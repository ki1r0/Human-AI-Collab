import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime.bolt_harness.agent import BoltAgentError, BoltHarnessAgent, _project_observation, parse_tool_call
from runtime.bolt_harness.executor import MAX_SKILL_DURATION_S as EXECUTOR_DEFAULT_SKILL_DURATION_S


class BoltHarnessAgentTests(unittest.TestCase):
    def setUp(self):
        self.task = {"task_id": "bolt_insert", "target_ids": ["bolt_socket_1"]}

    def assert_all_prompt_examples_parse(self, prompt):
        examples = [json.loads(line) for line in prompt.splitlines() if line.startswith('{"tool":')]
        expected_tools = [
            "observe", "execute_skill", "execute_skill", "execute_skill", "execute_skill",
            "nudge", "retract", "check_task",
        ]
        if any(item["tool"] == "send_help_request" for item in examples):
            expected_tools.append("send_help_request")
        expected_tools.append("stop")
        self.assertEqual([item["tool"] for item in examples], expected_tools)
        for item in examples:
            with self.subTest(example=item):
                self.assertNotIn("result", item)
                parsed = parse_tool_call(
                    json.dumps(item), completed=item.get("skill") == "release_and_retract",
                    task_spec=self.task,
                    available_evidence_refs=set(item.get("evidence_refs", ())),
                )
                self.assertEqual(parsed["tool"], item["tool"])
        self.assertEqual([item for item in examples if item["tool"] == "check_task"],
                         [{"tool": "check_task"}])
        return examples

    def test_malformed_and_non_object_responses_are_rejected(self):
        for content in ("not json", "[]", '{"tool":"stop","private":true}', '{"tool":[]}',
                        '{"tool":"check_task","result":{"completed":true}}',
                        '{"tool":"execute_skill","skill":"check_task","target_id":"bolt_socket_1","mode":"default"}',
                        '```json\n{"tool":"stop"}\n``` trailing text'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_tool_call(content, task_spec=self.task)
        self.assertEqual(parse_tool_call('```json\n{"tool":"stop"}\n```')["tool"], "stop")
        self.assertEqual(parse_tool_call(json.dumps('{"tool":"stop"}'))["tool"], "stop")
        with self.assertRaisesRegex(ValueError, "does not contain a JSON object"):
            parse_tool_call(json.dumps("\"tool\": \"stop\""))

    def test_nudge_retract_and_skill_ranges_are_bounded(self):
        calls = (
            '{"tool":"nudge","frame":"assembly","direction":"x_positive","step_class":"fine","max_duration_s":1.01}',
            '{"tool":"retract","direction":"z_positive","distance_class":"full","max_duration_s":5.1}',
            '{"tool":"execute_skill","skill":"insert_and_seat","target_id":"bolt_socket_1","mode":"compliant","max_duration_s":30.1}',
            '{"tool":"nudge","frame":"assembly","direction":"x_positive","step_class":"fine","max_duration_s":0}',
        )
        for content in calls:
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_tool_call(content, task_spec=self.task)

    def test_help_request_uses_declared_target_and_actual_public_evidence_refs(self):
        request = {
            "tool": "send_help_request", "target": "bolt_socket_1",
            "operation": "inspect the bolt and socket alignment",
            "allowed_scope": "specified assembly only",
            "desired_postconditions": ["bolt remains held", "insertion path is safe to reassess"],
            "evidence_refs": ["observation:obs-1", "tool_history:0"],
            "observed_problem": "Insertion did not complete; cause is unverified.",
        }
        parsed = parse_tool_call(
            json.dumps(request), task_spec=self.task,
            available_evidence_refs={"observation:obs-1", "tool_history:0"},
        )
        self.assertEqual(parsed, request)
        for change in (
            {"target": "hidden_bolt"},
            {"evidence_refs": ["observation:not-in-context"]},
            {"evidence_refs": []},
            {"ground_truth": "blocked"},
        ):
            invalid = {**request, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_tool_call(
                    json.dumps(invalid), task_spec=self.task,
                    available_evidence_refs={"observation:obs-1", "tool_history:0"},
                )

    def test_public_contact_projection_accepts_only_two_boolean_statuses(self):
        observation = {
            "observation_id": "obs-1", "frames": {"wrist": "data:image/png;base64,aGk="},
            "finger_bolt_contacts": [True, False],
        }
        projected, _ = _project_observation(observation)
        self.assertEqual(projected["finger_bolt_contacts"], [True, False])
        with self.assertRaisesRegex(ValueError, "two booleans"):
            _project_observation({**observation, "finger_bolt_contacts": [1, 1]})

    def test_release_requires_boolean_completion_signal(self):
        release = json.dumps({"tool": "execute_skill", "skill": "release_and_retract",
                              "target_id": "bolt_socket_1", "mode": "default", "max_duration_s": 5})
        with self.assertRaisesRegex(ValueError, "release_authorized=True"):
            parse_tool_call(release, task_spec=self.task, completed=False)
        parsed_release = parse_tool_call(release, task_spec=self.task, completed=True)
        self.assertEqual(parsed_release["skill"], "release_and_retract")
        self.assertEqual(parsed_release["max_duration_s"], 5)
        with self.assertRaisesRegex(ValueError, "completed must be a boolean"):
            parse_tool_call(release, task_spec=self.task, completed=1)

    def test_execute_skill_target_must_be_declared_by_task(self):
        call = json.dumps({"tool": "execute_skill", "skill": "pick", "target_id": "hidden_bolt",
                           "mode": "default", "max_duration_s": 10})
        with self.assertRaisesRegex(ValueError, "declared by task_spec"):
            parse_tool_call(call, task_spec=self.task)

    def test_check_task_history_contains_only_boolean_completion(self):
        observation = {"frames": {"wrist": "data:image/png;base64,aGk="}}
        agent = BoltHarnessAgent("http://model.test/chat/completions", "test")
        result = {"choices": [{"message": {"content": '{"tool":"stop"}'}}]}
        history = [{"call": {"tool": "check_task"}, "result": {"completed": False, "reason": "private"}}]
        with patch.dict(os.environ, {}, clear=False), patch(
            "runtime.bolt_harness.agent.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(result).encode())
        ), self.assertRaisesRegex(ValueError, "only a boolean completed"):
            agent.next_tool_call(observation, self.task, history)

    def test_actual_multimodal_request_contains_only_allowlisted_public_data(self):
        sentinel = "PRIVATE_EVALUATOR_SENTINEL_7f31"
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "wrist.png"
            frame.write_bytes(b"png bytes")
            observation = {
                "observation_id": "obs-1",
                "frames": {"wrist": str(frame)},
                "tcp_pose": [0, 0, 0, 1, 0, 0, 0],
                "finger_bolt_contacts": [True, True],
                "evaluator_private": sentinel,
                "object_estimates": [{"target_id": "bolt_socket_1", "class_name": "socket", "pose": [1, 2, 3, 1, 0, 0, 0], "private": sentinel}],
            }
            task = {**self.task, "goal": "seat the bolt", "ground_truth": sentinel,
                    "nominal_dimensions": {"bolt_diameter_m": 0.006, "evaluator_detail": sentinel}}
            history = [{"call": {"tool": "observe"}, "result": {"status": "completed", "private_trace": sentinel}}]
            response = {"choices": [{"message": {"content": '{"tool":"stop"}'}}]}
            captured = {}
            trace_events = []

            def fake_urlopen(request, timeout):
                captured["body"] = request.data.decode("utf-8")
                captured["headers"] = dict(request.header_items())
                captured["timeout"] = timeout
                return io.BytesIO(json.dumps(response).encode("utf-8"))

            agent = BoltHarnessAgent("http://model.test/v1/chat/completions", "public-vlm", timeout_s=4,
                                     trace_callback=trace_events.append)
            with patch.dict(os.environ, {"HRC_M1_API_KEY": "test-credential"}, clear=False), patch(
                "runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=fake_urlopen
            ), patch("runtime.bolt_harness.agent.parse_tool_call", wraps=parse_tool_call) as parser:
                call = agent.next_tool_call(observation, task, history)
                parser.assert_called_once()

        request_body = json.loads(captured["body"])
        user_content = request_body["messages"][1]["content"]
        structured = json.loads(user_content[0]["text"])
        image_parts = [part for part in user_content if part["type"] == "image_url"]
        traced_request = trace_events[0]["payload"]
        traced_user = traced_request["messages"][1]["content"]
        traced_context = json.loads(traced_user[0]["text"])
        self.assertEqual(call, {"tool": "stop"})
        self.assertEqual(request_body["model"], "public-vlm")
        self.assertEqual(captured["timeout"], 4)
        self.assertEqual(len(image_parts), 1)
        self.assertEqual(image_parts[0]["image_url"]["url"], "data:image/png;base64," + base64.b64encode(b"png bytes").decode())
        self.assertEqual(structured["observation"]["frames"], ["wrist"])
        self.assertEqual(structured["observation"]["finger_bolt_contacts"], [True, True])
        self.assertEqual(structured["task_spec"]["nominal_dimensions"], {"bolt_diameter_m": 0.006})
        self.assertEqual(traced_context["pose_format"], "position [x,y,z] in metres; quaternion [w,x,y,z]")
        self.assertEqual(traced_context["public_progress"], {
            "completed_skills": [], "check_phase": "pre_release", "release_authorized": False,
            "task_success": None, "latest_boolean_check_result": None,
            "last_call": {"tool": "observe"},
            "last_result": {"status": "completed"},
        })
        self.assertNotIn("completed", structured)
        self.assertEqual(traced_context["tools"]["execute_skill"]["mode_by_skill"], {
            "pick": "default", "transport": "default", "insert_and_seat": "compliant",
            "release_and_retract": "default",
        })
        self.assertIn("send_help_request", traced_context["tools"])
        self.assertIn("no helper executes", traced_context["tools"]["send_help_request"]["semantics"])
        self.assertIn("`force_limit` or `stalled` is returned insertion feedback, not an automatic help trigger",
                      request_body["messages"][0]["content"])
        self.assertIn('"evidence_refs":["observation:obs-1"]', request_body["messages"][0]["content"])
        self.assertTrue(traced_context["tools"]["execute_skill"]["release_and_retract_requires_release_authorized"])
        self.assertNotIn("release_and_retract_requires_completed", traced_context["tools"]["execute_skill"])
        self.assertIn("release_authorized=true",
                      traced_context["tools"]["execute_skill"]["skill_descriptions"]["release_and_retract"])
        self.assertEqual(traced_context["tools"]["execute_skill"]["arguments"]["skill"], [
            "insert_and_seat", "pick", "release_and_retract", "transport",
        ])
        self.assertIn("while retaining the grasp",
                      traced_context["tools"]["execute_skill"]["skill_descriptions"]["transport"])
        self.assertIn("optional", traced_context["tools"]["execute_skill"]["arguments"]["max_duration_s"])
        self.assertIn("executor default 30", traced_context["tools"]["execute_skill"]["arguments"]["max_duration_s"])
        check_task_schema = traced_context["tools"]["check_task"]
        self.assertEqual(check_task_schema["arguments"], {})
        self.assertIn("true permits", check_task_schema["returns_read_only"]["pre_release"]["release_authorized"])
        self.assertIn("fresh true", check_task_schema["returns_read_only"]["post_release"]["task_success"])
        self.assertNotIn("result", check_task_schema)
        self.assertEqual([part["image_url"]["url"] for part in traced_user if part["type"] == "image_url"], ["frame://wrist"])
        self.assertEqual(trace_events[-1]["raw_response"], json.dumps(response))
        self.assertNotIn("test-credential", json.dumps(trace_events))
        self.assertNotIn("data:image", json.dumps(trace_events))
        self.assertNotIn(sentinel, captured["body"])
        self.assertNotIn("test-credential", captured["body"])
        self.assertNotIn("evaluator_private", captured["body"])
        self.assertIn("Bearer", str(captured["headers"]))

    def test_help_examples_cite_visible_evidence_initially_and_after_stall(self):
        frame = "data:image/png;base64,aGk="
        stalled_history = [
            {
                "call": {"tool": "execute_skill", "skill": "pick", "target_id": "bolt_socket_1",
                         "mode": "default"},
                "result": {"status": "completed", "observation": {
                    "observation_id": "obs-a-pick", "frames": {"wrist": frame},
                }},
            },
            {
                "call": {"tool": "execute_skill", "skill": "insert_and_seat", "target_id": "bolt_socket_1",
                         "mode": "compliant"},
                "result": {"status": "stalled", "observation": {
                    "observation_id": "obs-m-stalled", "frames": {"wrist": frame},
                }},
            },
        ]
        scenarios = (
            ({"observation_id": "obs-z-initial", "frames": {"wrist": frame}}, ()),
            ({"observation_id": "obs-z-stalled", "frames": {"wrist": frame}}, stalled_history),
            ({"frames": {"wrist": frame}}, ()),
        )
        requests = []
        responses = [
            {"choices": [{"message": {"content": '{"tool":"stop"}'}}]}
            for _ in scenarios
        ]

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(json.dumps(responses[len(requests) - 1]).encode())

        agent = BoltHarnessAgent("http://model.test/chat/completions", "test")
        with patch("runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=fake_urlopen):
            for observation, history in scenarios:
                self.assertEqual(agent.next_tool_call(observation, self.task, history)["tool"], "stop")

        expected_current_refs = ("observation:obs-z-initial", "observation:obs-z-stalled")
        for index, request in enumerate(requests):
            system_prompt = request["messages"][0]["content"]
            context = json.loads(request["messages"][1]["content"][0]["text"])
            help_examples = [
                json.loads(line) for line in system_prompt.splitlines()
                if line.startswith('{"tool":"send_help_request"')
            ]
            if index == 2:
                self.assertEqual(help_examples, [])
                continue

            self.assertEqual(len(help_examples), 1)
            observable_refs = set()
            current_id = context["observation"].get("observation_id")
            if current_id:
                observable_refs.add(f"observation:{current_id}")
            for history_index, event in enumerate(context["tool_history"]):
                observable_refs.add(f"tool_history:{history_index}")
                result_observation = event["result"].get("observation", {})
                if result_observation.get("observation_id"):
                    observable_refs.add(f"observation:{result_observation['observation_id']}")
            self.assertLessEqual(set(help_examples[0]["evidence_refs"]), observable_refs)
            self.assertEqual(help_examples[0]["evidence_refs"], [expected_current_refs[index]])

    def test_skill_call_prompt_example_matches_parser_and_retry_names_validation_error(self):
        observation = {"frames": {"wrist": "data:image/png;base64,aGk="}}
        example_call = {
            "tool": "execute_skill", "skill": "pick", "target_id": "bolt_socket_1", "mode": "default",
        }
        responses = (
            {"choices": [{"message": {"content": '```json\n{"tool":"execute_skill pick"}\n```'}}]},
            {"choices": [{"message": {"content": json.dumps(example_call)}}]},
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(json.dumps(responses[len(requests) - 1]).encode())

        agent = BoltHarnessAgent("http://model.test/chat/completions", "test")
        with patch("runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=fake_urlopen):
            call = agent.next_tool_call(observation, self.task)

        expected = {**example_call, "max_duration_s": EXECUTOR_DEFAULT_SKILL_DURATION_S}
        initial_prompt = requests[0]["messages"][0]["content"]
        retry_prompt = requests[1]["messages"][0]["content"]
        example = json.dumps(example_call, ensure_ascii=True, separators=(",", ":"))
        self.assertIn(example, initial_prompt)
        self.assertEqual(parse_tool_call(example, task_spec=self.task), expected)
        self.assertIn("tool`=`execute_skill`", initial_prompt)
        self.assertIn("optional", initial_prompt)
        self.assertIn("executor default of 30 seconds", initial_prompt)
        self.assertIn("are not instantaneous", initial_prompt)
        self.assertIn("no explanation appended", initial_prompt)
        self.assertIn("do not repeat a successfully completed skill", initial_prompt)
        self.assertNotIn("max_duration_s", example)
        self.assertIn("unknown tool", retry_prompt)
        self.assertIn("execute_skill pick", retry_prompt)
        self.assertEqual(call, expected)
        self.assertEqual(len(requests), 2)

        prompt_examples = self.assert_all_prompt_examples_parse(initial_prompt)
        skill_examples = [item for item in prompt_examples if item["tool"] == "execute_skill"]
        self.assertEqual([item["skill"] for item in skill_examples], [
            "pick", "transport", "insert_and_seat", "release_and_retract",
        ])
        self.assertTrue(all("max_duration_s" not in item for item in skill_examples))
        self.assertIn('`{"tool":"check_task"}`', initial_prompt)
        self.assertIn("never an `execute_skill.skill`", initial_prompt)
        self.assertIn("never a call argument", initial_prompt)
        self.assert_all_prompt_examples_parse(retry_prompt)

        invalid_description = json.dumps({
            "tool": "execute_skill", "skill": "transport while maintaining the grasp",
            "target_id": "bolt_socket_1", "mode": "default",
        })
        with self.assertRaisesRegex(ValueError, "unsupported execute_skill skill"):
            parse_tool_call(invalid_description, task_spec=self.task)

    def test_invalid_skill_description_retry_repeats_all_exact_skill_examples(self):
        observation = {"frames": {"wrist": "data:image/png;base64,aGk="}}
        responses = (
            {"choices": [{"message": {"content": json.dumps({
                "tool": "execute_skill", "skill": "transport while maintaining the grasp",
                "target_id": "bolt_socket_1", "mode": "default",
            })}}]},
            {"choices": [{"message": {"content": json.dumps({
                "tool": "execute_skill", "skill": "transport", "target_id": "bolt_socket_1", "mode": "default",
            })}}]},
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(json.dumps(responses[len(requests) - 1]).encode())

        agent = BoltHarnessAgent("http://model.test/chat/completions", "test")
        with patch("runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=fake_urlopen):
            call = agent.next_tool_call(observation, self.task)

        self.assertEqual(call["skill"], "transport")
        self.assertEqual(call["max_duration_s"], EXECUTOR_DEFAULT_SKILL_DURATION_S)
        retry_prompt = requests[1]["messages"][0]["content"]
        self.assertIn("unsupported execute_skill skill", retry_prompt)
        self.assertIn("transport while maintaining the grasp", retry_prompt)
        examples = self.assert_all_prompt_examples_parse(retry_prompt)
        self.assertEqual([item["skill"] for item in examples if item["tool"] == "execute_skill"], [
            "pick", "transport", "insert_and_seat", "release_and_retract",
        ])

    def test_execute_skill_omission_uses_executor_default_and_explicit_budget_is_preserved(self):
        omitted = json.dumps({
            "tool": "execute_skill", "skill": "pick", "target_id": "bolt_socket_1", "mode": "default",
        })
        explicit = json.dumps({
            "tool": "execute_skill", "skill": "pick", "target_id": "bolt_socket_1", "mode": "default",
            "max_duration_s": 17,
        })
        self.assertEqual(parse_tool_call(omitted, task_spec=self.task)["max_duration_s"],
                         EXECUTOR_DEFAULT_SKILL_DURATION_S)
        self.assertEqual(parse_tool_call(explicit, task_spec=self.task)["max_duration_s"], 17)

        limited_task = {**self.task, "tool_limits": {"skill_max_duration_s": 20}}
        self.assertEqual(parse_tool_call(omitted, task_spec=limited_task)["max_duration_s"], 20)

    def test_public_progress_summarizes_only_successful_skills_and_last_public_result(self):
        observation = {"frames": {"wrist": "data:image/png;base64,aGk="}}
        history = [
            {"call": {"tool": "execute_skill", "skill": "pick", "target_id": "bolt_socket_1",
                      "mode": "default"},
             "result": {"status": "completed", "private_trace": "PROGRESS_PRIVATE_SENTINEL"}},
            {"call": {"tool": "execute_skill", "skill": "transport", "target_id": "bolt_socket_1",
                      "mode": "default"},
             "result": {"status": "succeeded"}},
        ]
        after_insert_failure = history + [
            {"call": {"tool": "execute_skill", "skill": "insert_and_seat", "target_id": "bolt_socket_1",
                      "mode": "compliant"},
             "result": {"status": "motion_timeout"}},
            {"call": {"tool": "check_task"}, "result": {"completed": False}},
        ]
        after_insert_stall = history + [
            {"call": {"tool": "execute_skill", "skill": "insert_and_seat", "target_id": "bolt_socket_1",
                      "mode": "compliant"},
             "result": {"status": "stalled"}},
        ]
        requests = []
        response = {"choices": [{"message": {"content": '{"tool":"stop"}'}}]}

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(json.dumps(response).encode())

        agent = BoltHarnessAgent("http://model.test/chat/completions", "test")
        with patch("runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=fake_urlopen):
            agent.next_tool_call(observation, self.task, history)
            agent.next_tool_call(observation, self.task, after_insert_failure)
            agent.next_tool_call(observation, self.task, after_insert_stall)

        contexts = [json.loads(request["messages"][1]["content"][0]["text"]) for request in requests]
        self.assertEqual(contexts[0]["public_progress"], {
            "completed_skills": ["pick", "transport"],
            "check_phase": "pre_release", "release_authorized": False, "task_success": None,
            "latest_boolean_check_result": None,
            "last_call": {"tool": "execute_skill", "skill": "transport"},
            "last_result": {"status": "succeeded"},
        })
        self.assertEqual(contexts[1]["public_progress"], {
            "completed_skills": ["pick", "transport"],
            "check_phase": "pre_release", "release_authorized": False, "task_success": None,
            "latest_boolean_check_result": {"phase": "pre_release", "release_authorized": False},
            "last_call": {"tool": "check_task"},
            "last_result": {"phase": "pre_release", "release_authorized": False},
        })
        self.assertEqual(contexts[1]["tool_history"][-1]["result"], {
            "phase": "pre_release", "release_authorized": False,
        })
        self.assertEqual(contexts[2]["tool_history"][-1]["result"], {"status": "stalled"})
        self.assertNotIn("PROGRESS_PRIVATE_SENTINEL", json.dumps(requests))

    def test_phase_scoped_completion_semantics_in_pre_and_post_release_requests(self):
        release = {
            "tool": "execute_skill", "skill": "release_and_retract", "target_id": "bolt_socket_1",
            "mode": "default", "max_duration_s": 5,
        }
        observation = {"frames": {"wrist": "data:image/png;base64,aGk="}}
        contexts = []
        cases = (
            ([{"call": {"tool": "check_task"}, "result": {"completed": True}}], True),
            ([
                {"call": {"tool": "check_task"}, "result": {"completed": True}},
                {"call": release, "result": {"status": "completed"}},
                {"call": {"tool": "check_task"}, "result": {"completed": True}},
            ], True),
            ([
                {"call": {"tool": "check_task"}, "result": {"completed": True}},
                {"call": release, "result": {"status": "completed"}},
            ], False),
        )
        response = {"choices": [{"message": {"content": '{"tool":"stop"}'}}]}

        def fake_urlopen(request, timeout):
            contexts.append(json.loads(request.data))
            return io.BytesIO(json.dumps(response).encode())

        agent = BoltHarnessAgent("http://model.test/chat/completions", "test")
        with patch("runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=fake_urlopen):
            for history, current_signal in cases:
                self.assertEqual(
                    agent.next_tool_call(observation, self.task, history, completed=current_signal),
                    {"tool": "stop"},
                )

        prompt = contexts[0]["messages"][0]["content"]
        required_order = prompt.split("Required order is ", 1)[1].split(" The check_task", 1)[0]
        required_sequence = (
            "`pick`", "`transport`", "`insert_and_seat`", "`check_task`",
            "`release_and_retract`", "`check_task` again",
        )
        sequence_positions = [required_order.index(part) for part in required_sequence]
        self.assertEqual(sequence_positions, sorted(sequence_positions))
        self.assertIn("Before release, `release_authorized: true` permits release_and_retract only", prompt)
        self.assertIn("only a fresh `task_success: true` completes the task", prompt)
        self.assertIn("A pre-release true remains valid until physical action", prompt)
        self.assertIn("repeating check_task without intervening physical action adds no evidence", prompt)
        self.assertIn("does not prescribe a next action", prompt)
        self.assertNotIn("`completed` value", prompt)
        self.assertIn("A stop never proves completion", prompt)
        public_cases = (
            {
                "check_phase": "pre_release", "release_authorized": True, "task_success": None,
                "latest_boolean_check_result": {"phase": "pre_release", "release_authorized": True},
                "last_call": {"tool": "check_task"},
                "last_result": {"phase": "pre_release", "release_authorized": True},
            },
            {
                "check_phase": "post_release", "release_authorized": None, "task_success": True,
                "latest_boolean_check_result": {"phase": "post_release", "task_success": True},
                "last_call": {"tool": "check_task"},
                "last_result": {"phase": "post_release", "task_success": True},
            },
            {
                "check_phase": "post_release", "release_authorized": None, "task_success": False,
                "latest_boolean_check_result": {"phase": "pre_release", "release_authorized": True},
                "last_call": {"tool": "execute_skill", "skill": "release_and_retract"},
                "last_result": {"status": "completed"},
            },
        )
        for request, (expected_history, _), expected_progress in zip(contexts, cases, public_cases):
            public = json.loads(request["messages"][1]["content"][0]["text"])
            self.assertIn("release_authorized=true", public["tools"]["check_task"]["semantics"])
            self.assertIn("task_success=true", public["tools"]["check_task"]["semantics"])
            self.assertIn("without claiming task completion", public["tools"]["stop"]["semantics"])
            self.assertNotIn("completed", public)
            self.assertEqual(public["public_progress"], {
                "completed_skills": ["release_and_retract"]
                if expected_progress["check_phase"] == "post_release" else [],
                **expected_progress,
            })
            self.assertEqual(len(public["tool_history"]), len(expected_history))
            self.assertEqual(public["tool_history"][-1]["call"], expected_history[-1]["call"])
            self.assertTrue(all("completed" not in event["result"] for event in public["tool_history"]))
            self.assertEqual(public["tool_history"][0]["result"], {
                "phase": "pre_release", "release_authorized": True,
            })
            self.assertNotIn("seat_ready", json.dumps(request))
            self.assertNotIn("evaluator", json.dumps(request).lower())
            self.assertNotIn("ground_truth", json.dumps(request).lower())

    def test_one_malformed_model_response_gets_one_retry_and_raw_trace(self):
        calls = []
        traces = []
        replies = (
            {"choices": [{"message": {"content": "not-json"}}]},
            {"choices": [{"message": {"content": '{"tool":"stop"}'}}]},
        )

        def fake_urlopen(request, timeout):
            calls.append(json.loads(request.data))
            return io.BytesIO(json.dumps(replies[len(calls) - 1]).encode())

        agent = BoltHarnessAgent("http://model.test/chat/completions", "test", trace_callback=traces.append)
        observation = {"frames": {"wrist": "data:image/png;base64,aGk="}}
        with patch("runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(agent.next_tool_call(observation, self.task), {"tool": "stop"})
        self.assertEqual(len(calls), 2)
        retry_prompt = calls[1]["messages"][0]["content"]
        self.assertIn("previous response failed tool validation", retry_prompt)
        self.assertIn("malformed JSON", retry_prompt)
        self.assertIn('"tool":"execute_skill"', retry_prompt)
        self.assertEqual([event["attempt"] for event in traces], [1, 1, 2, 2])
        self.assertEqual(traces[1]["raw_response"], json.dumps(replies[0]))

        calls.clear()
        agent = BoltHarnessAgent("http://model.test/chat/completions", "test")
        with patch("runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=[
            io.BytesIO(b"malformed"), io.BytesIO(b"malformed"), io.BytesIO(json.dumps(replies[1]).encode()),
        ]) as urlopen, self.assertRaises(BoltAgentError):
            agent.next_tool_call(observation, self.task)
        self.assertEqual(urlopen.call_count, 2)

    def test_past_release_history_is_not_gated_by_current_completion_bit(self):
        release = {"tool": "execute_skill", "skill": "release_and_retract", "target_id": "bolt_socket_1",
                   "mode": "default", "max_duration_s": 5}
        observation = {"frames": {"wrist": "data:image/png;base64,aGk="}}
        response = {"choices": [{"message": {"content": '{"tool":"stop"}'}}]}
        captured = {}

        def fake_urlopen(request, timeout):
            captured["context"] = json.loads(request.data)["messages"][1]["content"][0]["text"]
            return io.BytesIO(json.dumps(response).encode())

        agent = BoltHarnessAgent("http://model.test/chat/completions", "test")
        history = [{"call": release, "result": {"status": "succeeded"}}]
        with patch("runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(agent.next_tool_call(observation, self.task, history, completed=False), {"tool": "stop"})
        public_context = json.loads(captured["context"])
        self.assertEqual(public_context["tool_history"][0]["call"], release)
        self.assertEqual(public_context["public_progress"]["check_phase"], "post_release")
        self.assertIsNone(public_context["public_progress"]["release_authorized"])
        self.assertFalse(public_context["public_progress"]["task_success"])
        self.assertNotIn("completed", public_context)

    def test_model_transport_failure_is_not_replaced_with_a_success(self):
        agent = BoltHarnessAgent("http://model.test/chat/completions", "test")
        observation = {"frames": {"wrist": "data:image/png;base64,aGk="}}
        with patch("runtime.bolt_harness.agent.urllib.request.urlopen", side_effect=TimeoutError):
            with self.assertRaises(BoltAgentError):
                agent.next_tool_call(observation, self.task)


if __name__ == "__main__":
    unittest.main()
