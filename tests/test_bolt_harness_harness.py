"""Unit-only harness tests: dummy callbacks/executor and mocked HTTP, no sim or live model."""

import io
import json
import unittest
from unittest.mock import patch

from runtime.bolt_harness.agent import BoltHarnessAgent
from runtime.bolt_harness.harness import BoltHarnessLoop


TASK = {"task_id": "bolt_insert", "target_ids": ["socket_1"]}
TEST_ONLY_FRAME = "data:image/png;base64,aGk="
PRIVATE_SENTINEL = "PRIVATE_CHECKER_DETAIL_TEST_ONLY"


def _test_only_observation():
    return {
        "observation_id": "unit-frame-1",
        "frames": {"wrist": TEST_ONLY_FRAME},
        "joint_positions": [0.0] * 9,
        "wrench": [0.0] * 6,
        "finger_bolt_contacts": [True, True],
        "private_debug": PRIVATE_SENTINEL,
    }


def _release_call():
    return {
        "tool": "execute_skill", "skill": "release_and_retract", "target_id": "socket_1",
        "mode": "default", "max_duration_s": 5,
    }


class DummyAgent:
    """Test-only action queue; does not make or represent a model request."""

    def __init__(self, calls):
        self.calls = list(calls)
        self.trace_callback = None
        self.inputs = []

    def next_tool_call(self, observation, task_spec, history, *, completed=False):
        self.inputs.append((observation, task_spec, list(history), completed))
        return self.calls.pop(0)


class DummyExecutor:
    """Test-only executor; no physical actions are performed."""

    def __init__(self):
        self.skills = []

    def execute_skill(self, skill, target_id, *, mode, max_duration_s, completed=False):
        self.skills.append((skill, target_id, mode, max_duration_s, completed))
        return {"status": "completed"}


class DummyNudgeExecutor(DummyExecutor):
    """Test-only executor with a nudge method for readiness-invalidation coverage."""

    def __init__(self):
        super().__init__()
        self.nudges = []

    def nudge(self, **arguments):
        self.nudges.append(arguments)
        return {"status": "completed"}


class ForceLimitExecutor(DummyExecutor):
    def execute_skill(self, skill, target_id, *, mode, max_duration_s, completed=False):
        self.skills.append((skill, target_id, mode, max_duration_s, completed))
        return {"status": "force_limit"}


class StalledExecutor(ForceLimitExecutor):
    def execute_skill(self, skill, target_id, *, mode, max_duration_s, completed=False):
        self.skills.append((skill, target_id, mode, max_duration_s, completed))
        return {"status": "stalled"}


class MotionTimeoutExecutor(ForceLimitExecutor):
    def execute_skill(self, skill, target_id, *, mode, max_duration_s, completed=False):
        self.skills.append((skill, target_id, mode, max_duration_s, completed))
        return {"status": "motion_timeout"}


class GuardedRetractExecutor(DummyExecutor):
    def retract(self, **_arguments):
        return {"status": "force_limit"}


class NativeGuardExceptionExecutor(DummyExecutor):
    def execute_skill(self, *_args, **_kwargs):
        raise RuntimeError("native force guard")


class BoltHarnessLoopUnitTests(unittest.TestCase):
    def make_loop(
        self, calls, task_checks, *, executor=None, trace=None, max_calls=16, help_request_callback=None
    ):
        agent = DummyAgent(calls)
        executor = executor or DummyExecutor()
        checks = iter(task_checks)
        traces = trace if trace is not None else []
        loop = BoltHarnessLoop(
            agent,
            executor,
            task_spec=TASK,
            observation_callback=_test_only_observation,
            check_task_callback=lambda: next(checks),
            trace_callback=traces.append,
            help_request_callback=help_request_callback,
            max_calls=max_calls,
        )
        return loop, agent, executor, traces

    def test_dummy_loop_requires_pre_release_readiness_and_post_release_success(self):
        loop, agent, executor, traces = self.make_loop(
            [
                {"tool": "check_task"},
                _release_call(),
                {"tool": "check_task"},
            ],
            [{"held": True, "seat_ready": True}, {"task_success": True}],
        )

        result = loop.run()

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.task_success)
        self.assertTrue(result.held)
        self.assertTrue(result.seat_ready)
        self.assertEqual(result.status, "success")
        self.assertEqual(executor.skills, [("release_and_retract", "socket_1", "default", 5.0, True)])
        self.assertEqual([inputs[3] for inputs in agent.inputs], [False, True, False])
        self.assertEqual(result.history[0]["result"], {"completed": True})
        self.assertEqual(result.history[-1]["result"], {"completed": True})
        self.assertIn("tool_result", [event["event"] for event in traces])
        self.assertNotIn(PRIVATE_SENTINEL, json.dumps(traces))

    def test_dummy_stop_and_unverified_seat_never_claim_success(self):
        loop, _, _, _ = self.make_loop([{"tool": "stop"}], [])

        result = loop.run()

        self.assertEqual(result.status, "stopped")
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(result.task_success)

    def test_dummy_release_without_completed_signal_never_reaches_executor(self):
        loop, _, executor, _ = self.make_loop([_release_call()], [])

        result = loop.run()

        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(result.task_success)
        self.assertEqual(executor.skills, [])

    def test_dummy_motion_invalidates_cached_held_seat_ready_gate(self):
        nudge = {
            "tool": "nudge", "frame": "assembly", "direction": "x_positive",
            "step_class": "fine", "max_duration_s": 0.5,
        }
        executor = DummyNudgeExecutor()
        loop, agent, _, _ = self.make_loop(
            [{"tool": "check_task"}, nudge, _release_call()],
            [{"held": True, "seat_ready": True}],
            executor=executor,
        )

        result = loop.run()

        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(len(executor.nudges), 1)
        self.assertEqual(executor.skills, [])
        self.assertEqual([item[3] for item in agent.inputs], [False, True, False])

    def test_unsupported_dummy_nudge_fails_without_claiming_dispatch(self):
        nudge = {
            "tool": "nudge", "frame": "assembly", "direction": "x_positive",
            "step_class": "fine", "max_duration_s": 0.5,
        }
        loop, _, executor, _ = self.make_loop([nudge], [])

        result = loop.run()

        self.assertEqual(result.status, "failed")
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(result.history[0]["result"], {"status": "invalid_action"})
        self.assertEqual(executor.skills, [])

    def test_private_exception_is_returned_to_runner_but_not_public_trace(self):
        failure = RuntimeError(PRIVATE_SENTINEL)
        traces = []
        agent = DummyAgent([{"tool": "check_task"}])
        loop = BoltHarnessLoop(
            agent,
            DummyExecutor(),
            task_spec=TASK,
            observation_callback=_test_only_observation,
            check_task_callback=lambda: (_ for _ in ()).throw(failure),
            trace_callback=traces.append,
        )

        result = loop.run()

        self.assertNotEqual(result.exit_code, 0)
        self.assertIs(result.private_exception, failure)
        self.assertNotIn(PRIVATE_SENTINEL, json.dumps(traces))

    def test_dummy_max_calls_is_a_nonzero_runner_result(self):
        loop, _, _, _ = self.make_loop([{"tool": "observe"}], [], max_calls=1)

        result = loop.run()

        self.assertEqual(result.status, "max_calls")
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(result.calls, 1)
        self.assertFalse(result.task_success)

    def test_help_request_is_a_terminal_local_outbox_boundary_not_task_success(self):
        request = {
            "tool": "send_help_request", "target": "socket_1",
            "operation": "inspect the bolt and socket alignment", "allowed_scope": "specified assembly only",
            "desired_postconditions": ["bolt remains held"], "evidence_refs": ["observation:unit-frame-1"],
            "observed_problem": "Insertion did not complete; cause is unverified.",
        }
        persisted = []
        loop, agent, executor, traces = self.make_loop(
            [request], [], help_request_callback=lambda value: persisted.append(dict(value)),
        )

        result = loop.run()

        self.assertEqual(result.status, "help_requested")
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(result.task_success)
        self.assertEqual(result.calls, 1)
        self.assertEqual(executor.skills, [])
        self.assertEqual(persisted[0]["request_id"], "help-01")
        self.assertEqual(result.help_request, persisted[0])
        tool_result = next(event for event in traces if event["event"] == "tool_result")
        self.assertEqual(tool_result["result"]["delivery"], "local_outbox")
        self.assertNotIn("human_received", json.dumps(traces))
        self.assertNotIn(PRIVATE_SENTINEL, json.dumps(traces))

    def test_help_request_without_local_outbox_is_rejected(self):
        request = {
            "tool": "send_help_request", "target": "socket_1", "operation": "inspect",
            "allowed_scope": "assembly only", "desired_postconditions": ["bolt remains held"],
            "evidence_refs": ["observation:unit-frame-1"], "observed_problem": "Unverified symptom.",
        }
        loop, _, _, _ = self.make_loop([request], [])

        result = loop.run()

        self.assertEqual(result.status, "failed")
        self.assertFalse(result.task_success)
        self.assertIsNone(result.help_request)

    def test_help_request_can_reference_an_actual_tool_history_entry(self):
        request = {
            "tool": "send_help_request", "target": "socket_1", "operation": "inspect alignment",
            "allowed_scope": "specified assembly only", "desired_postconditions": ["bolt remains held"],
            "evidence_refs": ["tool_history:0"], "observed_problem": "Insertion remains incomplete; cause unverified.",
        }
        observations = iter((
            {**_test_only_observation(), "observation_id": "initial"},
            {**_test_only_observation(), "observation_id": "after-observe"},
        ))
        persisted = []
        agent = DummyAgent([{"tool": "observe"}, request])
        loop = BoltHarnessLoop(
            agent, DummyExecutor(), task_spec=TASK,
            observation_callback=lambda: next(observations),
            check_task_callback=lambda: {"held": False, "seat_ready": False},
            trace_callback=lambda _event: None,
            help_request_callback=lambda value: persisted.append(dict(value)),
        )

        result = loop.run()

        self.assertEqual(result.status, "help_requested")
        self.assertEqual(agent.inputs[1][2][0]["call"], {"tool": "observe"})
        self.assertEqual(persisted[0]["evidence_refs"], ["tool_history:0"])

    def test_insertion_force_limit_returns_fresh_public_state_to_model_without_auto_help(self):
        insert = {
            "tool": "execute_skill", "skill": "insert_and_seat", "target_id": "socket_1",
            "mode": "compliant", "max_duration_s": 5,
        }
        agent = DummyAgent([insert, {"tool": "stop"}])
        executor = ForceLimitExecutor()
        observations = iter((
            {**_test_only_observation(), "observation_id": "before"},
            {**_test_only_observation(), "observation_id": "after-force-limit"},
        ))
        traces = []
        loop = BoltHarnessLoop(
            agent, executor, task_spec=TASK,
            observation_callback=lambda: next(observations),
            check_task_callback=lambda: {"held": False, "seat_ready": False},
            trace_callback=traces.append,
        )

        result = loop.run()

        self.assertEqual(result.status, "stopped")
        self.assertFalse(result.task_success)
        self.assertEqual(result.calls, 2)
        self.assertEqual(agent.inputs[1][0]["observation_id"], "after-force-limit")
        self.assertEqual(agent.inputs[1][2][-1]["result"]["status"], "force_limit")
        self.assertEqual(agent.inputs[1][2][-1]["result"]["observation"]["observation_id"], "after-force-limit")
        self.assertEqual(executor.skills[0][0], "insert_and_seat")
        self.assertNotIn("send_help_request", [item["call"]["tool"] for item in result.history])

    def test_stalled_insert_reaches_model_and_model_can_choose_help(self):
        insert = {
            "tool": "execute_skill", "skill": "insert_and_seat", "target_id": "socket_1",
            "mode": "compliant", "max_duration_s": 5,
        }
        request = {
            "tool": "send_help_request", "target": "socket_1", "operation": "inspect alignment",
            "allowed_scope": "specified assembly only", "desired_postconditions": ["bolt remains held"],
            "evidence_refs": ["observation:after-stall"],
            "observed_problem": "Insertion stalled; cause is unverified.",
        }
        agent = DummyAgent([insert, request])
        observations = iter((
            {**_test_only_observation(), "observation_id": "before"},
            {**_test_only_observation(), "observation_id": "after-stall"},
        ))
        persisted = []
        loop = BoltHarnessLoop(
            agent, StalledExecutor(), task_spec=TASK,
            observation_callback=lambda: next(observations),
            check_task_callback=lambda: {"held": False, "seat_ready": False},
            trace_callback=lambda _event: None,
            help_request_callback=lambda value: persisted.append(dict(value)),
        )

        result = loop.run()

        self.assertEqual(result.status, "help_requested")
        self.assertEqual(result.exit_code, 1)
        self.assertFalse(result.task_success)
        self.assertEqual(result.calls, 2)
        self.assertEqual(agent.inputs[1][0]["observation_id"], "after-stall")
        self.assertEqual(agent.inputs[1][2][-1]["result"]["status"], "stalled")
        self.assertEqual(persisted[0]["evidence_refs"], ["observation:after-stall"])

    def test_generic_insertion_motion_timeout_remains_fatal(self):
        insert = {
            "tool": "execute_skill", "skill": "insert_and_seat", "target_id": "socket_1",
            "mode": "compliant", "max_duration_s": 5,
        }
        loop, agent, _, _ = self.make_loop(
            [insert, {"tool": "send_help_request"}], [], executor=MotionTimeoutExecutor(),
        )

        result = loop.run()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.calls, 1)
        self.assertEqual(len(agent.inputs), 1)
        self.assertIsNone(result.help_request)

    def test_force_limit_without_bilateral_contact_is_fatal(self):
        insert = {
            "tool": "execute_skill", "skill": "insert_and_seat", "target_id": "socket_1",
            "mode": "compliant", "max_duration_s": 5,
        }
        agent = DummyAgent([insert, {"tool": "stop"}])
        observations = iter((
            {**_test_only_observation(), "observation_id": "before"},
            {**_test_only_observation(), "observation_id": "unsafe", "finger_bolt_contacts": [True, False]},
        ))
        loop = BoltHarnessLoop(
            agent, ForceLimitExecutor(), task_spec=TASK,
            observation_callback=lambda: next(observations),
            check_task_callback=lambda: {"held": False, "seat_ready": False},
            trace_callback=lambda _event: None,
        )

        result = loop.run()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.calls, 1)
        self.assertIn("bilateral", result.error)
        self.assertFalse(result.task_success)

    def test_retract_force_limit_remains_fatal_and_does_not_reach_model_again(self):
        retract = {
            "tool": "retract", "direction": "z_positive", "distance_class": "short", "max_duration_s": 1,
        }
        loop, agent, _, _ = self.make_loop([retract, {"tool": "stop"}], [], executor=GuardedRetractExecutor())

        result = loop.run()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.calls, 1)
        self.assertEqual(len(agent.inputs), 1)

    def test_native_executor_guard_exception_remains_fatal(self):
        insert = {
            "tool": "execute_skill", "skill": "insert_and_seat", "target_id": "socket_1",
            "mode": "compliant", "max_duration_s": 5,
        }
        loop, agent, _, _ = self.make_loop(
            [insert, {"tool": "send_help_request"}], [], executor=NativeGuardExceptionExecutor(),
        )

        result = loop.run()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.calls, 1)
        self.assertIsInstance(result.private_exception, RuntimeError)
        self.assertEqual(len(agent.inputs), 1)

    def test_mock_transport_wires_raw_agent_and_tool_results_to_one_trace_sink(self):
        traces = []
        response = {"choices": [{"message": {"content": '{"tool":"stop"}'}}]}
        agent = BoltHarnessAgent("http://model.test/chat/completions", "unit-test")
        loop = BoltHarnessLoop(
            agent,
            DummyExecutor(),
            task_spec=TASK,
            observation_callback=_test_only_observation,
            check_task_callback=lambda: {"held": False, "seat_ready": False},
            trace_callback=traces.append,
        )
        with patch(
            "runtime.bolt_harness.agent.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(response).encode()),
        ):
            result = loop.run()

        events = [event["event"] for event in traces]
        self.assertIn("request", events)
        self.assertIn("response", events)
        self.assertIn("tool_result", events)
        self.assertIn("run_finished", events)
        self.assertNotEqual(result.exit_code, 0)
        self.assertNotIn(PRIVATE_SENTINEL, json.dumps(traces))
        request = next(event["payload"] for event in traces if event["event"] == "request")
        request_context = json.loads(request["messages"][1]["content"][0]["text"])
        self.assertEqual(request_context["observation"]["frames"], ["wrist"])


if __name__ == "__main__":
    unittest.main()
