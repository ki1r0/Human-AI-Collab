import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hrc_m1.contracts import Decision, EpisodeState, Observation, SkillResult
from hrc_m1.observer import ObserverVerifier, VLMVerdict, _parse_assessment
from hrc_m1.planner import HttpPlanner
from hrc_m1.state_machine import M1StateMachine
from hrc_m1.vader import assessment_context, expected_outcome
from hrc_repair.contracts import Observation as RepairObservation, PlannerAction
from hrc_repair.planner import HttpRepairPlanner, PlannerUnavailable
from hrc_repair.run import _fresh_action, _safe_stop_action, _scatter_action_error, _scatter_recovery_action
from hrc_repair.state_recognizer import assess_initial, assess_pick, assess_place, expected_initial_state, expected_place_outcome, planner_observation, visual_success
from tools.repair_qwen_server import normalize_messages


class VaderPipelineTests(unittest.TestCase):
    def test_outcome_description_uses_configured_parts(self):
        result = expected_outcome(
            {
                "source_part": "Hub_Cover_Output_Top",
                "target_part": "Casing_Top",
                "target_frame": "socket_hub_output",
            },
            "insert",
        )
        self.assertIn("Hub Cover Output Top", result)
        self.assertIn("socket hub output", result)
        self.assertIn("Casing Top", result)

    def test_vqa_assessment_is_attached_to_planner_context(self):
        context = assessment_context({"skill_id": "insert"}, "cover is seated", "FAILED", "cover remains above socket")
        self.assertEqual(context["expected_outcome"], "cover is seated")
        self.assertEqual(context["vqa_assessment"]["verdict"], "FAILED")
        self.assertIn("remains above", context["vqa_assessment"]["evidence"])

    def test_visual_failure_replans_only_when_enabled(self):
        observation = Observation("episode", "obs0", 0.0)
        decision = Decision.from_dict({
            "schema_version": "m1-decision-v1",
            "episode_id": "episode",
            "observation_id": "obs0",
            "action": "PICK",
            "target_part": "Hub_Cover_Output_Top",
        })
        for enabled, expected_state in ((True, EpisodeState.REPLAN), (False, EpisodeState.SAFE_HOLD)):
            sm = M1StateMachine("episode", replan_on_visual_failure=enabled)
            sm.reset()
            sm.accept_observation(observation)
            sm.accept_decision(decision)
            sm.skill_started("pick")
            sm.skill_finished(SkillResult("episode", "obs0", "pick", "SUCCEEDED"))
            sm.verify("FAILED")
            self.assertEqual(sm.state, expected_state)

    def test_vlm_parser_prefers_verdict_field_and_preserves_assessment(self):
        verdict, evidence = _parse_assessment('{"verdict":"FAILED","assessment":"Part is not seated."}')
        self.assertEqual(verdict, VLMVerdict.FAILED)
        self.assertEqual(evidence, "Part is not seated.")
        verdict, _ = _parse_assessment("FAILED: image does not show a SUCCESS state")
        self.assertEqual(verdict, VLMVerdict.FAILED)

    def test_verifier_sends_expected_outcome_and_image(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.png"
            frame.write_bytes(b"test image")
            captured = {}

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def read(self):
                    return json.dumps({"choices": [{"message": {"content": '{"verdict":"FAILED","assessment":"Part is still above the socket."}'}}]}).encode()

            def fake_urlopen(request, timeout):
                captured["payload"] = json.loads(request.data.decode())
                return Response()

            verifier = ObserverVerifier("http://local/v1/chat/completions", "qwen-vl")
            observation = Observation("episode", "obs0", 0.0, frames={"head_rgb": str(frame)})
            with patch.dict(os.environ, {"HRC_M1_VLM_API_KEY": "local"}):
                with patch("hrc_m1.observer.urllib.request.urlopen", side_effect=fake_urlopen):
                    result = verifier.verify(observation, expected_outcome="Part is seated in the socket")
            self.assertEqual(result.verdict, VLMVerdict.FAILED)
            blocks = captured["payload"]["messages"][0]["content"]
            self.assertIn("Part is seated in the socket", blocks[0]["text"])
            self.assertEqual(blocks[1]["type"], "image_url")

    def test_vader_planner_does_not_leak_local_frame_paths(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                action = {"schema_version": "m1-decision-v1", "episode_id": "episode", "observation_id": "obs0", "action": "ABORT"}
                return json.dumps({"choices": [{"message": {"content": json.dumps(action)}}]}).encode()

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode())
            return Response()

        observation = Observation(
            "episode", "obs0", 0.0,
            frames={"head_rgb": "/private/run/frame.png"},
            recent_skill={"vqa_assessment": {"verdict": "FAILED", "evidence": "Part not seated"}},
        )
        planner = HttpPlanner("http://local/v1/chat/completions", "text-planner", pipeline="vader")
        with patch.dict(os.environ, {"HRC_M1_API_KEY": "local"}):
            with patch("hrc_m1.planner.urllib.request.urlopen", side_effect=fake_urlopen):
                planner.next_decision(observation, EpisodeState.REPLAN)
        user = json.loads(captured["payload"]["messages"][1]["content"])
        self.assertEqual(user["observation"]["frames"], {})
        self.assertNotIn("/private/run/frame.png", json.dumps(user))
        self.assertIn("vqa_assessment", user["observation"]["recent_skill"])

    def test_vader_planner_binds_fixed_part_and_socket_to_llm_action(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                action = {"schema_version": "m1-decision-v1", "episode_id": "episode", "observation_id": "obs0", "action": "GUARDED_INSERT"}
                return json.dumps({"choices": [{"message": {"content": json.dumps(action)}}]}).encode()

        observation = Observation("episode", "obs0", 0.0)
        planner = HttpPlanner("http://local/v1/chat/completions", "text-planner", pipeline="vader")
        with patch.dict(os.environ, {"HRC_M1_API_KEY": "local"}):
            with patch("hrc_m1.planner.urllib.request.urlopen", return_value=Response()):
                decision, _ = planner.next_decision(observation, EpisodeState.PLAN)
        self.assertEqual(decision.target_part, "Hub_Cover_Output_Top")
        self.assertEqual(decision.target_socket, "socket_hub_output")

    def test_local_server_translates_openai_image_blocks(self):
        messages = [
            {"role": "system", "content": "Return JSON only"},
            {"role": "user", "content": [
                {"type": "text", "text": "Check the part"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ]},
        ]
        normalized = normalize_messages(messages)
        self.assertEqual(normalized[0]["content"], [{"type": "text", "text": "Return JSON only"}])
        self.assertEqual(normalized[1]["content"][1], {"type": "image", "url": "data:image/png;base64,AA=="})

    def test_m0_vader_expected_outcome_uses_registered_parts(self):
        expected = expected_place_outcome({"scene": {
            "cover_prim": "/World/envs/env_0/Hub_Cover_Output_Top",
            "hub_prim": "/World/envs/env_0/Casing_Top",
            "registered_goal_frame": "socket_hub_output",
        }})
        self.assertIn("Hub Cover Output Top", expected)
        self.assertIn("socket hub output", expected)
        self.assertIn("Casing Top", expected)
        self.assertIn("broad circular flange", expected)
        self.assertIn("raised center opening is part of the cover", expected)
        self.assertIn("release is checked separately", expected)
        self.assertIn("separate ring resting on the tabletop", expected)

    def test_m0_vader_attaches_vqa_text_without_copying_frame_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "head_rgb.png"
            frame.write_bytes(b"image")
            hand_frame = Path(directory) / "left_hand_rgb.png"
            hand_frame.write_bytes(b"occluded image")

            class Verdict:
                value = "SUCCESS"

            class Result:
                verdict = Verdict()
                evidence = "Cover appears seated and gripper is clear."
                prompt_hash = "prompt"
                response_hash = "response"

            class Verifier:
                def verify(self, observation, *, expected_outcome):
                    self.frames = observation.frames
                    self.expected = expected_outcome
                    return Result()

            verifier = Verifier()
            observation = RepairObservation("episode", "obs1", 0, 1.0, recent_skill={"skill_id": "place"})
            updated, result = assess_place(verifier, observation, [str(hand_frame), str(frame)], "cover seated")
        self.assertEqual(len(verifier.frames), 1)
        self.assertEqual(next(iter(verifier.frames.values())), str(frame))
        self.assertEqual(verifier.expected, "cover seated")
        self.assertTrue(visual_success(updated))
        self.assertEqual(updated.recent_skill["expected_outcome"], "cover seated")
        self.assertNotIn(str(frame), json.dumps(updated.to_dict()))
        self.assertEqual(result.verdict.value, "SUCCESS")

    def test_public_contact_failure_overrules_positive_place_vlm(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "head_rgb.png"
            frame.write_bytes(b"image")

            class Result:
                verdict = VLMVerdict.SUCCESS
                evidence = "Cover appears seated."
                prompt_hash = "prompt"
                response_hash = "response"

            class Verifier:
                def verify(self, observation, *, expected_outcome):
                    return Result()

            observation = RepairObservation(
                "episode", "obs1", 0, 1.0,
                recent_skill={"skill_id": "place", "feedback": {"public_sensor": {
                    "blocked_guard": True,
                    "released": False,
                    "release_contact_free": True,
                }}},
            )
            updated, _ = assess_place(Verifier(), observation, [str(frame)], "cover seated")
        self.assertEqual(updated.held, "unknown")
        self.assertEqual(updated.placed, "no")
        self.assertEqual(updated.release_observed, "no")
        self.assertTrue(visual_success(updated))

    def test_unconfirmed_after_place_state_recovers_to_stop(self):
        observation = RepairObservation(
            "episode", "obs1", 0, 1.0, held="unknown", placed="no", release_observed="no",
            recent_skill={"skill_id": "place", "feedback": {"public_sensor": {
                "placement_observed": False,
                "blocked_guard": False,
            }}},
        )
        place = PlannerAction("obs1", 0, "place", {"target": "hub"})
        self.assertIn("held=yes", _scatter_action_error(place, observation))
        self.assertEqual(_scatter_recovery_action(observation, help_budget=1), "stop")
        self.assertEqual(_safe_stop_action(observation), PlannerAction("obs1", 0, "stop", {}))

    def test_m0_vader_initial_scatter_observation_requires_a_supported_unheld_cover(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "initial_head_rgb.png"
            frame.write_bytes(b"image")

            class Result:
                verdict = VLMVerdict.SUCCESS
                evidence = "Cover rests separately on its support."
                prompt_hash = "prompt"
                response_hash = "response"

            class Verifier:
                def verify(self, observation, *, expected_outcome):
                    self.expected = expected_outcome
                    self.frames = observation.frames
                    return Result()

            verifier = Verifier()
            observation = RepairObservation("episode", "obs0", 0, 1.0)
            expected = expected_initial_state({"scene": {"cover_prim": "Hub_Cover_Output_Top", "hub_prim": "Casing_Top"}})
            updated, _ = assess_initial(verifier, observation, [str(frame)], expected)
        self.assertEqual(updated.held, "no")
        self.assertEqual(updated.placed, "no")
        self.assertEqual(verifier.expected, expected)
        self.assertIn("physical tabletop support", expected)

    def test_repair_rgb_pick_assessment_sets_held_and_keeps_paths_private(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = []
            for alias in ("head_rgb", "left_hand_rgb", "right_hand_rgb"):
                frame = Path(directory) / f"after_pick_{alias}.png"
                frame.write_bytes(b"image")
                frames.append(str(frame))

            class Verdict:
                value = "SUCCESS"

            class Result:
                verdict = Verdict()
                evidence = "The cover is pinched and clear of its support."
                prompt_hash = "prompt"
                response_hash = "response"

            class Verifier:
                def verify(self, observation, *, expected_outcome):
                    self.frames = observation.frames
                    self.expected = expected_outcome
                    return Result()

            verifier = Verifier()
            observation = RepairObservation("episode", "obs1", 0, 1.0, held="unknown")
            updated, result = assess_pick(verifier, observation, frames, "robot visibly holds cover")
        self.assertEqual(len(verifier.frames), 3)
        self.assertEqual(updated.held, "yes")
        self.assertEqual(updated.placed, "no")
        self.assertTrue(visual_success(updated))
        self.assertNotIn(directory, json.dumps(updated.to_dict()))
        self.assertEqual(result.verdict.value, "SUCCESS")

    def test_m0_vader_uses_right_hand_view_after_public_release(self):
        with tempfile.TemporaryDirectory() as directory:
            head = Path(directory) / "1231_head_rgb.png"
            right = Path(directory) / "1231_right_hand_rgb.png"
            head.write_bytes(b"head")
            right.write_bytes(b"right")

            class Verdict:
                value = "SUCCESS"

            class Result:
                verdict = Verdict()
                evidence = "Cover is seated."

            class Verifier:
                def verify(self, observation, *, expected_outcome):
                    self.frames = observation.frames
                    return Result()

            verifier = Verifier()
            observation = RepairObservation("episode", "obs3", 0, 1.0, placed="yes", release_observed="yes")
            assess_place(verifier, observation, [str(head), str(right)], "cover seated")
        self.assertEqual(list(verifier.frames.values()), [str(right)])

    def test_repair_place_assessment_can_use_all_camera_views(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = []
            for alias in ("head_rgb", "left_hand_rgb", "right_hand_rgb"):
                frame = Path(directory) / f"post_place_{alias}.png"
                frame.write_bytes(b"image")
                frames.append(str(frame))

            class Result:
                verdict = VLMVerdict.SUCCESS
                evidence = "Visible from all views."

            class Verifier:
                def verify(self, observation, *, expected_outcome):
                    self.frames = observation.frames
                    return Result()

            verifier = Verifier()
            assess_place(verifier, RepairObservation("episode", "obs", 0, 1.0), frames, "seated", all_views=True)
        self.assertEqual(len(verifier.frames), 3)

    def test_m0_vader_lmp_context_drops_prior_turn_ids_and_keeps_vqa(self):
        observation = RepairObservation(
            "episode", "obs_0001", 0, 1.0,
            recent_skill={"observation_id": "obs_0000", "control_epoch": 0, "vqa_assessment": {"verdict": "FAILED"}},
        )
        planner_obs = planner_observation(observation)
        self.assertEqual(planner_obs.observation_id, "obs_0001")
        self.assertNotIn("observation_id", planner_obs.recent_skill)
        self.assertNotIn("control_epoch", planner_obs.recent_skill)
        self.assertEqual(planner_obs.recent_skill["vqa_assessment"]["verdict"], "FAILED")

    def test_vader_scatter_rejects_repeated_pick_after_live_grasp(self):
        observation = RepairObservation(
            "episode", "obs_0001", 0, 1.0, held="yes", placed="no",
            recent_skill={"skill_id": "pick", "vqa_assessment": {"verdict": "SUCCESS"}},
        )
        repeated_pick = PlannerAction("obs_0001", 0, "pick", {"target": "cover"})
        place = PlannerAction("obs_0001", 0, "place", {"target": "hub"})
        self.assertIn("held=no", _scatter_action_error(repeated_pick, observation, place_once=True))
        self.assertIsNone(_scatter_action_error(place, observation, place_once=True))

    def test_m0_vader_retries_once_when_planner_returns_stale_ids(self):
        class Planner:
            def __init__(self):
                self.histories = []
                self.actions = [
                    PlannerAction("obs_0000", 0, "help", {"request_type": "clear_target_area", "target": "hub", "request": "clear blocker"}),
                    PlannerAction("obs_0001", 0, "help", {"request_type": "clear_target_area", "target": "hub", "request": "clear blocker"}),
                ]

            def next_action(self, observation, history, budgets):
                self.histories.append(history)
                return self.actions.pop(0), {"provider": "test"}

        planner = Planner()
        observation = RepairObservation("episode", "obs_0001", 0, 1.0)
        action, meta = _fresh_action(planner, observation, [], {"help_remaining": 1}, 0, retry_invalid_output=True)
        self.assertEqual(action.observation_id, "obs_0001")
        self.assertIn("copy its observation_id and control_epoch exactly", planner.histories[1][0]["protocol_error"])
        self.assertTrue(meta["stale_action_retry"])

    def test_m0_vader_retries_once_after_planner_format_error(self):
        class Planner:
            def __init__(self):
                self.histories = []

            def next_action(self, observation, history, budgets):
                self.histories.append(history)
                if len(self.histories) == 1:
                    try:
                        raise ValueError("not one JSON action")
                    except ValueError as exc:
                        raise PlannerUnavailable("planner response invalid") from exc
                return PlannerAction("obs_0001", 0, "help", {"request_type": "clear_target_area", "target": "hub", "request": "clear blocker"}), {"provider": "test"}

        planner = Planner()
        observation = RepairObservation("episode", "obs_0001", 0, 1.0)
        action, meta = _fresh_action(planner, observation, [], {"help_remaining": 1}, 0, retry_invalid_output=True)
        self.assertEqual(action.observation_id, "obs_0001")
        self.assertIn("Return exactly one JSON object", planner.histories[1][0]["protocol_error"])
        self.assertTrue(meta["action_format_retry"])

    def test_repair_planner_disables_help_when_budget_is_empty(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                action = {"observation_id": "obs_0000", "control_epoch": 0, "action": "finish", "args": {}}
                return json.dumps({"choices": [{"message": {"content": json.dumps(action)}}]}).encode()

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode())
            return Response()

        planner = HttpRepairPlanner("http://local/v1/chat/completions", "text-planner")
        with patch.dict(os.environ, {"HRC_REPAIR_API_KEY": "local"}):
            with patch("hrc_repair.planner.urllib.request.urlopen", side_effect=fake_urlopen):
                planner.next_action(RepairObservation("episode", "obs_0000", 0, 1.0), [], {"help_remaining": 0})
        user = json.loads(captured["payload"]["messages"][1]["content"])
        self.assertFalse(user["help_allowed"])


if __name__ == "__main__":
    unittest.main()
