import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hrc_m1.contracts import Decision, EpisodeState, Observation, SkillResult
from hrc_m1.evaluator import IndependentSeatEvaluator, SeatMeasurement
from hrc_m1.logger import EventLogger
from hrc_m1.state_machine import M1StateMachine


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.episode = "e1"
        self.obs = Observation(
            episode_id=self.episode,
            observation_id="e1:obs:0000",
            timestamp=0.0,
            sensor_availability={"rgb": True},
        )

    def test_decision_rejects_unknown_and_evaluator_fields(self):
        base = {
            "schema_version": "m1-decision-v1",
            "episode_id": self.episode,
            "observation_id": self.obs.observation_id,
            "action": "PICK",
            "target_part": "Hub_Cover_Output_Top",
        }
        with self.assertRaises(ValueError):
            Decision.from_dict({**base, "unknown": 1})
        with self.assertRaises(ValueError):
            Decision.from_dict({**base, "ground_truth": {"success": True}})

    def test_state_machine_rejects_stale_decision_and_completes_release_path(self):
        sm = M1StateMachine(self.episode)
        sm.reset()
        sm.accept_observation(self.obs)
        with self.assertRaises(ValueError):
            sm.accept_decision(Decision.from_dict({"schema_version": "m1-decision-v1", "episode_id": self.episode, "observation_id": "stale", "action": "OBSERVE"}))
        decision = Decision.from_dict({"schema_version": "m1-decision-v1", "episode_id": self.episode, "observation_id": self.obs.observation_id, "action": "RELEASE_RETRACT", "target_part": "Hub_Cover_Output_Top"})
        sm.accept_decision(decision)
        sm.skill_started("release_retract")
        sm.skill_finished(SkillResult(self.episode, self.obs.observation_id, "release_retract", "SUCCEEDED"))
        sm.verify("SUCCESS")
        self.assertEqual(sm.state, EpisodeState.DONE)

    def test_unknown_verification_holds_and_explicit_abort_is_safe(self):
        sm = M1StateMachine(self.episode)
        sm.reset()
        sm.accept_observation(self.obs)
        decision = Decision.from_dict(
            {
                "schema_version": "m1-decision-v1",
                "episode_id": self.episode,
                "observation_id": self.obs.observation_id,
                "action": "PICK",
                "target_part": "Hub_Cover_Output_Top",
            }
        )
        sm.accept_decision(decision)
        sm.skill_started("pick")
        sm.skill_finished(SkillResult(self.episode, self.obs.observation_id, "pick", "FAILED"))
        sm.verify("UNKNOWN")
        self.assertEqual(sm.state, EpisodeState.SAFE_HOLD)
        sm.abort("no safe recovery")
        self.assertEqual(sm.state, EpisodeState.ABORT)

    def test_pending_calibration_cannot_be_task_success(self):
        measurement = SeatMeasurement(
            axial_depth_m=0.062,
            radial_error_m=0.0,
            tilt_deg=0.0,
            penetration_m=0.0,
            released=True,
            stable=True,
            contact_valid=True,
            settle_speed_mps=0.0,
            settle_window_s=1.0,
        )
        pending = IndependentSeatEvaluator(calibration_status="pending").evaluate(measurement)
        self.assertTrue(pending.physical_criteria_passed)
        self.assertFalse(pending.task_success)
        self.assertEqual(pending.status, "PROVISIONAL_PASS")

    def test_settle_boundary_allows_float_roundoff(self):
        measurement = SeatMeasurement(
            axial_depth_m=0.062,
            radial_error_m=0.0,
            tilt_deg=0.0,
            penetration_m=0.0,
            released=True,
            stable=True,
            contact_valid=True,
            settle_speed_mps=0.0,
            settle_window_s=0.49999999999999994,
        )
        result = IndependentSeatEvaluator(calibration_status="pending").evaluate(measurement)
        self.assertTrue(result.physical_criteria_passed)
        self.assertEqual(result.status, "PROVISIONAL_PASS")

    def test_private_logger_file_and_credential_redaction(self):
        with tempfile.TemporaryDirectory() as directory:
            with EventLogger(directory, {"api_key": "do-not-write", "task_id": "m1"}) as logger:
                logger.event("test", {"token": "do-not-write", "value": 1})
                logger.evaluator_event("truth", {"fault_cause": "private"})
            manifest = json.loads(Path(directory, "manifest.json").read_text())
            events = Path(directory, "events.jsonl").read_text()
            self.assertNotIn("do-not-write", json.dumps(manifest))
            self.assertNotIn("do-not-write", events)
            self.assertEqual(Path(directory, "evaluator_private.jsonl").stat().st_mode & 0o777, 0o600)

    def test_human_jog_bound_is_enforced_by_state_machine(self):
        sm = M1StateMachine(self.episode)
        sm.reset()
        sm.accept_observation(self.obs)
        sm.safe_stop("test")
        sm.request_help("test")
        sm.take_control()
        with self.assertRaises(ValueError):
            sm.human_jog({"type": "cartesian_delta", "delta_m": [0.011, 0.0, 0.0]})

    def test_handoff_requires_fresh_memory_update_before_replan(self):
        sm = M1StateMachine(self.episode)
        sm.reset()
        sm.accept_observation(self.obs)
        sm.safe_stop("guarded insertion failed")
        sm.request_help("alignment correction needed")
        sm.take_control()
        sm.return_control()
        self.assertIsNone(sm.pending_decision)
        fresh = Observation(
            episode_id=self.episode,
            observation_id="e1:obs:0001",
            timestamp=1.0,
            sensor_availability={"rgb": True},
        )
        sm.accept_observation(fresh)
        self.assertEqual(sm.state, EpisodeState.UPDATE_MEMORY)
        sm.update_memory()
        self.assertEqual(sm.state, EpisodeState.REPLAN)

    def test_model_boundary_loads_local_frame_as_data_uri(self):
        from hrc_m1.observer import _frame_data_uri

        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.png"
            frame.write_bytes(b"png-test")
            uri = _frame_data_uri(str(frame))
            self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_unconfigured_vlm_is_explicitly_unknown(self):
        from hrc_m1.observer import ObserverVerifier, VLMVerdict

        result = ObserverVerifier().verify(self.obs)
        self.assertEqual(result.verdict, VLMVerdict.UNKNOWN)
        self.assertIn("not configured", result.evidence)

    def test_planner_timeout_becomes_unavailable_without_rule_fallback(self):
        from hrc_m1.planner import HttpPlanner, PlannerUnavailable

        planner = HttpPlanner("http://127.0.0.1:9/v1/chat/completions", "test-model")
        with patch.dict(os.environ, {"HRC_M1_API_KEY": "test-only"}, clear=False):
            with patch("hrc_m1.planner.urllib.request.urlopen", side_effect=TimeoutError):
                with self.assertRaises(PlannerUnavailable):
                    planner.next_decision(self.obs, EpisodeState.PLAN)

    def test_pick_cannot_infer_hold_without_public_grasp_signal(self):
        from hrc_m1.roco_adapter import RocoTaskAdapter

        adapter = object.__new__(RocoTaskAdapter)
        adapter.env = SimpleNamespace()
        self.assertEqual(adapter._held_status(), ("UNKNOWN", "public_grasp_sensor_unavailable"))
        adapter.env.grasp_verification = lambda: ("HELD_CONFIRMED", "calibrated_pair_sensor")
        self.assertEqual(adapter._held_status(), ("HELD_CONFIRMED", "calibrated_pair_sensor"))


if __name__ == "__main__":
    unittest.main()
