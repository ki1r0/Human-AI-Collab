"""Unit-only episode wiring tests; no Isaac simulation or live model is used."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from runtime.bolt_harness import episode


PRIVATE_SENTINEL = "PRIVATE_EPISODE_TEST_SENTINEL"


def _skill_call(skill, mode):
    return {
        "tool": "execute_skill", "skill": skill, "target_id": "socket_1",
        "mode": mode, "max_duration_s": 30.0,
    }


def _settings():
    return {
        "task_spec": {
            "task_id": "unit_bolt_insert",
            "target_id": "socket_1",
            "socket_id": "cover_socket_1",
            "nominal_dimensions": {"bolt_diameter_m": 0.006, "bolt_length_m": 0.026},
            "assembly_direction": [0.0, 0.0, -1.0],
            "private_fixture_note": PRIVATE_SENTINEL,
        },
        "executor": {"max_gripper_force_n": 12.0, "max_insertion_force_n": 4.0},
        "tolerances": {
            "max_cap_support_gap_m": 0.001,
            "min_casing_entry_depth_m": 0.001,
            "max_shaft_radial_error_m": 0.001,
            "max_penetration_m": 0.001,
            "max_axial_motion_since_release_m": 0.001,
            "max_relative_linear_speed_mps": 0.01,
            "max_relative_angular_speed_radps": 0.1,
            "held_seat_dwell_s": 0.25,
            "stable_dwell_s": 0.5,
            "max_sample_gap_s": 0.2,
        },
        "sampling_criteria": {
            "cap_support": {
                "max_cap_patch_error_m": 0.001,
                "max_contact_separation_m": 0.001,
                "min_normal_alignment_cosine": 0.8,
                "normal_axis_sign": 1,
                "min_contact_force_n": 0.1,
            },
            "min_cap_grasp_local_y_m": 0.018,
            "max_cap_grasp_patch_error_m": 0.001,
            "max_cap_grasp_contact_separation_m": 0.001,
            "min_cap_grasp_force_n": 0.1,
            "max_cap_grasp_normal_alignment_cosine": 0.5,
            "max_robot_contact_separation_m": 0.001,
            "min_robot_contact_force_n": 0.1,
            "max_controller_tcp_linear_speed_mps": 0.01,
            "max_controller_tcp_angular_speed_radps": 0.1,
            "max_controller_arm_joint_speed_radps": 0.1,
            "max_tcp_tracking_error_m": 0.001,
            "max_tcp_orientation_tracking_error_rad": 0.01,
            "release_gripper_width_m": 0.04,
            "min_pickup_root_lift_m": 0.001,
        },
    }


@dataclass
class _Measurement:
    physically_held: bool


@dataclass
class _Status:
    seat_ready: bool
    task_success: bool


@dataclass
class _Sample:
    time_s: float
    measurement: _Measurement
    status: _Status
    pickup_proven: bool


class _Camera:
    def __init__(self):
        self.frame = np.asarray([1], dtype=np.int64)
        self._timestamp_last_update = np.asarray([0.0], dtype=np.float64)
        self.data = SimpleNamespace(output={"rgb": np.full((1, 4, 6, 3), 32, dtype=np.uint8)})


class _Env:
    """Duck-typed runner fixture; its step is not a physics simulation."""

    def __init__(self, *, initially_held=False):
        self.num_envs = 1
        self.cfg = SimpleNamespace(sim=SimpleNamespace(dt=1.0 / 12.0), decimation=1)
        self.actions = np.zeros((1, 6), dtype=np.float32)
        self.episode_length_buf = np.asarray([0], dtype=np.int64)
        self.max_episode_length = 1000
        self.wrench_baseline_stats = {"stationary_zero_reference_only": True}
        self.wrench = np.zeros((1, 6), dtype=np.float32)
        self.phase = "held_at_start" if initially_held else "unheld"
        self.step_calls = 0
        self.steps_per_skill = 1
        self._robot = SimpleNamespace(_data=SimpleNamespace(_sim_timestamp=0.0))
        self.task_camera = _Camera()
        self.scene = SimpleNamespace(sensors={"task_rgb_camera": self.task_camera})
        self.private_contact_source_calls = 0
        self.gt_private_value = PRIVATE_SENTINEL

    def step(self, action):
        self.step_calls += 1
        self.episode_length_buf[0] += 1
        self._robot._data._sim_timestamp += 1.0 / 12.0
        self.task_camera.frame[0] += 1
        self.task_camera._timestamp_last_update[0] = self._robot._data._sim_timestamp
        return None, None, np.asarray([False]), np.asarray([False]), {}

    def public_wrench(self):
        return self.wrench.copy()

    def get_public_executor_state(self):
        return {
            "tcp_pose": [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]],
            "tcp_velocity": [[0.0] * 6],
            "joint_positions": [[0.0] * 9],
            "joint_velocities": [[0.0] * 9],
            "gripper_width_m": 0.04,
            "finger_bolt_contacts": [True, True],
            "private_debug": PRIVATE_SENTINEL,
        }

    def get_private_bolt_contact_source(self):
        self.private_contact_source_calls += 1
        return {"filter_map": [{"private": PRIVATE_SENTINEL}]}


class _OffGridCameraEnv(_Env):
    """Unit fixture with sensor acquisition cadence offset from episode-start time."""

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(sim=SimpleNamespace(dt=1.0 / 120.0), decimation=2)
        self._robot._data._sim_timestamp = 203.0 / 60.0
        self.task_camera.frame[0] = 7
        self.task_camera._timestamp_last_update[0] = 201.0 / 60.0
        self.next_camera_time = 206.0 / 60.0
        self.steps_per_skill = 10

    def step(self, action):
        self.step_calls += 1
        self.episode_length_buf[0] += 1
        self._robot._data._sim_timestamp += 1.0 / 60.0
        if self._robot._data._sim_timestamp + 1e-12 >= self.next_camera_time:
            self.task_camera.frame[0] += 1
            self.task_camera._timestamp_last_update[0] = self.next_camera_time
            self.next_camera_time += 1.0 / 12.0
        return None, None, np.asarray([False]), np.asarray([False]), {}


class _Adapter:
    def __init__(self, env, *_args, **_kwargs):
        self.env = env
        self._trace = []

    def observe(self):
        phase = self.env.phase
        held = phase in {"held_at_start", "picked", "transported", "inserted"}
        status = _Status(seat_ready=phase == "inserted", task_success=phase == "released")
        sample = _Sample(
            self.env._robot._data._sim_timestamp,
            _Measurement(physically_held=held),
            status,
            pickup_proven=phase in {"transported", "inserted", "released"},
        )
        self._trace.append(sample)
        return status


class _Executor:
    def __init__(self, env, **_kwargs):
        self.env = env
        self.skills = []

    def execute_skill(self, skill, _target_id, **_kwargs):
        self.skills.append(skill)
        self.env.phase = {
            "pick": "picked",
            "transport": "transported",
            "insert_and_seat": "inserted",
            "release_and_retract": "released",
        }[skill]
        for _ in range(self.env.steps_per_skill):
            self.env.step(np.zeros((1, 6), dtype=np.float32))
        return {"status": "completed"}


class _FailedInsertionExecutor(_Executor):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.private_diagnostics = {"insertion": None}

    def execute_skill(self, skill, _target_id, **_kwargs):
        if skill != "insert_and_seat":
            return super().execute_skill(skill, _target_id, **_kwargs)
        self.skills.append(skill)
        self.env.phase = "transported"
        self.env.step(np.zeros((1, 6), dtype=np.float32))
        endpoint = {
            "step": 1,
            "phase": "endpoint",
            "actual_pose_at_seat": False,
            "command_pose_at_seat": True,
            "at_seat": False,
        }
        self.private_diagnostics["insertion"] = {
            "source": "public_executor_state_and_public_wrench",
            "exit_reason": "no_progress_before_seat",
            "sample_limit": 2048,
            "dropped_sample_count": 0,
            "samples": [endpoint],
            "endpoint": endpoint,
        }
        return {"status": "motion_timeout"}


class _Recorder:
    def __init__(self, path, *, fps):
        self.path = path
        self.fps = fps
        self.frames = []

    @property
    def frame_count(self):
        return len(self.frames)

    def append(self, rgb, sim_time_s):
        self.frames.append((rgb.copy(), sim_time_s))

    def close(self):
        if self.frames:
            self.path.write_bytes(b"unit-test video placeholder; not physical evidence")
            return None
        return "no frames"


class BoltEpisodeUnitTests(unittest.TestCase):
    def run_with_fakes(
        self, env, directory, mode="scripted", *, agent=None, executor_cls=_Executor,
        recorder_cls=_Recorder, settings=None,
    ):
        patches = [
            patch.object(episode, "BoltSeatSamplingAdapter", _Adapter),
            patch.object(episode, "BoltSkillExecutor", executor_cls),
            patch.object(episode, "EpisodeRGBVideoRecorder", recorder_cls),
            patch.object(episode, "_save_png", side_effect=lambda path, _rgb: path.write_bytes(b"unit png")),
        ]
        if agent is not None:
            patches.append(patch.object(episode, "BoltHarnessAgent", return_value=agent))
        with patches[0], patches[1], patches[2], patches[3]:
            if agent is not None:
                with patches[4]:
                    return episode.run_bolt_episode(env, settings or _settings(), directory, mode)
            return episode.run_bolt_episode(env, settings or _settings(), directory, mode)

    def test_scripted_dummy_path_runs_same_checks_and_finalizes_test_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            env = _Env()
            result = self.run_with_fakes(env, directory)

            self.assertEqual(result.exit_code, 0)
            self.assertTrue(result.task_success)
            self.assertTrue(result.video_finalized)
            self.assertFalse(result.model_evidence)
            self.assertEqual(env.private_contact_source_calls, 1)
            self.assertEqual(env.step_calls, 4)
            self.assertNotIn("step", env.__dict__)
            self.assertTrue((result.run_dir / "episode.mp4").is_file())
            self.assertTrue((result.run_dir / "private/evaluator_trace.jsonl").is_file())
            summary = json.loads((result.run_dir / "run_summary.json").read_text())
            self.assertEqual(summary["agent_mode"], "scripted")
            self.assertTrue(summary["starts_ungrasped"])
            self.assertTrue(summary["physical_cap_grasp"])
            public_trace = (result.run_dir / "public_trace.jsonl").read_text()
            config_snapshot = (result.run_dir / "config.yaml").read_text()
            self.assertNotIn(PRIVATE_SENTINEL, public_trace)
            self.assertNotIn(PRIVATE_SENTINEL, config_snapshot)

    def test_vlm_help_request_is_persisted_locally_without_claiming_task_success(self):
        class HelpAgent:
            trace_callback = None

            def next_tool_call(self, observation, _task, _history, *, completed=False):
                self.trace_callback({"event": "request"})
                self.trace_callback({"event": "response"})
                return {
                    "tool": "send_help_request", "target": "socket_1",
                    "operation": "inspect bolt and socket alignment",
                    "allowed_scope": "specified assembly only",
                    "desired_postconditions": ["bolt remains held", "insertion can be reassessed safely"],
                    "evidence_refs": [f"observation:{observation['observation_id']}"],
                    "observed_problem": "Insertion has not completed; cause is unverified.",
                }

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings()
            settings["model"] = {
                "endpoint": "http://127.0.0.1:18081/v1/chat/completions", "model": "test-cosmos",
            }
            env = _Env()
            result = self.run_with_fakes(
                env, directory, "vlm", agent=HelpAgent(), settings=settings,
            )

            self.assertEqual(result.status, "help_requested")
            self.assertEqual(result.exit_code, 1)
            self.assertFalse(result.task_success)
            self.assertTrue(result.model_evidence)
            self.assertEqual(env.step_calls, 0)
            request = json.loads((result.run_dir / "help_request.json").read_text())
            self.assertEqual(set(request), {
                "request_id", "target", "operation", "allowed_scope", "desired_postconditions",
                "evidence_refs", "observed_problem",
            })
            self.assertEqual(request["target"], "socket_1")
            self.assertEqual(request["request_id"], "help-01")
            self.assertEqual(request["evidence_refs"], ["observation:task_rgb_camera-00000001"])
            self.assertIn("cause is unverified", request["observed_problem"])
            summary = json.loads((result.run_dir / "run_summary.json").read_text())
            self.assertEqual(summary["status"], "help_requested")
            self.assertFalse(summary["task_success"])
            self.assertTrue(summary["help_requested"])
            self.assertEqual(summary["help_delivery"], "local_outbox")
            self.assertNotIn("helper_gt_success", json.dumps(request))
            self.assertNotIn(PRIVATE_SENTINEL, (result.run_dir / "public_trace.jsonl").read_text())

    def test_existing_help_outbox_is_preserved_without_starting_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "help_request.json"
            request_path.write_text("preexisting request", encoding="utf-8")
            env = _Env()

            result = episode.run_bolt_episode(env, _settings(), directory, "scripted")

            self.assertEqual(result.status, "output_conflict")
            self.assertEqual(env.step_calls, 0)
            self.assertEqual(request_path.read_text(encoding="utf-8"), "preexisting request")

    def test_task_projection_uses_only_allowlisted_public_fields(self):
        normalized, snapshot = episode._validated_settings(_settings(), "scripted")

        serialized = json.dumps({"task": normalized["task_spec"], "snapshot": snapshot})
        self.assertNotIn(PRIVATE_SENTINEL, serialized)
        self.assertNotIn("nominal_pose", normalized["task_spec"])
        self.assertIn("assembly_direction", normalized["task_spec"])

    def test_exact_final_pose_is_rejected_from_public_task_settings(self):
        settings = _settings()
        settings["task_spec"]["nominal_pose"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        with self.assertRaisesRegex(ValueError, "exact final root coordinates"):
            episode._validated_settings(settings, "scripted")

    def test_missing_calibration_fails_before_stepping(self):
        settings = _settings()
        del settings["tolerances"]["max_penetration_m"]
        with tempfile.TemporaryDirectory() as directory:
            env = _Env()
            result = episode.run_bolt_episode(env, settings, directory, "scripted")

            self.assertNotEqual(result.exit_code, 0)
            self.assertEqual(env.step_calls, 0)
            self.assertFalse(result.task_success)
            self.assertTrue((result.run_dir / "private/exception.txt").is_file())

    def test_initial_held_bolt_is_rejected_before_any_tool_action(self):
        with tempfile.TemporaryDirectory() as directory:
            env = _Env(initially_held=True)
            result = self.run_with_fakes(env, directory)

            self.assertNotEqual(result.exit_code, 0)
            self.assertFalse(result.task_success)
            self.assertEqual(env.step_calls, 0)
            self.assertNotIn("step", env.__dict__)

    def test_off_grid_initial_capture_uses_sensor_time_and_keeps_frame_association(self):
        recorders = []

        class CadenceRecorder(_Recorder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                recorders.append(self)

            def append(self, rgb, acquisition_time_s):
                if self.frames:
                    self.assert_cadence(acquisition_time_s)
                super().append(rgb, acquisition_time_s)

            def assert_cadence(self, acquisition_time_s):
                self_delta = acquisition_time_s - self.frames[-1][1]
                if not np.isclose(self_delta, 1.0 / 12.0, atol=1e-10, rtol=0.0):
                    raise ValueError(f"camera cadence {self_delta} is not 1/12 second")

        with tempfile.TemporaryDirectory() as directory:
            env = _OffGridCameraEnv()
            result = self.run_with_fakes(env, directory, recorder_cls=CadenceRecorder)

            self.assertEqual(result.exit_code, 0)
            timestamps = [frame_time for _rgb, frame_time in recorders[0].frames]
            self.assertEqual(timestamps[0], 201.0 / 60.0)
            for previous, current in zip(timestamps, timestamps[1:]):
                self.assertAlmostEqual(current - previous, 1.0 / 12.0, places=10)

            public_events = [
                json.loads(line) for line in (result.run_dir / "public_trace.jsonl").read_text().splitlines()
            ]
            camera_events = [event for event in public_events if event.get("event") == "camera_frame"]
            self.assertEqual(camera_events[0]["frame_id"], "task_rgb_camera-00000007")
            self.assertEqual(camera_events[1]["frame_id"], "task_rgb_camera-00000008")
            self.assertAlmostEqual(camera_events[0]["timestamp_s"], 201.0 / 60.0)
            self.assertAlmostEqual(camera_events[1]["timestamp_s"], 206.0 / 60.0)

            private_rows = [
                json.loads(line)
                for line in (result.run_dir / "private/evaluator_trace.jsonl").read_text().splitlines()
            ]
            first_new_frame_sample = next(
                row for row in private_rows if np.isclose(row["sample"]["time_s"], 206.0 / 60.0)
            )
            self.assertEqual(first_new_frame_sample["camera_frame_id"], "task_rgb_camera-00000008")

    def test_failed_insertion_diagnostics_are_saved_privately_only(self):
        with tempfile.TemporaryDirectory() as directory:
            env = _Env()
            result = self.run_with_fakes(
                env, directory, executor_cls=_FailedInsertionExecutor
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertFalse(result.task_success)
            diagnostic_path = result.run_dir / "private/executor_diagnostics.json"
            self.assertTrue(diagnostic_path.is_file())
            diagnostics = json.loads(diagnostic_path.read_text())
            self.assertEqual(diagnostics["exit_reason"], "no_progress_before_seat")
            self.assertFalse(diagnostics["endpoint"]["actual_pose_at_seat"])
            self.assertFalse(diagnostics["endpoint"]["at_seat"])
            self.assertEqual(set(diagnostics), {
                "source", "exit_reason", "sample_limit", "dropped_sample_count", "samples", "endpoint",
            })

            public_trace = (result.run_dir / "public_trace.jsonl").read_text()
            self.assertNotIn("no_progress_before_seat", public_trace)
            self.assertNotIn("actual_pose_at_seat", public_trace)

    def test_vlm_transport_failure_never_falls_back_to_scripted_calls(self):
        class FailingModel:
            trace_callback = None

            def next_tool_call(self, *_args, **_kwargs):
                raise RuntimeError("unit transport failure")

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings()
            settings["model"] = {"endpoint": "http://127.0.0.1:18081/v1/chat/completions", "model": "qwen-test"}
            env = _Env()
            failing_model = FailingModel()
            with patch.object(episode, "BoltSeatSamplingAdapter", _Adapter), patch.object(
                episode, "BoltSkillExecutor", _Executor
            ), patch.object(episode, "EpisodeRGBVideoRecorder", _Recorder), patch.object(
                episode, "_save_png", side_effect=lambda path, _rgb: path.write_bytes(b"unit png")
            ), patch.object(episode, "BoltHarnessAgent", return_value=failing_model) as model_factory, patch.object(
                episode, "_ScriptedDevelopmentAgent", side_effect=AssertionError("VLM fallback invoked")
            ):
                result = episode.run_bolt_episode(env, settings, directory, "vlm")

            self.assertNotEqual(result.exit_code, 0)
            self.assertFalse(result.task_success)
            self.assertEqual(env.step_calls, 0)
            self.assertEqual(model_factory.call_args.kwargs["endpoint"], settings["model"]["endpoint"])
            self.assertEqual(model_factory.call_args.kwargs["model"], "qwen-test")
            self.assertFalse(result.model_evidence)
            self.assertNotIn("step", env.__dict__)

    def test_dummy_vlm_actions_without_request_response_evidence_cannot_succeed(self):
        class QueuedTestAgent:
            def __init__(self):
                self.trace_callback = None
                self.calls = [
                    _skill_call("pick", "default"),
                    _skill_call("transport", "default"),
                    _skill_call("insert_and_seat", "compliant"),
                    {"tool": "check_task"},
                    _skill_call("release_and_retract", "default"),
                    {"tool": "check_task"},
                ]

            def next_tool_call(self, *_args, **_kwargs):
                return self.calls.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings()
            settings["model"] = {"endpoint": "http://127.0.0.1:18081/v1/chat/completions", "model": "qwen-test"}
            env = _Env()
            agent = QueuedTestAgent()
            with patch.object(episode, "BoltSeatSamplingAdapter", _Adapter), patch.object(
                episode, "BoltSkillExecutor", _Executor
            ), patch.object(episode, "EpisodeRGBVideoRecorder", _Recorder), patch.object(
                episode, "_save_png", side_effect=lambda path, _rgb: path.write_bytes(b"unit png")
            ), patch.object(episode, "BoltHarnessAgent", return_value=agent):
                result = episode.run_bolt_episode(env, settings, directory, "vlm")

            self.assertTrue(result.task_success)
            self.assertTrue(result.video_finalized)
            self.assertFalse(result.model_evidence)
            self.assertNotEqual(result.exit_code, 0)

    def test_successful_dummy_evaluator_is_still_nonzero_without_finalized_video(self):
        class MissingVideoRecorder(_Recorder):
            def close(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            env = _Env()
            with patch.object(episode, "BoltSeatSamplingAdapter", _Adapter), patch.object(
                episode, "BoltSkillExecutor", _Executor
            ), patch.object(episode, "EpisodeRGBVideoRecorder", MissingVideoRecorder), patch.object(
                episode, "_save_png", side_effect=lambda path, _rgb: path.write_bytes(b"unit png")
            ):
                result = episode.run_bolt_episode(env, _settings(), directory, "scripted")

            self.assertTrue(result.task_success)
            self.assertFalse(result.video_finalized)
            self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
