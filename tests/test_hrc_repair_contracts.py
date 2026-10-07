from hrc_repair.contracts import HelpReport, Observation, PlannerAction
from hrc_repair.adapter import ContractAdapter, IsaacPersistentWorkerAdapter
from hrc_repair.observer import MetricsObserver
import json
import io
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def test_help_action_rejects_extra_motion_fields():
    try:
        PlannerAction.from_dict({"observation_id": "o", "control_epoch": 0, "action": "help", "args": {"request_type": "clear_target_area", "target": "hub", "request": "clear", "delta_m": [1, 2, 3]}})
    except ValueError:
        return
    raise AssertionError("help accepted an unregistered motion field")


def test_pick_action_accepts_only_registered_cover_target():
    action = PlannerAction.from_dict({"observation_id": "o", "control_epoch": 0, "action": "pick", "args": {"target": "cover"}})
    assert action.action == "pick"
    try:
        PlannerAction.from_dict({"observation_id": "o", "control_epoch": 0, "action": "pick", "args": {"target": "cover", "pose": [0, 0, 0]}})
    except ValueError:
        return
    raise AssertionError("pick accepted an unregistered pose field")


def test_help_report_keeps_rgb_paths_out_of_planner_serialization():
    report = HelpReport("h", "clear_target_area", "hub", "succeeded", "cleared", 1.0, 2.0, 0, 1, {"head_rgb": "/private/frame.png"})
    assert "frames" not in report.to_dict()


def test_stepwise_adapter_only_releases_the_expected_skill_command():
    with tempfile.TemporaryDirectory() as directory:
        adapter = IsaacPersistentWorkerAdapter(
            scenario="nominal", repo_root=Path(directory), config={}, out_dir=Path(directory)
        )
        adapter._command_path = Path(directory) / "command.json"
        adapter._awaiting_action = "pick"
        adapter._send_action("place")
        assert adapter._awaiting_action == "pick"
        assert not adapter._command_path.exists()
        adapter._send_action("pick")
        assert adapter._awaiting_action is None
        assert json.loads(adapter._command_path.read_text()) == {"action": "pick"}


def test_adapter_preserves_configured_right_gripper_release():
    with tempfile.TemporaryDirectory() as directory, patch("hrc_repair.adapter.subprocess.Popen") as popen:
        adapter = IsaacPersistentWorkerAdapter(
            scenario="nominal", repo_root=Path(directory),
            config={"controller": {"m0_release_right_after_lift": True}},
            out_dir=Path(directory),
        )
        adapter._start_worker()
        assert popen.call_args.kwargs["env"]["M1_RELEASE_RIGHT_AFTER_LIFT"] == "1"
        adapter._log_handle.close()


def test_rgb_repair_scatter_enables_skill_observation_boundaries_and_replan():
    with tempfile.TemporaryDirectory() as directory, patch("hrc_repair.adapter.subprocess.Popen") as popen:
        adapter = IsaacPersistentWorkerAdapter(
            scenario="nominal", repo_root=Path(directory),
            config={
                "controller": {"m0_correct_orientation_at_insertion_above": True},
                "scene": {"initial_layout": "scatter", "scatter_physical_supports": True},
            },
            out_dir=Path(directory), rgb_observer=True, initial_rgb_observation=True,
        )
        with patch.dict(os.environ, {"M1_GRASP_CORRECTION_X_DEG": "-4", "M1_GRASP_CORRECTION_Y_DEG": "-5"}):
            adapter._start_worker()
        worker_env = popen.call_args.kwargs["env"]
        assert worker_env["M1_M0_STEPWISE_ACTIONS"] == "1"
        assert worker_env["M1_M0_REPLAN_AFTER_HELP"] == "1"
        assert worker_env["M1_GRASP_CORRECTION_X_DEG"] == "-4"
        assert worker_env["M1_GRASP_CORRECTION_Y_DEG"] == "-5"
        assert worker_env["M1_M0_REPAIR_RGB"] == "1"
        assert worker_env["M1_M0_VADER_VQA"] == "0"
        assert worker_env["M1_CORRECT_ORIENTATION_AT_INSERTION_ABOVE"] == "1"
        assert worker_env["M1_RELEASE_RIGHT_AT_INSERTION_ABOVE"] == "0"
        adapter._log_handle.close()


def test_insertion_orientation_correction_is_recorded_in_resolved_config():
    from hrc_repair import run

    config = {"controller": {}}
    with (
        patch.object(run, "load_config", return_value=(Path("config.yaml"), config)),
        patch.object(run, "preflight", return_value={"errors": []}),
        patch.object(run, "run_episode", return_value={"pipeline_success": True}) as run_episode,
        patch.object(sys, "stdout", io.StringIO()),
    ):
        assert run.main(["--correct-orientation-at-insertion-above"]) == 0
    resolved = run_episode.call_args.args[1]
    assert resolved["controller"]["m0_correct_orientation_at_insertion_above"] is True


def test_right_gripper_handoff_at_insertion_above_is_recorded_and_forwarded():
    from hrc_repair import run

    config = {"controller": {}}
    with (
        patch.object(run, "load_config", return_value=(Path("config.yaml"), config)),
        patch.object(run, "preflight", return_value={"errors": []}),
        patch.object(run, "run_episode", return_value={"pipeline_success": True}) as run_episode,
        patch.object(sys, "stdout", io.StringIO()),
    ):
        assert run.main(["--dual-gripper", "--release-right-at-insertion-above"]) == 0
    resolved = run_episode.call_args.args[1]
    assert resolved["controller"]["m0_release_right_at_insertion_above"] is True

    with tempfile.TemporaryDirectory() as directory, patch("hrc_repair.adapter.subprocess.Popen") as popen:
        adapter = IsaacPersistentWorkerAdapter(
            scenario="nominal", repo_root=Path(directory),
            config={"controller": {"m0_release_right_at_insertion_above": True}},
            out_dir=Path(directory),
        )
        adapter._start_worker()
        assert popen.call_args.kwargs["env"]["M1_RELEASE_RIGHT_AT_INSERTION_ABOVE"] == "1"
        adapter._log_handle.close()


def test_place_rgb_requirement_includes_centering_and_bolt_hole_alignment():
    from hrc_repair.state_recognizer import expected_place_outcome

    prompt = expected_place_outcome({"scene": {}}).lower()
    assert "concentric" in prompt
    assert "bolt holes" in prompt
    assert "unknown" in prompt


def test_dual_gripper_cli_transports_with_both_grippers_unless_release_is_requested():
    from hrc_repair import run

    cases = (
        ([], False, None),
        (["--release-right-after-lift"], True, None),
        (["--episode-wall-timeout-s", "3600"], False, 3600),
    )
    for extra_args, expected_release, expected_timeout in cases:
        config = {"controller": {"m0_release_right_after_lift": True}}
        with (
            patch.object(run, "load_config", return_value=(Path("config.yaml"), config)),
            patch.object(run, "preflight", return_value={"errors": []}),
            patch.object(run, "run_episode", return_value={"pipeline_success": True}) as run_episode,
            patch.object(sys, "stdout", io.StringIO()),
        ):
            assert run.main(["--dual-gripper", *extra_args]) == 0
        resolved = run_episode.call_args.args[1]
        assert resolved["controller"]["m0_synchronous_dual_transport"] is True
        assert resolved["controller"]["m0_release_right_after_lift"] is expected_release
        if expected_timeout is not None:
            assert resolved["budgets"]["episode_wall_timeout_s"] == expected_timeout


def test_blocked_contract_requires_help_then_fresh_epoch():
    adapter = ContractAdapter(scenario="blocked")
    obs = adapter.reset("e", 0, __import__("pathlib").Path("/tmp"))
    first = adapter.place(obs, __import__("pathlib").Path("/tmp"))
    assert first.outcome == "failed"
    report = adapter.help(obs, __import__("pathlib").Path("/tmp"))
    assert report.status == "succeeded"
    obs2 = Observation("e", "o2", 2, 1.0, held="yes", target_visible="yes")
    second = adapter.place(obs2, __import__("pathlib").Path("/tmp"))
    assert second.outcome == "succeeded"


def test_observer_preserves_unknown_for_nonphysical_result():
    adapter = ContractAdapter(scenario="nominal")
    obs = adapter.reset("e", 0, __import__("pathlib").Path("/tmp"))
    result = adapter.place(obs, __import__("pathlib").Path("/tmp"))
    public = MetricsObserver().after_skill("e", 0, result)
    assert public.placed == "yes"
    assert public.release_observed == "yes"


def test_public_boundary_drops_evaluator_verdict_and_simulator_stability():
    from hrc_repair.adapter import _public_contact_summary
    from hrc_repair.run import _public_skill_result
    from hrc_repair.contracts import SkillResult

    sensors = _public_contact_summary({"records": [{
        "label": "retract",
        "hub_casing_force_norm": 2.0,
        "hub_speed_mps": 0.0,
        "gripper_to_hub_contact_force_norm_N": {"left": 0.0},
        "gripper_joint": [0.05],
    }]})
    assert sensors == {
        "support_contact": True,
        "release_contact_free": True,
        "released": True,
        "blocked_guard": False,
    }

    with tempfile.TemporaryDirectory() as directory:
        adapter = IsaacPersistentWorkerAdapter(
            scenario="nominal", repo_root=Path(directory),
            config={"experiment": {"feedback_mode": "public_contact", "task_mode": "placement"}},
            out_dir=Path(directory),
        )
        adapter._run_dir = Path(directory)
        physical_result = adapter._result_from_metrics(
            Observation("e", "o", 0, 1.0),
            {"placement_score": {"success": False}, "physical_place_score": {"total": 1, "max": 4}, "records": [{"hub_speed_mps": 0.0}]},
            started=1.0,
        )
        assert physical_result.outcome == "unknown"
        assert physical_result.failure_code is None
        assert "placement_score" not in physical_result.feedback

    public = _public_skill_result(SkillResult(
        "e", "o", 0, "place", "completed", "unknown", "goal_not_reached",
        feedback={
            "metrics_path": "/private/metrics.json",
            "physical_place_score": {"total": 2, "max": 4},
            "public_evidence": "physical place completed",
            "public_sensor": {
                **sensors, "stable_retract": True, "placement_observed": True,
                "hub_speed_mps": 0.0,
            },
            "stderr_tail": "private stack trace",
            "returncode": 1,
        },
        frames_after=["/private/frame.png"],
    ))
    assert public.failure_code is None
    assert public.outcome == "unknown"
    assert public.frames_after == []
    assert "metrics_path" not in public.feedback
    assert "stderr_tail" not in public.feedback
    assert "returncode" not in public.feedback
    observation = MetricsObserver(feedback_mode="public_contact").after_skill("e", 0, public)
    serialized = json.dumps(observation.to_dict())
    assert "goal_not_reached" not in serialized
    assert "physical_place_score" not in serialized
    assert "placement_observed" not in serialized
    assert "stable_retract" not in serialized
    assert "/private/" not in serialized
