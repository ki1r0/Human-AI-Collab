"""Independent M0 evaluator over simulator artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def placement_success(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Evaluate M0 placement separately from the stricter M1 seat gate."""
    insertion = metrics.get("insertion_metrics", {}) if isinstance(metrics, dict) else {}
    evaluator = config.get("evaluator", {}) if isinstance(config, dict) else {}
    half = evaluator.get("goal_region_half_extents_m", [0.012, 0.012, 0.020])
    try:
        radial_limit = float(max(float(half[0]), float(half[1])))
        axial_limit = float(half[2])
    except (TypeError, ValueError, IndexError):
        radial_limit, axial_limit = 0.012, 0.020
    position = insertion.get("insert_hub_root_m", [])
    target = insertion.get("seat_target_root_m", [])
    radial = insertion.get("radial_error_m")
    axial = insertion.get("axial_error_m")
    if radial is None and len(position) >= 2 and len(target) >= 2:
        radial = ((float(position[0]) - float(target[0])) ** 2 + (float(position[1]) - float(target[1])) ** 2) ** 0.5
    if axial is None and len(position) >= 3 and len(target) >= 3:
        axial = abs(float(position[2]) - float(target[2]))
    strict = insertion.get("strict_checks", {})
    release_free = bool(strict.get("release_contact_free", False))
    support_contact = float(insertion.get("insert_hub_casing_force_norm_N", 0.0) or 0.0) > 1.0e-3
    retract_drift = float(insertion.get("post_settle_retract_drift_m", insertion.get("retract_drift_m", 999.0)) or 999.0)
    release_motion = float(insertion.get("release_to_insert_motion_m", 999.0) or 999.0)
    release_motion_limit = float(evaluator.get("placement_release_motion_tolerance_m", 0.03))
    retract_limit = float(evaluator.get("placement_retract_drift_tolerance_m", 0.03))
    checks = {
        "support_contact": support_contact,
        "radial_error": radial is not None and float(radial) <= radial_limit,
        "axial_error": axial is not None and float(axial) <= axial_limit,
        "release_contact_free": release_free,
        "release_motion_bounded": release_motion <= release_motion_limit,
        "stable_retract": retract_drift <= retract_limit,
    }
    score = {
        "scheme": "REPAIR_M0_PLACEMENT_V1",
        "success": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "limits": {"radial_m": radial_limit, "axial_m": axial_limit, "release_motion_m": release_motion_limit, "retract_drift_m": retract_limit},
        "measurements": {"radial_error_m": radial, "axial_error_m": axial, "support_force_N": insertion.get("insert_hub_casing_force_norm_N"), "release_to_insert_motion_m": insertion.get("release_to_insert_motion_m"), "post_settle_retract_drift_m": insertion.get("post_settle_retract_drift_m")},
        "interpretation": (
            "Full-physics scattered task; this gate checks final placement, while pick and visual decisions are recorded separately."
            if str(config.get("scene", {}).get("start_stage", "")) == "scattered"
            else "M0 placement from a physically held preplace state; not full pick-and-carry or bolt/6-DoF seating evidence."
        ),
    }
    return bool(score["success"]), score


def evaluate_artifact(out_dir: Path, *, online_status: str, helper_used: bool, scenario: str, feedback_mode: str) -> dict[str, Any]:
    metrics_paths = sorted(out_dir.glob("place_*/metrics.json"))
    latest: dict[str, Any] = {}
    if metrics_paths:
        try:
            latest = json.loads(metrics_paths[-1].read_text(encoding="utf-8"))
        except ValueError:
            latest = {}
    score = latest.get("physical_place_score", {}) if isinstance(latest, dict) else {}
    config: dict[str, Any] = {}
    resolved_path = out_dir / "config_resolved.yaml"
    if resolved_path.exists():
        try:
            import yaml
            loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
            config = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            config = {}
    placement_ok, placement = placement_success(latest, config) if config.get("experiment", {}).get("task_mode") == "placement" else (False, {})
    gt_success = placement_ok if config.get("experiment", {}).get("task_mode") == "placement" else bool(isinstance(score, dict) and int(score.get("total", 0)) >= int(score.get("max", 4)) and int(score.get("max", 0)) == 4)
    if config.get("experiment", {}).get("task_mode") != "placement" and "SUCCESS" in str(latest.get("insertion_verdict", "")):
        gt_success = True
    if scenario == "blocked" and latest.get("blocked_scenario_not_wired"):
        gt_success = False
    return {
        "task_success_gt": gt_success,
        "pipeline_success": bool(gt_success and online_status == "DONE"),
        "post_help_continuation_success": bool(gt_success and helper_used),
        "online_status": online_status,
        "scenario": scenario,
        "feedback_mode": feedback_mode,
        "latest_physics_artifact": str(metrics_paths[-1]) if metrics_paths else None,
        "physical_score": score,
        "placement_score": placement,
        "insertion_verdict": latest.get("insertion_verdict"),
        "video_artifacts": [
            str(p)
            for pattern in ("place_*/episode.mp4", "place_*/inner_wall_probe.mp4", "place_*/rollout.mp4")
            for p in out_dir.glob(pattern)
            if p.is_file() and p.stat().st_size > 0
        ],
    }


__all__ = ["evaluate_artifact"]
