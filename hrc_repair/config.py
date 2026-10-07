"""Configuration loading and P0 preflight checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("M0 config must contain a YAML object")
    return config_path, data


def _resolve(value: Any, base: Path) -> Any:
    if isinstance(value, str) and (value.startswith(".") or "/" in value) and not value.startswith("/"):
        candidate = (base / value).resolve()
        return str(candidate)
    if isinstance(value, dict):
        return {str(k): _resolve(v, base) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, base) for v in value]
    return value


def resolved_config(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    result = _resolve(config, repo_root())
    result.setdefault("provenance", {})
    result["provenance"].update({"repo_root": str(repo_root()), "config_path": str(config_path)})
    return result


def preflight(config_path: Path, config: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    root = repo_root()
    errors: list[str] = []
    warnings: list[str] = []
    scene = config.get("scene", {})
    robot = config.get("robot", {})
    model = config.get("model", {})
    evaluator = config.get("evaluator", {})
    for rel in ("assets/parts/Hub Cover Output.usd", "assets/parts/Casing Top.usd", "tools/run_m1_isaac_strict_success.sh"):
        if not (root / rel).exists():
            errors.append(f"missing required binding: {rel}")
    required = {
        "scene.robot_prim": scene.get("robot_prim"),
        "scene.cover_prim": scene.get("cover_prim"),
        "scene.hub_prim": scene.get("hub_prim"),
        "scene.registered_goal_frame": scene.get("registered_goal_frame"),
        "scene.meters_per_unit": scene.get("meters_per_unit"),
        "scene.physics_dt_s": scene.get("physics_dt_s"),
        "robot.adapter_entrypoint": robot.get("adapter_entrypoint"),
        "robot.joint_names_in_action_order": robot.get("joint_names_in_action_order"),
        "evaluator.require_release": evaluator.get("require_release"),
    }
    for name, value in required.items():
        if value is None or value == "" or value == []:
            errors.append(f"missing required config field: {name}")
    if config.get("experiment", {}).get("feedback_mode") == "privileged_debug":
        warnings.append("feedback_mode=privileged_debug: public observer is derived from debug metrics, not RGB-only evidence")
    if model.get("allow_silent_fallback", True):
        errors.append("model.allow_silent_fallback must be false")
    if model.get("rgb_observer_enabled", False):
        if not (os.environ.get("HRC_REPAIR_VLM_ENDPOINT") or os.environ.get("HRC_REPAIR_PLANNER_ENDPOINT")):
            errors.append("RGB observer enabled but HRC_REPAIR_VLM_ENDPOINT/HRC_REPAIR_PLANNER_ENDPOINT is not configured")
        if not os.environ.get(str(model.get("api_key_env_name", "HRC_REPAIR_API_KEY"))):
            errors.append("RGB observer credential is not configured in the named environment variable")
        if not str(config.get("experiment", {}).get("feedback_mode", "")).startswith("public_"):
            errors.append("RGB REPAIR observation requires public feedback mode; privileged metrics cannot enter the online observer")
        if config.get("experiment", {}).get("backend") != "isaac_inprocess":
            errors.append("RGB REPAIR observation requires the persistent Isaac worker backend")
        if not config.get("controller", {}).get("replan_after_help", False):
            errors.append("RGB REPAIR adaptation requires an explicit planner decision after helper intervention")
        if str(scene.get("start_stage", "")) == "scattered" and not config.get("controller", {}).get("stepwise_actions", False):
            errors.append("scattered RGB REPAIR requires separate planner-controlled pick and place boundaries")
        if not any(camera.get("enabled") for camera in config.get("cameras", []) if isinstance(camera, dict)):
            errors.append("RGB observer enabled but no camera is enabled")
    if robot.get("use_magic_combine", True):
        errors.append("robot.use_magic_combine must be false")
    if robot.get("allow_part_pose_writes_during_episode", True):
        errors.append("robot.allow_part_pose_writes_during_episode must be false")
    if config.get("experiment", {}).get("backend") == "isaac_subprocess":
        warnings.append("Isaac subprocess backend serializes one simulator writer per episode")
    if strict and errors:
        raise RuntimeError("M0 preflight failed: " + "; ".join(errors))
    return {
        "status": "PASS" if not errors else "WARN",
        "config": str(config_path),
        "repo_root": str(root),
        "errors": errors,
        "warnings": warnings,
        "bindings": {
            "robot_prim": scene.get("robot_prim"),
            "cover_prim": scene.get("cover_prim"),
            "hub_prim": scene.get("hub_prim"),
            "goal_frame": scene.get("registered_goal_frame"),
            "cameras": config.get("cameras", []),
            "physics_dt_s": scene.get("physics_dt_s"),
            "meters_per_unit": scene.get("meters_per_unit"),
        },
    }


__all__ = ["load_config", "resolved_config", "preflight", "repo_root"]
