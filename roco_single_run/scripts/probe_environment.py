#!/usr/bin/env python3
"""Stage A integrity probe for the official RoCo Task 1 agent environment.

This script must be run with Isaac Sim's Python.  It deliberately applies only
the post-reset joint positions, so the probe validates the environment contract
without commanding a task motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Template-Galaxea-Lab-Agent-Direct-v0",
    help="Registered Isaac Lab environment ID.",
)
parser.add_argument("--num-steps", type=int, default=3, help="Number of safe hold steps.")
parser.add_argument("--seed", type=int, default=2026, help="Environment reset seed.")
parser.add_argument("--output", type=Path, required=True, help="JSON result path.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import Galaxea_Lab_External.tasks  # noqa: E402,F401
from Galaxea_Lab_External.robots import ACTIVE_ROBOT_BUNDLE  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _tensor_summary(value: torch.Tensor) -> dict:
    """Return a compact, JSON-safe tensor summary."""
    finite = torch.isfinite(value) if torch.is_floating_point(value) else torch.ones_like(value, dtype=torch.bool)
    summary = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite_fraction": float(finite.float().mean().item()),
    }
    if value.numel() and finite.any():
        finite_values = value[finite]
        summary["min"] = float(finite_values.min().item())
        summary["max"] = float(finite_values.max().item())
    return summary


def _as_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().any().item())
    return bool(value)


def main() -> None:
    if args.num_steps < 1:
        raise ValueError("--num-steps must be positive")

    result = {
        "stage": "A",
        "classification": "confirmatory",
        "task": args.task,
        "seed": args.seed,
        "requested_hold_steps": args.num_steps,
        "active_robot_bundle": ACTIVE_ROBOT_BUNDLE.name,
        "status": "RUNNING",
    }
    env = None

    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
        env = gym.make(args.task, cfg=env_cfg)
        base_env = env.unwrapped
        observations, reset_info = env.reset(seed=args.seed)
        policy_obs = observations["policy"]

        controlled_indices = list(base_env._joint_idx)
        controlled_names = [base_env.robot.joint_names[index] for index in controlled_indices]
        hold_action = base_env.robot.data.joint_pos[:, controlled_indices].detach().clone()

        result.update(
            {
                "sim_dt_s": float(base_env.physics_dt),
                "control_dt_s": float(base_env.step_dt),
                "decimation": int(base_env.cfg.decimation),
                "action_space": str(env.action_space),
                "observation_space": str(env.observation_space),
                "robot_joint_count": len(base_env.robot.joint_names),
                "robot_joint_names": list(base_env.robot.joint_names),
                "controlled_joint_indices": controlled_indices,
                "controlled_joint_names_env_order": controlled_names,
                "hold_action": hold_action.detach().cpu().tolist(),
                "reset_info_keys": sorted(reset_info.keys()),
                "observations_at_reset": {
                    name: _tensor_summary(value) for name, value in sorted(policy_obs.items())
                },
            }
        )

        if hold_action.shape != (1, 14):
            raise RuntimeError(f"Expected one 14-D action, received {tuple(hold_action.shape)}")
        if ACTIVE_ROBOT_BUNDLE.name != "r1":
            raise RuntimeError(f"Expected checkpoint-compatible R1 bundle, got {ACTIVE_ROBOT_BUNDLE.name!r}")

        per_step = []
        for step_index in range(args.num_steps):
            observations, reward, terminated, truncated, info = env.step(hold_action)
            current = base_env.robot.data.joint_pos[:, controlled_indices]
            max_abs_drift = float(torch.max(torch.abs(current - hold_action)).item())
            score, score_time_s = base_env.evaluate_score()
            per_step.append(
                {
                    "step": step_index + 1,
                    "max_abs_joint_drift": max_abs_drift,
                    "reward": float(torch.as_tensor(reward).detach().cpu().flatten()[0].item()),
                    "terminated": _as_bool(terminated),
                    "truncated": _as_bool(truncated),
                    "score": int(score),
                    "score_time_s": float(score_time_s),
                    "info_keys": sorted(info.keys()),
                }
            )

        final_policy_obs = observations["policy"]
        required_observations = {
            "head_rgb",
            "left_hand_rgb",
            "right_hand_rgb",
            "head_depth",
            "left_hand_depth",
            "right_hand_depth",
            "left_arm_joint_pos",
            "right_arm_joint_pos",
            "left_gripper_joint_pos",
            "right_gripper_joint_pos",
        }
        missing = sorted(required_observations - set(final_policy_obs))
        camera_keys = [
            "head_rgb",
            "left_hand_rgb",
            "right_hand_rgb",
            "head_depth",
            "left_hand_depth",
            "right_hand_depth",
        ]
        bad_cameras = [
            key
            for key in camera_keys
            if final_policy_obs[key].numel() == 0 or not torch.isfinite(final_policy_obs[key].float()).any()
        ]
        result["steps"] = per_step
        result["observations_after_steps"] = {
            name: _tensor_summary(value) for name, value in sorted(final_policy_obs.items())
        }
        result["checks"] = {
            "r1_bundle": ACTIVE_ROBOT_BUNDLE.name == "r1",
            "controlled_dof_count_14": len(controlled_indices) == 14,
            "required_observations_present": not missing,
            "all_camera_tensors_nonempty_and_partly_finite": not bad_cameras,
            "hold_steps_completed": len(per_step) == args.num_steps,
        }
        result["missing_observations"] = missing
        result["bad_camera_tensors"] = bad_cameras
        result["status"] = "PASS" if all(result["checks"].values()) else "FAIL"
        print("ROCO_STAGE_A_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        result["status"] = "ERROR"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        print("ROCO_STAGE_A_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
