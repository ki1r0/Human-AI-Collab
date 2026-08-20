#!/usr/bin/env python3
"""Save one synchronized live RoCo R1 policy observation package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Galaxea-Lab-Agent-Direct-v0")
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import Galaxea_Lab_External.tasks  # noqa: E402,F401
from Galaxea_Lab_External.robots import ACTIVE_ROBOT_BUNDLE  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


CAMERA_KEYS = ("head_rgb", "left_hand_rgb", "right_hand_rgb")


def _summary(value: torch.Tensor) -> dict:
    numeric = value.float()
    finite = torch.isfinite(numeric)
    finite_values = numeric[finite]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite_fraction": float(finite.float().mean().item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "mean": float(finite_values.mean().item()),
        "std": float(finite_values.std().item()),
    }


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def main() -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "stage": "C",
        "classification": "confirmatory",
        "task": args.task,
        "seed": args.seed,
        "active_robot_bundle": ACTIVE_ROBOT_BUNDLE.name,
        "camera_order": list(CAMERA_KEYS),
        "status": "RUNNING",
    }
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        base_env = env.unwrapped
        reset_observations, _ = env.reset(seed=args.seed)
        reset_policy = reset_observations["policy"]
        hold_action = base_env.robot.data.joint_pos[:, base_env._joint_idx].detach().clone()

        observations, _, terminated, truncated, _ = env.step(hold_action)
        policy = observations["policy"]

        qpos_parts = (
            policy["left_arm_joint_pos"],
            policy["left_gripper_joint_pos"].unsqueeze(-1),
            policy["right_arm_joint_pos"],
            policy["right_gripper_joint_pos"].unsqueeze(-1),
        )
        qpos = torch.cat(qpos_parts, dim=-1)
        qpos_names = (
            [base_env.robot.joint_names[index] for index in base_env._left_arm_joint_idx]
            + [base_env.robot.joint_names[base_env._left_gripper_dof_idx[0]]]
            + [base_env.robot.joint_names[index] for index in base_env._right_arm_joint_idx]
            + [base_env.robot.joint_names[base_env._right_gripper_dof_idx[0]]]
        )

        arrays = {key: policy[key].detach().cpu().numpy() for key in CAMERA_KEYS}
        qpos_array = qpos.detach().cpu().numpy()
        timestamp_s = float(base_env.episode_length_buf[0].item() * base_env.step_dt)
        package_path = output_dir / "observation.npz"
        np.savez_compressed(
            package_path,
            **arrays,
            qpos=qpos_array,
            timestamp_s=np.asarray(timestamp_s, dtype=np.float64),
        )

        png_paths = {}
        for key, array in arrays.items():
            path = output_dir / f"{key}.png"
            Image.fromarray(array[0]).save(path)
            png_paths[key] = str(path)

        reset_differences = {
            key: float(
                torch.mean(
                    torch.abs(policy[key].float() - reset_policy[key].float())
                ).item()
            )
            for key in CAMERA_KEYS
        }
        result.update(
            {
                "sim_dt_s": float(base_env.physics_dt),
                "control_dt_s": float(base_env.step_dt),
                "timestamp_s": timestamp_s,
                "terminated": bool(torch.as_tensor(terminated).any().item()),
                "truncated": bool(torch.as_tensor(truncated).any().item()),
                "policy_observation_keys": sorted(policy),
                "camera_summaries": {key: _summary(policy[key]) for key in CAMERA_KEYS},
                "reset_to_step_mean_abs_pixel_difference": reset_differences,
                "qpos_summary": _summary(qpos),
                "qpos_order": qpos_names,
                "qpos_values": qpos_array.tolist(),
                "array_sha256": {
                    **{key: _sha256_array(value) for key, value in arrays.items()},
                    "qpos": _sha256_array(qpos_array),
                },
                "package": str(package_path),
                "png_files": png_paths,
            }
        )

        camera_shape_ok = all(value.shape == (1, 240, 320, 3) for value in arrays.values())
        camera_dtype_ok = all(value.dtype == np.uint8 for value in arrays.values())
        camera_content_ok = all(value.max() > 0 and value.std() > 1 for value in arrays.values())
        checks = {
            "r1_bundle": ACTIVE_ROBOT_BUNDLE.name == "r1",
            "camera_shape_1x240x320x3": camera_shape_ok,
            "camera_dtype_uint8": camera_dtype_ok,
            "camera_nonblack_nonconstant": camera_content_ok,
            "qpos_shape_1x14": qpos_array.shape == (1, 14),
            "qpos_dtype_float32": qpos_array.dtype == np.float32,
            "qpos_all_finite": bool(np.isfinite(qpos_array).all()),
            "qpos_names_14": len(qpos_names) == 14,
            "npz_written": package_path.is_file(),
            "pngs_written": all(Path(path).is_file() for path in png_paths.values()),
        }
        result["checks"] = checks
        result["status"] = "PASS" if all(checks.values()) else "FAIL"
        print("ROCO_STAGE_C_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        result["status"] = "ERROR"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        print("ROCO_STAGE_C_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        (output_dir / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
