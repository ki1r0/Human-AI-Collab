#!/usr/bin/env python3
"""Run the official RoCo ACT policy on the M1 Hub-Cover scene.

This runner intentionally keeps the learned policy untouched.  It reuses the
strict checkpoint/stats loader from ``roco_single_run.roco_policy`` and only
adapts the official 14-D action order to the M1 environment's joint-target
interface.  A short smoke episode is the default because the M1 scene is a
new visual/geometry domain for the public RoCo checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--stats", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--max-steps", type=int, default=20)
parser.add_argument("--temporal-decay", type=float, default=0.1)
parser.add_argument("--expected-checkpoint-sha256", default="a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1")
parser.add_argument("--expected-stats-sha256", default="4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e")
parser.add_argument("--camera-update-stride", type=int, default=20)
parser.add_argument("--render-interval", type=int, default=100)
parser.add_argument("--no-cameras", action="store_true", help="Only for loader/contract debugging; ACT requires RGB cameras and will fail without them.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from hrc_m1.roco_env import make_env_classes  # noqa: E402
from roco_single_run.roco_policy import RocoActPolicy  # noqa: E402


def _policy_qpos(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        (
            obs["left_arm_joint_pos"],
            obs["left_gripper_joint_pos"].reshape(obs["left_gripper_joint_pos"].shape[0], -1),
            obs["right_arm_joint_pos"],
            obs["right_gripper_joint_pos"].reshape(obs["right_gripper_joint_pos"].shape[0], -1),
        ),
        dim=-1,
    )


def _policy_images(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack((obs["head_rgb"], obs["left_hand_rgb"], obs["right_hand_rgb"]), dim=1)


def _snapshot(env: object, label: str) -> dict[str, object]:
    state = env.root_states()["hub"][0].detach().cpu().tolist()
    body_id = env.left_arm_cfg.body_ids[0]
    link = env.robot.data.body_state_w[0, body_id, :7].detach().cpu().tolist()
    return {
        "label": label,
        "hub_root_state": state,
        "left_link6_state": link,
        "hub_speed_mps": float(torch.linalg.vector_norm(env.root_states()["hub"][0, 7:10]).item()),
    }


def main() -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_cls, env_cls = make_env_classes()
    cfg = cfg_cls()
    cfg.seed = int(args.seed)
    cfg.action_control = True
    cfg.spawn_cameras = not args.no_cameras
    cfg.update_cameras = not args.no_cameras
    cfg.camera_update_stride = max(1, int(args.camera_update_stride))
    cfg.sim.render_interval = max(1, int(args.render_interval))
    env = env_cls(cfg)
    result: dict[str, object] = {
        "status": "RUNNING",
        "policy": "official_roco_act_adapter",
        "checkpoint": str(args.checkpoint.resolve()),
        "stats": str(args.stats.resolve()),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "action_order": "environment=[L arm6,R arm6,L grip,R grip]; policy=[L arm6,L grip,R arm6,R grip]",
        "cameras": not args.no_cameras,
        "records": [],
    }
    try:
        reset_output = env.reset(seed=args.seed)
        # Isaac Lab's DirectRLEnv returns (observations, info), while older
        # RoCo wrappers expose observations directly. Accept both without
        # changing the policy contract.
        if isinstance(reset_output, tuple):
            reset_output = reset_output[0]
        obs = reset_output["policy"]
        if args.no_cameras:
            raise RuntimeError("ACT requires three RGB camera observations; rerun without --no-cameras")
        policy = RocoActPolicy(
            args.checkpoint,
            args.stats,
            device=str(env.device),
            temporal_decay=args.temporal_decay,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_stats_sha256=args.expected_stats_sha256,
        )
        result["checkpoint_sha256"] = policy.checkpoint_sha256
        result["stats_sha256"] = policy.stats_sha256
        result["parameter_count"] = policy.parameter_count
        result["records"].append(_snapshot(env, "reset"))
        action_records: list[list[float]] = []
        inference_times: list[float] = []
        for step_index in range(int(args.max_steps)):
            qpos = _policy_qpos(obs)
            images = _policy_images(obs)
            if qpos.shape != (1, 14):
                raise RuntimeError(f"Unexpected qpos shape {tuple(qpos.shape)}")
            if images.ndim != 5 or images.shape[1] != 3 or images.shape[-1] != 3:
                raise RuntimeError(f"Unexpected camera shape {tuple(images.shape)}")
            start = time.perf_counter()
            trace = policy.predict_trace(qpos, images)
            inference_times.append(time.perf_counter() - start)
            action = trace["environment_action"]
            if action.shape != (1, 14) or not torch.isfinite(action).all():
                raise RuntimeError(f"Invalid ACT action shape/value: {tuple(action.shape)}")
            arm_values = action[0, :12]
            grip_values = action[0, 12:14]
            if torch.any(torch.abs(arm_values) > 3.3) or torch.any((grip_values < -0.01) | (grip_values > 0.06)):
                raise RuntimeError("Grossly unsafe ACT action; no clipping or repair was applied")
            action_records.append(action[0].detach().cpu().tolist())
            obs, _reward, terminated, truncated, _info = env.step(action)
            obs = obs["policy"]
            result["records"].append(_snapshot(env, f"step_{step_index + 1}"))
            if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
                result["termination"] = "terminated_or_truncated"
                break
        result["actions"] = action_records
        result["inference_time_s"] = inference_times
        result["executed_steps"] = len(action_records)
        result["status"] = "COMPLETED_SMOKE"
    except Exception as exc:
        result["status"] = "FAILED"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        # Camera-backed Isaac Lab close can wait on an RTX worker after the
        # artifact is already written. Stop the simulation app explicitly and
        # keep cleanup best-effort so a smoke launcher returns deterministically.
        try:
            env.sim.stop()
        except Exception:
            pass
        try:
            app.close()
        except Exception:
            pass
    print(json.dumps({"status": result["status"], "output": str(args.output_dir / "result.json")}))
    return 0 if result["status"] == "COMPLETED_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
