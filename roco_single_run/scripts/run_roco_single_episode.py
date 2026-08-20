#!/usr/bin/env python3
"""Run and record one genuine live closed-loop RoCo ACT episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Galaxea-Lab-Agent-Direct-v0")
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--stats", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--max-steps", type=int, default=590)
parser.add_argument("--video-fps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import Galaxea_Lab_External.tasks  # noqa: E402,F401
from Galaxea_Lab_External.robots import ACTIVE_ROBOT_BUNDLE  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from roco_policy import CAMERA_NAMES, RocoActPolicy, file_sha256  # noqa: E402


CONTAINER_IMAGE = "nvcr.io/nvidia/isaac-lab:2.3.0"
SOURCE_COMMIT = "094a1f76d18c207caec198315f23b1a60dbca94f"


def policy_qpos(observations: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        (
            observations["left_arm_joint_pos"],
            observations["left_gripper_joint_pos"].unsqueeze(-1),
            observations["right_arm_joint_pos"],
            observations["right_gripper_joint_pos"].unsqueeze(-1),
        ),
        dim=-1,
    )


def policy_images(observations: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(tuple(observations[key] for key in CAMERA_NAMES), dim=1)


def flag(value: object) -> bool:
    return bool(torch.as_tensor(value).any().item())


def scalar(value: object) -> float:
    return float(torch.as_tensor(value).reshape(-1)[0].item())


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def compose_frame(
    observations: dict[str, torch.Tensor], *, step: int, score: int, label: str
) -> np.ndarray:
    arrays = [
        observations[key][0].detach().cpu().numpy().astype(np.uint8, copy=False)
        for key in CAMERA_NAMES
    ]
    canvas = np.zeros((270, 960, 3), dtype=np.uint8)
    canvas[30:, :, :] = np.concatenate(arrays, axis=1)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((6, 7), f"{label}  step={step}  score={score}", fill=(255, 255, 255))
    draw.text((325, 7), "head | left wrist | right wrist", fill=(200, 220, 255))
    return np.asarray(image)


def object_pose_vector(base_env: object) -> tuple[list[str], np.ndarray]:
    names = sorted(base_env.obj_dict)
    poses = np.concatenate(
        [
            base_env.obj_dict[name].data.root_state_w[0, :7]
            .detach()
            .cpu()
            .numpy()
            for name in names
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return names, poses


def main() -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "episode.mp4"
    trace_path = output_dir / "trace.npz"
    result_path = output_dir / "result.json"
    result: dict[str, object] = {
        "stage": "F",
        "classification": "confirmatory" if args.seed == 23 else "preregistered_sensitivity",
        "task": args.task,
        "seed": args.seed,
        "seed_sequence": [23, 17, 42, 2026],
        "container_image": CONTAINER_IMAGE,
        "official_source_commit": SOURCE_COMMIT,
        "active_robot_bundle": ACTIVE_ROBOT_BUNDLE.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "stats": str(args.stats.resolve()),
        "max_steps": args.max_steps,
        "video_fps": args.video_fps,
        "policy_inputs": list(CAMERA_NAMES) + ["qpos14"],
        "privileged_inputs_to_policy": False,
        "status": "RUNNING",
    }
    env = None
    writer = None
    policy = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        observations, _ = env.reset(seed=args.seed)
        base_env = env.unwrapped
        policy_observations = observations["policy"]

        policy = RocoActPolicy(
            args.checkpoint,
            args.stats,
            device=args.device,
            temporal_decay=0.1,
        )
        policy.reset()
        writer = imageio.get_writer(
            video_path,
            fps=args.video_fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
            ffmpeg_log_level="error",
        )

        qpos_before_samples: list[np.ndarray] = []
        qpos_after_samples: list[np.ndarray] = []
        policy_action_samples: list[np.ndarray] = []
        environment_action_samples: list[np.ndarray] = []
        score_samples: list[int] = []
        reward_samples: list[float] = []
        terminated_samples: list[bool] = []
        truncated_samples: list[bool] = []
        inference_time_samples: list[float] = []
        aggregate_count_samples: list[int] = []
        aggregate_oldest_weight_samples: list[float] = []
        aggregate_newest_weight_samples: list[float] = []
        normalized_qpos_min_samples: list[float] = []
        normalized_qpos_max_samples: list[float] = []
        raw_chunk_min_samples: list[float] = []
        raw_chunk_max_samples: list[float] = []
        camera_hash_samples: list[list[str]] = []
        object_pose_samples: list[np.ndarray] = []
        object_names: list[str] = []
        score_transitions: list[dict[str, int]] = []

        initial_score, _ = base_env.evaluate_score()
        current_score = int(initial_score)
        initial_object_names, initial_object_poses = object_pose_vector(base_env)
        object_names = initial_object_names
        initial_qpos = policy_qpos(policy_observations).detach().cpu().numpy()
        initial_camera_hashes = [
            array_sha256(policy_observations[key].detach().cpu().numpy())
            for key in CAMERA_NAMES
        ]
        writer.append_data(
            compose_frame(policy_observations, step=0, score=current_score, label="RESET")
        )
        video_frame_count = 1
        success = False
        termination_reason = "max_steps"

        for step in range(args.max_steps):
            qpos = policy_qpos(policy_observations)
            images = policy_images(policy_observations)
            if qpos.shape != (1, 14) or images.shape != (1, 3, 240, 320, 3):
                raise RuntimeError(
                    f"Live contract changed: qpos={tuple(qpos.shape)}, images={tuple(images.shape)}"
                )
            if qpos.dtype != torch.float32 or images.dtype != torch.uint8:
                raise RuntimeError(
                    f"Live dtype changed: qpos={qpos.dtype}, images={images.dtype}"
                )
            if not torch.isfinite(qpos).all():
                raise RuntimeError("Non-finite live qpos")

            camera_hashes = [
                array_sha256(policy_observations[key].detach().cpu().numpy())
                for key in CAMERA_NAMES
            ]
            start = time.perf_counter()
            trace = policy.predict_trace(qpos, images)
            inference_time = time.perf_counter() - start
            policy_action = trace["denormalized_policy_action"]
            environment_action = trace["environment_action"]
            if not torch.isfinite(policy_action).all() or not torch.isfinite(
                environment_action
            ).all():
                raise RuntimeError("Non-finite learned action")
            arm_values = policy_action[0, [*range(6), *range(7, 13)]]
            gripper_values = policy_action[0, [6, 13]]
            if torch.any(torch.abs(arm_values) > 3.3) or torch.any(
                (gripper_values < -0.01) | (gripper_values > 0.06)
            ):
                raise RuntimeError(
                    "Grossly unsafe raw action (not clipped): "
                    + repr(policy_action.detach().cpu().tolist())
                )

            qpos_before_samples.append(qpos.detach().cpu().numpy()[0])
            policy_action_samples.append(policy_action.detach().cpu().numpy()[0])
            environment_action_samples.append(
                environment_action.detach().cpu().numpy()[0]
            )
            camera_hash_samples.append(camera_hashes)
            inference_time_samples.append(inference_time)
            weights = trace["aggregation_weights"]
            aggregate_count_samples.append(int(weights.numel()))
            aggregate_oldest_weight_samples.append(float(weights[0].item()))
            aggregate_newest_weight_samples.append(float(weights[-1].item()))
            normalized_qpos_min_samples.append(float(trace["normalized_qpos"].min().item()))
            normalized_qpos_max_samples.append(float(trace["normalized_qpos"].max().item()))
            raw_chunk_min_samples.append(float(trace["raw_chunk"].min().item()))
            raw_chunk_max_samples.append(float(trace["raw_chunk"].max().item()))

            observations, reward, terminated, truncated, _ = env.step(environment_action)
            policy_observations = observations["policy"]
            qpos_after = policy_qpos(policy_observations)
            explicit_score, _ = base_env.evaluate_score()
            new_score = int(explicit_score)
            reward_value = scalar(reward)
            is_terminated = flag(terminated)
            is_truncated = flag(truncated)
            _, object_poses = object_pose_vector(base_env)

            qpos_after_samples.append(qpos_after.detach().cpu().numpy()[0])
            score_samples.append(new_score)
            reward_samples.append(reward_value)
            terminated_samples.append(is_terminated)
            truncated_samples.append(is_truncated)
            object_pose_samples.append(object_poses)
            if new_score != current_score:
                score_transitions.append(
                    {"step": step + 1, "from": current_score, "to": new_score}
                )
                print(
                    f"ROCO_SCORE_TRANSITION step={step + 1} {current_score}->{new_score}",
                    flush=True,
                )
            current_score = new_score
            writer.append_data(
                compose_frame(
                    policy_observations,
                    step=step + 1,
                    score=current_score,
                    label="TERMINAL" if current_score >= 6 else "POST ACTION",
                )
            )
            video_frame_count += 1
            if step % 25 == 0:
                print(
                    f"ROCO_LEARNED_PROGRESS step={step + 1}/{args.max_steps} "
                    f"score={current_score} inference_s={inference_time:.4f}",
                    flush=True,
                )

            if current_score >= 6:
                success = True
                termination_reason = "task_success"
                break
            if is_terminated:
                termination_reason = "native_terminated_before_success"
                break
            if is_truncated:
                termination_reason = "native_truncated_before_success"
                break

        executed_steps = len(score_samples)
        writer.close()
        writer = None

        final_qpos = policy_qpos(policy_observations).detach().cpu().numpy()
        final_camera_hashes = [
            array_sha256(policy_observations[key].detach().cpu().numpy())
            for key in CAMERA_NAMES
        ]
        _, final_object_poses = object_pose_vector(base_env)
        np.savez_compressed(
            trace_path,
            initial_qpos_policy_order=initial_qpos,
            final_qpos_policy_order=final_qpos,
            qpos_before_policy_order=np.asarray(qpos_before_samples, dtype=np.float32),
            qpos_after_policy_order=np.asarray(qpos_after_samples, dtype=np.float32),
            policy_actions=np.asarray(policy_action_samples, dtype=np.float32),
            environment_actions=np.asarray(environment_action_samples, dtype=np.float32),
            score=np.asarray(score_samples, dtype=np.int16),
            reward=np.asarray(reward_samples, dtype=np.float32),
            terminated=np.asarray(terminated_samples, dtype=np.bool_),
            truncated=np.asarray(truncated_samples, dtype=np.bool_),
            inference_time_s=np.asarray(inference_time_samples, dtype=np.float32),
            aggregation_count=np.asarray(aggregate_count_samples, dtype=np.int16),
            aggregation_oldest_weight=np.asarray(
                aggregate_oldest_weight_samples, dtype=np.float32
            ),
            aggregation_newest_weight=np.asarray(
                aggregate_newest_weight_samples, dtype=np.float32
            ),
            normalized_qpos_min=np.asarray(normalized_qpos_min_samples, dtype=np.float32),
            normalized_qpos_max=np.asarray(normalized_qpos_max_samples, dtype=np.float32),
            raw_chunk_min=np.asarray(raw_chunk_min_samples, dtype=np.float32),
            raw_chunk_max=np.asarray(raw_chunk_max_samples, dtype=np.float32),
            camera_sha256=np.asarray(camera_hash_samples, dtype="S64"),
            initial_camera_sha256=np.asarray(initial_camera_hashes, dtype="S64"),
            final_camera_sha256=np.asarray(final_camera_hashes, dtype="S64"),
            object_names=np.asarray(object_names, dtype="S64"),
            initial_object_poses=initial_object_poses,
            object_poses_after_step=np.asarray(object_pose_samples, dtype=np.float32),
            final_object_poses=final_object_poses,
        )

        all_arrays_finite = bool(
            np.isfinite(np.asarray(qpos_before_samples)).all()
            and np.isfinite(np.asarray(qpos_after_samples)).all()
            and np.isfinite(np.asarray(policy_action_samples)).all()
            and np.isfinite(np.asarray(environment_action_samples)).all()
            and np.isfinite(np.asarray(object_pose_samples)).all()
        )
        video_sha256 = file_sha256(video_path)
        trace_sha256 = file_sha256(trace_path)
        video_size = video_path.stat().st_size
        trace_size = trace_path.stat().st_size
        expected_video_frames = executed_steps + 1
        # One reset frame followed by every post-action observation, including
        # the terminal/final state. Intermediate frames are the next policy input.
        artifacts_ok = (
            video_size > 0
            and trace_size > 0
            and video_frame_count == expected_video_frames
        )
        checks = {
            "r1_bundle": ACTIVE_ROBOT_BUNDLE.name == "r1",
            "checkpoint_hash_matches": policy.checkpoint_sha256
            == "a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1",
            "stats_hash_matches": policy.stats_sha256
            == "4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e",
            "strict_model_load": not policy.missing_keys and not policy.unexpected_keys,
            "physics_dt_0_01": abs(float(base_env.physics_dt) - 0.01) < 1e-9,
            "control_dt_0_05": abs(float(base_env.step_dt) - 0.05) < 1e-9,
            "one_inference_per_step": policy.timestep == executed_steps,
            "all_arrays_finite": all_arrays_finite,
            "task_success_score_ge_6": success and max(score_samples, default=0) >= 6,
            "success_termination": termination_reason == "task_success",
            "video_trace_complete": artifacts_ok,
        }
        result.update(
            {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "termination_reason": termination_reason,
                "task_success": success,
                "executed_steps": executed_steps,
                "physics_dt_s": float(base_env.physics_dt),
                "control_dt_s": float(base_env.step_dt),
                "initial_score": int(initial_score),
                "best_score": max(score_samples, default=int(initial_score)),
                "final_score": current_score,
                "score_transitions": score_transitions,
                "policy_timestep": policy.timestep,
                "checkpoint_sha256": policy.checkpoint_sha256,
                "stats_sha256": policy.stats_sha256,
                "checkpoint_tensor_count": policy.checkpoint_tensor_count,
                "parameter_count": policy.parameter_count,
                "missing_keys": policy.missing_keys,
                "unexpected_keys": policy.unexpected_keys,
                "camera_order": list(CAMERA_NAMES),
                "qpos_order": "Larm6,Lgrip,Rarm6,Rgrip",
                "action_environment_order": "Larm6,Rarm6,Lgrip,Rgrip",
                "temporal_decay": policy.temporal_decay,
                "inference_time_s": {
                    "mean": float(np.mean(inference_time_samples)),
                    "median": float(np.median(inference_time_samples)),
                    "max": float(np.max(inference_time_samples)),
                },
                "normalized_qpos_range": [
                    min(normalized_qpos_min_samples),
                    max(normalized_qpos_max_samples),
                ],
                "raw_chunk_range": [min(raw_chunk_min_samples), max(raw_chunk_max_samples)],
                "video": str(video_path),
                "video_codec": "H.264/yuv420p",
                "video_frame_size": [960, 270],
                "video_frame_count": video_frame_count,
                "expected_video_frame_count": expected_video_frames,
                "video_size": video_size,
                "video_sha256": video_sha256,
                "trace": str(trace_path),
                "trace_size": trace_size,
                "trace_sha256": trace_sha256,
                "object_state_usage": "post_action_audit_only_not_policy_input",
                "checks": checks,
            }
        )
        print("ROCO_LEARNED_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        result["status"] = "ERROR"
        result["termination_reason"] = "exception"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        print("ROCO_LEARNED_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        if writer is not None:
            writer.close()
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
