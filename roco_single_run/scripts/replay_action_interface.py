#!/usr/bin/env python3
"""Replay one pinned official trajectory through the live RoCo action interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Galaxea-Lab-Agent-Direct-v0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import Galaxea_Lab_External.tasks  # noqa: E402,F401
from Galaxea_Lab_External.VLA.ACT.policy_wrapper import (  # noqa: E402
    DataReplayPolicyWrapper,
)
from Galaxea_Lab_External.robots import ACTIVE_ROBOT_BUNDLE  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


DATA_SHA256 = "dd7ade042484b4fc5a868df6f17df0a90ccdebbd6b1e309d916bbd2e2450d434"
DATA_SIZE = 951_664_104
CAMERA_KEYS = ("head_rgb", "left_hand_rgb", "right_hand_rgb")


class CompatibleDataReplayPolicyWrapper(DataReplayPolicyWrapper):
    """Repair only the upstream base/subclass device-property mismatch."""

    @property
    def device(self) -> torch.device:
        return self._replay_device

    @device.setter
    def device(self, value: str | torch.device) -> None:
        self._replay_device = torch.device(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def policy_to_environment_numpy(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (values[..., :6], values[..., 7:13], values[..., 6:7], values[..., 13:14]),
        axis=-1,
    )


def environment_to_policy_numpy(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (values[..., :6], values[..., 12:13], values[..., 6:12], values[..., 13:14]),
        axis=-1,
    )


def scalar(value: object) -> float:
    return float(torch.as_tensor(value).reshape(-1)[0].item())


def flag(value: object) -> bool:
    return bool(torch.as_tensor(value).any().item())


def main() -> None:
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "stage": "E",
        "classification": "confirmatory",
        "task": args.task,
        "seed": args.seed,
        "seed_provenance": "heuristic_only_dataset_omits_reset_seed",
        "active_robot_bundle": ACTIVE_ROBOT_BUNDLE.name,
        "dataset": str(data_path),
        "dataset_expected_size": DATA_SIZE,
        "dataset_expected_sha256": DATA_SHA256,
        "wrapper_class": "DataReplayPolicyWrapper",
        "status": "RUNNING",
    }
    env = None
    try:
        actual_size = data_path.stat().st_size
        actual_sha256 = file_sha256(data_path)
        result["dataset_actual_size"] = actual_size
        result["dataset_actual_sha256"] = actual_sha256
        if actual_size != DATA_SIZE or actual_sha256 != DATA_SHA256:
            raise RuntimeError("Pinned replay dataset size/hash mismatch")

        with h5py.File(data_path, "r") as source:
            file_policy_actions = np.concatenate(
                (
                    source["actions/left_arm_action"][:],
                    source["actions/left_gripper_action"][:][:, None],
                    source["actions/right_arm_action"][:],
                    source["actions/right_gripper_action"][:][:, None],
                ),
                axis=1,
            ).astype(np.float32, copy=False)
            file_policy_qpos = np.concatenate(
                (
                    source["observations/left_arm_joint_pos"][:],
                    source["observations/left_gripper_joint_pos"][:][:, None],
                    source["observations/right_arm_joint_pos"][:],
                    source["observations/right_gripper_joint_pos"][:][:, None],
                ),
                axis=1,
            ).astype(np.float32, copy=False)
            current_time = source["current_time"][:]

        data_next_qpos_error = np.abs(file_policy_actions[:-1] - file_policy_qpos[1:])
        final_hold_error = np.abs(file_policy_actions[-1] - file_policy_qpos[-1])
        data_contract_max_error = float(
            max(data_next_qpos_error.max(), final_hold_error.max())
        )

        replay = CompatibleDataReplayPolicyWrapper(str(data_path), device=args.device)
        wrapper_actions = replay.actions.detach().cpu().numpy()
        file_environment_actions = policy_to_environment_numpy(file_policy_actions)
        wrapper_file_max_error = float(np.abs(wrapper_actions - file_environment_actions).max())

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        observations, _ = env.reset(seed=args.seed)
        base_env = env.unwrapped
        policy_observations = observations["policy"]
        reset_qpos = policy_qpos(policy_observations).detach().cpu().numpy()
        reset_qpos_mae = float(np.abs(reset_qpos[0] - file_policy_qpos[0]).mean())
        Image.fromarray(policy_observations["head_rgb"][0].detach().cpu().numpy()).save(
            output_dir / "live_head_reset.png"
        )

        live_before: list[np.ndarray] = []
        live_after: list[np.ndarray] = []
        commands_environment: list[np.ndarray] = []
        scores: list[float] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        success = False

        for _ in range(replay.max_steps):
            current_policy_qpos = policy_qpos(policy_observations)
            images = torch.stack(
                tuple(policy_observations[key] for key in CAMERA_KEYS), dim=1
            )
            command = replay.predict(current_policy_qpos, images)
            before = current_policy_qpos.detach().cpu().numpy()[0]
            observations, reward, terminated, truncated, _ = env.step(command)
            policy_observations = observations["policy"]
            after = policy_qpos(policy_observations).detach().cpu().numpy()[0]
            score = scalar(reward)
            is_terminated = flag(terminated)
            is_truncated = flag(truncated)

            live_before.append(before)
            live_after.append(after)
            commands_environment.append(command.detach().cpu().numpy()[0])
            scores.append(score)
            terminated_flags.append(is_terminated)
            truncated_flags.append(is_truncated)

            if score >= 6.0:
                success = True
                break
            if is_terminated or is_truncated:
                break

        before_array = np.asarray(live_before, dtype=np.float32)
        after_array = np.asarray(live_after, dtype=np.float32)
        command_environment_array = np.asarray(commands_environment, dtype=np.float32)
        command_policy_array = environment_to_policy_numpy(command_environment_array)

        # A successful step may return post-reset observations; exclude only that
        # terminal sample from robot tracking metrics.
        tracking_count = len(after_array) - int(success)
        tracking_after = after_array[:tracking_count]
        tracking_before = before_array[:tracking_count]
        tracking_commands = command_policy_array[:tracking_count]
        tracking_abs_error = np.abs(tracking_after - tracking_commands)
        requested_motion = tracking_commands - tracking_before
        observed_motion = tracking_after - tracking_before
        demanded = np.abs(requested_motion) > 0.001
        direction_agreement = float(
            np.mean((requested_motion[demanded] * observed_motion[demanded]) > 0)
        )
        mean_tracking_error = float(tracking_abs_error.mean())
        max_tracking_error = float(tracking_abs_error.max())

        Image.fromarray(policy_observations["head_rgb"][0].detach().cpu().numpy()).save(
            output_dir / "live_head_final.png"
        )
        trace_path = output_dir / "trace.npz"
        np.savez_compressed(
            trace_path,
            file_policy_actions=file_policy_actions,
            file_policy_qpos=file_policy_qpos,
            wrapper_environment_actions=wrapper_actions,
            live_before_policy_qpos=before_array,
            live_after_policy_qpos=after_array,
            executed_environment_actions=command_environment_array,
            score=np.asarray(scores, dtype=np.float32),
            terminated=np.asarray(terminated_flags, dtype=np.bool_),
            truncated=np.asarray(truncated_flags, dtype=np.bool_),
        )

        reached_end = len(after_array) == replay.max_steps
        premature_done = bool(
            (terminated_flags[-1] or truncated_flags[-1]) if terminated_flags else True
        ) and not success
        all_finite = bool(
            np.isfinite(wrapper_actions).all()
            and np.isfinite(before_array).all()
            and np.isfinite(after_array).all()
            and np.isfinite(command_environment_array).all()
        )
        checks = {
            "r1_bundle": ACTIVE_ROBOT_BUNDLE.name == "r1",
            "dataset_size_matches": actual_size == DATA_SIZE,
            "dataset_hash_matches": actual_sha256 == DATA_SHA256,
            "wrapper_shape_590x14": wrapper_actions.shape == (590, 14),
            "wrapper_file_exact": wrapper_file_max_error == 0.0,
            "dataset_action_is_next_qpos_exact": data_contract_max_error == 0.0,
            "current_time_metadata_known_zero": bool(np.all(current_time == 0.0)),
            "reset_qpos_mae_le_0_01": reset_qpos_mae <= 0.01,
            "all_finite": all_finite,
            "mean_tracking_error_le_0_10": mean_tracking_error <= 0.10,
            "direction_agreement_ge_0_80": direction_agreement >= 0.80,
            "reached_end_or_task_success": reached_end or success,
            "no_premature_done": not premature_done,
            "trace_written": trace_path.is_file(),
        }
        result.update(
            {
                "control_dt_s": float(base_env.step_dt),
                "physics_dt_s": float(base_env.physics_dt),
                "dataset_samples": int(len(file_policy_actions)),
                "dataset_current_time_unique": np.unique(current_time).tolist(),
                "dataset_action_next_qpos_max_abs_error": data_contract_max_error,
                "wrapper_file_max_abs_error": wrapper_file_max_error,
                "reset_qpos_policy_order": reset_qpos.tolist(),
                "dataset_reset_qpos_policy_order": file_policy_qpos[0].tolist(),
                "reset_qpos_mae": reset_qpos_mae,
                "executed_steps": int(len(after_array)),
                "tracking_steps": int(tracking_count),
                "tracking_mean_abs_error": mean_tracking_error,
                "tracking_max_abs_error": max_tracking_error,
                "demanded_motion_elements": int(demanded.sum()),
                "direction_agreement_fraction": direction_agreement,
                "best_score": max(scores) if scores else None,
                "final_score": scores[-1] if scores else None,
                "task_success": success,
                "reached_replay_end": reached_end,
                "terminated_final": terminated_flags[-1] if terminated_flags else None,
                "truncated_final": truncated_flags[-1] if truncated_flags else None,
                "checks": checks,
            }
        )
        result["status"] = "PASS" if all(checks.values()) else "FAIL"
        print("ROCO_STAGE_E_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        result["status"] = "ERROR"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        print("ROCO_STAGE_E_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
