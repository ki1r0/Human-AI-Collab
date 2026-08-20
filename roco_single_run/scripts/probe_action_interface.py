#!/usr/bin/env python3
"""Run a reversible, held-target probe through the learned action interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Galaxea-Lab-Agent-Direct-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--settle-steps", type=int, default=5)
parser.add_argument("--phase-steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import Galaxea_Lab_External.tasks  # noqa: E402,F401
from Galaxea_Lab_External.robots import ACTIVE_ROBOT_BUNDLE  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from roco_policy import policy_to_environment_order  # noqa: E402


OFFSET = torch.tensor(
    [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, -0.005,
      0.03, 0.02, 0.01, -0.01, -0.02, -0.03, -0.004],
    dtype=torch.float32,
)


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


def flag(value: object) -> bool:
    return bool(torch.as_tensor(value).any().item())


def scalar(value: object) -> float:
    return float(torch.as_tensor(value).reshape(-1)[0].item())


def main() -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "stage": "E-equivalent",
        "classification": "confirmatory_follow_up",
        "task": args.task,
        "seed": args.seed,
        "active_robot_bundle": ACTIVE_ROBOT_BUNDLE.name,
        "offset_policy_order": OFFSET.tolist(),
        "settle_steps": args.settle_steps,
        "phase_steps": args.phase_steps,
        "status": "RUNNING",
    }
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        observations, _ = env.reset(seed=args.seed)
        base_env = env.unwrapped
        policy_observations = observations["policy"]

        reset_target = policy_qpos(policy_observations).detach().clone()
        for _ in range(args.settle_steps):
            observations, _, terminated, truncated, _ = env.step(
                policy_to_environment_order(reset_target)
            )
            if flag(terminated) or flag(truncated):
                raise RuntimeError("Done during baseline settle")
            policy_observations = observations["policy"]

        baseline = policy_qpos(policy_observations).detach().clone()
        offset = OFFSET.to(device=baseline.device).unsqueeze(0)
        phases = (
            ("positive", baseline + offset),
            ("return_1", baseline),
            ("negative", baseline - offset),
            ("return_2", baseline),
        )

        live_samples: list[np.ndarray] = []
        command_samples: list[np.ndarray] = []
        score_samples: list[float] = []
        phase_results: list[dict[str, object]] = []
        any_done = False

        for phase_name, target in phases:
            phase_start = policy_qpos(policy_observations).detach().clone()
            command = policy_to_environment_order(target)
            for _ in range(args.phase_steps):
                observations, reward, terminated, truncated, _ = env.step(command)
                policy_observations = observations["policy"]
                live = policy_qpos(policy_observations).detach()
                live_samples.append(live.cpu().numpy()[0])
                command_samples.append(command.detach().cpu().numpy()[0])
                score_samples.append(scalar(reward))
                any_done = any_done or flag(terminated) or flag(truncated)
                if flag(terminated) or flag(truncated):
                    break

            phase_end = policy_qpos(policy_observations).detach()
            requested = (target - phase_start)[0]
            observed = (phase_end - phase_start)[0]
            demanded = torch.abs(requested) > 0.003
            direction_fraction = float(
                torch.mean(((requested[demanded] * observed[demanded]) > 0).float()).item()
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    requested.unsqueeze(0), observed.unsqueeze(0), dim=-1
                ).item()
            )
            final_mae = float(torch.mean(torch.abs(phase_end - target)).item())
            phase_results.append(
                {
                    "name": phase_name,
                    "requested_policy_displacement": requested.cpu().tolist(),
                    "observed_policy_displacement": observed.cpu().tolist(),
                    "demanded_channels": int(demanded.sum().item()),
                    "direction_agreement_fraction": direction_fraction,
                    "displacement_cosine_similarity": cosine,
                    "final_target_mean_abs_error": final_mae,
                }
            )
            if any_done:
                break

        live_array = np.asarray(live_samples, dtype=np.float32)
        command_array = np.asarray(command_samples, dtype=np.float32)
        score_array = np.asarray(score_samples, dtype=np.float32)
        trace_path = output_dir / "trace.npz"
        np.savez_compressed(
            trace_path,
            baseline_policy_qpos=baseline.cpu().numpy(),
            live_policy_qpos=live_array,
            command_environment_order=command_array,
            score=score_array,
        )

        sentinel = torch.arange(14, device=baseline.device, dtype=torch.float32).unsqueeze(0)
        sentinel_result = policy_to_environment_order(sentinel).cpu().tolist()[0]
        expected_sentinel = list(range(6)) + list(range(7, 13)) + [6, 13]
        phase_count_ok = len(phase_results) == 4
        phase_errors_ok = phase_count_ok and all(
            float(phase["final_target_mean_abs_error"]) <= 0.02
            for phase in phase_results
        )
        phase_cosines_ok = phase_count_ok and all(
            float(phase["displacement_cosine_similarity"]) >= 0.90
            for phase in phase_results
        )
        phase_directions_ok = phase_count_ok and all(
            float(phase["direction_agreement_fraction"]) >= 0.90
            for phase in phase_results
        )
        final_return_mae = float(
            torch.mean(torch.abs(policy_qpos(policy_observations) - baseline)).item()
        )
        all_finite = bool(
            np.isfinite(live_array).all()
            and np.isfinite(command_array).all()
            and all(
                np.isfinite(float(phase[key]))
                for phase in phase_results
                for key in (
                    "direction_agreement_fraction",
                    "displacement_cosine_similarity",
                    "final_target_mean_abs_error",
                )
            )
        )
        checks = {
            "r1_bundle": ACTIVE_ROBOT_BUNDLE.name == "r1",
            "physics_dt_0_01": abs(float(base_env.physics_dt) - 0.01) < 1e-9,
            "control_dt_0_05": abs(float(base_env.step_dt) - 0.05) < 1e-9,
            "sentinel_reorder_exact": sentinel_result == expected_sentinel,
            "all_four_phases_completed": phase_count_ok and len(live_array) == 40,
            "all_finite": all_finite,
            "no_done": not any_done,
            "every_phase_target_mae_le_0_02": phase_errors_ok,
            "every_phase_cosine_ge_0_90": phase_cosines_ok,
            "every_phase_direction_ge_0_90": phase_directions_ok,
            "final_return_mae_le_0_02": final_return_mae <= 0.02,
            "trace_written": trace_path.is_file(),
        }
        result.update(
            {
                "physics_dt_s": float(base_env.physics_dt),
                "control_dt_s": float(base_env.step_dt),
                "baseline_policy_qpos": baseline.cpu().tolist(),
                "sentinel_environment_order": sentinel_result,
                "phase_results": phase_results,
                "executed_steps": int(len(live_array)),
                "best_score": float(score_array.max()) if len(score_array) else None,
                "final_score": float(score_array[-1]) if len(score_array) else None,
                "final_return_mean_abs_error": final_return_mae,
                "checks": checks,
            }
        )
        result["status"] = "PASS" if all(checks.values()) else "FAIL"
        print("ROCO_STAGE_E_EQUIVALENT_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        result["status"] = "ERROR"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        print("ROCO_STAGE_E_EQUIVALENT_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
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
