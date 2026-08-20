#!/usr/bin/env python3
"""Run one bounded official RoCo rule-policy episode as a physics oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Template-Galaxea-Lab-External-Direct-v0",
    help="Official scripted-policy environment ID.",
)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--max-steps", type=int, default=700)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--classification",
    choices=("confirmatory", "exploratory"),
    default="confirmatory",
    help="Experiment classification recorded in the result.",
)
parser.add_argument(
    "--save-phase-images",
    action="store_true",
    help="Save RGB observations at ring pick/mount phase boundaries.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import Galaxea_Lab_External.tasks  # noqa: E402,F401
from Galaxea_Lab_External.robots import ACTIVE_ROBOT_BUNDLE  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _as_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().any().item())
    return bool(value)


def _object_poses(base_env) -> dict:
    return {
        name: {
            "position_xyz_m": obj.data.root_state_w[0, :3].detach().cpu().tolist(),
            "quaternion_wxyz": obj.data.root_state_w[0, 3:7].detach().cpu().tolist(),
        }
        for name, obj in sorted(base_env.obj_dict.items())
    }


def _save_rgb_observations(observations: dict, output_dir: Path, physics_step: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for key in ("head_rgb", "left_hand_rgb", "right_hand_rgb"):
        target = output_dir / f"physics_{physics_step:04d}_{key}.png"
        Image.fromarray(observations["policy"][key][0].detach().cpu().numpy()).save(target)
        saved[key] = str(target)
    return saved


def main() -> None:
    result = {
        "stage": "B",
        "classification": args.classification,
        "task": args.task,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "active_robot_bundle": ACTIVE_ROBOT_BUNDLE.name,
        "policy": "official GalaxeaRulePolicy",
        "success_threshold": 6,
        "status": "RUNNING",
    }
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
        env_cfg.seed = args.seed
        env_cfg.record_data = False
        env = gym.make(args.task, cfg=env_cfg, use_action=True)
        base_env = env.unwrapped
        observations, _ = env.reset(seed=args.seed)
        initial_score, initial_time_s = base_env.evaluate_score()
        total_policy_steps = int(base_env.rule_policy.total_time_steps.item())
        zero_action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=base_env.device)

        result.update(
            {
                "sim_dt_s": float(base_env.physics_dt),
                "control_dt_s": float(base_env.step_dt),
                "decimation": int(base_env.cfg.decimation),
                "rule_policy_total_physics_steps": total_policy_steps,
                "initial_score": int(initial_score),
                "initial_score_time_s": float(initial_time_s),
                "initial_object_poses": _object_poses(base_env),
                "score_transitions": [],
                "phase_snapshots": [],
            }
        )

        snapshot_physics_steps = {}
        for phase_name in ("count_step_11", "count_step_12", "count_step_13", "count_step_14"):
            for boundary_index, value in enumerate(getattr(base_env.rule_policy, phase_name)):
                snapshot_physics_steps.setdefault(int(value.item()), []).append(
                    f"{phase_name}[{boundary_index}]"
                )
        snapshot_physics_steps.setdefault(total_policy_steps - 5, []).append("pre_schedule_end")

        best_score = int(initial_score)
        last_transition_score = int(initial_score)
        final_terminated = False
        final_truncated = False
        last_reward_score = int(initial_score)
        steps_completed = 0
        for step_index in range(args.max_steps):
            with torch.inference_mode():
                observations, reward, terminated, truncated, _ = env.step(zero_action)
            steps_completed = step_index + 1
            score = int(torch.as_tensor(reward).detach().cpu().flatten()[0].item())
            last_reward_score = score
            best_score = max(best_score, score)
            final_terminated = _as_bool(terminated)
            final_truncated = _as_bool(truncated)
            policy_count = int(base_env.rule_policy.count)

            if score != last_transition_score:
                result["score_transitions"].append(
                    {
                        "env_step": steps_completed,
                        "physics_step": policy_count,
                        "score": score,
                        "previous_score": last_transition_score,
                        "object_poses": _object_poses(base_env),
                    }
                )
                last_transition_score = score
            if policy_count in snapshot_physics_steps:
                snapshot = {
                    "env_step": steps_completed,
                    "physics_step": policy_count,
                    "phase_boundaries": snapshot_physics_steps[policy_count],
                    "score": score,
                    "ring_arm": base_env.rule_policy.gear_to_pin_map["ring_gear"]["arm"],
                    "ring_pose": _object_poses(base_env)["ring_gear"],
                    "left_gripper_position_m": float(
                        base_env.robot.data.joint_pos[0, base_env._left_gripper_dof_idx[0]].item()
                    ),
                    "right_gripper_position_m": float(
                        base_env.robot.data.joint_pos[0, base_env._right_gripper_dof_idx[0]].item()
                    ),
                    "object_poses": _object_poses(base_env),
                }
                if args.save_phase_images:
                    snapshot["rgb_files"] = _save_rgb_observations(
                        observations, args.output.parent / "phase_images", policy_count
                    )
                result["phase_snapshots"].append(snapshot)
            if steps_completed % 25 == 0 or score >= 6:
                print(
                    f"ROCO_STAGE_B_PROGRESS env_step={steps_completed} "
                    f"physics_step={policy_count}/{total_policy_steps} score={score}",
                    flush=True,
                )
            if final_terminated or final_truncated:
                break

        final_score, final_score_time_s = base_env.evaluate_score()
        result.update(
            {
                "steps_completed": steps_completed,
                "best_score": best_score,
                "last_reward_score_before_auto_reset": last_reward_score,
                "final_score": int(final_score),
                "final_score_time_s": float(final_score_time_s),
                "terminated": final_terminated,
                "truncated": final_truncated,
                "final_object_poses": _object_poses(base_env),
                "final_observation_keys": sorted(observations["policy"].keys()),
            }
        )
        result["checks"] = {
            "r1_bundle": ACTIVE_ROBOT_BUNDLE.name == "r1",
            "official_score_reached_6": best_score >= 6,
            "stable_score_6_at_schedule_end": last_reward_score >= 6,
            "completed_before_bound": steps_completed <= args.max_steps,
            "not_time_truncated": not final_truncated,
        }
        result["status"] = "PASS" if all(result["checks"].values()) else "FAIL"
        print("ROCO_STAGE_B_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        result["status"] = "ERROR"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        print("ROCO_STAGE_B_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
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
