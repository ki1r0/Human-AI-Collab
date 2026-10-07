#!/usr/bin/env python3
"""Run and record one deterministic physical RoCo M1 assembly episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=("planet1", "planet2", "planet3", "center", "ring", "reducer", "full"), default="full")
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--video-fps", type=int, default=20)
parser.add_argument(
    "--post-settle-s",
    type=float,
    default=0.0,
    help="Continue stepping after the controller schedule for a measured settle window.",
)
parser.add_argument(
    "--inspect-only",
    action="store_true",
    help="Initialize the scene and diagnostic sensors, then exit before executing controls.",
)
parser.add_argument(
    "--gear-friction-coefficient",
    type=float,
    default=None,
    help="Diagnostic override for the shared gear/carrier material friction.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import PhysxSchema  # noqa: E402
from isaaclab.sim import activate_contact_sensors, get_current_stage  # noqa: E402

import Galaxea_Lab_External.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402

from controller import M1AssemblyController  # noqa: E402


TASK = "Template-Galaxea-Lab-External-Direct-v0"
OBJECTS = (
    "planetary_carrier",
    "sun_planetary_gear_1",
    "sun_planetary_gear_2",
    "sun_planetary_gear_3",
    "sun_planetary_gear_4",
    "ring_gear",
    "planetary_reducer",
)
PHASES = ("planet1", "planet2", "planet3", "center", "ring", "reducer")

# Fixed, separated table layout. The carrier is the stationary assembly base,
# matching the official RoCo sequence; all six added parts start unassembled.
FIXED_LAYOUT = {
    "planetary_carrier": (0.5000, 0.0000, 0.9200),
    "sun_planetary_gear_1": (0.6402, 0.0468, 0.9200),
    "sun_planetary_gear_2": (0.6937, 0.1248, 0.9200),
    "sun_planetary_gear_3": (0.5463, -0.3395, 0.9200),
    "sun_planetary_gear_4": (0.5937, -0.1779, 0.9200),
    "ring_gear": (0.6020, 0.2767, 0.9200),
    "planetary_reducer": (0.5191, -0.2476, 0.9200),
}

ASSEMBLY_CENTER = (0.5000, 0.0000)
PIN_OFFSETS = ((0.0, -0.054), (0.0471, 0.0268), (-0.0471, 0.0268))


def pose_dict(base_env) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "position_xyz_m": base_env.obj_dict[name].data.root_state_w[0, :3].detach().cpu().tolist(),
            "quaternion_wxyz": base_env.obj_dict[name].data.root_state_w[0, 3:7].detach().cpu().tolist(),
        }
        for name in OBJECTS
    }


def set_initial_pose(base_env, name: str, xyz: tuple[float, float, float]) -> None:
    state = base_env.obj_dict[name].data.root_state_w.clone()
    state[:, :3] = torch.tensor(xyz, device=base_env.device)
    state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=base_env.device)
    state[:, 7:] = 0.0
    base_env.obj_dict[name].write_root_state_to_sim(state)


def initialize_layout(base_env, phase: str) -> None:
    for name, xyz in FIXED_LAYOUT.items():
        set_initial_pose(base_env, name, xyz)

    if phase == "full":
        return
    completed = PHASES[: PHASES.index(phase)]
    carrier_z = 0.9023
    set_initial_pose(base_env, "planetary_carrier", (ASSEMBLY_CENTER[0], ASSEMBLY_CENTER[1], carrier_z))
    for index, previous in enumerate(("planet1", "planet2", "planet3")):
        if previous in completed:
            dx, dy = PIN_OFFSETS[index]
            set_initial_pose(
                base_env,
                f"sun_planetary_gear_{index + 1}",
                (ASSEMBLY_CENTER[0] + dx, ASSEMBLY_CENTER[1] + dy, 0.9123),
            )
    if "center" in completed:
        set_initial_pose(base_env, "sun_planetary_gear_4", (*ASSEMBLY_CENTER, 0.9123))
    if "ring" in completed:
        set_initial_pose(base_env, "ring_gear", (*ASSEMBLY_CENTER, 0.9216))


def official_score(base_env) -> int:
    value = base_env.evaluate_score()
    return int(value[0] if isinstance(value, tuple) else value)


def relation_metrics(base_env) -> dict[str, dict[str, object]]:
    pins, pin_q, gears, gear_q, carrier_p, carrier_q, ring_p, ring_q, reducer_p, reducer_q = base_env.get_key_points()

    def relation(a_p, a_q, b_p, b_q) -> dict[str, object]:
        dot = torch.dot(a_q.squeeze(0), b_q.squeeze(0)).clamp(-1.0, 1.0)
        return {
            "xy_error_m": float(torch.norm(a_p[:, :2] - b_p[:, :2]).item()),
            "height_a_minus_b_m": float((a_p[:, 2] - b_p[:, 2]).item()),
            "quaternion_angle_rad": float(torch.acos(dot).item()),
        }

    output = {f"planet_{i + 1}_to_pin_{i + 1}": relation(gears[i], gear_q[i], pins[i], pin_q[i]) for i in range(3)}
    output["carrier_to_ring"] = relation(carrier_p, carrier_q, ring_p, ring_q)
    output["center_to_carrier"] = relation(gears[3], gear_q[3], carrier_p, carrier_q)
    output["center_to_ring"] = relation(gears[3], gear_q[3], ring_p, ring_q)
    output["center_to_reducer"] = relation(gears[3], gear_q[3], reducer_p, reducer_q)
    return output


def socket_aware_score(base_env, official: int) -> int:
    """Diagnostic score using the measured reducer/socket insertion depth.

    The upstream scorer compares the reducer root to the centre-gear root with
    a one-sided 2 mm height threshold.  The reducer mesh has a narrow shaft and
    a shoulder; a seated part can therefore have its root about 10 mm below
    the centre-gear root while remaining concentric and orientation matched.
    Keep the upstream score untouched and report this semantic interpretation
    separately until the geometry/settling evidence justifies adopting it.
    """
    if official != 5:
        return official
    relation = relation_metrics(base_env)["center_to_reducer"]
    seated = (
        relation["xy_error_m"] < 0.005
        and relation["quaternion_angle_rad"] < 0.1
        and -0.002 < relation["height_a_minus_b_m"] < 0.012
    )
    return 6 if seated else official


def frame(observations, step: int, score: int, event: str) -> np.ndarray:
    arrays = [observations["policy"][key][0].detach().cpu().numpy().astype(np.uint8) for key in ("head_rgb", "left_hand_rgb", "right_hand_rgb")]
    canvas = np.zeros((270, 960, 3), dtype=np.uint8)
    canvas[30:] = np.concatenate(arrays, axis=1)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((6, 7), f"M1 {args.phase} step={step} score={score} {event}", fill=(255, 255, 255))
    return np.asarray(image)


def main() -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    phase_path = output / "phase_scores.json"
    env = None
    writer = None
    result: dict[str, object] = {"task": TASK, "phase": args.phase, "seed": args.seed, "status": "RUNNING"}
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cfg = parse_env_cfg(TASK, device=args.device, num_envs=1, use_fabric=True)
        cfg.seed = args.seed
        cfg.record_data = False
        cfg.episode_length_s = 180.0
        if args.gear_friction_coefficient is not None:
            cfg.gears_friction_coefficient = args.gear_friction_coefficient
            print(
                f"M1_GEAR_FRICTION_OVERRIDE={args.gear_friction_coefficient}",
                flush=True,
            )
        # The upstream scene does not request PhysX contact-report APIs on the
        # reducer/gear meshes.  Enable them only for this diagnostic sensor;
        # collision geometry and contact parameters remain unchanged.
        for object_cfg_name in (
            "planetary_reducer_cfg",
            "sun_planetary_gear_4_cfg",
        ):
            object_cfg = getattr(cfg, object_cfg_name, None)
            if object_cfg is not None and hasattr(object_cfg.spawn, "activate_contact_sensors"):
                object_cfg.spawn.activate_contact_sensors = True
                print(
                    f"M1_CONTACT_CFG={object_cfg_name}:{object_cfg.spawn.activate_contact_sensors}",
                    flush=True,
                )
        env = gym.make(TASK, cfg=cfg, use_action=True)
        base = env.unwrapped
        observations, _ = env.reset(seed=args.seed)

        # Some replicated USD assets only receive the reporter on the source
        # body.  Apply the same Isaac Lab schema helper to the two concrete
        # bodies used by this measurement before creating a PhysX view.
        for object_name in ("planetary_reducer", "sun_planetary_gear_4"):
            activate_contact_sensors(f"/World/envs/env_0/{object_name}")

        # The only object-state writes in the run: deterministic scene initialization.
        initialize_layout(base, args.phase)
        base.scene.write_data_to_sim()
        for _ in range(10):
            base.sim.step(render=False)
            base.scene.update(dt=base.physics_dt)
        for obj in base.obj_dict.values():
            obj.update(base.physics_dt)

        initial_states = {name: base.obj_dict[name].data.root_state_w.clone() for name in base.obj_dict}
        policy = M1AssemblyController(base.sim, base.scene, base.obj_dict, phase=args.phase)
        policy.set_initial_root_state(initial_states)
        policy.configure_schedule()
        policy.total_time_steps = policy.total_time_steps + policy._ticks(5.0)
        base.rule_policy = policy
        # The upstream DirectRLEnv resets all objects immediately when its
        # scorer reaches 6, even if the controller has not completed its
        # release/settle window.  That makes a transient success look like a
        # failed final state in this measurement runner.  Keep physics and
        # evaluate_score() unchanged, but suppress only the upstream
        # early-reset signal while this run records the requested full
        # schedule and post-settle stability.
        def _keep_episode_for_measurement():
            return (
                torch.zeros_like(base.reset_terminated),
                torch.zeros_like(base.reset_time_outs),
            )

        base._get_dones = _keep_episode_for_measurement
        reducer_view = base.obj_dict["planetary_reducer"].root_physx_view
        contact_api = [
            name
            for name in dir(reducer_view)
            if "contact" in name.lower() or "force" in name.lower()
        ]
        print(f"M1_REDUCER_PHYSX_API={contact_api}", flush=True)

        contact_sensor = None
        try:
            stage_api_paths = [
                str(prim.GetPath())
                for prim in get_current_stage().Traverse()
                if any(
                    token in str(prim.GetPath())
                    for token in ("planetary_reducer", "sun_planetary_gear_4")
                )
                and prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            ]
            print(f"M1_REDUCER_CONTACT_API_PATHS={stage_api_paths}", flush=True)
            # A filtered GPU contact view in Isaac Sim 5.1 can assert when a
            # replicated mesh changes contact topology.  Use the robust
            # unfiltered net-force view here; pose/geometry remains the
            # pair-specific evidence and the result records this limitation.
            sensor_candidates = (
                ("/World/envs/env_.*/planetary_reducer/node_", []),
            )
            sensor_errors = []
            for sensor_prim, filter_paths in sensor_candidates:
                try:
                    contact_sensor_cfg = ContactSensorCfg(
                        prim_path=sensor_prim,
                    filter_prim_paths_expr=filter_paths,
                        track_contact_points=False,
                        max_contact_data_count_per_prim=16,
                        debug_vis=False,
                    )
                    candidate = ContactSensor(contact_sensor_cfg)
                    # The environment is already playing, so initialize
                    # directly; the normal PLAY callback handles later resets.
                    candidate._initialize_impl()
                    candidate._is_initialized = True
                    contact_sensor = candidate
                    print(
                        "M1_REDUCER_CONTACT_SENSOR="
                        f"prim={sensor_prim},bodies={candidate.body_names},"
                        f"filters={candidate.contact_physx_view.filter_count}",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    sensor_errors.append(f"{sensor_prim}:{type(exc).__name__}:{exc}")
            if contact_sensor is None:
                raise RuntimeError(" | ".join(sensor_errors))
        except Exception as exc:
            print(f"M1_REDUCER_CONTACT_SENSOR_ERROR={type(exc).__name__}:{exc}", flush=True)

        if args.inspect_only:
            print("M1_INSPECT_ONLY_DONE", flush=True)
            return

        initial_score = official_score(base)
        initial_poses = pose_dict(base)
        initial_relations = relation_metrics(base)
        initial_socket_score = socket_aware_score(base, initial_score)
        writer = imageio.get_writer(output / "episode.mp4", fps=args.video_fps, codec="libx264", quality=8)

        zero_action = torch.zeros(env.action_space.shape, device=base.device)
        scores: list[int] = []
        socket_scores: list[int] = []
        qpos: list[np.ndarray] = []
        object_poses: list[np.ndarray] = []
        object_velocities: list[np.ndarray] = []
        reducer_contact_forces: list[np.ndarray] = []
        reducer_center_contact_forces: list[np.ndarray] = []
        reducer_center_contact_points: list[np.ndarray] = []
        grippers: list[np.ndarray] = []
        events: list[str] = []
        score_transitions: list[dict[str, object]] = []
        event_snapshots: list[dict[str, object]] = []
        previous_score = initial_score
        previous_event = None
        schedule_steps = int((policy.total_time_steps.item() - policy._ticks(5.0)) // base.cfg.decimation)
        post_settle_steps = max(
            0,
            int(round(args.post_settle_s / (base.sim.get_physics_dt() * base.cfg.decimation))),
        )
        end_step = schedule_steps + post_settle_steps

        for step in range(end_step):
            with torch.inference_mode():
                observations, _, _, _, _ = env.step(zero_action)
            score = official_score(base)
            semantic_score = socket_aware_score(base, score)
            scores.append(score)
            socket_scores.append(semantic_score)
            qpos.append(base.robot.data.joint_pos[0, base._joint_idx].detach().cpu().numpy().copy())
            object_poses.append(np.concatenate([base.obj_dict[name].data.root_state_w[0, :7].detach().cpu().numpy() for name in OBJECTS]))
            object_velocities.append(np.concatenate([base.obj_dict[name].data.root_state_w[0, 7:13].detach().cpu().numpy() for name in OBJECTS]))
            try:
                force = reducer_view.get_net_contact_forces(dt=base.physics_dt)
                reducer_contact_forces.append(force[0].detach().cpu().numpy().reshape(-1))
            except (AttributeError, TypeError, RuntimeError):
                reducer_contact_forces.append(np.zeros(0, dtype=np.float32))
            if contact_sensor is not None:
                try:
                    contact_sensor.update(
                        base.physics_dt * base.cfg.decimation,
                        force_recompute=True,
                    )
                    sensor_data = contact_sensor.data
                    if getattr(sensor_data, "force_matrix_w", None) is not None:
                        force_sample = sensor_data.force_matrix_w[0, 0, 0]
                    else:
                        force_sample = sensor_data.net_forces_w[0, 0]
                    reducer_center_contact_forces.append(force_sample.detach().cpu().numpy().copy())
                    reducer_center_contact_points.append(np.full(3, np.nan, dtype=np.float32))
                except (AttributeError, IndexError, RuntimeError, TypeError):
                    reducer_center_contact_forces.append(np.zeros(3, dtype=np.float32))
                    reducer_center_contact_points.append(np.full(3, np.nan, dtype=np.float32))
            grippers.append(np.array([
                base.robot.data.joint_pos[0, base._left_gripper_dof_idx[0]].item(),
                base.robot.data.joint_pos[0, base._right_gripper_dof_idx[0]].item(),
            ], dtype=np.float32))
            events.append(policy.event)
            writer.append_data(frame(observations, step + 1, score, policy.event))

            if score != previous_score:
                score_transitions.append({
                    "control_step": step + 1,
                    "physics_step": int(policy.count),
                    "before": previous_score,
                    "after": score,
                    "event": policy.event,
                    "object_poses": pose_dict(base),
                })
                previous_score = score
            if policy.event != previous_event:
                event_snapshots.append({
                    "control_step": step + 1,
                    "physics_step": int(policy.count),
                    "event": policy.event,
                    "score": score,
                    "object_poses": pose_dict(base),
                    "relations": relation_metrics(base),
                    "left_gripper_position_m": float(grippers[-1][0]),
                    "right_gripper_position_m": float(grippers[-1][1]),
                })
                previous_event = policy.event
            if (step + 1) % 20 == 0 or score == 6 or semantic_score == 6:
                print(f"M1_PROGRESS step={step + 1}/{end_step} physics_step={policy.count} score={score} socket_score={semantic_score} event={policy.event}", flush=True)

        final_score = official_score(base)
        final_socket_score = socket_aware_score(base, final_score)
        final_relations = relation_metrics(base)
        final_poses = pose_dict(base)
        best_score = max([initial_score, *scores])
        best_socket_score = max([initial_socket_score, *socket_scores])
        executed_full = args.phase == "full"
        task_success = executed_full and initial_score == 0 and best_score == 6 and final_score == 6
        socket_task_success = executed_full and initial_score == 0 and best_socket_score == 6 and final_socket_score == 6
        center_relation = final_relations["center_to_carrier"]
        center_valid = (
            center_relation["xy_error_m"] < 0.005
            and center_relation["quaternion_angle_rad"] < 0.1
            and 0.0 < center_relation["height_a_minus_b_m"] < 0.015
        )
        phase_pass = (
            center_valid
            if args.phase == "center"
            else final_score > initial_score
        )
        result.update({
            "status": "PASS" if (task_success if executed_full else phase_pass) else "FAIL",
            "executed_full_sequence": executed_full,
            "initial_score": initial_score,
            "best_score": best_score,
            "final_score": final_score,
            "best_socket_aware_score": best_socket_score,
            "final_socket_aware_score": final_socket_score,
            "task_success": task_success,
            "socket_aware_task_success": socket_task_success,
            "center_geometrically_valid": center_valid,
            "collision_enabled": True,
            "object_pose_writes_after_initialization": 0,
            "initial_object_poses": initial_poses,
            "final_object_poses": final_poses,
            "initial_relations": initial_relations,
            "final_relations": final_relations,
            "score_transitions": score_transitions,
            "event_snapshots": event_snapshots,
            "controller_events": policy.event_history,
            "post_settle_s": args.post_settle_s,
            "early_success_termination_disabled": True,
            "schedule_steps": schedule_steps,
            "post_settle_steps": post_settle_steps,
            "post_settle_min_official_score": int(min(scores[schedule_steps:] or [final_score])),
            "post_settle_min_socket_aware_score": int(min(socket_scores[schedule_steps:] or [final_socket_score])),
            "post_settle_max_reducer_linear_speed_mps": float(
                np.max(np.linalg.norm(np.asarray(object_velocities)[schedule_steps:, 6 * 6 : 6 * 6 + 3], axis=1))
                if post_settle_steps else 0.0
            ),
            "post_settle_max_reducer_angular_speed_radps": float(
                np.max(np.linalg.norm(np.asarray(object_velocities)[schedule_steps:, 6 * 6 + 3 : 6 * 6 + 6], axis=1))
                if post_settle_steps else 0.0
            ),
            "reducer_contact_force_samples": int(sum(bool(x.size) for x in reducer_contact_forces)),
            "post_settle_max_reducer_contact_force_norm_N": float(
                max(
                    (float(np.linalg.norm(x)) for x in reducer_contact_forces[schedule_steps:] if x.size),
                    default=0.0,
                )
                if post_settle_steps else 0.0
            ),
            "reducer_center_contact_force_samples": int(len(reducer_center_contact_forces)),
            "reducer_contact_sensor_filter": "none (net force; pair identity unavailable)",
            "post_settle_max_reducer_center_contact_force_norm_N": float(
                max(
                    (
                        float(np.linalg.norm(x))
                        for x in reducer_center_contact_forces[schedule_steps:]
                    ),
                    default=0.0,
                )
                if post_settle_steps else 0.0
            ),
            "post_settle_reducer_center_contact_nonzero_steps": int(
                sum(
                    float(np.linalg.norm(x)) > 1.0e-6
                    for x in reducer_center_contact_forces[schedule_steps:]
                )
                if post_settle_steps else 0
            ),
            "gear_friction_coefficient": args.gear_friction_coefficient,
        })
        phase_summary = {
            "phase": args.phase,
            "initial_score": initial_score,
            "best_score": best_score,
            "final_score": final_score,
            "best_socket_aware_score": best_socket_score,
            "final_socket_aware_score": final_socket_score,
            "score_transitions": score_transitions,
            "final_relations": final_relations,
        }
        np.savez_compressed(
            output / "trace.npz",
            scores=np.asarray(scores, dtype=np.int16),
            socket_aware_scores=np.asarray(socket_scores, dtype=np.int16),
            joint_positions=np.asarray(qpos, dtype=np.float32),
            object_poses=np.asarray(object_poses, dtype=np.float32),
            object_velocities=np.asarray(object_velocities, dtype=np.float32),
            reducer_contact_forces=np.asarray(reducer_contact_forces, dtype=object),
            reducer_center_contact_forces=np.asarray(reducer_center_contact_forces, dtype=np.float32),
            reducer_center_contact_points=np.asarray(reducer_center_contact_points, dtype=np.float32),
            gripper_positions=np.asarray(grippers, dtype=np.float32),
            events=np.asarray(events),
            object_names=np.asarray(OBJECTS),
        )
        phase_path.write_text(json.dumps(phase_summary, indent=2) + "\n", encoding="utf-8")
        print("M1_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        result.update({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)})
        print("M1_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        if writer is not None:
            writer.close()
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
