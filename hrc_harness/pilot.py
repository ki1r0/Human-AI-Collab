"""Offline guarded calibration: fixed nominal commands, full gravity, no part-pose feedback."""

import argparse
import copy
import json
import math
import os
import subprocess
import time
from pathlib import Path

from .evaluation import conjugate, multiply, rotate
from .isaac import IsaacSceneAdapter, configure_scene, load_task, reset_scene
from .runtime import Interrupted


class SeatContactReached(Exception):
    """End the placement descent; this is not a seating-success signal."""


def nominal_points(task, plan):
    """CAD/reset priors only, not measured runtime poses."""
    q = plan["grasp_quat_wxyz"]
    tool = rotate(q, task["frames"]["robot_tool_offset_m"])
    x, y = task["scene"]["cover_initial_xy"]
    x_offset = plan.get("grasp_link6_x_offset_m", 0)
    close = [x+x_offset+tool[0], y+plan["grasp_pair_offset_m"]+tool[1],
             plan["nominal_source_root_z_m"]+plan["grasp_link6_z_offset_m"]+tool[2]]
    cx, cy = task["scene"]["casing_initial_xy"]
    trim = plan.get("seat_target_trim_casing_local_m", [0.0, 0.0, 0.0])
    if len(trim) != 3 or not all(math.isfinite(value) for value in trim):
        raise ValueError("seat target trim must be a finite 3-vector in casing-local metres")
    seat_relative = [a+b for a, b in zip(task["frames"]["seated_relative_position_m"], trim)]
    offset = rotate(task["scene"].get("casing_initial_quat_wxyz", [1, 0, 0, 0]),
                    seat_relative)
    seat = [cx+offset[0]+x_offset+tool[0], cy+offset[1]+plan["grasp_pair_offset_m"]+tool[1],
            plan["nominal_casing_root_z_m"]+offset[2]+plan["grasp_link6_z_offset_m"]+tool[2]]
    return close, seat


def measure_grasp(checkpoints):
    """Post-run private measurements only, never a controller input."""
    phases = {row["phase"]: row for row in checkpoints}
    if not {"close", "held_dwell"} <= phases.keys():
        return {"status": "insufficient_lift_data"}
    a, b = phases["close"], phases["held_dwell"]
    hub_delta = [y-x for x, y in zip(a["private"]["hub_xyz"], b["private"]["hub_xyz"])]
    tcp_delta = [y-x for x, y in zip(a["public"]["tcp_xyz_m"], b["public"]["tcp_xyz_m"])]
    relative_xyz, relative_q = [], []
    for row in (a, b):
        inverse_tcp = conjugate(row["public"]["tcp_quat_wxyz"])
        relative_xyz.append(rotate(inverse_tcp, [h-t for h, t in
                            zip(row["private"]["hub_xyz"], row["public"]["tcp_xyz_m"])]))
        relative_q.append(multiply(inverse_tcp, row["private"]["hub_quat"]))
    dot = abs(sum(x*y for x, y in zip(*relative_q)))
    norm = math.prod(math.sqrt(sum(v*v for v in q)) for q in relative_q)
    return {"status": "measured_not_certified", "cover_lift_m": hub_delta[2],
            "tcp_lift_m": tcp_delta[2], "relative_translation_drift_m": math.dist(*relative_xyz),
            "relative_rotation_drift_deg": math.degrees(2*math.acos(min(1., dot/norm))),
            "final_pair_contact_N": b["public"]["pair_contact_N"]}


def run_pilot(adapter, plan, output, *, stage="grasp"):
    output = Path(output)
    started = time.monotonic()
    samples, checkpoints, commands = [], [], []
    phase, require_pair, missing_since = "settle", False, None
    contact_stop_armed = False
    original_step = adapter.idle_step
    dt = adapter.env.cfg.sim_dt
    safety = plan["safety"]
    completed, failure = [], None
    close, seat = nominal_points(adapter.task, plan)
    q = plan["grasp_quat_wxyz"]
    dual = plan.get("dual_arm_tcp_offset_m") is not None
    finger_ids, finger_names = adapter.env.robot.find_bodies(".*_gripper_link[12]" if dual else "left_gripper_link[12]")
    contact_names = [f"{side}_gripper_link{i}_contact" for side in (("left", "right") if dual else ("left",)) for i in (1, 2)]

    def sample():
        xyz, quat = adapter._tcp()
        opening = float(adapter.env.robot.data.joint_pos[0, adapter.env.left_gripper_cfg.joint_ids].abs().mean().item())
        row = {"physics_tick": adapter.tick, "phase": phase, "tcp_xyz_m": xyz, "tcp_quat_wxyz": quat,
               "qpos": adapter.env.robot.data.joint_pos[0].detach().cpu().tolist(),
               "qvel": adapter.env.robot.data.joint_vel[0].detach().cpu().tolist(),
               "gripper_link_poses_wxyz_m": {name: adapter.env.robot.data.body_state_w[0, index, :7].detach().cpu().tolist()
                                            for name, index in zip(finger_names, finger_ids)},
               "commanded_gripper_opening_m": adapter.env._gripper_target,
               "gripper_position_target_m": adapter.env.robot.data.joint_pos_target[0, adapter.env.left_gripper_cfg.joint_ids].detach().cpu().tolist(),
               "gripper_applied_effort_N": adapter.env.robot.data.applied_torque[0, adapter.env.left_gripper_cfg.joint_ids].detach().cpu().tolist(),
               "finger_net_contact_N": [float(adapter.torch.linalg.vector_norm(getattr(adapter.env, name).data.net_forces_w).item())
                                        for name in contact_names],
               "gripper_opening_m": opening, "contact_scalar_N": adapter._force("hub_contact"),
               "pair_contact_N": [adapter._force(name) for name in contact_names],
               "tracking_error_m": math.dist(xyz, adapter.target or adapter.hold_tcp)}
        samples.append(row)
        public.write(json.dumps(row, allow_nan=False)+"\n")
        public.flush()

    def step():
        original_step()
        sample()
        if adapter.env.cfg.spawn_cameras and adapter.tick % adapter.env.cfg.camera_update_stride == 0:
            adapter.observe(0)

    def check():
        nonlocal missing_since
        if time.monotonic()-started > plan["wall_limit_s"]:
            raise Interrupted("pilot_wall_limit")
        reason = adapter.protection(safety, motion=True)
        if reason:
            raise Interrupted(reason)
        pair = [adapter._force(name) for name in contact_names]
        if any(value is None for value in pair):
            raise Interrupted("missing_finger_sensor")
        if max(pair+samples[-1]["finger_net_contact_N"]) > safety["finger_force_limit_N"]:
            raise Interrupted("finger_force_limit")
        if require_pair and min(pair) < plan["pair_contact_probe_N"]:
            missing_since = adapter.tick if missing_since is None else missing_since
            if (adapter.tick-missing_since)*dt >= plan["pair_loss_window_s"]:
                raise Interrupted("pair_contact_lost")
        else:
            missing_since = None
        if contact_stop_armed and adapter._force("hub_contact") >= plan["seat_contact_stop_N"]:
            raise SeatContactReached

    def wait(seconds):
        for _ in range(math.ceil(seconds/dt)):
            check()
            adapter.idle_step()
            check()

    def mark():
        completed.append(phase)
        # Private checkpoint is for post-run labels only; no caller reads it for commands.
        checkpoints.append({"phase": phase, "physics_tick": adapter.tick,
                            "public": copy.deepcopy(samples[-1]), "private": copy.deepcopy(adapter.private_samples[-1])})
        print(f"pilot phase={phase} tick={adapter.tick}", flush=True)

    def move(target, *, speed=None, opening=None, insert=False, stop_contact=False):
        nonlocal contact_stop_armed
        xyz, _ = adapter._tcp()
        duration = max(2.0, math.dist(xyz, target)/(speed or plan["translation_speed_mps"]))
        if phase == "release" and opening is not None:
            duration = max(plan.get("release_ramp_s", 2.0), math.dist(xyz, target)/(speed or plan["translation_speed_mps"]))
        if dual:
            right = adapter.env.robot.data.body_state_w[0, adapter.env.right_arm_cfg.body_ids[0], :7].detach().cpu().tolist()
            tool = rotate(right[3:7], adapter.task["frames"]["robot_tool_offset_m"])
            right_xyz = [a+b for a, b in zip(right[:3], tool)]
            right_goal = [a+b for a, b in zip(target, plan["dual_arm_tcp_offset_m"])]
            duration = max(duration, math.dist(right_xyz, right_goal)/(speed or plan["translation_speed_mps"]))
        point = {"tcp_xyz_m": target, "quat_wxyz": q, "ticks": math.ceil(duration/dt)}
        if opening is not None:
            point["gripper_opening_m"] = opening
        commands.append({"phase": phase, **point})
        insertion = (xyz, target, [0, 0, -1], True) if insert else None
        if insert:
            adapter.insertion_elapsed_s = 0
            adapter.insertion_check.samples.clear()
        contact_stop_armed = stop_contact
        held_target = None
        contact_stopped = False
        try:
            if adapter._waypoint(point, check, insertion=insertion):
                raise Interrupted(adapter.insertion_report["reason"])
        except SeatContactReached:
            contact_stopped = True
            commands[-1]["stop_reason"] = "contact_scalar_trigger_not_seat_verification"
            commands[-1]["stopped_command_tcp_xyz_m"] = list(adapter.target)
            adapter.insertion_report["placement_stop_reason"] = commands[-1]["stop_reason"]
            # Cancel the remaining insertion target before any release motion.
            held_target = list(adapter._tcp()[0])
            adapter.safe_hold()
        finally:
            contact_stop_armed = False
        if held_target is None:
            held_target = list(adapter.target)
        if not contact_stopped:
            wait(plan["settle_s"])
        mark()
        return held_target

    adapter.idle_step = step
    with (output/"public_samples.jsonl").open("w") as public:
        try:
            sample()
            wait(plan["settle_s"])
            mark()
            if stage == "jaws":
                for opening in (.02, .045, .048, .05):
                    phase = f"jaw_target_{opening}"
                    xyz, quat = adapter._tcp()
                    point = {"tcp_xyz_m": xyz, "quat_wxyz": quat, "ticks": math.ceil(2/dt),
                             "gripper_opening_m": opening}
                    commands.append({"phase": phase, **point})
                    adapter._waypoint(point, check)
                    mark()
            else:
                entry_offset = plan.get("entry_offset_m", [0, 0, 0])
                outside = [value+delta for value, delta in zip(close, entry_offset)]
                phase = "approach"
                move([outside[0], outside[1], outside[2]+plan.get("approach_clearance_m", .213)],
                     opening=plan["approach_opening_m"])
                if stage == "reach":
                    phase = "reach_target"
                    move([seat[0], seat[1], plan["transport_tcp_z_m"]])
                    phase = "reach_preinsert"
                    move([seat[0], seat[1], seat[2]+plan["preinsert_clearance_m"]])
                    raise Interrupted("empty_hand_reach_check_complete")
                phase = "descend"
                move(outside)
                if any(entry_offset):
                    phase = "enter"
                    move(close)
                phase = "close"
                move(close, opening=plan["grasp_opening_m"])
                tail = samples[-math.ceil(.25/dt):]
                if not all(min(row["pair_contact_N"]) >= plan["pair_contact_probe_N"] for row in tail):
                    raise Interrupted("pair_contact_not_established")
                require_pair = True
                phase = "lift"
                move([close[0], close[1], close[2]+plan["lift_m"]], speed=plan.get("lift_speed_mps"))
                phase = "held_dwell"
                wait(1.0)
                mark()
                if stage == "insert":
                    phase = "rise"
                    move([close[0], close[1], plan["transport_tcp_z_m"]])
                    phase = "transfer"
                    move([seat[0], seat[1], plan["transport_tcp_z_m"]],
                         speed=plan.get("transfer_speed_mps"))
                    phase = "preinsert"
                    move([seat[0], seat[1], seat[2]+plan["preinsert_clearance_m"]],
                         speed=plan.get("preinsert_speed_mps"))
                    phase = "insert"
                    move(seat, speed=plan["insert_speed_mps"], insert=True)
                    # No release from an unmeasured seat: retain grasp for this calibration pilot.
                elif stage == "place":
                    phase = "rise"
                    move([close[0], close[1], plan["transport_tcp_z_m"]])
                    phase = "transfer"
                    move([seat[0], seat[1], plan["transport_tcp_z_m"]],
                         speed=plan.get("transfer_speed_mps"))
                    phase = "preinsert"
                    move([seat[0], seat[1], seat[2]+plan["preinsert_clearance_m"]],
                         speed=plan.get("preinsert_speed_mps"))
                    phase = "insert"
                    release_target = move(seat, speed=plan["insert_speed_mps"], insert=True,
                                          stop_contact=plan.get("seat_contact_stop_N") is not None)
                    window = samples[-math.ceil(.1/dt):]
                    minimum_contact = plan["seat_contact_min_N"]
                    if not window or any(row["contact_scalar_N"] is None for row in window) or not any(
                            row["contact_scalar_N"] >= minimum_contact for row in window):
                        raise Interrupted("no_seat_contact_before_release")
                    require_pair = False
                    phase = "release"
                    move(release_target, opening=plan["release_opening_m"])
                    phase = "release_dwell"
                    wait(plan["release_dwell_s"])
                    mark()
                    phase = "retract"
                    move([release_target[0], release_target[1], release_target[2]+plan["retract_clearance_m"]],
                         opening=plan["release_opening_m"])
                    phase = "settle_release"
                    wait(plan["release_dwell_s"])
                    mark()
        except Interrupted as error:
            failure = str(error)
        finally:
            adapter.safe_hold()
            adapter.idle_step = original_step
            public.flush()
    result = {"role": "offline_physics_calibration", "completed_phases": completed,
              "exit_reason": failure or "pilot_complete", "physics_ticks": adapter.tick,
              "production_profiles_authorized": False, "insertion_check": adapter.insertion_report,
              "commands": commands, "checkpoints": checkpoints,
              "grasp_measurement": measure_grasp(checkpoints),
              "terminal_evaluation": adapter.evaluate(), "private_layout": adapter.env.scatter_reset_report()}
    with (output/"calibration_private.json").open("w") as stream:
        os.chmod(stream.fileno(), 0o600)
        json.dump(result, stream, indent=2, allow_nan=False)
    summary = {key: result[key] for key in (
        "role", "completed_phases", "exit_reason", "physics_ticks",
        "production_profiles_authorized", "insertion_check", "grasp_measurement",
        "terminal_evaluation",
    )}
    (output/"metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return result


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/harness_pilot.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("jaws", "reach", "grasp", "insert", "place"), default="grasp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    plan = load_task(args.config)
    task = load_task(plan["task_config"])
    task["scene"].update(plan["scene"])
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (args.output/"manifest.json").write_text(json.dumps({"role": "offline_physics_calibration", "seed": args.seed,
        "stage": args.stage, "git_revision": revision, "task": task, "plan": plan}, indent=2)+"\n")
    app = AppLauncher(args).app
    env, adapter = None, None
    try:
        from hrc_m1.roco_env import make_env_classes
        cfg_cls, env_cls = make_env_classes()
        cfg = configure_scene(cfg_cls(), task, args.seed, cameras=not args.no_video)
        env = env_cls(cfg)
        reset_scene(env, args.seed)
        (args.output/"robot_setup.json").write_text(json.dumps({
            "joint_names": env.robot.joint_names,
            "joint_limits": env.robot.data.joint_pos_limits[0].detach().cpu().tolist(),
            "initial_qpos": env.robot.data.joint_pos[0].detach().cpu().tolist(),
            "torso_limits": "zero-width bundle posture" if cfg.torso_runtime_override else "authored zero-width limits",
            "mimic_initialization": "axis2 initialized at the same opening as axis1",
        }, indent=2)+"\n")
        adapter = IsaacSceneAdapter(env, plan, task, args.output)
        if not args.no_video:
            adapter.observe(0)
        result = run_pilot(adapter, plan, args.output, stage=args.stage)
        print(json.dumps({key: value for key, value in result.items()
                         if key not in {"commands", "checkpoints", "private_layout"}}), flush=True)
    finally:
        if adapter:
            adapter.close()
        if env:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
