"""Tick-stepped nominal Cartesian execution on the existing Isaac scene.

No online method reads part root poses. Initial layout changes are reset-only;
the sole runtime scene mutation is the scoped helper operation.
"""

import argparse
import json
import math
import time
import uuid
import os
from collections import deque
from pathlib import Path

from .contracts import Channel, HelpReport, MeasuredCost, ObservationPack
from .evaluation import rotate
from .evidence import InsertionCheck
from .runtime import CONTACT_TOOLS, Interrupted


class IsaacSceneAdapter:
    def __init__(self, env, config, task, output):
        import torch
        self.env, self.config, self.task, self.torch = env, config, task, torch
        self.output = Path(output)
        self.episode_id = uuid.uuid4().hex
        self.tick, self.stage = 0, "initial"
        self.target, self.previous_tcp = None, None
        self.target_quat = None
        self.last_camera_stamp = 0.0
        self.last_robot_stamp = time.monotonic()
        self.hold_tcp = self._tcp()[0]
        self.force_exposure, self.axial_delta = 0.0, None
        self.axial_stamp = self.last_robot_stamp
        self.insertion_check = InsertionCheck(task.get("online_check", {}))
        self.insertion_report, self.insertion_stamp = {}, self.last_robot_stamp
        self.contact_exposure_valid = True
        self.media = None
        capacity = math.ceil(task["evaluation"]["settle_window_s"] / env.cfg.sim_dt) + 2
        self.private_samples = deque(maxlen=capacity)
        self.private_stream = None
        self.private_reset = {}
        self.active_requires_holding = False
        self.hold_certified = bool(config["safety"]["hold_certified"])
        if int(env.cfg.decimation) != 1:
            raise ValueError("harness requires a guard check at each physics tick (decimation=1)")
        if env.cfg.hub_cfg.spawn.rigid_props.disable_gravity or env.cfg.hub_cfg.spawn.rigid_props.kinematic_enabled:
            raise ValueError("cover must be a dynamic rigid body with gravity enabled")
        if env.cfg.spawn_grasp_constraint:
            raise ValueError("temporary grasp constraints require a separately declared modeling study")

    def _tcp(self):
        state = self.env.robot.data.body_state_w[0, self.env.left_arm_cfg.body_ids[0]]
        position = state[:3].detach().cpu().tolist()
        quaternion = state[3:7].detach().cpu().tolist()
        offset = rotate(quaternion, self.task["frames"]["robot_tool_offset_m"])
        return [a+b for a, b in zip(position, offset)], quaternion

    def _force(self, name):
        sensor = getattr(self.env, name, None)
        matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
        if matrix is None or not matrix.numel() or not self.torch.isfinite(matrix).all():
            return None
        return float(self.torch.linalg.vector_norm(matrix).item())

    def _holding(self):
        config = self.task["sensors"]
        threshold = config["holding_pair_force_threshold_N"]
        if threshold is None or not config["holding_estimate_certified"]:
            return "unknown"
        values = [self._force("left_gripper_link1_contact"), self._force("left_gripper_link2_contact")]
        if any(value is None for value in values):
            return "unknown"
        return "yes" if all(value >= threshold for value in values) else "no"

    def observe(self, epoch):
        from hrc_m1.roco_adapter import RocoTaskAdapter
        now = time.monotonic()
        if self.media is None:
            import imageio.v2 as imageio
            self.media = RocoTaskAdapter(self.env, frame_dir=self.output / "frames")
            fps = 1 / (self.env.cfg.sim_dt * self.env.cfg.camera_update_stride)
            self.media.video_writer = imageio.get_writer(str(self.output / "episode.mp4"), fps=fps,
                                                        codec="libx264", pixelformat="yuv420p", macro_block_size=None)
        self.media.index += 1
        xyz, quat = self._tcp()
        forces = [self._force("left_gripper_link1_contact"), self._force("left_gripper_link2_contact")]
        joints = self.env.robot.data.joint_pos[0, self.env.left_gripper_cfg.joint_ids]
        opening = float(joints.abs().mean().item())
        released_limit = self.task["sensors"]["released_opening_threshold_m"]
        values = {
            "qpos": (self.env.robot.data.joint_pos[0].detach().cpu().tolist(), "joint_SI(rad_or_m)", "robot_joint_encoder"),
            "qvel": (self.env.robot.data.joint_vel[0].detach().cpu().tolist(), "joint_SI(rad_or_m)/s", "robot_joint_encoder"),
            "tcp_xyz_m": (xyz, "m", "link6_kinematics_with_nominal_tool_offset"),
            "tcp_quat_wxyz": (quat, "wxyz", "robot_link6_kinematics"),
            "tracking_error_m": (math.dist(xyz, self.target or self.hold_tcp), "m", "robot_command_or_hold_reference_tracking"),
            "gripper_opening_m": (opening, "m", "measured_finger_joint_positions"),
            "gripper_contact_N": (sum(forces) if all(f is not None for f in forces) else None, "N", "filtered_finger_contacts"),
            "contact_scalar_N": (self._force("hub_contact"), "N", "filtered_cover_casing_contact_scalar"),
            "contact_exposure_Ns": (self.force_exposure if self.contact_exposure_valid else None, "Ns", "integrated_filtered_contact_scalar"),
            "holding": (self._holding(), "state", "certified_filtered_pair_contact_proxy"),
            "tcp_axial_delta_mm": (self.axial_delta, "mm", "tcp_motion_proxy_not_part_depth"),
            "part_depth_est_mm": (None, "mm", "RGBD_estimator_not_installed"),
            "tilt_est_deg": (None, "deg", "RGBD_estimator_not_installed"),
            "target_visible": (None, "state", "public_visual_estimator_not_installed"),
            "visual_seated": (None, "state", "public_visual_estimator_not_installed"),
            "stable_observed": (None, "state", "public_visual_estimator_not_installed"),
            "released": (opening >= released_limit if released_limit is not None else None, "state", "measured_fingers"),
        }
        channels = {name: Channel(value, units, self.last_robot_stamp, source, valid=value is not None)
                    for name, (value, units, source) in values.items()}
        channels["tcp_axial_delta_mm"] = Channel(self.axial_delta, "mm", self.axial_stamp,
                                                 "tcp_motion_proxy_not_part_depth", self.axial_delta is not None)
        for name, key, units in (("insertion_state", "state", "state"),
                                 ("insertion_reason", "reason", "state"),
                                 ("insertion_elapsed_s", "elapsed_s", "s")):
            value = self.insertion_report.get(key)
            channels[name] = Channel(value, units, self.insertion_stamp,
                                     "rule_force_tcp_time_last_insertion_attempt", valid=value is not None)
        frames, stamps = {}, {}
        for alias in self.task["sensors"]["camera_aliases"]:
            path = self.media._save_image(self.env._last_obs.get(alias), alias)
            if path:
                frames[alias], stamps[alias] = path, self.last_camera_stamp
        return ObservationPack(self.episode_id, uuid.uuid4().hex, epoch, self.tick, now,
                               self.stage, channels, frames, stamps)

    def idle_step(self):
        self.env.step(self.torch.zeros((1, 14), device=self.env.device))
        self.tick += 1
        now = time.monotonic()
        self.last_robot_stamp = now
        if self.env._camera_tick % max(1, int(self.env.cfg.camera_update_stride)) == 0:
            self.last_camera_stamp = now
        xyz, _ = self._tcp()
        self.measured_speed = math.dist(xyz, self.previous_tcp) / float(self.env.cfg.sim_dt) if self.previous_tcp else 0.0
        self.previous_tcp = xyz
        force = self._force("hub_contact")
        if force is not None:
            self.force_exposure += force * float(self.env.cfg.sim_dt)
        else:
            self.contact_exposure_valid = False
        if self.media and self.env._camera_tick % max(1, int(self.env.cfg.camera_update_stride)) == 0:
            # Match the existing RoCo live recorder's four-camera layout;
            # overhead is a recording view, not an added policy observation.
            self.media._append_video([self.env._last_obs.get(alias) for alias in
                                      ("head_rgb", "overhead_rgb", "left_hand_rgb", "right_hand_rgb")])
        self._record_private()

    def safe_hold(self):
        self.env.clear_pose_target()
        self.env._joint_target = (
            self.env.robot.data.joint_pos[:, self.env.left_arm_cfg.joint_ids].clone(),
            self.env.robot.data.joint_pos[:, self.env.right_arm_cfg.joint_ids].clone())
        self.target = None
        self.target_quat = None
        self.hold_tcp = self._tcp()[0]

    def quiesce(self, check):
        limit = self.config["safety"].get("handover_joint_speed_radps")
        if not self.hold_certified or limit is None:
            raise Interrupted("uncertified_quiescent_hold")
        for _ in range(100):
            check()
            self.idle_step()
            check()
            if float(self.env.robot.data.joint_vel.abs().max().item()) <= limit:
                return
        raise Interrupted("hold_not_quiescent")

    def protection(self, safety, *, motion=False):
        required = ("workspace_xyz_m", "tcp_speed_limit_mps", "joint_speed_limit_radps", "force_limit_N", "tracking_limit_m")
        if motion and any(safety.get(name) is None for name in required):
            return "uncertified_safety_limits"
        force = self._force("hub_contact")
        if motion and force is None:
            return "missing_contact_sensor"
        if motion and self.active_requires_holding and self._holding() != "yes":
            return "grasp_lost_or_unverified"
        if force is not None and safety.get("force_limit_N") is not None and force > safety["force_limit_N"]:
            return "force_limit"
        xyz, _ = self._tcp()
        bounds = safety.get("workspace_xyz_m")
        if bounds and any(not low <= value <= high for value, (low, high) in zip(xyz, bounds)):
            return "workspace_limit"
        ids = list(self.env.left_arm_cfg.joint_ids) + list(self.env.right_arm_cfg.joint_ids)
        joint_speed = float(self.env.robot.data.joint_vel[:, ids].abs().max().item())
        if safety.get("joint_speed_limit_radps") is not None and joint_speed > safety["joint_speed_limit_radps"]:
            return "joint_speed_limit"
        if safety.get("tcp_speed_limit_mps") is not None and getattr(self, "measured_speed", 0) > safety["tcp_speed_limit_mps"]:
            return "tcp_speed_limit"
        if self.target and safety.get("tracking_limit_m") is not None and math.dist(xyz, self.target) > safety["tracking_limit_m"]:
            return "tracking_limit"
        return None

    def execute(self, spec, check):
        if spec.tool in {"observe", "inspect"}:
            for _ in range(max(1, int(self.env.cfg.camera_update_stride))):
                check()
                self.idle_step()
                check()
            return "COMPLETED", "fresh_camera_window", {}, MeasuredCost()
        profile = self.task["profiles"][spec.profile]
        if not profile["certified"] or not profile["waypoints"] or not self.hold_certified:
            raise Interrupted("uncertified_motion_profile_or_hold")
        if spec.tool in {"probe_xy", "probe_angle"} and not profile.get("unload_first"):
            raise Interrupted("probe_requires_certified_unload")
        start, _ = self._tcp()
        self.insertion_report = {}
        self.insertion_check.samples.clear()
        self.insertion_elapsed_s = 0.0
        self.active_requires_holding = spec.requires_holding
        exposure = self.force_exposure
        axis = self.task["frames"].get("socket_axis_world")
        if axis and self.task["frames"].get("certified"):
            if len(axis) != 3 or not all(math.isfinite(v) for v in axis) or abs(sum(v*v for v in axis)-1) > 1e-4:
                raise ValueError("insertion axis must be a finite unit vector")
        else:
            axis = None
        insert_points = [point for point in profile["waypoints"] if point.get("phase") == "insert"]
        insertion_goal = insert_points[-1]["tcp_xyz_m"] if insert_points else None
        insertion_axis = axis
        if axis and insertion_goal:
            travel = sum((b-a)*v for a, b, v in zip(start, insertion_goal, axis))
            insertion_axis = [v * (1 if travel > 0 else -1) for v in axis] if abs(travel) > 1e-9 else None
        stopped = False
        for waypoint in profile["waypoints"]:
            if waypoint.get("phase") == "release":
                if spec.tool == "pick_and_prealign":
                    raise ValueError("pick/prealign cannot release the cover")
                self.active_requires_holding = False
            insertion = (start, insertion_goal, insertion_axis, waypoint is insert_points[-1]) if (
                spec.tool in CONTACT_TOOLS and waypoint.get("phase") == "insert") else None
            if insertion is None:
                self.insertion_check.samples.clear()
            if self._waypoint(waypoint, check, insertion=insertion):
                stopped = True
                break
        end, _ = self._tcp()
        self.axial_delta = 1000 * sum((b-a)*v for a, b, v in zip(start, end, axis)) if axis else None
        self.axial_stamp = self.last_robot_stamp
        self.stage = "prealigned" if spec.tool == "pick_and_prealign" else self.stage
        minimum = profile.get("minimum_axial_progress_mm")
        status = "STALLED" if stopped or (minimum is not None and self.axial_delta is not None and abs(self.axial_delta) < minimum) else "COMPLETED"
        self.safe_hold()
        self.active_requires_holding = False
        actual = {"tcp_displacement_m": math.dist(start, end)}
        actual.update({name: value for name, value in self.insertion_report.items() if isinstance(value, (int, float))})
        reason = self.insertion_report["reason"] if stopped else (
            "measured_progress_window" if status == "STALLED" else "bounded_profile_complete")
        return status, reason, actual, MeasuredCost(contact_exposure_Ns=self.force_exposure - exposure)

    def _waypoint(self, point, check, *, insertion=None):
        start, initial_q = self._tcp()
        continuous = getattr(self, "target", None) is not None and getattr(self, "target_quat", None) is not None
        if continuous:
            # Reusing measured sag at each segment would abruptly unload the PD drives.
            start, initial_q = list(self.target), list(self.target_quat)
        target, quaternion = point["tcp_xyz_m"], point["quat_wxyz"]
        ticks = point["ticks"]
        if type(ticks) is not int or ticks < 1 or len(target) != 3 or len(quaternion) != 4:
            raise ValueError("invalid waypoint")
        if not all(math.isfinite(v) for v in target + quaternion) or abs(sum(v*v for v in quaternion)-1) > 1e-4:
            raise ValueError("nonfinite target or unnormalized quaternion")
        if math.dist(start, target) / (ticks * self.env.cfg.sim_dt) > self.config["safety"]["tcp_speed_limit_mps"]:
            raise Interrupted("command_speed_limit")
        bounds = self.config["safety"]["workspace_xyz_m"]
        if any(not low <= value <= high for value, (low, high) in zip(target, bounds)):
            raise Interrupted("command_workspace_limit")
        if sum(a*b for a, b in zip(initial_q, quaternion)) < 0:
            initial_q = [-v for v in initial_q]
        opening = point.get("gripper_opening_m")
        initial_opening = self.env._gripper_target if opening is not None else None
        dual_offset = self.config.get("dual_arm_tcp_offset_m")
        if dual_offset is not None:
            if len(dual_offset) != 3 or not all(math.isfinite(v) for v in dual_offset):
                raise ValueError("invalid dual arm offset")
            right_state = self.env.robot.data.body_state_w[0, self.env.right_arm_cfg.body_ids[0]]
            right_q = right_state[3:7].detach().cpu().tolist()
            right_tool = rotate(right_q, self.task["frames"]["robot_tool_offset_m"])
            right_start = [a+b for a, b in zip(right_state[:3].detach().cpu().tolist(), right_tool)]
            previous_right = right_start
            if continuous:
                right_start = [a+b for a, b in zip(start, dual_offset)]
                right_q = list(initial_q)
            right_goal = [a+b for a, b in zip(target, dual_offset)]
            if any(not low <= value <= high for value, (low, high) in zip(right_goal, bounds)):
                raise Interrupted("command_workspace_limit")
            if math.dist(right_start, right_goal)/(ticks*self.env.cfg.sim_dt) > self.config["safety"]["tcp_speed_limit_mps"]:
                raise Interrupted("command_speed_limit")
            if sum(a*b for a, b in zip(right_q, quaternion)) < 0:
                right_q = [-v for v in right_q]
        for index in range(ticks):
            check()
            fraction = (index + 1) / ticks
            if opening is not None:
                # Large target jumps destabilize the source R1 mimic jaws even without contact.
                self.env.set_gripper(initial_opening+(float(opening)-initial_opening)*fraction)
            self.target = [a + (b-a)*fraction for a, b in zip(start, target)]
            q = [a + (b-a)*fraction for a, b in zip(initial_q, quaternion)]
            norm = math.sqrt(sum(v*v for v in q))
            q = [v/norm for v in q]
            self.target_quat = q
            offset = rotate(q, self.task["frames"]["robot_tool_offset_m"])
            link_target = [a-b for a, b in zip(self.target, offset)]
            self.env.set_pose_target("left", self.torch.tensor([link_target], device=self.env.device),
                                     self.torch.tensor([q], device=self.env.device))
            if dual_offset is not None:
                rq = [a+(b-a)*fraction for a, b in zip(right_q, quaternion)]
                rn = math.sqrt(sum(v*v for v in rq))
                rq = [v/rn for v in rq]
                rt = rotate(rq, self.task["frames"]["robot_tool_offset_m"])
                rp = [a+(b-a)*fraction-c for a, b, c in zip(right_start, right_goal, rt)]
                left_joints = self.env._ik_target(self.env._pose_target[0], self.env._pose_target[1], self.env.left_arm_cfg)
                right_joints = self.env._ik_target(self.torch.tensor([rp], device=self.env.device),
                                                  self.torch.tensor([rq], device=self.env.device), self.env.right_arm_cfg)
                self.env.clear_pose_target()
                self.env._joint_target = (left_joints, right_joints)
                self.env.set_dual_gripper_targets(self.env._gripper_target, self.env._gripper_target)
            self.idle_step()
            if dual_offset is not None:
                state = self.env.robot.data.body_state_w[0, self.env.right_arm_cfg.body_ids[0]]
                measured = state.detach().cpu().tolist()
                offset = rotate(measured[3:7], self.task["frames"]["robot_tool_offset_m"])
                actual = [a+b for a, b in zip(measured[:3], offset)]
                expected = [a+(b-a)*fraction for a, b in zip(right_start, right_goal)]
                if any(not low <= value <= high for value, (low, high) in zip(actual, bounds)):
                    raise Interrupted("right_workspace_limit")
                if math.dist(actual, previous_right)/self.env.cfg.sim_dt > self.config["safety"]["tcp_speed_limit_mps"]:
                    raise Interrupted("right_tcp_speed_limit")
                limit = self.config["safety"].get("tracking_limit_m")
                if limit is not None and math.dist(actual, expected) > limit:
                    raise Interrupted("right_tracking_limit")
                previous_right = actual
            check()
            if insertion is not None:
                origin, goal, axis, final_insert = insertion
                xyz, _ = self._tcp()
                self.insertion_elapsed_s += self.env.cfg.sim_dt
                progress = 1000 * sum((b-a)*v for a, b, v in zip(origin, xyz, axis)) if axis else None
                remaining = 1000 * sum((b-a)*v for a, b, v in zip(xyz, goal, axis)) if axis else None
                self.insertion_report = self.insertion_check.update(
                    self.insertion_elapsed_s, progress, remaining, self._force("hub_contact"), self._holding())
                self.insertion_stamp = self.last_robot_stamp
                if self.insertion_report["state"] == "BLOCKED":
                    return True
        if (insertion is not None and final_insert and self.insertion_report["state"] == "IN_PROGRESS" and
                self.insertion_report["reason"] != "tcp_target_reached_not_seating_success"):
            self.insertion_report.update(state="BLOCKED", reason="insertion_profile_ended_before_target")
            return True
        return False

    def help(self, request, check):
        if not self.hold_certified:
            return HelpReport("REJECTED", "none", "Safe hold has not been certified.", 0, 0)
        if (request.target, request.operation, request.allowed_scope) != (
                "socket_hub_output", "clear_target_area", "target_area_only"):
            return HelpReport("REJECTED", "none", "Outside allowed scope.", 0, 0)
        started = time.monotonic()
        check()
        self.env.clear_target_blocker()
        for _ in range(5):
            check()
            self.idle_step()
            check()
        return HelpReport("COMPLETED", request.operation, "Scoped intervention attempted; verify fresh observations.",
                          time.monotonic() - started, 1)

    def _record_private(self):
        from hrc_harness.evaluation import record_isaac_sample
        sample = record_isaac_sample(self.env, self.task)
        self.private_samples.append(sample)
        if self.private_stream is None:
            self.private_stream = (self.output / "physics_private_gt.jsonl").open("w")
            os.chmod(self.private_stream.fileno(), 0o600)
        self.private_stream.write(json.dumps({"physics_tick": self.tick, **sample}) + "\n")
        self.private_stream.flush()

    def evaluate(self):
        from .evaluation import judge
        return {"backend": "isaac", **judge(self.private_samples, self.task), "physics_ticks": self.tick,
                "private_reset": self.private_reset, "retained_dwell_samples": len(self.private_samples)}

    def close(self):
        if self.media:
            self.media.close()
        if self.private_stream:
            self.private_stream.close()


def configure_scene(cfg, task, seed=0, *, blocked=False, cameras=True):
    """Apply task-scoped reset geometry before any runtime physics or commands."""
    scene = task["scene"]
    cfg.seed, cfg.decimation, cfg.episode_length_s = seed, 1, 1_000_000
    cfg.sim.dt = cfg.sim_dt = scene["physics_dt_s"]
    cfg.spawn_cameras = cfg.update_cameras = cameras
    cfg.camera_update_stride = cfg.sim.render_interval = 10
    cfg.spawn_physical_supports = cfg.scatter_reset = True
    cfg.harness_cover_support_offsets = scene.get("cover_support_xy_offsets_m", {})
    cfg.torso_runtime_override = bool(scene.get("fixed_bundle_torso_posture", False))
    if cfg.torso_runtime_override:
        # Reset-only bundle posture; retain zero-width limits, not movable torso DOFs.
        cfg.torso_limit_half_range = 0.0
    cfg.spawn_target_blocker = blocked
    cfg.spawn_grasp_constraint = False
    cfg.hub_cfg.spawn.rigid_props.disable_gravity = False
    cfg.hub_cfg.spawn.rigid_props.kinematic_enabled = False
    cfg.casing_cfg.spawn.rigid_props.kinematic_enabled = scene["casing_kinematic_fixture"]
    cfg.casing_cfg.spawn.rigid_props.disable_gravity = scene["casing_kinematic_fixture"]
    for part in ("cover", "casing"):
        quaternion = scene.get(f"{part}_initial_quat_wxyz")
        if quaternion is not None:
            if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion) or abs(sum(v*v for v in quaternion)-1) > 1e-4:
                raise ValueError(f"invalid {part} reset quaternion")
            getattr(cfg, "hub_cfg" if part == "cover" else "casing_cfg").init_state.rot = tuple(quaternion)
    cfg.hub_reset_pos = (*scene["cover_initial_xy"], cfg.hub_reset_pos[2])
    cfg.casing_reset_pos = (*scene["casing_initial_xy"], cfg.casing_reset_pos[2])
    cfg.table_top_z = scene["table_top_z_m"]
    cfg.table_center_xy = tuple(scene.get("table_center_xy", cfg.table_center_xy))
    cfg.table_cfg.init_state.pos = (*cfg.table_center_xy, cfg.table_top_z - cfg.table_cfg.spawn.size[2]/2)
    cfg.scatter_support_top_z = scene["support_top_z_m"]
    cfg.scatter_hub_support_top_z = scene.get("cover_support_top_z_m", cfg.scatter_support_top_z)
    for name in ("hub_support_n_cfg", "hub_support_s_cfg", "hub_support_e_cfg", "hub_support_w_cfg", "casing_support_cfg"):
        support = getattr(cfg, name)
        top = cfg.scatter_support_top_z if name == "casing_support_cfg" else cfg.scatter_hub_support_top_z
        height = top + cfg.scatter_spawn_margin_m - cfg.table_top_z
        if height <= 0:
            raise ValueError("support surface must be above the table")
        support.spawn.size = (*support.spawn.size[:2], height)
    if scene.get("robot_contact_offset_m") is not None:
        cfg.robot_cfg.spawn.collision_props.contact_offset = scene["robot_contact_offset_m"]
        cfg.robot_cfg.spawn.collision_props.rest_offset = 0.0
    # The slave jaws mimic axis1; opening only the master at reset violates that constraint.
    for side in ("left", "right"):
        cfg.robot_cfg.init_state.joint_pos[f"{side}_gripper_axis2"] = cfg.robot_cfg.init_state.joint_pos[f"{side}_gripper_axis1"]
    return cfg


def reset_scene(env, seed):
    # The source custom reset omits unactuated mimic DOFs; include both slave jaws.
    slaves, _ = env.robot.find_joints(".*_gripper_axis2")
    env._reset_joint_ids = sorted(set(env._reset_joint_ids + slaves))
    env.prepare_scatter_reset()
    offsets = env.cfg.harness_cover_support_offsets
    if offsets:
        report = env.scatter_reset_report()
        for side, xy in offsets.items():
            support = getattr(env, f"hub_support_{side}")
            state = support.data.default_root_state.clone()
            state[:, 0] = env.cfg.hub_reset_pos[0] + xy[0]
            state[:, 1] = env.cfg.hub_reset_pos[1] + xy[1]
            support.data.default_root_state[:] = state
            support.write_root_state_to_sim(state)
            for row in report["supports"]:
                if row["prim_path"] == str(support.cfg.prim_path):
                    row["root_position_m"][:2] = state[0, :2].detach().cpu().tolist()
        env._scatter_reset_report = report
    env.reset(seed=seed)


def main():
    from isaaclab.app import AppLauncher
    from .run import METHODS, load_config, run_episode
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/harness_isaac.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="proposed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blocked", action="store_true", help="Private reset condition, never part of online context")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Sensors/gravity/support smoke only; no model or manipulation")
    parser.add_argument("--branch-plan", type=Path, help="Offline script; record_calibration=false selects infrastructure smoke")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    task = load_task(config["task_config"])
    config["task_snapshot"] = task
    if args.smoke_steps:
        if args.smoke_steps < 1:
            parser.error("smoke steps must be positive")
        config["model"] = None
        config["tools"] = [item for item in config["tools"] if item["tool"] in {"observe", "stop"}]
        config["max_decisions"] = 1
        config["startup_observe_ticks"] = args.smoke_steps
    if config["backend"] != "isaac":
        parser.error("physical runner requires isaac configuration")
    app = AppLauncher(args).app
    env, adapter = None, None
    try:
        from hrc_m1.roco_env import make_env_classes
        cfg_cls, env_cls = make_env_classes()
        cfg = configure_scene(cfg_cls(), task, args.seed, blocked=args.blocked)
        env = env_cls(cfg)
        reset_scene(env, args.seed)
        adapter = IsaacSceneAdapter(env, config, task, args.output)
        adapter.private_reset = {"condition": "blocked" if args.blocked else "nominal",
                                 "layout": env.scatter_reset_report()}
        plan = json.loads(args.branch_plan.read_text()) if args.branch_plan else None
        print(run_episode(adapter, config, args.output, method=args.method, seed=args.seed, branch_plan=plan))
    finally:
        if adapter:
            adapter.close()
        if env:
            env.close()
        app.close()


def load_task(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


if __name__ == "__main__":
    main()
