#!/usr/bin/env python3
"""Owned development-only Isaac loading/physics smoke for bolt insertion."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _write_private_native_physics_trace(env, evidence_root: Path) -> Path:
    trace_reader = getattr(env, "finalize_private_physics_trace", None)
    if not callable(trace_reader):
        raise RuntimeError("environment does not expose its private native physics trace")
    trace = trace_reader()
    private_dir = evidence_root / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    path = private_dir / "native_physics_trace.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(trace, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    return path


def _write_private_condition_manifest(
    run_dir: Path,
    *,
    condition: str,
    selection_source: str,
    collider_evidence: dict[str, object],
) -> Path:
    if selection_source not in {"command_line", "config", "default"}:
        raise ValueError("condition selection source must be command_line, config, or default")
    manifest = {
        "condition": condition,
        "selection_source": selection_source,
        "input_flag": "--condition" if selection_source == "command_line" else None,
        "condition_collider_evidence": collider_evidence,
    }
    path = run_dir / "private_condition_manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    return path


def _record_zero_control_sample(summary: dict[str, object], stage: str, env) -> None:
    control = env.get_factory_control_snapshot()
    if control is None:
        return
    control["stage"] = stage
    control["post_step_sim_timestamp_s"] = float(env._robot._data._sim_timestamp)
    control["post_step_joint7_position_rad"] = float(env._robot.data.joint_pos[0, 6].item())
    control["post_step_joint7_velocity_rad_s"] = float(env._robot.data.joint_vel[0, 6].item())
    control["post_step_tcp_angular_velocity_world_rad_s"] = (
        env.fingertip_midpoint_angvel[0].detach().cpu().tolist()
    )
    trace = summary.setdefault("zero_command_control_trace", [])
    if isinstance(trace, list):
        trace.append(control)


def _record_task_rgb_camera_frame(env, trace: list[dict[str, float]]) -> None:
    camera = env.scene.sensors.get("task_rgb_camera")
    if camera is None:
        raise RuntimeError("task_rgb_camera is missing from env.scene.sensors")
    _ = camera.data
    frame_index = int(camera.frame[0].item())
    if trace and frame_index == int(trace[-1]["frame_index"]):
        return
    trace.append(
        {
            "frame_index": frame_index,
            "sim_timestamp_s": float(env._robot._data._sim_timestamp),
        }
    )


def _capture_task_rgb_keyframe(env, path: Path, *, refresh: bool) -> dict[str, object]:
    import imageio.v3 as imageio
    import numpy as np

    camera = env.scene.sensors.get("task_rgb_camera")
    if camera is None:
        raise RuntimeError("task_rgb_camera is missing from env.scene.sensors")
    if refresh:
        env.sim.render()
        camera.update(0.0, force_recompute=True)
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        raise RuntimeError("task_rgb_camera has no rendered output['rgb'] frame")
    if rgb.ndim == 4:
        if rgb.shape[0] != 1:
            raise RuntimeError(f"task_rgb_camera expected one environment, got {tuple(rgb.shape)}")
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise RuntimeError(f"task_rgb_camera returned unexpected image shape {tuple(rgb.shape)}")
    frame = rgb[..., :3].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.size and float(np.nanmax(frame)) <= 1.0:
            frame = np.rint(frame * 255.0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)
    return {
        "path": str(path),
        "frame_index": int(camera.frame[0].item()),
        "sim_timestamp_s": float(env._robot._data._sim_timestamp),
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "source": "Isaac Sim task_rgb_camera output['rgb']",
    }


def _task_rgb_camera_diagnostics(env, cfg, *, app_enabled: bool) -> dict[str, object]:
    import torch

    camera = env.scene.sensors.get("task_rgb_camera")
    if camera is None:
        raise RuntimeError("task_rgb_camera is missing from env.scene.sensors")
    output = camera.data.output
    rgb = output.get("rgb") if isinstance(output, dict) else None
    if rgb is None:
        raise RuntimeError("task_rgb_camera has no rendered output['rgb'] frame")
    shape = tuple(int(value) for value in rgb.shape)
    if rgb.ndim == 4:
        if rgb.shape[0] != 1:
            raise RuntimeError(f"task_rgb_camera expected one environment, got {shape}")
        rgb = rgb[0]
    if rgb.ndim != 3 or tuple(rgb.shape[:2]) != (cfg.task_rgb_camera.height, cfg.task_rgb_camera.width):
        raise RuntimeError(f"task_rgb_camera returned unexpected image shape {shape}")
    if rgb.shape[-1] not in (3, 4):
        raise RuntimeError(f"task_rgb_camera expected RGB/RGBA channels, got {shape}")
    if rgb.is_floating_point() and not bool(torch.isfinite(rgb).all()):
        raise RuntimeError("task_rgb_camera returned non-finite RGB values")
    frame_trace = env._task_rgb_camera_frame_trace
    observed_frames = [item for item in frame_trace if int(item["frame_index"]) > 0]
    fresh_frames = [int(item["frame_index"]) for item in observed_frames]
    if len(fresh_frames) < 2 or any(
        current != previous + 1 for previous, current in zip(fresh_frames, fresh_frames[1:])
    ):
        raise RuntimeError(f"task_rgb_camera did not produce sequential fresh frames: {fresh_frames[-8:]}")
    return {
        "prim_path": cfg.task_rgb_camera.prim_path,
        "update_period_s": cfg.task_rgb_camera.update_period,
        "offset_pos_m": list(cfg.task_rgb_camera.offset.pos),
        "offset_quat_wxyz": list(cfg.task_rgb_camera.offset.rot),
        "focal_length_mm": cfg.task_rgb_camera.spawn.focal_length,
        "sim_render_interval_steps": cfg.sim.render_interval,
        "app_enable_cameras": bool(app_enabled),
        "shape": shape,
        "dtype": str(rgb.dtype),
        "minimum": float(rgb.min().item()),
        "maximum": float(rgb.max().item()),
        "mean": float(rgb.float().mean().item()),
        "sim_timestamp_s": float(env._robot._data._sim_timestamp),
        "frame_count_observed": len(fresh_frames),
        "first_frame_index": fresh_frames[0],
        "last_frame_index": fresh_frames[-1],
        "frame_timestamp_delta_s_sample": [
            round(float(current["sim_timestamp_s"] - previous["sim_timestamp_s"]), 9)
            for previous, current in list(zip(observed_frames, observed_frames[1:]))[:8]
        ],
        "source": "Isaac Sim camera output['rgb']",
    }


def _record_private_bolt_contacts(env, source: dict[str, object], summary: dict[str, object], stage: str) -> None:
    from runtime.bolt_harness.measurement import read_filtered_contact_reports

    def as_list(value):
        if callable(getattr(value, "detach", None)):
            value = value.detach()
        if callable(getattr(value, "cpu", None)):
            value = value.cpu()
        if callable(getattr(value, "tolist", None)):
            value = value.tolist()
        return value

    def json_safe(value):
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        return value

    def raw_buffer_sample(view, dt_s: float, filter_index: int) -> dict[str, object]:
        forces, points, normals, separations, counts, starts = view.get_contact_data(dt_s)
        count_rows = as_list(counts)
        start_rows = as_list(starts)
        count = int(count_rows[0][filter_index])
        start = int(start_rows[0][filter_index])
        raw_contacts = []
        for index in range(start, start + min(count, 8)):
            raw_contacts.append(
                {
                    "buffer_index": index,
                    "force": json_safe(as_list(forces[index])),
                    "point_w": json_safe(as_list(points[index])),
                    "normal_w": json_safe(as_list(normals[index])),
                    "separation": json_safe(as_list(separations[index])),
                }
            )
        return {"count": count, "start": start, "raw_contacts_first_8": raw_contacts}

    view = source["contact_physx_view"]
    filters = source["filter_map"]
    dt_s = float(source["dt_s"])
    capacity = int(source["contact_data_capacity"])
    contacts = []
    counts = []
    invalid_reports = []
    for item in filters:
        filter_index = int(item["filter_index"])
        try:
            reports = read_filtered_contact_reports(
                view,
                dt_s=dt_s,
                sensor_index=0,
                filter_index=filter_index,
            )
        except ValueError as exc:
            # This private observer is diagnostic-only; preserve bad raw samples without interrupting control.
            try:
                raw = raw_buffer_sample(view, dt_s, filter_index)
            except Exception as raw_exc:
                raw = {"capture_error": f"{type(raw_exc).__name__}: {raw_exc}"}
            invalid_reports.append(
                {
                    "filter_index": filter_index,
                    "filter_prim_path": item["filter_prim_path"],
                    "category": item["category"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_buffer": raw,
                }
            )
            counts.append(None)
            continue
        counts.append(len(reports))
        if reports:
            contacts.extend(
                {
                    "filter_index": int(item["filter_index"]),
                    "filter_prim_path": item["filter_prim_path"],
                    "category": item["category"],
                    "force_n": report.force_n,
                    "force_vector_w_n": list(report.force_vector_w_n),
                    "force_magnitude_n": report.force_magnitude_n,
                    "point_w_m": list(report.point_w_m),
                    "normal_w": list(report.normal_w),
                    "separation_m": report.separation_m,
                }
                for report in reports
            )
    total_count = sum(count for count in counts if count is not None)
    total_count_known = not invalid_reports
    capacity_reached = total_count >= capacity or not total_count_known
    state = summary.setdefault(
        "private_bolt_contact_buffer",
        {
            "configured_capacity": capacity,
            "max_contact_data_count_per_prim": int(source["max_contact_data_count_per_prim"]),
            "sensor_body_names": list(source["sensor_body_names"]),
            "sample_count": 0,
            "peak_contact_points_in_sample": 0,
            "capacity_reached_or_truncation_possible": False,
            "raw_report_validation_error_count": 0,
            "raw_report_validation_errors": [],
        },
    )
    state["sample_count"] += 1
    state["peak_contact_points_in_sample"] = max(state["peak_contact_points_in_sample"], total_count)
    state["capacity_reached_or_truncation_possible"] |= capacity_reached
    state["raw_report_validation_error_count"] += len(invalid_reports)
    if invalid_reports and len(state["raw_report_validation_errors"]) < 32:
        state["raw_report_validation_errors"].extend(invalid_reports[: 32 - len(state["raw_report_validation_errors"])])
    trace = summary.setdefault("private_bolt_contact_samples", [])
    trace.append(
        {
            "sim_timestamp_s": float(env._robot._data._sim_timestamp),
            "stage": stage,
            "counts_by_filter": counts,
            "total_contact_points": total_count,
            "total_contact_points_known": total_count_known,
            "capacity_reached_or_truncation_possible": capacity_reached,
            "invalid_reports": invalid_reports,
            "contacts": contacts,
        }
    )
    if invalid_reports:
        first = invalid_reports[0]
        raise RuntimeError(
            "private bolt contact report validation failed; raw signed buffer sample was saved for "
            f"{first['filter_prim_path']}"
        )


def _record_robot_fixture_contacts(env, source: dict[str, object], summary: dict[str, object]) -> None:
    import torch

    diagnostics = summary.setdefault(
        "robot_fixture_contact_diagnostics",
        {
            "filter_map": source["filter_map"],
            "sample_count": 0,
            "reporting_threshold_n": 1e-4,
            "peak_force_n_by_body_category": {},
            "peak_sim_time_s_by_body_category": {},
            "active_sample_count_by_body_category": {},
            "last_active_sample_by_body_category": {},
            "sensor_errors": [],
        },
    )
    diagnostics["sample_count"] += 1
    timestamp = float(env._robot._data._sim_timestamp)
    for body_name, sensor in source["sensors"].items():
        matrix = sensor.data.force_matrix_w
        if matrix is None or matrix.ndim != 4 or matrix.shape[0] != 1 or matrix.shape[1] != 1:
            message = f"{body_name}: unexpected force_matrix_w shape {None if matrix is None else tuple(matrix.shape)}"
            if message not in diagnostics["sensor_errors"]:
                diagnostics["sensor_errors"].append(message)
            continue
        if matrix.shape[2] != len(source["filter_map"]):
            message = f"{body_name}: matrix has {matrix.shape[2]} filters, expected {len(source['filter_map'])}"
            if message not in diagnostics["sensor_errors"]:
                diagnostics["sensor_errors"].append(message)
            continue
        forces_n = torch.linalg.vector_norm(matrix[0, 0], dim=-1).detach().cpu().tolist()
        category_forces: dict[str, float] = {}
        for item, force_n in zip(source["filter_map"], forces_n):
            category = str(item["category"])
            category_forces[category] = max(category_forces.get(category, 0.0), float(force_n))
        for category, force_n in category_forces.items():
            key = f"{body_name}/{category}"
            if force_n > diagnostics["peak_force_n_by_body_category"].get(key, 0.0):
                diagnostics["peak_force_n_by_body_category"][key] = force_n
                diagnostics["peak_sim_time_s_by_body_category"][key] = timestamp
            if force_n > diagnostics["reporting_threshold_n"]:
                counts = diagnostics["active_sample_count_by_body_category"]
                counts[key] = counts.get(key, 0) + 1
                diagnostics["last_active_sample_by_body_category"][key] = {
                    "sim_timestamp_s": timestamp,
                    "force_n": force_n,
                }


def _record_tcp_kinematic_diagnostics(env, summary: dict[str, object]) -> dict[str, float]:
    import torch

    data = env._robot.data
    hand_idx = env.wrench_body_idx
    qdot = env.joint_vel[:, :7]
    controller_jacobian_twist = torch.bmm(
        env.fingertip_midpoint_jacobian, qdot.unsqueeze(-1)
    ).squeeze(-1)
    raw_jacobians = env._robot.root_physx_view.get_jacobians()
    raw_hand_jacobian = raw_jacobians[:, hand_idx - 1, :6, :7]
    tcp_world = env.fingertip_midpoint_pos + env.scene.env_origins
    tcp_from_com_world = tcp_world - data.body_com_pos_w[:, hand_idx]
    angular_axes = raw_hand_jacobian[:, 3:6, :].transpose(1, 2)
    com_to_tcp_shift = torch.cross(angular_axes, tcp_from_com_world[:, None, :], dim=-1).transpose(1, 2)
    com_adjusted_jacobian = torch.cat(
        (raw_hand_jacobian[:, :3, :] + com_to_tcp_shift, raw_hand_jacobian[:, 3:6, :]), dim=1
    )
    com_adjusted_jacobian_twist = torch.bmm(com_adjusted_jacobian, qdot.unsqueeze(-1)).squeeze(-1)
    fd_twist = torch.cat((env.ee_linvel_fd, env.ee_angvel_fd), dim=-1)
    reported_twist = torch.cat((env.fingertip_midpoint_linvel, env.fingertip_midpoint_angvel), dim=-1)
    body_com_velocity = data.body_com_vel_w[:, hand_idx]
    tcp_from_link_world = tcp_world - data.body_link_pos_w[:, hand_idx]
    legacy_com_tcp_linear = body_com_velocity[:, :3] + torch.cross(
        body_com_velocity[:, 3:6], tcp_from_link_world, dim=-1
    )

    error_values = {
        "reported_link_tcp_velocity_vs_fd_linear_mps": torch.linalg.vector_norm(
            reported_twist[:, :3] - fd_twist[:, :3], dim=-1
        ),
        "legacy_com_tcp_velocity_vs_fd_linear_mps": torch.linalg.vector_norm(
            legacy_com_tcp_linear - fd_twist[:, :3], dim=-1
        ),
        "current_controller_jacobian_vs_fd_linear_mps": torch.linalg.vector_norm(
            controller_jacobian_twist[:, :3] - fd_twist[:, :3], dim=-1
        ),
        "com_to_tcp_jacobian_candidate_vs_fd_linear_mps": torch.linalg.vector_norm(
            com_adjusted_jacobian_twist[:, :3] - fd_twist[:, :3], dim=-1
        ),
    }
    metrics = {name: float(value[0].item()) for name, value in error_values.items()}
    q = env.joint_pos[0]
    qdot_single = env.joint_vel[0]
    torque = env.joint_torque[0]
    effort_limits = data.joint_effort_limits[0]
    pos_limits = data.joint_pos_limits[0]
    limit_margins = torch.minimum(q - pos_limits[:, 0], pos_limits[:, 1] - q)
    torque_utilization = torque.abs() / effort_limits.clamp_min(1e-9)
    snapshot = {
        "sim_timestamp_s": float(env._robot._data._sim_timestamp),
        "finite_difference_tcp_twist": fd_twist[0].detach().cpu().tolist(),
        "reported_link_frame_tcp_twist": reported_twist[0].detach().cpu().tolist(),
        "legacy_com_based_tcp_linear_velocity": legacy_com_tcp_linear[0].detach().cpu().tolist(),
        "current_controller_jacobian_times_qdot": controller_jacobian_twist[0].detach().cpu().tolist(),
        "com_to_tcp_shifted_jacobian_candidate_times_qdot": com_adjusted_jacobian_twist[0]
        .detach()
        .cpu()
        .tolist(),
        "panda_hand_com_offset_local_m": data.body_com_pos_b[0, hand_idx].detach().cpu().tolist(),
        "joint_positions_rad": q.detach().cpu().tolist(),
        "joint_position_limits_rad": pos_limits.detach().cpu().tolist(),
        "joint_position_limit_margins_rad": limit_margins.detach().cpu().tolist(),
        "joint_velocities_rad_s": qdot_single.detach().cpu().tolist(),
        "joint_torque_Nm": torque.detach().cpu().tolist(),
        "joint_effort_limits_Nm": effort_limits.detach().cpu().tolist(),
        "joint_effort_utilization": torque_utilization.detach().cpu().tolist(),
        "errors": metrics,
    }
    diagnostics = summary.setdefault(
        "tcp_kinematic_diagnostics",
        {"sample_count": 0, "max_errors": {}, "sample_at_max_error": {}, "latest_sample": None},
    )
    diagnostics["sample_count"] += 1
    for name, value in metrics.items():
        if value > diagnostics["max_errors"].get(name, -1.0):
            diagnostics["max_errors"][name] = value
            diagnostics["sample_at_max_error"][name] = snapshot
    diagnostics["latest_sample"] = snapshot
    return metrics


def _install_executor_trace(
    env,
    summary: dict[str, object],
    private_contact_source: dict[str, object],
    robot_fixture_contact_source: dict[str, object],
    keyframe_dir: Path,
):
    trace: list[dict[str, object]] = []
    gripper_commands: list[dict[str, object]] = []
    skill = {"name": "unassigned"}
    summary["executor_step_trace"] = trace
    summary["gripper_command_trace"] = gripper_commands
    private_state = summary.setdefault(
        "simulator_private_ground_truth", {"not_passed_to_agent_or_model": True}
    )
    private_control_trace = private_state.setdefault("pick_control_trace", [])
    original_step = env.step
    original_setter = env.set_gripper_target_width
    active_gripper_command: dict[str, float | None] = {
        "target_width_m": 0.08,
        "total_effort_cap_n": None,
    }
    trajectory_keyframes = summary.setdefault("trajectory_keyframes", {})
    trajectory_events: set[str] = set()

    def save_trajectory_keyframe(event: str, skill_name: str, bolt_pos, bolt_quat) -> None:
        if event in trajectory_events:
            return
        keyframe = _capture_task_rgb_keyframe(
            env, keyframe_dir / f"{event}.png", refresh=True
        )
        _record_task_rgb_camera_frame(env, env._task_rgb_camera_frame_trace)
        keyframe.update({"event": event, "skill": skill_name})
        trajectory_keyframes[event] = keyframe
        private_state = summary.setdefault(
            "simulator_private_ground_truth", {"not_passed_to_agent_or_model": True}
        )
        private_state.setdefault("trajectory_keyframe_state", {})[event] = {
            "sim_time_s": float(env._robot._data._sim_timestamp),
            "bolt_root_pos_m": list(bolt_pos),
            "bolt_root_quat_wxyz": list(bolt_quat),
        }
        trajectory_events.add(event)

    def traced_setter(width_m: float, max_force_n: float) -> None:
        gripper_commands.append(
            {
                "sim_timestamp_s": float(env._robot._data._sim_timestamp),
                "target_width_m": float(width_m),
                "total_effort_cap_n": float(max_force_n),
                "per_finger_effort_cap_n": float(max_force_n) / 2.0,
            }
        )
        original_setter(width_m, max_force_n)
        active_gripper_command["target_width_m"] = float(width_m)
        active_gripper_command["total_effort_cap_n"] = float(max_force_n)

    def traced_step(action):
        result = original_step(action)
        _record_task_rgb_camera_frame(env, env._task_rgb_camera_frame_trace)
        _record_private_bolt_contacts(env, private_contact_source, summary, skill["name"])
        _record_robot_fixture_contacts(env, robot_fixture_contact_source, summary)
        kinematic_errors = _record_tcp_kinematic_diagnostics(env, summary)
        state = env.get_public_executor_state()
        contacts = env.get_bilateral_grasp_contact()
        bolt_pos = env._bolt.data.root_pos_w[0].detach().cpu().tolist()
        bolt_quat = env._bolt.data.root_quat_w[0].detach().cpu().tolist()
        if skill["name"] == "pick":
            if bolt_pos[2] >= env.cfg.bolt_reset_pos_m[2] + 0.01:
                save_trajectory_keyframe("pick_lift_start", skill["name"], bolt_pos, bolt_quat)
            if bolt_pos[2] >= 0.85:
                save_trajectory_keyframe("pick_lift_clear", skill["name"], bolt_pos, bolt_quat)
        elif skill["name"] == "transport":
            pickup_xy = env.cfg.bolt_reset_pos_m[:2]
            socket_xy = env.cfg.bolt_seat_root_pos_m[:2]
            moved_xy = math.hypot(bolt_pos[0] - pickup_xy[0], bolt_pos[1] - pickup_xy[1])
            socket_xy_error = math.hypot(bolt_pos[0] - socket_xy[0], bolt_pos[1] - socket_xy[1])
            if moved_xy >= 0.05:
                save_trajectory_keyframe("transport_departure", skill["name"], bolt_pos, bolt_quat)
            if socket_xy_error <= 0.02:
                save_trajectory_keyframe("transport_over_socket", skill["name"], bolt_pos, bolt_quat)
        finger_force_matrix_w_n = {}
        for side, sensor in env._finger_bolt_sensors.items():
            matrix = sensor.data.force_matrix_w
            if matrix is None or matrix.ndim != 4 or matrix.shape[0] != 1 or matrix.shape[2] < 1:
                raise RuntimeError(f"{side} finger force_matrix_w is unavailable for independent contact comparison")
            finger_force_matrix_w_n[side] = matrix[0, 0, 0].detach().cpu().tolist()
        private_report_sums_w_n = {side: [0.0, 0.0, 0.0] for side in ("left", "right")}
        private_samples = summary.get("private_bolt_contact_samples", [])
        latest_sample = private_samples[-1] if private_samples else {}
        for report in latest_sample.get("contacts", []):
            path = str(report.get("filter_prim_path", ""))
            side = "left" if path.endswith("/panda_leftfinger") else (
                "right" if path.endswith("/panda_rightfinger") else None
            )
            if side is not None:
                vector = report["force_vector_w_n"]
                private_report_sums_w_n[side] = [
                    a + float(b) for a, b in zip(private_report_sums_w_n[side], vector)
                ]
        pair_force_comparison = {
            side: {
                "force_matrix_w_n": finger_force_matrix_w_n[side],
                "private_contact_report_vector_sum_w_n": private_report_sums_w_n[side],
                "matrix_minus_report_sum_w_n": [
                    float(a) - float(b)
                    for a, b in zip(finger_force_matrix_w_n[side], private_report_sums_w_n[side])
                ],
                "matrix_plus_report_sum_w_n": [
                    float(a) + float(b)
                    for a, b in zip(finger_force_matrix_w_n[side], private_report_sums_w_n[side])
                ],
            }
            for side in ("left", "right")
        }
        bolt_root_lin_vel = getattr(env._bolt.data, "root_lin_vel_w", None)
        bolt_root_ang_vel = getattr(env._bolt.data, "root_ang_vel_w", None)
        private_control_trace.append(
            {
                "sim_timestamp_s": float(env._robot._data._sim_timestamp),
                "skill": skill["name"],
                "bolt_root_pos_m": bolt_pos,
                "bolt_root_quat_wxyz": bolt_quat,
                "bolt_root_lin_vel_w_mps": (
                    bolt_root_lin_vel[0].detach().cpu().tolist()
                    if bolt_root_lin_vel is not None
                    else None
                ),
                "bolt_root_ang_vel_w_radps": (
                    bolt_root_ang_vel[0].detach().cpu().tolist()
                    if bolt_root_ang_vel is not None
                    else None
                ),
                "tcp_pose_w": state["tcp_pose"][0].detach().cpu().tolist(),
                "commanded_tcp_pose_w": state["commanded_tcp_pose"][0].detach().cpu().tolist(),
                "finger_joint_pos_m": env._robot.data.joint_pos[0, env._finger_joint_ids]
                .detach()
                .cpu()
                .tolist(),
                "finger_joint_vel_mps": env._robot.data.joint_vel[0, env._finger_joint_ids]
                .detach()
                .cpu()
                .tolist(),
                "measured_gripper_width_m": float(state["gripper_width_m"]),
                "commanded_gripper_width_m": active_gripper_command["target_width_m"],
                "total_gripper_effort_cap_n": active_gripper_command["total_effort_cap_n"],
                "per_finger_effort_cap_n": (
                    None
                    if active_gripper_command["total_effort_cap_n"] is None
                    else float(active_gripper_command["total_effort_cap_n"]) / 2.0
                ),
                "finger_bolt_contacts": list(state["finger_bolt_contacts"]),
                "finger_contact_force_matrix_w_n": finger_force_matrix_w_n,
                "bolt_filtered_contact_force_vectors_w_n": private_report_sums_w_n,
            }
        )
        trace.append(
            {
                "skill": skill["name"],
                "sim_timestamp_s": float(env._robot._data._sim_timestamp),
                "action_delta": action.detach().reshape(-1).cpu().tolist(),
                "tcp_pose": state["tcp_pose"][0].detach().cpu().tolist(),
                "commanded_tcp_pose": state["commanded_tcp_pose"][0].detach().cpu().tolist(),
                "tcp_velocity": state["tcp_velocity"][0].detach().cpu().tolist(),
                "joint_velocities": state["joint_velocities"][0].detach().cpu().tolist(),
                "gripper_width_m": float(state["gripper_width_m"]),
                "finger_bolt_contacts": list(state["finger_bolt_contacts"]),
                "finger_bolt_contact_forces_n": {
                    "left": float(contacts["left_force_n"][0].item()),
                    "right": float(contacts["right_force_n"][0].item()),
                },
                "finger_bolt_contact_force_matrix_w_n": finger_force_matrix_w_n,
                "private_contact_force_vector_comparison_w_n": pair_force_comparison,
                "wrench_assembly_N_Nm": env.public_wrench()[0].detach().cpu().tolist(),
                "tcp_kinematic_error_metrics": kinematic_errors,
            }
        )
        return result

    env.step = traced_step
    env.set_gripper_target_width = traced_setter
    return skill, trace, gripper_commands


def _measure_pick_retention(
    trace: list[dict[str, object]],
    *,
    initial_bolt_root_z_m: float,
    planned_lift_m: float,
    cap_pad_overlap_m: float,
    cap_width_m: float,
    width_contact_allowance_m: float,
    measured_width_m: float,
    final_bilateral_contact: bool,
) -> dict[str, object]:
    held = [sample for sample in trace if all(sample.get("finger_bolt_contacts", (False, False)))]
    if not held:
        return {
            "passed": False,
            "reason": "no bilateral contact samples were measured",
            "bilateral_contact_samples": 0,
            "final_bilateral_contact": final_bilateral_contact,
            "measured_gripper_width_m": measured_width_m,
            "cap_width_m": cap_width_m,
        }

    first = held[0]
    first_root = first["bolt_root_pos_m"]
    first_tcp = first["tcp_pose_w"][:3]
    baseline_relative = [float(first_root[i]) - float(first_tcp[i]) for i in range(3)]
    relative_drifts = []
    for sample in held:
        root = sample["bolt_root_pos_m"]
        tcp = sample["tcp_pose_w"][:3]
        relative = [float(root[i]) - float(tcp[i]) for i in range(3)]
        relative_drifts.append(
            [relative[i] - baseline_relative[i] for i in range(3)]
        )
    max_relative_slip_m = max(
        math.sqrt(sum(component * component for component in drift))
        for drift in relative_drifts
    )
    max_lateral_slip_m = max(
        math.hypot(drift[0], drift[1]) for drift in relative_drifts
    )
    last_root = trace[-1]["bolt_root_pos_m"] if trace else first_root
    measured_lift_m = float(last_root[2]) - float(initial_bolt_root_z_m)
    minimum_lift_m = max(0.0, float(planned_lift_m) - float(cap_pad_overlap_m))
    width_limit_m = float(cap_width_m) + float(width_contact_allowance_m)
    width_within_cap = 0.0 <= float(measured_width_m) <= width_limit_m
    slip_within_pad = max_relative_slip_m <= float(cap_pad_overlap_m)
    lift_retained = measured_lift_m >= minimum_lift_m
    passed = bool(
        final_bilateral_contact
        and width_within_cap
        and slip_within_pad
        and lift_retained
    )
    return {
        "passed": passed,
        "bilateral_contact_samples": len(held),
        "final_bilateral_contact": final_bilateral_contact,
        "initial_bilateral_bolt_tcp_offset_w_m": baseline_relative,
        "max_bolt_tcp_relative_slip_m": max_relative_slip_m,
        "max_lateral_bolt_tcp_slip_m": max_lateral_slip_m,
        "measured_bolt_lift_m": measured_lift_m,
        "planned_tcp_lift_m": float(planned_lift_m),
        "minimum_bolt_lift_after_cap_overlap_m": minimum_lift_m,
        "cap_pad_overlap_allowance_m": float(cap_pad_overlap_m),
        "measured_gripper_width_m": float(measured_width_m),
        "cap_width_m": float(cap_width_m),
        "width_contact_allowance_m": float(width_contact_allowance_m),
        "maximum_gripper_width_from_cap_m": width_limit_m,
        "width_within_physical_cap": width_within_cap,
        "relative_slip_within_cap_pad_overlap": slip_within_pad,
        "bolt_lift_retained": lift_retained,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config/bolt_insertion.yaml"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stage", choices=("hold", "pick", "episode"), default="hold")
    parser.add_argument(
        "--condition",
        choices=("S0", "S2"),
        help="per-run private scene condition override; defaults to the config condition (S0)",
    )
    parser.add_argument("--agent-mode", choices=("vlm", "scripted"))
    parser.add_argument(
        "--model-endpoint",
        help="per-run VLM endpoint override; requires --model-name and --stage episode --agent-mode vlm",
    )
    parser.add_argument(
        "--model-name",
        help="per-run VLM model identifier override; requires --model-endpoint and --stage episode --agent-mode vlm",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir")
    parser.add_argument("--_simulation-child", action="store_true", help=argparse.SUPPRESS)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0", enable_cameras=True)
    args = parser.parse_args()
    if args.stage != "hold" and args.steps is not None:
        parser.error("--steps is only applicable to --stage hold")
    if args.stage == "episode" and args.agent_mode is None:
        parser.error("--stage episode requires --agent-mode vlm|scripted")
    if args.stage != "episode" and args.agent_mode is not None:
        parser.error("--agent-mode is only applicable to --stage episode")
    has_model_endpoint = args.model_endpoint is not None
    has_model_name = args.model_name is not None
    if has_model_endpoint != has_model_name:
        parser.error("--model-endpoint and --model-name must be provided together")
    if has_model_endpoint and (args.stage != "episode" or args.agent_mode != "vlm"):
        parser.error("model overrides are only applicable to --stage episode --agent-mode vlm")
    if not args._simulation_child:
        if args.output_dir:
            run_dir = Path(args.output_dir).expanduser()
            if not run_dir.is_absolute():
                run_dir = REPO_ROOT / run_dir
        else:
            run_dir = REPO_ROOT / "artifacts/bolt_insertion/development" / (
                f"loading_smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
        child = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--output-dir",
                str(run_dir),
                "--_simulation-child",
            ],
            check=False,
        )
        summary_path = run_dir / "smoke_summary.json"
        if summary_path.is_file():
            try:
                child_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                return (
                    0
                    if child_summary.get("run_passed") is True and child.returncode == 0
                    else 2
                )
            except (OSError, json.JSONDecodeError):
                return 2
        return child.returncode if child.returncode != 0 else 2
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    run_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(
        "artifacts/bolt_insertion/development"
    ) / f"loading_smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, object] = {
        "run_kind": f"development_{args.stage}_physics_smoke",
        "stage": args.stage,
        "task_success": None,
        "final_success_claimed": False,
        "steps_requested": args.steps,
        "device": args.device,
        "config_path": str(config_path.resolve()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    launcher = None
    simulation_app = None
    env = None
    private_physics_trace_path: Path | None = None
    try:
        launcher = AppLauncher(args)
        simulation_app = launcher.app
        import torch
        import isaacsim.core.utils.torch as torch_utils
        import yaml

        from runtime.bolt_harness.env import BoltInsertionEnv
        from runtime.bolt_harness.executor import BoltSkillExecutor, make_nominal_bolt_motion_plan
        from runtime.bolt_harness.env_cfg import BOLT_USD, CASING_USD, COVER_USD, PANDA_USD, make_env_cfg

        settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if has_model_endpoint:
            if not isinstance(settings, dict):
                raise ValueError("run config must be a mapping to apply per-run model overrides")
            settings["model"] = {"endpoint": args.model_endpoint, "model": args.model_name}
        condition = args.condition if args.condition is not None else settings.get("condition", "S0")
        condition_source = "command_line" if args.condition is not None else (
            "config" if isinstance(settings, dict) and "condition" in settings else "default"
        )
        pose_window_tolerances = settings.get("pose_window_tolerances") if isinstance(settings, dict) else None
        if args.stage == "episode":
            expected_pose_fields = {
                "max_position_excursion_m",
                "max_orientation_excursion_rad",
            }
            if not isinstance(pose_window_tolerances, dict) or set(pose_window_tolerances) != expected_pose_fields:
                raise ValueError(
                    "episode mode requires explicit pose_window_tolerances with exactly "
                    "max_position_excursion_m and max_orientation_excursion_rad"
                )
            pose_window_tolerances = {
                key: float(value) for key, value in pose_window_tolerances.items()
            }
            if any(not math.isfinite(value) or value < 0.0 for value in pose_window_tolerances.values()):
                raise ValueError("pose-window excursion limits must be finite and non-negative")
        steps = 0 if args.stage in {"pick", "episode"} else (
            args.steps if args.steps is not None else int(settings.get("smoke_steps", 120))
        )
        if args.stage == "hold" and steps < 1:
            raise ValueError("smoke step count must be positive")
        summary["steps_requested"] = steps
        cfg = make_env_cfg(device=args.device, condition=condition)
        if isinstance(settings, dict):
            cfg.pose_window_tolerances = (
                dict(pose_window_tolerances) if args.stage == "episode" else None
            )
            cfg.bolt_mass_kg = float(settings.get("bolt_mass_kg", cfg.bolt_mass_kg))
            cfg.bolt.spawn.mass_props.mass = cfg.bolt_mass_kg
            cfg.wrench_warmup_steps = int(settings.get("wrench_warmup_steps", cfg.wrench_warmup_steps))
            cfg.wrench_stationary_sample_count = int(
                settings.get("wrench_stationary_sample_count", cfg.wrench_stationary_sample_count)
            )
            cfg.task_translation_prop_gain = float(
                settings.get("task_translation_prop_gain", cfg.task_translation_prop_gain)
            )
            cfg.ctrl.default_task_prop_gains[:3] = [cfg.task_translation_prop_gain] * 3
            cfg.task_rot_deriv_scale = float(
                settings.get("task_rot_deriv_scale", cfg.task_rot_deriv_scale)
            )
            cfg.episode_length_s = float(settings.get("episode_length_s", cfg.episode_length_s))
            cfg.tcp_tracking_stop_error_m = float(
                settings.get("tcp_tracking_stop_error_m", cfg.tcp_tracking_stop_error_m)
            )
            cfg.tcp_tracking_stop_orientation_rad = float(
                settings.get("tcp_tracking_stop_orientation_rad", cfg.tcp_tracking_stop_orientation_rad)
            )
            cfg.tcp_linear_speed_stop_mps = float(
                settings.get("tcp_linear_speed_stop_mps", cfg.tcp_linear_speed_stop_mps)
            )
            cfg.tcp_angular_speed_stop_radps = float(
                settings.get("tcp_angular_speed_stop_radps", cfg.tcp_angular_speed_stop_radps)
            )
        env = BoltInsertionEnv(cfg)
        env.reset(seed=args.seed)
        condition_manifest_path = _write_private_condition_manifest(
            run_dir,
            condition=condition,
            selection_source=condition_source,
            collider_evidence=env._private_condition_collider_evidence,
        )
        summary["private_condition_manifest"] = str(condition_manifest_path.relative_to(run_dir))
        env._task_rgb_camera_frame_trace = []
        summary["task_rgb_camera_keyframes"] = {
            "initial_free_bolt": _capture_task_rgb_keyframe(
                env, run_dir / "keyframes" / "initial_free_bolt.png", refresh=True
            )
        }
        _record_task_rgb_camera_frame(env, env._task_rgb_camera_frame_trace)
        summary["task_rgb_camera_frame_trace"] = env._task_rgb_camera_frame_trace
        private_contact_source = env.get_private_bolt_contact_source()
        robot_fixture_contact_source = env.get_private_robot_fixture_contact_source()
        summary["private_bolt_contact_filter_map"] = private_contact_source["filter_map"]
        summary["private_bolt_contact_filter_count"] = len(private_contact_source["filter_map"])
        summary["robot_fixture_contact_filter_map"] = robot_fixture_contact_source["filter_map"]
        summary["panda_joint_physics_usd"] = env._robot_joint_physics_info
        summary["panda_hand_com_offset_local_m"] = (
            env._robot.data.body_com_pos_b[0, env.wrench_body_idx].detach().cpu().tolist()
        )
        summary["effective_control_config"] = {
            "factory_nullspace_reference_rad": list(env.cfg.ctrl.default_dof_pos_tensor),
            "factory_task_prop_gains": env.task_prop_gains[0].detach().cpu().tolist(),
            "factory_task_deriv_gains": env.task_deriv_gains[0].detach().cpu().tolist(),
            "factory_rotation_derivative_scale": cfg.task_rot_deriv_scale,
            "physics_dt_s": float(env.physics_dt),
            "action_control_period_s": float(env.physics_dt * env.cfg.decimation),
            "panda_gravity_disabled": bool(env.cfg.robot.spawn.rigid_props.disable_gravity),
            "panda_composed_rigid_body_gravity": env._robot_gravity_body_flags,
            "tcp_tracking_stop_error_m": cfg.tcp_tracking_stop_error_m,
            "tcp_linear_speed_stop_mps": cfg.tcp_linear_speed_stop_mps,
            "tcp_angular_speed_stop_radps": cfg.tcp_angular_speed_stop_radps,
        }
        arm_mass = env._robot.root_physx_view.get_generalized_mass_matrices()[0, :7, :7]
        jacobian = env.fingertip_midpoint_jacobian[0]
        try:
            operational_inertia = torch.linalg.inv(jacobian @ torch.linalg.solve(arm_mass, jacobian.T))
            operational_inertia_diag = torch.diagonal(operational_inertia).detach().cpu().tolist()
        except RuntimeError:
            operational_inertia_diag = None
        body_inertias = env._robot.root_physx_view.get_inertias()[0].detach().cpu().tolist()
        summary["panda_runtime_inertia"] = {
            "body_names_in_view_order": list(env._robot.body_names),
            "factory_augmented_link_inertias": body_inertias,
            "factory_augmented_arm_generalized_mass_matrix": arm_mass.detach().cpu().tolist(),
            "factory_task_space_effective_inertia_diagonal": operational_inertia_diag,
        }

        start_pos = env._bolt.data.root_pos_w[0].detach().cpu().tolist()
        start_quat = env._bolt.data.root_quat_w[0].detach().cpu().tolist()
        start_bolt_velocity = env._bolt.data.root_vel_w[0].detach().cpu().tolist()
        start_joint_pos = env._robot.data.joint_pos[0].detach().cpu().tolist()
        reset_state = env.get_public_executor_state()
        summary["reset_state"] = {
            "tcp_pose": reset_state["tcp_pose"][0].detach().cpu().tolist(),
            "commanded_tcp_pose": reset_state["commanded_tcp_pose"][0].detach().cpu().tolist(),
            "tcp_velocity": reset_state["tcp_velocity"][0].detach().cpu().tolist(),
            "joint_positions": start_joint_pos,
            "joint_velocities": env._robot.data.joint_vel[0].detach().cpu().tolist(),
            "bolt_root_pose": start_pos + start_quat,
            "bolt_root_velocity": start_bolt_velocity,
            "motion_safety_armed": bool(env.motion_safety_armed),
        }
        warmup_joint_speeds = []
        warmup_tcp_linear_speeds = []
        warmup_tcp_angular_speeds = []
        warmup_tracking_errors = []
        warmup_contact_trace = []
        bolt_support_trace = []
        for _ in range(cfg.wrench_warmup_steps):
            _, _, terminated, truncated, _ = env.step(torch.zeros((1, 6), device=env.device))
            _record_task_rgb_camera_frame(env, env._task_rgb_camera_frame_trace)
            if bool(terminated.any() or truncated.any()):
                raise RuntimeError("environment terminated during wrench warmup")
            _record_private_bolt_contacts(env, private_contact_source, summary, "warmup")
            _record_zero_control_sample(summary, "warmup", env)
            warmup_joint_speeds.append(float(env._robot.data.joint_vel.abs().max().item()))
            state = env.get_public_executor_state()
            warmup_tcp_linear_speeds.append(float(torch.linalg.vector_norm(state["tcp_velocity"][0, :3]).item()))
            warmup_tcp_angular_speeds.append(float(torch.linalg.vector_norm(state["tcp_velocity"][0, 3:]).item()))
            warmup_tracking_errors.append(float(state["tcp_tracking_error_m"][0].item()))
            contacts = env.get_bilateral_grasp_contact()
            warmup_contact_trace.append(
                (float(contacts["left_force_n"][0].item()), float(contacts["right_force_n"][0].item()))
            )
            bolt_support_trace.append(
                {
                    "sim_timestamp_s": float(env._robot._data._sim_timestamp),
                    "root_position_m": env._bolt.data.root_pos_w[0].detach().cpu().tolist(),
                    "root_quaternion_wxyz": env._bolt.data.root_quat_w[0].detach().cpu().tolist(),
                    "root_velocity_mps_radps": env._bolt.data.root_vel_w[0].detach().cpu().tolist(),
                }
            )

        sample_count = cfg.wrench_stationary_sample_count
        tcp_samples = list(env._tcp_position_history)[-sample_count:]
        tcp_drift_m = float("inf")
        if len(tcp_samples) == sample_count:
            tcp_window = torch.stack([sample[0] for sample in tcp_samples])
            tcp_drift_m = float(torch.linalg.vector_norm(tcp_window - tcp_window[0], dim=-1).max().item())
        env_steps_per_window = max(1, (sample_count + cfg.decimation - 1) // cfg.decimation)
        max_joint_speed = max(warmup_joint_speeds[-env_steps_per_window:], default=float("inf"))
        recent_contacts = warmup_contact_trace[-env_steps_per_window:]
        bilateral_contacts_clear = bool(recent_contacts) and all(
            max(force_pair) < cfg.grasp_contact_min_force_n for force_pair in recent_contacts
        )
        stationary = (
            bool(env.motion_safety_armed)
            and tcp_drift_m <= cfg.wrench_stationary_tcp_drift_limit_m
            and max_joint_speed <= cfg.wrench_stationary_joint_speed_limit_rad_s
        )
        baseline_stats = None
        baseline_error = None
        if stationary and bilateral_contacts_clear:
            try:
                baseline_stats = env.set_stationary_wrench_baseline(
                    stationary=True, bilateral_contacts_clear=True
                )
            except RuntimeError as exc:
                baseline_error = str(exc)
        else:
            baseline_error = "warmup failed TCP/joint stability or bilateral no-contact criteria"

        bolt_pre_skill_pos = env._bolt.data.root_pos_w[0].detach().cpu().tolist()
        bolt_pre_skill_quat = env._bolt.data.root_quat_w[0].detach().cpu().tolist()
        bolt_pre_skill_velocity = env._bolt.data.root_vel_w[0].detach().cpu().tolist()
        support_window = bolt_support_trace[-sample_count:]
        support_reference_pos = torch.tensor(support_window[0]["root_position_m"])
        support_reference_quat = torch.tensor(support_window[0]["root_quaternion_wxyz"])
        support_position_drift_m = max(
            float(
                torch.linalg.vector_norm(
                    torch.tensor(sample["root_position_m"]) - support_reference_pos
                ).item()
            )
            for sample in support_window
        )
        support_orientation_drift_rad = max(
            float(
                2.0
                * torch.acos(
                    torch.clamp(
                        torch.abs(torch.dot(support_reference_quat, torch.tensor(sample["root_quaternion_wxyz"]))),
                        0.0,
                        1.0,
                    )
                ).item()
            )
            for sample in support_window
        )
        support_max_linear_speed = max(
            float(torch.linalg.vector_norm(torch.tensor(sample["root_velocity_mps_radps"][:3])).item())
            for sample in support_window
        )
        support_max_angular_speed = max(
            float(torch.linalg.vector_norm(torch.tensor(sample["root_velocity_mps_radps"][3:])).item())
            for sample in support_window
        )
        # The audited bolt tip is the local -Y endpoint; transform the live root
        # pose instead of reusing the spawn-time USD bounds cache.
        bolt_tip_local_m = torch.tensor((0.0, -0.026149998, 0.0), device=env.device).unsqueeze(0)
        bolt_tip_world_m = torch.tensor(bolt_pre_skill_pos, device=env.device).unsqueeze(0) + torch_utils.quat_apply(
            torch.tensor(bolt_pre_skill_quat, device=env.device).unsqueeze(0), bolt_tip_local_m
        )
        bolt_tip_gap_m = float(bolt_tip_world_m[0, 2].item()) - cfg.table_top_z_m
        bolt_spawn_bounds_min = env._scene_collision_info["bolt"].get("world_bounds_min_m")
        expected_bolt_quat = torch.tensor(cfg.bolt_reset_quat_wxyz, device=env.device)
        observed_bolt_quat = torch.tensor(bolt_pre_skill_quat, device=env.device)
        upright_quat_alignment = float(
            torch.abs(torch.dot(expected_bolt_quat, observed_bolt_quat)).item()
        )
        bolt_nominal_position_error_m = float(
            torch.linalg.vector_norm(
                torch.tensor(bolt_pre_skill_pos) - torch.tensor(cfg.bolt_reset_pos_m)
            ).item()
        )
        bolt_upright_orientation_error_rad = float(
            2.0 * torch.acos(torch.clamp(torch.tensor(upright_quat_alignment), 0.0, 1.0)).item()
        )
        bolt_linear_speed = float(torch.linalg.vector_norm(torch.tensor(bolt_pre_skill_velocity[:3])).item())
        bolt_angular_speed = float(torch.linalg.vector_norm(torch.tensor(bolt_pre_skill_velocity[3:])).item())
        bolt_tip_supported = (
            abs(bolt_tip_gap_m) <= 0.00015
            and bolt_nominal_position_error_m <= 0.0005
            and bolt_upright_orientation_error_rad <= 0.005
            and support_position_drift_m <= 0.0002
            and support_orientation_drift_rad <= 0.005
        )
        bolt_support_diagnostics = {
            "root_position_m": bolt_pre_skill_pos,
            "root_quaternion_wxyz": bolt_pre_skill_quat,
            "root_velocity_mps_radps": bolt_pre_skill_velocity,
            "live_tip_position_m": bolt_tip_world_m[0].detach().cpu().tolist(),
            "spawn_collision_bounds_min_m_diagnostic_only": bolt_spawn_bounds_min,
            "table_top_z_m": cfg.table_top_z_m,
            "tip_gap_to_table_m": bolt_tip_gap_m,
            "upright_quaternion_alignment": upright_quat_alignment,
            "nominal_root_position_error_m": bolt_nominal_position_error_m,
            "upright_orientation_error_rad": bolt_upright_orientation_error_rad,
            "endpoint_linear_speed_mps": bolt_linear_speed,
            "endpoint_angular_speed_radps": bolt_angular_speed,
            "settle_window_sample_count": len(support_window),
            "settle_window_position_drift_m": support_position_drift_m,
            "settle_window_orientation_drift_rad": support_orientation_drift_rad,
            "settle_window_max_linear_speed_mps": support_max_linear_speed,
            "settle_window_max_angular_speed_radps": support_max_angular_speed,
            "tip_supported_and_stationary": bolt_tip_supported,
        }
        summary["simulator_private_ground_truth"] = {
            "not_passed_to_agent_or_model": True,
            "bolt_upright_tip_support_before_skill": bolt_support_diagnostics,
            "gripper_native_physics_material_bindings": env._runtime_grip_material_bindings,
            "scene_default_physics_material_cfg": {
                "static_friction": getattr(cfg.sim.physics_material, "static_friction", None),
                "dynamic_friction": getattr(cfg.sim.physics_material, "dynamic_friction", None),
                "restitution": getattr(cfg.sim.physics_material, "restitution", None),
            },
        }

        if args.stage == "episode":
            if baseline_stats is None:
                raise RuntimeError(
                    f"episode mode requires a verified stationary wrench zero: {baseline_error}"
                )
            if not bolt_tip_supported:
                raise RuntimeError(
                    "episode mode requires the unheld bolt to be stationary upright on its table tip"
                )
            from runtime.bolt_harness.episode import run_bolt_episode

            episode_result = run_bolt_episode(
                env,
                settings,
                run_dir / "episode",
                agent_mode=args.agent_mode,
            )
            private_physics_trace_path = _write_private_native_physics_trace(
                env, run_dir / "episode"
            )
            summary.update(
                {
                    "selected_cad_steps": settings.get("selected_cad_steps"),
                    "loaded_assets": {
                        "panda": str(PANDA_USD),
                        "bolt": str(BOLT_USD),
                        "fixed_casing": str(CASING_USD),
                        "fixed_cover": str(COVER_USD),
                    },
                    "panda_native_actuator_limits": {
                        name: {
                            "effort_limit_sim": getattr(actuator, "effort_limit_sim", None),
                            "velocity_limit_sim": getattr(actuator, "velocity_limit_sim", None),
                            "stiffness": getattr(actuator, "stiffness", None),
                            "damping": getattr(actuator, "damping", None),
                            "friction": getattr(actuator, "friction", None),
                        }
                        for name, actuator in cfg.robot.actuators.items()
                    },
                    "episode_result": {
                        "status": episode_result.status,
                        "exit_code": episode_result.exit_code,
                        "task_success": episode_result.task_success,
                        "video_finalized": episode_result.video_finalized,
                        "model_evidence": episode_result.model_evidence,
                        "run_dir": str(episode_result.run_dir),
                        "error": episode_result.error,
                    },
                    "private_native_physics_trace": str(
                        private_physics_trace_path.relative_to(run_dir)
                    ),
                    "task_success": bool(episode_result.task_success),
                    "final_success_claimed": bool(
                        episode_result.exit_code == 0 and episode_result.task_success
                    ),
                    "wrench_stationary_zero_reference": baseline_stats,
                    "wrench_baseline_error": baseline_error,
                    "task_rgb_camera_keyframes": {
                        "initial_free_bolt": summary["task_rgb_camera_keyframes"][
                            "initial_free_bolt"
                        ],
                        "last_real_rgb": _capture_task_rgb_keyframe(
                            env, run_dir / "keyframes" / "last_real_rgb.png", refresh=True
                        ),
                    },
                    "task_rgb_camera": _task_rgb_camera_diagnostics(
                        env, cfg, app_enabled=args.enable_cameras
                    ),
                    "task_rgb_camera_frame_trace": env._task_rgb_camera_frame_trace,
                }
            )
            summary["simulator_private_ground_truth"].update(
                {
                    "scene_collision_info": env._scene_collision_info,
                    "bolt_world_bounds_size_m": env._scene_collision_info["bolt"][
                        "world_bounds_size_m"
                    ],
                    "fixture_pose_m": {
                        "casing_root": cfg.assembly_frame_pos_m,
                        "cover_root": cfg.cover_pos_m,
                        "cover_quat_wxyz": cfg.cover_quat_wxyz,
                        "nominal_bolt_seat_root": cfg.bolt_seat_root_pos_m,
                    },
                },
            )
            summary["run_passed"] = episode_result.exit_code == 0
            summary["runner_exit_code"] = 0 if summary["run_passed"] else 2
            summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            (run_dir / "smoke_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            return episode_result.exit_code

        pick_stage_passed = False
        pick_result: dict[str, str] | None = None
        transport_result: dict[str, str] | None = None
        gripper_force_cap_n: float | None = None
        executor_trace: list[dict[str, object]] = []
        gripper_command_trace: list[dict[str, object]] = []
        if args.stage == "pick":
            if baseline_stats is None:
                raise RuntimeError(f"pick stage requires a verified stationary hold and wrench zero: {baseline_error}")
            if not bolt_tip_supported:
                raise RuntimeError("pick stage requires a stationary upright free bolt resting on the table tip")
            if not isinstance(settings, dict) or "max_gripper_force_n" not in settings:
                raise ValueError("pick stage requires explicit max_gripper_force_n in the YAML config")
            gripper_force_cap_n = float(settings["max_gripper_force_n"])
            native_finger_limit_n = float(env._native_finger_effort_limits.min().item())
            if gripper_force_cap_n <= 0.0 or gripper_force_cap_n / 2.0 > native_finger_limit_n:
                raise ValueError(
                    f"total gripper cap must be positive and at most {2.0 * native_finger_limit_n:.1f} N"
                )
            if "grasp_cap_pad_overlap_m" not in settings:
                raise ValueError("pick stage requires the CAD-measured grasp_cap_pad_overlap_m")
            cap_pad_overlap_m = float(settings["grasp_cap_pad_overlap_m"])
            cap_piece = next(
                (
                    piece
                    for piece in env._scene_collision_info["bolt"]["split_proxy_info"]["pieces"]
                    if piece["name"] == "cap"
                ),
                None,
            )
            if cap_piece is None:
                raise RuntimeError("the composed CAD-derived bolt cap proxy is missing")
            cap_local_bounds = cap_piece["bounds_root_local_source_units_xyz"]
            # The top-down grasp quaternion maps bolt-local Z across the Panda fingers.
            cap_width_m = float(cap_local_bounds[2][1] - cap_local_bounds[2][0]) * cfg.cad_stage_scale
            cap_height_m = float(cap_local_bounds[1][1] - cap_local_bounds[1][0]) * cfg.cad_stage_scale
            if not 0.0 < cap_pad_overlap_m < cap_height_m:
                raise ValueError(
                    f"CAD cap-pad overlap {cap_pad_overlap_m} m must be within cap height {cap_height_m} m"
                )
            finger_contact_offsets = [
                float(item["contact_offset_m"])
                for item in env._robot_collision_offset_info
                if "/panda_leftfinger/" in item["prim_path"]
                or "/panda_rightfinger/" in item["prim_path"]
            ]
            width_contact_allowance_m = (
                max(finger_contact_offsets, default=0.0)
                + float(env._scene_collision_info["bolt"]["contact_offset_m"])
            )
            plan = make_nominal_bolt_motion_plan(max_gripper_force_n=gripper_force_cap_n)
            summary["gripper_force_cap"] = {
                "requested_total_n": gripper_force_cap_n,
                "requested_per_finger_n": gripper_force_cap_n / 2.0,
                "native_per_finger_effort_limit_n": native_finger_limit_n,
                "below_native_limit": gripper_force_cap_n / 2.0 <= native_finger_limit_n,
                "calibration_status": "candidate cap; lift/transport must empirically verify adequacy",
                "grasp_target_width_m": plan.grasp_gripper_width_m,
                "grasp_width_completion_is_not_a_success_condition": True,
                "cap_width_m_measured_from_composed_cad_proxy": cap_width_m,
                "width_contact_allowance_m": width_contact_allowance_m,
                "cap_pad_overlap_allowance_m": cap_pad_overlap_m,
            }
            skill_marker, executor_trace, gripper_command_trace = _install_executor_trace(
                env,
                summary,
                private_contact_source,
                robot_fixture_contact_source,
                run_dir / "keyframes",
            )
            executor = BoltSkillExecutor(env, target_id="bolt01", plan=plan)
            skill_marker["name"] = "pick"
            pick_result = executor.execute_skill("pick", "bolt01")
            if pick_result.get("status") == "completed":
                skill_marker["name"] = "transport"
                transport_result = executor.execute_skill("transport", "bolt01")
            final_executor_state = env.get_public_executor_state()
            summary["pick_stage_controller_endpoint"] = summary.get(
                "tcp_kinematic_diagnostics", {}
            ).get("latest_sample")
            final_contact_state = env.get_bilateral_grasp_contact()
            final_bilateral_contact = bool(final_contact_state["bilateral_contact"][0].item())
            final_gripper_width = float(final_executor_state["gripper_width_m"])
            bilateral_trace = [
                sample for sample in executor_trace
                if all(sample["finger_bolt_contacts"])
            ]
            max_contact_force_per_finger_n = {
                side: max(
                    (float(sample["finger_bolt_contact_forces_n"][side]) for sample in bilateral_trace),
                    default=0.0,
                )
                for side in ("left", "right")
            }
            initial_root = summary["simulator_private_ground_truth"][
                "bolt_upright_tip_support_before_skill"
            ]["root_position_m"]
            retention = _measure_pick_retention(
                summary["simulator_private_ground_truth"]["pick_control_trace"],
                initial_bolt_root_z_m=float(initial_root[2]),
                planned_lift_m=(
                    float(plan.pick_waypoints[2].position_m[2])
                    - float(plan.pick_waypoints[1].position_m[2])
                ),
                cap_pad_overlap_m=cap_pad_overlap_m,
                cap_width_m=cap_width_m,
                width_contact_allowance_m=width_contact_allowance_m,
                measured_width_m=final_gripper_width,
                final_bilateral_contact=final_bilateral_contact,
            )
            summary["pick_retention"] = retention
            pick_stage_passed = bool(
                pick_result.get("status") == "completed"
                and transport_result is not None
                and transport_result.get("status") == "completed"
                and retention["passed"]
                and not summary["private_bolt_contact_buffer"]["capacity_reached_or_truncation_possible"]
            )
            summary["pick_result"] = pick_result
            summary["transport_result"] = transport_result or {"status": "not_attempted"}
            summary["pick_stage_final_state"] = {
                "tcp_pose": final_executor_state["tcp_pose"][0].detach().cpu().tolist(),
                "gripper_width_m": final_gripper_width,
                "finger_bolt_contacts": list(final_executor_state["finger_bolt_contacts"]),
                "finger_contact_forces_n": {
                    "left": float(final_contact_state["left_force_n"][0].item()),
                    "right": float(final_contact_state["right_force_n"][0].item()),
                },
            }
            summary["gripper_force_cap"]["bilateral_contact_samples"] = len(bilateral_trace)
            summary["gripper_force_cap"]["max_measured_contact_force_per_finger_n"] = max_contact_force_per_finger_n
            summary["gripper_force_cap"]["empirically_sufficient_for_pick_and_transport"] = pick_stage_passed

        smoke_steps_completed = 0
        unexpected_done = False
        smoke_joint_speeds = []
        smoke_tcp_linear_speeds = []
        smoke_tcp_angular_speeds = []
        smoke_tracking_errors = []
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(torch.zeros((1, 6), device=env.device))
            _record_task_rgb_camera_frame(env, env._task_rgb_camera_frame_trace)
            if bool(terminated.any() or truncated.any()):
                unexpected_done = True
                break
            _record_private_bolt_contacts(env, private_contact_source, summary, "stationary_hold")
            _record_zero_control_sample(summary, "stationary_hold", env)
            state = env.get_public_executor_state()
            smoke_joint_speeds.append(float(env._robot.data.joint_vel.abs().max().item()))
            smoke_tcp_linear_speeds.append(float(torch.linalg.vector_norm(state["tcp_velocity"][0, :3]).item()))
            smoke_tcp_angular_speeds.append(float(torch.linalg.vector_norm(state["tcp_velocity"][0, 3:]).item()))
            smoke_tracking_errors.append(float(state["tcp_tracking_error_m"][0].item()))
            smoke_steps_completed += 1
        end_pos = env._bolt.data.root_pos_w[0].detach().cpu().tolist()
        wrench = env.public_wrench()[0].detach().cpu()
        end_joint_pos = env._robot.data.joint_pos[0].detach().cpu().tolist()
        end_quat = env._bolt.data.root_quat_w[0].detach().cpu().tolist()
        end_control_state = env.get_public_executor_state()
        end_contacts = env.get_bilateral_grasp_contact()
        table_half = [dimension / 2.0 for dimension in cfg.table_size_m]
        bolt_end_retained_at_table = (
            abs(end_pos[0] - cfg.table_center_m[0]) <= table_half[0]
            and abs(end_pos[1] - cfg.table_center_m[1]) <= table_half[1]
            and cfg.table_top_z_m - 0.02 <= end_pos[2] <= cfg.table_top_z_m + 0.15
        )
        table_retained = bolt_tip_supported if args.stage == "pick" else bolt_end_retained_at_table
        summary["simulator_private_ground_truth"].update(
            {
                "bolt_start_root_pose": start_pos + start_quat,
                "bolt_end_root_pose": end_pos + end_quat,
                "bolt_end_retained_at_table_height": bolt_end_retained_at_table,
                "bolt_start_joint_positions_rad": start_joint_pos,
                "bolt_end_joint_positions_rad": end_joint_pos,
                "scene_collision_info": env._scene_collision_info,
                "bolt_world_bounds_size_m": env._scene_collision_info["bolt"][
                    "world_bounds_size_m"
                ],
                "fixture_pose_m": {
                    "casing_root": cfg.assembly_frame_pos_m,
                    "cover_root": cfg.cover_pos_m,
                    "cover_quat_wxyz": cfg.cover_quat_wxyz,
                    "nominal_bolt_seat_root": cfg.bolt_seat_root_pos_m,
                },
            }
        )
        bolt_collision_info = env._scene_collision_info["bolt"]
        bolt_pieces = bolt_collision_info.get("split_proxy_info", {}).get("pieces", [])
        split_proxies_valid = (
            len(bolt_pieces) == 2
            and bolt_collision_info.get("source_visual_collision_enabled") is False
            and all(
                piece.get("collision_enabled") is True
                and piece.get("approximation") == "convexHull"
                and piece.get("physx_hull_vertex_limit") == 64
                for piece in bolt_pieces
            )
        )
        authored_colliders = bool(
            split_proxies_valid
            and all(env._scene_collision_info[name].get("mesh_count", 0) > 0 for name in ("casing", "cover"))
        )
        bounds_size = env._scene_collision_info["bolt"]["world_bounds_size_m"]
        bounds_correct = 0.04 <= max(bounds_size) <= 0.07
        no_automatic_reset = env._reset_count == 1
        panda_contact_offset_max = max(
            (item["contact_offset_m"] for item in env._robot_collision_offset_info), default=float("inf")
        )
        cover_contact_offset = env._scene_collision_info["cover"]["contact_offset_m"]
        hand_cover_geometric_clearance_m = 0.001
        contact_offset_budget_m = panda_contact_offset_max + cover_contact_offset
        hand_cover_clearance_safe = (
            all(item["collision_enabled"] for item in env._robot_collision_offset_info)
            and hand_cover_geometric_clearance_m > contact_offset_budget_m
        )
        summary.update(
            {
                "loaded_assets": {
                    "panda": str(PANDA_USD),
                    "bolt": str(BOLT_USD),
                    "fixed_casing": str(CASING_USD),
                    "fixed_cover": str(COVER_USD),
                },
                "robot_body_names": list(env._robot.body_names),
                "robot_joint_names": list(env._robot.joint_names),
                "panda_joint_physics_usd": env._robot_joint_physics_info,
                "panda_native_actuator_limits": {
                    name: {
                        "effort_limit_sim": getattr(actuator, "effort_limit_sim", None),
                        "velocity_limit_sim": getattr(actuator, "velocity_limit_sim", None),
                        "stiffness": getattr(actuator, "stiffness", None),
                        "damping": getattr(actuator, "damping", None),
                        "friction": getattr(actuator, "friction", None),
                    }
                    for name, actuator in cfg.robot.actuators.items()
                },
                "panda_robot_gravity_disabled_factory_style": bool(cfg.robot.spawn.rigid_props.disable_gravity),
                "panda_composed_rigid_body_gravity": env._robot_gravity_body_flags,
                "panda_collision_shapes_task_layer": env._robot_collision_offset_info,
                "nominal_seat_hand_cover_clearance": {
                    "geometric_whole_hand_clearance_m_from_cad_audit": hand_cover_geometric_clearance_m,
                    "max_panda_contact_offset_m": panda_contact_offset_max,
                    "cover_contact_offset_m": cover_contact_offset,
                    "pair_contact_offset_budget_m": contact_offset_budget_m,
                    "remaining_nominal_clearance_m": hand_cover_geometric_clearance_m - contact_offset_budget_m,
                    "all_panda_collision_shapes_enabled": all(
                        item["collision_enabled"] for item in env._robot_collision_offset_info
                    ),
                },
                "motion_safety_armed": bool(env.motion_safety_armed),
                "factory_nullspace_reference_rad": list(cfg.ctrl.default_dof_pos_tensor),
                "factory_control_uses_persistent_commanded_tcp_target": True,
                "commanded_tcp_pose_end": end_control_state["commanded_tcp_pose"][0].detach().cpu().tolist(),
                "tcp_tracking_error_end_m": float(end_control_state["tcp_tracking_error_m"][0].item()),
                "tcp_orientation_tracking_error_end_rad": float(
                    end_control_state["tcp_orientation_tracking_error_rad"][0].item()
                ),
                "tcp_safety_stop_limits": {
                    "tracking_error_m": cfg.tcp_tracking_stop_error_m,
                    "orientation_error_rad": cfg.tcp_tracking_stop_orientation_rad,
                    "linear_speed_mps": cfg.tcp_linear_speed_stop_mps,
                    "angular_speed_radps": cfg.tcp_angular_speed_stop_radps,
                },
                "wrench_body": env.cfg.wrench_body_name,
                "wrench_source": "root_physx_view.get_link_incoming_joint_force",
                "wrench_frame_transform": "isaaclab_tasks.direct.forge.forge_utils.change_FT_frame",
                "wrench_finite": bool(torch.isfinite(wrench).all()),
                "wrench_source_frame": env.wrench_source_frame,
                "wrench_api_doc": env.wrench_api_doc,
                "wrench_child_joint_prim_path": env.wrench_joint_prim_path,
                "wrench_child_joint_local_pos_m": env._wrench_joint_local_pos.detach().cpu().tolist(),
                "wrench_child_joint_local_quat_wxyz": env._wrench_joint_local_quat.detach().cpu().tolist(),
                "wrench_end_N_Nm": wrench.tolist(),
                "wrench_stationary_zero_reference": baseline_stats,
                "wrench_baseline_error": baseline_error,
                "wrench_sign_and_load_calibration_performed": False,
                "tcp_offset_from_hand_local_m": cfg.tcp_offset_from_hand_local_m,
                "tcp_stability_drift_m": tcp_drift_m,
                "joint_speed_max_rad_s_during_baseline_window": max_joint_speed,
                "warmup_joint_speed_trace_max_rad_s": max(warmup_joint_speeds, default=float("inf")),
                "warmup_tcp_linear_speed_max_mps": max(warmup_tcp_linear_speeds, default=float("inf")),
                "warmup_tcp_angular_speed_max_radps": max(warmup_tcp_angular_speeds, default=float("inf")),
                "warmup_tracking_error_max_m": max(warmup_tracking_errors, default=float("inf")),
                "smoke_joint_speed_max_rad_s": max(smoke_joint_speeds, default=float("inf")),
                "smoke_tcp_linear_speed_max_mps": max(smoke_tcp_linear_speeds, default=float("inf")),
                "smoke_tcp_angular_speed_max_radps": max(smoke_tcp_angular_speeds, default=float("inf")),
                "smoke_tracking_error_max_m": max(smoke_tracking_errors, default=float("inf")),
                "bilateral_contacts_clear_during_baseline_window": bilateral_contacts_clear,
                "bilateral_contact_end_forces_n": {
                    "left": float(end_contacts["left_force_n"][0].item()),
                    "right": float(end_contacts["right_force_n"][0].item()),
                },
                "bolt_reset_known_layout_is_provisional": True,
                "bolt_gravity_enabled": not bool(cfg.bolt.spawn.rigid_props.disable_gravity),
                "bolt_reset_is_free_rigid_object": True,
                "table_top_z_m": cfg.table_top_z_m,
                "geometry_pose_status": (
                    "candidate #01 is CAD-only; the active task-layer split convex proxy is source-derived and "
                    "has not yet passed physical cover/casing contact calibration"
                ),
                "steps_completed": int(env.common_step_counter),
                "smoke_steps_completed": smoke_steps_completed,
                "unexpected_done": unexpected_done,
                "cad_spawn_scale_m_per_source_unit": cfg.cad_stage_scale,
                "bolt_world_bounds_size_m": env._scene_collision_info["bolt"]["world_bounds_size_m"],
                "tcp_body": cfg.tcp_body_name if not env._tcp_uses_hand_offset else "panda_hand plus measured local offset",
                "private_ground_truth_diagnostics": "See simulator_private_ground_truth; never included in agent/model inputs",
                "task_success": None,
                "final_success_claimed": False,
            }
        )
        summary["bolt_retained_at_table_height"] = table_retained
        summary["cad_collision_meshes_authored"] = authored_colliders
        summary["bolt_split_proxies_authored_with_gpu_hull_limit"] = split_proxies_valid
        summary["bolt_dimensions_verified_from_composed_usd"] = bounds_correct
        summary["nominal_hand_cover_clearance_exceeds_contact_offsets"] = hand_cover_clearance_safe
        summary["single_episode_reset_count"] = env._reset_count
        private_physics_trace_path = _write_private_native_physics_trace(env, run_dir)
        summary["private_native_physics_trace"] = str(
            private_physics_trace_path.relative_to(run_dir)
        )
        summary["task_rgb_camera_keyframes"]["last_real_rgb"] = _capture_task_rgb_keyframe(
            env, run_dir / "keyframes" / "last_real_rgb.png", refresh=True
        )
        _record_task_rgb_camera_frame(env, env._task_rgb_camera_frame_trace)
        summary["task_rgb_camera"] = _task_rgb_camera_diagnostics(env, cfg, app_enabled=args.enable_cameras)
        checks_passed = bool(
            summary["wrench_finite"]
            and baseline_stats is not None
            and table_retained
            and authored_colliders
            and bounds_correct
            and hand_cover_clearance_safe
            and no_automatic_reset
            and not summary["private_bolt_contact_buffer"]["capacity_reached_or_truncation_possible"]
            and env.motion_safety_armed
            and not unexpected_done
        )
        summary["pick_stage_passed"] = pick_stage_passed if args.stage == "pick" else None
        summary["smoke_passed"] = bool(
            checks_passed
            and (
                smoke_steps_completed == steps
                if args.stage == "hold"
                else pick_stage_passed
            )
        )
        summary["run_passed"] = bool(summary["smoke_passed"])
        summary["runner_exit_code"] = 0 if summary["run_passed"] else 2
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return 0 if summary["smoke_passed"] else 2
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        summary["run_passed"] = False
        summary["runner_exit_code"] = 2
        summary["smoke_passed"] = False
        summary["final_success_claimed"] = False
        if env is not None:
            if private_physics_trace_path is None:
                try:
                    evidence_root = run_dir / "episode" if args.stage == "episode" else run_dir
                    private_physics_trace_path = _write_private_native_physics_trace(
                        env, evidence_root
                    )
                    summary["private_native_physics_trace"] = str(
                        private_physics_trace_path.relative_to(run_dir)
                    )
                except Exception as trace_exc:
                    summary["private_native_physics_trace_error"] = (
                        f"{type(trace_exc).__name__}: {trace_exc}"
                    )
            try:
                summary.setdefault("task_rgb_camera_keyframes", {})["post_error_real_rgb"] = (
                    _capture_task_rgb_keyframe(
                        env, run_dir / "keyframes" / "post_error_real_rgb.png", refresh=True
                    )
                )
            except Exception as camera_exc:
                summary["post_error_camera_error"] = f"{type(camera_exc).__name__}: {camera_exc}"
            try:
                private_state = summary.setdefault(
                    "simulator_private_ground_truth",
                    {"not_passed_to_agent_or_model": True},
                )
                private_state["last_factory_control_snapshot"] = env.get_factory_control_snapshot()
                failure_state = env.get_public_executor_state()
                private_state["failure_diagnostics"] = {
                    "simulation_step_counter": int(env.common_step_counter),
                    "motion_safety_armed": bool(env.motion_safety_armed),
                    "panda_composed_rigid_body_gravity": env._robot_gravity_body_flags,
                    "tcp_pose": failure_state["tcp_pose"][0].detach().cpu().tolist(),
                    "commanded_tcp_pose": failure_state["commanded_tcp_pose"][0].detach().cpu().tolist(),
                    "tcp_velocity": failure_state["tcp_velocity"][0].detach().cpu().tolist(),
                    "joint_positions": env._robot.data.joint_pos[0].detach().cpu().tolist(),
                    "joint_velocities": env._robot.data.joint_vel[0].detach().cpu().tolist(),
                    "factory_nullspace_reference_rad": list(env.cfg.ctrl.default_dof_pos_tensor),
                    "last_factory_task_wrench": (
                        env.applied_wrench[0].detach().cpu().tolist()
                        if getattr(env, "applied_wrench", None) is not None
                        else None
                    ),
                    "last_factory_joint_torque_Nm": (
                        env.joint_torque[0].detach().cpu().tolist()
                        if getattr(env, "joint_torque", None) is not None
                        else None
                    ),
                    "tcp_jacobian_joint7_angular": env.fingertip_midpoint_jacobian[0, 3:6, 6]
                    .detach()
                    .cpu()
                    .tolist(),
                    "panda_collision_shapes_task_layer": env._robot_collision_offset_info,
                    "scene_collision_info": env._scene_collision_info,
                }
            except Exception as diagnostic_exc:
                summary["diagnostic_error"] = f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return 2
    finally:
        if env is not None:
            try:
                env.close()
            except SystemExit as close_exit:
                if close_exit.code not in (None, 0):
                    raise
        if simulation_app is not None:
            try:
                simulation_app.close()
            except SystemExit as close_exit:
                # Some Kit shutdown paths raise SystemExit(0), which must not mask
                # the runner's already-determined failure return code.
                if close_exit.code not in (None, 0):
                    raise


if __name__ == "__main__":
    raise SystemExit(main())
