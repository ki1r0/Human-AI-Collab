#!/usr/bin/env python3
"""Capture seat contacts or negative evaluator probes; never evaluates task success."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts/bolt_insertion/contact_calibration"
CONTAINER_REPO = Path("/workspace/Human-AI-Collab")
ISAAC_IMAGE = "nvcr.io/nvidia/isaac-lab:2.3.0"
CONTACT_CAPACITY = 1024
PROBE_RESET_OFFSETS_M = {
    # Small clearance above the audited nominal seat; no robot grasp or pose writes.
    "gravity-hover": (0.0, 0.0, 0.005),
    # Horizontal reset: local +Y lies tangent to the selected hole; the cap
    # envelope starts 2 mm above the audited cover-bearing reference plane.
    "cover-away": (0.014, 0.0, 0.031397),
}
PROBE_RESET_QUATERNIONS_WXYZ = {
    "cover-away": (1.0, 0.0, 0.0, 0.0),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_name(probe: str | None = None) -> str:
    now = _utc_now()
    label = probe or "seat"
    return f"gpu2-{label}-{now.strftime('%Y%m%dT%H%M%S')}-{now.microsecond:06d}Z"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _python_value(value: Any) -> Any:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "tolist", None)):
        value = value.tolist()
    if isinstance(value, tuple):
        return [_python_value(item) for item in value]
    if isinstance(value, list):
        return [_python_value(item) for item in value]
    if callable(getattr(value, "item", None)):
        value = value.item()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return float(value)


def _float(value: Any) -> float:
    return float(_python_value(value))


def _pair_value(values: Any, first: int, second: int) -> Any:
    try:
        return values[first, second]
    except (TypeError, IndexError):
        return values[first][second]


def _row(values: Any, index: int) -> list[float]:
    row = _python_value(values[index])
    if not isinstance(row, list):
        row = [row]
    return [float(value) for value in row]


def _vector(values: Any, index: int = 0) -> list[float]:
    row = _python_value(values[index])
    if not isinstance(row, list):
        raise TypeError("expected a vector-valued simulator field")
    return [float(value) for value in row]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _time_s(env: Any) -> float:
    return _float(env._robot._data._sim_timestamp)


def _bolt_runtime_body_metadata(env: Any, bolt_root: Any) -> dict[str, Any]:
    from pxr import PhysxSchema, Usd, UsdPhysics

    body_prims = [prim for prim in Usd.PrimRange(bolt_root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    bodies = []
    for prim in body_prims:
        rigid_body = UsdPhysics.RigidBodyAPI(prim)
        physx_body = PhysxSchema.PhysxRigidBodyAPI(prim)
        gravity_attr = physx_body.GetDisableGravityAttr()
        kinematic_attr = rigid_body.GetKinematicEnabledAttr()
        bodies.append(
            {
                "prim_path": str(prim.GetPath()),
                "is_bolt_root": prim == bolt_root,
                "disable_gravity": gravity_attr.Get() if gravity_attr.IsValid() else None,
                "kinematic_enabled": kinematic_attr.Get() if kinematic_attr.IsValid() else None,
            }
        )

    view = env._bolt.root_physx_view
    return {
        "root_prim_path": str(bolt_root.GetPath()),
        "rigid_body_count_under_bolt": len(body_prims),
        "rigid_bodies": bodies,
        "physx_view_masses_kg_raw": _python_value(view.get_masses()),
        "physx_view_inertias_kg_m2_raw": _python_value(view.get_inertias()),
        "physx_view_api": "env._bolt.root_physx_view",
    }


def _bolt_axis_from_quaternion(quaternion_wxyz: list[float]) -> list[float]:
    w, x, y, z = quaternion_wxyz
    return [
        -2.0 * w * z + 2.0 * x * y,
        1.0 - 2.0 * (x * x + z * z),
        2.0 * w * x + 2.0 * y * z,
    ]


def _raw_contact_sample(
    env: Any,
    source: dict[str, Any],
    *,
    sample_index: int,
    phase: str,
) -> dict[str, Any]:
    from runtime.bolt_harness.measurement import (
        BOLT_TIP_LOCAL_Y_M,
        CAP_SUPPORT_PLANE_Z_M,
        CAP_UNDERSIDE_LOCAL_Y_M,
        CASING_INTERIOR_FLOOR_Z_M,
    )

    view = source["contact_physx_view"]
    dt_s = float(source["dt_s"])
    forces, points, normals, separations, counts, starts = view.get_contact_data(dt_s)
    force_matrix = _python_value(view.get_contact_force_matrix(dt_s))
    net_contact_forces = _python_value(view.get_net_contact_forces(dt_s))
    gravity_world = [float(value) for value in source["gravity_world_mps2"]]
    gravity_norm = math.sqrt(sum(value * value for value in gravity_world))
    upward = [-value / gravity_norm for value in gravity_world] if gravity_norm else None
    net_force_world = [float(value) for value in net_contact_forces[0]]
    net_force_up_n = (
        sum(net_force_world[i] * upward[i] for i in range(3)) if upward is not None else None
    )
    expected_static_weight_n = float(source["bolt_mass_kg_actual"]) * gravity_norm

    bolt = env._bolt.data
    root_pos = _vector(bolt.root_pos_w)
    root_quat = _vector(bolt.root_quat_w)
    root_linvel = _vector(bolt.root_lin_vel_w)
    root_angvel = _vector(bolt.root_ang_vel_w)
    native_transform = _vector(env._bolt.root_physx_view.get_transforms())
    native_velocity = _vector(env._bolt.root_physx_view.get_velocities())
    if len(native_transform) != 7 or len(native_velocity) != 6:
        raise ValueError(
            "unexpected native PhysX root state shape: "
            f"transform={len(native_transform)}, velocity={len(native_velocity)}"
        )
    native_quat_wxyz = [native_transform[6], *native_transform[3:6]]
    data_quat_norm = math.sqrt(sum(value * value for value in root_quat))
    native_quat_norm = math.sqrt(sum(value * value for value in native_quat_wxyz))
    orientation_dot = sum(
        (root_quat[index] / data_quat_norm) * (native_quat_wxyz[index] / native_quat_norm)
        for index in range(4)
    )
    orientation_error_rad = 2.0 * math.acos(min(1.0, abs(orientation_dot)))
    position_delta = [root_pos[i] - native_transform[i] for i in range(3)]
    linear_delta = [root_linvel[i] - native_velocity[i] for i in range(3)]
    angular_delta = [root_angvel[i] - native_velocity[i + 3] for i in range(3)]
    axis = _bolt_axis_from_quaternion(root_quat)
    cap_underside_center = [
        root_pos[i] + axis[i] * CAP_UNDERSIDE_LOCAL_Y_M for i in range(3)
    ]
    bolt_tip = [root_pos[i] + axis[i] * BOLT_TIP_LOCAL_Y_M for i in range(3)]
    timestamp = _time_s(env)
    filters = []
    total_count = 0
    filter_matrix_sum = [0.0, 0.0, 0.0]

    for item in source["filter_map"]:
        filter_index = int(item["filter_index"])
        count = int(_float(_pair_value(counts, 0, filter_index)))
        start = int(_float(_pair_value(starts, 0, filter_index)))
        if count < 0:
            raise ValueError(f"negative PhysX contact count for filter {filter_index}: {count}")
        if count and start < 0:
            raise ValueError(f"negative PhysX contact start for non-empty filter {filter_index}: {start}")

        contacts = []
        contact_force_sum = [0.0, 0.0, 0.0]
        contact_force_sum_valid = True
        for point_index in range(start, start + count):
            force_raw = _row(forces, point_index)
            point_world = _row(points, point_index)
            normal_world = _row(normals, point_index)
            separation_raw = _row(separations, point_index)
            if len(point_world) != 3 or len(normal_world) != 3:
                raise ValueError("PhysX returned a non-3D contact point or normal")
            force_raw_json = [value if math.isfinite(value) else None for value in force_raw]
            force_scalar_raw = force_raw[0] if len(force_raw) == 1 else None
            force_scalar_finite = force_scalar_raw is not None and math.isfinite(force_scalar_raw)
            normal_norm = math.sqrt(sum(value * value for value in normal_world))
            force_vector = None
            if force_scalar_finite and all(math.isfinite(value) for value in normal_world):
                force_vector = [force_scalar_raw * value for value in normal_world]
                for axis_index in range(3):
                    contact_force_sum[axis_index] += force_vector[axis_index]
            else:
                contact_force_sum_valid = False
            raw_dot = sum(normal_world[i] * axis[i] for i in range(3))
            cosine = raw_dot / normal_norm if normal_norm and math.isfinite(normal_norm) else None
            if cosine is not None and not math.isfinite(cosine):
                cosine = None
            if cosine is None or abs(cosine) <= 1e-12:
                sign = "zero_or_invalid"
            else:
                sign = "positive_axis" if cosine > 0.0 else "negative_axis"
            contacts.append(
                {
                    "sim_timestamp_s": timestamp,
                    "force_raw": _json_safe(force_raw_json),
                    "force_raw_repr": [repr(value) for value in force_raw],
                    "force_scalar_raw_n": force_scalar_raw if force_scalar_finite else None,
                    "force_scalar_raw_repr": repr(force_scalar_raw) if force_scalar_raw is not None else None,
                    "force_scalar_raw_status": (
                        "finite_scalar"
                        if force_scalar_finite
                        else "nonfinite_scalar"
                        if len(force_raw) == 1
                        else "unexpected_component_count"
                    ),
                    "force_vector_scalar_times_normal_w_n": _json_safe(force_vector),
                    "point_world_m": _json_safe(point_world),
                    "normal_world_raw": _json_safe(normal_world),
                    "normal_dot_bolt_axis_raw": _json_safe(raw_dot),
                    "normal_bolt_axis_cosine": cosine,
                    "normal_axis_sign_unverified": sign,
                    "separation_raw_m": _json_safe(separation_raw),
                }
            )
        total_count += count
        filters.append(
            {
                "filter_index": filter_index,
                "filter_prim_path": item["filter_prim_path"],
                "source_prim_path": item["source_prim_path"],
                "category": item["category"],
                "reported_count": count,
                "reported_start_index": start,
                "contacts": contacts,
            }
        )
        pair_matrix_force = [
            float(value) for value in force_matrix[0][filter_index]
        ]
        for axis_index in range(3):
            filter_matrix_sum[axis_index] += pair_matrix_force[axis_index]
        matrix_minus_contact_sum = (
            [pair_matrix_force[i] - contact_force_sum[i] for i in range(3)]
            if contact_force_sum_valid
            else None
        )
        filters[-1].update(
            {
                "pair_force_matrix_w_n_raw": _json_safe(pair_matrix_force),
                "contact_force_sum_scalar_times_normal_w_n": (
                    contact_force_sum if contact_force_sum_valid else None
                ),
                "matrix_minus_contact_sum_w_n": _json_safe(matrix_minus_contact_sum),
            }
        )

    capacity = int(source["contact_data_capacity"])
    return {
        "sample_index": sample_index,
        "phase": phase,
        "wall_time_utc": _utc_now().isoformat(),
        "sim_timestamp_s": timestamp,
        "bolt_root_pose_world": {
            "position_m": root_pos,
            "quaternion_wxyz": root_quat,
        },
        "bolt_root_velocity_world": {
            "linear_mps": root_linvel,
            "angular_radps": root_angvel,
        },
        "bolt_root_physx_view_native": {
            "transform_xyz_qxyzw_raw": native_transform,
            "velocity_linear_xyz_angular_xyz_raw": native_velocity,
            "data_minus_native_position_error_m": math.sqrt(sum(value * value for value in position_delta)),
            "data_minus_native_orientation_error_rad": orientation_error_rad,
            "data_minus_native_linear_velocity_error_mps": math.sqrt(sum(value * value for value in linear_delta)),
            "data_minus_native_angular_velocity_error_radps": math.sqrt(sum(value * value for value in angular_delta)),
        },
        "bolt_local_positive_y_axis_world": axis,
        "bolt_cap_underside_center_world_m": cap_underside_center,
        "cap_support_plane_gap_m": cap_underside_center[2] - CAP_SUPPORT_PLANE_Z_M,
        "bolt_tip_world_m": bolt_tip,
        "tip_to_casing_interior_floor_clearance_m": bolt_tip[2] - CASING_INTERIOR_FLOOR_Z_M,
        "pair_force_matrix_w_n_raw": _json_safe(force_matrix),
        "net_contact_forces_on_sensor_w_n_raw": _json_safe(net_contact_forces),
        "filter_matrix_sum_w_n_raw": _json_safe(filter_matrix_sum),
        "net_minus_filter_matrix_sum_w_n_raw": _json_safe(
            [net_force_world[i] - filter_matrix_sum[i] for i in range(3)]
        ),
        "gravity_reference": {
            "world_acceleration_mps2": gravity_world,
            "actual_bolt_mass_kg": float(source["bolt_mass_kg_actual"]),
            "expected_static_weight_n": expected_static_weight_n,
            "net_contact_force_projected_up_n": _json_safe(net_force_up_n),
            "up_world": upward,
        },
        "contact_buffer": {
            "dt_s": dt_s,
            "configured_capacity": capacity,
            "max_contact_data_count_per_prim": int(source["max_contact_data_count_per_prim"]),
            "reported_total_contact_points": total_count,
            "capacity_reached_or_truncation_possible": total_count >= capacity,
            "counts_by_filter": [item["reported_count"] for item in filters],
            "filters": filters,
        },
    }


def _new_filter_stats(filter_map: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(item["filter_index"]): {
            "filter_index": int(item["filter_index"]),
            "filter_prim_path": item["filter_prim_path"],
            "source_prim_path": item["source_prim_path"],
            "category": item["category"],
            "samples": 0,
            "samples_with_contacts": 0,
            "reported_contact_points": 0,
            "peak_points_in_sample": 0,
            "force_scalar_raw_finite_count": 0,
            "force_scalar_raw_sum_n_signed": 0.0,
            "force_scalar_raw_min_n": None,
            "force_scalar_raw_max_n": None,
            "force_scalar_raw_min_time_s": None,
            "force_scalar_raw_max_time_s": None,
            "force_scalar_raw_negative_count": 0,
            "force_scalar_raw_zero_count": 0,
            "force_scalar_raw_unusable_count": 0,
            "separation_min_m": None,
            "separation_max_m": None,
            "normal_axis_cosine_min": None,
            "normal_axis_cosine_max": None,
            "normal_axis_sign_counts_unverified": {
                "positive_axis": 0,
                "negative_axis": 0,
                "zero_or_invalid": 0,
            },
            "first_contact_time_s": None,
            "last_contact_time_s": None,
        }
        for item in filter_map
    }


def _update_stats(stats: dict[str, Any], sample: dict[str, Any]) -> None:
    timestamp = float(sample["sim_timestamp_s"])
    for filter_record in sample["contact_buffer"]["filters"]:
        current = stats[str(filter_record["filter_index"])]
        contacts = filter_record["contacts"]
        current["samples"] += 1
        current["reported_contact_points"] += len(contacts)
        current["peak_points_in_sample"] = max(current["peak_points_in_sample"], len(contacts))
        if not contacts:
            continue
        current["samples_with_contacts"] += 1
        if current["first_contact_time_s"] is None:
            current["first_contact_time_s"] = timestamp
        current["last_contact_time_s"] = timestamp
        for contact in contacts:
            force = contact["force_scalar_raw_n"]
            if force is None:
                current["force_scalar_raw_unusable_count"] += 1
            else:
                force = float(force)
                current["force_scalar_raw_finite_count"] += 1
                current["force_scalar_raw_sum_n_signed"] += force
                if force < 0.0:
                    current["force_scalar_raw_negative_count"] += 1
                elif force == 0.0:
                    current["force_scalar_raw_zero_count"] += 1
                low = current["force_scalar_raw_min_n"]
                high = current["force_scalar_raw_max_n"]
                if low is None or force < low:
                    current["force_scalar_raw_min_n"] = force
                    current["force_scalar_raw_min_time_s"] = timestamp
                if high is None or force > high:
                    current["force_scalar_raw_max_n"] = force
                    current["force_scalar_raw_max_time_s"] = timestamp
            separation = contact["separation_raw_m"]
            if len(separation) == 1:
                value = float(separation[0])
                low = current["separation_min_m"]
                high = current["separation_max_m"]
                current["separation_min_m"] = value if low is None else min(low, value)
                current["separation_max_m"] = value if high is None else max(high, value)
            cosine = contact["normal_bolt_axis_cosine"]
            if cosine is not None:
                cosine = float(cosine)
                low = current["normal_axis_cosine_min"]
                high = current["normal_axis_cosine_max"]
                current["normal_axis_cosine_min"] = cosine if low is None else min(low, cosine)
                current["normal_axis_cosine_max"] = cosine if high is None else max(high, cosine)
            sign = contact["normal_axis_sign_unverified"]
            current["normal_axis_sign_counts_unverified"][sign] += 1


def _finalize_phase_stats(stats: dict[str, Any]) -> dict[str, Any]:
    for current in stats.values():
        count = current["force_scalar_raw_finite_count"]
        current["force_scalar_raw_mean_n_signed"] = (
            current["force_scalar_raw_sum_n_signed"] / count if count else None
        )
    return stats


def _make_probe_adapter(env: Any, source: dict[str, Any]) -> tuple[Any, Any, Any]:
    from runtime.bolt_harness.evaluator import BoltSeatTolerances
    from runtime.bolt_harness.measurement import (
        BoltSeatSamplingAdapter,
        BoltSeatSamplingCriteria,
        CapSupportContactCriteria,
    )

    tolerances = BoltSeatTolerances(
        max_cap_support_gap_m=0.00025,
        min_casing_entry_depth_m=0.015,
        max_shaft_radial_error_m=0.0002,
        max_penetration_m=0.0001,
        max_axial_motion_since_release_m=0.0002,
        max_relative_linear_speed_mps=0.002,
        max_relative_angular_speed_radps=0.01,
        held_seat_dwell_s=0.25,
        stable_dwell_s=1.0,
        max_sample_gap_s=0.033333333,
    )
    criteria = BoltSeatSamplingCriteria(
        cap_support=CapSupportContactCriteria(
            max_cap_patch_error_m=0.0001,
            max_contact_separation_m=0.0001,
            min_normal_alignment_cosine=0.95,
            normal_axis_sign=1,
            min_contact_force_n=0.10,
        ),
        min_cap_grasp_local_y_m=0.017850002289,
        max_cap_grasp_patch_error_m=0.0002,
        max_cap_grasp_contact_separation_m=0.0001,
        min_cap_grasp_force_n=0.05,
        max_cap_grasp_normal_alignment_cosine=0.25,
        max_robot_contact_separation_m=0.0001,
        min_robot_contact_force_n=0.02,
        max_controller_tcp_linear_speed_mps=0.002,
        max_controller_tcp_angular_speed_radps=0.01,
        max_controller_arm_joint_speed_radps=0.01,
        max_tcp_tracking_error_m=0.003,
        max_tcp_orientation_tracking_error_rad=0.005,
        release_gripper_width_m=0.079,
        min_pickup_root_lift_m=0.005,
    )
    return (
        BoltSeatSamplingAdapter(env, tolerances, criteria, contact_source=source),
        tolerances,
        criteria,
    )


def _capture_evaluator_sample(
    adapter: Any,
    *,
    phase: str,
    sample_index: int,
) -> dict[str, Any]:
    status = adapter.observe()
    trace = adapter._trace[-1]
    return _json_safe(
        {
            "sample_index": sample_index,
            "phase": phase,
            "time_s": trace.time_s,
            "geometry": asdict(trace.geometry),
            "measurement": asdict(trace.measurement),
            "status": asdict(status),
            "private_contact_evidence": {
                "cover_contact_count": len(trace.cover_contacts),
                "casing_contact_count": len(trace.casing_contacts),
                "robot_contact_count": len(trace.robot_contacts),
                "left_cap_grasp_contact_count": len(trace.left_grasp_contacts),
                "right_cap_grasp_contact_count": len(trace.right_grasp_contacts),
                "left_cap_grasp": trace.left_cap_grasp,
                "right_cap_grasp": trace.right_cap_grasp,
                "bilateral_contact": trace.bilateral_contact,
                "physically_held": trace.measurement.physically_held,
                "insertion_provenance_valid": trace.measurement.insertion_provenance_valid,
                "controller_stopped": trace.measurement.controller_stopped,
                "robot_contact_or_support": trace.measurement.robot_contact_or_support,
                "pickup_proven": trace.pickup_proven,
                "pickup_sequence_valid": trace.pickup_sequence_valid,
                "held_inserted_sample_count": trace.held_inserted_sample_count,
                "release_started": trace.release_started,
                "pickup_root_lift_m": trace.pickup_root_lift_m,
            },
        }
    )


def _probe_geometry_valid(row: dict[str, Any], tolerances: Any) -> bool:
    geometry = row["geometry"]
    measurement = row["measurement"]
    return (
        geometry["through_selected_cover_hole"]
        and geometry["entered_selected_casing_opening"]
        and measurement["casing_entry_depth_m"] >= tolerances.min_casing_entry_depth_m
        and measurement["shaft_radial_error_m"] <= tolerances.max_shaft_radial_error_m
        and measurement["cap_support_gap_m"] <= tolerances.max_cap_support_gap_m
        and measurement["cap_support_contact"]
        and measurement["penetration_m"] <= tolerances.max_penetration_m
        and not measurement["premature_bottoming"]
    )


def _validate_probe(
    probe: str,
    rows: list[dict[str, Any]],
    *,
    stable_window_samples: int,
    tolerances: Any,
) -> dict[str, Any]:
    no_public_success = bool(rows) and all(
        not row["status"]["seat_ready"] and not row["status"]["task_success"]
        for row in rows
    )
    never_held_or_proven = bool(rows) and all(
        not row["measurement"]["physically_held"]
        and not row["measurement"]["insertion_provenance_valid"]
        and not row["measurement"]["robot_contact_or_support"]
        for row in rows
    )
    window = rows[-stable_window_samples:] if len(rows) >= stable_window_samples else []
    low_motion = bool(window) and all(
        row["measurement"]["relative_linear_speed_mps"]
        <= tolerances.max_relative_linear_speed_mps
        and row["measurement"]["relative_angular_speed_radps"]
        <= tolerances.max_relative_angular_speed_radps
        for row in window
    )

    if probe == "gravity-hover":
        initial = rows[0] if rows else None
        initial_hover_rejected = bool(
            initial
            and not _probe_geometry_valid(initial, tolerances)
            and initial["measurement"]["cap_support_gap_m"]
            > tolerances.max_cap_support_gap_m
            and not initial["measurement"]["cap_support_contact"]
        )
        gravity_seated_without_provenance = bool(window) and all(
            _probe_geometry_valid(row, tolerances)
            and not row["measurement"]["physically_held"]
            and not row["measurement"]["insertion_provenance_valid"]
            and not row["measurement"]["robot_contact_or_support"]
            for row in window
        )
        checks = {
            "initial_hover_geometry_rejected": initial_hover_rejected,
            "gravity_only_seated_geometry_observed": gravity_seated_without_provenance,
            "seat_ready_and_task_success_never_projected": no_public_success,
            "no_grasp_or_robot_support_proven": never_held_or_proven,
            "gravity_seat_stable_for_window": low_motion,
        }
        details = {
            "initial_cap_support_gap_m": (
                initial["measurement"]["cap_support_gap_m"] if initial else None
            ),
            "stable_window_samples": stable_window_samples,
        }
    else:
        cover_rest_without_socket_entry = bool(window) and all(
            not row["geometry"]["through_selected_cover_hole"]
            and not row["geometry"]["entered_selected_casing_opening"]
            and row["measurement"]["shaft_radial_error_m"]
            > tolerances.max_shaft_radial_error_m
            and row["private_contact_evidence"]["cover_contact_count"] > 0
            and row["private_contact_evidence"]["casing_contact_count"] == 0
            for row in window
        )
        checks = {
            "cover_contact_away_from_selected_socket": cover_rest_without_socket_entry,
            "seat_ready_and_task_success_never_projected": no_public_success,
            "no_grasp_or_robot_support_proven": never_held_or_proven,
            "cover_rest_stable_for_window": low_motion,
        }
        details = {"stable_window_samples": stable_window_samples}

    return {
        "probe": probe,
        "probe_validation_passed": all(checks.values()),
        "checks": checks,
        "details": details,
        "task_success_claimed": False,
        "actual_task_episode": False,
    }


def _run_inside_container(args: argparse.Namespace) -> int:
    probe = getattr(args, "probe", None)
    if probe not in (None, *PROBE_RESET_OFFSETS_M):
        raise ValueError(f"unsupported negative probe: {probe!r}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_contacts.jsonl"
    evaluator_trace_path = output_dir / "evaluator_trace.jsonl"
    native_physics_trace_path = output_dir / "native_physics_trace.json"
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "run_kind": "bolt_negative_evaluator_probe_only" if probe else "bolt_seat_contact_sensor_calibration_only",
        "calibration_scope": (
            f"unheld {probe} gravity-only negative evaluator probe; not a task episode"
            if probe
            else "unheld nominal-seat sensor observations; not a task episode"
        ),
        "probe": probe,
        "probe_reset_offset_from_nominal_seat_m": (
            list(PROBE_RESET_OFFSETS_M[probe]) if probe else [0.0, 0.0, 0.0]
        ),
        "probe_reset_quaternion_wxyz": (
            list(PROBE_RESET_QUATERNIONS_WXYZ[probe])
            if probe in PROBE_RESET_QUATERNIONS_WXYZ
            else None
        ),
        "bolt_pose_writes_after_reset": False,
        "robot_action_during_probe": "zero action at each control step",
        "calibration_status": "raw_observations_unverified",
        "task_success_claimed": False,
        "actual_task_episode": False,
        "started_at_utc": _utc_now().isoformat(),
        "requested_warmup_samples": args.warmup_samples,
        "requested_measurement_samples": args.samples,
        "camera_rendering_enabled": bool(args.enable_cameras),
        "raw_contacts_path": str(raw_path),
        "evaluator_trace_path": str(evaluator_trace_path) if probe else None,
        "native_physics_trace_path": str(native_physics_trace_path),
        "stdout_path": None,
        "normal_direction_sign_calibrated": False,
        "contact_force_thresholds_calibrated": False,
        "penetration_tolerances_calibrated": False,
        "capacity_reached_or_truncation_possible": False,
    }
    env = None
    simulation_app = None
    exit_code = 0
    stats_by_phase: dict[str, dict[str, Any]] = {}
    sample_index = 0
    probe_rows: list[dict[str, Any]] = []

    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(args)
        simulation_app = launcher.app

        import torch

        from runtime.bolt_harness.env import BoltInsertionEnv
        from runtime.bolt_harness.env_cfg import BOLT_USD, make_env_cfg

        cfg = make_env_cfg(device=args.device)
        cfg.bolt_contact_max_data_count_per_prim = CONTACT_CAPACITY
        reset_position = tuple(cfg.bolt_seat_root_pos_m)
        if probe:
            offset = PROBE_RESET_OFFSETS_M[probe]
            reset_position = tuple(reset_position[index] + offset[index] for index in range(3))
        cfg.bolt_reset_pos_m = reset_position
        cfg.bolt_reset_quat_wxyz = tuple(
            PROBE_RESET_QUATERNIONS_WXYZ.get(probe, cfg.bolt_seat_root_quat_wxyz)
        )
        cfg.bolt.init_state.pos = cfg.bolt_reset_pos_m
        cfg.bolt.init_state.rot = cfg.bolt_reset_quat_wxyz
        if bool(cfg.bolt.spawn.rigid_props.disable_gravity):
            raise RuntimeError("seat calibration requires the bolt's configured gravity to remain enabled")

        env = BoltInsertionEnv(cfg)
        env.reset(seed=0)
        collision = env._scene_collision_info["bolt"]
        if collision["collision_approximation"] not in {
            "convexDecomposition",
            "sdf",
            "splitConvexHull",
        }:
            raise RuntimeError(f"unsupported bolt collider for representation comparison: {collision}")
        if not math.isclose(collision["contact_offset_m"], 0.00005, rel_tol=0.0, abs_tol=2e-9):
            raise RuntimeError(f"bolt contact offset changed from the existing 50 um setting: {collision}")
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        bolt_root = stage.GetPrimAtPath("/World/envs/env_0/Bolt")
        bolt_meshes = [prim for prim in Usd.PrimRange(bolt_root) if prim.IsA(UsdGeom.Mesh)]
        collider_offsets = []
        collision_mesh_metadata = []
        for mesh in bolt_meshes:
            physx_collision = PhysxSchema.PhysxCollisionAPI(mesh)
            collision_api = UsdPhysics.CollisionAPI(mesh)
            mesh_collision = UsdPhysics.MeshCollisionAPI(mesh)
            has_hull_api = mesh.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI)
            hull_vertex_limit = None
            if has_hull_api:
                hull_api = PhysxSchema.PhysxConvexHullCollisionAPI(mesh)
                hull_vertex_limit = hull_api.GetHullVertexLimitAttr().Get()
            points = UsdGeom.Mesh(mesh).GetPointsAttr().Get() or []
            collider_offsets.append(
                {
                    "prim_path": str(mesh.GetPath()),
                    "contact_offset_m": float(physx_collision.GetContactOffsetAttr().Get()),
                    "rest_offset_m": float(physx_collision.GetRestOffsetAttr().Get()),
                }
            )
            mesh_approximation = mesh_collision.GetApproximationAttr().Get()
            has_sdf_api = mesh.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
            sdf_values = {"resolution": None, "margin_m": None, "narrow_band_thickness_m": None}
            if has_sdf_api:
                sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI(mesh)
                sdf_values = {
                    "resolution": sdf_api.GetSdfResolutionAttr().Get(),
                    "margin_m": sdf_api.GetSdfMarginAttr().Get(),
                    "narrow_band_thickness_m": sdf_api.GetSdfNarrowBandThicknessAttr().Get(),
                }
            collision_mesh_metadata.append(
                {
                    "prim_path": str(mesh.GetPath()),
                    "collision_enabled": bool(collision_api.GetCollisionEnabledAttr().Get()),
                    "mesh_approximation": str(mesh_approximation),
                    "source_mesh_vertex_count": len(points),
                    "physx_convex_hull_api_applied": bool(has_hull_api),
                    "physx_hull_vertex_limit": hull_vertex_limit,
                    "source_hull_vertices_exceed_limit": (
                        len(points) > int(hull_vertex_limit) if hull_vertex_limit is not None else None
                    ),
                    "physx_sdf_api_applied": bool(has_sdf_api),
                    **sdf_values,
                }
            )
        if not collider_offsets or any(
            not math.isclose(item["contact_offset_m"], 0.00005, rel_tol=0.0, abs_tol=2e-9)
            or not math.isclose(item["rest_offset_m"], 0.0, rel_tol=0.0, abs_tol=2e-9)
            for item in collider_offsets
        ):
            raise RuntimeError(f"bolt collider offsets differ from the existing 50 um/zero settings: {collider_offsets}")
        if collision["collision_approximation"] == "sdf":
            if any(
                not item["physx_sdf_api_applied"]
                or item["resolution"] is None
                or int(item["resolution"]) != 640
                or item["margin_m"] is None
                or not math.isclose(float(item["margin_m"]), 0.0, rel_tol=0.0, abs_tol=1e-12)
                for item in collision_mesh_metadata
            ):
                raise RuntimeError(
                    "active bolt SDF does not match requested resolution=640/margin=0: "
                    f"{collision_mesh_metadata}"
                )
        if collision["collision_approximation"] == "splitConvexHull":
            split_info = collision.get("split_proxy_info") or {}
            active_paths = set(collision.get("active_collision_prim_paths") or ())
            pieces = split_info.get("pieces") or []
            if (
                collision.get("source_visual_collision_enabled") is not False
                or len(pieces) != 2
                or {piece.get("name") for piece in pieces} != {"shaft", "cap"}
                or {piece.get("prim_path") for piece in pieces} != active_paths
            ):
                raise RuntimeError(f"split bolt collision source/proxy map is incomplete: {collision}")
            active_metadata = {
                item["prim_path"]: item for item in collision_mesh_metadata if item["collision_enabled"]
            }
            for piece in pieces:
                active = active_metadata.get(piece["prim_path"])
                if (
                    active is None
                    or active["mesh_approximation"] != "convexHull"
                    or active["physx_hull_vertex_limit"] != 64
                    or not active["physx_convex_hull_api_applied"]
                ):
                    raise RuntimeError(
                        "split bolt collider is not active as a 64-vertex convex hull: "
                        f"piece={piece}, runtime={active}"
                    )
                piece["runtime_mesh_vertex_count"] = active["source_mesh_vertex_count"]
                piece["runtime_hull_vertex_limit"] = active["physx_hull_vertex_limit"]
                piece["runtime_source_vertices_exceed_limit"] = active["source_hull_vertices_exceed_limit"]

        source = env.get_private_bolt_contact_source()
        probe_adapter = probe_tolerances = probe_criteria = None
        if probe:
            probe_adapter, probe_tolerances, probe_criteria = _make_probe_adapter(env, source)
        body_metadata = _bolt_runtime_body_metadata(env, bolt_root)
        actual_masses = body_metadata["physx_view_masses_kg_raw"]
        if (
            not isinstance(actual_masses, list)
            or not actual_masses
            or not isinstance(actual_masses[0], list)
            or not actual_masses[0]
        ):
            raise RuntimeError(f"unexpected actual PhysX bolt mass tensor: {actual_masses!r}")
        source["bolt_mass_kg_actual"] = float(actual_masses[0][0])
        source["gravity_world_mps2"] = [float(value) for value in cfg.sim.gravity]
        filter_map = source["filter_map"]
        if int(source["max_contact_data_count_per_prim"]) != CONTACT_CAPACITY:
            raise RuntimeError("private contact view did not receive the requested 1024-point buffer")
        if int(source["contact_data_capacity"]) < CONTACT_CAPACITY:
            raise RuntimeError(f"unexpected private contact capacity: {source['contact_data_capacity']}")
        for phase in ("post_reset", "warmup", "measurement"):
            stats_by_phase[phase] = _new_filter_stats(filter_map)

        summary.update(
            {
                "device": args.device,
                "physics_dt_s": float(env.physics_dt),
                "control_period_s": float(env.physics_dt * env.cfg.decimation),
                "seat_reset_config": {
                    "root_position_m": list(cfg.bolt_reset_pos_m),
                    "root_quaternion_wxyz": list(cfg.bolt_reset_quat_wxyz),
                    "root_positive_y_axis_world": _bolt_axis_from_quaternion(
                        list(cfg.bolt_reset_quat_wxyz)
                    ),
                    "offset_from_nominal_seat_m": (
                        list(PROBE_RESET_OFFSETS_M[probe]) if probe else [0.0, 0.0, 0.0]
                    ),
                },
                "gravity_enabled": not bool(cfg.bolt.spawn.rigid_props.disable_gravity),
                "gravity_world_mps2": list(source["gravity_world_mps2"]),
                "expected_static_bolt_weight_n": source["bolt_mass_kg_actual"]
                * math.sqrt(sum(value * value for value in source["gravity_world_mps2"])),
                "bolt_mass_kg_configured": float(cfg.bolt_mass_kg),
                "bolt_mass_kg_physx_actual": source["bolt_mass_kg_actual"],
                "bolt_source_usd": str(BOLT_USD),
                "bolt_collision_info_runtime": collision,
                "bolt_runtime_contact_offsets": collider_offsets,
                "bolt_runtime_collision_mesh_metadata": collision_mesh_metadata,
                "bolt_runtime_body_metadata": body_metadata,
                "private_contact_source": {
                    "dt_s": float(source["dt_s"]),
                    "capacity": int(source["contact_data_capacity"]),
                    "max_contact_data_count_per_prim": int(source["max_contact_data_count_per_prim"]),
                    "sensor_body_names": list(source["sensor_body_names"]),
                    "filter_map": filter_map,
                },
                "evaluator_tolerances": (
                    asdict(probe_tolerances) if probe_tolerances is not None else None
                ),
                "evaluator_sampling_criteria": (
                    asdict(probe_criteria) if probe_criteria is not None else None
                ),
            }
        )

        trace_file = evaluator_trace_path.open("x", encoding="utf-8") if probe else None

        def record_sample(raw_file: Any, phase: str) -> None:
            nonlocal sample_index
            sample = _raw_contact_sample(env, source, sample_index=sample_index, phase=phase)
            raw_file.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
            raw_file.flush()
            _update_stats(stats_by_phase[phase], sample)
            summary["capacity_reached_or_truncation_possible"] |= sample["contact_buffer"][
                "capacity_reached_or_truncation_possible"
            ]
            if probe_adapter is not None and trace_file is not None:
                trace_row = _capture_evaluator_sample(
                    probe_adapter, phase=phase, sample_index=sample_index
                )
                trace_file.write(json.dumps(trace_row, sort_keys=True, allow_nan=False) + "\n")
                trace_file.flush()
                probe_rows.append(trace_row)
            sample_index += 1

        try:
            with raw_path.open("x", encoding="utf-8") as raw_file:
                record_sample(raw_file, "post_reset")

                zero_action = torch.zeros((1, 6), dtype=torch.float32, device=env.device)
                for phase, count in (("warmup", args.warmup_samples), ("measurement", args.samples)):
                    for _ in range(count):
                        _, _, terminated, truncated, _ = env.step(zero_action)
                        if bool(terminated.any()) or bool(truncated.any()):
                            raise RuntimeError(
                                "calibration environment ended unexpectedly; no task result is inferred"
                            )
                        record_sample(raw_file, phase)
        finally:
            if trace_file is not None:
                trace_file.close()

        summary["sample_count_including_post_reset"] = sample_index
        summary["filter_statistics_by_phase"] = {
            phase: _finalize_phase_stats(stats) for phase, stats in stats_by_phase.items()
        }
        summary["capture_complete"] = not summary["capacity_reached_or_truncation_possible"]
        summary["calibration_status"] = (
            "raw_observations_unverified"
            if summary["capture_complete"]
            else "raw_capture_capacity_saturated_unverified"
        )
        if probe:
            stable_window_samples = max(
                2, math.ceil(probe_tolerances.stable_dwell_s / summary["control_period_s"])
            )
            validation = _validate_probe(
                probe,
                probe_rows,
                stable_window_samples=stable_window_samples,
                tolerances=probe_tolerances,
            )
            summary["probe_validation"] = validation
            if not validation["probe_validation_passed"] and exit_code == 0:
                exit_code = 3
        summary["finished_at_utc"] = _utc_now().isoformat()
        if not summary["capture_complete"]:
            exit_code = 2
        print(
            "BOLT_CONTACT_CALIBRATION_ONLY="
            + json.dumps(
                {
                    "sample_count": sample_index,
                    "capture_complete": summary["capture_complete"],
                    "probe_validation_passed": summary.get("probe_validation", {}).get(
                        "probe_validation_passed"
                    ),
                    "task_success_claimed": False,
                    "raw_contacts_path": str(raw_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as exc:
        exit_code = 1
        summary["calibration_status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                native_trace = env.finalize_private_physics_trace()
                _write_json(native_physics_trace_path, _json_safe(native_trace))
                summary["native_physics_trace_sample_count"] = int(
                    native_trace["sample_count"]
                )
            except Exception:
                summary["native_physics_trace_error"] = traceback.format_exc()
                exit_code = exit_code or 1
        summary["inner_process_exit_code"] = exit_code
        summary["finished_at_utc"] = summary.get("finished_at_utc", _utc_now().isoformat())
        _write_json(summary_path, summary)
        if env is not None:
            try:
                env.close()
            except Exception:
                summary["env_close_error"] = traceback.format_exc()
                exit_code = exit_code or 1
        summary["inner_process_exit_code"] = exit_code
        _write_json(summary_path, summary)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                summary["simulation_app_close_error"] = traceback.format_exc()
                exit_code = exit_code or 1
    return exit_code


def _docker_preflight() -> dict[str, Any]:
    apps = subprocess.run(
        ["nvidia-smi", "-i", "2", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ids = subprocess.run(["docker", "ps", "-q"], check=True, capture_output=True, text=True).stdout.split()
    containers = []
    if ids:
        inspected = json.loads(
            subprocess.run(["docker", "inspect", *ids], check=True, capture_output=True, text=True).stdout
        )
        for item in inspected:
            requests = item.get("HostConfig", {}).get("DeviceRequests") or []
            requested_ids = sorted(
                {
                    device_id
                    for request in requests
                    for device_id in (request.get("DeviceIDs") or [])
                }
            )
            count_requested = any(int(request.get("Count", 0)) != 0 for request in requests)
            containers.append(
                {
                    "name": item["Name"].lstrip("/"),
                    "device_ids": requested_ids,
                    "ambiguous_gpu_request": bool(count_requested and not requested_ids),
                }
            )
    gpu2_claimants = [
        item
        for item in containers
        if "2" in item["device_ids"] or item["ambiguous_gpu_request"]
    ]
    if apps or gpu2_claimants:
        raise RuntimeError(
            f"GPU2 is not exclusively available: compute_apps={apps!r}, docker_claimants={gpu2_claimants!r}"
        )
    return {"gpu2_compute_apps": apps, "active_docker_containers": containers}


def _launch_docker(args: argparse.Namespace) -> int:
    run_id = _run_name(args.probe)
    output_dir = (ARTIFACT_ROOT / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = output_dir / "stdout.log"
    summary_path = output_dir / "summary.json"
    preflight = _docker_preflight()
    container_name = f"bolt-contact-calibration-{run_id}"
    in_container_script = CONTAINER_REPO / "tools/calibrate_bolt_contact.py"
    in_container_output = CONTAINER_REPO / output_dir.relative_to(REPO_ROOT)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--gpus=device=2",
        "--shm-size=16g",
        "-w",
        str(CONTAINER_REPO),
        "--entrypoint",
        "/isaac-sim/python.sh",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PRIVACY_CONSENT=Y",
        "-e",
        "OMNI_KIT_ACCEPT_EULA=YES",
        "-e",
        "OMNI_ENV_PRIVACY_CONSENT=YES",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        f"PYTHONPATH={CONTAINER_REPO}",
        "-v",
        f"{REPO_ROOT}:{CONTAINER_REPO}:rw",
        ISAAC_IMAGE,
        str(in_container_script),
        "--container-run",
        "--headless",
        "--device",
        "cuda:0",
        "--output-dir",
        str(in_container_output),
        "--samples",
        str(args.samples),
        "--warmup-samples",
        str(args.warmup_samples),
    ]
    if args.probe:
        command.extend(["--probe", args.probe])
    if args.enable_cameras:
        command.append("--enable_cameras")
    command_record = {
        "created_at_utc": _utc_now().isoformat(),
        "docker_argv": command,
        "docker_command_display": shlex.join(command),
        "container_name": container_name,
        "gpu_request": "device=2 only; in-container CUDA device is cuda:0",
        "repo_mount": f"{REPO_ROOT}:{CONTAINER_REPO}:rw",
        "roco_root_set": False,
        "external_roco_pythonpath_added": False,
        "camera_rendering_enabled": bool(args.enable_cameras),
        "probe": args.probe,
        "probe_reset_offset_from_nominal_seat_m": (
            list(PROBE_RESET_OFFSETS_M[args.probe]) if args.probe else [0.0, 0.0, 0.0]
        ),
        "probe_reset_quaternion_wxyz": (
            list(PROBE_RESET_QUATERNIONS_WXYZ[args.probe])
            if args.probe in PROBE_RESET_QUATERNIONS_WXYZ
            else None
        ),
        "gpu_preflight": preflight,
    }
    _write_json(output_dir / "command.json", command_record)
    _write_json(
        summary_path,
        {
            "run_kind": (
                "bolt_negative_evaluator_probe_only"
                if args.probe
                else "bolt_seat_contact_sensor_calibration_only"
            ),
            "calibration_scope": (
                f"unheld {args.probe} gravity-only negative evaluator probe; not a task episode"
                if args.probe
                else "unheld nominal-seat sensor observations; not a task episode"
            ),
            "calibration_status": "container_not_started",
            "task_success_claimed": False,
            "actual_task_episode": False,
            "probe": args.probe,
            "probe_reset_offset_from_nominal_seat_m": (
                list(PROBE_RESET_OFFSETS_M[args.probe]) if args.probe else [0.0, 0.0, 0.0]
            ),
            "probe_reset_quaternion_wxyz": (
                list(PROBE_RESET_QUATERNIONS_WXYZ[args.probe])
                if args.probe in PROBE_RESET_QUATERNIONS_WXYZ
                else None
            ),
            "bolt_pose_writes_after_reset": False,
            "robot_action_during_probe": "zero action at each control step",
            "output_dir": str(output_dir),
            "raw_contacts_path": str(output_dir / "raw_contacts.jsonl"),
            "evaluator_trace_path": (
                str(output_dir / "evaluator_trace.jsonl") if args.probe else None
            ),
            "stdout_path": str(stdout_path),
            "command_path": str(output_dir / "command.json"),
            "camera_rendering_enabled": bool(args.enable_cameras),
            "gpu_preflight": preflight,
        },
    )
    print(
        f"Launching GPU2 {args.probe or 'seat-only'} calibration; stdout: {stdout_path}",
        flush=True,
    )
    try:
        with stdout_path.open("wb") as stdout_file:
            process = subprocess.run(command, stdout=stdout_file, stderr=subprocess.STDOUT, check=False)
        process_exit = int(process.returncode)
    except Exception as exc:
        with stdout_path.open("ab") as stdout_file:
            stdout_file.write(f"Host launcher error: {type(exc).__name__}: {exc}\n".encode("utf-8"))
        process_exit = 127

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = {
            "run_kind": (
                "bolt_negative_evaluator_probe_only"
                if args.probe
                else "bolt_seat_contact_sensor_calibration_only"
            ),
            "calibration_scope": (
                f"unheld {args.probe} gravity-only negative evaluator probe; not a task episode"
                if args.probe
                else "unheld nominal-seat sensor observations; not a task episode"
            ),
            "calibration_status": "container_failed_before_summary",
            "task_success_claimed": False,
            "actual_task_episode": False,
            "probe": args.probe,
            "camera_rendering_enabled": bool(args.enable_cameras),
        }
    capture_complete = summary.get("capture_complete") is True
    inner_exit_code = summary.get("inner_process_exit_code")
    effective_exit = process_exit
    if effective_exit == 0 and isinstance(inner_exit_code, int) and inner_exit_code != 0:
        effective_exit = inner_exit_code
    if effective_exit == 0 and not capture_complete:
        effective_exit = 1
        if summary.get("calibration_status") not in {
            "failed",
            "raw_capture_capacity_saturated_unverified",
            "container_failed_before_summary",
        }:
            summary["calibration_status"] = "capture_incomplete"
    summary.update(
        {
            "output_dir": str(output_dir),
            "raw_contacts_path": str(output_dir / "raw_contacts.jsonl"),
            "stdout_path": str(stdout_path),
            "command_path": str(output_dir / "command.json"),
            "docker_process_exit_code": process_exit,
            "wrapper_exit_code": effective_exit,
            "exit_code_interpretation": (
                "negative evaluator probe capture completed only; not task success"
                if effective_exit == 0 and capture_complete
                else "raw probe capture completed, but validation failed; not task success"
                if capture_complete
                and summary.get("probe_validation", {}).get("probe_validation_passed") is False
                else "calibration process failed or capture was incomplete; not task success"
            ),
            "task_success_claimed": False,
            "actual_task_episode": False,
            "finished_at_utc": _utc_now().isoformat(),
        }
    )
    _write_json(summary_path, summary)
    print(
        f"GPU2 container exit={process_exit}; wrapper exit={effective_exit}; "
        f"summary: {summary_path}; raw: {output_dir / 'raw_contacts.jsonl'}",
        flush=True,
    )
    return effective_exit


def _host_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=240, help="post-warmup control-step samples")
    parser.add_argument("--warmup-samples", type=int, default=120)
    parser.add_argument("--probe", choices=tuple(PROBE_RESET_OFFSETS_M))
    parser.add_argument("--enable_cameras", action="store_true")
    return parser


def _container_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-run", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--warmup-samples", type=int, default=120)
    parser.add_argument("--probe", choices=tuple(PROBE_RESET_OFFSETS_M))
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0")
    return parser


def main() -> int:
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--container-run", action="store_true")
    mode, _ = mode_parser.parse_known_args()
    if mode.container_run:
        args = _container_parser().parse_args()
        if args.samples <= 0 or args.warmup_samples < 0:
            raise SystemExit("--samples must be positive and --warmup-samples non-negative")
        return _run_inside_container(args)

    args = _host_parser().parse_args()
    if args.samples <= 0 or args.warmup_samples < 0:
        raise SystemExit("--samples must be positive and --warmup-samples non-negative")
    try:
        return _launch_docker(args)
    except Exception as exc:
        print(f"Calibration launch refused/failed before container run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
