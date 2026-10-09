"""Quaternion math and private terminal scoring; only math is used online."""

import math


def rotate(q, vector):
    w, x, y, z = q
    norm = math.sqrt(sum(v * v for v in q))
    if not norm:
        raise ValueError("zero quaternion")
    w, x, y, z = (v / norm for v in q)
    vx, vy, vz = vector
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx]


def conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]


def record_isaac_sample(env, task):
    """Private sampler: results are never consulted by commands, monitors or guards."""
    import torch
    roots = env.root_states()
    hub, casing = roots["hub"][0], roots["casing"][0]
    sensor = env.hub_contact
    matrix = getattr(sensor.data, "force_matrix_w", None)
    support_contact = bool(torch.linalg.vector_norm(matrix).item() > 1e-3) if matrix is not None and matrix.numel() else None
    penetration = None
    view = getattr(sensor, "contact_physx_view", None)
    if view is not None:
        try:
            _, _, _, distances, counts, _ = view.get_contact_data(dt=float(env.cfg.sim_dt))
            count = int(counts.reshape(-1)[0].item())
            if count:
                values = distances.reshape(-1)[:count]
                values = values[torch.isfinite(values)]
                if values.numel():
                    penetration = max(0.0, -float(values.min().item()))
        except (RuntimeError, AttributeError):
            pass
    limit = task["sensors"]["released_opening_threshold_m"]
    openings = [float(env.robot.data.joint_pos[0, grip.joint_ids].abs().mean().item())
                for grip in (env.left_gripper_cfg, env.right_gripper_cfg)]
    finger_matrices = [getattr(getattr(getattr(env, f"{side}_gripper_link{i}_contact", None), "data", None),
                               "force_matrix_w", None) for side in ("left", "right") for i in (1, 2)]
    released = None
    if limit is not None and all(value is not None and value.numel() for value in finger_matrices):
        released = min(openings) >= limit and all(float(torch.linalg.vector_norm(value).item()) < 1e-3 for value in finger_matrices)
    return {"hub_xyz": hub[:3].detach().cpu().tolist(), "hub_quat": hub[3:7].detach().cpu().tolist(),
            "casing_xyz": casing[:3].detach().cpu().tolist(), "casing_quat": casing[3:7].detach().cpu().tolist(),
            "speed_mps": float(torch.linalg.vector_norm(hub[7:10]).item()),
            "support_contact": support_contact, "penetration_m": penetration,
            "released": released, "dt_s": float(env.cfg.sim_dt)}


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def judge(samples, task):
    config = task["evaluation"]
    if not samples:
        return {"seat_gt": "UNKNOWN", "registration_gt": "UNKNOWN", "reason": "no_private_samples",
                "physical_validation": False}
    if task["frames"]["socket_axis_local"] is None or task["frames"]["plug_axis_local"] is None:
        return {"seat_gt": "UNKNOWN", "registration_gt": "UNKNOWN",
                "reason": "insertion_axes_not_certified", "physical_validation": False}
    required = {"hub_xyz", "hub_quat", "casing_xyz", "casing_quat", "speed_mps",
                "released", "support_contact", "penetration_m", "dt_s"}
    if any(not required <= sample.keys() for sample in samples):
        return {"seat_gt": "UNKNOWN", "registration_gt": "UNKNOWN",
                "reason": "incomplete_private_samples", "physical_validation": False}
    last = samples[-1]
    casing_q = last["casing_quat"]
    offset = rotate(conjugate(casing_q), [a-b for a, b in zip(last["hub_xyz"], last["casing_xyz"])])
    error = [a-b for a, b in zip(offset, task["frames"]["seated_relative_position_m"])]
    axis = task["frames"]["socket_axis_local"]
    axis_norm = math.sqrt(sum(v*v for v in axis))
    axis = [v / axis_norm for v in axis]
    axial = sum(a*b for a, b in zip(error, axis))
    radial_vector = [v - axial*a for v, a in zip(error, axis)]
    radial = math.sqrt(sum(v*v for v in radial_vector))
    local_q = multiply(conjugate(casing_q), last["hub_quat"])
    target_q = task["frames"]["seated_relative_quat_wxyz"]
    orientation_error_q = multiply(conjugate(target_q), local_q)
    if orientation_error_q[0] < 0:
        orientation_error_q = [-value for value in orientation_error_q]
    orientation_dot = abs(sum(a*b for a, b in zip(local_q, target_q)))
    orientation_norm = math.sqrt(sum(v*v for v in local_q) * sum(v*v for v in target_q))
    orientation_error = (math.degrees(2 * math.acos(max(-1, min(1, orientation_dot / orientation_norm))))
                         if orientation_norm > 1e-12 else None)
    hub_axis = rotate(local_q, task["frames"]["plug_axis_local"])
    tilt = math.degrees(math.acos(max(-1, min(1, sum(a*b for a, b in zip(hub_axis, axis))))))
    dwell = 0.0
    complete_signals = True
    for sample in reversed(samples):
        required = (sample["released"], sample["support_contact"], sample["penetration_m"], sample["speed_mps"])
        if any(value is None for value in required):
            complete_signals = False
            break
        if (not sample["released"] or not sample["support_contact"] or
                sample["speed_mps"] > config["settle_speed_mps"] or
                sample["penetration_m"] > config["max_penetration_m"]):
            break
        sample_offset = rotate(conjugate(sample["casing_quat"]), [a-b for a, b in
                               zip(sample["hub_xyz"], sample["casing_xyz"])])
        sample_error = [a-b for a, b in zip(sample_offset, task["frames"]["seated_relative_position_m"])]
        sample_axial = sum(a*b for a, b in zip(sample_error, axis))
        sample_radial = math.sqrt(max(0, sum(v*v for v in sample_error)-sample_axial*sample_axial))
        sample_axis = rotate(multiply(conjugate(sample["casing_quat"]), sample["hub_quat"]),
                             task["frames"]["plug_axis_local"])
        sample_tilt = math.degrees(math.acos(max(-1, min(1, sum(a*b for a, b in zip(sample_axis, axis))))))
        if (abs(sample_axial) > config["axial_tolerance_m"] or sample_radial > config["radial_tolerance_m"]
                or sample_tilt > config["tilt_tolerance_deg"]):
            break
        dwell += sample["dt_s"]
    passed = (abs(axial) <= config["axial_tolerance_m"] and radial <= config["radial_tolerance_m"]
              and tilt <= config["tilt_tolerance_deg"] and dwell >= config["settle_window_s"])
    calibrated = config["certified"] and task["frames"]["certified"]
    seat = ("PASS" if passed else "FAIL") if calibrated and complete_signals else "UNKNOWN"
    symmetry = task["frames"].get("registration_symmetry_quats")
    registration_error = None
    registration = "UNKNOWN"
    if symmetry and config.get("registration_certified"):
        target = task["frames"]["seated_relative_quat_wxyz"]
        registration_error = min(math.degrees(2 * math.acos(max(-1, min(1, abs(sum(
            a*b for a, b in zip(local_q, multiply(target, q)))))))) for q in symmetry)
        registration = "PASS" if registration_error <= config["registration_tolerance_deg"] else "FAIL"
    return {"seat_gt": seat, "registration_gt": registration,
            "geometry": {"axial_error_m": axial, "radial_error_m": radial, "tilt_deg": tilt,
                         # Evaluator-only post-run residuals; never included in observations or commands.
                         "position_error_casing_local_m": error,
                         "radial_error_vector_casing_local_m": radial_vector,
                         "orientation_error_quat_casing_local_wxyz": orientation_error_q,
                         "full_orientation_error_deg": orientation_error,
                         "stable_released_dwell_s": dwell, "registration_error_deg": registration_error},
            "physical_validation": bool(calibrated and complete_signals),
            "reason": "calibrated" if calibrated else "provisional_frames_or_tolerances"}
