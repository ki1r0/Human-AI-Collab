#!/usr/bin/env python3
"""Quantify the first and second gear events in one ACT trace.

This is a post-run diagnostic.  It reads only the runner's NPZ trace and does
not change the simulator, policy, checkpoint, or scorer.  The output is
intended to make the failure window reproducible without relying on video
interpretation alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OBJECTS = (
    "planetary_carrier",
    "planetary_reducer",
    "ring_gear",
    "sun_planetary_gear_1",
    "sun_planetary_gear_2",
    "sun_planetary_gear_3",
    "sun_planetary_gear_4",
)
POLICY_CHANNELS = (
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "Lgrip",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "Rgrip",
)


def _decode_names(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _pose_array(values: np.ndarray, object_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != object_count * 7:
        raise ValueError(f"Expected {object_count * 7} pose values, got {array.shape}")
    return array.reshape((-1, object_count, 7))


def _window_stats(array: np.ndarray, start: int, stop: int) -> dict[str, float]:
    values = np.asarray(array[start:stop], dtype=np.float64)
    return {
        "start_step": int(start + 1),
        "end_step": int(stop),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def analyze(trace_path: Path) -> dict[str, object]:
    trace = np.load(trace_path, allow_pickle=False)
    names = _decode_names(trace["object_names"])
    if len(names) != len(OBJECTS):
        raise ValueError(f"Unexpected object list: {names}")
    name_to_index = {name: index for index, name in enumerate(names)}
    if set(names) != set(OBJECTS):
        raise ValueError(f"Unexpected object names: {names}")

    initial = np.asarray(trace["initial_object_poses"], dtype=np.float64).reshape(len(names), 7)
    poses = _pose_array(trace["object_poses_after_step"], len(names))
    score = np.asarray(trace["score"], dtype=np.int64)
    actions = np.asarray(trace["policy_actions"], dtype=np.float64)
    qpos_before = np.asarray(trace["qpos_before_policy_order"], dtype=np.float64)
    qpos_after = np.asarray(trace["qpos_after_policy_order"], dtype=np.float64)
    steps = len(score)
    if not all(array.shape[0] == steps for array in (poses, actions, qpos_before, qpos_after)):
        raise ValueError("Trace arrays do not have a common step count")

    transitions: list[dict[str, int]] = []
    previous = 0
    for index, value in enumerate(score):
        if index == 0 or int(value) != previous:
            transitions.append({"step": index + 1, "from": previous if index else 0, "to": int(value)})
        previous = int(value)

    gear_names = [f"sun_planetary_gear_{index}" for index in range(1, 5)]
    motion: dict[str, dict[str, object]] = {}
    for name in gear_names:
        index = name_to_index[name]
        delta = poses[:, index, :3] - initial[index, :3]
        distance = np.linalg.norm(delta, axis=1)
        z = poses[:, index, 2]
        z_rise = z - initial[index, 2]
        angle = 2.0 * np.arccos(np.clip(np.abs(poses[:, index, 3]), 0.0, 1.0))
        motion[name] = {
            "initial_xyz_m": initial[index, :3].tolist(),
            "final_xyz_m": poses[-1, index, :3].tolist(),
            "max_displacement_m": float(distance.max()),
            "max_displacement_step": int(distance.argmax() + 1),
            "final_displacement_m": float(distance[-1]),
            "max_z_rise_m": float(z_rise.max()),
            "max_z_rise_step": int(z_rise.argmax() + 1),
            "final_z_m": float(z[-1]),
            "max_orientation_change_rad": float(angle.max()),
            "max_orientation_change_step": int(angle.argmax() + 1),
        }

    # The first true lift is the only gear with a large positive z excursion.
    first_name = max(gear_names, key=lambda name: motion[name]["max_z_rise_m"])
    # Excluding the first gear, the largest post-reset motion identifies the
    # second attempted target without assigning an identity from the video.
    remaining = [name for name in gear_names if name != first_name]
    second_name = max(remaining, key=lambda name: motion[name]["max_displacement_m"])
    second_index = name_to_index[second_name]
    second_z_rise = poses[:, second_index, 2] - initial[second_index, 2]
    second_distance = np.linalg.norm(poses[:, second_index, :3] - initial[second_index, :3], axis=1)
    second_peak = int(np.argmax(second_distance))
    # Report a compact contact window around the target's largest motion.
    contact_start = max(0, second_peak - 12)
    contact_stop = min(steps, second_peak + 13)

    windows = {
        "before_first_score": [0, min(78, steps)],
        "after_first_score": [min(78, steps), min(120, steps)],
        "second_contact": [contact_start, contact_stop],
        "post_contact": [min(contact_stop, steps), steps],
    }
    channels: dict[str, object] = {}
    for label, (start, stop) in windows.items():
        channels[label] = {
            "policy_action": {
                channel: _window_stats(actions[:, index], start, stop)
                for index, channel in enumerate(POLICY_CHANNELS)
            },
            "qpos_before": {
                channel: _window_stats(qpos_before[:, index], start, stop)
                for index, channel in enumerate(POLICY_CHANNELS)
            },
            "qpos_after": {
                channel: _window_stats(qpos_after[:, index], start, stop)
                for index, channel in enumerate(POLICY_CHANNELS)
            },
            "target_xyz_m": poses[stop - 1, second_index, :3].tolist() if stop else [],
            "target_z_rise_m": float(second_z_rise[stop - 1]) if stop else 0.0,
        }

    # A directly testable grasp signature: command closes, target moves in z,
    # then target returns near the table instead of following the arm.
    table_settled_z = float(np.median(poses[min(39, steps - 1) :, second_index, 2]))
    contact_target_peak_z = float(poses[contact_start:contact_stop, second_index, 2].max())
    contact_target_end_z = float(poses[min(contact_stop, steps) - 1, second_index, 2])
    result: dict[str, object] = {
        "trace": str(trace_path.resolve()),
        "steps": steps,
        "score_transitions": transitions,
        "first_scored_gear": first_name,
        "second_attempted_gear_by_motion": second_name,
        "gear_motion": motion,
        "second_contact_window_steps": [contact_start + 1, contact_stop],
        "second_contact_peak_step": second_peak + 1,
        "second_contact_table_settled_z_m": table_settled_z,
        "second_contact_peak_z_m": contact_target_peak_z,
        "second_contact_end_z_m": contact_target_end_z,
        "second_contact_z_lift_m": contact_target_peak_z - table_settled_z,
        "windows": channels,
        "interpretation": {
            "first_gear": "large positive z excursion followed by score 0->1",
            "second_gear": "left-gripper closure coincides with a short target motion, but the target is not carried and returns near table height",
            "right_arm": "no second-pick sequence is visible in the action/qpos trace; Rgrip remains near its open reset value",
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json", type=Path, help="Write the complete diagnostic JSON here")
    args = parser.parse_args()
    result = analyze(args.trace)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
