#!/usr/bin/env python3
"""Apply or remove one audited rule-oracle ring-rotation variant."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PINNED_COMMIT = "094a1f76d18c207caec198315f23b1a60dbca94f"
OFFICIAL_ROTATION = """        if gear_id == 4:
            rot_deg = 60
        else:
            rot_deg = 30"""
ZERO_RING_ROTATION = """        if gear_id == 4:
            rot_deg = 60
        elif gear_id == 5:
            rot_deg = 0
        else:
            rot_deg = 30"""
OFFICIAL_REDUCER_HEIGHT = """        elif gear_id == 6: # Reducer
            root_state = self.sun_planetary_gear_4.data.root_state_w.clone()
            if self.count == count_step[0]:
                self.current_target_position = root_state[:, :3].clone()
                self.current_target_orientation = root_state[:, 3:7].clone()
            obj_height_offset = 0.023 + 0.02
            mount_height_offset = 0.025"""
RAISED_REDUCER_HEIGHT = """        elif gear_id == 6: # Reducer
            root_state = self.sun_planetary_gear_4.data.root_state_w.clone()
            if self.count == count_step[0]:
                self.current_target_position = root_state[:, :3].clone()
                self.current_target_orientation = root_state[:, 3:7].clone()
            obj_height_offset = 0.023 + 0.02
            mount_height_offset = 0.030"""


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--ring-rotation-deg", type=int, choices=(0, 30), required=True)
    parser.add_argument(
        "--reducer-mount-height-m",
        type=float,
        choices=(0.025, 0.030),
        default=0.025,
    )
    args = parser.parse_args()

    checkout = args.checkout.resolve()
    head = _git_head(checkout)
    if head != PINNED_COMMIT:
        raise RuntimeError(f"Expected RoCo commit {PINNED_COMMIT}, found {head}")

    policy_path = (
        checkout
        / "source"
        / "Galaxea_Lab_External"
        / "Galaxea_Lab_External"
        / "robots"
        / "galaxea_rule_policy.py"
    )
    source = policy_path.read_text(encoding="utf-8")
    official_count = source.count(OFFICIAL_ROTATION)
    zero_count = source.count(ZERO_RING_ROTATION)
    if (official_count, zero_count) not in ((1, 0), (0, 1)):
        raise RuntimeError(
            "Unexpected ring rotation source layout: "
            f"official={official_count}, zero_variant={zero_count}"
        )

    target = ZERO_RING_ROTATION if args.ring_rotation_deg == 0 else OFFICIAL_ROTATION
    replacement = OFFICIAL_ROTATION if args.ring_rotation_deg == 0 else ZERO_RING_ROTATION
    status = "already-selected"
    if source.count(target) != 1:
        source = source.replace(replacement, target)
        status = "patched"

    official_height_count = source.count(OFFICIAL_REDUCER_HEIGHT)
    raised_height_count = source.count(RAISED_REDUCER_HEIGHT)
    if (official_height_count, raised_height_count) not in ((1, 0), (0, 1)):
        raise RuntimeError(
            "Unexpected reducer mount-height source layout: "
            f"official={official_height_count}, raised_variant={raised_height_count}"
        )
    height_target = (
        RAISED_REDUCER_HEIGHT
        if args.reducer_mount_height_m == 0.030
        else OFFICIAL_REDUCER_HEIGHT
    )
    height_replacement = (
        OFFICIAL_REDUCER_HEIGHT
        if args.reducer_mount_height_m == 0.030
        else RAISED_REDUCER_HEIGHT
    )
    if source.count(height_target) != 1:
        source = source.replace(height_replacement, height_target)
        status = "patched"

    policy_path.write_text(source, encoding="utf-8")

    result = {
        "checkout": str(checkout),
        "commit": head,
        "policy_path": str(policy_path),
        "ring_rotation_deg": args.ring_rotation_deg,
        "reducer_mount_height_m": args.reducer_mount_height_m,
        "status": status,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
