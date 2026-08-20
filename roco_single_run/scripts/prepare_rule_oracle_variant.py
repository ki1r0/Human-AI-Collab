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
        policy_path.write_text(source, encoding="utf-8")
        status = "patched"

    result = {
        "checkout": str(checkout),
        "commit": head,
        "policy_path": str(policy_path),
        "ring_rotation_deg": args.ring_rotation_deg,
        "status": status,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
