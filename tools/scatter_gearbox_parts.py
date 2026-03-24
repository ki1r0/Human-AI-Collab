#!/usr/bin/env python3
"""Scatter gearbox parts in the room with hardcoded transforms.

Usage:
  ./isaaclab.sh -p /absolute/path/to/tools/scatter_gearbox_parts.py
  ./isaaclab.sh -p /absolute/path/to/tools/scatter_gearbox_parts.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()



from pxr import Gf, Usd, UsdGeom  # type: ignore


# Hardcoded scene layout (meters, degrees)
PART_LAYOUT = {
    "Orange_01": {"t": (-0.85, 0.55, 1.95), "r": (0.0, 0.0, 0.0)},
    "plasticpail_a01": {"t": (0.75, -0.55, 0.88), "r": (0.0, 0.0, -15.0)},
    "utilitybucket_a01": {"t": (-0.75, -0.55, 0.90), "r": (0.0, 0.0, 20.0)},
    "Casing_Top": {"t": (0.85, 0.55, 1.75), "r": (90.0, 10.0, 20.0)},
    "Casing_Base": {"t": (-0.85, 0.15, 1.55), "r": (90.0, -20.0, -35.0)},
    "Hub_Cover_Output": {"t": (0.45, 0.35, 1.30), "r": (90.0, 0.0, 45.0)},
    "Hub_Cover_Input": {"t": (-0.45, 0.35, 1.32), "r": (90.0, 0.0, -30.0)},
    "Hub_Cover_Small": {"t": (0.00, 0.55, 1.62), "r": (90.0, 15.0, 20.0)},
    "Input_Shaft": {"t": (-0.25, -0.15, 1.85), "r": (90.0, 0.0, 90.0)},
    "Output_Shaft": {"t": (0.25, -0.15, 1.82), "r": (90.0, 0.0, 0.0)},
    "Transfer_Shaft": {"t": (0.65, 0.10, 1.50), "r": (90.0, 90.0, 0.0)},
    "Output_Gear": {"t": (-0.65, 0.10, 1.45), "r": (90.0, 0.0, 0.0)},
    "Transfer_Gear": {"t": (0.00, -0.45, 1.35), "r": (90.0, 0.0, 0.0)},
    "M6_Hub_Bolt": {"t": (0.55, -0.05, 1.95), "r": (90.0, 0.0, 0.0)},
    "M10_Casing_Bolt": {"t": (-0.55, -0.05, 1.95), "r": (90.0, 0.0, 0.0)},
    "M10_Casing_Bolt_01": {"t": (0.20, 0.10, 1.95), "r": (90.0, 25.0, 0.0)},
    "M10_Casing_Nut": {"t": (-0.20, 0.10, 1.95), "r": (90.0, 0.0, 0.0)},
    "Oil_Level_Indicator": {"t": (0.00, -0.05, 1.10), "r": (90.0, 0.0, 0.0)},
    "Breather_Plug": {"t": (0.00, 0.25, 1.95), "r": (90.0, 0.0, 0.0)},
}


def _get_existing_scale(prim) -> Gf.Vec3f:
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            value = op.Get()
            return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))
    return Gf.Vec3f(0.005, 0.005, 0.005)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default=str(Path(_REPO_ROOT) / "assets" / "simple_room_scene.usd"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        print(f"[ERROR] Failed to open scene: {args.scene}")
        return 1

    updated = 0
    missing = []
    for part, target in PART_LAYOUT.items():
        prim_path = f"/Root/{part}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            missing.append(prim_path)
            continue

        t = target["t"]
        r = target["r"]
        s = _get_existing_scale(prim)

        print(
            f"{'DRY' if args.dry_run else 'SET'} {prim_path} "
            f"translate={t} rotateXYZ={r} scale=({s[0]:.6f},{s[1]:.6f},{s[2]:.6f})"
        )
        if not args.dry_run:
            common = UsdGeom.XformCommonAPI(prim)
            common.SetXformVectors(
                Gf.Vec3d(*t),
                Gf.Vec3f(*r),
                s,
                Gf.Vec3f(0.0, 0.0, 0.0),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
                Usd.TimeCode.Default(),
            )
        updated += 1

    if missing:
        print("[WARN] Missing prims:")
        for p in missing:
            print(f"  {p}")

    if not args.dry_run:
        stage.GetRootLayer().Save()
        print(f"[OK] Updated {updated} parts and saved: {args.scene}")
    else:
        print(f"[OK] Dry-run complete for {updated} parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
