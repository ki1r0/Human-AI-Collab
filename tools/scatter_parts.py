#!/usr/bin/env python3
"""Scatter gearbox parts across the table and floor so they don't overlap.

Modifies ONLY the xformOp:translate attribute of each part prim in the scene file.
Keeps rotation, scale, and pivot ops unchanged.

Table surface is at approximately Z ≈ 2.35 (based on Orange_01 at Z=2.39).
Floor is at approximately Z ≈ 0.0 (based on pail/bucket positions).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Usd, UsdGeom, Gf  # noqa: E402

# ---------------------------------------------------------------------------
# Hardcoded layout — parts scattered across the table with generous spacing.
#
# Coordinate system (Z-up):
#   Table surface ≈ Z = 2.40  (Orange_01 sits at 2.39)
#   Floor         ≈ Z = 0.02  (pails/buckets sit at 0.02–0.04)
#   Table center  ≈ (X=0, Y=0)
#   Table extent  ≈ X: [-0.50, 0.50],  Y: [-0.70, 0.70]
#
# Layout: 4-column grid on the table, well-spaced.
# Large parts at the edges, small parts in the middle.
# ---------------------------------------------------------------------------

TABLE_Z = 2.40   # just above the table surface
FLOOR_Z = 0.02   # just above the floor

# Room extents (approx): X: [-2.5, 2.5],  Y: [-2.5, 2.5]
# Table center ≈ (0, 0), table extent ≈ X: [-0.50, 0.50], Y: [-0.70, 0.70]
# Parts at scale 0.002 are smaller in scene, but we keep generous spacing.

PART_POSITIONS = {
    # --- Large casings: far apart on floor ---
    "Casing_Base":        (-1.80,  1.50, FLOOR_Z),   # floor, back-left corner
    "Casing_Top":         ( 1.80,  1.50, FLOOR_Z),   # floor, back-right corner

    # --- Gears: on table, well separated ---
    "Output_Gear":        (-0.40,  0.50, TABLE_Z),   # table back-left
    "Transfer_Gear":      ( 0.40, -0.50, TABLE_Z),   # table front-right

    # --- Hub covers: mix of table and floor ---
    "Hub_Cover_Input":    (-1.80, -1.50, FLOOR_Z),   # floor, front-left corner
    "Hub_Cover_Output":   ( 1.80, -1.50, FLOOR_Z),   # floor, front-right corner
    "Hub_Cover_Small":    (-1.00, -1.50, FLOOR_Z),   # floor, front-center-left

    # --- Shafts: floor, spread along walls ---
    "Output_Shaft":       ( 1.80,  0.00, FLOOR_Z),   # floor, right wall
    "Input_Shaft":        (-1.80,  0.00, FLOOR_Z),   # floor, left wall
    "Transfer_Shaft":     ( 0.00,  1.80, FLOOR_Z),   # floor, back wall

    # --- Small parts: scattered on table and nearby floor ---
    "M10_Casing_Bolt":    (-0.40, -0.50, TABLE_Z),   # table front-left
    "M10_Casing_Bolt_01": ( 0.40,  0.50, TABLE_Z),   # table back-right
    "M10_Casing_Nut":     ( 0.00, -1.80, FLOOR_Z),   # floor, front wall
    "M6_Hub_Bolt":        (-1.00,  0.80, FLOOR_Z),   # floor, mid-left
    "Breather_Plug":      ( 1.00, -0.80, FLOOR_Z),   # floor, mid-right
    "Oil_Level_Indicator":( 0.00,  0.00, TABLE_Z),    # table center
}


def scatter(scene_path: str, dry_run: bool = False) -> int:
    stage = Usd.Stage.Open(scene_path)
    dp = stage.GetDefaultPrim()
    if dp is None or not dp.IsValid():
        print("[ERROR] No defaultPrim")
        return 1

    tc = Usd.TimeCode.Default()
    n_moved = 0

    for child in dp.GetChildren():
        name = child.GetName()
        if name not in PART_POSITIONS:
            continue
        new_pos = PART_POSITIONS[name]

        xf = UsdGeom.Xformable(child)
        ops = xf.GetOrderedXformOps()
        if not ops:
            print(f"  [SKIP] {name}: no xformOps")
            continue

        # Find the translate op (first op should be translate)
        translate_op = None
        for op in ops:
            if op.GetOpName() == "xformOp:translate" and "pivot" not in op.GetOpName():
                translate_op = op
                break

        if translate_op is None:
            print(f"  [SKIP] {name}: no translate op found")
            continue

        old_pos = translate_op.Get(tc)
        if dry_run:
            print(f"  [DRY] {name}: ({old_pos[0]:.3f}, {old_pos[1]:.3f}, {old_pos[2]:.3f}) -> "
                  f"({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})")
        else:
            translate_op.Set(Gf.Vec3d(*new_pos))
            print(f"  [SET] {name}: ({old_pos[0]:.3f}, {old_pos[1]:.3f}, {old_pos[2]:.3f}) -> "
                  f"({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})")
        n_moved += 1

    if not dry_run and n_moved > 0:
        stage.GetRootLayer().Save()

    print(f"\n  {n_moved} parts {'would be' if dry_run else ''} repositioned.")
    return 0


def verify(scene_path: str) -> int:
    """Verify no two parts overlap (center-to-center distance check)."""
    stage = Usd.Stage.Open(scene_path)
    dp = stage.GetDefaultPrim()
    tc = Usd.TimeCode.Default()

    positions = {}
    for child in dp.GetChildren():
        name = child.GetName()
        if name not in PART_POSITIONS:
            continue
        xf = UsdGeom.Xformable(child)
        ops = xf.GetOrderedXformOps()
        for op in ops:
            if op.GetOpName() == "xformOp:translate" and "pivot" not in op.GetOpName():
                pos = op.Get(tc)
                positions[name] = (pos[0], pos[1], pos[2])
                break

    # Check pairwise distances
    MIN_DISTANCE = 0.50  # 50cm minimum between part centers
    names = list(positions.keys())
    n_pass = 0
    n_fail = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = positions[a], positions[b]
            dx = pa[0] - pb[0]
            dy = pa[1] - pb[1]
            dz = pa[2] - pb[2]
            dist = (dx**2 + dy**2 + dz**2) ** 0.5
            if dist < MIN_DISTANCE:
                print(f"  [FAIL] {a} <-> {b}: distance = {dist:.3f}m (min {MIN_DISTANCE}m)")
                n_fail += 1
            else:
                n_pass += 1

    # Check positions match expected values
    n_pos_pass = 0
    n_pos_fail = 0
    for name, expected in PART_POSITIONS.items():
        if name not in positions:
            print(f"  [FAIL] {name}: not found in scene")
            n_pos_fail += 1
            continue
        actual = positions[name]
        if (abs(actual[0] - expected[0]) > 0.001 or
            abs(actual[1] - expected[1]) > 0.001 or
            abs(actual[2] - expected[2]) > 0.001):
            print(f"  [FAIL] {name}: position mismatch actual={actual} expected={expected}")
            n_pos_fail += 1
        else:
            n_pos_pass += 1

    total_checks = n_pass + n_fail + n_pos_pass + n_pos_fail
    total_pass = n_pass + n_pos_pass
    total_fail = n_fail + n_pos_fail

    print(f"\n{'='*60}")
    print(f"  Scatter verification: {total_pass}/{total_checks} passed  ({total_fail} failed)")
    print(f"    Distance checks: {n_pass} pass, {n_fail} fail")
    print(f"    Position checks: {n_pos_pass} pass, {n_pos_fail} fail")
    print(f"{'='*60}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Only verify, don't modify.")
    parser.add_argument("--scene", default=os.path.join(_REPO_ROOT, "assets", "simple_room_scene.usd"))
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Scatter Parts — spread gearbox parts across the table")
    print(f"  Scene: {os.path.basename(args.scene)}")
    print(f"{'='*60}\n")

    if args.verify:
        sys.exit(verify(args.scene))
    else:
        rc = scatter(args.scene, dry_run=args.dry_run)
        if rc == 0 and not args.dry_run:
            print("\nRunning verification...\n")
            sys.exit(verify(args.scene))
        sys.exit(rc)
