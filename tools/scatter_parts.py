#!/usr/bin/env python3
"""Scatter gearbox parts across the table and floor so they don't overlap.

Run via the repo wrapper (handles pxr / LD_LIBRARY_PATH / launcher discovery):

    docker compose exec hac tools/run_tool.sh tools/scatter_parts.py --dry-run
    docker compose exec hac tools/run_tool.sh tools/scatter_parts.py
    docker compose exec hac tools/run_tool.sh tools/scatter_parts.py --verify

The wrapper is required because `pxr` ships as the `omni.usd.libs` Omniverse
extension; it is only on PYTHONPATH after a Kit app starts. This script is a
pure offline USD edit (no Kit) so it needs `tools/run_tool.sh` to set
PYTHONPATH and LD_LIBRARY_PATH before launching Isaac Sim's Python.

Per-part scatter spec lives in assembly/scatter_layout.yaml (override with
--layout). Each entry under `parts:` is one of:
  - bbox-contact (preferred):
        {xy: [x, y], surface: table|floor, rot: [rx, ry, rz], z: <override>}
        Rotation (if any) is applied first, then Z is derived so the world-AABB
        bottom rests on <surface>_surface_z + spawn_margin, independent of the
        part's geometry pivot. This is the Phase 0 hover fix / diversity path.
  - legacy absolute:
        {pos: [x, y, z]}
        Sets xformOp:translate directly; rotation preserved.

Surface constants (table_surface_z, floor_surface_z, spawn_margin, and the
legacy table_z/floor_z) also live in that YAML under `surfaces:`.
"""

import os
import sys

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Usd, UsdGeom, Gf  # noqa: E402

# Default scatter-layout spec (positions + surface constants live here, not in code).
DEFAULT_LAYOUT_PATH = os.path.join(_REPO_ROOT, "assembly", "scatter_layout_007050.yaml")

SPAWN_MARGIN = 0.005  # default clearance above the surface; overridden by the layout


def load_layout(path: str):
    """Load the scatter-layout YAML. Returns (surfaces: dict, parts: dict).

    surfaces holds table_surface_z / floor_surface_z / spawn_margin (and legacy
    table_z / floor_z). parts maps part name -> entry (bbox-contact or pos form).
    """
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    surfaces = data.get("surfaces", {}) or {}
    parts = data.get("parts", {}) or {}
    if not parts:
        raise ValueError(f"scatter layout {path!r} has no 'parts:' entries")
    return surfaces, parts


def _is_dict_spec(entry) -> bool:
    """A bbox-contact / explicit entry (has xy/surface/rot/z), not a legacy pos."""
    return isinstance(entry, dict) and "pos" not in entry


def _find_op(xf, op_type):
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == op_type and "pivot" not in op.GetOpName():
            return op
    return None


def _surface_z(surface_name: str, surfaces: dict) -> float:
    if surface_name == "table":
        return float(surfaces.get("table_surface_z", -0.1))
    return float(surfaces.get("floor_surface_z", 0.0))


def _world_bbox_min_z(prim, tc) -> float:
    cache = UsdGeom.BBoxCache(tc, includedPurposes=[UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(prim)
    aabb = bbox.ComputeAlignedRange()
    return float(aabb.GetMin()[2])


def _set_rotate_xyz(xf, rot_xyz):
    op = _find_op(xf, UsdGeom.XformOp.TypeRotateXYZ)
    if op is None:
        op = xf.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
    op.Set(Gf.Vec3f(*rot_xyz))
    return op


def scatter(scene_path: str, surfaces: dict, parts: dict, dry_run: bool = False) -> int:
    spawn_margin = float(surfaces.get("spawn_margin", SPAWN_MARGIN))
    stage = Usd.Stage.Open(scene_path)
    dp = stage.GetDefaultPrim()
    if dp is None or not dp.IsValid():
        print("[ERROR] No defaultPrim")
        return 1

    tc = Usd.TimeCode.Default()
    n_moved = 0

    for child in dp.GetChildren():
        name = child.GetName()
        if name not in parts:
            continue
        entry = parts[name]

        xf = UsdGeom.Xformable(child)
        translate_op = _find_op(xf, UsdGeom.XformOp.TypeTranslate)
        if translate_op is None:
            print(f"  [SKIP] {name}: no translate op found")
            continue

        old_pos = translate_op.Get(tc)
        rot_msg = ""

        if _is_dict_spec(entry):
            x, y = entry["xy"]
            surface = entry.get("surface", "floor")
            rot = entry.get("rot")
            z_override = entry.get("z")

            # 1) Apply rotation first so bbox reflects new orientation.
            if rot is not None:
                rot_msg = f", rot -> {tuple(rot)}"
                if not dry_run:
                    _set_rotate_xyz(xf, rot)

            # 2) Derive Z via bbox contact unless an explicit override is given.
            if z_override is not None:
                new_z = float(z_override)
            else:
                target_bottom = _surface_z(surface, surfaces) + spawn_margin
                if dry_run:
                    new_z = target_bottom
                else:
                    translate_op.Set(Gf.Vec3d(x, y, 0.0))
                    bbox_min_z = _world_bbox_min_z(child, tc)
                    new_z = target_bottom - bbox_min_z

            new_pos = (x, y, new_z)
        else:
            new_pos = entry["pos"]  # legacy absolute — translate-only, rotation preserved

        old_fmt = f"({old_pos[0]:.3f}, {old_pos[1]:.3f}, {old_pos[2]:.3f})"
        new_fmt = f"({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})"
        if dry_run:
            print(f"  [DRY] {name}: {old_fmt} -> {new_fmt}{rot_msg}")
        else:
            translate_op.Set(Gf.Vec3d(*new_pos))
            print(f"  [SET] {name}: {old_fmt} -> {new_fmt}{rot_msg}")
        n_moved += 1

    if not dry_run and n_moved > 0:
        stage.GetRootLayer().Save()

    print(f"\n  {n_moved} parts {'would be' if dry_run else ''} repositioned.")
    return 0


def verify(scene_path: str, surfaces: dict, parts: dict) -> int:
    """Verify pairwise spacing and per-part placement.

    For pos-form entries, the translate must match exactly (legacy strict check).
    For bbox-contact entries, XY must match exactly; Z is checked via bbox-contact —
    the world-AABB bottom must be within HOVER_TOL of (surface + spawn_margin).
    """
    spawn_margin = float(surfaces.get("spawn_margin", SPAWN_MARGIN))
    stage = Usd.Stage.Open(scene_path)
    dp = stage.GetDefaultPrim()
    tc = Usd.TimeCode.Default()

    positions = {}
    bbox_min_z = {}
    for child in dp.GetChildren():
        name = child.GetName()
        if name not in parts:
            continue
        xf = UsdGeom.Xformable(child)
        translate_op = _find_op(xf, UsdGeom.XformOp.TypeTranslate)
        if translate_op is None:
            continue
        pos = translate_op.Get(tc)
        positions[name] = (float(pos[0]), float(pos[1]), float(pos[2]))
        try:
            bbox_min_z[name] = _world_bbox_min_z(child, tc)
        except Exception:
            pass

    # Pairwise distance check
    MIN_DISTANCE = 0.03
    names = list(positions.keys())
    n_pass = 0
    n_fail = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = positions[a], positions[b]
            dx, dy, dz = pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < MIN_DISTANCE:
                print(f"  [FAIL] {a} <-> {b}: distance = {dist:.3f}m (min {MIN_DISTANCE}m)")
                n_fail += 1
            else:
                n_pass += 1

    # Per-part placement check
    HOVER_TOL = 0.02  # 2 cm — bbox bottom must be within this of (surface + margin)
    n_pos_pass = 0
    n_pos_fail = 0
    for name, entry in parts.items():
        if name not in positions:
            print(f"  [FAIL] {name}: not found in scene")
            n_pos_fail += 1
            continue
        actual = positions[name]

        if _is_dict_spec(entry):
            ex, ey = entry["xy"]
            if abs(actual[0] - ex) > 0.001 or abs(actual[1] - ey) > 0.001:
                print(f"  [FAIL] {name}: XY mismatch actual=({actual[0]:.3f},{actual[1]:.3f}) "
                      f"expected=({ex:.3f},{ey:.3f})")
                n_pos_fail += 1
                continue
            if "z" in entry:
                if abs(actual[2] - float(entry["z"])) > 0.001:
                    print(f"  [FAIL] {name}: Z override mismatch actual={actual[2]:.3f} "
                          f"expected={entry['z']:.3f}")
                    n_pos_fail += 1
                    continue
            elif name in bbox_min_z:
                target_bottom = _surface_z(entry.get("surface", "floor"), surfaces) + spawn_margin
                if abs(bbox_min_z[name] - target_bottom) > HOVER_TOL:
                    print(f"  [FAIL] {name}: hovering — bbox bottom Z={bbox_min_z[name]:.3f}, "
                          f"expected ≈ {target_bottom:.3f}")
                    n_pos_fail += 1
                    continue
            n_pos_pass += 1
        else:
            expected = entry["pos"]
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
    parser.add_argument("--layout", default=DEFAULT_LAYOUT_PATH,
                        help="Scatter-layout YAML (positions + surface constants).")
    args = parser.parse_args()

    surfaces, parts = load_layout(args.layout)

    print(f"\n{'='*60}")
    print(f"  Scatter Parts — spread gearbox parts across the table")
    print(f"  Scene:  {os.path.basename(args.scene)}")
    print(f"  Layout: {os.path.basename(args.layout)} ({len(parts)} parts)")
    print(f"{'='*60}\n")

    if args.verify:
        sys.exit(verify(args.scene, surfaces, parts))
    else:
        rc = scatter(args.scene, surfaces, parts, dry_run=args.dry_run)
        if rc == 0 and not args.dry_run:
            print("\nRunning verification...\n")
            sys.exit(verify(args.scene, surfaces, parts))
        sys.exit(rc)
