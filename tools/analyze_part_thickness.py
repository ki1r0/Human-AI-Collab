#!/usr/bin/env python3
"""Analyze part dimensions to compute the correct fit offsets.

For each child part, determine:
1. Thin axis (disc normal for hub covers, shaft axis for bolts)
2. Half-thickness (how far from center to the contact face)
3. Which face is the "contact" face (head of bolt, mounting face of hub cover)
"""
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()
from pxr import Gf, Usd, UsdGeom


def get_root_local_points(stage, root_prim):
    pts = []
    xf_cache = UsdGeom.XformCache()
    root_to_world = xf_cache.GetLocalToWorldTransform(root_prim)
    root_to_world_inv = root_to_world.GetInverse()
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        prim_to_world = xf_cache.GetLocalToWorldTransform(prim)
        prim_to_root = prim_to_world * root_to_world_inv
        for p in points:
            wp = prim_to_root.Transform(Gf.Vec3d(p[0], p[1], p[2]))
            pts.append((wp[0], wp[1], wp[2]))
    return pts


PARTS_DIR = os.path.join(_REPO, "assets", "parts")


def analyze_part(part_name):
    usd_path = os.path.join(PARTS_DIR, f"{part_name}.usd")
    if not os.path.isfile(usd_path):
        print(f"  {part_name}: file not found")
        return
    stage = Usd.Stage.Open(usd_path)
    dp = stage.GetDefaultPrim()
    pts = get_root_local_points(stage, dp)

    if not pts:
        print(f"  {part_name}: no vertices")
        return

    mins = [min(p[i] for p in pts) for i in range(3)]
    maxs = [max(p[i] for p in pts) for i in range(3)]
    extents = [maxs[i] - mins[i] for i in range(3)]

    axis_names = ['X', 'Y', 'Z']
    print(f"\n  {part_name}:")
    print(f"    BBox: {', '.join(f'{axis_names[i]}=[{mins[i]:.2f}, {maxs[i]:.2f}] ext={extents[i]:.2f}' for i in range(3))}")

    # For each axis, count vertices near each face
    for ax in range(3):
        face_lo = sum(1 for p in pts if abs(p[ax] - mins[ax]) < 0.5)
        face_hi = sum(1 for p in pts if abs(p[ax] - maxs[ax]) < 0.5)
        print(f"    {axis_names[ax]}- face ({mins[ax]:.2f}): {face_lo} vertices")
        print(f"    {axis_names[ax]}+ face ({maxs[ax]:.2f}): {face_hi} vertices")

    # For bolt-like parts, analyze the cross-section at different heights along the shaft
    shaft_axis = extents.index(max(extents))
    print(f"    Longest axis: {axis_names[shaft_axis]} (shaft direction)")
    print(f"    Cross-section analysis along {axis_names[shaft_axis]}:")

    n_slices = 20
    for s in range(n_slices + 1):
        pos = mins[shaft_axis] + (maxs[shaft_axis] - mins[shaft_axis]) * s / n_slices
        # Get cross-section vertices within 0.5mm
        cross = [(p[(shaft_axis+1)%3], p[(shaft_axis+2)%3])
                 for p in pts if abs(p[shaft_axis] - pos) < 0.5]
        if len(cross) > 5:
            # Compute max radius from center
            dists = [math.sqrt(c[0]**2 + c[1]**2) for c in cross]
            max_r = max(dists)
            print(f"      {axis_names[shaft_axis]}={pos:7.2f}: {len(cross):5d} pts, max_r={max_r:.2f}")


def main():
    parts = [
        "M6 Hub Bolt",
        "M10 Casing Bolt",
        "M10 Casing Nut",
        "Hub Cover Output",
        "Hub Cover Input",
        "Hub Cover Small",
        "Oil Level Indicator",
        "Breather Plug",
    ]
    print(f"\n{'='*70}")
    print(f"  PART THICKNESS / GEOMETRY ANALYSIS")
    print(f"{'='*70}")
    for p in parts:
        analyze_part(p)


if __name__ == "__main__":
    main()
