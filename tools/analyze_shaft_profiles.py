#!/usr/bin/env python3
"""Inspect radial profiles for shaft and gear assets."""

from __future__ import annotations

import numpy as np

from _bootstrap import REPO_ROOT, ensure_pxr_paths

ensure_pxr_paths()

from pxr import Gf, Usd, UsdGeom  # type: ignore  # noqa: E402

PARTS_DIR = REPO_ROOT / "assets" / "parts"
SHAFTS = (
    ("Output Shaft", "Y"),
    ("Transfer Shaft", "X"),
    ("Input Shaft", "Y"),
)
GEARS = ("Output Gear", "Transfer Gear")


def _open_stage(part_name: str):
    stage_path = PARTS_DIR / f"{part_name}.usd"
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open stage: {stage_path}")
    return stage


def get_world_points(stage) -> np.ndarray:
    """Return all mesh vertices in world space for a USD stage."""
    points: list[list[float]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        mesh_points = mesh.GetPointsAttr().Get()
        if mesh_points is None:
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in mesh_points:
            world_point = matrix.Transform(Gf.Vec3d(point))
            points.append([world_point[0], world_point[1], world_point[2]])
    return np.array(points)


def _print_shaft_profile(part_name: str, shaft_axis: str, slices: int = 40) -> None:
    print(f"\n{'=' * 60}")
    print(f"{part_name} (shaft axis = {shaft_axis})")
    print("=" * 60)

    points = get_world_points(_open_stage(part_name))
    axis_index = {"X": 0, "Y": 1, "Z": 2}[shaft_axis]
    other_axes = [idx for idx in range(3) if idx != axis_index]

    axis_min, axis_max = points[:, axis_index].min(), points[:, axis_index].max()
    print(f"  Range along {shaft_axis}: [{axis_min:.2f}, {axis_max:.2f}]")

    slice_width = (axis_max - axis_min) / (slices * 1.5)
    for i in range(slices + 1):
        value = axis_min + (axis_max - axis_min) * i / slices
        mask = np.abs(points[:, axis_index] - value) < slice_width
        if mask.sum() < 3:
            continue
        radius = np.sqrt(points[mask, other_axes[0]] ** 2 + points[mask, other_axes[1]] ** 2)
        print(f"  {shaft_axis}={value:7.2f}: max_r={radius.max():6.2f}  n={mask.sum()}")


def _print_gear_profile(part_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{part_name} (disc normal = Z)")
    print("=" * 60)

    points = get_world_points(_open_stage(part_name))
    print(
        "  BBox:"
        f" X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]"
        f" Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]"
        f" Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]"
    )

    radius_xy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    inner_points = points[radius_xy < 30]
    if len(inner_points) == 0:
        print("  No points found near the center hole.")
        return

    inner_radius = np.sqrt(inner_points[:, 0] ** 2 + inner_points[:, 1] ** 2)
    print(f"  Inner hole: min_r={inner_radius.min():.2f}, points near center: {len(inner_points)}")

    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    for z_value in np.linspace(z_min, z_max, 10):
        mask = np.abs(points[:, 2] - z_value) < 1.5
        if mask.sum() < 3:
            continue
        radius = np.sqrt(points[mask, 0] ** 2 + points[mask, 1] ** 2)
        print(f"  Z={z_value:6.2f}: min_r={radius.min():6.2f} max_r={radius.max():6.2f}")


def main() -> int:
    for part_name, axis in SHAFTS:
        _print_shaft_profile(part_name, axis)
    for part_name in GEARS:
        _print_gear_profile(part_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
