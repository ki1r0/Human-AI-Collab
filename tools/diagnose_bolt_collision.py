#!/usr/bin/env python3
"""Offline audit of the source bolt profile and installed PhysX SDF schema."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools._bootstrap import ensure_pxr_paths  # noqa: E402

ensure_pxr_paths()
from pxr import Gf, PhysxSchema, Usd, UsdGeom  # noqa: E402


def _triangles(mesh: UsdGeom.Mesh) -> list[tuple[int, int, int]]:
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    result: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        for i in range(1, count - 1):
            result.append((int(face[0]), int(face[i]), int(face[i + 1])))
    if cursor != len(indices):
        raise ValueError("face counts do not consume all mesh vertex indices")
    return result


def _vertex_components(
    points: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    weld_tolerance: float = 1e-4,
) -> list[list[int]]:
    """Group triangle-soup faces sharing a welded position, not just an index."""
    parent = list(range(len(triangles)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_face_for_position: dict[tuple[int, int, int], int] = {}
    for face_index, triangle in enumerate(triangles):
        for vertex_index in triangle:
            position = points[vertex_index]
            key = tuple(round(value / weld_tolerance) for value in position)
            previous = first_face_for_position.setdefault(key, face_index)
            union(face_index, previous)

    groups: dict[int, list[int]] = {}
    for face_index in range(len(triangles)):
        groups.setdefault(find(face_index), []).append(face_index)
    return sorted(groups.values(), key=len, reverse=True)


def _welded_topology(
    points: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    weld_tolerance: float = 1e-4,
) -> dict[str, int | bool]:
    position_ids: dict[tuple[int, int, int], int] = {}
    welded_triangles = []
    for triangle in triangles:
        welded = []
        for index in triangle:
            key = tuple(round(value / weld_tolerance) for value in points[index])
            welded.append(position_ids.setdefault(key, len(position_ids)))
        welded_triangles.append(tuple(welded))

    edge_directions: dict[tuple[int, int], list[int]] = {}
    for triangle in welded_triangles:
        for start, end in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[start], triangle[end]
            edge = (min(a, b), max(a, b))
            edge_directions.setdefault(edge, []).append(1 if a < b else -1)
    boundary_edges = sum(len(uses) == 1 for uses in edge_directions.values())
    nonmanifold_edges = sum(len(uses) > 2 for uses in edge_directions.values())
    winding_conflicts = sum(len(uses) == 2 and uses[0] == uses[1]
                            for uses in edge_directions.values())
    return {
        "welded_vertex_count": len(position_ids),
        "welded_edge_count": len(edge_directions),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "shared_edge_winding_conflict_count": winding_conflicts,
        "closed_oriented_two_manifold_at_tolerance": not (boundary_edges or nonmanifold_edges or winding_conflicts),
    }


def _bounds(points: list[tuple[float, float, float]]) -> list[list[float]]:
    return [[min(point[axis] for point in points), max(point[axis] for point in points)]
            for axis in range(3)]


def _determinant3(matrix) -> float:
    a, b, c = (float(matrix[0][i]) for i in range(3))
    d, e, f = (float(matrix[1][i]) for i in range(3))
    g, h, i = (float(matrix[2][i]) for i in range(3))
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _signed_volume(points: list[tuple[float, float, float]],
                  triangles: list[tuple[int, int, int]]) -> float:
    """Oriented triangle volume; positive is outward for right-handed winding."""
    volume6 = 0.0
    for ia, ib, ic in triangles:
        ax, ay, az = points[ia]
        bx, by, bz = points[ib]
        cx, cy, cz = points[ic]
        volume6 += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
    return volume6 / 6.0


def _section_profile(
    points: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    y: float,
) -> dict[str, float | int] | None:
    hits: list[tuple[float, float]] = []
    eps = 1e-9
    for triangle in triangles:
        vertices = [points[index] for index in triangle]
        section_points: list[tuple[float, float]] = []
        for start, end in ((0, 1), (1, 2), (2, 0)):
            a, b = vertices[start], vertices[end]
            da, db = a[1] - y, b[1] - y
            if abs(da) <= eps:
                section_points.append((a[0], a[2]))
            if da * db < -(eps * eps):
                t = da / (da - db)
                section_points.append((a[0] + t * (b[0] - a[0]),
                                       a[2] + t * (b[2] - a[2])))
        unique = list(dict.fromkeys((round(x, 10), round(z, 10)) for x, z in section_points))
        hits.extend(unique)
    if not hits:
        return None
    radii = sorted(math.hypot(x, z) for x, z in hits)
    return {
        "y_source_units": y,
        "surface_intersection_samples": len(radii),
        "radius_min_source_units": radii[0],
        "radius_median_source_units": radii[len(radii) // 2],
        "radius_p95_source_units": radii[min(len(radii) - 1, int(0.95 * len(radii)))],
        "radius_max_source_units": radii[-1],
    }


def audit(asset: Path, instance_scale_m_per_source_unit: float) -> dict[str, object]:
    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError(f"could not open bolt asset: {asset}")
    root = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError(f"no UsdGeom.Mesh under {root.GetPath()}")

    cache = UsdGeom.XformCache()
    root_to_world = cache.GetLocalToWorldTransform(root)
    world_to_root = root_to_world.GetInverse()
    root_to_stage_determinant = _determinant3(root_to_world)
    mesh_records = []
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        mesh_to_root = cache.GetLocalToWorldTransform(prim) * world_to_root
        points = [tuple(float(value) for value in mesh_to_root.Transform(Gf.Vec3d(*point)))
                  for point in (mesh.GetPointsAttr().Get() or [])]
        triangles = _triangles(mesh)
        islands = _vertex_components(points, triangles)
        topology = _welded_topology(points, triangles)
        orientation = str(mesh.GetOrientationAttr().Get())
        orientation_factor = {"rightHanded": 1.0, "leftHanded": -1.0}.get(orientation)
        signed_volume_root = _signed_volume(points, triangles)
        orientation_adjusted_volume_root = (
            signed_volume_root * orientation_factor if orientation_factor is not None else None
        )
        task_scale_determinant = instance_scale_m_per_source_unit ** 3
        orientation_adjusted_volume_task_m3 = (
            orientation_adjusted_volume_root * root_to_stage_determinant * task_scale_determinant
            if orientation_adjusted_volume_root is not None else None
        )
        if orientation_adjusted_volume_task_m3 is None or abs(orientation_adjusted_volume_task_m3) < 1e-18:
            winding_classification = "undetermined"
        else:
            winding_classification = (
                "outward" if orientation_adjusted_volume_task_m3 > 0 else "inward"
            )
        components = []
        for face_indices in islands:
            vertex_indices = sorted({index for face_index in face_indices for index in triangles[face_index]})
            component_points = [points[index] for index in vertex_indices]
            component_bounds = _bounds(component_points)
            welded_positions = {
                tuple(round(value / 1e-4) for value in point) for point in component_points
            }
            components.append({
                "triangle_count": len(face_indices),
                "source_point_index_count": len(vertex_indices),
                "welded_position_count_at_1e-4_source_unit_tolerance": len(welded_positions),
                "bounds_source_units_xyz": component_bounds,
                "bounds_at_task_scale_m_xyz": [
                    [low * instance_scale_m_per_source_unit, high * instance_scale_m_per_source_unit]
                    for low, high in component_bounds
                ],
                "max_radius_about_local_y_axis_source_units": max(
                    math.hypot(points[index][0], points[index][2]) for index in vertex_indices
                ),
            })
        ymin, ymax = _bounds(points)[1]
        sample_ys = sorted({
            ymin + 0.01,
            -12.0, -10.0, -8.0, -5.0, 0.0, 3.0, 6.0, 8.0,
            8.8, 9.0, 10.0, 11.0, 12.0, ymax - 0.01,
        })
        profile = [item for y in sample_ys
                   if (item := _section_profile(points, triangles, y)) is not None]

        sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI
        sdf = sdf_api.Apply(prim)  # in-memory only; this stage is never exported
        sdf_defaults = {
            "sdfResolution": int(sdf.GetSdfResolutionAttr().Get()),
            "sdfSubgridResolution": int(sdf.GetSdfSubgridResolutionAttr().Get()),
            "sdfNarrowBandThickness": float(sdf.GetSdfNarrowBandThicknessAttr().Get()),
            "sdfMargin": float(sdf.GetSdfMarginAttr().Get()),
            "sdfEnableRemeshing": bool(sdf.GetSdfEnableRemeshingAttr().Get()),
        }
        extent_units = max(axis[1] - axis[0] for axis in _bounds(points))
        extent_m = extent_units * instance_scale_m_per_source_unit
        min_res_for_0_1mm = math.ceil(extent_m / 0.0001)
        mesh_records.append({
            "prim_path": str(prim.GetPath()),
            "point_count": len(points),
            "triangle_count": len(triangles),
            "welded_topology_at_1e-4_source_unit_tolerance": topology,
            "orientation_winding_audit": {
                "mesh_orientation_token": orientation,
                "raw_triangle_signed_volume_root_source_units3": signed_volume_root,
                "orientation_adjusted_signed_volume_root_source_units3": orientation_adjusted_volume_root,
                "orientation_adjusted_signed_volume_task_instance_m3": orientation_adjusted_volume_task_m3,
                "mesh_to_root_linear_determinant": _determinant3(mesh_to_root),
                "root_to_stage_linear_determinant": root_to_stage_determinant,
                "task_uniform_scale_linear_determinant": task_scale_determinant,
                "task_instance_winding_classification": winding_classification,
            },
            "source_bounds_units_xyz": _bounds(points),
            "source_bounds_at_task_scale_m_xyz": [
                [low * instance_scale_m_per_source_unit, high * instance_scale_m_per_source_unit]
                for low, high in _bounds(points)
            ],
            "connected_surface_components_by_welded_positions": components,
            "radial_section_profile_about_local_y": profile,
            "sdf_api_can_apply": bool(sdf_api.CanApply(prim)),
            "sdf_api_defaults_in_memory_only": sdf_defaults,
            "sdf_max_aabb_extent_at_task_scale_m": extent_m,
            "sdf_spacing_at_resolution_512_mm": extent_m / 512.0 * 1000.0,
            "minimum_integer_resolution_for_spacing_le_0_1mm": min_res_for_0_1mm,
            "sdf_spacing_at_minimum_resolution_mm": extent_m / min_res_for_0_1mm * 1000.0,
        })

    return {
        "scope": "read-only USD geometry; no Kit, physics stepping, or GPU simulation; SDF schema applied only in memory",
        "asset": str(asset),
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "task_instance_scale_m_per_source_unit": instance_scale_m_per_source_unit,
        "root_path": str(root.GetPath()),
        "mesh_prim_count": len(mesh_records),
        "meshes": mesh_records,
    }


def _self_test() -> None:
    pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
           (1.0, 1.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0)]
    # The second triangle duplicates the first component's vertex positions
    # under different indices, as in this asset's triangle-soup USD encoding.
    tris = [(0, 1, 2), (2, 3, 0), (4, 5, 6)]
    groups = _vertex_components(pts, tris)
    assert sorted(len(group) for group in groups) == [1, 2]
    topology = _welded_topology(pts, tris[:2])
    assert topology["boundary_edge_count"] == 4 and not topology["closed_oriented_two_manifold_at_tolerance"]
    tetra = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    outward_faces = [(0, 2, 1), (0, 3, 2), (0, 1, 3), (1, 2, 3)]
    assert math.isclose(_signed_volume(tetra, outward_faces), 1.0 / 6.0)
    points = [(1.0, -1.0, 0.0), (1.0, 1.0, 0.0),
              (0.0, 1.0, 1.0), (0.0, -1.0, 1.0)]
    profile = _section_profile(points, [(0, 1, 2), (0, 2, 3)], 0.0)
    assert profile is not None and profile["surface_intersection_samples"] >= 2
    assert math.ceil(0.0523 / 0.0001) == 523
    print("bolt collision geometry audit self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ROOT / "assets/parts/M6 Hub Bolt.usd")
    parser.add_argument("--instance-scale", type=float, default=0.002,
                        help="task's meters per source unit, not source-stage metadata")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.instance_scale <= 0 or not math.isfinite(args.instance_scale):
        parser.error("--instance-scale must be positive and finite")
    print(json.dumps(audit(args.asset.resolve(), args.instance_scale), indent=2))


if __name__ == "__main__":
    main()
