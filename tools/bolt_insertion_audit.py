#!/usr/bin/env python3
"""Read-only, fast USD unit/transform inventory for bolt insertion candidate #01."""
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
from pxr import Gf, Usd, UsdGeom  # noqa: E402


def matrix_rows(matrix):
    return [[float(matrix[i][j]) for j in range(4)] for i in range(4)]


def xyz(value):
    return [float(value[i]) for i in range(3)]


def point_bounds(points):
    return [[min(p[i] for p in points), max(p[i] for p in points)] for i in range(3)]


def xform_ops(prim):
    result = []
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        value = op.Get()
        try:
            value = [float(v) for v in value]
        except (TypeError, ValueError):
            value = str(value)
        result.append({"op": str(op.GetOpName()), "type": str(op.GetOpType()), "value": value})
    return {"order": [str(x) for x in xf.GetXformOpOrderAttr().Get() or []], "ops": result}


def root_geometry(stage, root):
    cache = UsdGeom.XformCache()
    root_world = cache.GetLocalToWorldTransform(root)
    world_to_root = root_world.GetInverse()
    points, meshes, triangles = [], [], 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        local_to_root = cache.GetLocalToWorldTransform(prim) * world_to_root
        mesh_points = mesh.GetPointsAttr().Get() or []
        points.extend(tuple(float(v) for v in local_to_root.Transform(Gf.Vec3d(*p)))
                      for p in mesh_points)
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        triangles += sum(max(0, int(n) - 2) for n in counts)
        meshes.append({"path": str(prim.GetPath()), "point_count": len(mesh_points),
                       "triangle_count": sum(max(0, int(n) - 2) for n in counts)})
    if not points:
        raise RuntimeError(f"no mesh points under {root.GetPath()}")
    return point_bounds(points), meshes, points


def axis_y_diameter(points, lo_fraction=0.15, hi_fraction=0.62):
    """Estimate a Y-axis shank OD from mid-shank vertices; report spread, not a fit claim."""
    ys = [p[1] for p in points]
    lo, hi = min(ys), max(ys)
    span = hi - lo
    radii = sorted(math.hypot(p[0], p[2]) for p in points
                   if lo + lo_fraction * span <= p[1] <= lo + hi_fraction * span)
    if not radii:
        raise RuntimeError("no vertices in selected shank band")
    return {"band_y": [lo + lo_fraction * span, lo + hi_fraction * span],
            "diameter_median_units": 2.0 * radii[len(radii) // 2],
            "diameter_p95_units": 2.0 * radii[min(len(radii) - 1, int(.95 * len(radii)))],
            "vertex_samples": len(radii)}


def asset_report(path):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open {path}")
    root = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    if not root or not root.IsValid():
        raise RuntimeError(f"missing default root: {path}")
    bounds, meshes, points = root_geometry(stage, root)
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    return {"path": str(path.relative_to(ROOT)), "stage_meters_per_unit": mpu,
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)), "root_path": str(root.GetPath()),
            "root_local_to_stage_matrix": matrix_rows(UsdGeom.XformCache().GetLocalToWorldTransform(root)),
            "root_xform": xform_ops(root), "root_local_mesh_bounds_units": bounds,
            "root_local_mesh_bounds_if_root_scale_ignored_m": [[v * mpu for v in a] for a in bounds],
            "meshes": meshes, "triangle_count": sum(m["triangle_count"] for m in meshes),
            "_points": points}


def scene_report(path):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open {path}")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    cache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    wanted = {"Casing_Top", "M6_Hub_Bolt_01_top", "M6_Hub_Bolt_01",
              "Hub_Cover_Output_Top", "Hub_Cover_Output"}
    result = {"path": str(path.relative_to(ROOT)), "stage_meters_per_unit": mpu,
              "up_axis": str(UsdGeom.GetStageUpAxis(stage)), "instances": {}}
    for prim in stage.Traverse():
        if prim.GetName() not in wanted:
            continue
        world = cache.GetLocalToWorldTransform(prim)
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = box.GetMin(), box.GetMax()
        basis = [world.TransformDir(Gf.Vec3d(*(1 if i == j else 0 for i in range(3)))).GetLength()
                 for j in range(3)]
        result["instances"][str(prim.GetPath())] = {
            "name": prim.GetName(), "world_matrix": matrix_rows(world), "world_translation": xyz(world.GetRow3(3)),
            "world_linear_basis_lengths": basis, "xform": xform_ops(prim),
            "world_bounds_stage_units": [xyz(mn), xyz(mx)],
            "world_dimensions_m": [float(mx[i] - mn[i]) * mpu for i in range(3)],
            "world_bounds_m": [[float(mn[i]) * mpu for i in range(3)],
                                [float(mx[i]) * mpu for i in range(3)]],
        }
    return result


def self_test():
    pts = [(0, 0, 0), (0, 10, 3)] + [(r, y, 0) for y in (2, 3, 4, 5, 6) for r in (-3, 3)]
    measured = axis_y_diameter(pts)
    assert measured["vertex_samples"] == 10 and measured["diameter_median_units"] == 6
    assert point_bounds(pts) == [[-3, 3], [0, 10], [0, 3]]
    print("bolt insertion USD audit self-test: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "assets/simple_room_scene.usd")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    assets = [ROOT / "assets/parts/Casing Top.usd", ROOT / "assets/parts/Hub Cover Output.usd",
              ROOT / "assets/parts/M6 Hub Bolt.usd"]
    report = {"scope": "offline read-only USD inventory; no Kit, physics, or GPU simulation",
              "assets": {}, "simple_room_scene": scene_report(args.scene)}
    for path in assets:
        data = asset_report(path)
        if path.name == "M6 Hub Bolt.usd":
            shaft = axis_y_diameter(data.pop("_points"))
            data["estimated_y_axis_shank_diameter"] = shaft
            data["estimated_y_axis_shank_diameter_at_source_stage_m"] = {
                key: value * data["stage_meters_per_unit"]
                for key, value in shaft.items() if key.endswith("_units")}
        else:
            data.pop("_points")
        report["assets"][path.name] = data
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
