#!/usr/bin/env python3
"""Tune gear bores and preserve concave mating geometry with SDF colliders."""

from __future__ import annotations

import math
import os
import struct
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402


PARTS = os.path.join(_REPO, "assets", "parts")
TARGET_CLEARANCE = 0.50  # asset units; scene scale 0.002 => 1.00 mm
BORE_BAND = 0.25
FITS = (
    ("Transfer Gear", "Transfer Shaft", 0, 22.1, 11.05),
    ("Output Gear", "Output Shaft", 1, 15.5, 13.50),
)
SDF_ASSETS = (
    "Transfer Gear", "Output Gear", "Casing Base", "Casing Top",
    "Input Shaft", "Transfer Shaft", "Output Shaft",
    "M6 Hub Bolt", "M10 Casing Bolt", "M10 Casing Nut",
    "Breather Plug", "Oil Level Indicator",
)
GEAR_PROXIES = {
    "Transfer Gear": (30, 46.0, 50.13, 11.05),
    "Output Gear": (48, 74.0, 80.58, 13.50),
}


def _mesh_points(name: str) -> np.ndarray:
    stage = Usd.Stage.Open(os.path.join(PARTS, f"{name}.usd"))
    points = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName() == "collision_gear":
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in UsdGeom.Mesh(prim).GetPointsAttr().Get() or []:
            p = matrix.Transform(Gf.Vec3d(point))
            points.append((p[0], p[1], p[2]))
    return np.asarray(points)


def _shaft_radius(name: str, axis: int, mount: float, half_thickness: float) -> float:
    points = _mesh_points(name)
    # The gear approaches from the shaft's positive end.  Its bore must clear
    # every shaft section swept by the leading gear face, not just the final
    # center plane.
    section = points[points[:, axis] >= mount - half_thickness - 0.1]
    radial_axes = [i for i in range(3) if i != axis]
    return float(np.linalg.norm(section[:, radial_axes], axis=1).max())


def _expand_xy(x: float, y: float, bore: float, delta: float) -> tuple[float, float]:
    radius = math.hypot(x, y)
    if radius == 0 or radius > bore + BORE_BAND:
        return x, y
    scale = (radius + delta) / radius
    return x * scale, y * scale


def _update_usd(name: str, bore: float, delta: float) -> int:
    path = os.path.join(PARTS, f"{name}.usd")
    stage = Usd.Stage.Open(path)
    changed = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName() == "collision_gear":
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        inverse = matrix.GetInverse()
        points = list(mesh.GetPointsAttr().Get() or [])
        updated = []
        for point in points:
            world = matrix.Transform(Gf.Vec3d(point))
            x, y = _expand_xy(float(world[0]), float(world[1]), bore, delta)
            changed += x != world[0] or y != world[1]
            local = inverse.Transform(Gf.Vec3d(x, y, world[2]))
            updated.append(Gf.Vec3f(local))
        mesh.GetPointsAttr().Set(updated)
    stage.GetRootLayer().Save()
    return changed


def _update_binary_stl(name: str, bore: float, delta: float) -> int:
    path = os.path.join(PARTS, f"{name}.stl")
    data = bytearray(open(path, "rb").read())
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path} is not a binary STL")
    stage = Usd.Stage.Open(os.path.join(PARTS, f"{name}.usd"))
    mesh_prim = next(prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh))
    matrix = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inverse = matrix.GetInverse()
    changed = 0
    for triangle in range(count):
        base = 84 + triangle * 50
        vertices = []
        for index in range(3):
            offset = base + 12 + index * 12
            x, y, z = struct.unpack_from("<fff", data, offset)
            world = matrix.Transform(Gf.Vec3d(x, y, z))
            x2, y2 = _expand_xy(float(world[0]), float(world[1]), bore, delta)
            changed += x2 != world[0] or y2 != world[1]
            local = inverse.Transform(Gf.Vec3d(x2, y2, world[2]))
            vertices.append((local[0], local[1], local[2]))
            struct.pack_into("<fff", data, offset, local[0], local[1], local[2])
        a, b, c = (np.asarray(v) for v in vertices)
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length:
            normal /= length
        struct.pack_into("<fff", data, base, *normal)
    with open(path, "wb") as stream:
        stream.write(data)
    return changed


def _set_sdf(name: str) -> int:
    path = os.path.join(PARTS, f"{name}.usd")
    stage = Usd.Stage.Open(path)
    changed = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        if collision.GetApproximationAttr().Get() != "sdf":
            collision.GetApproximationAttr().Set("sdf")
            changed += 1
    stage.GetRootLayer().Save()
    return changed


def _author_gear_proxy(name: str, teeth: int, root_radius: float,
                       tip_radius: float, half_thickness: float) -> None:
    path = os.path.join(PARTS, f"{name}.usd")
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    proxy_path = root.GetPath().AppendChild("collision_gear")

    for prim in list(stage.Traverse()):
        if prim.IsA(UsdGeom.Mesh) and prim.GetPath() != proxy_path:
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)

    bore_radius = float(np.linalg.norm(_mesh_points(name)[:, :2], axis=1).min())
    segments = teeth * 2
    radii = [tip_radius if index % 2 == 0 else root_radius for index in range(segments)]
    points = []
    for z, inner in ((-half_thickness, False), (half_thickness, False),
                     (-half_thickness, True), (half_thickness, True)):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            radius = bore_radius if inner else radii[index]
            points.append(Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), z))

    outer_bottom, outer_top, inner_bottom, inner_top = (0, segments, 2 * segments, 3 * segments)
    indices = []
    for index in range(segments):
        nxt = (index + 1) % segments
        indices.extend((outer_bottom + index, inner_bottom + index, inner_bottom + nxt, outer_bottom + nxt))
        indices.extend((outer_top + index, outer_top + nxt, inner_top + nxt, inner_top + index))
        indices.extend((outer_bottom + index, outer_bottom + nxt, outer_top + nxt, outer_top + index))
        indices.extend((inner_bottom + index, inner_top + index, inner_top + nxt, inner_bottom + nxt))

    proxy = UsdGeom.Mesh.Define(stage, proxy_path)
    proxy.CreatePointsAttr(points)
    proxy.CreateFaceVertexCountsAttr([4] * segments * 4)
    proxy.CreateFaceVertexIndicesAttr(indices)
    proxy.CreateSubdivisionSchemeAttr("none")
    proxy.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(proxy.GetPrim()).GetApproximationAttr().Set("sdf")
    stage.GetRootLayer().Save()


def main() -> int:
    for gear, shaft, axis, mount, half_thickness in FITS:
        gear_points = _mesh_points(gear)
        bore = float(np.linalg.norm(gear_points[:, :2], axis=1).min())
        shaft = _shaft_radius(shaft, axis, mount, half_thickness)
        delta = max(0.0, shaft + TARGET_CLEARANCE - bore)
        if delta <= 1e-4:
            print(f"PASS {gear}: clearance already >= {TARGET_CLEARANCE:.3f} asset units")
            continue
        usd_vertices = _update_usd(gear, bore, delta)
        stl_vertices = _update_binary_stl(gear, bore, delta)
        print(
            f"UPDATED {gear}: bore {bore:.3f} -> {bore + delta:.3f}, "
            f"USD vertices={usd_vertices}, STL vertices={stl_vertices}"
        )
    for asset in SDF_ASSETS:
        print(f"COLLIDER {'UPDATED' if _set_sdf(asset) else 'PASS'} {asset}: approximation=sdf")
    for asset, config in GEAR_PROXIES.items():
        _author_gear_proxy(asset, *config)
        print(f"COLLIDER UPDATED {asset}: watertight toothed annulus proxy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
