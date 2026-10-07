#!/usr/bin/env python3
"""Repair the source casing's M6 clearance bores.

The CAD mesh has 5.6-unit bores while the supplied M6 bolt shank is 5.88
units in diameter.  This tool makes only the twelve measured inner rings into
6.5-unit clearance bores.  The rest of each mesh is untouched, and the same
vertex operation is applied to the source STL so the USD/STL pair remains
consistent.  It is intentionally opt-in and supports ``--dry-run``.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Gf, Usd, UsdGeom  # noqa: E402


# Circle fits on the actual source top face, not on the authored socket hints.
BORES = (
    (40.0003, 0.1324), (-40.0005, 0.1320),
    (40.0019, 80.1334), (-39.9974, 80.1322),
    (54.0013, -26.6663), (-53.9979, -26.6680),
    (7.9998, -26.6677), (54.1121, -72.4371),
    (7.9994, -72.6674), (-7.9994, -26.6672),
    (-7.9995, -72.6672), (-54.0159, -72.2708),
)

TARGET_RADIUS = 3.25
INNER_RADIUS = 2.0
OUTER_RADIUS = 3.6


def _move_xy(x: float, y: float) -> tuple[float, float, bool]:
    best = min(BORES, key=lambda c: (x - c[0]) ** 2 + (y - c[1]) ** 2)
    dx, dy = x - best[0], y - best[1]
    radius = math.hypot(dx, dy)
    if radius < INNER_RADIUS or radius >= OUTER_RADIUS:
        return x, y, False
    # Move only the inner annulus outward.  The outer casing surface and all
    # neighboring features are left exactly where the source CAD put them.
    new_radius = max(radius, TARGET_RADIUS)
    scale = new_radius / radius
    return best[0] + dx * scale, best[1] + dy * scale, new_radius != radius


def _mesh_data(stage: Usd.Stage):
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    root_world = cache.GetLocalToWorldTransform(root)
    root_inv = root_world.GetInverse()
    rows = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = list(mesh.GetPointsAttr().Get() or [])
        to_root = cache.GetLocalToWorldTransform(prim) * root_inv
        rows.append((mesh, points, to_root, to_root.GetInverse()))
    return rows


def repair_usd(path: str, dry_run: bool) -> int:
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"cannot open {path}")
    changed = 0
    for mesh, points, to_root, from_root in _mesh_data(stage):
        updated = []
        for point in points:
            root_point = to_root.Transform(Gf.Vec3d(point))
            x, y, did_change = _move_xy(float(root_point[0]), float(root_point[1]))
            changed += did_change
            updated.append(from_root.Transform(Gf.Vec3d(x, y, float(root_point[2]))))
        if not dry_run:
            mesh.GetPointsAttr().Set([Gf.Vec3f(p) for p in updated])
    if not dry_run:
        stage.GetRootLayer().Save()
    return changed


def repair_stl(path: str, usd_path: str, dry_run: bool) -> int:
    raw = bytearray(open(path, "rb").read())
    if len(raw) < 84:
        raise ValueError(f"invalid STL {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"only binary STL is supported: {path}")
    stage = Usd.Stage.Open(usd_path)
    rows = _mesh_data(stage)
    if not rows:
        return 0
    _, _, to_root, from_root = rows[0]
    changed = 0
    for index in range(count):
        base = 84 + index * 50
        vertices = []
        for corner in range(3):
            offset = base + 12 + corner * 12
            raw_point = Gf.Vec3d(struct.unpack_from("<3f", raw, offset))
            root_point = to_root.Transform(raw_point)
            x, y, did_change = _move_xy(float(root_point[0]), float(root_point[1]))
            changed += did_change
            local = from_root.Transform(Gf.Vec3d(x, y, float(root_point[2])))
            struct.pack_into("<3f", raw, offset, float(local[0]), float(local[1]), float(local[2]))
            vertices.append((float(local[0]), float(local[1]), float(local[2])))
        a, b, c = (Gf.Vec3d(v) for v in vertices)
        normal = Gf.Cross(b - a, c - a)
        if normal.GetLength() > 1e-12:
            normal.Normalize()
        struct.pack_into("<3f", raw, base, float(normal[0]), float(normal[1]), float(normal[2]))
    if not dry_run:
        with open(path, "wb") as stream:
            stream.write(raw)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", default=os.path.join(_REPO, "assets", "parts"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    total = 0
    for name in ("Casing Top", "Casing Base"):
        usd = os.path.join(args.parts_dir, f"{name}.usd")
        stl = os.path.join(args.parts_dir, f"{name}.stl")
        usd_changed = repair_usd(usd, args.dry_run)
        stl_changed = repair_stl(stl, usd, args.dry_run) if os.path.exists(stl) else 0
        total += usd_changed
        print(f"REPAIR {'PLAN' if args.dry_run else 'APPLY'} {name}: "
              f"usd_vertices={usd_changed} stl_vertices={stl_changed} "
              f"target_radius={TARGET_RADIUS:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
