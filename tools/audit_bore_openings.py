#!/usr/bin/env python3
"""Check whether the source casing meshes leave open axial paths at bolt bores.

This is deliberately a geometry-only check.  It does not infer success from
the authored ``socket_*`` Xforms; it ray-casts the actual source mesh triangles
at the canonical Magic Assembly M6 locations and reports the mesh crossings.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Gf, Usd, UsdGeom  # noqa: E402


M6_TOP = (
    (40.0, 0.0, "socket_bolt_hub_1"),
    (-40.0, 0.0, "socket_bolt_hub_2"),
    (40.0, 80.0, "socket_bolt_hub_3"),
    (-40.0, 80.0, "socket_bolt_hub_4"),
    (54.0, -27.0, "socket_bolt_hub_5"),
    (-54.0, -27.0, "socket_bolt_hub_6"),
    (8.26, -26.47, "socket_bolt_hub_8"),
    (54.0, -72.69, "socket_bolt_hub_9"),
    (7.76, -72.52, "socket_bolt_hub_10"),
    (-7.96, -26.37, "socket_bolt_hub_11"),
    (-8.0, -72.69, "socket_bolt_hub_13"),
    (-53.82, -72.43, "socket_bolt_hub_14"),
)

GEAR_TOP = (
    (31.0, -49.69, "socket_gear_input"),
    (-31.0, -49.69, "socket_gear_transfer"),
    (0.0, 39.75, "socket_gear_output"),
)

M10_TOP = (
    (60.0, 120.0, "socket_bolt_casing_1"),
    (-60.0, 120.0, "socket_bolt_casing_2"),
    (60.0, -125.0, "socket_bolt_casing_3"),
    (-60.0, -125.0, "socket_bolt_casing_4"),
    (95.0, 0.0, "socket_bolt_casing_5"),
    (-95.0, 0.0, "socket_bolt_casing_6"),
)


@dataclass
class Triangle:
    a: Gf.Vec3d
    b: Gf.Vec3d
    c: Gf.Vec3d


def _root_triangles(stage: Usd.Stage) -> list[Triangle]:
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    root_world = cache.GetLocalToWorldTransform(root)
    root_inv = root_world.GetInverse()
    out: list[Triangle] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get() or []
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        xf = cache.GetLocalToWorldTransform(prim) * root_inv
        cursor = 0
        for count in counts:
            face = indices[cursor : cursor + int(count)]
            cursor += int(count)
            if len(face) < 3:
                continue
            p0 = xf.Transform(Gf.Vec3d(pts[face[0]]))
            for i in range(1, len(face) - 1):
                out.append(
                    Triangle(
                        p0,
                        xf.Transform(Gf.Vec3d(pts[face[i]])),
                        xf.Transform(Gf.Vec3d(pts[face[i + 1]])),
                    )
                )
    return out


def _ray_z(tri: Triangle, x: float, y: float) -> float | None:
    """Return z at a +Z ray hit with a triangle, or None."""
    ax, ay = float(tri.a[0]), float(tri.a[1])
    bx, by = float(tri.b[0]), float(tri.b[1])
    cx, cy = float(tri.c[0]), float(tri.c[1])
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-10:
        return None
    u = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den
    v = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den
    w = 1.0 - u - v
    if u < -1e-7 or v < -1e-7 or w < -1e-7:
        return None
    return u * float(tri.a[2]) + v * float(tri.b[2]) + w * float(tri.c[2])


def _crossings(triangles: list[Triangle], x: float, y: float) -> list[float]:
    values = sorted(z for tri in triangles if (z := _ray_z(tri, x, y)) is not None)
    # A shared edge produces duplicate intersections.  Collapse those before
    # reporting the cross-section; otherwise tessellation looks like a solid.
    unique: list[float] = []
    for value in values:
        if not unique or abs(value - unique[-1]) > 1e-4:
            unique.append(value)
    return unique


def _top_surface_triangles(triangles: list[Triangle], min_z: float = 26.0) -> list[Triangle]:
    """Keep triangles that can contribute the exterior Z+ skin crossing."""
    return [tri for tri in triangles
            if min(float(tri.a[2]), float(tri.b[2]), float(tri.c[2])) > min_z]


def _fit_circle(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Small least-squares circle fit used only for the inner hole ring."""
    if len(points) < 3:
        return None
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for x, y in points:
        row = [2.0 * x, 2.0 * y, 1.0]
        rhs = x * x + y * y
        for i in range(3):
            for j in range(3):
                ata[i][j] += row[i] * row[j]
            atb[i] += row[i] * rhs
    matrix = [ata[i] + [atb[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(matrix[row][col]))
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        if abs(matrix[col][col]) < 1e-12:
            return None
        for row in range(col + 1, 3):
            factor = matrix[row][col] / matrix[col][col]
            for j in range(col, 4):
                matrix[row][j] -= factor * matrix[col][j]
    sol = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        sol[row] = matrix[row][3]
        for j in range(row + 1, 3):
            sol[row] -= matrix[row][j] * sol[j]
        sol[row] /= matrix[row][row]
    cx, cy = sol[0], sol[1]
    return cx, cy, max(0.0, sol[2] + cx * cx + cy * cy) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="Casing Top")
    ap.add_argument("--kind", choices=("m6", "m10", "gear"), default="m6")
    ap.add_argument("--radius", type=float, default=3.0,
                    help="probe ring radius in source asset units")
    ap.add_argument("--profile", action="store_true",
                    help="also report the open-radius profile at each bore")
    ap.add_argument("--asset", default=os.path.join(_REPO, "assets", "parts"))
    args = ap.parse_args()
    path = os.path.join(args.asset, f"{args.part}.usd")
    stage = Usd.Stage.Open(path)
    if stage is None:
        print(f"ERROR cannot open {path}")
        return 2
    triangles = _root_triangles(stage)
    # M10 casing pockets are cut into the lower flange (z≈20.25), whereas M6
    # hub bores are on the upper shell (z≈27.95).  Use the relevant exterior
    # skin for each audit rather than silently treating M10 pockets as absent.
    top_triangles = _top_surface_triangles(triangles, min_z=18.0 if args.kind == "m10" else 26.0)
    print(f"PART {args.part}: triangles={len(triangles)}")
    failures = 0
    offsets = ((0.0, 0.0), (args.radius, 0.0), (-args.radius, 0.0),
               (0.0, args.radius), (0.0, -args.radius))
    if args.kind == "m6":
        points_of_interest = M6_TOP
    elif args.kind == "m10":
        points_of_interest = M10_TOP
    else:
        points_of_interest = GEAR_TOP
    for x, y, name in points_of_interest:
        probes = [_crossings(triangles, x + dx, y + dy) for dx, dy in offsets]
        top_hits = [hits[0] if hits else None for hits in probes]
        # A center path is open when the central ray does not hit the top skin.
        # The ring probes distinguish a genuine opening from an empty location.
        blocked = top_hits[0] is not None and top_hits[0] > 20.0
        status = "BLOCKED" if blocked else "OPEN/UNCERTAIN"
        failures += blocked
        top_radii = []
        top_points = []
        for tri in top_triangles:
            for point in (tri.a, tri.b, tri.c):
                radial = ((float(point[0]) - x) ** 2 +
                          (float(point[1]) - y) ** 2) ** 0.5
                if radial < 20.0:
                    top_radii.append(radial)
                    if 1.0 <= radial <= 5.0:
                        top_points.append((float(point[0]), float(point[1])))
        nearest_top = min(top_radii) if top_radii else None
        fit = _fit_circle(top_points)
        print(f"BORE {status} {name} center=({x:.2f},{y:.2f}) "
              f"nearest_top_vertex={nearest_top!r} "
              f"inner_ring_fit={fit!r} "
              f"center_crossings={probes[0]} ring_crossings={probes[1:]}")
        if args.profile:
            profile = []
            for radius in [i * 0.5 for i in range(0, 25)]:
                # Four cardinal probes are enough to expose a small circular
                # opening and avoid depending on the exact triangulation seam.
                open_count = 0
                for dx, dy in ((radius, 0.0), (-radius, 0.0),
                               (0.0, radius), (0.0, -radius)):
                    hits = _crossings(top_triangles, x + dx, y + dy)
                    if not hits:
                        open_count += 1
                if open_count:
                    profile.append((radius, open_count))
            print(f"  OPEN_PROFILE {name}: {profile}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
