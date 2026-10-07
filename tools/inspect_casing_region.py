#!/usr/bin/env python3
"""Inspect source casing vertices around a nominated XY region."""

from __future__ import annotations

import argparse
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()
from pxr import Gf, Usd, UsdGeom  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="Casing Top")
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--radius", type=float, default=20.0)
    args = ap.parse_args()
    stage = Usd.Stage.Open(os.path.join(_REPO, "assets", "parts", f"{args.part}.usd"))
    if stage is None:
        raise SystemExit("cannot open part")
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    root_inv = cache.GetLocalToWorldTransform(root).GetInverse()
    points = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        to_root = cache.GetLocalToWorldTransform(prim) * root_inv
        for p in mesh.GetPointsAttr().Get() or []:
            q = to_root.Transform(Gf.Vec3d(p))
            d = math.hypot(float(q[0]) - args.x, float(q[1]) - args.y)
            if d <= args.radius:
                points.append((d, float(q[0]), float(q[1]), float(q[2])))
    print(f"PART {args.part} center=({args.x},{args.y}) points={len(points)}")
    if not points:
        return 0
    points.sort()
    print("NEAREST", [(round(d, 4), round(x, 4), round(y, 4), round(z, 4))
                       for d, x, y, z in points[:24]])
    bins = {}
    for _, _, _, z in points:
        key = round(z, 1)
        bins[key] = bins.get(key, 0) + 1
    print("Z_BINS", sorted(bins.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
