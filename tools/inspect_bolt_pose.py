#!/usr/bin/env python3
"""Print source M6 bolt world bounds for the controlled mating pose."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths
ensure_pxr_paths()
from pxr import Gf, Usd, UsdGeom  # noqa: E402


def main():
    stage = Usd.Stage.Open(os.path.join(_REPO, "assets", "parts", "M6 Hub Bolt.usd"))
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    root_world = cache.GetLocalToWorldTransform(root)
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        # Controlled trial: uniform scene scale .002 and +90° about X.
        rotation = Gf.Matrix4d(1.0)
        rotation.SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0))
        to_root = cache.GetLocalToWorldTransform(prim) * root_world.GetInverse()
        groups = {"head": [], "shaft": []}
        for point in points:
            root_point = to_root.Transform(Gf.Vec3d(point))
            p = rotation.Transform(root_point) * 0.002
            groups["head" if float(root_point[1]) > 6.5 else "shaft"].append(p)
        print("MESH", prim.GetPath(), "points", len(points))
        for name, values in groups.items():
            if not values:
                continue
            mins = [min(float(p[i]) for p in values) for i in range(3)]
            maxs = [max(float(p[i]) for p in values) for i in range(3)]
            print(name, "source_bbox", mins, maxs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
