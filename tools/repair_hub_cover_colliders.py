#!/usr/bin/env python3
"""Use concave SDF colliders for the three hub covers.

Their mounting bosses and recessed faces are not faithfully represented by a
convex decomposition.  The visible meshes are unchanged; only the authored
collision approximation is upgraded, with an optional high-resolution PhysX
SDF attribute when running inside Isaac Sim.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths
ensure_pxr_paths()

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

try:
    from pxr import PhysxSchema  # type: ignore
except ImportError:
    PhysxSchema = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", default=os.path.join(_REPO, "assets", "parts"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for name in ("Hub Cover Output", "Hub Cover Input", "Hub Cover Small"):
        path = os.path.join(args.parts_dir, f"{name}.usd")
        stage = Usd.Stage.Open(path)
        changed = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            approximation = UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr()
            if approximation.Get() != "sdf":
                approximation.Set("sdf")
                changed += 1
            if PhysxSchema is not None and not args.dry_run:
                sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                sdf.GetSdfResolutionAttr().Set(256)
        if not args.dry_run:
            stage.GetRootLayer().Save()
        print(f"REPAIR {'PLAN' if args.dry_run else 'APPLY'} {name}: "
              f"collision_meshes_updated={changed} approximation=sdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
