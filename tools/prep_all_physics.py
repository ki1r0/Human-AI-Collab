#!/usr/bin/env python3
"""tools/prep_all_physics.py — Batch-apply physics APIs for manual GUI assembly.

Iterates through every .usd part file in the assets directory and applies:

  Root Xform (/World):
      UsdPhysics.RigidBodyAPI   — simulated rigid body
      UsdPhysics.MassAPI        — mass = 0.2 kg

  Every Mesh prim:
      UsdPhysics.CollisionAPI       — collision detection
      UsdPhysics.MeshCollisionAPI   — approximation = "convexDecomposition"

convexDecomposition is critical so that shaft bores and gear teeth remain
physically open — convexHull would seal them shut and cause incorrect collisions.

Run:
    python3.11 tools/prep_all_physics.py
    python3.11 tools/prep_all_physics.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()


from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore

# ---------------------------------------------------------------------------
MASS_KG = 0.2
COLLISION_APPROX = "convexDecomposition"


def _find_all_meshes(stage, root_path):
    """Return all Mesh prims under root_path."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []
    return [p for p in Usd.PrimRange(root) if p.GetTypeName() == "Mesh"]


def prep_part(usd_path: str, dry_run: bool) -> bool:
    """Apply physics APIs to a single part USD file."""
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as exc:
        print(f"    [ERROR] Cannot open {usd_path}: {exc}")
        return False

    dp = stage.GetDefaultPrim()
    if dp is None or not dp.IsValid():
        print(f"    [ERROR] No defaultPrim in {usd_path}")
        return False

    root_path = str(dp.GetPath())
    meshes = _find_all_meshes(stage, root_path)

    if not meshes:
        print(f"    [WARN] No Mesh prims found under {root_path}")

    if dry_run:
        print(f"      [DRY] {root_path} <- RigidBodyAPI + MassAPI({MASS_KG} kg)")
        for m in meshes:
            print(f"      [DRY] {m.GetPath()} <- CollisionAPI + MeshCollisionAPI({COLLISION_APPROX})")
        return True

    # -- RigidBodyAPI on root --
    UsdPhysics.RigidBodyAPI.Apply(dp)
    print(f"      + {root_path}  RigidBodyAPI")

    # -- MassAPI on root --
    mass_api = UsdPhysics.MassAPI.Apply(dp)
    mass_api.GetMassAttr().Set(MASS_KG)
    print(f"      + {root_path}  MassAPI  mass={MASS_KG} kg")

    # -- CollisionAPI + MeshCollisionAPI on every mesh --
    for mesh_prim in meshes:
        mp = str(mesh_prim.GetPath())
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_col.GetApproximationAttr().Set(COLLISION_APPROX)
        print(f"      + {mp}  CollisionAPI + MeshCollisionAPI({COLLISION_APPROX})")

    stage.GetRootLayer().Save()
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parts-dir",
                        default=os.path.join(_REPO_ROOT, "assets", "parts"))
    args = parser.parse_args(argv)

    parts_dir = args.parts_dir
    print(f"\n{'='*65}")
    print(f"  Physics Prep — convexDecomposition for manual GUI assembly")
    print(f"  Parts dir : {os.path.relpath(parts_dir, _REPO_ROOT)}")
    print(f"  Mass      : {MASS_KG} kg")
    print(f"  Approx    : {COLLISION_APPROX}")
    if args.dry_run:
        print(f"  MODE      : DRY-RUN")
    print(f"{'='*65}\n")

    if not os.path.isdir(parts_dir):
        print(f"[ERROR] Parts directory not found: {parts_dir}")
        return 1

    usd_files = sorted(f for f in os.listdir(parts_dir) if f.endswith(".usd"))
    if not usd_files:
        print("[WARN] No .usd files found.")
        return 0

    n_ok = n_fail = 0
    for fname in usd_files:
        path = os.path.join(parts_dir, fname)
        print(f"  {'DRY-RUN' if args.dry_run else 'PREP   '}  {fname}")
        if prep_part(path, args.dry_run):
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{'='*65}")
    print(f"  Done: {n_ok} OK, {n_fail} failed")
    print(f"{'='*65}\n")

    if n_fail == 0:
        print("All assets have been prepped with Convex Decomposition physics.")
        print("You may now open them in Isaac Sim to manually find the socket/plug coordinates.\n")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
