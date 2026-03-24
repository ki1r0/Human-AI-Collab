#!/usr/bin/env python3
"""tools/prep_asset_physics.py — Apply UsdPhysics APIs to all gearbox part USDs.

Physics authoring per prim
--------------------------
Root Xform  (/World)
    UsdPhysics.RigidBodyAPI   — makes the prim a simulated rigid body
    UsdPhysics.MassAPI        — sets mass = 0.1 kg

Mesh prim   (/World/node_/mesh_)
    UsdPhysics.CollisionAPI       — enables collision detection
    UsdPhysics.MeshCollisionAPI   — sets approximation = convexDecomposition

PhysxSchema (Isaac Sim runtime-only)
    If PhysxSchema is importable (full Isaac Sim environment), additional
    PhysX-specific attributes are authored.  In a standalone pxr build the
    import fails silently and only the standard UsdPhysics layer is written.

Run from repo root:

    PYTHONPATH=<usd_lib> <python3.11> tools/prep_asset_physics.py
    PYTHONPATH=<usd_lib> <python3.11> tools/prep_asset_physics.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap — add USD lib and repo root to sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore

# PhysxSchema is only available inside a full Isaac Sim runtime.
try:
    from pxr import PhysxSchema  # type: ignore
    _HAS_PHYSX_SCHEMA = True
except ImportError:
    PhysxSchema = None
    _HAS_PHYSX_SCHEMA = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MASS_KG = 0.1
_MESH_APPROXIMATION = "convexDecomposition"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_mesh_prim(stage: Usd.Stage, root_path: str) -> Optional[Usd.Prim]:
    """Walk children of *root_path* depth-first and return the first Mesh prim."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Mesh":
            return prim
    return None


def _find_all_mesh_prims(stage: Usd.Stage, root_path: str) -> list:
    """Walk children of *root_path* depth-first and return ALL Mesh prims."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []
    return [prim for prim in Usd.PrimRange(root) if prim.GetTypeName() == "Mesh"]


def apply_physics(usd_path: str, dry_run: bool) -> bool:
    """Open *usd_path*, apply physics APIs, save in-place.

    Returns True on success, False on any error.
    """
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as exc:
        print(f"    [ERROR] Cannot open {usd_path}: {exc}")
        return False

    default_prim = stage.GetDefaultPrim()
    if default_prim is None or not default_prim.IsValid():
        print(f"    [ERROR] No defaultPrim in {usd_path}")
        return False

    root_path = str(default_prim.GetPath())  # typically /World

    # ── Mesh prims ───────────────────────────────────────────────────────────
    mesh_prims = _find_all_mesh_prims(stage, root_path)
    if not mesh_prims:
        print(f"    [WARN]  No Mesh prim found under {root_path} in {usd_path}")
        # Non-fatal: still apply rigid body to the root.

    if dry_run:
        print(f"      [DRY-RUN] {root_path} ← RigidBodyAPI, MassAPI({_DEFAULT_MASS_KG} kg)")
        for mp in mesh_prims:
            print(f"      [DRY-RUN] {mp.GetPath()} ← CollisionAPI, "
                  f"MeshCollisionAPI(approx={_MESH_APPROXIMATION})")
        if _HAS_PHYSX_SCHEMA:
            print(f"      [DRY-RUN] PhysxSchema available — would author PhysxRigidBodyAPI")
        return True

    # ── Apply RigidBodyAPI to root Xform ─────────────────────────────────────
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(default_prim)
    print(f"      + {root_path}  RigidBodyAPI")

    # ── Apply MassAPI to root Xform ───────────────────────────────────────────
    mass_api = UsdPhysics.MassAPI.Apply(default_prim)
    mass_api.GetMassAttr().Set(_DEFAULT_MASS_KG)
    print(f"      + {root_path}  MassAPI  mass={_DEFAULT_MASS_KG} kg")

    # ── Apply CollisionAPI + MeshCollisionAPI to ALL Mesh prims ──────────────
    for mesh_prim in mesh_prims:
        mesh_path = str(mesh_prim.GetPath())

        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        print(f"      + {mesh_path}  CollisionAPI")

        mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_col_api.GetApproximationAttr().Set(_MESH_APPROXIMATION)
        print(f"      + {mesh_path}  MeshCollisionAPI  approx={_MESH_APPROXIMATION}")

    # ── Optional PhysxSchema (Isaac Sim runtime only) ────────────────────────
    if _HAS_PHYSX_SCHEMA:
        try:
            PhysxSchema.PhysxRigidBodyAPI.Apply(default_prim)
            print(f"      + {root_path}  PhysxRigidBodyAPI  (runtime)")
            for mesh_prim in mesh_prims:
                PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
                print(f"      + {mesh_prim.GetPath()}  PhysxCollisionAPI  (runtime)")
        except Exception as exc:
            print(f"      [WARN]  PhysxSchema authoring failed (non-fatal): {exc}")

    # ── Save ─────────────────────────────────────────────────────────────────
    try:
        stage.GetRootLayer().Save()
    except Exception as exc:
        print(f"    [ERROR] Save failed for {usd_path}: {exc}")
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="prep_asset_physics.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be authored without modifying any files.",
    )
    parser.add_argument(
        "--parts-dir",
        default=os.path.join(_REPO_ROOT, "assets", "parts"),
        help="Directory containing part .usd files (default: assets/parts/).",
    )
    args = parser.parse_args(argv)

    parts_dir: str = args.parts_dir

    print(f"\n{'='*60}")
    print(f"  Physics Prep: UsdPhysics API authoring on gearbox parts")
    print(f"  Parts directory : {os.path.relpath(parts_dir, _REPO_ROOT)}")
    print(f"  Mass            : {_DEFAULT_MASS_KG} kg per part")
    print(f"  Approximation   : {_MESH_APPROXIMATION}")
    print(f"  PhysxSchema     : {'available' if _HAS_PHYSX_SCHEMA else 'NOT available (standalone pxr)'}")
    if args.dry_run:
        print("  MODE            : DRY-RUN — no files will be written")
    print(f"{'='*60}\n")

    if not os.path.isdir(parts_dir):
        print(f"[ERROR] Parts directory not found: {parts_dir}")
        return 1

    usd_files = sorted(
        f for f in os.listdir(parts_dir) if f.endswith(".usd")
    )

    if not usd_files:
        print("[WARN] No .usd files found in parts directory.")
        return 0

    n_ok = n_fail = 0

    for filename in usd_files:
        usd_path = os.path.join(parts_dir, filename)
        part_name = os.path.splitext(filename)[0]
        action = "DRY-RUN" if args.dry_run else "PHYSICS "
        print(f"  {action}  {filename}")
        ok = apply_physics(usd_path, dry_run=args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"  Done: {n_ok} OK, {n_fail} failed")
    print(f"{'='*60}\n")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
