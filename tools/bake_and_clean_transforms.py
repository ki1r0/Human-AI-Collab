#!/usr/bin/env python3
"""tools/bake_and_clean_transforms.py — Bake messy xformOpOrder stacks into clean SRT.

The CAD importer and XformCommonAPI generate xformOpOrder stacks that the
Isaac Sim UI Manipulator cannot handle (e.g. pivot ops, orient quaternions,
named scale suffixes). This script flattens target prims to a standard 3-op
stack:

    xformOp:translate
    xformOp:rotateXYZ
    xformOp:scale

Algorithm per baked prim:
    1. Evaluate the composed local transform matrix from the current ops.
    2. ClearXformOpOrder() — wipe the messy stack.
    3. Decompose the matrix via Gf.Transform into T, R (Euler XYZ), S.
    4. Add the three standard ops and set the decomposed values.

Run:
    python3.11 tools/bake_and_clean_transforms.py
    python3.11 tools/bake_and_clean_transforms.py --dry-run
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


from pxr import Gf, Usd, UsdGeom  # type: ignore

# The three op names that constitute a "clean" stack
_CLEAN_OPS = ("xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale")
_LOCAL_PART_HINTS = ("/parts/", "./parts/", "\\parts\\")
_RECONSTRUCTION_TOL = 1e-6
_DECOMP_STAGE = None
_DECOMP_PRIM = None
_RECOMP_STAGE = None
_RECOMP_PRIM = None


def _is_already_clean(ops: list) -> bool:
    """Return True if the op stack is already the standard TRS triple
    with correct precision (translate=Double, rotateXYZ=Float, scale=Float).
    """
    if len(ops) != 3:
        return False
    names = tuple(op.GetOpName() for op in ops)
    if names != _CLEAN_OPS:
        return False
    # Also verify precision: scale MUST be PrecisionFloat for XformCommonAPI
    precisions = [op.GetPrecision() for op in ops]
    expected = [
        UsdGeom.XformOp.PrecisionDouble,  # translate
        UsdGeom.XformOp.PrecisionFloat,   # rotateXYZ
        UsdGeom.XformOp.PrecisionFloat,   # scale
    ]
    return precisions == expected


def _evaluate_local_matrix(xformable, tc) -> Gf.Matrix4d:
    """Compute a prim's fully composed local-to-parent matrix via USD APIs."""
    return xformable.GetLocalTransformation(tc)


def _decompose_matrix(mat: Gf.Matrix4d):
    """Decompose a 4x4 matrix into (translate, rotateXYZ_degrees, scale).

    Uses XformCommonAPI accumulation on a temporary matrix-op prim so
    decomposition matches Isaac Sim manipulator conventions.
    """
    global _DECOMP_STAGE, _DECOMP_PRIM
    if _DECOMP_STAGE is None:
        _DECOMP_STAGE = Usd.Stage.CreateInMemory()
        _DECOMP_PRIM = UsdGeom.Xform.Define(_DECOMP_STAGE, "/_decompose").GetPrim()
    assert _DECOMP_PRIM is not None

    xf = UsdGeom.Xformable(_DECOMP_PRIM)
    xf.ClearXformOpOrder()
    for attr in [a.GetName() for a in _DECOMP_PRIM.GetAttributes() if a.GetName().startswith("xformOp:")]:
        _DECOMP_PRIM.RemoveProperty(attr)
    xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(mat)

    api = UsdGeom.XformCommonAPI(_DECOMP_PRIM)
    t, rotate_xyz, scale, _pivot, _order = api.GetXformVectorsByAccumulation(Usd.TimeCode.Default())
    return (
        Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])),
        Gf.Vec3f(float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2])),
        Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])),
    )


def _compose_matrix(translate: Gf.Vec3d, rotate_xyz: Gf.Vec3f, scale: Gf.Vec3d) -> Gf.Matrix4d:
    """Compose a local matrix using XformCommonAPI to match runtime semantics."""
    global _RECOMP_STAGE, _RECOMP_PRIM
    if _RECOMP_STAGE is None:
        _RECOMP_STAGE = Usd.Stage.CreateInMemory()
        _RECOMP_PRIM = UsdGeom.Xform.Define(_RECOMP_STAGE, "/_recompose").GetPrim()
    assert _RECOMP_PRIM is not None

    xf = UsdGeom.Xformable(_RECOMP_PRIM)
    xf.ClearXformOpOrder()
    for attr in [a.GetName() for a in _RECOMP_PRIM.GetAttributes() if a.GetName().startswith("xformOp:")]:
        _RECOMP_PRIM.RemoveProperty(attr)
    api = UsdGeom.XformCommonAPI(_RECOMP_PRIM)
    api.SetTranslate(translate)
    api.SetRotate(rotate_xyz, UsdGeom.XformCommonAPI.RotationOrderXYZ)
    api.SetScale(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))
    return xf.GetLocalTransformation(Usd.TimeCode.Default())


def _max_abs_matrix_diff(a: Gf.Matrix4d, b: Gf.Matrix4d) -> float:
    """Return maximum absolute element-wise difference between two 4x4 matrices."""
    max_diff = 0.0
    for row in range(4):
        for col in range(4):
            d = abs(float(a[row][col]) - float(b[row][col]))
            if d > max_diff:
                max_diff = d
    return max_diff


def _iter_asset_paths_from_list_op(list_op) -> list[str]:
    if not list_op:
        return []
    paths = []
    try:
        items = list_op.GetAddedOrExplicitItems()
    except Exception:
        items = []
    for item in items:
        asset_path = getattr(item, "assetPath", "") or ""
        if asset_path:
            paths.append(str(asset_path))
    return paths


def _prim_references_local_part(prim) -> bool:
    """Return True when a prim references/payloads a local part usd."""
    refs = _iter_asset_paths_from_list_op(prim.GetMetadata("references"))
    payloads = _iter_asset_paths_from_list_op(prim.GetMetadata("payload"))
    payloads += _iter_asset_paths_from_list_op(prim.GetMetadata("payloads"))
    for asset_path in refs + payloads:
        if any(hint in asset_path for hint in _LOCAL_PART_HINTS):
            return True
    return False


def bake_prim(prim, tc, dry_run: bool) -> bool:
    """Bake a single prim's xformOpOrder to clean TRS. Returns True if modified."""
    xf = UsdGeom.Xformable(prim)
    if not xf:
        return False

    ops = xf.GetOrderedXformOps()
    if not ops:
        return False  # no xform ops at all — skip

    if _is_already_clean(ops):
        return False  # already standard

    path = str(prim.GetPath())
    old_names = [op.GetOpName() for op in ops]

    # Step 1: Evaluate local matrix from current ops
    local_mat = _evaluate_local_matrix(xf, tc)

    # Step 2: Decompose
    translate, rotate_xyz, scale = _decompose_matrix(local_mat)

    reconstructed = _compose_matrix(translate, rotate_xyz, scale)
    max_diff = _max_abs_matrix_diff(local_mat, reconstructed)
    if max_diff > _RECONSTRUCTION_TOL:
        print(
            f"      [SKIP] {path}  decomposition_error={max_diff:.6g} "
            f"(keeping original ops)"
        )
        return False

    if dry_run:
        print(f"      [DRY] {path}")
        print(f"            old ops: {old_names}")
        print(f"            T=({translate[0]:.4f}, {translate[1]:.4f}, {translate[2]:.4f})")
        print(f"            R=({rotate_xyz[0]:.4f}, {rotate_xyz[1]:.4f}, {rotate_xyz[2]:.4f})")
        print(f"            S=({scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f})")
        print(f"            reconstruction_error={max_diff:.3e}")
        return True

    # Step 3: Clear old ops AND remove stale xformOp attributes
    xf.ClearXformOpOrder()
    # Remove all xformOp:* attributes so we can recreate with correct precision
    attrs_to_remove = [
        a.GetName() for a in prim.GetAttributes()
        if a.GetName().startswith("xformOp:")
    ]
    for attr_name in attrs_to_remove:
        prim.RemoveProperty(attr_name)

    # Step 4: Add clean TRS ops (precision must match XformCommonAPI expectations)
    # translate = PrecisionDouble (Vec3d), rotateXYZ = PrecisionFloat (Vec3f),
    # scale = PrecisionFloat (Vec3f) — Isaac Sim Move Tool requires this.
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(translate)
    xf.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(rotate_xyz)
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2]))
    )

    print(f"      + {path}  {old_names} -> [translate, rotateXYZ, scale]")
    return True


def process_file(usd_path: str, dry_run: bool, all_prims: bool = False) -> bool:
    """Open a USD file, bake all Xformable prims, save."""
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as exc:
        print(f"    [ERROR] Cannot open {usd_path}: {exc}")
        return False

    dp = stage.GetDefaultPrim()
    if dp is None or not dp.IsValid():
        print(f"    [ERROR] No defaultPrim in {usd_path}")
        return False

    tc = Usd.TimeCode.Default()
    n_baked = 0

    # Default mode only bakes defaultPrim, which is the manipulator target.
    # Full-depth mode is optional for one-off cleanup.
    prims = Usd.PrimRange(dp) if all_prims else [dp]
    for prim in prims:
        if bake_prim(prim, tc, dry_run):
            n_baked += 1

    if n_baked == 0:
        print(f"      (all prims already clean)")
        return True

    if not dry_run:
        stage.GetRootLayer().Save()
        print(f"      saved ({n_baked} prims baked)")

    return True


def process_scene_file(usd_path: str, dry_run: bool) -> bool:
    """Open a scene USD and bake prims that reference/payload local part files."""
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as exc:
        print(f"    [ERROR] Cannot open {usd_path}: {exc}")
        return False

    dp = stage.GetDefaultPrim()
    if dp is None or not dp.IsValid():
        print(f"    [ERROR] No defaultPrim in {usd_path}")
        return False

    tc = Usd.TimeCode.Default()
    n_baked = 0

    for prim in Usd.PrimRange(dp):
        if prim == dp:
            continue
        if _prim_references_local_part(prim):
            if bake_prim(prim, tc, dry_run):
                n_baked += 1

    if n_baked == 0:
        print(f"      (all scene prims already clean)")
        return True

    if not dry_run:
        stage.GetRootLayer().Save()
        print(f"      saved ({n_baked} scene prims baked)")

    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parts-dir",
                        default=os.path.join(_REPO_ROOT, "assets", "parts"))
    parser.add_argument("--file", dest="single_file",
                        help="Process a single USD file instead of the parts dir.")
    parser.add_argument("--scene", action="store_true",
                        help="Also bake gearbox prims in the scene file.")
    parser.add_argument("--all-prims", action="store_true",
                        help="Bake every Xformable prim in each part file (default: defaultPrim only).")
    args = parser.parse_args(argv)

    parts_dir = args.parts_dir
    print(f"\n{'='*65}")
    print(f"  Bake & Clean Transforms — flatten to standard TRS stack")
    print(f"  Target: [xformOp:translate, xformOp:rotateXYZ, xformOp:scale]")
    if args.single_file:
        print(f"  File: {args.single_file}")
    else:
        print(f"  Parts dir: {os.path.relpath(parts_dir, _REPO_ROOT)}")
    if args.scene:
        print(f"  Scene: will also process simple_room_scene.usd")
    if args.dry_run:
        print(f"  MODE: DRY-RUN")
    print(f"{'='*65}\n")

    n_ok = n_fail = 0

    if args.single_file:
        # Process a single file
        print(f"  {'DRY-RUN' if args.dry_run else 'BAKE   '}  {os.path.basename(args.single_file)}")
        if process_file(args.single_file, args.dry_run, all_prims=args.all_prims):
            n_ok += 1
        else:
            n_fail += 1
    else:
        # Process all parts
        if not os.path.isdir(parts_dir):
            print(f"[ERROR] Parts directory not found: {parts_dir}")
            return 1

        usd_files = sorted(f for f in os.listdir(parts_dir) if f.endswith(".usd"))
        if not usd_files:
            print("[WARN] No .usd files found.")
            return 0

        for fname in usd_files:
            path = os.path.join(parts_dir, fname)
            print(f"  {'DRY-RUN' if args.dry_run else 'BAKE   '}  {fname}")
            if process_file(path, args.dry_run, all_prims=args.all_prims):
                n_ok += 1
            else:
                n_fail += 1

    # Process scene file if requested
    if args.scene:
        scene_path = os.path.join(_REPO_ROOT, "assets", "simple_room_scene.usd")
        print(f"\n  {'DRY-RUN' if args.dry_run else 'BAKE   '}  simple_room_scene.usd (scene prims)")
        if process_scene_file(scene_path, args.dry_run):
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{'='*65}")
    print(f"  Done: {n_ok} OK, {n_fail} failed")
    print(f"{'='*65}\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
