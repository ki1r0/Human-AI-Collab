#!/usr/bin/env python3
"""Offline smoke test for MagicAssemblyManager.hover().

Runs the real hover() code path against the scene USD (no Kit / no GUI), so we
can verify the bbox-contact Z logic without a live Isaac Sim session.

Run via the repo wrapper (handles pxr / LD_LIBRARY_PATH):
    docker compose run --rm hac tools/run_tool.sh tools/test_hover.py
    docker compose run --rm hac tools/run_tool.sh tools/test_hover.py --part Input_Shaft
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from tools._bootstrap import ensure_pxr_paths  # noqa: E402

ensure_pxr_paths()

from pxr import Usd, UsdGeom  # noqa: E402

from runtime.magic_assembly import MagicAssemblyManager  # noqa: E402


def _world_bbox_z(stage, prim, tc):
    cache = UsdGeom.BBoxCache(tc, includedPurposes=[UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None, None
    return float(rng.GetMin()[2]), float(rng.GetMax()[2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="Input_Shaft")
    ap.add_argument("--surface", default="table")
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument(
        "--scene",
        default=os.path.join(_REPO_ROOT, "assets", "simple_room_scene.usd"),
    )
    args = ap.parse_args()

    print(f"\n{'='*64}\n  hover() smoke test — part={args.part!r} scene={os.path.basename(args.scene)}\n{'='*64}")

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        print("[ERROR] could not open stage")
        return 1
    tc = Usd.TimeCode.Default()

    ma = MagicAssemblyManager(
        stage_fn=lambda: stage,
        use_omni_commands=False,          # offline: pure USD edits, no Kit commands
        logger=lambda m: print("   ", m),
    )

    # Locate the part prim through the manager's own resolver.
    part_path, part_prim = ma._resolve_target(stage, args.part)
    if part_prim is None:
        print(f"[FAIL] part {args.part!r} not found in scene")
        return 1

    # Surface height as hover() will compute it.
    surface_z = ma._resolve_surface_z(stage, args.surface)
    detected = ma._table_surface_top_z(stage) if args.surface == "table" else None
    print(f"\n  resolved surface {args.surface!r} -> z = {surface_z}")
    if args.surface == "table":
        print(f"  detected tabletop mesh top z = {detected}")
        if detected is None:
            print("  [WARN] tabletop NOT detected — surface came from fallback constant")

    before_min, before_max = _world_bbox_z(stage, part_prim, tc)
    print(f"  BEFORE: bbox bottom z = {before_min:.4f}   (top z = {before_max:.4f})")

    ok = ma.hover(args.part, surface=args.surface, margin=args.margin)
    print(f"\n  hover() returned: {ok}")

    after_min, after_max = _world_bbox_z(stage, part_prim, tc)
    expected = float(surface_z) + float(args.margin)
    print(f"  AFTER : bbox bottom z = {after_min:.4f}   (top z = {after_max:.4f})")
    print(f"  EXPECT: bottom ≈ surface+margin = {expected:.4f}")

    tol = 1e-4
    passed = ok and abs(after_min - expected) <= tol
    # Orientation/XY preserved: height (top-bottom span) must be unchanged.
    span_before = before_max - before_min
    span_after = after_max - after_min
    span_ok = abs(span_before - span_after) <= tol
    print(f"\n  bbox height preserved (Z-only move): {span_before:.4f} -> {span_after:.4f} "
          f"[{'OK' if span_ok else 'CHANGED'}]")

    print(f"\n{'='*64}")
    print(f"  RESULT: {'PASS' if (passed and span_ok) else 'FAIL'} "
          f"(contact err = {abs(after_min - expected):.2e})")
    print(f"{'='*64}\n")
    # Do NOT save the stage — this is a read-only smoke test.
    return 0 if (passed and span_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
