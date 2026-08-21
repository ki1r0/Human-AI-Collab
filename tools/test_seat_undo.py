#!/usr/bin/env python3
"""Offline smoke test for auto-seat + folded undo on flip/upright/put_down.

For each op: record the part's world transform, run the op with seat=True
(verify its bbox bottom lands on the tabletop), then /undo and verify the part
returns to its exact original transform (one undo reverses op + seat).

    docker compose run --rm hac tools/run_tool.sh tools/test_seat_undo.py
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

TC = Usd.TimeCode.Default()


def _world_matrix(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(TC)


def _bbox_bottom(prim):
    cache = UsdGeom.BBoxCache(TC, includedPurposes=[UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return None if rng.IsEmpty() else float(rng.GetMin()[2])


def _matrix_max_diff(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4))


# Round-trip tolerance: the rotation primitives author Euler rotateXYZ xformOps,
# whose decomposition round-trips to ~1e-4 (worst near 90°). 1e-3 ≈ 0.3 mm /
# 0.06°, well below any physical relevance for magic assembly.
RT_TOL = 1e-3
SEAT_TOL = 1e-4


def _run_case(ma, stage, part, label, call, seat):
    path, prim = ma._resolve_target(stage, part)
    m0 = _world_matrix(prim)
    surface_z = ma._resolve_surface_z(stage, "table")
    expect_bottom = float(surface_z) + 0.005

    ok = call()
    bottom = _bbox_bottom(prim)
    seat_err = abs(bottom - expect_bottom)
    # Only assert contact when seating was requested.
    seated_ok = ok and (seat_err <= SEAT_TOL if seat else True)

    undone = ma.undo()
    m2 = _world_matrix(prim)
    rt_err = _matrix_max_diff(m0, m2)
    roundtrip_ok = undone and rt_err <= RT_TOL

    status = "PASS" if (seated_ok and roundtrip_ok) else "FAIL"
    seat_str = f"seated bottom={bottom:.4f} err {seat_err:.1e}" if seat else "no-seat (control)"
    print(f"  [{status}] {label:18s} | {seat_str:28s} | "
          f"undo round-trip max|Δmatrix|={rt_err:.1e}")
    return seated_ok and roundtrip_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="Input_Shaft")
    ap.add_argument("--scene", default=os.path.join(_REPO_ROOT, "assets", "simple_room_scene.usd"))
    args = ap.parse_args()

    print(f"\n{'='*72}\n  seat+undo smoke test — part={args.part!r} scene={os.path.basename(args.scene)}\n{'='*72}")
    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        print("[ERROR] could not open stage")
        return 1
    ma = MagicAssemblyManager(stage_fn=lambda: stage, use_omni_commands=False,
                              logger=lambda m: None)  # quiet; we print our own lines

    p = args.part
    cases = [
        # (label, call, seat) — no-seat controls isolate the rotation-op error.
        ("flip (no-seat)",    lambda: ma.flip(p, axis="x", seat=False),         False),
        ("flip (seat)",       lambda: ma.flip(p, axis="x", seat=True),          True),
        ("upright (no-seat)", lambda: ma.upright(p, axis="x", seat=False),      False),
        ("upright (seat)",    lambda: ma.upright(p, axis="x", seat=True),       True),
        ("put_down (seat)",   lambda: ma.put_down(p, drop=0.15, seat=True),     True),
    ]
    all_ok = True
    for label, call, seat in cases:
        all_ok &= _run_case(ma, stage, p, label, call, seat)

    print(f"\n{'='*72}\n  RESULT: {'PASS' if all_ok else 'FAIL'}\n{'='*72}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
