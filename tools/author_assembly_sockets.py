#!/usr/bin/env python3
"""tools/author_assembly_sockets.py — Add Socket and Plug Xforms to real part USDs.

Socket / Plug convention
------------------------
Socket  an Xform child of the parent prim that defines where a child part
        snaps TO.  Named "socket_<role>" (e.g. socket_center, socket_top).
Plug    an Xform child of the child prim that defines the point that snaps
        ONTO a socket.  Named "plug" or "plug_<role>".

These Xforms are added directly under each part's defaultPrim (/World) so
that, when the part USD is referenced into a scene, the socket/plug prims
appear as direct children of the reference prim — exactly where
MagicAssemblyManager._find_socket_world() and _find_plug_local() search.

Geometry notes (all parts are Y-up, metersPerUnit=0.01 i.e. mm units,
already centered so centroid = world origin):

  Part                 Long axis  Half-extent used for endpoint sockets
  -------------------------------------------------------------------
  Input Shaft          Y          ±102.5 mm
  Output Shaft         Y          ±84.8  mm
  Transfer Shaft       X          ±50.1  mm  (lies along X in part space)
  M10 Casing Bolt      Y          ±58.2  mm
  M6 Hub Bolt          Y          ±13.1  mm
  Output Gear          flat XY    thin Z  (disc, centre at origin)
  Transfer Gear        flat XY    thin Z  (disc, centre at origin)
  Casing Base          flat box   Z ±27.9 mm (housing, sockets at centre)
  Casing Top           flat box   Z ±27.9 mm
  Hub Cover Input/Small thin flat Y
  Hub Cover Output      flat XZ
  Breather Plug         round     all ≈12.6 mm
  Oil Level Indicator   round     all ≈11 mm
  M10 Casing Nut        thin disc

Run from repo root:

    PYTHONPATH=<usd_lib> <python3.11> tools/author_assembly_sockets.py
    PYTHONPATH=<usd_lib> <python3.11> tools/author_assembly_sockets.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Gf, Usd, UsdGeom  # type: ignore


# ---------------------------------------------------------------------------
# Socket / Plug catalogue
# ---------------------------------------------------------------------------
# Key  : part name (without .usd extension)
# Value: list of (xform_name, (tx, ty, tz))
#
# Coordinate frame: Y-up (part file space), millimetre units.
# All parts are centred so (0, 0, 0) = geometric centroid.
#
# Naming rules kept consistent with MagicAssemblyManager lookups:
#   socket_*  → _find_socket_world() candidate list (exact or startswith)
#   plug*     → _find_plug_local()   candidate list (exact or startswith)

Vec3 = Tuple[float, float, float]
Attachment = Tuple[str, Vec3]

PART_ATTACHMENTS: Dict[str, List[Attachment]] = {
    # ── Gearbox housing ──────────────────────────────────────────────────
    "Casing Base": [
        ("socket_center",      (0.0,    0.0,   0.0)),   # kinematic assembly anchor
        ("socket_shaft_input", (0.0,    0.0,   0.0)),   # alias — shaft enters centre
        ("socket_bolt_a",      (75.0,  100.0,  20.0)),  # corner bolt seats (approx.)
        ("socket_bolt_b",      (-75.0, 100.0,  20.0)),
        ("socket_bolt_c",      (75.0, -100.0,  20.0)),
        ("socket_bolt_d",      (-75.0,-100.0,  20.0)),
        ("plug",               (0.0,    0.0,   0.0)),   # if casing snaps onto a rig
    ],
    "Casing Top": [
        ("socket_center",      (0.0,   0.0,  0.0)),
        ("plug",               (0.0,   0.0,  0.0)),
    ],
    # ── Shafts ────────────────────────────────────────────────────────────
    # Input Shaft: cylinder along Y, half-extent ±102.5 mm
    "Input Shaft": [
        ("plug",               (0.0, -102.5,  0.0)),   # bottom/drive end
        ("socket_top",         (0.0,  102.5,  0.0)),   # output end (gear mounts here)
        ("socket_center",      (0.0,   0.0,   0.0)),
    ],
    # Output Shaft: cylinder along Y, half-extent ±84.8 mm
    "Output Shaft": [
        ("plug",               (0.0,  -84.8,  0.0)),
        ("socket_top",         (0.0,   84.8,  0.0)),
        ("socket_center",      (0.0,    0.0,  0.0)),
    ],
    # Transfer Shaft: cylinder along X, half-extent ±50.1 mm
    "Transfer Shaft": [
        ("plug",               (-50.1,  0.0,  0.0)),   # one end
        ("socket_top",         ( 50.1,  0.0,  0.0)),   # other end
        ("socket_center",      (  0.0,  0.0,  0.0)),
    ],
    # ── Gears (flat discs, centred) ────────────────────────────────────
    "Output Gear": [
        ("plug",               (0.0,  0.0,  0.0)),     # bore centre — snaps to shaft
        ("socket_center",      (0.0,  0.0,  0.0)),
    ],
    "Transfer Gear": [
        ("plug",               (0.0,  0.0,  0.0)),
        ("socket_center",      (0.0,  0.0,  0.0)),
    ],
    # ── Hub covers ─────────────────────────────────────────────────────
    "Hub Cover Input": [
        ("plug",               (0.0,  0.0,  0.0)),
        ("socket_center",      (0.0,  0.0,  0.0)),
    ],
    "Hub Cover Output": [
        ("plug",               (0.0,  0.0,  0.0)),
        ("socket_center",      (0.0,  0.0,  0.0)),
    ],
    "Hub Cover Small": [
        ("plug",               (0.0,  0.0,  0.0)),
        ("socket_center",      (0.0,  0.0,  0.0)),
    ],
    # ── Fasteners ──────────────────────────────────────────────────────
    # M10 Casing Bolt: along Y, half-extent ±58.2 mm
    "M10 Casing Bolt": [
        ("plug",               (0.0, -58.2,  0.0)),    # threaded tip
        ("socket_center",      (0.0,   0.0,  0.0)),
    ],
    "M10 Casing Nut": [
        ("plug",               (0.0,  0.0,  0.0)),
    ],
    # M6 Hub Bolt: along Y, half-extent ±13.1 mm
    "M6 Hub Bolt": [
        ("plug",               (0.0, -13.1,  0.0)),
        ("socket_center",      (0.0,   0.0,  0.0)),
    ],
    # ── Accessories ────────────────────────────────────────────────────
    "Breather Plug": [
        ("plug",               (0.0,  0.0,  0.0)),
    ],
    "Oil Level Indicator": [
        ("plug",               (0.0,  0.0,  0.0)),
    ],
}


# ---------------------------------------------------------------------------
# Core authoring
# ---------------------------------------------------------------------------

def _add_xform(stage, parent_path: str, name: str, translate: Vec3) -> bool:
    """Define or overwrite an Xform child with a translate op.

    Returns True if the prim was newly created (or already existed and was
    updated), False if it had to be skipped for a naming clash.
    """
    prim_path = parent_path.rstrip("/") + "/" + name
    existing = stage.GetPrimAtPath(prim_path)

    if existing and existing.IsValid():
        # Update existing translate — idempotent re-run support.
        xf = UsdGeom.Xformable(existing)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and not op.IsInverseOp():
                op.Set(Gf.Vec3d(translate[0], translate[1], translate[2]))
                return True
        # No translate op yet — add one.
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(translate[0], translate[1], translate[2])
        )
        return True

    # Define new Xform.
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(translate[0], translate[1], translate[2])
    )
    return True


def author_part(usd_path: str, attachments: List[Attachment], dry_run: bool) -> bool:
    """Open *usd_path*, add/update all attachment Xforms, save in-place.

    Returns True on success.
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

    parent_path = str(default_prim.GetPath())  # typically "/World"

    for name, translate in attachments:
        if dry_run:
            print(f"      [DRY-RUN] {parent_path}/{name}  translate={translate}")
        else:
            _add_xform(stage, parent_path, name, translate)
            print(f"      + {parent_path}/{name}  ({translate[0]:+.1f}, {translate[1]:+.1f}, {translate[2]:+.1f})")

    if not dry_run:
        try:
            stage.GetRootLayer().Save()
        except Exception as exc:
            print(f"    [ERROR] Save failed: {exc}")
            return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="author_assembly_sockets.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be authored without writing files.")
    args = parser.parse_args(argv)

    parts_dir = os.path.join(_REPO_ROOT, "assets", "parts")

    n_ok = n_fail = n_skip = 0

    print(f"\n{'='*60}")
    print(f"  Authoring Socket/Plug Xforms on real gearbox parts")
    print(f"  Parts directory: {os.path.relpath(parts_dir, _REPO_ROOT)}")
    if args.dry_run:
        print("  MODE: DRY-RUN — no files will be written")
    print(f"{'='*60}\n")

    for part_name, attachments in PART_ATTACHMENTS.items():
        usd_path = os.path.join(parts_dir, part_name + ".usd")
        if not os.path.isfile(usd_path):
            print(f"  SKIP  {part_name}.usd  (file not found)")
            n_skip += 1
            continue

        action = "DRY-RUN" if args.dry_run else "AUTHORING"
        print(f"  {action}  {part_name}.usd")
        ok = author_part(usd_path, attachments, dry_run=args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"  Done: {n_ok} OK, {n_fail} failed, {n_skip} skipped")
    print(f"{'='*60}\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
