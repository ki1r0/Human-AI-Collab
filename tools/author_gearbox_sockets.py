#!/usr/bin/env python3
"""tools/author_gearbox_sockets.py — Author assembly-manual-accurate Socket/Plug Xforms.

Two strategies:
  1. "Parent" parts (Casing_Top, Casing_Base) — sockets are placed at positions
     derived from circle-fitting the bore openings on the mesh geometry.
  2. "Child" parts (hub covers, gears, shafts, bolts, etc.) — plug_main is
     placed at the **circular feature center** for disc-shaped parts (hub covers)
     or at the BBox centroid for other parts.

Run:
    python3 tools/author_gearbox_sockets.py
    python3 tools/author_gearbox_sockets.py --dry-run
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()


from pxr import Gf, Sdf, Tf, Usd, UsdGeom  # type: ignore

# ═══════════════════════════════════════════════════════════════════════════
#  PARENT PART SOCKETS — bore centers from circle-fitting mesh geometry
# ═══════════════════════════════════════════════════════════════════════════
# Coordinates are in mm, part-local (mesh geometry centered at origin by
# the node_ offset).  Casing half-thickness = 27.95 mm.
#
# Bore centers (from iterative circle fit on bore-edge vertices):
#   Output (large):   (  0.0,  39.75 ) — user-tuned bore center for output shaft install
#   Input (right):    ( 31.0, -49.69)  — inner bore edge fit, 668 vertices
#   Transfer (left):  (-31.0, -49.69)  — inner bore edge fit, 660 vertices

PARENT_SOCKET_CONFIG: Dict[str, Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]]] = {
    "Casing Top": {
        "sockets": {
            # Hub cover sockets on EXTERIOR face (Z+)
            # Output bore center refined from ring-fit on top face.
            "socket_hub_output":  {"translate": (0.0,   39.75,  27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_hub_input":   {"translate": (31.0, -49.69,  27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_hub_small":   {"translate": (-31.0,-49.69,  27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            # M6 hub bolt sockets (exterior face, on bore flange bosses)
            "socket_bolt_hub_1":  {"translate": (40.0,   0.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_2":  {"translate": (-40.0,  0.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_3":  {"translate": (40.0,  80.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_4":  {"translate": (-40.0, 80.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_5":  {"translate": (54.0, -27.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_6":  {"translate": (-54.0,-27.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
        },
        "plugs": {
            "plug_casing_mate":   {"translate": (0.0,   0.0,  -27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
        },
    },
    "Casing Base": {
        "sockets": {
            # Hub cover sockets on EXTERIOR face (Z-)
            "socket_hub_output":   {"translate": (0.0,   39.75, -27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_hub_small_1":  {"translate": (31.0, -49.69, -27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_hub_small_2":  {"translate": (-31.0,-49.69, -27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            # M6 hub bolt sockets mirrored directly from Casing Top as requested.
            "socket_bolt_hub_1":   {"translate": (40.0,   0.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_2":   {"translate": (-40.0,  0.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_3":   {"translate": (40.0,  80.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_4":   {"translate": (-40.0, 80.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_5":   {"translate": (54.0, -27.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_6":   {"translate": (-54.0,-27.0,   27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_8":   {"translate": (8.26, -26.47,  27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_9":   {"translate": (54.0, -72.69,  27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_10":  {"translate": (7.76, -72.52,  27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_11":  {"translate": (-7.96,-26.37,  27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_13":  {"translate": (-8.0, -72.69,  27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_bolt_hub_14":  {"translate": (-53.82,-72.43, 27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            # Internal gear sockets (center plane Z=0) — step 11
            "socket_gear_input":   {"translate": (31.0, -49.69,   0.0), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_gear_transfer":{"translate": (-31.0,-49.69,   0.0), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_gear_output":  {"translate": (0.0,   39.75,   0.0), "rotate_xyz": (0.0, 0.0, 0.0)},
            # Casing mate socket — receives Casing_Top (parting face Z+)
            "socket_casing_mate":  {"translate": (0.0,   0.0,    27.9), "rotate_xyz": (0.0, 0.0, 0.0)},
            # Oil level indicator sockets (side holes)
            "socket_oil_1":        {"translate": (95.0,   0.0,    0.0), "rotate_xyz": (0.0, 0.0, 0.0)},
            "socket_oil_2":        {"translate": (-95.0,  0.0,    0.0), "rotate_xyz": (0.0, 0.0, 0.0)},
            # Breather plug socket
            "socket_breather":     {"translate": (0.0,  138.6,    0.0), "rotate_xyz": (0.0, 0.0, 0.0)},
        },
        "plugs": {},
    },
}

# ═══════════════════════════════════════════════════════════════════════════
#  CHILD PART SCHEMA — plug_main at circle center or BBox centroid
# ═══════════════════════════════════════════════════════════════════════════
# Disc-shaped parts (hub covers) use circle fitting to find the disc center.
# Other parts use BBox centroid.
#
# "plug_mode": "disc" → compute_disc_circle_center() (least-squares circle
#   fit on outer-ring vertices in the cross-section plane).
# "plug_mode": "bbox" (default) → compute_mesh_centroid() (BBox midpoint).

CHILD_PART_SCHEMA: Dict[str, Dict[str, object]] = {
    "Input Shaft":         {"sockets": [{"name": "socket_gear", "translate": (0.0, 0.0, 0.0)}]},
    "Transfer Shaft":      {"sockets": [{"name": "socket_gear", "translate": (0.0, 0.0, 0.0)}]},
    "Output Shaft":        {"sockets": [{"name": "socket_gear", "translate": (0.0, 0.0, 0.0)}]},
    "Transfer Gear":       {"sockets": []},
    "Output Gear":         {"sockets": []},
    # plug_main is explicitly pinned to the fitted mounting-circle center.
    "Hub Cover Output":    {"sockets": [], "plug_mode": "fixed", "plug_translate": (0.0, 0.0, -3.66859)},
    "Hub Cover Input":     {"sockets": [], "plug_mode": "disc"},
    "Hub Cover Small":     {"sockets": [], "plug_mode": "disc"},
    "M6 Hub Bolt":         {"sockets": []},
    "M10 Casing Bolt":     {"sockets": []},
    "M10 Casing Nut":      {"sockets": []},
    "Oil Level Indicator":  {"sockets": []},
    "Breather Plug":       {"sockets": []},
}


# ---------------------------------------------------------------------------
# Geometry computation
# ---------------------------------------------------------------------------

def _get_root_local_points(stage: Usd.Stage, root_prim: Usd.Prim) -> list:
    """Collect all mesh vertices transformed into root-prim-local coords."""
    pts = []
    xf_cache = UsdGeom.XformCache()
    root_to_world = xf_cache.GetLocalToWorldTransform(root_prim)
    root_to_world_inv = root_to_world.GetInverse()
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        prim_to_world = xf_cache.GetLocalToWorldTransform(prim)
        prim_to_root = prim_to_world * root_to_world_inv
        for p in points:
            wp = prim_to_root.Transform(Gf.Vec3d(p[0], p[1], p[2]))
            pts.append((wp[0], wp[1], wp[2]))
    return pts


def _lstsq_circle_fit(coords_2d: list) -> Tuple[float, float, float]:
    """Least-squares circle fit on 2D points. Returns (cx, cy, radius).

    Solves the over-determined system:
      2*x*a + 2*y*b + c = x^2 + y^2
    where a=cx, b=cy, c = r^2 - cx^2 - cy^2.
    """
    n = len(coords_2d)
    # Build matrices manually (no numpy dependency)
    # A^T A x = A^T b  (3x3 normal equations)
    ata = [[0.0]*3 for _ in range(3)]
    atb = [0.0]*3
    for x, y in coords_2d:
        row = [2.0*x, 2.0*y, 1.0]
        rhs = x*x + y*y
        for i in range(3):
            for j in range(3):
                ata[i][j] += row[i] * row[j]
            atb[i] += row[i] * rhs

    # Solve 3x3 system via Gaussian elimination
    M = [ata[i][:] + [atb[i]] for i in range(3)]
    for col in range(3):
        # Partial pivoting
        max_row = col
        for row in range(col+1, 3):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-12:
            return 0.0, 0.0, 0.0
        for row in range(col+1, 3):
            factor = M[row][col] / pivot
            for j in range(col, 4):
                M[row][j] -= factor * M[col][j]
    # Back substitution
    sol = [0.0]*3
    for i in range(2, -1, -1):
        sol[i] = M[i][3]
        for j in range(i+1, 3):
            sol[i] -= M[i][j] * sol[j]
        sol[i] /= M[i][i]

    cx, cy = sol[0], sol[1]
    r_sq = sol[2] + cx*cx + cy*cy
    radius = math.sqrt(max(0.0, r_sq))
    return cx, cy, radius


def compute_disc_circle_center(
    stage: Usd.Stage, root_prim: Usd.Prim
) -> Optional[Gf.Vec3d]:
    """Compute the center of the mounting face circle via least-squares fit.

    The mounting face is the flat face of the disc-shaped part that seats into
    a casing bore.  This function:

    1. Identifies the thin axis (disc normal direction).
    2. Selects vertices on the positive-direction face (within 0.5 mm of max).
    3. Finds the innermost ring of vertices (the mounting rim) on that face.
    4. Fits a circle to the ring via least-squares.
    5. Returns the 3D circle center (with thin-axis component = BBox midpoint).

    This gives the center of the mounting rim, which may differ from the BBox
    centroid when the mounting boss is eccentric to the disc body.
    """
    pts = _get_root_local_points(stage, root_prim)
    if len(pts) < 10:
        return None

    # BBox
    mins = [min(p[i] for p in pts) for i in range(3)]
    maxs = [max(p[i] for p in pts) for i in range(3)]
    extents = [maxs[i] - mins[i] for i in range(3)]
    mids = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]

    # Thin axis = disc normal (smallest extent)
    thin_axis = extents.index(min(extents))
    plane_axes = [i for i in range(3) if i != thin_axis]
    pa, pb = plane_axes

    # Get vertices on the positive face of the thin axis
    face_max = maxs[thin_axis]
    face_pts = [(p[pa], p[pb]) for p in pts if p[thin_axis] > face_max - 0.5]

    if len(face_pts) < 10:
        return None

    # Compute BBox center of face vertices
    face_a = [fp[0] for fp in face_pts]
    face_b = [fp[1] for fp in face_pts]
    center_a = (min(face_a) + max(face_a)) / 2.0
    center_b = (min(face_b) + max(face_b)) / 2.0

    # Compute distances from face BBox center
    dists = [math.sqrt((fp[0] - center_a)**2 + (fp[1] - center_b)**2)
             for fp in face_pts]

    # Find the innermost ring with enough vertices (>= 40 pts).
    # Bin distances into 1mm-wide rings and find the first dense ring.
    max_dist = max(dists) if dists else 0
    bin_width = 1.0
    n_bins = max(1, int(max_dist / bin_width) + 1)

    for b_idx in range(n_bins):
        r_lo = b_idx * bin_width
        r_hi = r_lo + bin_width
        ring = [(face_pts[i][0], face_pts[i][1])
                for i in range(len(face_pts))
                if r_lo <= dists[i] < r_hi + 1.0]  # ±1mm tolerance
        if len(ring) >= 40:
            cx, cy, radius = _lstsq_circle_fit(ring)
            if radius > 1.0:
                result = [0.0, 0.0, 0.0]
                result[pa] = cx
                result[pb] = cy
                result[thin_axis] = mids[thin_axis]
                return Gf.Vec3d(result[0], result[1], result[2])

    # Fallback: fit all face vertices
    cx, cy, radius = _lstsq_circle_fit(face_pts)
    if radius < 1.0:
        return None

    result = [0.0, 0.0, 0.0]
    result[pa] = cx
    result[pb] = cy
    result[thin_axis] = mids[thin_axis]
    return Gf.Vec3d(result[0], result[1], result[2])


def compute_mesh_centroid(
    stage: Usd.Stage, root_prim: Usd.Prim
) -> Optional[Gf.Vec3d]:
    """Compute the BBox centroid of all Mesh descendants in root-local space."""
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )
    bbox = cache.ComputeLocalBound(root_prim)
    rng = bbox.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mid = rng.GetMidpoint()
    return Gf.Vec3d(mid[0], mid[1], mid[2])


# ---------------------------------------------------------------------------
# XformCommonAPI helper
# ---------------------------------------------------------------------------

def _apply_xform_common(stage: Usd.Stage, parent_path: str, name: str,
                         translate: Tuple[float, float, float],
                         rotate_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> bool:
    """Define or update an Xform child using UsdGeom.XformCommonAPI."""
    prim_path = parent_path.rstrip("/") + "/" + name
    existing = stage.GetPrimAtPath(prim_path)

    if existing and existing.IsValid():
        xf = UsdGeom.Xformable(existing)
        xf.ClearXformOpOrder()
    else:
        UsdGeom.Xform.Define(stage, prim_path)

    prim = stage.GetPrimAtPath(prim_path)
    common = UsdGeom.XformCommonAPI(prim)
    common.SetXformVectors(
        Gf.Vec3d(*translate),
        Gf.Vec3f(*rotate_xyz),
        Gf.Vec3f(1.0, 1.0, 1.0),
        Gf.Vec3f(0.0, 0.0, 0.0),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
        Usd.TimeCode.Default(),
    )
    return True


# ---------------------------------------------------------------------------
# Authoring functions
# ---------------------------------------------------------------------------

def author_parent_part(usd_path: str, part_config: dict, dry_run: bool) -> bool:
    """Author parent part (casing) with hardcoded socket/plug positions."""
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
    sockets = part_config.get("sockets", {})
    plugs = part_config.get("plugs", {})
    all_names = set(sockets.keys()) | set(plugs.keys())

    if dry_run:
        for name, cfg in {**sockets, **plugs}.items():
            t = cfg["translate"]
            print(f"      [DRY] {root_path}/{name}  translate=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
        return True

    # Remove stale socket_*/plug_* Xforms not in the new config
    for child in dp.GetChildren():
        cname = child.GetName()
        if (cname.startswith("socket_") or cname.startswith("plug_")) \
                and cname not in all_names:
            stage.RemovePrim(child.GetPath())

    for name, cfg in {**sockets, **plugs}.items():
        _apply_xform_common(stage, root_path, name,
                            translate=cfg["translate"],
                            rotate_xyz=cfg["rotate_xyz"])

    stage.GetRootLayer().Save()
    return True


def author_child_part(usd_path: str, part_name: str, schema: dict,
                      dry_run: bool) -> Tuple[bool, Optional[Gf.Vec3d]]:
    """Author child part — plug_main at circle center or BBox centroid."""
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as exc:
        print(f"    [ERROR] Cannot open {usd_path}: {exc}")
        return False, None

    dp = stage.GetDefaultPrim()
    if dp is None or not dp.IsValid():
        print(f"    [ERROR] No defaultPrim in {usd_path}")
        return False, None

    root_path = str(dp.GetPath())

    # --- Compute plug position ---
    plug_mode = str(schema.get("plug_mode", "bbox"))
    if plug_mode == "fixed":
        fixed = schema.get("plug_translate")
        if not (isinstance(fixed, (tuple, list)) and len(fixed) == 3):
            print(f"    [ERROR] plug_mode=fixed requires plug_translate=(x,y,z) for {part_name}")
            return False, None
        center = Gf.Vec3d(float(fixed[0]), float(fixed[1]), float(fixed[2]))
        method = "fixed"
    elif plug_mode == "disc":
        center = compute_disc_circle_center(stage, dp)
        method = "disc-circle"
    else:
        center = compute_mesh_centroid(stage, dp)
        method = "bbox-centroid"

    if center is None:
        print(f"    [ERROR] Failed to compute {method} for {part_name}")
        return False, None

    plug_translate = (center[0], center[1], center[2])

    # Gather all expected names
    socket_defs = schema.get("sockets", [])
    all_names = {"plug_main"} | {s["name"] for s in socket_defs}

    if dry_run:
        print(f"      [DRY] plug_main ({method}) = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        for sdef in socket_defs:
            t = sdef["translate"]
            print(f"      [DRY] {sdef['name']}  translate=({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})")
        return True, center

    # Remove stale socket_*/plug_* Xforms
    for child in dp.GetChildren():
        cname = child.GetName()
        if (cname.startswith("socket_") or cname.startswith("plug_")) \
                and cname not in all_names:
            stage.RemovePrim(child.GetPath())

    # Author plug_main
    _apply_xform_common(stage, root_path, "plug_main", translate=plug_translate)
    print(f"      plug_main ({method}) = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")

    # Author fixed sockets (e.g., socket_gear on shafts)
    for sdef in socket_defs:
        _apply_xform_common(stage, root_path, sdef["name"], translate=sdef["translate"])
        t = sdef["translate"]
        print(f"      {sdef['name']} = ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})")

    stage.GetRootLayer().Save()
    return True, center


# ---------------------------------------------------------------------------
# Backward-compatible GEARBOX_SOCKET_CONFIG (used by other scripts)
# ---------------------------------------------------------------------------

def build_full_config() -> Dict[str, Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]]]:
    """Build the full GEARBOX_SOCKET_CONFIG from parent + child schemas."""
    cfg = dict(PARENT_SOCKET_CONFIG)
    for part_name, schema in CHILD_PART_SCHEMA.items():
        entry: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]] = {
            "sockets": {},
            "plugs": {
                "plug_main": {"translate": (0.0, 0.0, 0.0), "rotate_xyz": (0.0, 0.0, 0.0)},
            },
        }
        for sdef in schema.get("sockets", []):
            entry["sockets"][sdef["name"]] = {
                "translate": sdef["translate"],
                "rotate_xyz": (0.0, 0.0, 0.0),
            }
        cfg[part_name] = entry
    return cfg


GEARBOX_SOCKET_CONFIG = build_full_config()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="author_gearbox_sockets.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--parts-dir",
        default=os.path.join(_REPO_ROOT, "assets", "parts"),
    )
    args = parser.parse_args(argv)

    parts_dir = args.parts_dir
    print(f"\n{'='*65}")
    print(f"  Gearbox Socket Authoring (circle-center + hardcoded parents)")
    print(f"  Parts directory: {os.path.relpath(parts_dir, _REPO_ROOT)}")
    if args.dry_run:
        print(f"  MODE: DRY-RUN")
    print(f"{'='*65}\n")

    n_ok = n_fail = 0

    # --- Parent parts (hardcoded sockets) ---
    for part_name, part_config in sorted(PARENT_SOCKET_CONFIG.items()):
        usd_path = os.path.join(parts_dir, f"{part_name}.usd")
        if not os.path.isfile(usd_path):
            print(f"  [SKIP]  {part_name}.usd not found")
            continue
        ns = len(part_config.get("sockets", {}))
        np_ = len(part_config.get("plugs", {}))
        print(f"  [PARENT] {part_name}.usd  ({ns} sockets, {np_} plugs)")
        ok = author_parent_part(usd_path, part_config, dry_run=args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    # --- Child parts (circle center or BBox centroid) ---
    for part_name, schema in sorted(CHILD_PART_SCHEMA.items()):
        usd_path = os.path.join(parts_dir, f"{part_name}.usd")
        if not os.path.isfile(usd_path):
            print(f"  [SKIP]  {part_name}.usd not found")
            continue
        ns = len(schema.get("sockets", []))
        mode = schema.get("plug_mode", "bbox")
        print(f"  [CHILD]  {part_name}.usd  (plug_main[{mode}] + {ns} sockets)")
        ok, center = author_child_part(usd_path, part_name, schema,
                                        dry_run=args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{'='*65}")
    print(f"  Done: {n_ok} OK, {n_fail} failed")
    print(f"{'='*65}\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
