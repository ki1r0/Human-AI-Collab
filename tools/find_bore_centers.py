#!/usr/bin/env python3
"""Find exact bore centers by iterative circle fitting.

Strategy: Start from approximate bore center, compute distances from that center
for face vertices, find the densest ring, fit a circle to refine the center,
and iterate until convergence.
"""
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()
from pxr import Gf, Usd, UsdGeom


def get_root_local_points(stage, root_prim):
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


def lstsq_circle_fit(coords_2d):
    n = len(coords_2d)
    if n < 3:
        return 0.0, 0.0, 0.0
    ata = [[0.0]*3 for _ in range(3)]
    atb = [0.0]*3
    for x, y in coords_2d:
        row = [2.0*x, 2.0*y, 1.0]
        rhs = x*x + y*y
        for i in range(3):
            for j in range(3):
                ata[i][j] += row[i] * row[j]
            atb[i] += row[i] * rhs
    M = [ata[i][:] + [atb[i]] for i in range(3)]
    for col in range(3):
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


def find_bore_center_iterative(face_pts_xy, approx_cx, approx_cy, label,
                                 expected_r_min=10.0, expected_r_max=60.0,
                                 n_iters=5):
    """Iteratively find bore center.

    1. Compute distances from current center estimate
    2. Find densest 1mm ring within expected radius range
    3. Fit circle to that ring to get refined center
    4. Repeat
    """
    cx, cy = approx_cx, approx_cy

    print(f"\n  {label}")
    print(f"  Starting from ({cx:.2f}, {cy:.2f})")

    for it in range(n_iters):
        dists = [math.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in face_pts_xy]

        # Find densest ring in expected radius range
        bin_width = 0.5
        best_bin = None
        best_count = 0

        for r_lo_half in range(int(expected_r_min * 2), int(expected_r_max * 2)):
            r_lo = r_lo_half * 0.5
            r_hi = r_lo + bin_width
            count = sum(1 for d in dists if r_lo <= d < r_hi)
            if count > best_count:
                best_count = count
                best_bin = (r_lo, r_hi)

        if best_bin is None or best_count < 10:
            print(f"    Iter {it}: no dense ring found")
            break

        # Expand to adjacent bins to get more points
        r_lo, r_hi = best_bin
        # Extend by ±1mm for more robust fit
        ring_pts = [(face_pts_xy[i][0], face_pts_xy[i][1])
                    for i in range(len(face_pts_xy))
                    if (r_lo - 1.0) <= dists[i] < (r_hi + 1.0)]

        new_cx, new_cy, new_r = lstsq_circle_fit(ring_pts)

        delta = math.sqrt((new_cx - cx)**2 + (new_cy - cy)**2)
        print(f"    Iter {it}: ring r=[{r_lo:.1f}-{r_hi:.1f}], {best_count} pts (expanded: {len(ring_pts)}), "
              f"center=({new_cx:.4f}, {new_cy:.4f}), radius={new_r:.3f}, delta={delta:.4f}")

        cx, cy = new_cx, new_cy

        if delta < 0.001:
            print(f"    Converged!")
            break

    # Final: show ALL rings at the converged center
    print(f"\n    Final ring analysis from ({cx:.4f}, {cy:.4f}):")
    dists = [math.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in face_pts_xy]
    max_dist = max(dists) if dists else 0

    print(f"    {'R_range':>12s} {'Count':>6s} {'CX':>10s} {'CY':>10s} {'Radius':>8s}")
    for r_lo_int in range(0, int(max_dist) + 1):
        r_lo = float(r_lo_int)
        r_hi = r_lo + 1.0
        ring = [(face_pts_xy[i][0], face_pts_xy[i][1])
                for i in range(len(face_pts_xy))
                if r_lo <= dists[i] < r_hi]
        if len(ring) >= 50:
            rcx, rcy, rr = lstsq_circle_fit(ring)
            marker = " <-- BORE EDGE" if len(ring) >= 200 else ""
            print(f"    {r_lo:5.0f}-{r_hi:5.0f}mm  {len(ring):5d}  {rcx:10.4f}  {rcy:10.4f}  {rr:8.3f}{marker}")

    return cx, cy


def main():
    parts_dir = os.path.join(_REPO, "assets", "parts")

    for part_name in ["Casing Top", "Casing Base"]:
        print(f"\n{'='*70}")
        print(f"  {part_name}")
        print(f"{'='*70}")

        usd_path = os.path.join(parts_dir, f"{part_name}.usd")
        stage = Usd.Stage.Open(usd_path)
        dp = stage.GetDefaultPrim()
        pts = get_root_local_points(stage, dp)

        mins = [min(p[i] for p in pts) for i in range(3)]
        maxs = [max(p[i] for p in pts) for i in range(3)]
        print(f"  BBox: Z=[{mins[2]:.2f}, {maxs[2]:.2f}]")

        # Z+ face (parting face / exterior for Top, parting face for Base)
        z_plus = maxs[2]
        face_zp = [(p[0], p[1]) for p in pts if abs(p[2] - z_plus) < 0.5]

        # Z- face
        z_minus = mins[2]
        face_zm = [(p[0], p[1]) for p in pts if abs(p[2] - z_minus) < 0.5]

        print(f"\n  Z+ face ({z_plus:.2f}): {len(face_zp)} vertices")

        # Find bore centers on Z+ face
        out_cx, out_cy = find_bore_center_iterative(
            face_zp, 0.0, 17.0, "OUTPUT BORE on Z+",
            expected_r_min=20.0, expected_r_max=50.0)

        inp_cx, inp_cy = find_bore_center_iterative(
            face_zp, 31.0, -49.69, "INPUT BORE on Z+",
            expected_r_min=15.0, expected_r_max=35.0)

        xfr_cx, xfr_cy = find_bore_center_iterative(
            face_zp, -31.0, -49.69, "TRANSFER BORE on Z+",
            expected_r_min=15.0, expected_r_max=35.0)

        print(f"\n  Z- face ({z_minus:.2f}): {len(face_zm)} vertices")

        if len(face_zm) > 100:
            find_bore_center_iterative(
                face_zm, 0.0, 17.0, "OUTPUT BORE on Z-",
                expected_r_min=20.0, expected_r_max=50.0)

            find_bore_center_iterative(
                face_zm, 31.0, -49.69, "INPUT BORE on Z-",
                expected_r_min=15.0, expected_r_max=35.0)

            find_bore_center_iterative(
                face_zm, -31.0, -49.69, "TRANSFER BORE on Z-",
                expected_r_min=15.0, expected_r_max=35.0)

        print(f"\n  SUMMARY for {part_name}:")
        print(f"    Output bore center (Z+): ({out_cx:.4f}, {out_cy:.4f})")
        print(f"    Input bore center  (Z+): ({inp_cx:.4f}, {inp_cy:.4f})")
        print(f"    Transfer bore center(Z+):({xfr_cx:.4f}, {xfr_cy:.4f})")


if __name__ == "__main__":
    main()
