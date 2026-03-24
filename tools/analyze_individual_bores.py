#!/usr/bin/env python3
"""Targeted bore analysis: isolate each bore opening on the casing Z+/Z- faces
and fit circles to find exact centers."""
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


def analyze_bore_region(face_pts_xy, approx_cx, approx_cy, search_radius, label):
    """Extract vertices near (approx_cx, approx_cy) on the face and fit circles."""
    # Get all face vertices within search_radius of the approximate center
    region_pts = [(x, y) for x, y in face_pts_xy
                  if math.sqrt((x - approx_cx)**2 + (y - approx_cy)**2) < search_radius]

    print(f"\n  {label}")
    print(f"    Search: center=({approx_cx:.1f}, {approx_cy:.1f}), radius={search_radius}mm")
    print(f"    Vertices in region: {len(region_pts)}")

    if len(region_pts) < 10:
        print("    Too few vertices")
        return None, None, None

    # Compute local centroid
    cx_local = sum(p[0] for p in region_pts) / len(region_pts)
    cy_local = sum(p[1] for p in region_pts) / len(region_pts)

    # Compute distances from local centroid
    dists = [math.sqrt((p[0] - cx_local)**2 + (p[1] - cy_local)**2) for p in region_pts]

    # Bin into 0.5mm rings
    max_dist = max(dists)
    bin_width = 0.5
    n_bins = int(max_dist / bin_width) + 1

    print(f"    Local centroid: ({cx_local:.3f}, {cy_local:.3f})")
    print(f"    Ring analysis:")
    print(f"    {'Bin':>8s} {'Count':>6s} {'CX':>10s} {'CY':>10s} {'Radius':>8s}")

    best_ring = None
    best_count = 0

    for b_idx in range(n_bins):
        r_lo = b_idx * bin_width
        r_hi = r_lo + bin_width
        ring = [(region_pts[i][0], region_pts[i][1])
                for i in range(len(region_pts))
                if r_lo <= dists[i] < r_hi]
        if len(ring) >= 15:
            cx, cy, radius = lstsq_circle_fit(ring)
            marker = ""
            if len(ring) >= 100:
                marker = " <--- DENSE"
            if len(ring) > best_count:
                best_count = len(ring)
                best_ring = (cx, cy, radius, len(ring))
            print(f"    {r_lo:6.1f}-{r_hi:5.1f}  {len(ring):5d}  {cx:10.3f}  {cy:10.3f}  {radius:8.3f}{marker}")

    # Also do a multi-ring fit: collect all vertices in the densest contiguous range
    # Find rings with bore-edge characteristics (relatively narrow band of dense vertices)
    dense_bins = []
    for b_idx in range(n_bins):
        r_lo = b_idx * bin_width
        r_hi = r_lo + bin_width
        ring_count = sum(1 for i in range(len(region_pts))
                        if r_lo <= dists[i] < r_hi)
        if ring_count >= 30:
            dense_bins.append((b_idx, ring_count))

    if dense_bins:
        # Group consecutive dense bins
        clusters = []
        current = [dense_bins[0]]
        for i in range(1, len(dense_bins)):
            if dense_bins[i][0] - dense_bins[i-1][0] <= 2:
                current.append(dense_bins[i])
            else:
                clusters.append(current)
                current = [dense_bins[i]]
        clusters.append(current)

        print(f"\n    Bore edge clusters:")
        for ci, cluster in enumerate(clusters):
            r_lo = cluster[0][0] * bin_width
            r_hi = (cluster[-1][0] + 1) * bin_width
            total = sum(c[1] for c in cluster)
            cluster_pts = [(region_pts[i][0], region_pts[i][1])
                          for i in range(len(region_pts))
                          if r_lo <= dists[i] < r_hi]
            cx, cy, radius = lstsq_circle_fit(cluster_pts)
            print(f"      Cluster {ci}: r=[{r_lo:.1f}-{r_hi:.1f}mm], {total} pts, "
                  f"center=({cx:.4f}, {cy:.4f}), radius={radius:.3f}mm")

    # Overall fit of all region points
    cx_all, cy_all, r_all = lstsq_circle_fit(region_pts)
    print(f"\n    All-points fit: center=({cx_all:.4f}, {cy_all:.4f}), radius={r_all:.3f}mm")

    return best_ring


def analyze_casing_face(part_name, face_z, face_tol=0.5):
    """Analyze all three bores on a specific face of the casing."""
    usd_path = os.path.join(_REPO, "assets", "parts", f"{part_name}.usd")
    stage = Usd.Stage.Open(usd_path)
    dp = stage.GetDefaultPrim()
    pts = get_root_local_points(stage, dp)

    # Get face vertices (XY coords)
    face_pts = [(p[0], p[1]) for p in pts if abs(p[2] - face_z) < face_tol]
    print(f"\n{'='*70}")
    print(f"  {part_name} face Z={face_z:.2f}  ({len(face_pts)} vertices)")
    print(f"{'='*70}")

    # Analyze each bore region with generous search radius
    # Output bore: approximate center near (0, 17)
    analyze_bore_region(face_pts, 0.0, 17.0, 55.0, "OUTPUT BORE (large)")

    # Input bore: approximate center near (31, -50)
    analyze_bore_region(face_pts, 31.0, -50.0, 40.0, "INPUT BORE")

    # Transfer bore: approximate center near (-31, -50)
    analyze_bore_region(face_pts, -31.0, -50.0, 40.0, "TRANSFER BORE")

    # Also check bolt hole positions
    # M6 bolt holes are at known approximate positions
    bolt_positions = [
        (40.0, 0.0, "BOLT_HOLE_1"),
        (-40.0, 0.0, "BOLT_HOLE_2"),
        (40.0, 80.0, "BOLT_HOLE_3"),
        (-40.0, 80.0, "BOLT_HOLE_4"),
        (54.0, -27.0, "BOLT_HOLE_5"),
        (-54.0, -27.0, "BOLT_HOLE_6"),
    ]
    for bx, by, blabel in bolt_positions:
        region_pts = [(x, y) for x, y in face_pts
                      if math.sqrt((x - bx)**2 + (y - by)**2) < 10.0]
        if len(region_pts) >= 10:
            cx, cy, r = lstsq_circle_fit(region_pts)
            print(f"\n  {blabel}: {len(region_pts)} pts near ({bx}, {by}), "
                  f"fit center=({cx:.3f}, {cy:.3f}), r={r:.3f}")


if __name__ == "__main__":
    # Casing Top - Z+ face (exterior, where hub covers mount)
    analyze_casing_face("Casing Top", 27.95)
    # Casing Top - Z- face (parting face / interior)
    analyze_casing_face("Casing Top", -27.95)

    # Casing Base - Z+ face (parting face, mates with Casing Top)
    analyze_casing_face("Casing Base", 27.95)
    # Casing Base - Z- face (exterior, where hub covers mount)
    analyze_casing_face("Casing Base", -27.95)
