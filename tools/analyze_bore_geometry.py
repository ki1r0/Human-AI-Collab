#!/usr/bin/env python3
"""Analyze casing bore geometry to find exact circle centers on each face.

For each casing, identifies circular bore openings on the Z+ and Z- faces,
fits circles to them, and reports the centers and radii.
Also analyzes hub cover mounting faces for plug position verification.
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
    """Get all mesh vertices in root-prim-local coordinates."""
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
    """Least-squares circle fit. Returns (cx, cy, radius)."""
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


def analyze_face_bores(pts, face_axis, face_value, face_tol, label):
    """Analyze circular features on a specific face of the casing.

    face_axis: 0=X, 1=Y, 2=Z
    face_value: the coordinate value on that axis for the face
    face_tol: tolerance for selecting face vertices
    """
    plane_axes = [i for i in range(3) if i != face_axis]
    pa, pb = plane_axes

    # Get face vertices
    face_pts_2d = [(p[pa], p[pb]) for p in pts
                    if abs(p[face_axis] - face_value) < face_tol]

    print(f"\n  --- {label} (axis={face_axis}, val={face_value:.2f}, tol={face_tol}) ---")
    print(f"  Face vertices: {len(face_pts_2d)}")

    if len(face_pts_2d) < 10:
        print("  Too few vertices on face")
        return

    # Compute face BBox
    as_ = [fp[0] for fp in face_pts_2d]
    bs_ = [fp[1] for fp in face_pts_2d]
    center_a = (min(as_) + max(as_)) / 2.0
    center_b = (min(bs_) + max(bs_)) / 2.0
    print(f"  Face BBox center: ({['X','Y','Z'][pa]}={center_a:.3f}, {['X','Y','Z'][pb]}={center_b:.3f})")

    # Compute distances from face center
    dists = [math.sqrt((fp[0] - center_a)**2 + (fp[1] - center_b)**2)
             for fp in face_pts_2d]

    # Bin into 1mm rings and find dense rings (potential bore edges)
    max_dist = max(dists)
    bin_width = 1.0
    n_bins = int(max_dist / bin_width) + 1

    print(f"\n  Ring analysis (distance from face center):")
    print(f"  {'Bin':>6s} {'Count':>6s} {'Circle CX':>10s} {'Circle CY':>10s} {'Radius':>8s}  Axes=({['X','Y','Z'][pa]},{['X','Y','Z'][pb]})")

    dense_rings = []
    for b_idx in range(n_bins):
        r_lo = b_idx * bin_width
        r_hi = r_lo + bin_width
        ring = [(face_pts_2d[i][0], face_pts_2d[i][1])
                for i in range(len(face_pts_2d))
                if r_lo <= dists[i] < r_hi]
        if len(ring) >= 20:
            cx, cy, radius = lstsq_circle_fit(ring)
            dense_rings.append((b_idx, len(ring), cx, cy, radius))
            marker = " <--- DENSE" if len(ring) >= 100 else ""
            print(f"  {r_lo:5.0f}-{r_hi:4.0f}  {len(ring):5d}  {cx:10.3f}  {cy:10.3f}  {radius:8.3f}{marker}")

    # Find clusters of dense rings (bore edges)
    if dense_rings:
        print(f"\n  Identified bore-edge ring clusters:")
        clusters = []
        current_cluster = [dense_rings[0]]
        for i in range(1, len(dense_rings)):
            if dense_rings[i][0] - dense_rings[i-1][0] <= 2:  # consecutive or gap of 1
                current_cluster.append(dense_rings[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [dense_rings[i]]
        clusters.append(current_cluster)

        for ci, cluster in enumerate(clusters):
            total_pts = sum(r[1] for r in cluster)
            # Collect all points in this cluster and fit a single circle
            r_lo_all = cluster[0][0] * bin_width
            r_hi_all = (cluster[-1][0] + 1) * bin_width
            cluster_pts = [(face_pts_2d[i][0], face_pts_2d[i][1])
                          for i in range(len(face_pts_2d))
                          if r_lo_all <= dists[i] < r_hi_all]
            cx, cy, radius = lstsq_circle_fit(cluster_pts)
            axis_labels = ['X','Y','Z']
            print(f"    Cluster {ci}: r=[{r_lo_all:.0f}-{r_hi_all:.0f}mm], "
                  f"{total_pts} pts, "
                  f"center=({axis_labels[pa]}={cx:.3f}, {axis_labels[pb]}={cy:.3f}), "
                  f"radius={radius:.3f}mm")


def analyze_casing(part_name):
    """Full analysis of a casing part."""
    print(f"\n{'='*70}")
    print(f"  ANALYZING: {part_name}")
    print(f"{'='*70}")

    usd_path = os.path.join(_REPO, "assets", "parts", f"{part_name}.usd")
    stage = Usd.Stage.Open(usd_path)
    dp = stage.GetDefaultPrim()
    pts = get_root_local_points(stage, dp)

    # BBox
    mins = [min(p[i] for p in pts) for i in range(3)]
    maxs = [max(p[i] for p in pts) for i in range(3)]
    print(f"  BBox: X=[{mins[0]:.2f}, {maxs[0]:.2f}], Y=[{mins[1]:.2f}, {maxs[1]:.2f}], Z=[{mins[2]:.2f}, {maxs[2]:.2f}]")
    print(f"  Total vertices: {len(pts)}")

    # Analyze Z+ face (parting face / top exterior)
    analyze_face_bores(pts, face_axis=2, face_value=maxs[2], face_tol=0.5,
                       label=f"{part_name} Z+ face (z={maxs[2]:.2f})")

    # Analyze Z- face (bottom exterior)
    analyze_face_bores(pts, face_axis=2, face_value=mins[2], face_tol=0.5,
                       label=f"{part_name} Z- face (z={mins[2]:.2f})")


def analyze_hub_cover(part_name):
    """Analyze hub cover mounting face geometry."""
    print(f"\n{'='*70}")
    print(f"  ANALYZING: {part_name}")
    print(f"{'='*70}")

    usd_path = os.path.join(_REPO, "assets", "parts", f"{part_name}.usd")
    stage = Usd.Stage.Open(usd_path)
    dp = stage.GetDefaultPrim()
    pts = get_root_local_points(stage, dp)

    mins = [min(p[i] for p in pts) for i in range(3)]
    maxs = [max(p[i] for p in pts) for i in range(3)]
    extents = [maxs[i] - mins[i] for i in range(3)]
    thin_axis = extents.index(min(extents))

    print(f"  BBox: X=[{mins[0]:.2f}, {maxs[0]:.2f}], Y=[{mins[1]:.2f}, {maxs[1]:.2f}], Z=[{mins[2]:.2f}, {maxs[2]:.2f}]")
    print(f"  Extents: X={extents[0]:.2f}, Y={extents[1]:.2f}, Z={extents[2]:.2f}")
    print(f"  Thin axis: {['X','Y','Z'][thin_axis]} (disc normal)")

    # Analyze both faces of the thin axis
    analyze_face_bores(pts, face_axis=thin_axis, face_value=maxs[thin_axis], face_tol=0.5,
                       label=f"{part_name} + face ({['X','Y','Z'][thin_axis]}={maxs[thin_axis]:.2f})")
    analyze_face_bores(pts, face_axis=thin_axis, face_value=mins[thin_axis], face_tol=0.5,
                       label=f"{part_name} - face ({['X','Y','Z'][thin_axis]}={mins[thin_axis]:.2f})")


if __name__ == "__main__":
    analyze_casing("Casing Top")
    analyze_casing("Casing Base")
    analyze_hub_cover("Hub Cover Output")
    analyze_hub_cover("Hub Cover Input")
    analyze_hub_cover("Hub Cover Small")
