#!/usr/bin/env python3
"""Vectorized, read-only static fit check for M6 hub bolt 01 / output cover."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools._bootstrap import ensure_pxr_paths  # noqa: E402
ensure_pxr_paths()
from pxr import Gf, Usd, UsdGeom  # noqa: E402


def matrix_array(matrix):
    return np.asarray([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=float)


def transform_points(points, matrix):
    m = matrix_array(matrix)
    return np.asarray(points, dtype=float) @ m[:3, :3] + m[3, :3]


def read_meshes(path):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"cannot open {path}")
    root = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    if not root or not root.IsValid():
        raise RuntimeError(f"missing default root in {path}")
    cache = UsdGeom.XformCache()
    root_inv = cache.GetLocalToWorldTransform(root).GetInverse()
    triangles, points = [], []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        xf = cache.GetLocalToWorldTransform(prim) * root_inv
        local = transform_points(mesh.GetPointsAttr().Get() or [], xf)
        points.extend(local)
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        cursor = 0
        for count in mesh.GetFaceVertexCountsAttr().Get() or []:
            face = indices[cursor:cursor + int(count)]
            cursor += int(count)
            if len(face) >= 3:
                triangles.extend((local[[face[0], face[i], face[i + 1]]]
                                  for i in range(1, len(face) - 1)))
    if not triangles:
        raise RuntimeError(f"no triangles in {path}")
    return stage, root, cache, np.asarray(points), np.asarray(triangles)


def frame_local(stage, root, cache, name):
    prim = stage.GetPrimAtPath(f"{root.GetPath()}/{name}")
    if not prim or not prim.IsValid():
        raise RuntimeError(f"missing frame {name}")
    root_world = cache.GetLocalToWorldTransform(root)
    return cache.GetLocalToWorldTransform(prim) * root_world.GetInverse()


def fit_matrix(translation, rotation_xyz):
    rx, ry, rz = rotation_xyz
    rotation = (Gf.Rotation(Gf.Vec3d(1, 0, 0), rx)
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz))
    result = Gf.Matrix4d(1.0)
    result.SetRotate(rotation)
    result.SetTranslateOnly(Gf.Vec3d(*translation))
    return result


def clip_y(poly, bound, keep_above):
    if not poly:
        return []
    out = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        ina = a[1] >= bound if keep_above else a[1] <= bound
        inb = b[1] >= bound if keep_above else b[1] <= bound
        if ina != inb:
            t = (bound - a[1]) / (b[1] - a[1])
            out.append(a + t * (b - a))
        if inb:
            out.append(b)
    return out


def polygon_min_radius(poly):
    """Minimum XZ radius of a polygon; used only on vector-filtered candidates."""
    p = np.asarray(poly, dtype=float)[:, (0, 2)]
    if len(p) < 2:
        return float(np.linalg.norm(p[0])) if len(p) else math.inf
    edge = np.roll(p, -1, axis=0) - p
    cross = edge[:, 0] * (-p[:, 1]) - edge[:, 1] * (-p[:, 0])
    if np.all(cross >= -1e-9) or np.all(cross <= 1e-9):
        return 0.0
    a = p
    t = np.clip(-np.sum(a * edge, axis=1) / np.maximum(np.sum(edge * edge, axis=1), 1e-20), 0, 1)
    closest = a + t[:, None] * edge
    return float(np.min(np.linalg.norm(closest, axis=1)))


def triangle_min_radii(tris):
    """Vectorized projected triangle-to-axis distances, an exact broad-phase lower bound."""
    p = tris[:, :, (0, 2)]
    a = p
    edge = np.roll(p, -1, axis=1) - p
    t = np.clip(-np.sum(a * edge, axis=2) /
                np.maximum(np.sum(edge * edge, axis=2), 1e-20), 0, 1)
    closest = a + t[:, :, None] * edge
    d2 = np.sum(closest * closest, axis=2)
    edge = np.roll(p, -1, axis=1) - p
    cross = edge[:, :, 0] * (-p[:, :, 1]) - edge[:, :, 1] * (-p[:, :, 0])
    inside = np.all(cross >= -1e-9, axis=1) | np.all(cross <= 1e-9, axis=1)
    result = np.sqrt(np.min(d2, axis=1))
    result[inside] = 0.0
    return result


def shaft_measure(points):
    y = points[:, 1]
    r = np.hypot(points[:, 0], points[:, 2])
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    mid = r[(y >= lo + .15 * span) & (y <= lo + .62 * span)]
    radius = float(np.percentile(mid, 95))
    head = r[(y >= lo + .55 * span) & (r >= radius * 1.4)]
    if not len(head):
        raise RuntimeError("could not isolate cap radial envelope from bolt vertices")
    head_y = y[(y >= lo + .55 * span) & (r >= radius * 1.4)]
    head_bottom = float(head_y.min())
    return {"axis_y_bounds_source_units": [lo, hi], "shaft_y_interval_source_units": [lo, head_bottom],
            "shaft_radius_p95_source_units": radius, "shaft_diameter_p95_source_units": 2 * radius,
            "shaft_diameter_median_source_units": 2 * float(np.median(mid)),
            "cap_bottom_y_source_units_estimate": head_bottom,
            "cap_outer_radius_source_units": float(head.max())}


def clipped_surface_clearance(tris, y0, y1, radius, mm_per_unit):
    tri_y = tris[:, :, 1]
    y_overlap = (tri_y.min(axis=1) <= y1) & (tri_y.max(axis=1) >= y0)
    radial_lower = triangle_min_radii(tris)
    ordered = np.flatnonzero(y_overlap)
    ordered = ordered[np.argsort(radial_lower[ordered])]
    minimum, tri_index, axial_span = math.inf, None, None
    exact_count = 0
    for index in ordered:
        if radial_lower[index] > minimum:
            break
        poly = [v.copy() for v in tris[index]]
        poly = clip_y(poly, y0, True)
        poly = clip_y(poly, y1, False)
        if poly:
            exact_count += 1
            distance = polygon_min_radius(poly)
            if distance < minimum:
                minimum, tri_index = distance, int(index)
                axial_span = [float(min(v[1] for v in poly)), float(max(v[1] for v in poly))]
    if tri_index is None:
        return {"status": "NO_MESH_SURFACE_IN_SHAFT_INTERVAL", "shaft_radius_source_units": radius,
                "axial_overlap_triangles": int(y_overlap.sum())}
    return {"status": "SURFACE_WITHIN_SHAFT_ENVELOPE" if minimum <= radius else "CLEAR",
            "shaft_radius_source_units": radius, "min_surface_radius_source_units": minimum,
            "margin_mm": (minimum - radius) * mm_per_unit, "closest_triangle_index": tri_index,
            "closest_surface_axial_span_source_units": axial_span,
            "triangles_exactly_clipped_until_minimum_proven": exact_count,
            "axial_overlap_triangles": int(y_overlap.sum())}


def planar_support(cover_tris, y_plane, shaft_r, cap_r, hole_r, mm_per_unit):
    v1 = cover_tris[:, 1] - cover_tris[:, 0]
    v2 = cover_tris[:, 2] - cover_tris[:, 0]
    normal = np.cross(v1, v2)
    normal_len = np.linalg.norm(normal, axis=1)
    axial = np.abs(normal[:, 1]) / np.maximum(normal_len, 1e-12)
    yspan = np.ptp(cover_tris[:, :, 1], axis=1)
    mean_y = cover_tris[:, :, 1].mean(axis=1)
    selected = (axial > .98) & (yspan < .05) & (np.abs(mean_y - y_plane) < 1.0)
    faces = cover_tris[selected]
    if not len(faces):
        return {"status": "NO_COPLANAR_COVER_FACE_FOUND", "cap_bottom_y_source_units": y_plane}
    # Three radii across the potential bearing annulus, 48 azimuths each.
    radii = np.linspace(shaft_r + .12, max(shaft_r + .13, cap_r - .12), 3)
    angles = np.arange(48) * (2 * math.pi / 48)
    queries = np.asarray([(r * math.cos(a), r * math.sin(a)) for r in radii for a in angles])
    tri = faces[:, :, (0, 2)]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    good = np.abs(den) > 1e-12
    a, b, c, den = a[good], b[good], c[good], den[good]
    face_y = faces[good, :, 1].mean(axis=1)
    qx, qz = queries[:, 0:1], queries[:, 1:2]
    u = ((b[:, 1] - c[:, 1]) * (qx - c[:, 0]) + (c[:, 0] - b[:, 0]) * (qz - c[:, 1])) / den
    v = ((c[:, 1] - a[:, 1]) * (qx - c[:, 0]) + (a[:, 0] - c[:, 0]) * (qz - c[:, 1])) / den
    covered = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
    supported = covered.any(axis=1)
    surface_y = np.full(len(queries), np.nan)
    for i in np.flatnonzero(supported):
        surface_y[i] = float(face_y[np.flatnonzero(covered[i])].max())
    per_radius = []
    ring_gaps = []
    bearing_valid = []
    for index, radius in enumerate(radii):
        start, stop = index * len(angles), (index + 1) * len(angles)
        ring_supported = supported[start:stop]
        is_bearing = radius > hole_r + .05
        if is_bearing:
            bearing_valid.extend(bool(v) for v in ring_supported)
        per_radius.append({"radius_source_units": float(radius),
                           "support_fraction": float(ring_supported.mean()),
                           "inside_aperture": bool(not is_bearing),
                           "samples": int(len(ring_supported))})
        ring_gaps.append(None if not ring_supported.any() else [
            float(np.nanmin(gaps_all := (y_plane - surface_y[start:stop][ring_supported])) * mm_per_unit),
            float(np.nanmax(gaps_all) * mm_per_unit)])
    bearing_fraction = float(np.mean(bearing_valid)) if bearing_valid else 0.0
    bearing_gaps = [y_plane - surface_y[i] for i, ok in enumerate(supported)
                    if ok and radii[i // len(angles)] > hole_r + .05]
    return {"status": "SAMPLED_ANNULAR_BEARING" if bearing_fraction >= .9 else "PARTIAL_OR_NO_BEARING",
            "cap_bottom_y_source_units_estimate": y_plane,
            "sampled_support_fraction_excluding_aperture": bearing_fraction,
            "all_probe_fraction_including_aperture_diagnostic": float(supported.mean()),
            "samples": int(len(queries)),
            "support_by_radius": per_radius, "gap_mm_by_radius": ring_gaps,
            "coplanar_cover_triangles_considered": int(len(faces)),
            "covered_bearing_gap_mm_min_max": None if not bearing_gaps else
                [float(min(bearing_gaps) * mm_per_unit), float(max(bearing_gaps) * mm_per_unit)],
            "note": "48 azimuth samples per radius; rings inside measured aperture are diagnostic, not bearing failures"}


def ray_z_crossings(tris, x, y):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    good = np.abs(den) > 1e-10
    a, b, c, den = a[good], b[good], c[good], den[good]
    u = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / den
    v = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / den
    w = 1.0 - u - v
    inside = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
    z = u * a[:, 2] + v * b[:, 2] + w * c[:, 2]
    ordered = np.sort(z[inside])
    return [float(v) for i, v in enumerate(ordered)
            if i == 0 or v - ordered[i - 1] > 1e-4]


def main():
    partdir = ROOT / "assets/parts"
    case_stage, case_root, case_cache, case_points, case_tris = read_meshes(partdir / "Casing Top.usd")
    bolt_stage, bolt_root, bolt_cache, bolt_points, bolt_tris = read_meshes(partdir / "M6 Hub Bolt.usd")
    _, _, _, _, cover_tris = read_meshes(partdir / "Hub Cover Output.usd")
    socket = frame_local(case_stage, case_root, case_cache, "socket_bolt_hub_1")
    plug = frame_local(bolt_stage, bolt_root, bolt_cache, "plug_main")
    fit = fit_matrix((0, 0, 0), (90, 0, 0))
    bolt_pose = plug.GetInverse() * fit * socket
    old = bolt_pose.GetRow3(3)
    bolt_pose.SetTranslateOnly(Gf.Vec3d(float(old[0]), float(old[1]), 29.2))
    cover_pose = fit_matrix((0.0, 43.41859, 31.0), (-90, 180, 0))

    scene = Usd.Stage.Open(str(ROOT / "assets/simple_room_scene.usd"))
    scene_mpu = float(UsdGeom.GetStageMetersPerUnit(scene))
    scene_cache = UsdGeom.XformCache()
    bolt_instance = next((p for p in scene.Traverse() if p.GetName() == "M6_Hub_Bolt_01_top"), None)
    case_instance = next((p for p in scene.Traverse() if p.GetName() == "Casing_Top"), None)
    if bolt_instance is None or case_instance is None:
        raise RuntimeError("simple_room_scene lacks expected bolt/casing instance")
    bolt_world = scene_cache.GetLocalToWorldTransform(bolt_instance)
    case_world = scene_cache.GetLocalToWorldTransform(case_instance)
    bolt_basis = [bolt_world.TransformDir(Gf.Vec3d(*(1 if i == j else 0 for i in range(3)))).GetLength()
                  for j in range(3)]
    case_basis = [case_world.TransformDir(Gf.Vec3d(*(1 if i == j else 0 for i in range(3)))).GetLength()
                  for j in range(3)]
    if max(abs(a - b) for a, b in zip(bolt_basis, case_basis)) > 1e-6:
        raise RuntimeError(f"bolt and casing have different scene scales: {bolt_basis} vs {case_basis}")
    scene_scale = float(sum(bolt_basis) / 3)
    if abs(scene_scale - .002) > 1e-5:
        raise RuntimeError(f"scene scale changed from reported inventory: {scene_scale}")

    shaft = shaft_measure(bolt_points)
    y0, y1 = shaft["shaft_y_interval_source_units"]
    radius = shaft["shaft_radius_p95_source_units"]
    mm_per_source_unit = scene_scale * scene_mpu * 1000
    socket_t = socket.GetRow3(3)
    cover_axis_local = cover_pose.GetInverse().Transform(Gf.Vec3d(float(socket_t[0]), float(socket_t[1]), 31.0))

    def posed_meshes(pose):
        inv = pose.GetInverse()
        case_local = transform_points(case_tris.reshape(-1, 3), inv).reshape(-1, 3, 3)
        cover_xf = cover_pose * inv
        cover_local = transform_points(cover_tris.reshape(-1, 3), cover_xf).reshape(-1, 3, 3)
        return case_local, cover_local

    def fit_at_pose(pose):
        case_local, cover_local = posed_meshes(pose)
        case_fit = clipped_surface_clearance(case_local, y0, y1, radius, mm_per_source_unit)
        cover_fit = clipped_surface_clearance(cover_local, y0, y1, radius, mm_per_source_unit)
        seat = planar_support(cover_local, shaft["cap_bottom_y_source_units_estimate"], radius,
                              shaft["cap_outer_radius_source_units"],
                              cover_fit.get("min_surface_radius_source_units", math.inf), mm_per_source_unit)
        return case_fit, cover_fit, seat

    case_old, cover_old, seat_old = fit_at_pose(bolt_pose)
    old_t = bolt_pose.GetRow3(3)
    contact_gap_mm = (seat_old.get("covered_bearing_gap_mm_min_max") or [0.0])[0]
    shift_source = contact_gap_mm / mm_per_source_unit
    axis = bolt_pose.TransformDir(Gf.Vec3d(0, 1, 0))
    axis.Normalize()
    corrected_t = Gf.Vec3d(*old_t) - axis * shift_source
    contact_pose = Gf.Matrix4d(bolt_pose)
    contact_pose.SetTranslateOnly(corrected_t)
    case_clear, cover_clear, support = fit_at_pose(contact_pose)

    tip = contact_pose.Transform(Gf.Vec3d(0, y0, 0))
    case_top_z = float(case_points[:, 2].max())
    centerline_hits_z = ray_z_crossings(case_tris, float(socket_t[0]), float(socket_t[1]))
    interior_floor_z = max((z for z in centerline_hits_z if z < case_top_z - 1.0), default=None)

    result = {
        "scope": "static source meshes posed in authored casing frame; no threading, simulation, or robot motion",
        "assembly_source": "runtime/ui.py::combine_casing_top + runtime/magic_assembly.py child overrides",
        "task_scale": {"simple_room_meters_per_unit": scene_mpu, "bolt_instance_basis": bolt_basis,
                       "casing_instance_basis": case_basis, "source_unit_to_scene_m": scene_scale * scene_mpu,
                       "bolt_scene_shank_diameter_mm_from_source_profile":
                           shaft["shaft_diameter_median_source_units"] * scene_scale * scene_mpu * 1000},
        "candidate": {"bolt": "M6_Hub_Bolt_01_top", "cover": "Hub_Cover_Output", "socket": "socket_bolt_hub_1",
                      "socket_source_xyz": [float(socket_t[i]) for i in range(3)],
                      "output_cover_local_bolt_axis_at_cover_midplane": [float(cover_axis_local[i]) for i in range(3)],
                      "cover_pose_source": {"translation": [0.0, 43.41859, 31.0], "rotation_xyz_deg": [-90, 180, 0]},
                      "authored_bolt_root_translation_source": [float(old_t[i]) for i in range(3)],
                      "measured_gap_at_authored_pose_mm": contact_gap_mm,
                      "calculated_contact_bolt_root_translation_source": [float(corrected_t[i]) for i in range(3)],
                      "contact_translation_delta_mm": [float(corrected_t[i] - old_t[i]) * mm_per_source_unit
                                                        for i in range(3)]},
        "bolt_profile_source_units": shaft,
        "output_cover_aperture": {"minimum_radial_mesh_distance_source_units":
                                      cover_clear.get("min_surface_radius_source_units"),
                                  "corresponding_diameter_mm":
                                      2 * cover_clear.get("min_surface_radius_source_units", 0) * mm_per_source_unit,
                                  "cap_to_aperture_radial_overlap_mm":
                                      (shaft["cap_outer_radius_source_units"] -
                                       cover_clear.get("min_surface_radius_source_units", 0)) * mm_per_source_unit},
        "whole_shank_surface_clearance_at_contact_pose": {"cover": cover_clear, "casing": case_clear},
        "cap_bearing_check": support,
        "casing_axial_entry": {"outermost_casing_mesh_z_source": case_top_z,
                                "bolt_tip_z_at_contact_pose_source": float(tip[2]),
                                "shaft_tip_below_outer_surface_mm": (case_top_z - float(tip[2])) * mm_per_source_unit,
                                "centerline_mesh_crossings_z_source": centerline_hits_z,
                                "nearest_interior_floor_crossing_z_source": interior_floor_z,
                                "tip_to_floor_crossing_clearance_mm": None if interior_floor_z is None else
                                    (float(tip[2]) - interior_floor_z) * mm_per_source_unit},
        "limits": ["mesh-surface/cylinder test is conservative over the measured shaft interval",
                   "calculated seat closes the measured static cap-to-cover gap; contact is mesh-based, not dynamic",
                   "does not establish grasp accessibility, contact dynamics, or full task success"],
    }
    print(json.dumps(result, indent=2))


def self_test():
    assert abs(polygon_min_radius(np.asarray([[4., -1., -1.], [4., 1., 1.], [5., 0., 0.]])) - 4) < 1e-8
    slab = clip_y([np.array([0., -2., 0.]), np.array([1., 2., 0.]), np.array([0., 2., 1.])], -1, True)
    slab = clip_y(slab, 1, False)
    assert len(slab) >= 3 and all(-1 - 1e-9 <= p[1] <= 1 + 1e-9 for p in slab)
    tri = np.asarray([[[4., -1., 0.], [4., 1., 0.], [4., 0., 1.]],
                      [[0., 0., 0.], [1., 0., 0.], [0., 1., 1.]]])
    assert np.allclose(triangle_min_radii(tri), [4, 0])
    print("geometry feasibility self-test: PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
