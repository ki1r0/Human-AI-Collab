#!/usr/bin/env python3
"""Offline PXR audit of local Panda hand/finger collision geometry and bolt cap."""
from __future__ import annotations

import json
import math
import ast
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools._bootstrap import ensure_pxr_paths  # noqa: E402
ensure_pxr_paths()
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402


PANDA = ROOT / "assets/vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
PANDA_PROPS = PANDA.parent / "Props"
PROP_NAMES = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
BOLT = ROOT / "assets/parts/M6 Hub Bolt.usd"
CVR = ROOT / "assets/parts/Hub Cover Output.usd"
ENV_CFG = ROOT / "runtime/bolt_harness/env_cfg.py"
SCENE_SCALE = 0.002
TCP_HAND_Z_M = 0.1034
CAP_BOTTOM_Y_SOURCE = 8.9250011444


def all_prims(stage):
    return Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())


def matrix_np(matrix):
    return np.asarray([[float(matrix[i][j]) for j in range(4)] for i in range(4)])


def transform(points, matrix):
    m = matrix_np(matrix)
    p = np.asarray(points, dtype=float)
    return p @ m[:3, :3] + m[3, :3]


def bounds(points):
    p = np.asarray(points, dtype=float)
    return [[float(p[:, i].min()), float(p[:, i].max())] for i in range(3)]


def env_cfg_value(name):
    module = ast.parse(ENV_CFG.read_text())
    config = next(n for n in module.body if isinstance(n, ast.ClassDef)
                  and n.name == "BoltInsertionEnvCfg")
    for node in config.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {ENV_CFG}")


def quat_rotate(vector, wxyz):
    v = np.asarray(vector, dtype=float)
    q = np.asarray(wxyz, dtype=float)
    w, u = q[0], q[1:]
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def prop_collision_points(robot_stage, hand):
    robot_cache = UsdGeom.XformCache()
    hand_inv = robot_cache.GetLocalToWorldTransform(hand).GetInverse()
    groups, records = {}, []
    for name in PROP_NAMES:
        prop_stage = Usd.Stage.Open(str(PANDA_PROPS / f"{name}.usd"))
        if prop_stage is None:
            raise RuntimeError(f"cannot open local Panda collision prop {name}")
        root = prop_stage.GetDefaultPrim() or prop_stage.GetPseudoRoot()
        prop_cache = UsdGeom.XformCache()
        root_inv = prop_cache.GetLocalToWorldTransform(root).GetInverse()
        body = body_prim(robot_stage, name)
        body_to_hand = robot_cache.GetLocalToWorldTransform(body) * hand_inv
        pts_for_body = []
        for prim in all_prims(prop_stage):
            if not prim.IsA(UsdGeom.Mesh) or not has_collision_api(prim, root):
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get() or []
            mesh_to_root = prop_cache.GetLocalToWorldTransform(prim) * root_inv
            local = transform(pts, mesh_to_root)
            hand_local = transform(local, body_to_hand)
            if len(hand_local):
                pts_for_body.append(hand_local)
                records.append({"body": name, "asset": str((PANDA_PROPS / f"{name}.usd").relative_to(ROOT)),
                                "mesh_path": str(prim.GetPath()), "vertex_count": int(len(hand_local)),
                                "collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
                                "mesh_collision_api": bool(prim.HasAPI(UsdPhysics.MeshCollisionAPI)),
                                "bounds_hand_local_m": bounds(hand_local)})
        if not pts_for_body:
            raise RuntimeError(f"no collision meshes in local Panda prop {name}")
        groups[name] = np.vstack(pts_for_body)
    return groups, records


def asset_mesh_points(path):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"cannot open {path}")
    root = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World") or stage.GetPseudoRoot()
    cache = UsdGeom.XformCache()
    root_inv = cache.GetLocalToWorldTransform(root).GetInverse()
    groups = []
    for prim in all_prims(stage):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        xf = cache.GetLocalToWorldTransform(prim) * root_inv
        pts = transform(mesh.GetPointsAttr().Get() or [], xf)
        if len(pts):
            groups.append(pts)
    if not groups:
        raise RuntimeError(f"no mesh points in {path}")
    return stage, root, np.vstack(groups)


def body_prim(stage, name):
    matches = [p for p in all_prims(stage) if p.GetName() == name]
    rigid = [p for p in matches if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    choices = rigid or matches
    if len(choices) != 1:
        raise RuntimeError(f"expected one {name} body, found {[str(p.GetPath()) for p in choices]}")
    return choices[0]


def has_collision_api(prim, stop):
    current = prim
    while current and current.IsValid():
        if current.HasAPI(UsdPhysics.CollisionAPI):
            enabled = current.GetAttribute("physics:collisionEnabled").Get()
            return enabled is not False
        if current == stop:
            break
        current = current.GetParent()
    return False


def mesh_points(stage, root, frame):
    cache = UsdGeom.XformCache()
    frame_inv = cache.GetLocalToWorldTransform(frame).GetInverse()
    groups = {name: [] for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger")}
    records = []
    all_points = []
    for prim in all_prims(stage):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        owner = None
        parent = prim
        while parent and parent.IsValid():
            if parent.GetName() in groups:
                owner = parent.GetName()
                break
            if parent == root:
                break
            parent = parent.GetParent()
        if owner is None or not has_collision_api(prim, root):
            continue
        mesh = UsdGeom.Mesh(prim)
        xf = cache.GetLocalToWorldTransform(prim) * frame_inv
        pts = transform(mesh.GetPointsAttr().Get() or [], xf)
        if not len(pts):
            continue
        groups[owner].append(pts)
        all_points.append(pts)
        records.append({"path": str(prim.GetPath()), "owner": owner,
                        "point_count": int(len(pts)),
                        "bounds_hand_local_stage_units": bounds(pts)})
    merged = {k: np.vstack(v) for k, v in groups.items() if v}
    return merged, records, (np.vstack(all_points) if all_points else np.empty((0, 3)))


def joint_report(stage):
    result = []
    for prim in all_prims(stage):
        if "finger_joint" not in prim.GetName():
            continue
        attrs = {str(a.GetName()): a.Get() for a in prim.GetAttributes()
                 if a.GetName().startswith(("physics:", "drive:")) and a.Get() is not None}
        for key, val in list(attrs.items()):
            try:
                attrs[key] = [float(x) for x in val]
            except (TypeError, ValueError):
                attrs[key] = str(val)
        rels = {}
        for name in ("physics:body0", "physics:body1"):
            rels[name] = [str(x) for x in prim.GetRelationship(name).GetTargets()]
        result.append({"path": str(prim.GetPath()), "type": str(prim.GetTypeName()),
                       "attributes": attrs, "bodies": rels})
    return result


def bolt_report(scene_scale=SCENE_SCALE):
    stage = Usd.Stage.Open(str(BOLT))
    if stage is None:
        raise RuntimeError(f"cannot open {BOLT}")
    root = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    cache = UsdGeom.XformCache()
    points = []
    root_inv = cache.GetLocalToWorldTransform(root).GetInverse()
    for prim in all_prims(stage):
        if prim.IsA(UsdGeom.Mesh):
            xf = cache.GetLocalToWorldTransform(prim) * root_inv
            points.extend(transform(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], xf))
    p = np.asarray(points)
    head = p[p[:, 1] >= CAP_BOTTOM_Y_SOURCE - .02]
    head_box = bounds(head)
    return {"path": str(BOLT.relative_to(ROOT)), "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "upright_axis": "source +Y maps to world +Z", "source_units_per_scene_meter": 1 / scene_scale,
            "cap_bottom_y_source_units": CAP_BOTTOM_Y_SOURCE,
            "cap_top_y_source_units": float(p[:, 1].max()),
            "cap_height_mm_at_env_scale": (float(p[:, 1].max()) - CAP_BOTTOM_Y_SOURCE) * scene_scale * 1000,
            "cap_head_bounds_source_units_xyz": head_box,
            "cap_head_width_xz_mm_at_env_scale": [(head_box[i][1] - head_box[i][0]) * scene_scale * 1000
                                                   for i in (0, 2)]}


def stage_inventory():
    stage = Usd.Stage.Open(str(PANDA))
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    hand = body_prim(stage, "panda_hand")
    cache = UsdGeom.XformCache()
    hand_inv = cache.GetLocalToWorldTransform(hand).GetInverse()
    result = []
    for prim in stage.Traverse():
        path_name = str(prim.GetPath()).lower()
        if not any(key in path_name for key in ("hand", "finger", "tool_center")):
            continue
        attrs = {str(a.GetName()): str(a.Get()) for a in prim.GetAttributes()
                 if a.GetName().startswith("physics:") and a.Get() is not None}
        if prim.IsA(UsdGeom.Gprim) or "finger_joint" in path_name or prim.GetName() in {
            "panda_hand", "panda_leftfinger", "panda_rightfinger", "tool_center"
        }:
            row = {"path": str(prim.GetPath()), "type": str(prim.GetTypeName()),
                           "schemas": [str(x) for x in prim.GetAppliedSchemas()], "physics_attrs": attrs,
                           "collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
                           "rigid_body_api": bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))}
            if prim.GetName() in PROP_NAMES:
                local = cache.GetLocalToWorldTransform(prim) * hand_inv
                row["hand_local_matrix"] = [[float(local[i][j]) for j in range(4)] for i in range(4)]
            result.append(row)
    props = []
    for name in PROP_NAMES:
        prop_path = PANDA_PROPS / f"{name}.usd"
        prop_stage = Usd.Stage.Open(str(prop_path))
        if prop_stage is None:
            props.append({"path": str(prop_path.relative_to(ROOT)), "error": "cannot open"})
            continue
        cache = UsdGeom.XformCache()
        prims = []
        for prim in all_prims(prop_stage):
            if not (prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI)
                    or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                continue
            row = {"path": str(prim.GetPath()), "type": str(prim.GetTypeName()),
                   "schemas": [str(x) for x in prim.GetAppliedSchemas()],
                   "collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
                   "rigid_body_api": bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))}
            if prim.IsA(UsdGeom.Mesh):
                pts = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
                xf = cache.GetLocalToWorldTransform(prim)
                world = transform(pts, xf)
                row["point_count"] = len(pts)
                row["stage_bounds"] = bounds(world) if len(world) else None
                row["ancestor_collision"] = has_collision_api(prim, prop_stage.GetPseudoRoot())
            prims.append(row)
        props.append({"path": str(prop_path.relative_to(ROOT)),
                      "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(prop_stage)),
                      "up_axis": str(UsdGeom.GetStageUpAxis(prop_stage)),
                      "default_prim": str((prop_stage.GetDefaultPrim() or prop_stage.GetPseudoRoot()).GetPath()),
                      "prims": prims})
    return {"root": str(root.GetPath()), "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "selected_prims": result, "referenced_prop_stages": props}


def audit():
    tcp_offset_m = np.asarray(env_cfg_value("tcp_offset_from_hand_local_m"), dtype=float)
    scene_scale = float(env_cfg_value("cad_stage_scale"))
    if not np.allclose(tcp_offset_m[:2], 0.0):
        raise RuntimeError("this top-down clearance calculation requires a hand-local Z-only TCP offset")
    tcp_hand_z_m = float(tcp_offset_m[2])
    stage = Usd.Stage.Open(str(PANDA))
    if stage is None:
        raise RuntimeError(f"cannot open {PANDA}")
    hand = body_prim(stage, "panda_hand")
    frame = UsdGeom.XformCache().GetLocalToWorldTransform(hand)
    inv = frame.GetInverse()
    bodies = {name: body_prim(stage, name) for name in
              ("panda_hand", "panda_leftfinger", "panda_rightfinger")}
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    groups, mesh_records = prop_collision_points(stage, hand)
    all_collision = np.vstack(list(groups.values()))
    if not all_collision.size:
        raise RuntimeError("no collision-tagged hand/finger mesh points found")

    local_frames = {}
    body_report = {}
    for name, prim in bodies.items():
        local = UsdGeom.XformCache().GetLocalToWorldTransform(prim) * inv
        local_frames[name] = local
        row = local.GetRow3(3)
        body_report[name] = {"path": str(prim.GetPath()),
                             "hand_local_matrix": [[float(local[i][j]) for j in range(4)] for i in range(4)],
                             "hand_local_translation_stage_units": [float(row[i]) for i in range(3)],
                             "collision_mesh_count": sum(r["body"] == name for r in mesh_records),
                             "collision_bounds_hand_local_m": [[v * mpu for v in a]
                                                               for a in bounds(groups[name])] if name in groups else None}

    left = groups.get("panda_leftfinger")
    right = groups.get("panda_rightfinger")
    left_center = np.mean(bounds(left), axis=1) if left is not None else None
    right_center = np.mean(bounds(right), axis=1) if right is not None else None
    delta = right_center - left_center
    close_axis = delta / np.linalg.norm(delta)
    left_proj = left @ close_axis
    right_proj = right @ close_axis
    gap_units = float(right_proj.min() - left_proj.max())

    # For a top-down grasp, local +Z is directed downward; the env TCP offset is hand-local +Z.
    all_z = all_collision[:, 2]
    finger_z = np.vstack([left, right])[:, 2]
    hand_local_bounds = bounds(all_collision)
    finger_local_bounds = bounds(np.vstack([left, right]))
    dist_bottom_finger_tcp_m = tcp_hand_z_m - float(finger_z.max()) * mpu
    dist_bottom_whole_tcp_m = tcp_hand_z_m - float(all_z.max()) * mpu
    palm_max_z_m = float(groups["panda_hand"][:, 2].max()) * mpu

    tool = [p for p in all_prims(stage) if p.GetName() == "tool_center"]
    tool_report = []
    for p in tool:
        rel = UsdGeom.XformCache().GetLocalToWorldTransform(p) * inv
        tool_report.append({"path": str(p.GetPath()), "hand_local_matrix": [[float(rel[i][j]) for j in range(4)] for i in range(4)],
                            "hand_local_translation_m": [float(rel.GetRow3(3)[i]) * mpu for i in range(3)]})

    cap = bolt_report(scene_scale)
    # A 1 mm pad-to-cover margin is explicit task geometry, not inferred physics.
    target_tcp_above_cover_m = .001 - dist_bottom_whole_tcp_m
    pad_vertical_span_m = (float(finger_local_bounds[2][1] - finger_local_bounds[2][0]) * mpu)
    cap_height_m = cap["cap_height_mm_at_env_scale"] / 1000.0
    cap_overlap_m = max(0.0, min(cap_height_m - .001, pad_vertical_span_m))
    palm_clearance_m = target_tcp_above_cover_m + tcp_hand_z_m - palm_max_z_m
    tool_is_rigid_body = any(p.GetName() == "tool_center" and
                             p.HasAPI(UsdPhysics.RigidBodyAPI) for p in all_prims(stage))

    return {
        "scope": "read-only USD collision geometry; no Kit, physics, GPU, or grasp dynamics",
        "sources": {"env_cfg": "runtime/bolt_harness/env_cfg.py", "panda_usd": str(PANDA.relative_to(ROOT)),
                    "bolt_usd": cap["path"]},
        "panda_stage": {"meters_per_unit": mpu, "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                        "root_prim": str((stage.GetDefaultPrim() or stage.GetPseudoRoot()).GetPath()),
                        "hand_path": str(hand.GetPath()),
                        "hand_local_to_stage_matrix": [[float(frame[i][j]) for j in range(4)] for i in range(4)]},
        "tcp": {"env_tcp_body_name": "tool_center", "tool_center_usd_prims": tool_report,
                "fallback_hand_local_offset_m": tcp_offset_m.tolist(),
                "tool_center_has_usd_rigid_body_api": tool_is_rigid_body,
                "runtime_fallback_status": "not measured by this offline USD audit; env.py branches on articulation body_names",
                "top_down_assumption": "hand-local +Z points down in world; this places finger distal ends toward the cover"},
        "finger_closure": {"axis_hand_local_unit_from_collision_centers": close_axis.tolist(),
                           "left_to_right_collision_center_m": (delta * mpu).tolist(),
                           "default_collision_inner_gap_m": gap_units * mpu,
                           "cap_width_target_mm": cap["cap_head_width_xz_mm_at_env_scale"],
                           "joint_authored_data": joint_report(stage)},
        "collision_geometry": {"bodies": body_report, "collision_meshes": mesh_records,
                               "whole_hand_finger_bounds_hand_local_m":
                                   [[v * mpu for v in a] for a in hand_local_bounds],
                               "finger_union_bounds_hand_local_m":
                                   [[v * mpu for v in a] for a in finger_local_bounds],
                               "finger_distal_bottom_hand_local_z_m": float(finger_local_bounds[2][1]) * mpu,
                               "finger_distal_bottom_relative_to_tcp_world_up_m": dist_bottom_finger_tcp_m,
                               "whole_collision_bottom_relative_to_tcp_world_up_m": dist_bottom_whole_tcp_m},
        "bolt_cap": cap,
        "nominal_geometry_grasp": {"cover_plane_reference": "clearances are relative to the cover top plane",
                                    "distal_bottom_clearance_above_cover_m": 0.001,
                                    "tcp_z_relative_to_cover_plane_m": target_tcp_above_cover_m,
                                    "finger_pad_vertical_span_m": pad_vertical_span_m,
                                    "bolt_cap_height_m": cap_height_m,
                                    "estimated_vertical_overlap_with_cap_m": cap_overlap_m,
                                    "palm_clearance_to_cover_at_that_tcp_m": palm_clearance_m,
                                    "note": "height aligns the lowest collision point 1 mm above cover; cap-side grasp feasibility remains geometric, not force-closure"},
    }


def self_test():
    axis = np.asarray([0., 1., 0.])
    axis /= np.linalg.norm(axis)
    assert np.allclose(axis, [0, 1, 0])
    assert math.isclose(TCP_HAND_Z_M * 1000, 103.4)
    assert np.allclose(env_cfg_value("tcp_offset_from_hand_local_m"), [0.0, 0.0, 0.1034])
    assert math.isclose(env_cfg_value("cad_stage_scale"), SCENE_SCALE)
    print("Panda gripper USD audit self-test: PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    elif "--inventory" in sys.argv:
        print(json.dumps(stage_inventory(), indent=2))
    else:
        print(json.dumps(audit(), indent=2))
