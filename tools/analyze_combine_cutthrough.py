#!/usr/bin/env python3
"""Analyze combine logic for all connecting pairs.

For each (child, parent, plug, socket) pair, compute the child's final
transform after combine and check:
1. The plug point lands on the socket point (alignment check)
2. The child mesh doesn't cut through the parent mesh (penetration check)
   - Uses the child's BBox in the parent's local frame
   - Checks if the child extends past the parent's surface into its interior

Reports the child's position relative to the parent surface normal.
"""
import math
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Gf, Sdf, Usd, UsdGeom
from runtime.magic_assembly import MagicAssemblyManager, HARDCODED_FIT_OFFSETS, DEFAULT_SOCKET_MAP, DEFAULT_PLUG_MAP


# All connecting pairs from the assembly manual
CONNECT_PAIRS = [
    # (child_prim_name, parent_prim_name, plug_name, socket_name)
    # Hub covers on Casing Top
    ("Hub_Cover_Output", "Casing_Top",   "plug_main", "socket_hub_output"),
    ("Hub_Cover_Input",  "Casing_Top",   "plug_main", "socket_hub_input"),
    ("Hub_Cover_Small",  "Casing_Top",   "plug_main", "socket_hub_small"),
    # Hub covers on Casing Base
    ("Hub_Cover_Output", "Casing_Base",  "plug_main", "socket_hub_output"),
    ("Hub_Cover_Small",  "Casing_Base",  "plug_main", "socket_hub_small_1"),
    ("Hub_Cover_Small",  "Casing_Base",  "plug_main", "socket_hub_small_2"),
    # Gears on shafts
    ("Transfer_Gear",    "Transfer_Shaft","plug_main", "socket_gear"),
    ("Output_Gear",      "Output_Shaft",  "plug_main", "socket_gear"),
    # Shafts into Casing Base
    ("Input_Shaft",      "Casing_Base",   "plug_main", "socket_gear_input"),
    ("Transfer_Shaft",   "Casing_Base",   "plug_main", "socket_gear_transfer"),
    ("Output_Shaft",     "Casing_Base",   "plug_main", "socket_gear_output"),
    # Casing mating
    ("Casing_Top",       "Casing_Base",   "plug_casing_mate", "socket_casing_mate"),
    # Bolts
    ("M6_Hub_Bolt",      "Casing_Top",    "plug_main", "socket_bolt_hub_1"),
    ("M6_Hub_Bolt",      "Casing_Base",   "plug_main", "socket_bolt_hub_1"),
    ("M10_Casing_Bolt",  "Casing_Base",   "plug_main", "socket_bolt_casing_1"),
    # Nuts
    ("M10_Casing_Nut",   "Casing_Base",   "plug_main", "socket_nut_casing_1"),
    # Accessories
    ("Oil_Level_Indicator", "Casing_Base", "plug_main", "socket_oil_1"),
    ("Breather_Plug",       "Casing_Base", "plug_main", "socket_breather"),
]

PARTS_DIR = os.path.join(_REPO, "assets", "parts")


def build_test_scene():
    """Build a temp scene with all parts at origin."""
    all_part_names = set()
    for c, p, _, _ in CONNECT_PAIRS:
        all_part_names.add(c.replace("_", " "))
        all_part_names.add(p.replace("_", " "))

    tmpdir = tempfile.mkdtemp(prefix="combine_test_")
    scene_path = os.path.join(tmpdir, "test_scene.usda")
    stage = Usd.Stage.CreateNew(scene_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    for part_name in sorted(all_part_names):
        prim_name = part_name.replace(" ", "_")
        xform = UsdGeom.Xform.Define(stage, f"/World/{prim_name}")
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
        usd_path = os.path.join(PARTS_DIR, f"{part_name}.usd")
        xform.GetPrim().GetReferences().AddReference(usd_path)

    stage.GetRootLayer().Save()
    return scene_path, all_part_names


def restore_scene(scene_path, all_part_names):
    layer = Sdf.Layer.Find(scene_path)
    if layer:
        layer.Clear()
    else:
        if os.path.exists(scene_path):
            os.remove(scene_path)
        layer = Sdf.Layer.CreateNew(scene_path)
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    for part_name in sorted(all_part_names):
        prim_name = part_name.replace(" ", "_")
        xform = UsdGeom.Xform.Define(stage, f"/World/{prim_name}")
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
        usd_path = os.path.join(PARTS_DIR, f"{part_name}.usd")
        xform.GetPrim().GetReferences().AddReference(usd_path)
    stage.GetRootLayer().Save()


def get_bbox_in_world(prim):
    """Get the bounding box corners in world space."""
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )
    bbox = cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedRange()
    return rng


def analyze_combine_pair(scene_path, all_part_names, child_name, parent_name,
                          plug_name, socket_name):
    """Combine a pair and analyze the result for alignment and penetration."""
    # Re-open stage fresh
    stage = Usd.Stage.Open(scene_path)
    logs = []
    ma = MagicAssemblyManager(
        lambda: stage,
        use_omni_commands=False,
        logger=lambda msg: logs.append(msg),
    )

    # Get pre-combine bboxes
    child_prim = stage.GetPrimAtPath(f"/World/{child_name}")
    parent_prim = stage.GetPrimAtPath(f"/World/{parent_name}")

    if not child_prim or not child_prim.IsValid():
        return {"error": f"child {child_name} not found"}
    if not parent_prim or not parent_prim.IsValid():
        return {"error": f"parent {parent_name} not found"}

    child_bbox_pre = get_bbox_in_world(child_prim)
    parent_bbox = get_bbox_in_world(parent_prim)

    # Get socket and plug positions before combine
    tc = Usd.TimeCode.Default()
    socket_prim = parent_prim.GetChild(socket_name) if socket_name else None
    plug_prim_child = child_prim.GetChild(plug_name) if plug_name else None

    socket_world = None
    plug_world_pre = None
    if socket_prim and socket_prim.IsValid():
        xf = UsdGeom.Xformable(socket_prim)
        mat = xf.ComputeLocalToWorldTransform(tc)
        socket_world = Gf.Vec3d(mat.GetRow3(3)[0], mat.GetRow3(3)[1], mat.GetRow3(3)[2])
    if plug_prim_child and plug_prim_child.IsValid():
        xf = UsdGeom.Xformable(plug_prim_child)
        mat = xf.ComputeLocalToWorldTransform(tc)
        plug_world_pre = Gf.Vec3d(mat.GetRow3(3)[0], mat.GetRow3(3)[1], mat.GetRow3(3)[2])

    # Do the combine
    ok = ma.combine(child_name, parent_name, plug_name, socket_name)
    if not ok:
        return {"error": "combine failed", "logs": logs}

    # Re-read stage to see result
    stage = Usd.Stage.Open(scene_path)

    # Find the reparented child
    new_child_path = f"/World/{parent_name}/{child_name}"
    new_child_prim = stage.GetPrimAtPath(new_child_path)
    if not new_child_prim or not new_child_prim.IsValid():
        return {"error": f"child not found at {new_child_path} after combine"}

    # Get child bbox after combine
    child_bbox_post = get_bbox_in_world(new_child_prim)

    # Get plug world position after combine
    plug_path_post = f"{new_child_path}/{plug_name}"
    plug_prim_post = stage.GetPrimAtPath(plug_path_post)
    plug_world_post = None
    if plug_prim_post and plug_prim_post.IsValid():
        xf = UsdGeom.Xformable(plug_prim_post)
        mat = xf.ComputeLocalToWorldTransform(tc)
        plug_world_post = Gf.Vec3d(mat.GetRow3(3)[0], mat.GetRow3(3)[1], mat.GetRow3(3)[2])

    # Socket world after combine (should be same)
    socket_path = f"/World/{parent_name}/{socket_name}"
    socket_prim_post = stage.GetPrimAtPath(socket_path)
    socket_world_post = None
    if socket_prim_post and socket_prim_post.IsValid():
        xf = UsdGeom.Xformable(socket_prim_post)
        mat = xf.ComputeLocalToWorldTransform(tc)
        socket_world_post = Gf.Vec3d(mat.GetRow3(3)[0], mat.GetRow3(3)[1], mat.GetRow3(3)[2])

    # Get child world origin after combine
    child_xf = UsdGeom.Xformable(new_child_prim)
    child_world_mat = child_xf.ComputeLocalToWorldTransform(tc)
    child_origin = Gf.Vec3d(
        child_world_mat.GetRow3(3)[0],
        child_world_mat.GetRow3(3)[1],
        child_world_mat.GetRow3(3)[2],
    )

    # Alignment check
    alignment_dist = None
    if plug_world_post and socket_world_post:
        d = plug_world_post - socket_world_post
        alignment_dist = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

    # Penetration check: does child bbox overlap with parent interior?
    # For casings (parent), the surface normals are:
    #   Z+: exterior top / parting face
    #   Z-: exterior bottom / parting face
    # A hub cover on Z+ face should have its bbox ABOVE (or at) Z=27.9
    # A hub cover on Z- face should have its bbox BELOW (or at) Z=-27.9
    child_min = child_bbox_post.GetMin()
    child_max = child_bbox_post.GetMax()
    parent_min = parent_bbox.GetMin()
    parent_max = parent_bbox.GetMax()

    result = {
        "ok": ok,
        "alignment_dist": alignment_dist,
        "child_bbox_pre": (child_bbox_pre.GetMin(), child_bbox_pre.GetMax()),
        "child_bbox_post": (child_min, child_max),
        "parent_bbox": (parent_min, parent_max),
        "child_origin": child_origin,
        "socket_world": socket_world_post,
        "plug_world": plug_world_post,
    }

    # Check for Z-axis penetration (most common for hub covers on casings)
    if socket_world_post:
        sz = socket_world_post[2]
        # Socket on Z+ face: child should be at or above sz
        if sz > 0:
            # Child's min Z should be >= socket Z (minus small tolerance for bearing seat)
            penetration_z = sz - child_min[2]
            result["surface_z"] = sz
            result["child_min_z"] = child_min[2]
            result["penetration_z"] = penetration_z
            result["penetration_dir"] = "into Z+ surface"
        elif sz < 0:
            # Socket on Z- face: child's max Z should be <= socket Z
            penetration_z = child_max[2] - sz
            result["surface_z"] = sz
            result["child_max_z"] = child_max[2]
            result["penetration_z"] = penetration_z
            result["penetration_dir"] = "into Z- surface"

    return result


def main():
    print(f"\n{'='*75}")
    print(f"  COMBINE CUT-THROUGH ANALYSIS")
    print(f"{'='*75}")

    scene_path, all_part_names = build_test_scene()

    n_ok = 0
    n_warn = 0
    n_fail = 0

    for child_name, parent_name, plug_name, socket_name in CONNECT_PAIRS:
        print(f"\n  --- {child_name} -> {parent_name} ({plug_name} -> {socket_name}) ---")

        result = analyze_combine_pair(
            scene_path, all_part_names,
            child_name, parent_name, plug_name, socket_name,
        )

        if "error" in result:
            print(f"    ERROR: {result['error']}")
            if "logs" in result:
                for log in result["logs"][-5:]:
                    print(f"      {log}")
            n_fail += 1
            restore_scene(scene_path, all_part_names)
            continue

        # Report alignment
        ad = result.get("alignment_dist")
        if ad is not None:
            status = "OK" if ad < 1.0 else "WARN" if ad < 5.0 else "FAIL"
            print(f"    Alignment: plug-socket distance = {ad:.4f}mm [{status}]")
        else:
            print(f"    Alignment: could not measure (plug/socket missing)")

        # Report positions
        if result.get("socket_world"):
            sw = result["socket_world"]
            print(f"    Socket world: ({sw[0]:.2f}, {sw[1]:.2f}, {sw[2]:.2f})")
        if result.get("plug_world"):
            pw = result["plug_world"]
            print(f"    Plug world:   ({pw[0]:.2f}, {pw[1]:.2f}, {pw[2]:.2f})")
        co = result.get("child_origin")
        if co:
            print(f"    Child origin: ({co[0]:.2f}, {co[1]:.2f}, {co[2]:.2f})")

        # Report bbox
        cb = result.get("child_bbox_post")
        pb = result.get("parent_bbox")
        if cb:
            print(f"    Child BBox:   [{cb[0][0]:.1f},{cb[0][1]:.1f},{cb[0][2]:.1f}] -> "
                  f"[{cb[1][0]:.1f},{cb[1][1]:.1f},{cb[1][2]:.1f}]")
        if pb:
            print(f"    Parent BBox:  [{pb[0][0]:.1f},{pb[0][1]:.1f},{pb[0][2]:.1f}] -> "
                  f"[{pb[1][0]:.1f},{pb[1][1]:.1f},{pb[1][2]:.1f}]")

        # Report penetration
        pen = result.get("penetration_z")
        if pen is not None:
            pen_dir = result.get("penetration_dir", "")
            if pen > 1.0:  # more than 1mm penetration
                print(f"    PENETRATION: {pen:.2f}mm {pen_dir}")
                n_warn += 1
            else:
                print(f"    No significant penetration ({pen:.2f}mm)")
                n_ok += 1
        else:
            n_ok += 1

        # Restore for next pair
        restore_scene(scene_path, all_part_names)

    print(f"\n{'='*75}")
    print(f"  RESULTS: {n_ok} OK, {n_warn} warnings, {n_fail} failed")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
