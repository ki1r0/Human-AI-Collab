#!/usr/bin/env python3
"""tools/import_stl_parts.py — Convert STL files to centered USD part files,
then reference them into a demo scene USD.

Workflow
--------
1. For each .stl in assets/parts/ that has no matching .usd:
   a. Parse binary or ASCII STL.
   b. Write a USD file (World → node_ → mesh_) matching the layout of the
      existing parts (metersPerUnit=0.01, upAxis=Y, OmniPBR material).
   c. Center the geometry axis via BBoxCache (rigorously_center_prim).
2. Create assets/simple_room_demo.usd:
   - Sublayers assets/simple_room_scene.usd (inherits room + 4 existing parts).
   - Adds references to every *new* part with the same transform convention
     already used for the existing 4 parts:
       xformOp:translate  (position on table)
       xformOp:orient     quaternion
       xformOp:scale      (0.2, 0.2, 0.2)   ← user-requested part scale
       xformOp:rotateX    90                  ← Y-up → Z-up
       xformOp:scale      (0.01, 0.01, 0.01) ← mm → m unit conversion

Usage (run from repo root)
--------------------------
    PYTHONPATH=<usd_lib> <python3.11> tools/import_stl_parts.py
"""
from __future__ import annotations

import math
import os
import struct
import sys
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # type: ignore
from runtime.asset_utils import rigorously_center_prim  # noqa: E402


# ---------------------------------------------------------------------------
# STL reader
# ---------------------------------------------------------------------------

Vec3 = Tuple[float, float, float]


def _read_stl(path: str) -> Tuple[List[Vec3], List[Vec3]]:
    """Return (vertices, face_normals) from a binary or ASCII STL file.

    vertices : flat list of length 3*N_tri – each triangle contributes 3.
    face_normals : flat list of length 3*N_tri – one normal per face vertex
                   (faceVarying interpolation in USD).
    """
    with open(path, "rb") as f:
        raw = f.read()

    # Detect ASCII vs binary.  Binary STLs occasionally have "solid" in the
    # 80-byte header, so we use the file-size cross-check instead.
    header = raw[:80]
    is_ascii = False
    if header.lstrip().startswith(b"solid"):
        n_tri_bin = struct.unpack_from("<I", raw, 80)[0]
        expected_size = 80 + 4 + n_tri_bin * 50
        if abs(expected_size - len(raw)) > 4:
            is_ascii = True

    if is_ascii:
        return _read_stl_ascii(raw.decode("utf-8", errors="replace"))
    else:
        return _read_stl_binary(raw)


def _read_stl_binary(raw: bytes) -> Tuple[List[Vec3], List[Vec3]]:
    n_tri = struct.unpack_from("<I", raw, 80)[0]
    vertices: List[Vec3] = []
    normals: List[Vec3] = []
    offset = 84
    for _ in range(n_tri):
        nx, ny, nz = struct.unpack_from("<3f", raw, offset)
        v0 = struct.unpack_from("<3f", raw, offset + 12)
        v1 = struct.unpack_from("<3f", raw, offset + 24)
        v2 = struct.unpack_from("<3f", raw, offset + 36)
        offset += 50
        n = (float(nx), float(ny), float(nz))
        vertices += [
            (float(v0[0]), float(v0[1]), float(v0[2])),
            (float(v1[0]), float(v1[1]), float(v1[2])),
            (float(v2[0]), float(v2[1]), float(v2[2])),
        ]
        normals += [n, n, n]
    return vertices, normals


def _read_stl_ascii(text: str) -> Tuple[List[Vec3], List[Vec3]]:
    vertices: List[Vec3] = []
    normals: List[Vec3] = []
    current_normal: Vec3 = (0.0, 0.0, 0.0)
    tri_verts: List[Vec3] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("facet normal"):
            parts = line.split()
            current_normal = (float(parts[2]), float(parts[3]), float(parts[4]))
            tri_verts = []
        elif line.startswith("vertex"):
            parts = line.split()
            tri_verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("endfacet"):
            if len(tri_verts) == 3:
                vertices += tri_verts
                normals += [current_normal, current_normal, current_normal]
            tri_verts = []
    return vertices, normals


# ---------------------------------------------------------------------------
# USD writer
# ---------------------------------------------------------------------------

def _make_part_usd(
    stl_path: str,
    usd_path: str,
) -> bool:
    """Convert *stl_path* to a centered USD at *usd_path*.

    Returns True on success.
    """
    name = os.path.basename(stl_path)
    print(f"  Converting: {name}")

    # 1. Parse STL.
    try:
        verts, face_normals = _read_stl(stl_path)
    except Exception as exc:
        print(f"    [ERROR] Failed to read STL: {exc}")
        return False

    n_tri = len(verts) // 3
    print(f"    triangles: {n_tri:,}  vertices: {len(verts):,}")

    # 2. Create USD stage.
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)   # centimetres (matching existing parts)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # /World  (defaultPrim — MagicAssembly replaces its xformOps at runtime)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    world.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1, 0, 0, 0))
    world.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1, 1, 1))

    # /World/Looks  + DefaultMaterial  (OmniPBR, grey)
    looks_scope = UsdGeom.Scope.Define(stage, "/World/Looks")
    mat = UsdShade.Material.Define(stage, "/World/Looks/DefaultMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/DefaultMaterial/DefaultMaterial")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
    ).Set("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.627451, 0.627451, 0.627451)
    )

    # /World/node_  (intermediate Xform — our centering offset lives here)
    node = UsdGeom.Xform.Define(stage, "/World/node_")
    node.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    node.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1, 0, 0, 0))
    node.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1, 1, 1))

    # /World/node_/mesh_
    mesh_prim_path = "/World/node_/mesh_"
    mesh = UsdGeom.Mesh.Define(stage, mesh_prim_path)

    pts_vt = Vt.Vec3fArray(len(verts))
    for i, v in enumerate(verts):
        pts_vt[i] = Gf.Vec3f(v[0], v[1], v[2])

    fvc = Vt.IntArray(n_tri, 3)          # all triangles
    fvi = Vt.IntArray(list(range(len(verts))))

    nrm_vt = Vt.Vec3fArray(len(face_normals))
    for i, n in enumerate(face_normals):
        nrm_vt[i] = Gf.Vec3f(n[0], n[1], n[2])

    mesh.GetPointsAttr().Set(pts_vt)
    mesh.GetFaceVertexCountsAttr().Set(fvc)
    mesh.GetFaceVertexIndicesAttr().Set(fvi)
    mesh.GetNormalsAttr().Set(nrm_vt)
    UsdGeom.Primvar(mesh.GetNormalsAttr()).SetInterpolation("faceVarying")
    mesh.GetSubdivisionSchemeAttr().Set("none")

    # Bind the DefaultMaterial.
    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding_api.Bind(mat)

    # 3. Center via BBoxCache.
    node_prim = stage.GetPrimAtPath("/World/node_")
    centroid = rigorously_center_prim(node_prim, stage, logger=lambda m: None)
    if centroid is None:
        print("    [WARN] No geometry centroid found — file saved un-centered.")
    else:
        cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
        print(f"    centroid_before=({cx:.3f}, {cy:.3f}, {cz:.3f})")

    # 4. Save.
    stage.GetRootLayer().Save()
    print(f"    → saved: {os.path.relpath(usd_path, _REPO_ROOT)}")
    return True


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

# Transform used in simple_room_scene.usd for all gearbox parts:
#   scale=(0.01,0.01,0.01)  mm→m unit conversion
#   rotateX=90              Y-up (part) → Z-up (scene)
#   scale=(0.2,0.2,0.2)     user-requested part scale
#   orient=quaternion        part orientation
#   translate=(x,y,z)        world position in metres

# Grid layout for 11 new parts on the table.
# Existing parts occupy roughly x∈[0..0.8], y∈[0..1.4], z≈1.7
# We spread the new parts on the opposite side: x∈[-2.0..-0.5], y∈[0..1.5]
_NEW_PART_GRID: list[tuple[str, tuple[float, float, float]]] = [
    ("Casing Top",            (-1.5,  0.5,  1.1)),
    ("Hub Cover Output",      (-1.0,  0.5,  1.1)),
    ("Hub Cover Small",       (-0.5,  0.5,  1.1)),
    ("Input Shaft",           ( 0.0,  0.5,  1.1)),
    ("M10 Casing Bolt",       (-1.5,  0.0,  1.1)),
    ("M10 Casing Nut",        (-1.0,  0.0,  1.1)),
    ("Oil Level Indicator",   (-0.5,  0.0,  1.1)),
    ("Output Gear",           ( 0.0,  0.0,  1.1)),
    ("Output Shaft",          (-1.5, -0.5,  1.1)),
    ("Transfer Gear",         (-1.0, -0.5,  1.1)),
    ("Transfer Shaft",        (-0.5, -0.5,  1.1)),
]


def _prim_name(part_name: str) -> str:
    """'Casing Top' → 'Casing_Top'"""
    return part_name.replace(" ", "_")


def _build_demo_scene(
    scene_path: str,
    new_part_names: list[str],
    parts_dir: str,
) -> None:
    """Create assets/simple_room_demo.usd.

    Sublayers simple_room_scene.usd and adds references to all *new_part_names*.
    """
    assets_dir = os.path.dirname(scene_path)
    base_scene = os.path.join(assets_dir, "simple_room_scene.usd")

    # Remove stale file so CreateNew succeeds.
    if os.path.exists(scene_path):
        os.remove(scene_path)

    stage = Usd.Stage.CreateNew(scene_path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    layer = stage.GetRootLayer()

    # Sublayer the original scene so the room + 4 existing parts are inherited.
    # Must assign the list, not append to the returned copy.
    layer.subLayerPaths = ["./simple_room_scene.usd"]

    # Define /Root in this layer so we can author child prims here.
    # "def Xform" gives it a concrete spec that Isaac Sim can expand.
    root_xf = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root_xf.GetPrim())

    # Build a lookup for the grid positions.
    grid_map = {name: pos for name, pos in _NEW_PART_GRID}

    for part_name in new_part_names:
        pname = _prim_name(part_name)
        prim_path = "/Root/" + pname
        ref_path = "./parts/" + part_name + ".usd"
        pos = grid_map.get(part_name, (0.0, 0.0, 1.1))

        xform = UsdGeom.Xform.Define(stage, prim_path)
        prim = xform.GetPrim()

        # Set transform ops BEFORE adding the reference, so AddXformOp does
        # not see the referenced file's defaultPrim ops in the composed stage.
        # Transform stack (matches existing 4 parts in simple_room_scene.usd):
        #   translate → orient → scale(0.2) → rotateX(90) → scale(0.01)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(pos[0], pos[1], pos[2])
        )
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Quatf(1, 0, 0, 0)
        )
        xform.AddScaleOp(
            UsdGeom.XformOp.PrecisionFloat, "halfScale"
        ).Set(Gf.Vec3f(0.2, 0.2, 0.2))
        xform.AddRotateXOp(UsdGeom.XformOp.PrecisionFloat).Set(90.0)
        xform.AddScaleOp(
            UsdGeom.XformOp.PrecisionFloat, "unitScale"
        ).Set(Gf.Vec3f(0.01, 0.01, 0.01))

        # Add USD reference last (its xformOpOrder is weaker than our local ops).
        prim.GetReferences().AddReference(ref_path)

        print("    /Root/%-22s  ref=%-35s  pos=(%.2f,%.2f,%.2f)" % (
            pname, ref_path, pos[0], pos[1], pos[2]))

    layer.Save()
    print("  → saved: " + os.path.relpath(scene_path, _REPO_ROOT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parts_dir = os.path.join(_REPO_ROOT, "assets", "parts")
    assets_dir = os.path.join(_REPO_ROOT, "assets")
    demo_scene = os.path.join(assets_dir, "simple_room_demo.usd")

    # The 4 parts already imported before this script was written:
    _ALREADY_IMPORTED = {"Breather Plug", "Casing Base", "Hub Cover Input", "M6 Hub Bolt"}

    # Collect all STL stems that are NOT in the already-imported set.
    all_stl = sorted(
        f for f in os.listdir(parts_dir) if f.lower().endswith(".stl")
    )
    new_stl = [f for f in all_stl if f[:-4] not in _ALREADY_IMPORTED]

    # Which of those still need conversion?
    needs_conversion = [
        f for f in new_stl
        if not os.path.exists(os.path.join(parts_dir, f[:-4] + ".usd"))
    ]

    if not needs_conversion:
        print("[INFO] All STL files already have .usd counterparts — skipping conversion.")
    else:
        print("\n" + "="*60)
        print("  Converting %d STL files to USD" % len(needs_conversion))
        print("="*60)

    converted_ok: list[str] = []
    failed: list[str] = []

    for stl_file in needs_conversion:
        stl_path = os.path.join(parts_dir, stl_file)
        usd_path = os.path.join(parts_dir, stl_file[:-4] + ".usd")
        ok = _make_part_usd(stl_path, usd_path)
        if ok:
            converted_ok.append(stl_file[:-4])
        else:
            failed.append(stl_file[:-4])

    if needs_conversion:
        print("\n  Converted: %d  Failed: %d" % (len(converted_ok), len(failed)))
        if failed:
            for f in failed:
                print("    FAIL: " + f)

    # All new parts to reference in the demo scene (whether just converted or already existing).
    all_new_part_names = [f[:-4] for f in new_stl
                          if os.path.exists(os.path.join(parts_dir, f[:-4] + ".usd"))]

    # Build demo scene referencing all new parts.
    print("\n" + "="*60)
    print("  Building " + os.path.relpath(demo_scene, _REPO_ROOT))
    print("="*60)
    _build_demo_scene(demo_scene, all_new_part_names, parts_dir)

    print("\nDone.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
