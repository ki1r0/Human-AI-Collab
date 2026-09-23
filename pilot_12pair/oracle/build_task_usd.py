"""Build lightweight USD fixture scenes from the five-task recipes.

This builder uses USD primitives for the *mutation fixture* while referencing the
repository's CAD parts for appearance.  The fixture primitives carry both their
display geometry and collision API, so the visible intervention and collision
intervention cannot silently diverge.  Isaac calibration is still required before
these scenes are accepted as physical benchmark domains.

Run with a USD-capable Python, for example:

    PYTHONPATH=. conda run -n lingbot-va python -m pilot_12pair.oracle.build_task_usd
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

from .constrained_tasks import DEFAULT_CONFIG, load_config, validate_config


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _mm(value: float) -> float:
    return float(value) / 1000.0


def _add_material(stage, path: str, color: tuple[float, float, float]):
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _apply_visual_and_collision(prim, material) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    geom = UsdGeom.Gprim(prim)
    geom.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.48, 0.52)])
    if material is not None:
        UsdShade.MaterialBindingAPI(prim).Bind(material)
    UsdPhysics.CollisionAPI.Apply(prim)


def _cube(stage, parent: str, name: str, center_mm: tuple[float, float, float], size_mm: tuple[float, float, float], material):
    from pxr import Gf, UsdGeom

    path = f"{parent}/{_slug(name)}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*(_mm(x) for x in center_mm)))
    cube.AddScaleOp().Set(Gf.Vec3f(*(_mm(x) for x in size_mm)))
    _apply_visual_and_collision(cube.GetPrim(), material)
    return cube.GetPrim()


def _cylinder(stage, parent: str, name: str, center_mm: tuple[float, float, float], radius_mm: float, height_mm: float, material):
    from pxr import Gf, UsdGeom

    path = f"{parent}/{_slug(name)}"
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.GetRadiusAttr().Set(_mm(radius_mm))
    cyl.GetHeightAttr().Set(_mm(height_mm))
    cyl.AddTranslateOp().Set(Gf.Vec3d(*(_mm(x) for x in center_mm)))
    _apply_visual_and_collision(cyl.GetPrim(), material)
    return cyl.GetPrim()


def _fixture(stage, task: dict, variant: str, material):
    """Create neutral fixture geometry whose positive pieces define an aperture/slot."""
    from pxr import UsdGeom

    parent = "/World/mutation_fixture"
    UsdGeom.Xform.Define(stage, parent)
    geo = task["geometry"]
    if task["task_id"] == "HCF-01":
        opening = geo["round_hole_diameter_mm"] if variant == "HARD" else geo["keyhole_lobe_diameter_mm"]
        half = opening / 2.0
        wall = 4.0
        # Four positive bars make the same visible aperture used by the collision.
        _cube(stage, parent, "north", (0, half + wall / 2, 0), (opening + 2 * wall, wall, 6), material)
        _cube(stage, parent, "south", (0, -half - wall / 2, 0), (opening + 2 * wall, wall, 6), material)
        _cube(stage, parent, "east", (half + wall / 2, 0, 0), (wall, opening, 6), material)
        _cube(stage, parent, "west", (-half - wall / 2, 0, 0), (wall, opening, 6), material)
        if variant == "COMMUTABLE":
            _cylinder(stage, parent, "head_clearance_lobe", (0, 0, 3.5), opening / 2, 1.0, material)
    elif task["task_id"] == "WSG-01":
        outer = geo["washer_outer_diameter_mm"] / 2 + 5
        inner = 12.0
        bars = [(0, outer, outer - inner, 5), (0, -outer, outer - inner, 5), (outer, 0, 5, outer - inner)]
        if variant == "HARD":
            bars.append((-outer, 0, 5, outer - inner))
        for i, (x, y, sx, sy) in enumerate(bars):
            _cube(stage, parent, f"shroud_{i}", (x, y, 0), (sx, sy, 6), material)
    elif task["task_id"] == "KEY-01":
        width = geo["key_width_mm"] + 2 * geo["keyway_entry_clearance_mm"]
        if variant == "HARD":
            _cube(stage, parent, "closed_end_barrier", (0, 0, 0), (width, 5, 8), material)
        else:
            _cube(stage, parent, "upper_guide", (0, geo["side_access_width_mm"] / 2 + 3, 0), (width, 6, 8), material)
            _cube(stage, parent, "lower_guide", (0, -geo["side_access_width_mm"] / 2 - 3, 0), (width, 6, 8), material)
    elif task["task_id"] == "CAS-01":
        width = geo["bolt_head_envelope_mm"][0]
        depth = geo["captive_pocket_depth_mm"] if variant == "COMMUTABLE" else 8.0
        _cube(stage, parent, "top_lug", (0, 0, 0), (width + 4, width + 4, depth), material)
        if variant == "COMMUTABLE":
            _cube(stage, parent, "captive_pocket_left", (-width / 2 - 2, 0, depth / 2), (4, width + 6, depth), material)
            _cube(stage, parent, "captive_pocket_right", (width / 2 + 2, 0, depth / 2), (4, width + 6, depth), material)
    elif task["task_id"] == "DOW-01":
        d = geo["dowel_diameter_mm"]
        slot = geo["captive_slot_width_mm"] if variant == "COMMUTABLE" else d - 1.0
        _cube(stage, parent, "slot_left", (-slot / 2 - 3, 0, 0), (6, d + 8, 8), material)
        _cube(stage, parent, "slot_right", (slot / 2 + 3, 0, 0), (6, d + 8, 8), material)
    else:  # pragma: no cover - guarded by the five-task config
        raise ValueError(task["task_id"])


def build_one(config: dict, task: dict, variant: str, out_path: Path, repo_root: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    from pxr import Sdf
    world.GetPrim().CreateAttribute("task:task_id", Sdf.ValueTypeNames.String).Set(task["task_id"])
    world.GetPrim().CreateAttribute("task:variant", Sdf.ValueTypeNames.String).Set(variant)
    world.GetPrim().CreateAttribute("task:render_collision_contract", Sdf.ValueTypeNames.String).Set("shared mutation parameters")
    material = _add_material(stage, "/World/Looks/MutationFeature", (0.42, 0.48, 0.52))
    parts_scope = UsdGeom.Xform.Define(stage, "/World/parts")
    for index, part in enumerate(task["parts"]):
        path = f"/World/parts/{_slug(part)}"
        # Keep source-authored xformOps inside a child reference.  The wrapper owns
        # the benchmark placement and therefore never collides with source ops.
        wrapper = UsdGeom.Xform.Define(stage, path)
        prim = UsdGeom.Xform.Define(stage, f"{path}/Asset").GetPrim()
        source = repo_root / config["source_assets"][part]["visual"]
        rel = Path(__import__("os").path.relpath(source, out_path.parent))
        prim.GetReferences().AddReference(str(rel))
        wrapper.AddTranslateOp().Set((0.22 * index, 0.0, 0.02))
        wrapper.AddScaleOp().Set((0.001, 0.001, 0.001))
    _fixture(stage, task, variant, material)
    stage.GetRootLayer().Save()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "scenes" / "generated")
    args = parser.parse_args()
    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        raise SystemExit("invalid config:\n- " + "\n- ".join(errors))
    repo_root = Path(__file__).resolve().parents[2]
    for task in config["tasks"]:
        for variant in ("HARD", "COMMUTABLE"):
            out = args.out_dir / f"{task['task_id'].lower()}_{variant.lower()}.usda"
            build_one(config, task, variant, out, repo_root)
    print(f"wrote 10 USD fixture scenes to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
