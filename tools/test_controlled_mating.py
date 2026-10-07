#!/usr/bin/env python3
"""Controlled collision-on mating checks for real gearbox assets."""

from __future__ import annotations

import os
import math
import argparse
import json
from dataclasses import dataclass

from isaacsim.simulation_app import SimulationApp


APP = SimulationApp({"headless": True})

import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(REPO, "assets", "parts")
SCALE = (0.002, 0.002, 0.002)


@dataclass
class Trial:
    name: str
    source: str
    target: str
    source_pos: tuple[float, float, float]
    target_pos: tuple[float, float, float]
    source_rot: tuple[float, float, float]
    axis: int
    velocity: float
    expected: float
    tolerance: float
    target_rot: tuple[float, float, float] = (0.0, 0.0, 0.0)


TRIALS = (
    Trial("Transfer_Gear_to_Transfer_Shaft", "Transfer Gear", "Transfer Shaft",
          (0.145, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 90.0, 0.0), 0, -0.04, 0.0442, 0.006),
    Trial("Output_Gear_to_Output_Shaft", "Output Gear", "Output Shaft",
          (0.0, 0.22, 0.6), (0.0, 0.0, 0.6), (-90.0, 0.0, 0.0), 1, -0.08, 0.0310, 0.008),
    Trial("Breather_Plug_to_Casing_Base", "Breather Plug", "Casing Base",
          (0.0, 2.33, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 0.0), 1, -0.025, 2.2899, 0.003),
    Trial("M6_Hub_Bolt_to_Casing_Top", "M6 Hub Bolt", "Casing Top",
          # The Magic Assembly child-local z=29.2 is multiplied by the
          # scene's 0.002 part scale.  With the casing root at z=0, the
          # seated bolt root is z=0.0428 (head lower face at z=0.0558).
          (2.08, 0.0, 0.11), (2.0, 0.0, 0.0), (90.0, 0.0, 0.0), 2, -0.02, 0.0428, 0.006),
)


def load_trials(path: str) -> tuple[Trial, ...]:
    global _CONFIG_TRIAL_ITEMS
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    _CONFIG_TRIAL_ITEMS = tuple(payload["trials"])
    return tuple(
        Trial(
            name=item["name"],
            source=item["source"],
            target=item["target"],
            source_pos=tuple(item["source_pos"]),
            target_pos=tuple(item["target_pos"]),
            source_rot=tuple(item["source_rot"]),
            target_rot=tuple(item.get("target_rot", (0.0, 0.0, 0.0))),
            axis=int(item["axis"]),
            velocity=float(item["velocity"]),
            expected=float(item["expected"]),
            tolerance=float(item["tolerance"]),
        )
        for item in payload["trials"]
    )


# The casing uses the same repaired M6 annulus at each numbered socket, but the
# centers are intentionally non-uniform.  Expanding the manifest below lets us
# exercise every canonical fastener step at its own measured hole center rather
# than treating one representative bolt as proof for all 24 placements.
_M6_SOCKET_CENTERS = {
    # These are the centers used by repair_casing_m6_bores.py.  The source
    # mesh's authored socket labels are close, but a few are offset by enough
    # to consume the sub-millimeter M6 clearance after the repair.
    1: (40.0003, 0.1324), 2: (-40.0005, 0.1320),
    3: (40.0019, 80.1334), 4: (-39.9974, 80.1322),
    5: (54.0013, -26.6663), 6: (-53.9979, -26.6680),
    8: (7.9998, -26.6677), 9: (54.1121, -72.4371),
    10: (7.9994, -72.6674), 11: (-7.9994, -26.6672),
    13: (-7.9995, -72.6672), 14: (-54.0159, -72.2708),
}
# Canonical IDs name the twelve hub bolts by ordinal (01..12), while the
# physical fixture's socket labels skip 07 and 12.  Keep that authored order
# explicit so per-ID expansion never indexes a non-existent socket.
_M6_SOCKET_ORDER = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14)
_M10_SOCKET_CENTERS = {
    1: (60.0, 120.0), 2: (-60.0, 120.0), 3: (60.0, -125.0),
    4: (-60.0, -125.0), 5: (95.0, 0.0), 6: (-95.0, 0.0),
}
_M10_TOP_EXPECTED_Z = {
    # The negative-Y mounting pockets open into a shallower interior shelf in
    # the source Casing Top mesh.  Their collision-on rest pose is therefore
    # -71.5 asset units (-0.143 m), while the other four use -86 units.
    3: -0.143,
    4: -0.143,
}


def expand_canonical_trials(trials: tuple[Trial, ...]) -> tuple[Trial, ...]:
    """Clone grouped entries so every canonical combine ID gets a run."""
    expanded: list[Trial] = []
    for trial in trials:
        # The JSON loader attaches no IDs to Trial; grouped names are paired
        # with the same ordered IDs used by physics_validation.json below.
        # Keep the built-in/fixture-only trials unchanged.
        ids = next((item.get("canonical_steps", []) for item in _CONFIG_TRIAL_ITEMS
                    if item.get("name") == trial.name), [])
        if not ids:
            expanded.append(trial)
            continue
        for step_id in ids:
            source_pos = trial.source_pos
            expected = trial.expected
            if "m6_hub_bolt" in step_id:
                ordinal = int(step_id.split("_")[-2])
                number = _M6_SOCKET_ORDER[ordinal - 1]
                cx, cy = _M6_SOCKET_CENTERS[number]
                source_pos = (cx * 0.002, cy * 0.002, trial.source_pos[2])
            elif "m10_casing_bolt" in step_id:
                number = int(step_id.split("_")[-1])
                cx, cy = _M10_SOCKET_CENTERS[number]
                source_pos = (cx * 0.002, cy * 0.002, trial.source_pos[2])
                expected = _M10_TOP_EXPECTED_Z.get(number, expected)
            expanded.append(Trial(
                name=f"{trial.name}__{step_id}", source=trial.source,
                target=trial.target, source_pos=source_pos,
                target_pos=trial.target_pos, source_rot=trial.source_rot,
                target_rot=trial.target_rot, axis=trial.axis,
                velocity=trial.velocity, expected=expected,
                tolerance=trial.tolerance,
            ))
    return tuple(expanded)


# Populated by load_trials so expansion can retain the exact authored IDs
# without adding a second config format.
_CONFIG_TRIAL_ITEMS: tuple[dict, ...] = ()


def add_asset(stage, path: str, asset: str, position, rotation, kinematic: bool):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
    xform.AddScaleOp().Set(Gf.Vec3f(*SCALE))
    prim = xform.GetPrim()
    prim.GetReferences().AddReference(os.path.join(PARTS, f"{asset}.usd"))
    UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(kinematic)
    for child in Usd.PrimRange(prim):
        if not child.IsA(UsdGeom.Mesh) or not child.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision = PhysxSchema.PhysxCollisionAPI.Apply(child)
        collision.GetContactOffsetAttr().Set(0.0001)
        collision.GetRestOffsetAttr().Set(-0.00005)
        if UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Get() == "sdf":
            sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(child)
            sdf.GetSdfResolutionAttr().Set(512 if asset.startswith("Casing ") else 256)
    return prim


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run one named trial")
    parser.add_argument("--config", help="JSON trial config (overrides built-in baseline)")
    parser.add_argument("--expand-canonical", action="store_true",
                        help="run one isolated trial for every canonical_steps ID")
    parser.add_argument("--start-index", type=int, default=0,
                        help="slice expanded trials from this zero-based index")
    parser.add_argument("--max-trials", type=int,
                        help="limit the number of trials after slicing")
    parser.add_argument("--steps", type=int, default=960,
                        help="simulation steps (default: 960 at 240 Hz)")
    args = parser.parse_args(argv)
    configured = load_trials(args.config) if args.config else TRIALS
    if args.expand_canonical and args.config:
        configured = expand_canonical_trials(configured)
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    configured = configured[args.start_index:]
    if args.max_trials is not None:
        configured = configured[:args.max_trials]
    trials = tuple(t for t in configured if not args.only or t.name == args.only)
    if not trials:
        print(f"ERROR unknown trial {args.only!r}")
        return 2
    context = omni.usd.get_context()
    context.new_stage()
    APP.update()
    stage = context.get_stage()
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    physics = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    physics.GetGravityMagnitudeAttr().Set(0.0)

    bodies = []
    for index, trial in enumerate(trials):
        # Keep independent interface trials from colliding with each other.
        # Four columns leave more than twice the largest casing diameter.
        scene_offset = (float(index % 4) * 1.2, float(index // 4) * 1.2, 0.0)
        target_path = f"/World/trial_{index}/target"
        source_path = f"/World/trial_{index}/source"
        UsdGeom.Xform.Define(stage, f"/World/trial_{index}")
        target_pos = tuple(float(trial.target_pos[i]) + scene_offset[i] for i in range(3))
        source_pos = tuple(float(trial.source_pos[i]) + scene_offset[i] for i in range(3))
        add_asset(stage, target_path, trial.target, target_pos, trial.target_rot, True)
        source = add_asset(stage, source_path, trial.source, source_pos, trial.source_rot, False)
        velocity = [0.0, 0.0, 0.0]
        velocity[trial.axis] = trial.velocity
        UsdPhysics.RigidBodyAPI(source).GetVelocityAttr().Set(Gf.Vec3f(*velocity))
        bodies.append(source)

        colliders = [child for child in Usd.PrimRange(source)
                     if child.IsA(UsdGeom.Mesh) and child.HasAPI(UsdPhysics.CollisionAPI)]
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
        details = []
        for collider in colliders:
            size = cache.ComputeWorldBound(collider).ComputeAlignedRange().GetSize()
            details.append((str(collider.GetPath()), tuple(round(float(v), 4) for v in size),
                            UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()))
        print(f"SETUP {trial.name}: {details}")

    APP.update()
    initial = [UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetRow3(3) for body in bodies]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for trial, body in zip(trials, bodies):
        bounds = bbox_cache.ComputeWorldBound(body).ComputeAlignedRange()
        print(f"POSE {trial.name}: min={tuple(round(float(v), 5) for v in bounds.GetMin())} "
              f"max={tuple(round(float(v), 5) for v in bounds.GetMax())}")
    physx = omni.physx.get_physx_interface()
    physx.start_simulation()
    time_step = 1.0 / 240.0
    for step in range(args.steps):
        physx.update_simulation(time_step, step * time_step)
        physx.update_transformations(True, True, True)

    failures = 0
    for index, (trial, body, start) in enumerate(zip(trials, bodies, initial)):
        final = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetRow3(3)
        bounds = bbox_cache.ComputeWorldBound(body).ComputeAlignedRange()
        scene_offset = (float(index % 4) * 1.2,
                        float(index // 4) * 1.2, 0.0)
        expected_world = trial.expected + scene_offset[trial.axis]
        axial_error = abs(float(final[trial.axis]) - expected_world)
        lateral_axes = [axis for axis in range(3) if axis != trial.axis]
        lateral_drift = math.sqrt(sum((float(final[axis]) - float(start[axis])) ** 2 for axis in lateral_axes))
        passed = axial_error <= trial.tolerance and lateral_drift <= 0.004
        failures += not passed
        print(
            f"PHYSICS {'PASS' if passed else 'FAIL'} {trial.name}: "
            f"final={tuple(round(float(v), 5) for v in final)}, "
            f"axial_error={axial_error:.5f}m, lateral_drift={lateral_drift:.5f}m"
        )
        print(f"  FINAL_BBOX {trial.name}: min={tuple(round(float(v), 5) for v in bounds.GetMin())} "
              f"max={tuple(round(float(v), 5) for v in bounds.GetMax())}")
    physx.reset_simulation()
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
