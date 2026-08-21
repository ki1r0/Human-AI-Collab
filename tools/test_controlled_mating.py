#!/usr/bin/env python3
"""Controlled collision-on mating checks for real gearbox assets."""

from __future__ import annotations

import os
import math
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


TRIALS = (
    Trial("Transfer_Gear_to_Transfer_Shaft", "Transfer Gear", "Transfer Shaft",
          (0.145, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 90.0, 0.0), 0, -0.04, 0.0442, 0.006),
    Trial("Output_Gear_to_Output_Shaft", "Output Gear", "Output Shaft",
          (0.0, 0.22, 0.6), (0.0, 0.0, 0.6), (-90.0, 0.0, 0.0), 1, -0.08, 0.0310, 0.008),
    Trial("Breather_Plug_to_Casing_Base", "Breather Plug", "Casing Base",
          (0.0, 2.33, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 0.0), 1, -0.025, 2.2899, 0.003),
    Trial("M6_Hub_Bolt_to_Casing_Top", "M6 Hub Bolt", "Casing Top",
          (2.08, 0.0, 0.11), (2.0, 0.0, 0.0), (90.0, 0.0, 0.0), 2, -0.02, 0.0584, 0.006),
)


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


def main() -> int:
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
    for index, trial in enumerate(TRIALS):
        target_path = f"/World/trial_{index}/target"
        source_path = f"/World/trial_{index}/source"
        UsdGeom.Xform.Define(stage, f"/World/trial_{index}")
        add_asset(stage, target_path, trial.target, trial.target_pos, (0.0, 0.0, 0.0), True)
        source = add_asset(stage, source_path, trial.source, trial.source_pos, trial.source_rot, False)
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
    physx = omni.physx.get_physx_interface()
    physx.start_simulation()
    time_step = 1.0 / 240.0
    for step in range(960):
        physx.update_simulation(time_step, step * time_step)
        physx.update_transformations(True, True, True)

    failures = 0
    for trial, body, start in zip(TRIALS, bodies, initial):
        final = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetRow3(3)
        axial_error = abs(float(final[trial.axis]) - trial.expected)
        lateral_axes = [axis for axis in range(3) if axis != trial.axis]
        lateral_drift = math.sqrt(sum((float(final[axis]) - float(start[axis])) ** 2 for axis in lateral_axes))
        passed = axial_error <= trial.tolerance and lateral_drift <= 0.004
        failures += not passed
        print(
            f"PHYSICS {'PASS' if passed else 'FAIL'} {trial.name}: "
            f"final={tuple(round(float(v), 5) for v in final)}, "
            f"axial_error={axial_error:.5f}m, lateral_drift={lateral_drift:.5f}m"
        )
    physx.reset_simulation()
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
