#!/usr/bin/env python3
"""Record a collision-on PhysX rollout of the validated assembly interfaces.

This is deliberately an evidence recorder rather than a policy/environment
implementation.  Each configured interface is placed in its own grid cell,
the source part is driven along its insertion axis, and the target is held
fixed.  The same 17 interface trials used by ``test_controlled_mating.py`` are
run in one Isaac Sim stage so the resulting MP4 shows real rendered geometry
while the accompanying JSON records the numerical pass/fail results.

Run inside the Isaac container, for example::

    tools/run_tool.sh tools/record_physics_rollout.py \
        --config assembly/physics_validation.json \
        --out validation_logs/assembly_physics_rollout.mp4

The video is intentionally labelled as an interface-rollout recording.  It
does not claim that a robot policy performed the canonical 73-step sequence.
The canonical sequence coverage is checked separately by the physics manifest
report and the Magic Assembly playback test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaacsim.simulation_app import SimulationApp


APP = SimulationApp({"headless": True})

import imageio.v2 as imageio  # noqa: E402
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
PARTS = REPO / "assets" / "parts"
SCALE = (0.002, 0.002, 0.002)


def _load_trials(path: Path) -> tuple[dict, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(payload["trials"])


def _add_asset(stage, path: str, asset: str, position, rotation, kinematic: bool):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
    xform.AddScaleOp().Set(Gf.Vec3f(*SCALE))
    prim = xform.GetPrim()
    prim.GetReferences().AddReference(str(PARTS / f"{asset}.usd"))
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.GetKinematicEnabledAttr().Set(kinematic)
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


def _camera(stage, width: int, height: int, focus=None):
    # The grid occupies roughly x/y=[0, 4.8] metres.  A high oblique camera
    # keeps all independent mating cells in view while preserving depth cues.
    if focus is None:
        camera_position = (2.4, 2.4, 8.4)
        camera_target = (2.4, 2.4, 0.0)
    else:
        camera_position = (float(focus[0]) + 0.65, float(focus[1]) + 0.65, 0.95)
        camera_target = (float(focus[0]), float(focus[1]), 0.03)
    camera = rep.create.camera(position=camera_position, look_at=camera_target)
    product = rep.create.render_product(camera, (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])

    dome = UsdLux.DomeLight.Define(stage, "/World/rollout_dome")
    dome.GetIntensityAttr().Set(700.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/rollout_key")
    distant.GetIntensityAttr().Set(1200.0)
    distant.GetAngleAttr().Set(25.0)
    distant.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, -25.0))
    return annotator


def _read_frame(annotator):
    data = annotator.get_data()
    if data is None:
        return None
    frame = data[:, :, :3] if data.ndim == 3 and data.shape[2] == 4 else data
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    return frame.astype("uint8", copy=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "assembly" / "physics_validation.json"))
    parser.add_argument("--only", help="record one named trial as a close-up")
    parser.add_argument("--out", default=str(REPO / "validation_logs" / "assembly_physics_rollout.mp4"))
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--steps", type=int, default=None,
                        help="physics steps; defaults to the config value")
    parser.add_argument("--capture-every", type=int, default=8,
                        help="capture one rendered frame every N physics steps")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args(argv)
    if args.capture_every < 1:
        parser.error("--capture-every must be positive")

    config_path = Path(args.config)
    all_trials = _load_trials(config_path)
    trials = tuple(trial for trial in all_trials
                   if not args.only or trial["name"] == args.only)
    if not trials:
        parser.error(f"unknown trial: {args.only!r}")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    steps = int(args.steps if args.steps is not None else config_payload.get("steps", 960))
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
    starts = []
    for index, trial in enumerate(trials):
        offset = (float(index % 4) * 1.2, float(index // 4) * 1.2, 0.0)
        target_pos = tuple(float(trial["target_pos"][i]) + offset[i] for i in range(3))
        source_pos = tuple(float(trial["source_pos"][i]) + offset[i] for i in range(3))
        trial_root = f"/World/trial_{index}"
        UsdGeom.Xform.Define(stage, trial_root)
        _add_asset(stage, f"{trial_root}/target", trial["target"], target_pos,
                   trial.get("target_rot", (0.0, 0.0, 0.0)), True)
        source = _add_asset(stage, f"{trial_root}/source", trial["source"], source_pos,
                            trial["source_rot"], False)
        velocity = [0.0, 0.0, 0.0]
        velocity[int(trial["axis"])] = float(trial["velocity"])
        UsdPhysics.RigidBodyAPI(source).GetVelocityAttr().Set(Gf.Vec3f(*velocity))
        bodies.append(source)
        # source_pos is already in world coordinates (including the grid
        # offset); keep it as the baseline for lateral-drift measurement.
        starts.append(tuple(source_pos))

    focus = None
    if args.only:
        focus = trials[0]["target_pos"]
    annotator = _camera(stage, args.width, args.height, focus=focus)
    APP.update()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_out) if args.metrics_out else out_path.with_suffix(".json")
    physx = omni.physx.get_physx_interface()
    # Match the controlled-mating harness: one Kit update before starting
    # PhysX lets the USD references/camera resolve without advancing the
    # manually stepped bodies.  We do not render during this run because a
    # Replicator step advances PhysX in this Kit build.
    APP.update()
    physx.start_simulation()
    dt = 1.0 / 240.0
    trajectory = [[
        tuple(float(v) for v in UsdGeom.Xformable(body)
              .ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetRow3(3))
        for body in bodies
    ]]
    for step in range(steps):
        physx.update_simulation(dt, step * dt)
        physx.update_transformations(True, True, True)
        if (step + 1) % args.capture_every == 0 or step == steps - 1:
            trajectory.append([
                tuple(float(v) for v in UsdGeom.Xformable(body)
                      .ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetRow3(3))
                for body in bodies
            ])

    results = []
    for index, (trial, body, start) in enumerate(zip(trials, bodies, starts)):
        final = trajectory[-1][index]
        offset = (float(index % 4) * 1.2, float(index // 4) * 1.2, 0.0)
        axis = int(trial["axis"])
        expected_world = float(trial["expected"]) + offset[axis]
        axial_error = abs(float(final[axis]) - expected_world)
        lateral_axes = [a for a in range(3) if a != axis]
        lateral_drift = math.sqrt(sum((float(final[a]) - start[a]) ** 2
                                      for a in lateral_axes))
        passed = axial_error <= float(trial["tolerance"]) and lateral_drift <= 0.004
        results.append({
            "name": trial["name"], "passed": bool(passed),
            "final": [float(v) for v in final],
            "axial_error_m": axial_error, "lateral_drift_m": lateral_drift,
            "canonical_steps": trial.get("canonical_steps", []),
        })
    # Stop/reset before rendering.  During replay the source bodies are made
    # kinematic and are placed at the poses captured above.  This preserves
    # the exact physical trajectory while preventing the renderer from
    # injecting extra simulation steps.
    physx.reset_simulation()
    for body in bodies:
        UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Set(True)
        UsdPhysics.RigidBodyAPI(body).GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    writer = imageio.get_writer(
        str(out_path), fps=args.fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=None, ffmpeg_log_level="error",
    )
    captured = 0
    for poses in trajectory:
        for body, pose in zip(bodies, poses):
            ops = UsdGeom.Xformable(body).GetOrderedXformOps()
            if not ops:
                raise RuntimeError(f"no xform ops on {body.GetPath()}")
            ops[0].Set(Gf.Vec3d(*pose))
        physx.update_transformations(True, True, True)
        rep.orchestrator.step(rt_subframes=1)
        frame = _read_frame(annotator)
        if frame is not None:
            writer.append_data(frame)
            captured += 1
    writer.close()

    payload = {
        "kind": "collision_on_interface_rollout_video",
        "config": str(config_path), "video": str(out_path),
        "only": args.only,
        "video_frames": captured, "fps": args.fps,
        "physics_steps": steps, "capture_every": args.capture_every,
        "trajectory_samples": len(trajectory),
        "results": results,
        "all_pass": all(item["passed"] for item in results),
        "interpretation": "Rendered replay of a recorded collision-on PhysX trajectory for simultaneous isolated interface trials; not a robot-policy rollout or a full sequential 73-step dynamic assembly.",
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_pass"] and captured > 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
