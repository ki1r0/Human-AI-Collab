#!/usr/bin/env python3
"""Render an Isaac scene replay of a previously recorded Hub/Casing PhysX trace."""

import argparse
import json
from pathlib import Path

from isaacsim.simulation_app import SimulationApp


APP = SimulationApp({"headless": True})

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics, UsdLux  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    trace_path = run_dir / "trace.jsonl"
    config_path = run_dir / "configuration.json"
    video_path = run_dir / "physics_trace_scene_replay.mp4"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    config = json.loads(config_path.read_text())
    start = {"sim_time_s": 0.0,
             "cover_root_state": config["preinsert_xyz_m"] + config["cover_quat_wxyz"] + [0.] * 6,
             "actuator_enabled": True}
    indices = set(range(0, len(rows), 24))
    indices.update((len(rows) - 1,
                    next(i for i, row in enumerate(rows) if row["contact_count"]),
                    next(i for i, row in enumerate(rows) if not row["actuator_enabled"])))
    replay = [start] + [rows[i] for i in sorted(indices)]

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    repo = Path(__file__).resolve().parents[1]
    source_assets = {
        "cover": repo / "assets/parts/Hub Cover Output.usd",
        "casing": repo / "assets/parts/Casing Top.usd",
    }

    def add_part(name, xyz, quat):
        root = UsdGeom.Xform.Define(stage, f"/World/{name}")
        root.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(quat[0], Gf.Vec3d(*quat[1:4])))
        root.AddScaleOp().Set(Gf.Vec3f(0.002, 0.002, 0.002))
        prim = root.GetPrim()
        prim.GetReferences().AddReference(str(source_assets[name.lower()]))
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.GetKinematicEnabledAttr().Set(True)
        body.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        return root

    cover = add_part("Cover", config["preinsert_xyz_m"], config["cover_quat_wxyz"])
    casing = add_part("Casing", config["casing_xyz_m"], [1., 0., 0., 0.])
    pedestal = UsdGeom.Cube.Define(stage, "/World/FixturePedestal")
    pedestal.CreateSizeAttr().Set(1.0)
    pedestal.AddTranslateOp().Set(Gf.Vec3d(0., 0., 0.118))
    pedestal.AddScaleOp().Set(Gf.Vec3d(0.65, 0.65, 0.05))
    pedestal.CreateDisplayColorAttr().Set([Gf.Vec3f(0.19, 0.24, 0.30)])
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr().Set(1300.)
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(cover).GetOrderedXformOps()}
    translate = ops["xformOp:translate"]
    orient = ops["xformOp:orient"]
    body = UsdPhysics.RigidBodyAPI(cover)
    body.GetKinematicEnabledAttr().Set(True)
    body.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    omni.timeline.get_timeline_interface().stop()
    omni.physx.get_physx_interface().reset_simulation()

    camera = rep.create.camera(position=(0.42, 0.45, 0.52),
                               look_at=(0.0, 0.0795, 0.25),
                               clipping_range=(0.001, 2.0))
    product = rep.create.render_product(camera, (args.width, args.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
        small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = small = ImageFont.load_default()

    out = imageio.get_writer(str(video_path), fps=args.fps, codec="libx264",
                             quality=8, pixelformat="yuv420p", macro_block_size=None,
                             ffmpeg_log_level="error")
    rendered = 0
    try:
        for item in replay:
            state = item["cover_root_state"]
            translate.Set(Gf.Vec3d(*state[:3]))
            orient.Set(Gf.Quatd(state[3], Gf.Vec3d(*state[4:7])))
            omni.physx.get_physx_interface().update_transformations(True, True, True)
            rep.orchestrator.step(rt_subframes=1)
            data = rgb.get_data()
            if data is None:
                raise RuntimeError("Isaac RGB annotator returned no frame")
            frame = np.asarray(data[:, :, :3], dtype=np.uint8).copy()
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            timestamp = float(item["sim_time_s"])
            draw.rectangle((0, 0, args.width, 62), fill=(12, 20, 30))
            draw.text((16, 7), "HUB COVER PHYSX TRACE - ISAAC SCENE REPLAY", font=font, fill="white")
            draw.text((18, 38),
                      f"t={timestamp:05.2f}s | actuator {'ON' if item.get('actuator_enabled') else 'OFF'}"
                      " | recorded states, no physics stepping during video",
                      font=small, fill=(255, 210, 105))
            out.append_data(np.asarray(image))
            rendered += 1
    finally:
        out.close()

    metadata = {
        "type": "isaac_rendered_replay_of_recorded_physx_trace_not_live_simulation",
        "video": str(video_path), "source_trace": str(trace_path),
        "source_assets": {key: str(value) for key, value in source_assets.items()},
        "source_trace_rows": len(rows), "rendered_frames": rendered, "fps": args.fps,
        "physics_duration_s": float(rows[-1]["sim_time_s"]),
        "live_physics_used_as_video_source": False,
        "scene_method": "Isaac render stage reconstructed from original USD assets; part transforms are the exact recorded trace samples",
        "runtime_pose_writes": "video-only replay of recorded trace; none in source physical test",
        "sampling": "10 Hz samples plus exact first-contact, actuator-off, and final states",
    }
    (args.run_dir / "physics_trace_scene_replay.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        APP.close()
