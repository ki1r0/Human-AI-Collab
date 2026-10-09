#!/usr/bin/env python3
"""Isolated collision-on Hub/Casing diagnostic; no runtime part-pose writes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def ray_hits(triangles, origin, direction):
    """Positive ray/triangle crossings, with shared-edge duplicates collapsed."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(det) > 1e-12
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
    delta = origin - triangles[:, 0]
    u = inv * np.einsum("ij,ij->i", delta, h)
    q = np.cross(delta, edge1)
    v = inv * (q @ direction)
    distance = inv * np.einsum("ij,ij->i", edge2, q)
    values = np.sort(distance[valid & (u >= -1e-7) & (v >= -1e-7)
                              & (u+v <= 1.0000001) & (distance > 1e-7)])
    return values[np.r_[True, np.diff(values) > 2e-6]].tolist() if len(values) else []


def section_circles(audit):
    section = next(s for s in audit["cross_sections"]
                   if abs(s["cover_root_z_m"]-.062) < 1e-6 and abs(s["slice_z_m"]-.052) < 1e-6)
    fits = {}
    for part, boundary in (("cover", max), ("casing", min)):
        points = np.array([[boundary(r[part+"_radii_m"])*math.cos(r["angle_rad"]),
                            boundary(r[part+"_radii_m"])*math.sin(r["angle_rad"])]
                           for r in section["rays"]])
        coefficients = np.linalg.lstsq(np.column_stack((2*points, np.ones(len(points)))),
                                       np.sum(points*points, axis=1), rcond=None)[0]
        center = coefficients[:2]
        radius = math.sqrt(coefficients[2]+center@center)
        fits[part] = {"center_offset_from_socket_m": center.tolist(), "radius_m": radius,
                      "fit_rms_m": float(np.sqrt(np.mean((np.linalg.norm(points-center, axis=1)-radius)**2)))}
    fits["relative_center_correction_m"] = (np.array(fits["casing"]["center_offset_from_socket_m"])
                                           -fits["cover"]["center_offset_from_socket_m"]).tolist()
    fits["radial_clearance_m"] = fits["casing"]["radius_m"]-fits["cover"]["radius_m"]
    fits["method"] = "Least-squares circle on 16 input-mesh rays at casing-relative z=52 mm; cover root z=62 mm."
    return fits


def mesh_triangles(stage, root_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    triangles, details = [], []
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        attrs = {attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
                 if attr.GetName().startswith(("physics:", "physxRigidBody:",
                                               "physxCollision:", "physxSDFMeshCollision:"))}
        if attrs:
            details.append({"path": str(prim.GetPath()), "applied_schemas": prim.GetAppliedSchemas(),
                            "attributes": attrs})
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        points = (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError(f"non-triangular collision mesh: {prim.GetPath()}")
        triangles.append(points[indices.reshape(-1, 3)])
    return np.concatenate(triangles), details


def geometry_audit(stage, config):
    from pxr import Gf, Usd, UsdGeom
    cover, cover_settings = mesh_triangles(stage, "/World/Cover")
    casing, casing_settings = mesh_triangles(stage, "/World/Casing")
    cover -= np.asarray(config["preinsert_xyz_m"])
    casing -= np.asarray(config["casing_xyz_m"])
    socket_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath("/World/Casing/socket_hub_output"))
    plug_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath("/World/Cover/plug_main"))
    socket = np.asarray(socket_matrix.GetRow3(3)) - config["casing_xyz_m"]
    plug = np.asarray(plug_matrix.GetRow3(3)) - config["preinsert_xyz_m"]
    frames_aligned_root = socket - plug
    profiles = []
    for root_z in (config["nominal_relative_xyz_m"][2], float(frames_aligned_root[2])):
        seated = cover + [0, config["nominal_relative_xyz_m"][1], root_z]
        for z in np.arange(0.046, 0.0601, 0.002):
            sections = []
            for tri in (seated, casing):
                selected = tri[(tri[:, :, 2].min(axis=1) <= z)
                               & (tri[:, :, 2].max(axis=1) >= z)]
                sections.append(selected)
            clearances, rays = [], []
            for angle in np.linspace(0.017, 2*math.pi+0.017, 16, endpoint=False):
                direction = np.array([math.cos(angle), math.sin(angle), 0.0])
                hits = [ray_hits(tri, np.array([socket[0], socket[1], z]), direction)
                        if len(tri) else [] for tri in sections]
                nearby = [[radius for radius in group if radius < 0.16] for group in hits]
                gap = min(nearby[1]) - max(nearby[0]) if all(nearby) else None
                if gap is not None:
                    clearances.append(gap)
                rays.append({"angle_rad": float(angle), "cover_radii_m": nearby[0],
                             "casing_radii_m": nearby[1], "radial_gap_m": gap})
            profiles.append({"cover_root_z_m": root_z, "slice_z_m": float(z),
                             "min_sampled_radial_gap_m": min(clearances) if clearances else None,
                             "rays": rays})
    source_metadata = []
    for path in config["asset_paths"].values():
        source = Usd.Stage.Open(path)
        source_metadata.append({"path": path, "meters_per_unit": UsdGeom.GetStageMetersPerUnit(source),
                                "up_axis": str(UsdGeom.GetStageUpAxis(source)),
                                "collision_meshes": [
                                    {"path": str(p.GetPath()), "schemas": p.GetAppliedSchemas(),
                                     "approximation": str(p.GetAttribute("physics:approximation").Get())}
                                    for p in source.Traverse() if p.IsA(UsdGeom.Mesh)]})
    return {"source_metadata": source_metadata, "spawn_scale": 0.002,
            "insertion_axis_world": [0, 0, -1], "cover_thin_axis_local": [0, 1, 0],
            "socket_relative_xyz_m": socket.tolist(), "plug_offset_world_m": plug.tolist(),
            "socket_frame_world_matrix": np.asarray(socket_matrix).tolist(),
            "plug_frame_world_matrix_at_preinsert": np.asarray(plug_matrix).tolist(),
            "frame_coincident_root_relative_xyz_m": frames_aligned_root.tolist(),
            "nominal_root_relative_xyz_m": config["nominal_relative_xyz_m"],
            "frame_vs_nominal_axial_difference_m": config["nominal_relative_xyz_m"][2]-frames_aligned_root[2],
            "cover_settings": cover_settings, "casing_settings": casing_settings,
            "cross_sections": profiles,
            "clearance_caveat": "Triangle sections audit collider input meshes; cooked SDF clearance is measured by dynamic contacts."}


def self_test():
    tri = np.array([[[1., -1., -1.], [1., 1., -1.], [1., 0., 1.]]])
    assert np.allclose(ray_hits(tri, np.zeros(3), np.array([1., 0., 0.])), [1.])
    assert ray_hits(tri, np.zeros(3), np.array([-1., 0., 0.])) == []
    assert ray_hits(np.repeat(tri, 2, axis=0), np.zeros(3), np.array([1., 0., 0.])) == [1.]
    result = summarize({"casing_xyz_m": [0, 0, 0], "nominal_relative_xyz_m": [0, 0, 0],
                        "cover_quat_wxyz": [1, 0, 0, 0], "preinsert_xyz_m": [0, 0, 1], "dt_s": .1,
                        "acceptance": {"axial_m": .01, "radial_m": .01, "tilt_deg": 2,
                                       "penetration_m": .001, "settle_speed_mps": .01,
                                       "settle_window_s": .1, "min_descent_m": .1}},
                       [{"cover_root_state": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], "contact_count": 1,
                         "sim_time_s": .1, "min_separation_m": 0, "contact_force_world_N": [0, 0, 1]}], 1, 0.)
    json.dumps(result)  # Result flags must be JSON-native, not numpy.bool_.
    print("geometry self-test PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--collision-mode", choices=("authored", "scene_sdf"), default="authored")
    parser.add_argument("--preinsert-from-geometry", type=Path)
    parser.add_argument("--no-scene-video", action="store_true",
                        help="Run physics/trace only; live Replicator capture is currently blocked")
    known, _ = parser.parse_known_args()
    if known.self_test:
        self_test()
        return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error("--output-dir required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    import faulthandler
    stack_log = (args.output_dir/"python_stacks.log").open("w")
    faulthandler.dump_traceback_later(60, repeat=True, file=stack_log)
    try:
        physical_test(args)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        stack_log.close()
        app.close()


def physical_test(args):
    import imageio.v2 as imageio
    import torch
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg

    output = args.output_dir
    repo = Path(__file__).resolve().parents[1]
    config = {"stage": "1_physical", "role": "isolated_feasibility_not_policy_search",
              "asset_paths": {"cover": str(repo/"assets/parts/Hub Cover Output.usd"),
                              "casing": str(repo/"assets/parts/Casing Top.usd")},
              "casing_xyz_m": [0., 0., 0.2], "preinsert_xyz_m": [0., 0.08683718, 0.302],
              "nominal_relative_xyz_m": [0., 0.08683718, 0.062],
              "cover_quat_wxyz": [0., 0., 2**-0.5, 2**-0.5],
              "dt_s": 1/240, "duration_s": 32., "capture_hz": 10,
              "cover_mass_kg": 5.7, "gravity_mps2": 9.81,
              "collision_mode": args.collision_mode,
              "linear_speed_cap_mps": 0.002, "angular_speed_cap_radps": 0.05,
              "contact_offset_m": 0.0001, "rest_offset_m": -0.00005,
              "pose_writes_during_test": 0, "collisions_enabled": True,
              "render_sync": "Fabric forward + Replicator delta_time=0 capture; no renderer pose commands",
              "scene_video_requested": not args.no_scene_video,
              "fixture": "kinematic Casing; dynamic Cover under gravity with a velocity cap; no robot or grasp",
              "acceptance": {"axial_m": 0.008, "radial_m": 0.004, "tilt_deg": 2.,
                             "penetration_m": 0.001, "settle_speed_mps": 0.01,
                             "settle_window_s": 0.5, "min_descent_m": 0.025}}
    if args.collision_mode == "scene_sdf":
        config.update({"linear_speed_cap_mps": 0.1, "angular_speed_cap_radps": 0.05,
                       "fixture": "kinematic Casing; dynamic Cover; force-only descent, then full-gravity support test; no robot or grasp",
                       "drive": "force-only vertical velocity servo for 26 s, then 6 s full-gravity load; no lateral or angular drive",
                       "drive_speed_mps": 0.002, "drive_net_force_cap_N": 20.,
                       "drive_duration_s": 26., "sdf_resolution": 256,
                       "sdf_margin": 0., "sdf_narrow_band_thickness": 0.})
    if args.preinsert_from_geometry:
        fit = section_circles(json.loads(args.preinsert_from_geometry.read_text()))
        config["geometric_preinsert_source"] = str(args.preinsert_from_geometry)
        config["section_circle_fit"] = fit
        for coordinate, correction in enumerate(fit["relative_center_correction_m"]):
            config["preinsert_xyz_m"][coordinate] += correction
            config["nominal_relative_xyz_m"][coordinate] += correction
        config["acceptance"]["axial_m"] = 0.001
        config["acceptance"]["radial_m"] = 0.001
        config["pretest_pose_initialization_writes"] = 1
    (output/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=config["dt_s"], device=args.device, use_fabric=True, render_interval=24))
    stage = omni.usd.get_context().get_stage()
    objects = {}
    for role in ("casing", "cover"):
        kinematic = role == "casing"
        body_cfg = RigidObjectCfg(
            prim_path=f"/World/{role.title()}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=config["asset_paths"][role], scale=(0.002,)*3,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.32, 0.48, 0.64) if kinematic else (0.85, 0.44, 0.12)),
                mass_props=sim_utils.MassPropertiesCfg(mass=15. if kinematic else config["cover_mass_kg"]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=kinematic, disable_gravity=kinematic,
                    max_linear_velocity=100. if kinematic else config["linear_speed_cap_mps"],
                    max_angular_velocity=100. if kinematic else config["angular_speed_cap_radps"],
                    max_depenetration_velocity=0.2,
                    solver_position_iteration_count=64, solver_velocity_iteration_count=16),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=config["contact_offset_m"], rest_offset=config["rest_offset_m"]),
                activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(config["casing_xyz_m"] if kinematic else config["preinsert_xyz_m"]),
                rot=(1., 0., 0., 0.) if kinematic else tuple(config["cover_quat_wxyz"])))
        objects[role] = RigidObject(body_cfg)
    if args.collision_mode == "scene_sdf":
        # Match existing M1 scene setup, only in the composed test layer.
        for role, body in objects.items():
            for prim in Usd.PrimRange(stage.GetPrimAtPath(body.cfg.prim_path)):
                if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
                    "sdf" if role == "cover" else "none")
                if role == "cover":
                    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                    sdf.CreateSdfResolutionAttr().Set(config["sdf_resolution"])
                    sdf.CreateSdfMarginAttr().Set(config["sdf_margin"])
                    sdf.CreateSdfNarrowBandThicknessAttr().Set(config["sdf_narrow_band_thickness"])
    contact = ContactSensor(ContactSensorCfg(
        prim_path="/World/Cover", filter_prim_paths_expr=["/World/Casing"],
        track_contact_points=True, max_contact_data_count_per_prim=1024, update_period=0.))
    sim_utils.CuboidCfg(size=(0.65, 0.65, 0.05),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.19, 0.24, 0.30))).func(
            "/World/FixturePedestal", sim_utils.CuboidCfg(size=(0.65, 0.65, 0.05)),
            translation=(0., 0., 0.118))
    sim_utils.DomeLightCfg(intensity=1300.).func("/World/Light", sim_utils.DomeLightCfg(intensity=1300.))
    annotators = []
    print("diagnostic: creating two cameras", flush=True)
    for eye in (() if args.no_scene_video else ((0.42, 0.45, 0.52), (0., 0.0795, 0.85))):
        camera = rep.create.camera(position=eye, look_at=(0., 0.0795, 0.25),
                                   clipping_range=(0.001, 10.))
        product = rep.create.render_product(camera, (640, 480))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        annotators.append(annotator)
    print("diagnostic: auditing composed collider geometry", flush=True)
    audit = geometry_audit(stage, config)
    audit["circle_fit_at_legacy_nominal"] = section_circles(audit)
    (output/"geometry_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    stage.Flatten().Export(str(output/"initial_scene.usda"))
    print("diagnostic: resetting physics", flush=True)
    sim.reset()
    for body in objects.values():
        body.update(config["dt_s"])
    (output/"runtime_dynamics.json").write_text(json.dumps({
        role: {"mass_kg": body.root_physx_view.get_masses().cpu().tolist(),
               "inertia_kg_m2": body.root_physx_view.get_inertias().cpu().tolist()}
        for role, body in objects.items()}, indent=2)+"\n")
    writer = None if args.no_scene_video else imageio.get_writer(
        str(output/"physical_insertion.mp4"), fps=config["capture_hz"],
        codec="libx264", pixelformat="yuv420p", macro_block_size=None)
    frames, rows, first_contact = 0, [], None
    render_pose_delta = 0.

    def capture(label=None):
        nonlocal frames, render_pose_delta
        if writer is None:
            return
        before = objects["cover"].root_physx_view.get_transforms().clone()
        sim.forward()
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=1)
        data = [np.asarray(a.get_data())[:, :, :3].copy() for a in annotators]
        after = objects["cover"].root_physx_view.get_transforms()
        render_pose_delta = max(render_pose_delta, float(torch.max(torch.abs(after-before)).item()))
        frame = np.concatenate(data, axis=1)
        writer.append_data(frame)
        frames += 1
        if label:
            imageio.imwrite(output/f"{label}.png", frame)

    try:
        for _ in range(4):
            sim.render()
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        sim.render()  # Process pause while Isaac's render() explicitly suppresses physics.
        if args.preinsert_from_geometry:
            # Isaac reset warms up physics. Restore the declared initial condition
            # once before t=0; never write pose or velocity in the test loop.
            body = objects["cover"]
            body.write_root_pose_to_sim(torch.tensor([config["preinsert_xyz_m"]+config["cover_quat_wxyz"]],
                                                     device=args.device))
            body.write_root_velocity_to_sim(torch.zeros((1, 6), device=args.device))
            body.update(config["dt_s"])
        (output/"initial_state.json").write_text(json.dumps({
            role: body.data.root_state_w[0].detach().cpu().tolist()
            for role, body in objects.items()}, indent=2)+"\n")
        capture("preinsertion")
        # Timeline play is asynchronous. Process it without physics, otherwise
        # SimulationContext.step() resumes via app.update() and advances an
        # unlogged rendering interval (0.1 s here) before the first driven tick.
        sim.set_setting("/app/player/playSimulations", False)
        timeline.play()
        sim.app.update()
        sim.set_setting("/app/player/playSimulations", True)
        assert sim.is_playing(), "timeline resume was not processed"
        initial_timeline_time = timeline.get_current_time()
        with (output/"trace.jsonl").open("w") as stream:
            for tick in range(round(config["duration_s"]/config["dt_s"])):
                external_z = 0.
                if args.collision_mode == "scene_sdf" and tick*config["dt_s"] < config["drive_duration_s"]:
                    vz = float(objects["cover"].data.root_state_w[0, 9].item())
                    net_force = np.clip(config["cover_mass_kg"] * (-config["drive_speed_mps"]-vz)
                                        / config["dt_s"], -config["drive_net_force_cap_N"],
                                        config["drive_net_force_cap_N"])
                    external_z = float(config["cover_mass_kg"]*config["gravity_mps2"] + net_force)
                objects["cover"].set_external_force_and_torque(
                    torch.tensor([[[0., 0., external_z]]], device=args.device),
                    torch.zeros((1, 1, 3), device=args.device), is_global=True)
                objects["cover"].write_data_to_sim()
                sim.step(render=False)
                for body in objects.values():
                    body.update(config["dt_s"])
                contact.update(config["dt_s"])
                state = objects["cover"].data.root_state_w[0].detach().cpu().tolist()
                force = contact.data.force_matrix_w[0, 0, 0].detach().cpu().tolist()
                raw = contact.contact_physx_view.get_contact_data(dt=config["dt_s"])
                count = int(raw[4].reshape(-1)[0].item())
                start = int(raw[5].reshape(-1)[0].item())
                distances = raw[3].reshape(-1)[start:start+count]
                separation = float(distances.min().item()) if count else None
                row = {"tick": tick+1, "sim_time_s": (tick+1)*config["dt_s"],
                       "cover_root_state": state, "casing_root_state": objects["casing"].data.root_state_w[0].detach().cpu().tolist(),
                       "contact_force_world_N": force, "contact_count": count,
                       "applied_force_world_N": [0., 0., external_z],
                       "min_separation_m": separation}
                target = np.asarray(config["casing_xyz_m"])+config["nominal_relative_xyz_m"]
                row.update({"insertion_depth_m": config["preinsert_xyz_m"][2]-state[2],
                            "axial_target_error_m": state[2]-float(target[2]),
                            "radial_alignment_error_m": float(np.linalg.norm(np.asarray(state[:2])-target[:2])),
                            "actuator_enabled": tick*config["dt_s"] < config.get("drive_duration_s", 0),
                            "penetration_m": max(0., -separation) if separation is not None else 0.})
                row["timeline_elapsed_s"] = timeline.get_current_time()-initial_timeline_time
                if args.preinsert_from_geometry and tick == 0:
                    assert abs(row["insertion_depth_m"]) < 0.00005 and abs(state[9]) < .003, (
                        "initialization advanced unlogged physics", row)
                if count and (tick+1) % 24 == 0:
                    row["contact_points_m"] = raw[1].reshape(-1, 3)[start:start+count].detach().cpu().tolist()
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                rows.append(row)
                if count and first_contact is None:
                    first_contact = row["sim_time_s"]
                    capture("first_contact")
                elif tick == round(config.get("drive_duration_s", -1)/config["dt_s"]):
                    capture("actuator_removed")
                elif (tick+1) % 24 == 0:
                    capture()
                if (tick+1) % 1200 == 0:
                    print(json.dumps({"time_s": row["sim_time_s"], "cover_xyz_m": state[:3],
                                      "force_N": float(np.linalg.norm(force)), "contact_count": count}), flush=True)
        capture("final_state")
    finally:
        if writer is not None:
            writer.close()
    result = summarize(config, rows, frames, render_pose_delta)
    (output/"physical_state.json").write_text(json.dumps(rows[-1], indent=2)+"\n")
    (output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    # Fabric owns rendered dynamic transforms; physical_state.json is the final
    # state. Do not label an unchanged USD stage export as a final physical scene.
    print(json.dumps(result, indent=2), flush=True)


def summarize(config, rows, frames, render_pose_delta):
    final = rows[-1]
    first_contact = next((r["sim_time_s"] for r in rows if r["contact_count"]), None)
    nominal = np.asarray(config["casing_xyz_m"])+config["nominal_relative_xyz_m"]
    position = np.asarray(final["cover_root_state"][:3])
    quat = np.asarray(final["cover_root_state"][3:7])
    angle = math.degrees(2*math.acos(np.clip(abs(quat@config["cover_quat_wxyz"]), 0., 1.)))
    errors = {"axial_m": abs(float(position[2]-nominal[2])),
              "radial_m": float(np.linalg.norm(position[:2]-nominal[:2])), "tilt_deg": angle}
    tail = rows[-round(config["acceptance"]["settle_window_s"]/config["dt_s"]):]
    penetration = max([max(0., -row["min_separation_m"]) for row in rows
                       if row["min_separation_m"] is not None] or [0.])
    stable = all(np.linalg.norm(row["cover_root_state"][7:10]) <= config["acceptance"]["settle_speed_mps"]
                 and row["contact_count"] > 0 for row in tail)
    descent = config["preinsert_xyz_m"][2]-position[2]
    checks = {key: value <= config["acceptance"][key] for key, value in errors.items()}
    checks.update({"contact_and_settle": stable, "penetration": penetration <= config["acceptance"]["penetration_m"],
                   "descent": bool(descent >= config["acceptance"]["min_descent_m"]),
                   "preinsert_contact_free": rows[0]["contact_count"] == 0,
                   "render_did_not_advance_body": (render_pose_delta is not None and render_pose_delta < 1e-7)
                   if config.get("scene_video_requested", True) else None})
    passed = all(value for value in checks.values() if value is not None)
    result = {"stage": 1, "status": "PASS" if passed else "FAIL", "checks": checks,
              "errors": errors, "descent_m": float(descent), "max_contact_penetration_m": penetration,
              "first_contact_time_s": first_contact, "render_pose_delta": render_pose_delta,
              "peak_contact_force_N": max(float(np.linalg.norm(r["contact_force_world_N"])) for r in rows),
              "peak_linear_speed_mps": max(float(np.linalg.norm(r["cover_root_state"][7:10])) for r in rows),
              "final_physical_state": final, "video_frames": frames,
              "classification": "PHYSICS_SEATED_AT_NOMINAL_PRIOR" if passed else "GEOMETRY_OR_CONTACT_BLOCKER",
              "next_stage": "deterministic_controller" if passed else "BLOCKED_by_stage_1"}
    result["final_window_peak_linear_speed_mps"] = max(float(np.linalg.norm(r["cover_root_state"][7:10])) for r in tail)
    result["final_window_peak_angular_speed_radps"] = max(float(np.linalg.norm(r["cover_root_state"][10:13])) for r in tail if len(r["cover_root_state"]) >= 13) if len(final["cover_root_state"]) >= 13 else None
    result["final_window_position_range_m"] = np.ptp(np.array([r["cover_root_state"][:3] for r in tail]), axis=0).tolist()
    if not stable and all(checks[k] for k in ("axial_m", "radial_m", "tilt_deg")):
        result["classification"] = "CONTACT_INSTABILITY_AT_SEATED_POSE"
    return result


if __name__ == "__main__":
    main()
