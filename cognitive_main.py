"""Minimal cognitive loop demo (no UI).

Runs a physics-step callback at 60 Hz:
- Captures the latest camera frame each step.
- Every 60 steps (1 Hz), triggers the VLM worker with the current frame.
- Logs belief updates in the main thread (no USD ghost visualization).

Usage:
  ./isaaclab.sh -p nvidia_tutorial/cognitive_main.py --usd nvidia_tutorial/simple_room_scene.usd
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
from collections import deque
from typing import Any, Dict, Sequence

from isaaclab.app import AppLauncher

from rc_belief_manager import BeliefManager
from rc_cognitive_worker import cognitive_worker
from rc_config import TABLE_PRIM_PATH


def _point_in_aabb(p: Sequence[float], aabb_min: Sequence[float], aabb_max: Sequence[float]) -> bool:
    return (
        aabb_min[0] <= p[0] <= aabb_max[0]
        and aabb_min[1] <= p[1] <= aabb_max[1]
        and aabb_min[2] <= p[2] <= aabb_max[2]
    )


def _extract_orange_pos(snapshot: Dict[str, Any]) -> Sequence[float] | None:
    if not isinstance(snapshot, dict):
        return None
    orange = None
    if isinstance(snapshot.get("objects"), dict):
        orange = snapshot["objects"].get("orange")
    if orange is None:
        orange = snapshot.get("orange")
    if not isinstance(orange, dict):
        return None
    for k in ("estimated_pos_3d", "position", "pos", "location"):
        v = orange.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return v
    return None


def _update_robot_target(target: Sequence[float]) -> None:
    # TODO: Hook this into your robot controller.
    print(f"[INFO] Robot target updated to: {list(target)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cognitive loop demo using SimulationContext.")
    parser.add_argument(
        "--usd",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "simple_room_scene.usd"),
        help="Path to the USD stage to open.",
    )
    parser.add_argument(
        "--table_prim",
        type=str,
        default=TABLE_PRIM_PATH,
        help="USD prim path for the table used in the safety overlap check.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # This script relies on camera rendering / Replicator, so ensure camera extensions are enabled.
    if hasattr(args, "enable_cameras") and not args.enable_cameras:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import Isaac/Omni modules only after the app is running.
    import omni.timeline
    import omni.usd
    import json
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext

    from rc_camera import init_camera, get_latest_rgb_uint8
    from rc_ghost_visualizer import GhostVisualizer
    from rc_config import CAPTURE_FPS, INQUIRY_FRAME_COUNT

    def _compute_aabb(prim_path: str):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        aabb = bbox.ComputeAlignedRange()
        return aabb.GetMin(), aabb.GetMax()

    # Load stage.
    usd_path = os.path.abspath(args.usd)
    print(f"[INFO] Opening stage: {usd_path}")
    sim_utils.open_stage(usd_path)
    # Keep the scene static on startup; user must click toolbar PLAY.
    # NOTE: `timeline.stop()` would hide the PLAY button in Isaac Lab's toolbar via AppLauncher hooks.
    try:
        omni.timeline.get_timeline_interface().pause()
    except Exception:
        pass
    # Ensure PLAY remains visible for interactive GUI runs.
    try:
        import omni.kit.widget.toolbar

        toolbar = omni.kit.widget.toolbar.get_instance()
        play_button_group = toolbar._builtin_tools._play_button_group  # type: ignore[attr-defined]
        if play_button_group is not None:
            play_btn = play_button_group._play_button  # type: ignore[attr-defined]
            play_btn.visible = True
            play_btn.enabled = True
    except Exception:
        pass

    # Setup simulation.
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device=getattr(args, "device", "cpu"))
    sim = SimulationContext(sim_cfg)

    # Enable RTX sensors if available (helps camera pipelines).
    try:
        sim.set_setting("/isaaclab/render/rtx_sensors", True)
    except Exception:
        pass

    sim.reset()
    # Reset may warm up render/physics; force PAUSE again for a static initial state.
    try:
        omni.timeline.get_timeline_interface().pause()
    except Exception:
        pass
    print("[INFO] Simulation ready.")

    # Initialize camera + prims.
    init_camera()
    belief_manager = BeliefManager(initial_state={"objects": {"orange": {"belief_status": "unknown", "stale": False}}})
    ghost_visualizer = GhostVisualizer(logger=print)

    # Worker thread and queues.
    from rc_long_term_memory import LongTermMemory
    from rc_short_term_memory import ShortTermMemory

    short_memory = ShortTermMemory(ttl_sec=6.0, logger=print)
    long_memory = LongTermMemory(logger=print)

    image_queue: queue.Queue = queue.Queue(maxsize=2)
    command_queue: queue.Queue = queue.Queue()
    worker = threading.Thread(
        target=cognitive_worker,
        args=(belief_manager, short_memory, long_memory, image_queue, command_queue),
        name="cognitive_worker",
        daemon=False,
    )
    worker.start()

    # Precompute table AABB (static scene assumption).
    table_aabb = _compute_aabb(args.table_prim)
    if table_aabb is None:
        print(f"[WARN] Table prim not found or has no bounds: {args.table_prim}. Safety check disabled.")

    sim_dt = sim.get_physics_dt()
    capture_stride = max(1, int(round(1.0 / (max(1e-6, sim_dt) * CAPTURE_FPS))))
    batch_frames: deque = deque(maxlen=INQUIRY_FRAME_COUNT)

    step_count = 0

    def physics_step_cb(event):
        nonlocal step_count, table_aabb, batch_frames
        step_count += 1

        # Capture at CAPTURE_FPS (e.g. 5 Hz -> every ~12 physics steps at 60 Hz).
        if step_count % capture_stride == 0:
            frame = get_latest_rgb_uint8()
            if frame is not None:
                batch_frames.append(frame)

        # Trigger inference every 60 physics steps (1 Hz at 60 Hz sim).
        # Send the latest 5-frame trajectory (oldest -> newest).
        if step_count % 60 == 0 and len(batch_frames) >= INQUIRY_FRAME_COUNT:
            try:
                frames = list(batch_frames)
                image_queue.put_nowait((frames, {"type": "periodic_1hz", "step": step_count}))
                print(f"[INFO] Triggered VLM at step={step_count} (frames={len(frames)})")
            except queue.Full:
                # Drop if worker is still busy; keep sim real-time.
                pass

        # Consume worker outputs (non-blocking).
        while True:
            try:
                msg = command_queue.get_nowait()
            except queue.Empty:
                break

            status = msg.get("status")
            if status != "Done":
                print(f"[WARN] Worker status={status}: {msg}")
                continue

            snapshot = belief_manager.get_snapshot()
            if getattr(ghost_visualizer, "enabled", False):
                ghost_visualizer.sync_ghosts(snapshot)
                print(f"[INFO] Belief update (ghost=on): {json.dumps(snapshot)}")
            else:
                print(f"[INFO] Belief update (ghost=off): {json.dumps(snapshot)}")

            # Also log the direct worker update for debugging.
            belief_update = msg.get("belief_update")
            if belief_update is not None:
                print(f"[INFO] Worker belief_update: {json.dumps(belief_update)}")

            pos = _extract_orange_pos(snapshot)
            if pos is None:
                continue

            # Safety overlap check against table.
            if table_aabb is not None:
                aabb_min, aabb_max = table_aabb
                if _point_in_aabb(pos, aabb_min, aabb_max):
                    print("[WARN] Ghost position overlaps with table AABB. Skipping target update.")
                    continue

            _update_robot_target(pos)

    sim.add_physics_callback("cognitive_loop", physics_step_cb)

    try:
        timeline = omni.timeline.get_timeline_interface()
        # Ensure PAUSE before entering the main loop.
        try:
            timeline.pause()
        except Exception:
            pass

        while simulation_app.is_running():
            # In interactive GUI workflows, only advance physics when the timeline is PLAYing.
            if timeline.is_playing():
                sim.step()
            else:
                # Keep UI responsive without advancing physics.
                sim.render()
    finally:
        # Shutdown worker thread.
        try:
            image_queue.put_nowait(None)
        except Exception:
            pass
        worker.join(timeout=5.0)
        simulation_app.close()


if __name__ == "__main__":
    main()
