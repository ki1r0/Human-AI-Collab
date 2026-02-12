#!/usr/bin/env python3
"""Headless verification that timeline PLAY advances simulation time.

This script launches Isaac Lab app headless, opens the simple-room USD,
installs rc_ui timeline hooks, triggers PLAY, waits ~2s, and asserts the
timeline time increased.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


THIS_DIR = os.path.dirname(__file__)
TUTORIAL_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(TUTORIAL_DIR, ".."))

for p in (TUTORIAL_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.append(p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usd",
        type=str,
        default=os.path.join(TUTORIAL_DIR, "simple_room_scene.usd"),
        help="USD stage path",
    )
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    # Build launcher args exactly as Isaac Lab expects.
    launch_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launch_parser)
    launch_args = launch_parser.parse_args([])
    launch_args.headless = True
    launch_args.enable_cameras = True
    launch_args.device = "cpu"

    app_launcher = AppLauncher(launch_args)
    simulation_app = app_launcher.app

    try:
        import isaaclab.sim as sim_utils
        import omni.kit.app
        import omni.timeline

        sim_utils.open_stage(os.path.abspath(args.usd))

        # Let stage settle.
        for _ in range(30):
            simulation_app.update()

        import rc_ui

        # Install the same hooks used by the main app.
        rc_ui._configure_quiet_kit_logging()
        rc_ui._ensure_asset_browser_cache_dir()
        rc_ui._install_startup_autostop_hooks()
        rc_ui._install_timeline_run_hooks()

        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()

        for _ in range(10):
            simulation_app.update()

        t0 = float(timeline.get_current_time())
        rc_ui.on_play()

        deadline = time.time() + float(args.seconds)
        while time.time() < deadline:
            simulation_app.update()

        t1 = float(timeline.get_current_time())

        if t1 <= t0 + 1e-6:
            print(f"FAIL: timeline did not advance (t0={t0:.6f}, t1={t1:.6f})")
            return 1

        print(f"SUCCESS: Simulation is ticking (t0={t0:.6f}, t1={t1:.6f}, dt={t1 - t0:.6f})")
        return 0
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
