"""Primary entry point for the Human-AI-Collab demo.

Standalone example:
  ./isaaclab.sh -p /absolute/path/to/Human-AI-Collab/main.py \
      --usd /absolute/path/to/Human-AI-Collab/assets/simple_room_scene.usd

Script Editor example:
  execute this file from Isaac Sim after adding the repo root to sys.path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rc_paths import ASSETS_DIR, PROJECT_ROOT, load_runtime_env_defaults


load_runtime_env_defaults()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

simulation_app = None


def _apply_scene_fallbacks() -> None:
    try:
        import omni.usd

        from rc_scene_setup import apply_local_scene_fallbacks

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        result = apply_local_scene_fallbacks(stage, logger=print)
        if result.get("changed"):
            print(
                f"[ASSET] Applied {result['changed']} local scene fallback(s); skipped {result.get('skipped', 0)} already-resolved prim(s).",
                flush=True,
            )
    except Exception as exc:
        print(f"[WARN] Scene fallback setup skipped: {exc}", flush=True)


try:
    import omni.ui  # noqa: F401
except ModuleNotFoundError:
    from isaaclab.app import AppLauncher
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usd",
        type=str,
        default=str(ASSETS_DIR / "simple_room_scene.usd"),
        help="Optional: USD stage to open on startup (standalone mode only).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    def _append_kit_arg(raw: str, item: str) -> str:
        raw = (raw or "").strip()
        if item in raw:
            return raw
        return f"{raw} {item}".strip() if raw else item

    safe_kit_args = args.kit_args if hasattr(args, "kit_args") else ""
    safe_kit_args = _append_kit_arg(safe_kit_args, "--/renderer/multiGpu/autoEnable=0")
    safe_kit_args = _append_kit_arg(safe_kit_args, "--/renderer/multiGpu/enabled=0")
    safe_kit_args = _append_kit_arg(safe_kit_args, "--/renderer/multiGpu/maxGpuCount=1")

    active_gpu = os.getenv("ISAAC_ACTIVE_GPU", "").strip()
    if not active_gpu:
        dev = str(getattr(args, "device", "") or "")
        if dev.startswith("cuda:"):
            active_gpu = dev.split(":", 1)[1].strip()
    if active_gpu.isdigit():
        safe_kit_args = _append_kit_arg(safe_kit_args, f"--/renderer/activeGpu={active_gpu}")

    args.kit_args = safe_kit_args

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import omni.timeline

        omni.timeline.get_timeline_interface().stop()
    except Exception:
        pass

    try:
        import isaaclab.sim as sim_utils

        if args.usd:
            sim_utils.open_stage(os.path.abspath(args.usd))
            _apply_scene_fallbacks()
            try:
                import omni.timeline

                omni.timeline.get_timeline_interface().stop()
            except Exception:
                pass
    except Exception as exc:
        print(f"[WARN] Failed to open USD stage: {exc}", flush=True)

_PROJECT_PACKAGES = ("rc_ui", "rc_config", "rc_state", "rc_log", "agent", "sensor", "belief", "memory", "control")
for _name in list(sys.modules.keys()):
    if any(_name == pkg or _name.startswith(pkg + ".") for pkg in _PROJECT_PACKAGES):
        sys.modules.pop(_name, None)

import rc_ui

_apply_scene_fallbacks()
rc_ui.run()

if simulation_app is not None:
    try:
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()
