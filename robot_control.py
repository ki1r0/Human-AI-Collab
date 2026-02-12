import os
import sys

# Allow importing sibling modules even when loaded by Kit.
_THIS_DIR = os.path.dirname(__file__)
# Ensure absolute path is present even when Script Editor runs a temp copy.
_ABS_DIR = "/workspace/IsaacLab/nvidia_tutorial"
for _p in (_THIS_DIR, _ABS_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

simulation_app = None  # set only in standalone (when we launch Kit ourselves)

try:
    import omni.ui  # noqa: F401
except ModuleNotFoundError:
    # Launch Kit when executed standalone.
    repo_root = os.path.abspath(os.path.join(_THIS_DIR, ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from isaaclab.app import AppLauncher
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usd",
        type=str,
        default=os.path.join(_THIS_DIR, "simple_room_scene.usd"),
        help="Optional: USD stage to open on startup (standalone mode only).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Force safer rendering defaults for this demo to avoid OOM from multi-GPU render graph behavior.
    # Users can still override by passing explicit --kit_args on CLI.
    def _append_kit_arg(raw: str, item: str) -> str:
        raw = (raw or "").strip()
        if item in raw:
            return raw
        return f"{raw} {item}".strip() if raw else item

    safe_kit_args = args.kit_args if hasattr(args, "kit_args") else ""
    safe_kit_args = _append_kit_arg(safe_kit_args, "--/renderer/multiGpu/autoEnable=0")
    safe_kit_args = _append_kit_arg(safe_kit_args, "--/renderer/multiGpu/enabled=0")
    safe_kit_args = _append_kit_arg(safe_kit_args, "--/renderer/multiGpu/maxGpuCount=1")

    # Best-effort render GPU pinning:
    # 1) explicit env override ISAAC_ACTIVE_GPU
    # 2) infer from --device cuda:N
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

    # Make sure the timeline is PAUSED on startup so the scene is static until the user clicks PLAY.
    # NOTE: `timeline.stop()` would hide the PLAY button in Isaac Lab's toolbar via AppLauncher hooks.
    try:
        import omni.timeline

        _tl = omni.timeline.get_timeline_interface()
        _tl.pause()
    except Exception:
        pass

    # Open the requested stage after Kit is running.
    try:
        import isaaclab.sim as sim_utils

        if args.usd:
            sim_utils.open_stage(os.path.abspath(args.usd))
            # Some configs/extensions may auto-play after a stage load. Force PAUSE again.
            try:
                import omni.timeline

                omni.timeline.get_timeline_interface().pause()
            except Exception:
                pass
    except Exception as exc:
        print(f"[WARN] Failed to open USD stage: {exc}")

# Reset local rc_* modules when re-running from Script Editor.
# Drop modules first, then import once to keep a single shared STATE instance.
for _name in list(sys.modules.keys()):
    if _name == "rc_ui" or _name.startswith("rc_"):
        sys.modules.pop(_name, None)

import rc_ui

rc_ui.run()

# If we launched Kit ourselves, keep the app alive until the user closes it.
# (When running from Script Editor, Kit is already running and we must return.)
if simulation_app is not None:
    try:
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()
