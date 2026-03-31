#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import REPO_ROOT, ensure_pxr_paths

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.paths import ASSETS_DIR  # noqa: E402
from runtime.scene_setup import apply_local_scene_fallbacks  # noqa: E402


def _open_stage_with_pxr(stage_path: Path):
    try:
        from pxr import Usd  # type: ignore
    except Exception:
        ensure_pxr_paths()
        from pxr import Usd  # type: ignore
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open stage with pxr: {stage_path}")
    return stage, None


def _open_stage_with_simulation_app(stage_path: Path):
    from isaacsim.simulation_app import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.usd

        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(stage_path)):
            raise RuntimeError(f"Failed to open stage with omni.usd: {stage_path}")
        for _ in range(60):
            app.update()
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError(f"Stage unavailable after opening with SimulationApp: {stage_path}")
        return stage, app
    except Exception:
        app.close()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply release-scene localization and hygiene fixes, then optionally save the stage."
    )
    parser.add_argument(
        "--stage",
        default=str(ASSETS_DIR / "simple_room_scene.usd"),
        help="Stage to inspect and localize.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the stage after applying local asset fallbacks and rigid-body cleanup.",
    )
    args = parser.parse_args()

    stage_path = Path(args.stage).expanduser().resolve()
    app = None
    try:
        try:
            stage, app = _open_stage_with_pxr(stage_path)
            print("[LOCALIZE] opened stage with pxr")
        except Exception as exc:
            print(f"[LOCALIZE] pxr open failed ({exc}); retrying via SimulationApp")
            stage, app = _open_stage_with_simulation_app(stage_path)
            print("[LOCALIZE] opened stage with SimulationApp")

        result = apply_local_scene_fallbacks(stage, logger=print)
        print(f"[LOCALIZE] result={result}")
        if args.save:
            stage.GetRootLayer().Save()
            print(f"[LOCALIZE] saved {stage_path}")
        return 0
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
