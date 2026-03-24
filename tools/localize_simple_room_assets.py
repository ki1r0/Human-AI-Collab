#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import REPO_ROOT, ensure_pxr_paths

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from pxr import Usd  # noqa: E402
except Exception:
    ensure_pxr_paths()
    from pxr import Usd  # noqa: E402

from rc_paths import ASSETS_DIR  # noqa: E402
from rc_scene_setup import apply_local_scene_fallbacks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror remote simple-room assets locally and optionally save a localized stage.")
    parser.add_argument(
        "--stage",
        default=str(ASSETS_DIR / "simple_room_scene.usd"),
        help="Stage to inspect and localize.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the stage after replacing remote references with local mirrored assets.",
    )
    args = parser.parse_args()

    stage_path = Path(args.stage).expanduser().resolve()
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open stage: {stage_path}")

    result = apply_local_scene_fallbacks(stage, logger=print)
    print(f"[LOCALIZE] result={result}")
    if args.save:
        stage.GetRootLayer().Save()
        print(f"[LOCALIZE] saved {stage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
