"""Inspect reset-time world bounds for the Hub bore and R1 gripper links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdGeom  # noqa: E402

from hrc_m1.roco_env import make_env_classes  # noqa: E402


def main() -> int:
    cfg_cls, env_cls = make_env_classes()
    env = env_cls(cfg_cls())
    try:
        stage = env.sim.get_initial_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        roots = [
            "/World/envs/env_0/Hub_Cover_Output_Top",
            "/World/envs/env_0/Robot/left_gripper_link1",
            "/World/envs/env_0/Robot/left_gripper_link2",
        ]
        result: dict[str, object] = {}
        for root_path in roots:
            prim = stage.GetPrimAtPath(root_path)
            entries = []
            for child in Usd.PrimRange(prim):
                if not child.IsA(UsdGeom.Mesh):
                    continue
                box = cache.ComputeWorldBound(child).ComputeAlignedBox()
                entries.append({
                    "path": str(child.GetPath()),
                    "min": [float(v) for v in box.GetMin()],
                    "max": [float(v) for v in box.GetMax()],
                })
            result[root_path] = entries
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
