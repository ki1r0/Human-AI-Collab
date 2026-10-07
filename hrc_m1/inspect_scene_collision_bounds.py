"""Inspect world-space bounds of the M1 fixture and all R1 collision prims.

This is a read-only Isaac diagnostic.  It does not step the simulation or
modify USD.  The output is used to distinguish a real assembly obstruction
from an initial Robot↔Table overlap caused by the calibration fixture.
"""

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

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from hrc_m1.roco_env import make_env_classes  # noqa: E402


def _bounds(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> dict[str, object]:
    bound = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    lo, hi = bound.GetMin(), bound.GetMax()
    return {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "min": [float(lo[i]) for i in range(3)],
        "max": [float(hi[i]) for i in range(3)],
    }


def _overlap(a: dict[str, object], b: dict[str, object]) -> bool:
    amin, amax = a["min"], a["max"]
    bmin, bmax = b["min"], b["max"]
    return all(float(amin[i]) <= float(bmax[i]) and float(bmin[i]) <= float(amax[i]) for i in range(3))


def main() -> int:
    cfg_cls, env_cls = make_env_classes()
    env = env_cls(cfg_cls())
    try:
        stage = env.sim.get_initial_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
            useExtentsHint=True,
        )
        roots = {
            "table": "/World/envs/env_0/Table",
            "casing": "/World/envs/env_0/Casing_Top",
            "hub": "/World/envs/env_0/Hub_Cover_Output_Top",
            "robot": "/World/envs/env_0/Robot",
        }
        result: dict[str, object] = {"collision_prims": {}}
        for name, root_path in roots.items():
            root = stage.GetPrimAtPath(root_path)
            rows = []
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    rows.append(_bounds(cache, prim))
            result["collision_prims"][name] = rows

        table = result["collision_prims"]["table"]
        robot = result["collision_prims"]["robot"]
        result["robot_table_aabb_overlaps"] = [
            {"robot": row["path"], "table": table_row["path"]}
            for row in robot
            for table_row in table
            if _overlap(row, table_row)
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
