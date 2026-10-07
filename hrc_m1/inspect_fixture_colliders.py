"""Print spawned fixture/Hub collision prims and world-space bounds in Isaac."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from hrc_m1.roco_env import make_env_classes  # noqa: E402


def main() -> int:
    cfg_cls, env_cls = make_env_classes()
    env = env_cls(cfg_cls())
    try:
        stage = env.sim.get_initial_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"],
            useExtentsHint=True,
        )
        for name in ("Table", "Casing_Top", "Hub_Cover_Output_Top"):
            root = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                bound = cache.ComputeWorldBound(prim).GetBox()
                lo, hi = bound.GetMin(), bound.GetMax()
                local_to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                approx = ""
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    approx = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
                print({
                    "body": name,
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "world_min": [float(lo[i]) for i in range(3)],
                    "world_max": [float(hi[i]) for i in range(3)],
                    "local_to_world_rows": [
                        [float(local_to_world.GetRow(r)[c]) for c in range(4)]
                        for r in range(4)
                    ],
                    "approximation": approx,
                }, flush=True)
        return 0
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
