"""Print authored R1 gripper collision prims and local bounds inside Isaac Sim."""

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
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if "left_gripper_link" not in path or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            box = cache.ComputeLocalBound(prim).GetBox()
            lo, hi = box.GetMin(), box.GetMax()
            approx = ""
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
            print({
                "path": path,
                "type": prim.GetTypeName(),
                "min": [float(lo[i]) for i in range(3)],
                "max": [float(hi[i]) for i in range(3)],
                "approximation": approx,
            }, flush=True)
        return 0
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
