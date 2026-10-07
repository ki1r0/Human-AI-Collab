#!/usr/bin/env python3
"""Print scene prim paths, world transforms, and source bounds for key parts."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths
ensure_pxr_paths()
from pxr import Usd, UsdGeom  # noqa: E402


def main():
    scene = os.path.join(_REPO, "assets", "simple_room_scene.usd")
    stage = Usd.Stage.Open(scene)
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if prim.GetName() not in {
            "Casing_Base", "Casing_Top", "M6_Hub_Bolt_01_base",
            "M6_Hub_Bolt_01_top", "Hub_Cover_Input", "Hub_Cover_Input_Top",
        }:
            continue
        xf = UsdGeom.Xformable(prim)
        world = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        row = world.GetRow3(3)
        print("PRIM", prim.GetPath(), "name", prim.GetName(),
              "world_t", tuple(round(float(v), 6) for v in row))
        for child in prim.GetChildren():
            if child.GetName().startswith(("socket_bolt_hub_", "socket_bolt_casing_",
                                           "socket_nut_casing_", "socket_hub_", "socket_gear",
                                           "socket_casing_mate", "socket_oil", "socket_breather",
                                           "plug_")):
                cxf = UsdGeom.Xformable(child)
                cworld = cxf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                crow = cworld.GetRow3(3)
                print("  CHILD", child.GetName(),
                      tuple(round(float(v), 6) for v in crow))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
