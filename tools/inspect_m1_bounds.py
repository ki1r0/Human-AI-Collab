"""Print unscaled USD bounds for the M1 cover and casing inside Isaac Kit."""
from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdGeom  # noqa: E402

for path in ("assets/parts/Hub Cover Output.usd", "assets/parts/Casing Top.usd"):
    stage = Usd.Stage.Open(path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    prim = stage.GetDefaultPrim()
    bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    print(path, "default", prim.GetPath(), "min", bound.GetMin(), "max", bound.GetMax(), "size", bound.GetSize(), "mpu", UsdGeom.GetStageMetersPerUnit(stage), flush=True)
    for mesh in stage.Traverse():
        if mesh.IsA(UsdGeom.Mesh):
            mesh_bound = cache.ComputeWorldBound(mesh).ComputeAlignedRange()
            print(" mesh", mesh.GetPath(), "min", mesh_bound.GetMin(), "max", mesh_bound.GetMax(), "size", mesh_bound.GetSize(), flush=True)
app.close()
