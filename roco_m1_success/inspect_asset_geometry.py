#!/usr/bin/env python3
"""Print the authored local bounds and transforms of the RoCo gearbox assets."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

from pxr import Usd, UsdGeom  # noqa: E402


def main() -> None:
    root = Path(args.asset_dir)
    print(f"M1_ASSET_ROOT={root} exists={root.exists()}", flush=True)
    for filename in (
        "planetary_reducer_3x_scale.usd",
        "sun_planetary_gear_3x_scale.usd",
        "planetary_carrier_3x_scale.usd",
        "OakTableLarge.usd",
    ):
        path = root / ("Gearbox" if "gear" in filename or "reducer" in filename or "carrier" in filename else "Props") / filename
        print(f"M1_ASSET_CHECK={path} exists={path.exists()}", flush=True)
        if not path.exists():
            print(f"M1_ASSET_MISSING={path}", flush=True)
            continue
        stage = Usd.Stage.Open(str(path))
        print(f"M1_ASSET_OPENED={stage is not None}", flush=True)
        print(f"M1_ASSET={path}", flush=True)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default)
        for prim in stage.Traverse():
            if prim.GetPath().pathString.count("/") <= 1:
                continue
            if not prim.IsA(UsdGeom.Boundable):
                continue
            extent = UsdGeom.Boundable(prim).GetExtentAttr().Get()
            if extent is None:
                continue
            print(
                f"M1_ASSET_EXTENT={prim.GetPath()} extent="
                f"{tuple(tuple(float(value) for value in point) for point in extent)} "
                f"xform={xform_cache.GetLocalToWorldTransform(prim)}",
                flush=True,
            )


try:
    main()
finally:
    app.close()
