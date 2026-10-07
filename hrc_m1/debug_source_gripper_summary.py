"""Inspect the source R1 gripper prims and their authored collision geometry."""

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


def main() -> int:
    path = "/workspace/gearboxAssembly/source/Galaxea_Lab_External/assets/Robots/Galaxea/r1_DVT_colored_cam_pos.usd"
    stage = Usd.Stage.Open(path)
    rows = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        name = str(prim.GetName()).lower()
        path_text = str(prim.GetPath())
        if "gripper" not in name and "gripper" not in path_text.lower():
            continue
        attrs = {}
        for attr_name in ("physics:approximation", "physics:collisionEnabled", "physics:contactOffset", "physics:restOffset", "physxSDFMeshCollision:sdfResolution"):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.HasAuthoredValue():
                try:
                    attrs[attr_name] = attr.Get()
                except Exception:
                    attrs[attr_name] = "<unreadable>"
        row = {"path": path_text, "type": prim.GetTypeName(), "apis": [str(api) for api in prim.GetAppliedSchemas()], "attrs": attrs}
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            if points:
                row["point_count"] = len(points)
                row["local_min"] = [float(min(point[i] for point in points)) for i in range(3)]
                row["local_max"] = [float(max(point[i] for point in points)) for i in range(3)]
        rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rows, indent=2, default=str), flush=True)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
