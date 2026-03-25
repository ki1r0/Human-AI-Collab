#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim.simulation_app import SimulationApp


def _iter_asset_values(value):
    from pxr import Sdf

    if isinstance(value, Sdf.AssetPath):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, Sdf.AssetPath):
                yield item


def _nested_rigid_bodies(stage):
    from pxr import UsdPhysics

    def _rigid_body_enabled(prim) -> bool:
        if not prim or not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return False
        rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
        enabled_attr = rigid_api.GetRigidBodyEnabledAttr()
        if enabled_attr and enabled_attr.IsValid():
            value = enabled_attr.Get()
            if value is False:
                return False
        return True

    findings = []
    for prim in stage.Traverse():
        if not _rigid_body_enabled(prim):
            continue
        ancestor = prim.GetParent()
        while ancestor and ancestor.IsValid():
            if _rigid_body_enabled(ancestor):
                findings.append((prim.GetPath().pathString, ancestor.GetPath().pathString))
                break
            ancestor = ancestor.GetParent()
    return findings


def _missing_asset_paths(stage):
    findings = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            try:
                values = list(_iter_asset_values(attr.Get()))
            except Exception:
                continue
            for asset in values:
                raw_path = str(getattr(asset, "path", "") or "").strip()
                if not raw_path:
                    continue
                resolved = str(getattr(asset, "resolvedPath", "") or "").strip()
                if resolved:
                    continue
                findings.append((prim.GetPath().pathString, attr.GetName(), raw_path))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the release scene for nested rigid bodies and unresolved asset paths.")
    parser.add_argument("--scene", required=True, help="USD scene to open.")
    parser.add_argument("--wait-updates", type=int, default=60, help="Kit update ticks to allow stage loading before auditing.")
    args = parser.parse_args()

    app = SimulationApp({"headless": True})
    try:
        import omni.usd

        scene_path = str(Path(args.scene).expanduser().resolve())
        ctx = omni.usd.get_context()
        if not ctx.open_stage(scene_path):
            raise RuntimeError(f"Failed to open stage: {scene_path}")

        for _ in range(max(args.wait_updates, 0)):
            app.update()

        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError(f"Stage unavailable after opening: {scene_path}")

        print(f"[AUDIT] scene={scene_path}")

        nested = _nested_rigid_bodies(stage)
        print(f"[AUDIT] nested_enabled_rigid_bodies={len(nested)}")
        for child_path, ancestor_path in nested:
            print(f"[AUDIT][RIGID] child={child_path} ancestor={ancestor_path}")

        missing_assets = _missing_asset_paths(stage)
        print(f"[AUDIT] unresolved_asset_paths={len(missing_assets)}")
        for prim_path, attr_name, raw_path in missing_assets:
            print(f"[AUDIT][ASSET] prim={prim_path} attr={attr_name} path={raw_path}")
    finally:
        app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
