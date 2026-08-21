#!/usr/bin/env python3
"""Author precision contact offsets on registered gearbox colliders in the scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim.simulation_app import SimulationApp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="assets/simple_room_scene.usd")
    parser.add_argument("--registry", default="assembly/asset_registry.yaml")
    parser.add_argument("--contact-offset", type=float, default=0.0001)
    parser.add_argument("--rest-offset", type=float, default=-0.00005)
    args = parser.parse_args()

    app = SimulationApp({"headless": True})
    try:
        import yaml
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        registry = yaml.safe_load(Path(args.registry).read_text())
        names = set()
        for part in registry.get("parts", []):
            names.add(part["name"])
            names.update(part.get("alternate_names") or [])
            prim_path = part.get("prim_path") or ""
            if prim_path:
                names.add(prim_path.rstrip("/").split("/")[-1])

        stage = Usd.Stage.Open(args.scene)
        if stage is None:
            raise RuntimeError(f"failed to open {args.scene}")

        authored = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not any(component in names for component in str(prim.GetPath()).split("/")):
                continue
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.GetContactOffsetAttr().Set(args.contact_offset)
            collision.GetRestOffsetAttr().Set(args.rest_offset)
            if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf":
                sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                path_components = set(str(prim.GetPath()).split("/"))
                casing = bool(path_components & {"Casing_Base", "Casing_Top"})
                sdf.GetSdfResolutionAttr().Set(512 if casing else 256)
            authored += 1

        stage.GetRootLayer().Save()
        print(
            f"AUTHORED {authored} registered collider(s): "
            f"contact_offset={args.contact_offset} m, rest_offset={args.rest_offset} m"
        )
        return 0 if authored else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
