#!/usr/bin/env python3
from __future__ import annotations

from isaacsim.simulation_app import SimulationApp

TARGETS = (
    "/Root/Output_Shaft",
    "/Root/Output_Shaft/Output_Gear",
    "/Root/Transfer_Shaft",
    "/Root/Transfer_Shaft/Transfer_Gear",
)


def _describe_prim(stage, path: str) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(path)
    print(f"[INSPECT] path={path}")
    if prim is None or not prim.IsValid():
        print("[INSPECT]   valid=False")
        return

    print(f"[INSPECT]   type={prim.GetTypeName() or '<none>'}")
    print(f"[INSPECT]   applied_schemas={list(prim.GetAppliedSchemas())}")

    xform = UsdGeom.Xformable(prim)
    try:
        reset_stack = xform.GetResetXformStack()
    except Exception:
        reset_stack = None
    print(f"[INSPECT]   reset_xform_stack={reset_stack}")

    rigid = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
    enabled_attr = rigid.GetRigidBodyEnabledAttr() if rigid else None
    enabled_value = enabled_attr.Get() if enabled_attr and enabled_attr.IsValid() else None
    has_authored = enabled_attr.HasAuthoredValue() if enabled_attr and enabled_attr.IsValid() else False
    print(f"[INSPECT]   rigid_body_enabled={enabled_value!r} authored={has_authored}")

    for attr_name in ("physics:rigidBodyEnabled", "xformOpOrder"):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            print(f"[INSPECT]   attr[{attr_name}]={attr.Get()!r} authored={attr.HasAuthoredValue()}")

    descendants = []
    for subprim in Usd.PrimRange(prim):
        if subprim.GetPath() == prim.GetPath():
            continue
        if subprim.HasAPI(UsdPhysics.RigidBodyAPI):
            descendants.append(subprim.GetPath().pathString)
    if descendants:
        print(f"[INSPECT]   rigid_body_descendants={descendants}")

    for spec in prim.GetPrimStack():
        try:
            spec_layer = spec.layer.identifier
            spec_path = spec.path.pathString
            api_schemas = spec.GetInfo("apiSchemas")
        except Exception:
            continue
        print(f"[INSPECT]   stack layer={spec_layer} path={spec_path} apiSchemas={api_schemas}")


def main() -> int:
    app = SimulationApp({"headless": True})
    try:
        import omni.usd

        scene_path = "/workspace/Human-AI-Collab/assets/simple_room_scene.usd"
        ctx = omni.usd.get_context()
        if not ctx.open_stage(scene_path):
            raise RuntimeError(f"Failed to open stage: {scene_path}")
        for _ in range(60):
            app.update()
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError(f"Stage unavailable after opening: {scene_path}")

        for path in TARGETS:
            _describe_prim(stage, path)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
