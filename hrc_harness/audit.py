"""Read-only USD inspection; collider presence is not proof that a socket is open."""

import argparse
import json
from pathlib import Path


def audit_asset(path, scale=0.002):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open USD: {path}")
    frames, meshes, rigid = [], [], []
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if any(token in name for token in ("socket", "plug", "grasp")):
            frames.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                           "local_transform": [list(row) for row in UsdGeom.Xformable(prim).GetLocalTransformation()]
                           if prim.IsA(UsdGeom.Xformable) else None})
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, stage.GetDefaultPrim())
            points = np.column_stack((points, np.ones(len(points)))) @ np.asarray(transform)
            points = points[:, :3] * scale
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            planes = None
            if np.all(counts == 3):
                triangles = points[np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
                cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                areas = np.linalg.norm(cross, axis=1)/2
                normals = cross / np.maximum(2*areas[:, None], 1e-15)
                aligned = np.abs(normals) > np.cos(np.deg2rad(1))
                planes = (areas[:, None]*aligned).sum(axis=0).tolist()
            collision = UsdPhysics.CollisionAPI(prim) if prim.HasAPI(UsdPhysics.CollisionAPI) else None
            approximation = UsdPhysics.MeshCollisionAPI(prim) if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
            meshes.append({"path": str(prim.GetPath()), "points": len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or []),
                           "collision_enabled": collision.GetCollisionEnabledAttr().Get() if collision else False,
                           "approximation": approximation.GetApproximationAttr().Get() if approximation else None,
                           "root_frame_bounds_m": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
                           "axis_aligned_planar_area_m2": planes})
    return {"path": str(path), "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)), "default_prim": str(stage.GetDefaultPrim().GetPath()),
            "rigid_bodies": rigid, "frames": frames, "meshes": meshes,
            "hole_clearance_verified": False, "physical_seating_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("assets", nargs="*", type=Path, default=[
        Path("assets/parts/Hub Cover Output.usd"), Path("assets/parts/Casing Top.usd")])
    args = parser.parse_args()
    report = {"assets": [audit_asset(path) for path in args.assets]}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
