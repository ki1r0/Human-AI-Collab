#!/usr/bin/env python3
"""Report manipulation-critical geometry and collider properties for gearbox assets."""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402


PARTS = os.path.join(_REPO, "assets", "parts")
SCENE = os.path.join(_REPO, "assets", "simple_room_scene.usd")
GEAR_FITS = (
    ("Transfer Gear", "Transfer Shaft", 0, 22.1, 11.05),
    ("Output Gear", "Output Shaft", 1, 15.5, 13.50),
)
SDF_ASSETS = {
    "Transfer Gear", "Output Gear", "Casing Base", "Casing Top",
    "Input Shaft", "Transfer Shaft", "Output Shaft",
    "M6 Hub Bolt", "M10 Casing Bolt", "M10 Casing Nut",
    "Breather Plug", "Oil Level Indicator",
}


def mesh_points(path: str, *, skip_proxy: bool = False) -> np.ndarray:
    stage = Usd.Stage.Open(path)
    points = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or (skip_proxy and prim.GetName() == "collision_gear"):
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in UsdGeom.Mesh(prim).GetPointsAttr().Get() or []:
            p = matrix.Transform(Gf.Vec3d(point))
            points.append((p[0], p[1], p[2]))
    return np.asarray(points)


def radial_clearance(gear: str, shaft: str, axis: int, mount: float,
                     half_thickness: float) -> tuple[float, float, float]:
    gear_points = mesh_points(os.path.join(PARTS, f"{gear}.usd"), skip_proxy=True)
    shaft_points = mesh_points(os.path.join(PARTS, f"{shaft}.usd"))

    gear_radius = np.linalg.norm(gear_points[:, :2], axis=1)
    bore_radius = float(gear_radius.min())

    radial_axes = [i for i in range(3) if i != axis]
    # The positive-end approach sweeps all sections above the leading face.
    section = shaft_points[shaft_points[:, axis] >= mount - half_thickness - 0.1]
    shaft_radius = float(np.linalg.norm(section[:, radial_axes], axis=1).max())
    return bore_radius, shaft_radius, bore_radius - shaft_radius


def audit_asset(path: str) -> dict:
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    approximations = []
    collision_meshes = 0
    for mesh in meshes:
        if mesh.HasAPI(UsdPhysics.CollisionAPI):
            collision_meshes += 1
            approximation = UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get()
            if approximation:
                approximations.append(str(approximation))
    mass = UsdPhysics.MassAPI(root).GetMassAttr().Get() if root.HasAPI(UsdPhysics.MassAPI) else None
    points = mesh_points(path)
    extent = np.ptp(points, axis=0) if len(points) else np.zeros(3)
    return {
        "meshes": len(meshes),
        "collision_meshes": collision_meshes,
        "approx": sorted(set(approximations)),
        "rigid": root.HasAPI(UsdPhysics.RigidBodyAPI),
        "mass": mass,
        "extent": extent,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
    }


def main() -> int:
    failures = 0
    for name in sorted(os.path.splitext(f)[0] for f in os.listdir(PARTS) if f.endswith(".usd")):
        result = audit_asset(os.path.join(PARTS, f"{name}.usd"))
        expected_approximation = ["sdf"] if name in SDF_ASSETS else ["convexDecomposition"]
        collision_ok = result["collision_meshes"] >= 1 and result["approx"] == expected_approximation
        physics_ok = result["rigid"] and result["mass"] is not None
        status = "PASS" if collision_ok and physics_ok else "FAIL"
        failures += status == "FAIL"
        extent = ",".join(f"{v:.2f}" for v in result["extent"])
        print(
            f"ASSET {status} {name}: extent=[{extent}] units, mpu={result['meters_per_unit']}, "
            f"meshes={result['collision_meshes']}/{result['meshes']} collision, "
            f"approx={result['approx']}, mass={result['mass']} kg"
        )

    for gear, shaft, axis, mount, half_thickness in GEAR_FITS:
        bore, radius, clearance = radial_clearance(gear, shaft, axis, mount, half_thickness)
        status = "PASS" if clearance >= 0.49 else "FAIL"
        failures += status == "FAIL"
        print(
            f"FIT {status} {gear} -> {shaft}: bore={bore:.3f}, "
            f"shaft={radius:.3f}, radial_clearance={clearance:.3f} asset units"
        )

    scene = Usd.Stage.Open(SCENE)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    critical_names = {name.replace(" ", "_") for pair in GEAR_FITS for name in pair[:2]}
    for prim in scene.Traverse():
        if prim.GetName() not in critical_names:
            continue
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
        offsets = []
        for child in Usd.PrimRange(prim):
            if not child.IsA(UsdGeom.Mesh):
                continue
            offsets.append((
                child.GetAttribute("physxCollision:contactOffset").Get(),
                child.GetAttribute("physxCollision:restOffset").Get(),
            ))
        print(
            f"SCENE {prim.GetPath()}: extent=[{extent[0]:.4f},{extent[1]:.4f},{extent[2]:.4f}] "
            f"stage units, mpu={UsdGeom.GetStageMetersPerUnit(scene)}, offsets={offsets}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
