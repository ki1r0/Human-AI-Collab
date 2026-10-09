"""Offline USD tests for the source-derived bolt collision proxies."""

import importlib.util
import math
import unittest

from runtime.bolt_harness.collision import author_split_bolt_convex_proxies


HAS_SCIPY = importlib.util.find_spec("scipy") is not None
try:
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
except ImportError:
    Gf = PhysxSchema = Usd = UsdGeom = UsdPhysics = None
HAS_PXR_PHYSX = Usd is not None


def _box_mesh_data(bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    points = [
        (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
        (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1),
    ]
    quads = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    triangles = [
        triangle
        for quad in quads
        for triangle in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3]))
    ]
    return points, triangles


@unittest.skipUnless(HAS_SCIPY and HAS_PXR_PHYSX, "requires installed Isaac PXR/PhysX and SciPy")
class BoltCollisionProxyTests(unittest.TestCase):
    def _stage_with_split_bolt(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        root = UsdGeom.Xform.Define(stage, "/World/Bolt").GetPrim()
        UsdGeom.Xformable(root).AddScaleOp().Set(Gf.Vec3d(0.002, 0.002, 0.002))
        UsdPhysics.RigidBodyAPI.Apply(root)

        shaft_points, shaft_faces = _box_mesh_data((-1.5, 1.5, -2.0, 0.0, -1.0, 1.0))
        cap_points, cap_faces = _box_mesh_data((-2.0, 2.0, 0.0, 2.0, -1.5, 1.5))
        points = shaft_points + cap_points
        faces = shaft_faces + [tuple(index + len(shaft_points) for index in face) for face in cap_faces]
        source = UsdGeom.Mesh.Define(stage, "/World/Bolt/node_/mesh_")
        source.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        source.CreateFaceVertexCountsAttr([3] * len(faces))
        source.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
        UsdGeom.Xformable(source.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(3.0, 1.0, -2.0))

        collision = UsdPhysics.CollisionAPI.Apply(source.GetPrim())
        collision.GetCollisionEnabledAttr().Set(True)
        physx = PhysxSchema.PhysxCollisionAPI.Apply(source.GetPrim())
        physx.GetContactOffsetAttr().Set(0.00005)
        physx.GetRestOffsetAttr().Set(0.0)
        binding = source.GetPrim().CreateRelationship("material:binding:physics")
        binding.SetTargets(["/World/BoltPhysicsMaterial"])
        return stage, root, source, tuple(source.GetPointsAttr().Get()), binding

    def test_authors_two_root_local_hidden_colliders_and_preserves_body_materials(self):
        stage, root, source, original_points, binding = self._stage_with_split_bolt()
        result = author_split_bolt_convex_proxies(
            stage,
            bolt_root_path="/World/Bolt",
            source_mesh_path="/World/Bolt/node_/mesh_",
            split_y_source_units=1.0,
        )

        self.assertTrue(root.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertFalse(UsdPhysics.CollisionAPI(source.GetPrim()).GetCollisionEnabledAttr().Get())
        self.assertEqual(tuple(source.GetPointsAttr().Get()), original_points)
        self.assertEqual(binding.GetTargets(), ["/World/BoltPhysicsMaterial"])
        self.assertEqual(result["split_y_source_units"], 1.0)
        self.assertEqual(len(result["pieces"]), 2)

        by_name = {piece["name"]: piece for piece in result["pieces"]}
        shaft = by_name["shaft"]
        cap = by_name["cap"]
        self.assertEqual(shaft["bounds_root_local_source_units_xyz"][1], [-1.0, 1.0])
        self.assertEqual(cap["bounds_root_local_source_units_xyz"][1], [1.0, 3.0])
        self.assertEqual(shaft["bounds_root_local_source_units_xyz"][0], [1.5, 4.5])
        self.assertEqual(cap["bounds_root_local_source_units_xyz"][2], [-3.5, -0.5])

        for piece in result["pieces"]:
            prim = stage.GetPrimAtPath(piece["prim_path"])
            self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            self.assertEqual(str(UsdGeom.Imageable(prim).GetPurposeAttr().Get()), "guide")
            self.assertEqual(str(UsdGeom.Imageable(prim).GetVisibilityAttr().Get()), "invisible")
            self.assertEqual(
                str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()), "convexHull"
            )
            self.assertEqual(
                PhysxSchema.PhysxConvexHullCollisionAPI(prim).GetHullVertexLimitAttr().Get(), 64
            )
            self.assertAlmostEqual(
                PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get(), 0.00005
            )
            self.assertEqual(
                stage.GetRelationshipAtPath(f"{prim.GetPath()}.material:binding:physics").GetTargets(),
                ["/World/BoltPhysicsMaterial"],
            )

    def test_requires_existing_dynamic_body_and_source_under_root(self):
        stage, root, source, _, _ = self._stage_with_split_bolt()
        root.RemoveAPI(UsdPhysics.RigidBodyAPI)
        with self.assertRaisesRegex(ValueError, "must already have RigidBodyAPI"):
            author_split_bolt_convex_proxies(
                stage,
                bolt_root_path="/World/Bolt",
                source_mesh_path="/World/Bolt/node_/mesh_",
                split_y_source_units=1.0,
            )


if __name__ == "__main__":
    unittest.main()
