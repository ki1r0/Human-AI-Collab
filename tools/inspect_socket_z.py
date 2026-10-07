"""Inspect source CAD vertices around the Hub Cover output interface."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from pxr import Gf, Usd, UsdGeom  # noqa: E402
import math  # noqa: E402

for filename, x0, y0 in (("Casing Top", 0.0, 43.42), ("Casing Top", 0.0, 39.75), ("Hub Cover Output", 0.0, 0.0)):
    stage = Usd.Stage.Open(f"assets/parts/{filename}.usd")
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    root_inv = cache.GetLocalToWorldTransform(root).GetInverse()
    values = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        to_root = cache.GetLocalToWorldTransform(prim) * root_inv
        for point in mesh.GetPointsAttr().Get() or []:
            q = to_root.Transform(Gf.Vec3d(point))
            distance = math.hypot(float(q[0]) - x0, float(q[1]) - y0)
            if distance <= 90.0:
                values.append((distance, float(q[0]), float(q[1]), float(q[2])))
    print(filename, "points", len(values), flush=True)
    if values:
        values.sort()
        print("nearest", [(round(d, 2), round(x, 2), round(y, 2), round(z, 2)) for d, x, y, z in values[:24]], flush=True)
        print("z_range", min(v[3] for v in values), max(v[3] for v in values), flush=True)
        # The socket registry contains both the old 39.75 mm fit and the
        # composed-stage 43.42 mm override.  Report the radial profile of the
        # actual top-surface vertices so this discrepancy can be audited from
        # geometry rather than inferred from a controller outcome.
        top = sorted(v[0] for v in values if filename != "Casing Top" or v[3] > 20.0)
        if top:
            n = len(top)
            quantiles = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
            print("radial_quantiles", [(q, round(top[min(n - 1, int(q * n))], 3)) for q in quantiles], flush=True)
            bands = []
            for lo, hi in ((0, 40), (40, 45), (45, 47), (47, 49), (49, 51), (51, 55), (55, 60), (60, 70), (70, 90)):
                count = sum(lo <= d < hi for d in top)
                if count:
                    bands.append((lo, hi, count))
            print("radial_bands", bands, flush=True)

app.close()
