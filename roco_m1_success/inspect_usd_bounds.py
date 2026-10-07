from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom

root = "/home/sunsiliang/roco_runtime/gearboxAssembly/source/Galaxea_Lab_External/assets/Gearbox"
for name in (
    "sun_planetary_gear_3x_scale.usd",
    "planetary_reducer_3x_scale.usd",
    "ring_gear_3x_scale.usd",
    "planetary_carrier_3x_scale.usd",
):
    stage = Usd.Stage.Open(f"{root}/{name}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )
    print(name, cache.ComputeWorldBound(stage.GetPseudoRoot()).GetRange(), flush=True)
app.close()
