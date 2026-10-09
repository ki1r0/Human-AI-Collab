# Candidate #01 Static Geometry Feasibility

## Decision

Use `M6_Hub_Bolt_01_top` → Output cover hole at `Casing_Top/socket_bolt_hub_1` for the bounded no-thread axial insertion task. Its full modeled shank cylinder clears both meshes, and the cap has sampled annular support when moved to the measured contact pose. This is a geometry-only selection, not a simulated or robotic success claim.

## Measured Scale And Placement

- `simple_room_scene.usd`: `metersPerUnit=1`, Z-up. The composed bolt and casing root transforms each have basis lengths `[0.002000000095, 0.002000000095, 0.002000000095]`; asset root matrices are identity. Standalone asset metadata says `metersPerUnit=0.01`, Y-up (which would interpret the raw shaft as `58.84 mm`), while the simple-room composition uses its own stage units and composed scale; the measured scene-world result is the task reference here.
- The bolt's local-Y shaft interval is `[-13.074999, 8.925001]` source units; p95 shaft radius `2.942001`, median OD `5.883998`, and estimated cap-bottom plane `Y=8.925001`. Cap outer radius is `5.773503` source units.
- At the actual scene scale, the shaft OD is `11.767997 mm`, or `1.961x` nominal 6 mm. Do not silently treat this asset as a physically nominal M6 fastener; no asset or scene scale was changed.
- Casing socket center is `(40, 0, 27.9)` source units. Applying the authored Output-cover pose `(0,43.41859,31)`, rotation `(-90,180,0)`, maps the bolt axis at the cover midplane to cover-local `(-40, 0, -43.41859)`, radial position `59.035362` from the cover origin.

## Clearance And Seat

- The nearest Output-cover mesh surface along the full shaft axial interval is `3.647927` source units from the bolt axis. The p95 shaft radius is `2.942001`, leaving `1.411853 mm` radial clearance at the measured scene scale. The corresponding local aperture estimate is `14.591709 mm` across; the maximum head envelope exceeds that aperture radius by `4.251153 mm` (this is not the underside bearing overlap; see correction below).
- The nearest casing mesh surface over the same shaft interval is `3.110965` source units from the axis, leaving `0.337928 mm` radial clearance. The closest surface section is bolt-local axial `[-4.970820,-4.724726]` source units. The result is positive but narrow; contact tolerances/dynamics remain untested.
- The authored bolt root Z `29.2` leaves a measured `0.190950 mm` cap-to-cover gap. Translating along the insertion axis by `-0.09547491` source units (`-0.190950 mm`) gives contact root translation `(40, 0, 29.10452509)`. At that pose, the two bearing radii outside the aperture, `4.357752` and `5.653503` source units, each have `48/48` covered samples. The inner test radius `3.062001` is inside the aperture and correctly excluded from bearing coverage.
- At the contact pose, the bolt tip is at casing Z `16.029526`; the casing's outermost mesh Z is `27.948326`, so the tip extends `23.837601 mm` below that top plane. The centerline's lower casing crossings are Z `6.948158` and `5.548164`; the tip remains `18.162737 mm` above the nearest interior-floor crossing. No modeled bottoming is indicated.
- The axial fit check tests the entire shank interval against transformed cover and casing triangles, not only the centerline. No thread rotation is required by this static clearance geometry; an active straight insertion to the cap seat is geometrically feasible at the actual scene scale.

## Evidence And Limits

Inventory: [`geometry_inventory.json`](geometry_inventory.json). Probe: [`geometry_feasibility.py`](geometry_feasibility.py), using installed `pxr` through `tools/run_tool.sh`; its self-test passed. The final probe also passed and retested full-shank clearance at the calculated contact pose.

Not tested: collision/contact dynamics, grasp affordance or robot reachability, force closure, active controller behavior, randomized trials, or the task evaluator. This report does not certify P2+ or full task success.

## Bearing-Ring Interpretation Correction

The later CAD-derived cap/shaft split audit refined the support interpretation above. At the bolt's actual underside plane (`root-local Y=8.925001`), the cap extends only to radius `5.000001` source units. Therefore the sampled `4.357752` ring is on the cap underside and lies outside the `3.647927` cover aperture, but the `5.653503` ring is outside the bolt underside: its `48/48` cover-surface samples do **not** demonstrate bolt-cap bearing. The measured radial overlap at the actual underside is `2.704148 mm`; real support/contact still requires calibrated collision evidence.
