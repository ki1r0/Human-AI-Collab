# Gearbox part and manipulation-fidelity matrix

Status: `PART_MANIPULATION_FIDELITY: PARTIAL`

The integrated registry contains 57 named components represented by 15 reusable CAD asset files. The two gear-on-shaft insertions and the breather-plug insertion now pass controlled collision-on PhysX tests. The M6 hub-bolt interface still stops at the casing surface, and gear-to-gear rolling contact, covers, bearings, casing closure, oil indicators, and M10 fasteners do not yet have equivalent controlled-physics proof. The entire set therefore cannot be called manipulation-grade yet.

## Units and physics convention

- Source CAD stages use `metersPerUnit=0.01`; the canonical assembly stage uses `metersPerUnit=1.0` and instantiated part scale `0.002`.
- Every reusable part root currently carries a nominal mass of 0.2 kg. That is a placeholder, not a measured bill-of-materials value. No family-specific center of mass or inertia was added in this task.
- Manipulation-critical scene colliders use contact offset 0.0001 m and rest offset -0.00005 m. These are close to the targeted RoCo reference values (0.0001 m and -0.0005 m) while retaining a smaller negative rest offset for this tighter gearbox geometry.
- No new dependency or global physics-material override was added. Default scene material/friction remains in force. CCD and unrelated solver settings were not changed because the controlled tests did not diagnose those as the failure source.

## Complete component inventory

| Registry components | Count | Reusable visual/collision source | Default state and destination | Mating axis / nominal transform | Current collision model |
|---|---:|---|---|---|---|
| `Casing_Base` | 1 | `assets/parts/Casing Base.usd` | floor fixture at `[-1.8,1.5,0.02]`; assembly root | sockets define shaft Z insertion, cover normals, side ports, and `Casing_Top` closure | detailed CAD mesh, SDF |
| `Casing_Top` | 1 | `assets/parts/Casing Top.usd` | floor at `[1.8,1.5,0.02]`; mates to `Casing_Base/socket_casing_mate` | `(0,0,0)`, rotate Y 180 degrees relative to the mating socket | detailed CAD mesh, SDF |
| `Input_Shaft`, `Transfer_Shaft`, `Output_Shaft` | 3 | matching files under `assets/parts/` | floor scatter; then `Casing_Base/socket_gear_*` | Input: Rx -90, local Z -74.9; Transfer: Ry -90, Z -27; Output: Rx -90, Z -18.32 | detailed CAD mesh, SDF |
| `Transfer_Gear`, `Output_Gear` | 2 | matching USD and STL files under `assets/parts/` | pre-assembled on `Transfer_Shaft` / `Output_Shaft` | Transfer `(22.1,0,0)`, Ry 90; Output `(0,15.5,0)`, Rx -90 | original visual CAD plus toothed annulus SDF proxy |
| `Bearing_Input_Bottom`, `Bearing_Input_Top`, `Bearing_Transfer`, `Bearing_Output_Top`, `Bearing_Output_Bottom` | 5 | composed/pre-assembled scene geometry | fixed initial condition on named shaft; not sequenced | axial offsets `-100.75`, `2.732481`, `-50.563`, `51.156502`, `-51.156502` asset units | existing composed collision; not upgraded individually |
| `Hub_Cover_Output_Base`, `Hub_Cover_Output_Top` | 2 | `assets/parts/Hub Cover Output.usd` | named `Casing_Base`/`Casing_Top` output sockets | cover-normal placement with runtime seat correction | convex decomposition |
| `Hub_Cover_Input_Top` | 1 | `assets/parts/Hub Cover Input.usd` | `Casing_Top/socket_hub_input` | Ry 90, Rz 90, axial seat 6.3 asset units | convex decomposition |
| `Hub_Cover_Small_Base_01`, `Hub_Cover_Small_Base_02`, `Hub_Cover_Small_Top` | 3 | `assets/parts/Hub Cover Small.usd` | named small-cover sockets on the casings | Ry 90 plus part-specific phase/seat | convex decomposition |
| `M6_Hub_Bolt_01_base`…`12_base`, `M6_Hub_Bolt_01_top`…`12_top` | 24 | `assets/parts/M6 Hub Bolt.usd` | runtime-scattered; each targets its numbered casing socket | local shaft Y mapped to casing normal by Rx 90 | SDF; insertion currently fails |
| `M10_Casing_Bolt_01`…`06` | 6 | `assets/parts/M10 Casing Bolt.usd` | table/runtime clones; numbered `Casing_Top` through-bolt sockets | local shaft Y mapped by Rx 90 | SDF; unvalidated physically |
| `M10_Casing_Nut_01`…`06` | 6 | `assets/parts/M10 Casing Nut.usd` | each mates to its same-numbered M10 bolt | coaxial with bolt, measured axial seat | SDF; unvalidated physically |
| `Oil_Level_Indicator_01`, `Oil_Level_Indicator_02` | 2 | `assets/parts/Oil Level Indicator.usd` | side ports `Casing_Base/socket_oil_1` and `_2` | side insertion, ±13 asset-unit fit correction | SDF; unvalidated physically |
| `Breather_Plug` | 1 | `assets/parts/Breather Plug.usd` | floor at `[1.0,-0.8,0.02]`; `Casing_Base/socket_breather` | casing Y insertion; physical shoulder seat at world Y 2.2899 m in isolated test | SDF; controlled test passes |

Visual meshes remain the imported CAD. Except for the two gear bores, no visible dimensions were changed. All 15 reusable assets retain at least one collider and a rigid-body mass.

## Mating-pair evidence

| Source → target | Interface / nominal mating geometry | Required clearance | Change made | Validation and result |
|---|---|---:|---|---|
| `Transfer_Gear` → `Transfer_Shaft` | gear bore over X-axis shaft; center X=22.1, half-thickness 11.05 asset units | 0.5 asset radial = 1.0 mm scene | bore set to 13.483 against maximum swept shaft radius 12.983; original tooth CAD remains visual; watertight toothed annulus SDF proxy preserves teeth and open bore | static fit PASS; controlled collision-on insertion PASS, final X=0.04082 m, nominal error 3.38 mm |
| `Output_Gear` → `Output_Shaft` | gear bore over Y-axis shaft; center Y=15.5, half-thickness 13.5 | 0.5 asset radial = 1.0 mm scene | bore set to 33.504 against maximum swept shaft radius 33.004; matching STL updated; toothed annulus SDF proxy | static fit PASS; controlled collision-on insertion PASS, final Y=0.02918 m, nominal error 1.82 mm |
| `Transfer_Gear` ↔ `Input_Shaft` integral pinion | external tooth mesh at runtime phase 88.5 degrees | running tooth clearance, not yet measured | teeth retained in collision proxy; no center-distance change | assembled phase has a kinematic marker audit only; controlled rotation/mesh UNVALIDATED |
| `Transfer_Gear` ↔ `Output_Gear` | external tooth mesh at runtime output phase -52.5 degrees | running tooth clearance, not yet measured | teeth retained in collision proxy | controlled rotation/mesh UNVALIDATED |
| `Input_Shaft` / `Transfer_Shaft` / `Output_Shaft` → `Casing_Base` | shaft/bearing modules into three named casing bores along Z | bore-specific | shaft colliders changed from convex hull/decomposition behavior to SDF so stepped profiles are preserved | instance staging resolves; final physical insertion basin UNVALIDATED |
| `Breather_Plug` → `Casing_Base/socket_breather` | male plug into casing side port along Y; shoulder limits depth | existing CAD gap | plug and casing use SDF; no visual shrink/enlargement | controlled collision-on insertion PASS, final Y=2.28988 m; no snap, teleport, or collision disable |
| `Oil_Level_Indicator_01/_02` → `Casing_Base` | side plug/socket | not yet measured | indicator and casing use SDF | UNVALIDATED |
| six named hub covers → casing sockets | cover/housing face and pilot geometry | flush face with no penetration | retained CAD and convex decomposition | kinematic placement audit only; controlled physics UNVALIDATED |
| 24 named M6 hub bolts → casing/hub-cover holes | fastener/hole along casing normal | measured bolt shank radius 2.94 versus a nominal local feature radius about 3.6 asset units | M6 and casing switched to SDF; contact offsets authored | FAIL: nominal collision-on approach stops at outer surface (Z=0.08049 m versus expected seat 0.0584 m). The source casing collider does not yield a dependable through-hole; must be rebuilt locally around the fastener bores before use as a robot insertion task |
| six M10 bolts → `Casing_Top` and six nuts → bolts | through-bolt then threaded/axial nut seat | not yet measured | bolt and nut switched to SDF | UNVALIDATED |
| `Casing_Top` → `Casing_Base` | housing closure at parting faces | flush, no overlap | casing CAD switched to SDF; existing Y-180 transform retained | static/kinematic audit only; controlled closure UNVALIDATED |
| five bearings → three shafts | bearing/shaft seats | not yet measured | no change; pre-assembled initial condition | UNVALIDATED as independent mating pairs |

## What changed and why

`tools/tune_gear_bores.py` measures the full swept shaft profile rather than only the final gear plane. It applies the smallest bore enlargement that gives 0.5 asset-unit radial clearance to both the USD visual source and matching STL. At the scene scale this is 1 mm radial clearance: large enough to exceed the two 0.1 mm contact envelopes and SDF discretization, but small relative to the 100–161 asset-unit gear diameters.

The original gear CAD mesh is now visual-only. `collision_gear` is a watertight annular proxy with the original tooth count and root/tip radii, an open bore, and SDF collision. This avoids the convex hull failure that sealed the bore while preserving external tooth contact. Shafts, casings, fasteners, nuts, and insertion accessories use SDF only where concavity/profile fidelity matters; simple covers remain on the cheaper existing convex decomposition.

`tools/author_part_contact_offsets.py` authors the measured contact/rest values onto 52 registered colliders in the canonical composed scene. `tools/check_part_fidelity.py` audits all reusable assets, collision approximations, masses, extents, gear path clearances, and composed-scene offsets. `tools/test_controlled_mating.py` drives four isolated approaches with native PhysX, gravity off, target kinematic, source dynamic, collisions enabled throughout, and no post-contact correction.

## RoCo comparison and limitation

The targeted RoCo reference uses high-detail gear collision with 0.0001 m contact offset, -0.0005 m rest offset, low gear friction, damping, and increased solver iterations. This repository now matches the essential behavioral class for the two tested gear-on-shaft placements: real bore passage with tooth geometry still collidable. It does not yet establish the requested full RoCo-class success basin because no offset/angle perturbation grid or gear-to-gear rotation test was completed. The M6 failure and untested interface families keep the overall status `PARTIAL`.
