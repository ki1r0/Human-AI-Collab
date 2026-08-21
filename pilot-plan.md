# Real-part 12-pair gearbox pilot plan

Status: `PILOT_PLAN: PASS`

This is a planning deliverable. It preserves the falsification question in `PILOT_12_PAIR_EXECUTION_PLAN.md`: does a model reproduce the demonstrated order, or identify the physical constraint that makes the reverse order feasible or infeasible? It does not execute the participant/model study. Every operation below names a component from `assembly/asset_registry.yaml`; no placeholder parts are used.

The pilot must not begin data collection until the M6 bore collider and the pair-specific HARD/COMMUTABLE geometry variants pass their registered physics oracles. The current M6 limitation is documented in `part-fidelity-matrix.md`.

## Initial scene and dependency graph

Use `assets/simple_room_scene.usd`, `assembly/asset_registry.yaml`, and `assembly/instances/canonical_grouped.yaml`. `Casing_Base` and `Casing_Top` begin separated. The shaft modules consist of:

- `Input_Shaft` with `Bearing_Input_Bottom` and `Bearing_Input_Top` pre-assembled;
- `Transfer_Shaft` with `Transfer_Gear` and `Bearing_Transfer` pre-assembled;
- `Output_Shaft` with `Output_Gear`, `Bearing_Output_Top`, and `Bearing_Output_Bottom` pre-assembled.

All covers, numbered M6 hub bolts, numbered M10 casing bolt/nut pairs, `Oil_Level_Indicator_01`, `Oil_Level_Indicator_02`, and `Breather_Plug` begin loose at their registry/runtime scatter poses.

```text
validate scene
├─ prepare Casing_Base: flip → three covers → their M6 bolts ─┐
├─ prepare Casing_Top:  flip → three covers → their M6 bolts ─┤ (parallel workstreams)
└─ stage/insert three shaft modules into Casing_Base ──────────┘
            ↓
mate Casing_Top to Casing_Base → upright assembly
            ├─ six M10 bolt→nut pairs
            └─ Breather_Plug + two Oil_Level_Indicators
```

Closure is the main occlusion point: shaft/gear inspection and internal access must occur before `Casing_Top` is mated unless the COMMUTABLE geometry retains an explicit visible/access corridor. Each M6 bolt depends on its assigned cover. Each M10 nut depends on its same-numbered bolt.

## Full actual assembly order

Groups marked parallel/interchangeable may be reordered only where the DAG permits it.

| Order | Actor | Exact part(s) and destination | Operation and reason |
|---:|---|---|---|
| 1 | Human | whole scene | Verify all registered components, clear work areas, and accept the initial poses. This provides a recoverable baseline. |
| 2 | Human | `Casing_Base` | Focus, confirm pose, and flip to expose the base-cover face. |
| 3 | Robot | `Hub_Cover_Output_Base` → `Casing_Base/socket_hub_output` | Align cover normal and seat the output cover. |
| 4 | Robot | `Hub_Cover_Small_Base_01` → `socket_hub_small_1` | Seat first small base cover; interchangeable with steps 3/5. |
| 5 | Robot | `Hub_Cover_Small_Base_02` → `socket_hub_small_2` | Seat second small base cover. |
| 6 | Human/robot | `M6_Hub_Bolt_01_base`, `_02_base`, `_03_base`, `_04_base` → numbered `Casing_Base` sockets | Secure `Hub_Cover_Output_Base`; insertion/fastening contact is scientifically useful but is execution-gated on the M6 collider repair. |
| 7 | Human/robot | `M6_Hub_Bolt_05_base`, `_06_base`, `_07_base`, `_08_base` → numbered sockets | Secure `Hub_Cover_Small_Base_01`. |
| 8 | Human/robot | `M6_Hub_Bolt_09_base`, `_10_base`, `_11_base`, `_12_base` → numbered sockets | Secure `Hub_Cover_Small_Base_02`, then unfocus the base. |
| 9 | Human | `Casing_Top` | Focus, confirm pose, and flip to expose its cover face; may run alongside base preparation. |
| 10 | Robot | `Hub_Cover_Input_Top` → `Casing_Top/socket_hub_input` | Seat the input-side cover. |
| 11 | Robot | `Hub_Cover_Output_Top` → `socket_hub_output` | Seat the output-side cover. |
| 12 | Robot | `Hub_Cover_Small_Top` → `socket_hub_small` | Seat the small top cover. |
| 13 | Human/robot | `M6_Hub_Bolt_01_top`, `_02_top`, `_03_top`, `_04_top` → numbered sockets | Secure `Hub_Cover_Output_Top`. |
| 14 | Human/robot | `M6_Hub_Bolt_05_top`, `_06_top`, `_07_top`, `_08_top` → numbered sockets | Secure `Hub_Cover_Input_Top`. |
| 15 | Human/robot | `M6_Hub_Bolt_09_top`, `_10_top`, `_11_top`, `_12_top` → numbered sockets | Secure `Hub_Cover_Small_Top`, then unfocus the top. |
| 16 | Human | `Casing_Base`, `Input_Shaft`, `Output_Shaft`, `Transfer_Shaft` | Refocus/check poses and apply only the conditional flips indicated by the pose checks. |
| 17 | Robot | all three shaft modules → their `Casing_Base/socket_gear_*` | Stage 0.15 m above the sockets simultaneously; this creates a controlled, visible pre-insertion state. |
| 18 | Robot | `Input_Shaft` → `socket_gear_input` | Insert the input module along casing Z. Its integral pinion must phase with `Transfer_Gear`. |
| 19 | Robot | `Output_Shaft` → `socket_gear_output` | Insert the output module; `Output_Gear` must remain seated on the shaft and phase with `Transfer_Gear`. |
| 20 | Robot | `Transfer_Shaft` → `socket_gear_transfer` | Insert the transfer module and verify both intended gear contacts. The generated DAG permits the three insertions as one simultaneous group; the pilot executes them serially for attributable contact logs. |
| 21 | Human then robot | inspect gear train; `Casing_Top` → `Casing_Base/socket_casing_mate` | Human confirms internal state; robot lowers the top with the Y-180 mating orientation. This is the irreversible/occluding transition for an episode. |
| 22 | Human | assembled `Casing_Base` | Upright and stabilize the housing before through-fastening. |
| 23 | Robot/human | `M10_Casing_Bolt_01` → `Casing_Top/socket_bolt_casing_1`, then `M10_Casing_Nut_01` → bolt | Install the first paired fastener; bolt precedes its nut. |
| 24 | Robot/human | corresponding `_02` bolt then `_02` nut | Same primitive at socket 2; independent of other completed pairs. |
| 25 | Robot/human | corresponding `_03` bolt then `_03` nut | Same at socket 3. |
| 26 | Robot/human | corresponding `_04` bolt then `_04` nut | Same at socket 4. |
| 27 | Robot/human | corresponding `_05` bolt then `_05` nut | Same at socket 5. |
| 28 | Robot/human | corresponding `_06` bolt then `_06` nut | Same at socket 6. |
| 29 | Robot | `Breather_Plug` → `Casing_Base/socket_breather` | Align along the side-port axis and seat on its shoulder; this interface has a passing collision-on test. |
| 30 | Robot | `Oil_Level_Indicator_01` → `socket_oil_1`; `Oil_Level_Indicator_02` → `socket_oil_2` | Insert into the two side ports. These are interchangeable after casing mate. |
| 31 | Human | complete gearbox | Verify cover flushness, shaft freedom, gear response, fastener seating, accessory seating, and end the episode. |

## Twelve matched pilot pairs

Both domains in a pair show the same successful A→B demonstration, use the same exact part names, cameras, motions, prompt, and completed final relations. A single visible local geometry parameter on an existing named part changes B→A feasibility. Geometry variants retain the same registry identifier; no answer-bearing names or colors are exposed.

| Pair | Demonstrated A→B with actual components | HARD condition | COMMUTABLE condition | Ground-truth decision and scientific value |
|---|---|---|---|---|
| ACC-01 | A: insert `Input_Shaft` into `Casing_Base`; B: insert `Transfer_Shaft` | `Transfer_Shaft`/`Transfer_Gear` occupies the only remaining swept input-module corridor | a visible local relief in `Casing_Base` preserves that corridor | Is B→A collision-free? Tests access reasoning across two geared shaft modules. |
| ACC-02 | A: insert `Output_Shaft`; B: insert `Transfer_Shaft` | `Transfer_Gear` blocks the output module's reverse swept path | a visible output-bore lead-in/relief preserves the path | Same final three-dimensional relations; different reverse accessibility. |
| ACC-03 | A: mount `Output_Gear` on `Output_Shaft`; B: insert the output module into `Casing_Base` | casing walls enclose the gear-mounting end after shaft insertion | the output opening remains wide enough for later `Output_Gear` placement | Distinguishes gear/shaft mating knowledge from replay of the demonstrated order. |
| ACC-04 | A: insert `Output_Shaft`; B: seat `Hub_Cover_Output_Base` | the cover seals the only output-module insertion port | a visible opposite-side corridor through the existing casing geometry remains open | Tests whether the model recognizes closure of an insertion port. |
| FAS-01 | A: mate `Casing_Top` to `Casing_Base`; B: insert `M10_Casing_Bolt_01` | the bolt path becomes continuous only when casing lugs align | a visible captive recess in `Casing_Top` retains the same bolt before closure without interference | Closure/through-bolt constraint using an existing bolt and housing. |
| FAS-02 | A: insert `M10_Casing_Bolt_01`; B: seat `M10_Casing_Nut_01` | the nut has no retaining seat before the bolt is present | a visible captive pocket in the existing `Casing_Top` retains `M10_Casing_Nut_01` | Tests bolt/nut dependency rather than semantic fastening priors. |
| FAS-03 | A: mate `Casing_Top`; B: insert `M10_Casing_Bolt_02` | socket 2 is a closed through-bore created by aligned halves | socket 2 is an open-sided retaining slot that permits bolt preload | Replicates closure logic at another location without changing part identity. |
| FAS-04 | A: seat `Hub_Cover_Output_Top`; B: insert `M6_Hub_Bolt_01_top` | the bolt cannot be aligned/retained before the cover is seated | a visible captive feature in `Hub_Cover_Output_Top` retains the same bolt | Tests local cover-fastener ordering; collect only after the M6 collider passes. |
| TST-01 | A: inspect `Input_Shaft` pinion ↔ `Transfer_Gear`; B: mate `Casing_Top` | opaque `Casing_Top` removes all views of that mesh | a transparent region in the same `Casing_Top` preserves the view | Can inspection occur after closure? Tests visual access, not action replay. |
| TST-02 | A: rotate `Input_Shaft` and verify `Output_Shaft` response; B: mate `Casing_Top` | closure removes actuation/observation access | existing exposed shaft ends remain accessible after closure | Tests functional causality through the real gear train. |
| TST-03 | A: gauge `Output_Gear` ↔ `Transfer_Gear` backlash; B: mate `Casing_Top` | casing blocks the gauge corridor | a visible local port in `Casing_Top` preserves that corridor | Tests tool-path access while holding the named gears and final state constant. |
| TST-04 | A: verify `Bearing_Output_Top`, `Bearing_Output_Bottom`, and `Output_Shaft` seating; B: mate `Casing_Top` | all seating references are occluded | a sight region on the same `Casing_Top` preserves the seating references | Tests whether closure destroys evidence needed for a correct inspection decision. |

For each pair, the HARD feasible terminal set is `{A→B}` and the COMMUTABLE set is `{A→B, B→A}`. `A only` and `B only` remain failures because the declared task requires both named relations. Calibration must place the intervention at least twice the measured simulator/evaluator tolerance away from the feasibility threshold.

## Episode procedure and actor protocol

1. Reset to the named initial scene and load one anonymous geometry variant.
2. Human performs the specified inspection/pose confirmation or hands the named component into the robot pickup region. Robot performs controlled align/approach/insert primitives. Actor, speeds, camera path, and timing are identical within a pair.
3. Record one successful A→B demonstration. Never show B→A in the canonical input.
4. Independently reset and execute A→B, B→A, A-only, and B-only with the physics/visibility/function oracle. Pair labels are frozen only after the registered reliability threshold is met.
5. Ask the three existing questions: demonstrated order, whether B can precede A, and feasible terminal procedures. Omit pair ID and evaluator-only fields from model requests.

The human may stop a run for unsafe contact, a dropped part, or an incorrectly posed fixture. Recovery is episode-level reset; no manual correction after contact is counted as success.

## Reset, success, failure, and logging

Reset uses `SequenceRunner.reset()` plus scene reload when geometry variants change. `test_instance_playback.py` proves reset clears pose-check state and grouped undo reverses simultaneous operations; `test_seat_undo.py` proves flip/upright/put-down round trips. Before each episode verify exact loose-part poses, zero completed DAG steps, and unmodified variant identity.

Physical success requires the named child to reach its registered relation and seated pose without collision disabling, snap after contact, unintended penetration, excessive force, or human correction. Gear tasks additionally require free coherent rotation/response; inspection tasks require a registered camera/tool ray; covers and casing require flush contact; fasteners and accessories require their shoulder/seat criterion.

Failures include blocked insertion path, jam, drop, wrong socket, gear not seated or not transmitting motion, cover/casing gap, bolt or nut not retained, plug/indicator not seated, lost inspection visibility, wrong next-action decision, or unnecessary intervention.

Reuse existing streams: RGB/video, robot state, issued actions, episode/DAG timing, pose-check results, actor and interaction labels. Add only per-pair oracle outputs needed by the existing pilot schemas: minimum clearance/contact, final relative pose, collision/force events, rotation transfer or ray visibility, and failure reason. Do not build a second logging framework.

## Pilot go/no-go gates

- All 24 domains preserve A→B and visibly expose the one changed mechanism.
- For each domain and four terminal procedures, run the existing plan's 20-trial perturbation batch: A→B ≥95% in both variants; B→A ≥95% COMMUTABLE and ≤5% HARD; A-only/B-only fail the declared task 100%.
- M6 pair FAS-04 remains no-go until the casing fastener holes support controlled collision-on insertion.
- ACC and TST gear pairs remain no-go until gear-to-gear contact/rotation and shaft-module insertion basins are measured.
- Reviewers must agree the physical intervention is visible without labels; public manifests must contain no evaluator-only keys.

This sequence is intentionally diverse but bounded: shaft insertion, gear placement/meshing, housing closure, cover seating, fastener/nut dependency, accessory insertion, and functional/visibility checks all use the integrated gearbox's real named components.
