# M1 Task Contract

## Task identity

- Task: seat the red `Hub_Cover_Output_Top` onto `Casing_Top/socket_hub_output`.
- Manual step: Step 2 in `Gearbox assembly.pdf`.
- Source sequence: `step_14_hub_cover_output_top` in `assembly/gearbox_sequence.yaml`.
- Expanded instance: `inst_025_combine_hub_cover_output_top` in `assembly/instances/canonical_grouped.yaml`.
- Interface trial: `Hub_Cover_Output_to_Casing_Top` in `assembly/physics_validation.json`.
- Dynamic part: `/World/Hub_Cover_Output_Top`, asset `assets/parts/Hub Cover Output.usd`.
- Fixed target: `Casing_Top`, asset `assets/parts/Casing Top.usd`.
- Mating relation: `plug_main → socket_hub_output`.
- M6 fastening is out of scope for M1.

## Episode reset

The reset must create a dedicated station containing the R1 robot articulation and cameras from the pinned RoCo runtime, a fixed/static `Casing_Top`, and one dynamic loose output hub cover. Other gearbox pieces are omitted from the station before the episode begins. The cover is not attached and no parent/reparent relation may be created during an episode.

The reset records the nominal fixture pose and verifies that the hub entrance faces the robot, the cover is on the calibrated table/grasp region, and both are within the calibration envelope. Any fault profile is selected before reset and is recorded only in evaluator-private data.

## Coordinate contract

All public positions are metres, rotations are quaternions in `(w, x, y, z)` order, and action deltas are expressed in the robot base frame unless the adapter says otherwise. The target transform is:

```text
T_world_socket = T_world_casing × T_casing_socket
a_world = R_world_socket × a_local_hub
preinsert = socket_origin - sign * clearance * a_world
```

`sign`, clearance, insertion depth, radial tolerance and tilt tolerance must be filled from the calibration manifest after the RoCo composed-stage regression. The isolated trial's `axis=2` is not itself a world-axis contract.

## Agent/evaluator boundary

Agent-visible data is limited to public RGB/depth frames, robot qpos/velocities, gripper state, documented public contact/force data if present, task ID, prior skill results and human messages. The agent never receives evaluator live pose, contact labels, penetration values, fault cause or private success flags.

The evaluator alone may read simulator ground truth and computes seating depth, mating-frame error, radial error, tilt, penetration/contacts and post-release stability. A VLM verdict is tri-state (`SUCCESS`, `FAILED`, `UNKNOWN`) and cannot by itself claim millimetre-level physical seating.

## Allowed actions

The planner may emit only `OBSERVE`, `PICK`, `MOVE_TO_PREINSERT`, `GUARDED_INSERT`, `RELEASE_RETRACT`, `VERIFY`, `ASK_HUMAN`, `REPLAN`, or `ABORT`. It may not emit joint arrays, arbitrary poses, USD paths, Python/shell, teleport, snap, reparent, or Magic Assembly calls.

## Human handoff contract

Control ownership is exactly one of `AUTO`, `HUMAN`, `SAFE_STOP`. A handoff requires `TAKE_CONTROL`; return requires `RETURN_CONTROL_AND_DONE` or `ABORT`. The human may jog the held cover to correct alignment but must not complete the final seat. Returning control invalidates the previous insertion trajectory and forces a new observation and plan.
The implementation makes this explicit as `REOBSERVE → UPDATE_MEMORY → REPLAN`; the pre-handoff decision is cleared before the fresh observation is accepted.
