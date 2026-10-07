# REPAIR M0 implementation status

This repository now contains a minimal runnable `hrc_repair` package that
implements the plan's public contracts and control-flow skeleton:

```text
reset → public observation → one registered skill → result →
help handoff (epoch invalidation) → fresh observation → replan → finish/stop
```

The package reuses the existing strict Isaac launcher through the
`IsaacSubprocessAdapter` and the persistent `IsaacPersistentWorkerAdapter`.
It never writes a part pose during a skill and it
keeps `gt_trajectory.jsonl` separate from public observations.  The fast
`configs/repair_m0_contract_smoke.yaml` backend is only a protocol test; it is
not physics evidence.

## Current physical boundary

The current Isaac M1 preplace controller is still a scripted physical skill,
not an ACT or REPAIR policy.  The latest single-arm probes have real dynamic
Hub/Casing contact and a real release attempt, but do not pass the calibrated
M1 6-DoF/bolt-seat acceptance.  The M0 route is intentionally **single-arm
left-gripper**; an earlier dual-arm probe showed zero right-gripper Hub contact
and is retained as a negative control rather than being counted as a dual-arm
success.  The Isaac scene has an optional real colliding target cuboid
controlled by `HRC_M1_BLOCKED=1`.

The persistent-worker backend (`--backend isaac_inprocess`) now keeps one
Isaac stage alive across the blocked place, helper handoff, and post-help place.
It does not reset or rewrite Hub/robot poses between those actions.  The older
subprocess evidence remains separately labeled
`blocked_physical_reset_per_skill`; it is not mixed into the persistent result.

`feedback_mode=privileged_debug` is deliberate for this first integration:
the observer reads the strict runner's private metrics and exposes only a
tri-state public result.  A sensor-only observer must replace it before a
formal model comparison.

## 2026-10-02 Isaac physical M0 runs

The reproducible candidate configuration is
[`configs/repair_m0_physical_candidate.yaml`](../configs/repair_m0_physical_candidate.yaml).
It uses the existing strict Isaac subprocess launcher, real gravity and
physical staging supports, a left-arm real-contact grasp, a bounded closed-loop
preplace descent, release, and stable retract.  The preplace branch does not
use the full pick-and-carry detour: the object is already above the registered
socket, so the controller descends from the measured grasp frame.  No Hub pose
write, weld, grasp joint, or kinematic attachment is used after reset.

### Nominal

[`runs/repair_m0_physical_candidate_nominal_fix3_20261002/metrics.json`](../runs/repair_m0_physical_candidate_nominal_fix3_20261002/metrics.json)
is a successful `REPAIR_M0_PLACEMENT_V1` run.  All six placement checks pass:
support contact, radial error, axial error, release contact-free, bounded
release motion, and stable retract.  Measurements are radial `3.45 mm`, axial
`16.49 mm`, release motion `5.76 mm`, and post-settle retract drift `9.91 mm`.
The live three-camera artifact is
[`rollout.mp4`](../runs/repair_m0_physical_candidate_nominal_fix3_20261002/place_01/rollout.mp4)
(a symlink to `inner_wall_probe.mp4`), with `video_error=null` and 1219 frames.

The earlier `fix1` and `fix2` artifacts are retained as negative controls.  The
old high-Z preplace detour produced repeatable 35.2/45.6 mm errors; changing
only staging-pad collision handling produced 28.0/28.2 mm errors.  These are
not silently overwritten.

### Blocked → help → continue

[`runs/repair_m0_physical_candidate_blocked_fix3_20261002/metrics.json`](../runs/repair_m0_physical_candidate_blocked_fix3_20261002/metrics.json)
records a real blocked scenario.  The first episode used a colliding target
blocker and failed placement (Hub–Casing force `0 N`, radial error `45.5 mm`,
axial error `114.4 mm`).  The event log then records `help_001` removing only
the registered blocker, increments the control epoch, and starts a second
place action.  The second physical episode passes the same six placement
checks; `post_help_continuation_success=true`, `help_requests=1`, and
`place_attempts=2`.  Its live-camera videos are under `place_01/` (blocked)
and `place_02/` (post-help).

This is a valid M0 protocol/physics demonstration, but the adapter limitation
is explicit: helper recovery currently starts a fresh Isaac subprocess reset;
it is not yet an in-process handoff preserving the same dynamic state.  The
artifact records this as `blocked_physical_reset_per_skill` rather than hiding
it.  The planner is the deterministic `rule_dev_only` protocol exerciser in
these runs, not an LLM result.

## 2026-10-03 persistent blocked→help→place result

The current reproducible M0 configuration is
[`configs/repair_m0_physical_candidate.yaml`](../configs/repair_m0_physical_candidate.yaml).
It uses a real dynamic Hub, full gravity, physical staging supports, a real
single-left-gripper grasp, a registered colliding `arm_edge` blocker, and one
Isaac worker for the complete episode.  The first place is guarded before
blocker contact (`physical_contact=false`) so the helper handoff does not
inject an uncontrolled collision impulse; the blocker is nevertheless a
physical scene object and is moved only by the registered helper command.

The no-video physics acceptance is
[`runs/repair_m0_persistent_blocked_segmented_release_clear_20261003/metrics.json`](../runs/repair_m0_persistent_blocked_segmented_release_clear_20261003/metrics.json).
It reports `task_success_gt=true`, `pipeline_success=true`,
`post_help_continuation_success=true`, and `placement_score.success=true`.
All six M0 placement checks pass: support contact, radial error, axial error,
release contact-free, bounded release motion, and stable retract.  Measured
radial error is 4.78 mm, axial error 17.26 mm, release motion 5.67 mm, and
post-settle retract drift 10.21 mm.  The stricter assembly verdict remains
`CONTACTED_SEAT_NOT_STABLE`; no 6-DoF or bolt-hole claim is made.

The matching live-camera artifact is
[`rollout.mp4`](../runs/repair_m0_persistent_blocked_final_camera_20261003/place_01/rollout.mp4),
backed by [`inner_wall_probe.mp4`](../runs/repair_m0_persistent_blocked_final_camera_20261003/place_01/inner_wall_probe.mp4).
The metrics record 1,229 synchronized 480×360 live-camera frames with
`video_error=null`; the MP4 is non-empty and has an ISO MP4 container.  The
event log records `blocked_guarded_attempt` → `blocked_waiting_help` →
`helper_applied` → `post_help_resume` in the same worker episode.

The final release fix is explicit and reproducible: the arm withdraws in ten
measured 20 mm physics segments, then records `release_clearance` only after
the jaws have actually left the annulus.  This avoids the failed one-shot IK
retract and does not relax placement tolerances or teleport the Hub.

The unguarded-contact negative is retained at
[`runs/repair_m0_persistent_blocked_arm_edge_20261003/`](../runs/repair_m0_persistent_blocked_arm_edge_20261003/):
allowing the arm-edge blocker to collide with the held cover produces a real
contact/ejection failure.  It is not used as a success claim.  The guarded
route is the accepted M0 handoff because it keeps the robot in a safe hold
before that destructive collision, while still requiring the helper and the
robot to continue in the same live episode.

## Commands

```bash
python3 -m hrc_repair.preflight --config configs/repair_m0.yaml --strict
python3 -m hrc_repair.run --config configs/repair_m0.yaml --method repair --scenario nominal --seed 100
python3 -m hrc_repair.run --config configs/repair_m0_contract_smoke.yaml --method repair --scenario blocked --seed 0

# Persistent same-episode blocked→help→place physics baseline.
M1_M0_BLOCKER_PROFILE=arm_edge M1_M0_PRECONTACT_GUARD=1 \
python3 -m hrc_repair.run --config configs/repair_m0_physical_candidate.yaml \
  --backend isaac_inprocess --method repair --scenario blocked --seed 100 \
  --out runs/repair_m0_persistent_blocked_final_camera_20261003
```

The first physical command may take several minutes because it starts the
Isaac container and advances 0.01 s PhysX steps.  Each episode gets its own
directory under `runs/repair_m0/`; failures are retained.

## 2026-10-05 strict pure REPAIR planner reproduction

The accepted no-leakage result is documented in
[`repair_m0_pure_reproduction_20261005.md`](repair_m0_pure_reproduction_20261005.md)
and stored at
[`runs/repair_m0_pure_qwen_blocked_20261005_r5/`](../runs/repair_m0_pure_qwen_blocked_20261005_r5/).
It uses `configs/repair_m0_pure.yaml`, the local Qwen3-VL-2B HTTP planner,
`public_contact` observations, and the persistent Isaac worker.  The actual
planner trace is `place → help → place → finish`; the physical evaluator and
the protocol both pass (`task_success_gt=true`, `pipeline_success=true`,
`online_status=DONE`).

The accepted run has an opaque episode ID and exposes neither private metrics,
poses, scores, nor filesystem paths to the planner.  Earlier runs remain
explicitly excluded when they used privileged/rule fallback, an ambiguous
prompt, or leaked scenario/path metadata.  This is still an M0 preplace
placement result: it does not claim full pick-and-carry, ACT/VLA control,
6-DoF bolt-seat alignment, or RGB-only sensing.
