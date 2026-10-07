# M1 Hub-Cover output-top policy rollout

## Scope

Task: `step_14_hub_cover_output_top` (`inst_025_combine_hub_cover_output_top`).
The dynamic child is `Hub_Cover_Output_Top`; the fixed parent is `Casing_Top`;
the calibrated interface is `plug_main -> socket_hub_output`.

The direct baseline uses the existing RoCo/Isaac Lab R1 control stack, but
replaces the official gearbox objects with the M1 USD parts. It is a scripted
oracle made of measured Cartesian waypoints and closed-loop state feedback; it
does not write the Hub pose after reset, make the Hub kinematic, or create a
grasp joint. It must not be described as a learned-policy result.

## Reproducible direct baseline

Configuration: [`configs/m1_direct_policy_success.json`](../configs/m1_direct_policy_success.json)

Launcher:

```bash
M1_GPU_DEVICE=0 \
M1_DIRECT_OUTPUT_DIR=/absolute/path/to/output \
./tools/run_m1_direct_policy_success.sh
```

The tested run is
[`validation_logs/m1_direct_hold_release_drop_20260930/metrics.json`](../validation_logs/m1_direct_hold_release_drop_20260930/metrics.json).
Its measured final state is:

- final Hub-root error: `0.00413 m` (target tolerance `0.01 m`);
- final Hub speed: `< 1e-6 m/s`;
- final Hub--Casing contact force: `55.9 N`;
- both finger--Hub filtered contacts after release: `0 N`;
- verdict: `SEAT_RELEASE_CANDIDATE`.

The release record itself is intentionally not treated as the seat: the Hub is
released above the socket, then gravity is enabled and the final stable contact
is measured. The final orientation is within about 9 degrees of the calibrated
MagicAssembly orientation. The static USD fidelity and physics-manifest checks
remain separate evidence; this rollout is the dynamic contact test.

## Visualization

[`state_trace_rollout.mp4`](../validation_logs/m1_direct_hold_release_drop_20260930/state_trace_rollout.mp4)
is a generated state-trace visualization (top/side schematic from the measured
Isaac states), not a camera-RGB recording. It shows the reset, radial
one-inside/one-outside grasp, lift, closed-loop transport, release, gravity
drop, and final seated state. It is encoded as H.264 `yuv420p` for common
player compatibility. The JSON metrics are the physical source of truth.

The first full-resolution attempt was stopped while diagnosing an overly
frequent RTX refresh; it is not used as evidence.  The corrected run below is
the camera artifact to use.

The completed live-camera run is
[`inner_wall_probe.mp4`](../validation_logs/m1_direct_camera_policy_success_20261001_final/inner_wall_probe.mp4).
It was captured during the same Isaac/PhysX rollout, not rendered from the
state-trace JSON afterward.  It contains three RoCo-order RGB views
(`head | left wrist | right wrist`) at `320x240` per camera, concatenated to
`960x240`, encoded as H.264 `yuv420p`, with 941 decoded frames at 20 fps.
The RTX sensors refresh every 10 control steps and the latest live frame is
repeated between refreshes; physics and actions are not changed.  The companion
metrics report has `video_frame_count=941`, `video_error=null`, and
`insertion_verdict=SEAT_RELEASE_CANDIDATE`.

Reproduce it with:

```bash
M1_GPU_DEVICE=1 \
M1_DIRECT_CAMERA_OUTPUT_DIR=/absolute/path/to/output \
M1_CAMERA_UPDATE_STRIDE=10 M1_RENDER_INTERVAL=5 \
M1_CAMERA_WIDTH=320 M1_CAMERA_HEIGHT=240 \
M1_STEP_SCALE=0.5 M1_RELEASE_OPEN_STEPS=1000 M1_GRAVITY_SETTLE_STEPS=100 \
./tools/run_m1_direct_camera_policy_success.sh
```

This launcher follows RoCo's camera flag, three-sensor ordering, `imageio`
writer, and `H.264/yuv420p` encoding.  Use `M1_CAMERA_UPDATE_STRIDE=1` when
fresh sensor data on every control step is required; it is substantially slower
for this high-resolution CAD scene.

For a visible Isaac Sim window on a workstation with X11, set
`M1_HEADLESS=0`; the launcher passes `DISPLAY`, `XAUTHORITY`, and the X11
socket into the pinned container.

## RoCo learned-policy attempt

The official RoCo repository contains ACT (Action Chunking with Transformers),
not a model or module named “ACL”. The adapter therefore uses the pinned ACT
checkpoint already present at
`/home/sunsiliang/roco_runtime/checkpoints/roco_model_act_2/` and preserves its
strict hashes. The public checkpoint has 83,923,087 parameters and receives the
three RGB cameras plus the 14-D joint state. Its output is mapped exactly from
`[L arm6, L grip, R arm6, R grip]` to the M1 environment's
`[L arm6, R arm6, L grip, R grip]` absolute joint targets.

Launcher:

```bash
ROCO_GPU_DEVICE=0 ROCO_M1_ACT_STEPS=1 \
./roco_single_run/run_m1_act_episode.sh
```

Test artifact:
[`roco_single_run/runs/m1_act_smoke_seed23_final/result.json`](../roco_single_run/runs/m1_act_smoke_seed23_final/result.json)

The 20-step learned-policy smoke completed with valid finite actions and the
expected hashes. The Hub stayed at its reset pose throughout these 20 actions;
ACT did not reach or grasp the Hub. This is a pipeline/adapter smoke, not an M1
task-success claim. A longer ACT run is expected to require M1-specific
training or a checkpoint trained on this visual/geometry domain; no action was
replaced by a rule policy when testing ACT.

## Current conclusion

The earlier direct run is retained as a scripted-oracle/candidate artifact,
not as proof of a physically valid assembly: it released above the socket and
did not satisfy the final 6DoF orientation check. The new strict Isaac runner
(`tools/run_m1_isaac_strict_success.sh`) keeps gravity enabled from reset,
uses physical staging supports, requires one-inner/one-outer tip contact,
transports while closed, and scores the result with the six-point rubric in
[`m1_isaac_roco6_scoring.zh-CN.md`](m1_isaac_roco6_scoring.zh-CN.md). Only a
`SUCCESS`/6-of-6 artifact is a complete trajectory. The learned RoCo ACT path
 remains a separate adapter smoke and is not included in this strict rollout.

The current best Isaac/PhysX placement route is a **hybrid dual-arm route**:
both arms synchronously lift the Hub under gravity, the right arm opens and
withdraws vertically after the lift, and the left arm remains closed for the
transport, socket insertion, release, and stable retract. This is still a
scripted Cartesian/IK oracle; ACT is not loaded and no ground-truth Hub pose is
fed back to an action model.

The completed fast camera run is
[`inner_wall_probe.mp4`](../validation_logs/m1_dual_reset_y0_sync_hybrid_camera_fast_20261003/inner_wall_probe.mp4),
with metrics at
[`metrics.json`](../validation_logs/m1_dual_reset_y0_sync_hybrid_camera_fast_20261003/metrics.json).
It contains 1,349 live-camera frames at `480x360`, and the MP4 is a valid
ISO-BMFF file. The physical-place rubric is `4/4`: dynamic supported reset,
load-bearing lift, finite transport/socket seat, and release/stable retract.
This is a complete physical **placement** result, but not yet a strict
assembly result: the six-point score is `2/6` in the camera run because tip
topology is disabled for the lower-overhead video run, transport contact is
reported, and final orientation/bolt alignment remain outside the strict
tolerances. The matching no-video strict-point run reports
`orientation_error_deg=3.51` and `bolt_hole_alignment_max_error_m=0.00565`;
therefore it must not be described as a 6-DoF bolt-ready success.

The controlled post-lift orientation ablation is retained at
[`validation_logs/m1_dual_reset_y0_sync_hybrid_postlift_orient_20261003/metrics.json`](../validation_logs/m1_dual_reset_y0_sync_hybrid_postlift_orient_20261003/metrics.json).
It attempted to rotate the dynamically held Hub while it was still above the
fixture, using ordinary IK and no pose write. The Hub slipped during the
rotation (lift after close `0.0564 m` versus left link-6 motion `0.1408 m`),
and the run ended at the source pose with `INSERTION_NOT_VERIFIED` and a
physical-place score of `2/4`. This is a negative control result: the
remaining yaw error cannot be fixed by simply twisting the frictional
one-inner/one-outer pinch in mid-air.

The source Hub mesh was also checked independently. Its four output-cover
bolt-hole centers are approximately `(x,z)=(±40,±40)` authored units; under
the canonical scale and rotation this agrees with the evaluator's four local
points to sub-millimetre-to-millimetre precision. The strict failure is
therefore currently a measured held-seat orientation/settling issue, not
evidence that the CAD file has missing bolt holes.

The attempted partial-release variant is also retained at
[`validation_logs/m1_dual_reset_y0_sync_hybrid_partial_release_20261003/metrics.json`](../validation_logs/m1_dual_reset_y0_sync_hybrid_partial_release_20261003/metrics.json).
It opened the right gripper after the supported lift while leaving the left
gripper partially open (`release_opening=0.03`) so that socket contact and
gravity could settle the Hub. This did not improve the route: the left
inner/outer contact was lost during the lift (Hub lift `0.0564 m` versus left
link-6 motion `0.1408 m`), the Hub remained at the source pose, and the result
was `INSERTION_NOT_VERIFIED`, `CONTACT_WITHOUT_HELD_FOLLOW`, physical score
`2/4`, and RoCo-style score `1/6`. The evidence supports keeping the
full-load-bearing left pinch closed until the pre-insert release; it does not
support calling partial opening a physically successful placement.

A separate contact-material sensitivity run at Hub↔Casing static/dynamic
friction `0.15/0.10` is at
[`validation_logs/m1_dual_reset_y0_sync_hybrid_lowseatfriction_20261003/metrics.json`](../validation_logs/m1_dual_reset_y0_sync_hybrid_lowseatfriction_20261003/metrics.json).
It was intentionally isolated from the nominal `0.45/0.35` baseline. The
lower interface friction caused the cover to slip during the lift (`2/4`
physical score, `CONTACT_WITHOUT_HELD_FOLLOW`) before it reached the Casing;
it is therefore not an alignment fix. The friction values are now explicit
run parameters so a later material-calibration experiment can be reproduced
without changing the nominal result.

## Reproducibility audit (2026-10-03)

The earlier `m1_dual_reset_y0_sync_hybrid_final_strict_20261003` result is
retained as a historical 4/4 physical-placement artifact, but it is not a
stable strict baseline. Three fresh, separately named runs identify the
current blocker:

* [`m1_dual_reset_y0_sync_m1friction0_reanchorseat_nohold_grip108_p30_20261003`](../validation_logs/m1_dual_reset_y0_sync_m1friction0_reanchorseat_nohold_grip108_p30_20261003/metrics.json)
  reached pre-insert with a real left pinch, then lost the Hub at the first
  seat segment with Hub–Casing friction `0/0`.
* [`m1_dual_reset_y0_exact_old_baseline_20261003`](../validation_logs/m1_dual_reset_y0_exact_old_baseline_20261003/metrics.json)
  restored nominal `0.45/0.35` Hub–Casing material and reproduced lift and
  transport, but the 40-step pre-insert hold dropped the Hub back to source;
  the later contact was mis-seated (`orientation_error_deg=38.66`, bolt-hole
  max error `0.1442 m`).
* [`m1_true_dual_reanchor_followlink_20261003`](../validation_logs/m1_true_dual_reanchor_followlink_20261003/metrics.json)
  kept both R1 arms closed through seat and refreshed both measured grasp
  frames. It still lost the Hub during transport (`ROCO 3/6`, physical 2/4).

Position-only and fixed-waypoint IK ablations are also preserved in
`m1_dual_reset_y0_positiononly_seat_20261003` and
`m1_dual_reset_y0_fixedik_baseline_20261003`; the former loses object-follow
during lift and the latter fails to establish finger contact. These results
do not change the acceptance rule: M1 is complete only when one run has
physical placement 4/4, strict 6-DoF pose and bolt alignment, contact-free
release/retract, and (for a video claim) a live-camera MP4 with
`video_error=null`.

## Single-arm versus dual-arm evidence

The first M1 baseline was **single-arm (left arm)**. It established that the
left fingers can make a load-bearing radial inner/outer-wall grasp and carry
the Hub, but its final seat/orientation was not strict-success. The current
best complete placement controller is the **hybrid dual-arm route** described
above. It uses both arms only for the supported lift, then removes the right
arm before transport because keeping both arms closed caused the right inner
finger to collide with the Casing during insertion.

The original dual-arm lift-only ablation is retained at
[`validation_logs/m1_lift_only_dual_friction10_20261002/metrics.json`](../validation_logs/m1_lift_only_dual_friction10_20261002/metrics.json):
it predates the synchronized hybrid route and is not used in the current
success claim.

The single-arm lift-only calibration
[`validation_logs/m1_lift_only_disable_support_friction10_fix_20261002/metrics.json`](../validation_logs/m1_lift_only_disable_support_friction10_fix_20261002/metrics.json)
does show a real dynamic lift with both left finger contacts and a Hub/link6
lift of roughly `0.13 m`; it is a grasp/lift check, not placement success.

The failed full dual-arm routes are retained in the
`m1_dual_reset_y0_full_release_transport*` directories; their failure was a
right-finger/Casing collision during insertion. The successful physical
artifact is now the hybrid M1 placement run above, while the strict
6-DoF/bolt-ready M1 claim remains open. The separate M0 preplace placement
run is documented in [`reports/repair_m0_status.md`](repair_m0_status.md).

## Follow-up contact-stability probes (2026-10-03)

The latest two controlled ablations further isolate the handoff failure:

* [`m1_dual_sync_correction_vertical_orientation_20261003`](../validation_logs/m1_dual_sync_correction_vertical_orientation_20261003/metrics.json)
  added one closed-loop preinsert correction, re-anchored the measured grasp
  frame, and attempted an in-air orientation correction before a vertical-only
  seat.  The orientation correction ejected the dynamic Hub; the final root
  error was about `0.265 m` and the physical score remained `2/4`.  This is a
  negative control and is not a placement claim.
* [`m1_dual_sync_correction_vertical_no_orientation_20261003`](../validation_logs/m1_dual_sync_correction_vertical_no_orientation_20261003/metrics.json)
  removed that rotation and retained only the preinsert correction plus
  vertical-only descent.  It preserved the dynamic lift and reached casing
  contact, but the segment-14 right-arm handoff was not repeatably stable: the
  final root error was `0.0976 m`, orientation error `29.51°`, and physical
  score `2/4`.  The result is `INSERTION_NOT_VERIFIED`.

These runs explain the current runtime and status: both arms are simulated
under PhysX during the lift/transport, but the nominal M1 controller is still
a scripted IK oracle (not ACT).  The remaining work is contact-stable handoff
and 6-DoF seating, not model inference.

The subsequent probes were run to answer why the current M1 route is slow and
why it is not yet a strict success:

* [`m1_dual_hybrid_freeze_preinsert10_seat_jacobian_transport020_20261003`](../validation_logs/m1_dual_hybrid_freeze_preinsert10_seat_jacobian_transport020_20261003/metrics.json)
  restored the known `0.20 m` transport clearance and held the measured arm
  joints for ten physics steps at preinsert. The Hub still slipped back to its
  staging pose before the seat (`2/4` physical, `4/6` RoCo-style). Joint
  freezing is therefore not a valid grasp-preservation solution.
* [`m1_true_dual_sync_transport_seat_jacobian_20261003`](../validation_logs/m1_true_dual_sync_transport_seat_jacobian_20261003/metrics.json)
  enabled an explicit synchronized dual-transport controller: both wrist IK
  targets are computed from one measured state and applied in one physics step.
  It avoided the old left-then-right update ordering during transport, but the
  final seat became unstable (`2/4` physical, `3/6` RoCo-style), with large
  contact impulses. This is retained as a negative control, not a success.
* [`m1_hybrid_m1friction0_vertical_noreanchor_20261003`](../validation_logs/m1_hybrid_m1friction0_vertical_noreanchor_20261003/metrics.json)
  used zero Hub–Casing friction and vertical-only final descent without
  reanchoring. The Hub lost the grasp before seat and never contacted the
  socket (`2/4`, `4/6`).

These runs confirm that the current bottleneck is real contact/IK stability at
the Casing interface, not ACT inference. Strict M1 remains open; the old
release-before-seat artifact must not be presented as a closed-gripper place.

## Historical-route replays (2026-10-03)

Two parameter-audit replays explain why the earlier 4/4 artifact is not yet a
stable baseline. [`m1_repro_historical_hybrid_20261003`](../validation_logs/m1_repro_historical_hybrid_20261003/metrics.json)
omitted the historical `transport-direct` route and used the default Hub reset
Y; it developed an 81 kN contact impulse and numerical divergence. The direct
route replay [`m1_repro_historical_hybrid_direct_20261003`](../validation_logs/m1_repro_historical_hybrid_direct_20261003/metrics.json)
still used the default reset Y, so the right arm never reached the Hub.

The fully aligned replay [`m1_repro_historical_hybrid_direct_y0_20261003`](../validation_logs/m1_repro_historical_hybrid_direct_y0_20261003/metrics.json)
did establish both inner/outer dual-arm contacts and a real lift, but the
right-arm handoff produced 74 mm release drift; the result was
`INSERTION_NOT_VERIFIED`, physical `2/4`, orientation error `55.6°`, and bolt
alignment error `0.158 m`. These are retained as reproducible negative
artifacts. They show that the current issue is the physical handoff/held-pose
stability, not a missing ACT checkpoint or a simulator hang.

## Latest near-seat diagnostic (2026-10-04)

[`m1_single_true_gravity_nearseat_release008_20261004`](../validation_logs/m1_single_true_gravity_nearseat_release008_20261004/metrics.json)
was intentionally run with `preinsert_height=0.08 m` to test whether a
shorter gravity-assisted release would improve seating. It did not. The cover
was already inside the collision-sensitive socket corridor before release;
opening the left jaw generated an approximately `8.0 kN` filtered gripper
contact impulse, after which PhysX diverged (`insert_root_error≈95.99 m`,
`orientation_error≈118.97°`). The run is a failed numerical/contact probe,
not a placement result. The next replay therefore restores the historical
`0.14 m` preinsert height and hybrid dual-arm handoff while retaining
full-gravity physics.

## Follow-up true-gravity parameter probes (2026-10-04)

The parameter-audit runs below are retained separately from the accepted M0
placement artifact and do not close the M1 strict gate:

* [`m1_truegravity_historical_hybrid_replay_20261004`](../validation_logs/m1_truegravity_historical_hybrid_replay_20261004/metrics.json)
  was a deliberate replay with the wrong default reset-Y. It is a negative
  configuration control (`hub_gravity=true`, but the Hub was initialized at
  the historical staging offset rather than `M1_HUB_RESET_Y=0`).
* [`m1_truegravity_dual_handoff_above_y0_20261004`](../validation_logs/m1_truegravity_dual_handoff_above_y0_20261004/metrics.json)
  corrected the reset-Y and used synchronized dual lift/transport, but the
  right-arm handoff and contact impulses destabilized the seat (`2/4`
  physical, insertion root error about `113 mm`, orientation about `51.4°`).
* [`m1_single_true_gravity_freeze_release_20261004`](../validation_logs/m1_single_true_gravity_freeze_release_20261004/metrics.json)
  held the measured arm joints while opening the gripper. The Hub was lost
  during the controlled seat before release, so the freeze option cannot be
  counted as a successful place.
* [`m1_single_true_gravity_extra025_freeze_release_20261004`](../validation_logs/m1_single_true_gravity_extra025_freeze_release_20261004/metrics.json)
  added 25 mm controlled-seat depth and failed at the same pre-release
  handoff, confirming that extra depth alone does not fix it.
* [`m1_single_true_gravity_pos1_offset_freeze_release_20261004`](../validation_logs/m1_single_true_gravity_pos1_offset_freeze_release_20261004/metrics.json)
  tested a small `(+1°, +1°)` grasp correction. It made the contact unstable:
  after a `402.9 N` seat contact the released Hub received roughly `19.2 kN`
  impulse and diverged (`final_root_error≈147 mm`, orientation about
  `115.5°`). It is a negative control.

The closest true-gravity route remains
[`m1_single_true_gravity_notilt_offset_freeze_release_20261004`](../validation_logs/m1_single_true_gravity_notilt_offset_freeze_release_20261004/metrics.json):
it carries the dynamic Hub with one left gripper, reaches casing contact,
opens without a large release drift, and retracts stably. Its pre-release
pose is still outside the strict 6-DoF/bolt tolerances (`orientation≈6.60°`,
bolt max error≈22.4 mm), and the final released pose is logged separately
(`orientation≈2.36°`, root error≈5.2 mm, bolt max error≈7.2 mm). It is therefore
the current single-arm diagnostic candidate, not a strict M1 success.

The follow-up contact-after-seat yaw correction
[`m1_single_true_gravity_notilt_postseat_yawfix_20261004`](../validation_logs/m1_single_true_gravity_notilt_postseat_yawfix_20261004/metrics.json)
is also a negative control. Although the correction was applied only after
the Hub reached casing contact, it increased release drift to `37.9 mm` and
left the final orientation error at about `6.23°`; physical placement remained
`2/4`. The controller therefore does not yet have a validated strict M1 seat
trajectory.

The no-tilt replay with an additional 5 s post-release physics dwell
[`m1_single_true_gravity_notilt_offset_postsettle500_20261004`](../validation_logs/m1_single_true_gravity_notilt_offset_postsettle500_20261004/metrics.json)
did not change the final pose: the Hub remained supported with final root
error `5.23 mm`, axial error `5.18 mm`, orientation error `2.36°`, and bolt
alignment max error `7.17 mm`. This shows the residual is a repeatable
geometric/seat calibration error rather than an insufficient settle time.

## Geometry and seat-depth audit (2026-10-04)

The source socket authoring script still contains the historical output-bore
fit at `y=39.75 mm`, while the composed MagicAssembly registry and M1
calibration manifest use the audited override `y=43.42 mm` (the composed
dynamic-root target is `0.08683718 m`). A read-only inspection of the actual
`Casing Top.usd` vertices found the open top-surface region centered at
`43.42 mm` with a nearest boundary of about `46.76 mm`; the `39.75 mm` query
has a different boundary profile (about `49.72 mm`). The old fit is therefore
not interchangeable with the composed-stage target. No USD geometry was
changed in this audit.

The independent collision-on probe reports a free gravity-seat root offset of
about `0.067636 m`, but that is a *result* of unconstrained contact, not a
command target for a clamped robot. The single-arm full-gravity replay
[`m1_single_true_gravity_calibrated_depth0676_20261004`](../validation_logs/m1_single_true_gravity_calibrated_depth0676_20261004/metrics.json)
set the controlled-seat target to that value while keeping all other settings
equal to the closest no-tilt route. It reached contact (`insert_root_error
≈12.7 mm`, orientation `≈7.57°`) but drove the cover too deeply for the
clamped configuration: release drift was `≈2.69 m` and the post-release
state numerically diverged (`final_root_error≈5905.6 m`). This is a negative
control. The controller must approach the physical support from above and
let contact settle; it must not force the free-seat depth while gripping.

At the start of the latest full-gravity single-arm route, the Hub already
acquires a roughly `7.5°` pose error during the segmented lift (before it
reaches the Casing). This identifies the next controller experiment as
grasp-frame/orientation stabilization, not another socket-center change. The
follow-link-orientation replay is stored separately at
[`m1_single_true_gravity_followlink_seat062_20261004`](../validation_logs/m1_single_true_gravity_followlink_seat062_20261004/metrics.json)
and remains subject to the same strict `4/4` physical and six-DoF acceptance
rules.

The compliant-gripper replay
[`m1_single_true_gravity_compliant_gripper_20261004`](../validation_logs/m1_single_true_gravity_compliant_gripper_20261004/metrics.json)
lowered the R1 gripper actuator from the authored `25,000/1,000` stiffness/
damping to `4,000/200` with a `60 N` effort cap. It reduced peak early contact
force, but the jaws no longer established a centered stable clamp: the Hub
already rotated about `28.6°` by insertion and the release/retract impulses
were still large (`≈7.6 kN` and `≈24.5 kN`). Physical score remained `2/4`.
This excludes a simple “make the gripper softer” fix; the original stiff
actuator is needed for the load-bearing lift, while the grasp geometry itself
still needs redesign or a mechanically stable two-sided contact.

The safe-plane small-orientation correction replay
[`m1_single_true_gravity_small_orientation_correction_20261004`](../validation_logs/m1_single_true_gravity_small_orientation_correction_20261004/metrics.json)
used sixteen ordinary IK segments with a maximum `0.5°` wrist correction per
segment. It still lost the dynamic Hub during the correction/approach and
returned to its supported reset (`insert_root_error≈0.265 m`, no Hub–Casing
contact, physical `2/4`). Thus even a bounded high-Z rotation is not a valid
way to repair the current frictional grasp. The next experiment changes only
the gripper actuator compliance, leaving the mesh, gravity, target frame, and
strict evaluator unchanged.

The completed follow-link replay
[`m1_single_true_gravity_followlink_seat062_20261004`](../validation_logs/m1_single_true_gravity_followlink_seat062_20261004/metrics.json)
is a negative control: refreshing the wrist orientation from the instantaneous
link state during lift/transport caused a large attitude excursion instead of
stabilizing the part (`insert_root_error≈86.9 mm`, orientation `≈84.4°`,
physical `2/4`). It did preserve release/retract numerically, but it never
reached the socket pose. The controller should therefore keep the measured
grasp orientation during translation and use a separate, bounded correction
stage only when the Hub is safely above the socket.

## Dual-arm synchronized route timeout (2026-10-04)

The run [`m1_dual_true_gravity_sync_clearance018_20261004_r2`](../validation_logs/m1_dual_true_gravity_sync_clearance018_20261004_r2/TIMEOUT.md)
started a real full-gravity two-arm route with synchronized lift and
transport, a right-arm release at the insertion-above waypoint, and no pose
write or fixed grasp constraint. Isaac remained CPU-active for approximately
40 minutes but produced no `metrics.json`; it was stopped at the explicit
wall-time limit. This is an execution-timeout/convergence negative, not a
physical success or a scored physical failure. It confirms that the current
dual-arm path is too slow to serve as the next MVP diagnostic without first
shortening the route or disabling expensive telemetry.

## Dual-arm lift and transport ablations (2026-10-04)

The short synchronized-lift probe
[`m1_dual_true_gravity_sync_lift_only_20261004`](../validation_logs/m1_dual_true_gravity_sync_lift_only_20261004/metrics.json)
is a valid grasp diagnostic: with full gravity and a dynamic Hub, both R1
arms reported nonzero contact on all four gripper links, peak reported
contact was about `89.9 N`, and the Hub followed the wrists by
`0.13817 m`. It used no pose write and no grasp constraint, but it stopped
before insertion, so it is not a placement result.

The complete synchronized route with both arms kept through transport
[`m1_dual_true_gravity_sync_fast_place_20261004`](../validation_logs/m1_dual_true_gravity_sync_fast_place_20261004/metrics.json)
scored `2/4`: the Hub reached a casing contact impulse but the insertion root
error was `221.4 mm` and orientation error `68.2°`; the right gripper reached
about `5.7 kN`. Releasing the right arm after the lift and transporting with
the left arm (`m1_dual_sync_lift_left_transport_20261004`) returned the Hub to
its supported reset after an unstable insertion attempt (`2/4`). Keeping the
right jaw open in place (`m1_dual_sync_lift_right_release_in_place_20261004`)
and using a direct high-Z route
(`m1_dual_sync_lift_direct_left_transport_20261004`) did not fix the pose:
the best direct route had `7.6 mm` radial but `114.4 mm` axial error and
`22.6°` orientation error, with multi-kN contact impulses. Increasing only
gripper friction to `10/8`
(`m1_dual_sync_lift_high_friction_direct_20261004`) still scored `2/4` and
ended at `140.5 mm` axial and `61.9°` orientation error.

These runs isolate the current boundary: the two-arm grasp itself is
physically load-bearing, while the available IK transport/release path is not
yet a stable six-DoF assembly controller. `M1_Z_OFFSET_M` is now exposed in
the strict launcher so future grasp-frame tests remain reproducible; no USD
geometry or asset collision was changed by these ablations.

The final single-arm replay of the historical transport branch
[`m1_single_true_gravity_graspcorrect_direct_20261004`](../validation_logs/m1_single_true_gravity_graspcorrect_direct_20261004/metrics.json)
used full gravity, dynamic supports, friction `10/8`, and the historical
`-4°/-5°` grasp correction. It reproduced neither a strict seat nor a valid
place: score `2/4`, insertion root error `86.1 mm`, radial error `35.1 mm`,
axial error `78.6 mm`, orientation error `42.3°`, and bolt alignment error
`128.5 mm`. The old historical run that scored transport used a different
non-dynamic Hub state and is therefore not a valid M1 result.

## Full-gravity reproduction audit (2026-10-04)

The current source was replayed with the historical hybrid parameters in
[`m1_reproduce_hybrid_strict_current_20261004/metrics.json`](../validation_logs/m1_reproduce_hybrid_strict_current_20261004/metrics.json).
This is an actual full-gravity run (`hub_gravity=true`, physical supports,
dual synchronous lift, one-inner/one-outer tip topology), but it is **not** a
placement success: the physical-place score is `2/4`. The lift evidence is
valid; the dynamic Hub reaches the transport path with four active gripper
contacts. During the first controlled-seat segment it is displaced to a
mis-seated pose (`insert_root_error_m=0.09136 m`, orientation error `38.66°`,
bolt-hole error `0.14416 m`).

The matched no-hold ablation is
[`m1_reproduce_hybrid_nohold_20261004/metrics.json`](../validation_logs/m1_reproduce_hybrid_nohold_20261004/metrics.json).
Removing the 40-step preinsert hold does not solve the problem: the Hub is
lost at the first seat segment and returns to the source support (`2/4`, root
error `0.26526 m`). This isolates the blocker to the final dynamic seat
controller/contact path, not to the hold dwell. Neither run is promoted to a
strict M1 trajectory.

The subsequent high-friction replay
[`m1_single_tiltcorr05_friction10_fullgravity_20261004`](../validation_logs/m1_single_tiltcorr05_friction10_fullgravity_20261004/metrics.json)
restored the load-bearing `10/8` gripper friction, but retained the in-air
orientation correction. It held through transport and reached the preinsert
stage, then the correction/controlled-seat action ejected the Hub back to the
source support (`2/4`, `4/6`, insertion not verified). This is evidence
against rotating the loaded single-arm grasp immediately before seating.

The first release-clearance replay
[`m1_single_releaseclearance_fullgravity_20261004`](../validation_logs/m1_single_releaseclearance_fullgravity_20261004/metrics.json)
disabled that rotation but increased each insertion segment to 60 steps. The
longer dwell changed the contact trajectory: the Hub slipped during the
horizontal preinsert path and never reached the socket (`2/4`, insertion not
verified). It is not a placement result; the next run restores the previously
stable 30-step insertion segments while retaining the explicit release-clearance
waypoint. The 30-step replay
[`m1_single_releaseclearance30_fullgravity_20261004`](../validation_logs/m1_single_releaseclearance30_fullgravity_20261004/metrics.json)
restored the transport path and reached the socket (`insert` was about
`16.5 mm` from the target with real Hub--Casing contact). However, translating
the whole opened tool by `+120 mm` in Y to clear the outer finger swept the
inner finger across the annulus and ejected the Hub to its source support.
The run therefore remains `2/4`, `4/6`, and is not a placement success. The
next release test uses a short vertical clearance instead of a lateral sweep.

The vertical-clearance replay
[`m1_release_verticalclearance08_fullgravity_20261004`](../validation_logs/m1_release_verticalclearance08_fullgravity_20261004/metrics.json)
confirmed the same failure mode: even an upward `80 mm` TCP clearance while a
residual outer-finger contact remained pulled/ejected the Hub to the source
support. It remains `2/4`, `4/6`, and is not a valid placement. The release
controller must therefore let the seated geometry clear the tool through the
ordinary final retract; it must not insert an intermediate TCP clearance
waypoint.

The no-hold free-seat-depth replay
[`m1_single_calibrateddepth0676_nohold_fullgravity_20261004`](../validation_logs/m1_single_calibrateddepth0676_nohold_fullgravity_20261004/metrics.json)
also failed: forcing `seat_depth=0.06763625 m` reached contact but produced
about `26 mm` release drift and returned the Hub to the source support. The
free-gravity rest height is therefore not a valid clamped-seat command target.

## Single-arm tilt-correction replay (2026-10-04)

The first no-hold replay with the bounded `0.5°` orientation-correction
stage is recorded in
[`m1_single_tiltcorr05_fullgravity_20261004/metrics.json`](../validation_logs/m1_single_tiltcorr05_fullgravity_20261004/metrics.json).
It used one R1 arm, full gravity, physical supports, ordinary IK actions, and
no Hub pose writes. The initial one-inner/one-outer tip topology was observed,
but the low-friction `4/3` gripper lost held-follow during the horizontal
preinsert approach. The Hub returned to its supported source state before
the socket, giving physical-place `2/4`, RoCo-style `2/6`,
`candidate_verdict=CONTACT_WITHOUT_HELD_FOLLOW`, and
`insertion_verdict=INSERTION_NOT_VERIFIED`. The correction stage therefore
did not constitute a placement attempt or success. The next replay changes
only the gripper friction back to the previously load-bearing `10/8` setting
and uses eight insertion segments.

## Single-arm opening-width replay (2026-10-04)

The bounded `10 mm` grasp-opening test is recorded in
[`m1_single_opening010_fullgravity_20261004/metrics.json`](../validation_logs/m1_single_opening010_fullgravity_20261004/metrics.json).
It is a genuine full-gravity, physical-supports run with one R1 arm and a
dynamic Hub; no Hub pose or velocity was written. The one-inner/one-outer tip
topology and load-bearing lift were observed (peak gripper contact about
`166.7 N`, lift about `137.8 mm`), and the transport stage completed. The
larger opening changed the final grasp frame, however: at the controlled seat
the Hub was `58.7 mm` from the socket target, with about `33.2°` orientation
error and `122.2 mm` maximum bolt-hole error. Releasing therefore left the
Hub unsupported and it returned to the source support (`2/4` physical-place,
`4/6` RoCo-style, `INSERTION_NOT_VERIFIED`). This is a negative grasp-frame
ablation, not a successful trajectory; the nominal `5 mm` opening remains the
better single-arm candidate.

## Preinsert gravity-release replay (2026-10-04)

The nominal `5 mm` grasp was also tested with `--release-at-preinsert` at a
world-Z preinsert height of `1.090 m` (about `28 mm` above the calibrated
socket root):
[`m1_single_release_at_preinsert090_fullgravity_20261004/metrics.json`](../validation_logs/m1_single_release_at_preinsert090_fullgravity_20261004/metrics.json).
The pick, one-inner/one-outer contact, lift, and finite transport were real,
but opening before the controlled seat was unstable. The Hub acquired a
nonphysical-looking high-speed collision state immediately after release
(`17.7 m/s`, then `66.7 m/s`) and left the fixture; it ended at the source
support. The run scored `2/4` physical-place and `4/6` RoCo-style with
`INSERTION_NOT_VERIFIED`. It is therefore rejected as a placement trajectory;
the remaining route must keep the part supported by the gripper until the
socket-seat action is complete, then perform a physically clear release.

## Small grasp-orientation correction replay (2026-10-04)

The bounded `(-1.0°, +0.5°)` grasp-frame correction was tested in
[`m1_single_graspcorr_m1_p05_fullgravity_20261004/metrics.json`](../validation_logs/m1_single_graspcorr_m1_p05_fullgravity_20261004/metrics.json).
It retained full gravity, physical supports, a dynamic Hub, and ordinary
single-arm IK actions. The pick/lift/transport stages remained valid, but the
correction did not cancel the seat tilt (`6.66°` at `insert`, `27.8 mm` max
bolt-hole error). The Hub then lost support during release (`256.1 mm` drift)
and returned to the source support. The run scored `2/4` and `4/6`; it is not
a placement success. This rules out that small correction direction/magnitude
as the next controller change.

## Dual-arm reanchor replay (2026-10-04)

The current dual-arm hybrid replay is recorded in
[`m1_dual_hybrid_reanchor_above_current_20261004/metrics.json`](../validation_logs/m1_dual_hybrid_reanchor_above_current_20261004/metrics.json).
Both R1 arms performed the initial grasp and synchronized lift under full
gravity with physical supports; the right arm then released and the left arm
continued the transport. The run recorded one-inner/one-outer grasp topology,
about `0.143 m` lift, and finite transport, so the first four grasp/transport
checks were not the blocker. However, the Hub was reset to its supported source
state before the insertion check (`insert_root_error≈0.265 m`, no Hub--Casing
contact), giving physical-place `2/4`, RoCo-style `4/6`, and
`INSERTION_NOT_VERIFIED`. Re-anchoring the insertion frame at the above-bore
waypoint did not rescue the current dynamics. This is a negative dual-arm
controller replay, not a successful rollout.

## Controlled-seat position-only replay (2026-10-04)

The valid baseline-matched position-only ablation is recorded in
[`m1_dual_hybrid_seat_position_only_exact_20261004/metrics.json`](../validation_logs/m1_dual_hybrid_seat_position_only_exact_20261004/metrics.json).
It uses the same reset (`y=0`), gripper friction (`10/8`), transport clearance
(`0.20 m`), dual-arm lift, right-arm handoff, and `4×30/8×30` action timing as
the current hybrid route; only the final controlled-seat IK is switched to
position-only. Grasp and lift remain valid (`0.143 m` Hub lift, `43/43`
transport-contact samples), but the Hub returns to its source support during
the final descent (`insert` root `[0.3000, 0.0000, 1.07995]`, no verified
Hub--Casing seat). The run scores physical-place `2/4`, RoCo-style `4/6`, and
`INSERTION_NOT_VERIFIED`. Position-only IK therefore does not fix the current
seat failure; the next change must address the preinsert grasp frame or the
contact/seat trajectory rather than merely changing the final IK mode.

## Controlled-seat full-XYZ replay (2026-10-04)

The matching full-XYZ replay is recorded in
[`m1_dual_hybrid_full_xyz_seat_exact_20261004/metrics.json`](../validation_logs/m1_dual_hybrid_full_xyz_seat_exact_20261004/metrics.json).
It restores ordinary pose IK for the controlled seat while keeping the same
full-gravity, physical-support, dual-arm lift and right-arm handoff baseline.
The Hub was transported with `43/43` contact samples, but the preinsert state
still had roughly `95.8 N` Hub--Casing contact and about `22 mm` lateral error.
The first controlled seat segment ejected the Hub back to its source support.
The run therefore remains physical-place `2/4`, RoCo-style `4/6`, and
`INSERTION_NOT_VERIFIED`. Switching between vertical-only, position-only,
Jacobian-position, and full-XYZ seat control has not removed this failure.

## Higher-preinsert and right-arm handoff replays (2026-10-04)

Raising the preinsert waypoint to `0.20 m` above the socket is recorded in
[`m1_dual_hybrid_preinsert020_exact_20261004/metrics.json`](../validation_logs/m1_dual_hybrid_preinsert020_exact_20261004/metrics.json).
It reduced the measured Hub--Casing force at preinsert to about `5.7 N`, but
the first seat segment still ejected the dynamic Hub; the result is again
`2/4`, `4/6`, and `INSERTION_NOT_VERIFIED`. Thus lower preinsert contact by
itself is not sufficient evidence of a stable grasp or a valid seat.

The attempted dual-support handoff in
[`m1_dual_hold_to_above_release_right_exact_20261004/metrics.json`](../validation_logs/m1_dual_hold_to_above_release_right_exact_20261004/metrics.json)
kept both arms on the Hub until above the socket and released the right arm
there. It is rejected as an unsafe route: only `23/43` transport samples
contained the Hub, the preinsert speed was about `0.176 m/s`, and transient
contact forces reached approximately `21 kN` and `8.9 kN`. It scored physical
place `2/4`, RoCo-style `3/6`, and did not verify insertion. The large force
spikes are a collision/hand-off failure, not proof of successful two-arm
placement.

## Historical no-hold replay with closed-gripper seat (2026-10-04)

To isolate the long preinsert dwell, the current runner was replayed with the
historical nominal parameters and `preinsert_hold_steps=0`:
[`m1_dual_hybrid_nohold_exact_current_20261004/metrics.json`](../validation_logs/m1_dual_hybrid_nohold_exact_current_20261004/metrics.json).
This is still a real two-arm lift followed by a right-arm withdrawal and a
left-arm closed-gripper seat; it uses full gravity, physical supports, no Hub
pose writes, and the `10/8` gripper material. The hold change did not rescue
the route. At preinsert the Hub was already about `53 mm` radially from the
socket center with about `95.8 N` Hub--Casing contact; the first seat segments
generated multi-kN gripper loads and the Hub remained outside the socket
(`insert_root_error≈66.7 mm`, `release_drift≈107.6 mm`). The result is
physical-place `2/4`, RoCo-style `4/6`, and `INSERTION_NOT_VERIFIED`.
This rules out the preinsert dwell alone as the cause; the remaining blocker
is the collision-sensitive grasp frame/approach, not simply waiting too long.

## Compensated zero-tilt dual-arm replay (2026-10-04)

The run
[`m1_arm_effort_compensated_zero_tilt_20261004`](../validation_logs/m1_arm_effort_compensated_zero_tilt_20261004/metrics.json)
kept the high-effort actuator overrides and shifted the commanded preinsert
X by `+6 mm`, but removed the historical `-4°/-5°` initial wrist correction.
The dynamic Hub was genuinely grasped and lifted (`≈141 mm` displacement with
both left-finger contacts), but the left-arm transport became physically
unreachable at the second side-above segment. The Hub then left the valid
scene (`insert_root_error≈1.367 m`, orientation≈180°), giving physical-place
`2/4`, RoCo-style `2/6`, and `INSERTION_NOT_VERIFIED`. This is a negative
controller/reachability ablation; the X compensation is not interpreted as a
validated calibration until a route remains physically stable through the
socket approach.

## High-effort single-arm zero-tilt replay (2026-10-04)

[`m1_single_high_effort_zero_tilt_offset_20261004`](../validation_logs/m1_single_high_effort_zero_tilt_offset_20261004/metrics.json)
used the zero-tilt single-arm route with run-scoped arm effort/stiffness
overrides (`200/1600/150/10`) and the `+15/-5 mm` preinsert offset that was
stable with the authored actuator settings. The dynamic Hub did acquire a
real two-sided pinch, but the stronger arm controller diverged during the
transport (Hub speed exceeded `200 m/s`, and the final root error exceeded
`16 km`). It scored physical-place `2/4`, RoCo-style `3/6`, and
`INSERTION_NOT_VERIFIED`. This is a negative actuator-compliance ablation;
the next route restores the authored arm controller and changes only the
controlled-seat depth.

## Single-arm y=0 release-settle replay and final-placement score (2026-10-04)

The reproducible baseline was rerun with the reset explicitly fixed at
`(0.30, 0.00, 1.08) m`, `0.20 m` transport clearance, no wrist correction,
the authored arm controller, `10/8` gripper friction, vertical-only controlled
seat, measured preinsert re-anchoring, and `500` post-release physics steps:
[`m1_single_y0_notilt_postsettle500_score_20261004/metrics.json`](../validation_logs/m1_single_y0_notilt_postsettle500_score_20261004/metrics.json).
This is a single-left-arm trajectory. It is not an ACT rollout and it does not
write the Hub pose or use a hidden grasp constraint.

The dynamic Hub was lifted by `137.5 mm` and remained in finite, load-bearing
contact for all `43/43` transport samples. The strict evaluator still reports
`CONTACTED_SEAT_NOT_STABLE`, physical-place `2/4`, and RoCo-style `4/6`: the
pre-release snapshot is `6.60°` from the canonical orientation and its maximum
bolt-hole error is `22.42 mm`. Those strict checks remain failed.

Because the released dynamic part settles after that snapshot, the run also
emits a separate, explicitly relaxed final-placement diagnostic:
`M1_POST_RELEASE_PLACEMENT_4_POINT_V1 = 4/4`. At the final post-settle state,
radial error is `0.77 mm`, axial error `5.18 mm`, orientation error `2.36°`,
maximum bolt-hole error `7.17 mm`, and post-settle drift `10.85 mm`; all gripper
contact forces are zero at retract and the Hub remains supported by the Casing.
This `4/4` means *released and stably placed on the socket* under the stated
relaxed thresholds. It is not strict M1 success, not bolt-ready alignment, and
not evidence of a learned policy. The strict score and this final-placement
diagnostic are intentionally kept separate.

## Single-arm y=0 live-camera replay (2026-10-04)

The live RTX-camera replay is complete at
[`m1_single_y0_notilt_postsettle500_camera_20261004/inner_wall_probe.mp4`](../validation_logs/m1_single_y0_notilt_postsettle500_camera_20261004/inner_wall_probe.mp4),
with matching metrics at
[`metrics.json`](../validation_logs/m1_single_y0_notilt_postsettle500_camera_20261004/metrics.json).
The MP4 contains `3061` frames, uses H.264/yuv420p, is about `6.7 MB`, and
reports `video_error=null`. It is a live three-camera recording (head, left
hand, right hand) with a refresh every ten control steps; the last frame is
not a synthetic render or a pose-only visualization. Its physical metrics
match the no-video replay: strict `2/4`, RoCo-style `4/6`, and final-placement
diagnostic `4/4`.

The camera artifact is therefore a valid visualization of the scripted,
full-gravity single-arm trajectory. It still does not turn the run into an ACT
rollout or into strict bolt-ready assembly.

## Release/contact and orientation correction ablations (2026-10-04)

Three controlled ablations were run after the `4/4` final-placement baseline:

* [`m1_single_y0_notilt_release_clearance020_20261004`](../validation_logs/m1_single_y0_notilt_release_clearance020_20261004/metrics.json)
  moved the opened wrist upward by `20 mm` before recording the clearance.
  It cleared contact only at the very end, but dragged/ejected the dynamic Hub
  during that motion (`final_root_error≈1.169 m`, post-placement score `2/4`).
  It is rejected; vertical clearance is not a safe release primitive here.
* [`m1_single_y0_notilt_release_open080_20261004`](../validation_logs/m1_single_y0_notilt_release_open080_20261004/metrics.json)
  commanded an `80 mm` gripper opening. The actual R1 gripper joint did not
  reach that target (the release snapshot remained around `0.033 m` and still
  carried about `94 N`), and the final Hub drifted about `112 mm`. This is a
  negative actuator-limit result, not evidence that the physical part cannot
  be released.
* [`m1_single_y0_notilt_correct_before_seat_step1_20261004`](../validation_logs/m1_single_y0_notilt_correct_before_seat_step1_20261004/metrics.json)
  applied a `1°`-per-segment wrist correction before the axial seat. The
  correction moved the Hub laterally and increased the pre-release error
  (`radial≈23.7 mm`, orientation≈`8.0°`, maximum bolt error≈`41.0 mm`), so it
  is also rejected. The ordinary no-correction route remains the best physical
  placement candidate.

These experiments leave the strict M1 blocker explicit: the released dynamic
part can settle on the socket, but the current R1 grasp/controller combination
does not yet provide a stable, contact-free, bolt-aligned pre-release state.

## Single-arm y=0 release-open wait ablation (2026-10-04)

The release-wait hypothesis was tested without changing the grasp, seat target,
or object pose. The run
[`m1_single_y0_notilt_release_wait300_20261004`](../validation_logs/m1_single_y0_notilt_release_wait300_20261004/metrics.json)
held the arm joints fixed while commanding the same `0.05 m` opening for `300`
physics steps, then allowed `500` post-release settle steps. It completed under
full gravity with physical supports and the single-left-arm route.

The extra wait did not change the strict pre-release snapshot: it still reports
`6.595°` orientation error, `15.82 mm` axial error, and `22.42 mm` maximum
bolt-hole error. The snapshot recorded at the beginning of the open command
still has about `111.5 N` on one finger, so the strict evaluator remains
`CONTACTED_SEAT_NOT_STABLE`, physical-place `2/4`, and RoCo-style `4/6`.
This is not evidence that the part is physically impossible to release: at the
later retract sample both finger contacts are `0 N`, Hub speed is effectively
zero, and the final casing support force is about `55.9 N`.

The separate relaxed final-placement diagnostic is `4/4`: final radial error
`0.687 mm`, axial error `4.728 mm`, orientation error `2.132°`, maximum
bolt-hole error `6.836 mm`, and post-settle retract drift `11.28 mm`. Thus the
longer open wait confirms a stable ordinary place candidate, but it does not
repair the strict pre-release alignment/release gate and is not ACT or learned
policy evidence.

## High-waypoint orientation-correction ablation (2026-10-04)

The next controller hypothesis was tested in isolation at
[`m1_single_y0_notilt_correct_above_step1_20261004`](../validation_logs/m1_single_y0_notilt_correct_above_step1_20261004/metrics.json):
before entering the socket corridor, the held wrist was rotated toward the
canonical Hub orientation in eight bounded `1°` increments. All other values
were copied from the stable single-arm baseline, including explicit `y=0`
reset, `0.20 m` transport clearance, vertical-only seat, frozen-joint release,
300 opening steps, and 500 settle steps.

This was a negative physical result, not a failed metric implementation. The
first divergent evidence appears during the high-waypoint/preinsert transition:
the preinsert sample reaches about `9.65 kN`/`3.72 kN` on the two fingers and
the Hub speed rises to about `1.02 m/s`; the controlled-seat sample reaches
about `13.14 kN`/`5.94 kN`. The Hub then leaves the valid placement route:
strict score `3/6`, physical-place `2/4`, relaxed final-placement `2/4`, final
radial error `136.46 mm`, final orientation error `178.40°`, and final support
force `60.1 kN`. The high-waypoint wrist correction is therefore rejected and
must not replace the stable no-correction baseline.

## Final-seat position-only ablation (2026-10-04)

The stable single-arm route was rerun with only the final-seat position-DLS
override enabled:
[`m1_single_y0_notilt_seat_position_only_20261004`](../validation_logs/m1_single_y0_notilt_seat_position_only_20261004/metrics.json).
The hypothesis was that removing orientation tracking during the axial seat
would let contact passively correct the Hub tilt. It did the opposite. The
first divergence is at `insert`: Hub speed is about `23.9 m/s`, one filtered
contact reaches about `99.1 kN`, and Hub–Casing contact is absent. During
release/retract the speed grows to about `171/504.6 m/s`; final radial error is
`55.62 m` and final orientation error `112.12°`. Physical-place and relaxed
final-placement scores are both `2/4`. Position-only final-seat control is
rejected; the full-pose no-correction baseline remains the only stable route.

## Dual-arm post-seat orientation-correction ablation (2026-10-04)

The historical dual-arm hybrid route was the closest strict-placement
candidate: both arms lifted the dynamic Hub, the right arm withdrew after the
lift, and the left arm reached the socket with pre-release orientation error
`3.51°` and maximum bolt-hole error `5.65 mm`. To test whether that residual
could be removed after socket contact, the run
[`m1_dual_postseatcorr35_20261004`](../validation_logs/m1_dual_postseatcorr35_20261004/metrics.json)
changed only `--seat-yaw-correction-after-seat-deg 3.5`; gravity, collision,
contact, dual-lift, and acceptance settings were otherwise retained.

This was a negative physical result. The left load-bearing pinch was not
maintained through the socket approach (`held_contact_and_follow=false` and
only `15/43` transport samples retained contact). The Hub left the valid
scene, yielding physical-place `2/4` and relaxed post-placement `1/4`.
Post-seat wrist rotation is rejected; it does not show that the CAD interface
is impossible. The uncorrected dual route remains a historical near-candidate,
not a strict M1 success.

## Small lateral release-clearance ablation (2026-10-05)

The next release hypothesis changed only the opened-gripper clearance: after
the same full-gravity single-arm route opened the fingers to `0.05 m`, the
controller moved the measured arm frame by `+5 mm` in Y before the release
snapshot and then performed the ordinary retract. The run is
[`m1_single_y0_notilt_release_clearance005_20261005`](../validation_logs/m1_single_y0_notilt_release_clearance005_20261005/metrics.json).
No Hub pose, velocity, collision mode, or evaluator threshold was changed.

This also failed physically. Pick/lift and transport remained valid (`43/43`
transport samples retained the one-inside/one-outside contact topology), but
the insert snapshot was still outside the strict seat gate: axial error
`15.82 mm`, orientation error `6.595°`, and maximum bolt-hole error
`22.42 mm`. The 5 mm clearance reduced the measured remaining finger contact
at the clearance sample to about `75.2 N`, but did not make the release
contact-free. During retract the Hub was dragged out of the assembly, with
`1.339 m` retract drift and final radial error `0.745 m`; relaxed final
placement was only `2/4`. The strict result is `CONTACTED_SEAT_NOT_STABLE`,
RoCo-style `4/6`, physical-place `2/4`, and no video was requested for this
diagnostic. A small lateral move is therefore not a valid release primitive.

## Post-seat closed-grasp hold ablation (2026-10-05)

The stable single-arm baseline was rerun with a new physically grounded
controller option, `--post-seat-hold-steps 300`. After the dynamic Hub reached
the socket, the measured robot joint positions were held with the gripper
closed for 300 physics steps before the release sample. This does not write
the Hub pose or relax collision; it tests whether contact and gravity can
settle the part while still held. The artifact is
[`m1_single_y0_notilt_postseat_hold300_20261004`](../validation_logs/m1_single_y0_notilt_postseat_hold300_20261004/metrics.json).

The hypothesis was rejected. The hold reduced the insert speed to
`0.00162 m/s`, but the pre-release orientation error was `7.01°` (baseline
`6.60°`) and the maximum bolt-hole error was `23.27 mm`. Opening then left a
large residual finger contact (`≈4.94 kN`), moved the Hub off the socket, and
ended with physical-place `2/4` and relaxed final-placement `2/4`. Extra
closed-grasp settling therefore does not repair the strict alignment gate and
is not part of the nominal route.

## Integrated adapter smoke recheck (2026-10-04)

The current `hrc_m1.run` entry point was independently rechecked in the pinned
Isaac Lab 2.3.0 container with `--backend roco --smoke --smoke-steps 3`, seed
`1201`, and cameras enabled. The run
[`m1_integrated_smoke_recheck_20261004`](../runs/m1_integrated_smoke_recheck_20261004/)
returned `SIM_SMOKE_ONLY` with `task_success=false` and produced the manifest,
event log, six non-empty RGB frames, and `episode.mp4`. This confirms the
current reset/camera/step integration but is deliberately not counted as a
robot-held assembly or learned-policy result.

## Adapter-level skill smoke (2026-10-04)

The collision-on adapter trace
[`m1_adapter_skill_smoke_recheck_20261004`](../runs/m1_adapter_skill_smoke_recheck_20261004/)
ran `pick → preinsert → insert → verify → release_retract` through
`hrc_m1.run --backend roco --skill-smoke --planner rule`. Every motion skill
returned `SUCCEEDED`; `pick` returned `HELD_CONFIRMED` using the public
filtered-contact/object-follow check (`64.471/73.916 N`, Hub follow
`0.082923 m` versus link-6 `0.113525 m`). The `insert` skill deliberately
returned only `SEAT_CANDIDATE`; the independent evaluator remained `UNKNOWN`,
and the run is labeled `SKILL_SMOKE_ONLY` with `task_success=false`. The
non-empty three-camera MP4 and event log are useful adapter evidence but do not
replace a verified physical seat/release episode or a real model-controlled
rollout.
