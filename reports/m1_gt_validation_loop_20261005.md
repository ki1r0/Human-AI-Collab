# M1 GT/scripted validation loop — 2026-10-05

This is the continuation log for the strict physical GT/scripted controller.
REPAIR and ACT evidence remain separate. The six-point acceptance thresholds
are unchanged.

## Starting evidence

The strongest existing dynamic full-route run is
`validation_logs/m1_dual_reset_y0_sync_hybrid_final_strict_20261003/metrics.json`.
It reached 4/6: reset, grasp/lift, and collision-free transport passed. The
measured seat was close in translation (0.84 mm radial, 0.04 mm axial), but
the held Hub still had 3.51 degrees orientation error and 5.65 mm maximum
bolt-hole error. Release drift (0 mm) and retract drift (0.094 mm) were stable.

## Iteration 1 protocol

Hypothesis: the remaining orientation/bolt error is partly caused by an
aggressive in-air wrist correction. Enabling the existing physical
`correct-orientation-at-insertion-above` action with a bounded 0.5-degree
per-segment step should reduce the residual while preserving the already
validated transport and release route.

Prediction: orientation <= 2 degrees and bolt-hole max <= 3 mm, with no loss
of dynamic grasp, collision-free transport, or release/retract stability.

Single changed control variable: enable the existing above-socket orientation
correction and set `held_orientation_step_deg=0.5`; scene, physics, route,
thresholds, and evaluator are unchanged. A failure is recorded as evidence,
not hidden by a threshold or pose/state write.

## Results

Results are appended after each run with the complete output directory and
the measured strict components.

### Preliminary replay audit

The first replay attempt used the launcher defaults and therefore reset the
Hub at `y=0.45 m` (the environment's collision-regression default), whereas
the historical near-success artifact has `hub_reset_position_m=[0.30, 0.00,
1.08]`. It was intentionally not treated as a controller result: the Hub was
already displaced to `y≈0.45 m` during approach, the right gripper never made
contact, and the run ended at 3/6. A corrected replay with explicit
`M1_HUB_RESET_X=0.30`, `M1_HUB_RESET_Y=0.0`, and `M1_HUB_RESET_Z=1.08` is the
valid audit run.

### Iteration 0b: corrected reset replay

The explicit `y=0` replay restored the expected two-sided grasp and the first
four strict points. It still failed at insertion: the right-jaw release had a
residual 13 N contact on one fingertip and the Hub tilt grew from near-zero at
lift to roughly 12 degrees during the right-tool withdrawal. The Hub then
missed the socket (`radial_error_m=0.553 m`, `axial_error_m=1.098 m`). This is
recorded as a release/withdrawal impulse, not as an orientation-seat result.

### Iteration 1 protocol

Hypothesis: the right withdrawal action, rather than the left transport, is
injecting the observed torque. Open the right gripper after the supported lift
but leave that already-open tool at its measured lift pose (`release-right-in-
place`); the left gripper remains the sole load-bearing controller for the
same direct transport and gravity release path.

Prediction: the Hub remains within the prior transport envelope, with no
post-release right-finger contact and a substantially smaller seat orientation
error. This is a single route/control change; no geometry, thresholds, or
evaluator behavior is changed.

### Iteration 1 result: right release left in place

The run is `validation_logs/m1_gt_iter1_right_in_place_20261005/metrics.json`.
It completed the corrected reset, dual grasp, synchronized lift, direct
transport, controlled seat, physical release, and retract without a runtime
error. The strict score remained **4/6**: components 1--4 passed, while the
6-DoF seat and release gate failed. Leaving the right tool in place removed the
large withdrawal ejection seen in iteration 0b, but did not remove the opening
impulse: the `right_release_after_lift` sample still carried `13.13 N` on the
right inner contact. The seat was `54.43 mm` radial, `22.69 mm` axial,
`16.04°` orientation, and `86.15 mm` maximum bolt error. The release and
retract drift checks themselves passed. This is an informative improvement,
not a success: the opening dwell, not only the subsequent withdrawal, is a
causal blocker.

### Iteration 2 protocol

Hypothesis: the right-jaw opening actuator is changing the loaded right-arm
IK branch during the release dwell. Add the opt-in
`--freeze-right-release-open` action, which holds both measured arm joint
targets while the right gripper opens; the Hub remains dynamic and the later
right withdrawal is unchanged. This is a single controller change, with no
scene, threshold, evaluator, or object-state modification.

Prediction: the right inner contact force and the release-induced angular
impulse decrease, while the left load-bearing pinch and transport remain
unchanged.

### Iteration 2 result: freeze right release opening

The source change is in `hrc_m1/debug_inner_wall_grasp.py` and is exposed by
`M1_FREEZE_RIGHT_RELEASE_OPEN=1` in
`tools/run_m1_isaac_strict_success.sh`. Syntax checks passed before the run.
The complete result is
`validation_logs/m1_gt_iter2_freeze_right_release_open_20261005/metrics.json`.

The right-finger residual decreased from `13.13 N` to `4.24 N`, confirming
that the opening dwell was physically involved. However, holding the arm
joints rather than its Cartesian pose changed the load-bearing left-arm
response: the Hub already had a larger attitude/velocity perturbation at the
release snapshot and reached a mis-seated final state. The strict score again
was **4/6**. Seat errors were `39.32 mm` radial, `24.41 mm` axial, `47.99°`
orientation, and `91.90 mm` maximum bolt error; release-contact-free, release
drift, and retract stability passed. Therefore this is a failed controller
ablation, not a 6/6 result. The residual force reduction does not justify
keeping this option as the default route.

The next bounded test must preserve the measured Cartesian pose of both arms
while the right jaw opens (rather than freezing raw joints), or use a staged
open/withdraw action that clears the right inner finger before the full
opening. No strict threshold or evaluator rule is changed.

### Iteration 3 result: Cartesian pose hold during right opening

The run is `validation_logs/m1_gt_iter3_hold_right_release_pose_20261005/metrics.json`.
The new mode computed ordinary IK targets from the measured left and right
TCP poses and held both target tuples during the right-jaw opening. It did not
write the Hub state or relax collision/acceptance logic. The right inner
contact did **not** clear (it increased to `29.16 N` at the release sample),
and the subsequent controlled insertion diverged catastrophically: the Hub
reached `[4.60, 23.79, -508.80] m`, with approximately `100 m/s` speed at the
insert sample and a final retract drift of `3118 m`. Strict score remained
**4/6** (components 1--4 only); seat, contact-at-seat, and retract stability
failed. This is a third independent release-controller attempt and is
rejected.

### Loop disposition

The three bounded release repairs have now been tested against the same
explicit reset, scene, controller route, and strict evaluator:

1. leave the opened right arm in place — reduced withdrawal ejection but left
   `13.13 N` opening contact and a `16.04°` seat error;
2. freeze raw arm joints while opening — reduced right residual force to
   `4.24 N` but changed the load-bearing response and produced `47.99°` seat
   error;
3. hold both measured Cartesian TCP poses with ordinary IK — retained
   `29.16 N` contact and diverged at insertion.

None reaches strict 6/6. Per the predeclared stopping rule, no further blind
release timing/trajectory tuning is justified in this loop. The GT/scripted
M1 validation remains incomplete; the exact blocker is a physically stable
right-hand release and left-hand-supported six-DoF seat, not a score threshold
or evaluator issue. REPAIR and ACT results remain separate.

### Watchable live-camera replay

For visual inspection, the closest non-divergent route was rerun with live
RTX cameras and no controller changes at
`validation_logs/m1_gt_video_right_in_place_20261005/`. The MP4 is
`inner_wall_probe.mp4`, with `2501` frames at `480x360`, and the metrics report
`video_error=null`. The matching run remains a scripted GT result with strict
score `4/6` and `insertion_verdict=INSERTION_NOT_VERIFIED`; the video is
provided to inspect the real grasp/transport/release behavior, not as evidence
of a completed M1 assembly.

## Resumed dual-arm repair audit (2026-10-05)

### Question 1 diagnosis

The latest non-divergent video run did reach the Casing's above-socket region;
the left arm was not simply out of reach. Its `insertion_above` Hub root was
approximately `[0.527, 0.109, 1.133] m`, while the calibrated socket target is
`[0.550, 0.087, 1.062] m`. By the `preinsert` sample, the dynamic Hub had
slipped to approximately `[0.526, 0.129, 1.099] m`; one left fingertip had
lost contact and the other carried the remaining load. The resulting final
seat error was `54.43 mm` radial, `22.69 mm` axial, and `16.04°` orientation.
The primary failure is therefore a load-bearing/contact-frame failure during
the final approach, not a Casing-distance or arm-reach failure.

### Dual-arm hypothesis D1 and protocol

The next baseline keeps both physical grippers closed through lift, horizontal
transport, above-socket alignment, preinsert descent, and controlled seat. The
right arm is not released after lift. Both arm targets are computed from the
same measured Hub state and applied in one environment action using
`--synchronous-dual-transport`; release occurs only after the seat stage. This
tests the user's requested cooperative path without changing the scene,
socket target, strict thresholds, or evaluator.

Prediction: both finger pairs remain load-bearing through `preinsert` and
`insert`, the Hub stays within the calibrated socket frame, and final release
can be performed without the preinsert one-sided-load failure. A failure will
be localized by the first contact/pose divergence in the trace.

### D1 baseline result: synchronized dual-arm route

The baseline run is
`validation_logs/m1_gt_dual_cooperative_baseline_20261005/metrics.json`.
Both arms remained closed through lift and the entire transport; the route used
same-step dual-arm IK and released only after the controlled seat attempt. It
scored **3/6**: the grasp and lift passed, but collision-free transport failed
at the final approach. The Hub reached the above-socket region with both inner
contacts still present (about `77--80 N`), but the outer contacts were lost.
During preinsert, the left inner contact rose to about `3.69 kN` while the
right contacts disappeared. The insert sample had about `0.057 m` root error,
`0.319 m/s` speed, and no Casing contact; the later release/retract retained
large left-finger forces and `0.169 m` retract drift. This confirms that
cooperative lift/transport is feasible, but the fixed lift-time grasp frames
are inconsistent with the actual two-arm pose at the socket approach.

### Dual-arm hypothesis D2

At the above-socket waypoint, re-anchor **both** TCP-to-Hub offsets and
orientations from the measured dynamic state before starting horizontal
registration and axial descent. This is ordinary feedback on the two real
arms; it does not write the Hub pose, create a constraint, or change the seat
target. The prediction is that the two arms will remain geometrically
consistent through preinsert, eliminating the one-sided multi-kN wrench.

### D2 result: dual-arm above-socket re-anchor

The run is `validation_logs/m1_gt_dual_reanchor_above_20261006/metrics.json`.
Re-anchoring did not improve the route. The preinsert contact was reduced from
the baseline's multi-kN left-only impulse, but the Hub still drifted off the
socket axis (`56.02 mm` radial, `27.16 mm` axial) and rotated to `45.45°`.
At the insert sample the right inner contact reached about `5.07 kN`, while
the left contacts were lost; the strict score remained **3/6** and transport
still failed. Re-anchoring the existing frame is therefore not sufficient and
is not enabled as the default.

### Dual-arm hypothesis D3

The orthogonal route's final side/corner leg may be injecting the two-arm
wrench. Use the existing opt-in direct high-Z route so the two synchronized arm
targets move directly to the calibrated socket vertical line, avoiding that
side-above corner. Keep both grippers closed and release only after controlled
seat. This changes one route variable only.

### D3 result: direct high-Z cooperative route

The run is `validation_logs/m1_gt_dual_direct_cooperative_20261006/metrics.json`.
The direct leg preserved both physical pinches through the high-Z transport:
the trace shows both inner/outer fingertip topology through
`closed_loop_direct_above_8`, with left/right contact forces still present
(approximately `16--119 N` on the left and `24--139 N` on the right at the
above-socket waypoint). Thus the side/corner transport leg was not the only
failure source. During the subsequent horizontal registration, the left outer
contact disappeared at segment 6 and the right outer contact disappeared by
segment 8. The first axial preinsert step then began with both pinches no
longer load-bearing, generated `6.54 kN` Hub–Casing force, and moved the Hub
to `[0.531, 0.029, 1.100] m` instead of the target preinsert height. The
strict result remained **3/6** (transport, six-DoF/bolt seat, and release/
retract failed); final radial error was `46.97 mm`, axial error `27.53 mm`,
orientation error `85.26°`, and post-retract drift `92.19 mm`.

The evidence rejects “only the orthogonal corner route” as the complete
explanation. The remaining failure is a fixed, lift-time two-arm grasp frame
becoming inconsistent during the horizontal registration and then imposing an
over-constrained descent. The next test therefore enables measured adaptive
grasp-frame feedback for both arms; no evaluator threshold, scene pose, or
contact model is changed.

### Dual-arm hypothesis D4

With both grippers closed, recompute each wrist's current measured TCP-to-Hub
vector at every registration segment instead of reusing the lift-time vector.
This should absorb millimetre-scale real slip before the axial descent and
avoid the one-sided wrench seen at the first preinsert segment. The change is
implemented symmetrically for left and right arms behind the existing opt-in
`--adaptive-grasp-frame` flag; the default controller remains unchanged.

### D4 result: symmetric adaptive grasp-frame feedback

The run is `validation_logs/m1_gt_dual_adaptive_direct_20261006/metrics.json`.
Adaptive vectors did not preserve the two-arm pinch. In the direct high-Z leg
the right outer contact was already absent at `insertion_above`; during
horizontal registration both outer contacts then disappeared by segment 5,
while the surviving inner contacts rose into the hundreds of newtons. The
first axial preinsert segment produced a large left-side impulse (about
`4.61 kN`) and the Hub remained off-axis at
`[0.512, -0.019, 1.142] m`, with approximately `603 N` Hub–Casing force at
the final preinsert sample. The strict result was still **3/6**; final radial
error `76.80 mm`, axial error `238.23 mm`, orientation error `54.04°`, and
post-retract drift `177.68 mm`.

Adaptive feedback therefore worsened this route and is not enabled by default.
The remaining evidence points to the two arms' simultaneous pose-IK descent,
not merely stale grasp offsets: the right arm loses its outer contact while
the object is still above the socket, before the casing is contacted. The next
bounded test will keep the demonstrated dual-arm lift/transport but use the
existing position-only IK mode for the horizontal registration, then restore
full pose IK for the final orientation/seat stage. This tests whether the
orientation branch is injecting the loss, without writing a rigid-body pose.

### D5 result: position-only high-Z transport

The first D5 invocation was interrupted before it wrote an output, so it is
not counted. The completed retry is
`validation_logs/m1_gt_dual_position_transport_retry_20261006/metrics.json`.
Position-only IK brought the Hub close to the socket center at the end of
horizontal registration (`[0.546, 0.083, 1.190] m`), with both grippers still
in contact, though the right contact peaked near `9.9 kN`. The controller then
returned to full pose IK for the first axial preinsert segment. Both grippers
lost contact there, followed by numerical/physical divergence (the logged
Hub position reached hundreds of metres). Strict score stayed **3/6**.
This isolates the failure to the mode switch at descent: position-only
registration alone is insufficient, and the single segment that restores
pose IK destabilizes the held object.

### Dual-arm hypothesis D6

Keep position-only IK for high-Z transport, horizontal registration, preinsert
descent, and controlled seat. Once the Hub reaches the seat waypoint, use the
existing bounded orientation-to-authored-pose correction while both arms are
still engaged. For that correction, command both wrists with the same
incremental rotation around the Hub; the earlier helper rotated only the left
grasp frame and left the right target fixed. This tests the requested
“seat first, orient last” sequence with dynamic contact and the unchanged
strict evaluator.

### D6 result: position-controlled seat, then orientation correction

The run is `validation_logs/m1_gt_dual_seat_then_orient_20261006/metrics.json`.
This is the first resumed dual-arm route to pass the transport component:
the strict score rose from **3/6 to 4/6**, and all 19 transport samples retained
both arms in the contact trace. Position-only control kept the Hub near the
socket center during the approach, but it reached the seat waypoint still
`44.7 mm` too high and `25.6°` off orientation, with a `13.6 mm` radial error.
The Hub–Casing force at that sample was `190 N`. The after-seat rotation reduced
orientation error to `3.26°`, but the Hub was not actually seated; it moved to
`[0.567, 0.092, 1.120] m` and the subsequent retract ended at `1.302 m` Z.
Release/retract and final seat checks failed. This shows the large orientation
alignment must occur above the socket so the Hub can enter; only a residual
adjustment can safely be left to the end.

### Dual-arm hypothesis D7

Use the same direct high-Z dual-arm route and position-only control through
transport and descent. At the above-socket waypoint, first rotate both wrists
together toward the authored Hub orientation while the assembly has clearance.
Refresh both measured grasp frames, register X/Y over the socket, then descend
and seat. Retain the bounded after-seat correction for any small residual.
This tests whether entering with the correct orientation closes the remaining
seat gap while preserving the demonstrated dual-arm transport.

### D7 result: preorient above socket, then seat

The run is `validation_logs/m1_gt_dual_preorient_then_seat_20261006/metrics.json`.
It retained the first four strict points (**4/6**) and aligned the Hub to about
1.8 degrees at the above-socket orientation correction. It then reached the
preinsert sample at approximately `[0.5445, 0.0831, 1.1209] m` rather than the
requested `[0.5500, 0.08684, 1.1400] m`. At the insert sample it was still
`22.31 mm` radial, `23.79 mm` axial, and `12.09°` rotationally off; maximum
bolt-hole error was `48.99 mm`. The after-seat correction increased the
orientation error to `29.05°` and the retract pulled the cover upward. Thus
preorientation alone did not solve the final registration/seat.

### Dual-arm hypothesis D8

The opt-in preinsert correction computes a bounded residual as
`target - measured`, but the prior code added that residual to `target`, which
commands `2*target - measured` and doubles the measured error. It also moved
only the left wrist although D7 keeps both arms engaged. Correct the waypoint
to `measured + bounded_residual` and route it through the existing closed-loop
helper, which applies the left/right pose-IK targets synchronously. Run once
with one correction enabled and otherwise keep D7's scene, physics, route,
orientation timing, seed, and strict thresholds unchanged.

Prediction: the correction sample moves toward the preinsert target without
overshooting, both arms remain in contact, and the subsequent seat improves in
translation without a new high-force casing impact. If it does not, the trace
will distinguish reach/IK error from a physical two-arm contact-wrench limit.

### D8 result: bounded dual-arm preinsert correction

The run is `validation_logs/m1_gt_dual_preinsert_fix_20261006/metrics.json`.
The arithmetic and action path are now correct: the bounded residual is added
to the measured root and both wrists use the synchronous closed-loop helper.
However, this did not improve seating. The root changed only about 1--2 mm
during the correction despite a roughly 20 mm preinsert residual; both arms
remained in contact, with right fingertip forces around `1.5--1.9 kN`. At
`controlled_seat_2`, the Hub jumped to `[0.579, 0.116, 1.162] m`, reached
`2.55 m/s`, and the left inner fingertip force peaked at `10.0 kN`. The insert
ended `24.39 mm` radial, `27.31 mm` axial, `14.48°` orientation, and `54.50 mm`
maximum bolt error from target. The strict score remains **4/6**; release and
retract remain unstable. This rejects a simple extra Cartesian correction as
the cure and localizes the main seat-stage event to the first controlled-seat
steps under an imbalanced, high-force two-arm grasp.

### Dual-arm hypothesis D9

Enable the existing `reanchor_before_seat` action after the preinsert hold so
both arms refresh their TCP-to-Hub vectors from the measured state immediately
before descent. D8's position correction was followed by a hold and then seat
using frames captured earlier, while the right fingertips carried much more
load than the left; D8's abrupt `controlled_seat_2` excursion is consistent
with stale dual-arm frames imposing incompatible targets. Change only this
re-anchoring option from D8. Keep the correction fix, seed, route, scene,
physics, orientation stages, and strict metrics unchanged.

Prediction: the first seat segments remain finite and avoid the `2.55 m/s` /
`10 kN` excursion while both wrists stay in contact; the measured seat pose
should move closer to the unchanged target. If not, the trace will show that
the force spike is intrinsic to the two-arm pinch/seat geometry rather than a
stale-frame error.

### D9 result: re-anchor both grasp frames before seat

The run is `validation_logs/m1_gt_dual_reanchor_before_seat_20261006/metrics.json`.
This was a real improvement in seat stability: the insert speed dropped from
D8's `0.241 m/s` to `0.055 m/s`, radial error from `24.39 mm` to `11.26 mm`,
and orientation error from `14.48°` to `4.82°`. The strict score nevertheless
remains **4/6**. The cover stopped about `40.20 mm` above the target, did not
make measured Casing contact, and retained `46.95 mm` maximum bolt error. The
right outer fingertip lost contact during `controlled_seat_4` as the root
shifted about 9 mm laterally; release/retract drift remained `113.13 mm` after
settling. Re-anchoring removes D8's extreme second-segment impulse, but the
combined XY-and-Z seat command still lets lateral motion destabilize the
two-arm pinch before the Hub reaches the socket.

### Dual-arm hypothesis D10

Keep D9's pre-seat re-anchor and change only the existing
`seat_vertical_only` option. The target is already within about `6.5 mm` radial
at the start of seat; D9's first meaningful lateral slip occurs in the same
phase as the right outer fingertip contact loss. Hold measured X/Y during
descent and command only axial motion, allowing the real Casing geometry to
guide the last part of insertion. No pose write, gravity change, threshold, or
seat depth change is made.

Prediction: the Hub retains its pre-seat radial position, the right two-sided
pinch survives deeper descent, and Casing contact appears before the final
seat sample. The unchanged strict seat/release evaluator decides success.

### D10 result: vertical-only controlled seat

The run is `validation_logs/m1_gt_dual_vertical_seat_20261006/metrics.json`.
Vertical-only held the insert radial error to `5.55 mm`, but the Hub stopped
`44.19 mm` above the seat target with no Casing force at the insert sample;
orientation was `5.15°` and bolt error `47.68 mm`. The right outer fingertip
lost contact at `controlled_seat_3`, after which the Hub made little axial
progress. The after-seat wrist correction lifted it back to `1.149 m`; when
the grippers opened, contact-free release passed but the unsupported Hub fell
through and ended at `z=-0.036 m` (`1.185 m` post-settle drift). Strict score
remains **4/6**. Holding X/Y fixed prevents the lateral kick, but does not
preserve the right-hand pinch or complete insertion.

### Dual-arm hypothesis D11

Keep D10's position-only transport/preinsert approach, pre-seat dual-arm
re-anchor, and vertical-only seat target. Restore full pose IK only for the
controlled seat. D10's position-only seat preserved X/Y but the right outer
finger lost contact at segment 3; position-only IK does not constrain the
loaded wrist orientation. D5's earlier full-pose re-entry failed at the
preinsert descent before the two-arm frame was re-anchored, so this test
re-enters only after the measured preinsert hold/re-anchor. The small opt-in
`--seat-full-pose-ik` selects that phase boundary; all physical settings and
strict acceptance checks remain unchanged.

Prediction: both wrist orientations and right two-sided contact persist deeper
into the vertical seat, producing Casing contact and reducing axial error. If
the re-entry again destabilizes the dynamic Hub, that rules out pose IK at the
final seat and points to the physical two-hand grasp geometry/retention as the
remaining blocker.

### D11 result: full-pose IK only for the final seat

The run is `validation_logs/m1_gt_dual_full_pose_seat_20261006/metrics.json`.
Restoring full pose IK at the seat did not preserve the right outer fingertip:
it was lost by `controlled_seat_4`, as in D10. The Hub ended `42.13 mm` above
the target with no Casing contact, `5.61°` orientation error, and `48.12 mm`
maximum bolt error. The strict score remains **4/6**; the after-seat rotation
and release again produced drift. The remaining limitation is therefore not
just position-only wrist rotation: in this measured grasp, the two-arm
configuration loses its right-hand load-bearing contact and does not drive the
Hub down to the seat, even when full pose IK is restored.

### Dual-arm hypothesis D12 (handoff diagnostic)

Run the same route but use the existing `release_right_at_insertion_above`
handoff: keep both arms load-bearing through lift and direct transport to the
above-socket waypoint, then open/withdraw the right arm and finish the
preinsert/seat with the left. This tests whether dual-arm coupling itself is
the descent blocker. It is a diagnostic alternative, **not** counted as
satisfying the requested two-arm descent unless the later task evidence shows
the right arm still contributes through the physical support/seat phase.
All scene, reset, seat target, timing, physics and evaluator settings remain
the same.

Prediction: the left arm reaches the previously demonstrated stable single-arm
seat while the dual-arm transport remains collision-free. If that works, the
next design question is a supported handoff later in the descent; if it fails,
the right-arm handoff impulse is itself a blocker.

### D12 result: release-right-at-above handoff diagnostic

The run is `validation_logs/m1_gt_dual_handoff_above_20261006/metrics.json`.
The raw six-point result is **3/6** (`INSERTION_NOT_VERIFIED`). Both grippers
remain in contact through the high-Z transport to `insertion_above`; the left
gripper is the only support after the intentional right-hand release. During
the opening/withdrawal, the Hub shifts from
`[0.5612, 0.0766, 1.1919] m` to `[0.5624, 0.0985, 1.1398] m` (about 56.6 mm)
and rotates roughly 43 degrees from the target. The left-only preinsert/seat
then loses one left fingertip by seat segment 2. The insert sample is
`[0.5225, 0.1452, 1.0969] m`: 64.54 mm radial, 34.85 mm axial, 43.36 degrees
orientation error, and 153.04 mm maximum bolt-point error. Casing contact is
present but the assembly is not seated/aligned. The unconditional 6-degree
after-seat wrist correction reaches 50.2 kN measured Casing contact and
0.554 m/s Hub speed, so it is not a safe recovery for a failed seat.

Two evaluation caveats are recorded rather than hidden. First, the JSON's
transport point is 0 because its scorer still expects all four finger contacts
after `release_right_at_insertion_above`; the trace shows the four-contact
transport itself completed to the above-socket handoff. The raw 3/6 is retained
without rewriting the scorer, and this diagnostic does not meet the requested
bilateral descent. Second, `release` and `retract` were recorded at the exact
reset pose, not at the Hub's preceding physical state. The local task config
defaults to a 60-second episode (`hrc_m1/roco_env.py`), but this run inherited
the wrapper's 180-second override; it reached the environment timeout and
auto-reset before valid release evidence was captured. Thus D12's seat-stage
trace is useful, but its release/retract measurements are censored by timeout.

This exposed a protocol error: recent long scripted diagnostics used a longer
episode horizon instead of the task's native 60 seconds. Do not extend the
horizon again. Next, restore 60 seconds and test shorter waypoint hold times
as a controller-speed change, keeping the D11 bilateral route and all target
geometry/strict criteria fixed. Do not apply the after-seat yaw command unless
the measured seat is stable; prefer solving orientation before descent.

### Dual-arm hypothesis D13 (native-horizon timing)

Return to D11's bilateral route (no right-arm handoff) and change only the
controller hold-time scale to `0.30`. The environment episode horizon is
restored to its native `60 s`; simulation dt remains `0.01 s` and environment
step dt remains `0.05 s`. This shortens the number of physics steps spent
holding each interpolated IK waypoint; it does not change the target path,
seat pose, physics, thresholds, or simulated time-step. The run must finish
before the native timeout to count as a complete rollout. In particular,
verify `release`/`retract` are not the exact reset state before interpreting
those measurements.

Prediction: the same bilateral lift, direct transport, preinsert and vertical
seat complete within 60 simulated seconds without the timeout auto-reset seen
in D12. If the compressed controller loses either fingertip or fails to
advance the Hub, the record will show whether 30-step waypoint dwell itself
was important for the physical hold. Strict acceptance remains unchanged.

### D13 result: 0.30 waypoint hold-time scale under the native 60 s horizon

The run is `validation_logs/m1_gt_dual_native60_scale030_20261006/metrics.json`.
It completed under the native 60-second cap, but the physical result regressed
to **3/6**. The cover was at `[0.5662, 0.0619, 1.1093] m` at the insert sample:
`29.71 mm` radial, `47.29 mm` axial, `52.22°` orientation error, and
`125.50 mm` maximum bolt-point error. The left finger pair lost contact during
the above-socket orientation correction; preinsert then stayed about 49 mm
off in Y. The first numerical blow-up starts at after-seat yaw correction
segment 6 (Hub speed `3.54 m/s`, followed by `10.66 m/s` at segment 9), after
all fingertip contacts had already gone to zero. The later release/retract
positions are out-of-domain and are not physical seating evidence.

This rejects uniform `0.30` scaling: target waypoints advance faster than the
arm/object can track, so grasp-frame error accumulates before seating. D13
does prove the route can fit inside the official horizon only by making that
unsafe timing change. Keep the native 60-second cap; do not use this run to
justify changing it. Next test a route-level reduction of redundant high-Z
waypoints while retaining the normal per-waypoint dwell, and remove the
after-seat yaw action from that diagnostic because it acts after the seat has
already failed and generates extreme contact/velocity spikes. Any resulting
score remains subject to the same physical 6-point criteria.

### Dual-arm hypothesis D14 (shorter route, normal settling per waypoint)

Keep D11's bilateral grasp/seat setup, native 60-second horizon, and 30 physics
steps per interpolated target. Use the existing direct high-Z transport mode
to replace the rise/side/above sequence, reduce insertion waypoints from 8 to
4 (without shortening each target's settling time), and turn off the
after-seat yaw command. The above-socket orientation correction remains the
orientation alignment step. This is an explicit route/discretization test:
each waypoint now spans more distance, but gets the original 1.5 seconds of
actuator/physics settling. The per-snapshot episode clock will verify that no
timeout reset contaminated release/retract.

Prediction: fewer redundant free-space moves fit the controller path under
60 seconds without the accumulated IK lag seen at D13's 0.45-second waypoint
dwell. If the larger interpolation increments instead drop either grasp,
the contact trace will identify which route leg needs denser waypoints; the
episode horizon and strict evaluator remain unchanged.

### D14 result: direct high-Z route with four insertion waypoints

The run is `validation_logs/m1_gt_dual_native60_direct4_20261006/metrics.json`.
The raw JSON says **3/6**, but this is not a valid completed rollout: the
episode clock shows `insertion_above` at 51.0 s, above-socket orientation
correction at 57.0 s, and an automatic environment reset at the 60.0 s cap
during `closed_loop_preinsert_xy`. The subsequent Hub states, score, and
release/retract checks are post-reset artifacts. Before timeout, the dynamic
Hub had reached `[0.5502, 0.0860, 1.2204] m` with all four fingertip contacts;
the orientation-correction endpoint was `[0.5418, 0.0862, 1.2027] m` with
all four contacts still present. This is meaningful approach evidence, not
insertion success.

The trajectory contains 2,260 environment steps at the normal scale (113.0 s),
so the reduced path still cannot fit the native 60 s horizon. D15 will change
only `step_scale` to `0.50` on D14's direct/four-waypoint route. That gives an
estimated 1,130 steps (56.5 s), leaving a small margin before timeout; each
waypoint still receives 15 physics steps (0.75 s), between D13's failed 9 and
D14's 30. The new per-record clock will verify actual completion and expose
any reset immediately.

### D15 result: scale 0.50, complete inside the native episode

The run is `validation_logs/m1_gt_dual_native60_direct4_scale050_20261006/metrics.json`.
It completed at 56.5 s without a reset and scored **4/6**; all 15/15 transport
samples retained the required contacts. At `insert`, the Hub root was
`[0.54668, 0.08828, 1.10593] m`: radial error `3.62 mm`, but still `43.93 mm`
above the target. Hub-Casing force was zero, insertion speed was `0.160 m/s`,
orientation error `3.95°`, and maximum bolt-point error `48.59 mm`. The left
inner fingertip lost contact during seat segment 3 while the right pair
remained engaged; the cover then stopped descending. Opening did not clear
the remaining Hub contacts (left outer `2.81 kN`, right contacts about
`0.97--1.26 kN`), and the ordinary upward retract lifted the Hub `182 mm`.
So the episode clock and bilateral transport now pass, but seat and release
do not. The 4/6 is a valid completed episode, not success.

### Dual-arm hypothesis D16 (Jacobian position seat)

Keep all D15 path, physics, timing, and target settings fixed; enable only the
existing `seat_jacobian_position` controller for the vertical controlled-seat
phase. D15's preinsert radial error was only `5.3 mm`, but the Hub advanced
only `13.7 mm` axially over the four seat waypoints and one left fingertip
lost contact. The Jacobian-position branch computes bounded joint deltas
from the measured TCP position instead of re-solving the left full-pose IK
target at every seat waypoint; the right target and physical two-arm grasp
remain active. Prediction: the left TCP tracks the axial descent more
continuously, the Hub reaches measured Casing contact, and both load-bearing
grasp contacts persist farther into seating. Strict thresholds and the
native 60-second horizon remain unchanged.

### D16 result: Jacobian position seat did not resolve the descent stall

The run is `validation_logs/m1_gt_dual_native60_direct4_jacobian_seat_20261006/metrics.json`.
It completed without a timeout reset in 56.5 simulated seconds and scored
**4/6** (physical-place score **2/4**): bilateral lift/transport passed, but
seating and release did not. At `insert`, the Hub root was
`[0.54363, 0.08808, 1.10593] m`, about `6.49 mm` radial and `43.93 mm` axial
from the target; orientation error was `3.48°`, bolt-point error `48.30 mm`,
and Hub speed `0.081 m/s`. No Casing contact was measured. The left inner
fingertip had lost Hub contact while the right arm remained heavily loaded.
During retract the Hub was pulled upward by about `177 mm`, confirming it had
not been released in a seated state.

D16 changed the seat controller branch but produced essentially the same
44-mm axial stall as D15. Repeatedly changing IK mode is not justified by this
evidence. Next diagnose the right TCP's near-static Z during the seat phase and
the asymmetric fingertip loads against the actual commanded targets and joint
motion, then change one cause-specific factor before another full rollout.
The full run takes minutes of wall time even though it represents only
56.5 seconds of simulation because each environment action advances five
0.01-second physics steps, with articulation/contact evaluated on every step;
Isaac Sim startup and rendering add overhead. This run's JSON reports zero
captured video frames, so it did not produce a watchable rollout video.

### D16 recorded replay / repeatability check (2026-10-06)

The same recorded D16 controller and physics settings were replayed with camera
capture enabled in
`validation_logs/m1_gt_dual_native60_direct4_jacobian_seat_video_20261006T2248/`.
The episode retained the 60-second horizon and seed 1201; no acceptance
threshold or action timing was changed. The live MP4 contains 1,071 decoded
frames at 20 fps (three 480x360 camera views arranged side by side).

This replay scored **3/6** (physical-place **2/4**), one RoCo-style point below
the prior D16 result of 4/6. Contact began dropping during
`closed_loop_preinsert_xy_2`; at `insert`, radial error was `43.86 mm`, axial
error `8.67 mm`, orientation error `11.08°`, and maximum bolt-point error
`64.70 mm`. The verdict remains `INSERTION_NOT_VERIFIED`. Thus the recorded
controller settings do not yet produce a repeatable transport/seat result;
this replay changed no controller logic or success criteria.

### D17: image-2 XY hold, slower descent dwell, and timed post-seat hold

The user identified the above-socket pose as correct and asked for a slow
vertical descent followed by release only after the Hub Cover is seated. The
run is `validation_logs/m1_gt_vertical_descent_hold80_video_20261006T2305/`.
It kept the native 60 s episode, seed 1201, physical setup, and evaluator. The
preinsert XY offsets were set to the measured image-2 Hub position, the
in-air orientation correction was disabled, insertion waypoints were held
for 45 rather than 30 environment steps, and the existing post-seat joint
hold was set to 80 steps (2 simulated seconds). No thresholds changed.

The episode completed in 58.1 s and recorded 1,163 H.264 frames. It scored
**4/6** RoCo-style and **2/4** physical-place; insertion remained
`INSERTION_NOT_VERIFIED`. At `insertion_above`, the Hub root was
`[0.53093, 0.08012, 1.20668] m`. By `preinsert` it had drifted to
`[0.52393, 0.07282, 1.12193] m`. During controlled seat, the first waypoint
lost the left link-1 contact; finger loads rose to 3.16--5.91 kN, while the
Hub moved laterally and stopped descending. At the end of the 80-step hold,
its root was `[0.54365, 0.06965, 1.09649] m`: `34.5 mm` above the target,
`18.3 mm` radial error, and zero measured Hub-Casing force. Thus the timed
hold did not establish seating. The scheduled release still began with
`1,001.5 N` left- and `163.6 N` right-fingertip contact; release drift was
`19.4 mm`. The final Hub was tilted `35.37°`, with `116.8 mm` maximum
bolt-point error and `42.9 mm` post-settle retract drift.

This confirms the user's observation: the release is still effectively
mid-air. The failure is not simply a short wait: the 80-step wait is
time-based, not contact-based, and the image-2 XY pose was about `20 mm`
radially away from the seat target. Holding that offset through a vertical
descent left the Hub about `18 mm` off-axis at the seat phase. The next test
should keep the part grasped while aligning above the socket center, then use
smaller vertical waypoints; opening should follow measured seating rather
than a fixed delay.

### D18: socket-center registration with finer global waypoints

To address D17's measured radial offset, this replay restored zero preinsert
XY offsets and split the existing route into 8 waypoints of 22 environment
steps each; their total dwell is approximately the same as D17's 4×45-step
setting. Seed, physics, native 60 s episode, and evaluator were unchanged.
The run is `validation_logs/m1_gt_centered_8waypoint_video_20261006T2358/`.

This attempt scored **3/6** (physical-place **2/4**), worse than D17. The
four-waypoint XY registration did not carry the object toward the target:
from `insertion_above` `[0.54152, 0.06701, 1.20612] m`, the Hub's Y coordinate
moved away from the target during `closed_loop_preinsert_xy`; by waypoint 4
the two outer contacts were lost, and the gripper contacts were all gone at
`preinsert`. The Hub then contacted the Casing at `[0.49965, 0.07294,
1.07152] m`, but was `51.3 mm` radially off target, `8.42°` misoriented, and
`63.9 mm` out on maximum bolt-point error. It stayed nearly still after
release (0.25 mm release drift; 0.97 mm post-settle retract drift), but this
was a stable misplaced part, not a successful seat. One left-fingertip contact
also remained at the release snapshot (`28.8 N`).

The current dual-arm fixed grasp frame cannot safely execute this full
20-mm-level XY correction: it loses contact and lets the Hub fall laterally.
Reject this registration route. The next controlled test returns to D17's
image-2 XY hold and lowers the preinsert waypoint so that the final axial
descent is shorter, leaving all other D17 variables unchanged.

### D19: lower preinsert waypoint with image-2 XY held

The single changed parameter from D17 was `preinsert_height_m`, reduced from
`0.14 m` to `0.105 m`; all other D17 settings, the 60 s horizon, seed, and
acceptance conditions were held. The video and metrics are in
`validation_logs/m1_gt_vertical_preinsert105_hold80_video_20261007T0010/`.

This also scored **4/6** (physical-place **2/4**) and did not verify
insertion. The Hub reached `preinsert` at `[0.52420, 0.06831, 1.09913] m`,
but the controlled-seat path then became unstable: at waypoint 3 the measured
Hub speed reached `2.48 m/s` and a right fingertip reported `8.93 kN`; the Hub
was driven back upward to `z=1.11550 m` by the end of the seat command. There
was no measured Hub-Casing contact at `insert`; axial error was `53.5 mm`,
orientation error `18.79°`, and maximum bolt-point error `80.3 mm`. Release
began with multi-kN fingertip contacts, and retract pulled the Hub upward by
`182 mm`.

Lowering the preinsert endpoint is rejected: it increases loading/instability
instead of making the final descent more gradual. Together, D17--D19 show
that timed holds and waypoint/height tuning alone do not reliably realize the
requested seat-then-release behavior. The next action should target the
controlled-seat arm tracking itself, not further change the release timing or
relax the evaluator.

### Runtime release condition rationale

D17 is the concrete failure case: the scheduled open began with zero measured
Hub-Casing force and a `34.5 mm` axial gap; the Hub then moved `19.4 mm` while
fingertips were still loaded. Git/revision metadata can identify the command
but cannot inspect live PhysX contact at the instant of opening. Primary keys,
transactions, and uniqueness constraints protect stored records, not actuator
actions; static types cannot prove a dynamic object is supported. Ordinary
offline or post-run tests observe the failure after the open command and
cannot prevent it. Therefore the requested “seat first, then open” behavior
uses a narrowly scoped runtime condition at the physical release action,
reusing the existing relaxed placement tolerances and adding no scoring or
acceptance threshold.

### D20: discarded configuration-mismatch diagnostic

The first 8-waypoint guarded-release attempt is saved at
`validation_logs/m1_gt_seat8x28_guarded_release_video_20261007T0036/`.
It is not a valid comparison with D17: the launch command omitted D17's
`hub_reset_y=0.0 m`, `10/8` gripper friction, and `4x30` lift settings, so it
used the default `y=0.45 m`, friction `4/3`, and lift `8x60`. It also followed
the non-direct transport branch. The Hub was already out of the workspace by
the end of transport (`z=-0.397 m`), before the requested descent could be
evaluated. The runtime guard correctly skipped release. Keep this artifact as
a command-configuration diagnostic; do not compare its score to D17.

### D21: matched D17 transport, 8x28 controlled seat, guarded release

The corrected run is
`validation_logs/m1_gt_seat8x28_guarded_release_matched_video_20261007T0046/`.
It restores D17's seed, reset pose, grip friction, lift settings, direct
transport route, and 60 s horizon; changes are limited to the slower 8x28
controlled-seat route, the seated-only release condition, and simultaneous
dual-jaw opening if that condition passes. The rollout reached the same
above-socket pose as D17: `[0.53093, 0.08012, 1.20668] m`; the video contains
887 frames and `video_error` is null.

The seat check failed, so no gripper release occurred. Before the check the
Hub was `[0.52189, 0.10768, 1.09793] m`, moving at `0.0840 m/s`, with no
measured Hub-Casing force. Readiness reported `35.0 mm` radial error,
`35.9 mm` axial gap, `16.06°` orientation error, and `64.6 mm` maximum
bolt-point error. The 8-waypoint trajectory therefore did not convert the
correct above-socket pose into a physical seat; adding dwell/waypoints alone
is insufficient. Peak fingertip force reached `9.86 kN` on the right link,
consistent with the current seat controller over-constraining the two arms.
Next compare a symmetric dual-arm position-only IK seat against the current
left-Jacobian/right-pose-IK combination, without changing the route, reset,
or release criterion.

### D22: symmetric dual-arm position-only seat

The run is `validation_logs/m1_gt_seatpos8x28_guarded_release_video_20261007T0055/`.
It keeps D21's matched transport and 8x28 guarded seat, but disables the
left-only Jacobian path and uses position-only IK for both arms. This reduced
the final orientation error from `16.1°` (D21) to `6.81°` and produced
transient Hub-Casing contact during seat waypoints 6--8 (peak `587 N`). The
80-step closed-joint hold then lost Casing contact; at the release check the
Hub was moving at `0.0821 m/s`, `46.6 mm` radially and `26.9 mm` axially from
the seat target, so the guard again kept both jaws closed. Score remained
`4/6` (physical-place `2/4`), `INSERTION_NOT_VERIFIED`.

This is a better orientation/contact response but not a stable seat: keeping
the Hub's measured X/Y fixed during the final descent prevents correcting the
remaining radial offset. Next retain symmetric position-only IK and let the
same gradual seat trajectory make its measured lateral correction toward the
socket center while descending; keep the same release guard and thresholds.

### D23: symmetric position-only IK with lateral correction during descent

The run is
`validation_logs/m1_gt_seatpos_diagonal8x28_guarded_release_video_20261007T0106/`.
Relative to D22, the only change was allowing the 8x28 seat path to correct
X/Y while descending. The Hub reached the same above-socket pose, but the
combined diagonal correction caused a `27.998 kN` transient Hub-Casing force
and `0.79 m/s` Hub speed at waypoint 7. At the release check it had bounced
to `[0.57066, 0.06159, 1.12763] m`; despite low speed and `48.9 N` residual
contact, it was `32.6 mm` radial, `65.6 mm` axial, `64.12°` rotational, and
`179.8 mm` bolt-point error from the seat target. Score remained `4/6`
(physical-place `2/4`), and release was blocked.

D21--D23 are three independent, physically motivated controller changes:
finer guarded descent; symmetric dual-arm position-only IK; and gradual XY
correction during descent. None produced a verified seat. D22 did create
brief contact, while D23 shows that asking the same grasp to correct laterally
under load can eject the Hub. Stop this iteration here as requested; do not
relax acceptance or remove the release guard. A next attempt needs a different
grasp/contact-control treatment, not more dwell or waypoint tuning.

### D24 planned: fixed overhead view for geometric diagnosis

The existing rollout video contains head and wrist views only; those views
hide parallax between the Hub center and socket. Add one RGB camera fixed above
the socket center `(0.55, 0.08683718, 2.05) m`, looking straight down, and show
the four cameras in a 2x2 video layout. First run a short camera-only probe and
inspect that the casing and socket are visible without changing physics,
controller inputs, or acceptance criteria. If framing is sound, use the same
view in the next full rollout to distinguish camera-perspective error from
true XY misalignment.

### D24: overhead camera and matched D22 full rollout

The 10-frame camera probe passed. The overhead view is centered on the registered
socket XY and clearly frames the casing and Hub cover. The matched full rollout
is `validation_logs/m1_gt_overhead_baseline_d22_20261007T0150/`; it uses D22's
controller settings, adding only the camera. The four-view video is
`inner_wall_probe.mp4` (887 frames, 960x720, no video error). It reproduces
D22's `4/6` ROCO-style and `2/4` physical-place scores; release remains blocked.

The overhead projection agrees with the measured Hub-root offset: the socket
optical center is pixel `(240, 180)` and the final measured Hub root projects to
approximately `(217, 181)`, matching the 46.6 mm world-X error. At the
preinsert hold, the Hub root was `[0.52274, 0.08060, 1.12330] m` versus the
socket XY `[0.55000, 0.08684] m`; it was already 27.3 mm left of the socket.
During controlled-seat waypoint 8 it reached `[0.51753, 0.08801, 1.08757] m`
with 534.6 N Hub-Casing contact and 0.0165 m/s speed, but the configured
80-step post-seat hold then pulled it to `[0.50344, 0.08497, 1.08886] m`,
raised speed to 0.0821 m/s, and eliminated Casing contact. The runtime release
guard correctly kept the grippers closed.

The USD source frame inspection found `plug_main` has no local X/Y translation;
the model bounds are symmetric around those axes. Combined with the camera
projection, this supports a real X misalignment rather than a root-frame-only
artifact. Next test a measured X lead (`preinsert_offset_x_m=+0.01366 m`, from
D22's waypoint-8 X error) and remove the 80-step post-seat hold so the existing
release guard checks the part immediately after the slow descent. Keep the Y
target, seat depth, score thresholds, and guard unchanged. This isolates the
known preinsert X offset and avoids the observed post-contact drag.

### D25: excessive one-shot X lead rejected

The full rollout is `validation_logs/m1_gt_overhead_xlead_immediate_guard_20261007T0205/`
(`inner_wall_probe.mp4`, 847 frames, 960x720, no video error). The proposed
`+13.66 mm` preinsert offset moved the commanded target `32.47 mm` from D24's
setting. The first severe failure occurred at controller step 648, on the first
downward `closed_loop_preinsert` waypoint: Hub speed reached 0.995 m/s and
Hub-Casing force 13.7 kN. Later records peaked at 71.9 kN; the final Hub was
`[0.50133, 0.03448, 1.04267] m`, with 26.14° orientation error. Transport
scoring fell to 0/1, total score to `3/6`, and the release guard kept both jaws
closed. The `32.47 mm` correction was too large for the current dual-arm route;
the linear one-to-one lead assumption is falsified. The 80-step hold was
removed but could not be evaluated independently because the grasp had already
failed before seat completion.

Resume the loop under the user's current instruction (superseding the earlier
D21--D23 stop note). D26 will use the nominal preinsert X target (`0.0 m`), an
`18.81 mm` shift from D24 rather than D25's `32.47 mm`, and keep post-seat hold
at zero. All other D22 settings, the guarded release, and scoring thresholds
remain unchanged. Inspect the overhead frames for XY tracking and the contact
trace for the first stable seating point before selecting another correction.

### D26: nominal socket-center X target, immediate seat check

The run `validation_logs/m1_gt_overhead_center_preinsert_noposthold_20261007T0212/`
completed with the same 847-frame four-view recording. The zero X offset
avoided D25's high-energy impact, but did not track the target in Y. At the
above-socket waypoint the Hub was `[0.53093, 0.08012, 1.20668] m`; after the
high-plane preinsert XY move it was `[0.54616, 0.04394, 1.18889] m`, a 42.9 mm
Y shift away from the socket center. At the guarded seat check it was
`[0.56604, 0.01677, 1.07623] m` (71.9 mm radial error, 14.2 mm axial error,
11.52° orientation error). The Hub was slow at 0.0036 m/s with 55.8 N casing
contact, but was nowhere near the socket XY target; release correctly remained
blocked. Score was `3/6`.

The X change alone therefore exposes a stale-grasp-frame problem: transport
continues to use the lift-time dual-arm link-to-Hub offsets even after the Hub
and wrists have shifted relative to one another. The existing
`--reanchor-at-insertion-above` path refreshes both arms' offsets from measured
poses before this sensitive lateral move. D27 will enable only that option
relative to D26; retain X offset `0.0 m`, Y offset, immediate seat check,
slow descent, guard, and all thresholds.

### D27: re-anchor both arm frames above the socket

The valid retry is
`validation_logs/m1_gt_overhead_reanchor_center_nopost_retry_20261007T0318Z/`
(`inner_wall_probe.mp4`, 847 frames at 960x720, no video error). A preceding
attempt failed in viewport initialization before the rollout and is not a
control result. D27 adds only `--reanchor-at-insertion-above` to D26.

Re-anchoring did not resolve the lateral tracking error. The Hub began the
high-plane registration at `[0.53093, 0.08012, 1.20668] m`; after four XY
segments it was `[0.54523, 0.03764, 1.18845] m`, moving 42.3 mm away from the
socket's Y coordinate despite the commanded target being approximately
`[0.55000, 0.07990] m`. At the end of the subsequent preinsert descent it was
`[0.54967, 0.03152, 1.09043] m`. The vertical seat then began with a 55.3 mm
Y error; as configured, the seat held X/Y fixed instead of correcting that
error while descending.

The seat generated a 62.7 kN Hub-Casing force transient and 2.40 m/s Hub speed.
At the guarded check the Hub was `[0.51462, 0.03995, 1.04641] m`: radial error
58.7 mm, axial error 15.6 mm, orientation error 30.57°, and maximum bolt-point
error 110.4 mm. The guard correctly kept both grippers closed. Scores were
`3/6` ROCO-style and `2/4` physical-place. D27 slightly reduced radial error
versus D26 (71.9 mm to 58.7 mm), but worsened orientation and contact loading;
it is not a successful placement.

The trace localizes the next issue to Cartesian wrist tracking during the
high-plane XY/preinsert actions, not camera parallax: the Hub's measured root
position confirms the overhead view, while full-pose IK fails to keep the two
grasping wrists at their commanded targets. D28 will keep D27's measured
re-anchor and all targets, routes, release guards, and thresholds, and enable
the existing position-only IK option for transport and preinsert registration.
This tests whether removing the unnecessary wrist-orientation constraint lets
both wrists track XY without twisting the physical pinch.

### D28: position-only transport and preinsert registration

The rollout is
`validation_logs/m1_gt_overhead_position_only_reanchor_nopost_20261007T0337Z/`
(`inner_wall_probe.mp4`, 847 frames at 960x720, no video error). The command
line was checked to confirm that `--transport-position-only`,
`--reanchor-at-insertion-above`, `--seat-position-only`, and
`--release-only-if-seated` all reached the simulator.

This improved high-plane tracking: after the XY registration, the Hub was
`[0.54467, 0.08046, 1.21563] m`, about 5.4 mm radially from the commanded
`[0.55000, 0.07990] m` target, versus a 42.3 mm Y error in D27. Collision-free
transport recovered (ROCO-style score `4/6`), with the same `2/4`
physical-place score. The Hub then drifted during lowering; at the preinsert
check it was `[0.54073, 0.07686, 1.12679] m`, and at the end of the seat it
was `[0.54852, 0.07186, 1.11390] m`. The final state remained 15.0 mm radially
and 51.9 mm axially above the target, with 15.74° orientation error and
71.2 mm maximum bolt-point error. There was no Hub-Casing contact at the seat
check, so release remained blocked. The Hub descended only 12.9 mm during the
seat trajectory instead of the required roughly 64.8 mm.

Position-only IK fixed the XY transport failure but did not provide enough
vertical wrist motion under the dual-arm grasp. D29 will preserve D28's
position-only transport/reanchor path and switch only the final controlled
seat back to the existing full-pose IK mode. This tests whether the constrained
seat phase needs orientation-coupled wrist motion after the better XY
registration; release guarding and thresholds stay unchanged.

### D29: full-pose IK only for the final seat

The run is
`validation_logs/m1_gt_overhead_position_only_fullpose_seat_reanchor_20261007T0351Z/`
(`inner_wall_probe.mp4`, 847 frames at 960x720, no video error). It preserves
D28's position-only transport/reanchor and changes only the final seat back to
full-pose IK.

This did not improve seating. The rollout remained `4/6` ROCO-style and `2/4`
physical-place; no Hub-Casing contact was measured and the seated-only release
guard kept both grippers closed. The final Hub root was
`[0.54677, 0.07315, 1.11412] m`, with 14.1 mm radial error, 52.1 mm axial
error, 17.82° orientation error, and 74.8 mm maximum bolt-point error. It
descended only 12.7 mm during the seat path, essentially the same as D28's
12.9 mm. The two IK modes therefore fail at the same stage: the commanded
dual-arm seat does not move the wrist/Hub assembly down to the physical socket.

D30 will return to D28's position-only seat and disable only `seat-vertical-only`.
At D28's preinsert point the Hub was 13.6 mm radially from the target, just
outside the unchanged 12 mm limit. The current vertical-only mode freezes that
residual XY error for the whole descent. With the improved registration, an
8x28 gradual diagonal seat asks for only about 1.7 mm lateral correction per
segment; test that measured correction under PhysX, retaining the same camera,
grasp, dual-arm route, re-anchor, and release guard.

### D30: diagonal correction during the slow seat

The four-camera rollout is
`validation_logs/m1_gt_overhead_diagonal_seat_posonly_reanchor_20261007T0401Z/`
(`inner_wall_probe.mp4`, 847 frames at 960x720, no video error). This differs
from D28 only by allowing gradual XY correction during the 8x28 seat.

The Hub descended 27.1 mm, more than twice D28's 12.9 mm, but moved away from
the XY target during that descent. It ended at `[0.55965, 0.06919, 1.09973] m`
with 20.1 mm radial error, 37.7 mm axial error, 17.81° orientation error, and
58.4 mm maximum bolt-point error. Hub-Casing force remained zero, so it had not
reached the socket; the release guard kept both jaws closed. Scores remained
`4/6` and `2/4`.

The per-arm trace is asymmetric: during the seat, left link6 lowered only
6.4 mm (`1.12765` to `1.12128 m`), while right link6 lowered 41.8 mm
(`1.15811` to `1.11627 m`). The left arm is the limiting tracker; its lag lets
the Hub drift in X while the right arm continues the descent. D31 will keep
D30's diagonal path and position-only IK, and enable the existing bounded
Jacobian-position update for the left arm during controlled seat only. The
right arm remains on its current position-only IK target. This isolates the
left-wrist tracking shortfall without changing timing, targets, reset, or
acceptance criteria.

### D31: bounded Jacobian update for the left wrist

The run is
`validation_logs/m1_gt_overhead_diagonal_jacobian_seat_posonly_reanchor_20261007T0410Z/`
(`inner_wall_probe.mp4`, 847 frames at 960x720, no video error). It adds the
existing left-arm Jacobian-position seat update to D30; the right arm keeps
position-only IK.

This did not seat the Hub. The score stayed `4/6` and `2/4`; there was no
Hub-Casing contact, and the release guard kept both grippers closed. Final Hub
root was `[0.55167, 0.07252, 1.11250] m`, with 14.4 mm radial error, 50.5 mm
axial error, 16.34° orientation error, and 69.9 mm maximum bolt-point error.
The left wrist lowered only 3.1 mm and the right 20.8 mm, so the left Jacobian
update did not remove the vertical reach/tracking limit.

The robot bundle defines a nonzero torso posture `(0.5, -0.8, 0.5, 0.0)` for
the R1 arm workspace, but this environment currently locks the torso at its
authored all-zero posture because Isaac Sim 5.1 was observed to let widened
zero-width torso joints diverge. The measured seat failures now give a concrete
reason to test that documented workspace posture: both arms plateau well above
the socket even when commanded through different IK modes. D32 will make a
run-scoped opt-in to the existing torso reset/limit override, keep D31's
controller and geometry unchanged, and check whether the posture restores the
required vertical reach without torso drift or loss of the Hub grasp. The
default remains locked unless the full rollout proves this mode stable.

### D32 result: torso override invalidates the calibrated grasp route

The valid retry is
`validation_logs/m1_gt_overhead_torso_workspace_diag_jacobian_retry_20261007T0421Z/`
(`inner_wall_probe.mp4`, four views, 847 frames at 960x720, no video error).
A preceding invocation failed during viewport initialization before physics
and is excluded. With the nonzero torso posture, the reset arm poses changed
substantially from D31: left link6 moved to `[0.437, 0.222, 1.279] m` and right
link6 to `[0.356, -0.290, 1.072] m`. The intended Hub pinch was not formed:
only two same-side contacts were measured, neither right fingertip contacted,
and the candidate was `CONTACT_WITHOUT_HELD_FOLLOW`. The Hub displaced 72 mm,
then lost both grippers during lift; the later commanded arm path diverged and
does not provide seat-reach evidence. Scores were **2/6** and **2/4**.

Reject the torso override as an isolated change; keep its default off. The
valid D28--D31 route already completed dynamic bilateral lift and transport.
Its measured D28 preinsert Hub was `[0.54073, 0.07686, 1.12679] m` versus the
socket XY target `[0.55000, 0.08684] m`; the configured preinsert Y offset
deliberately shifted the waypoint another `6.93 mm` away from socket center,
and the vertical-only seat then froze that residual. D33 was intended to
return to the D28 route and change only `preinsert_offset_y_m` from
`-0.00693427 m` to `0.0 m`, but the first launch omitted D28's
`transport_clearance_z_m=0.20 m` and inherited the wrapper default `0.55 m`.
Do not compare that rollout as an isolated Y-offset test. D34 repeats the
planned change with the transport clearance explicitly matched to D28; the
seat controller, simulated-time horizon, physics, release guard, and scoring
remain unchanged.

### D33 result: route mismatch invalidates the preinsert-offset comparison

The full rollout and video are in
`validation_logs/m1_gt_overhead_preinsert_center_posonly_reanchor_20261007T0432Z/`
(`inner_wall_probe.mp4`, 847 frames, 960x720, no video error). It scored
`4/6`; bilateral grasp/lift and 11/11 transport contact samples passed, but
the Hub missed seat and guarded release was skipped. At `insertion_above` the
root was `[0.56330, 0.08717, 1.55956] m`, confirming that the unintended
`0.55 m` clearance changed the trajectory. During preinsert it moved to
`[0.51426, 0.10013, 1.11860] m`; the seat endpoint was
`[0.52685, 0.10689, 1.09390] m`. The video is genuine and watchable, but the
large tracking error cannot be attributed to the Y-offset change alone. The
60-second horizon and all acceptance thresholds were still unchanged.

### D34 result: matched center-Y preinsert registration

D34 is the valid matched run in
`validation_logs/m1_gt_overhead_preinsert_center_matched_posonly_reanchor_20261007T0445Z/`
(`inner_wall_probe.mp4`, 847 frames at 960x720, no video error). It matches
D28's `0.20 m` transport clearance and changes only the preinsert Y offset to
zero. Score remains **4/6** (`2/4` physical-place): bilateral grasp/lift and
all 11 transport samples passed; the unchanged seated-release guard skipped
release.

The preinsert Hub moved to `[0.54158, 0.08124, 1.12874] m`, reducing radial
error from D28's `13.6 mm` to `10.1 mm`. During the eight-segment vertical seat
it descended only `13.5 mm` and ended at `[0.54926, 0.07645, 1.11529] m`.
Final radial error was `10.4 mm`, axial error `53.3 mm`, orientation error
`16.34°`, bolt-point error `73.5 mm`; Hub-Casing force was zero. The corrected
XY registration therefore improves the radial metric but does not explain the
vertical stall. Keep the center-Y waypoint. D35 will preserve this full D34
configuration and enable the existing above-socket held-orientation
correction with a bounded `0.5°` per-segment step. This checks whether the
measured `16.34°` attitude error is preventing the socket approach while the
part still has clearance; the descent, 60-second horizon, and release gate
remain unchanged.

### D35 result: above-socket orientation correction improves XY/orientation only

D35 is
`validation_logs/m1_gt_overhead_center_y_orientation_above_20261007T0459Z/`
(`inner_wall_probe.mp4`, 935 frames at 960x720, no video error). It retains
D34's matched route and adds the bounded `0.5°` above-socket correction. The
score remains **4/6** (`2/4` physical-place); all bilateral grasp/lift and
transport points pass, while seat and release remain blocked.

The final root was `[0.54724, 0.08393, 1.11251] m`: radial error improved
from D34's `10.4 mm` to `4.0 mm`, and orientation error from `16.34°` to
`11.07°`; bolt-point error improved from `73.5 mm` to `60.2 mm`. However, the
Hub descended only `11.6 mm` from preinsert and remained `50.5 mm` above the
seat target. There was no Hub-Casing contact, final speed was `0.153 m/s`, and
the release guard kept the jaws closed. D35 confirms that bounded
preorientation helps the 6-DoF pose but does not explain the seat-height
stall.

D36 preserves the entire D35 route and changes only `seat_vertical_only` from
true to false, allowing the existing 8x28 gradual XYZ seat to correct the
remaining measured XY residual while descending. D30's diagonal seat already
showed greater axial progress than vertical-only, but it drifted from a less
centered preinsert pose and had no above-socket orientation correction. D36
tests that same geometric correction from D35's better aligned state. The
target, native 60-second horizon, physics, release guard, and acceptance stay
fixed.

### D36 result: diagonal seat does not overcome the axial stall

D36 is
`validation_logs/m1_gt_overhead_center_y_orientation_diagseat_retry_20261007T0511Z/`
(`inner_wall_probe.mp4`, 935 frames at 960x720). It scores **4/6** (`2/4`
physical-place): grasp, lift, and collision-free transport pass; seating and
release fail. Final radial error is `2.97 mm`, axial error `50.66 mm`,
orientation error `10.80°`, and maximum bolt-point error `58.02 mm`. Hub-to-
Casing force is zero, so the part never reaches the socket; the release guard
correctly keeps both grippers closed. Diagonal XY correction is not the cause
of the vertical stall.

### D37 result: two-wrist Jacobian seat still stalls

D37 is
`validation_logs/m1_gt_overhead_direct_positiononly_dualjacobian_20261007T0952Z/`
(`inner_wall_probe.mp4`, 935 frames at 960x720; no video error). It preserves
D36's valid route and applies the existing bounded positional-Jacobian update
to both loaded wrists during the final controlled-seat segment. Score remains
**4/6** (`2/4` physical-place). The Hub descends `11.02 mm` during the eight
seat segments, ending at `[0.54846, 0.08432, 1.11286] m`: radial error
`2.96 mm`, axial error `50.86 mm`, orientation error `11.83°`, and maximum
bolt-point error `61.34 mm`. There is still no Hub-Casing contact, and the
release guard correctly skips release.

The wrists themselves move only `2.68 mm` downward on the left and `17.47 mm`
on the right over the seat, despite a nominal `61.9 mm` root descent target.
Neither shoulder joint reaches its soft limit (right joint 2 ends at `3.098`
rad vs `3.229` rad upper bound), so the remaining failure is not explained by
the previously observed transport saturation. This falsifies the hypothesis
that applying the same positional Jacobian to both wrists alone will recover
the seat. Next, record each segment's commanded joint targets, measured joint
positions/velocities, and wrist target error; that will separate a bad
Jacobian target from actuator/grasp/contact under-tracking before changing the
seat controller again. The official horizon, physical scoring, contact
conditions, and release gate remain unchanged.

### D38 result: measured joints lag commanded positions

D38 is
`validation_logs/m1_gt_overhead_dualjacobian_tracking_telemetry_20261007T1008Z/`
(`inner_wall_probe.mp4`, 935 frames at 960x720; no video error). Repeating
D37's exact motion with target/actual joint telemetry reproduces the same
**4/6** score and final pose to the recorded precision: radial error
`2.96 mm`, axial error `50.86 mm`, orientation error `11.83°`, bolt-point
error `61.34 mm`, and zero Hub-Casing force. Release remains correctly
blocked.

The controller issues changing position targets, but by controlled-seat
segment 8 the measured joints lag their commanded positions by up to `0.065 rad`
(left) and `0.064 rad` (right), while several measured joint velocities are
still nonzero (left wrist joint 6: `3.80 rad/s`). This is consistent with the
eight coarse target updates not tracking the moving seat trajectory. Next
experiment: recompute the same bounded Jacobian correction at each existing
environment step inside each seat segment, keeping the same total step count
and 60-second horizon. This changes feedback update frequency only; it does
not extend the task or weaken acceptance.

### D39 result: per-environment-step feedback improves alignment, not seating

D39 is
`validation_logs/m1_gt_overhead_dualjacobian_feedbackstep_20261007T1026Z/`
(`inner_wall_probe.mp4`, 935 frames at 960x720; no video error). It keeps the
D38 physical route and total simulated steps but recomputes each arm's
position-Jacobian update every environment step. Score remains **4/6**
(`2/4` physical-place). Relative to D38, orientation error improves from
`11.83°` to `8.82°` and bolt-point error from `61.34 mm` to `53.25 mm`; radial
error stays near `3.03 mm`. However, Hub descent changes only from `11.02 mm`
to `12.39 mm`; final axial error is still `49.49 mm`, Hub-Casing force remains
zero, and release is still correctly blocked. The wrists remain roughly
`3.19 mm` (left) and `16.83 mm` (right) lower than at seat start. Per-step
feedback alone does not explain or fix the axial stop.

Next diagnostic: test whether Robot↔Casing collision obstructs the final
approach. Use the existing calibration-only collision-pair filter while
keeping Hub↔Casing and both gripper↔Hub contacts enabled. This run is strictly
causal diagnosis, not a valid success result; if motion changes, the fix must
be a collision-free physical arm path, not leaving the filter enabled.

### D40 diagnostic: Robot-Casing collision is a major descent obstruction

D40 is
`validation_logs/m1_gt_overhead_robot_casing_collision_probe_20261007T1040Z/`.
It uses the D39 controller and filters all 30 Robot collision shapes against
the single Casing collider, while preserving Hub-Casing and gripper-Hub
contacts. This is **not an acceptance run**. With the pair filter, the Hub
descends `41.21 mm` from preinsert (versus D39's `12.39 mm`) and develops
`223.29 N` Hub-Casing contact. Final axial error drops to `17.17 mm`, but
radial error is `11.73 mm`, orientation error `12.35°`, maximum bolt-point
error `33.23 mm`; the official score remains **4/6**, release remains
blocked. This is strong causal evidence that Robot-Casing collision blocks
most of the descent, but not enough to identify which arm/link or to claim a
valid placement. The next test should isolate the obstructing side, then
replace the diagnostic filter with an actual collision-free arm path.

### D41-D42 diagnostics: both arms obstruct the descent; right is primary

D41 filters only the left arm's nine collider shapes and preserves every
right-arm↔Casing collision:
`validation_logs/m1_gt_overhead_robot_casing_left_probe_20261007T1100Z/`.
It scores **4/6**; the Hub descends `30.94 mm` from preinsert and ends
`27.44 mm` axially high, `5.24 mm` radial error, with `131.84 N` Hub-Casing
force. The guarded release remains blocked.

D42 filters only the right arm's nine collider shapes:
`validation_logs/m1_gt_overhead_robot_casing_right_probe_20261007T1110Z/`.
It also scores **4/6**; descent is `38.31 mm`, axial error `23.58 mm`, radial
error `23.47 mm`, orientation error `11.70°`, bolt-point error `49.66 mm`,
and Hub-Casing force `223.38 N`. Release remains blocked.

Together with D39 (no filter, `12.39 mm` descent) and D40 (both sides filtered,
`41.21 mm` descent), these controlled comparisons show that **both arm sides
hit the casing during descent, with the right side the larger single-side
obstruction**. Filtering remains diagnostic only. The next valid controller
test changes the grasp/wrist approach geometry while restoring all nominal
robot-fixture collisions; do not promote either filtered rollout to success.

### D43 result: `radial_y90` loses the load-bearing grasp

D43 is `validation_logs/m1_gt_overhead_radialy90_clearance_20261007T1115Z/`
(`inner_wall_probe.mp4`, 935 frames at 960x720; no video error). It restores all
normal Robot-Casing collisions and changes only the grasp/wrist orientation to
`radial_y90`. The run ends at **1/6** (dynamic reset only), with verdict
`CONTACT_WITHOUT_HELD_FOLLOW`: gripper/link motion is `151.84 mm`, while the
Hub moves only `0.20 mm` after close. Reported peak gripper contact is high
(`588.18 N`), but the grasp/lift criterion is not met. The Hub remains about
`274.66 mm` radially from the seat target, has zero Hub-Casing force, and the
insertion verdict is `INSERTION_NOT_VERIFIED`. The existing release guard
correctly skips opening the grippers because seating was not verified.

This rejects `radial_y90` as a usable grasp correction; it fails earlier than
the prior `radial` runs and does not resolve the arm-Casing descent obstruction
seen in D39-D42. Keep the accepted collision and release checks unchanged. The
next experiment should return to the load-bearing `radial` grasp and adjust the
arm path/clearance at the casing, rather than further changing grasp angle or
masking collisions.

### D44 protocol: RoCo-scale tabletop and measured radial pinch

Hypothesis: the previous strict scene makes the work surface visually and
physically inconsistent with the task: a `1.5 x 1.2 m` box is placed at
`z=-0.05 m`, while independent staging supports hold the parts around `z=1 m`.
Use a `3.0 x 2.0 m` tabletop with its top at `z=0.934 m`, scatter-reset the
dynamic Hub and Casing onto that collision surface, and remove elevated staging
supports. Start the Hub at `(0.30, 0.43)` so its CAD footprint is separated
from the Casing and fully inside the table bounds. Revert to the known-valid
`radial` pinch (`0.085 m` pair offset); D39 contact telemetry measured inner
and outer radii at approximately `0.058 m` and `0.123 m`, whereas D43's
`radial_y90` missed both walls. Keep the controller, full gravity, strict
scoring, all nominal robot-fixture collisions, and release guard unchanged.

Prediction: reset geometry reports both CAD bottoms within the existing `2 mm`
spawn margin of the large tabletop; bilateral tip telemetry again reports one
inner-wall and one outer-wall contact; Hub follows both arms during lift and
transport. This tests the requested scene/grasp corrections, not a scoring
threshold change. The run is `validation_logs/m1_gt_roco_table_radial_contact_20261007/`.

### D44 result: table is fixed; Hub is ejected before a bilateral pinch

D44 completed with a valid 2,683-frame video and no encoder error. The new
`3.0 x 2.0 m` tabletop at `z=0.934 m` is supporting both dynamic CAD parts
directly: their measured reset bottoms are `2 mm` above the surface and both
footprints are within the table bounds. This removes the previous raised-pad /
low-table mismatch.

The requested radial contact did not pass in this source location. With the Hub
at `(0.30, 0.43)`, the left approach already tips and displaces it before the
second arm closes; the Hub root moves from `(0.30, 0.43, 0.948)` to about
`(0.356, 0.467, 1.013)` during the first insertion approach, with high angular
velocity. At closure only a left inner-wall contact is classified; the outer
wall and right-arm contacts are absent. A reported `7.49 kN` peak is therefore
a collision impulse, not evidence of a grasp. The physical score is **1/4**
and RoCo-style score **2/6**; insertion is unverified and the release guard
correctly keeps the grippers closed.

### D45 protocol: separate the parts in X; retain the validated radial pinch

The evidence isolates the remaining immediate cause: the `(0.30, 0.43)` source
pose requires the left arm to reach diagonally across the table and destabilizes
the unsupported Hub before the mirrored jaw pair is established. D45 moves the
Hub to `(0.15, 0.0)`, leaving its full CAD footprint on a slightly wider
`3.2 x 2.0 m` tabletop and creating a `50 mm` X gap from the Casing footprint.
The Hub returns to the centerline used by the prior successful radial contact
test. The gripper orientation, `85 mm` radial pair offset, controller, horizon,
full gravity, collision pairs, scoring thresholds, and release guard are
unchanged. Prediction: no reset overlap/ejection; both inner and outer wall
contacts are present before lift. Run output:
`validation_logs/m1_gt_roco_table_radial_contact_x015_20261007/`.

### D45 result: direct tabletop reset still destabilizes the grasp

D45 completed with a valid 2,683-frame video. The `3.2 x 2.0 m` tabletop
contains both parts, and their reset bottoms are each `2 mm` above the surface.
However, during the left then right approach the Hub is displaced and rotated;
at the right approach its speed reaches `0.20 m/s`, and at closure the Hub is
moving at `1.03 m/s`. Only one transient right-link contact is reported
(`780.62 N`); no sample verifies one-inner/one-outer contact. The final Hub is
still at the source `(0.150, 0.000, 0.948)`, about `422 mm` radially from the
seat, so the strict score is **2/6** and release is correctly skipped.

The trace shows the remaining cause: moving the Hub onto the `z=0.934 m`
tabletop makes the existing high-to-low radial approach traverse the annulus
from too far above and destabilize the part. It is not a table-size failure.
Next preserve the enlarged tabletop and direct Casing support, but use the
existing physical Hub staging pads to present the Hub at its previously
validated grasp height (`z≈1.08 m`). This keeps the object dynamic and
gravity-supported while separating the approach-height effect from the table
change. D46 output: `validation_logs/m1_gt_roco_table_staged_radial_20261008/`.

### D46 protocol: raised physical cradle, known-valid radial contact

Use the `3.2 x 2.0 m` tabletop at `z=0.934 m`; keep the Casing directly
supported at `z=0.935 m`; support the dynamic Hub on the existing four kinematic
staging pads with their top at `z=1.066 m`. The CAD-derived Hub root therefore
remains about `1.08 m`, matching the prior radial grasp evidence while a real
support reaction remains enabled. Place Hub at `(0.30, 0.0)` and Casing at
`(0.55, 0.0)`, restore the measured `radial` tool pose and `0.085 m` offset,
and retain all nominal robot-fixture collisions, full gravity, the 60 s horizon,
strict scoring, and the seated-only release guard. Prediction: the two-arm
contact topology and held-follow test recover; the remaining question is
whether the existing dual-arm seat path can reach Hub-Casing contact without
the arm-casing interference diagnosed in D39-D42.

### D46 result: the raised Hub and Casing are too close for a clean grasp

D46 completed with a valid 2,683-frame video and no encoder error. The large
`3.2 x 2.0 m` table and reset support checks pass: the Casing is supported
`2 mm` above the tabletop, and the Hub is supported `2 mm` above its physical
staging pads. However, the Hub was placed at `(0.30, 0.0)` while the Casing was
at `(0.55, 0.0)`. Their projected X bounds overlap (`Hub 0.17–0.43 m`,
`Casing 0.33–0.77 m`), leaving only `19 mm` of vertical clearance between the
Hub's predicted bottom and the Casing's top. During closure the Hub is pushed
to `(0.446, 0.008, 1.063)` and then `(0.649, -0.079, 1.063)`; the trace shows
roughly `56 N` Hub-Casing force, while every gripper-to-Hub fingertip force is
zero and no inner/outer wall contact is classified. Score is **2/6** (the
strict grasp and transport criteria fail); no insertion is verified and the
release guard correctly keeps both grippers closed. Thus the enlarged table
and raised physical support do not by themselves fix the approach: the staged
Hub is too close to the Casing/arm path.

### D47 protocol: keep the validated height, move the Hub clear of the Casing

Keep the same enlarged table, four physical Hub pads, direct Casing support,
radial orientation, `85 mm` radial pair offset, controller, strict score,
nominal collision pairs, and release guard. Change only the Hub reset X from
`0.30 m` to `0.15 m`, retaining its supported grasp height near `1.08 m` and
the Casing at `0.55 m`. This leaves an estimated `50 mm` gap between the
Hub/Casing horizontal footprints while preserving the previously measured
inner/outer-wall pinch geometry. Run output:
`validation_logs/m1_gt_roco_table_staged_radial_x015_20261008/`.

### D47 result: x=0.15 m is unstable before grasp

D47 completed with a valid 2,683-frame video and no encoder error. Reset
geometry is inside the enlarged tabletop, with the Hub bottom `2 mm` above its
four physical supports and the Casing bottom `2 mm` above its support. But
after only 50 gravity-settle steps, before either jaw closes, the Hub has moved
from `(0.15, 0, 1.082)` to `(1.148, -0.402, 0.948)`—a `1.084 m` displacement.
No fingertip force/contact or Hub-Casing force is recorded. The dynamic body is
already off its support at the `gravity_reset_settle` checkpoint, so later arm
motion cannot grasp it; score is **1/6**, with verdict
`COLLISION_EJECTION_OR_APPROACH_DRIFT`. The current telemetry does not identify
which robot/support collision caused the initial ejection; moving the Hub
toward the robot-side table edge is rejected.

### D48 protocol: move the source away in Y while retaining safe X

Retain Hub `x=0.30 m`, the `z≈1.08 m` physical-pad presentation, casing
`(0.55, 0.0)`, large table, radial tool orientation, `85 mm` pinch offset,
controller, collisions, score, and release guard. Change only Hub `y` to
`0.43 m`, separating the two parts laterally without moving the Hub closer to
the robot's X-side boundary. The low direct-table D44 attempt at this Y was
destabilized by a longer high-to-low approach; this run restores the validated
raised grasp height to isolate that factor. Run output:
`validation_logs/m1_gt_roco_table_staged_radial_y043_20261008/`.

### D48 result: lateral source placement is outside the right arm's reach

D48 completed with a valid 2,683-frame video and no encoder error. The Hub
remains stable on the physical cradle through both approach checkpoints, but
the right gripper does not reach it: immediately before closure, the left jaw
links are around `y=0.46–0.58 m` while the right jaw links remain around
`y=-0.08–0.01 m`, versus the Hub center at `y=0.43 m`. No fingertip contact is
classified. Closure pushes the Hub `140 mm` in X and it drops `134 mm` onto the
table; no load-bearing grasp follows. Score is **1/6**, insertion is unverified,
and release remains guarded. This rules out moving the source laterally as a
dual-arm clearance fix.

### D49 protocol: preserve the reachable grasp; move the Casing along X

Return the Hub to the validated radial-grasp pose `(0.30, 0.0)` at the same
physical support height. Move only the Casing from `x=0.55 m` to `x=0.75 m`
(keep `y=0` and its table support). This opens an estimated `100 mm` gap
between their CAD X bounds while retaining the source position previously
shown to admit bilateral inner/outer-wall contact. Keep the controller,
transport, collisions, scoring, and guarded release unchanged. This tests
whether a longer but collision-free X transfer lets the grasp survive and
allows seating on the enlarged table. Run output:
`validation_logs/m1_gt_roco_table_staged_radial_casing_x075_20261008/`.
