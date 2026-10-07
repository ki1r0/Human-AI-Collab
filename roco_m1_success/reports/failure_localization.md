# Failure localization log

## Planet 3, first attempt

- Preceding official score: 2.
- Expected relation: third sun gear concentric with the remaining carrier pin,
  within 2 mm XY, 0.1 rad quaternion angle, and 12 mm height difference.
- Actual settled relation: 0.00037 m XY, 0.01040 m height, and 0.12893 rad
  quaternion angle.
- Reachability/grasp: passed; the robot approached, grasped, lifted,
  transported, inserted, released, and retreated.
- Contact evidence: score briefly reached 3 immediately after release, then
  returned to 2 as the carrier yawed and the released gear relaxed in the
  opposite direction.
- Source path: the non-rotating planet branch of
  `GalaxeaRulePolicy.mount_gear_to_target()`.
- Classification: controller release-orientation compensation, not target XY,
  grasp, asset collision invalidity, or evaluator implementation.
- First attempted fix: 15 degrees of wrist pre-rotation. It was too large,
  changed the carrier's contact response, and produced a 0.13205 rad mismatch
  in the opposite configuration (best/final score 2/2).
- Minimal fix under test: retain the official IK target and add a 4-degree
  counter-rotation during only planet 3's final insertion/release hold. This
  covers the 3.3-degree amount by which the first attempt exceeded the
  official tolerance; no object pose or physics setting changes.

## Reducer, first clearance-drop attempt

- Preceding official score: 5; transient best score: 6; settled score: 5.
- Expected relation: reducer concentric and orientation-matched with the center
  gear, with `center_z - reducer_z < 0.002 m` after release.
- Actual settled relation: 0.00070 m XY, 0.01904 rad orientation, but
  `center_z - reducer_z = 0.01001 m`.
- Reachability/grasp/alignment: passed. Score 6 was observed while the reducer
  was held above the stack, but that is not accepted as assembly evidence.
- Physical failure: after opening at 40 mm clearance, the reducer fell through
  the center bore to table height. It briefly crossed the valid height during
  retreat, then continued falling.
- Classification: release height / uncontrolled axial impact.
- Minimal fix under test: release at 25 mm clearance. This reduces the free-fall
  distance so the reducer's transverse shoulder can mate, without reintroducing
  the upstream 30 mm held press that compressed and dragged the stack.

### 25 mm result and revised localization

The 25 mm attempt failed earlier and more severely (best/final 5/2). At the
start of the axial approach, the reducer was 5.06 mm off the center gear. The
23.4 x 15.0 mm reducer shaft therefore struck the face instead of entering the
gear's only slightly larger rectangular bore; the held descent lifted and
scattered the stack. Lowering alone is falsified.

The revised minimal fix continuously corrects measured reducer-to-center XY
error (limited to 10 mm per command), doubles only the axial alignment time,
and then inserts to 15 mm clearance. The correction remains a robot IK command;
it does not write object state or alter collision/contact behavior.

### Closed-loop 15 mm result and controlled-slip revision

At the final release boundary the reducer relation itself was valid (1.60 mm
XY, 0.0122 rad, reducer 0.71 mm below center), but the held press had already
reduced the other five relations to four. This confirms that a low held target
is the wrong mechanism even when centered.

The controller now restores the non-contact 40 mm alignment that preserved all
five preceding relations. Instead of fully opening and free-falling, it first
commands a 10 mm gripper opening for a controlled gravity slip, then fully
opens, settles, and retreats. This is a physical compliant insertion using the
existing gripper actuator; there is no state write, attachment, or collision
change.

### 10 mm slip result and state-based arrest

The 10 mm slip preserved all five preceding relations but still passed through
the valid height in about one 50 ms control interval (best/final 6/5). The next
bounded attempt uses a 7.5 mm slip target, closes when reducer root height is
within 20 mm of center root height, holds for 0.25 s, applies a 1-degree wrist
wedge in the close-fitting rectangular bore, and then fully opens. The wedge is
ordinary collision/friction mating, not an attachment or state constraint.

### State-based arrest result and pre-contact interference fit

The close-and-wrist-wedge attempt retained score 6 only transiently, then the
post-contact arm rotation transferred energy into the stack and scattered it
(final score 1). Post-contact rotation is rejected.

Video plus the STL cross sections show that the reducer otherwise slides fully
through the close rectangular bore, whereas the official height rule requires
a shallow insertion. The next minimal attempt applies a 5-degree roll before
contact (inside the existing 0.1-rad quaternion tolerance), holds the arm
stationary, performs the 10 mm controlled slip, then fully releases. This tests
a passive interference fit without any post-contact arm motion.

### 5-degree fit result and narrow-axis seating bias

The tilt did not hold score 6 and caused avoidable orientation motion (final
score 5), so it is rejected. The narrow-axis shaft/bore half-clearance is only
about 0.2 mm. The next attempt restores zero tilt and adds a 1 mm target bias
along that narrow axis, still far inside the official 5 mm XY relation. This
tests shallow wall-supported seating with the same gentle slip release.

The 1 mm command produced only 0.40 mm measured XY error at release because it
was canceled by the grasp/IK offset; the reducer again slid through (best/final
6/5). The bounded follow-up uses 4 mm commanded narrow-axis bias, expected to
produce roughly 3.4 mm measured seating error, still below the unchanged 5 mm
official relation threshold.

### 4 mm narrow-axis bias result

The reducer-only Isaac Sim run used the existing command
`ROCO_PHASE=reducer ROCO_SEED=23 ROCO_OUTPUT_DIR=roco_m1_success/runs/dev_reducer_bias4mm ./roco_m1_success/run_successful_assembly.sh`.
The only trajectory change was the `0.004 m` y-axis bias at
`controller.py:172`; the source still uses the existing collision-enabled
physics and writes no object poses after initialization.

At the release snapshot (control step 126, physics step 630), the measured
reducer-to-center XY error was `0.003807404 m` (3.807 mm), with the reducer
`0.022838652 m` above the center. This confirms that the 4 mm command produced
the intended wall-side offset, within the unchanged 5 mm XY relation limit.
The reducer then passed through the rectangular bore during settling: the
final center-to-reducer relation was 0.761 mm XY but 9.991 mm below the center
height, so the insertion relation did not remain valid.

The trace localizes the score transition to one 50 ms interval, before full
opening or retreat: at control step 126 / physics step 630 the reducer was
0.935689 m and the center 0.912850 m; at step 127 / 635 it was 0.927075 m;
at step 128 / 640 it was 0.902754 m and the center 0.912747 m. Thus the
reducer fell 32.935 mm over two control intervals, its XY error collapsed from
3.807 mm to 0.701 mm, and the official reducer height difference became
`0.009993 m` (center above reducer), exceeding the unchanged `0.002 m`
criterion in `galaxea_lab_agent_env.py:309-317`. The gripper was still at
0.009534 m at this transition, within the controlled-slip phase
(`controller.py:186-190`), so retreat timing is not the cause.

The official score was transient best `6/6`, then settled to final `5/6`; the
run status is `FAIL` and `task_success` is false. Artifacts are in
`runs/dev_reducer_bias4mm/` (`result.json`, `phase_scores.json`, `trace.npz`,
`episode.mp4`, and `console.log`). Therefore 4 mm improves the measured
lateral seating offset versus the 1 mm run's 0.396 mm, but does not complete
the reducer assembly or pass the full score.

### 4 mm bias with 25 mm release clearance (bounded follow-up)

Hypothesis: keeping the successful 4 mm narrow-axis bias but reducing the
release-height clearance from 40 mm to 25 mm would shorten the free-fall and
let the reducer remain at the official shallow insertion height. This changed
only the reducer controller pose target (`controller.py:171` at the time of
the run); the 4 mm y bias, actuator schedule, collision settings, and official
score rules were unchanged.

The exact run command was:

`ROCO_GPU_DEVICE=3 ROCO_PHASE=reducer ROCO_SEED=23 ROCO_OUTPUT_DIR=/home/sunsiliang/Human-AI-Collab/roco_m1_success/runs/dev_reducer_bias4mm_clearance25mm_abs ./roco_m1_success/run_successful_assembly.sh`

The run falsified the hypothesis. It reached a transient best score of 6/6
at control step 93, but the reducer was already disturbing the stack before
release: at control step 114 its root was 0.925243 m while the center was
0.925879 m, and the center had been lifted from its nominal 0.914 m height.
At the release snapshot (control step 126, physics step 630), the reducer
was only 2.560 mm from the center in XY but 22.329 mm below it; the other
assembled relations had already been displaced. The score fell 6 -> 5 at
control step 106, briefly recovered to 6 at step 107, then fell 5 -> 4 at
step 115, 4 -> 3 at step 117, 3 -> 2 at step 120, 2 -> 1 at step 123, and
1 -> 0 at step 134. The final score was 0/6, `status=FAIL`, and
`task_success=false` (the final center-to-reducer XY error was 1.168 mm, but
the center was 32.156 mm above the reducer and the center itself was no
longer valid).

This identifies the 25 mm pose as pre-contact interference: the reducer is
lowered far enough during the held IK approach to push/lift the existing
stack, rather than merely arresting the later free-fall. It is rejected. The
working tree therefore restores the best-known 4 mm controller behavior with
40 mm release clearance (`controller.py:171-172`). Full artifacts for the
bounded run are `runs/dev_reducer_bias4mm_clearance25mm_abs/result.json`,
`phase_scores.json`, `trace.npz`, `episode.mp4`, and `console.log`.

### 4 mm bias with 35 mm release clearance (bounded midpoint)

Hypothesis: 35 mm would be far enough above the assembled center to avoid the
25 mm pre-contact interference, while reducing the free-fall seen with 40 mm.
The run changed only `release_height` from `0.040` to `0.035` m at
`controller.py:171` during the experiment. The 4 mm y bias, actuator timing,
collision/contact settings, and official scoring code were unchanged.

The exact run command was:

`ROCO_GPU_DEVICE=3 ROCO_PHASE=reducer ROCO_SEED=23 ROCO_OUTPUT_DIR=/home/sunsiliang/Human-AI-Collab/roco_m1_success/runs/dev_reducer_bias4mm_clearance35mm_abs ./roco_m1_success/run_successful_assembly.sh`

The 35 mm approach was clean rather than pre-contact interference. At control
step 125 / physics step 625 the score was 6, the reducer was at `0.933975 m`,
and the center was at `0.912948 m`; at the release snapshot (step 126 / 630)
the reducer was `(0.499772, 0.001899, 0.921631) m`, the center was
`(0.500068, 0.002724, 0.912868) m`, and the measured release relation was
0.876 mm XY with the reducer 8.762 mm above the center. The center did not
lift or scatter during the held approach (`center_geometrically_valid=true`).

It still free-fell after release: by control step 127 / physics step 635 the
reducer z was `0.902830 m` (an 18.800 mm drop in one 50 ms interval), the
center-minus-reducer height was `+9.956 mm`, and the score changed 6 -> 5.
The final reducer pose was `(0.498541, 0.001350, 0.901194) m`, a displacement
of `(-1.231, -0.549, -20.437) mm` from release; its final XY relation was
1.018 mm and it remained 9.991 mm below the center. The official run result
was `initial_score=5`, `best_score=6`, `final_score=5`, `status=FAIL`,
`task_success=false`, with collision enabled and zero post-initialization
object-pose writes. (The score briefly recovered to 6 during retreat, but
settled back to 5.)

This establishes the requested bracket without further sweeping: 25 mm causes
pre-contact interference and final 0/6; 35 mm avoids that interference but
still free-falls and finishes 5/6; 40 mm likewise finishes 5/6. The midpoint
is therefore rejected, and the working tree restores the best-known 40 mm
configuration at `controller.py:171` (with the unchanged 4 mm bias at line
172). Full artifacts are `runs/dev_reducer_bias4mm_clearance35mm_abs/`:
`result.json`, `phase_scores.json`, `trace.npz`, `episode.mp4`, and
`console.log`.

### 5 mm y-bias at 40 mm release clearance (bounded follow-up)

Hypothesis: increasing only the established 4 mm narrow-axis command to 5 mm
could provide enough wall engagement to arrest the reducer while retaining the
40 mm clearance that avoids pre-contact interference. During the experiment
only `release_height[:, 1]` changed from `0.004` to `0.005` m at
`controller.py:172`; release clearance, timing, environment/contact settings,
and official scorer thresholds were unchanged.

The exact run command was:

`ROCO_GPU_DEVICE=3 ROCO_PHASE=reducer ROCO_SEED=23 ROCO_OUTPUT_DIR=/home/sunsiliang/Human-AI-Collab/roco_m1_success/runs/dev_reducer_bias5mm_clearance40mm_abs ./roco_m1_success/run_successful_assembly.sh`

The actual release XY error was `5.103737 mm` at control step 126 / physics
step 630, exceeding the official strict `<5 mm` reducer relation limit by
0.104 mm. Therefore 6/6 was attained transiently before release (for example,
at step 93), but not at release: the release snapshot scored 5/6. The reducer
was at `(0.500863, 0.004635, 0.935830) m`, the center at
`(0.500249, -0.000431, 0.912846) m`, and the reducer was 22.984 mm above the
center. Unlike the 35/40 mm runs, it did not free-fall through the bore. Its
z stayed around 0.935 m while the trajectory developed side interference:
the final reducer pose was `(0.488797, 0.010998, 0.936683) m`, with a
12.607 mm XY relation and 0.472 rad relative quaternion angle. Its
release-to-final displacement was `(-12.065, +6.363, +0.853) mm`. The
official result was `initial_score=5`, `best_score=6`, `final_score=4`,
`status=FAIL`, and `task_success=false`; collision remained enabled and
post-initialization object-pose writes remained zero.

The 5 mm command is rejected immediately for violating the release XY limit
and for causing contact/interference rather than controlled seating. The
working tree restores the 4 mm bias at `controller.py:172`; all other 40 mm
trajectory values remain unchanged. Full artifacts are
`runs/dev_reducer_bias5mm_clearance40mm_abs/`: `result.json`,
`phase_scores.json`, `trace.npz`, `episode.mp4`, and `console.log`.

### 4.5 mm y-bias at 40 mm release clearance (bounded midpoint)

Hypothesis: a 4.5 mm y-bias could preserve the 4 mm run's release alignment
while adding enough wall engagement to prevent the reducer's post-release
fall. During the experiment only `release_height[:, 1]` changed from `0.004`
to `0.0045` m at `controller.py:172`; the 40 mm clearance, timing,
environment/contact settings, and official scorer thresholds were unchanged.

The exact run command was:

`ROCO_GPU_DEVICE=3 ROCO_PHASE=reducer ROCO_SEED=23 ROCO_OUTPUT_DIR=/home/sunsiliang/Human-AI-Collab/roco_m1_success/runs/dev_reducer_bias4p5mm_clearance40mm_abs ./roco_m1_success/run_successful_assembly.sh`

The actual release snapshot (control step 126 / physics step 630) was score
6/6, with reducer pose `(0.500899, 0.004034, 0.935800) m`, center pose
`(0.500300, -0.000273, 0.912842) m`, XY error `4.349 mm`, reducer 22.958 mm
above center, and relative quaternion angle `0.0106 rad`. Thus 6/6 was
attained both before and at release, and the release XY error remained below
the strict 5 mm limit.

The reducer did not remain stably seated. During the first settle snapshot
(control step 141 / physics step 705), it was still at `0.935131 m` rather
than free-falling, but its XY error had grown to `5.457 mm` and the score was
5. It then descended; at step 150 / physics step 750 it was at `0.916354 m`
and score 6, but at step 151 / 755 it reached `0.907170 m`, 4.394 mm below
the center, and score 5. This is delayed wall/contact interaction followed by
axial overshoot, not stable retention. Relative orientation stayed within the
official tolerance (release `0.0106 rad`, final `0.0200 rad`).

The final reducer pose was `(0.497603, -0.006398, 0.901200) m`, giving a
release-to-final displacement of `(-3.296, -10.432, -34.600) mm` (36.288 mm
norm). Final reducer-center XY error was `0.876 mm`, but the reducer was
9.980 mm below center. The official result was `initial_score=5`,
`best_score=6`, `final_score=5`, `status=FAIL`, `task_success=false`, with
collision enabled and zero post-initialization object-pose writes.

The midpoint is rejected because it does not achieve stable final 6/6 even
though its release XY error is valid. The working tree restores the 4 mm bias
at `controller.py:172`; full artifacts are
`runs/dev_reducer_bias4p5mm_clearance40mm_abs/`: `result.json`,
`phase_scores.json`, `trace.npz`, `episode.mp4`, and `console.log`.

### 4.75 mm y-bias at 40 mm release clearance (final bounded midpoint)

Hypothesis: a 4.75 mm command could add enough narrow-axis wall engagement to
retain the reducer while remaining inside the official 5 mm release XY limit.
Only `release_height[:, 1]` changed from `0.004` to `0.00475` m at
`controller.py:172` during this run. The 40 mm clearance at line 171, actuator
timing, collision/contact settings, and official scorer thresholds were
unchanged.

The exact run command was:

`ROCO_GPU_DEVICE=3 ROCO_PHASE=reducer ROCO_SEED=23 ROCO_OUTPUT_DIR=/home/sunsiliang/Human-AI-Collab/roco_m1_success/runs/dev_reducer_bias4p75mm_clearance40mm_abs ./roco_m1_success/run_successful_assembly.sh`

At release (control step 126 / physics step 630), the official snapshot was
6/6. The reducer pose was `(0.500866, 0.004113, 0.935827) m`, the center pose
was `(0.500262, -0.000359, 0.912844) m`, and the measured reducer-center XY
error was `4.512 mm`, below the strict 5 mm limit. The reducer was 22.983 mm
above the center and the recorded relative quaternion angle was `0.00621 rad`.
Thus the command attained 6/6 before and at release, with a valid release
relation.

The trace has no direct contact-force channel, so wall contact is localized by
the first post-release lateral/rotational and score signature. At control step
141 / physics step 705 the XY error was still 4.963 mm and score 6; at step
142 / physics step 710 it crossed to 5.031 mm, the relative angle was
`0.02903 rad`, and the score changed 6 -> 5. This is the first sampled
kinematic wall/contact signature, 0.80 s after release; the reducer had moved
only about 0.98 mm in z and remained roughly 22.885 mm above the center, so it
was lateral interference rather than the 4 mm run's immediate free-fall. The
score briefly recovered to 6 at step 145 / physics step 725 (XY 4.173 mm).
At step 161 / physics step 805, when the controller entered retreat, the score
fell 6 -> 5 with 11.653 mm XY error; the relative angle then grew to `0.51408
rad` by the final sample.

The official result was `initial_score=5`, `best_score=6`, `final_score=5`,
`status=FAIL`, and `task_success=false`. The final reducer pose was
`(0.499054, -0.017407, 0.940564) m`; release-to-final displacement was
`(-1.812, -21.519, +4.737) mm` (22.109 mm norm). Its final relation was
18.609 mm XY, 29.367 mm above the center, and `0.51408 rad` relative angle.
The center remained geometrically valid, collision stayed enabled, and there
were zero post-initialization object-pose writes.

Artifacts are in
`runs/dev_reducer_bias4p75mm_clearance40mm_abs/`: `result.json`,
`phase_scores.json`, `trace.npz`, `episode.mp4`, and `console.log`. The
temporary command was restored immediately afterward: the working tree now has
the best-known 40 mm clearance and 4 mm bias at `controller.py:171-172`, with
the 4 mm comment at lines 187-189.

This final midpoint does not produce stable 6/6. Together with 4 mm and
4.5 mm (valid release but later loss) and 5 mm (invalid release/interference),
and the 25/35/40 mm clearance bracket, the scalar bias/height approach is
exhausted for the tested bounded range under this passive slip trajectory. The
next mechanism-level hypothesis is contact-aware retention: keep the reducer
partially guided or compliantly grasped while actively controlling insertion
depth, then release only after contact/pose settles. That redesign is not
implemented here.

### Guided grasp-to-seat retention experiment (restored after interference)

Hypothesis: retaining the reducer grasp while moving from the validated 40 mm
clearance to the previously observed 15 mm seated target would replace the
passive free-fall with a slow, controlled insertion. The 4 mm y-bias and all
official scoring, scene, physics, geometry, and threshold settings were kept
unchanged. The temporary controller-only diff reused the existing 0.75 s
release window: it added `guided_height = target + [0, 0, 0.015]`, returned the
existing IK arm action while leaving the gripper closed, and moved the existing
`0.04` open command to the following settle window. No schedule duration was
added.

The exact run command was:

`ROCO_GPU_DEVICE=3 ROCO_PHASE=reducer ROCO_SEED=23 ROCO_OUTPUT_DIR=/home/sunsiliang/Human-AI-Collab/roco_m1_success/runs/dev_reducer_guided_seat15mm_clearance40mm_abs ./roco_m1_success/run_successful_assembly.sh`

Pre-contact validity was intact at control step 125 / physics step 625:
score 6/6, reducer-center XY error `3.394 mm`, center-minus-reducer height
`-26.079 mm`, and relative angle `0.00947 rad`. At the guided descent start
(step 126 / 630), score was still 6/6 and the reducer was at
`(0.500165, 0.001587, 0.929498) m`, with 1.737 mm XY error and 14.860 mm
above the center. The measured right gripper joint remained about `0.00688 m`,
consistent with retaining the grasp.

The trace has no direct contact-force channel. Its earliest kinematic
interference is at the guided command itself: from step 125 to 126 the reducer
dropped 9.491 mm while the center rose 1.728 mm. The first score-confirmed
contact/interference is one 50 ms control interval later, step 127 / physics
step 635: score changed 6 -> 4, the reducer was at `0.922798 m`, the center
had risen to `0.916032 m`, and the relative angle jumped to `0.06049 rad`.
The score then oscillated 4 -> 5 -> 2 during the still-held guided descent;
this is pre-release stack disturbance, not post-release free-fall.

At the first open command (settle boundary, step 141 / physics step 705), the
reducer had reached a geometrically seated pose `(0.499921, 0.001893,
0.913698) m` against center `(0.500079, 0.002391, 0.913657) m`: 0.522 mm XY,
0.041 mm center-minus-reducer height, and `0.03661 rad` relative angle. But
the other relations had already been damaged; the official score at that
release snapshot was only 3/6. The reducer then fell to `0.901950 m` while
the disturbed center rose to `0.921037 m` at step 142, score 2/6.

The official result was `initial_score=5`, `best_score=6`, `final_score=4`,
`status=FAIL`, and `task_success=false`; the center remained geometrically
valid, collision stayed enabled, and post-initialization object-pose writes
were zero. The final reducer pose was `(0.495264, -0.001795, 0.928238) m`.
Final reducer-center relation was 2.359 mm XY, 16.994 mm above the center,
and `0.26786 rad` relative angle. The artifacts are in
`runs/dev_reducer_guided_seat15mm_clearance40mm_abs/`: `result.json`,
`phase_scores.json`, `trace.npz`, `episode.mp4`, and `console.log`; all were
written and non-empty, and the trace contains 200 samples.

The held guided descent is rejected as unsafe because it presses/disturbs the
already assembled stack before opening, despite briefly reaching the desired
reducer pose. The temporary change was restored immediately. The working
controller is again the prior best-known passive 40 mm / 4 mm version at
`controller.py:171-172`, with its controlled-slip branch at lines 187-190.
No further variation was run. The evidence supports a contact-aware mechanism
that can sense or limit insertion force while preserving lateral guidance, but
this experiment does not select a new scalar or implement that redesign.

### Contact-awareness audit and stack-motion stop experiment (restored)

Hypothesis: the first harmful stack motion could be detected from an existing
kinematic state signal, and a small positive-z threshold could stop the guided
descent and open the gripper before more of the stack was disturbed. The
temporary controller-only experiment kept the validated 40 mm release
clearance, 4 mm y-bias, official scorer, scene, physics, and schedule
unchanged. It compared the current `sun_planetary_gear_4` center pose against
the previous control sample after the reducer release boundary. At
`controller.py` during the run, `center_delta[:, 2] > 0.0005` m (0.5 mm) set a
latched contact flag and returned the existing 0.04 m open command; no sensor,
threshold, or scorer was added elsewhere.

The exact validation command was:

`ROCO_GPU_DEVICE=3 ROCO_PHASE=reducer ROCO_SEED=23 ROCO_OUTPUT_DIR=/home/sunsiliang/Human-AI-Collab/roco_m1_success/runs/dev_reducer_contact_motion_stop15mm_clearance40mm_abs ./roco_m1_success/run_successful_assembly.sh`

The trace identifies the first harmful disturbance at 6.30 s, control sample
126 / physics step 630. The preceding sample (control 125 / physics 625) had
center `(0.500249, -0.000456, 0.912910)` m and reducer
`(0.499875, 0.002918, 0.938989)` m. At the disturbance sample the center was
`(0.500522, 0.003287, 0.914638)` m, a `(+0.272, +3.742, +1.728)` mm jump,
while the reducer was `(0.500165, 0.001587, 0.929498)` m and dropped
`(+0.290, -1.331, -9.491)` mm. The official score was still 6/6 at that
sample. The controller logged `M1_REDUCER_CONTACT physics_step=630
center_dz=+0.001728m` in `console.log`. The first score-confirmed damage was
6 -> 5 at control 128 / physics 640 (6.40 s): center
`(0.503472, -0.000533, 0.921281)` m and reducer
`(0.502612, -0.001585, 0.919316)` m. The score then reached 2 at physics 645
and continued to oscillate while the stack settled.

The 0.5 mm signal was able to recognize the already-occurring upward stack
impulse, but it was not a pre-contact stop signal. The controller is called
once per 0.05 s environment step and the environment applies each action for
five physics steps (`galaxea_lab_external_env.py:835-854`). Thus the detector
first ran at physics 630, after the physics 625-629 release interval had
already produced the 1.728 mm impulse. Opening from that point did not prevent
the subsequent score loss. At the release event snapshot (control 126 / 630),
the center-to-reducer relation was 1.737 mm XY, -14.860 mm height, and
0.02674 rad; the reducer was still above the stack. At the first score loss
(physics 640), the recorded score was 5/6. The final result was
`initial_score=5`, `best_score=6`, `final_score=0`, `status=FAIL`, and
`task_success=false`; final reducer pose was `(0.624798, 0.056821,
0.908854)` m, 137.879 mm from its detector/release pose, with final
center-to-reducer XY error 138.595 mm, center-minus-reducer height 2.741 mm,
and relative angle 1.00451 rad. Collision remained enabled and there were
zero post-initialization object-pose writes.

The other existing runtime signals do not provide an earlier reliable
contact trigger in this run. The external environment's observation path
exports camera data plus joint position and velocity only
(`galaxea_lab_external_env.py:221-285`); it does not export joint effort,
wrist force, contact force, or a contact-sensor buffer. The active R1 USD does
set `activate_contact_sensors=True` (`robots/galaxea_robots.py:19-38`), but
`_setup_scene` only creates the robot, cameras, and rigid objects
(`galaxea_lab_external_env.py:118-169`), with no `ContactSensor` entity. The
policy imports `ContactSensorCfg` at `robots/galaxea_rule_policy.py:22` but
does not instantiate or read one. The available joint velocity is not a stall
signal: finite differences of the recorded qpos around the first disturbance
were 0.766, 0.498, 0.289, 0.166, and 0.096 rad/s (samples 125-129), i.e. a
commanded-motion decay rather than a contact stall. Gripper positions were
also only measured after the fact; there is no desired-vs-actual grasp error
channel in `trace.npz`.

The failed temporary behavior was restored to the best-known passive
controller: 40 mm clearance and 4 mm y-bias at
`controller.py:169-172`, controlled slip at `controller.py:186-190`, then
the existing open/retreat phases at `controller.py:191-193`. The final source
passes `python -m py_compile roco_m1_success/controller.py`. No additional
validation was started. The minimal future mechanism-level change is to wire
an actual link/contact or joint-wrench signal through the active scene and
controller, then calibrate its onset against this trace; that sensor path was
not added in this bounded task.

Artifacts (all non-empty) are in
`runs/dev_reducer_contact_motion_stop15mm_clearance40mm_abs/`:
`result.json`, `phase_scores.json`, `trace.npz`, `episode.mp4`, and
`console.log` (including the detector marker and full score transitions).
