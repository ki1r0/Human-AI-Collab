# M1 Progress

## Current status

- Goal: execute `M1_gearbox_HRC_Codex_plan.md` end-to-end.
- Current phase: Phase 1/2 implementation and collision-on regression; the corrected one-inside/one-outside pick/lift and public held predicate now pass after a table-collider repair, while socket insertion/release remain for G1.
- Last evidence: `validation_logs/m1_physics_trial_default_20260926T1810Z/` reproduced the real Isaac Lab collision-on reset-only interface probe using the task config's default SDF-margin=0 trial parameters and produced an MP4, public event log, and private evaluator record. The earlier `m1_physics_trial_sdf0_20260926T1630Z/` remains the detailed calibration record.
- Current gate status: G0 is **PROVISIONAL** for the composed collision-on interface probe but still blocked for final robot-held insertion/release calibration. G1 public pick/held verification passes in `validation_logs/m1_public_held_skill_smoke_20261001T050000/`; socket insertion, release and post-seat stability remain blocked. G2 is blocked, and G3/G4 wait for a real operator; G5 is partial because GitHub credentials are unavailable.

## Completed

- Read the complete execution plan and Ponytail instructions.
- Recorded main/RoCo commits, dirty state, LFS status, container/GPU versions and current asset hashes.
- Confirmed RoCo task registry launches in `nvcr.io/nvidia/isaac-lab:2.3.0` with the read-only checkout.
- Confirmed the RoCo environment's current sensors/action path is planetary-gear-specific and cannot be reused as a complete M1 task.
- Wrote `M1_PRECHECK.md` and `M1_TASK_CONTRACT.md`.
- Added the isolated `hrc_m1` contracts, state machine, strict planner/VLM boundaries, evaluator, logger, task config, RoCo R1 scene and adapter.
- Host unit tests: `python3 -m unittest discover -s tests -p 'test_hrc_m1*.py'` — 12 tests passed, covering schema/evaluator-boundary rejection, fail/unknown/abort state transitions, HIL handoff memory invalidation, bounded jogs, VLM UNKNOWN, planner timeout safety, logging redaction, image boundaries, settle boundaries, and public grasp verification.
- RoCo container smoke with `--enable_cameras`: scene reset, RGB/qpos/qvel/gripper observation and zero-action step completed. The run is explicitly `SIM_SMOKE_ONLY`.
- Task-scoped SDF collision approximation is used for the dynamic Hub Cover and the authored triangle mesh is retained for the fixed Casing socket. The RoCo desk is not used as the M1 fixture. The analytic M1 table avoids the unrelated desk collision fallback.
- The contact sensor initially overflowed its default four-point GPU buffer. It is now configured with `track_contact_points=True` and `max_contact_data_count_per_prim=512`; `validation_logs/m1_contact_capacity_smoke_20260925T103545Z/` confirms the scene starts without the assert.
- A slow, 1930-physics-step (`386` environment steps at `dt=0.01`, decimation 5) collision-on reset-only probe completed at `validation_logs/m1_physics_trial_sdf0_20260926T1630Z/` after explicit SDF margin calibration. It measured contact, `0.0986 mm` penetration, `0.432 mm` radial error, `1.870°` tilt, `0.00326 m/s` settle speed and a `0.55 s` stable window. The independent evaluator returned `PROVISIONAL_PASS`; `task_success` remains false because calibration is pending and this is not a robot-held episode. The same result was reproduced without CLI overrides at `validation_logs/m1_physics_trial_default_20260926T1810Z/`; the earlier `m1_physics_trial_magic_boundary_20260925T1055Z/` remains preserved as historical evidence.
- The stable post-revert five-skill trace `validation_logs/m1_skill_smoke_reverted_20260925T1200Z/` completed `pick → preinsert → insert → verify → release_retract` with `PASS_TRACE` and a valid MP4. Its structured results still report `held_status`/seat verification as `UNKNOWN`, so it is a controller trace rather than G1 success.
- A targeted TCP/tool-orientation experiment was preserved as `validation_logs/m1_skill_smoke_tcpfix_20260925T1140Z/`; it diverged catastrophically (private evaluator position/velocity magnitudes reached 170.8 m/43.7 m/s), so that uncalibrated transform was removed and the stable version was rerun. This failed trace is not used as evidence of capability.
- A follow-up world-to-base-frame correction plus the pinned RoCo TCP transform was tested at `validation_logs/m1_skill_smoke_framefix_20260925T1230Z/`; it no longer diverged, but the cover still remained at reset (`radial_error_m=0.4007`, no contact). The transform is therefore retained as an explicit **uncalibrated G1 candidate**, not a success claim; further R1 gripper/TCP calibration is required.
- A fault-no-help safety baseline using the explicitly dev-only rule planner completed at `validation_logs/m1_fault_no_help_rule_20260925T1110Z/`: after the first fixed `pick` trace, the absent VLM returned `UNKNOWN` and the state machine ended `SAFE_ABORT` without continuing insertion. This is a safety/contract trace, not an autonomous-model result.
- An equivalent fault-HIL run with no operator attached is recorded at `validation_logs/m1_fault_hil_waiting_20260925T1120Z/`; after the same observable `UNKNOWN`, it entered `ASK_HUMAN` and ended `WAITING_FOR_OPERATOR`. No synthetic human commands were injected.
- The exact MagicAssembly child-local override is retained only as reset-time calibration input. The evaluated trial advances the dynamic rigid body through PhysX after one initial velocity; no per-step pose writes, snap, reparent, or MagicAssembly combine path is used.
- Two implementation causes of the earlier apparent large penetration were fixed: reset no longer starts the cover overlapping the casing/robot, and zero action now holds the robot's current joint targets rather than sweeping both arms to literal zero. The USD XYZ `(-90, 180, 0)` orientation is represented with the verified `wxyz=(0,0,+sqrt(1/2),+sqrt(1/2))` quaternion.
- Added optional stdin HIL broker (`hrc_m1.human_help`) with bounded 10 mm jogs, explicit TAKE_CONTROL/RETURN_CONTROL, and no direct USD pose writes.
- Ran a dedicated collision-on R1 grasp probe (central-bore and external side-grasp orientations) with a temporary Hub-to-gripper contact filter. The probe is recorded in `validation_logs/m1_grasp_blocker_20260926.md`: the Hub remained at reset for small openings and was ejected when the gripper was driven into the collision boundary; no stable lift/held state was observed. This remains an unresolved grasp-path blocker, not a model failure and not evidence to relax geometry.
- The Hub SDF collider was then calibrated with explicit zero margin and zero narrow-band thickness because the cover's thin axis is only about 28 mm. A repeat central-bore closure still ejected the Hub at the collision boundary, so the blocker is not explained by an implicit SDF inflation. The setting is retained in `hrc_m1/roco_env.py` and the calibration manifest for the next interface regression.
- The post-calibration collision-on seat regression `validation_logs/m1_physics_trial_sdf0_20260926T1630Z/` remains `PROVISIONAL_PASS`: axial `0.067636 m`, radial `0.000432 m`, tilt `1.870°`, penetration `0.0000986 m`, contact true, stable `0.55 s`, settle speed `0.003262 m/s`. This confirms the SDF change did not break the isolated interface; it still does not establish a robot-held episode.
- Historical hold-gate regression: `validation_logs/m1_skill_smoke_hold_gate_20260926T1700Z/` returned `FAILED / HELD_UNVERIFIED` after 300 steps with reason `public_grasp_sensor_unavailable`; no preinsert/insert skill was attempted. This closed the false-success path before the later public sensor wiring; the current public `pick` result is recorded below.
- Added two filtered left-gripper-to-Hub ContactSensors. The calibrated pre-open regression `validation_logs/m1_inner_wall_probe_sensors_20260929T140000/` reports `0 N` on both fingers and `0 m` Hub displacement, so that earlier waypoint did not make contact; the corrected one-inside/one-outside route is now promoted through the public adapter's `HELD_CONFIRMED` predicate in the later full-scene smoke.
- The opening-boundary follow-up `validation_logs/m1_inner_wall_probe_midopen_20260929T145000/` and `validation_logs/m1_inner_wall_probe_maxopen_20260929T144000/` crossed into collision ejection at actual openings of about `0.0442 m` and `0.0461 m` (Hub displacement `0.335 m` and `0.673 m`). Neither is a held grasp; the transition is retained as calibration evidence.
- A fresh `fault_no_help` regression after the hold gate, `validation_logs/m1_fault_no_help_prompt_hash_20260926T1930Z/`, reached `SAFE_ABORT` after the failed/unknown pick and did not attempt insertion. The event trace contains `HELD_UNVERIFIED` followed by verifier `UNKNOWN` and `completed: SAFE_ABORT`; the rule planner remains dev-only safety plumbing, not G2 evidence.
- Git provenance in container-generated manifests was repaired by using an explicit per-checkout Git `safe.directory`. Host mock and Isaac smoke regression `validation_logs/m1_manifest_git_smoke_20260926T1830Z/` now record main HEAD `0e280c6f0f2aaf91589ff27fb7b130b11a2d7e31`, RoCo HEAD `094a1f76d18c207caec198315f23b1a60dbca94f`, and both dirty states truthfully instead of `unavailable`.
- Isaac environment seeding is now set before scene construction. `validation_logs/m1_seed_smoke_20260926T1900Z/` shows `Environment seed : 1201`, completes reset/camera/one-step smoke, and retains the same truthful Git provenance.
- The state machine now requires `REOBSERVE → UPDATE_MEMORY → REPLAN` after human control is returned (and clears the pre-handoff decision); the new host regression verifies stale trajectory invalidation before replanning.
- Planner and VLM event records now carry request/prompt hashes alongside response hashes; manifests record the decision schema, model configuration status, and a public fault-profile hash while evaluator labels remain in the private stream.
- Current gate ledger is in `M1_ACCEPTANCE.md`; it intentionally reports G0/G1 blocked pending insertion/release, G2 blocked without a real model endpoint, and G3/G4 waiting for a real operator.
- The requested one-inside/one-outside Hub Cover topology is now tested explicitly. The isolated PhysX diagnostic `validation_logs/m1_inner_outer_isolated_video_20260929T223000/` records both finger contacts (about `70.5 N`/`65.7 N`) and Hub-follow during lift (`0.0896 m` Hub versus `0.1134 m` link-6), with an MP4. The first full-scene attempt was invalidated by the analytic table intersecting the robot; after moving that task-scoped table below the robot, `validation_logs/m1_inner_outer_fullscene_tablefix_20260930T100000/` reports both contacts and object-follow with all scene collisions enabled. The public adapter regression subsequently reports `HELD_CONFIRMED`; B6 is now specifically an insertion/release blocker.
- Added a finite, resumable diagnostic sweep in `tools/m1_grasp_continuation.sh`, with explicit candidate state under `validation_logs/m1_grasp_continuation/`. The user-level `m1-grasp-continuation.timer` is enabled and configured to retry on a five-hour interval without catch-up execution at session startup. This host-side runner can resume experiments after a tool/model-quota interruption; it cannot resume the language-model conversation itself.
- Measured the fixture split and found the old analytic table (`z=0.25..0.35 m`) intersected R1 `base_link` and `torso_link1` at reset (`validation_logs/m1_scene_collision_bounds_20260930.json`). Moved only the task-scoped table to center `z=-0.10 m` (top `z=-0.05 m`) so it remains a floor collision without penetrating the robot. The corrected full-scene one-inside/one-outside run `validation_logs/m1_inner_outer_fullscene_tablefix_20260930T100000/` reports both finger contacts (`66.72 N`/`62.41 N`) and Hub-follow (`0.08966 m` Hub versus `0.11350 m` link-6), with no collision filters enabled. This is a pick/lift regression pass; insertion, seating, release, and public held verification remain open.
- The public adapter now starts its grasp reference immediately before the calibrated lift and returns `HELD_CONFIRMED` only when both filtered finger force sensors and Hub/object-follow pass. `validation_logs/m1_public_held_skill_smoke_20261001T050000/events.jsonl` records `HELD_CONFIRMED`, forces `64.471/73.916 N`, Hub follow `0.082923 m` versus link-6 `0.113525 m`; the full trace remains `SKILL_SMOKE_ONLY` and `task_success=false` because insertion/release are not accepted.
- Insertion/release diagnostics were run with collision-on physics and no object pose writes: lateral release left residual contact or dragged the Hub (`validation_logs/m1_inner_outer_release_clearance_wait_20261001T000000/`); one-shot and incremental axial withdrawal cleared contacts only after moving the Hub off-axis (`validation_logs/m1_inner_outer_release_axial_clearance_20261001T020000/`, `validation_logs/m1_inner_outer_release_axial_incremental_20261001T030000/`). The insertion verdict remains `INSERTION_NOT_VERIFIED`.
- A follow-up diagnostic added an opt-in measured-grasp-frame insertion target (`--preserve-lift-grasp-frame`) and a `--stop-after-preinsert` boundary. The camera-enabled long trace was terminated after approximately 30 minutes without a metrics file (`validation_logs/m1_inner_outer_preserve_grasp_frame_20261001T060000_ABORTED.md`); the no-video full trace was likewise terminated after 20 minutes (`validation_logs/m1_inner_outer_preserve_grasp_frame_20261001T070000_ABORTED.md`). These are execution-time diagnostics only and provide no physical pass/fail evidence.
- Rechecked the calibration manifest against the current USD bytes; both canonical asset hashes now match exactly. The manifest records the repaired task-scoped table (`top_z=-0.05 m`) and the calibrated Hub reset (`y=0.45 m`).
- Ran the manipulation-critical USD audit in the pinned Isaac container (`validation_logs/check_part_fidelity_20260930.txt`): all 15 part assets passed rigid/mass/collider checks, and both gear-to-shaft fits passed with `0.500` asset-unit radial clearance. This is a static geometry/collider audit, not a robot-held insertion proof.
- The canonical physics manifest covers all `49/49` combine steps with no missing, unknown, or duplicate references (`validation_logs/check_physics_manifest_20260930.txt`).

## Next commands

1. Implement and validate a controlled pick→preinsert→guarded-insert→seat→release/retract episode with independent held/seat/release checks and inspect the generated MP4/state trace; the remaining blocker is the calibrated release/clearance path, not the public pick predicate.
3. Configure real LLM/VLM providers if credentials are supplied; otherwise retain a blocked, auditable result.
4. Run nominal and fault-no-help after a real planner/VLM is configured; run fault-HIL only with a real operator and mark it `WAITING_FOR_OPERATOR` until then.
5. Package the local HRC deliverable; GitHub creation/push remains credential-gated.

## Rules for every subsequent failure

Preserve the first failing log and manifest, identify the first divergent observation, change one evidence-backed cause, rerun the smallest failing test, then rerun the affected gate. Never lower physical criteria, disable collision, teleport/reparent, or call Magic Assembly in an evaluated episode.

## 2026-10-04 strict-placement continuation

The best reproducible full-gravity single-left-arm route is
`validation_logs/m1_single_y0_notilt_postsettle500_score_20261004/metrics.json`.
It uses an explicit reset `(0.30, 0.00, 1.08) m`, `0.20 m` transport clearance,
the authored arm controller, no Hub pose writes, and no grasp constraint. The
Hub is lifted `0.1375 m` and all `43/43` transport samples retain finite state
and two-finger contact. The live-camera replay is
`validation_logs/m1_single_y0_notilt_postsettle500_camera_20261004/`, with
3061 frames and `video_error=null`.

The strict evaluator remains `CONTACTED_SEAT_NOT_STABLE`, physical-place
`2/4`, RoCo-style `4/6`; strict pre-release orientation is `6.595°` and max
bolt-hole error `22.42 mm`. A separate `M1_POST_RELEASE_PLACEMENT_4_POINT_V1`
diagnostic is `4/4` after 500 release-settle steps (final radial `0.77 mm`,
axial `5.18 mm`, orientation `2.36°`, max bolt error `7.17 mm`, drift
`10.85 mm`). This is a normal-place candidate, not strict M1 or ACT evidence.

Evidence-driven negative replays are retained in
`reports/m1_policy_rollout.md`: 20 mm vertical release clearance ejected the
Hub; an 80 mm command exceeded the effective gripper range and did not clear
contact; a 1°/segment wrist correction worsened the seat. The next unresolved
strict issue is the R1 grasp/controller's ability to achieve a contact-free,
bolt-aligned pre-release state without disturbing the valid post-release
placement route.

The follow-up `validation_logs/m1_single_y0_notilt_release_wait300_20261004/`
kept the arm fixed during a 300-step opening wait and then settled for 500
steps. It confirms the ordinary place candidate at `4/4` (final radial
`0.687 mm`, axial `4.728 mm`, orientation `2.132°`, bolt-hole max `6.836 mm`,
support force `55.9 N`, and zero finger contact at retract). It does not alter
the strict verdict: the pre-release sample remains `6.595°`/`22.42 mm` from
the strict pose/bolt thresholds and the sample taken at open-command start
still contains about `111.5 N` finger contact. The strict M1 gate therefore
remains open; the relaxed final-placement score is not promoted to strict
success.

The isolated high-waypoint orientation correction
`validation_logs/m1_single_y0_notilt_correct_above_step1_20261004/` is a
negative controller ablation. Eight 1-degree wrist increments before the
socket corridor caused the first divergence at preinsert (about `9.65 kN` and
`3.72 kN` finger contact, Hub speed about `1.02 m/s`), followed by final radial
error `136.46 mm` and orientation error `178.40°`; relaxed placement fell to
`2/4`. It is rejected; the no-correction route remains the best candidate.

The integrated M1 adapter was rechecked in the pinned Isaac Lab 2.3.0
container with `hrc_m1.run --backend roco --smoke --smoke-steps 3` at seed
`1201`. The run exited successfully as `SIM_SMOKE_ONLY` and produced six
non-empty RGB frames (head/left-hand/right-hand at reset and step 3), an
`episode.mp4`, manifest, event log, and metrics under
`runs/m1_integrated_smoke_recheck_20261004/`. This validates reset/camera/step
integration only; it is not an M1 task-success claim.

The same adapter then ran a collision-on low-level skill trace with
`hrc_m1.run --backend roco --skill-smoke --planner rule`, seed `1201`, under
`runs/m1_adapter_skill_smoke_recheck_20261004/`. The five skills all returned
motion `SUCCEEDED`; `pick` specifically returned
`HELD_CONFIRMED` from filtered two-finger contact and Hub/object-follow
(`64.471/73.916 N`, Hub follow `0.082923 m` versus link-6 `0.113525 m`). The
trace has reset/preinsert/insert/verify/release frames and a non-empty
three-camera MP4. Its `insert` result is only `SEAT_CANDIDATE`, the independent
evaluator is `UNKNOWN`, and metrics intentionally remain
`SKILL_SMOKE_ONLY`/`task_success=false`. This promotes the adapter pick/skill
path to a reproducible diagnostic, not to a complete M1 physical success.

The single-arm final-seat position-only ablation
`validation_logs/m1_single_y0_notilt_seat_position_only_20261004/` is a
negative result. It diverged at insert (Hub speed `23.9 m/s`, one finger
contact about `99.1 kN`, no Hub–Casing contact) and ended with radial error
`55.62 m`, orientation error `112.12°`, and relaxed placement `2/4`. Removing
orientation tracking is therefore not a safe correction; the previous
full-pose no-correction route remains the best physically stable candidate.

The new `--post-seat-hold-steps 300` ablation
(`validation_logs/m1_single_y0_notilt_postseat_hold300_20261004/`) held the
measured arm joints after socket contact with the gripper still closed. It
reduced insert speed to `0.00162 m/s` but increased orientation error to
`7.01°`; release retained about `4.94 kN` finger contact and dragged the Hub
away. Physical-place and relaxed final placement were both `2/4`. The option
is rejected and is now recorded explicitly in the metrics schema.

The dual-arm post-seat correction ablation
`validation_logs/m1_dual_postseatcorr35_20261004/` changed only the bounded
post-seat wrist correction (`3.5°`) on the closest historical dual-arm route.
It was rejected: only `15/43` transport samples retained contact, the Hub left
the valid scene, physical-place was `2/4`, and relaxed final-placement was
`1/4`. Rotating the loaded wrist after socket contact is not a safe strict-
alignment repair. The uncorrected dual route remains a historical
near-candidate, not M1 success.

The 2026-10-05 single-arm release-clearance diagnostic
`validation_logs/m1_single_y0_notilt_release_clearance005_20261005/` moved the
opened arm frame only `+5 mm` in Y before ordinary retract. Pick/lift and
transport still passed (`43/43` contact-retaining samples), but strict seat
alignment remained `15.82 mm` axial, `6.595°` orientation, and `22.42 mm`
maximum bolt-hole error. Residual release contact was still about `75.2 N`;
retract drift reached `1.339 m`, final radial error `0.745 m`, and relaxed
placement was `2/4`. It is a negative release experiment, not a success
trajectory; the strict result remains `CONTACTED_SEAT_NOT_STABLE` and RoCo
style `4/6`.

M0 is independently complete and rechecked with
`python3 -m hrc_repair.preflight --config configs/repair_m0.yaml --strict`:
`PASS`. The persistent blocked→help→continue artifact remains
`runs/repair_m0_persistent_blocked_final_camera_20261003/` with
`task_success_gt=true`, `pipeline_success=true`, and
`post_help_continuation_success=true`.
