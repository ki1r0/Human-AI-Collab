# M1 Blockers

This file contains active blockers only. Existing dirty files are not M1 blockers because they predate this execution; they are recorded in `M1_PRECHECK.md` and must not be overwritten.

| ID | Status | Evidence | Impact | Next action |
|---|---|---|---|---|
| B0 | OPEN | RoCo checkout has 27 dirty/untracked items and `git lfs fsck` pointer errors | Cannot use it as a clean release checkout | Keep it read-only; publish commit/dirty-state provenance and use a separate adapter path |
| B1 | OPEN | No standalone Isaac Sim on host; only local Docker images are available | Host-native `scripts/env.sh` path is unavailable | Run all Isaac/RoCo tests through the pinned Docker image |
| B2 | OPEN | No live human operator has participated in this session | G3/G4 cannot be truthfully marked PASS | Complete all non-human work; mark HIL gates `WAITING_FOR_OPERATOR` and provide runbook |
| B3 | OPEN | No planner/VLM endpoint or credential variables are configured (`HRC_M1_*` absent) | G2 cannot run a real model; a rule planner is contract-only | Keep the HTTP boundaries strict; configure real providers before nominal/HIL evaluation |
| B4 | OPEN | The composed-stage collision-on reset-only probe is `PROVISIONAL_PASS` (`validation_logs/m1_physics_trial_default_20260926T1810Z/`), and the later full-scene one-inside/one-outside pick plus public held predicate pass (`validation_logs/m1_public_held_skill_smoke_20261001T050000/`). Guarded insertion, release stability, and measured dynamics remain unverified | G0 is only provisional and G1 cannot be reported as a physical pick/insert/release success | Validate collision-on robot-held preinsert/guarded-insert/release, source or measure inertia/friction, then freeze the calibration manifest |
| B5 | OPEN / INSERTION | The public R1 `pick` now returns `HELD_CONFIRMED` from filtered pair contact plus Hub/object-follow, but the same trace only reaches `SEAT_CANDIDATE`; release experiments either retain finger contact or drag the Hub off the socket axis (`validation_logs/m1_inner_outer_release_axial_incremental_20261001T030000/`) | A valid grasp does not establish that the held cover can be inserted and released without collision or loss of alignment | Calibrate a physically valid insertion/release motion and add independent seat and post-release stability checks |
| B6 | OPEN / GRASP REGRESSION PASS, INSERTION UNVERIFIED | The initial collision-on probe in `validation_logs/m1_grasp_blocker_20260926.md` tested the proposed central-bore route with `q=(0,1,0,0)` and an external side route, but did not establish a robot-held lift. Those central-bore trials placed **both** fingers inside the bore and are the wrong topology for the requested one-inside/one-outside clamp. The corrected radial topology was first isolated in `validation_logs/m1_inner_outer_isolated_video_20260929T223000/`, then rerun with every scene collision enabled after repairing the analytic table placement. The bounds report `validation_logs/m1_scene_collision_bounds_20260930.json` showed the old table penetrating R1 `base_link`/`torso_link1`; `hrc_m1/roco_env.py` now places its top at `z=-0.05 m`. The corrected full-scene run `validation_logs/m1_inner_outer_fullscene_tablefix_20260930T100000/` reports both finger contacts (`66.72 N`/`62.41 N`), Hub lift `0.08966 m` versus link-6 `0.11350 m`, and `candidate_verdict=HELD_CANDIDATE` with no collision filters. The same evidence is now wired into the public adapter callback: `validation_logs/m1_public_held_skill_smoke_20261001T050000/events.jsonl` records `HELD_CONFIRMED`, forces `64.471/73.916 N`, Hub follow `0.082923 m` versus link-6 `0.113525 m`. | The previous Robot↔Table obstruction is resolved and the public pick/held regression passes. G1 remains blocked because socket insertion, seating, release stability, and a complete pick→preinsert→insert→verify→release trace are still unvalidated. | Keep the public held predicate; validate insertion/release with a physically controlled path. Current release experiments show the one-inside/one-outside clamp cannot yet clear the ring without residual contact or dragging the Hub; do not use snap, reparent, teleport, kinematic toggles, or evaluator truth as a substitute. |
| B7 | OPEN / STRICT PRE-RELEASE ALIGNMENT | The best full-gravity single-arm replay reaches the socket and settles after release, but the strict pre-release sample is `6.595°` from the canonical orientation, `15.82 mm` axial error, and `22.42 mm` maximum bolt-hole error. The release-open-start sample still carries about `111.5 N` on one finger; after the separately logged 300-step opening wait, the later retract sample has `0 N` finger contact. A separate final-placement diagnostic is `4/4` only after release-settle, so it does not satisfy strict M1. Evidence: `validation_logs/m1_single_y0_notilt_release_wait300_20261004/metrics.json`. The high-waypoint 1-degree orientation correction was separately rejected after preinsert finger forces reached about `9.65/3.72 kN` and final placement fell to `2/4`: `validation_logs/m1_single_y0_notilt_correct_above_step1_20261004/metrics.json`. Final-seat position-only was also rejected after a `23.9 m/s` insert divergence and `55.62 m` final radial error: `validation_logs/m1_single_y0_notilt_seat_position_only_20261004/metrics.json`. | Strict 6DoF/bolt alignment and release-contact-free-at-open checks cannot be claimed, although ordinary post-release placement is physically stable on the no-correction route. | Preserve the valid `m1_single_y0_notilt_postsettle500_camera_20261004` place candidate and the wait ablation; do not use high-waypoint wrist rotation or position-only seating. Continue only with a physically justified gripper/fixture or release-timing change that produces auditable pre-release alignment. Do not weaken strict thresholds. |

Additional evidence: the dual-arm near-candidate was tested with a bounded
`3.5°` post-seat wrist correction in
`validation_logs/m1_dual_postseatcorr35_20261004/metrics.json`. It lost held
contact during transport (`15/43` contact samples), left the valid scene, and
scored physical-place `2/4` / relaxed placement `1/4`; post-seat wrist
rotation is therefore rejected as an alignment repair.

Additional evidence: a 300-step closed-grasp hold after the single-arm seat in
`validation_logs/m1_single_y0_notilt_postseat_hold300_20261004/metrics.json`
reduced speed but increased orientation error to `7.01°`; release retained
about `4.94 kN` contact and dragged the Hub, yielding physical-place `2/4` and
relaxed placement `2/4`. Post-seat settling time is rejected as a repair.

Additional evidence: the 2026-10-05 `+5 mm` Y release-clearance test in
`validation_logs/m1_single_y0_notilt_release_clearance005_20261005/metrics.json`
still retained about `75.2 N` finger contact at the clearance sample and then
dragged the Hub during retract (`1.339 m` drift; final radial error `0.745 m`).
It scored physical-place `2/4`, relaxed placement `2/4`, and RoCo-style `4/6`.
Small lateral clearance is therefore rejected as a release fix; strict seat
alignment and contact-free retract remain open.

Additional dual-arm GT loop evidence (2026-10-05) is recorded in
`reports/m1_gt_validation_loop_20261005.md`. Three bounded release-controller
attempts all retained the first four points but failed the strict seat/release
gate: right-in-place (`16.04°` seat orientation, `86.15 mm` bolt error), raw
joint freeze during right opening (`47.99°`, `91.90 mm`), and Cartesian TCP
pose hold during right opening (insertion diverged to approximately `100 m/s`
and final retract drift `3118 m`). These runs do not weaken the thresholds;
they close the current release-repair loop without a 6/6 GT result.
