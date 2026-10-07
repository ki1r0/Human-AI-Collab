# M1 acceptance snapshot

This is the current evidence ledger for `M1_gearbox_HRC_Codex_plan.md`.

| Gate | Status | Evidence / reason |
|---|---|---|
| G0 physical interface | PROVISIONAL / BLOCKED_FOR_G1 | `validation_logs/m1_physics_trial_default_20260926T1810Z/` reproduces the real collision-on reset-only probe from the task config defaults after explicit Hub SDF margin=0 calibration. The detailed companion record `m1_physics_trial_sdf0_20260926T1630Z/` measured contact, `0.0000986 m` penetration, `0.000432 m` radial error, `1.870°` tilt, `0.00326 m/s` settle speed and `0.55 s` stable window; evaluator status is `PROVISIONAL_PASS`. Final G0 remains blocked until robot-held insertion/release and dynamics sources are calibrated. |
| G1 fixed skills | BLOCKED (public pick/held passes; insertion/release unvalidated) | The earlier central-bore trials put both fingers inside the bore and are not evidence against the requested topology. After repairing the Robot↔Table AABB overlap (`validation_logs/m1_scene_collision_bounds_20260930.json`, table top moved to `z=-0.05 m`), the corrected one-inside/one-outside radial pose passed a full-collision pick/lift regression in `validation_logs/m1_inner_outer_fullscene_tablefix_20260930T100000/`: both finger sensors reported `66.72 N`/`62.41 N`, and the Hub followed `0.08966 m` of a `0.11350 m` link-6 lift. The calibrated contact/object-follow check is now exposed through the public adapter and independently exercised in `validation_logs/m1_public_held_skill_smoke_20261001T050000/`, whose `pick` result is `HELD_CONFIRMED` with `64.471/73.916 N` and `0.082923/0.113525 m` follow. Insertion trials remain failures: residual finger contact or controller-path drift prevents a verified seat/release (see `validation_logs/m1_inner_outer_release_axial_incremental_20261001T030000/`). |
| G2 autonomous LLM/VLM | BLOCKED | Strict HTTP planner/verifier boundaries exist; planner/VLM request and response hashes are recorded without exposing credentials or evaluator labels. No `HRC_M1_*` provider endpoint or credential is configured. Rule planner is explicitly dev-only and cannot count. |
| G3 human intervention | WAITING_FOR_OPERATOR | `validation_logs/m1_fault_hil_waiting_20260925T1120Z/` demonstrates the runner entering `ASK_HUMAN` and stopping without an operator. The bounded stdin HIL broker and runbook exist; no real operator participated. |
| G4 recovery | WAITING_FOR_OPERATOR | Requires G3 plus a fresh observation and autonomous post-handoff insertion. No synthetic human commands are counted. |
| G5 reproduction/package | PARTIAL | Unit tests, configs, manifests, container commands and MP4 evidence are present. `validation_logs/m1_manifest_git_smoke_20260926T1830Z/` verifies checkout provenance, and `validation_logs/m1_seed_smoke_20260926T1900Z/` verifies seed injection before Isaac scene construction. GitHub CLI/auth is unavailable; no remote `HRC` repository was created or pushed. |

The latest hold-gate safety baseline `validation_logs/m1_fault_no_help_prompt_hash_20260926T1930Z/`
also ended `SAFE_ABORT`: the collision-on pick returned `HELD_UNVERIFIED`, the
absent verifier returned `UNKNOWN`, and no insertion was attempted. It used
`RulePlanner` only as a dev-only contract exerciser, so it is not G2 evidence.
The earlier `validation_logs/m1_fault_no_help_rule_20260925T1110Z/` trace is
preserved as historical evidence.

## Verified commands

```text
python3 -m unittest discover -s tests -p 'test_hrc_m1*.py'  # 12 passed
python3 -m compileall -q hrc_m1                              # exit 0
python3 -m json.tool hrc_m1/calibration_manifest.json         # exit 0
git diff --check                                               # exit 0
```

The simulator smoke and collision-on probe used `nvcr.io/nvidia/isaac-lab:2.3.0`, Isaac Sim 5.1,
Isaac Lab 0.47.1, one RTX A5000, `--enable_cameras`, and the read-only RoCo
checkout.  It completed reset, RGB/qpos/qvel/gripper observation and zero-step
execution.  The task-scoped dynamic Hub collider is SDF and the fixed Casing
retains its authored triangle mesh; the M1 support table is analytic. Contact
points use a 512-entry private buffer to avoid the earlier GPU overflow.

No status in this file should be read as a claim that the action model
understands assembly physics or that M1 has passed.
