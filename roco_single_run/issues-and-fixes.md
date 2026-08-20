# Issues and fixes

## ISSUE-001 — Orchestra installer unavailable

- **Symptom:** `npx @orchestra-research/ai-research-skills install --all` exited 127: `npx: command not found`.
- **Hypothesis:** Node.js/npm is not installed on the host.
- **Evidence:** `command -v node`, `npm`, `npx`, `bun`, and `deno` returned no paths.
- **Experiment:** Inspected the official Orchestra repository and its current install layout at commit `773a52944ba4747a18bd4ae9ade53fff041adcbc`.
- **Result:** The official package copies the same skill directories into Codex's skills path (normally via a canonical directory and symlinks).
- **Fix:** Used Codex's supported GitHub skill installer to install all 98 exact skill directories directly under `~/.codex/skills` from the pinned commit.
- **Regression status:** PASS — 98 new `SKILL.md` files installed; `autoresearch` read successfully. A fresh Codex turn may be needed for automatic skill discovery.

## ISSUE-002 — Host Python environment is incompatible

- **Symptom:** Host `python` is Conda base Python 3.13.11 and cannot import torch, Isaac Sim, or Isaac Lab.
- **Hypothesis:** Running RoCo from the active shell would mix an unsupported Python with missing packages and possibly sticky Conda paths.
- **Evidence:** `which python` -> `/home/sunsiliang/miniconda3/bin/python`; all three imports raised `ModuleNotFoundError`; `CONDA_DEFAULT_ENV=base`.
- **Experiment:** Inspected existing local runtimes and Docker images.
- **Result:** `nvcr.io/nvidia/isaac-lab:2.3.0` is locally cached and matches the official RoCo environment pins.
- **Fix:** Use the isolated container; do not alter host Conda or the dirty host IsaacLab checkout.
- **Regression status:** PASS — Stage A launched Kit, the R1 scene, RTX cameras, and repeated physics steps in the container.

## ISSUE-003 — Container driver metadata mismatch

- **Symptom:** The Isaac Lab 2.3.0 image declares `MIN_DRIVER_VERSION=570.169`; host driver is 550.163.01.
- **Hypothesis:** Isaac Sim/RTX startup may fail despite CUDA compute working through forward compatibility.
- **Evidence:** Docker image config and `nvidia-smi` disagree on the version floor.
- **Experiment:** Ran `/isaac-sim/python.sh` with GPU 2 and imported torch.
- **Result:** torch 2.7.0+cu128 reports CUDA available and identifies the RTX A5000.
- **Fix:** No driver mutation was needed. Preserve the isolated container and keep GPU/renderer errors as hard launch gates.
- **Regression status:** PASS for the selected R1 workload — two Stage A scene launches and RTX camera steps completed; the version mismatch remains a documented portability risk.

## ISSUE-004 — No official learned checkpoint is published

- **Symptom:** Official README requires `--checkpoint='Your-VLA-Checkpoint-File-Path'`; no `.ckpt` or `dataset_stats.pkl` exists in the repo.
- **Hypothesis:** The organizer baseline weights were never added to the public GitHub artifacts.
- **Evidence:** Full repository search and public documentation search found only placeholders; paper says baselines were provided but gives no direct current artifact.
- **Experiment:** Searched official sources first, then public RoCo-compatible models.
- **Result:** Found `yjsm1203/roco_model_act_2` with architecture-matching checkpoint and stats, trained on official plus additional data.
- **Fix:** Use this pinned third-party candidate provisionally and mark checkpoint provenance as APPROXIMATION. Continue searching for an organizer artifact.
- **Regression status:** BLOCKER remains for organizer weights. Three third-party
  ACT releases were pinned, strictly loaded, and rolled out; none exceeded 1/6.

## ISSUE-005 — Upstream preprocessing mismatch

- **Symptom:** `VLA_agent.py` casts images to float but does not divide by 255; `ACTPolicyWrapper` does not standardize qpos.
- **Hypothesis:** Copying the runner literally feeds inputs on the wrong scale relative to training.
- **Evidence:** `act/utils.py` divides RGB by 255 and standardizes qpos; `imitate_episodes.py` does the same during evaluation.
- **Experiment:** Static end-to-end data-flow comparison across runner, wrapper, loader, and evaluator.
- **Result:** Training/eval behavior is consistent; deployment runner is incomplete.
- **Fix:** The adapter applies training-faithful RGB and qpos preprocessing with
  runtime assertions and logs both raw/normalized ranges.
- **Regression status:** PASS in all policy-only and learned live runs.

## ISSUE-006 — R1/R1 Lite embodiment drift

- **Symptom:** Current `robot_bundles.py` activates R1 Lite, while README says R1 is default and the public simulation data/checkpoint candidate use R1.
- **Hypothesis:** The April/May 2026 R1 Lite extension changed the repository default after the December 2025 ACT checkpoint was trained.
- **Evidence:** Git history shows commit `d2ae3a7` switched the active bundle to R1 Lite; the checkpoint was published 2025-12-26.
- **Experiment:** Compared current README, bundle history, checkpoint dates, and dataset embodiment metadata.
- **Result:** R1 is the checkpoint-compatible official embodiment for this rollout.
- **Fix:** Use the repository-supported one-line R1 bundle selection in the isolated checkout and record the patch.
- **Regression status:** PASS for the selected embodiment contract; all three
  checkpoint families pass physical-range checks and live R1 execution.

## ISSUE-007 — Native success termination bug

- **Symptom:** Agent environment `_get_dones()` evaluates `self.evaluate_score() == 6`, but `evaluate_score()` returns `(score, time_cost)`.
- **Hypothesis:** A successful learned episode will not terminate on score and may reset only via rule-policy count or timeout.
- **Evidence:** Direct source trace at current pinned commit.
- **Experiment:** Static type/value comparison.
- **Result:** Tuple-to-int equality is always false.
- **Fix:** Runner will separately read `(score, time_cost)` and declare success only from `score >= 6`; environment timeout remains a termination condition. Upstream code will not be silently rewritten.
- **Regression status:** PASS for explicit score handling. Nine traces recorded
  per-step score, including a genuine 0→1 transition, without false success.

## ISSUE-008 — Git LFS attribute warning

- **Symptom:** Checkout reported 22 STL files that “should have been pointers, but weren't.”
- **Hypothesis:** Later `.gitattributes` changes cover historical ordinary blobs; this may be noisy without leaving required USD assets unresolved.
- **Evidence:** `git lfs ls-files --size` lists 76 materialized assets; `git lfs status` is clean; no text LFS pointers remain in the worktree.
- **Experiment:** Enumerated LFS files and searched for pointer headers.
- **Result:** All listed runtime USD assets are materialized; warning concerns tracking history/attributes.
- **Fix:** No content rewrite. Preserve official checkout and validate assets again before launch.
- **Regression status:** PASS for pointer resolution and repeated Stage A/F asset
  loading.

## ISSUE-009 — Invalid collision offsets and absent table textures

- **Symptom:** The initial Stage A scene emitted six `PxShape::setContactOffset` errors and repeated missing references for `OakTable_N.png` and `OakTable_R.png`.
- **Hypothesis:** Current upstream gear assets set `contact_offset=0`, which violates PhysX 5.1's positive-and-greater-than-rest rule; the committed oak-table USD references two files absent from every official tree revision.
- **Evidence:** The failures are in `runs/stage_a/console.log`; source inspection found three negative-rest assets with zero contact offset and one positive-rest asset with an even smaller contact offset. Git tree/history contains neither texture.
- **Experiment:** Added an idempotent preparation step pinned to the official commit. It preserves the authors' negative 0.5 mm gear rest offset, uses a 0.1 mm positive gear contact offset, uses 1.0/0.5 mm contact/rest offsets for the carrier, and supplies 4×4 neutral normal/roughness maps.
- **Result:** The identical Stage A regression completed with no `[Error]`, missing-asset, PhysX-error, or traceback lines. Robot, camera, timing, and hold-step observations remained consistent.
- **Fix:** Run `scripts/prepare_official_checkout.py` before any experiment. Treat the neutral non-color textures as a visual APPROXIMATION and the collision change as a required PhysX compatibility deviation.
- **Regression status:** PASS — result SHA-256 `c6a2694b3192f8d3ebca574af5bf0d3251cfb559e3cc7f9b5fbeadaaa8417d1e`.

## ISSUE-010 — R1 paired with a later R1-Lite workspace offset

- **Symptom:** The first official rule-oracle run completed its schedule but reached only best score 3/6; gear 2, gear 4, and the carrier/ring relationship never scored.
- **Hypothesis:** Selecting R1 alone is insufficient because current table/external cfg offsets were changed from 0.20 m to 0.15 m after R1 Lite became the default.
- **Evidence:** December 2025 R1 source uses 0.20 m everywhere; May 2026 commit `5631142` changes the table/external environment to 0.15 m; current learned-agent cfg still uses 0.20 m.
- **Experiment:** Restore only the table and external environment to the R1/checkpoint-era 0.20 m workspace and repeat the locked oracle.
- **Result:** Supported but insufficient — best score improved from 3 to 4 and all three pin gears mounted; ring/carrier still failed.
- **Fix:** `prepare_official_checkout.py` now asserts/restores 0.20 m for the table, external environment, and agent environment whenever the R1 integration is prepared.
- **Regression status:** PARTIAL — 0.20 m retained; the completed randomized
  sweep did not produce stable score 6.

## ISSUE-011 — Randomization does not explain ring insertion failure

- **Symptom:** The corrected-workspace oracle remained below score 6 and appeared sensitive to the randomized ring start.
- **Hypothesis:** At least one of preregistered seeds 17, 23, and 42 would exceed score 4 because a more reachable start would enable the ring grasp.
- **Evidence:** Boundary state and RGB snapshots were recorded for every ring pick/mount phase in all three runs.
- **Experiment:** Repeated the identical 0.20 m R1 oracle with seeds 17/23/42, changing no policy, physics, score, timing, or task parameters.
- **Result:** Refuted — best scores were 2/4/2. All rings were securely grasped, lifted from about 0.901 m to 1.099 m, transported over the carrier, and descended. Seed 23 lost three scored carrier-pin relations specifically during the 30-degree ring rotation.
- **Fix:** Do not seed-select and do not alter the demonstrated grasp target. Retain seed 23 as the most diagnostic start and isolate the collision compatibility margin.
- **Regression status:** FAIL for Stage B. Result hashes: seed 17 `b505d5f151a6695a3e0492a297ed08a2213c5a306b34806984b114f42b53d9f3`; seed 23 `3e638010dbca67539016cd73363cdfe4710181b731b3f5976eb29027fee89496`; seed 42 `073c0cfb500a0aaef3599b706bc262680e3aeef24b483df55f81b6239f6fac16`.

## ISSUE-012 — First collision compatibility patch enlarges insertion envelopes

- **Symptom:** Upstream `contact_offset=0.0` is rejected by PhysX 5.1, but the first legal patch used 0.1 mm for gears and 1.0 mm for the carrier.
- **Hypothesis:** The added carrier envelope initiates ring/stack contact before the intended seated pose and contributes to the destructive insertion rotation.
- **Evidence:** Upstream commit `8f57e9b` deliberately changed gear rest offsets to -0.5 mm so surfaces could overlap and slide onto pins. PhysX requires a positive contact offset greater than rest offset; only a positive epsilon is needed for the negative-rest gears, and the carrier needs only an epsilon above its 0.5 mm rest offset.
- **Experiment:** Preregistered Stage B iteration 3 reduces the compatibility margin to 1 micrometre: gear/reducer contact offset 0.000001 m and carrier contact offset 0.000501 m, with rest offsets unchanged.
- **Result:** Refuted — the minimum-margin Stage A regression passed cleanly, but seed 23 regressed from best score 4 to 3 and mounted only two pin gears. Ring descent still added one point and rotation still removed a pin relation.
- **Fix:** Restore the better-performing legal offsets: 0.1 mm contact with -0.5 mm rest for ring/sun/reducer, and 1.0/0.5 mm contact/rest for the carrier.
- **Regression status:** FAIL for H8. Stage A result SHA-256 `0783b83f584aebdfdad60f1e209cfc541d104293f63a6d4947ebdeeb54cb4263`; Stage B result SHA-256 `c2d52896be0255e79cdcf0d72daa7cebe76759924ab3d99f17e3728539f66bdb`.

## ISSUE-013 — Ring rotation destroys already-scored assembly relations

- **Symptom:** With both tested legal contact margins, score increases at ring descent and then decreases during the commanded 30-degree wrist rotation.
- **Hypothesis:** Rotation is unnecessary once the ring/carrier relationship scores and mechanically pushes the partially seated pin gears out of tolerance.
- **Evidence:** Baseline seed 23 scores 3 before the ring and 4 at step 2,520, then 0 at step 2,820. The minimum-margin run scores 2, then 3, then 1 at the same boundaries. Ring pose changes by roughly 25–30 degrees during that interval.
- **Experiment:** Restore the selected collision offsets and change only the ring case from 30 degrees to 0 degrees while preserving the full 3-second hold, descent, release, and subsequent official phases.
- **Result:** Supported but insufficient — seed 23 preserved all three pin relations, held score 4 after ring release, and improved the run best from 4 to 5. The later reducer descent reduced the score to 2.
- **Fix:** Retain zero ring rotation in the exploratory repaired-oracle branch; diagnose reducer insertion separately. This is a trajectory deviation and must not be mislabeled as the untouched official expert.
- **Regression status:** PARTIAL. Result SHA-256 `a3b775b0fdfd72046aac6ddc8f2181b192c60742b869e12037217bf41b606575`.

## ISSUE-014 — Reducer descent destabilizes the four-point assembly

- **Symptom:** With ring rotation removed, the assembly remains at score 4 through ring release, reaches score 5 when the reducer aligns above the center gear, then falls to 2 during reducer descent/release.
- **Hypothesis:** The official 25 mm reducer mount offset presses the reducer too deeply into the stack under the current collision/runtime regime; a 30 mm endpoint will avoid the destabilizing impulse and still let the part settle.
- **Evidence:** At score 5 (step 3,180), reducer XY is within 1.7 mm of gear 4 but its center is still about 74 mm higher. The fifth point is therefore a one-sided evaluator artifact, not proof of seating. Score loss occurs later in the 3,220–3,270 release interval.
- **Experiment:** Retain zero ring rotation and change only reducer `mount_height_offset` from 0.025 m to 0.030 m. Save every reducer phase boundary and every score increase/decrease.
- **Result:** Supported but insufficient — at 30 mm the run held score 5 through step 3,250 instead of collapsing to 2, but release still disturbed the assembly; last pre-reset reward was 4 and the 3,315 snapshot score was 3.
- **Fix:** Retain 30 mm for the next exploratory branch and isolate the visibly incomplete gripper release.
- **Regression status:** PARTIAL. The run had one `omni.syntheticdata` discarded-frame error and is diagnostic only. Result SHA-256 `75c261fe87b5274ebcfad875ee98b20318dcd54b87f2bd72b3bf6cdd47dfcf2a`.

## ISSUE-015 — Reducer release ends while gripper is still closed

- **Symptom:** The arm retreats at step 3,270 while the right gripper remains clamped around the reducer, after which the reducer and stack are displaced.
- **Hypothesis:** Extending only the reducer open-command phase from 0.5 s to 1.5 s lets the gripper clear the part before retreat.
- **Evidence:** Right gripper position is 0.006844 m at release start (step 3,220) and only 0.007173 m at the retreat boundary (step 3,270), despite a 0.04 m target. The 3,315 RGB frame visibly shows the retreating closed gripper and disturbed gearbox.
- **Experiment:** Keep zero ring rotation and the 0.030 m reducer endpoint; change `time_step_14` release duration only from 0.5 s to 1.5 s. Total schedule becomes 3,420 physics steps / 684 environment steps, within the locked 700-step bound.
- **Result:** Refuted — after 1.5 s the gripper was slightly more closed (0.006810 m) and the score had fallen from 5 to 3. The failure is not insufficient time.
- **Fix:** Do not add more release time. Diagnose the reducer/ring contact that exists before the open command.
- **Regression status:** FAIL for H11. Clean result SHA-256 `f289481b4550a929df318022f1d223bb03dc2b5933cafe952667469d38030ed9`.

## ISSUE-016 — Reducer endpoint compresses ring before release

- **Symptom:** At the 30 mm reducer endpoint, the ring drops about 10.3 mm during descent and the right gripper cannot open under contact load.
- **Hypothesis:** A 40 mm reducer endpoint restores contact clearance, lets the reducer drop gently when released, and permits the gripper to open.
- **Evidence:** Ring z is 0.921566 m at descent start (step 3,170) and 0.911341 m at release start (3,220); reducer z is 0.924517 m. Middle gear and ring are nearly coplanar, showing the reducer has driven the ring down rather than seated above it.
- **Experiment:** Retain the 1.5 s release window and change only reducer mount height 0.030 -> 0.040 m. All targets/timings otherwise remain fixed.
- **Result:** Partial — at 40 mm, ring z remained 0.9216–0.9217 m through descent, eliminating compression. However, the gripper still closed and dragged the intact stack laterally; stable score remained 4.
- **Fix:** Retain 40 mm as the clearance branch and isolate gripper actuation authority.
- **Regression status:** PARTIAL. Clean result SHA-256 `a4e2ef3ec2a03259c3082ba547226f94fa796beddda0fd54d7eaf663ce28eff3`.

## ISSUE-017 — R1 gripper cannot open under reducer contact load

- **Symptom:** Even with reducer/ring clearance, a 0.04 m open target moves the right gripper from 0.006805 m to 0.005918 m over 1.5 s while the grasped stack follows laterally.
- **Hypothesis:** Restoring the earlier official R1 200 N gripper effort limit gives the release enough authority; the current 100 N cap saturates under contact/friction.
- **Evidence:** Current `GALAXEA_R1_CFG` uses `effort_limit_sim=100.0`; the pre-December R1 configuration used an explicit 200 N gripper effort limit. Velocity limit is not the observed bottleneck because motion is in the wrong direction.
- **Experiment:** Keep zero ring rotation, 40 mm reducer clearance, and 1.5 s release; change only R1 gripper effort limit 100 -> 200 N. Stiffness, damping, velocity, friction, armature, and all motion stay fixed.
- **Result:** Refuted — the clean 200 N run completed all 3,420 physics steps but regressed from best/terminal scores 5/4 to 3/1 and disturbed the already-working early assembly phases.
- **Fix:** Restore the 100 N configuration. Test reducer-only grasp preload rather than changing global gripper dynamics.
- **Regression status:** FAIL for H13. Result SHA-256 `be30d19d9d51e2484952f34d69516d05d88e4929754a16f839e0025dd0f1e53a`.

## ISSUE-018 — Reducer zero-target grasp may create unnecessary preload

- **Symptom:** At 100 N the reducer is held near 0.0068 m by a 0 m close target and the fingers move inward, not outward, during the 0.04 m release command.
- **Hypothesis:** A reducer-only 0.007 m close target supplies enough geometric retention for transport without storing the squeeze/contact load that prevents release.
- **Evidence:** The reducer's observed clamped position is about 6.8 mm. The shared controller source currently commands 0 m for every object's close phase, but `gear_id == 6` can be isolated without touching pins or ring.
- **Experiment:** Keep the best clearance trajectory and 100 N actuator configuration; change only the reducer close target from 0 to 0.007 m.
- **Result:** Refuted — right-gripper position reached 0.007058 m at the close/lift boundary, but the reducer remained within 0.3 mm of its source and returned fully to the table by the next phase. Best/terminal score was 4/4.
- **Fix:** Restore the official 0 m reducer close target. Do not use a weaker grasp or a non-reference teleport/release mechanism in the learned-policy baseline.
- **Regression status:** FAIL for H14. Clean result SHA-256 `725b90eb7e13bc2e2c7c607018133771d7880396206a23baa8ae9907d67f2550`.

## ISSUE-019 — Live gripper observations omit a feature dimension

- **Symptom:** The first Stage C probe failed while concatenating qpos: arm observations are `(1, 6)` but each gripper observation is `(1,)`.
- **Hypothesis:** The policy contract requires treating each scalar gripper as one feature, producing `(1, 1)` before concatenation.
- **Evidence:** The live tensor error was `Tensors must have same number of dimensions: got 2 and 1`; the official runner likewise builds a 14-D vector from six arm values plus each scalar gripper.
- **Experiment:** Unsqueeze only the last dimension of both live gripper observations, retain the canonical `[L arm, L grip, R arm, R grip]` ordering, and repeat the probe.
- **Result:** Supported — after adding the singleton dimensions, live qpos is finite float32 `(1, 14)` and its explicit names match the canonical order.
- **Fix:** Explicit `.unsqueeze(-1)` on both gripper observations in the Stage C probe.
- **Regression status:** PASS in Stage C.

## ISSUE-020 — Static hold cannot distinguish a correct frame from a stale frame

- **Symptom:** Reset and first held-step RGB arrays were byte-identical in all three cameras even though the structural Stage C checks passed.
- **Hypothesis:** The scene and attached cameras are static under an absolute-position hold, so equality is expected but cannot positively prove frame refresh.
- **Evidence:** The three saved views are valid and distinct on visual inspection, while their reset-to-step mean absolute differences are exactly zero.
- **Experiment:** After saving the canonical observation, command only a bounded +0.05/-0.05 rad change in the first left/right controlled arm joints for five steps and compare all new camera tensors against clones of the saved tensors.
- **Result:** Supported — the active check produced mean absolute pixel changes of 5.18 (head), 35.14 (left wrist), and 22.01 (right wrist), while the saved canonical package remained pre-perturbation.
- **Fix:** Add an active, diagnostic-only freshness perturbation; keep it outside all policy input and rollout paths.
- **Regression status:** PASS in Stage C.

## ISSUE-021 — Top-level Galaxea import requires a running Kit app

- **Symptom:** The first policy-only launch failed before checkpoint construction with `ModuleNotFoundError: omni.physics`.
- **Hypothesis:** Importing `Galaxea_Lab_External.VLA...` executes the package root, which eagerly imports task/environment modules that require Isaac Sim, even though ACT itself is standalone PyTorch.
- **Evidence:** The traceback enters `Galaxea_Lab_External/__init__.py -> tasks -> isaaclab.assets` before reaching ACT policy code.
- **Experiment:** Put the official checkout's `VLA/ACT` directory on `PYTHONPATH` and import the exact official `act.policy.ACTPolicy` module directly.
- **Result:** Supported — the direct `act.policy` import reached stats loading without importing any Isaac environment module.
- **Fix:** Use the direct official ACT subpackage in policy-only and learned-runner processes; environment imports remain after `AppLauncher` in simulator processes.
- **Regression status:** PASS through Stage D and all learned runs.

## ISSUE-022 — Stats pickle uses a NumPy 2 private module name

- **Symptom:** The second Stage D attempt failed at `pickle.load()` with `No module named 'numpy._core'`.
- **Hypothesis:** The stats were serialized by NumPy 2, while the Isaac Lab image ships NumPy 1.x and exposes the equivalent implementation as `numpy.core`.
- **Evidence:** `pickletools` finds only `numpy._core.multiarray._reconstruct`, `numpy.ndarray`, and `numpy.dtype` globals; the pinned file contains the expected five arrays and no arbitrary application classes.
- **Experiment:** After SHA-256 verification and before deserialization, alias `numpy._core` and `numpy._core.multiarray` to their NumPy 1.x equivalents, then retain all strict shape/finite/std checks.
- **Result:** Supported — the aliases load the exact pinned arrays; all four required arrays are finite `(14,)`, both standard-deviation vectors are positive, and Stage D inference passes.
- **Fix:** Add the narrow module aliases only in the pinned stats loader.
- **Regression status:** PASS in Stage D.

## ISSUE-023 — Reference DataReplay wrapper assigns a read-only property

- **Symptom:** Stage E attempt 1 exits from `DataReplayPolicyWrapper.__init__`
  with `AttributeError: property 'device' ... has no setter` before creating the
  environment.
- **Hypothesis:** The wrapper subclass was updated to accept an explicit device,
  but the base class retained a getter-only `device` property.
- **Evidence:** `PolicyWrapper.device` defines only a getter at line 83, while
  `DataReplayPolicyWrapper.__init__` assigns `self.device = device` at line 367
  in the pinned upstream checkout.
- **Experiment:** Subclass the reference wrapper locally and add only a
  `torch.device`-backed setter/getter, inheriting its HDF5 loader, action tensor,
  step counter, reset, and `predict()` unchanged.
- **Result:** Attempt 1 is retained as ERROR; the compatibility subclass then
  loaded the exact 590 actions and completed the full live replay.
- **Fix:** Use `CompatibleDataReplayPolicyWrapper` only for this upstream API
  mismatch. Do not patch the official checkout or trajectory.
- **Regression status:** PASS for wrapper construction/loading; the independent
  replay gate still failed its separate direction criterion.

## ISSUE-024 — Per-step residual direction metric rejects close tracking

- **Symptom:** Full replay mean absolute command error is only 0.0100 and all
  trajectories visually/numerically track, yet the locked direction agreement
  is 0.626 against a 0.80 threshold.
- **Hypothesis:** Comparing the current residual request to only the next 50 ms
  displacement counts inertial settling around small targets as wrong-direction
  motion even when the absolute-position controller is functioning correctly.
- **Evidence:** On the same failed trace, agreement rises from 0.626 for residuals
  over 0.001 to 0.806 over 0.002 and 0.906 over 0.01. Actual action-transition
  direction agreement is 0.976, and all per-joint command/live correlations are
  0.940–0.998. These are diagnostic post-hoc values, not pass evidence.
- **Experiment:** Preserve replay FAIL. On fresh seed 42, command bounded positive,
  return, negative, and return targets for ten steps each, preregistering phase
  settling error, displacement cosine, and direction thresholds.
- **Result:** The fresh test passed every preregistered phase: direction agreement
  1.0, displacement cosine 0.9924–0.9940, target MAE 0.00186–0.00188, and final
  return error 0.001878 across all 40 action steps.
- **Fix:** Retain replay FAIL unchanged and use the independent equivalent result
  as the Stage E acceptance route. The learned runner uses the same shared mapping.
- **Regression status:** PASS in Stage E iteration 2.

## ISSUE-025 — First learned candidate run activates almost only the left arm

- **Symptom:** Seed 23 reaches score 1 at step 81 but makes no further scored
  progress through step 590; video shows the right arm staying at reset.
- **Hypothesis:** This may be reset-layout sensitivity, or `roco_model_act_2` may
  have learned a dominant first left-arm subtask from its padded integrated data
  rather than a full bimanual sequence.
- **Evidence:** The left arm genuinely mounts gear 2. Across the whole run,
  right-arm action standard deviations are 0.012–0.037 and right-gripper std is
  0.000642, while left-arm std is 0.099–0.448 and left-gripper std is 0.0102.
  All cameras are fresh and the shared action interface already passed, so this
  is not attributable to stale input or reorder failure.
- **Experiment:** Run the unchanged preregistered seeds 17, 42, and 2026. If the
  same unilateral behavior repeats, reject this checkpoint for full-task success
  and test the separately pinned base ACT candidate whose model card describes
  later assembly phases.
- **Result:** Seeds 23, 17, 42, and 2026 FAIL at best/final 1/1, 0/0, 0/0,
  and 0/0. Seeds 17/42/2026
  reproduced the near-static right side even more strongly (joint std
  0.0010–0.0048) while failing the left-arm first placement.
- **Fix:** None. Do not synthesize right-arm actions, splice policies, or alter
  temporal history.
- **Regression status:** REFUTED for `_2` as a full-task policy. Test the base
  candidate independently; do not mix or splice checkpoints.

## ISSUE-026 — Base ACT candidate repeats unilateral behavior across layouts

- **Symptom:** The base `roco_model_act` candidate scores zero on each fixed
  seed despite its card describing later assembly phases.
- **Hypothesis:** Its different training data might activate later/right-arm
  phases after a seed-sensitive left-arm start.
- **Evidence:** The checkpoint/stats pass the complete Stage D gate and all four
  live runs have 590 unique inputs per camera, finite actions, full video, and
  clean shutdown. Right-arm joint standard deviations remain approximately
  0.001-0.006 while the left arm is visibly active.
- **Experiment:** Pinned revision/hashes independently, then ran the unchanged
  590-step contract at seeds 23, 17, 42, and 2026.
- **Result:** Refuted — best scores are `[0,0,0,0]`; no score transition or
  right-arm task phase appears.
- **Fix:** None. Do not splice the base checkpoint with `_2` or manufacture a
  post-pin start state.
- **Regression status:** REFUTED for the base checkpoint as a full-task policy.

## ISSUE-027 — No public ACT checkpoint completes Task 1

- **Symptom:** The final public `_1` terminal epoch also completes 590 valid
  learned steps at score zero with a near-static right arm.
- **Hypothesis:** A separately trained terminal epoch, selected before tensor or
  rollout inspection, could be behaviorally broader than both best checkpoints.
- **Evidence:** Immutable epoch 2800 and its stats match preregistered hashes;
  all Stage D gates pass; seed-23 left/right joint action standard deviations
  are 0.039-0.174 versus 0.0013-0.0044. Video, camera freshness, finite arrays,
  action-interface validation, and shutdown all pass.
- **Experiment:** Run the unchanged live seed-23 contract with the terminal
  epoch after strict policy gating.
- **Result:** Refuted — best/final score 0/0. Across `_2`, base, and `_1`, nine
  genuine trials score `[1,0,0,0,0,0,0,0,0]`.
- **Fix:** No faithful local code fix exists. The organizer source publishes no
  ACT checkpoint or matching stats; a capable checkpoint or new training is
  required. SmolVLA/RL substitution, action mirroring, oracle splicing, or
  object teleports are outside the requested ACT fidelity contract.
- **Regression status:** BLOCKER for learned task success; pipeline integrity
  remains PASS and overall reproduction status is PARTIAL.

## ISSUE-028 — Wrapper evidence hard links are rejected

- **Symptom:** The new one-command smoke completed a live ACT step and shutdown,
  then `ln` returned `Operation not permitted`; container-side Git lookup also
  returned unavailable.
- **Hypothesis:** The workspace mount supports files/symlinks but not hard links,
  and the runtime container does not ship the Git executable.
- **Evidence:** Canonical result/log/trace/video exist and are valid; failure
  occurs only at the first post-run hard-link command. The result contains
  `integration_git_commit=unavailable`.
- **Experiment:** Preserve attempt 1, replace aliases with relative symlinks,
  resolve the commit on the host, pass it as a runner argument, and rerun in a
  fresh directory.
- **Result:** Pending fresh wrapper regression.
- **Fix:** Relative symlink aliases and explicit host provenance.
- **Regression status:** PENDING.
