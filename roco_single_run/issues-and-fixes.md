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
- **Regression status:** OPEN — download, load, and rollout pending.

## ISSUE-005 — Upstream preprocessing mismatch

- **Symptom:** `VLA_agent.py` casts images to float but does not divide by 255; `ACTPolicyWrapper` does not standardize qpos.
- **Hypothesis:** Copying the runner literally feeds inputs on the wrong scale relative to training.
- **Evidence:** `act/utils.py` divides RGB by 255 and standardizes qpos; `imitate_episodes.py` does the same during evaluation.
- **Experiment:** Static end-to-end data-flow comparison across runner, wrapper, loader, and evaluator.
- **Result:** Training/eval behavior is consistent; deployment runner is incomplete.
- **Fix:** Planned adapter applies training-faithful RGB and qpos preprocessing with runtime assertions and logs both raw/normalized ranges.
- **Regression status:** OPEN — unit and live-observation tests pending.

## ISSUE-006 — R1/R1 Lite embodiment drift

- **Symptom:** Current `robot_bundles.py` activates R1 Lite, while README says R1 is default and the public simulation data/checkpoint candidate use R1.
- **Hypothesis:** The April/May 2026 R1 Lite extension changed the repository default after the December 2025 ACT checkpoint was trained.
- **Evidence:** Git history shows commit `d2ae3a7` switched the active bundle to R1 Lite; the checkpoint was published 2025-12-26.
- **Experiment:** Compared current README, bundle history, checkpoint dates, and dataset embodiment metadata.
- **Result:** R1 is the checkpoint-compatible official embodiment for this rollout.
- **Fix:** Use the repository-supported one-line R1 bundle selection in the isolated checkout and record the patch.
- **Regression status:** PARTIAL — Stage A verified the live R1 joint names, indices, camera paths, and reset state; checkpoint range comparison remains pending.

## ISSUE-007 — Native success termination bug

- **Symptom:** Agent environment `_get_dones()` evaluates `self.evaluate_score() == 6`, but `evaluate_score()` returns `(score, time_cost)`.
- **Hypothesis:** A successful learned episode will not terminate on score and may reset only via rule-policy count or timeout.
- **Evidence:** Direct source trace at current pinned commit.
- **Experiment:** Static type/value comparison.
- **Result:** Tuple-to-int equality is always false.
- **Fix:** Runner will separately read `(score, time_cost)` and declare success only from `score >= 6`; environment timeout remains a termination condition. Upstream code will not be silently rewritten.
- **Regression status:** OPEN — runtime score test pending.

## ISSUE-008 — Git LFS attribute warning

- **Symptom:** Checkout reported 22 STL files that “should have been pointers, but weren't.”
- **Hypothesis:** Later `.gitattributes` changes cover historical ordinary blobs; this may be noisy without leaving required USD assets unresolved.
- **Evidence:** `git lfs ls-files --size` lists 76 materialized assets; `git lfs status` is clean; no text LFS pointers remain in the worktree.
- **Experiment:** Enumerated LFS files and searched for pointer headers.
- **Result:** All listed runtime USD assets are materialized; warning concerns tracking history/attributes.
- **Fix:** No content rewrite. Preserve official checkout and validate assets again before launch.
- **Regression status:** PASS for pointer resolution; Stage A asset loading pending.

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
- **Regression status:** PARTIAL — 0.20 m retained; randomized ring-placement sweep pending.

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
- **Result:** Pending.
- **Fix:** Pending Stage B iteration 5.
- **Regression status:** OPEN.
