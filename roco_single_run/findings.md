# Research Findings

## Research Question

Can the public RoCo gearbox environment and ACT execution path complete one genuine learned closed-loop Task 1 rollout on this machine?

## Current Understanding

The relevant benchmark is the AAAI 2026 planetary-gearbox environment, not the newer IROS 2026 industrial-board benchmark and not this repository's existing Franka scene. The paper's simulation benchmark and public demonstrations use the dual-arm Galaxea R1. The current official source additionally supports R1 Lite and currently activates it, but that post-dates the public R1 simulation data and the available ACT checkpoint candidate. The supported one-line R1 bundle switch is therefore required for checkpoint compatibility.

The official ACT training data flow is internally clear even though the deployment scripts are inconsistent: three RGB views are stacked in `[head_rgb, left_hand_rgb, right_hand_rgb]` order and divided by 255; qpos is standardized with `qpos_mean/qpos_std`; ACT predicts a 100-action normalized chunk; temporal aggregation combines overlapping chunks; actions are denormalized with `action_mean/action_std`; policy order `[left arm, left gripper, right arm, right gripper]` is remapped to environment order `[left arm, right arm, left gripper, right gripper]` before absolute joint-position control.

The public official repository contains no learned checkpoint or normalization
statistics. Three compatible same-author public ACT releases were pinned and
tested as explicit third-party approximations. All load and execute faithfully,
but none completes the task.

## Key Results

- Official source pinned: `rocochallenge/gearboxAssembly@094a1f76d18c207caec198315f23b1a60dbca94f`.
- All 76 listed LFS assets in the official checkout are materialized; no unresolved LFS pointers were found.
- A locally cached Isaac Lab 2.3.0 container matches the official runtime versions and exposes GPU inference.
- The policy/environment 14-D reordering is supported by source history and current environment joint-index construction.
- Stage A passes after a pinned compatibility preparation: the live R1 task runs at 100 Hz physics / 20 Hz control and exposes three 240×320 uint8 RGB plus three float32 depth streams.
- The public source has two runtime defects independent of policy quality: invalid PhysX contact offsets and absent oak-table normal/roughness textures.
- Stage C passes with three visually inspected, actively refreshed RGB feeds and canonical finite 14-D qpos.
- Stage D strictly loads the 83,923,087-parameter candidate and paired stats, then passes normalization, chunk, temporal aggregation, denormalization, reorder, and reset checks on a real live observation.
- The full official DataReplay episode executed all 590 actions but failed its locked residual-direction threshold and scored 0 on a visibly different reset layout. A fresh independently preregistered equivalent-interface probe then passed all 14 channels in both directions with 1.0 direction agreement and about 0.00188 target error.
- Four genuine closed-loop `roco_model_act_2` episodes completed at scores `[1,0,0,0]`. The seed-23 left arm genuinely mounted one gear, but right-arm action variance stayed tiny and no right-arm task phase appeared on any seed; this candidate is refuted as a full-task solution under the reproduced contract.
- Four separately pinned base-checkpoint episodes completed at `[0,0,0,0]` and
  repeated the same unilateral behavior despite distinct weights and stats.
- A provenance-selected terminal `_1` epoch passed the full policy gate but its
  seed-23 live episode scored 0 and again kept the right arm near reset.
- Nine learned episodes total have complete fresh-camera/action/state traces,
  independently decodable 591-frame H.264 video, and clean shutdown evidence.
- The advertised host wrapper passed a fresh prepare/launch/live ACT
  inference/action/evidence/shutdown regression and records Git/runtime/GPU
  provenance plus conventional manifest/log/metrics/video/trace aliases.

## Patterns and Insights

Several apparent runtime failures can be predicted from source mismatches rather than policy quality: unnormalized images, unnormalized qpos, R1-vs-R1-Lite embodiment drift, and incorrect 14-D ordering. The reproduction runner must make these contracts explicit and assert them instead of copying either upstream deployment script wholesale.

## Lessons and Constraints

- Do not use the host Conda base Python; it is Python 3.13 and lacks the robotics stack.
- Do not modify `/home/sunsiliang/IsaacLab`; it contains unrelated dirty user changes and does not match the official pin.
- Do not use the existing Franka/magic-assembly project path as a substitute for the Galaxea R1 environment.
- Do not claim an official checkpoint: the organizer repository publishes only a placeholder path.
- Treat environment success termination independently: current `_get_dones()` compares the `(score, time)` tuple to `6`, so native success termination is defective.
- Always run the commit-pinned preparation step; its neutral table normal/roughness maps are an explicit visual approximation.

## Open Questions

- Can an organizer/private or newly trained full-task ACT checkpoint reach
  explicit score 6 under this now-validated execution contract?
- Is the repeated unilateral behavior primarily training-phase coverage or
  sensitivity to the small documented environment compatibility deviations?

## Optimization Trajectory

Stages A, C, D, and E pass; Stage B is a diagnosed rule-oracle failure with best
transient score 5. Stage F executed nine genuine learned trials across all
available same-author ACT candidates. Best learned score is 1/6, so the project
is PARTIAL and blocked on a capable matching checkpoint rather than pipeline
execution.
