# Research Findings

## Research Question

Can the public RoCo gearbox environment and ACT execution path complete one genuine learned closed-loop Task 1 rollout on this machine?

## Current Understanding

The relevant benchmark is the AAAI 2026 planetary-gearbox environment, not the newer IROS 2026 industrial-board benchmark and not this repository's existing Franka scene. The paper's simulation benchmark and public demonstrations use the dual-arm Galaxea R1. The current official source additionally supports R1 Lite and currently activates it, but that post-dates the public R1 simulation data and the available ACT checkpoint candidate. The supported one-line R1 bundle switch is therefore required for checkpoint compatibility.

The official ACT training data flow is internally clear even though the deployment scripts are inconsistent: three RGB views are stacked in `[head_rgb, left_hand_rgb, right_hand_rgb]` order and divided by 255; qpos is standardized with `qpos_mean/qpos_std`; ACT predicts a 100-action normalized chunk; temporal aggregation combines overlapping chunks; actions are denormalized with `action_mean/action_std`; policy order `[left arm, left gripper, right arm, right gripper]` is remapped to environment order `[left arm, right arm, left gripper, right gripper]` before absolute joint-position control.

The public official repository contains no learned checkpoint or normalization statistics. A compatible public candidate exists at `yjsm1203/roco_model_act_2`, including `policy_best.ckpt` and `dataset_stats.pkl`, but it is third-party and trained on the official data plus additional demonstrations. Its task success must be established empirically and its provenance remains a fidelity deviation.

## Key Results

- Official source pinned: `rocochallenge/gearboxAssembly@094a1f76d18c207caec198315f23b1a60dbca94f`.
- All 76 listed LFS assets in the official checkout are materialized; no unresolved LFS pointers were found.
- A locally cached Isaac Lab 2.3.0 container matches the official runtime versions and exposes GPU inference.
- The policy/environment 14-D reordering is supported by source history and current environment joint-index construction.

## Patterns and Insights

Several apparent runtime failures can be predicted from source mismatches rather than policy quality: unnormalized images, unnormalized qpos, R1-vs-R1-Lite embodiment drift, and incorrect 14-D ordering. The reproduction runner must make these contracts explicit and assert them instead of copying either upstream deployment script wholesale.

## Lessons and Constraints

- Do not use the host Conda base Python; it is Python 3.13 and lacks the robotics stack.
- Do not modify `/home/sunsiliang/IsaacLab`; it contains unrelated dirty user changes and does not match the official pin.
- Do not use the existing Franka/magic-assembly project path as a substitute for the Galaxea R1 environment.
- Do not claim an official checkpoint: the organizer repository publishes only a placeholder path.
- Treat environment success termination independently: current `_get_dones()` compares the `(score, time)` tuple to `6`, so native success termination is defective.

## Open Questions

- Can Kit fully launch on driver 550.163.01 even though the container metadata names 570.169 as minimum?
- What exact dtype/range do live Isaac camera tensors have in this environment?
- Does the third-party ACT checkpoint load with zero missing/unexpected keys and plausible stats?
- Does its initial-state distribution match the current official R1 environment reset?
- Can a compact official demonstration be obtained for the replay action-interface diagnostic?

## Optimization Trajectory

No learned-policy rollout has run yet. The project remains at the provenance and environment-integrity gate.

