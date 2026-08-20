# Stage B iteration 7 — reducer contact clearance

Classification: **EXPLORATORY**.

## Hypothesis

H12: raising the reducer endpoint from 30 mm to 40 mm prevents premature ring compression, unjams the gripper, and preserves the stack through release.

## Locked procedure

1. Retain selected collisions, R1 0.20 m workspace, seed 23, zero ring rotation, and the 1.5 s reducer release observation window.
2. Change only reducer `mount_height_offset` from 0.030 m to 0.040 m. Preserve XY/orientation target, pickup, descent/release/retreat commands, score, cameras, and 3,420-step schedule.
3. Run on GPU 3 and save all score transitions, reducer/ring/middle poses, gripper state, and RGB frames.

## Prediction and decision rule

- H12 is supported if ring z remains within 3 mm of its pre-descent height at step 3,220 and right-gripper position increases during release.
- Stage B passes only at stable end score 6 with a clean console and visual/pose evidence of a seated reducer and intact ring/pin assembly.
- If the ring remains stable but the gripper stays closed, diagnose gripper/reducer friction or command authority.
- If the reducer remains visibly floating after release, reduce clearance in one bounded interpolation rather than accepting an evaluator-only score.
