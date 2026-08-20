# Stage B iteration 6 — extended reducer release

Classification: **EXPLORATORY**.

## Hypothesis

H11: extending reducer release from 0.5 s to 1.5 s allows the right gripper to clear the reducer before arm retreat and preserves the assembled stack.

## Locked procedure

1. Retain selected collision settings, R1 0.20 m workspace, seed 23, zero ring rotation, and reducer mount endpoint 0.030 m.
2. Change only the third duration in `time_step_14` from 0.5 s to 1.5 s. Reducer pickup, high/low targets, release command, retreat target, control rate, score, and all other timings remain fixed.
3. Run on a less-loaded identical RTX A5000 because iteration 5 logged one discarded camera frame on GPU 1.
4. Complete the resulting 3,420-physics-step / 684-environment-step schedule within the existing 700-step bound. Save all score transitions, boundary poses, gripper state, and RGB frames.

## Prediction and decision rule

- H11 is supported if right-gripper position increases materially before the new step-3,370 retreat boundary and score remains at least 5 through retreat.
- Stage B passes only at stable end score 6 with visually and kinematically seated parts and a console free of renderer/physics/asset/Python errors.
- If the gripper remains nearly closed for 1.5 s, diagnose gripper contact/command actuation rather than adding more time.
- If it opens but the reducer is lifted or the stack still shifts, next adjust only release/retreat geometry using the captured poses.
