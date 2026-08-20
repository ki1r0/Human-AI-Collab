# Stage B iteration 4 — zero ring insertion rotation

Classification: **EXPLORATORY**.

## Hypothesis

H9: holding the ring wrist orientation fixed after descent preserves the existing carrier-pin relations and allows the remaining official phases to reach score 6.

## Rationale

In the selected-contact seed-23 run, score is 3 before ring insertion, rises to 4 at descent step 2,520, and collapses to 0 during the following 30-degree wrist rotation. The minimum-contact variant shows the same causal boundary (2 -> 3 -> 1). Descent already satisfies a ring-related score condition; rotation is the first common interval that destroys the prior assembly.

## Locked procedure

1. Restore the selected legal collision settings: ring/sun/reducer contact/rest `0.0001/-0.0005` m and carrier `0.001/0.0005` m.
2. Change only the ring (`gear_id == 5`) rotation from 30 degrees to 0 degrees. Keep the 3-second phase duration, descent target, gripper timing, release, retreat, and every subsequent official phase unchanged. Gear 4 retains its 60-degree insertion rotation.
3. Run seed 23 in R1 at the 0.20 m workspace with the identical 20 Hz control, camera snapshots, official score, and 700-step bound.

## Prediction and decision rule

- H9 is supported if the score at step 2,820 remains at least 4 and the run exceeds the baseline best score 4.
- Stage B passes only if official score reaches at least 6 without timeout or runtime error.
- If the ring score is lost before release even without rotation, next vary only the descent height.
- If score is preserved through release but later phases fail, diagnose the reducer/final-gear relationship without changing ring motion again.
