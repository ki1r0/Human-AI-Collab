# Stage B iteration 5 — shallower reducer endpoint

Classification: **EXPLORATORY**.

## Hypothesis

H10: raising the reducer mount endpoint by 5 mm prevents the final insertion/release from destabilizing the four-point assembly while still allowing the reducer to settle on the middle gear.

## Locked procedure

1. Retain the selected legal collision settings, R1 0.20 m workspace, seed 23, and accepted exploratory zero-ring-rotation repair.
2. Change only the reducer `mount_height_offset` from 0.025 m to 0.030 m. Preserve its XY target, orientation, pickup, phase durations, gripper release, retreat, solver, and score function.
3. Record all score increases and decreases, all object poses, and RGB/state snapshots at every ring and reducer phase boundary.
4. Complete the full 3,320-step schedule even if score transiently reaches 6.

## Prediction and decision rule

- H10 is supported if the score remains at least 4 through reducer release and the reducer's final pose is visibly/kinematically seated rather than floating.
- Stage B passes only if score is at least 6 on the last pre-reset schedule step and the saved final phase evidence shows a physically assembled gearbox.
- A transient score 6 before descent/release is explicitly insufficient because the upstream evaluator accepts arbitrarily high parts due to one-sided signed-height tests.
- If the assembly still destabilizes, compare the captured 3,170/3,220/3,270 reducer poses and next adjust only the endpoint in the evidence-supported direction.
