# Stage B iteration 2 — randomized-start sensitivity

Classification: **EXPLORATORY**.

## Hypothesis

H7: with the corrected R1 workspace, rule-oracle completion is sensitive to the randomized ring start. At least one of three preregistered valid seeds will improve on score 4, and ring pick success will correlate with a reachable initial placement/arm assignment.

## Locked procedure

Run seeds **17, 23, and 42** independently using the identical R1 0.20 m environment, official rule trajectory, legal contact offsets, score, cameras, 20 Hz control, and 700-step bound. Use one isolated A5000 per seed and save full logs, results, plus RGB and state snapshots at every ring pick/mount phase boundary.

## Decision rule

- H7 is supported if any seed exceeds score 4 and phase snapshots show corresponding ring displacement/lift.
- Stage B passes only if a seed reaches score at least 6 without timeout/error.
- If every seed is at most 4 or every ring pick fails, reject seed sensitivity as sufficient and inspect the ring grasp target. Do not select a seed merely because it is best below the official threshold.
