# Stage B iteration 9 — reducer-only gentle grasp

Classification: **EXPLORATORY**.

## Hypothesis

H14: reducing only the reducer close-command preload from 0 to 7 mm preserves its pickup while allowing the right gripper to withdraw during the 40 mm clearance release.

## Locked procedure

1. Retain selected collision settings, R1 0.20 m workspace, seed 23, zero ring rotation, 40 mm reducer endpoint, and 1.5 s release.
2. Restore the R1 global gripper effort limit to 100 N after the 200 N regression.
3. Change only the reducer pickup close target from 0 to 0.007 m. Every pin/ring grasp remains at 0 m; stiffness, damping, velocity, friction, armature, and all arm targets remain unchanged.
4. Run the 3,420-step schedule on GPU 3 with full boundary RGB/state evidence and strict console scan.

## Prediction and decision rule

- H14 is supported only if the reducer is lifted and transported, then right-gripper position exceeds 0.01 m during release without displacing the seated ring by more than 5 mm.
- Stage B passes only at stable score 6 with physically seated reducer evidence and a clean console.
- If the reducer slips before transport, reject 7 mm as under-grasping; do not interpret later score as a release test.
- If it remains clamped near 7 mm, reject grasp preload as the cause and inspect the reducer/finger material interaction before changing another motion.
