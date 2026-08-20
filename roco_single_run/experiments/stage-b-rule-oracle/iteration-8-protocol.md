# Stage B iteration 8 — historical gripper effort

Classification: **EXPLORATORY**.

## Hypothesis

H13: restoring the earlier official R1 200 N gripper effort cap allows the right gripper to release the reducer under contact load without dragging the seated stack.

## Locked procedure

1. Retain selected collision settings, R1 0.20 m workspace, seed 23, zero ring rotation, 40 mm reducer endpoint, and 1.5 s release.
2. Change only `GALAXEA_R1_CFG` gripper `effort_limit_sim` from 100 N to 200 N. Preserve stiffness 25,000, damping 1,000, velocity 0.07 m/s, joint friction 0.2, armature 0.2, scene friction, and all motion.
3. Run the 3,420-step schedule on GPU 3 with full boundary RGB/state evidence and strict console scan.

## Prediction and decision rule

- H13 is supported if right-gripper position increases above 0.01 m during release and ring/carrier displacement remains below 5 mm.
- Stage B passes only at stable score 6 with seated reducer evidence and a clean console.
- If effort doubles but the gripper still closes, reject actuator saturation and next test a release action that directly clears the fingers without changing grasp physics.
- If release succeeds but score remains 5, decompose the missing relationship before changing another motion.
