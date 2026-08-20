# Stage A protocol — environment integrity

Classification: **CONFIRMATORY**.

## Hypothesis

H1: the locally cached Isaac Lab 2.3.0 image can launch the pinned official RoCo R1 Task 1 environment with all required assets and cameras.

## Locked procedure

1. Verify the official commit and every LFS-managed file before launch.
2. Run the official task-list command inside the container with only the external extension mounted/installed.
3. Launch `Template-Galaxea-Lab-Agent-Direct-v0` headless with cameras on GPU 2.
4. Reset once, inspect robot embodiment and joint names, step a safe hold-position action for several steps, and shut down.
5. Record exit code, Kit errors, asset-resolution errors, GPU/driver errors, reset behavior, and physics progression.

## Prediction

The task registers, the R1 assets resolve, reset returns observations, all three cameras render, and safe position targets advance physics without infrastructure failure. If Kit rejects driver 550.163.01, the failure should occur before environment creation and distinguish runtime compatibility from RoCo code defects.

## Pass criteria

- No unresolved LFS pointer or missing USD dependency.
- Isaac Sim and official task register successfully.
- R1 joint names and 14 controlled DoFs resolve.
- Reset and repeated physics steps complete.
- Process shuts down cleanly.

