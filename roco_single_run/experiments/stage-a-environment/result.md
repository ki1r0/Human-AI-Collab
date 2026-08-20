# Stage A result — environment integrity

Final status: **PASS after one documented compatibility iteration**.

## Runs

| Run | Result | Evidence |
|---|---|---|
| Initial | FAIL gate despite probe-level checks passing | Six invalid PhysX contact-offset errors and unresolved `OakTable_N.png` / `OakTable_R.png` references in `runs/stage_a/console.log`. |
| Regression | PASS | Exit 0; probe status PASS; no `[Error]`, missing-asset, PhysX-error, or traceback lines in `runs/stage_a_regression/console.log`. Result SHA-256: `c6a2694b3192f8d3ebca574af5bf0d3251cfb559e3cc7f9b5fbeadaaa8417d1e`. |

## Measured contract

- Isaac Sim 5.1 / Isaac Lab 2.3.0 launched on one RTX A5000 through host driver 550.163.01.
- Active robot: Galaxea R1, 20 joints total, 14 controlled DoFs.
- Environment order: six left-arm joints, six right-arm joints, left gripper, right gripper.
- Physics: 0.01 s; decimation: 5; control period: 0.05 s (20 Hz).
- Each head/left-wrist/right-wrist RGB tensor is `(1,240,320,3)` `uint8` with live nonconstant content.
- Each matching depth tensor is `(1,240,320,1)` `float32`; background infinity is expected in head and left-wrist depth.
- Three hold-position steps completed to simulation time 0.15 s with no termination or timeout.

## Compatibility changes

`scripts/prepare_official_checkout.py` pins and verifies official commit `094a1f76d18c207caec198315f23b1a60dbca94f`, selects the supported R1 bundle, corrects invalid contact/rest offset pairs, and supplies neutral normal/roughness textures for the two public-repository omissions. The neutral maps are a visual **APPROXIMATION**; they do not replace a missing color/albedo texture.
