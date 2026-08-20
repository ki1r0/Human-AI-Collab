# Stage D — real ACT policy-only test

Classification: **CONFIRMATORY**.

## Inputs

- Third-party candidate `yjsm1203/roco_model_act_2` pinned at revision `52344a203e0739638cb2c7b11ea632e7b2eb2608`.
- `policy_best.ckpt` SHA-256 `a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1`.
- `dataset_stats.pkl` SHA-256 `4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e`.
- The exact pre-perturbation live Stage C observation NPZ.

Checkpoint provenance is **APPROXIMATION**: it is not organizer-published and combines official RoCo demonstrations with additional data.

## Locked contract

1. Instantiate the official ACT architecture/configuration and strictly load the pinned state dict.
2. Convert RGB NHWC→NCHW and divide uint8 values by 255 exactly once.
3. Normalize qpos with paired `qpos_mean/qpos_std`.
4. Infer a `(1,100,14)` normalized action chunk.
5. Apply the official competition wrapper's exponential temporal decay `0.1`.
6. Denormalize with paired `action_mean/action_std`.
7. Reorder `[L6,Lgrip,R6,Rgrip]` to `[L6,R6,Lgrip,Rgrip]`.
8. Run two predictions, reset temporal state, and reproduce the first prediction.

## Pass criteria

- Hashes match; strict load has no missing/unexpected keys and loads a nonempty real state dict.
- Stats are finite `(14,)` arrays with positive standard deviations.
- A saved real RoCo observation produces finite `(1,100,14)` output and plausible denormalized actions.
- Temporal aggregation advances and changes the second action.
- Reset empties history and deterministically reproduces the first output.
- The reorder sentinel exactly maps to `[0..5,7..12,6,13]`.
