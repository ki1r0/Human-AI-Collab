# Stage F final ACT candidate — `yjsm1203/roco_model_act_1`

Classification: **EXPLORATORY CANDIDATE SELECTION, THEN CONFIRMATORY GATES**.
This does not alter the failed `_2` or base-checkpoint results.

## Pinned provenance and non-outcome selection

- Repository: `yjsm1203/roco_model_act_1` (third-party).
- Immutable revision: `fb02c3ef63ea946bf9cc4d0447e2ef0598b10e78`.
- Selected checkpoint: terminal/latest published training artifact
  `policy_epoch_2800_seed_0.ckpt`.
- Expected checkpoint size / SHA-256: 336,117,389 bytes /
  `8696496e8daf4ead6059d1b721a08f7e454a8de0954b6c762afab8fee9726c5c`.
- `dataset_stats.pkl` expected size / SHA-256: 33,607 bytes /
  `316990648acfcdcd71d72631bc1ab731583c07f8f24a37c8f9db5e585d3b5476`.

The repository has no `policy_best.ckpt`. Its model card declares ACT with the
same 100-query/512-hidden/3200-feedforward configuration and publishes epochs
0 through 2800. The terminal/latest epoch is selected solely from provenance,
before loading any tensor or observing any rollout. Training-curve images may
be inspected for artifact coherence but may not change the selection.

## Locked gates and execution

1. Download only the selected checkpoint, matching stats, and small training
   curves from the immutable revision outside Git; reject a size/hash mismatch.
2. Run the full Stage D policy-only gate on the unchanged saved live Stage C
   observation. Require strict state-dict load, matching finite stats,
   normalization, finite `(1,100,14)` chunk, exact reorder, temporal
   advance/reset, and broad physical bounds.
3. Do not begin a rollout unless every Stage D criterion passes. Report action
   activity relative to prior candidates, but do not tune or splice policies.
4. If the gate passes, run the unchanged Stage F seed-23 episode for 590 steps.
   If needed, retain the fixed sensitivity order 17, 42, 2026.
5. Never mix `_1` tensors or temporal history with another checkpoint.

The learned PASS criteria remain explicit official score at least 6, genuine
live closed loop, complete trace/video/manifest, and clean shutdown.
