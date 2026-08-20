# Stage F alternative candidate — `yjsm1203/roco_model_act`

Classification: **EXPLORATORY CANDIDATE SELECTION, THEN CONFIRMATORY GATES**.
This does not alter or supersede the failed `_2` four-seed result.

## Pinned provenance

- Repository: `yjsm1203/roco_model_act` (third-party).
- Immutable revision: `144d1f73e6ed1f80e04dd969235160452cc517fc`.
- `policy_best.ckpt`: 336,112,647 bytes, expected SHA-256
  `9a4c081f104f74e1318e06f477b72f1845c4038648f720a08a72fc3f4854953e`.
- `dataset_stats.pkl`: 33,607 bytes, expected SHA-256
  `308b4e453a276135b66b3753bba46a38c7602c8f5b1eed8be7a08b5dd26ec1cc`.

The direct model card says it is the same ACT shape/config trained on all
filtered simulation gearbox episodes, with pin insertion omitted and later
gear/cover phases retained; it also discloses one known failed training episode.
It remains an **APPROXIMATION**, not an organizer checkpoint.

## Locked gates

1. Download only the two files above from the immutable revision outside Git;
   reject any size/hash mismatch.
2. Run the complete Stage D policy-only test on the exact saved live Stage C
   observation: strict state-dict load, paired finite stats, normalization,
   finite `(1,100,14)` chunk, temporal advance/reset, broad physical bounds,
   and exact reorder.
3. Do not begin simulation rollout unless every Stage D check passes.
4. If it passes, run the unchanged Stage F runner at seed 23 for 590 steps. If
   needed, use the same fixed seed sensitivity order 17, 42, 2026.
5. Keep this checkpoint's temporal history and stats isolated. Never splice its
   actions with `_2`, a replay, the rule oracle, or another learned model.

The learned PASS criteria remain unchanged: explicit official score at least 6,
genuine live closed loop, complete video/trace/manifest, and clean shutdown.
