# Stage F — genuine ACT closed-loop learned rollout

Classification: **CONFIRMATORY** for seed 23. If it fails, seeds 17, 42, and
2026 are a preregistered fixed sensitivity sequence, with no intervening policy
or task tuning.

## Locked inputs

- Official task: `Template-Galaxea-Lab-Agent-Direct-v0`, prepared R1 checkout.
- Candidate checkpoint revision:
  `yjsm1203/roco_model_act_2@52344a203e0739638cb2c7b11ea632e7b2eb2608`.
- Checkpoint SHA-256:
  `a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1`.
- Stats SHA-256:
  `4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e`.
- Training-faithful adapter passed at Stage D; action interface passed at Stage E.
- 0.01 s physics, decimation 5, 0.05 s / 20 Hz policy loop.
- Maximum 590 learned steps per episode.

Checkpoint provenance remains **APPROXIMATION** because it is third-party and
uses official plus additional demonstrations.

## Locked execution

1. Reset one live environment and reset ACT temporal history once.
2. At every step, read fresh live head/left-wrist/right-wrist uint8 RGB and live
   14-D qpos. No saved demonstration frame or privileged object state enters the
   model.
3. Apply the Stage D contract unchanged: RGB `/255`, qpos standardization, real
   ACT inference, decay-0.1 overlapping-chunk aggregation, action
   denormalization, and exact environment reorder.
4. Reject non-finite or grossly unsafe raw outputs; do not clip or replace them.
5. Step the environment exactly once per inference. Use privileged object state
   only after the step for score/trace evidence, never for policy input.
6. Stop on explicit official `evaluate_score() >= 6`, native done, or step 590.
7. Stream a synchronized 20 fps H.264 composite of all three live policy cameras
   and save every qpos/action/score plus audited object poses to NPZ.

## PASS criteria

- Exact checkpoint/stats hashes and strict real model load.
- R1 embodiment; input/action shapes, dtypes, finiteness, and timing remain valid
  for every executed step.
- At least one action is inferred from each corresponding fresh live observation;
  no replayed action stream or privileged input is used.
- Explicit official task score reaches at least 6 before another termination.
- Result status is PASS with `termination_reason=task_success`.
- Video has exactly `executed_steps + 1` frames and covers reset through terminal
  observation; trace and manifest are nonempty and hash-verified.
- Post-run console audit finds no simulator/GPU error, and the container exits
  cleanly.

If no preregistered seed reaches score 6, the reproduction remains PARTIAL/FAIL
regardless of plausible motion, video quality, or intermediate score.
