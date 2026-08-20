# Stage E iteration 2 — independent equivalent action-interface probe

Classification: **CONFIRMATORY FOLLOW-UP**, preregistered after the full replay
failed its original per-step direction criterion.

## Rationale

The full replay is not reclassified. Its mean absolute target error was 0.0100,
but its locked direction test treated every residual target error over 0.001 as
a new request. At that scale, a controller can remain close to target while
inertia moves one step opposite a tiny correction. Post-hoc stratification is
diagnostic only and cannot pass Stage E.

This independent run tests the actual interface semantics with held, reversible
targets so each requested displacement has time to settle. It uses a fresh seed
42 and no replay trace values.

## Locked method

1. Launch the unchanged prepared R1 learned-agent environment at seed 42.
2. Hold the reset robot position for five 0.05 s control steps and take the live
   settled qpos as the baseline.
3. In policy order, apply this bounded offset vector for ten steps:
   `[-.04,-.03,-.02,-.01,.01,.02,-.005,.03,.02,.01,-.01,-.02,-.03,-.004]`.
4. Hold the baseline for ten steps, then the negated offset for ten steps, then
   the baseline for ten steps. This exercises both directions of every one of
   the 14 controlled channels without changing the task, controller, or physics.
5. Convert every policy-order target through the same shared
   `policy_to_environment_order` function used by learned deployment.
6. Save all commands/live qpos, reward/done flags, and exact phase metrics.

## Pass criteria

- R1, 0.01 s physics, 0.05 s control, finite 14-D qpos/action tensors.
- The shared reorder sentinel is exact: `[0..5,7..12,6,13]`.
- All four phases execute ten steps without termination or truncation.
- For every phase, final mean absolute target error is at most 0.02.
- For every phase, cosine similarity between requested and observed phase
  displacement is at least 0.90.
- For every phase, at least 90% of channels requested by more than 0.003 move in
  the requested direction.
- The final return-to-baseline mean absolute error is at most 0.02.

Task score is recorded but not a pass criterion; this is explicitly an
equivalent action-interface test, not an assembly rollout.
