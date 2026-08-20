# Stage B protocol — official rule-policy oracle

Classification: **CONFIRMATORY**.

## Hypothesis

H5: after the Stage A compatibility preparation, the official R1 rule policy can physically complete Task 1 and reach the repository's intended assembly score of 6.

## Locked procedure

1. Use pinned official commit `094a1f76d18c207caec198315f23b1a60dbca94f` with the Stage A compatibility preparation.
2. Launch `Template-Galaxea-Lab-External-Direct-v0` headless with cameras on GPU 2, one environment, seed 2026.
3. Disable demonstration recording but make no change to the rule-policy trajectory, score function, scene randomization, physics step, or controller.
4. Pass a zero placeholder action; the official external environment ignores it and calls `GalaxeaRulePolicy.get_action()` once per 20 Hz environment step.
5. Stop at the first official score of 6, termination/truncation, or 700 environment steps. Record score transitions and object poses.

## Prediction

The official expert reaches score 6 within its 3,320-physics-step schedule (664 environment steps) without a simulator error or time truncation.

## Pass criteria

- R1 remains the active embodiment.
- The unmodified official `evaluate_score()` path reaches at least 6.
- No environment timeout, NaN, missing asset, PhysX error, or uncaught exception occurs.
- The process exits cleanly and leaves a machine-readable result and full console log.
