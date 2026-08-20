# Stage C — synchronized observation probe

Classification: **CONFIRMATORY**.

## Contract

Use the pinned R1 agent environment after the Stage A compatibility preparation. After reset, apply one absolute joint-position hold action and save one synchronized package containing:

1. `head_rgb`, `left_hand_rgb`, `right_hand_rgb` in canonical policy order;
2. qpos in `[L arm 6, L gripper, R arm 6, R gripper]` order;
3. simulation timestamp and full tensor statistics;
4. reset-to-step pixel differences as a stale-frame diagnostic.

Because a held static scene may be byte-identical, save the canonical package first, then apply a bounded 0.05 rad bilateral shoulder perturbation for five steps. Require nonzero head change and mean absolute pixel difference above 0.1 in both moving wrist cameras. The perturbation is diagnostic-only and is never supplied to the learned policy.

## Pass criteria

- Each RGB tensor is `(1, 240, 320, 3)`, `uint8`, nonblack, and nonconstant.
- qpos is finite float32 with shape `(1, 14)` and explicit names/order.
- Saved PNGs and NPZ values hash back to the live tensors.
- Both wrist streams and the head stream update after the bounded active perturbation.
- The console has no renderer, asset, physics, CUDA, NaN, or Python error.
- Visual inspection confirms correct head/left-wrist/right-wrist viewpoints and orientation.
