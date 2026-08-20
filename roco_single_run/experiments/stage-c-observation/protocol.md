# Stage C — synchronized observation probe

Classification: **CONFIRMATORY**.

## Contract

Use the pinned R1 agent environment after the Stage A compatibility preparation. After reset, apply one absolute joint-position hold action and save one synchronized package containing:

1. `head_rgb`, `left_hand_rgb`, `right_hand_rgb` in canonical policy order;
2. qpos in `[L arm 6, L gripper, R arm 6, R gripper]` order;
3. simulation timestamp and full tensor statistics;
4. reset-to-step pixel differences as a stale-frame diagnostic.

## Pass criteria

- Each RGB tensor is `(1, 240, 320, 3)`, `uint8`, nonblack, and nonconstant.
- qpos is finite float32 with shape `(1, 14)` and explicit names/order.
- Saved PNGs and NPZ values hash back to the live tensors.
- The console has no renderer, asset, physics, CUDA, NaN, or Python error.
- Visual inspection confirms correct head/left-wrist/right-wrist viewpoints and orientation.
