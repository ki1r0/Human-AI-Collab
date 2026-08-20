# Stage C result — synchronized observation probe

Status: **PASS**.

- Canonical camera order: `head_rgb`, `left_hand_rgb`, `right_hand_rgb`.
- All three tensors are live CUDA `uint8` arrays with shape `(1, 240, 320, 3)`, finite fraction 1.0, nonzero variance, and measured maxima 230/227/228.
- qpos is finite CUDA float32 with shape `(1, 14)` in explicit `[L arm 6, L gripper, R arm 6, R gripper]` order.
- Visual inspection confirmed distinct head overview, left-wrist, and right-wrist views with expected orientation and visible assembly objects.
- A diagnostic-only bilateral shoulder perturbation changed mean absolute pixels by 5.18/35.14/22.01 for head/left/right, proving that all feeds refresh. The saved policy package precedes this perturbation.
- NPZ arrays hash exactly to the live-tensor hashes in the manifest.
- No renderer, asset, physics, CUDA, NaN, or Python errors were found.

Evidence:

- Manifest SHA-256: `ac6c508bfb2c28f23dccd58e67634ead188e40a79ef8b620484508adbe4864f0`.
- Observation NPZ SHA-256: `89ebdd20bed69282bfe7edbc4753519799b8ffa191f2d5e69d7d764737c27083`.
- Console SHA-256: `8379547a0221e07a425a2f5d954c3e3d60b82a59dece4c6e5b3108d2ff5e33df`.
- Runtime artifacts: `artifacts/observation_probe/`.

The first attempt exposed scalar gripper batching; the second established the structural and visual contract but left freshness inconclusive under a static hold. Both were preserved. The third attempt passed the strengthened active-freshness gate.
