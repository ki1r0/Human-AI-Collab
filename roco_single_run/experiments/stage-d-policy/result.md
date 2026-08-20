# Stage D result — real ACT policy-only test

Status: **PASS**.

- Loaded pinned `policy_best.ckpt` (SHA-256 `a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1`) and paired stats (SHA-256 `4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e`).
- Strict state-dict load: 344 tensors, 83,923,087 model parameters, zero missing keys, zero unexpected keys.
- The exact saved Stage C package normalized to RGB range `[0, 0.9294]` and qpos z-score range `[-1.1642, 1.1767]`.
- Output chunk was finite float32 `(1,100,14)`, normalized range `[-2.1268, 2.1216]`.
- First denormalized policy action was physically plausible and near the live R1 pose; grippers were 0.0382/0.0347 m.
- Exact environment reorder produced `[L6,R6,Lgrip,Rgrip]` and the sentinel mapped to `[0..5,7..12,6,13]`.
- Second inference used official wrapper decay 0.1 with oldest/newest weights `[0.4750208, 0.5249792]` and changed the aggregate.
- Reset cleared all temporal state and reproduced the first action within `1e-6`.

Evidence:

- `artifacts/policy_probe/result.json`
- `artifacts/policy_probe/console.log`

Two failed attempts are retained: top-level package import incorrectly required a live Kit app, then NumPy 2 pickle naming was incompatible with the image's NumPy 1.x. Both were repaired without altering model weights or numeric arrays.
