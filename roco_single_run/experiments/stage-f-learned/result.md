# Stage F result — learned closed-loop episodes

Overall status: **IN PROGRESS; no task-success episode yet**.

## Seed 23

Status: **FAIL** (`max_steps`, best/final score 1/1).

- Strictly loaded 344 tensors / 83,923,087 parameters with exact checkpoint and
  stats hashes.
- Executed 590 fresh-observation inferences and 590 live actions at 20 Hz with
  finite qpos/actions and no native done.
- Score transitioned 0→1 at step 81 when the left arm mounted
  `sun_planetary_gear_2` on its carrier pin; it stayed 1 through step 590.
- All three camera hashes were unique for all 590 policy inputs.
- Right-arm output varied little (per-joint standard deviations 0.012–0.037)
  relative to the active left arm (0.099–0.448), and the right gripper standard
  deviation was 0.000642. Visual audit confirms the right arm never attempted a
  task phase.
- Video verified as H.264/yuv420p, 960×270, 20 fps, 591 frames, 29.55 s.
- No simulator/GPU traceback or error was present; only expected headless GLFW
  warnings matched the broad text audit.

Evidence under ignored run directory `runs/learned_seed23/`:

- `result.json` SHA-256
  `6cf082e7b71fc351f705b5ff04590020d88cb94e2cff6804534935bfd7b0adbd`.
- `trace.npz` SHA-256
  `2b58499984843532b558d69c63b156e30a161f2d7a90030d1969dbca98874251`.
- `episode.mp4` SHA-256
  `9146dceb2aad642fdefde31a9887bc532956a98983009277c8f5027493275d30`.
- `console.log` SHA-256
  `1684a19506b5dcdd67a7590a45563355f585febcf880813643d19f19d0ff0f4c`.

Next locked run: seed 17, unchanged policy and runner.

## Seed 17

Status: **FAIL** (`max_steps`, best/final score 0/0).

- Completed the same 590-inference/590-action contract with fresh live frames,
  exact model load, finite trace, no done, and no simulator/GPU error.
- The left arm approached the first gear but did not mount it. Right-arm action
  standard deviations fell further to 0.0010–0.0048 and right-gripper std was
  0.000499, again matching a visually static right arm.
- All three camera hashes were unique at every step. Video verified at H.264,
  591 frames, 29.55 s.
- Evidence hashes: result `f11e0ab4...cdbb15`, trace
  `841f479a...8b6206`, video `bac5a6cf...edcef`, console
  `806c603f...c4eff`.

Next locked run: seed 42, unchanged policy and runner.

## Seed 42

Status: **FAIL** (`max_steps`, best/final score 0/0).

- Completed all model, camera-freshness, trace, 591-frame video, and shutdown
  checks with no score transition.
- Right-arm action standard deviations again remained 0.0011–0.0047 and the
  right-gripper std was 0.000526. This is the third visually unilateral run.
- Evidence hashes: result `20139c2c...1b7fe`, trace
  `1ba8a99c...f0ede`, video `d736f856...67d6f`, console
  `77c36d2d...b3ec1`.

Next locked run: seed 2026, unchanged policy and runner.

## Seed 2026

Status: **FAIL** (`max_steps`, best/final score 0/0).

- Completed all 590 learned steps and every integrity/evidence check without a
  score transition.
- Right-arm action standard deviations remained 0.0011–0.0055 and right-gripper
  std 0.000602, reproducing the unilateral policy for a fourth layout.
- Evidence hashes: result `7ed921a9...ca78c`, trace
  `c8c9c3af...7cfd6`, video `17cd31a5...5517a`, console
  `1a402e7a...6ca4e`.

## `roco_model_act_2` conclusion

The fixed four-seed sequence is **FAIL** with best scores `[1,0,0,0]`. Every run
strictly loaded the model, used 590 fresh triplet-camera observations, produced
finite actions, recorded a verified 591-frame H.264 episode, and shut down
cleanly. The repeated near-static right arm refutes this checkpoint as a public
full-task Task 1 solution under the reproduced contract. The only scored event
was one genuine left-arm gear placement on seed 23.

The reproduction remains in progress with a separately pinned base ACT
candidate whose own model card describes broader later assembly phases. Its
strict policy-only gate is recorded below.

## Alternative base ACT policy gate

Status: **PASS**.

`yjsm1203/roco_model_act@144d1f73e6ed1f80e04dd969235160452cc517fc`
passed the full Stage D gate against the saved live observation. Its exact
checkpoint/statistics hashes are `9a4c081f...4953e` and
`308b4e45...ec1cc`. All 344 tensors (83,923,087 parameters) loaded strictly;
the raw chunk was finite with shape `(1,100,14)`; training-faithful
normalization, exact reorder, broad physical bounds, temporal advance/change,
and reset reproducibility all passed. Its first action was numerically distinct
from the refuted `_2` checkpoint.

Evidence under ignored run directory `runs/policy_probe_base/`:

- `result.json` SHA-256
  `5ff91561958dab35f2c06c30ffd3a4d44bc2ef993a1c13ebe91bb6ba3731558b`.
- `console.log` SHA-256
  `f79b6a5e26040612c13ac8468a8fad769a684b64981fa4f28711f15fbb4473c4`.

Next locked run: base candidate seed 23 under the unchanged 590-step live
closed-loop contract.
