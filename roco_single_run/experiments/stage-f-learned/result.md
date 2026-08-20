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
