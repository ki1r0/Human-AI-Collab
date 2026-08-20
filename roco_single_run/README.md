# RoCo 2026 single-run reproduction

Status: **PARTIAL — faithful live ACT pipeline works; public policies do not
complete Task 1**.

This integration reconstructs the AAAI 2026 RoCo planetary-gearbox Task 1 pipeline around the pinned official `rocochallenge/gearboxAssembly` source. It deliberately does not reuse this repository's single-Franka kinematic assembly path.

The stable external reference checkout is currently:

```text
/home/sunsiliang/roco_runtime/gearboxAssembly
```

The selected isolated runtime is the locally cached
`nvcr.io/nvidia/isaac-lab:2.3.0` image. Environment, observation, real-policy,
and equivalent action-interface gates pass. The official/repaired rule oracle
remains diagnosed below score 6. Nine genuine learned episodes across all three
public same-author ACT releases reached best scores
`[1,0,0,0, 0,0,0,0, 0]`; none meets the required score 6.

## One-command episode

After the external checkout/checkpoint preparation described by the research
log, launch the best-observed pinned candidate with:

```bash
ROCO_GPU_DEVICE=3 ./roco_single_run/run_roco_single_episode.sh
```

The command idempotently checks/prepares the pinned official checkout, launches
one fresh headless container, runs 590 live ACT steps by default, records a
three-camera H.264 video plus complete trace/result/console evidence under a new
`roco_single_run/runs/` directory, and shuts Isaac down. It exits normally for
a technically valid task failure; inspect `result.json` and require
`status=PASS`, `task_success=true`, and `best_score>=6` for task success.
The wrapper also supplies the conventional `run_manifest.json`, `metrics.json`,
`run.log`, `video.mp4`, and `action_trace.npz` names as relative symlinks to the
canonical artifacts.

Useful deterministic overrides are environment variables:

```bash
ROCO_SEED=17 \
ROCO_OUTPUT_DIR=/absolute/path/to/run \
ROCO_GPU_DEVICE=3 \
./roco_single_run/run_roco_single_episode.sh --max-steps 590
```

`ROCO_CHECKPOINT`, `ROCO_STATS`, their two `*_SHA256` values, candidate ID, and
immutable candidate revision may also be overridden as a matched set. The
wrapper deliberately defaults to `roco_model_act_2`, the only public candidate
that produced any official score in testing.

## Reproduced contract

- Task: `Template-Galaxea-Lab-Agent-Direct-v0`, Galaxea R1, one environment.
- Cameras: `[head_rgb,left_hand_rgb,right_hand_rgb]`, live uint8
  `(1,240,320,3)`; ACT input `(1,3,3,240,320)` float32 divided by 255 once.
- State: 14-D `[L arm6,L gripper,R arm6,R gripper]`, standardized with the
  checkpoint-paired stats.
- Policy: real 83,923,087-parameter ACT, 100-action chunks, decay-0.1 temporal
  aggregation, one fresh inference per environment step.
- Action: denormalized absolute joint positions, reordered to
  `[L arm6,R arm6,L gripper,R gripper]` for the environment.
- Timing: 0.01 s physics, decimation 5, 0.05 s / 20 Hz control.

## Known blocker

The organizer repository exposes a checkpoint path but publishes no ACT
checkpoint or matching statistics. All available third-party candidates pass
the exact policy and action-interface contracts, but their live output is
strongly unilateral: the right arm stays near reset after the left subtask
misses or stalls. No policy output was replaced, clipped, mirrored, spliced, or
supplemented with privileged state. A successful learned Task 1 episode now
requires a capable checkpoint (or new training), not another integration fix.

See:

- `research-log.md` for the chronological trace;
- `fidelity-matrix.md` for source-to-local contracts and deviations;
- `issues-and-fixes.md` for evidence-driven failure analysis;
- `research-state.yaml` and `findings.md` for persistent project state.

Primary evidence is summarized in `experiments/stage-f-learned/result.md`;
large videos and traces remain in ignored `runs/` directories.
The one-command regression is under `runs/launcher_smoke_seed23/` (one-step
infrastructure smoke only, not a task-success trial).
