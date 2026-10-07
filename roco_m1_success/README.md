# RoCo M1 deterministic assembly

Cleanup note (2026-10-06): most historical rollout MP4s under `runs/` were
removed from the working tree to focus on REPAIR M0 and current control-policy
validation. Run metrics, traces, phase scores, and logs were retained. The
known 4 mm reducer video remains at
`runs/dev_reducer_bias4mm/episode.mp4`; a small number of old root-owned videos
could not be moved to Trash and may still be present.

This package runs the official Galaxea R1, RoCo gearbox assets, simulator, and
official score with a minimal deterministic controller built from RoCo's
Differential IK primitives.

Run one isolated phase:

```bash
ROCO_PHASE=reducer ./roco_m1_success/run_successful_assembly.sh
```

Run the full sequence:

```bash
./roco_m1_success/run_successful_assembly.sh
```

Each run writes `console.log`, `result.json`, `trace.npz`, `episode.mp4`, and
`phase_scores.json` below `roco_m1_success/runs/`.
