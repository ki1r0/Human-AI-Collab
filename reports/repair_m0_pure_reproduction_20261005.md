# Strict pure REPAIR M0 reproduction (2026-10-05)

## Result

The accepted run is:

[`runs/repair_m0_pure_qwen_blocked_20261005_r5/`](../runs/repair_m0_pure_qwen_blocked_20261005_r5/)

It is a real Isaac persistent-worker episode with the local
`Qwen/Qwen3-VL-2B-Instruct` HTTP planner and `feedback_mode: public_contact`.
The planner action trace is:

```text
place(obs_0000, target=hub)
  → guarded_abort/contact_guard
help(obs_0001, clear_target_area(hub))
  → helper succeeds; control epoch 0 → 2
place(obs_0002, target=hub)
  → physical placement completes
finish(obs_0003)
```

The recorded summary is `task_success_gt=true`, `pipeline_success=true`,
`post_help_continuation_success=true`, and `online_status=DONE`.  The
independent `REPAIR_M0_PLACEMENT_V1` evaluator passed all six placement checks:
support contact, radial error, axial error, release contact-free, bounded
release motion, and stable retract.

Measured placement values:

| measurement | value |
|---|---:|
| radial error | 4.149 mm |
| axial error | 14.589 mm |
| release-to-insert motion | 7.451 mm |
| post-settle retract drift | 7.452 mm |
| support force | 14.830 N |

The stricter `M1_PHYSICAL_PREPLACE_CONTROLLED_4_POINT_V1` score is **2/4**:
dynamic supported preplace and real contact before release pass; the
gravity-seated 6-DoF/bolt-alignment and release/stable-retract components are
not claimed as passed by that stricter score.  Thus this is an M0
preplace-to-socket REPAIR reproduction, not a full pick-and-carry, ACT, VLA,
or bolt-installation result.

## No-leakage boundary

The planner receives only the public observation, the last two public history
records, public budgets, and the planner prompt.  In the accepted run:

- the episode ID is opaque (`episode_21bf7037891747b1`); it contains no method,
  scenario, or seed;
- private `metrics.json`, pose, evaluator score, contact records, and
  `metrics_path` are retained only for the evaluator/logger;
- private frame filesystem paths are removed from public `recent_skill`; the
  MP4 remains available on disk;
- a recursive public-key audit found zero forbidden keys, and a value scan of
  all four planner user payloads found no private paths/scenario labels;
- the planner chose all four actions; no rule planner fallback or semantic
  action override was used.  The only output repair is JSON/schema formatting.

`gt_trajectory.jsonl` and the private per-skill `metrics.json` are deliberately
not planner inputs.  `scenario=blocked` is an evaluator-side run argument; the
initial planner observation does not identify it.  The later guarded failure is
public contact/guard evidence, which is part of the REPAIR protocol rather than
ground-truth pose feedback.

## Reproduction

The planner server used for this run was a local OpenAI-compatible endpoint
backed by the cached Qwen model.  It was launched in the Isaac image with
GPU 0 and the model snapshot mounted read-only:

```bash
docker run --rm --gpus 'device=0' --network=host \
  --entrypoint /isaac-sim/python.sh \
  -v "$PWD:/workspace/Human-AI-Collab:ro" \
  -v "$PWD/pilot_12pair/outputs/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct:/models/qwen:ro" \
  human-ai-collab:latest \
  /workspace/Human-AI-Collab/tools/repair_qwen_server.py \
  --model /models/qwen/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
  --host 0.0.0.0 --port 18081 --max-new-tokens 128
```

Then run the pure configuration:

```bash
HRC_REPAIR_PLANNER_ENDPOINT=http://127.0.0.1:18081/v1/chat/completions \
HRC_REPAIR_PLANNER_MODEL=Qwen/Qwen3-VL-2B-Instruct \
HRC_REPAIR_API_KEY=local \
M1_M0_BLOCKER_PROFILE=arm_edge M1_M0_PRECONTACT_GUARD=1 \
python3 -m hrc_repair.run \
  --config configs/repair_m0_pure.yaml \
  --method repair --scenario blocked --seed 100 \
  --out runs/repair_m0_pure_qwen_blocked_20261005_r5
```

Preflight and regression checks used for this run:

```bash
python3 -m hrc_repair.preflight --config configs/repair_m0_pure.yaml --strict
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Both passed.  The accepted live-camera artifact is
[`inner_wall_probe.mp4`](../runs/repair_m0_pure_qwen_blocked_20261005_r5/place_01/inner_wall_probe.mp4);
[`rollout.mp4`](../runs/repair_m0_pure_qwen_blocked_20261005_r5/place_01/rollout.mp4)
is a symlink to the same non-empty MP4.

## Excluded attempts

Earlier attempts are kept for audit but excluded from the result:

- `repair_m0_repro_20261005`: privileged-debug/rule fallback;
- `repair_m0_pure_qwen_blocked_20261005`: model repeated `place` after a
  successful public placement and exhausted the place budget;
- `_r2`: the prompt reminder caused an early false `finish`;
- `_r3`: public `episode_id` exposed `blocked`;
- `_r4`: public `frames_after` exposed the private absolute run path.

These failures motivated the final public-boundary fixes; none is silently
counted as a success.
