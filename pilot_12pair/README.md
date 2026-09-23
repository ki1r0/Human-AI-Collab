# CausalAssembly minimal proof of concept

This package runs two execution-plan baselines on one real, synchronized two-view
assembly task. It is a pipeline and model-capability proof, not evidence for the plan's
matched-physical-mutation hypothesis.

## Implemented smoke task

- Source: EAI Bench subject S002, gearbox segment 3, explicitly restricted to the
  0–800 second assembly episode because one action label repeats later.
- Action A: put the small hub cover on the casing top.
- Action B: put the input hub cover on the casing top.
- Observed order: A then B, extracted from `Y2_GT.xlsx`.
- Provisional process label: both A>B and B>A are feasible; single actions do not
  complete the declared two-action goal.

The process label is not backed by a matched physical mutation or independent oracle.
Accordingly, all necessity metrics from this smoke task are diagnostic only.

## Baseline 1: sequence replay

Run from the repository root:

```bash
python3 -m pilot_12pair.run_smoke
```

The completed reference run produced:

```text
SequenceRecall  1.00
NecessityAcc    0.00
ImitationGap    1.00
FSEM            0.00
FalseRestriction 0.50
PMC             undefined (no matched pair)
```

This is the expected sanity pattern for a method that permits only the demonstrated
order.

## Baseline 2: Qwen3-VL-2B direct video

The launcher reuses the repository's existing GPU image, freezes 16 uniformly sampled
side-by-side frames, verifies their hashes, loads an immutable model snapshot, makes one
deterministic call, and strictly parses it once:

```bash
PILOT_GPU_INDEX=0 ./pilot_12pair/launch_qwen3_vl_2b.sh \
  --output-dir pilot_12pair/outputs/eai_s002_hub_cover_order_rerun_001
```

Use a new explicit output directory for each repetition; the preserved reference run is
write-once and is not silently overwritten.

Defaults:

```text
model     Qwen/Qwen3-VL-2B-Instruct
revision  89644892e4d85e24eaac8bacfd4f463576704203
runtime   cosmos-reason2:cu1281 (image digest recorded in raw output)
decoding  BF16, SDPA, do_sample=false, seed=0
input     16 two-view composite frames with explicit temporal metadata
```

The validated v2 run parsed strictly and produced SequenceRecall 1.00,
NecessityAcc 0.00, FSEM 0.00, UnsafePermission 0.667, and Brier 1.00. The model recalled
the shown order, rejected the unshown reverse order with probability 1.0, and
inconsistently marked both incomplete single-action candidates feasible. This is one
observational example, not a model ranking or causal result.

An earlier v1 run is deliberately preserved under a separate config and output name.
It is invalidated because its format example instantiated one answer pattern, its video
timestamps defaulted incorrectly, and its fenced JSON was parsed permissively. Do not
report v1 metrics.

For an independent Transformers 5.14 environment, build the separate image and use its
separate config/output directory:

```bash
docker build -f pilot_12pair/docker/qwen3_vl.Dockerfile \
  -t causalassembly-qwen:transformers-5.14 .

PILOT_VLM_IMAGE=causalassembly-qwen:transformers-5.14 \
./pilot_12pair/launch_qwen3_vl_2b.sh \
  --model-config pilot_12pair/config/qwen3_vl_2b_direct_transformers_5_14.json \
  --output-dir pilot_12pair/outputs/eai_s002_hub_cover_order_tf5_14
```

The stronger Qwen3.5-4B candidate also has its own config; it has not been executed in
this MVP.

## Data roots and tests

The defaults are:

```text
/media/sunsiliang/CoAI/EAI_bench
/media/sunsiliang/CoAI/CollabAI_obj_det
```

Override them without editing tracked configs:

```bash
EAI_BENCH_ROOT=/path/to/EAI_bench \
COLLABAI_OBJ_DET_ROOT=/path/to/CollabAI_obj_det \
python3 -m pilot_12pair.run_smoke
```

Run tests with:

```bash
python3 -m unittest discover -s pilot_12pair/tests -v
```

Outputs include frozen-frame hashes, input hashes, raw responses, normalized
predictions, and metrics beneath `pilot_12pair/outputs/`. Direct-run evidence is
write-once; use a different `--output-dir` for a repetition or ablation.

## Confidentiality and boundaries

The external materials are marked I2R Confidential. Videos, workbooks, detections,
derived frames, and model caches stay outside Git through `.gitignore`. Do not publish
them or the derived outputs without authorization.

Direct model code loads only the public manifest and frozen model-facing frames. The
evaluator manifest and timing observations are isolated scorer/sequence-baseline inputs.
See `PREPARATION.md`, `RESEARCH.md`, and `DEVIATIONS.md` before extending the experiment.

## Action-policy extension

The VLM pilot above does not test closed-loop manipulation. The proposed physical
action benchmark is specified in `ACTION_MODEL_TASK_SPEC.md`; current VLA/WAM interfaces
and reproduction requirements are summarized in `ACTION_MODEL_RESEARCH.md`. Its
machine-readable, evaluator-only prototype is
`config/action_hub_cover_fastener.prototype.json`.

The flagship intervention uses the existing small hub cover and three fasteners. Round
holes require cover-first assembly, while head-clearance keyholes permit either order.
The primary episode begins with all fasteners retained, so an action policy must remove
them in the hard world but place the cover directly in the commutable world. This is a
new versioned extension and does not retroactively alter the frozen 12-pair VLM plan.

## Five-task action MVP

The first five-task constrained action contract is in
[`CONSTRAINED_TASKS.md`](CONSTRAINED_TASKS.md). It uses the existing gearbox parts for
hub-cover/bolt, washer/gear, key/gear, casing/through-bolt, and casing/dowel probes.
Each task has matched `HARD` and `COMMUTABLE` mechanism variants, an evaluator-only
relation oracle, and a render/collision scene recipe:

```bash
python3 -m pilot_12pair.oracle.generate_task_manifests
python3 -m pilot_12pair.oracle.build_scene_recipes
python3 -m pilot_12pair.oracle.check_task_pack \
  --write-report pilot_12pair/reports/five_task_contract.md
```

The report is an offline contract check. It is not yet an Isaac physics result; a scene
builder and paired swept-volume/contact calibration must pass before model success is
reported.

## LingBot-VA base deployment

The official model is isolated in a Python 3.10 environment and kept outside Git. See
[`lingbot_va/README.md`](lingbot_va/README.md) and the pinned adapter config for the
exact checkpoint/runtime. Once the checkpoint finishes downloading:

```bash
LINGBOT_VA_GPU=0 ./pilot_12pair/lingbot_va/run_base_i2va.sh
```

This produces one native video/action chunk from the official Franka example input. It
is a deployment smoke baseline, not a custom gearbox completion score. Use a separately
rendered/calibrated task observation directory for any later task run.
