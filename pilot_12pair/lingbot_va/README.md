# LingBot-VA base deployment

The official checkout and 24.4 GB checkpoint are intentionally kept outside Git.
The repository adapter is `run_base_i2va.sh`; it uses the official `franka_i2av`
interface, one GPU, SDPA (`attn_mode="torch"`), CPU offload for VAE/text encoder,
and one generated action/video chunk by default.

## Current local setup

```bash
conda activate lingbot-va
git -C /home/sunsiliang/Downloads/lingbot-va log -1 --oneline
du -sh /home/sunsiliang/models/lingbot-va-base
```

The environment is pinned in `config/lingbot_va_base_i2va.json`. The checkpoint is
downloaded from `robbyant/lingbot-va-base` and is not copied into this repository.

## Official example smoke

```bash
LINGBOT_VA_GPU=0 \
LINGBOT_VA_PROMPT='Pick up the object, place it on the assembly fixture, and finish the demonstrated manipulation.' \
./pilot_12pair/lingbot_va/run_base_i2va.sh
```

Outputs are write-once under `pilot_12pair/outputs/lingbot_va_base_i2va_smoke/`:
`run_manifest.json`, `demo.mp4`, and the upstream action/latent tensors. Use a new
`LINGBOT_VA_OUTPUT` directory for every repetition.

## Custom task observation

The input directory must contain exactly the three filenames expected by the official
Franka config:

```text
observation.images.cam_high.png
observation.images.cam_left_wrist.png
observation.images.cam_right_wrist.png
```

Use a rendered task scene only after its recipe has passed Isaac calibration. A copy of
one image into three camera slots is acceptable for an interface smoke test, but must
not be reported as a multi-view task evaluation.

For a pre-USD interface smoke, the repository includes an unlabeled Pillow sketch
generator. It is useful only to verify that the model consumes the three image slots:

```bash
PYTHONPATH=. conda run -n lingbot-va python \
  -m pilot_12pair.lingbot_va.make_synthetic_observations \
  --task-id HCF-01 --variant HARD \
  --out-dir pilot_12pair/outputs/lingbot_va_synth_hcf_hard
```

Those sketches are intentionally not benchmark media and carry no task score.

The combined task-smoke launcher generates one such sketch and invokes the base model:

```bash
./pilot_12pair/lingbot_va/run_task_smoke.sh HCF-01 HARD
```

On this four-A5000 host, the official 224x320 / 512-token / 5+10-step setting is
CPU-offload bound and can take longer than a short smoke window. For a fast,
explicitly non-scoring interface check, use a separate output directory:

```bash
LINGBOT_VA_TEXT_MAX_LENGTH=64 \
LINGBOT_VA_VIDEO_STEPS=1 LINGBOT_VA_ACTION_STEPS=1 \
LINGBOT_VA_HEIGHT=64 LINGBOT_VA_WIDTH=96 \
LINGBOT_VA_TASK_OUTPUT=pilot_12pair/outputs/lingbot_va_hcf01_hard_fast_lowres \
./pilot_12pair/lingbot_va/run_task_smoke.sh HCF-01 HARD
```

The run manifest records every fast-smoke override. The generated action tensor
proves that the base interface is live; it is not a task-completion or causal
understanding score.

Set `LINGBOT_VA_TASK_OUTPUT` to a new directory for each run; this prevents outputs
from different variants from being overwritten.

This base checkpoint has no adaptation for the gearbox embodiment. A native action
chunk or generated video proves deployment and non-degenerate inference, not causal
constraint understanding. Completion scores require an action adapter, a shared
controller, calibrated task USDs, and task-matched post-training data.
