# LingBot-VA base smoke report

Date: 2026-09-23  
Official checkout: `/home/sunsiliang/Downloads/lingbot-va`, revision `7c6ffa9`  
Checkpoint: `/home/sunsiliang/models/lingbot-va-base` (`robbyant/lingbot-va-base`)  
Environment: `lingbot-va` (Python 3.10.16, Torch 2.9.0+cu126, Diffusers 0.36.0,
Transformers 4.55.2, USD 25.11)

## Result

The base model produced a native Franka i2va output from the HCF-01/HARD
interface smoke observation. The successful output is under
`outputs/lingbot_va_hcf01_hard_fast_lowres/model_output/`:

- `actions_0.pt`: `(1, 30, 4, 20, 1)`, `bfloat16`, all finite, 929 nonzero entries;
- `latents_0.pt`: `(1, 48, 4, 4, 18)`, `bfloat16`, all finite, 13,799 nonzero entries;
- `demo.mp4`: readable, 13 frames, 288x64, 10 fps;
- `run_manifest.json`: records the model, source revision, input, and all smoke
  overrides.

The run intentionally used `text_max_length=64`, `height=64`, `width=96`, and one
video/action diffusion step. This is a deployment/interface smoke, not a task
completion score and not evidence of causal constraint understanding.

## Reproduction

```bash
LINGBOT_VA_GPU=0 \
LINGBOT_VA_TEXT_MAX_LENGTH=64 \
LINGBOT_VA_VIDEO_STEPS=1 LINGBOT_VA_ACTION_STEPS=1 \
LINGBOT_VA_HEIGHT=64 LINGBOT_VA_WIDTH=96 \
LINGBOT_VA_TASK_OUTPUT=pilot_12pair/outputs/lingbot_va_hcf01_hard_fast_lowres \
./pilot_12pair/lingbot_va/run_task_smoke.sh HCF-01 HARD
```

The official default (`224x320`, text length `512`, `5+10` steps) loaded correctly
but was CPU-offload bound and did not write an output within 20 minutes on this
host; that diagnostic directory is retained separately as
`outputs/lingbot_va_hcf01_hard_base_retry1/`.

## Corrections made during the loop

1. The upstream server's legacy top-level `configs` import was using a second
   configuration object; the adapter now mutates `wan_va_server.VA_CONFIGS`.
2. The optional `flash_attn` stub now has a `ModuleSpec`, allowing Diffusers' optional
   dependency probe to pass while inference remains on official `attn_mode="torch"`.
3. The launcher exports the pinned upstream checkout path for run provenance.
