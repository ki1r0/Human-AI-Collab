# Research Log

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-09-23 | bootstrap | Reviewed the CausalAssembly plan, the existing action-task prototype, the gearbox asset registry, and the official LingBot-VA repository. The first MVP will use five matched HARD/COMMUTABLE tasks and keep the model environment separate from Isaac. |
| 2 | 2026-09-23 | bootstrap | Confirmed local hardware: four RTX A5000 GPUs with 23,028 MiB each. The official base checkpoint is listed at 24.4 GB and requires offload/single-GPU inference settings. |
| 3 | 2026-09-23 | inner-loop | Began installing a dedicated Python 3.10 LingBot-VA environment and locked the task-pack geometry/oracle contract before downloading model weights. |
| 4 | 2026-09-23 | validation | Completed the isolated `lingbot-va` environment (Python 3.10.16, Torch 2.9.0+cu126, Diffusers 0.36.0, Transformers 4.55.2, USD 25.11), downloaded all 7 safetensors, and passed 24 repository tests plus source-CAD/USD audits. |
| 5 | 2026-09-23 | failure-fix | First base invocation failed before inference because the upstream server imports a separate top-level `configs` module; the adapter was mutating `wan_va.configs`. It was corrected to mutate `wan_va_server.VA_CONFIGS`. A second optional-dependency check exposed the need for a `ModuleSpec` on the flash-attn stub; that was fixed and import smoke passed. |
| 6 | 2026-09-23 | baseline | Official default 224x320, 512-token, 5+10-step run loaded the checkpoint but produced no file after 20 minutes on CPU-offloaded A5000 execution; it was interrupted safely and retained as a diagnostic output directory. |
| 7 | 2026-09-23 | baseline | Explicit fast interface smoke (64x96, text length 64, 1+1 steps) completed. `actions_0.pt` has shape `(1,30,4,20,1)`, finite values and 929 nonzero entries; `latents_0.pt` is finite/nonzero; `demo.mp4` is readable (13 frames, 288x64, 10 fps). This is deployment evidence only, not task success or physics understanding. |
