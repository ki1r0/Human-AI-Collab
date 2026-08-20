# Research Log

Chronological, append-only record for the RoCo reproduction.

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-20 | bootstrap | Read the full reproduction mission and confirmed pre-existing user changes: modified `.gitignore`, untracked `PILOT_12_PAIR_EXECUTION_PLAN.md`, and untracked `pilot_12pair/`. No user work was altered. |
| 2 | 2026-08-20 | bootstrap | The required Orchestra Research skills were absent. The official `npx @orchestra-research/ai-research-skills install --all` command failed because Node/npx is not installed. Installed all 98 skill directories with Codex's GitHub skill installer from Orchestra commit `773a52944ba4747a18bd4ae9ade53fff041adcbc`; loaded `autoresearch` and its templates/routing instructions. |
| 3 | 2026-08-20 | bootstrap | Captured host baseline: Conda base Python 3.13.11 has no torch/isaacsim/isaaclab; driver 550.163.01; four RTX A5000 GPUs; host Isaac Sim reports `5.1.0-rc.19`; host IsaacLab is a dirty `v2.2.1-213-g42e61645c9` checkout and must not be modified. |
| 4 | 2026-08-20 | literature | Verified the AAAI 2026 gearbox benchmark is the relevant RoCo generation. The existing workspace is a separate single-Franka Isaac Sim 5.1 project with kinematic assembly support; it is not the dual-arm Galaxea R1 benchmark and will not be mixed into the baseline. |
| 5 | 2026-08-20 | literature | Pinned official `rocochallenge/gearboxAssembly` at `094a1f76d18c207caec198315f23b1a60dbca94f` and official devkit at `23522d72af214158d3c56ee2f171888c3e74698d`. Stable checkouts are under `/home/sunsiliang/roco_runtime/`. |
| 6 | 2026-08-20 | literature | Traced camera, proprioception, ACT, normalization, temporal aggregation, action reordering, environment step, score, reset, rule-policy, and replay paths. Found upstream contradictions in preprocessing, active robot selection, standardized deployment ordering, and success termination. |
| 7 | 2026-08-20 | bootstrap | Selected the local `nvcr.io/nvidia/isaac-lab:2.3.0` image as the initial isolated runtime. It contains Python 3.11.13, torch 2.7.0+cu128, Isaac Lab 0.47.1/2.3.0, and Isaac Sim 5.1.0. CUDA inference sees an RTX A5000 despite the image metadata requesting a newer driver. Full Kit launch remains to be tested. |

