# RoCo fidelity matrix

Status vocabulary: **EXACT**, **FUNCTIONALLY EQUIVALENT**, **APPROXIMATION**, **UNKNOWN**.

This is the initial matrix. Runtime measurements replace source-only entries as each gate passes.

| Item | Reference behavior and source | Local implementation | Shape | dtype | Range / units | Ordering | Frequency | Status | Validation performed |
|---|---|---|---|---|---|---|---|---|---|
| Environment generation | AAAI 2026 Task 1, `Template-Galaxea-Lab-Agent-Direct-v0`; official `scripts/VLA_agent.py` | Same official task from pinned checkout plus required collision/texture preparation and R1-era 0.20 m workspace offset | 1 env | N/A | SI physics | N/A | env step configured below | FUNCTIONALLY EQUIVALENT | Stage A launched/reset/stepped; Stage B initial 0.15 m oracle scored 3/6; 0.20 m regression pending |
| Robot embodiment | Paper/public simulation data use Galaxea R1; current source supports R1 and R1 Lite; `robots/robot_bundles.py` | Select supported `GALAXEA_R1_BUNDLE` for checkpoint/data compatibility | 2×6 arm joints + 2 scalar grippers; torso excluded | float32 tensors | arm rad; gripper m | left/right names in articulation order | control loop | FUNCTIONALLY EQUIVALENT | Live R1 verified: 20 joints; controlled indices `[4,6,8,10,12,14,5,7,9,11,13,15,16,18]` |
| Head RGB | `head_rgb` from `head_camera`; `galaxea_robots.py:221-239`, agent env `_get_observations()` | Read live `obs['policy']['head_rgb']` | `(1,240,320,3)` raw; `(1,3,240,320)` model view | raw uint8; model float32 | raw measured `[0,229]`; training expects `[0,1]` after `/255` | camera 0 | rendered each env step | EXACT | Stage A live probe |
| Left wrist RGB | `left_hand_rgb` from `left_hand_camera` | Same live observation | same | raw uint8 | raw measured `[0,231]` | camera 1 | same | EXACT | Stage A live probe |
| Right wrist RGB | `right_hand_rgb` from `right_hand_camera` | Same live observation | same | raw uint8 | raw measured `[0,234]` | camera 2 | same | EXACT | Stage A live probe |
| Depth | Environment records `distance_to_image_plane`, but ACT `camera_names` and training loader consume only RGB | Probe and save depth, exclude it from ACT input | `(1,240,320,1)` | float32 | metres / image-plane distance; background may be `inf` | head,left,right | rendered with cameras | EXACT | Stage A measured all three streams; finite fractions 0.766/0.892/1.0 after steps |
| RGB preprocessing | `act/utils.py:56-74` and `imitate_episodes.py:141-148`: NHWC→NCHW, float, `/255` | Assert raw range, convert NCHW float32, divide by 255 exactly once | `(1,3,3,240,320)` | float32 | `[0,1]` | head,left,right | each inference | EXACT | Training source traced; numeric runtime test pending |
| qpos composition | `VLA_agent.py:84-88`: `[left arm, left gripper, right arm, right gripper]` | Same explicit concatenation | `(1,14)` | float32 | arm rad; gripper m | L0..L5,Lgrip,R0..R5,Rgrip | each inference | EXACT | Source/history traced; runtime values pending |
| qpos normalization | `act/utils.py:74`, generated stats include mean/std; canonical VLA runner omits it | Standardize with checkpoint `qpos_mean/qpos_std`; fail if absent/wrong shape | `(1,14)` | float32 | standardized | unchanged | each inference | FUNCTIONALLY EQUIVALENT | Training source traced; checkpoint stats pending |
| ACT architecture | Official `ACTPolicyWrapper`: ResNet-18, hidden 512, FF 3200, 4 encoder, 7 decoder, 8 heads | Use official class/config | input state 14; output `(1,100,14)` | float32 | normalized actions | policy order | queried every env step with temporal aggregation | EXACT | Source traced; weight load pending |
| Checkpoint | Organizer repo provides no file; candidate `yjsm1203/roco_model_act_2@52344a2` | Download pinned `policy_best.ckpt` and matching stats; verify hashes/state dict | ~336 MB | PyTorch state dict | learned weights | official architecture | N/A | APPROXIMATION | Provenance/file listing verified; download/load pending |
| Action normalization | `act/utils.py:73`; checkpoint output trained in standardized action space | Model output remains normalized until aggregation | `(1,100,14)` | float32 | standardized | policy order | each query | EXACT | Source traced; numeric checkpoint test pending |
| Temporal aggregation | Official wrapper stores overlapping 100-action chunks; canonical ACT eval uses exponential weight `k=0.01`, current wrapper uses `0.1` | Resolve against checkpoint-era reference; expose and log exact coefficient | `(1,14)` output | float32 | standardized | policy order | one aggregate per step | UNKNOWN | Source discrepancy identified; history/checkpoint-era comparison pending |
| Action denormalization | `action * action_std + action_mean` | Require paired stats and apply after aggregation | `(1,14)` | float32 | absolute joint positions: arm rad, gripper m | policy order | each step | EXACT | Source traced; stats range pending |
| Policy→environment reorder | Official `VLA_agent.py:104-109` | `[0:6,7:13,6,13]` | `(1,14)` | float32 | unchanged units | L arm,R arm,L grip,R grip | each step | EXACT | Git history proves correction; unit tests pending |
| Environment action semantics | Agent env `_joint_idx = left arm + right arm + left grip + right grip`; `_apply_action()` calls `set_joint_position_target` | Pass reordered absolute joint targets to `env.step()` | `(1,14)` | float32 | rad and m | environment order | once per environment step | EXACT | Source traced; replay/runtime pending |
| Physics timing | Agent cfg `sim_dt=0.01`, `decimation=5`, default env step 20 Hz | Preserve 20 Hz unless checkpoint-era evidence requires wrapper's claimed 50 Hz | N/A | N/A | seconds | N/A | 100 Hz physics, 20 Hz env | EXACT | Stage A measured `physics_dt=0.01`, `step_dt=0.05`; dataset devkit also says 20 Hz |
| Reset | Official env randomizes objects and writes robot defaults; policy temporal state must reset separately | Seed, env reset, then policy reset; record initial state | N/A | N/A | SI | N/A | episode boundary | FUNCTIONALLY EQUIVALENT | Source traced; deterministic runtime test pending |
| Termination/success | Intended `score == 6`; current `_get_dones()` incorrectly compares `(score,time)` tuple to 6 | Read official `evaluate_score()` result explicitly; success iff score >= 6; also honor timeout | scalar | int/float | score 0..6 intended | N/A | each step | FUNCTIONALLY EQUIVALENT | Static bug confirmed; runtime score validation pending |
| Evidence | Mission requires manifest/log/metrics/video/action trace | Timestamped run directories under `runs/` | per-run | JSON/NPZ/MP4/text | N/A | N/A | every run | FUNCTIONALLY EQUIVALENT | Design pending implementation |

## Known reference contradictions

1. `scripts/VLA_agent.py` casts RGB to float without `/255`; training code divides by 255.
2. `ACTPolicyWrapper` loads only action stats and does not normalize qpos; training/eval code standardizes qpos.
3. `deploy_policy.py` normalizes RGB but builds qpos in environment order and does not reorder actions.
4. `ACTPolicyWrapper` declares 50 Hz; environment and public data are 20 Hz.
5. Current code activates R1 Lite while the README calls R1 the default and the candidate checkpoint uses R1 data.
6. Current `_get_dones()` compares the full `(score, time_cost)` tuple to integer 6.
7. Current gear/carrier configs request PhysX-invalid contact/rest offset pairs; public table USD references absent normal/roughness maps.
