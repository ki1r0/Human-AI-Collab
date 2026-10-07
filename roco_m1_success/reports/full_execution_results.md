# M1 完整规则流程结果

清理说明（2026-10-06）：本次报告对应的历史 `episode.mp4` 已从工作区移入系统回收站；`result.json`、`phase_scores.json`、`trace.npz` 和 `console.log` 等数值与轨迹证据保留。其他历史 rollout 视频的清理状态见 `roco_m1_success/README.md`。

本次运行按既定计划从分离初始布局启动完整 M1。用户已经确认 reducer-only 的 40 mm clearance / 4 mm bias 终态是可接受的，因此本次没有继续调整 reducer、几何体或官方评分器。

## 运行命令与范围

```bash
ROCO_GPU_DEVICE=1 \
ROCO_PHASE=full \
ROCO_SEED=23 \
ROCO_OUTPUT_DIR=/home/sunsiliang/Human-AI-Collab/roco_m1_success/runs/full_rule_seed23_20260909 \
./roco_m1_success/run_successful_assembly.sh
```

运行入口是 `roco_m1_success/run_episode.py`，控制器是当前的确定性 `M1AssemblyController`。`full` 模式将 carrier 固定在装配基座上，把三个行星齿轮、中心齿轮、ring 和 reducer 放在分散的桌面位置，然后依次执行六个阶段。初始化之后没有再次写入物体 root state；结果中的 `object_pose_writes_after_initialization` 为 `0`。碰撞和官方 `evaluate_score()` 均保持启用。

## 结果

本次完整运行真实执行了 1070 个 20 Hz 控制步，并正常退出。官方结果是：

| 项目 | 结果 |
|---|---:|
| 初始 score | 0 |
| 最高 score | 2 |
| 最终 score | 0 |
| `task_success` | false |
| `status` | FAIL |

score 变化定位如下：

| 阶段 | 事件 | 控制步 | 变化 |
|---|---|---:|---:|
| planet1 | release 后稳定 | 127 | 0 → 1 |
| planet3 | release 后稳定 | 467 | 1 → 2 |
| center | release / retreat 后 | 636–671 | 保持 2，没有新增分数 |
| ring | align/insert 过程中 | 789–802 | 2 → 1 → 2 → 1 → 0 |
| reducer | descend 过程中 | 908–912 | 0 → 1 → 0 |

完整流程在 ring 之前已经存在未满足的装配关系：planet2 没有新增分数，center 完成后也仍只有 2 分。仅凭分数不能精确判定这些阶段的接触原因。**ring 阶段是已有分数明显丢失的阶段**：ring 被搬到装配区并开始对准、插入时，官方 score 反复变化并最终降为 0。reducer 随后仍然运行并产生了视频，但它是在已有堆栈已经失稳之后执行的，因此不能用这次 full run 判断用户认可的 reducer-only 终态。

最终关系也显示前序装配已被破坏，例如：

- `carrier_to_ring` XY 误差约 `35.7 mm`，姿态误差约 `0.151 rad`；
- `center_to_carrier` XY 误差约 `78.2 mm`；
- `center_to_reducer` XY 误差约 `282.0 mm`；
- 三个行星齿轮到对应 pin 的 XY 误差约 `52–94 mm`。

这些数值描述的是 full run 的最终状态，不代表此前已确认的 reducer-only 4 mm rollout。由于 ring 阶段已经出现明确失败，按执行计划跳过 seed17 重复运行，避免把同一失败机制重复消耗 GPU 时间。

## 产物

完整产物目录：

`/home/sunsiliang/Human-AI-Collab/roco_m1_success/runs/full_rule_seed23_20260909/`

其中包括：

- `episode.mp4`：从 reset 到结束的三视角合成视频，约 6.8 MB；
- `result.json`：完整 score transitions、事件快照、初始/最终物体状态和关系指标；
- `phase_scores.json`：阶段摘要；
- `trace.npz`：1070 个样本，包含 `scores (1070,)`、`joint_positions (1070,14)`、`object_poses (1070,49)`、`gripper_positions (1070,2)`、`events (1070,)` 和 `object_names (7,)`；
- `console.log`：启动、Isaac Sim、控制器事件和退出日志。

## ACT 数据准备结论

这次规则 full run 适合做运动和失败诊断证据，但目前还不是 ACT 可直接训练的 demonstration。它保存了三视角合成 MP4 和 post-step 关节轨迹，却没有逐控制步保存：

- 三路独立原始 RGB 张量；
- 与每个 observation 对齐的 14-D policy-order qpos；
- 控制器在 `env.step()` 前实际发出的 14-D action；
- 每个 action 与下一个 observation 的精确时间对齐；
- demonstration 成功/失败标签及其 episode 元数据。

已有的 `roco_single_run` ACT runner 可以正确执行和记录 learned rollout：它使用 `[head_rgb,left_hand_rgb,right_hand_rgb]`、14-D qpos、100-step action chunk、训练配套统计量和环境 action reorder。已有的 `replay_action_interface.py` 也能读取官方 HDF5 的 images/qpos/actions，并验证 action 等于下一时刻 qpos。但仓库中没有把当前 Galaxea 规则控制器直接导出为 ACT 训练 HDF5 的完整采集脚本；`VLA/ACT/act/imitate_episodes.py` 是原始 ALOHA/模拟示例，不能直接把本次 `trace.npz` 当作训练集。

因此本轮没有训练，也没有把 full 失败轨迹伪装成成功 demonstration。最小可行的数据采集路径是：在规则控制器的每个环境 step 记录三路 live RGB、policy-order qpos、实际环境 action 和时间戳，按 ACT 兼容的 HDF5 结构写入；只保留用户认可的成功 episode 作为正向 demonstrations，同时把 full run 的失败作为单独诊断数据。之后才有意义用 `imitate_episodes.py` 中的 ACT 模型配置改出 R1/14-D 数据训练入口，并用现有 `run_roco_single_episode.sh` 做独立 learned rollout。

当前可用于 ACT 的外部数据仍是已有官方/第三方 HDF5 与配套 checkpoint；本次新 full run 没有产生可直接训练的相机状态动作数据集。ACT 完整 rollout 已由另一条独立 learned runner 运行并单独记录，不能用本次规则视频替代。
