# ACT 推理链独立审查：第二个 gear 抓取失败

审查对象是 `learned_full_act_seed23_20260909T082411Z` 的真实闭环 rollout，以及本地 ACT adapter、RoCo wrapper、上游 ACT evaluation 和一份可用的官方 HDF5 示范。审查只读，不修改 checkpoint、动作源、环境、评分器或 runner；本文件也不把任何规则控制器动作混入 ACT 实验。

## 结论摘要

目前没有发现一个能够单独解释第二个 gear 失败的本地推理/预处理 bug。`roco_single_run/roco_policy.py` 的输入顺序、RGB 缩放、qpos 归一化、输出动作重排和状态性时间聚合，与训练数据和 RoCo wrapper 的关键约定一致；实际第一颗 gear 能够被抓取并得到分数 1，也不支持“动作通道完全接错”的解释。

失败更像是模型在闭环分布外的阶段切换/接触状态上失配：第一次拾取后，左臂继续发出第二次接触尝试，约在 320--335 步只造成一次短暂的部分夹持和旋转；右臂没有形成第二次抓取序列。以下仍是待通过对照实验检验的假设，而不是仅凭一次 rollout 得出的 checkpoint 因果证明。

## 1. 推理契约逐项核对

| 项目 | 当前本地实现 | 对照证据 | 判断 |
|---|---|---|---|
| 相机顺序 | `head_rgb`, `left_hand_rgb`, `right_hand_rgb` | `CAMERA_NAMES`、ACT 数据加载器和模型配置使用同一顺序 | 一致 |
| 图像范围 | 原始 `uint8`，转为 `[0,1]` 后交给 `ACTPolicy` | 训练 loader 在进入 policy 前除以 255；policy 内部再做 ImageNet normalization | 一致 |
| qpos 顺序 | `[左臂6, 左夹爪, 右臂6, 右夹爪]` | 数据集 merge/训练路径使用该 policy 顺序 | 一致 |
| qpos 归一化 | `(qpos-qpos_mean)/qpos_std` | ACT 训练与官方 ALOHA eval 使用同一统计量 | 一致 |
| 动作顺序 | policy 输出 `[左臂6, 左夹爪, 右臂6, 右夹爪]`，送环境前变为 `[左臂6, 右臂6, 左夹爪, 右夹爪]` | Galaxea 环境 joint index 为左臂、右臂、左夹爪、右夹爪；独立接口 probe 已通过 | 一致 |
| 夹爪量纲 | 归一化动作反变换后直接使用原始夹爪位置单位 | 首次夹取时目标与实测左夹爪均有闭合响应 | 暂无明显 bug |
| query/horizon | 每个 20 Hz 环境步重新查询；chunk horizon 100 | 数据集每 episode 590 个控制样本，RoCo wrapper 同样使用 100 | 训练/任务时间尺度一致 |
| 时间聚合 | 只保留仍在 horizon 内的预测；按时间顺序收集；指数权重使较新预测权重更高 | RoCo `policy_wrapper.py` 是同样的 `.1`、newest-first 语义 | 一致 |
| stale prediction | `0 <= timestep-start < 100` 才纳入；过期 chunk 排除 | 与 wrapper 的有效列/零掩码逻辑等价 | 一致 |

上游文件中确实存在容易混淆的其他路径：通用 `deploy_policy.py` 的 qpos/action 顺序与 gearbox 任务不同，`scripts/VLA_agent.py` 也缺少 RGB `/255`。这些路径不是本次 runner 使用的路径；不能把它们的缺陷归因到本次 rollout。另一个实际差异是通用 wrapper 宣称 50 Hz，而 RoCo gearbox 数据和本地 runner 是 20 Hz；本地 20 Hz 与 590-step 示范长度相符，不能在没有对照实验的情况下改成 50 Hz。

## 2. 时间聚合消融

为避免把“上游 ACT 代码”和“RoCo wrapper 约定”混为一谈，使用同一 `_2` checkpoint、同一 stats 和一个本地官方 HDF5，在离线示范观测上先比较聚合器。模型对 590 个观测批量推理，然后只改变聚合方式；这不是闭环成功率测试。

| 配置 | qpos 已归一化时 overall action MAE |
|---|---:|
| 当前 RoCo：`k=.1`、newest-first | **0.00501** |
| 上游 ACT：`k=.01`、oldest-first | 0.00552 |
| `k=.01`、newest-first | 0.00538 |
| `k=.1`、oldest-first | 0.00636 |
| 只取当前 chunk 的第一动作 | 0.00631 |

当前配置在该示范上的 MAE 最低。按时间窗口（0--170、170--320、320--420、420--520、520--590）分别为 `0.00684, 0.00536, 0.00308, 0.00539, 0.00201`；`k=.01` oldest-first 为 `0.00732, 0.00582, 0.00277, 0.00697, 0.00239`。因此，虽然 `k=.01` oldest-first 是值得保留的上游风格消融，但仅凭源码差异不能把当前 newest-first 判定为 bug，也不应在没有 rollout 结果前把它作为修复。

该消融的限制很重要：只有一份 HDF5，行为克隆 MAE 不等于接触任务成功率；示范上的平均动作误差也不能排除闭环视觉/物理分布偏移。

## 3. 失败窗口的实证定位

来自 `seed23_diagnosis.json` 和 `trace.npz` 的对象状态与动作统计如下：

* `sun_planetary_gear_2` 在第一次阶段从约 `z=.9200 m` 上升到约 `z=1.1007 m`，随后稳定到 carrier 附近；官方分数在约第 79 步从 0 变成 1。这证明至少左侧输入、动作重排、夹爪单位和基本仿真闭环能够共同完成一次真实拾取/安装。
* `sun_planetary_gear_1` 在 320--335 步附近只从约 `z=.9027 m` 短暂升到约 `z=.9175 m`，同时旋转约 `.96 rad`，随后回到桌面高度。它没有达到 carrier，也没有形成稳定搬运。
* 同一窗口左夹爪目标约由 `.0343 m` 收到 `.0167--.0183 m`，实测夹爪跟随；右夹爪保持接近打开值（约 `.0385 m` 目标、`.0370 m` 实测），右臂动作方差很低。因而当前证据是“左侧部分接触/夹持失败后没有恢复”，不是“右夹爪已经成功夹住后掉落”。
* 全段 policy action 的右臂 6 个通道标准差约为 `.0026--.0124`，右夹爪约 `.00057`；第二阶段之后更小。这个事实说明模型没有明显发出第二个右臂抓取动作，但不说明为什么它没有发出该动作。

结合参考 HDF5 的阶段分布（左臂动作主要在 0--170，随后应发生阶段切换，右臂从后段开始活动），最需要验证的是闭环中观测是否已足够接近模型学到的阶段切换条件。当前证据无法区分：物体布局/相机/机器人形体的 domain shift、checkpoint 的阶段覆盖不足，还是接触后的状态反馈偏离。

## 4. 按证据排序的候选原因

1. **闭环分布偏移或阶段切换失败（目前最高优先级）**。示范离线误差正常，但 live episode 在第一次接触后进入不同的物体姿态/视觉状态；模型因此继续左侧搜索，没有转到第二个抓取策略。
2. **checkpoint 数据覆盖不足**。`yjsm1203/roco_model_act_2` model card 描述的是融合后的约 241 episodes、固定长度处理、100-step chunk、三相机和 14 维动作；本地 stats 中的示例数量与 model card 的固定长度描述并不完全相同。公开 model card 不能证明它覆盖当前 seed 的完整、多次成功双臂序列。
3. **接触/夹爪跟踪导致模型输入进一步偏离**。第二次尝试确实有左夹爪闭合和 15.6 mm 瞬态上升，所以这是合理的机械/观测反馈候选，但第一次拾取有效，尚无证据表明全局夹爪缩放错误。
4. **时间聚合配置不匹配**。这是可检验的低优先级候选；离线示范反而支持当前 `.1` newest-first，因此不能先验地当成主因。

## 5. 最小、可证伪的下一轮实验

不改模型、seed、场景、物理和评分，只使用 ACT 产生动作，按同一输出目录格式运行：

1. 当前基线：`temporal_decay=0.1`，newest-first。
2. 上游风格消融：实现/运行 `k=0.01`、oldest-first；不要通过改变语义而只传一个容易误读的负 decay，记录实际聚合顺序和权重。
3. 若前两者仍不能区分，可做一次“不聚合/当前 chunk 第一动作”对照，不再扫描更多参数。

每次至少记录 250--360 步的 raw ACT chunk、聚合前后动作、夹爪目标/实测值、end-effector pose、目标 gear 的 z/XY/旋转、评分转移及视频。判定重点是：

* 第一次安装是否仍然保留；
* 第二个 gear 是否出现稳定上升并到达 carrier，而不是仅有小于约 20 mm 的瞬态抬升；
* 右臂/右夹爪是否真正发出阶段性动作；
* 不同聚合器是否只改变平滑轨迹，还是改变了模型的阶段选择。

若 `.01` oldest-first 仍显示左臂部分接触、右臂无抓取，且 raw chunk 本身也不包含右臂闭合/搬运意图，则时间聚合不再是首要解释，后续应检查 checkpoint 数据覆盖和视觉/物理域偏移。若 raw chunk 包含右臂意图但聚合后被压掉，才有理由继续研究聚合器。任何规则动作、镜像动作、物体 teleport、评分修改或人为恢复都不能计为 ACT 完整 rollout。

## 6. 审查边界

本审查没有运行 Isaac Sim，也没有声称已经得到成功的 ACT 完整 rollout。离线结果只覆盖一份 HDF5，且现有 live runner 尚未保存完整 raw chunk/TCP 轨迹；对象位姿是仿真 privileged trace，只用于事后定位，不能作为 policy 输入。若下一轮仍失败，应报告为模型/闭环实验失败，而不是用规则控制替换 ACT 来填补成功结果。

主要复核文件：

* `roco_single_run/roco_policy.py`
* `roco_single_run/scripts/run_roco_single_episode.py`
* `roco_single_run/experiments/second_gear_recovery/diagnosis.md`
* `roco_single_run/experiments/second_gear_recovery/seed23_diagnosis.json`
* `/home/sunsiliang/roco_runtime/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT/policy_wrapper.py`
* `/home/sunsiliang/roco_runtime/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT/act/imitate_episodes.py`
