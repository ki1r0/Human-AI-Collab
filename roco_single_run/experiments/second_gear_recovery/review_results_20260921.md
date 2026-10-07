# 2026-09-21 ACT 第二个 gear 实验审查

本文件只审查已有 rollout 产物和当前实验代码，不启动仿真、不修改 ACT 权重、动作源、场景几何或评分逻辑。目标是判断 H1/H2/H3 是否真正检验了各自假设，以及现有指标是否足以支持因果结论。

## 当前结论

H1 和 H2 都没有解决问题；第一轮 H3 也没有证明“延长到 1180 步”这一解释，因为它在第 684 个控制步被环境内部的 rule-policy schedule 提前终止。第一轮 H3 的确提供了新的证据：同样的 checkpoint 和 `.1` newest-first 聚合在第 79 步得到分数 1，但在原生 cutoff 处没有机会继续到 1180 步；它没有带来第二次安装。第 684 步的 `1→0` 是环境在终止后自动 reset、而 runner 又读取 reset 后状态造成的记录伪影，不能解释为装配关系被 ACT 动作物理破坏。修正 H3 已完成：在不改变 ACT 计算的情况下执行满 1180 步，最高/最终分数仍为 `1/1`，没有第二次安装。

修正后的 H3 使用 `--use-env-time-limit`。该选项只把内部 rule-policy 的 schedule cutoff 提高到环境 60 s 的上限，使 1180 个 20 Hz 控制步有机会执行；它不是对未经修改原生终止规则的复现。结果是 `executed_steps=1180`、最高/最终分数 `1/1`，因此新增时间段没有救回任务。

## 实验逐项核对

| 实验 | 实际配置与产物 | 结果 | 对原假设的支持 |
|---|---|---|---|
| H1 newest-only | 同 checkpoint、seed 23、590 步，`temporal_decay=1000`；`secondgear_newest_seed23_20260909T092724Z` | 590 次 ACT，score `0/0/0`，无评分转移，所有数组/视频/trace 完整 | 不支持把历史聚合作为可直接修复；但因整个轨迹都改变，不能单独证明它不是第二次失败的必要条件 |
| H2 original-style | 同 checkpoint、seed 23、590 步，`temporal_decay=-0.01` 在本地聚合器中等价于 oldest-first；`secondgear_originalagg_seed23_20260921T040004Z` | 590 次 ACT，score `0/0/0`，未拾取任何 gear；权重方向和有限性已核对 | 不支持上游 oldest-first `k=.01` 是本任务的修复 |
| H3a 原生长时域尝试 | `.1` newest-first，`max_steps=1180`，保存 raw chunk；`secondgear_longhorizon_seed23_20260921T040730Z` | 实际仅 684 步；79 步 `0→1`，684 步记录 `1→0`，`native_terminated_before_success` | 不能作为 1180 步测试；终止前第 683 步仍为 score 1，不能把 reset 后的 0 当作物理失败 |
| H3b 修正长时域 | `.1` newest-first，`max_steps=1180`，`--record-raw-chunks --use-env-time-limit`；`secondgear_longhorizon_envlimit_seed23_20260921T041727Z` | 1180 步、最高/最终 `1/1`、无第二次安装；视频/trace 完整 | 延长有效控制时长不是当前阻碍；不再继续扫描时间或聚合参数 |

H1/H2 的 checkpoint、stats、seed、任务和物理 dt/control dt 保持一致；两者均为真实 ACT 动作，不含规则动作替换。H2 的 `-0.01` 只是利用当前实现的权重指数和顺序表达 oldest-first，报告时应明确这一点，避免误称为改了模型。

## H3a 的终止与指标审计

环境中的 `_get_dones()` 除了 60 s timeout 外，还检查 `rule_policy.count >= rule_policy.total_time_steps`。rule policy 的 `total_time_steps` 约束在 684 个环境控制步处触发；这与 H3a 的 `executed_steps=684` 和终止原因完全吻合。因此：

* `max_steps=1180` 是请求上限，不是实际执行长度；H3a 不能被描述为完成 1180 步。
* `score` trace 是 runner 在每次 `env.step` 返回后重新调用 `evaluate_score()` 得到的值；H3a 的转移为第 79 步 `0→1`、第 684 步记录 `1→0`。在第 684 步，环境先因 native rule cutoff 设置 termination 并执行 `_reset_idx()`，随后 runner 才读取 score 和 object pose，所以第 684 步的 0 是 reset 后新布局的 score。第 683 步仍为 score 1；`final_score=0` 不能用作原 episode 的最终物理状态。
* H3a 的 `raw_normalized_chunks` 形状为 `(684,100,14)`，全部有限；视频 685 帧与 `executed_steps+1` 一致。它是可审计的部分 rollout，但不是完整长时域 rollout。
* reset 前最后一个可用状态是第 683 步：score 仍为 1，`sun_planetary_gear_2` 的位置约为 `(0.5638877, 0.0278380, 0.9131486)`，carrier 约为 `(0.6108109, 0.0010106, 0.9029301)`，与第一次安装关系一致。第 684 步 trace 中出现的 `(0.6891, 0.2511, 0.9200)` 是 reset 后的新初始布局，不是 ACT 把 gear 推到那里的证据。

H2 报告中的“gear 2 maximum positive Z displacement = -0.01060 m”表述应改为“最大 `Δz=-0.01060 m`，没有正向抬升”。负值不是一个正位移。该命名问题不改变 H2 的对象状态结论，但会误导读者。

## H3a 对照性的限制：初始机械状态相同，但不是 bit-exact replay

H3a 与原始 `.1` baseline 的 `initial_qpos_policy_order` 和 `initial_object_poses` 逐元素完全相同，object name 也相同。这确认机械 reset 状态可比。但三路初始 camera hash 不同，且第一步之后 policy actions 已不同：第 2 步 qpos 最大差约 `2.17e-4`，第 2 步 action 已有非零差；到前 590 步 action 的最大差约 `0.4256`。所以 H3a 的前 590 步不是原始 baseline 的逐步复播，不能把两段轨迹的差异归因于“仅多执行了 590 步”。

更准确的说法是：两次运行从相同记录的机械 reset 开始，使用相同 checkpoint/聚合器和环境配置，但相机渲染/输入存在差异，闭环反馈很快放大了微小差异。H3a 只能回答“在一次相同 seed 的 fresh live run 中，原生环境是否允许执行到 1180 步”（答案是否定的）；它不能回答“额外 horizon 是否救回任务”，也不能回答严格的 counterfactual “保持前 590 个 action 完全不变后再延长”。

## Raw chunk 的阶段意图：当前动作与未来预测必须分开

H3a 的 raw chunk 允许区分两个概念：

1. 已执行的聚合动作中，约 321--684 步左夹爪持续收紧，右夹爪约 `0.038--0.041 m`，仍接近打开；右臂当前聚合动作的方差很小。因此实际控制没有形成一个清晰的右夹爪闭合/搬运序列。
2. 不能据此断言模型完全没有右臂意图。chunk 中较远未来的右臂预测与 chunk 第一项可能有明显差异；这可能表示模型预测了未来动作，但时间聚合和闭环执行尚未把它变成当前控制。后续审查必须分别报告 `raw[t,0]`、未来项（例如 `raw[t,99]`）和最终执行的 aggregated action，不能只看一个窗口的聚合均值。

因此目前最稳妥的因果描述是：ACT 在 live 闭环中没有把第二次有效抓取推进到评分；它是否“没有预测右臂动作”，不能由当前右臂实际动作直接推出。候选原因仍包括视觉/物理分布偏移、阶段反馈失配、checkpoint 数据覆盖不足和接触后恢复能力不足；时间聚合修复已被 H1/H2 的结果显著削弱，但还不是形式上的单变量证明。

## 修正 H3 的审查标准

`--use-env-time-limit` 在 runner 中把 `base_env.rule_policy.total_time_steps` 替换为环境最大控制时长对应的 physics steps；它不改变 ACT 计算、动作重排、评分器、物体几何或接触参数。这个改动属于为诊断暴露环境原生终止边界，不应被写成模型修复。H3b 已经完成，结果仍为最高/最终 `1/1`。

修正 H3 完成后检查：

* `result.json` 中 `use_env_time_limit=true`、原始/有效 rule horizon、`executed_steps`、`termination_reason` 是否相互一致；
* `trace.npz` 是否为 `(executed_steps,100,14)` 的 `raw_normalized_chunks`，有限且逐步对应 ACT inference；
* 首 590 步是否仍只达到最高 score 1，以及第 591--1180 步是否出现稳定第二 gear 搬运或只是继续扰动已有装配；
* 每个窗口同时给出左/右夹爪的 executed aggregate、raw chunk first、raw future-end，并配合 gear 的 z/XY/rotation 和 score transition；
* 如果 score 先到 1 又回到 0，先检查是否发生了 env reset：只有在 reset 前的连续 object pose/score 证据确认装配确实被破坏时，才能报告“长时域继续执行破坏已有关系”；若 0 只出现在 reset 后，必须报告为终止/reset 记录伪影，而不是报告为物理失败或第二次抓取成功。如果 native timeout 在 1180 前再次触发，仍按实际步数结束，不补造缺失轨迹。

## 停止判断

截至目前，H1、H2 和修正后的 H3b 都未产生 score 6；两个聚合消融和一次有效长时域测试均失败。已经有三类有明确假设的失败尝试：newest-only、original-style aggregation、延长有效控制时长。应停止继续扫描聚合参数，按证据转向 checkpoint 数据覆盖/视觉-物理域一致性分析或明确报告当前权重无法完成 full ACT rollout。任何规则动作或 scorer 改写都不能用来填补 score 6。
