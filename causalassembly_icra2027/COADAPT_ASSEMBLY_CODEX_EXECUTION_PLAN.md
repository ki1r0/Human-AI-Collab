# CoAdapt-Assembly 执行计划（交给另一个 Codex）

> 这是实现规范，不是概念草稿。请从仓库根目录开始工作，先审计、再实现、再验证。
> 不要直接扩建完整 benchmark，也不要在没有证据时改写研究主张。

## 0. 给执行 Codex 的总指令

你正在实现一个小规模、可证伪的 HRI assembly pilot。项目最终研究问题是：

> 在协作装配中，机器人能否从交互历史中区分任务物理约束、机器人自身能力限制、人类稳定偏好/能力和人类临时意图，并在未见过的组合或动态变化下选择正确的协作动作？

暂定项目名：

```text
CoAdapt-Assembly:
Disentangling Task, Self, and Partner Constraints
for Adaptive Human-Robot Assembly
```

必须遵守以下规则：

1. 保留用户已有修改，不回退、不覆盖无关文件。
2. 所有新实现放入仓库根目录下的 `coadapt_assembly/`。
3. 不要先做完整 benchmark。先完成本文定义的 12-pair pilot 和 go/no-go 判断。
4. 不要把 `runtime/magic_assembly.py` 当作物理 oracle。它可以摆放零件、生成演示和提供高层技能，但不能证明碰撞、可达性、承重或真实物理可行性。
5. 模型不得看到 ground-truth DAG、socket 名称、USD prim path、profile label、能力标签或 evaluator manifest。
6. 不要把 VLM 问答准确率当作最终结论。主要证据必须来自下游协作动作和结果。
7. 第一阶段不要训练低层抓取策略，不要训练完整 VLA，不要做视频 world-model rollout。
8. RL 不是默认贡献。只有在简单非 RL 方法失败且 RL 显著超过强 baseline 时才加入。
9. 任何真实 API key 只能放在未跟踪的 `.env` 或本地配置中，绝不能写入代码、日志或 manifest。
10. 如果仓库现实与本文冲突，保持本文的科学约束，只调整软件接口，并把偏差写入 `coadapt_assembly/DEVIATIONS.md`。

---

## 1. 开始前必须阅读的仓库内容

从仓库根目录依次阅读以下文件。不要假设旧文档里的设计已经实现。

### 1.1 环境和启动方式

| 路径 | 用途 | 执行要求 |
|---|---|---|
| `README.md` | Docker、Isaac Sim/Isaac Lab、GPU、X11 和环境变量总说明 | 先确认运行条件，不要重新发明启动链 |
| `docker-compose.yml` | 容器、GPU、volume、环境变量 | 不要提交 `.env`；保留现有 `hac` service |
| `docker/run_demo.sh` | 容器入口 | 了解它最终调用 `run_main.sh` |
| `run_main.sh` | Isaac Lab launcher 和主场景入口 | 当前主场景是 `assets/simple_room_scene.usd` |
| `tools/check_setup.py` | 主机环境预检 | 在运行 Isaac 前先执行 |
| `docs/docker_run.md` | 本地 Docker 说明 | 本地运行时参考 |
| `docs/cloud_deployment.md` | 云端 GPU 和显示方案 | 本地 GPU 不足时参考 |

### 1.2 当前装配系统

| 路径 | 用途 | 注意事项 |
|---|---|---|
| `runtime/magic_assembly.py` | `combine`/`separate`、socket map、gearbox 高层拼装 | 只可作为高层 executor/演示生成器，不能作为物理 oracle |
| `runtime/ui.py` | `/combine_*` 快捷命令、运行循环、UI | 可复用命令入口，不要把 benchmark 逻辑硬塞进 UI |
| `runtime/llm_commander.py` | 当前 `combine/separate/noop` action contract | benchmark 使用独立 action schema，必要时写 adapter |
| `agent/reason2.py` | 当前推理提示、belief update、action parsing | 只能作为已有 baseline 参考，不要把其内部状态当 ground truth |
| `memory/short_term.py` | STM、task state、dialogue state | 当前存的是 agent belief，不应读取 evaluator hidden state |
| `belief/manager.py` | belief 管理和更新 | 可复用接口思想，先避免大改 |
| `sensor/camera.py`、`sensor/perception.py` | 摄像头与感知入口 | Sensor-only track 使用这些接口或新 adapter |
| `control/franka.py` | Franka 控制 | 第一阶段可绕过低层控制；集成阶段再使用 |
| `runtime/config.py` | camera、robot、capture、GT monitor 配置 | 注意当前存在 GT monitor；benchmark model-facing path 必须隔离 |

### 1.3 Gearbox 资产

优先检查：

```text
assets/parts/
assets/CAD_edit/
assets/simple_room_scene.usd
tools/author_gearbox_sockets.py
tools/author_assembly_sockets.py
tools/scatter_gearbox_parts.py
tools/prep_asset_physics.py
tools/prep_all_physics.py
tools/inspect_nested_rigid_bodies.py
```

当前可见的关键零件包括：

```text
Casing Base / Casing Top
Input Shaft / Transfer Shaft / Output Shaft
Transfer Gear / Output Gear
Output Key / Transfer Key
Thin washer
M6 Hub Bolt
M10 Casing Bolt / M10 Casing Nut
Hub Cover Input / Output / Small
Bearings
Oil Level Indicator / Breather Plug
```

资产名称、坐标和 socket 只能供 scene builder/evaluator 使用。模型输入中使用匿名、自然语言动作名。

### 1.4 已有研究计划和重要警告

| 路径 | 用途 |
|---|---|
| `causalassembly_icra2027/PILOT_12_PAIR_EXECUTION_PLAN.md` | 原始 CausalAssembly 12-pair 物理约束 pilot；复用 oracle、泄漏审计和统计原则 |
| `causalassembly_icra2027/README.md` | 当前论文中心和状态 |
| `causalassembly_icra2027/latex/main.tex` | 旧 manuscript，仅参考，不要先改论文 |
| `reason/hri_oracle_plan_audit_2026_08/final_report.md` | 解释为什么宽泛 HRI taxonomy 和 naive oracle delta 不足 |
| `docs/implementation_plan.md` | 旧 Stage-1 感知/STM 设计；确认哪些是规划、哪些已经实现 |
| `docs/tasks.md` | 旧任务清单；其中 GT 初始化不代表最终 sensor-only benchmark |

旧的 12-pair 计划不能原样删除。新的 pilot 在其基础上加入 task/self/partner confounding，但仍应复用：

- public/evaluator manifest 隔离；
- 匿名 domain ID；
- physics oracle 独立性；
- paired bootstrap；
- leakage audit；
- go/no-go gate。

---

## 2. 最终研究对象和明确删减

### 2.1 保留的四个潜变量

```text
G_T       task constraints / task feasibility
C_R       robot embodiment and capability
theta_H   stable human preference, skill, reliability, style
z_H(t)    current human intent, workload, availability, temporary state
```

模型观察：

```text
RGB-D or structured public observations
robot proprioception
human and robot action history
natural-language interaction
observable success/failure outcomes
```

模型选择：

```text
EXECUTE / DELEGATE / ASSIST / JOINT / WAIT / WARN / ASK / INSPECT
```

### 2.2 暂时删除

- Big Five 等宽泛心理人格建模；
- 泛化的 Theory of Mind claim；
- 每一步生成未来视频；
- 从头训练低层抓取或灵巧手；
- 用 PPO 背 gearbox 固定步骤；
- 把真实人类作为第一阶段唯一数据源；
- 把所有 failure 都解释成某个内部模块错误；
- 单一总分遮盖 success、time、preference 和 safety 的权衡。

### 2.3 中心可证伪假设

```text
H1: Strong embodied/VLM agents can often identify what the human did,
    but fail when identical early behavior has different latent causes.

H2: A factorized task/self/partner model reduces collaboration regret
    on held-out causal combinations compared with role classification
    and monolithic history prompting.

H3: Interaction history improves performance only when the system can
    update the correct latent factor rather than memorize a human ID.

H4: Active ASK/INSPECT helps only in cases that are observationally
    underdetermined, and its value must exceed its interaction cost.
```

---

## 3. 要创建的目录结构

```text
coadapt_assembly/
  README.md
  DEVIATIONS.md
  DECISIONS.md
  pyproject.toml
  config/
    actions.yaml
    pairs.yaml
    profiles.yaml
    robot_capabilities.yaml
    splits.yaml
    models.example.yaml
  schemas/
    public_episode.schema.json
    evaluator_episode.schema.json
    prediction.schema.json
    event.schema.json
  domains/
    public_manifest.jsonl
    evaluator_manifest.jsonl
  humans/
    profile.py
    policy.py
    state_machine.py
    language_wrapper.py
  environment/
    abstract_state.py
    high_level_executor.py
    isaac_scene_adapter.py
    physics_oracle.py
    observation_adapter.py
  pilot/
    generate_pairs.py
    generate_histories.py
    validate_pairs.py
    render_media.py
  agents/
    fixed_active.py
    reactive.py
    role_classifier.py
    direct_history_model.py
    factorized_bayes.py
    proposed_factorized_planner.py
    oracle.py
  runners/
    run_pilot.py
    run_baselines.py
    run_integrated.py
  evaluation/
    validate_predictions.py
    metrics.py
    statistics.py
    make_tables.py
    make_figures.py
  audits/
    leakage.py
    observation_equivalence.py
    manifest_separation.py
    oracle_stability.py
  tests/
    test_schemas.py
    test_manifest_separation.py
    test_pair_invariants.py
    test_metrics.py
    test_human_policy_reproducibility.py
    test_no_hidden_state_in_requests.py
  outputs/
    raw/
    normalized/
    metrics/
    figures/
  reports/
    repo_audit.md
    oracle_validation.md
    leakage_audit.md
    pilot_results.md
    go_no_go.md
```

不要修改旧 `pilot_12pair/` 约定或 manuscript 来伪装新 pilot 已经完成。

---

## 4. Action contract

统一 action JSON，所有高层 baseline 必须使用相同 schema：

```json
{
  "type": "EXECUTE|DELEGATE|ASSIST|JOINT|WAIT|WARN|ASK|INSPECT",
  "skill": "insert|place|bolt|close|hold|lift|inspect|test|none",
  "target": "anonymous_part_or_relation_id",
  "confidence": 0.0,
  "attribution": {
    "task_constraint": 0.25,
    "robot_capability": 0.25,
    "human_profile": 0.25,
    "human_transient_state": 0.25
  },
  "rationale": "optional and unscored"
}
```

要求：

- `confidence` 和 attribution probability 在 `[0,1]`；
- attribution 总和允许小数误差，validator 统一归一化或拒绝；
- parse failure 必须计入结果，不能无限重试；
- rationale 只用于错误分析，不按文风评分；
- 多个动作均可行时，evaluator 保存合法动作集合和每个动作成本。

---

## 5. Public/evaluator 数据隔离

### 5.1 Public episode 示例

```json
{
  "domain_id": "D-Q7K2",
  "history_budget": 2,
  "task_goal": "complete the current gearbox subassembly",
  "observations": ["media_or_public_event_paths"],
  "public_objects": ["part_1", "part_2", "casing"],
  "available_actions": ["EXECUTE", "DELEGATE", "ASSIST", "JOINT", "WAIT", "WARN", "ASK", "INSPECT"],
  "current_step": "anonymous_state_description"
}
```

### 5.2 Evaluator-only episode 示例

```json
{
  "domain_id": "D-Q7K2",
  "source_pair": "TP-01",
  "latent_cause": "task_constraint",
  "task_graph": "private_path",
  "robot_capability": "private_profile_id",
  "human_profile": "private_profile_id",
  "human_transient_state": "private_state_id",
  "valid_actions": ["WARN", "DELEGATE"],
  "action_costs": {"WARN": 0.2, "DELEGATE": 0.4, "EXECUTE": 5.0},
  "oracle_action": "WARN"
}
```

必须写自动测试，确保模型请求的序列化内容不包含：

```text
latent_cause
task_graph
robot_capability profile ID
human_profile ID
human_transient_state ID
oracle_action
action_costs
USD prim path
socket / plug names
HARD / COMMUTABLE / preference / incapable 等答案词
```

---

## 6. 12-pair pilot 的具体设计

每个 pair 包含两个 domain。两者前期公共观察和人类行为尽可能匹配，但隐藏原因不同，因而测试时的最优协作动作不同。

### Family A: Task constraint vs human preference

| ID | 共同观察历史 | Domain 1 | Domain 2 | 测试时应分化的行为 |
|---|---|---|---|---|
| TP-01 | 人类总是先放 thin washer 再装 output gear | gear 会遮住 washer seat，顺序是硬约束 | 有 side-loading washer slot，顺序可交换但人类偏好 washer-first | 几何突变后仍必须保持 vs 可按人类新选择调整 |
| TP-02 | 人类总是先放 output key 再装 output gear | gear 封闭 keyway | open-ended keyway 允许后插 key | 人类改变顺序时阻止 vs 接受 |
| TP-03 | 人类总是先检查 gear mesh 再关 casing | opaque casing 关闭后不可检查 | inspection window 关闭后仍可检查 | 关闭前必须提醒检查 vs 不必打断 |

### Family B: Robot incapability vs human preference

| ID | 共同观察历史 | Domain 1 | Domain 2 | Counterfactual test |
|---|---|---|---|---|
| RP-01 | 人类一直拧 M10 casing bolt | robot gripper/tool geometry 无法进入 | robot 可执行，但人类偏爱 bolting | 人类临时忙碌：请求/等待人类 vs robot 接管 |
| RP-02 | 人类一直搬 casing top | robot payload/grasp 限制，单独不可行 | robot 可搬，但人类偏爱该步骤 | 人类疲劳或不可用：JOINT/求助 vs robot 自行执行 |
| RP-03 | 人类一直插 output gear | robot 低摩擦 grasp 在该零件上不可靠 | robot grasp 可靠，人类只是偏好该步骤 | 时间压力下交给人类 vs robot 并行接管 |

### Family C: Human preference vs human capability

| ID | 共同观察历史 | Domain 1 | Domain 2 | Counterfactual test |
|---|---|---|---|---|
| HC-01 | 人类持续避开 bolting | 不喜欢但能够稳定完成 | 对齐能力差且高错误率 | robot 忙碌时可 delegate vs 应等待/辅助而非盲目 delegate |
| HC-02 | 人类只使用固定 gear 顺序 | 稳定程序偏好 | 不会可靠执行替代合法顺序 | 新顺序更快时建议切换 vs 保持/辅助原顺序 |
| HC-03 | 人类总在 inspection step 停顿 | 谨慎偏好 | 视觉判断能力不足，需要 robot 检查 | 可让人类自行确认 vs robot 必须 ASSIST/INSPECT |

### Family D: Stable profile vs transient state

| ID | 共同观察历史 | Domain 1 | Domain 2 | Counterfactual test |
|---|---|---|---|---|
| ST-01 | 人类当前避开 heavy part | 长期不愿搬重物 | 本轮临时疲劳 | 跨 episode 持续分工 vs 状态恢复后重新评估 |
| ST-02 | 人类当前在 shaft insertion 上很慢 | 稳定低技能 | 当前零件姿态异常造成一次性延迟 | 永久接管 vs 修复状态后继续让人类执行 |
| ST-03 | 人类从主动变为等待 | 稳定 supportive style | 临时更改下一步意图 | 持续由 robot lead vs 检测 intent shift 后重新分工 |

### Pair 合法性要求

1. 在需要归因的时间点之前，两域的公共 observation/event history 必须匹配到预注册容差。
2. 不要求视频像素完全相同，但不能用颜色、文件名、时长或语言泄漏标签。
3. 每个 pair 的 counterfactual test 必须让至少一个最优动作不同。
4. 如果两域所有合理动作都相同，该 pair 无法检验归因，必须重做。
5. 物理原因必须由独立 oracle 验证；偏好/能力/临时状态由参数化 human policy 产生。
6. 对模型而言，单次观察不足的场景允许 `ASK`/`INSPECT`，但必须计算代价。
7. 模型不能直接看到 pair 的另一成员。

---

## 7. Virtual human agent

不要一开始做完整 humanoid motion synthesis。第一版 human agent 是可复现的高层行为策略，并可用一个简单 avatar、ghost hand 或动作标记在 Isaac 中可视化。

### 7.1 Stable profile

```python
HumanProfile(
    task_preferences,
    order_preferences,
    per_skill_success_probability,
    per_skill_duration_distribution,
    initiative_style,
    communication_reliability,
)
```

### 7.2 Dynamic state

```python
HumanState(
    current_intent,
    current_subtask,
    workload,
    available,
    temporary_error_modifier,
    plan_revision,
)
```

### 7.3 Policy rules

- profile/state 加 seed 后必须完全可复现；
- 同一个 profile 在合理随机性下表现多样，但统计特性稳定；
- LLM 只能把已经确定的 action/intent 转成语言，不能改变 latent state 或 oracle action；
- 每个 event 写入 neutral public event 和 private evaluator event 两份日志；
- 提供 `scripted`, `stochastic`, `language_wrapped` 三种模式；pilot 的主结果使用 `scripted` 或受控 `stochastic`。

### 7.4 需要测试

- 同 seed 完全重现；
- 不同 seed 不改变 profile 的目标统计特性；
- profile label 不出现在 public log；
- mid-episode shift 只在指定 timestep 发生；
- 人类行为不是根据被评测模型身份改变。

---

## 8. 环境和 oracle 的实现顺序

### Track A: Abstract high-level evaluator

先实现纯 Python 高层状态机，以便快速验证 benchmark logic：

```text
state + action -> next state + observable outcome + cost
```

用途：

- schema 和 metric 单元测试；
- human policy 调试；
- baseline 接口调试；
- 验证每个 pair 的最优动作确实不同。

它不能支持“真实物理”论文结论，只是实现预检。

### Track B: Isaac physical validation

对 TP-01 到 TP-03 和 RP-01 到 RP-03 实现 Isaac scene adapter：

- 复用 `assets/parts/` 和 `assets/CAD_edit/`；
- 必要时增加机械上合理、可见的 slot/window/tool/collision proxy；
- 每次 action candidate 从相同初始状态独立执行；
- 保存 collision、contact、reachability、force、closure、visibility 和 task outcome；
- 每种物理条件至少 20 次小扰动；
- 预注册 success/failure threshold；
- 不能用手写 `if variant == HARD: fail` 代替物理结果。

`runtime/magic_assembly.py` 可以把成功动作 snap 到最终位姿，但物理 oracle 必须在 snap 前或独立轨迹中验证路径和约束。

### Track C: Sensor-only observation

物理和逻辑通过后，再添加模型输入：

- fixed camera 和 task-critical close view；
- RGB-D 或多帧 RGB；
- human/robot action event；
- 不含 prim/socket/variant 信息的语言；
- 最终集成时再加入 Franka proprioception。

---

## 9. 数据划分

不要按 episode 随机切分，否则相同 profile/geometry 会泄漏。

```text
Split S1: unseen human instance, seen factor combination
Split S2: held-out composition of seen preference/capability factors
Split S3: within-episode intent or reliability shift
Split S4: changed task geometry or robot capability
Split S5: second assembly product (full benchmark only)
```

历史预算：

```text
0 / 1 / 2 / 4 prior interactions with the same partner
```

绘制 performance-versus-history curve。禁止使用 human ID 作为模型输入。

---

## 10. Pilot baselines

按以下顺序实现，不要一开始接大模型：

1. **Fixed Active Robot**：只按固定 DAG 执行，不适应人类。
2. **Last-Action Reactive**：只看人类上一动作。
3. **Discrete Role Classifier**：把人分为少量 role 后选动作，模拟 HRIBench-style abstraction。
4. **Direct History Model**：给 VLM/LLM 完整公开历史，直接输出统一 action JSON。
5. **Factorized Bayesian Baseline**：显式维护 `G_T/C_R/theta_H/z_H(t)` 的概率。
6. **Proposed Factorized Planner**：belief update + constraint-aware allocation。
7. **Oracle**：获得 evaluator hidden variables，仅作上界。

完整 benchmark 后再加入：

- COOPERA-style long-term memory baseline；
- recurrent/meta-RL partner adaptation；
- LingBot-VA、pi0.5、GR00T 或 ACT integrated policy；
- real-human condition。

公平性要求：

- Track A/B 中所有 cognition baseline 使用同一 high-level executor；
- integrated VLA 单独报告，不能与 oracle-skill VLM 混成一个总榜；
- 每个 model 记录版本、日期、prompt、seed、采样方式、延迟和费用；
- API/model 不可用时明确记录，不得伪造结果。

---

## 11. 推荐方法

实现一个最小的 `Factorized Belief Planner`，不要先做大模型训练：

```text
Observation/Event Encoder
        |
        v
Belief over G_T, C_R, theta_H, z_H(t)
        |
        v
Feasible Action Set + Cost/Preference Model
        |
        v
EXECUTE / DELEGATE / ASSIST / JOINT / WAIT / WARN
        |
        +---- insufficient evidence ----> ASK / INSPECT
        |
        v
Observed outcome -> factor-specific belief update
```

第一版用规则/Bayesian update 即可。只有下面条件同时成立时才加入 RL：

1. passive factorized planner 在 underdetermined cases 明显失败；
2. ASK/INSPECT 有多个成本不同的选择；
3. RL 学的是 evidence/action selection，不是低层抓取；
4. RL 显著超过 greedy information gain、Thompson sampling、exact small-state Bayesian planning 和随机 probe；
5. 提升在 held-out composition 上存在，而不是只记住 12 个 pair。

---

## 12. 指标

### Primary behavioral metrics

```text
TaskSuccess
ValidActionRate
AllocationRegret = chosen_action_cost - oracle_action_cost
IrreversibleErrorRate
PreferenceViolationRate
HumanIdleTime
RobotIdleTime
AdaptationLatency after a registered shift
PersonalizationGain between history budgets 0 and 4
AULC over history budget
```

### Diagnostic metrics

```text
AttributionAccuracy
AttributionBrierScore
PairwiseCausalConsistency:
  both members of a matched pair receive the correct distinct decision
AskRate
UsefulAskRate
FalseInterventionRate
ErrorPreventionRate
```

统计要求：

- 以 pair/template 为 bootstrap 单位，不把同一 pair 的两个 domain 拆开；
- 报告 95% paired-bootstrap CI；
- action 可多解时以 valid action set 和 regret 评分；
- attribution 是辅助诊断，行为结果是主要证据；
- 分别报告四个 family，不能只给总平均；
- 记录 parse/API/oracle failure，不得丢弃失败 episode。

---

## 13. Ablation

至少运行：

```text
Full factorized model
- task constraint factor G_T
- robot capability factor C_R
- stable human profile theta_H
- dynamic human state z_H(t)
- interaction history
- ASK/INSPECT
shuffled history control
wrong-profile control
wrong-task-constraint control
```

如果移除某因子没有影响对应 family，不能声称模型在使用该因子。

---

## 14. 实施阶段和验收标准

### Phase 0: Repository audit

任务：

1. 读取第 1 节全部资源。
2. 运行 `python tools/check_setup.py`。
3. 确认 Docker、GPU、Isaac image、主场景和 gearbox assets。
4. 搜索已有 human agent、task graph、event log 和 evaluator；不要重复实现已经存在的模块。
5. 创建 `coadapt_assembly/reports/repo_audit.md`，记录存在、缺失和复用决定。
6. 记录 git 状态；若因 ownership 不能读取，记录问题，不要擅自改全局 git 配置。

验收：明确确认当前仓库没有现成 virtual-human benchmark，并列出复用路径。

### Phase 1: Schemas, abstract evaluator, tests

任务：

1. 创建目录结构和 schemas。
2. 实现 public/evaluator manifest separation。
3. 实现 action validator 和所有 metric。
4. 用手工 prediction 构造 deterministic sanity tests。
5. 实现 abstract state transition evaluator。

建议命令：

```bash
python -m pytest coadapt_assembly/tests -q
python -m coadapt_assembly.pilot.generate_pairs --abstract-only
python -m coadapt_assembly.pilot.validate_pairs --abstract-only
```

验收：测试全部通过；12 个 pair 的两成员有不同 oracle action；public requests 无隐藏字段。

### Phase 2: Virtual human

任务：

1. 实现 stable profile 和 dynamic state。
2. 为 12 pairs 生成匹配历史。
3. 实现 deterministic/stochastic human policy。
4. 实现 mid-episode shift。
5. 运行 reproducibility 和 observation-equivalence audit。

验收：同 seed 可重现；profile label 无泄漏；匹配历史通过预注册相似度检查。

### Phase 3: Three-pair physical smoke test

先只实现：

```text
TP-01 washer/gear
TP-02 key/gear
RP-01 bolt/gripper/tool access
```

任务：

1. 建立参数化 Isaac scene。
2. 独立执行所有 relevant candidate actions。
3. 每个 variant 运行至少 20 次扰动。
4. 保存 failure reason、collision/access/force evidence。
5. 渲染 neutral media。

验收：oracle outcome 稳定，匹配 pair 的差异由目标机制产生，人工可从允许观察中识别相关证据。

如果这三个 smoke pair 无法稳定实现，暂停，不扩展其余九个。

### Phase 4: Baseline falsification pilot

任务：

1. 先在 abstract track 跑 Fixed、Reactive、Role、Bayesian、Oracle。
2. 再在 3 个 physical smoke pairs 跑 Direct History Model。
3. 比较 history recall、attribution 和 action regret。
4. 运行 text-only、shuffled-history、wrong-profile 控制。

验收：生成 `reports/pilot_results.md`，包括逐 pair 错误，不只给平均数。

### Phase 5: Expand to all 12 pairs

只有 Phase 4 达到 GO gate 才做：

1. 实现其余九个 pair。
2. 每个物理 pair 进行稳定性验证。
3. 运行完整 baseline 和 ablation。
4. 冻结 manifest、prompt 和 media hash。
5. 生成表格、图和 paired CI。

### Phase 6: Second assembly product

只有 gearbox 上的 gap 成立才选择第二产品。优先选择能复用动作词汇但改变结构的产品，例如：

```text
wheel-hub assembly
casing-fastener assembly
simple shaft-bearing assembly
```

不能仅通过重命名 gearbox 零件形成“新任务”。必须改变 topology、工具或可行顺序。

### Phase 7: Integrated VLA and real humans

最后执行：

1. 将高层 subgoal 接入一个 VLA/skill policy；LingBot-VA 可作为候选 integrated baseline。
2. 单独报告 perception/planning/execution failure。
3. 设计 counterbalanced within-subject real-human study。
4. 用 power analysis 决定主研究人数；先做小规模可用性 pilot。
5. 真实人类条件至少覆盖稳定偏好、能力不匹配和 mid-task shift。
6. 报告 makespan、错误、错误干预、NASA-TLX、协作流畅度和信任。
7. 检查 simulation model ranking 是否预测 real-human ranking。

---

## 15. Go/No-Go gate

### Data-quality gate

以下全部通过：

- public/evaluator manifest 完全隔离；
- 12 pairs 在 abstract evaluator 中有正确且不同的最优动作；
- 物理 smoke pairs 的 intended outcome 在扰动下稳定；
- matched histories 不通过标签词、长度、颜色、文件名泄漏；
- 机械熟悉的 reviewer 能看到必要物理证据；
- model-facing observation 不需要隐藏 simulator state 才能回答。

### GO

至少满足：

1. 两个不同强 baseline 在 matched pairs 上明显低于 oracle；
2. 它们能复述 human action，但 PairwiseCausalConsistency 明显更低；
3. factorized baseline 相对 direct history/role classifier 降低 allocation regret；
4. 改错或打乱对应 factor 会定向破坏相关 family 的表现；
5. 结果在 part name、颜色、prompt 和 seed 控制后仍存在。

### REDESIGN

- 模型和人类都看不出物理差异；
- abstract evaluator 的 pair 实际上有相同最优动作；
- direct model 只靠文件名或语言模板分类；
- factorized method 的提升来自额外 oracle state；
- virtual human 的 profile 不稳定或不同 profile 行为不可区分。

### STOP/PIVOT

- 强 direct baselines 在 12 pairs 上接近 oracle，且 held-out composition 也稳定；
- 第二任务无法复现任何 gap；
- real-human behavior 与 scripted profile 完全不一致，且无法通过校准修复；
- 计算/工程成本远高于可获得证据。

不要移动阈值来保住 hypothesis。把负结果完整记录。

---

## 16. 最终产物清单

不得在以下内容完成前声称 pilot 完成：

- [ ] `repo_audit.md` 说明现有资源和缺口。
- [ ] 12 pairs / 24 domains 有匿名 public manifest。
- [ ] evaluator manifest 与 public manifest 自动隔离。
- [ ] abstract evaluator 验证每个 pair 的 oracle action 分化。
- [ ] virtual human profile/state 可复现且不泄漏。
- [ ] TP-01、TP-02、RP-01 通过 Isaac physical smoke validation。
- [ ] Fixed、Reactive、Role、Direct、Factorized、Oracle baseline 可运行。
- [ ] 所有 prediction 符合同一 schema。
- [ ] primary metrics、paired CI、逐 family 结果存在。
- [ ] leakage、observation equivalence、oracle stability audit 存在。
- [ ] 所有 parse/API/simulator failure 保留在 raw logs。
- [ ] `go_no_go.md` 明确给出 GO、REDESIGN 或 STOP。
- [ ] 报告明确说明这是 pilot，不夸大为完整 benchmark。

---

## 17. 执行时的汇报格式

每个阶段结束时向用户报告：

```text
Completed:
- 实际完成的文件/实验

Evidence:
- 测试、日志、数值或截图路径

Deviations:
- 与本文不同的地方及原因

Next gate:
- 下一阶段开始前必须满足的条件
```

遇到以下情况必须停下来询问用户，而不是自行扩大范围：

- 需要购买或使用付费模型 API；
- 需要新的私有 CAD 或真实人类数据；
- 需要改变研究中心 claim；
- 需要使用远程服务器或大规模 GPU；
- 需要删改现有 manuscript 或旧实验结果；
- 发现与该 benchmark 实质同构的新论文。

第一条实际执行指令是：

```text
从 Phase 0 开始，只做仓库审计和 Phase 1 的 schema/abstract evaluator。
不要先启动大模型、Isaac 批量渲染、RL、VLA 或真实人类实验。
```

