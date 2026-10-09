# Bolt Insertion Harness 状态报告

日期：2026-10-10

## 范围与结论

本文记录仓库内独立的 `runtime/bolt_harness` bolt-insertion 工作：Franka Panda 在 Isaac Lab / Factory 物理仿真中拾取 M6 bolt、运输、插入并验证。它不是 VADER 或 REPAIR 的复现，也不是完整 HRC M0/M1 系统；没有实际 helper 执行。当前 Cosmos 配对只证明一次 S0 通过、一次 S2 失败，不能据此宣称鲁棒性或 benchmark 成绩。

依据：[bolt-insertion progress](../artifacts/bolt_insertion/progress.md)、真实配对运行的 [S0 summary](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s0/episode/run_summary.json)、[S0 public trace](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s0/episode/public_trace.jsonl)、[S0 video](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s0/episode/episode.mp4)、[S2 summary](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s2/episode/run_summary.json)、[S2 public trace](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s2/episode/public_trace.jsonl) 和 [S2 video](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s2/episode/episode.mp4)。本报告未读取 private evaluator/native physics traces。

## 系统与信息边界

- **模型是高层工具选择器，不是端到端控制器。** Cosmos-Reason2-8B 每轮返回一个 JSON tool call。工具包括 `observe`、`check_task`、`send_help_request`、`stop`，以及 `execute_skill`、`nudge`、`retract`；skill 名称、模式、目标和动作幅度受 schema 限制。[system prompt](../runtime/bolt_harness/agent.py#L46) [tool schema](../runtime/bolt_harness/agent.py#L80) [harness loop](../runtime/bolt_harness/harness.py#L108)
- **运动由固定坐标 waypoint 与手写反馈代码执行。** motion plan 提供 pick/transport/preinsert/seat/retract TCP targets；每步读取当前 TCP pose，计算受限的 6D delta，再交给 Factory step/controller。模型不输出关节指令或连续轨迹。[waypoints](../runtime/bolt_harness/executor.py#L216) [executor](../runtime/bolt_harness/executor.py#L257) [feedback delta](../runtime/bolt_harness/executor.py#L1088) [Factory step](../runtime/bolt_harness/executor.py#L1123)
- **物理环境是 Factory 仿真。** bolt environment 继承 Isaac Lab `FactoryEnv`，配置继承 `FactoryEnvCfg`，并引用仓库 Panda 与 CAD 零件资产；“物理”指仿真物理，不是实机实验。[env.py](../runtime/bolt_harness/env.py#L32) [env_cfg.py](../runtime/bolt_harness/env_cfg.py#L1) [asset paths](../runtime/bolt_harness/env_cfg.py#L22)
- **prompt 明确给了名义顺序。** 系统指令要求 `pick → transport → insert_and_seat → check_task → (release_authorized=true 时) release_and_retract → check_task`，并说明 `stalled`/`force_limit` 不会自动触发 help，是否求助由模型选择。parser 对 schema 和 release authorization 有校验，但 prompt 顺序不等同于完整序列规划器。[prompt](../runtime/bolt_harness/agent.py#L46) [release validation](../runtime/bolt_harness/agent.py#L313)
- **模型输入不止 RGB。** 每轮包括当前 task RGB、关节位置/速度、TCP pose/速度、wrench、gripper width、双指接触、task spec、tool history 和 progress。[public observation](../runtime/bolt_harness/episode.py#L480) [agent projection](../runtime/bolt_harness/agent.py#L165)
- **在线 evaluator 布尔值会反馈给模型，但这本身不是信息边界违规。** pre-release `held` 读取 `latest_measurement.physically_held`，并与 evaluator 的 `seat_ready` 合成 `release_authorized`；release 后反馈 `task_success`。权威执行计划 §7.2 明确允许 VLM 接收当前装配完成状态及释放后的最终结果，也允许关节/TCP/校准力觉等公开状态；§3.1 明确采用解析抓取/搬运与 Factory 柔顺或阻抗插入的混合路线。因此固定 waypoint/解析控制与受限完成布尔符合计划设计，不是纯视觉自主判断，也不应单凭此判为违规。仍需在 P7 核对真实 payload 没有泄露禁止的 GT pose、内部几何、evaluator 分项或隐藏原因。public trace 只暴露 phase-labeled bool。[check callback](../runtime/bolt_harness/episode.py#L546) [Boolean projection](../runtime/bolt_harness/harness.py#L213)
- **视频不含 HTTP 等待时间。** harness 在模型推理时暂停环境，视频按仿真相机帧记录。[serial loop](../runtime/bolt_harness/harness.py#L35) [camera recording](../runtime/bolt_harness/episode.py#L446) S0 summary 记载仿真时间约 67.33 s、墙钟约 239.54 s、772 帧；S2 约 70.77 s、241.88 s、814 帧。按 progress 记录的 12 fps 解码，视频播放时长约 64.3 s 和 67.8 s；不能把播放时长当作端到端响应延迟。

## 配对结果

使用同一 Cosmos-Reason2-8B 服务。每个模型 response 对应一次 harness tool call。

| 条件 | 实际选择序列 | 结果 |
|---|---|---|
| S0 | `pick → transport → insert_and_seat → check_task(true) → release_and_retract → check_task(true)`；6 requests/responses/calls | `success`, exit 0, `task_success=true`, `help_requested=false`；视频 772 帧。见 [summary](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s0/episode/run_summary.json) 与 [trace](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s0/episode/public_trace.jsonl)。 |
| S2 | `pick → transport → insert_and_seat(stalled) → check_task(false) → insert_and_seat`；5 requests/responses/calls | 第二次插入返回 `invalid_action`；`failed`, exit 1, `task_success=false`, `help_requested=false`；视频 814 帧。public trace 没有给出更具体的 invalid-action 原因。见 [summary](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s2/episode/run_summary.json) 与 [trace](../artifacts/bolt_insertion/development_owned_20261009_cosmos8b_s0s2_pair_01/run_s2/episode/public_trace.jsonl)。 |

S2 的 `send_help_request` 是可选终止工具，模型没有选择它；两次配对都没有实际 help request、helper 执行或 human receipt。[help dispatch](../runtime/bolt_harness/harness.py#L147) [local persistence](../runtime/bolt_harness/episode.py#L499)

配对运行早于 executor-history logging 和 help-example ref 的后续修订。旧 S2 prompt 示例引用 `observation:task_rgb_camera-00000290`；该 ID 在当时提供给模型的 pick history 中，前次审计确认它是有效的历史 public ref，不能把配对描述成 invalid-ref 故障。当前代码优先用可用的当前 observation ref 生成示例；这项后续代码变化没有通过新配对运行验证。[current example builder](../runtime/bolt_harness/agent.py#L481) [current observation preference](../runtime/bolt_harness/agent.py#L597)

## P0–P7 状态

阶段定义依据权威本地计划 `/home/sunsiliang/桌面/BOLT_INSERTION_HARNESS_EXECUTION_PLAN_CN.md` §11；执行器路线与输入权限分别见 §3.1、§7.2。以下按原阶段原义评估本仓库证据；阶段完成仅表示该阶段要求的开发证据，不扩展成 VADER/REPAIR 复现或论文级验证。

| 阶段 | 状态 | 依据与未完成项 |
|---|---|---|
| P0 环境定位 | 已完成基础定位并可运行 | 仓库内可定位 Isaac Lab/Factory 环境、Panda/CAD parts、Cosmos 模型配置及配对产物；容器 AppLauncher 初始化和实际 S0/S2 运行均有记录。 |
| P1 步骤选择/几何 | 部分完成 | 已选择 bolt insertion，代码有 bolt/cover/socket 对应、名义 TCP waypoint 和尺寸配置；现有配对不等于完整核验实例映射、截面与碰撞图，几何证据仍需复核。 |
| P2 自定义环境 | 已实现并在目标场景运行 | Factory 派生单环境、动态 bolt、Panda/CAD 与力觉路径已用于真实仿真 episode；当前结论限于该配置，不外推到其他场景。 |
| P3 evaluator | 已实现并有配对正/负结果，完整性仍待核 | S0 pre-release/最终完成为 true，S2 seat-ready 为 false；这证明 evaluator 在两次运行中参与判定，但尚不能据此声称计划列出的关键负例/正例集合全部验收。 |
| P4 真实拾取/搬运 | 已演示 | S0、S2 都从未抓持状态开始，summary 记录 physical cap grasp；两条 public trace 中 pick 与 transport 均完成。 |
| P5 真实接触插入 | 名义 S0 已完成一次，泛化未验证 | S0 在保持夹持时 seat-ready，随后 release/retract 并最终成功；S2 插入 stalled、检查为 false，重试 invalid_action。不能把单次成功说成稳定成功率。 |
| P6 harness/model | 已集成并完成配对，当前修订待复跑 | 真实 Cosmos 请求、响应、工具动作及反馈均在配对 public trace 中；S0 六次、S2 五次。配对后 executor-history 与 help-example ref 修订没有重新接模型验证。 |
| P7 最终完整运行 | 当前版本待完成 | 有一条修订前 S0 从未抓持 bolt 开始、真实模型调用、全流程成功及视频/trace/config 的证据；但当前版本仍需完成信息边界核对并重跑 S0。重跑通过后再推进 constraints；本报告没有启动该运行。 |

## 已有测试记录与限制

[progress 文件](../artifacts/bolt_insertion/progress.md)记录：focused executor-history tests 45 passed；host 全套 164 项中 161 passed、3 skipped、0 errors；Isaac Lab container suite 167 passed、0 skipped、外层 exit 0。这里引用既有记录，不代表本次重新运行；报告编写过程未启动仿真、模型服务或测试。配对之后的源码修订没有 paired rerun，因此 S0/S2 仍是修订前行为证据。

本次没有新增 gate、hash、冻结 contract 或 benchmark，也没有修改其他文件。
