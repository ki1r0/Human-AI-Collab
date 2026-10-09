# HRC Harness 搭建记录

2026-10-08 scope 更新：当前默认 pipeline 仅到本地 help request outbox，随后安全停止；
不运行 helper/干预/恢复，也不为请求预留恢复 contact。以下完整闭环是 10-07
旧 scope 的历史 contract 测试，不代表现在运行的流程。当前说明与来源见
`reports/hrc_harness_pipeline_sources_20261008.md`。

依据：`HRC_Harness_M0_M1_Build_Plan_ZH.md`，2026-10-07。
任务：`Hub_Cover_Output_Top -> Casing_Top/socket_hub_output`。

## 当前结果

新增独立 `hrc_harness` 包，复用现有 Isaac 场景与图像/视频工具。
没有修改 REPAIR/VADER/control-policy 的执行逻辑，也没有将旧 runner
的真实零件位姿查询、precontact blocker 查询当作新 harness 的公共反馈。
工作区中那些模块的并行改动仍原样保留。

已实现公共观测与独立 GT 边界、append-only ledger、有限工具 registry、
每 tick 取消/保护、统一 contact/probe/help 预算、ownership/epoch、
求助后新观测三值验证、共享 multimodal proposer、经验 count estimator、
depth-2 selector，以及独立 reset/prefix replay 的 branch 采集接口。
不新增 hash、冻结 contract 或实验准入机制；未认证运动的限制用于安全边界。

这些是框架与接口完成情况，**不是物理 M0/M1 已验收，也不是严格论文复现结果**。

## 验证

| 检查 | 实测结果 | 边界 |
|---|---|---|
| `unittest discover` | 56 项通过，含 20 项新 harness 测试 | 软件不变量，不是物理成功率 |
| 既有 REPAIR function tests | 13 项通过 | 单独执行，unittest 不发现这些裸函数 |
| compileall / diff whitespace | 通过 | 不替代 runtime |
| nominal contract | finish，1 contact、0 help | 明确 synthetic |
| blocked contract | stall→help→fresh verification→resume→finish，2 contacts、1 help | 明确 synthetic |
| depth-2 算术例 | Q(probe)=0.572，direct=0.23，help=0.50 | toy numbers，仅检查奖励/成本不重复计算 |
| Isaac 5.1 / Lab 2.3 sensor smoke | 60 physics ticks，dt=0.01，sim=0.6s，退出码 0 | 没有抓取、坐合、probe 或模型推理 |

最新物理 smoke：`runs/harness/isaac_sensor_smoke_20261007_02/`。
保存三路 RGB、关节/TCP/接触观测、10Hz 视频、60 条私有物理采样。
runtime wall 约 4.08s，**不包括 Kit 启动与 scene 初始化**。
contact/probe/help 均为 0；`seat_gt` 与 `registration_gt` 为 UNKNOWN。
初始旧 camera buffer 时间未确认时明确标 stale，后续 camera/robot acquisition
时间分别保留，重复读取不会伪造新的传感器采样时间。

contract artifacts：`runs/harness/contract_nominal_20261007/` 与
`runs/harness/contract_blocked_20261007/`。这些不纳入真实 calibration。

## P0 资产发现

读 USD 的真实结果见 `reports/harness_asset_audit_20261007.json`：

- 两份资产：`metersPerUnit=0.01`、`upAxis=Y`、default prim `/World`。
- 两份资产均有 rigid body 与启用的 SDF collider；casing visual mesh 约 35 万 points。
- cover 的 `/World/plug_main` 为 `(0,0,-3.66859)`，不是旧假设中的 y 向偏移。
- casing 的 `socket_hub_output` 为 `(0,39.75,27.9)`；两者 frame 朝向都是单位旋转。
- collider 存在不证明孔畅通；frame 单位旋转不证明插入轴。

保留现有 spawn scale=0.002，未擅自再乘一次 metadata 的 0.01。
旧 combine 的 `(0,0.08683718,0.062)` 仅作为未认证名义先验。
插入轴与旋转对称性尚未认证，不据此给物理完成打 PASS。

## 与计划的差距

| 阶段 | 已有 | 尚需完成 |
|---|---|---|
| P0 | USD 单位/frame/body/collider 审计、真实 scene/sensor 启动 | 实测孔 clearance、插入轴、fixture frame、clocking/symmetry |
| P1 | 无零件真值纠正的 tick-stepped nominal Cartesian executor | 已批准硬限值、认证 safe hold/grasp/pick/prealign/seat/release profiles |
| P2 | contract M0 与 helper handover/fresh-verification | 真实 nominal 和 blocked→help→机器人完成回合；公共 visual 判断准确性 |
| P3 | XY/angle/speed whitelist、统一计费、卸载声明和取消接口 | 真实 probe 幅度/卸载轨迹/只改变一个因素的 pilot 认证 |
| P4 | branch replay/record、grouped split、fit、coverage/Brier、unsupported | 真实 branch 数据、prefix replay 一致性、动作排序验证、误差/不确定性校准 |
| P5 | 五种共享接口的 comparison policies、trace/summary | 批量物理实验、跨 seed/severity/OOD 比较、完整论文统计与报告 |

ACT 暂未注册为低层 backend；现有整条轨迹尚未验证为可中断 seating/probe。
RGB-D depth/tilt estimator 暂无，返回 null；HTTP 模型提供独立标注的视觉推断，
不能改写测量，也不等于已经过准确性验证。真实模型服务本次没有调用。
registration 当前 UNKNOWN，不将单步坐合等同于后续螺栓可装配。
没有真实经验表时，proposed 明确记录 unsupported fallback，而非伪造概率。

## 待用户确认

1. force、速度、workspace 与 handover 停稳硬限值的批准来源；确认前保持接触工具禁用。
2. 默认以 `seat_gt` 为 primary endpoint，孔位配准单独记录，是否接受？
3. 默认使用现有仿真 helper 的 scoped blocker clearing，还是先只使用 placeholder？

运行命令、工具 profile 格式、calibration 数据约束见 `hrc_harness/README.md`。
本次源码尚未提交/push；原工作区的并行修改不属于本次实现。
