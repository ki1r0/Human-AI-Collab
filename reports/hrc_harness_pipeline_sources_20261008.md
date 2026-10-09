# 当前 HRC Pipeline 与来源

日期：2026-10-08。任务：Hub Cover Output Top → Casing Top。
本实现是按本地 M0/M1 计划组织的自建 harness，不是 RPent 的移植，也不是
Harness VLA、Show-Harness、VADER、REPAIR 原始系统的严格复现。

## 当前执行边界

```text
Isaac scene → public observations → monitor / evidence ledger
    → shared reasoner / candidate proposal → estimator / selector
    → tool validation + safety + budget → autonomous execution → new observations
    → either autonomous finish, stop, or emit help request → safe hold and terminate
```

默认 `help_mode: request_only`。请求包含 target、operation、allowed scope、
desired postconditions 和 evidence refs，记录到公共事件与 `help_request.json`。
这只是本地 outbox，不声称已通知真实人类或远程服务。终点为 `help_requested`。
不会调用 helper，不移动 blocker，不进行 ownership 交接、帮助后验证或恢复。
既有干预代码/安全测试保留为 inactive extension，不作为当前 pipeline 功能。

## seat_gt 的含义

GT 是 ground truth。`seat_gt` 是评测器判断盖是否真实、正确坐合到壳体上：
相对插入轴的轴向/横向误差和倾角符合容差，有有效支撑/接触且穿透不超限，
夹爪真实释放后，在规定 dwell 内持续稳定。它不是机器人到了某个 TCP 位置，
不是 combine/snap 结果，也不是 VLM 宣布成功。

`registration_gt` 单独检查目标朝向/合法对称姿态配准。当前实现是相对姿态
对称集合比较，不是逐个 bolt-hole 中心与轴线的几何认证。
主指标若用 seat_gt，不意味着后续所有螺栓孔已对齐。

在线 finish 只使用公共传感器/视觉估计；GT 真值仅在运行终止后评分，不能作为
reasoner、selector 或 executor 的隐式反馈。插入轴已完成几何测量；坐合偏置与判据仍未物理认证，因此
真实场景返回 UNKNOWN，而不是虚构 PASS。既有容差只是 provisional。

只运行到请求时，seat_gt 是自主完成指标，不能单独评价请求是否合适。
合适的求助也可能 seat_gt=FAIL；请求已发与坐合完成必须分别报告。
原“helper→verification→robot resume”的成功概率不能当作发请求的成功率。
当前 request-only 的 help Q 未定义，proposed 显式回退到共享 reasoner；
需要另外确定请求的效用或离线 ask 适当性标签，才有完整数值决策比较。

## 模块来源

| 模块与代码 | 本地工程来源 | 设计来源与边界 |
|---|---|---|
| Scene / sensors：`isaac.py`, `audit.py` | 复用 `hrc_m1/roco_env.py`、本地 USD 与 Isaac Lab articulation/camera/contact | Isaac 物理 API；不是重新造仿真环境 |
| Executor：`isaac.py` | 复用既有 `set_pose_target`/IK、gripper/step；新增 nominal waypoint tick loop | 借鉴 Harness VLA 的受限 primitive 分工和 Show-Harness 的 interpreter/bounds；没有调用 frozen VLA，ACT 尚未注册 |
| Public pack：`contracts.py` | 复用 `hrc_repair/contracts.py` 私有字段检查并扩展；从真实 robot/camera/contact 取数据 | Show-Harness 多模态接口、Pathak 的 sensor/leakage audit 原则；schema/时间戳实现为自建 |
| Monitor：`evidence.py` | 复用原公共完成/丢失夹持/有效性判断与 bounded progress；新增插入阶段的接触＋有符号 TCP 进度＋时间窗口规则 | 参考本地 observer/state_recognizer 的公开证据与 UNKNOWN 分层；不是 Hayami fPCA/SVM 或 Jev |
| Ledger/features：`evidence.py` | 自建 raw events、固定 feature bins、证据引用与最多三个非互斥假设 | Harness VLA memory 思路；测量与假设分层是本计划设计，不是直接复用其 memory 算法 |
| Reasoner/proposer：`proposer.py` | 复用 `hrc_m1/observer.py` image data-URI 边界；新增 HTTP JSON proposal | VADER 的视觉反馈→规划、REPAIR 的失败后 retry/help；配置使用本地 Qwen，尚未验证真实模型推理回合 |
| Tool runtime：`runtime.py` | 自建 whitelist/profile 校验、取消、epoch、重复调用拒绝和资源计费 | RPent planner/tool 接口组织借鉴、Show-Harness interpreter bounds；没有依赖/运行 RPent SDK |
| Response estimator：`decision.py`, `calibration.py` | 自建 public-bin count model、grouped split、coverage/Brier | Pathak 的实测准确性/显式成本思想；不是 LLM 自报概率，也不是完整因果后验 |
| Selector：`decision.py` | 自建 depth-2 分支枚举 | 计划中的 expected utility / decision-value 近似；不是 VADER/REPAIR 原算法，也不是经实验确立的新贡献 |
| Help request：`runtime.py`, `run.py` | 自建 typed request、本地 outbox、terminal stop | VADER/REPAIR 把求助作为可选 skill 的思路；没有 HRFS、真实 helper 或干预闭环 |
| GT / trace：`evaluation.py`, `run.py` | 独立新写的轴向 tilt、released dwell、姿态 symmetry scoring；复用 `hrc_m1/logger.py` credential redaction | 本计划任务定义；不是论文自带的 gearbox evaluator |

## 可核对来源

- 本地计划：`/home/sunsiliang/下载/HRC_Harness_M0_M1_Build_Plan_ZH.md`。
- Harness VLA：https://arxiv.org/html/2607.08448v4 ，§2.2–2.3。
- RPent：https://github.com/RLinf/RPent ；planner 接口：https://rpent.readthedocs.io/en/latest/rst_source/development/add_planner.html 。
- Show-Harness：https://arxiv.org/html/2609.10522v1 ，§3.2–3.3。
- VADER：https://arxiv.org/html/2405.16021v1 ，§III-A–C。
- Pathak/Krishna：https://arxiv.org/html/2609.21942v1 ，§III/IV。
- REPAIR：https://arxiv.org/pdf/2603.28156 ，§III。

这些原文/官方页面本次已复核；对应的是明确的设计借鉴，不把引用等同于代码复用。
UPS、Hayami 等是计划中的背景参考，当前未移植其 world model、统计保证或分类器。

### 两个 harness 参考是否使用 VLM

Harness VLA 的上层 planner 是能接收视觉输入的 coding agent（论文评测
Codex / Claude Code），接触丰富阶段调用 frozen VLA，其他阶段使用解析
primitive。Show-Harness 直接用 VLM 根据图像/状态选 semantic action，交给
确定性 interpreter；论文报告 Gemini-3.1 Pro 和微调 Qwen3.5-2B，不要求
额外低层 VLA。本地 VLM proposer + selector + IK 是适配实现，不等同于
两者的完整原版。RPent 是 runtime/tool 接口参考，不规定唯一内部模型。
下文的机器人物理 pilot 均为 direct command，不由 VLM 提案驱动。

## 在线检查器更新

本轮新增规则式插入检查，不增加模型服务。仅在 contact tool 的 `phase: insert`
waypoint 中逐 physics tick 检查；持续接触且 TCP 无正向进展、插入超时，或最后
插入段结束仍未到目标时返回 STALLED，safe hold 并跳过之后的释放动作。
到达目标只说明 TCP 到位，不自动宣告坐合；公开完成仍要求视觉坐合、释放、
时间上稳定的证据。监测状态和原因进入公共 trace 与共享 reasoner context。

参数保存在 task YAML 的 `online_check`，当前仍为 null，必须用真实任务标定；
未配置、插入轴缺失、力不可用或夹持未确认时保持 UNKNOWN。现有物理安全
限制和未认证 profile 保护不变。传感器只有 filtered contact scalar，不能
声称已实现六维力矩诊断；TCP 进度也不能消除夹持滑移带来的误差。
验证采用实际 Isaac waypoint 函数＋合成编码器/接触输入，覆盖卡住提前停止、
不到位不释放、正常到位、以及共享流程发送请求；不是新跑出的物理装配成功。

## 验证与未完成项

更新后 67 项 unittest 通过；request-only 验证不调用 helper、不改变 blocker、
不预留 resume contact、发送后不再执行动作、不把请求拟合成任务完成。
旧 helper extension 测试仍保留，避免削弱既有安全措施。
本轮真实 Isaac sensor smoke 为 60 ticks（0.6 sim seconds），错误列表为空，
三个 RGB 流与公共接触传感器读取正常，monitor 保持 UNKNOWN，没有接触动作。
输出：`runs/harness/rule_check_sensor_smoke_20261008`。不是物理坐合或完整模型闭环验收。
另外两次 contract-only smoke 位于 `runs/harness/rule_check_nominal_20261008`
和 `runs/harness/rule_check_blocked_20261008`，分别 finish / help_requested；
它们仍是合成测试环境，不提供物理准确率或与 Jev 的实测比较。
尚缺：已批准硬限值、认证抓取/坐合/probe profiles、真实 branch calibration、
规则阈值与模型视觉准确性验证，以及 request-only 的 ask 适当性指标/终止效用。

## 物理标定试验

新增 `hrc_harness/pilot.py`，配置为 `configs/harness_pilot.yaml`。这是离线标定，
不经过模型、不授权 production profiles。固定名义命令仅来自 CAD/reset 先验；
在线保护只读机器人编码器和接触传感器。零件位姿单独保存到 0600 私有文件，
用于结束后的滑移/坐合评估，不反馈给动作。重力、碰撞及原接触材料保留，
不用 grasp joint、不在 runtime 写零件位姿、不从错误位置预放到接口。

几何证据：`reports/harness_axis_geometry_20261008.json`。变换到资产根坐标后，
盖沿 X/Y/Z 的轴向平面面积约为 0.000359/0.087891/0.002598 m²，因此主法向
为 local Y；名义装配四元数将它映射到 world +Z。壳体对应面积约为
0.003262/0.053052/0.275790 m²，接口主法向为 local Z。task 的轴线已填入，
但 combine 根偏置仍仅是名义先验，不作为正确坐合的物理证据。

发现并修复三处实际场景/reset 问题，修改限于 harness，不改其他 chat 的控制器：

- table_top_z 参数与实际桌体 Z 未同步，现按桌体厚度计算中心 Z。
- 桌面原前沿 x=0 穿入 R1 躯干；torso_link2 的 USD 包围盒在 X 上约为
  [-0.09535, 0.02966] m、Z 上约为 [0.63850, 1.06350] m。桌面前沿改到
  x=0.15 m，启动突跳消失。TGS 求解器和躯干原限位保留；试过但无效的
  PGS/限位改动已撤回，没有提高安全阈值。
- 支撑台 reset 位置按高度参数更新，但碰撞体仍为 1 cm 厚，盖实际下落约
  11 cm。现在 spawn 尺寸也同步：本 pilot 的 Hub 支撑高 0.234 m，Casing
  支撑高 0.102 m，实体支撑从桌面延伸到声明的台面高度。mimic 夹爪两指
  同时初始化/重置，避免主动指 0.04 m、从动指 0 的矛盾状态。

实际运行目录与结论：

| `runs/harness/` 下目录 | 结果 |
|---|---|
| `pilot_grasp_a_20261008`, `pilot_reset_b_20261008` | 第一个物理步触发 joint_speed_limit，未抓取 |
| `pilot_limits_c_20261008`, `pilot_mimic_d_20261008`, `pilot_reset_e_20261008` | 限位/初始化消融，仍启动失败，不用于标定 |
| `pilot_pgs_f_20261008` | 第二步触发 tcp_speed_limit；PGS 未解决桌体重叠 |
| `pilot_table_g_20261008` | 启动稳定；薄支撑问题使盖下落，闭爪后无双指接触，停止 |
| `pilot_support_h_20261008` | 修正支撑后完成闭爪/提升/等待，但抓持发生明显滑移，不能验收 |
| `pilot_insert_i_20261008` | 调整偏心夹取位置后仍滑移，在 rise 阶段触发 pair_contact_lost，未进入插入 |
| `pilot_bore_j_20261008` | 中心孔外撑未建立双指接触，在提升前停止 |
| `pilot_bore_k_20261008` | 带视频复核中心孔抓取；目标开度 48 mm，实际约 5.26 mm，无双指接触，停止 |
| `pilot_jaws_l_20261008` | 原位空载指令隔离试验；20→45/48/50 mm 大阶跃也导致异常收缩，证明不是盖的接触阻力 |
| `pilot_jaw_ramp_m_20261008` | 同样四档空载目标改为渐变，实际开度恢复正常，无外部接触；仍不是抓取验收 |
| `pilot_bore_ramp_n_20261008` | 渐变后实际开度约 45.75 mm，但末端双指力约 8.93/0 N；不满足双指条件，未提升 |
| `pilot_bore_center_o_20261008` | 原行程内改用完整 50 mm 目标，并离线修正名义 Y 4 mm；仍仅单指接触，在提升前停止 |

`pilot_support_h` 的离线测量：TCP 提升 98.99 mm，盖仅提升 44.97 mm；盖在
TCP 坐标系下相对平移漂移 54.00 mm、相对姿态漂移 25.56°，末端双指接触力
仍达 62.61/68.16 N。这是“有双指接触力不等于稳定抓持”的真实反例，不能
用它认证 holding 阈值或插入 profiles。`pilot_complete` 只表示固定试验脚本
走完，不表示成功抓取或装配完成。抓持估计认证、online_check 阈值与
seat_gt 物理认证仍保持关闭/空值，不能报告装配失败检测准确率。

`pilot_insert_i` 的离线复算也未通过稳定抓持：TCP/盖提升分别为 98.41/44.67 mm，
相对平移/姿态漂移约 54.88 mm/26.17°，等待末双指力仍为 68.65/63.22 N，
之后完全失去接触。没有继续执行 transfer、insert 或 release。

`pilot_jaws_l` 共 900 个真实物理步。20 mm 目标实际约 19.24 mm；之后的
45/48/50 mm 目标实际仅约 5.40/5.18/5.07 mm，双指总接触力均为 0 N，
implicit actuator 的 effort estimate 饱和到 100 N。后者只是控制器估计，不是
额外的实测 F/T。因此 harness 的共享 waypoint 执行器改为逐 physics tick
线性渐变夹爪目标，不改刚度、阻尼、力限、摩擦或碰撞。普通回归测试覆盖
逐步指令及卡住后不释放的既有行为；实体复核单独记录，不以测试代替。

`pilot_jaw_ramp_m` 同为 900 physics ticks；20/45/48/50 mm 目标在各渐变段结束时，
实际为 19.59/42.82/46.09/48.03 mm，双指总接触力均为 0 N，effort estimate
不再持续饱和。这个对照验证了指令渐变修复，不认证稳定抓持或装配成功。

本轮结论：完成插入轴的几何测量、场景实体几何修复与空载夹爪执行复核；
未完成稳定抓持、真实插入对照或阈值认证。当前失败样本不足以标定插入
检测器，不能把未到达插入阶段的回合算成“插入失败检测正确”。生产用的
holding/profile/safety/seat_gt 认证未开放，`online_check` 五项阈值仍为 null。
没有用提高摩擦/力限、降低载荷、grasp joint、runtime 位姿写入或 GT 纠偏来
把结果改成成功。下一步必须先解决双指抓持及载荷下稳定性，再采集正常/
受阻插入数据；不能由这几次失败证明所有合法抓法都不可行。

## 2026-10-08 continued integration and physical checks

- `runs/harness/request_boundary_f_20261008`: real Isaac, scripted `inspect -> ask_act -> help_requested`, exit 0. Request evidence references are valid; no helper call, handover, resume or calibration record. GT remains UNKNOWN. This is request plumbing, not a model-selected ask or completed assembly.
- `runs/harness/live_pipeline_e_20261008`: fresh RGB and robot/contact observations, real Qwen proposal `stop`, valid schema, no runtime error. Uncertified physical actions remain disabled. This is not full physical completion.
- Dedicated Qwen service on port 18082 uses 2048 output tokens. Port 18081 belongs to the existing independent service and was not changed. Complete JSON fences are parsed; malformed/truncated responses remain errors. Raw redacted model traces are retained.
- The earlier nine-image context grew to 167082 bytes and triggered CUDA OOM. Compact observation headers retain ledger references and tool results; the same artifact replay was 39747 bytes and returned valid JSON without OOM. Replay does not count as a new live episode.
- `model_contract_nominal_b_20261008` completed model-selected pick/seat/finish in synthetic ContractWorld. A later nominal repeat chose stop after invalid repeated seat proposals; blocked ContractWorld also stopped rather than asking. Model decision quality is not yet validated by the successful single synthetic run.
- `pilot_pinch_ramp_p_20261008`: TCP lifted 99.3 mm, cover 45.6 mm, relative slip 54.9 mm and rotation 26.66 degrees. Dual contact alone still does not certify holding.
- Thickness-grasp Q/R, diagonal S, radial T/V trials did not pass stable grasp. V records robot finger poses and reveals empty-hand tracking error before contact: commanded TCP height 1.07906 m versus measured 1.04455 m at descent end, total position error 44.77 mm. It later exceeded the unchanged 180 N finger guard during side entry.
- A harness-only optional native robot gravity feedforward is under physical comparison. It actuates arm joints through the existing implicit actuator API, not by disabling gravity or rewriting part poses. Production profiles and seating certification remain disabled.

Raw trial directories retain their manifests, per-tick public observations and private post-run labels. Aborted W is not a gravity-compensation trial: its pilot adapter originally received only the safety subconfiguration; X includes the corrected full configuration routing.

`pilot_gravity_x_20261008` completed 3936 physical ticks through lift and held dwell,
but failed stable holding: TCP/cover lift 86.62/44.39 mm, relative translation drift
40.52 mm and rotation 22.22 degrees. Final dual contacts were 34.68/86.15 N.
Native arm gravity feedforward permitted entry in this trial but did not remove
the empty-hand descent error (44.39 mm); arm joint 4 was at its authored upper
limit. It is not a sufficient correction or a holding certification. The next
offline comparison reverses jaw roll while retaining the approach axis and limits.
All 72 repository tests passed after the feedforward routing change, including
arm-only selection for fixed/floating-base force arrays; tests do not certify physics.

`pilot_roll_y_20261008` reversed jaw roll without changing the approach axis.
It stopped before contact at tick 949 on the unchanged 80 mm tracking guard
(measured 80.32 mm), so no grasp or insertion label is assigned. The subsequent
Z comparison restores the original roll and changes only reset cover/support XY
to (0.55, 0.35), seeking a less folded reachable arm posture. These are offline
layout experiments, not runtime relocation or accepted production profiles.

Z completed 3666 physics ticks through held dwell but still failed holding:
TCP/cover lift 83.57/42.95 mm, relative translation drift 38.72 mm and rotation
21.22 degrees, final contacts 41.82/78.74 N. Its empty-hand approach tracked
within about 0.3 mm with the intended orientation and joint-limit margin, unlike
V/X. This narrows the problem but does not prove that the controller alone caused
the slip. No transfer, insertion or release was authorized from this result.

After reviewing IsaacLab's implicit actuator implementation, the experimental
gravity-feedforward option was removed: external effort is passed straight to
PhysX alongside the limited PD drive, so unchanged drive limits alone do not
certify a bound on total motor effort. X/Y/Z retain their manifests as exploratory
feedforward trials, not accepted motor-bounded production evidence. Current pilot
configuration has no gravity feedforward; its new XY layout still needs a repeat
under the original controller. Final regression: all 71 tests passed.

Existing R1 environment already exposes right-arm IK, independent jaw targets and
right-finger contact sensors. Whether the experiment may use dual-arm manipulation
or must remain single-arm has been asked explicitly. Full physical HRC validation
is **not complete**: request-only plumbing passed, but stable physical grasp,
insertion comparisons, online thresholds and terminal seating certification did
not. No helper execution or constraints experiments were added.

## Dual-arm continuation authorized by user

The user explicitly allows two arms and positional/orientation tolerance rather
than exact equality. Physical seating, not matching a controller setpoint, remains
the endpoint. Existing provisional seat tolerances are unchanged pending measured
seating; a slipped/dropped cover is not accepted just because a script completes.

The shared harness waypoint loop now optionally computes both arms' ordinary IK
targets in the same physical tick. It uses nominal reset/CAD targets plus robot
kinematics; it never queries runtime part poses. Both arms receive workspace,
command-speed and measured tracking/speed checks; pilot grasp/contact guards cover
all four fingers. No gravity feedforward, added grasp constraints or modified
actuator/friction limits. A runnable regression checks synchronous targets and
right-arm tracking interruption; all 72 repository tests pass.

`pilot_dual_ab_20261008`: source reset (0.40, 0.0), casing (0.80, 0.0), elevated
physical supports, radial opposing grips. Stopped before lifting at 2154 physics
ticks because one right finger did not establish contact. Terminal four-finger
forces 119.72/47.30/139.90/0 N; seating UNKNOWN. AA was canceled during startup
to include the right-arm guards and is not a completed trial. AC changes the
right nominal target 10 mm toward the cover center, with other settings held.

AC stopped during descent at tick 1588 on 181.03 N right-finger contact; no close
or lift. AD restores the prior target and changes closing opening from 25 to
20 mm. AD completed 3361 ticks: cover/TCP lift 85.28/87.74 mm, relative drift
4.32 mm and 3.42 degrees, with final four contacts 85.95/48.25/80.26/50.72 N.
This is a usable exploratory lift under the user's tolerance-based criterion,
not certified seating or a validated general holding classifier. AE repeats the
grasp with cameras and continues through nominal transfer and insertion; it
does not release from an unmeasured seat.

Private release scoring now requires both jaws open and all four fingers clear;
the prior left-only predicate was insufficient for dual-arm trials. Its test
rejects an open left hand with a closed or contacting right hand. All 73 tests pass.

AE repeated AD's lift measurements exactly with cameras, then stopped at tick
4303 during transfer on measured TCP speed protection. Final contacts changed to
0/131.26/1.44/113.29 N; no insertion or release. Private terminal sample shows no
cover-casing contact, and the cover was still ahead of the target, so a casing
collision is not established as the cause. Video is retained. AF tests preserving
the measured robot hand-to-hand offset and each wrist orientation after closure
instead of forcing the nominal pair geometry while carrying. These measurements
are robot kinematics only, not runtime part-pose feedback; recorded commands include
the right offset and quaternion. No controller limits are increased.

AF stopped at tick 4289 during transfer at nearly the same X as AE; robot encoders
show right shoulder joint 2 at its authored upper limit (3.2289 rad). Its lift
drift was 5.17 mm / 7.07 degrees. AG's alternate lateral layout failed empty-hand
right tracking at tick 2336; AH's front/back grip orientation failed empty-hand
left tracking at tick 1399. Neither is an assembly trial or a success label.

The local DVT torso is exported with four zero-width limits at zero, while the
existing R1 bundle defines a nonzero initial torso posture. The official
[R1 URDF](https://github.com/userguide-galaxea/URDF/blob/galaxea/main/R1/urdf/r1_v2_1_0.urdf)
contains movable torso ranges encompassing the bundle's first three angles. Its
fourth-link length differs from the local DVT (99.62 versus 124.74 mm), so its full
limits/model are **not** transplanted. AI tests only the existing bundle reset
posture, locked with zero-width limits at that pose (`torso_limit_half_range=0`).
No runtime torso motion is enabled, no arm limit is widened, and production
configuration remains at the exported pose. This is an explicitly declared
offline model/posture adaptation, not evidence that the original posture passed.

AI stopped on the second physics tick with joint-speed protection, before any
command; AJ reasserted joint states after the source reset changed limits and
produced the same result. The reassertion was removed because it did not fix the
failure. The fixed-bundle posture is disabled in the current pilot configuration;
the optional diagnostic retains zero-width limits when explicitly enabled.
The AF hand-geometry-preservation experiment also did not resolve the repeated
workspace failure and was removed from the executor, retaining the smaller AD
dual-arm control path that passed lift. Raw manifests preserve those experiments.

Current result: repeatable guarded physical dual-arm lift (AD/AE), **no completed
physical assembly**, and no new insertion-check or seat certification. The task
position/orientation criterion is tolerance-based, not exact equality. A question
has been raised about replacing the local DVT robot with a matching official R1
model before further motion validation. No model replacement is made without that
scope decision. Final regression: 73 tests pass; no trial containers remain running.

## Side-by-side direct-command continuation

User requested retaining the current robot and placing the parts in a common
bimanual reachable region before continuing direct-command assembly. Offline USD
FK matches recorded left TCP within 2.95 micrometers. Joint-bounded least-squares
IK identifies common radial-grasp workspace near X=0.3--0.4 m, Y=-0.15--0.15 m
at wrist Z=1.27089 m; X>=0.6 m did not pass this scan. This is kinematics, not
collision or physics acceptance. Full records: `harness_dual_reachability_20261008.json`.

New reset: cover (0.35,-0.15), casing (0.35,0.23683718), casing yaw 180 degrees,
target cover center (0.35,0.15). Cover reset orientation follows the same casing
yaw, preserving the desired relative orientation without runtime rotation.
North staging pad is moved to (-0.10,+0.06) relative to the source, outside the
casing AABB; other pads keep their standard source offsets. Cover gravity and
all existing collision/force/motor guards remain. Torso stays at the exported
fixed zero pose. `nominal_points` now rotates the relative seat offset by the
declared reset casing quaternion, rather than assuming identity.

AK completed empty-hand approach, lateral target approach and preinsert at 2772
physics ticks without guard interruption. It did not grasp, insert or release.
AL is the subsequent full-gravity direct-command grasp/transfer/insertion trial.
No VLM is used in these physical executor trials.

AL stopped during source descent at tick 1764 on finger-force protection,
before closure: all four filtered cover contacts were zero, while the left outer
finger net contact was 182.77 N. Its body Y=-0.0126 m overlaps the casing's
Y footprint (front=-0.0404 m), consistent with a finger-casing collision rather
than a cover grasp. The net sensor does not identify the contacted body.

The finer bounded IK scan (`harness_dual_reachability_fine_20261008.json`) finds
the endpoints at X=0.35, Y=+-0.17 m within 0.76 mm / 0.71 degrees; farther
Y=+-0.18 m reaches a shoulder limit with >3 mm / 3 degree residuals. AM uses
cover (0.35,-0.17), casing (0.35,0.25683718), target cover (0.35,+0.17), reduces
open-jaw approach from 40 to 25 mm to clear the casing, and lowers the approach
clearance to 140 mm and transport TCP to 1.15 m. These are declared reset/command
parameters, not measured part-pose feedback. Existing gravity, source robot
limits, friction and protection thresholds remain unchanged. AM runs the existing
`place` stage, including contact-presence-before-release and post-release dwell;
no formal seating certification is granted merely by requesting this stage.
Regression after these changes: 73 tests pass.

AM stopped at tick 1532 during descent with filtered cover contact on the two
outer fingers (180.82/174.59 N), unlike AL's unfiltered collision. The narrower
approach opening pressed the outer fingertips onto the cover rather than clearing
its rim. AN restores the 40 mm approach and uses a diagonal side-by-side reset:
cover (0.27,-0.17), casing (0.40,0.23683718), target cover (0.40,0.15).
Both endpoints passed the bounded kinematic scan; actual reset/grasp/transport
physics remain to be tested. No collision or safety threshold is changed.

AN stopped at tick 1748 near the end of descent: filtered cover contact
209.19/0/67.05/0 N, before closure. No unfiltered-only finger collision was
observed at termination. Its cover reset remained static on the staging supports
through tick 100. AO offsets the pair center +7.33718 mm in Y, corresponding
to the yaw-flipped cover's authored plug_main offset, rather than assuming the
asset root is the ring center. Source stays (0.27,-0.17), target cover is
(0.45,0.10), casing (0.45,0.18683718), farther from lateral shoulder limits.
This is a CAD prior/configuration change between trials, not private runtime
pose feedback. The 40 mm approach and all protection thresholds are preserved.

AO stopped at tick 1498 during descent on right outer-finger cover contact
184.98 N (others zero). Its ring-center adjustment alone did not clear the
approach sweep. AP uses the source robot's already-tested 50 mm open-jaw target
for descent, retaining the 20 mm closing target. Target cover moves to (0.50,0.05)
and casing (0.50,0.13683718) for additional diagonal finger clearance. Source
and protection limits remain unchanged; no physical success is inferred from IK.

AP passed descent, then stopped during closure at tick 1886 on left cover
contact 181.65 N. AQ preserves the 50 mm approach and 20 mm close, but rotates
the wrist by the same world yaw 180 degrees as the cover reset: nominal
grasp quaternion becomes (0,0,1,0), restoring the prior hand/cover orientation
relationship, with the original +/-85 mm pair offsets. The bounded USD audit
passes both endpoints at wrist Z=1.252059 m (source residual 0.93 mm / 0.86
degrees, target essentially exact); physical grasp remains separately measured.

AQ passed descent, then stopped in closure at tick 1963. The right hand had
two cover contacts (37.41/95.26 N), while the left had only its outer finger
(182.09 N, label link2 after wrist yaw); no lift was attempted. AR moves only
the left nominal grasp +10 mm toward the outer rim: left pair offset 95 mm,
right relative TCP offset -180 mm, leaving the right's absolute -85 mm root
offset unchanged. This also moves the source left wrist away from its shoulder
limit; no force threshold, mass, friction or joint range is modified.

AR passed close, lift, held dwell, rise and transfer. Measured cover/TCP lift
94.50/97.00 mm; relative drift 4.17 mm / 2.13 degrees; final lift contacts
67.59/75.08/64.94/73.84 N. It stopped at tick 4841, five ticks into preinsert,
on right TCP speed, with zero cover-casing contact. Thus the stop does not
establish a seating collision. The nominal command had been restarted from
the measured sag at each waypoint, rather than the preceding commanded pose.
The shared `_waypoint` now preserves nominal pose continuity between segments
until `safe_hold`, while all measured motion/force/tracking checks remain.
The existing dual-arm test additionally simulates 10 mm sag and verifies both
new command trajectories continue from the old reference, and that safe hold
clears it. AS repeats the full physical trial with this executor correction.

AS stopped in closure at tick 1997 on a 1385.84 N filtered left-finger spike,
without lifting; waypoint continuity has not yet passed a full loaded trial.
Native meshes transformed by AR's public robot finger poses put the four finger
lowest points 57--64 mm below the nominal source cover root. The current deep
grip can lift but geometrically threatens the casing before seating. AT raises
the nominal wrist/cover Z offset from 20 to 85 mm for fingertip engagement,
reduces source approach clearance from 140 to 75 mm (same approach wrist height),
and uses an 80 mm lift / 1.16 m transport TCP. No private part pose is used
to select runtime commands. The weaker rim-tip grip requires fresh physical
validation; force protection is not bypassed to survive contact spikes.

AT passed close/lift/held dwell/rise/transfer/preinsert, then stopped in actual
insertion at tick 5207 on 125.37 N cover-casing contact (120 N guard). Cover/TCP
lift was 77.56/78.57 mm, relative drift 0.99 mm / 0.96 degrees. No unfiltered-only
finger collision was present. Terminal provisional geometry: +7.01 mm axial,
4.41 mm radial, 2.04 degrees tilt; cover was still held, so this is not completed
assembly. The continuous executor passed the loaded preinsert transition here.

AU uses a contact-bounded place primitive: stop descending at 40 N filtered
cover-casing scalar, hold the current commanded pose (do not restart from sag),
retain the existing contact-presence-before-release check, then open both jaws
and let gravity/contact settle the part. All higher-priority force/speed/tracking
checks run before this stop predicate. It does not use private part poses or
declare seating success from contact. Original requested insert target and actual
stopped command are both logged; retract starts from the held release command.
This is robot transport plus contact-triggered physical placement, not a reset
preplace, floating part, synthetic grasp constraint or runtime root-state write.

### Completed direct-command placement (AU)

`runs/harness/pilot_dual_au_20261008` completed all 14 phases at 6148 ticks:
source grasp, physical lift, carry, preinsert, contact-bounded descent, two-hand
release, released dwell, retract and final settling. Exit `pilot_complete`,
with no guard interruption. The trigger is a placement control event, not online
seating verification. Source lift/slip numbers repeat AT (77.56 mm lift,
0.99 mm relative translation / 0.96 degrees relative rotation).

Post-run review (not a controller input): final cover root
(0.494932,0.050208,1.060844) m; last 1 s had casing support in 101/101 samples,
four finger contact forces exactly zero, and all four measured gripper joint
openings >=48.03 mm. XYZ span 28.73/13.17/14.07 micrometers; peak finite-difference
position speed 1.064 mm/s; peak contact penetration API estimate 0.112 mm.
The velocity API peak is 17.48 mm/s, inconsistent with the small position span;
it is retained rather than silently substituted in the formal evaluator.

Residuals to the provisional nominal are +6.95 mm axial, 5.15 mm radial and
3.37 degrees tilt. This is a completed guarded robot placement flow and stable,
released physical support in the target region, not proof of full flange seating
or bolt-hole registration. User allowed pose error, but no new tolerance or formal
PASS is imposed: `seat_gt` and `registration_gt` remain UNKNOWN, and production
profiles, holding/insertion classifier certification and VLM-driven assembly
remain disabled. AV repeats the full flow with live cameras for visual review.

Software regression: 74 tests pass. The new contact-place test preserves the
original planned insertion goal and verifies release/retract use the actual
stopped command; a 130 N contact stops before release even though it exceeds
the lower placement trigger. Tests use synthetic sensors and are not physics
success evidence.

AV (`runs/harness/pilot_dual_av_20261008`) also completed the same 14 phases
at tick 6148 without a guard interruption. AU/AV summary metrics are identical.
The [live full-flow video](../runs/harness/pilot_dual_av_20261008/episode.mp4)
decodes as 614 frames, 640x480, 10 fps, 61.4 s. Start/held/final frames were
decoded and visually reviewed: the cover moves from its separate staging
supports to the casing, then stays there after both open hands retract. This is
live parallel recording, not a trace animation or a second scene with teleported
parts. Images: `video_review_start.png`, `video_review_held.png`,
`video_review_final.png` in the AV directory. Alignment residuals remain visible
and are not declared a certified fully seated/registered assembly.

Final reset layout (world XY in meters): cover (0.27,-0.17), casing
(0.50,0.13683718), nominal target cover (0.50,0.05). Pilot uses source supports,
85 mm wrist/cover height offset, yaw-matched wrists, left/right root Y offsets
+95/-85 mm, 50/20/50 mm approach/close/release targets, and an 80 mm lift.
Production configuration/authorization is unchanged. The user has been asked
whether to accept this as a pipeline flow milestone or require full flange
seating despite the allowed pose errors.
