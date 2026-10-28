 CAPE-Assembly Peg MVP 实施状态

日期：2026-10-26。状态：开发与物理标定中，未达到完整 MVP 验收。

## 2026-10-28 角速度修复的接入范围

- 已修复并验证的是`assembly-harness-controlled-transfer-20261023/adapters/isaac-runtime/generic_backend.py`的Panda native控制路径；其`launch_check.py`默认RUNTIME直接加载该正式共享后端，不只存在于诊断副本中。
- 修复包括夹爪更新覆盖机械臂插值目标、参考重置后位置/速度起点不一致，以及匹配高刚度的阻尼配置。保留原角速度、关节限位和力保护；没有用差分读数替代原始API读数来规避保护。
- 同前态接触对照中，原异常步API/差分角速度从4.9105/0.1510降至0.0613/0.0555rad/s；相关测试14项通过。TGS子步末速度与整步平均速度本就不要求完全相等，不能将全部差异都解释为缓存错误。
- **适用补丁已接入CAPE与xArm共用后端，物理回归仍待完成。** 本MVP通过`panda-unified-assembly-peg+hub-20261018/launch.py`加载该目录自己的`generic_backend.py`，并将源码复制到回合private目录。夹爪物理步更新现仅按joint_ids写夹爪，不再重写机械臂目标。该后端无controlled-transfer的native子步插值/速度前馈组合，不照搬不适用的参考重置代码；Panda高PD分支原本已用配套阻尼，OSC和xArm增益、控制律及安全判据保持不变。
- 新增实际源代码setter执行测试，覆盖Panda双指、xArm驱动关节和双臂索引；全套实验测试345通过、3跳过。物理适配器验证已启动，路径为实验目录下`runs/angular-patch-20261028/cape-osc-adapter`与`runs/angular-patch-20261028/xarm-adapter`。两项均非VLM装配rollout，不计任务成功；结果尚待核对。旧回合的源码快照不修改。
- 此结论不覆盖xArm真机，也不代表自主装配成功或MVP整体完成。完整证据：`/home/sunsiliang/robot-assembly-experiments/experiments/assembly-harness-controlled-transfer-20261023/diagnostics/angular-velocity-root-cause-20261028/README_CN.md`。

## 已实现

- `assembly_bench/peg_mvp.py`：与工具名称无关的私有过程 evaluator，八类 checkpoint、
  历史/终态分开、按秒计时、缺测和间断处理、可恢复重试、终态谓词上下界。
- 非特权控制来源必须持续成立；初始化无效、未标定或私有探针均不能生成正式成功。
- `assembly_bench.cli rescore-peg`：只用实际采样离线重评，输出 JSON、事件和 HTML 摘要，拒绝覆盖旧输出。
- `assembly_bench.cli evaluate-peg-mvp`：调用最新实验 `launch.py --method ours`，
  不是旧 `CurrentPipelinePolicy`。固定实例、独立 run、批量 CSV，不自动挑选最佳尝试。
- `configs/bench/tasks/peg_mvp.yaml`：M6 两点钟方向孔、Panda、0.001 比例及明确验收要求。
- `data/bench/peg_mvp/instances.json`：4 个开发实例，20 个独立实例，直立/平躺各10个测试实例。
- 最新实验后端 `--peg-mvp`：目标零件专属手指力、机器人-目标接触、相对运动、
  保守整手包围区域撤离证据，均只写私有记录，不改变 actor 的力反馈或安全上限。
- 后续启动保存并挂载本回合 scorer 源文件副本，避免运行中编辑影响已启动的 scorer。

当前仍是开发 evaluator，`calibrated=False`，不能将 candidate success 当作正式成绩。
主轨先用 Panda；xArm 的兼容性问题不与这个 MVP 的有效任务成绩混合。

## 真实物理证据

实验根目录：
`/home/sunsiliang/robot-assembly-experiments/experiments/panda-unified-assembly-peg+hub-20261018`

- `runs/cape-peg-mvp-20261026/calibration-seat-01`：初始化座合探针完成。
  约12mm进入、帽底接近零间隙，但后续支撑报告空；不是机器人装配成功。
- `calibration-contact-source-02`：解析刚体配置证实 dynamic、重力开启、sleep threshold=0；
  原始接触矩阵和详细报告依然为空。没有通过放宽阈值解决。
- `calibration-no-stabilization-03`：仅关闭数值稳定化的对照也出现空支撑报告；未推广该设置。
- `calibration-cpu-contact-04`：CPU 碰撞后端也出现相同现象，尚不能归因于 GPU。
- `calibration-load-response-05`：该对照已作废。随后发现`advance()`内的局部
  `diagnostic`变量遮蔽了外层探针类型，导致请求载荷没有实际施加；原记录中的载荷字段
  只是请求值，不能当作实测/已施加证据。已改为从实际写入的外力缓冲状态记录，并补测试。
- `calibration-native-contact-06`：原生睡眠查询返回false，原生报告也为空，不能再简单归因于睡眠。
- `calibration-bearing-surface-07`：增加真实碰撞形状的帽底环面射线查询；
  仅记录几何证据，不覆盖原力报告，不据此自动认证标定。
- `calibration-query-control-08`：桌面查询正对照可命中；随后发现回调命中字段误写为
  `collider`，实际安装的API使用`collision`。因此07/08的零命中不能证明没有夹具表面。
- `calibration-query-api-09`：修正命中字段后的真实复测；另补回调错误捕获，避免C++
  回调异常被误解释为正常零命中。未因此自动改变座合成功判据。
- `ours-dev-attempt01/dev-u0`：最新 ours 的非特权开发回合，模型已选择抓取候选并开始物理运动；
  约234秒仿真时间时，目标专属双指力约1.89/1.90N且离开初始支撑高度。
  尚未确认完成搬运/插入/释放，不计入独立验收集。

当前控制器到抓取即已超过最初计划草案的180秒，因此开发运行上限暂定600秒仿真时间；
正式预算仍待物理参考轨迹的耗时分布确认，再对所有同轨方法统一使用。
不能把旧180秒草案与新600秒开发尝试直接作公平成绩比较。

第一批开发回合期间 scorer 仍在迭代，不能把它们称为固定版本正式比较。
已有重评报告：`reports/cape_peg_mvp_20261026/seat01/episode_report.html`。
目标映射离线核对：私有目标投影约(468.75,391.66)，与模型选择的孔口像素一致；
该次视觉深度推定的入口平面与私有CAD平面有差异，后续需核查几何采样，不能由投影吻合就声称深度正确。

## 验收要求与剩余

物理校准起步10条正例、20条反例，另留10条独立验收回合；有限验收集不等于永不误判的保证。
错孔、悬空、假插入、机器人托住等反例不得被接受；合理不同完成姿态、松手后正常落座、
失败后恢复不得仅因旧 held-seat/控制器停留顺序被误拒。

ours 至少完成直立和平躺各一个完整物理回合，再按固定实例全量运行、报告样本数与真实成功率。
不通过改变 prompt/evaluator 迎合测试集，不将 UNKNOWN 记成成功。

剩余：解决支撑测量；补齐完整物理正负例和独立验收；验证初态有效性覆盖；
完善关键帧/连续量曲线报告；接入一个可替换 policy 的端到端验证。
当前新增 MVP/原 bench 定向测试42项通过（另4个subtests）；实验后端最新全量264项通过、3项跳过。
其中能力描述及相机启动修复定向8项通过。
这些测试不替代物理装配或 evaluator 标定。

## 后续迭代证据

- 首个ours开发回合结束于角速度保护：49,831物理步附近、约415秒仿真时间。
  自动重评记录控制零件约225秒、取起保持约233秒、搬运到邻域约346秒；
  从未确认孔口对准/有效进入/座合/释放，未计成功。
- 从公开RGB-D定位到自适应ROI误绑定：所选像素(469,392)曾返回约37像素外另一孔。
  已增加选点必须属于当前椭圆的检查，以及小孔的半像素量化残差处理。
  真实回放返回约(468.80,391.77)，入口平面约0.745898m；函数未使用GT。
- `osc-controller-validation-10`因新增诊断变量遮蔽错误失败；保留。修复后的
  `osc-controller-validation-12`已返回`adapter_validation_complete, passed=true`。
  MVP后续ours默认使用已验证的官方OSC后端，不提升原安全阈值。首轮旧后端数据不混作同配置成绩。
- 真实碰撞表面查询已确认正例的八个承托环面采样命中cover，原力报告仍空。
  新分支明确标注`support_evidence=collider_geometry`，不把几何查询冒充力测量，
  校准状态仍pending，旧force分支和原报告保留。
- `endpoint-calibration-development-13`正在运行10正例/20反例，包含不同初始yaw、
  50微米以内横向扰动、1-3mm释放高度，以及偏孔和倒置反例；仅验证终态谓词，
  不能认证机器人抓取/搬运过程。独立endpoint验收使用另一组seed。

## 实际孔壁与第二次 ours 回合

- `calibration-bore-clearance-14`与`calibration-shaft-sweep-15`发现：旧统一圆孔
  近似对seed4给出约158微米穿透，而实际碰撞孔壁的采样包络偏差约25微米。
  连续球扫掠在原100微米数值容差下无casing命中，桌面阳性对照有命中。
  因此v0.3改用实际casing碰撞体扫掠判定通道，不再把旧半径近似当真实孔壁。
  cover通孔、头部承托、深度、倾角、释放、稳定和过程来源检查仍保留。
  原解析测量继续记录；缺少扫掠数据的历史回合不能补造该证据。
- `endpoint-calibration-development-13`跨开发版本运行，只用于排查，不作为统一版本正式成绩。
  部分预期正例在释放后倾倒；预期标签不等于最终物理真值，需逐项复核而不能直接称误拒。
- `endpoint-acceptance-collider-v03-16`使用另一组seed验证v0.3终态谓词。
  校准初始化、源码和原始测量单独保存，绝不记为机器人自主装配成功。
- `ours-dev-attempt02-osc-rimfix/dev-u0`在1800秒墙钟上限结束，39,564物理步、
  1,648视频帧。双指仍保持螺栓；最后TCP约(0.417,0.022,0.776)，
  搬运目标约(0.460,-0.070,0.800)，尚未完成插入。不是成功或evaluator误拒。
- 发现能力描述把旧关节驱动的0.0025rad步进、0.008rad跟踪限额错误报告给OSC。
  已按实际控制器报告不适用字段为null，并修正真实SE3参考速度为0.010m/s、0.060rad/s。
  这仅修复模型输入，不改变控制律、保护阈值或现有回合记录。
- `ours-dev-attempt03-horizontal-v03/dev-h0`为平躺初态的全物理开发回合。
  开始运行时的后端与评分器源码已保存在各回合private目录；不把途中代码修改算作已生效。
- 第三次尝试在模型动作前因Isaac首帧深度annotator返回空数组失败，不算策略失败。
  相机readiness现对原生camera.py该特定空数组异常最多8次纯渲染重试，不填假深度，
  不推进物理；其他RuntimeError仍立即抛出。第四次平躺尝试保存到
  `ours-dev-attempt04-horizontal-camera-retry/dev-h0`，已产生真实物理记录。
- 第二次回合搬运阶段的204个一秒采样中，187个姿态参考误差超过0.02rad暂停阈值；
  位置参考误差最大约1.25mm，未超过2mm暂停阈值。直接阻塞是等待姿态跟踪收敛，
  不是旧关节步进限额。尚需查明转动误差来源，未以放宽阈值代替修复。
- 报告新增`max_valid_entry_m`：只有通过选定cover孔和casing通道验证的轴向深度才计入。
  原`max_entry_m`保留但明确是全程轴向投影，远离孔口的投影不能被解释为插入进度。

当前仍未达到MVP验收标准：不得报告已完成两种初态的自主装配，或已通过全部物理标定。

## 可执行入口

```bash
python -m assembly_bench.cli evaluate-peg-mvp \
  --experiment-root /path/to/panda-unified-assembly-peg+hub-20261018 \
  --asset-root /path/to/Human-AI-Collab/assets \
  --split dev --limit 1 --gpu 3 --budget-s 1800 --output /new/output

python -m assembly_bench.cli rescore-peg --run /existing/run --output /new/score
```

需使用已配置相关依赖的 Python；本机使用实验目录 `.venv/bin/python`。
私有指标、原始接触和源码快照不得开放给 actor 文件工具。所有尝试与视频保留。
