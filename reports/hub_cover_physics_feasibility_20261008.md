# Hub Cover → Casing Top：独立物理可行性诊断

日期：2026-10-08，Asia/Singapore。**按用户确认的物理判据，Stage 1 通过；旧版固定 nominal-height 检查仍会显示 FAIL，但不再作为装配失败判据。Isaac 场景回放视频已生成。** 没有恢复机器人或策略搜索。

结论：这是动态刚体、碰撞开启、无机器人/吸附/运行时位姿覆盖的慢速力驱动插入；26 s 后执行器归零，重力和接触继续作用 6 s。撤力后 1441/1441 个采样都记录到 Hub↔Casing 接触，最后 1 s 的位置有限差分速度峰值仅 0.0024 mm/s，位置稳定。按“真实物理流程并自然接触、orientation/位置允许误差”的标准，物理落位通过。这里不是从高处完全自由落体，而是先慢速驱动到配合区，再撤掉执行器观察自然沉降。

**2.525 mm 的来源**：最终 Hub 根节点世界高度 `0.2645247 m`，减去配置里的 Casing 根高度 `0.200 m` 和旧 nominal 根节点高度 `0.062 m`，即 `+0.0025247 m`。它是相对配置中旧的 62 mm 名义根节点位置，不是相对最终接触点或独立验证的装配成功状态；你指出手动/Magic Assembly 的参考高度可能不准确是对的，因此不能据此判失败。输入网格的平面测量估计完整水平法兰贴合时根节点应为 61.837 mm，但这只是平面几何估计；当前部件有轻微倾斜且有实际接触支撑，不把该估计用作硬阈值。

尚未解决的是接触力/PhysX 速度遥测与逐帧位移不一致，故不把力值或 velocity API 峰值当成可信载荷/稳定性判据。实际位姿、持续接触和最后 1 s 的有限差分运动支持上述通过结论。

[Isaac 场景回放视频（323 帧，10 fps）](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/physics_trace_scene_replay.mp4) · [回放说明](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/physics_trace_scene_replay.json)

[指标视频：明确标注为数据回放，不是现场录像](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/telemetry_playback.mp4) · [指标总览截图](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/telemetry_summary.png) · [物理结果](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/result.json) · [完整 trace](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/trace.jsonl)

## 范围与方法

已停止机器人夹爪/策略参数搜索。本次场景没有机器人、抓取约束、magic assembly 或运行时零件位姿覆盖。Casing 是固定的运动学夹具；Hub 是开启重力的动态刚体。两者碰撞始终开启，原始 USD 文件未保存修改；所有碰撞配置覆盖仅存在于新测试场景。

代码：[diagnose_hub_cover_feasibility.py](../tools/diagnose_hub_cover_feasibility.py)。数据根目录：[hub_feasibility_20261008](../validation_logs/hub_feasibility_20261008/)。有效物理证据：`stage1_aligned_verified/`；`stage1_fabric_video/` 的物理结果与其一致，但两者的现场图像均未通过渲染同步审核。

测试协议：

- 240 Hz 物理步长；有效运动证据来自 7680 行物理 trace。之后按该 trace 在 Isaac 场景中离线渲染 323 帧、10 fps；此前双视角现场采集未能与 PhysX 状态同步。
- 先初始化 PhysX，再暂停时间线，设置一次已对齐的预插入初始位姿与零速度。恢复时间线时禁止额外推进物理；首步检查实际下降量和速度。
- 盖子质量 5.7 kg；重力 9.81 m/s²；沿世界 −Z 轴插入。
- 前 26 s 仅施加世界 Z 向力，无横向力或转矩：`F_external,z = mg + clip(m(-0.002-vz)/dt, -20, 20)`。因此执行器力在 35.917–75.917 N 之间，净轴向驱动力限制为 ±20 N；目标下降速度 2 mm/s。无关节或驱动约束。
- 26 s 后执行器力严格清零，保留重力与接触，观察 6 s。最后 0.5 s 检查接触与速度，而非只检查最终一帧的位置。
- 旧脚本另外计算了相对 nominal 根节点位置、orientation 和 PhysX velocity 阈值。这些是诊断数值，不是用户确认的成功定义；当前按“碰撞开启、动态运动、撤力后重力支持并稳定”判读，允许最终位姿和 orientation 有误差。
- 记录 PhysX 实际质量/惯量，不仅记录配置；录制前后直接比较 PhysX 位姿，检查渲染没有推进物理。

这些仅是此隔离测试的诊断标准，不替换任务的正式评分，也不代表完整 pick/transport/release 已通过。5.7 kg 是当前仿真采用的估计质量，不是实物标定值。

## 已观测的几何与碰撞问题

### 1. 命名 mating frame 并不是精确的孔中心

在 Casing 相对高度 52 mm 的输入三角网格截面，用 16 条射线拟合圆：

| 项目 | 测量值 |
|---|---:|
| Hub 插入边缘半径 | 99.9611 mm |
| Casing 孔半径 | 100.1273 mm |
| 半径差（不是直径差） | 0.1662 mm |
| 孔中心相对命名 socket frame 的 Y 偏差 | +0.7142 mm |
| 两个圆拟合 RMS 残差 | 12.2 / 9.3 µm |
| 未修正对齐时采样的最小径向间隙 | 约 −0.5463 mm |

因此按命名 frame 精确对齐并不等于按接触几何对齐。本次初始 XY 修正由截面拟合计算得到，不是参数扫描。此处是输入网格的局部截面测量，不能直接当成整个插入路径或 cooked SDF 的精度保证。

来源：`stage1_scene_sdf/geometry_audit.json` 的原始射线；`stage1_aligned_verified/configuration.json` 的 `section_circle_fit` 保存拟合方法、修正与残差。

命名 socket 的位置相对 Casing 为 `(0, 79.5000, 55.8000) mm`；当前 nominal seating root 为 `(0, 86.8372, 62.0000) mm`。仅让 plug/socket 原点重合会把盖子根节点放到 55.8000 mm，比现有 nominal seat 低 6.2000 mm。不能直接把这两个高度当成同一个成功定义；测试保留了两个 frame 的完整矩阵和两种高度的截面。

方向也不一致：该姿态下 socket frame 的 +Z 是世界 +Z，而 `plug_main` frame 的 +Z 是世界 +Y。盖子的几何薄轴/插入轴是其局部 Y，旋转后才对应世界 Z。因此不能未经校准就要求这两个命名 frame 完整 6-DoF 重合。

另对输入网格的水平面做了面积分组：盖子法兰底面相对根节点为 −5.941 mm，Casing 上表面为 +55.896 mm，水平面重合估计要求根节点相对高度为 61.837 mm，与旧 nominal 62 mm 只差 0.163 mm。因此最后高出约 2.5 mm 并非仅由 6.2 mm 的 frame 定义差异造成。[表面测量与方法](../validation_logs/hub_feasibility_20261008/geometry_surface_planes.json)。该估计不替代全周接触检查，倾斜后各处间隙不同。

### 2. 原始属性声明不足以保证 PhysX 使用 SDF

两个资产的网格都声明 `physics:approximation="sdf"`，但原始 applied schemas 没有 `PhysxSDFMeshCollisionAPI`。第一次按 authored 属性直接加载时，PhysX 明确报告动态 Hub 的三角网格回退到 convex hull。这一运行不是有效的凹形孔槽装配验证，不能用其停止位置判定实物不可装配。

后续隔离场景显式设置动态 Hub 为 SDF、固定 Casing 为合法的三角网格，沿用现有 M1 场景的 SDF resolution=256、margin=0、narrow-band thickness=0；没有禁用任何零件碰撞。

### 3. 接触数值配置仍需要审查

当前 Hub 最大尺寸约 260 mm，resolution=256 对应约 1.016 mm 的名义 SDF 网格间距，明显大于上述 0.166 mm 半径差。这个比较说明精度需要验证，并不单独证明 SDF 必然无法表达该配合。

另外，现有 `hrc_m1/roco_env.py` 将 SDF margin 和 narrow band 都设为零。NVIDIA 文档说明，margin 扩展的是 **SDF 距离查询域**，不是实体碰撞面的膨胀量；narrow band 决定表面附近高分辨率采样的范围，域外只有低分辨率采样。将二者归零不能简单解释为“取消实体膨胀”。这是接触数值异常的待验证嫌疑，而非已确认根因。[NVIDIA SDF schema 文档](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html)

## 运行有效性记录

| 目录 | 结果与用途 |
|---|---|
| `stage1_physical` | 无效对照：动态碰撞回退凸包；首轮相机近裁剪面过远，零件不可见；结果 JSON 从完整 trace 恢复。不得用于物理可行性结论。 |
| `stage1_scene_sdf` | 凹形接触对照：接近 nominal seat，但撤去执行器后速度与接触力振荡；初始化还含 reset warm-up 速度，不是最终协议证据。 |
| `stage1_geometry_aligned` | 几何中心对齐对照：能到达 seating 附近，但恢复时间线引入约 9 mm 的未记录初始下降，且末窗速度未通过；不能算最终协议通过。 |
| `stage1_aligned_clocked` | 摄像头初始化阶段卡住，未产生物理 trace；单列为基础设施失败，不是装配失败。 |
| `stage1_aligned_verified` | 时序检查通过、物理 trace 完整，但最终轴向误差 2.525 mm、末窗速度未通过。视频审核发现 USD 渲染没有跟随 PhysX，不能作为运动视频使用。 |
| `stage1_live_output_synced` | 直接发布 USD 输出的录制验证：0.1 s 时渲染变换一致性检查失败，中止；不是完整物理运行。 |
| `stage1_fabric_video` | 标准 Fabric 模式复核：物理结果重复，但截图仍几乎不变，现场视频无效。 |
| `stage1_zero_time_capture` | 按官方零时间推进采集方式尝试修复录制，但在初始 `rep.orchestrator.step()` 阻塞；没有开始新的物理 trace。已保存 Python 堆栈并停止进程。 |

没有运行新的机器人控制或自主策略实验。前述启动/记录问题均单独保留，没有挑选一个好看的视频冒充通过。

## 最终物理结果

有效记录包含 7680 个物理 tick（32 s），撤去执行器后的 1440 tick（6 s）施力均为零。首步下降 8.358 µm、速度 2.000 mm/s，初始接触为零；渲染前后的 PhysX 位姿变化为零。运行时 Hub 质量为 5.6999998 kg，惯量已保存；Casing 是固定夹具，其 runtime mass=1 不代表配置的自由动态质量。

| 测量 | 结果 |
|---|---:|
| 首次接触 | 16.333 s |
| 最大插入行程 / 目标行程 | 40.045 / 40.000 mm |
| 26 s 撤去执行器前，轴向残差 | +0.296 mm |
| 32 s 最终插入行程 | 37.475 mm |
| 最终根节点相对旧 62 mm nominal 的高度差 | +2.525 mm，仅作位置比较；不判失败 |
| 最终横向误差 / 姿态误差 | 0.133 mm / 1.488°，用户允许误差 |
| 最大 contact API 穿透估计 | 0.272 mm；不是独立 CAD 交叠测量 |
| 最后 0.5 s，PhysX velocity API 峰值 | 26.685 mm/s；与位姿差分不一致，不作通过/失败标准 |
| 同窗口角速度峰值 | 0.377 rad/s |
| 同窗口 XYZ 位置范围 | 1.58 / 2.41 / 10.82 µm |
| 执行器实际最大施力 | 69.560 N，未超过预设 75.917 N 总力上限 |
| contact API 报告的峰值 | 6538 N，**不可直接解释为真实载荷** |

最后位置波动很小，不能把速度通道的尖峰直接描述为肉眼可见的大幅振动。进一步复核发现，该窗口接触力均值约为 `(-1476,-1299,+160) N`，但由质量乘平均速度变化估计的净力约为 `(-0.09,-0.08,-0.32) N`；该场景只有重力、已归零的执行器和 Hub↔Casing 接触。这些量未形成可信的力学闭合，可能涉及接触求解/位置修正或力遥测语义，尚未定位。不能据此声称承受了 6.5 kN 的真实装配载荷。

按用户判据的分类：**Stage 1 物理落位 PASS**。撤去执行器后的最初几秒存在约 2.25 mm 的姿态/位置调整，之后稳定；末段接触点大多落在 Casing 顶面高度附近。接触力与速度遥测仍未校准，因此不据此估算实际载荷。该结果不代表机器人控制器、完整双臂任务或正式评分满分通过。

## 录像限制、交付与阶段状态

旧的 `physical_insertion.mp4` 仍是渲染同步失败的陈旧画面，不应作为运动证据。现在新增的 `physics_trace_scene_replay.mp4` 是在 Isaac 中用原始 USD 网格重绘的 32 s 物理轨迹回放：323 帧、960×720、10 fps，包含预插入、首次接触、撤去执行器和最终状态。它逐帧读取实际 trace 位姿，**不是现场并行录制，也不是第二次物理运行**；物理测试本身未 teleport，视频中的位姿回写只发生在离线渲染回放阶段。

另保留 `telemetry_playback.mp4` 作为指标动画。新的场景视频有 323 帧，解码抽查确认画面随时间改变；抽帧见 [开始](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/scene_replay_start.png)、[首次接触附近](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/scene_replay_first_contact.png)、[撤去执行器](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/scene_replay_actuator_removed.png)、[最终状态](../validation_logs/hub_feasibility_20261008/stage1_aligned_verified/scene_replay_final.png)。

录制阻塞证据：[Python 堆栈](../validation_logs/hub_feasibility_20261008/stage1_zero_time_capture/python_stacks.log) · [引擎日志](../validation_logs/hub_feasibility_20261008/zero_time_capture_engine.log)。尝试遵循了 NVIDIA 的 `delta_time=0.0` 采集方式，但这不能证明该调用在本地组合中正常工作。[官方采集示例](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_getting_started.html)

阶段 1：按用户物理判据 PASS；旧 nominal/velocity 诊断项保留为数值记录。阶段 2：NOT RUN。阶段 3：NOT ANALYZED。机器人仍按要求排除；没有启动策略搜索。结构化状态见 [stage_status.json](../validation_logs/hub_feasibility_20261008/stage_status.json)。

## 最小下一步（本轮不自动执行）

用户若希望继续完整诊断链，下一步才是确定性 Cartesian/IK 插入（Stage 2）；本轮没有执行。若要精确研究实际接触载荷，再单独校验 PhysX 接触/速度遥测；当前通过结论不依赖这些不一致的通道。

以上未执行新的 collider 参数扫描或政策实验。原始资产、生产控制器和正式评分规则均未因本诊断而修改。
