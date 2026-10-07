# 原始 Gearbox USD 的物理装配验证

## 结论（当前迭代）

原来的 mutation counterpart 已删除。当前验证只针对 `assets/parts/` 中的原始部件、`assembly/instances/canonical_grouped.yaml` 定义的 Magic Assembly 顺序，以及 Isaac Sim 的真实碰撞开启 PhysX。

这次循环已经完成：

- 15 个可复用原始 USD 资产通过结构/碰撞资产审计；每个资产都有可见网格、碰撞网格和刚体质量。
- canonical sequence 的 49 个 `combine` 步骤已经逐 ID 独立展开，并在 49/49 个 collision-on PhysX 案例中通过；另有 2 个无 canonical ID 的齿轮—轴 fixture 通过。展开总数为 51 个物理案例，无遗漏、无未知 ID。
- 原来的 17/17 个 grouped interface 试验也仍然通过；逐 ID 结果汇总在 `validation_logs/physics_validation_canonical_summary.json`，每批原始日志保留在同目录。
- 产生了 Isaac Replicator 渲染的视频：`validation_logs/assembly_physics_rollout.mp4`（121 帧，960 个手动 PhysX step 的轨迹回放，960×640，24 fps）。对应数值 sidecar 是 `validation_logs/assembly_physics_rollout.json`。

这证明的是“每个 canonical 装配 ID 在校准后的单接口 collision-on 试验中可以到达稳定座位”。它还不是机器人执行的完整 73-step sequential dynamic rollout；机器人可达性、抓取、顺序累计误差、齿轮在旋转载荷下的啮合仍需单独验证。

## 检查—修复—再检查循环

### 1. 发现源几何问题

`tools/audit_bore_openings.py` 对 Casing Top/Base 的横截面做射线/三角形审计。原始 M6 孔的拟合内半径约为 2.81 asset units，而 M6 螺杆半径约为 2.94，说明源碰撞网格会在孔口发生干涉；因此不能只修改 Magic 偏移。

### 2. 修复原始 USD/STL

- `tools/repair_casing_m6_bores.py` 在实测 12 个 M6 孔中心的局部环带内，将目标内半径扩大到 3.25 asset units，并同步更新 Casing Top/Base 的 USD 和 STL。可见 casing 外形没有被整体缩放。
- `tools/repair_hub_cover_colliders.py` 将 Hub Cover Input/Output/Small 的碰撞近似改为 SDF，保持可见网格不变；凸分解无法表达的凹陷和安装凸台因此由 SDF 保留。
- `runtime/magic_assembly.py` 的 shaft、casing closure、oil indicator、M10 bolt/nut 座位改为 collision-on 试验测得的物理 rest pose；这不是通过关闭碰撞或事后 teleport 伪造成功。
- M6 使用实测的每孔轴心；M10 Casing Top 的负 Y 侧第 3/4 个 pocket 经 collision-on 复验确认为较浅台阶，因此其 Magic seat 从统一的 `-86` 修正为 `-71.5` asset units，避免把螺栓强行推进壳体实体。

### 3. 再检查

资产审计命令：

```bash
docker compose run --rm --no-deps --entrypoint bash hac -lc \
  'cd /workspace/Human-AI-Collab && tools/run_tool.sh tools/check_part_fidelity.py'
```

结果：所有 15 个资产 `ASSET PASS`；Transfer Gear 和 Output Gear 的孔—轴静态间隙分别为 0.500 asset units；所有碰撞网格的接触/恢复偏移合法。

canonical 覆盖命令：

```bash
python3 tools/check_physics_manifest.py
```

结果：

```text
canonical_combine_steps: 49
covered_step_references: 49
unique_covered_steps: 49
missing: []
unknown: []
duplicates: []
interface_trials: 17
```

逐 canonical ID 的独立复验命令（按批次串行运行，避免多个 Isaac Kit 同时占用 GPU）：

```bash
docker compose run --rm --no-deps --entrypoint bash hac -lc \
  'cd /workspace/Human-AI-Collab && tools/run_tool.sh tools/test_controlled_mating.py \
    --config assembly/physics_validation.json --expand-canonical \
    --start-index 0 --max-trials 8 --steps 960'
```

`--start-index/--max-trials` 依次覆盖 0、8、16、20、24、32、40、48；修复后汇总为：

```text
canonical_steps=49, observed_unique_steps=49, pass=49, fail=0,
missing=[], unknown=[], failed_final_ids=[]
```

旧批次中暴露的 M6 轴心偏移和 M10 浅 pocket 失败仍保留在日志中；汇总脚本按“后一次修复复验覆盖前一次失败”规则生成最终证据，避免删除失败历史。

Magic 的无物理序列一致性回归：`python3 tools/test_instance_playback.py`，20/20 通过；Isaac 中 `play_instance.py --variant canonical_grouped --pose correct`，73/73 步完成，`validate_state: ok=True`。

## 17 个 collision-on 物理接口

`assembly/physics_validation.json` 是可复现实验配置。每个 trial 使用原始 USD、目标 kinematic、源 dynamic、零重力、240 Hz、960 steps，并以轴向误差和横向漂移判定。

| 物理接口 | 覆盖的 canonical step | 结果 |
|---|---|---|
| Transfer Gear → Transfer Shaft | 预装 fixture（无 canonical combine） | PASS |
| Output Gear → Output Shaft | 预装 fixture（无 canonical combine） | PASS |
| Hub Cover Output → Casing Top | `inst_025` | PASS |
| Hub Cover Input → Casing Top | `inst_024` | PASS |
| Hub Cover Small → Casing Top | `inst_026` | PASS |
| 12 个 M6 Hub Bolt → Casing Top | `inst_027`–`inst_038` | PASS |
| Input Shaft → Casing Base | `inst_052` | PASS |
| Transfer Shaft → Casing Base | `inst_054` | PASS |
| Output Shaft → Casing Base | `inst_053` | PASS |
| Casing Top → Casing Base | `inst_055` | PASS |
| 6 个 M10 Casing Bolt → Casing Top | `inst_057`–`inst_062`（第 3/4 个 socket 使用 -71.5 asset-unit 实际 seat） | PASS |
| Breather Plug → Casing Base | `inst_071` | PASS |
| 两个 Oil Level Indicator → Casing Base | `inst_072`–`inst_073` | PASS |
| Hub Cover Output → Casing Base | `inst_005` | PASS |
| 两个 Hub Cover Small → Casing Base | `inst_006`–`inst_007` | PASS |
| 12 个 M6 Hub Bolt → Casing Base | `inst_008`–`inst_019` | PASS |
| 6 个 M10 Casing Nut → 对应 M10 Bolt | `inst_063`–`inst_068` | PASS |

部分代表性终点误差：M6 顶部/底部轴向误差约 2.80/2.83 mm；Casing closure 0.12 mm；M10 nut 0.19 mm；所有接口横向漂移均小于 1.10 mm。

## 视频证据

`tools/record_physics_rollout.py` 先运行完整 960-step collision-on PhysX，并保存每个采样时刻的真实部件位姿；之后停止仿真，将这条轨迹以 kinematic replay 送入 Isaac Replicator 渲染。这样不会因渲染器自己的 Kit step 重复推进物理时间。视频 sidecar 的 `all_pass` 为 `true`，视频有 121 个非空 RGB 帧，可由 Isaac 环境中的 imageio 解码。

该视频是 17 个 grouped interface 的同时隔离回放；49 个逐 ID 案例的数值终点由上述独立批次日志和 summary JSON 覆盖，未把一个小视频误称为 49 个机器人 rollout。

为便于检查孔和螺栓细节，另有单接口近景视频：`validation_logs/m6_hub_bolt_rollout.mp4`，对应 `M6_Hub_Bolt_to_Casing_Top`，同样是 960 个真实 PhysX step 的轨迹回放，sidecar 的 `all_pass` 为 `true`。

运行方式：

```bash
docker compose run --rm --no-deps --entrypoint bash hac -lc \
  'cd /workspace/Human-AI-Collab && tools/run_tool.sh tools/record_physics_rollout.py \
    --config assembly/physics_validation.json \
    --out validation_logs/assembly_physics_rollout.mp4'
```

## 仍未声称已证明的内容

- `Transfer Gear ↔ Input Shaft` 和 `Transfer Gear ↔ Output Gear` 的外齿在相对旋转、中心距扰动和负载下的连续啮合尚未通过动态试验；当前只证明了齿轮自身保留了齿形碰撞代理以及两个 gear-on-shaft 孔轴插入通过。
- 五个预装 bearing 的独立座合没有作为 canonical combine 步骤，当前仍属于初始 fixture。
- 还没有将 Franka/实际动作策略接入这些物理接口，也没有证明末端执行器可达性、抓取稳定性或真实 sequential rollout。
- 因此不能把本报告解读为模型已经理解装配物理，也不能把它解读为完整 Magic Assembly 已由机器人无障碍执行。
