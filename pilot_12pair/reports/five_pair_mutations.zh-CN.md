# 五个 constrained pair：具体装配关系、mutation 与当前真实性

## 先纠正一个容易误解的表述

此前把这些 USD 称为“经过几何一致性审计的 fixture”，容易让人理解成“已经是
可以在 Isaac 中直接播放、齿轮也已经完成物理啮合校准的最终场景”。这个理解不对。

目前审计实际覆盖的是：

1. 配置里的源 STL 包围盒与登记尺寸是否一致；
2. 生成 USD 是否能引用仓库中的源零件；
3. 任务参数的单位、可见 mutation 参数和 fixture 碰撞 API 是否存在；以及
4. 离线 oracle 的顺序逻辑是否自洽。

`pilot_12pair/scenes/generated/*.usda` 是 **pre-Isaac fixture**：源零件作为外观引用，
mutation 目前由独立的正几何体（条块/圆柱）表示。它们不是把孔、槽、护罩或 pocket
真正布尔写回 `Casing Top`、`Output Gear` 等 CAD 网格，也没有完成 swept-volume、接触、
可达性、摄像机可见性或控制器校准。因此，当前离线结果不能被当作物理实验结果。

## 五个 pair 分别对应什么

`A→B` 是统一展示的顺序；实验真正比较的是在同一目标下，`B→A` 是否因为局部机构
改变而变得可行。`A-only` 和 `B-only` 始终不算完成。

| Pair | A→B 的具体装配关系 | HARD 中为什么必须 A→B | COMMUTABLE 中改变了什么 | 与原始 gearbox 顺序的关系 |
|---|---|---|---|---|
| **HCF-01** | 放好 small hub cover → 插入并固定一个 M6 hub bolt | 圆孔不能让已保留的螺栓头穿过 | 在同一局部区域增加 keyhole 大 lobe，螺栓先保留也能让 cover 通过并旋锁 | 对应原计划中 `Hub_Cover_Small_Top` 之后安装其 M6 bolts 的局部关系（原计划步骤 12→15 的简化单螺栓 probe） |
| **WSG-01** | 将薄垫圈放到 output-shaft shoulder → 安装 output gear | 齿轮落下后封住垫圈肩部，没有侧向入口 | 在齿轮座护罩上开径向侧槽，齿轮先装好仍能从侧面放入垫圈 | **不是**原始演示中的独立步骤；原始场景把 `Output_Gear` 作为 output shaft 模块的一部分预装。它是从真实零件抽出的新局部 probe |
| **KEY-01** | 将 Output Key 完全推入 shaft keyway → 安装 output gear | 齿轮裙部覆盖唯一的键槽入口 | 保留一个侧向开放的 keyway window，齿轮先装后仍可插 key | 当前 gearbox 计划没有这个独立 `Output Key` 步骤；是使用仓库零件的新局部 probe |
| **CAS-01** | 合上 casing top/base → 插入并坐实 M10 through-bolt | casing 未合拢时上下凸耳不形成连续贯穿孔 | 在 top-side lug 增加 captive pocket，使 bolt 可先被保持，再完成合盖 | 直接对应原计划步骤 21→23：合盖后安装 M10 bolt |
| **DOW-01** | 合上 casing top/base → 插入 locating dowel | 两个半孔在分离状态不能稳定承托整根 dowel | 增加开放式 retaining slot，可在合盖前保持 dowel | 原始计划没有该独立 dowel 顺序；是使用现有 `Dowel Pin` 的新增局部 probe |

示意图（只表达约束拓扑，不冒充 CAD）见：

![五个 pair 的 mutation 示意](../outputs/geometry_demo/five_pair_mutation_schematic.png)

## 1. HCF-01：small hub cover / M6 headed bolt

### 任务动作

- 初始：`Casing Top` 上的 `Hub Cover Small` 松放在旁边；一个 `M6 Hub Bolt` 在可见
  rack；guide bore 为空。
- **A — `seat_cover`**：把 small cover 对准 casing top 的 registration features，完成
  cover seating，并把 yaw 从预锁角校正到最终角。
- **B — `retain_bolt`**：将 M6 bolt 沿 guide bore 插入并完成 quarter-turn/retention。
- 目标：cover 已坐实、bolt 已保持，不能 teleport 或 snap。

### mutation 的精确参数

| 参数 | 值 |
|---|---:|
| M6 shaft diameter | 5.88 mm |
| M6 head diameter（碰撞包络） | 11.70 mm |
| HARD round-hole diameter | 6.80 mm |
| COMMUTABLE keyhole neck | 6.80 mm |
| COMMUTABLE lobe diameter | 13.60 mm |
| pre-lock yaw → final yaw | -12° → 0° |
| head underside gap | 5.35 mm |
| cover-under-head clearance | 0.35 mm |

HARD 中 `11.70 > 6.80`，因此 bolt 先装会被 head 卡住；COMMUTABLE 中 lobe 提供
`13.60 - 11.70 = 1.90 mm` 的直径余量，再以 -12° 的 bayonet motion 进入 6.80 mm
的窄颈。登记的干预余量是 0.95 mm。

### 当前实现状态

源 `Hub Cover Small.usd` 与 `M6 Hub Bolt.usd` 被引用了，但当前生成器用四根正方条表示
round/keyhole 周围的 fixture，COMMUTABLE 的 lobe 还是独立圆柱；它不是 cover 网格中的
真实布尔 keyhole。更重要的是，仓库现有 M6 collider 在 controlled collision-on 插入中
仍会停在 casing 表面（见根目录 `part-fidelity-matrix.md`），所以 HCF-01 目前只能算
任务契约/几何 fixture，不能算已通过物理装配验证。

## 2. WSG-01：washer / output gear

### 任务动作

- 初始：`Output Shaft` 固定在 fixture，`Thin washer` 和 `Output Gear` 分开放在 tray。
- **A — `seat_washer`**：将 1 mm 薄垫圈平放到 output-shaft shoulder。
- **B — `gear_seated`**：沿 shaft 轴线下降 output gear，直至 gear seat 完成。

### mutation 的精确参数

| 参数 | 值 |
|---|---:|
| washer outer diameter / thickness | 72.0 / 1.0 mm |
| Output Gear bounding diameter | 161.157 mm |
| gear-skirt bottom clearance | 1.25 mm |
| COMMUTABLE radial slot width | 2.00 mm |
| slot 相对 1 mm washer 的 nominal余量 | 0.75 mm |

HARD 是封闭 annular shroud；齿轮落下后垫圈没有垂直或径向路径。COMMUTABLE 只在
shroud 上增加一个可见径向槽，齿轮仍可先装，垫圈可以从侧面进入。

### 当前实现状态

这个 pair 并不是人类原始视频/`pilot-plan.md` 中已经确认的独立步骤。原始 gearbox
初始状态把 `Output_Gear` 和 `Output_Shaft` 作为模块预装，然后在计划步骤 19 插入
output module。因此 WSG-01 是一个新的、可控的局部因果 probe，不应写成“复现了原始
assembly step”。当前 USD 的 shroud 也是正方条 fixture，不是 Output Gear 或 shaft
周围的真实护罩布尔修改。

## 3. KEY-01：Output Key / output gear

### 任务动作

- 初始：shaft keyway entrance、key 和 gear bore 都可见。
- **A — `insert_output_key`**：沿 keyway 将 key 完全推到底。
- **B — `mount_output_gear`**：把 output gear 套上 keyed shaft 并坐实。

### mutation 的精确参数

| 参数 | 值 |
|---|---:|
| key length × width × height | 29.70 × 19.80 × 14.00 mm |
| keyway entry clearance | 0.50 mm |
| gear bore length | 27.00 mm |
| COMMUTABLE side-access width | 21.00 mm |

HARD 把 gear skirt 下的唯一键槽入口闭合；COMMUTABLE 保留一个 21 mm 侧向开放窗口，
所以 gear 先装后仍能从侧面插 key。

### 当前实现状态

`Output Key` 来自仓库中的 CAD_edit 资源，但它没有出现在原始 `pilot-plan.md` 的
独立 assembly DAG 中。当前 generated USD 只用导入的 shaft/gear 外观加正方条表示
barrier/guide，并未把 keyway window 写成真实 shaft/gear 几何。因此它是“参数化 probe
已定义”，不是“真实 keyway 物理已验证”。

## 4. CAS-01：casing closure / M10 through-bolt

### 任务动作

- 初始：`Casing Base` 与 `Casing Top` 分离，M10 bolt 在 rack，top/base lug 可见。
- **A — `close_casing`**：把 top 降下并坐到 base 上，完成 mating。
- **B — `insert_casing_bolt`**：从 top 侧插入 M10 through-bolt，直到 head seat。

这是五个 pair 中最接近原始 assembly DAG 的一个：原计划步骤 21 先合 casing，步骤
23 再装第一个 M10 bolt/nut pair。

### mutation 的精确参数

| 参数 | 值 |
|---|---:|
| casing bounding box | 220.037 × 277.267 × 55.897 mm |
| bolt head envelope | 19.514 × 16.900 mm |
| bolt axis length | 116.45 mm |
| through-bore diameter | 20.50 mm |
| lug alignment tolerance | 0.50 mm |
| COMMUTABLE captive pocket depth | 18.00 mm |
| head clearance in pocket | 1.50 mm |

HARD 的 separated lugs 只有在合盖后才形成贯穿孔；COMMUTABLE 在 top-side lug 增加
visible captive pocket，使 bolt 在合盖时保持而不阻碍 mating。

### 当前实现状态

当前 fixture 的 `top_lug` 和 pocket walls 是独立 cube，不是 `Casing Top` 的真实
孔/槽。仓库的 M10 fastener 也尚未有等价的 controlled collision-on proof，所以不能
把 CAS-01 的 reverse success 当作已测得的 casing physics。

## 5. DOW-01：casing closure / locating dowel

### 任务动作

- 初始：casing halves 分离，dowel upright 在 tray，split bore/retention feature 可见。
- **A — `close_casing`**：合上 casing。
- **B — `insert_locating_dowel`**：沿对齐后的 bore 插入 dowel，直到稳定 seated。

### mutation 的精确参数

| 参数 | 值 |
|---|---:|
| dowel diameter / length | 13.80 / 59.70 mm |
| aligned bore diameter | 14.40 mm |
| bore depth per casing half | 29.85 mm |
| COMMUTABLE retaining-slot width | 15.20 mm |
| required radial clearance | 0.30 mm |

HARD 只有两个在空间上分开的 half-bores，任一半都不能独立稳定保持整根 dowel；
COMMUTABLE 以 15.20 mm 的开放 slot 先保持 13.80 mm dowel，再允许 casing closure。

### 当前实现状态

这是使用仓库 `Dowel Pin_rescaled.usd` 的新 probe；原始 gearbox 计划没有这条独立
dowel 顺序。当前 `slot_left/right` 是正方条 fixture，尚未成为 casing top/base 的
真实 split-bore 或 open-slot 几何。

## 离线 oracle 与真实物理结果的区别

当前配置的离线契约只断言：

```text
A→B: HARD 和 COMMUTABLE 都应可行
B→A: HARD 不可行；COMMUTABLE 可行
A-only/B-only: 都不能完成目标
```

这只是根据参数记录写出的逻辑 oracle。真正发表结果前，必须重新生成 Isaac scene，
将 mutation 作为真实可见几何写入正确的 casing/shaft/cover，完成 paired controller
和 seed 校准，再记录 swept-volume、contact/penetration、最终相对位姿、可达性、可见性
和终止条件。

## 齿轮：当前到底建模到什么程度

`assets/parts/Output Gear.usd` 的源视觉 mesh 确实包含齿形。用 USD 直接读取得到：

- visual mesh：50,544 个点、16,848 个 authored triangular faces；
- `collision_gear`：384 个点、384 个 authored polygons（渲染时四边形扇形拆成 768 个三角形）；
- stage `metersPerUnit=0.01`；
- visual 与 collision 都来自同一个 `Output Gear.usd`，但 collision 是为物理稳定性
  做的 toothed-annulus proxy，不是 visual CAD 的逐面复制。

因此可以准确地说：**齿在源视觉模型中存在，且碰撞代理保留了齿圈的外部拓扑意图；不能
准确地说已经证明了两个齿轮逐齿啮合、旋转时无重叠。** 根目录
`part-fidelity-matrix.md` 明确把 `Transfer_Gear ↔ Output_Gear` 的 running tooth
clearance/rotation 标成 `UNVALIDATED`。当前已经通过的是两个 gear-on-shaft 的受控插入，
不是 gear-to-gear running mesh。

## 可直接查看的演示

这些输出是本次新生成的，不是之前 288×64 的 LingBot smoke video：

- [五个 pair 的 mutation 概念图](../outputs/geometry_demo/five_pair_mutation_schematic.png)
- [Output Gear 源视觉网格 vs collision proxy](../outputs/geometry_demo/output_gear_visual_vs_collision.png)
- [Output Gear 旋转诊断视频](../outputs/geometry_demo/output_gear_visual_vs_collision.mp4)
- [演示 manifest](../outputs/geometry_demo/manifest.json)

重新生成：

```bash
conda run -n lingbot-va python -m pilot_12pair.oracle.render_mutation_schematic
conda run -n lingbot-va python -m pilot_12pair.oracle.render_geometry_demo --video-frames 48
```

这段视频只旋转并排显示的真实 source visual mesh 和当前 collision proxy，目的是让
几何差异可见；它不模拟 gear-to-gear 接触，也不宣称播放后齿会自动一一扣住。要完成
“齿不重叠”的演示，下一步必须在 Isaac 中固定真实 `Transfer_Gear`/`Output_Gear`
pose，扫描 phase 与 center distance，在 collision-on 条件下运行旋转并导出 contact、
penetration 和 torque 日志。
