# 五任务约束动作 MVP

这是 12-pair 计划的第一版动作策略扩展，范围明确限定为五个任务，并非最终的
12-pair benchmark。每个任务都使用仓库中已有的零件，并固定示范顺序 `A -> B`。
每个任务都有两个渲染变体：

| 任务 | A 后 B | HARD 变体中的反向顺序 | COMMUTABLE 变体中的反向顺序 | 零件 |
|---|---|---|---|---|
| HCF-01 | 安装 hub cover → 固定 M6 螺栓 | 螺栓头无法通过圆孔 | keyhole 的扩展槽允许螺栓头通过 | casing top、small cover、M6 bolt |
| WSG-01 | 安装垫圈 → 安装 output gear | 齿轮护罩遮住垫圈安装位 | 可见的径向槽允许从侧面插入 | output shaft、thin washer、output gear |
| KEY-01 | 插入 output key → 安装 output gear | 齿轮只覆盖键槽入口 | 侧向开放的键槽仍然可达 | output shaft、output key、output gear |
| CAS-01 | 闭合 casing → 插入贯穿螺栓 | 两半的凸耳不会形成贯穿孔 | captive pocket 可在闭合过程中保持螺栓 | casing base/top、M10 bolt |
| DOW-01 | 闭合 casing → 插入定位销 | 分体孔无法稳定承托定位销 | 开放式 retaining slot 可以保持定位销 | casing base/top、dowel pin |

精确测量的源零件包围盒和变异参数位于
`config/constrained_tasks.json`；生成的仅包含数据的场景 recipe 位于
`scenes/recipes/`。所有长度均以毫米编写，并且只转换一次为 stage 米制单位。
渲染/碰撞规则必须严格一致：可见特征和碰撞表示必须由同一个参数源生成。

## 离线契约与预期结果

运行：

```bash
python3 -m pilot_12pair.oracle.generate_task_manifests
python3 -m pilot_12pair.oracle.build_scene_recipes
python3 -m unittest pilot_12pair.tests.test_constrained_tasks -v
```

离线 oracle 仅用于检查任务契约。它预期：两个变体的 `A>B` 都可行；HARD 的
`B>A` 不可行；COMMUTABLE 的 `B>A` 可行；只执行其中一个动作时都不能完成任务。
在报告模型分数之前，必须由 Isaac 独立验证 swept-volume、接触、可达性、可见性和
终止条件。

## 研究设计依据

任务集遵循三个设计检查。第一，问题优先表述是：当前动作策略可能通过重放示范流程
完成装配，但在可见机构发生干预后，该流程可能已经物理无效。第二，边界探测让同一
策略逐一跨越不同的机构边界：螺栓头间隙、侧向可达性、键槽可达性、预留螺栓的保持，
以及定位保持。第三，简化性检查将每个领域限制为两个命名动作和四种终止候选，因此
错误不会被长时域规划或语言回答隐藏。

两句话的核心论断是：*一次成功的装配轨迹并不能证明动作模型知道哪一种顺序在物理上
是必要的。我们让目标和 A→B 示范保持相同，只改变一个可见机构，然后测量模型是否
恰好在 B→A 变得可行时改变第一步操作分支。* 这只是诊断性论断，并不声称某一个
成功 fixture 就能证明模型具备一般性的物理理解。

## 进行科学实验仍需完成的工作

- 编写 Isaac builder，把每个 recipe 转换为渲染 USD layer；
- 在不使用 teleport 或 snap attachment 的情况下，完成成对的控制器/随机种子校准；
- 为每个领域制作一条标准的、成功的 A→B 示范；
- 完成近距离视角的可见性和信息泄漏审计；以及
- 如果要评估 LingBot-VA 的任务完成能力，而不只是原生 action-output smoke，准备任务
  专用的 adaptation 数据。
