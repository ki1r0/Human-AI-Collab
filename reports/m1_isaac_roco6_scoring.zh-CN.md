# M1 Isaac Sim：ROCO 风格六分制

当前 M1 是脚本化物理 oracle，尚未加入 ACT。每次 rollout 都在 `metrics.json` 写入 `roco_style_score`，总分为 0–6；每项只有在对应的 PhysX/位姿证据满足时才得 1 分，不用视频主观判断。

| 分值 | 阶段 | 得分条件 |
|---:|---|---|
| 1 | 动态支撑 | Hub 从 reset 开始受重力，落在环形 staging pads 上，settle 后速度 ≤ 0.01 m/s。 |
| 1 | 一内一外抓取 | 两个指爪均有 Hub 接触力，并且接触点半径分别落在内壁 0.050–0.080 m、外壁 0.115–0.145 m。 |
| 1 | 抬升跟随 | Hub 实际位移 ≥ 20 mm，且不小于 link6 位移的 50%；不能用写入 Hub 位姿代替。 |
| 1 | 无碰撞搬运 | 运输采样点均为有限状态、速度 ≤ 1 m/s，至少 70% 的运输采样仍保持双指接触；不允许掉落、弹飞或穿过 Casing。 |
| 1 | 坐合与 6DoF 对齐 | Casing–Hub 有接触；径向误差 ≤ 4 mm、轴向误差 ≤ 8 mm、姿态误差 ≤ 2°；四个 output-top bolt hole 的最大误差 ≤ 3 mm。 |
| 1 | 释放与稳定撤离 | 开爪后两个指爪均无 Hub 接触，释放漂移 ≤ 5 mm，撤离前后 Hub 漂移 ≤ 20 mm，且已坐合。 |

只有 6/6 才允许 `insertion_verdict=SUCCESS`。例如“确实夹住但还没搬到 Casing”应得到 2/6 或 3/6，而不是成功；`candidate_verdict` 和 `SEAT_RELEASE_CANDIDATE` 仍是调试标签，不等于满分。

运行结果示例：

```bash
M1_NO_VIDEO=1 tools/run_m1_isaac_strict_success.sh
python3 - <<'PY'
import json
d=json.load(open('validation_logs/m1_isaac_strict_success/metrics.json'))
print(d['roco_style_score'])
PY
```

相机录制版本仍使用同一个评分，只需不要设置 `M1_NO_VIDEO=1`；MP4 是证据附件，分数以 `metrics.json` 中的物理记录为准。
