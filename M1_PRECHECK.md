# M1 Precheck

更新时间：2026-09-25（Asia/Singapore）

本文件记录 M1 执行前的只读盘点。主工程和 RoCo checkout 中已有的未提交内容均保留；本 M1 只新增独立路径，不在 RoCo checkout 内修改文件。

## 工程与版本

| 项目 | 结果 |
|---|---|
| 主工程 | `/home/sunsiliang/Human-AI-Collab` |
| 主工程 HEAD | `0e280c6f0f2aaf91589ff27fb7b130b11a2d7e31`，branch `main` |
| 主工程工作树 | 有既有修改和未跟踪文件；见执行开始时的 `git status --short`，未清理/覆盖 |
| RoCo checkout | `/home/sunsiliang/roco_runtime/gearboxAssembly` |
| RoCo HEAD | `094a1f76d18c207caec198315f23b1a60dbca94f`，branch `main` |
| RoCo IsaacLab submodule | `-3c6e67bb5c7ada942a6d1884ab69338f57596f77 IsaacLab` |
| RoCo 工作树 | 27 个既有修改/未跟踪项；不修改 |
| RoCo LFS | `git lfs fsck` 报告多项 tracked STL 为 `unexpectedGitObject`；这不是 M1 新增的修复 |
| 主机 | Linux x86_64，kernel `7.0.0-31-generic` |
| GPU | 4 × NVIDIA RTX A5000，driver `580.173.02`，每张 23028 MiB |
| 主机 Python | 3.13.11；不用于 Isaac runtime |
| Isaac runtime | Docker image `nvcr.io/nvidia/isaac-lab:2.3.0`，image ID `sha256:1b204053c9facaa168b9aacd014061b479dabc0569f35f626dd60ea9d667188a` |
| Isaac Python | container Python 3.11.13，Torch `2.7.0+cu128`，IsaacLab `0.47.1` |
| 已构建主工程镜像 | `human-ai-collab:latest`，image ID `sha256:086add3786e78c9275689651d876c1c70f12ed5366d151bdca169753fe430c1a` |

## 已确认入口

只读挂载 RoCo checkout 到 Isaac Lab image 后，以下命令成功：

```bash
docker run --rm --gpus all --ipc=host --network=host \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_ENV_PRIVACY_CONSENT=YES \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace/gearboxAssembly/source/Galaxea_Lab_External \
  -v /home/sunsiliang/roco_runtime/gearboxAssembly:/workspace/gearboxAssembly:ro \
  nvcr.io/nvidia/isaac-lab:2.3.0 \
  /workspace/gearboxAssembly/scripts/list_envs.py
```

注册的环境：

- `Template-Galaxea-Lab-Agent-Direct-v0`
- `Template-Galaxea-Lab-External-Direct-v0`

对应日志由该命令产生于 Isaac Kit container log；本次命令退出码为 0。尚未把它记录为 M1 simulator smoke，因为它没有加载 M1 端盖任务。

## 当前 RoCo 接口事实

从 checkout 源码核对：

- `GalaxeaLabExternalEnvCfg.sim_dt = 0.01`，`decimation = 5`；控制步约 20 Hz。
- `action_space = 14`，但 `GalaxeaLabExternalEnv` 的内部 `_apply_action` 使用 rule policy 产生的 joint-position target，不是一个通用 14-D Cartesian action adapter。
- observation 返回 `head_rgb`、`left_hand_rgb`、`right_hand_rgb`、对应 depth，以及左右臂 6-D joint position/velocity 和单值夹爪 position/velocity。
- `GalaxeaRulePolicy`、`evaluate_score()` 和环境 reset 都硬编码 planetary carrier/ring/sun gear/reducer；不能直接用于 `Hub_Cover_Output_Top → Casing_Top`。
- `GalaxeaLabExternalEnv` 的当前 scene 不包含主工程的 casing/cover USD。

因此 M1 必须新增独立 RoCo task adapter/scene 层，复用 RoCo 的 R1 articulation、相机和 controller 边界；不能仅修改 `executor_mode` 或调用现有 `MagicAssemblyManager.combine()`。

## M1 资产与序列核对

计划附件中的路径 `assets/assets/parts/...` 是附件归档结构；当前主工程实际路径为 `assets/parts/...`。关键当前文件存在：

- `assembly/gearbox_sequence.yaml`
- `assembly/asset_registry.yaml`
- `assembly/scatter_layout_007050.yaml`
- `assembly/physics_validation.json`
- `assets/parts/Hub Cover Output.usd`
- `assets/parts/Casing Top.usd`
- `assets/AI_robot.usd`
- `runtime/magic_assembly.py`
- `runtime/llm_commander.py`

关键当前 SHA-256：

```text
4ef9d16e8a9566df05333074852d2e8a987c3a7b97b7cd730254bebca99a175  assets/parts/Hub Cover Output.usd
999e82054a4f096b71f08733b4cb2ea1f27fa7a38a474a0c456c1240f9afb810  assets/parts/Hub Cover Output.stl
36bc3032a594050413be9fc6d433d63b657a524f875275c78449c1cd26b80931  assets/parts/Casing Top.usd
2fefcb6d2d9be42b7d274d649c27e70abd131fee200e3d787a7a675779c73f5b  assets/parts/Casing Top.stl
98c35b7a851293f8cbe22db6122a5c9e41601352d620f105391fb987060d62f0  assembly/physics_validation.json
8ca0e313dc338ab52c6b1e4b7ba12be69e6580e21ca140cacff48d5a4f31bf4d  assembly/gearbox_sequence.yaml
fba61048153bf5e4f02a0747006e54d0b1a12b5582446fb7e063d9cb4dde9577  assembly/instances/canonical_grouped.yaml
f39eb490543ce47d82035178e81b834bba21e71d94f93b611d55f9b33f422f12  runtime/magic_assembly.py
```

## 预先存在的物理证据

`assembly/physics_validation.json` 中 `Hub_Cover_Output_to_Casing_Top` 对应 `inst_025_combine_hub_cover_output_top`，当前报告为 PASS。该证据是 collision-on 的隔离接口试验，不是 RoCo robot rollout；M1 必须在 RoCo scene 中回归，并单独记录机器人抓取、可达性、释放和静置。

## 当前阻断/风险

1. RoCo 原 checkout 的 dirty/LFS 状态禁止作为干净发布依赖。
2. 主机没有 standalone Isaac Sim；必须通过本地 Docker image 运行。
3. 计划要求的真实 HIL 需要有人在线操作 TAKE_CONTROL/RETURN_CONTROL；Codex 不能代替真人，未完成时 G3/G4 应为 `WAITING_FOR_OPERATOR`。
4. 当前 RoCo rule policy 是另一种 planetary gearbox 任务；M1 adapter 需要明确复用 R1 controller 的接口，不能把其 planetary skill 当作 M1 成功。
5. 质量、惯量、摩擦的真实测量来源尚未在当前主工程证据中发现；没有来源的数值只能标 `provisional`。

6. 2026-09-26 的 R1 collision-on grasp audit（`validation_logs/m1_grasp_blocker_20260926.md`）确认当前 Hub cover 的 broad CAD envelope 约 0.260 m、薄轴约 0.028 m；R1 当前测试夹爪的有效对向指爪中心间距约 0.123 m。中央孔和外侧对向夹持均未产生稳定 lift，达到碰撞边界会弹射 Hub；后续在校准 contact offset 后，预张开内壁路径的两个过滤传感器均为 0 N、Hub 位移为 0 m。因此当前结论是“内壁抓取路径尚未验证”的 G1 阻断，而不是 Hub 无法抓取的硬事实，也不是可由 planner/VLM 修复的感知问题。

7. 为避免薄盖板被隐式 SDF margin 放大，M1 dynamic Hub collider 已显式设置 SDF margin=0、narrow-band thickness=0；`validation_logs/m1_physics_trial_sdf0_20260926T1630Z/` 的 collision-on 接口回归仍为 `PROVISIONAL_PASS`。这只证明坐合接口回归未被该校准破坏，不改变 B6“内壁抓取尚未验证”的结论。
