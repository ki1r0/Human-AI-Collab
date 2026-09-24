# LingBot-VA base smoke 报告

日期：2026-09-23  
官方 checkout：`/home/sunsiliang/Downloads/lingbot-va`，revision `7c6ffa9`  
Checkpoint：`/home/sunsiliang/models/lingbot-va-base`（`robbyant/lingbot-va-base`）  
环境：`lingbot-va`（Python 3.10.16、Torch 2.9.0+cu126、Diffusers 0.36.0、
Transformers 4.55.2、USD 25.11）

## 结果

Base 模型已经从 HCF-01/HARD 的接口 smoke observation 生成了原生 Franka i2va
输出。成功输出位于
`outputs/lingbot_va_hcf01_hard_fast_lowres/model_output/`：

- `actions_0.pt`：形状 `(1, 30, 4, 20, 1)`，`bfloat16`，全部为有限值，其中 929 个元素非零；
- `latents_0.pt`：形状 `(1, 48, 4, 4, 18)`，`bfloat16`，全部为有限值，其中 13,799 个元素非零；
- `demo.mp4`：可读取，13 帧，尺寸 288x64，10 fps；
- `run_manifest.json`：记录模型、源代码 revision、输入以及所有 smoke 覆盖参数。

本次运行特意使用了 `text_max_length=64`、`height=64`、`width=96`，以及一次
video/action diffusion step。这是部署/接口 smoke，不是任务完成分数，也不能作为
理解因果约束的证据。

## 复现命令

```bash
LINGBOT_VA_GPU=0 \
LINGBOT_VA_TEXT_MAX_LENGTH=64 \
LINGBOT_VA_VIDEO_STEPS=1 LINGBOT_VA_ACTION_STEPS=1 \
LINGBOT_VA_HEIGHT=64 LINGBOT_VA_WIDTH=96 \
LINGBOT_VA_TASK_OUTPUT=pilot_12pair/outputs/lingbot_va_hcf01_hard_fast_lowres \
./pilot_12pair/lingbot_va/run_task_smoke.sh HCF-01 HARD
```

官方默认设置（`224x320`、文本长度 `512`、`5+10` steps）能够正确加载模型，但在
本机上受 CPU offload 限制，20 分钟内没有写出结果；该诊断目录单独保留在
`outputs/lingbot_va_hcf01_hard_base_retry1/`。

## 循环中完成的修正

1. 上游服务器的旧式顶层 `configs` import 使用了第二个配置对象；适配器现在直接修改
   `wan_va_server.VA_CONFIGS`。
2. 可选依赖 `flash_attn` 的 stub 现在带有 `ModuleSpec`，因此可以通过 Diffusers 的
   可选依赖探测，同时推理仍使用官方的 `attn_mode="torch"` 路径。
3. launcher 现在导出固定的上游 checkout 路径，用于记录运行 provenance。
