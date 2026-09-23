# LingBot-VA

- **Primary sources:** [official repository](https://github.com/robbyant/lingbot-va), [paper](https://arxiv.org/abs/2601.21998), [base checkpoint](https://huggingface.co/robbyant/lingbot-va-base).
- **Model role here:** first action/world-action deployment baseline; use the base checkpoint before any task-specific post-training.
- **Official interface facts:** the repository provides image-to-video-action generation and server/client inference. The model uses a 30-dimensional multi-embodiment action layout; the released Franka/Robotwin configs select Cartesian end-effector and gripper channels. The official README lists Python 3.10.16, PyTorch 2.9.0, CUDA 12.6, and roughly 18 GB VRAM for single-GPU image-to-video-action inference with offload.
- **Relevance to this benchmark:** the base checkpoint can test whether the pipeline is wired and whether output actions are non-degenerate. It is not a calibrated custom-embodiment policy, so task success requires later adaptation and a common controller.
- **Reproducibility record:** checkout `7c6ffa9` was inspected on 2026-09-23; the model repository page reported 24.4 GB for `lingbot-va-base` at inspection time.
