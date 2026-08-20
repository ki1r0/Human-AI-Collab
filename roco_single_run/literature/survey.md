# Primary-source survey

## RoCo Challenge paper

- Title: *RoCo Challenge at AAAI 2026: Benchmarking Robotic Collaborative Manipulation for Assembly Towards Industrial Automation*
- Authors: Haichao Liu et al.
- Date: 2026-03-16
- URL: https://arxiv.org/abs/2603.15469
- Relevance: Defines Task 1 (assembly from scratch), Task 2 (partial state), Task 3 (recovery), the Galaxea R1 simulation platform, multi-modal demonstrations, and ACT/π0.5 baselines.

## Official gearbox benchmark

- Repository: https://github.com/rocochallenge/gearboxAssembly
- Pinned commit: `094a1f76d18c207caec198315f23b1a60dbca94f`
- Local checkout: `/home/sunsiliang/roco_runtime/gearboxAssembly`
- Relevance: Primary authority for environment creation, cameras, state/action ordering, rule policies, ACT wrapper, reset, control, and score logic.

## Official RoCo dataset devkit

- Repository: https://github.com/rocochallenge/roco_dataset_devkit
- Pinned commit: `23522d72af214158d3c56ee2f171888c3e74698d`
- Dataset: https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025
- Relevance: Confirms the public R1 simulation split, RGB-D + qpos + actions, 20 Hz data, and a 2.32 TB total release.

## Official challenge site

- URL: https://rocochallenge.github.io/RoCo2026/
- Relevance: Competition documentation and artifact hub. No organizer-published ACT checkpoint was found in the public repository or indexed documentation.

## Candidate learned checkpoint (non-official)

- Model: https://huggingface.co/yjsm1203/roco_model_act_2
- Revision: `52344a203e0739638cb2c7b11ea632e7b2eb2608`
- Relevance: Public 336 MB ACT checkpoint with matching `dataset_stats.pkl`; trained on 241 episodes combining official RoCo data and additional collected data.
- Fidelity: RoCo-compatible but third-party, not an organizer baseline artifact.

## Generation selection

The IROS 2026 challenge uses industrial-board and brick assembly. This machine's workspace contains gearbox assets and the requested paper/repository target is the AAAI 2026 planetary gearbox. The gearbox generation is therefore selected; no code from the later industrial-board generation will be mixed into the baseline.

