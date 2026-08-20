# RoCo 2026 single-run reproduction

Status: **IN PROGRESS — not yet a successful learned-policy reproduction**.

This integration reconstructs the AAAI 2026 RoCo planetary-gearbox Task 1 pipeline around the pinned official `rocochallenge/gearboxAssembly` source. It deliberately does not reuse this repository's single-Franka kinematic assembly path.

The stable external reference checkout is currently:

```text
/home/sunsiliang/roco_runtime/gearboxAssembly
```

The selected isolated runtime is the locally cached `nvcr.io/nvidia/isaac-lab:2.3.0` image. A one-command learned rollout will be documented here only after the staged environment, observation, policy, and replay/action-interface gates pass.

See:

- `research-log.md` for the chronological trace;
- `fidelity-matrix.md` for source-to-local contracts and deviations;
- `issues-and-fixes.md` for evidence-driven failure analysis;
- `research-state.yaml` and `findings.md` for persistent project state.

