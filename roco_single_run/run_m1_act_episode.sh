#!/usr/bin/env bash
set -euo pipefail

# Minimal learned-policy M1 smoke.  The public RoCo ACT checkpoint is a
# gearbox-domain policy; this launcher makes the domain shift explicit and
# records a strict failure/smoke artifact rather than silently substituting a
# rule policy.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${ROCO_M1_ACT_CHECKPOINT:-/home/sunsiliang/roco_runtime/checkpoints/roco_model_act_2/policy_best.ckpt}"
STATS="${ROCO_M1_ACT_STATS:-/home/sunsiliang/roco_runtime/checkpoints/roco_model_act_2/dataset_stats.pkl}"
OUT="${ROCO_M1_ACT_OUTPUT_DIR:-$ROOT_DIR/roco_single_run/runs/m1_act_smoke_seed23}"
GPU="${ROCO_GPU_DEVICE:-0}"
STEPS="${ROCO_M1_ACT_STEPS:-20}"

mkdir -p "$OUT"
exec docker run --rm --gpus "device=$GPU" --ipc=host --network=host \
  --entrypoint /isaac-sim/python.sh -w /workspace/Human-AI-Collab \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/workspace/roco_runtime/torch_cache -e HRC_M1_ROOT=/workspace/Human-AI-Collab \
  -e PYTHONPATH=/workspace/Human-AI-Collab:/workspace/gearboxAssembly/source/Galaxea_Lab_External:/workspace/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT:/workspace/Human-AI-Collab/roco_single_run \
  -v "$ROOT_DIR:/workspace/Human-AI-Collab" \
  -v /home/sunsiliang/roco_runtime/gearboxAssembly:/workspace/gearboxAssembly:ro \
  -v /home/sunsiliang/roco_runtime/checkpoints:/workspace/roco_runtime/checkpoints:ro \
  -v /home/sunsiliang/roco_runtime/torch_cache:/workspace/roco_runtime/torch_cache \
  nvcr.io/nvidia/isaac-lab:2.3.0 \
  -m roco_single_run.scripts.run_m1_act_episode \
  --checkpoint "/workspace/roco_runtime/checkpoints/roco_model_act_2/policy_best.ckpt" \
  --stats "/workspace/roco_runtime/checkpoints/roco_model_act_2/dataset_stats.pkl" \
  --output-dir "/workspace/Human-AI-Collab/$(realpath --relative-to="$ROOT_DIR" "$OUT")" \
  --max-steps "$STEPS" --enable_cameras --headless
