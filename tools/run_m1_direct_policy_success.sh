#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${M1_DIRECT_OUTPUT_DIR:-$ROOT_DIR/validation_logs/m1_direct_policy_rerun}"
GPU="${M1_GPU_DEVICE:-0}"
mkdir -p "$OUT"

exec docker run --rm --gpus "device=$GPU" --ipc=host --network=host \
  --entrypoint /isaac-sim/python.sh -w /workspace/Human-AI-Collab \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONUNBUFFERED=1 -e HRC_M1_ROOT=/workspace/Human-AI-Collab \
  -e PYTHONPATH=/workspace/Human-AI-Collab:/workspace/gearboxAssembly/source/Galaxea_Lab_External \
  -v "$ROOT_DIR:/workspace/Human-AI-Collab" \
  -v /home/sunsiliang/roco_runtime/gearboxAssembly:/workspace/gearboxAssembly:ro \
  nvcr.io/nvidia/isaac-lab:2.3.0 -m hrc_m1.debug_inner_wall_grasp \
  --output-dir "/workspace/Human-AI-Collab/$(realpath --relative-to="$ROOT_DIR" "$OUT")" \
  --orientation radial --radial-offset 0.085 --opening 0.018 --approach-opening 0.045 \
  --z-offset 0.041 --gripper-contact-offset 0.0001 \
  --grasp-correction-x-deg=-4 --grasp-correction-y-deg=-5 \
  --preserve-lift-grasp-frame --insert-after-lift --release-before-seat --hub-gravity \
  --preinsert-height-m 0.14 --release-opening 0.05 \
  --release-clearance-x 0 --release-clearance-y 0 --release-clearance-z 0 \
  --release-open-steps 1000 --gravity-settle-steps 500 --step-scale 0.5 --no-video --headless
