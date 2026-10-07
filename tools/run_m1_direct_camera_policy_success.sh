#!/usr/bin/env bash
set -euo pipefail

# Camera-backed companion to run_m1_direct_policy_success.sh.  The direct
# physics baseline remains unchanged; this launcher enables the same RTX
# camera path and yuv420p writer used by RoCo's episode runner.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${M1_DIRECT_CAMERA_OUTPUT_DIR:-$ROOT_DIR/validation_logs/m1_direct_camera_policy_success}"
GPU="${M1_GPU_DEVICE:-0}"
STRIDE="${M1_CAMERA_UPDATE_STRIDE:-10}"
RENDER_INTERVAL="${M1_RENDER_INTERVAL:-5}"
VIDEO_WIDTH="${M1_CAMERA_WIDTH:-320}"
VIDEO_HEIGHT="${M1_CAMERA_HEIGHT:-240}"
STEP_SCALE="${M1_STEP_SCALE:-0.5}"
RELEASE_OPEN_STEPS="${M1_RELEASE_OPEN_STEPS:-1000}"
GRAVITY_SETTLE_STEPS="${M1_GRAVITY_SETTLE_STEPS:-500}"
HEADLESS="${M1_HEADLESS:-1}"
DOCKER_DISPLAY_ARGS=()
APP_ARGS=(--headless --enable_cameras)
if [[ "$HEADLESS" == "0" ]]; then
  : "${DISPLAY:?M1_HEADLESS=0 requires DISPLAY}"
  : "${XAUTHORITY:?M1_HEADLESS=0 requires XAUTHORITY}"
  [[ -d /tmp/.X11-unix ]] || { echo "M1_HEADLESS=0 requires /tmp/.X11-unix" >&2; exit 2; }
  [[ -f "$XAUTHORITY" ]] || { echo "XAUTHORITY is not a readable file: $XAUTHORITY" >&2; exit 2; }
  DOCKER_DISPLAY_ARGS=(
    -e "DISPLAY=$DISPLAY"
    -e XAUTHORITY=/tmp/m1.Xauthority
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v "$XAUTHORITY:/tmp/m1.Xauthority:ro"
  )
  APP_ARGS=(--enable_cameras)
fi
mkdir -p "$OUT"

exec docker run --rm --gpus "device=$GPU" --ipc=host --network=host "${DOCKER_DISPLAY_ARGS[@]}" \
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
  --release-open-steps "$RELEASE_OPEN_STEPS" --gravity-settle-steps "$GRAVITY_SETTLE_STEPS" --step-scale "$STEP_SCALE" \
  --camera-update-stride "$STRIDE" --render-interval "$RENDER_INTERVAL" \
  --video-width "$VIDEO_WIDTH" --video-height "$VIDEO_HEIGHT" \
  "${APP_ARGS[@]}"
