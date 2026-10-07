#!/usr/bin/env bash
set -euo pipefail

: "${HRC_REPAIR_PLANNER_ENDPOINT:?Set the OpenAI-compatible Qwen planner endpoint}"
: "${HRC_REPAIR_API_KEY:?Set the key expected by the local Qwen endpoint}"
export HRC_REPAIR_VLM_ENDPOINT="${HRC_REPAIR_VLM_ENDPOINT:-$HRC_REPAIR_PLANNER_ENDPOINT}"
export HRC_REPAIR_VLM_MODEL="${HRC_REPAIR_VLM_MODEL:-${HRC_REPAIR_PLANNER_MODEL:-Qwen/Qwen3-VL-2B-Instruct}}"

# Carry the demonstrated inner/outer pinch, then use the high-clearance route
# and conservative seat/release checks from the measured Isaac diagnostic.
export M1_GPU_DEVICE="${M1_GPU_DEVICE:-0}"
export M1_GRASP_CORRECTION_X_DEG=-4
export M1_GRASP_CORRECTION_Y_DEG=-5
export M1_TRANSPORT_CLEARANCE_Z=0.55
export M1_TRANSPORT_POSITION_ONLY=1
export M1_TRANSPORT_DIRECT=1
export M1_REANCHOR_AT_INSERTION_ABOVE=1
export M1_SEAT_POSITION_ONLY=1
export M1_SEAT_VERTICAL_ONLY=1
export M1_RELEASE_ONLY_IF_SEATED=1
export M1_CONTROLLED_SEAT_SEGMENTS=8
export M1_CONTROLLED_SEAT_SEGMENT_STEPS=28
export M1_STEP_SCALE=0.5
export M1_ISAAC_IPC_MODE=private

exec python3 -m hrc_repair.run \
  --config configs/repair_m0_scattered_pure.yaml \
  --method repair --scenario nominal --seed 0 --official-rgb --dual-gripper \
  --episode-wall-timeout-s 3600 \
  --out runs/repair_m0_official_rgb_dual_corrected_direct_v1/nominal/seed_0000 "$@"
