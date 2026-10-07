#!/usr/bin/env bash
set -euo pipefail

: "${HRC_REPAIR_PLANNER_ENDPOINT:?Set the OpenAI-compatible Qwen planner endpoint}"
: "${HRC_REPAIR_API_KEY:?Set the key expected by the local Qwen endpoint}"
export HRC_REPAIR_VLM_ENDPOINT="${HRC_REPAIR_VLM_ENDPOINT:-$HRC_REPAIR_PLANNER_ENDPOINT}"
export HRC_REPAIR_VLM_MODEL="${HRC_REPAIR_VLM_MODEL:-${HRC_REPAIR_PLANNER_MODEL:-Qwen/Qwen3-VL-2B-Instruct}}"

exec python3 -m hrc_repair.run \
  --config configs/repair_m0_scattered_pure.yaml \
  --method repair --scenario nominal --seed 0 --official-rgb --dual-gripper \
  --cover-xy 0.43 0.0 --casing-xy 0.55 0.42 \
  --out runs/repair_m0_official_rgb_dual_transport/nominal/seed_0000 "$@"
