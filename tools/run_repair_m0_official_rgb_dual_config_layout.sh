#!/usr/bin/env bash
set -euo pipefail

: "${HRC_REPAIR_PLANNER_ENDPOINT:?Set the OpenAI-compatible Qwen planner endpoint}"
: "${HRC_REPAIR_API_KEY:?Set the key expected by the local Qwen endpoint}"
export HRC_REPAIR_VLM_ENDPOINT="${HRC_REPAIR_VLM_ENDPOINT:-$HRC_REPAIR_PLANNER_ENDPOINT}"
export HRC_REPAIR_VLM_MODEL="${HRC_REPAIR_VLM_MODEL:-${HRC_REPAIR_PLANNER_MODEL:-Qwen/Qwen3-VL-2B-Instruct}}"

# Reproduce the config's full-physics scattered reset without per-run pose overrides.
exec python3 -m hrc_repair.run \
  --config configs/repair_m0_scattered_pure.yaml \
  --method repair --scenario nominal --seed 0 --official-rgb --dual-gripper \
  --episode-wall-timeout-s 3600 \
  --out runs/repair_m0_official_rgb_dual_config_layout/nominal/seed_0000 "$@"
