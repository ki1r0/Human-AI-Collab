#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

COMMANDER_VALUE="${COMMANDER_API_KEY:-${GEMINI_API_KEY:-${GOOGLE_API_KEY:-${OPENAI_API_KEY:-}}}}"
if [[ -z "${COMMANDER_VALUE}" ]]; then
  echo "[ERROR] COMMANDER_API_KEY is required. Set COMMANDER_API_KEY (or GEMINI_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY) in your environment or .env before launching." >&2
  exit 2
fi
export COMMANDER_API_KEY="${COMMANDER_VALUE}"

if [[ -z "${COSMOS_BASE_URL:-}${COSMOS_CHAT_COMPLETIONS_URL:-}${COSMOS_HOST:-}" ]]; then
  echo "[INFO] Cosmos is not configured. The pipeline will bypass Cosmos and continue with the commander-only path." >&2
fi

export HAC_PREFER_LOCAL_SCENE_ASSETS="${HAC_PREFER_LOCAL_SCENE_ASSETS:-1}"
export HAC_AUTO_DOWNLOAD_SCENE_ASSETS="${HAC_AUTO_DOWNLOAD_SCENE_ASSETS:-0}"
export HAC_ENABLE_ROOM_SHELL_FALLBACK="${HAC_ENABLE_ROOM_SHELL_FALLBACK:-0}"

# Optional pre-step: rewrite scatter layout into simple_room_scene.usd before
# launching the sim. Off by default so normal `docker compose up hac` is a
# pure scene load. Set HAC_RESCATTER=1 to refresh part poses (e.g. after
# editing PART_POSITIONS, or for diversity tests that randomize the layout).
# HAC_RESCATTER=dry runs scatter_parts.py with --dry-run (no scene mutation).
if [[ -n "${HAC_RESCATTER:-}" && "${HAC_RESCATTER}" != "0" ]]; then
  SCATTER_ARGS=()
  if [[ "${HAC_RESCATTER}" == "dry" || "${HAC_RESCATTER}" == "dry-run" ]]; then
    SCATTER_ARGS+=("--dry-run")
    echo "[INFO] HAC_RESCATTER=${HAC_RESCATTER}: running scatter_parts.py --dry-run (scene NOT modified)" >&2
  else
    echo "[INFO] HAC_RESCATTER=${HAC_RESCATTER}: rewriting scatter layout in simple_room_scene.usd" >&2
  fi
  "${REPO_ROOT}/tools/run_tool.sh" tools/scatter_parts.py "${SCATTER_ARGS[@]}"
fi

exec "${REPO_ROOT}/run_main.sh" "$@"
