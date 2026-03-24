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

exec "${REPO_ROOT}/run_main.sh" "$@"
