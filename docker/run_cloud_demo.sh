#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

COMMANDER_VALUE="${COMMANDER_API_KEY:-${GEMINI_API_KEY:-${GOOGLE_API_KEY:-${OPENAI_API_KEY:-}}}}"
if [[ -z "${COMMANDER_VALUE}" ]]; then
  echo "[ERROR] COMMANDER_API_KEY is required." >&2
  exit 2
fi
export COMMANDER_API_KEY="${COMMANDER_VALUE}"

if [[ -z "${PUBLIC_IP:-}" ]]; then
  echo "[ERROR] PUBLIC_IP is required for cloud WebRTC streaming." >&2
  exit 2
fi

export LIVESTREAM=1
export HEADLESS=1
export ENABLE_CAMERAS=1
export HAC_PREFER_LOCAL_SCENE_ASSETS="${HAC_PREFER_LOCAL_SCENE_ASSETS:-1}"
export HAC_AUTO_DOWNLOAD_SCENE_ASSETS="${HAC_AUTO_DOWNLOAD_SCENE_ASSETS:-0}"
export HAC_ENABLE_ROOM_SHELL_FALLBACK="${HAC_ENABLE_ROOM_SHELL_FALLBACK:-0}"

echo "[CLOUD] Starting public WebRTC stream for ${PUBLIC_IP}:49100 (TCP) / 47998 (UDP)." >&2
echo "[CLOUD] Restrict streaming ports and the web viewer to your client IP." >&2

exec "${REPO_ROOT}/run_main.sh" \
  --livestream 1 \
  --enable_cameras \
  --rendering_mode performance \
  "$@"
