#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

load_env_defaults() {
  local env_file="${1:-}"
  [[ -f "${env_file}" ]] || return 0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#export }"
    [[ -n "${line// /}" ]] || continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    [[ "${line}" == *=* ]] || continue
    local key="${line%%=*}"
    local value="${line#*=}"
    key="$(printf '%s' "${key}" | xargs)"
    [[ -n "${key}" ]] || continue
    if [[ -z "${!key+x}" ]]; then
      export "${key}=${value}"
    fi
  done < "${env_file}"
}

for env_file in \
  "${SCRIPT_DIR}/config/runtime_env.env" \
  "${SCRIPT_DIR}/config/runtime_env.local.env"
do
  load_env_defaults "${env_file}"
done

for shell_file in \
  "${SCRIPT_DIR}/config/runtime_env.sh" \
  "${SCRIPT_DIR}/config/runtime_env.local.sh"
do
  if [[ -f "${shell_file}" ]]; then
    # shellcheck disable=SC1090
    source "${shell_file}"
  fi
done

find_launcher() {
  local candidate=""
  for candidate in \
    "${ISAACLAB_LAUNCHER:-}" \
    "${ISAACLAB_ROOT:-}/isaaclab.sh" \
    "/isaac-sim/isaaclab.sh" \
    "${SCRIPT_DIR}/../isaaclab.sh" \
    "${SCRIPT_DIR}/../../isaaclab.sh"
  do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    printf '%s\n' "${candidate}"
    return 0
  done
  if command -v isaaclab.sh >/dev/null 2>&1; then
    command -v isaaclab.sh
    return 0
  fi
  return 1
}

LAUNCHER="$(find_launcher || true)"
if [[ -z "${LAUNCHER}" ]]; then
  echo "[ERROR] Could not find isaaclab.sh. Set ISAACLAB_LAUNCHER=/path/to/isaaclab.sh or ISAACLAB_ROOT=/path/to/IsaacLab." >&2
  exit 1
fi

exec "${LAUNCHER}" -p "${SCRIPT_DIR}/main.py" --usd "${SCRIPT_DIR}/assets/simple_room_scene.usd" "$@"
