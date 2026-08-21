#!/usr/bin/env bash
# tools/run_tool.sh — run a standalone Python tool (e.g. scatter_parts.py)
# with Isaac Sim's USD (pxr) bindings available.
#
# Usage (typically via docker compose exec):
#   docker compose exec hac tools/run_tool.sh tools/scatter_parts.py --dry-run
#   docker compose exec hac tools/run_tool.sh tools/scatter_parts.py
#
# Why this exists:
#   `isaaclab.sh -p script.py` and `python.sh script.py` both invoke Isaac Sim's
#   Python, but neither puts the `pxr` USD bindings on PYTHONPATH unless a Kit
#   app is started — pxr ships as the `omni.usd.libs` Omniverse extension,
#   which Kit loads at runtime. Offline tools that only edit USD files (no Kit
#   needed) hit `ModuleNotFoundError: No module named 'pxr'` without this setup.
#
# What this wrapper does:
#   1. Globs /isaac-sim/extscache (and fallbacks) for `omni.usd.libs-*`.
#   2. Prepends the extension dir to PYTHONPATH and `<dir>/bin` to
#      LD_LIBRARY_PATH so libusd_*.so resolves before Python starts.
#      (LD_LIBRARY_PATH must be set pre-exec; setting it from inside Python
#      does not help — the dynamic linker reads it at process startup.)
#   3. Locates an `isaaclab.sh` (preferred) or falls back to Isaac Sim's
#      `python.sh`, and execs it with the script + remaining args.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <script.py> [args...]" >&2
  exit 2
fi

# Source repo env defaults (mirrors run_main.sh; harmless if files absent).
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
  "${REPO_ROOT}/config/runtime_env.env" \
  "${REPO_ROOT}/config/runtime_env.local.env"
do
  load_env_defaults "${env_file}"
done

# --- Locate the omni.usd.libs extension (provides `pxr`) ---------------------
find_usd_libs_dir() {
  local search_roots=(
    "${ISAACSIM_ROOT:-}/extscache"
    "${ISAACLAB_ROOT:-}/extscache"
    "/isaac-sim/extscache"
    "/workspace/isaaclab/_isaac_sim/extscache"
  )
  local root match
  for root in "${search_roots[@]}"; do
    [[ -n "${root}" && -d "${root}" ]] || continue
    # Latest version wins (lexicographic sort works for the +commit suffixes).
    match="$(ls -d "${root}"/omni.usd.libs-* 2>/dev/null | sort | tail -n1)"
    if [[ -n "${match}" && -d "${match}/pxr" && -d "${match}/bin" ]]; then
      printf '%s\n' "${match}"
      return 0
    fi
  done
  return 1
}

USD_LIBS_DIR="$(find_usd_libs_dir || true)"
if [[ -z "${USD_LIBS_DIR}" ]]; then
  {
    echo "[ERROR] Could not find omni.usd.libs extension under any of:"
    echo "          \$ISAACSIM_ROOT/extscache, \$ISAACLAB_ROOT/extscache,"
    echo "          /isaac-sim/extscache, /workspace/isaaclab/_isaac_sim/extscache"
    echo "        Set ISAACSIM_ROOT or run this inside the hac container."
  } >&2
  exit 1
fi

export PYTHONPATH="${USD_LIBS_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${USD_LIBS_DIR}/bin:${LD_LIBRARY_PATH:-}"

# --- Locate the Python launcher ----------------------------------------------
find_launcher() {
  local candidates=(
    "${ISAACLAB_LAUNCHER:-}"
    "${ISAACLAB_ROOT:-}/isaaclab.sh"
    "/workspace/isaaclab/isaaclab.sh"
    "/workspace/IsaacLab/isaaclab.sh"
    "/isaac-sim/isaaclab.sh"
  )
  local c
  for c in "${candidates[@]}"; do
    [[ -n "${c}" && -x "${c}" ]] && { printf '%s\n' "${c}"; return 0; }
  done
  # Fall back to Isaac Sim's bundled python.sh — offline USD tools don't need Kit.
  for c in \
    "/workspace/isaaclab/_isaac_sim/python.sh" \
    "/isaac-sim/python.sh"
  do
    [[ -x "${c}" ]] && { printf '%s\n' "${c}"; return 0; }
  done
  return 1
}

LAUNCHER="$(find_launcher || true)"
if [[ -z "${LAUNCHER}" ]]; then
  {
    echo "[ERROR] Could not find isaaclab.sh or python.sh."
    echo "        Set ISAACLAB_LAUNCHER=/path/to/isaaclab.sh or run inside the hac container."
  } >&2
  exit 1
fi

# isaaclab.sh uses `-p script.py args...`; python.sh uses `script.py args...`.
if [[ "$(basename "${LAUNCHER}")" == "isaaclab.sh" ]]; then
  exec "${LAUNCHER}" -p "$@"
else
  exec "${LAUNCHER}" "$@"
fi
