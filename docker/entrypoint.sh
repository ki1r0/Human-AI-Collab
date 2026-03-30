#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export HAC_REPO_ROOT="${REPO_ROOT}"

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

for shell_file in \
  "${REPO_ROOT}/config/runtime_env.sh" \
  "${REPO_ROOT}/config/runtime_env.local.sh"
do
  if [[ -f "${shell_file}" ]]; then
    # shellcheck disable=SC1090
    source "${shell_file}"
  fi
done

export ISAACLAB_ROOT="${ISAACLAB_ROOT:-/isaac-sim}"

prepend_pythonpath_if_dir() {
  local dir="${1:-}"
  [[ -n "${dir}" && -d "${dir}" ]] || return 0
  case ":${PYTHONPATH:-}:" in
    *":${dir}:"*) ;;
    *)
      if [[ -n "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="${dir}:${PYTHONPATH}"
      else
        export PYTHONPATH="${dir}"
      fi
      ;;
  esac
}

prepend_pythonpath_if_dir "/opt/hac-python"
prepend_pythonpath_if_dir "/isaac-sim/python_packages"

if [[ -z "${ISAACLAB_LAUNCHER:-}" ]]; then
  for candidate in \
    "/isaac-sim/isaaclab.sh" \
    "/workspace/isaaclab/isaaclab.sh" \
    "/workspace/IsaacLab/isaaclab.sh"
  do
    if [[ -x "${candidate}" ]]; then
      export ISAACLAB_LAUNCHER="${candidate}"
      break
    fi
  done
fi

if [[ $# -eq 0 ]]; then
  set -- "${REPO_ROOT}/docker/run_demo.sh"
fi

exec "$@"
