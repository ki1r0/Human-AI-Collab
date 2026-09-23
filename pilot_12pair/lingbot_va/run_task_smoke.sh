#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
task_id=${1:-HCF-01}
variant=${2:-HARD}
gpu=${LINGBOT_VA_GPU:-0}
task_tag=$(printf '%s_%s' "${task_id}" "${variant}" | tr '[:upper:]' '[:lower:]')
out_root="${LINGBOT_VA_TASK_OUTPUT:-${repo_root}/pilot_12pair/outputs/lingbot_va_${task_tag}}"
input_dir="${out_root}/synthetic_observation"

PYTHONPATH="${repo_root}" conda run --no-capture-output -n "${LINGBOT_VA_ENV:-lingbot-va}" \
  python -m pilot_12pair.lingbot_va.make_synthetic_observations \
  --task-id "${task_id}" --variant "${variant}" --out-dir "${input_dir}"

LINGBOT_VA_GPU="${gpu}" \
LINGBOT_VA_INPUT_DIR="${input_dir}" \
LINGBOT_VA_OUTPUT="${out_root}/model_output" \
LINGBOT_VA_PROMPT="Finish the assembly task shown in the observations. Use safe physical actions and stop only when both named parts are seated." \
"${repo_root}/pilot_12pair/lingbot_va/run_base_i2va.sh"
