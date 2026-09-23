#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
lingbot_repo=${LINGBOT_VA_REPO:-/home/sunsiliang/Downloads/lingbot-va}
lingbot_env=${LINGBOT_VA_ENV:-lingbot-va}
model_path=${LINGBOT_VA_MODEL:-/home/sunsiliang/models/lingbot-va-base}
input_dir=${LINGBOT_VA_INPUT_DIR:-"${lingbot_repo}/example/franka"}
save_root=${LINGBOT_VA_OUTPUT:-"${repo_root}/pilot_12pair/outputs/lingbot_va_base_i2va_smoke"}
gpu=${LINGBOT_VA_GPU:-0}
prompt=${LINGBOT_VA_PROMPT:-"Pick up the object, place it on the assembly fixture, and finish the demonstrated manipulation."}
chunks=${LINGBOT_VA_NUM_CHUNKS:-1}
video_steps=${LINGBOT_VA_VIDEO_STEPS:-5}
action_steps=${LINGBOT_VA_ACTION_STEPS:-10}
text_max_length=${LINGBOT_VA_TEXT_MAX_LENGTH:-512}
height=${LINGBOT_VA_HEIGHT:-224}
width=${LINGBOT_VA_WIDTH:-320}

if [[ ! -d "${lingbot_repo}" ]]; then
  echo "LingBot-VA checkout not found: ${lingbot_repo}" >&2
  exit 2
fi

export PYTHONPATH="${lingbot_repo}:${repo_root}:${PYTHONPATH:-}"
export LINGBOT_VA_REPO="${lingbot_repo}"
export CUDA_VISIBLE_DEVICES="${gpu}"

exec conda run --no-capture-output -n "${lingbot_env}" \
  python -m torch.distributed.run --nproc_per_node=1 \
  "${repo_root}/pilot_12pair/lingbot_va/run_base_i2va.py" \
  --model-path "${model_path}" \
  --input-dir "${input_dir}" \
  --save-root "${save_root}" \
  --prompt "${prompt}" \
  --num-chunks "${chunks}" \
  --video-steps "${video_steps}" \
  --action-steps "${action_steps}" \
  --text-max-length "${text_max_length}" \
  --height "${height}" \
  --width "${width}" \
  --gpu 0
