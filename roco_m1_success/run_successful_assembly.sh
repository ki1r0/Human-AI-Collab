#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"
runtime_dir="${ROCO_RUNTIME_DIR:-/home/sunsiliang/roco_runtime}"
gpu_device="${ROCO_GPU_DEVICE:-3}"
container_image="${ROCO_CONTAINER_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.0}"
phase="${ROCO_PHASE:-full}"
seed="${ROCO_SEED:-23}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${phase}_seed${seed}_$$"
output_dir="${ROCO_OUTPUT_DIR:-${script_dir}/runs/${run_id}}"

mkdir -p "${output_dir}"
echo "M1 output directory: ${output_dir}"

docker run --rm \
  --entrypoint /isaac-sim/python.sh \
  --gpus "device=${gpu_device}" \
  --ipc=host \
  --network=host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONUNBUFFERED=1 \
  -e "PYTHONPATH=${script_dir}:${runtime_dir}/gearboxAssembly/source/Galaxea_Lab_External" \
  -v "${workspace_dir}:${workspace_dir}" \
  -v "${runtime_dir}:${runtime_dir}" \
  "${container_image}" \
  "${script_dir}/run_episode.py" \
  --headless --enable_cameras \
  --phase "${phase}" \
  --seed "${seed}" \
  --output-dir "${output_dir}" \
  "$@" 2>&1 | tee "${output_dir}/console.log"
