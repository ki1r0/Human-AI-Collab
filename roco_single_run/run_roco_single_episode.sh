#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"
runtime_dir="${ROCO_RUNTIME_DIR:-/home/sunsiliang/roco_runtime}"
gpu_device="${ROCO_GPU_DEVICE:-3}"
container_image="${ROCO_CONTAINER_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.0}"
seed="${ROCO_SEED:-23}"

checkpoint="${ROCO_CHECKPOINT:-${runtime_dir}/checkpoints/roco_model_act_2/policy_best.ckpt}"
stats="${ROCO_STATS:-${runtime_dir}/checkpoints/roco_model_act_2/dataset_stats.pkl}"
checkpoint_sha256="${ROCO_CHECKPOINT_SHA256:-a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1}"
stats_sha256="${ROCO_STATS_SHA256:-4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e}"
candidate_id="${ROCO_CANDIDATE_ID:-yjsm1203/roco_model_act_2}"
candidate_revision="${ROCO_CANDIDATE_REVISION:-52344a203e0739638cb2c7b11ea632e7b2eb2608}"
seed_sequence="${ROCO_SEED_SEQUENCE:-23,17,42,2026}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)_seed${seed}_$$"
output_dir="${ROCO_OUTPUT_DIR:-${script_dir}/runs/${run_id}}"

checkout="${runtime_dir}/gearboxAssembly"
act_dir="${checkout}/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT"
galaxea_source="${checkout}/source/Galaxea_Lab_External"

for required in "${checkout}/.git" "${act_dir}/act/policy.py" "${checkpoint}" "${stats}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required RoCo artifact is missing: ${required}" >&2
        exit 2
    fi
done

python3 "${script_dir}/scripts/prepare_official_checkout.py" "${checkout}"
if [[ -e "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to overwrite non-empty output directory: ${output_dir}" >&2
    exit 2
fi
mkdir -p "${output_dir}"

echo "RoCo output directory: ${output_dir}"
docker run --rm \
    --entrypoint /isaac-sim/python.sh \
    --gpus "device=${gpu_device}" \
    --ipc=host \
    --network=host \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e PYTHONUNBUFFERED=1 \
    -e TORCH_HOME="${runtime_dir}/torch_cache" \
    -e PYTHONPATH="${script_dir}:${act_dir}:${galaxea_source}" \
    -v "${workspace_dir}:${workspace_dir}" \
    -v "${runtime_dir}:${runtime_dir}" \
    "${container_image}" \
    "${script_dir}/scripts/run_roco_single_episode.py" \
    --headless \
    --enable_cameras \
    --checkpoint "${checkpoint}" \
    --stats "${stats}" \
    --expected-checkpoint-sha256 "${checkpoint_sha256}" \
    --expected-stats-sha256 "${stats_sha256}" \
    --candidate-id "${candidate_id}" \
    --candidate-revision "${candidate_revision}" \
    --seed-sequence "${seed_sequence}" \
    --seed "${seed}" \
    --output-dir "${output_dir}" \
    "$@" 2>&1 | tee "${output_dir}/run.log"

# Conventional mission filenames are hard links to the canonical artifacts;
# the large video and trace are not duplicated.
ln "${output_dir}/result.json" "${output_dir}/run_manifest.json"
ln "${output_dir}/result.json" "${output_dir}/metrics.json"
ln "${output_dir}/run.log" "${output_dir}/console.log"
ln "${output_dir}/episode.mp4" "${output_dir}/video.mp4"
ln "${output_dir}/trace.npz" "${output_dir}/action_trace.npz"
