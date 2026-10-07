#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"
runtime_dir="${ROCO_RUNTIME_DIR:-/home/sunsiliang/roco_runtime}"
gpu_device="${ROCO_GPU_DEVICE:-3}"
container_image="${ROCO_CONTAINER_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.0}"
seed="${ROCO_SEED:-23}"
headless="${ROCO_HEADLESS:-1}"
livestream="${ROCO_LIVESTREAM:-0}"

checkpoint="${ROCO_CHECKPOINT:-${runtime_dir}/checkpoints/roco_model_act_2/policy_best.ckpt}"
stats="${ROCO_STATS:-${runtime_dir}/checkpoints/roco_model_act_2/dataset_stats.pkl}"
checkpoint_sha256="${ROCO_CHECKPOINT_SHA256:-a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1}"
stats_sha256="${ROCO_STATS_SHA256:-4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e}"
candidate_id="${ROCO_CANDIDATE_ID:-yjsm1203/roco_model_act_2}"
candidate_revision="${ROCO_CANDIDATE_REVISION:-52344a203e0739638cb2c7b11ea632e7b2eb2608}"
seed_sequence="${ROCO_SEED_SEQUENCE:-23,17,42,2026}"
integration_git_commit="$(git -C "${workspace_dir}" rev-parse HEAD)"
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
docker_args=(
    --rm
    --entrypoint /isaac-sim/python.sh
    --gpus "device=${gpu_device}"
    --ipc=host
    --network=host
)
app_args=(--headless --enable_cameras)
if [[ "${livestream}" == "1" ]]; then
    if [[ "${headless}" == "0" ]]; then
        echo "Choose either ROCO_LIVESTREAM=1 or ROCO_HEADLESS=0, not both." >&2
        exit 2
    fi
    app_args=(--enable_cameras --livestream 2)
elif [[ "${livestream}" != "0" ]]; then
    echo "ROCO_LIVESTREAM must be 0 or 1." >&2
    exit 2
fi
if [[ "${headless}" == "0" ]]; then
    if [[ -z "${DISPLAY:-}" ]] || [[ ! -d /tmp/.X11-unix ]]; then
        echo "Visible mode requires DISPLAY and /tmp/.X11-unix." >&2
        exit 2
    fi
    if [[ -z "${XAUTHORITY:-}" ]] || [[ ! -f "${XAUTHORITY}" ]]; then
        echo "Visible mode requires XAUTHORITY to name a readable X11 authority file." >&2
        exit 2
    fi
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
    minimum_gui_driver="580.65.06"
    if [[ "$(printf '%s\n%s\n' "${minimum_gui_driver}" "${driver_version}" | sort -V | head -n 1)" != "${minimum_gui_driver}" ]]; then
        echo "Native Isaac Sim 5.1 GUI is unsupported on NVIDIA driver ${driver_version}." >&2
        echo "NVIDIA's tested Linux driver is ${minimum_gui_driver}; use ROCO_LIVESTREAM=1 or upgrade the host driver." >&2
        exit 2
    fi
    docker_args+=(
        -e "DISPLAY=${DISPLAY}"
        -e XAUTHORITY=/tmp/roco.Xauthority
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
        -v "${XAUTHORITY}:/tmp/roco.Xauthority:ro"
    )
    app_args=(--enable_cameras)
fi

docker run "${docker_args[@]}" \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e PYTHONUNBUFFERED=1 \
    -e TORCH_HOME="${runtime_dir}/torch_cache" \
    -e PYTHONPATH="${script_dir}:${act_dir}:${galaxea_source}" \
    -v "${workspace_dir}:${workspace_dir}" \
    -v "${runtime_dir}:${runtime_dir}" \
    "${container_image}" \
    "${script_dir}/scripts/run_roco_single_episode.py" \
    "${app_args[@]}" \
    --checkpoint "${checkpoint}" \
    --stats "${stats}" \
    --expected-checkpoint-sha256 "${checkpoint_sha256}" \
    --expected-stats-sha256 "${stats_sha256}" \
    --candidate-id "${candidate_id}" \
    --candidate-revision "${candidate_revision}" \
    --seed-sequence "${seed_sequence}" \
    --integration-git-commit "${integration_git_commit}" \
    --seed "${seed}" \
    --output-dir "${output_dir}" \
    "$@" 2>&1 | tee "${output_dir}/run.log"

# Conventional mission filenames are relative symlinks to the canonical
# artifacts; the large video and trace are not duplicated.
ln -s result.json "${output_dir}/run_manifest.json"
ln -s result.json "${output_dir}/metrics.json"
ln -s run.log "${output_dir}/console.log"
ln -s episode.mp4 "${output_dir}/video.mp4"
ln -s trace.npz "${output_dir}/action_trace.npz"
