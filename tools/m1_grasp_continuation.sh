#!/usr/bin/env bash
# Resumable host-side continuation for the corrected Hub-cover grasp sweep.
#
# This is intentionally a finite, stateful queue.  It does not fake a grasp,
# alter the Hub pose during an episode, or declare success from a target pose.
# If a terminal/tool/model quota interrupts the assistant, rerunning this
# script (or its user service) resumes from existing metrics.json files.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT}/validation_logs/m1_grasp_continuation"
STATE="${OUT_ROOT}/state.tsv"
LOG="${OUT_ROOT}/runner.log"
LOCK="${OUT_ROOT}/runner.lock"
IMAGE="nvcr.io/nvidia/isaac-lab:2.3.0"
ROCO="/home/sunsiliang/roco_runtime/gearboxAssembly"

mkdir -p "${OUT_ROOT}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another continuation runner is active" >&2
  exit 0
fi

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${LOG}"
}

run_candidate() {
  local name="$1" orientation="$2" offset="$3" opening="$4"
  local out="${OUT_ROOT}/${name}"
  local metrics="${out}/metrics.json"
  mkdir -p "${out}"
  if [[ -s "${metrics}" ]]; then
    log "SKIP ${name}: metrics already exists"
    return 0
  fi
  log "START ${name}: orientation=${orientation} offset=${offset} opening=${opening}"
  set +e
  docker run --rm --gpus 'device=0' --ipc=host --network=host \
    --entrypoint /isaac-sim/python.sh -w /workspace/Human-AI-Collab \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -e PYTHONUNBUFFERED=1 -e HRC_M1_ROOT=/workspace/Human-AI-Collab \
    -e ROCO_ROOT=/workspace/gearboxAssembly \
    -e PYTHONPATH=/workspace/Human-AI-Collab:/workspace/gearboxAssembly/source/Galaxea_Lab_External \
    -v "${ROOT}:/workspace/Human-AI-Collab" \
    -v "${ROCO}:/workspace/gearboxAssembly:ro" "${IMAGE}" \
    -m hrc_m1.debug_inner_wall_grasp \
    --output-dir "/workspace/Human-AI-Collab/validation_logs/m1_grasp_continuation/${name}" \
    --orientation "${orientation}" --radial-offset "${offset}" \
    --opening "${opening}" --approach-opening 0.045 \
    --z-offset 0.037 --gripper-contact-offset 0.0001 \
    --disable-fixture-collisions --gripper-only-collisions \
    --no-video --headless --enable_cameras >>"${LOG}" 2>&1
  local rc=$?
  set -e
  if [[ ${rc} -eq 0 && -s "${metrics}" ]]; then
    printf '%s\tDONE\t%s\n' "${name}" "${metrics}" >>"${STATE}"
    log "DONE ${name}"
  else
    printf '%s\tFAILED\t%s\n' "${name}" "${rc}" >>"${STATE}"
    log "FAILED ${name}: rc=${rc}; runner will continue"
  fi
}

# This queue is the corrected topology in an isolated contact diagnostic:
# both gripper links remain collision-enabled, while table/casing and other
# robot links are disabled only to identify viable two-sided clamps.  Existing
# artifacts are never overwritten, so this queue is safe to resume after a
# host/tool interruption.  A candidate is promoted to the full scene only
# after inspecting its contact/object-follow metrics; that promotion remains
# a separate, auditable run.
# Keep the queue explicit and reviewable.
while read -r name orientation offset opening; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  run_candidate "${name}" "${orientation}" "${offset}" "${opening}"
done <<'QUEUE'
# name orientation offset_m opening_m
radial_085_020 radial 0.085 0.020
radial_0975_020 radial 0.0975 0.020
radial_105_020 radial 0.105 0.020
radial_085_039 radial 0.085 0.039
radial_0975_039 radial 0.0975 0.039
radial_105_039 radial 0.105 0.039
radial_outward_0975_020 radial_outward 0.0975 0.020
radial_outward_105_020 radial_outward 0.105 0.020
QUEUE

log "QUEUE_COMPLETE: inspect metrics.json and candidate_verdict fields before promoting any run"
