#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.cloud"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.cloud.yml"

failures=0
warnings=0

pass() {
  printf '[PASS] %s\n' "$1"
}

warn() {
  printf '[WARN] %s\n' "$1"
  warnings=$((warnings + 1))
}

fail() {
  printf '[FAIL] %s\n' "$1"
  failures=$((failures + 1))
}

env_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "${ENV_FILE}" 2>/dev/null | tail -n 1
}

echo "Human-AI-Collab cloud preflight"
echo

if [[ "$(uname -s)" == "Linux" ]]; then
  pass "Linux host detected"
else
  fail "Cloud deployment requires a Linux host"
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  pass "Docker engine is available"
else
  fail "Docker engine is unavailable"
fi

if docker compose version >/dev/null 2>&1; then
  pass "Docker Compose is available"
else
  fail "Docker Compose is unavailable"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  pass "NVIDIA GPU detected: ${gpu_name} (${vram_mib} MiB VRAM)"
  if [[ "${vram_mib:-0}" -lt 16000 ]]; then
    warn "GPU VRAM is below NVIDIA's 16 GB minimum recommendation"
  fi
else
  fail "NVIDIA GPU is unavailable"
fi

memory_mib="$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)"
if [[ "${memory_mib:-0}" -ge 32000 ]]; then
  pass "System memory is ${memory_mib} MiB"
else
  warn "System memory is ${memory_mib} MiB; NVIDIA recommends at least 32 GB"
fi

if [[ -f "${ENV_FILE}" ]]; then
  pass ".env.cloud exists"
else
  fail "Missing .env.cloud; copy it from .env.cloud.example"
fi

public_ip="$(env_value PUBLIC_IP)"
if [[ -n "${public_ip}" ]]; then
  pass "PUBLIC_IP is configured"
else
  fail "PUBLIC_IP is missing in .env.cloud"
fi

commander_key="$(env_value COMMANDER_API_KEY)"
if [[ -n "${commander_key}" ]]; then
  pass "COMMANDER_API_KEY is configured"
else
  fail "COMMANDER_API_KEY is missing in .env.cloud"
fi

base_image="$(env_value ISAACLAB_BASE_IMAGE)"
if [[ -n "${base_image}" ]]; then
  pass "Base image configured: ${base_image}"
  if docker manifest inspect "${base_image}" >/dev/null 2>&1; then
    pass "Base image manifest is accessible"
  else
    fail "Cannot access ${base_image}; run docker login nvcr.io or choose an accessible image"
  fi
else
  fail "ISAACLAB_BASE_IMAGE is missing in .env.cloud"
fi

if docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" config --quiet >/dev/null 2>&1; then
  pass "Cloud Compose configuration is valid"
else
  fail "Cloud Compose configuration is invalid"
fi

echo
printf 'Summary: %s failure(s), %s warning(s)\n' "${failures}" "${warnings}"
exit "${failures}"
