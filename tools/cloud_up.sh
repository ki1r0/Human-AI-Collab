#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.cloud"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.cloud.yml"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${PROJECT_ROOT}/.env.cloud.example" "${ENV_FILE}"
  echo "Created ${ENV_FILE}. Set PUBLIC_IP and COMMANDER_API_KEY, then run this script again." >&2
  exit 2
fi

bash "${SCRIPT_DIR}/cloud_preflight.sh"

cd "${PROJECT_ROOT}"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up --build -d

public_ip="$(sed -n 's/^PUBLIC_IP=//p' "${ENV_FILE}" | tail -n 1)"

echo
echo "Human-AI-Collab cloud stack is starting."
echo "Viewer: http://${public_ip}:8210"
echo "Logs:   docker compose --env-file .env.cloud -f docker-compose.cloud.yml logs -f hac-cloud"
echo
echo "The first launch can take several minutes while images and shader caches are prepared."
