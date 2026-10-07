#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT_DIR/tools/run_repair_m0_official_rgb.sh" \
  --transport-direct \
  --out runs/repair_m0_official_rgb_direct_transport/nominal/seed_0000 "$@"
