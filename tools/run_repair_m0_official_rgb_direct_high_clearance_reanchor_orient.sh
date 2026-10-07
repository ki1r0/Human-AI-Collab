#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT_DIR/tools/run_repair_m0_official_rgb.sh" \
  --transport-direct --transport-clearance-z-m 0.40 \
  --reanchor-at-insertion-above --correct-orientation-at-insertion-above \
  --out runs/repair_m0_official_rgb_direct_high_clearance_reanchor_orient/nominal/seed_0000 "$@"
