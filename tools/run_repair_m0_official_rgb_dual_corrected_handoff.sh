#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT_DIR/tools/run_repair_m0_official_rgb_dual_orientation_reanchor_v5.sh" \
  --release-right-at-insertion-above \
  --out runs/repair_m0_official_rgb_dual_corrected_handoff/nominal/seed_0000 "$@"
