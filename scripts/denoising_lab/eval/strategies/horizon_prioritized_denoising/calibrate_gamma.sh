#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

# PYTHONUNBUFFERED ensures we see progress in real time over SSH
exec env PYTHONUNBUFFERED=1 uv run python "$SCRIPT_DIR/calibrate_gamma.py" \
    --obs-dir "${OBS_DIR:-/tmp/schedule_calibration/observations}" \
    --output-dir "${OUTPUT_DIR:-/tmp/gamma_calibration}" \
    "$@"
