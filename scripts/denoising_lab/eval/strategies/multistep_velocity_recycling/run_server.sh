#!/usr/bin/env bash
# Launch the GR00T N1.6 server with AB2 velocity recycling denoising.
# Run from the repo root in the main (model) venv.
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/multistep_velocity_recycling/run_server.sh
#   bash scripts/denoising_lab/eval/strategies/multistep_velocity_recycling/run_server.sh --port 5556
set -euo pipefail

uv run python scripts/denoising_lab/eval/strategies/multistep_velocity_recycling/run_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper \
    --verbose \
    "$@"
