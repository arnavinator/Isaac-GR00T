#!/usr/bin/env bash
# Launch the GR00T N1.6 server with optimized non-uniform timestep schedule.
# Run from the repo root in the main (model) venv.
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/run_server.sh
#   bash scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/run_server.sh --port 5556
#   bash scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/run_server.sh --schedule 0.0 0.1 0.4 0.85
set -euo pipefail

uv run python scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/run_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper \
    --verbose \
    "$@"
