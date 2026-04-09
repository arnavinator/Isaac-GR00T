#!/usr/bin/env bash
# Launch the GR00T N1.6 server with receding-horizon warm-start denoising.
# Run from the repo root in the main (model) venv.
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_server.sh
#   bash scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_server.sh --port 5556
#   bash scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_server.sh --tau-start 0.5 --n-executed 4
set -euo pipefail

uv run python scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper \
    --port 5555 \
    "$@"
