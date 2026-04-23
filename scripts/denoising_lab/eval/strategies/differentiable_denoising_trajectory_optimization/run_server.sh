#!/usr/bin/env bash
# Launch the GR00T N1.6 server with differentiable denoising trajectory optimization (DDTO).
# Run from the repo root in the main (model) venv.
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/run_server.sh
#   bash scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/run_server.sh --port 5556
#   bash scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/run_server.sh --eta 0.2
set -euo pipefail

uv run python scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/run_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper \
    --port 5555 \
    "$@"
