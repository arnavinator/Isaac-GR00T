#!/usr/bin/env bash
# Launch the GR00T N1.6 server with convergence-gated iterative refinement.
# Run from the repo root in the main (model) venv.
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/convergence_gated_iterative_refinement/run_server.sh
#   bash scripts/denoising_lab/eval/strategies/convergence_gated_iterative_refinement/run_server.sh --verbose
#   bash scripts/denoising_lab/eval/strategies/convergence_gated_iterative_refinement/run_server.sh --theta 0.3 --K-max 4
set -euo pipefail

uv run python scripts/denoising_lab/eval/strategies/convergence_gated_iterative_refinement/run_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper \
    --port 5555 \
    "$@"
