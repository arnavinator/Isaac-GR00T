#!/usr/bin/env bash
# Launch the standard GR00T N1.6 server (baseline 4-step Euler denoising).
# Run from the repo root in the main (model) venv.
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/baseline_euler/run_server.sh
#   bash scripts/denoising_lab/eval/strategies/baseline_euler/run_server.sh --port 5556
set -euo pipefail

uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper \
    --port 5555 \
    "$@"
