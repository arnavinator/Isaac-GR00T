#!/usr/bin/env bash
# Run the reproducible benchmark for the noise-space mode selection strategy.
# Run from the repo root. Requires the server to be running (see run_server.sh).
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/noise_space_mode_selection/run_eval.sh
#   bash scripts/denoising_lab/eval/strategies/noise_space_mode_selection/run_eval.sh --n-episodes 50
set -euo pipefail

gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n-episodes 15 --seed 42 \
    --n-envs 2 --port 5555 \
    --score-dims 12 --score-horizon 16 --noise-keyframes 0 \
    --lambda-smooth 0.1 --lambda-mag 0.0 --lambda-anchor 2.0
    --max-episode-steps 400 \
    --output-dir ~/my_Isaac-GR00T/scripts/denoising_lab/eval/benchmark_results/noise_space_mode_selection_pc \
    --strategy-name noise_space_mode_selection \
    "$@"
