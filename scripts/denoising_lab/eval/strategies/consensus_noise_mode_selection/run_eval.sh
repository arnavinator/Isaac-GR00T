#!/usr/bin/env bash
# Run the reproducible benchmark for the consensus noise mode selection strategy.
# Run from the repo root. Requires the server to be running (see run_server.sh).
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/consensus_noise_mode_selection/run_eval.sh
#   bash scripts/denoising_lab/eval/strategies/consensus_noise_mode_selection/run_eval.sh --n-episodes 50
set -euo pipefail

gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --n-episodes 15 --seed 42 \
    --n-envs 2 --port 5555 \
    --max-episode-steps 480 \
    --output-dir scripts/denoising_lab/eval/benchmark_results/consensus_noise_mode_selection \
    --strategy-name consensus_noise_mode_selection \
    "$@"
