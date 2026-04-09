#!/usr/bin/env bash
# Run the reproducible benchmark for the optimized non-uniform timestep schedule strategy.
# Run from the repo root. Requires the server to be running (see run_server.sh).
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/run_eval.sh
#   bash scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/run_eval.sh --n-episodes 50
set -euo pipefail

gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n-episodes 10 --seed 42 \
    --n-envs 2 --port 5555 \
    --max-episode-steps 480 \
    --output-dir /tmp/benchmark_results/optimized_nonuniform_timestep_schedule \
    --strategy-name optimized_nonuniform_timestep_schedule \
    "$@"
