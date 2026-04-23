#!/usr/bin/env bash
# Run the reproducible benchmark for the convergence-gated iterative refinement strategy.
# Run from the repo root. Requires the server to be running (see run_server.sh).
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/convergence_gated_iterative_refinement/run_eval.sh
#   bash scripts/denoising_lab/eval/strategies/convergence_gated_iterative_refinement/run_eval.sh --n-episodes 50
set -euo pipefail

gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
               robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --n-episodes 15 --seed 42 \
    --n-envs 2 --port 5555 \
    --max-episode-steps 400 480 \
    --output-dir ~/my_Isaac-GR00T/scripts/denoising_lab/eval/benchmark_results/convergence_gated_iterative_refinement \
    --strategy-name convergence_gated_iterative_refinement \
    "$@"
