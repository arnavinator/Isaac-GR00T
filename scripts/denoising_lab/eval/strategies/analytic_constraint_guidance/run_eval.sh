#!/usr/bin/env bash
# Run the reproducible benchmark for the analytic constraint guidance strategy.
# Run from the repo root. Requires the server to be running (see run_server.sh).
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/run_eval.sh
#   bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/run_eval.sh --n-episodes 50
set -euo pipefail

gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --n-episodes 50 --seed 100 \
    --n-envs 1 --port 5556 \
    --max-episode-steps 480 \
    --output-dir scripts/denoising_lab/eval/4run_benchmark_results/analytic_constraint_guidance_3 \
    --strategy-name analytic_constraint_guidance \
    "$@"
