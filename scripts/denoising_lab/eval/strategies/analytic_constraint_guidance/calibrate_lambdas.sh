#!/usr/bin/env bash
# Grid search for constraint guidance lambda hyperparameters.
#
# Focuses on universally-hard seeds (45, 52 from OpenDrawer; 54, 55, 56 from
# CoffeeServeMug) to find lambda configs that flip failures to successes.
#
# The script loads the model ONCE and iterates over a 3x3x3 grid of lambda
# values, re-patching the action head between evaluations.
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/calibrate_lambdas.sh
#
#   # Override grid values:
#   bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/calibrate_lambdas.sh \
#       --lambda-smooth 0.001 0.005 0.02 --eta 0.05 0.1 0.2
#
#   # Run all 15 seeds (full comparison instead of hard-only screening):
#   bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/calibrate_lambdas.sh \
#       --seeds-only  # (pass empty to disable seed filtering)
set -euo pipefail

uv run python scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/calibrate_lambdas.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
                robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --max-episode-steps 400 480 \
    --n-episodes 15 --seed 42 \
    --seeds-only 45 52 54 55 56 \
    --lambda-smooth 0.002 0.005 0.01 \
    --lambda-discrete 0.005 0.01 0.02 \
    --lambda-mode 0.001 0.003 0.01 \
    --eta 0.1 \
    --n-envs 2 \
    --output-dir scripts/denoising_lab/eval/benchmark_results/analytic_constraint_guidance_calibration \
    "$@"
