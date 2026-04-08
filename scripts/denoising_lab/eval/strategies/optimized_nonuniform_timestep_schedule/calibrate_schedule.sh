#!/usr/bin/env bash
# Calibrate the optimal non-uniform timestep schedule.
# Run from the repo root in the main (model) venv.
#
# This starts a temporary baseline server, spawns the sim-venv to collect
# observations, then runs the grid search.  Everything in one command.
#
# Usage:
#   bash scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/calibrate_schedule.sh
#
#   # Custom env / episodes:
#   bash .../calibrate_schedule.sh \
#       --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
#                   robocasa_panda_omron/CloseDrawer_PandaOmron_Env \
#       --n-episodes 10 --n-candidates 2000
#
#   # Skip collection (use pre-saved observations):
#   bash .../calibrate_schedule.sh --obs-dir /tmp/my_obs --n-candidates 2000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run python "$SCRIPT_DIR/calibrate_schedule.py" \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n-episodes 5 \
    --seed 42 \
    --output-dir /tmp/schedule_calibration \
    "$@"
