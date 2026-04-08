# Baseline Euler (Stock GR00T N1.6)

**Control group** -- the unmodified 4-step Euler denoising from the GR00T N1.6
checkpoint. All other strategies in `DENOISING_STRATEGIES.md` are compared
against this baseline.

## What it does

- 4 Euler steps at `dt = 0.25`, timestep schedule `0 -> 250 -> 500 -> 750`
- Each step: DiT predicts velocity, `actions = actions + 0.25 * velocity`
- No guidance, no schedule changes, no early stopping
- 4 NFEs (neural function evaluations) per action chunk

## How to run

From the **repo root**:

```bash
# Terminal 1 (model venv) -- start the standard server
bash scripts/denoising_lab/eval/strategies/baseline_euler/run_server.sh

# Terminal 2 (sim venv) -- run the reproducible benchmark
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n-episodes 10 --seed 42 \
    --output-dir /tmp/benchmark_results/baseline_euler \
    --strategy-name baseline_euler
```

## Expected characteristics

| Metric | Expected |
|--------|----------|
| NFEs per chunk | 4 |
| Inference latency | ~64 ms / chunk (single A100) |
| Success rate | Env-dependent; 10-60% on typical RoboCasa tasks |
| Action horizon | 16 timesteps (Panda), first 8 executed |

## Notes

- This is the reference point. When evaluating a new strategy, run both
  `baseline_euler` and the new strategy with the same `--seed` and compare
  `summary.json` outputs.
- The server uses `--verbose` to log denoising steps. Remove the flag for
  slightly lower latency in production runs.
