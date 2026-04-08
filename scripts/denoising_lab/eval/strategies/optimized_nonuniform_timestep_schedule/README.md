## Strategy 2: Optimized Non-Uniform Timestep Schedule

**Category:** Drop-in replacement | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

GR00T's 4 denoising steps are evenly spaced at $\tau \in \{0.00, 0.25, 0.50, 0.75\}$. But the velocity field $v(a_t^\tau, \tau, o_t, l_t)$ is not uniformly complex across $\tau$ — it changes rapidly near $\tau = 0$ (where noise dominates and gross structure must emerge) and near $\tau = 1$ (where fine details are resolved), but is relatively smooth in between. By concentrating steps where the velocity field changes fastest, we can reduce discretization error *for free*.

This is NVIDIA's own insight: the **Align Your Steps** paper from NVIDIA Research demonstrates that optimizing timestep placement can dramatically improve few-step sampling quality.

### Mathematical Formulation

Instead of uniform spacing, use an optimized schedule $\{\tau_0, \tau_1, \tau_2, \tau_3\}$ with corresponding step sizes $\Delta\tau_i = \tau_{i+1} - \tau_i$ (where $\tau_4 = 1.0$):

$$a_t^{\tau_{i+1}} = a_t^{\tau_i} + \Delta\tau_i \cdot v(a_t^{\tau_i},\; \tau_i,\; o_t,\; l_t)$$

The schedule $\{\tau_0, \tau_1, \tau_2, \tau_3\}$ is found by minimizing the expected discretization error over a validation set:

$$\{\tau_i^*\} = \arg\min_{\{\tau_i\}} \; \mathbb{E}_{o_t, l_t}\left[\left\| a_{t,\text{fine}}^1 - a_{t,\text{coarse}}^1(\{\tau_i\}) \right\|^2\right]$$

where $a_{t,\text{fine}}^1$ is a high-fidelity reference (e.g., 64-step Euler) and $a_{t,\text{coarse}}^1(\{\tau_i\})$ is the 4-step result with the candidate schedule.

**Example hypothetical schedule** (to be determined empirically):

| | Uniform (current) | Optimized (hypothetical) |
|-|-------------------|--------------------------|
| $\tau_0$ | 0.000 | 0.000 |
| $\tau_1$ | 0.250 | 0.080 |
| $\tau_2$ | 0.500 | 0.350 |
| $\tau_3$ | 0.750 | 0.820 |
| $\tau_4$ (target) | 1.000 | 1.000 |

This concentrates 2 steps in the early phase ($\tau < 0.35$) where coarse structure emerges, and 2 steps in the late phase ($\tau > 0.82$) for fine refinement.

### Pseudocode

```python
def denoise_optimized_schedule(a_noise, vl_embeds, state_embeds, embodiment_id):
    """4-step Euler with optimized non-uniform timestep schedule."""
    # Optimized schedule (found via grid search on validation episodes)
    schedule = [0.000, 0.080, 0.350, 0.820]  # τ values
    tau_end = 1.0

    a = a_noise
    for i, tau in enumerate(schedule):
        tau_next = schedule[i + 1] if i + 1 < len(schedule) else tau_end
        dt = tau_next - tau
        tau_bucket = int(tau * 1000)

        velocity = DiT(a, tau_bucket, vl_embeds, state_embeds, embodiment_id)
        a = a + dt * velocity
    return a


def find_optimal_schedule(denoising_lab, observations, n_candidates=1000):
    """One-time offline calibration: grid search for optimal 4-step schedule.

    Run this once on a validation set of observations, then hard-code the
    winning schedule into denoise_optimized_schedule() for inference.
    This function is NOT called at inference time.
    """
    best_schedule, best_error = None, float('inf')

    # Reference: 64-step Euler (high-fidelity)
    reference_actions = [
        denoising_lab.denoise(obs, num_steps=64, seed=0)
        for obs in observations
    ]

    # Grid search over candidate schedules
    for _ in range(n_candidates):
        tau_1 = np.random.uniform(0.02, 0.30)
        tau_2 = np.random.uniform(tau_1 + 0.05, 0.60)
        tau_3 = np.random.uniform(tau_2 + 0.05, 0.95)
        schedule = [0.0, tau_1, tau_2, tau_3]

        errors = []
        for obs, ref in zip(observations, reference_actions):
            candidate = denoise_with_schedule(denoising_lab, obs, schedule, seed=0)
            errors.append(torch.norm(candidate - ref).item())

        mean_error = np.mean(errors)
        if mean_error < best_error:
            best_error = mean_error
            best_schedule = schedule

    return best_schedule
```

### How It Replaces Action Chunking

Fully transparent. Same 4 Euler steps, same output format, same `MultiStepWrapper` integration. The only change is the $\tau$ values at which each step is evaluated. No code changes outside the denoising loop.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Moderate-to-significant. AYS reports meaningful improvement in the few-step regime for image diffusion models. The gain depends on how non-uniform the velocity field complexity is across $\tau$ — if it's relatively uniform, benefit is small. |
| **Risk** | Low. This is still Euler integration; we're just moving the step positions. The DiT is trained to handle all $\tau$ values, so any valid schedule works. |
| **Latency** | Identical — 4 NFEs × ~16ms = ~64ms. |
| **Implementation** | Easy — change the schedule array and adjust step-size computation. The schedule search is a **one-time offline calibration** (~1 GPU-hour on a validation set), after which the optimal schedule is hard-coded for all future inference. |

### Prior Work

- **Sabour et al., "Align Your Steps: Optimizing Sampling Schedules in Diffusion Models"** — arXiv:2404.14507. Uses stochastic calculus to derive optimal noise-level schedules for a given solver and model. Particularly effective in the few-step regime.
- **Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3)** — arXiv:2403.03206. Introduced logit-normal timestep sampling during training, which implicitly creates a non-uniform distribution over $\tau$ values. GR00T's $\text{Beta}(1.5, 1.0)$ training distribution serves a similar purpose.

### How to run (inference)

From the **repo root**:

```bash
# Terminal 1 (model venv) — start the non-uniform schedule server
bash scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/run_server.sh

# Terminal 2 (sim venv) — run the reproducible benchmark
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n-episodes 10 --seed 42 \
    --output-dir /tmp/benchmark_results/optimized_nonuniform_timestep_schedule \
    --strategy-name optimized_nonuniform_timestep_schedule
```

To use a custom schedule (e.g., found via calibration below):

```bash
bash scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/run_server.sh \
    --schedule 0.0 0.1 0.4 0.85
```

### How to run (schedule calibration)

The default schedule `[0.0, 0.08, 0.35, 0.82]` is a starting hypothesis.
`calibrate_schedule.py` finds the best schedule for your embodiment and task
distribution in a single command.  It is a **one-time offline GPU job** — run
it once, get a schedule, hard-code it for all future inference.

#### Quick start

```bash
# One command — collects observations from sim rollouts, then grid-searches.
# Runs in the model venv.  The sim venv is invoked automatically as a subprocess.
bash scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/calibrate_schedule.sh
```

Or with custom arguments:

```bash
uv run python scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/calibrate_schedule.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
                robocasa_panda_omron/CloseDrawer_PandaOmron_Env \
    --n-episodes 5 --seed 42 \
    --n-candidates 2000 \
    --output-dir /tmp/schedule_calibration
```

If you already have saved observations (e.g., from `interactive_rollout.py`),
skip collection entirely:

```bash
uv run python .../calibrate_schedule.py \
    --obs-dir /tmp/my_saved_obs \
    --output-dir /tmp/schedule_calibration
```

#### What the flags mean

The script has **two phases**, and the flags control different phases:

**Phase 1 flags** control observation collection (sim rollouts):

| Flag | Default | What it controls |
|------|---------|------------------|
| `--env-names` | *(required)* | Which sim environments to collect observations from |
| `--n-episodes` | 5 | How many rollout episodes to run **per env** |
| `--obs-per-episode` | 4 | Observations saved per episode (evenly spaced through the episode) |
| `--obs-dir` | *(none)* | If set, **skips Phase 1 entirely** — uses pre-collected `.npz` files |

With 2 envs × 5 episodes × 4 obs/episode you get **40 observations**.
These are camera images + proprioceptive state snapshots from the sim, saved
as `.npz` files.  The quality of the *actions* during collection does not
matter — the observations are only used as conditioning inputs for the grid
search.

**Phase 2 flags** control the schedule search:

| Flag | Default | What it controls |
|------|---------|------------------|
| `--n-candidates` | 1000 | How many random 4-step schedules to try |
| `--reference-steps` | 64 | Euler steps for the "ground truth" reference |

Every candidate schedule still uses exactly **4 denoising steps** (same as
baseline Euler).  `--n-candidates` controls how many *different placements*
of those 4 steps are tried — e.g., `[0.0, 0.12, 0.41, 0.87]` vs
`[0.0, 0.05, 0.33, 0.79]` vs 998 others.  The step size (`dt`) for each
step varies per candidate (it equals the gap to the next τ value), but every
candidate integrates the full interval from τ=0 to τ=1.

#### How it works

**Phase 1 — Collect observations.**  Starts a temporary baseline GR00T server
in a background thread, then spawns `_collect_observations.py` as a subprocess
in the sim venv (robocasa).  The subprocess runs rollouts, saving
evenly-spaced observations as `.npz` files.  Skipped when `--obs-dir` is
provided.

**Phase 2 — Grid search.**  Loads the model into `DenoisingLab`, encodes each
saved observation through the Eagle VLM backbone, then:

1. **Compute references** — For each observation, run `--reference-steps`-step
   Euler to get the "ground truth" action chunk.
2. **For each of `--n-candidates` random schedules** — Run 4-step non-uniform
   Euler on every observation (from the *same* initial noise as the reference)
   and compute the mean L2 distance to the reference.
3. **Pick the winner** — The schedule with the lowest mean error is returned,
   alongside the uniform baseline's error for comparison.

Because the reference and each candidate start from identical noise (same
`seed`), the error isolates the discretization gap — no stochastic variance.

#### Concrete example

```
calibrate_schedule.py \
    --env-names .../OpenDrawer .../CloseDrawer \
    --n-episodes 5 --obs-per-episode 4 \
    --n-candidates 2000

Phase 1: Collect observations
├── Start temporary baseline server (standard 4-step Euler)
├── Run 5 episodes of OpenDrawer → save 20 .npz snapshots
├── Run 5 episodes of CloseDrawer → save 20 .npz snapshots
└── 40 observations total

Phase 2: Grid search (all on GPU, no sim needed)
├── Encode 40 observations through Eagle backbone
├── Compute 64-step Euler reference for each (40 × 64 = 2,560 NFEs)
├── Try 2000 random 4-step schedules:
│   └── Each: run 4-step Euler on all 40 obs (2000 × 40 × 4 = 320,000 NFEs)
├── Also evaluate uniform baseline [0.0, 0.25, 0.5, 0.75]
└── Output: best schedule + improvement % over uniform
```

#### Output

The script writes `calibration_result.json` to `--output-dir`:

```json
{
  "best_schedule": [0.0, 0.06, 0.37, 0.83],
  "best_error": 1.234567,
  "uniform_schedule": [0.0, 0.25, 0.5, 0.75],
  "uniform_error": 1.456789,
  "improvement_pct": 15.24,
  ...
}
```

It also prints a ready-to-use `--schedule` flag for `run_server.sh`.

#### Using the result

```bash
# Plug the calibrated schedule into the server
bash .../run_server.sh --schedule 0.0 0.06 0.37 0.83
```

Or hard-code it as the new default in `strategy.py`:

```python
DEFAULT_SCHEDULE: list[float] = [0.0, 0.06, 0.37, 0.83]
```

#### Resource estimates

| n_candidates | n_observations | NFEs (approx) | Time (L40) |
|:------------:|:--------------:|:-------------:|:----------:|
| 500          | 10             | ~21k          | ~5 min     |
| 1000         | 20             | ~81k          | ~20 min    |
| 2000         | 20             | ~161k         | ~40 min    |

Phase 1 (observation collection) adds ~1 minute per episode on top.

#### Tips

- **More observations help more than more candidates.**  20 obs × 1000
  candidates beats 5 obs × 4000 candidates.
- **Task diversity matters.**  Include observations from different episodes,
  different timesteps, and (if applicable) different task descriptions.
- **Sanity check the improvement.**  If `improvement_pct` is < 5 %, the
  velocity field is roughly uniform and this strategy has minimal impact.

---
