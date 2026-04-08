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

---
