## Strategy 1: Single-Step RK4

**Category:** Drop-in replacement | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

Classical fourth-order Runge-Kutta (RK4) achieves $O(\Delta\tau^4)$ global error for multi-step integration — compared to Euler's $O(\Delta\tau)$ — by evaluating the velocity field at four carefully chosen points within a single step and taking a weighted average. Since GR00T already uses 4 NFEs (one per Euler step), we can instead use those same 4 NFEs as a single RK4 step spanning the entire $\tau \in [0, 1]$ interval. The benefit comes from RK4's ability to capture flow curvature: it samples the velocity at the start, midpoint (twice with different inputs), and endpoint of the interval, then blends them to approximate the true integral.

### Mathematical Formulation

Starting from $a_t^0 \sim \mathcal{N}(0, I)$ with a single step of size $\Delta\tau = 1.0$:

$$k_1 = v(a_t^0,\; 0,\; o_t,\; l_t)$$

$$k_2 = v\!\left(a_t^0 + \tfrac{1}{2} k_1,\; 0.5,\; o_t,\; l_t\right)$$

$$k_3 = v\!\left(a_t^0 + \tfrac{1}{2} k_2,\; 0.5,\; o_t,\; l_t\right)$$

$$k_4 = v\!\left(a_t^0 + k_3,\; 1.0,\; o_t,\; l_t\right)$$

$$a_t^1 = a_t^0 + \tfrac{1}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)$$

**Note on timestep discretization:** The DiT expects integer timestep buckets in $[0, 999]$. The evaluations above map to buckets $\{0, 500, 500, 999\}$. The two evaluations at $\tau = 0.5$ use *different action inputs* ($a_t^0 + \frac{1}{2}k_1$ vs $a_t^0 + \frac{1}{2}k_2$), so they produce different velocities despite the same timestep.

### Pseudocode

```python
def denoise_rk4(a_noise, vl_embeds, state_embeds, embodiment_id):
    """Single RK4 step: 4 NFEs, 4th-order accuracy."""
    a = a_noise  # (B, 50, 128), pure noise

    k1 = DiT(a,              tau_bucket=0,   vl_embeds, state_embeds, embodiment_id)
    k2 = DiT(a + 0.5 * k1,  tau_bucket=500, vl_embeds, state_embeds, embodiment_id)
    k3 = DiT(a + 0.5 * k2,  tau_bucket=500, vl_embeds, state_embeds, embodiment_id)
    k4 = DiT(a + 1.0 * k3,  tau_bucket=999, vl_embeds, state_embeds, embodiment_id)

    a_denoised = a + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return a_denoised
```

### How It Replaces Action Chunking

Unchanged. RK4 produces the same-shaped output $(B, 50, 128)$ which is decoded to $(B, 16, 29)$ per the existing `decode_action()` pipeline. The `MultiStepWrapper` executes 8 of 16 steps as before. The only change is *how* the ODE is integrated — the action representation, normalization, and chunking are untouched.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Depends on trajectory curvature. RK4 captures curvature via its 4-point weighted average, so if the flow has any nonlinearity, RK4 will trace it more faithfully than Euler. However, rectified flow is specifically designed to produce *near-linear* trajectories — and for a perfectly linear flow, Euler is already exact in 1 step. The practical gain depends on how much residual curvature exists after GR00T's training. Moderate improvement expected. |
| **Risk** | **Distribution mismatch at intermediate evaluations.** During training, the DiT sees noised actions at each $\tau$ drawn from the interpolation $a_t^\tau = (1-\tau)\epsilon + \tau a_t^1$. RK4's intermediate evaluations (e.g., $a_t^0 + 0.5 k_1$ for $k_2$) produce states that are *extrapolations* of the learned velocity, not interpolations between noise and data — they may not resemble any distribution the model was trained on. Similarly, $k_4$ evaluates at $\tau = 0.999$, a region where the training distribution ($\text{Beta}(1.5, 1.0)$ biased toward lower $\tau$) provides sparse coverage. Both effects could degrade the quality of velocity predictions at those evaluation points. |
| **Latency** | Identical — 4 NFEs × ~16ms = ~64ms. |
| **Implementation** | Trivial — ~15 lines of code in the denoising loop. |

### Prior Work

- **Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM)** — arXiv:2206.00364. Systematically compared ODE solvers (Euler, Heun, RK4) for diffusion model sampling. Found Heun's method (2nd-order) to be the best cost-accuracy tradeoff.
- **torchdiffeq** — Standard neural ODE library provides `odeint(method='rk4')` as a drop-in solver.

### How to run

From the **repo root**:

```bash
# Terminal 1 (model venv) — start the RK4 server
bash scripts/denoising_lab/eval/strategies/single_step_rk4/run_server.sh

# Terminal 2 (sim venv) — run the reproducible benchmark
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n-episodes 10 --seed 42 \
    --output-dir /tmp/benchmark_results/single_step_rk4 \
    --strategy-name single_step_rk4
```

To override server options (e.g., port):

```bash
bash scripts/denoising_lab/eval/strategies/single_step_rk4/run_server.sh --port 5556
```

---
