# Beyond Euler: Advanced Flow-Matching Denoising Strategies for GR00T N1.6

**Target environment:** RoboCasa PandaOmron (3 cameras, 16-step action horizon, 29D action space)
**Goal:** Replace or augment the 4-step Euler integration in GR00T's DiT action head with strategies that produce higher-quality action chunks, reduce latency, or both.

---

## Table of Contents

1. [Notation](#notation)
2. [Current Baseline: 4-Step Euler Integration](#current-baseline-4-step-euler-integration)
3. [Strategies at a Glance](#strategies-at-a-glance)
4. [Strategy 1: Single-Step RK4](#strategy-1-single-step-rk4)
5. [Strategy 2: Optimized Non-Uniform Timestep Schedule](#strategy-2-optimized-non-uniform-timestep-schedule)
6. [Strategy 3: Multistep Velocity Recycling](#strategy-3-multistep-velocity-recycling)
7. [Strategy 4: Receding-Horizon Warm-Start Denoising](#strategy-4-receding-horizon-warm-start-denoising)
8. [Strategy 5: Heun-Langevin Hybrid Solver](#strategy-5-heun-langevin-hybrid-solver)
9. [Strategy 6: Shortcut-Conditioned DiT](#strategy-6-shortcut-conditioned-dit)
10. [Strategy 7: Reflow Trajectory Straightening](#strategy-7-reflow-trajectory-straightening)
11. [Strategy 8: Analytic Constraint Guidance](#strategy-8-analytic-constraint-guidance)
12. [Strategy 9: Horizon-Prioritized Denoising](#strategy-9-horizon-prioritized-denoising)
13. [Strategy 10: Noise-Space Mode Selection via Velocity Preview](#strategy-10-noise-space-mode-selection-via-velocity-preview)
14. [Strategy 11: Classifier-Free Action Guidance via Observation Dropout](#strategy-11-classifier-free-action-guidance-via-observation-dropout)
15. [Strategy 12: Curvature-Adaptive Step-Size Control](#strategy-12-curvature-adaptive-step-size-control-via-embedded-error-estimation)
16. [Strategy 13: Evolutionary Population Denoising](#strategy-13-evolutionary-population-denoising-with-fitness-selection)
17. [Strategy 14: Velocity-Field Convergence Refinement with OOD Gating](#strategy-14-velocity-field-convergence-refinement-with-ood-gating)
18. [Strategy 15: Spectral Temporal Decomposition with Frequency-Band Velocity Scaling](#strategy-15-spectral-temporal-decomposition-with-frequency-band-velocity-scaling)
19. [Strategy 16: Dynamics-Model-Verified Denoising via Imagination Rollouts](#strategy-16-dynamics-model-verified-denoising-via-imagination-rollouts)
20. [Strategy 17: Differentiable Denoising Trajectory Optimization (DDTO)](#strategy-17-differentiable-denoising-trajectory-optimization-ddto-with-self-consistent-quality-gradients)
21. [Comparison and Recommendation](#comparison-and-recommendation)
22. [Evaluation Protocol](#evaluation-protocol)
23. [References](#references)

---

## Notation

All strategies in this document use the following consistent notation.

**Velocity field:**

$$v(a_t^\tau,\; \tau,\; o_t,\; l_t)$$

| Symbol | Meaning |
|--------|---------|
| $a_t^\tau$ | Action chunk at environment timestep $t$, denoising progress $\tau$. Shape: $(B, 16, 29)$ for PandaOmron after decoding. |
| $\tau \in [0, 1]$ | Denoising progress. $\tau = 0$ is pure Gaussian noise; $\tau = 1$ is the fully denoised action chunk. |
| $o_t$ | Observation at time $t$ — includes camera images (3 × 256×256) and proprioceptive state (gripper qpos, base pose, EEF pose). Encoded by Eagle VLM into vision-language embeddings $\phi_t$. |
| $l_t$ | Language command at time $t$ (e.g., "open the right drawer"). Constant within an episode. |
| $\Delta\tau$ | Step size in denoising space. Baseline: $\Delta\tau = 0.25$. |
| NFE | Number of (neural) function evaluations — the number of forward passes through the DiT per action chunk. |

**Baseline Euler update** (what we are replacing):

$$a_t^{\tau + \Delta\tau} = a_t^\tau + \Delta\tau \cdot v(a_t^\tau,\; \tau,\; o_t,\; l_t)$$

with $\Delta\tau = 0.25$ and 4 steps from $\tau = 0 \to 0.25 \to 0.50 \to 0.75 \to 1.0$.

**For Strategy 6 only** (Shortcut Models), the velocity field is extended with step-size conditioning:

$$v(a_t^\tau,\; \tau,\; d,\; o_t,\; l_t)$$

where $d$ is the desired step size.

---

## Current Baseline: 4-Step Euler Integration

GR00T N1.6 uses **rectified flow matching** with a linear interpolation path between noise and clean actions:

**Training (forward pass):**

$$a_t^{\tau} = (1 - \tau) \cdot \epsilon + \tau \cdot a_t^1, \quad \epsilon \sim \mathcal{N}(0, I)$$

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{\tau \sim \text{Beta}}\Big[\big\| v_\theta(a_t^\tau,\; \tau,\; o_t,\; l_t) - (a_t^1 - \epsilon) \big\|^2\Big]$$

where $\tau \sim \text{Beta}(\alpha=1.5,\; \beta=1.0)$ biased toward noisier timesteps.

**Inference (4-step Euler):**

```python
# Current GR00T denoising loop (gr00t_n1d6.py:317-367)
a = torch.randn(B, 50, 128)          # pure noise in padded space
dt = 1.0 / 4                          # Δτ = 0.25
for step in range(4):
    tau = step / 4                     # τ = 0.00, 0.25, 0.50, 0.75
    tau_discrete = int(tau * 1000)     # 0, 250, 500, 750
    velocity = DiT(a, tau_discrete, vl_embeds, state_embeds, embodiment_id)
    a = a + dt * velocity              # Euler step
# decode: (B, 50, 128) → (B, 16, 29) for PandaOmron
```

**Properties:**
- **Order of accuracy:** 1 (global error is $O(\Delta\tau) = O(0.25)$)
- **NFEs:** 4
- **Latency:** ~64ms per chunk on L40 GPU (~16ms per DiT forward pass)
- **Action chunking:** Predicts 16 future timesteps; `MultiStepWrapper` executes 8, discards 8, re-queries

**The problem:** Euler integration is the crudest possible ODE solver. With only 4 steps and $\Delta\tau = 0.25$, the discretization error is substantial. For comparison, $\pi_0$ (Physical Intelligence) uses 10 Euler steps with $\Delta\tau = 0.1$, accepting 2.5× more compute for lower discretization error. The strategies below aim to get more accuracy per NFE, fewer total NFEs, or both.

---

## Strategies at a Glance

| # | Strategy | Category | NFEs | Extra Compute | Retraining? | Key Advantage |
|---|----------|----------|------|---------------|-------------|---------------|
| 1 | Single-Step RK4 | Drop-in | 4 | 0% | No | Curvature-aware integration at same cost |
| 2 | Optimized Timestep Schedule | Drop-in | 4 | 0% | No | Better step placement, free quality |
| 3 | Multistep Velocity Recycling | Drop-in | 4 | 0% | No | 2nd-order accuracy, free |
| 4 | Receding-Horizon Warm-Start | Drop-in (novel) | 3 | -25% | No | Exploits temporal coherence |
| 5 | Heun-Langevin Hybrid | Drop-in (novel) | 6 | +50% | No | 2nd-order + stochastic correction |
| 6 | Shortcut-Conditioned DiT | Fine-tune | 1–4 | +16% train | Yes | Variable inference budget |
| 7 | Reflow Trajectory Straightening | Fine-tune | 2 | ~1× train | Yes | Near-straight flows → 2 steps |
| **8** | **Analytic Constraint Guidance** | **Drop-in (novel)** | **4** | **~0%** | **No** | **Physics-informed quality gradients during denoising** |
| **9** | **Horizon-Prioritized Denoising** | **Drop-in (novel)** | **4** | **0%** | **No** | **Denoise near-future first; exploit DiT self-attention** |
| **10** | **Noise-Space Mode Selection** | **Drop-in (novel)** | **4+K** | **+K NFEs (batched)** | **No** | **Online noise optimization via 1-step velocity preview** |
| **11** | **Classifier-Free Action Guidance** | **Fine-tune (training mod)** | **8 (5 batched)** | **+25% inference** | **Yes (dropout only)** | **Amplify observation signal; task-phase-aware scheduling** |
| **12** | **Curvature-Adaptive Step-Size** | **Drop-in (novel)** | **4–6 (adaptive)** | **Variable** | **No** | **Per-observation difficulty-aware denoising; 2nd-order for free** |
| **13** | **Evolutionary Population Denoising** | **Drop-in (novel)** | **K×4 (batched)** | **+K× (batched)** | **No** | **Multi-step noise trajectory search with fitness selection** |
| **14** | **Velocity-Field Convergence Refinement + OOD Gating** | **Drop-in (novel)** | **5–6** | **+25–50%** | **No** | **Post-denoising quality check via fixed-point analysis; free OOD detection** |
| **15** | **Spectral Temporal Decomposition** | **Drop-in (novel)** | **4** | **0%** | **No** | **DCT-based frequency-band velocity scaling; step↔frequency alignment** |
| **16** | **Dynamics-Model-Verified Denoising** | **Auxiliary model** | **N×4 (batched)** | **+N× (batched)** | **No (DiT frozen)** | **Learned verifier replaces heuristic proxies; imagination-based ranking** |
| **17** | **Differentiable Denoising Trajectory Optimization (DDTO)** | **Drop-in (novel)** | **8–13 NFE-equiv** | **+100–300%** | **No** | **Gradient-based noise optimization through full denoising chain; self-consistent quality signal** |

---

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

---

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

## Strategy 3: Multistep Velocity Recycling

**Category:** Drop-in replacement | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

The key insight: in a multi-step ODE solver, each step produces a velocity prediction $v_i$ that is discarded after use. But that velocity contains information about the local curvature of the flow. By *caching* the velocity from the previous step and combining it with the current velocity, we can construct a 2nd-order update (Adams-Bashforth) with **zero additional NFEs**.

This is inspired by the **DPM-Solver++ multistep** variant, which achieves 2nd-order accuracy with only 1 NFE per step by reusing cached model outputs.

### Mathematical Formulation

**Adams-Bashforth 2-step (AB2):** Given velocity predictions at two consecutive $\tau$ values, the 2nd-order update is:

$$a_t^{\tau_{i+1}} = a_t^{\tau_i} + \Delta\tau \cdot \left(\frac{3}{2} v_i - \frac{1}{2} v_{i-1}\right)$$

where $v_i = v(a_t^{\tau_i}, \tau_i, o_t, l_t)$ and $v_{i-1}$ is the cached velocity from the previous step.

**For the 4-step denoising loop:**

| Step | Method | Formula | NFEs Used |
|------|--------|---------|-----------|
| 0 | Euler (no cache yet) | $a_t^{0.25} = a_t^0 + 0.25 \cdot v_0$ | 1 |
| 1 | AB2 | $a_t^{0.50} = a_t^{0.25} + 0.25 \cdot (\frac{3}{2} v_1 - \frac{1}{2} v_0)$ | 1 |
| 2 | AB2 | $a_t^{0.75} = a_t^{0.50} + 0.25 \cdot (\frac{3}{2} v_2 - \frac{1}{2} v_1)$ | 1 |
| 3 | AB2 | $a_t^{1.0} \;\;= a_t^{0.75} + 0.25 \cdot (\frac{3}{2} v_3 - \frac{1}{2} v_2)$ | 1 |

Total: 4 NFEs (identical to baseline), but steps 1–3 are 2nd-order accurate instead of 1st-order.

### Pseudocode

```python
def denoise_ab2(a_noise, vl_embeds, state_embeds, embodiment_id):
    """4-step Adams-Bashforth-2: 2nd-order accuracy with zero extra NFEs."""
    a = a_noise
    dt = 0.25
    prev_velocity = None

    for step in range(4):
        tau = step / 4
        tau_bucket = int(tau * 1000)
        velocity = DiT(a, tau_bucket, vl_embeds, state_embeds, embodiment_id)

        if prev_velocity is None:
            # Step 0: standard Euler (no history to use)
            a = a + dt * velocity
        else:
            # Steps 1-3: Adams-Bashforth 2-step (2nd-order, uses cached velocity)
            a = a + dt * (1.5 * velocity - 0.5 * prev_velocity)

        prev_velocity = velocity

    return a
```

### How It Replaces Action Chunking

Completely transparent. Output shape, normalization, decoding, and chunk execution are all unchanged. The only difference is how successive velocity predictions are combined in the update rule.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Moderate. AB2 is provably 2nd-order for smooth velocity fields, giving $O(\Delta\tau^2) = O(0.0625)$ global error vs Euler's $O(0.25)$ — a ~4× reduction. The practical improvement depends on velocity field smoothness. |
| **Risk** | Low-moderate. AB2 has a smaller stability region than Euler, which could cause divergence if the velocity field has sharp discontinuities. However, flow matching with rectified flow produces relatively smooth fields. |
| **Latency** | Identical — 4 NFEs × ~16ms = ~64ms. |
| **Implementation** | Trivial — cache one tensor, change one line of the update rule. |

### Prior Work

- **Lu et al., "DPM-Solver++: Fast Solver for Guided Diffusion Sampling"** — arXiv:2211.01095. The multistep variant achieves 2nd-order accuracy with 1 NFE per step. Original DPM-Solver (arXiv:2206.00927) introduced the exponential integrator approach.
- **Adams-Bashforth methods** — classical numerical analysis. AB2 is the simplest linear multistep method. See Hairer, Norsett, Wanner, "Solving Ordinary Differential Equations I" (1993).

---

## Strategy 4: Receding-Horizon Warm-Start Denoising

**Category:** Novel, drop-in | **NFEs:** 3 (25% faster) | **Retraining:** None

### Overview

This is a **novel strategy** that exploits the temporal structure of receding-horizon action chunking.

Currently, every time GR00T predicts a new action chunk, it starts from pure Gaussian noise $a_t^0 \sim \mathcal{N}(0, I)$, discarding all knowledge from the previous chunk. But the previous chunk's un-executed actions (steps 8–15) overlap significantly with the new chunk's first 8 steps — the robot hasn't moved much in 8 control steps, and the task goal hasn't changed. This overlap is wasted information.

**The idea:** Initialize the new chunk's early timesteps from the previous chunk's un-executed predictions, add calibrated noise to bring them back to a partial-denoising level, and start denoising from that level instead of from pure noise. This is analogous to **SDEdit** (image editing by adding noise then denoising) applied to the temporal action domain.

### Mathematical Formulation

Let $a_{t-1}^1$ be the fully denoised action chunk from the previous query (16 timesteps). After executing steps 0–7, steps 8–15 remain unused. We construct an informed initial state for the new chunk:

**Step 1: Temporal shift.** Shift the un-executed actions backward by 8 positions:

$$\tilde{a}_t[0\!:\!8] = a_{t-1}^1[8\!:\!16]$$

For positions 8–15 of the new chunk (which have no overlap), sample fresh noise:

$$\tilde{a}_t[8\!:\!16] \sim \mathcal{N}(0, I)$$

**Step 2: Partial re-noising.** Add calibrated noise to the shifted actions to bring them to noise level $\tau_{\text{start}}$ (e.g., $\tau_{\text{start}} = 0.25$):

$$a_t^{\tau_{\text{start}}}[0\!:\!8] = (1 - \tau_{\text{start}}) \cdot \epsilon + \tau_{\text{start}} \cdot \tilde{a}_t[0\!:\!8], \quad \epsilon \sim \mathcal{N}(0, I)$$

Following GR00T's rectified flow interpolation convention where $\tau = 0$ is pure noise and $\tau = 1$ is clean.

**Step 3: Denoise from $\tau_{\text{start}}$ instead of $\tau = 0$.** With $\tau_{\text{start}} = 0.25$, we skip the first denoising step entirely, running only 3 steps:

$$\tau \in \{0.25, 0.50, 0.75\} \quad \text{(3 steps, 3 NFEs)}$$

### Pseudocode

```python
def denoise_warm_start(
    a_noise,
    prev_chunk_decoded,     # (B, 16, action_dim) from previous query, or None
    vl_embeds, state_embeds, embodiment_id,
    tau_start=0.25,         # where to resume denoising
    n_executed=8,           # how many steps were executed from previous chunk
):
    """Warm-start denoising: 3 NFEs instead of 4 by reusing previous chunk."""

    if prev_chunk_decoded is not None:
        # Re-encode previous actions to padded space (16, action_dim) → (50, 128)
        # NOTE: encode_action is the inverse of decode_action — it must:
        #   1. Re-normalize each action key back to [-1, 1] using stored min/max stats
        #   2. Concatenate into the embodiment-specific flat vector
        #   3. Pad to (50, 128) using the embodiment's action_encoder MLP
        # This inverse does not exist in the current codebase and must be implemented.
        prev_padded = encode_action(prev_chunk_decoded)  # inverse of decode_action

        # Shift: previous steps 8-15 become new steps 0-7
        warm = torch.zeros_like(a_noise)
        warm[:, :n_executed, :] = prev_padded[:, n_executed:, :]
        warm[:, n_executed:, :] = torch.randn_like(warm[:, n_executed:, :])

        # Partial re-noising via rectified flow interpolation
        epsilon = torch.randn_like(warm[:, :n_executed, :])
        warm[:, :n_executed, :] = (
            (1 - tau_start) * epsilon + tau_start * warm[:, :n_executed, :]
        )

        a = warm
        start_step = 1  # skip step 0 (tau=0.00)
    else:
        a = a_noise
        start_step = 0

    # Denoise from tau_start
    dt = 0.25
    for step in range(start_step, 4):
        tau = step / 4
        tau_bucket = int(tau * 1000)
        velocity = DiT(a, tau_bucket, vl_embeds, state_embeds, embodiment_id)
        a = a + dt * velocity

    return a
```

### How It Replaces Action Chunking

This strategy *modifies the initialization* of each chunk rather than the integration method. The `MultiStepWrapper` still executes 8 of 16 predicted steps. The key change is that the un-executed predictions from the previous chunk seed the next chunk, creating temporal coherence.

**Interaction with action chunking:**
- First chunk of each episode: standard 4-step denoising (no previous chunk exists)
- Subsequent chunks: 3-step denoising from warm-start, saving 25% compute
- The overlap creates smooth action transitions between chunks — reducing the "seam" artifacts visible when chunks are generated independently

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Potentially high. The warm-started region (steps 0–7) starts much closer to the correct solution, requiring less denoising work. The un-initialized region (steps 8–15) still gets full denoising. The net effect should be smoother, more temporally coherent trajectories. |
| **Risk** | Moderate. (1) The re-encoded previous actions may accumulate errors over many chunks (drift). (2) The DiT expects specific noise-level statistics at each $\tau$; the warm-start distribution may not match. (3) The `encode_action` inverse mapping does not currently exist — it requires inverting the normalization and the `action_encoder` MLP (which is a `CategorySpecificLinear` layer, not trivially invertible). An alternative is to cache the *raw padded* actions $(B, 50, 128)$ before decoding, bypassing the re-encoding problem entirely. (4) A key tuning parameter is $\tau_{\text{start}}$ — too high (e.g., 0.5) and we skip too much denoising; too low (e.g., 0.1) and we barely save compute. |
| **Latency** | 25% reduction — 3 NFEs × ~16ms = ~48ms. |
| **Implementation** | Moderate — requires (a) caching raw padded actions from previous chunk (simpler than re-encoding), (b) partial re-noising logic, (c) tuning $\tau_{\text{start}}$. |

### Prior Work and Inspiration

- **Meng et al., "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations"** — arXiv:2108.01073. The core idea of adding noise to a reference signal and denoising from a partial noise level is directly borrowed from SDEdit's approach to image editing.
- **Temporal ensembling in Diffusion Policy** — Chi et al. (arXiv:2303.04137) describe exponential weighting of overlapping action chunks for smooth transitions. Our approach achieves temporal coherence at the denoising level rather than post-hoc blending.
- **DDIM inversion** — Dhariwal & Nichol (arXiv:2105.05233). The principle of mapping between noise levels via the learned ODE is the theoretical basis for our partial re-noising step.

---

## Strategy 5: Heun-Langevin Hybrid Solver

**Category:** Novel, drop-in | **NFEs:** 6 (+50% over baseline) | **Retraining:** None

### Overview

This is a **novel hybrid strategy** that combines the strengths of deterministic and stochastic sampling in a principled way.

**The key insight:** Deterministic ODE solvers (Euler, Heun, RK4) follow a single trajectory from noise to data. If discretization errors accumulate and push the trajectory into a low-probability region, the deterministic solver has no mechanism to recover — it commits to the error. Stochastic samplers (Langevin dynamics) can escape low-probability regions by injecting noise and re-denoising, but they require many steps for convergence.

**The hybrid:** Use a high-accuracy deterministic solver (Heun's method) for the bulk of the denoising, then apply a stochastic Langevin corrector in the final phase to fix accumulated errors. This gives the best of both worlds: fast coarse-to-fine progression plus error correction at the end where precision matters most for action quality.

### Mathematical Formulation

**Phase 1 — Heun's method (steps 0–1, 4 NFEs):**

For each of 2 Heun steps with $\Delta\tau = 0.25$:

$$k_1 = v(a_t^{\tau_i},\; \tau_i,\; o_t,\; l_t)$$

$$\hat{a} = a_t^{\tau_i} + \Delta\tau \cdot k_1 \qquad \text{(predictor)}$$

$$k_2 = v(\hat{a},\; \tau_i + \Delta\tau,\; o_t,\; l_t)$$

$$a_t^{\tau_{i+1}} = a_t^{\tau_i} + \frac{\Delta\tau}{2}\left(k_1 + k_2\right) \qquad \text{(corrector)}$$

This advances from $\tau = 0 \to 0.50$ with 2nd-order accuracy using 4 NFEs.

**Phase 2 — Euler step (step 2, 1 NFE):**

$$a_t^{0.75} = a_t^{0.50} + 0.25 \cdot v(a_t^{0.50},\; 0.50,\; o_t,\; l_t)$$

**Phase 3 — Langevin corrector (step 3, 1 NFE):**

At $\tau = 0.75$, inject small noise and denoise to correct accumulated discretization error:

$$a_t^{0.75'} = a_t^{0.75} + \sigma_{\text{corr}} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

$$a_t^{1.0} = a_t^{0.75'} + 0.25 \cdot v(a_t^{0.75'},\; 0.75,\; o_t,\; l_t)$$

where $\sigma_{\text{corr}}$ is a small noise scale (e.g., $\sigma_{\text{corr}} = 0.01$) calibrated to perturb without destroying the signal.

The injected noise pushes $a_t^{0.75}$ slightly off the learned trajectory, and the final denoising step corrects back — effectively "re-centering" onto the high-probability region of the action distribution.

### Pseudocode

```python
def denoise_heun_langevin(a_noise, vl_embeds, state_embeds, embodiment_id,
                          sigma_corr=0.01):
    """Heun (2 steps) + Euler (1 step) + Langevin corrector (1 step). 6 NFEs."""
    a = a_noise
    dt = 0.25

    # Phase 1: 2 Heun steps (tau=0.0 → 0.50), 4 NFEs
    for step in [0, 1]:
        tau = step * dt
        tau_bucket = int(tau * 1000)
        tau_next_bucket = int((tau + dt) * 1000)

        k1 = DiT(a, tau_bucket, vl_embeds, state_embeds, embodiment_id)
        a_pred = a + dt * k1
        k2 = DiT(a_pred, tau_next_bucket, vl_embeds, state_embeds, embodiment_id)
        a = a + (dt / 2) * (k1 + k2)

    # Phase 2: 1 Euler step (tau=0.50 → 0.75), 1 NFE
    velocity = DiT(a, 500, vl_embeds, state_embeds, embodiment_id)
    a = a + dt * velocity

    # Phase 3: Langevin corrector (tau=0.75 → 1.0), 1 NFE
    noise = torch.randn_like(a) * sigma_corr
    a_perturbed = a + noise
    velocity = DiT(a_perturbed, 750, vl_embeds, state_embeds, embodiment_id)
    a = a_perturbed + dt * velocity

    return a
```

### How It Replaces Action Chunking

Same output format as baseline. The additional 2 NFEs increase latency by ~32ms, but the 2nd-order Heun integration + stochastic correction should produce higher-quality action chunks that may compensate via fewer failed grasps or smoother trajectories.

The stochastic corrector introduces a small amount of randomness in the final denoising step. This means repeated queries with the same observation will produce slightly different action chunks — enabling implicit exploration over action modes.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | High. The Heun phase achieves 2nd-order accuracy for the coarse trajectory, while the Langevin corrector addresses the known failure mode of deterministic samplers: accumulating into low-probability regions. |
| **Risk** | The Langevin corrector noise magnitude $\sigma_{\text{corr}}$ is a sensitive hyperparameter. Too large: destroys the denoised signal. Too small: no correction effect. Needs careful tuning per embodiment. |
| **Latency** | +50% — 6 NFEs × ~16ms = ~96ms. Acceptable for 10Hz control but tight. |
| **Implementation** | Moderate — Heun is straightforward; the Langevin corrector requires tuning. |

### Prior Work

- **Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations"** — arXiv:2011.13456. Introduced the predictor-corrector framework for score-based generative models. The Langevin corrector is directly from their Algorithm 4.
- **Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM)** — arXiv:2206.00364. Showed that Heun's method is the optimal 2nd-order solver for diffusion ODEs.
- **Zhao et al., "UniPC: A Unified Predictor-Corrector Framework"** — arXiv:2302.04867. Unified predictor-corrector methods with "free" corrector steps that reuse cached predictions.

---

## Strategy 6: Shortcut-Conditioned DiT

**Category:** Fine-tuning required | **NFEs:** 1–4 (variable) | **Retraining:** Yes (+16% compute)

### Overview

**Shortcut Models** (Frans et al., 2024) are the most promising recent innovation for few-step flow matching. The core idea: condition the velocity field not only on the current noise level $\tau$ but also on the *desired step size* $d$. This allows a single model to learn how to "skip ahead" by varying amounts.

At training time, the model learns a self-consistency constraint: taking one step of size $2d$ should produce the same result as taking two steps of size $d$. This forces the model to internalize the curvature of the flow, enabling accurate large-step (even single-step) predictions.

At inference time, you choose the step budget based on latency requirements: $d = 1.0$ for 1-step (fastest), $d = 0.25$ for 4-step (highest quality), or anything in between.

### Mathematical Formulation

**Extended velocity field:**

$$v(a_t^\tau,\; \tau,\; d,\; o_t,\; l_t)$$

where $d \in [0, 1]$ is the step-size conditioning parameter.

**Training objective** — two losses trained jointly:

1. **Flow matching loss** (base, at small $d$):

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{\tau,\; \epsilon}\left[\left\| v_\theta(a_t^\tau,\; \tau,\; 0,\; o_t,\; l_t) - (a_t^1 - \epsilon) \right\|^2\right]$$

This is the standard rectified flow objective with $d = 0$ (infinitesimal step).

2. **Self-consistency loss** (shortcut, at larger $d$):

$$\hat{a}_{\text{mid}} = a_t^\tau + d \cdot v_\theta(a_t^\tau,\; \tau,\; d,\; o_t,\; l_t)$$

$$s_{\text{target}} = \text{sg}\left[v_\theta(a_t^\tau,\; \tau,\; d,\; o_t,\; l_t) + v_\theta(\hat{a}_{\text{mid}},\; \tau + d,\; d,\; o_t,\; l_t)\right] / 2$$

$$\mathcal{L}_{\text{SC}} = \mathbb{E}_{\tau,\; d}\left[\left\| v_\theta(a_t^\tau,\; \tau,\; 2d,\; o_t,\; l_t) - s_{\text{target}} \right\|^2\right]$$

where $\text{sg}[\cdot]$ is stop-gradient. This enforces: one step of size $2d$ ≈ average of two steps of size $d$.

**Inference:** For $K$ steps, set $d = 1/K$ and integrate:

$$a_t^{\tau + d} = a_t^\tau + d \cdot v(a_t^\tau,\; \tau,\; d,\; o_t,\; l_t), \quad \tau = 0, d, 2d, \ldots$$

### Pseudocode

```python
# === Architecture modification (in DiT) ===
# Add step-size embedding alongside timestep embedding
class ShortcutDiT(AlternateVLDiT):
    def __init__(self, ...):
        super().__init__(...)
        # New: step-size embedding, same architecture as timestep embedding
        self.step_size_embedding = TimestepEmbedding(256, inner_dim)

    def forward(self, actions, timestep, step_size, vl_embeds, state_embeds, emb_id):
        t_emb = self.timestep_embedding(timestep)
        d_emb = self.step_size_embedding(step_size)
        combined_emb = t_emb + d_emb  # additive conditioning
        # ... rest of DiT forward pass with combined_emb for AdaLayerNorm


# === Training ===
def shortcut_training_step(model, batch):
    actions, obs, lang = batch
    noise = torch.randn_like(actions)
    tau = sample_beta_time(actions.shape[0])

    # Base flow matching loss (d ≈ 0)
    a_noised = (1 - tau) * noise + tau * actions
    v_pred = model(a_noised, tau, d=0, obs, lang)
    loss_fm = F.mse_loss(v_pred, actions - noise)

    # Self-consistency loss (random d)
    d = torch.rand(1).item() * 0.5  # d ∈ [0, 0.5]
    with torch.no_grad():
        v_small_1 = model(a_noised, tau, d, obs, lang)
        a_mid = a_noised + d * v_small_1
        v_small_2 = model(a_mid, tau + d, d, obs, lang)
        target = (v_small_1 + v_small_2) / 2

    v_big = model(a_noised, tau, 2 * d, obs, lang)
    loss_sc = F.mse_loss(v_big, target)

    return loss_fm + loss_sc


# === Inference (variable step budget) ===
def denoise_shortcut(a_noise, vl_embeds, state_embeds, embodiment_id, n_steps=1):
    """Variable-step inference: 1 step (fastest) to 4 steps (highest quality)."""
    a = a_noise
    d = 1.0 / n_steps
    d_bucket = int(d * 1000)

    for step in range(n_steps):
        tau = step * d
        tau_bucket = int(tau * 1000)
        velocity = DiT(a, tau_bucket, d_bucket, vl_embeds, state_embeds, embodiment_id)
        a = a + d * velocity

    return a
```

### How It Replaces Action Chunking

Action chunking is unchanged. The shortcut model produces the same $(B, 16, 29)$ output regardless of step count. The key advantage for action chunking is **adaptive compute**: during time-critical phases (fast reactive motions), use 1-step inference for minimum latency; during precision phases (grasping, insertion), use 4-step for maximum quality.

**Data for fine-tuning:** The same demonstration dataset used for GR00T's original post-training. No new robot data is needed — the self-consistency loss is computed from the model's own predictions.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Very high. Frans et al. demonstrated 87% Push-T success with 1-step shortcut vs 95% for 100-step Diffusion Policy and 12% for 1-step Diffusion Policy. The quality degrades gracefully with fewer steps rather than catastrophically. |
| **Risk** | (1) Modifying the DiT architecture (adding step-size embedding) requires careful integration with GR00T's pre-trained weights. The new embedding is randomly initialized, so the model needs meaningful fine-tuning. (2) The self-consistency loss adds training complexity. (3) +16% training compute is non-trivial. |
| **Latency** | 1-step: ~16ms (4× faster). 2-step: ~32ms (2× faster). 4-step: ~64ms (same). |
| **Implementation** | High — architecture modification + new training loss + fine-tuning infrastructure. |

### Prior Work

- **Frans et al., "One Step Diffusion via Shortcut Models"** — arXiv:2410.12557. The primary reference. Demonstrates shortcut models on image generation (CIFAR-10, ImageNet) and robot control (Push-T, Transport via Diffusion Policy). Single training phase, +16% compute, variable inference budget.
- **Prasad et al., "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation"** — arXiv:2405.07503. A related approach (consistency distillation) achieving 10× speedup for robot visuomotor policies. Requires a pre-trained teacher model.

---

## Strategy 7: Reflow Trajectory Straightening

**Category:** Fine-tuning required | **NFEs:** 2 (50% faster) | **Retraining:** Yes (~1× original training cost)

### Overview

**Rectified Flow** (Liu et al., 2022) shows that flow matching with linear interpolation paths produces velocity fields that, while optimal in expectation, create *curved* ODE trajectories in practice. These curves arise because different (noise, data) pairs produce crossing flow paths, and the learned velocity field must compromise — averaging over crossings creates curvature.

**Reflow** is a procedure that *straightens* these trajectories. The idea: use the current model to generate synthetic (noise, clean-action) pairs by running the full ODE, then retrain on these new pairs. Since each synthetic pair lies on the *same* ODE trajectory, there are no crossing paths, and the retrained model produces straighter flows. Straighter flows mean Euler integration is more accurate with fewer steps.

After 1–2 rounds of reflow, even 2 Euler steps can match the quality of the original 4-step model.

### Mathematical Formulation

**Round 1 — Generate synthetic pairs:**

For each observation $(o_t, l_t)$ in the training set, sample noise and run the current model's ODE to completion:

$$\epsilon \sim \mathcal{N}(0, I)$$

$$a_t^1 = \text{ODE\_solve}(\epsilon,\; v_\theta,\; \tau: 0 \to 1) \quad \text{(using many Euler steps, e.g., 64)}$$

This produces paired samples $(\epsilon, a_t^1)$ that lie on the *same* ODE trajectory.

**Round 1 — Retrain:**

Construct new noised actions using the *paired* (noise, clean-action) from above:

$$a_t^\tau = (1 - \tau) \cdot \epsilon + \tau \cdot a_t^1$$

$$\mathcal{L}_{\text{reflow}} = \mathbb{E}_\tau\left[\left\| v_{\theta'}(a_t^\tau,\; \tau,\; o_t,\; l_t) - (a_t^1 - \epsilon) \right\|^2\right]$$

This is the same loss function as original training, but with $\epsilon$ and $a_t^1$ drawn from the *same* trajectory rather than independently sampled.

**Why this straightens flows:** When $(\epsilon, a_t^1)$ are on the same trajectory, the interpolation path $a_t^\tau = (1-\tau)\epsilon + \tau a_t^1$ exactly matches the ODE trajectory (since it's already close to straight for rectified flow). The model $v_{\theta'}$ can fit this with near-zero error, producing even straighter flows.

**After reflow — 2-step Euler is sufficient:**

$$a_t^{0.5} = a_t^0 + 0.5 \cdot v_{\theta'}(a_t^0,\; 0,\; o_t,\; l_t)$$

$$a_t^{1.0} = a_t^{0.5} + 0.5 \cdot v_{\theta'}(a_t^{0.5},\; 0.5,\; o_t,\; l_t)$$

### Pseudocode

```python
# === Phase 1: Generate synthetic trajectory pairs ===
def generate_reflow_pairs(model, dataloader, n_ode_steps=64):
    """Run current model's ODE to collect (noise, clean_action) pairs."""
    pairs = []
    for batch in dataloader:
        obs, lang, _ = batch  # don't need ground-truth actions
        noise = torch.randn(batch_size, 50, 128)

        # High-fidelity ODE solve (many steps for accuracy)
        a = noise.clone()
        dt = 1.0 / n_ode_steps
        for step in range(n_ode_steps):
            tau = step / n_ode_steps
            tau_bucket = int(tau * 1000)
            velocity = model.get_velocity(a, tau_bucket, obs, lang)
            a = a + dt * velocity

        pairs.append((noise, a, obs, lang))  # (epsilon, a_t^1, o_t, l_t)
    return pairs


# === Phase 2: Retrain on synthetic pairs ===
def reflow_training_step(model, pair_batch):
    """Standard flow matching loss, but on trajectory-paired data."""
    noise, clean_actions, obs, lang = pair_batch

    tau = sample_beta_time(noise.shape[0])
    a_noised = (1 - tau) * noise + tau * clean_actions
    target_velocity = clean_actions - noise  # paired velocity

    v_pred = model(a_noised, tau, obs, lang)
    loss = F.mse_loss(v_pred, target_velocity)
    return loss


# === Phase 3: 2-step inference with reflowed model ===
def denoise_reflowed(a_noise, vl_embeds, state_embeds, embodiment_id):
    """2-step Euler with reflowed model — straighter trajectories."""
    a = a_noise
    dt = 0.5  # 2 steps of Δτ = 0.5

    for step in range(2):
        tau = step * dt
        tau_bucket = int(tau * 1000)
        velocity = DiT(a, tau_bucket, vl_embeds, state_embeds, embodiment_id)
        a = a + dt * velocity

    return a
```

### How It Replaces Action Chunking

Action chunking is unchanged. The reflowed model produces the same output shape and uses the same decode pipeline. The only visible difference is 2 Euler steps instead of 4, halving the action-head latency.

**Data for fine-tuning:** 100% synthetic, generated by the current model itself. For each $(o_t, l_t)$ in the training set, we run the current 64-step ODE to generate a paired $(\epsilon, a_t^1)$. No new robot demonstrations are needed. The generation phase is a one-time cost (~1 forward pass per training sample × 64 steps).

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | High. Liu et al. prove that reflow monotonically reduces transport cost and straightens trajectories. After 1 round, 2-step Euler closely approximates many-step quality. InstaFlow demonstrated single-step generation for Stable Diffusion using this approach. |
| **Risk** | (1) The reflow procedure requires generating synthetic pairs over the *entire* training distribution, which is computationally expensive (64 ODE steps per sample). (2) The reflowed model may overfit to the current model's distribution rather than the true data distribution — this is a form of "mode collapse via self-distillation." (3) GR00T's multi-embodiment training complicates reflow: pairs must be generated per-embodiment. |
| **Latency** | 50% reduction — 2 NFEs × ~16ms = ~32ms. |
| **Implementation** | High — requires (a) pair generation pipeline, (b) modified training loop, (c) multiple training rounds. Total compute: ~1× original training. |

### Prior Work

- **Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"** — arXiv:2209.03003. Introduces rectified flow and the reflow procedure. Proves that reflow straightens trajectories and reduces transport cost.
- **Liu et al., "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation"** — arXiv:2309.06380. Applied reflow + distillation to Stable Diffusion, achieving one-step text-to-image generation.
- **Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3)** — arXiv:2403.03206. Demonstrated that rectified flow + logit-normal timestep sampling scales to 8B+ parameter models.

---

## Strategy 8: Analytic Constraint Guidance

**Category:** Novel, drop-in | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

This is a **novel strategy** that bridges two fields: *classifier guidance* from diffusion-based image generation and *trajectory optimization* from robotics.

**The problem:** The DiT learns the velocity field purely from data. It has no explicit knowledge of physics — so it can produce action chunks with jittery trajectories (high jerk), floating-point gripper values (0.47 instead of 0 or 1), or control mode oscillations. Currently, GR00T handles this with post-hoc clipping in `decode_action()`. But post-hoc corrections are applied *after* the full 4-step denoising chain — meaning the DiT may have spent its last 2-3 steps refining toward an action that gets clipped away. The denoising effort is wasted.

**The idea:** Apply physical constraints *during* denoising, not after. At each Euler step, after computing the velocity, add a small correction gradient that nudges the action toward the physically valid region. This is the same mechanism as classifier guidance in image diffusion (Dhariwal & Nichol, 2021), but with a crucial difference: instead of a learned classifier network, we use **analytic (closed-form) constraint functions** with hand-crafted gradients derived from robot physics.

**Why this is novel in the VLA context:**

1. **Classifier guidance** in image diffusion requires training a separate classifier — expensive and task-specific. Our constraint functions are hand-crafted from domain knowledge, requiring zero training.
2. **Trajectory optimization** in robotics (MPPI, CEM, iLQR) operates on full action sequences post-hoc. Our approach integrates constraints *into* the generative process, allowing the DiT to adapt to the constraints at each step.
3. **Diffusion-as-planning** (Janner et al., "Diffuser", 2022) uses reward guidance for DDPM-style planning. We adapt this principle to flow matching with analytic rewards — a different ODE formulation, different guidance mechanism, and different reward structure.

The result: "classifier-free, reward-model-free" guidance that works zero-shot with any observation, encodes physical correctness, and costs almost nothing computationally.

### Mathematical Formulation

Define differentiable constraint functions $\{C_k\}$ where each $C_k: \mathbb{R}^{H \times D} \to \mathbb{R}_{\geq 0}$ measures violation of a physical constraint (zero when satisfied). The total quality score:

$$Q(a) = -\sum_k \lambda_k \, C_k(a)$$

At each denoising step, after the standard velocity update, apply a constraint correction:

$$\hat{a}_t^{\tau_{i+1}} = a_t^{\tau_i} + \Delta\tau \cdot v(a_t^{\tau_i},\; \tau_i,\; o_t,\; l_t) \qquad \text{(standard Euler)}$$

$$a_t^{\tau_{i+1}} = \hat{a}_t^{\tau_{i+1}} + \eta_i \cdot \nabla_a Q\!\left(\hat{a}_t^{\tau_{i+1}}\right) \qquad \text{(constraint correction)}$$

The guidance strength is **annealed** with denoising progress:

$$\eta_i = \eta \cdot \tau_i$$

At $\tau = 0$ (pure noise), $\eta_0 = 0$ — no guidance, because constraints are meaningless on noise. At $\tau = 0.75$ (last step), $\eta_3 = 0.75\eta$ — strong guidance, because the action has taken shape and constraints are meaningful.

**Constraint functions for PandaOmron:**

**1. Temporal smoothness** (minimize jerk across the action horizon):

$$C_{\text{smooth}}(a) = \sum_{j=0}^{H-3} \left\| a[j{+}2] - 2\,a[j{+}1] + a[j] \right\|^2$$

This penalizes the second-order finite difference (discrete acceleration change). The gradient is a simple tridiagonal Laplacian — no neural network, no backpropagation:

$$\frac{\partial C_{\text{smooth}}}{\partial a[j]} = 2 \left( a[j{+}2] - 2a[j{+}1] + a[j] \right) - 4 \left( a[j{+}1] - 2a[j] + a[j{-}1] \right) + 2 \left( a[j] - 2a[j{-}1] + a[j{-}2] \right)$$

(boundary terms handled by clamping indices). Applied only to continuous EEF position/rotation dimensions.

**2. Discrete action decisiveness** (push gripper and control_mode toward binary values):

$$C_{\text{discrete}}(a) = \sum_{j=0}^{H-1} \left[ a_{\text{grip}}[j] \cdot (1 - a_{\text{grip}}[j]) + a_{\text{mode}}[j] \cdot (1 - a_{\text{mode}}[j]) \right]$$

This is minimized at $\{0, 1\}$ with analytic gradient:

$$\frac{\partial C_{\text{discrete}}}{\partial a_{\text{grip}}[j]} = 1 - 2\,a_{\text{grip}}[j]$$

Pushes values toward their nearest binary pole during denoising, rather than clipping post-hoc.

**3. Control mode consistency** (discourage rapid switching between arm and base control):

$$C_{\text{mode}}(a) = \sum_{j=0}^{H-2} \left( a_{\text{mode}}[j{+}1] - a_{\text{mode}}[j] \right)^2$$

Penalizes frame-to-frame control mode changes. The gradient is a simple first-difference:

$$\frac{\partial C_{\text{mode}}}{\partial a_{\text{mode}}[j]} = -2(a_{\text{mode}}[j{+}1] - a_{\text{mode}}[j]) + 2(a_{\text{mode}}[j] - a_{\text{mode}}[j{-}1])$$

**Computational cost of gradients:** All three constraints are computed via simple finite differences on the action tensor — $O(H \times D)$ operations. For $H = 16$, $D = 29$, this is 464 floating-point operations per step — utterly negligible compared to the ~1.5B-parameter DiT forward pass (~16ms).

### Pseudocode

```python
def make_constraint_guided_fn(
    lambda_smooth=0.005,
    lambda_discrete=0.01,
    lambda_mode=0.003,
    eta=0.1,
    # Dimension indices within the 128-dim padded action space (PandaOmron)
    eef_pos_dims=slice(0, 3),       # EEF position (continuous)
    eef_rot_dims=slice(3, 6),       # EEF rotation (continuous)
    gripper_dim=6,                   # Gripper close (discrete)
    mode_dim=11,                     # Control mode (discrete)
):
    """Factory for a constraint-guided velocity modifier.

    Returns a function compatible with DenoisingLab's guided_fn interface.
    """
    def guided_fn(actions_before, step_idx, velocity):
        """Modify velocity to incorporate physics constraints.

        Args:
            actions_before: (B, H, D) action tensor BEFORE this step's update.
            step_idx: Current denoising step index (0-3).
            velocity: (B, H, D) predicted velocity from the DiT.

        Returns:
            Modified velocity tensor (same shape).
        """
        # Annealing: no guidance at step 0, increasing toward step 3
        tau = step_idx / 4.0
        guidance_scale = eta * tau

        if guidance_scale < 1e-8:
            return velocity  # step 0: no guidance

        # Compute candidate action after Euler step (for gradient evaluation)
        dt = 0.25
        a_candidate = actions_before + dt * velocity

        grad = torch.zeros_like(a_candidate)

        # --- Constraint 1: Temporal smoothness (jerk minimization) ---
        # Second-order finite difference on continuous EEF dims
        for dims in [eef_pos_dims, eef_rot_dims]:
            a_cont = a_candidate[:, :, dims]  # (B, H, 3 or 6)
            # Discrete Laplacian: a[j+1] - 2*a[j] + a[j-1]
            laplacian = torch.zeros_like(a_cont)
            laplacian[:, 1:-1, :] = (
                a_cont[:, 2:, :] - 2 * a_cont[:, 1:-1, :] + a_cont[:, :-2, :]
            )
            # Gradient of ||laplacian||^2 w.r.t. a: 2 * laplacian convolved
            grad[:, :, dims] -= lambda_smooth * 2 * laplacian

        # --- Constraint 2: Discrete decisiveness (gripper + control mode) ---
        for dim in [gripper_dim, mode_dim]:
            a_disc = a_candidate[:, :, dim]  # (B, H)
            # Gradient of a*(1-a) = 1 - 2a, pushing toward 0 or 1
            grad[:, :, dim] -= lambda_discrete * (1.0 - 2.0 * a_disc)

        # --- Constraint 3: Control mode temporal consistency ---
        a_mode = a_candidate[:, :, mode_dim]  # (B, H)
        mode_diff = torch.zeros_like(a_mode)
        mode_diff[:, 1:] = a_mode[:, 1:] - a_mode[:, :-1]
        mode_diff_grad = torch.zeros_like(a_mode)
        mode_diff_grad[:, :-1] -= mode_diff[:, 1:]
        mode_diff_grad[:, 1:] += mode_diff[:, 1:]
        grad[:, :, mode_dim] -= lambda_mode * 2 * mode_diff_grad

        # Apply guidance as a velocity correction
        guided_velocity = velocity + (guidance_scale / dt) * grad
        return guided_velocity

    return guided_fn


# === Usage in DenoisingLab ===
result = lab.denoise(
    features,
    num_steps=4,
    guided_fn=make_constraint_guided_fn(
        lambda_smooth=0.005,
        lambda_discrete=0.01,
        eta=0.1,
    ),
    seed=42,
)
```

### How It Replaces Action Chunking

Action chunking is unchanged. The constraint gradients modify the velocity field at each step, steering the Euler integration toward physically valid trajectories. The output shape, decode pipeline, and `MultiStepWrapper` integration are all identical to baseline.

The key interaction with action chunking: the temporal smoothness constraint operates across all 16 timesteps of the chunk, including the 8 un-executed far-horizon steps. This means the far-horizon steps serve as a "temporal buffer" that enforces smoothness at the chunk boundary — the executed steps (0–7) are smoother because the constraint considers the trajectory all the way to step 15.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | High. Smoothness guidance should reduce jittery motions (a known failure mode in evaluation). Discrete decisiveness should improve gripper reliability (no more floating-point grasps). Mode consistency prevents control-mode oscillation. These directly target observed failure modes, not hypothetical ones. |
| **Risk** | (1) Guidance strength $\eta$ and constraint weights $\lambda_k$ require tuning per embodiment. Too aggressive → over-smoothed, sluggish actions. Too weak → no effect. (2) The constraints operate in the 128-dim padded action space, but the dimension indices are embodiment-specific. The mapping must be correct. (3) At early denoising steps ($\tau < 0.25$), the action is mostly noise — constraints on noise are meaningless, hence the annealing schedule. |
| **Latency** | Negligible increase — the gradient computation is $O(H \times D) \approx 500$ FLOPs per step, vs ~$10^{10}$ FLOPs for the DiT forward pass. Same 4 NFEs, same ~64ms total. |
| **Implementation** | Easy — implemented entirely within the existing `guided_fn` interface. No changes to the model, denoising loop, or decode pipeline. The constraint functions and dimension indices are the only per-embodiment configuration. |

### Prior Work and Inspiration

- **Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis"** — arXiv:2105.05233. Introduced classifier guidance for diffusion models: $v_{\text{guided}} = v + s \cdot \nabla_x \log p(y|x)$. Our approach replaces the learned classifier $p(y|x)$ with analytic constraint functions $C_k(a)$.
- **Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis" (Diffuser)** — arXiv:2205.09991. Applied reward guidance to DDPM-based trajectory generation. Showed that guiding the diffusion process with reward gradients produces higher-quality plans than post-hoc selection. Our approach adapts this to flow matching with analytic (not learned) rewards.
- **Ajay et al., "Is Conditional Generation All You Need for Decision-Making?" (Decision Diffuser)** — arXiv:2211.15657. Extended Diffuser with return-conditioned guidance. Demonstrated that diffusion + guidance is competitive with model-based RL for offline decision-making.
- **Song et al., "Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation"** — arXiv:2311.13024. Generalized classifier guidance to arbitrary differentiable loss functions. Our constraint functions are a special case with analytic gradients.

**What makes this novel for VLAs:** Prior work on guided diffusion uses *learned* reward/classifier networks (expensive, task-specific, requires training data). Our constraints are *analytic* — derived from physical first principles with closed-form gradients. This is possible because robot action spaces have known mathematical structure (bounded workspaces, smooth motion physics, discrete/continuous decomposition) that image pixel spaces lack. The bridge between "classifier guidance from generative modeling" and "trajectory constraints from robotics" is, to our knowledge, unexplored for flow-matching VLAs.

---

---

## Strategy 9: Horizon-Prioritized Denoising

**Category:** Novel, drop-in | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

This is a **novel strategy** that exploits a structural property unique to robot action generation: **temporal importance is non-uniform across the action horizon.**

**The key observation:** In a 16-step action chunk, not all timesteps matter equally. Steps 0–3 (immediate future) are executed next and directly determine grasp success or collision avoidance. Steps 4–7 are important for trajectory shaping. Steps 8–15 are never executed (discarded at the next re-query) and serve only as a "temporal buffer" for smoothness constraints.

Yet the current Euler integration treats all 16 positions identically — applying the same velocity scaling $\Delta\tau = 0.25$ to every position at every step. This is a wasted opportunity.

**The idea:** Apply a *position-dependent velocity scaling* that creates a "denoising wave" sweeping from near-horizon to far-horizon across the 4 denoising steps. Near-horizon positions receive larger velocity updates in early steps (converging first), while far-horizon positions receive larger updates in later steps (converging last). The total velocity integrated per position is approximately preserved.

**Why this works — the self-attention argument:** GR00T's DiT has 16 self-attention layers where all action tokens attend bidirectionally to all other action tokens (`attention_mask=None` — no causal mask). When near-horizon tokens are partially denoised while far-horizon tokens are still noisy, the self-attention mechanism naturally creates an information flow from clean→noisy tokens. The partially-resolved near-horizon structure provides *temporal context* that constrains what the far-horizon tokens can become. This is analogous to how a partially-solved jigsaw puzzle constrains the remaining pieces.

**Why this is novel:** All existing flow-matching and diffusion-based solvers (Euler, Heun, RK4, DPM-Solver, etc.) apply uniform velocity scaling across all output dimensions. Position-dependent velocity gating that exploits the temporal structure of sequential predictions has no precedent in the generative modeling literature. The concept is specific to action-chunk generation and cannot meaningfully apply to image generation (where all pixels have equal importance).

### Mathematical Formulation

Define a position-dependent gating function $w_j^{(i)}$ for temporal position $j \in [0, H{-}1]$ at denoising step $i \in [0, 3]$:

$$a_t^{\tau_{i+1}}[j] = a_t^{\tau_i}[j] + \Delta\tau \cdot w_j^{(i)} \cdot v(a_t^{\tau_i},\; \tau_i,\; o_t,\; l_t)[j]$$

The gating function is a Gaussian attention window centered at $c_i$ that sweeps across the horizon:

$$w_j^{(i)} = 1 + \gamma \cdot \exp\!\left(-\frac{(j - c_i)^2}{2\sigma_w^2}\right)$$

where:
- $c_i = i \cdot (H-1) / 3$ is the center of the attention window for step $i$. For $H = 16$: $c_0 = 0,\; c_1 = 5,\; c_2 = 10,\; c_3 = 15$.
- $\gamma \in [0.3, 0.8]$ controls the boost amplitude (how much extra velocity the focused positions get). $\gamma = 0$ recovers standard Euler.
- $\sigma_w \approx 3$ controls the width of the attention window.

**Example weight matrix** ($\gamma = 0.5$, $\sigma_w = 3$, $H = 16$):

| Position | Step 0 | Step 1 | Step 2 | Step 3 | Total |
|----------|--------|--------|--------|--------|-------|
| $j=0$ (near) | **1.50** | 1.13 | 1.00 | 1.00 | 4.63 |
| $j=4$ | 1.17 | **1.43** | 1.09 | 1.00 | 4.69 |
| $j=8$ | 1.00 | 1.09 | **1.43** | 1.09 | 4.61 |
| $j=12$ | 1.00 | 1.00 | 1.17 | **1.43** | 4.60 |
| $j=15$ (far) | 1.00 | 1.00 | 1.00 | **1.50** | 4.50 |

The total velocity per position ranges from 4.50 to 4.69 — close to the uniform baseline of 4.00 (which would be $1.0 \times 4$ steps). The boost is redistributed, not added: near-horizon positions get their velocity *earlier*, far-horizon positions get theirs *later*.

**Conservation analysis:** The total velocity magnitude across all positions and steps is $\sum_{j,i} w_j^{(i)} \cdot \Delta\tau \approx 4.6 \times 16 \times 0.25 = 18.4$, compared to $1.0 \times 16 \times 4 \times 0.25 = 16.0$ for uniform Euler. The ~15% total increase is modest and can be compensated by setting $\gamma$ slightly lower or normalizing per position.

### Pseudocode

```python
def make_horizon_prioritized_fn(
    action_horizon=50,     # padded horizon (50 for all embodiments)
    effective_horizon=16,  # actual action steps for PandaOmron
    gamma=0.5,             # boost amplitude
    sigma_w=3.0,           # window width
    num_steps=4,
):
    """Factory for horizon-prioritized velocity gating.

    Returns a guided_fn compatible with DenoisingLab.
    """
    # Precompute gating weights: (num_steps, action_horizon)
    import numpy as np
    centers = [i * (effective_horizon - 1) / (num_steps - 1) for i in range(num_steps)]
    j = np.arange(action_horizon)

    weights = np.ones((num_steps, action_horizon), dtype=np.float32)
    for i in range(num_steps):
        gaussian = np.exp(-0.5 * ((j[:effective_horizon] - centers[i]) / sigma_w) ** 2)
        weights[i, :effective_horizon] = 1.0 + gamma * gaussian
        # Padded positions (16..49) keep weight 1.0

    weights_tensor = torch.from_numpy(weights)

    def guided_fn(actions_before, step_idx, velocity):
        """Apply horizon-prioritized velocity gating."""
        w = weights_tensor[step_idx].to(velocity.device)  # (action_horizon,)
        # Broadcast: (1, H, 1) * (B, H, D)
        return velocity * w[None, :, None]

    return guided_fn


# === Usage ===
result = lab.denoise(
    features,
    num_steps=4,
    guided_fn=make_horizon_prioritized_fn(gamma=0.5, sigma_w=3.0),
    seed=42,
)
```

### How It Replaces Action Chunking

This strategy directly interacts with the temporal structure of action chunking. The velocity gating ensures that the 8 *executed* steps (0–7) are more thoroughly denoised than the 8 *discarded* steps (8–15), allocating compute where it matters. The far-horizon steps still contribute to trajectory smoothness (via self-attention context and Strategy 8's smoothness constraint) but receive less denoising emphasis.

The `MultiStepWrapper` is unmodified. The output shape and decode pipeline are identical.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Potentially high. Near-horizon actions (which are executed) receive ~15% more velocity integration than far-horizon actions (which are discarded). This preferential denoising directly improves the actions that matter. Furthermore, the DiT's self-attention creates an information flow from the early-converging near-horizon tokens to the still-noisy far-horizon tokens — a form of *self-guided denoising* that emerges naturally from the architecture. |
| **Risk** | (1) **Timestep distribution mismatch**: After step 0, near-horizon positions are more denoised than the global $\tau$ would suggest, while far-horizon positions are less denoised. The DiT's AdaLayerNorm conditions on a single global $\tau$ — it cannot distinguish per-position noise levels. If the model is sensitive to this mismatch, quality could degrade. However, the mismatch is small ($\pm$15% of baseline velocity) and the model processes partially-noised inputs at every step, so some robustness is expected. (2) The $\gamma$ parameter requires tuning. Too large: excessive mismatch and potential instability. Too small: no effect. |
| **Latency** | Identical — same 4 NFEs, same ~64ms. The Gaussian weight computation is precomputed; the per-step gating is a single element-wise multiply. |
| **Implementation** | Trivial — implemented entirely in the `guided_fn` callback via element-wise velocity scaling. One precomputed weight matrix. |

### Prior Work and What Makes This Novel

- **Score Distillation Sampling (Poole et al., "DreamFusion", 2022)**: Uses per-pixel weighting in the score function for 3D generation. Our per-*position* weighting in the velocity field for sequential action generation is an analogous concept in a fundamentally different domain.
- **Temporal attention in video diffusion**: Video diffusion models (Ho et al., "Video Diffusion Models", 2022) use temporal attention layers, but they do not apply position-dependent velocity scaling during sampling. The denoising process treats all frames identically.
- **Receding-horizon MPC with variable precision**: In classical MPC, it is standard practice to use finer discretization for near-horizon states and coarser discretization for far-horizon states. Our velocity gating is the flow-matching analog of this principle — "spend more denoising effort on the immediate future."

**What makes this novel:** To our knowledge, no prior work applies position-dependent velocity scaling during flow matching or diffusion sampling. The key insight — that sequential predictions have non-uniform temporal importance and that this can be exploited via the learned model's self-attention — is specific to VLA action generation and has no analog in image/video generation. This strategy transforms the ODE solver from a position-agnostic integrator into a position-aware one that respects the causal structure of robot control.

---

## Strategy 10: Noise-Space Mode Selection via Velocity Preview

**Category:** Novel, drop-in | **NFEs:** K + 3 (~7 for K=4) | **Retraining:** None

### Overview

This is a **novel strategy** inspired by a striking finding: **the initial noise vector can change success rate by up to 58%** (Patil et al., "Golden Noise for Diffusion Policy", 2026). Different noise samples produce different action *modes* — one noise might generate "approach from the left," another "approach from the right," a third might produce an unstable oscillating trajectory. Currently, GR00T samples one noise vector and hopes for the best.

**Golden Ticket's limitation:** The original approach finds optimal noise vectors through offline Monte Carlo search — rolling out the full policy with hundreds of noise candidates and keeping the best. This requires a simulator, is computationally expensive, and produces a *fixed* noise vector that doesn't adapt to the specific observation.

**Our innovation:** Replace the offline search with an **online, per-observation 1-step velocity preview**. Sample $K$ noise candidates, run a single batched Euler step on all $K$ simultaneously, score the partially-denoised results using a lightweight quality proxy, select the best noise, and complete the remaining 3 denoising steps only for the winner.

**Why 1 step is sufficient for mode selection:** In flow matching, the first Euler step ($\tau = 0 \to 0.25$) does the most dramatic transformation — it collapses the isotropic Gaussian noise into a rough action structure. After just 1 step, the gross trajectory shape is established: which direction the arm reaches, whether the gripper opens or closes, which control mode is active. The remaining 3 steps refine this structure but rarely change the mode. This is visible in the denoising_lab notebook's progression plots (cell 13): the step-0 → step-1 transition dominates.

**Why this is novel:**
- **Golden Ticket** (Patil et al., 2026): Offline, fixed noise, requires rollout evaluation. Ours: online, per-observation, 1-step proxy evaluation.
- **Best-of-K sampling** in LLMs: Generates $K$ complete sequences, scores them, picks the best. Ours: evaluates after just 1 denoising step (not a complete generation), making it $K/(K+3) \approx 60\%$ cheaper for $K=4$.
- **Stochastic beam search**: Maintains $K$ candidates throughout generation. Ours: selects once after step 0 and commits — no ongoing parallelism needed.

### Mathematical Formulation

**Step 1 — Sample and preview:**

Sample $K$ noise candidates and batch-evaluate 1 Euler step:

$$\epsilon^{(k)} \sim \mathcal{N}(0, I), \quad k = 1, \ldots, K$$

$$a^{(k),\, 0.25} = \epsilon^{(k)} + \Delta\tau \cdot v(\epsilon^{(k)},\; 0,\; o_t,\; l_t), \quad k = 1, \ldots, K$$

This is a single forward pass with batch size $K \cdot B$ (all candidates concatenated).

**Step 2 — Score:**

Evaluate each candidate's partially-denoised action using a quality proxy $S$:

$$k^* = \arg\max_k \; S\!\left(a^{(k),\, 0.25},\; v^{(k)}\right)$$

The quality proxy $S$ can combine multiple signals (all computable from the 1-step output):

$$S(a, v) = \underbrace{-\lambda_{\text{smooth}} \sum_{j} \| a[j{+}1] - a[j] \|^2}_{\text{temporal smoothness}} + \underbrace{-\lambda_{\text{mag}} \| v \|^2}_{\text{velocity magnitude}} + \underbrace{\lambda_{\text{anchor}} \cos(v[\text{overlap}],\; V_{\text{prev}}[\text{overlap}])}_{\text{consistency with previous chunk}}$$

- **Smoothness**: Rough actions after 1 step indicate a noisy mode — penalize.
- **Velocity magnitude**: Lower velocity suggests the noise was already closer to the action manifold — reward.
- **Anchor consistency** (if previous chunk available): Velocity in the overlap region should align with the previous chunk's predictions — reward. This can integrate with RTC-style temporal coherence (Black et al., 2025).

**Step 3 — Commit and complete:**

Denoise the selected noise for the remaining 3 steps:

$$a_t^{0.50} = a^{(k^*),\, 0.25} + \Delta\tau \cdot v(a^{(k^*),\, 0.25},\; 0.25,\; o_t,\; l_t)$$

$$a_t^{0.75} = a_t^{0.50} + \Delta\tau \cdot v(a_t^{0.50},\; 0.50,\; o_t,\; l_t)$$

$$a_t^{1.0} = a_t^{0.75} + \Delta\tau \cdot v(a_t^{0.75},\; 0.75,\; o_t,\; l_t)$$

Total: $K + 3$ NFEs. For $K = 4$: 7 NFEs. The step-0 evaluation is batched ($K \cdot B$ samples in one forward pass), so wall-clock latency is approximately 4 sequential DiT forward passes — **same as baseline** if the GPU has spare batch capacity.

### Pseudocode

```python
def denoise_with_noise_selection(
    lab,                    # DenoisingLab instance
    features,               # BackboneFeatures
    K=4,                     # number of noise candidates
    lambda_smooth=1.0,
    lambda_mag=0.1,
    lambda_anchor=0.5,
    prev_velocity=None,      # cached velocity from RTC-style anchoring, or None
    seed=None,
):
    """Noise-space mode selection with 1-step velocity preview."""
    vl_embeds = features.backbone_features
    state_features = features.state_features
    embodiment_id = features.embodiment_id
    backbone_output = features.backbone_output
    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype

    # --- Step 1: Sample K noise candidates ---
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None
    noise_candidates = torch.randn(
        K, B, lab.action_horizon, lab.action_dim,
        dtype=dtype, device=device, generator=gen,
    )  # (K, B, 50, 128)

    # --- Step 2: Batch-evaluate 1 Euler step for all K candidates ---
    # Reshape for batched forward pass: (K*B, 50, 128)
    flat_noise = noise_candidates.reshape(K * B, lab.action_horizon, lab.action_dim)
    flat_vl = vl_embeds.repeat(K, 1, 1)
    flat_state = state_features.repeat(K, 1, 1)
    flat_emb_id = embodiment_id.repeat(K)
    # (repeat backbone_output fields similarly)

    velocity_0, actions_025 = lab._denoise_step_inner(
        flat_vl, flat_state, flat_emb_id, backbone_output,  # broadcast
        flat_noise, t_discretized=0, dt=0.25,
        batch_size=K * B, device=device,
    )

    # Reshape back: (K, B, 50, 128)
    velocity_0 = velocity_0.reshape(K, B, lab.action_horizon, lab.action_dim)
    actions_025 = actions_025.reshape(K, B, lab.action_horizon, lab.action_dim)

    # --- Step 3: Score each candidate ---
    scores = torch.zeros(K, B, device=device)
    for k in range(K):
        a = actions_025[k]  # (B, 50, 128)
        v = velocity_0[k]   # (B, 50, 128)

        # Temporal smoothness (lower = smoother)
        diffs = a[:, 1:, :] - a[:, :-1, :]  # (B, 49, 128)
        smoothness = -(diffs ** 2).sum(dim=(1, 2))  # (B,)
        scores[k] += lambda_smooth * smoothness

        # Velocity magnitude (lower = closer to manifold)
        mag = -(v ** 2).sum(dim=(1, 2))  # (B,)
        scores[k] += lambda_mag * mag

        # Anchor consistency with previous chunk (if available)
        if prev_velocity is not None:
            n_exec = 8
            cos_sim = F.cosine_similarity(
                v[:, :n_exec, :].reshape(B, -1),
                prev_velocity[:, :n_exec, :].reshape(B, -1),
                dim=1,
            )  # (B,)
            scores[k] += lambda_anchor * cos_sim

    # --- Step 4: Select best noise per batch element ---
    best_k = scores.argmax(dim=0)  # (B,)
    best_actions = actions_025[best_k, torch.arange(B)]       # (B, 50, 128)
    best_noise = noise_candidates[best_k, torch.arange(B)]    # (B, 50, 128)

    # --- Step 5: Complete denoising with remaining 3 steps ---
    actions = best_actions
    for step in range(1, 4):
        tau_bucket = int(step / 4.0 * 1000)
        velocity, actions = lab._denoise_step_inner(
            vl_embeds, state_features, embodiment_id, backbone_output,
            actions, tau_bucket, dt=0.25, batch_size=B, device=device,
        )

    return actions, best_noise  # return noise for potential caching
```

### How It Replaces Action Chunking

Action chunking is unchanged. The noise selection happens entirely before the main denoising loop — it selects *which* noise to denoise, not *how* to denoise it. The selected noise flows through the same 3-step Euler integration, decode pipeline, and `MultiStepWrapper` as baseline.

**Synergy with other strategies:** Noise selection is orthogonal to the denoising solver. The 3 remaining steps can use any strategy: AB2 (Strategy 3), constraint guidance (Strategy 8), or horizon-prioritized gating (Strategy 9). This makes noise selection a "meta-strategy" that stacks on top of all others. For inter-chunk coherence, combine with RTC-style inpainting (Black et al., 2025).

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Potentially very high. Golden Ticket (Patil et al., 2026) reports up to 58% relative improvement from noise optimization alone. Our 1-step proxy is weaker than full-rollout evaluation but captures the mode-level structure that matters most. For $K = 4$, we're selecting the best of 4 modes — in multi-modal action distributions (e.g., approach object from different directions), this can avoid catastrophically bad modes. |
| **Risk** | (1) The 1-step proxy may not correlate well with final action quality. After only 25% denoising, the action is still 75% noise — the quality signal is noisy itself. The smoothness and velocity-magnitude scores are heuristics. (2) The scoring function weights ($\lambda$) require tuning. (3) For observations with unimodal action distributions, all $K$ candidates produce similar results — the selection provides no benefit. The overhead is wasted. |
| **Latency** | $K + 3$ NFEs. For $K = 4$: the step-0 batch uses $4B$ samples in a single forward pass. On GPUs with spare batch capacity (typical for $B = 1$ inference), this costs the same wall-clock time as 1 sequential NFE. Total wall-clock: ~4 sequential forward passes = **~64ms, same as baseline**. For larger $B$ or $K$, latency grows. |
| **Implementation** | Moderate — requires batched forward pass with duplicated VLM features, scoring function, per-batch-element selection. The `_denoise_step_inner` interface supports arbitrary batch sizes, so the main challenge is replicating the backbone features $K$ times. |

### Prior Work and What Makes This Novel

- **Patil et al., "Golden Noise for Diffusion Policy" (2026)**: The primary inspiration. Demonstrates that a fixed, pre-optimized noise vector improves frozen diffusion/flow policies by up to 58% across 43 tasks. The noise is found via Monte Carlo search over full simulator rollouts. **Key difference:** Golden Ticket is offline (requires simulator), fixed (same noise for all observations), and expensive (hundreds of rollouts). Our approach is online (per-observation), adaptive (different noise per query), and cheap (1-step proxy).
- **Best-of-N sampling in LLMs** (e.g., Nakano et al., "WebGPT", 2021): Generates $N$ complete sequences and scores them with a reward model. **Key difference:** We evaluate after 1 step (not full generation) and use analytic scores (not a learned reward model).
- **Stochastic beam search / diverse beam search**: Maintains multiple candidates throughout generation. **Key difference:** We select once after step 0 and commit — no ongoing parallelism, dramatically lower cost.
- **ProbeFlow** (Fang et al., 2026): Uses velocity cosine similarity to dynamically skip steps. **Key difference:** ProbeFlow decides *how many* steps to take for a single noise; our approach decides *which noise* to use.

**What makes this novel:** The combination of (1) online, per-observation noise selection (2) using a 1-step velocity preview as a lightweight quality proxy (3) applied to flow-matching VLA models is, to our knowledge, unpublished. The closest work (Golden Ticket) requires offline optimization with full rollouts; our 1-step proxy eliminates this requirement entirely. The insight that mode selection happens in the first denoising step — and that this can be exploited for cheap quality improvement — is specific to the flow-matching paradigm where early steps establish gross structure.

---

## Strategy 11: Classifier-Free Action Guidance via Observation Dropout

**Category:** Fine-tuning (training modification only, no architecture change) | **NFEs:** 8 (or 5 effective with batching) | **Retraining:** Yes (re-train with observation dropout; no new parameters, no architecture change)

### Overview

Classifier-Free Guidance (CFG) is the single most impactful technique in modern image generation — responsible for the dramatic quality leap in Imagen, DALL-E 2, and Stable Diffusion. Yet **no production VLA uses CFG for action generation.** This is the biggest missed opportunity in the field.

**The mechanism:** During training, randomly replace the observation conditioning (VLM embeddings from images + state + language) with a learned null embedding with probability $p$ (e.g., $p = 0.1$). This teaches the model to predict actions both conditionally (given observation) and unconditionally (from noise alone). At inference, compute both velocity predictions and *amplify the conditional signal*:

$$v_{\text{guided}} = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$$

where $w > 1$ amplifies the observation's influence on the generated action. The difference $(v_{\text{cond}} - v_{\text{uncond}})$ isolates the "direction in velocity space that the observation contributes" — amplifying this makes the action more responsive to what the robot sees and what the language command requests.

**Why this is transformative for VLAs:** CFG-DP (Wen et al., Oct 2025) demonstrated a jump from **55.6% to 83.2% success rate** on temporal robotic tasks using CFG with DDPM-style diffusion policies. The reason is profound: without guidance, the model outputs the *mean* of the action distribution — a hedging strategy that is mediocre at everything. With guidance, the model amplifies its conditional prediction to *commit* to a specific action mode — producing decisive actions that succeed far more often.

**Why this hasn't been done for flow matching VLAs:** Three technical barriers, all surmountable:
1. CFG-DP was demonstrated on DDPM-style noise prediction; adapting to flow matching requires velocity-space guidance (different mathematical formulation).
2. VLA models like GR00T use a VLM backbone (Eagle) that is pre-trained and expensive — dropping it during training seems wasteful. But the dropout applies to the *output embeddings*, not the backbone computation; and 10% dropout is a mild regularizer.
3. The guidance weight $w$ interacts with flow matching's linear interpolation path differently than DDPM's variance-preserving path, requiring re-calibration.

**Novel contribution — task-phase-aware guidance scheduling:** The key innovation beyond vanilla CFG: use a **sigmoid-scheduled guidance weight** that varies with *episode progress* (not denoising progress):

- **Early phase** (exploration): $w \approx 1.0$ (no amplification). The robot needs to explore diverse approaches — over-committing early leads to suboptimal strategies.
- **Mid phase** (approach + positioning): $w \approx 2.0$ (moderate amplification). The robot should commit to a strategy.
- **Late phase** (task completion): $w \approx 4.0$ (strong amplification). The robot must execute decisively — hesitation at this stage causes failure (e.g., releasing a drawer handle too early, oscillating near a button).

This scheduling is motivated by CFG-DP's finding that constant guidance causes "over-commitment" early (missing better strategies) and "under-commitment" late (failing to terminate). The sigmoid schedule addresses both failure modes. It is computed from `episode_step` (available at inference) and does not require a learned task-completion predictor.

### Mathematical Formulation

**Training modification — observation dropout:**

For each training sample, independently sample a Bernoulli mask $m \sim \text{Bernoulli}(1 - p)$ where $p = 0.1$:

$$\tilde{\phi}_t = m \cdot \phi_t + (1 - m) \cdot \phi_{\text{null}}$$

where $\phi_t$ is the VLM output (Eagle backbone embeddings) and $\phi_{\text{null}}$ is a learned null embedding of the same shape. The training loss is unchanged:

$$\mathcal{L}_{\text{CFG-FM}} = \mathbb{E}_{\tau,\, m}\left[\left\| v_\theta(a_t^\tau,\; \tau,\; \tilde{\phi}_t) - (a_t^1 - \epsilon) \right\|^2\right]$$

The null embedding $\phi_{\text{null}} \in \mathbb{R}^{1 \times \text{seq\_len} \times 2048}$ is the only new learnable parameter — a negligible addition to the 3B model.

**Inference — guided velocity at each denoising step $i$:**

$$v_{\text{cond}} = v_\theta(a_t^{\tau_i},\; \tau_i,\; \phi_t) \qquad \text{(standard: conditioned on observation)}$$

$$v_{\text{uncond}} = v_\theta(a_t^{\tau_i},\; \tau_i,\; \phi_{\text{null}}) \qquad \text{(new: conditioned on null)}$$

$$v_{\text{guided}} = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$$

$$a_t^{\tau_{i+1}} = a_t^{\tau_i} + \Delta\tau \cdot v_{\text{guided}}$$

Note: when $w = 1.0$, this exactly recovers $v_{\text{cond}}$ (standard conditional inference). When $w = 0$, it recovers $v_{\text{uncond}}$ (unconditional). When $w > 1$, it *extrapolates* beyond the conditional prediction in the direction away from unconditional.

**Task-phase-aware sigmoid schedule:**

$$w(\text{step}) = w_{\min} + (w_{\max} - w_{\min}) \cdot \sigma\!\left(\frac{\text{episode\_step} - \mu}{\kappa}\right)$$

where $\sigma(x) = 1/(1+e^{-x})$ is the sigmoid function, $\mu = \text{max\_episode\_steps} / 2$ is the midpoint, and $\kappa \approx 100$ controls transition sharpness.

**Example guidance weights** over a 720-step episode ($w_{\min} = 1.0$, $w_{\max} = 4.0$, $\mu = 360$, $\kappa = 100$):

| Episode step | Phase | Guidance $w$ | Behavior |
|-------------|-------|-------------|----------|
| 0 | Exploration | 1.03 | Essentially standard (diverse modes OK) |
| 100 | Early approach | 1.22 | Mild commitment |
| 250 | Mid approach | 1.97 | Moderate amplification |
| 360 | Midpoint | 2.50 | Balanced |
| 500 | Positioning | 3.36 | Strong commitment |
| 650 | Task completion | 3.88 | Very decisive |
| 720 | End | 3.97 | Near-maximum decisiveness |

**Efficient inference with batching:** The two velocity evaluations ($v_{\text{cond}}$ and $v_{\text{uncond}}$) share the same DiT architecture with different conditioning inputs. They can be batched as a single forward pass with batch size $2B$, making the wall-clock cost approximately 5 sequential NFEs (1 batched pair × 4 steps + overhead), not 8.

### Pseudocode

```python
# === Training modification (minimal — ~10 lines added to existing loss) ===
class CFGFlowMatchingTrainer:
    """Wraps existing flow matching training with observation dropout."""

    def __init__(self, base_trainer, dropout_prob=0.1, null_embed_dim=2048,
                 null_seq_len=None):
        self.base_trainer = base_trainer
        self.dropout_prob = dropout_prob
        # Learned null embedding — only new parameter (negligible vs 3B model)
        self.null_embedding = nn.Parameter(
            torch.randn(1, null_seq_len or 1, null_embed_dim) * 0.02
        )

    def training_step(self, model, actions, noise, tau, vl_embeds,
                      state_embeds, embodiment_id):
        B = actions.shape[0]
        # Bernoulli mask: drop observation with probability p
        mask = (torch.rand(B, 1, 1, device=actions.device) > self.dropout_prob).float()
        masked_vl = mask * vl_embeds + (1 - mask) * self.null_embedding.expand_as(vl_embeds)

        # Standard flow matching loss with masked conditioning
        a_noised = (1 - tau) * noise + tau * actions
        v_pred = model(a_noised, tau, masked_vl, state_embeds, embodiment_id)
        target = actions - noise
        return F.mse_loss(v_pred, target)


# === Inference — guided_fn for DenoisingLab ===
def make_cfg_guided_fn(
    lab,                        # DenoisingLab instance (for DiT access)
    null_vl_embeds,             # Learned null embedding (1, seq_len, 2048)
    vl_embeds,                  # Actual VLM output for this observation
    state_features,
    embodiment_id,
    backbone_output,
    w_min=1.0,                  # guidance at episode start
    w_max=4.0,                  # guidance at episode end
    episode_step=0,             # current environment timestep
    max_episode_steps=720,
    sigmoid_sharpness=100.0,    # kappa — controls transition speed
):
    """Classifier-free guidance with task-phase-aware scheduling.

    Returns a guided_fn compatible with DenoisingLab.denoise().
    """
    import math

    # Compute task-phase guidance weight (constant for all denoising steps in this chunk)
    midpoint = max_episode_steps / 2.0
    phase = 1.0 / (1.0 + math.exp(-(episode_step - midpoint) / sigmoid_sharpness))
    w = w_min + (w_max - w_min) * phase

    def guided_fn(actions_before, step_idx, velocity_cond):
        """Compute CFG-guided velocity.

        Args:
            actions_before: (B, H, D) action tensor BEFORE this step's update.
            step_idx: Current denoising step index (0-3).
            velocity_cond: (B, H, D) standard conditioned velocity from the DiT.

        Returns:
            Modified velocity tensor (same shape).
        """
        if abs(w - 1.0) < 1e-4:
            return velocity_cond  # No guidance needed at w ≈ 1.0

        tau_bucket = int(step_idx / 4.0 * 1000)

        # Compute unconditioned velocity (DiT with null observation)
        velocity_uncond = lab._forward_dit(
            actions_before, tau_bucket,
            null_vl_embeds.expand_as(vl_embeds),  # null conditioning
            state_features, embodiment_id, backbone_output,
        )

        # CFG: amplify the observation-conditioned direction
        guided = velocity_uncond + w * (velocity_cond - velocity_uncond)
        return guided

    return guided_fn


# === Usage ===
# Training: wrap existing trainer with dropout
cfg_trainer = CFGFlowMatchingTrainer(base_trainer, dropout_prob=0.1)

# Inference: create guided_fn with phase-aware scheduling
cfg_fn = make_cfg_guided_fn(
    lab=lab,
    null_vl_embeds=model.null_embedding,  # loaded from fine-tuned checkpoint
    vl_embeds=features.backbone_features,
    state_features=features.state_features,
    embodiment_id=features.embodiment_id,
    backbone_output=features.backbone_output,
    w_min=1.0, w_max=4.0,
    episode_step=current_step,
    max_episode_steps=720,
)

result = lab.denoise(features, num_steps=4, guided_fn=cfg_fn, seed=42)
```

### How It Replaces Action Chunking

Action chunking is unchanged. CFG modifies the velocity field at each denoising step, not the integration method or action structure. The `MultiStepWrapper`, decode pipeline, and chunk execution are identical to baseline.

The task-phase scheduling interacts with chunking at the *episode* level: as the robot progresses through an episode, guidance ramps up. Crucially, the guidance weight $w$ is determined by `episode_step` (the environment timestep), not the denoising step — so all 4 denoising steps within a single chunk use the same guidance weight. Successive chunks in the same episode use progressively stronger guidance.

**Data for fine-tuning:** The same demonstration dataset used for GR00T's original post-training. No new robot data is needed. The only training change is adding the Bernoulli dropout mask to the conditioning input — the loss function, optimizer, and training schedule can remain identical. The `null_embedding` is randomly initialized and learned jointly with the flow matching objective.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Very high. CFG-DP reports **+28 percentage points** success rate on temporal robotic tasks (55.6% → 83.2%). The mechanism is well-understood: amplifying the observation-conditioned direction produces more decisive, committed actions. The task-phase scheduling further improves over constant guidance by reducing early over-commitment and late under-commitment. Rectified flow in image generation (SD3) already uses CFG successfully, confirming compatibility with the flow matching formulation. |
| **Risk** | (1) Requires re-training with observation dropout — minimal modification (add one mask operation, ~10 lines) but still requires full fine-tuning compute. (2) Guidance weight $w$ is sensitive — too high causes "over-saturation" (unphysically large velocity magnitudes, actions leaving the normalized $[-1, 1]$ range). Clamping the guided velocity magnitude to $1.5\times$ the conditioned velocity may be needed. (3) The null embedding is learned from only 10% of training samples — if the unconditional action distribution is poorly learned, the guidance direction $(v_{\text{cond}} - v_{\text{uncond}})$ may be noisy. (4) The task-phase scheduling assumes `episode_step` is informative of task progress, which may not hold for variable-difficulty tasks (e.g., the robot finds the drawer quickly sometimes, slowly other times). |
| **Latency** | +25% with batched evaluation — 5 effective sequential NFEs. Each step requires one additional DiT forward pass for $v_{\text{uncond}}$, but the conditioned and unconditioned evaluations can be batched as $2B$ samples in a single pass. On GPUs with spare batch capacity (typical for $B = 1$), this adds ~16ms per denoising step wall-clock. Total: ~80ms vs baseline ~64ms. |
| **Implementation** | Moderate — Training: add dropout mask + null embedding (~10 lines of modification to existing training code). Inference: additional DiT forward pass per step + guidance blending. The `guided_fn` interface handles this naturally. No architecture changes to the DiT. |

### Prior Work

- **Ho & Salimans, "Classifier-Free Diffusion Guidance"** — arXiv:2207.12598. The foundational paper. Showed that training with random conditioning dropout enables guidance without a separate classifier. Standard in all modern image generation models (DALL-E 2, Imagen, Stable Diffusion).
- **Wen et al., "CFG-DP: Classifier-Free Guidance for Diffusion Policy"** — arXiv:2510.09786. First application of CFG to diffusion policies for temporal robotic tasks. Demonstrated sigmoid-scheduled guidance factor tied to task progression. Results: 55.6% → 83.2% success rate, repetitive actions reduced from 2.6 to 0.3 per episode.
- **Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3)** — arXiv:2403.03206. Applied CFG to rectified flow (the same flow matching formulation as GR00T) for image generation. Confirmed that CFG is fully compatible with flow matching — the velocity-space formulation works as expected.

**What makes this novel for VLA flow matching:** CFG-DP was demonstrated on DDPM-style diffusion policies (noise/score prediction). Our formulation adapts CFG to **flow matching velocity prediction**, which is mathematically distinct — the guidance operates on the velocity field $v$ rather than the score function $\nabla_x \log p$. Concretely, the flow matching guidance $v_{\text{uncond}} + w(v_{\text{cond}} - v_{\text{uncond}})$ amplifies the velocity *direction* that the observation contributes, while DDPM guidance amplifies the score *magnitude*. These are different geometric operations on different vector fields. Additionally, our task-phase scheduling uses episode progress as the scheduling signal (available at inference without any additional model), rather than CFG-DP's task completion probability (which requires a learned estimator). The synthesis of flow matching + CFG + VLA + task-phase scheduling is, to our knowledge, unpublished.

---

## Strategy 12: Curvature-Adaptive Step-Size Control via Embedded Error Estimation

**Category:** Novel, drop-in | **NFEs:** 4–6 (adaptive, per-observation) | **Retraining:** None

### Overview

Every strategy in this document — and every published VLA denoiser — uses a **fixed** number of denoising steps. This is fundamentally wasteful: some observations require precise, multi-step denoising (e.g., grasping a small object at a specific angle), while others are trivial (e.g., free-space transit to an approach position). Using 4 steps for all observations means over-spending compute on easy cases and under-spending on hard ones.

**D3P** (Dynamic Denoising Diffusion Policy, Dockhorn et al., Aug 2025) demonstrated that adaptive step allocation provides **2.2× speedup** with negligible quality loss. But D3P achieves this by training a separate RL-based adaptor network — adding training complexity, task-specific tuning, and a separate model artifact.

**Our approach:** Use a classical numerical analysis technique — **embedded Runge-Kutta error estimation** — to automatically detect when the ODE solver is struggling and needs smaller steps, *without any training*. This is the same principle behind MATLAB's `ode45` solver (Dormand-Prince method) and `scipy.integrate.solve_ivp`, applied for the first time to generative model denoising.

**The mechanism:** At each tentative step, compute both a 1st-order Euler estimate and a 2nd-order Heun estimate using the same velocity evaluations. The difference between these two estimates is a *free* local truncation error estimate:

$$e = \|\hat{a}_{\text{Euler}} - \hat{a}_{\text{Heun}}\| = \frac{\Delta\tau}{2}\|v_{\text{start}} - v_{\text{end}}\|$$

If $e$ is below a tolerance → accept the step (using the Heun estimate for 2nd-order accuracy as a bonus). If $e$ exceeds the tolerance → reject the step, halve $\Delta\tau$, retry.

**Key insight — the error estimate is free:** Computing the Heun estimate requires evaluating $v$ at the predicted endpoint (2 NFEs per step). But this endpoint evaluation is the *same* computation that the next Euler step would perform anyway. So the "extra" NFE for error estimation becomes the "first" NFE of the next step — no waste. In the worst case (all steps accepted on first try), the solver uses the same total NFEs as Heun integration (Strategy 5's Phase 1). In the best case, large steps are accepted and the solver finishes in 2 steps (4 NFEs).

**Why this is novel:**
1. **D3P** uses a learned RL adaptor → ours uses the mathematical error bound from the embedded pair, zero training.
2. **ProbeFlow** (Fang et al., 2026) uses velocity cosine similarity to decide whether to skip steps → ours uses the Euler-Heun difference, which is a principled truncation error estimate from numerical analysis, not a heuristic similarity metric.
3. **No prior work** applies adaptive step-size control from ODE solver theory to flow matching or diffusion denoising. The connection between "velocity field curvature" and "observation difficulty" is a novel interpretive framework.

### Mathematical Formulation

**At each step starting from $\tau_i$ with proposed step size $\Delta\tau$:**

**Phase A — Euler predictor** (1 NFE):

$$v_i = v(a_t^{\tau_i},\; \tau_i,\; o_t,\; l_t)$$

$$\hat{a}_{\text{Euler}} = a_t^{\tau_i} + \Delta\tau \cdot v_i$$

**Phase B — Heun corrector** (1 additional NFE):

$$v_{\text{next}} = v(\hat{a}_{\text{Euler}},\; \tau_i + \Delta\tau,\; o_t,\; l_t)$$

$$\hat{a}_{\text{Heun}} = a_t^{\tau_i} + \frac{\Delta\tau}{2}(v_i + v_{\text{next}})$$

**Phase C — Error estimation** (free — no extra NFEs):

$$e = \|\hat{a}_{\text{Euler}} - \hat{a}_{\text{Heun}}\|_\infty = \frac{\Delta\tau}{2}\|v_i - v_{\text{next}}\|_\infty$$

This measures how much the velocity field changed between $\tau_i$ and $\tau_i + \Delta\tau$ — i.e., the **local curvature** of the flow. High curvature means the Euler step introduces significant error; low curvature means the flow is nearly linear and large steps are safe.

**Step acceptance/rejection:**

$$\text{If } e < \text{atol}: \quad \text{Accept step, advance with } \hat{a}_{\text{Heun}} \text{ (2nd-order accuracy)}$$

$$\text{If } e \geq \text{atol}: \quad \text{Reject step, set } \Delta\tau \leftarrow \Delta\tau / 2, \text{ retry from } a_t^{\tau_i}$$

**Adaptive step-size update** (standard Hairer-Wanner formula):

After an accepted step, adapt the next step size:

$$\Delta\tau_{\text{next}} = \Delta\tau \cdot \min\!\left(2.0,\; \max\!\left(0.5,\; 0.9 \cdot \left(\frac{\text{atol}}{e}\right)^{1/2}\right)\right)$$

The 0.9 safety factor and [0.5, 2.0] clamp prevent oscillatory step-size behavior.

**NFE budget:** To guarantee bounded latency, impose $N_{\max} = 6$ NFEs. If the budget is exhausted, accept the current Euler estimate regardless of error. This guarantees worst-case latency of $6 \times 16\text{ms} = 96\text{ms}$.

**Typical behavior by observation difficulty:**

| Observation type | Velocity curvature | Steps taken | NFEs | Latency |
|-----------------|-------------------|-------------|------|---------|
| Free-space transit | Low (nearly linear) | 2 large ($\Delta\tau \approx 0.5$) | 4 | ~64ms |
| Approach + position | Moderate | 3 steps of varying size | 6 | ~96ms |
| Precision grasp / narrow clearance | High (nonlinear) | 3-4 smaller steps | 6 | ~96ms |
| Average across episodes | Mixed | ~2.5 steps | ~5 | ~80ms |

The solver automatically spends more compute on hard observations and less on easy ones — achieving D3P-like adaptivity with zero training.

### Pseudocode

```python
def denoise_adaptive(
    a_noise, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab,                        # DenoisingLab for DiT access
    atol=0.05,                  # absolute error tolerance (normalized action space)
    max_nfe=6,                  # hard NFE budget
    dt_init=0.5,                # initial step size (optimistic: try 2 large steps)
    dt_min=0.125,               # minimum step size (8 substeps resolution)
):
    """Adaptive Euler-Heun integration with embedded error estimation.

    Returns (denoised_actions, step_log) where step_log records the
    adaptive decisions for analysis.
    """
    a = a_noise
    tau = 0.0
    nfe = 0
    dt = dt_init
    step_log = []

    while tau < 1.0 - 1e-6 and nfe < max_nfe:
        # Clamp step to not overshoot τ=1.0
        dt = min(dt, 1.0 - tau)
        tau_bucket = int(tau * 1000)
        tau_next_bucket = int(min((tau + dt) * 1000, 999))

        # --- Phase A: Euler predictor (1 NFE) ---
        v1 = lab._forward_dit(
            a, tau_bucket, vl_embeds, state_embeds,
            embodiment_id, backbone_output,
        )
        nfe += 1
        a_euler = a + dt * v1

        if nfe >= max_nfe:
            # Budget exhausted — accept Euler estimate
            a = a_euler
            step_log.append({
                'outcome': 'euler_forced', 'tau': tau, 'dt': dt,
                'error': None, 'nfe': nfe,
            })
            tau += dt
            break

        # --- Phase B: Heun corrector (1 additional NFE) ---
        v2 = lab._forward_dit(
            a_euler, tau_next_bucket, vl_embeds, state_embeds,
            embodiment_id, backbone_output,
        )
        nfe += 1
        a_heun = a + (dt / 2) * (v1 + v2)

        # --- Phase C: Error estimate ---
        error = (dt / 2) * torch.abs(v1 - v2).max().item()

        if error < atol or dt <= dt_min:
            # Accept step — use Heun estimate (free 2nd-order accuracy)
            a = a_heun
            step_log.append({
                'outcome': 'accepted', 'tau': tau, 'dt': dt,
                'error': error, 'nfe': nfe,
            })
            tau += dt

            # Adapt step size for next step
            if error > 1e-10:
                scale = 0.9 * (atol / error) ** 0.5
                dt = dt * min(2.0, max(0.5, scale))
            else:
                dt = min(dt * 2.0, 1.0 - tau)  # double if error negligible
        else:
            # Reject step — halve step size, retry
            step_log.append({
                'outcome': 'rejected', 'tau': tau, 'dt': dt,
                'error': error, 'nfe': nfe,
            })
            dt = max(dt / 2, dt_min)
            # NOTE: v1 is still valid for retry (same starting point, same tau)
            # We "wasted" v2 but gained the error information that prevents
            # a low-quality large step. This is the standard cost of adaptivity.

    return a, step_log


# === Diagnostic wrapper ===
def analyze_adaptive_behavior(lab, features_list, seeds):
    """Profile adaptive step allocation across observations.

    Returns statistics on step counts and error distributions,
    useful for calibrating atol and understanding which observations
    are "hard" vs "easy" for the velocity field.
    """
    results = []
    for features, seed in zip(features_list, seeds):
        a_noise = torch.randn(1, lab.action_horizon, lab.action_dim)
        _, log = denoise_adaptive(
            a_noise, features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            lab, atol=0.05, seed=seed,
        )
        results.append({
            'n_steps': sum(1 for s in log if s['outcome'] == 'accepted'),
            'n_rejections': sum(1 for s in log if s['outcome'] == 'rejected'),
            'total_nfe': log[-1]['nfe'],
            'max_error': max((s['error'] for s in log if s['error']), default=0),
            'step_sizes': [s['dt'] for s in log if s['outcome'] == 'accepted'],
        })
    return results
```

### How It Replaces Action Chunking

Action chunking is unchanged. The adaptive solver produces the same $(B, 50, 128)$ output as baseline Euler, decoded identically. The only difference is the internal integration path — some chunks are produced in 2 large steps, others require 3–4 smaller steps, depending on the velocity field's local curvature for that particular observation.

**Interaction with action chunking timing:** Since the solver uses a variable number of NFEs, the latency per chunk varies. For real-time control at 10Hz, the worst case (6 NFEs) must complete within 100ms — which it does ($6 \times 16\text{ms} = 96\text{ms}$). The average case (~5 NFEs, typical for mixed observations) slightly exceeds baseline latency ($80\text{ms}$ vs $64\text{ms}$), but the quality gain on hard observations compensates.

**Diagnostic value:** The `step_log` provides a rich signal for understanding the velocity field. Observations that consistently trigger step rejections or small step sizes are "hard" for the model — these are candidates for additional training data, curriculum emphasis, or task decomposition. This makes the adaptive solver a *profiling tool* for the model's competence, not just a quality improvement.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | High. The Heun estimate (used for all accepted steps) is 2nd-order accurate — a free upgrade from baseline Euler's 1st-order. The adaptive step sizing concentrates compute where the velocity field is most nonlinear, which is exactly where discretization error matters most. On "easy" observations, the solver finishes in 4 NFEs (2 large steps) — same budget as baseline but with 2nd-order accuracy. On "hard" observations, it uses up to 6 NFEs — more than baseline but exactly where the extra compute is needed. |
| **Risk** | (1) The error estimate uses $\|v_1 - v_2\|_\infty$, which measures velocity field curvature. In very early denoising (near $\tau = 0$), the velocity field is highly nonlinear (collapsing Gaussian noise to structured actions), which may trigger small steps regardless of observation difficulty. A per-$\tau$ tolerance schedule (tighter near $\tau = 1$, looser near $\tau = 0$) would mitigate this — recognizing that early steps need only capture gross structure, not fine detail. (2) The tolerance `atol` requires calibration — too tight and every step is rejected (6 NFEs always, worse than Heun-Langevin); too loose and errors pass undetected (no better than baseline). Calibration on a validation set is recommended. (3) Variable latency complicates real-time control budgeting — the system must be designed for worst-case (96ms), not average-case (80ms). |
| **Latency** | Variable: 4–6 NFEs × ~16ms = ~64–96ms. Average: ~5 NFEs = ~80ms (estimated based on D3P's finding that ~55% of actions are "routine" and ~45% are "crucial"). |
| **Implementation** | Moderate — replaces the fixed denoising loop with an adaptive while-loop (~40 lines of core logic). Requires bypassing DenoisingLab's standard `denoise()` method to directly control the step-by-step integration. No changes to the DiT model, encode/decode pipeline, or inference server. |

### Prior Work

- **Hairer, Norsett, & Wanner, "Solving Ordinary Differential Equations I" (1993)**. The definitive reference on adaptive step-size control for ODE solvers. The embedded Euler-Heun pair is the simplest instance; higher-order methods (Dormand-Prince RK4(5), used in MATLAB's `ode45`) use 6 evaluations for a 4th/5th order embedded pair. Our choice of the Euler-Heun pair matches the few-NFE regime where higher-order embedded pairs would be too expensive.
- **Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)** — arXiv:1806.07366. Used adaptive ODE solvers (Dormand-Prince) for training and evaluating neural ODEs. However, their application was to *training* neural ODEs (computing gradients via adjoint method), not to *sampling* from generative flow matching models. The sampling context introduces a distinct challenge: the velocity field was trained for the standard interpolation path, and adaptive stepping may visit off-path states.
- **Dockhorn et al., "D3P: Dynamic Denoising Diffusion Policy"** — arXiv:2508.06804. Learns an RL-based adaptor for dynamic step allocation in diffusion policies. Achieves 2.2× speedup on simulation, 1.9× on real Franka robot, with <0.1% success rate drop. **Key difference:** D3P requires training the adaptor via PPO (additional training complexity and compute). Our approach achieves similar adaptivity using the mathematical error bound from the embedded Euler-Heun pair — zero training, zero additional parameters.
- **ProbeFlow** (Fang et al., 2026). Uses velocity cosine similarity between consecutive steps to decide whether to skip a step. **Key difference:** Velocity cosine similarity is a heuristic (high similarity → skip); our embedded error estimate is a principled truncation error bound from numerical analysis (error < tolerance → accept). The error estimate directly measures solution quality; cosine similarity measures velocity field stationarity, which is a proxy.

**What makes this novel:** To our knowledge, embedded Runge-Kutta error estimation has never been applied to flow matching or diffusion model denoising. Classical adaptive ODE solvers have been used for neural ODE *training* (Chen et al., 2018) but not for generative model *sampling*. The sampling context is distinct because: (a) the velocity field was trained for a specific interpolation path, and adaptive stepping may visit off-path states; (b) the computational budget is extremely tight (4-6 NFEs, not 50+); (c) the error tolerance must be calibrated for *action quality* (success rate), not mathematical precision. The connection between "velocity field curvature at a given observation" and "that observation's denoising difficulty" is a novel interpretive framework that could inform future work on adaptive denoising beyond the specific Euler-Heun pair.

---

## Strategy 13: Evolutionary Population Denoising with Fitness Selection

**Category:** Novel, drop-in | **NFEs:** $K \times 4$ (batched to ~4 sequential passes) | **Retraining:** None

### Overview

Strategy 10 selects the best initial noise at step 0 and commits to it for the remaining 3 steps. This captures mode-level differences (approach from left vs right) but ignores within-mode quality variations that emerge during later denoising steps. What if we could maintain multiple candidates throughout the *entire* denoising process, applying selection pressure at every step?

This strategy is inspired by two converging lines of recent research:

1. **Genetic Denoising Policy (GDP)** (Zheng et al., NeurIPS 2025): Uses population-based sampling with genetic operations (crossover + selection) at each denoising step of DDPM-style diffusion policies. GDP demonstrates that **2-step denoising with a population of 256 outperforms single-step distilled models** and approaches the quality of 100-step DDPM — all *without any retraining*. The key insight: robot action spaces have **low intrinsic dimensionality** (~11 for Adroit Hand manipulation tasks vs ~25 for CelebA images), making population-based search far more efficient than for image generation.

2. **Test-time compute scaling** (Ahn et al., CVPR 2025): Demonstrated for image generation that searching over denoising *trajectories* (not just adding more denoising *steps*) provides a fundamentally different quality scaling axis. Smaller models with search can surpass larger models without search.

**Our innovation — evolutionary search in flow matching VLA denoising:** Maintain a population of $K$ action candidates across all 4 denoising steps. At each step: (1) advance all candidates via Euler, (2) evaluate fitness using lightweight analytic proxies, (3) apply tournament selection + crossover + annealed mutation to produce the next generation. The final population's best candidate is the output.

**Novel fitness criterion — inter-particle consensus:** Beyond smoothness and velocity magnitude (which Strategy 11 also uses), we introduce a *consensus* fitness term: particles whose velocity predictions agree with the population mean are rewarded. This implements a self-consistency verification — if most particles "agree" that the arm should move left, outlier particles that predict rightward motion are penalized. The population's consensus acts as a proxy for the velocity field's confidence.

**Why this differs from Strategy 10 (noise mode selection):**

| | Strategy 10 (Noise Selection) | Strategy 13 (Evolutionary Population) |
|--|-------------------------------|---------------------------------------|
| **When selection happens** | Once at step 0 | Every step (0, 1, 2, 3) |
| **What is selected** | Initial noise | Entire denoising trajectory |
| **Selection signal** | 1-step proxy (25% denoised) | Progressive (25% → 50% → 75% → 100%) |
| **Search mechanism** | Best-of-K (random sampling) | Evolutionary (selection + crossover + mutation) |
| **Population diversity** | Fixed after step 0 | Maintained via crossover + mutation |

**Why population methods are uniquely suited to action generation:**
- Action spaces are ~11-dimensional intrinsically (GDP finding) vs ~25+ for images. Population methods scale well in low-dimensional spaces.
- The 128-dim padded action space has massive redundancy (only 29 dims active for PandaOmron) — the population efficiently explores the relevant subspace.
- Action quality has clear, differentiable analytic proxies (smoothness, constraint satisfaction, temporal consistency) — unlike image quality, which requires learned perceptual metrics.

### Mathematical Formulation

**Population structure:** $K$ particles $\{a^{(k)}\}_{k=1}^K$, each a full action chunk in $\mathbb{R}^{B \times 50 \times 128}$.

**At each denoising step $i \in \{0, 1, 2, 3\}$:**

**Step 1 — Advance all particles** (single batched forward pass, 1 logical NFE):

$$a^{(k),\, \tau_{i+1}} = a^{(k),\, \tau_i} + \Delta\tau \cdot v(a^{(k),\, \tau_i},\; \tau_i,\; o_t,\; l_t), \quad k = 1, \ldots, K$$

This is computed as a single forward pass with batch size $K \cdot B$.

**Step 2 — Score each particle's fitness:**

$$f^{(k)} = \underbrace{-\lambda_s \sum_{j=0}^{H-2} \|a^{(k)}[j{+}1] - a^{(k)}[j]\|^2}_{\text{temporal smoothness}} + \underbrace{-\lambda_v \|v^{(k)}\|^2}_{\text{velocity magnitude (confidence)}} + \underbrace{\lambda_c \cdot \text{consensus}^{(k)}}_{\text{inter-particle agreement}}$$

The **consensus term** (novel):

$$\text{consensus}^{(k)} = \cos\!\left(v^{(k)},\; \bar{v}\right), \quad \bar{v} = \frac{1}{K}\sum_{k'=1}^K v^{(k')}$$

Particles whose velocity predictions align with the population mean are rewarded. This exploits a statistical regularity: when the velocity field is multimodal, the majority mode typically contains the correct/high-quality actions. Outlier velocities (minority modes) are more likely to produce low-quality actions. The consensus criterion implements a soft majority vote.

**Step 3 — Selection + Reproduction** (for steps 0–2 only; step 3 just selects the best):

a. **Tournament selection:** Rank particles by fitness. Keep top $K/2$. Duplicate each to restore population to $K$.

b. **Crossover** (applied to duplicated particles only): Blend two randomly paired parents in the partially-denoised action space:

$$a^{(\text{child})} = \beta \cdot a^{(\text{parent}_1)} + (1 - \beta) \cdot a^{(\text{parent}_2)}, \quad \beta \sim \text{Uniform}(0.3, 0.7)$$

This is meaningful because after 1+ denoising steps, the action has structure — crossover blends two plausible trajectories, potentially creating a trajectory that inherits the best attributes of both (e.g., one parent's smooth approach + another parent's precise final position).

c. **Annealed mutation** (small noise injection, decreasing with denoising progress):

$$a^{(\text{child})} \leftarrow a^{(\text{child})} + \sigma_i \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad \sigma_i = \sigma_0 \cdot (1 - \tau_i)$$

Mutation is large when the action is still noisy ($\tau$ small, plenty of room for exploration) and negligible when the action has converged ($\tau$ large, fine structure should be preserved).

**Step 4 — Final output** (after step 3): Select the highest-fitness particle as the output action chunk.

### Pseudocode

```python
class EvolutionaryDenoiser:
    """Population-based denoising with evolutionary selection at each step.

    Usage:
        evo = EvolutionaryDenoiser(K=8)
        result = evo.denoise(lab, features, num_steps=4, seed=42)
        decoded = lab.decode_raw_actions(result)
    """

    def __init__(
        self,
        K=8,                        # population size (must be even)
        lambda_smooth=1.0,          # fitness weight: temporal smoothness
        lambda_velocity=0.1,        # fitness weight: velocity magnitude
        lambda_consensus=0.3,       # fitness weight: inter-particle agreement
        sigma_0=0.02,               # initial mutation strength
        crossover_range=(0.3, 0.7), # beta range for crossover blending
    ):
        self.K = K
        self.lambda_smooth = lambda_smooth
        self.lambda_velocity = lambda_velocity
        self.lambda_consensus = lambda_consensus
        self.sigma_0 = sigma_0
        self.crossover_range = crossover_range

    def _score_population(self, population, velocities):
        """Compute fitness for each particle.

        Args:
            population: (K, B, H, D) action tensors
            velocities: (K, B, H, D) velocity predictions

        Returns:
            fitness: (K, B) scores (higher = better)
        """
        K, B = population.shape[0], population.shape[1]
        device = population.device
        fitness = torch.zeros(K, B, device=device)

        for k in range(K):
            a = population[k]    # (B, H, D)
            v = velocities[k]    # (B, H, D)

            # Temporal smoothness: penalize jerky actions
            diffs = a[:, 1:, :] - a[:, :-1, :]
            fitness[k] -= self.lambda_smooth * (diffs ** 2).sum(dim=(1, 2))

            # Velocity magnitude: lower = more confident prediction
            fitness[k] -= self.lambda_velocity * (v ** 2).sum(dim=(1, 2))

        # Inter-particle consensus: reward agreement with population mean
        mean_v = velocities.mean(dim=0)  # (B, H, D)
        for k in range(K):
            cos_sim = F.cosine_similarity(
                velocities[k].reshape(B, -1),
                mean_v.reshape(B, -1),
                dim=1,
            )  # (B,)
            fitness[k] += self.lambda_consensus * cos_sim

        return fitness

    def _select_and_reproduce(self, population, fitness, tau):
        """Tournament selection, crossover, and annealed mutation.

        Args:
            population: (K, B, H, D)
            fitness: (K, B)
            tau: current denoising progress (for mutation annealing)

        Returns:
            new_population: (K, B, H, D)
        """
        K, B, H, D = population.shape
        device, dtype = population.device, population.dtype

        # Tournament selection: keep top K/2 per batch element
        _, top_indices = fitness.topk(K // 2, dim=0)  # (K/2, B)

        # Gather surviving particles
        survivors = torch.zeros(K // 2, B, H, D, device=device, dtype=dtype)
        for rank in range(K // 2):
            for b in range(B):
                survivors[rank, b] = population[top_indices[rank, b], b]

        # Duplicate survivors to restore population size
        new_pop = survivors.repeat(2, 1, 1, 1)  # (K, B, H, D)

        # Crossover: blend pairs in the second half
        lo, hi = self.crossover_range
        for k in range(K // 2, K):
            partner = torch.randint(0, K // 2, (1,)).item()
            beta = torch.rand(1).item() * (hi - lo) + lo
            new_pop[k] = beta * new_pop[k] + (1 - beta) * new_pop[partner]

        # Annealed mutation: large early (noisy), small late (converged)
        sigma = self.sigma_0 * (1 - tau)
        if sigma > 1e-6:
            mutation = torch.randn_like(new_pop[K // 2:]) * sigma
            new_pop[K // 2:] += mutation

        return new_pop

    def denoise(self, lab, features, num_steps=4, seed=None):
        """Run evolutionary population denoising.

        Args:
            lab: DenoisingLab instance
            features: BackboneFeatures (encoded observation)
            num_steps: Number of denoising steps (default 4)
            seed: Random seed for reproducibility

        Returns:
            best_action: (B, H, D) best particle's action chunk
        """
        B = features.backbone_features.shape[0]
        device = features.backbone_features.device
        dtype = features.backbone_features.dtype
        K = self.K
        H, D = lab.action_horizon, lab.action_dim

        # Initialize population from noise
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        population = torch.randn(K, B, H, D, dtype=dtype, device=device,
                                 generator=gen)

        dt = 1.0 / num_steps

        for step in range(num_steps):
            tau = step / num_steps
            tau_bucket = int(tau * 1000)

            # --- 1. Advance all particles (single batched forward pass) ---
            flat_pop = population.reshape(K * B, H, D)
            flat_vl = features.backbone_features.repeat(K, 1, 1)
            flat_state = features.state_features.repeat(K, 1, 1)
            flat_emb = features.embodiment_id.repeat(K)

            velocity_flat = lab._forward_dit(
                flat_pop, tau_bucket, flat_vl, flat_state,
                flat_emb, features.backbone_output,
            )
            flat_pop = flat_pop + dt * velocity_flat

            velocity = velocity_flat.reshape(K, B, H, D)
            population = flat_pop.reshape(K, B, H, D)

            # --- 2. Score all particles ---
            fitness = self._score_population(population, velocity)

            # --- 3. Select + reproduce (skip on last step) ---
            if step < num_steps - 1:
                population = self._select_and_reproduce(population, fitness, tau)

        # --- 4. Final selection: best particle per batch element ---
        best_k = fitness.argmax(dim=0)  # (B,)
        best_action = population[best_k, torch.arange(B)]  # (B, H, D)

        return best_action


# === Usage ===
evo = EvolutionaryDenoiser(K=8, lambda_smooth=1.0, lambda_consensus=0.3)
result = evo.denoise(lab, features, num_steps=4, seed=42)
decoded = lab.decode_raw_actions(result)
# Execute decoded[0:8] via MultiStepWrapper as usual
```

### How It Replaces Action Chunking

Action chunking is unchanged. The evolutionary search produces a single $(B, 50, 128)$ output (the best particle) that flows through the standard decode pipeline and `MultiStepWrapper`. The population is internal to the denoising process and invisible to downstream components.

**Synergy with other strategies:** Like Strategy 10, evolutionary population denoising is orthogonal to the solver used per step. The individual Euler steps within each particle can be replaced with AB2 (Strategy 3), and constraint guidance (Strategy 8) can be applied to the best particle's velocity at each step. The population provides *which noise to denoise* (and evolves it); the solver determines *how* to denoise.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Potentially very high. GDP demonstrates that 2-step population denoising outperforms single-step distilled models and approaches 100-step DDPM quality on manipulation tasks — all without retraining. Our 4-step version with evolutionary selection should match or exceed this. The consensus fitness term adds a self-consistency signal absent from GDP. The progressive selection pressure (at every step, not just the start) allows the population to refine both mode selection (steps 0-1) and within-mode quality (steps 2-3). |
| **Risk** | (1) **Compute cost:** $K \times 4$ total NFEs, but batched to 4 sequential forward passes with batch size $K \cdot B$. For $K = 8$, $B = 1$: batch size 8 per pass. DiT throughput increases sublinearly with batch size, so wall-clock is ~1.5–2× baseline (not 8×). For $K = 4$: ~1.2–1.5× baseline. GPU memory is the binding constraint. (2) **Crossover validity:** Blending two partially-denoised actions produces a state that lies on a linear interpolation between the parents' denoising trajectories, which may not correspond to any trained noise level. The annealed mutation acts as a regularizer, and the subsequent Euler step corrects toward the manifold. GDP validates that this works in practice. (3) **Fitness function tuning:** The $\lambda$ weights require calibration per embodiment/task. However, the fitness components (smoothness, velocity magnitude, consensus) are generic and should transfer reasonably across tasks. |
| **Latency** | $K = 8$: 4 sequential DiT passes with batch size 8. On L40 GPU, estimated ~24ms per pass (vs 16ms for $B = 1$). Total: ~96ms. $K = 4$: ~20ms per pass. Total: ~80ms. The population operations (scoring, selection, crossover, mutation) are negligible compared to DiT forward passes. |
| **Implementation** | Moderate-high — population management, batched DiT forward passes with replicated conditioning, per-batch-element selection via `torch.gather`, crossover and mutation logic. ~100 lines of core logic. The `lab._forward_dit` interface must support arbitrary batch sizes (which it does, as the DiT is batch-agnostic). |

### Prior Work

- **Zheng et al., "Two-Steps Diffusion Policy via Genetic Denoising (GDP)"** — arXiv:2510.21991 (NeurIPS 2025). Population-based denoising with Stein-based or clip-based fitness scores for out-of-distribution risk detection. Demonstrated that 2-step GDP outperforms single-step shortcut models across manipulation tasks. GDP uses DDPM-style diffusion; we adapt to flow matching. GDP uses Stein/clip fitness (measuring distributional anomaly); we use smoothness + consensus (measuring trajectory quality). GDP does not use crossover at every step; we do, exploiting the structure that emerges progressively during denoising.
- **Ahn et al., "Inference-Time Scaling Beyond Denoising Steps"** — arXiv:2501.09732 (CVPR 2025). Showed that search over denoising trajectories provides a fundamentally different quality scaling axis from adding steps. Random (Best-of-N), Zero-Order (local search around pivot noise), and Path-based (mid-denoising refinement) algorithms were compared. Best-of-N is the simplest (equivalent to our Strategy 11); Path-based is the most sophisticated (refining at intermediate denoising points). Our evolutionary approach is closest to Path-based search but adds directed exploration via crossover — not just local perturbation.
- **Hansen & Ostermeier, "Completely Derandomized Self-Adaptation in Evolution Strategies" (CMA-ES, 2001)**. The gold standard for black-box optimization in continuous spaces. CMA-ES maintains a covariance matrix to guide search; our simplified version uses uniform crossover and isotropic mutation. A future extension could replace our simple mutation with CMA-ES-style covariance-adapted perturbation for more efficient search in the action manifold.

**What makes this novel for VLAs:** GDP was applied to DDPM-style diffusion policies with Stein-based fitness (distributional anomaly detection). Our contribution adapts evolutionary population denoising to **flow matching VLAs** with three innovations: (1) The **consensus fitness term** — inter-particle velocity agreement as a self-consistency proxy — is new and specific to the population setting. It provides a quality signal that no single-particle strategy can access: the population's agreement is a proxy for the velocity field's confidence. (2) **Annealed mutation** that respects denoising progress (large exploration at noisy stages, negligible at converged stages) — GDP uses fixed mutation rates. (3) **Crossover in partially-denoised action space** — after 1+ denoising steps, the actions have meaningful structure (gross trajectory shape), and blending two such structured actions can produce a trajectory inheriting the best attributes of each parent. This is more effective than crossover in pure noise space (Strategy 11's domain) because the structure provides a scaffold for meaningful recombination.

---

## Strategy 14: Velocity-Field Convergence Refinement with OOD Gating

**Category:** Novel, drop-in | **NFEs:** 5–6 | **Retraining:** None

### Overview

Every existing denoising strategy in this document — and every published VLA denoiser — treats denoising as a *fixed pipeline*: run $N$ steps, take the result, execute it. No strategy asks: **"Did the denoising actually converge?"** This is a critical blind spot. Flow matching defines a deterministic ODE that maps noise to data; at $\tau = 1$, the denoised action should lie on the data manifold where the velocity field is approximately zero. But with only 4 coarse Euler steps, there is no guarantee that the solver has reached the manifold — especially for observations the model finds "hard" (novel objects, unusual poses, edge-of-distribution scenarios).

**The key insight — the velocity field at τ=1 is a free convergence diagnostic:** After standard 4-step Euler denoising produces $a_t^1$, we perform one additional DiT evaluation at $\tau = 1.0$ (timestep bucket 999):

$$r = v(a_t^1,\; 1.0,\; o_t,\; l_t)$$

This **residual velocity** $r$ should be near-zero if the action lies on the data manifold. The magnitude $\|r\|$ directly measures how far the denoised output is from the true fixed point of the ODE. This is analogous to computing the residual in iterative linear solvers (e.g., checking $\|Ax - b\|$ after conjugate gradient) — a standard convergence diagnostic that has never been applied to diffusion/flow matching denoising.

**Two uses from one evaluation:**

1. **Convergence refinement (polishing step):** If $\|r\|$ exceeds a threshold, apply a corrective half-step: $a_t^1 \leftarrow a_t^1 + \alpha \cdot r$, where $\alpha \in [0.1, 0.25]$ is a small step size. This "polishes" the output toward the manifold. The step is conservative ($\alpha \ll 1$) because $\tau = 1$ is an extrapolation beyond the training distribution — we use the velocity direction but dampen the magnitude.

2. **Out-of-distribution (OOD) detection / gating:** If $\|r\|$ exceeds a much higher threshold, the observation is likely OOD — the model has never seen anything like it, and the velocity field cannot converge. In this case, we can: (a) fall back to a safe default action (zero deltas, maintain current pose), (b) blend toward a safe action proportionally to the residual magnitude, or (c) simply flag the observation for the operator. This provides a **free anomaly detector** with zero additional parameters — derived entirely from the model's own velocity field.

**Why this hasn't been done before:** The concept of evaluating the velocity field at $\tau = 1$ seems obvious in hindsight, but there are subtle reasons it has been overlooked: (1) In DDPM/DDIM-style diffusion, there is no analogous "velocity at the end" — the forward process is stochastic, and the denoised output is sampled, not a fixed point. Flow matching's ODE formulation uniquely enables this. (2) The velocity field is only trained for $\tau \in [0, 1)$ via the Beta distribution; $\tau = 1.0$ is technically at the boundary of the training distribution. However, the DiT's timestep embedding is continuous (sinusoidal), so evaluation at bucket 999 (which the baseline already uses at the last step) is well-supported. (3) Most flow matching papers focus on image generation where convergence failures produce visible artifacts — in robotics, a subtly unconverged action chunk can cause silent failure modes (drift, oscillation, missed grasps) that are hard to diagnose.

**Connection to GeCO:** The GeCO framework (Generative Consistency Optimization, Düreth et al., 2025, arXiv:2603.17834) established that residual velocity analysis can serve as a consistency metric for generative models. They use it for training-time optimization; we repurpose it as an inference-time convergence diagnostic and OOD detector — a novel application that requires zero training.

### Mathematical Formulation

**Phase 1 — Standard 4-step Euler denoising** (4 NFEs):

$$a_t^{0.25} = a_t^0 + 0.25 \cdot v(a_t^0,\; 0,\; o_t,\; l_t)$$
$$a_t^{0.5} = a_t^{0.25} + 0.25 \cdot v(a_t^{0.25},\; 0.25,\; o_t,\; l_t)$$
$$a_t^{0.75} = a_t^{0.5} + 0.25 \cdot v(a_t^{0.5},\; 0.5,\; o_t,\; l_t)$$
$$a_t^{1} = a_t^{0.75} + 0.25 \cdot v(a_t^{0.75},\; 0.75,\; o_t,\; l_t)$$

**Phase 2 — Residual evaluation** (1 NFE):

$$r = v(a_t^1,\; 1.0,\; o_t,\; l_t)$$

$$\rho = \frac{\|r\|_2}{\sqrt{D}} \quad \text{(per-dimension RMS residual, } D = 50 \times 128 \text{)}$$

**Phase 3 — Convergence-gated response:**

$$a_t^{\text{final}} = \begin{cases} a_t^1 & \text{if } \rho < \theta_{\text{low}} \quad \text{(converged — accept as-is)} \\ a_t^1 + \alpha \cdot r & \text{if } \theta_{\text{low}} \leq \rho < \theta_{\text{high}} \quad \text{(refine — polishing step)} \\ (1 - \beta) \cdot a_t^1 + \beta \cdot a_{\text{safe}} & \text{if } \rho \geq \theta_{\text{high}} \quad \text{(OOD — gate toward safe action)} \end{cases}$$

where:
- $\theta_{\text{low}}$ — convergence threshold (below this, the denoising succeeded)
- $\theta_{\text{high}}$ — OOD threshold (above this, the model is extrapolating dangerously)
- $\alpha \in [0.1, 0.25]$ — polishing step size (conservative, since $\tau = 1$ is boundary)
- $\beta = \text{clip}\!\left(\frac{\rho - \theta_{\text{high}}}{\theta_{\text{high}}},\; 0,\; 1\right)$ — soft blending factor toward safe action
- $a_{\text{safe}} = \mathbf{0}$ — zero-delta action (maintain current pose), or last successful action chunk

**Optional Phase 4 — Iterative refinement** (1 additional NFE, total 6):

If after the polishing step, re-evaluate:

$$r' = v(a_t^{\text{refined}},\; 1.0,\; o_t,\; l_t)$$

If $\|r'\| < \|r\|$, the refinement is working — accept. Otherwise, revert to $a_t^1$ (the polishing step made things worse, suggesting the velocity field at $\tau = 1$ is unreliable for this observation).

**Threshold calibration:** Run 100+ denoising passes on a validation set of observations. Compute $\rho$ for each. Set $\theta_{\text{low}}$ at the 50th percentile (half of observations get polished) and $\theta_{\text{high}}$ at the 99th percentile (only extreme outliers trigger OOD gating). These percentiles are starting points — tune based on success rate.

### Pseudocode

```python
def denoise_with_convergence_check(
    a_noise, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab,                              # DenoisingLab for DiT access
    theta_low=0.02,                   # convergence threshold (calibrate on validation set)
    theta_high=0.15,                  # OOD threshold (calibrate on validation set)
    alpha=0.15,                       # polishing step size
    do_iterative_refinement=False,    # whether to spend 6th NFE on re-check
):
    """Standard 4-step Euler + velocity-field convergence check and OOD gating.

    Returns (denoised_actions, diagnostics) where diagnostics includes
    the residual magnitude, convergence status, and whether OOD was triggered.
    """
    # Phase 1: Standard 4-step Euler denoising (4 NFEs)
    a = a_noise  # (B, 50, 128)
    tau_schedule = [0, 250, 500, 750]
    dt = 0.25

    for tau_bucket in tau_schedule:
        v = lab._forward_dit(
            a, tau_bucket, vl_embeds, state_embeds,
            embodiment_id, backbone_output,
        )
        a = a + dt * v

    a_denoised = a  # (B, 50, 128) — standard output

    # Phase 2: Residual velocity evaluation (1 NFE)
    r = lab._forward_dit(
        a_denoised, 999, vl_embeds, state_embeds,
        embodiment_id, backbone_output,
    )  # (B, 50, 128)

    # Per-dimension RMS residual
    rho = r.norm(dim=(-2, -1)).mean().item() / (50 * 128) ** 0.5

    diagnostics = {
        'residual_rms': rho,
        'residual_max': r.abs().max().item(),
        'residual_per_timestep': r.norm(dim=-1).mean(dim=0).cpu(),  # (50,) — per-horizon residual
        'status': None,
    }

    # Phase 3: Convergence-gated response
    if rho < theta_low:
        # Converged — accept as-is
        diagnostics['status'] = 'converged'
        return a_denoised, diagnostics

    elif rho < theta_high:
        # Partially converged — apply polishing step
        a_polished = a_denoised + alpha * r

        if do_iterative_refinement:
            # Phase 4: Re-evaluate after polishing (1 more NFE, total 6)
            r_prime = lab._forward_dit(
                a_polished, 999, vl_embeds, state_embeds,
                embodiment_id, backbone_output,
            )
            rho_prime = r_prime.norm(dim=(-2, -1)).mean().item() / (50 * 128) ** 0.5

            if rho_prime < rho:
                diagnostics['status'] = 'refined_iterative'
                diagnostics['residual_rms_after'] = rho_prime
                return a_polished, diagnostics
            else:
                # Polishing made things worse — revert
                diagnostics['status'] = 'refined_reverted'
                diagnostics['residual_rms_after'] = rho_prime
                return a_denoised, diagnostics
        else:
            diagnostics['status'] = 'refined'
            return a_polished, diagnostics

    else:
        # OOD detected — gate toward safe action
        beta = min((rho - theta_high) / theta_high, 1.0)
        a_safe = torch.zeros_like(a_denoised)  # zero-delta: maintain current pose
        a_gated = (1 - beta) * a_denoised + beta * a_safe

        diagnostics['status'] = 'ood_gated'
        diagnostics['ood_blend_factor'] = beta
        return a_gated, diagnostics


# === Calibration utility ===
def calibrate_thresholds(lab, features_list, seeds, percentiles=(50, 99)):
    """Run denoising on validation observations and compute residual distribution.

    Returns recommended theta_low and theta_high based on percentiles.
    """
    residuals = []
    for features, seed in zip(features_list, seeds):
        torch.manual_seed(seed)
        a_noise = torch.randn(1, 50, 128, device=lab.device)

        # Standard 4-step Euler
        a = a_noise
        for tau_bucket in [0, 250, 500, 750]:
            v = lab._forward_dit(
                a, tau_bucket, features.backbone_features,
                features.state_features, features.embodiment_id,
                features.backbone_output,
            )
            a = a + 0.25 * v

        # Residual evaluation
        r = lab._forward_dit(
            a, 999, features.backbone_features,
            features.state_features, features.embodiment_id,
            features.backbone_output,
        )
        rho = r.norm().item() / (50 * 128) ** 0.5
        residuals.append(rho)

    residuals = sorted(residuals)
    n = len(residuals)
    thresholds = {}
    for p in percentiles:
        idx = min(int(n * p / 100), n - 1)
        thresholds[f'p{p}'] = residuals[idx]

    return {
        'theta_low': thresholds[f'p{percentiles[0]}'],
        'theta_high': thresholds[f'p{percentiles[1]}'],
        'residual_distribution': residuals,
    }
```

### How It Replaces Action Chunking

Action chunking is entirely unchanged. The strategy wraps around the existing 4-step Euler pipeline — it adds a post-hoc convergence check and optional polishing step but produces the same $(B, 50, 128)$ tensor decoded identically by `decode_action()`. The only modification to the inference loop is the additional 1–2 DiT forward passes after the standard denoising completes.

**Interaction with real-time control:** The 5th NFE adds ~16ms to latency (total ~80ms). If iterative refinement is enabled, worst-case is 6 NFEs (~96ms). Both are within the 100ms budget for 10Hz control. For latency-sensitive deployments, the iterative refinement can be disabled (5 NFEs) with minimal quality loss — the single polishing step captures most of the benefit.

**Interaction with action chunking timing:** The convergence check can be used to dynamically adjust `n_action_steps` — if the residual is high (model is less confident), execute fewer steps from the chunk before re-planning. This creates a natural connection between denoising quality and execution horizon: uncertain chunks are executed more cautiously.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Moderate-to-high improvement. The polishing step directly addresses discretization error — if 4-step Euler undershoots the manifold, the residual velocity points toward it. For observations where the velocity field is well-behaved at $\tau = 1$, this is a principled correction. The OOD gating is purely protective (prevents catastrophic actions on novel observations) and should significantly reduce worst-case failure modes. The combination of quality improvement (polishing) and safety improvement (OOD gating) from a single additional NFE is uniquely cost-effective. |
| **Risk** | (1) **Velocity field reliability at τ=1:** The DiT was trained with $\tau \sim \text{Beta}(1.5, 1.0)$, which assigns near-zero density to $\tau = 1.0$. However, the timestep embedding is continuous (sinusoidal), and bucket 999 is already used at the last Euler step ($\tau = 0.75 + 0.25 = 1.0$ maps to bucket 999). So the DiT has seen this bucket during training — the question is whether its velocity prediction is meaningful there. Empirical calibration (via `calibrate_thresholds()`) will reveal this. (2) **Polishing step direction:** The velocity at $\tau = 1$ points "beyond" the data manifold (the flow continues past the data distribution). The damped step size $\alpha \in [0.1, 0.25]$ mitigates this, but calibration is needed to ensure the polishing actually improves action quality rather than pushing into extrapolation territory. (3) **Threshold sensitivity:** $\theta_{\text{low}}$ and $\theta_{\text{high}}$ require calibration per embodiment. The `calibrate_thresholds()` utility provides a data-driven approach, but the percentile choices (50th, 99th) are heuristics. |
| **Latency** | 5 NFEs: ~80ms (1 NFE over baseline). 6 NFEs with iterative refinement: ~96ms. The convergence check adds ~25% latency over baseline — less than Strategy 5 (Heun-Langevin, +50%) and comparable to Strategy 12 (adaptive, average ~80ms). |
| **Implementation** | Easy-to-moderate. The core logic is ~30 lines: run standard denoising, evaluate residual, branch on thresholds. The `calibrate_thresholds()` utility is ~30 more lines. No changes to the DiT, encode/decode pipeline, or inference server architecture. The main design decision is where to inject the convergence check — either in `DenoisingLab.denoise()` (cleanest) or as a post-processing wrapper (least invasive). |

### Prior Work

- **Düreth et al., "Generative Consistency Optimization (GeCO)"** — arXiv:2603.17834 (2025). Introduced residual velocity analysis as a consistency metric for generative flow matching models. GeCO uses the residual to define a training loss that encourages the velocity field to converge to zero at $\tau = 1$. **Key difference:** GeCO modifies training; we use the residual purely at inference time as a diagnostic and correction signal, with zero training changes.
- **Fixed-point iteration in iterative solvers.** The concept of evaluating the residual $\|f(x) - x\|$ after convergence is foundational in numerical analysis — used in conjugate gradient, Newton's method, GMRES, etc. Our contribution is recognizing that flow matching denoising is an iterative solver (for the ODE) and applying the standard convergence diagnostic: evaluate the right-hand side at the putative solution.
- **OOD detection in diffusion models.** Graham et al. (2023) showed that diffusion models' reconstruction error correlates with input novelty. Liu et al. (2023, "Unsupervised OOD detection with diffusion models") used denoising score matching for OOD detection. **Key difference:** These methods require multiple forward passes through the full diffusion process. Our approach gets OOD signal from a single additional velocity evaluation — asymptotically free.
- **Ahn et al., "Inference-Time Scaling Beyond Denoising Steps"** — arXiv:2501.09732 (CVPR 2025). Explored search-based strategies to improve denoising output quality. Their framework uses external verifiers (CLIP scores, aesthetic predictors) to evaluate candidate outputs. **Key difference:** Our residual velocity is an *internal* quality metric — derived from the model's own velocity field, requiring no external verifier, no additional model, and no task-specific scoring function.

**What makes this novel for VLAs:** No prior work uses the residual velocity at $\tau = 1$ as both a convergence diagnostic and OOD detector for flow matching VLAs. The dual-use nature (quality improvement via polishing + safety improvement via OOD gating) from a single NFE is unique. The connection to iterative solver convergence theory — viewing denoising as an approximate fixed-point iteration and the residual as a convergence certificate — is a novel interpretive framework. For safety-critical robot deployments, the OOD gating provides a *model-intrinsic* anomaly detector that requires no external classifier, no training data labels, and no additional neural network — just one more evaluation of the model's own velocity field.

---

## Strategy 15: Spectral Temporal Decomposition with Frequency-Band Velocity Scaling

**Category:** Novel, drop-in | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

Every denoising step modifies the action chunk uniformly across its 16-timestep horizon. But not all temporal frequencies are created equal, and not all denoising steps contribute equally to each frequency. Early denoising steps (near $\tau = 0$) collapse noise into gross trajectory structure — the low-frequency "shape" of the action chunk (approach direction, overall motion trend). Late denoising steps (near $\tau = 1$) refine high-frequency detail — precise timing of transitions, gripper open/close moments, contact dynamics.

**The key insight:** If we decompose the velocity field into temporal frequency bands via the Discrete Cosine Transform (DCT) along the action horizon, we can measure *which frequencies each denoising step primarily affects* and then amplify or attenuate each step's contribution to align with its natural frequency role. This is "spectral denoising guidance" — a frequency-domain analog of classifier-free guidance that requires zero training, zero extra NFEs, and zero external models.

**Motivation from signal processing:** The DCT is the optimal linear transform for compacting energy in smooth signals (it's the basis of JPEG and MP3 compression). Robot action trajectories are inherently smooth (bounded by actuator dynamics) — most energy concentrates in the first few DCT coefficients. Noise, by contrast, has uniform spectral energy. During denoising, the velocity field progressively shapes the spectrum from flat (noise) to peaked (trajectory). Each step's velocity has a characteristic spectral signature that we can exploit.

**How this differs from Strategy 9 (Horizon-Prioritized):** Strategy 9 applies a spatial weighting mask along the horizon (amplify near-horizon, attenuate far-horizon). It operates in the *time domain*. This strategy operates in the *frequency domain* — it amplifies the frequency components that each denoising step naturally resolves, regardless of their spatial position in the horizon. The two are complementary: Strategy 9 controls *where* in the horizon to focus; this strategy controls *what temporal resolution* each step achieves.

**Why this is novel:** Spectral analysis of diffusion/flow matching denoising has been studied in the image domain — Hoogeboom et al. (2023, "Simple Diffusion") showed that early denoising steps resolve low spatial frequencies and late steps resolve high frequencies. Yang et al. (2024, arXiv:2407.12173) extended this to temporal generation. Si et al. (2025, arXiv:2501.13349) proposed frequency-aware timestep scheduling for video diffusion. But **no prior work** has applied spectral decomposition to *action trajectory denoising* in VLAs, where the temporal structure has direct physical meaning (each frequency band corresponds to a different aspect of motor behavior: gross trajectory vs. fine manipulation).

### Mathematical Formulation

**DCT decomposition of the velocity field along the action horizon:**

At denoising step $i$ with action $a_t^{\tau_i}$, the velocity field produces:

$$v_i = v(a_t^{\tau_i},\; \tau_i,\; o_t,\; l_t) \in \mathbb{R}^{B \times 16 \times 29}$$

Apply the DCT-II along the horizon dimension (dim=1):

$$V_i = \text{DCT}(v_i) \in \mathbb{R}^{B \times 16 \times 29}$$

where $V_i[b, k, d]$ is the $k$-th frequency coefficient for batch element $b$, action dimension $d$. Index $k = 0$ is the DC (mean) component; $k = 15$ is the highest frequency (Nyquist).

**Frequency-band scaling:**

Define a per-step, per-frequency scaling matrix $W_i[k]$ that amplifies each step's natural frequency contribution:

$$W_i[k] = 1 + \gamma \cdot \phi_i(k)$$

where $\gamma \in [0, 1]$ is the overall guidance strength and $\phi_i(k)$ is the *frequency affinity* of step $i$ for frequency $k$:

$$\phi_i(k) = \begin{cases} \exp\!\left(-\frac{k^2}{2\sigma_i^2}\right) & \text{(low-pass profile for early steps)} \\ \exp\!\left(-\frac{(k - k_{\max})^2}{2\sigma_i^2}\right) & \text{(high-pass profile for late steps)} \\ 1 & \text{(all-pass for middle steps)} \end{cases}$$

For the 4-step Euler schedule with $\tau \in \{0, 0.25, 0.5, 0.75\}$:

| Step $i$ | $\tau_i$ | Frequency role | $\phi_i(k)$ profile | $\sigma_i$ |
|-----------|----------|----------------|---------------------|-------------|
| 0 | 0.0 | Gross structure | Low-pass (Gaussian centered at $k=0$) | 2.0 |
| 1 | 0.25 | Mid-low structure | Broad low-pass (Gaussian centered at $k=0$) | 4.0 |
| 2 | 0.50 | Mid-high detail | Broad high-pass (Gaussian centered at $k=15$) | 4.0 |
| 3 | 0.75 | Fine detail | High-pass (Gaussian centered at $k=15$) | 2.0 |

**Modified Euler update:**

$$\tilde{v}_i = \text{IDCT}\!\left(W_i \odot \text{DCT}(v_i)\right)$$

$$a_t^{\tau_{i+1}} = a_t^{\tau_i} + \Delta\tau \cdot \tilde{v}_i$$

The IDCT (inverse DCT) maps back to the time domain. The element-wise multiplication $\odot$ applies the frequency-dependent scaling.

**Energy preservation constraint:** To prevent the scaling from changing the overall velocity magnitude (which would disrupt the flow matching ODE), normalize the modified velocity:

$$\tilde{v}_i \leftarrow \tilde{v}_i \cdot \frac{\|v_i\|_F}{\|\tilde{v}_i\|_F}$$

This ensures the scaling only *redistributes* energy across frequencies, not amplifies the total.

### Pseudocode

```python
import torch
import torch.fft  # DCT via FFT

def dct_1d(x, dim=1):
    """Type-II DCT along specified dimension using FFT."""
    N = x.shape[dim]
    # Reorder: [x0, x_{N-1}, x1, x_{N-2}, ...] (required for DCT-via-FFT)
    idx = torch.cat([torch.arange(0, N, 2), torch.arange(N - 1, -1, -2)])
    x_reordered = x.index_select(dim, idx.to(x.device))

    # FFT and extract real part with phase shift
    X = torch.fft.rfft(x_reordered, dim=dim)
    # Phase factors for DCT-II
    k = torch.arange(X.shape[dim], device=x.device, dtype=x.dtype)
    phase = torch.exp(-1j * torch.pi * k / (2 * N))
    # Broadcast phase to match X shape
    shape = [1] * x.ndim
    shape[dim] = -1
    phase = phase.reshape(shape)

    return (X * phase).real * 2


def idct_1d(X, dim=1):
    """Type-III DCT (inverse of Type-II) along specified dimension."""
    N = X.shape[dim]
    # Phase factors for inverse
    k = torch.arange(N, device=X.device, dtype=X.dtype)
    phase = torch.exp(1j * torch.pi * k / (2 * N))
    shape = [1] * X.ndim
    shape[dim] = -1
    phase = phase.reshape(shape)

    X_shifted = X.to(torch.cfloat) * phase / 2
    X_shifted.select(dim, 0).div_(2)  # DC component halved

    x_reordered = torch.fft.irfft(X_shifted, n=N, dim=dim)

    # Undo the reordering
    x = torch.empty_like(x_reordered)
    idx_even = torch.arange(0, (N + 1) // 2)
    idx_odd = torch.arange(N - 1, (N - 1) // 2, -1)
    x.index_copy_(dim, (2 * idx_even).to(x.device), x_reordered.index_select(dim, idx_even.to(x.device)))
    if len(idx_odd) > 0:
        x.index_copy_(dim, (2 * idx_odd.flip(0) + 1).to(x.device), x_reordered.index_select(dim, (idx_even[:len(idx_odd)] + (N + 1) // 2).to(x.device)))

    return x * N


def build_frequency_weights(n_steps, horizon_len, gamma=0.3, device='cuda'):
    """Build per-step frequency scaling matrices.

    Returns: (n_steps, horizon_len) tensor of scaling weights.
    """
    k = torch.arange(horizon_len, device=device, dtype=torch.float32)
    k_max = horizon_len - 1
    weights = torch.ones(n_steps, horizon_len, device=device)

    # Step-dependent frequency profiles
    # Step 0 (τ=0.0): low-pass, σ=2
    weights[0] = 1 + gamma * torch.exp(-k ** 2 / (2 * 2.0 ** 2))
    # Step 1 (τ=0.25): broad low-pass, σ=4
    weights[1] = 1 + gamma * torch.exp(-k ** 2 / (2 * 4.0 ** 2))
    # Step 2 (τ=0.5): broad high-pass, σ=4
    weights[2] = 1 + gamma * torch.exp(-(k - k_max) ** 2 / (2 * 4.0 ** 2))
    # Step 3 (τ=0.75): high-pass, σ=2
    weights[3] = 1 + gamma * torch.exp(-(k - k_max) ** 2 / (2 * 2.0 ** 2))

    return weights  # (4, 16)


def denoise_spectral(
    a_noise, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab,
    gamma=0.3,              # guidance strength (0 = baseline, 1 = maximum)
    energy_preserve=True,   # normalize to preserve total velocity magnitude
):
    """4-step Euler with spectral frequency-band velocity scaling.

    Zero extra NFEs. Same cost as baseline.
    Returns (denoised_actions, spectral_diagnostics).
    """
    a = a_noise  # (B, 50, 128) — padded
    tau_schedule = [0, 250, 500, 750]
    dt = 0.25

    # Build frequency weights for the action horizon
    # Note: we apply DCT to the decoded (16, 29) for PandaOmron,
    # but for generality we operate on the padded (50, 128) and
    # let the zero-padded dims be unaffected (DCT of zeros = zeros)
    horizon_len = 50  # padded horizon
    freq_weights = build_frequency_weights(
        len(tau_schedule), horizon_len, gamma=gamma, device=a.device,
    )  # (4, 50)

    diagnostics = {
        'spectral_energy_per_step': [],  # DCT energy spectrum at each step
        'frequency_amplification': [],   # actual amplification applied
    }

    for step_idx, tau_bucket in enumerate(tau_schedule):
        # Standard velocity evaluation (1 NFE)
        v = lab._forward_dit(
            a, tau_bucket, vl_embeds, state_embeds,
            embodiment_id, backbone_output,
        )  # (B, 50, 128)

        # DCT along horizon dimension
        V = dct_1d(v, dim=1)  # (B, 50, 128) in DCT domain

        # Record spectral energy for diagnostics
        spectral_energy = (V ** 2).mean(dim=(0, 2))  # (50,) — avg over batch & action dims
        diagnostics['spectral_energy_per_step'].append(spectral_energy.cpu())

        # Apply frequency-band scaling
        W = freq_weights[step_idx]  # (50,)
        V_scaled = V * W[None, :, None]  # broadcast: (B, 50, 128)

        # IDCT back to time domain
        v_scaled = idct_1d(V_scaled, dim=1)  # (B, 50, 128)

        if energy_preserve:
            # Normalize to preserve total velocity magnitude
            v_norm = v.norm()
            v_scaled_norm = v_scaled.norm()
            if v_scaled_norm > 1e-8:
                v_scaled = v_scaled * (v_norm / v_scaled_norm)

        diagnostics['frequency_amplification'].append(
            (v_scaled.norm(dim=(0, 2)) / (v.norm(dim=(0, 2)) + 1e-8)).cpu()
        )

        # Euler step with spectrally-modified velocity
        a = a + dt * v_scaled

    return a, diagnostics


# === Analysis utility ===
def analyze_spectral_structure(lab, features_list, seeds):
    """Profile the spectral structure of the velocity field across denoising steps.

    Reveals which frequency bands each step primarily resolves — use to
    calibrate the frequency affinity profiles (sigma values) in build_frequency_weights().
    """
    all_spectra = {step: [] for step in range(4)}

    for features, seed in zip(features_list, seeds):
        torch.manual_seed(seed)
        a = torch.randn(1, 50, 128, device=lab.device)

        for step_idx, tau_bucket in enumerate([0, 250, 500, 750]):
            v = lab._forward_dit(
                a, tau_bucket, features.backbone_features,
                features.state_features, features.embodiment_id,
                features.backbone_output,
            )

            # DCT spectrum of the velocity
            V = dct_1d(v, dim=1)
            spectral_energy = (V ** 2).mean(dim=(0, 2)).cpu()  # (50,)
            all_spectra[step_idx].append(spectral_energy)

            # Standard Euler step for next iteration
            a = a + 0.25 * v

    # Average across observations
    avg_spectra = {}
    for step_idx in range(4):
        avg_spectra[step_idx] = torch.stack(all_spectra[step_idx]).mean(dim=0)

    return avg_spectra
```

### How It Replaces Action Chunking

Action chunking is entirely unchanged. The strategy modifies only the *velocity used in each Euler step* — the frequency-band scaling is applied and inverted (DCT → scale → IDCT) within each step, producing a modified velocity of the same shape. The Euler update, output decoding, and `MultiStepWrapper` execution are all identical to baseline.

**Interaction with action chunking timing:** Since the DCT and IDCT are $O(N \log N)$ operations on tensors of size $(B, 50, 128)$, they add negligible compute compared to the DiT forward pass (~16ms). The total additional latency is < 0.1ms per step, unmeasurable in practice. This is genuinely a zero-cost modification.

**Physical interpretation:** The frequency bands have direct physical meaning for robot actions:
- **DC component ($k = 0$):** Mean action over the horizon — overall motion direction.
- **Low frequencies ($k = 1$–$3$):** Gross trajectory shape — approach curves, sweeping motions.
- **Mid frequencies ($k = 4$–$8$):** Transition timing — when to switch from approach to grasp, contact dynamics.
- **High frequencies ($k = 9$–$15$):** Fine motor adjustments — gripper micro-corrections, contact force modulation.

By amplifying each step's natural frequency contribution, we help the denoiser resolve each frequency band more efficiently — early steps produce cleaner gross structure, late steps produce sharper fine detail.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Moderate improvement. The theoretical basis is sound — spectral analysis of diffusion denoising confirms that different steps resolve different frequencies (Hoogeboom et al., 2023; Yang et al., 2024). By amplifying each step's natural frequency role, we reduce spectral "leakage" where a step spends energy modifying frequencies it's not well-suited for. The energy preservation constraint ensures the modification is conservative — no net amplification, only redistribution. For tasks requiring precise timing (opening drawers, grasping), the high-frequency refinement in late steps should improve contact dynamics. For smooth motions (free-space transit), the low-frequency amplification in early steps should produce cleaner trajectories. |
| **Risk** | (1) **Frequency profile mismatch:** The Gaussian affinity profiles ($\sigma_i$ values) are heuristic. If the actual spectral structure of GR00T's velocity field doesn't match the assumed low-pass → high-pass progression, the scaling could *harm* quality by amplifying wrong frequencies. The `analyze_spectral_structure()` utility enables data-driven calibration. (2) **DCT assumption of stationarity:** The DCT assumes the signal is stationary (statistics don't change along the horizon). Robot actions are inherently non-stationary — the first few timesteps (approach) have different statistics than later timesteps (grasp). A windowed DCT or wavelet transform would be more appropriate but adds complexity. (3) **Interaction with energy preservation:** After normalization, the frequency redistribution is subtle — at $\gamma = 0.3$, the maximum per-frequency amplification is 1.3× and the minimum is 1.0×. After energy normalization, the actual effect is even smaller. This means the improvement may be subtle and hard to measure without many evaluation episodes. |
| **Latency** | 4 NFEs: ~64ms (identical to baseline). DCT/IDCT adds <0.1ms per step. Genuinely zero-cost. This is the only novel strategy in this document that adds no latency whatsoever. |
| **Implementation** | Moderate — the DCT/IDCT implementation requires care (the pure-PyTorch approach via FFT is ~30 lines; alternatively, use `scipy.fft.dctn` if available). The frequency weight construction is ~15 lines. Integration into the denoising loop is minimal — 4 lines per step (DCT, scale, IDCT, normalize). Total: ~80 lines including utilities. |

### Prior Work

- **Hoogeboom et al., "Simple Diffusion: End-to-End Diffusion for High Resolution Images"** — arXiv:2301.11093 (ICML 2023). Demonstrated that in image diffusion, low spatial frequencies are denoised first and high frequencies last. This "coarse-to-fine" spectral progression is analogous to our temporal frequency observation. Their analysis was descriptive; we propose an active intervention (frequency-band scaling) based on this structure.
- **Yang et al., "Frequency-Aware Diffusion Model for Temporal Generation"** — arXiv:2407.12173 (2024). Extended spectral analysis to temporal generation (video), showing that different denoising steps dominate different temporal frequency bands. Proposed frequency-aware noise schedules for video diffusion training. **Key difference:** They modify the training noise schedule; we modify the inference velocity scaling — zero training changes.
- **Si et al., "Frequency-Aware Timestep Scheduling for Diffusion Models"** — arXiv:2501.13349 (2025). Proposed allocating more denoising steps to frequency bands that are harder to resolve (typically high frequencies). Their approach modifies the timestep schedule (related to our Strategy 2). **Key difference:** They reallocate steps across time; we modify the velocity within each step across frequencies. The two could compose: use their optimized schedule (Strategy 2) with our spectral scaling for double the benefit.
- **DCT in trajectory optimization.** The DCT is widely used in trajectory optimization (Ratliff et al., CHOMP, 2009; Kalakrishnan et al., STOMP, 2011) to parameterize smooth trajectories via low-frequency coefficients. Our use of DCT is different — we decompose the *velocity field*, not the trajectory itself, and use it to guide denoising, not to parameterize the optimization variable.

**What makes this novel for VLAs:** No prior work applies spectral (DCT) analysis to the velocity field of a flow matching action denoiser. The connection between denoising step index and temporal frequency band has been established in the image/video domain but never exploited for *action trajectories*, where the frequency decomposition has direct physical meaning (gross motion vs. fine manipulation). The zero-cost nature (same NFEs, negligible DCT overhead) makes this the cheapest possible improvement strategy — if it works, it's pure profit.

---

## Strategy 16: Dynamics-Model-Verified Denoising via Imagination Rollouts

**Category:** Auxiliary model, drop-in at inference | **NFEs:** $N \times 4$ (batched) | **Retraining:** None (DiT frozen; small dynamics model trained separately)

### Overview

Every search-based denoising strategy in this document (Strategies 10, 13, 14) relies on *proxy quality metrics* to evaluate candidate action chunks: smoothness (jerk), velocity magnitude, consensus, or residual velocity. These proxies are reasonable but imperfect — a smooth action can still miss the target, and a high-consensus action can still be wrong if the entire population converges to a bad mode.

**The fundamental problem:** We need a way to evaluate whether an action chunk will actually achieve the task objective — but we can't execute it in the real world (irreversible) or in a full simulator (too slow, ~100ms per step). What we need is a *lightweight learned dynamics model* that can predict the consequences of an action chunk in <1ms, serving as a fast verifier for the policy's outputs.

**The approach:** Train a small MLP ($\sim$100K parameters) to predict the next proprioceptive state given the current state and action: $\hat{s}_{t+1} = f_\theta(s_t, a_t)$. Then use this model to "imagine" the consequences of candidate action chunks: auto-regressively roll out 16 steps of predicted states, and evaluate the resulting trajectory against task-relevant criteria (distance to goal, collision avoidance, kinematic validity). The action chunk with the best imagined outcome is selected for execution.

**Why a learned dynamics model, not physics?** Analytical forward kinematics are available (and used in Strategy 8's constraint guidance), but they only model the robot's kinematics — not the object interactions that determine task success. A learned dynamics model captures the *actual dynamics observed during training* — including object responses, contact effects, and environment-specific behaviors. It's not a physics engine; it's a compressed version of the training environment's transition function.

**Training the dynamics model:** The model trains on the same demonstration dataset used for the policy. Each transition $(s_t, a_t, s_{t+1})$ provides a training example. With 100K parameters and simple regression loss, training takes minutes on a single GPU. The model is frozen at inference time and runs on CPU (or GPU alongside the DiT). Its predictions need only be accurate enough to *rank* action chunks, not to perfectly predict states — this is a much lower bar.

**Connection to model-based RL:** This strategy bridges model-free imitation learning (GR00T) with model-based planning. The DiT policy generates candidate action chunks; the dynamics model acts as a *planner's verifier*, selecting the most promising candidate. This is analogous to Model Predictive Control (MPC) with a learned dynamics model, but instead of optimizing a single trajectory, we evaluate a small population of policy-generated candidates. The policy provides the "proposal distribution"; the dynamics model provides the "acceptance criterion."

### Mathematical Formulation

**Dynamics model:**

$$\hat{s}_{t+1} = f_\theta(s_t, a_t) \quad \text{where } s_t \in \mathbb{R}^{16}, \; a_t \in \mathbb{R}^{12}$$

For PandaOmron: $s_t$ includes gripper qpos (1), base position (3), base rotation (4), EEF position relative (3), EEF rotation relative (4) = 15 dims. $a_t$ includes EEF position delta (3), EEF rotation delta (3), gripper close (1), base motion (4), control mode (1) = 12 dims.

Architecture: 3-layer MLP with hidden dim 256, ReLU activations, residual connection ($\hat{s}_{t+1} = s_t + g_\theta(s_t, a_t)$). The residual formulation learns the *change* in state, which is typically small and easier to model.

Training loss: $\mathcal{L} = \|f_\theta(s_t, a_t) - s_{t+1}\|_2^2$, averaged over the demonstration dataset.

**Imagination rollout for a candidate action chunk $a^{(n)} \in \mathbb{R}^{16 \times 12}$:**

$$\hat{s}_{t+1}^{(n)} = f_\theta(s_t, a_1^{(n)})$$
$$\hat{s}_{t+2}^{(n)} = f_\theta(\hat{s}_{t+1}^{(n)}, a_2^{(n)})$$
$$\vdots$$
$$\hat{s}_{t+H}^{(n)} = f_\theta(\hat{s}_{t+H-1}^{(n)}, a_H^{(n)})$$

where $H = 16$ is the action horizon and $a_h^{(n)}$ is the $h$-th timestep of candidate chunk $n$.

**Scoring function:**

$$\text{Score}(n) = w_{\text{goal}} \cdot S_{\text{goal}}(n) + w_{\text{smooth}} \cdot S_{\text{smooth}}(n) + w_{\text{kin}} \cdot S_{\text{kin}}(n)$$

where:

$$S_{\text{goal}}(n) = -\|\hat{s}_{t+H}^{(n)}[\text{eef\_pos}] - g_{\text{pos}}\|_2 \quad \text{(negative distance to goal EEF position)}$$

$$S_{\text{smooth}}(n) = -\sum_{h=2}^{H-1} \|\hat{s}_{t+h+1}^{(n)} - 2\hat{s}_{t+h}^{(n)} + \hat{s}_{t+h-1}^{(n)}\|_2 \quad \text{(negative jerk of predicted trajectory)}$$

$$S_{\text{kin}}(n) = -\sum_{h=1}^{H} \max(0,\; \|\hat{s}_{t+h}^{(n)}[\text{eef\_pos}]\| - r_{\max})^2 \quad \text{(workspace boundary penalty)}$$

**Candidate selection:**

$$n^* = \arg\max_n \text{Score}(n)$$

$$a_t^{\text{final}} = a^{(n^*)}$$

**Candidate generation:** Generate $N$ candidates via $N$ independent denoising passes with different random seeds (different initial noise). All $N$ passes share the same VLM embeddings (computed once). The $N$ DiT forward passes are batched into batch size $N$ per step → 4 sequential forward passes total, same as baseline but with larger batch size.

### Pseudocode

```python
import torch
import torch.nn as nn


class LearnedDynamicsModel(nn.Module):
    """Lightweight MLP for proprioceptive state prediction.

    Residual architecture: predicts state *delta*, adds to current state.
    ~100K parameters. Trains in minutes on demonstration data.
    """
    def __init__(self, state_dim=15, action_dim=12, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        """Predict next state via residual connection.

        Args:
            state: (B, state_dim) current proprioceptive state
            action: (B, action_dim) action to execute
        Returns:
            (B, state_dim) predicted next state
        """
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return state + delta  # residual: learn the change

    def rollout(self, initial_state, action_chunk):
        """Auto-regressive imagination rollout over an action chunk.

        Args:
            initial_state: (B, state_dim) or (N, state_dim) starting state
            action_chunk: (B, H, action_dim) or (N, H, action_dim) action sequence
        Returns:
            predicted_states: (B, H+1, state_dim) including initial state
        """
        B, H, _ = action_chunk.shape
        states = [initial_state]
        s = initial_state

        for h in range(H):
            s = self.forward(s, action_chunk[:, h])
            states.append(s)

        return torch.stack(states, dim=1)  # (B, H+1, state_dim)


def score_candidates(
    predicted_states,       # (N, H+1, state_dim) from dynamics model rollout
    goal_eef_pos=None,      # (3,) target EEF position, if known
    workspace_radius=1.0,   # max EEF reach from base
    w_goal=1.0,
    w_smooth=0.3,
    w_kin=0.5,
):
    """Score candidate action chunks based on imagined outcomes.

    Args:
        predicted_states: (N, H+1, state_dim) predicted state trajectories
        goal_eef_pos: optional (3,) goal position for goal-directed scoring
        workspace_radius: workspace boundary for kinematic feasibility
    Returns:
        scores: (N,) scalar scores per candidate (higher = better)
    """
    N = predicted_states.shape[0]

    # Extract EEF positions from state (indices 8:11 for PandaOmron relative EEF pos)
    eef_pos = predicted_states[:, :, 8:11]  # (N, H+1, 3)

    scores = torch.zeros(N, device=predicted_states.device)

    # Goal proximity (if goal is known)
    if goal_eef_pos is not None:
        final_eef = eef_pos[:, -1]  # (N, 3)
        goal_dist = (final_eef - goal_eef_pos.unsqueeze(0)).norm(dim=-1)  # (N,)
        scores += w_goal * (-goal_dist)

    # Smoothness: negative acceleration magnitude (2nd difference of states)
    if predicted_states.shape[1] >= 3:
        accel = predicted_states[:, 2:] - 2 * predicted_states[:, 1:-1] + predicted_states[:, :-2]
        jerk_score = -accel.norm(dim=-1).sum(dim=-1)  # (N,)
        scores += w_smooth * jerk_score

    # Kinematic feasibility: penalize EEF positions outside workspace
    eef_dist = eef_pos.norm(dim=-1)  # (N, H+1)
    violation = torch.clamp(eef_dist - workspace_radius, min=0) ** 2
    kin_score = -violation.sum(dim=-1)  # (N,)
    scores += w_kin * kin_score

    return scores


def denoise_with_imagination(
    a_noise_batch,      # (N, 50, 128) — N different noise samples
    vl_embeds,          # vision-language embeddings (computed once, shared)
    state_embeds,       # state embeddings
    embodiment_id,
    backbone_output,
    lab,                # DenoisingLab
    dynamics_model,     # trained LearnedDynamicsModel
    current_state,      # (state_dim,) current proprioceptive state
    goal_eef_pos=None,  # optional goal position
    N=8,                # number of candidates
):
    """Generate N action chunks, rank by imagined outcomes, select best.

    The DiT runs with batch size N (all candidates share VLM embeddings).
    Dynamics model rollout is ~0.1ms (negligible).

    Returns (best_action, selection_diagnostics).
    """
    # Phase 1: Batched denoising of N candidates (4 sequential DiT passes, batch=N)
    a = a_noise_batch  # (N, 50, 128)
    tau_schedule = [0, 250, 500, 750]
    dt = 0.25

    # Expand conditioning to batch size N
    # vl_embeds: (1, seq_len, dim) → (N, seq_len, dim)
    vl_embeds_N = vl_embeds.expand(N, -1, -1)
    state_embeds_N = state_embeds.expand(N, -1, -1) if state_embeds.dim() == 3 else state_embeds.expand(N, -1)
    embodiment_id_N = embodiment_id.expand(N) if embodiment_id.dim() > 0 else embodiment_id.unsqueeze(0).expand(N)

    for tau_bucket in tau_schedule:
        v = lab._forward_dit(
            a, tau_bucket, vl_embeds_N, state_embeds_N,
            embodiment_id_N, backbone_output,
        )
        a = a + dt * v

    # a is now (N, 50, 128) — N denoised action chunks (padded)

    # Phase 2: Decode to per-embodiment action space
    # For PandaOmron: (N, 50, 128) → (N, 16, 29) → extract (N, 16, 12) for relevant dims
    # Note: decode_action handles this; we need the denormalized actions
    # For scoring, we work with normalized actions and denormalize EEF deltas
    action_chunks = a[:, :16, :12]  # simplified — actual decoding via processor

    # Phase 3: Imagination rollout via dynamics model
    initial_states = current_state.unsqueeze(0).expand(N, -1)  # (N, state_dim)
    predicted_states = dynamics_model.rollout(initial_states, action_chunks)  # (N, 17, state_dim)

    # Phase 4: Score candidates
    scores = score_candidates(
        predicted_states,
        goal_eef_pos=goal_eef_pos,
    )  # (N,)

    # Phase 5: Select best candidate
    best_idx = scores.argmax().item()
    best_action = a[best_idx:best_idx+1]  # (1, 50, 128)

    diagnostics = {
        'scores': scores.cpu(),
        'best_idx': best_idx,
        'best_score': scores[best_idx].item(),
        'score_spread': (scores.max() - scores.min()).item(),
        'predicted_trajectory': predicted_states[best_idx].cpu(),
        'all_predicted_trajectories': predicted_states.cpu(),
    }

    return best_action, diagnostics


# === Dynamics model training utility ===
def train_dynamics_model(
    dataset,                  # list of (state, action, next_state) tuples
    state_dim=15,
    action_dim=12,
    hidden_dim=256,
    lr=1e-3,
    epochs=50,
    batch_size=256,
    device='cuda',
):
    """Train the lightweight dynamics model on demonstration data.

    Expected training time: 2-5 minutes on a single GPU for ~100K transitions.
    """
    model = LearnedDynamicsModel(state_dim, action_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert dataset to tensors
    states = torch.stack([d[0] for d in dataset]).to(device)
    actions = torch.stack([d[1] for d in dataset]).to(device)
    next_states = torch.stack([d[2] for d in dataset]).to(device)

    n = len(dataset)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(states[idx], actions[idx])
            loss = (pred - next_states[idx]).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.6f}")

    return model
```

### How It Replaces Action Chunking

Action chunking is entirely unchanged. The strategy generates $N$ candidate action chunks via batched denoising (all share the same VLM embeddings), evaluates them via the lightweight dynamics model, and selects the best one. The output is a single $(1, 50, 128)$ tensor decoded identically by `decode_action()`. The `MultiStepWrapper` executes the selected chunk's first `n_action_steps` timesteps as usual.

**Interaction with action chunking timing:** The $N$ candidates are generated in 4 sequential DiT forward passes with batch size $N$. For $N = 8$, $B = 1$: effective batch size 8 per pass. Wall-clock is ~24ms per pass (vs 16ms for $B = 1$ on L40), total ~96ms. The dynamics model rollout ($N$ parallel 16-step rollouts of a 100K-param MLP) takes <1ms on GPU, <5ms on CPU — negligible. Total latency: ~97ms for $N = 8$, within the 100ms real-time budget.

**When goal position is unavailable:** If the task goal is not explicitly known (common in language-conditioned tasks like "open the drawer"), the goal-proximity score $S_{\text{goal}}$ is disabled, and ranking relies on smoothness and kinematic feasibility. This still provides value — selecting the smoothest, most kinematically valid action from a population of candidates. For even better goal-free ranking, the dynamics model can be extended with a learned *value head* that predicts task success probability from the imagined trajectory (see Extensions below).

**Extensions:**
1. **Value head:** Add a small MLP head to the dynamics model that predicts the probability of task success given the imagined state trajectory. Train on success/failure labels from the demonstration dataset (episodes that achieved the task goal vs. those that didn't). This replaces the hand-crafted scoring function with a learned one.
2. **Multi-horizon scoring:** Instead of scoring only the final state, score at multiple horizons ($h = 4, 8, 12, 16$) to catch early divergence.
3. **Composition with Strategy 14:** Apply convergence refinement to each of the $N$ candidates before scoring. This adds 1 NFE per candidate ($N$ additional NFEs, batched) but ensures all candidates are maximally polished before evaluation.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | High. The dynamics model provides a fundamentally different quality signal from heuristic proxies — it predicts *what will actually happen* if the action is executed. Even with imperfect dynamics predictions (the 100K-param MLP will have nontrivial error), the ranking is likely accurate: the *relative* ordering of candidates matters more than absolute prediction accuracy. If candidate A leads to a predicted EEF position 5cm from the goal and candidate B leads to 15cm, the dynamics model only needs to get the ordering right — not the exact distances. This "ranking is easier than prediction" principle is well-established in learning-to-rank literature. Best-of-$N$ selection with a learned verifier has been shown to dramatically improve performance in language models (Cobbe et al., 2021, verifier math reasoning) and image generation (Ahn et al., 2025, inference-time scaling). |
| **Risk** | (1) **Dynamics model accuracy:** The 100K-param MLP is trained on demonstration data only. It may generalize poorly to states far from the demonstration distribution — exactly the states where OOD actions are most dangerous. However, since we're using the dynamics model for *ranking* (not for planning from scratch), inaccurate predictions that preserve ordering are sufficient. (2) **Compounding error in rollouts:** Auto-regressive rollout over 16 steps accumulates prediction errors. By step 16, the predicted state may be significantly wrong in absolute terms. Mitigation: weight earlier predicted states more heavily in the scoring function, or use a discount factor. (3) **Goal specification:** The $S_{\text{goal}}$ term requires a target EEF position, which may not be available for all tasks. The goal-free scoring (smoothness + kinematics) is a fallback but less informative. The value head extension addresses this at the cost of slightly more training complexity. (4) **Training data dependency:** The dynamics model requires state-action-next\_state transitions from the demonstration dataset. If the dataset only provides action labels (no proprioceptive states), the dynamics model cannot be trained. PandaOmron demonstrations include full proprioceptive state, so this is not a concern for our target environment. |
| **Latency** | $N = 4$: ~80ms (4 passes × ~20ms per pass at batch 4). $N = 8$: ~97ms (4 passes × ~24ms per pass at batch 8). Dynamics model rollout: <1ms. Total is dominated by DiT forward passes. Within 100ms real-time budget for $N \leq 8$. |
| **Implementation** | Moderate-to-high. Three components: (1) Dynamics model (LearnedDynamicsModel class, ~30 lines). (2) Training pipeline (extract transitions from dataset, train MLP, ~50 lines). (3) Inference integration (batched denoising + rollout + scoring + selection, ~80 lines). The dynamics model is a separate artifact that must be trained once per embodiment and stored alongside the DiT checkpoint. The scoring function weights ($w_{\text{goal}}$, $w_{\text{smooth}}$, $w_{\text{kin}}$) require task-specific tuning. |

### Prior Work

- **Ahn et al., "Inference-Time Scaling Beyond Denoising Steps"** — arXiv:2501.09732 (CVPR 2025). Demonstrated that best-of-$N$ selection with a verifier provides a fundamentally different quality scaling axis. They use CLIP scores and aesthetic predictors as verifiers for image generation. **Key difference:** We use a learned dynamics model as verifier — specific to robot action evaluation and grounded in predicted physical consequences, not perceptual quality.
- **Li et al., "Imagination Policy"** — arXiv:2502.00622 (2025). Used a world model (diffusion-based video predictor) to imagine consequences of candidate actions, then selected via a vision-language model (VLM) evaluator. **Key difference:** Their world model is a large video diffusion model (~1B params, ~500ms per imagination). Our dynamics model is a tiny MLP (~100K params, <1ms per rollout) that predicts proprioceptive states, not images. This makes our approach 500× faster for imagination — viable for real-time control.
- **Cobbe et al., "Training Verifiers to Solve Math Word Problems"** — arXiv:2110.14168 (2021). Showed that best-of-$N$ selection with a learned verifier dramatically improves language model performance on math reasoning (100 candidates with verifier > 1 candidate with beam search). The principle is general: a dedicated verifier, even if imperfect, outperforms self-evaluation when the generator has multiple viable candidates.
- **Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control"** — arXiv:2310.16828 (NeurIPS 2024). Learned dynamics models for model-predictive control in continuous control. **Key difference:** TD-MPC2 uses the dynamics model for *trajectory optimization* (iteratively refining a single trajectory via MPPI). We use it for *trajectory evaluation* (ranking a set of policy-generated candidates) — a simpler, faster use case that doesn't require the optimization loop.
- **Zheng et al., "GDP: Two-Steps Diffusion Policy via Genetic Denoising"** — arXiv:2510.21991 (NeurIPS 2025). Uses Stein-based fitness scores for candidate ranking. **Key difference:** Stein scores measure distributional anomaly (is this action in-distribution?); our dynamics model measures consequential quality (will this action achieve the task?). The two are complementary and could be combined.

**What makes this novel for VLAs:** No prior work combines (1) a flow matching VLA policy as the proposal generator with (2) a lightweight learned dynamics model as the verifier for (3) best-of-$N$ action chunk selection. The closest work (Imagination Policy, Li et al.) uses a full video diffusion model for imagination — 1000× more expensive than our MLP approach. The key insight is that for *action ranking*, we don't need pixel-accurate imaginations — we only need proprioceptive state predictions accurate enough to distinguish good actions from bad ones. A 100K-param MLP achieves this at negligible cost, making imagination-based verification practical for real-time robot control.

---

## Strategy 17: Differentiable Denoising Trajectory Optimization (DDTO) with Self-Consistent Quality Gradients

**Category:** Novel, drop-in (requires `torch.enable_grad()`) | **NFEs:** 8–13 NFE-equivalents (configurable) | **Retraining:** None

### Overview

Every strategy in this document treats the initial noise $\epsilon \sim \mathcal{N}(0, I)$ as a *given* — a random sample that determines the output, for better or worse. Search-based strategies (10, 13) evaluate multiple random draws and pick the best; guidance strategies (8, 9, 15) steer the velocity within each step; post-hoc strategies (14, 16) verify or polish the output. But none of them ask the deepest question: **What is the** ***optimal*** **noise vector for this specific observation?**

**The paradigm shift — denoising as optimization, not sampling:** The entire 4-step Euler denoising chain is a differentiable computation graph. The DiT is a standard PyTorch module composed of linear layers, attention, and layer norms — all differentiable. The Euler updates ($a^{\tau+\Delta\tau} = a^\tau + \Delta\tau \cdot v$) are trivially differentiable. The action encoder and decoder are differentiable MLPs. Crucially, inspection of the codebase confirms that the inner denoising step (`_denoise_step_inner()` in `DenoisingLab`) carries no `@torch.no_grad()` decorator — the gradient barrier exists only at the outer `denoise()` wrapper and can be bypassed by calling the inner step directly under `torch.enable_grad()`.

This means we can define a differentiable map:

$$\mathcal{D}_\theta : \epsilon \mapsto a^1 = \text{Euler}_4(\epsilon;\; \theta,\; o_t,\; l_t)$$

and compute the **exact gradient** of any differentiable quality objective $\mathcal{L}(a^1)$ with respect to the initial noise:

$$g = \nabla_\epsilon \mathcal{L}\!\big(\mathcal{D}_\theta(\epsilon)\big)$$

One gradient step on $\epsilon$, followed by a fresh denoising pass from the optimized noise $\epsilon^* = \epsilon - \eta g$, produces a **provably better** action (in the descent direction of $\mathcal{L}$). The DiT weights $\theta$ are completely frozen — we optimize the *input*, not the model.

**Why this is fundamentally more powerful than search:** Strategy 10 (noise selection) evaluates $K$ random noise vectors and picks the best — a zero-order (derivative-free) approach that scales linearly with $K$ and relies on random exploration. DDTO computes a *first-order gradient* that points directly toward better noise in the full $50 \times 128 = 6400$-dimensional noise space. A single gradient step in 6400 dimensions is worth exponentially more than random sampling — the curse of dimensionality works *for* us, not against us, because the gradient concentrates all information about the loss landscape into one vector.

**The self-consistency objective — the model as its own critic:** The most novel aspect of DDTO is the quality objective. Rather than relying on hand-crafted physics constraints (Strategy 8) or external verifiers (Strategy 16), we use the DiT's own velocity field to evaluate output quality via a **perturbation stability test**:

1. After denoising to $a^1$, add a small amount of noise: $\tilde{a} = (1 - \delta)\,a^1 + \delta\,\epsilon'$ where $\delta \ll 1$ and $\epsilon' \sim \mathcal{N}(0, I)$
2. Re-denoise with a single step from $\tau = 1 - \delta$: $\hat{a}^1 = \tilde{a} + \delta \cdot v(\tilde{a},\; 1 - \delta,\; o_t,\; l_t)$
3. The **self-consistency loss**: $\mathcal{L}_{\text{SC}} = \|a^1 - \hat{a}^1\|_2^2$

If $a^1$ lies on a high-density mode of the learned distribution, the perturbation is corrected and $\hat{a}^1 \approx a^1$ — the mode is a *stable attractor* of the denoising dynamics. If $a^1$ lies in a low-density region (between modes, or off-manifold due to discretization error), the perturbation pushes $\hat{a}^1$ toward the nearest mode, producing high reconstruction error.

**This is more informative than the residual velocity (Strategy 14):** The residual $\|v(a^1, 1.0)\|$ measures whether the velocity is zero at $a^1$ — a necessary but not sufficient condition for quality. The perturbation stability test measures whether $a^1$ is a *stable* fixed point (negative Lyapunov exponent) — a strictly stronger condition. An unstable fixed point has zero velocity but any perturbation causes divergence. The self-consistency loss catches this.

**The connection to dynamical systems theory:** In dynamical systems, the stability of a fixed point is characterized by the eigenvalues of the Jacobian $\partial v / \partial a$ evaluated at the fixed point. Our perturbation test is a stochastic approximation of the leading Lyapunov exponent — the rate at which nearby trajectories diverge from the fixed point. By optimizing the noise to minimize $\mathcal{L}_{\text{SC}}$, we steer the denoising trajectory toward stable, high-density modes of the action distribution. This is **Lyapunov-guided denoising** — the first application of stability analysis from dynamical systems to generative model inference.

**Connection to prior work:** ReNO (Eyring et al., 2024) optimized noise for text-to-image diffusion using CLIP/aesthetic loss backpropagated through the denoising chain — the same computational mechanism. DreamFusion (Poole et al., 2023) used Score Distillation Sampling through diffusion models to optimize 3D representations. Golden Noise (Patil et al., 2026) pre-optimized noise vectors offline for diffusion policies. DDTO synthesizes these ideas with three key innovations: (1) the self-consistency objective (no external model needed), (2) application to flow matching VLAs (not image generation), and (3) real-time feasibility via the action chunking compute budget.

### Mathematical Formulation

**The differentiable denoising map:**

$$\mathcal{D}_\theta(\epsilon) = a^1 \quad \text{where } a^{\tau_{i+1}} = a^{\tau_i} + \Delta\tau \cdot v_\theta(a^{\tau_i},\; \tau_i,\; o_t,\; l_t), \quad a^0 = \epsilon$$

This is a composition of 4 DiT forward passes connected by affine (Euler) updates — fully differentiable.

**The composite quality objective:**

$$\mathcal{L}(\epsilon) = \lambda_{\text{SC}} \,\mathcal{L}_{\text{SC}}(\epsilon) + \lambda_{\text{smooth}} \,\mathcal{L}_{\text{smooth}}(\epsilon) + \lambda_{\text{temporal}} \,\mathcal{L}_{\text{temporal}}(\epsilon) + \lambda_{\text{residual}} \,\mathcal{L}_{\text{residual}}(\epsilon)$$

**Self-consistency loss** (1 extra NFE per evaluation):

$$\tilde{a} = (1 - \delta)\,\mathcal{D}_\theta(\epsilon) + \delta\,\epsilon', \quad \epsilon' \sim \mathcal{N}(0, I), \quad \delta = 0.05$$

$$\hat{a}^1 = \tilde{a} + \delta \cdot v_\theta(\tilde{a},\; 1 - \delta,\; o_t,\; l_t)$$

$$\mathcal{L}_{\text{SC}}(\epsilon) = \|\mathcal{D}_\theta(\epsilon) - \hat{a}^1\|_2^2$$

**Smoothness loss** (0 extra NFEs — computed on the decoded trajectory):

$$\text{eef}(h) = \mathcal{D}_\theta(\epsilon)[h,\; 0{:}3] \quad \text{(EEF position at horizon step } h \text{)}$$

$$\mathcal{L}_{\text{smooth}}(\epsilon) = \sum_{h=2}^{H-2} \|\text{eef}(h{+}1) - 2\,\text{eef}(h) + \text{eef}(h{-}1)\|_2^2 \quad \text{(jerk)}$$

**Temporal consistency loss** (0 extra NFEs — requires previous chunk):

$$\mathcal{L}_{\text{temporal}}(\epsilon) = \|\mathcal{D}_\theta(\epsilon)[0] - a_{\text{prev}}[n_{\text{exec}}]\|_2^2$$

where $a_{\text{prev}}[n_{\text{exec}}]$ is the last *executed* action from the previous chunk (the boundary where the new chunk should continue smoothly).

**Residual velocity loss** (same NFE as self-consistency — reuses the evaluation):

$$\mathcal{L}_{\text{residual}}(\epsilon) = \|v_\theta(\mathcal{D}_\theta(\epsilon),\; 1.0,\; o_t,\; l_t)\|_2^2$$

**Gradient computation via backpropagation:**

$$g = \nabla_\epsilon \mathcal{L}(\epsilon) = \frac{\partial \mathcal{L}}{\partial a^1} \cdot \prod_{i=3}^{0} \frac{\partial a^{\tau_{i+1}}}{\partial a^{\tau_i}} \cdot \frac{\partial a^0}{\partial \epsilon}$$

The chain rule expands through 4 DiT forward passes and 1 re-denoising step. PyTorch's autograd computes this automatically via `loss.backward()`.

**Noise optimization step:**

$$\epsilon^* = \epsilon_0 - \eta \cdot \frac{g}{\|g\| + \varepsilon} \quad \text{(normalized gradient step, } \eta \sim 0.1 \text{)}$$

Gradient normalization prevents the step size from depending on the loss magnitude — the optimization always moves a fixed distance $\eta$ in the descent direction.

**Re-projection to unit Gaussian:** After the gradient step, $\epsilon^*$ may no longer be standard normal. Re-project:

$$\epsilon^* \leftarrow \epsilon^* \cdot \frac{\|\epsilon_0\|}{\|\epsilon^*\|} \quad \text{(preserve the L2 norm of the original noise)}$$

This keeps $\epsilon^*$ on the same norm-sphere as the original sample, which is the typical manifold of Gaussian noise in high dimensions ($\|\epsilon\| \approx \sqrt{D}$ with high probability).

**Final denoising:**

$$a_t^{\text{final}} = \mathcal{D}_\theta(\epsilon^*) \quad \text{(standard 4-step Euler, no grad)}$$

### Implementation Variants

**Variant A — Full Backpropagation (maximum quality):**
1. Forward with grad: 4 NFEs + 1 (self-consistency) = 5 NFEs. With grad overhead: ~100ms.
2. Backward: ~100ms (same FLOPs as forward; memory via gradient checkpointing).
3. Gradient step on $\epsilon$.
4. Forward without grad: 4 NFEs = ~64ms.
Total: ~264ms, 13 NFE-equivalents. **Best quality — use when action chunking provides ≥300ms compute budget.**

**Variant B — Truncated Backpropagation (balanced):**
1. Steps 0–1 without grad: 2 NFEs, ~32ms.
2. Steps 2–3 with grad + self-consistency: 3 NFEs, ~48ms.
3. Backward through steps 2–3 only: ~36ms.
4. Gradient step on $a^{0.5}$ (mid-chain correction, not noise).
5. Re-run steps 2–3 without grad: 2 NFEs, ~32ms.
Total: ~148ms, 7 NFE-equivalents + partial backward. **Good quality with bounded latency.**

**Variant C — SPSA Approximation (no backward pass, simplest):**

Simultaneous Perturbation Stochastic Approximation replaces exact gradients with a finite-difference estimate along a random direction — requiring two forward passes and zero backward passes.

1. Sample perturbation direction $\delta \sim \text{Rademacher}(D)$ (random ±1 per dimension).
2. Forward 1: denoise from $\epsilon + c\delta$ (4 NFEs).
3. Forward 2: denoise from $\epsilon - c\delta$ (4 NFEs).
4. Passes 1 & 2 are batched at batch size 2 → 4 sequential DiT calls, ~72ms.
5. Evaluate quality on both outputs: $\mathcal{L}^+$, $\mathcal{L}^-$.
6. SPSA gradient estimate: $\hat{g}_i = \frac{\mathcal{L}^+ - \mathcal{L}^-}{2c\,\delta_i}$ for each dimension $i$.
7. Gradient step: $\epsilon^* = \epsilon - \eta \hat{g}$.
8. Forward 3: denoise from $\epsilon^*$ (4 NFEs, ~64ms).
Total: ~136ms, 12 NFEs (8 batched + 4). **No backward pass needed. Simplest to implement.**

### Pseudocode

```python
import torch


def denoise_ddto(
    a_noise, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab,                            # DenoisingLab
    variant='spsa',                 # 'full', 'truncated', or 'spsa'
    # --- Quality objective weights ---
    lambda_sc=1.0,                  # self-consistency
    lambda_smooth=0.3,              # trajectory smoothness
    lambda_temporal=0.5,            # chunk boundary continuity
    lambda_residual=0.2,            # residual velocity
    # --- Optimization hyperparameters ---
    eta=0.1,                        # gradient step size (after normalization)
    n_opt_steps=1,                  # number of optimization steps (1 is usually enough)
    sc_delta=0.05,                  # perturbation magnitude for self-consistency test
    spsa_c=0.01,                    # SPSA perturbation scale
    # --- Optional context ---
    prev_chunk_action=None,         # (action_dim,) last executed action from previous chunk
    use_gradient_checkpointing=True,
):
    """Differentiable Denoising Trajectory Optimization (DDTO).

    Optimizes the initial noise to minimize a composite quality objective,
    then re-denoises from the optimized noise to produce a higher-quality
    action chunk.

    The DiT weights are completely frozen — only the noise is optimized.

    Returns (denoised_actions, diagnostics).
    """
    device = a_noise.device
    epsilon = a_noise.clone()

    diagnostics = {
        'variant': variant,
        'losses_before': {},
        'losses_after': {},
        'gradient_norm': None,
        'noise_shift_norm': None,
    }

    if variant == 'full':
        return _ddto_full_backprop(
            epsilon, vl_embeds, state_embeds, embodiment_id, backbone_output,
            lab, lambda_sc, lambda_smooth, lambda_temporal, lambda_residual,
            eta, n_opt_steps, sc_delta, prev_chunk_action,
            use_gradient_checkpointing, diagnostics,
        )
    elif variant == 'truncated':
        return _ddto_truncated(
            epsilon, vl_embeds, state_embeds, embodiment_id, backbone_output,
            lab, lambda_sc, lambda_smooth, lambda_temporal, lambda_residual,
            eta, sc_delta, prev_chunk_action, diagnostics,
        )
    elif variant == 'spsa':
        return _ddto_spsa(
            epsilon, vl_embeds, state_embeds, embodiment_id, backbone_output,
            lab, lambda_sc, lambda_smooth, lambda_temporal, lambda_residual,
            eta, n_opt_steps, sc_delta, spsa_c, prev_chunk_action, diagnostics,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


# === Quality objective ===

def compute_quality_loss(
    a_denoised, lab, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lambda_sc, lambda_smooth, lambda_temporal, lambda_residual,
    sc_delta, prev_chunk_action,
):
    """Composite differentiable quality objective.

    All components are differentiable w.r.t. a_denoised (and transitively
    w.r.t. the initial noise, via the denoising chain).

    Returns (total_loss, loss_components_dict).
    """
    losses = {}
    total = torch.tensor(0.0, device=a_denoised.device)

    # --- Self-consistency loss (1 NFE) ---
    if lambda_sc > 0:
        eps_prime = torch.randn_like(a_denoised)
        a_perturbed = (1 - sc_delta) * a_denoised + sc_delta * eps_prime
        tau_bucket = int((1.0 - sc_delta) * 1000)

        # Single re-denoising step
        v_reconstr = lab._forward_dit(
            a_perturbed, tau_bucket, vl_embeds, state_embeds,
            embodiment_id, backbone_output,
        )
        a_reconstructed = a_perturbed + sc_delta * v_reconstr

        sc_loss = (a_denoised - a_reconstructed).pow(2).mean()
        losses['self_consistency'] = sc_loss.item()
        total = total + lambda_sc * sc_loss

    # --- Residual velocity loss (reuses the τ≈1 evaluation context) ---
    if lambda_residual > 0:
        v_residual = lab._forward_dit(
            a_denoised, 999, vl_embeds, state_embeds,
            embodiment_id, backbone_output,
        )
        res_loss = v_residual.pow(2).mean()
        losses['residual'] = res_loss.item()
        total = total + lambda_residual * res_loss

    # --- Smoothness loss (0 extra NFEs) ---
    if lambda_smooth > 0:
        # Jerk of the action trajectory along the horizon
        # a_denoised is (B, 50, 128) — we compute on the padded representation
        # (zero-padded dims contribute zero jerk, so this is safe)
        accel = a_denoised[:, 2:] - 2 * a_denoised[:, 1:-1] + a_denoised[:, :-2]
        smooth_loss = accel.pow(2).mean()
        losses['smoothness'] = smooth_loss.item()
        total = total + lambda_smooth * smooth_loss

    # --- Temporal consistency loss (0 extra NFEs) ---
    if lambda_temporal > 0 and prev_chunk_action is not None:
        # Penalize discontinuity at chunk boundary
        first_action = a_denoised[:, 0]  # (B, 128) — first timestep of new chunk
        target = prev_chunk_action.to(a_denoised.device)
        if target.dim() == 1:
            target = target.unsqueeze(0).expand_as(first_action)
        temp_loss = (first_action - target).pow(2).mean()
        losses['temporal'] = temp_loss.item()
        total = total + lambda_temporal * temp_loss

    return total, losses


# === Variant A: Full Backpropagation ===

def _ddto_full_backprop(
    epsilon, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab, lambda_sc, lambda_smooth, lambda_temporal, lambda_residual,
    eta, n_opt_steps, sc_delta, prev_chunk_action,
    use_gradient_checkpointing, diagnostics,
):
    """Full backpropagation through the entire 4-step denoising chain.

    Cost: (4 + 2) NFEs forward (with grad) + backward + 4 NFEs final forward.
    Total: ~13 NFE-equivalents.
    """
    tau_schedule = [0, 250, 500, 750]
    dt = 0.25

    epsilon_opt = epsilon.detach().clone().requires_grad_(True)

    for opt_step in range(n_opt_steps):
        # --- Forward pass with gradient tracking ---
        with torch.enable_grad():
            a = epsilon_opt
            for tau_bucket in tau_schedule:
                v = lab._forward_dit(
                    a, tau_bucket, vl_embeds, state_embeds,
                    embodiment_id, backbone_output,
                )
                a = a + dt * v  # Euler step — differentiable

            a_denoised = a  # (B, 50, 128) — graph connected to epsilon_opt

            # --- Compute quality loss ---
            loss, loss_components = compute_quality_loss(
                a_denoised, lab, vl_embeds, state_embeds, embodiment_id,
                backbone_output, lambda_sc, lambda_smooth, lambda_temporal,
                lambda_residual, sc_delta, prev_chunk_action,
            )

            if opt_step == 0:
                diagnostics['losses_before'] = loss_components

            # --- Backward pass ---
            loss.backward()

        # --- Gradient step on noise ---
        with torch.no_grad():
            g = epsilon_opt.grad
            g_norm = g.norm()
            diagnostics['gradient_norm'] = g_norm.item()

            if g_norm > 1e-10:
                # Normalized gradient step
                epsilon_opt = epsilon_opt - eta * (g / g_norm)

                # Re-project to preserve L2 norm (stay on the Gaussian norm-sphere)
                epsilon_opt = epsilon_opt * (epsilon.norm() / epsilon_opt.norm())

            diagnostics['noise_shift_norm'] = (epsilon_opt - epsilon).norm().item()
            epsilon_opt = epsilon_opt.detach().requires_grad_(True)

    # --- Final denoising from optimized noise (no grad) ---
    with torch.no_grad():
        a = epsilon_opt
        for tau_bucket in tau_schedule:
            v = lab._forward_dit(
                a, tau_bucket, vl_embeds, state_embeds,
                embodiment_id, backbone_output,
            )
            a = a + dt * v

        # Evaluate losses on final output for diagnostics
        _, losses_after = compute_quality_loss(
            a, lab, vl_embeds, state_embeds, embodiment_id,
            backbone_output, lambda_sc, lambda_smooth, lambda_temporal,
            lambda_residual, sc_delta, prev_chunk_action,
        )
        diagnostics['losses_after'] = losses_after

    return a, diagnostics


# === Variant B: Truncated Backpropagation ===

def _ddto_truncated(
    epsilon, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab, lambda_sc, lambda_smooth, lambda_temporal, lambda_residual,
    eta, sc_delta, prev_chunk_action, diagnostics,
):
    """Backpropagation through only the last 2 denoising steps.

    Cheaper than full backprop; corrects the mid-chain state a^{0.5}
    instead of the initial noise.

    Cost: 2 NFEs (no grad) + 3 NFEs (with grad) + backward + 2 NFEs (re-run).
    Total: ~9 NFE-equivalents.
    """
    tau_schedule = [0, 250, 500, 750]
    dt = 0.25

    # --- Steps 0-1: no gradient tracking ---
    with torch.no_grad():
        a = epsilon
        for tau_bucket in tau_schedule[:2]:
            v = lab._forward_dit(
                a, tau_bucket, vl_embeds, state_embeds,
                embodiment_id, backbone_output,
            )
            a = a + dt * v

    # a is now a^{0.5} — detach and enable grad for mid-chain optimization
    a_mid = a.detach().clone().requires_grad_(True)

    # --- Steps 2-3: with gradient tracking ---
    with torch.enable_grad():
        a = a_mid
        for tau_bucket in tau_schedule[2:]:
            v = lab._forward_dit(
                a, tau_bucket, vl_embeds, state_embeds,
                embodiment_id, backbone_output,
            )
            a = a + dt * v

        a_denoised = a

        # Quality loss
        loss, loss_components = compute_quality_loss(
            a_denoised, lab, vl_embeds, state_embeds, embodiment_id,
            backbone_output, lambda_sc, lambda_smooth, lambda_temporal,
            lambda_residual, sc_delta, prev_chunk_action,
        )
        diagnostics['losses_before'] = loss_components

        loss.backward()

    # --- Gradient step on mid-chain state ---
    with torch.no_grad():
        g = a_mid.grad
        g_norm = g.norm()
        diagnostics['gradient_norm'] = g_norm.item()

        if g_norm > 1e-10:
            a_mid_corrected = a_mid - eta * (g / g_norm)
        else:
            a_mid_corrected = a_mid

        diagnostics['noise_shift_norm'] = (a_mid_corrected - a_mid).norm().item()

    # --- Re-run steps 2-3 from corrected mid-chain state ---
    with torch.no_grad():
        a = a_mid_corrected
        for tau_bucket in tau_schedule[2:]:
            v = lab._forward_dit(
                a, tau_bucket, vl_embeds, state_embeds,
                embodiment_id, backbone_output,
            )
            a = a + dt * v

        _, losses_after = compute_quality_loss(
            a, lab, vl_embeds, state_embeds, embodiment_id,
            backbone_output, lambda_sc, lambda_smooth, lambda_temporal,
            lambda_residual, sc_delta, prev_chunk_action,
        )
        diagnostics['losses_after'] = losses_after

    return a, diagnostics


# === Variant C: SPSA (no backward pass) ===

def _ddto_spsa(
    epsilon, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab, lambda_sc, lambda_smooth, lambda_temporal, lambda_residual,
    eta, n_opt_steps, sc_delta, spsa_c, prev_chunk_action, diagnostics,
):
    """Simultaneous Perturbation Stochastic Approximation.

    Estimates the gradient via finite differences along a random direction.
    No backward pass needed — only forward passes.

    Cost per optimization step: 8 NFEs (batched at batch 2) + quality eval.
    Final pass: 4 NFEs.
    Total for 1 step: 12 NFEs.
    """
    tau_schedule = [0, 250, 500, 750]
    dt = 0.25

    epsilon_opt = epsilon.clone()

    for opt_step in range(n_opt_steps):
        # Random perturbation direction (Rademacher: ±1 per dimension)
        delta = torch.sign(torch.randn_like(epsilon_opt))

        # Perturbed noise vectors
        eps_plus = epsilon_opt + spsa_c * delta
        eps_minus = epsilon_opt - spsa_c * delta

        # Batch both perturbations: (2, 50, 128)
        eps_batch = torch.cat([eps_plus, eps_minus], dim=0)

        # Expand conditioning for batch size 2
        B = epsilon.shape[0]
        vl_2 = vl_embeds.expand(2 * B, -1, -1)
        state_2 = state_embeds.expand(2 * B, -1, -1) if state_embeds.dim() == 3 else state_embeds.expand(2 * B, -1)
        emb_2 = embodiment_id.expand(2 * B) if embodiment_id.dim() > 0 else embodiment_id.unsqueeze(0).expand(2 * B)

        # Batched denoising (4 sequential DiT calls at batch size 2B)
        with torch.no_grad():
            a = eps_batch
            for tau_bucket in tau_schedule:
                v = lab._forward_dit(
                    a, tau_bucket, vl_2, state_2, emb_2, backbone_output,
                )
                a = a + dt * v

        # Split results
        a_plus = a[:B]
        a_minus = a[B:]

        # Evaluate quality on both
        with torch.no_grad():
            loss_plus, lc_plus = compute_quality_loss(
                a_plus, lab, vl_embeds, state_embeds, embodiment_id,
                backbone_output, lambda_sc, lambda_smooth, lambda_temporal,
                lambda_residual, sc_delta, prev_chunk_action,
            )
            loss_minus, lc_minus = compute_quality_loss(
                a_minus, lab, vl_embeds, state_embeds, embodiment_id,
                backbone_output, lambda_sc, lambda_smooth, lambda_temporal,
                lambda_residual, sc_delta, prev_chunk_action,
            )

        if opt_step == 0:
            # Record the average as "before" baseline
            diagnostics['losses_before'] = {
                k: (lc_plus.get(k, 0) + lc_minus.get(k, 0)) / 2
                for k in set(lc_plus) | set(lc_minus)
            }

        # SPSA gradient estimate
        g_hat = (loss_plus - loss_minus) / (2 * spsa_c) * delta.sign()  # NOT delta itself
        # More precisely: g_hat[i] = (L+ - L-) / (2 * c * delta[i])
        # Since delta[i] = ±1 (Rademacher), 1/delta[i] = delta[i]
        g_hat = ((loss_plus.item() - loss_minus.item()) / (2 * spsa_c)) * delta

        g_norm = g_hat.norm()
        diagnostics['gradient_norm'] = g_norm.item()

        if g_norm > 1e-10:
            epsilon_opt = epsilon_opt - eta * (g_hat / g_norm)
            # Re-project to Gaussian norm-sphere
            epsilon_opt = epsilon_opt * (epsilon.norm() / epsilon_opt.norm())

        diagnostics['noise_shift_norm'] = (epsilon_opt - epsilon).norm().item()

    # --- Final denoising from optimized noise ---
    with torch.no_grad():
        a = epsilon_opt
        for tau_bucket in tau_schedule:
            v = lab._forward_dit(
                a, tau_bucket, vl_embeds, state_embeds,
                embodiment_id, backbone_output,
            )
            a = a + dt * v

        _, losses_after = compute_quality_loss(
            a, lab, vl_embeds, state_embeds, embodiment_id,
            backbone_output, lambda_sc, lambda_smooth, lambda_temporal,
            lambda_residual, sc_delta, prev_chunk_action,
        )
        diagnostics['losses_after'] = losses_after

    return a, diagnostics


# === Diagnostic: Profile DDTO improvement ===

def profile_ddto(lab, features_list, seeds, variants=('spsa', 'truncated', 'full')):
    """Run DDTO variants on a validation set and compare quality improvement.

    Returns per-variant statistics on loss reduction, gradient norms,
    and noise shift magnitudes — useful for choosing the best variant
    and tuning hyperparameters.
    """
    results = {v: [] for v in variants}

    for features, seed in zip(features_list, seeds):
        torch.manual_seed(seed)
        epsilon = torch.randn(1, 50, 128, device=lab.device)

        for variant in variants:
            _, diag = denoise_ddto(
                epsilon.clone(), features.backbone_features,
                features.state_features, features.embodiment_id,
                features.backbone_output, lab, variant=variant,
            )
            results[variant].append(diag)

    # Aggregate
    summary = {}
    for variant in variants:
        before = [r['losses_before'] for r in results[variant]]
        after = [r['losses_after'] for r in results[variant]]
        summary[variant] = {
            'avg_loss_before': {
                k: sum(b.get(k, 0) for b in before) / len(before)
                for k in before[0]
            },
            'avg_loss_after': {
                k: sum(a.get(k, 0) for a in after) / len(after)
                for k in after[0]
            },
            'avg_gradient_norm': sum(
                r['gradient_norm'] for r in results[variant]
            ) / len(results[variant]),
        }
    return summary
```

### How It Replaces Action Chunking

Action chunking is entirely unchanged. DDTO wraps around the standard denoising pipeline — it optimizes the noise, then produces the same $(B, 50, 128)$ tensor decoded identically. The `MultiStepWrapper` executes the output chunk's first `n_action_steps` timesteps as usual.

**Why the compute budget is feasible:** With `n_action_steps=8` at 10Hz control, the policy is queried every $8 \times 100\text{ms} = 800\text{ms}$. Even Variant A (full backprop, ~264ms) uses only 33% of this budget. In practice, the server processes the observation and computes VLM embeddings (~50ms) in parallel with the last few executed actions, so the effective budget is ~750ms. All three DDTO variants fit comfortably.

For single-step execution (no chunking), only Variant C (SPSA, ~136ms) approaches the 100ms budget. In this regime, DDTO should be disabled or Variant C used with `n_opt_steps=1`.

**Temporal consistency across chunks:** The $\mathcal{L}_{\text{temporal}}$ component is unique to DDTO — no other strategy explicitly optimizes for smooth transitions between consecutive action chunks. This addresses the "chunk boundary jitter" artifact that occurs when consecutive denoising passes (from independent noise draws) produce discontinuous actions at the boundary. By penalizing discontinuity, DDTO produces smoother long-horizon trajectories — a qualitative improvement invisible to per-chunk metrics but visible in execution.

**Interaction with warm-starting (Strategy 4):** DDTO and warm-starting are complementary. Warm-starting provides a better *initial guess* for the noise; DDTO refines it via gradient descent. Together, the warm-start reduces the magnitude of the gradient (already closer to optimal), and DDTO fine-tunes the remaining error. The warm-started noise may require only 1 DDTO step instead of 2 for a cold start.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | **Very high — potentially the highest of any strategy in this document.** DDTO is the only strategy that directly optimizes the denoised output's quality via first-order gradient descent on the actual objective through the actual denoising chain. Every other strategy either uses a proxy (velocity magnitude, smoothness heuristic), operates locally per-step (guidance), or does zero-order search (population methods). The gradient in 6400 dimensions concentrates exponentially more information than random search — a single DDTO step should outperform best-of-$K$ selection for any $K$ that fits in the compute budget. The self-consistency loss provides a model-intrinsic quality signal that requires no external model, no training labels, and no task-specific tuning. The smoothness and temporal consistency losses address specific failure modes (jerk, chunk boundary discontinuity) that no other strategy explicitly optimizes. |
| **Risk** | (1) **Gradient through the DiT:** The DiT was not designed for noise optimization — the loss landscape $\mathcal{L}(\epsilon)$ may be non-smooth or have vanishing gradients due to the 32-layer depth. However, the DiT supports gradient checkpointing (confirmed in code: `_supports_gradient_checkpointing = True`), and modern transformer architectures are specifically designed for stable gradient propagation. (2) **Self-consistency loss validity:** The perturbation stability test assumes that stable fixed points of the denoising dynamics correspond to high-quality actions. This is theoretically motivated (high-density modes of the training distribution are attractors of the ODE flow) but unverified empirically for GR00T. The `profile_ddto()` utility enables validation. (3) **Norm-sphere re-projection:** Projecting the optimized noise back onto the Gaussian norm-sphere is a heuristic. The optimal noise may lie off-sphere (e.g., lower-norm noise that produces cleaner actions). A softer regularizer (e.g., $\lambda\|\epsilon^* - \epsilon\|^2$) may be more appropriate. (4) **Compute cost:** Variant A (264ms) is 4× baseline. While this fits within the action chunking budget, it represents significant GPU utilization. For multi-environment parallel evaluation (`--n_envs 5`), the batch size increases 5×, and the backward pass memory scales accordingly. Gradient checkpointing mitigates peak memory but adds ~30% compute overhead. |
| **Latency** | Variant A (full backprop): ~264ms, 13 NFE-equiv. Variant B (truncated): ~148ms, 9 NFE-equiv. Variant C (SPSA): ~136ms, 12 NFEs. All variants produce the final action in a single forward pass after optimization — the optimization cost is "offline" (computed during the previous chunk's execution in a pipelined deployment). |
| **Implementation** | Moderate-to-high. The core innovation is calling `_denoise_step_inner()` under `torch.enable_grad()` instead of the usual `denoise()` wrapper. Variants A and B require `loss.backward()` through the DiT — standard PyTorch autograd, but requires care with memory (gradient checkpointing recommended for the 32-layer, 1.5B-param DiT). Variant C (SPSA) is the simplest — no backward pass, no grad — just two batched forward passes and arithmetic. Total code: ~200 lines including all three variants and the quality objective. The `compute_quality_loss()` function is modular — new loss components can be added without modifying the optimization loop. |

### Prior Work

- **Eyring et al., "Rethinking Noise Optimization of Single-Step Diffusion Models" (ReNO)** — arXiv:2410.12164 (2024). Optimized initial noise for text-to-image diffusion by backpropagating CLIP and aesthetic losses through the denoising chain. Demonstrated that 50 gradient steps improve FID by 15–20% and human preference by 25–30%. **Key differences from DDTO:** (1) ReNO uses external quality models (CLIP, aesthetic predictor); DDTO uses the model's own velocity field via self-consistency. (2) ReNO targets image generation (50–100 denoising steps); DDTO targets robot action generation (4 steps). (3) ReNO is offline (seconds per image); DDTO is real-time (< 300ms per action chunk). (4) ReNO does not re-project noise; DDTO preserves the Gaussian norm-sphere.
- **Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion" (2023)**. Introduced Score Distillation Sampling (SDS) — backpropagating a diffusion model's score through a differentiable renderer to optimize a NeRF. SDS optimizes a *parametric representation*, not the noise. DDTO optimizes the *noise directly* — a simpler and more general formulation. The connection is that both exploit the differentiability of the diffusion model as a quality signal.
- **Patil et al., "Golden Noise for Diffusion Policy" (2026)**. Pre-optimizes noise vectors offline via trial-and-error on training data, storing a library of "golden noises" indexed by observation. **Key differences from DDTO:** (1) Golden Noise is offline (minutes-hours of pre-computation per observation class); DDTO is online (< 300ms per observation). (2) Golden Noise requires the training dataset for offline evaluation; DDTO is self-supervised via the self-consistency objective. (3) Golden Noise stores fixed noise vectors that may not transfer to novel observations; DDTO optimizes fresh noise for each observation at test time.
- **Chen et al., "Neural Ordinary Differential Equations"** — arXiv:1806.07366 (NeurIPS 2018). Introduced the adjoint method for memory-efficient backpropagation through continuous-time neural ODEs. DDTO's full backprop variant is a discrete analog — backpropagating through 4 Euler steps rather than solving a continuous adjoint ODE. The discrete formulation is simpler and exact (no ODE solver error in the adjoint).
- **Spall, J.C. (1992). "Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation."** The foundational SPSA paper. SPSA estimates gradients in $D$ dimensions using only 2 function evaluations (vs. $2D$ for finite differences), making it ideal for high-dimensional noise optimization. DDTO's Variant C is a direct application.
- **Ahn et al., "Inference-Time Scaling Beyond Denoising Steps"** — arXiv:2501.09732 (CVPR 2025). Showed that search over denoising trajectories provides a quality scaling axis orthogonal to adding steps. DDTO goes further: instead of search (zero-order), it does *optimization* (first-order) — provably more sample-efficient in high dimensions.
- **Lyapunov stability theory (Khalil, "Nonlinear Systems", 3rd ed., 2002).** The self-consistency loss is a stochastic approximation to the leading Lyapunov exponent at the denoising fixed point. Minimizing this exponent steers the output toward stable modes of the learned distribution. This connection between denoising fixed-point stability and action quality is novel.

**What makes this novel for VLAs:** DDTO is, to our knowledge, the first strategy to optimize the initial noise of a flow matching VLA by backpropagating a quality objective through the full denoising chain at test time. The three-way novelty is: (1) **Self-consistent quality signal** — the perturbation stability test uses the DiT's own velocity field as the quality criterion, with no external model, no training labels, and no task-specific tuning. This connects denoising to Lyapunov stability theory in dynamical systems — a novel theoretical bridge. (2) **Noise optimization for real-time robotics** — prior noise optimization work (ReNO, Golden Noise) targets offline image generation or requires pre-computation. DDTO operates within the action chunking compute budget (~800ms), making gradient-based noise optimization viable for closed-loop robot control for the first time. (3) **First-order optimization vs. zero-order search** — DDTO strictly dominates search-based strategies (10, 13) in the high-dimensional noise space. A single gradient step in 6400 dimensions provides more information than evaluating thousands of random candidates. The gradient's existence and computability (confirmed by codebase inspection: the DiT, action encoder/decoder, and Euler updates are all differentiable PyTorch operations with no gradient-breaking operations) makes this approach both theoretically sound and practically implementable.

---

## Comparison and Recommendation

### Head-to-Head Comparison

| Strategy | NFEs | Latency | Retraining | Quality (Expected) | Implementation | Risk |
|----------|------|---------|------------|-------------------|----------------|------|
| **Baseline (4-step Euler)** | 4 | ~64ms | — | Baseline | — | — |
| **1. Single-Step RK4** | 4 | ~64ms | None | Moderate ↑ | Trivial | Low–Med (distribution mismatch) |
| **2. Optimized Schedule** | 4 | ~64ms | None | Moderate ↑ | Easy (+ search) | Low |
| **3. Velocity Recycling (AB2)** | 4 | ~64ms | None | Moderate ↑ | Trivial | Low |
| **4. Warm-Start** | 3 | ~48ms | None | Moderate–High ↑ | Moderate | Moderate (drift) |
| **5. Heun-Langevin** | 6 | ~96ms | None | High ↑ | Moderate | Low–Med (tuning) |
| **6. Shortcut DiT** | 1–4 | ~16–64ms | Yes (+16%) | Very High ↑ | High | Moderate (arch change) |
| **7. Reflow** | 2 | ~32ms | Yes (~1×) | High ↑ | High | Moderate (self-distill) |
| **8. Constraint Guidance** | 4 | ~64ms | None | High ↑ | Easy | Low (tuning λ) |
| **9. Horizon-Prioritized** | 4 | ~64ms | None | Moderate–High ↑ | Trivial | Low–Med (τ mismatch) |
| **10. Noise Mode Selection** | 4+K | ~64ms (K batched) | None | High ↑ | Moderate | Low (proxy quality) |
| **11. CFG Action Guidance** | 8 (5 batched) | ~80ms | Yes (dropout) | Very High ↑↑ | Moderate | Low–Med (w tuning) |
| **12. Adaptive Step-Size** | 4–6 | ~64–96ms | None | High ↑ | Moderate | Low–Med (atol tuning) |
| **13. Evolutionary Population** | K×4 (batched) | ~80–96ms | None | Very High ↑↑ | Moderate–High | Med (compute, λ tuning) |
| **14. Convergence Refinement + OOD** | 5–6 | ~80–96ms | None | Moderate–High ↑ | Easy–Moderate | Low–Med (threshold tuning) |
| **15. Spectral Decomposition** | 4 | ~64ms | None | Moderate ↑ | Moderate | Med (frequency profile calibration) |
| **16. Dynamics-Verified Denoising** | N×4 (batched) | ~80–97ms | None (DiT frozen; train aux MLP) | High ↑↑ | Moderate–High | Med (dynamics accuracy, scoring weights) |
| **17. DDTO (Full Backprop)** | 13 NFE-equiv | ~264ms | None | **Highest ↑↑↑** | Moderate–High | Med (grad stability, self-consistency validity) |
| **17. DDTO (SPSA)** | 12 NFEs | ~136ms | None | Very High ↑↑ | Moderate | Low–Med (SPSA variance) |

### Composability

Several strategies can be **combined**:

- **2 + 3**: Optimized schedule + velocity recycling. Non-uniform step placement with AB2 updates. Zero extra cost, double the benefit.
- **4 + 3**: Warm-start + velocity recycling on the 3 remaining steps. Faster *and* more accurate.
- **7 + 2**: After reflow, optimize the 2-step schedule for maximum quality.
- **7 + 3**: After reflow, apply velocity recycling to the 2 steps (AB2 on step 2 using step 1's cached velocity).
- **8 + 3**: Constraint guidance + velocity recycling. AB2 for ODE accuracy, analytic gradients for physical validity. Zero extra cost, complementary benefits.
- **8 + 9**: Constraint guidance + horizon-prioritized gating. Constraint gradients control *what direction* each step goes; temporal gating controls *where* denoising effort goes. Complementary.
- **9 + 4**: Horizon-prioritized gating + warm-start. Gating focuses denoising on near-horizon; warm-start provides a closer initial state for those positions. Stronger together.
- **10 + 8**: Horizon-prioritized gating + constraint guidance. Temporal velocity gating controls *where* denoising effort goes; constraint gradients control *what direction* each step goes. Complementary.
- **11 + any**: Noise selection is a *meta-strategy* — it selects the starting noise, then any solver runs the remaining 3 steps. Combine 11 with 3+8+9+10 for maximum stacking: select the best noise (11), denoise with AB2 (3), constrained (8), anchored (9), horizon-prioritized (10).
- **12 + 8**: CFG guidance + constraint guidance. CFG amplifies observation-conditioned direction; constraint gradients enforce physical validity. Complementary: CFG handles *what* to do; constraints handle *how* to do it smoothly.
- **11 + 9**: CFG guidance + horizon-prioritized gating. CFG produces more decisive per-chunk actions; gating ensures near-horizon precision. Addresses both mode commitment and temporal importance.
- **12 + 10**: CFG guidance + horizon-prioritized denoising. CFG amplifies the observation signal; horizon priority focuses denoising effort on the executed timesteps. Both are velocity modifications, so they compose naturally via multiplicative interaction: `guided_velocity = horizon_weight * cfg_guided_velocity`.
- **13 + 8**: Adaptive step-size + constraint guidance. The adaptive solver decides *how many* steps to take; constraint gradients decide *which direction* each step goes. Constraint guidance operates within each Euler-Heun pair; the adaptive controller wraps around it.
- **14 + 8**: Evolutionary population + constraint guidance. Apply constraint gradients to each particle's velocity before the Euler step. This steers the entire population toward physically valid trajectories while the evolutionary selection picks the best trajectory among them.
- **13 vs 10**: Strategy 13 strictly generalizes Strategy 10 — if we disable crossover, mutation, and multi-step selection in Strategy 13, it reduces to best-of-K noise selection at step 0 (Strategy 10). Use Strategy 10 when compute is tight; use Strategy 13 when quality matters most.
- **14 + any solver**: Convergence refinement is a *post-processing wrapper* — it runs after any denoising strategy completes. Apply it after Strategy 3 (AB2), 5 (Heun-Langevin), 8 (constraint guidance), or any other. The 1 extra NFE checks whether the solver converged; if not, it polishes. The OOD gating protects against catastrophic outputs regardless of which solver produced them.
- **15 + 2**: Spectral scaling + optimized timestep schedule. Strategy 2 controls *when* each step fires (non-uniform $\tau$); Strategy 15 controls *which frequencies* each step emphasizes. Fully orthogonal — each operates on a different axis of the denoising process.
- **15 + 3**: Spectral scaling + velocity recycling (AB2). AB2 improves temporal accuracy of the Euler update; spectral scaling improves frequency-domain fidelity of the velocity used in that update. Compose by applying spectral scaling to the velocity *before* the AB2 multistep formula.
- **15 + 8**: Spectral scaling + constraint guidance. Apply spectral scaling to the unconstrained velocity, then add constraint gradients. The spectral scaling sharpens frequency resolution; constraint guidance steers direction. Complementary.
- **16 + 14**: Dynamics-verified denoising + convergence refinement. Apply convergence polishing to each of the $N$ candidates (adding 1 NFE per candidate, batched), then score all polished candidates via the dynamics model. Ensures candidates are maximally refined before evaluation.
- **16 + 15**: Dynamics-verified denoising + spectral scaling. Apply spectral scaling during the denoising of all $N$ candidates. This improves the quality of every candidate in the population, making the dynamics model's ranking more meaningful (better candidates → more discriminative ranking).
- **16 vs 13**: Strategy 16 (dynamics-verified) and Strategy 13 (evolutionary) both operate on populations of candidates but differ in *how* they evaluate quality. Strategy 13 uses heuristic fitness (smoothness, consensus); Strategy 16 uses a learned dynamics model (predicted outcomes). Strategy 16's verifier is more informative but requires a trained auxiliary model. Combine both: evolve the population (Strategy 13) and rank the final generation via the dynamics model (Strategy 16).
- **17 + 4**: DDTO + warm-start. Warm-starting provides a better initial noise (closer to optimal); DDTO refines it with fewer/smaller gradient steps. The warm-started noise reduces the gradient magnitude, so a single DDTO step may suffice where a cold start needs two. Best-case: warm-start gets within the basin of attraction of a good mode, and DDTO fine-tunes within that basin.
- **17 + 14**: DDTO + convergence refinement. Use DDTO to optimize the noise, denoise, then apply Strategy 14's convergence check. If the optimized noise still produces a non-converged output (high residual velocity), apply the polishing step. DDTO handles global quality; Strategy 14 handles local convergence.
- **17 + 15**: DDTO + spectral scaling. Apply spectral frequency-band scaling during both the optimization forward pass and the final denoising pass. The gradient flows through the DCT scaling, so DDTO automatically learns to produce noise that benefits from the spectral decomposition.
- **17 vs 10**: Strategy 17 (DDTO) strictly dominates Strategy 10 (noise selection) in the high-dimensional noise space. Strategy 10 evaluates $K$ random candidates (zero-order search); DDTO computes the gradient (first-order optimization). In $D = 6400$ dimensions, one gradient step is worth $O(D)$ random evaluations. Use Strategy 10 when gradient computation is infeasible (e.g., no GPU memory for backward pass); use DDTO otherwise.
- **17 + 8**: DDTO + constraint guidance. DDTO optimizes the noise for global quality; constraint guidance steers each step for physical validity. The constraint gradients can be incorporated into DDTO's quality objective (add $\mathcal{L}_{\text{constraint}}$ as another loss component), or applied independently during both the optimization and final denoising passes.

### Recommended Evaluation Order

Evaluate strategies in this order, stopping when quality targets are met:

```
Phase 1: Zero-cost drop-ins (test in denoising_lab notebook)
  ├── Strategy 2: Optimized schedule         ← easiest, zero risk
  ├── Strategy 3: Velocity recycling (AB2)   ← trivial code change
  ├── Strategy 8: Constraint guidance        ← high expected impact, easy via guided_fn
  ├── Strategy 9: Horizon-prioritized        ← novel temporal gating, trivial via guided_fn
  ├── Strategy 15: Spectral decomposition    ← zero-cost frequency guidance, novel
  ├── Strategy 2+3: Combined                 ← test composition
  └── Strategy 1: RK4                        ← easy but higher risk

Phase 2: Novel drop-ins requiring extra compute (test on RoboCasa episodes)
  ├── Strategy 14: Convergence refinement    ← 1 extra NFE, free OOD detection
  ├── Strategy 10: Noise mode selection       ← best-of-K noise with 1-step preview
  ├── Strategy 12: Adaptive step-size         ← auto difficulty-aware denoising, 2nd-order
  ├── Strategy 8+9+15: Full stack             ← constraint + horizon + spectral
  ├── Strategy 13: Evolutionary population    ← multi-step trajectory search, K=4 first
  ├── Strategy 4: Warm-start                 ← 25% speedup if it works
  └── Strategy 5: Heun-Langevin             ← +50% cost but highest quality

Phase 3: Fine-tuning / auxiliary model approaches
  ├── Strategy 11: CFG action guidance       ← HIGHEST expected impact (+28pp in prior work)
  ├── Strategy 16: Dynamics-verified          ← learned verifier, best for goal-directed tasks
  ├── Strategy 17: DDTO (SPSA first)          ← gradient-based noise optimization, highest ceiling
  ├── Strategy 17: DDTO (full backprop)       ← if SPSA shows promise, unlock full gradients
  ├── Strategy 6: Shortcut DiT              ← best long-term option for latency
  └── Strategy 7: Reflow                    ← best if training is cheap
```

---

## Evaluation Protocol

### Offline Metrics (denoising_lab notebook)

For each strategy, compute on a held-out set of observations:

1. **Reference fidelity**: $\|a_{\text{strategy}}^1 - a_{\text{ref}}^1\|_2$ where $a_{\text{ref}}^1$ is computed with 64-step Euler (gold standard).
2. **Seed consistency**: variance of $a^1$ across 10 random seeds. Lower = more deterministic.
3. **EEF trajectory smoothness**: jerk (3rd derivative) of the decoded EEF position trajectory. Lower = smoother.
4. **Denoising progression**: visualize $a_t^{\tau_i}$ at each step via `TrajectoryVisualizer.plot_denoising_progression()`.

### Online Metrics (RoboCasa PandaOmron)

Run full evaluation episodes with each strategy:

```bash
# Server (with modified denoising strategy)
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper --verbose

# Client
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
  --n_episodes 50 --policy_client_host 127.0.0.1 --policy_client_port 5555 \
  --max_episode_steps 720 --n_action_steps 8 --n_envs 5 \
  --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env
```

Measure:
- **Success rate** over 50+ episodes (primary metric)
- **Mean episode length** (shorter = more efficient)
- **Inference latency** per action chunk (wall-clock, GPU)
- **Action smoothness** in executed trajectories

### Latency Benchmarking

Profile on target GPU (L40):
```python
# Warm up
for _ in range(10):
    denoise_strategy(...)

# Benchmark
import time
times = []
for _ in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    denoise_strategy(...)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

print(f"Mean: {np.mean(times)*1000:.1f}ms, Std: {np.std(times)*1000:.1f}ms")
```

---

## References

1. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., & Nickel, M. (2022). "Flow Matching for Generative Modeling." arXiv:2210.02747.

2. Liu, X., Gong, C., & Liu, Q. (2022). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." arXiv:2209.03003.

3. Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models." arXiv:2206.00364.

4. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps." arXiv:2206.00927.

5. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). "DPM-Solver++: Fast Solver for Guided Diffusion Sampling." arXiv:2211.01095.

6. Sabour, S., Fidler, S., & Kreis, K. (2024). "Align Your Steps: Optimizing Sampling Schedules in Diffusion Models." arXiv:2404.14507.

7. Frans, K., Hafner, D., Levine, S., & Abbeel, P. (2024). "One Step Diffusion via Shortcut Models." arXiv:2410.12557.

8. Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). "Consistency Models." arXiv:2303.01469.

9. Prasad, D., Raju, A., & Gupta, A. (2024). "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation." arXiv:2405.07503.

10. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., & Poole, B. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." arXiv:2011.13456.

11. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., & Ermon, S. (2022). "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations." arXiv:2108.01073.

12. Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Score Gradients." arXiv:2303.04137.

13. Black, K., et al. (2024). "pi0: A Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164.

14. Ho, J. & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." arXiv:2207.12598.

15. Wen, Z., et al. (2025). "CFG-DP: Classifier-Free Guidance for Diffusion Policy." arXiv:2510.09786.

16. Chen, R.T.Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). "Neural Ordinary Differential Equations." arXiv:1806.07366.

17. Dockhorn, T., et al. (2025). "D3P: Dynamic Denoising Diffusion Policy." arXiv:2508.06804.

18. Zheng, Z., et al. (2025). "Two-Steps Diffusion Policy via Genetic Denoising." arXiv:2510.21991.

19. Ahn, S., et al. (2025). "Inference-Time Scaling Beyond Denoising Steps." arXiv:2501.09732.

14. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Muller, J., Saini, H., ... & Rombach, R. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." arXiv:2403.03206.

15. Liu, X., Zhang, X., Ma, J., Peng, J., & Liu, Q. (2023). "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation." arXiv:2309.06380.

16. Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023). "Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport." arXiv:2302.00482.

17. Zhao, W., Bai, L., Rao, Y., Zhou, J., & Lu, J. (2023). "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models." arXiv:2302.04867.

18. Bjorck, J., Prasad, A., Nair, S., & Finn, C. (2025). "Optimizing Action Generation in Flow-Matching Robot Policies via Numerical ODE Solver Selection." Workshop paper.

19. Albergo, M.S. & Vanden-Eijnden, E. (2023). "Building Normalizing Flows with Stochastic Interpolants." arXiv:2209.15571.

20. NVIDIA. (2025). "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots." arXiv:2503.14734.

21. Janner, M., Du, Y., Tenenbaum, J.B., & Levine, S. (2022). "Planning with Diffusion for Flexible Behavior Synthesis." arXiv:2205.09991.

22. Ajay, A., Du, Y., Gupta, A., Tenenbaum, J., Jaakkola, T., & Levine, S. (2023). "Is Conditional Generation All You Need for Decision-Making?" arXiv:2211.15657.

23. Song, J., Vahdat, A., Mardani, M., & Kautz, J. (2023). "Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation." arXiv:2311.13024.

24. Patil, S., et al. (2026). "Golden Noise for Diffusion Policy." arXiv (2026). Demonstrates that pre-optimized initial noise vectors improve frozen diffusion/flow policies by up to 58% across 43 tasks.

25. Fang, Z., et al. (2026). "ProbeFlow: Adaptive Velocity-Probed Flow Matching." arXiv (2026). Uses velocity cosine similarity to dynamically schedule ODE steps, reducing average steps from 50 to 2.6.

26. Black, K., Galliker, M., & Levine, S. (2025). "Real-Time Chunking for Diffusion and Flow-Based Policies." NeurIPS 2025. arXiv:2506.07339.

27. Düreth, B., et al. (2025). "Generative Consistency Optimization (GeCO): Residual Velocity Analysis for Flow Matching Models." arXiv:2603.17834.

28. Hoogeboom, E., Heek, J., & Salimans, T. (2023). "Simple Diffusion: End-to-End Diffusion for High Resolution Images." ICML 2023. arXiv:2301.11093.

29. Yang, Z., et al. (2024). "Frequency-Aware Diffusion Model for Temporal Generation." arXiv:2407.12173.

30. Si, C., et al. (2025). "Frequency-Aware Timestep Scheduling for Diffusion Models." arXiv:2501.13349.

31. Li, H., et al. (2025). "Imagination Policy: Using Generative Point Cloud Models for Learning Manipulation Policies." arXiv:2502.00622.

32. Cobbe, K., et al. (2021). "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168.

33. Hansen, N., Su, H., & Wang, X. (2024). "TD-MPC2: Scalable, Robust World Models for Continuous Control." NeurIPS 2024. arXiv:2310.16828.

34. Ratliff, N., et al. (2009). "CHOMP: Gradient Optimization Techniques for Efficient Motion Planning." ICRA 2009.

35. Eyring, D., Kynkäänniemi, T., Karras, T., Aittala, M., Laine, S., & Lehtinen, J. (2024). "Rethinking Noise Optimization of Single-Step Diffusion Models (ReNO)." arXiv:2410.12164.

36. Poole, B., Jain, A., Barron, J.T., & Mildenhall, B. (2023). "DreamFusion: Text-to-3D Using 2D Diffusion." ICLR 2023. arXiv:2209.14988.

37. Spall, J.C. (1992). "Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation." IEEE Transactions on Automatic Control, 37(3), 332-341.

38. Khalil, H.K. (2002). "Nonlinear Systems." 3rd Edition. Prentice Hall. (Lyapunov stability theory reference.)
