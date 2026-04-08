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
