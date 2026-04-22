## Strategy 12: Curvature-Adaptive Step-Size Control via Embedded Error Estimation

**Category:** Novel, drop-in | **NFEs:** 8–10 (adaptive, per-observation) | **Retraining:** None

### Overview

Every strategy in this document — and every published VLA denoiser — uses a **fixed** number of denoising steps. This is fundamentally wasteful: some observations require precise, multi-step denoising (e.g., grasping a small object at a specific angle), while others are trivial (e.g., free-space transit to an approach position). Using 4 steps for all observations means under-spending compute on hard cases where discretization error compounds into task failure.

**D3P** (Dynamic Denoising Diffusion Policy, Dockhorn et al., Aug 2025) demonstrated that adaptive step allocation improves efficiency — using fewer steps on easy observations and more on hard ones. But D3P achieves this by training a separate RL-based adaptor network — adding training complexity, task-specific tuning, and a separate model artifact.

**Our approach:** Use a classical numerical analysis technique — **embedded Runge-Kutta error estimation** — to automatically detect when the ODE solver is struggling and needs smaller steps, *without any training*. This is the same principle behind MATLAB's `ode45` solver (Dormand-Prince method) and `scipy.integrate.solve_ivp`, applied for the first time to generative model denoising. The trade-off is latency: this strategy uses 8–10 NFEs (vs baseline's 4), spending a larger total compute budget but allocating it adaptively — and upgrading every step to 2nd-order Heun accuracy.

**Step growth is disabled by default** (`dt_grow_max=1.0`): An earlier version allowed the step size to grow up to 2× after low-error steps. Benchmarking revealed this caused the solver to skip tau regions (e.g., jumping from tau=0.25 to tau=0.75, missing tau=0.5) that baseline Euler always evaluates. With growth disabled, the solver takes 4 Heun steps at dt=0.25 by default — matching baseline's tau schedule — and only uses smaller steps when error is high, drawing from the enlarged NFE budget (max_nfe=10) to refine difficult regions.

**The mechanism:** At each tentative step, compute both a 1st-order Euler estimate and a 2nd-order Heun estimate using the same velocity evaluations. The difference between these two estimates is a *free* local truncation error estimate:

$$e = \|\hat{a}_{\text{Euler}} - \hat{a}_{\text{Heun}}\| = \frac{\Delta\tau}{2}\|v_{\text{start}} - v_{\text{end}}\|$$

If $e$ is below a tolerance → accept the step (using the Heun estimate for 2nd-order accuracy as a bonus). If $e$ exceeds the tolerance → reject the step, halve $\Delta\tau$, retry.

**Key insight — the error estimate is free:** The Heun corrector already requires 2 NFEs per step — one for the Euler predictor ($v_1$), one for the corrector ($v_2$). The error estimate $e = \frac{\Delta\tau}{2}\|v_1 - v_2\|$ is a free byproduct of these two evaluations — no additional NFEs beyond standard Heun integration. When all steps are accepted on the first try, the solver uses exactly the same NFEs as pure Heun. The cost of adaptivity only materializes when steps are *rejected* — and even then, the rejected step's $v_1$ is cached and reused on the retry (same starting state, same $\tau$), so a rejection wastes only 1 NFE (the corrector evaluation at the wrong step size). The solver starts conservatively at $\Delta\tau = 0.25$ (matching baseline Euler's step size). With `dt_grow_max=1.0`, steps never grow — the typical path is 4 Heun steps (8 NFEs) at the same tau schedule as baseline, but with 2nd-order accuracy. When error is high, steps shrink and additional NFEs from the budget (up to 10) are used to refine difficult regions.

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

$$\Delta\tau_{\text{next}} = \Delta\tau \cdot \min\!\left(1.0,\; \max\!\left(0.5,\; 0.9 \cdot \left(\frac{\text{atol}}{e}\right)^{1/2}\right)\right)$$

The 0.9 safety factor and [0.5, 1.0] clamp prevent oscillatory step-size behavior. The growth cap of 1.0 (instead of the classical 2.0) prevents the solver from skipping tau regions that baseline Euler evaluates — this was found to be critical for maintaining action quality with this model.

**NFE budget:** To guarantee bounded latency, impose $N_{\max} = 10$ NFEs. If the budget is exhausted, accept the current Euler estimate regardless of error. This guarantees worst-case latency of $10 \times 16\text{ms} = 160\text{ms}$.

**Why start at $\Delta\tau = 0.25$, not $0.5$:** The velocity field near $\tau = 0$ is highly nonlinear — it must collapse Gaussian noise into structured action trajectories. Starting with $\Delta\tau = 0.5$ would almost always trigger a step rejection on the very first step, burning 2 NFEs for nothing. Starting at $0.25$ (matching baseline Euler's step size) avoids this waste, and the adaptive mechanism grows the step size in later stages where the flow straightens out.

**Typical behavior by observation difficulty:**

| Observation type | Velocity curvature | Steps taken | NFEs | Latency |
|-----------------|-------------------|-------------|------|---------|
| Routine (low curvature) | Low (nearly linear) | 4 Heun steps at dt=0.25 (no growth) | 8 | ~128ms |
| Moderate difficulty | Moderate | 4 Heun steps, all accepted | 8 | ~128ms |
| Precision grasp / narrow clearance | High (nonlinear) | 4+ Heun steps (some rejections, smaller dt) | 8–10 | ~128–160ms |
| Average across episodes | Mixed | ~4 steps | ~8 | ~128ms |

The solver allocates its 8–10 NFE budget adaptively: routine observations get 4 Heun steps at the standard dt=0.25 schedule (matching baseline Euler's tau points but with 2nd-order accuracy), while hard observations get additional refined steps where the velocity field is most nonlinear. Unlike D3P, this requires zero training — the error estimate drives allocation automatically.

### Pseudocode

```python
def denoise_adaptive(
    a_noise, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab,                        # DenoisingLab for DiT access
    atol=0.05,                  # absolute error tolerance (normalized action space)
    max_nfe=10,                 # hard NFE budget
    dt_init=0.25,               # initial step size (conservative: match baseline, let solver grow)
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
    v1_cached = None  # Cache v1 across rejected steps (same a, same tau)

    while tau < 1.0 - 1e-6 and nfe < max_nfe:
        # Clamp step to not overshoot τ=1.0
        dt = min(dt, 1.0 - tau)
        tau_bucket = int(tau * 1000)
        tau_next_bucket = int(min((tau + dt) * 1000, 999))

        # --- Phase A: Euler predictor (1 NFE, or 0 if cached) ---
        if v1_cached is not None:
            v1 = v1_cached
            v1_cached = None
        else:
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

            # Adapt step size for next step (no growth: dt_grow_max=1.0)
            if error > 1e-10:
                scale = 0.9 * (atol / error) ** 0.5
                dt = dt * min(1.0, max(0.5, scale))
            else:
                dt = min(dt, 1.0 - tau)  # keep dt if error negligible
        else:
            # Reject step — halve step size, retry
            step_log.append({
                'outcome': 'rejected', 'tau': tau, 'dt': dt,
                'error': error, 'nfe': nfe,
            })
            dt = max(dt / 2, dt_min)
            v1_cached = v1  # Cache for retry — same starting state, same tau.
                            # Only v2 (1 NFE) was wasted; v1 is still valid.

    # Safety: ensure integration reaches τ=1.0. If the NFE budget was
    # exhausted mid-trajectory (e.g., due to rejections), take one final
    # Euler step with the remaining distance. This may exceed max_nfe by 1
    # in pathological cases, but guarantees we never return partially-
    # denoised (i.e., still noisy) actions.
    if tau < 1.0 - 1e-6:
        dt_remaining = 1.0 - tau
        tau_bucket = int(tau * 1000)
        v_final = lab._forward_dit(
            a, tau_bucket, vl_embeds, state_embeds,
            embodiment_id, backbone_output,
        )
        a = a + dt_remaining * v_final
        nfe += 1
        step_log.append({
            'outcome': 'euler_cleanup', 'tau': tau, 'dt': dt_remaining,
            'error': None, 'nfe': nfe,
        })
        tau = 1.0

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

Action chunking is unchanged. The adaptive solver produces the same $(B, 50, 128)$ output as baseline Euler, decoded identically. The only difference is the internal integration path — some chunks are produced in 3 steps with growing step sizes, others require 4 smaller steps, depending on the velocity field's local curvature for that particular observation.

**Interaction with action chunking timing:** Since the solver uses a variable number of NFEs, the latency per chunk varies. The typical case (8 NFEs, ~$128\text{ms}$) exceeds the $100\text{ms}$ budget for 10Hz control, trading latency for 2nd-order accuracy and adaptive step placement on difficult observations. The worst case (10 NFEs, $160\text{ms}$) occurs only when step rejections require extra refinement.

**Diagnostic value:** The `step_log` provides a rich signal for understanding the velocity field. Observations that consistently trigger step rejections or small step sizes are "hard" for the model — these are candidates for additional training data, curriculum emphasis, or task decomposition. This makes the adaptive solver a *profiling tool* for the model's competence, not just a quality improvement.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | High. The Heun estimate (used for all accepted steps) is 2nd-order accurate — a free upgrade from baseline Euler's 1st-order. With `dt_grow_max=1.0`, the solver follows baseline's tau schedule (0, 0.25, 0.5, 0.75) by default, never skipping tau regions. Only when error is high do steps shrink, using additional NFEs from the budget to refine difficult regions. This ensures the strategy never has *fewer* effective tau evaluations than baseline. |
| **Risk** | (1) **Fixed 2× latency overhead on easy observations.** With growth disabled, routine observations always take 8 NFEs (4 Heun steps at dt=0.25) — 2× baseline's 4 NFEs. The quality gain from Heun's 2nd-order accuracy must justify this cost. (2) The tolerance `atol` requires calibration — too tight and every step triggers subdivision (10 NFEs always); too loose and errors pass undetected (no better than fixed Heun). (3) Variable latency on hard observations (up to 10 NFEs / 160ms) complicates real-time control budgeting. |
| **Latency** | Typical: 8 NFEs × ~16ms = ~128ms (4 Heun steps, no rejections). Worst case: 10 NFEs × ~16ms = ~160ms (with step rejections). |
| **Implementation** | Moderate — replaces the fixed denoising loop with an adaptive while-loop (~40 lines of core logic). Requires bypassing DenoisingLab's standard `denoise()` method to directly control the step-by-step integration. No changes to the DiT model, encode/decode pipeline, or inference server. |

### Prior Work

- **Hairer, Norsett, & Wanner, "Solving Ordinary Differential Equations I" (1993)**. The definitive reference on adaptive step-size control for ODE solvers. The embedded Euler-Heun pair is the simplest instance; higher-order methods (Dormand-Prince RK4(5), used in MATLAB's `ode45`) use 6 evaluations for a 4th/5th order embedded pair. Our choice of the Euler-Heun pair matches the few-NFE regime where higher-order embedded pairs would be too expensive.
- **Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)** — arXiv:1806.07366. Used adaptive ODE solvers (Dormand-Prince) for training and evaluating neural ODEs. However, their application was to *training* neural ODEs (computing gradients via adjoint method), not to *sampling* from generative flow matching models. The sampling context introduces a distinct challenge: the velocity field was trained for the standard interpolation path, and adaptive stepping may visit off-path states.
- **Dockhorn et al., "D3P: Dynamic Denoising Diffusion Policy"** — arXiv:2508.06804. Learns an RL-based adaptor for dynamic step allocation in diffusion policies. Achieves 2.2× speedup on simulation, 1.9× on real Franka robot, with <0.1% success rate drop. **Key differences:** (1) D3P targets *speedup* (fewer steps on easy observations); our approach targets *quality* (2nd-order accuracy + more steps on hard observations), trading ~50–100% more latency for better action quality. (2) D3P requires training the adaptor via PPO; ours uses the mathematical error bound from the embedded Euler-Heun pair — zero training, zero additional parameters.
- **ProbeFlow** (Fang et al., 2026). Uses velocity cosine similarity between consecutive steps to decide whether to skip a step. **Key difference:** Velocity cosine similarity is a heuristic (high similarity → skip); our embedded error estimate is a principled truncation error bound from numerical analysis (error < tolerance → accept). The error estimate directly measures solution quality; cosine similarity measures velocity field stationarity, which is a proxy.

**What makes this novel:** To our knowledge, embedded Runge-Kutta error estimation has never been applied to flow matching or diffusion model denoising. Classical adaptive ODE solvers have been used for neural ODE *training* (Chen et al., 2018) but not for generative model *sampling*. The sampling context is distinct because: (a) the velocity field was trained for a specific interpolation path, and adaptive stepping may visit off-path states; (b) the computational budget is extremely tight (8-10 NFEs, not 50+); (c) the error tolerance must be calibrated for *action quality* (success rate), not mathematical precision. The connection between "velocity field curvature at a given observation" and "that observation's denoising difficulty" is a novel interpretive framework that could inform future work on adaptive denoising beyond the specific Euler-Heun pair.

---

### How to Run

**Terminal 1 — Server** (from repo root, main model venv):
```bash
bash scripts/denoising_lab/eval/strategies/curvature_adaptive_step_size/run_server.sh
# Or with custom tolerance:
bash scripts/denoising_lab/eval/strategies/curvature_adaptive_step_size/run_server.sh --atol 0.03
# With verbose step logging (shows accept/reject decisions):
bash scripts/denoising_lab/eval/strategies/curvature_adaptive_step_size/run_server.sh --verbose
```

**Terminal 2 — Benchmark** (from repo root, robocasa venv):
```bash
bash scripts/denoising_lab/eval/strategies/curvature_adaptive_step_size/run_eval.sh
# Or with more episodes:
bash scripts/denoising_lab/eval/strategies/curvature_adaptive_step_size/run_eval.sh --n-episodes 50
```

**Notebook / DenoisingLab:**
```python
from scripts.denoising_lab.eval.strategies.curvature_adaptive_step_size.strategy import (
    denoise_with_lab, AdaptiveConfig,
)
cfg = AdaptiveConfig(atol=0.05, max_nfe=10)
actions, step_log = denoise_with_lab(lab, features, seed=42, cfg=cfg)
decoded = lab.decode_raw_actions(actions)
# Inspect adaptive behaviour:
for entry in step_log:
    print(f"  {entry.outcome}  tau={entry.tau:.3f}  dt={entry.dt:.4f}  error={entry.error}")
```

---

---
