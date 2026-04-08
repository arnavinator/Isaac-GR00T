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

**What makes this novel for VLAs:** No prior work uses the residual velocity at $\tau = 1$ as both a convergence diagnostic and OOD detector for *standard time-conditioned* flow matching VLAs. The dual-use nature (quality improvement via polishing + safety improvement via OOD gating) from a single NFE is unique. The connection to iterative solver convergence theory — viewing denoising as an approximate fixed-point iteration and the residual as a convergence certificate — is a novel interpretive framework.

**Empirical caveat from GeCO:** GeCO (Zhang et al., arXiv:2603.17834) uses velocity field magnitude for OOD detection in robot flow matching, but requires retraining with a *time-unconditional* objective. Critically, GeCO's paper found that the standard time-conditioned flow matching velocity at τ=1 gives only **~0.53 AUROC** for OOD detection (near random). This suggests our OOD gating component may be empirically weak for standard time-conditioned models like GR00T. The convergence polishing step (Phase 3's refinement branch) is unaffected by this concern. **Enhancement:** If retraining is feasible, adopt GeCO's time-unconditional training for dramatically stronger OOD detection (0.93 AUROC). For zero-training deployment, rely on the polishing step and treat OOD gating as experimental.

---
