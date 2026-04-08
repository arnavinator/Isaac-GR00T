## Strategy 17: Differentiable Denoising Trajectory Optimization (DDTO)

**Category:** Novel, drop-in (requires `torch.enable_grad()`) | **NFEs:** 5–9 NFE-equivalents (configurable) | **Retraining:** None

### Overview

Every strategy in this document treats the initial noise $\epsilon \sim \mathcal{N}(0, I)$ as a *given* — a random sample that determines the output, for better or worse. Search-based strategies (10, 13) evaluate multiple random draws and pick the best; guidance strategies (8, 9, 15) steer the velocity within each step. But none of them ask the deepest question: **What is the** ***optimal*** **noise vector for this specific observation?**

**The paradigm shift — denoising as optimization, not sampling:** The entire denoising chain is a differentiable computation graph. The DiT is a standard PyTorch module composed of linear layers, attention, and layer norms — all differentiable. The Euler updates ($a^{\tau+\Delta\tau} = a^\tau + \Delta\tau \cdot v$) are trivially differentiable. Crucially, the inner denoising step (`_denoise_step_inner()` in `DenoisingLab`) carries no `@torch.no_grad()` decorator — the gradient barrier exists only at the outer `denoise()` wrapper and can be bypassed by calling the inner step directly under `torch.enable_grad()`.

This means we can backpropagate through the DiT to compute the exact gradient of a quality objective with respect to the initial noise — then update ε to improve quality, and re-denoise from the optimized noise. The DiT weights $\theta$ are completely frozen — we optimize the *input*, not the model.

**Why this is more powerful than search:** Strategy 10 (noise selection) evaluates $K$ random noise vectors and picks the best — a zero-order (derivative-free) approach. DDTO computes a *first-order gradient* that points directly toward better noise in the full $50 \times 128 = 6400$-dimensional noise space. A single gradient step in 6400 dimensions provides far more information than random sampling, because the gradient concentrates the entire local loss landscape into one vector.

**The key design decision: 1-step backprop, not 4-step.** Backpropagating through all 4 DiT calls is expensive (~264ms, ~6GB activation memory). Instead, we backprop through **a single DiT call at step 0 only**. This is sufficient because step 0 is where mode selection happens — after 1 Euler step, the gross trajectory structure (approach direction, gripper intent, control mode) is established. The gradient $\partial L / \partial \epsilon$ through 1 DiT call tells us exactly how to change ε to improve the step-0 output. The cost is 1 forward+backward through 1 DiT call (~1.5GB activations, ~48ms) plus 4 standard Euler steps for the final denoising (~64ms). Total: ~112ms.

### Mathematical Formulation

**1-step differentiable probe:**

$$v_0 = v_\theta(\epsilon,\; 0,\; o_t,\; l_t)$$

$$a^{0.25} = \epsilon + 0.25 \cdot v_0$$

**Quality objective** (deterministic, computed on the partially-denoised $a^{0.25}$):

$$\mathcal{L}(\epsilon) = \lambda_{\text{smooth}} \,\mathcal{L}_{\text{smooth}} + \lambda_{\text{temporal}} \,\mathcal{L}_{\text{temporal}}$$

**Smoothness loss** (jerk of the emerging trajectory):

$$\mathcal{L}_{\text{smooth}}(\epsilon) = \sum_{h=2}^{H-2} \|a^{0.25}[h{+}1] - 2\,a^{0.25}[h] + a^{0.25}[h{-}1]\|_2^2$$

Even at 25% denoised, the gross trajectory shape is visible — smooth trajectories have low jerk even when noisy, while between-mode trajectories have high jerk (the averaging of two different motion directions creates discontinuities).

**Temporal consistency loss** (continuity with previous chunk):

$$\mathcal{L}_{\text{temporal}}(\epsilon) = \|a^{0.25}[0] - a_{\text{prev}}[n_{\text{exec}}]\|_2^2$$

where $a_{\text{prev}}[n_{\text{exec}}]$ is the last executed action from the previous chunk.

**On-mode regularizer via gradient-norm penalty:**

Mode-averaging is one of the primary failure modes of flow matching with few Euler steps. When the action distribution is multi-modal (approach from left vs right), the velocity field at step 0 averages between modes, and 4-step Euler can land between them — producing an invalid action that's a weighted average of two valid strategies.

We can detect between-mode regions through a property of the velocity field's Jacobian: **on a mode, the velocity is locally insensitive to input perturbations (small Jacobian). Between modes, the velocity is highly sensitive (large Jacobian).** This is because between modes, nearby noise vectors map to different modes, producing wildly different velocities.

The gradient $g = \partial \mathcal{L} / \partial \epsilon$ that we already compute for the quality objective is propagated through the chain rule:

$$g = \frac{\partial \mathcal{L}}{\partial a^{0.25}} \cdot \frac{\partial a^{0.25}}{\partial \epsilon} = \frac{\partial \mathcal{L}}{\partial a^{0.25}} \cdot \left(I + 0.25 \cdot \frac{\partial v_0}{\partial \epsilon}\right)$$

The term $\partial v_0 / \partial \epsilon$ is the Jacobian we care about. When this Jacobian has large eigenvalues (between modes), $\|g\|$ is large. When the Jacobian is small (on-mode), $\|g\|$ is small. So **$\|g\|$ is already an implicit on-mode signal** embedded in the gradient we're computing anyway.

To explicitly push toward on-mode regions, we add a gradient-norm penalty as a second-order regularizer:

$$\mathcal{L}_{\text{mode}}(\epsilon) = \|g\|^2 = \left\|\frac{\partial \mathcal{L}_{\text{quality}}}{\partial \epsilon}\right\|^2$$

The gradient of $\|g\|^2$ w.r.t. $\epsilon$ is a Hessian-vector product, computed via:

```python
g_mode = torch.autograd.grad(g, epsilon, grad_outputs=g, retain_graph=True)
```

This adds one more backward pass through the same 1 DiT call — roughly doubling the backward cost but still operating on a single DiT call, not 4.

**Combined gradient:**

$$g_{\text{total}} = g + \lambda_{\text{mode}} \cdot g_{\text{mode}}$$

The first term ($g$) pushes ε toward better quality. The second term ($g_{\text{mode}}$) pushes ε toward regions where the quality landscape is flat — i.e., stable modes where the output is robust to noise perturbations.

**Noise update:**

$$\epsilon^* = \epsilon - \eta \cdot \frac{g_{\text{total}}}{\|g_{\text{total}}\| + \varepsilon}$$

**Re-projection to the Gaussian norm-sphere:**

$$\epsilon^* \leftarrow \epsilon^* \cdot \frac{\|\epsilon\|}{\|\epsilon^*\|}$$

**Final denoising from optimized noise:**

$$a_t^{\text{final}} = \mathcal{D}_\theta(\epsilon^*) \quad \text{(standard 4-step Euler, no grad)}$$

### Implementation Variants

**Variant A — 1-Step Backprop with Mode Regularizer (recommended):**
1. Forward step 0 with grad: 1 NFE, ~24ms (with grad overhead).
2. Compute $\mathcal{L}_{\text{quality}}$ on $a^{0.25}$.
3. First backward: compute $g = \partial \mathcal{L} / \partial \epsilon$. ~24ms.
4. Second backward (Hessian-vector product): compute $g_{\text{mode}} = \partial \|g\|^2 / \partial \epsilon$. ~24ms.
5. Gradient step on $\epsilon$. Re-project to norm-sphere.
6. Forward 4-step Euler from $\epsilon^*$ without grad: 4 NFEs, ~64ms.
**Total: ~136ms, 5 NFEs + 2 backward passes through 1 DiT call.**
Memory: ~1.5GB activation cache for 1 DiT call (32 layers × 1536 dim). No gradient checkpointing needed.

**Variant B — 1-Step Backprop without Mode Regularizer (simplest):**
1. Forward step 0 with grad: 1 NFE, ~24ms.
2. Compute $\mathcal{L}_{\text{quality}}$ on $a^{0.25}$.
3. Backward: compute $g = \partial \mathcal{L} / \partial \epsilon$. ~24ms.
4. Gradient step on $\epsilon$. Re-project.
5. Forward 4-step Euler from $\epsilon^*$ without grad: 4 NFEs, ~64ms.
**Total: ~112ms, 5 NFEs + 1 backward pass through 1 DiT call.**
Simpler, cheaper, no Hessian. Use this as the baseline; add mode regularizer if between-mode artifacts are observed.

### Pseudocode

```python
import torch


def denoise_ddto(
    epsilon,                     # (B, 50, 128) initial noise
    vl_embeds,                   # vision-language embeddings (from Eagle backbone)
    state_embeds,                # state embeddings
    embodiment_id,
    backbone_output,
    lab,                         # DenoisingLab instance
    # --- Quality objective weights ---
    lambda_smooth=1.0,           # trajectory smoothness
    lambda_temporal=0.5,         # chunk boundary continuity
    lambda_mode=0.1,             # on-mode regularizer (0 = disable)
    # --- Optimization hyperparameters ---
    eta=0.1,                     # gradient step size (after normalization)
    # --- Optional context ---
    prev_chunk_action=None,      # (128,) last executed action from previous chunk
):
    """Differentiable Denoising Trajectory Optimization (DDTO).

    Optimizes the initial noise via 1-step backprop through the DiT
    to minimize smoothness + temporal consistency losses, with an optional
    gradient-norm regularizer that pushes toward stable modes.

    The DiT weights are completely frozen — only the noise is optimized.

    Returns (denoised_actions, diagnostics).
    """
    device = epsilon.device
    tau_schedule = [0, 250, 500, 750]
    dt = 0.25

    diagnostics = {
        'quality_loss_before': None,
        'quality_loss_after': None,
        'gradient_norm': None,
        'mode_gradient_norm': None,
        'noise_shift_norm': None,
    }

    # ================================================================
    # Phase 1: 1-step forward with grad → compute quality + gradient
    # ================================================================
    eps = epsilon.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        # Single DiT forward pass (1 NFE)
        v0 = lab._forward_dit(
            eps, tau_bucket=0,
            vl_embeds=vl_embeds,
            state_embeds=state_embeds,
            embodiment_id=embodiment_id,
            backbone_output=backbone_output,
        )
        a_025 = eps + dt * v0  # partially denoised action (B, 50, 128)

        # --- Quality loss ---
        loss = torch.tensor(0.0, device=device)

        # Smoothness: jerk of emerging trajectory
        accel = a_025[:, 2:] - 2 * a_025[:, 1:-1] + a_025[:, :-2]
        smooth_loss = accel.pow(2).mean()
        loss = loss + lambda_smooth * smooth_loss

        # Temporal consistency with previous chunk
        if prev_chunk_action is not None and lambda_temporal > 0:
            target = prev_chunk_action.to(device)
            if target.dim() == 1:
                target = target.unsqueeze(0).expand_as(a_025[:, 0])
            temp_loss = (a_025[:, 0] - target).pow(2).mean()
            loss = loss + lambda_temporal * temp_loss

        diagnostics['quality_loss_before'] = loss.item()

        # --- First backward: quality gradient ---
        g = torch.autograd.grad(loss, eps, create_graph=(lambda_mode > 0))[0]
        diagnostics['gradient_norm'] = g.norm().item()

        # --- Optional: on-mode regularizer via gradient-norm penalty ---
        if lambda_mode > 0:
            g_norm_sq = g.pow(2).sum()
            g_mode = torch.autograd.grad(g_norm_sq, eps)[0]
            diagnostics['mode_gradient_norm'] = g_mode.norm().item()
            g_total = g + lambda_mode * g_mode
        else:
            g_total = g

    # ================================================================
    # Phase 2: Update noise
    # ================================================================
    with torch.no_grad():
        g_total_norm = g_total.norm()
        if g_total_norm > 1e-10:
            eps_opt = eps - eta * (g_total / g_total_norm)
            # Re-project to Gaussian norm-sphere
            eps_opt = eps_opt * (epsilon.norm() / eps_opt.norm())
        else:
            eps_opt = eps

        diagnostics['noise_shift_norm'] = (eps_opt - epsilon).norm().item()

    # ================================================================
    # Phase 3: Full 4-step Euler from optimized noise (no grad)
    # ================================================================
    with torch.no_grad():
        a = eps_opt
        for tau_bucket in tau_schedule:
            v = lab._forward_dit(
                a, tau_bucket, vl_embeds, state_embeds,
                embodiment_id, backbone_output,
            )
            a = a + dt * v

        # Evaluate quality on final output for diagnostics
        accel_final = a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]
        diagnostics['quality_loss_after'] = accel_final.pow(2).mean().item()

    return a, diagnostics
```

### How It Replaces Action Chunking

Action chunking is entirely unchanged. DDTO wraps around the standard denoising pipeline — it optimizes the noise, then produces the same $(B, 50, 128)$ tensor decoded identically. The `MultiStepWrapper` executes the output chunk's first `n_action_steps` timesteps as usual.

**Why the compute budget is feasible:** With `n_action_steps=8` at 10Hz control, the policy is queried every $8 \times 100\text{ms} = 800\text{ms}$. Variant A (136ms) uses only 17% of this budget. The server processes the observation and computes VLM embeddings (~50ms) in parallel with the last few executed actions, so the effective budget is ~750ms. Both variants fit comfortably.

**Temporal consistency across chunks:** The $\mathcal{L}_{\text{temporal}}$ component is unique to DDTO — no other strategy explicitly optimizes for smooth transitions between consecutive action chunks. By penalizing discontinuity at the chunk boundary, DDTO produces smoother long-horizon trajectories.

**Composition with other strategies:** DDTO optimizes *which noise* to denoise; the remaining 4 Euler steps can use any solver — AB2 (Strategy 3), constraint guidance (Strategy 8), or horizon-prioritized gating (Strategy 9). DDTO is a noise optimizer, not a solver replacement, so it stacks cleanly on top.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | High. DDTO is the only strategy that uses first-order gradient information to optimize noise in the 6400-dimensional space. The quality losses (smoothness, temporal consistency) are deterministic and physically meaningful — their gradients through 1 DiT call point toward genuinely better noise vectors. The on-mode regularizer ($\lambda_{\text{mode}}$) provides an additional signal: large $\|g\|$ indicates a between-mode region where the velocity field is sensitive to input perturbations, and the regularizer pushes ε toward more stable regions. However, a single gradient step on a non-convex landscape provides a local improvement, not a global optimum — the practical benefit depends on how smooth the loss landscape is around the sampled ε. |
| **Risk** | (1) **Non-convexity:** The loss landscape through the DiT is highly non-convex. A single gradient step may not improve the final 4-step output if the landscape changes significantly between ε and ε*. This is mitigated by the normalized step size (η=0.1, a small displacement). (2) **1-step proxy vs 4-step quality:** We optimize quality at $\tau=0.25$ but care about quality at $\tau=1.0$. The correlation between step-0 quality and final quality is strong for mode-level properties (approach direction) but weak for fine details (gripper timing). (3) **Hessian-vector product cost:** The on-mode regularizer requires a second backward pass. For $\lambda_{\text{mode}} = 0$ (Variant B), this is skipped, saving ~24ms. (4) **Memory:** 1 DiT call's activation cache is ~1.5GB — manageable on L40 (48GB), but non-trivial for multi-env evaluation. |
| **Latency** | Variant A (with mode regularizer): ~136ms. Variant B (without): ~112ms. Both within the 800ms action chunking budget. |
| **Implementation** | Moderate. The key change: call `_forward_dit()` (which returns velocity without the Euler update) under `torch.enable_grad()`, compute loss, call `torch.autograd.grad()`. The DiT supports gradient checkpointing (`_supports_gradient_checkpointing = True`). Total: ~80 lines. |

### Prior Work

- **Eyring et al., "Rethinking Noise Optimization of Single-Step Diffusion Models" (ReNO)** — arXiv:2410.12164 (2024). Optimized initial noise for text-to-image diffusion by backpropagating CLIP and aesthetic losses through the denoising chain. **Key differences:** ReNO uses external quality models (CLIP); DDTO uses physics-based quality losses (smoothness, temporal consistency). ReNO backprops through 50+ diffusion steps; DDTO backprops through 1 DiT call. ReNO is offline (seconds per image); DDTO is real-time (~112ms per action chunk).
- **Patil et al., "Golden Noise for Diffusion Policy" (2026)**. Pre-optimizes noise vectors offline via Monte Carlo rollouts. **Key differences:** Golden Noise is offline (minutes of pre-computation); DDTO is online (single gradient step per query). Golden Noise requires a simulator for evaluation; DDTO uses analytic quality losses.
- **Poole et al., "DreamFusion" (2023)**. Score Distillation Sampling through diffusion models. Both DreamFusion and DDTO exploit the differentiability of the generative model — DreamFusion optimizes a NeRF, DDTO optimizes the noise input.

**What makes this novel for VLAs:** DDTO is the first strategy to optimize VLA noise via exact gradients through the DiT at test time, using a 1-step backprop design that keeps the cost tractable for real-time control. The on-mode regularizer — penalizing $\|g\|^2$ to push noise toward regions where the velocity field is locally insensitive to input perturbations — is a novel mechanism for avoiding between-mode artifacts, grounded in the observation that the Jacobian $\partial v / \partial \epsilon$ has larger eigenvalues at mode boundaries than at mode centers.

---
