## Strategy 17: Differentiable Denoising Trajectory Optimization (DDTO)

**Category:** Novel, drop-in (requires `torch.enable_grad()`) | **NFEs:** 5–9 NFE-equivalents (configurable) | **Retraining:** None

### Overview

Every strategy in this document treats the initial noise $\epsilon \sim \mathcal{N}(0, I)$ as a *given* — a random sample that determines the output, for better or worse. Search-based strategies (10, 13) evaluate multiple random draws and pick the best; guidance strategies (8, 9, 15) steer the velocity within each step. But none of them ask the deepest question: **What is the** ***optimal*** **noise vector for this specific observation?**

**The paradigm shift — denoising as optimization, not sampling:** The entire denoising chain is a differentiable computation graph. The DiT is a standard PyTorch module composed of linear layers, attention, and layer norms — all differentiable. The Euler updates ($a^{\tau+\Delta\tau} = a^\tau + \Delta\tau \cdot v$) are trivially differentiable. Crucially, the inner denoising step (`_denoise_step_inner()` in `DenoisingLab`) carries no `@torch.no_grad()` decorator — the gradient barrier exists only at the outer `denoise()` wrapper and can be bypassed by calling the inner step directly under `torch.enable_grad()`.

This means we can backpropagate through the DiT to compute the exact gradient of a quality objective with respect to the initial noise — then update ε to improve quality, and re-denoise from the optimized noise. The DiT weights $\theta$ are completely frozen — we optimize the *input*, not the model.

**Why this is more powerful than search:** Strategy 10 (noise selection) evaluates $K$ random noise vectors and picks the best — a zero-order (derivative-free) approach. DDTO computes a *first-order gradient* that points directly toward better noise in the full $50 \times 128 = 6400$-dimensional noise space. A single gradient step in 6400 dimensions provides far more information than random sampling, because the gradient concentrates the entire local loss landscape into one vector.

**The key design decision: 1-step backprop, not 4-step.** Backpropagating through all 4 DiT calls is expensive (~264ms, ~6GB activation memory). Instead, we backprop through **a single DiT call at step 0 only**. This is sufficient because step 0 is where mode selection happens — after 1 Euler step, the gross trajectory structure (approach direction, gripper intent, control mode) is established. The gradient $\partial L / \partial \epsilon$ through 1 DiT call tells us exactly how to change ε to improve the step-0 output. The cost is 1 forward+backward through 1 DiT call (~1.5GB activations, ~48ms) plus 4 standard Euler steps for the final denoising (~64ms). Total: ~112ms.

**Why the fully-extrapolated proxy, not the 0.25-stepped state:** We compute the quality loss on $a_{1.0}^* = \epsilon + v_0$, the fully-extrapolated proxy, rather than the partially-denoised $a^{0.25} = \epsilon + 0.25 \cdot v_0$. In rectified flow, $v(\epsilon, 0) \approx \text{data} - \epsilon$, so $\epsilon + v(\epsilon, 0) \approx \text{data}_{\text{predicted}}$. This proxy is **signal-dominated** — gradients through it reflect actual action quality, not noise artifacts. The $a^{0.25}$ alternative is 75% noise and 25% signal; optimizing through it is essentially gradient descent on noise structure, which Strategy 10 discovered and corrected for its own scoring heuristics. Note: the proxy is for loss computation only — the actual Euler integration still advances from $a^{0.25}$ (the properly-stepped state at $\tau = 0.25$).

### Mathematical Formulation

**1-step differentiable probe:**

$$v_0 = v_\theta(\epsilon,\; 0,\; o_t,\; l_t)$$

$$a^{1.0*} = \epsilon + v_0 \quad \text{(fully-extrapolated proxy for quality loss — signal-dominated)}$$

$$a^{0.25} = \epsilon + 0.25 \cdot v_0 \quad \text{(for Euler continuation after noise update)}$$

**Quality objective** (deterministic, computed on the fully-extrapolated proxy $a^{1.0*}$):

$$\mathcal{L}(\epsilon) = \lambda_{\text{smooth}} \,\mathcal{L}_{\text{smooth}} + \lambda_{\text{anchor}} \,\mathcal{L}_{\text{anchor}}$$

**Smoothness loss** (temporal roughness of the predicted trajectory):

$$\mathcal{L}_{\text{smooth}}(\epsilon) = \sum_{h=0}^{H-2} \|a^{1.0*}[h{+}1] - a^{1.0*}[h]\|_2^2$$

Because the fully-extrapolated proxy is signal-dominated ($a^{1.0*} \approx \text{data}_{\text{predicted}}$), consecutive-step deltas measure actual trajectory roughness rather than noise structure. Rough predicted trajectories indicate unstable or between-mode solutions.

**Anchor consistency loss** (continuity with previous chunk's predicted-but-unexecuted tail):

$$\mathcal{L}_{\text{anchor}}(\epsilon) = \sum_{j=0}^{n-1} w_j \,\|a^{1.0*}[j] - a_{\text{prev}}[n_{\text{exec}} + j]\|_2^2$$

where $a_{\text{prev}}$ is the full denoised action tensor from the previous chunk (cached across calls, cleared on episode reset), $n_{\text{exec}}$ is the number of executed steps per chunk, $n = \min(n_{\text{exec}},\, H - n_{\text{exec}})$ is the overlap length, and $w_j = \gamma^j / \sum_i \gamma^i$ are geometrically decaying weights ($\gamma$ = `anchor_decay`, default 0.5). Step 0 of the overlap gets ~50% of the total weight, reflecting that near-horizon predictions are more reliable.

Both tensors ($a^{1.0*}$ and $a_{\text{prev}}$) are action estimates in the same normalized $[-1, 1]$ space, making L2 distance a natural metric. The decay weighting focuses the gradient signal on the most trustworthy overlap region.

**On-mode regularizer via gradient-norm penalty:**

Mode-averaging is one of the primary failure modes of flow matching with few Euler steps. When the action distribution is multi-modal (approach from left vs right), the velocity field at step 0 averages between modes, and 4-step Euler can land between them — producing an invalid action that's a weighted average of two valid strategies.

We can detect between-mode regions through a property of the velocity field's Jacobian: **on a mode, the velocity is locally insensitive to input perturbations (small Jacobian). Between modes, the velocity is highly sensitive (large Jacobian).** This is because between modes, nearby noise vectors map to different modes, producing wildly different velocities.

The gradient $g = \partial \mathcal{L} / \partial \epsilon$ that we already compute for the quality objective is propagated through the chain rule:

$$g = \frac{\partial \mathcal{L}}{\partial a^{1.0*}} \cdot \frac{\partial a^{1.0*}}{\partial \epsilon} = \frac{\partial \mathcal{L}}{\partial a^{1.0*}} \cdot \left(I + \frac{\partial v_0}{\partial \epsilon}\right)$$

The Jacobian $\partial v_0 / \partial \epsilon$ appears with coefficient **1.0** (not the 0.25 that an $a^{0.25}$ proxy would give). This 4× amplification makes $\|g\|$ a much sharper on-mode/between-mode discriminator: between-mode regions, where the Jacobian has large eigenvalues, produce proportionally larger $\|g\|$, while on-mode regions (small Jacobian) keep $\|g\|$ small.

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
2. Compute fully-extrapolated proxy $a^{1.0*} = \epsilon + v_0$. Compute $\mathcal{L}_{\text{quality}}$ on $a^{1.0*}$.
3. First backward: compute $g = \partial \mathcal{L} / \partial \epsilon$. ~24ms.
4. Second backward (Hessian-vector product): compute $g_{\text{mode}} = \partial \|g\|^2 / \partial \epsilon$. ~24ms.
5. Gradient step on $\epsilon$. Re-project to norm-sphere.
6. Forward 4-step Euler from $\epsilon^*$ without grad: 4 NFEs, ~64ms.
**Total: ~136ms, 5 NFEs + 2 backward passes through 1 DiT call.**
Memory: ~1.5GB activation cache for 1 DiT call (32 layers × 1536 dim). No gradient checkpointing needed.

**Variant B — 1-Step Backprop without Mode Regularizer (simplest):**
1. Forward step 0 with grad: 1 NFE, ~24ms.
2. Compute fully-extrapolated proxy $a^{1.0*} = \epsilon + v_0$. Compute $\mathcal{L}_{\text{quality}}$ on $a^{1.0*}$.
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
    lambda_anchor=0.5,           # anchor consistency with previous chunk
    lambda_mode=0.1,             # on-mode regularizer (0 = disable)
    # --- Anchor parameters ---
    anchor_decay=0.5,            # geometric decay per overlap step
    n_exec=8,                    # action steps executed per chunk
    # --- Optimization hyperparameters ---
    eta=0.1,                     # gradient step size (after normalization)
    # --- Previous chunk context ---
    prev_actions=None,           # (B, 50, 128) full denoised actions from previous chunk
):
    """Differentiable Denoising Trajectory Optimization (DDTO).

    Optimizes the initial noise via 1-step backprop through the DiT,
    computing quality losses on the fully-extrapolated proxy a_1.0* = ε + v_0
    (signal-dominated, ≈ data_predicted in rectified flow).

    The DiT weights are completely frozen — only the noise is optimized.

    Returns (denoised_actions, diagnostics).
    """
    device = epsilon.device
    tau_schedule = [0, 250, 500, 750]
    dt = 0.25
    H = epsilon.shape[1]  # 50

    diagnostics = {
        'quality_loss_before': None,
        'quality_loss_after': None,
        'gradient_norm': None,
        'mode_gradient_norm': None,
        'noise_shift_norm': None,
    }

    # ================================================================
    # Phase 1: 1-step forward with grad → fully-extrapolated proxy
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
        # Fully-extrapolated proxy for scoring (signal-dominated)
        # In rectified flow: ε + v(ε,0) ≈ data_predicted
        a_1_star = eps + 1.0 * v0  # (B, 50, 128)

        # --- Quality loss on fully-extrapolated proxy ---
        loss = torch.tensor(0.0, device=device)

        # Smoothness: temporal roughness of predicted trajectory
        diffs = a_1_star[:, 1:, :] - a_1_star[:, :-1, :]
        smooth_loss = diffs.pow(2).sum(dim=(1, 2)).mean()
        loss = loss + lambda_smooth * smooth_loss

        # Anchor consistency with previous chunk (overlap-region, decay-weighted)
        if prev_actions is not None and lambda_anchor > 0:
            n_overlap = min(n_exec, H - n_exec)
            candidate_near = a_1_star[:, :n_overlap, :]                  # (B, n, D)
            prev_tail = prev_actions[:, n_exec:n_exec + n_overlap, :]    # (B, n, D)

            # Geometric decay weights: step 0 gets ~50% of total weight
            weights = torch.tensor(
                [anchor_decay ** j for j in range(n_overlap)],
                device=device,
            )
            weights = weights / weights.sum()

            sq_dist = (candidate_near - prev_tail).pow(2).sum(dim=2)  # (B, n)
            anchor_loss = (sq_dist * weights.unsqueeze(0)).sum(dim=1).mean()
            loss = loss + lambda_anchor * anchor_loss

        diagnostics['quality_loss_before'] = loss.item()

        # --- First backward: quality gradient ---
        # Chain rule: g = ∂L/∂a_1.0* · (I + ∂v_0/∂ε)
        # The Jacobian appears with coefficient 1.0, giving 4× sharper
        # mode discrimination than the 0.25 proxy alternative.
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
        diffs_final = a[:, 1:, :] - a[:, :-1, :]
        diagnostics['quality_loss_after'] = diffs_final.pow(2).sum(dim=(1, 2)).mean().item()

    return a, diagnostics
```

### Anchor Consistency Design

The anchor term compares the candidate's fully-extrapolated proxy with the previous chunk's predicted-but-unexecuted tail. This reuses the same design validated in Strategy 10, adapted for gradient-based optimization:

| Aspect | Detail |
|--------|--------|
| **What is compared** | Candidate proxy steps [0, n) vs. previous chunk steps [n_exec, n_exec + n) |
| **Space** | Both tensors are action estimates in the same normalized [-1, 1] space |
| **Metric** | Decay-weighted L2 distance — differentiable w.r.t. ε via the proxy |
| **Weighting** | Geometric decay: step $j$ gets weight $\gamma^j / \sum \gamma^i$. Default $\gamma=0.5$ gives step 0 ~50% of total weight |
| **Rationale** | Near-horizon predictions are more reliable; decay focuses the gradient signal on the most trustworthy overlap |
| **Episode boundaries** | `prev_actions` must be cleared on episode reset to prevent cross-episode distortion |

**Why overlap-region comparison instead of single-point:** The original formulation compared only $a[0]$ with the last executed action — a single scalar comparison in 128-dimensional space. The anchor consistency loss compares the full overlap region (up to `n_exec` timesteps), weighted by reliability. This provides a much richer gradient signal: the gradient $\partial \mathcal{L}_{\text{anchor}} / \partial \epsilon$ encodes how to shift ε to improve alignment across the entire overlap, not just at one point.

### How It Replaces Action Chunking

Action chunking is entirely unchanged. DDTO wraps around the standard denoising pipeline — it optimizes the noise, then produces the same $(B, 50, 128)$ tensor decoded identically. The `MultiStepWrapper` executes the output chunk's first `n_action_steps` timesteps as usual.

**Why the compute budget is feasible:** With `n_action_steps=8` at 10Hz control, the policy is queried every $8 \times 100\text{ms} = 800\text{ms}$. Variant A (136ms) uses only 17% of this budget. The server processes the observation and computes VLM embeddings (~50ms) in parallel with the last few executed actions, so the effective budget is ~750ms. Both variants fit comfortably.

**Temporal consistency across chunks:** The $\mathcal{L}_{\text{anchor}}$ component is unique to DDTO — no other strategy explicitly optimizes the gradient of temporal coherence with respect to the initial noise. By backpropagating the anchor consistency loss through the DiT, DDTO shifts ε in a direction that reduces chunk-boundary discontinuities, producing smoother long-horizon trajectories.

**Composition with other strategies:** DDTO optimizes *which noise* to denoise; the remaining 4 Euler steps can use any solver — AB2 (Strategy 3), constraint guidance (Strategy 8), or horizon-prioritized gating (Strategy 9). DDTO is a noise optimizer, not a solver replacement, so it stacks cleanly on top. It also composes naturally with Strategy 10: use noise-space mode selection to choose the best of $K$ candidates, then apply DDTO's gradient refinement to the winner — coarse global search followed by fine local optimization. This hybrid is worth exploring if empirical results show DDTO getting stuck in bad basins, but the mode regularizer ($\lambda_{\text{mode}}$) already pushes toward stable modes, so the standalone formulation is the right starting point.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | High. DDTO is the only strategy that uses first-order gradient information to optimize noise in the 6400-dimensional space. The quality losses (smoothness, anchor consistency) are computed on the fully-extrapolated proxy $a_{1.0}^* \approx \text{data}_{\text{predicted}}$ — a signal-dominated estimate where gradients reflect actual action quality rather than noise structure. The on-mode regularizer ($\lambda_{\text{mode}}$) benefits from the proxy choice: the chain rule $g = \partial \mathcal{L}/\partial a^{1.0*} \cdot (I + \partial v_0 / \partial \epsilon)$ gives the velocity Jacobian 4× more influence on $\|g\|$ compared to the $a^{0.25}$ alternative, making the on-mode/between-mode discrimination sharper. However, a single gradient step on a non-convex landscape provides a local improvement, not a global optimum — the practical benefit depends on how smooth the loss landscape is around the sampled ε. |
| **Risk** | (1) **Non-convexity:** The loss landscape through the DiT is highly non-convex. A single gradient step may not improve the final 4-step output if the landscape changes significantly between ε and ε*. This is mitigated by the normalized step size (η=0.1, a small displacement). (2) **Proxy-to-final correlation:** We optimize quality on $a_{1.0}^*$ (a 1-step extrapolation) but care about quality of the full 4-step output. The fully-extrapolated proxy is signal-dominated and correlates strongly with final output for mode-level properties (approach direction, gripper intent), though fine details (exact timing) may diverge. (3) **Hessian-vector product cost:** The on-mode regularizer requires a second backward pass. For $\lambda_{\text{mode}} = 0$ (Variant B), this is skipped, saving ~24ms. (4) **Memory:** 1 DiT call's activation cache is ~1.5GB — manageable on L40 (48GB), but non-trivial for multi-env evaluation. |
| **Latency** | Variant A (with mode regularizer): ~136ms. Variant B (without): ~112ms. Both within the 800ms action chunking budget. |
| **Implementation** | Moderate. The key change: call `_forward_dit()` (which returns velocity without the Euler update) under `torch.enable_grad()`, compute $a_{1.0}^* = \epsilon + v_0$, compute loss on $a_{1.0}^*$, call `torch.autograd.grad()`. The DiT supports gradient checkpointing (`_supports_gradient_checkpointing = True`). Total: ~100 lines. |

### Prior Work

- **Eyring et al., "Rethinking Noise Optimization of Single-Step Diffusion Models" (ReNO)** — arXiv:2410.12164 (2024). Optimized initial noise for text-to-image diffusion by backpropagating CLIP and aesthetic losses through the denoising chain. **Key differences:** ReNO uses external quality models (CLIP); DDTO uses physics-based quality losses (smoothness, anchor consistency). ReNO backprops through 50+ diffusion steps; DDTO backprops through 1 DiT call. ReNO is offline (seconds per image); DDTO is real-time (~112ms per action chunk).
- **Patil et al., "Golden Noise for Diffusion Policy" (2026)**. Pre-optimizes noise vectors offline via Monte Carlo rollouts. **Key differences:** Golden Noise is offline (minutes of pre-computation); DDTO is online (single gradient step per query). Golden Noise requires a simulator for evaluation; DDTO uses analytic quality losses.
- **Poole et al., "DreamFusion" (2023)**. Score Distillation Sampling through diffusion models. Both DreamFusion and DDTO exploit the differentiability of the generative model — DreamFusion optimizes a NeRF, DDTO optimizes the noise input.

**What makes this novel for VLAs:** DDTO is the first strategy to optimize VLA noise via exact gradients through the DiT at test time, using a 1-step backprop design that keeps the cost tractable for real-time control. The fully-extrapolated proxy $a_{1.0}^* = \epsilon + v_0 \approx \text{data}_{\text{predicted}}$ ensures that gradients reflect action quality rather than noise structure — a correction validated by Strategy 10's experience. The on-mode regularizer — penalizing $\|g\|^2$ to push noise toward regions where the velocity field is locally insensitive to input perturbations — is a novel mechanism for avoiding between-mode artifacts, and the proxy choice amplifies its discriminative power by expressing the full Jacobian $\partial v_0 / \partial \epsilon$ with coefficient 1.0 in the chain rule.

---

### How to Run

**Terminal 1 — Server** (from repo root, main model venv):
```bash
bash scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/run_server.sh
# Or with custom parameters:
bash scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/run_server.sh \
    --eta 0.2 --lambda-mode 0.0 --lambda-smooth 2.0
```

**Terminal 2 — Benchmark** (from repo root, robocasa venv):
```bash
bash scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/run_eval.sh
# Or with more episodes:
bash scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/run_eval.sh --n-episodes 50
```

**Notebook / DenoisingLab:**
```python
from scripts.denoising_lab.eval.strategies.differentiable_denoising_trajectory_optimization.strategy import (
    denoise_with_lab, DDTOConfig,
)
cfg = DDTOConfig(lambda_smooth=1.0, lambda_anchor=0.5, eta=0.1)
actions, diagnostics = denoise_with_lab(lab, features, seed=42, cfg=cfg)
decoded = lab.decode_raw_actions(actions)

# Subsequent chunks: pass prev_actions for anchor consistency
actions2, diag2 = denoise_with_lab(lab, features2, cfg=cfg, prev_actions=actions)
```

### Hyperparameter Tuning

**Lambda grid search** (`calibrate_lambdas.py`):

Loads the model once, starts a ZMQ server, and iterates over a grid of quality-loss weights and step sizes. Each config re-patches the action head and launches the eval client subprocess.

```bash
uv run python scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/calibrate_lambdas.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
                robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --max-episode-steps 400 480 \
    --n-episodes 15 --seed 42 \
    --lambda-smooth 0.5 1.0 \
    --lambda-anchor 0.25 0.5 1.0 \
    --lambda-mode 0.0 0.05 0.1 \
    --eta 0.05 0.1 0.2 \
    --output-dir /tmp/calibration_results/ddto
```

---
