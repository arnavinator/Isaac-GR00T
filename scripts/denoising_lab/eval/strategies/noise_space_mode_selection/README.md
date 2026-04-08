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
