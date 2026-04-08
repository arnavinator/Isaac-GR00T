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
