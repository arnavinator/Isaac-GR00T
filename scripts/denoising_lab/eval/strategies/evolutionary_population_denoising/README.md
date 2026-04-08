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

**What makes this novel for VLAs:** GDP (Clemente et al., arXiv:2510.21991, NeurIPS 2025) directly implements population-based denoising with per-step fitness selection for robot DDPM policies — making the *core mechanism* (population + per-step selection) not novel. Our contribution adapts this to **flow matching VLAs** with three specific innovations beyond GDP: (1) The **consensus fitness term** — inter-particle velocity agreement as a self-consistency proxy — is new and specific to the population setting. (2) **Annealed mutation** that respects denoising progress (GDP uses selection only, no crossover or mutation). (3) **Crossover in partially-denoised action space** — blending structured actions after 1+ denoising steps. **Enhancement:** Adopt GDP's Stein-based fitness (distributional anomaly) as an additional scoring term alongside our smoothness/consensus. Compare selection-only (GDP-style) vs. full evolutionary operators to determine the marginal value of crossover/mutation.

---
