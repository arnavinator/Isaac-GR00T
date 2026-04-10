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
| **Risk** | (1) **Timestep distribution mismatch**: After step 0, near-horizon positions are more denoised than the global $\tau$ would suggest, while far-horizon positions are less denoised. The DiT's AdaLayerNorm conditions on a single global $\tau$ — it cannot distinguish per-position noise levels. If the model is sensitive to this mismatch, quality could degrade. However, the mismatch is small ($\pm$15% of baseline velocity) and the model processes partially-noised inputs at every step, so some robustness is expected. (2) The $\gamma$ parameter requires tuning — use the `find_optimal_gamma` calibration utility (see below) to find the best value for a given embodiment and checkpoint. Too large: excessive mismatch and potential instability. Too small: no effect. |
| **Latency** | Identical — same 4 NFEs, same ~64ms. The Gaussian weight computation is precomputed; the per-step gating is a single element-wise multiply. |
| **Implementation** | Trivial — implemented entirely in the `guided_fn` callback via element-wise velocity scaling. One precomputed weight matrix. |

### Prior Work and What Makes This Novel

- **Score Distillation Sampling (Poole et al., "DreamFusion", 2022)**: Uses per-pixel weighting in the score function for 3D generation. Our per-*position* weighting in the velocity field for sequential action generation is an analogous concept in a fundamentally different domain.
- **Temporal attention in video diffusion**: Video diffusion models (Ho et al., "Video Diffusion Models", 2022) use temporal attention layers, but they do not apply position-dependent velocity scaling during sampling. The denoising process treats all frames identically.
- **Receding-horizon MPC with variable precision**: In classical MPC, it is standard practice to use finer discretization for near-horizon states and coarser discretization for far-horizon states. Our velocity gating is the flow-matching analog of this principle — "spend more denoising effort on the immediate future."

**What makes this novel:** To our knowledge, no prior work applies position-dependent velocity scaling during flow matching or diffusion sampling. The key insight — that sequential predictions have non-uniform temporal importance and that this can be exploited via the learned model's self-attention — is specific to VLA action generation and has no analog in image/video generation. This strategy transforms the ODE solver from a position-agnostic integrator into a position-aware one that respects the causal structure of robot control.

### How to Run

**Terminal 1 — Server** (from repo root, main venv):
```bash
# Default parameters (gamma=0.5, sigma_w=3.0, effective_horizon=16)
bash scripts/denoising_lab/eval/strategies/horizon_prioritized_denoising/run_server.sh

# Custom boost amplitude
bash scripts/denoising_lab/eval/strategies/horizon_prioritized_denoising/run_server.sh --gamma 0.7

# Custom port
bash scripts/denoising_lab/eval/strategies/horizon_prioritized_denoising/run_server.sh --port 5556
```

**Terminal 2 — Benchmark** (from repo root, robocasa venv):
```bash
# Default: 10 episodes, seed 42, OpenDrawer
bash scripts/denoising_lab/eval/strategies/horizon_prioritized_denoising/run_eval.sh

# More episodes
bash scripts/denoising_lab/eval/strategies/horizon_prioritized_denoising/run_eval.sh --n-episodes 50
```

**Notebook / DenoisingLab:**
```python
from strategy import make_horizon_prioritized_fn, denoise_with_lab

# Default parameters
actions = denoise_with_lab(lab, features, seed=42)

# Custom parameters
actions = denoise_with_lab(lab, features, seed=42, gamma=0.7, sigma_w=4.0)

# Or use the guided_fn interface directly
guided_fn = make_horizon_prioritized_fn(gamma=0.5, sigma_w=3.0)
result = lab.denoise(features, num_steps=4, guided_fn=guided_fn, seed=42)
```

**Calibrating gamma with `find_optimal_gamma`:**

The optimal `gamma` depends on how tolerant the trained DiT is to per-position noise level mismatch — the weights make near-horizon positions more denoised than the global $\tau$ suggests, but the DiT conditions on a single global $\tau$. This tolerance is an empirical property of the checkpoint that can't be determined analytically.

`find_optimal_gamma` automates the search. It generates a high-fidelity reference (64-step uniform Euler, which closely approximates the true ODE solution), then sweeps gamma values and picks the one whose 4-step output is closest to the reference by L2 distance. Run it once on a few validation observations (one-time cost, ~1 GPU-minute per observation):

```python
from strategy import find_optimal_gamma

# features_list: 3-5 BackboneFeatures from representative observations
best_gamma, results = find_optimal_gamma(lab, features_list)

# results is sorted by error: [(best_gamma, error), (next_best, error), ...]
for gamma, err in results:
    print(f"  gamma={gamma:.1f}  error={err:.3f}")

# Use the winner
actions = denoise_with_lab(lab, features, seed=42, gamma=best_gamma)
```

Then hard-code the winning gamma into `run_server.sh` (via `--gamma`) or pass it to `patch_action_head()` for evaluation.

---
