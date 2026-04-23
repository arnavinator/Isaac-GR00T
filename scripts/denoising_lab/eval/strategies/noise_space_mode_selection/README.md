## Strategy 10: Noise-Space Mode Selection via Velocity Preview

**Category:** Novel, drop-in | **NFEs:** K + 3 (~8 for K=5) | **Retraining:** None

### Overview

This is a **novel strategy** inspired by a striking finding: **the initial noise vector can change success rate by up to 58%** (Patil et al., "Golden Noise for Diffusion Policy", 2026). Different noise samples produce different action *modes* — one noise might generate "approach from the left," another "approach from the right," a third might produce an unstable oscillating trajectory. Currently, GR00T samples one noise vector and hopes for the best.

**Golden Ticket's limitation:** The original approach finds optimal noise vectors through offline Monte Carlo search — rolling out the full policy with hundreds of noise candidates and keeping the best. This requires a simulator, is computationally expensive, and produces a *fixed* noise vector that doesn't adapt to the specific observation.

**Our innovation:** Replace the offline search with an **online, per-observation 1-step velocity preview**. Sample $K$ noise candidates, run a single batched Euler step on all $K$ simultaneously, score the **fully-extrapolated action proxy** $a_{1.0}^* = \epsilon + v(\epsilon, 0)$ using a lightweight quality proxy, select the best noise, and complete the remaining 3 denoising steps only for the winner.

**Why the fully-extrapolated proxy works:** In rectified flow, $v(\epsilon, 0) \approx \text{data} - \epsilon$, so $\epsilon + v(\epsilon, 0) \approx \text{data}_{\text{predicted}}$. This single-step extrapolation is dominated by signal (not noise), giving the scoring heuristics meaningful action-quality information at zero extra NFEs. An earlier version scored the 25%-denoised $a_{0.25} = \epsilon + 0.25v$ instead, but that proxy was 75% noise and scoring it was essentially measuring noise structure — which caused the strategy to underperform baseline. The fully-extrapolated proxy resolves this.

**Why 1 step is sufficient for mode selection:** In flow matching, the first Euler step ($\tau = 0 \to 0.25$) does the most dramatic transformation — it collapses the isotropic Gaussian noise into a rough action structure. The velocity at $\tau = 0$ already encodes the full trajectory intent, so the extrapolated proxy $a_{1.0}^*$ reveals which direction the arm reaches, whether the gripper opens or closes, and which control mode is active. The remaining 3 steps refine this structure but rarely change the mode.

**Why this is novel:**
- **Golden Ticket** (Patil et al., 2026): Offline, fixed noise, requires rollout evaluation. Ours: online, per-observation, 1-step proxy evaluation.
- **Best-of-K sampling** in LLMs: Generates $K$ complete sequences, scores them, picks the best. Ours: evaluates after just 1 denoising step (not a complete generation), making it $K/(K+3) \approx 60\%$ cheaper for $K=4$.
- **Stochastic beam search**: Maintains $K$ candidates throughout generation. Ours: selects once after step 0 and commits — no ongoing parallelism needed.

### Mathematical Formulation

**Step 1 — Sample and preview:**

Sample $K$ noise candidates and batch-evaluate 1 Euler step:

$$\epsilon^{(k)} \sim \mathcal{N}(0, I), \quad k = 1, \ldots, K$$

$$v^{(k)} = v(\epsilon^{(k)},\; 0,\; o_t,\; l_t), \quad k = 1, \ldots, K$$

$$a^{(k),\, 0.25} = \epsilon^{(k)} + \Delta\tau \cdot v^{(k)} \quad \text{(for denoising continuation)}$$

$$a^{(k),\, 1.0*} = \epsilon^{(k)} + v^{(k)} \quad \text{(fully-extrapolated proxy for scoring)}$$

This is a single forward pass with batch size $K \cdot B$ (all candidates concatenated).

**Step 2 — Score:**

Evaluate each candidate's fully-extrapolated action proxy using a quality score $S$:

$$k^* = \arg\max_k \; S\!\left(a^{(k),\, 1.0*},\; v^{(k)},\; a_{\text{prev}}\right)$$

The quality proxy $S$ combines three signals (all computable from the 1-step output):

$$S(a^*, v, a_{\text{prev}}) = \underbrace{-\lambda_{\text{smooth}} \sum_{j} \| a^*[j{+}1] - a^*[j] \|^2}_{\text{temporal smoothness}} + \underbrace{-\lambda_{\text{mag}} \| v \|^2}_{\text{velocity magnitude}} + \underbrace{-\lambda_{\text{anchor}} \sum_{j=0}^{n-1} w_j \| a^*[j] - a_{\text{prev}}[n_{\text{exec}} + j] \|^2}_{\text{action-space anchor consistency}}$$

where $w_j = \gamma^j / \sum_i \gamma^i$ are geometrically decaying weights ($\gamma$ = `anchor_decay`, default 0.5). Step 0 of the overlap gets ~50% of the total weight.

- **Smoothness**: Rough predicted actions indicate a noisy or unstable mode — penalize.
- **Velocity magnitude**: Lower velocity suggests the noise was already closer to the action manifold — reward.
- **Anchor consistency** (enabled after the first chunk): The candidate's predicted near-future (steps 0 to $n-1$) should align with the previous chunk's predicted-but-unexecuted tail (steps $n_{\text{exec}}$ to $n_{\text{exec}} + n - 1$). Both tensors are action estimates in the same normalized space, making L2 distance a natural metric. The decay weighting ensures nearer predictions (which are more reliable) dominate the comparison. The server patch caches the full denoised `prev_actions` across calls and clears it on episode reset.

**Step 3 — Commit and complete:**

Denoise the selected noise for the remaining 3 steps:

$$a_t^{0.50} = a^{(k^*),\, 0.25} + \Delta\tau \cdot v(a^{(k^*),\, 0.25},\; 0.25,\; o_t,\; l_t)$$

$$a_t^{0.75} = a_t^{0.50} + \Delta\tau \cdot v(a_t^{0.50},\; 0.50,\; o_t,\; l_t)$$

$$a_t^{1.0} = a_t^{0.75} + \Delta\tau \cdot v(a_t^{0.75},\; 0.75,\; o_t,\; l_t)$$

Total: $K + 3$ NFEs. For $K = 5$: 8 NFEs. The step-0 evaluation is batched ($K \cdot B$ samples in one forward pass), so wall-clock latency is approximately 4 sequential DiT forward passes — **same as baseline** if the GPU has spare batch capacity.

### Pseudocode

```python
def denoise_with_noise_selection(
    lab,                     # DenoisingLab instance
    features,                # BackboneFeatures
    K=5,                     # number of noise candidates
    lambda_smooth=1.0,
    lambda_mag=0.1,
    lambda_anchor=0.5,
    anchor_decay=0.5,        # geometric decay for distance weighting
    prev_actions=None,       # cached denoised actions from previous chunk (auto-managed in server)
    seed=None,
):
    """Noise-space mode selection with 1-step velocity preview."""
    vl_embeds = features.backbone_features
    state_features = features.state_features
    B = vl_embeds.shape[0]
    n_exec = 8  # steps executed per chunk

    # --- Step 1: Sample K noise candidates ---
    noise_candidates = torch.randn(K, B, H, D)  # (K, B, 50, 128)

    # --- Step 2: Batch-evaluate 1 Euler step for all K candidates ---
    flat_noise = noise_candidates.reshape(K * B, H, D)
    # (replicate VLM features K times for batched forward pass)
    velocity_0 = evaluate_velocity(flat_noise, t_bucket=0, ...)
    actions_025 = flat_noise + 0.25 * velocity_0     # for denoising continuation
    actions_1_star = flat_noise + 1.0 * velocity_0   # fully-extrapolated proxy

    # Reshape back: (K, B, H, D)
    actions_1_star = actions_1_star.reshape(K, B, H, D)
    actions_025 = actions_025.reshape(K, B, H, D)
    velocities = velocity_0.reshape(K, B, H, D)

    # --- Step 3: Score each candidate using fully-extrapolated proxy ---
    scores = torch.zeros(K, B)
    for k in range(K):
        a = actions_1_star[k]  # (B, H, D) — signal-dominated proxy
        v = velocities[k]      # (B, H, D)

        # Temporal smoothness
        diffs = a[:, 1:, :] - a[:, :-1, :]
        scores[k] -= lambda_smooth * (diffs ** 2).sum(dim=(1, 2))

        # Velocity magnitude
        scores[k] -= lambda_mag * (v ** 2).sum(dim=(1, 2))

        # Action-space anchor consistency (distance-weighted L2)
        if prev_actions is not None:
            n_overlap = min(n_exec, H - n_exec)
            candidate_near = a[:, :n_overlap, :]                  # (B, n, D)
            prev_tail = prev_actions[:, n_exec:n_exec+n_overlap]  # (B, n, D)

            weights = [anchor_decay ** j for j in range(n_overlap)]
            weights = weights / sum(weights)  # normalize

            sq_dist = ((candidate_near - prev_tail) ** 2).sum(dim=2)  # (B, n)
            scores[k] -= lambda_anchor * (sq_dist * weights).sum(dim=1)

    # --- Step 4: Select best noise per batch element ---
    best_k = scores.argmax(dim=0)                                 # (B,)
    best_actions = actions_025[best_k, torch.arange(B)]           # (B, H, D)

    # --- Step 5: Complete denoising with remaining 3 steps ---
    actions = best_actions
    for step in range(1, 4):
        tau_bucket = int(step / 4.0 * 1000)
        velocity = evaluate_velocity(actions, tau_bucket, ...)
        actions = actions + 0.25 * velocity

    return actions
```

### Anchor Consistency Design

The anchor term compares the candidate's extrapolated proxy with the previous chunk's predicted-but-unexecuted tail. This is a direct measure of temporal coherence:

| Aspect | Detail |
|--------|--------|
| **What is compared** | Candidate proxy steps [0, n) vs. previous chunk steps [n_exec, n_exec + n) |
| **Space** | Both tensors are action estimates in the same normalized [-1, 1] space |
| **Metric** | Weighted L2 distance (lower distance = higher score) |
| **Weighting** | Geometric decay: step $j$ gets weight $\gamma^j / \sum \gamma^i$. Default $\gamma=0.5$ gives step 0 ~50% of total weight |
| **Rationale** | Near-horizon predictions (step 0) are more reliable than far-horizon (step 7). The decay focuses the comparison on the most trustworthy overlap region |
| **Episode boundaries** | `patch_action_head` returns a `reset()` callable that clears cached `prev_actions`. This must be hooked into `policy.reset()` to prevent stale cross-episode distortion |

**Why L2 in action space instead of cosine similarity on velocities:**
The previous approach compared velocities at mismatched denoising stages (candidate at $\tau=0$, previous chunk at $\tau=0.75$). These velocity vectors have fundamentally different magnitudes and semantics. Cosine similarity between them is a weak, noisy signal. The L2 action-space comparison avoids this by operating on action estimates (same normalized space, same semantics, same timesteps).

### How It Replaces Action Chunking

Action chunking is unchanged. The noise selection happens entirely before the main denoising loop — it selects *which* noise to denoise, not *how* to denoise it. The selected noise flows through the same 3-step Euler integration, decode pipeline, and `MultiStepWrapper` as baseline.

**Synergy with other strategies:** Noise selection is orthogonal to the denoising solver. The 3 remaining steps can use any strategy: AB2 (Strategy 3), constraint guidance (Strategy 8), or horizon-prioritized gating (Strategy 9). This makes noise selection a "meta-strategy" that stacks on top of all others.

### Configuration

```python
@dataclass
class NoiseSelectionConfig:
    K: int = 5               # noise candidates to evaluate
    lambda_smooth: float = 1.0   # temporal smoothness weight
    lambda_mag: float = 0.1      # velocity magnitude weight
    lambda_anchor: float = 0.5   # anchor consistency weight
    anchor_decay: float = 0.5    # geometric decay per overlap step
    noise_type: str = "gaussian" # "gaussian" (N(0,1)) or "uniform" (variance-matched)
    num_steps: int = 4           # denoising steps
    n_exec_steps: int = 8        # action steps executed per chunk
```

**Noise distribution options:**
- `"gaussian"` (default): Standard normal N(0,1). Matches the distribution used during model training.
- `"uniform"`: Uniform[-sqrt(3), sqrt(3)], variance-matched to N(0,1). Bounded support means no extreme outlier candidates; may provide more uniform coverage of the noise space for mode exploration. The per-dimension value range overlaps heavily with the Gaussian — any specific value (e.g., 0.75) is valid under both distributions.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Potentially very high. Golden Ticket (Patil et al., 2026) reports up to 58% relative improvement from noise optimization alone. The fully-extrapolated proxy $a_{1.0}^*$ gives a signal-dominated estimate of the clean action, enabling meaningful quality scoring. For $K = 5$, we're selecting the best of 5 modes — in multi-modal action distributions (e.g., approach object from different directions), this can avoid catastrophically bad modes. The server patch caches `prev_actions` across calls and clears on episode reset for clean anchor consistency. |
| **Risk** | (1) The scoring function weights ($\lambda$) require tuning — use `calibrate_lambdas.py` for systematic grid search. (2) For observations with unimodal action distributions, all $K$ candidates produce similar results — the selection provides no benefit. (3) The single-step proxy is a first-order extrapolation that may diverge for highly curved velocity fields, but this only affects scoring accuracy, not the denoised output. |
| **Latency** | $K + 3$ NFEs. For $K = 5$: the step-0 batch uses $5B$ samples in a single forward pass. On GPUs with spare batch capacity (typical for $B = 1$ inference), this costs the same wall-clock time as 1 sequential NFE. Total wall-clock: ~4 sequential forward passes. Use `profile_k_runtime.py` to measure actual latency for different $K$ values on your hardware. |
| **Implementation** | Moderate — requires batched forward pass with duplicated VLM features, scoring function, per-batch-element selection, and episode-reset cache clearing. |

### Prior Work and What Makes This Novel

- **Patil et al., "Golden Noise for Diffusion Policy" (2026)**: The primary inspiration. Demonstrates that a fixed, pre-optimized noise vector improves frozen diffusion/flow policies by up to 58% across 43 tasks. The noise is found via Monte Carlo search over full simulator rollouts. **Key difference:** Golden Ticket is offline (requires simulator), fixed (same noise for all observations), and expensive (hundreds of rollouts). Our approach is online (per-observation), adaptive (different noise per query), and cheap (1-step proxy).
- **Best-of-N sampling in LLMs** (e.g., Nakano et al., "WebGPT", 2021): Generates $N$ complete sequences and scores them with a reward model. **Key difference:** We evaluate after 1 step (not full generation) and use analytic scores (not a learned reward model).
- **Stochastic beam search / diverse beam search**: Maintains multiple candidates throughout generation. **Key difference:** We select once after step 0 and commit — no ongoing parallelism, dramatically lower cost.
- **ProbeFlow** (Fang et al., 2026): Uses velocity cosine similarity to dynamically skip steps. **Key difference:** ProbeFlow decides *how many* steps to take for a single noise; our approach decides *which noise* to use.

**What makes this novel:** The combination of (1) online, per-observation noise selection (2) using a 1-step velocity preview as a lightweight quality proxy (3) applied to flow-matching VLA models is, to our knowledge, unpublished. The closest work (Golden Ticket) requires offline optimization with full rollouts; our 1-step proxy eliminates this requirement entirely. The insight that mode selection happens in the first denoising step — and that this can be exploited for cheap quality improvement — is specific to the flow-matching paradigm where early steps establish gross structure.

---

### How to Run

**Terminal 1 — Server** (from repo root, main model venv):
```bash
bash scripts/denoising_lab/eval/strategies/noise_space_mode_selection/run_server.sh
# Or with custom parameters:
bash scripts/denoising_lab/eval/strategies/noise_space_mode_selection/run_server.sh \
    --K 8 --lambda-smooth 2.0 --anchor-decay 0.5 --noise-type uniform
```

**Terminal 2 — Benchmark** (from repo root, robocasa venv):
```bash
bash scripts/denoising_lab/eval/strategies/noise_space_mode_selection/run_eval.sh
# Or with more episodes:
bash scripts/denoising_lab/eval/strategies/noise_space_mode_selection/run_eval.sh --n-episodes 50
```

**Notebook / DenoisingLab:**
```python
from scripts.denoising_lab.eval.strategies.noise_space_mode_selection.strategy import (
    denoise_with_lab, NoiseSelectionConfig,
)
cfg = NoiseSelectionConfig(K=5, lambda_smooth=1.0, lambda_mag=0.1, anchor_decay=0.5)
actions, best_noise, last_velocity = denoise_with_lab(lab, features, seed=42, cfg=cfg)
decoded = lab.decode_raw_actions(actions)

# Subsequent chunks: pass prev_actions for anchor consistency
actions2, _, _ = denoise_with_lab(lab, features2, cfg=cfg, prev_actions=actions)
```

### Hyperparameter Tuning

**Lambda grid search** (`calibrate_lambdas.py`):

Loads the model once, starts a ZMQ server, and iterates over a grid of scoring weights. Each config re-patches the action head and launches the eval client subprocess.

```bash
uv run python scripts/denoising_lab/eval/strategies/noise_space_mode_selection/calibrate_lambdas.py \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --max-episode-steps 480 \
    --K 8 \
    --n-episodes 3 --seed 42 \
    --lambda-smooth 0.7 1.0 \
    --lambda-mag 0.01 \
    --lambda-anchor 1.0 2.0 \
    --noise-type gaussian uniform \
    --output-dir ~/my_Isaac-GR00T/scripts/denoising_lab/eval/strategies/noise_space_mode_selection/noise_space_mode_selection1
```

**Runtime profiling** (`profile_k_runtime.py`):

Measures inference latency for different K values.  Uses the two-venv architecture: launches `collect_obs.py` as a subprocess in the robocasa venv to gather observations, then benchmarks each K value in the model venv with CUDA synchronization.

```bash
# From repo root, model venv (single terminal — no server needed):
uv run python scripts/denoising_lab/eval/strategies/noise_space_mode_selection/profile_k_runtime.py \
    --K-values 3 5 8 12 \
    --n-warmup 5 --n-iters 50 \
    --output-dir /tmp/noise_sel_profile
```

---
