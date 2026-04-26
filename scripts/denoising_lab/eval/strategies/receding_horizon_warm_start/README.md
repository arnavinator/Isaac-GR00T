## Strategy 4: Receding-Horizon Warm-Start Denoising

**Category:** Drop-in (not novel — see GPC, BRIDGER, STEP, OFP) | **NFEs:** 2–4 (mode-dependent) | **Retraining:** None

### Overview

This strategy exploits the temporal structure of receding-horizon action chunking.

Currently, every time GR00T predicts a new action chunk, it starts from pure Gaussian noise $a_t^0 \sim \mathcal{N}(0, I)$, discarding all knowledge from the previous chunk. But the previous chunk's un-executed actions (steps 8–15) overlap significantly with the new chunk's first 8 steps — the robot hasn't moved much in 8 control steps, and the task goal hasn't changed. This overlap is wasted information.

**Two warm-start modes:**

**partial_denoise (default):** Initialize from the shifted un-executed tail, re-noise to $\tau_{\text{start}} = 0.5$ (preserving 50% of the warm signal), and denoise from there with 2 NFEs. In v1 ($\tau_{\text{start}} = 0.25$), only 25% of the warm signal survived — barely different from pure noise. The higher $\tau_{\text{start}}$ was motivated by the analysis that the un-executed tail (steps 8–15) is the *least reliable* part of the prediction, so moderate signal preservation (50%) balances warm information against the risk of preserving prediction errors.

**noise_bias:** Keep all 4 NFEs. Blend the shifted un-executed tail into fresh Gaussian noise at strength $\beta$ (default 0.15), then renormalize per-sample to preserve the expected Gaussian norm. The DiT sees near-standard noise at $\tau = 0$ but biased toward the warm trajectory. This avoids the tau/noise-level mismatch of partial_denoise while still injecting temporal coherence.

### Mathematical Formulation

Let $a_{t-1}^1$ be the fully denoised action chunk from the previous query (16 timesteps). After executing steps 0–7, steps 8–15 remain unused. We construct an informed initial state for the new chunk:

**Step 1: Temporal shift.** Shift the un-executed actions backward by 8 positions:

$$\tilde{a}_t[0\!:\!8] = a_{t-1}^1[8\!:\!16]$$

For positions 8–15 of the new chunk (which have no overlap), sample fresh noise:

$$\tilde{a}_t[8\!:\!16] \sim \mathcal{N}(0, I)$$

**Step 2a (partial_denoise mode): Partial re-noising.** Add calibrated noise to the shifted actions to bring them to noise level $\tau_{\text{start}}$ (default $\tau_{\text{start}} = 0.5$):

$$a_t^{\tau_{\text{start}}}[0\!:\!8] = (1 - \tau_{\text{start}}) \cdot \epsilon + \tau_{\text{start}} \cdot \tilde{a}_t[0\!:\!8], \quad \epsilon \sim \mathcal{N}(0, I)$$

Following GR00T's rectified flow interpolation convention where $\tau = 0$ is pure noise and $\tau = 1$ is clean.

**Step 3a: Denoise from $\tau_{\text{start}}$ instead of $\tau = 0$.** With $\tau_{\text{start}} = 0.5$, we skip the first two denoising steps, running only 2 steps:

$$\tau \in \{0.50, 0.75\} \quad \text{(2 steps, 2 NFEs)}$$

**Step 2b (noise_bias mode):** Instead of re-noising, blend the shifted actions into fresh Gaussian noise and renormalize:

$$a_t^0 = \text{normalize}\big((1 - \beta) \cdot \epsilon + \beta \cdot \tilde{a}_t\big) \quad \text{where } \|\text{normalize}(x)\| = \|\epsilon\|$$

**Step 3b:** Denoise from $\tau = 0$ using all 4 standard Euler steps (4 NFEs).

### Pseudocode

```python
def denoise_warm_start(
    a_noise,
    prev_chunk,         # (B, 50, 128) raw padded actions from previous query, or None
    vl_embeds, state_embeds, embodiment_id,
    mode="partial_denoise",  # "partial_denoise" or "noise_bias"
    tau_start=0.5,           # partial_denoise: where to resume denoising
    beta=0.15,               # noise_bias: blend strength
    n_executed=8,            # how many steps were executed from previous chunk
):
    """Warm-start denoising from previous chunk's un-executed actions."""

    if prev_chunk is not None:
        # Shift: previous steps 8-15 become new steps 0-7
        warm = torch.randn_like(a_noise)
        remaining = a_noise.shape[1] - n_executed
        warm[:, :remaining, :] = prev_chunk[:, n_executed:, :]

        if mode == "noise_bias":
            # Blend warm signal into fresh noise, renormalize to preserve Gaussian norm
            epsilon = torch.randn_like(a_noise)
            biased = (1 - beta) * epsilon + beta * warm
            a = biased * (epsilon.norm() / biased.norm())  # per-sample renorm
            start_step = 0  # full 4-step Euler
        else:
            # Partial re-noising via rectified flow interpolation
            epsilon = torch.randn_like(warm[:, :remaining, :])
            warm[:, :remaining, :] = (
                (1 - tau_start) * epsilon + tau_start * warm[:, :remaining, :]
            )
            a = warm
            start_step = round(tau_start * 4)  # skip first step(s)
    else:
        a = a_noise
        start_step = 0

    # Denoise
    dt = 0.25
    for step in range(start_step, 4):
        tau = step / 4
        tau_bucket = int(tau * 1000)
        velocity = DiT(a, tau_bucket, vl_embeds, state_embeds, embodiment_id)
        a = a + dt * velocity

    return a
```

### How It Replaces Action Chunking

This strategy *modifies the initialization* of each chunk rather than the integration method. The `MultiStepWrapper` still executes 8 of 16 predicted steps. The key change is that the un-executed predictions from the previous chunk seed the next chunk, creating temporal coherence.

**Interaction with action chunking:**
- First chunk of each episode: standard 4-step denoising (no previous chunk exists)
- Subsequent chunks (partial_denoise, default): 2-step denoising from warm-start, saving 50% compute
- Subsequent chunks (noise_bias): full 4-step denoising with biased initialization, same compute
- The overlap creates smooth action transitions between chunks — reducing the "seam" artifacts visible when chunks are generated independently

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Depends on mode. **noise_bias** preserves all 4 NFEs, so quality should be at least baseline with a potential coherence boost. **partial_denoise** (tau_start=0.5) preserves 50% of the warm signal — a much stronger starting point than v1's 25%, but with only 2 NFEs for correction. The un-initialized region (steps 8–15) still gets full denoising. |
| **Risk** | Moderate. (1) The un-executed tail (steps 8–15) is the least reliable part of the prediction — farthest from the observation. Preserving too much of an inaccurate signal can hurt. (2) In noise_bias mode, the renormalized biased noise has correct norm but non-isotropic distribution — the DiT handles this gracefully for small beta (~0.15). (3) In partial_denoise mode, the actual noise level of the warm-started input may not perfectly match the tau bucket. (4) Tuning: partial_denoise has tau_start (higher = more signal, fewer steps), noise_bias has beta (higher = more bias, more OOD). |
| **Latency** | partial_denoise (default): 50% reduction — 2 NFEs x ~16ms = ~32ms. noise_bias: same as baseline — 4 NFEs x ~16ms = ~64ms. |
| **Implementation** | Implemented. Caches raw padded actions $(B, 50, 128)$ before decoding, bypassing the re-encoding problem. |

### Prior Work and Inspiration

- **Meng et al., "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations"** — arXiv:2108.01073. The core idea of adding noise to a reference signal and denoising from a partial noise level is directly borrowed from SDEdit's approach to image editing.
- **Temporal ensembling in Diffusion Policy** — Chi et al. (arXiv:2303.04137) describe exponential weighting of overlapping action chunks for smooth transitions. Our approach achieves temporal coherence at the denoising level rather than post-hoc blending.
- **DDIM inversion** — Dhariwal & Nichol (arXiv:2105.05233). The principle of mapping between noise levels via the learned ODE is the theoretical basis for our partial re-noising step.

**Note — this strategy is NOT novel for VLAs.** Multiple concurrent works implement the same warm-start mechanism for robot diffusion/flow policies:
- **GPC** (Kurtz & Burdick, arXiv:2502.13406, ICRA 2026): `U_0 = (1-α)ε + α·U_{k-1}` for flow matching robot policies — essentially identical.
- **BRIDGER** (Chen et al., arXiv:2402.16075): Trains a conditional predictor for warm-start initialization via stochastic interpolants.
- **STEP** (Li et al., arXiv:2602.08245): Spatiotemporal consistency predictor for warm-start; outperforms BRIDGER by 21.6% with 2 denoising steps.
- **OFP** (Li et al., arXiv:2603.12480): Self-distilled one-step flow policy with warm-start from temporal correlations.

**Enhancement:** Adopt STEP's spatiotemporal consistency predictor for higher-quality warm initialization if retraining budget is available. For a zero-training approach, GPC's formulation matches ours and validates the core mechanism.

### How to run

From the **repo root**:

```bash
# Terminal 1 (model venv) — start the warm-start server
bash scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_server.sh

# Terminal 2 (sim venv) — run the reproducible benchmark
bash scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_eval.sh
```

To override defaults (e.g., more episodes):

```bash
bash scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_eval.sh \
    --n-episodes 50
```

To tune warm-start parameters:

```bash
# partial_denoise with higher tau (1 NFE, 75% warm signal):
bash scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_server.sh \
    --tau-start 0.75 --mode partial_denoise

# noise_bias mode (4 NFEs, biased initialization):
bash scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_server.sh \
    --mode noise_bias --beta 0.15
```

---
