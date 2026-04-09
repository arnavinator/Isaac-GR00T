## Strategy 4: Receding-Horizon Warm-Start Denoising

**Category:** Drop-in (not novel — see GPC, BRIDGER, STEP, OFP) | **NFEs:** 3 (25% faster) | **Retraining:** None

### Overview

This is a **novel strategy** that exploits the temporal structure of receding-horizon action chunking.

Currently, every time GR00T predicts a new action chunk, it starts from pure Gaussian noise $a_t^0 \sim \mathcal{N}(0, I)$, discarding all knowledge from the previous chunk. But the previous chunk's un-executed actions (steps 8–15) overlap significantly with the new chunk's first 8 steps — the robot hasn't moved much in 8 control steps, and the task goal hasn't changed. This overlap is wasted information.

**The idea:** Initialize the new chunk's early timesteps from the previous chunk's un-executed predictions, add calibrated noise to bring them back to a partial-denoising level, and start denoising from that level instead of from pure noise. This is analogous to **SDEdit** (image editing by adding noise then denoising) applied to the temporal action domain.

### Mathematical Formulation

Let $a_{t-1}^1$ be the fully denoised action chunk from the previous query (16 timesteps). After executing steps 0–7, steps 8–15 remain unused. We construct an informed initial state for the new chunk:

**Step 1: Temporal shift.** Shift the un-executed actions backward by 8 positions:

$$\tilde{a}_t[0\!:\!8] = a_{t-1}^1[8\!:\!16]$$

For positions 8–15 of the new chunk (which have no overlap), sample fresh noise:

$$\tilde{a}_t[8\!:\!16] \sim \mathcal{N}(0, I)$$

**Step 2: Partial re-noising.** Add calibrated noise to the shifted actions to bring them to noise level $\tau_{\text{start}}$ (e.g., $\tau_{\text{start}} = 0.25$):

$$a_t^{\tau_{\text{start}}}[0\!:\!8] = (1 - \tau_{\text{start}}) \cdot \epsilon + \tau_{\text{start}} \cdot \tilde{a}_t[0\!:\!8], \quad \epsilon \sim \mathcal{N}(0, I)$$

Following GR00T's rectified flow interpolation convention where $\tau = 0$ is pure noise and $\tau = 1$ is clean.

**Step 3: Denoise from $\tau_{\text{start}}$ instead of $\tau = 0$.** With $\tau_{\text{start}} = 0.25$, we skip the first denoising step entirely, running only 3 steps:

$$\tau \in \{0.25, 0.50, 0.75\} \quad \text{(3 steps, 3 NFEs)}$$

### Pseudocode

```python
def denoise_warm_start(
    a_noise,
    prev_chunk_decoded,     # (B, 16, action_dim) from previous query, or None
    vl_embeds, state_embeds, embodiment_id,
    tau_start=0.25,         # where to resume denoising
    n_executed=8,           # how many steps were executed from previous chunk
):
    """Warm-start denoising: 3 NFEs instead of 4 by reusing previous chunk."""

    if prev_chunk_decoded is not None:
        # Re-encode previous actions to padded space (16, action_dim) → (50, 128)
        # NOTE: encode_action is the inverse of decode_action — it must:
        #   1. Re-normalize each action key back to [-1, 1] using stored min/max stats
        #   2. Concatenate into the embodiment-specific flat vector
        #   3. Pad to (50, 128) using the embodiment's action_encoder MLP
        # This inverse does not exist in the current codebase and must be implemented.
        prev_padded = encode_action(prev_chunk_decoded)  # inverse of decode_action

        # Shift: previous steps 8-15 become new steps 0-7
        warm = torch.zeros_like(a_noise)
        warm[:, :n_executed, :] = prev_padded[:, n_executed:, :]
        warm[:, n_executed:, :] = torch.randn_like(warm[:, n_executed:, :])

        # Partial re-noising via rectified flow interpolation
        epsilon = torch.randn_like(warm[:, :n_executed, :])
        warm[:, :n_executed, :] = (
            (1 - tau_start) * epsilon + tau_start * warm[:, :n_executed, :]
        )

        a = warm
        start_step = 1  # skip step 0 (tau=0.00)
    else:
        a = a_noise
        start_step = 0

    # Denoise from tau_start
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
- Subsequent chunks: 3-step denoising from warm-start, saving 25% compute
- The overlap creates smooth action transitions between chunks — reducing the "seam" artifacts visible when chunks are generated independently

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Potentially high. The warm-started region (steps 0–7) starts much closer to the correct solution, requiring less denoising work. The un-initialized region (steps 8–15) still gets full denoising. The net effect should be smoother, more temporally coherent trajectories. |
| **Risk** | Moderate. (1) The re-encoded previous actions may accumulate errors over many chunks (drift). (2) The DiT expects specific noise-level statistics at each $\tau$; the warm-start distribution may not match. (3) The `encode_action` inverse mapping does not currently exist — it requires inverting the normalization and the `action_encoder` MLP (which is a `CategorySpecificLinear` layer, not trivially invertible). An alternative is to cache the *raw padded* actions $(B, 50, 128)$ before decoding, bypassing the re-encoding problem entirely. (4) A key tuning parameter is $\tau_{\text{start}}$ — too high (e.g., 0.5) and we skip too much denoising; too low (e.g., 0.1) and we barely save compute. |
| **Latency** | 25% reduction — 3 NFEs × ~16ms = ~48ms. |
| **Implementation** | Moderate — requires (a) caching raw padded actions from previous chunk (simpler than re-encoding), (b) partial re-noising logic, (c) tuning $\tau_{\text{start}}$. |

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
bash scripts/denoising_lab/eval/strategies/receding_horizon_warm_start/run_server.sh \
    --tau-start 0.5 --n-executed 4
```

---
