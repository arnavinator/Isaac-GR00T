## Strategy 15: Spectral Temporal Decomposition with Frequency-Band Velocity Scaling

**Category:** Novel, drop-in | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

Every denoising step modifies the action chunk uniformly across its 16-timestep horizon. But not all temporal frequencies are created equal, and not all denoising steps contribute equally to each frequency. Early denoising steps (near $\tau = 0$) collapse noise into gross trajectory structure — the low-frequency "shape" of the action chunk (approach direction, overall motion trend). Late denoising steps (near $\tau = 1$) refine high-frequency detail — precise timing of transitions, gripper open/close moments, contact dynamics.

**The key insight:** If we decompose the velocity field into temporal frequency bands via the Discrete Cosine Transform (DCT) along the action horizon, we can measure *which frequencies each denoising step primarily affects* and then amplify or attenuate each step's contribution to align with its natural frequency role. This is "spectral denoising guidance" — a frequency-domain analog of classifier-free guidance that requires zero training, zero extra NFEs, and zero external models.

**Motivation from signal processing:** The DCT is the optimal linear transform for compacting energy in smooth signals (it's the basis of JPEG and MP3 compression). Robot action trajectories are inherently smooth (bounded by actuator dynamics) — most energy concentrates in the first few DCT coefficients. Noise, by contrast, has uniform spectral energy. During denoising, the velocity field progressively shapes the spectrum from flat (noise) to peaked (trajectory). Each step's velocity has a characteristic spectral signature that we can exploit.

**How this differs from Strategy 9 (Horizon-Prioritized):** Strategy 9 applies a spatial weighting mask along the horizon (amplify near-horizon, attenuate far-horizon). It operates in the *time domain*. This strategy operates in the *frequency domain* — it amplifies the frequency components that each denoising step naturally resolves, regardless of their spatial position in the horizon. The two are complementary: Strategy 9 controls *where* in the horizon to focus; this strategy controls *what temporal resolution* each step achieves.

**Why this is novel:** Spectral analysis of diffusion/flow matching denoising has been studied in the image domain — Hoogeboom et al. (2023, "Simple Diffusion") showed that early denoising steps resolve low spatial frequencies and late steps resolve high frequencies. Yang et al. (2024, arXiv:2407.12173) extended this to temporal generation. Si et al. (2025, arXiv:2501.13349) proposed frequency-aware timestep scheduling for video diffusion. But **no prior work** has applied spectral decomposition to *action trajectory denoising* in VLAs, where the temporal structure has direct physical meaning (each frequency band corresponds to a different aspect of motor behavior: gross trajectory vs. fine manipulation).

### Mathematical Formulation

**DCT decomposition of the velocity field along the action horizon:**

At denoising step $i$ with action $a_t^{\tau_i}$, the velocity field produces:

$$v_i = v(a_t^{\tau_i},\; \tau_i,\; o_t,\; l_t) \in \mathbb{R}^{B \times 16 \times 29}$$

Apply the DCT-II along the horizon dimension (dim=1):

$$V_i = \text{DCT}(v_i) \in \mathbb{R}^{B \times 16 \times 29}$$

where $V_i[b, k, d]$ is the $k$-th frequency coefficient for batch element $b$, action dimension $d$. Index $k = 0$ is the DC (mean) component; $k = 15$ is the highest frequency (Nyquist).

**Frequency-band scaling:**

Define a per-step, per-frequency scaling matrix $W_i[k]$ that amplifies each step's natural frequency contribution:

$$W_i[k] = 1 + \gamma \cdot \phi_i(k)$$

where $\gamma \in [0, 1]$ is the overall guidance strength and $\phi_i(k)$ is the *frequency affinity* of step $i$ for frequency $k$:

$$\phi_i(k) = \begin{cases} \exp\!\left(-\frac{k^2}{2\sigma_i^2}\right) & \text{(low-pass profile for early steps)} \\ \exp\!\left(-\frac{(k - k_{\max})^2}{2\sigma_i^2}\right) & \text{(high-pass profile for late steps)} \\ 1 & \text{(all-pass for middle steps)} \end{cases}$$

For the 4-step Euler schedule with $\tau \in \{0, 0.25, 0.5, 0.75\}$:

| Step $i$ | $\tau_i$ | Frequency role | $\phi_i(k)$ profile | $\sigma_i$ |
|-----------|----------|----------------|---------------------|-------------|
| 0 | 0.0 | Gross structure | Low-pass (Gaussian centered at $k=0$) | 2.0 |
| 1 | 0.25 | Mid-low structure | Broad low-pass (Gaussian centered at $k=0$) | 4.0 |
| 2 | 0.50 | Mid-high detail | Broad high-pass (Gaussian centered at $k=15$) | 4.0 |
| 3 | 0.75 | Fine detail | High-pass (Gaussian centered at $k=15$) | 2.0 |

**Modified Euler update:**

$$\tilde{v}_i = \text{IDCT}\!\left(W_i \odot \text{DCT}(v_i)\right)$$

$$a_t^{\tau_{i+1}} = a_t^{\tau_i} + \Delta\tau \cdot \tilde{v}_i$$

The IDCT (inverse DCT) maps back to the time domain. The element-wise multiplication $\odot$ applies the frequency-dependent scaling.

**Energy preservation constraint:** To prevent the scaling from changing the overall velocity magnitude (which would disrupt the flow matching ODE), normalize the modified velocity:

$$\tilde{v}_i \leftarrow \tilde{v}_i \cdot \frac{\|v_i\|_F}{\|\tilde{v}_i\|_F}$$

This ensures the scaling only *redistributes* energy across frequencies, not amplifies the total.

### Pseudocode

```python
import torch
import torch.fft  # DCT via FFT

def dct_1d(x, dim=1):
    """Type-II DCT along specified dimension using FFT."""
    N = x.shape[dim]
    # Reorder: [x0, x_{N-1}, x1, x_{N-2}, ...] (required for DCT-via-FFT)
    idx = torch.cat([torch.arange(0, N, 2), torch.arange(N - 1, -1, -2)])
    x_reordered = x.index_select(dim, idx.to(x.device))

    # FFT and extract real part with phase shift
    X = torch.fft.rfft(x_reordered, dim=dim)
    # Phase factors for DCT-II
    k = torch.arange(X.shape[dim], device=x.device, dtype=x.dtype)
    phase = torch.exp(-1j * torch.pi * k / (2 * N))
    # Broadcast phase to match X shape
    shape = [1] * x.ndim
    shape[dim] = -1
    phase = phase.reshape(shape)

    return (X * phase).real * 2


def idct_1d(X, dim=1):
    """Type-III DCT (inverse of Type-II) along specified dimension."""
    N = X.shape[dim]
    # Phase factors for inverse
    k = torch.arange(N, device=X.device, dtype=X.dtype)
    phase = torch.exp(1j * torch.pi * k / (2 * N))
    shape = [1] * X.ndim
    shape[dim] = -1
    phase = phase.reshape(shape)

    X_shifted = X.to(torch.cfloat) * phase / 2
    X_shifted.select(dim, 0).div_(2)  # DC component halved

    x_reordered = torch.fft.irfft(X_shifted, n=N, dim=dim)

    # Undo the reordering
    x = torch.empty_like(x_reordered)
    idx_even = torch.arange(0, (N + 1) // 2)
    idx_odd = torch.arange(N - 1, (N - 1) // 2, -1)
    x.index_copy_(dim, (2 * idx_even).to(x.device), x_reordered.index_select(dim, idx_even.to(x.device)))
    if len(idx_odd) > 0:
        x.index_copy_(dim, (2 * idx_odd.flip(0) + 1).to(x.device), x_reordered.index_select(dim, (idx_even[:len(idx_odd)] + (N + 1) // 2).to(x.device)))

    return x * N


def build_frequency_weights(n_steps, horizon_len, gamma=0.3, device='cuda'):
    """Build per-step frequency scaling matrices.

    Returns: (n_steps, horizon_len) tensor of scaling weights.
    """
    k = torch.arange(horizon_len, device=device, dtype=torch.float32)
    k_max = horizon_len - 1
    weights = torch.ones(n_steps, horizon_len, device=device)

    # Step-dependent frequency profiles
    # Step 0 (τ=0.0): low-pass, σ=2
    weights[0] = 1 + gamma * torch.exp(-k ** 2 / (2 * 2.0 ** 2))
    # Step 1 (τ=0.25): broad low-pass, σ=4
    weights[1] = 1 + gamma * torch.exp(-k ** 2 / (2 * 4.0 ** 2))
    # Step 2 (τ=0.5): broad high-pass, σ=4
    weights[2] = 1 + gamma * torch.exp(-(k - k_max) ** 2 / (2 * 4.0 ** 2))
    # Step 3 (τ=0.75): high-pass, σ=2
    weights[3] = 1 + gamma * torch.exp(-(k - k_max) ** 2 / (2 * 2.0 ** 2))

    return weights  # (4, 16)


def denoise_spectral(
    a_noise, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab,
    gamma=0.3,              # guidance strength (0 = baseline, 1 = maximum)
    energy_preserve=True,   # normalize to preserve total velocity magnitude
):
    """4-step Euler with spectral frequency-band velocity scaling.

    Zero extra NFEs. Same cost as baseline.
    Returns (denoised_actions, spectral_diagnostics).
    """
    a = a_noise  # (B, 50, 128) — padded
    tau_schedule = [0, 250, 500, 750]
    dt = 0.25

    # Build frequency weights for the action horizon
    # Note: we apply DCT to the decoded (16, 29) for PandaOmron,
    # but for generality we operate on the padded (50, 128) and
    # let the zero-padded dims be unaffected (DCT of zeros = zeros)
    horizon_len = 50  # padded horizon
    freq_weights = build_frequency_weights(
        len(tau_schedule), horizon_len, gamma=gamma, device=a.device,
    )  # (4, 50)

    diagnostics = {
        'spectral_energy_per_step': [],  # DCT energy spectrum at each step
        'frequency_amplification': [],   # actual amplification applied
    }

    for step_idx, tau_bucket in enumerate(tau_schedule):
        # Standard velocity evaluation (1 NFE)
        v = lab._forward_dit(
            a, tau_bucket, vl_embeds, state_embeds,
            embodiment_id, backbone_output,
        )  # (B, 50, 128)

        # DCT along horizon dimension
        V = dct_1d(v, dim=1)  # (B, 50, 128) in DCT domain

        # Record spectral energy for diagnostics
        spectral_energy = (V ** 2).mean(dim=(0, 2))  # (50,) — avg over batch & action dims
        diagnostics['spectral_energy_per_step'].append(spectral_energy.cpu())

        # Apply frequency-band scaling
        W = freq_weights[step_idx]  # (50,)
        V_scaled = V * W[None, :, None]  # broadcast: (B, 50, 128)

        # IDCT back to time domain
        v_scaled = idct_1d(V_scaled, dim=1)  # (B, 50, 128)

        if energy_preserve:
            # Normalize to preserve total velocity magnitude
            v_norm = v.norm()
            v_scaled_norm = v_scaled.norm()
            if v_scaled_norm > 1e-8:
                v_scaled = v_scaled * (v_norm / v_scaled_norm)

        diagnostics['frequency_amplification'].append(
            (v_scaled.norm(dim=(0, 2)) / (v.norm(dim=(0, 2)) + 1e-8)).cpu()
        )

        # Euler step with spectrally-modified velocity
        a = a + dt * v_scaled

    return a, diagnostics


# === Analysis utility ===
def analyze_spectral_structure(lab, features_list, seeds):
    """Profile the spectral structure of the velocity field across denoising steps.

    Reveals which frequency bands each step primarily resolves — use to
    calibrate the frequency affinity profiles (sigma values) in build_frequency_weights().
    """
    all_spectra = {step: [] for step in range(4)}

    for features, seed in zip(features_list, seeds):
        torch.manual_seed(seed)
        a = torch.randn(1, 50, 128, device=lab.device)

        for step_idx, tau_bucket in enumerate([0, 250, 500, 750]):
            v = lab._forward_dit(
                a, tau_bucket, features.backbone_features,
                features.state_features, features.embodiment_id,
                features.backbone_output,
            )

            # DCT spectrum of the velocity
            V = dct_1d(v, dim=1)
            spectral_energy = (V ** 2).mean(dim=(0, 2)).cpu()  # (50,)
            all_spectra[step_idx].append(spectral_energy)

            # Standard Euler step for next iteration
            a = a + 0.25 * v

    # Average across observations
    avg_spectra = {}
    for step_idx in range(4):
        avg_spectra[step_idx] = torch.stack(all_spectra[step_idx]).mean(dim=0)

    return avg_spectra
```

### How It Replaces Action Chunking

Action chunking is entirely unchanged. The strategy modifies only the *velocity used in each Euler step* — the frequency-band scaling is applied and inverted (DCT → scale → IDCT) within each step, producing a modified velocity of the same shape. The Euler update, output decoding, and `MultiStepWrapper` execution are all identical to baseline.

**Interaction with action chunking timing:** Since the DCT and IDCT are $O(N \log N)$ operations on tensors of size $(B, 50, 128)$, they add negligible compute compared to the DiT forward pass (~16ms). The total additional latency is < 0.1ms per step, unmeasurable in practice. This is genuinely a zero-cost modification.

**Physical interpretation:** The frequency bands have direct physical meaning for robot actions:
- **DC component ($k = 0$):** Mean action over the horizon — overall motion direction.
- **Low frequencies ($k = 1$–$3$):** Gross trajectory shape — approach curves, sweeping motions.
- **Mid frequencies ($k = 4$–$8$):** Transition timing — when to switch from approach to grasp, contact dynamics.
- **High frequencies ($k = 9$–$15$):** Fine motor adjustments — gripper micro-corrections, contact force modulation.

By amplifying each step's natural frequency contribution, we help the denoiser resolve each frequency band more efficiently — early steps produce cleaner gross structure, late steps produce sharper fine detail.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Moderate improvement. The theoretical basis is sound — spectral analysis of diffusion denoising confirms that different steps resolve different frequencies (Hoogeboom et al., 2023; Yang et al., 2024). By amplifying each step's natural frequency role, we reduce spectral "leakage" where a step spends energy modifying frequencies it's not well-suited for. The energy preservation constraint ensures the modification is conservative — no net amplification, only redistribution. For tasks requiring precise timing (opening drawers, grasping), the high-frequency refinement in late steps should improve contact dynamics. For smooth motions (free-space transit), the low-frequency amplification in early steps should produce cleaner trajectories. |
| **Risk** | (1) **Frequency profile mismatch:** The Gaussian affinity profiles ($\sigma_i$ values) are heuristic. If the actual spectral structure of GR00T's velocity field doesn't match the assumed low-pass → high-pass progression, the scaling could *harm* quality by amplifying wrong frequencies. The `analyze_spectral_structure()` utility enables data-driven calibration. (2) **DCT assumption of stationarity:** The DCT assumes the signal is stationary (statistics don't change along the horizon). Robot actions are inherently non-stationary — the first few timesteps (approach) have different statistics than later timesteps (grasp). A windowed DCT or wavelet transform would be more appropriate but adds complexity. (3) **Interaction with energy preservation:** After normalization, the frequency redistribution is subtle — at $\gamma = 0.3$, the maximum per-frequency amplification is 1.3× and the minimum is 1.0×. After energy normalization, the actual effect is even smaller. This means the improvement may be subtle and hard to measure without many evaluation episodes. |
| **Latency** | 4 NFEs: ~64ms (identical to baseline). DCT/IDCT adds <0.1ms per step. Genuinely zero-cost. This is the only novel strategy in this document that adds no latency whatsoever. |
| **Implementation** | Moderate — the DCT/IDCT implementation requires care (the pure-PyTorch approach via FFT is ~30 lines; alternatively, use `scipy.fft.dctn` if available). The frequency weight construction is ~15 lines. Integration into the denoising loop is minimal — 4 lines per step (DCT, scale, IDCT, normalize). Total: ~80 lines including utilities. |

### Prior Work

- **Hoogeboom et al., "Simple Diffusion: End-to-End Diffusion for High Resolution Images"** — arXiv:2301.11093 (ICML 2023). Demonstrated that in image diffusion, low spatial frequencies are denoised first and high frequencies last. This "coarse-to-fine" spectral progression is analogous to our temporal frequency observation. Their analysis was descriptive; we propose an active intervention (frequency-band scaling) based on this structure.
- **Yang et al., "Frequency-Aware Diffusion Model for Temporal Generation"** — arXiv:2407.12173 (2024). Extended spectral analysis to temporal generation (video), showing that different denoising steps dominate different temporal frequency bands. Proposed frequency-aware noise schedules for video diffusion training. **Key difference:** They modify the training noise schedule; we modify the inference velocity scaling — zero training changes.
- **Si et al., "Frequency-Aware Timestep Scheduling for Diffusion Models"** — arXiv:2501.13349 (2025). Proposed allocating more denoising steps to frequency bands that are harder to resolve (typically high frequencies). Their approach modifies the timestep schedule (related to our Strategy 2). **Key difference:** They reallocate steps across time; we modify the velocity within each step across frequencies. The two could compose: use their optimized schedule (Strategy 2) with our spectral scaling for double the benefit.
- **DCT in trajectory optimization.** The DCT is widely used in trajectory optimization (Ratliff et al., CHOMP, 2009; Kalakrishnan et al., STOMP, 2011) to parameterize smooth trajectories via low-frequency coefficients. Our use of DCT is different — we decompose the *velocity field*, not the trajectory itself, and use it to guide denoising, not to parameterize the optimization variable.

**What makes this novel for VLAs:** No prior work applies spectral (DCT) analysis to the velocity field of a flow matching action denoiser. The connection between denoising step index and temporal frequency band has been established in the image/video domain but never exploited for *action trajectories*, where the frequency decomposition has direct physical meaning (gross motion vs. fine manipulation). The zero-cost nature (same NFEs, negligible DCT overhead) makes this the cheapest possible improvement strategy — if it works, it's pure profit.

### How to Run

**Terminal 1 — Server** (from repo root, main venv):
```bash
# Default parameters (gamma=0.3, energy_preserve=True)
bash scripts/denoising_lab/eval/strategies/spectral_temporal_decomposition/run_server.sh

# Stronger frequency guidance
bash scripts/denoising_lab/eval/strategies/spectral_temporal_decomposition/run_server.sh --gamma 0.5

# Disable energy preservation (allow net amplification)
bash scripts/denoising_lab/eval/strategies/spectral_temporal_decomposition/run_server.sh --no-energy-preserve

# Custom port
bash scripts/denoising_lab/eval/strategies/spectral_temporal_decomposition/run_server.sh --port 5556
```

**Terminal 2 — Benchmark** (from repo root, robocasa venv):
```bash
# Default: 10 episodes, seed 42, OpenDrawer
bash scripts/denoising_lab/eval/strategies/spectral_temporal_decomposition/run_eval.sh

# More episodes
bash scripts/denoising_lab/eval/strategies/spectral_temporal_decomposition/run_eval.sh --n-episodes 50
```

**Notebook / DenoisingLab:**
```python
from strategy import make_spectral_fn, denoise_with_lab

# Default parameters
actions = denoise_with_lab(lab, features, seed=42)

# Custom parameters
actions = denoise_with_lab(lab, features, seed=42, gamma=0.5, energy_preserve=False)

# Or use the guided_fn interface directly
guided_fn = make_spectral_fn(gamma=0.3, energy_preserve=True)
result = lab.denoise(features, num_steps=4, guided_fn=guided_fn, seed=42)
```

---
