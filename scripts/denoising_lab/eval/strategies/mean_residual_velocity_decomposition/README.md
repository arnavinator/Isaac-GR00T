## Strategy: Mean-Residual Velocity Decomposition

**Category:** Novel, drop-in | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

Spectral analysis of GR00T's velocity field during 4-step Euler denoising (`spectral_temporal_decomposition/analyze_spectral_structure.py`, 160 runs over 32 observations) reveals that the velocity evolves in a structurally asymmetric way across denoising steps:

| Component | Step 0 | Step 1 | Step 2 | Step 3 | Trend |
|-----------|--------|--------|--------|--------|-------|
| DC energy (k=0) | 1.72 | 1.92 | 2.25 | 2.74 | **+59% growth** |
| k=1 energy | 0.23 | 0.26 | 0.32 | 0.41 | +78% growth |
| k=2 energy | 0.19 | 0.25 | 0.33 | 0.43 | +126% growth |
| Bins k>=3 (each) | ~0.12 | ~0.12 | ~0.12 | ~0.12 | Flat (noise floor) |
| Total energy | 8.09 | 8.33 | 8.70 | 9.18 | +13.5% |
| Spectral centroid | 19.0 | 18.3 | 17.1 | 15.7 | Decreasing |

**The key finding:** The velocity field has two structurally distinct components that evolve differently:

1. **Mean component** (DC / uniform correction): The horizon-averaged velocity grows 59% from step 0 to step 3. This represents the model's increasing confidence about *where* the trajectory should be — an overall translation of the action chunk.

2. **Residual component** (structured / position-dependent correction): The deviation from the mean stays approximately constant. This represents the trajectory's *temporal structure* — when to accelerate, decelerate, open/close gripper.

Yet standard Euler integration treats both identically: `a += dt * v`. This strategy decomposes the velocity and applies different scaling to each component, counteracting the growing DC dominance that may under-resolve trajectory detail at later steps.

**Why this is novel:** All existing ODE solvers (Euler, Heun, RK4, AB2) and velocity-modification strategies (CFG, spectral scaling, horizon weighting) operate on the full velocity vector. No prior work decomposes the velocity into mean and residual components and integrates them with different effective step sizes. The decomposition is motivated by empirical spectral evidence specific to VLA action denoising — the DC/residual asymmetry doesn't arise in image or video generation where the spectral evolution follows a different pattern (low-freq→high-freq, per Hoogeboom et al. 2023).

**How this differs from existing strategies:**
- **Horizon-prioritized denoising (Strategy 9):** Applies *position-dependent* weights along the horizon. This strategy applies *component-dependent* weights on the velocity's mean vs. residual structure. The two are orthogonal and could compose.
- **Spectral temporal decomposition (Strategy 15):** Applies frequency-band scaling via DCT with heuristic Gaussian profiles. This strategy uses a single, empirically grounded decomposition (mean vs. residual) that requires no DCT and no frequency-profile assumptions.
- **AB2 / RK4 (Strategies 1, 3):** Change the ODE solver. This strategy modifies the velocity before integration — compatible with any solver.

### Mathematical Formulation

At denoising step $i$ with velocity $v_i \in \mathbb{R}^{B \times H \times D}$:

**Decomposition:**

$$\bar{v}_i = \frac{1}{H} \sum_{j=0}^{H-1} v_i[:, j, :] \quad \in \mathbb{R}^{B \times 1 \times D}$$

$$\tilde{v}_i = v_i - \bar{v}_i \quad \in \mathbb{R}^{B \times H \times D}$$

where $\bar{v}_i$ is the mean (uniform) component and $\tilde{v}_i$ is the residual (structured) component.

**Modified velocity:**

$$v_i^* = \begin{cases} v_i & \text{if } i < i_{\text{onset}} \\ \bar{v}_i + \rho \cdot \tilde{v}_i & \text{if } i \geq i_{\text{onset}} \end{cases}$$

where $\rho \in \mathbb{R}^+$ is the residual scaling factor and $i_{\text{onset}}$ is the step at which decomposition begins.

**Energy preservation (optional):**

$$v_i^* \leftarrow v_i^* \cdot \frac{\|v_i\|_F}{\|v_i^*\|_F}$$

This ensures the scaling only *redistributes* velocity between mean and residual without changing the total magnitude.

**Euler update:**

$$a^{\tau_{i+1}} = a^{\tau_i} + \Delta\tau \cdot v_i^*$$

**Note on orthogonality:** The mean and residual are orthogonal in the horizon dimension ($\langle \bar{v}_i, \tilde{v}_i \rangle = 0$), so:

$$\|v_i^*\|^2 = \|\bar{v}_i\|^2 + \rho^2 \|\tilde{v}_i\|^2$$

This means $\rho > 1$ increases the velocity magnitude and $\rho < 1$ decreases it, unless energy preservation is applied.

### Pseudocode

```python
def denoise_mean_residual(
    a_noise, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab,
    rho=1.15,           # residual scaling (1.0 = baseline)
    onset=2,            # first step to apply decomposition
    energy_preserve=True,
):
    """4-step Euler with mean-residual velocity decomposition.

    Zero extra NFEs. Same cost as baseline.
    """
    a = a_noise
    dt = 0.25

    for step_idx, tau_bucket in enumerate([0, 250, 500, 750]):
        v = lab._forward_dit(
            a, tau_bucket, vl_embeds, state_embeds,
            embodiment_id, backbone_output,
        )

        if step_idx >= onset:
            # Decompose into mean and residual
            v_mean = v.mean(dim=1, keepdim=True)  # (B, 1, D)
            v_res = v - v_mean                      # (B, H, D)

            # Scale residual component
            v_modified = v_mean + rho * v_res

            if energy_preserve:
                v_norm = v.norm()
                vm_norm = v_modified.norm()
                if vm_norm > 1e-8:
                    v_modified = v_modified * (v_norm / vm_norm)

            v = v_modified

        a = a + dt * v

    return a
```

### How It Replaces Action Chunking

Completely transparent. The decomposition modifies only the velocity used in each Euler step. The output shape, decoding, normalization, and `MultiStepWrapper` execution are all identical to baseline. The mean computation is $O(BHD)$ — negligible compared to the DiT forward pass (~16ms).

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | The strategy tests a specific hypothesis about DC dominance. If the model over-produces uniform corrections at later steps (rho > 1 helps), this means trajectory structure is being under-resolved and the strategy directly fixes it. If the model's DC growth is correctly calibrated (rho = 1 optimal), the strategy provides a useful null result. The energy preservation constraint ensures modifications are conservative. |
| **Risk** | Low. (1) At rho=1.0, the strategy is exactly baseline — no risk of regression at the default. (2) The mean-residual decomposition is exact and invertible — no information loss. (3) Energy preservation prevents magnitude changes that could disrupt the ODE. (4) The onset parameter limits modification to later steps where empirical evidence is strongest. Main risk: the modification is too subtle to produce measurable effects in the 10-50 episode evaluation range. |
| **Latency** | 4 NFEs: ~64ms (identical to baseline). Mean computation adds <0.01ms per step. |
| **Implementation** | Trivial — ~10 lines of core logic. No external dependencies. No DCT. No learned parameters. |

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `rho` | 1.15 | 0.5 - 2.0 | Residual scaling factor. 1.0 = baseline. >1.0 boosts structural detail. <1.0 boosts uniformity. |
| `onset` | 2 | 0 - 3 | First denoising step to apply decomposition. 2 = apply only at steps 2-3 (where DC dominance is strongest). |
| `energy_preserve` | True | bool | Normalize modified velocity to preserve total magnitude. |

**Suggested calibration sweep:**
- `rho` in {0.8, 0.9, 1.0, 1.1, 1.15, 1.2, 1.3}
- `onset` in {1, 2}
- `energy_preserve`: True (always, for safety)

### Prior Work

- **Hoogeboom et al., "Simple Diffusion" (ICML 2023):** Showed that image diffusion resolves low spatial frequencies first, high frequencies last. Our spectral analysis of GR00T shows the *opposite* pattern for action trajectories — later steps are *more* low-frequency dominated. This divergence motivates a different intervention.
- **Yang et al., "Frequency-Aware Diffusion for Temporal Generation" (2024):** Extended spectral analysis to video. The frequency progression in video diffusion doesn't match what we observe in action denoising either — confirming that action trajectories are a distinct modality requiring dedicated analysis.
- **Lu et al., "DPM-Solver++" (NeurIPS 2022):** Multistep solver that reuses cached velocities. Operates on the full velocity vector. Our decomposition is complementary — it could be applied within a DPM-Solver step.

**What makes this novel:** The mean-residual velocity decomposition is, to our knowledge, the first strategy that exploits the structural asymmetry between uniform and structured velocity components in flow matching denoising. The decomposition is motivated by empirical spectral analysis specific to VLA action generation — a domain where the velocity's spectral evolution is qualitatively different from image/video generation.

### How to Run

**Terminal 1 — Server** (from repo root, main venv):
```bash
# Default parameters (rho=1.15, onset=2, energy_preserve=True)
bash scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/run_server.sh

# Stronger residual boost
bash scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/run_server.sh --rho 1.3

# Dampen residual (test opposite direction)
bash scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/run_server.sh --rho 0.8

# Apply from step 1 onward
bash scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/run_server.sh --onset 1

# Custom port
bash scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/run_server.sh --port 5556
```

**Terminal 2 — Benchmark** (from repo root, robocasa venv):
```bash
# Default: 15 episodes, seed 42, OpenDrawer
bash scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/run_eval.sh

# More episodes
bash scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/run_eval.sh --n-episodes 50
```

**Notebook / DenoisingLab:**
```python
from strategy import make_mean_residual_fn, denoise_with_lab

# Default parameters
actions = denoise_with_lab(lab, features, seed=42)

# Custom parameters
actions = denoise_with_lab(lab, features, seed=42, rho=1.3, onset=1)

# Or use the guided_fn interface directly
guided_fn = make_mean_residual_fn(rho=1.15, onset=2, energy_preserve=True)
result = lab.denoise(features, num_steps=4, guided_fn=guided_fn, seed=42)
```

### Hyperparameter Calibration

**Grid search** (`calibrate_lambdas.py`):

Loads the model once, starts a ZMQ server, and iterates over a grid of `(rho, onset, energy_preserve)` values. Each config re-patches the action head and launches the eval client subprocess. Results are ranked by combined success rate.

```bash
uv run python scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/calibrate_lambdas.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
                robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --max-episode-steps 400 480 \
    --n-episodes 15 --seed 42 \
    --rho 1.0 1.05 1.10 1.15 1.25 1.50 \
    --onset 1 2 3 \
    --energy-preserve True False \
    --output-dir ./calibration_results/mean_residual
```

This sweeps 36 configs (6 rho x 3 onset x 2 energy_preserve). `rho=1.0` serves as the baseline control. To focus on hard seeds only (faster, more discriminative):

```bash
uv run python scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/calibrate_lambdas.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
                robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --max-episode-steps 400 480 \
    --n-episodes 15 --seed 42 \
    --seeds-only 42 45 46 47 49 52 \
    --rho 1.0 1.05 1.10 1.15 1.25 1.50 \
    --onset 1 2 3 \
    --energy-preserve True False \
    --output-dir ./calibration_results/mean_residual_hard_seeds
```

---
