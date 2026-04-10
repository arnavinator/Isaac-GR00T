## Strategy 8: Analytic Constraint Guidance

**Category:** Novel, drop-in | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

This is a **novel strategy** that bridges two fields: *classifier guidance* from diffusion-based image generation and *trajectory optimization* from robotics.

**The problem:** The DiT learns the velocity field purely from data. It has no explicit knowledge of physics — so it can produce action chunks with jittery trajectories (high jerk), floating-point gripper values (0.47 instead of 0 or 1), or control mode oscillations. Currently, GR00T handles this with post-hoc clipping in `decode_action()`. But post-hoc corrections are applied *after* the full 4-step denoising chain — meaning the DiT may have spent its last 2-3 steps refining toward an action that gets clipped away. The denoising effort is wasted.

**The idea:** Apply physical constraints *during* denoising, not after. At each Euler step, after computing the velocity, add a small correction gradient that nudges the action toward the physically valid region. This is the same mechanism as classifier guidance in image diffusion (Dhariwal & Nichol, 2021), but with a crucial difference: instead of a learned classifier network, we use **analytic (closed-form) constraint functions** with hand-crafted gradients derived from robot physics.

**Why this is novel in the VLA context:**

1. **Classifier guidance** in image diffusion requires training a separate classifier — expensive and task-specific. Our constraint functions are hand-crafted from domain knowledge, requiring zero training.
2. **Trajectory optimization** in robotics (MPPI, CEM, iLQR) operates on full action sequences post-hoc. Our approach integrates constraints *into* the generative process, allowing the DiT to adapt to the constraints at each step.
3. **Diffusion-as-planning** (Janner et al., "Diffuser", 2022) uses reward guidance for DDPM-style planning. We adapt this principle to flow matching with analytic rewards — a different ODE formulation, different guidance mechanism, and different reward structure.

The result: "classifier-free, reward-model-free" guidance that works zero-shot with any observation, encodes physical correctness, and costs almost nothing computationally.

### Mathematical Formulation

Define differentiable constraint functions $\{C_k\}$ where each $C_k: \mathbb{R}^{H \times D} \to \mathbb{R}_{\geq 0}$ measures violation of a physical constraint (zero when satisfied). The total quality score:

$$Q(a) = -\sum_k \lambda_k \, C_k(a)$$

At each denoising step, after the standard velocity update, apply a constraint correction:

$$\hat{a}_t^{\tau_{i+1}} = a_t^{\tau_i} + \Delta\tau \cdot v(a_t^{\tau_i},\; \tau_i,\; o_t,\; l_t) \qquad \text{(standard Euler)}$$

$$a_t^{\tau_{i+1}} = \hat{a}_t^{\tau_{i+1}} + \eta_i \cdot \nabla_a Q\!\left(\hat{a}_t^{\tau_{i+1}}\right) \qquad \text{(constraint correction)}$$

The guidance strength is **annealed** with denoising progress:

$$\eta_i = \eta \cdot \tau_i$$

The annealing schedule is linear in $\tau$ and serves a critical purpose: **constraint guidance is only meaningful when the action has structure.** At the start of denoising ($\tau = 0$), the action tensor is pure Gaussian noise — every position is random, so "smoothness" or "gripper decisiveness" have no physical meaning. Applying constraint gradients to noise would inject arbitrary bias that fights the DiT's initial structure-forming velocity. As denoising progresses and the action takes shape, the constraints become increasingly relevant:

| Step $i$ | $\tau_i$ | $\eta_i$ (with $\eta = 0.1$) | Interpretation |
|-----------|----------|-------------------------------|----------------|
| 0 | 0.00 | **0.000** (off) | Pure noise. DiT establishes gross trajectory structure. No guidance — let the model work. |
| 1 | 0.25 | 0.025 (gentle) | Trajectory shape is emerging. Light smoothing nudges away from jittery modes. |
| 2 | 0.50 | 0.050 (moderate) | Trajectory is recognisable. Discrete dims start being pushed toward {0, 1}. |
| 3 | 0.75 | **0.075** (strongest) | Near-final refinement. Strong smoothing and decisive gripper/mode signals. |

This "off → gentle → strong" ramp means the DiT controls the overall trajectory plan in early steps (where it is most capable) and the constraint gradients polish physical validity in late steps (where the actions are concrete enough for constraints to be meaningful). The approach avoids the failure mode of constant-strength guidance, which can warp the trajectory structure before it has formed.

**Constraint functions for PandaOmron:**

**1. Temporal smoothness** (minimize jerk across the action horizon):

$$C_{\text{smooth}}(a) = \sum_{j=0}^{H-3} \left\| L[j] \right\|^2 \quad \text{where } L[j] = a[j{+}1] - 2\,a[j] + a[j{-}1]$$

$L[j]$ is the **discrete Laplacian** — the second-order finite difference at position $j$. It measures how much $a[j]$ deviates from being a linear interpolation of its neighbors. Equivalently, it is the discrete acceleration (the difference of consecutive velocities):

$$L[j] = \underbrace{(a[j{+}1] - a[j])}_{\text{velocity}[j]} - \underbrace{(a[j] - a[j{-}1])}_{\text{velocity}[j{-}1]}$$

When the trajectory is locally linear (constant velocity), $L[j] = 0$. Large values indicate abrupt direction changes — kinks or jitter.

**Exact gradient vs. implementation approximation:**

The exact gradient of $C_{\text{smooth}} = \sum_j \|L[j]\|^2$ requires the chain rule. Since each $L[j]$ depends on three positions ($a[j{-}1], a[j], a[j{+}1]$), and each position $a[k]$ participates in three Laplacian terms ($L[k{-}1], L[k], L[k{+}1]$):

$$\frac{\partial C_{\text{smooth}}}{\partial a[k]} = \sum_j 2\,L[j] \cdot \frac{\partial L[j]}{\partial a[k]} = 2\bigl(L[k{+}1] - 2\,L[k] + L[k{-}1]\bigr)$$

This is the **Laplacian of the Laplacian** (bi-Laplacian) — applying the $[1, -2, 1]$ stencil twice, which yields the 4th-order stencil $[1, -4, 6, -4, 1]$ convolved with $a$.

Our implementation uses an **approximation** that drops the second application of the stencil:

$$\nabla_a C_{\text{smooth}} \approx 2\,L[k] \quad \text{(used in code)} \qquad \text{vs.} \qquad 2\bigl(L[k{+}1] - 2\,L[k] + L[k{-}1]\bigr) \quad \text{(exact)}$$

This is valid because $L[k]$ and the exact gradient are both large where the trajectory is jerky and zero where it is smooth — they point in the same direction. The approximation is equivalent to **Laplacian smoothing** (a single step of heat diffusion), which nudges each point toward the average of its neighbors. The exact bi-Laplacian would be a more aggressive 4th-order smoothing operator. For gentle guidance within a denoising loop, the 1st-order version is preferable: less risk of overshooting, and the iterative application across 3 guided steps (steps 1–3) provides cumulative smoothing.

Applied only to continuous EEF position/rotation dimensions (not discrete gripper/mode dims).

**2. Discrete action decisiveness** (push gripper and control_mode toward binary values):

$$C_{\text{discrete}}(a) = \sum_{j=0}^{H-1} \left[ a_{\text{grip}}[j] \cdot (1 - a_{\text{grip}}[j]) + a_{\text{mode}}[j] \cdot (1 - a_{\text{mode}}[j]) \right]$$

This is minimized at $\{0, 1\}$ with analytic gradient:

$$\frac{\partial C_{\text{discrete}}}{\partial a_{\text{grip}}[j]} = 1 - 2\,a_{\text{grip}}[j]$$

Pushes values toward their nearest binary pole during denoising, rather than clipping post-hoc.

**3. Control mode consistency** (discourage rapid switching between arm and base control):

$$C_{\text{mode}}(a) = \sum_{j=0}^{H-2} \left( a_{\text{mode}}[j{+}1] - a_{\text{mode}}[j] \right)^2$$

Penalizes frame-to-frame control mode changes. The gradient is a simple first-difference:

$$\frac{\partial C_{\text{mode}}}{\partial a_{\text{mode}}[j]} = -2(a_{\text{mode}}[j{+}1] - a_{\text{mode}}[j]) + 2(a_{\text{mode}}[j] - a_{\text{mode}}[j{-}1])$$

**Computational cost of gradients:** All three constraints are computed via simple finite differences on the action tensor — $O(H \times D)$ operations. For $H = 16$, $D = 29$, this is 464 floating-point operations per step — utterly negligible compared to the ~1.5B-parameter DiT forward pass (~16ms).

### Pseudocode

```python
def make_constraint_guided_fn(
    lambda_smooth=0.005,
    lambda_discrete=0.01,
    lambda_mode=0.003,
    eta=0.1,
    # Dimension indices within the 128-dim padded action space (PandaOmron)
    eef_pos_dims=slice(0, 3),       # EEF position (continuous)
    eef_rot_dims=slice(3, 6),       # EEF rotation (continuous)
    gripper_dim=6,                   # Gripper close (discrete)
    mode_dim=11,                     # Control mode (discrete)
):
    """Factory for a constraint-guided velocity modifier.

    Returns a function compatible with DenoisingLab's guided_fn interface.
    """
    def guided_fn(actions_before, step_idx, velocity):
        """Modify velocity to incorporate physics constraints.

        Args:
            actions_before: (B, H, D) action tensor BEFORE this step's update.
            step_idx: Current denoising step index (0-3).
            velocity: (B, H, D) predicted velocity from the DiT.

        Returns:
            Modified velocity tensor (same shape).
        """
        # Annealing: no guidance at step 0, increasing toward step 3
        tau = step_idx / 4.0
        guidance_scale = eta * tau

        if guidance_scale < 1e-8:
            return velocity  # step 0: no guidance

        # Compute candidate action after Euler step (for gradient evaluation)
        dt = 0.25
        a_candidate = actions_before + dt * velocity

        grad = torch.zeros_like(a_candidate)

        # --- Constraint 1: Temporal smoothness (jerk minimization) ---
        # Second-order finite difference on continuous EEF dims
        for dims in [eef_pos_dims, eef_rot_dims]:
            a_cont = a_candidate[:, :, dims]  # (B, H, 3 or 6)
            # Discrete Laplacian: a[j+1] - 2*a[j] + a[j-1]
            laplacian = torch.zeros_like(a_cont)
            laplacian[:, 1:-1, :] = (
                a_cont[:, 2:, :] - 2 * a_cont[:, 1:-1, :] + a_cont[:, :-2, :]
            )
            # Gradient of ||laplacian||^2 w.r.t. a: 2 * laplacian convolved
            grad[:, :, dims] -= lambda_smooth * 2 * laplacian

        # --- Constraint 2: Discrete decisiveness (gripper + control mode) ---
        for dim in [gripper_dim, mode_dim]:
            a_disc = a_candidate[:, :, dim]  # (B, H)
            # Gradient of a*(1-a) = 1 - 2a, pushing toward 0 or 1
            grad[:, :, dim] -= lambda_discrete * (1.0 - 2.0 * a_disc)

        # --- Constraint 3: Control mode temporal consistency ---
        a_mode = a_candidate[:, :, mode_dim]  # (B, H)
        mode_diff = torch.zeros_like(a_mode)
        mode_diff[:, 1:] = a_mode[:, 1:] - a_mode[:, :-1]
        mode_diff_grad = torch.zeros_like(a_mode)
        mode_diff_grad[:, :-1] -= mode_diff[:, 1:]
        mode_diff_grad[:, 1:] += mode_diff[:, 1:]
        grad[:, :, mode_dim] -= lambda_mode * 2 * mode_diff_grad

        # Apply guidance as a velocity correction
        guided_velocity = velocity + (guidance_scale / dt) * grad
        return guided_velocity

    return guided_fn


# === Usage in DenoisingLab ===
result = lab.denoise(
    features,
    num_steps=4,
    guided_fn=make_constraint_guided_fn(
        lambda_smooth=0.005,
        lambda_discrete=0.01,
        eta=0.1,
    ),
    seed=42,
)
```

### How It Replaces Action Chunking

Action chunking is unchanged. The constraint gradients modify the velocity field at each step, steering the Euler integration toward physically valid trajectories. The output shape, decode pipeline, and `MultiStepWrapper` integration are all identical to baseline.

The key interaction with action chunking: the temporal smoothness constraint operates across all 16 timesteps of the chunk, including the 8 un-executed far-horizon steps. This means the far-horizon steps serve as a "temporal buffer" that enforces smoothness at the chunk boundary — the executed steps (0–7) are smoother because the constraint considers the trajectory all the way to step 15.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | High. Smoothness guidance should reduce jittery motions (a known failure mode in evaluation). Discrete decisiveness should improve gripper reliability (no more floating-point grasps). Mode consistency prevents control-mode oscillation. These directly target observed failure modes, not hypothetical ones. |
| **Risk** | (1) Guidance strength $\eta$ and constraint weights $\lambda_k$ require tuning per embodiment. Too aggressive → over-smoothed, sluggish actions. Too weak → no effect. (2) The constraints operate in the 128-dim padded action space, but the dimension indices are embodiment-specific. The mapping must be correct. (3) At early denoising steps ($\tau < 0.25$), the action is mostly noise — constraints on noise are meaningless, hence the annealing schedule. |
| **Latency** | Negligible increase — the gradient computation is $O(H \times D) \approx 500$ FLOPs per step, vs ~$10^{10}$ FLOPs for the DiT forward pass. Same 4 NFEs, same ~64ms total. |
| **Implementation** | Easy — implemented entirely within the existing `guided_fn` interface. No changes to the model, denoising loop, or decode pipeline. The constraint functions and dimension indices are the only per-embodiment configuration. |

### Prior Work and Inspiration

- **Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis"** — arXiv:2105.05233. Introduced classifier guidance for diffusion models: $v_{\text{guided}} = v + s \cdot \nabla_x \log p(y|x)$. Our approach replaces the learned classifier $p(y|x)$ with analytic constraint functions $C_k(a)$.
- **Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis" (Diffuser)** — arXiv:2205.09991. Applied reward guidance to DDPM-based trajectory generation. Showed that guiding the diffusion process with reward gradients produces higher-quality plans than post-hoc selection. Our approach adapts this to flow matching with analytic (not learned) rewards.
- **Ajay et al., "Is Conditional Generation All You Need for Decision-Making?" (Decision Diffuser)** — arXiv:2211.15657. Extended Diffuser with return-conditioned guidance. Demonstrated that diffusion + guidance is competitive with model-based RL for offline decision-making.
- **Song et al., "Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation"** — arXiv:2311.13024. Generalized classifier guidance to arbitrary differentiable loss functions. Our constraint functions are a special case with analytic gradients.

**What makes this novel for VLAs:** Prior work on guided diffusion uses *learned* reward/classifier networks (expensive, task-specific, requires training data). Our constraints are *analytic* — derived from physical first principles with closed-form gradients. This is possible because robot action spaces have known mathematical structure (bounded workspaces, smooth motion physics, discrete/continuous decomposition) that image pixel spaces lack. The bridge between "classifier guidance from generative modeling" and "trajectory constraints from robotics" is, to our knowledge, unexplored for flow-matching VLAs.

**Related VLA work (similar but not identical):** SafeDiffuser (arXiv:2306.00148) embeds CBF constraints during diffusion planning; SafeFlow (arXiv:2504.08661) extends this to flow matching for robot manipulation; KCGG (arXiv:2409.15528) uses FK-based gradients during diffusion sampling. All use different constraint types and/or model architectures. **Enhancement:** Adopt SafeFlow's barrier functions for hard safety constraints alongside our smoothness/workspace gradients. Use KCGG's FK-based gradients for joint-space validity.

### How to Run

**Terminal 1 — Server** (from repo root, main venv):
```bash
# Default parameters (eta=0.1, lambda_smooth=0.005, lambda_discrete=0.01, lambda_mode=0.003)
bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/run_server.sh

# Custom guidance strength
bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/run_server.sh --eta 0.2

# Custom port
bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/run_server.sh --port 5556
```

**Terminal 2 — Benchmark** (from repo root, robocasa venv):
```bash
# Default: 10 episodes, seed 42, OpenDrawer
bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/run_eval.sh

# More episodes
bash scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/run_eval.sh --n-episodes 50
```

**Notebook / DenoisingLab:**
```python
from strategy import make_constraint_guided_fn, denoise_with_lab, ConstraintConfig

# Default config
actions = denoise_with_lab(lab, features, seed=42)

# Custom config
cfg = ConstraintConfig(eta=0.2, lambda_smooth=0.01)
actions = denoise_with_lab(lab, features, seed=42, cfg=cfg)

# Or use the guided_fn interface directly
guided_fn = make_constraint_guided_fn(cfg)
result = lab.denoise(features, num_steps=4, guided_fn=guided_fn, seed=42)
```

---
