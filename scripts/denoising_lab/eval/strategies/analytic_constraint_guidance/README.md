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

At $\tau = 0$ (pure noise), $\eta_0 = 0$ — no guidance, because constraints are meaningless on noise. At $\tau = 0.75$ (last step), $\eta_3 = 0.75\eta$ — strong guidance, because the action has taken shape and constraints are meaningful.

**Constraint functions for PandaOmron:**

**1. Temporal smoothness** (minimize jerk across the action horizon):

$$C_{\text{smooth}}(a) = \sum_{j=0}^{H-3} \left\| a[j{+}2] - 2\,a[j{+}1] + a[j] \right\|^2$$

This penalizes the second-order finite difference (discrete acceleration change). The gradient is a simple tridiagonal Laplacian — no neural network, no backpropagation:

$$\frac{\partial C_{\text{smooth}}}{\partial a[j]} = 2 \left( a[j{+}2] - 2a[j{+}1] + a[j] \right) - 4 \left( a[j{+}1] - 2a[j] + a[j{-}1] \right) + 2 \left( a[j] - 2a[j{-}1] + a[j{-}2] \right)$$

(boundary terms handled by clamping indices). Applied only to continuous EEF position/rotation dimensions.

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

---

---
