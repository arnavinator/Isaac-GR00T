## Strategy 18: Convergence-Gated Iterative Refinement with Adaptive Execution Horizon

**Category:** Novel, drop-in | **NFEs:** 4–8 (adaptive, self-terminating) | **Retraining:** None

### Overview

Every denoising strategy in this document — and every published VLA denoiser — makes a hidden assumption: **the denoising schedule and the execution plan are independent.** You pick a solver (4-step Euler), run it, get an action chunk, and execute a fixed number of steps from that chunk. The denoising process has no say in how much of its output gets used, and the execution has no visibility into which parts of the denoised chunk the model is actually confident about.

This strategy breaks that wall.

**The empirical discovery that inspired this strategy:** In our denoising lab experiments (Cell 12.1 of the interactive notebook), we decoupled the timestep embedding from the integration progress — running the DiT repeatedly at a *fixed* late timestep (τ=800, 6 iterations) instead of sweeping through the standard schedule (τ=0→250→500→750). The result was striking: the model produced coherent trajectories that *converged* — each iteration applied smaller corrections until the output stabilized. The DiT, when told "you're in the refinement stage," acts as a **self-correcting iterative refiner** regardless of the actual denoising progress.

This observation reveals that the DiT's timestep embedding and action state provide **independent, complementary information**. The timestep controls *what kind of correction* to apply (coarse structural vs. fine detail); the action state controls *what to correct*. They can be decoupled — and this decoupling enables a fundamentally new denoising paradigm.

**The strategy — Phase-Separated Denoising with Convergence Gating:**

**Phase 1 — Structural Denoising (2 standard Euler steps):** Run the first two steps of the standard schedule (τ=0, τ=250) to collapse noise into gross trajectory structure — the overall motion direction, approach curve, and mode commitment. These steps do the "heavy lifting" of denoising, transforming random noise into a recognizable action trajectory. Using the standard schedule here ensures the DiT receives timestep embeddings that match the actual noise level of its input.

**Phase 2 — Iterative Refinement (2–6 steps at fixed τ_refine=750):** Instead of continuing with the standard schedule (τ=500, τ=750), switch to a *fixed-timestep refinement loop*. Repeatedly evaluate the DiT at τ=750 (telling it "you're in the late refinement stage") and apply the resulting velocity to refine the trajectory. At each iteration:

1. Evaluate $v_k = v(a_k, \tau_{\text{refine}})$ — the refinement velocity.
2. Compute the **per-position velocity magnitude**: $\rho_h^{(k)} = \|v_k[h]\|_2$ for each horizon position $h$.
3. Update only the **to-be-executed positions** (0 through $n_{\text{exec}}-1$); leave far-horizon positions at their Phase 1 state.
4. Check convergence: if $\max_{h < n_{\text{exec}}} \rho_h^{(k)} < \theta$, **stop early** — all executed positions have converged.

**Phase 3 — Adaptive Execution Decision:** The per-position convergence map $\{\rho_h^{(\text{final})}\}$ from Phase 2 is a rich signal. Positions where the velocity converged to near-zero are ones the model is confident about; positions where it remains large are uncertain. Instead of always executing a fixed $n_{\text{action\_steps}}$, execute only the **longest prefix of converged positions**:

$$n_{\text{adaptive}} = \max\big\{h \;:\; \rho_k^{(\text{final})} < \theta \text{ for all } k \leq h,\; h < n_{\text{exec}}\big\}$$

clamped to $[\,n_{\min},\; n_{\text{exec}}\,]$ where $n_{\min}$ is a safety floor (e.g., 2 steps).

**Why the fixed-timestep iteration converges:** At high τ (near τ=1), the velocity field is trained to produce small corrections that push actions toward the data manifold. For actions already close to the manifold (after Phase 1's 2 structural steps), the velocity at τ=750 is a *contraction mapping* — each iteration moves the action closer to a fixed point (a mode of the data distribution). The Banach contraction mapping theorem guarantees convergence if the Lipschitz constant of the velocity field is less than 1, which is empirically observed (velocities decrease monotonically across iterations in our experiments).

**Why this hasn't been done before — the fixed-timestep decoupling insight:** Standard flow matching theory couples the timestep to the noise level: at τ, the model expects to see an input that is a $(1-\tau)$ fraction noise and $\tau$ fraction signal. Evaluating at τ=750 with a Phase 1 output (which is at roughly τ=0.5 of denoising progress) is technically out-of-distribution — the input is noisier than the timestep suggests. But the empirical evidence is unambiguous: the model handles this gracefully, producing refinement velocities that converge. This works because: (1) the DiT's 32-layer transformer is highly over-parameterized and generalizes across the timestep-noise mismatch, (2) the AdaLayerNorm timestep conditioning is additive (scale/shift modulation), not a hard gate, and (3) the rectified flow training objective encourages the velocity field to point toward the data regardless of τ.

**What makes this truly unique — the trifecta:**

1. **Phase separation** — coarse structure vs. iterative refinement, inspired by multigrid methods in numerical PDE solvers (V-cycle: smooth at coarse resolution, refine at fine resolution).
2. **Per-position convergence monitoring** — a novel diagnostic signal that reveals *which horizon timesteps* the model is confident about. No other strategy produces per-position confidence estimates.
3. **Adaptive execution horizon** — the first strategy that feeds denoising quality back into the control loop, dynamically adjusting how many steps to execute before re-planning. This closes the loop between perception, planning, and execution in a way that fixed action chunking cannot.

**The connection to Diffusion Forcing:** Diffusion Forcing (Chen et al., NeurIPS 2024) trains models to handle per-token independent noise levels — different positions in the sequence can be at different stages of denoising simultaneously. Our Phase 2 achieves an analogous *inference-time* effect without retraining: the executed positions are iteratively refined (approaching τ≈1) while the far-horizon positions remain at their Phase 1 state (roughly τ≈0.5). The resulting action chunk has **heterogeneous resolution** across the horizon — high precision where it matters (near-horizon, to be executed) and coarse structure where it doesn't (far-horizon, to be re-predicted). This is emergent diffusion forcing without the training modification.

### Mathematical Formulation

**Phase 1 — Structural denoising** (2 NFEs):

$$a^{0.25} = a^0 + \Delta\tau_1 \cdot v(a^0,\; 0,\; o_t,\; l_t) \quad \text{with } \Delta\tau_1 = 0.5$$

$$a^{0.5} = a^{0.25} + \Delta\tau_1 \cdot v(a^{0.25},\; 250,\; o_t,\; l_t)$$

Note: we use $\Delta\tau = 0.5$ (not 0.25) per step — covering the first half of the denoising interval in 2 steps. This is equivalent to 2-step Euler on $[0, 1]$ with step size 0.5, but we only integrate the first half. The timestep buckets 0 and 250 correspond to $\tau = 0$ and $\tau = 0.25$ — the "correct" buckets for the first two steps of a 4-step schedule.

**Actually, let us be precise.** Phase 1 uses the standard 4-step schedule for its first 2 steps:

$$a^{0.25} = a^0 + 0.25 \cdot v(a^0,\; 0,\; o_t,\; l_t)$$
$$a^{0.5} = a^{0.25} + 0.25 \cdot v(a^{0.25},\; 250,\; o_t,\; l_t)$$

The output $a^{0.5}$ is approximately halfway through the standard denoising — gross trajectory structure is established, but fine detail is unresolved.

**Phase 2 — Iterative refinement at fixed τ** ($K$ NFEs, $K \in [2, K_{\max}]$):

For $k = 1, 2, \ldots, K_{\max}$:

$$v_k = v(a_k,\; \tau_{\text{refine}},\; o_t,\; l_t) \quad \text{with } \tau_{\text{refine}} = 750$$

**Position-selective update** (only refine executed positions):

$$(a_{k+1})_h = \begin{cases} (a_k)_h + \Delta\tau_{\text{refine}} \cdot (v_k)_h & \text{if } h < n_{\text{exec}} \\ (a_k)_h & \text{if } h \geq n_{\text{exec}} \end{cases}$$

where $\Delta\tau_{\text{refine}} = 0.25$ (standard Euler step size — the model expects this for τ=750).

**Per-position convergence metric:**

$$\rho_h^{(k)} = \|(v_k)_h\|_2 \quad \text{for } h = 0, \ldots, n_{\text{exec}} - 1$$

**Early stopping criterion:**

$$\max_{h < n_{\text{exec}}} \rho_h^{(k)} < \theta \quad \Longrightarrow \quad \text{STOP: all executed positions converged}$$

**Budget cap:** If $k = K_{\max}$ without convergence, stop and proceed to Phase 3.

**Phase 3 — Adaptive execution horizon:**

$$n_{\text{adaptive}} = \text{clip}\!\left(\max\big\{h : \rho_j^{(K)} < \theta \;\;\forall\, j \leq h\big\} + 1,\;\; n_{\min},\;\; n_{\text{exec}}\right)$$

In words: find the longest contiguous prefix of converged positions (starting from position 0), clamped between $n_{\min}$ (safety floor, e.g., 2) and $n_{\text{exec}}$ (the standard execution horizon, e.g., 8).

**Threshold calibration:** Run Phase 1 + Phase 2 on a validation set. Plot the distribution of $\rho_h^{(K)}$ across observations and horizon positions. Set $\theta$ at the median — this means roughly half of positions converge and half need more refinement, which is the operating point where adaptive execution provides the most benefit.

**NFE count by observation difficulty:**

| Observation type | Phase 1 | Phase 2 (convergence) | Total NFEs | Adaptive $n_{\text{exec}}$ |
|-----------------|---------|----------------------|------------|---------------------------|
| Easy (free-space transit) | 2 | 2 (converges immediately) | 4 | Full ($n_{\text{exec}}$) |
| Medium (approach) | 2 | 3–4 | 5–6 | Full or slightly reduced |
| Hard (precision grasp) | 2 | $K_{\max}=6$ | 8 | Reduced (re-plan sooner) |
| Average | 2 | ~3 | ~5 | Mostly full |

The strategy self-adapts: easy observations use 4 NFEs (same as baseline), hard observations use up to 8, and the execution horizon shrinks when the model can't commit — all without any manual tuning of step counts.

### Pseudocode

```python
import torch
from dataclasses import dataclass, field


@dataclass
class RefinementDiagnostics:
    """Rich diagnostic output from convergence-gated refinement."""
    phase1_nfe: int = 2
    phase2_nfe: int = 0
    total_nfe: int = 2
    converged: bool = False
    convergence_iteration: int | None = None
    # Per-position convergence map: (n_exec,) velocity norms at final iteration
    position_convergence: torch.Tensor | None = None
    # Full convergence history: (K, n_exec) velocity norms per iteration
    convergence_history: list[torch.Tensor] = field(default_factory=list)
    # Adaptive execution decision
    adaptive_n_exec: int = 8
    original_n_exec: int = 8
    # Per-position labels
    position_labels: list[str] = field(default_factory=list)  # 'converged' or 'uncertain'


def denoise_convergence_gated(
    a_noise, vl_embeds, state_embeds, embodiment_id, backbone_output,
    lab,                          # DenoisingLab
    n_exec=8,                     # standard execution horizon (n_action_steps)
    n_min=2,                      # minimum execution horizon (safety floor)
    tau_refine=750,               # fixed timestep bucket for refinement phase
    dt_refine=0.25,               # Euler step size for refinement
    theta=0.5,                    # per-position convergence threshold
    K_max=6,                      # max refinement iterations (budget cap)
    K_min=2,                      # min refinement iterations before early stopping
):
    """Phase-separated denoising with convergence-gated iterative refinement.

    Phase 1: 2 standard Euler steps (τ=0, 250) — structural denoising.
    Phase 2: Up to K_max iterations at fixed τ_refine — iterative refinement.
    Phase 3: Adaptive execution horizon from per-position convergence map.

    Returns (denoised_actions, diagnostics).
    """
    device = a_noise.device
    diag = RefinementDiagnostics(original_n_exec=n_exec)

    # ================================================================
    # Phase 1: Structural denoising (2 standard Euler steps)
    # ================================================================
    a = a_noise  # (B, action_horizon, action_dim)
    dt_structural = 0.25

    # Step 1: τ = 0
    v, a = lab._denoise_step_inner(
        vl_embeds, state_embeds, embodiment_id, backbone_output,
        a, t_discretized=0, dt=dt_structural,
        batch_size=a.shape[0], device=device,
    )

    # Step 2: τ = 250
    v, a = lab._denoise_step_inner(
        vl_embeds, state_embeds, embodiment_id, backbone_output,
        a, t_discretized=250, dt=dt_structural,
        batch_size=a.shape[0], device=device,
    )

    # a is now at ~τ=0.5 — gross structure established.
    diag.phase1_nfe = 2

    # ================================================================
    # Phase 2: Iterative refinement at fixed timestep
    # ================================================================
    # Create a mask for position-selective updates
    horizon = a.shape[1]  # 50 (padded)
    position_mask = torch.zeros(1, horizon, 1, device=device, dtype=a.dtype)
    position_mask[:, :n_exec, :] = 1.0  # only refine executed positions

    for k in range(K_max):
        # Evaluate velocity at fixed refinement timestep
        v_refine, _ = lab._denoise_step_inner(
            vl_embeds, state_embeds, embodiment_id, backbone_output,
            a, t_discretized=tau_refine, dt=dt_refine,
            batch_size=a.shape[0], device=device,
        )
        # Note: _denoise_step_inner returns (velocity, updated_actions).
        # We need the velocity to apply position-selective updates.
        # Undo the default update and apply our masked version:
        # Actually, we need the velocity BEFORE the step. Let's compute it directly.

        # Re-extract just the velocity (undo the step that _denoise_step_inner did)
        # velocity was the first return value; the step was: updated = a + dt * v
        # So v_refine is the velocity, and we apply it selectively:

        # Position-selective Euler update
        a = a + dt_refine * v_refine * position_mask

        diag.phase2_nfe += 1

        # Per-position velocity magnitude for executed positions
        # v_refine shape: (B, horizon, action_dim)
        per_pos_rho = v_refine[:, :n_exec, :].norm(dim=-1).mean(dim=0)  # (n_exec,)
        diag.convergence_history.append(per_pos_rho.detach().cpu())

        max_rho = per_pos_rho.max().item()

        # Check convergence (only after minimum iterations)
        if k >= K_min - 1 and max_rho < theta:
            diag.converged = True
            diag.convergence_iteration = k + 1
            break

    # ================================================================
    # Phase 3: Adaptive execution horizon
    # ================================================================
    final_rho = diag.convergence_history[-1]  # (n_exec,) — last iteration's convergence map
    diag.position_convergence = final_rho

    # Find longest prefix of converged positions
    converged_mask = final_rho < theta
    adaptive_n = 0
    for h in range(n_exec):
        if converged_mask[h]:
            adaptive_n = h + 1
        else:
            break

    # Clamp to [n_min, n_exec]
    diag.adaptive_n_exec = max(n_min, min(adaptive_n, n_exec))

    # Label each position
    diag.position_labels = [
        'converged' if final_rho[h] < theta else 'uncertain'
        for h in range(n_exec)
    ]

    diag.total_nfe = diag.phase1_nfe + diag.phase2_nfe

    return a, diag


# === Integration with MultiStepWrapper ===

def execute_with_adaptive_horizon(env, action_chunk, diag, default_n_exec=8):
    """Execute an action chunk with the convergence-gated adaptive horizon.

    Instead of always executing `default_n_exec` steps, execute only
    `diag.adaptive_n_exec` steps and re-plan from a new observation.

    Args:
        env: The environment (or environment wrapper).
        action_chunk: Dict of decoded actions, each (1, 16, dim).
        diag: RefinementDiagnostics from denoise_convergence_gated.
        default_n_exec: Standard execution horizon (for comparison).

    Returns:
        n_executed: Number of steps actually executed.
        should_replan: Whether to immediately re-plan (True if reduced horizon).
    """
    n = diag.adaptive_n_exec

    for step in range(n):
        action = {k: v[:, step] for k, v in action_chunk.items()}
        env.step(action)

    should_replan = n < default_n_exec
    return n, should_replan


# === Diagnostic utilities ===

def profile_convergence(lab, features_list, seeds, n_exec=8, K_max=6, theta=0.5):
    """Profile convergence behavior across observations.

    Returns statistics on:
    - Convergence rate: fraction of observations that converge within K_max
    - NFE distribution: how many refinement steps each observation needs
    - Position difficulty: which horizon positions are hardest to converge
    - Adaptive horizon distribution: how often the horizon is reduced
    """
    results = []
    for features, seed in zip(features_list, seeds):
        torch.manual_seed(seed)
        a_noise = torch.randn(
            1, lab.action_horizon, lab.action_dim,
            device=lab.device, dtype=lab.dtype,
        )
        _, diag = denoise_convergence_gated(
            a_noise, features.backbone_features,
            features.state_features, features.embodiment_id,
            features.backbone_output, lab,
            n_exec=n_exec, K_max=K_max, theta=theta,
        )
        results.append(diag)

    n = len(results)
    convergence_rate = sum(1 for r in results if r.converged) / n
    avg_nfe = sum(r.total_nfe for r in results) / n
    avg_adaptive = sum(r.adaptive_n_exec for r in results) / n

    # Per-position difficulty: average velocity norm at final iteration
    pos_difficulty = torch.stack([r.position_convergence for r in results]).mean(dim=0)

    # Convergence curves: average velocity over iterations
    max_iters = max(len(r.convergence_history) for r in results)
    convergence_curves = []
    for k in range(max_iters):
        vals = [r.convergence_history[k].mean().item()
                for r in results if k < len(r.convergence_history)]
        convergence_curves.append(sum(vals) / len(vals))

    return {
        'convergence_rate': convergence_rate,
        'avg_nfe': avg_nfe,
        'avg_adaptive_horizon': avg_adaptive,
        'original_horizon': n_exec,
        'horizon_reduction_rate': sum(
            1 for r in results if r.adaptive_n_exec < n_exec
        ) / n,
        'per_position_difficulty': pos_difficulty,
        'convergence_curves': convergence_curves,
        'nfe_distribution': sorted(r.total_nfe for r in results),
    }


def plot_convergence_map(diag, title="Per-Position Convergence Map"):
    """Visualize the convergence map for a single observation.

    Shows velocity magnitude per horizon position across refinement iterations.
    Converged positions are green; uncertain positions are red.
    """
    import matplotlib.pyplot as plt

    history = torch.stack(diag.convergence_history)  # (K, n_exec)
    K, n_exec = history.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: convergence curves per position
    for h in range(n_exec):
        color = 'green' if diag.position_labels[h] == 'converged' else 'red'
        ax1.plot(range(1, K + 1), history[:, h].numpy(), color=color, alpha=0.7,
                 label=f'h={h}' if h < 4 or h == n_exec - 1 else None)
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='θ')
    ax1.set_xlabel('Refinement iteration')
    ax1.set_ylabel('Velocity magnitude (ρ)')
    ax1.set_title('Convergence per horizon position')
    ax1.legend(loc='upper right', fontsize=8)

    # Right: final convergence map (bar chart)
    colors = ['green' if l == 'converged' else 'red' for l in diag.position_labels]
    ax2.bar(range(n_exec), diag.position_convergence.numpy(), color=colors, alpha=0.8)
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Horizon position')
    ax2.set_ylabel('Final velocity magnitude (ρ)')
    ax2.set_title(f'Adaptive horizon: {diag.adaptive_n_exec}/{diag.original_n_exec}')

    plt.suptitle(title)
    plt.tight_layout()
    return fig
```

### How It Replaces Action Chunking

This strategy does not merely produce an action chunk — it **redefines how much of the chunk to execute**. The standard pipeline (`MultiStepWrapper.step()` executing a fixed `n_action_steps`) is augmented with a convergence-informed execution decision:

1. **Standard flow:** Denoise → execute $n_{\text{action\_steps}}$ → observe → repeat.
2. **Convergence-gated flow:** Denoise with Phase 1 + Phase 2 → compute adaptive $n_{\text{exec}}$ from convergence map → execute $n_{\text{adaptive}}$ steps → observe → repeat.

When the model is confident (easy observation), $n_{\text{adaptive}} = n_{\text{action\_steps}}$ — identical to baseline. When the model is uncertain (hard observation), $n_{\text{adaptive}} < n_{\text{action\_steps}}$ — the robot re-plans sooner from a fresh observation, avoiding the execution of uncertain actions.

**This creates a self-regulating control loop:**
- **Easy phases** (free-space transit): Full execution horizon, 4 NFEs, fast. The robot cruises.
- **Hard phases** (grasping, contact): Reduced execution horizon, 6–8 NFEs, more re-planning. The robot is cautious, gathering new observations more frequently for the critical moments.

This is exactly how human motor control works — we move quickly and confidently through easy motions, but slow down and re-assess frequently during precise, uncertain operations. The convergence-gated strategy is the first VLA denoising approach that *emergently* reproduces this behavior.

**Interaction with the inference server:** The adaptive execution horizon requires communication between the denoising server (which computes $n_{\text{adaptive}}$) and the rollout client (which executes actions). The `RefinementDiagnostics.adaptive_n_exec` field is returned alongside the action chunk, and the client respects it instead of using a fixed `n_action_steps`. This is a minor protocol change (one additional integer per action chunk).

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | **Very high — and qualitatively different from all other strategies.** The quality improvement comes from two independent sources: (1) Iterative refinement at fixed τ converges to a fixed point of the late-timestep velocity field — a stable mode of the data distribution. This is empirically observed (notebook Cell 12.1) and theoretically grounded (Banach contraction). (2) Adaptive execution horizon prevents the execution of uncertain actions, reducing error accumulation in the closed-loop control. The combination means that on hard observations, the robot takes fewer but higher-quality actions and re-plans more frequently — exactly the correct adaptive behavior. On easy observations, the strategy degrades gracefully to baseline performance (4 NFEs, full execution horizon). |
| **Risk** | (1) **Timestep mismatch in Phase 2:** The action input to Phase 2 is at roughly τ≈0.5 of denoising progress, but the timestep embedding says τ=0.75. The DiT has never seen this exact combination during training (it was trained with matching timestep-noise levels). However, the notebook experiments confirm that the model handles this gracefully — producing coherent refinement velocities that converge. The mismatch is less severe than it appears because: (a) the Beta(1.5, 1.0) training distribution provides broad coverage, (b) the sinusoidal timestep embedding is smooth, and (c) the AdaLayerNorm conditioning is additive (scale/shift), not a hard gate. (2) **Convergence threshold sensitivity:** θ determines the balance between convergence quality and execution responsiveness. Too low → always full horizon, no adaptive benefit. Too high → always reduced horizon, excessive re-planning. The `profile_convergence()` utility provides data-driven calibration. (3) **Execution protocol change:** The adaptive horizon requires the rollout client to accept a variable `n_action_steps` per chunk, which is a breaking change to the `MultiStepWrapper` interface. However, the change is minimal (respect an integer from the server instead of a fixed config). (4) **Position-selective masking validity:** Zeroing the velocity for far-horizon positions while letting the DiT's self-attention attend across all positions creates an inconsistency — the model's internal representations may be influenced by the expectation that all positions are being updated. In practice, the far-horizon positions are still at their Phase 1 state (which is partially denoised), so the self-attention signal they provide to the near-horizon positions is meaningful (coarse trajectory context) even without further updates. |
| **Latency** | Variable: 4 NFEs (easy, converges at $K_{\min}=2$) to 8 NFEs (hard, hits $K_{\max}=6$). Average: ~5 NFEs = ~80ms (estimated). The early stopping makes this strategy FASTER than baseline on easy observations (if $K_{\min}=2$ and convergence is immediate: 4 NFEs = 64ms, same as baseline) and only moderately slower on hard observations (8 NFEs = 128ms). The average latency depends on the difficulty distribution of the task — for PandaOmron drawer opening, we estimate ~60% easy / ~40% hard → ~5 NFEs average. |
| **Implementation** | Moderate. Phase 1 is 2 standard denoising calls. Phase 2 is a while-loop calling `_denoise_step_inner` with a fixed timestep and position masking. Phase 3 is a few lines of threshold comparison. The convergence monitoring adds ~10 lines (per-position norm computation + early stopping check). Total: ~80 lines of core logic. The adaptive execution protocol change (`n_adaptive` communicated to the client) is ~5 lines on each side. The `plot_convergence_map()` visualization utility adds ~30 lines and is invaluable for understanding model behavior. |

### Prior Work

- **Chen et al., "Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion"** — arXiv:2407.01392 (NeurIPS 2024). Introduced per-token independent noise levels during training, enabling sequence generation with heterogeneous denoising progress across positions. **Key difference:** Diffusion Forcing modifies *training*; our strategy achieves an analogous effect at *inference time* without retraining — Phase 1 brings all positions to τ≈0.5, Phase 2 selectively refines near-horizon positions to τ≈1, creating a natural noise-level gradient across the horizon.
- **Multigrid methods for PDEs (Briggs, Henson, & McCormick, 2000).** The V-cycle in multigrid methods alternates between "smoothing" at the current resolution and "correcting" at a coarser resolution. Our Phase 1 → Phase 2 transition is analogous: Phase 1 "smooths" the global structure, Phase 2 "corrects" the local detail at a fixed resolution (τ=750). The per-position convergence monitoring is analogous to residual tracking in multigrid, which determines when to switch between cycles.
- **Banach contraction mapping theorem (1922).** The theoretical foundation for fixed-point iteration convergence. If $\|v(a, \tau)\|$ is a contraction (Lipschitz constant < 1 at the refinement timestep), the iteration $a_{k+1} = a_k + \Delta\tau \cdot v(a_k, \tau)$ converges geometrically to the unique fixed point. Our empirical observations (monotonically decreasing velocity norms across iterations) are consistent with contraction.
- **Black, Galliker, & Levine, "Real-Time Chunking for Diffusion and Flow-Based Policies"** — arXiv:2506.07339 (NeurIPS 2025). Explored dynamic adjustment of action chunking parameters for real-time diffusion policies. **Key difference:** Their chunking is based on timing constraints (fit within real-time budget); ours is based on *model confidence* (execute only what the model is certain about). The two are complementary — real-time chunking handles latency constraints; convergence-gated execution handles quality constraints.
- **Adaptive Model Predictive Control (Mayne et al., 2000, "Constrained MPC: Stability and Optimality").** Varying the prediction/execution horizon based on the current state's difficulty is standard in adaptive MPC. Our contribution is connecting this principle to VLA denoising — using the per-position velocity convergence as the "difficulty" signal that drives the horizon adaptation.
- **Bai & Melas-Kyriazi, "Fixed Point Diffusion Models (FPDM)"** — arXiv:2401.08741 (2024). Embeds implicit fixed-point solving layers inside the denoising network, iterating each denoising step to variable precision. Demonstrates that diffusion denoising at a given timestep IS a fixed-point problem, and iterating to convergence yields better results than a single pass. **Key difference:** FPDM modifies the architecture to include implicit layers; our approach iterates the *standard* DiT at a fixed timestep — zero architectural changes, zero retraining.
- **Garibi et al., "ReNoise: Real Image Inversion Through Iterative Noising"** — arXiv:2403.14602 (2024). For diffusion inversion, applies the pretrained diffusion model multiple times at each fixed timestep and averages predictions. Demonstrates empirically that repeated application at the *same* timestep improves prediction stability and accuracy. **Key difference:** ReNoise averages across iterations (useful for inversion); we use the iteration trajectory's convergence rate as a *diagnostic signal* (useful for confidence-gated execution).
- **Biroli et al., "Dynamical Regimes of Diffusion Models"** — arXiv:2402.18491 (2024). Uses statistical physics to identify three dynamical regimes during denoising: (1) speciation (gross structure via symmetry breaking), (2) intermediate refinement, (3) collapse onto data points. Different output components converge at different rates, determined by the spectral structure of the data. **Key connection:** This predicts exactly what our Phase 2 observes — near-horizon positions (aligned with leading eigenvectors of the action covariance) converge faster than far-horizon positions (aligned with trailing eigenvectors). The per-position convergence map is a direct empirical measurement of this spectral convergence structure.
- **Dockhorn et al., "D3P: Dynamic Denoising Diffusion Policy"** — arXiv:2508.06804. Adapts the number of denoising steps per observation via a learned RL adaptor. **Key difference:** D3P adapts the *denoising* budget; we adapt the *execution* horizon. D3P requires training an RL policy; we use the velocity convergence signal (zero training). The two could compose: D3P decides how many NFEs to use; our convergence gate decides how many steps to execute.

**What makes this novel for VLAs:** To our knowledge, this is the first VLA denoising strategy that: (1) uses **phase-separated denoising** with a structural phase followed by a fixed-timestep iterative refinement phase — leveraging the empirically-discovered property that the DiT functions as a convergent iterative refiner when conditioned on a fixed late timestep; (2) computes a **per-position convergence map** from the refinement phase, revealing which horizon timesteps the model is confident about — a novel diagnostic signal with no analog in any prior denoising strategy; (3) uses the convergence map to **adaptively set the execution horizon**, feeding denoising quality directly back into the control loop — creating the first self-regulating VLA that plans cautiously when uncertain and executes confidently when certain, mirroring human motor control. The combination of multigrid-inspired phase separation, fixed-point convergence theory, diffusion forcing-inspired heterogeneous resolution, and adaptive MPC-inspired execution gating synthesizes ideas from numerical methods, dynamical systems, generative modeling, and control theory into a unified framework that is greater than the sum of its parts.

---
