"""Curvature-Adaptive Step-Size Control for GR00T N1.6.

Replaces the fixed 4-step Euler loop with an adaptive Euler-Heun integrator
that automatically adjusts step sizes based on the velocity field's local
curvature.  The embedded Euler-Heun error estimate is free (no extra NFEs
beyond standard Heun).

With dt_grow_max=1.0 (default), step sizes never increase — the solver takes
4 Heun steps at dt=0.25, matching baseline's tau schedule while achieving
2nd-order accuracy.  Steps can still shrink when error is high, using
additional NFEs from the budget to refine difficult regions.

All accepted steps use 2nd-order Heun accuracy.

NFEs: 8-10 (adaptive).  Max latency bounded by max_nfe parameter.

Usage (server):
    from strategy import patch_action_head, AdaptiveConfig
    patch_action_head(policy.model.action_head, cfg=AdaptiveConfig(atol=0.05))

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab
    actions, step_log = denoise_with_lab(lab, features, seed=42)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveConfig:
    """Tunable parameters for curvature-adaptive step-size control."""

    atol: float = 0.05
    """Absolute error tolerance (in normalized action space)."""

    max_nfe: int = 10
    """Hard NFE budget.  Guarantees bounded worst-case latency."""

    dt_init: float = 0.25
    """Initial step size (conservative: matches baseline Euler)."""

    dt_min: float = 0.125
    """Minimum step size (prevents infinite subdivision)."""

    safety_factor: float = 0.9
    """Safety factor for step-size adaptation (Hairer-Wanner)."""

    dt_grow_max: float = 1.0
    """Maximum step-size growth factor per accepted step.  Capped at 1.0 to
    prevent the solver from skipping tau regions (e.g., tau=0.5) that baseline
    Euler always evaluates.  Set >1.0 to re-enable adaptive growth."""

    dt_shrink_min: float = 0.5
    """Minimum step-size shrink factor per rejected step."""


@dataclass
class StepLogEntry:
    """Recorded information for a single adaptive step."""

    outcome: str  # 'accepted', 'rejected', 'euler_forced', 'euler_cleanup'
    tau: float
    dt: float
    error: float | None
    nfe: int


# ---------------------------------------------------------------------------
# Shared helper: evaluate DiT velocity field
# ---------------------------------------------------------------------------


def _evaluate_velocity(action_head, actions, t_bucket, vl_embeds, state_features,
                       embodiment_id, backbone_output):
    """Forward pass through the DiT to get the predicted velocity."""
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device

    timesteps = torch.full((batch_size,), t_bucket, device=device)
    action_features = action_head.action_encoder(actions, timesteps, embodiment_id)

    if action_head.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

    sa_embs = torch.cat((state_features, action_features), dim=1)

    if action_head.config.use_alternate_vl_dit:
        model_output = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=timesteps,
            image_mask=backbone_output.image_mask,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
        )
    else:
        model_output = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=timesteps,
        )

    pred = action_head.action_decoder(model_output, embodiment_id)
    return pred[:, -action_head.action_horizon:]


# ---------------------------------------------------------------------------
# Core adaptive denoising function
# ---------------------------------------------------------------------------


def denoise_adaptive(action_head, vl_embeds, state_features, embodiment_id,
                     backbone_output, *, cfg=None, seed=None):
    """Adaptive Euler-Heun integration with embedded error estimation.

    At each step:
      1. Euler predictor: a_euler = a + dt * v1           (1 NFE)
      2. Heun corrector:  a_heun = a + dt/2 * (v1 + v2)  (1 NFE)
      3. Error estimate:  e = dt/2 * |v1 - v2|_inf        (free)
      4. Accept if e < atol, else reject and halve dt

    Args:
        action_head: Gr00tN1d6ActionHead instance.
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output.
        cfg: AdaptiveConfig.
        seed: Random seed for initial noise.

    Returns:
        (denoised_actions, step_log): Actions (B, H, D) and list of StepLogEntry.
    """
    if cfg is None:
        cfg = AdaptiveConfig()

    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    H = action_head.action_horizon
    D = action_head.action_dim
    num_buckets = action_head.num_timestep_buckets

    # Initial noise
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None
    actions = torch.randn(B, H, D, dtype=dtype, device=device, generator=gen)

    tau = 0.0
    nfe = 0
    dt = cfg.dt_init
    step_log: list[StepLogEntry] = []
    v1_cached = None  # Cache v1 across rejected steps (same a, same tau)

    while tau < 1.0 - 1e-6 and nfe < cfg.max_nfe:
        # Clamp step to not overshoot tau=1.0
        dt = min(dt, 1.0 - tau)
        tau_bucket = int(tau * num_buckets)
        tau_next_bucket = min(int((tau + dt) * num_buckets), num_buckets - 1)

        # --- Phase A: Euler predictor (1 NFE, or 0 if cached from rejection) ---
        if v1_cached is not None:
            v1 = v1_cached
            v1_cached = None
        else:
            v1 = _evaluate_velocity(
                action_head, actions, tau_bucket,
                vl_embeds, state_features, embodiment_id, backbone_output,
            )
            nfe += 1
        a_euler = actions + dt * v1

        if nfe >= cfg.max_nfe:
            # Budget exhausted -- accept Euler estimate
            actions = a_euler
            step_log.append(StepLogEntry(
                outcome='euler_forced', tau=tau, dt=dt, error=None, nfe=nfe,
            ))
            tau += dt

            if action_head.verbose:
                print(f"[Adaptive] FORCED Euler  tau={tau - dt:.3f}->{tau:.3f}  "
                      f"dt={dt:.4f}  nfe={nfe}")
            break

        # --- Phase B: Heun corrector (1 additional NFE) ---
        v2 = _evaluate_velocity(
            action_head, a_euler, tau_next_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )
        nfe += 1
        a_heun = actions + (dt / 2) * (v1 + v2)

        # --- Phase C: Error estimate (free -- no extra NFEs) ---
        error = (dt / 2) * torch.abs(v1.float() - v2.float()).max().item()

        if error < cfg.atol or dt <= cfg.dt_min:
            # Accept step -- use Heun estimate (free 2nd-order accuracy)
            actions = a_heun
            step_log.append(StepLogEntry(
                outcome='accepted', tau=tau, dt=dt, error=error, nfe=nfe,
            ))

            if action_head.verbose:
                print(f"[Adaptive] ACCEPT  tau={tau:.3f}->{tau + dt:.3f}  "
                      f"dt={dt:.4f}  error={error:.6f}  nfe={nfe}")
            tau += dt

            # Adapt step size for next step (Hairer-Wanner formula)
            if error > 1e-10:
                scale = cfg.safety_factor * (cfg.atol / error) ** 0.5
                dt = dt * min(cfg.dt_grow_max, max(cfg.dt_shrink_min, scale))
            else:
                dt = dt * cfg.dt_grow_max
            # Clamp: never exceed remaining integration distance
            dt = min(dt, 1.0 - tau)
        else:
            # Reject step -- halve step size, retry
            step_log.append(StepLogEntry(
                outcome='rejected', tau=tau, dt=dt, error=error, nfe=nfe,
            ))

            if action_head.verbose:
                print(f"[Adaptive] REJECT  tau={tau:.3f}  dt={dt:.4f}  "
                      f"error={error:.6f}  nfe={nfe}  halving dt")

            dt = max(dt / 2, cfg.dt_min)
            v1_cached = v1  # Cache for retry (same starting state)

    # Safety: ensure integration reaches tau=1.0
    if tau < 1.0 - 1e-6:
        dt_remaining = 1.0 - tau
        tau_bucket = int(tau * num_buckets)
        v_final = _evaluate_velocity(
            action_head, actions, tau_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )
        actions = actions + dt_remaining * v_final
        nfe += 1
        step_log.append(StepLogEntry(
            outcome='euler_cleanup', tau=tau, dt=dt_remaining, error=None, nfe=nfe,
        ))

        if action_head.verbose:
            print(f"[Adaptive] CLEANUP Euler  tau={tau:.3f}->{1.0:.3f}  "
                  f"dt={dt_remaining:.4f}  nfe={nfe}")

    if action_head.verbose:
        af = actions.float()
        n_accepted = sum(1 for s in step_log if s.outcome == 'accepted')
        n_rejected = sum(1 for s in step_log if s.outcome == 'rejected')
        print(f"[Adaptive] Final  shape={tuple(actions.shape)}  "
              f"mean={af.mean():.4f}  std={af.std():.4f}  "
              f"total_nfe={nfe}  accepted={n_accepted}  rejected={n_rejected}")

    return actions, step_log


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg=None):
    """Monkey-patch the action head to use adaptive step-size denoising.

    Replaces ``get_action_with_features()`` in-place.
    """
    if cfg is None:
        cfg = AdaptiveConfig()

    @torch.no_grad()
    def adaptive_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions, _log = denoise_adaptive(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, cfg=cfg,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = adaptive_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None):
    """Run adaptive Euler-Heun denoising via DenoisingLab.

    Args:
        lab: DenoisingLab instance.
        features: BackboneFeatures from lab.encode_features().
        seed: Random seed.
        cfg: AdaptiveConfig.

    Returns:
        (torch.Tensor, list[StepLogEntry]): Raw actions and step log.
        Decode actions with lab.decode_raw_actions(actions).
    """
    if cfg is None:
        cfg = AdaptiveConfig()

    with torch.no_grad():
        return denoise_adaptive(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            cfg=cfg, seed=seed,
        )
