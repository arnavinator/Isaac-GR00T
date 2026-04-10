"""Analytic Constraint Guidance for GR00T N1.6.

Applies physics-based constraint gradients *during* denoising to improve action
quality.  Three analytic constraints with hand-crafted gradients:

1. **Temporal smoothness** -- jerk minimisation via second-order finite differences
   on continuous EEF dims (position + rotation).
2. **Discrete decisiveness** -- push gripper and control_mode toward binary {0,1}
   values during denoising instead of post-hoc clipping.
3. **Control-mode consistency** -- penalise frame-to-frame mode switching.

Guidance strength is annealed with denoising progress (zero at step 0, strongest
at the last step).  Same 4 NFEs, negligible extra compute (~500 FLOPs per step).

Usage (server):
    from strategy import patch_action_head
    patch_action_head(policy.model.action_head)

Usage (notebook with DenoisingLab):
    from strategy import make_constraint_guided_fn, denoise_with_lab
    actions = denoise_with_lab(lab, features, seed=42)
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
class ConstraintConfig:
    """Tunable constraint guidance parameters.

    Dimension indices are for the 128-dim padded action space and default to the
    PandaOmron layout.  Adjust for other embodiments.
    """

    # Constraint weights
    lambda_smooth: float = 0.005
    lambda_discrete: float = 0.01
    lambda_mode: float = 0.003

    # Overall guidance strength (annealed by tau)
    eta: float = 0.1

    # Denoising steps (must match the baseline)
    num_steps: int = 4

    # Dimension indices within the 128-dim padded action vector (PandaOmron)
    eef_pos_start: int = 0
    eef_pos_end: int = 3
    eef_rot_start: int = 3
    eef_rot_end: int = 6
    gripper_dim: int = 6
    mode_dim: int = 11


# ---------------------------------------------------------------------------
# Constraint gradient computation
# ---------------------------------------------------------------------------


def compute_constraint_gradient(
    actions: torch.Tensor, cfg: ConstraintConfig,
) -> torch.Tensor:
    """Compute the aggregate constraint gradient nabla Q(a).

    Q(a) = -sum_k lambda_k C_k(a), so the returned gradient points toward
    the constraint-satisfying region (lower total violation).

    Args:
        actions: ``(B, H, D)`` action tensor to evaluate constraints on.
        cfg: Constraint configuration.

    Returns:
        Gradient tensor of the same shape as *actions*.
    """
    grad = torch.zeros_like(actions)

    # ---- 1. Temporal smoothness (jerk minimisation) ----
    # Penalise second-order finite differences on continuous EEF dims.
    for start, end in [
        (cfg.eef_pos_start, cfg.eef_pos_end),
        (cfg.eef_rot_start, cfg.eef_rot_end),
    ]:
        a_cont = actions[:, :, start:end]  # (B, H, D_sub)
        # Discrete Laplacian: a[j+1] - 2*a[j] + a[j-1]
        # This is discrete acceleration — the difference of consecutive velocities.
        # Boundary positions (j=0, j=H-1) keep laplacian=0 since they have only one neighbor.
        laplacian = torch.zeros_like(a_cont)
        laplacian[:, 1:-1, :] = (
            a_cont[:, 2:, :] - 2 * a_cont[:, 1:-1, :] + a_cont[:, :-2, :]
        )
        # Negative gradient of C_smooth (pushes toward smoother trajectory)
        grad[:, :, start:end] -= cfg.lambda_smooth * 2.0 * laplacian

    # ---- 2. Discrete decisiveness (gripper + control mode) ----
    # C_discrete = a*(1-a); gradient = 1 - 2a; pushes toward {0, 1}.
    for dim in [cfg.gripper_dim, cfg.mode_dim]:
        a_disc = actions[:, :, dim]  # (B, H)
        grad[:, :, dim] -= cfg.lambda_discrete * (1.0 - 2.0 * a_disc)

    # ---- 3. Control-mode temporal consistency ----
    # Penalise frame-to-frame mode switching.
    a_mode = actions[:, :, cfg.mode_dim]  # (B, H)
    mode_diff = torch.zeros_like(a_mode)
    mode_diff[:, 1:] = a_mode[:, 1:] - a_mode[:, :-1]

    mode_grad = torch.zeros_like(a_mode)
    mode_grad[:, :-1] -= mode_diff[:, 1:]   # -diff[j+1]
    mode_grad[:, 1:] += mode_diff[:, 1:]    # +diff[j]
    grad[:, :, cfg.mode_dim] -= cfg.lambda_mode * 2.0 * mode_grad

    return grad


# ---------------------------------------------------------------------------
# Shared helper: evaluate the DiT velocity field
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
# Core denoising function
# ---------------------------------------------------------------------------


def denoise_with_guidance(action_head, vl_embeds, state_features, embodiment_id,
                          backbone_output, *, cfg=None, initial_noise=None):
    """4-step Euler with analytic constraint guidance applied after each step.

    Mathematical formulation (per step i):
        a_hat   = a + dt * v(a, tau_i)            (standard Euler)
        eta_i   = eta * tau_i                     (annealing)
        a_new   = a_hat + eta_i * grad_Q(a_hat)   (constraint correction)

    Args:
        action_head: ``Gr00tN1d6ActionHead`` instance.
        vl_embeds: Vision-language embeddings ``(B, seq_len, 2048)``.
        state_features: Encoded state ``(B, state_horizon, hidden_dim)``.
        embodiment_id: Embodiment IDs ``(B,)``.
        backbone_output: Full backbone output.
        cfg: Constraint configuration.  Defaults to PandaOmron settings.
        initial_noise: Optional starting noise ``(B, action_horizon, action_dim)``.

    Returns:
        Denoised actions ``(B, action_horizon, action_dim)``.
    """
    if cfg is None:
        cfg = ConstraintConfig()

    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    num_buckets = action_head.num_timestep_buckets
    num_steps = cfg.num_steps
    dt = 1.0 / num_steps

    if initial_noise is not None:
        actions = initial_noise.to(device=device, dtype=dtype)
    else:
        actions = torch.randn(
            batch_size, action_head.action_horizon, action_head.action_dim,
            dtype=dtype, device=device,
        )

    for step in range(num_steps):
        tau = step / float(num_steps)
        t_bucket = int(tau * num_buckets)

        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )

        # Standard Euler step
        actions = actions + dt * velocity

        # Constraint correction (annealed)
        guidance_scale = cfg.eta * tau
        if guidance_scale > 1e-8:
            grad = compute_constraint_gradient(actions, cfg)
            actions = actions + guidance_scale * grad

        if action_head.verbose:
            print(
                f"[ConstraintGuidance] Step {step}/{num_steps}  "
                f"tau={tau:.3f}  bucket={t_bucket}  "
                f"guidance_scale={guidance_scale:.4f}  "
                f"a_norm={actions.float().norm():.4f}  "
                f"v_norm={velocity.float().norm():.4f}"
            )

    if action_head.verbose:
        af = actions.float()
        print(
            f"[ConstraintGuidance] Final  shape={tuple(actions.shape)}  "
            f"mean={af.mean():.4f}  std={af.std():.4f}"
        )

    return actions


# ---------------------------------------------------------------------------
# DenoisingLab guided_fn interface
# ---------------------------------------------------------------------------


def make_constraint_guided_fn(cfg: ConstraintConfig | None = None):
    """Factory for a constraint-guided velocity modifier.

    Returns a function compatible with ``DenoisingLab.denoise(guided_fn=...)``.
    The DenoisingLab loop undoes the default Euler update and applies:
        ``a_new = a_before + dt * guided_fn(a_before, step_idx, velocity)``

    We return a modified velocity that achieves the same result as the two-step
    formulation (Euler + constraint correction).
    """
    if cfg is None:
        cfg = ConstraintConfig()

    def guided_fn(
        actions_before: torch.Tensor, step_idx: int, velocity: torch.Tensor,
    ) -> torch.Tensor:
        tau = step_idx / float(cfg.num_steps)
        guidance_scale = cfg.eta * tau

        if guidance_scale < 1e-8:
            return velocity  # step 0: no guidance

        dt = 1.0 / cfg.num_steps
        a_candidate = actions_before + dt * velocity
        grad = compute_constraint_gradient(a_candidate, cfg)
        # Fold the correction into the velocity so the Euler update produces
        # a_before + dt*velocity + guidance_scale*grad = a_candidate + guidance_scale*grad
        return velocity + (guidance_scale / dt) * grad

    return guided_fn


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg: ConstraintConfig | None = None):
    """Monkey-patch the action head to use constraint-guided denoising.

    Replaces ``get_action_with_features()`` in-place.
    """
    if cfg is None:
        cfg = ConstraintConfig()

    @torch.no_grad()
    def guided_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions = denoise_with_guidance(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, cfg=cfg,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = guided_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None):
    """Run constraint-guided denoising via DenoisingLab.

    Uses the ``guided_fn`` interface so all intermediates are recorded.

    Args:
        lab: ``DenoisingLab`` instance.
        features: ``BackboneFeatures`` from ``lab.encode_features()``.
        seed: Random seed for initial noise.
        cfg: Constraint configuration.

    Returns:
        ``torch.Tensor`` -- raw actions ``(B, action_horizon, action_dim)``.
        Decode with ``lab.decode_raw_actions(actions)``.
    """
    if cfg is None:
        cfg = ConstraintConfig()

    guided_fn = make_constraint_guided_fn(cfg)
    result = lab.denoise(
        features, num_steps=cfg.num_steps, guided_fn=guided_fn, seed=seed,
    )
    return result.action_pred
