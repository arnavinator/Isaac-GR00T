"""Mean-Residual Velocity Decomposition.

Decomposes the velocity field at each denoising step into its horizon
mean (uniform correction) and residual (structured, position-dependent
correction), then scales the residual to counteract the empirically
observed DC dominance at later steps.

Spectral analysis of GR00T's velocity field shows:
  - DC (mean) energy grows 59% from step 0 to step 3
  - Residual (structured) energy stays approximately constant
  - Later steps predominantly "translate" the trajectory uniformly

By scaling the residual component (rho > 1.0), we boost trajectory
structure that the model's growing DC dominance may under-resolve.

Same 4 NFEs, <0.01 ms overhead per step (one mean + one multiply).

Usage (server):
    from strategy import patch_action_head
    patch_action_head(policy.model.action_head, rho=1.15)

Usage (notebook with DenoisingLab):
    from strategy import make_mean_residual_fn, denoise_with_lab
    actions = denoise_with_lab(lab, features, seed=42)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# Core: mean-residual velocity modification
# ---------------------------------------------------------------------------


def _modify_velocity(
    velocity: torch.Tensor,
    rho: float,
    energy_preserve: bool = True,
) -> torch.Tensor:
    """Decompose velocity into mean + residual and scale residual by rho.

    Args:
        velocity: ``(B, H, D)`` velocity from the DiT.
        rho: Residual scaling factor.  ``1.0`` = identity (baseline).
        energy_preserve: Normalise to preserve total velocity magnitude.

    Returns:
        Modified velocity, same shape and dtype as input.
    """
    orig_dtype = velocity.dtype
    v = velocity.float()

    # Decompose
    v_mean = v.mean(dim=1, keepdim=True)  # (B, 1, D)
    v_res = v - v_mean                     # (B, H, D)

    # Scale residual
    v_mod = v_mean + rho * v_res

    if energy_preserve:
        v_norm = v.norm()
        vm_norm = v_mod.norm()
        if vm_norm > 1e-8:
            v_mod = v_mod * (v_norm / vm_norm)

    return v_mod.to(orig_dtype)


# ---------------------------------------------------------------------------
# DiT velocity evaluation (shared helper)
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


def denoise_mean_residual(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *,
    rho: float = 1.15,
    onset: int = 2,
    energy_preserve: bool = True,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """4-step Euler with mean-residual velocity decomposition.

    At steps >= onset: velocity -> decompose -> scale residual -> Euler update.
    Zero extra NFEs.  Same cost as baseline.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` instance.
        vl_embeds: Vision-language embeddings ``(B, seq_len, 2048)``.
        state_features: Encoded state ``(B, state_horizon, hidden_dim)``.
        embodiment_id: Embodiment IDs ``(B,)``.
        backbone_output: Full backbone output.
        rho: Residual scaling factor.  ``1.0`` recovers baseline Euler.
        onset: First denoising step to apply the decomposition (0-3).
        energy_preserve: Normalise modified velocity to preserve magnitude.
        initial_noise: Optional starting noise ``(B, action_horizon, action_dim)``.

    Returns:
        Denoised actions ``(B, action_horizon, action_dim)``.
    """
    num_steps = 4
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    num_buckets = action_head.num_timestep_buckets
    dt = 1.0 / num_steps
    H = action_head.action_horizon

    if initial_noise is not None:
        actions = initial_noise.to(device=device, dtype=dtype)
    else:
        actions = torch.randn(
            batch_size, H, action_head.action_dim,
            dtype=dtype, device=device,
        )

    for step in range(num_steps):
        tau = step / float(num_steps)
        t_bucket = int(tau * num_buckets)

        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )

        if step >= onset:
            velocity = _modify_velocity(velocity, rho, energy_preserve)

        actions = actions + dt * velocity

        if action_head.verbose:
            vf = velocity.float()
            print(
                f"[MeanRes] Step {step}/{num_steps}  "
                f"tau={tau:.3f}  bucket={t_bucket}  "
                f"rho={'%.2f' % rho if step >= onset else 'n/a'}  "
                f"v_norm={vf.norm():.4f}  a_norm={actions.float().norm():.4f}"
            )

    if action_head.verbose:
        af = actions.float()
        print(
            f"[MeanRes] Final  shape={tuple(actions.shape)}  "
            f"mean={af.mean():.4f}  std={af.std():.4f}"
        )

    return actions


# ---------------------------------------------------------------------------
# DenoisingLab guided_fn interface
# ---------------------------------------------------------------------------


def make_mean_residual_fn(
    rho: float = 1.15,
    onset: int = 2,
    energy_preserve: bool = True,
):
    """Factory for mean-residual velocity modifier.

    Returns a function compatible with ``DenoisingLab.denoise(guided_fn=...)``.

    Args:
        rho: Residual scaling factor.  ``1.0`` recovers baseline.
        onset: First step to apply decomposition (0-3).
        energy_preserve: Normalise to preserve velocity magnitude.

    Returns:
        A ``guided_fn(actions_before, step_idx, velocity) -> velocity``.
    """

    def guided_fn(
        actions_before: torch.Tensor, step_idx: int, velocity: torch.Tensor,
    ) -> torch.Tensor:
        if step_idx < onset:
            return velocity
        return _modify_velocity(velocity, rho, energy_preserve)

    return guided_fn


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(
    action_head,
    rho: float = 1.15,
    onset: int = 2,
    energy_preserve: bool = True,
):
    """Monkey-patch the action head to use mean-residual denoising.

    Replaces ``get_action_with_features()`` in-place.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` to patch.
        rho: Residual scaling factor.
        onset: First denoising step to apply decomposition.
        energy_preserve: Normalise to preserve velocity magnitude.
    """

    @torch.no_grad()
    def mean_residual_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions = denoise_mean_residual(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output,
            rho=rho, onset=onset, energy_preserve=energy_preserve,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = mean_residual_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------


def denoise_with_lab(
    lab, features, *, seed=None,
    rho: float = 1.15, onset: int = 2, energy_preserve: bool = True,
):
    """Run mean-residual denoising via DenoisingLab.

    Uses the ``guided_fn`` interface so all intermediates are recorded.

    Args:
        lab: ``DenoisingLab`` instance (model loaded).
        features: Encoded features from ``lab.encode_features_from_sim_obs()``.
        seed: Random seed for reproducibility.
        rho: Residual scaling factor.  ``1.0`` = baseline.
        onset: First step to apply decomposition.
        energy_preserve: Normalise to preserve velocity magnitude.

    Returns:
        ``torch.Tensor`` -- raw actions ``(B, action_horizon, action_dim)``.
        Decode with ``lab.decode_raw_actions(actions)``.
    """
    guided_fn = make_mean_residual_fn(rho=rho, onset=onset, energy_preserve=energy_preserve)
    result = lab.denoise(
        features, num_steps=4, guided_fn=guided_fn, seed=seed,
    )
    return result.action_pred
