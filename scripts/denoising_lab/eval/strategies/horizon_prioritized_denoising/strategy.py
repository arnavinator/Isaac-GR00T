"""Horizon-Prioritized Denoising for GR00T N1.6.

Applies position-dependent velocity scaling that creates a "denoising wave"
sweeping from near-horizon to far-horizon across the 4 denoising steps.

Near-horizon positions (executed next) receive larger velocity updates in early
steps, while far-horizon positions (discarded at re-query) receive larger
updates in later steps.  The total velocity integrated per position is
approximately preserved.

The gating function is a Gaussian attention window centered at c_i that sweeps
across the horizon:

    w_j^{(i)} = 1 + gamma * exp(-(j - c_i)^2 / (2 * sigma_w^2))

Same 4 NFEs, identical latency to baseline.

Usage (server):
    from strategy import patch_action_head
    patch_action_head(policy.model.action_head)

Usage (notebook with DenoisingLab):
    from strategy import make_horizon_prioritized_fn, denoise_with_lab
    actions = denoise_with_lab(lab, features, seed=42)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------


def build_horizon_weights(
    num_steps: int = 4,
    action_horizon: int = 50,
    effective_horizon: int = 16,
    gamma: float = 0.5,
    sigma_w: float = 3.0,
) -> torch.Tensor:
    """Build per-step position-dependent gating weights.

    Args:
        num_steps: Number of denoising steps.
        action_horizon: Padded horizon length (50 for GR00T).
        effective_horizon: Actual action steps for the embodiment (16 for PandaOmron).
        gamma: Boost amplitude.  ``gamma=0`` recovers standard Euler.
        sigma_w: Width of the Gaussian attention window.

    Returns:
        ``(num_steps, action_horizon)`` tensor of gating weights.
    """
    # Sweep centers evenly across the effective horizon
    centers = [i * (effective_horizon - 1) / (num_steps - 1) for i in range(num_steps)]
    j = np.arange(action_horizon)

    weights = np.ones((num_steps, action_horizon), dtype=np.float32)
    for i in range(num_steps):
        gaussian = np.exp(-0.5 * ((j[:effective_horizon] - centers[i]) / sigma_w) ** 2)
        weights[i, :effective_horizon] = 1.0 + gamma * gaussian
        # Padded positions (effective_horizon..action_horizon-1) keep weight 1.0

    return torch.from_numpy(weights)


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


def denoise_horizon_prioritized(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *,
    weights: torch.Tensor,
    num_steps: int = 4,
    effective_horizon: int = 16,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """4-step Euler with horizon-prioritized velocity gating.

    At each step, velocity is element-wise multiplied by position-dependent
    weights before the Euler update.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` instance.
        vl_embeds: Vision-language embeddings ``(B, seq_len, 2048)``.
        state_features: Encoded state ``(B, state_horizon, hidden_dim)``.
        embodiment_id: Embodiment IDs ``(B,)``.
        backbone_output: Full backbone output.
        weights: Precomputed ``(num_steps, action_horizon)`` gating weights
            from ``build_horizon_weights``.
        num_steps: Number of denoising steps.
        effective_horizon: Actual action steps (for verbose logging only).
        initial_noise: Optional starting noise ``(B, action_horizon, action_dim)``.

    Returns:
        Denoised actions ``(B, action_horizon, action_dim)``.
    """
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    num_buckets = action_head.num_timestep_buckets
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

        # Position-dependent velocity gating: (B, H, D) * (1, H, 1)
        w = weights[step]  # (action_horizon,)
        actions = actions + dt * velocity * w[None, :, None]

        if action_head.verbose:
            w_eff = w[:effective_horizon]
            print(
                f"[HorizonPrioritized] Step {step}/{num_steps}  "
                f"tau={tau:.3f}  bucket={t_bucket}  "
                f"w_min={w_eff.min():.3f}  w_max={w_eff.max():.3f}  "
                f"a_norm={actions.float().norm():.4f}  "
                f"v_norm={velocity.float().norm():.4f}"
            )

    if action_head.verbose:
        af = actions.float()
        print(
            f"[HorizonPrioritized] Final  shape={tuple(actions.shape)}  "
            f"mean={af.mean():.4f}  std={af.std():.4f}"
        )

    return actions


# ---------------------------------------------------------------------------
# DenoisingLab guided_fn interface
# ---------------------------------------------------------------------------


def make_horizon_prioritized_fn(
    action_horizon: int = 50,
    effective_horizon: int = 16,
    gamma: float = 0.5,
    sigma_w: float = 3.0,
    num_steps: int = 4,
):
    """Factory for horizon-prioritized velocity gating.

    Returns a function compatible with ``DenoisingLab.denoise(guided_fn=...)``.
    """
    weights = build_horizon_weights(
        num_steps, action_horizon, effective_horizon, gamma, sigma_w,
    )

    def guided_fn(
        actions_before: torch.Tensor, step_idx: int, velocity: torch.Tensor,
    ) -> torch.Tensor:
        w = weights[step_idx].to(device=velocity.device, dtype=velocity.dtype)
        return velocity * w[None, :, None]

    return guided_fn


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(
    action_head,
    gamma: float = 0.5,
    sigma_w: float = 3.0,
    effective_horizon: int = 16,
):
    """Monkey-patch the action head to use horizon-prioritized denoising.

    Replaces ``get_action_with_features()`` in-place.  Weights are built once
    here and reused for every subsequent inference call.
    """
    num_steps = action_head.num_inference_timesteps
    weights = build_horizon_weights(
        num_steps, action_head.action_horizon, effective_horizon, gamma, sigma_w,
    ).to(device=action_head.device, dtype=action_head.dtype)

    @torch.no_grad()
    def hp_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions = denoise_horizon_prioritized(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output,
            weights=weights, num_steps=num_steps,
            effective_horizon=effective_horizon,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = hp_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------


def denoise_with_lab(
    lab, features, *, seed=None,
    gamma: float = 0.5, sigma_w: float = 3.0, effective_horizon: int = 16,
):
    """Run horizon-prioritized denoising via DenoisingLab.

    Uses the ``guided_fn`` interface so all intermediates are recorded.

    Returns:
        ``torch.Tensor`` -- raw actions ``(B, action_horizon, action_dim)``.
        Decode with ``lab.decode_raw_actions(actions)``.
    """
    guided_fn = make_horizon_prioritized_fn(
        action_horizon=lab.action_horizon,
        effective_horizon=effective_horizon,
        gamma=gamma, sigma_w=sigma_w,
        num_steps=lab.num_inference_timesteps,
    )
    result = lab.denoise(
        features, num_steps=lab.num_inference_timesteps,
        guided_fn=guided_fn, seed=seed,
    )
    return result.action_pred


# ---------------------------------------------------------------------------
# Offline calibration utility
# ---------------------------------------------------------------------------


def find_optimal_gamma(
    lab, features_list, *,
    gamma_candidates=None,
    sigma_w: float = 3.0,
    effective_horizon: int = 16,
    reference_steps: int = 64,
    seed: int = 0,
):
    """Search for the best gamma value against a high-fidelity reference.

    Generates a many-step Euler reference (approximating the true ODE solution),
    then evaluates each gamma candidate by L2 distance to that reference.  Run
    once on a handful of validation observations, then hard-code the winner.

    Args:
        lab: ``DenoisingLab`` instance.
        features_list: List of ``BackboneFeatures`` from different observations.
        gamma_candidates: Iterable of gamma values to try.  Defaults to
            ``[0.0, 0.1, 0.2, ..., 1.0]``.
        sigma_w: Gaussian window width (held fixed during search).
        effective_horizon: Actual action steps for the embodiment.
        reference_steps: Number of Euler steps for the high-fidelity reference.
        seed: Seed for reproducibility (noise generation).

    Returns:
        Tuple of ``(best_gamma, results)`` where ``results`` is a list of
        ``(gamma, mean_error)`` sorted by error ascending.
    """
    if gamma_candidates is None:
        gamma_candidates = [round(g * 0.1, 1) for g in range(11)]  # 0.0 .. 1.0

    # Compute high-fidelity references (many-step uniform Euler = no guidance)
    references = []
    for feat in features_list:
        ref = lab.denoise(feat, num_steps=reference_steps, seed=seed)
        references.append(ref.action_pred)

    results = []
    for gamma in gamma_candidates:
        errors = []
        for feat, ref in zip(features_list, references):
            candidate = denoise_with_lab(
                lab, feat, seed=seed,
                gamma=gamma, sigma_w=sigma_w,
                effective_horizon=effective_horizon,
            )
            errors.append((candidate - ref).float().norm().item())
        mean_err = float(np.mean(errors))
        results.append((gamma, mean_err))

    results.sort(key=lambda x: x[1])
    best_gamma = results[0][0]
    return best_gamma, results
