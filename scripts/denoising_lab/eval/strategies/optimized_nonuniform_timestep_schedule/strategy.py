"""Optimized Non-Uniform Timestep Schedule for GR00T N1.6.

Replaces the uniform timestep schedule {0.00, 0.25, 0.50, 0.75} with an
optimized non-uniform schedule that concentrates steps where the velocity
field changes fastest.  Same 4 NFEs, same Euler integrator — only the tau
positions change.

The default schedule is a starting hypothesis.  Use ``find_optimal_schedule``
to calibrate on a validation set (one-time offline, ~1 GPU-hour).

Usage (server):
    from strategy import patch_action_head
    patch_action_head(policy.model.action_head)

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab
    actions = denoise_with_lab(lab, features, seed=42)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature


# Default schedule — concentrate steps in the early (structure) and late
# (refinement) phases.  Replace with empirically optimized values.
DEFAULT_SCHEDULE: list[float] = [0.000, 0.080, 0.350, 0.820]


# ---------------------------------------------------------------------------
# Shared helper
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

def denoise_nonuniform(action_head, vl_embeds, state_features, embodiment_id,
                       backbone_output, *, schedule=None, initial_noise=None):
    """Euler integration with a non-uniform timestep schedule.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` instance.
        vl_embeds: Vision-language embeddings ``(B, seq_len, 2048)``.
        state_features: Encoded state ``(B, state_horizon, hidden_dim)``.
        embodiment_id: Embodiment IDs ``(B,)``.
        backbone_output: Full backbone output.
        schedule: List of tau values at which to evaluate (ascending, starting
            at 0.0).  Defaults to ``DEFAULT_SCHEDULE``.
        initial_noise: Optional starting noise ``(B, action_horizon, action_dim)``.

    Returns:
        Denoised actions ``(B, action_horizon, action_dim)``.
    """
    if schedule is None:
        schedule = DEFAULT_SCHEDULE

    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    num_buckets = action_head.num_timestep_buckets

    if initial_noise is not None:
        actions = initial_noise.to(device=device, dtype=dtype)
    else:
        actions = torch.randn(
            batch_size, action_head.action_horizon, action_head.action_dim,
            dtype=dtype, device=device,
        )

    tau_end = 1.0
    num_steps = len(schedule)

    for i, tau in enumerate(schedule):
        tau_next = schedule[i + 1] if i + 1 < num_steps else tau_end
        dt = tau_next - tau
        t_bucket = min(int(tau * num_buckets), num_buckets - 1)

        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )

        if action_head.verbose:
            print(
                f"[NonUniform] Step {i}/{num_steps}  tau={tau:.3f}  "
                f"dt={dt:.3f}  bucket={t_bucket}  "
                f"a_norm={actions.float().norm():.4f}  "
                f"v_norm={velocity.float().norm():.4f}"
            )

        actions = actions + dt * velocity

    if action_head.verbose:
        af = actions.float()
        print(
            f"[NonUniform] Final  shape={tuple(actions.shape)}  "
            f"mean={af.mean():.4f}  std={af.std():.4f}"
        )

    return actions


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------

def patch_action_head(action_head, schedule=None):
    """Monkey-patch the action head to use non-uniform timestep Euler.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` to patch in-place.
        schedule: Custom tau schedule (list of floats).  Defaults to
            ``DEFAULT_SCHEDULE``.
    """
    _schedule = list(schedule) if schedule is not None else None

    @torch.no_grad()
    def nonuniform_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions = denoise_nonuniform(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, schedule=_schedule,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = nonuniform_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------

def denoise_with_lab(lab, features, *, seed=None, schedule=None):
    """Run non-uniform schedule denoising using a DenoisingLab instance.

    Returns:
        ``torch.Tensor`` — raw actions ``(B, action_horizon, action_dim)``.
        Decode with ``lab.decode_raw_actions(actions)``.
    """
    device = lab.device
    dtype = lab.dtype

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    noise = torch.randn(
        features.backbone_features.shape[0],
        lab.action_horizon, lab.action_dim,
        dtype=dtype, device=device, generator=gen,
    )

    with torch.no_grad():
        return denoise_nonuniform(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            schedule=schedule, initial_noise=noise,
        )


# ---------------------------------------------------------------------------
# Offline calibration utility
# ---------------------------------------------------------------------------

def find_optimal_schedule(lab, features_list, *, n_candidates=1000,
                          reference_steps=64, seed=0):
    """Grid-search for the best 4-step non-uniform schedule.

    Run once on a set of validation observations, then hard-code the winning
    schedule into ``DEFAULT_SCHEDULE`` (or pass it to ``patch_action_head``).

    Args:
        lab: ``DenoisingLab`` instance.
        features_list: List of ``BackboneFeatures`` from different observations.
        n_candidates: Number of random schedules to try.
        reference_steps: Number of Euler steps for the high-fidelity reference.
        seed: Seed for reproducibility (noise + schedule sampling).

    Returns:
        Tuple of ``(best_schedule, best_error)`` where ``best_schedule`` is a
        list of 4 floats and ``best_error`` is the mean L2 deviation from the
        reference.
    """
    import numpy as np

    rng = np.random.RandomState(seed)

    # Compute high-fidelity reference for each observation
    references = []
    for feat in features_list:
        ref = lab.denoise(feat, num_steps=reference_steps, seed=seed)
        references.append(ref.action_pred)

    best_schedule = list(DEFAULT_SCHEDULE)
    best_error = float("inf")

    for _ in range(n_candidates):
        tau_1 = rng.uniform(0.02, 0.30)
        tau_2 = rng.uniform(tau_1 + 0.05, 0.60)
        tau_3 = rng.uniform(tau_2 + 0.05, 0.95)
        schedule = [0.0, float(tau_1), float(tau_2), float(tau_3)]

        errors = []
        for feat, ref in zip(features_list, references):
            candidate = denoise_with_lab(lab, feat, seed=seed, schedule=schedule)
            errors.append((candidate - ref).float().norm().item())

        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_error = mean_error
            best_schedule = schedule

    return best_schedule, best_error
