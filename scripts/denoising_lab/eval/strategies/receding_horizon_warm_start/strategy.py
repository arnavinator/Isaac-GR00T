"""Receding-Horizon Warm-Start Denoising for GR00T N1.6.

Exploits temporal overlap between consecutive action chunks.  Instead of
starting every chunk from pure noise, the un-executed tail of the previous
chunk is shifted forward, partially re-noised via rectified-flow interpolation,
and used as the initial state for the new chunk.  This lets us skip the first
denoising step, reducing NFEs from 4 to 3 (25 % faster) while improving
temporal coherence.

First chunk of each episode uses standard 4-step Euler (no prior data).

Usage (server):
    from strategy import patch_action_head
    reset_fn = patch_action_head(policy.model.action_head)
    # Hook reset_fn into policy.reset() — see run_server.py

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab
    # First call (no prior):
    actions = denoise_with_lab(lab, features, seed=42)
    # Subsequent calls (warm start from previous prediction):
    actions = denoise_with_lab(lab, features, prev_actions=actions, seed=42)
"""

from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature


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
# Standard Euler fallback (for first chunk or cold start)
# ---------------------------------------------------------------------------

def _euler_denoise(action_head, vl_embeds, state_features, embodiment_id,
                   backbone_output, actions, num_steps=4):
    """Standard N-step Euler denoising (matches baseline)."""
    dt = 1.0 / num_steps
    num_buckets = action_head.num_timestep_buckets

    for step in range(num_steps):
        tau = step / float(num_steps)
        t_bucket = int(tau * num_buckets)

        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )

        if action_head.verbose:
            print(
                f"[WarmStart/Euler] Step {step}/{num_steps}  tau={tau:.3f}  "
                f"bucket={t_bucket}  "
                f"a_norm={actions.float().norm():.4f}  "
                f"v_norm={velocity.float().norm():.4f}"
            )

        actions = actions + dt * velocity

    return actions


# ---------------------------------------------------------------------------
# Core denoising function
# ---------------------------------------------------------------------------

def denoise_warm_start(action_head, vl_embeds, state_features, embodiment_id,
                       backbone_output, *, prev_actions=None, tau_start=0.25,
                       n_executed=8, num_steps=4, initial_noise=None):
    """Warm-start denoising from previous chunk's un-executed actions.

    When ``prev_actions`` is provided, shifts the un-executed tail forward,
    partially re-noises to level ``tau_start``, and denoises from there
    (skipping the first step).  When ``prev_actions`` is ``None``, falls back
    to standard Euler.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` instance.
        vl_embeds: Vision-language embeddings ``(B, seq_len, 2048)``.
        state_features: Encoded state ``(B, state_horizon, hidden_dim)``.
        embodiment_id: Embodiment IDs ``(B,)``.
        backbone_output: Full backbone output.
        prev_actions: Previous chunk's raw padded actions
            ``(B, action_horizon, action_dim)`` or ``None``.
        tau_start: Noise level to re-noise to (default 0.25 = skip 1 step).
        n_executed: Number of steps executed from the previous chunk (default 8).
        num_steps: Total number of Euler steps for the full schedule (default 4).
        initial_noise: Optional starting noise (used only when ``prev_actions``
            is ``None``).

    Returns:
        Denoised actions ``(B, action_horizon, action_dim)``.
    """
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    action_horizon = action_head.action_horizon
    action_dim = action_head.action_dim
    num_buckets = action_head.num_timestep_buckets
    dt = 1.0 / num_steps

    if prev_actions is None:
        # Cold start: standard Euler from noise
        if initial_noise is not None:
            actions = initial_noise.to(device=device, dtype=dtype)
        else:
            actions = torch.randn(
                batch_size, action_horizon, action_dim,
                dtype=dtype, device=device,
            )
        if action_head.verbose:
            print("[WarmStart] Cold start — using standard 4-step Euler")
        return _euler_denoise(
            action_head, vl_embeds, state_features, embodiment_id,
            backbone_output, actions, num_steps=num_steps,
        )

    # ---- Warm start from previous chunk ----
    prev = prev_actions.to(device=device, dtype=dtype)

    # Snap tau_start to the nearest step boundary so the re-noising level
    # matches the first DiT evaluation tau exactly.
    start_step = round(tau_start * num_steps)
    actual_tau_start = start_step / float(num_steps)

    if action_head.verbose:
        print(
            f"[WarmStart] Warm start  tau_start={tau_start}  "
            f"actual_tau_start={actual_tau_start}  "
            f"n_executed={n_executed}  prev_shape={tuple(prev.shape)}"
        )

    # 1. Shift: move un-executed tail (positions n_executed:) to the front
    remaining = action_horizon - n_executed
    warm = torch.randn(batch_size, action_horizon, action_dim,
                        dtype=dtype, device=device)
    warm[:, :remaining, :] = prev[:, n_executed:, :]

    # 2. Partial re-noising via rectified-flow interpolation
    #    a_tau = (1 - tau) * epsilon + tau * a_clean
    #    Only re-noise the shifted portion; the tail is fresh noise already.
    epsilon = torch.randn(batch_size, remaining, action_dim,
                          dtype=dtype, device=device)
    warm[:, :remaining, :] = (
        (1.0 - actual_tau_start) * epsilon
        + actual_tau_start * warm[:, :remaining, :]
    )

    # 3. Denoise from actual_tau_start (skip the first start_step steps)
    #    With num_steps=4 and tau_start=0.25 the remaining steps are at
    #    tau in {0.25, 0.50, 0.75}, each with dt=0.25.
    actions = warm

    for step in range(start_step, num_steps):
        tau = step / float(num_steps)
        t_bucket = int(tau * num_buckets)

        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )

        if action_head.verbose:
            print(
                f"[WarmStart] Step {step}/{num_steps}  tau={tau:.3f}  "
                f"bucket={t_bucket}  "
                f"a_norm={actions.float().norm():.4f}  "
                f"v_norm={velocity.float().norm():.4f}"
            )

        actions = actions + dt * velocity

    if action_head.verbose:
        af = actions.float()
        print(
            f"[WarmStart] Final  shape={tuple(actions.shape)}  "
            f"mean={af.mean():.4f}  std={af.std():.4f}"
        )

    return actions


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------

def patch_action_head(action_head, tau_start=0.25, n_executed=8):
    """Monkey-patch the action head for warm-start denoising.

    The patched action head caches the raw ``action_pred`` from each inference
    call and uses it to warm-start the next call.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` to patch in-place.
        tau_start: Noise level for re-noising (default 0.25 → 3 NFEs).
        n_executed: Steps executed from each chunk (default 8).

    Returns:
        A ``reset()`` callable that clears the cached state.  **Must** be
        hooked into the policy's ``reset()`` method — see ``run_server.py``.
    """
    action_head._warm_start_prev = None

    # Save original method for cold-start fallback
    _original_fn = action_head.get_action_with_features

    @torch.no_grad()
    def warm_start_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        prev = action_head._warm_start_prev

        if prev is not None:
            actions = denoise_warm_start(
                action_head, backbone_features, state_features,
                embodiment_id, backbone_output,
                prev_actions=prev, tau_start=tau_start,
                n_executed=n_executed,
            )
        else:
            # Cold start: delegate to original 4-step Euler
            result = _original_fn(
                backbone_features, state_features, embodiment_id, backbone_output,
            )
            actions = result["action_pred"]

        # Cache for next call
        action_head._warm_start_prev = actions.clone()

        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = warm_start_get_action_with_features

    def reset():
        """Clear warm-start cache (call on episode reset)."""
        action_head._warm_start_prev = None

    return reset


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------

def denoise_with_lab(lab, features, *, seed=None, prev_actions=None,
                     tau_start=0.25, n_executed=8):
    """Run warm-start denoising using a DenoisingLab instance.

    Args:
        lab: ``DenoisingLab`` instance.
        features: ``BackboneFeatures`` from ``lab.encode_features()``.
        seed: Random seed for initial noise (cold-start only).
        prev_actions: Previous chunk's raw padded actions, or ``None`` for
            cold start.  Typically the return value of a previous
            ``denoise_with_lab`` call.
        tau_start: Noise level for re-noising.
        n_executed: Steps executed from the previous chunk.

    Returns:
        ``torch.Tensor`` — raw actions ``(B, action_horizon, action_dim)``.
        Decode with ``lab.decode_raw_actions(actions)``.
    """
    device = lab.device
    dtype = lab.dtype

    noise = None
    if prev_actions is None:
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
        return denoise_warm_start(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            prev_actions=prev_actions, tau_start=tau_start,
            n_executed=n_executed, initial_noise=noise,
        )
