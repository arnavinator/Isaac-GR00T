"""Receding-Horizon Warm-Start Denoising for GR00T N1.6.

Exploits temporal overlap between consecutive action chunks.  Instead of
starting every chunk from pure noise, the un-executed tail of the previous
chunk is shifted forward and used to inform the initial state for the new
chunk.

Two warm-start modes:

  partial_denoise (default) — Re-noise the shifted actions via rectified-flow
      interpolation to tau_start, then denoise from there.  Skips the first
      step(s), reducing NFEs.  Default tau_start=0.5 preserves 50% of the
      warm signal and uses 2 NFEs (was 0.25 / 3 NFEs in v1).

  noise_bias — Keep all 4 NFEs.  Blend the shifted actions into the initial
      noise at strength beta, then renormalize to preserve Gaussian norm
      statistics.  The DiT sees near-standard noise at tau=0 but biased toward
      the warm trajectory.

First chunk of each episode uses standard 4-step Euler (no prior data).

Usage (server):
    from strategy import patch_action_head
    reset_fn = patch_action_head(policy.model.action_head, mode="partial_denoise")
    # Hook reset_fn into policy.reset() — see run_server.py

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab
    actions = denoise_with_lab(lab, features, seed=42)
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
                       backbone_output, *, prev_actions=None, tau_start=0.5,
                       n_executed=8, num_steps=4, initial_noise=None,
                       mode="partial_denoise", beta=0.15):
    """Warm-start denoising from previous chunk's un-executed actions.

    Two modes:

    **partial_denoise** (default): Shift un-executed tail forward, re-noise to
    ``tau_start`` via rectified-flow interpolation, denoise from there.
    Default ``tau_start=0.5`` preserves 50% of the warm signal and runs
    2 NFEs.

    **noise_bias**: Shift un-executed tail forward, blend into fresh Gaussian
    noise at strength ``beta``, renormalize to preserve expected Gaussian norm,
    then run the full 4-step Euler.  Keeps all 4 NFEs while injecting warm
    trajectory information into the initial noise.

    When ``prev_actions`` is ``None``, both modes fall back to standard Euler.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` instance.
        vl_embeds: Vision-language embeddings ``(B, seq_len, 2048)``.
        state_features: Encoded state ``(B, state_horizon, hidden_dim)``.
        embodiment_id: Embodiment IDs ``(B,)``.
        backbone_output: Full backbone output.
        prev_actions: Previous chunk's raw padded actions
            ``(B, action_horizon, action_dim)`` or ``None``.
        tau_start: Noise level for partial_denoise mode (default 0.5).
        n_executed: Number of steps executed from the previous chunk (default 8).
        num_steps: Total number of Euler steps for the full schedule (default 4).
        initial_noise: Optional starting noise (cold-start only).
        mode: ``"partial_denoise"`` or ``"noise_bias"``.
        beta: Blend strength for noise_bias mode (default 0.15).

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

    # 1. Shift: move un-executed tail (positions n_executed:) to the front
    remaining = action_horizon - n_executed
    warm = torch.randn(batch_size, action_horizon, action_dim,
                        dtype=dtype, device=device)
    warm[:, :remaining, :] = prev[:, n_executed:, :]

    if mode == "noise_bias":
        # ---- Noise-bias mode: full 4 NFEs with biased initialization ----
        epsilon = torch.randn(batch_size, action_horizon, action_dim,
                              dtype=dtype, device=device)
        biased = (1.0 - beta) * epsilon + beta * warm

        # Renormalize per-sample to preserve expected Gaussian norm.
        # This ensures the DiT sees input at tau=0 with standard statistics.
        eps_norm = epsilon.float().flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1)
        biased_norm = biased.float().flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1)
        safe_norm = torch.clamp(biased_norm, min=1e-8)
        actions = (biased.float() * (eps_norm / safe_norm)).to(dtype)

        start_step = 0

        if action_head.verbose:
            print(
                f"[WarmStart] Noise-bias mode  beta={beta}  "
                f"n_executed={n_executed}  eps_norm={eps_norm.mean():.2f}  "
                f"biased_norm={biased_norm.mean():.2f}"
            )
    else:
        # ---- Partial-denoise mode: re-noise and skip early steps ----
        start_step = round(tau_start * num_steps)
        actual_tau_start = start_step / float(num_steps)

        if action_head.verbose:
            print(
                f"[WarmStart] Partial-denoise mode  tau_start={tau_start}  "
                f"actual_tau_start={actual_tau_start}  "
                f"n_executed={n_executed}  prev_shape={tuple(prev.shape)}"
            )

        # Partial re-noising via rectified-flow interpolation
        epsilon = torch.randn(batch_size, remaining, action_dim,
                              dtype=dtype, device=device)
        warm[:, :remaining, :] = (
            (1.0 - actual_tau_start) * epsilon
            + actual_tau_start * warm[:, :remaining, :]
        )

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

def patch_action_head(action_head, tau_start=0.5, n_executed=8,
                      mode="partial_denoise", beta=0.15):
    """Monkey-patch the action head for warm-start denoising.

    The patched action head caches the raw ``action_pred`` from each inference
    call and uses it to warm-start the next call.  State is keyed by
    ``client_id`` so multiple clients can share one server safely.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` to patch in-place.
        tau_start: Noise level for partial_denoise mode (default 0.5 -> 2 NFEs).
        n_executed: Steps executed from each chunk (default 8).
        mode: ``"partial_denoise"`` or ``"noise_bias"``.
        beta: Blend strength for noise_bias mode (default 0.15).

    Returns:
        A ``reset(options)`` callable that clears the cached state.  **Must** be
        hooked into the policy's ``reset()`` method — see ``run_server.py``.
    """
    _client_state: dict = {}  # client_id -> prev_actions tensor

    # Save original method for cold-start fallback
    _original_fn = action_head.get_action_with_features

    def _get_cid():
        return getattr(action_head, "_current_client_id", None)

    @torch.no_grad()
    def warm_start_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        cid = _get_cid()
        prev = _client_state.get(cid)

        if prev is not None:
            actions = denoise_warm_start(
                action_head, backbone_features, state_features,
                embodiment_id, backbone_output,
                prev_actions=prev, tau_start=tau_start,
                n_executed=n_executed, mode=mode, beta=beta,
            )
        else:
            # Cold start: delegate to original 4-step Euler
            result = _original_fn(
                backbone_features, state_features, embodiment_id, backbone_output,
            )
            actions = result["action_pred"]

        # Cache for next call
        _client_state[cid] = actions.clone()

        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = warm_start_get_action_with_features

    def reset(options=None):
        """Clear warm-start cache (call on episode reset)."""
        cid = options.get("client_id") if options else None
        if cid is not None:
            _client_state.pop(cid, None)
        else:
            _client_state.clear()

    return reset


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------

def denoise_with_lab(lab, features, *, seed=None, prev_actions=None,
                     tau_start=0.5, n_executed=8, mode="partial_denoise",
                     beta=0.15):
    """Run warm-start denoising using a DenoisingLab instance.

    Args:
        lab: ``DenoisingLab`` instance.
        features: ``BackboneFeatures`` from ``lab.encode_features()``.
        seed: Random seed for initial noise (cold-start only).
        prev_actions: Previous chunk's raw padded actions, or ``None`` for
            cold start.
        tau_start: Noise level for partial_denoise mode.
        n_executed: Steps executed from the previous chunk.
        mode: ``"partial_denoise"`` or ``"noise_bias"``.
        beta: Blend strength for noise_bias mode.

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
            mode=mode, beta=beta,
        )
