"""Multistep Velocity Recycling (Adams-Bashforth 2) for GR00T N1.6.

Replaces the standard Euler update with Adams-Bashforth 2-step (AB2) for
steps 1-3, achieving 2nd-order accuracy with zero additional NFEs.  Step 0
uses standard Euler (no cached velocity yet).

The AB2 update reuses the velocity from the *previous* step:
    a_{i+1} = a_i + dt * (1.5 * v_i - 0.5 * v_{i-1})

This captures local curvature of the flow at no extra cost.

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

def denoise_ab2(action_head, vl_embeds, state_features, embodiment_id,
                backbone_output, *, num_steps=4, initial_noise=None):
    """4-step denoising with Adams-Bashforth 2 velocity recycling.

    Step 0 uses standard Euler (no previous velocity).  Steps 1-3 use the AB2
    multistep update which is 2nd-order accurate.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` instance.
        vl_embeds: Vision-language embeddings ``(B, seq_len, 2048)``.
        state_features: Encoded state ``(B, state_horizon, hidden_dim)``.
        embodiment_id: Embodiment IDs ``(B,)``.
        backbone_output: Full backbone output.
        num_steps: Number of denoising steps (default 4).
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

    prev_velocity = None

    for step in range(num_steps):
        tau = step / float(num_steps)
        t_bucket = int(tau * num_buckets)

        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )

        if prev_velocity is None:
            # Step 0: standard Euler (no history)
            actions = actions + dt * velocity
            method = "Euler"
        else:
            # Steps 1+: Adams-Bashforth 2-step (2nd-order)
            actions = actions + dt * (1.5 * velocity - 0.5 * prev_velocity)
            method = "AB2"

        if action_head.verbose:
            print(
                f"[AB2] Step {step}/{num_steps}  tau={tau:.3f}  "
                f"bucket={t_bucket}  method={method}  "
                f"a_norm={actions.float().norm():.4f}  "
                f"v_norm={velocity.float().norm():.4f}"
            )

        prev_velocity = velocity

    if action_head.verbose:
        af = actions.float()
        print(
            f"[AB2] Final  shape={tuple(actions.shape)}  "
            f"mean={af.mean():.4f}  std={af.std():.4f}"
        )

    return actions


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------

def patch_action_head(action_head):
    """Monkey-patch the action head to use AB2 velocity recycling.

    Replaces ``get_action_with_features()`` in-place.
    """

    @torch.no_grad()
    def ab2_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions = denoise_ab2(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = ab2_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------

def denoise_with_lab(lab, features, *, seed=None, num_steps=4):
    """Run AB2 denoising using a DenoisingLab instance.

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
        return denoise_ab2(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            num_steps=num_steps, initial_noise=noise,
        )
