"""Single-Step RK4 denoising strategy for GR00T N1.6.

Replaces the 4-step Euler loop with a single classical Runge-Kutta (RK4) step
spanning the full denoising interval tau in [0, 1].  Uses the same 4 NFEs but
achieves O(dt^4) local truncation error vs Euler's O(dt).

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
# Shared helper: evaluate the DiT velocity field at (actions, t_bucket)
# ---------------------------------------------------------------------------

def _evaluate_velocity(action_head, actions, t_bucket, vl_embeds, state_features,
                       embodiment_id, backbone_output):
    """Forward pass through the DiT to get the predicted velocity.

    Mirrors the inner body of ``Gr00tN1d6ActionHead.get_action_with_features``
    but returns only the velocity (no Euler update).
    """
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

def denoise_rk4(action_head, vl_embeds, state_features, embodiment_id,
                backbone_output, *, initial_noise=None):
    """Single RK4 step from tau=0 to tau=1.  4 NFEs, 4th-order accuracy.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` instance.
        vl_embeds: Vision-language embeddings ``(B, seq_len, 2048)``.
        state_features: Encoded state features ``(B, state_horizon, hidden_dim)``.
        embodiment_id: Embodiment IDs ``(B,)``.
        backbone_output: Full backbone output (attention masks, etc.).
        initial_noise: Optional starting noise ``(B, action_horizon, action_dim)``.

    Returns:
        Denoised actions ``(B, action_horizon, action_dim)``.
    """
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    num_buckets = action_head.num_timestep_buckets  # 1000

    if initial_noise is not None:
        a = initial_noise.to(device=device, dtype=dtype)
    else:
        a = torch.randn(
            batch_size, action_head.action_horizon, action_head.action_dim,
            dtype=dtype, device=device,
        )

    # k1: velocity at tau=0 (bucket 0)
    k1 = _evaluate_velocity(
        action_head, a, 0,
        vl_embeds, state_features, embodiment_id, backbone_output,
    )
    if action_head.verbose:
        print(f"[RK4] k1  tau=0.0    a_norm={a.float().norm():.4f}  v_norm={k1.float().norm():.4f}")

    # k2: velocity at tau=0.5 (bucket 500), input = a + 0.5*k1
    a_mid1 = a + 0.5 * k1
    k2 = _evaluate_velocity(
        action_head, a_mid1, num_buckets // 2,
        vl_embeds, state_features, embodiment_id, backbone_output,
    )
    if action_head.verbose:
        print(f"[RK4] k2  tau=0.5    a_norm={a_mid1.float().norm():.4f}  v_norm={k2.float().norm():.4f}")

    # k3: velocity at tau=0.5 (bucket 500), input = a + 0.5*k2
    a_mid2 = a + 0.5 * k2
    k3 = _evaluate_velocity(
        action_head, a_mid2, num_buckets // 2,
        vl_embeds, state_features, embodiment_id, backbone_output,
    )
    if action_head.verbose:
        print(f"[RK4] k3  tau=0.5    a_norm={a_mid2.float().norm():.4f}  v_norm={k3.float().norm():.4f}")

    # k4: velocity at tau=1.0 (bucket 999), input = a + k3
    a_end = a + k3
    k4 = _evaluate_velocity(
        action_head, a_end, num_buckets - 1,
        vl_embeds, state_features, embodiment_id, backbone_output,
    )
    if action_head.verbose:
        print(f"[RK4] k4  tau=1.0    a_norm={a_end.float().norm():.4f}  v_norm={k4.float().norm():.4f}")

    # RK4 weighted combination
    a_denoised = a + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if action_head.verbose:
        af = a_denoised.float()
        print(
            f"[RK4] Final  shape={tuple(a_denoised.shape)}  "
            f"mean={af.mean():.4f}  std={af.std():.4f}"
        )

    return a_denoised


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------

def patch_action_head(action_head):
    """Monkey-patch the action head to use single-step RK4 denoising.

    Replaces ``get_action_with_features()`` in-place.
    """

    @torch.no_grad()
    def rk4_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions = denoise_rk4(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = rk4_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------

def denoise_with_lab(lab, features, *, seed=None):
    """Run RK4 denoising using a DenoisingLab instance.

    Args:
        lab: ``DenoisingLab`` instance (provides action_head and device).
        features: ``BackboneFeatures`` from ``lab.encode_features()``.
        seed: Random seed for initial noise.

    Returns:
        ``torch.Tensor`` of shape ``(B, action_horizon, action_dim)`` — raw
        model output.  Decode with ``lab.decode_raw_actions(actions)``.
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
        return denoise_rk4(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            initial_noise=noise,
        )
