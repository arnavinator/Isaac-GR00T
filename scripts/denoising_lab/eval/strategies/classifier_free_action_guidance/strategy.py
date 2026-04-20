"""Classifier-Free Action Guidance via Observation Dropout for GR00T N1.6.

Applies CFG-style guidance in velocity space during flow-matching denoising.
At each step, computes both a conditioned velocity (standard) and an
unconditioned velocity (null observation), then amplifies the conditional
direction:

    v_guided = v_uncond + w * (v_cond - v_uncond)

When w=1.0 this recovers the standard conditional velocity.  When w>1 it
amplifies the observation-conditioned signal, producing more decisive actions.

**Important:** Full CFG quality requires the model to be fine-tuned with
observation dropout (p=0.1).  Without that training, the null-embedding
velocity is not well-calibrated.  The guidance direction (v_cond - v_uncond)
still provides useful signal but the guidance weight should be kept moderate
(w <= 2.0).

Features:
- Task-phase-aware sigmoid schedule for guidance weight (lab/notebook mode)
- Fixed guidance weight for server mode
- Batched cond+uncond evaluation for efficiency (2B samples per pass)
- Compatible with models fine-tuned with observation dropout or standard models

Usage (server):
    from strategy import patch_action_head, CFGConfig
    patch_action_head(policy.model.action_head, cfg=CFGConfig(guidance_weight=1.5))

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab
    actions = denoise_with_lab(lab, features, seed=42, episode_step=200)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CFGConfig:
    """Tunable parameters for classifier-free action guidance."""

    guidance_weight: float = 1.5
    """Fixed guidance weight w for server mode.  Set to 1.0 to disable."""

    w_min: float = 1.0
    """Minimum guidance weight (episode start) for sigmoid schedule."""

    w_max: float = 4.0
    """Maximum guidance weight (episode end) for sigmoid schedule."""

    sigmoid_sharpness: float = 100.0
    """Sigmoid transition sharpness (kappa).  Larger = sharper transition."""

    max_episode_steps: int = 720
    """Maximum episode steps (for sigmoid midpoint calculation)."""

    use_sigmoid_schedule: bool = False
    """Use episode-step-aware sigmoid schedule instead of fixed weight."""

    num_steps: int = 4
    """Number of denoising steps."""

    null_embed_path: str | None = None
    """Path to a saved null embedding tensor.  If None, uses zeros."""

    velocity_clamp_ratio: float = 2.0
    """Clamp guided velocity magnitude to this ratio of conditioned velocity.
    Prevents over-saturation from high guidance weights.  Set to 0 to disable."""


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


def _repeat_backbone_output(backbone_output, n):
    """Repeat backbone_output tensor fields n times along the batch dimension."""
    repeated = BatchFeature()
    for key, val in backbone_output.items():
        if isinstance(val, torch.Tensor):
            repeated[key] = val.repeat(n, *([1] * (val.ndim - 1)))
        else:
            repeated[key] = val
    return repeated


# ---------------------------------------------------------------------------
# Guidance weight computation
# ---------------------------------------------------------------------------


def compute_guidance_weight(episode_step, cfg):
    """Compute guidance weight using sigmoid schedule or fixed value.

    Args:
        episode_step: Current environment timestep (0-based).
        cfg: CFGConfig.

    Returns:
        Guidance weight w (float).
    """
    if not cfg.use_sigmoid_schedule:
        return cfg.guidance_weight

    midpoint = cfg.max_episode_steps / 2.0
    phase = 1.0 / (1.0 + math.exp(-(episode_step - midpoint) / cfg.sigmoid_sharpness))
    return cfg.w_min + (cfg.w_max - cfg.w_min) * phase


def _get_null_vl_embeds(vl_embeds, cfg):
    """Get the null VLM embedding for unconditional velocity evaluation.

    If a saved null embedding is available (from fine-tuning with dropout),
    loads and expands it.  Otherwise, uses zeros.

    Expected null embedding shapes: (1, seq_len, 2048) or (1, 1, 2048).
    The embedding is expanded to match vl_embeds (B, seq_len, 2048).
    """
    if cfg.null_embed_path is not None:
        null_embed = torch.load(cfg.null_embed_path, map_location=vl_embeds.device,
                                weights_only=True)
        # Ensure 3D: (batch, seq_len, dim)
        if null_embed.dim() == 2:
            null_embed = null_embed.unsqueeze(0)
        if null_embed.dim() != 3:
            raise ValueError(
                f"Null embedding must be 2D or 3D, got {null_embed.dim()}D "
                f"with shape {null_embed.shape}"
            )
        # Handle sequence length mismatch: expand single-token to full seq_len
        if null_embed.shape[1] != vl_embeds.shape[1]:
            if null_embed.shape[1] == 1:
                null_embed = null_embed.expand(-1, vl_embeds.shape[1], -1)
            else:
                # Truncate or pad to match — truncation is the safe default
                seq_len = vl_embeds.shape[1]
                if null_embed.shape[1] > seq_len:
                    null_embed = null_embed[:, :seq_len, :]
                else:
                    pad = torch.zeros(
                        null_embed.shape[0], seq_len - null_embed.shape[1],
                        null_embed.shape[2], device=vl_embeds.device,
                        dtype=null_embed.dtype,
                    )
                    null_embed = torch.cat([null_embed, pad], dim=1)
        # Expand batch dim: (1, seq, dim) -> (B, seq, dim)
        return null_embed.expand(vl_embeds.shape[0], -1, -1).to(dtype=vl_embeds.dtype)
    else:
        return torch.zeros_like(vl_embeds)


def _clamp_velocity(v_guided, v_cond, clamp_ratio):
    """Clamp guided velocity magnitude to prevent over-saturation.

    Args:
        v_guided: Guided velocity (B, H, D).
        v_cond: Conditioned velocity (B, H, D).
        clamp_ratio: Max ratio of guided/conditioned magnitude.

    Returns:
        Clamped guided velocity.
    """
    if clamp_ratio <= 0:
        return v_guided

    cond_norm = v_cond.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    guided_norm = v_guided.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    max_norm = clamp_ratio * cond_norm
    scale = torch.where(guided_norm > max_norm, max_norm / guided_norm,
                        torch.ones_like(guided_norm))
    return v_guided * scale.to(v_guided.dtype)


# ---------------------------------------------------------------------------
# Core denoising function
# ---------------------------------------------------------------------------


def denoise_with_cfg(action_head, vl_embeds, state_features, embodiment_id,
                     backbone_output, *, cfg=None, seed=None, episode_step=0):
    """4-step Euler with classifier-free guidance at each step.

    Uses batched evaluation: cond and uncond velocities are computed in a single
    forward pass with batch size 2B for efficiency.

    Args:
        action_head: Gr00tN1d6ActionHead instance.
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output.
        cfg: CFGConfig.
        seed: Random seed for initial noise.
        episode_step: Current environment timestep (for sigmoid schedule).

    Returns:
        Denoised actions (B, action_horizon, action_dim).
    """
    if cfg is None:
        cfg = CFGConfig()

    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    H = action_head.action_horizon
    D = action_head.action_dim
    num_buckets = action_head.num_timestep_buckets
    num_steps = cfg.num_steps
    dt = 1.0 / num_steps

    # Compute guidance weight (constant across all denoising steps in this chunk)
    w = compute_guidance_weight(episode_step, cfg)

    # Get null embedding for unconditional velocity
    null_vl = _get_null_vl_embeds(vl_embeds, cfg)

    # Initial noise
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None
    actions = torch.randn(B, H, D, dtype=dtype, device=device, generator=gen)

    for step in range(num_steps):
        tau = step / float(num_steps)
        t_bucket = int(tau * num_buckets)

        if abs(w - 1.0) < 1e-4:
            # No guidance needed -- standard forward pass
            velocity = _evaluate_velocity(
                action_head, actions, t_bucket,
                vl_embeds, state_features, embodiment_id, backbone_output,
            )
        else:
            # Batched cond + uncond evaluation (2B samples in one pass)
            flat_actions = actions.repeat(2, 1, 1)          # (2B, H, D)
            flat_vl = torch.cat([vl_embeds, null_vl], dim=0)  # (2B, seq, 2048)
            flat_state = state_features.repeat(2, 1, 1)       # (2B, ...)
            flat_emb = embodiment_id.repeat(2)                 # (2B,)
            flat_backbone = _repeat_backbone_output(backbone_output, 2)

            flat_velocity = _evaluate_velocity(
                action_head, flat_actions, t_bucket,
                flat_vl, flat_state, flat_emb, flat_backbone,
            )

            v_cond = flat_velocity[:B]     # (B, H, D)
            v_uncond = flat_velocity[B:]   # (B, H, D)

            # CFG: amplify the observation-conditioned direction
            velocity = v_uncond + w * (v_cond - v_uncond)

            # Clamp to prevent over-saturation
            velocity = _clamp_velocity(velocity, v_cond, cfg.velocity_clamp_ratio)

        actions = actions + dt * velocity

        if action_head.verbose:
            print(f"[CFG] Step {step}/{num_steps}  tau={tau:.3f}  "
                  f"bucket={t_bucket}  w={w:.3f}  "
                  f"a_norm={actions.float().norm():.4f}  "
                  f"v_norm={velocity.float().norm():.4f}")

    if action_head.verbose:
        af = actions.float()
        print(f"[CFG] Final  shape={tuple(actions.shape)}  "
              f"mean={af.mean():.4f}  std={af.std():.4f}")

    return actions


# ---------------------------------------------------------------------------
# DenoisingLab guided_fn interface
# ---------------------------------------------------------------------------


def make_cfg_guided_fn(action_head, vl_embeds, state_features, embodiment_id,
                       backbone_output, cfg=None, episode_step=0):
    """Factory for a CFG-guided velocity modifier.

    Returns a function compatible with ``DenoisingLab.denoise(guided_fn=...)``.

    Args:
        action_head: Gr00tN1d6ActionHead (for DiT forward passes).
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output.
        cfg: CFGConfig.
        episode_step: Current environment timestep.

    Returns:
        guided_fn compatible with DenoisingLab.denoise().
    """
    if cfg is None:
        cfg = CFGConfig(use_sigmoid_schedule=True)

    w = compute_guidance_weight(episode_step, cfg)
    null_vl = _get_null_vl_embeds(vl_embeds, cfg)

    def guided_fn(actions_before, step_idx, velocity_cond):
        if abs(w - 1.0) < 1e-4:
            return velocity_cond

        tau_bucket = int(step_idx / float(cfg.num_steps) * action_head.num_timestep_buckets)

        v_uncond = _evaluate_velocity(
            action_head, actions_before, tau_bucket,
            null_vl, state_features, embodiment_id, backbone_output,
        )

        guided = v_uncond + w * (velocity_cond - v_uncond)
        guided = _clamp_velocity(guided, velocity_cond, cfg.velocity_clamp_ratio)
        return guided

    return guided_fn


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg=None):
    """Monkey-patch the action head to use classifier-free guidance.

    Replaces ``get_action_with_features()`` in-place.  Uses a fixed guidance
    weight (no episode_step tracking in server mode).
    """
    if cfg is None:
        cfg = CFGConfig()

    @torch.no_grad()
    def cfg_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions = denoise_with_cfg(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, cfg=cfg,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = cfg_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None, episode_step=0):
    """Run CFG-guided denoising via DenoisingLab.

    Uses the guided_fn interface so all intermediates are recorded.

    Args:
        lab: DenoisingLab instance.
        features: BackboneFeatures from lab.encode_features().
        seed: Random seed.
        cfg: CFGConfig.  Defaults to sigmoid-scheduled guidance.
        episode_step: Current environment timestep (for sigmoid schedule).

    Returns:
        torch.Tensor -- raw actions (B, action_horizon, action_dim).
    """
    if cfg is None:
        cfg = CFGConfig(use_sigmoid_schedule=True)

    guided_fn = make_cfg_guided_fn(
        lab.action_head,
        features.backbone_features, features.state_features,
        features.embodiment_id, features.backbone_output,
        cfg=cfg, episode_step=episode_step,
    )
    result = lab.denoise(
        features, num_steps=cfg.num_steps, guided_fn=guided_fn, seed=seed,
    )
    return result.action_pred
