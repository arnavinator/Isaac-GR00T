"""Noise-Space Mode Selection via Velocity Preview for GR00T N1.6.

Samples K noise candidates, evaluates 1 Euler step on all K simultaneously
(single batched forward pass), scores the single-step fully-extrapolated
action proxy a_1.0* = noise + velocity with lightweight quality proxies,
selects the best noise, then completes the remaining 3 denoising steps for
the winner.

Total NFEs: K + 3 (K batched in 1 pass + 3 sequential).

Usage (server):
    from strategy import patch_action_head
    patch_action_head(policy.model.action_head)

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab, NoiseSelectionConfig
    cfg = NoiseSelectionConfig(K=5)
    actions, best_noise, last_velocity = denoise_with_lab(lab, features, seed=42, cfg=cfg)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NoiseSelectionConfig:
    """Tunable parameters for noise-space mode selection."""

    K: int = 5
    """Number of noise candidates to evaluate."""

    lambda_smooth: float = 1.0
    """Weight for temporal smoothness score (penalises jerky actions)."""

    lambda_mag: float = 0.1
    """Weight for velocity magnitude score (lower mag = closer to manifold)."""

    lambda_anchor: float = 0.5
    """Weight for anchor consistency with previous chunk."""

    num_steps: int = 4
    """Number of denoising steps (remaining steps after selection)."""

    n_exec_steps: int = 8
    """Overlap region for anchor consistency (action steps executed per chunk)."""


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


def _repeat_backbone_output(backbone_output, K):
    """Repeat backbone_output tensor fields K times along the batch dimension."""
    repeated = BatchFeature()
    for key, val in backbone_output.items():
        if isinstance(val, torch.Tensor):
            repeated[key] = val.repeat(K, *([1] * (val.ndim - 1)))
        else:
            repeated[key] = val
    return repeated


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_candidates(action_proxy, velocities, K, B, cfg, prev_velocity=None):
    """Score K noise candidates using single-step fully-extrapolated proxy.

    Args:
        action_proxy: (K, B, H, D) fully-extrapolated action estimates
            a_1.0* = noise + velocity(noise, tau=0).  This is the model's
            single-step prediction of the clean action, providing a much
            stronger quality signal than the 25%-denoised a_0.25.
        velocities: (K, B, H, D) velocity predictions.
        K: Number of candidates.
        B: Batch size.
        cfg: NoiseSelectionConfig.
        prev_velocity: Optional (B, H, D) cached velocity from previous chunk.

    Returns:
        scores: (K, B) quality scores (higher = better).
    """
    device = action_proxy.device
    scores = torch.zeros(K, B, device=device, dtype=torch.float32)

    for k in range(K):
        a = action_proxy[k].float()  # (B, H, D)
        v = velocities[k].float()    # (B, H, D)

        # Temporal smoothness: penalise jerky predicted actions
        diffs = a[:, 1:, :] - a[:, :-1, :]  # (B, H-1, D)
        smoothness = -(diffs ** 2).sum(dim=(1, 2))  # (B,)
        scores[k] += cfg.lambda_smooth * smoothness

        # Velocity magnitude: lower magnitude = noise was closer to manifold
        mag = -(v ** 2).sum(dim=(1, 2))  # (B,)
        scores[k] += cfg.lambda_mag * mag

        # Anchor consistency with previous chunk (if available)
        if prev_velocity is not None:
            n = min(cfg.n_exec_steps, v.shape[1], prev_velocity.shape[1])
            cos_sim = F.cosine_similarity(
                v[:, :n, :].reshape(B, -1),
                prev_velocity[:, :n, :].float().reshape(B, -1),
                dim=1,
            )  # (B,)
            scores[k] += cfg.lambda_anchor * cos_sim

    return scores


# ---------------------------------------------------------------------------
# Core denoising function
# ---------------------------------------------------------------------------


def denoise_with_noise_selection(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *, cfg=None, seed=None, prev_velocity=None,
):
    """Noise-space mode selection: sample K noises, preview, select, complete.

    Args:
        action_head: Gr00tN1d6ActionHead instance.
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output (BatchFeature).
        cfg: NoiseSelectionConfig. Defaults to K=5.
        seed: Random seed for noise generation.
        prev_velocity: Optional (B, H, D) velocity from previous chunk for
            anchor consistency scoring.

    Returns:
        (actions, best_noise, last_velocity) tuple:
            actions: Denoised actions (B, action_horizon, action_dim).
            best_noise: Selected noise vectors (B, action_horizon, action_dim),
                useful for caching / warm-starting future chunks.
            last_velocity: Velocity from the final denoising step (B, H, D),
                used for prev_velocity caching in the server patch.
    """
    if cfg is None:
        cfg = NoiseSelectionConfig()

    K = cfg.K
    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    H = action_head.action_horizon
    D = action_head.action_dim
    num_buckets = action_head.num_timestep_buckets
    dt = 1.0 / cfg.num_steps

    # --- Step 1: Sample K noise candidates ---
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None
    noise_candidates = torch.randn(K, B, H, D, dtype=dtype, device=device,
                                   generator=gen)

    # --- Step 2: Batch-evaluate 1 Euler step for all K candidates ---
    flat_noise = noise_candidates.reshape(K * B, H, D)
    flat_vl = vl_embeds.repeat(K, 1, 1)
    flat_state = state_features.repeat(K, 1, 1)
    flat_emb = embodiment_id.repeat(K)
    flat_backbone = _repeat_backbone_output(backbone_output, K)

    velocity_flat = _evaluate_velocity(
        action_head, flat_noise, 0,  # tau=0 -> t_bucket=0
        flat_vl, flat_state, flat_emb, flat_backbone,
    )
    actions_flat = flat_noise + dt * velocity_flat

    # Single-step fully-extrapolated proxy for scoring: a_1.0* = noise + v
    # In rectified flow, v(noise, 0) ≈ data - noise, so noise + v ≈ data.
    # This proxy is dominated by signal (not noise), giving the scoring
    # heuristics meaningful action-quality information at zero extra NFEs.
    actions_1_star_flat = flat_noise + 1.0 * velocity_flat

    # Reshape back: (K, B, H, D)
    velocities = velocity_flat.reshape(K, B, H, D)
    actions_025 = actions_flat.reshape(K, B, H, D)
    actions_1_star = actions_1_star_flat.reshape(K, B, H, D)

    # --- Step 3: Score each candidate using the fully-extrapolated proxy ---
    scores = _score_candidates(actions_1_star, velocities, K, B, cfg, prev_velocity)

    # --- Step 4: Select best noise per batch element ---
    best_k = scores.argmax(dim=0)  # (B,)
    batch_idx = torch.arange(B, device=device)
    actions = actions_025[best_k, batch_idx]  # (B, H, D)
    best_noise = noise_candidates[best_k, batch_idx]  # (B, H, D)

    if action_head.verbose:
        print(f"[NoiseSelection] K={K} candidates scored. "
              f"Best indices: {best_k.tolist()}  "
              f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")

    # --- Step 5: Complete denoising with remaining 3 steps ---
    last_velocity = None
    for step in range(1, cfg.num_steps):
        tau = step / float(cfg.num_steps)
        t_bucket = int(tau * num_buckets)

        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )
        actions = actions + dt * velocity
        last_velocity = velocity

        if action_head.verbose:
            print(f"[NoiseSelection] Step {step}/{cfg.num_steps}  "
                  f"tau={tau:.3f}  bucket={t_bucket}  "
                  f"a_norm={actions.float().norm():.4f}  "
                  f"v_norm={velocity.float().norm():.4f}")

    if action_head.verbose:
        af = actions.float()
        print(f"[NoiseSelection] Final  shape={tuple(actions.shape)}  "
              f"mean={af.mean():.4f}  std={af.std():.4f}")

    return actions, best_noise, last_velocity


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg=None):
    """Monkey-patch the action head to use noise-space mode selection.

    Replaces ``get_action_with_features()`` in-place.  Caches the final
    denoising velocity across calls to enable anchor consistency scoring
    via ``prev_velocity``.
    """
    if cfg is None:
        cfg = NoiseSelectionConfig()

    _prev_velocity = [None]  # mutable closure state for cross-chunk caching

    @torch.no_grad()
    def patched_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions, _best_noise, last_velocity = denoise_with_noise_selection(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, cfg=cfg,
            prev_velocity=_prev_velocity[0],
        )
        _prev_velocity[0] = last_velocity.detach()
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = patched_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience (DenoisingLab interface)
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None, prev_velocity=None):
    """Run noise-space mode selection via DenoisingLab.

    Args:
        lab: DenoisingLab instance.
        features: BackboneFeatures from lab.encode_features().
        seed: Random seed for reproducibility.
        cfg: NoiseSelectionConfig.
        prev_velocity: Optional (B, H, D) velocity from previous chunk.

    Returns:
        (actions, best_noise, last_velocity) tuple:
            actions: torch.Tensor -- raw actions (B, action_horizon, action_dim).
            best_noise: torch.Tensor -- selected noise (B, action_horizon, action_dim).
            last_velocity: torch.Tensor -- velocity from final denoising step (B, H, D).
        Decode actions with lab.decode_raw_actions(actions).
    """
    if cfg is None:
        cfg = NoiseSelectionConfig()

    with torch.no_grad():
        return denoise_with_noise_selection(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            cfg=cfg, seed=seed, prev_velocity=prev_velocity,
        )
