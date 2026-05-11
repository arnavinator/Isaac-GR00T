"""Consensus Noise Mode Selection for GR00T N1.6.

Samples K noise candidates, fully denoises all K through D Euler steps,
then selects the candidate whose EEF trajectory is closest to the consensus
(mean) of all K.  Scoring uses three physically meaningful metrics computed
on cumulative EEF position/rotation trajectories: closeness to mean position,
closeness to mean rotation, and jerk minimization.

Total NFEs: K * D (e.g. 8 * 4 = 32), batched as D sequential DiT passes
each with batch size K * B.

Usage (server):
    from strategy import patch_action_head
    reset_fn = patch_action_head(policy.model.action_head)

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab, ConsensusConfig
    cfg = ConsensusConfig(K=8)
    actions, best_noise, last_velocity = denoise_with_lab(lab, features, seed=42, cfg=cfg)
    decoded = lab.decode_raw_actions(actions, features.states)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConsensusConfig:
    """Tunable parameters for consensus noise mode selection."""

    K: int = 8
    """Number of noise candidates to evaluate."""

    num_steps: int = 4
    """Number of Euler denoising steps.  All K candidates go through all D steps."""

    lambda_pos: float = 1.0
    """Weight for EEF position closeness-to-mean score."""

    lambda_rot: float = 0.5
    """Weight for EEF rotation closeness-to-mean score."""

    lambda_jerk: float = 0.1
    """Weight for jerk minimization score (third finite difference of
    cumulative EEF position)."""

    action_horizon: int = 16
    """Number of meaningful timesteps to extract for scoring.
    PandaOmron uses 16; the remaining timesteps (up to 50) are padding."""

    eef_pos_slice: tuple[int, int] = (0, 3)
    """Start and end indices in the per-timestep action vector for EEF position."""

    eef_rot_slice: tuple[int, int] = (3, 6)
    """Start and end indices in the per-timestep action vector for EEF rotation."""


# ---------------------------------------------------------------------------
# Shared helpers (copied from noise_space_mode_selection/strategy.py)
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


def _normalize_score(score):
    """Normalize a (K, B) score tensor to [0, 1] per batch element.

    Maps worst (most negative) to 0 and best (least negative / zero) to 1.
    If all candidates have identical scores for a batch element, returns
    uniform 1/K (no preference).
    """
    score_min = score.min(dim=0, keepdim=True).values  # (1, B)
    score_max = score.max(dim=0, keepdim=True).values  # (1, B)
    denom = score_max - score_min
    safe = denom > 0
    result = torch.where(safe, (score - score_min) / denom, torch.ones_like(score) / score.shape[0])
    return result


def _score_candidates(candidates, K, B, cfg):
    """Score K fully-denoised candidates on cumulative EEF trajectories.

    Computes three terms on cumulative position/rotation trajectories
    in the normalized action space, each rescaled to [0, 1] per batch
    element before combining with lambda weights:
      1. Position closeness to mean  (1 = closest to consensus)
      2. Rotation closeness to mean  (1 = closest to consensus)
      3. Jerk minimization           (1 = smoothest trajectory)

    Args:
        candidates: (K, B, H, D) fully-denoised action tensors.
        K: Number of candidates.
        B: Batch size.
        cfg: ConsensusConfig.

    Returns:
        scores: (K, B) total score per candidate per batch element.
    """
    h = cfg.action_horizon
    ps, pe = cfg.eef_pos_slice
    rs, re = cfg.eef_rot_slice

    pos_deltas = candidates[:, :, :h, ps:pe].float()  # (K, B, h, 3)
    rot_deltas = candidates[:, :, :h, rs:re].float()  # (K, B, h, 3)

    # Cumulative trajectories with origin (matches notebook: [0, cumsum(deltas)])
    origin = torch.zeros(K, B, 1, 3, dtype=pos_deltas.dtype, device=pos_deltas.device)
    cumpos = torch.cat([origin, torch.cumsum(pos_deltas, dim=2)], dim=2)  # (K, B, h+1, 3)
    cumrot = torch.cat([origin, torch.cumsum(rot_deltas, dim=2)], dim=2)  # (K, B, h+1, 3)

    mean_cumpos = cumpos.mean(dim=0)  # (B, h+1, 3)
    mean_cumrot = cumrot.mean(dim=0)  # (B, h+1, 3)

    # Position closeness: negative mean squared distance to consensus
    pos_diff = cumpos - mean_cumpos.unsqueeze(0)       # (K, B, h+1, 3)
    pos_score = -(pos_diff ** 2).mean(dim=(2, 3))      # (K, B)

    # Rotation closeness: same metric on cumulative rotation
    rot_diff = cumrot - mean_cumrot.unsqueeze(0)        # (K, B, h+1, 3)
    rot_score = -(rot_diff ** 2).mean(dim=(2, 3))       # (K, B)

    # Normalize each to [0, 1] so lambdas control relative importance
    scores = cfg.lambda_pos * _normalize_score(pos_score) \
           + cfg.lambda_rot * _normalize_score(rot_score)

    # Jerk minimization: third finite difference of cumulative position
    if h >= 4 and cfg.lambda_jerk > 0:
        jerk = (cumpos[:, :, 3:, :]
                - 3 * cumpos[:, :, 2:-1, :]
                + 3 * cumpos[:, :, 1:-2, :]
                - cumpos[:, :, :-3, :])                 # (K, B, h-2, 3)
        jerk_score = -(jerk ** 2).mean(dim=(2, 3))      # (K, B)
        scores = scores + cfg.lambda_jerk * _normalize_score(jerk_score)

    return scores


# ---------------------------------------------------------------------------
# Core denoising function
# ---------------------------------------------------------------------------


def denoise_consensus(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *, cfg=None, seed=None,
):
    """Consensus noise mode selection: sample K, denoise all, score, select.

    Args:
        action_head: Gr00tN1d6ActionHead instance.
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output (BatchFeature).
        cfg: ConsensusConfig.  Defaults to K=8, 4 steps.
        seed: Random seed for noise generation.

    Returns:
        (actions, best_noise, last_velocity) tuple:
            actions: Denoised actions (B, action_horizon, action_dim).
            best_noise: Selected noise vectors (B, action_horizon, action_dim).
            last_velocity: Velocity from the final denoising step (B, H, D).
    """
    if cfg is None:
        cfg = ConsensusConfig()

    K = cfg.K
    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    H = action_head.action_horizon
    D = action_head.action_dim
    num_buckets = action_head.num_timestep_buckets
    dt = 1.0 / cfg.num_steps

    # --- Step 1: Sample K noise candidates ---
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    noise_candidates = torch.randn(K, B, H, D, dtype=dtype, device=device, generator=gen)

    # --- Step 2: Replicate conditioning for batched inference ---
    flat_vl = vl_embeds.repeat(K, 1, 1)
    flat_state = state_features.repeat(K, 1, 1)
    flat_emb = embodiment_id.repeat(K)
    flat_backbone = _repeat_backbone_output(backbone_output, K)

    # --- Step 3: Full D-step Euler denoising for all K candidates ---
    flat_actions = noise_candidates.reshape(K * B, H, D)
    last_velocity_flat = None

    for step in range(cfg.num_steps):
        tau = step / float(cfg.num_steps)
        t_bucket = int(tau * num_buckets)

        velocity = _evaluate_velocity(
            action_head, flat_actions, t_bucket,
            flat_vl, flat_state, flat_emb, flat_backbone,
        )
        flat_actions = flat_actions + dt * velocity
        last_velocity_flat = velocity

    # --- Step 4: Score in normalized action space ---
    all_candidates = flat_actions.reshape(K, B, H, D)
    scores = _score_candidates(all_candidates, K, B, cfg)

    # --- Step 5: Select best per batch element ---
    best_k = scores.argmax(dim=0)  # (B,)
    batch_idx = torch.arange(B, device=device)
    actions = all_candidates[best_k, batch_idx]          # (B, H, D)
    best_noise = noise_candidates[best_k, batch_idx]     # (B, H, D)
    last_velocity = last_velocity_flat.reshape(K, B, H, D)[best_k, batch_idx]

    if action_head.verbose:
        print(f"[Consensus] K={K} candidates scored over {cfg.num_steps} steps. "
              f"Best indices: {best_k.tolist()}  "
              f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    return actions, best_noise, last_velocity


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg=None):
    """Monkey-patch the action head to use consensus noise mode selection.

    Replaces ``get_action_with_features()`` in-place.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` to patch in-place.
        cfg: ``ConsensusConfig``.

    Returns:
        A ``reset()`` callable.  Must be hooked into the policy's ``reset()``
        method for correct episode boundary handling.
    """
    if cfg is None:
        cfg = ConsensusConfig()

    @torch.no_grad()
    def patched_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions, _best_noise, _last_velocity = denoise_consensus(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, cfg=cfg,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = patched_get_action_with_features

    def reset(options=None):
        pass

    return reset


# ---------------------------------------------------------------------------
# Notebook convenience (DenoisingLab interface)
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None):
    """Run consensus noise mode selection via DenoisingLab.

    Args:
        lab: DenoisingLab instance.
        features: BackboneFeatures from lab.encode_features().
        seed: Random seed for reproducibility.
        cfg: ConsensusConfig.

    Returns:
        (actions, best_noise, last_velocity) tuple.
        Decode actions with lab.decode_raw_actions(actions, features.states).
    """
    if cfg is None:
        cfg = ConsensusConfig()

    with torch.no_grad():
        return denoise_consensus(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            cfg=cfg, seed=seed,
        )
