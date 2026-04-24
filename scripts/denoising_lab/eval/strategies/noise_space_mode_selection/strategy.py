"""Noise-Space Mode Selection via Velocity Preview for GR00T N1.6.

Samples K noise candidates, evaluates 1 Euler step on all K simultaneously
(single batched forward pass), scores the single-step fully-extrapolated
action proxy a_1.0* = noise + velocity with lightweight quality proxies,
selects the best noise, then completes the remaining 3 denoising steps for
the winner.

Anchor consistency uses distance-weighted L2 in action space: the candidate's
extrapolated proxy (steps 0..n-1) is compared to the previous chunk's
predicted-but-unexecuted tail (steps n_exec..end).  Comparison weights decay
geometrically so nearer predictions dominate.

Total NFEs: K + 3 (K batched in 1 pass + 3 sequential).

Usage (server):
    from strategy import patch_action_head
    reset_fn = patch_action_head(policy.model.action_head)

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab, NoiseSelectionConfig
    cfg = NoiseSelectionConfig(K=8)
    actions, best_noise, last_velocity = denoise_with_lab(lab, features, seed=42, cfg=cfg)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NoiseSelectionConfig:
    """Tunable parameters for noise-space mode selection."""

    K: int = 8
    """Number of noise candidates to evaluate.  Default 8 based on
    ``profile_k_runtime.py`` results showing K=8 provides good mode
    exploration with acceptable latency overhead."""

    lambda_smooth: float = 0.1
    """Weight for temporal smoothness score (penalises jerky actions)."""

    lambda_mag: float = 0.0
    """Weight for velocity magnitude score (lower mag = closer to manifold)."""

    lambda_anchor: float = 2.0
    """Weight for anchor consistency with previous chunk."""

    anchor_decay: float = 0.5
    """Per-step decay for distance-weighted anchor scoring.  Step 0 of the
    overlap gets weight 1.0, step j gets weight ``anchor_decay ** j``.
    Nearer predictions are more reliable so they dominate the score."""

    num_steps: int = 4
    """Number of denoising steps (remaining steps after selection)."""

    noise_type: str = "gaussian"
    """Noise distribution for candidates: ``"gaussian"`` (N(0,1), matches
    baseline training) or ``"uniform"`` (Uniform[-sqrt(3), sqrt(3)],
    variance-matched to N(0,1) for comparable norm statistics)."""

    n_exec_steps: int = 8
    """Overlap region for anchor consistency (action steps executed per chunk)."""

    score_dims: int | None = 12
    """Number of leading action dims to use for smoothness and magnitude scoring.
    Dims beyond this are padding in the multi-embodiment action space and carry
    only noise (the model has no training loss on them).  Set to ``None`` to
    score all dims.  Default 12 covers PandaOmron's meaningful dims: EEF
    position (3) + rotation (3) + gripper (1) + base_motion (4) + control_mode (1)."""

    score_horizon: int | None = 16
    """Number of leading timesteps to use for smoothness and magnitude scoring.
    Timesteps beyond this are padding in the multi-embodiment action space.
    Set to ``None`` (default) to score all timesteps — this includes temporal
    padding which empirically correlates with meaningful trajectory quality
    (Spearman rho ~0.3).  Set to 16 for PandaOmron to score only the
    meaningful action horizon."""

    noise_keyframes: int | None = None
    """When set, generates noise at this many temporal keyframes and linearly
    interpolates to the full action horizon.  Produces temporally smooth noise
    candidates.  ``None`` (default) disables smoothing — standard i.i.d. noise.
    Try 4-8 for smooth noise; 1 gives constant noise per candidate (extreme).
    Works with both ``"gaussian"`` and ``"uniform"`` noise types."""


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


def _score_candidates(action_proxy, velocities, K, B, cfg, prev_actions=None,
                      return_breakdown=False):
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
        prev_actions: Optional (B, H, D) cached denoised actions from
            previous chunk.  Used for action-space anchor consistency.
        return_breakdown: If True, return ``(scores, breakdown)`` where
            breakdown is a dict of per-component weighted contributions.

    Returns:
        scores: (K, B) quality scores (higher = better).
        breakdown (only when return_breakdown=True): dict with keys
            ``scores_smooth``, ``scores_mag``, ``scores_anchor``,
            ``scores_total`` — each (K, B).
    """
    device = action_proxy.device
    scores = torch.zeros(K, B, device=device, dtype=torch.float32)

    if return_breakdown:
        bd_smooth = torch.zeros(K, B, device=device, dtype=torch.float32)
        bd_mag = torch.zeros(K, B, device=device, dtype=torch.float32)
        bd_anchor = torch.zeros(K, B, device=device, dtype=torch.float32)

    for k in range(K):
        a = action_proxy[k].float()  # (B, H, D)
        v = velocities[k].float()    # (B, H, D)

        # Slice to meaningful dims and horizon for smoothness/magnitude.
        # None means "all" for that axis (PyTorch slice semantics).
        sd = cfg.score_dims
        sh = cfg.score_horizon
        a_scored = a[:, :sh, :sd]
        v_scored = v[:, :sh, :sd]

        # Temporal smoothness: penalise jerky predicted actions
        diffs = a_scored[:, 1:, :] - a_scored[:, :-1, :]  # (B, H-1, D')
        smoothness = -(diffs ** 2).mean(dim=(1, 2))  # (B,)
        smooth_contrib = cfg.lambda_smooth * smoothness
        scores[k] += smooth_contrib
        if return_breakdown:
            bd_smooth[k] = smooth_contrib

        # Velocity magnitude: lower magnitude = noise was closer to manifold
        mag = -(v_scored ** 2).mean(dim=(1, 2))  # (B,)
        mag_contrib = cfg.lambda_mag * mag
        scores[k] += mag_contrib
        if return_breakdown:
            bd_mag[k] = mag_contrib

        # Anchor consistency: distance-weighted L2 in action space.
        # Compare candidate's predicted near-future (steps 0..n-1) to the
        # previous chunk's predicted-but-unexecuted tail (steps n_exec..n_exec+n-1).
        # Both tensors are action estimates in the same normalised space.
        if prev_actions is not None:
            n = cfg.n_exec_steps
            H_scored = a_scored.shape[1]
            n_prev_tail = H_scored - n
            n_overlap = min(n, n_prev_tail)

            if n_overlap > 0:
                candidate_near = a_scored[:, :n_overlap, :]           # (B, n_overlap, D')
                prev_scored = prev_actions[:, :sh, :sd].float()
                prev_tail = prev_scored[:, n:n + n_overlap, :]        # (B, n_overlap, D')

                # Distance-weighted: step j gets weight decay^j
                weights = torch.tensor(
                    [cfg.anchor_decay ** j for j in range(n_overlap)],
                    device=device, dtype=torch.float32,
                )  # (n_overlap,)
                weights = weights / weights.sum()

                sq_dist = (candidate_near - prev_tail) ** 2  # (B, n_overlap, D)
                sq_dist_per_step = sq_dist.mean(dim=2)       # (B, n_overlap)
                weighted_dist = (sq_dist_per_step * weights.unsqueeze(0)).sum(dim=1)  # (B,)
                anchor_contrib = -cfg.lambda_anchor * weighted_dist
                scores[k] += anchor_contrib
                if return_breakdown:
                    bd_anchor[k] = anchor_contrib

    if return_breakdown:
        breakdown = {
            "scores_smooth": bd_smooth,
            "scores_mag": bd_mag,
            "scores_anchor": bd_anchor,
            "scores_total": scores,
        }
        return scores, breakdown
    return scores


# ---------------------------------------------------------------------------
# Core denoising function
# ---------------------------------------------------------------------------


def denoise_with_noise_selection(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *, cfg=None, seed=None, prev_actions=None, diagnostics_log=None,
):
    """Noise-space mode selection: sample K noises, preview, select, complete.

    Args:
        action_head: Gr00tN1d6ActionHead instance.
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output (BatchFeature).
        cfg: NoiseSelectionConfig. Defaults to K=8.
        seed: Random seed for noise generation.
        prev_actions: Optional (B, H, D) denoised actions from previous chunk
            for action-space anchor consistency scoring.
        diagnostics_log: Optional list to append per-chunk diagnostic dicts to.
            When provided, captures full scoring internals and candidate tensors.

    Returns:
        (actions, best_noise, last_velocity) tuple:
            actions: Denoised actions (B, action_horizon, action_dim).
            best_noise: Selected noise vectors (B, action_horizon, action_dim),
                useful for caching / warm-starting future chunks.
            last_velocity: Velocity from the final denoising step (B, H, D).
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

    n_kf = cfg.noise_keyframes
    sample_H = n_kf if (n_kf is not None and n_kf > 0 and n_kf < H) else H

    if cfg.noise_type == "uniform":
        bound = math.sqrt(3.0)
        raw_noise = torch.rand(
            K, B, sample_H, D, dtype=dtype, device=device, generator=gen,
        ) * (2 * bound) - bound
    else:
        raw_noise = torch.randn(
            K, B, sample_H, D, dtype=dtype, device=device, generator=gen,
        )

    if sample_H < H:
        # Interpolate coarse keyframes to full horizon.
        # F.interpolate expects (N, C, L): treat dims as channels, time as length.
        flat = raw_noise.reshape(K * B, sample_H, D).permute(0, 2, 1)  # (KB, D, n_kf)
        smooth = F.interpolate(flat, size=H, mode="linear", align_corners=True)  # (KB, D, H)
        noise_candidates = smooth.permute(0, 2, 1).reshape(K, B, H, D)  # (K, B, H, D)
    else:
        noise_candidates = raw_noise

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
    if diagnostics_log is not None:
        scores, breakdown = _score_candidates(
            actions_1_star, velocities, K, B, cfg, prev_actions,
            return_breakdown=True,
        )
    else:
        scores = _score_candidates(actions_1_star, velocities, K, B, cfg, prev_actions)

    # --- Step 4: Select best noise per batch element ---
    best_k = scores.argmax(dim=0)  # (B,)
    batch_idx = torch.arange(B, device=device)
    actions = actions_025[best_k, batch_idx]  # (B, H, D)
    best_noise = noise_candidates[best_k, batch_idx]  # (B, H, D)

    if action_head.verbose:
        print(f"[NoiseSelection] K={K} candidates scored. "
              f"Best indices: {best_k.tolist()}  "
              f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")

    # Capture diagnostics before denoising loop (scoring + candidate tensors)
    if diagnostics_log is not None:
        sorted_scores, _ = scores[:, 0].sort(descending=True)
        score_gap = (sorted_scores[0] - sorted_scores[1]).item() if K > 1 else 0.0

        diag = {
            "scores_smooth":        breakdown["scores_smooth"][:, 0].cpu().clone(),
            "scores_mag":           breakdown["scores_mag"][:, 0].cpu().clone(),
            "scores_anchor":        breakdown["scores_anchor"][:, 0].cpu().clone(),
            "scores_total":         breakdown["scores_total"][:, 0].cpu().clone(),
            "best_k":               best_k[0].item(),
            "score_gap":            score_gap,
            "score_mean":           scores[:, 0].mean().item(),
            "score_std":            scores[:, 0].std().item(),
            "score_min":            scores[:, 0].min().item(),
            "score_max":            scores[:, 0].max().item(),
            "noise_candidates":     noise_candidates[:, 0].cpu().clone(),
            "action_proxies_1star": actions_1_star[:, 0].cpu().clone(),
            "velocities":           velocities[:, 0].cpu().clone(),
            "prev_actions":         prev_actions[0].cpu().clone() if prev_actions is not None else None,
            "has_prev_actions":     prev_actions is not None,
            "denoising_actions":    [],
            "denoising_velocities": [],
        }

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

        if diagnostics_log is not None:
            diag["denoising_actions"].append(actions[0].cpu().clone())
            diag["denoising_velocities"].append(velocity[0].cpu().clone())

        if action_head.verbose:
            print(f"[NoiseSelection] Step {step}/{cfg.num_steps}  "
                  f"tau={tau:.3f}  bucket={t_bucket}  "
                  f"a_norm={actions.float().norm():.4f}  "
                  f"v_norm={velocity.float().norm():.4f}")

    if action_head.verbose:
        af = actions.float()
        print(f"[NoiseSelection] Final  shape={tuple(actions.shape)}  "
              f"mean={af.mean():.4f}  std={af.std():.4f}")

    if diagnostics_log is not None:
        diag["final_actions"] = actions[0].cpu().clone()
        diagnostics_log.append(diag)

    return actions, best_noise, last_velocity


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg=None, diagnostics_log=None):
    """Monkey-patch the action head to use noise-space mode selection.

    Replaces ``get_action_with_features()`` in-place.  Caches the denoised
    actions across calls to enable action-space anchor consistency scoring.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` to patch in-place.
        cfg: ``NoiseSelectionConfig``.
        diagnostics_log: Optional list to collect per-chunk diagnostics into.

    Returns:
        A ``reset()`` callable that clears the cached state.  **Must** be
        hooked into the policy's ``reset()`` method so that stale actions
        from a previous episode don't distort anchor scoring.
    """
    if cfg is None:
        cfg = NoiseSelectionConfig()

    _prev_actions = [None]  # mutable closure state for cross-chunk caching
    _chunk_idx = [0]

    @torch.no_grad()
    def patched_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions, _best_noise, last_velocity = denoise_with_noise_selection(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, cfg=cfg,
            prev_actions=_prev_actions[0],
            diagnostics_log=diagnostics_log,
        )
        _prev_actions[0] = actions.detach()

        if diagnostics_log is not None and len(diagnostics_log) > 0:
            diagnostics_log[-1]["chunk_idx"] = _chunk_idx[0]
        _chunk_idx[0] += 1

        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = patched_get_action_with_features

    def reset():
        """Clear cached prev_actions (call on episode reset)."""
        _prev_actions[0] = None
        _chunk_idx[0] = 0

    return reset


# ---------------------------------------------------------------------------
# Notebook convenience (DenoisingLab interface)
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None, prev_actions=None):
    """Run noise-space mode selection via DenoisingLab.

    Args:
        lab: DenoisingLab instance.
        features: BackboneFeatures from lab.encode_features().
        seed: Random seed for reproducibility.
        cfg: NoiseSelectionConfig.
        prev_actions: Optional (B, H, D) denoised actions from previous chunk.

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
            cfg=cfg, seed=seed, prev_actions=prev_actions,
        )
