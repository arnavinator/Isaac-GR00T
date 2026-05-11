"""Density-Aware Denoising via Velocity Divergence Estimation.

Estimates the divergence of the velocity field at each Euler denoising step
using Hutchinson's trace estimator with batched finite differences.  The
accumulated divergence gives a free log-likelihood estimate of the denoised
output under the learned distribution — the most principled quality metric
possible for a generative model.

Three operating modes:
  monitor  — Standard 4-step Euler with divergence logging.  Cost: +12%.
  guided   — Divergence-guided velocity scaling: amplify when converging
             on a mode, dampen when diverging.  Cost: +12%.
  rank     — Best-of-N candidate selection ranked by estimated log-likelihood,
             optionally augmented with anchor consistency.  Cost: +50% (N=4).

Dimension-restricted probes: Rademacher vectors are zeroed on multi-embodiment
padding dims, restricting the Hutchinson estimator to the meaningful action
subspace and dramatically improving signal-to-noise ratio.

Usage (server):
    from strategy import patch_action_head, DensityAwareConfig
    cfg = DensityAwareConfig(mode="guided", alpha=0.15)
    reset_fn = patch_action_head(policy.model.action_head, cfg=cfg)

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab, DensityAwareConfig
    cfg = DensityAwareConfig(mode="guided")
    actions, diag = denoise_with_lab(lab, features, seed=42, cfg=cfg)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DensityAwareConfig:
    """Tunable parameters for density-aware denoising."""

    mode: str = "guided"
    """Operating mode: ``"monitor"`` (diagnostic only), ``"guided"``
    (divergence-based velocity scaling), or ``"rank"`` (best-of-N by
    estimated log-likelihood)."""

    h: float = 1e-3
    """Finite-difference perturbation scale for the Hutchinson estimator.
    Bounded below by bfloat16 precision (~1e-3)."""

    alpha: float = 0.15
    """Guidance strength for guided mode.  The velocity scale factor
    lies in ``[1 - alpha, 1 + alpha]``.  0 = pure monitoring."""

    D0: float | None = None
    """Divergence normalization scale for guided mode.  ``None`` (default)
    auto-normalises per step as ``max(|D_hat|, 1.0)``.  Set to a fixed
    value from ``calibrate_divergence_scale()`` for more stable guidance."""

    N: int = 4
    """Number of noise candidates for rank mode."""

    num_steps: int = 4
    """Number of Euler denoising steps."""

    score_dims: int | None = 12
    """Restrict divergence probe to this many leading action dims.  Dims
    beyond this are multi-embodiment padding with no training loss —
    they add pure noise to the Hutchinson estimator.  Default 12 covers
    PandaOmron's meaningful dims (EEF pos/rot + gripper + base + mode).
    ``None`` to probe all dims."""

    score_horizon: int | None = None
    """Restrict divergence probe to this many leading timesteps.  ``None``
    (default) probes all timesteps.  Set to 16 for PandaOmron to match
    the meaningful action horizon."""

    lambda_anchor: float = 0.0
    """Anchor consistency weight for rank mode.  When > 0 and previous
    actions are available, rank score = log_likelihood + lambda_anchor *
    anchor_score.  0 = pure log-likelihood ranking."""

    anchor_decay: float = 0.5
    """Geometric decay for distance-weighted anchor scoring.  Step 0 of
    the overlap gets weight 1.0, step j gets ``anchor_decay ** j``."""

    n_exec_steps: int = 8
    """Overlap region size for anchor consistency (action steps executed
    per chunk by MultiStepWrapper)."""


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class DensityDiagnostics:
    """Rich diagnostic output from density-aware denoising."""

    divergences: list[float] = field(default_factory=list)
    cumulative_divergence: float = 0.0
    log_likelihood_estimate: float = 0.0
    noise_log_prob: float = 0.0
    density_trend: str = "unknown"
    guided_scales: list[float] | None = None
    candidate_log_likelihoods: list[float] | None = None
    best_candidate_idx: int | None = None
    log_likelihood_spread: float | None = None


# ---------------------------------------------------------------------------
# Shared helpers
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


def _make_probe_vector(shape, dtype, device, cfg):
    """Generate a Rademacher probe vector, zeroed on padding dims.

    Restricting the probe to meaningful dims focuses the Hutchinson
    estimator on the action subspace with training signal, cutting
    noise from multi-embodiment padding by ~33x for PandaOmron.
    """
    z = torch.sign(torch.randn(shape, dtype=dtype, device=device))
    z[z == 0] = 1.0

    if cfg.score_dims is not None:
        z[..., cfg.score_dims:] = 0.0
    if cfg.score_horizon is not None and z.ndim >= 3:
        z[..., cfg.score_horizon:, :] = 0.0

    return z


# ---------------------------------------------------------------------------
# Core: monitor / guided modes
# ---------------------------------------------------------------------------


def denoise_density_aware(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *, cfg=None, initial_noise=None, seed=None, prev_actions=None,
    diagnostics_log=None,
):
    """4-step Euler with velocity divergence estimation.

    At each step, evaluates the DiT at the current action and a small
    perturbation in a single batched forward pass (batch 2B), then uses
    Hutchinson's estimator to compute the divergence.

    In guided mode, scales the velocity by a factor derived from the
    divergence sign: amplify when converging on a mode, dampen when
    diverging away.

    Args:
        action_head: Gr00tN1d6ActionHead instance.
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output (BatchFeature).
        cfg: DensityAwareConfig.
        initial_noise: Optional starting noise (B, H, D).
        seed: Random seed for noise generation.
        prev_actions: Previous chunk's denoised actions for anchor (rank mode).
        diagnostics_log: Optional list to append per-chunk diagnostics to.

    Returns:
        (actions, diagnostics) tuple.
    """
    if cfg is None:
        cfg = DensityAwareConfig()

    if cfg.mode == "rank":
        return _density_ranked_best_of_n(
            action_head, vl_embeds, state_features, embodiment_id,
            backbone_output, cfg=cfg, seed=seed, prev_actions=prev_actions,
            diagnostics_log=diagnostics_log,
        )

    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    H = action_head.action_horizon
    D = action_head.action_dim
    num_buckets = action_head.num_timestep_buckets
    dt = 1.0 / cfg.num_steps

    if initial_noise is not None:
        actions = initial_noise.to(device=device, dtype=dtype)
    elif seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        actions = torch.randn(B, H, D, dtype=dtype, device=device, generator=gen)
    else:
        actions = torch.randn(B, H, D, dtype=dtype, device=device)

    diag = DensityDiagnostics(
        noise_log_prob=-0.5 * actions.float().pow(2).sum().item() / B,
        guided_scales=[] if cfg.mode == "guided" else None,
    )

    backbone_2 = _repeat_backbone_output(backbone_output, 2)

    for step in range(cfg.num_steps):
        tau = step / float(cfg.num_steps)
        t_bucket = int(tau * num_buckets)

        z = _make_probe_vector((B, H, D), dtype, device, cfg)

        a_batch = torch.cat([actions, actions + cfg.h * z], dim=0)
        vl_batch = vl_embeds.repeat(2, 1, 1)
        state_batch = state_features.repeat(2, 1, 1)
        emb_batch = embodiment_id.repeat(2)

        v_batch = _evaluate_velocity(
            action_head, a_batch, t_bucket,
            vl_batch, state_batch, emb_batch, backbone_2,
        )

        v = v_batch[:B]
        v_pert = v_batch[B:]

        jvp = (v_pert.float() - v.float()) / cfg.h
        div_est = (z.float() * jvp).sum().item() / B
        diag.divergences.append(div_est)

        if cfg.mode == "guided" and cfg.alpha > 0:
            D0 = cfg.D0 if cfg.D0 is not None else max(abs(div_est), 1.0)
            scale = 1.0 + cfg.alpha * math.tanh(-div_est / D0)
            v_step = v * scale
            diag.guided_scales.append(scale)
        else:
            v_step = v

        actions = actions + dt * v_step

        if action_head.verbose:
            vf = v.float()
            scale_str = ""
            if cfg.mode == "guided" and cfg.alpha > 0:
                scale_str = f"scale={diag.guided_scales[-1]:.3f}  "
            print(
                f"[DensityAware] Step {step}/{cfg.num_steps}  "
                f"tau={tau:.3f}  bucket={t_bucket}  "
                f"div={div_est:+.2f}  {scale_str}"
                f"v_norm={vf.norm():.4f}  a_norm={actions.float().norm():.4f}"
            )

    diag.cumulative_divergence = sum(diag.divergences)
    diag.log_likelihood_estimate = (
        diag.noise_log_prob
        - 0.5 * H * D * math.log(2 * math.pi)
        - dt * diag.cumulative_divergence
    )

    late_div = sum(diag.divergences[cfg.num_steps // 2:])
    if late_div < -0.1:
        diag.density_trend = "converging"
    elif late_div > 0.1:
        diag.density_trend = "diverging"
    else:
        diag.density_trend = "stable"

    if action_head.verbose:
        af = actions.float()
        print(
            f"[DensityAware] Final  shape={tuple(actions.shape)}  "
            f"mean={af.mean():.4f}  std={af.std():.4f}  "
            f"trend={diag.density_trend}  "
            f"log_lik={diag.log_likelihood_estimate:.2f}"
        )

    if diagnostics_log is not None:
        diagnostics_log.append({
            "divergences": list(diag.divergences),
            "cumulative_divergence": diag.cumulative_divergence,
            "log_likelihood_estimate": diag.log_likelihood_estimate,
            "density_trend": diag.density_trend,
            "guided_scales": list(diag.guided_scales) if diag.guided_scales else None,
        })

    return actions, diag


# ---------------------------------------------------------------------------
# Core: rank mode (best-of-N by log-likelihood)
# ---------------------------------------------------------------------------


def _density_ranked_best_of_n(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *, cfg, seed=None, prev_actions=None, diagnostics_log=None,
):
    """Generate N candidates, rank by estimated log-likelihood, select best.

    All N candidates (each with clean + perturbed) are denoised in a single
    set of batched forward passes (4 sequential at batch size 2NB).

    When ``cfg.lambda_anchor > 0`` and ``prev_actions`` is provided, the
    ranking combines log-likelihood with anchor consistency.
    """
    N = cfg.N
    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    H = action_head.action_horizon
    D = action_head.action_dim
    num_buckets = action_head.num_timestep_buckets
    dt = 1.0 / cfg.num_steps

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    noises = torch.randn(N, B, H, D, dtype=dtype, device=device, generator=gen)
    noise_log_probs = -0.5 * noises.float().pow(2).sum(dim=(2, 3))  # (N, B)

    actions = noises.clone()
    accumulated_div = torch.zeros(N, B, device=device)

    backbone_2n = _repeat_backbone_output(backbone_output, 2 * N)

    for step in range(cfg.num_steps):
        tau = step / float(cfg.num_steps)
        t_bucket = int(tau * num_buckets)

        z = _make_probe_vector((N, B, H, D), dtype, device, cfg)

        flat_a = actions.reshape(N * B, H, D)
        flat_a_pert = (actions + cfg.h * z).reshape(N * B, H, D)
        a_batch = torch.cat([flat_a, flat_a_pert], dim=0)

        vl_batch = vl_embeds.repeat(2 * N, 1, 1)
        state_batch = state_features.repeat(2 * N, 1, 1)
        emb_batch = embodiment_id.repeat(2 * N)

        v_batch = _evaluate_velocity(
            action_head, a_batch, t_bucket,
            vl_batch, state_batch, emb_batch, backbone_2n,
        )

        v_clean = v_batch[:N * B].reshape(N, B, H, D)
        v_pert = v_batch[N * B:].reshape(N, B, H, D)

        jvp = (v_pert.float() - v_clean.float()) / cfg.h
        div_per_candidate = (z.float() * jvp).sum(dim=(2, 3))  # (N, B)
        accumulated_div += div_per_candidate

        actions = actions + dt * v_clean

        if action_head.verbose:
            print(
                f"[DensityAware-Rank] Step {step}/{cfg.num_steps}  "
                f"tau={tau:.3f}  bucket={t_bucket}  "
                f"div_range=[{div_per_candidate.min():.1f}, "
                f"{div_per_candidate.max():.1f}]"
            )

    log_likelihoods = noise_log_probs - dt * accumulated_div  # (N, B)

    rank_scores = log_likelihoods.clone()

    if prev_actions is not None and cfg.lambda_anchor > 0:
        sd = cfg.score_dims
        sh = cfg.score_horizon
        n = cfg.n_exec_steps

        for k in range(N):
            a_scored = actions[k, :, :sh, :sd].float()
            prev_scored = prev_actions[:, :sh, :sd].float()
            H_scored = a_scored.shape[1]
            n_overlap = min(n, H_scored - n)
            if n_overlap > 0:
                candidate_near = a_scored[:, :n_overlap, :]
                prev_tail = prev_scored[:, n:n + n_overlap, :]
                weights = torch.tensor(
                    [cfg.anchor_decay ** j for j in range(n_overlap)],
                    device=device, dtype=torch.float32,
                )
                weights = weights / weights.sum()
                sq_dist = (candidate_near - prev_tail).pow(2).mean(dim=2)  # (B, n_overlap)
                anchor_score = -(sq_dist * weights.unsqueeze(0)).sum(dim=1)  # (B,)
                rank_scores[k] += cfg.lambda_anchor * anchor_score

    best_idx = rank_scores.argmax(dim=0)  # (B,)
    batch_idx = torch.arange(B, device=device)
    best_actions = actions[best_idx, batch_idx]  # (B, H, D)

    diag = DensityDiagnostics(
        cumulative_divergence=accumulated_div[best_idx, batch_idx].mean().item(),
        log_likelihood_estimate=(
            log_likelihoods[best_idx, batch_idx].mean().item()
            - 0.5 * H * D * math.log(2 * math.pi)
        ),
        noise_log_prob=noise_log_probs[best_idx, batch_idx].mean().item(),
        density_trend="rank_mode",
        candidate_log_likelihoods=log_likelihoods[:, 0].cpu().tolist(),
        best_candidate_idx=best_idx[0].item(),
        log_likelihood_spread=(
            log_likelihoods[:, 0].max() - log_likelihoods[:, 0].min()
        ).item(),
    )

    if action_head.verbose:
        ll = log_likelihoods[:, 0]
        print(
            f"[DensityAware-Rank] N={N}  best={best_idx[0].item()}  "
            f"best_ll={ll[best_idx[0]]:.2f}  worst_ll={ll.min():.2f}  "
            f"spread={ll.max() - ll.min():.2f}"
        )

    if diagnostics_log is not None:
        diagnostics_log.append({
            "mode": "rank",
            "N": N,
            "log_likelihoods": log_likelihoods[:, 0].cpu().tolist(),
            "best_candidate_idx": best_idx[0].item(),
            "log_likelihood_spread": diag.log_likelihood_spread,
            "accumulated_divergences": accumulated_div[:, 0].cpu().tolist(),
            "noise_log_probs": noise_log_probs[:, 0].cpu().tolist(),
            "has_anchor": prev_actions is not None and cfg.lambda_anchor > 0,
        })

    return best_actions, diag


# ---------------------------------------------------------------------------
# DenoisingLab guided_fn interface (limited — no divergence estimation)
# ---------------------------------------------------------------------------


def make_density_aware_fn(cfg=None):
    """Factory for a velocity modifier compatible with DenoisingLab.denoise().

    **Limitation:** The ``guided_fn`` interface receives a single pre-computed
    velocity, so it cannot perform the batched forward pass needed for
    divergence estimation.  This function provides the guided velocity
    *scaling* component only (using a fixed scale profile), not the full
    density-aware loop.  For true divergence estimation, use
    ``denoise_with_lab()`` or ``denoise_density_aware()`` directly.

    The scale profile applies conservative amplification at later steps
    (where the velocity field is empirically more converged), matching
    the guided mode's behaviour under typical negative-divergence conditions.
    """
    if cfg is None:
        cfg = DensityAwareConfig()

    _step_scales = {
        0: 1.0,
        1: 1.0,
        2: 1.0 + 0.5 * cfg.alpha,
        3: 1.0 + cfg.alpha,
    }

    def guided_fn(actions_before, step_idx, velocity):
        if cfg.mode != "guided" or cfg.alpha == 0:
            return velocity
        scale = _step_scales.get(step_idx, 1.0)
        return velocity * scale

    return guided_fn


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg=None, diagnostics_log=None):
    """Monkey-patch the action head to use density-aware denoising.

    Replaces ``get_action_with_features()`` in-place.  Caches denoised
    actions for cross-chunk anchor consistency in rank mode.  State is
    keyed by ``client_id`` so multiple clients can share one server safely.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` to patch in-place.
        cfg: ``DensityAwareConfig``.
        diagnostics_log: Optional list to collect per-chunk diagnostics.

    Returns:
        A ``reset(options)`` callable that clears cached state.  Must be hooked
        into the policy's ``reset()`` method so stale actions from a
        previous episode don't distort anchor scoring.
    """
    if cfg is None:
        cfg = DensityAwareConfig()

    _client_state: dict = {}  # client_id -> {"prev_actions": ..., "chunk_idx": ...}

    def _get_state():
        cid = getattr(action_head, "_current_client_id", None)
        if cid not in _client_state:
            _client_state[cid] = {"prev_actions": None, "chunk_idx": 0}
        return _client_state[cid], cid

    @torch.no_grad()
    def density_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        cs, _ = _get_state()
        actions, diag = denoise_density_aware(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output,
            cfg=cfg,
            prev_actions=cs["prev_actions"],
            diagnostics_log=diagnostics_log,
        )
        cs["prev_actions"] = actions.detach()

        if diagnostics_log is not None and len(diagnostics_log) > 0:
            diagnostics_log[-1]["chunk_idx"] = cs["chunk_idx"]
        cs["chunk_idx"] += 1

        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = density_get_action_with_features

    def reset(options=None):
        cid = options.get("client_id") if options else None
        if cid is not None:
            _client_state.pop(cid, None)
        else:
            _client_state.clear()

    return reset


# ---------------------------------------------------------------------------
# Notebook convenience (DenoisingLab interface)
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None):
    """Run density-aware denoising via DenoisingLab.

    Args:
        lab: DenoisingLab instance (model loaded).
        features: BackboneFeatures from lab.encode_features().
        seed: Random seed for reproducibility.
        cfg: DensityAwareConfig.

    Returns:
        (actions, diagnostics) tuple:
            actions: torch.Tensor -- raw actions (B, action_horizon, action_dim).
            diagnostics: DensityDiagnostics.
        Decode actions with lab.decode_raw_actions(actions).
    """
    if cfg is None:
        cfg = DensityAwareConfig()

    with torch.no_grad():
        return denoise_density_aware(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            cfg=cfg, seed=seed,
        )


# ---------------------------------------------------------------------------
# Calibration utility
# ---------------------------------------------------------------------------


def calibrate_divergence_scale(lab, features_list, seeds, *, h=1e-3):
    """Profile divergence distribution across observations and steps.

    Runs monitor-mode denoising on each (features, seed) pair and collects
    per-step divergence statistics.  Use the returned per-step std as the
    ``D0`` parameter for guided mode.

    Args:
        lab: DenoisingLab instance.
        features_list: List of BackboneFeatures.
        seeds: List of random seeds (one per features entry).
        h: Perturbation scale.

    Returns:
        Dict mapping step index to {mean, std, min, max, pct_negative}.
    """
    monitor_cfg = DensityAwareConfig(mode="monitor", h=h)
    step_divergences: dict[int, list[float]] = {i: [] for i in range(4)}

    for features, seed in zip(features_list, seeds):
        _, diag = denoise_with_lab(lab, features, seed=seed, cfg=monitor_cfg)
        for i, d in enumerate(diag.divergences):
            step_divergences[i].append(d)

    stats = {}
    for step_idx in range(4):
        vals = step_divergences[step_idx]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        stats[step_idx] = {
            "mean": mean,
            "std": std,
            "min": min(vals),
            "max": max(vals),
            "pct_negative": sum(1 for v in vals if v < 0) / len(vals),
        }
    return stats
