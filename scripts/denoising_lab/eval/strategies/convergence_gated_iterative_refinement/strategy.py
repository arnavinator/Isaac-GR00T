"""Convergence-Gated Iterative Refinement for GR00T N1.6.

Phase-separated denoising with per-position convergence monitoring and
adaptive execution horizon.

Phase 1: 2 standard Euler steps (tau=0, 250) — structural denoising.
Phase 2: Up to K_max iterations at fixed tau_refine — iterative refinement
         across the full action horizon with early stopping.
Phase 3: Adaptive execution horizon from per-position convergence map.

Total NFEs: 4 (easy, converges at K_min) to 2+K_max (hard, budget exhausted).

Usage (server):
    from strategy import patch_action_head, ConvergenceGatedConfig
    cfg = ConvergenceGatedConfig(theta=0.5)
    patch_action_head(policy.model.action_head, cfg=cfg)

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab, ConvergenceGatedConfig
    cfg = ConvergenceGatedConfig(theta=0.5)
    actions, diag = denoise_with_lab(lab, features, seed=42, cfg=cfg)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceGatedConfig:
    """Tunable parameters for convergence-gated iterative refinement."""

    n_exec: int = 8
    """Standard execution horizon (n_action_steps executed per chunk)."""

    n_min: int = 2
    """Minimum execution horizon (safety floor for adaptive horizon)."""

    tau_refine: int = 750
    """Fixed timestep bucket for the Phase 2 refinement loop."""

    dt_refine: float = 0.25
    """Euler step size for Phase 2 refinement updates."""

    theta: float = 0.5
    """Per-position convergence threshold on velocity L2 norm."""

    K_max: int = 6
    """Maximum number of Phase 2 refinement iterations (budget cap)."""

    K_min: int = 2
    """Minimum iterations before early stopping is allowed."""

    clamp_uncertain: bool = True
    """When True, replace uncertain tail positions (beyond adaptive_n_exec)
    with the last converged action.  Server-side approximation of adaptive
    execution that works without client protocol changes."""

    n_refine_horizon: int = 16
    """Number of positions to refine and monitor for convergence.  Should
    match the embodiment's meaningful action horizon.  Default 16 covers
    PandaOmron.  Positions beyond this are multi-embodiment padding."""


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class RefinementDiagnostics:
    """Rich diagnostic output from convergence-gated refinement."""

    phase1_nfe: int = 2
    phase2_nfe: int = 0
    total_nfe: int = 2
    converged: bool = False
    convergence_iteration: int | None = None
    position_convergence: torch.Tensor | None = None
    convergence_history: list[torch.Tensor] = field(default_factory=list)
    adaptive_n_exec: int = 8
    original_n_exec: int = 8
    position_labels: list[str] = field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Core denoising function
# ---------------------------------------------------------------------------


def denoise_convergence_gated(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *, cfg=None, seed=None, diagnostics_log=None,
):
    """Phase-separated denoising with convergence-gated iterative refinement.

    Args:
        action_head: Gr00tN1d6ActionHead instance.
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output (BatchFeature).
        cfg: ConvergenceGatedConfig.  Defaults to standard parameters.
        seed: Optional random seed for reproducible noise generation.
        diagnostics_log: Optional list to append RefinementDiagnostics to.

    Returns:
        (actions, diagnostics) tuple:
            actions: Denoised actions (B, action_horizon, action_dim).
            diagnostics: RefinementDiagnostics with convergence info.
    """
    if cfg is None:
        cfg = ConvergenceGatedConfig()

    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    H = action_head.action_horizon
    D = action_head.action_dim

    diag = RefinementDiagnostics(original_n_exec=cfg.n_exec)

    # Initial noise
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None
    actions = torch.randn(B, H, D, dtype=dtype, device=device, generator=gen)

    # ==================================================================
    # Phase 1: Structural denoising (2 standard Euler steps)
    # ==================================================================
    dt_structural = 0.25

    for t_bucket in [0, 250]:
        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )
        actions = actions + dt_structural * velocity

    diag.phase1_nfe = 2

    # ==================================================================
    # Phase 2: Iterative refinement at fixed timestep (full horizon)
    # ==================================================================
    refine_h = min(cfg.n_refine_horizon, H)

    for k in range(cfg.K_max):
        velocity = _evaluate_velocity(
            action_head, actions, cfg.tau_refine,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )

        actions = actions + cfg.dt_refine * velocity

        diag.phase2_nfe += 1

        per_pos_rho = velocity[:, :refine_h, :].float().norm(dim=-1).mean(dim=0)
        diag.convergence_history.append(per_pos_rho.detach().cpu())

        max_rho = per_pos_rho.max().item()

        if k >= cfg.K_min - 1 and max_rho < cfg.theta:
            diag.converged = True
            diag.convergence_iteration = k + 1
            break

    # ==================================================================
    # Phase 3: Adaptive execution horizon
    # ==================================================================
    final_rho = diag.convergence_history[-1]
    diag.position_convergence = final_rho

    converged_mask = final_rho < cfg.theta
    adaptive_n = 0
    for h in range(min(cfg.n_exec, len(converged_mask))):
        if converged_mask[h]:
            adaptive_n = h + 1
        else:
            break

    diag.adaptive_n_exec = max(cfg.n_min, min(adaptive_n, cfg.n_exec))

    diag.position_labels = [
        "converged" if final_rho[h] < cfg.theta else "uncertain"
        for h in range(min(refine_h, len(final_rho)))
    ]

    diag.total_nfe = diag.phase1_nfe + diag.phase2_nfe

    # Server-side adaptive approximation: replace uncertain tail with hold-steady
    if cfg.clamp_uncertain and diag.adaptive_n_exec < cfg.n_exec:
        last_good = diag.adaptive_n_exec - 1
        actions = actions.clone()
        actions[:, diag.adaptive_n_exec:cfg.n_exec, :] = (
            actions[:, last_good:last_good + 1, :]
        )

    if diagnostics_log is not None:
        diagnostics_log.append(diag)

    return actions, diag


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg=None, diagnostics_log=None):
    """Monkey-patch the action head to use convergence-gated iterative refinement.

    Replaces ``get_action_with_features()`` in-place.  Stateless per chunk —
    no cross-chunk caching, so no reset function is needed.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` to patch in-place.
        cfg: ``ConvergenceGatedConfig``.
        diagnostics_log: Optional list to collect per-chunk diagnostics.
    """
    if cfg is None:
        cfg = ConvergenceGatedConfig()

    @torch.no_grad()
    def patched_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions, diag = denoise_convergence_gated(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, cfg=cfg,
            diagnostics_log=diagnostics_log,
        )

        action_head._last_cgir_diagnostics = diag

        if action_head.verbose:
            labels = " ".join(
                "C" if l == "converged" else "U" for l in diag.position_labels
            )
            print(
                f"[CGIR] NFEs={diag.total_nfe} "
                f"(P1={diag.phase1_nfe}, P2={diag.phase2_nfe}) | "
                f"converged={diag.converged}"
                + (f" @iter={diag.convergence_iteration}" if diag.converged else "")
                + f" | horizon={diag.adaptive_n_exec}/{diag.original_n_exec}"
                + f" | positions=[{labels}]"
            )

        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = patched_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience (DenoisingLab interface)
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None, diagnostics_log=None):
    """Run convergence-gated iterative refinement via DenoisingLab.

    Args:
        lab: DenoisingLab instance.
        features: BackboneFeatures from lab.encode_features().
        seed: Random seed for reproducibility.
        cfg: ConvergenceGatedConfig.
        diagnostics_log: Optional list to collect per-call diagnostics.

    Returns:
        (actions, diagnostics) tuple:
            actions: torch.Tensor -- raw actions (B, action_horizon, action_dim).
            diagnostics: RefinementDiagnostics with convergence info.
        Decode actions with lab.decode_raw_actions(actions).
    """
    if cfg is None:
        cfg = ConvergenceGatedConfig()

    with torch.no_grad():
        return denoise_convergence_gated(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            cfg=cfg, seed=seed, diagnostics_log=diagnostics_log,
        )
