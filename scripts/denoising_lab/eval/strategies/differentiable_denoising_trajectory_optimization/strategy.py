"""Differentiable Denoising Trajectory Optimization (DDTO) for GR00T N1.6.

Optimises the initial noise ε via 1-step backprop through the DiT, computing
quality losses on the fully-extrapolated proxy a_1.0* = ε + v_0 (signal-
dominated in rectified flow), then re-denoises from the optimised noise with
a standard 4-step Euler integration.  The DiT weights are completely frozen —
only the noise input is updated.

Three phases:
  1. Forward step 0 under torch.enable_grad() → compute quality loss on the
     fully-extrapolated proxy → backprop to get ∂L/∂ε (+ optional HVP for the
     on-mode regulariser).
  2. Normalised gradient step on ε, re-project to the Gaussian norm-sphere.
  3. Full 4-step Euler denoising from ε* (no grad).

Total: 5 NFEs (1 with grad + 4 without) + 1–2 backward passes through 1 DiT
call.  Variant A (lambda_mode > 0) includes the Hessian-vector product for on-
mode regularisation; Variant B (lambda_mode = 0) skips it.

Usage (server):
    from strategy import patch_action_head
    reset_fn = patch_action_head(policy.model.action_head)

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab, DDTOConfig
    cfg = DDTOConfig(eta=0.1)
    actions, diagnostics = denoise_with_lab(lab, features, seed=42, cfg=cfg)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.feature_extraction_utils import BatchFeature

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def _math_sdp_context():
        return sdpa_kernel(SDPBackend.MATH)
except ImportError:
    def _math_sdp_context():
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DDTOConfig:
    """Tunable parameters for Differentiable Denoising Trajectory Optimization."""

    lambda_smooth: float = 1.0
    """Weight for temporal smoothness loss (penalises jerky predicted actions)."""

    lambda_anchor: float = 0.5
    """Weight for anchor consistency with previous chunk."""

    lambda_mode: float = 0.1
    """Weight for on-mode regulariser (gradient-norm penalty).  Set to 0 to
    disable (Variant B — simpler, no Hessian-vector product)."""

    anchor_decay: float = 0.5
    """Per-step geometric decay for distance-weighted anchor scoring.  Step 0
    of the overlap gets weight 1.0, step j gets weight ``anchor_decay ** j``."""

    n_exec_steps: int = 8
    """Number of action steps executed per chunk (overlap region size)."""

    eta: float = 0.1
    """Normalised gradient step size for the noise update."""

    num_steps: int = 4
    """Number of denoising steps (standard 4-step Euler)."""

    n_action_dims: int | None = 12
    """Number of active action dims for loss computation.  The raw tensor is
    padded to 128 dims for multi-embodiment support; only the first
    ``n_action_dims`` carry signal.  Set to ``None`` to use all dims.
    PandaOmron: 12."""

    n_action_horizon: int | None = 16
    """Number of active timesteps for loss computation.  The raw tensor is
    padded to 50 timesteps; only the first ``n_action_horizon`` are decoded
    into physical actions.  Set to ``None`` to use all timesteps.
    PandaOmron: 16."""


# ---------------------------------------------------------------------------
# Shared helper: evaluate DiT velocity field
# ---------------------------------------------------------------------------


def _evaluate_velocity(action_head, actions, t_bucket, vl_embeds, state_features,
                       embodiment_id, backbone_output):
    """Forward pass through the DiT to get the predicted velocity.

    No ``@torch.no_grad()`` decorator — gradient computation is controlled by
    the caller's context (``torch.enable_grad()`` in Phase 1, outer no_grad
    in Phase 3).
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


def _clone_backbone_output(backbone_output):
    """Clone tensor fields in a BatchFeature to escape inference-mode."""
    cloned = BatchFeature()
    for key, val in backbone_output.items():
        if isinstance(val, torch.Tensor):
            cloned[key] = val.clone()
        else:
            cloned[key] = val
    return cloned


# ---------------------------------------------------------------------------
# Core denoising function
# ---------------------------------------------------------------------------


def denoise_ddto(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *, cfg=None, seed=None, prev_actions=None,
):
    """Differentiable Denoising Trajectory Optimisation (DDTO).

    Optimises the initial noise via 1-step backprop through the DiT, then
    re-denoises from the optimised noise with standard Euler steps.

    Args:
        action_head: Gr00tN1d6ActionHead instance.
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output (BatchFeature).
        cfg: DDTOConfig.  Defaults to Variant A with default hyperparams.
        seed: Random seed for initial noise generation.
        prev_actions: Optional (B, H, D) denoised actions from previous chunk
            for anchor consistency scoring.

    Returns:
        (actions, diagnostics) tuple:
            actions: Denoised actions (B, action_horizon, action_dim).
            diagnostics: Dict with quality_loss_before, quality_loss_after,
                gradient_norm, mode_gradient_norm, noise_shift_norm.
    """
    if cfg is None:
        cfg = DDTOConfig()

    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    H = action_head.action_horizon
    D = action_head.action_dim
    num_buckets = action_head.num_timestep_buckets
    dt = 1.0 / cfg.num_steps

    # --- Sample initial noise ---
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    epsilon = torch.randn(B, H, D, dtype=dtype, device=device, generator=gen)

    diagnostics = {
        "quality_loss_before": None,
        "quality_loss_after": None,
        "gradient_norm": None,
        "mode_gradient_norm": None,
        "noise_shift_norm": None,
    }

    # ================================================================
    # Phase 1: 1-step forward with grad → fully-extrapolated proxy
    # ================================================================

    # Need both: inference_mode(False) to exit the @torch.inference_mode()
    # on get_action(), and enable_grad() to override the @torch.no_grad()
    # on patched_get_action_with_features().
    # eps MUST be created inside this block — tensors created under
    # inference_mode are "inference tensors" that cannot participate in
    # autograd (PyTorch raises "Inference tensors cannot be saved for
    # backward").
    with torch.inference_mode(False), torch.enable_grad():
        eps = epsilon.detach().clone().requires_grad_(True)

        # All tensors from the backbone were created under inference_mode and
        # are "inference tensors".  Clone them so autograd can save activations
        # for the backward pass.  Only needed for the Phase-1 grad call — the
        # no-grad Phase-3 Euler steps use the originals directly.
        vl_grad = vl_embeds.clone()
        sf_grad = state_features.clone()
        eid_grad = embodiment_id.clone()
        bo_grad = _clone_backbone_output(backbone_output)

        # Single DiT forward pass (1 NFE) — gradients tracked.
        # When lambda_mode > 0 (Variant A), the HVP requires double-backward
        # through the DiT's attention layers.  Flash / mem-efficient SDPA
        # kernels don't implement second-order derivatives; the math backend
        # does.  Force it for the forward pass so the backward graph is fully
        # differentiable.  Phase-3 Euler steps (no grad) still use the fast
        # backend.  Applied unconditionally — negligible cost for one pass.
        with _math_sdp_context():
            v0 = _evaluate_velocity(
                action_head, eps, 0,
                vl_grad, sf_grad, eid_grad, bo_grad,
            )

        # Fully-extrapolated proxy for quality scoring (signal-dominated).
        # In rectified flow: v(ε,0) ≈ data − ε, so ε + v(ε,0) ≈ data_predicted.
        a_1_star = eps + 1.0 * v0  # (B, H, D)

        # Slice to active dims — the raw (B, 50, 128) tensor is padded for
        # multi-embodiment support; only the first n_action_horizon timesteps
        # and n_action_dims action dims carry signal.  Computing losses on
        # padding wastes the gradient on meaningless dimensions.
        h_end = cfg.n_action_horizon if cfg.n_action_horizon is not None else H
        d_end = cfg.n_action_dims if cfg.n_action_dims is not None else D
        a_active = a_1_star[:, :h_end, :d_end]  # (B, h_end, d_end)

        # --- Quality loss on the proxy ---
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        # Smoothness: temporal roughness of predicted trajectory
        diffs = a_active[:, 1:, :] - a_active[:, :-1, :]
        smooth_loss = diffs.pow(2).sum(dim=(1, 2)).mean()
        loss = loss + cfg.lambda_smooth * smooth_loss

        # Anchor consistency with previous chunk (overlap-region, decay-weighted)
        if prev_actions is not None and cfg.lambda_anchor > 0:
            n_overlap = min(cfg.n_exec_steps, h_end - cfg.n_exec_steps)

            if n_overlap > 0:
                candidate_near = a_active[:, :n_overlap, :]
                prev_tail = prev_actions[:, cfg.n_exec_steps:cfg.n_exec_steps + n_overlap, :d_end]

                weights = torch.tensor(
                    [cfg.anchor_decay ** j for j in range(n_overlap)],
                    device=device, dtype=torch.float32,
                )
                weights = weights / weights.sum()

                sq_dist = (candidate_near - prev_tail).pow(2).sum(dim=2)  # (B, n_overlap)
                anchor_loss = (sq_dist.float() * weights.unsqueeze(0)).sum(dim=1).mean()
                loss = loss + cfg.lambda_anchor * anchor_loss

        diagnostics["quality_loss_before"] = loss.item()

        # --- First backward: quality gradient ---
        # Chain rule: g = ∂L/∂a_1.0* · (I + ∂v_0/∂ε)
        # The Jacobian appears with coefficient 1.0, giving 4× sharper mode
        # discrimination than the a_0.25 proxy alternative.
        g = torch.autograd.grad(loss, eps, create_graph=(cfg.lambda_mode > 0))[0]
        diagnostics["gradient_norm"] = g.float().norm().item()

        # --- Optional: on-mode regulariser via gradient-norm penalty ---
        if cfg.lambda_mode > 0:
            g_norm_sq = g.float().pow(2).sum()
            g_mode = torch.autograd.grad(g_norm_sq, eps)[0]
            diagnostics["mode_gradient_norm"] = g_mode.float().norm().item()
            g_total = g + cfg.lambda_mode * g_mode
        else:
            g_total = g

    # ================================================================
    # Phase 2: Update noise
    # ================================================================
    g_total_f = g_total.float()
    g_norm = g_total_f.norm()

    if g_norm > 1e-10:
        eps_opt = eps.float() - cfg.eta * (g_total_f / g_norm)

        # Re-project to Gaussian norm-sphere (per-sample)
        orig_norms = epsilon.float().norm(dim=(-1, -2), keepdim=True)
        opt_norms = eps_opt.norm(dim=(-1, -2), keepdim=True)
        eps_opt = eps_opt * (orig_norms / (opt_norms + 1e-10))
        eps_opt = eps_opt.to(dtype)
    else:
        eps_opt = epsilon

    diagnostics["noise_shift_norm"] = (eps_opt.float() - epsilon.float()).norm().item()

    if action_head.verbose:
        variant = "A (mode reg)" if cfg.lambda_mode > 0 else "B (no mode reg)"
        print(f"[DDTO] Variant {variant}  "
              f"loss={diagnostics['quality_loss_before']:.4f}  "
              f"||g||={diagnostics['gradient_norm']:.4f}  "
              f"shift={diagnostics['noise_shift_norm']:.4f}")

    # ================================================================
    # Phase 3: Full 4-step Euler from optimised noise (no grad)
    # ================================================================
    actions = eps_opt
    for step in range(cfg.num_steps):
        tau = step / float(cfg.num_steps)
        t_bucket = int(tau * num_buckets)

        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )
        actions = actions + dt * velocity

        if action_head.verbose:
            print(f"[DDTO] Step {step}/{cfg.num_steps}  "
                  f"tau={tau:.3f}  bucket={t_bucket}  "
                  f"a_norm={actions.float().norm():.4f}  "
                  f"v_norm={velocity.float().norm():.4f}")

    # Diagnostics: smoothness of final output (active dims only)
    diffs_final = actions[:, 1:h_end, :d_end] - actions[:, :h_end - 1, :d_end]
    diagnostics["quality_loss_after"] = (
        diffs_final.float().pow(2).sum(dim=(1, 2)).mean().item()
    )

    if action_head.verbose:
        af = actions.float()
        print(f"[DDTO] Final  shape={tuple(actions.shape)}  "
              f"mean={af.mean():.4f}  std={af.std():.4f}  "
              f"loss_after={diagnostics['quality_loss_after']:.4f}")

    return actions, diagnostics


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg=None):
    """Monkey-patch the action head to use DDTO.

    Replaces ``get_action_with_features()`` in-place.  Caches the denoised
    actions across calls to enable anchor consistency scoring.

    Returns:
        A ``reset()`` callable that clears the cached state.  **Must** be
        hooked into the policy's ``reset()`` method so that stale actions
        from a previous episode don't distort anchor scoring.
    """
    if cfg is None:
        cfg = DDTOConfig()

    _prev_actions = [None]  # mutable closure state for cross-chunk caching

    @torch.no_grad()
    def patched_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        # torch.enable_grad() inside denoise_ddto Phase 1 overrides
        # the outer no_grad for the 1-step backprop.
        actions, _diagnostics = denoise_ddto(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, cfg=cfg,
            prev_actions=_prev_actions[0],
        )
        _prev_actions[0] = actions.detach()
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = patched_get_action_with_features

    def reset():
        """Clear cached prev_actions (call on episode reset)."""
        _prev_actions[0] = None

    return reset


# ---------------------------------------------------------------------------
# Notebook convenience (DenoisingLab interface)
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None, prev_actions=None):
    """Run DDTO via DenoisingLab.

    Args:
        lab: DenoisingLab instance.
        features: BackboneFeatures from lab.encode_features().
        seed: Random seed for reproducibility.
        cfg: DDTOConfig.
        prev_actions: Optional (B, H, D) denoised actions from previous chunk.

    Returns:
        (actions, diagnostics) tuple:
            actions: torch.Tensor -- raw actions (B, action_horizon, action_dim).
            diagnostics: dict with quality loss and gradient info.
        Decode actions with lab.decode_raw_actions(actions).
    """
    if cfg is None:
        cfg = DDTOConfig()

    with torch.no_grad():
        return denoise_ddto(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            cfg=cfg, seed=seed, prev_actions=prev_actions,
        )
