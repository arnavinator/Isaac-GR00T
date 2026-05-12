"""Flow-Matching log-probability surrogate for GRPO.

This is the CORE algorithmic module — it provides the equivalent of
`dist.log_prob(action)` from grpo_cont.py, but for a flow-matching diffusion model.

Background:
In grpo_cont.py, the Gaussian policy gives an exact log-prob:
    log_prob = -0.5*log(2*pi) - ln(std) - 0.5*((action - mean)/std)^2

For a flow-matching model, there is NO closed-form log-probability.
Instead, we use the FM loss as a surrogate (from DPPO, Ren et al. 2024):
    log pi(action | obs) ≈ -MSE(v_theta(x_t, t | obs), velocity_target)

where:
    x_t = (1 - t) * epsilon + t * action       (noisy interpolation)
    velocity_target = action - epsilon           (true velocity field)
    v_theta = model's predicted velocity         (what the DiT outputs)

Key design decisions:
1. A SINGLE epsilon is used per action chunk, with K different tau values.
   Each action was generated from one denoising trajectory (one noise vector),
   so the surrogate should evaluate the velocity field along one path at
   multiple points — not across unrelated random paths.

2. When computing the importance ratio rho = pi_theta / pi_ref, the SAME
   (tau, epsilon) must be used for both models. This ensures the ratio reflects
   only the model quality difference, not estimation noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


def compute_fm_log_prob(
    action_head: nn.Module,
    backbone_output: dict,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    n_samples: int = 4,
    noise_beta_alpha: float = 1.5,
    noise_beta_beta: float = 1.0,
    noise_s: float = 1.0,
) -> torch.Tensor:
    """Compute FM log-probability surrogate for a batch of action chunks.

    This mirrors the forward() method of Gr00tN1d6ActionHead (gr00t_n1d6.py:149-257)
    but returns PER-SAMPLE loss instead of batch-mean, and accepts pre-specified
    (t, noise) for importance ratio consistency.

    Uses a SINGLE noise vector epsilon and K different timesteps tau.
    Each tau probes the velocity field at a different point along the same
    interpolation path (epsilon → action), giving K estimates that are
    averaged for variance reduction.

    Args:
        action_head: The Gr00tN1d6ActionHead module (with or without LoRA).
        backbone_output: Dict/BatchFeature from Eagle backbone containing:
            - backbone_features: [B, seq_len, 2048]
            - backbone_attention_mask, image_mask (optional)
        state_features: [B, state_horizon, 1536] pre-encoded state embeddings.
        embodiment_id: [B] embodiment IDs (e.g., 13 for PandaOmron).
        actions: [B, action_horizon, action_dim] the action chunk to evaluate.
        action_mask: [B, action_horizon, action_dim] binary mask for valid dims.
        timesteps: [K, B] pre-specified diffusion timesteps (continuous in [0, noise_s]).
            If None, samples K fresh timesteps from Beta distribution.
        noise: [B, action_horizon, action_dim] SINGLE noise vector for all K samples.
            If None, samples one fresh noise tensor.
        n_samples: Number of timestep samples for variance reduction (K).
        noise_beta_alpha: Alpha param for Beta distribution (default 1.5).
        noise_beta_beta: Beta param for Beta distribution (default 1.0).
        noise_s: Scaling factor for timestep (default from model config: 0.999).

    Returns:
        log_probs: [B] tensor of FM log-probability surrogates (negative MSE).
    """
    B = actions.shape[0]
    device = actions.device
    dtype = actions.dtype

    # Get vision-language embeddings from backbone output
    vl_embeds = backbone_output["backbone_features"]

    # Single noise vector shared across all K timestep samples.
    # This evaluates the velocity field along ONE interpolation path at K points,
    # rather than along K unrelated paths (which would add variance without benefit).
    if noise is not None:
        eps = noise  # [B, action_horizon, action_dim]
    else:
        eps = torch.randn_like(actions)

    # The velocity target is constant across all tau (it's a property of the
    # (action, noise) pair, not the interpolation point)
    velocity_target = actions - eps

    # Access masks safely for the DiT forward pass
    _image_mask = backbone_output.get("image_mask") if hasattr(backbone_output, "get") else getattr(backbone_output, "image_mask", None)
    _backbone_attn_mask = backbone_output.get("backbone_attention_mask") if hasattr(backbone_output, "get") else getattr(backbone_output, "backbone_attention_mask", None)

    # Accumulate log-probs across K timestep samples for variance reduction
    log_probs_accumulated = torch.zeros(B, device=device, dtype=torch.float32)

    for k in range(n_samples):
        # --- Sample or use pre-specified timestep ---
        if timesteps is not None:
            t = timesteps[k]  # [B]
        else:
            beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
            t = beta_dist.sample([B]).to(device=device, dtype=dtype)
            t = (1 - t) * noise_s

        # --- Construct noisy trajectory at this timestep ---
        # x_t = (1 - t) * epsilon + t * action
        t_expanded = t[:, None, None]  # [B, 1, 1]
        noisy_trajectory = (1 - t_expanded) * eps + t_expanded * actions

        # --- Forward pass through DiT ---
        num_timestep_buckets = action_head.num_timestep_buckets
        t_discretized = (t * num_timestep_buckets).long()

        action_features = action_head.action_encoder(
            noisy_trajectory, t_discretized, embodiment_id
        )

        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if action_head.config.use_alternate_vl_dit:
            model_output, _ = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=_backbone_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=_image_mask,
                backbone_attention_mask=_backbone_attn_mask,
            )
        else:
            model_output, _ = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=_backbone_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        # Decode velocity prediction
        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -actions.shape[1]:]

        # --- Per-sample MSE ---
        per_element_mse = F.mse_loss(pred_velocity, velocity_target, reduction="none")
        masked_mse = per_element_mse * action_mask
        valid_elements_per_sample = action_mask.sum(dim=(1, 2)).clamp(min=1.0)
        per_sample_mse = masked_mse.sum(dim=(1, 2)) / valid_elements_per_sample

        log_probs_accumulated += (-per_sample_mse).float()

    # Average across K timestep samples
    log_probs = log_probs_accumulated / n_samples

    return log_probs


def _sample_inference_jittered_timesteps(
    n_steps: int,
    B: int,
    noise_s: float,
    device: torch.device,
    dtype: torch.dtype,
    jitter_std: float = 0.02,
) -> torch.Tensor:
    """Sample timesteps from tight Gaussians centered on the inference schedule.

    The inference denoising uses deterministic timesteps (0, 0.25, 0.5, 0.75).
    These are the points where the velocity field prediction matters most for
    actual action generation. We add small Gaussian jitter for stochasticity
    (prevents overfitting to exact grid points) while staying close.

    Args:
        n_steps: Number of inference timesteps (typically 4).
        B: Batch size.
        noise_s: Maximum timestep value (0.999 from model config).
        device: Torch device.
        dtype: Torch dtype.
        jitter_std: Std of the Gaussian jitter (0.02 → 95% within ±0.04).

    Returns:
        timesteps: [K, B] tensor, one jittered timestep per inference step.
    """
    # Inference schedule: 0, 1/N, 2/N, ..., (N-1)/N
    centers = torch.arange(n_steps, device=device, dtype=dtype) / n_steps  # [K]

    # Sample from N(center, jitter_std) independently for each batch element
    jitter = torch.randn(n_steps, B, device=device, dtype=dtype) * jitter_std
    timesteps = centers[:, None] + jitter  # [K, B]

    # Clamp to valid range: must be >= 0 (especially for the τ=0 center)
    # and <= noise_s (the model's maximum training timestep)
    timesteps = timesteps.clamp(min=0.0, max=noise_s)

    return timesteps


def compute_fm_log_prob_pair(
    current_action_head: nn.Module,
    ref_action_head: nn.Module,
    backbone_output: dict,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    n_samples: int = 4,
    noise_s: float | None = None,
    jitter_std: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FM log-probs for both current and reference policy simultaneously.

    Uses a SINGLE shared noise vector and K shared timesteps for both models.
    Timesteps are sampled from tight Gaussians centered on the inference schedule
    (0, 0.25, 0.5, 0.75) — the actual denoising points where velocity prediction
    quality matters most for action generation.

    Args:
        current_action_head: Model being trained (with LoRA adapters).
        ref_action_head: Frozen reference model (for KL anchor).
        backbone_output: Shared backbone features (same for both models).
        state_features: Shared state embeddings (same for both models).
        embodiment_id: [B] embodiment IDs.
        actions: [B, action_horizon, action_dim] actions to evaluate.
        action_mask: [B, action_horizon, action_dim] valid dimension mask.
        n_samples: Number of timestep samples (K=4 matches the 4 inference steps).
        noise_s: Timestep scaling factor. If None, reads from config.
        jitter_std: Std of Gaussian jitter around inference timesteps (default 0.02).

    Returns:
        Tuple of (current_log_probs, ref_log_probs), each [B] tensors.
    """
    B = actions.shape[0]
    device = actions.device
    dtype = actions.dtype

    config = current_action_head.config
    if noise_s is None:
        noise_s = getattr(config, "noise_s", 0.999)

    # Timesteps: jittered around the 4 inference denoising points
    timesteps = _sample_inference_jittered_timesteps(
        n_steps=n_samples, B=B, noise_s=noise_s,
        device=device, dtype=dtype, jitter_std=jitter_std,
    )  # [K, B]

    # ONE shared noise vector — evaluates both models along the
    # same interpolation path, so the ratio isolates model quality difference
    shared_noise = torch.randn_like(actions)  # [B, action_horizon, action_dim]

    # Compute log-probs with shared (timesteps, noise)
    current_log_probs = compute_fm_log_prob(
        action_head=current_action_head,
        backbone_output=backbone_output,
        state_features=state_features,
        embodiment_id=embodiment_id,
        actions=actions,
        action_mask=action_mask,
        timesteps=timesteps,
        noise=shared_noise,
        n_samples=n_samples,
    )

    # Reference model: no gradient needed (frozen)
    with torch.no_grad():
        ref_log_probs = compute_fm_log_prob(
            action_head=ref_action_head,
            backbone_output=backbone_output,
            state_features=state_features,
            embodiment_id=embodiment_id,
            actions=actions,
            action_mask=action_mask,
            timesteps=timesteps,
            noise=shared_noise,
            n_samples=n_samples,
        )

    return current_log_probs, ref_log_probs
