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

    # Validate the action mask once — it's invariant across the k-loop.
    # An all-zero mask for any sample means compute_action_mask / the caller
    # produced a degenerate mask (bug upstream). Fail loudly here rather than
    # silently zero out a sample's log-prob contribution.
    valid_elements_per_sample = action_mask.sum(dim=(1, 2))
    assert (valid_elements_per_sample > 0).all(), (
        f"action_mask has sample(s) with zero valid elements: "
        f"{valid_elements_per_sample.tolist()}"
    )

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
            # NOTE: This call signature mirrors the model's pretraining forward()
            # (gr00t_n1d6.py:225-233), so the FM log-prob surrogate evaluates the
            # exact loss the model was trained with. AlternateVLDiT.forward()
            # accepts `encoder_attention_mask` but silently ignores it — the
            # cross-attention masks are built from `image_mask & backbone_attention_mask`
            # internally (dit.py:322-323). We pass it anyway for parity with the
            # pretraining forward.
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

        # --- Per-sample MSE in fp32 ---
        # Cast pred_velocity, velocity_target, and the mask to float32 BEFORE
        # computing MSE. Doing this in bf16 has two precision problems:
        #   1. bf16 has only 8 mantissa bits → element-wise (pred-target)^2 is
        #      noisy, and the noise floor swamps the small signal differences
        #      between the current LoRA-adapted policy and the reference.
        #   2. Summing ~192 (Panda 16×12) bf16 values accumulates rounding
        #      error that can drown the policy-quality difference between
        #      ref and current — making log_ratio noisy and inflating
        #      clipfrac and mean_log_ratio_abs.
        # The fp32 cast is cheap (a few hundred KB per minibatch) and keeps
        # gradients differentiable wrt the LoRA-adapted bf16 output.
        pred_v_f32 = pred_velocity.float()
        target_v_f32 = velocity_target.float()
        mask_f32 = action_mask.float()

        per_element_mse = F.mse_loss(pred_v_f32, target_v_f32, reduction="none")
        masked_mse = per_element_mse * mask_f32
        per_sample_mse = masked_mse.sum(dim=(1, 2)) / valid_elements_per_sample.float()

        log_probs_accumulated += -per_sample_mse  # already fp32

    # Average across K timestep samples
    log_probs = log_probs_accumulated / n_samples

    return log_probs


def _sample_jittered_timesteps(
    tau_centers: list[float],
    B: int,
    noise_s: float,
    device: torch.device,
    dtype: torch.dtype,
    jitter_std: float = 0.02,
) -> torch.Tensor:
    """Sample timesteps from tight Gaussians centered on user-specified τ values.

    Each center gets Gaussian jitter (std=0.02, so 95% within ±0.04). Choose centers
    to weight the FM log-prob evaluation toward the most important τ values.

    Example: Late-biased schedule [0, 0.25, 0.35, 0.5, 0.6, 0.75] has denser
    coverage in [0.5, 0.75] where velocity prediction errors have more impact
    (fewer Euler steps remaining to correct the action).

    Args:
        tau_centers: List of τ values in [0, noise_s]. K = len(tau_centers).
        B: Batch size.
        noise_s: Maximum timestep value (0.999 from model config).
        device: Torch device.
        dtype: Torch dtype.
        jitter_std: Std of the Gaussian jitter (default 0.02).

    Returns:
        timesteps: [K, B] tensor, one jittered timestep per center.
    """
    centers = torch.tensor(tau_centers, device=device, dtype=dtype)  # [K]
    K = centers.shape[0]

    # Sample from N(center, jitter_std) independently for each batch element
    jitter = torch.randn(K, B, device=device, dtype=dtype) * jitter_std
    timesteps = centers[:, None] + jitter  # [K, B]

    # Clamp to valid range
    timesteps = timesteps.clamp(min=0.0, max=noise_s)

    return timesteps


