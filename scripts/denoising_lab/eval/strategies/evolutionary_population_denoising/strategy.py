"""Evolutionary Population Denoising with Fitness Selection for GR00T N1.6.

Maintains a population of K action candidates throughout all 4 denoising steps.
At each step: (1) advance all candidates via batched Euler, (2) score fitness
using analytic proxies, (3) apply tournament selection + crossover + annealed
mutation.  The final population's best candidate is the output.

Key fitness signals:
- Temporal smoothness: penalise jerky action trajectories
- Velocity magnitude: lower = more confident prediction
- Inter-particle consensus: reward particles whose velocity agrees with the
  population mean (novel self-consistency signal)

Total NFEs: K * 4 (batched to 4 sequential passes with batch size K*B).

Usage (server):
    from strategy import patch_action_head, EvolutionaryConfig
    patch_action_head(policy.model.action_head, cfg=EvolutionaryConfig(K=8))

Usage (notebook with DenoisingLab):
    from strategy import denoise_with_lab, EvolutionaryConfig
    cfg = EvolutionaryConfig(K=8)
    actions = denoise_with_lab(lab, features, seed=42, cfg=cfg)
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
class EvolutionaryConfig:
    """Tunable parameters for evolutionary population denoising."""

    K: int = 8
    """Population size (must be even for tournament selection)."""

    lambda_smooth: float = 1.0
    """Fitness weight: temporal smoothness."""

    lambda_velocity: float = 0.1
    """Fitness weight: velocity magnitude (lower = better)."""

    lambda_consensus: float = 0.3
    """Fitness weight: inter-particle velocity agreement."""

    sigma_0: float = 0.02
    """Initial mutation strength (annealed by denoising progress)."""

    crossover_lo: float = 0.3
    """Lower bound for crossover blending coefficient."""

    crossover_hi: float = 0.7
    """Upper bound for crossover blending coefficient."""

    num_steps: int = 4
    """Number of denoising steps."""


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
# Fitness scoring
# ---------------------------------------------------------------------------


def _score_population(population, velocities, cfg):
    """Compute fitness for each particle in the population.

    Args:
        population: (K, B, H, D) action tensors.
        velocities: (K, B, H, D) velocity predictions.
        cfg: EvolutionaryConfig.

    Returns:
        fitness: (K, B) scores (higher = better).
    """
    K, B = population.shape[0], population.shape[1]
    device = population.device
    fitness = torch.zeros(K, B, device=device, dtype=torch.float32)

    for k in range(K):
        a = population[k].float()  # (B, H, D)
        v = velocities[k].float()  # (B, H, D)

        # Temporal smoothness: penalise jerky actions
        diffs = a[:, 1:, :] - a[:, :-1, :]
        fitness[k] -= cfg.lambda_smooth * (diffs ** 2).sum(dim=(1, 2))

        # Velocity magnitude: lower = more confident prediction
        fitness[k] -= cfg.lambda_velocity * (v ** 2).sum(dim=(1, 2))

    # Inter-particle consensus: reward agreement with population mean velocity
    mean_v = velocities.float().mean(dim=0)  # (B, H, D)
    for k in range(K):
        cos_sim = F.cosine_similarity(
            velocities[k].float().reshape(B, -1),
            mean_v.reshape(B, -1),
            dim=1,
        )  # (B,)
        fitness[k] += cfg.lambda_consensus * cos_sim

    return fitness


# ---------------------------------------------------------------------------
# Selection + reproduction
# ---------------------------------------------------------------------------


def _select_and_reproduce(population, fitness, tau, cfg):
    """Tournament selection, crossover, and annealed mutation.

    Args:
        population: (K, B, H, D) current population.
        fitness: (K, B) fitness scores.
        tau: Current denoising progress (for mutation annealing).
        cfg: EvolutionaryConfig.

    Returns:
        new_population: (K, B, H, D).
    """
    K, B, H, D = population.shape
    device, dtype = population.device, population.dtype
    half_K = K // 2

    # Tournament selection: keep top K/2 per batch element
    _, top_indices = fitness.topk(half_K, dim=0)  # (K/2, B)

    # Gather surviving particles using advanced indexing
    # top_indices[rank, b] gives the particle index for rank `rank` in batch `b`
    batch_idx = torch.arange(B, device=device).unsqueeze(0).expand(half_K, -1)
    survivors = population[top_indices, batch_idx]  # (K/2, B, H, D)

    # Duplicate survivors to restore population size
    new_pop = torch.cat([survivors, survivors.clone()], dim=0)  # (K, B, H, D)

    # Crossover: blend pairs in the second half with random partners
    for k in range(half_K, K):
        partner_idx = torch.randint(0, half_K, (1,), device=device).item()
        beta = torch.rand(1, device=device).item()
        beta = cfg.crossover_lo + beta * (cfg.crossover_hi - cfg.crossover_lo)
        new_pop[k] = beta * new_pop[k] + (1 - beta) * new_pop[partner_idx]

    # Annealed mutation: large early (noisy), small late (converged)
    sigma = cfg.sigma_0 * (1 - tau)
    if sigma > 1e-6:
        mutation = torch.randn(half_K, B, H, D, dtype=dtype, device=device) * sigma
        new_pop[half_K:] = new_pop[half_K:] + mutation

    return new_pop


# ---------------------------------------------------------------------------
# Core denoising function
# ---------------------------------------------------------------------------


def denoise_evolutionary(action_head, vl_embeds, state_features, embodiment_id,
                         backbone_output, *, cfg=None, seed=None):
    """Evolutionary population denoising with per-step fitness selection.

    Args:
        action_head: Gr00tN1d6ActionHead instance.
        vl_embeds: Vision-language embeddings (B, seq_len, 2048).
        state_features: Encoded state (B, state_horizon, hidden_dim).
        embodiment_id: Embodiment IDs (B,).
        backbone_output: Full backbone output.
        cfg: EvolutionaryConfig.
        seed: Random seed.

    Returns:
        Denoised actions (B, action_horizon, action_dim) from the best particle.
    """
    if cfg is None:
        cfg = EvolutionaryConfig()

    K = cfg.K
    if K < 2 or K % 2 != 0:
        raise ValueError(
            f"Population size K must be a positive even number, got {K}. "
            f"Tournament selection keeps K/2 survivors and duplicates them."
        )
    B = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    H = action_head.action_horizon
    D = action_head.action_dim
    num_buckets = action_head.num_timestep_buckets
    num_steps = cfg.num_steps
    dt = 1.0 / num_steps

    # Initialize population from noise
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None
    population = torch.randn(K, B, H, D, dtype=dtype, device=device, generator=gen)

    # Pre-replicate conditioning for batched forward passes
    flat_vl = vl_embeds.repeat(K, 1, 1)
    flat_state = state_features.repeat(K, 1, 1)
    flat_emb = embodiment_id.repeat(K)
    flat_backbone = _repeat_backbone_output(backbone_output, K)

    last_fitness = None

    for step in range(num_steps):
        tau = step / float(num_steps)
        t_bucket = int(tau * num_buckets)

        # --- 1. Advance all particles (single batched forward pass) ---
        flat_pop = population.reshape(K * B, H, D)

        velocity_flat = _evaluate_velocity(
            action_head, flat_pop, t_bucket,
            flat_vl, flat_state, flat_emb, flat_backbone,
        )
        flat_pop = flat_pop + dt * velocity_flat

        velocities = velocity_flat.reshape(K, B, H, D)
        population = flat_pop.reshape(K, B, H, D)

        # --- 2. Score all particles ---
        fitness = _score_population(population, velocities, cfg)
        last_fitness = fitness

        if action_head.verbose:
            best_score = fitness.max(dim=0).values.mean().item()
            worst_score = fitness.min(dim=0).values.mean().item()
            print(f"[Evo] Step {step}/{num_steps}  tau={tau:.3f}  "
                  f"fitness range: [{worst_score:.2f}, {best_score:.2f}]")

        # --- 3. Select + reproduce (skip on last step) ---
        if step < num_steps - 1:
            population = _select_and_reproduce(population, fitness, tau, cfg)

    # --- 4. Final selection: best particle per batch element ---
    best_k = last_fitness.argmax(dim=0)  # (B,)
    batch_idx = torch.arange(B, device=device)
    best_action = population[best_k, batch_idx]  # (B, H, D)

    if action_head.verbose:
        af = best_action.float()
        print(f"[Evo] Final  shape={tuple(best_action.shape)}  "
              f"mean={af.mean():.4f}  std={af.std():.4f}  "
              f"best_indices={best_k.tolist()}")

    return best_action


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(action_head, cfg=None):
    """Monkey-patch the action head to use evolutionary population denoising.

    Replaces ``get_action_with_features()`` in-place.
    """
    if cfg is None:
        cfg = EvolutionaryConfig()

    @torch.no_grad()
    def evo_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions = denoise_evolutionary(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output, cfg=cfg,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = evo_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------


def denoise_with_lab(lab, features, *, seed=None, cfg=None):
    """Run evolutionary population denoising via DenoisingLab.

    Args:
        lab: DenoisingLab instance.
        features: BackboneFeatures from lab.encode_features().
        seed: Random seed.
        cfg: EvolutionaryConfig.

    Returns:
        torch.Tensor -- raw actions (B, action_horizon, action_dim).
        Decode with lab.decode_raw_actions(actions).
    """
    if cfg is None:
        cfg = EvolutionaryConfig()

    with torch.no_grad():
        return denoise_evolutionary(
            lab.action_head,
            features.backbone_features, features.state_features,
            features.embodiment_id, features.backbone_output,
            cfg=cfg, seed=seed,
        )
