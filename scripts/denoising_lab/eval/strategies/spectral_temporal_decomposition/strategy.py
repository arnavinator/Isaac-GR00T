"""Spectral Temporal Decomposition with Frequency-Band Velocity Scaling.

Decomposes the velocity field into temporal frequency bands via the Discrete
Cosine Transform (DCT) along the action horizon and applies per-step scaling
that amplifies each step's natural frequency role:

    Step 0 (tau=0.00): low-pass  -- amplifies gross trajectory structure
    Step 1 (tau=0.25): broad low-pass -- mid-low structure
    Step 2 (tau=0.50): broad high-pass -- mid-high detail
    Step 3 (tau=0.75): high-pass -- fine motor adjustments

An optional energy-preservation normalisation ensures the scaling only
*redistributes* energy across frequencies without changing the total magnitude.

Same 4 NFEs, <0.1 ms overhead per step (DCT on 50-element vectors).

Usage (server):
    from strategy import patch_action_head
    patch_action_head(policy.model.action_head)

Usage (notebook with DenoisingLab):
    from strategy import make_spectral_fn, denoise_with_lab
    actions = denoise_with_lab(lab, features, seed=42)
    decoded = lab.decode_raw_actions(actions)
"""

from __future__ import annotations

import math

import torch
from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# DCT utilities (matrix-multiply approach -- exact, simple, fast for N<=50)
# ---------------------------------------------------------------------------


def build_dct_matrix(N: int, device=None) -> torch.Tensor:
    """Build the N x N orthonormal DCT-II matrix.

    M is orthogonal: M @ M^T = I, so the inverse DCT is simply M^T.

    Args:
        N: Transform length (= action horizon).
        device: Torch device.

    Returns:
        ``(N, N)`` float32 tensor.
    """
    n = torch.arange(N, device=device, dtype=torch.float32)
    k = torch.arange(N, device=device, dtype=torch.float32)
    # DCT-II basis: cos(pi * k * (2n+1) / (2N))
    M = torch.cos(math.pi * k[:, None] * (2.0 * n[None, :] + 1.0) / (2.0 * N))
    # Orthonormal scaling
    M[0] *= 1.0 / math.sqrt(N)
    M[1:] *= math.sqrt(2.0 / N)
    return M


def dct_1d(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Forward DCT-II along dim=1.

    Args:
        x: ``(B, N, D)`` input in time domain.
        M: ``(N, N)`` orthonormal DCT matrix from ``build_dct_matrix``.

    Returns:
        ``(B, N, D)`` in DCT (frequency) domain.
    """
    # M @ x per batch: (N, N) @ (B, N, D) -> (B, N, D)
    return M @ x


def idct_1d(X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Inverse DCT (DCT-III) along dim=1.

    For an orthonormal DCT matrix, the inverse is its transpose.

    Args:
        X: ``(B, N, D)`` input in DCT domain.
        M: ``(N, N)`` orthonormal DCT matrix from ``build_dct_matrix``.

    Returns:
        ``(B, N, D)`` in time domain.
    """
    # M^T @ X per batch: (N, N) @ (B, N, D) -> (B, N, D)
    return M.T @ X


# ---------------------------------------------------------------------------
# Frequency weight construction
# ---------------------------------------------------------------------------


def build_frequency_weights(
    num_steps: int,
    horizon_len: int,
    gamma: float = 0.3,
    device=None,
) -> torch.Tensor:
    """Build per-step frequency scaling matrices.

    Each step gets a Gaussian affinity profile that emphasises its natural
    frequency role (early steps -> low freq, late steps -> high freq).

    Args:
        num_steps: Number of denoising steps (must be 4).
        horizon_len: Length of the horizon (DCT domain size).
        gamma: Guidance strength in ``[0, 1]``.  ``gamma=0`` recovers baseline.
        device: Torch device.

    Returns:
        ``(num_steps, horizon_len)`` tensor of per-frequency weights >= 1.
    """
    if num_steps != 4:
        raise ValueError(
            f"Spectral strategy is designed for 4 denoising steps, got {num_steps}"
        )

    k = torch.arange(horizon_len, device=device, dtype=torch.float32)
    k_max = horizon_len - 1
    weights = torch.ones(num_steps, horizon_len, device=device)

    # Step 0 (tau=0.0): low-pass, sigma=2
    weights[0] = 1.0 + gamma * torch.exp(-k ** 2 / (2.0 * 2.0 ** 2))
    # Step 1 (tau=0.25): broad low-pass, sigma=4
    weights[1] = 1.0 + gamma * torch.exp(-k ** 2 / (2.0 * 4.0 ** 2))
    # Step 2 (tau=0.50): broad high-pass, sigma=4
    weights[2] = 1.0 + gamma * torch.exp(-(k - k_max) ** 2 / (2.0 * 4.0 ** 2))
    # Step 3 (tau=0.75): high-pass, sigma=2
    weights[3] = 1.0 + gamma * torch.exp(-(k - k_max) ** 2 / (2.0 * 2.0 ** 2))

    return weights


# ---------------------------------------------------------------------------
# Spectral velocity modification
# ---------------------------------------------------------------------------


def _spectrally_modify_velocity(
    velocity: torch.Tensor,
    step_idx: int,
    dct_matrix: torch.Tensor,
    freq_weights: torch.Tensor,
    energy_preserve: bool = True,
) -> torch.Tensor:
    """Apply DCT -> frequency scaling -> IDCT to velocity.

    All computation is done in float32 for numerical stability, then cast
    back to the input dtype.

    Args:
        velocity: ``(B, H, D)`` velocity from the DiT.
        step_idx: Current denoising step index.
        dct_matrix: ``(H, H)`` orthonormal DCT matrix.
        freq_weights: ``(num_steps, H)`` frequency scaling weights.
        energy_preserve: Normalise to preserve total velocity magnitude.

    Returns:
        Spectrally-modified velocity, same shape and dtype as input.
    """
    orig_dtype = velocity.dtype
    v_f = velocity.float()

    # Forward DCT
    V = dct_1d(v_f, dct_matrix)  # (B, H, D)

    # Apply frequency-band scaling: (1, H, 1) * (B, H, D)
    W = freq_weights[step_idx]  # (H,)
    V_scaled = V * W[None, :, None]

    # Inverse DCT
    v_scaled = idct_1d(V_scaled, dct_matrix)  # (B, H, D)

    if energy_preserve:
        v_norm = v_f.norm()
        vs_norm = v_scaled.norm()
        if vs_norm > 1e-8:
            v_scaled = v_scaled * (v_norm / vs_norm)

    return v_scaled.to(orig_dtype)


# ---------------------------------------------------------------------------
# Shared helper: evaluate the DiT velocity field
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


def denoise_spectral(
    action_head, vl_embeds, state_features, embodiment_id, backbone_output,
    *,
    dct_matrix: torch.Tensor,
    freq_weights: torch.Tensor,
    energy_preserve: bool = True,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """4-step Euler with spectral frequency-band velocity scaling.

    At each step: velocity -> DCT -> frequency scale -> IDCT -> Euler update.
    Zero extra NFEs.  Same cost as baseline.

    Args:
        action_head: ``Gr00tN1d6ActionHead`` instance.
        vl_embeds: Vision-language embeddings ``(B, seq_len, 2048)``.
        state_features: Encoded state ``(B, state_horizon, hidden_dim)``.
        embodiment_id: Embodiment IDs ``(B,)``.
        backbone_output: Full backbone output.
        dct_matrix: Precomputed ``(H, H)`` orthonormal DCT matrix from
            ``build_dct_matrix``.
        freq_weights: Precomputed ``(num_steps, H)`` frequency scaling weights
            from ``build_frequency_weights``.
        energy_preserve: Normalise velocity magnitude after spectral scaling.
        initial_noise: Optional starting noise ``(B, action_horizon, action_dim)``.

    Returns:
        Denoised actions ``(B, action_horizon, action_dim)``.
    """
    num_steps = 4
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    num_buckets = action_head.num_timestep_buckets
    dt = 1.0 / num_steps
    H = action_head.action_horizon

    if initial_noise is not None:
        actions = initial_noise.to(device=device, dtype=dtype)
    else:
        actions = torch.randn(
            batch_size, H, action_head.action_dim,
            dtype=dtype, device=device,
        )

    for step in range(num_steps):
        tau = step / float(num_steps)
        t_bucket = int(tau * num_buckets)

        velocity = _evaluate_velocity(
            action_head, actions, t_bucket,
            vl_embeds, state_features, embodiment_id, backbone_output,
        )

        # Spectrally modify the velocity
        v_modified = _spectrally_modify_velocity(
            velocity, step, dct_matrix, freq_weights, energy_preserve,
        )

        actions = actions + dt * v_modified

        if action_head.verbose:
            print(
                f"[Spectral] Step {step}/{num_steps}  "
                f"tau={tau:.3f}  bucket={t_bucket}  "
                f"v_orig_norm={velocity.float().norm():.4f}  "
                f"v_mod_norm={v_modified.float().norm():.4f}  "
                f"a_norm={actions.float().norm():.4f}"
            )

    if action_head.verbose:
        af = actions.float()
        print(
            f"[Spectral] Final  shape={tuple(actions.shape)}  "
            f"mean={af.mean():.4f}  std={af.std():.4f}"
        )

    return actions


# ---------------------------------------------------------------------------
# DenoisingLab guided_fn interface
# ---------------------------------------------------------------------------


def make_spectral_fn(
    gamma: float = 0.3,
    energy_preserve: bool = True,
):
    """Factory for spectral velocity modifier.

    Returns a function compatible with ``DenoisingLab.denoise(guided_fn=...)``.
    DCT matrix and frequency weights are lazily built on first call.
    """
    cache: dict = {}

    def guided_fn(
        actions_before: torch.Tensor, step_idx: int, velocity: torch.Tensor,
    ) -> torch.Tensor:
        device = velocity.device
        H = velocity.shape[1]
        key = (str(device), H)

        if key not in cache:
            cache[key] = {
                "M": build_dct_matrix(H, device=device),
                "W": build_frequency_weights(4, H, gamma=gamma, device=device),
            }

        return _spectrally_modify_velocity(
            velocity, step_idx,
            cache[key]["M"], cache[key]["W"],
            energy_preserve=energy_preserve,
        )

    return guided_fn


# ---------------------------------------------------------------------------
# Server patch
# ---------------------------------------------------------------------------


def patch_action_head(
    action_head,
    gamma: float = 0.3,
    energy_preserve: bool = True,
):
    """Monkey-patch the action head to use spectral denoising.

    Replaces ``get_action_with_features()`` in-place.  DCT matrix and frequency
    weights are built once here and reused for every subsequent inference call.
    """
    H = action_head.action_horizon
    dct_matrix = build_dct_matrix(H, device=action_head.device)
    freq_weights = build_frequency_weights(4, H, gamma=gamma, device=action_head.device)

    @torch.no_grad()
    def spectral_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output,
    ):
        actions = denoise_spectral(
            action_head, backbone_features, state_features,
            embodiment_id, backbone_output,
            dct_matrix=dct_matrix, freq_weights=freq_weights,
            energy_preserve=energy_preserve,
        )
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        })

    action_head.get_action_with_features = spectral_get_action_with_features


# ---------------------------------------------------------------------------
# Notebook convenience
# ---------------------------------------------------------------------------


def denoise_with_lab(
    lab, features, *, seed=None,
    gamma: float = 0.3, energy_preserve: bool = True,
):
    """Run spectral denoising via DenoisingLab.

    Uses the ``guided_fn`` interface so all intermediates are recorded.

    Returns:
        ``torch.Tensor`` -- raw actions ``(B, action_horizon, action_dim)``.
        Decode with ``lab.decode_raw_actions(actions)``.
    """
    guided_fn = make_spectral_fn(gamma=gamma, energy_preserve=energy_preserve)
    result = lab.denoise(
        features, num_steps=4, guided_fn=guided_fn, seed=seed,
    )
    return result.action_pred
