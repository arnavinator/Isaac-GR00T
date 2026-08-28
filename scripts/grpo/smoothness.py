"""Endpoint-roughness constraint for Jitter-GRPO (the "jerk constraint").

Self-contained primitives for the temporal-smoothness term described in
``jerk-constraint.md``. Kept out of ``fm_log_prob.py`` so every piece is
testable without a DiT: this module never touches the model.

The quantity being constrained is the roughness of the flow model's IMPLIED
ENDPOINT along the action-chunk horizon axis ``h``:

    a_hat(tau) = x_tau + (1 - tau) * v_theta(x_tau, tau)   ==   a + (1 - tau) * r

where ``r = v_theta - (a - eps)`` is the velocity residual. Two facts from the
measurements motivate targeting the endpoint rather than the residual:

  * ``HF(a_hat)`` separates the pretrained field from the GRPO-finetuned one by
    100-200x, against the residual's 1.5-2.1x.
  * The two checkpoints measured rank OPPOSITELY on residual roughness versus
    chunk roughness, so residual roughness does not control what you see.

Roughness is decomposed into two independent coordinates, ``R = 6 * M * HF``:

    D2 u[h] = u[h+2] - 2 u[h+1] + u[h]      curvature along the horizon
    M(u)    = mean(u^2)                     energy   -- how big the values are
    R(u)    = mean((D2 u)^2)                roughness -- how much they zig-zag
    HF(u)   = R(u) / (6 * M(u))             roughness per unit energy

The ``6`` is the squared norm of the ``(1, -2, 1)`` stencil, which pins
``HF = 1`` for a column that is white along ``h``. ``HF`` is scale-free, so it
reports the SHAPE of the sequence and not its size; ``M`` carries the size.
Verified calibration: constant 0.00, smooth half-sine 0.01, white 1.00,
alternating +-c 8/3.

See ``jerk-constraint.md`` sections 1, 6 and 7 for the derivations.
"""

from __future__ import annotations

import torch


# Floor on the HF denominator. Only bites for a degenerate all-zero slice, where
# the numerator is zero too, so HF evaluates to 0 rather than NaN. Without it a
# zero-energy row produces inf -> the trainer's non-finite guard silently drops
# the whole minibatch, which looks like a training stall rather than a bug.
SMOOTH_M_EPS = 1e-8

# Action keys whose values are discrete 0/1, thresholded at 0.5 by robocasa's
# key converters (`PandaOmronKeyConverter.unmap_action`). A grasp and a base-mode
# switch ARE step functions, so penalising their second difference would suppress
# grasping. Never included in the constrained dim set.
DEFAULT_DISCRETE_ACTION_KEYS: tuple[str, ...] = ("gripper_close", "control_mode")

# `base_motion` is gated by `control_mode` -- arm and base are mutually exclusive
# under robosuite's HybridMobileBase -- so in arm mode it is commanded but inert.
# Constraining an inert channel spends the adapter's limited capacity on dims
# that do not move the robot, so it is excluded unless explicitly requested.
DEFAULT_GATED_ACTION_KEYS: tuple[str, ...] = ("base_motion",)


def second_difference(u: torch.Tensor) -> torch.Tensor:
    """Second difference along the horizon axis (``dim=-2``).

    Args:
        u: ``[..., H, D]``. H must be >= 3 for a non-empty result.

    Returns:
        ``[..., H-2, D]``. Zero for a straight ramp, small for a gentle curve,
        large for a zig-zag.
    """
    if u.shape[-2] < 3:
        raise ValueError(
            f"second_difference needs a horizon of at least 3, got {u.shape[-2]}. "
            f"The (1,-2,1) stencil spans 3 rows."
        )
    return u[..., 2:, :] - 2.0 * u[..., 1:-1, :] + u[..., :-2, :]


def roughness_moments(u: torch.Tensor) -> torch.Tensor:
    """Per-row ``(R, M)`` as a ``[B, 2]`` tensor.

    Returning the raw moments rather than their ratio is deliberate. ``HF`` is a
    per-row ratio whose denominator is that row's own energy, so a near-idle chunk
    (arm holding still, ``M(a) -> 0``) reports ``HF(a_hat) -> HF(r)``, hundreds of
    times the value a moving chunk reports. An unweighted mean of such ratios is
    dominated by the idle rows. Keeping ``R`` and ``M`` separate lets the caller
    POOL them -- ``sum(R) / (6 * sum(M))`` -- which is energy-weighted by
    construction, so idle rows contribute in proportion to their (tiny) energy
    instead of dominating.

    Args:
        u: ``[B, H, D]``, already sliced to the valid horizon and dim set.

    Returns:
        ``[B, 2]`` with ``[:, 0] = R(u)`` and ``[:, 1] = M(u)``.
    """
    if u.dim() != 3:
        raise ValueError(f"roughness_moments expects [B, H, D], got {tuple(u.shape)}")
    d2 = second_difference(u)
    r_term = d2.pow(2).flatten(start_dim=1).mean(dim=1)
    m_term = u.pow(2).flatten(start_dim=1).mean(dim=1)
    return torch.stack((r_term, m_term), dim=1)


def pooled_hf(
    moments: torch.Tensor,
    *,
    detach_denominator: bool = True,
    eps: float = SMOOTH_M_EPS,
) -> torch.Tensor:
    """Energy-weighted ``HF`` over a batch: ``sum(R) / (6 * sum(M))``.

    A 0-dim tensor. Robust to rows of wildly differing energy in a way the mean of
    per-row ratios is not (see ``roughness_moments``), and identical in meaning at
    any batch size, so a threshold calibrated on one batch size transfers.

    ``detach_denominator`` carries the same requirement as in ``roughness_hf``:
    ``d(HF)/d(M) < 0``, so a live denominator would let the model satisfy the
    constraint by adding DC (constant-along-``h``) energy, which ``D2``
    annihilates -- ``R`` unchanged, ``M`` up, ``HF`` down, spectrum untouched.
    """
    if moments.dim() != 2 or moments.shape[1] != 2:
        raise ValueError(f"pooled_hf expects [B, 2], got {tuple(moments.shape)}")
    r_sum = moments[:, 0].sum()
    m_sum = moments[:, 1].sum()
    if detach_denominator:
        m_sum = m_sum.detach()
    return r_sum / (6.0 * m_sum + eps)


def roughness_hf(
    u: torch.Tensor,
    *,
    detach_denominator: bool = True,
    eps: float = SMOOTH_M_EPS,
) -> torch.Tensor:
    """Per-row ``HF(u) = R(u) / (6 * M(u))``.

    Args:
        u: ``[B, H, D]``, ALREADY sliced to the valid horizon and the
            constrained dim set. Slicing must happen before differencing --
            a ``D2`` that straddles the padded region of the model's
            ``(50, 128)`` output is meaningless.
        detach_denominator: Detach ``M``. Required in the loss:
            ``d(HF)/d(M) < 0``, so a live denominator lets the model satisfy the
            term by INFLATING the residual's energy instead of smoothing its
            spectrum. Detached, the term means exactly "reduce high-frequency
            energy at fixed scale" -- the reward terms set the scale, this term
            sets only the spectrum. Pass False for pure measurement.
        eps: Floor on the denominator (see ``SMOOTH_M_EPS``).

    Returns:
        ``[B]``: 0 = constant along h, 1 = white along h, 8/3 for a strictly
        alternating sequence. The attainable maximum is ``lambda_max(D2^T D2)``
        scaled, = 2.9839 at H=16; 8/3 is the alternating value, not the bound.
    """
    if u.dim() != 3:
        raise ValueError(f"roughness_hf expects [B, H, D], got {tuple(u.shape)}")
    d2 = second_difference(u)
    r_term = d2.pow(2).flatten(start_dim=1).mean(dim=1)
    m_term = u.pow(2).flatten(start_dim=1).mean(dim=1)
    if detach_denominator:
        m_term = m_term.detach()
    return r_term / (6.0 * m_term + eps)


def implied_endpoint(
    noisy_trajectory: torch.Tensor,
    pred_velocity: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """``a_hat(tau) = x_tau + (1 - tau) * v_theta``. MEASUREMENT ONLY.

    .. warning::
        **Do not use this to build the constraint.** It is not called by the
        production path and must not be. It equals ``a + (1 - tau) * r`` only when
        the ``eps`` inside ``x_tau`` is the same ``eps`` the velocity target uses.
        Under Jitter-GRPO it is not: ``noisy_trajectory`` is built from
        ``eps' = sqrt(1-lam^2) eps + lam xi``, so feeding it a K-loop velocity gives

            a_hat = a + (1-tau) r + (1-tau)^2 J (eps' - eps)

        whose last term is white, theta-independent, and at the production
        ``lam = 0.25`` large enough to invert the base-vs-finetuned discrimination
        the constraint depends on -- a silent no-op. The production path instead
        takes a dedicated clean forward at ``tau = 0`` and forms ``eps + v0``
        directly (``fm_log_prob.compute_fm_log_prob``).

        Kept because the identity ``a_hat = a + (1-tau) r`` is part of the theory
        (``jerk-constraint.md`` section 6) and is worth asserting in tests.

    Args:
        noisy_trajectory: ``[B, H, D]`` -- ``(1-t) * eps_input + t * actions``.
        pred_velocity: ``[B, H, D]`` -- the DiT's velocity prediction.
        t: ``[B]`` the timesteps at which ``pred_velocity`` was evaluated.
    """
    return noisy_trajectory + (1.0 - t)[:, None, None] * pred_velocity


def build_continuous_action_dims(
    modality_keys: list[str],
    key_dims: dict[str, int],
    *,
    discrete_keys: tuple[str, ...] | list[str] = DEFAULT_DISCRETE_ACTION_KEYS,
    gated_keys: tuple[str, ...] | list[str] = DEFAULT_GATED_ACTION_KEYS,
    include_gated: bool = False,
) -> tuple[list[int], list[str], int]:
    """Column indices of the continuous action dims, from the CHECKPOINT layout.

    Mirrors ``processing_gr00t_n1d6.decode_action``: walk the embodiment's action
    ``modality_keys`` IN ORDER, take each key's normalized ``dim``, and
    accumulate contiguous slices. The ordering lives in the checkpoint, not in
    the repo, so nothing here is hardcoded.

    Note this is the ACTION layout, which differs from the STATE layout in a way
    that is easy to trip over: ``end_effector_rotation`` is 3-dim axis-angle,
    while the state's ``end_effector_rotation_relative`` is a 4-dim quaternion.

    Args:
        modality_keys: Ordered action keys for this embodiment. A ``"prefix."``
            is tolerated and stripped for matching.
        key_dims: Per-key normalized dim count.
        discrete_keys: Always excluded (0/1, thresholded at 0.5).
        gated_keys: Excluded unless ``include_gated``.
        include_gated: Admit ``gated_keys`` into the constrained set.

    Returns:
        ``(dims, kept_key_names, total_valid_dims)``.
    """
    dims: list[int] = []
    kept: list[str] = []
    start = 0
    disc = {k.split(".")[-1] for k in discrete_keys}
    gated = {k.split(".")[-1] for k in gated_keys}
    for key in modality_keys:
        if key not in key_dims:
            raise KeyError(
                f"action key {key!r} is in modality_keys but has no entry in "
                f"key_dims (got {sorted(key_dims)}); the two must come from the "
                f"same embodiment's config"
            )
        width = int(key_dims[key])
        if width <= 0:
            raise ValueError(f"action key {key!r} has non-positive dim {width}")
        short = key.split(".")[-1]
        if short in disc:
            pass
        elif short in gated and not include_gated:
            pass
        else:
            dims.extend(range(start, start + width))
            kept.append(short)
        start += width
    if not dims:
        raise ValueError(
            f"no continuous action dims survived filtering. modality_keys="
            f"{modality_keys}, discrete={sorted(disc)}, gated={sorted(gated)}, "
            f"include_gated={include_gated}. The smoothness term would have "
            f"nothing to constrain."
        )
    return dims, kept, start


def describe_dim_selection(
    modality_keys: list[str],
    key_dims: dict[str, int],
    dims: list[int],
    *,
    horizon: int,
) -> str:
    """Human-readable layout table for the startup banner (self-verifying)."""
    lines = [
        f"    valid rectangle [:{horizon}, :{sum(int(key_dims[k]) for k in modality_keys)}]"
    ]
    chosen = set(dims)
    start = 0
    for key in modality_keys:
        width = int(key_dims[key])
        span = set(range(start, start + width))
        mark = "in C" if span <= chosen else ("excluded" if not (span & chosen) else "PARTIAL")
        lines.append(
            f"    [{start:2d}:{start + width:2d}] {key.split('.')[-1]:28s} "
            f"dim={width}  {mark}"
        )
        start += width
    return "\n".join(lines)
