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

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from smoothness import SMOOTH_INSTRUMENTS, roughness_moments

# Std of the Gaussian jitter applied to each tau center. Named rather than left as
# a bare default so the roughness constraint can record it in its calibration
# guard key. Note hf_ref does NOT depend on it -- the smooth pass runs on clean
# forwards from the original eps -- but this width changes what every other term
# in the loss evaluates, so a silent mismatch across a resume is worth refusing
# rather than inferring later.
TAU_JITTER_STD = 0.02

# `SMOOTH_INSTRUMENTS` is defined in `smoothness` rather than here so
# `grpo_config` can validate against it without importing this model-facing
# module; it is imported above so callers of `compute_fm_log_prob` need not
# learn a second module name.


def inference_schedule(action_head: nn.Module) -> tuple[list[float], float]:
    """The PRODUCTION denoising schedule, read off the model config.

    Mirrors ``Gr00tN1d6ActionHead.get_action_with_features``
    (``gr00t/model/gr00t_n1d6/gr00t_n1d6.py:317-321``) exactly:

        dt     = 1.0 / num_inference_timesteps
        t_cont = t / float(num_inference_timesteps)   for t in range(N)

    Nothing here is hardcoded. At the shipped ``num_inference_timesteps = 4``
    this returns ``([0.0, 0.25, 0.5, 0.75], 0.25)``, but a checkpoint that
    overrides the config gets its own schedule -- which matters because a
    schedule mismatch would silently constrain a trajectory the robot never
    executes, i.e. exactly the failure this instrument exists to fix.

    UNITS. The returned values are CONTINUOUS t in [0, 1), not bucket indices.
    ``_dit_velocity`` does the bucketization itself (``(t * num_timestep_buckets)
    .long()``), matching what ``get_action_with_features`` does inline
    (``int(t_cont * num_timestep_buckets)``), and matching the existing smooth
    pass, which builds ``t0 = torch.zeros(B)`` -- a continuous 0.0, not bucket 0.
    Note ``noise_s`` (0.999) does NOT appear: it scales the Beta-sampled TRAINING
    timesteps (``gr00t_n1d6.py:140``), while inference walks the raw ``t/N`` grid.

    PRECISION. These are Python floats (float64), and a caller building a tensor
    from them must NOT narrow to bf16. Note the reason is NOT that bf16 cannot
    represent the timesteps: 0.75 is exactly representable in bf16 (it is
    2^-1 + 2^-2). What loses precision is the PRODUCT: bucketizing computes
    ``t * num_timestep_buckets``, and with 8 mantissa bits the nearest bf16 to
    750.0 is 752.0, so ``t=0.75`` bucketizes to 752 where production's
    Python-float ``int(t_cont * num_timestep_buckets)`` gives 750 -- a different
    AdaLayerNorm conditioning from the one the sampler actually uses. See the
    dtype comment in ``_smooth_chunk_rollout``.

    Returns:
        ``(t_values, dt)`` -- the per-step continuous timesteps and the Euler
        step size.
    """
    n_steps = int(action_head.num_inference_timesteps)
    if n_steps < 1:
        raise ValueError(
            f"num_inference_timesteps must be >= 1, got {n_steps}. The chunk "
            f"instrument rolls the production sampler out, so there is no "
            f"trajectory to difference below one step."
        )
    dt = 1.0 / n_steps
    return [i / float(n_steps) for i in range(n_steps)], dt


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
    noise_for_input: torch.Tensor | None = None,
    return_per_tau: bool = False,
    smooth_dims: torch.Tensor | None = None,
    smooth_horizon: int | None = None,
    smooth_no_grad: bool = False,
    smooth_instrument: str = "chunk",
) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
            If None, samples one fresh noise tensor. ALSO defines the velocity_target
            via velocity_target = actions - noise (this is the ORIGINAL ε in the
            Jitter-GRPO formulation).
        n_samples: Number of timestep samples for variance reduction (K).
        noise_beta_alpha: Alpha param for Beta distribution (default 1.5).
        noise_beta_beta: Beta param for Beta distribution (default 1.0).
        noise_s: Scaling factor for timestep (default from model config: 0.999).
        noise_for_input: Optional [K, B, action_horizon, action_dim] tensor of
            per-K DiT INPUT noise (Jitter-GRPO ε'_k). When provided, the
            noisy_trajectory at timestep k is built from noise_for_input[k]
            instead of `noise`, while velocity_target stays at (actions - noise).
            None = use `noise` for both target and input (current behavior,
            bit-identical when jitter is disabled). The 4-D shape with leading
            K is required so each τ can use an independent ξ-jitter.
        return_per_tau: When True, ALSO return the un-averaged [K, B] per-τ
            log-probs alongside the [B] mean. Intended for the once-per-iteration
            jitter diagnostic in train_grpo._jitter_gap_diagnostics, which needs
            the per-τ breakdown to fit
                gap(τ) ≈ (1-τ)² · λ² · ‖∇_x v_θ‖²_F
            i.e. to see WHERE along the denoising path the velocity field is
            noise-sensitive — a single K-averaged number cannot show that.
            Default False keeps the return type and the arithmetic
            bit-identical for every existing caller: the flag adds one Python
            `None` assignment on that path and nothing else. When True it does
            cost one extra elementwise negation per tau (`-per_sample_mse` is
            evaluated a second time rather than reusing the in-place `+=`
            operand) — negligible, and the only callsite is a no_grad
            diagnostic.
        smooth_dims: Optional 1-D LongTensor of action-dim column indices. When
            given (with `smooth_horizon`), the smoothness pass runs and reduces a
            trajectory to per-row roughness moments over
            `[:, :smooth_horizon, smooth_dims]`. See `smoothness.py` and
            `jerk-constraint.md` sections 6-7. WHICH trajectory is decided by
            `smooth_instrument`.

            Deliberately NOT taken from the K-loop: under Jitter-GRPO the K-loop's
            input is x'_tau, so its velocity carries the model's response to the
            eps-jitter, landing in a_hat as (1-tau)^2 J (eps'-eps) -- white,
            theta-independent, and at the production lambda=0.25 large enough to
            collapse the base-vs-finetuned discrimination the constraint needs.
            Every forward here uses the ORIGINAL, un-jittered eps.

            Pass `smooth_no_grad=True` when the caller only reads the moments
            (e.g. hf_ref calibration) -- no graph is then retained anywhere.
        smooth_horizon: Valid action horizon (e.g. 16 for PandaOmron). Slicing
            happens BEFORE differencing -- a second difference straddling the pad
            boundary of the (50, 128) output is meaningless.
        smooth_no_grad: Build the smooth forward(s) under `torch.no_grad()`.
            Correct and cheaper in activations whenever the caller only reads the
            moments, e.g. during hf_ref calibration.
        smooth_instrument: WHICH trajectory the roughness operator is applied to.

            "chunk" (default) -- the 4-step generated chunk, i.e. the thing the
                robot executes. A full production-schedule Euler rollout from the
                collected eps, with only the LAST velocity evaluation carrying a
                graph (see `_smooth_chunk_rollout`). Costs `num_inference_timesteps`
                DiT forwards but only ONE forward's activations.
            "endpoint" -- the historical instrument: the 1-step implied endpoint
                `a_hat(0) = eps + v_theta(eps, 0)`, from a single differentiable
                forward at tau=0. Kept bit-for-bit so old runs stay reproducible.

            Why the default moved. An empirical sweep over 16 checkpoints of a
            real run found the endpoint does NOT control physical trajectory
            jerk: over iterations 10-16 the endpoint HF FELL 9% while EEF path
            jerk ROSE 11% (Spearman rho over that window: +0.00), and a run
            constrained at coef 0.15 pinned endpoint HF at 3-6x base for six
            iterations while its executed chunks still degraded 2.2x -> 8.6x
            base. The 4-step chunk's HF correlates with path jerk at rho = +0.98
            overall and +0.96 over the late iterations. Same (1,-2,1) operator;
            different trajectory.

    Returns:
        log_probs: [B] tensor of FM log-probability surrogates (negative MSE).
        With `return_per_tau=True`, additionally the un-averaged [K, B] per-tau
        log-probs. With `smooth_dims`/`smooth_horizon`, additionally a 2-tuple
        `(moments, endpoint_moments)` of [B, 2] per-row roughness moments
        `(R, M)` -- raw moments, not their ratio, so the caller can POOL them
        (see `smoothness.roughness_moments`).

        `moments` is the CONSTRAINED instrument (whichever `smooth_instrument`
        selected); `endpoint_moments` is always the tau=0 endpoint, monitoring
        only. Under `smooth_instrument="endpoint"` the two are the same tensor
        object, so the caller must not assume they are independent. Under
        "chunk" the endpoint pair is a FREE byproduct: the rollout's first step
        is `v_theta(eps, t=0)`, which is exactly the endpoint's velocity, and it
        runs under `no_grad` so it carries no gradient.

        Extras are appended in a fixed order:
            neither            -> log_probs
            per_tau only       -> (log_probs, per_tau)            [unchanged]
            smooth only        -> (log_probs, (mom, endpoint_mom))
            both               -> (log_probs, per_tau, (mom, endpoint_mom))
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
    # (action, noise) pair, not the interpolation point). In Jitter-GRPO this
    # stays at the ORIGINAL eps even when the DiT input is the jittered
    # eps' = sqrt(1-λ²)·eps + λ·ξ — that asymmetry is what makes the loss in
    # expectation an FM-loss + Jacobian-norm regularizer.
    velocity_target = actions - eps

    # Resolve the per-K input-noise tensor. Only the [K, B, H, D] shape is
    # supported because the only call path that exercises this argument
    # (Jitter-GRPO in _grpo_update_inner) constructs it as such; carrying
    # broadcast logic for shapes nothing exercises is dead surface area.
    if noise_for_input is None:
        eps_input_all = None
    else:
        if not (
            noise_for_input.dim() == 4
            and noise_for_input.shape[0] == n_samples
            and noise_for_input.shape[1:] == eps.shape
        ):
            raise ValueError(
                f"noise_for_input must be None or shape "
                f"[{n_samples}, {eps.shape[0]}, {eps.shape[1]}, {eps.shape[2]}]; "
                f"got {tuple(noise_for_input.shape)}"
            )
        eps_input_all = noise_for_input

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
    # Per-τ terms, kept only when the caller asks.
    per_tau_log_probs: list[torch.Tensor] | None = [] if return_per_tau else None

    # --- Roughness (jerk constraint) bookkeeping ---
    # Argument validation, done once here rather than inside the smooth block
    # below. Every smooth forward uses the ORIGINAL eps and the production
    # timestep schedule, so it needs nothing from `timesteps` and is independent
    # of both the tau-jitter and the eps-jitter -- see the block after the K-loop.
    want_smooth = smooth_dims is not None or smooth_horizon is not None
    smooth_moments: torch.Tensor | None = None
    endpoint_moments: torch.Tensor | None = None
    if want_smooth:
        if smooth_dims is None or smooth_horizon is None:
            raise ValueError(
                "smooth_dims and smooth_horizon must be given together "
                f"(got smooth_dims={'set' if smooth_dims is not None else None}, "
                f"smooth_horizon={smooth_horizon})"
            )
        if smooth_instrument not in SMOOTH_INSTRUMENTS:
            raise ValueError(
                f"smooth_instrument must be one of {sorted(SMOOTH_INSTRUMENTS)}, "
                f"got {smooth_instrument!r}"
            )
        if not (3 <= smooth_horizon <= actions.shape[1]):
            raise ValueError(
                f"smooth_horizon={smooth_horizon} must lie in "
                f"[3, {actions.shape[1]}] (the padded action horizon). The lower "
                f"bound is the (1,-2,1) stencil's span: at 1 or 2 rows "
                f"`second_difference` raises from inside the loss path, which "
                f"would surface as a mid-training crash rather than a config "
                f"error. Rejected HERE, once, before the first forward -- the "
                f"metrics path guards the same condition by degrading to 'no "
                f"reading', but the loss cannot degrade, so it must not start."
            )
        if int(smooth_dims.max()) >= actions.shape[2]:
            raise ValueError(
                f"smooth_dims max index {int(smooth_dims.max())} exceeds the "
                f"action dim {actions.shape[2]}"
            )
        smooth_dims = smooth_dims.to(device=device)

    def _dit_velocity(noisy_trajectory, t):
        """One DiT forward -> pred_velocity.

        Factored so the roughness pass can reuse it verbatim rather than
        duplicating the call signature -- BOTH instruments go through it: the
        endpoint calls it once at tau=0, the chunk calls it once per Euler step
        via `_smooth_chunk_rollout`.
        """
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
            # cross-attention masks are built from
            # `image_mask & backbone_attention_mask` internally (dit.py:322-323).
            # We pass it anyway for parity with the pretraining forward.
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
        pred = action_head.action_decoder(model_output, embodiment_id)
        return pred[:, -actions.shape[1]:]

    for k in range(n_samples):
        # --- Sample or use pre-specified timestep ---
        if timesteps is not None:
            t = timesteps[k]  # [B]
        else:
            beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
            t = beta_dist.sample([B]).to(device=device, dtype=dtype)
            t = (1 - t) * noise_s

        # --- Construct noisy trajectory at this timestep ---
        # x_t = (1 - t) * epsilon + t * action.
        # In Jitter-GRPO (eps_input_all is not None), the DiT INPUT noise is
        # the jittered eps'_k = sqrt(1-λ²)·eps + λ·ξ_k (constructed by the
        # caller), but velocity_target above stays at (action - original_eps).
        # That asymmetry is what gives the loss its Jacobian-regularizer
        # interpretation in expectation.
        eps_input = eps if eps_input_all is None else eps_input_all[k]
        t_expanded = t[:, None, None]  # [B, 1, 1]
        noisy_trajectory = (1 - t_expanded) * eps_input + t_expanded * actions

        pred_velocity = _dit_velocity(noisy_trajectory, t)

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
        if per_tau_log_probs is not None:
            per_tau_log_probs.append(-per_sample_mse)

    # ── Roughness instrument: dedicated CLEAN forward(s) with the ORIGINAL eps ──
    # Deliberately NOT taken from the K-loop. Under Jitter-GRPO the K-loop's DiT
    # input is x'_tau (built from eps' = sqrt(1-lam^2)eps + lam*xi), so its
    # velocity carries the model's RESPONSE to that perturbation:
    #     a_hat = a + (1-tau)r + (1-tau)^2 * J * (eps' - eps)
    # The last term is white, theta-independent, and at the production lam=0.25
    # with the measured jacobian_fro_sq ~2.4 it DOMINATES: HF goes 0.000347 ->
    # 0.790 at tau=0 on a base-like chunk (2275x). Because the calibration would
    # be contaminated identically, hf_ref freezes above HF's theoretical maximum
    # for H=16 (2.984) and the hinge can never fire -- a silent no-op. Both
    # instruments below start from the clean, collected eps, so both are immune.
    #
    # WHICH trajectory gets differenced is `smooth_instrument`:
    #
    #   "endpoint" -- the historical one-step implied endpoint at tau = 0. There
    #       x_0 == eps exactly, so a_hat(0) = eps + v_theta(eps, 0) is literally
    #       the 1-step Euler endpoint, and the (1-tau)^2 leverage weight is 1.
    #       One differentiable DiT forward. Retained bit-for-bit so the runs
    #       calibrated against it stay reproducible.
    #
    #   "chunk" -- the 4-step GENERATED chunk, i.e. what the robot executes.
    #       Measured over 16 checkpoints of a real run: the endpoint's HF has
    #       Spearman rho = +0.00 with EEF path jerk over the late iterations
    #       (endpoint HF fell 9% while path jerk rose 11%), while the chunk's HF
    #       correlates at +0.98 overall / +0.96 late. Constraining the endpoint
    #       therefore does not control the deliverable; constraining the chunk
    #       does. Same operator, different trajectory.
    #
    # Cost: the endpoint pass is exactly one extra DiT forward (~1/K of the
    # K-loop, ~17% at the default K=6). The chunk pass is `num_inference_timesteps`
    # forwards but only ONE forward's ACTIVATIONS (see _smooth_chunk_rollout),
    # which is what keeps mini_batch_size=8 inside the 25.3 GB budget -- a fully
    # differentiated 4-step rollout was estimated at 29.1 GB and OOMs.
    if smooth_dims is not None:
        sl = (slice(None), slice(0, smooth_horizon))

        def _slice_f32(u):
            """Slice to the constrained rectangle and upcast, BEFORE differencing.

            A D2 straddling the pad boundary at h = smooth_horizon of the
            (50, 128) output would difference real actions against padding.

            The fp32 cast is applied to each operand as it enters, which matters
            for the ENDPOINT path: `eps + v0` is summed after both are upcast, so
            the squared quantity R does not lose mantissa to a bf16 addition.
            It does NOT apply to the chunk path, and must not: the Euler sum
            `x + dt*v` there is accumulated in bf16 on purpose, because that is
            precisely what production does (`gr00t_n1d6.py:328`). Upcasting the
            rollout would make the measured chunk differ from the executed one,
            which is the whole property `_smooth_chunk_rollout` exists to keep.
            """
            return u[sl].index_select(2, smooth_dims).float()

        with torch.no_grad() if smooth_no_grad else contextlib.nullcontext():
            if smooth_instrument == "endpoint":
                # x_0 == eps exactly, so a_hat(0) = eps + v_theta(eps, 0).
                # == a + r, since velocity_target = a - eps.
                t0 = torch.zeros(B, device=device, dtype=dtype)
                v0 = _dit_velocity(eps, t0)
                smooth_moments = roughness_moments(_slice_f32(eps) + _slice_f32(v0))
                # The constrained instrument IS the endpoint here, so the
                # monitoring pair is the same tensor -- no second forward, and
                # smooth/hf_mean == smooth/endpoint_hf_mean by construction.
                endpoint_moments = smooth_moments
            else:
                chunk, v0 = _smooth_chunk_rollout(
                    _dit_velocity, eps, action_head
                )
                smooth_moments = roughness_moments(_slice_f32(chunk))
                # FREE byproduct: the rollout's first step evaluates
                # v_theta(eps, t=0), which is exactly the endpoint's velocity, so
                # a_hat(0) = eps + v0 costs no forward. It comes from the
                # no_grad leg of the rollout, hence carries no gradient -- which
                # is all a monitoring metric needs.
                with torch.no_grad():
                    endpoint_moments = roughness_moments(
                        _slice_f32(eps) + _slice_f32(v0)
                    )

    # Average across K timestep samples
    log_probs = log_probs_accumulated / n_samples

    # Extras are appended in a fixed order so every existing caller's unpacking
    # keeps working: per_tau first (pre-existing), then the smooth pair.
    extras: list = []
    if per_tau_log_probs is not None:
        extras.append(torch.stack(per_tau_log_probs, dim=0))  # [K, B]
    if smooth_moments is not None:
        # A TUPLE, not two positional extras: the constrained instrument and the
        # monitoring endpoint travel together, so a caller cannot unpack one and
        # silently drop the other, and adding the endpoint did not renumber the
        # per_tau slot.
        extras.append((smooth_moments, endpoint_moments))     # each [B, 2]=(R,M)
    if extras:
        return (log_probs, *extras)

    return log_probs


def _smooth_chunk_rollout(
    dit_velocity,
    eps: torch.Tensor,
    action_head: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Production Euler rollout with only the LAST step differentiable.

    Runs the full ``inference_schedule(action_head)`` -- same step count, same
    ``dt``, same continuous ``t`` values as ``get_action_with_features`` -- from
    the collected ``eps``. The first N-1 velocity evaluations run under
    ``torch.no_grad()`` and their outputs are explicitly ``detach()``-ed, so the
    graph contains exactly one DiT forward.

    Two consequences, both deliberate:

    * **The forward VALUE is exact.** Nothing is approximated in the number the
      hinge compares against ``hf_ref``; it is the true 4-step chunk the sampler
      would produce from this ``eps``. Calibration and measurement therefore stay
      the same functional, which is what makes a frozen threshold meaningful.

    * **The gradient is BIASED.** It misses how theta shapes the first N-1 steps
      and hence the sampler path that step N is evaluated on -- roughly a quarter
      of the true gradient's magnitude at N=4. Accepted: differentiating all four
      steps costs 4 graph-forwards, estimated 29.1 GB against 25.3 GB available at
      ``mini_batch_size=8``, i.e. an OOM or a halved batch. The retained term
      still points downhill on the roughness of the executed chunk, which is what
      the constraint needs; ``smooth_coef`` may need raising to compensate for the
      smaller magnitude (suggested 0.15-0.5, see ``grpo_config``).

    Args:
        dit_velocity: ``(x, t) -> v``, the closure over the batch's conditioning.
        eps: ``[B, H_pad, D_pad]`` the CLEAN collected noise (never the jittered
            ``eps'``) -- the sampler's own starting point.
        action_head: read for ``num_inference_timesteps``.

    Returns:
        ``(chunk, v_first)`` -- the rolled-out chunk (graph on the last step
        only) and the FIRST step's velocity ``v_theta(eps, t=0)``, which is the
        endpoint instrument's velocity and comes free of charge. ``v_first`` is
        detached whenever N > 1.
    """
    t_values, dt = inference_schedule(action_head)
    last = len(t_values) - 1
    B = eps.shape[0]

    x = eps
    v_first: torch.Tensor | None = None
    for i, t_val in enumerate(t_values):
        # FLOAT64, deliberately, and NOT the batch dtype. `_dit_velocity`'s only
        # use of `t` is `(t * num_timestep_buckets).long()`, and it is that
        # PRODUCT, not the timestep itself, that bf16 cannot hold: 0.75 IS exact
        # in bf16 (2^-1 + 2^-2), but with 8 mantissa bits the nearest bf16 to
        # 750.0 is 752.0, so a bf16 `t` bucketizes to 752 where production's
        # Python-float `int(t_cont * num_timestep_buckets)` gives 750. Two
        # buckets is a DIFFERENT AdaLayerNorm conditioning from the one the
        # sampler uses, i.e. the exact "constrains a trajectory the robot never
        # executes" failure this instrument exists to fix. Not unique to N=4 and
        # not always in the same direction -- at N=8, t=0.375 -> 376 and
        # t=0.625 -> 624; at N=3, t=1/3 -> 334 vs 333.
        # fp32 would also be exact here (24 mantissa bits covers 0..1000);
        # fp64 is used because `t_values` are already Python floats, so it
        # matches their width with no conversion to reason about.
        # Cost is a [B] fp64 tensor per step -- 64 bytes at B=8.
        # (The endpoint instrument never hit this: t=0 bucketizes to 0 in any
        # dtype, which is why the pre-existing `t0 = zeros(B, dtype=dtype)` was
        # safe.)
        t_i = torch.full((B,), float(t_val), device=eps.device, dtype=torch.float64)
        # nullcontext on the final step only. torch.no_grad() nests correctly
        # inside an outer no_grad (the caller's smooth_no_grad path), so this
        # needs no special-casing there.
        ctx = torch.no_grad() if i < last else contextlib.nullcontext()
        with ctx:
            v = dit_velocity(x, t_i)
        if i == 0:
            v_first = v
        if i < last:
            # Explicit detach even though the no_grad above already severs the
            # graph: it states the intent at the point the value is CARRIED
            # FORWARD, and it keeps the invariant if the context ever changes.
            # `dt` is a Python float so `dt * v` stays in v's dtype, exactly as
            # production's `actions = actions + dt * pred_velocity` does.
            x = (x + dt * v).detach()
        else:
            # The one term the gradient flows through.
            x = x + dt * v
    return x, v_first


def _sample_jittered_timesteps(
    tau_centers: list[float],
    B: int,
    noise_s: float,
    device: torch.device,
    dtype: torch.dtype,
    jitter_std: float = TAU_JITTER_STD,
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


