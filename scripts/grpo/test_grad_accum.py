"""Tests for gradient accumulation (`GRPOConfig.gradient_accumulation_steps`).

Unlike test_balanced_fixes.py — which copies the sampler methods into stubs —
these tests drive the **real** `GRPOTrainer._grpo_update` /
`_grpo_update_inner` on CPU. The substitutions are:

  1. `_prepare_batch`  → builds tiny CPU tensors instead of re-encoding
     observations through the Eagle backbone.
  2. `compute_fm_log_prob` → a 2-parameter analytic stand-in for the K-loop DiT
     forward pass.
  3. the model → `_TinyModel`, a single 2-element trainable parameter standing
     in for the ~20M LoRA params.
  4. the episode buffer → a `_build_chunks()` stub over hand-built chunks.
  5. the optimizer → plain SGD (`_RecordingSGD`) rather than the production
     AdamW, so a reference trajectory can be replayed bit-for-bit. The
     assertions are optimizer-agnostic (they are about gradients and step
     cadence, not about AdamW's moments).
  6. the trainer is built with `GRPOTrainer.__new__` to skip `setup()` (no GPU,
     no model download, no ZMQ server thread).

Everything under test is production code: the accumulation window, the
zero_grad placement, the 1/k loss scale, the non-finite-loss guard, the
epoch-boundary flush, `clip_grad_norm_` + `optimizer.step()` cadence, and every
metric divisor.

The log-prob stub is deliberately **value-pinned**: its output value does not
depend on the trainable parameter (only its gradient does), which makes each
micro-batch's loss value AND gradient constant across the whole run. That is
what lets these tests state exact expectations about per-step gradients
(mean-of-window vs sum-of-window vs per-micro-batch) and about metric values
being *exactly* equal across k. `test_harness_grad_is_param_independent`
asserts that property directly instead of assuming it.

NOTE on the cross-k equality assertions: in PRODUCTION the metric values are
NOT bit-identical across k (within a window all k micro-batches see the same
un-stepped weights, so the log-probs differ). What the equality here pins is the
DIVISOR — n_micro_batches, not n_updates — which is the thing that would
silently inflate every curve k-fold if it regressed.

Run with the project venv (needs torch; CPU is fine):
    .venv/bin/python scripts/grpo/test_grad_accum.py
"""

import contextlib
import io
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

import train_grpo  # noqa: E402  (path set up above)
from grpo_config import GRPOConfig  # noqa: E402
from train_grpo import GRPOTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

@dataclass
class _Chunk:
    """Minimal ActionChunk stand-in — only the fields the update loop reads.

    `_prepare_batch` is stubbed, so the real ActionChunk's observation fields
    are unnecessary; `raw_action` is kept because the stub builds the action
    tensor from it (1x1 so the per-row mean is exact in float32).
    """
    advantage: float
    feat: float                      # drives the stub's per-row f / delta
    group_id: int = 0
    is_anchor: bool = False
    ref_log_prob: Optional[float] = 0.0
    base_log_prob: Optional[float] = None
    tau_samples: Optional[np.ndarray] = None
    raw_action: Optional[np.ndarray] = None


class _ActionHeadStub(nn.Module):
    """`self.model.action_head.model.eval()` needs to resolve to a Module."""

    def __init__(self):
        super().__init__()
        self.model = nn.Identity()


class _TinyModel(nn.Module):
    """One 2-element trainable parameter — stands in for the LoRA params."""

    def __init__(self, w0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.action_head = _ActionHeadStub()


class _RecordingSGD(torch.optim.SGD):
    """SGD that logs zero_grad/step events and snapshots the grad at step time.

    Plain SGD (no momentum, no weight decay) so a reference trajectory can be
    reproduced bit-for-bit from the recorded gradients. The snapshot is taken
    inside step(), i.e. AFTER `clip_grad_norm_` has (possibly) rescaled the
    buffer — so with `max_grad_norm` large enough to never clip, the snapshot
    IS the accumulated gradient.
    """

    def __init__(self, params, lr, events):
        super().__init__(params, lr=lr, momentum=0.0, weight_decay=0.0)
        self._events = events
        self._tracked = [p for group in self.param_groups for p in group["params"]]

    def zero_grad(self, *args, **kwargs):
        self._events.append(("zero_grad", None))
        super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        snapshot = torch.cat([
            (p.grad.detach().clone().reshape(-1) if p.grad is not None
             else torch.zeros(p.numel()))
            for p in self._tracked
        ])
        self._events.append(("step", snapshot))
        return super().step(*args, **kwargs)


class _ScaleGrad(torch.autograd.Function):
    """Identity forward, gradient scaled by `mult` on the way back.

    Lets a test produce the one situation the forward `isfinite(loss)` guard
    cannot catch: a perfectly finite loss whose BACKWARD blows up. The forward
    returns a clone of its input, so the log-prob VALUE — and therefore the loss,
    the ratio, and every metric — is untouched (asserted in
    test_nonfinite_gradient_drops_step_and_protects_weights).

    Both blow-up flavours matter, because `clip_grad_norm_` reports them
    differently and a guard written as `isnan(...)` would catch only the first:
      mult = inf  → the gradient tensor holds ±inf, and summing mixed signs in
                    the matmul backward yields NaN, so the reported norm is NaN.
      mult = 1e30 → the gradient stays finite but the fp32 sum-of-squares in the
                    norm overflows, so the reported norm is +inf.
    """

    @staticmethod
    def forward(ctx, x, mult):
        ctx.mult = mult
        return x.clone()

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.mult, None


@dataclass
class _Run:
    """Everything a test needs to assert on, from one _grpo_update() call."""
    result: dict
    events: list                     # ("prep", B) / ("zero_grad", None) / ("step", grad)
    records: list                    # per-micro-batch {adv, f, delta, nonfinite}
    w0: torch.Tensor
    w_final: torch.Tensor
    stdout: str
    config: GRPOConfig
    entries: list

    # -- convenience views -------------------------------------------------
    @property
    def step_grads(self) -> list:
        return [g for kind, g in self.events if kind == "step"]

    @property
    def n_zero_grads(self) -> int:
        return sum(1 for kind, _ in self.events if kind == "zero_grad")

    @property
    def event_kinds(self) -> list:
        return [kind for kind, _ in self.events]

    @property
    def trained_records(self) -> list:
        return [r for r in self.records if not r["nonfinite"]]

    def reference_grads(self) -> list:
        """Analytic per-micro-batch gradient of the UNSCALED loss, in order."""
        return [_reference_grad(self.config, r) for r in self.trained_records]


def _make_chunks(n_chunks: int, n_groups: int = 1) -> list:
    """n_chunks chunks with alternating advantage signs, all values distinct.

    Distinct advantages guarantee the per-minibatch z-score has non-zero std in
    every micro-batch (so no batch silently contributes a zeroed advantage),
    and alternating signs keep both the reinforcement and suppression branches
    of the surrogate live.
    """
    chunks = []
    for i in range(n_chunks):
        sign = 1.0 if i % 2 == 0 else -1.0
        adv = sign * (1.0 + 0.13 * i)
        feat = 0.25 * ((i % 7) + 1) * (1.0 if i % 3 else -1.0)
        chunks.append(_Chunk(
            advantage=adv,
            feat=feat,
            group_id=i % n_groups,
            ref_log_prob=0.0,
            # Set unconditionally so a kl_coef_base_model > 0 override doesn't
            # get every chunk dropped by the ready_indices filter. Ignored when
            # the base anchor is off (compute_base == False).
            base_log_prob=0.0,
            tau_samples=np.zeros(6, dtype=np.float32),
            raw_action=np.full((1, 1), feat, dtype=np.float32),
        ))
    return chunks


def _row_feature(actions: torch.Tensor, delta_scale: float = 0.05) -> tuple:
    """(f, delta) for a batch of actions — shared by the stub and the reference.

    f is the per-row d(log_prob)/dw (a 2-vector), delta is the per-row
    log-ratio (ref_log_prob is 0 for every chunk, so log_ratio == delta).
    At the default delta_scale=0.05, |delta| < 0.05 keeps ratio = exp(delta)
    strictly inside the clip window, so the surrogate's min/clamp branches
    coincide and the reference gradient is exactly -A * r * f. A larger
    delta_scale deliberately pushes some rows OUTSIDE the window to exercise the
    clamped branch (and make clipfrac non-zero); `_reference_grad` asserts the
    small-scale precondition, so tests using a large scale must not call it.
    """
    a = actions.reshape(actions.shape[0], -1).mean(dim=1).to(torch.float32)
    f = torch.stack([a, torch.sin(a)], dim=1)
    delta = delta_scale * torch.tanh(a)
    return f, delta


def _reference_grad(cfg: GRPOConfig, rec: dict) -> torch.Tensor:
    """d(loss)/dw for one micro-batch, derived independently of train_grpo.

    loss = mean_i[-min(A_i r_i, A_i clamp(r_i))] + kl_coef * mean_i[e^{x_i} - x_i - 1]
    with r_i = e^{delta_i}, x_i = -delta_i, A = per-minibatch z-scored advantage,
    and d(log_prob_i)/dw = f_i.
    """
    adv, f, delta = rec["adv"], rec["f"], rec["delta"]

    A = adv.clone().to(torch.float32)
    if A.numel() > 1:                                    # matches the loop's guard
        A = (A - A.mean()) / (A.std() + 1e-8)            # torch.std == ddof=1

    r = torch.exp(delta)
    lo = 1.0 - cfg.clip_eps_low
    hi = 1.0 + cfg.clip_eps_high
    assert bool(((r > lo) & (r < hi)).all()), (
        "harness invariant broken: ratio left the clip window, so the "
        "reference gradient formula no longer applies"
    )
    # clamp is the identity here → both surrogate branches have gradient A*r*f.
    grad_clip = -((A * r).unsqueeze(1) * f).mean(dim=0)

    x = -delta
    grad_kl = cfg.kl_coef_last_iter * (
        ((torch.exp(x) - 1.0).unsqueeze(1) * (-f)).mean(dim=0)
    )
    return grad_clip + grad_kl


def _reference_masses(cfg: GRPOConfig, rec: dict) -> tuple:
    """(N, D) alive loss mass for one micro-batch — the PAWS pooling terms.

    Mirrors the measure block in _grpo_update_inner: mass is measured on the
    UNWEIGHTED per-row loss |A_post * r| (so it never feeds back on k), N over
    group-negative rows whose lower clip hasn't saturated, D over group-positive
    rows that survive renorm and aren't upper-clipped. `r` is inside the clip
    window by harness construction, so both "alive" filters pass everything.
    """
    adv, delta = rec["adv"], rec["delta"]
    pre_pos = adv > 0
    A = adv.clone().to(torch.float32)
    if A.numel() > 1:
        A = (A - A.mean()) / (A.std() + 1e-8)
    r = torch.exp(delta)
    rl_abs = (A * r).abs()
    n_mass = float(rl_abs[~pre_pos].sum())
    d_mass = float(rl_abs[pre_pos & (A > 0)].sum())
    return n_mass, d_mass


def run_update(
    k: int,
    *,
    epochs: int = 1,
    n_chunks: int = 16,
    mb_size: int = 4,
    lr: float = 0.1,
    max_grad_norm: float = 1e9,
    nonfinite: tuple = (),
    nonfinite_grad: tuple = (),
    grad_blowup: float = float("inf"),
    prepare_returns_none: bool = False,
    balanced: bool = False,
    n_groups: int = 1,
    seed: int = 7,
    w0: tuple = (0.3, -0.2),
    iteration: int = 1,
    delta_scale: float = 0.05,
    config_overrides: Optional[dict] = None,
) -> _Run:
    """Drive the real _grpo_update() once and return everything observable.

    Args:
        k: gradient_accumulation_steps.
        nonfinite: 0-based indices of micro-batches whose loss is forced to
            +inf (simulating the bf16 ratio overflow the guard exists for).
        nonfinite_grad: 0-based indices of micro-batches whose loss stays
            FINITE but whose backward() blows up — the case the forward guard
            cannot see, handled by the gradient-side guard in
            _apply_accumulated_grads.
        grad_blowup: backward multiplier for those micro-batches. inf (default)
            makes clip_grad_norm_ report NaN; 1e30 keeps the gradients finite
            but overflows the fp32 norm, so it reports +inf. See _ScaleGrad.
        prepare_returns_none: make every _prepare_batch call return None
            (the "nothing trainable in this batch" path).
        balanced: use the balanced sampler instead of stratified.
        delta_scale: magnitude of the per-row log-ratio. The default keeps
            every ratio inside the clip window (clipfrac == 0); raise it to
            exercise the clamped surrogate branch, at the cost of
            invalidating `_reference_grad`.
        config_overrides: extra GRPOConfig kwargs (e.g. jitter_pos,
            positive_advantage_weight_scaling, kl_coef_base_model).
    """
    cfg_kwargs = dict(
        device="cpu",
        mini_batch_size=mb_size,
        update_epochs=epochs,
        gradient_accumulation_steps=k,
        balanced_minibatch_training=balanced,
        dynamic_epoch_training=False,
        per_iteration_advantage_norm=False,      # the deliberate design choice
        positive_advantage_weight_scaling=False,
        kl_coef_last_iter=0.2,
        kl_coef_base_model=0.0,                  # keeps base_log_prob out of play
        jitter_pos=0.0,
        jitter_neg=0.0,
        max_grad_norm=max_grad_norm,
        learning_rate=lr,
        seed=seed,
    )
    cfg_kwargs.update(config_overrides or {})
    cfg = GRPOConfig(**cfg_kwargs)

    chunks = _make_chunks(n_chunks, n_groups=n_groups)
    events: list = []
    records: list = []

    model = _TinyModel(w0)
    trainer = GRPOTrainer.__new__(GRPOTrainer)          # skip setup(): no GPU here
    trainer.config = cfg
    trainer.device = torch.device("cpu")
    trainer.model = model
    trainer.optimizer = _RecordingSGD(model.parameters(), lr=lr, events=events)
    trainer.buffer = types.SimpleNamespace(_build_chunks=lambda: list(chunks))
    trainer.iteration = iteration
    import threading
    trainer._model_lock = threading.RLock()

    def _prepare_batch_stub(self, batch):
        if prepare_returns_none:
            return None
        valid = [c for (c, _m) in batch]
        modes = [m for (_c, m) in batch]
        B = len(valid)
        actions = torch.stack([
            torch.from_numpy(c.raw_action) for c in valid
        ]).to(torch.float32).unsqueeze(-1)[:, :, 0]      # [B, 1, 1]
        f, delta = _row_feature(actions, delta_scale)
        adv = torch.tensor([c.advantage for c in valid], dtype=torch.float32)
        records.append({
            "adv": adv, "f": f, "delta": delta,
            "nonfinite": len(records) in nonfinite,
            "nonfinite_grad": len(records) in nonfinite_grad,
            "B": B,
        })
        events.append(("prep", B))
        return {
            "actions": actions,
            "action_masks": torch.ones_like(actions),
            "initial_noise": torch.zeros_like(actions),
            "advantages": adv,
            "backbone_output": {"backbone_features": torch.zeros(B, 1, 1)},
            "state_features": torch.zeros(B, 1, 1),
            "embodiment_id": torch.zeros(B, dtype=torch.long),
            "modes": modes,
        }, valid

    trainer._prepare_batch = types.MethodType(_prepare_batch_stub, trainer)

    def _fake_fm_log_prob(**kw):
        """Value-pinned surrogate: value == delta, d/dw == f (param-independent)."""
        f, delta = _row_feature(kw["actions"], delta_scale)
        w = model.w
        # The inner parenthesisation matters: `(f @ w) - (f @ w.detach())` is
        # bitwise 0.0 (identical matmuls), so the value is EXACTLY delta for
        # every w while the gradient is exactly f. Written as
        # `delta + f @ w - f @ w.detach()` it would associate left and lose
        # ~1e-8 of delta to rounding, making the loss weakly w-dependent and
        # the cross-k invariance checks below fail for a harness reason.
        wterm = f @ w
        if records and records[-1]["nonfinite_grad"]:
            # Finite value, inf gradient: exercises the gradient-side guard.
            wterm = _ScaleGrad.apply(wterm, grad_blowup)
        lp = delta + (wterm - (f @ w.detach()))
        if records and records[-1]["nonfinite"]:
            lp = lp + float("inf")
        if kw.get("return_per_tau"):
            # Honour the real compute_fm_log_prob contract: (mean, [K, B]).
            # Called by GRPOTrainer._jitter_gap_diagnostics on the first
            # minibatch of a jitter-enabled iteration. This surrogate is
            # value-pinned (no tau or noise_for_input dependence), so the honest
            # per-tau breakdown is the mean replicated K times — which satisfies
            # per_tau.mean(0) == lp exactly and makes the measured gap 0. That is
            # the correct answer FOR THIS FAKE, and it keeps this file's tests a
            # crash/plumbing check; the arithmetic is verified against a
            # tau- and noise-sensitive stand-in in test_jitter_metrics.py.
            K = int(kw["n_samples"])
            return lp, lp.detach().unsqueeze(0).expand(K, -1)
        return lp

    real_fm = train_grpo.compute_fm_log_prob
    train_grpo.compute_fm_log_prob = _fake_fm_log_prob
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = trainer._grpo_update()
    finally:
        train_grpo.compute_fm_log_prob = real_fm

    return _Run(
        result=result,
        events=events,
        records=records,
        w0=torch.tensor(w0, dtype=torch.float32),
        w_final=model.w.detach().clone(),
        stdout=buf.getvalue(),
        config=cfg,
        entries=[(c, "fixed") for c in chunks],
    )


def _sgd_replay(w0: torch.Tensor, lr: float, grads: list) -> torch.Tensor:
    """Reference trajectory: plain SGD fed `grads`, one step each."""
    p = nn.Parameter(w0.clone())
    opt = torch.optim.SGD([p], lr=lr, momentum=0.0, weight_decay=0.0)
    for g in grads:
        p.grad = g.clone()
        opt.step()
    return p.detach().clone()


# ---------------------------------------------------------------------------
# check() harness (same style as test_balanced_fixes.py)
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


def _close(a, b, atol=1e-6, rtol=1e-5) -> bool:
    return bool(torch.allclose(a, b, atol=atol, rtol=rtol))


# ---------------------------------------------------------------------------
# Harness self-checks — the reference math is only valid if these hold
# ---------------------------------------------------------------------------

def test_harness_grad_is_param_independent():
    """Per-micro-batch gradients must not depend on w (the reference assumes it).

    If this fails, every "step grad == mean of micro-batch grads" assertion
    below becomes meaningless, because the k=1 run's recorded grads would have
    been taken at different parameter values than the k=2 run's.
    """
    print("\n[Harness] Stub gradient is parameter-independent")

    a = run_update(1, w0=(0.3, -0.2))
    b = run_update(1, w0=(-5.0, 7.5))

    check("same micro-batch count for both w0", len(a.step_grads) == len(b.step_grads))
    all_equal = all(
        torch.equal(ga, gb) for ga, gb in zip(a.step_grads, b.step_grads)
    )
    check("recorded grads bitwise identical across w0", all_equal)
    check(
        "params actually moved (lr is doing something)",
        not torch.equal(a.w0, a.w_final),
    )
    # And the analytic reference agrees with autograd.
    ref = a.reference_grads()
    check(
        "analytic reference matches autograd for every micro-batch",
        len(ref) == len(a.step_grads)
        and all(_close(g, r, atol=1e-7) for g, r in zip(a.step_grads, ref)),
        f"first pair: {a.step_grads[0].tolist()} vs {ref[0].tolist()}",
    )


# ---------------------------------------------------------------------------
# k = 1 must be bit-identical to the pre-accumulation behavior
# ---------------------------------------------------------------------------

def test_k1_is_per_minibatch_update():
    """k=1: one zero_grad → backward(unscaled) → clip → step per micro-batch."""
    print("\n[k=1] Bit-identical to per-minibatch updates")

    r = run_update(1, n_chunks=16, mb_size=4, epochs=2)
    n_mb = 8  # ceil(16/4) per epoch x 2 epochs, stratified yields each entry once

    check(
        f"event sequence is exactly (prep, zero_grad, step) x {n_mb}",
        r.event_kinds == ["prep", "zero_grad", "step"] * n_mb,
        f"got {r.event_kinds}",
    )
    check("n_updates == n_micro_batches == 8", r.result["n_updates"] == 8
          and r.result["n_micro_batches"] == 8,
          f"n_updates={r.result['n_updates']}, "
          f"n_micro_batches={r.result.get('n_micro_batches')}")
    check("one zero_grad per micro-batch", r.n_zero_grads == n_mb, f"{r.n_zero_grads}")

    # The gradient at each step is the UNSCALED micro-batch gradient (no 1/k).
    ref = r.reference_grads()
    check(
        "every step grad == unscaled micro-batch grad",
        all(_close(g, e, atol=1e-7) for g, e in zip(r.step_grads, ref)),
    )
    check(
        "step grads are NOT halved (discriminates against a stray 1/k)",
        not any(_close(g, e / 2.0, atol=1e-7) for g, e in zip(r.step_grads, ref)),
    )

    # Nothing but those steps touched the parameters.
    check(
        "final params == SGD replay of the recorded per-mb grads (bitwise)",
        torch.equal(_sgd_replay(r.w0, r.config.learning_rate, r.step_grads), r.w_final),
        f"{r.w_final.tolist()}",
    )
    # k=1 emits no accumulation-only keys and no accumulation banner.
    check("no grad_accum_steps key at k=1", "grad_accum_steps" not in r.result)
    check("no n_partial_windows key at k=1", "n_partial_windows" not in r.result)
    check(
        "console output unchanged at k=1 (no accumulation banner)",
        "Gradient accumulation" not in r.stdout,
        r.stdout.strip(),
    )


# ---------------------------------------------------------------------------
# k > 1: step count, loss scaling, window contents
# ---------------------------------------------------------------------------

def test_k2_halves_optimizer_steps():
    print("\n[k=2] Half as many optimizer steps")

    k1 = run_update(1, n_chunks=16, mb_size=4, epochs=2)
    k2 = run_update(2, n_chunks=16, mb_size=4, epochs=2)

    check("k=1 → 8 steps", k1.result["n_updates"] == 8, str(k1.result["n_updates"]))
    check("k=2 → 4 steps", k2.result["n_updates"] == 4, str(k2.result["n_updates"]))
    check(
        "k=2 steps == k=1 steps / 2",
        k2.result["n_updates"] * 2 == k1.result["n_updates"],
    )
    check(
        "same number of micro-batches trained (8) in both",
        k1.result["n_micro_batches"] == 8 and k2.result["n_micro_batches"] == 8,
        f"{k1.result['n_micro_batches']} vs {k2.result['n_micro_batches']}",
    )
    check("k=2 → one zero_grad per window (4)", k2.n_zero_grads == 4, str(k2.n_zero_grads))
    check(
        "4 micro-batches per epoch is divisible by k → no partial windows",
        k2.result["n_partial_windows"] == 0,
        str(k2.result["n_partial_windows"]),
    )
    check("grad_accum_steps surfaced as 2", k2.result["grad_accum_steps"] == 2)
    check(
        "accumulation banner printed at k=2",
        "Gradient accumulation: 2 × 4 rows = 8 rows per optimizer step" in k2.stdout,
        k2.stdout.strip(),
    )
    check(
        "each epoch flushes independently (4 steps, not 8 mbs in 3 windows + 1)",
        k2.event_kinds.count("step") == 4,
    )


def test_loss_scaled_by_one_over_k():
    """The accumulated buffer must hold the MEAN, not the SUM, of window grads."""
    print("\n[Scaling] Accumulated gradient == mean of the window's micro-batch grads")

    per_mb = run_update(1, n_chunks=16, mb_size=4, epochs=2).step_grads

    for k, n_steps in ((2, 4), (4, 2)):
        r = run_update(k, n_chunks=16, mb_size=4, epochs=2)
        check(f"k={k}: {n_steps} steps", r.result["n_updates"] == n_steps,
              str(r.result["n_updates"]))
        ok_mean, ok_not_sum = True, True
        for j, g in enumerate(r.step_grads):
            window = per_mb[j * k:(j + 1) * k]
            mean_g = torch.stack(window).mean(dim=0)
            sum_g = torch.stack(window).sum(dim=0)
            if not _close(g, mean_g, atol=1e-7):
                ok_mean = False
            if _close(g, sum_g, atol=1e-7):
                ok_not_sum = False
        check(f"k={k}: every step grad == mean of its {k} micro-batch grads", ok_mean,
              f"step0={r.step_grads[0].tolist()}")
        check(f"k={k}: step grad is NOT the unscaled sum", ok_not_sum)
        # Total displacement is conserved: sum of window means == (1/k) * sum of all.
        total_new = torch.stack(r.step_grads).sum(dim=0)
        total_old = torch.stack(per_mb).sum(dim=0) / k
        check(f"k={k}: no micro-batch gradient discarded (mass check)",
              _close(total_new, total_old, atol=1e-6),
              f"{total_new.tolist()} vs {total_old.tolist()}")


def test_partial_window_flushed_at_epoch_end():
    """3 micro-batches/epoch with k=2 → window(2) + flushed partial(1) per epoch."""
    print("\n[Flush] Partial window flushed at every epoch boundary")

    per_mb = run_update(1, n_chunks=12, mb_size=4, epochs=2).step_grads
    check("baseline k=1 gives 6 micro-batches (3/epoch × 2)", len(per_mb) == 6,
          str(len(per_mb)))

    r = run_update(2, n_chunks=12, mb_size=4, epochs=2)
    check("k=2 → 4 steps (2 per epoch: one full window + one flush)",
          r.result["n_updates"] == 4, str(r.result["n_updates"]))
    check("n_partial_windows == 2 (one per epoch)",
          r.result["n_partial_windows"] == 2, str(r.result["n_partial_windows"]))
    check("all 6 micro-batches trained", r.result["n_micro_batches"] == 6,
          str(r.result["n_micro_batches"]))

    expected = [
        torch.stack(per_mb[0:2]).mean(dim=0),   # epoch 0, full window
        per_mb[2] / 2.0,                        # epoch 0, flushed partial (uniform 1/k)
        torch.stack(per_mb[3:5]).mean(dim=0),   # epoch 1, full window
        per_mb[5] / 2.0,                        # epoch 1, flushed partial
    ]
    check(
        "step grads == [mean(g0,g1), g2/k, mean(g3,g4), g5/k]",
        len(r.step_grads) == 4
        and all(_close(g, e, atol=1e-7) for g, e in zip(r.step_grads, expected)),
        f"got {[g.tolist() for g in r.step_grads]}",
    )
    # The 3rd step must NOT mix epoch 0's trailing micro-batch with epoch 1's
    # first one — that's what a missing epoch-boundary flush would produce.
    carryover = torch.stack([per_mb[2], per_mb[3]]).mean(dim=0)
    check(
        "no window carry-over across the epoch boundary",
        not _close(r.step_grads[1], carryover, atol=1e-7),
    )
    check(
        "no micro-batch gradient discarded (mass check)",
        _close(torch.stack(r.step_grads).sum(dim=0),
               torch.stack(per_mb).sum(dim=0) / 2.0, atol=1e-6),
    )

    # The flush must contribute exactly ONE grad_norms sample, holding the norm
    # of the (partial) accumulated gradient. Without this, the only test that
    # reads grad_norm_* runs a config with n_partial_windows == 0, so a
    # flush-path-specific regression in the record-then-step helper (missing
    # append, double append, wrong value) would go unnoticed. max_grad_norm is
    # 1e9 here, so the recorded step grads ARE the accumulated gradients.
    norms = [float(torch.linalg.vector_norm(g)) for g in r.step_grads]
    check(
        "grad-norm samples == optimizer steps, flushes included",
        math.isclose(r.result["grad_norm_mean"], float(np.mean(norms)), rel_tol=1e-6),
        f"{r.result['grad_norm_mean']} vs {float(np.mean(norms))} over {len(norms)} steps",
    )
    check(
        "grad_norm_max covers the flushed partial windows",
        math.isclose(r.result["grad_norm_max"], float(np.max(norms)), rel_tol=1e-6),
        f"{r.result['grad_norm_max']} vs {float(np.max(norms))}",
    )

    # A window larger than the whole epoch must still flush exactly once/epoch.
    r5 = run_update(5, n_chunks=12, mb_size=4, epochs=2)
    check("k=5 > 3 mbs/epoch → 1 flushed step per epoch (2 total)",
          r5.result["n_updates"] == 2, str(r5.result["n_updates"]))
    check("k=5: both windows counted as partial", r5.result["n_partial_windows"] == 2,
          str(r5.result["n_partial_windows"]))
    check(
        "k=5: step grad == (sum of that epoch's 3 grads)/5",
        _close(r5.step_grads[0], torch.stack(per_mb[0:3]).sum(dim=0) / 5.0, atol=1e-7),
    )


# ---------------------------------------------------------------------------
# Non-finite loss handling
# ---------------------------------------------------------------------------

def test_nonfinite_microbatch_does_not_poison_window():
    """A non-finite micro-batch is skipped; the window keeps its good grads."""
    print("\n[Non-finite] Skip only the bad micro-batch, keep the window")

    per_mb = run_update(1, n_chunks=16, mb_size=4, epochs=1).step_grads
    check("baseline k=1 gives 4 micro-batches", len(per_mb) == 4, str(len(per_mb)))

    r = run_update(2, n_chunks=16, mb_size=4, epochs=1, nonfinite=(1,))

    check("n_skipped_nonfinite == 1", r.result["n_skipped_nonfinite"] == 1,
          str(r.result["n_skipped_nonfinite"]))
    check("n_micro_batches == 3 (the skipped one didn't train)",
          r.result["n_micro_batches"] == 3, str(r.result["n_micro_batches"]))
    check("n_updates == 2 (one full window + one flush)",
          r.result["n_updates"] == 2, str(r.result["n_updates"]))
    check("n_partial_windows == 1", r.result["n_partial_windows"] == 1,
          str(r.result["n_partial_windows"]))

    # mb1 raised no zero_grad and no step; mb2 closed the window opened by mb0.
    check(
        "event sequence shows the skip inside an open window",
        r.event_kinds == [
            "prep", "zero_grad",   # mb0: window opens
            "prep",                # mb1: non-finite → skipped entirely
            "prep", "step",        # mb2: completes the window (no new zero_grad)
            "prep", "zero_grad",   # mb3: opens a fresh window
            "step",                # epoch-end flush
        ],
        f"got {r.event_kinds}",
    )
    expected = [
        torch.stack([per_mb[0], per_mb[2]]).mean(dim=0),
        per_mb[3] / 2.0,
    ]
    check(
        "window 1 grad == mean(g0, g2) — bad mb contributed nothing",
        _close(r.step_grads[0], expected[0], atol=1e-7),
        f"{r.step_grads[0].tolist()} vs {expected[0].tolist()}",
    )
    check(
        "window 1 grad != mean(g0, g1) (bad mb excluded, not merely finite-ized)",
        not _close(r.step_grads[0],
                   torch.stack([per_mb[0], per_mb[1]]).mean(dim=0), atol=1e-7),
    )
    check(
        "window 1 grad != g0/2 (window did not close early on the skip)",
        not _close(r.step_grads[0], per_mb[0] / 2.0, atol=1e-7),
    )
    check("trailing partial window grad == g3/k", _close(r.step_grads[1], expected[1],
                                                         atol=1e-7))
    check(
        "gradients are finite (no inf leaked into the buffer)",
        all(bool(torch.isfinite(g).all()) for g in r.step_grads),
    )


def test_all_nonfinite_takes_no_step():
    print("\n[Non-finite] Every micro-batch bad → no optimizer step at all")

    r = run_update(2, n_chunks=16, mb_size=4, epochs=1, nonfinite=(0, 1, 2, 3))

    check("no step events", r.event_kinds.count("step") == 0, str(r.event_kinds))
    check("n_skipped_nonfinite == 4", r.result.get("n_skipped_nonfinite") == 4,
          str(r.result))
    check("n_updates absent → train()'s did_update is False",
          r.result.get("n_updates", 0) == 0, str(r.result))
    check("no loss/grad keys emitted on the early-return path",
          "loss" not in r.result and "grad_norm_mean" not in r.result, str(r.result))
    check("params untouched (bitwise)", torch.equal(r.w0, r.w_final),
          f"{r.w0.tolist()} -> {r.w_final.tolist()}")


def test_nonfinite_gradient_drops_step_and_protects_weights():
    """Finite loss + inf BACKWARD → step dropped, weights untouched, counted.

    This is the failure the forward `isfinite(loss)` guard cannot see. Before the
    gradient-side guard existed, `clip_grad_norm_` turned the inf buffer into
    NaN (coef 0 × inf), `optimizer.step()` wrote NaN into every LoRA param, and
    the only visible symptom was later minibatches tripping the FORWARD guard —
    while grad_norm_* still looked normal, because the offending norm is
    excluded from grad_norms.
    """
    print("\n[Non-finite grad] Poisoned window is dropped, not stepped")

    per_mb = run_update(1, n_chunks=16, mb_size=4, epochs=1).step_grads
    r = run_update(2, n_chunks=16, mb_size=4, epochs=1, nonfinite_grad=(1,))

    check("n_nonfinite_grad_steps == 1", r.result["n_nonfinite_grad_steps"] == 1,
          str(r.result.get("n_nonfinite_grad_steps")))
    check("n_updates == 1 (only the clean window stepped)",
          r.result["n_updates"] == 1, str(r.result["n_updates"]))
    check("n_micro_batches == 4 (all four did reach backward)",
          r.result["n_micro_batches"] == 4, str(r.result["n_micro_batches"]))
    check("forward-guard counter untouched (loss stayed finite)",
          r.result["n_skipped_nonfinite"] == 0, str(r.result["n_skipped_nonfinite"]))

    check(
        "event sequence: poisoned window zeroes without stepping",
        r.event_kinds == [
            "prep", "zero_grad",   # mb0 opens window 1
            "prep", "zero_grad",   # mb1 closes it → non-finite → zero, NO step
            "prep", "zero_grad",   # mb2 opens window 2
            "prep", "step",        # mb3 closes it → clean step
        ],
        f"got {r.event_kinds}",
    )
    check(
        "the surviving step used ONLY the clean window's micro-batches",
        len(r.step_grads) == 1
        and _close(r.step_grads[0],
                   torch.stack([per_mb[2], per_mb[3]]).mean(dim=0), atol=1e-7),
        f"got {[g.tolist() for g in r.step_grads]}",
    )
    check("final params are finite (no NaN reached the weights)",
          bool(torch.isfinite(r.w_final).all()), str(r.w_final.tolist()))
    check(
        "final params == SGD replay of the one surviving step (bitwise)",
        torch.equal(_sgd_replay(r.w0, r.config.learning_rate, r.step_grads),
                    r.w_final),
    )
    check("grad_norm_mean excludes the dropped window",
          math.isclose(r.result["grad_norm_mean"],
                       float(torch.linalg.vector_norm(r.step_grads[0])),
                       rel_tol=1e-6),
          f"{r.result['grad_norm_mean']}")
    check("operator sees a WARNING on stdout",
          "non-finite accumulated gradient" in r.stdout, r.stdout.strip())

    # Not a k>1-only protection: k=1 gets the same guard.
    r1 = run_update(1, n_chunks=8, mb_size=4, epochs=1, nonfinite_grad=(0,))
    check("k=1: poisoned micro-batch dropped, clean one still steps",
          (r1.result["n_updates"], r1.result["n_nonfinite_grad_steps"]) == (1, 1),
          f"n_updates={r1.result['n_updates']}, "
          f"dropped={r1.result.get('n_nonfinite_grad_steps')}")
    check("k=1: final params finite", bool(torch.isfinite(r1.w_final).all()))

    # --- The EPOCH-BOUNDARY FLUSH path must be guarded too -------------------
    # The flush shares _apply_accumulated_grads, but nothing above ever poisons
    # a PARTIAL window: restricting the guard to full windows only
    # (`... and accum_count == accum_steps`) leaves every other check green
    # while the flush writes NaN into the weights. 3 micro-batches at k=2 =
    # one full window + one flushed partial; poison the partial one.
    rf = run_update(2, n_chunks=12, mb_size=4, epochs=1, nonfinite_grad=(2,))
    check("flush path: n_partial_windows == 1 (the poisoned one was attempted)",
          rf.result["n_partial_windows"] == 1, str(rf.result["n_partial_windows"]))
    check("flush path: the poisoned flush was dropped, not stepped",
          (rf.result["n_updates"], rf.result["n_nonfinite_grad_steps"]) == (1, 1),
          f"n_updates={rf.result['n_updates']}, "
          f"dropped={rf.result.get('n_nonfinite_grad_steps')}")
    check("flush path: no NaN reached the weights",
          bool(torch.isfinite(rf.w_final).all()), str(rf.w_final.tolist()))
    check("flush path: only the clean full window stepped",
          rf.event_kinds.count("step") == 1, str(rf.event_kinds))

    # --- Both blow-up flavours, not just the NaN one -------------------------
    # An inf-valued gradient tensor makes clip_grad_norm_ report NaN (mixed
    # signs cancel in the matmul backward). Large-but-finite gradients instead
    # overflow the fp32 norm and report +inf. A guard written as isnan() would
    # catch only the first, so pin both.
    r_inf = run_update(2, n_chunks=16, mb_size=4, epochs=1, nonfinite_grad=(1,),
                       grad_blowup=1e30)
    check("finite-but-huge gradients also drop the step (norm overflows to +inf)",
          (r_inf.result["n_updates"], r_inf.result["n_nonfinite_grad_steps"]) == (1, 1),
          f"n_updates={r_inf.result['n_updates']}, "
          f"dropped={r_inf.result.get('n_nonfinite_grad_steps')}")
    check("huge-gradient case: weights protected",
          bool(torch.isfinite(r_inf.w_final).all()), str(r_inf.w_final.tolist()))
    check("huge-gradient case reports inf (not nan) — both flavours covered",
          "(inf)" in r_inf.stdout and "(nan)" in r.stdout,
          f"huge={r_inf.stdout.strip()[:80]!r} nan={r.stdout.strip()[:80]!r}")


def test_all_windows_nonfinite_grad_leaves_model_untouched():
    """Every window poisoned → no step at all → iteration treated as skipped."""
    print("\n[Non-finite grad] All windows poisoned → model bit-identical")

    r = run_update(2, n_chunks=16, mb_size=4, epochs=2,
                   nonfinite_grad=tuple(range(8)))

    check("no step events", r.event_kinds.count("step") == 0, str(r.event_kinds))
    check("params bitwise unchanged", torch.equal(r.w0, r.w_final),
          f"{r.w0.tolist()} -> {r.w_final.tolist()}")
    check("n_updates == 0 → train()'s did_update is False",
          r.result.get("n_updates", 0) == 0, str(r.result))
    check("early return still reports WHY (4 dropped windows, 8 trained mbs)",
          r.result.get("n_nonfinite_grad_steps") == 4
          and r.result.get("n_micro_batches") == 8,
          str(r.result))
    check("no loss/grad keys on the early-return path",
          "loss" not in r.result and "grad_norm_mean" not in r.result,
          str(r.result))
    check("operator sees the summary line",
          "No optimizer step survived" in r.stdout, r.stdout.strip())


def test_no_step_when_nothing_prepared():
    """_prepare_batch returning None everywhere must not step on an empty buffer."""
    print("\n[Guard] Empty accumulation buffer is never stepped")

    r = run_update(2, n_chunks=16, mb_size=4, epochs=2, prepare_returns_none=True)

    check("no step events", r.event_kinds.count("step") == 0, str(r.event_kinds))
    check("no zero_grad events", r.n_zero_grads == 0, str(r.n_zero_grads))
    check("result is the empty early-return dict", r.result == {}, str(r.result))
    check("params untouched (bitwise)", torch.equal(r.w0, r.w_final))


# ---------------------------------------------------------------------------
# Metrics bookkeeping
# ---------------------------------------------------------------------------

def test_per_microbatch_metrics_invariant_to_k():
    """Loss/ratio/KL/clipfrac are per-micro-batch means → identical for every k.

    This is the regression test for the divisor: if any of them still divided by
    n_updates, every value would be inflated ~k-fold at k>1.

    The equality is exact only because this harness's log-prob VALUE is
    parameter-independent, so the micro-batch losses don't depend on when the
    optimizer stepped. In production they do shift slightly across k — what this
    pins is the divisor, not bit-equality of the curves. Note that with the
    default delta_scale, `clipfrac`, `n_skipped_nonfinite`, `actual_epochs` and
    `n_pos_flipped_by_renorm` are constants here; the discriminating entries are
    loss / clip_loss / kl_loss_last_iter / mean_ratio / mean_log_ratio_abs, and
    test_clipped_rows_metrics_invariant_to_k makes clipfrac non-trivial.
    """
    print("\n[Metrics] Per-micro-batch means are invariant to k")

    runs = {k: run_update(k, n_chunks=16, mb_size=4, epochs=2) for k in (1, 2, 4)}
    base = runs[1].result

    invariant_keys = (
        "loss", "clip_loss", "kl_loss_last_iter", "clipfrac", "mean_ratio",
        "mean_log_ratio_abs", "ratio_max", "ratio_min", "n_micro_batches",
        "n_pos_flipped_by_renorm", "actual_epochs", "n_skipped_nonfinite",
    )
    for k, r in runs.items():
        if k == 1:
            continue
        for key in invariant_keys:
            check(
                f"k={k}: {key} identical to k=1 ({base[key]!r})",
                r.result[key] == base[key],
                f"got {r.result[key]!r}",
            )

    check(
        "n_updates DOES change with k (8 / 4 / 2) — test isn't vacuous",
        [runs[k].result["n_updates"] for k in (1, 2, 4)] == [8, 4, 2],
        str([runs[k].result["n_updates"] for k in (1, 2, 4)]),
    )
    check(
        "loss is non-trivial (a k-fold inflation would be detectable)",
        abs(base["loss"]) > 1e-6,
        str(base["loss"]),
    )


def test_grad_norm_tracks_accumulated_gradient():
    print("\n[Metrics] grad_norm_* measures the accumulated gradient")

    r = run_update(2, n_chunks=16, mb_size=4, epochs=2)
    norms = [float(torch.linalg.vector_norm(g)) for g in r.step_grads]

    check("one grad-norm sample per optimizer step",
          len(norms) == r.result["n_updates"], f"{len(norms)} vs {r.result['n_updates']}")
    check("grad_norm_mean == mean over steps of ||accumulated grad||",
          math.isclose(r.result["grad_norm_mean"], float(np.mean(norms)), rel_tol=1e-6),
          f"{r.result['grad_norm_mean']} vs {float(np.mean(norms))}")
    check("grad_norm_max == max over steps of ||accumulated grad||",
          math.isclose(r.result["grad_norm_max"], float(np.max(norms)), rel_tol=1e-6),
          f"{r.result['grad_norm_max']} vs {float(np.max(norms))}")

    # With clipping active, the reported norm must still be the PRE-clip norm of
    # the accumulated gradient, while the applied gradient is the clipped one.
    per_mb = run_update(1, n_chunks=16, mb_size=4, epochs=1).step_grads
    window_mean = torch.stack(per_mb[0:2]).mean(dim=0)
    pre_clip_norm = float(torch.linalg.vector_norm(window_mean))
    tight = pre_clip_norm / 10.0
    rc = run_update(2, n_chunks=16, mb_size=4, epochs=1, max_grad_norm=tight)
    check(
        "clipping fires: applied grad norm == max_grad_norm",
        math.isclose(float(torch.linalg.vector_norm(rc.step_grads[0])), tight,
                     rel_tol=1e-5),
        f"{float(torch.linalg.vector_norm(rc.step_grads[0]))} vs {tight}",
    )
    check(
        "grad_norm_max reports the pre-clip ACCUMULATED norm",
        math.isclose(rc.result["grad_norm_max"], pre_clip_norm, rel_tol=1e-5),
        f"{rc.result['grad_norm_max']} vs {pre_clip_norm}",
    )


def test_n_updates_counts_real_steps():
    print("\n[Metrics] n_updates == number of real optimizer.step() calls")

    for k in (1, 2, 3, 4, 8):
        r = run_update(k, n_chunks=16, mb_size=4, epochs=2)
        real_steps = r.event_kinds.count("step")
        check(
            f"k={k}: n_updates ({r.result['n_updates']}) == observed steps ({real_steps})",
            r.result["n_updates"] == real_steps,
        )
        # 4 micro-batches per epoch, flushed per epoch.
        expected = 2 * math.ceil(4 / k)
        check(f"k={k}: steps == 2 epochs × ceil(4/{k}) = {expected}",
              real_steps == expected, str(real_steps))


# ---------------------------------------------------------------------------
# Interaction with the balanced sampler's early-terminating epochs
# ---------------------------------------------------------------------------

def test_balanced_sampler_epoch_boundary():
    """Balanced sampler + accumulation: steps == Σ_epochs ceil(mbs_e / k).

    _iter_balanced_minibatches anchors epoch length to ceil(n/mb_size) but can
    return early when the majority pool drains, so mbs_e is neither constant
    nor a multiple of k. The expected step count is derived by re-driving the
    production sampler with the same per-epoch seeds the update loop uses.
    """
    print("\n[Balanced] Early-terminating epochs still flush their partial window")

    n_chunks, mb_size, epochs, seed, iteration = 20, 4, 2, 7, 1

    # Mirror _grpo_update_inner's per-epoch RNG seeding to count micro-batches.
    probe = GRPOTrainer.__new__(GRPOTrainer)
    probe.config = GRPOConfig(
        device="cpu", mini_batch_size=mb_size, update_epochs=epochs,
        balanced_minibatch_training=True, dynamic_epoch_training=False, seed=seed,
    )
    entries = [(c, "fixed") for c in _make_chunks(n_chunks)]
    mbs_per_epoch = []
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + iteration * 100 + epoch)
        mbs_per_epoch.append(
            len(list(probe._iter_balanced_minibatches(entries, rng)))
        )
    check(f"sampler yields {mbs_per_epoch} micro-batches per epoch",
          all(m > 0 for m in mbs_per_epoch), str(mbs_per_epoch))

    for k in (1, 2, 3):
        r = run_update(k, n_chunks=n_chunks, mb_size=mb_size, epochs=epochs,
                       balanced=True, seed=seed, iteration=iteration)
        expected_steps = sum(math.ceil(m / k) for m in mbs_per_epoch)
        check(
            f"balanced k={k}: n_updates == Σ ceil(mbs_e/{k}) = {expected_steps}",
            r.result["n_updates"] == expected_steps,
            f"got {r.result['n_updates']} (mbs/epoch={mbs_per_epoch})",
        )
        check(
            f"balanced k={k}: every micro-batch trained ({sum(mbs_per_epoch)})",
            r.result["n_micro_batches"] == sum(mbs_per_epoch),
            f"got {r.result['n_micro_batches']}",
        )
        if k > 1:
            expected_partial = sum(1 for m in mbs_per_epoch if m % k)
            check(
                f"balanced k={k}: n_partial_windows == {expected_partial}",
                r.result["n_partial_windows"] == expected_partial,
                f"got {r.result['n_partial_windows']}",
            )


def test_multi_group_stratified_with_accumulation():
    """Sanity: multi-group stratified batches behave the same way."""
    print("\n[Stratified] Multi-group batches accumulate identically")

    per_mb = run_update(1, n_chunks=16, mb_size=4, epochs=1, n_groups=4).step_grads
    r = run_update(2, n_chunks=16, mb_size=4, epochs=1, n_groups=4)

    check("k=1 → 4 micro-batches", len(per_mb) == 4, str(len(per_mb)))
    check("k=2 → 2 steps", r.result["n_updates"] == 2, str(r.result["n_updates"]))
    ok = all(
        _close(g, torch.stack(per_mb[j * 2:(j + 1) * 2]).mean(dim=0), atol=1e-7)
        for j, g in enumerate(r.step_grads)
    )
    check("step grads == window means across 4 groups", ok)


# ---------------------------------------------------------------------------
# PAWS mass accounting (positive_advantage_weight_scaling)
# ---------------------------------------------------------------------------

def test_paws_mass_pools_per_trained_microbatch():
    """Pooled PAWS mass must correspond exactly to TRAINED rows, for any k.

    The mass commits per micro-batch right after the non-finite guard, so
    window boundaries must be irrelevant: N_iter/D_iter (and therefore
    pos_adv_weight_k) are invariant to k, and a skipped micro-batch contributes
    nothing.
    """
    print("\n[PAWS] Pooled mass == trained rows, independent of k")

    # target_ratio > 1 keeps k live (> 1) in both regimes the update passes
    # through — the pre-warm-up analytic prior (k == target_ratio) and the
    # measured ratio (k == target_ratio * N/D, and N ~ D here because the
    # per-minibatch z-score forces sum_pos|A| == sum_neg|A|). Neither the pooled
    # mass nor its k-invariance depends on the value.
    paws = dict(
        positive_advantage_weight_scaling=True,
        positive_advantage_weight_max=10.0,
        positive_advantage_weight_target_ratio=2.0,
    )

    runs = {
        k: run_update(k, n_chunks=16, mb_size=4, epochs=2,
                      config_overrides=paws)
        for k in (1, 2, 4)
    }
    base = runs[1].result

    # Independent expectation: sum the analytic per-micro-batch masses.
    exp_n, exp_d = 0.0, 0.0
    for rec in runs[1].trained_records:
        n_m, d_m = _reference_masses(runs[1].config, rec)
        exp_n += n_m
        exp_d += d_m
    check(
        "k=1 pooled masses match the analytic per-micro-batch sum",
        math.isclose(base["pos_adv_alive_neg_mass"], exp_n, rel_tol=1e-5)
        and math.isclose(base["pos_adv_pos_mass"], exp_d, rel_tol=1e-5),
        f"N={base['pos_adv_alive_neg_mass']} vs {exp_n}, "
        f"D={base['pos_adv_pos_mass']} vs {exp_d}",
    )
    check("PAWS weight k is actually > 1 (weighting is live, not a no-op)",
          base["pos_adv_weight_k"] > 1.0, str(base["pos_adv_weight_k"]))

    for k, r in runs.items():
        if k == 1:
            continue
        for key in ("pos_adv_alive_neg_mass", "pos_adv_pos_mass", "pos_adv_weight_k"):
            check(f"accum k={k}: {key} identical to k=1",
                  r.result[key] == base[key],
                  f"{r.result[key]} vs {base[key]}")

    # A dropped micro-batch must contribute no mass — under accumulation too.
    clean = run_update(2, n_chunks=16, mb_size=4, epochs=1,
                       config_overrides=paws)
    skipped = run_update(2, n_chunks=16, mb_size=4, epochs=1, nonfinite=(1,),
                         config_overrides=paws)
    exp_n2 = exp_d2 = 0.0
    for rec in skipped.trained_records:
        n_m, d_m = _reference_masses(skipped.config, rec)
        exp_n2 += n_m
        exp_d2 += d_m
    check(
        "non-finite micro-batch adds no mass (matches trained-only sum)",
        math.isclose(skipped.result["pos_adv_alive_neg_mass"], exp_n2, rel_tol=1e-5)
        and math.isclose(skipped.result["pos_adv_pos_mass"], exp_d2, rel_tol=1e-5),
        f"N={skipped.result['pos_adv_alive_neg_mass']} vs {exp_n2}",
    )
    check(
        "skipped-run mass is strictly less than the clean run's",
        skipped.result["pos_adv_alive_neg_mass"] < clean.result["pos_adv_alive_neg_mass"],
        f"{skipped.result['pos_adv_alive_neg_mass']} vs "
        f"{clean.result['pos_adv_alive_neg_mass']}",
    )


def test_paws_cold_start_uses_target_ratio_not_one():
    """PAWS must never open an iteration at k = 1.0, and must carry no state.

    The regression this pins: k used to be gated on a cross-iteration EMA of
    (N, D) that was NOT checkpointed, so the first update of a fresh run AND the
    first update after --resume-from ran the WHOLE iteration at k = 1.0 whatever
    positive_advantage_weight_target_ratio said. k = 1.0 is not a mild warm-up:
    the realized post-weighting ratio is k*D/N, so at N ~ D (which per-minibatch
    renorm forces, since the z-score makes sum_pos|A| == sum_neg|A|) it drops the
    net update direction k*D - N from a positive surplus to ~0 and inverts its
    sign. Observed consequence when resuming a 67%-success run: collapse to
    ~4% in one iteration.

    Four properties, all of which the old code violated:
      1. A one-micro-batch iteration — i.e. an iteration that is nothing BUT its
         cold start — runs at the analytic prior k == target_ratio, not 1.0.
      2. The realized ratio tracks target_ratio, i.e. the mechanism actually
         delivers its target over a full iteration.
      3. k_min/k_max bracket k_last, so a within-iteration excursion is visible.
      4. Two independent trainers with no shared history produce identical k —
         there is no cross-iteration state left to lose, so resume is seamless
         by construction rather than by remembering to persist something.
    """
    print("\n[PAWS] Cold start opens at target_ratio, and carries no state")

    tratio = 1.75
    paws = dict(
        positive_advantage_weight_scaling=True,
        positive_advantage_weight_max=10.0,
        positive_advantage_weight_target_ratio=tratio,
    )

    # Property 1, measured directly rather than through an aggregate: 4 chunks at
    # mb_size 4 for 1 epoch trains exactly ONE micro-batch, and the warm-up is at
    # least 1 micro-batch long, so k_last IS the cold-start k. Asserting on
    # k_last (not a dedicated "first" metric) keeps this keyed to a value the
    # weights actually saw.
    solo = run_update(1, n_chunks=4, mb_size=4, epochs=1, config_overrides=paws)
    check("the solo run trained exactly one micro-batch",
          solo.result["n_micro_batches"] == 1,
          str(solo.result.get("n_micro_batches")))
    check(
        "cold-start micro-batch uses k == target_ratio (not 1.0)",
        math.isclose(solo.result["pos_adv_weight_k"], tratio, rel_tol=1e-9),
        f"k={solo.result['pos_adv_weight_k']} vs {tratio}",
    )
    # k_min/k_max track MEASURED k's only, so an iteration whose single
    # micro-batch ran at the unmeasured prior must report neither — otherwise
    # k_min would be pinned to the config-derived prior and hide the real spread.
    check(
        "an all-prior iteration reports no k_min/k_max",
        "pos_adv_weight_k_min" not in solo.result
        and "pos_adv_weight_k_max" not in solo.result,
        f"min={solo.result.get('pos_adv_weight_k_min')} "
        f"max={solo.result.get('pos_adv_weight_k_max')}",
    )

    # Every trainer here is built fresh via GRPOTrainer.__new__ with no EMA
    # attributes of any kind, which is exactly the post-resume state.
    # 64 chunks / mb 4 / 2 epochs = 32 micro-batches, so the 8-micro-batch
    # warm-up gives way to the measured ratio for the remaining 24.
    r = run_update(1, n_chunks=64, mb_size=4, epochs=2, config_overrides=paws)
    res = r.result

    check(
        "k_last is live (measured from the pool, still > 1)",
        res["pos_adv_weight_k"] > 1.0,
        str(res["pos_adv_weight_k"]),
    )
    check(
        "k_min <= k_last <= k_max, and the bracket respects [1, max]",
        res["pos_adv_weight_k_min"] <= res["pos_adv_weight_k"]
        <= res["pos_adv_weight_k_max"]
        and res["pos_adv_weight_k_min"] >= 1.0
        and res["pos_adv_weight_k_max"] <= 10.0,
        f"min={res['pos_adv_weight_k_min']} last={res['pos_adv_weight_k']} "
        f"max={res['pos_adv_weight_k_max']}",
    )
    # Realized ratio is pooled per micro-batch at the k that micro-batch was
    # actually weighted by (Dw_iter), so it is NOT reconstructible from the
    # aggregate k_last * D / N — assert it differs from that reconstruction, which
    # is what the earlier version of this metric computed.
    naive = res["pos_adv_weight_k"] * res["pos_adv_pos_mass"] / res["pos_adv_alive_neg_mass"]
    check(
        "realized ratio is the weighted pool, not k_last * D / N",
        not math.isclose(res["pos_adv_realized_ratio"], naive, rel_tol=1e-9),
        f"realized={res['pos_adv_realized_ratio']} naive={naive}",
    )
    # Not exact: the warm-up micro-batches were weighted at the prior rather than
    # at the measured ratio, and they are pooled too. What the assertion pins is
    # that the DELIVERED ratio tracks the target rather than sitting at the 1.0
    # the old cold-start path produced.
    check(
        "realized reinforcement:erosion ratio tracks target_ratio",
        math.isclose(res["pos_adv_realized_ratio"], tratio, rel_tol=0.05),
        f"realized={res['pos_adv_realized_ratio']} vs target={tratio}",
    )

    # Resume-equivalence: a second, independent trainer over the same data must
    # reproduce k exactly. Under the EMA design this pair differed (one warm,
    # one cold) — which is precisely what made a resumed iteration untrainable.
    r2 = run_update(1, n_chunks=64, mb_size=4, epochs=2, config_overrides=paws)
    for key in ("pos_adv_weight_k", "pos_adv_weight_k_min",
                "pos_adv_realized_ratio", "pos_adv_alive_neg_mass",
                "pos_adv_pos_mass"):
        check(f"fresh-vs-fresh trainer: {key} identical",
              r2.result[key] == res[key],
              f"{r2.result[key]} vs {res[key]}")

    # A real __init__ (cheap: no model, no GPU, no server thread) must not
    # create the EMA slots at all — the state is gone, not merely unused.
    fresh = GRPOTrainer(GRPOConfig(device="cpu", **paws))
    check(
        "no cross-iteration PAWS attribute exists on a real trainer",
        not any(hasattr(fresh, a) for a in
                ("_pos_scale_N_ema", "_pos_scale_D_ema")),
    )

    # target_ratio must reach the weights on the very first iteration, so a
    # resume that CHANGES it is not silently a no-op.
    hot = run_update(1, n_chunks=4, mb_size=4, epochs=1,
                     config_overrides={**paws,
                                       "positive_advantage_weight_target_ratio": 3.5})
    check(
        "changing target_ratio changes the cold-start k",
        math.isclose(hot.result["pos_adv_weight_k"], 3.5, rel_tol=1e-9),
        f"{hot.result['pos_adv_weight_k']} vs 3.5 "
        f"(baseline solo run was {solo.result['pos_adv_weight_k']})",
    )

    # Under per_iteration_advantage_norm the zero-mean identity does not hold, so
    # there is no analytic prior and the cold start correctly stays at 1.0 — with
    # the warm-up held to a single micro-batch to bound the exposure.
    pin = run_update(1, n_chunks=4, mb_size=4, epochs=1,
                     config_overrides={**paws,
                                       "per_iteration_advantage_norm": True})
    check(
        "per_iteration_advantage_norm cold start stays at k = 1.0",
        math.isclose(pin.result["pos_adv_weight_k"], 1.0, rel_tol=1e-9),
        str(pin.result["pos_adv_weight_k"]),
    )


# ---------------------------------------------------------------------------
# Jitter-GRPO per-branch row accumulators
# ---------------------------------------------------------------------------

def test_jitter_branch_metrics_invariant_to_k():
    """Paired jitter doubles the micro-batches; per-branch metrics stay row-weighted.

    The `_fixed` / `_jitter` accumulators are per-ROW sums divided by row counts,
    so they must not shift when several micro-batches share an optimizer step.
    """
    print("\n[Jitter] Per-branch row metrics unaffected by accumulation")

    jit = dict(jitter_pos=0.05, jitter_neg=0.05, jitter_paired=True)
    plain = run_update(1, n_chunks=16, mb_size=4, epochs=1)
    j1 = run_update(1, n_chunks=16, mb_size=4, epochs=1, config_overrides=jit)
    j2 = run_update(2, n_chunks=16, mb_size=4, epochs=1, config_overrides=jit)

    check(
        "paired jitter doubles the micro-batch count (4 → 8)",
        plain.result["n_micro_batches"] == 4 and j1.result["n_micro_batches"] == 8,
        f"{plain.result['n_micro_batches']} / {j1.result['n_micro_batches']}",
    )
    check("jitter + k=2 halves the steps (8 → 4)", j2.result["n_updates"] == 4,
          str(j2.result["n_updates"]))
    check("jitter + k=2 still trains all 8 micro-batches",
          j2.result["n_micro_batches"] == 8, str(j2.result["n_micro_batches"]))

    for key in ("mean_ratio_fixed", "mean_ratio_jitter",
                "mean_log_ratio_abs_fixed", "mean_log_ratio_abs_jitter",
                "kl_loss_last_iter_fixed", "kl_loss_last_iter_jitter"):
        check(f"{key} present and identical across k", key in j1.result
              and key in j2.result and j1.result[key] == j2.result[key],
              f"{j1.result.get(key)} vs {j2.result.get(key)}")
    for key in ("clipfrac_fixed_pos", "clipfrac_fixed_neg",
                "clipfrac_jitter_pos", "clipfrac_jitter_neg"):
        if key in j1.result:
            check(f"{key} identical across k", j1.result.get(key) == j2.result.get(key),
                  f"{j1.result.get(key)} vs {j2.result.get(key)}")


def test_base_model_kl_metric_divisor():
    """kl_loss_base_model is a per-micro-batch mean too → invariant to k."""
    print("\n[Base KL] kl_loss_base_model divides by micro-batches, not steps")

    base_kl = dict(kl_coef_base_model=0.2)
    a = run_update(1, n_chunks=16, mb_size=4, epochs=2, config_overrides=base_kl)
    b = run_update(4, n_chunks=16, mb_size=4, epochs=2, config_overrides=base_kl)

    check("base-model KL term emitted", "kl_loss_base_model" in a.result, str(a.result))
    check("kl_loss_base_model identical across k",
          a.result["kl_loss_base_model"] == b.result["kl_loss_base_model"],
          f"{a.result['kl_loss_base_model']} vs {b.result['kl_loss_base_model']}")
    check("steps still drop by k (8 → 2)",
          (a.result["n_updates"], b.result["n_updates"]) == (8, 2),
          f"{a.result['n_updates']} / {b.result['n_updates']}")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_clipped_rows_metrics_invariant_to_k():
    """With the ratio pushed OUTSIDE the clip window, clipfrac is non-trivial.

    Every other test keeps |log_ratio| < 0.05, so `clipfrac` is exactly 0.0 and
    the surrogate's clamped branch never runs — which makes the clipfrac entry in
    test_per_microbatch_metrics_invariant_to_k a comparison of 0.0 against
    itself. Here delta_scale=0.35 puts some rows past 1 + clip_eps_high, so the
    clamp branch and the clipfrac accounting are actually exercised under
    accumulation.
    """
    print("\n[Clipping] Clamped surrogate branch under accumulation")

    kw = dict(n_chunks=16, mb_size=4, epochs=2, delta_scale=0.35)
    a = run_update(1, **kw)
    b = run_update(2, **kw)

    check("clipfrac is non-trivial (clamped branch really fires)",
          0.0 < a.result["clipfrac"] < 1.0, str(a.result["clipfrac"]))
    check("ratio_max exceeds the upper clip bound",
          a.result["ratio_max"] > 1.0 + a.config.clip_eps_high,
          f"{a.result['ratio_max']} vs {1.0 + a.config.clip_eps_high}")
    for key in ("clipfrac", "loss", "clip_loss", "mean_ratio",
                "mean_log_ratio_abs", "ratio_max", "ratio_min"):
        check(f"clipped {key} identical across k", a.result[key] == b.result[key],
              f"{a.result[key]} vs {b.result[key]}")
    check("clipped run still halves the optimizer steps (8 → 4)",
          (a.result["n_updates"], b.result["n_updates"]) == (8, 4),
          f"{a.result['n_updates']} / {b.result['n_updates']}")
    check("clipped run still trains all 8 micro-batches",
          b.result["n_micro_batches"] == 8, str(b.result["n_micro_batches"]))


def test_result_shapes_are_loggable():
    """Every dict _grpo_update_inner can return must survive the real _log_metrics.

    The accumulation work added two result keys and a new early-return shape
    (trained micro-batches but zero optimizer steps, which previously could not
    happen). This drives the production `_log_metrics` with a recording writer
    over every reachable shape and asserts no exception, no duplicate TB tag, and
    no non-finite scalar — the failure mode being a key that only shows up on a
    rare path and then throws or poisons a chart mid-run.
    """
    print("\n[Logging] All result shapes survive the real _log_metrics")

    class _RecordingWriter:
        def __init__(self):
            self.calls = []

        def add_scalar(self, tag, value, step):
            self.calls.append((tag, float(value), step))

    shapes = [
        ("k=1", run_update(1, n_chunks=16, mb_size=4, epochs=2).result),
        ("k=2", run_update(2, n_chunks=16, mb_size=4, epochs=2).result),
        ("k=2 one bad grad",
         run_update(2, n_chunks=16, mb_size=4, epochs=1, nonfinite_grad=(1,)).result),
        ("all grads bad (n_updates=0 with trained mbs)",
         run_update(2, n_chunks=16, mb_size=4, epochs=2,
                    nonfinite_grad=tuple(range(8))).result),
        ("all losses bad",
         run_update(2, n_chunks=16, mb_size=4, epochs=1, nonfinite=(0, 1, 2, 3)).result),
        ("nothing prepared",
         run_update(2, n_chunks=16, mb_size=4, epochs=2, prepare_returns_none=True).result),
    ]

    for label, stats in shapes:
        for dyn in (False, True):
            trainer = GRPOTrainer.__new__(GRPOTrainer)
            trainer.config = GRPOConfig(
                device="cpu", use_wandb=False, dynamic_epoch_training=dyn
            )
            trainer.writer = _RecordingWriter()
            try:
                trainer._log_metrics(
                    7, {"success_rate": 0.5}, stats, lr=1.5e-5, iter_time=1.0,
                    phase_times={"collect": 1.0, "update": 2.0}, lora_delta_norm=0.01,
                )
                raised = None
            except Exception as exc:  # noqa: BLE001 - the assertion IS "nothing raises"
                raised = f"{type(exc).__name__}: {exc}"
            check(f"{label} (dynamic_epoch={dyn}): logs without raising",
                  raised is None, str(raised))
            if raised is not None:
                continue
            tags = [t for t, _, _ in trainer.writer.calls]
            dupes = sorted({t for t in tags if tags.count(t) > 1})
            nonfinite = [t for t, v, _ in trainer.writer.calls if not math.isfinite(v)]
            check(f"{label} (dynamic_epoch={dyn}): no duplicate TB tags",
                  not dupes, str(dupes))
            check(f"{label} (dynamic_epoch={dyn}): no non-finite scalars",
                  not nonfinite, str(nonfinite))
            # The counters must actually be EMITTED, not merely loggable — a
            # deleted add_scalar block would otherwise pass every check above.
            for tag in ("train/n_updates", "train/n_micro_batches",
                        "train/n_skipped_nonfinite", "train/n_nonfinite_grad_steps"):
                check(f"{label} (dynamic_epoch={dyn}): {tag} emitted",
                      tag in tags, str(sorted(tags)))

    # The counters must make the "trained but nothing applied" case legible.
    dropped = dict(shapes[3][1])
    check(
        "all-bad-grads shape is self-explanatory in TB "
        "(n_updates=0, n_micro_batches>0, n_nonfinite_grad_steps>0)",
        dropped.get("n_updates", 0) == 0
        and dropped.get("n_micro_batches", 0) > 0
        and dropped.get("n_nonfinite_grad_steps", 0) > 0,
        str(dropped),
    )


def test_config_validation():
    print("\n[Config] gradient_accumulation_steps validation")

    check("default is 1 (no accumulation)",
          GRPOConfig().gradient_accumulation_steps == 1)
    for good in (1, 2, 3, 16):
        try:
            GRPOConfig(gradient_accumulation_steps=good)
            ok = True
        except ValueError:
            ok = False
        check(f"k={good} accepted", ok)
    for bad in (0, -1, -7):
        try:
            GRPOConfig(gradient_accumulation_steps=bad)
            raised = False
        except ValueError:
            raised = True
        check(f"k={bad} rejected with ValueError", raised)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_harness_grad_is_param_independent()
    test_k1_is_per_minibatch_update()
    test_k2_halves_optimizer_steps()
    test_loss_scaled_by_one_over_k()
    test_partial_window_flushed_at_epoch_end()
    test_nonfinite_microbatch_does_not_poison_window()
    test_all_nonfinite_takes_no_step()
    test_nonfinite_gradient_drops_step_and_protects_weights()
    test_all_windows_nonfinite_grad_leaves_model_untouched()
    test_no_step_when_nothing_prepared()
    test_per_microbatch_metrics_invariant_to_k()
    test_grad_norm_tracks_accumulated_gradient()
    test_n_updates_counts_real_steps()
    test_balanced_sampler_epoch_boundary()
    test_multi_group_stratified_with_accumulation()
    test_paws_mass_pools_per_trained_microbatch()
    test_paws_cold_start_uses_target_ratio_not_one()
    test_jitter_branch_metrics_invariant_to_k()
    test_base_model_kl_metric_divisor()
    test_clipped_rows_metrics_invariant_to_k()
    test_result_shapes_are_loggable()
    test_config_validation()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\033[32mAll tests passed.\033[0m")
