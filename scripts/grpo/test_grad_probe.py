"""Tests for the gradient-decomposition probe (`GRPOConfig.grad_probe_every`).

WHAT IS UNDER TEST. On a positive-advantage row the loss minimises
`MSE_theta(eps')` with `eps' = sqrt(1-lam^2) eps + lam xi`, and in expectation
over xi that ONE term is two gradients welded together:

    d MSE_theta(eps')/d theta = d MSE_theta(eps)/d theta + lam^2 dP/dtheta
                              = g_R                      + lam^2 g_P

The probe takes `g_jit` off the graph the training forward already built, runs
one extra clean-eps forward for `g_R`, and reports
`R = ||g_head|| / ||g_R||` and `cos(g_R, g_head)` where `g_head = g_jit - g_R`.

Like `test_grad_accum.py` (whose conventions this file follows) these tests drive
the **real** `GRPOTrainer._grpo_update` / `_grpo_update_inner` and the real
`_grad_probe_capture_jittered` / `_grad_probe_finish` /
`aggregate_grad_probes` / `select_grad_probe_rows` on CPU. The substitutions are:

  1. `_prepare_batch`  -> tiny CPU tensors instead of an Eagle re-encode.
  2. `compute_fm_log_prob` -> `_analytic_fm`, below.
  3. the model -> `_TinyModel`, one 2-element trainable parameter.
  4. the episode buffer -> a `_build_chunks()` stub over hand-built chunks.
  5. the optimizer -> plain SGD, so a trajectory is exactly reproducible.
  6. the trainer is built with `GRPOTrainer.__new__` to skip `setup()`.

WHY THIS FILE NEEDS ITS OWN STAND-IN rather than reusing
`test_grad_accum.run_update`. That harness's fake is (a) jitter-INSENSITIVE (it
drops `noise_for_input`) and (b) returns a **detached** `per_tau`. Both are fine
for what it pins and fatal here: with (a) `g_head` would be identically zero, and
with (b) `autograd.grad` cannot differentiate the jittered leg at all. So
`_analytic_fm` below is tau-sensitive, `noise_for_input`-sensitive, and
differentiable — and, being a closed form, lets the decomposition identity be
checked against a **hand-derived** `g_head` rather than against a second autograd
run of the same expression. `test_grad_accum.run_update` IS still reused, for
exactly the case its detached `per_tau` makes valuable: proving a probe that
raises costs the metric and not the iteration.

Run with the project venv (needs torch; CPU is fine):
    .venv/bin/python scripts/grpo/test_grad_probe.py
"""

import contextlib
import dataclasses
import io
import math
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_grpo  # noqa: E402  (path set up above)
import test_clip_floor as tcf  # noqa: E402
import test_grad_accum as tga  # noqa: E402
from grpo_config import GRPOConfig  # noqa: E402
from train_grpo import (  # noqa: E402
    GRPOTrainer,
    aggregate_grad_probes,
    flatten_param_grads,
    select_grad_probe_rows,
)

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


# ── The analytic stand-in ────────────────────────────────────────────────────
#
# Mirrors the real `compute_fm_log_prob`'s STRUCTURE with a velocity field that
# is LINEAR in the DiT input, so the Jacobian is a known constant and the
# per-sample gap has a closed form:
#
#   u_ki   = mean over (H, D) of the DiT INPUT noise for row i at tau k
#            (== eps_i when noise_for_input is None, else eps'_ki)
#   resid  = c_i + w0 (1 - tau_ki) u_ki + w1          c_i = mean(actions_i)
#   MSE_ki = resid^2
#   log_prob_i = -mean_k MSE_ki
#
# d resid / d u = w0 (1 - tau), i.e. the "Jacobian" is exactly `w0 (1-tau)` and
# the `(1-tau)^2 lam^2 ||J||^2` structure of the real regulariser appears
# verbatim in the gap. `velocity_target` is folded into `c_i`, matching the real
# function's property that the target stays at the ORIGINAL eps on both legs.
#
# Every call is RECORDED so a test can reconstruct the hand-derived gradient from
# the exact tensors production used — including the bf16-quantized `timesteps`
# and the xi draw, neither of which the test can predict.

H_DIM, D_DIM = 3, 4          # action horizon x dim; > 1 so the dim mean matters
TAUS = [0.0, 0.25, 0.5, 0.75]


def _analytic_fm_factory(model, calls, *, clean_grad_nonfinite=False,
                         detach=False):
    """Build the stand-in plus its call log.

    Args:
        clean_grad_nonfinite: make the CLEAN leg's GRADIENT non-finite, to
            exercise the non-finite-norm skip. Note it must be the gradient, not
            the value: adding `inf` to the returned log-prob leaves the gradient
            finite (autograd passes `d(x + const)/dx` straight through), so a
            value-only injection does not reach `r_norm` at all.
        detach: return a DETACHED `per_tau`, reproducing
            `test_grad_accum`'s fake, so `autograd.grad` on the jittered leg
            raises — the phase-1 failure path.
    """

    def _analytic_fm(**kw):
        actions = kw["actions"]                       # [B, H, D]
        eps = kw["noise"]                             # [B, H, D]
        ts = kw["timesteps"]                          # [K, B] (bf16 in prod)
        n = int(kw["n_samples"])
        nfi = kw.get("noise_for_input")               # [K, B, H, D] or None
        w = model.w
        B = actions.shape[0]
        c = actions.reshape(B, -1).mean(dim=1).to(torch.float32)      # [B]
        per_tau = []
        for k in range(n):
            x = eps if nfi is None else nfi[k]
            u = x.reshape(B, -1).mean(dim=1).to(torch.float32)        # [B]
            t = ts[k].to(torch.float32)                               # [B]
            resid = c + w[0] * (1.0 - t) * u + w[1]
            per_tau.append(-(resid ** 2))
        pt = torch.stack(per_tau, dim=0)                              # [n, B]
        lp = pt.mean(dim=0)
        # LEG CLASSIFICATION. Three DIFFERENT callers reach this stand-in, and
        # they must not be confused:
        #   "train"      the main loss forward. The ONLY one that passes
        #                `smooth_instrument`, which is what identifies it.
        #   "probe_clean" the gradient probe's clean-eps forward: no
        #                `noise_for_input`, no `return_per_tau`.
        #   "jitter_diag" `_jitter_gap_diagnostics`' two no_grad forwards, which
        #                fire once per iteration when jitter is active. Its CLEAN
        #                leg also has `noise_for_input=None`, so a classifier
        #                keyed only on that would mistake it for the training
        #                forward and every hand-derived expectation would be
        #                computed from an un-jittered tensor.
        if "smooth_instrument" in kw:
            leg = "train"
        elif nfi is None and not kw.get("return_per_tau"):
            leg = "probe_clean"
        else:
            leg = "jitter_diag"
        if clean_grad_nonfinite and leg == "probe_clean":
            # Identity forward, inf GRADIENT — reuses test_grad_accum's
            # `_ScaleGrad`, which exists for exactly this shape of injection.
            lp = tga._ScaleGrad.apply(lp, float("inf"))
        calls.append({
            "leg": leg,
            "n_samples": n,
            "timesteps": ts.detach().clone(),
            "noise": eps.detach().clone(),
            "actions": actions.detach().clone(),
            "noise_for_input": None if nfi is None else nfi.detach().clone(),
            "return_per_tau": bool(kw.get("return_per_tau")),
            "w": w.detach().clone(),
            "B": B,
        })
        if kw.get("return_per_tau"):
            extras = [pt.detach() if detach else pt]
        else:
            extras = []
        if kw.get("smooth_dims") is not None:
            # Honour the real return contract's smooth slot: a (constrained,
            # endpoint) PAIR of [B, 2] = (R, M) moments. Value-pinned via the
            # `x - x.detach()` trick so R/M — and hence pooled HF = R/(6M) =
            # 0.1 — are exact, while the gradient is the real one. Present so the
            # FOUR-WAY return unpack (smooth x probe) is exercised.
            pin = lp.unsqueeze(1) - lp.detach().unsqueeze(1)     # [B, 1], == 0
            mom = torch.cat(
                (torch.full_like(pin, 1.2) + pin, torch.full_like(pin, 2.0)),
                dim=1,
            )
            ep = torch.cat(
                (torch.full_like(pin, 0.9), torch.full_like(pin, 2.0)), dim=1
            ).detach()
            extras.append((mom, ep))
        if extras:
            return (lp, *extras)
        return lp

    return _analytic_fm


@dataclass
class _Chunk:
    """Minimal ActionChunk stand-in — only the fields the update loop reads."""
    advantage: float
    feat: float
    group_id: int = 0
    is_anchor: bool = False
    ref_log_prob: Optional[float] = 0.0
    base_log_prob: Optional[float] = 0.0
    tau_samples: Optional[np.ndarray] = None
    raw_action: Optional[np.ndarray] = None
    noise: Optional[np.ndarray] = None


def _make_chunks(advantages, *, n_groups=1, taus=TAUS):
    """One chunk per advantage, each with a DISTINCT action / eps / tau sample.

    Distinct taus per chunk matter: the stand-in weights the input noise by
    `(1 - tau)`, so a shared tau would hide any tau-subset mistake.

    MAGNITUDES ARE DELIBERATELY SMALL (|c| <= 0.6, |eps mean| <= 0.5, so the
    stand-in's MSE stays under ~0.5). The KL term in the real loss is
    `exp(-log_ratio) - ... = exp(MSE_theta - MSE_ref) - ...`; with
    `ref_log_prob = 0` a stand-in MSE of ~8 already puts `exp` near fp32
    overflow after one un-clipped SGD step, and the whole run then dies in the
    non-finite-loss guard with only two trained micro-batches. `max_grad_norm`
    is also finite in `run_probe` for the same reason.
    """
    chunks = []
    for i, adv in enumerate(advantages):
        feat = 0.05 * (i + 1) * (1.0 if i % 2 == 0 else -1.0)
        # Deterministic, distinct, small eps. Not torch.randn: the global RNG
        # stream is itself under test.
        noise = np.array(
            [[0.03 * (i + 1) + 0.02 * (h * D_DIM + d)
              for d in range(D_DIM)] for h in range(H_DIM)],
            dtype=np.float32,
        )
        chunks.append(_Chunk(
            advantage=float(adv),
            feat=feat,
            group_id=i % n_groups,
            tau_samples=np.array(
                [t + 0.013 * i for t in taus], dtype=np.float32
            ),
            raw_action=np.full((H_DIM, D_DIM), feat, dtype=np.float32),
            noise=noise,
        ))
    return chunks


class _ActionHeadStub(nn.Module):
    """`self.model.action_head.model.eval()` needs to resolve to a Module."""

    def __init__(self):
        super().__init__()
        self.model = nn.Identity()


class _TinyModel(nn.Module):
    """One 2-element trainable parameter — stands in for the LoRA params."""

    def __init__(self, w0=(0.4, -0.1)):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.action_head = _ActionHeadStub()


@dataclass
class _Run:
    result: dict
    calls: list
    stdout: str
    config: GRPOConfig
    w0: torch.Tensor
    w_final: torch.Tensor
    grad_final: torch.Tensor
    chunks: list

    @property
    def probe(self) -> dict:
        return self.result.get("_grad_probe") or {}

    @property
    def train_calls(self) -> list:
        return [c for c in self.calls if c["leg"] == "train"]

    @property
    def clean_calls(self) -> list:
        return [c for c in self.calls if c["leg"] == "probe_clean"]

    def row_to_chunk(self, train_call, row: int) -> int:
        """Which CHUNK a micro-batch row came from.

        The samplers shuffle, so a row index is NOT a chunk index. Each chunk's
        action tensor is filled with its unique `feat`, which makes the mapping
        recoverable from the recorded tensors alone.
        """
        f = float(train_call["actions"][row].reshape(-1)[0])
        for j, c in enumerate(self.chunks):
            # 1e-6, not 1e-9: `feat` is a Python float and the recorded tensor is
            # float32, so e.g. -0.1 comes back as -0.10000000149011612.
            if abs(c.feat - f) < 1e-6:
                return j
        raise AssertionError(f"row {row} feat {f} matches no chunk")

    def probed_rows(self) -> list:
        """Micro-batch row indices the FIRST probe selected, ascending."""
        tc, cc = self.train_calls[0], self.clean_calls[0]
        return sorted(
            int(((tc["noise"] - e).abs().sum(dim=(1, 2))).argmin())
            for e in cc["noise"]
        )


def run_probe(
    *,
    advantages=(2.0, 1.4, 1.1, 0.8, -1.0, -1.6),
    n_groups=1,
    mb_size=6,
    epochs=1,
    lr=0.02,
    w0=(0.4, -0.1),
    taus=TAUS,
    clean_grad_nonfinite=False,
    detach_per_tau=False,
    smooth=False,
    fm_override=None,
    config_overrides=None,
) -> _Run:
    """Drive the real `_grpo_update()` once with the probe wired up.

    Defaults give ONE micro-batch of six rows (stratified sampler, one group,
    `mb_size == len(advantages)`, one epoch), which is what lets a test derive
    the expected gradient by hand at the initial `w`: there is no intervening
    optimizer step.
    """
    cfg_kwargs = dict(
        device="cpu",
        mini_batch_size=mb_size,
        update_epochs=epochs,
        gradient_accumulation_steps=1,
        balanced_minibatch_training=False,
        dynamic_epoch_training=False,
        per_iteration_advantage_norm=False,
        positive_advantage_weight_scaling=False,
        kl_coef_last_iter=0.2,
        kl_coef_base_model=0.0,
        # Production-like: a real positive-side lambda, jitter_neg pinned at 0 so
        # the FREE erosion measurement is clean.
        jitter_pos=0.25,
        jitter_neg=0.0,
        jitter_paired=False,
        tau_centers=list(taus),
        grad_probe_every=1,
        # FINITE, unlike test_grad_accum's 1e9: the stand-in's KL gradient grows
        # like exp(MSE), so an unclipped step can send `w` somewhere that
        # overflows fp32 `exp` and kills the rest of the epoch in the
        # non-finite-loss guard — leaving too few trained micro-batches for the
        # cadence and trend assertions to mean anything.
        max_grad_norm=0.5,
        learning_rate=lr,
        seed=11,
    )
    cfg_kwargs.update(config_overrides or {})
    cfg = GRPOConfig(**cfg_kwargs)

    chunks = _make_chunks(advantages, n_groups=n_groups, taus=taus)
    calls: list = []
    model = _TinyModel(w0)

    trainer = GRPOTrainer.__new__(GRPOTrainer)      # skip setup(): no GPU here
    trainer.config = cfg
    trainer.device = torch.device("cpu")
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0
    )
    trainer.buffer = types.SimpleNamespace(_build_chunks=lambda: list(chunks))
    trainer.iteration = 1
    trainer._model_lock = threading.RLock()

    def _prepare_batch_stub(self, batch):
        valid = [c for (c, _m) in batch]
        modes = [m for (_c, m) in batch]
        B = len(valid)
        actions = torch.from_numpy(
            np.stack([c.raw_action for c in valid])
        ).to(torch.float32)                                   # [B, H, D]
        noise = torch.from_numpy(
            np.stack([c.noise for c in valid])
        ).to(torch.float32)
        return {
            "actions": actions,
            "action_masks": torch.ones_like(actions),
            "initial_noise": noise,
            "advantages": torch.tensor(
                [c.advantage for c in valid], dtype=torch.float32
            ),
            "backbone_output": {"backbone_features": torch.zeros(B, 1, 1)},
            "state_features": torch.zeros(B, 1, 1),
            "embodiment_id": torch.zeros(B, dtype=torch.long),
            "modes": modes,
        }, valid

    trainer._prepare_batch = types.MethodType(_prepare_batch_stub, trainer)

    fake = fm_override or _analytic_fm_factory(
        model, calls,
        clean_grad_nonfinite=clean_grad_nonfinite,
        detach=detach_per_tau,
    )
    real_fm = train_grpo.compute_fm_log_prob
    train_grpo.compute_fm_log_prob = fake
    # The roughness constraint's attributes live on the CLASS (they are the
    # OFF-state defaults a `__new__`-built trainer relies on), so `smooth_coef`
    # alone does not switch the feature on — the same trap
    # `test_smoothness._on_path_run` documents. Restored in the finally.
    _smooth_saved = {}
    if smooth:
        _smooth_saved = {
            a: getattr(GRPOTrainer, a)
            for a in ("smooth_active", "_smooth_dims", "_smooth_horizon",
                      "_smooth_eef_pos_dims", "_smooth_hf_ref",
                      "_smooth_calib_sum", "_smooth_calib_n",
                      "_smooth_calib_rows", "_smooth_n_exec")
        }
        GRPOTrainer.smooth_active = True
        GRPOTrainer._smooth_dims = torch.tensor([0, 1, 2])
        GRPOTrainer._smooth_horizon = H_DIM
        GRPOTrainer._smooth_eef_pos_dims = torch.tensor([0, 1, 2])
        GRPOTrainer._smooth_hf_ref = torch.tensor(0.0)
        GRPOTrainer._smooth_calib_sum = None
        GRPOTrainer._smooth_calib_n = 0
        GRPOTrainer._smooth_calib_rows = 0
        GRPOTrainer._smooth_n_exec = 2
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = trainer._grpo_update()
    finally:
        train_grpo.compute_fm_log_prob = real_fm
        for a, v in _smooth_saved.items():
            setattr(GRPOTrainer, a, v)

    return _Run(
        result=result,
        calls=calls,
        stdout=buf.getvalue(),
        config=cfg,
        w0=torch.tensor(w0, dtype=torch.float32),
        w_final=model.w.detach().clone(),
        grad_final=(
            model.w.grad.detach().clone() if model.w.grad is not None
            else torch.zeros(2)
        ),
        chunks=chunks,
    )


def _hand_g_head_rows(train_call, rows, *, tau_sub) -> torch.Tensor:
    """`g_head` over an EXPLICIT row list, derived by hand — no autograd.

    For the stand-in's `resid(u) = c + w0 (1-t) u + w1` and `MSE(u) = resid(u)^2`,

        d MSE / d w0 = 2 resid(u) (1-t) u
        d MSE / d w1 = 2 resid(u)

    so with `u' = u + d` (and therefore `resid(u') = r + w0 (1-t) d`, `r` being
    the CLEAN residual) the per-(row, tau) gap gradient is

        d gap / d w0 = 2 (1-t) [ resid(u') u' - r u ]
                     = 2 (1-t) d [ r + w0 (1-t) u' ]
        d gap / d w1 = 2 [ resid(u') - r ]
                     = 2 w0 (1-t) d

    Note the `w0` factor on the second line: `resid` depends on `w1` with slope 1
    on BOTH legs, so the w1 component of `g_head` is the DIFFERENCE of two
    residuals, not `2 (1-t) d`. Dropping it understates ||g_head|| by ~2.3x at
    w0 = 0.4 — which is exactly the kind of error this whole file exists to catch,
    and it is why the expectation is derived here rather than by re-running
    autograd on the same expression (a re-run would have agreed with the mistake).

    Then MEAN over the tau subset and MEAN over the rows, matching the two-step
    reduction both legs of the probe perform.
    """
    w = train_call["w"]
    w0, w1 = float(w[0]), float(w[1])
    eps_all = train_call["noise"]                              # [B, H, D]
    ts = train_call["timesteps"][:tau_sub].to(torch.float32)    # [tau_sub, B]
    nfi = train_call["noise_for_input"]
    acc = torch.zeros(2, dtype=torch.float32)
    for i in rows:
        u0 = float(eps_all[i].mean())
        c = float(train_call["actions"][i].reshape(-1).mean())
        row = torch.zeros(2, dtype=torch.float32)
        for k in range(tau_sub):
            t = float(ts[k, i])
            up = u0 if nfi is None else float(nfi[k, i].mean())
            d = up - u0
            r = c + w0 * (1.0 - t) * u0 + w1
            row[0] += 2.0 * (1.0 - t) * d * (r + w0 * (1.0 - t) * up)
            row[1] += 2.0 * w0 * (1.0 - t) * d
        acc += row / tau_sub
    return acc / len(rows)


def _hand_g_head(train_call, clean_call, *, tau_sub) -> torch.Tensor:
    """`_hand_g_head_rows` with the rows recovered from the clean leg's eps.

    Valid whenever each chunk appears in the micro-batch AT MOST ONCE. Under
    `jitter_paired=True` a chunk's two copies share an eps, so the mapping is
    ambiguous there and the caller must pass rows explicitly instead.
    """
    eps_all, eps_sub = train_call["noise"], clean_call["noise"]
    rows = [
        int(((eps_all - e).abs().sum(dim=(1, 2))).argmin())
        for e in eps_sub
    ]
    return _hand_g_head_rows(train_call, rows, tau_sub=tau_sub)


# ═════════════════════════════════════════════════════════════════════════════
# 1 / 2 / 3 — the off-switch invariant, RNG identity, and `.grad` isolation
# ═════════════════════════════════════════════════════════════════════════════

_TRAINING_KEYS_IGNORED = {"_grad_probe"}


def _same_training_stats(a: dict, b: dict) -> str:
    """"" when every shared pre-existing stat matches exactly, else a diff."""
    diffs = []
    for k in sorted((set(a) | set(b)) - _TRAINING_KEYS_IGNORED):
        if k not in a or k not in b:
            diffs.append(f"{k}: present in only one")
            continue
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float):
            if math.isnan(va) and math.isnan(vb):
                continue
        if va != vb:
            diffs.append(f"{k}: {va!r} vs {vb!r}")
    return "; ".join(diffs[:5])


def test_probe_off_emits_nothing():
    print("\n[Off] grad_probe_every=0 takes no probe branch at all")
    grad_calls = {"n": 0, "retain": 0}
    real_grad = torch.autograd.grad

    def _spy_grad(*a, **k):
        grad_calls["n"] += 1
        if k.get("retain_graph"):
            grad_calls["retain"] += 1
        return real_grad(*a, **k)

    torch.autograd.grad = _spy_grad
    try:
        off = run_probe(config_overrides=dict(grad_probe_every=0))
    finally:
        torch.autograd.grad = real_grad

    check("no `_grad_probe` key in the update stats",
          "_grad_probe" not in off.result,
          f"keys: {sorted(off.result)}")
    # Scoped to the TRAINING leg: `_jitter_gap_diagnostics` legitimately uses
    # `return_per_tau=True` on its own two no_grad forwards, and always has. What
    # must stay False with the probe off is the LOSS forward, because that is the
    # graph the probe would otherwise enlarge.
    check("`return_per_tau` is never requested on the TRAINING forward",
          all(not c["return_per_tau"] for c in off.train_calls),
          f"{[c['return_per_tau'] for c in off.train_calls]}")
    check("no extra clean-eps forward runs",
          off.clean_calls == [], f"{len(off.clean_calls)} clean call(s)")
    check("torch.autograd.grad is never called",
          grad_calls["n"] == 0, f"{grad_calls['n']} call(s)")
    check("... and therefore no retain_graph=True is taken",
          grad_calls["retain"] == 0, f"{grad_calls['retain']}")
    check("no banner line is printed",
          "Gradient-decomposition probe" not in off.stdout)
    # Non-vacuity: the same spy MUST fire with the probe on.
    torch.autograd.grad = _spy_grad
    try:
        on = run_probe()
    finally:
        torch.autograd.grad = real_grad
    check("... and the same spy DOES fire with the probe on",
          grad_calls["n"] >= 2 and grad_calls["retain"] >= 1,
          f"n={grad_calls['n']} retain={grad_calls['retain']}")
    check("... and the probe emits its family",
          bool(on.probe), f"{on.probe}")
    check("the CUDA-only VRAM key is absent on a CPU run",
          "_vram_peak_delta_gb" not in on.probe, f"{sorted(on.probe)}")


def test_probe_on_does_not_change_training():
    print("\n[Isolation] the probe changes NOTHING about the training step")
    torch.manual_seed(4242)
    _ = torch.randn(1)                       # anchor the stream
    off = run_probe(config_overrides=dict(grad_probe_every=0))
    rng_off = torch.randn(4).tolist()

    torch.manual_seed(4242)
    _ = torch.randn(1)
    on = run_probe()
    rng_on = torch.randn(4).tolist()

    check("final weights are BIT-identical",
          torch.equal(off.w_final, on.w_final),
          f"{off.w_final.tolist()} vs {on.w_final.tolist()}")
    # THE constraint that makes `autograd.grad` mandatory: `.backward()` would
    # have accumulated the probe's two extra gradients into this buffer, silently
    # changing the optimizer step.
    check("`p.grad` after the update is BIT-identical (autograd.grad does NOT "
          "accumulate)",
          torch.equal(off.grad_final, on.grad_final),
          f"{off.grad_final.tolist()} vs {on.grad_final.tolist()}")
    check("every pre-existing stat is unchanged",
          _same_training_stats(off.result, on.result) == "",
          _same_training_stats(off.result, on.result))
    check("the global RNG stream is unchanged (the clean forward consumes none)",
          rng_off == rng_on, f"{rng_off} vs {rng_on}")
    # Non-vacuity for the RNG check: the probe path must be distinguishable.
    check("... and that comparison is not vacuous (the arms really differ)",
          bool(on.probe) and not off.result.get("_grad_probe"),
          "one arm did not run the probe")
    # Non-vacuity for the grad check: a nonzero grad is being compared.
    check("... and the compared `p.grad` is non-zero",
          float(on.grad_final.abs().sum()) > 0.0,
          f"{on.grad_final.tolist()}")


def test_rng_leak_would_be_detected():
    print("\n[Isolation] a positive control for the RNG-stream assertion")

    def _stream(leak: bool):
        """Run one update, optionally leaking ONE draw inside the clean leg.

        The leak is injected exactly where a careless implementation would put
        it — letting `compute_fm_log_prob` sample its own eps because `noise` was
        not threaded through — so this proves the stream comparison in
        `test_probe_on_does_not_change_training` is able to fail.
        """
        m = _TinyModel()
        calls: list = []
        inner = _analytic_fm_factory(m, calls)

        def _fm(**kw):
            if (leak and kw.get("noise_for_input") is None
                    and not kw.get("return_per_tau")):
                torch.randn(1)
            return inner(**kw)

        torch.manual_seed(99)
        _ = torch.randn(1)
        _run_with_model(m, _fm)
        return torch.randn(4).tolist()

    clean = _stream(leak=False)
    leaked = _stream(leak=True)
    check("an injected draw in the clean leg IS detected",
          clean != leaked, f"clean {clean} vs leaked {leaked}")


def _run_with_model(model, fm):
    """Drive one `_grpo_update()` against a caller-supplied model and fake.

    Used only by the RNG positive control, which has to bind its wrapper to the
    same model instance the stand-in closes over.
    """
    cfg = GRPOConfig(
        device="cpu", mini_batch_size=6, update_epochs=1,
        balanced_minibatch_training=False, dynamic_epoch_training=False,
        per_iteration_advantage_norm=False,
        positive_advantage_weight_scaling=False,
        kl_coef_base_model=0.0, jitter_pos=0.25, jitter_neg=0.0,
        jitter_paired=False, tau_centers=list(TAUS), grad_probe_every=1,
        max_grad_norm=0.5, learning_rate=0.02, seed=11,
    )
    chunks = _make_chunks((2.0, 1.4, 1.1, 0.8, -1.0, -1.6))
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.config = cfg
    t.device = torch.device("cpu")
    t.model = model
    t.optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    t.buffer = types.SimpleNamespace(_build_chunks=lambda: list(chunks))
    t.iteration = 1
    t._model_lock = threading.RLock()

    def _prep(self, batch):
        valid = [c for (c, _m) in batch]
        B = len(valid)
        actions = torch.from_numpy(
            np.stack([c.raw_action for c in valid])).to(torch.float32)
        noise = torch.from_numpy(
            np.stack([c.noise for c in valid])).to(torch.float32)
        return {
            "actions": actions, "action_masks": torch.ones_like(actions),
            "initial_noise": noise,
            "advantages": torch.tensor(
                [c.advantage for c in valid], dtype=torch.float32),
            "backbone_output": {"backbone_features": torch.zeros(B, 1, 1)},
            "state_features": torch.zeros(B, 1, 1),
            "embodiment_id": torch.zeros(B, dtype=torch.long),
            "modes": [m for (_c, m) in batch],
        }, valid

    t._prepare_batch = types.MethodType(_prep, t)
    real = train_grpo.compute_fm_log_prob
    train_grpo.compute_fm_log_prob = fm
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return t._grpo_update()
    finally:
        train_grpo.compute_fm_log_prob = real


# ═════════════════════════════════════════════════════════════════════════════
# 4 — the decomposition identity, against a hand-derived closed form
# ═════════════════════════════════════════════════════════════════════════════

def test_decomposition_identity_against_closed_form():
    print("\n[Identity] g_jit - g_R equals the hand-derived lambda^2 g_P")
    r = run_probe()
    check("exactly one probe ran", r.probe.get("n_probes") == 1,
          f"{r.probe}")
    check("exactly one clean-eps forward ran",
          len(r.clean_calls) == 1, f"{len(r.clean_calls)}")

    tc, cc = r.train_calls[0], r.clean_calls[0]
    expected = _hand_g_head(tc, cc, tau_sub=len(TAUS))
    got = r.probe["g_headroom_norm"]
    check("||g_head|| matches the closed form",
          math.isclose(got, float(expected.norm()), rel_tol=1e-4),
          f"probe {got:.8g} vs hand {float(expected.norm()):.8g}")
    # Not just the norm: the DIRECTION too, via the reported cosine against g_R.
    # g_R is recoverable from the clean leg by autograd on a fresh copy.
    m = _TinyModel(tuple(tc["w"].tolist()))
    n_probe = cc["B"]
    c_v = cc["actions"].reshape(n_probe, -1).mean(dim=1)
    u_v = cc["noise"].reshape(n_probe, -1).mean(dim=1)
    ts = cc["timesteps"].to(torch.float32)
    mse = torch.zeros(n_probe)
    for k in range(ts.shape[0]):
        mse = mse + (c_v + m.w[0] * (1.0 - ts[k]) * u_v + m.w[1]) ** 2
    g_r = torch.autograd.grad((mse / ts.shape[0]).mean(), [m.w])[0]
    check("||g_R|| matches an independent clean-leg autograd",
          math.isclose(r.probe["g_reinforce_norm"], float(g_r.norm()),
                       rel_tol=1e-4),
          f"probe {r.probe['g_reinforce_norm']:.8g} vs {float(g_r.norm()):.8g}")
    cos_hand = float(
        expected.dot(g_r) / (expected.norm() * g_r.norm())
    )
    check("cos(g_R, g_head) matches the closed form",
          math.isclose(r.probe["cos_reinforce_headroom"], cos_hand,
                       rel_tol=1e-4, abs_tol=1e-6),
          f"probe {r.probe['cos_reinforce_headroom']:.6g} vs {cos_hand:.6g}")
    check("R == ||g_head|| / ||g_R||",
          math.isclose(r.probe["R_mean"],
                       r.probe["g_headroom_norm"] / r.probe["g_reinforce_norm"],
                       rel_tol=1e-9),
          f"{r.probe['R_mean']}")
    # Non-vacuity: the closed form must not be trivially small or trivially
    # equal to ||g_jit||.
    check("... and the measurement is non-trivial (g_head, g_R and g_jit all "
          "differ and are non-zero)",
          r.probe["g_headroom_norm"] > 1e-8
          and r.probe["g_reinforce_norm"] > 1e-8
          and abs(r.probe["g_jit_norm"] - r.probe["g_reinforce_norm"]) > 1e-8,
          f"{r.probe}")


def test_R_vanishes_as_lambda_goes_to_zero():
    print("\n[Identity] R -> 0 as lambda -> 0")
    # Several probes per arm, and the SAME seed for each arm, so the xi draw is
    # held fixed and only lambda moves. A SINGLE probe is not enough: g_head's
    # cross term `2 r w0 (1-tau) delta` has zero mean but a large per-sample
    # realization, so one draw's ||g_head|| is genuinely non-monotone in lambda
    # (measured: 0.0104 at 0.15 vs 0.0067 at 0.40 on a one-probe arm). The
    # EXPECTATION is monotone, which is what the mean over probes estimates.
    advs = (2.0, 1.7, 1.4, 1.2, 1.0, 0.8,
            -1.0, -1.2, -1.4, -1.6, -1.8, -2.0)
    got = {}
    for lam in (0.0, 0.05, 0.15, 0.40):
        torch.manual_seed(5150)
        r = run_probe(advantages=advs, mb_size=4, epochs=4,
                      config_overrides=dict(jitter_pos=lam))
        got[lam] = r.probe["R_mean"]
    # NOT `== 0.0`: with `n_probe_rows != B` the clean leg reduces a
    # differently-shaped tensor, so `g_jit - g_R` carries ~1 fp32 ULP even though
    # the two are mathematically identical at lambda == 0. The floor is ~1e-7
    # RELATIVE here and higher in production (bf16 activations); it is the
    # instrument's resolution limit, not a bug.
    check("lambda == 0 gives R at the fp noise floor (eps' IS eps)",
          got[0.0] < 1e-6, f"{got[0.0]!r}")
    ordered = [got[l] for l in (0.0, 0.05, 0.15, 0.40)]
    check("R increases monotonically in lambda",
          all(a < b for a, b in zip(ordered, ordered[1:])),
          f"{ordered}")
    check("... and lambda=0.40 is >10x lambda=0.05, i.e. materially non-zero",
          got[0.40] > 10 * max(got[0.05], 1e-12), f"{ordered}")


# ═════════════════════════════════════════════════════════════════════════════
# 5 / 6 — same tau, same rows, same eps; and the tau subset
# ═════════════════════════════════════════════════════════════════════════════

def test_both_legs_share_tau_eps_and_rows():
    print("\n[Same-tau] the clean leg reuses the training tensors verbatim")
    r = run_probe()
    tc, cc = r.train_calls[0], r.clean_calls[0]
    idx = [
        int(((tc["noise"] - e).abs().sum(dim=(1, 2))).argmin())
        for e in cc["noise"]
    ]
    check("the clean leg's eps is a literal gather of the training eps",
          torch.equal(cc["noise"], tc["noise"][idx]),
          "eps differs between the two legs")
    check("the clean leg's timesteps are `timesteps[:tau_sub][:, idx]` exactly",
          torch.equal(cc["timesteps"], tc["timesteps"][:len(TAUS)][:, idx]),
          f"{cc['timesteps']} vs {tc['timesteps'][:, idx]}")
    check("the clean leg's actions are the same rows",
          torch.equal(cc["actions"], tc["actions"][idx]))
    check("the clean leg passes noise_for_input=None (the ONLY difference)",
          cc["noise_for_input"] is None)
    check("the clean leg's n_samples equals the tau subset size",
          cc["n_samples"] == len(TAUS), f"{cc['n_samples']}")
    check("rows are ascending (canonical gather order)",
          idx == sorted(idx), f"{idx}")


def test_mismatched_tau_or_rows_is_detected():
    print("\n[Same-tau] mismatching tau or rows produces a DIFFERENT reading")

    def _arm(mutate):
        """One update; `mutate` corrupts the CLEAN leg's inputs only.

        `mutate="tau"`  -> different tau VALUES than the jittered leg used.
        `mutate="rows"` -> a different ROW SUBSET than the probe selected.

        Note a tau PERMUTATION within the subset is deliberately NOT one of the
        mutants: on the clean leg the DiT input is eps for every tau, so the
        per-row MSE is a MEAN over the subset and any permutation of it is
        bit-identical. That is not a gap in the test — a permutation is genuinely
        harmless. What breaks the identity is a different tau SET or a different
        row SET, which is what these two mutants do.
        """
        m = _TinyModel()
        calls: list = []
        inner = _analytic_fm_factory(m, calls)

        def _fm(**kw):
            is_clean = (
                "smooth_instrument" not in kw
                and kw.get("noise_for_input") is None
                and not kw.get("return_per_tau")
            )
            if mutate and is_clean:
                tc = [c for c in calls if c["leg"] == "train"][-1]
                kw = dict(kw)
                if mutate == "tau":
                    kw["timesteps"] = kw["timesteps"] + 0.25
                else:
                    n = kw["actions"].shape[0]
                    kw["actions"] = tc["actions"][-n:]
                    kw["noise"] = tc["noise"][-n:]
                    kw["timesteps"] = tc["timesteps"][
                        :kw["timesteps"].shape[0], -n:
                    ].contiguous()
            return inner(**kw)

        torch.manual_seed(31337)
        res = _run_with_model(m, _fm)
        return (res.get("_grad_probe") or {}).get("R_mean")

    good = _arm(None)
    check("the correctly-paired arm produced a reading",
          good is not None, "no probe ran")
    for mutant in ("tau", "rows"):
        bad = _arm(mutant)
        check(f"the mismatched-{mutant} arm produced a reading too",
              bad is not None, "no probe ran")
        check(f"... and it DIFFERS from the correct one, so a {mutant} mismatch "
              f"cannot go unnoticed",
              bad is not None and good is not None
              and not math.isclose(bad, good, rel_tol=1e-3),
              f"good {good} vs mismatched {bad}")


def test_tau_subset_applies_to_both_legs():
    print("\n[Tau subset] grad_probe_tau_subset slices BOTH legs identically")
    r = run_probe(config_overrides=dict(grad_probe_tau_subset=2))
    tc, cc = r.train_calls[0], r.clean_calls[0]
    check("the TRAINING forward still uses every tau (the loss is unchanged)",
          tc["n_samples"] == len(TAUS), f"{tc['n_samples']}")
    check("the clean leg uses only the first 2 taus",
          cc["n_samples"] == 2 and cc["timesteps"].shape[0] == 2,
          f"n_samples={cc['n_samples']} shape={tuple(cc['timesteps'].shape)}")
    idx = [
        int(((tc["noise"] - e).abs().sum(dim=(1, 2))).argmin())
        for e in cc["noise"]
    ]
    check("... and they are the FIRST two of the training taus, same rows",
          torch.equal(cc["timesteps"], tc["timesteps"][:2][:, idx]),
          f"{cc['timesteps']} vs {tc['timesteps'][:2][:, idx]}")
    check("tau_subset_size is reported",
          r.probe.get("tau_subset_size") == 2, f"{r.probe.get('tau_subset_size')}")
    expected = _hand_g_head(tc, cc, tau_sub=2)
    check("||g_head|| matches the closed form restricted to the SAME subset",
          math.isclose(r.probe["g_headroom_norm"], float(expected.norm()),
                       rel_tol=1e-4),
          f"probe {r.probe['g_headroom_norm']:.8g} vs "
          f"{float(expected.norm()):.8g}")
    # Non-vacuity: the subset must actually change the answer, or this test
    # would pass against an implementation that ignored the knob. Both arms at
    # the same RNG state so only the subset differs.
    torch.manual_seed(606)
    sub = run_probe(
        config_overrides=dict(grad_probe_tau_subset=2)
    ).probe["g_headroom_norm"]
    torch.manual_seed(606)
    full = run_probe().probe["g_headroom_norm"]
    check("... and the subset reading DIFFERS from the full-K one",
          not math.isclose(sub, full, rel_tol=1e-3),
          f"subset {sub} vs full {full}")
    torch.manual_seed(606)
    explicit_full = run_probe(config_overrides=dict(
        grad_probe_tau_subset=len(TAUS))).probe["g_headroom_norm"]
    check("tau_subset == len(tau_centers) reproduces the full-K reading",
          explicit_full == full,
          f"explicit {explicit_full} vs default-0 {full}")


# ═════════════════════════════════════════════════════════════════════════════
# 7 — the row cap and deterministic selection
# ═════════════════════════════════════════════════════════════════════════════

def test_row_cap_and_deterministic_selection():
    print("\n[Rows] grad_probe_max_rows is honoured; selection is deterministic")
    # Four positive rows, advantages 2.0 / 1.4 / 1.1 / 0.8.
    for cap, want in ((1, 1), (2, 2), (3, 3), (4, 4), (9, 4)):
        r = run_probe(config_overrides=dict(grad_probe_max_rows=cap))
        got = r.clean_calls[0]["B"]
        check(f"cap={cap} -> {want} row(s) in the clean forward",
              got == want, f"got {got}")
        check(f"... and n_pos_rows_mean reports {want}",
              r.probe.get("n_pos_rows_mean") == float(want),
              f"{r.probe.get('n_pos_rows_mean')}")

    # WHICH rows: the two highest |advantage| positives, i.e. CHUNKS 0 and 1
    # (advantages 2.0 and 1.4). Row indices are not chunk indices — the sampler
    # shuffles — so map back through each chunk's unique `feat`.
    r2 = run_probe(config_overrides=dict(grad_probe_max_rows=2))
    tc = r2.train_calls[0]
    picked = sorted(r2.row_to_chunk(tc, i) for i in r2.probed_rows())
    check("the two highest-|advantage| positive CHUNKS are selected",
          picked == [0, 1], f"picked chunks {picked}")
    # Repeatability of the whole path, at a pinned RNG state (xi comes from the
    # global stream, so the seed is part of "identical conditions").
    torch.manual_seed(808)
    a = run_probe(config_overrides=dict(grad_probe_max_rows=2)).probe["R_mean"]
    torch.manual_seed(808)
    b = run_probe(config_overrides=dict(grad_probe_max_rows=2)).probe["R_mean"]
    check("two runs at the same RNG state give the IDENTICAL R",
          a == b, f"{a} vs {b}")

    # The selector itself, including the tie case that motivated rejecting topk.
    check("select_grad_probe_rows takes the top-weight rows, ascending",
          select_grad_probe_rows([0, 3, 5, 7], [0.1, 9.0, 0.5, 4.0], 2)
          == [3, 7],
          f"{select_grad_probe_rows([0, 3, 5, 7], [0.1, 9.0, 0.5, 4.0], 2)}")
    check("ties break on the ROW INDEX (paired copies share an advantage)",
          select_grad_probe_rows([9, 2, 5], [1.0, 1.0, 1.0], 2) == [2, 5],
          f"{select_grad_probe_rows([9, 2, 5], [1.0, 1.0, 1.0], 2)}")
    check("a cap above the eligible count returns every row",
          select_grad_probe_rows([4, 1], [1.0, 2.0], 99) == [1, 4])
    check("cap=1 returns exactly one row",
          select_grad_probe_rows([4, 1, 7], [1.0, 2.0, 3.0], 1) == [7])


def test_paired_mode_excludes_fixed_rows():
    print("\n[Rows] paired mode probes only the JITTERED positive rows")
    # jitter_paired=True gives each chunk TWO entries. A "fixed" row's eps' IS
    # eps, so it contributes exactly 0 to `g_head`; including the 2 fixed
    # positives alongside the 2 jittered ones would therefore report HALF the
    # true R, with no other symptom. That factor of 2 in the headline number is
    # the whole reason the mode filter exists.
    #
    # Row identity is NOT recoverable from eps here (a chunk's two copies share
    # it), so this test works on the row COUNT and on the VALUE.
    r = run_probe(
        advantages=(2.0, 1.4, -1.0, -1.6),
        mb_size=8, config_overrides=dict(jitter_paired=True),
    )
    tc, cc = r.train_calls[0], r.clean_calls[0]
    nfi = tc["noise_for_input"]
    check("the training forward really received a jittered noise tensor",
          nfi is not None, "noise_for_input was None on the training leg")
    if nfi is None:
        return

    def _is_jittered(i: int) -> bool:
        return float((nfi[:, i] - tc["noise"][i]).abs().max()) > 0.0

    pos_rows = [
        i for i in range(tc["B"])
        if r.chunks[r.row_to_chunk(tc, i)].advantage > 0
    ]
    jit_pos = [i for i in pos_rows if _is_jittered(i)]
    fixed_pos = [i for i in pos_rows if not _is_jittered(i)]
    check("the batch holds BOTH jittered and fixed positive rows (so the check "
          "is not vacuous)",
          len(jit_pos) >= 1 and len(fixed_pos) >= 1,
          f"jittered {jit_pos}, fixed {fixed_pos}")
    check("the clean forward covers only the JITTERED positive rows",
          cc["B"] == len(jit_pos),
          f"{cc['B']} row(s) vs {len(jit_pos)} jittered positive(s) "
          f"({len(pos_rows)} positives in total)")
    exp_jit = _hand_g_head_rows(tc, jit_pos, tau_sub=len(TAUS))
    exp_all = _hand_g_head_rows(tc, sorted(pos_rows), tau_sub=len(TAUS))
    check("||g_head|| matches the JITTER-rows-only expectation",
          math.isclose(r.probe["g_headroom_norm"], float(exp_jit.norm()),
                       rel_tol=1e-4),
          f"probe {r.probe['g_headroom_norm']:.8g} vs "
          f"jitter-only {float(exp_jit.norm()):.8g}")
    check("... and NOT the DILUTED all-positive-rows expectation",
          not math.isclose(r.probe["g_headroom_norm"], float(exp_all.norm()),
                           rel_tol=1e-3),
          f"probe {r.probe['g_headroom_norm']:.8g} vs "
          f"diluted {float(exp_all.norm()):.8g}")
    check("... and the diluted value really is ~half, i.e. the error this "
          "guards against is material",
          math.isclose(float(exp_all.norm()),
                       float(exp_jit.norm()) * len(jit_pos) / len(pos_rows),
                       rel_tol=1e-4),
          f"{float(exp_all.norm())} vs {float(exp_jit.norm())}")


# ═════════════════════════════════════════════════════════════════════════════
# 8 — the erosion side and its jitter_neg guard
# ═════════════════════════════════════════════════════════════════════════════

def test_erosion_is_free_and_guarded():
    print("\n[Erosion] free at jitter_neg == 0, omitted above it")
    clean = run_probe()
    check("jitter_neg == 0 is reported as the provenance flag",
          clean.probe.get("jitter_neg_is_zero") == 1.0,
          f"{clean.probe.get('jitter_neg_is_zero')}")
    check("g_erosion_norm is present",
          clean.probe.get("g_erosion_norm") is not None, f"{clean.probe}")
    check("reinforce_over_erosion is present",
          clean.probe.get("reinforce_over_erosion") is not None)
    check("n_neg_rows_mean counts the 2 negative non-anchor rows",
          clean.probe.get("n_neg_rows_mean") == 2.0,
          f"{clean.probe.get('n_neg_rows_mean')}")
    check("reinforce_over_erosion == ||g_R|| / ||g_erosion||",
          math.isclose(
              clean.probe["reinforce_over_erosion"],
              clean.probe["g_reinforce_norm"] / clean.probe["g_erosion_norm"],
              rel_tol=1e-9),
          f"{clean.probe['reinforce_over_erosion']}")
    check("the erosion leg costs NO extra forward (still one clean call)",
          len(clean.clean_calls) == 1, f"{len(clean.clean_calls)}")

    dirty = run_probe(config_overrides=dict(jitter_neg=0.05))
    check("jitter_neg > 0 flips the provenance flag to 0.0",
          dirty.probe.get("jitter_neg_is_zero") == 0.0,
          f"{dirty.probe.get('jitter_neg_is_zero')}")
    check("... and g_erosion_norm is OMITTED (not silently mislabelled)",
          "g_erosion_norm" not in dirty.probe, f"{sorted(dirty.probe)}")
    check("... and so is reinforce_over_erosion",
          "reinforce_over_erosion" not in dirty.probe)
    check("... and n_neg_rows_mean is omitted with it",
          "n_neg_rows_mean" not in dirty.probe)
    check("the positive side still reports normally",
          dirty.probe.get("n_probes") == 1
          and dirty.probe.get("R_mean") is not None,
          f"{dirty.probe}")


# ═════════════════════════════════════════════════════════════════════════════
# 9 — aggregation / percentiles against a hand-computed set of probes
# ═════════════════════════════════════════════════════════════════════════════

def _rec(r, cos, *, gr=1.0, ero=None, npos=3, nneg=2):
    d = {
        "g_reinforce_norm": gr,
        "g_headroom_norm": gr * r,
        "g_jit_norm": gr * (1.0 + r),
        "g_erosion_norm": ero,
        "R": r,
        "cos": cos,
        "n_pos_rows": npos,
        "n_neg_rows": nneg if ero is not None else 0,
    }
    return d


def test_aggregation_values():
    print("\n[Aggregation] hand-computed percentiles, means and trend")
    rs = [0.5, 2.0, 14.0, 3.0, 1.0]
    coss = [0.9, -0.2, 0.4, 0.1, 0.6]
    recs = [
        _rec(r, c, gr=2.0, ero=4.0, npos=3 + i, nneg=2)
        for i, (r, c) in enumerate(zip(rs, coss))
    ]
    out = aggregate_grad_probes(
        recs, tau_subset_size=6, jitter_neg_is_zero=True,
        n_skipped=2, n_failed=0,
    )
    arr = np.array(rs, dtype=np.float64)
    check("n_probes", out["n_probes"] == 5, f"{out['n_probes']}")
    check("n_skipped is carried through", out["n_skipped"] == 2)
    check("tau_subset_size is carried through", out["tau_subset_size"] == 6)
    check("R_mean", math.isclose(out["R_mean"], float(arr.mean()),
                                 rel_tol=1e-12), f"{out['R_mean']}")
    for q in (10, 50, 90):
        check(f"R_p{q}",
              math.isclose(out[f"R_p{q}"], float(np.percentile(arr, q)),
                           rel_tol=1e-12),
              f"{out[f'R_p{q}']} vs {float(np.percentile(arr, q))}")
    check("R_max", out["R_max"] == 14.0)
    check("R_first is the FIRST probe (0.5), not the smallest",
          out["R_first"] == 0.5, f"{out['R_first']}")
    check("R_last is the LAST probe (1.0), not the largest",
          out["R_last"] == 1.0, f"{out['R_last']}")
    check("R_p50 == 2.0 for this set",
          out["R_p50"] == 2.0, f"{out['R_p50']}")
    check("cos_reinforce_headroom is the mean cosine",
          math.isclose(out["cos_reinforce_headroom"], float(np.mean(coss)),
                       rel_tol=1e-12),
          f"{out['cos_reinforce_headroom']}")
    check("cos_min surfaces the NEGATIVE outlier the mean hides",
          out["cos_min"] == -0.2 and out["cos_reinforce_headroom"] > 0.0,
          f"min {out['cos_min']} mean {out['cos_reinforce_headroom']}")
    check("g_reinforce_norm / g_headroom_norm / g_jit_norm are probe means",
          out["g_reinforce_norm"] == 2.0
          and math.isclose(out["g_headroom_norm"],
                           float(np.mean([2.0 * r for r in rs])), rel_tol=1e-12)
          and math.isclose(out["g_jit_norm"],
                           float(np.mean([2.0 * (1 + r) for r in rs])),
                           rel_tol=1e-12),
          f"{out}")
    check("n_pos_rows_mean is the mean over probes",
          math.isclose(out["n_pos_rows_mean"], float(np.mean([3, 4, 5, 6, 7])),
                       rel_tol=1e-12),
          f"{out['n_pos_rows_mean']}")
    check("reinforce_over_erosion is the mean of PER-PROBE ratios",
          math.isclose(out["reinforce_over_erosion"], 0.5, rel_tol=1e-12),
          f"{out['reinforce_over_erosion']}")
    check("no n_failed key when nothing failed", "n_failed" not in out)

    # Mixed erosion availability: only the probes that HAVE it may contribute.
    mixed = aggregate_grad_probes(
        [_rec(1.0, 0.5, gr=2.0, ero=4.0), _rec(3.0, 0.5, gr=2.0, ero=None)],
        tau_subset_size=4, jitter_neg_is_zero=True, n_skipped=0, n_failed=1,
    )
    check("erosion mean uses only the probes that carry it",
          mixed["g_erosion_norm"] == 4.0 and mixed["n_neg_rows_mean"] == 2.0,
          f"{mixed}")
    check("R_mean still pools BOTH probes",
          mixed["R_mean"] == 2.0, f"{mixed['R_mean']}")
    check("n_failed is reported when non-zero", mixed["n_failed"] == 1)

    # Empty: counters only, so a systematically-skipping probe is visible.
    empty = aggregate_grad_probes(
        [], tau_subset_size=6, jitter_neg_is_zero=False,
        n_skipped=7, n_failed=0,
    )
    check("no probes -> counters only, no fabricated R",
          empty == {"n_probes": 0, "n_skipped": 7, "tau_subset_size": 6,
                    "jitter_neg_is_zero": 0.0},
          f"{empty}")
    check("a zero erosion norm is excluded from the ratio rather than dividing "
          "by zero",
          "reinforce_over_erosion" not in aggregate_grad_probes(
              [_rec(1.0, 0.5, gr=2.0, ero=0.0)], tau_subset_size=4,
              jitter_neg_is_zero=True, n_skipped=0, n_failed=0),
          "a 0.0 erosion norm reached the division")


# ═════════════════════════════════════════════════════════════════════════════
# 10 — skip and failure accounting
# ═════════════════════════════════════════════════════════════════════════════

def test_skip_accounting():
    print("\n[Skips] too-few-rows, non-finite norms, and hard failures")
    # ONE positive row -> below the 2-row floor -> skipped, not measured.
    thin = run_probe(advantages=(2.0, -1.0, -1.1, -1.2, -1.3, -1.4))
    check("a micro-batch with < 2 positive rows is SKIPPED",
          thin.probe.get("n_probes") == 0 and thin.probe.get("n_skipped") == 1,
          f"{thin.probe}")
    check("... and no clean forward was spent on it",
          thin.clean_calls == [], f"{len(thin.clean_calls)}")
    check("... and the family still reports its counters (visible zero, not a "
          "missing curve)",
          "n_skipped" in thin.probe and "R_mean" not in thin.probe,
          f"{sorted(thin.probe)}")

    # Non-finite clean leg -> non-finite norms -> skipped after the forward.
    nf = run_probe(clean_grad_nonfinite=True)
    check("a non-finite norm is SKIPPED rather than emitted",
          nf.probe.get("n_probes") == 0 and nf.probe.get("n_skipped") == 1,
          f"{nf.probe}")
    check("... and no R/cos key is fabricated",
          "R_mean" not in nf.probe and "cos_reinforce_headroom" not in nf.probe,
          f"{sorted(nf.probe)}")

    # A detached per_tau makes phase 1's autograd.grad raise: the METRIC is lost,
    # the ITERATION is not. Both arms at the same RNG state, because xi comes off
    # the global stream and would otherwise differ between them.
    torch.manual_seed(2024)
    det = run_probe(detach_per_tau=True)
    check("a probe that RAISES is counted in n_failed",
          det.probe.get("n_failed", 0) >= 1, f"{det.probe}")
    check("... and training still completed normally",
          det.result.get("n_updates", 0) >= 1
          and det.result.get("n_micro_batches", 0) >= 1,
          f"n_updates={det.result.get('n_updates')}")
    check("... and it warns rather than failing silently",
          "gradient-decomposition probe (phase 1) failed" in det.stdout,
          det.stdout[-300:])
    # And the training result is untouched by the failure.
    torch.manual_seed(2024)
    off = run_probe(detach_per_tau=True,
                    config_overrides=dict(grad_probe_every=0))
    check("... and the weights match the probe-off run exactly",
          torch.equal(det.w_final, off.w_final),
          f"{det.w_final.tolist()} vs {off.w_final.tolist()}")
    check("... and no clean forward was spent after the phase-1 failure",
          det.clean_calls == [], f"{len(det.clean_calls)}")


def test_cadence_counts_trained_microbatches():
    print("\n[Cadence] every Nth TRAINED micro-batch")
    # 12 chunks / mb 4 / 2 epochs = 6 micro-batches.
    advs = (2.0, 1.7, 1.4, 1.2, 1.0, 0.8, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0)
    for every, want in ((1, 6), (2, 3), (3, 2), (6, 1), (99, 1)):
        r = run_probe(
            advantages=advs, mb_size=4, epochs=2,
            config_overrides=dict(grad_probe_every=every),
        )
        n_mb = r.result.get("n_micro_batches")
        got = r.probe.get("n_probes", 0) + r.probe.get("n_skipped", 0)
        check(f"every={every}: {want} probe attempt(s) over {n_mb} "
              f"micro-batches",
              got == want, f"got {got} (probes {r.probe.get('n_probes')}, "
                           f"skipped {r.probe.get('n_skipped')})")
    r1 = run_probe(advantages=advs, mb_size=4, epochs=2)
    check("at least 2 probes succeeded, so the trend is defined",
          r1.probe.get("n_probes", 0) >= 2, f"{r1.probe}")
    check("R_first and R_last come from DIFFERENT probes when theta drifts",
          r1.probe.get("n_probes", 0) < 2
          or r1.probe["R_first"] != r1.probe["R_last"],
          f"{r1.probe.get('R_first')} vs {r1.probe.get('R_last')}")


# ═════════════════════════════════════════════════════════════════════════════
# Plumbing: flatten helper, TB/wandb emission, banner, config validation
# ═════════════════════════════════════════════════════════════════════════════

def test_both_legs_run_at_the_same_theta():
    print("\n[Sequencing] the clean leg runs at the SAME theta as the jittered "
          "one")
    # This is what the phase-2 placement buys: AFTER `loss.backward()` (so the
    # two graphs never coexist) but BEFORE `optimizer.step()`. If it ran after
    # the step, `g_head` would be `lambda^2 g_P` PLUS one optimizer step of
    # policy drift, silently. The stand-in records `w` at every call, so the
    # property is directly observable.
    #
    # Run at gradient_accumulation_steps 1 AND 2: at k=2 the step fires on the
    # SECOND micro-batch of each window, which is the one an "after the window
    # closes" placement would corrupt (and the `n_updates == 0`-style reasoning
    # used elsewhere would not catch it).
    advs = (2.0, 1.4, 1.1, 0.8, -1.0, -1.6)
    for k in (1, 2):
        torch.manual_seed(4004)
        r = run_probe(advantages=advs, mb_size=6, epochs=4,
                      config_overrides=dict(gradient_accumulation_steps=k))
        pairs = list(zip(r.train_calls, r.clean_calls))
        check(f"k={k}: every micro-batch produced a probe pair "
              f"({len(r.train_calls)} train / {len(r.clean_calls)} clean)",
              len(r.train_calls) == len(r.clean_calls)
              and len(r.train_calls) >= 3,
              f"{len(r.train_calls)} vs {len(r.clean_calls)}")
        check(f"k={k}: the clean leg's theta equals the training leg's, "
              f"micro-batch by micro-batch",
              all(torch.equal(tc["w"], cc["w"]) for tc, cc in pairs),
              "; ".join(
                  f"mb{i}: {tc['w'].tolist()} vs {cc['w'].tolist()}"
                  for i, (tc, cc) in enumerate(pairs)
                  if not torch.equal(tc["w"], cc["w"])
              ))
        # Non-vacuity: theta must actually MOVE across micro-batches, or every
        # `w` would be equal and the check would be trivially satisfied.
        ws = [tuple(tc["w"].tolist()) for tc in r.train_calls]
        check(f"k={k}: ... and theta really did move during the iteration",
              len(set(ws)) > 1, f"{ws}")
        # And the probe still leaves the accumulation window alone.
        torch.manual_seed(4004)
        off = run_probe(advantages=advs, mb_size=6, epochs=4,
                        config_overrides=dict(
                            gradient_accumulation_steps=k, grad_probe_every=0))
        check(f"k={k}: weights and p.grad are bit-identical to the probe-off run",
              torch.equal(r.w_final, off.w_final)
              and torch.equal(r.grad_final, off.grad_final),
              f"{r.w_final.tolist()} vs {off.w_final.tolist()}")
        check(f"k={k}: n_updates is unchanged by the probe",
              r.result.get("n_updates") == off.result.get("n_updates"),
              f"{r.result.get('n_updates')} vs {off.result.get('n_updates')}")


def test_probe_alongside_the_roughness_constraint():
    print("\n[Smooth] the four-way return unpack (smooth x probe)")
    # compute_fm_log_prob appends extras in a FIXED order (per_tau, then the
    # smooth pair), so smooth_active x probe is a distinct unpack arm from either
    # alone. Getting it wrong is an immediate TypeError mid-iteration.
    torch.manual_seed(1717)
    on = run_probe(smooth=True, config_overrides=dict(smooth_coef=0.3))
    check("the run completed with both features on",
          on.result.get("n_micro_batches", 0) >= 1,
          f"{on.result.get('n_micro_batches')}; {on.stdout[-300:]}")
    check("the probe still emitted its family",
          on.probe.get("n_probes") == 1, f"{on.probe}")
    check("the roughness constraint still emitted its own",
          any(k.startswith("smooth_") for k in on.result),
          f"{sorted(k for k in on.result if k.startswith('smooth_'))}")
    check("pooled HF is the pinned 0.1 (so the smooth pair was really unpacked, "
          "not swallowed)",
          math.isclose(on.result.get("smooth_hf_mean", -1), 0.1, rel_tol=1e-6),
          f"{on.result.get('smooth_hf_mean')}")
    # And smooth-only / probe-only still work, so the enumeration is complete.
    torch.manual_seed(1717)
    smooth_only = run_probe(smooth=True, config_overrides=dict(
        smooth_coef=0.3, grad_probe_every=0))
    check("smooth-only (probe off) still unpacks",
          smooth_only.result.get("n_micro_batches", 0) >= 1
          and "_grad_probe" not in smooth_only.result,
          f"{sorted(smooth_only.result)}")
    check("... and the probe does not perturb the smooth reading",
          smooth_only.result.get("smooth_hf_mean")
          == on.result.get("smooth_hf_mean"),
          f"{smooth_only.result.get('smooth_hf_mean')} vs "
          f"{on.result.get('smooth_hf_mean')}")
    check("... nor the weights",
          torch.equal(smooth_only.w_final, on.w_final),
          f"{smooth_only.w_final.tolist()} vs {on.w_final.tolist()}")
    check("the probe's clean forward does NOT run the smooth instrument",
          all(c["n_samples"] == len(TAUS) for c in on.clean_calls),
          "the clean leg was handed smooth work")


def test_flatten_param_grads():
    print("\n[Flatten] unused params become zeros, never dropped")
    p1 = nn.Parameter(torch.zeros(2, 3))
    p2 = nn.Parameter(torch.zeros(4))
    g = flatten_param_grads((torch.ones(2, 3), None), [p1, p2])
    check("length is the FULL parameter count even with an unused param",
          g.numel() == 10, f"{g.numel()}")
    check("the unused slot is zeros",
          torch.equal(g[6:], torch.zeros(4)), f"{g[6:].tolist()}")
    check("the used slot is preserved in order",
          torch.equal(g[:6], torch.ones(6)))
    check("output is fp32 regardless of input dtype",
          flatten_param_grads(
              (torch.ones(2, 3, dtype=torch.bfloat16), None), [p1, p2]
          ).dtype is torch.float32)
    # Two calls must be subtractable — the property the zero-fill exists for.
    a = flatten_param_grads((torch.ones(2, 3), None), [p1, p2])
    b = flatten_param_grads((None, torch.ones(4)), [p1, p2])
    check("two results with DIFFERENT unused sets still subtract",
          (a - b).numel() == 10, "shape mismatch between two grad vectors")


def test_metrics_emission():
    print("\n[Logging] gradprobe/* tags, and the vram/ split")
    gp = {
        "n_probes": 3, "n_skipped": 1, "tau_subset_size": 6,
        "jitter_neg_is_zero": 1.0, "R_mean": 4.5, "R_p90": 9.0,
        "cos_reinforce_headroom": 0.3, "cos_min": -0.1,
        "g_reinforce_norm": 0.02, "g_headroom_norm": 0.09,
        "g_jit_norm": 0.11, "g_erosion_norm": 0.05,
        "reinforce_over_erosion": 0.4, "n_pos_rows_mean": 3.5,
        "n_neg_rows_mean": 2.0, "R_first": 3.0, "R_last": 6.0,
        "_vram_peak_delta_gb": 5.9,
    }
    t = tcf._log_probe(
        {"n_updates": 2, "n_micro_batches": 4, "loss": 0.1,
         "_grad_probe": dict(gp)},
        None, grad_probe_every=20,
    )
    tags = {tag for tag, _v, _s in t.writer.scalars}
    missing = [
        k for k in gp if k != "_vram_peak_delta_gb"
        and f"gradprobe/{k}" not in tags
    ]
    check("every gradprobe key reaches TB under the gradprobe/ prefix",
          not missing, f"missing {missing}")
    check("the VRAM key is re-prefixed to vram/grad_probe_peak_delta",
          "vram/grad_probe_peak_delta" in tags
          and "gradprobe/_vram_peak_delta_gb" not in tags,
          f"{sorted(x for x in tags if 'vram' in x or 'peak' in x)}")
    check("the nested dict does NOT also leak under train/",
          "train/_grad_probe" not in tags)
    vals = {tag: v for tag, v, _s in t.writer.scalars}
    check("values are written unmodified",
          vals["gradprobe/R_mean"] == 4.5
          and vals["vram/grad_probe_peak_delta"] == 5.9,
          f"{vals.get('gradprobe/R_mean')}")

    # Non-finite entries are DROPPED with a warning, not written.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t2 = tcf._log_probe(
            {"n_updates": 1, "_grad_probe": {
                "n_probes": 1, "n_skipped": 0, "tau_subset_size": 6,
                "jitter_neg_is_zero": 1.0, "R_mean": float("inf"),
                "cos_min": float("nan"), "R_p50": 2.0,
            }}, None,
        )
    tags2 = {tag for tag, _v, _s in t2.writer.scalars}
    check("non-finite gradprobe scalars are dropped, not charted",
          "gradprobe/R_mean" not in tags2 and "gradprobe/cos_min" not in tags2
          and "gradprobe/R_p50" in tags2,
          f"{sorted(x for x in tags2 if x.startswith('gradprobe/'))}")
    check("... and the drop is announced",
          "dropped non-finite" in buf.getvalue(), buf.getvalue()[-200:])

    # Absent when the probe never ran.
    t3 = tcf._log_probe({"n_updates": 1, "loss": 0.1}, None)
    check("no gradprobe/* curve exists when the probe is off",
          not any(tag.startswith("gradprobe/")
                  for tag, _v, _s in t3.writer.scalars))


def test_banner():
    print("\n[Banner] the startup line states cadence, cap, subset and cost")
    src = Path(train_grpo.__file__).read_text()
    check("the banner is gated on grad_probe_every > 0",
          "if self.config.grad_probe_every > 0:" in src)
    for frag in ("Gradient-decomposition probe: ON",
                 "trained micro-batch(es)",
                 "row(s) in the clean-",
                 "subset",
                 "added compute"):
        check(f"banner mentions {frag!r}", frag in src)
    check("a jitter_pos == 0 run is warned that R is structurally 0",
          "headroom term is identically zero by construction" in src)


def test_config_validation():
    print("\n[Config] validation and the off-state default")
    c = GRPOConfig(device="cpu")
    check("grad_probe_every defaults to 0 (OFF)", c.grad_probe_every == 0)
    check("grad_probe_max_rows defaults to 4", c.grad_probe_max_rows == 4)
    check("grad_probe_tau_subset defaults to 0 (all taus)",
          c.grad_probe_tau_subset == 0)
    for bad in (-1, -7):
        try:
            GRPOConfig(device="cpu", grad_probe_every=bad)
            raised = False
        except ValueError:
            raised = True
        check(f"grad_probe_every={bad} rejected", raised)
    for bad in (0, -1):
        try:
            GRPOConfig(device="cpu", grad_probe_max_rows=bad)
            raised = False
        except ValueError:
            raised = True
        check(f"grad_probe_max_rows={bad} rejected", raised)
    K = len(GRPOConfig(device="cpu").tau_centers)
    for bad in (-1, K + 1, 99):
        try:
            GRPOConfig(device="cpu", grad_probe_tau_subset=bad)
            raised = False
        except ValueError:
            raised = True
        check(f"grad_probe_tau_subset={bad} rejected (K={K})", raised)
    for ok in (0, 1, K):
        try:
            GRPOConfig(device="cpu", grad_probe_tau_subset=ok)
            raised = False
        except ValueError:
            raised = True
        check(f"grad_probe_tau_subset={ok} accepted", not raised)
    # Validated against the CONFIGURED tau_centers, not the default length.
    try:
        GRPOConfig(device="cpu", tau_centers=[0.0, 0.5],
                   grad_probe_tau_subset=3)
        raised = False
    except ValueError:
        raised = True
    check("the subset bound tracks a CUSTOM tau_centers length", raised)
    # Validation is unconditional — a typo surfaces before the feature is on.
    try:
        GRPOConfig(device="cpu", grad_probe_every=0, grad_probe_max_rows=0)
        raised = False
    except ValueError:
        raised = True
    check("companion knobs are validated even at grad_probe_every=0", raised)


def test_default_config_is_untouched():
    print("\n[Off] the shipped default config is unchanged in every other field")
    a = dataclasses.asdict(GRPOConfig(device="cpu"))
    b = dataclasses.asdict(GRPOConfig(device="cpu", grad_probe_every=0))
    check("grad_probe_every=0 is the default (the off-switch is the shipped "
          "state)", a == b, f"{[k for k in a if a[k] != b[k]]}")
    on = dataclasses.asdict(GRPOConfig(device="cpu", grad_probe_every=20))
    check("... and only that field differs when it is switched on",
          [k for k in a if a[k] != on[k]] == ["grad_probe_every"],
          f"{[k for k in a if a[k] != on[k]]}")


def test_reuse_existing_harness_off_path():
    print("\n[Off] the pre-existing test_grad_accum harness is unaffected")
    # tga's fake drops noise_for_input and detaches per_tau. At the default
    # grad_probe_every=0 it must be completely untouched by this feature.
    base = tga.run_update(k=1, n_groups=2, n_chunks=16, mb_size=4, epochs=2)
    check("no `_grad_probe` key appears in the legacy harness's stats",
          "_grad_probe" not in base.result, f"{sorted(base.result)}")
    check("... and no probe banner/warning is printed",
          "gradient-decomposition probe" not in base.stdout.lower(),
          base.stdout[-200:])
    # With the probe ON that harness's detached per_tau makes phase 1 raise, so
    # the metric is lost and the run is not: same weights as the off arm.
    on = tga.run_update(k=1, n_groups=2, n_chunks=16, mb_size=4, epochs=2,
                        config_overrides=dict(grad_probe_every=1))
    check("a probe failure on the legacy harness leaves the weights identical",
          torch.equal(base.w_final, on.w_final),
          f"max delta {float((base.w_final - on.w_final).abs().max()):.3e}")
    check("... and every legacy stat is unchanged",
          _same_training_stats(base.result, on.result) == "",
          _same_training_stats(base.result, on.result))


if __name__ == "__main__":
    test_probe_off_emits_nothing()
    test_probe_on_does_not_change_training()
    test_rng_leak_would_be_detected()
    test_decomposition_identity_against_closed_form()
    test_R_vanishes_as_lambda_goes_to_zero()
    test_both_legs_share_tau_eps_and_rows()
    test_mismatched_tau_or_rows_is_detected()
    test_tau_subset_applies_to_both_legs()
    test_row_cap_and_deterministic_selection()
    test_paired_mode_excludes_fixed_rows()
    test_erosion_is_free_and_guarded()
    test_aggregation_values()
    test_skip_accounting()
    test_cadence_counts_trained_microbatches()
    test_both_legs_run_at_the_same_theta()
    test_probe_alongside_the_roughness_constraint()
    test_flatten_param_grads()
    test_metrics_emission()
    test_banner()
    test_config_validation()
    test_default_config_is_untouched()
    test_reuse_existing_harness_off_path()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\033[32mAll grad-probe tests passed.\033[0m")
