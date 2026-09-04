"""Tests for the per-row MSE-referenced lower clip, the PAWS k floor, and the
three new diagnostics (`jitter/pos_clip_budget_used`, `drift/*`, `lora/cos_*`).

Features under test:
  A. `GRPOConfig.clip_low_mse_coef` — per-row lower clip bound
     `rho_floor_i = exp(-min(coef * MSE_ref_i, |ln(1 - clip_eps_low)|))`,
     materialised ONCE in `_grpo_update_inner` and shared by all FIVE consumers
     of the lower bound (the surrogate, PAWS's `alive_neg_mask`,
     `train/clipfrac`, `clip_killed_gradient`, the jitter `over_clip` split).
  B. `GRPOConfig.paws_k_floor_at_target` — floor the MEASURED PAWS `k` at
     `positive_advantage_weight_target_ratio` instead of at 1.0.
  C. `jitter/pos_clip_budget_used`, the `drift/*` per-row erosion-drift
     distribution, and the `lora/cos_step_*` weight-step direction cosines.

Conventions follow `test_grad_accum.py`: CPU only, the REAL
`GRPOTrainer._grpo_update` / `_grpo_update_inner` driven with analytic stand-ins,
and the trainer built via `GRPOTrainer.__new__` so `setup()` (GPU, model
download, ZMQ thread) never runs.

TWO harnesses:
  * `test_grad_accum.run_update` is reused verbatim for the OFF-switch
    bit-identity test, so that claim is made through the exact code path the
    accumulation tests already pin (the same precedent `test_smoothness.py` uses).
  * `run_rows()` below is a second, finer harness for the arithmetic: it pins
    each row's `ref_log_prob` AND its `log_ratio` independently, which
    `run_update` cannot (its stand-in derives the log-ratio from the action and
    fixes `ref_log_prob = 0`, i.e. `MSE_ref = 0`, which collapses every per-row
    budget to zero). Direct control of `(MSE_ref_i, log_ratio_i, A_i)` is what
    makes "this row sits 1e-3 nats above its OWN floor" expressible.

Run with the project venv (needs torch; CPU is fine):
    .venv/bin/python scripts/grpo/test_clip_floor.py
"""

import contextlib
import io
import math
import struct
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

import train_grpo  # noqa: E402  (path set up above)
import test_grad_accum as tga  # noqa: E402
from grpo_config import GRPOConfig  # noqa: E402
from train_grpo import GRPOTrainer, clip_killed_gradient  # noqa: E402


# ---------------------------------------------------------------------------
# check() harness (same style as test_grad_accum.py)
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


def close(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Row-level harness: full control of (MSE_ref, log_ratio, advantage) per row
# ---------------------------------------------------------------------------

@dataclass
class _Row:
    """One chunk, specified by exactly the quantities the clip depends on.

    advantage: RAW per-chunk advantage (pre-renorm).
    mse_ref:   MSE_ref = -ref_log_prob. Drives `rho_floor` when the feature is on.
    log_ratio: the log-ratio the stand-in produces for this row, EXACTLY.
    """
    advantage: float
    mse_ref: float
    log_ratio: float
    group_id: int = 0
    is_anchor: bool = False

    # Fields the update loop reads off an ActionChunk.
    @property
    def ref_log_prob(self) -> float:
        return -self.mse_ref


class _Chunk:
    """ActionChunk stand-in. `_prepare_batch` is stubbed, so only these matter."""

    def __init__(self, row: _Row):
        self.row = row
        self.advantage = row.advantage
        self.group_id = row.group_id
        self.is_anchor = row.is_anchor
        self.ref_log_prob = row.ref_log_prob
        self.base_log_prob = row.ref_log_prob
        self.tau_samples = np.zeros(6, dtype=np.float32)
        # Carries the target log-ratio into the stand-in through the action
        # tensor, so the stub needs no side channel keyed on row order.
        self.raw_action = np.full((1, 1), row.log_ratio, dtype=np.float32)


class _ActionHeadStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Identity()


class _TinyModel(nn.Module):
    """One 2-element trainable parameter — stands in for the ~20M LoRA params."""

    def __init__(self, w0=(0.3, -0.2)):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.action_head = _ActionHeadStub()


@dataclass
class _RowRun:
    result: dict
    spy_calls: list          # [(ratio, surr1, surr2, rho_floor, hi)] per micro-batch
    prep_batches: list       # [[_Chunk, ...]] per micro-batch, in order
    config: GRPOConfig
    stdout: str
    w_final: torch.Tensor


def run_rows(rows, *, mb_size=None, epochs=1, spy=True, **overrides) -> _RowRun:
    """Drive the real `_grpo_update` over hand-specified rows.

    The log-prob stand-in returns `ref_log_prob + log_ratio` per row plus a
    w-term pinned to exactly 0.0 in VALUE (`f@w - f@w.detach()`), so
    `log_ratio = current - ref` is EXACTLY the requested value while the
    gradient w.r.t. `w` is still real — the same trick `test_grad_accum` uses.

    `spy=True` wraps `train_grpo.clip_killed_gradient` to capture the production
    call's `(ratio, surr1, surr2, rho_floor, clip_eps_high)`. That call site is
    the only place the per-row bound is observable from outside, and capturing
    `surr2` there is what lets the loss's OWN clamp be cross-checked against the
    four metric consumers.
    """
    cfg_kwargs = dict(
        device="cpu",
        mini_batch_size=mb_size if mb_size is not None else len(rows),
        update_epochs=epochs,
        gradient_accumulation_steps=1,
        balanced_minibatch_training=False,
        dynamic_epoch_training=False,
        per_iteration_advantage_norm=True,     # buffer-wide: preserves row signs
        positive_advantage_weight_scaling=False,
        kl_coef_last_iter=0.0,                 # isolate the surrogate
        kl_coef_base_model=0.0,
        jitter_pos=0.0,
        jitter_neg=0.0,
        max_grad_norm=1e9,
        learning_rate=0.01,
        seed=11,
    )
    cfg_kwargs.update(overrides)
    cfg = GRPOConfig(**cfg_kwargs)

    chunks = [_Chunk(r) for r in rows]
    model = _TinyModel()
    trainer = GRPOTrainer.__new__(GRPOTrainer)      # skip setup(): no GPU here
    trainer.config = cfg
    trainer.device = torch.device("cpu")
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.learning_rate, momentum=0.0, weight_decay=0.0
    )
    trainer.buffer = types.SimpleNamespace(_build_chunks=lambda: list(chunks))
    trainer.iteration = 1
    import threading
    trainer._model_lock = threading.RLock()

    prep_batches: list = []

    def _prepare_batch_stub(self, batch):
        valid = [c for (c, _m) in batch]
        modes = [m for (_c, m) in batch]
        prep_batches.append(list(valid))
        B = len(valid)
        # [B, 1, 1]; the single element IS the row's target log-ratio.
        actions = torch.tensor(
            [[[c.row.log_ratio]] for c in valid], dtype=torch.float32
        )
        return {
            "actions": actions,
            "action_masks": torch.ones_like(actions),
            "initial_noise": torch.zeros_like(actions),
            "advantages": torch.tensor(
                [c.advantage for c in valid], dtype=torch.float32
            ),
            "backbone_output": {"backbone_features": torch.zeros(B, 1, 1)},
            "state_features": torch.zeros(B, 1, 1),
            "embodiment_id": torch.zeros(B, dtype=torch.long),
            "modes": modes,
        }, valid

    trainer._prepare_batch = types.MethodType(_prepare_batch_stub, trainer)

    def _fake_fm_log_prob(**kw):
        acts = kw["actions"]
        target_lr = acts.reshape(acts.shape[0], -1)[:, 0].to(torch.float32)
        batch = prep_batches[-1]
        ref = torch.tensor([c.ref_log_prob for c in batch], dtype=torch.float32)
        adv = torch.tensor([c.advantage for c in batch], dtype=torch.float32)
        # d(log_prob)/dw. Deliberately NOT a function of target_lr alone: several
        # scenarios here need every row at log_ratio == 0, and a feature that
        # vanished there (or that was identical across rows whose renormalized
        # advantages sum to zero) would make the ACCUMULATED gradient exactly
        # 0.0 — which _apply_accumulated_grads legitimately drops, taking
        # n_updates to 0 and every `pos_adv_*` / per-mb-mean stat with it.
        f = torch.stack([
            1.0 + target_lr + 0.10 * adv,
            0.5 + torch.sin(target_lr) - 0.07 * adv,
        ], dim=1)
        wterm = f @ model.w
        lp = (ref + target_lr) + (wterm - (f @ model.w.detach()))
        extras = []
        if kw.get("return_per_tau"):
            K = int(kw["n_samples"])
            extras.append(lp.detach().unsqueeze(0).expand(K, -1))
        if extras:
            return (lp, *extras)
        return lp

    spy_calls: list = []
    real_pred = train_grpo.clip_killed_gradient

    def _spy(ratio, s1, s2, lo, hi):
        spy_calls.append((
            ratio.detach().clone(), s1.detach().clone(), s2.detach().clone(),
            lo.detach().clone() if torch.is_tensor(lo) else lo, hi,
        ))
        return real_pred(ratio, s1, s2, lo, hi)

    real_fm = train_grpo.compute_fm_log_prob
    train_grpo.compute_fm_log_prob = _fake_fm_log_prob
    if spy:
        train_grpo.clip_killed_gradient = _spy
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = trainer._grpo_update()
    finally:
        train_grpo.compute_fm_log_prob = real_fm
        train_grpo.clip_killed_gradient = real_pred

    return _RowRun(
        result=result, spy_calls=spy_calls, prep_batches=prep_batches,
        config=cfg, stdout=buf.getvalue(),
        w_final=model.w.detach().clone(),
    )


def _cfg(**kw):
    """Minimal GRPOConfig for validation-only checks (device pinned to CPU)."""
    base = dict(device="cpu")
    base.update(kw)
    return GRPOConfig(**base)


def expected_floor(coef: float, mse_ref: float, clip_eps_low: float) -> float:
    """Reference implementation of the spec, re-derived independently.

    Includes the SNAP-UP (`max(..., 1 - clip_eps_low)`) so it implements the whole
    spec. Note this is NOT what makes a deleted-snap-up mutant detectable: this
    helper computes in float64, where the `exp(-(-log(1-eps)))` round-trip is
    already exact (old vs new differ by <= 5.6e-17), and the call site's `1e-7`
    tolerance is ~2 fp32 ULPs wide — comfortably wider than the ~3e-8 violation.
    `test_snapup_at_a_nonexact_eps` is what actually kills that mutant, by
    comparing production floors BITWISE at an eps where the round-trip is inexact.
    """
    flat = 1.0 - clip_eps_low
    if coef <= 0.0:
        return flat
    ceiling = -math.log(flat)
    budget = min(coef * max(mse_ref, 0.0), ceiling)
    return max(math.exp(-budget), flat)


# ---------------------------------------------------------------------------
# 1. OFF-switch: determinism + additivity (NOT a HEAD comparison -- see docstring)
# ---------------------------------------------------------------------------

def test_off_switch_determinism_and_additivity():
    """Determinism of the flags-off path + that the new keys are pure additions.

    NOT a HEAD comparison, and deliberately renamed to stop implying one. An
    earlier version of this test was called `test_off_switch_bit_identity` and
    claimed "bit-identical to HEAD's code path", but its two arms were the SAME
    config on the SAME tree — `GRPOConfig()` already defaults
    `clip_low_mse_coef=0.0` / `paws_k_floor_at_target=False`, so
    `dataclasses.asdict(a) == asdict(b)` and the comparison could only ever detect
    nondeterminism. It could not have caught a regression against HEAD, and in fact
    did not: an exact-boundary-ratio batch DID diverge from HEAD under the earlier
    `maximum(minimum(...))` surrogate (0.75x the gradient at a tie), which only a
    real differential test found.

    The genuine differential check cannot live in-tree (once this change is
    committed, HEAD contains it). Run it by hand against any pre-change commit REV:

        mkdir -p /tmp/headtree && for f in train_grpo grpo_config fm_log_prob \
            lora_dit smoothness episode_buffer grpo_server; do
          git show REV:scripts/grpo/$f.py > /tmp/headtree/$f.py; done
        # then drive tga.run_update against both trees and compare
        # result scalars, w_final (fp32 hex) and the post-run RNG draw.

    Scenarios that matter there, learned the hard way: exact-boundary ratios,
    asymmetric clip_eps, gradient_accumulation_steps > 1, PAWS on under both norm
    modes, jitter paired and unpaired, anchors on, mini_batch_size=1, and an
    iteration where every micro-batch is dropped by the non-finite guard.
    """
    print("\n[off] flags-off path is deterministic; new keys are pure additions")

    kw = dict(k=1, n_groups=2, n_chunks=16, mb_size=4, epochs=2, seed=7)

    torch.manual_seed(4242)
    base = tga.run_update(**kw)
    rng_base = torch.randn(4).tolist()

    torch.manual_seed(4242)
    off = tga.run_update(**kw, config_overrides=dict(
        clip_low_mse_coef=0.0, paws_k_floor_at_target=False))
    rng_off = torch.randn(4).tolist()

    # State the vacuity explicitly so nobody re-reads this as a HEAD comparison.
    import dataclasses as _dc
    check("the two arms are the SAME config (defaults ARE the off state)",
          _dc.asdict(base.config) == _dc.asdict(off.config),
          "explicit off-overrides differ from the defaults?")

    # Every PRE-EXISTING stat. `_drift_diag` is a pure addition and is compared
    # separately below, so it is excluded from the "unchanged" claim rather than
    # making that claim vacuous by comparing dicts of dicts.
    def _scalars(res):
        return {k: v for k, v in res.result.items() if not k.startswith("_")}

    sa, sb = _scalars(base), _scalars(off)
    check("same stat key set", set(sa) == set(sb),
          f"only-in-base {sorted(set(sa) - set(sb))}, "
          f"only-in-off {sorted(set(sb) - set(sa))}")
    diffs = [k for k in sa if sa[k] != sb[k]]
    check("the flags-off path is deterministic across runs", not diffs,
          "; ".join(f"{k}: {sa[k]!r} vs {sb[k]!r}" for k in diffs[:4]))
    check("... and reaches bit-identical final weights",
          torch.equal(base.w_final, off.w_final),
          f"max delta {float((base.w_final - off.w_final).abs().max()):.3e}")
    check("the global RNG stream is unchanged", rng_base == rng_off,
          f"{rng_base} vs {rng_off}")

    # ... and the guard is not vacuous: turning the mechanism on DOES move the
    # weights on this same harness (MSE_ref == 0 there, so every budget is 0 and
    # the floor is 1.0 — the maximally tight case, which must bite).
    torch.manual_seed(4242)
    on = tga.run_update(**kw, config_overrides=dict(clip_low_mse_coef=8.0))
    check("... and the comparison is not vacuous (coef>0 changes the weights)",
          not torch.equal(base.w_final, on.w_final))

    # New keys ARE emitted unconditionally, with the flags off.
    check("drift/* is emitted with both flags off (pure addition)",
          bool(base.result.get("_drift_diag")),
          str(base.result.get("_drift_diag")))
    check("drift/* carries all nine documented keys",
          set(base.result["_drift_diag"]) == {
              "neg_down_p10", "neg_down_p50", "neg_down_p90", "neg_down_max",
              "neg_rows", "neg_frac_over_budget", "budget_mean",
              "neg_frac_born_dead", "neg_born_rows"},
          str(sorted(base.result["_drift_diag"])))

    # Config defaults are the off state.
    c = GRPOConfig()
    check("clip_low_mse_coef defaults to 0.0", c.clip_low_mse_coef == 0.0)
    check("paws_k_floor_at_target defaults to False",
          c.paws_k_floor_at_target is False)


# ---------------------------------------------------------------------------
# 2. Per-row floor arithmetic, including the ceiling
# ---------------------------------------------------------------------------

def test_per_row_floor_arithmetic():
    print("\n[floor] per-row rho_floor arithmetic and the clip_eps_low ceiling")

    coef, lo_eps = 8.0, 0.2
    ceiling = -math.log(1.0 - lo_eps)      # 0.22314...
    # MSE_ref values spanning both regimes: coef*mse below the ceiling for the
    # first two, above it for the last two (ceiling binds at mse >= 0.02789).
    mses = [0.0005, 0.0100, 0.0400, 0.5000]
    rows = [
        _Row(advantage=+1.0, mse_ref=mses[0], log_ratio=0.0),
        _Row(advantage=+1.0, mse_ref=mses[1], log_ratio=0.0),
        _Row(advantage=-1.0, mse_ref=mses[2], log_ratio=0.0),
        _Row(advantage=-1.0, mse_ref=mses[3], log_ratio=0.0),
    ]
    r = run_rows(rows, clip_low_mse_coef=coef, clip_eps_low=lo_eps)
    check("exactly one micro-batch (single-batch scenario holds)",
          len(r.spy_calls) == 1, f"{len(r.spy_calls)} calls")

    ratio, _s1, _s2, rho_floor, hi = r.spy_calls[0]
    order = [c.row.mse_ref for c in r.prep_batches[0]]
    got = [float(v) for v in rho_floor]
    want = [expected_floor(coef, m, lo_eps) for m in order]
    check("rho_floor matches exp(-min(coef*MSE_ref, |ln(1-eps)|)) per row",
          all(close(g, w, 1e-7) for g, w in zip(got, want)),
          f"order={order} got={got} want={want}")
    check("clip_eps_high is still passed as the scalar epsilon", hi == 0.2)

    # The ceiling BINDS: the two large-MSE rows are pinned to 1 - clip_eps_low
    # exactly, i.e. the mechanism is never LOOSER than the flat clip.
    pinned = [g for g, m in zip(got, order) if coef * m >= ceiling]
    check("the min(...) ceiling binds (large MSE_ref rows pin to 1-clip_eps_low)",
          len(pinned) == 2 and all(close(p, 1.0 - lo_eps, 1e-7) for p in pinned),
          f"pinned={pinned}")
    # EXACT, not 1e-12-slack: the violation the snap-up prevents is ~6e-8 (one fp32
    # ULP), which a 1e-12 tolerance would swallow whole.
    _flat32 = float(torch.tensor(1.0 - lo_eps, dtype=torch.float32))
    check("no row is EVER below 1-clip_eps_low (tighter or equal, never looser)",
          all(g >= _flat32 for g in got), f"{got} vs flat32 {_flat32!r}")
    check("the un-pinned rows are strictly tighter than the flat clip",
          all(g > 1.0 - lo_eps for g, m in zip(got, order)
              if coef * m < ceiling),
          f"{got}")

    # Off => a full-of-scalar tensor equal to 1 - clip_eps_low, per row.
    r_off = run_rows(rows, clip_low_mse_coef=0.0, clip_eps_low=lo_eps)
    _rt, _a, _b, floor_off, _hi = r_off.spy_calls[0]
    check("coef=0 gives a [B] tensor of exactly 1-clip_eps_low",
          torch.is_tensor(floor_off) and floor_off.shape == (4,)
          and bool((floor_off == torch.full((4,), 1.0 - lo_eps)).all()),
          f"{floor_off.tolist()}")
    check("... on ratio's device and dtype",
          floor_off.dtype == _rt.dtype and floor_off.device == _rt.device)

    # MSE_ref is clamped at >= 0: a (nonsensical) POSITIVE ref_log_prob must not
    # produce a floor above 1.0, which would clip every unmoved row.
    bad = [
        _Row(advantage=+1.0, mse_ref=-0.05, log_ratio=0.0),   # ref_log_prob > 0
        _Row(advantage=-1.0, mse_ref=+0.01, log_ratio=0.0),
    ]
    rb = run_rows(bad, clip_low_mse_coef=coef, clip_eps_low=lo_eps)
    fb = [float(v) for v in rb.spy_calls[0][3]]
    check("negative MSE_ref is clamped to 0 -> floor exactly 1.0, never above",
          max(fb) <= 1.0 + 1e-12 and any(close(v, 1.0, 1e-12) for v in fb),
          f"{fb}")


# ---------------------------------------------------------------------------
# 3. All FIVE clip_eps_low consumers agree, per row
# ---------------------------------------------------------------------------

def test_five_consumers_agree():
    print("\n[agree] all five lower-bound consumers classify each row identically")

    coef, lo_eps, hi_eps = 8.0, 0.2, 0.2
    eps = 2e-3        # nats above/below a row's own budget

    # Four NEGATIVE-advantage rows straddling their OWN (different) floors, plus
    # two positives to keep the buffer-wide z-score well defined and its mean at
    # zero (so post-renorm signs equal pre-renorm ones and the bucket predictions
    # below are exact).
    #
    # Budgets: coef*mse for the first two (below the 0.2231 ceiling), pinned to
    # the ceiling for the third/fourth. So the four negatives straddle TWO
    # DIFFERENT thresholds — a single flat bound cannot reproduce this pattern,
    # which is the point.
    ceiling = -math.log(1.0 - lo_eps)
    neg_specs = [
        (0.0060, +1),    # budget 0.048   -> just INSIDE  (alive)
        (0.0060, -1),    # budget 0.048   -> just OUTSIDE (dead)
        (0.1000, +1),    # budget pinned to 0.2231 -> just INSIDE
        (0.1000, -1),    # budget pinned to 0.2231 -> just OUTSIDE
    ]
    rows = []
    for mse, side in neg_specs:
        budget = min(coef * mse, ceiling)
        # side=+1 -> |log_ratio| = budget - eps (inside); -1 -> budget + eps.
        rows.append(_Row(advantage=-1.0, mse_ref=mse,
                         log_ratio=-(budget - side * eps)))
    rows.append(_Row(advantage=+2.0, mse_ref=0.0100, log_ratio=0.0))
    rows.append(_Row(advantage=+2.0, mse_ref=0.0100, log_ratio=0.0))

    r = run_rows(
        rows, epochs=2,
        clip_low_mse_coef=coef, clip_eps_low=lo_eps, clip_eps_high=hi_eps,
        positive_advantage_weight_scaling=True,
        positive_advantage_weight_target_ratio=1.0,
        # over_clip (consumer 5) only accumulates when jitter is active. The
        # stand-in is value-pinned, so a jitter lambda cannot change any ratio;
        # it only switches the metric block on. jitter_paired=False keeps one
        # entry per chunk, so clipfrac_jitter_neg is a clean per-row fraction.
        jitter_pos=0.05, jitter_neg=0.05, jitter_paired=False,
    )
    check("exactly one micro-batch per epoch", len(r.spy_calls) == 2,
          f"{len(r.spy_calls)} calls")

    ratio, surr1, surr2, rho_floor, hi = r.spy_calls[0]
    order = r.prep_batches[0]
    adv_pre = torch.tensor([c.advantage for c in order])

    # --- consumer 1: the LOSS's own clamp ---------------------------------
    want_s2_clamp = torch.maximum(
        torch.minimum(ratio, torch.full_like(ratio, 1 + hi_eps)), rho_floor
    )
    # surr2 = A_post * clamp; recover A_post from surr1 = A_post * ratio.
    a_post = surr1 / ratio
    check("consumer 1 (surr2) applies exactly the per-row rho_floor",
          torch.allclose(surr2, a_post * want_s2_clamp, atol=1e-7),
          f"{surr2.tolist()} vs {(a_post * want_s2_clamp).tolist()}")
    below = ratio < rho_floor
    check("... and it MOVED only the rows below their own floor",
          torch.equal(want_s2_clamp != ratio, below),
          f"moved={(want_s2_clamp != ratio).tolist()} below={below.tolist()}")
    check("the batch really straddles per-row floors (2 in, 2 out of 4 negs)",
          int(below.sum()) == 2, f"below={below.tolist()}")

    # --- consumer 4: clip_killed_gradient --------------------------------
    dead = clip_killed_gradient(ratio, surr1, surr2, rho_floor, hi_eps)
    neg_mask = adv_pre < 0
    check("consumer 4 (clip_killed_gradient) kills exactly the below-floor negs",
          torch.equal(dead, below & neg_mask),
          f"dead={dead.tolist()} expected={(below & neg_mask).tolist()}")

    # --- consumer 3: train/clipfrac --------------------------------------
    want_clipfrac = float(
        ((ratio < rho_floor) | (ratio > 1 + hi_eps)).float().mean()
    )
    check("consumer 3 (train/clipfrac) counts the same below-floor rows",
          close(r.result["clipfrac"], want_clipfrac, 1e-9),
          f"{r.result['clipfrac']} vs {want_clipfrac}")

    # --- consumer 5: over_clip -> clipfrac_jitter_neg --------------------
    want_jit_neg = float(
        (((ratio < rho_floor) | (ratio > 1 + hi_eps)) & neg_mask
         ).float().sum() / int(neg_mask.sum())
    )
    check("consumer 5 (over_clip -> clipfrac_jitter_neg) agrees",
          close(r.result["clipfrac_jitter_neg"], want_jit_neg, 1e-9),
          f"{r.result.get('clipfrac_jitter_neg')} vs {want_jit_neg}")

    # --- consumer 2: PAWS alive_neg_mask ---------------------------------
    # N pools |row_loss| over ALIVE negatives, per trained micro-batch. Both
    # micro-batches see the same rows and (value-pinned stand-in) the same
    # ratios, so the iteration total is 2x the per-mb mass.
    row_loss = -torch.min(surr1, surr2)
    alive = neg_mask & (ratio >= rho_floor)
    want_n = 2.0 * float(row_loss[alive].abs().sum())
    check("consumer 2 (PAWS alive_neg_mass) excludes exactly the dead negs",
          close(r.result["pos_adv_alive_neg_mass"], want_n, 1e-5),
          f"{r.result.get('pos_adv_alive_neg_mass')} vs {want_n}")
    # Falsifiability: the FLAT bound would admit a different set of rows, so the
    # mass computed against it must differ from what the trainer reported.
    flat_alive = neg_mask & (ratio >= 1 - lo_eps)
    check("... and the FLAT-bound mass would have been different (test bites)",
          int(flat_alive.sum()) != int(alive.sum()),
          f"flat admits {int(flat_alive.sum())}, per-row admits {int(alive.sum())}")

    # One consolidated statement of the property: NO SINGLE SCALAR bound can
    # reproduce the classification this batch produced. Two negative rows exist
    # with ratio_i < ratio_j where the LOWER one is inside its floor and the
    # higher one is outside — impossible under any flat threshold, so this pins
    # that the bound really is per-row rather than a relabelled constant.
    neg_idx = [i for i in range(len(order)) if adv_pre[i] < 0]
    inverted = any(
        float(ratio[i]) < float(ratio[j])
        and not bool(below[i]) and bool(below[j])
        for i in neg_idx for j in neg_idx
    )
    check("no flat scalar bound could reproduce this classification",
          inverted,
          f"ratios={[round(float(ratio[i]), 5) for i in neg_idx]} "
          f"below={[bool(below[i]) for i in neg_idx]}")


# ---------------------------------------------------------------------------
# 4. A tight floor cannot kill a positive-advantage row
# ---------------------------------------------------------------------------

def test_positive_rows_survive_a_tight_floor():
    print("\n[four-case] a tight per-row floor never kills a positive row")

    coef, lo_eps, hi_eps = 8.0, 0.2, 0.2
    # MSE_ref ~ 0 => budget ~ 0 => rho_floor ~ 1.0, the tightest possible floor.
    # Every row's ratio is put BELOW it.
    rows = [
        _Row(advantage=+1.0, mse_ref=1e-6, log_ratio=-0.05),
        _Row(advantage=+1.0, mse_ref=1e-6, log_ratio=-0.05),
        _Row(advantage=-1.0, mse_ref=1e-6, log_ratio=-0.05),
        _Row(advantage=-1.0, mse_ref=1e-6, log_ratio=-0.05),
    ]
    r = run_rows(rows, clip_low_mse_coef=coef,
                 clip_eps_low=lo_eps, clip_eps_high=hi_eps)
    ratio, surr1, surr2, rho_floor, _hi = r.spy_calls[0]

    check("every row is below its own floor (setup precondition)",
          bool((ratio < rho_floor).all()),
          f"ratio={ratio.tolist()} floor={rho_floor.tolist()}")
    check("train/clipfrac reads 1.0 (the sign-agnostic false positive)",
          close(r.result["clipfrac"], 1.0, 1e-12), str(r.result["clipfrac"]))
    check("clipfrac_effective_pos is 0.0 — positives cannot die on the LOWER bound",
          close(r.result["clipfrac_effective_pos"], 0.0, 1e-12),
          str(r.result.get("clipfrac_effective_pos")))
    check("clipfrac_effective_neg is 1.0 — negatives do die there",
          close(r.result["clipfrac_effective_neg"], 1.0, 1e-12),
          str(r.result.get("clipfrac_effective_neg")))

    # Autograd oracle on the real surrogate: a positive row's clip gradient must
    # still flow with the ratio pinned below a floor of ~1.0.
    for a, label in ((+1.0, "A>0"), (-1.0, "A<0")):
        rho = torch.tensor(0.95, requires_grad=True)
        floor = torch.tensor(1.0)
        av = torch.tensor(a)
        (-torch.min(
            av * rho,
            av * torch.maximum(torch.minimum(rho, torch.tensor(1.2)), floor),
        )).backward()
        alive = float(rho.grad) != 0.0
        check(f"autograd: {label} with rho below a 1.0 floor -> "
              f"{'ALIVE' if a > 0 else 'DEAD'}",
              alive == (a > 0), f"grad={float(rho.grad)}")

    # Anchor rows carry a constant POSITIVE advantage, so the same inertness must
    # hold for them: a maximally tight floor must not kill an anchor row. The
    # anchor's advantage must be non-zero — a zero-advantage row has
    # surr1 == surr2 and is DOCUMENTED as reported dead whenever the clamp moved
    # (see clip_killed_gradient), which would test the wrong thing.
    arows = [
        _Row(advantage=+1.0, mse_ref=1e-6, log_ratio=-0.05),
        _Row(advantage=-1.0, mse_ref=1e-6, log_ratio=-0.05),
        _Row(advantage=+0.5, mse_ref=1e-6, log_ratio=-0.05, is_anchor=True),
    ]
    ra = run_rows(
        arows, clip_low_mse_coef=coef,
        clip_eps_low=lo_eps, clip_eps_high=hi_eps,
        include_anchor_groups=True, anchor_advantage=1.0,
    )
    ratio_a, s1_a, s2_a, floor_a, _h = ra.spy_calls[0]
    anchor_idx = [i for i, c in enumerate(ra.prep_batches[0]) if c.is_anchor]
    dead_a = clip_killed_gradient(ratio_a, s1_a, s2_a, floor_a, hi_eps)
    check("anchor rows get a rho_floor entry (they pass through the surrogate)",
          floor_a.numel() == ratio_a.numel() and len(anchor_idx) == 1,
          f"floor={floor_a.tolist()} anchor_idx={anchor_idx}")
    check("... and a maximally tight floor leaves the anchor row's gradient ALIVE",
          not bool(dead_a[anchor_idx[0]]), str(dead_a.tolist()))


# ---------------------------------------------------------------------------
# 5. (B) the PAWS k floor
# ---------------------------------------------------------------------------

def _paws_run(*, floor_at_target: bool, tratio: float, dead_negs: int,
              pos_log_ratio: float = 0.0, **over):
    """1 positive + 3 negatives; `dead_negs` of the negatives are clip-dead.

    Per-iteration renorm keeps raw magnitudes, so A = [1.5, -0.5, -0.5, -0.5]
    for raw advantages (+3, -1, -1, -1) (mean 0, ddof=1 std 2.0). Live negatives
    sit at ratio 1.0, so N/D = (n_alive_neg * 0.5) / (1.5 * exp(pos_log_ratio))
    and the measured k is clamp(tratio * N/D, floor, max) — reachable because
    `epochs=2` gives a SECOND micro-batch whose k is measured from the first's
    pooled mass (the first uses the unmeasured prior, D_iter being 0).
    """
    lo_eps = 0.2
    coef = 8.0
    ceiling = -math.log(1.0 - lo_eps)
    rows = [_Row(advantage=+3.0, mse_ref=0.10, log_ratio=pos_log_ratio)]
    for i in range(3):
        if i < dead_negs:
            # MSE_ref 0.10 pins the budget to the ceiling; sit just past it.
            rows.append(_Row(advantage=-1.0, mse_ref=0.10,
                             log_ratio=-(ceiling + 0.05)))
        else:
            rows.append(_Row(advantage=-1.0, mse_ref=0.10, log_ratio=0.0))
    kwargs = dict(
        clip_low_mse_coef=coef, clip_eps_low=lo_eps,
        positive_advantage_weight_scaling=True,
        positive_advantage_weight_target_ratio=tratio,
        positive_advantage_weight_max=10.0,
        paws_k_floor_at_target=floor_at_target,
    )
    kwargs.update(over)
    return run_rows(rows, epochs=2, **kwargs)


def test_paws_k_floor():
    print("\n[paws] k floors at target_ratio with the flag on, at 1.0 with it off")

    tratio = 1.75
    # 2 of 3 negatives clip-dead -> N/D = 0.5/1.5 = 1/3 -> tratio*N/D = 0.583.
    off = _paws_run(floor_at_target=False, tratio=tratio, dead_negs=2)
    on = _paws_run(floor_at_target=True, tratio=tratio, dead_negs=2)

    nd_off = off.result["pos_adv_alive_neg_mass"] / off.result["pos_adv_pos_mass"]
    check("scenario really has N/D < 1/tratio (the floor can bind)",
          nd_off * tratio < 1.0, f"N/D={nd_off:.4f}, tratio*N/D={nd_off*tratio:.4f}")
    check("flag OFF: measured k floors at 1.0",
          close(off.result["pos_adv_weight_k_min"], 1.0, 1e-9),
          f"k_min={off.result.get('pos_adv_weight_k_min')}")
    check("flag ON: measured k floors at target_ratio",
          close(on.result["pos_adv_weight_k_min"], tratio, 1e-9),
          f"k_min={on.result.get('pos_adv_weight_k_min')}")
    check("flag ON: k_last also reads target_ratio",
          close(on.result["pos_adv_weight_k"], tratio, 1e-9),
          f"k={on.result.get('pos_adv_weight_k')}")

    # N/D > 1: the floor is inert and both arms must agree exactly. The positive
    # row is put slightly below ratio 1.0 so N/D is STRICTLY above 1 — at exactly
    # N/D == 1 the `D_iter + _POS_SCALE_EPS` denominator puts the measured k a
    # part in 1e8 BELOW tratio, so a tratio floor would bind by that much and the
    # "inert" claim would be a knife-edge artefact rather than the property.
    off_hi = _paws_run(floor_at_target=False, tratio=tratio, dead_negs=0,
                       pos_log_ratio=-0.02)
    on_hi = _paws_run(floor_at_target=True, tratio=tratio, dead_negs=0,
                      pos_log_ratio=-0.02)
    nd_hi = (off_hi.result["pos_adv_alive_neg_mass"]
             / off_hi.result["pos_adv_pos_mass"])
    check("scenario 2 really has N/D > 1 (all negatives alive)",
          nd_hi > 1.0 + 1e-4, f"N/D={nd_hi:.6f}")
    check("scenario 2 puts tratio*N/D above the tratio floor",
          tratio * nd_hi > tratio, f"tratio*N/D={tratio * nd_hi:.6f}")
    check("N/D > 1: k is UNCHANGED by the flag (floor inert)",
          close(off_hi.result["pos_adv_weight_k"],
                on_hi.result["pos_adv_weight_k"], 1e-12)
          and close(off_hi.result["pos_adv_weight_k"], tratio * nd_hi, 1e-6),
          f"off={off_hi.result['pos_adv_weight_k']} "
          f"on={on_hi.result['pos_adv_weight_k']} want={tratio * nd_hi}")

    # The cap still wins over the floor.
    capped = _paws_run(floor_at_target=True, tratio=1.75, dead_negs=2)
    check("k never exceeds positive_advantage_weight_max",
          capped.result["pos_adv_weight_k"] <= capped.config.
          positive_advantage_weight_max + 1e-12)

    # The two OTHER k branches are untouched by the flag. With a SINGLE
    # micro-batch D_iter is 0 for it, so k comes from the unmeasured branch:
    #   per_iteration_advantage_norm=False -> the analytic prior min(max(t,1), max)
    #   per_iteration_advantage_norm=True  -> the hard 1.0 fallback (no prior)
    for per_iter_norm, want_k, label in (
        (False, tratio, "analytic prior"),
        (True, 1.0, "per_iteration_advantage_norm fallback"),
    ):
        ks = []
        for flag in (False, True):
            one = run_rows(
                [_Row(advantage=+3.0, mse_ref=0.10, log_ratio=0.0),
                 _Row(advantage=-1.0, mse_ref=0.10, log_ratio=-0.01),
                 _Row(advantage=-1.0, mse_ref=0.10, log_ratio=-0.02)],
                epochs=1,
                per_iteration_advantage_norm=per_iter_norm,
                positive_advantage_weight_scaling=True,
                positive_advantage_weight_target_ratio=tratio,
                paws_k_floor_at_target=flag,
            )
            ks.append((one.result.get("pos_adv_weight_k"),
                       "pos_adv_weight_k_min" in one.result))
        check(f"{label} branch unchanged by the flag (k == {want_k}, "
              f"no measured k_min)",
              ks[0] == ks[1] and close(ks[0][0], want_k, 1e-12)
              and ks[0][1] is False,
              f"off={ks[0]} on={ks[1]}")

    # Validation: flooring below the no-op point is rejected.
    for bad in (0.5, 0.99):
        try:
            GRPOConfig(paws_k_floor_at_target=True,
                       positive_advantage_weight_target_ratio=bad)
            raised = False
        except ValueError:
            raised = True
        check(f"target_ratio={bad} with the flag on is rejected", raised)
    try:
        GRPOConfig(paws_k_floor_at_target=True,
                   positive_advantage_weight_target_ratio=1.0)
        ok = True
    except ValueError:
        ok = False
    check("target_ratio=1.0 with the flag on is accepted", ok)


# ---------------------------------------------------------------------------
# 6. Monotonicity: smaller coef -> fewer alive erosion rows
# ---------------------------------------------------------------------------

def test_monotone_in_coefficient():
    print("\n[monotone] smaller clip_low_mse_coef -> fewer alive erosion rows")

    lo_eps = 0.2
    # Negative rows spread across the whole nat range so a coefficient sweep
    # crosses several of them one at a time.
    rows = [_Row(advantage=+3.0, mse_ref=0.02, log_ratio=0.0)]
    for i, lr in enumerate((-0.01, -0.03, -0.06, -0.10, -0.16)):
        rows.append(_Row(advantage=-0.6, mse_ref=0.02, log_ratio=lr))

    alive_counts, masses = [], []
    coefs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 0.0]     # 0.0 == the flat clip
    for coef in coefs:
        r = run_rows(rows, epochs=2, clip_low_mse_coef=coef,
                     clip_eps_low=lo_eps,
                     positive_advantage_weight_scaling=True)
        ratio, _s1, _s2, floor, _h = r.spy_calls[0]
        adv_pre = torch.tensor([c.advantage for c in r.prep_batches[0]])
        alive_counts.append(int(((adv_pre < 0) & (ratio >= floor)).sum()))
        masses.append(r.result["pos_adv_alive_neg_mass"])

    sweep = coefs[:-1]                               # ascending, excluding 0.0
    check("alive erosion rows are NON-DECREASING in clip_low_mse_coef",
          all(a <= b for a, b in zip(alive_counts[:-1], alive_counts[1:-1])),
          f"coefs={sweep} alive={alive_counts[:-1]}")
    check("... and the sweep is not degenerate (the count actually moves)",
          len(set(alive_counts[:-1])) > 1, f"alive={alive_counts[:-1]}")
    check("alive erosion MASS is likewise non-decreasing",
          all(a <= b + 1e-9 for a, b in zip(masses[:-1], masses[1:-1])),
          f"masses={[round(m, 5) for m in masses[:-1]]}")
    check("the flat clip (coef=0) admits at least as many rows as any coef",
          alive_counts[-1] >= max(alive_counts[:-1]),
          f"flat={alive_counts[-1]} vs {alive_counts[:-1]}")


# ---------------------------------------------------------------------------
# 7. C2: drift/* values against a hand-computed batch
# ---------------------------------------------------------------------------

def _quantile_linear(vals, p):
    """Independent re-derivation of torch.quantile's 'linear' interpolation."""
    v = sorted(float(x) for x in vals)
    if len(v) == 1:
        return v[0]
    pos = p * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (pos - lo) * (v[hi] - v[lo])


def test_drift_diagnostics_values():
    print("\n[drift] pooled per-row erosion-drift distribution, signed and one-sided")

    lo_eps, coef = 0.2, 8.0
    ceiling = -math.log(1.0 - lo_eps)
    # `neg_down_* = -log_ratio`, so log_ratio=-0.030 means the row travelled +0.030
    # DOWN (toward its floor). One row deliberately drifts UP (log_ratio > 0): the
    # lower clip cannot touch it, so it must NOT count as over budget even though
    # its MAGNITUDE exceeds the budget. That row is what exposes a two-sided test.
    neg = [
        (0.0020, -0.005),    # budget 0.0160  down +0.005 -> inside
        (0.0020, -0.030),    # budget 0.0160  down +0.030 -> OVER
        (0.0100, -0.050),    # budget 0.0800  down +0.050 -> inside
        (0.1000, -0.300),    # budget capped 0.2231, down +0.300 -> OVER
        (0.0100, +0.600),    # budget 0.0800  down -0.600 -> UP, must NOT count
    ]
    rows = [_Row(advantage=-1.0, mse_ref=m, log_ratio=lr) for m, lr in neg]
    rows.append(_Row(advantage=+2.5, mse_ref=0.01, log_ratio=-0.5))
    rows.append(_Row(advantage=+2.5, mse_ref=0.01, log_ratio=-0.5))
    rows.append(_Row(advantage=0.0, mse_ref=0.01, log_ratio=-0.9, is_anchor=True))

    r = run_rows(rows, clip_low_mse_coef=coef, clip_eps_low=lo_eps,
                 include_anchor_groups=True, anchor_advantage=1.0)
    d = r.result["_drift_diag"]

    down = [-lr for _m, lr in neg]
    budg = [min(coef * m, ceiling) for m, _lr in neg]
    want_over = sum(1 for x, b in zip(down, budg) if x > b) / len(neg)

    check("neg_rows counts pre-renorm-negative, NON-anchor rows only",
          d["neg_rows"] == len(neg), f"{d['neg_rows']} vs {len(neg)}")
    for pct, qq in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        want = _quantile_linear(down, qq)
        check(f"{pct} matches an independent linear-interpolation quantile",
              close(d[f"neg_down_{pct}"], want, 1e-6),
              f"{d[f'neg_down_{pct}']} vs {want}")
    check("max matches max(-log_ratio) over those rows",
          close(d["neg_down_max"], max(down), 1e-6),
          f"{d['neg_down_max']} vs {max(down)}")
    check("the UP-drifted row shows as a NEGATIVE percentile value, not |.|",
          d["neg_down_p10"] < 0.0, str(d["neg_down_p10"]))
    check("frac_over_budget is ONE-SIDED: up-drifted row excluded (2/5)",
          close(d["neg_frac_over_budget"], want_over, 1e-6)
          and close(want_over, 0.4, 1e-12),
          f"{d['neg_frac_over_budget']} vs {want_over}")
    check("... where a two-sided |log_ratio| test would have said 3/5",
          sum(1 for x, b in zip(down, budg) if abs(x) > b) / len(neg) == 0.6)
    check("budget_mean is the pooled mean of the per-row budgets",
          close(d["budget_mean"], sum(budg) / len(budg), 1e-6),
          f"{d['budget_mean']} vs {sum(budg)/len(budg)}")
    check("the positives' |log_ratio| (0.5) is NOT in the population",
          d["neg_down_max"] < 0.5, str(d["neg_down_max"]))

    # STRICT `>`: a row exactly AT its budget is not clipped by the loss (the clip
    # fires on `ratio < rho_floor`), so it must not be counted over budget.
    #
    # The metric compares `-log_ratio` against `-log(rho_floor)`. `rho_floor` does
    # NOT depend on log_ratio, so read the budget off a probe run and then set
    # `log_ratio = -budget` — fp32 negation is exact, so the tie is guaranteed and
    # needs no search. (An earlier attempt searched for a tie in `-log(ratio)`, i.e.
    # after an exp->log round-trip, which is a DIFFERENT quantity from `-log_ratio`;
    # the tie it found was not a tie for the metric, and the strict-vs-non-strict
    # mutation survived.)
    _probe = run_rows(
        [_Row(advantage=-1.0, mse_ref=0.0100, log_ratio=-0.001),
         _Row(advantage=+1.0, mse_ref=0.0100, log_ratio=-0.001)],
        clip_low_mse_coef=8.0, clip_eps_low=lo_eps)
    _bud32 = float((-torch.log(_probe.spy_calls[0][3][0])).float())
    exact = run_rows(
        [_Row(advantage=-1.0, mse_ref=0.0100, log_ratio=-_bud32),
         _Row(advantage=+1.0, mse_ref=0.0100, log_ratio=-0.001)],
        clip_low_mse_coef=8.0, clip_eps_low=lo_eps)
    _de = exact.result["_drift_diag"]
    check("the tie is genuinely BITWISE in the metric's own units (guards the guard)",
          close(_de["neg_down_max"], _bud32, 0.0),
          f"neg_down_max={_de['neg_down_max']!r} budget={_bud32!r}")
    check("a row exactly AT its budget is NOT over budget (strict >)",
          _de["neg_frac_over_budget"] == 0.0,
          f"{_de['neg_frac_over_budget']} (>= would give 1.0)")
    # Same tie, PAWS on: the alive-erosion mask is `r_det >= rho_floor`, so a row
    # sitting exactly ON its floor is ALIVE and must contribute to N. Under `>` it
    # would be excluded and N would collapse to 0 (it is the only negative row).
    exact_paws = run_rows(
        [_Row(advantage=-1.0, mse_ref=0.0100, log_ratio=-_bud32),
         _Row(advantage=+1.0, mse_ref=0.0100, log_ratio=-0.001)],
        clip_low_mse_coef=8.0, clip_eps_low=lo_eps,
        positive_advantage_weight_scaling=True,
        positive_advantage_weight_target_ratio=2.0)
    check("PAWS counts a row exactly ON its floor as ALIVE erosion (>= not >)",
          exact_paws.result["pos_adv_alive_neg_mass"] > 0.0,
          f"N={exact_paws.result.get('pos_adv_alive_neg_mass')!r}")

    # With the feature OFF the budget is the FLAT |ln(1-clip_eps_low)|.
    r_off = run_rows(rows, clip_low_mse_coef=0.0, clip_eps_low=lo_eps,
                     include_anchor_groups=True, anchor_advantage=1.0)
    d_off = r_off.result["_drift_diag"]
    want_off = sum(1 for x in down if x > ceiling) / len(neg)
    check("coef=0: frac_over_budget uses the flat |ln(1-clip_eps_low)| (1/5)",
          close(d_off["neg_frac_over_budget"], want_off, 1e-6)
          and close(want_off, 0.2, 1e-12),
          f"{d_off['neg_frac_over_budget']} vs {want_off}")
    check("... and the percentiles are unchanged by the coefficient",
          all(close(d[k], d_off[k], 1e-12) for k in (
              "neg_down_p10", "neg_down_p50", "neg_down_p90", "neg_down_max")))

    # POOLED over every trained micro-batch. This is the load-bearing choice: the
    # first micro-batch runs at theta == theta_ref (_compute_ref_log_probs runs
    # before _grpo_update), so a first-micro-batch-only reading measures no drift.
    # With mb_size=2 the 5 negative rows span several micro-batches and ALL of
    # them must be in the pool.
    two = run_rows(rows, mb_size=2, epochs=1, clip_low_mse_coef=coef,
                   clip_eps_low=lo_eps, include_anchor_groups=True,
                   anchor_advantage=1.0)
    d2 = two.result["_drift_diag"]
    check("drift/* POOLS every trained micro-batch (not just the first)",
          d2["neg_rows"] == len(neg),
          f"{d2['neg_rows']} vs {len(neg)} at mini_batch_size=2")
    check("... so the pooled max matches the single-batch max",
          close(d2["neg_down_max"], max(down), 1e-6),
          f"{d2['neg_down_max']} vs {max(down)}")

    # neg_frac_born_dead is PRE-STEP scoped. In this mb_size=2 decomposition the
    # first micro-batch is all-POSITIVE (verified: mb0 = [+2.5]), so no pre-step
    # micro-batch holds a negative row and the born keys must be ABSENT — a curve
    # gap. An earlier revision latched on "first micro-batch WITH a negative row",
    # which deferred the capture to mb1, past an optimizer.step(), and reported
    # post-step drift as at-birth death (measured 1.0 where the truth was 0.0).
    # PRECONDITION, asserted so a sampler change cannot make this vacuous: the
    # test only means something if micro-batch 0 really holds no negative row.
    _mb0_advs = [c.advantage for c in two.prep_batches[0]]
    check("precondition: micro-batch 0 holds no negative row",
          all(a > 0 for a in _mb0_advs), f"mb0 advantages={_mb0_advs}")
    check("born keys ABSENT when no PRE-STEP micro-batch holds a negative row",
          "neg_frac_born_dead" not in d2 and "neg_born_rows" not in d2,
          f"{sorted(d2)}")
    check("... while the pooled family is still emitted",
          d2["neg_rows"] == len(neg), str(d2.get("neg_rows")))
    check("single-micro-batch run: born rows == pooled rows",
          d["neg_born_rows"] == d["neg_rows"],
          f"{d['neg_born_rows']} vs {d['neg_rows']}")
    # ... and the VALUE is asserted, not just the key. 2 of the 5 negative rows are
    # past their own budget at theta == theta_ref (run_rows pins log_ratio, so the
    # measured value IS the at-birth value).
    check("neg_frac_born_dead VALUE matches the hand count (2/5)",
          close(d["neg_frac_born_dead"], want_over, 1e-6),
          f"{d['neg_frac_born_dead']} vs {want_over}")

    # No negative signal rows at all -> the family is ABSENT (a curve gap), not 0.
    pos_only = run_rows(
        [_Row(advantage=+1.0, mse_ref=0.01, log_ratio=0.01),
         _Row(advantage=+2.0, mse_ref=0.01, log_ratio=0.02)],
        clip_low_mse_coef=coef, clip_eps_low=lo_eps,
    )
    check("no negative signal rows -> _drift_diag absent, not a fabricated 0",
          "_drift_diag" not in pos_only.result,
          str(pos_only.result.get("_drift_diag")))


def test_pos_clip_budget_used():
    print("\n[jitter] pos_clip_budget_used shares the FLAT neg denominator")

    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.config = GRPOConfig(device="cpu", clip_eps_low=0.08,
                                jitter_pos=0.25, jitter_neg=0.05)
    trainer.device = torch.device("cpu")
    trainer.model = _TinyModel()          # only `.action_head` is read

    # Two rows (one positive, one negative) with hand-set gaps, produced by a
    # stand-in whose jittered log-prob is lower than the clean one by a fixed
    # per-row amount. gap = lp_clean - lp_jit.
    gap_pos, gap_neg = 0.06, 0.004
    K, B = 2, 2
    gaps = torch.tensor([gap_pos, gap_neg])

    def _fake(**kw):
        lp = torch.zeros(B)
        per_tau = torch.zeros(K, B)
        if kw.get("noise_for_input") is not None:
            per_tau = per_tau - gaps.unsqueeze(0)
        return lp, per_tau

    real_fm = train_grpo.compute_fm_log_prob
    train_grpo.compute_fm_log_prob = _fake
    try:
        out = trainer._jitter_gap_diagnostics(
            ready_backbone={"backbone_features": torch.zeros(B, 1, 1)},
            ready_state_features=torch.zeros(B, 1, 1),
            ready_embodiment_id=torch.zeros(B, dtype=torch.long),
            ready_actions=torch.zeros(B, 1, 1),
            ready_masks=torch.ones(B, 1, 1),
            ready_noise=torch.zeros(B, 1, 1),
            timesteps=torch.zeros(K, B),
            noise_for_input=torch.zeros(K, B, 1, 1),
            lam_row=torch.tensor([0.25, 0.05]),
            pos_adv_mask=torch.tensor([True, False]),
            fixed_row_mask=torch.tensor([False, False]),
            jitter_row_mask=torch.tensor([True, True]),
        )
    finally:
        train_grpo.compute_fm_log_prob = real_fm

    # 1e-6 tolerances: the gap is reduced in float32, so a 0.06-nat gap arrives
    # with ~1e-8 of representation error.
    lo_budget = -math.log(1.0 - 0.08)
    check("pos_clip_budget_used == gap_pos / |ln(1-clip_eps_low)|",
          close(out["pos_clip_budget_used"], gap_pos / lo_budget, 1e-6),
          f"{out.get('pos_clip_budget_used')} vs {gap_pos / lo_budget}")
    check("neg_clip_budget_used is unchanged (same flat denominator)",
          close(out["neg_clip_budget_used"], gap_neg / lo_budget, 1e-6),
          f"{out.get('neg_clip_budget_used')} vs {gap_neg / lo_budget}")
    check("the two shares are on ONE denominator "
          "(ratio == gap_pos/gap_neg)",
          close(out["pos_clip_budget_used"] / out["neg_clip_budget_used"],
                gap_pos / gap_neg, 1e-4))
    # The flat denominator must NOT follow clip_low_mse_coef.
    trainer.config = GRPOConfig(device="cpu", clip_eps_low=0.08,
                               clip_low_mse_coef=8.0,
                               jitter_pos=0.25, jitter_neg=0.05)
    train_grpo.compute_fm_log_prob = _fake
    try:
        out2 = trainer._jitter_gap_diagnostics(
            ready_backbone={"backbone_features": torch.zeros(B, 1, 1)},
            ready_state_features=torch.zeros(B, 1, 1),
            ready_embodiment_id=torch.zeros(B, dtype=torch.long),
            ready_actions=torch.zeros(B, 1, 1),
            ready_masks=torch.ones(B, 1, 1),
            ready_noise=torch.zeros(B, 1, 1),
            timesteps=torch.zeros(K, B),
            noise_for_input=torch.zeros(K, B, 1, 1),
            lam_row=torch.tensor([0.25, 0.05]),
            pos_adv_mask=torch.tensor([True, False]),
            fixed_row_mask=torch.tensor([False, False]),
            jitter_row_mask=torch.tensor([True, True]),
        )
    finally:
        train_grpo.compute_fm_log_prob = real_fm
    check("clip_low_mse_coef does NOT change either budget share "
          "(cross-run comparability)",
          close(out2["pos_clip_budget_used"], out["pos_clip_budget_used"], 1e-12)
          and close(out2["neg_clip_budget_used"],
                    out["neg_clip_budget_used"], 1e-12))


# ---------------------------------------------------------------------------
# 8. C3: lora/cos_step_* against hand-constructed weight deltas
# ---------------------------------------------------------------------------

class _RefDiT(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.lora_A = nn.Parameter(torch.tensor(w, dtype=torch.float32))


class _RefActionHead(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.model = _RefDiT(w)


class _RefModel(nn.Module):
    """Minimal model whose trainable param is named like a real LoRA param."""

    def __init__(self, w=(0.0, 0.0, 0.0)):
        super().__init__()
        self.action_head = _RefActionHead(w)

    @property
    def w(self):
        return self.action_head.model.lora_A


def _cos_trainer(**over):
    cfg = GRPOConfig(device="cpu", **over)
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.config = cfg
    t.device = torch.device("cpu")
    t.model = _RefModel()
    t._lora_init_params = {
        n: p.detach().clone() for n, p in t.model.named_parameters()
        if p.requires_grad
    }
    t._lora_prev_params = None
    t._lora_prev_step = None
    t._lora_cos_ref = None
    t._lora_cos_ref_source = None
    t._lora_cos_n_logged = 0
    t._lora_cos_ref_logged = False
    return t


def _set_w(t, vals):
    with torch.no_grad():
        t.model.w.copy_(torch.tensor(vals, dtype=torch.float32))


def _cos(u, v):
    u, v = np.array(u, dtype=np.float64), np.array(v, dtype=np.float64)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def test_lora_step_cosines():
    print("\n[lora] cos_step_prev / cumulative / early against hand-built deltas")

    t = _cos_trainer(cos_ref_iterations=2)
    # W0 = (0,0,0) is the init snapshot.
    _set_w(t, (1.0, 0.0, 0.0))                          # W1
    out1 = t._compute_lora_step_cosines()
    check("first logged iteration emits nothing (seeds the history)",
          out1 == {}, str(out1))

    _set_w(t, (1.0, 1.0, 0.0))                          # W2; step = (0,1,0)
    out2 = t._compute_lora_step_cosines()
    check("second call: no cos_step_prev yet (no predecessor step)",
          "cos_step_prev" not in out2, str(out2))
    check("second call: step_norm == ||W2 - W1||",
          close(out2["step_norm"], 1.0, 1e-6), str(out2.get("step_norm")))
    check("second call: cos_step_cumulative == cos(step, W1 - W0)",
          close(out2["cos_step_cumulative"], _cos((0, 1, 0), (1, 0, 0)), 1e-6),
          str(out2.get("cos_step_cumulative")))

    _set_w(t, (1.0, 2.0, 0.0))                          # W3; step = (0,1,0)
    out3 = t._compute_lora_step_cosines()
    check("third call: cos_step_prev == +1 for two identical steps",
          close(out3["cos_step_prev"], 1.0, 1e-6), str(out3.get("cos_step_prev")))
    check("third call: cos_step_cumulative == cos((0,1,0), (1,1,0))",
          close(out3["cos_step_cumulative"], _cos((0, 1, 0), (1, 1, 0)), 1e-6),
          str(out3.get("cos_step_cumulative")))
    check("cos_step_early absent until the reference freezes",
          "cos_step_early" not in out2 and "cos_step_early" not in out3,
          f"{out2}, {out3}")
    # cos_ref_iterations=2 -> frozen at the end of the 2nd logged iteration
    # (out3's call), against W3 - W0 = (1, 2, 0). Note the SPAN is 3 iterations of
    # motion, not 2: the FIRST logged iteration only seeds _lora_prev_params and
    # returns before the counter advances. The provenance string states both
    # The span is n_logged + 1 here, but the string deliberately reports only the
    # COUNTER: when the first logged iteration makes no weight change the span is
    # n_logged instead, and the two cases are indistinguishable at the freeze site.
    check("reference freezes after cos_ref_iterations logged iterations",
          t._lora_cos_ref is not None
          and t._lora_cos_ref_source == "frozen_after_2_logged_iters_of_run",
          str(t._lora_cos_ref_source))
    l_early = (1.0, 2.0, 0.0)
    check("... and L_early == W_now - W_init at the freeze point",
          torch.allclose(t._lora_cos_ref["action_head.model.lora_A"],
                         torch.tensor(l_early)),
          str(t._lora_cos_ref))
    check("... stored on CPU (device-resident extra stays at 2 snapshots)",
          t._lora_cos_ref["action_head.model.lora_A"].device.type == "cpu")

    _set_w(t, (1.0, 3.0, 0.0))                          # W4; step = (0,1,0)
    out4 = t._compute_lora_step_cosines()
    check("cos_step_early == cos(step, L_early)",
          close(out4["cos_step_early"], _cos((0, 1, 0), l_early), 1e-6),
          f"{out4.get('cos_step_early')} vs {_cos((0, 1, 0), l_early)}")

    # THE SIGN FLIP: reverse the step direction and every cosine must go negative.
    _set_w(t, (1.0, 2.0, 0.0))                          # W5; step = (0,-1,0)
    out5 = t._compute_lora_step_cosines()
    check("sign flip: cos_step_prev == -1",
          close(out5["cos_step_prev"], -1.0, 1e-6), str(out5.get("cos_step_prev")))
    check("sign flip: cos_step_early == -cos(forward step, L_early)",
          close(out5["cos_step_early"], -_cos((0, 1, 0), l_early), 1e-6),
          str(out5.get("cos_step_early")))
    check("sign flip: cos_step_cumulative also negative here",
          out5["cos_step_cumulative"] < 0.0,
          str(out5.get("cos_step_cumulative")))
    check("step_norm is sign-agnostic (still 1.0)",
          close(out5["step_norm"], 1.0, 1e-6), str(out5.get("step_norm")))

    # Orthogonal step -> ~0 on every reference that is not itself orthogonal.
    _set_w(t, (1.0, 2.0, 1.0))                          # step = (0,0,1)
    out6 = t._compute_lora_step_cosines()
    check("orthogonal step -> cos_step_early ~ 0",
          abs(out6["cos_step_early"]) < 1e-6, str(out6.get("cos_step_early")))
    check("orthogonal step -> cos_step_prev ~ 0",
          abs(out6["cos_step_prev"]) < 1e-6, str(out6.get("cos_step_prev")))

    # ZERO step: no curves, and the history must SURVIVE so the next real step is
    # still compared against the last real step.
    prev_step_before = t._lora_prev_step[
        "action_head.model.lora_A"].detach().clone()
    n_logged_before = t._lora_cos_n_logged
    out7 = t._compute_lora_step_cosines()               # weights unchanged
    check("zero step emits nothing at all", out7 == {}, str(out7))
    check("... and leaves step_prev untouched",
          torch.equal(t._lora_prev_step["action_head.model.lora_A"],
                      prev_step_before))
    check("... and does not advance the logged-iteration counter",
          t._lora_cos_n_logged == n_logged_before)
    _set_w(t, (1.0, 2.0, 2.0))                          # step = (0,0,1) again
    out8 = t._compute_lora_step_cosines()
    check("the next real step still sees the pre-zero predecessor (cos == +1)",
          close(out8["cos_step_prev"], 1.0, 1e-6), str(out8.get("cos_step_prev")))

    # A trainer with no init snapshot (built via __new__ without setup) degrades.
    bare = GRPOTrainer.__new__(GRPOTrainer)
    bare.config = GRPOConfig(device="cpu")
    bare.device = torch.device("cpu")
    check("no _lora_init_params -> {} instead of AttributeError",
          bare._compute_lora_step_cosines() == {})


def test_lora_cos_ref_from_paths():
    print("\n[lora] L_early loaded from two checkpoint paths")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "iter_0001").mkdir()
        (d / "iter_0005").mkdir()
        torch.save({"lora_A": torch.tensor([1.0, 0.0, 0.0])},
                   d / "iter_0001" / "lora_weights.pt")
        torch.save({"lora_A": torch.tensor([1.0, 4.0, 0.0])},
                   d / "iter_0005" / "lora_weights.pt")

        t = _cos_trainer()
        t._lora_cos_ref = t._load_cos_ref_direction(
            str(d / "iter_0001"), str(d / "iter_0005"))
        t._lora_cos_ref_source = "paths"
        l_early = (0.0, 4.0, 0.0)                       # W(b) - W(a)
        check("L_early == W(b) - W(a) with the DiT prefix applied",
              torch.allclose(t._lora_cos_ref["action_head.model.lora_A"],
                             torch.tensor(l_early)),
              str(t._lora_cos_ref))
        check("_dit_param_prefix resolves by module identity",
              t._dit_param_prefix() == "action_head.model.",
              t._dit_param_prefix())

        _set_w(t, (1.0, 0.0, 0.0))
        t._compute_lora_step_cosines()                  # seed
        _set_w(t, (1.0, 1.0, 0.0))                      # step = (0,1,0)
        out = t._compute_lora_step_cosines()
        check("cos_step_early is available on the SECOND call (no warm-up wait)",
              close(out["cos_step_early"], _cos((0, 1, 0), l_early), 1e-6),
              str(out.get("cos_step_early")))
        check("... and the own-run freeze does not overwrite an explicit ref",
              t._lora_cos_ref_source == "paths", str(t._lora_cos_ref_source))

        # A .pt file directly is accepted too.
        t2 = _cos_trainer()
        ref2 = t2._load_cos_ref_direction(
            str(d / "iter_0001" / "lora_weights.pt"),
            str(d / "iter_0005" / "lora_weights.pt"))
        check("a lora_weights.pt path works as well as its directory",
              torch.allclose(ref2["action_head.model.lora_A"],
                             torch.tensor(l_early)))

        # Hard failures.
        torch.save({"lora_A": torch.tensor([1.0, 0.0])},
                   d / "bad_shape.pt")
        torch.save({"lora_B": torch.tensor([1.0, 0.0, 0.0])},
                   d / "bad_keys.pt")
        for name, path in (("shape mismatch", "bad_shape.pt"),
                           ("key-set mismatch", "bad_keys.pt")):
            try:
                _cos_trainer()._load_cos_ref_direction(
                    str(d / path), str(d / "iter_0005" / "lora_weights.pt"))
                raised = False
            except RuntimeError:
                raised = True
            check(f"{name} hard-fails with RuntimeError", raised)
        try:
            _cos_trainer()._load_cos_ref_direction(
                str(d / "iter_0001"), str(d / "iter_0001"))
            raised = False
        except RuntimeError:
            raised = True
        check("two IDENTICAL checkpoints hard-fail (degenerate reference)", raised)

        # Config-level validation.
        try:
            GRPOConfig(cos_ref_lora_paths=(str(d / "iter_0001"),
                                           str(d / "iter_0005")))
            ok = True
        except ValueError:
            ok = False
        check("existing 2-tuple passes config validation", ok)
        for label, bad in (
            ("a non-existent path", (str(d / "nope"), str(d / "iter_0005"))),
            ("a 1-tuple", (str(d / "iter_0001"),)),
            ("a bare string", str(d / "iter_0001")),
        ):
            try:
                GRPOConfig(cos_ref_lora_paths=bad)
                raised = False
            except ValueError:
                raised = True
            check(f"cos_ref_lora_paths with {label} is rejected", raised)
        for bad_n in (0, -3):
            try:
                GRPOConfig(cos_ref_iterations=bad_n)
                raised = False
            except ValueError:
                raised = True
            check(f"cos_ref_iterations={bad_n} rejected", raised)


# ---------------------------------------------------------------------------
# 9. Startup banner and clip_killed_gradient's dual low-bound form
# ---------------------------------------------------------------------------

def test_clip_killed_gradient_accepts_both_forms():
    print("\n[predicate] clip_killed_gradient takes a float epsilon OR a tensor bound")

    lo, hi = 0.2, 0.2
    ratio = torch.tensor([0.5, 0.96, 1.5, 0.5, 0.96, 1.5])
    adv = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    surr1 = adv * ratio
    surr2 = adv * torch.clamp(ratio, 1 - lo, 1 + hi)

    by_float = clip_killed_gradient(ratio, surr1, surr2, lo, hi)
    by_tensor = clip_killed_gradient(
        ratio, surr1, surr2, torch.full_like(ratio, 1 - lo), hi)
    check("float epsilon and the equivalent tensor bound agree exactly",
          torch.equal(by_float, by_tensor),
          f"{by_float.tolist()} vs {by_tensor.tolist()}")
    check("the legacy float form still reproduces the four-case table",
          by_float.tolist() == [False, False, True, True, False, False],
          str(by_float.tolist()))

    # A PER-ROW tensor bound: row 0 and row 1 share a ratio but not a bound.
    r2 = torch.tensor([0.90, 0.90])
    a2 = torch.tensor([-1.0, -1.0])
    floor = torch.tensor([0.85, 0.95])
    s2 = a2 * torch.maximum(torch.minimum(r2, torch.full_like(r2, 1 + hi)), floor)
    dead = clip_killed_gradient(r2, a2 * r2, s2, floor, hi)
    check("per-row bound: identical ratios classify DIFFERENTLY by their own floor",
          dead.tolist() == [False, True], str(dead.tolist()))

    # Values are bitwise equal to torch.clamp, gradients differ only at ties.
    torch.manual_seed(0)
    x = torch.randn(20000) * 0.3 + 1.0
    a = torch.clamp(x, 1 - lo, 1 + hi)
    b = torch.maximum(torch.minimum(x, torch.full_like(x, 1 + hi)),
                      torch.full_like(x, 1 - lo))
    check("maximum(minimum(...)) is BITWISE equal to clamp in value",
          torch.equal(a, b))


def test_banner():
    print("\n[banner] one line per enabled mechanism, with resolved arithmetic")

    # The banner block lives inside train(); exercise it via the same print
    # arguments by re-running the resolved arithmetic the operator will read.
    coef, lo_eps = 8.0, 0.2
    ceiling = -math.log(1.0 - lo_eps)
    check("banner probe constants bracket the measured MSE_ref range",
          train_grpo.MSE_REF_BANNER_PROBES == (0.0023, 0.0297),
          str(train_grpo.MSE_REF_BANNER_PROBES))
    check("ceiling-binding threshold is |ln(1-eps)|/coef",
          close(ceiling / coef, 0.0278929, 1e-6), str(ceiling / coef))
    check("at the EARLY probe the ceiling does NOT bind at coef=8",
          coef * train_grpo.MSE_REF_BANNER_PROBES[0] < ceiling)
    check("at the LATE probe the ceiling DOES bind at coef=8",
          coef * train_grpo.MSE_REF_BANNER_PROBES[1] >= ceiling)

    # Drive the real banner by calling train() far enough to print it, without
    # running an iteration: num_iterations=0 makes the loop body unreachable, and
    # _last_updated_iteration=0 takes the final-save path's "no update ran" exit.
    def _run_banner(**over):
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.config = GRPOConfig(device="cpu", num_iterations=0, **over)
        t.device = torch.device("cpu")
        t._start_iteration = 1
        t._last_updated_iteration = 0
        t._scene_seed_pool = lambda: (1,)
        t._resolved_num_async_vector_env = lambda: t.config.group_size
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            t.train()
        return buf.getvalue()

    off = _run_banner()
    check("no per-row-clip banner line when coef == 0",
          "MSE-referenced lower clip" not in off, off)
    check("no PAWS-floor banner line when the flag is off",
          "k floored at target_ratio" not in off, off)

    on = _run_banner(clip_low_mse_coef=coef, clip_eps_low=lo_eps)
    check("banner announces the mechanism",
          "Per-row MSE-referenced lower clip: ON" in on, on)
    check("banner prints the resolved formula with the numeric ceiling",
          "budget_i = min(8 x MSE_ref_i" in on and f"{ceiling:.4f}" in on, on)
    check("banner prints the uniform inflation factor",
          f"{1.0 + coef:.2f}x" in on, on)
    check("banner prints the resolved budget at both representative MSE_refs",
          all(f"at MSE_ref={p:.4f}" in on
              for p in train_grpo.MSE_REF_BANNER_PROBES), on)
    check("banner flags which probe hits the CEILING", "(CEILING)" in on, on)

    paws = _run_banner(positive_advantage_weight_scaling=True,
                       paws_k_floor_at_target=True,
                       positive_advantage_weight_target_ratio=1.75)
    check("banner announces the PAWS k floor with the resolved interval",
          "PAWS k floored at target_ratio: ON" in paws
          and "[1.75, 10]" in paws, paws)
    inert = _run_banner(paws_k_floor_at_target=True,
                        positive_advantage_weight_target_ratio=1.75)
    check("banner says INERT when the flag is set without PAWS enabled",
          "INERT" in inert, inert)


# ---------------------------------------------------------------------------
# 10. _log_metrics emits the new families under the documented prefixes
# ---------------------------------------------------------------------------

class _RecordingWriter:
    def __init__(self):
        self.scalars = []
        self.texts = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), step))

    def add_text(self, tag, text, global_step=None):
        self.texts.append((tag, text, global_step))


def _log_probe(update_stats, lora_cosines, *, source=None, logged=False,
               **cfg_over):
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.config = GRPOConfig(device="cpu", use_wandb=False, **cfg_over)
    t.writer = _RecordingWriter()
    t._lora_cos_ref_source = source
    t._lora_cos_ref_logged = logged
    t._log_metrics(5, {"success_rate": 0.5}, update_stats, lr=1.5e-5,
                   iter_time=1.0, phase_times={"collect": 1.0},
                   lora_delta_norm=0.5, lora_cosines=lora_cosines)
    return t


def test_log_metrics_new_families():
    print("\n[logging] drift/*, lora/cos_step_* and the provenance text summary")

    drift = {
        # PRODUCTION key names, not a hand-invented family: `_emit` is generic
        # over dict keys, so a fixture using retired names makes every assertion
        # here trivially true and cannot catch a metric rename. These must stay in
        # sync with _drift_stats()'s output (asserted as a set below).
        "neg_down_p10": 0.01, "neg_down_p50": 0.03,
        "neg_down_p90": 0.09, "neg_down_max": 0.12,
        "neg_rows": 6, "neg_frac_over_budget": 0.33,
    }
    cos = {"step_norm": 0.004, "cos_step_prev": 0.81,
           "cos_step_cumulative": 0.44, "cos_step_early": -0.49}
    t = _log_probe({"n_updates": 3, "loss": 1.0, "_drift_diag": drift}, cos)
    tags = [tag for tag, _v, _s in t.writer.scalars]

    check("every drift/* key reaches TB under the drift/ prefix",
          all(f"drift/{k}" in tags for k in drift), str(sorted(tags)))
    check("every cosine reaches TB under the lora/ prefix",
          all(f"lora/{k}" in tags for k in cos), str(sorted(tags)))
    check("the pre-existing lora/weight_delta_norm is untouched",
          "lora/weight_delta_norm" in tags)
    check("the nested dict does NOT leak as a train/ scalar",
          "train/_drift_diag" not in tags,
          str([x for x in tags if "diag" in x]))
    check("no duplicate TB tags", len(tags) == len(set(tags)),
          str(sorted({x for x in tags if tags.count(x) > 1})))

    # Non-finite entries are dropped by _emit, not written.
    tn = _log_probe(
        {"n_updates": 3, "_drift_diag": {**drift, "neg_rows": float("nan")}},
        {**cos, "cos_step_early": float("inf")},
    )
    bad = [tag for tag, v, _s in tn.writer.scalars if not math.isfinite(v)]
    check("non-finite drift/lora entries are dropped, not written", not bad,
          str(bad))
    ok_tags = [tag for tag, _v, _s in tn.writer.scalars]
    check("... and the finite siblings still get through",
          "drift/neg_down_p50" in ok_tags
          and "lora/cos_step_prev" in ok_tags)

    # Ungated on n_updates: the drift family survives a discarded update.
    t0 = _log_probe({"n_updates": 0, "_drift_diag": drift}, cos)
    tags0 = [tag for tag, _v, _s in t0.writer.scalars]
    check("drift/* is NOT gated on n_updates > 0",
          all(f"drift/{k}" in tags0 for k in drift), str(sorted(tags0)))

    # Provenance text: one write per DISTINCT source, never one per iteration.
    t_none = _log_probe({"n_updates": 3}, {}, source=None)
    check("unresolved reference writes the 'none' provenance once",
          [tag for tag, _x, _s in t_none.writer.texts] == ["lora/cos_ref_source"]
          and "'none'" in t_none.writer.texts[0][1],
          str(t_none.writer.texts))
    check("... and latches on the source string",
          t_none._lora_cos_ref_logged == "none",
          str(t_none._lora_cos_ref_logged))
    t_again = _log_probe({"n_updates": 3}, cos, source=None, logged="none")
    check("the same source is not re-written on later iterations",
          t_again.writer.texts == [], str(t_again.writer.texts))
    t_frozen = _log_probe({"n_updates": 3}, cos, source="frozen_after_2_logged_iters_of_run",
                          logged="none")
    check("a newly resolved source supersedes the 'none' note (one more write)",
          len(t_frozen.writer.texts) == 1
          and "frozen_after_2_logged_iters_of_run" in t_frozen.writer.texts[0][1]
          and t_frozen.writer.texts[0][2] == 0,
          str(t_frozen.writer.texts))
    check("... and the frozen note states the blind spot it carries",
          "already turned" in t_frozen.writer.texts[0][1],
          str(t_frozen.writer.texts))

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "a").mkdir()
        (d / "b").mkdir()
        for sub in ("a", "b"):
            torch.save({"lora_A": torch.zeros(2)}, d / sub / "lora_weights.pt")
        t_paths = _log_probe({"n_updates": 3}, cos, source="paths",
                             cos_ref_lora_paths=(str(d / "a"), str(d / "b")))
        check("the 'paths' provenance names both checkpoints",
              str(d / "a") in t_paths.writer.texts[0][1]
              and str(d / "b") in t_paths.writer.texts[0][1],
              str(t_paths.writer.texts))

    # A caller that does not use the diagnostic at all must not touch add_text —
    # several existing harnesses pass a writer stub with add_scalar only.
    t_absent = _log_probe({"n_updates": 3}, None, source=None)
    check("lora_cosines=None writes no provenance text at all",
          t_absent.writer.texts == [], str(t_absent.writer.texts))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regression tests for defects an audit found UNCAUGHT by this file. Each one
# corresponds to a mutation the suite previously passed with.
# ---------------------------------------------------------------------------

def test_snapup_at_a_nonexact_eps():
    """The snap-up must bite. Requires an eps where the fp32 round-trip is NOT exact.

    Mutation "delete `torch.maximum(..., flat)`" previously survived because the only
    per-row-floor test pinned clip_eps_low=0.2, one of the 745/999 eps values where
    `exp(-(-log(1-eps)))` already round-trips exactly, so the interesting case never
    occurred.
    """
    print("\n[snap] the snap-up is exercised at an eps where exp() does NOT round-trip")
    lo_eps = 0.067                      # audited: exp round-trip lands 1 ULP BELOW
    flat32 = float(torch.tensor(1.0 - lo_eps, dtype=torch.float32))
    naive = float(torch.exp(torch.tensor(
        math.log(1.0 - lo_eps), dtype=torch.float32)))
    check("the chosen eps really is a non-exact round-trip (guards the guard)",
          naive < flat32, f"naive={naive!r} flat={flat32!r}")

    coef = 8.0
    ceiling = -math.log(1.0 - lo_eps)
    big = ceiling / coef * 4.0           # deep in the ceiling-pinned regime
    r = run_rows([_Row(advantage=-1.0, mse_ref=big, log_ratio=-0.001),
                  _Row(advantage=+1.0, mse_ref=big, log_ratio=-0.001)],
                 clip_low_mse_coef=coef, clip_eps_low=lo_eps)
    floors = [float(x) for x in r.spy_calls[0][3]]
    check("a ceiling-pinned floor is EXACTLY the flat floor, not one ULP below",
          all(f == flat32 for f in floors),
          f"{[float(x) for x in floors]} vs flat32 {flat32!r}")
    check("... and without the snap-up it would have been strictly looser",
          naive < flat32 and all(f >= flat32 for f in floors))


def test_clamp_gradient_at_a_tie():
    """The PRODUCTION surrogate must be gradient-identical to scalar clamp at a tie.

    Mutation "revert to maximum(minimum(...))" previously survived because the only
    assertion compared the two forms by VALUE (they are bitwise equal there) and a
    first attempt at this test compared local lambdas rather than the production path
    — so reverting the surrogate changed nothing the test could see.

    This drives the real `_grpo_update_inner` twice with a negative-advantage row
    whose ratio is (a) BITWISE equal to its own `rho_floor` and (b) exactly ONE fp32
    ULP above it. The legitimate gradient difference between those two batches is
    O(1 ULP), so `grad_norm_mean` must agree to ~1e-5 relative. Under
    `maximum(minimum(...))` the at-tie row's surrogate gradient is 0.75x (the outer
    `torch.min` does NOT compensate), so the two disagree by ~20-25%.
    """
    print("\n[tie] production surrogate: gradient at an exact bound == just inside")
    coef, mse, lo_eps = 8.0, 0.01, 0.2

    def ulp_up(x: float) -> float:
        b = struct.unpack(">I", struct.pack(">f", x))[0] + 1
        return struct.unpack(">f", struct.pack(">I", b))[0]

    # Exact tie: log_ratio == -(coef * mse_ref) in fp32 makes ratio and rho_floor
    # the exp() of the same fp32 input, hence bitwise equal.
    bud = float(torch.tensor(coef, dtype=torch.float32)
                * torch.tensor(mse, dtype=torch.float32))
    tie_lr = -bud
    r_tie = run_rows([_Row(advantage=-1.0, mse_ref=mse, log_ratio=tie_lr),
                      _Row(advantage=+1.0, mse_ref=mse, log_ratio=-0.001)],
                     clip_low_mse_coef=coef, clip_eps_low=lo_eps)
    ratio_t, _, _, floor_t, _ = r_tie.spy_calls[0]
    check("the tie is genuinely BITWISE (guards the guard)",
          ratio_t[0].item() == floor_t[0].item(),
          f"{ratio_t[0].item()!r} vs {floor_t[0].item()!r}")

    # One ULP ABOVE the floor: step log_ratio up in fp32 ULPs until ratio moves off
    # the bound by exactly one ULP. Inside the band, so the clip cannot touch it.
    target = ulp_up(floor_t[0].item())
    near_lr, found = None, False
    for k in range(1, 64):
        b = struct.unpack(">I", struct.pack(">f", tie_lr))[0] - k   # toward zero
        cand = struct.unpack(">f", struct.pack(">I", b))[0]
        if float(torch.exp(torch.tensor(cand, dtype=torch.float32))) == target:
            near_lr, found = cand, True
            break
    check("found a log_ratio landing exactly ONE ULP above the floor",
          found, f"searched 63 ULPs from {tie_lr!r}")
    if not found:
        return
    r_near = run_rows([_Row(advantage=-1.0, mse_ref=mse, log_ratio=near_lr),
                       _Row(advantage=+1.0, mse_ref=mse, log_ratio=-0.001)],
                      clip_low_mse_coef=coef, clip_eps_low=lo_eps)
    ratio_n, _, _, floor_n, _ = r_near.spy_calls[0]
    check("... and that row is strictly INSIDE the band (not clipped)",
          ratio_n[0].item() > floor_n[0].item(),
          f"{ratio_n[0].item()!r} vs {floor_n[0].item()!r}")

    g_tie = r_tie.result["grad_norm_mean"]
    g_near = r_near.result["grad_norm_mean"]
    rel = abs(g_tie - g_near) / max(g_near, 1e-30)
    check("grad_norm_mean at the tie == one ULP inside (clamp semantics)",
          rel < 1e-5, f"tie={g_tie!r} near={g_near!r} rel={rel:.3e} "
                      f"(maximum/minimum would give ~0.20-0.25)")


def test_drift_drops_nonfinite_rows():
    """A non-finite row must never produce a NaN percentile.

    Honest about its reach: the finalizer's `isfinite` filter is a BACKSTOP that
    appears unreachable in practice. A non-finite `-log(rho_floor)` needs
    `clip_eps_low >= 1.0` (rejected at config time), and a non-finite `log_ratio`
    NaNs `kl_per_row_last_iter` and hence the loss even at `kl_coef_last_iter == 0`
    (`0.0 * nan == nan`), so the micro-batch is dropped upstream. What this test
    pins is the OBSERVABLE contract — no NaN ever reaches a curve — not the filter.
    """
    print("\n[drift] non-finite rows are dropped from the pooled distribution")
    ok = run_rows([_Row(advantage=-1.0, mse_ref=0.01, log_ratio=-0.05),
                   _Row(advantage=-1.0, mse_ref=0.01, log_ratio=-0.06),
                   _Row(advantage=+1.0, mse_ref=0.01, log_ratio=-0.01)],
                  clip_low_mse_coef=8.0, clip_eps_low=0.2)
    check("baseline: both negative rows counted", ok.result["_drift_diag"]["neg_rows"] == 2,
          str(ok.result["_drift_diag"]["neg_rows"]))
    # mse_ref = -inf -> MSE_ref = +inf is clamped to the ceiling, so the FLOOR stays
    # finite; a NaN ref_log_prob is what produces a non-finite floor, and that also
    # NaNs the loss so the micro-batch is dropped upstream. Assert the observable:
    # the family is absent rather than containing a NaN.
    nan = run_rows([_Row(advantage=-1.0, mse_ref=float("nan"), log_ratio=-0.05),
                    _Row(advantage=+1.0, mse_ref=0.01, log_ratio=-0.01)],
                   mb_size=1, clip_low_mse_coef=8.0, clip_eps_low=0.2)
    dd = nan.result.get("_drift_diag")
    check("a NaN MSE_ref never yields a NaN percentile",
          dd is None or all(v == v for v in dd.values()), str(dd))


def test_paws_k_binding_fractions():
    """floor / cap binding fractions, asserted by VALUE against a hand count.

    An earlier version checked only key presence and range [0, 1]. Eight plausible
    mutations survived that, including SWAPPING the two metric names and forcing the
    comparison floor to ignore `paws_k_floor_at_target` — i.e. the one question the
    metric exists to answer ("is this the same floor the loss used?").
    """
    print("\n[paws] k floor/cap binding fractions, by value")

    # Independent re-derivation of what the production counters must produce, from
    # the same prefix-pool recurrence the loss uses.
    def expected(masses, tratio, floor_on, cap):
        floor = tratio if floor_on else 1.0
        N = D = 0.0
        n_meas = fl = cp = 0
        for n_mass, d_mass in masses:
            if D > 0.0:
                raw = tratio * N / (D + 1e-8)
                n_meas += 1
                if raw < floor:
                    fl += 1
                elif raw > cap:
                    cp += 1
            N += n_mass
            D += d_mass
        return (fl / n_meas, cp / n_meas, n_meas) if n_meas else (None, None, 0)

    rows = [_Row(advantage=-1.0, mse_ref=0.01, log_ratio=-0.02, group_id=0),
            _Row(advantage=+1.0, mse_ref=0.01, log_ratio=-0.01, group_id=0),
            _Row(advantage=-1.0, mse_ref=0.01, log_ratio=-0.02, group_id=1),
            _Row(advantage=+1.0, mse_ref=0.01, log_ratio=-0.01, group_id=1)]
    common = dict(mb_size=2, epochs=2, positive_advantage_weight_scaling=True,
                  positive_advantage_weight_max=5.0,
                  per_iteration_advantage_norm=True)

    for floor_on in (True, False):
        r = run_rows(rows, positive_advantage_weight_target_ratio=2.0,
                     paws_k_floor_at_target=floor_on, **common)
        res = r.result
        tag = "ON" if floor_on else "OFF"
        check(f"flag {tag}: both binding fractions emitted",
              "pos_adv_k_floor_binds_frac" in res
              and "pos_adv_k_cap_binds_frac" in res,
              str(sorted(k for k in res if k.startswith("pos_adv_k"))))
        # The counters' denominator is the number of MEASURED micro-batches, which
        # is n_micro_batches - 1 (the first has an empty prefix pool, D_iter == 0).
        n_meas = res["n_micro_batches"] - 1
        f_frac = res["pos_adv_k_floor_binds_frac"]
        c_frac = res["pos_adv_k_cap_binds_frac"]
        check(f"flag {tag}: each fraction is a whole multiple of 1/n_measured",
              n_meas > 0
              and abs(f_frac * n_meas - round(f_frac * n_meas)) < 1e-9
              and abs(c_frac * n_meas - round(c_frac * n_meas)) < 1e-9,
              f"n_measured={n_meas} floor={f_frac} cap={c_frac}")
        check(f"flag {tag}: fractions cannot sum above 1 (floor < cap enforced)",
              f_frac + c_frac <= 1.0 + 1e-12, f"{f_frac} + {c_frac}")
        # THE discriminating assertion: with tratio=2.0 > 1.0 the floor binds on
        # strictly MORE micro-batches when the flag is on than when it is off, and
        # the cap (5.0) is never reached here. A mutant that hardwires the
        # comparison floor to 1.0, or to tratio, or swaps the two curves, breaks
        # one of these.
        check(f"flag {tag}: cap never binds at max=5.0", c_frac == 0.0, str(c_frac))
    on = run_rows(rows, positive_advantage_weight_target_ratio=2.0,
                  paws_k_floor_at_target=True, **common).result
    off = run_rows(rows, positive_advantage_weight_target_ratio=2.0,
                   paws_k_floor_at_target=False, **common).result
    check("floor binds STRICTLY more often with the flag on (floor 2.0 vs 1.0)",
          on["pos_adv_k_floor_binds_frac"] > off["pos_adv_k_floor_binds_frac"],
          f"on={on['pos_adv_k_floor_binds_frac']} "
          f"off={off['pos_adv_k_floor_binds_frac']}")
    check("... and with the flag ON the floor is `tratio`, so it binds everywhere "
          "the measurement is below 2.0",
          on["pos_adv_k_floor_binds_frac"] == 1.0,
          str(on["pos_adv_k_floor_binds_frac"]))
    check("the retired k_raw_min is gone (it could never surface the cap)",
          "pos_adv_weight_k_raw_min" not in on, str(sorted(on)))
    # A cap-binding configuration. Measured N/D ~= 0.99 in this harness, so
    # `k_raw = tratio * N/D` clears the cap only when tratio > max — reachable only
    # with the flag OFF, since validation forbids tratio >= max when it is on.
    capped = run_rows(rows, positive_advantage_weight_target_ratio=10.0,
                      paws_k_floor_at_target=False, mb_size=2, epochs=2,
                      positive_advantage_weight_scaling=True,
                      positive_advantage_weight_max=1.5,
                      per_iteration_advantage_norm=True).result
    check("the CAP fraction can be non-zero (it is not hardwired to 0)",
          capped["pos_adv_k_cap_binds_frac"] > 0.0,
          str(capped["pos_adv_k_cap_binds_frac"]))
    check("... and the floor fraction is then 0 (elif: they are exclusive)",
          capped["pos_adv_k_floor_binds_frac"] == 0.0,
          str(capped["pos_adv_k_floor_binds_frac"]))


def test_born_dead_is_prestep_scoped_not_pooled():
    """The born-dead FRACTION must come from pre-step rows, not the pooled set.

    Previously only `neg_born_rows` (the count) was pinned, so a mutant computing
    the fraction over ALL micro-batches — exactly the post-step contamination the
    pre-step scoping exists to prevent — passed the whole suite.

    Construction: two negative rows, one micro-batch each, at
    `gradient_accumulation_steps=1` so only micro-batch 0 is pre-step. The pre-step
    row is INSIDE its budget; the post-step row is far outside it. So the pre-step
    fraction must be 0.0 while the pooled fraction is 0.5 — the two are
    distinguishable, which is what the earlier scenario could not do.
    """
    print("\n[drift] born-dead fraction is pre-step scoped, not pooled")
    coef, mse, lo_eps = 8.0, 0.01, 0.2
    budget = coef * mse                      # 0.08 nats
    r = run_rows([_Row(advantage=-1.0, mse_ref=mse, log_ratio=-0.001, group_id=0),
                  _Row(advantage=-1.0, mse_ref=mse, log_ratio=-0.500, group_id=1)],
                 mb_size=1, epochs=1, gradient_accumulation_steps=1,
                 clip_low_mse_coef=coef, clip_eps_low=lo_eps)
    d = r.result["_drift_diag"]
    check("precondition: exactly one micro-batch is pre-step",
          d["neg_born_rows"] == 1, str(d.get("neg_born_rows")))
    check("precondition: the pooled set holds both rows",
          d["neg_rows"] == 2, str(d.get("neg_rows")))
    check("pooled frac_over_budget sees the far-outside row (0.5)",
          close(d["neg_frac_over_budget"], 0.5, 1e-6),
          str(d["neg_frac_over_budget"]))
    check("born-dead frac is 0.0 — the pre-step row is INSIDE its budget",
          d["neg_frac_born_dead"] == 0.0,
          f"{d['neg_frac_born_dead']} (would be 0.5 if computed over the pool)")
    check("... and the two genuinely differ, so the assertion discriminates",
          d["neg_frac_born_dead"] != d["neg_frac_over_budget"],
          f"born={d['neg_frac_born_dead']} pooled={d['neg_frac_over_budget']}")
    # Mirror image: pre-step row OUTSIDE, post-step row inside -> born 1.0, pooled 0.5.
    #
    # `kl_coef_last_iter > 0` is REQUIRED here, and the reason is worth recording: a
    # negative row past its budget is clip-DEAD by construction, so with both KL
    # coefficients at 0 it contributes no gradient at all, the accumulated gradient
    # is zero, `_apply_accumulated_grads` drops the step (n_zero_grad_steps=1), and
    # theta therefore stays theta_ref — making micro-batch 1 legitimately pre-step
    # too (born_rows=2). That is CORRECT behaviour, not a defect; the KL term simply
    # gives the step something to move on so the post-step case is reachable.
    r2 = run_rows([_Row(advantage=-1.0, mse_ref=mse, log_ratio=-0.500, group_id=0),
                   _Row(advantage=-1.0, mse_ref=mse, log_ratio=-0.001, group_id=1)],
                  mb_size=1, epochs=1, gradient_accumulation_steps=1,
                  clip_low_mse_coef=coef, clip_eps_low=lo_eps,
                  kl_coef_last_iter=0.1)
    d2 = r2.result["_drift_diag"]
    check("mirror precondition: a step really fired, so mb1 is post-step",
          d2["neg_born_rows"] == 1, str(d2.get("neg_born_rows")))
    check("mirror: born-dead frac is 1.0 when the PRE-STEP row is outside",
          d2["neg_frac_born_dead"] == 1.0
          and close(d2["neg_frac_over_budget"], 0.5, 1e-6),
          f"born={d2['neg_frac_born_dead']} pooled={d2['neg_frac_over_budget']} "
          f"budget={budget}")
    # And the zero-gradient variant, which documents the correct opposite behaviour.
    r3 = run_rows([_Row(advantage=-1.0, mse_ref=mse, log_ratio=-0.500, group_id=0),
                   _Row(advantage=-1.0, mse_ref=mse, log_ratio=-0.001, group_id=1)],
                  mb_size=1, epochs=1, gradient_accumulation_steps=1,
                  clip_low_mse_coef=coef, clip_eps_low=lo_eps)
    check("no step fired (clip-dead row, no KL) -> EVERY micro-batch is pre-step",
          r3.result["_drift_diag"]["neg_born_rows"] == 2,
          str(r3.result["_drift_diag"].get("neg_born_rows")))


def test_paws_floor_vs_cap_validation():
    """target_ratio >= max collapses k to a constant; the config must refuse."""
    print("\n[cfg] paws_k_floor_at_target requires target_ratio < max")
    def mk(**kw):
        try:
            _cfg(**kw); return None
        except ValueError as e:
            return str(e)
    check("target ABOVE the cap is rejected",
          mk(paws_k_floor_at_target=True,
             positive_advantage_weight_target_ratio=15.0,
             positive_advantage_weight_max=10.0) is not None)
    check("target EQUAL to the cap is also rejected (same collapse)",
          mk(paws_k_floor_at_target=True,
             positive_advantage_weight_target_ratio=10.0,
             positive_advantage_weight_max=10.0) is not None)
    check("target BELOW the cap is accepted",
          mk(paws_k_floor_at_target=True,
             positive_advantage_weight_target_ratio=2.25,
             positive_advantage_weight_max=5.0) is None)
    check("with the flag OFF the pairing is unconstrained (floor is 1.0)",
          mk(paws_k_floor_at_target=False,
             positive_advantage_weight_target_ratio=15.0,
             positive_advantage_weight_max=10.0) is None)


def test_drift_pooling_multiplicity():
    """neg_rows scales with update_epochs; resetting per epoch must be visible."""
    print("\n[drift] pooled row count scales with update_epochs")
    rows = [_Row(advantage=-1.0, mse_ref=0.01, log_ratio=-0.05),
            _Row(advantage=-1.0, mse_ref=0.01, log_ratio=-0.06),
            _Row(advantage=+1.0, mse_ref=0.01, log_ratio=-0.01)]
    seen = {}
    for ep in (1, 2, 3):
        r = run_rows(rows, epochs=ep, clip_low_mse_coef=8.0, clip_eps_low=0.2)
        seen[ep] = r.result["_drift_diag"]["neg_rows"]
    check("neg_rows == 2 negatives x update_epochs",
          seen == {1: 2, 2: 4, 3: 6}, str(seen))


def test_born_dead_admits_prestep_microbatches_at_accum():
    """At gradient_accumulation_steps=k, micro-batches 0..k-1 are all pre-step."""
    print("\n[drift] born-dead pools every PRE-STEP micro-batch at accum > 1")
    rows = [_Row(advantage=-1.0, mse_ref=0.01, log_ratio=-0.05, group_id=i % 2)
            for i in range(4)]
    rows += [_Row(advantage=+1.0, mse_ref=0.01, log_ratio=-0.01, group_id=i % 2)
             for i in range(2)]
    one = run_rows(rows, mb_size=1, epochs=1, clip_low_mse_coef=8.0,
                   clip_eps_low=0.2, gradient_accumulation_steps=1)
    many = run_rows(rows, mb_size=1, epochs=1, clip_low_mse_coef=8.0,
                    clip_eps_low=0.2, gradient_accumulation_steps=4)
    b1 = one.result["_drift_diag"].get("neg_born_rows", 0)
    b4 = many.result["_drift_diag"].get("neg_born_rows", 0)
    check("accum=1 captures only micro-batch 0", b1 <= 1, str(b1))
    check("accum=4 captures strictly more pre-step rows", b4 > b1, f"{b4} vs {b1}")


def test_ceiling_uses_clip_eps_low_not_high():
    """The budget ceiling must come from clip_eps_low, never clip_eps_high.

    Every other per-row-floor test pins `clip_eps_low == clip_eps_high` (0.2/0.2),
    so swapping the two survived them all — and when `clip_eps_high > clip_eps_low`
    the snap-up masks the swap anyway (both land on the flat floor). Detecting it
    requires `clip_eps_low > clip_eps_high` with a budget BETWEEN the two ceilings,
    which is what this builds. `test_jitter_metrics.py` forces `lo != hi` for the
    same reason.
    """
    print("\n[ceil] the budget ceiling is |ln(1-clip_eps_low)|, not clip_eps_high")
    lo_eps, hi_eps, coef = 0.2, 0.05, 8.0
    ceil_lo = -math.log(1.0 - lo_eps)          # 0.2231 -- correct
    ceil_hi = -math.log(1.0 - hi_eps)          # 0.0513 -- the mutant's
    mse = 0.0125                                # coef*mse = 0.10, strictly between
    check("precondition: the budget lies BETWEEN the two candidate ceilings",
          ceil_hi < coef * mse < ceil_lo,
          f"{ceil_hi:.4f} < {coef*mse:.4f} < {ceil_lo:.4f}")
    r = run_rows([_Row(advantage=-1.0, mse_ref=mse, log_ratio=-0.001),
                  _Row(advantage=+1.0, mse_ref=mse, log_ratio=-0.001)],
                 clip_low_mse_coef=coef, clip_eps_low=lo_eps, clip_eps_high=hi_eps)
    got = float(r.spy_calls[0][3][0])
    want = expected_floor(coef, mse, lo_eps)
    wrong = max(math.exp(-min(coef * mse, ceil_hi)), 1.0 - lo_eps)
    check("floor matches the clip_eps_low ceiling",
          close(got, want, 1e-7), f"got={got!r} want={want!r}")
    check("... and is NOT what the clip_eps_high ceiling would give",
          not close(got, wrong, 1e-7), f"got={got!r} wrong={wrong!r}")

if __name__ == "__main__":
    test_off_switch_determinism_and_additivity()
    test_snapup_at_a_nonexact_eps()
    test_ceiling_uses_clip_eps_low_not_high()
    test_clamp_gradient_at_a_tie()
    test_drift_drops_nonfinite_rows()
    test_paws_k_binding_fractions()
    test_born_dead_is_prestep_scoped_not_pooled()
    test_paws_floor_vs_cap_validation()
    test_drift_pooling_multiplicity()
    test_born_dead_admits_prestep_microbatches_at_accum()
    test_per_row_floor_arithmetic()
    test_five_consumers_agree()
    test_positive_rows_survive_a_tight_floor()
    test_paws_k_floor()
    test_monotone_in_coefficient()
    test_drift_diagnostics_values()
    test_pos_clip_budget_used()
    test_lora_step_cosines()
    test_lora_cos_ref_from_paths()
    test_clip_killed_gradient_accepts_both_forms()
    test_banner()
    test_log_metrics_new_families()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\033[32mAll clip-floor tests passed.\033[0m")
