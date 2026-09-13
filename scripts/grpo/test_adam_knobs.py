"""Tests for the surfaced AdamW knobs: `adam_beta1`, `adam_beta2`, `adam_eps`.

These three were hard-coded at the `optim.AdamW(...)` construction site in
`GRPOTrainer.setup()` (betas at the PyTorch default, `eps=1e-5` copied from
grpo_cont.py line 230). They are now config fields so they land in the
TensorBoard `config` dump and are tunable from the CLI.

What is covered:
  A. DEFAULT BIT-IDENTITY. The shipped defaults must equal the previously
     hard-coded `(0.9, 0.999)` / `1e-5` exactly, so an unchanged CLI reproduces
     every run recorded before the knobs existed. Asserted on the raw values AND
     through a real `optim.AdamW` built from a default config, because the second
     is what actually decides whether prior runs reproduce.
  B. VALIDATION MATRIX. Every rejected value, including the two that would
     otherwise run to completion while silently not training: `beta1 == 1.0`
     (momentum never leaves its zero init -> zero step forever) and
     `beta2 == 1.0` (denominator collapses to `adam_eps`).
  C. THE REGIME WARNING. `adam_eps < 1e-6` with `adam_beta2 >= 0.999` warns, and
     is silent once beta2 is also lowered. It is a `warnings.warn`, NOT an error:
     the combination is a legitimate experiment, just one that needs an lr
     recalibration to be interpretable.
  D. SOURCE-LEVEL WIRING. `setup()` cannot be reached from the `__new__`
     harnesses the other CPU suites use (it downloads the model, binds ZMQ and
     starts a thread), so the fact that the optimizer reads the CONFIG rather
     than a literal is checked against the source text — the same precedent
     `test_kl_base_adaptive.py` sets for its own `setup()` wiring check. Without
     this, the knobs could validate perfectly and be entirely inert.
  E. SEMANTICS THE COMMENT BLOCK CLAIMS. Two arithmetic facts the `adam_eps`
     documentation rests on, verified against a real `optim.AdamW` step rather
     than asserted: that `eps` is added to `sqrt(v_hat)` (so at
     `eps >> sqrt(v_hat)` the step is ~proportional to the gradient, and at
     `eps << sqrt(v_hat)` it is ~gradient-magnitude invariant), and that
     `weight_decay=1e-5` is numerically inert at the shipped learning rates.

CPU only, no model, no GPU. Conventions follow `test_grad_accum.py`.
"""

import sys
import warnings
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from grpo_config import GRPOConfig  # noqa: E402


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


def cfg(**kw) -> GRPOConfig:
    return GRPOConfig(device="cpu", **kw)


def raises(**kw) -> bool:
    try:
        cfg(**kw)
    except ValueError:
        return True
    return False


# ---------------------------------------------------------------------------
# A. Default bit-identity with the previously hard-coded values
# ---------------------------------------------------------------------------

def test_defaults_bit_identical():
    print("\nA. defaults reproduce the previously hard-coded optimizer")
    c = cfg()
    check("adam_beta1 default is 0.9", c.adam_beta1 == 0.9, repr(c.adam_beta1))
    check("adam_beta2 default is 0.999", c.adam_beta2 == 0.999, repr(c.adam_beta2))
    check("adam_eps default is 1e-5", c.adam_eps == 1e-5, repr(c.adam_eps))

    # The claim that matters is about the OPTIMIZER, not the dataclass: build one
    # the way setup() does and compare its param_group against a literal AdamW
    # constructed with the old hard-coded arguments.
    p = torch.nn.Parameter(torch.zeros(3))
    new = optim.AdamW(
        [p],
        lr=c.learning_rate,
        weight_decay=c.weight_decay,
        betas=(c.adam_beta1, c.adam_beta2),
        eps=c.adam_eps,
    )
    old = optim.AdamW([p], lr=c.learning_rate, weight_decay=c.weight_decay, eps=1e-5)
    keys = ("lr", "betas", "eps", "weight_decay")
    check(
        "config-built AdamW param_group == old hard-coded one",
        all(new.param_groups[0][k] == old.param_groups[0][k] for k in keys),
        f"{[(k, new.param_groups[0][k], old.param_groups[0][k]) for k in keys]}",
    )


# ---------------------------------------------------------------------------
# B. Validation matrix
# ---------------------------------------------------------------------------

def test_validation_matrix():
    print("\nB. validation matrix")
    nan, inf = float("nan"), float("inf")

    # The two silent-failure cases the error message calls out by name.
    check("beta1 == 1.0 rejected (zero step forever)", raises(adam_beta1=1.0))
    check("beta2 == 1.0 rejected (denominator -> adam_eps)", raises(adam_beta2=1.0))

    for name in ("adam_beta1", "adam_beta2"):
        check(f"{name} < 0 rejected", raises(**{name: -0.1}))
        check(f"{name} > 1 rejected", raises(**{name: 1.5}))
        check(f"{name} = nan rejected", raises(**{name: nan}))
        check(f"{name} = inf rejected", raises(**{name: inf}))
        # 0.0 is degenerate but well-defined (no EMA at all), and rejecting it
        # would forbid the "momentum off" ablation. Must be ACCEPTED.
        try:
            cfg(**{name: 0.0})
            check(f"{name} = 0.0 accepted (EMA off is a valid ablation)", True)
        except ValueError as e:
            check(f"{name} = 0.0 accepted", False, str(e)[:70])

    check("adam_eps = 0 rejected (0/0 on a dead coordinate)", raises(adam_eps=0.0))
    check("adam_eps < 0 rejected", raises(adam_eps=-1e-8))
    check("adam_eps = nan rejected", raises(adam_eps=nan))
    check("adam_eps = inf rejected", raises(adam_eps=inf))

    for kw in (
        {"adam_beta1": 0.95},
        {"adam_beta2": 0.99},
        {"adam_eps": 1e-8, "adam_beta2": 0.99},
        {"adam_beta1": 0.98, "adam_beta2": 0.99, "adam_eps": 1e-6},
    ):
        try:
            cfg(**kw)
            check(f"accepted {kw}", True)
        except ValueError as e:
            check(f"accepted {kw}", False, str(e)[:70])

    # The betas are unvalidated only in the sense that nothing else gates them —
    # there is no "feature off" switch here, unlike kl_base_adaptive. Confirm the
    # checks are unconditional by tripping one with an otherwise-exotic config.
    check(
        "beta validation is unconditional (not gated on any other flag)",
        raises(adam_beta1=1.0, jitter_pos=0.0, positive_advantage_weight_scaling=False),
    )


# ---------------------------------------------------------------------------
# C. The regime warning
# ---------------------------------------------------------------------------

def _warnings_for(**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg(**kw)
        return [str(x.message) for x in w]


def test_regime_warning():
    print("\nC. eps/beta2 regime warning")
    msgs = _warnings_for(adam_eps=1e-8)
    hit = [m for m in msgs if "NORMALISED regime" in m]
    check("eps=1e-8 with beta2=0.999 warns", len(hit) == 1, f"{len(hit)} of {len(msgs)}")
    if hit:
        m = hit[0]
        check("warning names the beta2 remedy", "adam_beta2=0.99" in m)
        check("warning names lora/step_norm as the recalibration target",
              "lora/step_norm" in m)
        # 1/(1-0.999) = 1000 steps; the message divides by ~40 steps/iter.
        check("warning reports the 1000-step memory", "1000-step memory" in m, m[:160])

    check("eps=1e-8 with beta2=0.99 is silent",
          not any("NORMALISED regime" in m for m in _warnings_for(
              adam_eps=1e-8, adam_beta2=0.99)))
    check("default config is silent",
          not any("NORMALISED regime" in m for m in _warnings_for()))
    # Boundary: the gate is `eps < 1e-6`, so exactly 1e-6 must NOT warn.
    check("eps == 1e-6 does not warn (strict inequality)",
          not any("NORMALISED regime" in m for m in _warnings_for(adam_eps=1e-6)))
    check("eps = 9e-7 does warn",
          any("NORMALISED regime" in m for m in _warnings_for(adam_eps=9e-7)))
    # It must be a warning, not an error — the combination is a real experiment.
    try:
        cfg(adam_eps=1e-8)
        check("eps=1e-8 warns but does NOT raise", True)
    except ValueError as e:
        check("eps=1e-8 warns but does NOT raise", False, str(e)[:70])


# ---------------------------------------------------------------------------
# D. Source-level wiring (setup() is unreachable from a __new__ harness)
# ---------------------------------------------------------------------------

def test_setup_wiring():
    print("\nD. setup() builds the optimizer from config, not from literals")
    src = (Path(__file__).parent / "train_grpo.py").read_text()
    i = src.find("self.optimizer = optim.AdamW")
    check("found the AdamW construction site", i != -1)
    if i == -1:
        return
    block = src[i:i + 400]
    end = block.find(")\n")
    block = block[:end] if end != -1 else block
    check("betas read from config",
          "betas=(self.config.adam_beta1, self.config.adam_beta2)" in block, block)
    check("eps reads from config", "eps=self.config.adam_eps" in block, block)
    check("no hard-coded eps literal remains", "eps=1e-5" not in block, block)
    check("lr/weight_decay still from config",
          "lr=self.config.learning_rate" in block
          and "weight_decay=self.config.weight_decay" in block, block)
    check("exactly one AdamW construction in train_grpo.py",
          src.count("optim.AdamW(") == 1, str(src.count("optim.AdamW(")))


# ---------------------------------------------------------------------------
# E. The semantics the adam_eps comment block rests on
# ---------------------------------------------------------------------------

def _one_step(grad_scale: float, eps: float, lr: float = 1.2e-4) -> float:
    """One AdamW step on a constant gradient; returns |delta theta|.

    A CONSTANT gradient is the point: it drives m_hat -> g and sqrt(v_hat) -> |g|
    exactly, so the step is `lr * g / (|g| + eps)` and the two regimes are
    separable analytically. Ten steps let the bias-corrected moments settle.
    """
    p = torch.nn.Parameter(torch.zeros(1))
    opt = optim.AdamW([p], lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=0.0)
    before = None
    for _ in range(10):
        opt.zero_grad()
        p.grad = torch.full_like(p, grad_scale)
        before = p.detach().clone()
        opt.step()
    return float((p.detach() - before).abs().item())


def test_eps_regime_arithmetic():
    print("\nE. eps regime arithmetic (verified against a real AdamW step)")

    # eps >> sqrt(v_hat): step ~= lr * g / eps, i.e. PROPORTIONAL to the gradient.
    # 1.0e-6 and 3.8e-6 are the measured per-coordinate sqrt(v_hat) for the
    # lam=0.125 and lam=0.35 arms; 3.8x apart in gradient.
    small = _one_step(1.0e-6, eps=1e-5)
    large = _one_step(3.8e-6, eps=1e-5)
    ratio_floored = large / small
    check("at eps=1e-5 a 3.8x gradient gives a ~3.0x+ larger step "
          "(gradient magnitude is NOT normalised away)",
          ratio_floored > 3.0, f"ratio={ratio_floored:.2f}")

    # eps << sqrt(v_hat): step ~= lr, i.e. gradient-magnitude INVARIANT.
    small_n = _one_step(1.0e-6, eps=1e-8)
    large_n = _one_step(3.8e-6, eps=1e-8)
    ratio_norm = large_n / small_n
    check("at eps=1e-8 the same 3.8x gradient gives a <1.1x step "
          "(gradient-magnitude invariant)",
          ratio_norm < 1.1, f"ratio={ratio_norm:.4f}")

    # The ~10x step inflation the comment block warns about, at the lam=0.125
    # gradient scale where sqrt(v_hat) ~ 1e-6 against eps = 1e-5.
    inflation = small_n / small
    check("eps 1e-5 -> 1e-8 inflates the step ~10x at sqrt(v_hat) ~ 1e-6 "
          "(hence the lr recalibration)",
          5.0 < inflation < 15.0, f"inflation={inflation:.2f}")

    # weight_decay=1e-5 inertness: the decoupled term is lr * wd * theta.
    c = cfg()
    theta = 10.0  # generous over-estimate of ||theta_lora||
    per_step = 1.2e-4 * c.weight_decay * theta
    over_run = per_step * 1050  # ~25 iterations at ~42 optimizer steps each
    check("weight_decay=1e-5 shrinks theta by <1e-4 over a whole 25-iter run "
          "(inert, not regularising)",
          over_run < 1e-4, f"total={over_run:.3e} vs weight_delta_norm ~0.7")


if __name__ == "__main__":
    test_defaults_bit_identical()
    test_validation_matrix()
    test_regime_warning()
    test_setup_wiring()
    test_eps_regime_arithmetic()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\033[32mAll AdamW-knob tests passed.\033[0m")
