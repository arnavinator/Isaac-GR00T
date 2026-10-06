"""Tests for `GRPOConfig.apply_jitter_fix` (jitter rows target a − ε′, not a − ε).

LOSS LEVEL, the REAL `compute_fm_log_prob` on test_exec_step_mask's stub DiT:
  [L1] False and the kwarg absent are bit-identical, with and without jitter.
  [L2] on, with jitter: matches a hand formula whose target is a − ε′_k.
  [L3] on, without noise_for_input: bit-identical to off.
  [L4] on: rows whose input noise IS ε are bit-identical to off.
  [L5] identity: (1−τ)²·MSE_fix,k == mean((â(x′_k) − a)²), â = x′ + (1−τ)v.
  [L6] mechanism: a field that lands EVERY input on a scores 0 under the fix and
       mean((ε′ − ε)²) under the original target.
DIAGNOSTICS:
  [D1] gap_* / headroom / budgets use the fix target; jacobian_fro_sq and
       gap_at_tau* keep the original target (equal to the fix-off values); one
       extra forward only when on; off == attribute absent.
TRAINER, the real ref pass + `_grpo_update` via test_exec_step_mask.run:
  [T1] the training forward gets apply_jitter_fix only when on, and its value
       equals the DEFAULT path evaluated at noise = ε′_k per τ.
  [T2] on with jitter off: step gradients and weights bit-identical to off.
  [T3] the chunk-gap survey passes the flag only when on.
CONFIG:
  [C1] default off; tyro --apply-jitter-fix; warns only when jitter is off.

Run: PYTHONPATH=/tmp/grpo_overlay .venv/bin/python scripts/grpo/test_jitter_fix.py
"""

import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_grpo  # noqa: E402
import test_exec_step_mask as esm  # noqa: E402  (stub DiT + real-fm trainer harness)
import test_jitter_metrics as tjm  # noqa: E402  (chunk-gap survey harness)
from fm_log_prob import compute_fm_log_prob  # noqa: E402
from grpo_config import GRPOConfig  # noqa: E402
from train_grpo import GRPOTrainer  # noqa: E402

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok or not detail else f"\n        {detail}"))
    if not ok:
        FAILURES.append(name)


def _maxdiff(a, b) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max())


def _common(head, inp):
    return dict(action_head=head, **inp)


def _hand_per_tau(W, actions, eps, mask, ts, nfi=None, fix=False) -> torch.Tensor:
    """[K, B] -MSE_k from the stub DiT's formula, independent of compute_fm_log_prob."""
    m = mask.float()
    valid = m.sum(dim=(1, 2))
    out = []
    for k in range(ts.shape[0]):
        te = ts[k][:, None, None]
        x_in = eps if nfi is None else nfi[k]
        x = (1 - te) * x_in + te * actions
        td = (ts[k] * 1000).long().to(x.dtype)
        v = x * W[None] + esm.MIX * x.mean(dim=1, keepdim=True) + esm.T_COEF * td[:, None, None]
        u = (actions - (x_in if fix else eps)).float()
        out.append(-((v.float() - u) ** 2 * m).sum(dim=(1, 2)) / valid)
    return torch.stack(out)


# ─────────────────────────────── loss level ───────────────────────────────

def test_L1_off_is_bit_identical():
    print("\n[L1] apply_jitter_fix=False == kwarg absent")
    head, inp = esm._Head(), esm._inputs(B=4, K=3)
    nfi = esm._jittered(inp)
    for name, extra in (("no jitter", {}), ("jitter", {"noise_for_input": nfi})):
        a = compute_fm_log_prob(**_common(head, inp), **extra)
        b = compute_fm_log_prob(**_common(head, inp), **extra, apply_jitter_fix=False)
        check(f"{name}: bitwise equal", torch.equal(a, b), f"max|diff| {_maxdiff(a, b):.3e}")


def test_L2_on_matches_hand_formula():
    print("\n[L2] on: target a − ε′_k")
    head, inp = esm._Head(), esm._inputs(B=4, K=3)
    nfi = esm._jittered(inp)
    _, got = compute_fm_log_prob(**_common(head, inp), noise_for_input=nfi,
                                 return_per_tau=True, apply_jitter_fix=True)
    W = head.model.W.detach()
    want = _hand_per_tau(W, inp["actions"], inp["noise"], inp["action_mask"],
                         inp["timesteps"], nfi=nfi, fix=True)
    orig = _hand_per_tau(W, inp["actions"], inp["noise"], inp["action_mask"],
                         inp["timesteps"], nfi=nfi, fix=False)
    check("per-τ values match the a − ε′ formula", torch.allclose(got, want, atol=1e-6),
          f"max|diff| {_maxdiff(got, want):.3e}")
    check("and differ from the a − ε formula (the test can fail)",
          _maxdiff(want, orig) > 1e-3, f"max|diff| {_maxdiff(want, orig):.3e}")


def test_L3_on_without_jitter_is_bit_identical():
    print("\n[L3] on, noise_for_input=None == off")
    head, inp = esm._Head(), esm._inputs(B=4, K=3)
    a = compute_fm_log_prob(**_common(head, inp))
    b = compute_fm_log_prob(**_common(head, inp), apply_jitter_fix=True)
    check("bitwise equal", torch.equal(a, b), f"max|diff| {_maxdiff(a, b):.3e}")


def test_L4_unjittered_rows_unchanged():
    print("\n[L4] on: rows fed the original ε are bit-identical to off")
    head, inp = esm._Head(), esm._inputs(B=4, K=3)
    nfi = esm._jittered(inp)
    nfi[:, 1] = inp["noise"][1]                       # row 1: a "fixed" / λ=0 row
    a = compute_fm_log_prob(**_common(head, inp), noise_for_input=nfi)
    b = compute_fm_log_prob(**_common(head, inp), noise_for_input=nfi, apply_jitter_fix=True)
    check("row 1 bitwise equal", torch.equal(a[1], b[1]), f"{a[1].item()} vs {b[1].item()}")
    check("jittered rows differ", bool((a[[0, 2, 3]] != b[[0, 2, 3]]).all()))


def test_L5_endpoint_identity():
    print("\n[L5] (1−τ)²·MSE_fix == mean((â(x′) − a)²)")
    head, inp = esm._Head(), esm._inputs(B=4, K=3)
    nfi = esm._jittered(inp)
    _, per_tau = compute_fm_log_prob(**_common(head, inp), noise_for_input=nfi,
                                     return_per_tau=True, apply_jitter_fix=True)
    W, a, ts, m = head.model.W.detach(), inp["actions"], inp["timesteps"], inp["action_mask"]
    ok = True
    for k in range(ts.shape[0]):
        te = ts[k][:, None, None]
        x = (1 - te) * nfi[k] + te * a
        td = (ts[k] * 1000).long().to(x.dtype)
        v = x * W[None] + esm.MIX * x.mean(dim=1, keepdim=True) + esm.T_COEF * td[:, None, None]
        a_hat = x + (1 - te) * v
        rhs = ((a_hat - a) ** 2 * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2))
        lhs = (1 - ts[k]) ** 2 * (-per_tau[k])
        ok &= torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-5)
    check("holds at every τ", ok)


class _LandOnA(nn.Module):
    """DiT whose velocity (A − x)/(1 − t) lands every input exactly on A."""

    def __init__(self, A):
        super().__init__()
        self.A = A

    def forward(self, hidden_states, timestep=None, **kw):
        t = timestep.to(hidden_states.dtype)[:, None, None] / 1000.0
        return (self.A - hidden_states) / (1.0 - t), None


def test_L6_mechanism():
    print("\n[L6] a field that lands every input on a")
    inp = esm._inputs(B=4, K=4)
    inp["timesteps"] = torch.tensor([0.0, 0.25, 0.5, 0.75])[:, None].expand(4, 4).clone()
    head = esm._Head()
    head.model = _LandOnA(inp["actions"])
    nfi = esm._jittered(inp, lam=0.25)
    fixed = -compute_fm_log_prob(**_common(head, inp), noise_for_input=nfi, apply_jitter_fix=True)
    orig = -compute_fm_log_prob(**_common(head, inp), noise_for_input=nfi)
    m = inp["action_mask"]
    shift = (((nfi - inp["noise"].unsqueeze(0)) ** 2 * m).sum(dim=(2, 3)) / m.sum(dim=(1, 2))).mean(0)
    check("fix: MSE == 0 (rewarded)", float(fixed.abs().max()) < 1e-10, f"{fixed.tolist()}")
    check("original: MSE == mean((ε′ − ε)²) (penalized)",
          torch.allclose(orig, shift, rtol=1e-5), f"{orig.tolist()} vs {shift.tolist()}")


# ─────────────────────────────── diagnostics ──────────────────────────────

def _diag(fix, lam_pos=0.25, lam_neg=0.05):
    head, inp = esm._Head(), esm._inputs(B=6, K=3, seed=4)
    t = GRPOTrainer.__new__(GRPOTrainer)
    cfg = dict(jitter_pos=lam_pos, jitter_neg=lam_neg, clip_eps_low=0.08,
               clip_eps_high=0.2, tau_centers=[0.0, 0.3, 0.6])
    if fix is not None:
        cfg["apply_jitter_fix"] = fix
    t.config = SimpleNamespace(**cfg)
    t.device = torch.device("cpu")
    t.model = SimpleNamespace(action_head=head)
    t._ref_mse_stats = {"pos_mean": 0.02}
    pos = torch.tensor([True, True, True, False, False, False])
    fixed = torch.tensor([False, False, False, False, False, True])
    lam_row = torch.where(pos, torch.full((6,), lam_pos), torch.full((6,), lam_neg))
    g = torch.Generator().manual_seed(11)
    xi = torch.randn(3, *inp["noise"].shape, generator=g)
    nfi = inp["noise"].unsqueeze(0).expand(3, -1, -1, -1).clone()
    jit, lam_j = ~fixed, lam_row[~fixed]
    nfi[:, jit] = ((1 - lam_j ** 2).sqrt()[None, :, None, None] * inp["noise"][jit].unsqueeze(0)
                   + lam_j[None, :, None, None] * xi[:, jit])
    n_calls = [0]
    real = train_grpo.compute_fm_log_prob

    def spy(**kw):
        n_calls[0] += 1
        return real(**kw)

    train_grpo.compute_fm_log_prob = spy
    try:
        out = t._jitter_gap_diagnostics(
            ready_backbone=inp["backbone_output"], ready_state_features=inp["state_features"],
            ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
            ready_masks=inp["action_mask"], ready_noise=inp["noise"],
            timesteps=inp["timesteps"], noise_for_input=nfi, lam_row=lam_row,
            pos_adv_mask=pos, fixed_row_mask=fixed, jitter_row_mask=jit)
    finally:
        train_grpo.compute_fm_log_prob = real
    return out, n_calls[0], head, inp, nfi, lam_row, pos, fixed


def test_D1_diagnostics_routing():
    print("\n[D1] gap diagnostics under the fix")
    off, n_off, *_ = _diag(False)
    absent, _, *_ = _diag(None)
    on, n_on, head, inp, nfi, lam_row, pos, fixed = _diag(True)
    check("off == attribute absent (every key, exact)", off == absent)
    check("forwards: 2 off, 3 on", (n_off, n_on) == (2, 3), f"{n_off}, {n_on}")
    W, a, eps, m, ts = (head.model.W.detach(), inp["actions"], inp["noise"],
                        inp["action_mask"], inp["timesteps"])
    clean = _hand_per_tau(W, a, eps, m, ts)
    gap_fix = clean - _hand_per_tau(W, a, eps, m, ts, nfi=nfi, fix=True)
    gap_vel = clean - _hand_per_tau(W, a, eps, m, ts, nfi=nfi, fix=False)
    jp, jn = pos & ~fixed, ~pos & ~fixed
    w = ((1 - ts) ** 2).mean(0)
    near = lambda x, y: abs(float(x) - float(y)) <= 1e-5 * max(1.0, abs(float(y)))
    check("gap_pos from the fix target", near(on["gap_pos"], gap_fix.mean(0)[jp].mean()))
    check("gap_neg from the fix target", near(on["gap_neg"], gap_fix.mean(0)[jn].mean()))
    check("headroom uses the fix-target gap",
          near(on["headroom_multiplier"], (0.02 + on["gap_pos"]) / 0.02))
    check("jacobian_fro_sq from the ORIGINAL target",
          near(on["jacobian_fro_sq"], (gap_vel.mean(0) / (w * lam_row ** 2))[jp].mean()))
    check("jacobian_fro_sq identical with the fix on and off",
          near(on["jacobian_fro_sq"], off["jacobian_fro_sq"]))
    for k in range(3):
        check(f"gap_at_tau{k} from the ORIGINAL target, equal to off",
              near(on[f"gap_at_tau{k}"], gap_vel[k][jp].mean())
              and near(on[f"gap_at_tau{k}"], off[f"gap_at_tau{k}"]))
    check("gap_pos differs from off (the test can fail)",
          abs(on["gap_pos"] - off["gap_pos"]) > 1e-4, f"{on['gap_pos']} vs {off['gap_pos']}")
    check("fixed-row selfcheck ~0", abs(on["gap_fixed_rows_selfcheck"]) < 1e-6)


# ──────────────────────────────── trainer ─────────────────────────────────

def _run_spied(**cfg_kw):
    """esm.run with every training forward re-evaluated at noise = ε′_k, same weights."""
    true_fm = train_grpo.compute_fm_log_prob
    seen = []

    def wrapper(**kw):
        out = true_fm(**kw)
        if "smooth_instrument" in kw and kw.get("noise_for_input") is not None:
            base = {k: kw[k] for k in ("action_head", "backbone_output", "state_features",
                                       "embodiment_id", "actions", "action_mask")}
            nfi, ts = kw["noise_for_input"], kw["timesteps"]
            with torch.no_grad():
                ref = torch.stack([true_fm(**base, timesteps=ts[k:k + 1], noise=nfi[k],
                                           n_samples=1) for k in range(ts.shape[0])]).mean(0)
            lp = out[0] if isinstance(out, tuple) else out
            seen.append((kw, lp.detach(), ref))
        return out

    train_grpo.compute_fm_log_prob = wrapper
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = esm.run(flag=False, **cfg_kw)
    finally:
        train_grpo.compute_fm_log_prob = true_fm
    return r, seen


def test_T1_training_forward():
    print("\n[T1] training forward (real ref pass + _grpo_update)")
    r_on, seen_on = _run_spied(jitter_pos=0.25, jitter_paired=False, apply_jitter_fix=True)
    r_off, seen_off = _run_spied(jitter_pos=0.25, jitter_paired=False)
    check("jittered training forwards ran", len(seen_on) > 0 and len(seen_off) > 0)
    check("on: every one carries apply_jitter_fix=True",
          all(kw.get("apply_jitter_fix") is True for kw, _, _ in seen_on))
    check("off: the kwarg is never passed",
          all("apply_jitter_fix" not in kw for kw, _, _ in seen_off))
    check("on: value == default path at noise = ε′_k (target a − ε′)",
          all(torch.allclose(lp, ref, atol=1e-6) for _, lp, ref in seen_on),
          f"max|diff| {max(_maxdiff(lp, ref) for _, lp, ref in seen_on):.3e}")
    check("off: value != that (the test can fail)",
          any(not torch.allclose(lp, ref, atol=1e-4) for _, lp, ref in seen_off))
    check("on and off take different steps",
          any(not torch.equal(a, b) for a, b in zip(r_on.step_grads, r_off.step_grads)))


def test_T2_on_without_jitter_is_bit_identical():
    print("\n[T2] on with jitter off == off")
    r_on, _ = _run_spied(apply_jitter_fix=True)
    r_off, _ = _run_spied()
    check("same number of steps", len(r_on.step_grads) == len(r_off.step_grads) > 0)
    check("step gradients bitwise equal",
          all(torch.equal(a, b) for a, b in zip(r_on.step_grads, r_off.step_grads)))
    check("final weights bitwise equal",
          torch.equal(r_on.policy.action_head.model.W, r_off.policy.action_head.model.W))


def test_T3_chunk_gap_survey():
    print("\n[T3] chunk-gap survey")
    for fix in (True, False):
        t, chunks, fake = tjm._survey_trainer(64, gap_of=lambda c: 0.05)
        t.config.apply_jitter_fix = fix
        kws = []

        def rec(**kw):
            kws.append(kw)
            return fake(**kw)

        real = train_grpo.compute_fm_log_prob
        train_grpo.compute_fm_log_prob = rec
        try:
            out = t._per_chunk_gap_survey(chunks)
        finally:
            train_grpo.compute_fm_log_prob = real
        ok = (all(kw.get("apply_jitter_fix") is True for kw in kws) if fix
              else all("apply_jitter_fix" not in kw for kw in kws))
        check(f"{'on' if fix else 'off'}: flag passed only when on ({len(kws)} calls)",
              ok and len(kws) > 0 and out is not None)


# ───────────────────────────────── config ─────────────────────────────────

def test_C1_config():
    print("\n[C1] config")
    check("default False", GRPOConfig(device="cpu").apply_jitter_fix is False)
    import tyro
    cfg = tyro.cli(GRPOConfig, args=["--device", "cpu", "--jitter-pos", "0.125",
                                     "--apply-jitter-fix"])
    check("tyro --apply-jitter-fix", cfg.apply_jitter_fix is True)

    def warned(**kw):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GRPOConfig(device="cpu", **kw)
        return any("apply_jitter_fix=True has no effect" in str(x.message) for x in w)

    check("warns when jitter is off", warned(apply_jitter_fix=True))
    check("no warning with jitter on", not warned(apply_jitter_fix=True, jitter_pos=0.125))


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All apply_jitter_fix tests passed.")


if __name__ == "__main__":
    main()
