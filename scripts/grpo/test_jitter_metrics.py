"""Tests for the jitter-gap / ratio-headroom instrumentation.

Covers the metrics added for the Track-A2 instrumentation:
  * `compute_fm_log_prob(..., return_per_tau=True)` contract
  * `GRPOTrainer._jitter_gap_diagnostics` arithmetic (gap, Jacobian norm,
    headroom multiplier, negative-side clip-budget, fixed-row self-check)
  * the EFFECTIVE clipfrac truth table
  * `_summarize_ref_mse`

Design notes:
  * `test_grad_accum.py` verifies the same code path with a *value-pinned*
    surrogate whose log-prob does not depend on tau or on `noise_for_input`, so
    the gap it measures is exactly 0. That checks the plumbing. This file uses a
    stand-in that IS tau- and noise-sensitive with a closed form, so the gap
    ARITHMETIC can be checked against an independently computed expectation.
  * The stand-in records every call's kwargs, so expectations are recomputed
    from the exact tensors the production code built (including the unseeded
    xi), rather than from a re-derivation that could drift.
  * CPU only; no GPU, no model download, no gr00t import.

Run: uv run python scripts/grpo/test_jitter_metrics.py
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_grpo  # noqa: E402
from fm_log_prob import compute_fm_log_prob  # noqa: E402
from train_grpo import GRPOTrainer, clip_killed_gradient  # noqa: E402

FAILURES: list[str] = []
GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  {GREEN}PASS{RESET}  {name}")
    else:
        print(f"  {RED}FAIL{RESET}  {name}" + (f"\n          {detail}" if detail else ""))
        FAILURES.append(name)


def close(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ───────────────────────── stub action head ──────────────────────────────
# Minimal surface that the REAL compute_fm_log_prob touches. Linear throughout
# so the velocity prediction has an exactly known dependence on the noisy
# trajectory, which makes the Jacobian analytically available.

class _Cfg:
    add_pos_embed = False
    use_alternate_vl_dit = False


class _StubHead(torch.nn.Module):
    """pred_velocity = SCALE * noisy_trajectory (elementwise).

    So MSE(pred, a - eps) with input built from eps_in is
        mean over valid dims of (SCALE*((1-t)*eps_in + t*a) - (a - eps))^2
    which is a closed form in (t, eps_in, a, eps) — everything the diagnostic
    needs to be checked against.
    """

    def __init__(self, horizon: int, dim: int, scale: float = 0.5):
        super().__init__()
        self.num_timestep_buckets = 1000
        self.config = _Cfg()
        self.scale = scale
        self.horizon = horizon
        self.dim = dim

    # compute_fm_log_prob calls action_encoder(traj, t_disc, emb_id) then
    # concatenates state_features in front, runs .model(...), then
    # action_decoder(...) and slices the last `horizon` rows back off.
    def action_encoder(self, noisy_trajectory, t_discretized, embodiment_id):
        return noisy_trajectory

    def model(self, hidden_states, encoder_hidden_states, encoder_attention_mask,
              timestep, return_all_hidden_states, **kw):
        return hidden_states, None

    def action_decoder(self, model_output, embodiment_id):
        return self.scale * model_output


def _make_inputs(B=6, H=4, D=8, K=3, n_state=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(B, H, D, generator=g)
    eps = torch.randn(B, H, D, generator=g)
    mask = torch.ones(B, H, D)
    mask[:, :, D // 2:] = 0.0            # half the dims padded, like Panda
    # Structured taus: distinct centres per k plus tight jitter, mirroring
    # production's _sample_jittered_timesteps(tau_centers, jitter_std=0.02).
    # NOT torch.rand(K, B): with i.i.d. taus every k has the same expected value,
    # so there is no tau gradient across k and any per-tau ordering assertion is
    # vacuous — which is exactly how a too-loose check hid that here before.
    centres = torch.linspace(0.0, 0.75, K)
    timesteps = (centres[:, None] + torch.randn(K, B, generator=g) * 0.02).clamp(0, 0.999)
    state_features = torch.zeros(B, n_state, D)
    backbone = {"backbone_features": torch.zeros(B, 1, D)}
    return dict(
        actions=actions, eps=eps, mask=mask, timesteps=timesteps,
        state_features=state_features, backbone=backbone,
        embodiment_id=torch.zeros(B, dtype=torch.long), K=K, B=B, H=H, D=D,
    )


def _expected_logprob_per_tau(inp, head, eps_input_all):
    """Closed form of what compute_fm_log_prob must return, per tau. [K, B]."""
    K, B = inp["K"], inp["B"]
    a, eps, mask = inp["actions"], inp["eps"], inp["mask"]
    target = a - eps
    n_valid = mask.sum(dim=(1, 2))
    out = torch.zeros(K, B, dtype=torch.float32)
    for k in range(K):
        t = inp["timesteps"][k][:, None, None]
        eps_in = eps if eps_input_all is None else eps_input_all[k]
        traj = (1 - t) * eps_in + t * a
        pred = head.scale * traj
        se = (pred.float() - target.float()) ** 2 * mask
        out[k] = -(se.sum(dim=(1, 2)) / n_valid)
    return out


# ──────────────────────── 1. per-tau contract ────────────────────────────

def test_per_tau_contract():
    print("\n[fm_log_prob] return_per_tau contract")
    inp = _make_inputs()
    head = _StubHead(inp["H"], inp["D"])
    common = dict(
        action_head=head, backbone_output=inp["backbone"],
        state_features=inp["state_features"], embodiment_id=inp["embodiment_id"],
        actions=inp["actions"], action_mask=inp["mask"],
        timesteps=inp["timesteps"], noise=inp["eps"], n_samples=inp["K"],
    )
    plain = compute_fm_log_prob(**common)
    check("default call returns a bare Tensor", isinstance(plain, torch.Tensor))

    got = compute_fm_log_prob(**common, return_per_tau=True)
    check("return_per_tau=True returns a 2-tuple",
          isinstance(got, tuple) and len(got) == 2)
    mean, per_tau = got
    check("per_tau shape is [K, B]", tuple(per_tau.shape) == (inp["K"], inp["B"]),
          f"got {tuple(per_tau.shape)}")
    check("per_tau is fp32", per_tau.dtype == torch.float32)
    check("mean is BIT-IDENTICAL to the default call",
          torch.equal(mean, plain),
          f"max|diff| = {(mean - plain).abs().max().item():.3e}")
    check("per_tau.mean(0) == mean (to fp32 round-off)",
          torch.allclose(per_tau.mean(dim=0), mean, atol=1e-6),
          f"max|diff| = {(per_tau.mean(0) - mean).abs().max().item():.3e}")

    exp = _expected_logprob_per_tau(inp, head, None)
    check("per_tau matches the closed form",
          torch.allclose(per_tau, exp, atol=1e-6),
          f"max|diff| = {(per_tau - exp).abs().max().item():.3e}")

    # With a per-K noise_for_input (the jitter path).
    g = torch.Generator().manual_seed(7)
    nfi = torch.randn(inp["K"], *inp["eps"].shape, generator=g)
    _, per_tau_j = compute_fm_log_prob(**common, noise_for_input=nfi,
                                       return_per_tau=True)
    exp_j = _expected_logprob_per_tau(inp, head, nfi)
    check("per_tau matches the closed form WITH noise_for_input",
          torch.allclose(per_tau_j, exp_j, atol=1e-6),
          f"max|diff| = {(per_tau_j - exp_j).abs().max().item():.3e}")
    check("velocity target stays anchored at the ORIGINAL eps "
          "(jitter changes the input only)",
          not torch.allclose(per_tau_j, per_tau, atol=1e-4))

    # Gradients must still flow with return_per_tau on.
    head_g = _StubHead(inp["H"], inp["D"])
    head_g.w = torch.nn.Parameter(torch.tensor(1.0))
    head_g.action_decoder = lambda mo, e: head_g.scale * mo * head_g.w
    m2, _ = compute_fm_log_prob(
        **{**common, "action_head": head_g}, return_per_tau=True)
    m2.sum().backward()
    check("gradient still flows to params with return_per_tau=True",
          head_g.w.grad is not None and torch.isfinite(head_g.w.grad).all())


# ─────────────────── 2. _jitter_gap_diagnostics arithmetic ───────────────

def _probe_trainer(lam_pos, lam_neg, clip_lo=0.08, clip_hi=0.2, ref_pos=None,
                   taus=(0.0, 0.3, 0.6)):
    """Trainer with only the attributes the diagnostics touch.

    Built via __new__ so no GPU, no model load, no setup(). Note this is exactly
    the shape that caught the `self._ref_mse_stats` AttributeError: an object
    that never ran __init__.
    """
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.config = SimpleNamespace(
        jitter_pos=lam_pos,
        jitter_neg=lam_neg,
        clip_eps_low=clip_lo,
        clip_eps_high=clip_hi,
        tau_centers=list(taus),
    )
    t.device = torch.device("cpu")
    if ref_pos is not None:
        t._ref_mse_stats = {"pos_mean": ref_pos}
    return t


def test_jitter_gap_arithmetic():
    print("\n[jitter] _jitter_gap_diagnostics arithmetic")
    LAM_POS, LAM_NEG, REF_POS = 0.25, 0.05, 0.02
    inp = _make_inputs(B=6, K=3, seed=3)
    head = _StubHead(inp["H"], inp["D"])
    trainer = _probe_trainer(LAM_POS, LAM_NEG, ref_pos=REF_POS)
    trainer.model = type("M", (), {"action_head": head})()

    # Rows 0-2 positive advantage, 3-5 negative. Row 5 is a paired "fixed" row.
    pos_mask = torch.tensor([True, True, True, False, False, False])
    fixed_mask = torch.tensor([False, False, False, False, False, True])
    lam_row = torch.where(pos_mask, torch.full((6,), LAM_POS),
                          torch.full((6,), LAM_NEG))

    # Build noise_for_input exactly as _grpo_update_inner does.
    g = torch.Generator().manual_seed(11)
    xi = torch.randn(inp["K"], *inp["eps"].shape, generator=g)
    nfi = inp["eps"].unsqueeze(0).expand(inp["K"], -1, -1, -1).clone()
    jit = ~fixed_mask
    lam_j = lam_row[jit]
    nfi[:, jit] = (
        (1.0 - lam_j * lam_j).sqrt()[None, :, None, None]
        * inp["eps"][jit].unsqueeze(0)
        + lam_j[None, :, None, None] * xi[:, jit]
    )

    out = trainer._jitter_gap_diagnostics(
        ready_backbone=inp["backbone"], ready_state_features=inp["state_features"],
        ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
        ready_masks=inp["mask"], ready_noise=inp["eps"],
        timesteps=inp["timesteps"], noise_for_input=nfi, lam_row=lam_row,
        pos_adv_mask=pos_mask, fixed_row_mask=fixed_mask,
        jitter_row_mask=~fixed_mask,
    )

    # Independent expectation from the closed form.
    lp_clean = _expected_logprob_per_tau(inp, head, None)
    lp_jit = _expected_logprob_per_tau(inp, head, nfi)
    gap_pt = lp_clean - lp_jit                                   # [K, B]
    gap_row = gap_pt.mean(dim=0)
    w_row = ((1.0 - inp["timesteps"]) ** 2).mean(dim=0)
    jac_row = gap_row / (w_row * lam_row ** 2)

    jp = jit & pos_mask
    jn = jit & ~pos_mask
    check("gap_pos matches the closed form",
          close(out["gap_pos"], gap_row[jp].mean(), 1e-5),
          f"{out['gap_pos']:.8f} vs {gap_row[jp].mean():.8f}")
    check("gap_neg matches the closed form",
          close(out["gap_neg"], gap_row[jn].mean(), 1e-5),
          f"{out['gap_neg']:.8f} vs {gap_row[jn].mean():.8f}")
    check("n_rows_pos counts only jitter AND positive rows",
          out["n_rows_pos"] == int(jp.sum()))
    check("jacobian_fro_sq divides out (1-tau)^2 * lam^2 PER ROW",
          close(out["jacobian_fro_sq"], jac_row[jp].mean(), 1e-5),
          f"{out['jacobian_fro_sq']:.8f} vs {jac_row[jp].mean():.8f}")

    for k in range(inp["K"]):
        check(f"gap_at_tau{k} matches the closed form",
              close(out[f"gap_at_tau{k}"], gap_pt[k][jp].mean(), 1e-5))
        check(f"tau{k}_value reports the ACTUAL sampled tau",
              close(out[f"tau{k}_value"], inp["timesteps"][k][jp].mean(), 1e-5))

    check("headroom_multiplier == (MSE_ref_pos + gap_pos) / MSE_ref_pos",
          close(out["headroom_multiplier"], (REF_POS + out["gap_pos"]) / REF_POS, 1e-9))
    check("headroom_ref_only == MSE_ref_pos", close(out["headroom_ref_only"], REF_POS))
    check("headroom_with_jitter == MSE_ref_pos + gap_pos",
          close(out["headroom_with_jitter"], REF_POS + out["gap_pos"], 1e-9))
    # NOT asserting headroom_multiplier > 1 here: _StubHead's MSE is not
    # minimised at eps (see _MinAtEpsHead's docstring), so gap_pos can legitimately
    # come out negative for this stand-in and a > 1 check would be seed luck.
    # The sign is asserted properly in test_jacobian_estimator_is_lambda_invariant.

    lo_budget = -math.log(1 - 0.08)
    check("neg_clip_budget_used == gap_neg / |log(1-clip_eps_low)|",
          close(out["neg_clip_budget_used"], out["gap_neg"] / lo_budget, 1e-9))
    check("n_rows_neg reported alongside n_rows_pos",
          out["n_rows_neg"] == int(jn.sum()))

    check("gap_fixed_rows_selfcheck is ~0 for a true fixed row",
          abs(out["gap_fixed_rows_selfcheck"]) < 1e-6,
          f"{out['gap_fixed_rows_selfcheck']:.3e}")


class _MinAtEpsHead(torch.nn.Module):
    """Stand-in whose FM residual vanishes exactly at eps_in == eps.

    Needed for the lambda-scaling checks below. `_StubHead` predicts
    `SCALE * x_t`, whose MSE against `(a - eps)` is a quadratic in `eps_in`
    minimised somewhere that is NOT `eps` — so perturbing `eps_in` can DECREASE
    its loss, producing negative gaps and no lambda ordering. That is a property
    of that stub, not of the estimator: in the real setting the reference policy
    *generated* `a` from `eps`, so `MSE(eps_in)` sits at a local minimum at
    `eps_in = eps` and the Taylor expansion the diagnostic inverts applies.

    This head encodes that property directly:

        pred_velocity = c * (x_t - x_t(eps)) + (a - eps)

    so the residual is `c * (1-tau) * (eps' - eps)` and

        gap(tau) = c^2 * (1-tau)^2 * mean_valid[(eps' - eps)^2]

    which is exactly the form `_jitter_gap_diagnostics` divides
    `(1-tau)^2 * lambda^2` out of. Since
    `E[(eps'-eps)^2] = (sqrt(1-lam^2)-1)^2 + lam^2 ~= lam^2`, the recovered
    `jacobian_fro_sq` must be ~c^2 for every lambda — which is the estimator's
    headline property and what makes the metric comparable across jitter_pos
    settings.
    """

    def __init__(self, actions, eps, timesteps, c=0.7):
        super().__init__()
        self.num_timestep_buckets = 1000
        self.config = _Cfg()
        self.actions, self.eps, self.timesteps, self.c = actions, eps, timesteps, c
        self.K = timesteps.shape[0]
        self.H = actions.shape[1]
        self._k = 0

    def action_encoder(self, noisy_trajectory, t_discretized, embodiment_id):
        return noisy_trajectory

    def model(self, hidden_states, **kw):
        return hidden_states, None

    def action_decoder(self, model_output, embodiment_id):
        # compute_fm_log_prob's K-loop calls this exactly once per tau, in
        # order, so a modulo-K counter recovers the exact float tau (rather than
        # the lossy t_discretized) without changing any production signature.
        t = self.timesteps[self._k % self.K][:, None, None]
        self._k += 1
        x_ref = (1 - t) * self.eps + t * self.actions
        out = torch.zeros_like(model_output)
        out[:, -self.H:] = (
            self.c * (model_output[:, -self.H:] - x_ref) + (self.actions - self.eps)
        )
        return out


def test_jacobian_estimator_is_lambda_invariant():
    print("\n[jitter] jacobian_fro_sq is comparable across jitter_pos")
    C = 0.7
    # Sized so the finite-xi noise on mean[(eps'-eps)^2] is small: 8 positive
    # rows x 64 valid dims x 4 taus ~ 2000 samples -> ~3% relative std.
    inp = _make_inputs(B=12, H=8, D=16, K=4, seed=21)
    head = _MinAtEpsHead(inp["actions"], inp["eps"], inp["timesteps"], c=C)
    trainer = _probe_trainer(0.25, 0.05, ref_pos=0.02)
    trainer.model = type("M", (), {"action_head": head})()

    pos_mask = torch.zeros(12, dtype=torch.bool)
    pos_mask[:8] = True
    fixed_mask = torch.zeros(12, dtype=torch.bool)

    g = torch.Generator().manual_seed(99)
    xi = torch.randn(inp["K"], *inp["eps"].shape, generator=g)

    results = {}
    for lam in (0.10, 0.25, 0.50):
        lam_row = torch.where(pos_mask, torch.full((12,), lam),
                              torch.full((12,), 0.05))
        nfi = inp["eps"].unsqueeze(0).expand(inp["K"], -1, -1, -1).clone()
        lj = lam_row
        nfi[:, :] = (
            (1.0 - lj * lj).sqrt()[None, :, None, None] * inp["eps"].unsqueeze(0)
            + lj[None, :, None, None] * xi
        )
        head._k = 0
        results[lam] = trainer._jitter_gap_diagnostics(
            ready_backbone=inp["backbone"], ready_state_features=inp["state_features"],
            ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
            ready_masks=inp["mask"], ready_noise=inp["eps"],
            timesteps=inp["timesteps"], noise_for_input=nfi, lam_row=lam_row,
            pos_adv_mask=pos_mask, fixed_row_mask=fixed_mask,
            jitter_row_mask=~fixed_mask,
        )

    gaps = [results[l]["gap_pos"] for l in (0.10, 0.25, 0.50)]
    check("gap_pos is strictly POSITIVE when MSE is minimised at eps",
          all(g > 0 for g in gaps), f"{[round(g, 6) for g in gaps]}")
    check("gap_pos grows monotonically with lam_pos",
          gaps[0] < gaps[1] < gaps[2], f"{[round(g, 6) for g in gaps]}")
    # gap ~ lam^2, so a 2.5x lam step should give ~6.25x gap.
    check("gap_pos scales as lam^2 (0.25/0.10 -> ~6.25x)",
          5.0 < gaps[1] / gaps[0] < 8.0, f"ratio {gaps[1] / gaps[0]:.2f}")
    jac = [results[l]["jacobian_fro_sq"] for l in (0.10, 0.25, 0.50)]
    # Tolerance covers the KNOWN upward bias: the estimator divides by lam^2 but
    # the true per-element variance of (eps'-eps) is (sqrt(1-lam^2)-1)^2 + lam^2,
    # so it reads +1.6% high at lam=0.25 and +7% at lam=0.50 (documented in
    # _jitter_gap_diagnostics). Plus finite-xi noise at this problem size.
    check("jacobian_fro_sq recovers c^2 for every lam (comparable across lam)",
          all(abs(v - C ** 2) / C ** 2 < 0.20 for v in jac),
          f"values {[round(v, 4) for v in jac]}, target c^2 = {C ** 2:.4f}")
    check("the known lam^2 bias is upward and small (lam=0.50 >= lam=0.10)",
          jac[2] >= jac[0] * 0.98,
          f"lam=0.10 -> {jac[0]:.4f}, lam=0.50 -> {jac[2]:.4f}")
    check("gap_at_tau{k} falls as tau rises (prefactor is (1-tau)^2)",
          _tau_ordering_ok(results[0.25], inp["K"]),
          str({f"tau{k}": (round(results[0.25][f'tau{k}_value'], 3),
                           round(results[0.25][f'gap_at_tau{k}'], 6))
               for k in range(inp["K"])}))
    # Quantitative version of the same claim: the gap ratio between the lowest
    # and highest tau must track ((1-tau_lo)/(1-tau_hi))^2 rather than being flat.
    pairs = sorted((results[0.25][f"tau{k}_value"], results[0.25][f"gap_at_tau{k}"])
                   for k in range(inp["K"]))
    (t_lo, g_lo), (t_hi, g_hi) = pairs[0], pairs[-1]
    predicted = ((1 - t_lo) / (1 - t_hi)) ** 2
    check("gap ratio across tau matches the (1-tau)^2 prefactor",
          0.7 * predicted < g_lo / g_hi < 1.4 * predicted,
          f"observed {g_lo / g_hi:.3f} vs predicted {predicted:.3f} "
          f"(tau {t_lo:.3f} -> {t_hi:.3f})")


def _tau_ordering_ok(out, K):
    """gap must be monotonically decreasing in tau, since gap ~ (1-tau)^2."""
    pairs = sorted((out[f"tau{k}_value"], out[f"gap_at_tau{k}"]) for k in range(K))
    gaps = [g for _, g in pairs]
    return all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))


def test_jitter_gap_edge_cases():
    print("\n[jitter] _jitter_gap_diagnostics edge cases")
    inp = _make_inputs(B=4, K=2, seed=5)
    head = _StubHead(inp["H"], inp["D"])
    nfi = inp["eps"].unsqueeze(0).expand(inp["K"], -1, -1, -1).clone() * 1.05

    # No positive rows at all -> no pos keys, no headroom, still no crash.
    t = _probe_trainer(0.25, 0.05, ref_pos=0.02)
    t.model = type("M", (), {"action_head": head})()
    out = t._jitter_gap_diagnostics(
        ready_backbone=inp["backbone"], ready_state_features=inp["state_features"],
        ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
        ready_masks=inp["mask"], ready_noise=inp["eps"], timesteps=inp["timesteps"],
        noise_for_input=nfi, lam_row=torch.full((4,), 0.05),
        pos_adv_mask=torch.zeros(4, dtype=torch.bool),
        fixed_row_mask=torch.zeros(4, dtype=torch.bool),
        jitter_row_mask=torch.ones(4, dtype=torch.bool),
    )
    check("all-negative minibatch: no gap_pos / headroom keys",
          "gap_pos" not in out and "headroom_multiplier" not in out, str(sorted(out)))
    check("all-negative minibatch: gap_neg still reported", "gap_neg" in out)
    check("all-negative minibatch: no fixed-row self-check key",
          "gap_fixed_rows_selfcheck" not in out)

    # Missing _ref_mse_stats entirely (trainer built via __new__) must not raise.
    t2 = _probe_trainer(0.25, 0.05, ref_pos=None)
    t2.model = type("M", (), {"action_head": head})()
    out2 = t2._jitter_gap_diagnostics(
        ready_backbone=inp["backbone"], ready_state_features=inp["state_features"],
        ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
        ready_masks=inp["mask"], ready_noise=inp["eps"], timesteps=inp["timesteps"],
        noise_for_input=nfi, lam_row=torch.full((4,), 0.25),
        pos_adv_mask=torch.ones(4, dtype=torch.bool),
        fixed_row_mask=torch.zeros(4, dtype=torch.bool),
        jitter_row_mask=torch.ones(4, dtype=torch.bool),
    )
    check("absent _ref_mse_stats: no raise, gap still reported",
          "gap_pos" in out2 and "headroom_multiplier" not in out2)

    # lam == 0 on every row: denominator guard must avoid inf/NaN.
    t3 = _probe_trainer(0.0, 0.0)
    t3.model = type("M", (), {"action_head": head})()
    out3 = t3._jitter_gap_diagnostics(
        ready_backbone=inp["backbone"], ready_state_features=inp["state_features"],
        ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
        ready_masks=inp["mask"], ready_noise=inp["eps"], timesteps=inp["timesteps"],
        noise_for_input=nfi, lam_row=torch.zeros(4),
        pos_adv_mask=torch.ones(4, dtype=torch.bool),
        fixed_row_mask=torch.zeros(4, dtype=torch.bool),
        jitter_row_mask=torch.ones(4, dtype=torch.bool),
    )
    check("lam == 0: jacobian_fro_sq is finite (guarded division)",
          math.isfinite(out3["jacobian_fro_sq"]), f"{out3['jacobian_fro_sq']}")
    check("lam == 0: every emitted value is finite",
          all(math.isfinite(v) for v in out3.values()),
          str({k: v for k, v in out3.items() if not math.isfinite(v)}))

    # The diagnostic must not consume RNG (runs stay comparable).
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    t3._jitter_gap_diagnostics(
        ready_backbone=inp["backbone"], ready_state_features=inp["state_features"],
        ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
        ready_masks=inp["mask"], ready_noise=inp["eps"], timesteps=inp["timesteps"],
        noise_for_input=nfi, lam_row=torch.full((4,), 0.25),
        pos_adv_mask=torch.ones(4, dtype=torch.bool),
        fixed_row_mask=torch.zeros(4, dtype=torch.bool),
        jitter_row_mask=torch.ones(4, dtype=torch.bool),
    )
    check("diagnostic consumes NO global RNG",
          torch.equal(before, torch.get_rng_state()))

    # No gradient may leak out of the diagnostic.
    head_g = _StubHead(inp["H"], inp["D"])
    head_g.w = torch.nn.Parameter(torch.tensor(1.0))
    head_g.action_decoder = lambda mo, e: head_g.scale * mo * head_g.w
    t4 = _probe_trainer(0.25, 0.05)
    t4.model = type("M", (), {"action_head": head_g})()
    t4._jitter_gap_diagnostics(
        ready_backbone=inp["backbone"], ready_state_features=inp["state_features"],
        ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
        ready_masks=inp["mask"], ready_noise=inp["eps"], timesteps=inp["timesteps"],
        noise_for_input=nfi, lam_row=torch.full((4,), 0.25),
        pos_adv_mask=torch.ones(4, dtype=torch.bool),
        fixed_row_mask=torch.zeros(4, dtype=torch.bool),
        jitter_row_mask=torch.ones(4, dtype=torch.bool),
    )
    check("diagnostic leaves no gradient on params (no_grad honoured)",
          head_g.w.grad is None)


# ─────────────────── 3. effective clipfrac truth table ───────────────────

def test_effective_clipfrac_truth_table():
    print("\n[clipfrac] effective (clip-term gradient dead) vs threshold clipfrac")
    LO, HI = 0.08, 0.2
    # (advantage, ratio, expect_grad_dead, label)
    cases = [
        (+1.0, 0.50, False, "A>0, rho below lower bound  -> min picks A*rho, ALIVE"),
        (+1.0, 0.96, False, "A>0, rho inside band        -> ALIVE"),
        (+1.0, 1.50, True,  "A>0, rho above upper bound  -> min picks A*(1+hi), DEAD"),
        (-1.0, 0.50, True,  "A<0, rho below lower bound  -> min picks A*(1-lo), DEAD"),
        (-1.0, 0.96, False, "A<0, rho inside band        -> ALIVE"),
        (-1.0, 1.50, False, "A<0, rho above upper bound  -> min picks A*rho, ALIVE"),
    ]
    adv = torch.tensor([c[0] for c in cases])
    ratio = torch.tensor([c[1] for c in cases])
    surr1 = adv * ratio
    surr2 = adv * torch.clamp(ratio, 1 - LO, 1 + HI)

    # Calls the PRODUCTION predicate, not a re-derivation. A re-implemented copy
    # here would keep passing if train_grpo's expression were changed.
    grad_dead = clip_killed_gradient(ratio, surr1, surr2, LO, HI)

    for i, (a, r, expect, label) in enumerate(cases):
        check(label, bool(grad_dead[i]) == expect,
              f"grad_dead={bool(grad_dead[i])} expected={expect}")

    # Independent oracle: autograd on the real loss decides the `expect` column,
    # so the two agree only if BOTH the predicate and the table are right.
    for i, (a, r, expect, label) in enumerate(cases):
        rho = torch.tensor(float(r), requires_grad=True)
        av = torch.tensor(float(a))
        loss = -torch.min(av * rho, av * torch.clamp(rho, 1 - LO, 1 + HI))
        loss.backward()
        by_autograd = float(rho.grad) == 0.0
        check(f"  autograd oracle agrees (A={a:+.0f}, rho={r})",
              by_autograd == bool(grad_dead[i]),
              f"autograd_dead={by_autograd} predicate={bool(grad_dead[i])}")

    # Zero-advantage rows: `surr1 == surr2` satisfies the min() half, but the
    # conjunction still needs the clamp to have moved. So in-band => ALIVE,
    # out-of-band => DEAD, even though the row has no gradient either way. This
    # pins the (corrected) docstring claim; an earlier version said in-band
    # A == 0 was counted DEAD, which is false.
    z_ratio = torch.tensor([1.00, 0.50, 1.50])
    z_adv = torch.zeros(3)
    z_dead = clip_killed_gradient(
        z_ratio, z_adv * z_ratio,
        z_adv * torch.clamp(z_ratio, 1 - LO, 1 + HI), LO, HI)
    check("A == 0 in-band  -> ALIVE (clip is not what killed it)",
          bool(z_dead[0]) is False, str(z_dead.tolist()))
    check("A == 0 below the lower bound -> DEAD",
          bool(z_dead[1]) is True, str(z_dead.tolist()))
    check("A == 0 above the upper bound -> DEAD",
          bool(z_dead[2]) is True, str(z_dead.tolist()))

    # Randomised agreement sweep: the predicate must match autograd everywhere,
    # not just on hand-picked points. A == 0 is included explicitly because it is
    # the only place `<=` vs `<` differs, and a pure randn sweep never hits it.
    g = torch.Generator().manual_seed(4)
    n_disagree = 0
    draws = [(0.0, r) for r in (0.5, 0.92, 1.0, 1.2, 1.5)]
    draws += [(float(torch.randn(1, generator=g)),
               float(torch.rand(1, generator=g) * 2.0)) for _ in range(400)]
    for a, r in draws:
        rho = torch.tensor(r, requires_grad=True)
        av = torch.tensor(a)
        (-torch.min(av * rho, av * torch.clamp(rho, 1 - LO, 1 + HI))).backward()
        pred = bool(clip_killed_gradient(
            torch.tensor([r]), torch.tensor([a * r]),
            torch.tensor([a * min(max(r, 1 - LO), 1 + HI)]), LO, HI)[0])
        # autograd reports 0 for a zero-advantage row regardless of the clamp, so
        # A == 0 rows are excluded from the equivalence claim and asserted above.
        if a != 0.0 and (float(rho.grad) == 0.0) != pred:
            n_disagree += 1
    check("predicate agrees with autograd over 400 random (A, rho) draws",
          n_disagree == 0, f"{n_disagree} disagreements")

    # The headline clipfrac counts 4 of 6 as clipped; only 2 are really dead.
    clamp_moved = (ratio < 1 - LO) | (ratio > 1 + HI)
    check("plain clipfrac OVERSTATES: 4/6 flagged vs 2/6 actually dead",
          int(clamp_moved.sum()) == 4 and int(grad_dead.sum()) == 2,
          f"clamp_moved={int(clamp_moved.sum())} grad_dead={int(grad_dead.sum())}")

    # The specific failure mode the metric exists for: a large jitter_pos puts
    # EVERY positive row below the lower bound with a fully live clip gradient.
    adv_p = torch.ones(8)
    ratio_p = torch.full((8,), 0.72)          # rho at jitter_pos ~= 0.6
    gd = clip_killed_gradient(
        ratio_p, adv_p * ratio_p,
        adv_p * torch.clamp(ratio_p, 1 - LO, 1 + HI), LO, HI)
    cm = (ratio_p < 1 - LO) | (ratio_p > 1 + HI)
    check("large jitter_pos: threshold clipfrac=1.0 but effective=0.0",
          float(cm.float().mean()) == 1.0 and float(gd.float().mean()) == 0.0,
          f"threshold={float(cm.float().mean())} effective={float(gd.float().mean())}")

    # The post-renorm bucketing is pinned BEHAVIOURALLY in
    # test_effective_clipfrac_aggregation_values (the "only post-renorm-positive
    # rows dead -> _pos == 1.0, _neg == 0.0" pair). A source-text grep was tried
    # here first and was wrong in both directions: it failed on a semantically
    # identical reformat, and it could not detect the two bucket bodies being
    # swapped, since both substrings remain present either way.


# ─────────────────────── 4. _summarize_ref_mse ───────────────────────────

class _Chunk:
    def __init__(self, lp, adv, blp=None):
        self.ref_log_prob, self.advantage, self.base_log_prob = lp, adv, blp


def test_summarize_ref_mse():
    print("\n[ref_mse] _summarize_ref_mse")
    t = _probe_trainer(0.25, 0.05)
    chunks = [_Chunk(-0.010, +1.0, -0.06), _Chunk(-0.012, +1.0, -0.062),
              _Chunk(-0.030, -1.0, -0.08), _Chunk(-0.034, -1.0, -0.084)]
    s = t._summarize_ref_mse(chunks, compute_base=True)
    check("mean == mean(-ref_log_prob)", close(s["mean"], 0.0215, 1e-9))
    check("pos_mean uses only advantage > 0", close(s["pos_mean"], 0.011, 1e-9))
    check("neg_mean uses only advantage <= 0", close(s["neg_mean"], 0.032, 1e-9))
    check("ratio_ceiling_max == exp(max MSE_ref)",
          close(s["ratio_ceiling_max"], math.exp(0.034), 1e-9))
    check("ratio_ceiling_mean == mean(exp(MSE_ref)), not exp(mean)",
          close(s["ratio_ceiling_mean"],
                float(np.exp([0.010, 0.012, 0.030, 0.034]).mean()), 1e-9))
    check("log_base_ratio == ref_log_prob - base_log_prob (positive = better than base)",
          close(s["log_base_ratio_mean"], 0.05, 1e-9))
    check("ceiling is far below 1+clip_eps_high (upper clip unreachable)",
          s["ratio_ceiling_max"] < 1 + t.config.clip_eps_high)

    check("no base anchor -> no drift keys",
          not any("base" in k for k in
                  t._summarize_ref_mse([_Chunk(-0.02, 1.0)], compute_base=False)))
    check("empty chunk list -> None", t._summarize_ref_mse([], False) is None)
    check("all ref_log_prob None -> None",
          t._summarize_ref_mse([_Chunk(None, 1.0)], False) is None)
    mixed = t._summarize_ref_mse([_Chunk(None, 1.0), _Chunk(-0.02, 1.0)], False)
    check("chunks with ref_log_prob=None are excluded, not treated as 0",
          close(mixed["mean"], 0.02, 1e-9))
    s_nb = t._summarize_ref_mse(
        [_Chunk(-0.02, 1.0, -0.07), _Chunk(-0.03, -1.0, None)], compute_base=True)
    check("base_log_prob=None rows excluded from the drift stat",
          close(s_nb["log_base_ratio_mean"], 0.05, 1e-9))
    check("every emitted ref_mse value is finite",
          all(math.isfinite(v) for v in s.values()))


# ───────────────── 5. end-to-end TB emission via _log_metrics ────────────

class _RecordingWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, float(value), step))


def test_log_metrics_emits_new_tags():
    print("\n[logging] _log_metrics emits the new tags and survives every shape")
    from grpo_config import GRPOConfig

    jitter_diag = {
        "gap_pos": 0.0576, "gap_neg": 0.0023, "n_rows_pos": 4,
        "jacobian_fro_sq": 2.251,
        "gap_at_tau0": 0.09, "gap_at_tau1": 0.04, "tau0_value": 0.0,
        "tau1_value": 0.5, "headroom_multiplier": 3.88,
        "headroom_ref_only": 0.02, "headroom_with_jitter": 0.0776,
        "neg_clip_budget_used": 0.0276, "gap_fixed_rows_selfcheck": 1e-7,
    }
    full = {
        "n_updates": 12, "n_micro_batches": 12, "n_skipped_nonfinite": 0,
        "n_nonfinite_grad_steps": 0, "loss": 0.3, "clip_loss": 0.3,
        "kl_loss_last_iter": 5e-4, "clipfrac": 0.5, "mean_ratio": 0.94,
        "mean_log_ratio_abs": 0.06, "grad_norm_mean": 0.03, "grad_norm_max": 0.2,
        "ratio_max": 1.01, "ratio_min": 0.72, "n_pos_flipped_by_renorm": 0,
        "mean_ratio_fixed": 1.0, "mean_log_ratio_abs_fixed": 5e-4,
        "kl_loss_last_iter_fixed": 1e-7,
        "mean_ratio_jitter": 0.94, "mean_log_ratio_abs_jitter": 0.06,
        "kl_loss_last_iter_jitter": 5e-4,
        "clipfrac_fixed_pos": 0.0, "clipfrac_fixed_neg": 0.1,
        "clipfrac_jitter_pos": 1.0, "clipfrac_jitter_neg": 0.12,
        "mean_ratio_fixed_pos": 1.0, "mean_ratio_fixed_neg": 1.0,
        "mean_ratio_jitter_pos": 0.9440, "mean_ratio_jitter_neg": 0.9977,
        "mean_log_ratio_abs_fixed_pos": 4e-4, "mean_log_ratio_abs_fixed_neg": 5e-4,
        "mean_log_ratio_abs_jitter_pos": 0.058,
        "mean_log_ratio_abs_jitter_neg": 0.0023,
        "clipfrac_effective_pos": 0.0, "clipfrac_effective_neg": 0.12,
        "_jitter_diag": jitter_diag,
    }
    ref_mse = {
        "mean": 0.0215, "p10": 0.0106, "p50": 0.021, "p90": 0.0328, "max": 0.034,
        "ratio_ceiling_mean": 1.0218, "ratio_ceiling_max": 1.0346,
        "pos_mean": 0.011, "neg_mean": 0.032, "log_base_ratio_mean": 0.05,
        "log_base_ratio_p10": 0.04, "log_base_ratio_min": 0.03,
    }

    MUST = [
        "train/mean_ratio_jitter_pos", "train/mean_ratio_jitter_neg",
        "train/mean_ratio_fixed_pos", "train/mean_ratio_fixed_neg",
        "train/mean_log_ratio_abs_jitter_pos",
        "train/mean_log_ratio_abs_jitter_neg",
        "train/clipfrac_jitter_pos", "train/clipfrac_jitter_neg",
        "train/clipfrac_effective_pos", "train/clipfrac_effective_neg",
        "jitter/gap_pos", "jitter/gap_neg", "jitter/jacobian_fro_sq",
        "jitter/headroom_multiplier", "jitter/neg_clip_budget_used",
        "jitter/gap_at_tau0", "jitter/tau0_value",
        "jitter/gap_fixed_rows_selfcheck",
        "ref_mse/mean", "ref_mse/pos_mean", "ref_mse/ratio_ceiling_max",
        "ref_mse/log_base_ratio_mean",
    ]

    # Shapes that must all survive: full, skipped-update, no-jitter, no-ref-mse,
    # and a trainer that never ran __init__ (no _ref_mse_stats attribute).
    no_jit = {k: v for k, v in full.items()
              if "jitter" in k or k == "_jitter_diag" or "fixed" in k}
    vanilla = {k: v for k, v in full.items() if k not in no_jit}
    shapes = [
        ("full jitter run", full, ref_mse, True),
        ("skipped update (n_updates=0)", {**full, "n_updates": 0}, ref_mse, True),
        ("vanilla (jitter off)", vanilla, ref_mse, True),
        ("no ref_mse (ref pass produced nothing)", full, None, True),
        ("trainer without __init__ (no _ref_mse_stats attr)", full, None, False),
        ("empty update_stats", {}, ref_mse, True),
    ]

    for label, stats, refm, set_attr in shapes:
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.config = GRPOConfig(device="cpu", use_wandb=False)
        trainer.writer = _RecordingWriter()
        if set_attr:
            trainer._ref_mse_stats = refm
        try:
            trainer._log_metrics(
                5, {"success_rate": 0.5}, stats, lr=1.5e-5, iter_time=1.0,
                phase_times={"collect": 1.0}, lora_delta_norm=0.5,
            )
            raised = None
        except Exception as exc:  # noqa: BLE001 — the assertion IS "nothing raises"
            raised = f"{type(exc).__name__}: {exc}"
        check(f"{label}: logs without raising", raised is None, str(raised))
        if raised is not None:
            continue
        tags = [t for t, _, _ in trainer.writer.calls]
        dupes = sorted({t for t in tags if tags.count(t) > 1})
        check(f"{label}: no duplicate TB tags", not dupes, str(dupes))
        nonfinite = [t for t, v, _ in trainer.writer.calls if not math.isfinite(v)]
        check(f"{label}: no non-finite scalars", not nonfinite, str(nonfinite))
        check(f"{label}: no nested dict leaked as a scalar tag",
              "train/_jitter_diag" not in tags, str([t for t in tags if "diag" in t]))
        if label == "full jitter run":
            missing = [t for t in MUST if t not in tags]
            check("full run: every new tag is actually emitted",
                  not missing, f"missing {missing}")
            # The jitter/* block must NOT be gated on n_updates.
        if label == "skipped update (n_updates=0)":
            check("skipped update: jitter/* still emitted (ungated on n_updates)",
                  "jitter/gap_pos" in tags, str(sorted(t for t in tags if "jitter/" in t)))
            check("skipped update: ref_mse/* still emitted",
                  "ref_mse/mean" in tags)
            # This is the hole that let the dead-code bug through: the keys were
            # written into the early-return dict but the emitter was gated on
            # n_updates > 0, so they never reached TB.
            check("skipped update: train/clipfrac_effective_* STILL emitted "
                  "(populated by micro-batches that trained)",
                  "train/clipfrac_effective_pos" in tags
                  and "train/clipfrac_effective_neg" in tags,
                  str(sorted(t for t in tags if "effective" in t)))
            check("skipped update: train/loss suppressed (no fake zeros)",
                  "train/loss" not in tags)
        if label == "vanilla (jitter off)":
            check("vanilla: no jitter/* curves at all",
                  not [t for t in tags if t.startswith("jitter/")],
                  str([t for t in tags if t.startswith("jitter/")]))
            check("vanilla: no train/*_jitter_* curves",
                  not [t for t in tags if "_jitter" in t],
                  str([t for t in tags if "_jitter" in t]))
            check("vanilla: effective clipfrac STILL emitted (not a jitter metric)",
                  "train/clipfrac_effective_neg" in tags)
        if refm is None:
            check(f"{label}: no ref_mse/* curves",
                  not [t for t in tags if t.startswith("ref_mse/")],
                  str([t for t in tags if t.startswith("ref_mse/")]))


def test_wandb_path_excludes_nested_dict():
    print("\n[logging] wandb mirror handles the nested _jitter_diag")
    from grpo_config import GRPOConfig

    captured = {}

    class _FakeWandb:
        @staticmethod
        def log(d, step=None):
            captured.update(d)

    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.config = GRPOConfig(device="cpu", use_wandb=True)
    trainer.writer = _RecordingWriter()
    trainer._ref_mse_stats = {"pos_mean": 0.011}
    sys.modules["wandb"] = _FakeWandb  # type: ignore[assignment]
    try:
        trainer._log_metrics(
            3, {"success_rate": 0.5},
            {"n_updates": 4, "loss": 0.3,
             "_jitter_diag": {"gap_pos": 0.058, "jacobian_fro_sq": 2.25}},
            lr=1e-5, iter_time=1.0, phase_times=None, lora_delta_norm=0.1,
        )
        raised = None
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    finally:
        sys.modules.pop("wandb", None)
    check("wandb path does not raise on the nested dict", raised is None, str(raised))
    check("wandb: no train/_jitter_diag key", "train/_jitter_diag" not in captured,
          str(sorted(captured)))
    check("wandb: jitter/* mirrored", captured.get("jitter/gap_pos") == 0.058,
          str(sorted(k for k in captured if k.startswith("jitter/"))))
    check("wandb: ref_mse/* mirrored", captured.get("ref_mse/pos_mean") == 0.011)
    check("wandb: every logged value is a scalar",
          all(isinstance(v, (int, float)) for v in captured.values()),
          str({k: type(v).__name__ for k, v in captured.items()
               if not isinstance(v, (int, float))}))

    # The bare-attribute-read hole: on a trainer built without __init__, a plain
    # `self._ref_mse_stats` raises AttributeError, which the `except Exception:
    # pass` around the wandb block would swallow — silently dropping the ENTIRE
    # iteration's wandb payload rather than one metric.
    captured.clear()
    t2 = GRPOTrainer.__new__(GRPOTrainer)
    t2.config = GRPOConfig(device="cpu", use_wandb=True)
    t2.writer = _RecordingWriter()          # note: no _ref_mse_stats attribute
    sys.modules["wandb"] = _FakeWandb  # type: ignore[assignment]
    try:
        t2._log_metrics(
            4, {"success_rate": 0.5}, {"n_updates": 2, "loss": 0.2},
            lr=1e-5, iter_time=1.0, phase_times=None, lora_delta_norm=0.1,
        )
    finally:
        sys.modules.pop("wandb", None)
    check("wandb: trainer without _ref_mse_stats still logs the rest "
          "(no swallowed AttributeError)",
          "train/loss" in captured and "train/n_updates" in captured,
          f"captured only: {sorted(captured)}")
    check("wandb: no ref_mse/* keys when the attribute is absent",
          not [k for k in captured if k.startswith("ref_mse/")],
          str([k for k in captured if k.startswith("ref_mse/")]))

    # The unfiltered-comprehension hole: clipfrac_effective_* must reach wandb
    # ONLY via the finite-filtered block. If they are left in the generic
    # `update_stats.items()` comprehension, the unfiltered value lands in
    # log_dict first and the filtered block cannot remove it (dict.update cannot
    # un-set a key), so a non-finite scalar would poison the wandb chart on every
    # n_updates > 0 iteration while TB printed "dropped ... rather than poisoning".
    captured.clear()
    t4 = GRPOTrainer.__new__(GRPOTrainer)
    t4.config = GRPOConfig(device="cpu", use_wandb=True)
    t4.writer = _RecordingWriter()
    t4._ref_mse_stats = None
    sys.modules["wandb"] = _FakeWandb  # type: ignore[assignment]
    try:
        t4._log_metrics(
            6, {"success_rate": 0.5},
            {"n_updates": 3, "loss": 0.2,
             "clipfrac_effective_pos": float("inf"),
             "clipfrac_effective_neg": float("nan")},
            lr=1e-5, iter_time=1.0, phase_times=None, lora_delta_norm=0.1,
        )
    finally:
        sys.modules.pop("wandb", None)
    eff = {k: v for k, v in captured.items() if "effective" in k}
    check("wandb: non-finite clipfrac_effective_* is NOT logged on the "
          "n_updates > 0 path (excluded from the generic comprehension)",
          not eff, f"leaked {eff}")
    check("wandb: the finite siblings on that iteration still get through",
          "train/loss" in captured, str(sorted(captured)))

    # Same hole on the TB side must already be closed by _emit.
    t5 = GRPOTrainer.__new__(GRPOTrainer)
    t5.config = GRPOConfig(device="cpu", use_wandb=False)
    t5.writer = _RecordingWriter()
    t5._ref_mse_stats = None
    t5._log_metrics(
        6, {"success_rate": 0.5},
        {"n_updates": 3, "loss": 0.2,
         "clipfrac_effective_pos": float("inf"),
         "mean_ratio_jitter_pos": float("inf"),
         "mean_log_ratio_abs_jitter_pos": float("nan"),
         "clipfrac_jitter_pos": 0.5},
        lr=1e-5, iter_time=1.0, phase_times=None, lora_delta_norm=0.1,
    )
    tb = [(t, v) for t, v, _ in t5.writer.calls]
    check("TB: non-finite sign-split ratio metrics are filtered too "
          "(they are reachable via a bf16 exp overflow that the loss guard misses)",
          all(math.isfinite(v) for _, v in tb),
          str([t for t, v in tb if not math.isfinite(v)]))
    check("TB: the FINITE sign-split sibling still gets through",
          "train/clipfrac_jitter_pos" in [t for t, _ in tb],
          str(sorted(t for t, _ in tb if "jitter" in t)))

    # The step argument must be the iteration, not a constant.
    steps = {s for _, _, s in t5.writer.calls}
    check("TB: every scalar is written at the iteration step, not 0",
          steps == {6}, f"steps seen: {sorted(steps)}")

    # Non-finite values must be filtered, not logged: one inf/nan poisons wandb
    # chart autoscale for the remainder of the run (the reason this file already
    # filters phase_times, grad_norms and ratio_max/min).
    for writer_name, stats, refm in (
        ("jitter", {"n_updates": 2,
                    "_jitter_diag": {"gap_pos": float("inf"), "gap_neg": 0.002}}, None),
        ("ref_mse", {"n_updates": 2}, {"pos_mean": 0.01, "ratio_ceiling_max": float("nan")}),
    ):
        t3 = GRPOTrainer.__new__(GRPOTrainer)
        t3.config = GRPOConfig(device="cpu", use_wandb=False)
        t3.writer = _RecordingWriter()
        t3._ref_mse_stats = refm
        t3._log_metrics(9, {"success_rate": 0.5}, stats, lr=1e-5, iter_time=1.0,
                        phase_times=None, lora_delta_norm=0.1)
        vals = [(t, v) for t, v, _ in t3.writer.calls]
        check(f"non-finite {writer_name} scalar is dropped, not logged",
              all(math.isfinite(v) for _, v in vals),
              str([t for t, v in vals if not math.isfinite(v)]))
        check(f"the FINITE sibling {writer_name} scalar still gets through",
              any(writer_name in t for t, _ in vals),
              str([t for t, _ in vals if writer_name in t]))


# ────────── 6. the diagnostic really runs at theta == theta_ref ──────────

def test_effective_clipfrac_aggregation_values():
    """Assert the VALUE of clipfrac_effective_*, not just that the key exists.

    Mutation testing showed every prior check was presence-only, so three
    independent corruptions of the aggregation all survived: using minibatch
    count as the denominator instead of row count, using the wrong bucket's
    denominator, and breaking the `n_pos + n_neg == trained rows` invariant.

    Method: replace the production predicate with a forced pattern, which pins
    the aggregation independently of what ratios the harness happens to produce.
      - nothing dead  -> both fractions must be exactly 0.0
      - everything dead -> both must be exactly 1.0, which is only true if each
        bucket's denominator is its OWN row count (catches all three mutations)
      - only post-renorm-positive rows dead -> _pos == 1.0 and _neg == 0.0,
        which additionally pins the bucketing to the post-renorm sign
    """
    print("\n[clipfrac] effective clipfrac AGGREGATION values")
    import test_grad_accum as tga

    real_pred = train_grpo.clip_killed_gradient
    # UNBALANCED pos/neg split on purpose. With the default balanced sampler at
    # ratio 0.5 and mini_batch_size=4 every minibatch is exactly 2 pos / 2 neg,
    # so n_rows_pos_total == n_rows_neg_total and a mutation that divides the pos
    # numerator by the NEG denominator is invisible. At ratio 0.25 the buckets
    # are 1 and 3, so the denominators are distinguishable.
    jit = dict(jitter_pos=0.25, jitter_neg=0.05, jitter_paired=True,
               balanced_minibatch_positive_adv_ratio=0.25)

    def _run_with(pred):
        train_grpo.clip_killed_gradient = pred
        try:
            return tga.run_update(1, n_chunks=16, mb_size=4, epochs=1,
                                  config_overrides=jit).result
        finally:
            train_grpo.clip_killed_gradient = real_pred

    none_dead = _run_with(lambda ratio, s1, s2, lo, hi: torch.zeros_like(ratio, dtype=torch.bool))
    check("nothing dead -> clipfrac_effective_pos == 0.0",
          none_dead.get("clipfrac_effective_pos") == 0.0,
          str(none_dead.get("clipfrac_effective_pos")))
    check("nothing dead -> clipfrac_effective_neg == 0.0",
          none_dead.get("clipfrac_effective_neg") == 0.0,
          str(none_dead.get("clipfrac_effective_neg")))

    all_dead = _run_with(lambda ratio, s1, s2, lo, hi: torch.ones_like(ratio, dtype=torch.bool))
    check("everything dead -> clipfrac_effective_pos == 1.0 "
          "(pins the _pos denominator to its own row count)",
          all_dead.get("clipfrac_effective_pos") == 1.0,
          str(all_dead.get("clipfrac_effective_pos")))
    check("everything dead -> clipfrac_effective_neg == 1.0 "
          "(pins the _neg denominator, and pins n_pos + n_neg == trained rows)",
          all_dead.get("clipfrac_effective_neg") == 1.0,
          str(all_dead.get("clipfrac_effective_neg")))

    # surr1 = A_post * ratio, so sign(surr1) == sign(A_post) for ratio > 0.
    # Marking exactly those rows dead isolates the post-renorm-positive bucket.
    pos_dead = _run_with(lambda ratio, s1, s2, lo, hi: s1 > 0)
    check("only post-renorm-positive rows dead -> _pos == 1.0",
          pos_dead.get("clipfrac_effective_pos") == 1.0,
          str(pos_dead.get("clipfrac_effective_pos")))
    check("only post-renorm-positive rows dead -> _neg == 0.0 "
          "(pins bucketing to the POST-renorm sign)",
          pos_dead.get("clipfrac_effective_neg") == 0.0,
          str(pos_dead.get("clipfrac_effective_neg")))

    # The production callsite must pass (clip_eps_low, clip_eps_high) in that
    # order. With the default 0.2/0.2 a swap is invisible, so force lo != hi.
    seen_args: list[tuple[float, float]] = []

    def _spy(ratio, s1, s2, lo, hi):
        seen_args.append((lo, hi))
        return real_pred(ratio, s1, s2, lo, hi)

    train_grpo.clip_killed_gradient = _spy
    try:
        tga.run_update(1, n_chunks=16, mb_size=4, epochs=1,
                       config_overrides=dict(clip_eps_low=0.08, clip_eps_high=0.4))
    finally:
        train_grpo.clip_killed_gradient = real_pred
    check("production callsite passes (clip_eps_low, clip_eps_high) in order",
          seen_args and all(a == (0.08, 0.4) for a in seen_args),
          f"observed {sorted(set(seen_args))}")


def test_diagnostic_failure_is_isolated():
    """A raising diagnostic must cost the metric, once, not the iteration."""
    print("\n[jitter] a failing diagnostic is isolated and not retried")
    import test_grad_accum as tga

    calls = {"n": 0}
    real = GRPOTrainer._jitter_gap_diagnostics

    def _boom(self, **kw):
        calls["n"] += 1
        raise RuntimeError("synthetic diagnostic failure")

    GRPOTrainer._jitter_gap_diagnostics = _boom
    try:
        # gradient_accumulation_steps=4 is load-bearing for the last assertion:
        # it keeps n_updates == 0 across the first four micro-batches, so a
        # sentinel bug (setting jitter_diag back to None instead of {}) would
        # retry the failing diagnostic on each of them. At accum_k=1 the first
        # step fires immediately and the n_updates guard masks the bug.
        r = tga.run_update(4, n_chunks=16, mb_size=4, epochs=2,
                           config_overrides=dict(jitter_pos=0.25, jitter_neg=0.05,
                                                 jitter_paired=True))
        raised = None
    except Exception as exc:  # noqa: BLE001
        r, raised = None, f"{type(exc).__name__}: {exc}"
    finally:
        GRPOTrainer._jitter_gap_diagnostics = real
    check("a raising diagnostic does not propagate out of the update",
          raised is None, str(raised))
    check("the iteration still trains normally",
          r is not None and r.result.get("n_updates", 0) > 0,
          str(sorted(r.result)) if r else "no result")
    check("no _jitter_diag key is emitted after a failure",
          r is not None and "_jitter_diag" not in r.result,
          str(r.result.get("_jitter_diag")) if r else "")
    # The except path sets a {} sentinel rather than None, so the failure is not
    # retried on every subsequent minibatch of the iteration.
    check("the failure is attempted ONCE, not once per minibatch "
          "(the except path must set a non-None sentinel)",
          calls["n"] == 1, f"invoked {calls['n']} times")


def test_diagnostic_runs_strictly_before_any_step():
    """Functional check that the gap is measured pre-step, not post-step.

    Drives the REAL `_grpo_update_inner` through test_grad_accum's harness with
    jitter enabled, and records the trainable weight at every
    `compute_fm_log_prob` call. The diagnostic's calls are identifiable by
    `return_per_tau=True`; all of them must observe the weight still at its
    initial value, i.e. before any optimizer.step() moved it.

    This is the regression test for the gating: the block that builds
    `noise_for_input` (and hence hosts the diagnostic) is conditional on the
    minibatch containing jitter rows, so under jitter_paired=True an all-"fixed"
    first minibatch is possible. Without the `n_updates == 0` guard the
    measurement would silently slide to a post-step minibatch and report policy
    drift as gap.
    """
    print("\n[jitter] diagnostic is taken at theta == theta_ref")
    import test_grad_accum as tga

    # AdamW populates optimizer.state lazily on the FIRST step (exp_avg /
    # exp_avg_sq per param), so an empty state dict is a reliable, functional
    # "no step has fired yet" witness that needs no access to the local
    # n_updates counter.
    seen_states: list[int] = []
    real_method = GRPOTrainer._jitter_gap_diagnostics

    def _recording(self, **kw):
        seen_states.append(len(getattr(self, "optimizer", {}).state)
                           if hasattr(self, "optimizer") else -1)
        return real_method(self, **kw)

    jit = dict(jitter_pos=0.25, jitter_neg=0.05, jitter_paired=True)
    GRPOTrainer._jitter_gap_diagnostics = _recording
    try:
        r = tga.run_update(1, n_chunks=16, mb_size=4, epochs=1,
                           config_overrides=jit)
    finally:
        GRPOTrainer._jitter_gap_diagnostics = real_method

    check("diagnostic was invoked exactly once per iteration",
          len(seen_states) == 1, f"invocations: {len(seen_states)}")
    check("diagnostic ran BEFORE any optimizer step (AdamW state still empty)",
          seen_states == [0], f"optimizer.state sizes at call time: {seen_states}")

    check("real update with jitter produces a _jitter_diag entry",
          "_jitter_diag" in r.result, str(sorted(r.result)))
    check("real update emits the sign-split ratio metrics",
          "mean_ratio_jitter_pos" in r.result
          and "mean_ratio_jitter_neg" in r.result, str(sorted(r.result)))
    check("real update emits the effective clipfrac",
          "clipfrac_effective_pos" in r.result
          or "clipfrac_effective_neg" in r.result, str(sorted(r.result)))

    # The value-pinned harness surrogate has no tau/noise dependence, so the
    # honest gap is 0 — assert exactly that rather than something vacuous.
    d = r.result["_jitter_diag"]
    # gap_pos must exist (the balanced sampler guarantees >=1 positive row);
    # gap_neg is conditional on the first minibatch drawing a negative row at
    # mb_size=4, so assert on it only when present rather than letting a
    # `.get(..., 0.0)` default make the check pass on absence.
    check("value-pinned surrogate => gap_pos is exactly 0 (no spurious signal)",
          abs(d["gap_pos"]) < 1e-9,
          str({k: v for k, v in d.items() if k.startswith("gap")}))
    if "gap_neg" in d:
        check("value-pinned surrogate => gap_neg is exactly 0", abs(d["gap_neg"]) < 1e-9,
              str(d["gap_neg"]))
    check("fixed-row self-check key is present in paired mode",
          "gap_fixed_rows_selfcheck" in d, str(sorted(d)))
    check("fixed-row self-check is 0 for the value-pinned surrogate",
          abs(d["gap_fixed_rows_selfcheck"]) < 1e-9,
          str(d["gap_fixed_rows_selfcheck"]))

    # ── The case the n_updates == 0 guard exists for ──────────────────────
    # Force the FIRST minibatch to contain only "fixed" entries. The block that
    # hosts the diagnostic is conditional on the minibatch having jitter rows,
    # so without the guard the measurement would slide to a later, POST-step
    # minibatch and report policy drift as gap. With the guard it must either
    # still be pre-step or be skipped entirely.
    real_strat = GRPOTrainer._iter_stratified_minibatches
    real_bal = GRPOTrainer._iter_balanced_minibatches

    def _fixed_first(self, entries, rng, _real=None):
        fixed = [e for e in entries if e[1] == "fixed"]
        jitter = [e for e in entries if e[1] == "jitter"]
        mb = self.config.mini_batch_size
        if len(fixed) >= mb:
            ordered = fixed[:mb] + jitter + fixed[mb:]
        else:
            ordered = fixed + jitter
        for i in range(0, len(ordered), mb):
            yield ordered[i:i + mb]

    seen_states.clear()
    GRPOTrainer._jitter_gap_diagnostics = _recording
    GRPOTrainer._iter_stratified_minibatches = _fixed_first
    GRPOTrainer._iter_balanced_minibatches = _fixed_first
    try:
        r2 = tga.run_update(1, n_chunks=16, mb_size=4, epochs=1,
                            config_overrides=jit)
    finally:
        GRPOTrainer._jitter_gap_diagnostics = real_method
        GRPOTrainer._iter_stratified_minibatches = real_strat
        GRPOTrainer._iter_balanced_minibatches = real_bal

    # Verified separately that this really does produce a fixed-only minibatch 0,
    # with the first jitter-bearing minibatch at index 1 — i.e. AFTER a step at
    # accum_k=1. So without the guard the diagnostic WOULD have run post-step.
    check("all-fixed first minibatch: measurement is SKIPPED, not taken "
          "post-step (this is what the n_updates == 0 guard buys)",
          len(seen_states) == 0,
          f"diagnostic ran {len(seen_states)} time(s) with optimizer.state sizes "
          f"{seen_states}; a non-empty list here means it measured after a step")
    check("all-fixed first minibatch: no _jitter_diag emitted rather than a "
          "drift-contaminated one",
          "_jitter_diag" not in r2.result,
          f"result carries {r2.result.get('_jitter_diag')}")
    check("all-fixed first minibatch: run still completes normally",
          r2.result.get("n_updates", 0) > 0, str(sorted(r2.result)))

    # Behavioural, not source-text: the all-fixed-first-minibatch case above is
    # what pins the n_updates == 0 guard, and test_diagnostic_failure_is_isolated
    # pins the try/except. Earlier versions grepped train_grpo.py for the guard
    # string, which both false-positived on a semantically identical reformat and
    # could not detect a swap of two identically-named branches.

    # The n_updates == 0 early-return path must still carry the diagnostics: the
    # gap was measured pre-step so it is still valid, and a blown-up gap is the
    # likeliest CAUSE of landing there (large |log_ratio| -> bf16 exp overflow ->
    # non-finite loss).
    r_dropped = tga.run_update(
        1, n_chunks=16, mb_size=4, epochs=1, config_overrides=jit,
        nonfinite_grad=tuple(range(8)),          # every window dropped
    )
    check("all-windows-dropped iteration reports n_updates == 0",
          r_dropped.result.get("n_updates", 0) == 0, str(sorted(r_dropped.result)))
    check("no-step iteration STILL carries _jitter_diag",
          "_jitter_diag" in r_dropped.result, str(sorted(r_dropped.result)))
    # `and`, not `or`: with `or` a mutation dropping either one individually
    # survived (M11 / M33 in the audit's mutation table).
    check("no-step iteration STILL carries BOTH effective clipfracs "
          "(micro-batches did train)",
          "clipfrac_effective_pos" in r_dropped.result
          and "clipfrac_effective_neg" in r_dropped.result,
          str(sorted(r_dropped.result)))

    r_nofinite = tga.run_update(
        1, n_chunks=16, mb_size=4, epochs=1, config_overrides=jit,
        nonfinite=tuple(range(8)),               # every loss non-finite
    )
    check("all-losses-non-finite iteration omits the effective clipfrac "
          "(nothing trained, so no rows to average)",
          "clipfrac_effective_pos" not in r_nofinite.result
          and "clipfrac_effective_neg" not in r_nofinite.result,
          str(sorted(r_nofinite.result)))
    check("all-losses-non-finite iteration STILL carries _jitter_diag "
          "(measured before the guard fired)",
          "_jitter_diag" in r_nofinite.result, str(sorted(r_nofinite.result)))


# ───────────── 7. per-chunk gap survey (Stage 0 + Stage 1) ───────────────

class _SurveyChunk:
    def __init__(self, ep, idx, succ, ref_lp, adv, H, D):
        self.episode_idx, self.chunk_idx, self.episode_success = ep, idx, succ
        self.ref_log_prob, self.advantage = ref_lp, adv
        self.tau_samples = np.linspace(0.0, 0.75, 6).astype(np.float32)
        self.initial_noise = np.zeros((H, D), dtype=np.float32)


def _survey_trainer(size, n_eps=8, chunks_succ=6, chunks_fail=10, H=4, D=8,
                    gap_of=None):
    """Trainer whose _prepare_batch is stubbed and whose fm_log_prob returns a
    log-prob chosen so that gap_i == gap_of(chunk) exactly."""
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.config = SimpleNamespace(
        jitter_pos=0.25, jitter_neg=0.05, clip_eps_low=0.08, clip_eps_high=0.2,
        tau_centers=[0.0, 0.25, 0.35, 0.5, 0.6, 0.75], mini_batch_size=8,
        seed=67, per_chunk_gap_survey_size=size,
    )
    t.device = torch.device("cpu")
    t.iteration = 3
    import threading
    t._model_lock = threading.RLock()
    t.model = SimpleNamespace(action_head=None)

    chunks = []
    for ep in range(n_eps):
        succ = ep % 2 == 0
        n = chunks_succ if succ else chunks_fail
        for i in range(n):
            chunks.append(_SurveyChunk(ep, i, succ, -0.004, +1.0 if succ else -1.0, H, D))

    def _prep(entries):
        cs = [c for c, _ in entries]
        B = len(cs)
        return ({"initial_noise": torch.zeros(B, H, D),
                 "backbone_output": {"backbone_features": torch.zeros(B, 1, D)},
                 "state_features": torch.zeros(B, 2, D),
                 "embodiment_id": torch.zeros(B, dtype=torch.long),
                 "actions": torch.zeros(B, H, D),
                 "action_masks": torch.ones(B, H, D)}, cs)
    t._prepare_batch = _prep

    def _fake(**kw):
        cs = _fake.current
        # lp_jit = ref_log_prob - gap  =>  gap_i = ref_log_prob_i - lp_jit_i
        return torch.tensor([c.ref_log_prob - gap_of(c) for c in cs],
                            dtype=torch.float32)
    _fake.current = None
    orig_prep = t._prepare_batch

    def _prep_track(entries):
        bd, cs = orig_prep(entries); _fake.current = cs; return bd, cs
    t._prepare_batch = _prep_track
    return t, chunks, _fake


def test_per_chunk_gap_survey():
    print("\n[chunk_gap] Stage 1: per-chunk gap survey")
    real = train_grpo.compute_fm_log_prob

    # (a) disabled by default -> exactly None, zero cost
    t, chunks, fake = _survey_trainer(0, gap_of=lambda c: 0.05)
    check("size=0 (default) -> returns None, no work done",
          t._per_chunk_gap_survey(chunks) is None)

    # (b) gap arithmetic + CV + planted correlations
    #     plant: successes have SMALLER gap (wider basins), and gap RISES with
    #     position within the episode.
    def planted(c):
        base = 0.040 if c.episode_success else 0.060
        n = 5 if c.episode_success else 9
        return base + 0.020 * (c.chunk_idx / n)
    t, chunks, fake = _survey_trainer(200, gap_of=planted)
    train_grpo.compute_fm_log_prob = fake
    torch.manual_seed(99); before = torch.get_rng_state().clone()
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            out = t._per_chunk_gap_survey(chunks)
    finally:
        train_grpo.compute_fm_log_prob = real
    check("returns a dict when enabled", isinstance(out, dict), str(type(out)))
    check("consumes NO global RNG (uses its own generator)",
          torch.equal(before, torch.get_rng_state()))
    exp = [planted(c) for c in chunks]
    check("mean gap matches the planted values",
          close(out["mean"], float(np.mean(exp)), 0.05),
          f"{out['mean']:.5f} vs ~{np.mean(exp):.5f}")
    check("cv is reported (THE decision statistic)", "cv" in out, str(sorted(out)))
    check("cv is positive and finite",
          out["cv"] > 0 and math.isfinite(out["cv"]), str(out.get("cv")))
    check("probe_lambda == jitter_pos (single probe, not the per-sign split)",
          close(out["probe_lambda"], 0.25))
    check("successes have the smaller mean gap, as planted",
          out["mean_succ"] < out["mean_fail"],
          f"succ={out['mean_succ']:.5f} fail={out['mean_fail']:.5f}")
    check("r_outcome is NEGATIVE (success <-> wider basin), as planted",
          out["r_outcome"] < -0.3, f"{out.get('r_outcome')}")
    check("r_position is POSITIVE (gap rises with position), as planted",
          out["r_position"] > 0.2, f"{out.get('r_position')}")
    check("both outcome classes were sampled (stratification works)",
          "mean_succ" in out and "mean_fail" in out, str(sorted(out)))
    check("percentiles ordered p10 <= p50 <= p90 <= max",
          out["p10"] <= out["p50"] <= out["p90"] <= out["max"],
          str({k: out[k] for k in ("p10", "p50", "p90", "max")}))
    check("every emitted value is finite",
          all(math.isfinite(v) for v in out.values()),
          str({k: v for k, v in out.items() if not math.isfinite(v)}))

    # (c) a FLAT gap must produce CV ~ 0 -> the kill signal
    t, chunks, fake = _survey_trainer(200, gap_of=lambda c: 0.05)
    train_grpo.compute_fm_log_prob = fake
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            flat = t._per_chunk_gap_survey(chunks)
    finally:
        train_grpo.compute_fm_log_prob = real
    check("constant gap -> cv ~ 0 (the 'per-chunk is dead' signal)",
          flat["cv"] < 1e-6, f"{flat['cv']:.3e}")
    check("constant gap -> r_outcome absent or ~0 (no spurious correlation)",
          abs(flat.get("r_outcome", 0.0)) < 1e-6, str(flat.get("r_outcome")))

    # (d) degenerate inputs
    t, chunks, fake = _survey_trainer(200, gap_of=lambda c: 0.05)
    check("too few usable chunks -> None", t._per_chunk_gap_survey(chunks[:8]) is None)
    for c in chunks:
        c.ref_log_prob = None
    check("all ref_log_prob None -> None", t._per_chunk_gap_survey(chunks) is None)


def test_stage0_cv_and_chunk_gap_logging():
    print("\n[chunk_gap] Stage 0 CV + TB emission")
    from grpo_config import GRPOConfig
    # Stage 0: gap_pos_cv comes out of the existing minibatch diagnostic
    inp = _make_inputs(B=8, K=3, seed=13)
    head = _MinAtEpsHead(inp["actions"], inp["eps"], inp["timesteps"], c=0.7)
    t = _probe_trainer(0.25, 0.05, ref_pos=0.004)
    t.model = SimpleNamespace(action_head=head)
    pos = torch.zeros(8, dtype=torch.bool); pos[:6] = True
    lam = torch.where(pos, torch.full((8,), 0.25), torch.full((8,), 0.05))
    g = torch.Generator().manual_seed(5)
    xi = torch.randn(inp["K"], *inp["eps"].shape, generator=g)
    nfi = ((1 - lam * lam).sqrt()[None, :, None, None] * inp["eps"].unsqueeze(0)
           + lam[None, :, None, None] * xi)
    out = t._jitter_gap_diagnostics(
        ready_backbone=inp["backbone"], ready_state_features=inp["state_features"],
        ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
        ready_masks=inp["mask"], ready_noise=inp["eps"], timesteps=inp["timesteps"],
        noise_for_input=nfi, lam_row=lam, pos_adv_mask=pos,
        fixed_row_mask=torch.zeros(8, dtype=torch.bool),
        jitter_row_mask=torch.ones(8, dtype=torch.bool))
    for k in ("gap_pos_cv", "gap_pos_min", "gap_pos_max"):
        check(f"Stage 0 emits {k}", k in out, str(sorted(out)))
    check("gap_pos_min <= gap_pos <= gap_pos_max",
          out["gap_pos_min"] <= out["gap_pos"] <= out["gap_pos_max"],
          f"{out['gap_pos_min']:.5f} {out['gap_pos']:.5f} {out['gap_pos_max']:.5f}")
    check("gap_pos_cv is finite and non-negative",
          math.isfinite(out["gap_pos_cv"]) and out["gap_pos_cv"] >= 0,
          str(out["gap_pos_cv"]))

    # chunk_gap/* must reach TB, ungated on n_updates, filtered for non-finite
    for label, nupd in (("n_updates>0", 3), ("n_updates==0", 0)):
        tr = GRPOTrainer.__new__(GRPOTrainer)
        tr.config = GRPOConfig(device="cpu", use_wandb=False)
        tr.writer = _RecordingWriter()
        tr._ref_mse_stats = None
        tr._chunk_gap_stats = {"n": 256, "mean": 0.05, "cv": 0.11,
                              "r_outcome": -0.21, "bad": float("inf")}
        tr._log_metrics(4, {"success_rate": 0.5}, {"n_updates": nupd},
                        lr=1e-5, iter_time=1.0, phase_times=None, lora_delta_norm=0.1)
        tags = [t_ for t_, _, _ in tr.writer.calls]
        check(f"{label}: chunk_gap/* emitted", "chunk_gap/cv" in tags,
              str(sorted(x for x in tags if "chunk_gap" in x)))
        check(f"{label}: non-finite chunk_gap entry dropped",
              "chunk_gap/bad" not in tags, str(tags))
    tr2 = GRPOTrainer.__new__(GRPOTrainer)
    tr2.config = GRPOConfig(device="cpu", use_wandb=False)
    tr2.writer = _RecordingWriter(); tr2._ref_mse_stats = None
    tr2._log_metrics(4, {"success_rate": 0.5}, {"n_updates": 3},
                     lr=1e-5, iter_time=1.0, phase_times=None, lora_delta_norm=0.1)
    check("survey disabled / attr absent -> no chunk_gap/* curves",
          not [x for x, _, _ in tr2.writer.calls if x.startswith("chunk_gap/")])
    check("config default keeps the survey OFF (zero cost)",
          GRPOConfig(device="cpu").per_chunk_gap_survey_size == 0)


# ───────────────────────────── run ───────────────────────────────────────

if __name__ == "__main__":
    test_per_tau_contract()
    test_jitter_gap_arithmetic()
    test_jacobian_estimator_is_lambda_invariant()
    test_jitter_gap_edge_cases()
    test_effective_clipfrac_truth_table()
    test_effective_clipfrac_aggregation_values()
    test_summarize_ref_mse()
    test_log_metrics_emits_new_tags()
    test_wandb_path_excludes_nested_dict()
    test_diagnostic_runs_strictly_before_any_step()
    test_diagnostic_failure_is_isolated()
    test_per_chunk_gap_survey()
    test_stage0_cv_and_chunk_gap_logging()
    print()
    if FAILURES:
        print(f"{RED}{len(FAILURES)} test(s) FAILED:{RESET}")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(f"{GREEN}All tests passed.{RESET}")
