"""CPU tests for the trajectory-roughness constraint (the "jerk constraint").

Runs without a GPU, without MuJoCo and without the 3B model. Covers:

  1. `roughness_hf` calibration against the values quoted in jerk-constraint.md
     (constant 0, white 1, alternating 8/3) plus scale-freeness and the
     zero-energy epsilon guard.
  2. `implied_endpoint` reproduces the identity `a_hat = a + (1-tau)*r` exactly,
     which is the whole basis for targeting the endpoint.
  3. `pooled_hf`: energy-weighted aggregation that is robust to near-idle rows,
     exactly associative over batch splits, and closes the add-DC exploit.
  4. `build_continuous_action_dims` reproduces the PandaOmron action layout and
     excludes discrete / gated keys.
  5. `compute_fm_log_prob` return contract, BOTH smooth instruments (the 1-step
     endpoint at tau=0 and the last-step-differentiable multi-step chunk rollout),
     and their INVARIANCE to jitter, against an analytic stand-in action head.
  6. The real `_grpo_update_inner`: smooth_coef=0 is BIT-IDENTICAL to a run
     without the feature; smooth_coef>0 changes the loss; calibration only
     accumulates while n_updates == 0; anchors use the same divisor as clip_loss.
  7. `smooth_ref.json` round-trip and guard-key rejection.

NOTE on the value-pinned fake: it pins (R, M) per ROW, which is what lets the
on-path tests assert exact penalties -- but it also makes pooled HF and a mean of
per-minibatch ratios agree for ANY split, so an assertion about row-weighting is
unfalsifiable under it. `test_grad_accum.SMOOTH_ROW_VARYING_MOMENTS` opts into
per-row moments that separate the two, and test_on_path uses it for exactly that
check (verified by mutation: swapping the pooled divisor for a mean now fails).

Usage:  .venv/bin/python scripts/grpo/test_smoothness.py
"""

import inspect
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoothness import (  # noqa: E402
    DEFAULT_DISCRETE_ACTION_KEYS,
    DEFAULT_GATED_ACTION_KEYS,
    build_continuous_action_dims,
    describe_dim_selection,
    implied_endpoint,
    pooled_hf,
    roughness_hf,
    roughness_moments,
    second_difference,
)

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
_failures: list[str] = []


def check(label, cond, detail=""):
    if cond:
        print(f"  {GREEN}PASS{RESET}  {label}")
    else:
        print(f"  {RED}FAIL{RESET}  {label}" + (f"  --> {detail}" if detail else ""))
        _failures.append(label)


def approx(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


# ───────────────────────────────────────────────────────────────────────────────
# 1. HF calibration
# ───────────────────────────────────────────────────────────────────────────────

def test_hf_calibration():
    print("\n[hf] calibration against jerk-constraint.md section 1")
    torch.manual_seed(0)
    B, H, D = 20000, 16, 6

    def hf(x):
        return float(roughness_hf(x, detach_denominator=False).mean())

    const = torch.randn(B, 1, D).expand(B, H, D).contiguous()
    white = torch.randn(B, H, D)
    sgn = torch.where(torch.arange(H)[None, :, None] % 2 == 0, 1.0, -1.0)
    alt = torch.randn(B, 1, D) * sgn
    ramp = torch.arange(H, dtype=torch.float32)[None, :, None].expand(B, H, D)

    check("constant along h -> HF == 0", approx(hf(const), 0.0, 1e-6))
    check("linear ramp along h -> HF == 0 (D2 kills a straight line)",
          approx(hf(ramp), 0.0, 1e-6))
    check("white along h -> HF == 1", approx(hf(white), 1.0, 0.01),
          f"got {hf(white):.4f}")
    check("alternating +-c -> HF == 8/3", approx(hf(alt), 8 / 3, 0.01),
          f"got {hf(alt):.4f}")
    check("scale-free: HF(10*white) == HF(white)",
          approx(hf(white * 10), hf(white), 1e-3))

    # Monotone in high-frequency content at comparable energy.
    cn = const / const.pow(2).mean().sqrt()
    wn = white / white.pow(2).mean().sqrt()
    seq = [hf((1 - f) * cn + f * wn) for f in (0.0, 0.25, 0.5, 1.0)]
    check("monotone increasing as white content rises",
          all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)),
          f"got {[round(v, 4) for v in seq]}")

    # Zero-energy row: numerator is 0 too, so the eps guard yields 0, not NaN/inf.
    z = roughness_hf(torch.zeros(4, H, D))
    check("all-zero slice -> 0.0, finite (eps guard)",
          bool(torch.isfinite(z).all()) and approx(float(z.max()), 0.0, 1e-9))

    # Denominator detachment is what stops the model satisfying the term by
    # inflating M instead of smoothing the spectrum.
    u = torch.randn(5, H, D, requires_grad=True)
    roughness_hf(u, detach_denominator=True).sum().backward()
    g_det = u.grad.clone()
    u.grad = None
    roughness_hf(u, detach_denominator=False).sum().backward()
    check("detach_denominator changes the gradient (M term suppressed)",
          not torch.allclose(g_det, u.grad, atol=1e-9))

    # The exploit a live denominator opens is NOT radial rescaling -- HF is
    # scale-free, so a -> c*a leaves it exactly unchanged. It is adding a DC
    # (constant-along-h) offset: D2 annihilates a constant, so R is untouched
    # while M rises, and HF = R/(6M) falls. The model could then "satisfy" the
    # constraint by inflating low-frequency energy without smoothing anything.
    #
    # Directional derivative along "add delta to every h of a column" is
    # g.sum(dim=1). With M detached it must be EXACTLY zero, because the
    # (1,-2,1) stencil sums to zero so D2^T of anything sums to zero over h.
    a = (torch.randn(3, H, D) + 2.0).requires_grad_(True)   # nonzero column means
    roughness_hf(a, detach_denominator=False).sum().backward()
    dc_live = a.grad.sum(dim=1)
    a.grad = None
    roughness_hf(a, detach_denominator=True).sum().backward()
    dc_det = a.grad.sum(dim=1)
    check("live denominator: adding DC energy REDUCES HF (the exploit exists)",
          float(dc_live.max()) < -1e-9,
          f"max DC sensitivity {float(dc_live.max()):.3e} should be < 0")
    check("detached denominator: DC sensitivity is exactly 0 (exploit closed)",
          float(dc_det.abs().max()) < 1e-6,
          f"got {float(dc_det.abs().max()):.3e}")
    # And confirm scale-freeness really does make radial rescaling a non-exploit.
    a2 = torch.randn(3, H, D)
    check("radial rescaling leaves HF unchanged (so it is not the exploit)",
          torch.allclose(roughness_hf(a2, detach_denominator=False),
                         roughness_hf(a2 * 7.0, detach_denominator=False),
                         atol=1e-5))

    check("second_difference rejects H < 3",
          _raises(lambda: second_difference(torch.zeros(1, 2, 3)), ValueError))
    check("roughness_hf rejects non-3D input",
          _raises(lambda: roughness_hf(torch.zeros(4, 4)), ValueError))


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def _raise_message(fn, exc):
    """The message of the expected exception, or None if it did not raise.

    Used where the assertion is not merely "it rejected" but "it rejected FOR
    THE RIGHT REASON" — a guard that hard-fails on every key at once would pass
    a bare `_raises` while telling the operator nothing.
    """
    try:
        fn()
    except exc as e:
        return str(e)
    except Exception:
        return None
    return None


# ───────────────────────────────────────────────────────────────────────────────
# 2. The endpoint identity
# ───────────────────────────────────────────────────────────────────────────────

def test_endpoint_identity():
    print("\n[endpoint] a_hat = x_tau + (1-tau)v  ==  a + (1-tau)r")
    torch.manual_seed(1)
    B, H, D = 7, 16, 12
    a = torch.randn(B, H, D)
    eps = torch.randn(B, H, D)
    r = torch.randn(B, H, D) * 0.1
    worst = 0.0
    for tau_val in (0.0, 0.25, 0.35, 0.5, 0.6, 0.75, 0.999):
        t = torch.full((B,), tau_val)
        x_tau = (1 - t)[:, None, None] * eps + t[:, None, None] * a
        v = (a - eps) + r                      # v_theta = target + residual
        got = implied_endpoint(x_tau, v, t)
        want = a + (1 - t)[:, None, None] * r
        worst = max(worst, float((got - want).abs().max()))
    check("identity holds to fp32 precision at every tau", worst < 1e-5,
          f"max abs err {worst:.2e}")

    # eps cancels structurally: a_hat is independent of eps given (a, r).
    t = torch.full((B,), 0.0)
    v1 = (a - eps) + r
    eps2 = torch.randn(B, H, D) * 5.0
    v2 = (a - eps2) + r
    e1 = implied_endpoint((1 - t)[:, None, None] * eps + t[:, None, None] * a, v1, t)
    e2 = implied_endpoint((1 - t)[:, None, None] * eps2 + t[:, None, None] * a, v2, t)
    check("a_hat is invariant to eps at fixed (a, r) -- no white-noise floor",
          torch.allclose(e1, e2, atol=1e-5))

    # A perfect model (r == 0) gives a_hat == a exactly.
    t = torch.full((B,), 0.4)
    x = (1 - t)[:, None, None] * eps + t[:, None, None] * a
    check("perfect model (r=0) -> a_hat == a",
          torch.allclose(implied_endpoint(x, a - eps, t), a, atol=1e-5))


# ───────────────────────────────────────────────────────────────────────────────
# 3. The hinge
# ───────────────────────────────────────────────────────────────────────────────

def test_pooled():
    print("\n[pooled] energy-weighted HF: robust to idle rows, batch-invariant")
    torch.manual_seed(11)
    H, D = 16, 6

    def chunk(scale):
        a = torch.cumsum(torch.randn(1, H, D), 1)
        a = a / a.pow(2).mean().sqrt() * (0.0432 ** 0.5) * scale
        r = torch.randn(1, H, D)
        r = r / r.pow(2).mean().sqrt() * (0.00047 ** 0.5)
        return a + r

    normal = [chunk(1.0) for _ in range(63)]
    u_clean = torch.cat(normal, 0)
    u_idle = torch.cat(normal + [chunk(0.0)], 0)      # one motionless chunk

    p_clean = float(pooled_hf(roughness_moments(u_clean), detach_denominator=False))
    p_idle = float(pooled_hf(roughness_moments(u_idle), detach_denominator=False))
    mean_ratio_clean = float(roughness_hf(u_clean, detach_denominator=False).mean())
    mean_ratio_idle = float(roughness_hf(u_idle, detach_denominator=False).mean())
    check("an idle row shifts POOLED HF by <2%",
          abs(p_idle - p_clean) / p_clean < 0.02,
          f"{p_clean:.5f} -> {p_idle:.5f}")
    check("the same idle row shifts the MEAN-OF-RATIOS far more (why pooling)",
          abs(mean_ratio_idle - mean_ratio_clean) / mean_ratio_clean
          > 5 * abs(p_idle - p_clean) / p_clean,
          f"mean {mean_ratio_clean:.5f} -> {mean_ratio_idle:.5f}")

    # Batch-size invariance is an EXACT algebraic property of a ratio of sums:
    # pooling a batch equals pooling its parts and recombining. (Comparing
    # independently drawn batches would only measure sampling noise.)
    big = torch.cat([chunk(1.0) for _ in range(96)], 0)
    whole = float(pooled_hf(roughness_moments(big), detach_denominator=False))
    parts = [roughness_moments(big[i:i + 8]) for i in range(0, 96, 8)]
    recombined = float(
        pooled_hf(torch.cat(parts, 0), detach_denominator=False))
    check("pooling is exactly associative over batch splits",
          approx(whole, recombined, 1e-6), f"{whole:.8f} vs {recombined:.8f}")
    sums = torch.stack([p.sum(dim=0) for p in parts]).sum(dim=0)
    check("pooled HF == sum(R)/(6*sum(M)) over the whole batch",
          approx(whole, float(sums[0] / (6.0 * sums[1])), 1e-6))

    # calibration values
    B = 4000
    w = torch.randn(B, H, D)
    c = torch.randn(B, 1, D).expand(B, H, D).contiguous()
    sgn = torch.where(torch.arange(H)[None, :, None] % 2 == 0, 1.0, -1.0)
    alt = torch.randn(B, 1, D) * sgn
    ph = lambda x: float(pooled_hf(roughness_moments(x), detach_denominator=False))
    check("pooled: constant -> 0", approx(ph(c), 0.0, 1e-6))
    check("pooled: white -> 1", approx(ph(w), 1.0, 0.02), f"got {ph(w):.4f}")
    check("pooled: alternating -> 8/3", approx(ph(alt), 8 / 3, 0.02))
    check("pooled: scale-free", approx(ph(w * 10), ph(w), 1e-3))

    # detached denominator closes the add-DC exploit, pooled form
    a = (torch.randn(6, H, D) + 2.0).requires_grad_(True)
    pooled_hf(roughness_moments(a), detach_denominator=False).backward()
    dc_live = a.grad.sum(dim=1)
    a.grad = None
    pooled_hf(roughness_moments(a), detach_denominator=True).backward()
    dc_det = a.grad.sum(dim=1)
    check("pooled+live denominator: adding DC reduces HF (exploit exists)",
          float(dc_live.max()) < -1e-9)
    check("pooled+detached: DC sensitivity exactly 0 (exploit closed)",
          float(dc_det.abs().max()) < 1e-6)

    check("pooled_hf rejects a non-[B,2] input",
          _raises(lambda: pooled_hf(torch.zeros(4, 3)), ValueError))
    check("roughness_moments rejects non-3D input",
          _raises(lambda: roughness_moments(torch.zeros(4, 4)), ValueError))
    z = pooled_hf(roughness_moments(torch.zeros(5, H, D)), detach_denominator=False)
    check("all-zero batch -> finite 0 (eps guard)",
          bool(torch.isfinite(z)) and approx(float(z), 0.0, 1e-9))


# ───────────────────────────────────────────────────────────────────────────────
# 4. Dim selection
# ───────────────────────────────────────────────────────────────────────────────

def test_dim_selection():
    print("\n[dims] PandaOmron action layout from the checkpoint ordering")
    keys = ["end_effector_position", "end_effector_rotation", "gripper_close",
            "base_motion", "control_mode"]
    kd = {"end_effector_position": 3, "end_effector_rotation": 3,
          "gripper_close": 1, "base_motion": 4, "control_mode": 1}

    dims, kept, total = build_continuous_action_dims(keys, kd)
    check("total valid dims == 12 (3+3+1+4+1; a quaternion would give 13)",
          total == 12, f"got {total}")
    check("default C == position + rotation -> [0..5]",
          dims == [0, 1, 2, 3, 4, 5], f"got {dims}")
    check("default kept keys", kept == ["end_effector_position",
                                        "end_effector_rotation"], f"got {kept}")
    check("gripper_close (idx 6) excluded -- a grasp IS a step function",
          6 not in dims)
    check("control_mode (idx 11) excluded", 11 not in dims)
    check("base_motion (7..10) excluded by default (gated by control_mode)",
          not set(range(7, 11)) & set(dims))

    dims_bm, kept_bm, _ = build_continuous_action_dims(keys, kd, include_gated=True)
    check("include_gated admits base_motion -> 10 dims",
          dims_bm == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], f"got {dims_bm}")
    check("discrete keys stay excluded even with include_gated",
          6 not in dims_bm and 11 not in dims_bm)

    # Prefixed keys (robocasa's converters use body./hand. prefixes).
    pkeys = ["body.end_effector_position", "body.end_effector_rotation",
             "hand.gripper_close", "body.base_motion", "body.control_mode"]
    pkd = {f"{k}": kd[k.split('.')[-1]] for k in pkeys}
    pdims, _, ptotal = build_continuous_action_dims(pkeys, pkd)
    check("prefixed keys give the same layout",
          pdims == dims and ptotal == total)

    # A different ordering must produce different indices -- proves nothing is
    # hardcoded and the order really comes from modality_keys.
    rkeys = ["gripper_close", "end_effector_position", "end_effector_rotation",
             "control_mode", "base_motion"]
    rdims, _, _ = build_continuous_action_dims(rkeys, kd)
    check("reordered modality_keys shifts the indices (order is not hardcoded)",
          rdims == [1, 2, 3, 4, 5, 6], f"got {rdims}")

    check("missing key_dims entry raises KeyError",
          _raises(lambda: build_continuous_action_dims(keys, {"gripper_close": 1}),
                  KeyError))
    check("non-positive dim raises",
          _raises(lambda: build_continuous_action_dims(
              ["a"], {"a": 0}), ValueError))
    check("empty C raises rather than silently constraining nothing",
          _raises(lambda: build_continuous_action_dims(
              ["gripper_close"], {"gripper_close": 1}), ValueError))

    table = describe_dim_selection(keys, kd, dims, horizon=16)
    check("banner table marks in-C and excluded rows",
          "in C" in table and "excluded" in table and "PARTIAL" not in table)


# ───────────────────────────────────────────────────────────────────────────────
# 5. compute_fm_log_prob return contract
# ───────────────────────────────────────────────────────────────────────────────

class _StubHead(torch.nn.Module):
    """Analytic stand-in for Gr00tN1d6ActionHead: no DiT, no backbone.

    `action_decoder` returns `scale * x_t + bias`, so `pred_velocity` is a known
    function of the noisy trajectory and both smooth instruments can be checked
    by hand. With `bias=0` the Euler rollout is a pure geometric progression:
    `x_{i+1} = (1 + dt*scale) x_i`, so the 4-step chunk is
    `(1 + dt*scale)^N * eps` in closed form.

    `num_inference_timesteps` is read by `inference_schedule`, so it is the one
    knob that decides the rollout length here -- deliberately settable so a test
    can prove nothing is hardcoded to 4.
    """

    def __init__(self, H_pad, D_pad, scale=0.5, bias=0.0, n_steps=4):
        super().__init__()
        self.num_timestep_buckets = 1000
        self.num_inference_timesteps = n_steps
        self.scale = scale
        self.bias = bias
        self._H, self._D = H_pad, D_pad
        # Per-forward record of the (t, x) each call saw, so a test can assert
        # the schedule the rollout actually walked rather than trusting the
        # closed form to have exercised the right t values.
        self.calls: list = []

        class _Cfg:
            add_pos_embed = False
            use_alternate_vl_dit = False
        self.config = _Cfg()

    def action_encoder(self, noisy, t_disc, emb):
        self.calls.append((t_disc.clone(), noisy.detach().clone()))
        return noisy            # pass through; model() ignores it anyway

    def model(self, **kw):
        return kw["hidden_states"], None

    def action_decoder(self, model_output, emb):
        return self.scale * model_output + self.bias


def test_fm_return_contract():
    print("\n[fm] compute_fm_log_prob return contract and smooth path")
    from fm_log_prob import compute_fm_log_prob

    torch.manual_seed(2)
    B, H_pad, D_pad, K = 3, 20, 8, 4
    H_valid, dims = 6, torch.tensor([0, 1, 2])
    head = _StubHead(H_pad, D_pad)
    actions = torch.randn(B, H_pad, D_pad)
    noise = torch.randn(B, H_pad, D_pad)
    mask = torch.zeros(B, H_pad, D_pad)
    mask[:, :H_valid, :4] = 1.0
    ts = torch.rand(K, B) * 0.9
    # state_features must concat with action_features along dim=1
    common = dict(
        action_head=head,
        backbone_output={"backbone_features": torch.randn(B, 5, 4)},
        state_features=torch.zeros(B, 0, D_pad),
        embodiment_id=torch.zeros(B, dtype=torch.long),
        actions=actions, action_mask=mask, timesteps=ts, noise=noise,
        n_samples=K,
    )

    plain = compute_fm_log_prob(**common)
    check("no flags -> bare [B] tensor", isinstance(plain, torch.Tensor)
          and tuple(plain.shape) == (B,))

    lp, per_tau = compute_fm_log_prob(**common, return_per_tau=True)
    check("return_per_tau contract unchanged: (log_probs, [K,B])",
          tuple(per_tau.shape) == (K, B) and torch.allclose(lp, plain, atol=1e-6))

    lp2, (mom, ep_mom) = compute_fm_log_prob(**common, smooth_dims=dims,
                                             smooth_horizon=H_valid)
    check("smooth only -> (log_probs, (moments, endpoint_moments))",
          tuple(mom.shape) == (B, 2) and tuple(ep_mom.shape) == (B, 2),
          f"got {tuple(mom.shape)}, {tuple(ep_mom.shape)}")
    check("adding the smooth output does NOT change log_probs",
          torch.allclose(lp2, plain, atol=0.0),
          f"max delta {float((lp2 - plain).abs().max()):.3e}")

    lp3, pt3, (mom3, _ep3) = compute_fm_log_prob(**common, return_per_tau=True,
                                                 smooth_dims=dims,
                                                 smooth_horizon=H_valid)
    check("both flags -> (log_probs, per_tau, (mom, ep_mom)) in that order",
          torch.allclose(pt3, per_tau, atol=0.0)
          and tuple(mom3.shape) == (B, 2))

    # --- The chunk instrument (default) --------------------------------------
    # The stub's velocity at x is scale*x, so one Euler step is
    # x -> x + dt*scale*x = (1 + dt*scale)*x, and N steps is a pure power.
    from fm_log_prob import inference_schedule
    sched, dt = inference_schedule(head)
    check("schedule is derived from num_inference_timesteps, not hardcoded",
          sched == [0.0, 0.25, 0.5, 0.75] and approx(dt, 0.25, 1e-12),
          f"got {sched}, dt={dt}")
    head8 = _StubHead(H_pad, D_pad, n_steps=8)
    s8, dt8 = inference_schedule(head8)
    check("a different num_inference_timesteps gives a different schedule",
          len(s8) == 8 and approx(dt8, 0.125, 1e-12) and approx(s8[1], 0.125, 1e-12),
          f"got {s8}, dt={dt8}")

    want_chunk = noise * (1.0 + dt * head.scale) ** len(sched)
    want_mom = roughness_moments(
        want_chunk[:, :H_valid].index_select(2, dims).float())
    check("chunk moments match a closed-form 4-step Euler rollout",
          torch.allclose(mom, want_mom, atol=1e-5),
          f"max delta {float((mom - want_mom).abs().max()):.3e}")

    # The rollout's forward VALUE must equal a plain no-grad rollout: last-step
    # differentiation changes the GRAPH, never the number.
    with torch.no_grad():
        x = noise.clone()
        for t_val in sched:
            x = x + dt * (head.scale * x)
    plain_mom = roughness_moments(x[:, :H_valid].index_select(2, dims).float())
    check("last-step-differentiable value == plain no-grad rollout value",
          torch.allclose(mom, plain_mom, atol=0.0),
          f"max delta {float((mom - plain_mom).abs().max()):.3e}")

    # The rollout must WALK the production schedule, not merely end where it
    # would. Bucketized t is what the DiT sees: int(t * num_timestep_buckets).
    head.calls.clear()
    compute_fm_log_prob(**common, smooth_dims=dims, smooth_horizon=H_valid)
    smooth_calls = head.calls[K:]      # the K-loop's forwards come first
    got_t = [int(c[0][0]) for c in smooth_calls]
    want_t = [int(t * head.num_timestep_buckets) for t in sched]
    check("the rollout visits exactly the production timestep buckets",
          got_t == want_t, f"got {got_t} want {want_t}")
    check("the rollout costs num_inference_timesteps DiT forwards",
          len(smooth_calls) == len(sched),
          f"got {len(smooth_calls)} forwards for {len(sched)} steps")

    # The FREE endpoint byproduct: the rollout's first step IS v(eps, t=0).
    want_ep = roughness_moments(
        (noise + head.scale * noise)[:, :H_valid].index_select(2, dims).float())
    check("endpoint moments are the tau=0 implied endpoint eps + v(eps,0)",
          torch.allclose(ep_mom, want_ep, atol=1e-5),
          f"max delta {float((ep_mom - want_ep).abs().max()):.3e}")
    check("the endpoint byproduct carries NO gradient (it is monitoring only)",
          not ep_mom.requires_grad)

    # --- The endpoint instrument reproduces the PREVIOUS behaviour -----------
    lp_e, (mom_e, ep_e) = compute_fm_log_prob(
        **common, smooth_dims=dims, smooth_horizon=H_valid,
        smooth_instrument="endpoint")
    t0 = torch.zeros(B)
    v0 = head.scale * noise                     # the stub's velocity at x = eps
    legacy = roughness_moments(
        (noise + v0)[:, :H_valid].index_select(2, dims))
    check("smooth_instrument='endpoint' reproduces the pre-change moments",
          torch.allclose(mom_e, legacy, atol=0.0),
          f"max delta {float((mom_e - legacy).abs().max()):.3e}")
    check("under 'endpoint' the constrained and monitoring pairs are the same",
          mom_e is ep_e)
    check("'endpoint' does not change log_probs either",
          torch.allclose(lp_e, plain, atol=0.0))
    check("the two instruments give DIFFERENT numbers (not a silent alias)",
          not torch.allclose(mom, mom_e, atol=1e-6),
          f"chunk {mom[:, 0].tolist()} vs endpoint {mom_e[:, 0].tolist()}")
    check("an unknown smooth_instrument raises",
          _raises(lambda: compute_fm_log_prob(
              **common, smooth_dims=dims, smooth_horizon=H_valid,
              smooth_instrument="residual"), ValueError))

    # THE critical property, for BOTH instruments: the smooth output must be
    # INVARIANT to jitter, because it is computed on its own clean forward(s)
    # from the original eps rather than reusing the K-loop's jittered one.
    # Without this the jitter Jacobian response dominates HF (measured
    # 0.000347 -> 0.790 at tau=0) and the constraint is a no-op.
    K_ = K
    nfi = torch.randn(K_, B, H_pad, D_pad) * 0.5 + noise.unsqueeze(0)
    _, (mom_j, ep_j) = compute_fm_log_prob(**common, noise_for_input=nfi,
                                           smooth_dims=dims,
                                           smooth_horizon=H_valid)
    check("chunk moments are IDENTICAL with and without jitter (clean rollout)",
          torch.allclose(mom, mom_j, atol=0.0),
          f"max delta {float((mom - mom_j).abs().max()):.3e}")
    check("endpoint moments are IDENTICAL with and without jitter",
          torch.allclose(ep_mom, ep_j, atol=0.0))
    _, (mom_ej, _) = compute_fm_log_prob(**common, noise_for_input=nfi,
                                         smooth_dims=dims,
                                         smooth_horizon=H_valid,
                                         smooth_instrument="endpoint")
    check("the endpoint INSTRUMENT is jitter-invariant too",
          torch.allclose(mom_e, mom_ej, atol=0.0))

    check("smooth_dims without smooth_horizon raises",
          _raises(lambda: compute_fm_log_prob(**common, smooth_dims=dims),
                  ValueError))
    # The smooth pass never reads `timesteps`, so it must work without them.
    _out = compute_fm_log_prob(**{**common, "timesteps": None},
                               smooth_dims=dims, smooth_horizon=H_valid)
    check("smooth path does NOT require explicit timesteps (own schedule)",
          isinstance(_out, tuple) and tuple(_out[1][0].shape) == (B, 2))
    # smooth_no_grad must give identical values with no graph, on both paths.
    _, (m_ng, ep_ng) = compute_fm_log_prob(**common, smooth_dims=dims,
                                           smooth_horizon=H_valid,
                                           smooth_no_grad=True)
    check("smooth_no_grad yields identical chunk moments with no graph",
          torch.allclose(m_ng, mom, atol=0.0) and not m_ng.requires_grad)
    check("smooth_no_grad yields identical endpoint moments with no graph",
          torch.allclose(ep_ng, ep_mom, atol=0.0) and not ep_ng.requires_grad)
    _, (m_ng_e, _) = compute_fm_log_prob(**common, smooth_dims=dims,
                                         smooth_horizon=H_valid,
                                         smooth_no_grad=True,
                                         smooth_instrument="endpoint")
    check("smooth_no_grad works for the endpoint instrument too",
          torch.allclose(m_ng_e, mom_e, atol=0.0) and not m_ng_e.requires_grad)
    check("smooth_horizon beyond the padded horizon raises",
          _raises(lambda: compute_fm_log_prob(
              **common, smooth_dims=dims, smooth_horizon=H_pad + 1), ValueError))
    check("out-of-range smooth_dims raises",
          _raises(lambda: compute_fm_log_prob(
              **common, smooth_dims=torch.tensor([D_pad]),
              smooth_horizon=H_valid), ValueError))


# ───────────────────────────────────────────────────────────────────────────────
# 5b. The last-step-differentiable rollout: value exact, gradient localized
# ───────────────────────────────────────────────────────────────────────────────

class _StepIdentifiableHead(torch.nn.Module):
    """A head whose every Euler step uses its OWN parameter.

    Step `i` returns `w[i] * x`, selected by the bucketized timestep. That makes
    the four steps separately identifiable in the gradient: only the parameter
    belonging to the LAST step may receive a non-zero grad, and each earlier
    one must be zero (or None) -- which is exactly the claim
    `_smooth_chunk_rollout` makes.
    """

    def __init__(self, n_steps=4, D_pad=4):
        super().__init__()
        self.num_timestep_buckets = 1000
        self.num_inference_timesteps = n_steps
        # Distinct values so a mixed-up step ordering shows up in the VALUE too.
        self.w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(0.5 + 0.25 * i)) for i in range(n_steps)
        ])
        self._step_of_bucket = {
            int((i / n_steps) * self.num_timestep_buckets): i
            for i in range(n_steps)
        }
        self._cur = 0

        class _Cfg:
            add_pos_embed = False
            use_alternate_vl_dit = False
        self.config = _Cfg()

    def action_encoder(self, noisy, t_disc, emb):
        # Route on the bucketized timestep, the same value the real DiT
        # conditions its AdaLayerNorm on.
        self._cur = self._step_of_bucket.get(int(t_disc[0]), 0)
        return noisy

    def model(self, **kw):
        return kw["hidden_states"], None

    def action_decoder(self, model_output, emb):
        return self.w[self._cur] * model_output


def test_rollout_gradient_localization():
    print("\n[rollout] value is exact; gradient reaches ONLY the last step")
    from fm_log_prob import _smooth_chunk_rollout, inference_schedule

    torch.manual_seed(5)
    B, H_pad, D_pad, N = 3, 12, 4, 4
    head = _StepIdentifiableHead(n_steps=N, D_pad=D_pad)
    eps = torch.randn(B, H_pad, D_pad)
    sched, dt = inference_schedule(head)

    def velocity(x, t):
        # Mirrors _dit_velocity's contract: bucketize, encode, decode.
        t_disc = (t * head.num_timestep_buckets).long()
        return head.action_decoder(head.action_encoder(x, t_disc, None), None)

    chunk, v_first = _smooth_chunk_rollout(velocity, eps, head)

    # 1. VALUE: identical to a plain no-grad rollout, to the bit.
    with torch.no_grad():
        x = eps.clone()
        for t_val in sched:
            t = torch.full((B,), float(t_val))
            x = x + dt * velocity(x, t)
    check("rollout value is bit-identical to a plain 4-step no-grad rollout",
          torch.equal(chunk, x),
          f"max delta {float((chunk - x).abs().max()):.3e}")

    # And to the closed form, so the test cannot pass by both being wrong.
    closed = eps.clone()
    for i in range(N):
        closed = closed * (1.0 + dt * float(head.w[i]))
    check("... and to the closed-form product of per-step factors",
          torch.allclose(chunk, closed, atol=1e-6),
          f"max delta {float((chunk - closed).abs().max()):.3e}")

    # 2. GRADIENT: only the LAST step's parameter is reached.
    chunk.pow(2).sum().backward()
    grads = [None if p.grad is None else float(p.grad.abs().sum())
             for p in head.w]
    check("earlier steps' parameters receive NO gradient",
          all(g is None or g == 0.0 for g in grads[:-1]),
          f"got {grads}")
    check("the LAST step's parameter receives a non-zero gradient",
          grads[-1] is not None and grads[-1] > 0.0, f"got {grads}")

    # 3. The first step's velocity is returned for the free endpoint metric, and
    #    it is detached (it came from the no_grad leg).
    check("v_first is v(eps, t=0), the endpoint instrument's velocity",
          torch.allclose(v_first, float(head.w[0]) * eps, atol=1e-6))
    check("v_first carries no graph", not v_first.requires_grad)

    # 4. The bias is quantified, not hidden: with the gradient restricted to the
    #    final step, the retained magnitude is a fraction of the full rollout's.
    #    Recompute with every step differentiable and compare.
    for p in head.w:
        p.grad = None
    x_full = eps.clone()
    for t_val in sched:
        t = torch.full((B,), float(t_val))
        x_full = x_full + dt * velocity(x_full, t)
    x_full.pow(2).sum().backward()
    full = [float(p.grad.abs().sum()) for p in head.w]
    check("a fully-differentiated rollout DOES reach every step (bias is real)",
          all(g > 0.0 for g in full), f"got {full}")
    check("the retained last-step gradient matches the full rollout's on that "
          "step (only the earlier terms are dropped)",
          approx(full[-1], grads[-1], 1e-4),
          f"last-step-only {grads[-1]:.6f} vs full {full[-1]:.6f}")

    # 5. N=1 is a degenerate but legal schedule: the single step IS the last.
    head1 = _StepIdentifiableHead(n_steps=1, D_pad=D_pad)

    def velocity1(x, t):
        t_disc = (t * head1.num_timestep_buckets).long()
        return head1.action_decoder(head1.action_encoder(x, t_disc, None), None)

    c1, v1 = _smooth_chunk_rollout(velocity1, eps, head1)
    c1.pow(2).sum().backward()
    check("N=1 rollout is differentiable through its only step",
          head1.w[0].grad is not None and float(head1.w[0].grad.abs()) > 0.0)
    check("N=1 still returns the first velocity for the endpoint metric",
          v1 is not None and torch.allclose(v1.detach(),
                                            float(head1.w[0]) * eps, atol=1e-6))

    # 6. BUCKET FIDELITY. The DiT conditions on int(t * num_timestep_buckets),
    # and bf16 has 8 mantissa bits -- too coarse for 0.75, which bucketizes to
    # 752 against production's 750. A two-bucket offset is a DIFFERENT
    # AdaLayerNorm conditioning from the sampler's, i.e. a trajectory the robot
    # never executes: the exact failure this instrument exists to fix. The
    # rollout therefore builds t in float64 regardless of the batch dtype.
    for n in (4, 8):
        h = _StepIdentifiableHead(n_steps=n, D_pad=D_pad)
        seen = []

        def vel(x, t, _h=h, _seen=seen):
            _seen.append(int((t * _h.num_timestep_buckets).long()[0]))
            return torch.zeros_like(x)

        # bf16 eps, the production dtype -- the rollout must NOT inherit it for t.
        _smooth_chunk_rollout(vel, eps.to(torch.bfloat16), h)
        sched_n, _ = inference_schedule(h)
        want = [int(t * h.num_timestep_buckets) for t in sched_n]   # Python float
        check(f"N={n}: buckets match production exactly, even from a bf16 batch",
              seen == want, f"got {seen} want {want}")
    # And prove the hazard is real, so the fp64 choice is not cargo-culted.
    bad = int((torch.tensor([0.75], dtype=torch.bfloat16) * 1000).long()[0])
    check("(the hazard is real: bf16 0.75 would bucketize to 752, not 750)",
          bad == 752, f"got {bad}")


# ───────────────────────────────────────────────────────────────────────────────
# 6. smooth_ref.json round-trip / guard key
# ───────────────────────────────────────────────────────────────────────────────

def test_ref_persistence():
    print("\n[persist] smooth_ref.json round-trip and guard-key rejection")
    import train_grpo as tg

    class _Cfg:
        tau_centers = [0.0, 0.25, 0.35, 0.5, 0.6, 0.75]
        embodiment_tag = "ROBOCASA_PANDA_OMRON"
        model_path = "nvidia/GR00T-N1.6-3B"
        smooth_coef = 0.15
        smooth_hf_ref_scale = 15.0
        smooth_calib_min_rows = 1
        smooth_instrument = "chunk"
        resume_from = None
        jitter_pos = 0.25
        jitter_neg = 0.05
        jitter_paired = False
        env_names = ["robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env"]

    t = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t.config = _Cfg()
    t.device = torch.device("cpu")
    t.smooth_active = True
    t._smooth_dims_list = [0, 1, 2, 3, 4, 5]
    t._smooth_horizon = 16
    t._smooth_schedule = (0.0, 0.25, 0.5, 0.75)
    t._smooth_schedule_dt = 0.25
    t._smooth_calib_iter = 1
    t._smooth_ref_source = "calibrated"
    t._smooth_ref_scale_applied = 15.0
    t._smooth_hf_ref = torch.tensor(0.0212)

    with tempfile.TemporaryDirectory() as d:
        ck = Path(d)
        t._save_smooth_ref(ck)
        check("smooth_ref.json written", (ck / tg.SMOOTH_REF_FILENAME).exists())
        loaded = t._load_smooth_ref(ck)
        check("round-trip preserves hf_ref",
              torch.allclose(loaded, t._smooth_hf_ref, atol=1e-9))

        payload = json.loads((ck / tg.SMOOTH_REF_FILENAME).read_text())
        check("guard records the full key set",
              set(payload["guard"]) == {"tau_centers", "jitter_std", "dims",
                                        "horizon", "embodiment_tag",
                                        "model_path", "jitter_pos", "jitter_neg",
                                        "jitter_paired", "instrument",
                                        "sampler_steps", "sampler_dt"},
              f"got {sorted(payload['guard'])}")
        check("the guard records the instrument and the sampler schedule",
              payload["guard"]["instrument"] == "chunk"
              and payload["guard"]["sampler_steps"] == 4
              and approx(payload["guard"]["sampler_dt"], 0.25, 1e-12),
              f"got {payload['guard'].get('instrument')!r} "
              f"steps={payload['guard'].get('sampler_steps')} "
              f"dt={payload['guard'].get('sampler_dt')}")
        check("env_names recorded OUTSIDE the hard-fail guard (warn-only)",
              "env_names" in payload and "env_names" not in payload["guard"])
        t._smooth_ref_scale_applied = 15.0
        t.config.smooth_hf_ref_scale = 9.0     # live config differs from baked-in
        t._save_smooth_ref(ck)
        _p2 = json.loads((ck / tg.SMOOTH_REF_FILENAME).read_text())
        check("sidecar records the BAKED-IN scale, not the live config value",
              approx(_p2.get("hf_ref_scale"), 15.0, 1e-9),
              f"got {_p2.get('hf_ref_scale')} (live config is 9.0)")
        check("hf_ref / recorded_scale reconstructs the base HF correctly",
              approx(_p2["hf_ref"] / _p2["hf_ref_scale"], 0.0212 / 15.0, 1e-9))
        t.config.smooth_hf_ref_scale = 15.0
        t._save_smooth_ref(ck)
        check("provenance recorded: source + the scale actually baked in",
              payload.get("hf_ref_source") == "calibrated"
              and approx(payload.get("hf_ref_scale"), 15.0, 1e-9),
              f"got source={payload.get('hf_ref_source')!r} "
              f"scale={payload.get('hf_ref_scale')!r}")
        t2 = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
        t2.config = _Cfg(); t2.device = torch.device("cpu")
        t2.smooth_active = True
        t2._smooth_dims_list = [0, 1, 2, 3, 4, 5]; t2._smooth_horizon = 16
        t2._smooth_schedule = (0.0, 0.25, 0.5, 0.75); t2._smooth_schedule_dt = 0.25
        t2._load_smooth_ref(ck)
        check("a resume INHERITS the baked-in scale, not this run's config",
              approx(t2._smooth_ref_scale_applied, 15.0, 1e-9)
              and t2._smooth_calib_iter == 1,
              f"scale={t2._smooth_ref_scale_applied} iter={t2._smooth_calib_iter}")
        check("jitter_std recorded from the named constant",
              approx(payload["guard"]["jitter_std"], tg.TAU_JITTER_STD, 1e-12))

        # Each guard field independently rejects.
        for field, bad in (
            ("tau_centers", [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]),
            ("jitter_std", 0.05),
            ("dims", [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]),
            ("horizon", 8),
            ("embodiment_tag", "GR1"),
            ("model_path", "some/other/model"),
            ("jitter_pos", 0.05),
            ("jitter_neg", 0.0),
            ("jitter_paired", True),
            ("instrument", "endpoint"),
            ("sampler_steps", 8),
            ("sampler_dt", 0.125),
        ):
            p2 = json.loads(json.dumps(payload))
            p2["guard"][field] = bad
            (ck / tg.SMOOTH_REF_FILENAME).write_text(json.dumps(p2))
            check(f"guard mismatch on {field!r} hard-fails",
                  _raises(lambda: t._load_smooth_ref(ck), RuntimeError))

        # THE instrument mismatch specifically. It matters more than the others
        # because the two instruments' BASE VALUES are similar (chunk 0.00141 vs
        # endpoint 0.00157), so a stale sidecar loads with no numeric red flag
        # while thresholding at 15x the wrong quantity.
        p_inst = json.loads(json.dumps(payload))
        p_inst["guard"]["instrument"] = "endpoint"
        (ck / tg.SMOOTH_REF_FILENAME).write_text(json.dumps(p_inst))
        msg = _raise_message(lambda: t._load_smooth_ref(ck), RuntimeError)
        check("an endpoint-calibrated ref is REJECTED for a chunk run",
              msg is not None and "instrument" in msg, f"got {msg!r}")
        # ... and symmetrically, a chunk-calibrated ref for an endpoint run.
        (ck / tg.SMOOTH_REF_FILENAME).write_text(json.dumps(payload))
        t_ep = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
        t_ep.config = _Cfg(); t_ep.config.smooth_instrument = "endpoint"
        t_ep.device = torch.device("cpu"); t_ep.smooth_active = True
        t_ep._smooth_dims_list = [0, 1, 2, 3, 4, 5]; t_ep._smooth_horizon = 16
        t_ep._smooth_schedule = (0.0, 0.25, 0.5, 0.75)
        t_ep._smooth_schedule_dt = 0.25
        msg2 = _raise_message(lambda: t_ep._load_smooth_ref(ck), RuntimeError)
        check("a chunk-calibrated ref is REJECTED for an endpoint run",
              msg2 is not None and "instrument" in msg2, f"got {msg2!r}")
        # Sanity: the same instrument loads cleanly, so the guard is not
        # rejecting everything.
        t_ep.config.smooth_instrument = "chunk"
        check("matching instrument still loads",
              t_ep._load_smooth_ref(ck) is not None)

        (ck / tg.SMOOTH_REF_FILENAME).unlink()
        check("absent file -> None (so the caller can decide)",
              t._load_smooth_ref(ck) is None)

    # Off state must not write anything.
    t.smooth_active = False
    with tempfile.TemporaryDirectory() as d:
        t._save_smooth_ref(Path(d))
        check("feature off -> no sidecar written",
              not (Path(d) / tg.SMOOTH_REF_FILENAME).exists())


# ───────────────────────────────────────────────────────────────────────────────
# 7. Calibration finalize
# ───────────────────────────────────────────────────────────────────────────────

def test_calibration_finalize():
    print("\n[calib] hf_ref = scale * mean(base HF), and the no-sample path")
    import train_grpo as tg

    class _Cfg:
        tau_centers = [0.0, 0.25, 0.35, 0.5, 0.6, 0.75]
        smooth_hf_ref_scale = 4.0
        smooth_coef = 0.15
        smooth_calib_min_rows = 1
        smooth_instrument = "chunk"

    t = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t.config = _Cfg()
    t.smooth_active = True
    t._smooth_hf_ref = None
    # Accumulator holds pooled (sum R, sum M). base = R/(6M).
    t._smooth_calib_sum = torch.tensor([1.2, 2.0])
    t._smooth_calib_n = 10
    t._smooth_calib_rows = 640
    out = t._smooth_finalize_calibration(1)
    want = 4.0 * (1.2 / (6.0 * 2.0))
    check("hf_ref == scale * pooled R/(6M)",
          approx(float(t._smooth_hf_ref), want, 1e-6),
          f"got {float(t._smooth_hf_ref):.6f} want {want:.6f}")
    check("calibration iteration recorded", t._smooth_calib_iter == 1)
    check("accumulator released after freezing", t._smooth_calib_sum is None)
    check("returns the scalar reference for TB",
          out.get("smooth_calib_samples") == 10
          and approx(out["smooth_hf_ref"], want, 1e-6))

    # A window below smooth_calib_min_rows must defer rather than freeze a
    # value measured on too few rows (an 8-row window spreads 0.64x-1.57x).
    class _CfgBig(_Cfg):
        smooth_calib_min_rows = 10_000
    t_small = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t_small.config = _CfgBig(); t_small.smooth_active = True
    t_small._smooth_hf_ref = None
    t_small._smooth_calib_sum = torch.tensor([1.2, 2.0])
    t_small._smooth_calib_n = 2; t_small._smooth_calib_rows = 16
    out_small = t_small._smooth_finalize_calibration(1)
    check("too few rows -> defers instead of freezing",
          t_small._smooth_hf_ref is None
          and out_small.get("smooth_calib_rows") == 16,
          f"hf_ref={t_small._smooth_hf_ref} out={out_small}")
    check("the deferred accumulator is PRESERVED for the next iteration",
          t_small._smooth_calib_sum is not None
          and float(t_small._smooth_calib_sum[1]) == 2.0)

    check("second call is a no-op (reference already frozen)",
          t._smooth_finalize_calibration(2) == {})

    # No samples: stays pending and warns rather than freezing a bogus 0.
    t2 = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t2.config = _Cfg()
    t2.smooth_active = True
    t2._smooth_hf_ref = None
    t2._smooth_calib_sum = torch.zeros(2)
    t2._smooth_calib_n = 0
    t2._smooth_calib_rows = 0
    check("zero samples -> stays uncalibrated (no zero threshold)",
          t2._smooth_finalize_calibration(1) == {}
          and t2._smooth_hf_ref is None)

    # Feature off: never calibrates.
    t3 = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t3.config = _Cfg()
    t3.smooth_active = False
    check("feature off -> finalize is a no-op",
          t3._smooth_finalize_calibration(1) == {})


# ───────────────────────────────────────────────────────────────────────────────
# 8. Real _grpo_update_inner: off-path bit-identity and on-path effect
# ───────────────────────────────────────────────────────────────────────────────

def test_update_integration():
    print("\n[update] real _grpo_update_inner: off is bit-identical, on bites")
    import test_grad_accum as tga

    # `run_update` in test_grad_accum drives the production
    # _grpo_update/_grpo_update_inner on CPU with analytic stand-ins. Reuse it so
    # the integration is exercised through the real code path.
    if not hasattr(tga, "run_update"):
        check("test_grad_accum exposes run_update for reuse", False,
              "helper missing -- integration test skipped")
        return

    kw = dict(k=1, n_groups=2, n_chunks=16, mb_size=4, seed=7)
    base = tga.run_update(**kw)
    off = tga.run_update(**kw, config_overrides=dict(smooth_coef=0.0))
    check("smooth_coef=0 leaves every train stat bit-identical",
          _same_stats(base, off), _stat_delta(base, off))
    check("smooth_coef=0 emits no smooth_* metrics at all",
          not any(str(key).startswith("smooth_") for key in _stats_of(off)),
          f"found {[k for k in _stats_of(off) if str(k).startswith('smooth_')]}")
    # The class-level OFF defaults are what make this work without setup().
    import train_grpo as tg
    check("GRPOTrainer.smooth_active defaults False at class level",
          tg.GRPOTrainer.smooth_active is False)
    check("class-level OFF defaults exist for every attr the update reads",
          all(hasattr(tg.GRPOTrainer, a) for a in
              ("_smooth_dims", "_smooth_horizon", "_smooth_hf_ref",
               "_smooth_calib_sum", "_smooth_calib_n")))
    # The strongest form of "off is a no-op": identical final weights.
    wb = torch.as_tensor(base.w_final).flatten()
    wo = torch.as_tensor(off.w_final).flatten()
    check("smooth_coef=0 leaves the final weights bit-identical",
          torch.equal(wb, wo),
          f"max delta {float((wb - wo).abs().max()):.3e}")

    # smooth_instrument must not leak into the OFF path either: the new default
    # is "chunk", which would add num_inference_timesteps forwards if the gate
    # were ever keyed on the instrument rather than on smooth_coef.
    for inst in ("chunk", "endpoint"):
        r = tga.run_update(
            **kw, config_overrides=dict(smooth_coef=0.0, smooth_instrument=inst))
        check(f"smooth_coef=0 with instrument={inst!r} is still bit-identical",
              _same_stats(base, r)
              and torch.equal(wb, torch.as_tensor(r.w_final).flatten()),
              _stat_delta(base, r))

    # NO EXTRA WORK and NO EXTRA RNG on the off path. `run_update` installs its
    # own fake over `train_grpo.compute_fm_log_prob`, so spying there is not
    # possible -- instead spy on `pooled_hf` / `roughness_moments`, which the
    # trainer imports and calls ONLY from inside the smooth block. Either firing
    # with smooth_coef=0 means the `smooth_active` gate leaked.
    import train_grpo as tg
    calls = {"pooled_hf": 0, "roughness_moments": 0}
    real_pooled, real_moments = tg.pooled_hf, tg.roughness_moments
    tg.pooled_hf = lambda *a, **k: (
        calls.__setitem__("pooled_hf", calls["pooled_hf"] + 1)
        or real_pooled(*a, **k))
    tg.roughness_moments = lambda *a, **k: (
        calls.__setitem__("roughness_moments", calls["roughness_moments"] + 1)
        or real_moments(*a, **k))
    try:
        torch.manual_seed(1234)
        _ = torch.randn(1)                 # anchor the stream
        tga.run_update(**kw, config_overrides=dict(smooth_coef=0.0))
        rng_off = torch.randn(4).tolist()
        off_calls = dict(calls)
        # And prove the spy WOULD have caught a leak, by running the on path.
        calls.update(pooled_hf=0, roughness_moments=0)
        _on_path_run(smooth_coef=0.5)
        on_calls = dict(calls)
    finally:
        tg.pooled_hf, tg.roughness_moments = real_pooled, real_moments
    check("smooth_coef=0 does no roughness work at all (the gate holds)",
          off_calls == {"pooled_hf": 0, "roughness_moments": 0},
          f"got {off_calls}")
    check("... and the same spy DOES fire when the feature is on",
          on_calls["pooled_hf"] > 0, f"got {on_calls}")

    # `smooth_coef=0.0` IS the GRPOConfig default, so comparing that override
    # against no-override at all would run two identical code paths and could
    # never fail -- a tautology dressed as an RNG check. What the trainer-side
    # arms below DO bind is narrower, and the comment must not overstate it:
    # `_on_path_run` swaps in a FAKE compute_fm_log_prob, so no DiT forward and no
    # Euler rollout runs in either arm. Comparing their RNG streams therefore
    # tests the TRAINER-side smooth block (the hinge, the accumulators, the
    # calibration branch) and says nothing about the smooth forwards themselves.
    # Those are covered separately, against the analytic stand-in head, in
    # test_smooth_forwards_draw_no_rng().
    torch.manual_seed(1234)
    _ = torch.randn(1)
    _on_path_run(smooth_coef=0.5)
    rng_on = torch.randn(4).tolist()
    check("the trainer-side smooth block draws no RNG (same stream as coef=0)",
          rng_off == rng_on, f"off {rng_off} vs on {rng_on}")
    # Guard the guard: if the two arms were secretly the same code path, this
    # would also pass. Prove the harness can tell them apart at all.
    check("... and that comparison is not vacuous (the arms differ)",
          off_calls != on_calls, f"off {off_calls} vs on {on_calls}")


def _stats_of(res):
    """`run_update` returns a _Run dataclass whose `.result` is the stats dict."""
    if isinstance(res, tuple):
        return res[0]
    return getattr(res, "result", res)


def _same_stats(a, b):
    sa, sb = _stats_of(a), _stats_of(b)
    keys = set(sa) & set(sb)
    for k in keys:
        va, vb = sa[k], sb[k]
        if isinstance(va, float) and isinstance(vb, float):
            if not (math.isnan(va) and math.isnan(vb)) and va != vb:
                return False
        elif va != vb:
            return False
    return True


def _stat_delta(a, b):
    sa, sb = _stats_of(a), _stats_of(b)
    diffs = [
        f"{k}: {sa[k]!r} vs {sb[k]!r}"
        for k in sorted(set(sa) & set(sb)) if sa[k] != sb[k]
    ]
    return "; ".join(diffs[:4])


# ───────────────────────────────────────────────────────────────────────────────
# 9. ON-path through the real _grpo_update_inner
# ───────────────────────────────────────────────────────────────────────────────

def _on_path_run(**over):
    """Drive the real _grpo_update_inner with the constraint ACTIVE.

    `test_grad_accum.run_update` builds the trainer via `GRPOTrainer.__new__`, so
    `_setup_smoothness()` never runs and `smooth_active` stays at the class-level
    False -- which means passing `smooth_coef>0` through `config_overrides` alone
    leaves the feature off. Its `_fake_fm_log_prob` also returns a bare tensor, so
    the `current_log_probs, smooth_hf_kb = fm_out` unpack would raise.

    This helper patches both: it injects the smoothness attributes onto the
    trainer class and swaps in a smooth-aware fake log-prob that returns a
    differentiable [K, B] HF built from the real primitives. Everything else --
    the divisor choice, the hinge, the calibration gate, the metrics -- is the
    production code path.
    """
    import test_grad_accum as tga
    import train_grpo as tg

    K = 6
    dims = torch.tensor([0, 1, 2, 3, 4, 5])
    horizon = 16

    saved = {a: getattr(tg.GRPOTrainer, a) for a in
             ("smooth_active", "_smooth_dims", "_smooth_horizon",
              "_smooth_eef_pos_dims", "_smooth_hf_ref", "_smooth_calib_sum",
              "_smooth_calib_n", "_smooth_n_exec", "_smooth_executed_stats")}
    tg.GRPOTrainer.smooth_active = True
    tg.GRPOTrainer._smooth_dims = dims
    tg.GRPOTrainer._smooth_horizon = horizon
    tg.GRPOTrainer._smooth_eef_pos_dims = torch.tensor([0, 1, 2])
    tg.GRPOTrainer._smooth_hf_ref = over.pop("_hf_ref", torch.tensor(0.0))
    tg.GRPOTrainer._smooth_calib_sum = over.pop("_calib_sum", None)
    tg.GRPOTrainer._smooth_calib_n = 0
    tg.GRPOTrainer._smooth_calib_rows = 0
    tg.GRPOTrainer._smooth_n_exec = 8
    # The shared harness's `actions` are [B, 1, 1] — too short for a horizon-16
    # second difference — so the real `_smooth_executed_stats` degrades to
    # (None, None) there. A caller that wants the executed-metric PLUMBING
    # exercised passes a stub returning known constants; the method's own
    # arithmetic is pinned separately in test_executed_metrics().
    _exec_stub = over.pop("_executed_stub", None)
    if _exec_stub is not None:
        tg.GRPOTrainer._smooth_executed_stats = _exec_stub
    try:
        kw = dict(k=1, n_groups=2, n_chunks=16, mb_size=4, seed=7)
        kw.update(over.pop("run_kw", {}))
        return tga.run_update(**kw, config_overrides=over)
    finally:
        for a, v in saved.items():
            setattr(tg.GRPOTrainer, a, v)


def test_on_path():
    print("\n[on-path] the constraint ACTIVE through the real _grpo_update_inner")
    import train_grpo as tg

    # The fake pins the pooled moments to R=1.2, M=2.0 -> pooled HF = 1.2/12 = 0.1
    # exactly, at any batch size, with a real gradient through R.
    HF_PIN = 0.1

    off = _on_path_run(smooth_coef=0.0)
    on = _on_path_run(smooth_coef=0.5)
    s_off, s_on = _stats_of(off), _stats_of(on)

    check("smooth_coef>0 emits smooth_* metrics (feature really engaged)",
          any(k.startswith("smooth_") for k in s_on),
          f"keys: {sorted(k for k in s_on if k.startswith('smooth_'))}")
    # Row-weighted (sum R / 6 sum M), not a mean of per-minibatch ratios.
    #
    # This needs the fake's ROW-VARYING moments to be observable at all: with
    # every row pinned to the same (R, M) the two statistics agree for ANY split,
    # so the check passed under either implementation -- a mutation to
    # mean-of-ratios was demonstrated to survive it. The fake loads M on the FIRST
    # row of each minibatch only, which makes the R/M mix depend on batch SIZE;
    # with the [4,4,2] split from n_chunks=10/mb_size=4 the two statistics then
    # differ and the expected value is computable in closed form.
    import test_grad_accum as _tga
    _tga.SMOOTH_ROW_VARYING_MOMENTS = True
    try:
        uneq = _stats_of(_on_path_run(
            smooth_coef=0.5, run_kw=dict(k=1, n_groups=1, n_chunks=10, mb_size=4)))
    finally:
        _tga.SMOOTH_ROW_VARYING_MOMENTS = False

    def _expected(sizes):
        # (pooled, mean-of-ratios) for M alternating 2.0/0.5 within each batch.
        pooled_r = pooled_m = 0.0
        ratios = []
        for n in sizes:
            r = 1.2 * n
            m = sum(2.0 if i == 0 else 0.5 for i in range(n))
            pooled_r += r
            pooled_m += m
            ratios.append(r / (6.0 * m))
        return pooled_r / (6.0 * pooled_m), sum(ratios) / len(ratios)

    want_pooled, want_mean = _expected([4, 4, 2])
    check("pooled and mean-of-ratios actually DIFFER here (check can fail)",
          abs(want_pooled - want_mean) > 1e-6,
          f"pooled {want_pooled:.6f} vs mean-of-ratios {want_mean:.6f}")
    check("hf_mean is row-weighted across unequal minibatch sizes",
          approx(uneq.get("smooth_hf_mean", -1), want_pooled, 1e-6),
          f"got {uneq.get('smooth_hf_mean')} want pooled {want_pooled:.6f} "
          f"(a per-minibatch mean would give {want_mean:.6f})")
    check("measured pooled HF equals the pinned value",
          approx(s_on.get("smooth_hf_mean", -1), HF_PIN, 1e-6),
          f"got {s_on.get('smooth_hf_mean')}")
    check("smooth_coef>0 changes train/loss vs smooth_coef=0",
          s_off.get("loss") != s_on.get("loss"),
          f"{s_off.get('loss')!r} vs {s_on.get('loss')!r}")
    wa = torch.as_tensor(off.w_final).flatten()
    wb = torch.as_tensor(on.w_final).flatten()
    check("smooth_coef>0 changes the final weights (a real gradient flows)",
          not torch.equal(wa, wb),
          f"max delta {float((wa - wb).abs().max()):.3e}")
    check("smooth_loss == coef * (HF - hf_ref), exactly",
          approx(s_on.get("smooth_loss", -1), 0.5 * HF_PIN, 1e-6),
          f"got {s_on.get('smooth_loss')} want {0.5 * HF_PIN}")
    check("active_frac == 1.0 when the pooled HF exceeds the threshold",
          approx(s_on.get("smooth_active_frac", 0.0), 1.0, 1e-9))
    check("smooth_loss scales linearly with smooth_coef",
          approx(_stats_of(_on_path_run(smooth_coef=1.0))["smooth_loss"],
                 2.0 * s_on["smooth_loss"], 1e-6))

    # --- The new instrument-provenance and monitoring metrics ----------------
    # Under the default "chunk" instrument, `hf_mean` (whatever is constrained)
    # and `chunk_hf_mean` (the chunk, named explicitly) are the same number.
    check("chunk_hf_mean is emitted and equals hf_mean under 'chunk'",
          approx(s_on.get("smooth_chunk_hf_mean", -1), HF_PIN, 1e-6)
          and approx(s_on["smooth_chunk_hf_mean"], s_on["smooth_hf_mean"], 0.0),
          f"got {s_on.get('smooth_chunk_hf_mean')} vs "
          f"hf_mean {s_on.get('smooth_hf_mean')}")
    # The fake pins the endpoint moments to R=0.9, M=2.0 -> HF 0.075, distinct
    # from the constrained 0.1, so a mixed-up wiring would be visible.
    check("endpoint_hf_mean is emitted from the FREE tau=0 byproduct",
          approx(s_on.get("smooth_endpoint_hf_mean", -1), 0.9 / 12.0, 1e-6),
          f"got {s_on.get('smooth_endpoint_hf_mean')} want {0.9/12.0}")
    check("endpoint_hf_mean is DISTINCT from hf_mean (not the same wire)",
          s_on["smooth_endpoint_hf_mean"] != s_on["smooth_hf_mean"])

    ep_run = _stats_of(_on_path_run(smooth_coef=0.5, smooth_instrument="endpoint"))
    check("under 'endpoint' the chunk_hf_mean key is ABSENT (no chunk exists)",
          "smooth_chunk_hf_mean" not in ep_run,
          f"got {ep_run.get('smooth_chunk_hf_mean')}")
    check("under 'endpoint' hf_mean is still the constrained quantity",
          approx(ep_run.get("smooth_hf_mean", -1), HF_PIN, 1e-6),
          f"got {ep_run.get('smooth_hf_mean')}")

    # The executed metrics: measured on `ready_actions`, no forward. The shared
    # harness's actions are [B,1,1], too short to difference, so the plumbing is
    # exercised with a stub returning known sums; the arithmetic itself is
    # pinned against real tensors in test_executed_metrics().
    def _exec_stub(_self, _actions):
        return (torch.tensor([2.4, 4.0]), (torch.tensor(3.0), torch.tensor(12.0)))

    ex = _stats_of(_on_path_run(smooth_coef=0.5, _executed_stub=_exec_stub))
    check("executed_hf_mean is pooled over minibatches the same way hf_mean is",
          approx(ex.get("smooth_executed_hf_mean", -1), 2.4 / (6.0 * 4.0), 1e-6),
          f"got {ex.get('smooth_executed_hf_mean')} want {2.4/24.0}")
    check("executed_jerk_ratio is the pooled sum|D2 a| / sum|a|",
          approx(ex.get("smooth_executed_jerk_ratio", -1), 0.25, 1e-6),
          f"got {ex.get('smooth_executed_jerk_ratio')} want 0.25")
    # Both are measurements, not hinge descriptions, so they must survive the
    # calibration iteration -- where the hinge deliberately reports nothing.
    ex_cal = _stats_of(_on_path_run(
        smooth_coef=0.5, _hf_ref=None, _calib_sum=torch.zeros(2),
        _executed_stub=_exec_stub,
        run_kw=dict(k=1, n_groups=2, n_chunks=16, mb_size=4)))
    check("the monitoring metrics survive the calibration iteration",
          "smooth_executed_hf_mean" in ex_cal
          and "smooth_endpoint_hf_mean" in ex_cal
          and "smooth_executed_jerk_ratio" in ex_cal,
          f"got {sorted(k for k in ex_cal if k.startswith('smooth_'))}")
    check("... while the hinge-describing metrics stay ABSENT there",
          "smooth_loss" not in ex_cal and "smooth_active_frac" not in ex_cal
          and "smooth_excess_mean" not in ex_cal,
          f"got {[k for k in ex_cal if k.startswith('smooth_')]}")

    # A threshold above the measured HF must make the term vanish entirely --
    # value AND gradient -- which is the whole point of the relu.
    hi = _on_path_run(smooth_coef=0.5, _hf_ref=torch.tensor(10.0))
    s_hi = _stats_of(hi)
    check("hf_ref above HF -> smooth_loss == 0 (hinge inactive)",
          approx(s_hi.get("smooth_loss", -1.0), 0.0, 1e-12),
          f"got {s_hi.get('smooth_loss')}")
    check("hf_ref above HF -> active_frac == 0",
          approx(s_hi.get("smooth_active_frac", 1.0), 0.0, 1e-12))
    check("inactive hinge leaves weights bit-identical to the off run",
          torch.equal(wa, torch.as_tensor(hi.w_final).flatten()),
          "zero gradient below the threshold")

    # R3: calibration only while n_updates == 0. At k=1 a step fires after every
    # micro-batch, so exactly ONE sample may be pooled; at k=4, exactly four.
    cal = _stats_of(_on_path_run(
        smooth_coef=0.5, _hf_ref=None, _calib_sum=torch.zeros(2),
        run_kw=dict(k=1, n_groups=2, n_chunks=16, mb_size=4)))
    check("during calibration the term does NOT enter the loss",
          "smooth_loss" not in cal or cal["smooth_loss"] == 0.0,
          f"got {cal.get('smooth_loss')}")
    # Calibration pools the WHOLE iteration, not just the pre-first-step window
    # (which is `gradient_accumulation_steps` micro-batches -- one at k=1, i.e.
    # mini_batch_size rows, where the pooled base HF spreads 0.64x-1.57x).
    check("calibration pools EVERY micro-batch of the iteration",
          cal.get("smooth_calib_added") == cal.get("n_micro_batches"),
          f"added {cal.get('smooth_calib_added')} of "
          f"{cal.get('n_micro_batches')} micro-batches")
    check("the strict-theta_base subtotal is reported separately for auditing",
          cal.get("smooth_calib_prestep_rows", 0) > 0
          and cal["smooth_calib_prestep_rows"] <= cal.get("smooth_rows", 0),
          f"prestep {cal.get('smooth_calib_prestep_rows')} of "
          f"{cal.get('smooth_rows')} rows")
    k4 = _stats_of(_on_path_run(
        smooth_coef=0.5, _hf_ref=None, _calib_sum=torch.zeros(2),
        run_kw=dict(k=4, n_groups=2, n_chunks=16, mb_size=4)))
    check("at k=4 the pre-step subtotal is 4 micro-batches' worth",
          k4.get("smooth_calib_prestep_rows") == 4 * 4,
          f"got {k4.get('smooth_calib_prestep_rows')}")

    # A non-finite reading must be rejected from the accumulator AND excluded
    # from the HF metrics, so hf_mean stays finite.
    bad = _stats_of(_on_path_run(
        smooth_coef=0.5, _hf_ref=None, _calib_sum=torch.zeros(2),
        run_kw=dict(k=1, n_groups=2, n_chunks=16, mb_size=4, nonfinite=(0,))))
    check("non-finite reading counted and excluded from calibration",
          bad.get("smooth_calib_nonfinite") == 1,
          f"nonfinite={bad.get('smooth_calib_nonfinite')}")
    check("clean micro-batches still calibrate after the rejection",
          bad.get("smooth_calib_added")
          == bad.get("n_micro_batches", 0) + bad.get("smooth_calib_nonfinite", 0) - 1,
          f"added={bad.get('smooth_calib_added')} "
          f"mbs={bad.get('n_micro_batches')} "
          f"rejected={bad.get('smooth_calib_nonfinite')}")
    bad2 = _stats_of(_on_path_run(
        smooth_coef=0.5, _hf_ref=torch.tensor(0.0),
        run_kw=dict(k=1, n_groups=2, n_chunks=16, mb_size=4, nonfinite=(0,))))
    check("hf_mean stays FINITE when one micro-batch reads non-finite",
          math.isfinite(bad2.get("smooth_hf_mean", float("nan")))
          and approx(bad2["smooth_hf_mean"], HF_PIN, 1e-6),
          f"got {bad2.get('smooth_hf_mean')}")
    check("the non-finite reading is counted rather than silently dropped",
          bad2.get("smooth_nonfinite_mbs") == 1,
          f"got {bad2.get('smooth_nonfinite_mbs')}")

    # relu convexity means a near-single-row minibatch reinstates the idle-row
    # blow-up pooling removes, so the term is skipped below SMOOTH_MIN_ROWS_PER_MB.
    tiny = _stats_of(_on_path_run(
        smooth_coef=0.5, run_kw=dict(k=1, n_groups=1, n_chunks=3, mb_size=1)))
    check("the term is skipped on under-filled minibatches",
          tiny.get("smooth_undersized_mbs", 0) > 0,
          f"undersized={tiny.get('smooth_undersized_mbs')}")
    # When the hinge never ran, smooth_loss/excess/active_frac are ABSENT rather
    # than 0.0. A reported 0.0 would be indistinguishable from "the field is
    # already smooth", which is the reading the docs tell an operator to expect at
    # the fixed point; undersized_mbs is what says why it is missing.
    check("hinge-describing metrics are absent, not a misleading 0.0",
          "smooth_loss" not in tiny and "smooth_active_frac" not in tiny
          and "smooth_excess_mean" not in tiny,
          f"got {[k for k in tiny if k.startswith('smooth_')]}")
    check("the field measurements ARE still reported on that path",
          "smooth_hf_mean" in tiny and "smooth_measured_mbs" in tiny)

    # The term is one scalar per minibatch, so anchors must not change its
    # magnitude -- it cannot be diluted by extra rows.
    plain = _stats_of(_on_path_run(smooth_coef=0.5))
    anch = _stats_of(_on_path_run(
        smooth_coef=0.5, include_anchor_groups=True, anchor_advantage=0.143,
        per_iteration_advantage_norm=True))
    check("anchors present -> smooth metrics still reported",
          "smooth_loss" in anch)
    check("anchors do not change the term's magnitude (row-count independent)",
          approx(anch["smooth_loss"], plain["smooth_loss"], 1e-6),
          f"anchors {anch['smooth_loss']:.8f} vs plain {plain['smooth_loss']:.8f}")


def test_round4_guards():
    """The four guards added after the third audit round."""
    print("\n[guards] discard reset, min_rows bound, hinge divisor, early-return")
    import train_grpo as tg

    class _Cfg:
        tau_centers = [0.0, 0.25, 0.35, 0.5, 0.6, 0.75]
        smooth_hf_ref_scale = 4.0
        smooth_coef = 0.5
        smooth_calib_min_rows = 512
        smooth_instrument = "chunk"

    # (1) A rejected calibration must reset the ROW counter too, or the
    # min_rows deferral becomes unreachable and the next iteration freezes
    # hf_ref from however few rows it happened to add.
    t = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t.config = _Cfg(); t.smooth_active = True; t._smooth_hf_ref = None
    t._smooth_calib_sum = torch.tensor([float("nan"), 2.0])
    t._smooth_calib_n = 40
    t._smooth_calib_rows = 600
    t._smooth_calib_iter = None
    t._smooth_ref_source = None
    t._smooth_ref_scale_applied = None
    out = t._smooth_finalize_calibration(1)
    check("rejected calibration resets the row counter",
          t._smooth_calib_rows == 0,
          f"rows left at {t._smooth_calib_rows} -> deferral unreachable")
    check("rejected calibration leaves hf_ref unset", t._smooth_hf_ref is None
          and out == {})
    check("rejected calibration does NOT stamp provenance",
          t._smooth_ref_source is None and t._smooth_ref_scale_applied is None,
          f"source={t._smooth_ref_source} scale={t._smooth_ref_scale_applied}")

    # (2) Accumulation must STOP once min_rows is met, so a frozen reference
    # carries as little post-update drift as possible.
    cal = _stats_of(_on_path_run(
        smooth_coef=0.5, _hf_ref=None, _calib_sum=torch.zeros(2),
        smooth_calib_min_rows=8,
        run_kw=dict(k=1, n_groups=2, n_chunks=16, mb_size=4)))
    check("accumulation stops at smooth_calib_min_rows",
          cal.get("smooth_calib_added") == 2,
          f"added {cal.get('smooth_calib_added')} mbs for a 8-row target at "
          f"4 rows/mb (expected 2); n_micro_batches="
          f"{cal.get('n_micro_batches')}")

    # (3) The hinge metrics divide by minibatches where the hinge RAN, not by
    # every finite reading. On the calibration iteration the hinge never runs,
    # so they must be absent rather than a misleading 0.0.
    check("calibration iteration reports no hinge metrics",
          "smooth_active_frac" not in cal and "smooth_excess_mean" not in cal,
          f"got {[k for k in cal if k.startswith('smooth_')]}")
    check("calibration iteration still reports the field measurement",
          "smooth_hf_mean" in cal and cal.get("smooth_measured_mbs", 0) > 0)
    # 9 chunks at mb_size=4 gives batches of 4, 4, 1: the 1-row batch has a FINITE
    # HF (so it counts in measured/finite) but is skipped by the row floor (so it
    # must NOT count in the hinge divisor). That separation is what makes the
    # divisor choice observable at all.
    mixed = _stats_of(_on_path_run(
        smooth_coef=0.5, run_kw=dict(k=1, n_groups=1, n_chunks=9, mb_size=4)))
    check("undersized minibatches are excluded from the hinge divisor",
          mixed.get("smooth_undersized_mbs", 0) > 0
          and mixed.get("smooth_hinge_mbs", 0)
          < mixed.get("smooth_measured_mbs", 0),
          f"hinge={mixed.get('smooth_hinge_mbs')} "
          f"measured={mixed.get('smooth_measured_mbs')} "
          f"undersized={mixed.get('smooth_undersized_mbs')}")
    check("active_frac == 1.0 over hinge-evaluated batches, not diluted by them",
          approx(mixed.get("smooth_active_frac", 0.0), 1.0, 1e-9),
          f"got {mixed.get('smooth_active_frac')} -- dividing by the finite count "
          f"instead would give "
          f"{mixed.get('smooth_hinge_mbs', 0) / max(mixed.get('smooth_measured_mbs', 1), 1):.3f}")

    # A SUCCESSFUL calibration must stamp provenance (the mirror of the
    # rejected-path assertion above).
    t4 = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t4.config = _Cfg(); t4.smooth_active = True; t4._smooth_hf_ref = None
    t4._smooth_calib_sum = torch.tensor([1.2, 2.0]); t4._smooth_calib_n = 40
    t4._smooth_calib_rows = 600; t4._smooth_calib_iter = None
    t4._smooth_ref_source = None; t4._smooth_ref_scale_applied = None
    t4._smooth_finalize_calibration(1)
    check("successful calibration stamps provenance and the applied scale",
          t4._smooth_ref_source == "calibrated"
          and approx(t4._smooth_ref_scale_applied, 4.0, 1e-9),
          f"source={t4._smooth_ref_source} scale={t4._smooth_ref_scale_applied}")

    # (4) On an iteration where no optimizer step survived, the HF readings are
    # still valid (they precede the loss guard) and must reach update_stats.
    dead = _stats_of(_on_path_run(
        smooth_coef=0.5, _hf_ref=torch.tensor(0.0),
        run_kw=dict(k=1, n_groups=2, n_chunks=16, mb_size=4,
                    nonfinite=(0, 1, 2, 3))))
    check("n_updates == 0 iteration still carries the smooth metrics",
          "smooth_measured_mbs" in dead and dead.get("smooth_measured_mbs", 0) > 0,
          f"got {[k for k in dead if k.startswith('smooth_')]}")
    check("... and reports why the term did not apply",
          dead.get("smooth_nonfinite_mbs", 0) > 0
          or dead.get("smooth_nonfinite_loss_mbs", 0) > 0,
          f"nonfinite_mbs={dead.get('smooth_nonfinite_mbs')} "
          f"nonfinite_loss={dead.get('smooth_nonfinite_loss_mbs')}")

    # (5) mini_batch_size below the row floor must FAIL rather than silently
    # disable the constraint for a whole run.
    from grpo_config import GRPOConfig
    t2 = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t2.config = GRPOConfig(smooth_coef=0.15,
                           mini_batch_size=tg.SMOOTH_MIN_ROWS_PER_MB - 1)
    check("mini_batch_size < SMOOTH_MIN_ROWS_PER_MB is rejected at setup",
          _raises(lambda: tg.GRPOTrainer._setup_smoothness(t2), ValueError))


def test_nan_guard():
    print("\n[nan] a non-finite HF must not poison hf_ref (permanent-stall guard)")
    import train_grpo as tg

    class _Cfg:
        tau_centers = [0.0, 0.25, 0.35, 0.5, 0.6, 0.75]
        smooth_hf_ref_scale = 4.0
        smooth_coef = 0.5
        smooth_calib_min_rows = 1
        smooth_instrument = "chunk"

    t = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t.config = _Cfg()
    t.smooth_active = True
    t._smooth_hf_ref = None
    t._smooth_calib_sum = torch.tensor([float("nan"), 2.0])
    t._smooth_calib_n = 3
    t._smooth_calib_rows = 640
    out = t._smooth_finalize_calibration(1)
    check("non-finite accumulator is discarded, not frozen",
          t._smooth_hf_ref is None and out == {})
    check("accumulator is reset so the next iteration can retry cleanly",
          t._smooth_calib_n == 0
          and bool(torch.isfinite(t._smooth_calib_sum).all()))

    # And prove why it matters: a NaN reference NaNs the penalty, which NaNs the
    # loss, which makes the guard drop every minibatch forever.
    nan_pen = (torch.tensor(0.1) - torch.tensor(float("nan"))).clamp(min=0.0)
    check("a NaN hf_ref would have NaN'd the penalty (clamp propagates NaN)",
          bool(torch.isnan(nan_pen)),
          "this is the failure the guard prevents")
    # Zero pooled energy must also be rejected, not divided by.
    t3 = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t3.config = _Cfg(); t3.smooth_active = True; t3._smooth_hf_ref = None
    t3._smooth_calib_sum = torch.tensor([0.0, 0.0]); t3._smooth_calib_n = 5
    t3._smooth_calib_rows = 640
    check("zero pooled energy is rejected rather than producing 0/0",
          t3._smooth_finalize_calibration(1) == {} and t3._smooth_hf_ref is None)


# ───────────────────────────────────────────────────────────────────────────────
# 12. The executed-chunk metrics (`ready_actions`, no forward)
# ───────────────────────────────────────────────────────────────────────────────

def test_executed_metrics():
    print("\n[executed] roughness of the chunks the robot actually executed")
    import train_grpo as tg

    torch.manual_seed(31)
    B, H_pad, D_pad, H_valid = 5, 50, 12, 16
    dims = [0, 1, 2, 3, 4, 5]

    t = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t._smooth_dims = torch.tensor(dims, dtype=torch.long)
    t._smooth_horizon = H_valid
    t._smooth_eef_pos_dims = torch.tensor([0, 1, 2], dtype=torch.long)

    a = torch.randn(B, H_pad, D_pad)
    # Pin the executed window explicitly: relying on the class default makes
    # this assertion depend on test ORDER, which is how a leaked class
    # attribute from another test silently changed what it measured.
    t._smooth_n_exec = H_valid
    mom, jerk = t._smooth_executed_stats(a)

    want_mom = roughness_moments(
        a[:, :H_valid].index_select(2, t._smooth_dims).float()).sum(dim=0)
    check("executed moments == roughness_moments over the SAME rectangle",
          mom is not None and torch.allclose(mom, want_mom, atol=1e-5),
          f"got {None if mom is None else mom.tolist()} want {want_mom.tolist()}")

    p = a[:, :H_valid, :3].float()
    want_num = (p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]).abs().sum()
    want_den = p.abs().sum()
    check("executed jerk == sum|D2 a| / sum|a| over the EEF-position dims",
          jerk is not None
          and approx(float(jerk[0]), float(want_num), 1e-3)
          and approx(float(jerk[1]), float(want_den), 1e-3),
          f"got {None if jerk is None else [float(x) for x in jerk]}")

    # It slices to the CONSTRAINED rectangle, so the padded (50, 128) region can
    # be arbitrary garbage and the reading must not move.
    a2 = a.clone()
    a2[:, H_valid:] = 1e6
    a2[:, :, 6:] = -1e6
    mom2, jerk2 = t._smooth_executed_stats(a2)
    check("padding beyond the horizon/dims does not reach the reading",
          torch.allclose(mom, mom2, atol=0.0)
          and approx(float(jerk[0]), float(jerk2[0]), 0.0),
          "the slice is happening AFTER the difference somewhere")

    # It must be non-differentiable: it is buffer data, and a gradient path here
    # would turn a monitoring metric into a silent second training objective.
    a3 = torch.randn(B, H_pad, D_pad, requires_grad=True)
    m3, j3 = t._smooth_executed_stats(a3)
    check("the executed reading carries no graph (monitoring only)",
          not m3.requires_grad and not j3[0].requires_grad)

    # A ramp has zero second difference, so the jerk ratio is exactly 0 -- the
    # calibration point that says the operator is really a 2nd difference.
    ramp = torch.zeros(2, H_pad, D_pad)
    ramp[:, :, :3] = torch.arange(H_pad, dtype=torch.float32)[None, :, None]
    _, j_ramp = t._smooth_executed_stats(ramp)
    check("a straight ramp gives jerk_ratio == 0",
          approx(float(j_ramp[0]) / float(j_ramp[1]), 0.0, 1e-6),
          f"got {float(j_ramp[0]) / float(j_ramp[1])}")

    # Degradation, not exceptions: a metrics path must never cost an iteration.
    t_off = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    check("no dims/horizon -> (None, None) rather than raising",
          t_off._smooth_executed_stats(a) == (None, None))
    check("a too-short action tensor -> (None, None), not a ValueError",
          t._smooth_executed_stats(torch.randn(B, 2, D_pad)) == (None, None))
    check("a non-3D action tensor -> (None, None)",
          t._smooth_executed_stats(torch.randn(B, H_pad)) == (None, None))
    check("None actions -> (None, None)", t._smooth_executed_stats(None)
          == (None, None))

    # The EEF span is derived from modality_keys, never a hardcoded index.
    from smoothness import build_key_dim_span
    keys = ["end_effector_position", "end_effector_rotation", "gripper_close",
            "base_motion", "control_mode"]
    kd = {"end_effector_position": 3, "end_effector_rotation": 3,
          "gripper_close": 1, "base_motion": 4, "control_mode": 1}
    check("EEF-position span for PandaOmron is [0,1,2]",
          build_key_dim_span(keys, kd) == [0, 1, 2])
    rkeys = ["gripper_close", "end_effector_position", "end_effector_rotation",
             "control_mode", "base_motion"]
    check("a reordered layout shifts the span (nothing is hardcoded)",
          build_key_dim_span(rkeys, kd) == [1, 2, 3],
          f"got {build_key_dim_span(rkeys, kd)}")
    pkeys = ["body.end_effector_position", "body.end_effector_rotation"]
    check("a prefixed key still matches",
          build_key_dim_span(pkeys, {k: 3 for k in pkeys}) == [0, 1, 2])
    check("an embodiment without the key yields [] (caller falls back, no guess)",
          build_key_dim_span(["gripper_close"], {"gripper_close": 1}) == [])


# ───────────────────────────────────────────────────────────────────────────────
# 13. Config validation of the instrument selector
# ───────────────────────────────────────────────────────────────────────────────

def test_instrument_config():
    print("\n[config] smooth_instrument validation and the new hf_ref_scale")
    from grpo_config import GRPOConfig
    from smoothness import SMOOTH_INSTRUMENTS

    check("the default instrument is 'chunk'",
          GRPOConfig().smooth_instrument == "chunk",
          f"got {GRPOConfig().smooth_instrument!r}")
    check("smooth_hf_ref_scale defaults to 15.0 (chunk-calibrated)",
          approx(GRPOConfig().smooth_hf_ref_scale, 15.0, 1e-12),
          f"got {GRPOConfig().smooth_hf_ref_scale}")
    check("smooth_coef still defaults to 0.0 (feature OFF)",
          GRPOConfig().smooth_coef == 0.0)
    for good in SMOOTH_INSTRUMENTS:
        check(f"smooth_instrument={good!r} is accepted",
              GRPOConfig(smooth_instrument=good).smooth_instrument == good)
    # Validated even with the feature OFF, so a typo surfaces at construction
    # rather than the first time someone switches smooth_coef on.
    msg = _raise_message(
        lambda: GRPOConfig(smooth_instrument="a_hat"), ValueError)
    check("an unknown instrument is rejected at config construction",
          msg is not None, f"got {msg!r}")
    check("... and the error names both valid values",
          msg is not None and "chunk" in msg and "endpoint" in msg,
          f"got {msg!r}")
    check("rejected even when smooth_coef == 0 (typos surface before use)",
          _raises(lambda: GRPOConfig(smooth_coef=0.0,
                                     smooth_instrument="ENDPOINT"), ValueError),
          "case-sensitivity is deliberate -- the value goes in the guard key")


def test_smooth_forwards_draw_no_rng():
    """The smooth FORWARDS themselves must consume no RNG, on the real code path.

    The on-path harness in `test_on_path` substitutes a fake `compute_fm_log_prob`,
    so neither arm ever executes a DiT forward or an Euler rollout -- an RNG
    assertion there cannot detect a draw inside those forwards, however it is
    worded. This test closes that hole by calling the REAL `compute_fm_log_prob`
    against the analytic stand-in head, with the smooth path on and off, and
    comparing the RNG stream afterwards. It is falsifiable: injecting a single
    `torch.randn(1)` into `_smooth_chunk_rollout` makes it fail, which the
    positive control at the end asserts.
    """
    print("\n[rng] the real smooth forwards draw no RNG")
    import fm_log_prob as fm

    torch.manual_seed(2)
    B, H_pad, D_pad, K = 3, 20, 8, 4
    H_valid, dims = 6, torch.tensor([0, 1, 2])
    head = _StubHead(H_pad, D_pad)
    head.num_inference_timesteps = 4          # the rollout reads this
    mask = torch.zeros(B, H_pad, D_pad)
    mask[:, :H_valid, :4] = 1.0
    common = dict(
        action_head=head,
        backbone_output={"backbone_features": torch.randn(B, 5, 4)},
        state_features=torch.zeros(B, 0, D_pad),
        embodiment_id=torch.zeros(B, dtype=torch.long),
        actions=torch.randn(B, H_pad, D_pad),
        action_mask=mask,
        timesteps=torch.rand(K, B) * 0.9,
        noise=torch.randn(B, H_pad, D_pad),
        n_samples=K,
    )

    def stream_after(**extra):
        torch.manual_seed(4321)
        _ = torch.randn(1)                    # anchor
        fm.compute_fm_log_prob(**common, **extra)
        return torch.randn(4).tolist()

    off = stream_after()
    for instrument in ("chunk", "endpoint"):
        got = stream_after(smooth_dims=dims, smooth_horizon=H_valid,
                           smooth_instrument=instrument)
        check(f"the real '{instrument}' smooth pass draws no RNG",
              off == got, f"off {off} vs {instrument} {got}")

    # POSITIVE CONTROL: the comparison above must be able to fail. Inject one
    # draw into the rollout and confirm it is detected.
    real = fm._smooth_chunk_rollout

    def leaky(*a, **k):
        torch.randn(1)                        # the leak we must catch
        return real(*a, **k)

    fm._smooth_chunk_rollout = leaky
    try:
        leaked = stream_after(smooth_dims=dims, smooth_horizon=H_valid,
                             smooth_instrument="chunk")
    finally:
        fm._smooth_chunk_rollout = real
    check("... and an injected draw IS detected (the check can fail)",
          off != leaked, "a leak in the rollout went unnoticed")


def test_audit_round_fixes():
    """Regressions for the defects the two independent audits surfaced."""
    print("\n[audit] legacy sidecar back-compat, guard scope, strict finiteness")
    import json as _json
    import tempfile
    from pathlib import Path as _Path
    import train_grpo as tg

    # --- 1. A PRE-CHANGE sidecar (no instrument/sampler keys) must resume under
    # "endpoint" -- that is the whole reason the instrument is retained -- and
    # must still be refused under "chunk", where hf_ref names another quantity.
    legacy = {
        "hf_ref": 0.006911625619977713,
        "guard": {
            "tau_centers": [0.0, 0.25, 0.35, 0.5, 0.6, 0.75],
            "jitter_std": 0.02, "jitter_pos": 0.25, "jitter_neg": 0.05,
            "jitter_paired": False, "dims": [0, 1, 2, 3, 4, 5], "horizon": 16,
            "embodiment_tag": "ROBOCASA_PANDA_OMRON",
            "model_path": "nvidia/GR00T-N1.6-3B",
        },
        "hf_ref_scale": 4.0, "calibrated_at_iteration": 1,
        "hf_ref_source": "calibrated",
        "env_names": ["robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env"],
    }

    def _loader(instrument):
        # Local stub: only the fields _load_smooth_ref / _smooth_guard_key read.
        class _Cfg:
            tau_centers = [0.0, 0.25, 0.35, 0.5, 0.6, 0.75]
            jitter_pos = 0.25
            jitter_neg = 0.05
            jitter_paired = False
            embodiment_tag = "ROBOCASA_PANDA_OMRON"
            model_path = "nvidia/GR00T-N1.6-3B"
            smooth_hf_ref_scale = 4.0

        t = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
        t.smooth_active = True
        t.device = torch.device("cpu")
        t._smooth_dims_list = [0, 1, 2, 3, 4, 5]
        t._smooth_horizon = 16
        t._smooth_schedule = (0.0, 0.25, 0.5, 0.75)
        t._smooth_schedule_dt = 0.25
        t._smooth_ref_scale_applied = None
        t._smooth_calib_iter = None
        t._smooth_ref_source = None
        t.config = _Cfg()
        t.config.smooth_instrument = instrument
        t.config.env_names = list(legacy["env_names"])
        d = _Path(tempfile.mkdtemp())
        (d / tg.SMOOTH_REF_FILENAME).write_text(_json.dumps(legacy))
        return t, d

    t_ep, d_ep = _loader("endpoint")
    try:
        got = t_ep._load_smooth_ref(d_ep)
        ok, why = (got is not None
                   and approx(float(got), legacy["hf_ref"], 1e-12)), ""
    except Exception as e:                                   # noqa: BLE001
        ok, why = False, f"raised {type(e).__name__}: {e}"
    check("a pre-change smooth_ref.json still resumes under 'endpoint'", ok,
          why or "instrument is IMPLIED (only the endpoint existed then), so a "
                 "missing key must backfill, not hard-fail")

    t_ch, d_ch = _loader("chunk")
    check("... and is still REFUSED under 'chunk'",
          _raises(lambda: t_ch._load_smooth_ref(d_ch), RuntimeError),
          "hf_ref measured on the endpoint is meaningless for the chunk")

    # --- 2. Sampler keys are scoped to "chunk": hf_ref for the endpoint provably
    # cannot depend on a schedule it never walks, so gating on it is a false
    # rejection.
    t_ep2, _ = _loader("endpoint")
    g_ep = t_ep2._smooth_guard_key()
    t_ch2, _ = _loader("chunk")
    g_ch = t_ch2._smooth_guard_key()
    check("guard omits sampler keys under 'endpoint'",
          "sampler_steps" not in g_ep and "sampler_dt" not in g_ep,
          f"got {sorted(g_ep)}")
    check("guard carries sampler keys under 'chunk'",
          g_ch.get("sampler_steps") == 4 and approx(g_ch.get("sampler_dt"), 0.25),
          f"got steps={g_ch.get('sampler_steps')} dt={g_ch.get('sampler_dt')}")
    check("guard always discriminates the instrument itself",
          g_ep.get("instrument") == "endpoint" and g_ch.get("instrument") == "chunk")

    # --- 3. An inf in M must NOT be admitted by a finite RATIO. R/(6*inf) is
    # 0.0, which passes a ratio test and then poisons the pooled denominator so
    # smooth/hf_mean reads exactly 0.0 -- "perfectly smooth", the opposite of an
    # overflowing field.
    poisoned = torch.tensor([[1.2, float("inf")], [1.2, 2.0]])
    ratio = float(pooled_hf(poisoned, detach_denominator=False))
    check("a finite pooled ratio can hide an inf moment (why the guard is strict)",
          math.isfinite(ratio) and not bool(torch.isfinite(poisoned).all()),
          f"ratio={ratio} (finite) while the moments are not")

    src = inspect.getsource(tg.GRPOTrainer._grpo_update_inner)
    check("the metric accumulator gates on the MOMENTS, not the pooled ratio",
          "if bool(torch.isfinite(mom_det).all()):" in src,
          "a `math.isfinite(hf_pooled_det)` gate here admits inf into smooth_m_sum")

    # --- 4. executed_jerk_ratio covers only the prefix MultiStepWrapper executes.
    t = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    t._smooth_horizon = 16
    t._smooth_dims = torch.arange(6)
    t._smooth_eef_pos_dims = torch.arange(3)
    t._smooth_n_exec = 8
    a = torch.zeros(1, 50, 128)
    # Rows 0..7 are a straight ramp (zero curvature); rows 8..15 zig-zag hard.
    a[0, :8, :3] = torch.arange(8, dtype=torch.float32).view(8, 1)
    a[0, 8:16, :3] = torch.tensor([0.0, 9.0] * 4).view(8, 1)
    _mom, jerk = t._smooth_executed_stats(a)
    check("executed_jerk_ratio ignores the DISCARDED tail (ramp prefix -> 0)",
          jerk is not None and approx(float(jerk[0]), 0.0, 1e-6),
          f"got numerator {None if jerk is None else float(jerk[0])}; a "
          f"full-horizon window would pick up the zig-zag in rows 8-15")
    t._smooth_n_exec = 16
    _m2, jerk_full = t._smooth_executed_stats(a)
    check("... and the full-horizon window DOES see it (test is not vacuous)",
          jerk_full is not None and float(jerk_full[0]) > 1.0,
          f"got {None if jerk_full is None else float(jerk_full[0])}")

    # --- 5. No EEF-position span -> the metric is dropped, not silently
    # redefined over dims that mix metres with radians.
    t._smooth_eef_pos_dims = None
    t._smooth_n_exec = 8
    check("no EEF span -> executed_jerk_ratio omitted (not redefined)",
          t._smooth_executed_stats(a)[1] is None)

    # --- 6. The fake writer must carry add_text, or the provenance branch raises
    # inside a bare `except Exception: pass` and drops the whole payload.
    import test_grad_accum as tga
    wsrc = inspect.getsource(tga.test_result_shapes_are_loggable)
    check("the logging harness's fake writer implements add_text",
          "def add_text(" in wsrc,
          "_log_metrics calls writer.add_text for smooth/instrument")


def main():
    print("=" * 74)
    print("Trajectory-roughness constraint (jerk constraint) -- CPU test suite")
    print("=" * 74)
    test_hf_calibration()
    test_endpoint_identity()
    test_pooled()
    test_dim_selection()
    test_fm_return_contract()
    test_rollout_gradient_localization()
    test_ref_persistence()
    test_calibration_finalize()
    test_update_integration()
    test_on_path()
    test_round4_guards()
    test_nan_guard()
    test_executed_metrics()
    test_instrument_config()
    test_smooth_forwards_draw_no_rng()
    test_audit_round_fixes()
    print()
    if _failures:
        print(f"{RED}{len(_failures)} test(s) FAILED:{RESET}")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"{GREEN}All trajectory-roughness tests passed.{RESET}")


if __name__ == "__main__":
    main()
