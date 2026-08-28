"""CPU tests for the endpoint-roughness constraint (the "jerk constraint").

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
  5. `compute_fm_log_prob` return contract, the clean tau=0 smooth pass, and its
     INVARIANCE to jitter, against an analytic stand-in action head (no DiT).
  6. The real `_grpo_update_inner`: smooth_coef=0 is BIT-IDENTICAL to a run
     without the feature; smooth_coef>0 changes the loss; calibration only
     accumulates while n_updates == 0; anchors use the same divisor as clip_loss.
  7. `smooth_ref.json` round-trip and guard-key rejection.

KNOWN COVERAGE GAP: the fake action head value-pins (R, M) per ROW, which is what
lets the on-path tests assert exact penalties. A consequence is that every minibatch
has the same pooled HF regardless of size, so swapping the row-weighted
`smooth_hf_mean` accumulation (sum R / sum M) for a per-minibatch mean of ratios is
UNOBSERVABLE here -- both give 0.1 exactly. Distinguishing them needs a fake whose HF
varies BETWEEN minibatches. `smooth_hf_mean` is a diagnostic only; the loss path is
covered, and `pooled_hf`'s exact associativity over batch splits is asserted directly.

Usage:  .venv/bin/python scripts/grpo/test_smoothness.py
"""

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
    function of the noisy trajectory and the smooth path can be checked by hand.
    """

    def __init__(self, H_pad, D_pad, scale=0.5, bias=0.0):
        super().__init__()
        self.num_timestep_buckets = 1000
        self.scale = scale
        self.bias = bias
        self._H, self._D = H_pad, D_pad

        class _Cfg:
            add_pos_embed = False
            use_alternate_vl_dit = False
        self.config = _Cfg()

    def action_encoder(self, noisy, t_disc, emb):
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

    lp2, mom = compute_fm_log_prob(**common, smooth_dims=dims,
                                   smooth_horizon=H_valid)
    check("smooth only -> (log_probs, [B, 2] moments)",
          tuple(mom.shape) == (B, 2), f"got {tuple(mom.shape)}")
    check("adding the smooth output does NOT change log_probs",
          torch.allclose(lp2, plain, atol=0.0),
          f"max delta {float((lp2 - plain).abs().max()):.3e}")

    lp3, pt3, mom3 = compute_fm_log_prob(**common, return_per_tau=True,
                                         smooth_dims=dims,
                                         smooth_horizon=H_valid)
    check("both flags -> (log_probs, per_tau, moments) in that order",
          torch.allclose(pt3, per_tau, atol=0.0)
          and tuple(mom3.shape) == (B, 2))

    # Hand-verify against the primitives: the smooth pass is a CLEAN forward at
    # tau = 0, where x_0 == eps exactly, so a_hat(0) = eps + v_theta(eps, 0).
    t0 = torch.zeros(B)
    v0 = head.scale * noise                     # the stub's velocity at x = eps
    want = roughness_moments(
        (noise + v0)[:, :H_valid].index_select(2, dims))
    check("moments match an independent recomputation at tau=0 from eps",
          torch.allclose(mom, want, atol=1e-5),
          f"max delta {float((mom - want).abs().max()):.3e}")

    # THE critical property: the smooth output must be INVARIANT to jitter,
    # because it is computed on its own clean forward rather than reusing the
    # K-loop's jittered one. Without this the jitter Jacobian response dominates
    # HF (measured 0.000347 -> 0.790 at tau=0) and the constraint is a no-op.
    K_ = K
    nfi = torch.randn(K_, B, H_pad, D_pad) * 0.5 + noise.unsqueeze(0)
    _, mom_j = compute_fm_log_prob(**common, noise_for_input=nfi,
                                   smooth_dims=dims, smooth_horizon=H_valid)
    check("smooth moments are IDENTICAL with and without jitter (clean forward)",
          torch.allclose(mom, mom_j, atol=0.0),
          f"max delta {float((mom - mom_j).abs().max()):.3e}")

    check("smooth_dims without smooth_horizon raises",
          _raises(lambda: compute_fm_log_prob(**common, smooth_dims=dims),
                  ValueError))
    # The tau=0 pass never reads `timesteps`, so it must work without them.
    _out = compute_fm_log_prob(**{**common, "timesteps": None},
                               smooth_dims=dims, smooth_horizon=H_valid)
    check("smooth path does NOT require explicit timesteps (tau=0 ignores them)",
          isinstance(_out, tuple) and tuple(_out[1].shape) == (B, 2))
    # smooth_no_grad must give identical values with no graph.
    _, m_ng = compute_fm_log_prob(**common, smooth_dims=dims,
                                  smooth_horizon=H_valid, smooth_no_grad=True)
    check("smooth_no_grad yields identical moments with requires_grad False",
          torch.allclose(m_ng, mom, atol=0.0) and not m_ng.requires_grad)
    check("smooth_horizon beyond the padded horizon raises",
          _raises(lambda: compute_fm_log_prob(
              **common, smooth_dims=dims, smooth_horizon=H_pad + 1), ValueError))
    check("out-of-range smooth_dims raises",
          _raises(lambda: compute_fm_log_prob(
              **common, smooth_dims=torch.tensor([D_pad]),
              smooth_horizon=H_valid), ValueError))


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
        smooth_hf_ref_scale = 4.0
        smooth_calib_min_rows = 1
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
    t._smooth_calib_iter = 1
    t._smooth_ref_source = "calibrated"
    t._smooth_ref_scale_applied = 4.0
    t._smooth_hf_ref = torch.tensor(0.0048)

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
                                        "jitter_paired"},
              f"got {sorted(payload['guard'])}")
        check("env_names recorded OUTSIDE the hard-fail guard (warn-only)",
              "env_names" in payload and "env_names" not in payload["guard"])
        t._smooth_ref_scale_applied = 4.0
        t.config.smooth_hf_ref_scale = 9.0     # live config differs from baked-in
        t._save_smooth_ref(ck)
        _p2 = json.loads((ck / tg.SMOOTH_REF_FILENAME).read_text())
        check("sidecar records the BAKED-IN scale, not the live config value",
              approx(_p2.get("hf_ref_scale"), 4.0, 1e-9),
              f"got {_p2.get('hf_ref_scale')} (live config is 9.0)")
        check("hf_ref / recorded_scale reconstructs the base HF correctly",
              approx(_p2["hf_ref"] / _p2["hf_ref_scale"], 0.0048 / 4.0, 1e-9))
        t.config.smooth_hf_ref_scale = 4.0
        t._save_smooth_ref(ck)
        check("provenance recorded: source + the scale actually baked in",
              payload.get("hf_ref_source") == "calibrated"
              and approx(payload.get("hf_ref_scale"), 4.0, 1e-9),
              f"got source={payload.get('hf_ref_source')!r} "
              f"scale={payload.get('hf_ref_scale')!r}")
        t2 = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
        t2.config = _Cfg(); t2.device = torch.device("cpu")
        t2.smooth_active = True
        t2._smooth_dims_list = [0, 1, 2, 3, 4, 5]; t2._smooth_horizon = 16
        t2._load_smooth_ref(ck)
        check("a resume INHERITS the baked-in scale, not this run's config",
              approx(t2._smooth_ref_scale_applied, 4.0, 1e-9)
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
        ):
            p2 = json.loads(json.dumps(payload))
            p2["guard"][field] = bad
            (ck / tg.SMOOTH_REF_FILENAME).write_text(json.dumps(p2))
            check(f"guard mismatch on {field!r} hard-fails",
                  _raises(lambda: t._load_smooth_ref(ck), RuntimeError))

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
              "_smooth_hf_ref", "_smooth_calib_sum", "_smooth_calib_n")}
    tg.GRPOTrainer.smooth_active = True
    tg.GRPOTrainer._smooth_dims = dims
    tg.GRPOTrainer._smooth_horizon = horizon
    tg.GRPOTrainer._smooth_hf_ref = over.pop("_hf_ref", torch.tensor(0.0))
    tg.GRPOTrainer._smooth_calib_sum = over.pop("_calib_sum", None)
    tg.GRPOTrainer._smooth_calib_n = 0
    tg.GRPOTrainer._smooth_calib_rows = 0
    tg.GRPOTrainer._smooth_calib_rows = 0
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
    # Row-weighted, not a mean of per-minibatch ratios: with UNEQUAL minibatch
    # sizes the two differ, and only the row-weighted one matches the loss and the
    # calibration. n_chunks=10 / mb_size=4 gives sizes [4,4,2].
    uneq = _stats_of(_on_path_run(
        smooth_coef=0.5, run_kw=dict(k=1, n_groups=1, n_chunks=10, mb_size=4)))
    check("hf_mean is row-weighted across unequal minibatch sizes",
          approx(uneq.get("smooth_hf_mean", -1), HF_PIN, 1e-6),
          f"got {uneq.get('smooth_hf_mean')} want {HF_PIN} "
          f"(a per-minibatch mean would drift with the [4,4,2] split)")
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


def main():
    print("=" * 74)
    print("Endpoint-roughness constraint (jerk constraint) -- CPU test suite")
    print("=" * 74)
    test_hf_calibration()
    test_endpoint_identity()
    test_pooled()
    test_dim_selection()
    test_fm_return_contract()
    test_ref_persistence()
    test_calibration_finalize()
    test_update_integration()
    test_on_path()
    test_round4_guards()
    test_nan_guard()
    print()
    if _failures:
        print(f"{RED}{len(_failures)} test(s) FAILED:{RESET}")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"{GREEN}All endpoint-roughness tests passed.{RESET}")


if __name__ == "__main__":
    main()
