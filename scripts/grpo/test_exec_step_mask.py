"""Tests for the executed-step loss mask (`GRPOConfig.mask_loss_with_n_action_steps`).

WHAT IS UNDER TEST. MultiStepWrapper plays steps 0..n_action_steps-1 of each
predicted chunk and discards the rest. With the flag on, the clipped surrogate's
importance ratio is computed over those executed steps only:

    lp_exec = -mean_k [ sum_{h < n_action_steps} mask * (v - u)^2
                        / sum_{h < n_action_steps} mask ]
    rho     = exp(lp_exec_theta - lp_exec_ref)

while the KL terms keep the full-horizon log-probs. Every consumer of the
surrogate's ratio (rho_floor, PAWS, clipfrac*, drift/*, gradprobe/*) follows the
executed-step pair.

Two layers:
  * LOSS LEVEL: the REAL `compute_fm_log_prob` on a stub head whose DiT gives each
    action step its OWN weight row (inputs are mixed across steps, weights are
    not), so "no gradient reaches an unexecuted step" is an exact zero, not a
    tolerance.
  * TRAINER LEVEL: the REAL `_compute_ref_log_probs`, `_grpo_update_inner`,
    `_grad_probe_finish`, `_summarize_ref_mse`, `_log_metrics` and the `train()`
    banner on CPU, driving that same stub through the real `compute_fm_log_prob`.

Off-switch bit-identity against HEAD is an OUT-OF-TREE differential, run when
this feature landed: materialise HEAD's `scripts/grpo/` with
`git archive <sha> scripts/grpo | tar -x -C /tmp/head`, then run one driver
against both trees that exercises `test_grad_accum.run_update` over a config
matrix, `test_vel_anchor._Harness` (ref pass + update), `test_grad_probe.run_probe`,
the real `compute_fm_log_prob` on `test_vel_anchor._Head` for every pre-existing
flag combination, and this file's real-fm harness with the flag off, and compare
weights, step gradients, full stats dicts, stdout and the RNG state bitwise.
Test [T1] is the in-tree half.

Run with the project venv (needs torch + peft; CPU is fine):
    PYTHONPATH=/tmp/grpo_overlay .venv/bin/python scripts/grpo/test_exec_step_mask.py
"""

import contextlib
import dataclasses
import io
import math
import sys
import tempfile
import threading
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_grpo  # noqa: E402
import test_grad_accum as tga  # noqa: E402
from episode_buffer import ActionChunk  # noqa: E402
from fm_log_prob import FMLogProbResult, VelAnchor, compute_fm_log_prob  # noqa: E402
from grpo_config import GRPOConfig  # noqa: E402
from train_grpo import GRPOTrainer  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


def _raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# Stub head: one weight row per action step
# ═════════════════════════════════════════════════════════════════════════════

H_PAD, H_VALID = 10, 6      # padded / embodiment action horizon
D_PAD, D_VALID = 4, 3       # padded / embodiment action dim
N_EXEC = 4                  # executed prefix used by most tests
T_COEF = 1e-3
MIX = 0.3


class _StepDiT(nn.Module):
    """out[:, h] = x[:, h] * W[h] + MIX * mean_h' x[:, h'] + T_COEF * t.

    d out[:, h] / d W[h'] == 0 for h' != h, so the gradient on W[h] comes ONLY
    from loss terms on output step h. MIX couples the steps through the INPUT,
    the way attention does, without coupling them through the weights.
    """

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(1.0 + 0.3 * torch.randn(H_PAD, D_PAD, generator=g))

    def forward(self, hidden_states, encoder_hidden_states=None,
                encoder_attention_mask=None, timestep=None,
                return_all_hidden_states=False, image_mask=None,
                backbone_attention_mask=None):
        x = hidden_states
        out = (x * self.W[None] + MIX * x.mean(dim=1, keepdim=True)
               + T_COEF * timestep.to(x.dtype)[:, None, None])
        return out, None


class _Head(nn.Module):
    """Action-head stand-in: identity encoder/decoder around `_StepDiT`."""

    def __init__(self, seed=0):
        super().__init__()
        self.num_timestep_buckets = 1000
        self.num_inference_timesteps = 4
        self.config = SimpleNamespace(add_pos_embed=False,
                                      use_alternate_vl_dit=False, noise_s=0.999)
        self.model = _StepDiT(seed)

    def action_encoder(self, noisy, t_disc, emb):
        return noisy

    def action_decoder(self, model_output, emb):
        return model_output


class _Policy(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.action_head = head


def _mask(B=None) -> torch.Tensor:
    m = torch.zeros(H_PAD, D_PAD)
    m[:H_VALID, :D_VALID] = 1.0
    return m if B is None else m.unsqueeze(0).expand(B, -1, -1).clone()


def _inputs(B=3, K=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(
        backbone_output={"backbone_features": torch.zeros(B, 1, D_PAD)},
        state_features=torch.zeros(B, 0, D_PAD),
        embodiment_id=torch.zeros(B, dtype=torch.long),
        actions=torch.randn(B, H_PAD, D_PAD, generator=g),
        action_mask=_mask(B),
        timesteps=torch.rand(K, B, generator=g) * 0.9,
        noise=torch.randn(B, H_PAD, D_PAD, generator=g),
        n_samples=K,
    )


def _jittered(inp, lam=0.25, seed=9):
    eps, K = inp["noise"], inp["n_samples"]
    g = torch.Generator().manual_seed(seed)
    xi = torch.randn((K,) + tuple(eps.shape), generator=g)
    return math.sqrt(1 - lam * lam) * eps.unsqueeze(0) + lam * xi


def _hand_lp(W, actions, noise, mask, ts, n_steps=None, nfi=None) -> torch.Tensor:
    """-mean_k masked MSE from the formula, independent of compute_fm_log_prob.

    `n_steps` zeroes the mask from that step on and normalises by what is left.
    """
    m = mask.clone().float()
    if n_steps is not None:
        m[:, n_steps:] = 0.0
    valid = m.sum(dim=(1, 2))
    u = (actions - noise).float()
    acc = torch.zeros(actions.shape[0])
    for k in range(ts.shape[0]):
        t = ts[k]
        x_in = noise if nfi is None else nfi[k]
        te = t[:, None, None]
        x = (1 - te) * x_in + te * actions
        td = (t * 1000).long().to(x.dtype)
        v = x * W[None] + MIX * x.mean(dim=1, keepdim=True) + T_COEF * td[:, None, None]
        acc = acc + ((v.float() - u) ** 2 * m).sum(dim=(1, 2)) / valid
    return -acc / ts.shape[0]


def _flat(x):
    if isinstance(x, torch.Tensor):
        return [x]
    out = []
    for y in x:
        out += _flat(y)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Loss level (the real compute_fm_log_prob)
# ═════════════════════════════════════════════════════════════════════════════

def test_L1_off_path_contract():
    print("\n[L1] n_exec_steps=None is the pre-existing call, bitwise, every flag set")
    head = _Head(seed=2)
    inp = _inputs()
    dims = torch.tensor([0, 1, 2])
    for name, fl in {
        "plain": {},
        "per_tau": dict(return_per_tau=True),
        "smooth": dict(smooth_dims=dims, smooth_horizon=H_VALID),
        "both": dict(return_per_tau=True, smooth_dims=dims, smooth_horizon=H_VALID),
    }.items():
        a = compute_fm_log_prob(action_head=head, **inp, **fl)
        b = compute_fm_log_prob(action_head=head, **inp, **fl, n_exec_steps=None)
        fa, fb = _flat(a), _flat(b)
        check(f"{name}: same positional return, bitwise",
              type(a) is type(b) and len(fa) == len(fb)
              and all(torch.equal(x, y) for x, y in zip(fa, fb)))
        s = compute_fm_log_prob(action_head=head, **inp, **fl, return_struct=True)
        check(f"{name}: struct exec fields are None when not requested",
              s.exec_log_probs is None and s.exec_per_tau is None)
    r = FMLogProbResult(torch.zeros(1), None, None, None, None)
    check("FMLogProbResult still builds from the five legacy positional fields",
          r.exec_log_probs is None and r.exec_per_tau is None)


def test_L2_values_match_hand_computation():
    print("\n[L2] exec_log_probs == the masked formula over steps < n (clean + jittered)")
    head = _Head(seed=3)
    inp = _inputs(B=4)
    W = head.model.W.detach()
    for label, nfi in (("clean", None), ("jittered", _jittered(inp))):
        r = compute_fm_log_prob(action_head=head, **inp, noise_for_input=nfi,
                                n_exec_steps=N_EXEC, return_struct=True)
        want_e = _hand_lp(W, inp["actions"], inp["noise"], inp["action_mask"],
                          inp["timesteps"], n_steps=N_EXEC, nfi=nfi)
        want_f = _hand_lp(W, inp["actions"], inp["noise"], inp["action_mask"],
                          inp["timesteps"], nfi=nfi)
        check(f"{label}: exec_log_probs == hand (mean over n*D_valid elements)",
              torch.allclose(r.exec_log_probs.detach(), want_e, atol=1e-6, rtol=1e-6),
              f"{r.exec_log_probs} vs {want_e}")
        check(f"{label}: log_probs still the full-horizon value",
              torch.allclose(r.log_probs.detach(), want_f, atol=1e-6, rtol=1e-6))
        check(f"{label}: the two differ (the mask is not a no-op here)",
              not torch.allclose(r.exec_log_probs.detach(), r.log_probs.detach(),
                                 atol=1e-4))
    # Normalisation: a MEAN over the executed prefix, not the prefix SUM over the
    # full valid count (which would read ~n/H_VALID of it).
    r = compute_fm_log_prob(action_head=head, **inp, n_exec_steps=N_EXEC,
                            return_struct=True)
    wrong = _hand_lp(W, inp["actions"], inp["noise"], inp["action_mask"],
                     inp["timesteps"], n_steps=N_EXEC) * (N_EXEC / H_VALID)
    check("normalised by the executed prefix's own valid count",
          not torch.allclose(r.exec_log_probs.detach(), wrong, atol=1e-6))


def test_L3_no_gradient_on_unexecuted_steps():
    print("\n[L3] d exec_log_probs / d W[h] == 0 exactly for every h >= n")
    head = _Head(seed=4)
    inp = _inputs(B=3)
    for nfi in (None, _jittered(inp)):
        head.model.W.grad = None
        r = compute_fm_log_prob(action_head=head, **inp, noise_for_input=nfi,
                                n_exec_steps=N_EXEC, return_struct=True)
        r.exec_log_probs.sum().backward()
        g = head.model.W.grad
        tag = "clean" if nfi is None else "jittered"
        check(f"{tag}: rows >= n_exec are exactly zero",
              torch.equal(g[N_EXEC:], torch.zeros_like(g[N_EXEC:])), f"{g}")
        check(f"{tag}: executed valid rows carry gradient",
              bool((g[:N_EXEC, :D_VALID].abs() > 0).all()))
        check(f"{tag}: padded dims stay zero", bool((g[:, D_VALID:] == 0).all()))
    head.model.W.grad = None
    r = compute_fm_log_prob(action_head=head, **inp, n_exec_steps=N_EXEC,
                            return_struct=True)
    r.log_probs.sum().backward()
    g = head.model.W.grad
    check("the full-horizon log_probs DO reach steps n..H_VALID-1 (control)",
          bool((g[N_EXEC:H_VALID, :D_VALID].abs() > 0).all()))


def test_L4_degenerate_prefix_is_bitwise_full():
    print("\n[L4] a prefix covering every valid step is bitwise the full log-prob")
    head = _Head(seed=5)
    inp = _inputs(B=3)
    for n in (H_VALID, H_VALID + 2, H_PAD):
        r = compute_fm_log_prob(action_head=head, **inp, return_per_tau=True,
                                noise_for_input=_jittered(inp),
                                n_exec_steps=n, return_struct=True)
        check(f"n={n}: exec_log_probs is bitwise log_probs",
              torch.equal(r.exec_log_probs, r.log_probs))
        check(f"n={n}: exec_per_tau is bitwise per_tau",
              torch.equal(r.exec_per_tau, r.per_tau))


def test_L5_per_tau_and_struct_fields():
    print("\n[L5] exec_per_tau: present iff return_per_tau, [K, B], averages to exec")
    head = _Head(seed=6)
    inp = _inputs(B=3, K=4)
    r = compute_fm_log_prob(action_head=head, **inp, n_exec_steps=N_EXEC,
                            return_per_tau=True, return_struct=True)
    check("exec_per_tau shape [K, B]", tuple(r.exec_per_tau.shape) == (4, 3))
    check("exec_per_tau.mean(0) == exec_log_probs",
          torch.allclose(r.exec_per_tau.mean(dim=0), r.exec_log_probs, atol=1e-7))
    r2 = compute_fm_log_prob(action_head=head, **inp, n_exec_steps=N_EXEC,
                             return_struct=True)
    check("exec_per_tau is None without return_per_tau", r2.exec_per_tau is None)
    check("exec_log_probs is identical with and without return_per_tau",
          torch.equal(r.exec_log_probs, r2.exec_log_probs))
    plain = compute_fm_log_prob(action_head=head, **inp)
    check("requesting exec leaves log_probs bitwise unchanged",
          torch.equal(r2.log_probs, plain))


def test_L6_validation():
    print("\n[L6] n_exec_steps validation")
    head = _Head()
    inp = _inputs()
    check("n_exec_steps without return_struct is rejected",
          _raises(lambda: compute_fm_log_prob(action_head=head, **inp,
                                              n_exec_steps=N_EXEC), ValueError))
    for bad in (0, -1, H_PAD + 1, True, 2.0, "3", np.int64(2)):
        check(f"n_exec_steps={bad!r} is rejected",
              _raises(lambda: compute_fm_log_prob(
                  action_head=head, **inp, n_exec_steps=bad, return_struct=True),
                  ValueError))
    m = _mask(3)
    m[1, :N_EXEC] = 0.0          # row 1 has no valid element in the prefix
    check("a row with no valid executed element is rejected",
          _raises(lambda: compute_fm_log_prob(
              action_head=head, **dict(inp, action_mask=m),
              n_exec_steps=N_EXEC, return_struct=True), ValueError))
    ok = compute_fm_log_prob(action_head=head, **inp, n_exec_steps=1,
                             return_struct=True)
    check("n_exec_steps=1 (replan every step) is accepted",
          bool(torch.isfinite(ok.exec_log_probs).all()))


def test_L7_no_rng_and_combinations():
    print("\n[L7] no RNG consumed; composes with vel_anchor and smoothness")
    head = _Head(seed=7)
    inp = _inputs()
    s0 = torch.get_rng_state()
    compute_fm_log_prob(action_head=head, **inp, n_exec_steps=N_EXEC,
                        return_per_tau=True, return_struct=True)
    check("torch RNG state unchanged", torch.equal(s0, torch.get_rng_state()))
    other = {"W": head.model.W.detach() + 0.1}
    anc = VelAnchor("params", other)
    r = compute_fm_log_prob(action_head=head, **inp, vel_anchor=anc,
                            n_exec_steps=N_EXEC, return_struct=True)
    r0 = compute_fm_log_prob(action_head=head, **inp, vel_anchor=anc,
                             return_struct=True)
    ex = compute_fm_log_prob(action_head=head, **inp, n_exec_steps=N_EXEC,
                             return_struct=True)
    check("with vel_anchor: anchor_dist is unchanged by n_exec_steps",
          torch.equal(r.anchor_dist, r0.anchor_dist))
    check("with vel_anchor: exec_log_probs equals the anchor-free call",
          torch.equal(r.exec_log_probs, ex.exec_log_probs))
    dims = torch.tensor([0, 1, 2])
    rs = compute_fm_log_prob(action_head=head, **inp, smooth_dims=dims,
                             smooth_horizon=H_VALID, n_exec_steps=N_EXEC,
                             return_struct=True)
    check("with smoothness: exec_log_probs equals the smooth-free call",
          torch.equal(rs.exec_log_probs, ex.exec_log_probs)
          and rs.smooth is not None)


# ═════════════════════════════════════════════════════════════════════════════
# Trainer level (the real ref pass + update, through the real compute_fm_log_prob)
# ═════════════════════════════════════════════════════════════════════════════

def _mask_np() -> np.ndarray:
    return _mask().numpy().astype(np.float32)


def _make_chunks(n=16, n_groups=2, seed=3, n_anchor=0) -> list:
    """Real ActionChunks with alternating-sign, all-distinct advantages."""
    g = torch.Generator().manual_seed(seed)
    chunks = []
    for i in range(n + n_anchor):
        anchor = i >= n
        sign = 1.0 if (i % 2 == 0 or anchor) else -1.0
        adv = 0.02 if anchor else sign * (1.0 + 0.13 * i)
        raw = (torch.randn(H_PAD, D_PAD, generator=g) * 0.5).numpy()
        noise = torch.randn(H_PAD, D_PAD, generator=g).numpy()
        chunks.append(ActionChunk(
            video_frames={}, state={}, language="task",
            action=raw[:H_VALID, :D_VALID].copy(),
            raw_action=raw, action_mask=_mask_np(), initial_noise=noise,
            advantage=adv, episode_idx=i // 4, chunk_idx=i % 4,
            episode_reward=1.0 if sign > 0 else 0.0, episode_success=sign > 0,
            group_id=99 if anchor else i % n_groups, is_anchor=anchor,
        ))
    return chunks


def _prepare_batch_stub(self, batch):
    pairs = [(c, m) for (c, m) in batch if c.raw_action is not None]
    if not pairs:
        return None
    valid = [c for c, _m in pairs]
    B = len(valid)
    self._test_batches.append((self._test_phase[0], list(valid)))
    return {
        "actions": torch.stack([torch.from_numpy(c.raw_action) for c in valid]).float(),
        "action_masks": torch.stack([torch.from_numpy(c.action_mask) for c in valid]).float(),
        "initial_noise": torch.stack(
            [torch.from_numpy(c.initial_noise) for c in valid]).float(),
        "advantages": torch.tensor([c.advantage for c in valid], dtype=torch.float32),
        "backbone_output": {"backbone_features": torch.zeros(B, 1, D_PAD)},
        "state_features": torch.zeros(B, 0, D_PAD),
        "embodiment_id": torch.zeros(B, dtype=torch.long),
        "modes": [m for _c, m in pairs],
    }, valid


@dataclasses.dataclass
class _Run:
    result: dict
    events: list
    calls: list          # every compute_fm_log_prob call: phase, kw, out, batch_idx
    batches: list        # (phase, [chunks]) per _prepare_batch call
    chunks: list
    policy: nn.Module
    trainer: GRPOTrainer
    stdout: str
    config: GRPOConfig

    @property
    def step_grads(self) -> list:
        return [g.reshape(H_PAD, D_PAD) for kind, g in self.events if kind == "step"]

    @property
    def train_calls(self) -> list:
        """Update-phase TRAINING forwards (the only calls passing smooth_instrument)."""
        return [c for c in self.calls
                if c["phase"] == "update" and "smooth_instrument" in c["kw"]]


def _config(**kw) -> GRPOConfig:
    base = dict(
        device="cpu", mini_batch_size=4, update_epochs=2,
        gradient_accumulation_steps=1, balanced_minibatch_training=False,
        dynamic_epoch_training=False, per_iteration_advantage_norm=False,
        positive_advantage_weight_scaling=False, kl_coef_last_iter=0.2,
        kl_coef_base_model=0.0, jitter_pos=0.0, jitter_neg=0.0,
        max_grad_norm=1e9, learning_rate=0.05, seed=7,
        tau_centers=[0.0, 0.3, 0.6], n_action_steps=N_EXEC,
    )
    base.update(kw)
    return GRPOConfig(**base)


def run(*, flag: bool, seed: int = 0, policy_seed: int = 0, n_chunks: int = 16,
        n_anchor: int = 0, epochs=None, chunks=None, after_ref=None, setup=None,
        **cfg_kw) -> _Run:
    if epochs is not None:
        cfg_kw["update_epochs"] = epochs
    cfg = _config(mask_loss_with_n_action_steps=flag, **cfg_kw)
    policy = _Policy(_Head(seed=policy_seed))
    chunks = chunks if chunks is not None else _make_chunks(n_chunks, n_anchor=n_anchor)
    events: list = []
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.config = cfg
    t.device = torch.device("cpu")
    t.model = policy
    t.optimizer = tga._RecordingSGD(policy.parameters(), lr=cfg.learning_rate,
                                    events=events)
    t.buffer = SimpleNamespace(_build_chunks=lambda: list(chunks))
    t.iteration = 1
    t._model_lock = threading.RLock()
    t._test_batches = []
    t._test_phase = ["ref"]
    t._prepare_batch = types.MethodType(_prepare_batch_stub, t)
    t._cache_encoded_features = lambda *a, **k: None
    t._ref_mse_stats = None
    t._chunk_gap_stats = None
    t._vel_anchor_start_stats = None
    if setup is not None:
        setup(t)

    calls: list = []
    real = train_grpo.compute_fm_log_prob

    def spy(**kw):
        out = real(**kw)
        calls.append({"phase": t._test_phase[0], "kw": kw, "out": out,
                      "batch_idx": len(t._test_batches) - 1})
        return out

    torch.manual_seed(seed)
    buf = io.StringIO()
    train_grpo.compute_fm_log_prob = spy
    try:
        with contextlib.redirect_stdout(buf):
            t._compute_ref_log_probs()
            if after_ref is not None:
                after_ref(chunks)
            t._test_phase[0] = "update"
            result = t._grpo_update()
    finally:
        train_grpo.compute_fm_log_prob = real
    return _Run(result, events, calls, t._test_batches, chunks, policy, t,
                buf.getvalue(), cfg)


def _lp(out, exec_side: bool) -> torch.Tensor:
    if isinstance(out, FMLogProbResult):
        return (out.exec_log_probs if exec_side else out.log_probs).detach()
    assert not exec_side
    return out.detach()


def _ready_rows(r: _Run, call) -> list:
    """The chunks a training forward scored, in row order (ready filter applied)."""
    batch = r.batches[call["batch_idx"]][1]
    cb = r.trainer.config.kl_coef_base_model > 0.0
    ex = r.trainer._exec_steps() is not None
    return [c for c in batch
            if c.ref_log_prob is not None and c.tau_samples is not None
            and (not cb or c.base_log_prob is not None)
            and (not ex or c.ref_log_prob_exec is not None)]


def _expected_stats(r: _Run, exec_side: bool) -> dict:
    """mean_ratio / mean_log_ratio_abs from the chosen pair; KL from the full pair."""
    ratios, labs, kls = [], [], []
    for call in r.train_calls:
        rows = _ready_rows(r, call)
        ref = torch.tensor([c.ref_log_prob_exec if exec_side else c.ref_log_prob
                            for c in rows], dtype=torch.float32)
        lr_ = _lp(call["out"], exec_side) - ref
        ratios.append(float(lr_.exp().mean()))
        labs.append(float(lr_.abs().mean()))
        x = (torch.tensor([c.ref_log_prob for c in rows], dtype=torch.float32)
             - _lp(call["out"], False))
        kls.append(float((x.exp() - x - 1.0).mean()))
    return {
        "mean_ratio": float(np.mean(ratios)),
        "mean_log_ratio_abs": float(np.mean(labs)),
        "kl_loss_last_iter": r.config.kl_coef_last_iter * float(np.mean(kls)),
    }


def _expected_clip_stats(r: _Run, exec_side: bool) -> dict:
    """Every other ratio consumer, recomputed from the chosen pair.

    Valid for anchor-free runs at the default per-minibatch z-score and a flat
    floor (clip_low_mse_coef == 0), which is how it is used below.
    """
    lo = 1 - r.config.clip_eps_low
    hi = 1 + r.config.clip_eps_high
    cf, rmax, rmin = [], [], []
    eff = {"pos": [0, 0], "neg": [0, 0]}
    N = D = 0.0
    for call in r.train_calls:
        rows = _ready_rows(r, call)
        ref = torch.tensor([c.ref_log_prob_exec if exec_side else c.ref_log_prob
                            for c in rows], dtype=torch.float32)
        ratio = (_lp(call["out"], exec_side) - ref).exp()
        pre = torch.tensor([c.advantage for c in rows], dtype=torch.float32)
        A = (pre - pre.mean()) / (pre.std() + 1e-8) if pre.numel() > 1 else pre
        moved = (ratio < lo) | (ratio > hi)
        cf.append(float(moved.float().mean()))
        rmax.append(float(ratio.max()))
        rmin.append(float(ratio.min()))
        s1 = A * ratio
        s2 = A * ratio.clamp(lo, hi)
        dead = moved & (s2 <= s1)
        for key, m in (("pos", A > 0), ("neg", A <= 0)):
            eff[key][0] += int(dead[m].sum())
            eff[key][1] += int(m.sum())
        rl = (-torch.min(s1, s2)).abs()
        N += float(rl[(pre <= 0) & (ratio >= lo)].sum())
        D += float(rl[(pre > 0) & (A > 0) & (ratio <= hi)].sum())
    return {
        "clipfrac": float(np.mean(cf)),
        "ratio_max": float(np.max(rmax)),
        "ratio_min": float(np.min(rmin)),
        "clipfrac_effective_pos": eff["pos"][0] / eff["pos"][1],
        "clipfrac_effective_neg": eff["neg"][0] / eff["neg"][1],
        "pos_adv_alive_neg_mass": N,
        "pos_adv_pos_mass": D,
    }


def test_T1_off_switch_in_tree():
    print("\n[T1] flag off: nothing requested, nothing stored, nothing printed")
    r = run(flag=False)
    check("no call received n_exec_steps or return_struct",
          not any({"n_exec_steps", "return_struct"} & set(c["kw"]) for c in r.calls))
    check("no chunk got ref_log_prob_exec",
          all(c.ref_log_prob_exec is None for c in r.chunks))
    check("ref_mse/* carries no exec_ key",
          not any(k.startswith("exec_") for k in (r.trainer._ref_mse_stats or {})))
    check("no executed-prefix console line",
          "executed steps" not in r.stdout and "Executed-step" not in r.stdout)
    want = _expected_stats(r, exec_side=False)
    check("mean_ratio is the full-horizon pair's",
          math.isclose(r.result["mean_ratio"], want["mean_ratio"], rel_tol=1e-6))
    r2 = run(flag=False)
    check("two flag-off runs are bitwise identical (weights and stats)",
          torch.equal(r.policy.action_head.model.W, r2.policy.action_head.model.W)
          and repr(r.result) == repr(r2.result))
    c = GRPOConfig(device="cpu")
    check("the flag defaults to False", c.mask_loss_with_n_action_steps is False)
    on = dataclasses.asdict(GRPOConfig(device="cpu", mask_loss_with_n_action_steps=True))
    off = dataclasses.asdict(c)
    check("switching it on changes exactly that one field",
          [k for k in off if off[k] != on[k]] == ["mask_loss_with_n_action_steps"])


def test_T2_ref_pass_values():
    print("\n[T2] ref pass stores ref_log_prob_exec == hand, full value untouched")
    on = run(flag=True)
    off = run(flag=False)
    W0 = _Head(seed=0).model.W.detach()
    ok_e = ok_f = True
    detail = ""
    for c in on.chunks:
        a = torch.from_numpy(c.raw_action)[None]
        e = torch.from_numpy(c.initial_noise)[None]
        m = torch.from_numpy(c.action_mask)[None]
        ts = torch.from_numpy(c.tau_samples)[:, None].to(torch.bfloat16)
        want_e = float(_hand_lp(W0, a, e, m, ts, n_steps=N_EXEC)[0])
        want_f = float(_hand_lp(W0, a, e, m, ts)[0])
        if not math.isclose(c.ref_log_prob_exec, want_e, rel_tol=1e-5, abs_tol=1e-6):
            ok_e, detail = False, f"{c.ref_log_prob_exec} vs {want_e}"
        if not math.isclose(c.ref_log_prob, want_f, rel_tol=1e-5, abs_tol=1e-6):
            ok_f = False
    check("every chunk's ref_log_prob_exec matches the hand formula", ok_e, detail)
    check("every chunk's ref_log_prob matches the full-horizon hand formula", ok_f)
    check("ref_log_prob and tau_samples are bitwise the flag-off run's",
          all(a.ref_log_prob == b.ref_log_prob
              and np.array_equal(a.tau_samples, b.tau_samples)
              for a, b in zip(on.chunks, off.chunks)))
    ref_calls = [c for c in on.calls if c["phase"] == "ref"]
    check("the ref call asked for the prefix via n_exec_steps=n_action_steps",
          ref_calls and all(c["kw"].get("n_exec_steps") == N_EXEC
                            and c["kw"].get("return_struct") for c in ref_calls))


def test_T3_surrogate_uses_exec_pair_kl_uses_full():
    print("\n[T3] surrogate ratio == exec pair; KL terms == full-horizon pair")
    r = run(flag=True)
    want = _expected_stats(r, exec_side=True)
    wrong = _expected_stats(r, exec_side=False)
    for k in ("mean_ratio", "mean_log_ratio_abs"):
        check(f"{k} is computed from the executed-prefix pair",
              math.isclose(r.result[k], want[k], rel_tol=1e-6),
              f"{r.result[k]} vs {want[k]}")
        check(f"... and is distinguishable from the full-horizon pair ({k})",
              not math.isclose(want[k], wrong[k], rel_tol=1e-4),
              f"{want[k]} vs {wrong[k]}")
    check("kl_loss_last_iter is computed from the FULL-horizon pair",
          math.isclose(r.result["kl_loss_last_iter"], want["kl_loss_last_iter"],
                       rel_tol=1e-6),
          f"{r.result['kl_loss_last_iter']} vs {want['kl_loss_last_iter']}")
    check("every training forward asked for the prefix",
          r.train_calls and all(c["kw"].get("n_exec_steps") == N_EXEC
                                for c in r.train_calls))
    check("optimizer steps fired", r.result.get("n_updates", 0) > 0)


def test_T3b_every_ratio_consumer_uses_exec_pair():
    print("\n[T3b] clipfrac / effective clipfrac / ratio tails / PAWS masses / anchor "
          "ratio follow the exec pair")
    keys = ("clipfrac", "ratio_max", "ratio_min", "clipfrac_effective_pos",
            "clipfrac_effective_neg")
    # Tight clips so a bound actually binds: the upper one under PAWS (which
    # also reports the masses), the lower one without it.
    for label, kw, side in (
        ("PAWS, upper bound binds", dict(
            positive_advantage_weight_scaling=True,
            positive_advantage_weight_target_ratio=1.5,
            clip_eps_low=0.02, clip_eps_high=0.05), "pos"),
        ("lower bound binds", dict(clip_eps_low=0.005, epochs=3), "neg"),
    ):
        r = run(flag=True, **kw)
        want = _expected_clip_stats(r, exec_side=True)
        wrong = _expected_clip_stats(r, exec_side=False)
        ks = keys + (("pos_adv_alive_neg_mass", "pos_adv_pos_mass")
                     if side == "pos" else ())
        check(f"{label}: precondition, clipfrac_effective_{side} > 0",
              want[f"clipfrac_effective_{side}"] > 0.0, f"{want}")
        for k in ks:
            check(f"{label}: {k} is the exec pair's",
                  math.isclose(r.result[k], want[k], rel_tol=1e-6, abs_tol=1e-12),
                  f"{r.result[k]} vs {want[k]}")
        differ = [k for k in ks
                  if not math.isclose(want[k], wrong[k], rel_tol=1e-4, abs_tol=1e-9)]
        check(f"{label}: the full-horizon pair would give different values",
              len(differ) >= 3, f"only {differ} differ")

    a = run(flag=True, include_anchor_groups=True, anchor_advantage=0.1,
            n_anchor=3, mini_batch_size=5, epochs=3, learning_rate=0.1)

    def anchor_ratio(exec_side):
        s = n = 0.0
        for call in a.train_calls:
            rows = _ready_rows(a, call)
            lp = _lp(call["out"], exec_side)
            for i, c in enumerate(rows):
                if c.is_anchor:
                    ref = c.ref_log_prob_exec if exec_side else c.ref_log_prob
                    s += math.exp(float(lp[i]) - float(torch.tensor(ref)))
                    n += 1
        return s / n

    got = a.result.get("mean_ratio_anchor", float("nan"))
    check("mean_ratio_anchor is the exec pair's",
          math.isclose(got, anchor_ratio(True), rel_tol=1e-5),
          f"{got} vs {anchor_ratio(True)}")
    check("... distinguishable from the full-horizon pair",
          not math.isclose(anchor_ratio(True), anchor_ratio(False), rel_tol=1e-4))


def test_T4_no_gradient_on_unexecuted_steps():
    print("\n[T4] every optimizer step: zero gradient on W[h >= n] (KL off)")
    on = run(flag=True, kl_coef_last_iter=0.0)
    off = run(flag=False, kl_coef_last_iter=0.0)
    check("steps fired", len(on.step_grads) > 2 and len(off.step_grads) > 2)
    check("flag on: rows >= n_action_steps are exactly zero at EVERY step",
          all(torch.equal(g[N_EXEC:], torch.zeros_like(g[N_EXEC:]))
              for g in on.step_grads))
    check("flag on: executed rows carry gradient",
          all(bool((g[:N_EXEC, :D_VALID].abs() > 0).any()) for g in on.step_grads))
    check("flag off (control): rows n..H_VALID-1 DO carry gradient",
          all(bool((g[N_EXEC:H_VALID, :D_VALID].abs() > 0).any())
              for g in off.step_grads))
    kl = run(flag=True, kl_coef_last_iter=0.5)
    check("flag on + KL: the full-horizon KL still reaches rows >= n after a step",
          any(bool((g[N_EXEC:H_VALID, :D_VALID].abs() > 0).any())
              for g in kl.step_grads[1:]))
    check("padded rows/dims never receive gradient",
          all(bool((g[H_VALID:] == 0).all()) and bool((g[:, D_VALID:] == 0).all())
              for g in on.step_grads + off.step_grads))


def test_T5_degenerate_prefix_matches_off():
    print("\n[T5] n_action_steps covering every valid step == flag off")
    for n in (H_VALID, H_PAD):
        on = run(flag=True, n_action_steps=n, kl_coef_last_iter=0.0)
        off = run(flag=False, n_action_steps=n, kl_coef_last_iter=0.0)
        check(f"n={n}, KL off: weights bitwise equal",
              torch.equal(on.policy.action_head.model.W,
                          off.policy.action_head.model.W))
        check(f"n={n}, KL off: every step gradient bitwise equal",
              len(on.step_grads) == len(off.step_grads)
              and all(torch.equal(a, b) for a, b in zip(on.step_grads, off.step_grads)))
        check(f"n={n}, KL off: stats identical",
              set(on.result) == set(off.result)
              and all(repr(on.result[k]) == repr(off.result[k]) for k in on.result))
        check(f"n={n}: ref_log_prob_exec is bitwise ref_log_prob",
              all(c.ref_log_prob_exec == c.ref_log_prob for c in on.chunks))
        check(f"n={n}: the inert NOTE is printed",
              "has no effect" in on.stdout)
        on_kl = run(flag=True, n_action_steps=n)
        off_kl = run(flag=False, n_action_steps=n)
        check(f"n={n}, KL on: weights equal to fp32 rounding",
              torch.allclose(on_kl.policy.action_head.model.W,
                             off_kl.policy.action_head.model.W, atol=1e-6, rtol=0))


def test_T6_rho_floor_and_drift_use_exec_mse_ref():
    print("\n[T6] clip_low_mse_coef budget and drift/* read the exec MSE_ref")
    # The stub's MSE_ref is O(1-6), so a small coefficient keeps every budget
    # below the |ln(1 - clip_eps_low)| ceiling, where exec and full differ.
    coef = 0.02
    r = run(flag=True, clip_low_mse_coef=coef, clip_eps_low=0.2, epochs=1)
    d = r.result.get("_drift_diag") or {}
    ceil = -math.log(1 - 0.2)

    def pooled(exec_side):
        bud, mref = [], []
        for call in r.train_calls:
            for c in _ready_rows(r, call):
                if c.advantage > 0 or c.is_anchor:
                    continue
                m = max(-(c.ref_log_prob_exec if exec_side else c.ref_log_prob), 0.0)
                bud.append(min(coef * m, ceil))
                mref.append(m)
        return float(np.mean(bud)), float(np.mean(mref)), max(bud)

    want_b, want_m, max_b = pooled(True)
    wrong_b, wrong_m, _ = pooled(False)
    check("precondition: budgets sit below the ceiling", max_b < ceil, f"{max_b}")
    check("drift/budget_mean == mean min(coef * MSE_ref_exec, ceiling)",
          math.isclose(d.get("budget_mean", float("nan")), want_b, rel_tol=1e-5),
          f"{d.get('budget_mean')} vs {want_b}")
    check("... distinguishable from the full-horizon budget",
          not math.isclose(want_b, wrong_b, rel_tol=1e-3), f"{want_b} vs {wrong_b}")
    check("drift/neg_mseref_all is the exec MSE_ref",
          math.isclose(d.get("neg_mseref_all", float("nan")), want_m, rel_tol=1e-5),
          f"{d.get('neg_mseref_all')} vs {want_m}")


def test_T7_ready_filter():
    print("\n[T7] a row without ref_log_prob_exec never reaches the loss")

    def drop_first(chunks):
        chunks[0].ref_log_prob_exec = None

    r = run(flag=True, after_ref=drop_first, epochs=1)
    first = r.chunks[0]
    n_prepared = sum(1 for ph, b in r.batches if ph == "update" for c in b if c is first)
    rows_scored = sum(int(c["kw"]["actions"].shape[0]) for c in r.train_calls)
    rows_prepared = sum(len(b) for ph, b in r.batches if ph == "update")
    check("the chunk was prepared at least once", n_prepared >= 1)
    check("... and every time it was dropped before the forward",
          rows_scored == rows_prepared - n_prepared,
          f"{rows_scored} vs {rows_prepared} - {n_prepared}")
    off = run(flag=False, after_ref=drop_first, epochs=1)
    check("flag off ignores the field (all prepared rows scored)",
          sum(int(c["kw"]["actions"].shape[0]) for c in off.train_calls)
          == sum(len(b) for ph, b in off.batches if ph == "update"))


def test_T8_combinations():
    print("\n[T8] composes with accumulation, anchors, jitter, PAWS, balanced, KL-base")
    combos = {
        "grad accumulation k=2": dict(gradient_accumulation_steps=2),
        "anchor rows": dict(include_anchor_groups=True, anchor_advantage=0.1,
                            n_anchor=3, mini_batch_size=5),
        "paired jitter": dict(jitter_pos=0.2, jitter_neg=0.05, epochs=1),
        "jitter-only": dict(jitter_pos=0.2, jitter_neg=0.0, jitter_paired=False),
        "PAWS": dict(positive_advantage_weight_scaling=True,
                     positive_advantage_weight_target_ratio=1.5),
        "balanced sampler": dict(balanced_minibatch_training=True),
        "per-iteration norm": dict(per_iteration_advantage_norm=True),
        "base-model KL": dict(kl_coef_base_model=0.3),
    }
    for name, kw in combos.items():
        r = run(flag=True, **kw)
        ok = r.result.get("n_updates", 0) > 0
        want = _expected_stats(r, exec_side=True)
        check(f"{name}: trains, and mean_ratio is the exec pair's",
              ok and math.isclose(r.result["mean_ratio"], want["mean_ratio"],
                                  rel_tol=1e-6),
              f"n_updates={r.result.get('n_updates')} "
              f"{r.result.get('mean_ratio')} vs {want['mean_ratio']}")
        if name == "anchor rows":
            check("anchor rows: mean_ratio_anchor present and finite",
                  math.isfinite(r.result.get("mean_ratio_anchor", float("nan"))))
        if name == "jitter-only":
            # Every row is jittered, so the first micro-batch always measures.
            jd = r.result.get("_jitter_diag") or {}
            jd_off = run(flag=False, **kw).result.get("_jitter_diag") or {}
            check("jitter-only: jitter/* measured, and identical to the flag-off "
                  "run (a full-horizon field diagnostic)",
                  "gap_pos" in jd and repr(jd) == repr(jd_off), f"{jd} vs {jd_off}")
        if name == "PAWS":
            check("PAWS: k and masses finite",
                  all(math.isfinite(r.result.get(k, float("nan")))
                      for k in ("pos_adv_weight_k", "pos_adv_alive_neg_mass",
                                "pos_adv_pos_mass")))
        if name == "base-model KL":
            check("base-model KL: kl_loss_base_model present and finite",
                  math.isfinite(r.result.get("kl_loss_base_model", float("nan"))))


def test_T9_vel_anchor_combination():
    print("\n[T9] composes with the velocity anchor (one struct call per forward)")

    def with_anchor(t):
        W = t.model.action_head.model.W.detach()
        t._vel_anchor = VelAnchor("params", {"W": W + 0.05})
        t._vel_anchor_source = "test"
        t._vel_anchor_split = (None, N_EXEC)
        t._vel_anchor_equals_start = False

    r = run(flag=True, vel_anchor_coef=0.5, setup=with_anchor)
    check("training forwards carried BOTH vel_anchor and n_exec_steps",
          r.train_calls and all("vel_anchor" in c["kw"]
                                and c["kw"].get("n_exec_steps") == N_EXEC
                                for c in r.train_calls))
    want = _expected_stats(r, exec_side=True)
    check("mean_ratio is the exec pair's",
          math.isclose(r.result["mean_ratio"], want["mean_ratio"], rel_tol=1e-6))
    va = r.result.get("_vel_anchor") or {}
    check("vel_anchor/train_mean present and > 0", va.get("train_mean", 0.0) > 0.0)
    st = r.trainer._vel_anchor_start_stats or {}
    check("ref pass still gathers vel_anchor/start_* (and the exec split)",
          "start_mean" in st and "start_exec_frac" in st, f"{st}")
    check("ref pass stored ref_log_prob_exec alongside",
          all(c.ref_log_prob_exec is not None for c in r.chunks))


def test_T10_smoothness_combination():
    print("\n[T10] composes with the roughness constraint")

    def with_smooth(t):
        t.smooth_active = True
        t._smooth_dims = torch.tensor([0, 1, 2])
        t._smooth_dims_list = [0, 1, 2]
        t._smooth_horizon = H_VALID
        t._smooth_hf_ref = torch.tensor(0.0)
        t._smooth_calib_sum = None
        t._smooth_n_exec = N_EXEC

    r = run(flag=True, smooth_coef=0.1, setup=with_smooth)
    check("trains", r.result.get("n_updates", 0) > 0)
    check("smooth/* stats present", "smooth_hf_mean" in r.result)
    want = _expected_stats(r, exec_side=True)
    check("mean_ratio is the exec pair's",
          math.isclose(r.result["mean_ratio"], want["mean_ratio"], rel_tol=1e-6))


def test_T11_grad_probe_scores_the_prefix():
    print("\n[T11] gradprobe: both legs score the executed prefix")
    seen = []
    real_cap = GRPOTrainer._grad_probe_capture_jittered

    def cap(self, **kw):
        seen.append(kw["per_tau"])
        return real_cap(self, **kw)

    GRPOTrainer._grad_probe_capture_jittered = cap
    try:
        r = run(flag=True, grad_probe_every=1, epochs=1)
    finally:
        GRPOTrainer._grad_probe_capture_jittered = real_cap
    gp = r.result.get("_grad_probe") or {}
    check("probes ran", gp.get("n_probes", 0) > 0, f"{gp}")
    train_exec_pt = [c["out"].exec_per_tau for c in r.train_calls
                     if c["out"].exec_per_tau is not None]
    check("phase 1 received the training forward's exec_per_tau",
          seen and all(any(s is e for e in train_exec_pt) for s in seen))
    clean = [c for c in r.calls if c["phase"] == "update"
             and "smooth_instrument" not in c["kw"]]
    check("the clean leg asked for the prefix too",
          clean and all(c["kw"].get("n_exec_steps") == N_EXEC for c in clean))
    # With jitter off the two legs are the same functional, so R is a null
    # reading. A leg scoring the full horizon would put R at O(1).
    check("jitter off: R is a null reading (legs agree)",
          gp.get("R_max", 1.0) < 1e-3, f"R_max={gp.get('R_max')}")


def test_T12_grad_probe_finish_unit():
    print("\n[T12] _grad_probe_finish: g_R is the gradient of the prefix MSE")
    for flag in (False, True):
        head = _Head(seed=11)
        pol = _Policy(head)
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.config = _config(mask_loss_with_n_action_steps=flag)
        t.device = torch.device("cpu")
        t.model = pol
        inp = _inputs(B=3, K=3)
        params = [head.model.W]
        state = {"flat_jit": torch.zeros(H_PAD * D_PAD), "g_jit_norm": 0.0,
                 "n_pos_rows": 2, "tau_sub": 3, "pos_idx": torch.tensor([0, 2])}
        rec = t._grad_probe_finish(
            state, probe_params=params, ready_backbone=inp["backbone_output"],
            ready_state_features=inp["state_features"],
            ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
            ready_masks=inp["action_mask"], ready_noise=inp["noise"],
            timesteps=inp["timesteps"],
        )
        W = head.model.W.detach().clone().requires_grad_(True)
        idx = torch.tensor([0, 2])
        lp = _hand_lp(W, inp["actions"][idx], inp["noise"][idx],
                      inp["action_mask"][idx], inp["timesteps"][:, idx],
                      n_steps=N_EXEC if flag else None)
        (g,) = torch.autograd.grad((-lp).mean(), W)
        check(f"flag={flag}: g_reinforce_norm == ||grad of the "
              f"{'prefix' if flag else 'full'} MSE||",
              rec is not None and math.isclose(rec["g_reinforce_norm"],
                                               float(g.norm()), rel_tol=1e-5),
              f"{rec and rec['g_reinforce_norm']} vs {float(g.norm())}")


def test_T13_ref_mse_exec_stats_and_logging():
    print("\n[T13] ref_mse/exec_* values, the console line, and TB emission")
    r = run(flag=True, epochs=1)
    s = r.trainer._ref_mse_stats or {}
    sig = [c for c in r.chunks if not c.is_anchor]
    mse_e = np.array([-c.ref_log_prob_exec for c in sig])
    mse_f = np.array([-c.ref_log_prob for c in sig])
    n_full = H_VALID * D_VALID
    n_ex = N_EXEC * D_VALID
    check("exec_mean == mean(-ref_log_prob_exec)",
          math.isclose(s.get("exec_mean", float("nan")), float(mse_e.mean()),
                       rel_tol=1e-9))
    check("exec_p10 / exec_p90 are the percentiles of MSE_ref_exec",
          math.isclose(s["exec_p10"], float(np.percentile(mse_e, 10)), rel_tol=1e-9)
          and math.isclose(s["exec_p90"], float(np.percentile(mse_e, 90)),
                           rel_tol=1e-9))
    check("exec_ratio_ceiling_max == exp(max MSE_ref_exec)",
          math.isclose(s.get("exec_ratio_ceiling_max", float("nan")),
                       math.exp(float(mse_e.max())), rel_tol=1e-9))
    pos = np.array([c.advantage > 0 for c in sig])
    check("exec_pos_mean / exec_neg_mean split by advantage sign",
          math.isclose(s["exec_pos_mean"], float(mse_e[pos].mean()), rel_tol=1e-9)
          and math.isclose(s["exec_neg_mean"], float(mse_e[~pos].mean()),
                           rel_tol=1e-9))
    want_share = float((mse_e * n_ex).sum() / (mse_f * n_full).sum())
    check("exec_share == pooled executed share of the residual energy",
          math.isclose(s.get("exec_share", float("nan")), want_share, rel_tol=1e-9),
          f"{s.get('exec_share')} vs {want_share}")
    check("exec_share is inside (0, 1)", 0.0 < s["exec_share"] < 1.0)
    check("the pre-existing ref_mse keys are unchanged by the flag",
          {k: v for k, v in s.items() if not k.startswith("exec_")}
          == (run(flag=False, epochs=1).trainer._ref_mse_stats or {}))
    check("console line printed", "MSE_ref over executed steps 0..3" in r.stdout)
    check("no inert NOTE when the prefix is a strict subset",
          "has no effect" not in r.stdout)

    class _W:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, value, step):
            self.scalars[tag] = value

        def add_text(self, *a, **k):
            pass

    for flag, rr in ((True, r), (False, run(flag=False, epochs=1))):
        rr.trainer.writer = _W()
        with contextlib.redirect_stdout(io.StringIO()):
            rr.trainer._log_metrics(1, {"success_rate": 0.5}, rr.result, lr=1e-5,
                                    iter_time=1.0)
        tags = rr.trainer.writer.scalars
        has = [t for t in tags if t.startswith("ref_mse/exec_")]
        if flag:
            check("flag on: ref_mse/exec_{mean,p10,p90,pos_mean,neg_mean,"
                  "ratio_ceiling_max,share} emitted",
                  sorted(has) == sorted(f"ref_mse/exec_{k}" for k in (
                      "mean", "p10", "p90", "pos_mean", "neg_mean",
                      "ratio_ceiling_max", "share")),
                  f"{has}")
        else:
            check("flag off: no ref_mse/exec_* scalar", not has, f"{has}")


def test_T14_config_validation():
    print("\n[T14] config validation (only when the flag is on)")
    for bad in (0, -1, True, 2.5):
        check(f"flag on, n_action_steps={bad!r} rejected",
              _raises(lambda: GRPOConfig(device="cpu", n_action_steps=bad,
                                         mask_loss_with_n_action_steps=True),
                      ValueError))
    check("flag OFF leaves n_action_steps unvalidated (untouched behaviour)",
          not _raises(lambda: GRPOConfig(device="cpu", n_action_steps=0)))
    for ok in (1, 8, 16):
        check(f"flag on, n_action_steps={ok} accepted",
              not _raises(lambda: GRPOConfig(device="cpu", n_action_steps=ok,
                                             mask_loss_with_n_action_steps=True)))
    try:
        import tyro
    except ImportError:
        print("  (tyro not installed; CLI parse check skipped)")
        return
    c = tyro.cli(GRPOConfig, args=["--device", "cpu",
                                   "--mask-loss-with-n-action-steps"])
    check("CLI: --mask-loss-with-n-action-steps turns it on",
          c.mask_loss_with_n_action_steps is True)
    c = tyro.cli(GRPOConfig, args=["--device", "cpu"])
    check("CLI: absent flag leaves it off", c.mask_loss_with_n_action_steps is False)


def test_T15_banner():
    print("\n[T15] the startup banner states the mask only when it is on")
    import test_vel_anchor as tva
    for flag in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            t, _calls, _saves = tva._loop_trainer(
                tmp, config_overrides=dict(mask_loss_with_n_action_steps=flag))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                t.train()
        line = "Executed-step loss mask: ON"
        if flag:
            check("flag on: banner line printed with the executed range",
                  line in out.getvalue() and "steps 0..7" in out.getvalue())
        else:
            check("flag off: no banner line", line not in out.getvalue())
    # The clip_low_mse_coef born-dead hint points at the MSE_ref the budget uses.
    for flag, key in ((False, "compare against ref_mse/p10"),
                      (True, "compare against ref_mse/exec_p10")):
        with tempfile.TemporaryDirectory() as tmp:
            t, _calls, _saves = tva._loop_trainer(tmp, config_overrides=dict(
                mask_loss_with_n_action_steps=flag, clip_low_mse_coef=8.0,
                jitter_neg=0.05))
            out = io.StringIO()
            with contextlib.redirect_stdout(out), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t.train()
        check(f"flag {'on' if flag else 'off'}: born-dead hint says {key!r}",
              key in out.getvalue())


def main():
    import re

    def _key(f):
        m = re.match(r"test_([LT])(\d+)([a-z]?)_", f.__name__)
        return (m.group(1) != "L", int(m.group(2)), m.group(3))

    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
    for t in sorted(tests, key=_key):
        t()
    print()
    if _failures:
        print(f"\033[31m{len(_failures)} FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\033[32mAll executed-step mask tests passed.\033[0m")


if __name__ == "__main__":
    main()
