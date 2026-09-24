"""Tests for the velocity-anchor drift penalty (`GRPOConfig.vel_anchor_coef`).

WHAT IS UNDER TEST.

    loss += vel_anchor_coef * reduce(D),
    D_row = mean_k  mean_valid  (v_theta(x_k) - v_anchor(x_k))^2

evaluated at the training forward's own DiT inputs (the jittered x' on positive
rows), where v_anchor is the base DiT (LoRA disabled) or frozen LoRA tensors
swapped in with `torch.func.functional_call`.

Two layers, numbered as in PLAN_vel_anchor.md section 3:
  * LOSS LEVEL (1-11): the REAL `compute_fm_log_prob` on a stub action head
    whose `model` holds a real PEFT-injected Linear, so both anchor paths act on
    real LoRA layers, and D is checked against an independent hand computation
    (the LoRA formula, never `disabled_adapters` / `functional_call`).
  * TRAINER LEVEL (12-23): the REAL `_grpo_update_inner`,
    `_compute_ref_log_probs`, `_jitter_gap_diagnostics`, `_setup_vel_anchor`,
    `_log_metrics` and `train()` on CPU, with test_grad_accum's value-pinned
    conventions (param-independent gradients, plain SGD, `__new__` trainers).

Off-switch bit-identity against HEAD is an OUT-OF-TREE differential, run when
this feature landed: materialise HEAD's `scripts/grpo/*.py` with `git show`,
then drive `test_grad_accum.run_update`, `test_grad_probe.run_probe` and the
real `compute_fm_log_prob` (flag combinations incl. the RNG-sampling path) on
both trees and compare weights, full stats dicts and the RNG state. Test 13 is
the in-tree half.

Run with the project venv (needs torch + peft; CPU is fine):
    .venv/bin/python scripts/grpo/test_vel_anchor.py
"""

import contextlib
import copy
import io
import math
import re
import sys
import tempfile
import threading
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: E402

import train_grpo  # noqa: E402
import test_grad_accum as tga  # noqa: E402
from fm_log_prob import (  # noqa: E402
    AnchorSplit,
    FMLogProbResult,
    VelAnchor,
    compute_fm_log_prob,
)
from grpo_config import GRPOConfig  # noqa: E402
from lora_dit import save_lora_checkpoint  # noqa: E402
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


def _close(a, b, atol=1e-6, rtol=1e-5) -> bool:
    return bool(torch.allclose(torch.as_tensor(a), torch.as_tensor(b),
                               atol=atol, rtol=rtol))


# ═════════════════════════════════════════════════════════════════════════════
# Loss-level harness: a real PEFT LoRA layer inside a stub action head
# ═════════════════════════════════════════════════════════════════════════════

T_COEF = 1e-3     # the stub DiT's timestep term, so different taus differ
LORA_SCALE = 2.0  # lora_alpha / r used by _Head


class _TinyDiT(nn.Module):
    """DiT stand-in: ONE real Linear that PEFT injects LoRA into.

    Accepts the real `AlternateVLDiT`-style keyword call and returns
    `(out, None)`. Every forward is recorded (adapter state, input) so a test
    can pair the current and anchor forwards.
    """

    def __init__(self, D):
        super().__init__()
        self.proj = nn.Linear(D, D)
        self.raise_when_disabled = False
        self.seen: list = []

    def forward(self, hidden_states, encoder_hidden_states=None,
                encoder_attention_mask=None, timestep=None,
                return_all_hidden_states=False, image_mask=None,
                backbone_attention_mask=None):
        disabled = bool(self.proj._disable_adapters)
        if self.raise_when_disabled and disabled:
            raise RuntimeError("boom inside the anchor forward")
        self.seen.append((disabled, hidden_states.detach().clone()))
        out = self.proj(hidden_states)
        return out + T_COEF * timestep.to(out.dtype)[:, None, None], None


class _Head(nn.Module):
    """Action-head stand-in: identity encoder/decoder around the PEFT DiT."""

    def __init__(self, D=4, rank=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_timestep_buckets = 1000
        self.num_inference_timesteps = 4

        class _Cfg:
            add_pos_embed = False
            use_alternate_vl_dit = False
            noise_s = 0.999
        self.config = _Cfg()
        self.model = _TinyDiT(D)
        inject_adapter_in_model(
            LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                       bias="none", target_modules=["proj"]),
            self.model, adapter_name="default",
        )
        # Mirror apply_lora_to_dit: only LoRA factors are trainable.
        for n, p in self.model.named_parameters():
            p.requires_grad = "lora_" in n

    def action_encoder(self, noisy, t_disc, emb):
        return noisy

    def action_decoder(self, model_output, emb):
        return model_output


class _Policy(nn.Module):
    """`policy.action_head.model` is the PEFT DiT, as the trainer expects."""

    def __init__(self, head):
        super().__init__()
        self.action_head = head


def _lora(head) -> dict:
    return {n: p for n, p in head.model.named_parameters() if "lora_" in n}


def _randomize_lora(head, seed, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in _lora(head).values():
            p.copy_(torch.randn(p.shape, generator=g) * scale)


def _random_lora_params(head, seed, scale=0.5) -> dict:
    """A DIFFERENT LoRA tensor set, DiT-relative names (a 'checkpoint')."""
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(p.shape, generator=g) * scale
            for n, p in _lora(head).items()}


def _inputs(B=3, H=6, D=4, K=3, seed=1, mask=None):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(B, H, D, generator=g)
    noise = torch.randn(B, H, D, generator=g)
    ts = torch.rand(K, B, generator=g) * 0.9
    return dict(
        backbone_output={"backbone_features": torch.zeros(B, 1, D)},
        state_features=torch.zeros(B, 0, D),
        embodiment_id=torch.zeros(B, dtype=torch.long),
        actions=actions,
        action_mask=torch.ones(B, H, D) if mask is None else mask,
        timesteps=ts, noise=noise, n_samples=K,
    )


def _jittered(inp, lam, seed=9):
    """[K, B, H, D] eps' = sqrt(1-lam^2) eps + lam xi, per-row lam."""
    eps, K = inp["noise"], inp["n_samples"]
    g = torch.Generator().manual_seed(seed)
    xi = torch.randn((K,) + tuple(eps.shape), generator=g)
    lam = torch.as_tensor(lam, dtype=torch.float32)[None, :, None, None]
    return (1.0 - lam * lam).sqrt() * eps.unsqueeze(0) + lam * xi


def _v_formula(head, x, td, lora=None):
    """Velocity by the LoRA FORMULA: base(x) + s * x A^T B^T + t-term.

    `lora=None` means "LoRA disabled"; otherwise a DiT-relative LoRA dict.
    Independent of both `disabled_adapters` and `functional_call`.
    """
    layer = head.model.proj
    out = layer.base_layer(x)
    if lora is not None:
        A = lora["proj.lora_A.default.weight"]
        Bm = lora["proj.lora_B.default.weight"]
        out = out + LORA_SCALE * (x @ A.T @ Bm.T)
    return out + T_COEF * td.to(out.dtype)[:, None, None]


def _hand_D(head, inp, anchor_lora, noise_for_input=None, split=None):
    """D [B] (and split parts) from scratch: build x_k, evaluate both fields."""
    a, eps, m = inp["actions"], inp["noise"], inp["action_mask"]
    ts, K = inp["timesteps"], inp["n_samples"]
    live = {n: p.detach() for n, p in _lora(head).items()}
    valid = m.sum(dim=(1, 2))
    D = torch.zeros(a.shape[0])
    grip = torch.zeros(a.shape[0])
    exe = torch.zeros(a.shape[0])
    per_tau = []
    with torch.no_grad():
        for k in range(K):
            x_in = eps if noise_for_input is None else noise_for_input[k]
            t = ts[k]
            x = (1 - t)[:, None, None] * x_in + t[:, None, None] * a
            td = (t * 1000).long()
            v_cur = _v_formula(head, x, td, live)
            v_anc = _v_formula(head, x, td, anchor_lora)
            sq = (v_cur.float() - v_anc.float()) ** 2 * m
            d_k = sq.sum(dim=(1, 2)) / valid
            per_tau.append(d_k)
            D += d_k
            if split is not None:
                gcols, n_exec = split
                grip += sq[:, :, list(gcols)].sum(dim=(1, 2)) / valid
                exe += sq[:, :n_exec].sum(dim=(1, 2)) / valid
    return D / K, grip / K, exe / K, torch.stack(per_tau, dim=0)


def _flat(x):
    """Flatten a compute_fm_log_prob return (tensor / nested tuples)."""
    if isinstance(x, torch.Tensor):
        return [x]
    out = []
    for y in x:
        out += _flat(y)
    return out


# ─── 1. Off-path contract ────────────────────────────────────────────────────

def test_01_off_path_contract():
    print("\n[1] Off path: no new args == vel_anchor=None, bitwise, all flag sets")
    head = _Head()
    _randomize_lora(head, 3)
    inp = _inputs()
    dims = torch.tensor([0, 1, 2])
    flag_sets = {
        "plain": {},
        "per_tau": dict(return_per_tau=True),
        "smooth": dict(smooth_dims=dims, smooth_horizon=4),
        "both": dict(return_per_tau=True, smooth_dims=dims, smooth_horizon=4),
    }
    for name, fl in flag_sets.items():
        a = compute_fm_log_prob(action_head=head, **inp, **fl)
        b = compute_fm_log_prob(action_head=head, **inp, **fl, vel_anchor=None)
        fa, fb = _flat(a), _flat(b)
        check(f"{name}: same return shape/type",
              type(a) is type(b) and len(fa) == len(fb))
        check(f"{name}: bitwise equal", all(torch.equal(x, y) for x, y in zip(fa, fb)))
        s = compute_fm_log_prob(action_head=head, **inp, **fl, return_struct=True)
        check(f"{name}: struct log_probs == legacy log_probs",
              torch.equal(s.log_probs, fa[0]))
        check(f"{name}: struct anchor fields are None",
              s.anchor_dist is None and s.anchor_split is None)
    raised = False
    try:
        compute_fm_log_prob(action_head=head, **inp, vel_anchor=VelAnchor("base"))
    except ValueError:
        raised = True
    check("vel_anchor without return_struct is rejected (no positional slot)", raised)


# ─── 2. Zero at the anchor ───────────────────────────────────────────────────

def test_02_zero_at_anchor():
    print("\n[2] D == 0 at the anchor")
    head = _Head()                                  # PEFT init: lora_B == 0
    inp = _inputs()
    r = compute_fm_log_prob(action_head=head, **inp, vel_anchor=VelAnchor("base"),
                            return_struct=True)
    check("base anchor, B == 0: D is exactly 0",
          torch.equal(r.anchor_dist, torch.zeros_like(r.anchor_dist)),
          f"{r.anchor_dist}")
    r.anchor_dist.sum().backward()
    grads = [p.grad for p in _lora(head).values()]
    check("base anchor, B == 0: zero gradient on every LoRA factor",
          all(g is None or torch.equal(g, torch.zeros_like(g)) for g in grads))

    head2 = _Head()
    _randomize_lora(head2, 4)
    same = {n: p.detach().clone() for n, p in _lora(head2).items()}
    r2 = compute_fm_log_prob(action_head=head2, **inp,
                             vel_anchor=VelAnchor("params", same),
                             return_struct=True)
    check("checkpoint anchor == current weights: D is exactly 0",
          torch.equal(r2.anchor_dist, torch.zeros_like(r2.anchor_dist)),
          f"{r2.anchor_dist}")


# ─── 3. Hand computation ─────────────────────────────────────────────────────

def test_03_hand_computation():
    print("\n[3] D matches an independent loop (both anchor kinds, clean + jittered)")
    head = _Head()
    _randomize_lora(head, 5)
    inp = _inputs(B=4)
    ckpt = _random_lora_params(head, 11)
    nfi = _jittered(inp, [0.25, 0.0, 0.125, 0.25])
    for label, anchor, anchor_lora in (
        ("base", VelAnchor("base"), None),
        ("params", VelAnchor("params", ckpt), ckpt),
    ):
        for jit_label, nf in (("clean", None), ("jittered", nfi)):
            r = compute_fm_log_prob(action_head=head, **inp, noise_for_input=nf,
                                    vel_anchor=anchor, return_struct=True)
            want = _hand_D(head, inp, anchor_lora, noise_for_input=nf)[0]
            check(f"{label}/{jit_label}: D == hand D",
                  _close(r.anchor_dist.detach(), want, atol=1e-6, rtol=1e-5),
                  f"{r.anchor_dist.detach().tolist()} vs {want.tolist()}")
            check(f"{label}/{jit_label}: D > 0 (the check is not vacuous)",
                  bool((want > 1e-4).all()))


# ─── 4. Same inputs ──────────────────────────────────────────────────────────

def test_04_same_inputs():
    print("\n[4] Current and anchor forwards see the identical (jittered) x'")
    head = _Head()
    _randomize_lora(head, 6)
    inp = _inputs(B=3, K=3)
    nfi = _jittered(inp, [0.25, 0.25, 0.0])
    for label, anchor in (("base", VelAnchor("base")),
                          ("params", VelAnchor("params", _random_lora_params(head, 2)))):
        head.model.seen.clear()
        compute_fm_log_prob(action_head=head, **inp, noise_for_input=nfi,
                            vel_anchor=anchor, return_struct=True)
        seen = head.model.seen
        K = inp["n_samples"]
        check(f"{label}: exactly two DiT forwards per tau", len(seen) == 2 * K,
              f"{len(seen)}")
        pairs_equal = all(torch.equal(seen[2 * k][1], seen[2 * k + 1][1])
                          for k in range(K))
        check(f"{label}: each tau's two forwards got bitwise-identical inputs",
              pairs_equal)
        a, ts = inp["actions"], inp["timesteps"]
        want_x = [(1 - ts[k])[:, None, None] * nfi[k] + ts[k][:, None, None] * a
                  for k in range(K)]
        check(f"{label}: and that input IS the jittered x'_k",
              all(torch.equal(seen[2 * k][1], want_x[k]) for k in range(K)))
        if label == "base":
            check("base: current forward has LoRA on, anchor forward has it off",
                  all((not seen[2 * k][0]) and seen[2 * k + 1][0] for k in range(K)))


# ─── 5. Gradient ─────────────────────────────────────────────────────────────

def test_05_gradient():
    print("\n[5] Gradient of D: finite difference, descent, anchor untouched")
    head = _Head()
    _randomize_lora(head, 8)
    inp = _inputs(B=3)
    ckpt = _random_lora_params(head, 12)
    for label, anchor in (("base", VelAnchor("base")),
                          ("params", VelAnchor("params", ckpt))):
        params = _lora(head)

        def total():
            return compute_fm_log_prob(action_head=head, **inp, vel_anchor=anchor,
                                       return_struct=True).anchor_dist.sum()

        for p in params.values():
            p.grad = None
        total().backward()
        ok = True
        detail = ""
        for name, p in params.items():
            idx = (0, 0)
            h = 1e-3
            with torch.no_grad():
                orig = float(p[idx])
                p[idx] = orig + h
                up = float(total())
                p[idx] = orig - h
                dn = float(total())
                p[idx] = orig
            fd = (up - dn) / (2 * h)
            ag = float(p.grad[idx])
            if not math.isclose(fd, ag, rel_tol=2e-3, abs_tol=1e-4):
                ok = False
                detail = f"{name}: autograd {ag:.6g} vs FD {fd:.6g}"
        check(f"{label}: autograd dD/dtheta == central finite difference", ok, detail)
        before = float(total())
        with torch.no_grad():
            for p in params.values():
                p -= 0.05 * p.grad
        after = float(total())
        check(f"{label}: one SGD step on D alone reduces D ({before:.4f} -> {after:.4f})",
              after < before)
        if anchor.params is not None:
            check("params: anchor tensors never require or receive grad",
                  all((not t.requires_grad) and t.grad is None
                      for t in anchor.params.values()))
        check(f"{label}: frozen base weights never receive grad",
              head.model.proj.base_layer.weight.grad is None)


# ─── 6. LoRA restored after a base-anchor call ───────────────────────────────

def test_06_lora_restored():
    print("\n[6] Adapters re-enabled after a base-anchor call, even on an exception")
    head = _Head()
    _randomize_lora(head, 9)
    inp = _inputs()
    layers = [m for m in head.model.modules() if isinstance(m, BaseTunerLayer)]
    x = torch.randn(2, 6, 4)

    def lora_on_and_active():
        flags_ok = all(not m._disable_adapters for m in layers)
        with torch.no_grad():
            differs = not torch.equal(head.model.proj(x),
                                      head.model.proj.base_layer(x))
        return flags_ok and differs

    compute_fm_log_prob(action_head=head, **inp, vel_anchor=VelAnchor("base"),
                        return_struct=True)
    check("after a normal call: adapters on and contributing", lora_on_and_active())
    head.model.raise_when_disabled = True
    raised = False
    try:
        compute_fm_log_prob(action_head=head, **inp, vel_anchor=VelAnchor("base"),
                            return_struct=True)
    except RuntimeError:
        raised = True
    head.model.raise_when_disabled = False
    check("the anchor forward really raised inside the disabled block", raised)
    check("after the exception: adapters on and contributing", lora_on_and_active())


# ─── 7. Checkpoint anchor leaves the live model untouched ────────────────────

def test_07_params_anchor_leaves_model_untouched():
    print("\n[7] functional_call anchor: live params, flags and saved LoRA unchanged")
    head = _Head()
    _randomize_lora(head, 10)
    policy = _Policy(head)
    inp = _inputs()
    live = dict(head.model.named_parameters())
    snap = {n: (p.detach().clone(), p.data_ptr(), p.requires_grad)
            for n, p in live.items()}
    keys_before = list(policy.state_dict().keys())
    active_before = head.model.proj.active_adapters
    with tempfile.TemporaryDirectory() as tmp:
        save_lora_checkpoint(policy, Path(tmp) / "before")
        compute_fm_log_prob(action_head=head, **inp,
                            vel_anchor=VelAnchor("params", _random_lora_params(head, 1)),
                            return_struct=True)
        save_lora_checkpoint(policy, Path(tmp) / "after")
        sb = torch.load(Path(tmp) / "before" / "lora_weights.pt")
        sa = torch.load(Path(tmp) / "after" / "lora_weights.pt")
    live_after = dict(head.model.named_parameters())
    check("same parameter objects",
          all(live_after[n] is live[n] for n in live))
    check("values unchanged (bitwise)",
          all(torch.equal(live_after[n].detach(), snap[n][0]) for n in live))
    check("storage unchanged (data_ptr)",
          all(live_after[n].data_ptr() == snap[n][1] for n in live))
    check("requires_grad unchanged",
          all(live_after[n].requires_grad == snap[n][2] for n in live))
    check("active adapter unchanged and adapters enabled",
          head.model.proj.active_adapters == active_before
          and not head.model.proj._disable_adapters)
    check("state_dict keys unchanged", list(policy.state_dict().keys()) == keys_before)
    check("save_lora_checkpoint output identical before/after",
          sb.keys() == sa.keys() and all(torch.equal(sb[k], sa[k]) for k in sb))


# ─── 8. Masking ──────────────────────────────────────────────────────────────

def test_08_masking():
    print("\n[8] Padded elements contribute nothing; per-row normalisation")
    head = _Head()
    _randomize_lora(head, 13)
    B, H, D = 3, 6, 4
    mask = torch.zeros(B, H, D)
    mask[0, :6, :4] = 1.0          # 24 valid
    mask[1, :3, :2] = 1.0          # 6 valid
    mask[2, :4, :3] = 1.0          # 12 valid
    inp = _inputs(B=B, H=H, D=D, mask=mask)
    r = compute_fm_log_prob(action_head=head, **inp, vel_anchor=VelAnchor("base"),
                            return_struct=True)
    want = _hand_D(head, inp, None)[0]
    check("masked D == hand D with the same mask (per-row valid counts)",
          _close(r.anchor_dist.detach(), want), f"{r.anchor_dist.tolist()} vs {want.tolist()}")
    full = _hand_D(head, _inputs(B=B, H=H, D=D), None)[0]
    check("and differs from the unmasked value (the mask is doing work)",
          not _close(r.anchor_dist.detach(), full, atol=1e-4, rtol=1e-3))
    # A mask-sum normaliser (not mean over all elements): row 1's value must be
    # its masked SUM over 6 elements / 6.
    sq_sum_row1 = want[1] * 6.0
    check("row with 6 valid elements normalised by 6",
          _close(r.anchor_dist.detach()[1] * 6.0, sq_sum_row1))


# ─── 9. Split fractions ──────────────────────────────────────────────────────

def test_09_split_fractions():
    print("\n[9] Gripper-column and executed-step parts")
    head = _Head()
    _randomize_lora(head, 14)
    inp = _inputs(B=3, H=6, D=4)
    gcol, n_exec = 2, 3
    r = compute_fm_log_prob(action_head=head, **inp, vel_anchor=VelAnchor("base"),
                            vel_anchor_split=(gcol, n_exec), return_struct=True)
    D, grip, exe, _ = _hand_D(head, inp, None, split=((gcol,), n_exec))
    check("split is an AnchorSplit", isinstance(r.anchor_split, AnchorSplit))
    check("gripper part == hand", _close(r.anchor_split.gripper, grip))
    check("exec part == hand", _close(r.anchor_split.exec, exe))
    other = [c for c in range(4) if c != gcol]
    r_other = compute_fm_log_prob(action_head=head, **inp,
                                  vel_anchor=VelAnchor("base"),
                                  vel_anchor_split=(other, None), return_struct=True)
    check("gripper part + other-columns part == D",
          _close(r.anchor_split.gripper + r_other.anchor_split.gripper,
                 r.anchor_dist.detach()))
    check("exec None -> exec part absent", r_other.anchor_split.exec is None)
    late = D - exe
    check("exec part + steps >= n_exec part == D (hand complement)",
          _close(r.anchor_split.exec + late, r.anchor_dist.detach()))
    check("split parts carry no graph",
          not r.anchor_split.gripper.requires_grad and not r.anchor_split.exec.requires_grad)
    for bad in ((9, None), (None, 0), (None, 99)):
        raised = False
        try:
            compute_fm_log_prob(action_head=head, **inp, vel_anchor=VelAnchor("base"),
                                vel_anchor_split=bad, return_struct=True)
        except ValueError:
            raised = True
        check(f"out-of-range split {bad} rejected", raised)


# ─── 10. No RNG use ──────────────────────────────────────────────────────────

def test_10_no_rng():
    print("\n[10] An anchored call consumes no RNG")
    head = _Head()
    _randomize_lora(head, 15)
    inp = _inputs()
    ckpt = _random_lora_params(head, 3)
    for label, anchor in (("base", VelAnchor("base")),
                          ("params", VelAnchor("params", ckpt))):
        s0 = torch.get_rng_state()
        compute_fm_log_prob(action_head=head, **inp, noise_for_input=_jittered(inp, [0.2] * 3),
                            vel_anchor=anchor, vel_anchor_split=(1, 2),
                            return_struct=True)
        check(f"{label}: torch RNG state unchanged", torch.equal(s0, torch.get_rng_state()))


# ─── 11. Struct return, every flag combination ───────────────────────────────

def test_11_struct_every_combination():
    print("\n[11] return_struct fills every field correctly in every combination")
    head = _Head()
    _randomize_lora(head, 16)
    inp = _inputs(B=3, K=3)
    K, B = 3, 3
    dims = torch.tensor([0, 1])
    base_lp = compute_fm_log_prob(action_head=head, **inp)
    base_pt = compute_fm_log_prob(action_head=head, **inp, return_per_tau=True)[1]
    base_sm = compute_fm_log_prob(action_head=head, **inp, smooth_dims=dims,
                                  smooth_horizon=4)[1]
    ok_all = True
    detail = ""
    n = 0
    for per_tau in (False, True):
        for smooth in (False, True):
            for anchor in (None, "base"):
                for split in ((False, True) if anchor else (False,)):
                    for apt in ((False, True) if anchor else (False,)):
                        kw = {}
                        if per_tau:
                            kw["return_per_tau"] = True
                        if smooth:
                            kw.update(smooth_dims=dims, smooth_horizon=4)
                        if anchor:
                            kw["vel_anchor"] = VelAnchor("base")
                        if split:
                            kw["vel_anchor_split"] = (1, 2)
                        if apt:
                            kw["vel_anchor_per_tau"] = True
                        r = compute_fm_log_prob(action_head=head, **inp, **kw,
                                                return_struct=True)
                        n += 1
                        conds = [
                            isinstance(r, FMLogProbResult),
                            torch.equal(r.log_probs, base_lp),
                            (r.per_tau is None) == (not per_tau),
                            (not per_tau) or torch.equal(r.per_tau, base_pt),
                            (r.smooth is None) == (not smooth),
                            (not smooth) or all(torch.equal(a, b) for a, b in
                                                zip(r.smooth, base_sm)),
                            (r.anchor_dist is None) == (anchor is None),
                            (anchor is None) or tuple(r.anchor_dist.shape)
                            == ((K, B) if apt else (B,)),
                            (r.anchor_split is None) == (not split),
                        ]
                        if not all(conds):
                            ok_all = False
                            detail = f"combo {kw.keys()} failed at {conds.index(False)}"
    check(f"all {n} combinations: fields present/absent/shaped correctly, and "
          f"log_probs / per_tau / smooth bitwise equal to the unanchored call",
          ok_all, detail)
    r1 = compute_fm_log_prob(action_head=head, **inp, vel_anchor=VelAnchor("base"),
                             return_struct=True)
    r2 = compute_fm_log_prob(action_head=head, **inp, vel_anchor=VelAnchor("base"),
                             vel_anchor_per_tau=True, return_struct=True)
    check("per-tau anchor_dist averages to the [B] value",
          _close(r2.anchor_dist.mean(dim=0), r1.anchor_dist, atol=1e-7))


# ═════════════════════════════════════════════════════════════════════════════
# Trainer-level harness (test_grad_accum conventions)
# ═════════════════════════════════════════════════════════════════════════════

class _ActionHeadStub(nn.Module):
    """`action_head.model.eval()` must resolve; the ref pass reads config.noise_s."""

    def __init__(self):
        super().__init__()
        self.model = nn.Identity()
        self.config = types.SimpleNamespace(noise_s=0.999)


class _TinyModel(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.action_head = _ActionHeadStub()


def _row_d(actions: torch.Tensor):
    """Stand-in D value (> 0) and its param-independent gradient d(D)/dw."""
    a = actions.reshape(actions.shape[0], -1).mean(dim=1).to(torch.float32)
    d0 = 0.02 + 0.05 * a * a
    g = torch.stack([torch.cos(a), 0.5 * a], dim=1)
    return d0, g


def _fake_factory(model, calls, phase, *, nan_train_call=None, d_from_w=False,
                  first_d=None, jit_extra=0.0, d_by_train_call=None,
                  zero_grad_train_calls=(), grad_blowup=False, d_zero=False):
    """Value-pinned stand-in honouring the real return contracts.

    log_prob: value delta, gradient f (test_grad_accum). D: value d0 (or
    d0 + ||w||^2 with d_from_w, or 0 with d_zero), gradient g. The jittered leg
    adds jit_extra * mean((eps' - eps)^2) to D so jac_part_pos is non-zero.
    d_by_train_call: {training-forward index: forced D value};
    zero_grad_train_calls: training-forward indices whose D gradient is 0;
    grad_blowup: finite D, infinite gradient on every training forward.
    """
    d_by_train_call = dict(d_by_train_call or {})
    if first_d is not None:
        d_by_train_call.setdefault(0, first_d)

    def fake(**kw):
        actions = kw["actions"]
        B = actions.shape[0]
        K = int(kw["n_samples"])
        f, delta = tga._row_feature(actions, 0.05)
        w = model.w
        lp = delta + ((f @ w) - (f @ w.detach()))
        leg = ("train" if "smooth_instrument" in kw
               else "diag" if kw.get("return_per_tau") else "other")
        rec = {"phase": phase[0], "leg": leg, "keys": set(kw),
               "w": w.detach().clone(), "B": B,
               "feat": actions.reshape(B, -1)[:, 0].clone()}
        per_tau = lp.unsqueeze(0).expand(K, -1) if kw.get("return_per_tau") else None
        smooth = None
        if kw.get("smooth_dims") is not None:
            pin = lp.unsqueeze(1) - lp.detach().unsqueeze(1)
            smooth = (torch.cat((torch.full_like(pin, 1.2) + pin,
                                 torch.full_like(pin, 2.0)), dim=1),
                      torch.cat((torch.full_like(pin, 0.9),
                                 torch.full_like(pin, 2.0)), dim=1).detach())
        anchor = kw.get("vel_anchor")
        if anchor is None:
            calls.append(rec)
            if kw.get("return_struct"):
                return FMLogProbResult(lp, per_tau, smooth, None, None)
            extras = [e for e in (per_tau, smooth) if e is not None]
            return (lp, *extras) if extras else lp
        d0, g = _row_d(actions)
        value = d0 + (w.detach() ** 2).sum() if d_from_w else d0
        if d_zero:
            value = torch.zeros_like(value)
        n_train = sum(1 for c in calls if c["leg"] == "train")
        if leg == "train" and n_train in d_by_train_call:
            value = torch.full_like(value, float(d_by_train_call[n_train]))
        if leg == "train" and n_train in zero_grad_train_calls:
            g = torch.zeros_like(g)
        nfi = kw.get("noise_for_input")
        if nfi is not None and jit_extra:
            eps = kw["noise"]
            value = value + jit_extra * ((nfi - eps.unsqueeze(0)) ** 2).mean(
                dim=(0, 2, 3))
        gw = g @ w
        if leg == "train" and grad_blowup:
            gw = tga._ScaleGrad.apply(gw, float("inf"))
        D = value + (gw - (g @ w.detach()))
        if leg == "train" and nan_train_call is not None and n_train == nan_train_call:
            D = D + float("nan")
        rec["D"] = D.detach().clone()
        rec["g"] = g.clone()
        calls.append(rec)
        split = None
        sp = kw.get("vel_anchor_split")
        if sp is not None:
            split = AnchorSplit(
                gripper=0.25 * D.detach() if sp[0] is not None else None,
                exec=0.6 * D.detach() if sp[1] is not None else None,
            )
        dist = D.unsqueeze(0).expand(K, -1) if kw.get("vel_anchor_per_tau") else D
        return FMLogProbResult(lp, per_tau, smooth, dist, split)

    return fake


def _prepare_batch_stub(self, batch):
    valid = [c for (c, _m) in batch]
    modes = [m for (_c, m) in batch]
    B = len(valid)
    actions = torch.stack([torch.from_numpy(c.raw_action) for c in valid]).to(
        torch.float32).unsqueeze(-1)[:, :, 0]                     # [B, 1, 1]
    adv = torch.tensor([c.advantage for c in valid], dtype=torch.float32)
    # Row -> advantage, in forward order (tga feats are NOT unique per chunk).
    getattr(self, "_test_prepared", []).append(adv.clone())
    return {
        "actions": actions,
        "action_masks": torch.ones_like(actions),
        "initial_noise": torch.full_like(actions, 0.1),
        "advantages": adv,
        "backbone_output": {"backbone_features": torch.zeros(B, 1, 1)},
        "state_features": torch.zeros(B, 1, 1),
        "embodiment_id": torch.zeros(B, dtype=torch.long),
        "modes": modes,
    }, valid


class _Harness:
    """A `__new__`-built trainer over tga chunks, with the stand-in patched in."""

    def __init__(self, *, coef=0.1, k=1, epochs=1, n_chunks=16, mb_size=4,
                 lr=0.1, grad_probe_every=0, n_anchor_chunks=0,
                 equals_start=False, w0=(0.3, -0.2), split=None,
                 fake_kwargs=None, config_overrides=None):
        cfg_kwargs = dict(
            device="cpu", mini_batch_size=mb_size, update_epochs=epochs,
            gradient_accumulation_steps=k, balanced_minibatch_training=False,
            dynamic_epoch_training=False, per_iteration_advantage_norm=False,
            positive_advantage_weight_scaling=False, kl_coef_last_iter=0.2,
            kl_coef_base_model=0.0, jitter_pos=0.0, jitter_neg=0.0,
            max_grad_norm=1e9, learning_rate=lr, seed=7,
            vel_anchor_coef=coef, grad_probe_every=grad_probe_every,
            include_anchor_groups=n_anchor_chunks > 0,
            anchor_advantage=0.15 if n_anchor_chunks else 0.0,
        )
        cfg_kwargs.update(config_overrides or {})
        self.config = GRPOConfig(**cfg_kwargs)
        self.chunks = tga._make_chunks(n_chunks)
        for i in range(n_anchor_chunks):
            feat = 0.11 * (i + 1)
            self.chunks.append(tga._Chunk(
                advantage=0.01, feat=feat, group_id=99, is_anchor=True,
                ref_log_prob=0.0, base_log_prob=0.0,
                tau_samples=np.zeros(len(self.config.tau_centers), dtype=np.float32),
                raw_action=np.full((1, 1), feat, dtype=np.float32),
            ))
        for c in self.chunks:
            c.tau_samples = np.zeros(len(self.config.tau_centers), dtype=np.float32)
        self.events: list = []
        self.calls: list = []
        self.phase = ["update"]
        self.model = _TinyModel(w0)
        self.w0 = self.model.w.detach().clone()
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.config = self.config
        t.device = torch.device("cpu")
        t.model = self.model
        t.optimizer = tga._RecordingSGD(self.model.parameters(), lr=lr, events=self.events)
        t.buffer = types.SimpleNamespace(_build_chunks=lambda: list(self.chunks))
        t.iteration = 1
        t._model_lock = threading.RLock()
        t._test_prepared = []
        t._prepare_batch = types.MethodType(_prepare_batch_stub, t)
        t._cache_encoded_features = lambda *a, **kw: None
        if coef > 0.0:
            t._vel_anchor = VelAnchor("base")
            t._vel_anchor_source = "base"
            t._vel_anchor_split = split
        t._vel_anchor_equals_start = equals_start
        self.trainer = t
        self.fake = _fake_factory(self.model, self.calls, self.phase,
                                  **(fake_kwargs or {}))
        self.stdout = ""

    @contextlib.contextmanager
    def _patched(self):
        real = train_grpo.compute_fm_log_prob
        train_grpo.compute_fm_log_prob = self.fake
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                yield
        finally:
            train_grpo.compute_fm_log_prob = real
            self.stdout += buf.getvalue()

    def ref_pass(self):
        self.phase[0] = "ref"
        with self._patched():
            self.trainer._compute_ref_log_probs()
        self.phase[0] = "update"

    def update(self) -> dict:
        self.phase[0] = "update"
        with self._patched():
            return self.trainer._grpo_update()

    @property
    def step_grads(self):
        return [g for kind, g in self.events if kind == "step"]

    @property
    def train_calls(self):
        return [c for c in self.calls if c["leg"] == "train" and c["phase"] == "update"]


class _FakeWriter:
    def __init__(self):
        self.scalars = {}
        self.texts = {}
        self.text_calls = []

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = value

    def add_text(self, tag, text, global_step=0):
        self.texts[tag] = text
        self.text_calls.append(tag)


def _log(trainer, update_stats):
    trainer.writer = _FakeWriter()
    with contextlib.redirect_stdout(io.StringIO()):
        trainer._log_metrics(1, {"success_rate": 0.5}, update_stats, lr=1e-5,
                             iter_time=1.0)
    return trainer.writer


# ─── 12. Loss composition ────────────────────────────────────────────────────

def test_12_loss_composition():
    print("\n[12] loss rises by exactly coef x reduced D (mean / sum-over-divisor)")
    c = 0.7
    off = _Harness(coef=0.0)
    r0 = off.update()
    on = _Harness(coef=c)
    r1 = on.update()
    reds = [float(tc["D"].mean()) for tc in on.train_calls]
    want = c * float(np.mean(reds))
    check("no anchor rows: result loss - off loss == coef * mean_mb mean_rows D",
          math.isclose(r1["loss"] - r0["loss"], want, rel_tol=1e-5, abs_tol=1e-8),
          f"{r1['loss'] - r0['loss']:.8g} vs {want:.8g}")
    check("vel_anchor/loss reports that same term",
          math.isclose(r1["_vel_anchor"]["loss"], want, rel_tol=1e-6))
    check("clip_loss unchanged by the penalty", r1["clip_loss"] == r0["clip_loss"])

    an_off = _Harness(coef=0.0, n_anchor_chunks=3, mb_size=5)
    ra0 = an_off.update()
    an = _Harness(coef=c, n_anchor_chunks=3, mb_size=5)
    ra1 = an.update()
    m = re.search(r"alongside (\d+) signal row", an.stdout)
    S = int(m.group(1)) if m else None
    check("anchor rows were in play (schedule banner printed)", S is not None,
          an.stdout[-400:])
    if S is not None:
        sums = [float(tc["D"].sum()) / S for tc in an.train_calls]
        means = [float(tc["D"].mean()) for tc in an.train_calls]
        want_a = c * float(np.mean(sums))
        check("anchors in play: delta loss == coef * mean_mb sum_rows D / signal_mb_size",
              math.isclose(ra1["loss"] - ra0["loss"], want_a, rel_tol=1e-5, abs_tol=1e-8),
              f"{ra1['loss'] - ra0['loss']:.8g} vs {want_a:.8g}")
        check("... and that is NOT the plain mean reduction (divisor path taken)",
              not math.isclose(want_a, c * float(np.mean(means)), rel_tol=1e-4))


# ─── 13. Off-switch bit-identity (in-tree half) ──────────────────────────────

def test_13_off_switch():
    print("\n[13] coef 0: nothing requested, nothing emitted, weights/stats identical")
    a = _Harness(coef=0.0, epochs=2)
    ra = a.update()
    with tempfile.TemporaryDirectory() as tmp:
        pol = _Policy(_Head())
        save_lora_checkpoint(pol, Path(tmp) / "iter_0001")
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter("always")
            b = _Harness(coef=0.0, epochs=2, config_overrides=dict(
                vel_anchor_path=str(Path(tmp) / "iter_0001")))
        check("path with coef 0 warns (inert)",
              any("inert" in str(x.message) for x in wlog))
        rb = b.update()
        tb = b.trainer
        tb.model_holder = None
        # setup with coef 0 builds nothing, even with a path set.
        tb.model = pol
        tb._setup_vel_anchor()
        check("_setup_vel_anchor with coef 0 builds no anchor", tb._vel_anchor is None)
        tb.model = b.model
    check("no fake call ever received vel_anchor / return_struct / split",
          not any({"vel_anchor", "return_struct", "vel_anchor_split",
                   "vel_anchor_per_tau"} & c["keys"] for c in a.calls + b.calls))
    check("no _vel_anchor stats key", "_vel_anchor" not in ra and "_vel_anchor" not in rb)
    check("weights identical", torch.equal(a.model.w, b.model.w))
    check("step grads identical",
          len(a.step_grads) == len(b.step_grads)
          and all(torch.equal(x, y) for x, y in zip(a.step_grads, b.step_grads)))
    check("stats identical",
          set(ra) == set(rb) and all(repr(ra[k]) == repr(rb[k]) for k in ra))
    w = _log(a.trainer, ra)
    check("no vel_anchor/* TB scalar or text",
          not any(t.startswith("vel_anchor/") for t in list(w.scalars) + list(w.texts)))


# ─── 14. Gradient accumulation ───────────────────────────────────────────────

def test_14_grad_accumulation():
    print("\n[14] Accumulation with the penalty on")
    per_mb = _Harness(coef=0.5, k=1, epochs=2)
    per_mb.update()
    acc = _Harness(coef=0.5, k=2, epochs=2)
    acc.update()
    ok = len(acc.step_grads) * 2 == len(per_mb.step_grads)
    for j, g in enumerate(acc.step_grads):
        mean_g = torch.stack(per_mb.step_grads[2 * j:2 * j + 2]).mean(dim=0)
        ok = ok and _close(g, mean_g, atol=1e-7)
    check("k=2 step gradient == mean of the two k=1 micro-batch gradients", ok)

    def penalty_part(h_on, h_off):
        return [x - y for x, y in zip(h_on.step_grads, h_off.step_grads)]

    full_on = _Harness(coef=0.5, k=1, mb_size=8)
    full_on.update()
    full_off = _Harness(coef=0.0, k=1, mb_size=8)
    full_off.update()
    split_on = _Harness(coef=0.5, k=2, mb_size=4)
    split_on.update()
    split_off = _Harness(coef=0.0, k=2, mb_size=4)
    split_off.update()
    same_rows = [set(c["feat"].tolist()) for c in full_on.train_calls] == [
        set(split_on.train_calls[2 * j]["feat"].tolist())
        | set(split_on.train_calls[2 * j + 1]["feat"].tolist())
        for j in range(len(full_on.train_calls))
    ]
    check("the k=2 windows hold exactly the rows of the full batches", same_rows)
    pf, ps = penalty_part(full_on, full_off), penalty_part(split_on, split_off)
    check("penalty gradient: k=2 x mb=4 accumulated == one mb=8 full batch",
          len(pf) == len(ps) and all(_close(x, y, atol=1e-7) for x, y in zip(pf, ps)),
          f"{[p.tolist() for p in pf]} vs {[p.tolist() for p in ps]}")
    check("and it is non-zero (the check is not vacuous)",
          all(float(p.abs().sum()) > 1e-4 for p in pf))


# ─── 15. Non-finite guard ────────────────────────────────────────────────────

def test_15_nonfinite_guard():
    print("\n[15] A NaN distance drops only its micro-batch")
    clean = _Harness(coef=0.3)
    rc = clean.update()
    h = _Harness(coef=0.3, fake_kwargs=dict(nan_train_call=1))
    r = h.update()
    check("n_skipped_nonfinite == 1", r["n_skipped_nonfinite"] == 1,
          str(r.get("n_skipped_nonfinite")))
    check("one fewer micro-batch trained",
          r["n_micro_batches"] == rc["n_micro_batches"] - 1)
    check("weights finite", bool(torch.isfinite(h.model.w).all()))
    check("vel_anchor/train_mean finite",
          math.isfinite(r["_vel_anchor"]["train_mean"]))


# ─── 16. Force-balance probe ─────────────────────────────────────────────────

def test_16_force_balance():
    print("\n[16] grad_ratio / grad_cos vs direct gradients; probe changes nothing")
    c = 0.4
    h = _Harness(coef=c, grad_probe_every=1, mb_size=4,
                 config_overrides=dict(kl_coef_last_iter=0.0))
    r = h.update()
    ratios, coses = [], []
    prepared = h.trainer._test_prepared
    check("one prepared batch per training forward (alignment precondition)",
          len(prepared) == len(h.train_calls))
    for tc, adv in zip(h.train_calls, prepared):
        actions = tc["feat"].reshape(-1, 1, 1)
        f, delta = tga._row_feature(actions, 0.05)
        A = (adv - adv.mean()) / (adv.std() + 1e-8)
        g_clip = -((A * torch.exp(delta)).unsqueeze(1) * f).mean(dim=0)
        g_pen = c * tc["g"].mean(dim=0)
        ratios.append(float(g_pen.norm() / g_clip.norm()))
        coses.append(float(torch.dot(g_pen, g_clip) / (g_pen.norm() * g_clip.norm())))
    va = r.get("_vel_anchor", {})
    check("grad_ratio == mean over probes of ||g_pen|| / ||g_clip||",
          "grad_ratio" in va and math.isclose(va["grad_ratio"], float(np.mean(ratios)),
                                              rel_tol=1e-5),
          f"{va.get('grad_ratio')} vs {np.mean(ratios)}")
    check("grad_cos == mean over probes of cos(g_pen, g_clip)",
          "grad_cos" in va and math.isclose(va["grad_cos"], float(np.mean(coses)),
                                            rel_tol=1e-5, abs_tol=1e-7),
          f"{va.get('grad_cos')} vs {np.mean(coses)}")
    h_off = _Harness(coef=c, grad_probe_every=0, mb_size=4,
                     config_overrides=dict(kl_coef_last_iter=0.0))
    r_off = h_off.update()
    check("optimizer steps bit-identical with and without the probe",
          len(h.step_grads) == len(h_off.step_grads)
          and all(torch.equal(x, y) for x, y in zip(h.step_grads, h_off.step_grads)))
    check("final weights bit-identical", torch.equal(h.model.w, h_off.model.w))
    check("absent at grad_probe_every == 0",
          not {"grad_ratio", "grad_cos"} & set(r_off.get("_vel_anchor", {})))
    h0 = _Harness(coef=0.0, grad_probe_every=1)
    r0 = h0.update()
    check("absent at coef == 0", "_vel_anchor" not in r0)


# ─── 17. Metrics ─────────────────────────────────────────────────────────────

VEL_TAGS = {"coef", "start_mean", "start_pos", "start_neg", "start_p90",
            "start_gripper_frac", "start_exec_frac", "train_mean",
            "train_last_epoch_mean", "loss", "grad_ratio", "grad_cos",
            "jac_part_pos"}


def test_17_metrics():
    print("\n[17] Every vel_anchor/* metric present+finite when on, absent when off")
    h = _Harness(coef=0.2, epochs=2, grad_probe_every=2, split=((0,), 1),
                 fake_kwargs=dict(d_from_w=True, jit_extra=0.5),
                 config_overrides=dict(jitter_pos=0.125, jitter_paired=False))
    h.ref_pass()
    start = dict(h.trainer._vel_anchor_start_stats or {})
    ref_calls = [c for c in h.calls if c["phase"] == "ref"]
    check("ref pass ran the anchored forward", bool(ref_calls)
          and all("vel_anchor" in c["keys"] for c in ref_calls))
    check("start_* measured at the START weights (before any optimizer step)",
          all(torch.equal(c["w"], h.w0) for c in ref_calls))
    ref_d = torch.cat([c["D"] for c in ref_calls])
    check("start_mean == mean of the ref-pass D over signal chunks",
          math.isclose(start.get("start_mean", float("nan")), float(ref_d.mean()),
                       rel_tol=1e-6))
    check("start_gripper_frac / start_exec_frac are the pooled part shares",
          math.isclose(start.get("start_gripper_frac", -1), 0.25, rel_tol=1e-5)
          and math.isclose(start.get("start_exec_frac", -1), 0.6, rel_tol=1e-5))
    r = h.update()
    w = _log(h.trainer, r)
    got = {t.split("/", 1)[1] for t in w.scalars if t.startswith("vel_anchor/")}
    check("every table scalar emitted", VEL_TAGS <= got, f"missing {VEL_TAGS - got}")
    check("all finite", all(math.isfinite(w.scalars[f"vel_anchor/{k}"]) for k in got))
    check("vel_anchor/source text written once",
          w.texts.get("vel_anchor/source") == "vel_anchor anchor = base")
    check("jac_part_pos moved out of jitter/*",
          "jitter/_vel_anchor_jac_part_pos" not in w.scalars
          and w.scalars.get("vel_anchor/jac_part_pos", 0.0) > 0.0)
    tcs = h.train_calls
    n_ep = len(tcs) // 2
    last = torch.cat([c["D"] for c in tcs[n_ep:]])
    allr = torch.cat([c["D"] for c in tcs])
    check("train_last_epoch_mean averages ONLY the final epoch",
          math.isclose(r["_vel_anchor"]["train_last_epoch_mean"], float(last.mean()),
                       rel_tol=1e-6)
          and not math.isclose(float(last.mean()), float(allr.mean()), rel_tol=1e-6))
    check("train_mean averages every trained row",
          math.isclose(r["_vel_anchor"]["train_mean"], float(allr.mean()), rel_tol=1e-6))
    off = _Harness(coef=0.0)
    off.ref_pass()
    ro = off.update()
    wo = _log(off.trainer, ro)
    check("off: no start stats, no vel_anchor/* scalars",
          off.trainer._vel_anchor_start_stats is None
          and not any(t.startswith("vel_anchor/") for t in wo.scalars))


# ─── 18. jac_part_pos ────────────────────────────────────────────────────────

def _diag_trainer(head):
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.config = GRPOConfig(device="cpu", jitter_pos=0.125)
    t.device = torch.device("cpu")
    t.model = _Policy(head)
    return t


def test_18_jac_part():
    print("\n[18] jac_part_pos == (jittered - clean) positive-row D, same weights")
    head = _Head()
    _randomize_lora(head, 21)
    B = 4
    inp = _inputs(B=B, K=3)
    lam = torch.tensor([0.125, 0.125, 0.0, 0.0])
    nfi = _jittered(inp, lam)
    pos = torch.tensor([True, True, False, False])
    common = dict(
        ready_backbone=inp["backbone_output"],
        ready_state_features=inp["state_features"],
        ready_embodiment_id=inp["embodiment_id"],
        ready_actions=inp["actions"], ready_masks=inp["action_mask"],
        ready_noise=inp["noise"], timesteps=inp["timesteps"],
        lam_row=lam, pos_adv_mask=pos,
        fixed_row_mask=torch.zeros(B, dtype=torch.bool),
        jitter_row_mask=torch.ones(B, dtype=torch.bool),
    )
    t = _diag_trainer(head)
    out = t._jitter_gap_diagnostics(noise_for_input=nfi, vel_anchor=VelAnchor("base"),
                                    **common)
    fm = dict(action_head=head, **inp)
    dc = compute_fm_log_prob(**fm, vel_anchor=VelAnchor("base"),
                             return_struct=True).anchor_dist
    dj = compute_fm_log_prob(**fm, noise_for_input=nfi, vel_anchor=VelAnchor("base"),
                             return_struct=True).anchor_dist
    want = float((dj - dc)[pos].mean())
    check("jac_part_pos matches two direct calls",
          math.isclose(out.get("_vel_anchor_jac_part_pos", float("nan")), want,
                       rel_tol=1e-5, abs_tol=1e-8),
          f"{out.get('_vel_anchor_jac_part_pos')} vs {want}")
    check("and is non-zero here", abs(want) > 1e-6)
    out_off = t._jitter_gap_diagnostics(noise_for_input=nfi, vel_anchor=None, **common)
    check("jitter/* values unchanged by the anchor",
          set(out_off) == set(out) - {"_vel_anchor_jac_part_pos"}
          and all(out_off[k] == out[k] for k in out_off))
    lam0 = torch.tensor([0.0, 0.0, 0.0, 0.0])
    out0 = t._jitter_gap_diagnostics(
        noise_for_input=_jittered(inp, lam0),
        vel_anchor=VelAnchor("base"), **{**common, "lam_row": lam0})
    check("plam == 0 (eps' == eps on positive rows): jac_part_pos == 0",
          out0.get("_vel_anchor_jac_part_pos") == 0.0,
          str(out0.get("_vel_anchor_jac_part_pos")))


# ─── 19. Anchor loading ──────────────────────────────────────────────────────

def _setup_trainer(policy, path, coef=0.1):
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.config = GRPOConfig(device="cpu", vel_anchor_coef=coef,
                          vel_anchor_path=None if path is None else str(path))
    t.device = torch.device("cpu")
    t.model = policy
    t._resolve_vel_anchor_split = lambda: ((2,), 3)
    with contextlib.redirect_stdout(io.StringIO()):
        t._setup_vel_anchor()
    return t


def test_19_anchor_loading():
    print("\n[19] Checkpoint anchor: validated, frozen, DiT-relative, not in state_dict")
    head = _Head()
    _randomize_lora(head, 30)
    policy = _Policy(head)
    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "iter_0003"
        save_lora_checkpoint(policy, ck)
        t = _setup_trainer(policy, ck)
        a = t._vel_anchor
        want_keys = {k for k in head.model.state_dict() if "lora_" in k}
        check("kind == params with exactly the DiT-relative LoRA keys",
              a.kind == "params" and set(a.params) == want_keys, f"{sorted(a.params)}")
        check("fp32, frozen", all(v.dtype == torch.float32 and not v.requires_grad
                                  for v in a.params.values()))
        ptrs = {v.data_ptr() for v in policy.state_dict().values()}
        check("not registered in model.state_dict() (no shared storage)",
              not any(v.data_ptr() in ptrs for v in a.params.values()))
        check("anchor == start weights detected", t._vel_anchor_equals_start)
        check("split resolved into the trainer", t._vel_anchor_split == ((2,), 3))
        sd = torch.load(ck / "lora_weights.pt")
        for label, mutate in (
            ("missing key", lambda d: d.pop(sorted(d)[0])),
            ("extra key", lambda d: d.__setitem__("proj.lora_A.other.weight",
                                                  torch.zeros(2, 4))),
            ("wrong shape", lambda d: d.__setitem__(
                "proj.lora_A.default.weight", torch.zeros(3, 4))),
        ):
            bad = copy.deepcopy(sd)
            mutate(bad)
            bd = Path(tmp) / f"bad_{label.replace(' ', '_')}"
            bd.mkdir()
            torch.save(bad, bd / "lora_weights.pt")
            raised = ""
            try:
                _setup_trainer(policy, bd)
            except RuntimeError as exc:
                raised = str(exc)
            check(f"{label}: setup fails with a clear error",
                  bool(raised) and ("does not match" in raised
                                    or "shape mismatch" in raised), raised[:120])
        raised = ""
        try:
            t._load_lora_state(Path(tmp) / "nowhere", label="vel_anchor_path")
        except RuntimeError as exc:
            raised = str(exc)
        check("missing file names the vel_anchor_path field", "vel_anchor_path" in raised)


# ─── 20. First-micro-batch assertion ─────────────────────────────────────────

def test_20_first_microbatch_assertion():
    print("\n[20] anchor == start weights: first micro-batch must read D ~ 0")
    ok = _Harness(coef=0.1, equals_start=True, fake_kwargs=dict(first_d=0.0))
    err = None
    try:
        ok.update()
    except RuntimeError as exc:
        err = exc
    check("passes when the first micro-batch reads D == 0", err is None, str(err))
    check("one-shot: the check disarms after it ran",
          ok.trainer._vel_anchor_equals_start is False)
    bad = _Harness(coef=0.1, equals_start=True, fake_kwargs=dict(first_d=1.0))
    raised = ""
    try:
        bad.update()
    except RuntimeError as exc:
        raised = str(exc)
    check("trips on a mis-mapped anchor (D = 1.0)", "first micro-batch" in raised,
          raised[:120])
    quiet = _Harness(coef=0.1, equals_start=False, fake_kwargs=dict(first_d=1.0))
    err = None
    try:
        quiet.update()
    except RuntimeError as exc:
        err = exc
    check("not armed when the anchor differs from the start weights", err is None)


# ─── 21. Config validation ───────────────────────────────────────────────────

def test_21_config_validation():
    print("\n[21] Config validation")
    for bad in (dict(vel_anchor_coef=-0.1), dict(vel_anchor_coef=float("nan")),
                dict(vel_anchor_coef=float("inf")), dict(vel_anchor_coef=True),
                dict(stop_after_iterations=0), dict(stop_after_iterations=-2),
                dict(stop_after_iterations=True), dict(stop_after_iterations=1.5),
                dict(vel_anchor_coef=0.1, vel_anchor_path="/nonexistent/iter_0003")):
        raised = False
        try:
            GRPOConfig(**bad)
        except ValueError:
            raised = True
        check(f"rejected: {bad}", raised)
    for good in (dict(), dict(vel_anchor_coef=0.0), dict(vel_anchor_coef=5.0),
                 dict(stop_after_iterations=1), dict(stop_after_iterations=None)):
        err = None
        try:
            GRPOConfig(**good)
        except ValueError as exc:
            err = exc
        check(f"accepted: {good}", err is None, str(err))
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "iter_0001"
        d.mkdir()
        raised = False
        try:
            GRPOConfig(vel_anchor_coef=0.1, vel_anchor_path=str(d))
        except ValueError:
            raised = True
        check("dir without lora_weights.pt rejected", raised)
        torch.save({}, d / "lora_weights.pt")
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            GRPOConfig(vel_anchor_coef=0.0, vel_anchor_path=str(d))
        check("path with coef 0 warns", any("inert" in str(x.message) for x in wl))
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            GRPOConfig(vel_anchor_coef=0.1, vel_anchor_path=str(d))
        check("path with coef > 0 does not warn", not any("inert" in str(x.message)
                                                          for x in wl))


# ─── 22. Resume behaviour ────────────────────────────────────────────────────

def test_22_resume_behaviour():
    print("\n[22] Resumed weights: base anchor D > 0; anchor == resume ckpt gives D == 0")
    head = _Head()
    _randomize_lora(head, 40)            # "resumed" trained LoRA
    policy = _Policy(head)
    inp = _inputs()
    tb = _setup_trainer(policy, None)
    check("base anchor on trained weights: not equal to start",
          tb._vel_anchor_equals_start is False)
    rb = compute_fm_log_prob(action_head=head, **inp, vel_anchor=tb._vel_anchor,
                             return_struct=True)
    check("base anchor on trained weights: D > 0 at the start",
          bool((rb.anchor_dist > 1e-4).all()), f"{rb.anchor_dist}")
    fresh = _setup_trainer(_Policy(_Head()), None)
    check("base anchor on a fresh (B == 0) model: equal to start",
          fresh._vel_anchor_equals_start is True)
    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "iter_0005"
        save_lora_checkpoint(policy, ck)
        tp = _setup_trainer(policy, ck)
        check("checkpoint anchor == resume weights: equal to start",
              tp._vel_anchor_equals_start is True)
        rp = compute_fm_log_prob(action_head=head, **inp, vel_anchor=tp._vel_anchor,
                                 return_struct=True)
        check("checkpoint anchor == resume weights: D exactly 0",
              torch.equal(rp.anchor_dist, torch.zeros_like(rp.anchor_dist)))


# ─── 23. stop_after_iterations through the real train() loop ─────────────────

def _loop_trainer(tmp, *, num_iterations=48, stop_after=1, start=1, lr=5e-5,
                  n_signal=24, n_anchor=0, config_overrides=None):
    from test_phase_timing_logs import FakeBuffer, STATS
    cfg = GRPOConfig(use_wandb=False, num_iterations=num_iterations, save_interval=1,
                     episode_dir=tmp, checkpoint_dir=str(Path(tmp) / "ckpt"),
                     learning_rate=lr, stop_after_iterations=stop_after,
                     **(config_overrides or {}))
    t = object.__new__(GRPOTrainer)
    t.config = cfg
    t.writer = _FakeWriter()
    t.iteration = start
    buf = FakeBuffer()
    stats = dict(STATS, n_signal_chunks=n_signal, n_anchor_chunks=n_anchor)
    buf.stats = lambda: dict(stats)
    t.buffer = buf
    t._consecutive_collect_failures = 0
    t._max_consecutive_collect_failures = 3
    t._collect_rollout_time = float("nan")
    t._collect_load_time = float("nan")
    t._start_iteration = start
    t._last_updated_iteration = start - 1
    t.optimizer = type("Opt", (), {"param_groups": [{"lr": 0.0}]})()
    calls, saves = [], []
    t._log_mem_snapshot = lambda *a, **k: None
    t._release_memory_to_os = lambda: None
    t._collect_episodes = lambda *a, **k: calls.append("collect")
    t._vram_snapshot = lambda **k: None
    t._log_vram = lambda *a, **k: None
    t._compute_ref_log_probs = lambda: calls.append("ref")
    t._grpo_update = lambda: {"n_updates": 3, "n_micro_batches": 6}
    t._compute_lora_delta_norm = lambda: 1.0

    def _save(it):
        saves.append(it)
        (Path(cfg.checkpoint_dir) / f"iter_{it:04d}").mkdir(parents=True, exist_ok=True)

    t._save_checkpoint = _save
    t._save_checkpoint_for_skipped_iter = lambda it: saves.append(("skipped", it))
    return t, calls, saves


def test_23_stop_after_iterations():
    print("\n[23] stop_after_iterations through the real train() loop")
    for start in (1, 4):
        with tempfile.TemporaryDirectory() as tmp:
            t, calls, saves = _loop_trainer(tmp, start=start)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                t.train()
            want_lr = 5e-5 * (1.0 - (start - 1) / 48)
            check(f"start {start}: exactly one iteration ran", calls == ["collect", "ref"],
                  str(calls))
            check(f"start {start}: LR is the num_iterations=48 schedule's value",
                  math.isclose(t.optimizer.param_groups[0]["lr"], want_lr, rel_tol=1e-12),
                  f"{t.optimizer.param_groups[0]['lr']} vs {want_lr}")
            check(f"start {start}: checkpoint saved once, no duplicate final save",
                  saves == [start], str(saves))
            check(f"start {start}: the stop is announced",
                  "stop_after_iterations=1" in out.getvalue())
    with tempfile.TemporaryDirectory() as tmp:
        t, calls, saves = _loop_trainer(tmp, stop_after=2, num_iterations=48)
        with contextlib.redirect_stdout(io.StringIO()):
            t.train()
        check("stop_after=2 runs exactly two iterations",
              calls == ["collect", "ref", "collect", "ref"] and saves == [1, 2],
              f"{calls} {saves}")
    with tempfile.TemporaryDirectory() as tmp:
        t, calls, saves = _loop_trainer(tmp, stop_after=1, n_signal=0)
        with contextlib.redirect_stdout(io.StringIO()):
            t.train()
        check("a skipped (no-signal) iteration also counts and stops",
              calls == ["collect"] and saves == [("skipped", 1)], f"{calls} {saves}")
    with tempfile.TemporaryDirectory() as tmp:
        t, calls, saves = _loop_trainer(tmp, stop_after=None, num_iterations=2)
        with contextlib.redirect_stdout(io.StringIO()):
            t.train()
        check("stop_after=None runs to num_iterations", saves == [1, 2], str(saves))


# ═════════════════════════════════════════════════════════════════════════════
# Audit regressions: each pins a behaviour a code audit found untested
# ═════════════════════════════════════════════════════════════════════════════

def _raises(fn, exc=Exception):
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return str(e) or type(e).__name__
    return ""


def test_a01_nonfinite_first_microbatch_while_armed():
    print("\n[A1] Armed first-micro-batch check: a NaN D is dropped, not a crash")
    h = _Harness(coef=0.2, equals_start=True,
                 fake_kwargs=dict(d_by_train_call={0: float("nan"), 1: 0.0}))
    err = _raises(h.update)
    check("NaN first micro-batch does not raise", err == "", err)
    h2 = _Harness(coef=0.2, equals_start=True,
                  fake_kwargs=dict(d_by_train_call={0: float("nan"), 1: 0.0}))
    r = h2.update()
    check("... it is dropped by the non-finite guard", r["n_skipped_nonfinite"] == 1)
    check("... and the check stayed armed, then passed on the next finite one",
          h2.trainer._vel_anchor_equals_start is False)
    h3 = _Harness(coef=0.2, equals_start=True,
                  fake_kwargs=dict(d_by_train_call={0: float("nan"), 1: 1.0}))
    check("a large finite D on the first FINITE micro-batch still trips it",
          "first micro-batch" in _raises(h3.update, RuntimeError))


def test_a02_tolerance_edges():
    print("\n[A2] Start tolerance is 1e-5, not looser")
    for d, trips in ((1e-3, True), (5e-6, False)):
        h = _Harness(coef=0.2, equals_start=True,
                     fake_kwargs=dict(d_by_train_call={0: d}))
        msg = _raises(h.update, RuntimeError)
        check(f"first micro-batch D = {d:g}: {'raises' if trips else 'passes'}",
              bool(msg) == trips, msg)


def test_a03_anchor_only_iteration_rescued():
    print("\n[A3] All-success iteration is trained when only the velocity anchor acts")
    base = dict(include_anchor_groups=True, anchor_advantage=0.0,
                kl_coef_base_model=0.0)
    for coef, want in ((0.1, ["collect", "ref"]), (0.0, ["collect"])):
        with tempfile.TemporaryDirectory() as tmp:
            t, calls, _saves = _loop_trainer(
                tmp, n_signal=0, n_anchor=5,
                config_overrides=dict(base, vel_anchor_coef=coef))
            with contextlib.redirect_stdout(io.StringIO()):
                t.train()
            check(f"vel_anchor_coef={coef}: {'trained' if coef else 'skipped'}",
                  calls == want, str(calls))


def test_a04_anchor_with_grad_probe_and_smoothness():
    print("\n[A4] Anchored forward keeps the grad probe and the smooth path intact")
    cfg = dict(jitter_pos=0.125, jitter_paired=False, kl_coef_last_iter=0.0)
    r0 = _Harness(coef=0.0, grad_probe_every=1, config_overrides=cfg).update()
    r1 = _Harness(coef=0.3, grad_probe_every=1, config_overrides=cfg).update()
    gp0, gp1 = r0.get("_grad_probe") or {}, r1.get("_grad_probe") or {}
    check("grad probe ran with the anchor on", gp1.get("n_probes", 0) > 0, str(gp1))
    check("gradprobe/* identical to coef 0 (value-pinned stand-in)",
          gp0 == gp1, f"{gp0} vs {gp1}")
    saved = {a: getattr(GRPOTrainer, a) for a in (
        "smooth_active", "_smooth_dims", "_smooth_horizon", "_smooth_eef_pos_dims",
        "_smooth_hf_ref", "_smooth_calib_sum", "_smooth_calib_n",
        "_smooth_calib_rows", "_smooth_n_exec")}
    try:
        GRPOTrainer.smooth_active = True
        GRPOTrainer._smooth_dims = torch.tensor([0])
        GRPOTrainer._smooth_horizon = 3
        GRPOTrainer._smooth_eef_pos_dims = torch.tensor([0])
        GRPOTrainer._smooth_hf_ref = torch.tensor(0.0)
        GRPOTrainer._smooth_calib_sum = None
        GRPOTrainer._smooth_calib_n = 0
        GRPOTrainer._smooth_calib_rows = 0
        GRPOTrainer._smooth_n_exec = 2
        s0 = _Harness(coef=0.0, config_overrides=dict(smooth_coef=0.2)).update()
        s1 = _Harness(coef=0.3, config_overrides=dict(smooth_coef=0.2)).update()
    finally:
        for a, v in saved.items():
            setattr(GRPOTrainer, a, v)
    sm0 = {k: v for k, v in s0.items() if k.startswith("smooth_")}
    sm1 = {k: v for k, v in s1.items() if k.startswith("smooth_")}
    check("smooth path active in both runs", bool(sm0) and bool(sm1))
    check("smooth/* identical to coef 0 (constrained vs endpoint not swapped)",
          sm0 == sm1, f"{sm0.get('smooth_hf_mean')} / {sm1.get('smooth_hf_mean')}")


def test_a05_ref_pass_all_zero_d():
    print("\n[A5] Ref pass with D == 0 everywhere (fresh base anchor, iteration 1)")
    h = _Harness(coef=0.2, split=((0,), 1), fake_kwargs=dict(d_zero=True))
    err = _raises(h.ref_pass)
    st = h.trainer._vel_anchor_start_stats or {}
    check("no exception (no 0/0)", err == "", err)
    check("start_mean / p90 / pos / neg read 0",
          st.get("start_mean") == 0.0 and st.get("start_p90") == 0.0
          and st.get("start_pos") == 0.0 and st.get("start_neg") == 0.0, str(st))
    check("the two fractions are absent (undefined at D == 0)",
          "start_gripper_frac" not in st and "start_exec_frac" not in st)


def test_a06_start_stats_exact():
    print("\n[A6] start_* values against a hand computation over signal chunks")
    h = _Harness(coef=0.2, n_anchor_chunks=3, mb_size=5)
    h.ref_pass()
    st = h.trainer._vel_anchor_start_stats
    sig = [c for c in h.chunks if not c.is_anchor]
    d = np.array([0.02 + 0.05 * c.feat ** 2 for c in sig])
    pos = np.array([c.advantage > 0 for c in sig])
    check("start_mean over signal chunks only (anchor rows excluded)",
          math.isclose(st["start_mean"], float(d.mean()), rel_tol=1e-6),
          f"{st['start_mean']} vs {d.mean()}")
    check("start_pos / start_neg split by advantage sign",
          math.isclose(st["start_pos"], float(d[pos].mean()), rel_tol=1e-6)
          and math.isclose(st["start_neg"], float(d[~pos].mean()), rel_tol=1e-6))
    check("start_p90 is the 90th percentile",
          math.isclose(st["start_p90"], float(np.percentile(d, 90)), rel_tol=1e-6))


def _expected_force_balance(h, c, keep):
    ratios, coses = [], []
    for j, (tc, adv) in enumerate(zip(h.train_calls, h.trainer._test_prepared)):
        if not keep(j):
            continue
        f, delta = tga._row_feature(tc["feat"].reshape(-1, 1, 1), 0.05)
        A = (adv - adv.mean()) / (adv.std() + 1e-8)
        g_clip = -((A * torch.exp(delta)).unsqueeze(1) * f).mean(dim=0)
        g_pen = c * tc["g"].mean(dim=0)
        if float(g_pen.norm()) == 0.0:
            continue
        ratios.append(float(g_pen.norm() / g_clip.norm()))
        coses.append(float(torch.dot(g_pen, g_clip) / (g_pen.norm() * g_clip.norm())))
    return float(np.mean(ratios)), float(np.mean(coses))


def test_a07_force_balance_cadence_and_zero_probe():
    print("\n[A7] Force balance: cadence respected; zero-penalty probes skipped")
    c = 0.4
    cfg = dict(kl_coef_last_iter=0.0)
    h = _Harness(coef=c, grad_probe_every=2, config_overrides=cfg)
    va = h.update()["_vel_anchor"]
    want_r, want_c = _expected_force_balance(h, c, lambda j: j % 2 == 0)
    check("grad_probe_every=2: means over every 2nd trained micro-batch only",
          math.isclose(va["grad_ratio"], want_r, rel_tol=1e-5)
          and math.isclose(va["grad_cos"], want_c, rel_tol=1e-5, abs_tol=1e-7),
          f"{va['grad_ratio']} vs {want_r}")
    hz = _Harness(coef=c, grad_probe_every=1, config_overrides=cfg,
                  fake_kwargs=dict(zero_grad_train_calls={0}))
    vz = hz.update()["_vel_anchor"]
    want_r, want_c = _expected_force_balance(hz, c, lambda j: True)
    check("a probe with zero penalty gradient is left out of BOTH means",
          math.isclose(vz["grad_ratio"], want_r, rel_tol=1e-5)
          and math.isclose(vz["grad_cos"], want_c, rel_tol=1e-5, abs_tol=1e-7),
          f"{vz['grad_ratio']} vs {want_r}")


class _FakeProcessor:
    """get_modality_configs / norm_params, shaped like the GR00T processor."""

    def __init__(self, keys, dims, horizon, tag="robocasa_panda_omron"):
        acfg = types.SimpleNamespace(modality_keys=list(keys),
                                     delta_indices=list(range(horizon)))
        self._cfg = {tag: {"action": acfg}}
        self.state_action_processor = types.SimpleNamespace(norm_params={
            tag: {"action": {k: {"dim": torch.tensor(d)} for k, d in zip(keys, dims)}}})

    def get_modality_configs(self):
        return self._cfg


def test_a08_resolve_split_real_method():
    print("\n[A8] _resolve_vel_anchor_split on a PandaOmron-shaped layout")
    import grpo_server
    keys = ["end_effector_position", "end_effector_rotation", "gripper_close",
            "base_motion", "control_mode"]
    dims = [3, 3, 1, 4, 1]
    real_mask_fn = grpo_server.compute_action_mask

    def run(proc, mask, n_action_steps=8):
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.config = GRPOConfig(device="cpu", n_action_steps=n_action_steps)
        t.processor = proc
        grpo_server.compute_action_mask = lambda probe: mask
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return t._resolve_vel_anchor_split()
        finally:
            grpo_server.compute_action_mask = real_mask_fn

    full = np.zeros((50, 128), dtype=np.float32)
    full[:16, :12] = 1.0
    check("gripper column 6, executed steps 8",
          run(_FakeProcessor(keys, dims, 16), full) == ((6,), 8))
    check("n_exec capped by the horizon",
          run(_FakeProcessor(keys, dims, 16), full, n_action_steps=32) == ((6,), 16))
    short = full.copy()
    short[:, 6] = 0.0
    check("gripper column outside the action mask -> gripper share dropped",
          run(_FakeProcessor(keys, dims, 16), short) == (None, 8))
    no_grip = [k for k in keys if k != "gripper_close"]
    no_dims = [d for k, d in zip(keys, dims) if k != "gripper_close"]
    check("no gripper_close key -> gripper share dropped",
          run(_FakeProcessor(no_grip, no_dims, 16), full) == (None, 8))

    class _Broken:
        def get_modality_configs(self):
            raise RuntimeError("no processor")
    check("an unresolvable layout degrades to (None, None), never raises",
          run(_Broken(), full) == (None, None))


def test_a09_checkpoint_anchor_not_equal_to_start():
    print("\n[A9] A checkpoint anchor that differs from the live weights is not 'start'")
    head = _Head()
    _randomize_lora(head, 50)
    policy = _Policy(head)
    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "iter_0002"
        save_lora_checkpoint(policy, ck)
        _randomize_lora(head, 51)                 # live weights move on
        t = _setup_trainer(policy, ck)
    check("equals_start is False (the first-micro-batch check stays disarmed)",
          t._vel_anchor_equals_start is False)


def test_a10_jac_part_excludes_fixed_rows():
    print("\n[A10] jac_part_pos uses jitter-positive rows only (paired-mode fixed rows out)")
    head = _Head()
    _randomize_lora(head, 60)
    B = 4
    inp = _inputs(B=B, K=3)
    lam = torch.tensor([0.125, 0.125, 0.0, 0.0])
    nfi = _jittered(inp, lam)                     # rows 2, 3: eps' == eps
    pos = torch.tensor([True, True, True, False])
    fixed = torch.tensor([False, False, True, False])
    t = _diag_trainer(head)
    out = t._jitter_gap_diagnostics(
        ready_backbone=inp["backbone_output"], ready_state_features=inp["state_features"],
        ready_embodiment_id=inp["embodiment_id"], ready_actions=inp["actions"],
        ready_masks=inp["action_mask"], ready_noise=inp["noise"],
        timesteps=inp["timesteps"], noise_for_input=nfi, lam_row=lam,
        pos_adv_mask=pos, fixed_row_mask=fixed, jitter_row_mask=~fixed,
        vel_anchor=VelAnchor("base"))
    fm = dict(action_head=head, **inp)
    dc = compute_fm_log_prob(**fm, vel_anchor=VelAnchor("base"), return_struct=True).anchor_dist
    dj = compute_fm_log_prob(**fm, noise_for_input=nfi, vel_anchor=VelAnchor("base"),
                             return_struct=True).anchor_dist
    want = float((dj - dc)[:2].mean())
    diluted = float((dj - dc)[:3].mean())
    got = out.get("_vel_anchor_jac_part_pos", float("nan"))
    check("equals the mean over jitter-positive rows (0, 1)",
          math.isclose(got, want, rel_tol=1e-5), f"{got} vs {want}")
    check("... not diluted by the fixed positive row", not math.isclose(got, diluted,
                                                                        rel_tol=1e-3))


def test_a11_source_text_once_and_ungated_emission():
    print("\n[A11] source text written once; train-side metrics emitted at n_updates 0")
    h = _Harness(coef=0.2)
    r = h.update()
    h.trainer.writer = _FakeWriter()
    with contextlib.redirect_stdout(io.StringIO()):
        h.trainer._log_metrics(1, {"success_rate": 0.5}, r, lr=1e-5, iter_time=1.0)
        h.trainer._log_metrics(2, {"success_rate": 0.5}, r, lr=1e-5, iter_time=1.0)
    check("vel_anchor/source written exactly once across two iterations",
          h.trainer.writer.text_calls.count("vel_anchor/source") == 1,
          str(h.trainer.writer.text_calls))
    hb = _Harness(coef=0.3, fake_kwargs=dict(grad_blowup=True))
    rb = hb.update()
    check("every window dropped (non-finite gradient): n_updates == 0",
          rb.get("n_updates", 0) == 0 and rb.get("n_nonfinite_grad_steps", 0) > 0,
          str({k: rb.get(k) for k in ("n_updates", "n_nonfinite_grad_steps")}))
    check("... the early-return stats still carry _vel_anchor",
          "train_mean" in (rb.get("_vel_anchor") or {}))
    check("... weights unchanged", torch.equal(hb.model.w, hb.w0))
    w = _log(hb.trainer, rb)
    check("... and vel_anchor/train_mean is still emitted to TB",
          "vel_anchor/train_mean" in w.scalars)


def test_a12_start_stats_reset_each_iteration():
    print("\n[A12] train() clears last iteration's start_* before the ref pass")
    with tempfile.TemporaryDirectory() as tmp:
        t, _calls, _saves = _loop_trainer(
            tmp, config_overrides=dict(vel_anchor_coef=0.1))
        t._vel_anchor_start_stats = {"start_mean": 123.0}   # stale
        with contextlib.redirect_stdout(io.StringIO()):
            t.train()
        check("stale start_mean not re-emitted",
              "vel_anchor/start_mean" not in t.writer.scalars, str(t.writer.scalars))
        check("vel_anchor/coef emitted every iteration",
              t.writer.scalars.get("vel_anchor/coef") == 0.1)


def test_a13_guards():
    print("\n[A13] Argument and wiring guards")
    h = _Harness(coef=0.1)
    h.trainer._vel_anchor = None
    check("coef > 0 without a built anchor refuses to train",
          "no anchor was built" in _raises(h.update, RuntimeError))
    check("VelAnchor rejects an unknown kind",
          bool(_raises(lambda: VelAnchor("bogus"), ValueError)))
    check("VelAnchor('params') needs params",
          bool(_raises(lambda: VelAnchor("params"), ValueError)))
    check("VelAnchor('base') rejects params",
          bool(_raises(lambda: VelAnchor("base", params={}), ValueError)))
    head = _Head()
    inp = _inputs()
    check("vel_anchor_split without vel_anchor rejected",
          bool(_raises(lambda: compute_fm_log_prob(action_head=head, **inp,
                                                   vel_anchor_split=(0, 1),
                                                   return_struct=True), ValueError)))
    check("vel_anchor_per_tau without vel_anchor rejected",
          bool(_raises(lambda: compute_fm_log_prob(action_head=head, **inp,
                                                   vel_anchor_per_tau=True,
                                                   return_struct=True), ValueError)))
    check("vel_anchor must be a VelAnchor",
          bool(_raises(lambda: compute_fm_log_prob(action_head=head, **inp,
                                                   vel_anchor="base",
                                                   return_struct=True), TypeError)))


TESTS = [
    test_01_off_path_contract,
    test_02_zero_at_anchor,
    test_03_hand_computation,
    test_04_same_inputs,
    test_05_gradient,
    test_06_lora_restored,
    test_07_params_anchor_leaves_model_untouched,
    test_08_masking,
    test_09_split_fractions,
    test_10_no_rng,
    test_11_struct_every_combination,
    test_12_loss_composition,
    test_13_off_switch,
    test_14_grad_accumulation,
    test_15_nonfinite_guard,
    test_16_force_balance,
    test_17_metrics,
    test_18_jac_part,
    test_19_anchor_loading,
    test_20_first_microbatch_assertion,
    test_21_config_validation,
    test_22_resume_behaviour,
    test_23_stop_after_iterations,
    test_a01_nonfinite_first_microbatch_while_armed,
    test_a02_tolerance_edges,
    test_a03_anchor_only_iteration_rescued,
    test_a04_anchor_with_grad_probe_and_smoothness,
    test_a05_ref_pass_all_zero_d,
    test_a06_start_stats_exact,
    test_a07_force_balance_cadence_and_zero_probe,
    test_a08_resolve_split_real_method,
    test_a09_checkpoint_anchor_not_equal_to_start,
    test_a10_jac_part_excludes_fixed_rows,
    test_a11_source_text_once_and_ungated_emission,
    test_a12_start_stats_reset_each_iteration,
    test_a13_guards,
]


if __name__ == "__main__":
    for fn in TESTS:
        fn()
    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for name in _failures:
            print(f"  - {name}")
        sys.exit(1)
    print("\033[32mAll velocity-anchor tests passed.\033[0m")
