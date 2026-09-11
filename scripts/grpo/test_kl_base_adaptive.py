"""CPU tests for the closed-loop base-model trust region (kl_base_adaptive).

Runs without a GPU, a model, or the collector. Two harnesses:

  * a bare `GRPOTrainer.__new__` shell with a stubbed config and `_ref_mse_stats`,
    which is enough to drive the real `_update_kl_base_coef` and
    `_kl_base_coef_now`;
  * `GRPOConfig` itself for the validation matrix.

The off-switch claim is the important one: with `kl_base_adaptive=False` every
consumer must read `config.kl_coef_base_model` exactly as before, so an
adaptive-off run is numerically unchanged.
"""

import math
import sys
import torch
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_grpo  # noqa: E402
from grpo_config import GRPOConfig  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  \033[32mPASS\033[0m  {name}")
    else:
        _FAIL += 1
        print(f"  \033[31mFAIL\033[0m  {name}" + (f"  ({detail})" if detail else ""))


def _trainer(**cfg_kw):
    """A `__new__` shell carrying only what the controller touches."""
    t = train_grpo.GRPOTrainer.__new__(train_grpo.GRPOTrainer)
    t.config = GRPOConfig(**cfg_kw)
    t._ref_mse_stats = None
    t._kl_base_coef = None
    # Seed the streak exactly as __init__ does. Without it, deleting a streak-reset
    # in the controller raises AttributeError instead of misbehaving, so two
    # mutations died on a harness artifact rather than on an assertion — a FAKE kill
    # that made the streak look covered when it was not.
    t._kl_base_below_streak = 0
    return t


def _drive(t, lbrs):
    """Feed a drift sequence, return the coefficient after each iteration."""
    out = []
    for x in lbrs:
        t._ref_mse_stats = None if x is None else {"log_base_ratio_mean": x}
        d = t._update_kl_base_coef()
        out.append((t._kl_base_coef_now(), d))
    return out


# ── 1. OFF switch ───────────────────────────────────────────────────────────
def test_off_switch():
    print("\n[off] kl_base_adaptive=False changes nothing")
    t = _trainer(kl_base_adaptive=False, kl_coef_base_model=0.1)
    check("_kl_base_coef_now returns the config value",
          t._kl_base_coef_now() == 0.1, str(t._kl_base_coef_now()))
    t._ref_mse_stats = {"log_base_ratio_mean": 99.0}   # would slam the controller
    d = t._update_kl_base_coef()
    check("controller is a no-op and emits nothing", d == {}, str(d))
    check("... and the coefficient is untouched",
          t._kl_base_coef_now() == 0.1 and t._kl_base_coef is None)
    check("no train/kl_base_* keys can be emitted (empty dict is falsy)", not d)


# ── 2. deadband semantics ───────────────────────────────────────────────────
def test_deadband():
    print("\n[band] tighten above hi*target, relax below lo*target, hold between")
    kw = dict(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.04,
              kl_base_adapt_rate=2.0, kl_base_deadband_hi=1.5, kl_base_deadband_lo=0.5)
    # hi = 0.06, lo = 0.02
    t = _trainer(**kw)
    (c1, d1), = _drive(t, [0.10])
    check("drift 0.10 > hi 0.06 -> x2", c1 == 2.0 and d1["kl_base_action"] == 1.0, str(c1))
    t = _trainer(**kw)
    seq = _drive(t, [0.01] * 3)
    check("one quiet iteration does NOT relax (patience)",
          seq[0][0] == 1.0 and seq[0][1]["kl_base_action"] == 0.0, str(seq[0][0]))
    check("quiet iterations at the START value cannot relax below it (relax floor)",
          [c for c, _ in seq] == [1.0, 1.0, 1.0], str([c for c, _ in seq]))
    # To see the relax branch actually move, there must be prior tightening to undo.
    t = _trainer(**kw)
    _drive(t, [0.10, 0.10])                       # -> 4.0
    seq = _drive(t, [0.01] * 3)
    check("after tightening, relax fires only on the `patience`-th quiet iteration",
          [c for c, _ in seq][:2] == [4.0, 4.0]
          and abs(seq[2][0] - 4.0 / 1.1) < 1e-12
          and seq[2][1]["kl_base_action"] == -1.0,
          str([c for c, _ in seq]))
    t = _trainer(**kw)
    seq = _drive(t, [0.01, 0.01, 0.04, 0.01, 0.01])
    # Assert the STREAK SERIES, not just the coefficient: the relax floor pins the
    # coefficient at 1.0 either way, so a coefficient-only check cannot see a broken
    # reset. Feeding below,below,in,below,below the streak must read 1,2,0,1,2.
    streaks = [d["kl_base_below_streak"] for _, d in seq]
    check("an in-band iteration RESETS the streak (series 1,2,0,1,2)",
          streaks == [1.0, 2.0, 0.0, 1.0, 2.0], str(streaks))
    # and a tighten resets it too
    t2 = _trainer(**kw)
    st2 = [d["kl_base_below_streak"] for _, d in _drive(t2, [0.01, 0.01, 0.10, 0.01])]
    check("a tighten RESETS the streak", st2 == [1.0, 2.0, 0.0, 1.0], str(st2))
    for x in (0.02, 0.04, 0.06):
        t = _trainer(**kw)
        (c, d), = _drive(t, [x])
        check(f"drift {x} inside [lo, hi] -> hold",
              c == 1.0 and d["kl_base_action"] == 0.0, str(c))
    # boundaries are exclusive on both sides: strictly greater / strictly less
    t = _trainer(**kw)
    (c, _), = _drive(t, [0.06 + 1e-12])
    check("just above hi engages", c == 2.0, str(c))


# ── 3. clamps ───────────────────────────────────────────────────────────────
def test_clamps():
    print("\n[clamp] coefficient stays inside [min, max] and the pin is reported")
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.04,
                 kl_base_adapt_rate=2.0, kl_base_coef_min=0.1, kl_base_coef_max=8.0)
    seq = _drive(t, [0.5] * 6)          # far above the band, climb until pinned
    coefs = [c for c, _ in seq]
    check("climbs 1->2->4->8 then pins at max",
          coefs == [2.0, 4.0, 8.0, 8.0, 8.0, 8.0], str(coefs))
    check("at_max flag is set once pinned", seq[-1][1]["kl_base_coef_at_max"] == 1.0)
    check("at_max flag is NOT set before pinning", seq[0][1]["kl_base_coef_at_max"] == 0.0)
    seq = _drive(t, [0.0] * 150)        # far below the band, relax until it stops
    coefs = [c for c, _ in seq]
    check("descent stops at the STARTING coefficient (1.0), not at kl_base_coef_min",
          coefs[-1] == 1.0 and min(coefs) == 1.0, str(coefs[-1]))
    check("...and slowly: >= 20 iterations of pure quiet to walk 8.0 -> 1.0",
          sum(1 for c, _ in seq if c > 1.0) >= 20, str(sum(1 for c, _ in seq if c > 1.0)))
    # kl_base_coef_min is now reachable ONLY when the start IS the min, since relax
    # is floored at the start. That configuration is covered by test_relax_floor.
    t_at_min = _trainer(kl_base_adaptive=True, kl_coef_base_model=0.1,
                        kl_base_target=0.04, kl_base_adapt_rate=2.0,
                        kl_base_coef_min=0.1, kl_base_coef_max=8.0)
    seq2 = _drive(t_at_min, [0.0] * 10)
    check("at_min flag is set when the start sits on the floor",
          seq2[-1][1]["kl_base_coef_at_min"] == 1.0 and seq2[-1][0] == 0.1,
          str(seq2[-1][0]))


# ── 4. missing reading HOLDS, never relaxes ─────────────────────────────────
def test_missing_reading_holds():
    print("\n[hold] a missing or non-finite drift reading holds, it does not relax")
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.04,
                 kl_base_adapt_rate=2.0)
    _drive(t, [0.5, 0.5])                       # climb to 4.0
    check("climbed to 4.0 first", t._kl_base_coef == 4.0, str(t._kl_base_coef))
    for label, stats in (("no _ref_mse_stats", None),
                         ("stats without the key", {}),
                         ("NaN reading", {"log_base_ratio_mean": float("nan")})):
        t._ref_mse_stats = stats
        d = t._update_kl_base_coef()
        check(f"{label} -> hold at 4.0", t._kl_base_coef == 4.0, str(t._kl_base_coef))
        check(f"{label} -> flagged, not silent", d.get("kl_base_coef_held_no_reading") == 1.0)
    # the failure this guards: relaxing on absent data disarms the controller
    check("absent data never decreases the coefficient", t._kl_base_coef == 4.0)


# ── 4b. relax is floored at the STARTING coefficient ────────────────────────
def test_relax_floor():
    print("\n[floor] relaxation undoes tightening but never goes below the start")
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.04,
                 kl_base_adapt_rate=2.0)
    _drive(t, [0.5, 0.5])                        # climb to 4.0
    check("climbed to 4.0", t._kl_base_coef == 4.0, str(t._kl_base_coef))
    coefs = [c for c, _ in _drive(t, [0.0] * 60)]
    check("relaxes back down...", min(coefs) < 4.0)
    check("...but stops exactly at the starting coefficient, never below",
          min(coefs) == 1.0, str(min(coefs)))
    check("and stays there indefinitely", coefs[-1] == 1.0, str(coefs[-1]))
    # a run that starts at the floor can only ever climb
    t2 = _trainer(kl_base_adaptive=True, kl_coef_base_model=0.1, kl_base_target=0.04,
                  kl_base_adapt_rate=2.0, kl_base_coef_min=0.1)
    c2 = [c for c, _ in _drive(t2, [0.0] * 30)]
    check("a start-at-floor config never drops below its start",
          min(c2) == 0.1, str(min(c2)))


# ── 4c. mutation-hardening: the five checks a mutation survey found missing ──
def test_action_and_pacing_semantics():
    print("\n[semantics] action reports EFFECT, and relax pacing is exact")
    kw = dict(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.04,
              kl_base_adapt_rate=2.0, kl_base_coef_max=4.0)
    # survivor 1: `action = -1.0 if new < prev else 0.0` — a relax that cannot move
    # (already at the start value) must report 0.0, not -1.0.
    t = _trainer(**kw)
    seq = _drive(t, [0.0] * 3)
    check("a relax that cannot move reports action=0.0, not -1.0",
          seq[2][1]["kl_base_action"] == 0.0 and seq[2][0] == 1.0,
          str((seq[2][0], seq[2][1]["kl_base_action"])))
    # ...and a relax that CAN move reports -1.0
    t = _trainer(**kw)
    _drive(t, [0.5, 0.5])                      # -> 4.0 (== cap)
    seq = _drive(t, [0.0] * 3)
    check("a relax that does move reports action=-1.0",
          seq[2][1]["kl_base_action"] == -1.0 and seq[2][0] < 4.0, str(seq[2][0]))
    # tighten is effect-based too: pinned at the cap must report 0.0
    t = _trainer(**kw)
    seq = _drive(t, [0.5] * 4)                 # 1->2, 2->4 (== cap), then pinned
    acts = [d["kl_base_action"] for _, d in seq]
    coefs = [c for c, _ in seq]
    check("a tighten that moves reports 1.0; once pinned at the cap it reports 0.0",
          acts == [1.0, 1.0, 0.0, 0.0] and coefs == [2.0, 4.0, 4.0, 4.0],
          f"acts={acts} coefs={coefs}")
    # survivor 2: relax pacing is EXACTLY one per `patience`, never compounding.
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.04,
                 kl_base_adapt_rate=2.0, kl_base_coef_max=64.0)
    _drive(t, [0.5] * 6)                       # -> 64.0
    seq = _drive(t, [0.0] * 12)
    n_moves = sum(1 for _, d in seq if d["kl_base_action"] == -1.0)
    check("exactly floor(12/patience)=4 relaxations in 12 quiet iterations",
          n_moves == 4, f"{n_moves} moves")
    check("...i.e. 64/1.1**4, not compounding per-iteration",
          abs(seq[-1][0] - 64.0 / 1.1 ** 4) < 1e-9, str(seq[-1][0]))


def test_defaults_and_warning():
    print("\n[defaults] the shipped defaults are the archive-derived ones")
    c = GRPOConfig()
    # survivor 4: nothing pinned the default target; a silent revert to 0.025 would
    # drive the coefficient to the cap on the corpus's two healthiest runs.
    check("kl_base_target default is 0.055 (viable window [0.0467, 0.0695])",
          c.kl_base_target == 0.055, str(c.kl_base_target))
    check("kl_base_coef_max default is 5.0, not 30 (over-correction must stay bounded)",
          c.kl_base_coef_max == 5.0, str(c.kl_base_coef_max))
    check("kl_base_adapt_rate default is 2.0 — 1.5 needs ~6 engaged iterations to "
          "reach useful authority and the archive says a run dies in two",
          c.kl_base_adapt_rate == 2.0, str(c.kl_base_adapt_rate))
    check("kl_base_relax_rate default is 1.1 and relax_patience is 3",
          c.kl_base_relax_rate == 1.1 and c.kl_base_relax_patience == 3,
          f"{c.kl_base_relax_rate}/{c.kl_base_relax_patience}")
    check("the default band brackets res11's healthy 0.0572-0.0701",
          c.kl_base_deadband_lo * c.kl_base_target <= 0.0572
          and c.kl_base_deadband_hi * c.kl_base_target >= 0.0701,
          f"[{c.kl_base_deadband_lo*c.kl_base_target:.4f}, "
          f"{c.kl_base_deadband_hi*c.kl_base_target:.4f}]")
    check("...and its top is below runB's it12 collapse drift 0.1042",
          c.kl_base_deadband_hi * c.kl_base_target < 0.1042)
    # survivor 3: the start-coefficient warning was never exercised.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        GRPOConfig(kl_base_adaptive=True, kl_coef_base_model=0.2)
        hits = [x for x in w if "STARTING coefficient" in str(x.message)]
    check("a low start coefficient WARNS (the 0.2 field default does)", len(hits) == 1,
          f"{len(w)} warnings, {len(hits)} matching")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        GRPOConfig(kl_base_adaptive=True, kl_coef_base_model=1.0)
        hits = [x for x in w if "STARTING coefficient" in str(x.message)]
    check("the recommended start does NOT warn", not hits, str(len(hits)))


def test_coef_zero_rejected_specifically():
    print("\n[cfg] coef<=0 is rejected by its OWN check, not incidentally")
    # survivor 5: asserting only `ValueError` passed for the wrong reason, because
    # 0.0 also fails the start-inside-clamp check. Pin the message.
    try:
        GRPOConfig(kl_base_adaptive=True, kl_coef_base_model=0.0)
        check("coef 0 rejected by the starves-the-controller check", False, "not rejected")
    except ValueError as e:
        check("coef 0 rejected by the starves-the-controller check",
              "requires kl_coef_base_model > 0" in str(e), str(e)[:80])


# ── 4d. the consumer / checkpoint / plumbing paths the __new__ harness CAN reach ──
def test_consumers_read_the_controlled_value():
    print("\n[consumers] the LOSS reads _kl_base_coef_now(), not the config")
    # Top mutation-survey gap: nothing asserted that _grpo_update actually routes
    # through _kl_base_coef_now(), so the loss could weight with the config value
    # while train/kl_base_coef plotted the controlled one — the exact
    # desynchronisation that method's docstring names. Patch the accessor rather
    # than the field: `run_update` builds its trainer internally, and patching the
    # accessor is also the sharper test (it fails if any consumer bypasses it).
    import test_grad_accum as tga
    kw = dict(k=1, n_groups=2, n_chunks=16, mb_size=4, epochs=1, seed=11)
    real = train_grpo.GRPOTrainer._kl_base_coef_now
    torch.manual_seed(3)
    a = tga.run_update(**kw, config_overrides=dict(kl_coef_base_model=0.37))
    try:
        train_grpo.GRPOTrainer._kl_base_coef_now = lambda self: 3.7
        torch.manual_seed(3)
        b = tga.run_update(**kw, config_overrides=dict(kl_coef_base_model=0.37))
    finally:
        train_grpo.GRPOTrainer._kl_base_coef_now = real
    ra, rb = a.result.get("kl_loss_base_model"), b.result.get("kl_loss_base_model")
    check("kl_loss_base_model scales 10x with _kl_base_coef_now()'s return",
          ra is not None and rb is not None and abs(rb / ra - 10.0) < 1e-3,
          f"{ra!r} -> {rb!r}")
    check("...and the WEIGHTS differ, so it reached the loss and not just a metric",
          not torch.equal(a.w_final, b.w_final))
    # compute_base must stay on the CONFIG value, or the controller could switch off
    # its own input by driving the coefficient toward zero.
    try:
        train_grpo.GRPOTrainer._kl_base_coef_now = lambda self: 3.7
        torch.manual_seed(3)
        c = tga.run_update(**kw, config_overrides=dict(kl_coef_base_model=0.0))
    finally:
        train_grpo.GRPOTrainer._kl_base_coef_now = real
    check("compute_base gates on the CONFIG value: config 0 => no base-KL term at all",
          "kl_loss_base_model" not in c.result, str(c.result.get("kl_loss_base_model")))


def test_checkpoint_roundtrip():
    print("\n[ckpt] optimizer.pt carries the coefficient, and the refresh is atomic")
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp()) / "iter_0010"; d.mkdir(parents=True)
    payload = {"optimizer_state": {"state": {}, "param_groups": []},
               "param_names": [], "kl_base_coef": 1.0}
    torch.save(payload, d / "optimizer.pt")
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0)
    t._kl_base_coef = 4.0
    t._refresh_kl_base_coef_in_checkpoint(d)
    got = torch.load(d / "optimizer.pt", weights_only=True)
    check("refresh patches kl_base_coef in place", got["kl_base_coef"] == 4.0, str(got))
    check("...and preserves the rest of the payload",
          set(got) == set(payload) and got["param_names"] == [])
    check("...and leaves no .tmp fragment", not (d / "optimizer.pt.tmp").exists())
    # off / no-state / legacy / missing must all be no-ops
    t.config = GRPOConfig(kl_base_adaptive=False)
    torch.save(payload, d / "optimizer.pt"); t._refresh_kl_base_coef_in_checkpoint(d)
    check("flag-off is a no-op",
          torch.load(d / "optimizer.pt", weights_only=True)["kl_base_coef"] == 1.0)
    t.config = GRPOConfig(kl_base_adaptive=True, kl_coef_base_model=1.0)
    t._kl_base_coef = None; t._refresh_kl_base_coef_in_checkpoint(d)
    check("no controller state is a no-op",
          torch.load(d / "optimizer.pt", weights_only=True)["kl_base_coef"] == 1.0)
    t._kl_base_coef = 4.0
    torch.save({"state": {}, "param_groups": []}, d / "optimizer.pt")
    t._refresh_kl_base_coef_in_checkpoint(d)
    legacy = torch.load(d / "optimizer.pt", weights_only=True)
    check("a legacy raw-state_dict payload is left untouched",
          "kl_base_coef" not in legacy and "param_groups" in legacy, str(list(legacy)))
    t._refresh_kl_base_coef_in_checkpoint(pathlib.Path("/nonexistent"))
    check("a missing dir does not raise", True)


def test_resume_matrix():
    print("\n[resume] all four documented resume cases, via _restore_kl_base_coef")
    # setup() itself is unreachable from a __new__ shell; the restore block was
    # extracted into this helper precisely so the matrix is testable on CPU.
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0)
    t._restore_kl_base_coef(4.0)
    check("(a) adaptive <- adaptive checkpoint: restored", t._kl_base_coef == 4.0, str(t._kl_base_coef))
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0)
    t._restore_kl_base_coef(None)
    check("(b) adaptive <- pre-feature checkpoint: starts at the config value",
          t._kl_base_coef == 1.0, str(t._kl_base_coef))
    t = _trainer(kl_base_adaptive=False, kl_coef_base_model=0.2)
    t._restore_kl_base_coef(4.0)
    check("(c) NON-adaptive <- adaptive checkpoint: ignored, falls back to config",
          t._kl_base_coef is None and t._kl_base_coef_now() == 0.2, str(t._kl_base_coef))
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_coef_max=3.0)
    t._restore_kl_base_coef(8.0)
    check("(d) a restored value above coef_max is CLAMPED, not used raw",
          t._kl_base_coef == 3.0, str(t._kl_base_coef))
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_coef_min=0.5)
    t._restore_kl_base_coef(0.01)
    check("...and below coef_min likewise", t._kl_base_coef == 0.5, str(t._kl_base_coef))


def test_relax_floor_min_clause():
    print("\n[floor] the min() in the relax floor: it must never RAISE the coefficient")
    # fix 10's nine-line-justified min() was entirely untested. Restored coefficient
    # BELOW a raised config start: a quiet iteration must not jump it up.
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=4.0, kl_base_target=0.04,
                 kl_base_coef_max=8.0)
    t._kl_base_coef = 1.0                      # e.g. resumed, then start was raised
    seq = _drive(t, [0.0] * 3)
    check("a quiet iteration does NOT raise a below-start coefficient",
          all(c == 1.0 for c, _ in seq), str([c for c, _ in seq]))
    check("...and reports action=0, not a 'relaxation'",
          all(d["kl_base_action"] == 0.0 for _, d in seq),
          str([d["kl_base_action"] for _, d in seq]))


def test_deadband_lo_boundary():
    print("\n[band] the lower edge is inclusive-hold")
    kw = dict(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.04,
              kl_base_coef_max=8.0)
    t = _trainer(**kw)                            # lo = 0.02 exactly -> HOLD
    st = [d["kl_base_below_streak"] for _, d in _drive(t, [0.02] * 3)]
    check("drift exactly at lo holds and does NOT accumulate the streak",
          st == [0.0, 0.0, 0.0], str(st))
    t = _trainer(**kw)
    st = [d["kl_base_below_streak"] for _, d in _drive(t, [0.02 - 1e-9] * 3)]
    check("just below lo does accumulate", st == [1.0, 2.0, 0.0], str(st))


def test_save_paths_carry_the_coefficient():
    print("\n[save] every save path either writes or refreshes the coefficient")
    import tempfile, pathlib, inspect
    # (i) _save_checkpoint's payload must contain the key. Build a minimal shell and
    # stub the two model-facing helpers; the payload construction is what matters.
    d = pathlib.Path(tempfile.mkdtemp())
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0)
    t.config.checkpoint_dir = str(d)
    t._kl_base_coef = 4.0
    t._lora_param_names = ["a"]
    t.optimizer = type("O", (), {"state_dict": lambda self: {"state": {}}})()
    t._save_smooth_ref = lambda _dir: None
    t.model = None                       # accessed as an argument before the stub runs
    real_save = train_grpo.save_lora_checkpoint
    try:
        # The stub must still create the dir: _save_checkpoint writes optimizer.pt
        # into it immediately afterwards.
        train_grpo.save_lora_checkpoint = lambda _m, _d: pathlib.Path(_d).mkdir(
            parents=True, exist_ok=True)
        t._save_checkpoint(7)
    finally:
        train_grpo.save_lora_checkpoint = real_save
    pay = torch.load(d / "iter_0007" / "optimizer.pt", weights_only=True)
    check("_save_checkpoint writes kl_base_coef into optimizer.pt",
          pay.get("kl_base_coef") == 4.0, str(pay.get("kl_base_coef", "ABSENT")))

    # (ii) the skipped-iteration early return must still refresh.
    seen = []
    t._refresh_kl_base_coef_in_checkpoint = lambda p: seen.append(p.name)
    t._last_updated_iteration = 7                      # iter_0007/ now exists
    t._save_checkpoint_for_skipped_iter(9)
    check("_save_checkpoint_for_skipped_iter refreshes when the dir exists",
          seen == ["iter_0007"], str(seen))

    # (iii) the END-OF-RUN final save must too. That path lives inside train(), so
    # assert the wiring at source level — the runtime path needs a full setup().
    src = inspect.getsource(train_grpo.GRPOTrainer.train)
    i = src.index("Final save skipped: iter_")
    check("train()'s final-save skip branch calls the refresh",
          "_refresh_kl_base_coef_in_checkpoint(final_dir)" in src[i:i + 900],
          "not found within the final-save branch")


def test_setup_wiring():
    print("\n[wiring] setup() passes the CHECKPOINT value to the restore helper")
    # setup() cannot run without a GPU and a model, so this is a source-level wiring
    # check, not a behavioural one. It exists because the four-case resume matrix is
    # only correct if setup() forwards the value it loaded — passing None instead
    # would silently restart the controller on every resume, and _restore_kl_base_coef
    # alone cannot detect that.
    import inspect
    src = inspect.getsource(train_grpo.GRPOTrainer.setup)
    check("setup() forwards the loaded payload value, not a literal None",
          "_restore_kl_base_coef(_resumed_kl_base_coef)" in src,
          "wiring changed — re-check the resume matrix")
    # setup() has TWO `if self.config.resume_from:` blocks; anchor on the one that
    # loads optimizer.pt. The binding must precede it AND sit at method-body indent,
    # or a fresh run NameErrors — the bug this check exists for.
    bind = src.index("_resumed_kl_base_coef = None")
    load = src.index("opt_path = Path(self.config.resume_from)")
    line = src[src.rindex("\n", 0, bind) + 1:bind]
    check("...and _resumed_kl_base_coef is bound before the optimizer.pt load",
          bind < load, f"bind@{bind} load@{load}")
    check("...at method-body indent, so it dominates every path (fresh run included)",
          len(line) - len(line.lstrip()) == 8, f"indent {len(line) - len(line.lstrip())}")


# ── 5. authority arithmetic ─────────────────────────────────────────────────
def test_authority():
    print("\n[authority] reported fraction matches coef*|e^-x - 1| / 0.9")
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.04,
                 kl_base_adapt_rate=2.0)
    (c, d), = _drive(t, [0.0181])              # runB's ignition drift
    check("a single quiet iteration holds the coefficient at its start", c == 1.0, str(c))
    want = c * abs(math.exp(-0.0181) - 1.0) / 0.9
    check("authority at coef 1.0, x=0.0181 is ~2%",
          abs(d["kl_base_authority_frac"] - want) < 1e-12 and 0.015 < want < 0.025,
          f"{d['kl_base_authority_frac']:.5f} vs {want:.5f}")


# ── 6. the runB trajectory ──────────────────────────────────────────────────
def test_runb_trajectory():
    print("\n[runB] replayed against the FULL measured drift series")
    # runB's ref_mse/log_base_ratio_mean, iterations 1-14. The FULL series, not
    # it9-14: truncating to it9-14 starts the below-band streak at zero and hid a
    # real self-disarm bug — the controller walked 1.0 -> 0.751 across the eleven
    # quiet iterations before the emergency, because a fresh run reads exactly 0
    # at it1 (PEFT zero-inits lora_B) and so begins accumulating the streak on
    # iteration one. Never truncate this fixture.
    lbr = [0.0, 0.00051, 0.00137, 0.00239, 0.00406, 0.00609, 0.00841, 0.01113,
           0.01432, 0.01808, 0.02953, 0.10421, 0.24568, 0.30229]
    t = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.04,
                 kl_base_adapt_rate=2.0, kl_base_coef_max=30.0)
    seq = _drive(t, lbr)
    coefs = [round(c, 4) for c, _ in seq]
    print(f"        drift : {lbr}")
    print(f"        coef  : {coefs}")
    check("NEVER relaxes below the starting coefficient across 11 quiet iterations",
          coefs[:11] == [1.0] * 11, str(coefs[:11]))
    check("engages the moment drift clears the band (it12)", coefs[11] == 2.0, str(coefs[11]))
    check("keeps climbing while drift stays high",
          coefs[12] == 4.0 and coefs[13] == 8.0, str(coefs[12:]))
    auth = seq[-1][1]["kl_base_authority_frac"]
    check("authority reaches >100% of the surrogate by it14",
          auth > 1.0, f"{auth:.3f}")
    # Target SENSITIVITY. runB's series jumps 0.02953 (it11) -> 0.10421 (it12) with
    # nothing between, so engagement lands at it12 for every target in roughly
    # [0.0197, 0.0695] — the data has no resolution inside that gap. What must hold
    # for ANY sane target is the pair of properties the controller exists for:
    # silent through the healthy window, engaged before the collapse completes.
    for tgt in (0.02, 0.025, 0.03, 0.04, 0.06):
        tt = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0,
                      kl_base_target=tgt, kl_base_adapt_rate=2.0)
        cc = [c for c, _ in _drive(tt, lbr)]
        # "Silent" means NEVER TIGHTENS during the healthy window — not that the
        # coefficient is untouched. At a loose target (0.06, band [0.03, 0.09])
        # runB's healthy drift sits BELOW the band, so the controller correctly
        # relaxes once. That is itself the argument against a loose target on this
        # task: the healthy operating point should sit INSIDE the band, or the
        # controller spends the quiet phase slowly disarming.
        check(f"target {tgt}: holds at the start value through all 11 quiet iterations",
              cc[:11] == [1.0] * 11, str(cc[:11]))
        check(f"target {tgt}: engaged by it12", cc[11] > 1.0, str(cc[11]))
    # Engaging at it11 instead needs hi < 0.02953, i.e. target < 0.0197 at the
    # default 1.5 deadband. Recorded so nobody re-derives it by fitting to one point.
    tt = _trainer(kl_base_adaptive=True, kl_coef_base_model=1.0, kl_base_target=0.019,
                  kl_base_adapt_rate=2.0)
    cc = [c for c, _ in _drive(tt, lbr)]
    check("target 0.019 (hi=0.0285) engages one iteration sooner, at it11",
          cc[10] == 2.0, str(cc))


# ── 7. validation matrix ────────────────────────────────────────────────────
def test_validation():
    print("\n[cfg] validation rejects every silent-failure config")
    ok = dict(kl_base_adaptive=True, kl_coef_base_model=1.0)
    GRPOConfig(**ok)
    check("the recommended config constructs", True)
    bad = [
        (dict(kl_base_adaptive=True, kl_coef_base_model=0.0), "coef 0 starves the controller"),
        (dict(**ok, kl_base_target=0.0), "target 0"),
        (dict(**ok, kl_base_target=float("nan")), "target NaN"),
        (dict(**ok, kl_base_adapt_rate=1.0), "rate 1.0 (no-op)"),
        (dict(**ok, kl_base_adapt_rate=0.5), "rate < 1 (inverted)"),
        (dict(**ok, kl_base_deadband_hi=0.9), "hi below 1.0"),
        (dict(**ok, kl_base_deadband_lo=1.1), "lo above 1.0"),
        (dict(**ok, kl_base_coef_min=0.0), "zero floor"),
        (dict(**ok, kl_base_coef_min=5.0, kl_base_coef_max=1.0), "min > max"),
        (dict(kl_base_adaptive=True, kl_coef_base_model=100.0), "start outside the clamp"),
        (dict(**ok, kl_base_relax_rate=1.0), "relax rate 1.0"),
        (dict(**ok, kl_base_relax_rate=3.0, kl_base_adapt_rate=2.0), "relax faster than tighten"),
        (dict(**ok, kl_base_relax_patience=0), "zero patience (disarms)"),
        (dict(**ok, kl_base_relax_patience=float("nan")), "patience nan (relax unreachable)"),
        (dict(**ok, kl_base_relax_patience=float("inf")), "patience inf"),
        (dict(**ok, kl_base_relax_patience=2.5), "patience as a float"),
        (dict(**ok, kl_base_relax_patience=True), "patience as a bool"),
        (dict(**ok, kl_base_target=0.5), "target 0.5 (tighten unreachable)"),
        (dict(**ok, kl_base_deadband_lo=0.0), "deadband_lo 0 (relax unreachable)"),
        (dict(kl_base_adaptive=True, kl_coef_base_model=5.0), "start == coef_max (no-op)"),
        (dict(**ok, kl_base_adapt_rate=float("nan")), "NaN adapt rate (NaNs the loss)"),
        (dict(**ok, kl_base_relax_rate=float("nan")), "NaN relax rate"),
        (dict(**ok, kl_base_deadband_hi=float("inf")), "inf deadband (silent disarm)"),
        (dict(**ok, kl_base_coef_max=float("inf")), "inf ceiling"),
    ]
    for kw, why in bad:
        try:
            GRPOConfig(**kw)
            check(f"rejects {why}", False, "NOT rejected")
        except ValueError:
            check(f"rejects {why}", True)
    # defect-5 regression: the adapt-rate check must not be masked by the
    # relax-rate check, or the error blames a knob the operator never set.
    try:
        GRPOConfig(**ok, kl_base_adapt_rate=0.5)
        check("adapt_rate<1 error names adapt_rate", False, "not rejected")
    except ValueError as e:
        check("adapt_rate<1 error names adapt_rate, not relax_rate",
              "kl_base_adapt_rate must be > 1.0" in str(e), str(e)[:90])

    # off means unvalidated-but-harmless: the knobs are inert
    GRPOConfig(kl_base_adaptive=False, kl_base_target=-1.0, kl_base_adapt_rate=0.1)
    check("knobs are not validated when the feature is off", True)


if __name__ == "__main__":
    test_off_switch()
    test_deadband()
    test_clamps()
    test_missing_reading_holds()
    test_relax_floor()
    test_action_and_pacing_semantics()
    test_defaults_and_warning()
    test_coef_zero_rejected_specifically()
    test_consumers_read_the_controlled_value()
    test_checkpoint_roundtrip()
    test_resume_matrix()
    test_relax_floor_min_clause()
    test_deadband_lo_boundary()
    test_save_paths_carry_the_coefficient()
    test_setup_wiring()
    test_authority()
    test_runb_trajectory()
    test_validation()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
