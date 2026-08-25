"""Tests for anchor groups (all-success groups) — buffer classification + update path.

Two halves:

  1. `EpisodeBuffer.compute_advantages` classification: signal / anchor / dead,
     the row budget, and bit-identity with the feature off. (The buffer's own
     `__main__` self-test covers the same ground from the data side; these
     assertions are here so one command covers the whole feature.)

  2. The REAL `GRPOTrainer._grpo_update_inner` driven on CPU with anchor rows
     present, reusing test_grad_accum.py's harness (tiny analytic log-prob stub,
     2-parameter model, stubbed `_prepare_batch`). This is where the risk lives:
     anchor rows must be excluded from the per-minibatch renorm statistics, the
     sampler pools, PAWS' alive-mass split, and every sign-keyed metric — and
     must NEVER be z-scored, or episode-length variation gets amplified to ±1
     and reproduces the time-scaling gradient that collapsed v2.

Run with the project venv (needs torch; CPU is fine):
    .venv/bin/python scripts/grpo/test_anchor_groups.py
"""

import contextlib
import math
import io
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from episode_buffer import EpisodeBuffer, GRPOEpisode  # noqa: E402
from grpo_config import GRPOConfig  # noqa: E402
import test_grad_accum as h  # noqa: E402  (harness: _Chunk, run_update)

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
_failures = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  {GREEN}PASS{RESET}  {label}")
    else:
        print(f"  {RED}FAIL{RESET}  {label}" + (f" — {detail}" if detail else ""))
        _failures.append(label)


def close(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ===========================================================================
# 1. Buffer classification
# ===========================================================================

def _buffer(outcomes: list[list[bool]], chunks_per_ep: int = 2) -> EpisodeBuffer:
    b = EpisodeBuffer()
    for gid, group in enumerate(outcomes):
        for succ in group:
            b.episodes.append(GRPOEpisode(
                video_frames=[{}] * chunks_per_ep, states=[{}] * chunks_per_ep,
                language="t", actions=[np.zeros((16, 12))] * chunks_per_ep,
                raw_actions=[np.zeros((50, 128))] * chunks_per_ep,
                action_masks=[np.ones((50, 128))] * chunks_per_ep,
                initial_noises=[np.zeros((50, 128))] * chunks_per_ep,
                success=succ, shaped_reward=0.0, env_name="t",
                episode_idx=len(b.episodes), num_steps=100,
                group_id=gid, env_seed=gid,
            ))
    return b


def test_classification():
    print("\n[buffer] signal / anchor / dead classification")
    # g0 all-success, g1 all-fail, g2 mixed 2/4, g3 singleton success.
    outcomes = [[True] * 4, [False] * 4, [True, True, False, False], [True]]

    off = _buffer(outcomes)
    adv_off = off.compute_advantages().copy()
    s_off = off.stats()
    check("off: all-success group is dead", bool(np.all(adv_off[:4] == 0.0)))
    check("off: n_anchor_groups == 0", s_off["n_anchor_groups"] == 0)
    check("off: all-success + all-fail + singleton = 3 dead", s_off["n_dead_groups"] == 3)

    on = _buffer(outcomes)
    adv_on = on.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True)
    s_on = on.stats()
    check("on: anchor group gets +0.2", bool(np.allclose(adv_on[:4], 0.2)))
    check("on: all-fail group STAYS dead", bool(np.all(adv_on[4:8] == 0.0)))
    # The assertion above is masked by the row budget in this fixture: group 0
    # consumes the whole default budget, so the all-fail episodes would read 0.0
    # even if they HAD been anchored and then dropped. Re-check with the budget
    # wide open, where an anchored all-fail group would show a positive value.
    unmasked = _buffer(outcomes)
    adv_um = unmasked.compute_advantages(
        anchor_advantage=0.2, include_anchor_groups=True, anchor_max_row_frac=100.0,
    )
    check("on: all-fail STAYS dead with the row budget wide open",
          bool(np.all(adv_um[4:8] == 0.0)),
          f"got {adv_um[4:8]} — an all-fail group must never be anchored")
    check("on: all-fail episodes are not flagged as anchors",
          not any(ep.is_anchor for ep in unmasked.episodes[4:8]))
    # Same masking applies to the singleton: it is last in index order, so the
    # default budget drops it for the wrong reason. Unmasked, an anchored
    # singleton would show a positive advantage.
    check("on: singleton STAYS dead with the row budget wide open",
          adv_um[12] == 0.0,
          f"got {adv_um[12]} — a 1-episode group must never be anchored")
    check("on: singleton is not flagged as an anchor",
          not unmasked.episodes[12].is_anchor)
    check("on: only the all-success group is an anchor group",
          unmasked.stats()["n_anchor_groups"] == 1)
    check("on: singleton STAYS dead", adv_on[12] == 0.0)
    check("on: mixed group bit-identical to off",
          bool(np.array_equal(adv_on[8:12], adv_off[8:12])))
    check("on: anchor episodes flagged", sum(ep.is_anchor for ep in on.episodes) == 4)
    check("on: n_anchor_groups == 1", s_on["n_anchor_groups"] == 1)
    check("on: dead count drops to 2 (all-fail + singleton)", s_on["n_dead_groups"] == 2)
    check("on: n_live_groups counts SIGNAL groups only", s_on["n_live_groups"] == 1)
    check("on: chunks inherit is_anchor",
          sum(1 for c in on._build_chunks() if c.is_anchor) == 8)

    # Advantage summaries stay signal-only so the curves keep their meaning.
    check("on: mean_advantage excludes anchors (mixed group is zero-sum)",
          close(s_on["mean_advantage"], 0.0))
    check("on: pct_positive_advantage excludes anchors",
          close(s_on["pct_positive_advantage"], 2 / 9),
          f"got {s_on['pct_positive_advantage']}")

    # Layer 1: flagged, admitted, zero advantage.
    l1 = _buffer(outcomes)
    adv_l1 = l1.compute_advantages(anchor_advantage=0.0, include_anchor_groups=True)
    check("layer 1: anchor advantage is exactly 0", bool(np.all(adv_l1[:4] == 0.0)))
    check("layer 1: anchors still flagged (KL-only)",
          sum(ep.is_anchor for ep in l1.episodes) == 4)


def test_row_budget():
    print("\n[buffer] anchor row budget")
    outcomes = [[True] * 4, [True, True, False, False]]  # 4 anchor eps, 4 signal eps
    # Signal chunks = 4 eps x 2 = 8. frac 0.25 -> budget 2 chunks -> 1 episode.
    bud = _buffer(outcomes)
    bud.compute_advantages(
        anchor_advantage=0.2, include_anchor_groups=True, anchor_max_row_frac=0.25,
    )
    s = bud.stats()
    check("budget 0.25: 1 anchor episode kept", s["n_anchor_episodes"] == 1)
    check("budget 0.25: 3 dropped", s["n_anchor_episodes_dropped"] == 3)
    check("budget 0.25: dropped episodes revert to advantage 0",
          bool(np.all(bud.advantages[1:4] == 0.0)))
    check("budget 0.25: group still counted as anchor", s["n_anchor_groups"] == 1)

    # A budget too small for even one episode still keeps one, rather than
    # silently deleting the feature.
    tiny = _buffer(outcomes)
    tiny.compute_advantages(
        anchor_advantage=0.2, include_anchor_groups=True, anchor_max_row_frac=1e-6,
    )
    check("sub-episode budget keeps exactly one anchor episode",
          tiny.stats()["n_anchor_episodes"] == 1)

    # No signal chunks at all -> budget waived (an all-success iteration).
    allsucc = _buffer([[True] * 4, [True] * 4])
    allsucc.compute_advantages(
        anchor_advantage=0.2, include_anchor_groups=True, anchor_max_row_frac=0.1,
    )
    s = allsucc.stats()
    check("all-success iter: budget waived, every anchor kept",
          s["n_anchor_episodes"] == 8 and s["n_anchor_episodes_dropped"] == 0)
    # The waiver must be visible: the drop-print only fires when something was
    # dropped, so without its own line an unbounded anchor pool leaves no trace.
    buf = io.StringIO()
    waived = _buffer([[True] * 4, [True] * 4])
    with contextlib.redirect_stdout(buf):
        waived.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True,
                                  anchor_max_row_frac=0.1)
    check("all-success iter: the waiver is logged", "WAIVED" in buf.getvalue(),
          f"stdout was {buf.getvalue()!r}")
    # It is not only the all-success case: an all-fail + all-success mix also has
    # zero signal chunks, and carries a non-zero std_reward so the trainer's
    # outer skip does not fire.
    buf2 = io.StringIO()
    mix = _buffer([[True] * 4, [False] * 4])
    with contextlib.redirect_stdout(buf2):
        mix.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True,
                               anchor_max_row_frac=0.01)
    check("all-fail + all-success mix also waives, and logs it",
          "WAIVED" in buf2.getvalue() and mix.stats()["std_reward"] > 1e-8,
          f"std_reward={mix.stats()['std_reward']}")
    check("all-success iter: std_reward == 0 (trainer skip must be anchor-aware)",
          s["std_reward"] < 1e-8)
    check("all-success iter: no dead groups", s["n_dead_groups"] == 0)


def test_fully_dropped_anchor_group_counts_as_dead():
    """An anchor group that loses every episode to the budget is dead, not live.

    Without the `n_dead += len(anchor_gids) - n_anchor_groups` adjustment,
    n_live_groups (documented as SIGNAL groups only) over-counts.
    """
    print("\n[buffer] a fully-dropped anchor group is accounted dead")
    # 2 all-success groups (4 eps x 2 chunks each) + 1 mixed group (8 signal
    # chunks). frac 0.25 -> budget 2 chunks -> only group 0's first episode fits;
    # group 1 loses every episode.
    b = _buffer([[True] * 4, [True] * 4, [True, True, False, False]])
    b.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True,
                         anchor_max_row_frac=0.25)
    st = b.stats()
    check("one anchor group survived", st["n_anchor_groups"] == 1, str(st))
    check("the fully-dropped anchor group counts as dead",
          st["n_dead_groups"] == 1, f"n_dead_groups={st['n_dead_groups']}")
    check("n_live_groups counts only the mixed group",
          st["n_live_groups"] == 1, f"n_live_groups={st['n_live_groups']}")
    check("groups sum to the total",
          st["n_anchor_groups"] + st["n_dead_groups"] + st["n_live_groups"]
          == st["n_groups"], str(st))


def test_flags_and_memo_are_reset_between_calls():
    """Re-running compute_advantages must not leave stale flags or chunks.

    The memo matters more than the flag: a chunk built under the previous config
    keeps its old NON-ZERO advantage, which passes the update's live filter even
    when the gate is now off — so it would train as an ordinary signal row.
    """
    print("\n[buffer] flags and the chunk memo reset across calls")
    outcomes = [[True] * 4, [True, True, False, False]]
    b = _buffer(outcomes)
    b.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True)
    first = b._build_chunks()
    stale = 0.2 / 2   # anchor_advantage / chunks_per_episode
    check("first pass produced anchor chunks",
          any(c.is_anchor for c in first) and
          any(abs(c.advantage - stale) < 1e-12 for c in first),
          str(sorted({round(c.advantage, 4) for c in first})))

    b.compute_advantages()          # same buffer, anchors now OFF
    second = b._build_chunks()
    check("the chunk memo was invalidated (fresh objects)",
          second is not first and all(c is not d for c, d in zip(second, first)))
    check("no chunk still claims to be an anchor",
          not any(c.is_anchor for c in second))
    check("no chunk retains the stale anchor advantage",
          not any(abs(c.advantage - stale) < 1e-12 for c in second),
          str(sorted({round(c.advantage, 4) for c in second})))
    check("episode flags were reset",
          not any(ep.is_anchor for ep in b.episodes))


def test_stats_chunk_counts_are_precisely_filtered():
    """`n_signal_chunks` must exclude anchors AND count only non-zero advantages.

    Both filters were unpinned: the only prior assertion used a Layer-1 fixture
    where `anchor_advantage=0.0` already forced anchors out via the advantage
    filter, so the anchor clause it meant to pin was redundant there.
    """
    print("\n[buffer] the chunk counts are precisely filtered")
    # anchor_advantage > 0 so the anchor episodes have a NON-ZERO advantage: the
    # advantage filter alone would then count them, and only `not ep.is_anchor`
    # keeps them out of n_signal_chunks.
    b = _buffer([[True] * 4, [False] * 4, [True, True, False, False]],
                chunks_per_ep=3)
    b.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True,
                         anchor_max_row_frac=100.0)
    st = b.stats()
    check("anchors have a non-zero advantage in this fixture",
          all(b.advantages[i] == 0.2 for i in range(4)), str(b.advantages[:4]))
    check("n_signal_chunks counts ONLY the mixed group (4 eps x 3 chunks)",
          st["n_signal_chunks"] == 12, f"got {st['n_signal_chunks']}")
    check("n_anchor_chunks counts ONLY the all-success group",
          st["n_anchor_chunks"] == 12, f"got {st['n_anchor_chunks']}")
    # The all-fail group has advantage 0 and is not an anchor: it must appear in
    # neither count, so a `>= 0.0` advantage filter would over-count by 12.
    check("the all-fail group is in neither count",
          st["n_signal_chunks"] + st["n_anchor_chunks"] == 24,
          f"signal={st['n_signal_chunks']} anchor={st['n_anchor_chunks']}")


def test_config_validation():
    print("\n[config] anchor validation")
    for kwargs, why in (
        (dict(anchor_advantage=-0.1, include_anchor_groups=True), "negative advantage"),
        (dict(anchor_advantage=0.2), "advantage without the gate"),
        (dict(anchor_max_row_frac=0.0), "non-positive row budget"),
    ):
        try:
            GRPOConfig(**kwargs)
            check(f"rejects {why}", False, "no ValueError raised")
        except ValueError:
            check(f"rejects {why}", True)
    # Checking only that SOMETHING raised is not enough: with the checks in the
    # wrong order a 0.0 budget trips the gate check instead, so the operator is
    # told to enable a flag rather than that the value is invalid.
    try:
        GRPOConfig(anchor_max_row_frac=0.0)
        check("0.0 budget reports the positivity error", False, "no raise")
    except ValueError as exc:
        check("0.0 budget reports the positivity error, not the gate error",
              "must be > 0" in str(exc), str(exc)[:90])
    check("defaults keep the feature off",
          GRPOConfig().include_anchor_groups is False
          and GRPOConfig().anchor_advantage == 0.0)


# ===========================================================================
# 2. Update path (real _grpo_update_inner on CPU)
# ===========================================================================

def _mixed_chunks(n_signal: int, n_anchor: int, anchor_adv: float) -> list:
    """Signal chunks with alternating advantage signs + anchor chunks.

    Anchor chunks deliberately carry DIFFERENT per-chunk advantages around
    `anchor_adv` (as `anchor_advantage / num_chunks` does in production, where
    episode lengths differ), so a test can detect the z-score-amplification bug:
    if anchors were renormalized, that ~2x spread becomes +/-1.

    Their `raw_action` is placed in [2.5, 3.5], outside the signal band
    (+/-0.25..1.75 from `h._make_chunks`), so `_anchor_rows` can identify them
    from the harness's recorded features REGARDLESS of advantage. Identifying
    them by advantage value instead would misclassify KL-only anchors
    (advantage == 0.0) as signal rows.
    """
    chunks = h._make_chunks(n_signal, n_groups=2)
    for i in range(n_anchor):
        raw = 2.5 + 0.1 * (i % 6)
        chunks.append(h._Chunk(
            advantage=anchor_adv / (1.0 + 0.5 * (i % 3)),   # 1x / 0.67x / 0.5x
            feat=raw,
            group_id=100 + (i % 2),
            is_anchor=True,
            ref_log_prob=0.0,
            base_log_prob=0.0,
            tau_samples=np.zeros(6, dtype=np.float32),
            raw_action=np.full((1, 1), raw, dtype=np.float32),
        ))
    return chunks


def _run(chunks: list, **overrides) -> h._Run:
    """run_update() over a hand-built chunk list (signal and/or anchor)."""
    orig = h._make_chunks
    h._make_chunks = lambda n_chunks, n_groups=1: list(chunks)
    try:
        return h.run_update(1, **overrides)
    finally:
        h._make_chunks = orig


def _anchor_rows(rec: dict) -> torch.Tensor:
    """Anchor rows in a recorded micro-batch, identified STRUCTURALLY.

    `rec["f"][:, 0]` is the row's mean raw_action, which `_mixed_chunks` places
    above 2.0 for anchors and within +/-1.75 for signal rows. Deliberately not
    keyed on the advantage: an anchor row at `anchor_advantage == 0` is
    indistinguishable from a signal row by value, which would silently turn the
    KL-only assertions below into statements about the wrong row set.
    """
    return rec["f"][:, 0] > 2.0


def _buffer_signal_std(chunks: list) -> tuple:
    """(mean, std) over the SIGNAL chunks' advantages — mirrors buffer_adv_*."""
    a = np.array([c.advantage for c in chunks if not c.is_anchor], dtype=np.float64)
    if a.size < 2:
        return 0.0, 0.0
    return float(a.mean()), float(a.std(ddof=1))


def _expected_clip_loss(
    cfg, adv, delta, anchor_mask, signal_mb_size,
    buf_mean, buf_std, *, z_anchors: bool, anchors_in_play: bool = True,
    anchor_advantage: float = 0.2, pool_mean_abs: float = 0.0,
) -> float:
    """Independent reimplementation of the per-minibatch clip loss.

    `z_anchors=False` is the documented behavior (anchors scaled by
    1/buffer_std, no mean subtraction; signal rows z-scored among themselves;
    constant `signal_mb_size` divisor). `z_anchors=True` is the bug this guards
    against — anchors pooled into the minibatch z-score, which is what would
    amplify episode-length variation to ±1. The two must not agree, or the
    assertion proves nothing.
    """
    A = adv.clone().to(torch.float32)
    if z_anchors:
        if A.numel() > 1:
            A = (A - A.mean()) / (A.std() + 1e-8)
    else:
        sig = A[~anchor_mask]
        if buf_std > 1e-8:
            anchor_scale = 1.0 / (buf_std + 1e-8)
        else:
            # No signal rows in the whole iteration: anchors are normalized by
            # the mean magnitude of the ITERATION-WIDE anchor pool, so rows land
            # at ~anchor_advantage. Must be the pool mean, not this batch's:
            # using the batch mean makes the per-batch errors nearly cancel in
            # the aggregate, which left the old version agreeing with the
            # implementation to within 27% of its tolerance while being unable to
            # tell the two normalizations apart at all.
            _ma = pool_mean_abs
            anchor_scale = (anchor_advantage / _ma) if _ma > 0.0 else 0.0
        if cfg.per_iteration_advantage_norm:
            sig_vals = ((A - buf_mean) / (buf_std + 1e-8)) if buf_std > 1e-8 else A
        elif sig.numel() > 1:
            sig_vals = (A - sig.mean()) / (sig.std() + 1e-8)
        elif buf_std > 1e-8:
            sig_vals = (A - buf_mean) / (buf_std + 1e-8)
        else:
            sig_vals = A
        A = torch.where(anchor_mask, adv.to(torch.float32) * anchor_scale, sig_vals)

    r = torch.exp(delta)
    surr1 = A * r
    surr2 = A * torch.clamp(r, 1 - cfg.clip_eps_low, 1 + cfg.clip_eps_high)
    row = -torch.min(surr1, surr2)
    if anchors_in_play:
        # Constant divisor for EVERY batch of the iteration, including the ones
        # the fractional quota left anchor-free.
        return float(row.sum() / signal_mb_size)
    return float(row.mean())


def _observed_signal_mb_size(r: h._Run) -> int:
    """Signal rows in the fullest recorded micro-batch = signal_mb_size."""
    return max(
        int((~_anchor_rows(rec)).sum()) for rec in r.trained_records
    )


def test_anchors_reach_the_update():
    print("\n[update] anchor rows train, quota respected")
    chunks = _mixed_chunks(n_signal=12, n_anchor=6, anchor_adv=0.2)
    r = _run(chunks, mb_size=8, epochs=1,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    check("anchor rows trained", r.result.get("n_anchor_rows_trained", 0) > 0,
          str(r.result.get("n_anchor_rows_trained")))
    check("mean_ratio_anchor emitted", "mean_ratio_anchor" in r.result)
    check("kl_loss_anchor emitted", "kl_loss_anchor" in r.result)
    # 6 anchor / 18 total -> round(8 * 1/3) = 3 anchors + 5 signal per batch.
    sizes = {b for (kind, b) in r.events if kind == "prep"}
    check("total rows per minibatch never exceeds mini_batch_size",
          max(sizes) <= 8, f"sizes={sorted(sizes)}")
    # `rec["B"] > 0` was vacuous (the stub would raise, and neither sampler
    # yields an empty list). Assert the quota instead: no batch may hold more
    # anchor rows than the reservation allows.
    per_batch = [int(_anchor_rows(rec).sum()) for rec in r.trained_records]
    check("no minibatch exceeds the anchor slot reservation",
          max(per_batch) <= 8 - _observed_signal_mb_size(r),
          f"anchors/batch={per_batch}, slots={8 - _observed_signal_mb_size(r)}")
    check("at least one batch actually received anchor rows", max(per_batch) > 0,
          str(per_batch))

    # The config gate. Production always pairs `is_anchor` with the advantage
    # `compute_advantages` assigned, so the honest feature-off representation of
    # an all-success group is (is_anchor=True, advantage=0.0) — exactly what
    # Layer 1 produces. Those rows must be dropped when the gate is off.
    gated = _mixed_chunks(n_signal=12, n_anchor=6, anchor_adv=0.0)
    g_off = _run(gated, mb_size=8, epochs=1)
    g_on = _run(gated, mb_size=8, epochs=1,
                config_overrides=dict(include_anchor_groups=True,
                                      anchor_advantage=0.0))
    check("gate off: anchor chunks filtered out",
          "n_anchor_rows_trained" not in g_off.result)
    check("gate on: the same chunks train",
          g_on.result.get("n_anchor_rows_trained", 0) > 0)
    check("gate off: fewer trained rows than gate on",
          sum(rec["B"] for rec in g_off.records)
          < sum(rec["B"] for rec in g_on.records),
          f"{sum(rec['B'] for rec in g_off.records)} vs "
          f"{sum(rec['B'] for rec in g_on.records)}")


def test_signal_only_run_is_bit_identical():
    print("\n[update] no-anchor path is bit-identical")
    signal = h._make_chunks(16, n_groups=2)
    base = _run(signal, mb_size=4, epochs=2)
    # Same chunks, feature ON but no all-success group present -> no anchor rows,
    # so every value must match the feature-off run exactly.
    withflag = _run(signal, mb_size=4, epochs=2,
                    config_overrides=dict(include_anchor_groups=True,
                                          anchor_advantage=0.2))
    for key in ("loss", "clip_loss", "kl_loss_last_iter", "clipfrac",
                "mean_ratio", "mean_log_ratio_abs", "n_updates",
                "n_micro_batches", "grad_norm_mean", "ratio_max", "ratio_min"):
        check(f"{key} unchanged when no anchor rows exist",
              close(base.result[key], withflag.result[key], 0.0),
              f"{base.result[key]} vs {withflag.result[key]}")
    check("final weights bit-identical",
          bool(torch.equal(base.w_final, withflag.w_final)))


def test_anchors_excluded_from_renorm():
    print("\n[update] anchors excluded from the per-minibatch z-score")
    chunks = _mixed_chunks(n_signal=6, n_anchor=6, anchor_adv=0.2)
    r = _run(chunks, mb_size=8, epochs=1,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    check("ran at least one minibatch", len(r.trained_records) > 0)

    buf_mean, buf_std = _buffer_signal_std(chunks)
    smb = _observed_signal_mb_size(r)
    check("buffer signal std is non-degenerate", buf_std > 1e-8, str(buf_std))

    # Compare the trainer's ACTUAL reported clip_loss (mean over micro-batches)
    # against two independent reimplementations: the documented one, and the one
    # where anchors get pooled into the minibatch z-score. Only the first may
    # match, or the assertion proves nothing.
    want = np.mean([
        _expected_clip_loss(r.config, rec["adv"], rec["delta"],
                            _anchor_rows(rec), smb, buf_mean, buf_std,
                            z_anchors=False)
        for rec in r.trained_records
    ])
    bug = np.mean([
        _expected_clip_loss(r.config, rec["adv"], rec["delta"],
                            _anchor_rows(rec), smb, buf_mean, buf_std,
                            z_anchors=True)
        for rec in r.trained_records
    ])
    got = r.result["clip_loss"]
    n_anchor_batches = sum(
        1 for rec in r.trained_records if bool(_anchor_rows(rec).any())
    )
    check("at least one anchor-bearing minibatch ran", n_anchor_batches > 0)
    check("clip_loss matches the anchors-excluded formula",
          close(got, want, 2e-6), f"got {got:.9f}, want {want:.9f}")
    check("clip_loss DIFFERS from the anchors-z-scored formula",
          not close(got, bug, 1e-4),
          f"both formulas give {got:.9f} — the assertion above has no power")


def test_loss_divisor_is_constant_across_batches():
    print("\n[update] under-filled batches don't spike the gradient")
    # 11 signal + 6 anchor at mb_size=8 leaves a trailing batch with a single
    # signal row. Weighting rows by the REALIZED signal count would give that
    # row (and the 3 anchor rows beside it) 5x their intended weight, making a
    # 4-row batch the largest step of the epoch.
    chunks = _mixed_chunks(n_signal=11, n_anchor=6, anchor_adv=0.2)
    r = _run(chunks, mb_size=8, epochs=1, max_grad_norm=1e9,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    per_batch = [
        (int((~_anchor_rows(rec)).sum()), float(g.norm()))
        for rec, g in zip(r.trained_records, r.step_grads)
    ]
    thin = [g for n, g in per_batch if n <= 1]
    full = [g for n, g in per_batch if n > 1]
    check("the scenario produced an under-filled batch", bool(thin) and bool(full),
          str(per_batch))
    if thin and full:
        ratio = max(thin) / max(full)
        check("under-filled batch's grad norm is not inflated", ratio < 2.0,
              f"thin={max(thin):.3f} vs full={max(full):.3f} (ratio {ratio:.2f})")

    # A single signal row must still be normalized (buffer-wide fallback), not
    # left raw at ~1/num_chunks scale. Checked through the aggregate loss, whose
    # reimplementation encodes that fallback.
    smb = _observed_signal_mb_size(r)
    buf_mean, buf_std = _buffer_signal_std(chunks)
    want = np.mean([
        _expected_clip_loss(r.config, rec["adv"], rec["delta"],
                            _anchor_rows(rec), smb, buf_mean, buf_std,
                            z_anchors=False)
        for rec in r.trained_records
    ])
    check("loss matches the constant-divisor + buffer-fallback formula",
          close(r.result["clip_loss"], want, 2e-6),
          f"got {r.result['clip_loss']:.9f}, want {want:.9f}")


def test_anchor_exposure_is_proportional():
    print("\n[update] anchor exposure tracks pool size, two-sided")
    # Bounds are two-sided on purpose. A one-sided "<= 1.5x" check cannot see
    # UNDER-delivery, which is what happens if the reserved slot count rounds
    # below the fractional target (the cap then pins every batch and the credit
    # grows without bound), and the pools must span the regimes where the target
    # rounds both up and down.
    for pool in (1, 2, 3, 5, 8, 13, 16, 24, 30, 40):
        chunks = _mixed_chunks(n_signal=40, n_anchor=pool, anchor_adv=0.2)
        r = _run(chunks, mb_size=8, epochs=1,
                 config_overrides=dict(include_anchor_groups=True,
                                       anchor_advantage=0.2))
        trained = r.result.get("n_anchor_rows_trained", 0)
        exposure = trained / pool
        check(f"pool={pool}: exposure in [0.85, 1.20] (got {exposure:.2f})",
              0.85 <= exposure <= 1.20,
              f"{trained} rows / {pool} chunks")

    # And the realized anchor:signal ROW ratio must track the pool ratio, which
    # is the quantity anchor_max_row_frac is documented to control.
    for pool, want in ((8, 0.20), (16, 0.40), (24, 0.60), (30, 0.75)):
        chunks = _mixed_chunks(n_signal=40, n_anchor=pool, anchor_adv=0.2)
        r = _run(chunks, mb_size=8, epochs=1,
                 config_overrides=dict(include_anchor_groups=True,
                                       anchor_advantage=0.2))
        n_a = r.result.get("n_anchor_rows_trained", 0)
        n_tot = sum(rec["B"] for rec in r.trained_records)
        got = n_a / max(n_tot - n_a, 1)
        check(f"pool={pool}: realized row ratio {got:.3f} within 15% of {want}",
              abs(got - want) <= 0.15 * want, f"got {got:.3f}, want {want}")


def test_divisor_is_gated_on_the_iteration_not_the_batch():
    """Anchor-free batches must still use the constant divisor.

    The fractional quota leaves some batches with no anchor row. Gating the
    divisor on "this batch holds an anchor" sends those through `.mean()`, so a
    trailing 1-signal-row batch weights its row at 1.0 instead of
    1/signal_mb_size — and which batches get that treatment is decided by the
    credit accumulator.
    """
    print("\n[update] the divisor is gated on the iteration, not the batch")
    chunks = _mixed_chunks(n_signal=34, n_anchor=3, anchor_adv=0.2)
    r = _run(chunks, mb_size=8, epochs=1, max_grad_norm=1e9,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    comp = [(rec["B"] - int(_anchor_rows(rec).sum()), int(_anchor_rows(rec).sum()))
            for rec in r.trained_records]
    check("the composition includes an anchor-FREE batch",
          any(a == 0 for _s, a in comp), str(comp))
    check("and an under-filled one", min(sg for sg, _a in comp) <= 2, str(comp))

    smb = _observed_signal_mb_size(r)
    buf_mean, buf_std = _buffer_signal_std(chunks)
    want = np.mean([
        _expected_clip_loss(r.config, rec["adv"], rec["delta"], _anchor_rows(rec),
                            smb, buf_mean, buf_std, z_anchors=False,
                            anchors_in_play=True)
        for rec in r.trained_records
    ])
    per_batch = np.mean([
        _expected_clip_loss(r.config, rec["adv"], rec["delta"], _anchor_rows(rec),
                            smb, buf_mean, buf_std, z_anchors=False,
                            anchors_in_play=bool(_anchor_rows(rec).any()))
        for rec in r.trained_records
    ])
    got = r.result["clip_loss"]
    check("clip_loss uses the constant divisor on every batch",
          close(got, want, 2e-6), f"got {got:.9f}, want {want:.9f}")
    check("and DIFFERS from the per-batch-gated variant",
          not close(got, per_batch, 1e-6),
          f"both give {got:.9f} — the assertion above has no power")

    # Both KL terms take the same divisor and must be gated the same way.
    def _kl_terms(per_batch_gate: bool) -> float:
        tot = 0.0
        for rec in r.trained_records:
            inv = -rec["delta"].to(torch.float64)
            kl = inv.exp() - inv - 1.0
            gated = bool(_anchor_rows(rec).any()) if per_batch_gate else True
            tot += float(kl.sum() / smb) if gated else float(kl.mean())
        return tot / len(r.trained_records) * r.config.kl_coef_last_iter

    check("kl_loss_last_iter uses the constant divisor on every batch",
          close(r.result["kl_loss_last_iter"], _kl_terms(False), 2e-9),
          f"got {r.result['kl_loss_last_iter']:.12f}, "
          f"want {_kl_terms(False):.12f}")
    check("and DIFFERS from the per-batch-gated KL",
          not close(r.result["kl_loss_last_iter"], _kl_terms(True), 1e-12),
          "the assertion above has no power")

    rb = _run(chunks, mb_size=8, epochs=1, max_grad_norm=1e9,
              config_overrides=dict(include_anchor_groups=True,
                                    anchor_advantage=0.2,
                                    kl_coef_base_model=0.2))
    tot_add = tot_gated = 0.0
    for rec in rb.trained_records:
        inv = -rec["delta"].to(torch.float64)
        kl = inv.exp() - inv - 1.0
        tot_add += float(kl.sum() / smb)
        tot_gated += (float(kl.sum() / smb) if bool(_anchor_rows(rec).any())
                      else float(kl.mean()))
    n = len(rb.trained_records)
    check("kl_loss_base_model uses the constant divisor on every batch",
          close(rb.result["kl_loss_base_model"], tot_add / n * 0.2, 2e-9),
          f"got {rb.result['kl_loss_base_model']:.12f}, "
          f"want {tot_add / n * 0.2:.12f}")
    check("and DIFFERS from the per-batch-gated base KL",
          not close(rb.result["kl_loss_base_model"], tot_gated / n * 0.2, 1e-12),
          "the assertion above has no power")

    # Gradient-norm uniformity: no batch may be an outlier purely because of how
    # many anchor rows the accumulator happened to give it.
    norms = [float(g.norm()) for g in r.step_grads]
    check("no batch's grad norm exceeds 3x the largest full batch",
          max(norms) <= 3.0 * sorted(norms)[len(norms) // 2],
          f"norms={[round(n, 3) for n in norms]}")


def test_paws_ema_untouched_when_anchors_off():
    """The EMA-fold guard must not change behavior with anchors disabled.

    Skipping the fold on a zero-mass iteration is right for an ANCHOR-ONLY
    iteration (no signal row pooled anything) but must not apply when anchors are
    off, where a zero-mass iteration is a legitimate all-rows-clip-dead one whose
    fold is part of the pre-anchor behavior.
    """
    print("\n[update] PAWS EMA fold is unchanged with anchors off")
    seen: dict = {}
    real = h.GRPOTrainer._grpo_update

    def spy(self):
        out = real(self)
        seen["N"] = self._pos_scale_N_ema
        seen["D"] = self._pos_scale_D_ema
        return out

    paws = dict(positive_advantage_weight_scaling=True,
                per_iteration_advantage_norm=True,
                clip_eps_low=1e-6, clip_eps_high=1e-6)  # every row clip-dead
    h.GRPOTrainer._grpo_update = spy
    try:
        _run(_mixed_chunks(16, 0, 0.2), mb_size=8, epochs=1,
             config_overrides=paws, pos_scale_ema=(1.0, 1.0))
        off = (seen["N"], seen["D"])
        seen.clear()
        _run(_mixed_chunks(0, 10, 0.2), mb_size=4, epochs=1,
             config_overrides=dict(**paws, include_anchor_groups=True,
                                   anchor_advantage=0.2),
             pos_scale_ema=(1.0, 1.0))
        anchor_only = (seen["N"], seen["D"])
    finally:
        h.GRPOTrainer._grpo_update = real

    check("anchors off: the fold still runs (EMA moves off its seed)",
          off != (1.0, 1.0), f"EMA stayed at {off}")
    # The distinguishing case for the guard's predicate: signal rows PRESENT but
    # zero pooled mass, with anchors also present. Keyed on `entries` the fold
    # runs (pre-anchor behavior); keyed on `anchors_in_play` it would be skipped.
    # Buildable by aligning each row's delta sign with its advantage sign so
    # negatives are lower-clip-dead and positives upper-clip-dead at once.
    aligned = []
    for i in range(8):
        adv = (1.0 + 0.1 * i) * (1.0 if i % 2 == 0 else -1.0)
        raw = 1.0 if adv > 0 else -1.0        # delta sign follows advantage sign
        aligned.append(h._Chunk(
            advantage=adv, feat=raw, group_id=i % 2, ref_log_prob=0.0,
            base_log_prob=0.0, tau_samples=np.zeros(6, dtype=np.float32),
            raw_action=np.full((1, 1), raw, dtype=np.float32)))
    for i in range(4):
        aligned.append(h._Chunk(
            advantage=0.2, feat=2.5, group_id=100, is_anchor=True,
            ref_log_prob=0.0, base_log_prob=0.0,
            tau_samples=np.zeros(6, dtype=np.float32),
            raw_action=np.full((1, 1), 2.5, dtype=np.float32)))
    seen.clear()
    h.GRPOTrainer._grpo_update = spy
    try:
        rz = _run(aligned, mb_size=8, epochs=1,
                  config_overrides=dict(positive_advantage_weight_scaling=True,
                                        per_iteration_advantage_norm=True,
                                        clip_eps_low=1e-6, clip_eps_high=1e-6,
                                        include_anchor_groups=True,
                                        anchor_advantage=0.2),
                  pos_scale_ema=(1.0, 1.0))
    finally:
        h.GRPOTrainer._grpo_update = real
    zero_mass = (rz.result.get("pos_adv_alive_neg_mass", 1.0) == 0.0
                 and rz.result.get("pos_adv_pos_mass", 1.0) == 0.0)
    check("built an iteration with signal rows and zero PAWS mass", zero_mass,
          f"N={rz.result.get('pos_adv_alive_neg_mass')} "
          f"D={rz.result.get('pos_adv_pos_mass')}")
    if zero_mass:
        # have_prior is True (seeded 1.0), so folding zero mass blends toward 0
        # rather than assigning it: 0.5*1.0 + 0.5*0.0 = 0.5. The observable fact
        # is simply that the EMA MOVED off its seed.
        check("signal rows present + zero mass: the fold still runs",
              (seen.get("N"), seen.get("D")) != (1.0, 1.0),
              f"EMA stayed at {(seen.get('N'), seen.get('D'))}; keying the guard "
              f"on anchors_in_play would skip the fold here")
    check("anchor-only: the fold is skipped (EMA left at its seed)",
          anchor_only == (1.0, 1.0), f"EMA moved to {anchor_only}")


def test_jitter_gap_excludes_anchors_from_both_buckets():
    """Anchors must be in NEITHER the fixed nor the jitter mask.

    Row-count inequalities cannot test this: under paired jitter half the signal
    rows are "fixed" and excluded, leaving slack exactly the size of the anchor
    leak, so `bucketed <= signal_rows` passes even with every anchor row in the
    negative bucket. So capture the masks the diagnostic actually receives and
    assert the partition directly. If the callee rebuilt the jitter set as
    `~fixed_row_mask`, `fixed | jitter` would cover every row and the anchors —
    whose gap is structurally 0, since they are never jittered — would drag
    `gap_neg`, and with it `neg_clip_budget_used` (documented as the ceiling on
    `jitter_neg`), toward zero.
    """
    print("\n[update] jitter gap masks partition signal rows only")
    seen: list = []
    real = h.GRPOTrainer._jitter_gap_diagnostics

    def spy(self, **kw):
        seen.append((kw["fixed_row_mask"].clone(),
                     kw["jitter_row_mask"].clone(),
                     kw["pos_adv_mask"].clone()))
        return real(self, **kw)

    h.GRPOTrainer._jitter_gap_diagnostics = spy
    try:
        for paired in (True, False):
            seen.clear()
            chunks = _mixed_chunks(n_signal=12, n_anchor=6, anchor_adv=0.2)
            r = _run(chunks, mb_size=8, epochs=1,
                     config_overrides=dict(include_anchor_groups=True,
                                           anchor_advantage=0.2,
                                           jitter_pos=0.25, jitter_neg=0.05,
                                           jitter_paired=paired))
            check(f"paired={paired}: the diagnostic ran", bool(seen))
            if not seen:
                continue
            fixed, jit, pos = seen[0]
            B = int(fixed.numel())
            n_anchor = int(_anchor_rows(r.trained_records[0]).sum())
            check(f"paired={paired}: the batch contained anchor rows",
                  n_anchor > 0, f"n_anchor={n_anchor}")
            check(f"paired={paired}: fixed and jitter masks are disjoint",
                  int((fixed & jit).sum()) == 0)
            uncovered = B - int((fixed | jit).sum())
            check(f"paired={paired}: exactly the anchor rows are in neither mask",
                  uncovered == n_anchor,
                  f"{uncovered} rows uncovered, {n_anchor} anchors in the batch "
                  f"(a ~fixed_row_mask jitter set would leave 0 uncovered)")
            check(f"paired={paired}: pos_adv_mask also excludes anchors",
                  int((pos & ~(fixed | jit)).sum()) == 0)
            # The masks being right is not enough — the callee must USE them.
            # Derive the buckets from the captured masks and compare against what
            # was emitted; rebuilding the jitter set as ~fixed_row_mask inside the
            # callee changes these counts while leaving the inputs untouched.
            jd = r.result.get("_jitter_diag", {})
            want_pos = int((jit & pos).sum())
            want_neg = int((jit & ~pos).sum())
            check(f"paired={paired}: n_rows_pos matches the jitter&pos mask",
                  jd.get("n_rows_pos", 0) == want_pos,
                  f"emitted {jd.get('n_rows_pos')}, mask gives {want_pos}")
            check(f"paired={paired}: n_rows_neg matches the jitter&neg mask",
                  jd.get("n_rows_neg", 0) == want_neg,
                  f"emitted {jd.get('n_rows_neg')}, mask gives {want_neg} "
                  f"(a ~fixed_row_mask jitter set would inflate this by the "
                  f"anchor row count)")
    finally:
        h.GRPOTrainer._jitter_gap_diagnostics = real


def test_delivery_in_the_cap_rounding_band():
    """Delivery must be 1.00x on the PRODUCTION sampler, not just the stratified one.

    The reservation is sized by solving for capacity (`slots x n_batches`). Sizing
    it as `ceil(mini_batch_size * pool_fraction)` instead implicitly assumes the
    stratified batch count; `_iter_balanced_minibatches` terminates early when its
    majority pool drains, so the realized count is smaller, the per-batch target
    exceeds the cap, every batch gets pinned, and rows are dropped. These
    compositions are the band where that rounding lands on a cap of 1.
    """
    print("\n[update] delivery in the cap-rounding band")
    for S, A, mb in ((56, 8, 8), (64, 9, 8), (72, 10, 8), (80, 11, 8),
                     (40, 12, 4), (20, 6, 4)):
        for balanced in (False, True):
            tag = "balanced" if balanced else "stratified"
            chunks = _mixed_chunks(n_signal=S, n_anchor=A, anchor_adv=0.2)
            r = _run(chunks, mb_size=mb, epochs=1, balanced=balanced,
                     config_overrides=dict(include_anchor_groups=True,
                                           anchor_advantage=0.2))
            got = r.result.get("n_anchor_rows_trained", 0)
            check(f"{tag} S={S} A={A} mb={mb}: all {A} anchor rows train",
                  got == A, f"trained {got}/{A} ({got / A:.3f})")
            sizes = {rec["B"] for rec in r.trained_records}
            check(f"{tag} S={S} A={A} mb={mb}: rows <= mini_batch_size",
                  max(sizes) <= mb, str(sorted(sizes)))

    # Above the coverage limit the cap genuinely binds — that must be WARNED, not
    # silent, since anchor_max_row_frac stops controlling delivered mass there.
    r = _run(_mixed_chunks(n_signal=2, n_anchor=40, anchor_adv=0.2), mb_size=8,
             epochs=1, config_overrides=dict(include_anchor_groups=True,
                                             anchor_advantage=0.2))
    check("beyond the coverage limit the shortfall is reported",
          "anchor row capacity" in r.stdout,
          "no capacity WARNING in stdout")


def test_min_expected_batches_never_overshoots():
    """The batch-count bound must never exceed the real count.

    It sizes the anchor slot reservation (`capacity = slots x n_batches`), so the
    direction is what matters, not exactness: under-estimating over-reserves (an
    extra optimizer step, delivery unaffected), over-estimating under-reserves and
    silently drops anchor rows.

    An earlier version of this test asserted `estimate == real` over a hand-picked
    list of shapes. That is FALSE in general — `ceil(n / mb)` is itself only a
    lower bound for the stratified sampler, which under-fills mid-epoch when a
    group queue drains early and can yield an extra batch — and the test passed
    only because none of its shapes hit that case.
    """
    print("\n[update] the batch-count bound is conservative in the right direction")
    trainer = h.GRPOTrainer.__new__(h.GRPOTrainer)
    over = under = exact = 0
    for S in (1, 2, 3, 5, 8, 12, 13, 20, 33, 40, 57):
        for mb in (1, 2, 3, 4, 5, 7, 8):
            for balanced in (False, True):
                for n_groups in (1, 2, 3, 5):
                    chunks = [
                        h._Chunk(
                            advantage=(1.0 + 0.1 * i) * (1.0 if i % 2 == 0 else -1.0),
                            feat=0.3, group_id=i % n_groups, ref_log_prob=0.0,
                            base_log_prob=0.0,
                            tau_samples=np.zeros(6, dtype=np.float32),
                            raw_action=np.full((1, 1), 0.3, dtype=np.float32))
                        for i in range(S)
                    ]
                    entries = [(c, "fixed") for c in chunks]
                    trainer.config = GRPOConfig(
                        device="cpu", balanced_minibatch_training=balanced)
                    est = trainer._min_expected_batches(entries, mb)
                    rng = np.random.default_rng(3)
                    sampler = (trainer._iter_balanced_minibatches if balanced
                               else trainer._iter_stratified_minibatches)
                    with contextlib.redirect_stdout(io.StringIO()):
                        real = len(list(sampler(entries, rng, mb)))
                    if est > real:
                        over += 1
                        check(f"S={S} mb={mb} balanced={balanced} groups={n_groups}: "
                              f"bound {est} exceeds real {real}", False)
                    elif est < real:
                        under += 1
                    else:
                        exact += 1
    n = over + under + exact
    check(f"bound never exceeds the real count over {n} shapes", over == 0,
          f"{over} overshoots")
    # NOT "exact in the common case": that holds only for the uniform group sizes
    # this fixture builds (group_id = i % n_groups). On the skewed (big, 1, 1, ...)
    # shapes that dynamic collection produces — the ones train_grpo.py itself calls
    # the common case — the stratified sampler yields far more batches than
    # ceil(n/mb) and the bound is loose by design. Assert the property that is
    # actually true and load-bearing: it is a bound, and it is not vacuous.
    check("the bound is exact on uniform group sizes", exact > 0.9 * n,
          f"exact in {exact}/{n}; conservative in {under}")
    skewed_over = 0
    for S in (20, 60, 120):
        for mb in (3, 5, 8, 12):
            for balanced in (False, True):
                sizes = [S - 3] + [1, 1, 1]
                gids = [g for g, k in enumerate(sizes) for _ in range(k)]
                chunks = [
                    h._Chunk(advantage=(1.0 + 0.1 * i) * (1.0 if i % 2 == 0 else -1.0),
                             feat=0.3, group_id=gids[i], ref_log_prob=0.0,
                             base_log_prob=0.0,
                             tau_samples=np.zeros(6, dtype=np.float32),
                             raw_action=np.full((1, 1), 0.3, dtype=np.float32))
                    for i in range(S)
                ]
                entries = [(c, "fixed") for c in chunks]
                trainer.config = GRPOConfig(
                    device="cpu", balanced_minibatch_training=balanced)
                est = trainer._min_expected_batches(entries, mb)
                sampler = (trainer._iter_balanced_minibatches if balanced
                           else trainer._iter_stratified_minibatches)
                with contextlib.redirect_stdout(io.StringIO()):
                    real = len(list(sampler(entries, np.random.default_rng(3), mb)))
                if est > real:
                    skewed_over += 1
    check("bound still never overshoots on SKEWED group sizes",
          skewed_over == 0, f"{skewed_over} overshoots")
    check("the bound is never zero or negative", exact + under + over == n and n > 0)
    # And the consequence that actually matters: delivery stays complete.
    for S, A, mb in ((8, 2, 4), (12, 3, 4), (40, 12, 4), (56, 8, 8)):
        for balanced in (False, True):
            r = _run(_mixed_chunks(n_signal=S, n_anchor=A, anchor_adv=0.2),
                     mb_size=mb, epochs=1, balanced=balanced,
                     config_overrides=dict(include_anchor_groups=True,
                                           anchor_advantage=0.2))
            got = r.result.get("n_anchor_rows_trained", 0)
            check(f"S={S} A={A} mb={mb} balanced={balanced}: all rows delivered",
                  got == A, f"{got}/{A}")


def test_key_properties_hold_under_both_samplers():
    """Re-check the load-bearing invariants on the PRODUCTION sampler.

    `run_update`'s `balanced` argument defaults to False, but production defaults
    to `balanced_minibatch_training=True`. Every numeric test here inherits that
    default, so a property that holds only on the stratified path would look
    verified. Exposure accounting was exactly such a case.
    """
    print("\n[update] invariants hold on both samplers")
    for balanced in (False, True):
        tag = "balanced" if balanced else "stratified"
        for pool in (1, 3, 8, 24):
            chunks = _mixed_chunks(n_signal=40, n_anchor=pool, anchor_adv=0.2)
            r = _run(chunks, mb_size=8, epochs=1, balanced=balanced,
                     config_overrides=dict(include_anchor_groups=True,
                                           anchor_advantage=0.2))
            trained = r.result.get("n_anchor_rows_trained", 0)
            exposure = trained / pool
            check(f"{tag}, pool={pool}: exposure in [0.85, 1.20] "
                  f"(got {exposure:.2f})", 0.85 <= exposure <= 1.20,
                  f"{trained} rows / {pool} chunks")
            sizes = {rec["B"] for rec in r.trained_records}
            check(f"{tag}, pool={pool}: rows/minibatch <= mini_batch_size",
                  max(sizes) <= 8, str(sorted(sizes)))
            per_batch = [int(_anchor_rows(rec).sum()) for rec in r.trained_records]
            check(f"{tag}, pool={pool}: anchor rows respect the reservation",
                  max(per_batch) <= 8 - _observed_signal_mb_size(r),
                  str(per_batch))

    # The balanced sampler's zero-negative-slot fallback must keep the reduced
    # batch size — dropping the mb_size argument there overfills every minibatch.
    src = Path(h.train_grpo.__file__).read_text()
    n = src.count("yield from self._iter_stratified_minibatches(entries, rng, mb_size)")
    check("both balanced fallbacks forward the reduced mb_size", n == 2,
          f"found {n}")
    # Drive the fallback with anchors present: signal_mb_size == 1 forces it.
    chunks = _mixed_chunks(n_signal=2, n_anchor=40, anchor_adv=0.2)
    r = _run(chunks, mb_size=8, epochs=1, balanced=True,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    sizes = {rec["B"] for rec in r.trained_records}
    check("balanced fallback with anchors keeps rows <= mini_batch_size",
          max(sizes) <= 8, str(sorted(sizes)))


def test_tiny_mini_batch_size_warns_instead_of_overfilling():
    print("\n[update] mini_batch_size=1 leaves no anchor slot")
    chunks = _mixed_chunks(n_signal=6, n_anchor=4, anchor_adv=0.2)
    r = _run(chunks, mb_size=1, epochs=1,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    sizes = {rec["B"] for rec in r.trained_records}
    check("rows per minibatch never exceeds mini_batch_size", sizes == {1}, str(sizes))
    check("the skipped anchor rows are reported, not silent",
          "no room for an anchor row" in r.stdout)
    check("no anchor row trained", r.result.get("n_anchor_rows_trained", 0) == 0)
    # And the signal rows must be scaled exactly as in an anchors-off run: with
    # anchors present but untrainable, treating the iteration as anchors-in-play
    # would divert every lone signal row into the anchor-aware renorm branch,
    # whose <2-signal-row fallback replaces the plain path's numel()>1 skip.
    off = _run(_mixed_chunks(n_signal=6, n_anchor=0, anchor_adv=0.2),
               mb_size=1, epochs=1)
    check("signal rows keep the anchors-off scale",
          close(r.result["clip_loss"], off.result["clip_loss"], 1e-9),
          f"anchors-off {off.result['clip_loss']:.9f} vs "
          f"{r.result['clip_loss']:.9f}")


def test_anchor_only_iteration_never_zscores_length():
    print("\n[update] anchor-only iteration keeps a coherent positive pull")
    # Every chunk is an anchor with a DIFFERENT per-chunk advantage (episode
    # length variation). If these were z-scored, half would flip negative and
    # the update would become a pure "prefer shorter episodes" gradient.
    chunks = _mixed_chunks(n_signal=0, n_anchor=12, anchor_adv=0.2)
    r = _run(chunks, mb_size=4, epochs=1,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    check("anchor-only iteration still trains", r.result.get("n_updates", 0) > 0,
          str(r.result))
    check("all trained rows are anchor rows",
          r.result.get("n_anchor_rows_trained", 0)
          == sum(rec["B"] for rec in r.trained_records))
    # No sign-keyed rows at all -> those denominators must stay empty.
    check("no pos/neg effective-clipfrac buckets (no signal rows)",
          "clipfrac_effective_pos" not in r.result
          and "clipfrac_effective_neg" not in r.result)
    check("no sign flips reported", r.result.get("n_pos_flipped_by_renorm", 0) == 0)
    # The surrogate is -A*r with A > 0, so the loss must be negative: the update
    # is pushing these log-probs UP, not z-scoring them into a wash.
    check("clip_loss is negative (reinforcing, not a wash)",
          r.result["clip_loss"] < 0.0, str(r.result["clip_loss"]))


def test_anchor_only_scale_is_numerically_correct():
    """Pin `anchor_scale` on the all-success path, at production chunk counts.

    Replacing `anchor_advantage / mean|A|` with 1.0 previously passed the whole
    suite. It is a ~40x magnitude error in production, where episodes are 30-65
    chunks so the raw per-chunk advantage is anchor_advantage/num_chunks — the
    old fixture's 1/1.5/2 divisors made the correct scale 1.39x, close enough to
    1.0 to hide it. This fixture uses realistic per-chunk magnitudes and checks
    the loss against an independent computation.
    """
    print("\n[update] anchor_scale on the all-success path")
    # Deliberately spread: with (30,45,65) the per-batch means sit close enough to
    # the pool mean that a batch-local implementation also passed. (5,50,95) makes
    # them diverge, so the assertion can actually tell them apart.
    adv_ep, n_chunks = 0.2, (5, 50, 95)
    chunks = []
    for i in range(12):
        raw = 2.5 + 0.1 * (i % 6)
        chunks.append(h._Chunk(
            advantage=adv_ep / n_chunks[i % 3], feat=raw, group_id=100 + (i % 2),
            is_anchor=True, ref_log_prob=0.0, base_log_prob=0.0,
            tau_samples=np.zeros(6, dtype=np.float32),
            raw_action=np.full((1, 1), raw, dtype=np.float32),
        ))
    r = _run(chunks, mb_size=4, epochs=1,
             config_overrides=dict(include_anchor_groups=True,
                                   anchor_advantage=adv_ep))
    check("the anchor-only iteration trained", r.result.get("n_updates", 0) > 0)

    pool_mean = float(np.mean([abs(c.advantage) for c in chunks]))
    want = np.mean([
        _expected_clip_loss(r.config, rec["adv"], rec["delta"],
                            _anchor_rows(rec), 4, 0.0, 0.0, z_anchors=False,
                            anchors_in_play=True, anchor_advantage=adv_ep,
                            pool_mean_abs=pool_mean)
        for rec in r.trained_records
    ])
    got = r.result["clip_loss"]
    check("clip_loss matches the POOL-mean anchor scale",
          close(got, want, 2e-6), f"got {got:.9f}, want {want:.9f}")
    # The pool-vs-batch counterfactual is INVISIBLE in the aggregate, structurally:
    # under a batch-local scale, batch i contributes -(B_i * mean_i) * (adv/mean_i)
    # * r̄, so mean_i cancels and every batch weighs the same; the mean over
    # batches then differs from the pool-scaled version by ~4e-5 relative, well
    # inside any sane tolerance. The difference IS visible per batch, so compare
    # the recorded per-step gradients — which is what the optimizer actually saw.
    lo, hi = 1 - r.config.clip_eps_low, 1 + r.config.clip_eps_high
    coef = r.config.kl_coef_last_iter

    def _expected_grad(rec, scale):
        A = rec["adv"].to(torch.float32) * scale
        ratio = torch.exp(rec["delta"])
        assert bool(((ratio > lo) & (ratio < hi)).all())     # unclamped branch
        g = -((A * ratio).unsqueeze(1) * rec["f"]).sum(dim=0) / 4
        x = -rec["delta"]
        g = g + coef * ((torch.exp(x) - 1.0).unsqueeze(1) * (-rec["f"])).sum(dim=0) / 4
        return g

    pool_err = batch_err = 0.0
    for rec, gstep in zip(r.trained_records, r.step_grads):
        pool_err += float((_expected_grad(rec, adv_ep / pool_mean) - gstep).norm())
        batch_err += float((_expected_grad(
            rec, adv_ep / float(rec["adv"].abs().mean())) - gstep).norm())
    check("per-batch gradients match the POOL-mean scale",
          pool_err < 1e-5, f"residual {pool_err:.3e}")
    check("and NOT a per-BATCH mean|A| scale",
          batch_err > 1e-3,
          f"pool residual {pool_err:.3e} vs per-batch {batch_err:.3e} — the "
          f"assertion above cannot distinguish the two normalizations")
    # An unscaled implementation (scale = 1.0) leaves rows ~1/40th of intended.
    bug = np.mean([
        float((-(rec["adv"].to(torch.float32)
                 * torch.exp(rec["delta"]))).sum() / 4)
        for rec in r.trained_records
    ])
    check("and DIFFERS from an unscaled (scale=1.0) implementation",
          not close(got, bug, 1e-6),
          f"both give {got:.9f} — the assertion above has no power")


def test_layer1_adds_no_clip_gradient():
    print("\n[update] layer 1 (advantage 0) contributes KL only")
    chunks = _mixed_chunks(n_signal=12, n_anchor=6, anchor_adv=0.0)
    # With kl coefficients at 0 AND anchor_advantage at 0, anchor rows must be
    # completely inert: the signal rows' clip_loss is divided by the SIGNAL row
    # count, so it matches a run where the anchors were never admitted.
    kw = dict(mb_size=8, epochs=1, config_overrides=dict(kl_coef_last_iter=0.0))
    off = _run(chunks, **kw)
    on = _run(chunks, mb_size=8, epochs=1,
              config_overrides=dict(kl_coef_last_iter=0.0,
                                    include_anchor_groups=True,
                                    anchor_advantage=0.0))
    check("layer 1 admits the rows", on.result.get("n_anchor_rows_trained", 0) > 0)
    # NOT "the same clip_loss as anchors-off": anchor rows displace signal rows,
    # so the batch compositions and the divisor both differ and the two values
    # legitimately diverge (~10x in this fixture). The property that IS true is
    # that anchor rows contribute exactly nothing to the numerator, checked
    # against an independent sum over the signal rows alone.
    smb_l1 = _observed_signal_mb_size(on)
    bm, bs = _buffer_signal_std(chunks)
    want = np.mean([
        _expected_clip_loss(on.config, rec["adv"], rec["delta"], _anchor_rows(rec),
                            smb_l1, bm, bs, z_anchors=False, anchors_in_play=True,
                            anchor_advantage=0.0)
        for rec in on.trained_records
    ])
    check("layer 1: anchor rows add exactly 0 to the clip numerator",
          close(on.result["clip_loss"], want, 2e-6),
          f"got {on.result['clip_loss']:.9f}, want {want:.9f}")
    check("layer 1: every anchor row's post-scale advantage is 0",
          close(0.0, 0.2 * 0.0), "anchor_scale * 0 == 0 by construction")
    # With KL on, the anchor rows must ADD KL rather than reallocate it. Note a
    # plain `.mean()` (the reallocating implementation) ALSO comes out above the
    # anchors-absent run, so `on > off` alone proves nothing — the additive value
    # has to be matched numerically against an independent computation.
    on_kl = _run(chunks, mb_size=8, epochs=1,
                 config_overrides=dict(kl_coef_last_iter=0.2,
                                       include_anchor_groups=True,
                                       anchor_advantage=0.0))
    smb = _observed_signal_mb_size(on_kl)
    want_add, want_mean = 0.0, 0.0
    for rec in on_kl.trained_records:
        # KL per row from the harness: inv = -delta, kl = e^inv - inv - 1.
        inv = -rec["delta"].to(torch.float64)
        kl = (inv.exp() - inv - 1.0)
        want_add += float(kl.sum() / smb) * 0.2      # additive (correct)
        want_mean += float(kl.mean()) * 0.2          # reallocating (the bug)
    n = len(on_kl.trained_records)
    got = on_kl.result["kl_loss_last_iter"]
    check("KL term matches the ADDITIVE reduction",
          close(got, want_add / n, 2e-9),
          f"got {got:.12f}, additive {want_add / n:.12f}")
    check("KL term DIFFERS from the reallocating (plain-mean) reduction",
          not close(got, want_mean / n, 1e-12),
          f"both give {got:.12f} — the assertion above has no power")


def test_anchors_excluded_from_paws():
    print("\n[update] anchors excluded from PAWS alive mass")
    chunks = _mixed_chunks(n_signal=12, n_anchor=12, anchor_adv=0.2)
    paws = dict(positive_advantage_weight_scaling=True,
                per_iteration_advantage_norm=True)
    # Baseline must be a run where the anchor chunks DON'T EXIST. Reusing the
    # same chunk list with the gate off does not work: those chunks carry a
    # non-zero advantage, so the gate-off run keeps them as ordinary SIGNAL
    # positives rather than dropping them, and the comparison would be measuring
    # the wrong difference.
    signal_only = _mixed_chunks(n_signal=12, n_anchor=0, anchor_adv=0.2)
    off = _run(signal_only, mb_size=8, epochs=1,
               config_overrides=paws, pos_scale_ema=(1.0, 1.0))
    on = _run(chunks, mb_size=8, epochs=1,
              config_overrides=dict(**paws, include_anchor_groups=True,
                                    anchor_advantage=0.2),
              pos_scale_ema=(1.0, 1.0))
    check("baseline has no anchor rows, test run does",
          off.result.get("n_anchor_rows_trained", 0) == 0
          and on.result.get("n_anchor_rows_trained", 0) > 0)
    # 12 anchor rows at 0.2 would add ~2.4 of loss mass to D if pooled — more
    # than doubling it. Excluded, D moves only by the epoch-length change.
    d_off, d_on = off.result["pos_adv_pos_mass"], on.result["pos_adv_pos_mass"]
    # `d_on < 2*d_off` had no power: the two are equal to the last digit because
    # the signal rows and their z-scores are identical in both runs, and even a
    # deliberately anchor-polluted D satisfied the bound. Assert EQUALITY — any
    # anchor contribution to D at all breaks it.
    check("PAWS positive mass is unchanged by the presence of anchor rows",
          close(d_on, d_off, 1e-9), f"D: {d_off:.6f} -> {d_on:.6f}")
    # And show the counterfactual is distinguishable: the anchor rows carry real
    # loss mass, so pooling them WOULD move D.
    anchor_mass = sum(
        float((rec["adv"][_anchor_rows(rec)].abs()
               * torch.exp(rec["delta"][_anchor_rows(rec)])).sum())
        for rec in on.trained_records
    )
    check("the anchor rows carry mass that pooling would have added",
          anchor_mass > 1e-6, f"anchor mass {anchor_mass:.6f}")

    # An anchor-ONLY iteration pools no mass at all, and must leave the EMA
    # alone rather than decaying it toward zero for no information.
    anchor_only = _mixed_chunks(n_signal=0, n_anchor=8, anchor_adv=0.2)
    ao = _run(anchor_only, mb_size=4, epochs=1,
              config_overrides=dict(**paws, include_anchor_groups=True,
                                    anchor_advantage=0.2),
              pos_scale_ema=(1.0, 1.0))
    check("anchor-only iter pools zero PAWS mass",
          ao.result["pos_adv_pos_mass"] == 0.0
          and ao.result["pos_adv_alive_neg_mass"] == 0.0)


def test_balanced_sampler_pools_exclude_anchors():
    print("\n[update] balanced sampler pools stay signal-only")
    # 6 positive + 6 negative signal rows (natural_pos_frac 0.5) and 12 anchor
    # rows. If anchors were pooled as positives the fraction would read 0.75.
    # The DYNAMIC ratio flag is required for this to be observable at all: with
    # it off, _effective_pos_ratio discards natural_pos_frac and returns the
    # configured constant, so the metric could not move either way.
    chunks = _mixed_chunks(n_signal=12, n_anchor=12, anchor_adv=0.2)
    cfg = dict(balanced_minibatch_positive_adv_ratio_dynamic=True,
               balanced_minibatch_positive_adv_ratio=0.5,
               balanced_minibatch_positive_adv_ratio_max=0.9)
    on = _run(chunks, mb_size=8, epochs=1, balanced=True,
              config_overrides=dict(**cfg, include_anchor_groups=True,
                                    anchor_advantage=0.2))
    # The two hypotheses are numerically distinct, which is what gives this
    # power: signal-only -> 6/12 = 0.5; anchors pooled as positives -> 18/24 =
    # 0.75. A signal-only baseline run confirms the 0.5 target independently.
    ref = _run(_mixed_chunks(n_signal=12, n_anchor=0, anchor_adv=0.2),
               mb_size=8, epochs=1, balanced=True, config_overrides=cfg)
    check("anchor rows trained", on.result.get("n_anchor_rows_trained", 0) > 0)
    check("balanced_pos_ratio tracks the SIGNAL fraction only",
          close(on.result["balanced_pos_ratio"], 0.5, 1e-9),
          f"got {on.result.get('balanced_pos_ratio')}; 0.75 would mean "
          f"anchors were pooled as positives")
    check("signal-only baseline reports the same 0.5",
          close(ref.result["balanced_pos_ratio"], 0.5, 1e-9),
          f"got {ref.result.get('balanced_pos_ratio')}")


def test_dynamic_epochs_exclude_anchors():
    print("\n[update] dynamic epochs ignore anchor episodes")
    # Drives the REAL tent formula in _grpo_update_inner. The harness's buffer
    # stub only carries _build_chunks, so attach the episodes/advantages the
    # dynamic-epoch block reads. 8 anchor episodes (all successful) + a 2/4
    # mixed group: pooling the anchors gives success_frac 10/12 -> 1 epoch,
    # excluding them gives 0.5 -> the full update_epochs.
    buf = _buffer([[True] * 8, [True, True, False, False]])
    buf.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True)
    bufchunks = buf._build_chunks()

    chunks = []
    for i, c in enumerate(bufchunks):
        chunks.append(h._Chunk(
            advantage=c.advantage, feat=0.25 * ((i % 7) + 1),
            group_id=c.group_id, is_anchor=c.is_anchor,
            ref_log_prob=0.0, base_log_prob=0.0,
            tau_samples=np.zeros(6, dtype=np.float32),
            raw_action=np.full((1, 1), 0.25 * ((i % 7) + 1), dtype=np.float32),
        ))

    real_update = h.GRPOTrainer._grpo_update

    def _with_episodes(self):
        # The dynamic-epoch block reads self.buffer.episodes / .advantages.
        self.buffer.episodes = buf.episodes
        self.buffer.advantages = buf.advantages
        return real_update(self)

    h.GRPOTrainer._grpo_update = _with_episodes
    try:
        r = _run(chunks, mb_size=8, epochs=4,
                 config_overrides=dict(dynamic_epoch_training=True,
                                       include_anchor_groups=True,
                                       anchor_advantage=0.2))
    finally:
        h.GRPOTrainer._grpo_update = real_update

    check("the real tent formula ran", "actual_epochs" in r.result)
    check("success_fraction is the SIGNAL fraction (0.5), not 10/12",
          close(r.result["success_fraction"], 0.5, 1e-9),
          f"got {r.result.get('success_fraction')}")
    check("epochs stay at the tent peak, not collapsed to 1",
          r.result["actual_epochs"] == 4,
          f"got {r.result.get('actual_epochs')}")


def test_anchor_dominated_buffer_still_trains_signal():
    print("\n[update] anchors can never crowd out every signal row")
    # anchor_max_row_frac is unbounded above, so the anchor pool can dwarf the
    # signal pool: 40 anchor vs 2 signal chunks puts the proportional quota at
    # round(8 * 0.952) = 8 == mini_batch_size. Without reserving a signal slot
    # that leaves signal_mb_size == 0, the inner sampler yields ZERO batches,
    # and the iteration silently trains nothing at all.
    chunks = _mixed_chunks(n_signal=2, n_anchor=40, anchor_adv=0.2)
    r = _run(chunks, mb_size=8, epochs=1,
             config_overrides=dict(include_anchor_groups=True,
                                   anchor_advantage=0.2))
    check("the iteration still trains", r.result.get("n_updates", 0) > 0,
          str(r.result.get("n_updates")))
    sizes = {rec["B"] for rec in r.trained_records}
    check("rows per minibatch never exceeds mini_batch_size",
          max(sizes) <= 8, str(sorted(sizes)))
    signal_rows = [int((~_anchor_rows(rec)).sum()) for rec in r.trained_records]
    check("every minibatch keeps at least one signal row",
          min(signal_rows) >= 1, str(signal_rows))
    check("anchor rows trained too", r.result.get("n_anchor_rows_trained", 0) > 0)


def _iteration_trained(outcomes: list, **cfg) -> bool:
    """Drive the REAL train() loop for one iteration; did it run the update?

    Behavioral, not a source-text check: an earlier version of this test located
    the guard with `src.index(<literal>)` and asserted on the substring, which
    both raises instead of failing when the source is reformatted and passes for
    a rewrite that keeps the substring while changing the logic.

    Every phase is stubbed (borrowing test_phase_timing_logs' harness) except the
    skip decision itself, and `stats()` comes from a REAL EpisodeBuffer over
    `outcomes`, so the composition reaching the guard is the one the buffer would
    actually produce.
    """
    import tempfile
    import test_phase_timing_logs as pt

    buf = _buffer(outcomes)
    buf.compute_advantages(**{k: v for k, v in cfg.items()
                              if k in ("anchor_advantage", "include_anchor_groups",
                                       "anchor_max_row_frac")})
    with tempfile.TemporaryDirectory() as tmp:
        tr = pt._stub_trainer_for_train_loop(tmp, **cfg)
        tr.buffer.stats = lambda: buf.stats()
        with contextlib.redirect_stdout(io.StringIO()):   # train() is chatty
            tr.train()
        return "ref" in tr.calls


def test_zero_gradient_step_is_not_an_update():
    """A step whose gradient is exactly zero must not consume an iteration.

    Reachable, not hypothetical: at LoRA init PEFT sets `lora_B = 0`, so
    base == ref == current and on an anchor-only Layer-1 iteration
    (`anchor_advantage = 0`) the clip term is 0 by construction and BOTH KL
    anchors are KL(p||p) = 0 — every loss term vanishes. Counting the step as an
    update sets `did_update=True`, which burns the iteration from
    `num_iterations`, writes a checkpoint named after it, advances the LR
    schedule, and destroys the retry the skip path exists to preserve.
    """
    print("\n[update] a zero-gradient step is not an update")
    # raw_action = 0 -> delta = 0.05*tanh(0) = 0 -> current == ref == base.
    chunks = [
        h._Chunk(advantage=0.0, feat=0.0, group_id=100 + (i % 2), is_anchor=True,
                 ref_log_prob=0.0, base_log_prob=0.0,
                 tau_samples=np.zeros(6, dtype=np.float32),
                 raw_action=np.zeros((1, 1), dtype=np.float32))
        for i in range(12)
    ]
    r = _run(chunks, mb_size=4, epochs=1,
             config_overrides=dict(include_anchor_groups=True,
                                   anchor_advantage=0.0,
                                   kl_coef_base_model=0.2))
    # n_updates == 0 returns the early dict, which carries no loss values — the
    # observable evidence is that every recorded step gradient was exactly zero.
    check("every step gradient was exactly zero",
          bool(r.step_grads) is False or all(
              float(g.norm()) == 0.0 for g in r.step_grads),
          f"norms={[float(g.norm()) for g in r.step_grads]}")
    check("no optimizer step is counted", r.result.get("n_updates", 0) == 0,
          f"n_updates={r.result.get('n_updates')} — did_update would be True and "
          f"the iteration would be burned")
    check("the dropped steps are reported, not silent",
          r.result.get("n_zero_grad_steps", 0) > 0,
          f"n_zero_grad_steps={r.result.get('n_zero_grad_steps')}")
    check("the weights did not move", bool(torch.equal(r.w0, r.w_final)))

    # A normal iteration must be untouched by the guard.
    ok = _run(_mixed_chunks(n_signal=12, n_anchor=6, anchor_adv=0.2), mb_size=8,
              epochs=1, config_overrides=dict(include_anchor_groups=True,
                                              anchor_advantage=0.2))
    check("a normal iteration still counts its steps",
          ok.result.get("n_updates", 0) > 0
          and ok.result.get("n_zero_grad_steps", 0) == 0,
          f"n_updates={ok.result.get('n_updates')} "
          f"n_zero_grad={ok.result.get('n_zero_grad_steps')}")


def test_trainer_skip_is_anchor_aware():
    """A 100%-success iteration must train only when it can actually learn.

    Regimes, all driven through the real loop:
      - all-success + anchor_advantage > 0  -> TRAIN (the headline case)
      - all-success + anchor_advantage == 0 -> SKIP. Layer 1 has no gradient
        there: clip_loss is identically 0 and the KL anchor is to theta_ref ==
        theta, so stepping applies only weight decay and carried momentum while
        consuming an iteration the pre-anchor code preserved for a retry.
      - ALL-FAIL + ALL-SUCCESS MIX -> same as all-success. This is the case a
        std_reward test cannot see: the 0/1 spread across groups gives
        std_reward = 0.5 while there are zero signal chunks.
      - all-fail, and anchors off -> SKIP.
    """
    print("\n[trainer] the skip decision is keyed on trainable chunks")
    on = dict(include_anchor_groups=True, anchor_advantage=0.2)
    # Layer 1 has two sub-cases. KL(base || current) is NOT degenerate at
    # theta == theta_ref once LoRA has moved, so with kl_coef_base_model > 0 (the
    # DEFAULT 0.2) an anchor-only iteration does carry a gradient — it pulls the
    # policy back toward the pretrained model on the solved states, which is the
    # gap Layer 1 exists to close. With that coefficient at 0 there is genuinely
    # nothing to learn and the iteration must be preserved for a retry.
    l1_base = dict(include_anchor_groups=True, anchor_advantage=0.0,
                   kl_coef_base_model=0.2)
    l1_nobase = dict(include_anchor_groups=True, anchor_advantage=0.0,
                     kl_coef_base_model=0.0)
    allsucc = [[True] * 4, [True] * 4]
    mix = [[True] * 4, [False] * 4]
    allfail = [[False] * 4, [False] * 4]
    mixed_group = [[True] * 4, [True, True, False, False]]

    for label, outcomes, cfg, want_train in (
        ("all-success + advantage>0 trains", allsucc, on, True),
        ("all-success + Layer 1 + base KL trains", allsucc, l1_base, True),
        ("all-success + Layer 1 + NO base KL skips", allsucc, l1_nobase, False),
        ("all-fail+all-success MIX + advantage>0 trains", mix, on, True),
        ("all-fail+all-success MIX + Layer 1 + base KL trains", mix, l1_base, True),
        ("all-fail+all-success MIX + Layer 1, no base KL skips", mix, l1_nobase,
         False),
        ("all-fail + advantage>0 skips", allfail, on, False),
        ("all-fail + Layer 1 + base KL skips (no anchor rows)", allfail, l1_base,
         False),
        ("all-success with anchors OFF skips", allsucc, {}, False),
        ("a mixed group always trains", mixed_group, l1_nobase, True),
    ):
        got = _iteration_trained(outcomes, **cfg)
        check(label, got is want_train, f"trained={got}, wanted {want_train}")

    # The mix is exactly where a std_reward-keyed guard fails: confirm the
    # composition really does carry a non-zero std_reward, so this is not just a
    # restatement of the all-success case.
    b = _buffer(mix)
    b.compute_advantages(anchor_advantage=0.0, include_anchor_groups=True)
    st = b.stats()
    check("the MIX has std_reward > 0 but zero signal chunks",
          st["std_reward"] > 1e-8 and st["n_signal_chunks"] == 0,
          f"std_reward={st['std_reward']} n_signal_chunks={st['n_signal_chunks']}")
    # And anchor CHUNKS, not groups, are what rescue an iteration: a group that
    # survived classification but contributes no rows must not count.
    check("stats exposes chunk counts, not just group counts",
          "n_signal_chunks" in st and "n_anchor_chunks" in st)


def test_ref_logprob_pass_admits_anchor_rows():
    """Anchor rows must get ref/base log-probs, or they can never train.

    Dropping the anchor clause from `_compute_ref_log_probs`' filter leaves the
    rows reserving minibatch slots while being silently discarded by the
    `ready_indices` readiness check — no error, just no anchor training.
    """
    print("\n[trainer] the ref-log-prob pass admits anchor rows")
    src = Path(h.train_grpo.__file__).read_text()
    n = src.count("abs(c.advantage) > 1e-12 or is_anchor_row(c, use_anchors)")
    check("both row filters are the same gated expression", n == 2, f"found {n}")
    check("neither filter reads c.is_anchor directly",
          "or c.is_anchor" not in src)

    # End-to-end: chunks whose ref_log_prob is absent are dropped at
    # ready_indices, so if the ref pass had skipped anchors, no anchor row would
    # appear in n_anchor_rows_trained.
    chunks = _mixed_chunks(n_signal=12, n_anchor=6, anchor_adv=0.2)
    for c in chunks:
        if c.is_anchor:
            c.ref_log_prob = None          # simulate "the ref pass skipped me"
    r = _run(chunks, mb_size=8, epochs=1,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    check("anchor rows without a ref log-prob train nothing",
          r.result.get("n_anchor_rows_trained", 0) == 0,
          "confirms the readiness filter is what makes the ref pass load-bearing")


def test_anchor_entries_are_never_jittered():
    """Anchor entries must be tagged "fixed": lambda is chosen by advantage sign."""
    print("\n[update] anchor entries are always tagged fixed")
    chunks = _mixed_chunks(n_signal=12, n_anchor=6, anchor_adv=0.2)
    r = _run(chunks, mb_size=8, epochs=1,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2,
                                   jitter_pos=0.25, jitter_neg=0.05,
                                   jitter_paired=True))
    src = Path(h.train_grpo.__file__).read_text()
    check("anchor entries are built as \"fixed\"",
          'anchor_entries = [(c, "fixed") for c in anchor_chunks]' in src)
    # The jitter gap diagnostic must not count anchor rows in either bucket.
    jd = r.result.get("_jitter_diag", {})
    n_sig_rows = sum(int((~_anchor_rows(rec)).sum()) for rec in r.trained_records)
    check("jitter buckets hold no more rows than exist as signal rows",
          (jd.get("n_rows_pos", 0) + jd.get("n_rows_neg", 0)) <= n_sig_rows,
          f"pos={jd.get('n_rows_pos')} neg={jd.get('n_rows_neg')} "
          f"signal_rows={n_sig_rows}")


class _RecordingWriter:
    """Captures add_scalar / add_text so the emission side can be asserted."""

    def __init__(self):
        self.calls: list = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, float(value), step))

    def add_text(self, *a, **kw):
        pass


def _emit(cfg_kwargs: dict, stats: dict, update_stats: dict) -> dict:
    """Run the real _log_metrics against a recording writer; return {tag: value}."""
    tr = h.GRPOTrainer.__new__(h.GRPOTrainer)
    tr.config = GRPOConfig(device="cpu", **cfg_kwargs)
    tr.writer = _RecordingWriter()
    tr._ref_mse_stats = None
    tr._chunk_gap_stats = None
    h.GRPOTrainer._log_metrics(tr, 7, stats, update_stats=update_stats,
                               lr=1e-5, iter_time=1.0)
    return {t: v for t, v, _ in tr.writer.calls}


def test_nonfinite_anchor_ratio_is_dropped_not_written():
    """An overflowed ratio must not be written to the anchor curves.

    `ratio = log_ratio.exp()` can overflow to +inf while the clipped loss stays
    finite — with A > 0 the clamp branch wins, and the k3 KL term is finite — so
    the minibatch trains normally and a raw running sum would carry inf into
    `mean_ratio_anchor`, poisoning chart autoscale for the rest of the run. Same
    policy as ratio_maxes/ratio_mins and the sign-split mean_ratio_* metrics.
    """
    print("\n[update] a non-finite anchor ratio is dropped, not written")
    chunks = _mixed_chunks(n_signal=12, n_anchor=6, anchor_adv=0.2)
    anchors = [c for c in chunks if c.is_anchor]
    # Unique raw_action so this chunk's rows are identifiable in the records,
    # which is what makes the finite-vs-trained divisor check possible below.
    OVERFLOW_RAW = 3.9
    anchors[0].ref_log_prob = -120.0        # log_ratio ~ +120 -> exp overflows
    anchors[0].feat = OVERFLOW_RAW
    anchors[0].raw_action = np.full((1, 1), OVERFLOW_RAW, dtype=np.float32)
    r = _run(chunks, mb_size=4, epochs=1,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    check("the iteration still trained", r.result.get("n_updates", 0) > 0)
    check("anchor rows were counted", r.result.get("n_anchor_rows_trained", 0) > 0)
    # Stronger than "absent or finite": the overflowing row must be EXCLUDED
    # from the running sum, so the remaining anchor rows still produce a value.
    # Merely dropping the key at emission time would also satisfy "not inf" while
    # silently erasing the metric for every well-behaved row in the iteration.
    mr = r.result.get("mean_ratio_anchor")
    check("mean_ratio_anchor is still emitted, and finite",
          mr is not None and math.isfinite(mr), f"got {mr!r}")
    check("and it is a plausible ratio (the inf row was excluded, not clamped)",
          mr is not None and 0.5 < mr < 2.0, f"got {mr!r}")
    kl = r.result.get("kl_loss_anchor")
    check("kl_loss_anchor is still emitted, and finite",
          kl is not None and math.isfinite(kl), f"got {kl!r}")
    # This is the ONLY fixture where the two divisors differ, so it is the only
    # place the finite-row divisor can be pinned: n_rows_anchor counts every
    # trained anchor row, n_rows_anchor_finite only those that contributed.
    n_trained = r.result["n_anchor_rows_trained"]
    finite_sum = 0.0
    n_finite = 0
    for rec in r.trained_records:
        m = _anchor_rows(rec)
        if not bool(m.any()):
            continue
        # Exclude the overflowing chunk's rows by their unique raw_action; every
        # other anchor row has ratio exp(delta) since its ref_log_prob is 0.
        keep = m & (rec["f"][:, 0] < OVERFLOW_RAW - 0.05)
        finite_sum += float(torch.exp(rec["delta"][keep].to(torch.float64)).sum())
        n_finite += int(keep.sum())
    check("the two divisors genuinely differ in this fixture",
          n_finite < n_trained, f"finite={n_finite} trained={n_trained}")
    if n_finite < n_trained:
        check("mean_ratio_anchor divides by the FINITE row count",
              abs(mr - finite_sum / n_finite) <= 1e-6 * abs(finite_sum / n_finite),
              f"got {mr:.9f}; finite-divisor {finite_sum / n_finite:.9f}, "
              f"trained-divisor {finite_sum / n_trained:.9f}")
    emitted = _emit(dict(include_anchor_groups=True, anchor_advantage=0.2),
                    {}, r.result)
    bad = {t: v for t, v in emitted.items()
           if "anchor" in t and not math.isfinite(v)}
    check("no non-finite ANCHOR scalar reached the writer", not bad, str(bad))
    # Scoped to anchor keys on purpose. The pre-existing `train/mean_ratio` is
    # also unfiltered and DOES go non-finite here — a longstanding behavior this
    # feature neither introduced nor should silently change, since that curve is
    # compared across existing runs. Noted, not fixed.
    check("the pre-existing train/mean_ratio is the only non-finite scalar "
          "(documents scope, not a passing property of this feature)",
          {t for t, v in emitted.items() if not math.isfinite(v)}
          <= {"train/mean_ratio"},
          str({t: v for t, v in emitted.items() if not math.isfinite(v)}))


def test_anchor_metric_values_are_numerically_pinned():
    """`mean_ratio_anchor` and `kl_loss_anchor` must be the right NUMBERS.

    Both were checked only for presence, finiteness, and a `0.5 < mr < 2.0` band
    that a 1.3x scale error passes — while the docs call mean_ratio_anchor "the
    one to watch" and tell the operator that saturation near 1 + clip_eps_high
    means the clip is bounding the retention move. A 1.3x error would read 1.31
    and look saturated when the true ratio is 1.01.
    """
    print("\n[update] anchor metric values are numerically pinned")
    chunks = _mixed_chunks(n_signal=12, n_anchor=6, anchor_adv=0.2)
    r = _run(chunks, mb_size=8, epochs=1,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2))
    n_rows = r.result.get("n_anchor_rows_trained", 0)
    check("anchor rows trained", n_rows > 0)

    # ref_log_prob is 0 for every chunk, so log_ratio == delta and the k3 KL
    # estimator is exp(-delta) + delta - 1, both row-weighted over anchor rows.
    ratio_sum = kl_sum = 0.0
    counted = 0
    for rec in r.trained_records:
        m = _anchor_rows(rec)
        if not bool(m.any()):
            continue
        d = rec["delta"][m].to(torch.float64)
        ratio_sum += float(torch.exp(d).sum())
        inv = -d
        kl_sum += float((torch.exp(inv) - inv - 1.0).sum())
        counted += int(m.sum())
    check("counted the same anchor rows the trainer did", counted == n_rows,
          f"test {counted} vs trainer {n_rows}")
    want_mr = ratio_sum / counted
    want_kl = 0.2 * kl_sum / counted        # kl_coef_last_iter default
    # Tolerances are relative and sized to float32 accumulation round-off
    # (~1e-7 relative), which is orders of magnitude tighter than the scale
    # errors that previously slipped through (1.3x on the ratio, 2x on the KL).
    check("mean_ratio_anchor matches an independent row-weighted mean",
          abs(r.result["mean_ratio_anchor"] - want_mr) <= 1e-6 * abs(want_mr),
          f"got {r.result['mean_ratio_anchor']:.12f}, want {want_mr:.12f}")
    check("kl_loss_anchor matches an independent row-weighted mean",
          abs(r.result["kl_loss_anchor"] - want_kl) <= 1e-5 * abs(want_kl),
          f"got {r.result['kl_loss_anchor']:.12f}, want {want_kl:.12f}")
    # A wrong divisor is the specific bug the finite/total split exists to avoid.
    check("and DIFFERS from dividing by the trained-row count when they differ",
          counted == n_rows or not close(r.result["mean_ratio_anchor"],
                                         ratio_sum / n_rows, 1e-9))


def test_anchor_metrics_actually_reach_the_writer():
    """The anchor curves must be emitted, and only when the feature is on.

    Nothing else in the suite touches _log_metrics, so deleting the add_scalar
    calls — or the keys from the writer's list — would otherwise be invisible.
    """
    print("\n[logging] anchor metrics reach the writer")
    stats = {"success_rate": 1.0, "mean_reward": 1.0, "std_reward": 0.0,
             "mean_num_steps": 30.0, "std_num_steps": 1.0,
             "n_groups": 3, "n_dead_groups": 1, "n_live_groups": 1,
             "group_success_min": 0.0, "group_success_median": 0.5,
             "group_success_max": 1.0, "mean_advantage": 0.0,
             "std_advantage": 1.0, "pct_positive_advantage": 0.5,
             "n_anchor_groups": 1, "n_anchor_episodes": 4,
             "n_anchor_episodes_dropped": 2, "n_signal_chunks": 30,
             "n_anchor_chunks": 12}
    upd = {"loss": 0.1, "clip_loss": 0.05, "kl_loss_last_iter": 1e-4,
           "clipfrac": 0.0, "mean_ratio": 1.0, "mean_log_ratio_abs": 1e-3,
           "n_updates": 3, "n_micro_batches": 3, "n_skipped_nonfinite": 0,
           "n_nonfinite_grad_steps": 0, "grad_norm_mean": 0.1,
           "grad_norm_max": 0.2, "ratio_max": 1.01, "ratio_min": 0.99,
           "actual_epochs": 1, "n_pos_flipped_by_renorm": 0,
           "n_anchor_rows_trained": 9, "mean_ratio_anchor": 1.02,
           "kl_loss_anchor": 3e-5}

    on = _emit(dict(include_anchor_groups=True, anchor_advantage=0.2), stats, upd)
    for tag, want in (("episode/n_anchor_groups", 1),
                      ("episode/n_anchor_episodes", 4),
                      ("episode/n_anchor_episodes_dropped", 2),
                      ("episode/n_signal_chunks", 30),
                      ("episode/n_anchor_chunks", 12),
                      ("train/n_anchor_rows_trained", 9),
                      ("train/mean_ratio_anchor", 1.02),
                      ("train/kl_loss_anchor", 3e-5)):
        check(f"emits {tag}", tag in on and close(on[tag], want, 1e-9),
              f"got {on.get(tag)!r}")

    off = _emit({}, {k: v for k, v in stats.items() if "anchor" not in k},
                {k: v for k, v in upd.items() if "anchor" not in k})
    new_tags = [t for t in off if "anchor" in t or "signal_chunks" in t]
    check("anchors off: no anchor/signal curves", not new_tags, str(new_tags))
    check("anchors off: the pre-existing episode/* keys are unchanged",
          {t for t in off if t.startswith("episode/")}
          == {t for t in on if t.startswith("episode/")
              and "anchor" not in t and "signal_chunks" not in t})


def test_wandb_key_set_unchanged_when_anchors_off():
    """buffer.stats() reports the anchor counters unconditionally; the wandb
    bulk-dump must not therefore hand an anchors-off run three new keys."""
    print("\n[logging] wandb key set is unchanged with anchors off")
    b = _buffer([[True] * 4, [True, True, False, False]])
    b.compute_advantages()                      # anchors OFF
    st = b.stats()
    check("stats() still carries the counters (TB/wandb gate them, not stats)",
          all(k in st for k in ("n_anchor_groups", "n_anchor_episodes",
                                "n_anchor_episodes_dropped")))
    # Behavioral, not a source grep: build the wandb payload the real
    # _log_metrics would send and compare its key set against an anchors-off
    # baseline. A source-substring check cannot see a key that stats() adds later
    # but the pop list forgets — which is exactly how two keys leaked.
    import types
    fake_wandb = types.ModuleType("wandb")
    sent: list = []
    fake_wandb.log = lambda d: sent.append(dict(d))
    sys.modules["wandb"] = fake_wandb
    try:
        upd = {"n_updates": 1, "n_micro_batches": 1, "n_skipped_nonfinite": 0,
               "n_nonfinite_grad_steps": 0, "loss": 0.1, "clip_loss": 0.05,
               "kl_loss_last_iter": 1e-4, "clipfrac": 0.0, "mean_ratio": 1.0,
               "mean_log_ratio_abs": 1e-3, "grad_norm_mean": 0.1,
               "grad_norm_max": 0.2, "ratio_max": 1.01, "ratio_min": 0.99,
               "actual_epochs": 1, "n_pos_flipped_by_renorm": 0}
        for cfg, tag in ((dict(use_wandb=True), "off"),
                         (dict(use_wandb=True, include_anchor_groups=True,
                               anchor_advantage=0.2), "on")):
            tr = h.GRPOTrainer.__new__(h.GRPOTrainer)
            tr.config = GRPOConfig(device="cpu", **cfg)
            tr.writer = _RecordingWriter()
            tr._ref_mse_stats = None
            tr._chunk_gap_stats = None
            sent.clear()
            h.GRPOTrainer._log_metrics(tr, 3, st, update_stats=upd)
            keys = set(sent[0]) if sent else set()
            leaked = {k for k in keys if "anchor" in k or "signal_chunks" in k}
            if tag == "off":
                check("wandb: anchors-off payload has no anchor/signal keys",
                      not leaked, str(sorted(leaked)))
            else:
                check("wandb: anchors-on payload does carry them",
                      len(leaked) >= 5, str(sorted(leaked)))
    finally:
        sys.modules.pop("wandb", None)


def test_fixed_branch_metrics_exclude_anchors():
    """`mean_ratio_fixed` must be a pure signal-row mean.

    Anchor entries are tagged "fixed", so dropping `& ~anchor_row_mask` from
    fixed_mask folds them into every `*_fixed` curve. Tested by EXACT equality:
    every signal chunk here shares one raw_action, so all signal rows share one
    ratio and `mean_ratio_fixed` must equal it to float precision. The anchor rows
    carry a different ratio, so any leak shifts the mean off that value. (A
    tolerance-based check on the default fixture could not see it — the anchors'
    ratio sat inside the signal rows' spread.)
    """
    print("\n[update] fixed-branch metrics are a pure signal-row mean")
    SIG_RAW, ANC_RAW = 1.0, -2.0
    chunks = []
    for i in range(12):
        chunks.append(h._Chunk(
            advantage=(1.0 + 0.13 * i) * (1.0 if i % 2 == 0 else -1.0),
            feat=SIG_RAW, group_id=i % 2, ref_log_prob=0.0, base_log_prob=0.0,
            tau_samples=np.zeros(6, dtype=np.float32),
            raw_action=np.full((1, 1), SIG_RAW, dtype=np.float32)))
    for i in range(6):
        chunks.append(h._Chunk(
            advantage=0.2 / (1.0 + 0.5 * (i % 3)), feat=ANC_RAW,
            group_id=100 + (i % 2), is_anchor=True, ref_log_prob=0.0,
            base_log_prob=0.0, tau_samples=np.zeros(6, dtype=np.float32),
            raw_action=np.full((1, 1), ANC_RAW, dtype=np.float32)))

    r = _run(chunks, mb_size=8, epochs=1,
             config_overrides=dict(include_anchor_groups=True, anchor_advantage=0.2,
                                   jitter_pos=0.25, jitter_neg=0.05,
                                   jitter_paired=True))
    check("anchor rows were trained (so there is something to leak)",
          r.result.get("n_anchor_rows_trained", 0) > 0)
    check("per-branch metrics were emitted", "mean_ratio_fixed" in r.result,
          str(sorted(k for k in r.result if k.endswith("_fixed"))))

    # ref_log_prob is 0 for every chunk, so log_ratio == delta == 0.05*tanh(raw).
    sig_ratio = float(np.exp(0.05 * np.tanh(SIG_RAW)))
    anc_ratio = float(np.exp(0.05 * np.tanh(ANC_RAW)))
    check("the two ratios are far apart (the test can discriminate)",
          abs(sig_ratio - anc_ratio) > 0.05,
          f"signal {sig_ratio:.4f} vs anchor {anc_ratio:.4f}")
    check("mean_ratio_fixed equals the signal rows' ratio exactly",
          close(r.result["mean_ratio_fixed"], sig_ratio, 1e-6),
          f"got {r.result['mean_ratio_fixed']:.6f}, signal-only {sig_ratio:.6f}, "
          f"any leak pulls it toward {anc_ratio:.6f}")
    # The JITTER branch needs the same check: anchors are tagged "fixed", so a
    # jitter set built as ~fixed_mask sweeps every anchor row into `_jitter`.
    # Under paired jitter a signal chunk's jitter copy uses eps' != eps, so its
    # ratio differs from sig_ratio — assert against the anchors-off run instead.
    ref = _run([c for c in chunks if not c.is_anchor], mb_size=8, epochs=1,
               config_overrides=dict(jitter_pos=0.25, jitter_neg=0.05,
                                     jitter_paired=True))
    check("mean_ratio_jitter matches the anchors-off run",
          close(r.result["mean_ratio_jitter"], ref.result["mean_ratio_jitter"], 1e-6),
          f"got {r.result['mean_ratio_jitter']:.6f}, anchors-off "
          f"{ref.result['mean_ratio_jitter']:.6f}; a leak pulls it toward "
          f"{anc_ratio:.6f}")

    # The dedupe guard in _with_anchor_rows is unreachable by construction: the
    # per-batch target is pool/n_batches, so an epoch's emissions never exceed the
    # pool and the permutation never wraps mid-epoch. Pin that invariant here
    # rather than testing the dead branch.
    for pool, n_batches in ((6, 2), (6, 3), (6, 12)):
        trainer = h.GRPOTrainer.__new__(h.GRPOTrainer)
        anchors = [(f"a{i}", "fixed") for i in range(pool)]
        inner = ([("s", "fixed")] for _ in range(n_batches))
        emitted = sum(
            len([x for x in b if x in anchors])
            for b in trainer._with_anchor_rows(inner, anchors, 4,
                                               np.random.default_rng(0))
        )
        check(f"pool={pool}, {n_batches} batches: emissions ({emitted}) never "
              f"exceed the pool", emitted <= pool, f"emitted {emitted}")


def test_dynamic_epochs_on_an_anchor_only_iteration():
    """The tent must not fire when there are no signal episodes to balance.

    With `live_ep_indices` empty the fallback (`total = max(0,1)`,
    `successful = 0`) gives `m = 0` and collapses to 1 epoch, while reporting
    `success_fraction = 0.0` for an iteration in which every episode SUCCEEDED.
    """
    print("\n[update] dynamic epochs on an anchor-only iteration")
    buf = _buffer([[True] * 4, [True] * 4])
    buf.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True)
    chunks = [
        h._Chunk(advantage=c.advantage, feat=2.5 + 0.1 * (i % 6),
                 group_id=c.group_id, is_anchor=True, ref_log_prob=0.0,
                 base_log_prob=0.0, tau_samples=np.zeros(6, dtype=np.float32),
                 raw_action=np.full((1, 1), 2.5 + 0.1 * (i % 6), dtype=np.float32))
        for i, c in enumerate(buf._build_chunks())
    ]
    real = h.GRPOTrainer._grpo_update

    def with_episodes(self):
        self.buffer.episodes = buf.episodes
        self.buffer.advantages = buf.advantages
        return real(self)

    h.GRPOTrainer._grpo_update = with_episodes
    try:
        r = _run(chunks, mb_size=4, epochs=6,
                 config_overrides=dict(dynamic_epoch_training=True,
                                       include_anchor_groups=True,
                                       anchor_advantage=0.2))
    finally:
        h.GRPOTrainer._grpo_update = real

    check("runs the configured epochs, not the collapsed 1",
          r.result.get("actual_epochs") == 6,
          f"actual_epochs={r.result.get('actual_epochs')} (update_epochs=6)")
    check("emits no success_fraction rather than a fabricated 0.0",
          "success_fraction" not in r.result,
          f"success_fraction={r.result.get('success_fraction')} on an iteration "
          f"where every episode succeeded")


def test_zero_chunk_anchor_episode_cannot_fake_a_trainable_iteration():
    """An anchor group with no CHUNKS must not make the skip decision say "train".

    The budget admits the first anchor episode unconditionally so a small value
    shrinks the anchor share rather than deleting the feature. Zero-chunk episodes
    can consume that floor, leaving `n_anchor_groups >= 1` with no rows — so the
    skip must consult chunk counts. Keying on group counts trains an iteration
    with literally nothing in it.
    """
    print("\n[trainer] a zero-chunk anchor group is not trainable")
    outcomes = [[True, True]]
    b = _buffer(outcomes, chunks_per_ep=2)
    for ep in b.episodes:
        ep.actions = []
        ep.video_frames = []
        ep.states = []
        ep.raw_actions = []
        ep.action_masks = []
        ep.initial_noises = []
    b.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True)
    st = b.stats()
    check("the group still counts as an anchor group", st["n_anchor_groups"] >= 1,
          str(st))
    check("but contributes zero anchor chunks", st["n_anchor_chunks"] == 0,
          f"n_anchor_chunks={st['n_anchor_chunks']}")
    check("and zero signal chunks", st["n_signal_chunks"] == 0)

    # The two keyings give OPPOSITE answers here, which is what makes this
    # fixture discriminating: chunks -> skip, groups -> train-with-nothing.
    chunk_keyed = st["n_signal_chunks"] == 0 and not (
        st["n_anchor_chunks"] > 0 and 0.2 > 0.0)
    group_keyed = st["n_signal_chunks"] == 0 and not (
        st["n_anchor_groups"] > 0 and 0.2 > 0.0)
    check("chunk-keyed and group-keyed decisions differ here",
          chunk_keyed != group_keyed, f"chunk={chunk_keyed} group={group_keyed}")

    # Drive the real loop: it must SKIP.
    def _zero_chunk_buffer():
        bb = _buffer(outcomes, chunks_per_ep=2)
        for ep in bb.episodes:
            ep.actions = []
            ep.video_frames = []
            ep.states = []
            ep.raw_actions = []
            ep.action_masks = []
            ep.initial_noises = []
        bb.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True)
        return bb

    import tempfile
    import test_phase_timing_logs as pt
    zb = _zero_chunk_buffer()
    with tempfile.TemporaryDirectory() as tmp:
        tr = pt._stub_trainer_for_train_loop(
            tmp, include_anchor_groups=True, anchor_advantage=0.2)
        tr.buffer.stats = lambda: zb.stats()
        with contextlib.redirect_stdout(io.StringIO()):
            tr.train()
        check("the real loop SKIPS a zero-chunk anchor iteration",
              "ref" not in tr.calls, f"calls={tr.calls}")


def test_step_count_rises_with_anchors():
    print("\n[update] the step-count increase is real and disclosed")
    # Anchor rows take minibatch slots, so the signal rows spread over more
    # batches -> more optimizer steps at the same LR. Pinned as a KNOWN effect
    # so it cannot change silently; the startup banner warns about it.
    signal_only = _mixed_chunks(n_signal=12, n_anchor=0, anchor_adv=0.2)
    with_anchors = _mixed_chunks(n_signal=12, n_anchor=12, anchor_adv=0.2)
    off = _run(signal_only, mb_size=8, epochs=2)
    on = _run(with_anchors, mb_size=8, epochs=2,
              config_overrides=dict(include_anchor_groups=True,
                                    anchor_advantage=0.2))
    check("anchors raise the per-iter optimizer step count",
          on.result["n_updates"] > off.result["n_updates"],
          f"{off.result['n_updates']} -> {on.result['n_updates']}")
    check("the increase is bounded by mini_batch_size/(mini_batch_size-slots)",
          on.result["n_updates"] <= 2 * off.result["n_updates"],
          f"{off.result['n_updates']} -> {on.result['n_updates']}")


if __name__ == "__main__":
    test_classification()
    test_row_budget()
    test_fully_dropped_anchor_group_counts_as_dead()
    test_flags_and_memo_are_reset_between_calls()
    test_stats_chunk_counts_are_precisely_filtered()
    test_config_validation()
    test_anchors_reach_the_update()
    test_signal_only_run_is_bit_identical()
    test_anchors_excluded_from_renorm()
    test_loss_divisor_is_constant_across_batches()
    test_anchor_exposure_is_proportional()
    test_divisor_is_gated_on_the_iteration_not_the_batch()
    test_paws_ema_untouched_when_anchors_off()
    test_jitter_gap_excludes_anchors_from_both_buckets()
    test_delivery_in_the_cap_rounding_band()
    test_min_expected_batches_never_overshoots()
    test_key_properties_hold_under_both_samplers()
    test_tiny_mini_batch_size_warns_instead_of_overfilling()
    test_anchor_only_iteration_never_zscores_length()
    test_anchor_only_scale_is_numerically_correct()
    test_layer1_adds_no_clip_gradient()
    test_anchors_excluded_from_paws()
    test_balanced_sampler_pools_exclude_anchors()
    test_dynamic_epochs_exclude_anchors()
    test_anchor_dominated_buffer_still_trains_signal()
    test_zero_gradient_step_is_not_an_update()
    test_trainer_skip_is_anchor_aware()
    test_ref_logprob_pass_admits_anchor_rows()
    test_anchor_entries_are_never_jittered()
    test_nonfinite_anchor_ratio_is_dropped_not_written()
    test_anchor_metric_values_are_numerically_pinned()
    test_anchor_metrics_actually_reach_the_writer()
    test_wandb_key_set_unchanged_when_anchors_off()
    test_fixed_branch_metrics_exclude_anchors()
    test_dynamic_epochs_on_an_anchor_only_iteration()
    test_zero_chunk_anchor_episode_cannot_fake_a_trainable_iteration()
    test_step_count_rises_with_anchors()

    if _failures:
        print(f"\n{RED}{len(_failures)} test(s) FAILED:{RESET}")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"\n{GREEN}All anchor-group tests passed.{RESET}")
