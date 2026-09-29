"""Tests for group_advantage_fixed_std: signal groups divide `r - mean` by 0.5.

Covers the buffer (`EpisodeBuffer.compute_advantages(fixed_std=...)`), the config
switch, the tyro flag and the `train()` call site. In-tree, the off path is checked
bit for bit against an independent re-implementation of the group-std formula.

The check against the PRE-CHANGE module cannot live in-tree (once merged, HEAD
contains the change). It was run out-of-tree: load HEAD's episode_buffer.py with
importlib.util.spec_from_file_location, build identical random buffers (with
anchors, the row budget and post-reopen truncation) in both modules, and compare
advantages, chunks, stats() and counters with fixed_std unset and None. 2000 cases,
identical.

Run with the project venv (CPU):
    .venv/bin/python scripts/grpo/test_group_adv_fixed_std.py
"""

import contextlib
import io
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from episode_buffer import EpisodeBuffer, GRPOEpisode, FIXED_GROUP_ADV_STD  # noqa: E402
from gripper_release import PostReopenFilter  # noqa: E402
from grpo_config import GRPOConfig  # noqa: E402

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
_failures = []

COUNTERS = ("_n_groups", "_n_dead_groups", "_n_anchor_groups",
            "_n_anchor_episodes", "_n_anchor_episodes_dropped")


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  {GREEN}PASS{RESET}  {label}")
    else:
        print(f"  {RED}FAIL{RESET}  {label}" + (f" — {detail}" if detail else ""))
        _failures.append(label)


def _episode(success: bool, gid: int, n_chunks: int = 2) -> GRPOEpisode:
    return GRPOEpisode(
        video_frames=[{}] * n_chunks, states=[{}] * n_chunks, language="t",
        actions=[np.zeros((16, 12))] * n_chunks,
        raw_actions=[np.zeros((50, 128))] * n_chunks,
        action_masks=[np.ones((50, 128))] * n_chunks,
        initial_noises=[np.zeros((50, 128))] * n_chunks,
        success=success, shaped_reward=0.0, env_name="t",
        episode_idx=0, num_steps=100, group_id=gid, env_seed=gid,
    )


def _buffer(groups: list, chunks: list | None = None) -> EpisodeBuffer:
    """One group per outcome list; `chunks` optionally gives per-episode counts."""
    b = EpisodeBuffer()
    for gid, outcomes in enumerate(groups):
        for j, s in enumerate(outcomes):
            ep = _episode(bool(s), gid, 2 if chunks is None else chunks[gid][j])
            ep.episode_idx = len(b.episodes)
            b.episodes.append(ep)
    return b


def _adv(b: EpisodeBuffer, **kw) -> np.ndarray:
    """compute_advantages (copied) without its anchor-budget / truncation logs."""
    with contextlib.redirect_stdout(io.StringIO()):
        return b.compute_advantages(**kw).copy()


def _k_of(k: int, g: int) -> list:
    return [True] * k + [False] * (g - k)


def _reference(b: EpisodeBuffer, scale: float | None = None) -> np.ndarray:
    """Independent signal-group formula; non-signal episodes stay 0."""
    rewards = np.array([float(ep.success) for ep in b.episodes])
    gids = np.array([ep.group_id for ep in b.episodes])
    out = np.zeros_like(rewards)
    for gid in np.unique(gids):
        m = gids == gid
        r = rewards[m]
        if len(r) <= 1:
            continue
        mean_r, std_r = r.mean(), r.std(ddof=1)
        if std_r < 1e-4:
            continue
        out[m] = (r - mean_r) / (std_r if scale is None else scale)
    return out


def _signal_mask(b: EpisodeBuffer) -> np.ndarray:
    gids = np.array([ep.group_id for ep in b.episodes])
    succ = np.array([ep.success for ep in b.episodes])
    return np.array([
        (gids == g).sum() > 1 and 0 < succ[gids == g].sum() < (gids == g).sum()
        for g in gids
    ])


def _random_groups(rng) -> list:
    groups = []
    for _ in range(int(rng.integers(1, 7))):
        g = int(rng.integers(1, 13))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            groups.append([True] * g)
        elif kind == 1:
            groups.append([False] * g)
        else:
            groups.append(list(rng.random(g) < rng.random()))
    return groups


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_constant():
    print("\n[constant] FIXED_GROUP_ADV_STD")
    check("is exactly 0.5", FIXED_GROUP_ADV_STD == 0.5)
    check("equals the largest population std of a [0, 1] reward (a 50/50 split)",
          float(np.std([0.0, 1.0])) == FIXED_GROUP_ADV_STD)


def test_off_path_matches_reference():
    print("\n[off] fixed_std=None is the group-std formula, bit for bit")
    rng = np.random.default_rng(0)
    ref_ok = none_ok = anchors_ok = True
    kw = dict(include_anchor_groups=True, anchor_advantage=0.15)
    for _ in range(300):
        groups = _random_groups(rng)
        dflt = _adv(_buffer(groups))
        ref_ok &= dflt.tobytes() == _reference(_buffer(groups)).tobytes()
        none_ok &= dflt.tobytes() == _adv(_buffer(groups), fixed_std=None).tobytes()
        anchors_ok &= (_adv(_buffer(groups), **kw).tobytes()
                       == _adv(_buffer(groups), fixed_std=None, **kw).tobytes())
    check("default == independent (r - mean) / std(ddof=1), 300 random buffers",
          ref_ok)
    check("explicit fixed_std=None == default", none_ok)
    check("explicit fixed_std=None == default with anchor groups on", anchors_ok)


def test_fixed_values():
    print("\n[on] signal groups divide by exactly 0.5")
    groups = [_k_of(1, 12), _k_of(6, 12), _k_of(10, 12), _k_of(11, 12),
              _k_of(1, 2), _k_of(3, 5), _k_of(7, 9)]
    b = _buffer(groups)
    got = _adv(b, fixed_std=FIXED_GROUP_ADV_STD)
    check("== independent (r - mean) / 0.5, bit for bit",
          got.tobytes() == _reference(_buffer(groups), 0.5).tobytes())
    gids = np.array([ep.group_id for ep in b.episodes])
    succ = np.array([ep.success for ep in b.episodes])
    table = {0: (1.8333, -0.1667), 1: (1.0, -1.0), 2: (0.3333, -1.6667),
             3: (0.1667, -1.8333)}
    for g, (want_s, want_f) in table.items():
        s = float(got[(gids == g) & succ][0])
        f = float(got[(gids == g) & ~succ][0])
        check(f"{groups[g].count(True)}/12: {s:+.4f} / {f:+.4f} as in the README table",
              abs(s - want_s) < 1e-4 and abs(f - want_f) < 1e-4)
    rng = np.random.default_rng(1)
    rand_ok = True
    for _ in range(300):
        groups = _random_groups(rng)
        rand_ok &= (_adv(_buffer(groups), fixed_std=0.5).tobytes()
                    == _reference(_buffer(groups), 0.5).tobytes())
    check("== reference on 300 random buffers (dead groups stay 0)", rand_ok)


def test_invariants():
    print("\n[on] zero sum / sign / bound; ratio within a group kept, across groups not")
    groups = [_k_of(1, 12), _k_of(6, 12), _k_of(10, 12), _k_of(11, 12), _k_of(3, 5)]
    on, off = _buffer(groups), _buffer(groups)
    a_on = _adv(on, fixed_std=0.5)
    a_off = _adv(off)
    gids = np.array([ep.group_id for ep in on.episodes])
    succ = np.array([ep.success for ep in on.episodes])
    check("every signal group sums to 0",
          all(abs(a_on[gids == g].sum()) < 1e-12 for g in range(len(groups))))
    check("successes > 0 and failures < 0",
          bool((a_on[succ] > 0).all() and (a_on[~succ] < 0).all()))
    check("|A| < 2 (|r - mean| < 1 in a mixed group)", float(np.abs(a_on).max()) < 2.0)
    ratio_ok = True
    for g in range(len(groups)):
        m = gids == g
        r = a_on[m] / a_off[m]
        std_g = np.array(groups[g], dtype=float).std(ddof=1)
        ratio_ok &= bool(np.allclose(r, r[0], rtol=0, atol=1e-12)
                         and abs(r[0] - std_g / 0.5) < 1e-12)
    check("each group is rescaled by exactly std_g / 0.5 (within-group ratio kept)",
          ratio_ok)
    lone_off = a_off[(gids == 3) & ~succ][0] / a_off[(gids == 1) & ~succ][0]
    lone_on = a_on[(gids == 3) & ~succ][0] / a_on[(gids == 1) & ~succ][0]
    check(f"a lone 11/12 failure vs a 6/12 failure: {lone_off:.2f}x -> {lone_on:.2f}x",
          abs(lone_off - 3.3166) < 1e-3 and abs(lone_on - 11 / 6) < 1e-12)


def test_classification_unchanged():
    print("\n[on] dead / anchor classification, anchor values and counters unchanged")
    groups = [[True] * 4, [False] * 4, _k_of(2, 4), [True], [False], [True] * 3,
              _k_of(1, 3), [True] * 5]
    for anchors in (False, True):
        for frac in (1.0, 0.25):
            kw = dict(include_anchor_groups=anchors, anchor_advantage=0.15,
                      anchor_max_row_frac=frac)
            on, off = _buffer(groups), _buffer(groups)
            a_on = _adv(on, fixed_std=0.5, **kw)
            a_off = _adv(off, **kw)
            sig = _signal_mask(on)
            counters_ok = all(getattr(on, c) == getattr(off, c) for c in COUNTERS)
            flags_ok = ([ep.is_anchor for ep in on.episodes]
                        == [ep.is_anchor for ep in off.episodes])
            rest_ok = a_on[~sig].tobytes() == a_off[~sig].tobytes()
            anchor_vals = a_on[[ep.is_anchor for ep in on.episodes]]
            anchor_ok = bool((anchor_vals == 0.15).all())
            with contextlib.redirect_stdout(io.StringIO()):
                st_on, st_off = on.stats(), off.stats()
            stats_ok = all(st_on[k] == st_off[k]
                           for k in ("n_signal_chunks", "n_anchor_chunks"))
            check(f"anchors={anchors} frac={frac}: counters, flags, non-signal "
                  f"values, chunk counts",
                  counters_ok and flags_ok and rest_ok and anchor_ok and stats_ok,
                  f"counters={counters_ok} flags={flags_ok} rest={rest_ok} "
                  f"anchor={anchor_ok} stats={stats_ok}")
            if anchors and frac == 1.0:
                check("anchor episodes get exactly anchor_advantage (not / 0.5)",
                      anchor_ok and anchor_vals.size > 0)


def test_per_chunk_split_and_truncation():
    print("\n[on] per-chunk split and post-reopen truncation keep the zero sum")
    groups = [_k_of(3, 5), _k_of(1, 4)]
    chunks = [[3, 7, 2, 5, 4], [6, 1, 9, 2]]
    b = _buffer(groups, chunks)
    adv = _adv(b, fixed_std=0.5)
    cs = b._build_chunks()
    check("each chunk carries A_ep / num_train_chunks",
          all(c.advantage == float(adv[c.episode_idx])
              / b.episodes[c.episode_idx].num_train_chunks for c in cs))
    check("every group's chunk advantages sum to 0",
          all(abs(sum(c.advantage for c in cs if c.group_id == g)) < 1e-12
              for g in range(len(groups))))
    check("the buffer-wide chunk mean is 0 (per-iteration norm keeps every sign)",
          abs(float(np.mean([c.advantage for c in cs]))) < 1e-12)

    import test_gripper_release as gr
    eps = [gr.make_episode(gr.failure_widths(), False, group_id=0),
           gr.make_episode(gr.success_widths(), True, group_id=0),
           gr.make_episode(gr.failure_widths(), False, group_id=0)]
    tb = gr.buffer_from(eps)
    ta = _adv(tb, post_reopen_filter=PostReopenFilter(3), fixed_std=0.5)
    tcs = tb._build_chunks()
    check("the fixture really truncates failures", tb._n_post_reopen_episodes_cut == 2,
          f"cut={tb._n_post_reopen_episodes_cut}")
    check("1/3 group: success +4/3, failures -2/3",
          abs(ta[1] - 4 / 3) < 1e-12 and abs(ta[0] + 2 / 3) < 1e-12
          and abs(ta[2] + 2 / 3) < 1e-12, f"{ta}")
    check("truncated group's chunks still sum to 0",
          abs(sum(c.advantage for c in tcs)) < 1e-12)
    check("truncated chunks carry A_ep / num_train_chunks",
          all(c.advantage == float(ta[c.episode_idx])
              / tb.episodes[c.episode_idx].num_train_chunks for c in tcs))


def test_reentry_rebuilds_chunks():
    print("\n[memo] changing fixed_std on the same buffer rebuilds the chunks")
    groups = [_k_of(1, 4), _k_of(3, 4)]
    b = _buffer(groups)
    _adv(b)
    off = [c.advantage for c in b._build_chunks()]
    _adv(b, fixed_std=0.5)
    on = [c.advantage for c in b._build_chunks()]
    want = [float(a) / 2 for a in _reference(_buffer(groups), 0.5) for _ in range(2)]
    _adv(b)
    back = [c.advantage for c in b._build_chunks()]
    check("chunks carry the fixed-scale values after re-entry", on == want,
          f"{on} vs {want}")
    check("switching back restores the group-std chunks exactly", back == off)


def test_argument_validation():
    print("\n[args] fixed_std is validated before any state changes")
    b = _buffer([_k_of(1, 4), [True] * 3])
    _adv(b, include_anchor_groups=True, anchor_advantage=0.15)
    before = b.advantages.copy()
    memo = b._build_chunks()
    flags = [ep.is_anchor for ep in b.episodes]
    bad = (0, 0.0, -0.5, float("nan"), float("inf"), -float("inf"), True, False,
           "0.5", [0.5])
    raised_all, kept = True, True
    for v in bad:
        try:
            _adv(b, fixed_std=v)
            raised_all = False
        except ValueError:
            pass
        except Exception:  # noqa: BLE001 — anything but a clean ValueError fails
            raised_all = False
        kept &= (b.advantages.tobytes() == before.tobytes() and b._chunks is memo
                 and [ep.is_anchor for ep in b.episodes] == flags)
    check("every invalid fixed_std raises ValueError", raised_all)
    check("a rejected call leaves advantages, anchor flags and the chunk memo alone",
          kept)
    accepted = True
    for v in (0.5, 1, 2.0, np.float64(0.5)):
        try:
            _adv(b, fixed_std=v)
        except ValueError:
            accepted = False
    check("finite positive numbers are accepted (float, int, np.float64)", accepted)


def test_config_validation_and_cli():
    print("\n[config] strict bool switch, default off, tyro spelling")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check("default is False", GRPOConfig().group_advantage_fixed_std is False)
        check("True is accepted",
              GRPOConfig(group_advantage_fixed_std=True).group_advantage_fixed_std
              is True)
        rejected = []
        for v in (0.5, 1, 0, "true", None, np.True_):
            try:
                GRPOConfig(group_advantage_fixed_std=v)
                rejected.append(False)
            except ValueError:
                rejected.append(True)
        check("non-bool values are rejected (0.5 would pass as truthy)", all(rejected),
              f"{rejected}")
        import tyro
        on = tyro.cli(GRPOConfig, args=["--group-advantage-fixed-std"])
        off = tyro.cli(GRPOConfig, args=["--no-group-advantage-fixed-std"])
        dflt = tyro.cli(GRPOConfig, args=[])
    check("tyro: --group-advantage-fixed-std / --no-group-advantage-fixed-std / "
          "default", on.group_advantage_fixed_std is True
          and off.group_advantage_fixed_std is False
          and dflt.group_advantage_fixed_std is False)


def test_train_call_site():
    print("\n[trainer] train() passes the switch through to compute_advantages")
    import test_phase_timing_logs as pt
    for flag, want in ((False, None), (True, FIXED_GROUP_ADV_STD)):
        with tempfile.TemporaryDirectory() as tmp:
            tr = pt._stub_trainer_for_train_loop(
                tmp, group_advantage_fixed_std=flag, include_anchor_groups=True,
                anchor_advantage=0.15)
            seen = []
            orig = tr.buffer.compute_advantages

            def rec(*a, _orig=orig, **k):
                seen.append((a, k))
                return _orig(*a, **k)

            tr.buffer.compute_advantages = rec
            with contextlib.redirect_stdout(io.StringIO()):
                tr.train()
        ok = (len(seen) == 1 and not seen[0][0] and "fixed_std" in seen[0][1]
              and seen[0][1]["fixed_std"] == want
              and type(seen[0][1]["fixed_std"]) is type(want))
        check(f"flag={flag}: one call with fixed_std={want!r}", ok, f"{seen}")
        kw = seen[0][1] if seen else {}
        check(f"flag={flag}: the other arguments are unchanged",
              kw.get("anchor_advantage") == 0.15
              and kw.get("include_anchor_groups") is True
              and kw.get("anchor_max_row_frac") == tr.config.anchor_max_row_frac,
              f"{kw}")


TESTS = [
    test_constant,
    test_off_path_matches_reference,
    test_fixed_values,
    test_invariants,
    test_classification_unchanged,
    test_per_chunk_split_and_truncation,
    test_reentry_rebuilds_chunks,
    test_argument_validation,
    test_config_validation_and_cli,
    test_train_call_site,
]


if __name__ == "__main__":
    for t in TESTS:
        t()
    if _failures:
        print(f"\n{RED}{len(_failures)} check(s) failed:{RESET}")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"\n{GREEN}All group_advantage_fixed_std tests passed.{RESET}")
