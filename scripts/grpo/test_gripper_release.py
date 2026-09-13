"""CPU suite for the post-reopen truncation of failing episodes.

Covers `gripper_release.py` (the detector primitives) and its wiring through
the real `EpisodeBuffer.compute_advantages` / `_build_chunks`, plus the
`GRPOConfig` validation matrix. No GPU, no MuJoCo, no model.

Run:  python scripts/grpo/test_gripper_release.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episode_buffer import EpisodeBuffer, GRPOEpisode  # noqa: E402
from gripper_release import (  # noqa: E402
    DEFAULT_CLOSE_BELOW,
    DEFAULT_MIN_CLOSED_CHUNKS,
    DEFAULT_MIN_TRAIN_CHUNKS,
    DEFAULT_ONSET_MARGIN,
    DEFAULT_OPEN_ABOVE,
    PostReopenFilter,
    close_cross_indices,
    gripper_widths,
    post_reopen_chunk_limit,
    post_reopen_detect,
    reopen_onset_index,
)

# Per-episode (success, num_chunks, onset_idx) measured with the SHIPPED
# defaults (close 0.035 / open 0.055 / onset margin 0.004) over the real
# 48-episode collection at
# ~/Desktop/isaaclab/IsaacAutomator/results/isaac-lab-v2/isaac/iter_0001
# (CoffeeServeMug_PandaOmron_Env, 4 groups x 12, 43 failures / 5 successes).
# This is the regression fixture for the N -> dropped-fraction curve that the
# default keep_chunks was chosen from; `test_real_npz_dir_matches_fixture`
# re-derives it from the raw .npz when that directory is present.
ITER_0001 = [
    (0, 50, 22), (0, 50, 24), (0, 50, 20), (0, 50, 15), (0, 50, 27), (0, 50, 22),
    (1, 42, 37), (0, 50, 23), (0, 50, 22), (0, 50, 19), (0, 50, 24), (1, 32, 26),
    (1, 33, 27), (0, 50, 23), (0, 50, 17), (0, 50, 18), (0, 50, 19), (0, 50, 19),
    (0, 50, 18), (0, 50, 20), (0, 50, 18), (0, 50, 20), (0, 50, 19), (0, 50, 24),
    (0, 50, 21), (0, 50, 18), (0, 50, 16), (0, 50, 22), (0, 50, 19), (0, 50, 23),
    (1, 36, 30), (0, 50, 21), (0, 50, 18), (0, 50, 25), (0, 50, 22), (0, 50, 18),
    (0, 50, 20), (0, 50, 21), (0, 50, 17), (0, 50, 23), (0, 50, 21), (0, 50, 48),
    (0, 50, 16), (1, 28, 21), (0, 50, 26), (0, 50, 16), (0, 50, 18), (0, 50, 17),
]

# The crossing indices the same collection yields — i.e. what the onset walk
# starts from. Pinned so a regression that collapses the onset back onto the
# crossing is caught by name rather than showing up as a plausible-looking
# shift in the drop curve.
ITER_0001_CROSS = [
    23, 25, 21, 16, 29, 23, 37, 24, 23, 20, 25, 28,
    28, 24, 18, 19, 20, 20, 19, 22, 18, 21, 20, 24,
    22, 19, 17, 23, 20, 24, 31, 22, 20, 26, 23, 19,
    21, 22, 18, 24, 22, 49, 17, 23, 27, 17, 18, 17,
]

REAL_NPZ_DIR = Path(
    "~/Desktop/isaaclab/IsaacAutomator/results/isaac-lab-v2/isaac/iter_0001"
).expanduser()

OPEN_W = 0.079   # measured fully-open width
MUG_W = 0.020    # measured width while holding the mug
EMPTY_W = 0.001  # measured width of a close on nothing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def states_from_widths(widths, key="gripper_qpos"):
    """Build a GRPOEpisode.states list carrying the given measured widths."""
    return [
        {key: np.array([[w / 2.0, -w / 2.0]], dtype=np.float32)} for w in widths
    ]


def make_episode(widths, success, group_id=0, n_chunks=None):
    n = len(widths) if n_chunks is None else n_chunks
    return GRPOEpisode(
        video_frames=[{} for _ in range(n)],
        states=states_from_widths(widths),
        language="pick the mug",
        actions=[np.zeros((16, 12), dtype=np.float32) for _ in range(n)],
        raw_actions=[np.zeros((50, 128), dtype=np.float32) for _ in range(n)],
        action_masks=[np.ones((50, 128), dtype=np.float32) for _ in range(n)],
        initial_noises=[np.zeros((50, 128), dtype=np.float32) for _ in range(n)],
        success=success,
        shaped_reward=0.0,
        env_name="robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env",
        episode_idx=0,
        num_steps=n * 8,
        group_id=group_id,
        env_seed=100067 + group_id * 1000,
    )


# The real transition is not a square wave: the fingers spend a chunk or two
# part-open on the way out, which is exactly what separates the ONSET from the
# crossing. Both shape helpers below reproduce that with one ramp chunk, so a
# regression that collapses the onset back onto the crossing shifts every
# assertion here by 1.
RAMP_W = 0.020

# Most primitive tests below isolate the hysteresis edges and the onset walk on
# short synthetic traces whose closed runs are 1-2 chunks long. They pass
# min_closed_chunks=1 to take the DWELL out of the picture and keep the asserted
# indices readable; the dwell is covered on its own by
# test_close_requires_a_dwell, and every test that goes through the real buffer
# or the real .npz uses the shipped default.
MC1 = {"min_closed_chunks": 1}


def failure_widths(n=50, close_at=13, onset_at=21, grasp=EMPTY_W):
    """Canonical failure: approach, close on nothing, reopen, meander.

    Closed over [close_at, onset_at), part-open at onset_at, open after — so
    the crossing is onset_at + 1 and the onset is onset_at.
    """
    w = [OPEN_W] * n
    for i in range(close_at, onset_at):
        w[i] = grasp
    w[onset_at] = RAMP_W
    return w


def success_widths(n=42, close_at=22, onset_at=36):
    """Canonical success: approach, grasp the mug, release, terminate."""
    w = [OPEN_W] * n
    for i in range(close_at, onset_at):
        w[i] = MUG_W
    w[onset_at] = 0.040  # above the mug hold, still below open_above
    return w


def buffer_from(episodes):
    b = EpisodeBuffer()
    for i, ep in enumerate(episodes):
        ep.episode_idx = i
        b.episodes.append(ep)
    return b


def expect_raises(exc, match, fn, label):
    try:
        fn()
    except exc as e:
        assert match in str(e), f"{label}: message {str(e)!r} lacks {match!r}"
        return
    raise AssertionError(f"{label}: expected {exc.__name__}, nothing raised")


# ---------------------------------------------------------------------------
# 1. Detector primitives
# ---------------------------------------------------------------------------

def test_width_extraction():
    st = states_from_widths([0.08, 0.02])
    assert np.allclose(gripper_widths(st), [0.08, 0.02])

    # A state horizon > 1 takes the MOST RECENT row, not the first: the width
    # that decides the cut must be the one the policy was queried with.
    st = [{"gripper_qpos": np.array([[0.04, -0.04], [0.0005, -0.0005]])}]
    assert np.allclose(gripper_widths(st), [0.001])

    # A flat (dim,) value is accepted — reshape(-1, 2) makes it one row.
    st = [{"gripper_qpos": np.array([0.04, -0.04])}]
    assert np.allclose(gripper_widths(st), [0.08])

    expect_raises(
        KeyError, "has no 'gripper_qpos' state key",
        lambda: gripper_widths([{"base_position": np.zeros(3)}]),
        "missing key",
    )
    expect_raises(
        ValueError, "expected (2,) or (state_horizon, 2)",
        lambda: gripper_widths([{"gripper_qpos": np.zeros((1, 7))}]),
        "wrong shape",
    )
    expect_raises(
        ValueError, "expected (2,) or (state_horizon, 2)",
        lambda: gripper_widths([{"gripper_qpos": np.float32(0.08)}]),
        "scalar",
    )
    expect_raises(
        ValueError, "is not finite",
        lambda: gripper_widths([{"gripper_qpos": np.array([[np.nan, 0.0]])}]),
        "NaN",
    )
    expect_raises(
        ValueError, "is not finite",
        lambda: gripper_widths([{"gripper_qpos": np.array([[np.inf, 0.0]])}]),
        "inf",
    )
    # A (0, 2) value satisfies every shape test and then IndexErrors on the
    # [-1]; a (1, 2, 2) value satisfies `shape[-1] == 2` and would be silently
    # reinterpreted as two horizon rows. Both must be rejected, loudly.
    expect_raises(
        ValueError, "expected (2,) or (state_horizon, 2)",
        lambda: gripper_widths([{"gripper_qpos": np.zeros((0, 2))}]),
        "empty",
    )
    expect_raises(
        ValueError, "expected (2,) or (state_horizon, 2)",
        lambda: gripper_widths([{"gripper_qpos": np.zeros((1, 2, 2))}]),
        "3-D",
    )
    expect_raises(
        ValueError, "is not numeric",
        lambda: gripper_widths([{"gripper_qpos": "ab"}]),
        "non-numeric",
    )
    # A custom key is honored.
    assert np.allclose(
        gripper_widths(states_from_widths([0.08], key="fingers"), "fingers"), [0.08]
    )
    print("  PASS: width extraction (horizon, flat, key, eight guards)")


def test_close_cross_hysteresis():
    # Never closes.
    assert close_cross_indices(np.array([0.08] * 10), **MC1) is None
    # Closes and stays closed to the end: nothing to trim (the "grasped it and
    # never let go" failure).
    assert close_cross_indices(np.array([0.08, 0.08, 0.001, 0.001]), **MC1) is None
    # Canonical close -> reopen.
    assert close_cross_indices(np.array([0.08, 0.001, 0.001, 0.079]), **MC1) == (1, 3)

    # Both hysteresis edges are STRICT and exclusive of the band. A width
    # sitting inside [close_below, open_above] must neither enter CLOSED nor
    # leave it.
    assert close_cross_indices(np.array([0.08, 0.045, 0.045, 0.08]), **MC1) is None, \
        "0.045 is inside the band and must not count as a close"
    assert close_cross_indices(np.array([0.08, 0.001, 0.045, 0.045]), **MC1) is None, \
        "0.045 is inside the band and must not count as a reopen"
    # Exactly ON each threshold: `< close_below` and `> open_above` are strict,
    # so neither boundary value triggers.
    assert close_cross_indices(np.array([0.08, 0.035, 0.08]), **MC1) is None
    assert close_cross_indices(np.array([0.08, 0.001, 0.055, 0.056]), **MC1) == (1, 3)

    # A real grasp (0.020) is a close too — the detector is about the OPENING,
    # not about whether the grasp held. Successes are excluded by the caller.
    assert close_cross_indices(np.array([0.08, MUG_W, MUG_W, 0.079]), **MC1) == (1, 3)

    # Only the FIRST cycle is reported, even with a later retry.
    assert close_cross_indices(
        np.array([0.08, 0.001, 0.079, 0.079, 0.001, 0.079])
    , **MC1) == (1, 2)

    # Custom thresholds are honored.
    assert close_cross_indices(np.array([0.08, 0.045, 0.08]), 0.05, 0.06, **MC1) == (1, 2)
    print("  PASS: hysteresis (both edges, band, strictness, retry, custom)")


def test_reopen_onset():
    """The onset is the base of the rising edge, not the crossing."""
    # One ramp chunk: crossing at 4, onset at 3.
    w = np.array([0.079, 0.001, 0.001, 0.020, 0.079])
    assert close_cross_indices(w, **MC1)[1] == 4
    assert reopen_onset_index(w, **MC1) == 3

    # A longer, monotone ramp is walked all the way back to its base.
    w = np.array([0.079, 0.001, 0.001, 0.008, 0.020, 0.040, 0.079])
    assert close_cross_indices(w, **MC1)[1] == 6
    assert reopen_onset_index(w, **MC1) == 3

    # A square transition has no ramp, so onset == crossing.
    w = np.array([0.079, 0.001, 0.001, 0.079])
    assert reopen_onset_index(w, **MC1) == close_cross_indices(w, **MC1)[1] == 3

    # None exactly when close_cross_indices is None.
    for w in (np.array([0.08] * 5), np.array([0.08, 0.001, 0.001])):
        assert close_cross_indices(w, **MC1) is None
        assert reopen_onset_index(w, **MC1) is None

    # The onset can never be the CLOSE chunk itself. What actually guarantees
    # that is the THRESHOLD, not the `onset - 1 > close_idx` bound: reaching
    # close_idx + 1 forces floor == w[close_idx], and the next test is then
    # `w[close_idx] > w[close_idx] + margin`, always False. (Loosening the bound
    # to `>=` is therefore undetectable — it is defence in depth, not the
    # mechanism.) This trace lands on close_idx + 1 via the threshold.
    w = np.array([0.079, 0.030, 0.040, 0.050, 0.079])
    assert close_cross_indices(w, **MC1) == (1, 4)
    assert reopen_onset_index(w, **MC1) == 2
    # Exhaustive, at the SHIPPED thresholds and dwell: over every 3-valued
    # trace up to length 9 that contains a cycle, the onset is strictly greater
    # than the close index and at most the crossing.
    import itertools
    n_checked = 0
    for length in range(3, 10):
        for combo in itertools.product((0.001, 0.020, 0.079), repeat=length):
            arr = np.array(combo)
            found = close_cross_indices(arr)
            if found is None:
                continue
            n_checked += 1
            onset = reopen_onset_index(arr)
            assert found[0] < onset <= found[1], (combo, found, onset)
    assert n_checked > 3000, n_checked
    print(f"  PASS: onset in (close, cross] on {n_checked} exhaustive traces")

    # The floor is the closed-phase MINIMUM, so the margin adapts to the grasp.
    # A close on the mug at 0.020 with a 0.025 ramp: 0.025 > 0.020 + 0.004.
    w = np.array([0.079, 0.020, 0.020, 0.025, 0.079])
    assert reopen_onset_index(w, **MC1) == 3
    # ...but hold-phase wobble inside the margin is NOT an onset.
    w = np.array([0.079, 0.020, 0.020, 0.023, 0.079])
    assert reopen_onset_index(w, **MC1) == 4, "0.023 is within floor + 0.004"
    print("  PASS: onset walks the rising edge, floor-relative, bounded by close")


def test_onset_ignores_mid_hold_blips():
    """A forward scan for 'first rise' would fire on these; the walk does not."""
    # episode_0044's real shape: 0.007 blip, back down to 0.003, then the ramp.
    w = np.array([0.079] * 3 + [0.001, 0.001, 0.007, 0.003, 0.003, 0.042, 0.077])
    assert close_cross_indices(w, **MC1) == (3, 9)
    assert reopen_onset_index(w, **MC1) == 8, "the blip at index 5 must not be reached"
    # episode_0039's real shape: a blip in the middle of the closed run.
    w = np.array([0.079] * 2 + [0.001, 0.007, 0.002, 0.001, 0.027, 0.076])
    assert reopen_onset_index(w, **MC1) == 6
    print("  PASS: isolated mid-hold blips are not mistaken for the onset")


# --- Reference mutants -------------------------------------------------------
# The detector's two load-bearing design choices are (1) walking BACKWARD from a
# confirmed crossing rather than scanning forward, and (2) taking the floor as
# the closed-phase MINIMUM rather than the width at the close chunk. Both were
# originally pinned only by the real-.npz test, which SKIPS when that collection
# is absent — so on any other machine the suite passed with either one mutated.
# These reference implementations make the difference testable without the data.

def _onset_forward_scan(widths, close_below=0.035, open_above=0.055,
                        onset_margin=0.004, min_closed_chunks=1):
    """MUTANT: scan forward for the first rising chunk above the threshold."""
    found = close_cross_indices(widths, close_below, open_above,
                                min_closed_chunks)
    if found is None:
        return None
    close_idx, cross_idx = found
    threshold = widths[close_idx:cross_idx].min() + onset_margin
    for i in range(close_idx + 1, cross_idx):
        if widths[i] > threshold and widths[i] < widths[i + 1]:
            return i
    return cross_idx


def _onset_floor_at_close(widths, close_below=0.035, open_above=0.055,
                          onset_margin=0.004, min_closed_chunks=1):
    """MUTANT: use the width AT the close chunk as the floor."""
    found = close_cross_indices(widths, close_below, open_above,
                                min_closed_chunks)
    if found is None:
        return None
    close_idx, cross_idx = found
    threshold = widths[close_idx] + onset_margin
    onset = cross_idx
    while (onset - 1 > close_idx and widths[onset - 1] > threshold
           and widths[onset - 1] < widths[onset]):
        onset -= 1
    return int(onset)


def _onset_no_monotonicity(widths, close_below=0.035, open_above=0.055,
                           onset_margin=0.004, min_closed_chunks=1):
    """MUTANT: drop the `w[o-1] < w[o]` term, keeping only the threshold."""
    found = close_cross_indices(widths, close_below, open_above,
                                min_closed_chunks)
    if found is None:
        return None
    close_idx, cross_idx = found
    threshold = widths[close_idx:cross_idx].min() + onset_margin
    onset = cross_idx
    while onset - 1 > close_idx and widths[onset - 1] > threshold:
        onset -= 1
    return int(onset)


def test_backward_walk_beats_a_forward_scan():
    """Pin choice (1) WITHOUT the real .npz: a blip that rises into a peak.

    The closed phase contains a blip that is itself rising (0.001 -> 0.007 ->
    0.010) before falling back. A forward scan fires on it; the walk, anchored
    at the confirmed crossing, never reaches it.
    """
    w = np.array([0.079, 0.001, 0.007, 0.010, 0.002, 0.001, 0.040, 0.079])
    assert close_cross_indices(w, **MC1) == (1, 7)
    assert reopen_onset_index(w, **MC1) == 6, "the real edge starts at 6"
    assert _onset_forward_scan(w) == 2, "a forward scan fires on the blip at 2"
    print("  PASS: the backward walk differs from a forward scan (6 vs 2)")


def test_floor_is_the_closed_phase_minimum():
    """Pin choice (2) WITHOUT the real .npz.

    The close chunk is entered on a partially-closed width (0.030) and the
    fingers then settle lower (0.001). Taking the floor at the close chunk puts
    the threshold at 0.034 — above the whole rising edge — so the mutant reports
    the crossing; the true minimum puts it at 0.005 and finds the edge.
    """
    w = np.array([0.079, 0.030, 0.001, 0.001, 0.020, 0.079])
    assert close_cross_indices(w, **MC1) == (1, 5)
    assert reopen_onset_index(w, **MC1) == 4
    assert _onset_floor_at_close(w) == 5, "mutant collapses onto the crossing"
    print("  PASS: the floor is the closed-phase minimum, not the close width")


def test_onset_monotonicity_guard():
    """A high plateau contiguous with the crossing must not be walked through.

    Fingers overshoot to the floor, the object then settles between them at a
    wider hold, and the release follows. The hold is above floor + margin and
    touches the crossing, so the threshold test alone would report the START of
    the hold as the onset. The `w[o-1] < w[o]` term bounds it to the real edge.
    """
    w = np.array([0.079] * 5 + [0.001, 0.001] + [0.020] * 8 + [0.079, 0.079])
    assert close_cross_indices(w, **MC1) == (5, 15)
    assert reopen_onset_index(w, **MC1) == 14, "must not walk back through the 0.020 hold"
    assert _onset_no_monotonicity(w) == 7, "without the term, the hold is walked"

    # The KNOWN COST of that term: a stepped release (open partway, hold, open
    # fully) is indistinguishable from the shape above, so the onset is reported
    # LATE by the plateau length — and two traces differing only by measurement
    # noise on the plateau disagree by several chunks. Pinned so the trade-off
    # is visible rather than discovered later. It errs toward keeping MORE
    # chunks, which is the safe direction.
    flat = np.array([0.079, 0.001, 0.001, 0.020, 0.020, 0.020, 0.079])
    noisy = np.array([0.079, 0.001, 0.001, 0.020, 0.020001, 0.0200011, 0.079])
    assert reopen_onset_index(flat, **MC1) == 5
    assert reopen_onset_index(noisy, **MC1) == 3
    print("  PASS: monotonicity guard bounds the walk (and its plateau cost)")


def test_onset_is_threshold_invariant():
    """The onset is anchored to the base of the edge, so it barely moves.

    Uses a trace with widths spread THROUGH the swept threshold box (0.032,
    0.044, 0.052, 0.061) rather than the canonical three-valued shape — on that
    shape the crossing is equally invariant, so the test could not tell the two
    apart and did not establish the property it claimed.
    """
    w = np.array([0.079, 0.061, 0.044, 0.032, 0.001, 0.001, 0.020, 0.052, 0.079])
    base = reopen_onset_index(w, **MC1)
    assert base == 6
    onsets, crossings = set(), set()
    for close_below in (0.030, 0.035, 0.040, 0.045, 0.050):
        for open_above in (0.050, 0.055, 0.060, 0.070):
            if close_below >= open_above:
                continue
            onsets.add(reopen_onset_index(w, close_below, open_above, **MC1))
            crossings.add(
                close_cross_indices(w, close_below, open_above, **MC1)[1]
            )
    assert onsets == {base}, f"onset moved across the threshold box: {onsets}"
    assert len(crossings) > 1, (
        f"the crossing must MOVE on this trace ({crossings}) or the test cannot "
        f"show the onset is the more robust of the two"
    )
    # And across the measured onset-margin plateau.
    for margin in (0.0025, 0.003, 0.004, 0.005, 0.0055):
        assert reopen_onset_index(w, onset_margin=margin, **MC1) == base, margin
    print(
        f"  PASS: onset invariant over 19 threshold pairs while the crossing "
        f"takes {len(crossings)} values; invariant over the margin plateau"
    )


def test_chunk_limit_semantics():
    # close 13, onset 21, crossing 22, of a 50-chunk episode.
    st = states_from_widths(failure_widths())
    assert close_cross_indices(gripper_widths(st))[1] == 22
    assert reopen_onset_index(gripper_widths(st)) == 21
    # N counts FROM the onset chunk: N=0 drops it, N=k keeps [onset, onset+k).
    for n, expected in ((0, 21), (1, 22), (3, 24), (4, 25), (10, 31)):
        assert post_reopen_chunk_limit(st, PostReopenFilter(n)) == expected, n
    # A window that runs past the end is "no cut", not a clamp to len().
    assert post_reopen_chunk_limit(st, PostReopenFilter(29)) is None
    assert post_reopen_chunk_limit(st, PostReopenFilter(28)) == 49
    # No close at all, and an empty episode.
    assert post_reopen_chunk_limit(states_from_widths([OPEN_W] * 9),
                                   PostReopenFilter(3)) is None
    assert post_reopen_chunk_limit([], PostReopenFilter(3)) is None
    print("  PASS: keep_chunks counts from the onset; overruns are no-ops")


def test_sign_convention_guard():
    """A mirrored qpos convention must RAISE, not silently disable the filter."""
    st = [
        {"gripper_qpos": np.array([[-w / 2.0, w / 2.0]], dtype=np.float32)}
        for w in failure_widths()
    ]
    expect_raises(
        ValueError, "never exceeds open_above",
        lambda: post_reopen_chunk_limit(st, PostReopenFilter(3)),
        "mirrored convention",
    )
    # The same guard catches a thresholds-vs-units mismatch (e.g. cm vs m).
    expect_raises(
        ValueError, "never exceeds open_above",
        lambda: post_reopen_chunk_limit(
            states_from_widths(failure_widths()),
            PostReopenFilter(3, close_below=3.5, open_above=5.5),
        ),
        "wrong units",
    )
    print("  PASS: sign-convention / unit-mismatch guard raises")


def test_filter_validation():
    ok = PostReopenFilter(3)
    assert (ok.keep_chunks, ok.close_below, ok.open_above, ok.onset_margin) == (
        3, 0.035, 0.055, 0.004
    )
    cases = [
        (lambda: PostReopenFilter(-1), "must be >= 0"),
        (lambda: PostReopenFilter(3.0), "must be an int"),
        (lambda: PostReopenFilter(True), "must be an int"),
        (lambda: PostReopenFilter(3, close_below=0.0), "must be > 0"),
        (lambda: PostReopenFilter(3, open_above=float("nan")), "must be finite"),
        (lambda: PostReopenFilter(3, onset_margin=0.0), "must be > 0"),
        (lambda: PostReopenFilter(3, onset_margin=float("inf")), "must be finite"),
        (lambda: PostReopenFilter(3, close_below=0.06), "must be <"),
        (lambda: PostReopenFilter(3, close_below=0.05, open_above=0.05), "must be <"),
        # A margin at or above open_above can never admit a chunk to the walk,
        # so the onset would silently collapse onto the crossing.
        # The bound is open_above - close_below, NOT open_above: the floor is a
        # width that entered CLOSED, so floor < close_below, and a margin of
        # 0.0205 already prevents any chunk clearing floor + margin on the
        # default band. (0.0549 < open_above passes a naive check and collapses
        # every one of the 43 measured failures onto the crossing.)
        (lambda: PostReopenFilter(3, onset_margin=0.020), "collapse onto the crossing"),
        (lambda: PostReopenFilter(3, onset_margin=0.0549), "collapse onto the crossing"),
        (lambda: PostReopenFilter(3, onset_margin=0.2), "collapse onto the crossing"),
        (lambda: PostReopenFilter(3, min_closed_chunks=0), "must be >= 1"),
        (lambda: PostReopenFilter(3, min_closed_chunks=2.0), "must be an int"),
        (lambda: PostReopenFilter(3, min_train_chunks=-1), "must be >= 0"),
        (lambda: PostReopenFilter(3, min_train_chunks=True), "must be an int"),
        # The band-gap check must fire with ITS OWN message. Equal thresholds
        # give a zero band, so the later onset_margin bound also trips and its
        # message likewise contains "must be <" — matching on that substring
        # would pass on the wrong error and blame the wrong knob.
        (lambda: PostReopenFilter(3, close_below=0.05, open_above=0.05),
         "needs a gap between the two"),
        # The margin bound at exact equality, on a binary-exact band (0.5-0.25).
        (lambda: PostReopenFilter(3, close_below=0.25, open_above=0.5,
                                  onset_margin=0.25),
         "collapse onto the crossing"),
        (lambda: PostReopenFilter(3, state_key=""), "non-empty string"),
        (lambda: PostReopenFilter(3, state_key=5), "non-empty string"),
        (lambda: PostReopenFilter(3, close_below="x"), "must be a real number"),
    ]
    for fn, match in cases:
        expect_raises(ValueError, match, fn, "PostReopenFilter validation")
    # Just under the bound is accepted, and numpy integers are a legitimate
    # keep_chunks (they arrive from sweeps and loaded configs).
    assert PostReopenFilter(3, onset_margin=0.0199).onset_margin == 0.0199
    assert PostReopenFilter(np.int64(4)).keep_chunks == 4
    print(f"  PASS: PostReopenFilter validation ({len(cases)} cases)")


# ---------------------------------------------------------------------------
# 1b. The two state-machine guards on the CLOSE edge
# ---------------------------------------------------------------------------

def test_close_requires_an_observed_open_first():
    """An episode that STARTS closed has no detectable close event.

    A close is a transition, not a state. Without this, an `init_state_npz_path`
    run whose branch point is after the grasp latches at chunk 0, reads the
    release of the PREVIOUS grasp as its reopen, and truncates every episode to
    `keep_chunks + 1` chunks — discarding the failed grasp entirely.
    """
    # closed 0-2, released and re-approaching 3-14, real grasp 15-26,
    # ramp at 27, open from 29.
    w = np.array([0.001] * 3 + [OPEN_W] * 12 + [0.001] * 12
                 + [RAMP_W, 0.001] + [OPEN_W] * 22)
    assert close_cross_indices(w) == (15, 29), close_cross_indices(w)
    assert reopen_onset_index(w) == 29
    st = states_from_widths(w)
    assert post_reopen_detect(st, PostReopenFilter(3)) == (29, 32)
    # The pre-fix behaviour, for the record: latch at 0, cross at 3, cut to 6.
    print("  PASS: an episode starting closed is not latched at chunk 0")


def test_close_requires_a_dwell():
    """A single transient dip through the band must not latch CLOSED.

    Hysteresis guards against chatter INSIDE the band; it does nothing about a
    one-sample excursion straight through it, and this signal carries ~6 mm
    single-sample blips. Without a dwell the next open sample — one chunk later
    — reads as the reopen and the whole failed grasp is amputated.
    """
    # one-chunk dip to 0.030 at chunk 3; real grasp 13-20, real onset 21.
    w = np.array([OPEN_W] * 3 + [0.030] + [OPEN_W] * 9
                 + [0.001] * 8 + [RAMP_W] + [OPEN_W] * 28)
    assert close_cross_indices(w) == (13, 22)
    assert reopen_onset_index(w) == 21
    assert post_reopen_detect(states_from_widths(w), PostReopenFilter(3)) == (21, 24)
    # A dwell of 1 reproduces the bug, so the knob is load-bearing.
    assert close_cross_indices(w, min_closed_chunks=1) == (3, 4)
    assert reopen_onset_index(w, min_closed_chunks=1) == 4

    # A TWO-chunk dip is rejected at the shipped dwell of 3 and accepted at 2 —
    # pins which dip lengths each setting tolerates.
    w2 = np.array([OPEN_W] * 3 + [0.030, 0.030] + [OPEN_W] * 8
                  + [0.001] * 8 + [RAMP_W] + [OPEN_W] * 28)
    assert reopen_onset_index(w2, min_closed_chunks=3) == 21
    assert reopen_onset_index(w2, min_closed_chunks=2) == 5

    # close_idx is the FIRST chunk of the dwell run, not the last — the floor
    # and the walk bound both depend on it.
    w3 = np.array([OPEN_W, 0.001, 0.001, 0.001, RAMP_W, OPEN_W])
    assert close_cross_indices(w3, min_closed_chunks=3)[0] == 1
    # A run shorter than the dwell resets, so two separated short runs never add.
    w4 = np.array([OPEN_W, 0.001, OPEN_W, 0.001, OPEN_W, 0.001, OPEN_W])
    assert close_cross_indices(w4, min_closed_chunks=2) is None
    print("  PASS: CLOSED needs a dwell; close_idx is the run start; runs reset")


def test_min_train_chunks_floor():
    """An implausibly short retained prefix is refused, not applied."""
    # onset 4 (close 1, dwell 3, cross 5): at keep=0 the limit is 4 < floor 5.
    w = [OPEN_W] + [0.001] * 3 + [RAMP_W] + [OPEN_W] * 45
    st = states_from_widths(w)
    assert reopen_onset_index(np.array(w)) == 4
    assert post_reopen_detect(st, PostReopenFilter(0)) == (4, None), "refused"
    assert post_reopen_detect(st, PostReopenFilter(0, min_train_chunks=0)) == (4, 4)
    # Above the floor it applies normally.
    assert post_reopen_detect(st, PostReopenFilter(1)) == (4, 5)
    print("  PASS: min_train_chunks refuses an implausible prefix, 0 disables it")


def test_open_above_guard_boundary_matches_the_crossing():
    """A trace whose MAX width is exactly `open_above` can never cross.

    The crossing test is `w > open_above`, so such a trace is a guaranteed silent
    no-op — the guard's `<=` must therefore match the crossing's strictness. With
    `<` it would slip through and return None.
    """
    expect_raises(
        ValueError, "never exceeds open_above",
        lambda: post_reopen_chunk_limit(
            states_from_widths([0.055, 0.001, 0.001, 0.001, 0.055]),
            PostReopenFilter(3),
        ),
        "max width exactly at open_above",
    )
    # A hair above passes the guard.
    st = states_from_widths([0.0551, 0.001, 0.001, 0.001, 0.0551])
    assert post_reopen_chunk_limit(st, PostReopenFilter(3)) is None
    print("  PASS: the max-width guard is strict in step with the crossing test")


def test_onset_threshold_boundary_is_strict():
    """`widths[onset-1] > threshold` is strict: equality must not be walked."""
    # floor 0.0, margin 0.004 -> threshold exactly 0.004; w[2] == 0.004.
    w = np.array([OPEN_W, 0.0, 0.004, RAMP_W, OPEN_W])
    assert close_cross_indices(w, min_closed_chunks=2) == (1, 4)
    assert reopen_onset_index(w, min_closed_chunks=2) == 3, "0.004 is ON the bound"
    print("  PASS: the onset threshold comparison is strict at equality")


def test_num_train_chunks_clamps():
    """`train_chunk_limit` is a public field; both clamps must hold."""
    ep = make_episode(failure_widths(), False)
    assert ep.num_train_chunks == 50
    ep.train_chunk_limit = -5
    assert ep.num_train_chunks == 0, "negative limit clamps to 0"
    ep.train_chunk_limit = 999
    assert ep.num_train_chunks == 50, "over-long limit clamps to num_chunks"
    ep.train_chunk_limit = 0
    assert ep.num_train_chunks == 0
    print("  PASS: num_train_chunks clamps at both ends")


def test_zero_chunk_episode_does_not_divide_by_zero():
    """A 0-chunk episode reaching _build_chunks must not ZeroDivisionError."""
    zero = make_episode([], False, n_chunks=0)
    b = buffer_from([zero, make_episode(failure_widths(), False),
                     make_episode(success_widths(), True)])
    b.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    assert len(b._build_chunks()) == 24 + 42, len(b._build_chunks())
    b2 = buffer_from([make_episode([], False, n_chunks=0),
                      make_episode(success_widths(), True)])
    b2.compute_advantages()
    assert len(b2._build_chunks()) == 42
    print("  PASS: a zero-chunk episode divides by 1, not 0")


def test_summary_log_line_is_correct_not_merely_present():
    """The summary line is the feature's only runtime observability surface.

    Three independent mutations survive a mere "is it printed" check: `n_fail`
    counting successes, `total_fail_chunks` using truncated lengths, and the
    percentage taken over all chunks instead of failure chunks. Assert the
    numbers.
    """
    import contextlib
    import io
    b = buffer_from([make_episode(failure_widths(), False) for _ in range(3)]
                    + [make_episode(success_widths(), True)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        b.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    text = out.getvalue()
    # 3 failures x 50 chunks, each cut to 24 -> 26 dropped each = 78/150 = 52.0%
    assert "in 3/3 failing episode(s)" in text, text
    assert "cut 3" in text, text
    assert "dropping 78/150 failure chunks (52.0%)" in text, text
    print("  PASS: the summary line's counts and percentage are all correct")


def test_first_error_is_the_first_one():
    """Two differently-broken episodes: the warning must name the FIRST."""
    import contextlib
    import io
    missing = make_episode(failure_widths(), False)
    for st in missing.states:
        st.pop("gripper_qpos")
        st["base_position"] = np.zeros(3, dtype=np.float32)
    nan = make_episode(failure_widths(), False)
    nan.states[5] = {"gripper_qpos": np.array([[np.nan, 0.0]])}
    b = buffer_from([missing, nan, make_episode(failure_widths(), False),
                     make_episode(success_widths(), True)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        b.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    text = out.getvalue()
    assert "no 'gripper_qpos' state key" in text, text
    assert "is not finite" not in text, "the SECOND error must not be reported"
    assert b.stats()["n_post_reopen_errors"] == 2
    print("  PASS: the warning reports the first error, not the last")


def test_defaults_agree_across_modules():
    """gripper_release.DEFAULT_* must be what GRPOConfig actually defaults to.

    Mirrors the GROUP_SEED_STRIDE agreement check in test_scene_seed_pool.py:
    the constants live in gripper_release and grpo_config imports them, so this
    pins that they have not been re-hardcoded.
    """
    from grpo_config import GRPOConfig
    c = GRPOConfig()
    assert c.post_reopen_close_width == DEFAULT_CLOSE_BELOW == 0.035
    assert c.post_reopen_open_width == DEFAULT_OPEN_ABOVE == 0.055
    assert c.post_reopen_onset_margin == DEFAULT_ONSET_MARGIN == 0.004
    assert c.post_reopen_min_closed_chunks == DEFAULT_MIN_CLOSED_CHUNKS == 3
    assert c.post_reopen_min_train_chunks == DEFAULT_MIN_TRAIN_CHUNKS == 5
    f = PostReopenFilter(3)
    assert (f.close_below, f.open_above, f.onset_margin) == (
        DEFAULT_CLOSE_BELOW, DEFAULT_OPEN_ABOVE, DEFAULT_ONSET_MARGIN)
    assert (f.min_closed_chunks, f.min_train_chunks) == (
        DEFAULT_MIN_CLOSED_CHUNKS, DEFAULT_MIN_TRAIN_CHUNKS)
    print("  PASS: DEFAULT_* constants agree with GRPOConfig and PostReopenFilter")


# ---------------------------------------------------------------------------
# 2. Wiring through the real EpisodeBuffer
# ---------------------------------------------------------------------------

def test_off_switch_is_bit_identical():
    """post_reopen_filter=None must reproduce the pre-feature behavior exactly."""
    eps = [make_episode(failure_widths(), False) for _ in range(3)]
    eps.append(make_episode(success_widths(), True))
    b = buffer_from(eps)
    adv = b.compute_advantages().copy()
    chunks = b._build_chunks()

    assert len(chunks) == sum(e.num_chunks for e in eps), "no chunk may be dropped"
    assert all(e.train_chunk_limit is None for e in b.episodes)
    assert b.num_train_chunks == b.num_chunks
    s = b.stats()
    assert s["n_post_reopen_episodes_cut"] == 0
    assert s["n_post_reopen_chunks_dropped"] == 0
    assert s["num_train_chunks"] == s["num_chunks"]
    # Per-chunk advantage is still A_ep / num_chunks.
    for c in chunks:
        assert c.advantage == adv[c.episode_idx] / eps[c.episode_idx].num_chunks
    print("  PASS: off-switch drops nothing and leaves every counter at 0")


def test_failures_cut_successes_whole():
    fail = make_episode(failure_widths(), False)            # onset 21, nc 50
    succ = make_episode(success_widths(), True)             # onset 36, nc 42
    b = buffer_from([fail, succ, make_episode(failure_widths(), False),
                     make_episode(success_widths(), True)])
    b.compute_advantages(post_reopen_filter=PostReopenFilter(3))

    for ep in b.episodes:
        if ep.success:
            assert ep.train_chunk_limit is None, "a success must never be cut"
            assert ep.num_train_chunks == ep.num_chunks
        else:
            assert ep.train_chunk_limit == 24
            assert ep.num_train_chunks == 24

    chunks = b._build_chunks()
    kept = {}
    for c in chunks:
        kept[c.episode_idx] = max(kept.get(c.episode_idx, -1), c.chunk_idx)
    assert kept[0] == 23 and kept[2] == 23, "failures truncated at 24 chunks"
    assert kept[1] == 41 and kept[3] == 41, "successes intact"
    s = b.stats()
    assert s["n_post_reopen_episodes_cut"] == 2
    assert s["n_post_reopen_chunks_dropped"] == 2 * 26
    assert s["num_chunks"] == 2 * 50 + 2 * 42
    assert s["num_train_chunks"] == 2 * 24 + 2 * 42
    print("  PASS: failures truncated, successes untouched, counters right")


def test_zero_sum_invariant_preserved():
    """Truncation RELOCATES an episode's credit; it must not shrink it."""
    eps = ([make_episode(failure_widths(onset_at=r), False)
            for r in (18, 22, 26)]
           + [make_episode(success_widths(), True)])
    b = buffer_from(eps)
    adv = b.compute_advantages(post_reopen_filter=PostReopenFilter(3)).copy()
    assert abs(adv.sum()) < 1e-9, "group-relative advantages must sum to 0"

    chunks = b._build_chunks()
    # Every episode's chunk advantages still sum to its episode advantage...
    for i, ep in enumerate(b.episodes):
        got = sum(c.advantage for c in chunks if c.episode_idx == i)
        assert abs(got - adv[i]) < 1e-9, (i, got, adv[i])
    # ...and therefore the whole group still sums to zero at the CHUNK level.
    assert abs(sum(c.advantage for c in chunks)) < 1e-9

    # The divisor really is the KEPT count: a cut episode's per-chunk advantage
    # is magnified by num_chunks / num_train_chunks relative to the uncut run.
    b2 = buffer_from([make_episode(failure_widths(onset_at=r), False)
                      for r in (18, 22, 26)] + [make_episode(success_widths(), True)])
    b2.compute_advantages()
    uncut = {c.episode_idx: c.advantage for c in b2._build_chunks()}
    cut = {c.episode_idx: c.advantage for c in chunks}
    for i, ep in enumerate(b.episodes):
        ratio = ep.num_chunks / ep.num_train_chunks
        assert abs(cut[i] - uncut[i] * ratio) < 1e-12, i
    print("  PASS: Sum A_chunk == A_ep and the group zero-sum both survive")


def test_idempotent_and_self_clearing():
    b = buffer_from([make_episode(failure_widths(), False),
                     make_episode(success_widths(), True)])
    b.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    first = [c.advantage for c in b._build_chunks()]
    assert b.episodes[0].train_chunk_limit == 24

    # Re-running with the same filter changes nothing.
    b.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    assert [c.advantage for c in b._build_chunks()] == first

    # Re-running with the feature OFF restores full lengths — the limit from the
    # previous call must not persist, and the memoized chunk list must not be
    # served stale.
    b.compute_advantages(post_reopen_filter=None)
    assert b.episodes[0].train_chunk_limit is None
    assert len(b._build_chunks()) == b.num_chunks
    assert b.stats()["n_post_reopen_chunks_dropped"] == 0

    # A different N re-cuts from scratch rather than compounding.
    b.compute_advantages(post_reopen_filter=PostReopenFilter(6))
    assert b.episodes[0].train_chunk_limit == 27
    print("  PASS: idempotent, self-clearing, no stale chunk memo")


def test_never_reopened_failure_kept_whole():
    """The 'grasped it and never let go' failure has no meander to trim."""
    w = [OPEN_W] * 50
    for i in range(23, 50):
        w[i] = MUG_W
    b = buffer_from([make_episode(w, False), make_episode(success_widths(), True)])
    b.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    assert b.episodes[0].train_chunk_limit is None
    assert b.stats()["n_post_reopen_episodes_cut"] == 0
    print("  PASS: a failure that never reopens is kept whole")


def test_dead_group_still_cut_but_contributes_nothing():
    """An all-fail group is dead; truncation must not resurrect it."""
    b = buffer_from([make_episode(failure_widths(), False) for _ in range(4)])
    adv = b.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    # The cut DOES happen — assert it, or every check below would pass equally
    # well with the filter inert.
    assert [ep.train_chunk_limit for ep in b.episodes] == [24] * 4
    assert b.stats()["n_post_reopen_chunks_dropped"] == 4 * 26
    assert np.all(adv == 0.0)
    assert b.stats()["n_dead_groups"] == 1
    assert b.stats()["n_signal_chunks"] == 0
    assert all(c.advantage == 0.0 for c in b._build_chunks())
    print("  PASS: an all-fail group stays dead after truncation")


def test_anchor_interaction():
    """Anchors are all-success so are never cut, but the row BUDGET shrinks."""
    def mk():
        # Group 0: all-success -> anchor. Group 1: mixed -> signal.
        return buffer_from(
            [make_episode(success_widths(), True, group_id=0) for _ in range(2)]
            + [make_episode(success_widths(), True, group_id=1),
               make_episode(failure_widths(), False, group_id=1)]
        )

    # A budget with room for both anchors: truncation touches only the failure.
    b = mk()
    b.compute_advantages(
        anchor_advantage=0.2, include_anchor_groups=True, anchor_max_row_frac=2.0,
        post_reopen_filter=PostReopenFilter(3),
    )
    assert all(ep.train_chunk_limit is None for ep in b.episodes if ep.success)
    assert b.episodes[3].train_chunk_limit == 24
    s = b.stats()
    # Signal chunks are counted POST-truncation: 42 (success) + 24 (cut failure).
    assert s["n_signal_chunks"] == 42 + 24, s["n_signal_chunks"]
    assert s["n_anchor_chunks"] == 2 * 42, s["n_anchor_chunks"]
    assert s["n_anchor_groups"] == 1 and s["n_anchor_episodes"] == 2

    # The budget's DENOMINATOR is the truncated signal count, so the same
    # anchor_max_row_frac admits strictly fewer anchor rows than it used to.
    # At the default frac=1.0: 84 anchor chunks against 92 signal chunks
    # uncut (both fit) vs 66 cut (only one fits).
    uncut = mk()
    uncut.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True)
    assert uncut.stats()["n_signal_chunks"] == 42 + 50
    assert uncut.stats()["n_anchor_episodes"] == 2
    assert uncut.stats()["n_anchor_episodes_dropped"] == 0

    cut = mk()
    cut.compute_advantages(
        anchor_advantage=0.2, include_anchor_groups=True,
        post_reopen_filter=PostReopenFilter(3),
    )
    assert cut.stats()["n_signal_chunks"] == 42 + 24
    assert cut.stats()["n_anchor_episodes"] == 1
    assert cut.stats()["n_anchor_episodes_dropped"] == 1
    print("  PASS: anchors uncut; the row budget sees truncated signal counts")


def test_detector_error_policy():
    """A GLOBAL misconfiguration raises; a SINGLE bad episode never does.

    The two are told apart by probing every episode with non-empty states,
    successes included — a wrong state key or mirrored sign breaks successes
    exactly as it breaks failures. An earlier version counted errors against the
    FAILING episodes only, which aborted the run whenever one bad episode
    happened to be the iteration's sole failure; that regression is pinned here.
    """
    def strip(ep):
        for st in ep.states:
            st.pop("gripper_qpos")
            st["base_position"] = np.zeros(3, dtype=np.float32)
        return ep

    # (a) EVERY episode unreadable -> RuntimeError naming the probe count.
    b = buffer_from([strip(make_episode(failure_widths(), False)),
                     strip(make_episode(success_widths(), True))])
    expect_raises(
        RuntimeError, "could not read the gripper state of a single one of the 2",
        lambda: b.compute_advantages(post_reopen_filter=PostReopenFilter(3)),
        "global misconfiguration",
    )
    # ...and the same buffer is fine with the feature off.
    b.compute_advantages()
    assert len(b._build_chunks()) == b.num_chunks

    # (b) THE REGRESSION: one bad episode that is the ONLY failure. Must warn,
    # not abort — this gets more likely as the policy improves.
    solo = make_episode(failure_widths(), False)
    solo.states[7] = {"gripper_qpos": np.array([[np.nan, 0.0]])}
    b1 = buffer_from([solo, make_episode(success_widths(), True)])
    b1.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    assert b1.stats()["n_post_reopen_errors"] == 1
    assert solo.train_chunk_limit is None, "the bad episode is kept whole"
    assert len(b1._build_chunks()) == b1.num_chunks

    # (c) THE DUAL REGRESSION: a zero-chunk failing episode must not vouch for a
    # buffer nothing else can read. It short-circuits before state_key is even
    # touched, so it is excluded from the probe denominator.
    zero = make_episode([], False, n_chunks=0)
    assert zero.states == [] and zero.num_chunks == 0
    b2 = buffer_from([zero] + [strip(make_episode(failure_widths(), False))
                               for _ in range(3)])
    expect_raises(
        RuntimeError, "could not read the gripper state of a single one of the 3",
        lambda: b2.compute_advantages(post_reopen_filter=PostReopenFilter(3)),
        "zero-chunk episode must not mask a misconfiguration",
    )

    # (d) One of three broken -> warn, keep that one whole, cut the others, and
    # every invariant survives the partial failure.
    eps = [strip(make_episode(failure_widths(), False)),
           make_episode(failure_widths(), False),
           make_episode(failure_widths(), False),
           make_episode(success_widths(), True)]
    b3 = buffer_from(eps)
    adv = b3.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    st = b3.stats()
    assert st["n_post_reopen_errors"] == 1
    assert st["n_post_reopen_detected"] == 2
    assert st["n_post_reopen_episodes_cut"] == 2
    assert b3.episodes[0].train_chunk_limit is None
    assert b3.episodes[1].train_chunk_limit == 24
    chunks = b3._build_chunks()
    for i in range(len(eps)):
        got = sum(c.advantage for c in chunks if c.episode_idx == i)
        assert abs(got - adv[i]) < 1e-9, i
    assert abs(sum(c.advantage for c in chunks)) < 1e-9
    print("  PASS: misconfig raises; a lone bad episode and a zero-chunk one do not")


def test_inert_filter_is_visible_in_the_log():
    """A keep_chunks larger than any episode cuts nothing — and BOTH counters
    look healthy, so the summary line must print anyway."""
    import contextlib
    import io
    b = buffer_from([make_episode(failure_widths(), False) for _ in range(3)]
                    + [make_episode(success_widths(), True)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        b.compute_advantages(post_reopen_filter=PostReopenFilter(10**6))
    s = b.stats()
    assert s["n_post_reopen_detected"] == 3, "a perfect hit rate..."
    assert s["n_post_reopen_episodes_cut"] == 0, "...while nothing is cut"
    assert s["num_train_chunks"] == s["num_chunks"]
    text = out.getvalue()
    assert "Post-reopen truncation" in text, "an inert filter must still log"
    assert "cut 0" in text, f"the cut count must be visible: {text!r}"
    print("  PASS: an inert filter still prints, with cut 0 visible")


def test_detected_vs_cut_counters():
    """`detected` is the hit rate; `episodes_cut` is not.

    An episode whose onset lands within keep_chunks of the end is DETECTED and
    correctly left whole. Conflating the two makes a policy that holds its grasp
    longer look like a detector failure.
    """
    early = make_episode(failure_widths(n=50, close_at=13, onset_at=21), False)
    late = make_episode(failure_widths(n=50, close_at=40, onset_at=48), False)
    b = buffer_from([early, late, make_episode(success_widths(), True)])
    b.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    s = b.stats()
    assert s["n_post_reopen_detected"] == 2, "both reopens WERE found"
    assert s["n_post_reopen_episodes_cut"] == 1, "only the early one is cut"
    assert late.train_chunk_limit is None
    assert s["n_post_reopen_errors"] == 0
    # An episode that never closes is a genuine miss: neither detected nor cut.
    b2 = buffer_from([make_episode([OPEN_W] * 50, False),
                      make_episode(success_widths(), True)])
    b2.compute_advantages(post_reopen_filter=PostReopenFilter(3))
    assert b2.stats()["n_post_reopen_detected"] == 0
    print("  PASS: detected vs cut vs miss are three distinct counters")


# ---------------------------------------------------------------------------
# 3. GRPOConfig wiring
# ---------------------------------------------------------------------------

def test_config():
    from grpo_config import GRPOConfig

    base = dict(env_names=["robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env"])
    assert GRPOConfig(**base).build_post_reopen_filter() is None, "default is OFF"

    f = GRPOConfig(**base, post_reopen_keep_chunks=4).build_post_reopen_filter()
    assert f == PostReopenFilter(4, 0.035, 0.055, 0.004, 3, 5, "gripper_qpos")
    f = GRPOConfig(
        **base, post_reopen_keep_chunks=0,
        post_reopen_close_width=0.03, post_reopen_open_width=0.06,
        post_reopen_onset_margin=0.005, post_reopen_min_closed_chunks=2,
        post_reopen_min_train_chunks=0, post_reopen_state_key="fingers",
    ).build_post_reopen_filter()
    assert f == PostReopenFilter(0, 0.03, 0.06, 0.005, 2, 0, "fingers")

    # A live rebuild: mutating the config between iterations must be reflected
    # (toy_train_grpo.py mutates config fields, so a cached object would stale).
    c = GRPOConfig(**base, post_reopen_keep_chunks=3)
    c.post_reopen_keep_chunks = 8
    assert c.build_post_reopen_filter().keep_chunks == 8

    # Validation fires at CONFIG construction, not at the first collection.
    cases = [
        (dict(post_reopen_keep_chunks=-1), "must be >= 0"),
        (dict(post_reopen_keep_chunks=3, post_reopen_close_width=0.09), "must be <"),
        (dict(post_reopen_keep_chunks=3, post_reopen_open_width=-1.0), "must be > 0"),
        (dict(post_reopen_keep_chunks=3, post_reopen_onset_margin=0.0), "must be > 0"),
        (dict(post_reopen_keep_chunks=3, post_reopen_onset_margin=0.1),
         "collapse onto the crossing"),
        # The four tuning knobs are inert with the feature off -> hard error.
        (dict(post_reopen_close_width=0.03), "post_reopen_keep_chunks is None"),
        (dict(post_reopen_open_width=0.06), "post_reopen_keep_chunks is None"),
        (dict(post_reopen_onset_margin=0.005), "post_reopen_keep_chunks is None"),
        (dict(post_reopen_min_closed_chunks=2), "post_reopen_keep_chunks is None"),
        (dict(post_reopen_min_train_chunks=9), "post_reopen_keep_chunks is None"),
        (dict(post_reopen_keep_chunks=3, post_reopen_min_closed_chunks=0),
         "must be >= 1"),
        (dict(post_reopen_keep_chunks=3, post_reopen_min_train_chunks=-1),
         "must be >= 0"),
        (dict(post_reopen_state_key="fingers"), "post_reopen_keep_chunks is None"),
    ]
    for kwargs, match in cases:
        expect_raises(
            ValueError, match, lambda k=kwargs: GRPOConfig(**base, **k),
            f"GRPOConfig({kwargs})",
        )
    print(f"  PASS: GRPOConfig wiring + validation ({len(cases)} rejections)")


# ---------------------------------------------------------------------------
# 3b. TB / wandb emission in train_grpo.py
# ---------------------------------------------------------------------------
# Nothing else in this suite imports train_grpo, so every one of the feature's
# new lines there — the six gated episode/n_post_reopen_* scalars, the ungated
# episode/num_chunks, and the README's promise that "an unfiltered run's
# episode/* key set is unchanged" — was unprotected by construction. Driven
# through the REAL _log_metrics on a `__new__`-built trainer, the technique
# test_scene_seed_pool.py and test_anchor_groups.py already use to run it on CPU.


class _RecordingWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, float(value), step))

    def add_text(self, *a, **kw):
        pass


def _emit_metrics(cfg, stats, iteration=4):
    import train_grpo as tg
    tr = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    tr.config = cfg
    tr.iteration = iteration
    tr.writer = _RecordingWriter()
    tr._ref_mse_stats = None
    tr._chunk_gap_stats = None
    tg.GRPOTrainer._log_metrics(tr, iteration, stats, update_stats=None,
                                lr=1e-5, iter_time=1.0)
    return {t: v for t, v, _ in tr.writer.calls}


POST_REOPEN_TAGS = (
    "episode/n_post_reopen_episodes_cut",
    "episode/n_post_reopen_detected",
    "episode/n_post_reopen_errors",
    "episode/n_post_reopen_implausible",
    "episode/n_post_reopen_chunks_dropped",
    "episode/num_train_chunks",
    "episode/post_reopen_kept_len_min",
    "episode/post_reopen_kept_len_median",
    "episode/post_reopen_kept_len_max",
)


def test_tb_emission():
    from grpo_config import GRPOConfig
    env = ["robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env"]

    def stats_for(N):
        b = buffer_from([make_episode(failure_widths(onset_at=o), False)
                         for o in (18, 21, 24)]
                        + [make_episode(success_widths(), True)])
        cfg = GRPOConfig(env_names=env, post_reopen_keep_chunks=N)
        b.compute_advantages(post_reopen_filter=cfg.build_post_reopen_filter())
        return cfg, b.stats()

    # OFF: not one post-reopen tag may appear — the key-set promise.
    cfg_off, st_off = stats_for(None)
    off = _emit_metrics(cfg_off, st_off)
    for tag in POST_REOPEN_TAGS:
        assert tag not in off, f"{tag} leaked into an unfiltered run"
    # ...but the raw collected count is ungated, so it IS there, and is what a
    # filtered run's num_train_chunks gets compared against.
    assert off["episode/num_chunks"] == 3 * 50 + 42

    # ON: every tag present, with the right values.
    cfg_on, st_on = stats_for(3)
    on = _emit_metrics(cfg_on, st_on)
    for tag in POST_REOPEN_TAGS:
        assert tag in on, f"{tag} missing from a filtered run"
    assert on["episode/num_chunks"] == 3 * 50 + 42
    assert on["episode/num_train_chunks"] == (21 + 24 + 27) + 42
    assert on["episode/n_post_reopen_detected"] == 3
    assert on["episode/n_post_reopen_episodes_cut"] == 3
    assert on["episode/n_post_reopen_errors"] == 0
    assert on["episode/n_post_reopen_implausible"] == 0
    assert on["episode/n_post_reopen_chunks_dropped"] == (29 + 26 + 23)
    assert on["episode/post_reopen_kept_len_min"] == 21
    assert on["episode/post_reopen_kept_len_median"] == 24
    assert on["episode/post_reopen_kept_len_max"] == 27

    # Every OFF tag must still be present ON (no key removed by the feature).
    assert set(off) <= set(on), set(off) - set(on)

    # An empty stats dict must not emit the episode/* block at all (collection
    # failed entirely; .get(..., 0) defaults would look like a real all-fail).
    empty = _emit_metrics(cfg_on, {})
    assert not any(t.startswith("episode/") for t in empty), empty
    print(f"  PASS: {len(POST_REOPEN_TAGS)} tags gated on the feature, "
          f"episode/num_chunks ungated, values correct, empty-stats block skipped")


def test_wandb_key_set_matches_tb():
    """The wandb pop-list must drop exactly the keys the TB side gates."""
    import train_grpo as tg
    import inspect
    src = inspect.getsource(tg.GRPOTrainer._log_metrics)
    # Every stats key the feature adds...
    added = {
        "n_post_reopen_episodes_cut", "n_post_reopen_detected",
        "n_post_reopen_errors", "n_post_reopen_implausible",
        "n_post_reopen_chunks_dropped", "num_train_chunks",
        "post_reopen_kept_len_min", "post_reopen_kept_len_median",
        "post_reopen_kept_len_max",
    }
    b = buffer_from([make_episode(failure_widths(), False),
                     make_episode(success_widths(), True)])
    b.compute_advantages()
    assert added <= set(b.stats()), added - set(b.stats())
    # ...must appear in the pop-list, so an anchors-off/filter-off wandb run has
    # the same key set as before the feature. `num_chunks` must NOT (it predates
    # the feature and is now emitted to TB unconditionally too).
    pop_block = src[src.index('if self.config.post_reopen_keep_chunks is None:'):]
    pop_block = pop_block[:pop_block.index("per_scene_success")]
    for k in added:
        assert f'"{k}"' in pop_block, f"{k} is not popped from wandb when off"
    assert '"num_chunks"' not in pop_block.replace('"num_train_chunks"', "")
    print(f"  PASS: all {len(added)} feature keys popped from wandb when off; "
          f"num_chunks correctly not popped")


# ---------------------------------------------------------------------------
# 4. Real-data regression: the N -> dropped-fraction curve
# ---------------------------------------------------------------------------

def _real_widths():
    """Gripper widths for every real episode, in sorted filename order."""
    out = []
    for path in sorted(REAL_NPZ_DIR.glob("episode_*.npz")):
        d = np.load(path, allow_pickle=True)
        nc = int(d["num_chunks"])
        out.append(gripper_widths(
            [{"gripper_qpos": d[f"state_gripper_qpos_{i}"]} for i in range(nc)]
        ))
    return out


def test_monotonicity_guard_is_free_on_real_data():
    """Ablate the guard against the real collection — the honest version.

    `test_real_npz_dir_matches_fixture` cannot establish this: it runs the
    shipped function (which INCLUDES the guard) against a fixture measured WITH
    the guard. This compares the shipped function to `_onset_no_monotonicity`
    on the same widths.
    """
    if not REAL_NPZ_DIR.is_dir():
        print(f"  SKIP: {REAL_NPZ_DIR} not present")
        return
    widths = _real_widths()
    for m in (0.0025, 0.003, 0.004, 0.005, 0.0055):
        shipped = [reopen_onset_index(w, 0.035, 0.055, m) for w in widths]
        ablated = [_onset_no_monotonicity(w, 0.035, 0.055, m) for w in widths]
        assert shipped == ablated, f"guard changed an onset at margin {m}"
    # BELOW the plateau it is not free. Pinned at both margins so the "free"
    # claim stays scoped to the plateau rather than drifting into "always".
    for m, n_expected in ((0.002, 1), (0.0015, 3), (0.001, 3)):
        shipped = [reopen_onset_index(w, 0.035, 0.055, m) for w in widths]
        ablated = [_onset_no_monotonicity(w, 0.035, 0.055, m) for w in widths]
        assert sum(a != b for a, b in zip(shipped, ablated)) == n_expected, m
    print("  PASS: guard changes 0 onsets inside the plateau, 1-3 below it")


def _curve_from(rows):
    """(dropped, fail_total, all_total) per N, via the SHIPPED limit rule.

    Drives `post_reopen_chunk_limit` on a synthetic trace built to reproduce
    each fixture row's (num_chunks, onset), rather than re-implementing the
    `None if limit >= len` rule — a re-implementation would keep agreeing with
    itself if the shipped rule changed.
    """
    fail_total = sum(nc for s, nc, _ in rows if not s)
    all_total = sum(nc for _, nc, _ in rows)
    out = {}
    for n in (0, 3, 4, 5, 6, 8, 10, 20, 30):
        dropped = 0
        for succ, nc, onset in rows:
            if succ or onset is None:
                continue
            st = states_from_widths(failure_widths(n=nc, close_at=max(onset - 8, 1),
                                                   onset_at=onset))
            assert reopen_onset_index(gripper_widths(st)) == onset
            limit = post_reopen_chunk_limit(st, PostReopenFilter(n))
            if limit is not None:
                dropped += nc - limit
        out[n] = (dropped, fail_total, all_total)
    return out


def test_iter_0001_curve():
    curve = _curve_from(ITER_0001)
    assert curve[0][1] == 2150 and curve[0][2] == 2321
    # Every N quoted in gripper_release.py, grpo_config.py and the README.
    expected = {0: 1251, 3: 1123, 4: 1081, 5: 1039, 6: 997,
                8: 913, 10: 829, 20: 409, 30: 45}
    for n, want in expected.items():
        assert curve[n][0] == want, (n, curve[n][0], want)
    # The shipped recommendation: N=3 drops ~52% of failure chunks.
    d, ft, at = curve[3]
    assert abs(100 * d / ft - 52.2) < 0.1, 100 * d / ft
    assert abs(100 * d / at - 48.4) < 0.1, 100 * d / at
    # N=4, the documented alternative framing.
    d4 = curve[4][0]
    assert abs(100 * d4 / ft - 50.3) < 0.1 and abs(100 * d4 / at - 46.6) < 0.1
    # 43/43 failures detected — the property the filter's usefulness rests on.
    assert sum(1 for s, _, o in ITER_0001 if not s and o is None) == 0
    # The onset really is EARLIER than the crossing (0-2 chunks, 1 on 36 of 43).
    lags = [c - o for (s, _, o), c in zip(ITER_0001, ITER_0001_CROSS) if not s]
    assert min(lags) == 0 and max(lags) == 2, (min(lags), max(lags))
    assert lags.count(1) == 36, lags.count(1)
    # Onset range: 15-27 for 42 of 43; episode_0041 at 48 is the exception.
    onsets = sorted(o for s, _, o in ITER_0001 if not s)
    assert onsets[0] == 15 and onsets[-1] == 48 and onsets[-2] == 27
    assert int(np.median(onsets)) == 20
    print("  PASS: iter_0001 N -> dropped-fraction curve (N=3: 52.2% / 48.4%)")


def test_real_npz_dir_matches_fixture():
    """Re-derive ITER_0001 from the raw .npz when that collection is present."""
    if not REAL_NPZ_DIR.is_dir():
        print(f"  SKIP: {REAL_NPZ_DIR} not present (fixture-only run)")
        return
    cfg = PostReopenFilter(4)
    rows, cross = [], []
    for path in sorted(REAL_NPZ_DIR.glob("episode_*.npz")):
        d = np.load(path, allow_pickle=True)
        nc = int(d["num_chunks"])
        states = [{"gripper_qpos": d[f"state_gripper_qpos_{i}"]} for i in range(nc)]
        w = gripper_widths(states, cfg.state_key)
        rows.append((
            int(bool(d["success"])), nc,
            reopen_onset_index(w, cfg.close_below, cfg.open_above, cfg.onset_margin),
        ))
        found = close_cross_indices(w, cfg.close_below, cfg.open_above)
        cross.append(None if found is None else found[1])
    assert rows == ITER_0001, (
        f"fixture drift: recomputed {len(rows)} rows from {REAL_NPZ_DIR} that "
        f"do not match ITER_0001"
    )
    assert cross == ITER_0001_CROSS, "crossing fixture drift"

    # The onset is invariant over the whole threshold box on REAL data — this
    # is the property that makes the two width knobs low-stakes, and the
    # synthetic version of the check cannot establish it.
    widths = []
    successes = []
    for path in sorted(REAL_NPZ_DIR.glob("episode_*.npz")):
        d = np.load(path, allow_pickle=True)
        nc = int(d["num_chunks"])
        widths.append(gripper_widths(
            [{"gripper_qpos": d[f"state_gripper_qpos_{i}"]} for i in range(nc)]
        ))
        successes.append(bool(d["success"]))
    base = [o for _, _, o in ITER_0001]
    n_pairs = 0
    for cb in (0.030, 0.035, 0.040, 0.045, 0.050):
        for oa in (0.050, 0.055, 0.060, 0.070):
            if cb >= oa:
                continue
            n_pairs += 1
            got = [reopen_onset_index(w, cb, oa, 0.004) for w in widths]
            assert got == base, f"onset moved at close<{cb} open>{oa}"

    # Onset-margin plateau. Two different widths, and the distinction matters:
    # only FAILING episodes are ever truncated, so the failures-only plateau is
    # the one the feature's behavior depends on. It is [0.0025, 0.0055]; the
    # all-48 plateau is the narrower [0.004, 0.005] because one SUCCESS
    # (episode_0011) holds the mug with 0.019 -> 0.023 -> 0.025 wobble that
    # straddles the margin. The shipped 0.004 sits inside both.
    def _moved(m):
        got = [reopen_onset_index(w, 0.035, 0.055, m) for w in widths]
        return [i for i in range(len(got)) if got[i] != base[i]]

    for m in (0.0025, 0.003, 0.004, 0.005, 0.0055):
        moved = [i for i in _moved(m) if not successes[i]]
        assert not moved, f"a FAILURE's onset moved at margin {m}: {moved}"
    for m in (0.004, 0.005):
        assert not _moved(m), f"some onset moved at margin {m}"
    # Pin the two edges so a change in either direction is caught, not absorbed.
    assert _moved(0.003) == [11] and successes[11], \
        "expected exactly episode_0011 (a success) to move at margin 0.003"
    assert len([i for i in _moved(0.006) if not successes[i]]) == 4, \
        "expected 4 failures to move at margin 0.006"
    assert len([i for i in _moved(0.002) if not successes[i]]) == 1
    print(
        f"  PASS: {len(rows)} real episodes reproduce the fixture exactly; the "
        f"onset is invariant over {n_pairs} threshold pairs, and over the "
        f"[0.0025, 0.0055] margin plateau on every failure"
    )


if __name__ == "__main__":
    print("=== post-reopen truncation ===\n")
    print("Detector primitives:")
    test_width_extraction()
    test_close_cross_hysteresis()
    test_close_requires_an_observed_open_first()
    test_close_requires_a_dwell()
    test_min_train_chunks_floor()
    test_open_above_guard_boundary_matches_the_crossing()
    test_onset_threshold_boundary_is_strict()
    test_reopen_onset()
    test_onset_ignores_mid_hold_blips()
    test_backward_walk_beats_a_forward_scan()
    test_floor_is_the_closed_phase_minimum()
    test_onset_monotonicity_guard()
    test_onset_is_threshold_invariant()
    test_chunk_limit_semantics()
    test_sign_convention_guard()
    test_filter_validation()
    print("\nEpisodeBuffer wiring:")
    test_num_train_chunks_clamps()
    test_zero_chunk_episode_does_not_divide_by_zero()
    test_off_switch_is_bit_identical()
    test_failures_cut_successes_whole()
    test_zero_sum_invariant_preserved()
    test_idempotent_and_self_clearing()
    test_never_reopened_failure_kept_whole()
    test_dead_group_still_cut_but_contributes_nothing()
    test_anchor_interaction()
    test_detector_error_policy()
    test_inert_filter_is_visible_in_the_log()
    test_detected_vs_cut_counters()
    test_summary_log_line_is_correct_not_merely_present()
    test_first_error_is_the_first_one()
    print("\nGRPOConfig:")
    test_config()
    test_defaults_agree_across_modules()
    print("\nTB / wandb emission:")
    test_tb_emission()
    test_wandb_key_set_matches_tb()
    print("\nReal-data regression:")
    test_iter_0001_curve()
    test_monotonicity_guard_is_free_on_real_data()
    test_real_npz_dir_matches_fixture()
    print("\nAll tests PASSED.")
