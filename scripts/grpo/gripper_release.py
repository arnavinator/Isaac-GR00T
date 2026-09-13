"""Post-reopen chunk truncation for FAILED grasp-and-place episodes.

Self-contained primitives, model-free and MuJoCo-free, so every piece is
testable on CPU without a DiT or a simulator. ``episode_buffer`` is the only
consumer; it calls :func:`post_reopen_detect` once per episode and stores the
limit on ``GRPOEpisode.train_chunk_limit``.

The problem
-----------
On ``CoffeeServeMug_PandaOmron_Env`` a failed episode has a stereotyped shape:
the policy approaches, closes the gripper (usually on **nothing**), reopens it a
few chunks later, and then flies the arm away and meanders until truncation.
Measured on ``iter_0001`` (48 episodes, 43 failures, 4 groups x 12):

* 43/43 failures contain exactly one detected close->reopen; the reopen onset
  lands at chunk 15-27 (median 20) of a 50-chunk episode for 42 of the 43.
  ``episode_0041`` is the exception at 48 -- it grasps the mug and never places
  it, so there is no meander and nothing to trim.
* **1/43** ever re-commands a close after the reopen (``episode_0003``, and only
  at +23 chunks). There is essentially no recovery attempt to preserve.
* EEF motion produced BY THE ACTION of chunk ``onset+k`` is 0.021 / 0.017 /
  0.029 m for k = 0, 1, 2, then **0.082 at k = 3** and 0.096 / 0.097 at k = 4, 5.
  The share of failures whose chunk moves more than 5 cm jumps from 21% at k = 2
  to **74% at k = 3**. For reference the approach phase -- everything strictly
  before the close -- runs at a median 0.005 m/chunk.
* The OBSERVATION at the start of chunk ``onset+k`` is 0.000 / 0.021 / 0.031 /
  0.045 m from the onset for k = 0..3, then 0.104 (k=4), 0.289 (k=8), plateauing
  near 0.7 m. So chunk onset+3 is still observed at the grasp site, but its
  action is the one that leaves.

So roughly **half of every failing episode is ballistic retreat and parking.**
Under ``A_ep / num_chunks`` that half carries half of the episode's negative
gradient weight, spent on behaviour that is a *consequence* of the failure
rather than its cause -- and that no successful episode in the group exhibits
at all (successes terminate on the place). Truncating it concentrates the
negative credit on the approach and the failed grasp.

Credit in a policy-gradient update attaches to a chunk's ACTION, so the two
profiles above disagree about where to cut by exactly one chunk and the action
one decides it: ``keep_chunks = 3`` retains the three chunks whose actions are
still local and drops from the first ballistic command.

The detector
------------
Everything runs on the measured gripper width ``w = qpos[0] - qpos[1]``, NOT on
the commanded ``action.gripper_close``. Three reasons: the measurement is what
physically happened; it covers a close on the mug and a close on nothing
identically, without needing to tell them apart; and it is unaffected by a
policy that emits a noisy gripper channel -- on this data 113 chunks command a
gripper value that is not even constant across their own executed substeps, so
"the chunk the command flipped in" is not as crisp a quantity as it sounds.

Three stages, in order:

1. **Close** (:func:`close_cross_indices`) -- a two-threshold hysteresis state
   machine. CLOSED is entered at the first width below ``close_below`` and left
   at the first width above ``open_above``.
2. **Crossing** -- that exit index. This is where the gripper is unambiguously
   open again, but it is 0-2 chunks LATE: the fingers take about a control chunk
   to travel, so by the time the width clears ``open_above`` the reopen is
   already underway.
3. **Onset** (:func:`reopen_onset_index`) -- walk BACKWARD from the crossing
   down the contiguous rising edge, while the width is both above
   ``floor + onset_margin`` (``floor`` = the minimum width during the closed
   phase) and strictly below its successor. The first chunk of that edge is the
   onset: the first observation in which the gripper has measurably begun to
   open. This is what ``keep_chunks`` is timed from.

Why walk backward rather than scan forward for the first rise: the closed phase
contains isolated upward blips (``episode_0039`` reads 0.007 mid-hold between
0.001 samples; ``episode_0044`` reads 0.007 then falls back to 0.003). A forward
scan fires on those. Anchoring at a confirmed crossing and walking back means
only the rising edge that actually reaches the open state is ever traversed.

Why the ``w[o-1] < w[o]`` term as well as the threshold: without it, a closed
phase whose floor is set EARLY and whose later hold sits above
``floor + onset_margin`` -- fingers briefly overshoot to 0.001, then the object
settles between them at 0.020, then release -- is one contiguous
above-threshold run, and the walk reports the start of the hold as the onset.
Within the margin plateau below, the term changes no onset on any of the 48 real
episodes; below the plateau it does (1 onset at margin 0.002, 3 at 0.0015 and
under), so "free" is a statement about the plateau, not about the term.

The cost of that term: a **stepped release** -- open partway, hold, then open
fully -- is indistinguishable in the width trace from "hold the object, then
release", so the walk stops at the top step and reports the onset LATE by the
plateau length. Two physically identical traces that differ only by measurement
noise on the plateau can therefore differ by several chunks. This is the
conservative direction (a late onset keeps MORE chunks, i.e. under-truncates),
and no rule on the width alone can separate the two shapes, so it is accepted
rather than fixed.

Calibration
-----------
Steady-state widths over ``iter_0001`` are cleanly trimodal:

===================  ==========================================
~0.000 - 0.004 m     closed on nothing (missed grasp)
~0.017 - 0.025 m     closed on the mug (real grasp)
~0.055 - 0.080 m     open (0.080 = fully open)
===================  ==========================================

94.7% of all 2321 chunks fall in those three bands; the other 5.3% are
transitions. Of the 26 chunks in [0.028, 0.042) -- the region that could trouble
``close_below`` -- every single one is a transient: the close chunk, the chunk
before it, or onset-1/onset/onset+1. So the hysteresis machine is unaffected,
but do not read the bands as covering every sample.

``close_below=0.035`` sits between the widest real grasp (~0.025) and the
narrowest approach-phase squeeze observed while the gripper was still commanded
open (0.0424, ``episode_0040``, taking "commanded open" to mean
``gripper_close[0] <= 0.5`` on the first EXECUTED substep; under the stricter
"no close in any of the 8 executed substeps" reading the narrowest is 0.0551, so
the usable valley is at least that wide). ``open_above=0.055`` sits below the
open cluster.

``onset_margin=0.004`` is the centre of a measured plateau. Over the 43
FAILURES -- the only episodes this filter ever truncates -- every value in
**[0.0025, 0.0055]** yields bit-identical onsets. Below that the walk starts
following a blip into the edge (``episode_0044``, one chunk early at 0.002);
above it, 4 failures stop one chunk short on the shallow part of the edge. Over
all 48 episodes the plateau is the narrower [0.004, 0.005], because one SUCCESS
(``episode_0011``) holds the mug with 0.019 -> 0.023 -> 0.025 wobble that
straddles the margin; that episode is never truncated, so it does not narrow the
usable range, but it is why the default sits at 0.004 rather than 0.003.

The onset is markedly more threshold-robust than the crossing it is derived
from. Over ``close_below`` in [0.030, 0.050] x ``open_above`` in [0.050, 0.070]
(19 pairs), the crossing index moves on 14 of the 19 while the onset is
**identical on all 48 episodes** -- it is anchored to the base of the rising
edge, which is the steepest and least threshold-sensitive part of the signal.

Silent no-op paths
------------------
Three shapes produce ``None`` (no truncation) with no error, because none of
them is distinguishable from a legitimately un-trimmable episode:

* a **gentler close** that never dips below ``close_below``;
* a **partial release** that never rises above ``open_above`` after closing;
* a close still held at the last chunk (the real ``episode_0041``).

The first two mean the detector has silently stopped working. The guard below
does not catch either -- it only checks that the gripper is observed open at
SOME point, which the approach phase satisfies. Watch
``episode/n_post_reopen_detected`` against the iteration's failure count; that
ratio is the hit rate and is the only thing that reveals them.
``episode/n_post_reopen_episodes_cut`` is NOT a hit rate -- an episode whose
onset lands within ``keep_chunks`` of the end is detected and correctly left
whole -- and neither counter catches a ``keep_chunks`` so large that nothing is
ever cut, which is why ``_apply_post_reopen_filter`` prints its summary
unconditionally.

Scope
-----
Calibrated for the PandaOmron parallel-jaw gripper (0.08 m open stroke) on a
grasp-and-place task. The mechanism generalizes -- any task whose failure mode
is "release early, then wander" -- but the three widths do not; re-derive them
from a width histogram before using this on another embodiment.

The ``max(width) > open_above`` guard below catches a MIRRORED sign convention
(which would otherwise pin the state machine in CLOSED from chunk 0 and disable
the feature silently) and a units mismatch. It does NOT catch a wrong
``state_key`` whose values happen to span a similar range: ``gripper_qvel`` is a
real ``(1, 2)`` key in these same ``.npz`` files and pointing the filter at it
returns a plausible, wrong answer with no error.
"""

import numbers
from dataclasses import dataclass

import numpy as np

# Modality key holding the parallel-jaw finger positions. For PandaOmron the
# stored value is (state_horizon, 2) with horizon 1.
GRIPPER_STATE_KEY = "gripper_qpos"

# Defaults for PandaOmron; see the module docstring for the calibration.
DEFAULT_CLOSE_BELOW = 0.035
DEFAULT_OPEN_ABOVE = 0.055
DEFAULT_ONSET_MARGIN = 0.004
DEFAULT_MIN_CLOSED_CHUNKS = 3
DEFAULT_MIN_TRAIN_CHUNKS = 5


@dataclass(frozen=True)
class PostReopenFilter:
    """Configuration for the post-reopen truncation.

    Frozen and self-validating so the same object can be built once from
    ``GRPOConfig`` and handed to ``compute_advantages`` every iteration without
    re-checking anything. ``None`` in place of an instance disables the feature
    entirely (and is bit-identical to the pre-feature behaviour).
    """

    # Chunks retained starting AT the reopen ONSET -- the first chunk in which
    # the gripper has measurably begun to open, not the later chunk in which the
    # width clears `open_above`. `keep_chunks=4` keeps onset, onset+1, onset+2,
    # onset+3 and drops everything from onset+4 to the end; `keep_chunks=0`
    # drops the onset chunk itself. Everything BEFORE the onset -- the whole
    # approach and the closed phase -- is always kept.
    keep_chunks: int
    close_below: float = DEFAULT_CLOSE_BELOW
    open_above: float = DEFAULT_OPEN_ABOVE
    # How far above the closed-phase floor the width must rise before a chunk
    # counts as "already opening". Measured plateau over the failures this
    # filter truncates: [0.0025, 0.0055]. See the module docstring.
    onset_margin: float = DEFAULT_ONSET_MARGIN
    # Consecutive sub-`close_below` chunks required before CLOSED latches. A
    # close is a DWELL, not a single sample: without this a lone transient dip
    # through the band latches the state machine and the very next open sample
    # is read as the reopen, truncating the episode to `keep_chunks + 1` chunks
    # and discarding the failed grasp entirely. Real closed phases run >= 7
    # chunks on the reference data and no episode there contains a
    # sub-threshold run shorter than 3, so the default is free.
    min_closed_chunks: int = DEFAULT_MIN_CLOSED_CHUNKS
    # Implausibility floor on the retained prefix. A limit below this keeps the
    # episode WHOLE and is counted + warned rather than applied: the approach
    # phase alone is ~13 chunks on the reference data, so a handful of retained
    # chunks means the detector latched onto something that is not the grasp.
    # Backstop for a detector failure not anticipated here — the two counters
    # cannot see one (`detected` reads a perfect hit rate in exactly that case).
    min_train_chunks: int = DEFAULT_MIN_TRAIN_CHUNKS
    state_key: str = GRIPPER_STATE_KEY

    def __post_init__(self):
        # numbers.Integral, not `int`: numpy integer scalars are the natural
        # thing to arrive here from a sweep or a loaded config, and they are
        # perfectly usable. `bool` is a subclass of int and is not.
        if isinstance(self.keep_chunks, bool) or not isinstance(
            self.keep_chunks, numbers.Integral
        ):
            raise ValueError(
                f"post_reopen_keep_chunks must be an int, got "
                f"{type(self.keep_chunks).__name__}"
            )
        if self.keep_chunks < 0:
            raise ValueError(
                f"post_reopen_keep_chunks must be >= 0, got {self.keep_chunks}. "
                f"0 drops the onset chunk itself; there is no value that means "
                f"'disabled' -- leave the knob at None for that."
            )
        for name, v in (
            ("post_reopen_min_closed_chunks", self.min_closed_chunks),
            ("post_reopen_min_train_chunks", self.min_train_chunks),
        ):
            if isinstance(v, bool) or not isinstance(v, numbers.Integral):
                raise ValueError(f"{name} must be an int, got {type(v).__name__}")
        if self.min_closed_chunks < 1:
            raise ValueError(
                f"post_reopen_min_closed_chunks must be >= 1, got "
                f"{self.min_closed_chunks}. 1 reproduces the single-sample latch "
                f"this knob exists to prevent; 2-3 is the useful range."
            )
        if self.min_train_chunks < 0:
            raise ValueError(
                f"post_reopen_min_train_chunks must be >= 0, got "
                f"{self.min_train_chunks}. 0 disables the implausibility floor."
            )
        for name, v in (
            ("post_reopen_close_width", self.close_below),
            ("post_reopen_open_width", self.open_above),
            ("post_reopen_onset_margin", self.onset_margin),
        ):
            if not isinstance(v, numbers.Real) or isinstance(v, bool):
                raise ValueError(
                    f"{name} must be a real number, got {type(v).__name__}"
                )
            if not np.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
            if v <= 0.0:
                raise ValueError(f"{name} must be > 0, got {v}")
        # Strict inequality, not >=: equal thresholds degenerate the state
        # machine into a single comparator, so a width sitting exactly on the
        # boundary would chatter open/closed on measurement noise and fire a
        # reopen mid-approach. The whole point of two thresholds is the gap.
        if self.close_below >= self.open_above:
            raise ValueError(
                f"post_reopen_close_width ({self.close_below}) must be < "
                f"post_reopen_open_width ({self.open_above}) -- the detector is "
                f"a hysteresis state machine and needs a gap between the two."
            )
        # A sanity bound, NOT a guarantee — be precise about which.
        #
        # Every width strictly inside the closed phase is below `open_above` by
        # construction (the crossing is the FIRST sample above it), so a chunk
        # can only clear `floor + margin` when `margin < open_above - floor`.
        # For a gripper whose floor lies in [0, close_below) — every physical
        # one — the value that holds for EVERY floor is
        # `margin < open_above - close_below`. At or above it the backward walk
        # provably cannot take a step on any trace, so the onset silently equals
        # the crossing and the feature reverts to the timing it exists to
        # replace with nothing in the logs saying so. Checking only
        # `margin < open_above` does not catch that: at the defaults it admits
        # 0.0549, which collapses all 43 measured failures onto the crossing.
        #
        # What this does NOT claim:
        #   - Not necessary. A trace whose floor is NEGATIVE (a mis-signed or
        #     mis-scaled width that slipped past the max() guard) can still walk
        #     at a margin this rejects.
        #   - Not sufficient. A value inside the bound can still collapse the
        #     onset onto the crossing on any particular episode — 0.0199 does so
        #     on 18 of the 43 measured failures. Only the empirical plateau in
        #     the module docstring speaks to a specific dataset.
        if self.onset_margin >= self.open_above - self.close_below:
            raise ValueError(
                f"post_reopen_onset_margin ({self.onset_margin}) must be < "
                f"post_reopen_open_width - post_reopen_close_width "
                f"({self.open_above - self.close_below:g}) -- at or above that "
                f"no chunk can ever clear floor + margin, so the onset would "
                f"silently collapse onto the crossing index."
            )
        if not isinstance(self.state_key, str) or not self.state_key:
            raise ValueError(
                f"post_reopen_state_key must be a non-empty string, got "
                f"{self.state_key!r}"
            )


def gripper_widths(
    states: "list[dict[str, np.ndarray]]",
    state_key: str = GRIPPER_STATE_KEY,
) -> np.ndarray:
    """Per-chunk gripper opening, in metres.

    ``states`` is ``GRPOEpisode.states`` -- one dict per chunk, holding the
    observation the policy was queried with. The width is ``qpos[0] - qpos[1]``
    of the MOST RECENT row when the stored value carries a state horizon > 1
    (PandaOmron stores horizon 1, so this is just the single row).

    Raises rather than returning a sentinel for a missing, wrong-shaped or
    non-finite value: a filter that quietly declines to filter is the failure
    mode this whole feature is most exposed to, and it is indistinguishable from
    a working one in every log line. A NaN in particular would make every
    comparison in the state machine False and disable detection outright.
    """
    widths = np.empty(len(states), dtype=np.float64)
    for i, state in enumerate(states):
        if state_key not in state:
            raise KeyError(
                f"post-reopen filter: chunk {i} has no '{state_key}' state key "
                f"(present: {sorted(state)}). The filter is calibrated for the "
                f"PandaOmron parallel-jaw gripper; disable it "
                f"(post_reopen_keep_chunks=None) for embodiments without one, or "
                f"point post_reopen_state_key at the right modality."
            )
        try:
            arr = np.asarray(state[state_key], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(
                f"post-reopen filter: '{state_key}' at chunk {i} is not numeric "
                f"({state[state_key]!r}): {e}"
            ) from e
        # ndim is pinned to 1 or 2, not just checked on the trailing axis: a
        # (1, 2, 2) value satisfies `shape[-1] == 2` and would be silently
        # reinterpreted by reshape(-1, 2) as two rows of a horizon. size is
        # checked because a (0, 2) value passes every shape test and then
        # IndexErrors on the [-1].
        if arr.ndim not in (1, 2) or arr.shape[-1] != 2 or arr.size == 0:
            raise ValueError(
                f"post-reopen filter: '{state_key}' at chunk {i} has shape "
                f"{arr.shape}; expected (2,) or (state_horizon, 2) — two finger "
                f"joints, optionally with a horizon."
            )
        row = arr.reshape(-1, 2)[-1]
        if not np.isfinite(row).all():
            raise ValueError(
                f"post-reopen filter: '{state_key}' at chunk {i} is not finite "
                f"({row}). Every comparison against a NaN is False, so this "
                f"would disable the detector rather than trip it."
            )
        widths[i] = row[0] - row[1]
    return widths


def close_cross_indices(
    widths: np.ndarray,
    close_below: float = DEFAULT_CLOSE_BELOW,
    open_above: float = DEFAULT_OPEN_ABOVE,
    min_closed_chunks: int = DEFAULT_MIN_CLOSED_CHUNKS,
) -> "tuple[int, int] | None":
    """``(close_idx, cross_idx)`` of the first close->reopen cycle, or ``None``.

    A three-state machine, and both of the states before CLOSED are load-bearing:

    1. **Waiting for open.** Nothing is detected until the gripper is observed
       OPEN (``> open_above``). A close is a TRANSITION, not a state: if the
       episode begins mid-grasp the transition was never observed, so there is
       no close event to find. Without this, an episode that starts closed --
       every episode of an ``init_state_npz_path`` run whose branch point is
       after the grasp -- latches at chunk 0, reads the release of the PREVIOUS
       grasp as its reopen, and gets truncated to ``keep_chunks + 1`` chunks.
    2. **Open.** CLOSED latches only after ``min_closed_chunks`` CONSECUTIVE
       samples below ``close_below``, and ``close_idx`` is the FIRST of them.
       Hysteresis protects against chatter inside the band but does nothing
       about a single-sample excursion straight through it, and this signal has
       ~6 mm one-sample blips. One transient dip during the approach would
       otherwise latch, and the next open sample -- one chunk later -- would be
       read as the reopen, again amputating the whole failed grasp.
    3. **Closed.** Returns at the first subsequent width above ``open_above``.

    ``None`` means the gripper never closes (by the above definition), or closes
    and is still closed at the last chunk (the "grasped it and never let go"
    failure -- no meander to trim).

    Both guards are free on the reference data: every one of the 48 episodes
    starts at 0.0793 (open), real closed phases run 7-25 chunks, and not one
    episode contains a sub-threshold run shorter than 3 chunks. Both fail
    CONSERVATIVELY -- an undetected close means no truncation at all, which is
    visible as ``episode/n_post_reopen_detected`` falling.

    ``cross_idx`` is where the gripper is unambiguously open again, which is
    0-2 chunks after it actually started opening; :func:`reopen_onset_index`
    recovers the earlier index. Only the FIRST cycle is reported. On the
    reference data that is not even a choice: no failure has a second COMPLETE
    cycle (the one retry, ``episode_0003``, closes again at chunk 38 and is
    never released before truncation), so first-cycle and last-cycle selection
    give identical onsets on all 43.
    """
    seen_open = False
    close_idx = None
    run_start = None
    run_len = 0
    for i, w in enumerate(widths):
        if close_idx is None:
            if not seen_open:
                if w > open_above:
                    seen_open = True
                continue
            if w < close_below:
                if run_start is None:
                    run_start = i
                run_len += 1
                if run_len >= min_closed_chunks:
                    close_idx = run_start
            else:
                run_start = None
                run_len = 0
        elif w > open_above:
            return close_idx, int(i)
    return None


def reopen_onset_index(
    widths: np.ndarray,
    close_below: float = DEFAULT_CLOSE_BELOW,
    open_above: float = DEFAULT_OPEN_ABOVE,
    onset_margin: float = DEFAULT_ONSET_MARGIN,
    min_closed_chunks: int = DEFAULT_MIN_CLOSED_CHUNKS,
) -> "int | None":
    """First chunk in which the gripper has measurably begun to reopen.

    Locates the close->reopen crossing, then walks BACKWARD down the contiguous
    rising edge that leads to it, keeping chunks that are both

      * above ``floor + onset_margin``, where ``floor`` is the minimum width
        over the closed phase (so the test adapts to a close on nothing at
        ~0.001 and a close on the mug at ~0.020 alike), and
      * strictly below their successor, so only a genuinely RISING run is
        traversed.

    Returns ``None`` exactly when :func:`close_cross_indices` does. The result
    is always in ``(close_idx, cross_idx]`` -- never the close chunk itself, so
    the closing motion can never be mistaken for the opening one. That is
    guaranteed by the THRESHOLD, not by the ``onset - 1 > close_idx`` bound: for
    the walk to reach ``close_idx + 1`` every element of
    ``[close_idx + 1, cross_idx)`` must exceed ``floor + onset_margin``, which
    forces ``floor == widths[close_idx]``, and the next test is then
    ``widths[close_idx] > widths[close_idx] + onset_margin`` -- always False.
    The bound is kept as a cheap, explicit statement of the invariant -- it never
    binds on any of the 48 real episodes nor on the exhaustive synthetic sweep in
    ``test_gripper_release.test_reopen_onset`` -- so a future change to the
    threshold logic cannot walk off the front.
    """
    found = close_cross_indices(widths, close_below, open_above, min_closed_chunks)
    if found is None:
        return None
    close_idx, cross_idx = found
    threshold = widths[close_idx:cross_idx].min() + onset_margin
    onset = cross_idx
    while (
        onset - 1 > close_idx
        and widths[onset - 1] > threshold
        and widths[onset - 1] < widths[onset]
    ):
        onset -= 1
    return int(onset)


def post_reopen_detect(
    states: "list[dict[str, np.ndarray]]",
    cfg: PostReopenFilter,
) -> "tuple[int | None, int | None]":
    """``(onset, limit)`` -- where the reopen began, and how much to keep.

    Both may be ``None``, and they mean DIFFERENT things, which is why this
    returns the pair rather than just the limit:

    * ``onset is None`` -- no close->reopen was found at all. Either the episode
      genuinely has none (it never closed, or it closed and never let go), or
      the detector has stopped working on this policy's gripper behaviour. This
      is the only case that belongs in a detector hit rate.
    * ``onset is not None, limit is None`` -- the reopen WAS found, but the
      truncation is not applied. Two sub-cases, both correct no-ops rather than
      misses: ``onset + keep_chunks`` already runs past the end of the episode
      (which becomes MORE common as the policy learns to hold its grasp longer,
      so counting it as a miss would raise a false alarm exactly when things are
      improving); or the limit is below ``cfg.min_train_chunks`` and is refused
      as implausible.
    * both set -- truncate to ``limit`` chunks.

    Callers that need to tell the two ``limit is None`` sub-cases apart should
    compare ``onset + cfg.keep_chunks`` against ``cfg.min_train_chunks``
    themselves; :meth:`EpisodeBuffer._apply_post_reopen_filter` does, so it can
    warn on the implausible one and stay quiet on the benign one.
    """
    if not states:
        return None, None
    widths = gripper_widths(states, cfg.state_key)
    # Sign-convention / units guard. Every grasp-and-place episode is observed
    # with the gripper open at some point (they all start that way), so a trace
    # that never exceeds `open_above` means the width is not being measured
    # correctly -- most likely a mirrored qpos convention, which makes every
    # width negative, pins the state machine in CLOSED from chunk 0, and turns
    # the entire feature into a silent no-op. This does NOT catch a wrong
    # state_key whose values happen to span a similar range; see the module
    # docstring's "Scope".
    if widths.max() <= cfg.open_above:
        raise ValueError(
            f"post-reopen filter: gripper width never exceeds open_above="
            f"{cfg.open_above} over {len(widths)} chunks (max "
            f"{widths.max():.4f}, min {widths.min():.4f}). Expected the gripper "
            f"to be observed OPEN at least once. Check the sign convention of "
            f"'{cfg.state_key}' (width is qpos[0] - qpos[1]) and the width "
            f"thresholds against a histogram of your own data."
        )
    onset = reopen_onset_index(
        widths, cfg.close_below, cfg.open_above, cfg.onset_margin,
        cfg.min_closed_chunks,
    )
    if onset is None:
        return None, None
    limit = onset + cfg.keep_chunks
    if limit >= len(states) or limit < cfg.min_train_chunks:
        return onset, None
    return onset, limit


def post_reopen_chunk_limit(
    states: "list[dict[str, np.ndarray]]",
    cfg: PostReopenFilter,
) -> "int | None":
    """Number of leading chunks to keep, or ``None`` for "keep everything".

    The limit half of :func:`post_reopen_detect`. Callers that need to tell a
    detector miss from a correct no-op must use that function instead.
    """
    return post_reopen_detect(states, cfg)[1]
