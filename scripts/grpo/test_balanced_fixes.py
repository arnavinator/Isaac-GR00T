"""Tests for Fix 1 (bidirectional balanced sampler) and Fix 2 (tent epoch scaling).

Runs without GPU, model, robocasa, or any heavy dependency. All imports come
from the stdlib + numpy, both of which are available in the base Python env.

Run with:
    python3 scripts/grpo/test_balanced_fixes.py
"""
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Minimal stubs so we can import the two methods under test without loading
# the full trainer (which requires CUDA, transformers, etc.).
# ---------------------------------------------------------------------------

@dataclass
class _FakeChunk:
    """Minimal stand-in for ActionChunk — only the fields _iter_balanced_minibatches needs."""
    advantage: float
    group_id: int = 0

    # Fields required by _iter_stratified_minibatches (group_id used for binning)


@dataclass
class _FakeConfig:
    balanced_minibatch_training: bool = True
    dynamic_epoch_training: bool = True
    balanced_minibatch_positive_adv_ratio: float = 0.5
    mini_batch_size: int = 8
    update_epochs: int = 4
    seed: int = 42


class _FakeTrainer:
    """Stub trainer that hosts only the two methods under test."""

    def __init__(self, pos_ratio: float = 0.5, mb_size: int = 8, update_epochs: int = 4):
        self.config = _FakeConfig(
            balanced_minibatch_positive_adv_ratio=pos_ratio,
            mini_batch_size=mb_size,
            update_epochs=update_epochs,
        )

    # ── Copy-paste the two methods from train_grpo.py (they use only self.config
    #    and standard python/numpy; no GPU tensors) ────────────────────────────

    def _iter_stratified_minibatches(self, entries, rng):
        """Simplified stratified fallback: flat shuffle, yield mb_size chunks."""
        indices = np.arange(len(entries))
        rng.shuffle(indices)
        mb = self.config.mini_batch_size
        for start in range(0, len(entries), mb):
            yield [entries[i] for i in indices[start:start + mb]]

    def _iter_balanced_minibatches(
        self,
        entries: list,
        rng: np.random.Generator,
    ) -> Iterator[list]:
        """Bidirectional balanced sampler (Fix 1) — exact copy from train_grpo.py."""
        if not entries:
            return

        pos_ratio = self.config.balanced_minibatch_positive_adv_ratio

        pos_indices = [i for i, (c, _) in enumerate(entries) if c.advantage > 0]
        neg_indices = [i for i, (c, _) in enumerate(entries) if c.advantage <= 0]

        if not pos_indices or not neg_indices:
            yield from self._iter_stratified_minibatches(entries, rng)
            return

        natural_pos_frac = len(pos_indices) / len(entries)

        mb_size = self.config.mini_batch_size
        n_pos_per_batch = max(1, round(pos_ratio * mb_size))
        n_neg_per_batch = mb_size - n_pos_per_batch

        if n_neg_per_batch <= 0:
            yield from self._iter_stratified_minibatches(entries, rng)
            return

        if natural_pos_frac < pos_ratio:
            minority_indices = pos_indices
            majority_indices = neg_indices
            n_minority_per_batch = n_pos_per_batch
            n_majority_per_batch = n_neg_per_batch
        else:
            minority_indices = neg_indices
            majority_indices = pos_indices
            n_minority_per_batch = n_neg_per_batch
            n_majority_per_batch = n_pos_per_batch

        minority_pool = list(rng.permutation(len(minority_indices)).astype(int))
        majority_pool = list(rng.permutation(len(majority_indices)).astype(int))

        n_batches = math.ceil(len(entries) / mb_size)
        minority_ptr = 0
        majority_ptr = 0

        for _ in range(n_batches):
            batch = []

            for _ in range(n_minority_per_batch):
                if minority_ptr >= len(minority_pool):
                    minority_pool = list(rng.permutation(len(minority_indices)).astype(int))
                    minority_ptr = 0
                batch.append(entries[minority_indices[minority_pool[minority_ptr]]])
                minority_ptr += 1

            taken = 0
            while taken < n_majority_per_batch and majority_ptr < len(majority_pool):
                batch.append(entries[majority_indices[majority_pool[majority_ptr]]])
                majority_ptr += 1
                taken += 1

            yield batch

            if majority_ptr >= len(majority_pool):
                return


def _make_tent_epochs(success_frac: float, update_epochs: int) -> int:
    """Tent epoch helper for test cases using clean fractional inputs.

    For the exact fractions used in the test table (0.0, 0.1, 0.25, …, 1.0)
    the float form is consistent with the integer formula in the real code.
    Use _make_tent_epochs_exact() when testing specific integer counts,
    especially at update_epochs >= 6 where ULP cancellation can differ.
    """
    update_scale = 2.0 * min(success_frac, 1.0 - success_frac)
    return max(1, math.floor(update_scale * update_epochs + 0.5))


def _make_tent_epochs_exact(k: int, n: int, update_epochs: int) -> int:
    """Exact integer tent formula matching train_grpo.py _grpo_update_inner.

    Real code:
        m = min(successful_eps, total_eps - successful_eps)
        actual_num_epochs = max(1, (4 * m * update_epochs + total_eps) // (2 * total_eps))

    Use for update_epochs >= 6 where ULP cancellation can affect float form.
    Example bug case: k=17, n=24, E=6 → float gives 3, integer gives 4.
    """
    m = min(k, n - k)
    return max(1, (4 * m * update_epochs + n) // (2 * n))


def compute_actual_epochs(dynamic_epoch_training: bool, success_frac: float, update_epochs: int) -> int:
    """Stub for the epoch-count branching in _grpo_update_inner.

    Mirrors the if/else at train_grpo.py:
        if self.config.dynamic_epoch_training:
            m = min(successful_eps, total_eps - successful_eps)
            actual_num_epochs = max(1, (4*m*E + n) // (2*n))  # integer tent
        else:
            actual_num_epochs = self.config.update_epochs  # tent NOT applied
    """
    if dynamic_epoch_training:
        return _make_tent_epochs(success_frac, update_epochs)
    else:
        return update_epochs  # no tent — exactly what the else branch does


def _make_entries(n_pos: int, n_neg: int):
    """Build a flat list of (chunk, 'fixed') entries with given +/- counts."""
    entries = []
    for _ in range(n_pos):
        entries.append((_FakeChunk(advantage=1.0), "fixed"))
    for _ in range(n_neg):
        entries.append((_FakeChunk(advantage=-1.0), "fixed"))
    return entries


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = f"  {FAIL}  {name}" + (f": {detail}" if detail else "")
        print(msg)
        _failures.append(name)


# ── Fix 2: Tent epoch scaling ────────────────────────────────────────────────

def test_tent_epoch_scaling():
    print("\n[Fix 2] Tent epoch scaling")

    # update_epochs=4 (the value used in the actual training runs)
    cases = {
        0.0:   1,   # all failure → min 1
        0.1:   1,   # scale=0.2, round(0.8)=1
        0.25:  2,   # scale=0.5, round(2.0)=2
        0.375: 3,   # scale=0.75, round(3.0)=3
        0.5:   4,   # scale=1.0, round(4.0)=4  ← full epochs at peak
        0.625: 3,   # scale=0.75, round(3.0)=3
        0.7:   2,   # scale=0.6, round(2.4)=2  ← key fix (old formula gave 3)
        0.75:  2,   # scale=0.5, round(2.0)=2
        0.9:   1,   # scale=0.2, round(0.8)=1
        1.0:   1,   # all success → min 1
    }
    for frac, expected in cases.items():
        got = _make_tent_epochs(frac, update_epochs=4)
        check(
            f"tent(frac={frac:.3f}, epochs=4) = {expected}",
            got == expected,
            f"got {got}",
        )

    # The critical case that caused the collapse: 70% success, update_epochs=4
    # Old formula: ceil(0.7 * 4) = ceil(2.8) = 3  ← TOO HIGH
    # New formula: round(0.6 * 4) = round(2.4) = 2 ← conservative
    old_result = math.ceil(0.7 * 4)
    new_result = _make_tent_epochs(0.7, update_epochs=4)
    check(
        "tent reduces epochs vs old formula at 70% success (3→2)",
        new_result < old_result,
        f"old={old_result}, new={new_result}",
    )

    # Symmetry: tent(f) == tent(1-f)
    for f in [0.1, 0.3, 0.4, 0.6, 0.7, 0.9]:
        a = _make_tent_epochs(f, update_epochs=4)
        b = _make_tent_epochs(1.0 - f, update_epochs=4)
        check(f"tent symmetric: tent({f}) == tent({1-f})", a == b, f"{a} vs {b}")

    # Always >= 1
    for f in [0.0, 0.01, 0.5, 0.99, 1.0]:
        got = _make_tent_epochs(f, update_epochs=4)
        check(f"tent always >=1 at frac={f}", got >= 1, f"got {got}")

    # dynamic_epoch_training=False path is unaffected — returns update_epochs unchanged
    # (this is tested at the trainer level in test_false_path_unchanged)


# ── Fix 1: Bidirectional balanced sampler ───────────────────────────────────

def test_balanced_sampler_too_few_positives():
    """Original behaviour: natural_pos_frac < pos_ratio → oversample positives."""
    print("\n[Fix 1] Too-few-positives case (original behaviour preserved)")

    trainer = _FakeTrainer(pos_ratio=0.5, mb_size=8)
    rng = np.random.default_rng(0)

    # 6 pos, 14 neg → natural_pos_frac=0.3 < 0.5
    entries = _make_entries(6, 14)
    batches = list(trainer._iter_balanced_minibatches(entries, rng))

    check("yields at least 1 batch", len(batches) >= 1)

    for i, batch in enumerate(batches):
        pos_in_batch = sum(1 for c, _ in batch if c.advantage > 0)
        neg_in_batch = sum(1 for c, _ in batch if c.advantage <= 0)
        # Each batch should have ~ pos_ratio=0.5 positives
        check(
            f"batch {i}: has >=1 positive and >=1 negative",
            pos_in_batch >= 1 and neg_in_batch >= 1,
            f"pos={pos_in_batch}, neg={neg_in_batch}",
        )

    # Over the full epoch, positives should be over-represented relative to
    # natural (6/20=30%) — cycling brings them to ~50% per batch
    all_pos = sum(sum(1 for c, _ in b if c.advantage > 0) for b in batches)
    all_total = sum(len(b) for b in batches)
    obs_pos_frac = all_pos / all_total
    check(
        "epoch pos fraction ≈ 0.5 (oversampled from natural 0.3)",
        abs(obs_pos_frac - 0.5) < 0.15,
        f"observed={obs_pos_frac:.2f}",
    )


def test_balanced_sampler_too_many_positives():
    """New behaviour: natural_pos_frac > pos_ratio → oversample negatives."""
    print("\n[Fix 1] Too-many-positives case (new behaviour — Fix 1)")

    trainer = _FakeTrainer(pos_ratio=0.5, mb_size=8)
    rng = np.random.default_rng(42)

    # 14 pos, 6 neg → natural_pos_frac=0.7 > 0.5  (mirrors the iter-4 collapse scenario)
    entries = _make_entries(14, 6)
    batches = list(trainer._iter_balanced_minibatches(entries, rng))

    check("yields at least 1 batch", len(batches) >= 1)

    for i, batch in enumerate(batches):
        pos_in_batch = sum(1 for c, _ in batch if c.advantage > 0)
        neg_in_batch = sum(1 for c, _ in batch if c.advantage <= 0)
        check(
            f"batch {i}: has >=1 positive and >=1 negative",
            pos_in_batch >= 1 and neg_in_batch >= 1,
            f"pos={pos_in_batch}, neg={neg_in_batch}",
        )

    all_pos = sum(sum(1 for c, _ in b if c.advantage > 0) for b in batches)
    all_total = sum(len(b) for b in batches)
    obs_pos_frac = all_pos / all_total

    # With negatives cycling, pos fraction should be ~0.5, NOT ~0.7 (natural)
    check(
        "epoch pos fraction ≈ 0.5 (negatives cycled from natural 0.7)",
        abs(obs_pos_frac - 0.5) < 0.15,
        f"observed={obs_pos_frac:.2f}",
    )

    # The old code would have fallen back to stratified here, giving ~0.7 pos frac
    # Verify we're NOT giving the natural (biased) distribution
    check(
        "pos fraction < natural 0.7 (not just falling back to stratified)",
        obs_pos_frac < 0.65,
        f"observed={obs_pos_frac:.2f}",
    )


def test_balanced_sampler_per_batch_ratio():
    """Each individual batch should respect the target ratio (4+4 for mb=8, ratio=0.5)."""
    print("\n[Fix 1] Per-batch ratio check")

    trainer = _FakeTrainer(pos_ratio=0.5, mb_size=8)

    for (n_pos, n_neg), label in [
        ((6, 14), "too-few-positives"),
        ((14, 6), "too-many-positives"),
        ((10, 10), "exactly-balanced"),
    ]:
        rng = np.random.default_rng(7)
        entries = _make_entries(n_pos, n_neg)
        batches = list(trainer._iter_balanced_minibatches(entries, rng))

        n_pos_per_batch = max(1, round(0.5 * 8))  # 4
        n_neg_per_batch = 8 - n_pos_per_batch       # 4

        # All full-size batches (skip last which may be smaller due to majority drain)
        full_batches = [b for b in batches if len(b) == 8]
        for i, batch in enumerate(full_batches):
            pos_count = sum(1 for c, _ in batch if c.advantage > 0)
            neg_count = sum(1 for c, _ in batch if c.advantage <= 0)
            check(
                f"{label} full-batch {i}: exactly {n_pos_per_batch}pos + {n_neg_per_batch}neg",
                pos_count == n_pos_per_batch and neg_count == n_neg_per_batch,
                f"pos={pos_count}, neg={neg_count}",
            )


def test_balanced_sampler_fallback_when_one_class_absent():
    """When all entries are positive or all negative, fall back to stratified."""
    print("\n[Fix 1] Fallback when one class absent")

    trainer = _FakeTrainer(pos_ratio=0.5, mb_size=4)
    rng = np.random.default_rng(0)

    # All positive
    entries_all_pos = _make_entries(10, 0)
    batches = list(trainer._iter_balanced_minibatches(entries_all_pos, rng))
    check("all-positive: still yields batches (stratified fallback)", len(batches) >= 1)
    all_neg_count = sum(sum(1 for c, _ in b if c.advantage <= 0) for b in batches)
    check("all-positive: no negatives in any batch", all_neg_count == 0)

    # All negative
    rng2 = np.random.default_rng(1)
    entries_all_neg = _make_entries(0, 10)
    batches2 = list(trainer._iter_balanced_minibatches(entries_all_neg, rng2))
    check("all-negative: still yields batches (stratified fallback)", len(batches2) >= 1)
    all_pos_count = sum(sum(1 for c, _ in b if c.advantage > 0) for b in batches2)
    check("all-negative: no positives in any batch", all_pos_count == 0)


def test_balanced_sampler_empty():
    """Empty entries → no batches, no crash."""
    print("\n[Fix 1] Empty entries")

    trainer = _FakeTrainer()
    rng = np.random.default_rng(0)
    batches = list(trainer._iter_balanced_minibatches([], rng))
    check("empty entries: zero batches yielded", len(batches) == 0)


def test_balanced_sampler_epoch_length_anchor():
    """n_batches is anchored to ceil(n_entries / mb_size) in both directions."""
    print("\n[Fix 1] Epoch length anchor")

    trainer = _FakeTrainer(pos_ratio=0.5, mb_size=8)

    for n_pos, n_neg in [(6, 14), (14, 6), (10, 10)]:
        n_entries = n_pos + n_neg
        expected_max_batches = math.ceil(n_entries / 8)

        rng = np.random.default_rng(99)
        entries = _make_entries(n_pos, n_neg)
        batches = list(trainer._iter_balanced_minibatches(entries, rng))

        check(
            f"({n_pos}pos, {n_neg}neg): n_batches <= ceil({n_entries}/8)={expected_max_batches}",
            len(batches) <= expected_max_batches,
            f"got {len(batches)} batches",
        )
        check(
            f"({n_pos}pos, {n_neg}neg): n_batches >= 1",
            len(batches) >= 1,
            f"got {len(batches)} batches",
        )


def test_false_path_unchanged():
    """dynamic_epoch_training=False: actual_num_epochs must always equal config.update_epochs.

    Uses compute_actual_epochs() — a stub that mirrors the if/else in
    _grpo_update_inner — so we actually exercise both branches rather than
    checking local variable assignments (which would be a tautology).
    """
    print("\n[Fix 2] dynamic_epoch_training=False path is unchanged")

    update_epochs = 4

    # False path: no tent applied regardless of what success_frac would be.
    # Verify compute_actual_epochs returns update_epochs exactly for all fractions.
    for frac in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        got = compute_actual_epochs(dynamic_epoch_training=False, success_frac=frac, update_epochs=update_epochs)
        check(
            f"dynamic_epoch=False, frac={frac}: actual_epochs=={update_epochs} (tent NOT applied)",
            got == update_epochs,
            f"got {got}",
        )

    # Confirm tent IS different from update_epochs at key fractions (so if the
    # False path accidentally applied tent, at least some of the above would fail).
    tent_differs_at = [f for f in [0.0, 0.1, 0.3, 0.7, 0.9, 1.0]
                       if compute_actual_epochs(True, f, update_epochs) != update_epochs]
    check(
        "tent gives ≠ update_epochs for at least one fraction (distinguishable from False path)",
        len(tent_differs_at) > 0,
        f"tent differs at: {tent_differs_at}",
    )

    # True path: tent IS applied and gives update_epochs only at exactly sf=0.5.
    got_at_half = compute_actual_epochs(dynamic_epoch_training=True, success_frac=0.5, update_epochs=update_epochs)
    check(
        f"dynamic_epoch=True, frac=0.5: actual_epochs=={update_epochs} (tent peak)",
        got_at_half == update_epochs,
        f"got {got_at_half}",
    )
    # At 0% and 100% success (all-one-sign), tent gives 1 — not update_epochs.
    for extreme_frac in [0.0, 1.0]:
        got = compute_actual_epochs(dynamic_epoch_training=True, success_frac=extreme_frac, update_epochs=update_epochs)
        check(
            f"dynamic_epoch=True, frac={extreme_frac}: actual_epochs==1 (tent floor)",
            got == 1,
            f"got {got}",
        )


def test_tent_matches_old_at_low_success():
    """At low success fractions, new formula is comparable to (or more conservative than) old."""
    print("\n[Fix 2] Tent vs old formula comparison")

    update_epochs = 4
    for frac in [0.0, 0.1, 0.2, 0.25, 0.3]:
        old = max(1, math.ceil(frac * update_epochs))
        new = _make_tent_epochs(frac, update_epochs)
        check(
            f"tent({frac:.2f}) >= old formula: {new} >= {old}",
            new >= old,
            f"old={old} new={new}",
        )

    # At high success, new is strictly less (the key fix)
    for frac in [0.7, 0.75, 0.8, 0.9, 1.0]:
        old = max(1, math.ceil(frac * update_epochs))
        new = _make_tent_epochs(frac, update_epochs)
        check(
            f"tent({frac:.2f}) <= old formula: {new} <= {old}  (conservative at high success)",
            new <= old,
            f"old={old} new={new}",
        )


def test_balanced_sampler_minority_cycles():
    """The minority pool should cycle with replacement when it's exhausted."""
    print("\n[Fix 1] Minority pool cycles (appears >1× per epoch)")

    trainer = _FakeTrainer(pos_ratio=0.5, mb_size=8)

    # 4 pos, 12 neg → natural_pos_frac=0.25 < 0.5 → positives are minority
    # With 3 full batches (ceil(16/8)=2, but with 12 negs → majority drains after 3):
    # Actually: n_batches=ceil(16/8)=2. majority=negs(12), n_majority_per_batch=4 → 3 batches until negs drain.
    # For loop runs min(n_batches=2, ...) so only 2 batches, 8 negs used.
    # 4 positives, 4 per batch → positives MUST cycle (4 pos, need 2*4=8 minority slots)
    n_pos, n_neg = 4, 12
    rng = np.random.default_rng(13)
    entries = _make_entries(n_pos, n_neg)
    batches = list(trainer._iter_balanced_minibatches(entries, rng))

    total_pos_appearances = sum(sum(1 for c, _ in b if c.advantage > 0) for b in batches)
    # If cycling works, total_pos_appearances > n_pos (each positive appeared >1×)
    check(
        f"minority positives appear >n_pos={n_pos} times across epoch (cycling)",
        total_pos_appearances > n_pos,
        f"total_pos_appearances={total_pos_appearances}",
    )


def test_iter4_scenario():
    """Exact replication of the balanced iter-4 scenario that caused collapse.

    iter 4: 14/20 positive-advantage episodes (success_frac=0.7, 5 groups × 4 eps)
    Each episode has ~47 chunks (852 chunks / 20 eps ≈ 42 chunks/ep, split into
    minibatches of 8). We simplify: 1 chunk per episode, 20 total entries.

    Old code: natural_pos_frac=0.7 >= pos_ratio=0.5 → falls back to stratified
               → each batch ~70% positive → per-minibatch z-score amplifies negatives
    New code: negatives are minority → cycle negatives, exhaust positives
               → each batch ≈ 50% positive / 50% negative
    """
    print("\n[Fix 1] Iter-4 collapse scenario (14pos/6neg, pos_ratio=0.5)")

    trainer = _FakeTrainer(pos_ratio=0.5, mb_size=8)
    rng = np.random.default_rng(17)

    # 14 pos, 6 neg exactly as in balanced iter 4
    entries = _make_entries(14, 6)
    batches = list(trainer._iter_balanced_minibatches(entries, rng))

    # Old code would have given ~70% positive per batch (stratified fallback)
    old_expected_pos_frac = 14 / 20  # = 0.7

    all_pos = sum(sum(1 for c, _ in b if c.advantage > 0) for b in batches)
    all_total = sum(len(b) for b in batches)
    obs_pos_frac = all_pos / all_total

    check(
        "iter-4: batch pos fraction ≈ 0.5 (not natural 0.7)",
        abs(obs_pos_frac - 0.5) < 0.15,
        f"obs={obs_pos_frac:.2f}, natural={old_expected_pos_frac}",
    )
    check(
        "iter-4: pos fraction significantly less than natural 0.7",
        obs_pos_frac < 0.65,
        f"obs={obs_pos_frac:.2f}",
    )

    # Verify tent epoch scaling: at success_frac=0.7, update_epochs=4
    tent_epochs = _make_tent_epochs(0.7, update_epochs=4)
    old_epochs = max(1, math.ceil(0.7 * 4))
    check(
        f"iter-4: tent gives {tent_epochs} epochs (not old formula's {old_epochs})",
        tent_epochs == 2 and old_epochs == 3,
        f"tent={tent_epochs}, old={old_epochs}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_tent_exact_integer_formula():
    """Verify _make_tent_epochs_exact avoids ULP cancellation at E >= 6.

    The float-based formula math.floor((2*(k/n) - ulp_error)*E + 0.5) can
    give the wrong answer when 1.0 - k/n differs from (n-k)/n by 1 ULP and
    the product lands just below a half-integer.  The integer formula is exact.
    """
    print("\n[Fix 2] Integer tent formula immune to ULP cancellation")

    # Known ULP bug cases (from audit): float formula gives wrong result, integer correct
    bug_cases = [
        (17, 24, 6, 4),   # float gives 3, correct is 4
        (25, 28, 7, 2),   # float gives 1, correct is 2 (audit: "just below 0.5")
    ]
    for k, n, E, expected in bug_cases:
        got = _make_tent_epochs_exact(k, n, E)
        check(
            f"integer tent(k={k},n={n},E={E})=={expected} (ULP bug case)",
            got == expected,
            f"got {got}",
        )

    # Symmetry: tent_exact(k, n, E) == tent_exact(n-k, n, E)
    for k, n, E in [(3, 8, 4), (7, 24, 6), (1, 10, 5)]:
        a = _make_tent_epochs_exact(k, n, E)
        b = _make_tent_epochs_exact(n - k, n, E)
        check(f"integer tent symmetric: k={k} vs k={n-k}, n={n}, E={E}", a == b, f"{a} vs {b}")

    # At k=n//2 (peak), give full E (or as close as integer division allows)
    for n, E in [(8, 4), (10, 5), (20, 4)]:
        k = n // 2
        got = _make_tent_epochs_exact(k, n, E)
        check(f"integer tent peak: k={k}, n={n}, E={E} → {E}", got == E, f"got {got}")

    # Always >= 1
    for k, n, E in [(0, 8, 4), (8, 8, 4), (1, 100, 4)]:
        got = _make_tent_epochs_exact(k, n, E)
        check(f"integer tent always >=1: k={k},n={n},E={E}", got >= 1, f"got {got}")


# ── Split: the two flags are independent ─────────────────────────────────────

def test_independent_flag_combinations():
    """balanced_minibatch_training and dynamic_epoch_training are independent.

    Drives all four on/off combinations on the iter-4-like input (14 pos, 6 neg,
    success_frac=0.7, update_epochs=4) and asserts:
      - the minibatch flag ALONE controls sampling: ON → epoch pos-fraction
        ≈ 0.5 (balanced), OFF → ≈ 0.7 (stratified / natural).
      - the dynamic-epoch flag ALONE controls epochs: ON → tent gives 2, OFF → 4.
    NOTE on scope: like the rest of this file, the dispatch (`_dispatch` below)
    and epoch (`compute_actual_epochs`) helpers MIRROR the real gating in
    train_grpo._grpo_update_inner rather than invoking it (the real path needs
    the GPU model + buffer). So this guards the *logic* of independence, not the
    exact flag names at the real dispatch sites — those are verified by
    inspection (train_grpo.py: epochs gate on dynamic_epoch_training, sampler
    gate on balanced_minibatch_training).
    """
    print("\n[Split] Independent balanced_minibatch_training × dynamic_epoch_training")

    n_pos, n_neg = 14, 6
    success_frac = n_pos / (n_pos + n_neg)  # 0.7
    update_epochs = 4

    def _dispatch(trainer, entries, rng):
        # Mirror train_grpo._grpo_update_inner's minibatch dispatch gating.
        if trainer.config.balanced_minibatch_training:
            return list(trainer._iter_balanced_minibatches(entries, rng))
        return list(trainer._iter_stratified_minibatches(entries, rng))

    for mb_flag in (True, False):
        for dyn_flag in (True, False):
            trainer = _FakeTrainer(pos_ratio=0.5, mb_size=8, update_epochs=update_epochs)
            trainer.config.balanced_minibatch_training = mb_flag
            trainer.config.dynamic_epoch_training = dyn_flag

            rng = np.random.default_rng(17)
            entries = _make_entries(n_pos, n_neg)
            batches = _dispatch(trainer, entries, rng)
            all_pos = sum(sum(1 for c, _ in b if c.advantage > 0) for b in batches)
            all_total = sum(len(b) for b in batches)
            obs_pos_frac = all_pos / all_total if all_total else 0.0

            # Sampler behavior depends ONLY on the minibatch flag.
            if mb_flag:
                check(
                    f"mb={mb_flag}, dyn={dyn_flag}: balanced sampler → pos_frac≈0.5",
                    abs(obs_pos_frac - 0.5) < 0.15,
                    f"obs={obs_pos_frac:.2f}",
                )
            else:
                check(
                    f"mb={mb_flag}, dyn={dyn_flag}: stratified → pos_frac≈natural 0.7",
                    obs_pos_frac > 0.6,
                    f"obs={obs_pos_frac:.2f}",
                )

            # Epoch count depends ONLY on the dynamic-epoch flag.
            epochs = compute_actual_epochs(dyn_flag, success_frac, update_epochs)
            expected_epochs = 2 if dyn_flag else update_epochs
            check(
                f"mb={mb_flag}, dyn={dyn_flag}: actual_epochs=={expected_epochs}",
                epochs == expected_epochs,
                f"got {epochs}",
            )


if __name__ == "__main__":
    test_tent_epoch_scaling()
    test_balanced_sampler_too_few_positives()
    test_balanced_sampler_too_many_positives()
    test_balanced_sampler_per_batch_ratio()
    test_balanced_sampler_fallback_when_one_class_absent()
    test_balanced_sampler_empty()
    test_balanced_sampler_epoch_length_anchor()
    test_false_path_unchanged()
    test_tent_matches_old_at_low_success()
    test_balanced_sampler_minority_cycles()
    test_iter4_scenario()
    test_tent_exact_integer_formula()
    test_independent_flag_combinations()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} test(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        total = sum(1 for n in dir() if n.startswith("test_"))
        print(f"\033[32mAll tests passed.\033[0m")
