"""CPU-only tests for the FROZEN SCENE SEED POOL (GRPOConfig.scene_seed_pool_size).

The feature replaces "every iteration draws brand-new RoboCasa scenes" with a
fixed pool of K scene seeds cycled deterministically across iterations, so the
success-rate curve stops being dominated by scene resampling. Four surfaces, all
covered here:

  1. `GRPOConfig.__post_init__` — base resolution + the four hard validations and
     the pass-alignment warning.
  2. `GRPOTrainer._scene_seed_pool` / `_scene_seeds_for_iteration` /
     `_scene_pool_pass` — the stateless cursor, and the collector argv it feeds.
  3. `collect_episodes.resolve_group_seed` and the real
     `EpisodeCollector.collect` group loop — that group N actually resets with
     `group_seeds[N]`, and that an over-run RAISES rather than wrapping (a wrap
     would put two groups of one iteration on the same scene).
  4. `EpisodeBuffer.stats()["per_scene_success"]` and the
     `episode/scene_sr/<seed>` + `episode/pool_pass` emission in
     `GRPOTrainer._log_metrics`, including its gate on the pool being enabled.

No GPU, no robocasa, no MuJoCo: the collector is driven with
test_turn_collection.py's fakes (the established pattern for exercising the real
EpisodeCollector control flow), and the trainer's argv builder is driven with a
faked `subprocess.Popen`.

Run with:
    .venv/bin/python scripts/grpo/test_scene_seed_pool.py
"""
import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import collect_episodes as ce                                 # noqa: E402
import train_grpo as tg                                       # noqa: E402
import test_turn_collection as tc                             # noqa: E402  (fakes)
from episode_buffer import EpisodeBuffer, GRPOEpisode         # noqa: E402
from grpo_config import GRPOConfig                            # noqa: E402


# ---------------------------------------------------------------------------
# Test harness (mirrors test_turn_collection.py / test_anchor_groups.py)
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


def _pool_config(**overrides) -> GRPOConfig:
    """A minimal pool-enabled config. K=12 / num_groups=4 / min_alive_groups=0
    is the canonical valid combination; overrides let a test break exactly one
    of those.

    `max_groups` is deliberately LEFT AT THE DATACLASS DEFAULT (5, i.e. greater
    than num_groups) rather than pinned equal to num_groups. Pinning it would
    hide the interaction that actually matters: the pool's minimum-K bound is
    num_groups alone, because the mandatory min_alive_groups == 0 makes the
    collector's dynamic extension unreachable, so max_groups must never inflate
    the requirement.
    """
    kwargs = dict(
        device="cpu",
        seed=67,
        num_groups=4,
        min_alive_groups=0,
        scene_seed_pool_size=12,
    )
    kwargs.update(overrides)
    return GRPOConfig(**kwargs)


def _trainer(config: GRPOConfig, iteration: int = 1):
    """A bare GRPOTrainer carrying only what the pool code paths read.

    `__new__` rather than `GRPOTrainer(config)`: the real __init__ loads the 3B
    model onto a GPU. Same technique test_anchor_groups.py uses to drive
    `_log_metrics` and `_grpo_update_inner` on CPU.
    """
    tr = tg.GRPOTrainer.__new__(tg.GRPOTrainer)
    tr.config = config
    tr.iteration = iteration
    return tr


class _FakeProc:
    """Minimal subprocess.Popen stand-in: no output, clean exit."""

    returncode = 0

    def __init__(self):
        self.stdout = iter(())

    def poll(self):
        return 0

    def wait(self):
        return 0


def _collector_argv(config: GRPOConfig, iteration: int) -> list:
    """The argv `_collect_via_subprocess` would hand the robocasa venv.

    Runs the REAL method with `subprocess.Popen` faked, so the argv under test is
    the one production builds — not a re-implementation that could drift.
    """
    captured: list = []

    def _fake_popen(cmd, **kwargs):
        captured.append(list(cmd))
        return _FakeProc()

    tr = _trainer(config, iteration)
    with mock.patch.object(tg.subprocess, "Popen", _fake_popen):
        failure = tr._collect_via_subprocess(
            env_name="robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env",
            episode_dir=Path("/tmp/does_not_need_to_exist"),
            max_steps=480,
            ff_steps=0,
        )
    assert failure is None, failure
    return captured[0]


# ---------------------------------------------------------------------------
# 1. The duplicated stride constant
# ---------------------------------------------------------------------------


def test_stride_constants_agree():
    """train_grpo mirrors collect_episodes.GROUP_SEED_STRIDE; it cannot drift.

    The trainer cannot import collect_episodes (that module imports
    gymnasium/robosuite at module scope and only exists in the robocasa venv), so
    the constant is duplicated. If the two ever disagree, every pool seed the
    trainer computes lands on a DIFFERENT scene than the one the collector's own
    formula would have produced for that slot — silently, since both values are
    valid seeds.
    """
    print("\n[stride] the duplicated GROUP_SEED_STRIDE literals agree")
    check("train_grpo.GROUP_SEED_STRIDE == collect_episodes.GROUP_SEED_STRIDE",
          tg.GROUP_SEED_STRIDE == ce.GROUP_SEED_STRIDE,
          f"trainer={tg.GROUP_SEED_STRIDE} collector={ce.GROUP_SEED_STRIDE}")
    check("the shared value is 1000", tg.GROUP_SEED_STRIDE == 1000,
          f"{tg.GROUP_SEED_STRIDE}")


# ---------------------------------------------------------------------------
# 2. Pool resolution and the cursor
# ---------------------------------------------------------------------------


def test_pool_resolution():
    """seed=67, K=12, base defaulted → exactly [100067, 101067, ..., 111067]."""
    print("\n[pool] default base resolution (seed=67, K=12)")
    cfg = _pool_config()
    check("scene_seed_pool_base resolved IN PLACE (not left None, so the TB "
          "config dump records the real scene set)",
          cfg.scene_seed_pool_base == 100_067,
          f"{cfg.scene_seed_pool_base}")
    check("default base == the seed block iteration 1 would have drawn under "
          "the old formula (seed + 1 * 100_000)",
          cfg.scene_seed_pool_base == cfg.seed + 100_000)

    pool = _trainer(cfg)._scene_seed_pool()
    want = [100_067 + j * 1000 for j in range(12)]
    check("pool == [100067, 101067, ..., 111067]", pool == want, f"{pool}")
    check("pool length == K", len(pool) == 12, f"{len(pool)}")

    # An explicit base is used verbatim, never re-derived.
    cfg2 = _pool_config(scene_seed_pool_base=500_000)
    check("explicit scene_seed_pool_base is honored verbatim",
          _trainer(cfg2)._scene_seed_pool()[:3] == [500_000, 501_000, 502_000],
          f"{_trainer(cfg2)._scene_seed_pool()[:3]}")

    # Disabled → empty pool, and nothing derived from it.
    off = GRPOConfig(device="cpu", scene_seed_pool_size=0)
    check("pool disabled → empty pool list", _trainer(off)._scene_seed_pool() == [])
    check("pool disabled → base left None (no silent resolution on an off run)",
          off.scene_seed_pool_base is None, f"{off.scene_seed_pool_base}")


def test_cursor_and_pass_alignment():
    """K=12, num_groups=4 → three iterations per pass, then wrap."""
    print("\n[pool] stateless cursor across iterations (K=12, num_groups=4)")
    cfg = _pool_config()
    expected = {
        1: [100_067, 101_067, 102_067, 103_067],
        2: [104_067, 105_067, 106_067, 107_067],
        3: [108_067, 109_067, 110_067, 111_067],
        4: [100_067, 101_067, 102_067, 103_067],   # wraps back to pass 1
    }
    for it, want in expected.items():
        got = _trainer(cfg, iteration=it)._scene_seeds_for_iteration()
        check(f"iteration {it} → {want}", got == want, f"{got}")

    check("iteration 1 starts at pool index 0 (the -1 in the cursor)",
          _trainer(cfg, 1)._scene_seeds_for_iteration()[0]
          == _trainer(cfg)._scene_seed_pool()[0])

    # pool_pass: 0 for iters 1-3, 1 for 4-6, ...
    passes = [_trainer(cfg, it)._scene_pool_pass(it) for it in range(1, 10)]
    check("pool_pass == [0,0,0,1,1,1,2,2,2] over iterations 1..9",
          passes == [0, 0, 0, 1, 1, 1, 2, 2, 2], f"{passes}")

    # The cursor is a PURE function of iteration — which is what makes
    # --resume-from correct with no checkpoint state. Recomputing it out of
    # order, or after visiting other iterations, must give the same answer.
    tr = _trainer(cfg, iteration=1)
    scrambled = {}
    for it in (7, 2, 41, 2, 7):
        tr.iteration = it
        scrambled.setdefault(it, []).append(tr._scene_seeds_for_iteration())
    check("cursor is stateless: revisiting an iteration reproduces its seeds "
          "exactly (resume-from correctness)",
          all(len(set(map(tuple, v))) == 1 for v in scrambled.values()),
          f"{scrambled}")
    check("a mid-run 'resume' at iteration 41 gets the same seeds as walking "
          "there sequentially",
          scrambled[41][0]
          == [_trainer(cfg)._scene_seed_pool()[(40 * 4 + g) % 12]
              for g in range(4)],
          f"{scrambled[41][0]}")

    check("pool disabled → empty seed list (nothing to append to the argv)",
          _trainer(GRPOConfig(device="cpu"), 3)._scene_seeds_for_iteration() == [])
    check("pool disabled → pool_pass is a constant 0 (never emitted anyway)",
          _trainer(GRPOConfig(device="cpu"), 3)._scene_pool_pass(3) == 0)


def test_distinct_within_iteration():
    """No seed may repeat WITHIN one iteration, for any valid (K, num_groups).

    This is the invariant the `K >= num_groups` validation exists to protect:
    two groups on one scene would correlate their group-relative advantages and
    double-count that scene in the iteration mean. Includes K=10 / num_groups=4,
    where K is NOT divisible by num_groups — the case that only warns, so the
    invariant has to hold without pass alignment. K == num_groups (4, 4) is the
    tightest legal pool and must still never repeat within an iteration.
    """
    print("\n[pool] no repeated seed inside any single iteration")
    for K, ng in [(4, 4), (12, 4), (10, 4), (7, 3), (13, 5), (100, 1)]:
        with warnings.catch_warnings():          # the non-divisible warning
            warnings.simplefilter("ignore")
            cfg = _pool_config(
                scene_seed_pool_size=K, num_groups=ng,
                # max_groups must stay >= num_groups (an unrelated, pre-existing
                # validation), but is otherwise left off the tightest-pool path.
                max_groups=max(ng, 5),
            )
        tr = _trainer(cfg)
        worst = None
        for it in range(1, 60):
            tr.iteration = it
            seeds = tr._scene_seeds_for_iteration()
            if len(set(seeds)) != len(seeds) or len(seeds) != ng:
                worst = (it, seeds)
                break
        check(f"K={K}, num_groups={ng}: {ng} distinct seeds every iteration "
              f"over 59 iterations", worst is None, f"iter {worst}")

        # And every seed used is a member of the frozen pool — the pool never
        # leaks a value derived some other way.
        tr.iteration = 17
        pool = set(tr._scene_seed_pool())
        check(f"K={K}, num_groups={ng}: every emitted seed is a pool member",
              set(tr._scene_seeds_for_iteration()) <= pool)


# ---------------------------------------------------------------------------
# 3. Config validations
# ---------------------------------------------------------------------------


def _raises(label: str, **overrides) -> None:
    """Assert GRPOConfig(**overrides) raises ValueError."""
    try:
        _pool_config(**overrides)
    except ValueError:
        check(f"{label} raises ValueError", True)
    except Exception as exc:                                 # noqa: BLE001
        check(f"{label} raises ValueError", False,
              f"got {type(exc).__name__}: {exc}")
    else:
        check(f"{label} raises ValueError", False, "no error raised")


def test_validations_raise():
    print("\n[config] every scene-pool validation is a hard error")
    # K < num_groups — an iteration would wrap onto itself.
    _raises("K(3) < num_groups(4)",
            scene_seed_pool_size=3, num_groups=4)
    # K == num_groups with the DEFAULT max_groups=5 must be ACCEPTED. This is
    # the documented "every iteration directly comparable" recipe, and it is the
    # combination a bound of max(num_groups, max_groups) would wrongly reject:
    # min_alive_groups == 0 is mandatory here, so the collector's dynamic
    # extension (min_alive_groups > 0 and max_groups > num_groups) can never
    # fire and max_groups slots are unreachable.
    try:
        cfg_tight = _pool_config(scene_seed_pool_size=4, num_groups=4)
        check("K == num_groups(4) accepted with default max_groups=5 "
              "(dynamic extension is unreachable at min_alive_groups=0)",
              cfg_tight.scene_seed_pool_size == 4
              and cfg_tight.max_groups > cfg_tight.num_groups)
    except ValueError as exc:
        check("K == num_groups(4) accepted with default max_groups=5",
              False, f"wrongly rejected: {exc}")
    # min_alive_groups > 0 desynchronises the stateless cursor.
    _raises("min_alive_groups(2) != 0", min_alive_groups=2, max_groups=12)
    # K < 1 while non-zero: 0 is the only disabling value.
    _raises("K = -1 (negative, would silently disable)",
            scene_seed_pool_size=-1)
    _raises("K = 0.5 (fractional, would build an empty pool)",
            scene_seed_pool_size=0.5)

    # init_state_npz_path overrides the scene entirely → the pool would be inert.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz") as fh:
        _raises("init_state_npz_path set (pool would be silently inert)",
                init_state_npz_path=fh.name)
        # ... and the same npz is accepted with the pool OFF, proving the error
        # comes from the combination rather than from the path itself.
        try:
            GRPOConfig(device="cpu", init_state_npz_path=fh.name,
                       min_alive_groups=0)
            check("the same npz is still accepted with the pool disabled", True)
        except Exception as exc:                             # noqa: BLE001
            check("the same npz is still accepted with the pool disabled",
                  False, f"{type(exc).__name__}: {exc}")

    # Every error message must explain the WHY, not just the what — these are
    # the config errors an operator hits at 2am.
    try:
        _pool_config(scene_seed_pool_size=3, num_groups=4)
    except ValueError as exc:
        msg = str(exc)
        check("K < num_groups message names the group-relative-advantage reason",
              "group-relative" in msg and "same seed" in msg, msg[:120])
    try:
        _pool_config(min_alive_groups=2, max_groups=12)
    except ValueError as exc:
        msg = str(exc)
        check("min_alive_groups message names the stateless-cursor reason",
              "stateless" in msg and "num_groups" in msg, msg[:120])


def test_non_divisible_warning():
    """K not a multiple of num_groups WARNS (does not raise)."""
    print("\n[config] pass-alignment warning when K % num_groups != 0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = _pool_config(scene_seed_pool_size=10, num_groups=4)
    msgs = [str(w.message) for w in caught]
    check("a warning fires", len(msgs) == 1, f"{msgs}")
    check("it is about pass alignment, not correctness",
          msgs and "multiple of num_groups" in msgs[0]
          and "pass mean" in msgs[0], f"{msgs}")
    check("construction SUCCEEDS (warning, not error)",
          cfg.scene_seed_pool_size == 10)
    check("the pool still resolves normally", len(
        _trainer(cfg)._scene_seed_pool()) == 10)

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        _pool_config(scene_seed_pool_size=12, num_groups=4)
    check("no warning when K IS a multiple of num_groups",
          not [w for w in caught2 if "num_groups" in str(w.message)],
          f"{[str(w.message)[:60] for w in caught2]}")


# ---------------------------------------------------------------------------
# 4. Collector argv (enabled and disabled)
# ---------------------------------------------------------------------------


def test_disabled_path_argv_unchanged():
    """With the pool off, the collector argv gains nothing at all.

    Byte-identity matters because the whole point of the A/B is comparing a
    pooled run against a pre-feature baseline: if the disabled path changed even
    one argument, the baseline would no longer be the baseline.
    """
    print("\n[argv] pool disabled → argv is byte-identical (no --group-seeds)")
    # Built from the SAME helper as the enabled config with only the pool flag
    # flipped, so the argv delta below isolates the feature. Hand-rolling a
    # second GRPOConfig here would let an unrelated field (max_groups, say)
    # differ and show up as a spurious argv diff.
    off = _pool_config(scene_seed_pool_size=0)
    argv_off = _collector_argv(off, iteration=1)
    check("no --group-seeds token anywhere in the argv",
          "--group-seeds" not in argv_off,
          f"{[a for a in argv_off if 'seed' in a]}")
    check("--seed is still the legacy per-iteration value "
          "(seed + iteration * 100_000)",
          argv_off[argv_off.index("--seed") + 1] == "100067",
          f"{argv_off[argv_off.index('--seed') + 1]}")

    # Enabling the pool must add EXACTLY the two --group-seeds tokens plus the
    # --log-scene-fingerprint switch, and change nothing else. Same config
    # otherwise (min_alive_groups already 0).
    on = _pool_config()
    argv_on = _collector_argv(on, iteration=1)
    i = argv_on.index("--group-seeds")
    stripped = [
        a for a in (argv_on[:i] + argv_on[i + 2:])
        if a != "--log-scene-fingerprint"
    ]
    check("enabling the pool adds exactly --group-seeds <csv> and "
          "--log-scene-fingerprint, and touches nothing else",
          stripped == argv_off,
          f"diff: {set(map(str, stripped)) ^ set(map(str, argv_off))}")
    check("--log-scene-fingerprint is passed for a pooled run (the pool's "
          "premise is unverifiable without it)",
          "--log-scene-fingerprint" in argv_on)
    check("--log-scene-fingerprint is NOT passed when the pool is off",
          "--log-scene-fingerprint" not in argv_off)
    check("--seed is UNCHANGED by the pool (other trainer RNGs depend on it)",
          argv_on[argv_on.index("--seed") + 1] == "100067",
          f"{argv_on[argv_on.index('--seed') + 1]}")


def test_enabled_argv_carries_this_iterations_seeds():
    print("\n[argv] pool enabled → --group-seeds carries the cursor's output")
    cfg = _pool_config()
    for it, want in [(1, "100067,101067,102067,103067"),
                     (2, "104067,105067,106067,107067"),
                     (4, "100067,101067,102067,103067")]:
        argv = _collector_argv(cfg, iteration=it)
        got = argv[argv.index("--group-seeds") + 1]
        check(f"iteration {it} argv → --group-seeds {want}", got == want,
              f"{got}")

    # And the CLI string the trainer emits round-trips through the collector's
    # own parser back to the integer list the cursor produced. This is the only
    # place the two halves of the feature actually meet.
    argv = _collector_argv(cfg, iteration=2)
    round_tripped = ce._comma_separated_ints(
        argv[argv.index("--group-seeds") + 1]
    )
    check("the emitted string round-trips through the collector's parser",
          round_tripped == _trainer(cfg, 2)._scene_seeds_for_iteration(),
          f"{round_tripped}")


def test_group_seeds_cli_parser():
    """`--group-seeds` type function: happy path and the failure modes."""
    print("\n[cli] --group-seeds comma-separated parsing")
    check("plain list parses", ce._comma_separated_ints("1,2,3") == [1, 2, 3])
    check("whitespace and a trailing comma are tolerated",
          ce._comma_separated_ints(" 100067, 101067 ,") == [100067, 101067])
    check("a single value parses", ce._comma_separated_ints("42") == [42])
    for bad in ("1,x,3", "", ",", "1.5"):
        try:
            ce._comma_separated_ints(bad)
            check(f"{bad!r} rejected", False, "no error raised")
        except Exception as exc:                             # noqa: BLE001
            check(f"{bad!r} rejected ({type(exc).__name__})",
                  "ArgumentTypeError" in type(exc).__name__,
                  f"got {type(exc).__name__}")


# ---------------------------------------------------------------------------
# 5. Collector consumption
# ---------------------------------------------------------------------------


def test_resolve_group_seed_default_derivation():
    """No group_seeds → the OLD base_seed + group_idx * STRIDE formula, exactly."""
    print("\n[collector] default derivation is unchanged (pure function)")
    base = 100_067
    got = [ce.resolve_group_seed(g, base) for g in range(6)]
    want = [base + g * ce.GROUP_SEED_STRIDE for g in range(6)]
    check("resolve_group_seed(g, base) == base + g * GROUP_SEED_STRIDE",
          got == want, f"{got}")
    check("explicit None is the same as omitting the argument",
          [ce.resolve_group_seed(g, base, None) for g in range(6)] == want)


def test_resolve_group_seed_pool_and_overrun():
    print("\n[collector] explicit seeds are used verbatim; over-run RAISES")
    seeds = [100_067, 101_067, 102_067, 103_067]
    got = [ce.resolve_group_seed(g, 999_999, seeds) for g in range(4)]
    check("group N gets group_seeds[N] verbatim", got == seeds, f"{got}")
    check("base_seed is IGNORED when explicit seeds are supplied",
          999_999 not in got)
    try:
        ce.resolve_group_seed(4, 999_999, seeds)
        check("group_idx past the list raises (does NOT wrap)", False,
              f"returned {ce.resolve_group_seed(4, 999_999, seeds)}")
    except IndexError as exc:
        check("group_idx past the list raises IndexError (does NOT wrap)", True)
        check("the error explains why wrapping is refused",
              "wrap" in str(exc) and "group-relative" in str(exc),
              str(exc)[:120])


def _drive_collect(group_size, num_envs, num_groups, base_seed,
                   group_seeds=None, max_groups=None, min_alive_groups=0):
    """Run the REAL EpisodeCollector.collect over test_turn_collection's fakes.

    Returns {group_id: set_of_env_seeds} so the caller can assert which scene
    each group actually reset with.
    """
    c = tc._make_collector(group_size=group_size, num_async_vector_env=num_envs)
    try:
        eps = c.collect(
            num_groups=num_groups,
            base_seed=base_seed,
            fast_forward_steps=0,        # FF off → plain seed-aligned groups
            fast_forward_pct=0.0,
            min_alive_groups=min_alive_groups,
            max_groups=max_groups,
            group_seeds=group_seeds,
        )
    finally:
        c.close()
    out: dict = {}
    for e in eps:
        out.setdefault(e["group_id"], set()).add(e["env_seed"])
    return out


def test_collector_consumes_group_seeds():
    """End-to-end through the real group loop: group N resets with seeds[N]."""
    print("\n[collector] collect(group_seeds=...) drives env.reset per group")
    seeds = [100_067, 101_067, 102_067]
    by_group = _drive_collect(group_size=2, num_envs=2, num_groups=3,
                              base_seed=555_000, group_seeds=seeds)
    check("one entry per group", sorted(by_group) == [0, 1, 2], f"{sorted(by_group)}")
    for g, want in enumerate(seeds):
        check(f"group {g} reset with group_seeds[{g}] == {want}",
              by_group[g] == {want}, f"{by_group[g]}")
    check("the default base_seed never leaks into any group",
          all(555_000 not in s for s in by_group.values()), f"{by_group}")

    # ... and without group_seeds the same driver reproduces the old formula.
    baseline = _drive_collect(group_size=2, num_envs=2, num_groups=3,
                              base_seed=555_000)
    check("group_seeds=None → base_seed + g * GROUP_SEED_STRIDE (unchanged)",
          baseline == {g: {555_000 + g * ce.GROUP_SEED_STRIDE} for g in range(3)},
          f"{baseline}")


def test_collector_rejects_short_seed_list():
    """A seed list shorter than the reachable group count is rejected up front."""
    print("\n[collector] a too-short --group-seeds list is rejected up front")
    try:
        _drive_collect(group_size=2, num_envs=2, num_groups=3,
                       base_seed=1000, group_seeds=[1, 2])
        check("collect() with 2 seeds for 3 groups raises", False,
              "no error raised")
    except ValueError as exc:
        check("collect() with 2 seeds for 3 groups raises ValueError", True)
        check("the error reports how many groups the call can reach",
              "can collect up to 3 groups" in str(exc), str(exc)[:140])
    except Exception as exc:                                 # noqa: BLE001
        check("collect() with 2 seeds for 3 groups raises ValueError", False,
              f"got {type(exc).__name__}: {exc}")

    # Under DYNAMIC collection the bound rises to max_groups: enough seeds for
    # num_groups but not for an extension must still be rejected, or the failure
    # would only surface on the iterations that actually extend.
    try:
        _drive_collect(group_size=2, num_envs=2, num_groups=2, max_groups=4,
                       min_alive_groups=1, base_seed=1000, group_seeds=[1, 2])
        check("dynamic mode: 2 seeds with num_groups=2, max_groups=4 raises",
              False, "no error raised")
    except ValueError as exc:
        check("dynamic mode: 2 seeds with num_groups=2, max_groups=4 raises "
              "ValueError", True)
        check("the dynamic-mode error bound is max_groups (4), not num_groups",
              "can collect up to 4 groups" in str(exc), str(exc)[:140])

    # REGRESSION (bug found while implementing): with dynamic mode OFF the loop
    # stops at num_groups, so num_groups seeds is SUFFICIENT even when
    # max_groups is larger. The DEFAULT GRPOConfig is exactly this shape
    # (num_groups=3, max_groups=5, min_alive_groups=0) and the trainer sends
    # num_groups seeds, so an over-strict `len < max_groups` check would have
    # failed EVERY iteration of the default pooled config.
    try:
        by_group = _drive_collect(group_size=2, num_envs=2, num_groups=3,
                                  max_groups=5, min_alive_groups=0,
                                  base_seed=1000, group_seeds=[11, 22, 33])
        check("dynamic mode OFF: num_groups(3) seeds accepted despite "
              "max_groups=5 (the default config's shape)",
              by_group == {0: {11}, 1: {22}, 2: {33}}, f"{by_group}")
    except Exception as exc:                                 # noqa: BLE001
        check("dynamic mode OFF: num_groups(3) seeds accepted despite "
              "max_groups=5", False, f"{type(exc).__name__}: {exc}")


def test_trainer_argv_feeds_collector_end_to_end():
    """The trainer's argv, parsed by the collector, must actually collect.

    The two halves of the feature are wired through a CLI string across a venv
    boundary, so neither half's unit tests can see a MISMATCH between how many
    seeds the trainer sends (num_groups) and how many the collector demands.
    This closes the loop: build the real argv, feed the real parser, and drive
    the real `collect()` with the parsed values.

    Run for the DEFAULT config shape (num_groups=3, max_groups=5,
    min_alive_groups=0) as well as the tidy K-aligned one — the default shape is
    where an over-strict collector-side length check fails every iteration.
    """
    print("\n[e2e] trainer argv → collector parser → collect(), both shapes")
    for label, cfg in (
        ("default shape (num_groups=3, max_groups=5)",
         _pool_config(scene_seed_pool_size=12, num_groups=3, max_groups=5)),
        ("aligned shape (num_groups=4, max_groups=4)",
         _pool_config(scene_seed_pool_size=12, num_groups=4)),
    ):
        argv = _collector_argv(cfg, iteration=2)

        def _val(flag):
            return argv[argv.index(flag) + 1]

        seeds = ce._comma_separated_ints(_val("--group-seeds"))
        check(f"{label}: trainer sends num_groups seeds",
              len(seeds) == cfg.num_groups, f"{len(seeds)} vs {cfg.num_groups}")
        try:
            by_group = _drive_collect(
                group_size=2, num_envs=2,
                num_groups=int(_val("--num-groups")),
                base_seed=int(_val("--seed")),
                group_seeds=seeds,
                max_groups=int(_val("--max-groups")),
                min_alive_groups=int(_val("--min-alive-groups")),
            )
            check(f"{label}: collect() accepts the argv and uses every seed "
                  f"in order",
                  by_group == {g: {s} for g, s in enumerate(seeds)},
                  f"{by_group}")
        except Exception as exc:                             # noqa: BLE001
            check(f"{label}: collect() accepts the argv", False,
                  f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 6. Per-scene stats + TB emission
# ---------------------------------------------------------------------------


def _buffer(groups: list[tuple[int, list[bool]]]) -> EpisodeBuffer:
    """One group per (env_seed, outcomes) pair. Mirrors the fake-episode shape
    used by episode_buffer.py's own __main__ self-test."""
    b = EpisodeBuffer()
    for gid, (seed, outcomes) in enumerate(groups):
        for succ in outcomes:
            b.episodes.append(GRPOEpisode(
                video_frames=[{}], states=[{}], language="t",
                actions=[np.zeros((16, 12))],
                raw_actions=[np.zeros((50, 128))],
                action_masks=[np.ones((50, 128))],
                initial_noises=[np.zeros((50, 128))],
                success=succ, shaped_reward=0.0, env_name="t",
                episode_idx=len(b.episodes), num_steps=100,
                group_id=gid, env_seed=seed,
            ))
    return b


def test_per_scene_success_stats():
    print("\n[buffer] stats()['per_scene_success'] is exact")
    b = _buffer([
        (100_067, [True, True, False, False]),   # mixed 2/4
        (101_067, [False] * 4),                  # ZERO successes
        (102_067, [True] * 4),                   # ALL successes
        (103_067, [True]),                       # singleton
    ])
    b.compute_advantages()
    per_scene = b.stats()["per_scene_success"]
    want = {100_067: (2, 4), 101_067: (0, 4), 102_067: (4, 4), 103_067: (1, 1)}
    check("per_scene_success == exact (n_success, n_total) per seed",
          per_scene == want, f"{per_scene}")
    check("a 0-success scene is PRESENT with (0, n), not omitted",
          per_scene[101_067] == (0, 4), f"{per_scene.get(101_067)}")
    check("an all-success scene reads (n, n)",
          per_scene[102_067] == (4, 4), f"{per_scene.get(102_067)}")
    check("counts sum to the buffer's episode count",
          sum(t for _, t in per_scene.values()) == b.num_episodes)
    check("successes sum to the buffer's success count",
          sum(s for s, _ in per_scene.values())
          == sum(1 for e in b.episodes if e.success))

    # Two group_ids sharing one env_seed POOL into one entry — the dict
    # accumulate is what makes that correct rather than last-write-wins.
    shared = _buffer([(700, [True, False]), (700, [True, True])])
    shared.compute_advantages()
    check("two groups on one seed pool into a single (2+1, 4) entry",
          shared.stats()["per_scene_success"] == {700: (3, 4)},
          f"{shared.stats()['per_scene_success']}")

    # The pre-existing per-GROUP curves must be untouched by the addition.
    st = b.stats()
    check("group_success_min/median/max unchanged by the new accumulation",
          (st["group_success_min"], st["group_success_max"]) == (0.0, 1.0),
          f"{st['group_success_min']}, {st['group_success_max']}")


class _RecordingWriter:
    """Captures add_scalar so the emission side can be asserted."""

    def __init__(self):
        self.calls: list = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, float(value), step))

    def add_text(self, *a, **kw):
        pass


def _emit(cfg: GRPOConfig, stats: dict, iteration: int = 4) -> dict:
    """Run the real _log_metrics against a recording writer; return {tag: value}."""
    tr = _trainer(cfg, iteration)
    tr.writer = _RecordingWriter()
    tr._ref_mse_stats = None
    tr._chunk_gap_stats = None
    tg.GRPOTrainer._log_metrics(tr, iteration, stats, update_stats=None,
                                lr=1e-5, iter_time=1.0)
    return {t: v for t, v, _ in tr.writer.calls}


def test_scene_sr_emission_gated_on_pool():
    print("\n[logging] episode/scene_sr/* + pool_pass are gated on the pool")
    b = _buffer([
        (100_067, [True, True, False, False]),
        (101_067, [False] * 4),
        (102_067, [True] * 4),
    ])
    b.compute_advantages()
    stats = b.stats()

    on = _emit(_pool_config(), stats, iteration=4)
    for seed, want in ((100_067, 0.5), (101_067, 0.0), (102_067, 1.0)):
        tag = f"episode/scene_sr/{seed}"
        check(f"emits {tag} == {want}",
              tag in on and abs(on[tag] - want) < 1e-9, f"got {on.get(tag)!r}")
    check("emits episode/pool_pass (iteration 4, K=12, ng=4 → pass 1)",
          on.get("episode/pool_pass") == 1.0, f"{on.get('episode/pool_pass')}")

    off = _emit(GRPOConfig(device="cpu", num_groups=4, max_groups=4,
                           min_alive_groups=0), stats, iteration=4)
    leaked = [t for t in off if "scene_sr" in t or "pool_pass" in t]
    check("pool OFF: no scene_sr/* and no pool_pass curves (a fresh seed set "
          "every iteration would mean 4 new single-point series per iter)",
          not leaked, f"{leaked}")
    check("pool OFF: the pre-existing episode/* key set is otherwise identical",
          {t for t in off if t.startswith("episode/")}
          == {t for t in on if t.startswith("episode/")
              and "scene_sr" not in t and t != "episode/pool_pass"},
          f"{sorted(set(off) ^ set(on))}")

    # Routed through _emit → the non-finite/non-numeric filter applies. A
    # hand-corrupted entry must cost that one scalar, not the iteration.
    bad_stats = dict(stats)
    bad_stats["per_scene_success"] = dict(stats["per_scene_success"])
    bad_stats["per_scene_success"][999] = (float("nan"), 1)
    try:
        got = _emit(_pool_config(), bad_stats, iteration=4)
        check("a non-finite per-scene rate is DROPPED, not written, and the "
              "other scenes still emit",
              "episode/scene_sr/999" not in got
              and "episode/scene_sr/100067" in got,
              f"{[t for t in got if 'scene_sr' in t]}")
    except Exception as exc:                                 # noqa: BLE001
        check("a non-finite per-scene rate is dropped rather than raising",
              False, f"{type(exc).__name__}: {exc}")


def test_wandb_payload_carries_no_dict_value():
    """`per_scene_success` is the only non-scalar stats() entry — it must never
    reach wandb.log as a dict of tuples, pool on or off."""
    print("\n[logging] wandb payload: per_scene_success is flattened or dropped")
    import types
    b = _buffer([(100_067, [True, False]), (101_067, [False, False])])
    b.compute_advantages()
    stats = b.stats()

    fake_wandb = types.ModuleType("wandb")
    sent: list = []
    fake_wandb.log = lambda d: sent.append(dict(d))
    sys.modules["wandb"] = fake_wandb
    try:
        for cfg, label in ((_pool_config(use_wandb=True), "pool on"),
                           (GRPOConfig(device="cpu", use_wandb=True,
                                       num_groups=4, max_groups=4,
                                       min_alive_groups=0), "pool off")):
            sent.clear()
            tr = _trainer(cfg, 4)
            tr.writer = _RecordingWriter()
            tr._ref_mse_stats = None
            tr._chunk_gap_stats = None
            tg.GRPOTrainer._log_metrics(tr, 4, stats, update_stats=None)
            payload = sent[0] if sent else {}
            check(f"{label}: 'per_scene_success' is not a payload key",
                  "per_scene_success" not in payload,
                  f"{[k for k in payload if 'scene' in k]}")
            check(f"{label}: every payload value is a scalar",
                  all(not isinstance(v, (dict, list, tuple))
                      for v in payload.values()),
                  f"{[k for k, v in payload.items() if isinstance(v, (dict, list, tuple))]}")
            flat = {k for k in payload if k.startswith("episode/scene_sr/")}
            if label == "pool on":
                check("pool on: flattened per-scene rates ARE in the payload",
                      flat == {"episode/scene_sr/100067",
                               "episode/scene_sr/101067"}, f"{flat}")
                check("pool on: episode/pool_pass is in the payload",
                      "episode/pool_pass" in payload)
            else:
                check("pool off: no per-scene keys in the payload",
                      not flat and "episode/pool_pass" not in payload,
                      f"{flat}")
    finally:
        sys.modules.pop("wandb", None)


# ---------------------------------------------------------------------------
# 6. Scene fingerprint (the pool's premise-check diagnostic)
# ---------------------------------------------------------------------------


def _bundle(layout=3, style=7, xml="<mujoco>kitchen A</mujoco>", state=(1.0, 2.0)):
    return {
        "ep_meta": {"layout_id": layout, "style_id": style, "lang": "serve mug"},
        "model_xml": xml,
        "sim_state": np.asarray(state, dtype=np.float64),
    }


def test_scene_fingerprint_discriminates():
    """The three parts must move independently, or the readout is unusable.

    The whole point is telling "same kitchen, different placements" (xml equal,
    state differs) from "not the same scene at all" (xml differs). If a sim_state
    change perturbed the xml digest, every pass would look like a fresh kitchen.
    """
    print("\n[fingerprint] layout/style/xml/state each move independently")
    base = ce.scene_fingerprint(_bundle())
    check("reports layout, style, xml and state",
          all(k in base for k in ("layout=3", "style=7", "xml=", "state=")), base)
    check("identical bundles → identical fingerprint",
          ce.scene_fingerprint(_bundle()) == base)

    def _part(fp, key):
        return next(p for p in fp.split() if p.startswith(key + "="))

    diff_state = ce.scene_fingerprint(_bundle(state=(1.0, 2.5)))
    check("a sim_state change moves ONLY state= (same kitchen, new placements)",
          _part(diff_state, "state") != _part(base, "state")
          and _part(diff_state, "xml") == _part(base, "xml"),
          f"{base} vs {diff_state}")

    diff_xml = ce.scene_fingerprint(_bundle(xml="<mujoco>kitchen B</mujoco>"))
    check("a model_xml change moves ONLY xml= (the pool-is-broken signal)",
          _part(diff_xml, "xml") != _part(base, "xml")
          and _part(diff_xml, "state") == _part(base, "state"),
          f"{base} vs {diff_xml}")

    diff_layout = ce.scene_fingerprint(_bundle(layout=9))
    check("a layout_id change is visible without comparing hashes",
          "layout=9" in diff_layout)

    # Memory layout must not masquerade as a scene change: a non-contiguous view
    # and a float32 array of the same values have to digest identically, or the
    # diagnostic invents the exact false alarm it exists to rule out.
    vals = np.asarray([1.0, 2.0], dtype=np.float64)
    noncontig = np.asarray([[1.0, 9.0], [2.0, 9.0]], dtype=np.float64)[:, 0]
    check("non-contiguous view of the same values digests identically",
          ce.scene_fingerprint(_bundle(state=noncontig))
          == ce.scene_fingerprint(_bundle(state=vals)),
          f"{noncontig.flags['C_CONTIGUOUS']}")
    check("float32 of exactly-representable values digests identically",
          ce.scene_fingerprint(_bundle(state=vals.astype(np.float32)))
          == ce.scene_fingerprint(_bundle(state=vals)))


def test_scene_fingerprint_never_raises():
    """Malformed input must cost the diagnostic, not 22 minutes of rollouts."""
    print("\n[fingerprint] degenerate bundles degrade, never raise")
    for label, arg in [
        ("None", None),
        ("not a dict", "nonsense"),
        ("empty dict", {}),
        ("ep_meta is not a dict", {"ep_meta": ["oops"], "model_xml": "x",
                                   "sim_state": np.zeros(2)}),
        ("sim_state is None", {"ep_meta": {}, "model_xml": "x",
                               "sim_state": None}),
        ("sim_state is unhashable junk", {"ep_meta": {}, "model_xml": "x",
                                          "sim_state": object()}),
        ("model_xml is bytes", {"ep_meta": {}, "model_xml": b"\xff\xfe",
                                "sim_state": np.zeros(2)}),
    ]:
        try:
            out = ce.scene_fingerprint(arg)
            check(f"{label} → returns a string ({out!r})", isinstance(out, str)
                  and out != "")
        except Exception as exc:                                 # noqa: BLE001
            check(f"{label} → returns a string", False,
                  f"raised {type(exc).__name__}: {exc}")
    check("a non-dict bundle reports n/a (singleton path captures none)",
          ce.scene_fingerprint(None) == "n/a")


def test_collector_publishes_fingerprint():
    """The collector fingerprints the PRISTINE branch point, with no extra RPC.

    Two properties that a naive wiring gets wrong: hashing `bundles[0]` after
    apply_scene_bundle has mutated its ep_meta in place (which would make the
    same scene read differently depending on measurement order), and leaving a
    previous group's fingerprint in place on a path that captures no bundle
    (which would attribute one group's scene to another).
    """
    print("\n[fingerprint] collector publishes it off the captured bundle")
    c = tc._make_collector(group_size=2, num_async_vector_env=2)
    try:
        c.log_scene_fingerprint = True
        before = len([m for m, _ in c.envs.calls if m == "get_scene_bundle"])
        c._align_envs_to_group_scene(100_067)
        after = len([m for m, _ in c.envs.calls if m == "get_scene_bundle"])
        check("no EXTRA get_scene_bundle RPC (reuses the captured bundle)",
              after - before == 1, f"{after - before} calls")
        check("a fingerprint is published for the group",
              isinstance(c._last_scene_fingerprint, str)
              and c._last_scene_fingerprint != "",
              f"{c._last_scene_fingerprint!r}")

        # A path that captures no bundle must CLEAR the previous value.
        c.group_size = 1
        c._align_envs_to_group_scene(101_067)
        check("singleton path clears the stale fingerprint (no misattribution)",
              c._last_scene_fingerprint is None,
              f"{c._last_scene_fingerprint!r}")
    finally:
        c.close()

    # Flag off → nothing computed at all, so the log line stays byte-identical.
    c2 = tc._make_collector(group_size=2, num_async_vector_env=2)
    try:
        check("flag off → default is off", c2.log_scene_fingerprint is False)
        c2._align_envs_to_group_scene(100_067)
        check("flag off → no fingerprint computed",
              c2._last_scene_fingerprint is None,
              f"{c2._last_scene_fingerprint!r}")
    finally:
        c2.close()


def test_fingerprint_appears_in_group_log_line():
    """It has to reach stdout next to the seed — that is the whole deliverable."""
    print("\n[fingerprint] shows up in the per-group log line, gated on the flag")
    import io
    from contextlib import redirect_stdout

    def _run(flag: bool) -> str:
        c = tc._make_collector(group_size=2, num_async_vector_env=2)
        c.log_scene_fingerprint = flag
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                c.collect(num_groups=2, base_seed=555_000,
                          fast_forward_steps=0, fast_forward_pct=0.0,
                          min_alive_groups=0, max_groups=None,
                          group_seeds=[100_067, 101_067])
        finally:
            c.close()
        return buf.getvalue()

    on = _run(True)
    group_lines = [l for l in on.splitlines() if "(seed=" in l]
    check("every group line carries the fingerprint next to its seed",
          len(group_lines) == 2
          and all("xml=" in l and "state=" in l for l in group_lines),
          f"{group_lines}")
    check("the fingerprint sits inside the same parenthesis as the seed",
          all(l.split("(seed=")[1].split(")")[0].startswith(("100067 ", "101067 "))
              for l in group_lines),
          f"{group_lines}")

    off = _run(False)
    off_lines = [l for l in off.splitlines() if "(seed=" in l]
    check("flag off → group line has seed only (byte-identical to before)",
          len(off_lines) == 2
          and all("xml=" not in l and "state=" not in l for l in off_lines),
          f"{off_lines}")
    check("flag off → seed parenthesis closes immediately after the number",
          all(l.split("(seed=")[1].split(")")[0].isdigit() for l in off_lines),
          f"{off_lines}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_stride_constants_agree()
    test_pool_resolution()
    test_cursor_and_pass_alignment()
    test_distinct_within_iteration()
    test_validations_raise()
    test_non_divisible_warning()
    test_disabled_path_argv_unchanged()
    test_enabled_argv_carries_this_iterations_seeds()
    test_group_seeds_cli_parser()
    test_resolve_group_seed_default_derivation()
    test_resolve_group_seed_pool_and_overrun()
    test_collector_consumes_group_seeds()
    test_collector_rejects_short_seed_list()
    test_trainer_argv_feeds_collector_end_to_end()
    test_per_scene_success_stats()
    test_scene_sr_emission_gated_on_pool()
    test_wandb_payload_carries_no_dict_value()
    test_scene_fingerprint_discriminates()
    test_scene_fingerprint_never_raises()
    test_collector_publishes_fingerprint()
    test_fingerprint_appears_in_group_log_line()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} test(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\033[32mAll scene-seed-pool tests passed.\033[0m")
