"""Tests for the time/* phase-timing instrumentation in train_grpo.py.

Covers the three curves added alongside the render-skipping work:
  * time/ref_logprob_seconds  — Phase 2b, previously untimed (~10% of an iter)
  * time/collect_rollout_seconds / time/collect_load_seconds — the two halves of
    Phase 1 (collector subprocess vs trainer-side npz read-back), which are
    indistinguishable on the aggregate time/collect_seconds curve.

The headline test drives the REAL `GRPOTrainer.train()` loop with stubbed phases,
so removing a `phase_times` entry from either log site fails the suite; the
sub-phase timers are checked with measurable sleeps so that moving a `time.time()`
call to the wrong side of the work it measures also fails.

Runs without a GPU or the robocasa venv:

    .venv/bin/python scripts/grpo/test_phase_timing_logs.py
"""

import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grpo_config import GRPOConfig  # noqa: E402
from train_grpo import GRPOTrainer  # noqa: E402

STATS = {"success_rate": 0.1, "mean_reward": 0.1, "std_reward": 0.3}

# Sleeps long enough to dominate scheduler noise, short enough to keep the suite
# fast. Asserted against half their value so the tests aren't flaky.
# Stale value pre-seeded into the timer fields by the fixtures, so a test
# asserting NaN/freshness fails if production forgets to set them.
STALE_SENTINEL = 999.0

ROLLOUT_SLEEP = 0.30
LOAD_SLEEP = 0.15


class FakeWriter:
    def __init__(self):
        self.scalars = {}          # tag -> value (last write wins)
        self.steps = {}            # tag -> step it was logged against

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = value
        self.steps[tag] = step

    def add_text(self, *args, **kwargs):
        pass


class FakeEpisode:
    """Enough shape for the cached-load FM-completeness check and the
    partial-collection warning (which groups episodes by group_id)."""

    def __init__(self, n_chunks, group_id=0):
        self.raw_actions = [object()] * n_chunks
        self.action_masks = [object()] * n_chunks
        self.initial_noises = [object()] * n_chunks
        self.group_id = group_id


class FakeBuffer:
    """Stands in for EpisodeBuffer: only the calls the trainer makes."""

    def __init__(self, n_loaded=48, chunks_per_episode=45, load_sleep=0.0):
        self.n_loaded = n_loaded
        self.chunks_per_episode = chunks_per_episode
        self.load_sleep = load_sleep
        self.episodes = []
        self.num_chunks = 0
        self.cleared = 0

    def clear(self):
        self.cleared += 1
        self.episodes = []
        self.num_chunks = 0

    def load_episodes(self, episode_dir):
        time.sleep(self.load_sleep)
        # Spread across group_size-sized groups so _warn_partial_collection
        # sees a complete collection and stays quiet.
        self.episodes = [
            FakeEpisode(self.chunks_per_episode, group_id=i // 8)
            for i in range(self.n_loaded)
        ]
        self.num_chunks = self.n_loaded * self.chunks_per_episode
        return self.n_loaded

    def compute_advantages(self, max_episode_steps=None):
        pass

    def stats(self):
        return dict(STATS)


def _bare_trainer(load_sleep=0.0, **config_kwargs):
    """A GRPOTrainer with just enough state for the methods under test —
    __init__/setup would load a 3B model."""
    trainer = object.__new__(GRPOTrainer)
    trainer.config = GRPOConfig(use_wandb=False, **config_kwargs)
    trainer.writer = FakeWriter()
    trainer.iteration = 1
    trainer.buffer = FakeBuffer(load_sleep=load_sleep)
    trainer._consecutive_collect_failures = 0
    trainer._max_consecutive_collect_failures = 3
    # Seed a stale FINITE sentinel, never NaN: assertions that a timer "is NaN"
    # would otherwise pass whether or not production actually reset it.
    trainer._collect_rollout_time = STALE_SENTINEL
    trainer._collect_load_time = STALE_SENTINEL
    return trainer


def _time_tags(trainer):
    return {
        k: v for k, v in trainer.writer.scalars.items() if k.startswith("time/")
    }


# ---------------------------------------------------------------------------
# End-to-end: the real train() loop, stubbed phases
# ---------------------------------------------------------------------------

EXPECTED_NORMAL_TAGS = {
    "time/iteration_seconds",
    "time/collect_seconds",
    "time/collect_rollout_seconds",
    "time/collect_load_seconds",
    "time/advantage_seconds",
    "time/ref_logprob_seconds",
    "time/update_seconds",
}


def _stub_trainer_for_train_loop(tmp, cached=False, num_iterations=1, **config_kwargs):
    """Stub every phase but keep the real train() loop and _log_metrics."""
    kwargs = dict(
        num_iterations=num_iterations,
        save_interval=1_000_000,   # never checkpoint
        episode_dir=tmp,
        **config_kwargs,
    )
    trainer = _bare_trainer(**kwargs)
    trainer._start_iteration = 1
    trainer._last_updated_iteration = 0
    trainer.optimizer = type(
        "Opt", (), {"param_groups": [{"lr": 1e-5}]}
    )()

    calls = []
    trainer._log_mem_snapshot = lambda *a, **k: None
    trainer._release_memory_to_os = lambda: None
    trainer._collect_episodes = lambda *a, **k: (
        calls.append("collect"),
        setattr(trainer, "_collect_rollout_time", 1.5),
        setattr(trainer, "_collect_load_time", 0.25),
    )[0]
    trainer._load_cached_episodes = lambda: (
        calls.append("cached"),
        setattr(trainer, "_collect_load_time", 0.25),
    )[0]
    trainer._vram_snapshot = lambda **k: None
    trainer._log_vram = lambda *a, **k: None
    trainer._compute_ref_log_probs = lambda: calls.append("ref")
    trainer._grpo_update = lambda: {"n_updates": 3, "n_micro_batches": 6}
    trainer._compute_lora_delta_norm = lambda: 1.0
    trainer._save_checkpoint = lambda *a, **k: None
    trainer._save_checkpoint_for_skipped_iter = lambda *a, **k: None
    trainer.calls = calls
    if cached:
        trainer.config.resume_from_collected_data = True
    return trainer


def test_train_loop_emits_every_phase_curve():
    """Drives the real loop: catches a phase_times entry dropped at the call site."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _stub_trainer_for_train_loop(tmp)
        trainer.train()
        tags = _time_tags(trainer)
        assert set(tags) == EXPECTED_NORMAL_TAGS, (
            f"missing/extra time curves: "
            f"{set(tags) ^ EXPECTED_NORMAL_TAGS}"
        )
        assert trainer.calls == ["collect", "ref"], trainer.calls
        # Every scalar must land on iteration 1, not some stale index.
        assert set(trainer.writer.steps[t] for t in tags) == {1}, trainer.writer.steps
        # The sub-phases came from the collection method, not from thin air.
        assert tags["time/collect_rollout_seconds"] == 1.5
        assert tags["time/collect_load_seconds"] == 0.25
    print(f"  [PASS] real train() emits all {len(EXPECTED_NORMAL_TAGS)} time/* curves")


def test_train_loop_cached_iter_gaps_rollout_only():
    """On a resumed iter, collect/collect_rollout must be gaps while the load
    that actually ran is logged."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "iter_0001").mkdir(parents=True)
        trainer = _stub_trainer_for_train_loop(tmp, cached=True)
        trainer.train()
        tags = _time_tags(trainer)
        assert trainer.calls == ["cached", "ref"], trainer.calls
        assert "time/collect_seconds" not in tags, tags
        assert "time/collect_rollout_seconds" not in tags, tags
        assert tags["time/collect_load_seconds"] == 0.25, tags
        assert "time/ref_logprob_seconds" in tags, tags
    print("  [PASS] real train() gaps only the sub-phases that didn't run")


# ---------------------------------------------------------------------------
# _log_metrics behavior
# ---------------------------------------------------------------------------


def test_nan_subphases_are_skipped():
    trainer = _bare_trainer()
    trainer._log_metrics(
        2,
        stats={},
        update_stats=None,
        lr=1e-5,
        iter_time=300.0,
        phase_times={
            "collect": float("nan"),
            "collect_rollout": float("nan"),
            "collect_load": 12.0,
            "advantage": 0.04,
        },
    )
    tags = _time_tags(trainer)
    assert "time/collect_seconds" not in tags, tags
    assert "time/collect_rollout_seconds" not in tags, tags
    assert tags["time/collect_load_seconds"] == 12.0, tags
    assert set(trainer.writer.steps.values()) == {2}, trainer.writer.steps
    print("  [PASS] NaN sub-phases skipped (clean gap, no misleading zero)")


# ---------------------------------------------------------------------------
# Phase 1 sub-phase timers
# ---------------------------------------------------------------------------


def test_collect_episodes_times_the_right_calls():
    """Sleeps make the assertion sensitive to WHERE the timers are taken."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _bare_trainer(episode_dir=tmp, load_sleep=LOAD_SLEEP)

        def fake_subprocess(env_name, episode_dir, max_steps, ff_steps):
            time.sleep(ROLLOUT_SLEEP)
            return None

        trainer._collect_via_subprocess = fake_subprocess
        trainer._collect_episodes("robocasa_panda_omron/Fake_Env", 0, 480)

        rollout, load = trainer._collect_rollout_time, trainer._collect_load_time
        assert rollout >= ROLLOUT_SLEEP * 0.5, (
            f"rollout timer {rollout:.3f}s missed a {ROLLOUT_SLEEP}s subprocess"
        )
        assert load >= LOAD_SLEEP * 0.5, (
            f"load timer {load:.3f}s missed a {LOAD_SLEEP}s load"
        )
        # Each timer must exclude the other's work.
        assert rollout < ROLLOUT_SLEEP + LOAD_SLEEP, (
            f"rollout timer {rollout:.3f}s appears to include the load"
        )
        assert load < ROLLOUT_SLEEP, (
            f"load timer {load:.3f}s appears to include the rollout"
        )
    print(
        f"  [PASS] rollout={rollout:.2f}s / load={load:.2f}s attributed to the "
        f"right calls"
    )


def test_failed_collection_leaves_load_time_nan():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _bare_trainer(episode_dir=tmp)
        trainer._collect_via_subprocess = (
            lambda env_name, episode_dir, max_steps, ff_steps: "non-zero exit code 1"
        )
        trainer._collect_episodes("robocasa_panda_omron/Fake_Env", 0, 480)

        assert math.isfinite(trainer._collect_rollout_time), (
            "the subprocess did run, so its time should be recorded"
        )
        assert trainer._collect_load_time != STALE_SENTINEL, (
            "load timer was never touched — the stale pre-seeded value survived"
        )
        assert math.isnan(trainer._collect_load_time), (
            "no load ran; expected the NaN sentinel"
        )
        assert trainer._consecutive_collect_failures == 1
    print("  [PASS] failed collection: rollout timed, load left as NaN")


def test_cached_load_marks_rollout_nan():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "iter_0001").mkdir(parents=True)
        trainer = _bare_trainer(episode_dir=tmp, load_sleep=LOAD_SLEEP)
        trainer._load_cached_episodes()

        assert trainer._collect_rollout_time != STALE_SENTINEL, (
            "rollout timer was never touched — the stale pre-seeded value survived"
        )
        assert math.isnan(trainer._collect_rollout_time), (
            "no rollouts ran; expected the NaN sentinel"
        )
        assert trainer._collect_load_time >= LOAD_SLEEP * 0.5, (
            f"load timer {trainer._collect_load_time:.3f}s missed the load"
        )
    print("  [PASS] cached load: rollout NaN, load timed")


def test_timers_not_carried_across_iterations_by_a_subclass():
    """The reset lives in the train loop, so a subclass overriding
    _collect_episodes (toy_train_grpo.py does) cannot leave last iteration's
    numbers to be re-logged as this iteration's."""
    with tempfile.TemporaryDirectory() as tmp:

        class ToyLikeTrainer(GRPOTrainer):
            def _collect_episodes(self, env_name, task_idx, max_steps):
                self.buffer.load_episodes(Path(tmp))   # no timer bookkeeping

        trainer = _stub_trainer_for_train_loop(tmp, num_iterations=2)
        trainer.__class__ = ToyLikeTrainer
        del trainer._collect_episodes          # unshadow the class method
        trainer._collect_rollout_time = 999.0  # stale values from a prior iter
        trainer._collect_load_time = 888.0
        trainer.train()

        tags = _time_tags(trainer)
        assert "time/collect_rollout_seconds" not in tags, (
            f"stale rollout time re-logged: {tags}"
        )
        assert "time/collect_load_seconds" not in tags, (
            f"stale load time re-logged: {tags}"
        )
        assert "time/collect_seconds" in tags, tags
    print("  [PASS] stale sub-phase timers can't leak into a later iteration")


def test_collector_cli_flag_only_appended_when_disabled():
    """The opt-out must cross the CLI boundary in the right direction.

    Inverting this condition would silently run every collection in the opposite
    render mode from what the config says, with no other symptom.
    """
    import inspect

    from collect_episodes import parse_args

    with tempfile.TemporaryDirectory() as tmp:
        for flag_value in (True, False):
            trainer = _bare_trainer(
                episode_dir=tmp, skip_intermediate_render=flag_value
            )
            trainer.iteration = 1
            captured = {}

            def fake_popen(cmd, **kwargs):
                captured["cmd"] = cmd
                raise RuntimeError("stop here — we only want the argv")

            import subprocess as _sp

            real_popen = _sp.Popen
            _sp.Popen = fake_popen
            try:
                trainer._collect_via_subprocess(
                    "robocasa_panda_omron/Fake_Env", Path(tmp), 480, 0
                )
            except RuntimeError:
                pass
            finally:
                _sp.Popen = real_popen

            cmd = captured["cmd"]
            present = "--no-skip-intermediate-render" in cmd
            assert present == (not flag_value), (
                f"config skip_intermediate_render={flag_value} produced "
                f"--no-skip-intermediate-render present={present}"
            )

    # And the collector must parse that flag to the matching value.
    sig = inspect.signature(parse_args)
    assert sig is not None
    import sys as _sys

    argv = _sys.argv
    base = ["collect_episodes.py", "--env-name", "e", "--output-dir", "/tmp/x"]
    try:
        _sys.argv = base
        assert parse_args().skip_intermediate_render is True
        _sys.argv = base + ["--no-skip-intermediate-render"]
        assert parse_args().skip_intermediate_render is False
    finally:
        _sys.argv = argv
    print("  [PASS] --no-skip-intermediate-render wiring is direction-correct")


def test_no_signal_skip_path_logs_collect_subphases():
    """The early-skip log site is a separate call site from the normal one and
    must carry the same Phase 1 sub-phases."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _stub_trainer_for_train_loop(tmp)
        # std_reward == 0 routes through the "no gradient signal" continue.
        trainer.buffer.stats = lambda: {
            "success_rate": 0.0, "mean_reward": 0.0, "std_reward": 0.0
        }
        trainer.train()
        tags = _time_tags(trainer)
        assert "time/collect_rollout_seconds" in tags, tags
        assert "time/collect_load_seconds" in tags, tags
        assert "time/collect_seconds" in tags, tags
        # The phases that never ran must not appear.
        assert "time/ref_logprob_seconds" not in tags, tags
        assert "time/update_seconds" not in tags, tags
    print("  [PASS] no-signal skip path logs the Phase 1 sub-phases")


TESTS = [
    test_train_loop_emits_every_phase_curve,
    test_train_loop_cached_iter_gaps_rollout_only,
    test_nan_subphases_are_skipped,
    test_collect_episodes_times_the_right_calls,
    test_failed_collection_leaves_load_time_nan,
    test_cached_load_marks_rollout_nan,
    test_timers_not_carried_across_iterations_by_a_subclass,
    test_collector_cli_flag_only_appended_when_disabled,
    test_no_signal_skip_path_logs_collect_subphases,
]


if __name__ == "__main__":
    print("=== phase-timing log tests ===\n")
    for test in TESTS:
        print(f"{test.__name__}:")
        test()
    print(f"\nAll {len(TESTS)} tests PASSED.")
