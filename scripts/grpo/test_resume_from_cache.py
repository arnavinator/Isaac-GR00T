"""Tests for resume_from_collected_data cache validation.

Synthesizes .npz files matching the format that collect_episodes.save_episodes
writes (collect_episodes.py:1768), then exercises GRPOTrainer's
_validate_collected_data_cache through every accept/reject path.

Runs without GPU, model, robocasa. Imports torch indirectly via grpo_config /
train_grpo, but never instantiates a CUDA tensor or a real model.

Run with:
    .venv/bin/python scripts/grpo/test_resume_from_cache.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make scripts/grpo/ importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent))


def _write_synthetic_npz(
    out_dir: Path,
    episode_idx: int,
    *,
    env_name: str,
    group_id: int,
    success: bool,
    num_chunks: int = 2,
    drop_keys: tuple[str, ...] = (),
    drop_top_keys: tuple[str, ...] = (),
    override_top_keys: dict | None = None,
) -> Path:
    """Write one episode_NNNN.npz with the keys EpisodeBuffer expects.

    The minimum keys to satisfy load_episodes + the resume-cache validator:
        language, env_name, success, max_progress, num_steps, num_chunks,
        group_id, env_seed, action_{i}, action_mask_{i}, raw_action_{i},
        initial_noise_{i}, video_{cam}_{i}, state_{key}_{i}.

    drop_keys lets a test omit specific per-chunk prefixes (e.g.,
    "raw_action_") to exercise the missing-key failure path.
    drop_top_keys lets a test omit specific top-level scalar keys (e.g.,
    "group_id") to exercise the missing-required-field path.
    override_top_keys lets a test replace a top-level scalar with an
    arbitrary value/dtype (e.g., success as np.array("True") to exercise
    the dtype-validation failure path). Applied AFTER drop_top_keys.
    """
    save_dict = {
        "language": "test instruction",
        "env_name": env_name,
        "success": success,
        "max_progress": 1.0 if success else 0.3,
        "num_steps": 50,
        "num_chunks": num_chunks,
        "group_id": group_id,
        "env_seed": 1000 + group_id,
    }
    for k in drop_top_keys:
        save_dict.pop(k, None)
    if override_top_keys:
        save_dict.update(override_top_keys)
    dropped_prefixes = set(drop_keys)
    for chunk_idx in range(num_chunks):
        # Required per-chunk keys
        if "action_" not in dropped_prefixes:
            save_dict[f"action_{chunk_idx}"] = np.zeros((16, 12), dtype=np.float32)
        if "action_mask_" not in dropped_prefixes:
            save_dict[f"action_mask_{chunk_idx}"] = np.ones((50, 128), dtype=np.float32)
        if "raw_action_" not in dropped_prefixes:
            save_dict[f"raw_action_{chunk_idx}"] = np.zeros((50, 128), dtype=np.float32)
        if "initial_noise_" not in dropped_prefixes:
            save_dict[f"initial_noise_{chunk_idx}"] = np.zeros((50, 128), dtype=np.float32)
        # Optional video / state keys (small to keep tests fast).
        save_dict[f"video_camera0_{chunk_idx}"] = np.zeros((4, 4, 3), dtype=np.uint8)
        save_dict[f"state_pos_{chunk_idx}"] = np.zeros((3,), dtype=np.float32)

    path = out_dir / f"episode_{episode_idx:04d}.npz"
    np.savez_compressed(path, **save_dict)
    return path


def _make_cache(
    cache_root: Path,
    iter_num: int,
    *,
    env_name: str,
    n_groups: int,
    group_size: int,
    n_alive_groups: int | None = None,
    drop_keys: tuple[str, ...] = (),
    drop_top_keys: tuple[str, ...] = (),
    partial_group_id: int | None = None,
    partial_group_actual_size: int = 0,
) -> Path:
    """Create cache_root/iter_NNNN/ populated with synthetic episodes.

    n_alive_groups defaults to n_groups (every group is mixed/alive: 1
    success, group_size-1 failures). Set lower to exercise the
    min_alive_groups path.

    partial_group_id (if set) overrides that group's episode count to
    partial_group_actual_size. Use a value < group_size for undercount
    (partial-group warning), or > group_size for overcount (raises).
    """
    if n_alive_groups is None:
        n_alive_groups = n_groups
    iter_dir = cache_root / f"iter_{iter_num:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    ep_idx = 0
    for gid in range(n_groups):
        size = (
            partial_group_actual_size
            if gid == partial_group_id
            else group_size
        )
        # First episode of group gets success if this group is "alive"
        # (mixed); rest of the group fails. Gives each group exactly
        # one success out of group_size, which is mixed under the
        # collector / validator's `0 < successes < len(s)` predicate.
        for in_group_idx in range(size):
            success = (gid < n_alive_groups) and (in_group_idx == 0)
            _write_synthetic_npz(
                iter_dir,
                ep_idx,
                env_name=env_name,
                group_id=gid,
                success=success,
                drop_keys=drop_keys,
                drop_top_keys=drop_top_keys,
            )
            ep_idx += 1
    return iter_dir


def _make_trainer(
    *,
    cache_root: Path,
    env_names: list[str],
    num_groups: int = 3,
    group_size: int = 4,
    min_alive_groups: int = 0,
    max_groups: int | None = None,
    resume_from: str = "grpo_data/grpo_checkpoints/iter_0050",
):
    """Construct a GRPOTrainer without loading the model.

    GRPOTrainer.__init__ avoids any GPU/model work — it just stores config,
    creates a torch.device('cuda') metadata object (which works without
    CUDA), and sets up an episode buffer. setup() is what loads the model,
    and we don't call setup() in these tests.
    """
    from grpo_config import GRPOConfig
    from train_grpo import GRPOTrainer

    if max_groups is None:
        max_groups = num_groups

    cfg = GRPOConfig(
        env_names=list(env_names),
        episode_dir=str(cache_root),
        num_groups=num_groups,
        group_size=group_size,
        num_async_vector_env=group_size,  # avoid divisibility complaints
        min_alive_groups=min_alive_groups,
        max_groups=max_groups,
        resume_from=resume_from,
        resume_from_collected_data=True,
        # Knobs we set explicitly to keep __post_init__ happy without
        # affecting cache validation:
        device="cpu",
        balanced_minibatch_training=False,
        clean_output=True,
        collector_server_host="",
    )
    return GRPOTrainer(cfg)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_happy_path_static_mode():
    """Cache that exactly matches config — should validate cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env"
        # iter_num=51 → task_idx = (51-1) % 1 = 0 → env[0] = `env`. ✓
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=3, group_size=4,
        )
        trainer = _make_trainer(cache_root=root, env_names=[env])
        # Should NOT raise
        trainer._validate_collected_data_cache(51)
    print("  PASS: happy path static mode")


def test_happy_path_dynamic_mode_extra_groups():
    """Dynamic mode: cache has MORE groups than num_groups (legitimate)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env"
        # 5 groups collected when num_groups=3 — dynamic mode produced extras.
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=5, group_size=4,
            n_alive_groups=2,
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
            min_alive_groups=2, max_groups=5,
        )
        trainer._validate_collected_data_cache(51)
    print("  PASS: happy path dynamic mode (extra groups)")


def test_round_robin_task_alignment():
    """Iter_num maps to env_names[(iter_num-1) % len(env_names)]."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        envs = [
            "robocasa_panda_omron/Task_A",
            "robocasa_panda_omron/Task_B",
            "robocasa_panda_omron/Task_C",
        ]
        # iter 51 → task_idx = 50 % 3 = 2 → Task_C
        _make_cache(
            root, iter_num=51,
            env_name=envs[2], n_groups=3, group_size=4,
        )
        trainer = _make_trainer(cache_root=root, env_names=envs)
        trainer._validate_collected_data_cache(51)
    print("  PASS: round-robin task alignment (iter 51 → env_names[2])")


def test_missing_directory():
    """Non-existent cache dir raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        # Don't create iter_0051/
        trainer = _make_trainer(cache_root=root, env_names=[env])
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "does not exist" in str(e), f"unexpected message: {e}"
    print("  PASS: missing directory raises FileNotFoundError")


def test_empty_directory():
    """Empty cache dir raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "iter_0051").mkdir(parents=True)
        env = "robocasa_panda_omron/Task_A"
        trainer = _make_trainer(cache_root=root, env_names=[env])
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "no episode_*.npz" in str(e), f"unexpected message: {e}"
    print("  PASS: empty directory raises FileNotFoundError")


def test_env_name_mismatch():
    """Cached env_name differs from round-robin expectation → RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cached_env = "robocasa_panda_omron/Task_A"
        config_env = "robocasa_panda_omron/Task_B"
        _make_cache(
            root, iter_num=51,
            env_name=cached_env, n_groups=3, group_size=4,
        )
        trainer = _make_trainer(cache_root=root, env_names=[config_env])
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "env_name mismatch" in str(e), f"unexpected message: {e}"
    print("  PASS: env_name mismatch raises RuntimeError")


def test_missing_raw_action_key():
    """Cache without raw_action_* keys → RuntimeError on FM-prereq check."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=3, group_size=4,
            drop_keys=("raw_action_",),
        )
        trainer = _make_trainer(cache_root=root, env_names=[env])
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "raw_action_" in str(e), f"unexpected message: {e}"
    print("  PASS: missing raw_action_ key raises RuntimeError")


def test_missing_initial_noise_key():
    """Cache without initial_noise_* keys → RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=3, group_size=4,
            drop_keys=("initial_noise_",),
        )
        trainer = _make_trainer(cache_root=root, env_names=[env])
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "initial_noise_" in str(e), f"unexpected message: {e}"
    print("  PASS: missing initial_noise_ key raises RuntimeError")


def test_too_few_groups():
    """Cache has fewer distinct group_ids than num_groups → RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        # Only 2 groups when config wants 3
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=2, group_size=4,
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
        )
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "distinct group_ids" in str(e), f"unexpected message: {e}"
            assert "num_groups=3" in str(e), f"unexpected message: {e}"
    print("  PASS: too few groups raises RuntimeError")


def test_min_alive_not_met_below_max():
    """Dynamic mode: min_alive not met AND not at max_groups cap → RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        # 3 groups, only 1 alive — but config wants 2 alive.
        # Cache is at num_groups (3) but max_groups is 5, so it didn't hit
        # the cap → unfinished collection.
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=3, group_size=4,
            n_alive_groups=1,
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
            min_alive_groups=2, max_groups=5,
        )
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "min_alive_groups" in str(e), f"unexpected message: {e}"
    print("  PASS: min_alive unmet below max_groups raises RuntimeError")


def test_min_alive_not_met_but_at_max_cap():
    """Dynamic mode: min_alive not met but cache hit max_groups cap → ACCEPT.

    Hitting max_groups is a legitimate collector exit condition (the task is
    too hard, but we tried), so the cache should be accepted.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        # 5 groups (= max_groups), only 1 alive — collector exhausted budget.
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=5, group_size=4,
            n_alive_groups=1,
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
            min_alive_groups=2, max_groups=5,
        )
        # Should NOT raise — hitting the cap is a valid exit condition.
        trainer._validate_collected_data_cache(51)
    print("  PASS: min_alive unmet but at max_groups cap → ACCEPT")


def test_partial_group_warns_not_raises():
    """Partial group (worker crash) → warning printed, not raised.

    Captures stdout to verify the warning fired, mirroring the trainer's
    existing partial-collection policy.
    """
    import io
    import contextlib

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        # Group 1 has only 2/4 episodes (worker crashed during original collect)
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=3, group_size=4,
            partial_group_id=1, partial_group_actual_size=2,
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # Should NOT raise
            trainer._validate_collected_data_cache(51)
        out = buf.getvalue()
        assert "WARNING: cached group 1 has 2/4" in out, (
            f"expected partial-group warning in stdout, got:\n{out}"
        )
    print("  PASS: partial group warns (not raises)")


def test_post_init_rejects_flag_without_resume_from():
    """GRPOConfig.__post_init__ rejects flag=True without resume_from."""
    from grpo_config import GRPOConfig
    try:
        GRPOConfig(resume_from_collected_data=True)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "requires resume_from" in str(e), f"unexpected message: {e}"
    print("  PASS: __post_init__ rejects flag without resume_from")


def test_validator_belt_and_suspenders():
    """Direct call to _validate_collected_data_cache without resume_from raises.

    __post_init__ already prevents this state from being reached normally,
    but the validator's own guard protects against future refactors that
    bypass dataclass validation.
    """
    from grpo_config import GRPOConfig
    from train_grpo import GRPOTrainer

    cfg = GRPOConfig()  # resume_from=None, flag=False — passes __post_init__
    trainer = GRPOTrainer(cfg)
    try:
        trainer._validate_collected_data_cache(51)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "without resume_from" in str(e), f"unexpected message: {e}"
    print("  PASS: validator's belt-and-suspenders guard fires")


def test_overcount_raises():
    """Group with MORE than group_size episodes → RuntimeError (Bug C).

    Indicates a manual cache merge or collector bug; within-group env_seed
    invariant is broken. Reject loudly rather than silently warn.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        # Group 1 has 8/4 episodes — overcount.
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=3, group_size=4,
            partial_group_id=1, partial_group_actual_size=8,
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
        )
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "MORE than group_size" in str(e), (
                f"unexpected message: {e}"
            )
    print("  PASS: overcount group raises RuntimeError")


def test_max_groups_lowered_rejects_cache():
    """Cache has more groups than current max_groups → reject (Bug B + Bug A1).

    User collected with max_groups=5, then lowered to max_groups=3 between
    save and resume. Cache has 5 groups + only 1 alive but min_alive=2.
    Original audit (Bug B) made `n_obs != max_groups` reject this case via
    the min_alive gate. Round-2 audit (Bug A1) added an UNCONDITIONAL
    `n_obs > max_groups` check that fires first, regardless of n_alive —
    this test now hits that path.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=5, group_size=4,
            n_alive_groups=1,
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
            min_alive_groups=2, max_groups=3,
        )
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "exceeding max_groups" in str(e), (
                f"unexpected message: {e}"
            )
    print("  PASS: max_groups lowered between save/resume → reject")


def test_max_groups_exceeded_with_sufficient_success_still_rejects():
    """Bug A1: n_obs > max_groups must reject even when n_alive >= min_alive.

    Round-2 audit found that the old min_alive_groups gate
    short-circuits when `n_alive >= min_alive`, allowing an over-collected
    cache to pass silently. The unconditional upper-bound check catches
    this case.

    Setup: cache has 5 groups, 3 alive (>= min_alive=2). Config has
    max_groups=3. Old check (`n_obs != max_groups` gated behind
    `n_alive < min_alive`) would have skipped because the n_alive gate is
    False. New unconditional `n_obs > max_groups` correctly rejects.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=5, group_size=4,
            n_alive_groups=3,  # ABOVE min_alive → old gate skipped
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
            min_alive_groups=2, max_groups=3,
        )
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "exceeding max_groups" in str(e), (
                f"unexpected message: {e}"
            )
    print("  PASS: n_obs > max_groups rejects even with sufficient success")


def test_missing_group_id_raises():
    """Cache .npz files lacking group_id key → RuntimeError (Bug D).

    Old behavior silently defaulted to group_id=0, merging all rollouts
    into one synthetic group and breaking GRPO's within-group invariant.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=3, group_size=4,
            drop_top_keys=("group_id",),
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
        )
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "group_id" in str(e), f"unexpected message: {e}"
    print("  PASS: missing group_id raises RuntimeError")


def test_missing_success_key_wrapped_error():
    """Missing 'success' scalar produces helpful (not raw KeyError) error (Bug E)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=3, group_size=4,
            drop_top_keys=("success",),
        )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
        )
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "Cache may be corrupted" in str(e), (
                f"unexpected message: {e}"
            )
            # Verify the error message includes the file path for triage
            assert "episode_" in str(e), f"unexpected message: {e}"
    print("  PASS: missing 'success' wrapped with helpful error")


def test_canonical_iter_name_required_when_flag_set():
    """resume_from_collected_data=True with non-iter_NNNN basename → raise (Bug A).

    Setup() should hard-fail rather than silently fall back to start_iteration=1
    (which would validate iter_0001/ — almost certainly the wrong cache).

    Imports `ITER_DIR_RE` from train_grpo so test and prod can't drift on
    the regex pattern. Earlier versions of this test duplicated the regex
    literal, which would have silently passed if prod regressed back to
    `\\d+` (Unicode-permissive).
    """
    from train_grpo import ITER_DIR_RE
    bad_paths = [
        "/tmp/checkpoints/best",
        "/tmp/checkpoints/latest",
        "/tmp/checkpoints/iter_50.bak",
        "/tmp/checkpoints/iter_50_v2",
        "/tmp/checkpoints/iter_NaN",
        "/tmp/checkpoints/iter_-5",
    ]
    for path in bad_paths:
        dir_name = Path(path).name
        m = ITER_DIR_RE.fullmatch(dir_name)
        assert m is None, (
            f"regex unexpectedly matched {dir_name!r} — would silently "
            f"validate the wrong cache"
        )
    # Canonical names DO match.
    for good_name in ("iter_50", "iter_0050", "iter_99999"):
        m = ITER_DIR_RE.fullmatch(good_name)
        assert m is not None, (
            f"regex unexpectedly rejected canonical name {good_name!r}"
        )
    print("  PASS: canonical iter_NNNN regex matches expected names")


def test_post_init_rejects_empty_resume_from():
    """resume_from='' with flag=True → ValueError (Bug N)."""
    from grpo_config import GRPOConfig
    for empty in ("", "  ", "\t\n"):
        try:
            GRPOConfig(
                resume_from=empty,
                resume_from_collected_data=True,
            )
            assert False, f"expected ValueError for resume_from={empty!r}"
        except ValueError as e:
            assert "non-empty path" in str(e), f"unexpected message: {e}"
    print("  PASS: empty/whitespace resume_from rejected")


def test_post_init_rejects_empty_env_names():
    """env_names=[] → ValueError (Bug H)."""
    from grpo_config import GRPOConfig
    try:
        GRPOConfig(env_names=[])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "env_names" in str(e), f"unexpected message: {e}"
    print("  PASS: empty env_names rejected")


def test_log_metrics_skips_nan_phase_times():
    """_log_metrics drops NaN phase_times entries from the TB writer (Bug Q).

    Captures all SummaryWriter.add_scalar calls via a fake writer and
    asserts the cached-iter `time/collect_seconds` was NOT written but
    other phase times were. Wandb is disabled here (`use_wandb=False`)
    so this test only covers the TB branch — the wandb branch uses the
    same `if not math.isnan(v)` filter and is structurally identical
    (see _log_metrics's wandb dict comprehension), but is not directly
    exercised by this test. A regression in JUST the wandb branch would
    not be caught by this test.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        # Cache existence isn't required for this test, but the trainer
        # constructor expects a valid config; we use the same helper.
        _make_cache(
            root, iter_num=51,
            env_name=env, n_groups=3, group_size=4,
        )
        trainer = _make_trainer(cache_root=root, env_names=[env])

        class _FakeWriter:
            def __init__(self):
                self.scalars: list[tuple[str, float, int]] = []
            def add_scalar(self, name, value, step):
                self.scalars.append((name, value, step))
            def close(self):
                pass

        trainer.writer = _FakeWriter()
        # Disable wandb — we're only verifying TB-side filtering.
        trainer.config.use_wandb = False

        trainer._log_metrics(
            iteration=51,
            stats={},  # empty stats → no episode/* logs
            phase_times={
                "collect": float("nan"),  # cached → NaN sentinel
                "advantage": 0.5,
                "update": 2.0,
            },
            lora_delta_norm=0.0,
        )

        names = [name for name, _, _ in trainer.writer.scalars]
        assert "time/collect_seconds" not in names, (
            f"cached-iter NaN should have been skipped; got names={names}"
        )
        assert "time/advantage_seconds" in names, (
            f"non-NaN phase_times should still log; got names={names}"
        )
        assert "time/update_seconds" in names, (
            f"non-NaN phase_times should still log; got names={names}"
        )
    print("  PASS: _log_metrics skips NaN phase_times entries")


def test_log_metrics_emits_lr_and_iter_time_on_early_skip():
    """F12 (Bug F regression cover): the early-skip path must still log
    train/learning_rate and time/iteration_seconds.

    Round 1 audit (Bug F) added `lr=lr, iter_time=time.time() - iter_start`
    to the std_reward<1e-8 _log_metrics call so curves don't gap there.
    Without this regression test, a revert that drops those args wouldn't
    be caught by any other test.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        _make_cache(root, iter_num=51, env_name=env, n_groups=3, group_size=4)
        trainer = _make_trainer(cache_root=root, env_names=[env])

        class _FakeWriter:
            def __init__(self):
                self.scalars: list[tuple[str, float, int]] = []
            def add_scalar(self, name, value, step):
                self.scalars.append((name, value, step))
            def close(self):
                pass

        trainer.writer = _FakeWriter()
        trainer.config.use_wandb = False

        # Simulate the early-skip call site: skip_reason="no_signal", with
        # lr and iter_time both passed (the post-Bug-F behavior).
        trainer._log_metrics(
            iteration=51,
            stats={},
            skip_reason="no_signal",
            lr=3e-5,
            iter_time=1.5,
            phase_times={
                "collect": 0.05,
                "advantage": 0.1,
            },
            lora_delta_norm=0.0,
        )

        names = {name for name, _, _ in trainer.writer.scalars}
        assert "train/learning_rate" in names, (
            f"early-skip path must log train/learning_rate; got {names}"
        )
        assert "time/iteration_seconds" in names, (
            f"early-skip path must log time/iteration_seconds; got {names}"
        )
    print("  PASS: early-skip path logs lr and iter_time")


def test_num_chunks_zero_raises():
    """Bug A2: num_chunks=0 silently passes spot-check, breaks downstream invariant.

    A corrupted file with num_chunks=0 makes `range(num_chunks)` empty,
    so the FM-key spot-check loop trivially passes. But the file has no
    chunks at all — `_load_single_episode` builds a 0-chunk episode that
    consumes one advantage in compute_advantages but contributes 0 chunks
    to _build_chunks, breaking the within-group `Σ A_chunk = 0`
    invariant the GRPO surrogate relies on. New explicit num_chunks > 0
    check rejects this.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        iter_dir = root / "iter_0051"
        iter_dir.mkdir(parents=True)
        # Hand-write 12 episodes with num_chunks=0 (override the helper's
        # default of 2). With num_chunks=0 the per-chunk action_*,
        # raw_action_*, etc. keys aren't written either.
        for ep_idx in range(12):
            gid = ep_idx // 4
            _write_synthetic_npz(
                iter_dir, ep_idx,
                env_name=env, group_id=gid,
                success=(ep_idx % 4 == 0),
                num_chunks=0,
            )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
        )
        try:
            trainer._validate_collected_data_cache(51)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "num_chunks" in str(e) and "positive integer" in str(e), (
                f"unexpected message: {e}"
            )
    print("  PASS: num_chunks=0 rejected with explicit error")


def _assert_dtype_rejection(
    *, key_name: str, override_value, expected_msg_terms: tuple[str, ...]
) -> None:
    """Build a 12-episode cache with the given top-level scalar overridden,
    then assert that _validate_collected_data_cache raises a RuntimeError
    whose message contains every term in `expected_msg_terms`.

    Used by the dtype-validation tests so each test case is one line —
    saves ~30 lines per case and makes it trivial to add a new dtype
    rejection case.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        iter_dir = root / "iter_0051"
        iter_dir.mkdir(parents=True)
        for ep_idx in range(12):
            gid = ep_idx // 4
            _write_synthetic_npz(
                iter_dir, ep_idx,
                env_name=env, group_id=gid,
                success=(ep_idx % 4 == 0),
                override_top_keys={key_name: override_value},
            )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
        )
        try:
            trainer._validate_collected_data_cache(51)
            assert False, (
                f"expected RuntimeError for {key_name}={override_value!r}"
            )
        except RuntimeError as e:
            for term in expected_msg_terms:
                assert term in str(e), (
                    f"missing {term!r} in error message: {e}"
                )


def test_dtype_validation_group_id_float():
    """Bug A3: group_id saved as float silently truncates via int() — reject."""
    _assert_dtype_rejection(
        key_name="group_id",
        override_value=np.float64(2),
        expected_msg_terms=("group_id", "dtype"),
    )
    print("  PASS: group_id float dtype rejected")


def test_dtype_validation_success_string():
    """Bug A3/A7: success saved as string evaluates True for any non-empty str — reject."""
    _assert_dtype_rejection(
        key_name="success",
        override_value="False",
        expected_msg_terms=("success", "dtype"),
    )
    print("  PASS: success string dtype rejected")


def test_dtype_validation_num_chunks_float():
    """Bug A3: num_chunks saved as float silently truncates via int() — reject."""
    _assert_dtype_rejection(
        key_name="num_chunks",
        override_value=np.float64(2.7),
        expected_msg_terms=("num_chunks", "dtype"),
    )
    print("  PASS: num_chunks float dtype rejected")


def test_dtype_validation_group_id_bool_rejected():
    """Bug A3: group_id with bool dtype silently coerces to 0/1 via int() — reject.

    bool dtype.kind == 'b'. int(np.array(True)) == 1, int(np.array(False)) == 0
    — would silently merge any True-tagged episodes into group 1 and
    False-tagged into group 0, breaking the within-group invariant.
    """
    _assert_dtype_rejection(
        key_name="group_id",
        override_value=np.bool_(True),
        expected_msg_terms=("group_id", "dtype"),
    )
    print("  PASS: group_id bool dtype rejected")


def test_dtype_validation_success_object_rejected():
    """Bug A3: success with object dtype silently coerces — reject.

    np.array(False, dtype=object) has dtype.kind == 'O'. The validator
    rejects on dtype rather than evaluating the value.
    """
    _assert_dtype_rejection(
        key_name="success",
        override_value=np.array(False, dtype=object),
        expected_msg_terms=("success", "dtype"),
    )
    print("  PASS: success object dtype rejected")


def test_load_cached_episodes_populates_buffer_and_resets_counter():
    """H3 (coverage gap): direct test of _load_cached_episodes.

    Builds a valid cache, calls the loader directly, and asserts the
    buffer is populated, the failure counter is reset, and chunk counts
    match. A regression in the loader's pop-and-load logic would not be
    caught by validator-only tests.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        _make_cache(root, iter_num=51, env_name=env, n_groups=3, group_size=4)
        trainer = _make_trainer(cache_root=root, env_names=[env])
        trainer.iteration = 51
        # Pre-condition: buffer empty, simulate failures from a prior iter
        # to verify the counter reset.
        trainer._consecutive_collect_failures = 2
        assert trainer.buffer.num_episodes == 0

        trainer._load_cached_episodes()

        assert trainer.buffer.num_episodes == 12, (
            f"expected 12 eps, got {trainer.buffer.num_episodes}"
        )
        # 12 eps × 2 chunks per default _make_cache = 24 chunks
        assert trainer.buffer.num_chunks == 24, (
            f"expected 24 chunks, got {trainer.buffer.num_chunks}"
        )
        assert trainer._consecutive_collect_failures == 0, (
            f"_load_cached_episodes must reset the counter on success; "
            f"got {trainer._consecutive_collect_failures}"
        )
    print("  PASS: _load_cached_episodes populates buffer and resets counter")


def test_load_cached_episodes_raises_when_dir_disappears_post_validate():
    """H3: cache deleted between setup and train → RuntimeError, not silent zero-load.

    Documents the validator-then-loader contract: if validation passes
    but the cache disappears (or contains no .npz) before the loader
    runs, the loader must raise rather than silently produce an empty
    buffer (which would break compute_advantages and downstream).
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        # Build cache, set up trainer, then delete the dir to simulate
        # a filesystem race / external cleanup between validate and load.
        iter_dir = _make_cache(
            root, iter_num=51, env_name=env, n_groups=3, group_size=4,
        )
        trainer = _make_trainer(cache_root=root, env_names=[env])
        trainer.iteration = 51
        # Validator runs at setup time; then the dir disappears.
        trainer._validate_collected_data_cache(51)
        import shutil as _sh
        _sh.rmtree(iter_dir)

        try:
            trainer._load_cached_episodes()
            assert False, "expected RuntimeError on missing cache"
        except RuntimeError as e:
            assert "validated at setup" in str(e), (
                f"unexpected message: {e}"
            )
    print(
        "  PASS: _load_cached_episodes raises when cache disappears "
        "post-validate"
    )


def test_episode_dir_normalized_to_absolute():
    """FS-F3: episode_dir resolved to absolute in __post_init__ so a CWD
    change between setup() and train() can't make the cached path point
    elsewhere. Verifies a relative input path is rewritten to absolute,
    and that an already-absolute path round-trips unchanged.
    """
    import os
    from grpo_config import GRPOConfig
    # Relative input → absolute output. We don't assert the resolved path
    # has any specific prefix because project conventions sometimes
    # symlink data dirs (e.g., grpo_data → /mnt/scratch/...), and
    # `.resolve()` follows symlinks.
    cfg = GRPOConfig(episode_dir="grpo_data/grpo_episodes")
    assert os.path.isabs(cfg.episode_dir), (
        f"episode_dir not normalized to absolute: {cfg.episode_dir!r}"
    )

    # Already-absolute input: must round-trip to an equivalent absolute
    # path (allowing for `~` expansion or symlink resolution by .resolve).
    with tempfile.TemporaryDirectory() as tmp:
        cfg_abs = GRPOConfig(episode_dir=tmp)
        assert os.path.isabs(cfg_abs.episode_dir), (
            f"absolute input lost absolute-ness: {cfg_abs.episode_dir!r}"
        )
        # The path should reach the same physical location (resolve may
        # canonicalize symlinks like /tmp → /private/tmp on macOS).
        assert Path(cfg_abs.episode_dir).resolve() == Path(tmp).resolve(), (
            f"absolute episode_dir resolved to a different location: "
            f"{cfg_abs.episode_dir!r} vs {tmp!r}"
        )
    print("  PASS: episode_dir normalized to absolute path")


def test_load_cached_raises_on_heterogeneous_fm_corruption():
    """FS-F4: validator's spot-check only sees file 0; loader-side
    consistency check catches missing FM keys in non-zero files.

    Concrete scenario: file 0 has raw_action / action_mask / initial_noise
    keys, but file 5 lacks them (manual merge across collector versions).
    Validator passes (single-sample spot-check). Loader silently sets the
    missing chunks' fields to None. Without the post-load consistency
    check, _prepare_batch would silently drop those chunks at training
    time — producing a mostly-dead minibatch with no signal.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = "robocasa_panda_omron/Task_A"
        iter_dir = root / "iter_0051"
        iter_dir.mkdir(parents=True)
        # File 0: all keys present (passes validator spot-check).
        # Files 1-11: drop raw_action_* keys per chunk.
        for ep_idx in range(12):
            gid = ep_idx // 4
            drop_keys = ()
            if ep_idx > 0:
                drop_keys = ("raw_action_",)
            _write_synthetic_npz(
                iter_dir, ep_idx,
                env_name=env, group_id=gid,
                success=(ep_idx % 4 == 0),
                drop_keys=drop_keys,
            )
        trainer = _make_trainer(
            cache_root=root, env_names=[env],
            num_groups=3, group_size=4,
        )
        trainer.iteration = 51

        # Validator passes (spot-check only inspects file 0, which is OK).
        trainer._validate_collected_data_cache(51)
        # Loader catches the heterogeneous corruption.
        try:
            trainer._load_cached_episodes()
            assert False, "expected RuntimeError on heterogeneous FM corruption"
        except RuntimeError as e:
            assert "raw_action" in str(e) and "manual cache merge" in str(e), (
                f"unexpected message: {e}"
            )
    print("  PASS: heterogeneous FM-key corruption caught at load time")


def test_main_calls_shutdown_on_setup_failure():
    """H5 (Bug B1 regression coverage): main()'s try/finally must invoke
    shutdown() even when setup() raises.

    Round 2 audit found setup() was OUTSIDE main()'s try block — fix
    moved it inside. Without this test, a regression where someone
    moves setup() back outside the try would silently leak the ZMQ
    collector socket on misconfigured runs.

    This test patches GRPOTrainer.setup() to raise immediately and
    GRPOTrainer.shutdown() to record being called. Then runs main() and
    asserts shutdown was invoked despite the setup failure.
    """
    import train_grpo

    captured = {"shutdown_called": False, "setup_called": False}
    orig_setup = train_grpo.GRPOTrainer.setup
    orig_shutdown = train_grpo.GRPOTrainer.shutdown

    def fake_setup(self):
        captured["setup_called"] = True
        raise RuntimeError("simulated setup failure")

    def fake_shutdown(self):
        captured["shutdown_called"] = True

    # Patch tyro to return a default config (no CLI args; happens during
    # module setup so __post_init__ runs cleanly).
    import sys as _sys
    fake_tyro = type(_sys)("tyro")  # bare module
    fake_tyro.cli = lambda cls: cls()
    _sys.modules["tyro"] = fake_tyro

    train_grpo.GRPOTrainer.setup = fake_setup
    train_grpo.GRPOTrainer.shutdown = fake_shutdown
    try:
        try:
            train_grpo.main()
            assert False, "expected RuntimeError to propagate from main"
        except RuntimeError as e:
            assert "simulated setup failure" in str(e)
    finally:
        train_grpo.GRPOTrainer.setup = orig_setup
        train_grpo.GRPOTrainer.shutdown = orig_shutdown
        _sys.modules.pop("tyro", None)

    assert captured["setup_called"], "fake setup() never ran"
    assert captured["shutdown_called"], (
        "main() must call shutdown() in the finally block when setup() "
        "raises — regression: setup() was moved outside try/finally?"
    )
    print("  PASS: main() invokes shutdown() when setup() raises")


def test_unicode_digits_in_resume_from_rejected():
    """Bug A4: full-width / Unicode digits don't match [0-9]+ regex.

    Python's `\\d+` matches all Nd-category Unicode digits (~580 codepoints,
    including full-width '０'-'９', Arabic-Indic '٠'-'٩'). int() then parses
    them. The fixed regex is `[0-9]+` (ASCII only) so a checkpoint named
    e.g. 'iter_０5' falls through to start_iteration=1 — and with the
    cache flag on, raises the canonical-name error.
    """
    from train_grpo import ITER_DIR_RE
    bad_unicode_paths = [
        "iter_０",         # Full-width zero
        "iter_０1２3",     # Mixed full-width + ASCII
        "iter_٥٠",         # Arabic-Indic 50
    ]
    for name in bad_unicode_paths:
        m = ITER_DIR_RE.fullmatch(name)
        assert m is None, (
            f"ASCII regex unexpectedly matched Unicode-digit name {name!r}"
        )
    # Sanity-check ASCII still matches.
    for good_name in ("iter_50", "iter_0050", "iter_99999"):
        m = ITER_DIR_RE.fullmatch(good_name)
        assert m is not None, (
            f"regex unexpectedly rejected canonical name {good_name!r}"
        )
    print("  PASS: Unicode digits rejected, ASCII digits accepted")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def main():
    tests = [
        # Original suite
        test_post_init_rejects_flag_without_resume_from,
        test_validator_belt_and_suspenders,
        test_happy_path_static_mode,
        test_happy_path_dynamic_mode_extra_groups,
        test_round_robin_task_alignment,
        test_missing_directory,
        test_empty_directory,
        test_env_name_mismatch,
        test_missing_raw_action_key,
        test_missing_initial_noise_key,
        test_too_few_groups,
        test_min_alive_not_met_below_max,
        test_min_alive_not_met_but_at_max_cap,
        test_partial_group_warns_not_raises,
        # Round-1 audit fixes
        test_overcount_raises,                             # Bug C
        test_max_groups_lowered_rejects_cache,             # Bug B (now A1 path)
        test_missing_group_id_raises,                      # Bug D
        test_missing_success_key_wrapped_error,            # Bug E
        test_canonical_iter_name_required_when_flag_set,   # Bug A
        test_post_init_rejects_empty_resume_from,          # Bug N
        test_post_init_rejects_empty_env_names,            # Bug H
        test_log_metrics_skips_nan_phase_times,            # Bug Q
        test_log_metrics_emits_lr_and_iter_time_on_early_skip,  # F12 (B-F regression cover)
        # Round-2 audit fixes
        test_max_groups_exceeded_with_sufficient_success_still_rejects,  # Bug A1
        test_num_chunks_zero_raises,                       # Bug A2
        test_dtype_validation_group_id_float,              # Bug A3
        test_dtype_validation_success_string,              # Bug A3/A7
        test_dtype_validation_num_chunks_float,            # Bug A3
        test_dtype_validation_group_id_bool_rejected,      # M8
        test_dtype_validation_success_object_rejected,     # M8
        test_unicode_digits_in_resume_from_rejected,       # Bug A4
        # Round-3 audit (coverage gaps)
        test_load_cached_episodes_populates_buffer_and_resets_counter,  # H3
        test_load_cached_episodes_raises_when_dir_disappears_post_validate,  # H3
        test_main_calls_shutdown_on_setup_failure,         # H5 (B1 regression cover)
        # Round-4 audit (filesystem fuzz)
        test_episode_dir_normalized_to_absolute,           # FS-F3
        test_load_cached_raises_on_heterogeneous_fm_corruption,  # FS-F4
    ]
    print(f"=== Running {len(tests)} cache-validation tests ===\n")
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests PASSED.")


if __name__ == "__main__":
    main()
