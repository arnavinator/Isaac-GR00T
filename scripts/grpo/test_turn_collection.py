"""CPU-only tests for multi-turn group collection (num_async_vector_env).

Verifies the turn-driver logic in collect_episodes.EpisodeCollector WITHOUT a
GPU, the GR00T model server, or robocasa/MuJoCo. We drive the REAL
EpisodeCollector with a fake vector env (FakeVectorEnv) and a fake policy
client (FakePolicyClient) patched in over gymnasium / PolicyClient, so the
actual control flow under test (_align_envs_to_group_scene,
_run_group_over_turns, _restart_at_branch_point, the per-env loop, and the
fast-forward + init-state branches) is exercised end-to-end.

Run with:
    scripts/grpo/../../.venv/bin/python scripts/grpo/test_turn_collection.py
(any interpreter with numpy + gymnasium + the gr00t package importable).
"""
import sys
from pathlib import Path
from unittest import mock

import numpy as np

# collect_episodes lives next to this file; import it directly.
sys.path.insert(0, str(Path(__file__).parent))
import collect_episodes as ce


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _one_obs() -> dict:
    """A single-env observation dict shaped like what the wrapper returns."""
    return {
        "video.image_0": np.zeros((3, 4, 4), dtype=np.uint8),
        "state.x": np.zeros(2, dtype=np.float32),
        "annotation.human.action.task_description": "do the thing",
    }


def _infer_num_active(obs: dict) -> int:
    for v in obs.values():
        if isinstance(v, np.ndarray):
            return int(v.shape[0])
        if isinstance(v, (tuple, list)):
            return len(v)
    return 1


class FakeVectorEnv:
    """Stands in for gym.vector.{Async,Sync}VectorEnv.

    Records reset seeds, RPC calls, and bundles applied so tests can assert the
    turn driver's behavior. Terminates every env once `step_count` exceeds
    `terminate_after` (default 0 → terminate on the first per-env-loop step),
    which lets the fast-forward prefix run `terminate_after` non-terminal steps.

    Models gymnasium's autoreset HONESTLY for the collector's real config,
    AsyncVectorEnv(shared_memory=False): under the DEFAULT AutoresetMode.NEXT_STEP
    a terminated env is flagged and its NEXT step() resets (discarding the
    action) instead of stepping — and a vector-level reset() does NOT clear that
    worker flag with shared_memory=False (only a step that consumes it, or
    autoreset_mode=DISABLED avoiding the flag entirely, does). `env.call()` never
    touches it. The fake therefore does NOT clear the flag on reset(); the only
    thing that prevents a cross-turn leak is constructing with
    autoreset_mode=DISABLED. `autoreset_fired` counts autoresets consumed by
    step(); in this lockstep fake all envs of a turn finish together, so any
    nonzero count means an autoreset leaked across a turn/group boundary and ate
    the branch-point restart. The fake captures `autoreset_mode` so tests can
    assert the collector requested DISABLED.
    """

    def __init__(self, num_envs: int, autoreset_mode=None):
        self.num_envs = num_envs
        self.autoreset_mode = autoreset_mode
        # DISABLED removes the autoreset state machine; anything else (incl. the
        # default None == NEXT_STEP) leaks across boundaries in this fake.
        self._disabled = (
            autoreset_mode is not None
            and getattr(autoreset_mode, "name", str(autoreset_mode)) == "DISABLED"
        )
        self.terminate_after = 0
        # Optional per-env outer-step thresholds (env i terminates when
        # step_count > terminate_at[i]); when None, all envs use terminate_after.
        # Lets a test produce STAGGERED termination (some envs done while siblings
        # are still active) to exercise the collector's mid-turn done-env handling.
        self.terminate_at = None
        self.reset_calls: list = []
        self.calls: list = []
        self.apply_bundle_args: list = []
        self.get_bundle_returns: list = []
        self.step_count = 0
        self._bundle_counter = 0
        self._autoreset = [False] * num_envs  # NEXT_STEP pending-autoreset flags
        self.autoreset_fired = 0

    def _batched_obs(self) -> dict:
        return {
            "video.image_0": np.zeros((self.num_envs, 3, 4, 4), dtype=np.uint8),
            "state.x": np.zeros((self.num_envs, 2), dtype=np.float32),
            "annotation.human.action.task_description": tuple(
                "do the thing" for _ in range(self.num_envs)
            ),
        }

    def reset(self, seed=None):
        self.reset_calls.append(seed)
        # Faithful to AsyncVectorEnv(shared_memory=False): a vector reset does
        # NOT clear the worker's pending NEXT_STEP autoreset flag. (Under
        # DISABLED there is no such flag to begin with.)
        return self._batched_obs(), {}

    def call(self, method: str, *args):
        # NOTE: like real gymnasium, env.call() does NOT touch the vector-level
        # autoreset flags — only reset()/step() do.
        self.calls.append((method, args))
        if method == "get_scene_bundle":
            out = []
            for _ in range(self.num_envs):
                self._bundle_counter += 1
                out.append({"id": self._bundle_counter})
            out = tuple(out)
            self.get_bundle_returns.append(out)
            return out
        if method == "apply_scene_bundle":
            self.apply_bundle_args.append(args[0])
            return tuple(_one_obs() for _ in range(self.num_envs))
        if method == "compute_dense_progress":
            return tuple(0.0 for _ in range(self.num_envs))
        raise ValueError(f"unexpected RPC call: {method}")

    def step(self, actions_full):
        self.step_count += 1
        terms = []
        for i in range(self.num_envs):
            if not self._disabled and self._autoreset[i]:
                # NEXT_STEP autoreset: reset this env, discard the action.
                self.autoreset_fired += 1
                terms.append(False)
            else:
                thresh = (self.terminate_at[i] if self.terminate_at is not None
                          else self.terminate_after)
                terms.append(self.step_count > thresh)
        if not self._disabled:
            self._autoreset = list(terms)  # flags for the NEXT step
        truncs = [False] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        infos = {
            "final_info": [
                {"success": np.array([i % 2 == 0]), "dones": np.array([True])}
                for i in range(self.num_envs)
            ]
        }
        return self._batched_obs(), rewards, terms, truncs, infos

    def close(self):
        pass


class FakePolicyClient:
    """Stands in for PolicyClient. Returns fresh, DISTINCT initial noise per
    query (mimics the unseeded torch.randn the real server draws), so tests can
    confirm that successive turns re-query rather than reuse cached actions."""

    def __init__(self, host=None, port=None):
        self.noise_counter = 0
        self.query_count = 0

    def get_action(self, observations):
        n = _infer_num_active(observations)
        self.query_count += 1
        T, D = 4, 6
        action = {"action": np.zeros((n, T, D), dtype=np.float32)}
        noise = np.array(
            [[float(self.noise_counter + j)] for j in range(n)], dtype=np.float32
        )
        self.noise_counter += n
        info = {
            "initial_noise": noise,
            "raw_actions": np.zeros((n, 50, 128), dtype=np.float32),
            "action_mask": np.ones((n, 50, 128), dtype=np.float32),
        }
        return action, info


def _make_collector(group_size: int, num_async_vector_env=None) -> ce.EpisodeCollector:
    """Construct a real EpisodeCollector with vector env + PolicyClient faked.

    The patched constructors forward the real `autoreset_mode` kwarg to the fake,
    so the fake faithfully (does not) leak autoreset depending on what the
    collector requests — making the DISABLED-mode fix testable.
    """
    with mock.patch.object(
        ce.gym.vector, "AsyncVectorEnv",
        lambda env_fns, **kw: FakeVectorEnv(len(env_fns), kw.get("autoreset_mode")),
    ), mock.patch.object(
        ce.gym.vector, "SyncVectorEnv",
        lambda env_fns, **kw: FakeVectorEnv(len(env_fns), kw.get("autoreset_mode")),
    ), mock.patch.object(ce, "PolicyClient", FakePolicyClient):
        return ce.EpisodeCollector(
            env_name="robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env",
            group_size=group_size,
            max_episode_steps=100,
            n_action_steps=8,
            server_host="127.0.0.1",
            server_port=5555,
            num_async_vector_env=num_async_vector_env,
        )


# ---------------------------------------------------------------------------
# Test harness (mirrors test_balanced_fixes.py)
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


def _noise_values(episodes: list) -> list:
    """First-chunk initial-noise scalar from each episode (distinct per query)."""
    return [float(ep["initial_noises"][0][0]) for ep in episodes]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_unchanged_when_num_envs_equals_group_size():
    """num_async_vector_env == group_size → 1 turn/group, identical to old path."""
    print("\n[Turn] Unchanged behavior when num_envs == group_size")
    c = _make_collector(group_size=4, num_async_vector_env=4)
    check("num_envs == group_size", c.num_envs == 4)
    check("turns_per_group == 1", c.turns_per_group == 1)

    eps = c._collect_one_group(group_seed=1000, group_id=0, success_weight=1.0)
    env = c.envs
    check("group_size episodes returned", len(eps) == 4, f"got {len(eps)}")
    check("reset called exactly once", len(env.reset_calls) == 1, f"{env.reset_calls}")
    check("reset uses num_envs seeds", env.reset_calls[0] == [1000] * 4, f"{env.reset_calls[0]}")
    check("apply_scene_bundle called once (single turn)",
          env.apply_bundle_args and len(env.apply_bundle_args) == 1,
          f"{len(env.apply_bundle_args)}")
    n_get = sum(1 for m, _ in env.calls if m == "get_scene_bundle")
    check("get_scene_bundle called exactly once (no extra RPC vs old path)",
          n_get == 1, f"{n_get}")
    check("step called once (single turn, terminate immediately)",
          env.step_count == 1, f"{env.step_count}")
    check("all episodes group_id==0", all(e["group_id"] == 0 for e in eps))
    check("no autoreset fired (single turn)", env.autoreset_fired == 0, f"{env.autoreset_fired}")
    c.close()


def test_multi_turn_divisible():
    """group_size=4, num_envs=2 → 2 turns; same captured bundle re-applied."""
    print("\n[Turn] Multi-turn (group_size=4, num_async_vector_env=2)")
    c = _make_collector(group_size=4, num_async_vector_env=2)
    check("num_envs == 2", c.num_envs == 2)
    check("turns_per_group == 2", c.turns_per_group == 2)

    eps = c._collect_one_group(group_seed=2000, group_id=3, success_weight=1.0)
    env = c.envs
    check("group_size (4) episodes returned across 2 turns", len(eps) == 4, f"got {len(eps)}")
    check("all episodes share group_id==3", all(e["group_id"] == 3 for e in eps))
    check("all episodes share env_seed==2000", all(e["env_seed"] == 2000 for e in eps))

    # One reset per turn: the align (turn 1) plus one before each restart turn,
    # to CLEAR gymnasium's NEXT_STEP autoreset flags before re-applying the
    # branch point (env.call does not clear them; a bare step() would autoreset).
    check("reset called once per turn (2)",
          len(env.reset_calls) == 2, f"{len(env.reset_calls)}")
    check("every reset uses num_envs (2) group seeds",
          all(rc == [2000, 2000] for rc in env.reset_calls), f"{env.reset_calls}")
    check("no autoreset leaked across a turn boundary (branch restart intact)",
          env.autoreset_fired == 0, f"autoreset_fired={env.autoreset_fired}")
    n_get = sum(1 for m, _ in env.calls if m == "get_scene_bundle")
    check("get_scene_bundle captured once", n_get == 1, f"{n_get}")
    check("apply_scene_bundle called once per turn (2)",
          len(env.apply_bundle_args) == 2, f"{len(env.apply_bundle_args)}")
    # Normal mode: every turn re-applies the SAME captured bundle.
    ids = [b.get("id") for b in env.apply_bundle_args]
    check("turns 2..k re-apply the SAME captured bundle (identical id)",
          len(set(ids)) == 1, f"ids={ids}")
    # Regression (Finding 2): each apply must receive a FRESH copy (distinct
    # object) so apply_scene_bundle's in-place ep_meta mutation cannot pollute
    # the stored branch point across turns (matters in SyncVectorEnv mode).
    check("each apply gets a fresh copy (distinct objects, no aliasing)",
          len({id(b) for b in env.apply_bundle_args}) == len(env.apply_bundle_args),
          f"{len({id(b) for b in env.apply_bundle_args})} distinct of {len(env.apply_bundle_args)}")
    check("step called once per turn (2 total)", env.step_count == 2, f"{env.step_count}")

    # Fresh noise per turn → all 4 rollouts got distinct initial noise.
    vals = _noise_values(eps)
    check("all 4 rollouts have DISTINCT initial noise (fresh per query)",
          len(set(vals)) == 4, f"noise={vals}")
    c.close()


def test_multi_turn_sync_num_envs_one():
    """group_size=3, num_envs=1 → SyncVectorEnv, 3 turns; bundle still captured."""
    print("\n[Turn] num_envs==1 with group_size>1 (SyncVectorEnv multi-turn)")
    c = _make_collector(group_size=3, num_async_vector_env=1)
    check("num_envs == 1", c.num_envs == 1)
    check("turns_per_group == 3", c.turns_per_group == 3)
    check("uses SyncVectorEnv (not async)", c._uses_async is False)

    eps = c._collect_one_group(group_seed=500, group_id=1, success_weight=1.0)
    env = c.envs
    check("group_size (3) episodes returned", len(eps) == 3, f"got {len(eps)}")
    check("all share group_id==1", all(e["group_id"] == 1 for e in eps))
    # Unification fix: group_size>1 captures+applies env0 bundle even at num_envs==1.
    n_get = sum(1 for m, _ in env.calls if m == "get_scene_bundle")
    check("get_scene_bundle captured once (not skipped at num_envs==1)", n_get == 1, f"{n_get}")
    check("apply_scene_bundle once per turn (3)", len(env.apply_bundle_args) == 3, f"{len(env.apply_bundle_args)}")
    ids = [b.get("id") for b in env.apply_bundle_args]
    check("same captured bundle re-applied each turn", len(set(ids)) == 1, f"ids={ids}")
    # Regression (Finding 2): SyncVectorEnv (num_envs==1) is the exact case the
    # deep-copy guards — each of the 3 applies must be a distinct fresh object.
    check("each turn applies a fresh copy (distinct objects)",
          len({id(b) for b in env.apply_bundle_args}) == 3,
          f"{len({id(b) for b in env.apply_bundle_args})} distinct")
    vals = _noise_values(eps)
    check("3 distinct initial noises across turns", len(set(vals)) == 3, f"noise={vals}")
    check("no autoreset leaked across turn boundaries (SyncVectorEnv)",
          env.autoreset_fired == 0, f"autoreset_fired={env.autoreset_fired}")
    c.close()


def test_singleton_group():
    """group_size=1 → 1 turn, no-bundle fast path (true singleton)."""
    print("\n[Turn] Singleton group_size==1 (no bundle needed)")
    c = _make_collector(group_size=1, num_async_vector_env=1)
    check("turns_per_group == 1", c.turns_per_group == 1)
    eps = c._collect_one_group(group_seed=7, group_id=0, success_weight=1.0)
    env = c.envs
    check("1 episode returned", len(eps) == 1, f"got {len(eps)}")
    # Singleton path returns reset obs directly — no scene-bundle RPCs.
    n_get = sum(1 for m, _ in env.calls if m == "get_scene_bundle")
    check("no get_scene_bundle (singleton fast path)", n_get == 0, f"{n_get}")
    check("no apply_scene_bundle (singleton fast path)", len(env.apply_bundle_args) == 0)
    c.close()


def test_fast_forward_multi_turn():
    """FF prefix runs ONCE (turn 1); turns 2..k restart from the captured
    post-FF bundle rather than re-running the lockstep prefix."""
    print("\n[Turn] Fast-forward multi-turn (group_size=4, num_envs=2, ff=2)")
    c = _make_collector(group_size=4, num_async_vector_env=2)
    c.envs.terminate_after = 2  # FF prefix (2 lockstep steps) stays non-terminal

    eps = c._collect_one_group_with_fast_forward(
        group_seed=300, group_id=5, fast_forward_steps=2, success_weight=1.0,
    )
    env = c.envs
    check("group_size (4) episodes returned", len(eps) == 4, f"got {len(eps)}")
    check("all share group_id==5", all(e["group_id"] == 5 for e in eps))

    # Step calls = ff_prefix (2, ONCE) + one per-env-loop step per turn (2).
    # If FF were re-run per turn it would be 2*2 + 2 = 6. Asserting 4 proves the
    # prefix ran exactly once.
    check("step count == ff_steps(2) + turns(2) == 4 (FF prefix runs once)",
          env.step_count == 4, f"{env.step_count}")

    # get_scene_bundle twice: align capture (turn1) + post-FF capture.
    n_get = sum(1 for m, _ in env.calls if m == "get_scene_bundle")
    check("get_scene_bundle called twice (align + post-FF)", n_get == 2, f"{n_get}")
    check("apply_scene_bundle once per turn (2)", len(env.apply_bundle_args) == 2, f"{len(env.apply_bundle_args)}")
    # Turn 2 restart applies a FRESH COPY of the POST-FF bundle (2nd
    # get_scene_bundle's [0]), NOT the original seed-aligned bundle from turn 1.
    post_ff_first = env.get_bundle_returns[1][0]
    applied = env.apply_bundle_args[-1]
    check("turn-2 restart uses the post-FF captured bundle (id match)",
          applied.get("id") == post_ff_first.get("id"),
          f"applied id={applied.get('id')}, post_ff id={post_ff_first.get('id')}")
    # Regression (Finding 2): must be a fresh deep copy, not the captured object,
    # so robosuite's in-place ep_meta mutation can't pollute the branch point.
    check("turn-2 bundle is a fresh copy, not the captured object (no aliasing)",
          applied is not post_ff_first)
    # Regression (Finding 1): turns 2..k must restore the SAME reduced truncation
    # budget the FF prefix left turn 1 with (fast_forward_steps * n_action_steps),
    # else turns 2..k run a longer episode horizon than turn 1 within one group.
    check("turn-2 bundle carries post-FF consumed_substeps budget (ff_steps*n_action_steps)",
          applied.get("consumed_substeps") == 2 * 8,
          f"got {applied.get('consumed_substeps')} (expected {2 * 8})")
    # The seed-aligned bundle applied during _align (turn-1 setup) must NOT carry
    # a consumed budget — only the post-FF capture is stamped.
    check("seed-align bundle has no consumed_substeps stamp",
          env.apply_bundle_args[0].get("consumed_substeps") is None)
    check("no autoreset leaked across turn boundary (FF multi-turn)",
          env.autoreset_fired == 0, f"autoreset_fired={env.autoreset_fired}")
    c.close()


def test_init_state_refetches_each_turn():
    """init-state mode re-fetches a FRESH bundle each turn (preserving the
    'fresh copy per apply' invariant), and never calls get_scene_bundle."""
    print("\n[Turn] init-state mode re-fetches fresh bundle per turn")
    c = _make_collector(group_size=4, num_async_vector_env=2)
    c._active_init_bundle_path = "/fake/init.npz"

    fetch_counter = {"n": 0}

    def _fake_get_init_bundle(path):
        fetch_counter["n"] += 1
        return {"id": f"init-{fetch_counter['n']}", "consumed_substeps": 0}

    c._get_init_bundle = _fake_get_init_bundle

    eps = c._collect_one_group(group_seed=900, group_id=2, success_weight=1.0)
    env = c.envs
    check("group_size (4) episodes returned", len(eps) == 4, f"got {len(eps)}")
    check("all share group_id==2", all(e["group_id"] == 2 for e in eps))
    # init-state skips env-0 scene capture entirely.
    n_get = sum(1 for m, _ in env.calls if m == "get_scene_bundle")
    check("get_scene_bundle NOT called in init-state mode", n_get == 0, f"{n_get}")
    # _get_init_bundle fetched once per turn (turn1 in _align + turn2 in restart).
    check("init bundle re-fetched once per turn (2)", fetch_counter["n"] == 2, f"{fetch_counter['n']}")
    ids = [b.get("id") for b in env.apply_bundle_args]
    check("each turn applies a FRESH init bundle (distinct ids)",
          len(set(ids)) == 2, f"ids={ids}")
    check("no autoreset leaked across turn boundary (init-state multi-turn)",
          env.autoreset_fired == 0, f"autoreset_fired={env.autoreset_fired}")
    c.close()


def test_autoreset_disabled_requested():
    """The collector must build the vector env with autoreset_mode=DISABLED on
    BOTH backends — the load-bearing fix that stops NEXT_STEP autoreset from
    eating the branch-point restart across turns/groups (critically for
    AsyncVectorEnv with shared_memory=False, where a vector reset() does NOT
    clear the worker's pending autoreset flag)."""
    print("\n[Turn] autoreset_mode=DISABLED requested on both backends")
    DISABLED = ce.gym.vector.AutoresetMode.DISABLED
    c_async = _make_collector(group_size=4, num_async_vector_env=2)  # → AsyncVectorEnv
    check("async path requests autoreset_mode=DISABLED",
          c_async.envs.autoreset_mode == DISABLED, f"{c_async.envs.autoreset_mode}")
    c_async.close()
    c_sync = _make_collector(group_size=3, num_async_vector_env=1)   # → SyncVectorEnv
    check("sync path requests autoreset_mode=DISABLED",
          c_sync.envs.autoreset_mode == DISABLED, f"{c_sync.envs.autoreset_mode}")
    c_sync.close()


def test_ff_fallback_yields_full_group():
    """If an env terminates DURING the FF prefix, the collector falls back to a
    normal seed-aligned multi-turn group — and must still yield group_size
    episodes (the fallback re-runs the FULL multi-turn collection from seed)."""
    print("\n[Turn] FF early-termination falls back to full multi-turn group")
    c = _make_collector(group_size=4, num_async_vector_env=2)
    # terminate_after=0 → the very first lockstep FF step terminates, so
    # _lockstep_step returns None and _collect_one_group_with_fast_forward
    # falls back to _collect_one_group.
    c.envs.terminate_after = 0

    eps = c._collect_one_group_with_fast_forward(
        group_seed=400, group_id=7, fast_forward_steps=3, success_weight=1.0,
    )
    env = c.envs
    check("fallback still yields group_size (4) episodes", len(eps) == 4, f"got {len(eps)}")
    check("all share group_id==7", all(e["group_id"] == 7 for e in eps))
    check("all share env_seed==400", all(e["env_seed"] == 400 for e in eps))
    # The fallback re-runs _align (a second reset/get_scene_bundle), so there are
    # two resets: the FF attempt's align + the fallback's align.
    check("fallback re-aligned (>=2 resets)", len(env.reset_calls) >= 2, f"{len(env.reset_calls)}")
    check("no autoreset leaked across turn boundary (FF fallback path)",
          env.autoreset_fired == 0, f"autoreset_fired={env.autoreset_fired}")
    c.close()


def test_collect_driver_multi_group_multi_turn():
    """End-to-end collect() over multiple groups with multi-turn collection:
    group_ids increment per group, each group yields group_size episodes across
    its turns, and per-group env_seeds follow GROUP_SEED_STRIDE."""
    print("\n[Turn] collect() driver: 2 groups x (group_size=4 over 2 turns)")
    c = _make_collector(group_size=4, num_async_vector_env=2)

    eps = c.collect(
        num_groups=2,
        base_seed=1000,
        success_weight=1.0,
        fast_forward_steps=0,      # FF disabled → normal seed-aligned groups
        fast_forward_pct=0.0,
        min_alive_groups=0,   # no dynamic extension → exactly num_groups
        max_groups=None,
    )
    check("total episodes == num_groups * group_size (8)", len(eps) == 8, f"got {len(eps)}")
    by_group: dict = {}
    for e in eps:
        by_group.setdefault(e["group_id"], []).append(e)
    check("two distinct group_ids {0,1}", set(by_group) == {0, 1}, f"{sorted(by_group)}")
    check("each group has group_size (4) episodes",
          all(len(v) == 4 for v in by_group.values()),
          f"{ {g: len(v) for g, v in by_group.items()} }")
    # Per-group seeds follow base_seed + group_idx * GROUP_SEED_STRIDE.
    seeds_by_group = {g: {e["env_seed"] for e in v} for g, v in by_group.items()}
    check("group 0 env_seed == base_seed (1000)", seeds_by_group[0] == {1000}, f"{seeds_by_group[0]}")
    check("group 1 env_seed == base_seed + GROUP_SEED_STRIDE (2000)",
          seeds_by_group[1] == {1000 + ce.GROUP_SEED_STRIDE}, f"{seeds_by_group[1]}")
    check("no autoreset leaked across any turn/group boundary",
          c.envs.autoreset_fired == 0, f"autoreset_fired={c.envs.autoreset_fired}")
    c.close()


def test_staggered_termination_within_turn():
    """Envs finishing at DIFFERENT times within a turn: the collector must keep
    stepping the still-active env while the finished one is excluded (active_indices)
    and ignored — producing one episode per env with the right per-env chunk counts,
    no crash. (This mid-turn done-while-sibling-active state is what the real
    autoreset/DISABLED handling and the MultiStepWrapper no-op guard protect.)"""
    print("\n[Turn] staggered within-turn termination (collector bookkeeping)")
    c = _make_collector(group_size=2, num_async_vector_env=2)  # 1 turn, 2 envs
    c.envs.terminate_at = [0, 2]  # env0 done after outer step 1; env1 after step 3
    eps = c._collect_one_group(group_seed=42, group_id=9, success_weight=1.0)
    check("2 episodes returned", len(eps) == 2, f"got {len(eps)}")
    check("both share group_id==9", all(e["group_id"] == 9 for e in eps))
    chunk_counts = sorted(len(e["actions"]) for e in eps)
    check("envs recorded different chunk counts (staggered: [1, 3])",
          chunk_counts == [1, 3], f"got {chunk_counts}")
    check("no autoreset fired (DISABLED) and no crash on the done env",
          c.envs.autoreset_fired == 0, f"autoreset_fired={c.envs.autoreset_fired}")
    c.close()


def test_multistep_no_op_when_done():
    """Under autoreset_mode=DISABLED the collector steps already-done envs (zero
    actions) while sibling envs are still active, so the REAL MultiStepWrapper.step
    MUST no-op cleanly when called on a done env rather than raise. (The pre-fix
    wrapper raised UnboundLocalError — env_state/truncated unbound — when the
    n_action_steps loop broke on its first iteration; the old NEXT_STEP autoreset
    hid this by resetting the env instead of stepping it.)"""
    print("\n[Turn] real MultiStepWrapper.step no-ops (no crash) when stepped while done")
    import gymnasium as gym
    from gymnasium import spaces
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

    class _MiniEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Dict({
                "video.cam": spaces.Box(0, 255, (4, 4, 3), np.uint8),
                "state.x": spaces.Box(-1, 1, (2,), np.float32),
            })
            self.action_space = spaces.Dict({"a": spaces.Box(-1, 1, (1,), np.float32)})

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return {"video.cam": np.zeros((4, 4, 3), np.uint8),
                    "state.x": np.zeros(2, np.float32)}, {"success": False}

        def step(self, action):  # ignore_done-style: never terminates on its own
            return ({"video.cam": np.zeros((4, 4, 3), np.uint8),
                     "state.x": np.zeros(2, np.float32)},
                    0.0, False, False, {"success": False})

    w = MultiStepWrapper(_MiniEnv(), video_delta_indices=np.array([0]),
                         state_delta_indices=np.array([0]), n_action_steps=4,
                         max_episode_steps=8, terminate_on_success=True)
    w.reset(seed=0)
    act = {"a": np.zeros((4, 1), np.float32)}
    done = False
    for _ in range(4):  # drive to truncation (8/4 = 2 outer steps) → self.done[-1]=True
        _, _, d, t, _ = w.step(act)
        done = bool(d or t)
        if done:
            break
    check("env reached a done state (self.done[-1] True)", bool(w.done) and w.done[-1])
    # Step AGAIN on the done env — must not raise (this is what the collector does).
    try:
        obs, rew, d2, t2, info = w.step(act)
        check("stepping a done MultiStepWrapper returns cleanly (no UnboundLocalError)", True)
        check("the no-op still reports done=True", bool(d2))
    except Exception as e:  # noqa: BLE001
        check("stepping a done MultiStepWrapper returns cleanly (no UnboundLocalError)",
              False, f"raised {type(e).__name__}: {e}")


def test_invalid_num_envs_rejected():
    """EpisodeCollector mirrors the config validation for standalone CLI runs."""
    print("\n[Turn] Invalid num_async_vector_env rejected at construction")
    for gs, nave, label in [(4, 3, "non-divisor"), (4, 5, "exceeds group_size"), (4, 0, "< 1")]:
        try:
            _make_collector(group_size=gs, num_async_vector_env=nave)
            check(f"group_size={gs}, num_envs={nave} ({label}) raises", False, "no error raised")
        except ValueError:
            check(f"group_size={gs}, num_envs={nave} ({label}) raises ValueError", True)
        except Exception as e:  # noqa: BLE001
            check(f"group_size={gs}, num_envs={nave} ({label}) raises ValueError",
                  False, f"got {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_unchanged_when_num_envs_equals_group_size()
    test_multi_turn_divisible()
    test_multi_turn_sync_num_envs_one()
    test_singleton_group()
    test_fast_forward_multi_turn()
    test_init_state_refetches_each_turn()
    test_autoreset_disabled_requested()
    test_ff_fallback_yields_full_group()
    test_collect_driver_multi_group_multi_turn()
    test_staggered_termination_within_turn()
    test_multistep_no_op_when_done()
    test_invalid_num_envs_rejected()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} test(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\033[32mAll turn-collection tests passed.\033[0m")
