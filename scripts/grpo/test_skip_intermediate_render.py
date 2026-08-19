"""Tests for MultiStepWrapper's skip_intermediate_render mode.

Only ONE camera frame per action chunk is ever read (`video_delta_indices=[0]`
→ `_get_obs` returns `self.obs[-1]`), so the wrapper keeps the robosuite camera
observables disabled for the whole chunk and takes that frame from a forced
render after the last substep. For PandaOmron that removes 21 of every 24
renders (3 cameras x 8 substeps).

The load-bearing detail is robosuite's `Observable` sampling state machine, so
these tests drive it rather than a hand-waved stand-in:

  * If `robosuite` is importable (the robocasa venv), the REAL
    `robosuite.utils.observables.Observable` is used.
  * Otherwise an embedded model of it runs, and
    `test_embedded_observable_matches_robosuite` guards against drift whenever
    the real class is available.

Either way no MuJoCo, GPU, or rendering is needed:

    .venv/bin/python scripts/grpo/test_skip_intermediate_render.py
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper  # noqa: E402

N_CAMS = 3
FRAME_HW = 4
CAM_KEYS = [f"video.cam_{i}" for i in range(N_CAMS)]
LANG_KEY = "annotation.human.action.task_description"
REAL_FRAME_SENTINEL = 255

# robosuite runs several physics substeps per control step (25 for PandaOmron:
# 20 Hz control over a 500 Hz sim). Only "more than one" matters for the
# sampling behavior under test, so use 5 to keep the physics-tick tags these
# tests stamp into frames inside uint8's range.
CONTROL_FREQ = 20
MODEL_DT = 0.01
N_PHYSICS_SUBSTEPS = int((1.0 / CONTROL_FREQ) / MODEL_DT)


# ---------------------------------------------------------------------------
# Observable: the real robosuite class when available, else a faithful model
# ---------------------------------------------------------------------------


class EmbeddedObservable:
    """Model of robosuite's `Observable` (utils/observables.py).

    Mirrors the three behaviors this feature depends on, and nothing else:

      * `update()` computes only when `_enabled`, and only when
        `not _sampled and _sampling_timestep - _current_delay >= _time_since_last_sample`
        — or unconditionally when `force=True`.
      * `_sampled` clears only after the timer crosses `_sampling_timestep`.
      * `reset()` (which `set_enabled()` calls) zeroes the timer and installs a
        float64 zeros value, but does NOT clear `_sampled`.

    That last point is why re-enabling mid-chunk cannot work; see
    MultiStepWrapper.step.
    """

    def __init__(self, name, sensor, sampling_rate=CONTROL_FREQ, data_shape=(1,)):
        self.name = name
        self._sensor = sensor
        self._sampling_timestep = 1.0 / sampling_rate
        self._data_shape = data_shape
        self._enabled = True
        self._active = True
        self._time_since_last_sample = 0.0
        self._current_delay = 0.0
        self._current_observed_value = np.zeros(self._data_shape)
        self._sampled = False

    def update(self, timestep, obs_cache, force=False):
        if self._enabled:
            self._time_since_last_sample += timestep
            if (
                not self._sampled
                and self._sampling_timestep - self._current_delay
                >= self._time_since_last_sample
            ) or force:
                self._current_observed_value = np.array(self._sensor(obs_cache))
                obs_cache[self.name] = np.array(self._current_observed_value)
                self._sampled = True
            if self._time_since_last_sample >= self._sampling_timestep:
                if not self._sampled:
                    self._current_observed_value = np.array(self._sensor(obs_cache))
                    obs_cache[self.name] = np.array(self._current_observed_value)
                self._time_since_last_sample %= self._sampling_timestep
                self._sampled = False

    def reset(self):
        self._time_since_last_sample = 0.0
        self._current_delay = 0.0
        self._current_observed_value = np.zeros(self._data_shape)

    def is_enabled(self):
        return self._enabled

    def is_active(self):
        return self._active

    def set_enabled(self, enabled):
        self._enabled = enabled
        self.reset()

    @property
    def obs(self):
        return self._current_observed_value


def _real_observable_factory():
    """Return a factory for the real robosuite Observable, or None."""
    try:
        from robosuite.utils.observables import Observable, sensor
    except Exception:
        return None

    def make(name, sensor_fn, sampling_rate=CONTROL_FREQ, data_shape=(1,)):
        return Observable(
            name, sensor(modality="image")(sensor_fn), sampling_rate=sampling_rate
        )

    return make


def _embedded_factory(name, sensor_fn, sampling_rate=CONTROL_FREQ, data_shape=(1,)):
    return EmbeddedObservable(
        name, sensor_fn, sampling_rate=sampling_rate, data_shape=data_shape
    )


REAL_FACTORY = _real_observable_factory()
OBSERVABLE_FACTORY = REAL_FACTORY or _embedded_factory


# ---------------------------------------------------------------------------
# Fake env: robosuite's step/reset observable plumbing, no MuJoCo
# ---------------------------------------------------------------------------


class FakeSimEnv(gym.Env):
    """Stand-in for GrootRoboCasaEnv over an observable-driven robosuite env.

    `tick` counts physics steps and tags every rendered frame, so a test can see
    WHICH sim state a kept frame came from — the difference between a correct
    implementation and one that renders a control step too early.
    """

    def __init__(self, terminate_at_substep=None, success_at_substep=None):
        self.observation_space = spaces.Dict(
            {
                **{
                    k: spaces.Box(0, 255, (FRAME_HW, FRAME_HW, 3), np.uint8)
                    for k in CAM_KEYS
                },
                "state.pos": spaces.Box(-1e9, 1e9, (3,), np.float32),
                LANG_KEY: spaces.Text(64),
            }
        )
        self.action_space = spaces.Dict(
            {"arm": spaces.Box(-1, 1, (3,), np.float32)}
        )
        self.terminate_at_substep = terminate_at_substep
        self.success_at_substep = success_at_substep

        self.tick = 0            # physics steps elapsed
        self.substep = 0         # control steps elapsed
        self.render_count = 0
        self.render_ticks = []
        self._obs_cache = {}
        self._observables = {
            **{
                f"cam_{i}_image": OBSERVABLE_FACTORY(
                    f"cam_{i}_image",
                    lambda cache, i=i: self._render(),
                    data_shape=(FRAME_HW, FRAME_HW, 3),
                )
                for i in range(N_CAMS)
            },
            "state_pos": OBSERVABLE_FACTORY(
                "state_pos",
                lambda cache: np.full(3, self.tick, dtype=np.float64),
                data_shape=(3,),
            ),
        }

    # --- rendering ---------------------------------------------------------
    def _render(self):
        """Frame carries a REAL_FRAME_SENTINEL pixel plus the physics tick.

        The sentinel distinguishes a genuinely rendered frame from an
        observable's zero-filled `_current_observed_value` even at tick 0; the
        tick pixel says which sim state the frame shows.
        """
        self.render_count += 1
        self.render_ticks.append(self.tick)
        frame = np.zeros((FRAME_HW, FRAME_HW, 3), dtype=np.uint8)
        frame[0, 0, :] = REAL_FRAME_SENTINEL
        frame[1, 1, :] = self.tick
        return frame

    # --- robosuite-side plumbing ------------------------------------------
    def _update_observables(self, force=False):
        for o in self._observables.values():
            o.update(timestep=MODEL_DT, obs_cache=self._obs_cache, force=force)

    def _get_observations(self, force_update=False):
        if force_update:
            self._update_observables(force=True)
        return {
            n: o.obs
            for n, o in self._observables.items()
            if o.is_enabled() and o.is_active()
        }

    # --- the interface skip_intermediate_render requires --------------------
    def set_camera_obs_enabled(self, enabled):
        for i in range(N_CAMS):
            o = self._observables[f"cam_{i}_image"]
            if o.is_enabled() != enabled:
                o.set_enabled(enabled)

    def recompute_observation(self):
        self.set_camera_obs_enabled(True)
        return self._to_groot(self._get_observations(force_update=True))

    # --- gym API -----------------------------------------------------------
    def _to_groot(self, raw):
        """Mirror get_basic_observation + get_groot_observation.

        Crucially, a camera key robosuite omitted (observable disabled) is
        BACKFILLED with a blank frame rather than dropped: gymnasium's
        PassiveEnvChecker asserts exact key-set equality on every step.
        """
        obs = {"state.pos": np.asarray(raw["state_pos"], dtype=np.float32),
               LANG_KEY: "do the thing"}
        for i, key in enumerate(CAM_KEYS):
            raw_key = f"cam_{i}_image"
            obs[key] = (
                np.asarray(raw[raw_key])
                if raw_key in raw
                else np.zeros((FRAME_HW, FRAME_HW, 3), dtype=np.uint8)
            )
        return obs

    def reset(self, seed=None, options=None):
        self.tick = 0
        self.substep = 0
        self._obs_cache = {}
        self.set_camera_obs_enabled(True)
        for o in self._observables.values():
            o.reset()
        # robosuite's reset ends with a forced update (environments/base.py).
        return self._to_groot(self._get_observations(force_update=True)), {
            "success": False
        }

    def step(self, action):
        self.substep += 1
        for _ in range(N_PHYSICS_SUBSTEPS):
            self.tick += 1
            self._update_observables()
        terminated = self.substep == self.terminate_at_substep
        success = self.substep == self.success_at_substep
        return (
            self._to_groot(self._get_observations()),
            0.0,
            terminated,
            False,
            {"success": success},
        )


class PassthroughWrapper(gym.Wrapper):
    """Stands in for the OrderEnforcing/PassiveEnvChecker wrappers gym.make()
    adds — gymnasium >= 1.0 does not forward attribute access through them."""


class FrameConsumingWrapper(gym.Wrapper):
    """Stands in for VideoRecordingWrapper (reads every substep's frames)."""

    consumes_every_substep_obs = True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _action(n_action_steps):
    return {"arm": np.zeros((n_action_steps, 3), dtype=np.float32)}


def _make(
    n_action_steps=8,
    skip=True,
    wrap=False,
    video_horizon=1,
    max_episode_steps=None,
    frame_consumer=False,
    **env_kwargs,
):
    env = FakeSimEnv(**env_kwargs)
    inner = env
    if wrap:
        inner = PassthroughWrapper(inner)
    if frame_consumer:
        inner = FrameConsumingWrapper(inner)
    wrapper = MultiStepWrapper(
        inner,
        video_delta_indices=np.array(list(range(-(video_horizon - 1), 1))),
        state_delta_indices=np.array([0]),
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
        terminate_on_success=True,
        skip_intermediate_render=skip,
    )
    return env, wrapper


def _frames_of(obs):
    """(n_cams, 1, H, W, 3) stack of the chunk's kept frames."""
    return np.stack([obs[k] for k in CAM_KEYS])


def _assert_real_frames(obs, context=""):
    """Every camera key must hold a genuinely rendered uint8 frame."""
    where = f" ({context})" if context else ""
    for key in CAM_KEYS:
        frame = np.asarray(obs[key])
        assert frame.dtype == np.uint8, (
            f"{key} is {frame.dtype}, not uint8{where} — an observable that "
            f"never sampled returns float64 zeros"
        )
        assert frame.reshape(-1, FRAME_HW, FRAME_HW, 3)[0, 0, 0, 0] == (
            REAL_FRAME_SENTINEL
        ), f"{key} is not a rendered frame{where}"


def _frame_tick(obs):
    """Physics tick the kept frame was rendered at (see FakeSimEnv._render)."""
    ticks = {
        int(np.asarray(obs[k]).reshape(-1, FRAME_HW, FRAME_HW, 3)[0, 1, 1, 0])
        for k in CAM_KEYS
    }
    assert len(ticks) == 1, f"cameras disagree on the render tick: {ticks}"
    return ticks.pop()


def _chunk_report(skip, n_chunks=4, n_action_steps=8, **kwargs):
    """Run n_chunks and return per-chunk (frame tag, state tag, dtype, renders)."""
    env, wrapper = _make(n_action_steps=n_action_steps, skip=skip, **kwargs)
    wrapper.reset()
    out = []
    for _ in range(n_chunks):
        before = env.render_count
        obs, _, done, _, _ = wrapper.step(_action(n_action_steps))
        has_frames = all(
            np.asarray(obs[k]).dtype == np.uint8 for k in CAM_KEYS
        )
        out.append(
            {
                "frame_tick": _frame_tick(obs) if has_frames else None,
                "frame_dtype": _frames_of(obs).dtype,
                "state_tick": int(np.asarray(obs["state.pos"]).max()),
                "renders": env.render_count - before,
                "done": bool(done),
            }
        )
        if done:
            break
    return out


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_embedded_observable_matches_robosuite():
    """Guard the embedded model against drift from the real class."""
    if REAL_FACTORY is None:
        print("  [SKIP] robosuite not importable; embedded model in use")
        return
    for force_at_start in (False, True):
        seqs = []
        for factory in (REAL_FACTORY, _embedded_factory):
            calls = []
            o = factory("x", lambda cache: calls.append("s") or np.zeros(3),
                        data_shape=(3,))
            cache = {}
            if force_at_start:
                o.update(MODEL_DT, cache, force=True)
            marks = []
            for step in range(3 * N_PHYSICS_SUBSTEPS):
                n = len(calls)
                o.update(MODEL_DT, cache)
                marks.append(len(calls) > n)
            seqs.append(marks)
        assert seqs[0] == seqs[1], (
            f"embedded model diverges from robosuite (force={force_at_start})"
        )
    print("  [PASS] embedded Observable matches robosuite's sampling behavior")


def test_baseline_renders_every_substep():
    """Without the flag: every substep renders (the behavior we're replacing)."""
    env, wrapper = _make(n_action_steps=8, skip=False)
    wrapper.reset()
    env.render_count = 0
    wrapper.step(_action(8))
    assert env.render_count == 8 * N_CAMS, env.render_count
    print(f"  [PASS] baseline renders {env.render_count} frames per chunk")


def test_skip_renders_once_per_chunk():
    env, wrapper = _make(n_action_steps=8, skip=True)
    wrapper.reset()
    env.render_count = 0
    obs, _, _, _, info = wrapper.step(_action(8))
    assert env.render_count == N_CAMS, env.render_count
    assert env.substep == 8, "all substeps must still be simulated"
    assert len(info["dones"]) == 8, info["dones"]
    print(f"  [PASS] skip renders {env.render_count} frames per chunk")


def test_frame_provenance_and_dtype_match_baseline():
    """THE regression test: the kept frame must come from the same sim state as
    the unskipped path, and stay uint8.

    Catches both ways the observable state machine bites: a re-enabled
    observable that cannot sample for a full control step (chunk 1 renders
    nothing and yields reset()'s float64 zeros), and a sample that lands on the
    first physics substep instead of the last (one control step stale).
    """
    base = _chunk_report(skip=False)
    skipped = _chunk_report(skip=True)
    for i, (b, s) in enumerate(zip(base, skipped)):
        assert s["renders"] > 0, f"chunk {i}: no render happened at all"
        assert s["frame_dtype"] == np.uint8, (
            f"chunk {i}: frames are {s['frame_dtype']}, not uint8 "
            f"(an unsampled observable returns float64 zeros)"
        )
        assert s["frame_tick"] == b["frame_tick"], (
            f"chunk {i}: frame came from sim tick {s['frame_tick']}, "
            f"baseline used {b['frame_tick']}"
        )
        assert s["state_tick"] == b["state_tick"], f"chunk {i}: state drifted"
        # The frame and the state in one observation must describe one instant.
        assert s["frame_tick"] == s["state_tick"], (
            f"chunk {i}: frame tick {s['frame_tick']} != state tick "
            f"{s['state_tick']} — image and proprio are out of sync"
        )
    print(
        f"  [PASS] frame provenance + dtype identical to baseline over "
        f"{len(base)} chunks (ticks {[b['frame_tick'] for b in base]})"
    )


def test_first_chunk_after_reset_is_real():
    """Isolate the reset case: reset() ends with a forced update, which is what
    leaves the observable unable to sample for a full control step."""
    env, wrapper = _make(n_action_steps=8, skip=True)
    wrapper.reset()
    obs, _, _, _, _ = wrapper.step(_action(8))
    _assert_real_frames(obs, "first chunk after reset")
    print("  [PASS] first chunk after reset carries a real frame")


def test_first_chunk_after_forced_update_is_real():
    """Same, for the scene-restore path: apply_scene_bundle reads observations
    with force_update=True mid-episode."""
    env, wrapper = _make(n_action_steps=8, skip=True)
    wrapper.reset()
    wrapper.step(_action(8))
    env._get_observations(force_update=True)   # mimic apply_scene_bundle step 5
    obs, _, _, _, _ = wrapper.step(_action(8))
    _assert_real_frames(obs, "chunk after a forced update")
    print("  [PASS] first chunk after a forced update carries a real frame")


def test_every_substep_keeps_the_full_key_set():
    """EVERY substep observation must carry the full key set, skipped or not.

    gymnasium's PassiveEnvChecker (inserted by gym.make between MultiStepWrapper
    and the base env) asserts observation keys == observation-space keys on
    every step, so dropping the video keys on skipped substeps kills the worker.
    """
    env, wrapper = _make(n_action_steps=8, skip=True)
    wrapper.reset()
    seen = []
    real_step = FakeSimEnv.step

    def spy(self, action):
        out = real_step(self, action)
        seen.append(sorted(out[0]))
        return out

    FakeSimEnv.step = spy
    try:
        wrapper.step(_action(8))
    finally:
        FakeSimEnv.step = real_step
    expected = sorted(CAM_KEYS + ["state.pos", LANG_KEY])
    assert len(seen) == 8, seen
    for i, keys in enumerate(seen):
        assert keys == expected, f"substep {i} key set changed: {keys}"
    print("  [PASS] every substep observation keeps the full key set")


def test_gym_make_chain_with_passive_env_checker():
    """THE regression test for the production wrapper chain.

    The collector builds envs with gym.make(), which inserts OrderEnforcing and
    PassiveEnvChecker between MultiStepWrapper and the base env. Earlier tests
    wrapped a bare env, so they never exercised the checker — and the checker is
    what failed on the real stack.
    """
    env_id = "test/SkipRenderFake-v0"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point=lambda **kw: FakeSimEnv(**kw))

    made = gym.make(env_id)
    chain = [type(t).__name__ for t in MultiStepWrapper._walk_chain(made)]
    assert "PassiveEnvChecker" in chain, (
        f"this test is only meaningful with the checker in the chain: {chain}"
    )

    wrapper = MultiStepWrapper(
        made,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=8,
        terminate_on_success=True,
        skip_intermediate_render=True,
    )
    base = made.unwrapped
    assert wrapper._render_gate is base, wrapper._render_gate

    wrapper.reset(seed=0)
    for chunk in range(3):
        obs, _, done, _, _ = wrapper.step(_action(8))
        _assert_real_frames(obs, f"gym.make chain, chunk {chunk}")
        if done:
            break
    wrapper.close()
    print(f"  [PASS] survives the real gym.make chain ({' -> '.join(chain)})")


def test_gate_lookup_emits_no_warnings():
    """Attribute probing must not rely on wrapper forwarding.

    gymnasium 0.29.1 (pinned by the robocasa collector venv) forwards unknown
    attributes down the chain and warns on every lookup; 1.x dropped forwarding.
    Probing type()/__dict__ instead is silent and correct on both.
    """
    import warnings

    env_id = "test/SkipRenderFake-v0"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point=lambda **kw: FakeSimEnv(**kw))
    made = gym.make(env_id)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gate = MultiStepWrapper._find_render_gate(made)
        consumer = MultiStepWrapper._find_substep_obs_consumer(made)
    made.close()
    assert gate is not None and consumer is None
    offenders = [
        str(w.message)
        for w in caught
        if "set_camera_obs_enabled" in str(w.message)
        or "recompute_observation" in str(w.message)
        or "consumes_every_substep_obs" in str(w.message)
    ]
    assert not offenders, f"gate lookup triggered forwarding warnings: {offenders[:2]}"
    print("  [PASS] gate lookup probes types directly (no forwarding warnings)")


def test_sync_vector_env_roundtrip():
    """The obs-space boundary is where a wrong dtype/shape/key-set blows up.

    Built through gym.make() so the chain matches what _make_collector_env
    produces in production (OrderEnforcing + PassiveEnvChecker included) — a
    bare env here would not exercise the checker.
    """
    env_id = "test/SkipRenderFake-v0"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point=lambda **kw: FakeSimEnv(**kw))

    def make():
        return MultiStepWrapper(
            gym.make(env_id),
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=8,
            terminate_on_success=True,
            skip_intermediate_render=True,
        )

    venv = SyncVectorEnv([make, make])
    try:
        venv.reset(seed=[0, 1])
        obs, _, _, _, _ = venv.step(
            {"arm": np.zeros((2, 8, 3), dtype=np.float32)}
        )
        for i in range(2):
            _assert_real_frames(
                {k: obs[k][i] for k in CAM_KEYS}, f"venv env {i}"
            )
    finally:
        venv.close()
    print("  [PASS] SyncVectorEnv round-trip keeps uint8, non-blank frames")


def test_early_termination_still_returns_real_frames():
    env, wrapper = _make(n_action_steps=8, skip=True, terminate_at_substep=3)
    wrapper.reset()
    env.render_count = 0
    obs, _, done, _, info = wrapper.step(_action(8))
    assert done and env.substep == 3, (done, env.substep)
    assert env.render_count == N_CAMS, env.render_count
    _assert_real_frames(obs, "early termination")
    assert len(info["dones"]) == 3, info["dones"]
    print("  [PASS] early termination renders the terminal observation")


def test_truncation_mid_chunk_still_returns_real_frames():
    env, wrapper = _make(n_action_steps=8, skip=True, max_episode_steps=5)
    wrapper.reset()
    env.render_count = 0
    obs, _, _, _, _ = wrapper.step(_action(8))
    assert env.substep == 5, env.substep
    assert env.render_count == N_CAMS, env.render_count
    _assert_real_frames(obs, "mid-chunk truncation")
    print("  [PASS] mid-chunk truncation renders the terminal observation")


def test_camera_obs_left_enabled_after_step():
    env, wrapper = _make(n_action_steps=8, skip=True, terminate_at_substep=2)
    wrapper.reset()
    wrapper.step(_action(8))
    assert all(
        env._observables[f"cam_{i}_image"].is_enabled() for i in range(N_CAMS)
    ), "cameras left disabled after a terminating chunk"
    wrapper.step(_action(8))   # no-op step on a done env
    assert all(
        env._observables[f"cam_{i}_image"].is_enabled() for i in range(N_CAMS)
    ), "cameras left disabled after a no-op step"
    obs, _ = wrapper.reset()
    _assert_real_frames(obs, "after reset")
    print("  [PASS] camera obs restored on every exit path")


def test_resolves_gate_through_wrapper_chain():
    env, wrapper = _make(n_action_steps=4, skip=True, wrap=True)
    assert wrapper._render_gate is env, wrapper._render_gate
    wrapper.reset()
    env.render_count = 0
    obs, _, _, _, _ = wrapper.step(_action(4))
    assert env.render_count == N_CAMS, env.render_count
    _assert_real_frames(obs, "through wrapper chain")
    print("  [PASS] render gate resolved through a wrapper chain")


def test_outer_gate_wins_over_base_env():
    """The walk must prefer the outermost implementer, which `.unwrapped` alone
    would skip."""

    class GatingWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.calls = []

        def set_camera_obs_enabled(self, enabled):
            self.calls.append(enabled)
            self.env.set_camera_obs_enabled(enabled)

        def recompute_observation(self):
            return self.env.recompute_observation()

    base = FakeSimEnv()
    gate = GatingWrapper(base)
    wrapper = MultiStepWrapper(
        gate,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=4,
        skip_intermediate_render=True,
    )
    assert wrapper._render_gate is gate, wrapper._render_gate
    wrapper.reset()
    wrapper.step(_action(4))
    assert gate.calls, "the outer gate was bypassed"
    print("  [PASS] outermost gate implementer wins over the base env")


def test_single_action_step_never_disables():
    env, wrapper = _make(n_action_steps=1, skip=True)
    wrapper.reset()
    env.render_count = 0
    obs, _, _, _, _ = wrapper.step(_action(1))
    assert env.render_count == N_CAMS, env.render_count
    _assert_real_frames(obs, "n_action_steps=1")
    print("  [PASS] n_action_steps=1 keeps the plain path")


def test_rejects_multi_frame_video_horizon():
    try:
        _make(n_action_steps=8, skip=True, video_horizon=2)
    except ValueError as e:
        assert "video_horizon == 1" in str(e), e
        print("  [PASS] video_horizon > 1 rejected at construction")
        return
    raise AssertionError("expected ValueError for video_horizon > 1")


def test_rejects_env_without_render_gate():
    class GatelessEnv(gym.Env):
        """An env that genuinely does not implement the gate — not a subclass
        hiding it behind __getattribute__, which would still DECLARE the methods
        on its type (see MultiStepWrapper._declares)."""

        def __init__(self):
            proto = FakeSimEnv()
            self.observation_space = proto.observation_space
            self.action_space = proto.action_space

    try:
        MultiStepWrapper(
            GatelessEnv(),
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=8,
            skip_intermediate_render=True,
        )
    except AttributeError as e:
        assert "set_camera_obs_enabled" in str(e), e
        print("  [PASS] env without the render-gate interface rejected")
        return
    raise AssertionError("expected AttributeError for an env without the gate")


def test_rejects_substep_frame_consumer():
    """VideoRecordingWrapper reads every substep's video keys; skipped substeps
    have none, so the combination must be refused up front."""
    try:
        _make(n_action_steps=8, skip=True, frame_consumer=True)
    except ValueError as e:
        assert "FrameConsumingWrapper" in str(e), e
        print("  [PASS] per-substep frame consumer in the chain rejected")
        return
    raise AssertionError("expected ValueError for a substep frame consumer")


def test_parity_with_baseline():
    """Rewards, dones, success, substep counts AND frame provenance must match
    the unskipped path."""
    for kwargs in (
        {},
        {"terminate_at_substep": 3},
        {"success_at_substep": 4},
        {"max_episode_steps": 6},
    ):
        base = _chunk_report(skip=False, **kwargs)
        skipped = _chunk_report(skip=True, **kwargs)
        strip = lambda rows: [   # noqa: E731
            {k: v for k, v in r.items() if k != "renders"} for r in rows
        ]
        assert strip(base) == strip(skipped), (
            f"parity broken for {kwargs}:\n  base={base}\n  skip={skipped}"
        )
        assert sum(r["renders"] for r in skipped) < sum(
            r["renders"] for r in base
        ), f"no renders saved for {kwargs}"
    print("  [PASS] outcomes and frame provenance identical to baseline")


TESTS = [
    test_embedded_observable_matches_robosuite,
    test_baseline_renders_every_substep,
    test_skip_renders_once_per_chunk,
    test_frame_provenance_and_dtype_match_baseline,
    test_first_chunk_after_reset_is_real,
    test_first_chunk_after_forced_update_is_real,
    test_every_substep_keeps_the_full_key_set,
    test_gym_make_chain_with_passive_env_checker,
    test_gate_lookup_emits_no_warnings,
    test_sync_vector_env_roundtrip,
    test_early_termination_still_returns_real_frames,
    test_truncation_mid_chunk_still_returns_real_frames,
    test_camera_obs_left_enabled_after_step,
    test_resolves_gate_through_wrapper_chain,
    test_outer_gate_wins_over_base_env,
    test_single_action_step_never_disables,
    test_rejects_multi_frame_video_horizon,
    test_rejects_env_without_render_gate,
    test_rejects_substep_frame_consumer,
    test_parity_with_baseline,
]


if __name__ == "__main__":
    which = "REAL robosuite Observable" if REAL_FACTORY else "embedded Observable model"
    print(f"=== skip_intermediate_render tests ({which}) ===\n")
    for test in TESTS:
        print(f"{test.__name__}:")
        test()
    print(f"\nAll {len(TESTS)} tests PASSED.")
