from collections import defaultdict, deque
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def stack_repeated(x, n, loc):
    return np.repeat(np.expand_dims(x, axis=loc), n, axis=loc)


def repeated_box(box_space, n, loc):
    return spaces.Box(
        low=stack_repeated(box_space.low, n, loc),
        high=stack_repeated(box_space.high, n, loc),
        shape=box_space.shape[:loc] + (n,) + box_space.shape[loc:],
        dtype=box_space.dtype,
    )


def repeated_space(space, n, loc=0):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n, loc)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n, loc)
        return result_space
    elif isinstance(space, spaces.Discrete):
        return spaces.MultiDiscrete([[space.n] for _ in range(n)])
    elif isinstance(space, spaces.Text):  # For language, we don't repeat and only keep the last one
        return space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def compress_dict_list(ds, recursive=False):
    """
    Args:
        ds: list of dicts with the same keys and the same value type
        recursive: whether to recursively compress nested dictionaries
    Returns:
        dict of lists with the same keys as the dicts in ds
    """
    if not ds:
        return {}

    # Assert that ds is a list of dictionaries
    if not isinstance(ds, list):
        raise TypeError(f"Expected a list of dictionaries, but got {type(ds)}")

    if not all(isinstance(d, dict) for d in ds):
        non_dict_indices = [i for i, d in enumerate(ds) if not isinstance(d, dict)]
        raise TypeError(
            f"All elements must be dictionaries. Found non-dictionary elements at indices: {non_dict_indices}"
        )

    # Check that all dictionaries have the same keys
    keys = set(ds[0].keys())
    for i, d in enumerate(ds[1:], 1):
        if set(d.keys()) != keys:
            missing_keys = keys - set(d.keys())
            extra_keys = set(d.keys()) - keys
            error_msg = f"Dictionary at index {i} has different keys than the first dictionary."
            if missing_keys:
                error_msg += f" Missing keys: {missing_keys}."
            if extra_keys:
                error_msg += f" Extra keys: {extra_keys}."
            raise ValueError(error_msg)

    result = defaultdict(list)
    for d in ds:
        for key, value in d.items():
            result[key].append(value)

    # Convert lists to numpy arrays or recursively compress nested dictionaries
    for key, value_list in result.items():
        # Check if all values are dictionaries and recursion is enabled
        if recursive and all(isinstance(v, dict) for v in value_list):
            result[key] = compress_dict_list(value_list, recursive=True)
        else:
            try:
                result[key] = np.array(value_list)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert values for key '{key}' to numpy array: {str(e)}"
                )

    return result


def aggregate(data, method="max"):
    if method == "max":
        # equivalent to any
        return np.max(data)
    elif method == "min":
        # equivalent to all
        return np.min(data)
    elif method == "mean":
        return np.mean(data)
    elif method == "sum":
        return np.sum(data)
    else:
        raise NotImplementedError()


class MultiStepWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_delta_indices,
        state_delta_indices,
        n_action_steps,
        max_episode_steps=None,
        reward_agg_method="max",
        terminate_on_success=False,
        skip_intermediate_render=False,
    ):
        """
        video_delta_indices: np.ndarray[int], please check `assert_delta_indices` to see the requirements
        state_delta_indices: np.ndarray[int] | None, please check `assert_delta_indices` to see the requirements
          if None, it means the model is vision-only
        skip_intermediate_render: bool, skip camera rendering on every substep
          except the last one of each action chunk. See `step` for why that is
          observationally equivalent, and the guards below for the two
          requirements it places on the env / delta indices. Off by default
          because consumers that read every substep's frames (e.g. eval video
          recording in rollout_policy.py) need all of them.
        """
        super().__init__(env)
        # Assign action space
        self._action_space = repeated_space(env.action_space, n_action_steps)

        # Assign delta indices and horizons
        self.video_delta_indices = video_delta_indices
        self.video_horizon = len(video_delta_indices)
        self.assert_delta_indices(self.video_delta_indices, self.video_horizon)
        if state_delta_indices is not None:
            self.state_delta_indices = state_delta_indices
            self.state_horizon = len(state_delta_indices)
            self.assert_delta_indices(self.state_delta_indices, self.state_horizon)
        else:
            self.state_horizon = None
            self.state_delta_indices = None

        # Assign observation space
        self._observation_space = self.convert_observation_space(
            self.observation_space,
            self.video_horizon,
            self.state_horizon,
        )

        # Assign other attributes
        self.max_episode_steps = max_episode_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.max_steps_needed = self.get_max_steps_needed()

        self.obs = deque(maxlen=self.max_steps_needed + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_action_steps + 1))
        self.terminate_on_success = terminate_on_success

        # --- skip_intermediate_render preconditions ---------------------------
        # Both are correctness requirements, not conveniences, so fail at
        # construction rather than silently feeding the policy black frames.
        self.skip_intermediate_render = skip_intermediate_render
        self._render_gate = None
        if skip_intermediate_render:
            if self.video_horizon != 1:
                # With video_horizon > 1, _get_obs reads self.obs[-2], [-3], ...
                # Those entries are EARLIER SUBSTEPS of the same chunk, whose
                # frames we would have skipped — the policy would receive
                # placeholders. Only the single-frame case is safe.
                raise ValueError(
                    "skip_intermediate_render requires video_horizon == 1 "
                    f"(video_delta_indices={video_delta_indices}); with a longer "
                    "video horizon, _get_obs reads intermediate substeps whose "
                    "frames would be placeholders."
                )
            if self.state_horizon is not None and self.state_horizon != 1:
                # recompute_observation() force-updates EVERY observable, not
                # just the cameras. Cameras are phase-protected (set_enabled
                # resets their timer first); low-dim observables are not, so an
                # extra sample per chunk walks their phase backwards. obs[-1]
                # stays correct, but a longer state horizon reads obs[-2], which
                # drifts from baseline.
                raise ValueError(
                    "skip_intermediate_render requires state_horizon == 1 "
                    f"(state_delta_indices={state_delta_indices}); a longer "
                    "state horizon reads intermediate substeps whose low-dim "
                    "sampling phase the forced render perturbs."
                )
            self._render_gate = self._find_render_gate(env)
            if self._render_gate is None:
                raise AttributeError(
                    "skip_intermediate_render requires an env in the wrapper "
                    "chain implementing set_camera_obs_enabled(), "
                    "recompute_observation() and prime_camera_obs() (see "
                    "RoboCasaEnv in "
                    "robocasa/utils/gym_utils/gymnasium_basic.py). Innermost "
                    f"env: {type(env.unwrapped).__name__}."
                )
            consumer = self._find_substep_obs_consumer(env)
            if consumer is not None:
                # e.g. VideoRecordingWrapper, which reads video.* out of every
                # substep's observation. Skipped substeps carry no video keys,
                # so it would fail its own "No video frame found" assertion.
                raise ValueError(
                    f"skip_intermediate_render is incompatible with "
                    f"{type(consumer).__name__} in the wrapper chain: it reads "
                    f"every substep's observation, but skipped substeps produce "
                    f"no video keys. Disable one of the two."
                )

    @staticmethod
    def _walk_chain(env):
        """Yield env, env.env, ... guarding against cycles."""
        target = env
        seen = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            yield target
            target = getattr(target, "env", None)

    @staticmethod
    def _declares(target, name):
        """True if `target` ITSELF defines `name` — never via wrapper forwarding.

        `hasattr` is wrong here across gymnasium versions: 0.29.1 (what the
        robocasa collector venv pins) forwards unknown attributes down the
        wrapper chain, so a plain hasattr on the outermost wrapper returns True
        for attributes it does not own — _find_render_gate would then return the
        wrapper instead of the env implementing them. It also emits a
        UserWarning per distinct attribute name. 1.x dropped forwarding
        entirely. Checking __dict__/type() is version-independent and
        silent.
        """
        # getattr(..., "__dict__", {}) not vars(): _walk_chain follows `.env`
        # without a type guard, and vars() raises TypeError on any object
        # without an instance dict.
        return name in getattr(target, "__dict__", {}) or hasattr(
            type(target), name
        )

    @classmethod
    def _find_render_gate(cls, env):
        """Return the first env in the wrapper chain that can gate rendering.

        gym.make() wraps the base env (OrderEnforcing, PassiveEnvChecker, ...),
        so the gate is never the object handed to this wrapper — we walk down to
        whoever actually implements it. Checking each level before descending
        means a wrapper that overrides the pair (e.g. to gate several sub-envs)
        wins over the base env.

        The gate's recompute_observation() must return observations in the same
        format step() does, so an observation-TRANSFORMING wrapper may not sit
        between this wrapper and the gate. The wrappers that appear there today
        (gym.make's OrderEnforcing / PassiveEnvChecker / TimeLimit, and
        VideoRecordingWrapper) all pass observations through unchanged —
        VideoRecordingWrapper *reads* them, which is rejected separately by
        _find_substep_obs_consumer.

        Returns None when nothing in the chain supports it.
        """
        for target in cls._walk_chain(env):
            if all(
                cls._declares(target, name)
                for name in (
                    "set_camera_obs_enabled",
                    "recompute_observation",
                    "prime_camera_obs",
                )
            ):
                return target
        return None

    @classmethod
    def _find_substep_obs_consumer(cls, env):
        """Return the first wrapper in the chain that reads EVERY substep's
        observation (marked with `consumes_every_substep_obs`), or None.

        Such a wrapper is incompatible with skip_intermediate_render, which only
        produces video keys once per chunk.
        """
        for target in cls._walk_chain(env):
            if cls._declares(target, "consumes_every_substep_obs") and getattr(
                target, "consumes_every_substep_obs", False
            ):
                return target
        return None

    def convert_observation_space(self, observation_space, video_horizon, state_horizon):
        """
        For video, the observation space will be (video_horizon,) + original shape
        For state (if not None), the observation space will be (state_horizon,) + original shape
        """
        new_observation_space = {}
        for k in observation_space.keys():
            if k.startswith("video"):
                box = observation_space[k]
                horizon = video_horizon
                new_observation_space[k] = repeated_space(box, horizon)
            elif k.startswith("state"):
                box = observation_space[k]
                if state_horizon is not None:
                    horizon = state_horizon
                else:
                    # Don't include the state in the observation space
                    continue
                new_observation_space[k] = repeated_space(box, horizon)
            elif k.startswith("annotation"):
                text = observation_space[k]
                new_observation_space[k] = text
            else:
                warnings.warn(f"Key without a prefix: {k}")
                box = observation_space[k]
                horizon = state_horizon
                new_observation_space[k] = repeated_space(box, horizon)

        return spaces.Dict(new_observation_space)

    def get_max_steps_needed(self):
        """
        Get the maximum number of steps that we need to cache.
        """
        video_max_steps_needed = (
            np.max(self.video_delta_indices) - np.min(self.video_delta_indices) + 1
        )
        if self.state_delta_indices is not None:
            state_max_steps_needed = (
                np.max(self.state_delta_indices) - np.min(self.state_delta_indices) + 1
            )
        else:
            state_max_steps_needed = 0
        return int(max(video_max_steps_needed, state_max_steps_needed))

    def assert_delta_indices(self, delta_indices: np.ndarray, horizon: int):
        # Check the length
        # (In this wrapper, this seems redundant because we get the horizon from the delta indices. But in the policy, the horizon is not derived from the delta indices but we need to make it consistent. To make the function consistent, we keep the check here.)
        assert len(delta_indices) == horizon, f"{delta_indices=}, {horizon=}"
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent (because in real robot experiments, we actually use the dt to get the observations, which requires the step to be consistent)
            assert np.all(np.diff(delta_indices) == delta_indices[1] - delta_indices[0]), (
                f"{delta_indices=}"
            )
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def reset(self, seed=None, options=None):
        """Resets the environment using kwargs."""
        obs, info = super().reset(seed=seed, options=options)

        self.obs = deque([obs] * (self.max_steps_needed + 1), maxlen=self.max_steps_needed + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_action_steps + 1))

        obs = self._get_obs(self.video_delta_indices, self.state_delta_indices)
        info = {k: [v] for k, v in info.items()}
        if "intermediate_signals" in info:
            # "intermediate_signals" contain the metrics for 5DC tasks to indicate language following
            info["intermediate_signals"] = {}
        return obs, info

    def step(self, action):
        """
        action: dict: key-value pairs where the values are of shape (n_action_steps,) + action_shape
        """
        states = []
        rewards = []
        dones = []
        # Defaults so that if the loop breaks on its FIRST iteration — which
        # happens when step() is called on an already-done env (e.g. a vector
        # env under autoreset_mode=DISABLED steps a done env while sibling envs
        # are still active) — `env_state` and `truncated` remain bound for the
        # post-loop assembly/return below. Without these, that path raises
        # UnboundLocalError (env_state at info["model"]=..., then truncated at
        # the return). In the normal path both are overwritten inside the loop,
        # so this is behavior-preserving; it just makes step() a clean no-op
        # (returns the cached obs with done=True) instead of crashing.
        env_state = {"states": [], "model": []}
        truncated = False
        # skip_intermediate_render: exactly ONE camera frame per chunk is ever
        # read. video_horizon == 1 is enforced at construction, so _get_obs below
        # returns self.obs[-1] alone; the per-substep renders robosuite performs
        # inside env.step() are pure waste (for RoboCasa, 3 x sim.render at full
        # camera resolution plus a flip/copy and a resize, every substep).
        #
        # Strategy: camera observables are disabled for the chunk, then PRIMED
        # back to their natural phase immediately before the last substep, so
        # robosuite takes that sample itself inside env.step(). Observationally
        # exact: the frame is captured at the same physics substep as baseline
        # AND before _post_action runs — Kitchen.update_state() writes fixture
        # visuals there (coffee-liquid / burner-flame / sink-water site_rgba,
        # cabinet interior sites), which a render issued after step() returns
        # would show one control step ahead of baseline.
        #
        # Plain set_enabled(True) on the last substep does NOT work, which is
        # why priming exists: Observable.set_enabled() calls reset(), which
        # zeroes the sampling timer but leaves _sampled set. After any
        # force_update (every env reset ends with one) _sampled is True, so a
        # re-enabled observable cannot sample for a full control step — the first
        # chunk renders nothing and yields reset()'s float64 zeros, and later
        # chunks sample on the FIRST physics substep, one control step stale.
        # Verified by simulating the real robosuite Observable.
        skip_render = self.skip_intermediate_render and self.n_action_steps > 1
        last_step = self.n_action_steps - 1
        # Nothing to gate if the env is already done: step() is a no-op that
        # breaks immediately. Toggling anyway would still hit
        # Observable.set_enabled -> reset(), which zeroes the observable's value
        # to float64 and its timer while leaving _sampled set — pointless churn
        # on a path that renders nothing.
        already_done = bool(self.done) and self.done[-1]
        try:
            if skip_render and not already_done:
                self._render_gate.set_camera_obs_enabled(False)
            for step in range(self.n_action_steps):
                act = {}
                for key, value in action.items():
                    act[key] = value[step, :]
                if len(self.done) > 0 and self.done[-1]:
                    # termination
                    break
                if skip_render and step == last_step:
                    # Hand the camera observables back the sampling phase they
                    # would have had if never disabled, so robosuite takes the
                    # sample itself INSIDE this step() — at the same physics
                    # substep, and crucially BEFORE _post_action mutates fixture
                    # visuals. See RoboCasaEnv.prime_camera_obs.
                    self._render_gate.prime_camera_obs()
                observation, reward, done, truncated, info = super().step(act)
                # TODO: assign meaningful values
                env_state = {"states": [], "model": []}
                states.append(env_state["states"])
                rewards.append(reward)
                dones.append(done)
                # NOTE: self.reward/self.done are updated BEFORE self.obs here
                # (the original order was obs first) so that `done` — including
                # the truncation override below — is known before we decide
                # whether this substep is the one that needs the real frames.
                # Nothing between the two reads self.obs, so the reorder is
                # behavior-preserving.
                self.reward.append(reward)
                if (self.max_episode_steps is not None) and (
                    len(self.reward) >= self.max_episode_steps
                ):
                    # truncation
                    done = True
                # Append to self.done BEFORE the fallible forced render below:
                # if recompute_observation() raises, self.reward and self.done
                # must not end up different lengths (the next step would then
                # mis-bill its truncation budget).
                self.done.append(done)
                if skip_render and done and step != last_step:
                    # The chunk ended early, so no primed sample was taken this
                    # substep. Fall back to a forced render. Its frame reflects
                    # fixture state one _post_action ahead of an in-step sample
                    # (see recompute_observation), which is harmless here: the
                    # collector discards the terminal observation of a chunk
                    # that ended.
                    observation = self._render_gate.recompute_observation()
                self.obs.append(observation)
                self._add_info(info)
        finally:
            if skip_render and not already_done:
                # Leave the env renderable. reset() and the scene save/restore
                # RPCs build observations outside this loop and need real
                # frames; the finally covers the early-break and exception
                # paths too.
                self._render_gate.set_camera_obs_enabled(True)

        observation = self._get_obs(self.video_delta_indices, self.state_delta_indices)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, "max")
        info = dict_take_last_n(self.info, self.n_action_steps)
        states = np.array(states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        info["states"] = states
        info["rewards"] = rewards
        info["model"] = env_state["model"]
        info["actions"] = action
        info["dones"] = dones
        if "intermediate_signals" in info:
            # "intermediate_signals" contain the metrics for 5DC tasks to indicate language following
            # Here we turn a list of dicts into a dict of lists
            """
            Example of the ultimate format of `info["intermediate_signals"]`:
            {
                'grasp_obj': [True, ..., True],
                'grasp_distractor_obj': [False, ..., False],
                'gripper_obj_dist': [0.0004638251298563212, ..., 0.0004638251298563212],
                'gripper_distractor_dist': [0.0023107511879928433, ..., 0.0023107511879928433]
            }
            The length is `n_action_steps`.
            """
            info["intermediate_signals"] = compress_dict_list(list(info["intermediate_signals"]))

        if self.terminate_on_success and any(info["success"]):
            # Terminate after this step.
            done = True

        return observation, reward, done, truncated, info

    def _get_obs(self, video_delta_indices, state_delta_indices):
        """
        Output:
        For video: (video_horizon,) + obs_shape
        For state (if not None): (state_horizon,) + obs_shape
        """
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                if key.startswith("video"):
                    """
                    NOTE:
                      We need to subtract 1 because video_delta_indices is 0-indexed.
                      E.g., video_delta_indices = np.array([-4, -3, -2, -1, 0])
                      Then when we select the observation,
                        it should be [obs[-5], obs[-4], obs[-3], obs[-2], obs[-1]]
                      (i.e., the latest observation is at the last index)
                    """
                    delta_indices = video_delta_indices - 1
                    this_obs = [self.obs[i][key] for i in delta_indices]
                    result[key] = np.stack(this_obs, axis=0)
                elif key.startswith("state"):
                    if state_delta_indices is not None:
                        delta_indices = state_delta_indices - 1
                    else:
                        raise ValueError(
                            f"state_delta_indices is None but `state` is still in the {self.observation_space=}"
                        )
                    this_obs = [self.obs[i][key] for i in delta_indices]
                    result[key] = np.stack(this_obs, axis=0)
                elif key.startswith("annotation"):
                    result[key] = self.obs[-1][key]
                else:
                    if state_delta_indices is not None:
                        delta_indices = state_delta_indices - 1
                    else:
                        raise ValueError(
                            f"state_delta_indices is None but `state` is still in the {self.observation_space=}"
                        )
                    this_obs = [self.obs[i][key] for i in delta_indices]
                    result[key] = np.stack(this_obs, axis=0)
            return result
        else:
            raise RuntimeError(f"Unsupported space type: {type(self.observation_space)=}")

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def get_rewards(self):
        return self.reward

    def get_attr(self, name):
        return getattr(self, name)

    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
