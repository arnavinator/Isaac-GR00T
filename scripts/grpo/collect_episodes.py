"""Episode collector for GRPO training.

This script runs in the ROBOCASA VENV (separate from the main .venv) and:
1. Connects to the GR00T model server via ZMQ (PolicyClient)
2. Collects episodes in groups (same seed within a group, different across groups)
3. Records observations, actions, initial noise tensors, and raw model outputs
4. Saves episodes as .npz files for the training loop to consume

Architecture context:
- The MODEL runs on GPU in Terminal 1 (main .venv, grpo_server.py)
- THIS SCRIPT runs on CPU in Terminal 2 (robocasa venv, collect_episodes.py)
- Communication via ZMQ (same machine, ~0.1ms latency per call)

Group structure (mirrors grpo_cont.py):
- Each group = group_size rollouts from the SAME initial state (same env seed)
- Different outcomes arise from policy noise (denoising randomness), NOT env randomness
- GRPO advantages compare rollouts WITHIN a group

Usage:
    # From robocasa venv (with server already running):
    python scripts/grpo/collect_episodes.py \\
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --group-size 5 --num-groups 12 \\
        --output-dir /tmp/grpo_episodes/iter_001 \\
        --server-host 127.0.0.1 \\
        --server-port 5555
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

# These imports are available in the robocasa venv
from gr00t.policy.server_client import PolicyClient

# Dense reward extraction (also in robocasa venv)
from dense_reward import classify_task_type, compute_dense_progress, compute_shaped_reward


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect episodes in groups for GRPO training via PolicyClient + SyncVectorEnv"
    )
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="Full env name (e.g., robocasa_panda_omron/OpenDrawer_PandaOmron_Env)"
    )
    parser.add_argument(
        "--group-size", type=int, default=5,
        help="Number of rollouts per group (G). All share the same env seed. Also the number of parallel envs."
    )
    parser.add_argument(
        "--num-groups", type=int, default=12,
        help="Number of groups (different initial states) per iteration."
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=720,
        help="Maximum steps per episode before truncation"
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=8,
        help="Number of steps to execute from each action chunk"
    )
    parser.add_argument(
        "--server-host", type=str, default="127.0.0.1",
        help="GR00T model server hostname"
    )
    parser.add_argument(
        "--server-port", type=int, default=5555,
        help="GR00T model server port"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save episode .npz files"
    )
    parser.add_argument(
        "--success-weight", type=float, default=1.0,
        help="Weight for binary success in shaped reward (1.0 = pure binary + time-scaled)"
    )
    parser.add_argument(
        "--fast-forward-steps", type=int, default=0,
        help="Outer steps to fast-forward before branching (0=disabled)"
    )
    parser.add_argument(
        "--fast-forward-pct", type=float, default=0.5,
        help="Fraction of groups that use fast-forward (0.0-1.0)"
    )
    parser.add_argument(
        "--debug-fast-forward", action="store_true",
        help="Save verification images after each branch point to --output-dir/debug_ff/"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for environment initialization"
    )
    return parser.parse_args()


def make_env(
    env_name: str,
    max_episode_steps: int,
    n_action_steps: int,
    video_delta_indices: np.ndarray | None = None,
    state_delta_indices: np.ndarray | None = None,
):
    """Create a gymnasium environment factory for SyncVectorEnv.

    This mirrors grpo_cont.py's make_env() (lines 178-198) but for RoboCasa.
    The env is wrapped with MultiStepWrapper for action chunking.

    Args:
        env_name: Full environment name for gym.make().
        max_episode_steps: Episode truncation length.
        n_action_steps: Steps to execute per action chunk.
        video_delta_indices: Temporal stacking indices for video observations.
            Defaults to np.array([0]) (single current frame, matches PandaOmron).
        state_delta_indices: Temporal stacking indices for state observations.
            Defaults to np.array([0]).

    Returns:
        Callable that creates the environment.
    """
    if video_delta_indices is None:
        video_delta_indices = np.array([0])
    if state_delta_indices is None:
        state_delta_indices = np.array([0])

    def _make():
        # Trigger robocasa's gymnasium env registration. The for-loop at the
        # bottom of robocasa/utils/gym_utils/gymnasium_groot.py calls
        # gym.register() for every (env, robot) pair (e.g.,
        # robocasa_panda_omron/OpenDrawer_PandaOmron_Env). Without these
        # imports, gym.make() raises NamespaceNotFound. Imports are inside
        # the factory to mirror the working pattern in
        # gr00t/eval/rollout_policy.py:82-90 (also works under AsyncVectorEnv,
        # where each subprocess has its own import state).
        import os
        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
        import robosuite  # noqa: F401

        # Headless MuJoCo rendering on Linux GPU hosts. egl is the standard
        # backend when there's no X display; matches rollout_policy.py.
        os.environ.setdefault("MUJOCO_GL", "egl")

        env = gym.make(env_name)

        # Apply MultiStepWrapper for action chunking
        # This executes n_action_steps from each predicted chunk
        from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
        env = MultiStepWrapper(
            env,
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        )
        return env

    return _make


class EpisodeCollector:
    """Collects episodes using PolicyClient + parallel envs.

    Each episode records:
    - Per-chunk: observation frames, states, actions, initial noise, raw model output
    - Episode-level: success, max progress, total steps

    The initial noise tensor is captured from the GRPO server (grpo_server.py) via
    a hook on torch.randn. It's the ε₀ that was denoised into the action — used during
    training to evaluate the FM log-prob along the actual denoising path.
    """

    def __init__(
        self,
        env_name: str,
        group_size: int,
        max_episode_steps: int,
        n_action_steps: int,
        server_host: str,
        server_port: int,
        seed: int = 42,
        debug_fast_forward: bool = False,
        output_dir: str = "/tmp/grpo_episodes",
    ):
        self.env_name = env_name
        self.group_size = group_size
        self.n_action_steps = n_action_steps
        self.task_type = classify_task_type(env_name)
        self.seed = seed
        self.debug_fast_forward = debug_fast_forward
        self.output_dir = Path(output_dir)

        # Create group_size parallel environments (one per rollout in a group)
        # All will be reset with the same seed per group
        env_fns = [
            make_env(env_name, max_episode_steps, n_action_steps)
            for _ in range(group_size)
        ]
        self.envs = gym.vector.SyncVectorEnv(env_fns)

        # Connect to model server
        self.policy_client = PolicyClient(host=server_host, port=server_port)

        print(f"Collector initialized:")
        print(f"  Env: {env_name} (task_type: {self.task_type})")
        print(f"  Group size (parallel envs): {group_size}")
        print(f"  Server: {server_host}:{server_port}")

    def collect(
        self,
        group_size: int,
        num_groups: int,
        base_seed: int,
        success_weight: float = 1.0,
        fast_forward_steps: int = 0,
        fast_forward_pct: float = 0.5,
    ) -> list[dict]:
        """Collect episodes organized in groups for GRPO.

        Each group consists of `group_size` rollouts from the SAME initial state
        (same env reset seed). Different groups have different seeds.
        This mirrors grpo_cont.py's structure where each group shares a seed:
            envs_list.append(make_env(..., seed=args.seed + ng, ...))

        Within a group, different outcomes arise from the policy's denoising noise
        (torch.randn in the action head), NOT from environmental randomness.
        GRPO advantages compare these outcomes against each other.

        When fast-forward is enabled for a group, one env runs solo for
        fast_forward_steps outer steps, then all envs branch from that
        intermediate state. This focuses gradient signal on the critical
        manipulation phase rather than the approach trajectory.

        Args:
            group_size: Number of rollouts per group (G = "answers per question").
            num_groups: Number of groups ("questions") to collect.
            base_seed: Starting seed for group generation.
            success_weight: Weight for binary success in shaped reward.
            fast_forward_steps: Outer steps to skip before branching (0=disabled).
            fast_forward_pct: Fraction of groups that use fast-forward.

        Returns:
            List of episode dicts ready for saving as .npz.
        """
        all_episodes = []
        start_time = time.time()
        total_episodes = group_size * num_groups
        rng = np.random.default_rng(base_seed)

        ff_enabled = fast_forward_steps > 0 and fast_forward_pct > 0
        print(f"\nCollecting {num_groups} groups × {group_size} rollouts = {total_episodes} episodes...")
        if ff_enabled:
            print(f"  Fast-forward: {fast_forward_steps} steps, {fast_forward_pct:.0%} of groups")

        for group_idx in range(num_groups):
            # All rollouts in this group start from the SAME initial state
            group_seed = base_seed + group_idx

            # Decide whether this group uses fast-forward
            use_ff = ff_enabled and rng.random() < fast_forward_pct

            if use_ff:
                group_episodes = self._collect_one_group_with_fast_forward(
                    group_seed=group_seed,
                    group_size=group_size,
                    group_id=group_idx,
                    fast_forward_steps=fast_forward_steps,
                    success_weight=success_weight,
                )
                ff_label = f"(branched at step {fast_forward_steps})"
            else:
                group_episodes = self._collect_one_group(
                    group_seed=group_seed,
                    group_size=group_size,
                    group_id=group_idx,
                    success_weight=success_weight,
                )
                ff_label = "(from seed)"

            all_episodes.extend(group_episodes)

            # Progress reporting
            n_done = len(all_episodes)
            elapsed = time.time() - start_time
            rate = n_done / elapsed * 60 if elapsed > 0 else 0
            group_successes = sum(e["success"] for e in group_episodes)
            print(
                f"  Group {group_idx+1}/{num_groups} (seed={group_seed}) {ff_label}: "
                f"{group_successes}/{group_size} success | "
                f"total: {n_done}/{total_episodes} ({rate:.0f} eps/min)"
            )

        elapsed = time.time() - start_time
        successes = sum(e["success"] for e in all_episodes)
        print(
            f"\nCollection complete: {total_episodes} episodes in {elapsed:.1f}s "
            f"({total_episodes/elapsed*60:.0f} eps/min)"
        )
        print(f"Success rate: {successes}/{total_episodes} ({100*successes/total_episodes:.0f}%)")
        # Only report dense progress when it actually contributed to the shaped
        # reward — with success_weight=1.0 (the default) the progress term is
        # multiplied by (1 - success_weight) = 0, so max_progress stays at the
        # per-episode init value of 0.0 and reporting it would be misleading.
        if success_weight < 1.0:
            print(f"Mean progress: {np.mean([e['max_progress'] for e in all_episodes]):.3f}")

        return all_episodes

    def _collect_one_group(
        self,
        group_seed: int,
        group_size: int,
        group_id: int,
        success_weight: float,
    ) -> list[dict]:
        """Collect one group: group_size rollouts from the same initial state.

        All environments are reset with the SAME seed, producing identical
        initial configurations. The policy's denoising noise causes different
        trajectories and outcomes.

        Per-env stepping (NOT SyncVectorEnv batched step) is used for two
        correctness reasons:

          1. SyncVectorEnv autoreset: when one env in the group terminates
             before another, gymnasium's NEXT_STEP / SAME_STEP autoreset
             modes produce mismatched-shape per-env info dicts (multi-substep
             arrays from active envs vs. length-1 from just-reset envs),
             which silently corrupts subsequent batching of info["success"].
          2. Skipping done envs: with per-env stepping we can simply omit
             done envs from each iteration's batch — no autoreset, no stale
             obs from a freshly-reset terminal env getting fed back to the
             policy.

        We still build a single batched observation for the server call
        (containing only the active envs) so denoising amortizes across
        the parallel rollouts.
        """
        # Reset each env with seed=group_seed AND align all G envs to env
        # 0's scene. RoboCasa picks per-instance kitchen layout, camera
        # noise, and textures at env construction time using each env's
        # own RNG; those choices live in the model XML, NOT in MjSimState,
        # so a plain seeded reset doesn't make the parallel envs match
        # visually. See _align_envs_to_group_scene for the recipe.
        observations_per_env = self._align_envs_to_group_scene(group_seed)

        episodes_in_progress = [
            self._new_episode(group_id, group_seed) for _ in range(group_size)
        ]
        done_flags = [False] * group_size

        while not all(done_flags):
            # Build a batched observation for the server from ACTIVE envs only.
            # `j` indexes into the batch (0..len(active_indices)-1); `env_idx`
            # is the original env's slot in [0, group_size).
            active_indices = [i for i, d in enumerate(done_flags) if not d]
            active_obs = [observations_per_env[i] for i in active_indices]
            batched_obs = self._batch_per_env_obs(active_obs)

            actions, initial_noise, raw_actions, action_masks_batch = (
                self._get_actions_from_server(batched_obs)
            )

            # Process each active env: record its observation/action, step it
            # individually, and harvest results.
            for j, env_idx in enumerate(active_indices):
                ep = episodes_in_progress[env_idx]
                obs = observations_per_env[env_idx]
                action_j = self._extract_per_env(actions, j)

                # --- Record observation, action chunk, and capture data ---
                ep["video_frames"].append(self._extract_video_single(obs))
                ep["states"].append(self._extract_state_single(obs))
                ep["actions"].append(action_j)
                if (raw_actions is not None and hasattr(raw_actions, '__getitem__')
                        and len(raw_actions) > j):
                    ep["raw_actions"].append(raw_actions[j])
                else:
                    ep["raw_actions"].append(None)
                if (action_masks_batch is not None and hasattr(action_masks_batch, '__getitem__')
                        and len(action_masks_batch) > j):
                    ep["action_masks"].append(action_masks_batch[j])
                else:
                    ep["action_masks"].append(None)
                if (initial_noise is not None and hasattr(initial_noise, '__getitem__')
                        and len(initial_noise) > j):
                    ep["initial_noises"].append(initial_noise[j])
                else:
                    ep["initial_noises"].append(None)
                if ep["language"] is None:
                    ep["language"] = self._extract_language_single(obs)

                # --- Step this env individually ---
                env = self.envs.envs[env_idx]
                next_obs, reward, terminated, truncated, info = env.step(action_j)
                observations_per_env[env_idx] = next_obs

                # H3 fix: count ACTUAL substeps run, not n_action_steps.
                # MultiStepWrapper.step early-breaks on termination, so a
                # chunk may have run fewer substeps than n_action_steps.
                # info["dones"] is the per-substep done array (length = actual
                # substeps); using its length avoids over-counting fast
                # successes, which previously biased time-scaled rewards low.
                if "dones" in info and hasattr(info["dones"], "__len__"):
                    substeps_run = len(info["dones"])
                else:
                    substeps_run = self.n_action_steps  # fallback
                ep["num_steps"] += substeps_run

                # Dense progress (only when it actually contributes to reward)
                if success_weight < 1.0:
                    progress = compute_dense_progress(env, self.task_type)
                    ep["max_progress"] = max(ep["max_progress"], progress)

                # Termination handling
                if terminated or truncated:
                    done_flags[env_idx] = True
                    success = self._reduce_success(info)
                    ep["success"] = success
                    ep["shaped_reward"] = compute_shaped_reward(
                        success, ep["max_progress"], success_weight
                    )

        return episodes_in_progress

    def _collect_one_group_with_fast_forward(
        self,
        group_seed: int,
        group_size: int,
        group_id: int,
        fast_forward_steps: int,
        success_weight: float,
    ) -> list[dict]:
        """Collect one group with lockstep fast-forward.

        Focuses GRPO signal on the critical manipulation phase by skipping
        the early approach trajectory.

        Phases:
          1. Reset all envs with the same seed → identical initial states.
          2. Lockstep fast-forward: query the server once per outer step
             (using env 0's obs — all G envs are in the same state) and
             apply the resulting chunk to every env. MuJoCo determinism
             keeps them bit-identical throughout the FF prefix.
          3. Continue all envs independently until done (per-env stepping).
             Within-group diversity then arises from the policy's denoising
             noise on each post-branch action chunk.

        If any env terminates during fast-forward (task solved early or
        error), falls back to normal collection from the seed.

        Determinism: this relies on RoboCasa's MuJoCo step being
        deterministic given identical (state, action). If per-step
        randomization is ever introduced upstream, the lockstep envs
        could diverge silently — use --debug-fast-forward to numerically
        verify sim states stay bit-identical across the group.
        """
        # Phase 1: Reset all envs and align them to env 0's scene. The
        # alignment is essential because lockstep stepping only produces
        # identical observations if the underlying scenes match (sim_state
        # alone isn't enough — cameras, textures, and kitchen geometry
        # are construction-time choices). See _align_envs_to_group_scene.
        observations_per_env = self._align_envs_to_group_scene(group_seed)

        # Phase 2: Lockstep fast-forward.
        #
        # We send env 0's observation to the server (any env's would be
        # identical) and apply the returned chunk to every env. We use
        # per-env stepping rather than SyncVectorEnv's batched .step() to
        # stay consistent with _collect_one_group and to avoid the
        # autoreset-cascade hazard if any env terminates.
        for step in range(fast_forward_steps):
            batched_obs = self._batch_per_env_obs([observations_per_env[0]])
            actions, _, _, _ = self._get_actions_from_server(batched_obs)
            action = self._extract_per_env(actions, 0)

            any_done = False
            for i in range(group_size):
                next_obs, _, terminated, truncated, _ = self.envs.envs[i].step(action)
                observations_per_env[i] = next_obs
                if terminated or truncated:
                    any_done = True

            if any_done:
                print(
                    f"    Env terminated at step {step} during fast-forward, "
                    "falling back to normal collection"
                )
                return self._collect_one_group(
                    group_seed, group_size, group_id, success_weight
                )

        # Debug: verify all envs really are at the same state after the
        # lockstep prefix. With deterministic MuJoCo this is a regression
        # check rather than a correctness concern.
        if self.debug_fast_forward:
            self._verify_branch_point(
                group_id, group_seed, fast_forward_steps, observations_per_env
            )

        # Phase 3: Continue all envs independently — same per-env loop as
        # _collect_one_group. We deliberately do NOT count the FF prefix
        # in episodes_in_progress[i].num_steps: time-scaled rewards within
        # a group should compare post-branch effort fairly across the
        # rollouts in that group, so the FF prefix is excluded for all of
        # them. Mixing FF and non-FF groups via fast_forward_pct < 1.0
        # means cross-group comparisons (e.g., logged mean_reward) will
        # favor branched groups numerically — this is a known artifact of
        # FF; group-relative advantages are unaffected since they
        # normalize WITHIN each group.
        episodes_in_progress = [
            self._new_episode(group_id, group_seed) for _ in range(group_size)
        ]
        done_flags = [False] * group_size

        while not all(done_flags):
            active_indices = [i for i, d in enumerate(done_flags) if not d]
            active_obs = [observations_per_env[i] for i in active_indices]
            batched_obs = self._batch_per_env_obs(active_obs)

            actions, initial_noise, raw_actions, action_masks_batch = (
                self._get_actions_from_server(batched_obs)
            )

            for j, env_idx in enumerate(active_indices):
                ep = episodes_in_progress[env_idx]
                obs = observations_per_env[env_idx]
                action_j = self._extract_per_env(actions, j)

                ep["video_frames"].append(self._extract_video_single(obs))
                ep["states"].append(self._extract_state_single(obs))
                ep["actions"].append(action_j)
                if (raw_actions is not None and hasattr(raw_actions, '__getitem__')
                        and len(raw_actions) > j):
                    ep["raw_actions"].append(raw_actions[j])
                else:
                    ep["raw_actions"].append(None)
                if (action_masks_batch is not None and hasattr(action_masks_batch, '__getitem__')
                        and len(action_masks_batch) > j):
                    ep["action_masks"].append(action_masks_batch[j])
                else:
                    ep["action_masks"].append(None)
                if (initial_noise is not None and hasattr(initial_noise, '__getitem__')
                        and len(initial_noise) > j):
                    ep["initial_noises"].append(initial_noise[j])
                else:
                    ep["initial_noises"].append(None)
                if ep["language"] is None:
                    ep["language"] = self._extract_language_single(obs)

                env = self.envs.envs[env_idx]
                next_obs, reward, terminated, truncated, info = env.step(action_j)
                observations_per_env[env_idx] = next_obs

                if "dones" in info and hasattr(info["dones"], "__len__"):
                    substeps_run = len(info["dones"])
                else:
                    substeps_run = self.n_action_steps
                ep["num_steps"] += substeps_run

                if success_weight < 1.0:
                    progress = compute_dense_progress(env, self.task_type)
                    ep["max_progress"] = max(ep["max_progress"], progress)

                if terminated or truncated:
                    done_flags[env_idx] = True
                    success = self._reduce_success(info)
                    ep["success"] = success
                    ep["shaped_reward"] = compute_shaped_reward(
                        success, ep["max_progress"], success_weight
                    )

        return episodes_in_progress

    # ─── Group scene alignment ───────────────────────────────────────────

    def _align_envs_to_group_scene(self, group_seed: int) -> list[dict]:
        """Reset all G envs to the SAME scene + dynamic state.

        RoboCasa picks layout, style, camera randomization, and textures
        at env construction time using a per-instance RNG. Those choices
        live in the model XML, NOT in MjSimState, so seeding reset()
        doesn't make parallel envs match visually. To get true within-
        group identity (so the only diversity in a group comes from
        policy denoising noise), we reset env 0 normally and force envs
        1..G-1 to adopt env 0's scene + state.

        The recipe mirrors robocasa's playback_dataset.py:227-258 and
        scripts/denoising_lab/eval/branching_rollout.py:399-427:

          1. wrapper.reset(seed=group_seed) on env 0 to establish a
             canonical scene.
          2. Capture env 0's ep_meta + model_xml + flattened sim state
             via the public robosuite API.
          3. For each i in 1..G-1, on env i:
             a. Pin env 0's ep_meta on the underlying robosuite env.
             b. wrapper.reset(seed=group_seed) — runs the underlying
                reset with the pinned ep_meta and re-initializes the
                wrapper's deques.
             c. edit_model_xml(env0_xml) + reset_from_xml_string +
                sim.reset() — rebuilds with env 0's exact cameras /
                textures / geometry (the construction-time randomness
                that wasn't in MjSimState).
             d. set_state_from_flattened(env0_sim_state) + sim.forward()
                + update_state() — applies env 0's exact qpos/qvel/act.
             e. Refresh wrapper.obs deque from a fresh underlying obs
                read, since wrapper.reset's deque was populated before
                the rebuild and is now stale.

        Returns a list of wrapper-stacked observations, one per env. When
        this function returns, the G envs are bit-identical (verifiable
        via --debug-fast-forward; both sim_state and obs diffs should be
        ~0).
        """
        from collections import deque as _deque

        # Phase 1: env 0 is the reference. Reset it normally; whatever
        # scene RoboCasa picked at construction time is now the target.
        obs_0, _ = self.envs.envs[0].reset(seed=group_seed)

        if self.group_size == 1:
            return [obs_0]

        # Phase 2: capture env 0's complete scene description.
        base_env_0 = self.envs.envs[0].unwrapped
        robosuite_env_0 = base_env_0.env
        env0_ep_meta = robosuite_env_0.get_ep_meta()
        env0_model_xml = robosuite_env_0.sim.model.get_xml()
        env0_sim_state = np.array(robosuite_env_0.sim.get_state().flatten())

        observations_per_env: list[dict] = [obs_0]

        # Phase 3: realign envs 1..G-1 to env 0's scene + state.
        for i in range(1, self.group_size):
            wrapper = self.envs.envs[i]
            base_env = wrapper.unwrapped
            robosuite_env = base_env.env

            # 3a. Pin env 0's metadata on the underlying robosuite env.
            # Older robocasa exposes set_attrs_from_ep_meta; newer
            # versions use set_ep_meta. Both apply layout_id, style_id,
            # and any other per-episode random choices.
            if hasattr(robosuite_env, "set_attrs_from_ep_meta"):
                robosuite_env.set_attrs_from_ep_meta(env0_ep_meta)
            elif hasattr(robosuite_env, "set_ep_meta"):
                robosuite_env.set_ep_meta(env0_ep_meta)

            # 3b. Hard reset with the pinned ep_meta. wrapper.reset() also
            # re-initializes the wrapper's obs deque, reward list, done
            # list, and info defaultdict. Per playback_dataset.py:239-241
            # this hard reset is required because reset_from_xml_string
            # below is only a "soft" reset that doesn't reload the model.
            wrapper.reset(seed=group_seed)

            # 3c. Rebuild the model from env 0's exact XML so that
            # cameras, textures, and kitchen geometry match bit-for-bit.
            xml = robosuite_env.edit_model_xml(env0_model_xml)
            robosuite_env.reset_from_xml_string(xml)
            robosuite_env.sim.reset()

            # 3d. Apply env 0's exact qpos/qvel/act/time. After the
            # rebuild the sim is at the model's home pose, NOT env 0's
            # seed-randomized post-reset pose, so this restore is what
            # makes the dynamic state match. update_state() (or
            # update_sites() on older versions) re-syncs robosuite's
            # internal observation/controller state to the new sim state.
            robosuite_env.sim.set_state_from_flattened(env0_sim_state)
            robosuite_env.sim.forward()
            if hasattr(robosuite_env, "update_state"):
                robosuite_env.update_state()
            elif hasattr(robosuite_env, "update_sites"):
                robosuite_env.update_sites()

            # 3e. Refresh the wrapper's obs deque. wrapper.reset above
            # populated it from the pre-rebuild state, so those frames
            # are stale. Re-read through the underlying env using the
            # same pipeline that GrootRoboCasaEnv.reset uses internally
            # (raw_obs → get_basic_observation → get_groot_observation).
            raw_obs = robosuite_env._get_observations()
            basic_obs = base_env.get_basic_observation(raw_obs)
            groot_obs = base_env.get_groot_observation(basic_obs)
            wrapper.obs = _deque(
                [groot_obs] * (wrapper.max_steps_needed + 1),
                maxlen=wrapper.max_steps_needed + 1,
            )
            # reward/done/info are still clean from wrapper.reset above —
            # nothing has been step()'d since.

            observations_per_env.append(
                wrapper._get_obs(
                    wrapper.video_delta_indices, wrapper.state_delta_indices
                )
            )

        return observations_per_env

    # ─── Debug helpers ───────────────────────────────────────────────────

    def _get_sim_state(self, env_idx: int) -> np.ndarray | None:
        """Read a sub-env's MuJoCo sim state as a flat array.

        Used by _verify_branch_point to numerically confirm that the G envs
        in a group are bit-identical after the lockstep FF prefix.
        """
        try:
            wrapper = self.envs.envs[env_idx]
            robosuite_env = wrapper.unwrapped.env
            return np.array(robosuite_env.sim.get_state().flatten())
        except (AttributeError, TypeError, IndexError):
            return None

    def _verify_branch_point(
        self,
        group_id: int,
        group_seed: int,
        ff_steps: int,
        observations_per_env: list[dict],
    ) -> None:
        """Debug verification: confirm all envs are at an identical state
        after the lockstep fast-forward prefix.

        With deterministic MuJoCo, the G envs see the same (state, action)
        pairs throughout the FF loop, so they should be bit-identical here.
        This routine is a regression check that fails loudly if upstream
        introduces step-time non-determinism (e.g., domain randomization)
        which would silently break the lockstep assumption.

        Saves a montage image showing camera views from all envs side-by-
        side, plus a numerical comparison of sim states and the wrapper-
        level observations the policy actually sees.

        Output saved to: {output_dir}/debug_ff/group{group_id}_seed{group_seed}.png

        Usage:
            python collect_episodes.py ... --debug-fast-forward --fast-forward-steps 10

        What to look for:
        - All camera views in the montage should be pixel-identical
        - sim_state_max_diff should be 0.0 (or <1e-10 for float precision)
        - obs_max_diff should be 0.0 for state keys, ~0 for video keys
        """
        debug_dir = self.output_dir / "debug_ff"
        debug_dir.mkdir(parents=True, exist_ok=True)

        print(f"    [DEBUG] Verifying branch point for group {group_id} (seed={group_seed})...")

        # 1. Compare sim states numerically
        sim_states = []
        for i in range(self.group_size):
            state = self._get_sim_state(i)
            if state is not None:
                sim_states.append(state)

        if len(sim_states) == self.group_size:
            ref = sim_states[0]
            max_diffs = [np.abs(s - ref).max() for s in sim_states[1:]]
            print(f"    [DEBUG] sim_state max diffs vs env 0: {max_diffs}")
            if all(d < 1e-10 for d in max_diffs):
                print(f"    [DEBUG] PASS: all sim states identical")
            else:
                print(f"    [DEBUG] FAIL: sim states differ!")

        # 2. Compare wrapper observations numerically. These are what the
        # policy actually sees, so this is the more faithful obs check than
        # poking the underlying robosuite env directly.
        ref_obs = observations_per_env[0]
        for i in range(1, self.group_size):
            for key in ref_obs:
                if isinstance(ref_obs[key], np.ndarray):
                    diff = np.abs(
                        ref_obs[key].astype(float)
                        - observations_per_env[i][key].astype(float)
                    ).max()
                    if diff > 1e-5:
                        print(f"    [DEBUG] obs key '{key}' differs: env 0 vs env {i}, max_diff={diff:.6f}")

        # 3. Render camera views from all envs and save as montage
        try:
            all_frames = []
            for i in range(self.group_size):
                obs = observations_per_env[i]
                env_frames = []
                for key in sorted(obs.keys()):
                    if not (key.startswith("video.") and "res256" in key):
                        continue
                    img = obs[key]
                    while img.ndim > 3:
                        img = img[0]
                    env_frames.append(img)

                if env_frames:
                    # Concatenate cameras horizontally for this env
                    row = np.concatenate(env_frames, axis=1)
                    all_frames.append(row)

            if all_frames:
                # Stack envs vertically: each row = one env's camera views.
                # All rows should be pixel-identical when lockstep FF is
                # working correctly.
                montage = np.concatenate(all_frames, axis=0)
                out_path = debug_dir / f"group{group_id:03d}_seed{group_seed}_ff{ff_steps}.png"

                try:
                    import cv2
                    cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
                    print(f"    [DEBUG] Montage saved: {out_path}")
                    print(f"    [DEBUG] Layout: {self.group_size} rows (envs) × {len(env_frames)} cols (cameras)")
                    print(f"    [DEBUG] All rows should look identical if lockstep is correct")
                except ImportError:
                    # Fall back to saving as .npy if cv2 not available
                    npy_path = out_path.with_suffix(".npy")
                    np.save(str(npy_path), montage)
                    print(f"    [DEBUG] Montage saved as numpy: {npy_path} (install cv2 for PNG)")
        except Exception as e:
            print(f"    [DEBUG] Could not render montage: {e}")

    def _new_episode(self, group_id: int = 0, env_seed: int = 0) -> dict:
        """Initialize a new episode tracking dict.

        We deliberately do NOT store a within-group env index: it collides
        across groups (each group has env indices 0..G-1) and adds no
        information beyond group_id + env_seed, which together uniquely
        identify the rollout.
        """
        return {
            "video_frames": [],
            "states": [],
            "actions": [],
            "raw_actions": [],       # Raw normalized actions (50x128) for FM log-prob
            "action_masks": [],      # Proper action masks (50x128) from model config
            "initial_noises": [],    # Initial noise tensors (50x128) used to denoise each action
            "language": None,        # Task instruction (extracted from first observation)
            "success": False,
            "max_progress": 0.0,
            "shaped_reward": 0.0,
            "env_name": self.env_name,
            "num_steps": 0,
            "group_id": group_id,
            "env_seed": env_seed,
        }

    def _get_actions_from_server(self, observations) -> tuple:
        """Query the GRPO server for actions, raw actions, initial noise, and action mask.

        The GRPO server captures the initial noise tensor, raw normalized action,
        and proper action mask from the denoising process and returns them in the info dict.

        Args:
            observations: Batched observations from vectorized env.

        Returns:
            Tuple of (actions, initial_noise, raw_actions, action_masks):
                - actions: dict or array for env.step()
                - initial_noise: (group_size, 50, 128) initial denoising noise or None
                - raw_actions: (group_size, 50, 128) normalized model output or None
                - action_masks: (group_size, 50, 128) valid dimension mask or None
        """
        result = self.policy_client.get_action(observations)

        if isinstance(result, tuple) and len(result) == 2:
            action_dict, info = result
        else:
            action_dict = result
            info = {}

        actions = action_dict
        initial_noise = info.get("initial_noise", None)
        raw_actions = info.get("raw_actions", None)
        action_masks = info.get("action_mask", None)

        return actions, initial_noise, raw_actions, action_masks

    def _extract_per_env(self, data, env_idx: int):
        """Extract per-environment data from batched structure.

        Handles both dict-of-arrays (e.g., {"action.eef_pos": (n_envs, 16, 3)})
        and plain arrays (e.g., (n_envs, action_dim)).
        """
        if isinstance(data, dict):
            return {k: np.array(v[env_idx]) if hasattr(v, '__getitem__') else v
                    for k, v in data.items()}
        elif hasattr(data, '__getitem__'):
            return np.array(data[env_idx])
        return data

    def _batch_per_env_obs(self, obs_list: list[dict]) -> dict:
        """Stack a list of single-env observation dicts into a batched dict.

        Used to construct the input to the policy server when we're stepping
        each env individually (no SyncVectorEnv batched step). The output
        format matches what SyncVectorEnv would produce: ndarrays gain a
        leading batch axis, language strings become a tuple per env.
        """
        if not obs_list:
            return {}
        batched = {}
        for key in obs_list[0]:
            vals = [obs[key] for obs in obs_list]
            if isinstance(vals[0], np.ndarray):
                batched[key] = np.stack(vals, axis=0)
            elif isinstance(vals[0], str):
                batched[key] = tuple(vals)
            else:
                batched[key] = vals
        return batched

    def _extract_video_single(self, obs: dict) -> dict[str, np.ndarray]:
        """Extract video frames from a single-env observation dict.

        Per-env counterpart to _extract_video — no batch-axis indexing.
        Strips the 'video.' prefix to match VLAStepData/processor expectations.
        """
        frames = {}
        if isinstance(obs, dict):
            for key, value in obs.items():
                if "image" in key or "video" in key:
                    clean_key = key.removeprefix("video.")
                    frames[clean_key] = np.array(value)
        return frames

    def _extract_state_single(self, obs: dict) -> dict[str, np.ndarray]:
        """Extract state values from a single-env observation dict.

        Per-env counterpart to _extract_state. Strips the 'state.' prefix
        and filters annotation/language keys.
        """
        state = {}
        if isinstance(obs, dict):
            for key, value in obs.items():
                if "image" not in key and "video" not in key and "language" not in key:
                    if "annotation" in key:
                        continue
                    clean_key = key.removeprefix("state.")
                    state[clean_key] = np.array(value)
        return state

    def _extract_language_single(self, obs: dict) -> str:
        """Extract task language instruction from a single-env observation dict."""
        if isinstance(obs, dict):
            for key, value in obs.items():
                if "language" in key or "annotation" in key or "task_description" in key:
                    if isinstance(value, (tuple, list)) and len(value) > 0:
                        return str(value[0])
                    elif isinstance(value, str):
                        return value
        return self.env_name.split("/")[-1]  # Fallback to env name

    def _reduce_success(self, info: dict) -> bool:
        """Reduce a single env's MultiStepWrapper info["success"] to a bool.

        MultiStepWrapper packs per-substep success bools into info["success"]
        as an array of length up to n_action_steps (multistep_wrapper.py:282).
        We treat the chunk as successful if ANY substep succeeded — same
        convention used by gr00t/eval/rollout_policy.py:303-313.

        Critically, naïve `bool(env_success)` would either raise (for
        multi-element arrays — "ambiguous truth value") or return True for
        any non-empty list (incl. all-False), so the np.any reduction is
        load-bearing for correctness, not a stylistic choice.
        """
        if "success" not in info:
            return False
        env_success = info["success"]
        if isinstance(env_success, (list, np.ndarray)):
            return bool(np.any(env_success))
        return bool(env_success)

    def close(self):
        """Clean up environments."""
        self.envs.close()


def save_episodes(episodes: list[dict], output_dir: str) -> None:
    """Save collected episodes as .npz files.

    Each episode is saved as episode_{idx}.npz with keys structured for
    EpisodeBuffer.load_episodes() to parse.

    Args:
        episodes: List of episode dicts from EpisodeCollector.collect().
        output_dir: Directory to write .npz files to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, ep in enumerate(episodes):
        save_dict = {
            "language": ep.get("language") or ep["env_name"].split("/")[-1],
            "env_name": ep["env_name"],
            "success": ep["success"],
            "max_progress": ep["max_progress"],
            "num_steps": ep["num_steps"],
            "num_chunks": len(ep["actions"]),
            "group_id": ep.get("group_id", 0),
            "env_seed": ep.get("env_seed", 0),
        }

        # Save per-chunk data
        for chunk_idx in range(len(ep["actions"])):
            # Video frames
            if chunk_idx < len(ep["video_frames"]):
                for cam_name, frame in ep["video_frames"][chunk_idx].items():
                    save_dict[f"video_{cam_name}_{chunk_idx}"] = frame

            # State
            if chunk_idx < len(ep["states"]):
                for state_key, state_val in ep["states"][chunk_idx].items():
                    save_dict[f"state_{state_key}_{chunk_idx}"] = state_val

            # Action (decoded physical action for env stepping)
            save_dict[f"action_{chunk_idx}"] = ep["actions"][chunk_idx]

            # Raw normalized action (50x128 tensor from model, for FM log-prob)
            if chunk_idx < len(ep.get("raw_actions", [])):
                raw_action = ep["raw_actions"][chunk_idx]
                if raw_action is not None:
                    save_dict[f"raw_action_{chunk_idx}"] = raw_action

            # Action mask from model (proper per-embodiment mask, not all-ones)
            if chunk_idx < len(ep.get("action_masks", [])):
                mask = ep["action_masks"][chunk_idx]
                if mask is not None:
                    save_dict[f"action_mask_{chunk_idx}"] = mask
                elif chunk_idx < len(ep.get("raw_actions", [])) and ep["raw_actions"][chunk_idx] is not None:
                    save_dict[f"action_mask_{chunk_idx}"] = np.ones_like(ep["raw_actions"][chunk_idx])
                else:
                    save_dict[f"action_mask_{chunk_idx}"] = np.ones((50, 128), dtype=np.float32)
            else:
                save_dict[f"action_mask_{chunk_idx}"] = np.ones((50, 128), dtype=np.float32)

            # Initial noise tensor (50×128) — the ε₀ that was denoised into this action
            if chunk_idx < len(ep.get("initial_noises", [])):
                ini_noise = ep["initial_noises"][chunk_idx]
                if ini_noise is not None:
                    save_dict[f"initial_noise_{chunk_idx}"] = ini_noise

        np.savez_compressed(output_path / f"episode_{idx:04d}.npz", **save_dict)

    print(f"Saved {len(episodes)} episodes to {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create collector
    collector = EpisodeCollector(
        env_name=args.env_name,
        group_size=args.group_size,
        max_episode_steps=args.max_episode_steps,
        n_action_steps=args.n_action_steps,
        server_host=args.server_host,
        server_port=args.server_port,
        seed=args.seed,
        debug_fast_forward=args.debug_fast_forward,
        output_dir=args.output_dir,
    )

    try:
        # Collect episodes in groups
        episodes = collector.collect(
            group_size=args.group_size,
            num_groups=args.num_groups,
            base_seed=args.seed,
            success_weight=args.success_weight,
            fast_forward_steps=args.fast_forward_steps,
            fast_forward_pct=args.fast_forward_pct,
        )

        # Save to disk
        save_episodes(episodes, args.output_dir)

    finally:
        collector.close()


if __name__ == "__main__":
    main()
