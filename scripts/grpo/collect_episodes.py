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
        env = gym.make(env_name, render_mode=None)

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
        trajectories and outcomes. One batch = one group (no sub-batching needed).
        """
        # Reset ALL envs with the SAME seed — identical initial states
        reset_seeds = [group_seed] * self.group_size
        observations, infos = self.envs.reset(seed=reset_seeds)

        # Track episodes for this group
        episodes_in_progress = [
            self._new_episode(group_id, group_seed) for _ in range(group_size)
        ]
        done_flags = [False] * group_size

        while not all(done_flags):
            # Query model server for actions
            actions, initial_noise, raw_actions, action_masks_batch = self._get_actions_from_server(observations)

            # Record observation and action for each active env
            for i in range(group_size):
                if done_flags[i]:
                    continue
                ep = episodes_in_progress[i]
                ep["video_frames"].append(self._extract_video(observations, i))
                ep["states"].append(self._extract_state(observations, i))
                ep["actions"].append(self._extract_per_env(actions, i))
                if raw_actions is not None and hasattr(raw_actions, '__getitem__') and len(raw_actions) > i:
                    ep["raw_actions"].append(raw_actions[i])
                else:
                    ep["raw_actions"].append(None)
                if action_masks_batch is not None and hasattr(action_masks_batch, '__getitem__') and len(action_masks_batch) > i:
                    ep["action_masks"].append(action_masks_batch[i])
                else:
                    ep["action_masks"].append(None)
                if initial_noise is not None and hasattr(initial_noise, '__getitem__') and len(initial_noise) > i:
                    ep["initial_noises"].append(initial_noise[i])
                else:
                    ep["initial_noises"].append(None)
                # Capture language from first observation (constant per episode)
                if ep["language"] is None:
                    ep["language"] = self._extract_language(observations, i)

            # Step all environments
            next_observations, rewards, terminations, truncations, infos = self.envs.step(actions)

            # Check for completions
            dones = terminations | truncations
            for i in range(group_size):
                if done_flags[i]:
                    continue

                ep = episodes_in_progress[i]
                ep["num_steps"] += self.n_action_steps

                # Track max progress
                progress = compute_dense_progress(
                    self.envs.envs[i] if hasattr(self.envs, 'envs') else self.envs,
                    self.task_type,
                )
                ep["max_progress"] = max(ep["max_progress"], progress)

                if dones[i]:
                    done_flags[i] = True
                    success = self._extract_success(infos, i)
                    ep["success"] = success
                    ep["shaped_reward"] = compute_shaped_reward(
                        success, ep["max_progress"], success_weight
                    )

            observations = next_observations

        return episodes_in_progress

    def _collect_one_group_with_fast_forward(
        self,
        group_seed: int,
        group_size: int,
        group_id: int,
        fast_forward_steps: int,
        success_weight: float,
    ) -> list[dict]:
        """Collect one group with fast-forward branching.

        Focuses GRPO signal on the critical manipulation phase by skipping the
        early approach trajectory. Adapted from branching_rollout.py.

        Phases:
          1. Reset all envs with the same seed
          2. Fast-forward env 0 for fast_forward_steps outer steps using the policy
          3. Save env 0's MuJoCo sim state at the branch point
          4. Restore that state into envs 1..G-1 (all now at identical branch point)
          5. Sync each env's MultiStepWrapper internal state
          6. Continue all envs independently until done (diverge via policy noise)

        If env 0 terminates during fast-forward (task solved early or error),
        falls back to normal collection from the seed.
        """
        # Phase 1: Reset all envs with same seed
        reset_seeds = [group_seed] * self.group_size
        observations, infos = self.envs.reset(seed=reset_seeds)

        # Phase 2: Fast-forward env 0 only
        # We step ALL envs (vectorized env requires it), but only env 0's trajectory matters.
        # The other envs' work is discarded — they'll be overwritten with env 0's state.
        consumed_substeps = 0
        for step in range(fast_forward_steps):
            actions, _, _, _ = self._get_actions_from_server(observations)
            observations, rewards, terms, truncs, infos = self.envs.step(actions)
            consumed_substeps += self.n_action_steps

            # If env 0 terminated during fast-forward, can't branch — fall back
            if terms[0] or truncs[0]:
                print(f"    Env 0 terminated at step {step} during fast-forward, falling back to normal collection")
                return self._collect_one_group(group_seed, group_size, group_id, success_weight)

        # Phase 3: Save env 0's MuJoCo sim state
        sim_state = self._get_sim_state(env_idx=0)
        if sim_state is None:
            print("    WARNING: Could not save sim state, falling back to normal collection")
            return self._collect_one_group(group_seed, group_size, group_id, success_weight)

        # Phase 4: Restore env 0's state into envs 1..G-1
        for i in range(1, group_size):
            self._restore_sim_state(env_idx=i, sim_state=sim_state)

        # Phase 5: Sync each restored env's MultiStepWrapper internal state
        # Read fresh observation from each restored env and update its wrapper
        for i in range(1, group_size):
            restored_obs = self._read_obs_after_restore(env_idx=i)
            self._sync_wrapper(env_idx=i, obs=restored_obs, consumed_substeps=consumed_substeps)

        # Re-read the vectorized observation (env 0 is already correct,
        # envs 1..G-1 now have matching state)
        observations = self._read_vectorized_obs()

        # Debug: verify all envs have identical state after branching
        if self.debug_fast_forward:
            self._verify_branch_point(group_id, group_seed, fast_forward_steps)

        # Phase 6: Continue all envs independently (same logic as _collect_one_group)
        episodes_in_progress = [
            self._new_episode(group_id, group_seed) for _ in range(group_size)
        ]
        done_flags = [False] * group_size

        while not all(done_flags):
            actions, initial_noise, raw_actions, action_masks_batch = self._get_actions_from_server(observations)

            for i in range(group_size):
                if done_flags[i]:
                    continue
                ep = episodes_in_progress[i]
                ep["video_frames"].append(self._extract_video(observations, i))
                ep["states"].append(self._extract_state(observations, i))
                ep["actions"].append(self._extract_per_env(actions, i))
                if raw_actions is not None and hasattr(raw_actions, '__getitem__') and len(raw_actions) > i:
                    ep["raw_actions"].append(raw_actions[i])
                else:
                    ep["raw_actions"].append(None)
                if action_masks_batch is not None and hasattr(action_masks_batch, '__getitem__') and len(action_masks_batch) > i:
                    ep["action_masks"].append(action_masks_batch[i])
                else:
                    ep["action_masks"].append(None)
                if initial_noise is not None and hasattr(initial_noise, '__getitem__') and len(initial_noise) > i:
                    ep["initial_noises"].append(initial_noise[i])
                else:
                    ep["initial_noises"].append(None)
                # Capture language from first observation (constant per episode)
                if ep["language"] is None:
                    ep["language"] = self._extract_language(observations, i)

            next_observations, rewards, terminations, truncations, infos = self.envs.step(actions)

            dones = terminations | truncations
            for i in range(group_size):
                if done_flags[i]:
                    continue

                ep = episodes_in_progress[i]
                ep["num_steps"] += self.n_action_steps

                progress = compute_dense_progress(
                    self.envs.envs[i] if hasattr(self.envs, 'envs') else self.envs,
                    self.task_type,
                )
                ep["max_progress"] = max(ep["max_progress"], progress)

                if dones[i]:
                    done_flags[i] = True
                    success = self._extract_success(infos, i)
                    ep["success"] = success
                    ep["shaped_reward"] = compute_shaped_reward(
                        success, ep["max_progress"], success_weight
                    )

            observations = next_observations

        return episodes_in_progress

    # ─── MuJoCo state save/restore helpers ───────────────────────────────
    # Adapted from branching_rollout.py:214-266

    def _get_sim_state(self, env_idx: int) -> np.ndarray | None:
        """Save the MuJoCo sim state from a sub-env as a flat array."""
        try:
            wrapper = self.envs.envs[env_idx]
            robosuite_env = wrapper.unwrapped.env
            return np.array(robosuite_env.sim.get_state().flatten())
        except (AttributeError, TypeError, IndexError):
            return None

    def _restore_sim_state(self, env_idx: int, sim_state: np.ndarray) -> None:
        """Restore a saved MuJoCo sim state into a sub-env."""
        wrapper = self.envs.envs[env_idx]
        robosuite_env = wrapper.unwrapped.env
        robosuite_env.sim.set_state_from_flattened(sim_state)
        robosuite_env.sim.forward()
        if hasattr(robosuite_env, "update_state"):
            robosuite_env.update_state()
        elif hasattr(robosuite_env, "update_sites"):
            robosuite_env.update_sites()

    def _read_obs_after_restore(self, env_idx: int) -> dict:
        """Read a fresh observation from a sub-env after state restoration."""
        wrapper = self.envs.envs[env_idx]
        base_env = wrapper.unwrapped
        robosuite_env = base_env.env
        raw_obs = robosuite_env._get_observations()
        basic_obs = base_env.get_basic_observation(raw_obs)
        return base_env.get_groot_observation(basic_obs)

    def _sync_wrapper(self, env_idx: int, obs: dict, consumed_substeps: int) -> None:
        """Sync a sub-env's MultiStepWrapper state after sim restoration.

        The MultiStepWrapper tracks an obs deque, reward list, and done list.
        After restoring the MuJoCo state (which the wrapper doesn't know about),
        we must update these so _get_obs() produces valid stacked observations.
        """
        from collections import deque as _deque
        wrapper = self.envs.envs[env_idx]
        wrapper.reward = [0.0] * consumed_substeps
        wrapper.done = [False] * consumed_substeps
        wrapper.obs = _deque(
            [obs] * (wrapper.max_steps_needed + 1),
            maxlen=wrapper.max_steps_needed + 1,
        )

    def _read_vectorized_obs(self) -> dict:
        """Read stacked observations from all sub-envs and batch them.

        After restoring state in individual sub-envs, the SyncVectorEnv's
        cached observations are stale. This reads fresh observations from
        each sub-env's MultiStepWrapper and combines them into the batched
        format that the policy server expects.
        """
        all_obs = []
        for i in range(self.group_size):
            wrapper = self.envs.envs[i]
            obs = wrapper._get_obs(wrapper.video_delta_indices, wrapper.state_delta_indices)
            all_obs.append(obs)

        # Stack into batched format: {key: (group_size, ...)}
        batched = {}
        for key in all_obs[0]:
            vals = [obs[key] for obs in all_obs]
            if isinstance(vals[0], np.ndarray):
                batched[key] = np.stack(vals, axis=0)
            elif isinstance(vals[0], str):
                batched[key] = tuple(vals)
            else:
                batched[key] = vals
        return batched

    def _verify_branch_point(self, group_id: int, group_seed: int, ff_steps: int) -> None:
        """Debug verification: confirm all envs have identical state after branching.

        Saves a montage image showing camera views from all envs side-by-side,
        plus numerical comparison of sim states. If the branch worked correctly,
        all envs should show the exact same robot pose, object positions, and
        kitchen scene.

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

        # 2. Compare observations numerically
        observations = []
        for i in range(self.group_size):
            obs = self._read_obs_after_restore(i)
            observations.append(obs)

        ref_obs = observations[0]
        for i in range(1, self.group_size):
            for key in ref_obs:
                if isinstance(ref_obs[key], np.ndarray):
                    diff = np.abs(ref_obs[key].astype(float) - observations[i][key].astype(float)).max()
                    if diff > 1e-5:
                        print(f"    [DEBUG] obs key '{key}' differs: env 0 vs env {i}, max_diff={diff:.6f}")

        # 3. Render camera views from all envs and save as montage
        try:
            all_frames = []
            for i in range(self.group_size):
                obs = observations[i]
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
                # Stack envs vertically: each row = one env's camera views
                # Top row = env 0 (source), rows below = restored envs
                # All rows should look identical if branch worked correctly
                montage = np.concatenate(all_frames, axis=0)
                out_path = debug_dir / f"group{group_id:03d}_seed{group_seed}_ff{ff_steps}.png"

                try:
                    import cv2
                    cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
                    print(f"    [DEBUG] Montage saved: {out_path}")
                    print(f"    [DEBUG] Layout: {self.group_size} rows (envs) × {len(env_frames)} cols (cameras)")
                    print(f"    [DEBUG] All rows should look identical if branch is correct")
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

    def _extract_video(self, observations, env_idx: int) -> dict[str, np.ndarray]:
        """Extract video frames for one environment from batched observations.

        Strips the 'video.' prefix so keys match VLAStepData/processor expectations
        (e.g., 'video.res256_image_side_0' → 'res256_image_side_0').
        """
        frames = {}
        if isinstance(observations, dict):
            for key, value in observations.items():
                if "image" in key or "video" in key:
                    if hasattr(value, '__getitem__') and len(value) > env_idx:
                        clean_key = key.removeprefix("video.")
                        frames[clean_key] = np.array(value[env_idx])
        return frames

    def _extract_state(self, observations, env_idx: int) -> dict[str, np.ndarray]:
        """Extract state values for one environment from batched observations.

        Strips the 'state.' prefix so keys match VLAStepData/processor expectations
        (e.g., 'state.gripper_qpos' → 'gripper_qpos'). Filters out annotation keys.
        """
        state = {}
        if isinstance(observations, dict):
            for key, value in observations.items():
                if "image" not in key and "video" not in key and "language" not in key:
                    if "annotation" in key:
                        continue
                    if hasattr(value, '__getitem__') and len(value) > env_idx:
                        clean_key = key.removeprefix("state.")
                        state[clean_key] = np.array(value[env_idx])
        return state

    def _extract_language(self, observations, env_idx: int) -> str:
        """Extract task language instruction for one env from batched observations.

        The language key in RoboCasa flat observations is typically
        'annotation.human.action.task_description' — a tuple/list of strings (one per env).
        Returns the string instruction (e.g., "open the drawer").
        """
        if isinstance(observations, dict):
            for key, value in observations.items():
                if "language" in key or "annotation" in key or "task_description" in key:
                    if isinstance(value, (tuple, list)) and len(value) > env_idx:
                        return str(value[env_idx])
                    elif isinstance(value, str):
                        return value
        return self.env_name.split("/")[-1]  # Fallback to env name

    def _extract_success(self, infos: dict, env_idx: int) -> bool:
        """Extract success flag from environment info dict."""
        # Try multiple locations where RoboCasa stores success
        if "success" in infos:
            val = infos["success"]
            if hasattr(val, '__getitem__'):
                return bool(val[env_idx])
            return bool(val)

        if "final_info" in infos:
            final = infos["final_info"]
            if isinstance(final, (list, np.ndarray)) and len(final) > env_idx:
                if isinstance(final[env_idx], dict):
                    return bool(final[env_idx].get("success", False))

        return False

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
