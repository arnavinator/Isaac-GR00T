"""Episode buffer and group-relative advantage computation for GRPO.

This module handles:
1. Loading collected episode data from .npz files (written by collect_episodes.py)
2. Computing group-relative advantages (the core GRPO normalization)
3. Yielding PyTorch mini-batches for the GRPO training loop

The advantage computation directly mirrors grpo_cont.py lines 325-364:
    means = final_group_reward.mean(dim=1, keepdim=True)
    stds  = final_group_reward.std(dim=1, keepdim=True)
    advantages = (final_group_reward - means) / (stds + 1e-8)

Key difference from grpo_cont.py:
- grpo_cont.py computes per-step rewards, then discounts them into a trajectory reward
- We use episodic rewards (binary success + dense progress) — no discounting needed
- Each episode gets ONE advantage, which is then divided by num_chunks and
  broadcast to each chunk in _build_chunks (mirroring grpo_cont.py:368-369).
  The division preserves the group-zero-sum invariant at the chunk level so
  every trajectory contributes equal gradient weight regardless of length.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


@dataclass
class ActionChunk:
    """A single action chunk from an episode, ready for GRPO training.

    One episode produces multiple action chunks (e.g., 720 steps / 8 exec steps = 90 chunks).
    Each chunk is one "token" for GRPO — analogous to one timestep in grpo_cont.py.

    The advantage stored here is `A_episode / num_chunks_in_episode`: the
    per-trajectory advantage spread evenly across chunks. This matches
    grpo_cont.py:368-369, where `advantages = advantages / num_steps` before
    being broadcast to every timestep in the trajectory.
    """
    # Observation data (to re-encode through backbone during training)
    video_frames: dict[str, np.ndarray]   # {camera_name: (H, W, 3) uint8}
    state: dict[str, np.ndarray]          # {state_key: (dim,) float32}
    language: str                          # Task instruction

    # Action produced by the policy (normalized, in [-1, 1])
    action: np.ndarray                    # (action_horizon, action_dim) float32

    # Raw normalized action from model output (50, 128) — for FM log-prob computation
    # This is the action in the DiT's internal space BEFORE decode_action() slices it
    raw_action: np.ndarray | None         # (50, 128) float32, or None if not available

    # Action mask for valid dimensions (handles multi-embodiment padding)
    action_mask: np.ndarray               # (50, 128) float32 when raw_action available

    # Initial noise tensor used during denoising to produce this action chunk.
    # This is the ε₀ in x_τ = (1-τ)ε₀ + τ*action — used during training to evaluate
    # the FM log-prob along the actual denoising path (not a random path).
    initial_noise: np.ndarray | None      # (50, 128) float32, or None if not available

    # GRPO advantage (same for all chunks in this episode)
    advantage: float

    # Episode-level metadata for logging
    episode_idx: int
    chunk_idx: int
    episode_reward: float
    episode_success: bool
    # Group this chunk's parent episode belongs to. Propagated from
    # GRPOEpisode.group_id in _build_chunks. Used by the stratified
    # minibatch iterator in train_grpo.py to bin chunks by group so each
    # minibatch can span all live groups. Defaults to 0 to keep the
    # dataclass constructor backward-compatible.
    group_id: int = 0

    # Pre-computed reference log-prob (set after collection, before GRPO update)
    ref_log_prob: float | None = None

    # Timestep samples used for ref_log_prob computation (reused during training)
    tau_samples: np.ndarray | None = None  # (K,) float32

    # --- Encoded-observation cache (populated in _compute_ref_log_probs) ----
    # The Eagle backbone and state encoder are frozen (no LoRA), so their outputs
    # are identical across all training epochs/minibatches. We run them once
    # during the ref log-prob pass and stash per-chunk slices here; _prepare_batch
    # then rebuilds a batched tensor from these slices instead of re-running the
    # backbone. Cleared with the rest of the chunk when buffer.clear() runs.
    # Shapes (unpadded, per-chunk):
    #   cached_backbone_features:  (seq_len, 2048)  bfloat16
    #   cached_backbone_attn_mask: (seq_len,)       bool
    #   cached_image_mask:         (seq_len,)       bool  (None if not provided)
    #   cached_state_features:     (state_horizon, 1536)  bfloat16
    #   cached_embodiment_id:      ()               long scalar tensor
    cached_backbone_features: "torch.Tensor | None" = None
    cached_backbone_attn_mask: "torch.Tensor | None" = None
    cached_image_mask: "torch.Tensor | None" = None
    cached_state_features: "torch.Tensor | None" = None
    cached_embodiment_id: "torch.Tensor | None" = None


@dataclass
class GRPOEpisode:
    """One complete episode collected from the simulation.

    Stores all data needed to reconstruct (obs, action) pairs for training.
    Loaded from .npz files written by collect_episodes.py.
    """
    # Per-chunk observation data
    video_frames: list[dict[str, np.ndarray]]   # len = num_chunks
    states: list[dict[str, np.ndarray]]         # len = num_chunks
    language: str                                 # Same for all chunks in episode

    # Per-chunk action data
    actions: list[np.ndarray]                    # len = num_chunks, each (horizon, dim)
    raw_actions: list[np.ndarray | None]         # len = num_chunks, each (50, 128) or None
    action_masks: list[np.ndarray]               # len = num_chunks
    initial_noises: list[np.ndarray | None]      # len = num_chunks, each (50, 128) or None

    # Episode-level reward signals
    success: bool                                # Binary task completion
    max_progress: float                          # Dense progress metric [0, 1]
    shaped_reward: float                         # Computed: w*success + (1-w)*progress

    # Metadata
    env_name: str
    episode_idx: int
    num_steps: int                               # Total env steps taken
    group_id: int = 0                            # Which group this episode belongs to
    env_seed: int = 0                            # Env reset seed (same within a group)

    @property
    def num_chunks(self) -> int:
        return len(self.actions)


class EpisodeBuffer:
    """Buffer for collected episodes with GRPO advantage computation.

    Usage:
        buffer = EpisodeBuffer()
        buffer.load_episodes("/tmp/grpo_episodes/iter_005/")
        buffer.compute_advantages(success_weight=0.7)
        for batch in buffer.iter_minibatches(batch_size=8):
            # train on batch
            ...
    """

    def __init__(self):
        self.episodes: list[GRPOEpisode] = []
        self.advantages: np.ndarray | None = None  # [num_episodes]
        self._chunks: list[ActionChunk] | None = None
        # Populated by compute_advantages; consumed by stats() so TB logging
        # can see how much signal each iteration actually carried.
        self._n_groups: int = 0
        self._n_dead_groups: int = 0

    def clear(self):
        """Clear buffer for next iteration.

        Explicitly nulls out per-chunk cached GPU tensors before dropping the
        chunk list. Without this, the tensors linger in the CUDA caching
        allocator's pool until the next allocator pass and can inflate
        observed GPU memory usage across iterations. After the fields are
        dropped, an empty_cache() hint encourages the allocator to release
        unused blocks back to the driver.
        """
        if self._chunks is not None:
            for chunk in self._chunks:
                chunk.cached_backbone_features = None
                chunk.cached_backbone_attn_mask = None
                chunk.cached_image_mask = None
                chunk.cached_state_features = None
                chunk.cached_embodiment_id = None
        self.episodes = []
        self.advantages = None
        self._chunks = None
        self._n_groups = 0
        self._n_dead_groups = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_episodes(self, episode_dir: str | Path) -> int:
        """Load all episode .npz files from a directory.

        Args:
            episode_dir: Directory containing episode_*.npz files.

        Returns:
            Number of episodes loaded.
        """
        episode_dir = Path(episode_dir)
        npz_files = sorted(episode_dir.glob("episode_*.npz"))

        for npz_path in npz_files:
            episode = self._load_single_episode(npz_path)
            if episode is not None:
                self.episodes.append(episode)

        return len(self.episodes)

    def _load_single_episode(self, path: Path) -> GRPOEpisode | None:
        """Load a single episode from .npz format.

        Expected .npz keys:
            - video_{camera}_{chunk_idx}: (H, W, 3) uint8
            - state_{key}_{chunk_idx}: (dim,) float32
            - action_{chunk_idx}: (horizon, dim) float32
            - action_mask_{chunk_idx}: (horizon, dim) float32
            - initial_noise_{chunk_idx}: (50, 128) float32
            - language: string
            - success: bool
            - max_progress: float
            - env_name: string
            - num_steps: int
            - num_chunks: int
        """
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            return None

        num_chunks = int(data["num_chunks"])
        language = str(data["language"])
        env_name = str(data["env_name"])
        success = bool(data["success"])
        max_progress = float(data["max_progress"])
        num_steps = int(data["num_steps"])
        group_id = int(data["group_id"]) if "group_id" in data else 0
        env_seed = int(data["env_seed"]) if "env_seed" in data else 0

        # Reconstruct per-chunk data
        video_frames = []
        states = []
        actions = []
        raw_actions = []
        action_masks = []
        initial_noises = []

        # Identify camera names from keys
        camera_names = set()
        for key in data.files:
            if key.startswith("video_") and key.count("_") >= 2:
                # Extract camera name: video_{camera}_{chunk_idx}
                parts = key.split("_")
                camera_name = "_".join(parts[1:-1])  # Handle multi-word camera names
                camera_names.add(camera_name)

        # Identify state keys
        state_keys = set()
        for key in data.files:
            if key.startswith("state_") and key.count("_") >= 2:
                parts = key.split("_")
                state_key = "_".join(parts[1:-1])
                state_keys.add(state_key)

        for i in range(num_chunks):
            # Video frames for this chunk
            chunk_video = {}
            for cam in camera_names:
                key = f"video_{cam}_{i}"
                if key in data:
                    chunk_video[cam] = data[key]
            video_frames.append(chunk_video)

            # State for this chunk
            chunk_state = {}
            for sk in state_keys:
                key = f"state_{sk}_{i}"
                if key in data:
                    chunk_state[sk] = data[key]
            states.append(chunk_state)

            # Action and mask
            actions.append(data[f"action_{i}"])
            action_masks.append(data[f"action_mask_{i}"])
            # Raw normalized action (may not exist in older collections)
            raw_key = f"raw_action_{i}"
            raw_actions.append(data[raw_key] if raw_key in data else None)
            # Initial noise tensor (may not exist in older collections)
            noise_key = f"initial_noise_{i}"
            initial_noises.append(data[noise_key] if noise_key in data else None)

        return GRPOEpisode(
            video_frames=video_frames,
            states=states,
            language=language,
            actions=actions,
            raw_actions=raw_actions,
            action_masks=action_masks,
            initial_noises=initial_noises,
            success=success,
            max_progress=max_progress,
            shaped_reward=0.0,  # Computed in compute_advantages()
            env_name=env_name,
            episode_idx=len(self.episodes),
            num_steps=num_steps,
            group_id=group_id,
            env_seed=env_seed,
        )

    def compute_advantages(self, success_weight: float = 1.0, max_episode_steps: int = 520) -> np.ndarray:
        """Compute group-relative advantages for all episodes (one per episode).

        This is the CORE GRPO computation, mirroring grpo_cont.py lines 362-364:
            means = final_group_reward.mean(dim=1, keepdim=True)
            stds  = final_group_reward.std(dim=1, keepdim=True)
            advantages = (final_group_reward - means) / (stds + 1e-8)

        Advantages are normalized WITHIN each group (episodes sharing the same
        group_id / env_seed). This compares rollouts from the same initial state,
        isolating the effect of policy noise from environmental randomness.

        After shaped reward computation, rewards are time-scaled: faster solutions
        get higher reward (reward / num_steps * max_episode_steps). This creates
        variance even in all-success groups where binary rewards are identical.

        Note: this returns ONE advantage per episode. The per-chunk division
        (A_episode / num_chunks, matching grpo_cont.py:368-369) happens later in
        _build_chunks when episodes are flattened into ActionChunks.

        Args:
            success_weight: Weight for binary success in shaped reward.
                reward = success_weight * success + (1 - success_weight) * max_progress
            max_episode_steps: Maximum episode steps (used for time-scaling normalization).

        Returns:
            advantages: [num_episodes] array of group-relative normalized per-episode advantages.
        """
        if not self.episodes:
            self.advantages = np.array([])
            self._n_groups = 0
            self._n_dead_groups = 0
            return self.advantages

        # Step 1: Compute shaped rewards per episode
        rewards = np.array([
            success_weight * float(ep.success) + (1 - success_weight) * ep.max_progress
            for ep in self.episodes
        ])

        # Step 1b: Time-scale rewards (faster solutions get higher reward)
        for i, ep in enumerate(self.episodes):
            if ep.num_steps > 0:
                rewards[i] = rewards[i] / ep.num_steps * max_episode_steps

        # Store shaped rewards in episodes
        for ep, r in zip(self.episodes, rewards):
            ep.shaped_reward = float(r)

        # Step 2: Group-relative normalization (per group, not global)
        # Same formula as grpo_cont.py line 364, applied per group:
        #   advantages[g] = (rewards[g] - rewards[g].mean()) / (rewards[g].std() + 1e-8)
        # NOTE: Use ddof=1 (Bessel's correction) to match PyTorch's tensor.std()
        self.advantages = np.zeros_like(rewards)

        # Identify unique groups
        group_ids = np.array([ep.group_id for ep in self.episodes])
        unique_groups = np.unique(group_ids)

        n_dead = 0
        for gid in unique_groups:
            mask = group_ids == gid
            group_rewards = rewards[mask]

            if len(group_rewards) <= 1:
                # Single episode in group — no comparison possible
                self.advantages[mask] = 0.0
                n_dead += 1
            else:
                mean_r = group_rewards.mean()
                std_r = group_rewards.std(ddof=1)
                # Threshold of 1e-4 (not 1e-8) prevents micro-std groups from
                # amplifying noise into giant advantages: with rewards ~ O(1)
                # and std=1e-6, the division produces ±1e6 advantages that
                # then dominate the per-minibatch z-score. With time-scaled
                # binary rewards (1.0 / num_steps * max_steps in [~1, ~5]),
                # any group_std < 1e-4 means the group is effectively
                # all-same-reward and provides no useful gradient signal.
                if std_r < 1e-4:
                    # No meaningful signal within group
                    self.advantages[mask] = 0.0
                    n_dead += 1
                else:
                    self.advantages[mask] = (group_rewards - mean_r) / std_r

        self._n_groups = int(len(unique_groups))
        self._n_dead_groups = int(n_dead)

        return self.advantages

    def _build_chunks(self) -> list[ActionChunk]:
        """Flatten episodes into individual action chunks for mini-batching.

        Each episode becomes N chunks (one per action query). Each chunk gets
        `A_episode / N` as its advantage — mirroring grpo_cont.py:368-369, which
        divides the per-trajectory advantage by `num_steps` before broadcasting
        to each timestep.

        Why divide: group-relative normalization guarantees Σ A_episode = 0
        within a group. Dividing by num_chunks preserves this invariant at the
        chunk level (Σ_chunks A_chunk = Σ_episodes A_episode = 0), so every
        trajectory contributes equal total gradient weight regardless of length.
        Without the division, long episodes would dominate the gradient purely
        by having more chunks.
        """
        if self._chunks is not None:
            return self._chunks

        assert self.advantages is not None, "Call compute_advantages() first"

        chunks = []
        for ep_idx, (episode, advantage) in enumerate(
            zip(self.episodes, self.advantages)
        ):
            n_chunks = max(episode.num_chunks, 1)
            per_chunk_advantage = float(advantage) / n_chunks
            for chunk_idx in range(episode.num_chunks):
                chunk = ActionChunk(
                    video_frames=episode.video_frames[chunk_idx],
                    state=episode.states[chunk_idx],
                    language=episode.language,
                    action=episode.actions[chunk_idx],
                    raw_action=episode.raw_actions[chunk_idx] if chunk_idx < len(episode.raw_actions) else None,
                    action_mask=episode.action_masks[chunk_idx],
                    initial_noise=episode.initial_noises[chunk_idx] if chunk_idx < len(episode.initial_noises) else None,
                    advantage=per_chunk_advantage,
                    episode_idx=ep_idx,
                    chunk_idx=chunk_idx,
                    episode_reward=episode.shaped_reward,
                    episode_success=episode.success,
                    group_id=episode.group_id,
                )
                chunks.append(chunk)

        self._chunks = chunks
        return chunks

    def iter_minibatches(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> Iterator[list[ActionChunk]]:
        """Yield mini-batches of action chunks for training.

        Mirrors grpo_cont.py's minibatch loop (lines 382-386):
            for start in range(0, grouped_batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idxs = b_inds[start:end]

        Args:
            batch_size: Number of action chunks per mini-batch.
            shuffle: Whether to randomly shuffle chunks (recommended for training).
            seed: Random seed for reproducible shuffling.

        Yields:
            Lists of ActionChunk objects, each list has length <= batch_size.
        """
        chunks = self._build_chunks()

        if not chunks:
            return

        # Create index permutation
        indices = np.arange(len(chunks))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        # Yield mini-batches
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            yield [chunks[i] for i in batch_indices]

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_chunks(self) -> int:
        """Total number of action chunks across all episodes."""
        return sum(ep.num_chunks for ep in self.episodes)

    @property
    def success_rate(self) -> float:
        """Fraction of episodes that succeeded."""
        if not self.episodes:
            return 0.0
        return sum(ep.success for ep in self.episodes) / len(self.episodes)

    @property
    def mean_progress(self) -> float:
        """Average dense progress across episodes."""
        if not self.episodes:
            return 0.0
        return np.mean([ep.max_progress for ep in self.episodes])

    def stats(self) -> dict:
        """Summary statistics for logging."""
        if not self.episodes:
            return {}

        rewards = [ep.shaped_reward for ep in self.episodes]
        num_steps_list = [ep.num_steps for ep in self.episodes]

        # Per-group success rate distribution. Each group's rate is
        # (n_successes_in_group / n_episodes_in_group); aggregated to
        # min / median / max so TB shows the spread without histograms.
        group_to_total: dict[int, int] = {}
        group_to_succ: dict[int, int] = {}
        for ep in self.episodes:
            group_to_total[ep.group_id] = group_to_total.get(ep.group_id, 0) + 1
            if ep.success:
                group_to_succ[ep.group_id] = group_to_succ.get(ep.group_id, 0) + 1
        per_group_success = [
            group_to_succ.get(gid, 0) / group_to_total[gid]
            for gid in group_to_total
        ]

        return {
            "num_episodes": self.num_episodes,
            "num_chunks": self.num_chunks,
            "success_rate": self.success_rate,
            "mean_progress": self.mean_progress,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_advantage": float(self.advantages.mean()) if self.advantages is not None else 0,
            "std_advantage": float(self.advantages.std()) if self.advantages is not None else 0,
            "pct_positive_advantage": float((self.advantages > 0).mean()) if self.advantages is not None else 0,
            # Group quality (populated by compute_advantages); diagnoses how
            # much of the iter's signal got filtered out by the dead-group
            # threshold downstream.
            "n_groups": self._n_groups,
            "n_dead_groups": self._n_dead_groups,
            "n_live_groups": max(0, self._n_groups - self._n_dead_groups),
            # Per-group success rate spread (min/median/max across groups).
            # Reveals when the iter average masks a bimodal "some seeds at
            # 100%, others at 0%" pattern.
            "group_success_min": float(min(per_group_success)) if per_group_success else 0.0,
            "group_success_median": float(np.median(per_group_success)) if per_group_success else 0.0,
            "group_success_max": float(max(per_group_success)) if per_group_success else 0.0,
            # Trajectory length stats. Catches the "model is rushing" failure
            # mode (mean_num_steps drops below baseline) before success_rate
            # collapse becomes visible.
            "mean_num_steps": float(np.mean(num_steps_list)),
            "std_num_steps": float(np.std(num_steps_list)),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Episode Buffer Self-Test ===\n")

    # Test 1: Per-group advantage computation (2 groups of 5)
    print("Test 1: Per-group advantage normalization")
    buffer = EpisodeBuffer()

    # Group 0: 3 successes, 2 failures (seed=100)
    # Group 1: 1 success, 4 failures (seed=200)
    for i in range(10):
        group_id = i // 5  # 0 for first 5, 1 for last 5
        ep = GRPOEpisode(
            video_frames=[{}],
            states=[{}],
            language="test task",
            actions=[np.zeros((16, 12))],
            raw_actions=[np.zeros((50, 128))],
            action_masks=[np.ones((50, 128))],
            initial_noises=[np.zeros((50, 128))],
            success=(i % 5 >= 2) if group_id == 0 else (i % 5 == 0),
            max_progress=(i % 5) / 5.0,
            shaped_reward=0.0,
            env_name="test_env",
            episode_idx=i,
            num_steps=100,
            group_id=group_id,
            env_seed=100 + group_id * 100,
        )
        buffer.episodes.append(ep)

    advantages = buffer.compute_advantages(success_weight=0.7)

    print("Rewards:", [f"{ep.shaped_reward:.3f}" for ep in buffer.episodes])
    print("Group IDs:", [ep.group_id for ep in buffer.episodes])
    print("Advantages:", [f"{a:.3f}" for a in advantages])

    # Verify per-group normalization: mean within each group should be ~0
    group0_adv = advantages[:5]
    group1_adv = advantages[5:]
    print(f"Group 0 mean: {group0_adv.mean():.6f} (should be ~0)")
    print(f"Group 1 mean: {group1_adv.mean():.6f} (should be ~0)")
    assert abs(group0_adv.mean()) < 1e-6, f"Group 0 mean should be ~0, got {group0_adv.mean()}"
    assert abs(group1_adv.mean()) < 1e-6, f"Group 1 mean should be ~0, got {group1_adv.mean()}"

    # Verify successes get positive advantages within their group
    # Group 0: episodes 2,3,4 succeed; Group 1: episode 5 succeeds
    assert advantages[4] > 0, "Group 0 success should have positive advantage"
    assert advantages[5] > 0, "Group 1 success should have positive advantage"
    assert advantages[0] < 0, "Group 0 failure should have negative advantage"
    assert advantages[6] < 0, "Group 1 failure should have negative advantage"
    print("  PASS: per-group normalization correct\n")

    # Test 2: Single-episode group (no signal)
    print("Test 2: Single-episode group gives zero advantage")
    buffer2 = EpisodeBuffer()
    buffer2.episodes.append(GRPOEpisode(
        video_frames=[{}], states=[{}], language="test",
        actions=[np.zeros((16, 12))], raw_actions=[np.zeros((50,128))],
        action_masks=[np.ones((50,128))], initial_noises=[np.zeros((50, 128))],
        success=True, max_progress=1.0, shaped_reward=0.0,
        env_name="test", episode_idx=0, num_steps=8,
        group_id=0, env_seed=42,
    ))
    adv2 = buffer2.compute_advantages()
    assert adv2[0] == 0.0, "Single-episode group should give zero advantage"
    print("  PASS\n")

    # Test 3: Mini-batch iteration
    print("Test 3: Mini-batch completeness")
    chunks = list(buffer.iter_minibatches(batch_size=3, seed=42))
    total_chunks = sum(len(batch) for batch in chunks)
    print(f"  {len(chunks)} batches, {total_chunks} total chunks")
    assert total_chunks == buffer.num_chunks
    print("  PASS")

    print("\nAll tests PASSED.")
