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
- We use episodic sparse binary rewards (task success) — no discounting needed
- Each episode gets ONE advantage, which is then divided by num_chunks and
  broadcast to each chunk in _build_chunks (mirroring grpo_cont.py:368-369).
  The division preserves the group-zero-sum invariant at the chunk level so
  every trajectory contributes equal gradient weight regardless of length.
- ANCHOR groups (all-success) are the one exception to the group-relative
  formula: their group-mean baseline gives exactly 0, so they optionally take a
  constant positive advantage instead. See compute_advantages and README
  "Anchor groups".
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

    # True when the parent episode belongs to an ANCHOR group (all-success;
    # see compute_advantages). Anchor rows carry a small constant positive
    # advantage (or 0 for KL-only anchoring) instead of a group-relative
    # z-score, so every downstream mechanism that is DEFINED by advantage sign
    # — the balanced sampler's pos/neg pools, the dynamic-epoch success
    # fraction, PAWS' alive-mass split, the pos/neg clipfrac buckets — must
    # exclude them. They are also never renormalized against minibatch-local
    # statistics; see _grpo_update_inner.
    is_anchor: bool = False

    # Pre-computed reference log-prob (set after collection, before GRPO update)
    ref_log_prob: float | None = None

    # Pre-computed BASE-MODEL log-prob (set after collection, before GRPO update).
    # Computed in the same no_grad pass as ref_log_prob but with LoRA adapters
    # disabled — i.e., evaluated against the pretrained DiT. Used for KL anchoring
    # to the base policy when GRPOConfig.kl_coef_base_model > 0; remains None when
    # the term is disabled (default), and the GRPO update path skips its KL
    # contribution accordingly.
    base_log_prob: float | None = None

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
    shaped_reward: float                         # Computed reward for advantages (binary success, time-scaled if enabled)

    # Metadata
    env_name: str
    episode_idx: int
    num_steps: int                               # Total env steps taken
    group_id: int = 0                            # Which group this episode belongs to
    env_seed: int = 0                            # Env reset seed (same within a group)
    is_anchor: bool = False                      # Set by compute_advantages for all-success groups

    @property
    def num_chunks(self) -> int:
        return len(self.actions)


class EpisodeBuffer:
    """Buffer for collected episodes with GRPO advantage computation.

    Usage:
        buffer = EpisodeBuffer()
        buffer.load_episodes("/tmp/grpo_episodes/iter_005/")
        buffer.compute_advantages()
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
        # Anchor-group bookkeeping (all-success groups admitted by
        # include_anchor_groups). Kept separate from _n_dead_groups so the
        # existing dead/live curves keep their meaning.
        self._n_anchor_groups: int = 0
        self._n_anchor_episodes: int = 0
        self._n_anchor_episodes_dropped: int = 0

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
        self._n_anchor_groups = 0
        self._n_anchor_episodes = 0
        self._n_anchor_episodes_dropped = 0
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
            shaped_reward=0.0,  # Computed in compute_advantages()
            env_name=env_name,
            episode_idx=len(self.episodes),
            num_steps=num_steps,
            group_id=group_id,
            env_seed=env_seed,
        )

    def compute_advantages(
        self,
        max_episode_steps: int = 520,
        anchor_advantage: float = 0.0,
        include_anchor_groups: bool = False,
        anchor_max_row_frac: float = 1.0,
    ) -> np.ndarray:
        """Compute group-relative advantages for all episodes (one per episode).

        This is the CORE GRPO computation, mirroring grpo_cont.py lines 362-364:
            means = final_group_reward.mean(dim=1, keepdim=True)
            stds  = final_group_reward.std(dim=1, keepdim=True)
            advantages = (final_group_reward - means) / (stds + 1e-8)

        Advantages are normalized WITHIN each group (episodes sharing the same
        group_id / env_seed). This compares rollouts from the same initial state,
        isolating the effect of policy noise from environmental randomness.

        Rewards are sparse binary (1.0 on task success, 0.0 otherwise). The
        time-scaling step below (faster solutions get higher reward) is
        currently DISABLED; see the block comment for the ablation rationale.

        Groups are classified three ways (see README "Anchor groups"):
          - SIGNAL (0 < k < G): group-relative z-score, formula untouched.
          - ANCHOR (k == G): all-success. `std_r == 0` makes the group-mean
            baseline give exactly 0, so with include_anchor_groups these get a
            constant `anchor_advantage` instead and are marked `is_anchor`.
          - DEAD (k == 0, or a singleton group): advantage 0, filtered before
            any forward pass.
        With include_anchor_groups=False, anchor groups fall through to DEAD and
        every value returned here is bit-identical to the pre-anchor behavior.

        Note: this returns ONE advantage per episode. The per-chunk division
        (A_episode / num_chunks, matching grpo_cont.py:368-369) happens later in
        _build_chunks when episodes are flattened into ActionChunks.

        Args:
            max_episode_steps: Maximum episode steps (used for time-scaling normalization).
            anchor_advantage: Per-episode advantage for anchor groups. 0.0 keeps
                anchor rows in the batch for the KL terms only.
            include_anchor_groups: Whether all-success groups become anchors at
                all. False = they stay dead (default, pre-anchor behavior).
            anchor_max_row_frac: Cap on anchor chunks as a multiple of the
                signal chunk count. Ignored when there are no signal chunks.

        Returns:
            advantages: [num_episodes] array of per-episode advantages (signal
            groups group-relative normalized, anchor groups constant).
        """
        # Any previously-built chunk list is now stale: it carries the old
        # advantages and is_anchor flags. Drop the memo so _build_chunks rebuilds
        # from the values computed here. Without this, a second call with a
        # different config returns chunks whose stale NON-ZERO advantage passes
        # the update's live filter even though the gate is now off — the flag
        # being gated does not help, because it is the advantage that admits them.
        self._chunks = None
        if not self.episodes:
            self.advantages = np.array([])
            self._n_groups = 0
            self._n_dead_groups = 0
            self._n_anchor_groups = 0
            self._n_anchor_episodes = 0
            self._n_anchor_episodes_dropped = 0
            return self.advantages

        # Step 1: Sparse binary reward per episode (1.0 on success, else 0.0).
        rewards = np.array([float(ep.success) for ep in self.episodes])

        # Step 1b: Time-scale rewards (faster solutions get higher reward)
        # DISABLED. The single-scene ablation experiments (toy_lr3.0e-5_v2/v3
        # on seed 305067) confirmed time-scaling was the dominant cause of the
        # success-rate collapse pattern observed across multiple LRs:
        #   - With time-scaling: all-success groups stay alive (variance from
        #     num_steps differences) → gradient becomes pure "be faster"
        #     pressure → policy moves toward fragile fast solutions →
        #     failures rise → pct_positive_advantage flips → avoidance
        #     gradient walks policy AWAY from working trajectories →
        #     ratio_min collapses, clipfrac >0.2, success collapses.
        #   - Without time-scaling: all-success/all-fail groups go std=0 →
        #     auto-filtered by the dead-group threshold below → only mixed
        #     groups contribute gradient. Convergence stops itself once a
        #     group hits all-success. Verified end-to-end in v3: success
        #     climbed 0.50 → 0.83 and HELD; clipfrac stayed near 0;
        #     mean_ratio stayed near 1.
        # `mean_num_steps` still dropped naturally in v3 without an explicit
        # speed bonus (failures hit truncation; successes terminate early),
        # so disabling this block does NOT cost us trajectory efficiency.
        # If you ever want a speed component back, prefer a CAPPED multiplier
        # (e.g., min(1.5, max_steps/num_steps)) so a fast success gets at
        # most 1.5× a slow one — not the original ~9× that drove the
        # asymmetric gradient collapse.
        # for i, ep in enumerate(self.episodes):
        #     if ep.num_steps > 0:
        #         rewards[i] = rewards[i] / ep.num_steps * max_episode_steps

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
        # Anchor groups are resolved in a second pass: the row budget is a
        # multiple of the SIGNAL chunk count, which isn't known until every
        # group has been classified.
        anchor_gids: list[int] = []
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
                #
                # Under the binary reward this is never a close call: the
                # per-group std is either exactly 0 (all G outcomes identical)
                # or at least 1/sqrt(G) — 3500x the threshold at G=8 — so the
                # branch below is an exact "were all outcomes the same?" test.
                if std_r < 1e-4:
                    if include_anchor_groups and group_rewards.min() >= 1.0 - 1e-9:
                        # ANCHOR: all-success. Resolved after the loop.
                        anchor_gids.append(int(gid))
                    else:
                        # DEAD: all-fail. Deliberately NOT given the anchor
                        # treatment — a shrunk baseline would hand every
                        # episode a negative advantage, which is uniform
                        # suppression with no target to move toward.
                        self.advantages[mask] = 0.0
                        n_dead += 1
                else:
                    self.advantages[mask] = (group_rewards - mean_r) / std_r

        self._n_groups = int(len(unique_groups))
        self._resolve_anchor_groups(
            group_ids, anchor_gids, anchor_advantage, anchor_max_row_frac
        )
        # An anchor group that lost every episode to the row budget contributes
        # nothing, so it counts as dead for the live/dead accounting.
        n_dead += len(anchor_gids) - self._n_anchor_groups
        self._n_dead_groups = int(n_dead)

        return self.advantages

    def _resolve_anchor_groups(
        self,
        group_ids: np.ndarray,
        anchor_gids: list[int],
        anchor_advantage: float,
        anchor_max_row_frac: float,
    ) -> None:
        """Assign anchor advantages, honoring the anchor row budget.

        Anchor episodes are walked in index order and admitted FIRST-FIT (an
        episode that doesn't fit is skipped, and a later shorter one may still be
        admitted) until the budget (`anchor_max_row_frac` x signal chunks) is
        exhausted; the rest revert to advantage 0 and are filtered as dead
        downstream. Because anchor advantages are constant rather than zero-sum
        within the group, dropping individual episodes distorts nothing — unlike
        a signal group, where it would break the Sum(A_ep) == 0 invariant that
        _build_chunks relies on.

        The budget has an implicit floor of ONE WHOLE EPISODE: the first anchor
        episode is admitted unconditionally so a small `anchor_max_row_frac`
        shrinks the anchor share rather than silently deleting the feature. At
        ~30-65 chunks per episode that can overshoot a small budget several-fold,
        so the realized chunk count is logged whenever anything is dropped.
        """
        self._n_anchor_groups = 0
        self._n_anchor_episodes = 0
        self._n_anchor_episodes_dropped = 0
        for ep in self.episodes:
            ep.is_anchor = False
        if not anchor_gids:
            return

        anchor_set = set(anchor_gids)
        is_anchor_ep = np.array(
            [int(gid) in anchor_set for gid in group_ids], dtype=bool
        )
        # Signal chunks = chunks of non-anchor episodes with a non-zero
        # advantage. Dead episodes contribute 0 to both sides.
        n_signal_chunks = sum(
            ep.num_chunks
            for i, ep in enumerate(self.episodes)
            if not is_anchor_ep[i] and self.advantages[i] != 0.0
        )
        # No SIGNAL chunks at all: there is no denominator to measure the budget
        # against, so it does not apply. Note this is not only the all-success
        # case — an all-fail + all-success mix also has zero signal chunks while
        # carrying a non-zero std_reward, so the trainer's outer skip does not
        # fire either. Logged, because it means anchor_max_row_frac (documented as
        # the compute knob) is not bounding anything this iteration.
        if n_signal_chunks == 0:
            budget = float("inf")
            n_anchor_chunks = sum(
                ep.num_chunks for i, ep in enumerate(self.episodes) if is_anchor_ep[i]
            )
            print(
                f"  Anchor row budget WAIVED: no signal (mixed-group) chunks this "
                f"iteration, so anchor_max_row_frac={anchor_max_row_frac:g} has no "
                f"denominator — admitting all {n_anchor_chunks} anchor chunk(s)."
            )
        else:
            budget = anchor_max_row_frac * n_signal_chunks

        used = 0
        kept_gids: set[int] = set()
        for i, ep in enumerate(self.episodes):
            if not is_anchor_ep[i]:
                continue
            # `not kept_gids` keeps the first anchor episode unconditionally: a
            # budget too small for even one episode should shrink the anchor's
            # share, not silently delete the feature.
            if used + ep.num_chunks <= budget or not kept_gids:
                ep.is_anchor = True
                self.advantages[i] = anchor_advantage
                used += ep.num_chunks
                kept_gids.add(ep.group_id)
                self._n_anchor_episodes += 1
            else:
                self.advantages[i] = 0.0
                self._n_anchor_episodes_dropped += 1

        self._n_anchor_groups = len(kept_gids)
        if self._n_anchor_episodes_dropped:
            print(
                f"  Anchor row budget ({anchor_max_row_frac:g} x "
                f"{n_signal_chunks} signal chunks): kept "
                f"{self._n_anchor_episodes} anchor episode(s) / {used} chunks, "
                f"dropped {self._n_anchor_episodes_dropped} to dead."
            )

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
                    is_anchor=episode.is_anchor,
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

    def stats(self) -> dict:
        """Summary statistics for logging."""
        if not self.episodes:
            return {}

        rewards = [ep.shaped_reward for ep in self.episodes]
        num_steps_list = [ep.num_steps for ep in self.episodes]

        # Advantage summaries are computed over SIGNAL episodes only (anchor
        # episodes carry a constant positive advantage, not a group-relative
        # z-score, so pooling them would push pct_positive_advantage toward 1
        # and make the curve mean something different from one run to the next).
        # With anchors off this selects every episode, so the values are
        # bit-identical to the pre-anchor behavior.
        adv = self.advantages
        if adv is not None and len(adv) == len(self.episodes):
            signal_adv = np.array(
                [a for a, ep in zip(adv, self.episodes) if not ep.is_anchor],
                dtype=np.float64,
            )
        elif adv is not None:
            signal_adv = np.asarray(adv, dtype=np.float64)
        else:
            signal_adv = None

        # Per-group success rate distribution. Each group's rate is
        # (n_successes_in_group / n_episodes_in_group); aggregated to
        # min / median / max so TB shows the spread without histograms.
        #
        # The SAME loop also accumulates by `env_seed` — the RoboCasa reset seed,
        # i.e. the SCENE. Keyed on env_seed rather than group_id because that is
        # what the frozen scene seed pool holds fixed across iterations: with the
        # pool on, `episode/scene_sr/<seed>` is one curve per scene over the whole
        # run, which is the only view that separates "the policy improved" from
        # "this iteration drew easier scenes". group_id cannot do that job — it is
        # just a per-iteration ordinal and points at a different scene every
        # iteration when the pool is off.
        #
        # Accumulating into a dict (rather than zipping the two indexings) is also
        # correct in the degenerate case where two group_ids share one seed: their
        # episodes simply pool into one scene entry, which is the right reading of
        # "how often did the policy solve THIS scene". That case is rejected by
        # GRPOConfig's `scene_seed_pool_size >= num_groups`
        # validation for pooled runs, but a hand-driven collector can still
        # produce it and this must not double-count or drop it.
        group_to_total: dict[int, int] = {}
        group_to_succ: dict[int, int] = {}
        seed_to_total: dict[int, int] = {}
        seed_to_succ: dict[int, int] = {}
        for ep in self.episodes:
            group_to_total[ep.group_id] = group_to_total.get(ep.group_id, 0) + 1
            seed_to_total[ep.env_seed] = seed_to_total.get(ep.env_seed, 0) + 1
            if ep.success:
                group_to_succ[ep.group_id] = group_to_succ.get(ep.group_id, 0) + 1
                seed_to_succ[ep.env_seed] = seed_to_succ.get(ep.env_seed, 0) + 1
        per_group_success = [
            group_to_succ.get(gid, 0) / group_to_total[gid]
            for gid in group_to_total
        ]
        # (n_success, n_total) per scene, NOT a pre-divided rate: the caller
        # (train_grpo._log_metrics) needs the denominator to tell "0/8 this
        # iteration" from "scene absent this iteration", and a partial group
        # (collector crash mid-group) from a full one.
        per_scene_success = {
            seed: (seed_to_succ.get(seed, 0), seed_to_total[seed])
            for seed in seed_to_total
        }

        return {
            "num_episodes": self.num_episodes,
            "num_chunks": self.num_chunks,
            "success_rate": self.success_rate,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_advantage": float(signal_adv.mean()) if signal_adv is not None and signal_adv.size else 0,
            "std_advantage": float(signal_adv.std()) if signal_adv is not None and signal_adv.size else 0,
            "pct_positive_advantage": float((signal_adv > 0).mean()) if signal_adv is not None and signal_adv.size else 0,
            # Group quality (populated by compute_advantages); diagnoses how
            # much of the iter's signal got filtered out by the dead-group
            # threshold downstream.
            "n_groups": self._n_groups,
            "n_dead_groups": self._n_dead_groups,
            # SIGNAL (mixed) groups only — anchor groups are neither dead nor a
            # source of improvement gradient, so they get their own counters and
            # this keeps its original meaning across the feature flag.
            "n_live_groups": max(
                0, self._n_groups - self._n_dead_groups - self._n_anchor_groups
            ),
            # Anchor groups (all-success, admitted by include_anchor_groups).
            # n_anchor_episodes_dropped > 0 means anchor_max_row_frac bit.
            "n_anchor_groups": self._n_anchor_groups,
            "n_anchor_episodes": self._n_anchor_episodes,
            "n_anchor_episodes_dropped": self._n_anchor_episodes_dropped,
            # Trainable CHUNK counts, which is what the trainer's skip decision
            # actually depends on. Group and episode counts are not substitutes:
            # an anchor group can survive classification and still contribute
            # zero chunks (every episode budget-dropped, or zero-chunk
            # episodes), and a buffer can have zero signal chunks while
            # std_reward is non-zero (an all-fail + all-success mix).
            "n_signal_chunks": sum(
                ep.num_chunks for i, ep in enumerate(self.episodes)
                if not ep.is_anchor and adv is not None and i < len(adv)
                and adv[i] != 0.0
            ),
            "n_anchor_chunks": sum(
                ep.num_chunks for ep in self.episodes if ep.is_anchor
            ),
            # Per-group success rate spread (min/median/max across groups).
            # Reveals when the iter average masks a bimodal "some seeds at
            # 100%, others at 0%" pattern.
            "group_success_min": float(min(per_group_success)) if per_group_success else 0.0,
            "group_success_median": float(np.median(per_group_success)) if per_group_success else 0.0,
            "group_success_max": float(max(per_group_success)) if per_group_success else 0.0,
            # PER-SCENE success counts: {env_seed: (n_success, n_total)}. The
            # only NON-SCALAR entry in this dict, so every consumer that bulk-
            # dumps stats() must handle it explicitly — train_grpo._log_metrics
            # expands it into `episode/scene_sr/<seed>` scalars (only when the
            # frozen scene pool is on) and drops it from the wandb payload rather
            # than letting a dict-of-tuples reach wandb.log.
            "per_scene_success": per_scene_success,
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
            shaped_reward=0.0,
            env_name="test_env",
            episode_idx=i,
            num_steps=100,
            group_id=group_id,
            env_seed=100 + group_id * 100,
        )
        buffer.episodes.append(ep)

    advantages = buffer.compute_advantages()

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
        success=True, shaped_reward=0.0,
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

    # Test 4: Anchor groups (all-success) vs dead groups (all-fail)
    print("\nTest 4: Anchor classification")

    def _mk_buffer(outcomes: list[list[bool]], n_chunks: int = 2) -> EpisodeBuffer:
        """One group per sub-list, one episode per bool."""
        b = EpisodeBuffer()
        for gid, group in enumerate(outcomes):
            for succ in group:
                b.episodes.append(GRPOEpisode(
                    video_frames=[{}] * n_chunks, states=[{}] * n_chunks,
                    language="t", actions=[np.zeros((16, 12))] * n_chunks,
                    raw_actions=[np.zeros((50, 128))] * n_chunks,
                    action_masks=[np.ones((50, 128))] * n_chunks,
                    initial_noises=[np.zeros((50, 128))] * n_chunks,
                    success=succ, shaped_reward=0.0, env_name="t",
                    episode_idx=len(b.episodes), num_steps=100,
                    group_id=gid, env_seed=gid,
                ))
        return b

    # Group 0 all-success (anchor), group 1 all-fail (dead), group 2 mixed (signal).
    outcomes = [[True] * 4, [False] * 4, [True, True, False, False]]

    # 4a: feature OFF reproduces the pre-anchor behavior exactly.
    off = _mk_buffer(outcomes)
    adv_off = off.compute_advantages().copy()
    assert np.all(adv_off[:8] == 0.0), "all-success + all-fail must be dead when off"
    assert off.stats()["n_dead_groups"] == 2
    assert off.stats()["n_anchor_groups"] == 0
    assert not any(ep.is_anchor for ep in off.episodes)

    # 4b: feature ON with a positive advantage.
    on = _mk_buffer(outcomes)
    adv_on = on.compute_advantages(anchor_advantage=0.2, include_anchor_groups=True)
    assert np.allclose(adv_on[:4], 0.2), f"anchor group should be +0.2, got {adv_on[:4]}"
    assert np.all(adv_on[4:8] == 0.0), "all-fail must STAY dead"
    assert np.allclose(adv_on[8:], adv_off[8:]), "mixed group must be untouched"
    s = on.stats()
    assert s["n_anchor_groups"] == 1 and s["n_anchor_episodes"] == 4
    assert s["n_dead_groups"] == 1 and s["n_live_groups"] == 1
    # Advantage summaries stay signal-only, so the mixed group's zero-sum holds.
    assert abs(s["mean_advantage"]) < 1e-9, s["mean_advantage"]
    assert s["pct_positive_advantage"] == 0.25, s["pct_positive_advantage"]
    assert all(c.is_anchor for c in on._build_chunks() if c.episode_idx < 4)
    print("  PASS: anchor=+0.2, all-fail dead, mixed unchanged")

    # 4c: Layer 1 (anchor_advantage=0) still marks anchors, for the KL terms.
    l1 = _mk_buffer(outcomes)
    adv_l1 = l1.compute_advantages(anchor_advantage=0.0, include_anchor_groups=True)
    assert np.all(adv_l1[:4] == 0.0)
    assert sum(ep.is_anchor for ep in l1.episodes) == 4, "anchors must still be flagged"
    assert l1.stats()["n_anchor_groups"] == 1
    print("  PASS: KL-only anchors flagged with zero advantage")

    # 4d: row budget. Group 2 (mixed) has 4 eps x 2 chunks = 8 signal chunks;
    # frac=0.25 -> budget 2 chunks -> only the first anchor episode fits.
    bud = _mk_buffer(outcomes)
    bud.compute_advantages(
        anchor_advantage=0.2, include_anchor_groups=True, anchor_max_row_frac=0.25,
    )
    assert bud.stats()["n_anchor_episodes"] == 1, bud.stats()["n_anchor_episodes"]
    assert bud.stats()["n_anchor_episodes_dropped"] == 3
    assert sum(1 for c in bud._build_chunks() if c.is_anchor) == 2
    print("  PASS: row budget caps anchor episodes")

    # 4e: all-success iteration — no signal chunks, so the budget is waived.
    allsucc = _mk_buffer([[True] * 4, [True] * 4])
    allsucc.compute_advantages(
        anchor_advantage=0.2, include_anchor_groups=True, anchor_max_row_frac=0.1,
    )
    s = allsucc.stats()
    assert s["n_anchor_episodes"] == 8 and s["n_anchor_episodes_dropped"] == 0
    assert s["n_anchor_groups"] == 2 and s["n_dead_groups"] == 0
    assert s["std_reward"] < 1e-8, "the trainer's skip-check must be anchor-aware here"
    print("  PASS: all-success iteration keeps every anchor row")

    print("\nAll tests PASSED.")
