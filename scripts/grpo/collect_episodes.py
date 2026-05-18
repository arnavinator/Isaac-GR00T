"""Episode collector for GRPO training.

This script runs in the ROBOCASA VENV (separate from the main .venv) and:
1. Connects to the GR00T model server via ZMQ (PolicyClient).
2. Collects episodes in groups (same seed within a group).
3. Records observations, actions, initial noise tensors, and raw model outputs.
4. Saves episodes as .npz files for the training loop to consume.

Architecture:
- The MODEL runs on GPU in Terminal 1 (main .venv, grpo_server.py).
- THIS SCRIPT runs on CPU in Terminal 2 (robocasa venv, collect_episodes.py).
- Communication via ZMQ (same machine, ~0.1ms latency per call).

Group structure:
- Each group = group_size rollouts from the SAME initial state (seed).
- Within-group diversity comes from policy denoising noise, NOT env randomness.
- GRPO advantages compare rollouts WITHIN a group.

Vector env strategy (matches scripts/denoising_lab/eval/robocasa_eval_benchmark.py):
- group_size > 1 → AsyncVectorEnv (subprocess workers, parallel sim across cores).
- group_size == 1 → SyncVectorEnv (no IPC overhead).

Cross-env scene replication:
- RoboCasa picks layout/cameras/textures at env construction time using a
  per-instance RNG; those choices live in the model XML, NOT MjSimState. So
  parallel envs in the same group render different scenes even with identical
  seeds. We fix this with composite RPC methods on GroupAlignmentWrapper
  (get_scene_bundle / apply_scene_bundle), invoked across subprocess workers
  via env.call(). Recipe mirrors robocasa's playback_dataset.py:227-258 and
  the in-tree scripts/denoising_lab/eval/branching_rollout.py:399-427.

Usage:
    python scripts/grpo/collect_episodes.py \\
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --group-size 5 --num-groups 12 \\
        --output-dir /tmp/grpo_episodes/iter_001 \\
        --server-host 127.0.0.1 --server-port 5555
"""

import argparse
import time
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

from gr00t.eval.rollout_policy import get_gym_env
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy.server_client import PolicyClient

# Local module: dense reward extraction (lives next to this script).
import dense_reward


# ---------------------------------------------------------------------------
# Worker-side memory diagnostics + per-call OS-release cleanup
# ---------------------------------------------------------------------------
#
# These helpers run inside AsyncVectorEnv subprocess workers (via env.call()
# RPCs from GroupAlignmentWrapper.apply_scene_bundle). Every group, that
# method calls robosuite.reset_from_xml_string(xml), which builds a fresh
# MjModel + MjData and replaces robosuite_env.sim. The previous pair becomes
# orphaned, but several mechanisms keep it pinned: cached refs in robosuite
#'s observable wrappers and sensor handlers, MuJoCo's C-side memory pools
# (textures, meshes, contact arrays) sticky at the glibc level, and EGL/GL
# framebuffer state from the renderer. Without explicit cleanup, each call
# adds ~0.5-1 GiB of resident memory per worker; 5 calls/iter x 5 workers
# crosses the system's RAM cliff into swap thrashing.
#
# We can't fix the underlying refs without modifying robosuite, but we can
# (a) trigger Python's GC to collect anything no longer reachable and (b)
# ask glibc to return freed heap pages to the kernel. The instrumentation
# logs RSS+Swap before and after so we can quantify the per-call leak and
# the cleanup's recovery.


def _read_proc_status_mb() -> tuple[float, float]:
    """Return (rss_mb, swap_mb) for the current process. Best-effort."""
    try:
        with open("/proc/self/status") as f:
            fields = {}
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    fields[k.strip()] = v.strip()
        rss_mb = int(fields.get("VmRSS", "0 kB").split()[0]) / 1024
        swap_mb = int(fields.get("VmSwap", "0 kB").split()[0]) / 1024
        return rss_mb, swap_mb
    except Exception:
        return 0.0, 0.0


def _log_worker_mem(label: str, extra: str = "") -> None:
    """Print this worker's memory footprint with the worker's PID prefix.

    Lines are flushed immediately so they propagate through the collector's
    stdout pipe to the trainer in real time. Best-effort: any failure is
    silently swallowed since this is a non-critical diagnostic.
    """
    try:
        import os
        rss_mb, swap_mb = _read_proc_status_mb()
        suffix = f" {extra}" if extra else ""
        print(
            f"  [worker_mem pid={os.getpid()} {label}] "
            f"RSS={rss_mb:.0f}MB Swap={swap_mb:.0f}MB Total={rss_mb + swap_mb:.0f}MB{suffix}",
            flush=True,
        )
    except Exception:
        pass


def _release_worker_memory_to_os() -> None:
    """Force memory back to the OS within a worker. Mirrors the trainer's
    _release_memory_to_os pattern.

    Two gc.collect() passes: first runs any finalizers (which can produce
    new garbage), second cleans that up. malloc_trim(0) returns freed glibc
    heap pages to the kernel; without it the Python objects are gone but
    the worker's RSS stays sticky-high. Best-effort: any failure is
    silently swallowed.
    """
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        # OSError if libc.so.6 absent (musl, macOS); AttributeError if the
        # symbol is missing. Optional cleanup must never crash the worker.
        pass


# Spacing between successive group seeds within one collect() call. Wide
# enough to land on materially-different RoboCasa scenes. Must be strictly
# smaller than the trainer's per-iter seed stride (100_000 in train_grpo.py)
# divided by num_groups, or two consecutive iters' seed ranges will overlap;
# at 1000 this is safe for num_groups <= 100 (num_groups=101 collides at
# the iter boundary).
GROUP_SEED_STRIDE = 1000


# ---------------------------------------------------------------------------
# Wrapper exposing composite scene-bundle methods over env.call()
# ---------------------------------------------------------------------------


class GroupAlignmentWrapper(MultiStepWrapper):
    """MultiStepWrapper extension with composite scene-bundle RPCs.

    EpisodeCollector calls these via AsyncVectorEnv.call() to align all G
    parallel envs in a group to env 0's exact scene + dynamic state. Without
    this, RoboCasa's per-instance construction-time random choices (kitchen
    layout, camera noise, procedural textures) cause parallel envs to render
    different scenes even when seeded identically — those choices live in the
    model XML, not MjSimState.

    Recipe (mirrors robocasa's playback_dataset.py:227-258 and the in-tree
    scripts/denoising_lab/eval/branching_rollout.py:399-427):
        set_ep_meta → reset → reset_from_xml_string → set_state_from_flattened.

    Also exposes get_sim_state_flat (debug verification) and
    compute_dense_progress (used when success_weight < 1.0). Both must cross
    the IPC boundary in async mode, so the parent process can't reach into
    wrapper.unwrapped.env directly.
    """

    def get_scene_bundle(self) -> dict:
        """Snapshot the underlying env's scene + dynamic state."""
        robosuite_env = self.env.unwrapped.env
        return {
            "ep_meta": robosuite_env.get_ep_meta(),
            "model_xml": robosuite_env.sim.model.get_xml(),
            "sim_state": np.array(robosuite_env.sim.get_state().flatten()),
        }

    def apply_scene_bundle(self, bundle: dict) -> dict:
        """Adopt a reference env's scene + dynamic state.

        Returns the post-restore wrapper-stacked observation. Refreshes the
        wrapper's internal state to match MultiStepWrapper.reset() so
        subsequent step()s start from a clean slate.
        """
        base_env = self.env.unwrapped
        robosuite_env = base_env.env

        # ─── Instrumentation: entry baseline ───────────────────────────
        # XML size correlates with MuJoCo MjModel size: roughly 1 KB of XML
        # per ~50-100 KB of native MjModel memory (textures + meshes + bodies).
        # Logging it lets us correlate per-call leak size with scene complexity
        # and confirm/refute the hypothesis that seed 600067 produces heavier
        # kitchens than seed 500067.
        _xml_size_kb = len(bundle.get("model_xml", "")) // 1024
        _log_worker_mem("apply_scene_bundle entry", extra=f"xml={_xml_size_kb}KB")

        # 1. Pin metadata (newer set_ep_meta / older set_attrs_from_ep_meta).
        if hasattr(robosuite_env, "set_attrs_from_ep_meta"):
            robosuite_env.set_attrs_from_ep_meta(bundle["ep_meta"])
        elif hasattr(robosuite_env, "set_ep_meta"):
            robosuite_env.set_ep_meta(bundle["ep_meta"])

        # 2. Hard reset (required before reset_from_xml_string per
        # playback_dataset.py:239-241 — that "soft" call doesn't reload model).
        self.env.reset()

        # 3. Rebuild model from reference XML.
        xml = robosuite_env.edit_model_xml(bundle["model_xml"])
        robosuite_env.reset_from_xml_string(xml)
        robosuite_env.sim.reset()

        # ─── Fix + instrumentation: free orphaned MjModel/MjData ───────
        # reset_from_xml_string above replaced robosuite_env.sim with a fresh
        # MjSim; the previous MjModel/MjData are no longer referenced by the
        # robosuite env, but Python and glibc both hold onto the memory. The
        # cleanup here drops Python-level refs (gc) and returns freed heap
        # pages to the kernel (malloc_trim). Two log lines bracket the call
        # so the per-call delta is visible in the collector log:
        #   pre-cleanup  - peak after the new MjModel allocation
        #   post-cleanup - what the cleanup actually got back
        # The difference between successive entries' RSS values is the
        # residual leak (memory that even malloc_trim can't recover, eg.
        # MuJoCo C-side allocations or fragmented heap).
        _log_worker_mem("apply_scene_bundle post-reset (pre-cleanup)")
        _release_worker_memory_to_os()
        _log_worker_mem("apply_scene_bundle post-cleanup")

        # 4. Apply reference dynamic state (qpos/qvel/act/time).
        robosuite_env.sim.set_state_from_flattened(bundle["sim_state"])
        robosuite_env.sim.forward()
        if hasattr(robosuite_env, "update_state"):
            robosuite_env.update_state()
        elif hasattr(robosuite_env, "update_sites"):
            robosuite_env.update_sites()

        # 4b. Re-apply realistic Panda gripper params. reset_from_xml_string
        # in step 3 builds a fresh MjModel from XML, which has the default
        # robosuite gripper params (forcerange=20N, kp=1000). The realistic
        # patch (70N, kp=5000) lives in patch_panda_gripper_realism — applied
        # to the live MjModel inside RoboCasaEnv.reset() but lost during the
        # XML reload. No-op for non-Panda robots since the patch only touches
        # actuators whose name contains "gripper_finger_joint".
        try:
            from robocasa.utils.gym_utils.gymnasium_basic import (
                patch_panda_gripper_realism,
            )
            patch_panda_gripper_realism(robosuite_env)
        except ImportError:
            pass

        # 5. Read fresh obs through the standard pipeline. force_update=True
        # is load-bearing: set_state_from_flattened changes MuJoCo physics
        # but doesn't refresh robosuite's observable cache. Without force,
        # _get_observations() returns the XML-loaded observation (step 3
        # state), not the post-restore observation (step 4 state).
        raw_obs = robosuite_env._get_observations(force_update=True)
        basic_obs = base_env.get_basic_observation(raw_obs)
        groot_obs = base_env.get_groot_observation(basic_obs)

        # 6. Refresh wrapper state (mirrors MultiStepWrapper.reset()).
        self.obs = deque(
            [groot_obs] * (self.max_steps_needed + 1),
            maxlen=self.max_steps_needed + 1,
        )
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_action_steps + 1))

        return self._get_obs(self.video_delta_indices, self.state_delta_indices)

    def get_sim_state_flat(self) -> np.ndarray:
        """Read MuJoCo sim state as a flat array (used by debug verify)."""
        return np.array(self.env.unwrapped.env.sim.get_state().flatten())

    def compute_dense_progress(self, task_type: str) -> float:
        """Continuous task progress in [0, 1] (used when success_weight < 1.0).

        Wrapped here because in async mode we can't reach into the underlying
        robosuite env from the parent process — this method runs in the
        subprocess worker via env.call().
        """
        return dense_reward.compute_dense_progress(self, task_type)


# ---------------------------------------------------------------------------
# Env factory (used by AsyncVectorEnv subprocess workers)
# ---------------------------------------------------------------------------


def _make_collector_env(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    n_action_steps: int,
    max_episode_steps: int,
):
    """Build one wrapped env. Defined at module level so spawn workers can
    import it for unpickling.

    Mirrors gr00t.eval.rollout_policy.create_eval_env's structure but uses
    GroupAlignmentWrapper instead of plain MultiStepWrapper, which exposes
    the composite RPCs that EpisodeCollector needs.
    """
    env = get_gym_env(env_name, env_idx, total_n_envs)
    return GroupAlignmentWrapper(
        env,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
        terminate_on_success=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect episodes in groups for GRPO training via PolicyClient + AsyncVectorEnv"
    )
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="Full env name (e.g., robocasa_panda_omron/OpenDrawer_PandaOmron_Env)",
    )
    parser.add_argument(
        "--group-size", type=int, default=5,
        help="Rollouts per group (G). All share the same env seed; also the number of parallel envs.",
    )
    parser.add_argument(
        "--num-groups", type=int, default=12,
        help="Number of groups (different initial states) per iteration.",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=720,
        help="Maximum steps per episode before truncation.",
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=8,
        help="Number of steps to execute from each action chunk.",
    )
    parser.add_argument(
        "--server-host", type=str, default="127.0.0.1", help="GR00T model server hostname.",
    )
    parser.add_argument(
        "--server-port", type=int, default=5555, help="GR00T model server port.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save episode .npz files.",
    )
    parser.add_argument(
        "--success-weight", type=float, default=1.0,
        help="Weight for binary success in shaped reward (1.0 = pure binary + time-scaled).",
    )
    parser.add_argument(
        "--fast-forward-steps", type=int, default=0,
        help="Outer steps to fast-forward before branching (0=disabled).",
    )
    parser.add_argument(
        "--fast-forward-pct", type=float, default=0.5,
        help="Fraction of groups that use fast-forward (0.0-1.0).",
    )
    parser.add_argument(
        "--debug-fast-forward", action="store_true",
        help="Save verification images after each branch point to --output-dir/debug_ff/.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for environment initialization.",
    )
    parser.add_argument(
        "--min-successful-groups", type=int, default=0,
        help="Min groups with >=1 success before stopping. 0 = disabled "
             "(always collect exactly num_groups). When >0, collector continues "
             "past num_groups (capped at max_groups) until criterion is met.",
    )
    parser.add_argument(
        "--max-groups", type=int, default=None,
        help="Hard cap on dynamic group collection. Defaults to num_groups "
             "(no dynamic collection). Must be <= 100 (GROUP_SEED_STRIDE limit).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# EpisodeCollector
# ---------------------------------------------------------------------------


class EpisodeCollector:
    """Collects episodes using PolicyClient + AsyncVectorEnv (or Sync for G=1).

    Each episode records:
    - Per-chunk: observation frames, states, decoded actions, raw model output,
      action mask, and the initial denoising noise tensor.
    - Episode-level: success, max progress, total substeps.

    The initial noise tensor is captured by the GRPO server (grpo_server.py)
    via a hook on torch.randn. It's the ε₀ that was denoised into the action —
    used during training to evaluate the FM log-prob along the actual
    denoising path.
    """

    def __init__(
        self,
        env_name: str,
        group_size: int,
        max_episode_steps: int,
        n_action_steps: int,
        server_host: str,
        server_port: int,
        debug_fast_forward: bool = False,
        output_dir: str = "/tmp/grpo_episodes",
    ):
        self.env_name = env_name
        self.group_size = group_size
        self.n_action_steps = n_action_steps
        # max_episode_steps is exposed so callers (e.g., CollectorServer) can
        # report it back to the trainer for cross-process config validation.
        self.max_episode_steps = max_episode_steps
        self.task_type = dense_reward.classify_task_type(env_name)
        self.debug_fast_forward = debug_fast_forward
        self.output_dir = Path(output_dir)

        env_fns = [
            partial(
                _make_collector_env,
                env_name=env_name,
                env_idx=i,
                total_n_envs=group_size,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
            )
            for i in range(group_size)
        ]

        # AsyncVectorEnv (subprocess workers, parallel MuJoCo) when G > 1;
        # SyncVectorEnv when G == 1 (no IPC overhead). Pattern matches
        # scripts/denoising_lab/eval/robocasa_eval_benchmark.py:336-343.
        if group_size > 1:
            self.envs = gym.vector.AsyncVectorEnv(
                env_fns, shared_memory=False, context="spawn",
            )
            self._uses_async = True
        else:
            self.envs = gym.vector.SyncVectorEnv(env_fns)
            self._uses_async = False

        self.policy_client = PolicyClient(host=server_host, port=server_port)

        print(f"Collector initialized:")
        print(f"  Env: {env_name} (task_type: {self.task_type})")
        print(f"  Group size (parallel envs): {group_size}")
        print(f"  Vector env: {'AsyncVectorEnv' if self._uses_async else 'SyncVectorEnv'}")
        print(f"  Server: {server_host}:{server_port}")

    # ─── Outer driver ─────────────────────────────────────────────────────

    def collect(
        self,
        num_groups: int,
        base_seed: int,
        success_weight: float = 1.0,
        fast_forward_steps: int = 0,
        fast_forward_pct: float = 0.5,
        min_successful_groups: int = 0,
        max_groups: int | None = None,
    ) -> list[dict]:
        """Collect groups of self.group_size rollouts each.

        Each group consists of group_size rollouts from the SAME initial state
        (same env reset seed). Different groups have different seeds. Within a
        group, different outcomes arise from the policy's denoising noise
        (torch.randn in the action head), NOT from environmental randomness;
        GRPO advantages compare these outcomes against each other.

        Fast-forward is decided ONCE per call (per training iteration), not
        per group. With probability `fast_forward_pct`, ALL groups in this
        call use lockstep FF; otherwise NONE do. This eliminates within-
        iteration FF/non-FF mixing, which would distort cross-group reward
        comparisons (FF rollouts have shorter num_steps and so larger time-
        scaled rewards). Long-run FF fraction across iterations still
        approaches `fast_forward_pct` because each call gets a different
        `base_seed` from the trainer.

        Dynamic group collection: when min_successful_groups > 0, after
        collecting `num_groups` groups the collector keeps adding one more
        group at a time until either (a) at least `min_successful_groups`
        groups have produced at least one successful rollout, or (b)
        `max_groups` groups have been collected (warning logged).
        """
        # Default max_groups = num_groups (disables dynamic mode regardless
        # of min_successful_groups, since we can't go beyond num_groups).
        if max_groups is None:
            max_groups = num_groups

        dynamic_mode = min_successful_groups > 0 and max_groups > num_groups

        # Validation. These constraints are also enforced statically in the
        # trainer config, but a misconfigured manual CLI invocation would
        # bypass that and silently produce wrong-shaped data.
        if num_groups < 1:
            raise ValueError(
                f"num_groups must be >= 1, got {num_groups}"
            )
        if max_groups < num_groups:
            raise ValueError(
                f"max_groups ({max_groups}) must be >= num_groups ({num_groups})"
            )
        if min_successful_groups > max_groups:
            raise ValueError(
                f"min_successful_groups ({min_successful_groups}) cannot "
                f"exceed max_groups ({max_groups}) — criterion would be unsatisfiable."
            )
        # GROUP_SEED_STRIDE × max_groups must stay below the trainer's per-iter
        # seed stride (100_000; see train_grpo.py) or two consecutive iters'
        # seed ranges overlap.
        if max_groups * GROUP_SEED_STRIDE > 100_000:
            raise ValueError(
                f"max_groups={max_groups} with GROUP_SEED_STRIDE={GROUP_SEED_STRIDE} "
                f"would overflow the trainer's per-iter seed stride (100_000), "
                f"causing seed collisions across iterations."
            )

        all_episodes: list[dict] = []
        start_time = time.time()
        rng = np.random.default_rng(base_seed)

        ff_enabled = fast_forward_steps > 0 and fast_forward_pct > 0
        # One Bernoulli for the whole iteration. Drawn from a base_seed-derived
        # rng so the FF/non-FF outcome is reproducible.
        use_ff_for_iteration = ff_enabled and rng.random() < fast_forward_pct

        # Header
        if dynamic_mode:
            print(
                f"\nCollecting {num_groups}+ groups (cap {max_groups}) "
                f"× {self.group_size} rollouts each..."
            )
            print(
                f"  Dynamic: continue past {num_groups} groups until "
                f">={min_successful_groups} groups have >=1 success "
                f"(or hit cap)."
            )
        else:
            total_episodes = self.group_size * num_groups
            print(
                f"\nCollecting {num_groups} groups × {self.group_size} rollouts "
                f"= {total_episodes} episodes..."
            )
        if ff_enabled:
            if use_ff_for_iteration:
                print(f"  Fast-forward: ALL groups branch at step {fast_forward_steps}")
            else:
                print(f"  Fast-forward: enabled (pct={fast_forward_pct:.0%}) but NOT this iteration")

        successful_groups = 0
        group_idx = 0
        while True:
            # Wide GROUP_SEED_STRIDE so consecutive groups land on far-apart
            # RoboCasa scenes (closer seeds tend to produce visually similar
            # kitchens/layouts, which dampens per-iteration diversity).
            group_seed = base_seed + group_idx * GROUP_SEED_STRIDE

            if use_ff_for_iteration:
                group_episodes = self._collect_one_group_with_fast_forward(
                    group_seed=group_seed,
                    group_id=group_idx,
                    fast_forward_steps=fast_forward_steps,
                    success_weight=success_weight,
                )
                ff_label = f"(branched at step {fast_forward_steps})"
            else:
                group_episodes = self._collect_one_group(
                    group_seed=group_seed,
                    group_id=group_idx,
                    success_weight=success_weight,
                )
                ff_label = "(from seed)"

            all_episodes.extend(group_episodes)

            group_successes = sum(e["success"] for e in group_episodes)
            # "Successful group" = at least one rollout succeeded. This is
            # an approximation for "group will produce a non-zero gradient
            # signal" — the strict condition is per-group reward std > 1e-4
            # (see episode_buffer.py:compute_advantages). Edge cases where
            # the approximation differs:
            #   - All G rollouts succeed at IDENTICAL num_steps: rewards
            #     all equal → std=0 → group dead, but counted here.
            #   - All G rollouts fail with IDENTICAL max_progress (only
            #     matters when success_weight<1.0): same situation.
            # Both require zero policy-noise-induced variance on
            # success/failure timing, which is vanishingly rare on
            # non-trivial RoboCasa tasks (typical num_steps spreads over
            # 10s of substeps within a successful group). If you observe
            # dynamic mode terminating with 4 "successful" groups but
            # _grpo_update_inner reports `Filtering N/N chunks ... dead
            # groups`, switch this criterion to a per-group reward std check
            # (would require the collector to compute time-scaled rewards,
            # which it doesn't currently do).
            if group_successes > 0:
                successful_groups += 1

            group_idx += 1

            n_done = len(all_episodes)
            elapsed = time.time() - start_time
            rate = n_done / elapsed * 60 if elapsed > 0 else 0
            if dynamic_mode:
                progress_str = (
                    f"successful groups: {successful_groups}/{min_successful_groups}, "
                    f"eps: {n_done}"
                )
                count_str = f"{group_idx}/{num_groups}+"
            else:
                progress_str = f"total: {n_done}/{self.group_size * num_groups}"
                count_str = f"{group_idx}/{num_groups}"
            print(
                f"  Group {count_str} (seed={group_seed}) {ff_label}: "
                f"{group_successes}/{self.group_size} success | "
                f"{progress_str} ({rate:.0f} eps/min)"
            )

            # Stop conditions. have_signal is True when the dynamic criterion
            # is met OR when dynamic mode is disabled (in which case we just
            # need have_min_groups).
            have_min_groups = group_idx >= num_groups
            have_signal = (
                not dynamic_mode
                or successful_groups >= min_successful_groups
            )
            at_max_cap = group_idx >= max_groups

            if have_min_groups and have_signal:
                break
            if at_max_cap:
                if not have_signal:
                    print(
                        f"  WARNING: hit max_groups={max_groups} cap with only "
                        f"{successful_groups}/{min_successful_groups} successful "
                        f"groups — proceeding with what was collected."
                    )
                break

        elapsed = time.time() - start_time
        successes = sum(e["success"] for e in all_episodes)
        total_eps = len(all_episodes)
        rate = total_eps / elapsed * 60 if elapsed > 0 else 0
        print(
            f"\nCollection complete: {total_eps} episodes from {group_idx} "
            f"groups in {elapsed:.1f}s ({rate:.0f} eps/min)"
        )
        # Defensive: total_eps is normally >= group_size after at least one
        # group, but a worker death producing an empty group_episodes list
        # would make this 0. Guard the division.
        success_pct = 100 * successes / total_eps if total_eps > 0 else 0.0
        print(
            f"Success rate: {successes}/{total_eps} "
            f"({success_pct:.0f}% episodes), "
            f"successful groups: {successful_groups}/{group_idx}"
        )
        # Only report dense progress when it actually contributed to the shaped
        # reward — with success_weight=1.0 (the default) the progress term is
        # multiplied by (1 - success_weight) = 0, so max_progress stays at the
        # per-episode init value of 0.0 and reporting it would be misleading.
        if success_weight < 1.0:
            print(f"Mean progress: {np.mean([e['max_progress'] for e in all_episodes]):.3f}")

        return all_episodes

    # ─── Per-group entry points ───────────────────────────────────────────

    def _collect_one_group(
        self,
        group_seed: int,
        group_id: int,
        success_weight: float,
    ) -> list[dict]:
        """Collect one group with all envs at the seed-aligned starting state."""
        observations = self._align_envs_to_group_scene(group_seed)
        return self._run_per_env_loop(
            observations, group_id, group_seed, success_weight
        )

    def _collect_one_group_with_fast_forward(
        self,
        group_seed: int,
        group_id: int,
        fast_forward_steps: int,
        success_weight: float,
    ) -> list[dict]:
        """Collect one group with a lockstep fast-forward prefix.

        All G envs are first aligned (same scene + dynamic state), then
        stepped in lockstep with env 0's action chunk for fast_forward_steps
        outer steps. After the prefix, envs continue independently and
        within-group diversity arises from the policy's denoising noise on
        each post-branch chunk.

        Falls back to _collect_one_group if any env terminates during FF.

        We deliberately do NOT count the FF prefix in episodes[i].num_steps:
        time-scaled rewards within a group should compare post-branch effort
        fairly. Mixing FF and non-FF groups (fast_forward_pct < 1.0) means
        cross-group comparisons (e.g., logged mean_reward) will favor branched
        groups numerically — a known artifact of FF; group-relative advantages
        are unaffected since they normalize WITHIN each group.
        """
        observations = self._align_envs_to_group_scene(group_seed)

        for step in range(fast_forward_steps):
            new_obs = self._lockstep_step(observations)
            if new_obs is None:
                print(
                    f"    Env terminated at step {step} during fast-forward, "
                    "falling back to normal collection"
                )
                return self._collect_one_group(
                    group_seed, group_id, success_weight
                )
            observations = new_obs

        if self.debug_fast_forward:
            self._verify_branch_point(
                group_id, group_seed, fast_forward_steps, observations
            )

        return self._run_per_env_loop(
            observations, group_id, group_seed, success_weight
        )

    # ─── Vector env primitives ────────────────────────────────────────────

    def _align_envs_to_group_scene(self, group_seed: int) -> list[dict]:
        """Reset all G envs to env 0's scene + dynamic state.

        Each subprocess env runs in its own MuJoCo instance with its own
        construction-time random scene. We can't change those choices from
        the parent process, so we use AsyncVectorEnv.call() to invoke the
        composite RPCs on GroupAlignmentWrapper inside each worker:
          1. Reset all envs with seed=group_seed (each ends up with its own
             scene because the RNG choices were baked at construction).
          2. get_scene_bundle on all envs (capture env 0's bundle).
          3. apply_scene_bundle(env0_bundle) on all envs — envs 1..G-1 align
             to env 0; env 0 re-applies its own (a no-op effect, but it runs
             in parallel with the other workers so no wall-time cost).

        Returns wrapper-stacked observations, one per env. Bit-identical
        across the group when this returns (verifiable via
        --debug-fast-forward).
        """
        seeds = [group_seed] * self.group_size
        vector_obs, _ = self.envs.reset(seed=seeds)

        if self.group_size == 1:
            # Nothing to align to — just return the lone env's obs.
            return self._unbatch_vector_obs(vector_obs)

        bundles = self.envs.call("get_scene_bundle")
        obs_tuple = self.envs.call("apply_scene_bundle", bundles[0])
        return list(obs_tuple)

    def _lockstep_step(
        self, observations_per_env: list[dict]
    ) -> list[dict] | None:
        """One outer step where every env steps with env 0's action chunk.

        Returns the new per-env observations, or None if any env terminated
        (signaling _collect_one_group_with_fast_forward to fall back).
        """
        # All G envs are bit-identical post-alignment, so env 0's obs alone
        # is enough for the server query.
        batched = self._batch_per_env_obs([observations_per_env[0]])
        actions, _, _, _ = self._get_actions_from_server(batched)
        actions_full = self._broadcast_actions(actions, self.group_size)

        next_obs, _, terms, truncs, _ = self.envs.step(actions_full)
        if any(terms) or any(truncs):
            return None
        return self._unbatch_vector_obs(next_obs)

    def _run_per_env_loop(
        self,
        observations_per_env: list[dict],
        group_id: int,
        group_seed: int,
        success_weight: float,
    ) -> list[dict]:
        """Step every env until it finishes, recording per-chunk data.

        Each outer step:
          - Batches active envs' obs and queries the policy server.
          - Builds a [G, T, dim] action tensor: real chunks for active envs,
            zeros for already-done envs (gymnasium auto-resets done envs in
            the background; we ignore the auto-reset obs by filtering on
            active_indices).
          - Calls vector env.step (parallel MuJoCo across subprocess workers).
          - Reads terminal info via final_info handling so autoreset doesn't
            clobber the terminating chunk's substep dones / success flag
            (pattern from robocasa_eval_benchmark.py:393-402).

        Episode num_steps starts at 0 in both normal and post-FF modes.
        """
        episodes = [
            self._new_episode(group_id, group_seed)
            for _ in range(self.group_size)
        ]
        done_flags = [False] * self.group_size

        while not all(done_flags):
            active_indices = [i for i, d in enumerate(done_flags) if not d]
            active_obs = [observations_per_env[i] for i in active_indices]
            batched = self._batch_per_env_obs(active_obs)

            actions_active, initial_noise, raw_actions, action_masks = (
                self._get_actions_from_server(batched)
            )
            actions_full = self._scatter_actions(actions_active, active_indices)

            # Optional dense progress (per-env RPC; only when needed).
            progresses = (
                self.envs.call("compute_dense_progress", self.task_type)
                if success_weight < 1.0
                else None
            )

            next_obs, _, terms, truncs, infos = self.envs.step(actions_full)

            for j, env_idx in enumerate(active_indices):
                ep = episodes[env_idx]
                obs = observations_per_env[env_idx]
                action_j = self._extract_per_env(actions_active, j)

                ep["video_frames"].append(self._extract_video_single(obs))
                ep["states"].append(self._extract_state_single(obs))
                ep["actions"].append(action_j)
                ep["raw_actions"].append(
                    raw_actions[j] if raw_actions is not None else None
                )
                ep["action_masks"].append(
                    action_masks[j] if action_masks is not None else None
                )
                ep["initial_noises"].append(
                    initial_noise[j] if initial_noise is not None else None
                )
                if ep["language"] is None:
                    ep["language"] = self._extract_language_single(obs)

                # H3 fix: count actual substeps (MultiStepWrapper.step()
                # early-breaks on termination, so the last chunk may run
                # fewer than n_action_steps substeps).
                env_dones = self._info_for_env(infos, "dones", env_idx)
                if env_dones is not None and hasattr(env_dones, "__len__"):
                    ep["num_steps"] += len(env_dones)
                else:
                    ep["num_steps"] += self.n_action_steps

                if progresses is not None:
                    ep["max_progress"] = max(
                        ep["max_progress"], progresses[env_idx]
                    )

                if terms[env_idx] or truncs[env_idx]:
                    done_flags[env_idx] = True
                    ep["success"] = self._success_for_env(infos, env_idx)
                    ep["shaped_reward"] = dense_reward.compute_shaped_reward(
                        ep["success"], ep["max_progress"], success_weight
                    )

            # Update obs only for envs we'll step again next iteration. Done
            # envs auto-reset in the background; we never consume their fresh
            # obs since they're filtered out via active_indices.
            for env_idx in active_indices:
                if not done_flags[env_idx]:
                    observations_per_env[env_idx] = self._extract_per_env(
                        next_obs, env_idx
                    )

        return episodes

    # ─── Debug verification ───────────────────────────────────────────────

    def _verify_branch_point(
        self,
        group_id: int,
        group_seed: int,
        ff_steps: int,
        observations_per_env: list[dict],
    ) -> None:
        """Confirm all envs are at an identical state after lockstep FF.

        With deterministic MuJoCo, the G envs see the same (state, action)
        pairs throughout the FF loop, so they should be bit-identical here.
        This routine fails loudly if upstream introduces step-time
        non-determinism (e.g., domain randomization) which would silently
        break the lockstep assumption.

        Saves a montage image showing camera views from all envs side-by-side,
        plus a numerical comparison of sim states and the wrapper-level
        observations the policy actually sees.

        Output: {output_dir}/debug_ff/group{group_id:03d}_seed{group_seed}_ff{ff_steps}.png

        What to look for:
        - All camera views in the montage should be pixel-identical.
        - sim_state max diffs should be 0.0 (or <1e-10 for float precision).
        - obs max diffs should be 0.0 for state keys, ~0 for video keys.
        """
        debug_dir = self.output_dir / "debug_ff"
        debug_dir.mkdir(parents=True, exist_ok=True)

        print(f"    [DEBUG] Verifying branch point for group {group_id} (seed={group_seed})...")

        # 1. Compare sim states numerically (RPC across all workers).
        sim_states = list(self.envs.call("get_sim_state_flat"))
        ref = sim_states[0]
        max_diffs = [np.abs(s - ref).max() for s in sim_states[1:]]
        print(f"    [DEBUG] sim_state max diffs vs env 0: {max_diffs}")
        if all(d < 1e-10 for d in max_diffs):
            print(f"    [DEBUG] PASS: all sim states identical")
        else:
            print(f"    [DEBUG] FAIL: sim states differ!")

        # 2. Compare wrapper observations (what the policy actually sees).
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

        # 3. Render camera views from all envs and save as a montage.
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
                    all_frames.append(np.concatenate(env_frames, axis=1))

            if all_frames:
                montage = np.concatenate(all_frames, axis=0)
                out_path = debug_dir / f"group{group_id:03d}_seed{group_seed}_ff{ff_steps}.png"
                try:
                    import cv2
                    cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
                    print(f"    [DEBUG] Montage saved: {out_path}")
                    print(f"    [DEBUG] Layout: {self.group_size} rows (envs) × {len(env_frames)} cols (cameras)")
                    print(f"    [DEBUG] All rows should look identical if lockstep is correct")
                except ImportError:
                    npy_path = out_path.with_suffix(".npy")
                    np.save(str(npy_path), montage)
                    print(f"    [DEBUG] Montage saved as numpy: {npy_path} (install cv2 for PNG)")
        except Exception as e:
            print(f"    [DEBUG] Could not render montage: {e}")

    # ─── Small helpers ────────────────────────────────────────────────────

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
        """Query the GRPO server.

        Returns (actions, initial_noise, raw_actions, action_masks). The GRPO
        server captures the initial noise tensor, raw normalized action, and
        per-embodiment action mask from the denoising process and returns them
        in the info dict; for vanilla PolicyClient those are None.
        """
        result = self.policy_client.get_action(observations)
        if isinstance(result, tuple) and len(result) == 2:
            action_dict, info = result
        else:
            action_dict, info = result, {}
        return (
            action_dict,
            info.get("initial_noise"),
            info.get("raw_actions"),
            info.get("action_mask"),
        )

    def _extract_per_env(self, data, env_idx: int):
        """Extract per-env data from a batched dict-of-arrays or array.

        Strings are preserved as bare Python strings — np.array("str") would
        produce a 0-dim numpy string array that _batch_per_env_obs would
        mistake for an ndarray and stack into a numpy string array instead
        of building the tuple-of-strings format the policy server expects
        for Text observations (annotation.human...).
        """
        def _one(v):
            if not hasattr(v, "__getitem__"):
                return v
            item = v[env_idx]
            return item if isinstance(item, str) else np.array(item)

        if isinstance(data, dict):
            return {k: _one(v) for k, v in data.items()}
        if hasattr(data, "__getitem__"):
            item = data[env_idx]
            return item if isinstance(item, str) else np.array(item)
        return data

    def _batch_per_env_obs(self, obs_list: list[dict]) -> dict:
        """Stack a list of single-env obs dicts into a vectorized dict.

        Output format matches what gym.vector envs would produce: ndarrays
        gain a leading batch axis, language strings become a tuple per env.
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

    def _unbatch_vector_obs(self, vector_obs: dict) -> list[dict]:
        """Convert vector env's batched dict to a per-env list of obs dicts."""
        return [self._extract_per_env(vector_obs, i) for i in range(self.group_size)]

    def _broadcast_actions(self, actions_one: dict, dst_size: int) -> dict:
        """Broadcast a [1, T, dim] action dict to [dst_size, T, dim].

        Used by lockstep FF, where every env steps with env 0's chunk.
        """
        out = {}
        for k, v in actions_one.items():
            tiled = np.broadcast_to(v, (dst_size,) + v.shape[1:])
            out[k] = np.array(tiled)  # copy so the array is writable
        return out

    def _scatter_actions(
        self, actions_active: dict, active_indices: list[int]
    ) -> dict:
        """Build [G, T, dim] from [num_active, T, dim], padding done envs with zeros.

        Done envs receive a zero action; gymnasium auto-resets them in the
        background and the dummy step's data is ignored by the caller (filtered
        out via active_indices). The per-env wasted compute is bounded by the
        slowest env's remaining episode length, which is typically the
        bottleneck anyway.
        """
        out = {}
        for k, v in actions_active.items():
            full = np.zeros((self.group_size,) + v.shape[1:], dtype=v.dtype)
            for j, env_idx in enumerate(active_indices):
                full[env_idx] = v[j]
            out[k] = full
        return out

    def _info_for_env(self, infos: dict, key: str, env_idx: int):
        """Read info[key][env_idx], preferring final_info on a terminating step.

        Under gymnasium's autoreset, infos[key] for a terminated env reflects
        the auto-reset episode's state, NOT the terminal episode's. The
        terminal info is preserved in infos["final_info"][env_idx]. Pattern
        from scripts/denoising_lab/eval/robocasa_eval_benchmark.py:393-402.
        """
        if "final_info" in infos and infos["final_info"][env_idx] is not None:
            return infos["final_info"][env_idx].get(key)
        if key in infos and infos[key] is not None:
            return infos[key][env_idx]
        return None

    def _success_for_env(self, infos: dict, env_idx: int) -> bool:
        """Reduce one env's MultiStepWrapper info["success"] to a bool.

        MultiStepWrapper packs per-substep success bools into info["success"]
        as an array of length up to n_action_steps; we treat the chunk as
        successful if ANY substep succeeded — same convention as
        gr00t/eval/rollout_policy.py:303-313.
        """
        s = self._info_for_env(infos, "success", env_idx)
        if s is None:
            return False
        if isinstance(s, (list, np.ndarray)):
            return bool(np.any(s))
        return bool(s)

    def _extract_video_single(self, obs: dict) -> dict:
        """Extract video frames from a single-env obs dict.

        Strips the 'video.' prefix to match VLAStepData/processor expectations.
        """
        frames = {}
        if isinstance(obs, dict):
            for key, value in obs.items():
                if "image" in key or "video" in key:
                    clean_key = key.removeprefix("video.")
                    frames[clean_key] = np.array(value)
        return frames

    def _extract_state_single(self, obs: dict) -> dict:
        """Extract state values from a single-env obs dict.

        Strips the 'state.' prefix and filters annotation/language keys.
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
        """Extract task language instruction from a single-env obs dict."""
        if isinstance(obs, dict):
            for key, value in obs.items():
                if "language" in key or "annotation" in key or "task_description" in key:
                    if isinstance(value, (tuple, list)) and len(value) > 0:
                        return str(value[0])
                    if isinstance(value, str):
                        return value
        return self.env_name.split("/")[-1]  # Fallback to env name

    def close(self):
        """Clean up environments."""
        self.envs.close()


# ---------------------------------------------------------------------------
# Save episodes to disk
# ---------------------------------------------------------------------------


def save_episodes(episodes: list[dict], output_dir: str) -> None:
    """Save collected episodes as .npz files.

    Each episode is saved as episode_{idx}.npz with keys structured for
    EpisodeBuffer.load_episodes() to parse.
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

        for chunk_idx in range(len(ep["actions"])):
            if chunk_idx < len(ep["video_frames"]):
                for cam_name, frame in ep["video_frames"][chunk_idx].items():
                    save_dict[f"video_{cam_name}_{chunk_idx}"] = frame

            if chunk_idx < len(ep["states"]):
                for state_key, state_val in ep["states"][chunk_idx].items():
                    save_dict[f"state_{state_key}_{chunk_idx}"] = state_val

            save_dict[f"action_{chunk_idx}"] = ep["actions"][chunk_idx]

            # Raw normalized action (50x128 tensor from model, for FM log-prob).
            if chunk_idx < len(ep.get("raw_actions", [])):
                raw_action = ep["raw_actions"][chunk_idx]
                if raw_action is not None:
                    save_dict[f"raw_action_{chunk_idx}"] = raw_action

            # Action mask from model (proper per-embodiment mask, not all-ones).
            # Hard-fail if missing: the GRPO server always attaches a per-
            # embodiment mask (grpo_server.py:252-256, with compute_action_mask
            # raising on derivation failure). A None mask reaching here means
            # the upstream contract broke (vanilla PolicyServer instead of
            # GRPOPolicyWrapper, or capture failure). Falling back to all-ones
            # would silently average FM-MSE over the padded region (e.g., 6400
            # elements instead of 192 valid for PandaOmron — ~33× underestimate),
            # which corrupts the log-prob signal without any visible error.
            if chunk_idx >= len(ep.get("action_masks", [])):
                raise RuntimeError(
                    f"Episode {idx} chunk {chunk_idx}: action_masks list is "
                    f"shorter than actions list (len={len(ep.get('action_masks', []))} "
                    f"vs {len(ep['actions'])}). The collector is misconfigured."
                )
            mask = ep["action_masks"][chunk_idx]
            if mask is None:
                raise RuntimeError(
                    f"Episode {idx} chunk {chunk_idx}: action_mask is None. "
                    f"The GRPO server (grpo_server.py) must attach a per-"
                    f"embodiment mask to info['action_mask']. Check that the "
                    f"server is wrapped in GRPOPolicyWrapper (not vanilla "
                    f"PolicyServer) and that compute_action_mask succeeded at "
                    f"server startup."
                )
            save_dict[f"action_mask_{chunk_idx}"] = mask

            # Initial noise tensor (50x128) — the ε₀ that was denoised into this action.
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
    np.random.seed(args.seed)

    collector = EpisodeCollector(
        env_name=args.env_name,
        group_size=args.group_size,
        max_episode_steps=args.max_episode_steps,
        n_action_steps=args.n_action_steps,
        server_host=args.server_host,
        server_port=args.server_port,
        debug_fast_forward=args.debug_fast_forward,
        output_dir=args.output_dir,
    )

    try:
        episodes = collector.collect(
            num_groups=args.num_groups,
            base_seed=args.seed,
            success_weight=args.success_weight,
            fast_forward_steps=args.fast_forward_steps,
            fast_forward_pct=args.fast_forward_pct,
            min_successful_groups=args.min_successful_groups,
            max_groups=args.max_groups,
        )
        save_episodes(episodes, args.output_dir)
    finally:
        collector.close()


if __name__ == "__main__":
    main()
