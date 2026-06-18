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
import copy
import json
import os
import time
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

# ─── Import-time noise suppression ───────────────────────────────────────
# Driven by the GRPO_CLEAN_OUTPUT=1 env var (set by train_grpo.py when
# config.clean_output is True). MUST run BEFORE the gymnasium / robocasa
# imports below, since several of these noises fire at import time:
#   - robosuite's logger emits [WARNING]/[INFO] from macros.py + __init__.py
#   - robocasa/__init__.py:294 has a bare `print(...)` for the mimicgen miss
#   - gymnasium.utils.passive_env_checker UserWarning fires per reset/step
# AsyncVectorEnv subprocess workers re-import this module on spawn and
# inherit GRPO_CLEAN_OUTPUT, so suppression applies to them too.
_CLEAN_OUTPUT = os.environ.get("GRPO_CLEAN_OUTPUT") == "1"
if _CLEAN_OUTPUT:
    import logging
    import warnings

    # Robosuite [WARNING]/[INFO] suppression. Two non-obvious facts about
    # robosuite's logger force the filter approach below:
    #   1. The actual logger name is "robosuite_logs", NOT "robosuite". See
    #      robosuite/utils/log_utils.py:63 (DefaultLogger logger_name kwarg
    #      default) and the module-level
    #      `ROBOSUITE_DEFAULT_LOGGER = DefaultLogger(...).get_logger()`.
    #   2. log_utils.py:89 calls `logger.setLevel(INFO)` when robosuite is
    #      imported (via DefaultLogger.__init__). A pre-import setLevel(ERROR)
    #      on the logger is therefore CLOBBERED to INFO before the first
    #      [robosuite WARNING] is even emitted. The level dial is unusable
    #      from outside without monkey-patching.
    # An attached Filter survives setLevel changes (Logger.handle checks
    # filter() and isEnabledFor() independently), so we pre-create the
    # logger by name and attach a filter that drops anything below ERROR.
    # When robosuite later calls getLogger("robosuite_logs") at log_utils
    # line 71/97, it gets THIS same logger instance and the filter persists.
    # The filter runs in Logger.handle() BEFORE callHandlers dispatches to
    # the StreamHandler, so the handler's stream destination (default stderr,
    # which redirect_stdout doesn't reach) is irrelevant — records are
    # dropped before they ever get written.
    # Catches: import-time WARNINGs (robosuite/macros.py "No private macro
    # file" trio, robosuite/__init__.py "Could not import robosuite_models"
    # and "Could not load mink-based whole-body IK", robocasa/macros.py's
    # mirror of the macro warnings via the same shared ROBOSUITE_DEFAULT_LOGGER)
    # AND runtime [robosuite INFO] (per-rollout "Loading controller
    # configuration from..." at composite_controller_factory.py:121).
    class _DropRobosuiteBelowError(logging.Filter):
        def filter(self, record):  # noqa: A003 — must match Filter API
            return record.levelno >= logging.ERROR
    logging.getLogger("robosuite_logs").addFilter(_DropRobosuiteBelowError())

    # gymnasium.utils.passive_env_checker UserWarning: "obs not within the
    # observation space" — fires per reset()/step() at runtime via
    # warnings.warn(). filterwarnings adds to the global filter list; the
    # warnings module checks it at warn() call time, so timing here is fine.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"gymnasium\.utils\.passive_env_checker",
    )

import gymnasium as gym
import numpy as np

# The mimicgen warning is a bare print() in robocasa/__init__.py — neither
# logging level nor warnings.filterwarnings reaches it. Redirect stdout for
# the import that triggers the robocasa side. Stderr is untouched, so real
# ImportErrors still surface.
#
# Important: rollout_policy lazy-imports robocasa inside its env_fn (see
# gr00t/eval/rollout_policy.py:79-90 — `import robocasa` lives in the inner
# `env_fn` body, fired only when env_fn() is called by gym.make). Just
# wrapping `from gr00t.eval.rollout_policy import ...` doesn't catch the
# print, because robocasa hasn't been touched yet at that point. We force-
# import robocasa inside the redirect block so its module-level mimicgen
# `print(...)` lands inside the swallow. Subsequent lazy imports (in env_fn
# AND in AsyncVectorEnv worker spawns, which inherit the cache via fresh
# import via env-var-driven re-run of this block) hit sys.modules and stay
# silent. Anything else printed at import time is expected to be silent in
# this codebase; if a future robocasa version adds a useful import-time
# print, set --no-clean-output to see it again.
if _CLEAN_OUTPUT:
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        from gr00t.eval.rollout_policy import get_gym_env
        try:
            import robocasa  # noqa: F401  -- eager to capture mimicgen print
        except ImportError:
            # collect_episodes.py runs in the robocasa venv where robocasa
            # is installed; if it isn't, suppression is best-effort and the
            # downstream env construction will surface the real ImportError
            # via the normal traceback path on stderr.
            pass
else:
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

    No-op when GRPO_CLEAN_OUTPUT=1 (config.clean_output=True) — the user
    opted out of memory diagnostics.
    """
    if _CLEAN_OUTPUT:
        return
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


def _parse_step_from_filename(npz_path: str) -> int | None:
    """Extract the outer-step index from `ep*_step*.npz`-style filenames.

    Mirrors scripts/denoising_lab/eval/branching_rollout.py:_parse_step_from_filename
    so an npz that lacks __step_info__ can still surface its branch step via
    the filename convention used by interactive_rollout.py.
    """
    import re
    m = re.search(r"step(\d+)", Path(npz_path).stem)
    return int(m.group(1)) if m else None


def _load_init_bundle(npz_path: str) -> dict:
    """Load a saved sim-state npz into the dict shape apply_scene_bundle expects.

    The npz contract (set by scripts/denoising_lab/eval/interactive_rollout.py
    and consumed by scripts/denoising_lab/eval/branching_rollout.py:182-210)
    uses double-underscore keys:
        __sim_state__   : np.ndarray, MuJoCo MjSimState flat
        __model_xml__   : str, the scene XML
        __ep_meta__     : str, JSON-serialized robosuite ep_meta dict
        __step_info__   : str (optional), JSON dict with keys `step` and
                          `n_action_steps` for the outer step the npz was
                          saved at — used to bill consumed_substeps against
                          MultiStepWrapper.max_episode_steps so the post-
                          restore rollout has only the REMAINING budget, not
                          a fresh full budget. Mirrors branching_rollout.py
                          Phase 3 wrapper-sync (lines 488-505).

    apply_scene_bundle (this file, GroupAlignmentWrapper) expects:
        sim_state, model_xml, ep_meta (as native dict, not JSON string),
        consumed_substeps (int, optional — defaults to 0 if not provided,
        which means "restored env starts with a fresh full budget"; this
        default preserves the existing fast-forward branch behavior).

    This helper does the key rename + json.loads + type validation in one
    place so the override path in _align_envs_to_group_scene stays a single
    line. All failure modes raise ValueError with the offending path quoted,
    so the user can grep their training log for the path that broke.
    """
    if not npz_path:
        # Defensive — GRPOConfig.__post_init__ rejects empty strings, but
        # collect_episodes.py is also runnable standalone via CLI.
        raise ValueError("init_state_npz_path is empty; expected a path string")
    data = dict(np.load(npz_path, allow_pickle=True))
    if "__sim_state__" not in data:
        raise ValueError(
            f"{npz_path}: missing __sim_state__. "
            "Re-save using scripts/denoising_lab/eval/interactive_rollout.py "
            "(it writes __sim_state__, __model_xml__, __ep_meta__)."
        )
    if "__model_xml__" not in data or "__ep_meta__" not in data:
        raise ValueError(
            f"{npz_path}: missing __model_xml__ and/or __ep_meta__ — "
            "apply_scene_bundle needs the full triple to restore the scene "
            "deterministically (XML rebuild + ep_meta layout pinning)."
        )
    # json.loads may raise JSONDecodeError if the npz was hand-edited or
    # serialized by a saver that doesn't follow the contract. Wrap with the
    # path so the user knows which file is broken.
    try:
        ep_meta = json.loads(str(data["__ep_meta__"]))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{npz_path}: __ep_meta__ is not valid JSON ({e}). "
            "Expected a JSON-serialized robosuite ep_meta dict; "
            "see interactive_rollout.py for the canonical save format."
        ) from e
    # apply_scene_bundle passes ep_meta to robosuite_env.set_ep_meta which
    # expects a dict (it indexes by key). A JSON scalar or list would either
    # crash deep in robosuite or silently store wrong state.
    if not isinstance(ep_meta, dict):
        raise ValueError(
            f"{npz_path}: __ep_meta__ JSON-decoded to {type(ep_meta).__name__}, "
            "expected dict (e.g., {'layout_id': ..., 'style_id': ..., ...})."
        )

    # Extract consumed_substeps from __step_info__ (canonical) or filename
    # (fallback). consumed_substeps = branch_step * n_action_steps tells
    # apply_scene_bundle how many sub-steps have ALREADY been spent against
    # the wrapper's max_episode_steps budget, so the post-restore rollout
    # truncates at the correct remaining horizon. If neither source resolves
    # both pieces of info, default to 0 (a fresh full budget — old behavior)
    # and warn so the user can fix the npz.
    consumed_substeps = 0
    branch_step: int | None = None
    saved_n_action_steps: int | None = None
    if "__step_info__" in data:
        try:
            step_info = json.loads(str(data["__step_info__"]))
            if isinstance(step_info, dict):
                branch_step = step_info.get("step")
                saved_n_action_steps = step_info.get("n_action_steps")
        except json.JSONDecodeError:
            # Bad __step_info__ JSON is non-fatal — fall through to filename
            # parsing. The user already gets a clear error if __ep_meta__ is
            # the broken JSON (above).
            pass
    if branch_step is None:
        branch_step = _parse_step_from_filename(npz_path)
    if branch_step is not None and saved_n_action_steps is not None:
        consumed_substeps = int(branch_step) * int(saved_n_action_steps)
        # Negative consumed_substeps would silently no-op the pre-fill
        # (`[0.0] * -40` is `[]` in Python). Catch it loudly here — typically
        # caused by a hand-edited __step_info__ with a negative `step` or
        # `n_action_steps`.
        if consumed_substeps < 0:
            raise ValueError(
                f"{npz_path}: __step_info__ produced consumed_substeps="
                f"{consumed_substeps} (negative). branch_step={branch_step}, "
                f"n_action_steps={saved_n_action_steps}; both must be >= 0."
            )
    elif branch_step is not None:
        # We know step but not n_action_steps. Warn rather than assume — a
        # wrong assumption silently breaks budget accounting in either
        # direction.
        import warnings
        warnings.warn(
            f"{npz_path}: __step_info__ provides branch_step={branch_step} but "
            f"not n_action_steps — cannot compute consumed_substeps. "
            f"Restored env will start with a fresh max_episode_steps budget "
            f"instead of the remaining budget after step {branch_step}.",
            stacklevel=3,
        )
    else:
        # Neither source resolved. This is normal for hand-built npzs but
        # surprising for interactive_rollout.py output. Warn once per load.
        import warnings
        warnings.warn(
            f"{npz_path}: no __step_info__ key and filename doesn't match "
            f"ep*_step*.npz — cannot bill any consumed sub-steps against "
            f"max_episode_steps. Restored env will start with a fresh full "
            f"budget. If this npz represents an intermediate trajectory step, "
            f"re-save with interactive_rollout.py to get correct budget accounting.",
            stacklevel=3,
        )

    return {
        "ep_meta": ep_meta,
        "model_xml": str(data["__model_xml__"]),
        "sim_state": np.asarray(data["__sim_state__"]),
        "consumed_substeps": consumed_substeps,
    }


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
        # Pre-fill reward/done with `consumed_substeps` placeholders so the
        # wrapper's truncation check (len(self.reward) >= max_episode_steps;
        # see multistep_wrapper.py:271-275) bills sub-steps that have already
        # elapsed in the saved trajectory against the budget. The post-restore
        # rollout then truncates after `max_episode_steps - consumed_substeps`
        # more sub-steps, matching Phase 3 of branching_rollout.py:488-505.
        #
        # consumed_substeps defaults to 0 when absent from the bundle. This
        # preserves the existing fast-forward branch behavior, where the FF
        # prefix runs INSIDE the wrapper and naturally grows self.reward —
        # no pre-fill needed there.
        #
        # Pre-filling with 0.0 / False is safe under MultiStepWrapper's
        # default reward_agg_method="max" (line 129) and done aggregator
        # "max" (line 281): zeros don't change max for non-negative rewards,
        # and False doesn't suppress True. info uses dict_take_last_n
        # (line 282), so it's windowed and unaffected.
        consumed_substeps = int(bundle.get("consumed_substeps", 0))
        # Sanity-check the pre-fill against the wrapper's budget — if the
        # saved trajectory's elapsed sub-steps already exhausted (or exceeded)
        # max_episode_steps, the very first sub-step after restore will fire
        # the truncation check (len(self.reward) >= max_episode_steps) and
        # the episode terminates with ~zero data, producing useless training
        # signal. Surface this loudly so the user can pick a less-advanced
        # branch point or raise max_episode_steps.
        if (
            consumed_substeps > 0
            and self.max_episode_steps is not None
            and consumed_substeps >= self.max_episode_steps
        ):
            import warnings
            warnings.warn(
                f"consumed_substeps={consumed_substeps} >= max_episode_steps="
                f"{self.max_episode_steps}; post-restore rollout will truncate "
                f"after a single sub-step. Either pick an npz from earlier in "
                f"the trajectory or raise --max-episode-steps.",
                stacklevel=2,
            )
        # The pre-fill is safe under reward_agg_method="max" because zeros
        # don't change the max of non-negative rewards. Under "sum" it's also
        # safe (zeros add nothing). Under "mean" it WOULD dilute (80 zeros
        # plus 8 real rewards of 0.5 → mean ~0.045 instead of 0.5). Warn so
        # any future change to the default aggregation method surfaces here.
        if consumed_substeps > 0 and getattr(
            self, "reward_agg_method", "max"
        ) not in ("max", "sum"):
            import warnings
            warnings.warn(
                f"apply_scene_bundle pre-fills self.reward with zeros, which "
                f"is safe under reward_agg_method in ('max', 'sum') but "
                f"dilutes results under reward_agg_method="
                f"{self.reward_agg_method!r}. The pre-fill matches "
                f"branching_rollout.py's pattern; if you genuinely need a "
                f"non-max aggregation, you'll need to skip the pre-fill "
                f"or compute the aggregate over only the post-restore tail.",
                stacklevel=2,
            )
        self.obs = deque(
            [groot_obs] * (self.max_steps_needed + 1),
            maxlen=self.max_steps_needed + 1,
        )
        self.reward = [0.0] * consumed_substeps
        self.done = [False] * consumed_substeps
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
        help="Rollouts per group (G). All share the same env seed. Defaults to "
             "the number of parallel envs unless --num-async-vector-env is set.",
    )
    parser.add_argument(
        "--num-async-vector-env", type=int, default=None,
        help="Physical AsyncVectorEnv worker count per group. Default None → "
             "group_size. Must be <= group_size and divide it evenly. When "
             "smaller than group_size, each group is collected over "
             "group_size // num_async_vector_env sequential turns.",
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
        "--min-alive-groups", type=int, default=0,
        help="Min ALIVE groups (mixed: 0 < group_successes < group_size, "
             "i.e., per-group reward std > 0 under success_weight=1.0) before "
             "stopping. 0 = disabled (always collect exactly num_groups). "
             "When >0, collector continues past num_groups (capped at "
             "max_groups) until criterion is met.",
    )
    parser.add_argument(
        "--max-groups", type=int, default=None,
        help="Hard cap on dynamic group collection. Defaults to num_groups "
             "(no dynamic collection). Must be <= 100 (GROUP_SEED_STRIDE limit).",
    )
    parser.add_argument(
        "--init-state-npz-path", type=str, default=None,
        help="Override every group's branch point with a saved sim state from "
             "this .npz (must contain __sim_state__, __model_xml__, __ep_meta__ "
             "as produced by scripts/denoising_lab/eval/interactive_rollout.py). "
             "Forces fast-forward off internally. Pair with --success-weight <1.0 "
             "to avoid dead-gradient stalls when starting from a hard state.",
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
        num_async_vector_env: int | None = None,
    ):
        self.env_name = env_name
        self.group_size = group_size            # LOGICAL rollouts per group
        # PHYSICAL AsyncVectorEnv worker count. None → group_size (one worker
        # per rollout = original behavior). Each group is then collected over
        # turns_per_group sequential turns of num_envs rollouts each. Mirror
        # the config-level validation so a standalone CLI run can't bypass it.
        self.num_envs = (
            group_size if num_async_vector_env is None else num_async_vector_env
        )
        if (
            not (1 <= self.num_envs <= self.group_size)
            or self.group_size % self.num_envs != 0
        ):
            raise ValueError(
                f"num_async_vector_env ({self.num_envs}) must satisfy "
                f"1 <= n <= group_size ({self.group_size}) and divide it evenly."
            )
        self.turns_per_group = self.group_size // self.num_envs
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
                total_n_envs=self.num_envs,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
            )
            for i in range(self.num_envs)
        ]

        # AsyncVectorEnv (subprocess workers, parallel MuJoCo) when num_envs > 1;
        # SyncVectorEnv when num_envs == 1 (no IPC overhead). Pattern matches
        # scripts/denoising_lab/eval/robocasa_eval_benchmark.py:336-343.
        #
        # Autoreset handling is gymnasium-VERSION DEPENDENT:
        #   * gymnasium >= 1.0 defaults to AutoresetMode.NEXT_STEP: a terminated
        #     env is flagged and its NEXT step() silently resets it (discarding
        #     the action). Because we re-use one persistent vector env across
        #     turns/groups and re-establish each branch point via the
        #     apply_scene_bundle RPC (which does NOT clear that flag — and for
        #     AsyncVectorEnv(shared_memory=False) a vector reset() does NOT clear
        #     the worker flag either), a leftover NEXT_STEP autoreset would make
        #     the FIRST step after a turn/group boundary reset to a fresh random
        #     scene instead of the branch point, corrupting the within-group GRPO
        #     invariant. We pass autoreset_mode=DISABLED to remove that flag.
        #   * Older gymnasium (e.g. the robocasa venv, ~0.26–0.29) has NO
        #     AutoresetMode and auto-resets on the SAME step as termination (with
        #     terminal info in info["final_info"], which is why _info_for_env
        #     reads final_info first). Same-step autoreset does NOT leak across a
        #     turn/group boundary, so no autoreset_mode is needed — and the enum
        #     doesn't exist, so passing it would raise AttributeError.
        # Detect support and pass the kwarg only when present (correct + safe on
        # both). Either way, _align/_restart call envs.reset() before
        # apply_scene_bundle, which on gymnasium>=1.0 also clears SyncVectorEnv's
        # per-env terminated guard (it asserts you don't step a just-terminated env).
        _autoreset_enum = getattr(gym.vector, "AutoresetMode", None)
        _autoreset_kw = (
            {"autoreset_mode": _autoreset_enum.DISABLED}
            if _autoreset_enum is not None
            else {}
        )
        if self.num_envs > 1:
            self.envs = gym.vector.AsyncVectorEnv(
                env_fns, shared_memory=False, context="spawn", **_autoreset_kw,
            )
            self._uses_async = True
        else:
            self.envs = gym.vector.SyncVectorEnv(env_fns, **_autoreset_kw)
            self._uses_async = False

        self.policy_client = PolicyClient(host=server_host, port=server_port)

        # Lazy-loaded saved init-state bundle (set per-collect() call). Keyed
        # by path so re-load happens only when the path changes — typical
        # overfitting runs reuse the same path across all iters.
        self._init_bundle: dict | None = None
        self._init_bundle_path: str | None = None
        # Per-call slot. _align_envs_to_group_scene reads this; collect()
        # writes it at the top of each invocation. Keeps the signature of
        # _align_envs_to_group_scene unchanged (1 caller, 1 caller's caller).
        self._active_init_bundle_path: str | None = None

        print(f"Collector initialized:")
        print(f"  Env: {env_name} (task_type: {self.task_type})")
        print(f"  Group size (logical rollouts/group): {self.group_size}")
        print(
            f"  Async vector envs (physical workers): {self.num_envs} "
            f"({self.turns_per_group} turn(s)/group)"
        )
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
        min_alive_groups: int = 0,
        max_groups: int | None = None,
        init_state_npz_path: str | None = None,
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

        Dynamic group collection: when min_alive_groups > 0, after
        collecting `num_groups` groups the collector keeps adding one more
        group at a time until either (a) at least `min_alive_groups` groups
        are ALIVE (mixed: 0 < group_successes < group_size — equivalently,
        per-group reward std > 0 under success_weight=1.0 with time-scaling
        disabled, the only regime this binary-mixed predicate is exact for),
        or (b) `max_groups` groups have been collected (warning logged).
        """
        # Default max_groups = num_groups (disables dynamic mode regardless
        # of min_alive_groups, since we can't go beyond num_groups).
        if max_groups is None:
            max_groups = num_groups

        dynamic_mode = min_alive_groups > 0 and max_groups > num_groups

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
        if min_alive_groups > max_groups:
            raise ValueError(
                f"min_alive_groups ({min_alive_groups}) cannot "
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

        # Saved init-state bundle (set per-call) — _align_envs_to_group_scene
        # reads from self._active_init_bundle_path; the lazy load itself runs
        # inside that method via _get_init_bundle so the trainer subprocess
        # only pays the I/O cost the first time the path is seen.
        self._active_init_bundle_path = init_state_npz_path

        ff_enabled = fast_forward_steps > 0 and fast_forward_pct > 0
        # Init-state override pre-empts the FF curriculum: with all groups
        # branching from the same saved scene, an extra model-rollout prefix
        # would just append more divergent post-restore behavior to a state
        # the user already chose. Skip the Bernoulli draw outright.
        if init_state_npz_path is not None:
            ff_enabled = False
        # One Bernoulli for the whole iteration. Drawn from OS entropy
        # (`default_rng()` with no argument) — NOT from `base_seed`. The
        # FF decision is a training-data sampling concern; `base_seed`
        # exists to control env initialization. Coupling them means
        # repeated `base_seed` values (e.g., toy_train_grpo.py reusing the
        # same fixed seed across iters/calls) lock the FF Bernoulli to a
        # deterministic outcome — `ff_pct=0.6` becomes either 100% or 0%
        # depending on which side of the threshold the seeded draw lands.
        # Fresh OS entropy decouples the two concerns: env init stays
        # reproducible from `base_seed`, and the FF draw actually achieves
        # the configured rate.
        # We accept losing bit-exact reproducibility of the FF curriculum
        # across re-runs at the same `--seed`. That property was already
        # partial — torch.randn during denoising isn't seeded, so two
        # re-runs already diverge on policy noise — and no analysis path
        # depends on FF being bit-reproducible.
        ff_rng = np.random.default_rng()
        use_ff_for_iteration = ff_enabled and ff_rng.random() < fast_forward_pct

        # Header
        if dynamic_mode:
            print(
                f"\nCollecting {num_groups}+ groups (cap {max_groups}) "
                f"× {self.group_size} rollouts each..."
            )
            print(
                f"  Dynamic: continue past {num_groups} groups until "
                f">={min_alive_groups} ALIVE (mixed) groups (or hit cap)."
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
        if init_state_npz_path is not None:
            # NOT guarded by _CLEAN_OUTPUT — init-state is a substantive
            # behavior change (all groups share a fixed scene), so the user
            # needs visible confirmation that the flag took effect. Matches
            # the FF prints above which are also unguarded for the same reason.
            print(f"  Init-state override: every group restores from {init_state_npz_path}")

        alive_groups = 0
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
            # "Alive group" = mixed: at least one rollout succeeded AND at
            # least one failed. This matches the trainer's gradient-signal
            # criterion exactly: under success_weight=1.0 with time-scaling
            # disabled (see episode_buffer.py:351-376 for why time-scaling
            # is off), per-group reward std > 0 iff the rewards span both
            # 0 and 1 — i.e., 0 < group_successes < group_size. Pure all-
            # success groups (group_successes == group_size) and pure all-
            # fail groups (group_successes == 0) both have std=0 and get
            # advantage-zeroed by compute_advantages, so neither contributes
            # gradient signal.
            #
            # NOTE: this binary-mixed predicate is exact only when
            # success_weight=1.0 with time-scaling disabled — the regime
            # this codebase trains in. For success_weight<1.0 or time-
            # scaled rewards, the strict criterion is per-group shaped-
            # reward std > 1e-4, which the collector does not currently
            # compute. If you switch reward shaping back on, audit this
            # site (and the resume_from_collected_data validator in
            # train_grpo.py:_validate_collected_data_cache) for the
            # same predicate.
            if 0 < group_successes < self.group_size:
                alive_groups += 1

            group_idx += 1

            n_done = len(all_episodes)
            elapsed = time.time() - start_time
            rate = n_done / elapsed * 60 if elapsed > 0 else 0
            if dynamic_mode:
                progress_str = (
                    f"alive groups: {alive_groups}/{min_alive_groups}, "
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
                or alive_groups >= min_alive_groups
            )
            at_max_cap = group_idx >= max_groups

            if have_min_groups and have_signal:
                break
            if at_max_cap:
                if not have_signal:
                    print(
                        f"  WARNING: hit max_groups={max_groups} cap with only "
                        f"{alive_groups}/{min_alive_groups} alive "
                        f"(mixed) groups — proceeding with what was collected."
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
            f"alive groups: {alive_groups}/{group_idx}"
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
        """Collect one group (group_size rollouts) from the seed-aligned scene,
        across turns_per_group turns of num_envs rollouts each."""
        observations, branch_bundle = self._align_envs_to_group_scene(group_seed)
        return self._run_group_over_turns(
            group_seed, group_id, success_weight, branch_bundle, observations
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
        observations, _seed_bundle = self._align_envs_to_group_scene(group_seed)

        for step in range(fast_forward_steps):
            new_obs = self._lockstep_step(observations)
            if new_obs is None:
                print(
                    f"    Env terminated at step {step} during fast-forward, "
                    "falling back to normal collection"
                )
                # Fallback collects the FULL multi-turn group from seed.
                return self._collect_one_group(
                    group_seed, group_id, success_weight
                )
            observations = new_obs

        if self.debug_fast_forward:
            self._verify_branch_point(
                group_id, group_seed, fast_forward_steps, observations
            )

        # Capture the POST-FF branch point so turns 2..k restart from the exact
        # state turn 1's lockstep reached. Re-running the FF prefix would diverge
        # (the server's denoising noise is unseeded). All num_envs envs are
        # bit-identical here (lockstep guarantee, asserted by _verify_branch_point
        # when debugging), so env 0's bundle IS the shared branch point. Skip the
        # capture for the true singleton (turns_per_group == 1). Note FF mode is
        # never combined with init_state (collect() disables FF when init_state is
        # set), so _run_group_over_turns always re-applies this captured bundle.
        post_ff_bundle = None
        if self.turns_per_group > 1:
            # Deep-copy the captured bundle so it stays pristine across turns:
            # apply_scene_bundle mutates ep_meta in place (see _get_init_bundle),
            # and in SyncVectorEnv env.call passes the arg with no pickle boundary.
            post_ff_bundle = copy.deepcopy(self.envs.call("get_scene_bundle")[0])
            # The FF prefix ran INSIDE the wrapper, so every env's truncation
            # counter (MultiStepWrapper.reward) grew by exactly
            # fast_forward_steps * n_action_steps substeps — reaching this capture
            # point requires that NO env terminated during the prefix (else
            # _lockstep_step returns None and we fall back), so each outer step
            # ran the full n_action_steps. Turn 1 continues from that live wrapper,
            # so its remaining horizon is max_episode_steps minus those substeps.
            # get_scene_bundle does NOT carry the wrapper budget, so without this
            # stamp turns 2..k would restore a FULL budget via apply_scene_bundle
            # (consumed_substeps defaults to 0) and run longer than turn 1 — a
            # within-group horizon asymmetry that biases group-relative advantages.
            post_ff_bundle["consumed_substeps"] = (
                fast_forward_steps * self.n_action_steps
            )

        return self._run_group_over_turns(
            group_seed, group_id, success_weight, post_ff_bundle, observations
        )

    # ─── Multi-turn group driver ──────────────────────────────────────────

    def _restart_at_branch_point(
        self, branch_bundle: dict, group_seed: int
    ) -> list[dict]:
        """Re-apply a group's branch point to all physical envs for a new turn.

        Turn 1 of a group is set up by _align_envs_to_group_scene (reset then
        apply_scene_bundle). Turns 2..k mirror that reset-then-apply. The vector
        env is constructed with autoreset_mode=DISABLED, so there is no NEXT_STEP
        autoreset to leak across the turn boundary — but the preceding
        self.envs.reset() is still required: SyncVectorEnv tracks a per-env
        "just terminated" guard and asserts you don't step a terminated env, and
        the previous turn left every env terminated; reset() clears that guard so
        the next turn's first step is legal. (For AsyncVectorEnv under DISABLED
        the reset is a harmless no-op w.r.t. autoreset.) The reset's own scene is
        immediately overwritten by apply_scene_bundle, so only its guard-clearing
        effect matters.

        For init-state mode we re-fetch a FRESH deep-copied bundle each turn via
        _get_init_bundle (preserving the existing "fresh copy per apply"
        invariant — see _get_init_bundle). For normal / fast-forward modes we
        re-apply a FRESH deep copy of the bundle captured on turn 1 (env 0's
        seed-aligned or post-FF scene). The deep copy is load-bearing:
        apply_scene_bundle mutates ep_meta in place, and in SyncVectorEnv
        (num_envs==1) env.call crosses no pickle boundary, so applying the stored
        branch_bundle directly would accumulate robosuite reset state across turns
        and make turns 2..k diverge from turn 1. apply_scene_bundle restores the
        full scene + dynamic sim state bit-identically and refreshes the wrapper's
        obs/reward/done deques, so each turn starts from an identical clean slate.
        """
        # Clear SyncVectorEnv's per-env terminated guard left by the previous
        # turn's terminating steps before re-applying the branch point.
        self.envs.reset(seed=[group_seed] * self.num_envs)
        if self._active_init_bundle_path is not None:
            bundle = self._get_init_bundle(self._active_init_bundle_path)
        else:
            bundle = copy.deepcopy(branch_bundle)
        obs_tuple = self.envs.call("apply_scene_bundle", bundle)
        return list(obs_tuple)

    def _run_group_over_turns(
        self,
        group_seed: int,
        group_id: int,
        success_weight: float,
        branch_bundle: dict | None,
        first_turn_obs: list[dict],
    ) -> list[dict]:
        """Collect group_size rollouts for one logical group across
        turns_per_group sequential turns of num_envs rollouts each.

        Turn 1 runs on first_turn_obs (already at the branch point, established
        by the caller). Turns 2..k re-apply the branch point to all physical
        envs for a bit-identical restart, then run another num_envs rollouts.
        Every rollout is tagged with the SAME group_id, so GRPO advantage
        normalization (which groups by group_id) treats all turns as one group.
        Within-group diversity comes from the policy's per-query denoising noise
        (unseeded) — fresh on every turn — NOT from env randomness.
        """
        group_episodes = self._run_per_env_loop(
            first_turn_obs, group_id, group_seed, success_weight
        )
        for _turn in range(1, self.turns_per_group):
            # branch_bundle is None only for the true singleton (group_size==1,
            # turns_per_group==1) where this loop never runs; in init-state mode
            # _restart_at_branch_point re-fetches regardless. Guard defensively.
            if branch_bundle is None and self._active_init_bundle_path is None:
                break
            turn_obs = self._restart_at_branch_point(branch_bundle, group_seed)
            group_episodes.extend(
                self._run_per_env_loop(
                    turn_obs, group_id, group_seed, success_weight
                )
            )
        return group_episodes

    # ─── Vector env primitives ────────────────────────────────────────────

    def _align_envs_to_group_scene(
        self, group_seed: int
    ) -> tuple[list[dict], dict | None]:
        """Reset all physical envs to env 0's scene + dynamic state.

        Each subprocess env runs in its own MuJoCo instance with its own
        construction-time random scene. We can't change those choices from
        the parent process, so we use AsyncVectorEnv.call() to invoke the
        composite RPCs on GroupAlignmentWrapper inside each worker:
          1. Reset all envs with seed=group_seed (each ends up with its own
             scene because the RNG choices were baked at construction).
          2. get_scene_bundle on all envs (capture env 0's bundle).
          3. apply_scene_bundle(env0_bundle) on all envs — envs 1..N-1 align
             to env 0; env 0 re-applies its own (a no-op effect, but it runs
             in parallel with the other workers so no wall-time cost).

        When self._active_init_bundle_path is set (overfitting / curriculum
        mode), step 2 is skipped and step 3 broadcasts the lazy-loaded saved
        bundle instead. Every env in every group then starts from the same
        fixed scene + sim state regardless of group_seed; only the policy's
        denoising noise drives within-group divergence.

        Returns (observations, branch_bundle): per-env wrapper-stacked obs
        (bit-identical across the num_envs physical envs when this returns —
        verifiable via --debug-fast-forward), plus env 0's captured scene
        bundle for the turn driver to RE-APPLY on later turns of this group.
        branch_bundle is None only for the true single-rollout group_size==1
        case (one env, one turn — nothing to restart). For init-state mode the
        returned bundle is the loaded fixed bundle; the driver re-fetches a
        fresh copy each turn (see _restart_at_branch_point).
        """
        seeds = [group_seed] * self.num_envs
        vector_obs, _ = self.envs.reset(seed=seeds)

        if self._active_init_bundle_path is not None:
            # Saved-state override: ignore env 0's scene entirely and broadcast
            # the loaded bundle to all envs (including the num_envs==1 case — the
            # user explicitly asked to start from this saved state).
            bundle = self._get_init_bundle(self._active_init_bundle_path)
            obs_tuple = self.envs.call("apply_scene_bundle", bundle)
            return list(obs_tuple), bundle

        if self.group_size == 1:
            # True singleton: one rollout, one env, one turn — nothing to align
            # to and no later turns to restart, so no branch bundle is needed.
            return self._unbatch_vector_obs(vector_obs), None

        # group_size > 1: capture env 0's bundle and broadcast it to all physical
        # envs — even when num_envs == 1 (SyncVectorEnv multi-turn), so that turn
        # 1 and every later turn restart from the SAME captured state. Snapshot a
        # pristine deep copy as the branch point BEFORE applying it: the turn-1
        # apply below mutates ep_meta in place (and in SyncVectorEnv the arg
        # crosses no pickle boundary), so reusing the same object as the stored
        # branch point would carry turn-1 reset state into turns 2..k.
        bundles = self.envs.call("get_scene_bundle")
        branch_bundle = copy.deepcopy(bundles[0])
        obs_tuple = self.envs.call("apply_scene_bundle", bundles[0])
        return list(obs_tuple), branch_bundle

    def _get_init_bundle(self, npz_path: str) -> dict:
        """Return a fresh bundle for one apply_scene_bundle broadcast.

        The npz is parsed once and cached (keyed by path string) — typical
        overfitting runs reuse the same path across all iters, so the npz is
        opened exactly once per collector process.

        Each call returns a NEW dict with FRESH copies of ep_meta (deepcopy)
        and sim_state (np.copy). This defensive copying matters because:

          - `robosuite_env.set_ep_meta(bundle["ep_meta"])` stores the dict by
            reference as `self._ep_meta`; later `robosuite_env.get_ep_meta()`
            calls during reset MUTATE that dict in-place (adding `layout_id`,
            `style_id`, `object_cfgs`, `lang`, `fixture_refs`, etc. — see e.g.
            external_dependencies/robocasa/.../kitchen.py:355 and the
            symmetric writers around line 1100+ in tabletop.py). Without a
            deepcopy here, the cached bundle would accumulate state from
            every iteration's reset and break determinism across groups.
            ep_meta can be nested (object_cfgs is a list of dicts), so a
            shallow copy is not enough.

          - sim_state is mutable; MuJoCo's set_state_from_flattened doesn't
            document copy-vs-reference semantics, so np.copy is cheap insurance.

          - In SyncVectorEnv (num_envs==1) the bundle reaches the wrapper
            by reference (no pickle boundary); in AsyncVectorEnv (num_envs>1)
            pickling already gives each worker a deep copy. The defensive
            copies here cover both cases uniformly. model_xml is left as the
            cached str (Python strings are immutable).

        Cost is negligible (ep_meta is a few KB of nested dicts; sim_state is
        ~60 floats); apply_scene_bundle itself is the expensive operation.
        """
        if self._init_bundle is None or self._init_bundle_path != npz_path:
            if not _CLEAN_OUTPUT:
                print(f"  [init-bundle] loading {npz_path}")
            self._init_bundle = _load_init_bundle(npz_path)
            self._init_bundle_path = npz_path
        return {
            "ep_meta": copy.deepcopy(self._init_bundle["ep_meta"]),
            "model_xml": self._init_bundle["model_xml"],
            "sim_state": np.copy(self._init_bundle["sim_state"]),
            # consumed_substeps is a plain int → safe to share by value.
            "consumed_substeps": self._init_bundle["consumed_substeps"],
        }

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
        actions_full = self._broadcast_actions(actions, self.num_envs)

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
            zeros for already-done envs (under autoreset_mode=DISABLED a done
            env's MultiStepWrapper.step either no-ops, if truncated, or harmlessly
            over-runs the inner env with the zero action — either way we ignore
            done envs by filtering on active_indices).
          - Calls vector env.step (parallel MuJoCo across subprocess workers).
          - Reads terminal info on the terminating step (final_info handling is
            kept as a defensive fallback; pattern from
            robocasa_eval_benchmark.py:393-402).

        Episode num_steps starts at 0 in both normal and post-FF modes.
        """
        episodes = [
            self._new_episode(group_id, group_seed)
            for _ in range(self.num_envs)
        ]
        done_flags = [False] * self.num_envs

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
            # envs are never re-read (filtered out via active_indices), so we
            # don't consume their post-step obs.
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
        for i in range(1, self.num_envs):
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
            for i in range(self.num_envs):
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
                    print(f"    [DEBUG] Layout: {self.num_envs} rows (envs) × {len(env_frames)} cols (cameras)")
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
        return [self._extract_per_env(vector_obs, i) for i in range(self.num_envs)]

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

        Done envs receive a zero action; under autoreset_mode=DISABLED their
        MultiStepWrapper.step either no-ops (if truncated) or over-runs the inner
        env with the zero action, and the result is ignored by the caller
        (filtered out via active_indices). The per-env wasted compute is bounded
        by the slowest env's remaining episode length, which is typically the
        bottleneck anyway.
        """
        out = {}
        for k, v in actions_active.items():
            full = np.zeros((self.num_envs,) + v.shape[1:], dtype=v.dtype)
            for j, env_idx in enumerate(active_indices):
                full[env_idx] = v[j]
            out[k] = full
        return out

    def _info_for_env(self, infos: dict, key: str, env_idx: int):
        """Read info[key][env_idx], preferring final_info when present.

        Under autoreset_mode=DISABLED (what we use) there is no autoreset, so the
        terminating step's regular infos[key] already holds the terminal value
        and the final_info branch is simply an unused, defensive fallback. (It is
        retained for robustness: under SAME_STEP-style autoreset the terminating
        step's regular info would reflect the reset episode and the terminal info
        would live in infos["final_info"][env_idx]. Pattern from
        scripts/denoising_lab/eval/robocasa_eval_benchmark.py:393-402.)
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
        num_async_vector_env=args.num_async_vector_env,
    )

    try:
        episodes = collector.collect(
            num_groups=args.num_groups,
            base_seed=args.seed,
            success_weight=args.success_weight,
            fast_forward_steps=args.fast_forward_steps,
            fast_forward_pct=args.fast_forward_pct,
            min_alive_groups=args.min_alive_groups,
            max_groups=args.max_groups,
            init_state_npz_path=args.init_state_npz_path,
        )
        save_episodes(episodes, args.output_dir)
    finally:
        collector.close()


if __name__ == "__main__":
    main()
