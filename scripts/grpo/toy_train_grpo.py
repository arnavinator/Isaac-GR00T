"""Toy GRPO experiment on a FIXED set of CoffeeServeMug seeds.

Diagnostic: verify the GRPO + LoRA training loop actually moves the policy by
removing env-distribution variance as a confound. Every iteration evaluates the
model on the SAME 7 seeds (one group per seed). Any trend in success_rate is
then attributable to policy change, not to a different draw of scenes.

Seed pick rationale (from the iter 1-5 production-run logs at run 1779157287):
each of these had 1/4 or 2/4 base-model success → strong intra-group variance
→ non-degenerate group-relative advantages every iter (no all-success or
all-fail saturation right away).

Usage (run separately at each LR, then compare TB curves):
    uv run python scripts/grpo/toy_train_grpo.py --learning-rate 1e-5
    uv run python scripts/grpo/toy_train_grpo.py --learning-rate 1e-4
    uv run python scripts/grpo/toy_train_grpo.py --learning-rate 3e-4

Each LR auto-derives its own grpo_data/toy_lr{LR}/ root for checkpoints,
episodes, and TB logs so head-to-head runs don't collide.

Design constraint: this file does NOT modify any production module under
scripts/grpo/. It subclasses GRPOTrainer/GRPOConfig and overrides only
_collect_episodes; everything else (LR sched, ref log-prob pass, GRPO update,
checkpointing) runs unchanged.
"""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Make sibling modules (grpo_config, train_grpo) importable when this script is
# launched as `python scripts/grpo/toy_train_grpo.py`. Also expose the repo
# root for gr00t.* imports that GRPOTrainer.setup() triggers.
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from grpo_config import GRPOConfig
from train_grpo import GRPOTrainer


# Env seeds where the production run logged 1/4 or 2/4 successes — well-mixed
# enough that the per-group reward std is comfortably above the dead-group
# threshold (1e-4 in episode_buffer.compute_advantages), so every seed
# contributes a real gradient signal instead of getting filtered out.
FIXED_SEEDS: list[int] = [
    # 101067,     # 0/4
    # 203067,     # 1/4
    # 303067,     # 0/4
    305067,     # 0/4
    305067,     # 0/4
    305067,     # 0/4
    # 402067,     # 0/4
    # 406067,     # 1/4
    # # 501067,     # 1/4
    # 502067,     # 0/4
    # 507067,     # 0/4
]


@dataclass
class ToyGRPOConfig(GRPOConfig):
    """GRPOConfig with toy-mode defaults. Every field is still CLI-overridable."""

    # Shorter run: this is a diagnostic, not a real training session.
    num_iterations: int = 15

    # Fewer epochs per iter: with noisy RL gradients, 5 epochs of re-fit on the
    # same 28-episode batch overfits to the minibatch much faster than it
    # improves the policy. 2 epochs is enough to see whether the gradient is
    # moving the right way.
    # update_epochs: int = 4

    update_epochs: int = 2
    jitter_lambda: float = 0.05

    # Cosmetic alignment: parent's startup print uses num_groups to describe
    # episodes-per-iter. Match the real (fixed-seed) count.
    num_groups: int = len(FIXED_SEEDS)

    min_successful_groups: int = 0

    max_groups: int = 8

    # Force ALL groups to fast-forward every iter (production used 0.8, which
    # mixes FF and non-FF iters and adds avoidable noise to the success curve).
    fast_forward_pct: float = 0
    fast_forward_steps: int | list[int] = 3

    save_interval: int = 1


class ToyGRPOTrainer(GRPOTrainer):
    """Trainer variant that draws env seeds from a FIXED list each iter.

    Mechanism: temporarily mutate (config.seed, config.num_groups,
    config.min_successful_groups, config.max_groups, self.iteration) around
    each subprocess call so the existing _collect_via_subprocess emits one
    group at the exact seed we want. State is restored in a finally block
    before returning, so the rest of the trainer (LR sched, advantage
    compute, GRPO update, checkpointing) sees the unmodified config.

    Re-tags loaded episodes with a globally-unique per-attempt group_id
    (the subprocess always writes group_id=0 when num_groups=1; without
    re-tagging, every attempt's episodes would collapse into a single
    pseudo-group spanning the iter) and the slot's env_seed (which
    _log_metrics buckets by). When config.min_successful_groups > 0,
    also runs an outer round-robin retry of any slot whose group had
    no task-successful rollouts, until config.min_successful_groups
    slots have succeeded or config.max_groups total subprocess attempts
    have been made — mirroring the production dynamic-collection
    semantics from collect_episodes.EpisodeCollector.collect, scoped
    to the FIXED_SEEDS list instead of fresh seeds. See
    _collect_episodes for the full retry algorithm.
    """

    def __init__(self, config: ToyGRPOConfig, fixed_seeds: list[int] | None = None):
        super().__init__(config)
        self.fixed_seeds = (
            list(fixed_seeds) if fixed_seeds is not None else list(FIXED_SEEDS)
        )

        # The collector-server route would need its own per-seed call surface
        # (CollectorClient.collect takes a single base_seed). The toy targets
        # subprocess mode; fail fast if the user accidentally points at a
        # collector_server config so they don't get silent wrong behavior.
        if self._collector_client is not None:
            raise RuntimeError(
                "ToyGRPOTrainer only supports subprocess collection. "
                "Set --collector-server-host '' (or remove the flag) to use "
                "the subprocess path, then re-launch."
            )

        # The toy's Phase-1 initial pass unconditionally attempts every fixed
        # seed once before any retries — that's the diagnostic invariant.
        # max_groups < len(FIXED_SEEDS) would cap that pass mid-way, which
        # is almost certainly not what the user intended. The parent
        # validator only enforces max_groups >= num_groups, and num_groups
        # is CLI-overridable independently of fixed_seeds, so a pathological
        # combo (e.g., --num-groups 1 --max-groups 1 with len(FIXED_SEEDS)=3)
        # would slip past the parent check without this guard.
        if self.config.max_groups < len(self.fixed_seeds):
            raise ValueError(
                f"max_groups ({self.config.max_groups}) must be >= "
                f"len(FIXED_SEEDS) ({len(self.fixed_seeds)}). The toy's "
                f"initial pass always attempts every fixed seed once before "
                f"any retries; max_groups < len(FIXED_SEEDS) would cap that "
                f"pass and is almost certainly a misconfiguration."
            )

    def _collect_episodes(self, env_name: str, task_idx: int, max_steps: int) -> None:
        """One subprocess call per fixed seed, retrying failed seeds until
        min_successful_groups slots succeed or max_groups attempts are made.

        Retry semantics match the production dynamic-collection path
        (collect_episodes.EpisodeCollector.collect): a "successful" group has
        at least one rollout that completed the task (group_successes > 0;
        see collect_episodes.py:714). After the initial pass through
        FIXED_SEEDS, if fewer than config.min_successful_groups slots have
        produced a successful group, round-robin retry the still-failed
        slots one at a time until any of:
          - >= min_successful_groups slots have succeeded
          - total attempts reaches config.max_groups
          - all slots have already succeeded (criterion exceeds
            len(FIXED_SEEDS), which is unsatisfiable by adding more
            attempts since each slot contributes at most 1).

        Each subprocess call gets a globally-unique group_id within the
        iter (incremented per attempt), so retries accumulate in the
        buffer alongside the initial pass and downstream advantage
        computation treats them as distinct groups — matching production,
        where dynamically-added groups never replace existing ones. The
        dead-group filter (per-group std < 1e-4 in compute_advantages)
        then handles all-fail and all-success groups uniformly. Episodes
        are also re-tagged with env_seed so _log_metrics can pool
        per-seed TB curves across the initial attempt and any retries of
        the same scene.

        Model-side denoising RNG is unseeded (see collect_episodes.py:
        640-642), so retries with the same env seed produce different
        rollouts on the same kitchen layout — same property the existing
        duplicate-FIXED_SEEDS pattern (e.g., [305067, 305067, 305067])
        already relies on.

        Note: ToyGRPOConfig defaults min_successful_groups=0, so by
        default no retries happen and behavior reduces to the original
        single-pass iteration over FIXED_SEEDS.
        """
        # Parent-equivalent setup
        self.buffer.clear()
        self._prune_old_episode_dirs()

        episode_dir = Path(self.config.episode_dir) / f"iter_{self.iteration:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        # Wipe any stale per-slot subdirs from an aborted previous toy run
        # (same checkpoint_dir → same iter_NNNN/slot_*/ layout). Also drop
        # any orphan top-level episode_*.npz that a re-run might glob.
        for stale_d in list(episode_dir.iterdir()):
            if stale_d.is_dir() and stale_d.name.startswith("slot_"):
                shutil.rmtree(stale_d, ignore_errors=True)
        for stale_f in episode_dir.glob("episode_*.npz"):
            stale_f.unlink()

        # Per-task fast_forward_steps (same resolution pattern as parent)
        if isinstance(self.config.fast_forward_steps, list):
            ff_steps = self.config.fast_forward_steps[task_idx]
        else:
            ff_steps = self.config.fast_forward_steps

        n_slots = len(self.fixed_seeds)
        min_succ = self.config.min_successful_groups
        # max_attempts caps total subprocess calls per iter (initial pass
        # plus any retries). __init__ enforces max_groups >= n_slots so the
        # initial pass always fits the budget, leaving room for >= 0 retries.
        max_attempts = self.config.max_groups
        # Per-attempt log display: when min_succ=0 Phase 2 is skipped, so
        # the actual cap on attempts is n_slots — not max_attempts. Pick
        # the right one so "Attempt N/M" reads honestly.
        attempt_cap_display = max_attempts if min_succ > 0 else n_slots
        if min_succ > 0:
            print(
                f"  [TOY] Collecting {n_slots} fixed seeds × "
                f"{self.config.group_size} rollouts each, then retrying "
                f"failed seeds round-robin until {min_succ} slots succeed "
                f"or {max_attempts} total attempts "
                f"(one subprocess per attempt)..."
            )
        else:
            print(
                f"  [TOY] Collecting {n_slots} fixed seeds × "
                f"{self.config.group_size} rollouts each "
                f"(no retry: min_successful_groups=0)..."
            )

        # Snapshot every field we mutate inside the per-attempt helper,
        # so the restoration in `finally` is exact even on exceptions.
        orig_num_groups = self.config.num_groups
        orig_min_succ = self.config.min_successful_groups
        orig_max_groups = self.config.max_groups
        orig_seed = self.config.seed
        orig_iter = self.iteration

        # Per-slot tracking. A "slot" is an index into FIXED_SEEDS; a slot
        # has "succeeded" once any of its attempts produced a group with
        # >= 1 task-successful rollout (matches the production
        # group_successes>0 criterion in collect_episodes.py:714).
        slot_attempts = [0] * n_slots
        slot_succeeded = [False] * n_slots
        n_attempts = 0           # total subprocess calls this iter
        n_successful_slots = 0   # distinct slots with >= 1 successful attempt
        next_group_id = 0        # globally-unique within this iter

        def _run_one_attempt(slot_idx: int, group_id: int) -> bool:
            """One subprocess call for FIXED_SEEDS[slot_idx], tagging
            loaded episodes with `group_id` and the slot's env_seed.
            Returns True iff the loaded group had >= 1 task-successful
            rollout. Side effects: appends loaded episodes to
            self.buffer.episodes; increments n_attempts and
            slot_attempts[slot_idx].
            """
            nonlocal n_attempts
            fixed_seed = self.fixed_seeds[slot_idx]
            # Unique subdir per (slot, attempt). Including slot_idx is
            # what makes duplicate entries in FIXED_SEEDS (e.g.,
            # [305067, 305067, 305067]) safe: without it, all three
            # Phase-1 slots would compute the same `seed_X` path and
            # each subprocess would overwrite the previous slot's
            # episode_*.npz files on disk. (The in-memory buffer
            # happens to survive because load_episodes runs between
            # subprocess calls, but the on-disk dir would only show
            # the LAST occupant — confusing for post-mortem inspection
            # and one numpy mmap default away from silent data
            # corruption.) load_episodes globs episode_*.npz within
            # one dir, so reusing one across retries of the SAME slot
            # would also re-load earlier attempts alongside the new one.
            attempt_idx = slot_attempts[slot_idx]
            if attempt_idx == 0:
                sub_dir = (
                    episode_dir
                    / f"slot_{slot_idx:02d}_seed_{fixed_seed:08d}"
                )
            else:
                sub_dir = (
                    episode_dir
                    / f"slot_{slot_idx:02d}_seed_{fixed_seed:08d}"
                      f"_retry_{attempt_idx:02d}"
                )
            sub_dir.mkdir(parents=True, exist_ok=True)

            # _collect_via_subprocess passes:
            #     --seed = self.config.seed + self.iteration * 100_000
            # Make this resolve to exactly fixed_seed regardless of attempt.
            self.config.seed = fixed_seed
            self.iteration = 0

            slot_attempts[slot_idx] = attempt_idx + 1
            n_attempts += 1

            tag = (
                f"slot {slot_idx + 1}/{n_slots}"
                if attempt_idx == 0
                else f"RETRY {attempt_idx} of slot {slot_idx + 1}"
            )
            print(
                f"  [TOY] Attempt {n_attempts}/{attempt_cap_display}: "
                f"env_seed={fixed_seed} ({tag}), group_id={group_id}"
            )

            # Wrap in try/except: _collect_via_subprocess RETURNS a
            # failure_reason for its known failure modes (timeout,
            # non-zero exit), but a raise can still come from
            # subprocess.Popen (OSError on a bad venv path) or from
            # stdout pipe iteration (IOError on a broken pipe).
            # Without this guard, such an exception would propagate
            # mid-loop with a partial buffer from earlier successful
            # attempts, and the parent's train() — which doesn't catch
            # — would tear down the run with that partial state still
            # in self.buffer. Treating as a failed attempt keeps
            # accounting consistent with the returned-failure_reason path.
            try:
                failure_reason = self._collect_via_subprocess(
                    env_name=env_name,
                    episode_dir=sub_dir,
                    max_steps=max_steps,
                    ff_steps=ff_steps,
                )
            except Exception as e:
                failure_reason = (
                    f"unexpected exception: {type(e).__name__}: {e}"
                )

            if failure_reason is not None:
                print(f"  [TOY]   FAILED: {failure_reason}")
                return False

            n_before = len(self.buffer.episodes)
            self.buffer.load_episodes(sub_dir)
            n_loaded = len(self.buffer.episodes) - n_before
            if n_loaded == 0:
                print(f"  [TOY]   produced 0 episodes")
                return False

            # Re-tag with our globally-unique group_id and the slot's
            # env_seed value. Collector always writes group_id=0 since
            # num_groups=1; without re-tagging, every attempt's
            # episodes would collapse into a single pseudo-group
            # spanning the iter.
            n_success_in_group = 0
            for ep in self.buffer.episodes[n_before:]:
                ep.group_id = group_id
                ep.env_seed = fixed_seed
                if ep.success:
                    n_success_in_group += 1
            had_success = n_success_in_group > 0
            print(
                f"  [TOY]   loaded {n_loaded} episodes, "
                f"{n_success_in_group}/{n_loaded} succeeded "
                f"(group {'ALIVE' if had_success else 'failed'})"
            )
            return had_success

        try:
            # Force the subprocess to emit exactly one group per call.
            # `min_successful_groups=0` disables the subprocess's own
            # dynamic mode; `max_groups=1` satisfies the validator's
            # max_groups>=num_groups invariant. The toy's outer Python
            # loop replaces subprocess-side dynamic collection.
            self.config.num_groups = 1
            self.config.min_successful_groups = 0
            self.config.max_groups = 1

            # Phase 1: full initial pass through every fixed seed.
            # Independent of min_successful_groups — every seed always
            # gets at least one attempt, matching the toy's
            # "fixed seed" diagnostic intent.
            for slot_idx in range(n_slots):
                had_success = _run_one_attempt(slot_idx, next_group_id)
                next_group_id += 1
                if had_success and not slot_succeeded[slot_idx]:
                    slot_succeeded[slot_idx] = True
                    n_successful_slots += 1

            # Phase 2: round-robin retry of still-failed slots. Skipped
            # entirely when min_successful_groups==0 (default) — in that
            # case behavior is identical to the original single-pass toy.
            # Each successful retry takes its slot out of the rotation;
            # the toy's interest is per-slot success, not extra
            # successful attempts on already-good slots.
            retry_cursor = 0
            while (
                min_succ > 0
                and n_successful_slots < min_succ
                and n_attempts < max_attempts
            ):
                # True fixed-order round-robin: walk slot indices forward
                # from retry_cursor (wrapping), skipping already-succeeded
                # slots. Naive `still_failed[cursor % len(still_failed)]`
                # was incorrect: removing a slot when it succeeds shifts
                # later slots' positions in the list, so the cursor
                # silently skipped a still-failed slot in the next pick.
                slot_idx: int | None = None
                for offset in range(n_slots):
                    cand = (retry_cursor + offset) % n_slots
                    if not slot_succeeded[cand]:
                        slot_idx = cand
                        retry_cursor = (cand + 1) % n_slots
                        break
                if slot_idx is None:
                    # All slots have succeeded but criterion is still
                    # unmet — only possible when min_successful_groups
                    # > len(FIXED_SEEDS), since each slot can
                    # contribute at most 1 to n_successful_slots.
                    # Adding more attempts to already-succeeded slots
                    # wouldn't help; stop.
                    print(
                        f"  [TOY] All {n_slots} slots have succeeded "
                        f"but min_successful_groups={min_succ} > "
                        f"len(FIXED_SEEDS)={n_slots} — criterion is "
                        f"unsatisfiable. Stopping retries."
                    )
                    break
                had_success = _run_one_attempt(slot_idx, next_group_id)
                next_group_id += 1
                if had_success:
                    # First success for this slot — bump the count.
                    # (We only pick still-failed slots above, so
                    # slot_succeeded[slot_idx] is guaranteed False here.)
                    slot_succeeded[slot_idx] = True
                    n_successful_slots += 1
        finally:
            self.config.num_groups = orig_num_groups
            self.config.min_successful_groups = orig_min_succ
            self.config.max_groups = orig_max_groups
            self.config.seed = orig_seed
            self.iteration = orig_iter

        # Total-failure handling, mirroring the parent's accounting so a
        # genuinely broken collector still trips the consecutive-failure abort.
        if not self.buffer.episodes:
            self._consecutive_collect_failures += 1
            print(
                f"  [TOY] WARNING: 0 episodes loaded across all "
                f"{n_attempts} attempts "
                f"({self._consecutive_collect_failures}/"
                f"{self._max_consecutive_collect_failures} consecutive failures)"
            )
            if (
                self._consecutive_collect_failures
                >= self._max_consecutive_collect_failures
            ):
                raise RuntimeError(
                    f"Toy collector failed "
                    f"{self._consecutive_collect_failures} consecutive iters. "
                    f"Check the [collector] log lines above."
                )
            return

        self._consecutive_collect_failures = 0
        # Actual loaded group count (not n_attempts): an attempt that
        # crashed or loaded 0 episodes contributes no group_id to the
        # buffer, so next_group_id (which increments per attempt) would
        # overstate. Pull the real count from buffer.episodes.
        n_groups_loaded = len({ep.group_id for ep in self.buffer.episodes})
        summary = (
            f"  [TOY] Loaded {len(self.buffer.episodes)} episodes from "
            f"{n_groups_loaded} groups across {n_attempts} subprocess attempts; "
            f"{n_successful_slots}/{n_slots} slots had >= 1 successful rollout"
        )
        if min_succ > 0:
            summary += f" (target: {min_succ})"
        summary += f" ({self.buffer.num_chunks} chunks total)"
        print(summary)
        # Final warning only when the attempt cap was the actual reason
        # for stopping short. The unsatisfiable-criterion branch
        # (min_succ > len(FIXED_SEEDS)) already printed its own message
        # above and exits with n_attempts < max_attempts, so don't
        # mislabel it as "hit attempt cap".
        if (
            min_succ > 0
            and n_successful_slots < min_succ
            and n_attempts >= max_attempts
        ):
            print(
                f"  [TOY] WARNING: hit attempt cap "
                f"({n_attempts}/{max_attempts}) with only "
                f"{n_successful_slots}/{min_succ} successful slots."
            )

    def _log_metrics(
        self,
        iteration,
        stats,
        update_stats=None,
        lr=None,
        iter_time=None,
        skip_reason=None,
        phase_times=None,
        lora_delta_norm=None,
    ):
        """Extend parent logging with per-seed success rates.

        Since the toy uses the SAME seeds every iter (unlike production,
        where each iter's seeds are disjoint), each seed's success curve
        IS that env init's learning curve. The iter-mean success_rate
        obscures bimodal "some seeds climbing, others flat" patterns;
        per-seed lines surface them.

        Bucketing: by ep.env_seed (the seed of the fixed initial state).
        With the retry path in _collect_episodes, a single fixed seed
        can produce multiple groups per iter — the initial attempt plus
        zero or more retries — each tagged with its own globally-unique
        group_id but the SAME env_seed. Pooling by env_seed combines
        those attempts into one curve per seed, which is what the
        diagnostic actually wants: "for THIS scene, what was the
        success rate this iter across however many rollouts I
        collected?" Duplicate entries in FIXED_SEEDS (e.g., the
        [305067]*3 default) likewise pool under their single seed value.
        """
        super()._log_metrics(
            iteration, stats, update_stats, lr, iter_time, skip_reason,
            phase_times=phase_times, lora_delta_norm=lora_delta_norm,
        )
        if self.writer is None:
            return

        seed_to_total: dict[int, int] = {}
        seed_to_succ: dict[int, int] = {}
        for ep in self.buffer.episodes:
            seed_to_total[ep.env_seed] = seed_to_total.get(ep.env_seed, 0) + 1
            if ep.success:
                seed_to_succ[ep.env_seed] = seed_to_succ.get(ep.env_seed, 0) + 1

        # dict.fromkeys preserves first-occurrence order and dedupes.
        for seed in dict.fromkeys(self.fixed_seeds):
            total = seed_to_total.get(seed, 0)
            if total == 0:
                # This seed's subprocess failed every attempt this iter —
                # leave the curve ungapped by skipping (TB plots will
                # interpolate visually).
                continue
            rate = seed_to_succ.get(seed, 0) / total
            self.writer.add_scalar(
                f"toy_seeds/seed_{seed}_success_rate", rate, iteration
            )


def main():
    try:
        import tyro
        config = tyro.cli(ToyGRPOConfig)
    except ImportError:
        print("tyro not available; using ToyGRPOConfig defaults")
        config = ToyGRPOConfig()

    # Isolate per-LR runs so head-to-head comparisons don't clobber each
    # other's checkpoints/episodes/TB logs. Derive the root from the parsed
    # learning_rate (post any CLI override). Use one decimal of mantissa
    # (`.1e`) so neighbor LRs like 1.5e-5 and 1.0e-5 don't both collapse to
    # the same dir name (`.0e` would round both to "1e-05"). Strip the
    # zero-pad in the exponent on both signs so the dir reads naturally.
    lr_tag = (
        f"{config.learning_rate:.1e}"
        .replace("e-0", "e-")
        .replace("e+0", "e+")
    )
    toy_root = Path(f"grpo_data/toy_lr{lr_tag}")
    config.checkpoint_dir = str(toy_root / "checkpoints")
    config.episode_dir = str(toy_root / "episodes")

    print("=" * 60)
    print("Toy GRPO experiment")
    print("=" * 60)
    print(f"  Output root:    {toy_root}")
    print(f"  Fixed seeds:    ({len(FIXED_SEEDS)}) {FIXED_SEEDS}")
    print(f"  Learning rate:  {config.learning_rate}")
    print(f"  Iterations:     {config.num_iterations}")
    print(f"  Update epochs:  {config.update_epochs}")
    print(f"  Group size:     {config.group_size}")
    print(f"  Mini-batch:     {config.mini_batch_size}")
    print(f"  KL coef:        last_iter={config.kl_coef_last_iter} "
          f"base_model={config.kl_coef_base_model}")
    print(f"  Clip eps lo/hi: {config.clip_eps_low} / {config.clip_eps_high}")
    print(f"  Jitter lambda:  {config.jitter_lambda}")
    print(f"  FF steps / pct: {config.fast_forward_steps} / "
          f"{config.fast_forward_pct}")
    print("=" * 60)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    trainer = ToyGRPOTrainer(config, fixed_seeds=FIXED_SEEDS)
    trainer.setup()
    try:
        trainer.train()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
