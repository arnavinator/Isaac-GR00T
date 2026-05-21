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
    update_epochs: int = 4

    # Cosmetic alignment: parent's startup print uses num_groups to describe
    # episodes-per-iter. Match the real (fixed-seed) count.
    num_groups: int = len(FIXED_SEEDS)

    # Disable dynamic group collection. Each iter uses exactly the fixed seed
    # set; if a seed produces an all-fail group, the dead-group filter
    # already handles it downstream.
    min_successful_groups: int = 0
    fast_forward_steps: int | list[int] = 3

    # Force ALL groups to fast-forward every iter (production used 0.8, which
    # mixes FF and non-FF iters and adds avoidable noise to the success curve).
    fast_forward_pct: float = 0

    save_interval: int = 1


class ToyGRPOTrainer(GRPOTrainer):
    """Trainer variant that draws env seeds from a FIXED list each iter.

    Mechanism: temporarily mutate (config.seed, config.num_groups,
    config.min_successful_groups, config.max_groups, self.iteration) around
    each subprocess call so the existing _collect_via_subprocess emits one
    group at the exact seed we want. State is restored in a finally block
    before returning, so the rest of the trainer (LR sched, advantage
    compute, GRPO update, checkpointing) sees the unmodified config.

    Re-tags loaded episodes with a unique per-seed group_id because the
    subprocess always writes group_id=0 when num_groups=1; without re-tagging,
    all 7 calls' episodes would merge into one giant pseudo-group and
    compute_advantages would normalize across them instead of within each seed.
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

    def _collect_episodes(self, env_name: str, task_idx: int, max_steps: int) -> None:
        """One subprocess call per fixed seed, accumulating into self.buffer."""
        # Parent-equivalent setup
        self.buffer.clear()
        self._prune_old_episode_dirs()

        episode_dir = Path(self.config.episode_dir) / f"iter_{self.iteration:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        # Wipe any stale per-seed subdirs from an aborted previous toy run
        # (same checkpoint_dir → same iter_NNNN/seed_* layout). Also drop
        # any orphan top-level episode_*.npz that a re-run might glob.
        for stale_d in list(episode_dir.iterdir()):
            if stale_d.is_dir() and stale_d.name.startswith("seed_"):
                shutil.rmtree(stale_d, ignore_errors=True)
        for stale_f in episode_dir.glob("episode_*.npz"):
            stale_f.unlink()

        # Per-task fast_forward_steps (same resolution pattern as parent)
        if isinstance(self.config.fast_forward_steps, list):
            ff_steps = self.config.fast_forward_steps[task_idx]
        else:
            ff_steps = self.config.fast_forward_steps

        print(
            f"  [TOY] Collecting {len(self.fixed_seeds)} fixed seeds × "
            f"{self.config.group_size} rollouts each "
            f"(one subprocess per seed)..."
        )

        # Snapshot every field we mutate inside the per-seed loop, so the
        # restoration in `finally` is exact even on exceptions.
        orig_num_groups = self.config.num_groups
        orig_min_succ = self.config.min_successful_groups
        orig_max_groups = self.config.max_groups
        orig_seed = self.config.seed
        orig_iter = self.iteration

        n_failed_seeds = 0
        try:
            # Force the subprocess to emit exactly one group per call.
            # `min_successful_groups=0` disables dynamic mode; `max_groups=1`
            # satisfies the validator's max_groups>=num_groups invariant.
            self.config.num_groups = 1
            self.config.min_successful_groups = 0
            self.config.max_groups = 1

            for seed_idx, fixed_seed in enumerate(self.fixed_seeds):
                sub_dir = episode_dir / f"seed_{fixed_seed:08d}"
                sub_dir.mkdir(parents=True, exist_ok=True)

                # _collect_via_subprocess passes:
                #     --seed = self.config.seed + self.iteration * 100_000
                # Make this resolve to exactly fixed_seed.
                self.config.seed = fixed_seed
                self.iteration = 0

                print(
                    f"  [TOY] Seed {seed_idx + 1}/{len(self.fixed_seeds)}: "
                    f"env_seed={fixed_seed}"
                )
                # Wrap in try/except: _collect_via_subprocess RETURNS a
                # failure_reason for its known failure modes (timeout,
                # non-zero exit), but a raise can still come from
                # subprocess.Popen (OSError if the robocasa venv path is
                # wrong) or from the stdout pipe iteration (IOError on a
                # broken pipe). Without this guard, such an exception
                # would propagate mid-loop with a partial buffer from
                # earlier successful seeds, and the parent's train() — which
                # doesn't catch — would tear down the run with that partial
                # state still in self.buffer. Treating it as a failed seed
                # keeps the failure accounting consistent with the
                # returned-failure_reason path.
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
                    print(f"  [TOY] Seed {fixed_seed} FAILED: {failure_reason}")
                    n_failed_seeds += 1
                    continue

                # Append this seed's 4 episodes onto the buffer, then re-tag
                # with a unique per-seed group_id (collector wrote 0 since
                # num_groups=1; without this re-tag, compute_advantages would
                # pool all 28 episodes into a single group and the
                # group-relative normalization would lose its meaning).
                n_before = len(self.buffer.episodes)
                self.buffer.load_episodes(sub_dir)
                n_loaded_this_seed = len(self.buffer.episodes) - n_before
                for ep in self.buffer.episodes[n_before:]:
                    ep.group_id = seed_idx
                    ep.env_seed = fixed_seed

                if n_loaded_this_seed == 0:
                    print(f"  [TOY] Seed {fixed_seed} produced 0 episodes")
                    n_failed_seeds += 1
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
                f"  [TOY] WARNING: 0 episodes loaded "
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
        n_succ_seeds = len(self.fixed_seeds) - n_failed_seeds
        print(
            f"  [TOY] Loaded {len(self.buffer.episodes)} episodes from "
            f"{n_succ_seeds}/{len(self.fixed_seeds)} seeds "
            f"({self.buffer.num_chunks} chunks total)"
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

        Indexing: _collect_episodes re-tags every episode with
        `group_id = enumerate(self.fixed_seeds)`, so group_id i ↔
        fixed_seeds[i]. We re-bucket by group_id here and emit one
        scalar per fixed seed.
        """
        super()._log_metrics(
            iteration, stats, update_stats, lr, iter_time, skip_reason,
            phase_times=phase_times, lora_delta_norm=lora_delta_norm,
        )
        if self.writer is None:
            return

        group_to_total: dict[int, int] = {}
        group_to_succ: dict[int, int] = {}
        for ep in self.buffer.episodes:
            group_to_total[ep.group_id] = group_to_total.get(ep.group_id, 0) + 1
            if ep.success:
                group_to_succ[ep.group_id] = group_to_succ.get(ep.group_id, 0) + 1

        for seed_idx, seed in enumerate(self.fixed_seeds):
            total = group_to_total.get(seed_idx, 0)
            if total == 0:
                # This seed's subprocess failed this iter — leave the curve
                # ungapped by skipping (TB plots will interpolate visually).
                continue
            rate = group_to_succ.get(seed_idx, 0) / total
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
    print(f"  KL coef:        {config.kl_coef}")
    print(f"  Clip eps:       {config.clip_eps}")
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
