"""Main GRPO training loop for GR00T N1.6 DiT.

This is the orchestrator that ties everything together:
1. Loads model + applies LoRA
2. Iterates: collect episodes → compute advantages → pre-compute ref log-probs → policy update

Key differences from grpo_cont.py:
- Episode collection is via subprocess (robocasa venv) + ZMQ server
- Log-prob uses FM surrogate instead of Gaussian distribution
- Advantages are episodic (group-relative on time-scaled rewards)
- Reference log-probs pre-computed per iteration (no deep-copied reference model)

Usage:
    uv run python scripts/grpo/train_grpo.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \\
        --num-iterations 200 \\
        --group-size 5 --num-groups 5

Hardware: Fits on A10G (24GB) with batch_size=4 and shared backbone.
"""

import sys
import dataclasses
import math
import os
import re
import shutil
import threading
import time
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grpo_config import GRPOConfig
from lora_dit import (
    apply_lora_to_dit,
    save_lora_checkpoint,
    load_lora_checkpoint,
    print_trainable_params,
    disabled_adapters,
)
from fm_log_prob import compute_fm_log_prob, _sample_jittered_timesteps
from episode_buffer import EpisodeBuffer, ActionChunk


# Canonical iter directory name pattern: 'iter_<ASCII digits>'.
# Exposed at module scope so the test suite can import it instead of
# duplicating the regex literal — duplicating the literal lets test and
# prod silently drift apart (e.g., a test using `\d+` would still pass
# even if prod regressed to `\d+`, missing Bug A4's regression coverage).
# [0-9]+ is intentionally narrower than `\d+` (which matches all
# Unicode digit categories, ~580 codepoints including full-width '０'-'９')
# so a checkpoint named with non-ASCII digits doesn't silently parse.
ITER_DIR_RE = re.compile(r"iter_([0-9]+)")


# Dynamic positive-advantage weighting (config.positive_advantage_weight_scaling).
# The tunables (k cap, target ratio) are config fields; these are the fixed
# smoothing constants. See _grpo_update_inner for the full algorithm.
_POS_SCALE_BETA = 0.5    # cross-iteration EMA weight on history
_POS_SCALE_PRIOR = 0.05  # within-iter seed from last iter's EMA (~a few % of an iter's mass)
_POS_SCALE_EPS = 1e-8


def is_anchor_row(chunk, include_anchor_groups: bool) -> bool:
    """Whether a chunk gets ANCHOR treatment this iteration.

    Reads the config gate rather than `chunk.is_anchor` alone, so a buffer
    flagged by an earlier `compute_advantages` call (or a resumed run whose
    config changed) can never admit anchor rows that the rest of the update
    would then treat as ordinary zero-advantage chunks. Shared by
    `_compute_ref_log_probs` and `_grpo_update_inner` so their row filters are
    the same expression by construction — if they diverged, a row could reach
    the update without a precomputed ref log-prob.
    """
    return include_anchor_groups and chunk.is_anchor


def clip_killed_gradient(
    ratio: torch.Tensor,
    surr1: torch.Tensor,
    surr2: torch.Tensor,
    clip_eps_low: float,
    clip_eps_high: float,
) -> torch.Tensor:
    """Which rows had their CLIP-TERM gradient zeroed by the clamp.

    Module-level (not inlined in _grpo_update_inner) so tests exercise the real
    predicate instead of re-deriving it — a re-derived copy would keep passing
    if this expression were changed.

    `torch.min(surr1, surr2)` returns the clamped bound iff `surr2 <= surr1`;
    that bound is a CONSTANT in `ratio` only when the clamp actually moved the
    ratio. Hence the conjunction. Verified against all four cases:

        A>0, rho < 1-lo : surr1 = A*rho < A*(1-lo) = surr2  -> min picks surr1, ALIVE
        A>0, rho > 1+hi : surr1 = A*rho > A*(1+hi) = surr2  -> min picks surr2, DEAD
        A<0, rho < 1-lo : surr1 = A*rho > A*(1-lo) = surr2  -> min picks surr2, DEAD
        A<0, rho > 1+hi : surr1 = A*rho < A*(1+hi) = surr2  -> min picks surr1, ALIVE

    i.e. positive-advantage rows can only ever die on the UPPER bound and
    negative ones only on the LOWER bound. That asymmetry is what the
    sign-agnostic `clipfrac` hides, and it is why a large `jitter_pos` — which
    pushes every positive row's ratio BELOW `1-clip_eps_low` — inflates
    `clipfrac` to ~1.0 while killing nothing.

    NAMING CAVEAT: this is the gradient of the CLIP TERM only. With
    kl_coef_last_iter / kl_coef_base_model > 0 (both default 0.2) a "dead" row
    still passes gradient through its KL terms, which depend on
    current_log_probs regardless of the clamp. So this is not "the row
    contributed no gradient", it is "the surrogate contributed no gradient".

    ZERO-ADVANTAGE ROWS. `A == 0` gives `surr1 == surr2`, so the `surr2 <= surr1`
    half is satisfied — but the conjunction still requires `clamp_moved`. So a
    zero-advantage row is reported DEAD only if its ratio is outside the band and
    ALIVE if inside, even though it has no gradient either way. That is the right
    call for a metric named after the clip (the clip is not what killed it), but
    it under-reports in one reachable case: under per-minibatch renorm a
    minibatch whose rows all share a single advantage value z-scores to exactly
    0.0 on every row (the `+1e-8` epsilon case documented in _grpo_update_inner),
    and those rows then land in the `_neg` bucket — `post_renorm_pos_mask` is
    all-False — diluting `clipfrac_effective_neg` toward the clamp-moved fraction
    instead of the 1.0 that "no gradient" would suggest. Watch
    `n_pos_flipped_by_renorm` and the `_neg` denominator if that curve looks odd.
    """
    clamp_moved = (ratio < 1 - clip_eps_low) | (ratio > 1 + clip_eps_high)
    return clamp_moved & (surr2 <= surr1)


class GRPOTrainer:
    """GRPO training loop for GR00T N1.6 DiT with LoRA.

    This class manages the full training pipeline:
    - Model setup (LoRA injection, persistent server)
    - Episode collection (launches collector subprocess)
    - Reference log-prob pre-computation (single no-grad pass)
    - Advantage computation (group-relative normalization)
    - Policy gradient update (clipped surrogate + KL penalty)
    - Checkpointing
    """

    def __init__(self, config: GRPOConfig):
        """Initialize the GRPO trainer.

        Args:
            config: Complete GRPO configuration.
        """
        self.config = config
        self.device = torch.device(config.device)

        # Will be set in setup()
        self.model = None
        self.optimizer = None
        self.iteration = 0

        # Iteration number of the last gradient update that actually fired.
        # When the skip-update path runs (collection failed or std_reward~0),
        # we use THIS as the checkpoint dir name instead of the current loop
        # iter — so resume from the saved checkpoint retries the skipped iter
        # rather than burning it from the num_iterations budget. Set in
        # setup(): 0 for a fresh run, resumed_iter for --resume-from.
        self._last_updated_iteration = 0

        # Episode buffer for current iteration's data
        self.buffer = EpisodeBuffer()

        # Logging
        self.writer = None  # TensorBoard/wandb writer

        # Dynamic positive-advantage weighting: cross-iteration EMA of the
        # per-iter alive-negative (N) / positive (D) loss mass. None until the
        # first update folds in its own masses. NOT persisted in checkpoints, so a
        # resumed run drops pre-resume EMA history and re-warms: the first update
        # after a fresh start OR a resume runs with k=1 (no weighting) while these
        # are None, then re-seeds — safe, but not seamless across resume.
        self._pos_scale_N_ema = None
        self._pos_scale_D_ema = None

        # Re-entrant lock serializing ALL model forward/backward passes
        # between the server thread (serving inference for the collector
        # subprocess) and the main thread (reference log-prob pass,
        # _grpo_update). Normally the collector subprocess has finished by
        # the time training phases run, but a late/stuck ZMQ request would
        # otherwise let the server thread fire a forward pass through the
        # model while the trainer is mid-backward(). Both paths take this
        # lock, so one waits for the other. RLock because each path takes
        # it at most once per call, but RLock costs nothing and guards
        # against accidental nesting in future edits.
        self._model_lock = threading.RLock()

        # Consecutive collector failures since the last successful collection.
        # We treat (subprocess timeout / non-zero exit / zero episodes loaded)
        # as failure modes and abort training after MAX_CONSECUTIVE_COLLECT_FAILURES
        # in a row. Without this guard, a misconfigured robocasa venv or a stuck
        # MuJoCo init would leave the trainer in a silent infinite no-op:
        # collector exits early → empty buffer → std_reward<1e-8 → iter skipped
        # → repeat forever, with the user discovering it hours later.
        self._consecutive_collect_failures = 0
        self._max_consecutive_collect_failures = 3

        # Rollout/load split of Phase 1, refreshed per iteration by
        # _collect_episodes / _load_cached_episodes and logged as
        # time/collect_rollout_seconds and time/collect_load_seconds. NaN means
        # "this sub-phase did not run", which _log_metrics turns into a gap
        # rather than a 0 data point.
        self._collect_rollout_time = float("nan")
        self._collect_load_time = float("nan")

        # Reference-MSE diagnostics, refreshed per iteration by
        # _compute_ref_log_probs and emitted as ref_mse/* (see _log_metrics).
        # MSE_ref = -ref_log_prob is the reference policy's own FM loss on the
        # action IT sampled, and it is the HARD CEILING on the importance ratio:
        #     log rho = MSE_ref - MSE_theta,  MSE_theta >= 0  =>  rho <= e^MSE_ref
        # so it is the total reinforcement headroom available on a chunk. It
        # costs nothing to log (the values are already on the chunks) and it is
        # the only direct read on positive-branch saturation: MSE_ref falling
        # toward 0 on positive-advantage chunks means the surrogate has nothing
        # left to gain there, no matter how the advantage is weighted.
        # None = the ref pass has not run this iteration (skipped iter).
        self._ref_mse_stats: dict | None = None

        # Per-chunk jitter-gap survey, refreshed per iteration by
        # _per_chunk_gap_survey and emitted as chunk_gap/*. None when
        # per_chunk_gap_survey_size == 0 (the default) or when the ref pass
        # produced too little usable data.
        self._chunk_gap_stats: dict | None = None

    def setup(self):
        """Load the model + LoRA, configure optimizer, validate the resume
        cache (when ``resume_from_collected_data=True``), and start the
        persistent inference server thread.

        Heavy work that's deferred from ``__init__`` so a misconfigured
        trainer fails fast at construction without paying for the multi-
        minute model load.
        """
        import gr00t.model  # noqa: F401 — registers model classes
        from transformers import AutoModel, AutoProcessor

        print("=" * 60)
        print("GRPO Training Setup")
        print("=" * 60)

        # Seed RNGs at the START of setup so LoRA-A's Kaiming init via
        # torch.randn (inside inject_adapter_in_model) is reproducible across
        # runs. main() already seeds before constructing the trainer, but
        # re-seeding here makes setup() self-contained — calling it from a
        # notebook or a custom entry point that forgot to seed will still
        # produce deterministic LoRA initialization.
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Compute start_iteration up-front so cached-collection validation
        # (when resume_from_collected_data=True) can run BEFORE the multi-
        # minute model load + LoRA injection. The actual LoRA-load happens in
        # Step 2b below; this block only parses the iter number from the
        # resume path so both paths share one source of truth.
        #
        # Use re.fullmatch with [0-9]+ instead of `startswith("iter_") +
        # split("_")[1]`: the old form silently accepted broken names
        # (e.g., "iter_50.bak" raised a cryptic ValueError; "iter_50_v2"
        # silently parsed as int("50") and ignored the v2 suffix;
        # "iter_-5" silently produced a negative iter number that broke
        # f-string padding downstream). The regex requires exactly "iter_"
        # followed by ASCII digits — anything else (including Unicode
        # digits like full-width "iter_０", which `\d+` would silently
        # accept and `int()` would parse) is treated as non-iter and
        # handled the same as a freshly named checkpoint ("best",
        # "latest"). [0-9]+ is intentionally narrower than \d+ so the
        # canonical-name guard is conservative.
        start_iteration = self._parse_resume_iteration()
        self._start_iteration = start_iteration
        # Resumed checkpoint represents iter (start_iteration - 1)'s end-of-update
        # state. For a fresh run, no update has fired yet → 0. The skip-save
        # path keys off this value to name checkpoints after real progress.
        self._last_updated_iteration = start_iteration - 1

        # Pre-flight: when resume_from_collected_data=True, validate that the
        # cached iter's episodes are usable BEFORE we sink minutes into model
        # load. Validation failure raises here; train() never runs.
        # The "should this iter use the cache?" decision is rederived in the
        # train loop as `iteration == start_iteration AND flag is set` —
        # cleaner than maintaining a one-shot mutable field across two methods.
        if self.config.resume_from_collected_data:
            self._validate_collected_data_cache(start_iteration)

        # --- Step 1: Load pretrained model ---
        print(f"\n[1/4] Loading model from {self.config.model_path}...")
        self.model = AutoModel.from_pretrained(self.config.model_path)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.eval()  # Start in eval mode (we manually control train/eval per component)

        # Load processor for action encoding/decoding
        self.processor = AutoProcessor.from_pretrained(self.config.model_path)
        self.processor.eval()

        # --- Step 2: Apply LoRA to DiT ---
        print(f"\n[2/4] Applying LoRA (rank={self.config.lora_rank})...")
        self.model = apply_lora_to_dit(
            self.model,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )

        # Cast trainable LoRA params from bf16 → fp32 for training.
        # Why: AdamW stores its momentum buffers (exp_avg, exp_avg_sq) in the
        # same dtype as the params. With bf16 LoRA at lr=1e-5, most Adam
        # updates are smaller than the bf16 ULP (~2^-7 × |param| ≈ 1e-4 for
        # typical LoRA values ~0.01) and round to zero, so the policy barely
        # moves regardless of gradient magnitude. Standard PEFT practice keeps
        # LoRA params in fp32 even when the base model is bf16.
        # Memory cost: ~80 MB extra (~20M LoRA params × 2 extra bytes), tiny
        # vs the ~6 GB frozen bf16 base. The frozen base model stays bf16 —
        # only trainable params (LoRA A/B) are upcast.
        # Forward pass: PEFT's LoraLayer.forward() handles dtype mismatch by
        # casting x to lora_A.weight.dtype (fp32) inside the LoRA branch and
        # casting the LoRA delta back to the base layer's dtype before the
        # residual add (peft/tuners/lora/layer.py); the base linear path
        # stays bf16-clean.
        n_upcast = 0
        for p in self.model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
                n_upcast += 1
        if n_upcast > 0:
            print(f"  Upcast {n_upcast} trainable LoRA params from bf16 → fp32 "
                  f"(prevents Adam moment underflow at lr={self.config.learning_rate})")

        stats = print_trainable_params(self.model)

        # --- Step 2b: Load LoRA checkpoint if resuming ---
        # start_iteration was already computed at the top of setup() (so the
        # cached-collection pre-flight could see it). This block just performs
        # the actual LoRA weight load.
        if self.config.resume_from:
            resume_path = Path(self.config.resume_from)
            print(f"\n  Resuming from: {resume_path}")
            load_lora_checkpoint(self.model, resume_path)
            if start_iteration > 1:
                print(f"  Continuing from iteration {start_iteration}")

        # Snapshot the trainable LoRA params for cumulative-drift logging
        # (lora/weight_delta_norm in _log_metrics). Resumed runs snapshot at
        # the resume point; fresh runs snapshot at PEFT init. The metric
        # tracks how far the policy has moved SINCE THIS RUN STARTED.
        # Cost: ~80 MB for ~20M fp32 LoRA params, dwarfed by the 6 GB frozen
        # base. Must run AFTER fp32 upcast AND after resume-load so the
        # baseline tensor dtypes/values match what the optimizer sees.
        self._lora_init_params = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

        # --- Step 3: Setup optimizer (only LoRA params) ---
        print("\n[3/4] Setting up optimizer...")
        # Capture (name, param) pairs in the SAME order that
        # model.parameters() yields. PyTorch documents named_parameters() and
        # parameters() as iterating in identical order (insertion order via
        # _parameters / _modules dicts), and the optimizer is constructed from
        # that order — so this list IS the optimizer's positional ordering.
        # Persisted alongside optimizer.pt (see _save_checkpoint) so resume
        # can detect a position permutation that the shape-only validation
        # in _validate_optimizer_state would otherwise miss.
        named_trainable = [
            (n, p) for n, p in self.model.named_parameters() if p.requires_grad
        ]
        self._lora_param_names = [n for n, _ in named_trainable]
        trainable_params = [p for _, p in named_trainable]

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-5,  # Same as grpo_cont.py line 230
        )

        # Load optimizer state if resuming
        if self.config.resume_from:
            opt_path = Path(self.config.resume_from) / "optimizer.pt"
            if opt_path.exists():
                payload = torch.load(opt_path, map_location=self.device)
                # New format wraps state_dict + param-name metadata. Legacy
                # format is the raw state_dict (pre-fix checkpoints).
                if (
                    isinstance(payload, dict)
                    and "optimizer_state" in payload
                    and "param_names" in payload
                ):
                    saved = payload["optimizer_state"]
                    self._validate_optimizer_param_names(payload["param_names"])
                else:
                    print(
                        "  WARNING: optimizer.pt was saved by an older trainer "
                        "version without parameter-name metadata. Falling back "
                        "to shape-only validation; same-shape param permutations "
                        "(from a peft/torch version bump between save and load) "
                        "could go undetected."
                    )
                    saved = payload
                self._validate_optimizer_state(saved)
                self.optimizer.load_state_dict(saved)
                print(f"  Optimizer state restored from {opt_path}")
            else:
                print(f"  WARNING: No optimizer.pt found at {opt_path}, starting fresh optimizer")

        print(f"  AdamW: lr={self.config.learning_rate}, wd={self.config.weight_decay}")
        print(f"  Trainable params in optimizer: {sum(p.numel() for p in trainable_params):,}")

        # --- Step 4: Setup logging ---
        print("\n[4/4] Setting up logging...")
        if self.config.use_wandb:
            try:
                import wandb
                run_name = self.config.wandb_run_name or f"grpo_{time.strftime('%m%d_%H%M')}"
                wandb.init(
                    project=self.config.wandb_project,
                    name=run_name,
                    config=vars(self.config),
                )
                print(f"  Wandb initialized: {self.config.wandb_project}/{run_name}")
            except ImportError:
                print("  Wandb not available, using TensorBoard only.")
                self.config.use_wandb = False

        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(self.config.checkpoint_dir) / "tb_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))
        print(f"  TensorBoard logs: {log_dir}")
        self._log_config()
        print("  Logged full run config to TensorBoard (tag: config)")

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # --- Start persistent server ---
        # The server shares self.model, so LoRA weight updates are reflected automatically
        self._server_handle = self._start_server_thread()

        print("\n" + "=" * 60)
        print("Setup complete. Ready to train.")
        print("=" * 60)

    def shutdown(self):
        """Clean up resources (server thread, tensorboard writer)."""
        if hasattr(self, '_server_handle') and self._server_handle is not None:
            self._stop_server_thread(self._server_handle)
            self._server_handle = None
        if self.writer is not None:
            self.writer.close()

    def train(self):
        """Main training loop.

        Structure mirrors grpo_cont.py lines 242-457:
            for update in range(1, num_updates+1):
                # Anneal LR
                # Collect episodes
                # Compute advantages
                # GRPO policy update (multiple epochs × minibatches)
                # Log metrics
        """
        print(f"\nStarting training: {self.config.num_iterations} iterations")
        total_eps = self.config.group_size * self.config.num_groups
        is_dynamic = (
            self.config.min_alive_groups > 0
            and self.config.max_groups > self.config.num_groups
        )
        if is_dynamic:
            max_eps = self.config.group_size * self.config.max_groups
            print(
                f"  Episodes per iteration: {total_eps}-{max_eps} "
                f"({self.config.num_groups}-{self.config.max_groups} groups × "
                f"{self.config.group_size}; dynamic, target "
                f">={self.config.min_alive_groups} alive groups)"
            )
        else:
            print(
                f"  Episodes per iteration: {total_eps} "
                f"({self.config.num_groups} groups × {self.config.group_size})"
            )
        if self._resolved_num_async_vector_env() != self.config.group_size:
            print(
                f"  Async vector envs: {self._resolved_num_async_vector_env()} "
                f"workers → {self._turns_per_group()} turns/group "
                f"(group_size={self.config.group_size})"
            )
        print(f"  Update epochs: {self.config.update_epochs}")
        print(f"  Mini-batch size: {self.config.mini_batch_size}")
        # Surface KL anchor strengths so the operator can confirm at a glance
        # whether the base-model anchor is active (default 0.0 = disabled).
        # Without this, an operator inspecting only logs has no way to tell
        # the two coefs apart from a typo'd CLI flag.
        print(
            f"  KL anchors: last_iter={self.config.kl_coef_last_iter} "
            f"base_model={self.config.kl_coef_base_model}"
            f"{' (disabled)' if self.config.kl_coef_base_model == 0.0 else ''}"
        )
        if self.config.balanced_minibatch_training:
            print(
                f"  Balanced mini-batch sampling: ON "
                f"(positive_adv_ratio={self.config.balanced_minibatch_positive_adv_ratio})"
            )
        if self.config.dynamic_epoch_training:
            print(
                f"  Dynamic epoch count: ON "
                f"(tent epochs=max(1, floor(2·min(sf,1-sf)·{self.config.update_epochs}+0.5)))"
            )
        if self.config.jitter_pos > 0.0 or self.config.jitter_neg > 0.0:
            # Surface the scheduling mode up-front. In paired mode, remind the
            # user to halve update_epochs if they want to match vanilla GRPO's
            # per-iter step budget; jitter-only mode already matches it.
            if self.config.jitter_paired:
                print(
                    f"  Jitter-GRPO: pos={self.config.jitter_pos} "
                    f"neg={self.config.jitter_neg} paired=True "
                    f"(fixed+jitter — 2× minibatches per epoch; "
                    f"halve update_epochs to match vanilla per-iter step count)"
                )
            else:
                print(
                    f"  Jitter-GRPO: pos={self.config.jitter_pos} "
                    f"neg={self.config.jitter_neg} paired=False "
                    f"(jitter-only — 1× minibatches per epoch, step count "
                    f"matches vanilla at the same update_epochs; no `_fixed` "
                    f"branch metrics)"
                )
        if self.config.include_anchor_groups:
            # Anchor groups reclassify all-success groups from dead to trainable.
            # Print the pairing advice: the per-minibatch z-score is what makes
            # a fixed anchor magnitude drift batch to batch, so per-iteration
            # norm is the intended companion.
            print(
                f"  Anchor groups: ON "
                f"(advantage={self.config.anchor_advantage:g}"
                f"{' — KL-only' if self.config.anchor_advantage == 0.0 else ''}, "
                f"row budget={self.config.anchor_max_row_frac:g}× signal rows)"
            )
            # Anchor rows occupy minibatch slots, so the signal rows spread over
            # MORE minibatches — i.e. more optimizer steps per iteration at the
            # same LR. Same caveat as jitter_paired's 2× warning below; worth
            # stating because it is a confound when comparing an anchors-on run
            # against an anchors-off baseline.
            _bound = (
                f"~2×" if self.config.anchor_max_row_frac <= 1.0
                else f"up to {self.config.mini_batch_size}×"
            )
            print(
                f"    NOTE: anchor rows take minibatch slots, so the per-iter "
                f"optimizer step count RISES ({_bound} at "
                f"anchor_max_row_frac={self.config.anchor_max_row_frac:g}), and "
                f"under active max_grad_norm clipping their added KL mass "
                f"rescales the signal gradient too. Lower update_epochs or "
                f"anchor_max_row_frac to match an anchors-off baseline."
            )
            if (
                self.config.anchor_advantage > 0.0
                and not self.config.per_iteration_advantage_norm
            ):
                print(
                    "    NOTE: per_iteration_advantage_norm=False — signal rows "
                    "renorm per minibatch while anchor rows use the buffer-wide "
                    "std, so the anchor:signal weight ratio wobbles with batch "
                    "composition. --per-iteration-advantage-norm is the intended "
                    "pairing."
                )
        print(f"  Estimated time: ~{self.config.num_iterations * 5 / 60:.1f} hours")

        for iteration in range(self._start_iteration, self.config.num_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            # Release memory back to OS before launching this iter's collector
            # subprocess. The collector spawns 5 AsyncVectorEnv workers (~5 GiB
            # RSS each); without this, glibc's heap retains ~2-4 GiB of dead
            # numpy/.npz allocations from the previous iter and squeezes the
            # workers into swap.
            self._log_mem_snapshot(f"iter {iteration} start (pre-release)")
            self._release_memory_to_os()
            self._log_mem_snapshot(f"iter {iteration} start (post-release)")

            # --- Learning rate annealing (mirrors grpo_cont.py lines 244-250) ---
            frac = 1.0 - (iteration - 1) / self.config.num_iterations
            frac = max(frac, 0.1)  # Don't decay below 10% of initial LR
            lr = frac * self.config.learning_rate
            self.optimizer.param_groups[0]["lr"] = lr

            # Clear last iteration's MSE_ref summary so an iteration that never
            # reaches the ref pass (collection failure, all-dead buffer) leaves a
            # gap in the ref_mse/* curves instead of silently re-emitting the
            # previous iteration's numbers at this step.
            self._ref_mse_stats = None
            self._chunk_gap_stats = None

            # --- Select task for this iteration (round-robin across env_names) ---
            # Each iteration focuses on ONE task and collects all num_groups for it.
            # With 8 tasks and 200 iterations, each task gets 25 full training updates.
            # This keeps group-relative advantages meaningful (same task within a group).
            task_idx = (iteration - 1) % len(self.config.env_names)
            env_name = self.config.env_names[task_idx]

            # Resolve per-task max_episode_steps
            if isinstance(self.config.max_episode_steps, list):
                max_steps = self.config.max_episode_steps[task_idx]
            else:
                max_steps = self.config.max_episode_steps

            print(f"\n{'─' * 50}")
            print(f"Iteration {iteration}/{self.config.num_iterations} | Task: {env_name.split('/')[-1]} | LR: {lr:.2e}")

            # ═══ Phase 1: Collect episodes (or reuse cached, when this iter
            # is the first resumed iter and resume_from_collected_data=True) ═══
            phase1_start = time.time()
            # Reset the Phase 1 sub-phase timers HERE rather than inside
            # _collect_episodes / _load_cached_episodes: subclasses override
            # those (toy_train_grpo.py does), and an override that doesn't set
            # the timers would otherwise leave the previous iteration's values
            # to be re-logged as if they were this iteration's.
            self._collect_rollout_time = float("nan")
            self._collect_load_time = float("nan")
            # Derive directly from the iter index instead of carrying a
            # mutable one-shot field across setup() and train(). Cache
            # validation has already passed in setup() (raised otherwise),
            # so reaching `iteration == start_iteration` with the flag on
            # means the cache is good to consume. After the first iter,
            # `iteration > start_iteration` so this is False forever — same
            # one-shot semantics, no field bookkeeping required.
            used_cached_collection = (
                self.config.resume_from_collected_data
                and iteration == self._start_iteration
            )
            if used_cached_collection:
                self._load_cached_episodes()
            else:
                self._collect_episodes(env_name, task_idx, max_steps)
            phase1_time = time.time() - phase1_start

            # ═══ Phase 2: Compute advantages ═══
            phase2_start = time.time()
            self.buffer.compute_advantages(
                max_episode_steps=max_steps,
                anchor_advantage=self.config.anchor_advantage,
                include_anchor_groups=self.config.include_anchor_groups,
                anchor_max_row_frac=self.config.anchor_max_row_frac,
            )
            stats = self.buffer.stats()
            phase2_time = time.time() - phase2_start

            # Skip update if no gradient signal (all same outcome). Anchor groups
            # are the exception: an all-success iteration has std_reward == 0 yet
            # every group is an anchor, and those rows are exactly the ones the
            # feature exists to train on. All-FAIL iterations still land here
            # (no anchors), as do all-success iterations with the feature off.
            # Keyed on the trainable CHUNK counts, not on std_reward. Those are
            # different questions: an all-fail + all-success mix has
            # std_reward = 0.5 yet zero signal chunks, so a std_reward test never
            # fires for it. `n_signal_chunks == 0` is strictly stronger — a mixed
            # group spans both reward values, so std_reward < 1e-8 implies no
            # mixed group — and it also catches the mix.
            #
            # Anchor chunks (not groups) rescue the iteration — counting CHUNKS
            # covers an anchor group that survived classification but contributes
            # no rows — provided the iteration can actually learn something from
            # them. Two ways it can:
            #   - anchor_advantage > 0: a real reinforcement gradient.
            #   - kl_coef_base_model > 0: KL(base || current) is NOT degenerate at
            #     theta == theta_ref once LoRA has moved, so it pulls the policy
            #     back toward the pretrained model ON THE SOLVED STATES. That is
            #     exactly the "trust region never covers the solved states" gap
            #     Layer 1 exists to close, and the coefficient defaults to 0.2, so
            #     skipping here would defeat the documented Layer-1 recipe.
            # KL(ref || current) alone does NOT qualify: its gradient is zero at
            # the start of the update and only re-anchors drift this same update
            # introduced, so a step would apply little but weight decay and
            # carried momentum while consuming an iteration the pre-anchor code
            # preserved for a retry.
            if (
                stats.get("n_signal_chunks", 0) == 0
                and not (
                    stats.get("n_anchor_chunks", 0) > 0
                    and (
                        self.config.anchor_advantage > 0.0
                        or self.config.kl_coef_base_model > 0.0
                    )
                )
            ):
                print(
                    f"  Skipping update: no trainable chunks "
                    f"(signal={stats.get('n_signal_chunks', 0)}, "
                    f"anchor={stats.get('n_anchor_chunks', 0)}, "
                    f"anchor_advantage={self.config.anchor_advantage:g})"
                )
                # Pass lr and iter_time so train/learning_rate and
                # time/iteration_seconds aren't dropped on the early-skip
                # path — TB curves with gaps are harder to read than
                # curves with continuous data including the skipped iters.
                self._log_metrics(
                    iteration, stats, skip_reason="no_signal",
                    lr=lr,
                    iter_time=time.time() - iter_start,
                    phase_times={
                        # Same NaN sentinel as the Phase 4 log site — keeps
                        # the cached-iter gap consistent across both the
                        # normal-completion path and this early-skip path.
                        "collect": (
                            float("nan")
                            if used_cached_collection
                            else phase1_time
                        ),
                        "collect_rollout": self._collect_rollout_time,
                        "collect_load": self._collect_load_time,
                        "advantage": phase2_time,
                    },
                    lora_delta_norm=self._compute_lora_delta_norm(),
                )
                # Save under the LAST UPDATED iter's name (not the current loop
                # iter), so resume from this checkpoint retries the current
                # iter rather than burning it from the num_iterations budget.
                # Skip the write if that dir already exists — overwriting it
                # would lose the prior on-disk state for no benefit (model
                # weights and optimizer moments are unchanged from then).
                if iteration % self.config.save_interval == 0:
                    self._save_checkpoint_for_skipped_iter(iteration)
                continue

            # ═══ Phase 2b: Pre-compute reference log-probs ═══
            # VRAM accounting, part 1: snapshot BEFORE the ref pass populates
            # the per-chunk feature cache, so `base` isolates the costs that do
            # NOT scale with buffer size (frozen bf16 weights + fp32 LoRA +
            # AdamW moments) from the cache, which scales with live-chunk count.
            # Needed because peak VRAM decomposes as
            #     base + per_chunk × n_live_chunks + per_row × mini_batch_size
            # and only the last term is what raising mini_batch_size buys.
            # Note: at iteration 1 `base` reads ~160 MB low — AdamW allocates
            # exp_avg/exp_avg_sq lazily on the first step().
            vram = self._vram_snapshot(reset_peak=True)
            phase2b_start = time.time()
            self._compute_ref_log_probs()
            phase2b_time = time.time() - phase2b_start
            if vram is not None:
                vram["ref_peak"] = (
                    torch.cuda.max_memory_allocated(self.device) / 1e9
                )
                # Live (not total) allocation after caching == base + cache.
                vram["fixed"] = torch.cuda.memory_allocated(self.device) / 1e9
                torch.cuda.reset_peak_memory_stats(self.device)

            # ═══ Phase 3: GRPO Policy Update ═══
            phase3_start = time.time()
            try:
                update_stats = self._grpo_update()
            except RuntimeError as e:
                # Report the measurement before propagating. An OOM here is the
                # EXPECTED outcome when probing for the largest feasible
                # mini_batch_size, and the per-row figure derived from the peak
                # is exactly what that probe is after — losing it to the
                # traceback would mean re-running the whole iteration to learn
                # nothing new. Catches RuntimeError rather than
                # torch.cuda.OutOfMemoryError so it also covers allocator paths
                # (e.g. cuBLAS workspace) that raise a bare RuntimeError; the
                # re-raise below means nothing is swallowed either way.
                self._log_vram(vram, oom="out of memory" in str(e).lower())
                raise
            phase3_time = time.time() - phase3_start
            self._log_vram(vram)

            # Treat an iter as "updated" only if at least one optimizer.step()
            # actually fired. Two paths lead to n_updates=0 here that the outer
            # chunk-keyed skip-check above does NOT catch:
            #   1. Every minibatch had non-finite loss (bf16 ratio overflow).
            #   2. Every accumulation window was dropped because its ACCUMULATED
            #      gradient was non-finite (see _apply_accumulated_grads). Unlike
            #      1 and 2 this one CAN coincide with n_micro_batches > 0 —
            #      minibatches trained, but no step was allowed to reach the
            #      weights. update_stats carries n_nonfinite_grad_steps so the
            #      three cases are distinguishable in the logs.
            # In all cases the model + optimizer are bit-identical to the
            # prior successful iter. Don't bump _last_updated_iteration, and
            # write the save (if scheduled) under the prior iter's name via
            # _save_checkpoint_for_skipped_iter — so resume retries this
            # iter rather than burning it from the num_iterations budget.
            did_update = update_stats.get("n_updates", 0) > 0
            if did_update:
                self._last_updated_iteration = iteration
            else:
                print(
                    f"  No gradient steps fired this iter (n_updates=0). "
                    f"Treating iter {iteration} as skipped — model state "
                    f"unchanged from iter {self._last_updated_iteration}."
                )

            # ═══ Phase 4: Logging and checkpointing ═══
            iter_time = time.time() - iter_start
            self._log_metrics(
                iteration, stats, update_stats, lr, iter_time,
                phase_times={
                    # NaN sentinel marks "no real collection ran" — the
                    # cached load is a few seconds (decompressing .npz video
                    # tensors), and logging that as a data point against an
                    # axis dominated by ~7 min × num_groups normal-collection
                    # points would compress the chart's autoscale toward zero
                    # for the rest of the run. _log_metrics skips NaN entries
                    # so TB shows a clean gap at the resumed iter.
                    "collect": (
                        float("nan") if used_cached_collection else phase1_time
                    ),
                    # Sub-phases of `collect`: rollout wall time (the collector
                    # subprocess: robocasa import + worker spawn + rollouts +
                    # npz writes) vs the trainer-side npz read-back. Without the
                    # split, a rollout regression and an I/O regression look
                    # identical on the `collect` curve. Both carry the NaN
                    # sentinel when the phase didn't run (see
                    # _collect_episodes / _load_cached_episodes).
                    "collect_rollout": self._collect_rollout_time,
                    "collect_load": self._collect_load_time,
                    "advantage": phase2_time,
                    # Phase 2b. Previously untimed, which left ~10% of every
                    # iteration unaccounted for when subtracting the logged
                    # phases from time/iteration_seconds.
                    "ref_logprob": phase2b_time,
                    "update": phase3_time,
                },
                lora_delta_norm=self._compute_lora_delta_norm(),
            )

            collect_label = (
                f"cached ({phase1_time:.2f}s)"
                if used_cached_collection
                else f"{phase1_time:.0f}s"
            )
            # Mirror the TB split in the console line: rollout vs npz read-back
            # inside collect, and the ref-logprob pass that used to be invisible
            # here (it was silently folded into `total`). Each part is shown only
            # when it ran — on the cached path `load` is the ONLY real work, so
            # they're reported independently rather than as a pair.
            parts = []
            if not math.isnan(self._collect_rollout_time):
                parts.append(f"rollout={self._collect_rollout_time:.0f}s")
            if not math.isnan(self._collect_load_time):
                parts.append(f"load={self._collect_load_time:.1f}s")
            if parts:
                collect_label += f" [{', '.join(parts)}]"
            print(
                f"  Time: collect={collect_label}, "
                f"advantage={phase2_time:.1f}s, "
                f"ref_logprob={phase2b_time:.0f}s, "
                f"update={phase3_time:.0f}s, "
                f"total={iter_time:.0f}s"
            )

            # Save checkpoint. When did_update is False, route through the
            # skipped-iter save path so the dir name reflects real progress.
            if iteration % self.config.save_interval == 0:
                if did_update:
                    self._save_checkpoint(iteration)
                else:
                    self._save_checkpoint_for_skipped_iter(iteration)

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

        # Final save under the last successfully-updated iter's name, so the
        # checkpoint always represents real progress. Skip if the run never
        # produced an update, or if a save_interval boundary already wrote
        # this dir during the loop.
        final_iter = self._last_updated_iteration
        if final_iter <= 0:
            print("Final save skipped: no successful update ran during training.")
        else:
            final_dir = Path(self.config.checkpoint_dir) / f"iter_{final_iter:04d}"
            if final_dir.exists():
                print(f"Final save skipped: iter_{final_iter:04d}/ already exists.")
            else:
                self._save_checkpoint(final_iter)

    def _release_memory_to_os(self):
        """Force memory back to the OS before starting a new iter.

        EpisodeBuffer.clear() drops Python references but glibc keeps freed
        allocations in its per-thread cache rather than returning them to
        the kernel. With ~2 GiB of episode .npz arrays loaded each iter,
        the heap grows monotonically and eventually squeezes the
        AsyncVectorEnv workers (~5 GiB each) into swap, where I/O contention
        with /mnt/scratch/swapfile makes collection 2-3x slower than its
        non-swapping baseline.
        """
        import gc
        import ctypes

        # Drop the previous iter's buffered episodes + cached chunk features
        # FIRST. Without this, gc.collect() and malloc_trim() can't release
        # any of it because self.buffer still holds live references to the
        # 25 episodes (~2-3 GiB of numpy arrays) and chunk-cached GPU
        # tensors. _collect_episodes() will call clear() again at Phase 1
        # start; that second call is a no-op on the now-empty buffer.
        self.buffer.clear()

        # gc.collect() before malloc_trim: breaks any reference cycles
        # between ActionChunks and parent episodes that would otherwise pin
        # numpy buffers past clear(). A single call already collects every
        # generation; the second pass picks up any garbage created by
        # finalizers run during the first pass (cheap insurance).
        gc.collect()
        gc.collect()

        if torch.cuda.is_available():
            # Synchronize first so any in-flight kernels finish and their
            # output tensors become eligible for the caching allocator to
            # reclaim. empty_cache() then returns those blocks to the driver.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Ask glibc to return freed heap pages to the kernel. Without this,
        # the heap is sticky-high even after Python has dropped all refs.
        # Best-effort: skipped on non-glibc libcs (musl, macOS).
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            # Best-effort: OSError if libc.so.6 absent (musl, macOS),
            # AttributeError if the symbol is missing (unusual builds).
            # Never let an optional cleanup crash training.
            pass

    def _vram_snapshot(self, reset_peak: bool = False) -> dict | None:
        """Start a per-iter VRAM measurement; returns None on non-CUDA hosts.

        Deliberately NOT gated on config.clean_output (unlike
        _log_mem_snapshot): the mini_batch_size ceiling is a hard operational
        constraint the operator has to size against, and clean_output defaults
        True, so gating would hide the numbers in exactly the runs that need
        them.
        """
        if not torch.cuda.is_available():
            return None
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        return {"base": torch.cuda.memory_allocated(self.device) / 1e9}

    def _log_vram(self, vram: dict | None, oom: bool = False) -> None:
        """Print + log the VRAM decomposition for one iteration.

        Peak VRAM during the update decomposes as

            base + per_chunk × n_cached_chunks + per_row × mini_batch_size

        Only the LAST term grows when mini_batch_size grows, which is why the
        three are reported separately: `per_row` is what you extrapolate to
        find the largest feasible mini_batch_size, while `cache` is a fixed
        cost per iteration that grows with group_size × num_groups (and with
        episode LENGTH, since chunks/episode = num_steps / n_action_steps).

        `n_cached_chunks` varies iter to iter (dead groups shrink it, dynamic
        group collection up to max_groups grows it, and include_anchor_groups
        keeps all-success groups that would otherwise have been dropped — capped
        by anchor_max_row_frac), so size the budget against the worst case —
        max_groups × group_size × chunks-per-episode — not against whatever the
        first iteration happens to report.

        Note `per_row` also absorbs the len(tau_centers) multiplier: the K-loop
        in compute_fm_log_prob accumulates into one loss, so autograd retains
        activations for all K DiT passes at once. Halving tau_centers roughly
        halves per_row.
        """
        if vram is None:
            return
        # `fixed` is absent if the ref pass itself OOM'd before reporting.
        fixed = vram.get("fixed")
        if fixed is None:
            return
        peak = torch.cuda.max_memory_allocated(self.device) / 1e9
        base = vram["base"]
        mb = self.config.mini_batch_size
        # Cheap: _build_chunks memoizes, so this returns
        # the same objects _compute_ref_log_probs hung the feature cache on.
        n_cached = sum(
            1
            for c in self.buffer._build_chunks()
            if c.cached_backbone_features is not None
        )
        per_row = (peak - fixed) / mb if mb > 0 else float("nan")
        per_chunk_mb = (fixed - base) / n_cached * 1000 if n_cached > 0 else float("nan")
        total_gb = (
            torch.cuda.get_device_properties(self.device).total_memory / 1e9
        )
        print(
            f"  [vram]{' OOM' if oom else ''} base={base:.2f} "
            f"cache={fixed - base:.2f} fixed={fixed:.2f} "
            f"ref_peak={vram['ref_peak']:.2f} upd_peak={peak:.2f} "
            f"/ {total_gb:.1f}GB | per_row={per_row:.4f}GB (mb={mb}, "
            f"K={len(self.config.tau_centers)}) "
            f"per_chunk={per_chunk_mb:.2f}MB (n={n_cached})"
        )
        if self.writer is not None and not oom:
            # Logged to TB as well as stdout so the ACROSS-iteration spread is
            # visible — the first iteration is rarely the high-water mark.
            for tag, val in (
                ("base", base), ("cache", fixed - base), ("fixed", fixed),
                ("ref_peak", vram["ref_peak"]), ("update_peak", peak),
                ("per_row", per_row), ("n_cached_chunks", n_cached),
            ):
                self.writer.add_scalar(f"vram/{tag}", val, self.iteration)

    def _log_mem_snapshot(self, label: str) -> None:
        """Log RSS+Swap of the trainer process. Used to detect cross-iter
        accumulation: if Total climbs across iters at the same label, the
        cleanup in _release_memory_to_os is missing something.

        No-op when config.clean_output=True — paired with the worker-side
        [worker_mem pid=...] suppression in collect_episodes.py.
        """
        if self.config.clean_output:
            return
        try:
            with open("/proc/self/status") as f:
                fields = {}
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        fields[k.strip()] = v.strip()
            rss_mb = int(fields.get("VmRSS", "0 kB").split()[0]) / 1024
            swap_mb = int(fields.get("VmSwap", "0 kB").split()[0]) / 1024
            print(f"  [mem {label}] RSS={rss_mb:.0f}MB Swap={swap_mb:.0f}MB Total={rss_mb + swap_mb:.0f}MB")
        except Exception:
            # Non-critical logging utility: /proc/self/status unavailable
            # (non-Linux), unexpected format, or any other parsing issue
            # should never crash training. Skip silently.
            pass

    def _collect_episodes(self, env_name: str, task_idx: int, max_steps: int):
        """Collect episodes for one iteration into self.buffer.

        Spawns a fresh subprocess of collect_episodes.py (in the robocasa
        venv), which writes episodes as .npz files to episode_dir; we then
        load them into self.buffer and run failure handling.

        Records the rollout/load split into self._collect_rollout_time and
        self._collect_load_time for the time/* TB curves. Both start as NaN so a
        phase that didn't run (e.g. load, when the subprocess failed) logs a gap
        instead of a misleading 0.
        """
        self._collect_rollout_time = float("nan")
        self._collect_load_time = float("nan")
        self.buffer.clear()

        # Prune BEFORE we create this iter's directory so the on-disk dir
        # count never exceeds `episode_dirs_to_keep`, even mid-collection or
        # after a crash. Pruning post-collection (the old order) left a
        # transient `keep+1` window between mkdir and the prune call — if the
        # trainer was killed in that window, the user saw keep+1 dirs. The
        # current iter's dir doesn't exist yet here, so there's no risk of
        # deleting a directory we're about to read from. Also runs on iters
        # whose collection later fails (the failure path early-returns), so
        # failed-iter dirs no longer linger an extra iteration.
        self._prune_old_episode_dirs()

        # Output directory for this iteration's episodes.
        # Remove any leftover episode_*.npz from a previous run before the
        # collector writes new files — without this, load_episodes() would
        # glob in stale data (e.g., if this iteration's config collects fewer
        # episodes than the previous one, old files would survive the
        # overwrite and contaminate advantage computation).
        # We do NOT rmtree the whole directory so debug outputs like
        # debug_ff/*.png are preserved for post-mortem inspection.
        episode_dir = Path(self.config.episode_dir) / f"iter_{self.iteration:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        for stale in episode_dir.glob("episode_*.npz"):
            stale.unlink()

        # Resolve per-task fast_forward_steps (same pattern as max_episode_steps).
        if isinstance(self.config.fast_forward_steps, list):
            ff_steps = self.config.fast_forward_steps[task_idx]
        else:
            ff_steps = self.config.fast_forward_steps

        total_episodes = self.config.group_size * self.config.num_groups
        is_dynamic = (
            self.config.min_alive_groups > 0
            and self.config.max_groups > self.config.num_groups
        )
        if is_dynamic:
            max_total = self.config.group_size * self.config.max_groups
            print(
                f"  Collecting {self.config.num_groups}+ groups (cap "
                f"{self.config.max_groups}) × {self.config.group_size} "
                f"rollouts = {total_episodes}-{max_total} episodes..."
            )
        else:
            print(
                f"  Collecting {self.config.num_groups} groups × "
                f"{self.config.group_size} = {total_episodes} episodes..."
            )

        # Run collection: spawn a fresh subprocess in the robocasa venv.
        rollout_start = time.time()
        failure_reason = self._collect_via_subprocess(
            env_name, episode_dir, max_steps, ff_steps,
        )
        self._collect_rollout_time = time.time() - rollout_start

        # Common post-processing: load episodes, then handle any failure.
        n_loaded = 0
        if failure_reason is None:
            load_start = time.time()
            n_loaded = self.buffer.load_episodes(episode_dir)
            self._collect_load_time = time.time() - load_start
            if n_loaded == 0:
                failure_reason = (
                    "zero episodes loaded (collector reported success but "
                    "produced no .npz files)"
                )

        if failure_reason is not None:
            self._consecutive_collect_failures += 1
            print(
                f"  WARNING: Collector failure ({self._consecutive_collect_failures}"
                f"/{self._max_consecutive_collect_failures} consecutive): {failure_reason}"
            )
            if self._consecutive_collect_failures >= self._max_consecutive_collect_failures:
                # Aborting rather than silently looping: empty buffer →
                # advantages are all-zero → iteration skipped → next iter
                # repeats the same failure mode. Without this guard the user
                # would discover the silent stall hours later.
                raise RuntimeError(
                    f"Collector failed {self._consecutive_collect_failures} consecutive "
                    f"iterations. Last reason: {failure_reason}. "
                    f"Common causes: robocasa venv path wrong, server port stuck in "
                    f"TIME_WAIT, MUJOCO_GL backend missing, or model OOM during inference. "
                    f"Check the [collector] log lines above this message."
                )
            return

        # Successful collection — reset the failure counter.
        self._consecutive_collect_failures = 0
        print(f"  Loaded {n_loaded} episodes ({self.buffer.num_chunks} chunks)")

        # Surface partial-success silently passing as success. We don't
        # increment the failure counter here (the load was technically
        # successful and may still produce useful gradients), but a sudden
        # drop in episode count usually points to MuJoCo worker crashes,
        # IPC stalls, or env-side termination bugs — worth seeing in the
        # log so the operator can investigate.
        self._warn_partial_collection(n_loaded, source="collection")

    def _warn_partial_collection(self, n_loaded: int, *, source: str) -> None:
        """Emit a 'fewer episodes than expected' warning, common to both the
        live-collection (`_collect_episodes`) and cached-load
        (`_load_cached_episodes`) paths.

        In dynamic mode the collector may produce more groups than
        `num_groups`, so the static `group_size * num_groups` lower bound
        would suppress this warning when actual collection > num_groups
        but lost episodes within those groups. Use the max of (configured
        minimum, actually-loaded distinct group_ids) as the expected
        group count — catches partial-loss within loaded groups AND
        static-mode under-collection. Doesn't catch entirely-missing
        groups in dynamic mode (no signal in the buffer for that).

        `source` toggles the warning's tail between "some workers may have
        failed silently" (live collection — likely transient subprocess
        failure) and "partial cache" (cached path — partial cache that
        nonetheless passed the validator's undercount-warns policy).
        """
        if not self.buffer.episodes:
            # Empty buffer is handled by the caller's failure path; nothing
            # meaningful to warn about here.
            return
        loaded_group_ids = len(set(ep.group_id for ep in self.buffer.episodes))
        expected_groups = max(self.config.num_groups, loaded_group_ids)
        expected_total = self.config.group_size * expected_groups
        if n_loaded < expected_total:
            pct = 100 * n_loaded / expected_total if expected_total > 0 else 0
            tail = (
                "some workers may have failed silently"
                if source == "collection"
                else "partial cache"
            )
            print(
                f"  WARNING: Only {n_loaded}/{expected_total} episodes "
                f"({pct:.0f}%) loaded across {loaded_group_ids} group(s) — "
                f"{tail}. Failure counter NOT incremented."
            )

    def _load_cached_episodes(self) -> None:
        """Load pre-collected episodes for the resumed iter — bypass collection.

        Mirrors the post-load tail of _collect_episodes (clear buffer, load,
        log success, surface partial-collection warnings, reset failure
        counter) but skips both the dispatch (subprocess / RPC) and the
        stale-file wipe — the cache IS the data we want to consume.

        Setup() has already validated this cache exists, has >= num_groups
        groups, and matches the round-robin task; reaching this method with
        an empty buffer would mean the directory was deleted between setup
        and train(), which we treat as a hard error (the operator's
        invariant is broken — better to stop than silently fall through).

        Takes no env_name/task_idx/max_steps args — unlike _collect_episodes,
        the cached path doesn't dispatch to any collector that would need
        them. The iter directory name is derived from self.iteration alone.
        """
        # No rollouts ran this iter, so the rollout timer keeps the NaN "did not
        # run" sentinel set by the caller; the load below is real work and is
        # timed. (Re-asserting NaN here keeps this method correct if it is ever
        # called outside the train loop.)
        self._collect_rollout_time = float("nan")
        self._collect_load_time = float("nan")
        self.buffer.clear()

        # Deliberately do NOT prune older iter dirs here. _prune_old_episode_dirs
        # runs inside _collect_episodes BEFORE the new iter's dir is created;
        # for cached iters there's no new dir to create, and the cached iter
        # itself is the dir we'd want to keep. The next iter's normal collect
        # path will run pruning, so on-disk count is bounded one iter later
        # than usual — fine.
        episode_dir = Path(self.config.episode_dir) / f"iter_{self.iteration:04d}"
        print(
            f"  resume_from_collected_data: reusing cached episodes at "
            f"{episode_dir} (skipping collection)."
        )

        load_start = time.time()
        n_loaded = self.buffer.load_episodes(episode_dir)
        self._collect_load_time = time.time() - load_start
        if n_loaded == 0:
            # Setup validated len(npz_files) > 0, so reaching here means the
            # cache was deleted out from under us. Don't fall through to
            # fresh collection — that would defeat the explicit user opt-in
            # and waste minutes.
            raise RuntimeError(
                f"resume_from_collected_data: cache validated at setup but "
                f"load_episodes returned 0 from {episode_dir}. Likely cause: "
                f"the directory was deleted between setup() and train() "
                f"(filesystem race or external cleanup)."
            )

        # Reset the consecutive-failure counter, mirroring the success path
        # in _collect_episodes. A successful cached load is still a successful
        # collection from the trainer's standpoint.
        self._consecutive_collect_failures = 0
        print(f"  Loaded {n_loaded} episodes ({self.buffer.num_chunks} chunks)")

        # Defense-in-depth against heterogeneous cache corruption that the
        # validator's spot-check (which only inspects npz_files[0]) would
        # miss. Concrete scenario the spot-check passes but this catches:
        # the operator manually merged a pre-FM-instrumentation cache with
        # a post-FM cache — file 0 has all the keys, files 1+ don't. The
        # loader silently sets raw_action=None on the missing files, then
        # _prepare_batch silently drops those chunks at training time
        # (filter on `c.raw_action is not None`), producing a mostly-dead
        # minibatch with no warning. Surface it loudly here.
        n_chunks_total = self.buffer.num_chunks
        n_chunks_with_fm = sum(
            1
            for ep in self.buffer.episodes
            for raw, mask, noise in zip(
                ep.raw_actions, ep.action_masks, ep.initial_noises
            )
            if raw is not None and mask is not None and noise is not None
        )
        if n_chunks_with_fm < n_chunks_total:
            raise RuntimeError(
                f"resume_from_collected_data: validator spot-check passed "
                f"on {episode_dir.name}/episode_0000.npz, but only "
                f"{n_chunks_with_fm}/{n_chunks_total} loaded chunks have "
                f"raw_action / action_mask / initial_noise populated. This "
                f"usually indicates a manual cache merge across collector "
                f"versions, or a partial save that left some files with "
                f"truncated chunk data. Disable the flag and re-collect."
            )

        # Surface partial-collection (mirrors the post-load tail of
        # _collect_episodes via the shared _warn_partial_collection helper).
        # The cache may contain partial groups even after passing setup
        # validation, since partial groups are warned-not-raised there too.
        self._warn_partial_collection(n_loaded, source="cache")

    def _resolved_num_async_vector_env(self) -> int:
        """Physical AsyncVectorEnv worker count per group (config value, or
        group_size when unset). __post_init__ guarantees it divides group_size
        and is <= group_size."""
        return (
            self.config.group_size
            if self.config.num_async_vector_env is None
            else self.config.num_async_vector_env
        )

    def _turns_per_group(self) -> int:
        """Sequential collection turns needed to fill one group of group_size
        rollouts with num_async_vector_env physical envs (1 in the default
        one-env-per-rollout case)."""
        return self.config.group_size // self._resolved_num_async_vector_env()

    def _collect_via_subprocess(
        self,
        env_name: str,
        episode_dir: Path,
        max_steps: int,
        ff_steps: int,
    ) -> str | None:
        """Spawn `python collect_episodes.py` for one iteration's collection.

        Returns a failure_reason string, or None on success. Pays the full
        startup cost (robocasa imports + AsyncVectorEnv worker spawn) every
        call.
        """
        robocasa_python = str(
            Path(__file__).parent.parent.parent
            / "gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
        )
        collector_script = str(Path(__file__).parent / "collect_episodes.py")

        cmd = [
            robocasa_python,
            "-u",  # unbuffered: per-group lines appear in real-time
            collector_script,
            "--env-name", env_name,
            "--group-size", str(self.config.group_size),
            "--num-async-vector-env", str(self._resolved_num_async_vector_env()),
            "--num-groups", str(self.config.num_groups),
            "--max-episode-steps", str(max_steps),
            "--n-action-steps", str(self.config.n_action_steps),
            "--fast-forward-steps", str(ff_steps),
            "--fast-forward-pct", str(self.config.fast_forward_pct),
            "--server-host", self.config.server_host,
            "--server-port", str(self.config.server_port),
            "--output-dir", str(episode_dir),
            # Iter-stride 100_000 leaves room for collect_episodes.py to space
            # its `num_groups` group seeds by 1000 (see GROUP_SEED_STRIDE in
            # collect_episodes.py) without crossing into the next iter's seed
            # range. Safe for num_groups <= 100; num_groups=101 collides at
            # the iter boundary (iter N's last seed == iter N+1's first seed).
            "--seed", str(self.config.seed + self.iteration * 100_000),
            # Dynamic group collection (config-driven). When
            # min_alive_groups=0 in config, collector behaves identically
            # to the old fixed-num_groups path.
            "--min-alive-groups", str(self.config.min_alive_groups),
            "--max-groups", str(self.config.max_groups),
        ]

        # Optional saved-state override. Only append when set so the existing
        # CLI behavior (no flag → no override) is unchanged for baseline runs.
        if self.config.init_state_npz_path is not None:
            cmd.extend(
                ["--init-state-npz-path", self.config.init_state_npz_path]
            )

        # The collector skips intermediate-substep rendering by default, so only
        # the opt-out needs to cross the CLI boundary.
        if not self.config.skip_intermediate_render:
            cmd.append("--no-skip-intermediate-render")

        # Always pass the list explicitly (nargs="*", so an empty list becomes a
        # bare flag) — the collector's default and the config default must not be
        # able to drift apart silently.
        cmd.append("--dropped-video-keys")
        cmd.extend(self.config.dropped_video_keys)

        # Stream collector output line-by-line so the user sees progress
        # instead of waiting for the whole subprocess to finish. Mirror the
        # collector's stdout/stderr to the trainer log with a "[collector]"
        # prefix. A background Timer enforces the wall clock even if the
        # subprocess hangs on stdout with no output (otherwise the blocking
        # read could wait forever).
        # Scale subprocess timeout from the EFFECTIVE upper bound on groups
        # this iter (matches the RPC client's scaling at __init__). When
        # dynamic mode is disabled (min_alive_groups=0 or max_groups
        # equals num_groups), the collector stops at num_groups so there's
        # no need to grant the dynamic-mode worst-case 70-min budget.
        effective_max_groups = (
            self.config.max_groups
            if self.config.min_alive_groups > 0
            and self.config.max_groups > self.config.num_groups
            else self.config.num_groups
        )
        timeout_s = 420 * effective_max_groups * self._turns_per_group()  # 7 min/group/turn
        # When clean_output is on, propagate via env var because the
        # collector's import-time suppression must run BEFORE argparse
        # (otherwise robocasa import noise has already fired). Copy the
        # current env so the subprocess keeps PATH / PYTHONPATH /
        # MUJOCO_GL / etc.; subprocess.Popen(env=...) replaces, not merges.
        # AsyncVectorEnv workers spawned inside the collector inherit the
        # env var too, so [worker_mem pid=...] is also suppressed.
        sub_env = None
        if self.config.clean_output:
            sub_env = os.environ.copy()
            sub_env["GRPO_CLEAN_OUTPUT"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=sub_env,
        )
        import threading as _threading
        timed_out = {"v": False}
        def _kill_on_timeout():
            if proc.poll() is None:
                timed_out["v"] = True
                proc.kill()
        killer = _threading.Timer(timeout_s, _kill_on_timeout)
        killer.daemon = True
        killer.start()
        try:
            for line in proc.stdout:
                sys.stdout.write(f"    [collector] {line}")
                sys.stdout.flush()
            proc.wait()
        finally:
            killer.cancel()

        if timed_out["v"]:
            return f"timeout after {timeout_s}s (subprocess killed)"
        if proc.returncode != 0:
            return f"non-zero exit code {proc.returncode}"
        return None

    def _parse_resume_iteration(self) -> int:
        """Parse the iter number from `resume_from`'s basename, returning the
        next iter to run.

        Returns 1 for fresh runs (resume_from is unset) AND for non-canonical
        resume paths (e.g., 'best/', 'latest/') — the latter preserves
        backward compat for operators using non-iter-style names.

        When resume_from_collected_data is set, requires canonical
        `iter_NNNN/` naming (raises ValueError otherwise) so the cache
        validator can deterministically infer the iter number; without
        canonical naming the validator would silently fall back to
        iter_0001/, almost certainly the wrong cache.
        """
        if not self.config.resume_from:
            return 1
        dir_name = Path(self.config.resume_from).name
        m = ITER_DIR_RE.fullmatch(dir_name)
        if m:
            return int(m.group(1)) + 1
        if self.config.resume_from_collected_data:
            raise ValueError(
                f"resume_from_collected_data=True requires resume_from "
                f"to follow the canonical iter_NNNN/ naming pattern; "
                f"got {self.config.resume_from!r} (basename "
                f"{dir_name!r}). The cache validator infers the iter "
                f"number from this directory name, and a non-canonical "
                f"name would silently fall back to validating iter_0001/ "
                f"— almost certainly the wrong cache. Either rename the "
                f"checkpoint dir to iter_<integer> (e.g., iter_0050) or "
                f"unset --resume-from-collected-data to start fresh "
                f"collection."
            )
        return 1

    def _validate_collected_data_cache(self, iter_num: int) -> None:
        """Pre-flight check that cached episodes for `iter_num` are usable.

        Called from setup() when resume_from_collected_data=True. Fails LOUDLY
        on any mismatch so the operator can't silently train on a stale cache
        from a prior config (the alternative — falling through to fresh
        collection — would defeat the whole point of the flag, since the user
        explicitly asked to skip the ~7-min × num_groups collection cost).

        Validates (in order):
          1. Cache directory exists and contains episode_*.npz files.
          2. Per-file reads (single pass over all .npz):
              a. file is parseable; per-file dtypes for `success` and
                 `group_id` are sane (bool/int/uint/float for success;
                 int/uint for group_id). Catches a future collector that
                 saves these as strings or floats.
              b. `group_id` is REQUIRED; missing key (old format) raises.
              c. `env_name` matches the round-robin task that train()
                 would assign to iter_num — `(iter_num - 1) % len(env_names)`.
          3. Spot-check the first .npz for FM-log-prob keys
             (raw_action_*, action_mask_*, initial_noise_* per chunk),
             with `num_chunks` dtype validation and a `> 0` guard
             (`num_chunks <= 0` would silently make the loop empty AND
             break the within-group `Σ A_chunk = 0` invariant downstream).
          4. Group-count invariants:
              a. `n_obs >= num_groups` (dynamic mode produced enough groups);
              b. `n_obs <= max_groups` UNCONDITIONALLY (collector never
                 produces more than max_groups; an over-collected cache
                 indicates manual merge or config drift);
              c. `min_alive_groups` criterion satisfied OR
                 `n_obs == max_groups` (legitimate collector exit when
                 the task is too hard for the current policy).
          5. Group sizes equal config.group_size:
              a. UNDERCOUNT (size < group_size) warns but doesn't raise
                 (mirrors the trainer's existing partial-collection policy
                 for live-collected episodes);
              b. OVERCOUNT (size > group_size) RAISES — within-group
                 `env_seed` invariant is broken.

        On success, prints a one-line summary so the operator can confirm at
        a glance which iter is being skipped and what the cache looked like.
        """
        if not self.config.resume_from:
            # resume_from=None is a SUPPORTED mode, not an error: "fresh model,
            # reuse iter_0001's episodes". _parse_resume_iteration returns 1 in
            # that case, so iter_num is 1 here, and setup() skips both
            # load_lora_checkpoint and the optimizer restore — the policy is
            # freshly initialized. Iteration 1 collects BEFORE any
            # optimizer.step(), so its cached episodes are on-policy for that
            # fresh policy by construction (see GRPOConfig.__post_init__).
            #
            # Guard the one combination that would be silently wrong: reusing
            # some LATER iteration's cache with no checkpoint. Those episodes
            # were produced by a TRAINED policy, so consuming them with fresh
            # weights would train on genuinely off-policy data.
            if iter_num != 1:
                raise ValueError(
                    f"resume_from_collected_data=True without resume_from "
                    f"reuses episode_dir/iter_0001/ against a fresh model, "
                    f"but the resolved start iteration is {iter_num}. Those "
                    f"episodes were collected by a TRAINED policy and are "
                    f"off-policy for fresh weights. Either pass the matching "
                    f"--resume-from iter_NNNN/ checkpoint, or point "
                    f"--episode-dir at a cache whose iter_0001/ is the one "
                    f"you want to reuse."
                )
            print(
                "  resume_from_collected_data without resume_from: starting "
                "from a FRESH model and reusing iter_0001's episodes "
                "(on-policy — collected before any gradient step). Ensure "
                "seed / model_path / lora_rank / lora_alpha / "
                "lora_target_modules match the run that wrote the cache; "
                "those are not validated below."
            )

        cache_dir = Path(self.config.episode_dir) / f"iter_{iter_num:04d}"

        # Wrap is_dir() in try/except: when the parent has no read/exec
        # permission, Path.is_dir() raises PermissionError instead of
        # returning False. Without this guard, the operator sees a raw
        # OSError stack trace instead of the actionable wrap-around.
        try:
            is_dir = cache_dir.is_dir()
        except OSError as e:
            raise RuntimeError(
                f"resume_from_collected_data: failed to stat {cache_dir} "
                f"({type(e).__name__}: {e}). Check that "
                f"{self.config.episode_dir} and its ancestors are readable "
                f"by the trainer process."
            ) from e
        if not is_dir:
            raise FileNotFoundError(
                f"resume_from_collected_data=True but cache directory does not "
                f"exist: {cache_dir}. Either disable the flag (start fresh "
                f"collection) or check that the prior run's collection for "
                f"iter {iter_num} actually completed."
            )

        # Use os.listdir + manual filter instead of Path.glob: when the
        # directory is unreadable (chmod 000), Path.glob silently returns
        # an empty iterator and the validator falsely reports "Cache is
        # empty" — operator might `rm -rf` the cache trying to fix it.
        # listdir surfaces the PermissionError directly so the operator
        # sees a `chmod` hint rather than a misleading "empty" message.
        try:
            all_entries = os.listdir(cache_dir)
        except OSError as e:
            raise RuntimeError(
                f"resume_from_collected_data: failed to list {cache_dir} "
                f"({type(e).__name__}: {e}). Check directory permissions; "
                f"the cache exists but the trainer process cannot read it."
            ) from e
        npz_files = sorted(
            cache_dir / name
            for name in all_entries
            if name.startswith("episode_") and name.endswith(".npz")
        )
        if not npz_files:
            raise FileNotFoundError(
                f"resume_from_collected_data=True but no episode_*.npz files "
                f"in {cache_dir}. Cache is empty — disable the flag, or check "
                f"the prior collection actually wrote files (it may have "
                f"crashed before save_episodes())."
            )

        # The round-robin task selection here MUST match train()'s formula at
        # the head of the iteration loop (`task_idx = (iteration - 1) %
        # len(env_names)`). If those drift apart, this validator silently
        # accepts caches that train() will later complain about.
        expected_env = self.config.env_names[
            (iter_num - 1) % len(self.config.env_names)
        ]

        # Single pass over the cache: read just the small scalar fields
        # (env_name, group_id, success) per file. .npz is a zip directory, so
        # numpy reads the metadata header without decompressing the large
        # video/state/action arrays — this is cheap (~50ms for 25 files).
        #
        # `with np.load(...) as data:` releases the underlying zipfile +
        # file descriptor on EVERY exit path (success, raise, GC). The
        # bare `data = np.load(...)` form leaks the FD until refcount-GC
        # runs, which is fragile across interpreter changes and not
        # guaranteed if a field read raises mid-block.
        group_to_successes: dict[int, list[bool]] = defaultdict(list)
        for path in npz_files:
            try:
                with np.load(path, allow_pickle=True) as data:
                    # Wrap the field reads inside the same try/except so a
                    # partial/corrupted file (e.g., missing 'success') produces
                    # the helpful "cache may be corrupted" message instead of
                    # a raw KeyError that bubbles up unwrapped.
                    actual_env = (
                        str(data["env_name"])
                        if "env_name" in data.files
                        else None
                    )
                    # group_id is REQUIRED. Old caches that lack it would
                    # silently default to 0 and merge unrelated rollouts
                    # into one synthetic group, breaking the within-group
                    # env_seed invariant that GRPO advantage normalization
                    # relies on. Hard-fail rather than silently corrupt
                    # training signal.
                    if "group_id" not in data.files:
                        raise KeyError(
                            f"missing 'group_id' key — likely a pre-group-id "
                            f"collector format. Disable "
                            f"resume_from_collected_data and re-collect with "
                            f"the current trainer."
                        )
                    # Tight dtype validation. A future collector that saved
                    # group_id as np.float64 (e.g., a debug-mode patch)
                    # would silently truncate via `int()` (`int(2.5) == 2`,
                    # collapsing groups 2.0 and 2.5 onto gid 2). A string-
                    # serialized success ("True"/"False") would evaluate
                    # `bool("False") == True` (non-empty string is truthy)
                    # and silently mark every episode successful. Dtype
                    # checks make these silent regressions impossible.
                    gid_arr = data["group_id"]
                    if gid_arr.dtype.kind not in ("i", "u"):
                        raise TypeError(
                            f"group_id has unexpected dtype "
                            f"{gid_arr.dtype} (expected integer)"
                        )
                    gid = int(gid_arr)
                    if gid < 0:
                        raise ValueError(
                            f"group_id={gid} is negative; expected >= 0"
                        )
                    succ_arr = data["success"]
                    # bool ('b'), int ('i'/'u'), float ('f') are the
                    # historical save formats. Reject objects/strings.
                    if succ_arr.dtype.kind not in ("b", "i", "u", "f"):
                        raise TypeError(
                            f"success has unexpected dtype "
                            f"{succ_arr.dtype} (expected bool/int/float)"
                        )
                    success = bool(succ_arr)
            except Exception as e:
                raise RuntimeError(
                    f"resume_from_collected_data: failed to read {path}: "
                    f"{type(e).__name__}: {e}. Cache may be corrupted; "
                    f"disable the flag and re-collect."
                )
            if actual_env != expected_env:
                raise RuntimeError(
                    f"resume_from_collected_data: env_name mismatch in "
                    f"{path.name}. Expected {expected_env!r} (round-robin "
                    f"task for iter {iter_num} given env_names="
                    f"{self.config.env_names}); cached file has "
                    f"{actual_env!r}. Likely cause: env_names config changed "
                    f"between save and resume, or the resume point's iter "
                    f"index doesn't align with the cached task. Disable the "
                    f"flag, or pick a resume_from where iter_num modulo "
                    f"len(env_names) lands on the cached task."
                )
            group_to_successes[gid].append(success)

        # Spot-check the first .npz for the FM-log-prob keys. All files in
        # an iter come from the same collector run, so a single sample is
        # representative — checking every file would just multiply I/O for
        # zero added safety.
        try:
            with np.load(npz_files[0], allow_pickle=True) as sample:
                sample_keys = set(sample.files)
                num_chunks_arr = sample["num_chunks"]
                if num_chunks_arr.dtype.kind not in ("i", "u"):
                    raise TypeError(
                        f"num_chunks has unexpected dtype "
                        f"{num_chunks_arr.dtype} (expected integer)"
                    )
                num_chunks = int(num_chunks_arr)
        except Exception as e:
            raise RuntimeError(
                f"resume_from_collected_data: failed to read sample file "
                f"{npz_files[0]}: {type(e).__name__}: {e}. Cache may be "
                f"corrupted; disable the flag and re-collect."
            )
        # num_chunks <= 0 would make `range(num_chunks)` empty and the
        # FM-key loop trivially pass — but the file has no chunks at all,
        # which propagates downstream: `_load_single_episode` builds a
        # 0-chunk episode that consumes one advantage in compute_advantages
        # (per-group success vote) without contributing any chunks back to
        # _build_chunks. The within-group `Σ A_chunk = 0` invariant that
        # the GRPO clipped surrogate relies on is broken — fail loudly.
        if num_chunks <= 0:
            raise RuntimeError(
                f"resume_from_collected_data: cached sample file "
                f"{npz_files[0].name} has num_chunks={num_chunks}; expected "
                f"a positive integer. Cache may be corrupted; disable the "
                f"flag and re-collect."
            )
        for chunk_idx in range(num_chunks):
            for prefix in ("raw_action_", "action_mask_", "initial_noise_"):
                key = f"{prefix}{chunk_idx}"
                if key not in sample_keys:
                    raise RuntimeError(
                        f"resume_from_collected_data: cached episode "
                        f"{npz_files[0].name} is missing key {key!r}. The FM "
                        f"log-prob surrogate requires raw_action / "
                        f"action_mask / initial_noise per chunk (see "
                        f"collect_episodes.save_episodes). Likely cause: "
                        f"cache predates the GRPOPolicyWrapper instrumentation "
                        f"that captures these tensors. Disable the flag and "
                        f"re-collect."
                    )

        n_groups_observed = len(group_to_successes)
        # "Alive" group = mixed (0 < group_successes < group_size). Matches
        # the collector's exit criterion (collect_episodes.py:_collect) and
        # the trainer's IMPROVEMENT-signal filter. Deliberately still
        # mixed-only under include_anchor_groups: anchor (all-success) groups
        # train, but they carry no within-group contrast, so they cannot
        # substitute for a mixed group when deciding whether a cache has enough
        # gradient signal to be worth resuming from. Compare against `len(s)`
        # rather than config.group_size so partial groups (worker crashes
        # losing some rollouts) are evaluated by what's actually in the cache —
        # which is what compute_advantages will see. For full groups
        # (`len(s) == group_size`) the two are equivalent.
        n_alive_groups = sum(
            1 for s in group_to_successes.values()
            if 0 < sum(s) < len(s)
        )

        # Dynamic mode is allowed to have produced MORE groups than num_groups,
        # so the bound is `>=` not `==`. A cache with fewer groups than
        # configured means the prior collection either crashed mid-run or was
        # collected under a smaller num_groups setting — both cases want a
        # fresh collection.
        if n_groups_observed < self.config.num_groups:
            raise RuntimeError(
                f"resume_from_collected_data: cache has {n_groups_observed} "
                f"distinct group_ids in {cache_dir}, but config requires "
                f"num_groups={self.config.num_groups}. Likely cause: prior "
                f"collection crashed before completing all groups, or "
                f"num_groups was raised between save and resume. Disable the "
                f"flag and re-collect."
            )

        # Unconditional upper bound. The collector never produces more than
        # max_groups by construction (EpisodeCollector.collect's loop caps
        # at max_groups). A cache with `n_obs > max_groups` therefore
        # indicates either a manual merge across iters or that the cache
        # was collected under a LARGER max_groups setting than the current
        # config admits. Both cases break the operator's stated config —
        # reject loudly. This check runs BEFORE the min_alive_groups
        # gate so it fires regardless of how many groups were alive
        # (the min_alive gate is intentionally short-circuited when
        # `n_alive >= min_alive`, which would otherwise silently pass an
        # over-collected cache).
        if n_groups_observed > self.config.max_groups:
            raise RuntimeError(
                f"resume_from_collected_data: cache has {n_groups_observed} "
                f"groups in {cache_dir}, exceeding max_groups="
                f"{self.config.max_groups}. The collector never produces "
                f"more than max_groups; the cache was either manually "
                f"merged across iters, or collected under a larger "
                f"max_groups setting that has since been lowered. Disable "
                f"the flag and re-collect."
            )

        # min_alive_groups is satisfied either by hitting the explicit
        # alive criterion OR by hitting the max_groups cap exactly (the
        # collector itself terminates on either condition — see
        # EpisodeCollector.collect in collect_episodes.py). Match
        # that exact contract here so caches accepted by the collector are
        # also accepted by this validator.
        #
        # Use `!=` not `<`: a `<` check would silently accept a cache with
        # MORE than max_groups groups (e.g., user lowered max_groups between
        # save and resume). The cache was collected under a different
        # max_groups setting, so the collector's exit reasoning doesn't
        # apply — fail loudly. `==` allows the cap-hit escape; anything
        # else (more or less) means the cache and config disagree.
        if (
            self.config.min_alive_groups > 0
            and n_alive_groups < self.config.min_alive_groups
            and n_groups_observed != self.config.max_groups
        ):
            raise RuntimeError(
                f"resume_from_collected_data: cache has {n_alive_groups} "
                f"alive (mixed: 0 < group_successes < group_size) groups, "
                f"but config requires "
                f"min_alive_groups={self.config.min_alive_groups}. "
                f"Cache has {n_groups_observed} groups vs config "
                f"max_groups={self.config.max_groups} — neither the "
                f"min_alive criterion nor the max_groups cap was hit "
                f"exactly, so the prior collection terminated abnormally "
                f"(or was collected under a different max_groups). Disable "
                f"the flag and re-collect."
            )

        # Partial groups (worker crashes during the original collection) are
        # surfaced but NOT a hard failure — _collect_episodes handles partial
        # collections the same way (warn, don't increment failure counter).
        # Forcing a full re-collection here would diverge from that policy.
        #
        # OVERCOLLECTED groups (size > group_size) are different: they
        # indicate either a manual cache merge (two iters' episodes glued
        # into one dir) or a collector bug that re-ran a group. Either way
        # the within-group `env_seed` invariant that GRPO advantage
        # computation relies on is broken, so reject loudly.
        expected_size = self.config.group_size
        undercount = [
            (gid, len(s))
            for gid, s in group_to_successes.items()
            if len(s) < expected_size
        ]
        overcount = [
            (gid, len(s))
            for gid, s in group_to_successes.items()
            if len(s) > expected_size
        ]
        if overcount:
            raise RuntimeError(
                f"resume_from_collected_data: {len(overcount)} group(s) in "
                f"{cache_dir} have MORE than group_size={expected_size} "
                f"episodes: {overcount}. This usually indicates a manual "
                f"merge of two iters' caches into one dir, or a collector "
                f"bug that re-ran a group. The within-group env_seed "
                f"invariant that GRPO advantage normalization relies on is "
                f"broken — disable the flag and re-collect."
            )
        if undercount:
            for gid, sz in undercount:
                print(
                    f"  WARNING: cached group {gid} has {sz}/{expected_size} "
                    f"episodes — worker(s) likely failed during the original "
                    f"collection. Continuing with partial group."
                )

        print(
            f"  resume_from_collected_data: validated cache at {cache_dir} "
            f"({len(npz_files)} episodes, {n_groups_observed} groups, "
            f"{n_alive_groups} alive/mixed). Iter {iter_num} will "
            f"skip collection."
        )

    def _validate_optimizer_param_names(self, saved_names: list[str]) -> None:
        """Verify the saved optimizer's param order matches the current model.

        AdamW serializes its state by integer position (an index into
        param_groups[i]['params']), and load_state_dict re-attaches by the
        SAME positional index. With many same-shape LoRA matrices in the DiT
        (32 layers × 8 target modules → ~512 LoRA tensors, with most
        ``lora_A.default.weight`` shapes identical at ``(rank, in_features)``),
        a position permutation between save and load — caused by a peft or
        torch version bump altering module traversal order — would silently
        mis-attach Adam moments (exp_avg, exp_avg_sq) to the wrong tensors.
        The shape-based check in ``_validate_optimizer_state`` cannot catch
        this. Compare the persisted name list to the current order to detect
        such permutations.

        Raises with an actionable message identifying the first mismatched
        position so the operator can correlate against checkpoints.
        """
        current_names = self._lora_param_names
        if saved_names == current_names:
            return  # Exact match — safe to load.

        if len(saved_names) != len(current_names):
            raise RuntimeError(
                f"Optimizer parameter count mismatch on resume: saved "
                f"{len(saved_names)} params, current {len(current_names)}. "
                f"Likely cause: lora_target_modules or lora_rank differs from "
                f"the checkpoint."
            )

        # Same length, different order — find the first divergence so the
        # error message is actionable.
        for i, (sn, cn) in enumerate(zip(saved_names, current_names)):
            if sn != cn:
                same_set = set(saved_names) == set(current_names)
                raise RuntimeError(
                    f"Optimizer parameter ORDER mismatch on resume at position "
                    f"{i}: saved name = '{sn}', current name = '{cn}'. "
                    f"Name SETS are {'identical' if same_set else 'different'}. "
                    f"Likely cause: a peft or torch version bump changed the "
                    f"LoRA module traversal order between save and load. "
                    f"Loading would silently mis-attach Adam moments "
                    f"(exp_avg, exp_avg_sq) to the wrong LoRA tensors. Either "
                    f"pin peft/torch versions across save and load, or restart "
                    f"training from scratch."
                )

    def _validate_optimizer_state(self, saved: dict) -> None:
        """Verify a saved optimizer state_dict matches the current optimizer's param layout.

        Defense-in-depth shape and group-count check. The PRIMARY safeguard
        against silent positional mis-attachment lives in
        ``_validate_optimizer_param_names`` (compares the persisted name list
        to the current order). This method covers (a) legacy checkpoints
        without name metadata and (b) the case where names match but a shape
        regression slipped through (rank changed without a corresponding
        target_modules change).

        Saved state may be empty if the checkpoint was written before any
        optimizer.step() — that's fine, no validation needed.
        """
        saved_groups = saved.get("param_groups", [])
        curr_groups = self.optimizer.param_groups

        if len(saved_groups) != len(curr_groups):
            raise RuntimeError(
                f"Optimizer state mismatch on resume: saved has "
                f"{len(saved_groups)} param groups, current has "
                f"{len(curr_groups)}. Likely cause: LoRA architecture differs "
                f"between checkpoint and current config (rank, alpha, target_modules)."
            )

        for gi, (sg, cg) in enumerate(zip(saved_groups, curr_groups)):
            n_saved = len(sg.get("params", []))
            n_curr = len(cg["params"])
            if n_saved != n_curr:
                raise RuntimeError(
                    f"Optimizer param count mismatch in group {gi}: saved "
                    f"{n_saved} params, current {n_curr}. Likely cause: "
                    f"lora_target_modules differs from checkpoint."
                )

        # Shape check via Adam's exp_avg tensors. Empty state (no .step() yet)
        # is valid and skipped.
        saved_state = saved.get("state", {})
        if not saved_state:
            return

        for gi, (sg, cg) in enumerate(zip(saved_groups, curr_groups)):
            for i, (sid, cp) in enumerate(zip(sg["params"], cg["params"])):
                if sid not in saved_state:
                    continue  # this param was never stepped; nothing to validate
                exp_avg = saved_state[sid].get("exp_avg")
                if exp_avg is None:
                    continue
                if tuple(exp_avg.shape) != tuple(cp.shape):
                    raise RuntimeError(
                        f"Optimizer state shape mismatch at group {gi}, "
                        f"position {i}: saved exp_avg shape "
                        f"{tuple(exp_avg.shape)}, current param shape "
                        f"{tuple(cp.shape)}. This means the trainable parameter "
                        f"order changed between save and load (e.g., PEFT or "
                        f"PyTorch version bump altered module traversal order). "
                        f"Loading would silently mis-attach Adam moments to the "
                        f"wrong tensors. Either pin peft/torch versions across "
                        f"save and load, or restart training from scratch."
                    )

    def _prune_old_episode_dirs(self):
        """Delete iter_*/ subdirs older than (current_iter - keep + 1).

        Called from _collect_episodes BEFORE the current iter's directory is
        created, so we never risk deleting a dir we're about to read from.
        With episode_dirs_to_keep=3 at the start of iteration 10, prunes
        iter_0001..iter_0007 and leaves iter_0008, iter_0009 on disk — the
        soon-to-be-created iter_0010 brings the on-disk count to exactly
        `keep`.
        """
        keep = self.config.episode_dirs_to_keep
        if keep <= 0:
            return  # disabled
        base = Path(self.config.episode_dir)
        if not base.is_dir():
            return
        cutoff = self.iteration - keep + 1  # inclusive lower bound to keep
        n_pruned = 0
        for d in base.iterdir():
            if not (d.is_dir() and d.name.startswith("iter_")):
                continue
            try:
                n = int(d.name[len("iter_"):])
            except ValueError:
                continue  # not an iter_NNNN dir, skip
            if n < cutoff:
                # ignore_errors=True so a stale-handle ENOTEMPTY on one dir
                # doesn't prevent us from pruning the others.
                shutil.rmtree(d, ignore_errors=True)
                n_pruned += 1
        if n_pruned > 0:
            print(f"  Pruned {n_pruned} old episode dirs (kept last {keep})")

    def _compute_ref_log_probs(self):
        """Pre-compute reference log-probs for all chunks using the current model.

        This replaces the deep-copied reference model. Since this runs BEFORE the
        GRPO update, the current model IS the reference (it hasn't been updated yet
        for this iteration). We store per-chunk ref_log_prob and tau_samples so the
        GRPO update can reuse the exact same timesteps.

        This matches grpo_cont.py's pattern where logprob_old is collected from the
        current policy at the start of each iteration.
        """
        chunks = self.buffer._build_chunks()
        if not chunks:
            return

        # Drop chunks from dead groups (per-group std < 1e-4 → advantage = 0).
        # They get filtered again in _grpo_update before any forward pass, so
        # computing their ref log-probs here would be pure waste — and they
        # also wouldn't get encoded-feature cache entries that nothing
        # downstream would use. The advantage is set to literal `0.0` upstream
        # (episode_buffer.py, compute_advantages), so `== 0.0` would also work;
        # `abs(x) > 1e-12`
        # is defense-in-depth against any future change that introduces float
        # noise in the per-group normalization path.
        #
        # ANCHOR chunks are kept regardless of advantage: at anchor_advantage=0
        # they train on the KL terms alone, which still needs ref (and base)
        # log-probs. The same predicate is used in _grpo_update_inner, so the
        # two passes always agree on the row set. Both read the config gate
        # rather than trusting `is_anchor` alone, so a buffer flagged by an
        # earlier call (or a resumed run whose config changed) can never admit
        # anchor rows that the update would then treat as ordinary zero-advantage
        # chunks.
        n_total = len(chunks)
        use_anchors = self.config.include_anchor_groups
        chunks = [
            c for c in chunks
            if abs(c.advantage) > 1e-12 or is_anchor_row(c, use_anchors)
        ]
        n_live = len(chunks)
        if n_live < n_total:
            n_anchor = sum(1 for c in chunks if c.is_anchor)
            anchor_note = f" ({n_anchor} anchor chunk(s) kept)" if n_anchor else ""
            print(
                f"  Skipping ref log-prob pass for {n_total - n_live}/"
                f"{n_total} dead-group chunks (advantage == 0){anchor_note}."
            )
        if not chunks:
            # Unreachable: train()'s chunk-keyed skip returns before this pass
            # whenever there are no trainable chunks. Kept as a cheap guard in
            # case a future caller reaches the ref pass directly.
            return

        batch_size = self.config.mini_batch_size * 2  # Larger batches OK (no grad)
        K = len(self.config.tau_centers)
        noise_s = getattr(self.model.action_head.config, "noise_s", 0.999)

        # DiT is already in eval mode from setup / after _grpo_update; do not flip
        # modes here or the current-pass in _grpo_update will drift from ref.
        #
        # We use torch.no_grad() (not torch.inference_mode()) because this pass
        # ALSO caches per-chunk backbone/state features onto the chunks for
        # reuse during _grpo_update. inference_mode() produces tensors that
        # cannot participate in a later autograd graph, which would break
        # _grpo_update; no_grad tensors can be used freely as non-grad inputs.
        #
        # Take the model lock: the server thread is likely idle (collector
        # subprocess has finished), but a stuck/late ZMQ request would
        # otherwise race our forward pass.

        n_computed = 0
        compute_base = self.config.kl_coef_base_model > 0.0
        with self._model_lock, torch.no_grad():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                # Wrap as (chunk, "fixed") tuples — _prepare_batch's new
                # signature takes (chunk, mode) entries. The ref pass always
                # uses original ε for the DiT input regardless of jitter
                # settings (Jitter-GRPO anchors the cached ref at the original
                # ε so both fixed and jitter branches share the same baseline),
                # so "fixed" is the correct tag here.
                result = self._prepare_batch([(c, "fixed") for c in batch])
                if result is None:
                    continue
                batch_data, valid_batch = result

                B = batch_data["actions"].shape[0]

                # Sample jittered timesteps for this batch
                timesteps = _sample_jittered_timesteps(
                    tau_centers=self.config.tau_centers,
                    B=B,
                    noise_s=noise_s,
                    device=self.device,
                    dtype=torch.bfloat16,
                )  # [K, B]

                # Compute log-probs using current model (= reference before update)
                ref_lp = compute_fm_log_prob(
                    action_head=self.model.action_head,
                    backbone_output=batch_data["backbone_output"],
                    state_features=batch_data["state_features"],
                    embodiment_id=batch_data["embodiment_id"],
                    actions=batch_data["actions"],
                    action_mask=batch_data["action_masks"],
                    timesteps=timesteps,
                    noise=batch_data["initial_noise"],
                    n_samples=K,
                )

                # Optionally compute BASE-MODEL log-prob with LoRA adapters
                # disabled — same (τ, ε), same cached backbone features (the
                # backbone has no LoRA so its output is identical regardless
                # of adapter state). The disabled_adapters() context just
                # toggles every LoraLayer's _disable_adapters flag, so no
                # second model is loaded and peak VRAM is unchanged from the
                # ref-only pass. Skipped entirely when kl_coef_base_model=0
                # so vanilla runs incur no extra DiT forward.
                if compute_base:
                    with disabled_adapters(self.model.action_head.model):
                        base_lp = compute_fm_log_prob(
                            action_head=self.model.action_head,
                            backbone_output=batch_data["backbone_output"],
                            state_features=batch_data["state_features"],
                            embodiment_id=batch_data["embodiment_id"],
                            actions=batch_data["actions"],
                            action_mask=batch_data["action_masks"],
                            timesteps=timesteps,
                            noise=batch_data["initial_noise"],
                            n_samples=K,
                        )

                # --- Cache per-chunk encoded features for _grpo_update reuse ---
                # The Eagle backbone + state encoder are frozen, so their output
                # is identical across all GRPO epochs. We only need to run them
                # once per iteration (here) instead of once per minibatch.
                self._cache_encoded_features(valid_batch, batch_data)

                # Store ref log-prob and the (tau, eps)-samples used for it
                tau_cpu = timesteps.float().cpu().numpy()  # [K, B]
                for i, chunk in enumerate(valid_batch):
                    chunk.ref_log_prob = ref_lp[i].item()
                    chunk.tau_samples = tau_cpu[:, i].astype(np.float32)
                if compute_base:
                    for i, chunk in enumerate(valid_batch):
                        chunk.base_log_prob = base_lp[i].item()
                n_computed += len(valid_batch)

        if compute_base:
            print(
                f"  Pre-computed ref_log_probs + base_log_probs for "
                f"{n_computed} chunks"
            )
        else:
            print(f"  Pre-computed ref_log_probs for {n_computed} chunks")

        # Both diagnostics split by advantage SIGN, so they see signal chunks
        # only — an anchor row is neither "good" nor "bad" relative to its group.
        signal_chunks = [c for c in chunks if not is_anchor_row(c, use_anchors)]
        self._ref_mse_stats = self._summarize_ref_mse(signal_chunks, compute_base)
        # Per-chunk gap survey (Stage 1). Runs HERE because it needs ref_log_prob,
        # tau_samples and the cached encoded features all in place — which is
        # exactly the state at the end of this pass — and because measuring at
        # theta == theta_ref keeps the gap free of policy-drift contamination, same
        # as _jitter_gap_diagnostics. Off by default; see the config docstring.
        try:
            self._chunk_gap_stats = self._per_chunk_gap_survey(signal_chunks)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  WARNING: per-chunk gap survey failed "
                f"({type(exc).__name__}: {exc}) — skipping chunk_gap/* this "
                f"iteration. Training is unaffected."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._chunk_gap_stats = None

    def _summarize_ref_mse(self, chunks: list, compute_base: bool) -> dict | None:
        """Summarize MSE_ref (= -ref_log_prob) over this iteration's live chunks.

        Why this is worth a TB curve. The FM surrogate's importance ratio is

            log rho = MSE_ref(a) - MSE_theta(a),     MSE_theta >= 0

        so `rho <= exp(MSE_ref)` is a HARD upper bound, and MSE_ref is therefore
        the entire reinforcement headroom on a chunk: no amount of advantage
        weighting (PAWS / tratio / balanced sampling) can move a positive row's
        surrogate further than that. Two things follow that nothing else in the
        metric set surfaces:

        1. `ratio_ceiling_mean = mean(exp(MSE_ref))` says whether clip_eps_high
           is even reachable. If it reads ~1.01 while clip_eps_high=0.2, the
           upper clip is inert by construction, not by tuning.
        2. MSE_ref on POSITIVE-advantage chunks is the direct measure of
           positive-branch saturation. The FM loss is least-squares, so the
           gradient is proportional to the residual; as the DiT fits a
           successful chunk, MSE_ref -> 0 and the reinforcement gradient
           vanishes. A `ref_mse/pos_mean` curve decaying toward zero while
           success rate plateaus IS that saturation, and it is the signal to
           reach for a non-saturating term (e.g. a larger jitter_pos, whose
           Jacobian penalty does not vanish as MSE_ref does) rather than a
           larger positive weight.

        Split by the chunk's group-relative advantage sign — the same
        classification used for the per-sign clipfrac buckets and for jitter's
        per-row lambda — because the two signs answer different questions
        (headroom to reinforce vs. distance already travelled from a good fit).

        When the base-model anchor is active we also emit MSE_base - MSE_ref
        = log(base_ratio), the CUMULATIVE drift of the current field from the
        pretrained one, in the same units. Free: base_log_prob is already on the
        chunks. Unlike kl_loss_base_model this is not scaled by a coefficient,
        so it is readable as a distance rather than as a loss contribution.

        Returns None when there is nothing to summarize, which keeps the curves
        absent (a gap) rather than emitting a misleading 0 — same convention as
        the per-branch jitter metrics.
        """
        if not chunks:
            return None

        # MSE_ref = -ref_log_prob (compute_fm_log_prob returns negative masked
        # MSE averaged over the K tau samples). Chunks whose ref pass was
        # skipped (_prepare_batch returned None for their batch) have
        # ref_log_prob still None and are excluded rather than treated as 0.
        mse = np.array(
            [-c.ref_log_prob for c in chunks if c.ref_log_prob is not None],
            dtype=np.float64,
        )
        if mse.size == 0:
            return None
        pos = np.array(
            [
                -c.ref_log_prob
                for c in chunks
                if c.ref_log_prob is not None and c.advantage > 0
            ],
            dtype=np.float64,
        )
        neg = np.array(
            [
                -c.ref_log_prob
                for c in chunks
                if c.ref_log_prob is not None and c.advantage <= 0
            ],
            dtype=np.float64,
        )

        stats = {
            "mean": float(mse.mean()),
            "p10": float(np.percentile(mse, 10)),
            "p50": float(np.percentile(mse, 50)),
            "p90": float(np.percentile(mse, 90)),
            "max": float(mse.max()),
            # exp() of the per-chunk bound, averaged — i.e. the mean attainable
            # ratio ceiling. Averaging after exp (not exp of the mean) keeps it
            # comparable to train/ratio_max, which is also a per-row extreme.
            "ratio_ceiling_mean": float(np.exp(mse).mean()),
            "ratio_ceiling_max": float(np.exp(mse.max())),
        }
        if pos.size:
            stats["pos_mean"] = float(pos.mean())
        if neg.size:
            stats["neg_mean"] = float(neg.mean())

        if compute_base:
            drift = np.array(
                [
                    c.ref_log_prob - c.base_log_prob
                    for c in chunks
                    if c.ref_log_prob is not None and c.base_log_prob is not None
                ],
                dtype=np.float64,
            )
            # ref_log_prob - base_log_prob = MSE_base - MSE_ref = log(base_ratio).
            # POSITIVE => the adapted field fits the sampled action BETTER than
            # the pretrained one; negative => it has been eroded past base.
            if drift.size:
                stats["log_base_ratio_mean"] = float(drift.mean())
                stats["log_base_ratio_p10"] = float(np.percentile(drift, 10))
                stats["log_base_ratio_min"] = float(drift.min())

        print(
            f"  MSE_ref: mean={stats['mean']:.5f} "
            f"p10={stats['p10']:.5f} p90={stats['p90']:.5f} "
            f"→ ratio ceiling e^MSE_ref: mean={stats['ratio_ceiling_mean']:.4f} "
            f"max={stats['ratio_ceiling_max']:.4f} "
            f"(clip_eps_high={self.config.clip_eps_high} → "
            f"{'REACHABLE' if stats['ratio_ceiling_max'] > 1 + self.config.clip_eps_high else 'UNREACHABLE'})"
        )
        return stats

    def _cache_encoded_features(self, valid_batch, batch_data):
        """Store per-chunk slices of the batched backbone/state output onto
        each chunk, so _prepare_batch can rebuild batches without re-running
        the backbone. Called from within _compute_ref_log_probs' no_grad block.

        We slice each batch tensor along the batch axis and detach()+clone()
        so the chunk owns its own storage — the large batch tensor can then be
        garbage-collected once the batch goes out of scope.
        """
        backbone_features = batch_data["backbone_output"]["backbone_features"]      # [B, seq, D]
        backbone_attn_mask = batch_data["backbone_output"].get("backbone_attention_mask")
        image_mask = batch_data["backbone_output"].get("image_mask")
        state_features = batch_data["state_features"]                               # [B, state_hz, 1536]
        embodiment_id = batch_data["embodiment_id"]                                 # [B]

        for i, chunk in enumerate(valid_batch):
            # Unpad to the chunk's true seq_len using its attention mask.
            # This keeps per-chunk cache as small as the chunk actually needs,
            # instead of carrying the batch-level padding forever.
            if backbone_attn_mask is not None:
                mask_i = backbone_attn_mask[i]
                valid_len = int(mask_i.sum().item())
                # Verify the mask's 1s are a contiguous prefix (left-aligned
                # valid tokens, right-padded with 0s). If Eagle ever changes
                # to right-padding or interleaved valid/invalid tokens, the
                # slice below would silently keep the WRONG tokens (e.g.,
                # padding zeros instead of real features) and the cached
                # backbone features fed to the DiT during the GRPO update
                # wouldn't match what the policy actually saw at inference
                # time. Cheap to verify; load-bearing for correctness.
                if valid_len < mask_i.shape[0]:
                    prefix_ok = bool(mask_i[:valid_len].all().item())
                    suffix_ok = not bool(mask_i[valid_len:].any().item())
                    if not (prefix_ok and suffix_ok):
                        raise RuntimeError(
                            f"backbone_attn_mask[{i}] is not contiguous-prefix "
                            f"(sum={valid_len}, len={mask_i.shape[0]}). This "
                            f"trainer assumes left-aligned valid tokens; "
                            f"Eagle backbone padding side appears to have "
                            f"changed. Either fix the cache slicing here to "
                            f"index by mask, or align padding side upstream."
                        )
                chunk.cached_backbone_features = backbone_features[i, :valid_len].detach().clone()
                chunk.cached_backbone_attn_mask = backbone_attn_mask[i, :valid_len].detach().clone()
                if image_mask is not None:
                    chunk.cached_image_mask = image_mask[i, :valid_len].detach().clone()
                else:
                    chunk.cached_image_mask = None
            else:
                chunk.cached_backbone_features = backbone_features[i].detach().clone()
                chunk.cached_backbone_attn_mask = None
                chunk.cached_image_mask = (
                    image_mask[i].detach().clone() if image_mask is not None else None
                )

            chunk.cached_state_features = state_features[i].detach().clone()
            chunk.cached_embodiment_id = embodiment_id[i].detach().clone()

    def _grpo_update(self) -> dict:
        """Run GRPO clipped surrogate policy gradient update on collected episodes.

        Uses pre-computed ref_log_probs (from _compute_ref_log_probs) and stored
        tau_samples for each chunk. Only the current model's log-prob is computed
        with gradients enabled.

        The DiT stays in eval mode so that dropout (LoRA + attention) is consistent
        with the reference log-prob pass. If you want to enable dropout, match the
        mode between _compute_ref_log_probs and this method.

        Returns:
            Dict of update statistics (loss, clipfrac, kl, etc.)
        """
        # Hold the model lock for the entire update — no server-thread inference
        # requests can fire forward passes through the same model while we
        # accumulate/apply gradients (which would corrupt autograd state).
        # Re-entrant so the surrounding no-op is safe if called from a context
        # that already holds the lock.
        with self._model_lock:
            return self._grpo_update_inner()

    def _grpo_update_inner(self) -> dict:
        # Keep DiT in eval mode to match _compute_ref_log_probs; gradients still flow
        # through LoRA params because requires_grad is set at the parameter level.
        self.model.action_head.model.eval()

        # Build live-only chunks for the GRPO update. Dead-group chunks
        # (advantage == 0 from per-group normalization in episode_buffer.py)
        # would otherwise pollute training in two ways:
        #   1. Per-minibatch renormalization: `(0 - mean) / std` for a dead
        #      chunk picks up arbitrary magnitude from the live chunks'
        #      subsample mean — competes with real signal.
        #   2. Variable minibatch composition after a per-batch filter: a
        #      minibatch that randomly lands on N_live=1 falls through the
        #      `numel() > 1` renorm guard and contributes an un-normalized
        #      tiny gradient at a different scale than other minibatches.
        # Filtering at the buffer level (here) keeps every minibatch
        # uniformly-sized live-only.
        #
        # Live chunks split two ways (same predicate as _compute_ref_log_probs):
        #   - SIGNAL: mixed-group chunks with a group-relative advantage. These
        #     drive every existing mechanism (sampler pools, renorm statistics,
        #     dynamic epochs, PAWS mass, pos/neg metrics) exactly as before.
        #   - ANCHOR: all-success chunks (config.include_anchor_groups). Held
        #     separate so none of those mechanisms sees them; they enter each
        #     minibatch through a fixed quota instead. With anchors off,
        #     anchor_chunks is empty and everything below is bit-identical.
        all_chunks = self.buffer._build_chunks()
        use_anchors = self.config.include_anchor_groups
        # Module-level `is_anchor_row` (gated on the config) is the single anchor
        # predicate; every consumer below goes through it rather than reading
        # `c.is_anchor` directly, and _compute_ref_log_probs uses the same
        # expression so the two row filters cannot diverge.
        def _is_anchor(c) -> bool:
            return is_anchor_row(c, use_anchors)

        live_chunks = [
            c for c in all_chunks
            if abs(c.advantage) > 1e-12 or is_anchor_row(c, use_anchors)
        ]
        signal_chunks = [c for c in live_chunks if not _is_anchor(c)]
        anchor_chunks = [c for c in live_chunks if _is_anchor(c)]
        n_total_chunks = len(all_chunks)
        n_live_chunks = len(live_chunks)
        n_signal_chunks = len(signal_chunks)
        n_anchor_chunks = len(anchor_chunks)
        if n_live_chunks < n_total_chunks:
            print(
                f"  Filtering {n_total_chunks - n_live_chunks}/"
                f"{n_total_chunks} chunks with zero advantage (dead groups). "
                f"Remaining live chunks: {n_live_chunks}."
            )
        if n_anchor_chunks:
            print(
                f"  Anchor rows: {n_anchor_chunks} chunk(s) from "
                f"{len({c.group_id for c in anchor_chunks})} all-success "
                f"group(s) at advantage {self.config.anchor_advantage:g}"
                f"{' (KL-only)' if self.config.anchor_advantage == 0.0 else ''}."
            )
        if n_live_chunks == 0:
            return {}

        # Buffer-wide advantage stats for per-iteration normalization
        # (config.per_iteration_advantage_norm). Computed ONCE over the SIGNAL
        # chunks' per-chunk advantages (A_ep / num_chunks). The buffer mean is
        # ≈0 by group-relative construction (Σ A_ep = 0 within each group), so
        # dividing by the buffer std preserves each chunk's good/bad sign — see
        # grpo_config.per_iteration_advantage_norm. ddof=1 matches torch.std()
        # (used by the per-minibatch path) and episode_buffer.compute_advantages.
        # buffer_adv_std == 0.0 (from the <2-signal-chunk fallback) disables the
        # renorm below, mirroring the per-mb numel()>1 guard. Anchor chunks are
        # excluded from the statistic (their constant positive advantage would
        # lift the mean off zero and reintroduce sign flips) but DO consume it as
        # a scale — see anchor_scale.
        need_buffer_stats = (
            self.config.per_iteration_advantage_norm or bool(anchor_chunks)
        )
        if need_buffer_stats and n_signal_chunks > 1:
            _adv_arr = np.array([c.advantage for c in signal_chunks], dtype=np.float64)
            buffer_adv_mean = float(_adv_arr.mean())
            buffer_adv_std = float(_adv_arr.std(ddof=1))
        else:
            buffer_adv_mean = 0.0
            buffer_adv_std = 0.0

        # Scale applied to anchor rows' raw per-chunk advantage. Anchor rows
        # never see minibatch-local statistics: an anchor-only minibatch has no
        # variance except `anchor_advantage / num_chunks`, so a z-score there
        # would amplify pure EPISODE LENGTH to ±1 and reproduce the time-scaling
        # gradient that collapsed v2. Two regimes:
        #   - Signal rows present: divide by the same buffer-wide std they use,
        #     so the documented ratio (an anchor row is anchor_advantage/|A_sig|
        #     of a signal row) holds regardless of batch composition.
        #   - No signal rows at all (all-success iteration): normalize by the
        #     mean anchor magnitude so rows land at ~anchor_advantage, close to
        #     where the first regime puts them. No mean subtraction either way,
        #     so length variation stays a proportional ~2x spread instead of a
        #     z-scored ±1.
        if not anchor_chunks:
            anchor_scale = 0.0
        elif buffer_adv_std > 1e-8:
            anchor_scale = 1.0 / (buffer_adv_std + 1e-8)
        else:
            _mean_abs = float(
                np.mean([abs(c.advantage) for c in anchor_chunks])
            )
            anchor_scale = (
                self.config.anchor_advantage / _mean_abs if _mean_abs > 0.0 else 0.0
            )

        # Jitter-GRPO scheduling. When jitter is active (jitter_pos or
        # jitter_neg > 0), each chunk's jitter entry uses DiT input noise
        # ε' = sqrt(1-λ²)·ε + λ·ξ (λ = jitter_pos or jitter_neg per the chunk's
        # advantage sign). jitter_paired decides how many entries a chunk gets:
        #   - True  (default): "fixed" + "jitter" per chunk → 2× minibatches →
        #     2× optimizer steps. Halve update_epochs MANUALLY to match a
        #     vanilla per-iter step budget. Keeps the fixed-vs-jitter branch
        #     diagnostic.
        #   - False: "jitter" only → 1× minibatches, so the per-iter step count
        #     matches a vanilla run at the same update_epochs (directly
        #     comparable). No "fixed" rows → no `_fixed` per-branch metrics.
        # When BOTH strengths are 0 (default), jitter is off and jitter_paired
        # is N/A: behavior is bit-identical to pre-jitter code (single "fixed"
        # tag per chunk; ξ-sampling block below is skipped).
        if self.config.jitter_pos > 0.0 or self.config.jitter_neg > 0.0:
            if self.config.jitter_paired:
                entries = (
                    [(c, "fixed") for c in signal_chunks]
                    + [(c, "jitter") for c in signal_chunks]
                )
            else:
                entries = [(c, "jitter") for c in signal_chunks]
        else:
            entries = [(c, "fixed") for c in signal_chunks]

        # Anchor entries are ALWAYS "fixed" (never jittered): the Jacobian
        # regularizer is defined per advantage sign and anchor rows are a third
        # class, so jittering them would feed rows with no λ into the per-branch
        # accounting for no benefit.
        anchor_entries = [(c, "fixed") for c in anchor_chunks]

        # Per-minibatch anchor quota, subtracted from mini_batch_size rather than
        # added on top — total rows per minibatch stay at mini_batch_size so the
        # documented per-row VRAM budget is unchanged.
        #
        # `anchor_slots` is the integer per-batch reservation; it sizes the signal
        # batches (`signal_mb_size = mb_size - anchor_slots`) and caps how many
        # anchor rows a batch may receive. The per-batch TARGET is not computed
        # here — _with_anchor_rows derives it from the batch count the sampler
        # actually produced (see there). A target floored at 1 would make a small
        # anchor pool ride along in EVERY minibatch — 1 anchor chunk against 100
        # signal chunks would train ~15x per epoch while each signal row trains
        # once, the over-sharpening risk the anchor magnitude is kept small to
        # avoid — which is why the target is fractional.
        mb_size = self.config.mini_batch_size
        anchor_slots = 0
        signal_mb_size = mb_size
        # True whenever anchor rows participate in this iteration at all. Gates
        # the loss divisor below — NOT `has_anchor_rows` (whether a particular
        # minibatch happens to hold one). Because the quota is fractional, the
        # credit accumulator leaves some batches anchor-free; gating per batch
        # would send those through `.mean()` and reintroduce exactly the
        # composition-dependent row weight the constant divisor exists to
        # prevent (a 1-signal-row trailing batch would weight its row at 1.0
        # instead of 1/signal_mb_size).
        if anchor_entries and entries:
            # Reserve at least 1 signal slot (so the signal batch is never
            # empty) and at least 1 anchor slot (so a sub-1.0 target can still
            # land). At mini_batch_size=1 there is no room for both: skip the
            # interleave and say so rather than silently exceeding mb_size.
            #
            # CEIL, not round: the slot count is the per-batch CAP in
            # _with_anchor_rows, so rounding down would pin every batch at the
            # cap, let `credit` grow without bound, and under-deliver the target
            # by up to a third (target 3.4 -> cap 3 -> realized 3.0). Ceil keeps
            # the cap >= the target so the accumulator can average to it.
            # Reserve enough slots that the epoch can actually DELIVER the pool:
            # capacity is slots x n_batches, and n_batches depends on the reduced
            # signal batch size, so solve for the smallest slots that fits.
            # `ceil(mb_size * frac)` alone assumed the STRATIFIED batch count;
            # _iter_balanced_minibatches (the default) terminates early when its
            # majority pool drains, so the realized count is smaller, the target
            # exceeds the cap, and every batch gets pinned — measured 0.83-0.91x
            # delivery in the band where the ceil lands on 1.
            # _min_expected_batches is a LOWER bound on the batch count, so it can
            # only over-reserve (an extra optimizer step) and never under-reserve
            # (dropped rows); the wrapper measures the real count and warns if
            # capacity still fell short.
            anchor_slots = 1
            while anchor_slots < mb_size - 1:
                if (anchor_slots * self._min_expected_batches(
                        entries, mb_size - anchor_slots) >= len(anchor_entries)):
                    break
                anchor_slots += 1
            anchor_slots = min(anchor_slots, mb_size - 1)
            if anchor_slots < 1:
                print(
                    f"  WARNING: mini_batch_size={mb_size} leaves no room for an "
                    f"anchor row alongside a signal row — {len(anchor_entries)} "
                    f"anchor row(s) will NOT be trained this iter. Raise "
                    f"mini_batch_size to >= 2."
                )
                anchor_slots = 0
            else:
                # max() is redundant given the `mb_size - 1` clamp above, but it
                # keeps this invariant local: signal_mb_size == 0 would make the
                # inner samplers fall back to config.mini_batch_size (via
                # `mb_size or self.config.mini_batch_size`, where 0 is falsy) and
                # overfill every minibatch.
                signal_mb_size = max(1, mb_size - anchor_slots)
                # The per-batch anchor target is computed inside
                # _with_anchor_rows from the batch count the sampler ACTUALLY
                # produces. It must not be estimated here: ceil(len(entries) /
                # signal_mb_size) is the STRATIFIED count, and
                # _iter_balanced_minibatches — the default — terminates early
                # when its majority pool drains, so the estimate overshoots and
                # the pool under-delivers (a 1-chunk pool trained ZERO rows).
                print(
                    f"  Anchor schedule: {len(anchor_entries)} anchor row(s) "
                    f"spread over the epoch, <= {anchor_slots}/minibatch, "
                    f"alongside {signal_mb_size} signal row(s). Target row ratio "
                    f"{len(anchor_entries) / len(entries):.2f} anchor:signal; each "
                    f"anchor row's ADVANTAGE is ~anchor_advantage/|A_signal| of a "
                    f"signal row's, so the gradient share is smaller than the "
                    f"row share."
                )

        # True only when anchor rows will actually appear in minibatches. NOT
        # `bool(anchor_chunks)`: at mini_batch_size=1 the reservation clamps to 0
        # slots and the rows are skipped with a warning, and treating that
        # iteration as anchors-in-play would still divert every signal row into
        # the anchor-aware renorm branch (whose <2-signal-row fallback replaces
        # the plain path's numel()>1 renorm skip), silently changing their scale
        # for no anchor training at all. `not entries` is the anchor-only path,
        # where the rows go through the stratified sampler at full mb_size.
        anchors_in_play = bool(anchor_chunks) and (anchor_slots > 0 or not entries)

        total_loss = 0.0
        total_clip_loss = 0.0
        # Two KL accumulators: one per anchor target.
        #   - last_iter: KL(ref || current), where ref is the start-of-iter
        #     policy snapshot (always active when kl_coef_last_iter > 0).
        #   - base_model: KL(base || current), where base is the pretrained
        #     DiT (LoRA adapters disabled). Skipped when kl_coef_base_model=0
        #     so vanilla runs incur no extra compute or memory.
        total_kl_last_iter = 0.0
        total_kl_base_model = 0.0
        total_ratio = 0.0
        total_log_ratio_abs = 0.0
        clipfracs = []
        # Per-minibatch diagnostics — surface gradient magnitude and ratio
        # distribution tails. grad_norm answers "is there any signal hitting
        # the LoRA params?"; ratio_max/min reveal when a near-1 mean_ratio
        # hides outlier minibatches doing all the clipping work.
        # Under gradient accumulation (config.gradient_accumulation_steps > 1)
        # grad_norms holds ONE entry per optimizer step — the norm of the
        # ACCUMULATED (1/k-averaged) gradient — while ratio_maxes/ratio_mins
        # stay per-minibatch. Expect grad_norm_* to read lower at k > 1: that's
        # noise cancelling between micro-batches, not weaker signal.
        grad_norms: list[float] = []
        ratio_maxes: list[float] = []
        ratio_mins: list[float] = []
        # n_updates counts REAL optimizer.step() calls; n_micro_batches counts
        # minibatches that actually contributed a backward(). They're equal at
        # gradient_accumulation_steps=1 and differ by ~k above it. The split
        # matters twice over: train()'s `did_update` gates checkpoint naming on
        # n_updates (so it must mean "a step fired"), while every
        # per-minibatch mean below (loss, clip_loss, kl, mean_ratio,
        # mean_log_ratio_abs) divides by n_micro_batches so those curves stay
        # comparable across k instead of being inflated k-fold.
        n_updates = 0
        n_micro_batches = 0
        # Minibatches dropped for NaN/Inf loss. The guard fires BEFORE
        # backward(), so a dropped minibatch adds nothing to the gradient
        # buffer and — see the accumulation block below — does not advance the
        # accumulation window either. So this stays "minibatches whose
        # contribution was discarded", exactly as before accumulation existed.
        n_skipped_nonfinite = 0
        # Optimizer steps SKIPPED because the accumulated gradient was
        # non-finite (backward() produced inf/NaN even though the forward loss
        # was finite). Distinct from n_skipped_nonfinite, which counts
        # minibatches rejected at the forward guard. Non-zero here means the
        # window's weights update was dropped to avoid writing NaN into the
        # LoRA params — see _apply_accumulated_grads. Expected to stay 0.
        n_nonfinite_grad_steps = 0
        # Optimizer steps skipped because the accumulated gradient was exactly
        # zero — no learning signal, so the window is dropped rather than spending
        # an iteration on it. See _apply_accumulated_grads.
        n_zero_grad_steps = 0
        # Sign-flip diagnostic: count of group-good chunks (pre-renorm adv > 0)
        # that advantage renorm pushed to a non-positive z-scored value. >0 under
        # per-minibatch norm (the artifact); exactly 0 under per-iteration norm.
        n_pos_flipped = 0

        # Whether the base-model KL anchor is active this run. Cached locally
        # so the per-mb hot path skips the dict lookup. Drives the decision
        # to load base_log_probs and compute KL(base || current) below.
        compute_base = self.config.kl_coef_base_model > 0.0

        # Per-branch row-level accumulators (Jitter-GRPO). Aggregated metrics
        # above stay per-mb so the jitter-off path produces bit-identical
        # TB curves; per-branch metrics use row-weighted means since the
        # fixed/jitter row counts in a single minibatch can differ.
        ratio_sum_fixed = 0.0
        ratio_sum_jitter = 0.0
        log_ratio_abs_sum_fixed = 0.0
        log_ratio_abs_sum_jitter = 0.0
        kl_per_row_sum_last_iter_fixed = 0.0
        kl_per_row_sum_last_iter_jitter = 0.0
        kl_per_row_sum_base_model_fixed = 0.0
        kl_per_row_sum_base_model_jitter = 0.0
        n_rows_fixed = 0
        n_rows_jitter = 0
        # Clipfrac split into {fixed,jitter} × {pos,neg} buckets, where
        # pos/neg is the chunk's PRE-renormalization advantage sign (the
        # group-relative GRPO advantage from the buffer — i.e., "good
        # chunk we want to reinforce" vs "bad chunk we want to suppress").
        # The asymmetric clip (clip_eps_low ≠ clip_eps_high) only activates
        # on one side per advantage sign — upper bound for pos, lower for
        # neg — so splitting by sign separates the two clipping mechanisms
        # that share the combined `clipfrac` headline number.
        clipfrac_sum_fixed_pos = 0
        clipfrac_sum_fixed_neg = 0
        clipfrac_sum_jitter_pos = 0
        clipfrac_sum_jitter_neg = 0
        n_rows_fixed_pos = 0
        n_rows_fixed_neg = 0
        n_rows_jitter_pos = 0
        n_rows_jitter_neg = 0
        # Sign-split ratio accumulators. The existing mean_ratio_{fixed,jitter}
        # pool both advantage signs, and because the jitter gap scales as
        # lambda^2 the two signs sit at very different ratios (at
        # jitter_pos=0.25 / jitter_neg=0.05 the biases are ~-0.058 vs ~-0.002),
        # so the pooled curve is dominated by the positive rows and NEITHER
        # branch is legible. Split by sign and each becomes a real signal:
        #   *_pos starts each iteration at e^-gap_pos and its movement away from
        #         that value IS headroom being consumed — the only direct
        #         "is the positive branch learning?" readout that exists.
        #   *_neg starts at ~1.0 and moves down; that is erosion.
        # Reuses the n_rows_{branch}_{sign} counters above as divisors.
        ratio_sum_fixed_pos = 0.0
        ratio_sum_fixed_neg = 0.0
        ratio_sum_jitter_pos = 0.0
        ratio_sum_jitter_neg = 0.0
        log_ratio_abs_sum_fixed_pos = 0.0
        log_ratio_abs_sum_fixed_neg = 0.0
        log_ratio_abs_sum_jitter_pos = 0.0
        log_ratio_abs_sum_jitter_neg = 0.0
        # EFFECTIVE clipfrac: rows whose CLIP-TERM gradient the clamp zeroed.
        # This is not what `clipfrac` measures. `clipfrac` is the sign-agnostic
        # test `(ratio < 1-lo) | (ratio > 1+hi)`, which for a positive-advantage
        # row is a false positive: with A>0 and rho < 1-lo,
        # min(A*rho, A*(1-lo)) = A*rho — the unclamped branch wins and the clip
        # term is fully alive. That distinction is cosmetic today
        # (rho_max_observed ~ 1.05) but becomes load-bearing the moment
        # jitter_pos rises past ~0.30, where gap_pos exceeds |log(1-lo)| and
        # EVERY positive row reports as "clipped" while training normally.
        # Predicate lives in the module-level clip_killed_gradient() so tests
        # exercise it directly; see there for the four-case derivation and for
        # why "clip-term gradient" is the accurate phrasing (KL terms still flow).
        clipfrac_eff_sum_pos = 0
        clipfrac_eff_sum_neg = 0
        # Dedicated row counters for the effective clipfrac. Deliberately NOT
        # the n_rows_{branch}_{sign} counters above: those only accumulate when
        # jitter is active, and the effective clipfrac is meaningful for vanilla
        # GRPO too.
        n_rows_pos_total = 0
        n_rows_neg_total = 0
        # Anchor-row accumulators. Row-weighted (sum / n_rows), like the
        # per-branch jitter metrics — anchor rows are excluded from every
        # sign-keyed metric, so these are the only readout of what they did.
        n_rows_anchor = 0          # trained anchor rows (the reported count)
        n_rows_anchor_finite = 0   # divisor for the ratio/KL means (finite rows)
        ratio_sum_anchor = 0.0
        kl_per_row_sum_anchor = 0.0

        # Once-per-iteration jitter gap measurement (None until it runs; stays
        # None when jitter is off, which leaves the jitter/* curves absent).
        jitter_diag: dict | None = None

        # ── Balanced training: dynamic epoch count ───────────────────────────
        # When dynamic_epoch_training=True, scale update_epochs using a tent function
        # of the positive-advantage fraction among live-group episodes:
        #
        #   m    = min(successful_eps, total_eps − successful_eps)
        #   actual_epochs = max(1, (4·m·update_epochs + total_eps) // (2·total_eps))
        #
        # This is the exact integer form of floor(2·min(sf,1-sf)·E + 0.5), where
        # sf = successful_eps / total_eps and E = update_epochs. Integer arithmetic
        # avoids ULP cancellation in `1.0 − sf` which can make the float version
        # give the wrong result when (4·m·E + n) / (2·n) lands just below a
        # half-integer (e.g. n=24, m=7, E=6 → exact 3.5, float gives 3.4999…
        # → floor(3.9999…) = 3 instead of the correct 4).
        #
        # The tent peaks at success_frac=0.5 (→ full update_epochs) and decays
        # symmetrically toward both extremes:
        #   - Near 0% or 100% success: asymmetric advantages, least informative
        #     signal → 1 epoch
        #   - Near 50% success: balanced +/- signal, most informative → full
        #     update_epochs
        #
        # This replaces the old monotonic formula ceil(success_frac × update_epochs),
        # which pathologically gave MORE epochs at high success (70% → 3 epochs),
        # exactly when the gradient signal is most asymmetric and most likely to
        # cause policy overshoot. The tent reduces epochs at both extremes.
        #
        # We count only episodes from LIVE groups (those with at least one
        # non-zero-advantage chunk in live_chunks). Dead groups — all-success
        # or all-fail with std<1e-4 — produce no gradient signal, so including
        # their episodes would inflate success_frac and keep actual_num_epochs
        # near update_epochs even when real training signal is sparse. ANCHOR
        # groups are excluded for the same reason: every one of their episodes
        # succeeded, so counting them would drive success_frac toward 1 and
        # collapse the tent to 1 epoch exactly when anchors were added.
        #
        # We use episode-level advantage sign (self.buffer.advantages[i] > 0)
        # rather than ep.success, for consistency with _iter_balanced_minibatches
        # which oversamples chunks with c.advantage > 0. Under the sparse binary
        # reward these coincide for live groups (a mixed group's successes get
        # positive advantage, failures negative), so this is equivalent to
        # counting ep.success while keeping the two mechanisms aligned.
        if self.config.dynamic_epoch_training:
            live_group_ids = {c.group_id for c in signal_chunks}
            live_ep_indices = [
                i for i, ep in enumerate(self.buffer.episodes)
                # `not ep.is_anchor` is belt-and-braces: an anchor group is
                # all-success, so none of its episodes ever contributes a signal
                # chunk and the group_id test already excludes them.
                if ep.group_id in live_group_ids and not ep.is_anchor
            ]
            successful_eps = sum(
                1 for i in live_ep_indices
                if self.buffer.advantages is not None and self.buffer.advantages[i] > 0
            )
            total_eps_collected = max(len(live_ep_indices), 1)
            success_frac = successful_eps / total_eps_collected  # float, for logging only
            # Exact integer tent: (4·m·E + n) // (2·n) where m = min(k, n-k).
            # Avoids the ULP cancellation that can corrupt math.floor(float + 0.5)
            # at specific integer counts when update_epochs >= 6.
            if not live_ep_indices:
                # No signal episodes at all (an anchor-only iteration). The tent
                # is a function of the pos/neg BALANCE among signal episodes, and
                # there is none to measure: the empty-list fallback
                # (total=max(0,1)=1, successful=0) would collapse it to 1 epoch
                # and report success_fraction=0.0 on an iteration where every
                # episode succeeded. Anchor rows carry a fixed small magnitude
                # rather than an asymmetric advantage distribution, so the
                # mechanism has nothing to correct — run the configured epochs
                # and emit no success_fraction rather than a fabricated one.
                actual_num_epochs = self.config.update_epochs
                success_frac = None
                print(
                    f"  Dynamic epochs: no signal episodes (anchor-only iter) — "
                    f"tent not applicable, running "
                    f"{actual_num_epochs}/{self.config.update_epochs} epochs"
                )
                _tent_applied = False
            else:
                _tent_applied = True
            m = min(successful_eps, total_eps_collected - successful_eps)
            E = self.config.update_epochs
            n = total_eps_collected
            if _tent_applied:
                actual_num_epochs = max(1, (4 * m * E + n) // (2 * n))
            if _tent_applied:
                update_scale = 2.0 * m / n  # float tent scale, for the print only
                # Always print: silence looks like dynamic epoch scaling is off when
                # it's actually running at full capacity (epochs == update_epochs near peak).
                print(
                    f"  Dynamic epochs: {successful_eps}/{total_eps_collected} "
                    f"positive-advantage live-group episodes "
                    f"(tent scale={update_scale:.2f}) "
                    f"→ {actual_num_epochs}/{self.config.update_epochs} epochs"
                )
        else:
            actual_num_epochs = self.config.update_epochs
            success_frac = None  # Not computed; omit from stats

        # --- Dynamic positive-advantage weighting: per-iteration state ---
        # (config.positive_advantage_weight_scaling). N_iter/D_iter pool the
        # UNWEIGHTED alive-negative / positive loss mass across ALL minibatches
        # of this iteration; a light seed from last iter's EMA (prior * *_ema)
        # warm-starts k within the iteration. `scaling` and `have_prior` are
        # defined unconditionally (both gate the end-of-iter fold below).
        # have_prior requires the prior EMA to hold a USABLE positive-mass
        # denominator (D_ema > 0), not merely be non-None: a degenerate prior
        # iteration with zero alive-positive mass (D_ema == 0) would otherwise
        # leave D_seed == 0, and — since k now excludes the current minibatch's
        # own mass — the first positives-bearing minibatch could divide N by only
        # +eps and spuriously saturate k at the cap. Treating D_ema == 0 as
        # "not warmed up" runs a safe k=1 iteration that re-seeds the EMA instead.
        # (D_ema == 0 is unreachable under the intended per_iteration_advantage_norm
        # pairing — mixed live groups always yield alive positive mass — so this is
        # purely defensive for the off-label per-minibatch-norm combination.)
        scaling = self.config.positive_advantage_weight_scaling
        have_prior = (
            self._pos_scale_D_ema is not None and self._pos_scale_D_ema > 0.0
        )
        if scaling:
            N_seed = _POS_SCALE_PRIOR * self._pos_scale_N_ema if have_prior else 0.0
            D_seed = _POS_SCALE_PRIOR * self._pos_scale_D_ema if have_prior else 0.0
            N_iter = 0.0
            D_iter = 0.0
            k_last = 1.0

        # Effective balanced-sampler positive ratio this iter, for logging — the
        # same value _iter_balanced_minibatches will use (via _effective_pos_ratio).
        # None when balanced sampling is off or there are no entries. Under the
        # dynamic flag this rises toward balanced_minibatch_positive_adv_ratio_max
        # as success climbs, which is how you SEE the sampler back off failure
        # oversampling at high success.
        eff_pos_ratio = None
        if self.config.balanced_minibatch_training and entries:
            _nat_pos_frac = sum(1 for (c, _m) in entries if c.advantage > 0) / len(entries)
            eff_pos_ratio = self._effective_pos_ratio(_nat_pos_frac)

        # ── Gradient accumulation (config.gradient_accumulation_steps = k) ────
        # k minibatches per optimizer step. Peak VRAM is unchanged (each
        # minibatch's graph is still freed by its own backward()); what changes
        # is the update DIRECTION — k independently z-scored micro-batch
        # gradients averaged together — and the step count, which drops by ~k.
        #
        # Window protocol, per micro-batch that survives the non-finite guard:
        #   accum_count == 0  → zero_grad() (start a fresh window)
        #   backward(loss / k) → contribute 1/k of this micro-batch's gradient
        #   accum_count == k  → clip + step + reset (close the window)
        # and at the END OF EVERY EPOCH any partial window is flushed, so a
        # micro-batch's gradient is never silently dropped on the floor. That
        # flush matters here in particular because _iter_balanced_minibatches
        # anchors epoch length to ceil(len(entries) / mb_size) but can terminate
        # EARLY when the majority pool drains — the number of micro-batches per
        # epoch is not a fixed multiple of k, and neither is it known up front.
        #
        # accum_count advances only for minibatches that actually reached
        # backward(), so a non-finite-loss skip neither pollutes the buffer nor
        # shortens the window: every full window carries exactly k trained
        # micro-batches. PAWS mass (N_iter/D_iter) likewise commits per trained
        # micro-batch, independent of window boundaries, so the documented
        # "pooled mass == trained rows" invariant is untouched by k.
        accum_steps = self.config.gradient_accumulation_steps
        accum_count = 0        # micro-batches currently in the gradient buffer
        n_partial_windows = 0  # windows flushed with < k micro-batches

        def _apply_accumulated_grads() -> None:
            """Clip + step on the gradients accumulated so far, then reset.

            Called at a full window boundary and at each epoch's trailing
            flush. On the normal path: records ONE grad_norms entry (the
            accumulated-gradient norm) and bumps n_updates by exactly one real
            optimizer.step(). If the accumulated gradient is NON-FINITE the step
            is SKIPPED instead (see below) — n_updates does not move, so it
            always counts steps that actually reached the weights.
            """
            nonlocal accum_count, n_updates, n_nonfinite_grad_steps
            nonlocal n_zero_grad_steps

            # Gradient clipping. clip_grad_norm_ returns the TOTAL norm
            # of the gradient vector BEFORE clipping — capture it for
            # the train/grad_norm_* diagnostics (independent of whether
            # clipping actually fired this step).
            pre_clip_grad_norm = nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            gnorm = float(pre_clip_grad_norm)

            # Non-finite ACCUMULATED gradient → discard the window, don't step.
            #
            # `clip_grad_norm_` (error_if_nonfinite=False) returns inf/nan in two
            # situations, and neither is salvageable:
            #   - The gradient tensors themselves hold inf/NaN, because backward()
            #     overflowed even though the forward `loss` was finite (so the
            #     isfinite guard upstream passed) — e.g. a bf16 overflow inside
            #     the DiT backward. Clipping makes it worse, not better: with
            #     total_norm=inf the clip coefficient is 0, and inf * 0 = NaN, so
            #     by this point the buffer is all-NaN.
            #   - The gradients are finite but LARGE enough that the fp32
            #     sum-of-squares in the norm itself overflows (|g| above roughly
            #     1e17 for a LoRA-shaped tensor). Then clip_coef=0 multiplies a
            #     finite buffer down to exactly 0.0 — a no-op step carrying only
            #     AdamW momentum and weight decay. Also worth dropping.
            # Either way the window's gradient is gone before we get here, so
            # there is nothing to rescue by stepping.
            #
            # Stepping on it would write NaN into every LoRA parameter and
            # permanently poison AdamW's moments; every later minibatch would
            # then trip the forward guard, the iteration would still report
            # n_updates > 0 (so did_update=True), and a save_interval boundary
            # would persist a NaN checkpoint that a later --resume-from loads.
            # Meanwhile grad_norm_* would look NORMAL, because the offending
            # non-finite norm is excluded from grad_norms. That failure is
            # unrecoverable and its first clear symptom arrives long after the
            # corruption, so we drop the window instead:
            #   - zero the buffer (it holds NaN, nothing is salvageable),
            #   - leave the weights at their last good value,
            #   - count + warn so the event is visible rather than inferred.
            # Training continues with the next window. If EVERY window of an
            # iteration is dropped this way, n_updates stays 0 and train()'s
            # existing skip path treats the iteration as not-updated (model
            # unchanged, checkpoint written under the prior iter's name) — which
            # is exactly right, because the model IS unchanged.
            # A step whose gradient is EXACTLY zero carries no learning signal.
            # Reachable and not hypothetical: at LoRA init PEFT sets lora_B = 0, so
            # base == ref == current and on an anchor-only Layer-1 iteration
            # (anchor_advantage = 0) the clip term is 0 by construction and BOTH KL
            # anchors are KL(p||p) = 0 — every loss term vanishes. Counting such a
            # step as an update sets did_update=True, which burns the iteration
            # from num_iterations, writes a checkpoint named after it, advances the
            # LR schedule, and defeats the retry the skip path exists to preserve.
            # Same handling as a non-finite gradient: drop the window, don't step.
            # (AdamW's decoupled weight decay would still shrink the weights a
            # little, which is exactly what we do NOT want to spend an iteration
            # on.)
            if gnorm == 0.0:
                n_zero_grad_steps += 1
                self.optimizer.zero_grad()
                accum_count = 0
                return

            if not math.isfinite(gnorm):
                n_nonfinite_grad_steps += 1
                print(
                    f"  WARNING: non-finite accumulated gradient "
                    f"({gnorm}) — skipping this optimizer step and "
                    f"discarding the window ({accum_count} micro-batch(es)). "
                    f"LoRA weights left unchanged."
                )
                # Explicit zero even though the next window would zero anyway:
                # if this was the LAST window of the epoch, accum_count drops to
                # 0 and the epoch-boundary flush won't fire, so without this the
                # NaN buffer would sit there until the next iteration's first
                # window. Cheap and refactor-proof.
                self.optimizer.zero_grad()
                accum_count = 0
                return

            grad_norms.append(gnorm)
            self.optimizer.step()
            n_updates += 1
            accum_count = 0

        # Announce the accumulation schedule only when it's actually on, so
        # vanilla (k=1) runs keep byte-identical console output.
        if accum_steps > 1:
            print(
                f"  Gradient accumulation: {accum_steps} × "
                f"{self.config.mini_batch_size} rows = "
                f"{accum_steps * self.config.mini_batch_size} rows per "
                f"optimizer step (LR held fixed)"
            )

        for epoch in range(actual_num_epochs):
            # Stratified minibatch sampling: every minibatch contains
            # chunks from all live groups (best-effort) — see
            # _iter_stratified_minibatches docstring for full rationale.
            # We bypass buffer.iter_minibatches because that yields from
            # the full chunk list (live + dead) AND uses a flat shuffle
            # that doesn't preserve group structure; both are wrong here.
            # Seed scheme matches the prior iter_minibatches contract so
            # iteration-to-iteration RNG state remains comparable.
            rng = np.random.default_rng(
                self.config.seed + self.iteration * 100 + epoch
            )

            if not entries:
                # Anchor-only iteration (every group came back all-success).
                # Stratify the anchor rows themselves at full mini_batch_size:
                # the balanced sampler has no sign classes to balance here, and
                # the stratified path still yields each row exactly once/epoch.
                batch_iter = self._iter_stratified_minibatches(anchor_entries, rng)
            elif self.config.balanced_minibatch_training:
                batch_iter = self._with_anchor_rows(
                    self._iter_balanced_minibatches(entries, rng, signal_mb_size),
                    anchor_entries, anchor_slots, rng,
                )
            else:
                batch_iter = self._with_anchor_rows(
                    self._iter_stratified_minibatches(entries, rng, signal_mb_size),
                    anchor_entries, anchor_slots, rng,
                )

            for batch in batch_iter:
                # --- Prepare batch tensors ---
                result = self._prepare_batch(batch)
                if result is None:
                    continue
                batch_data, valid_batch = result

                actions = batch_data["actions"]           # [B, horizon, dim]
                action_masks = batch_data["action_masks"] # [B, horizon, dim]
                initial_noise = batch_data["initial_noise"]  # [B, horizon, dim] or None
                advantages = batch_data["advantages"]     # [B]
                backbone_output = batch_data["backbone_output"]
                state_features = batch_data["state_features"]
                embodiment_id = batch_data["embodiment_id"]
                modes = batch_data["modes"]               # list[str] of length B

                # --- Compute importance ratio ---
                # Use pre-computed ref_log_probs (from _compute_ref_log_probs)
                # and stored tau_samples for consistency. When the base-model
                # KL anchor is active, base_log_prob is also required (computed
                # in the same no_grad pass as ref_log_prob, so absence here is
                # only possible across config edits between save/load).
                # Build ready_indices directly instead of calling list.index(c):
                # ActionChunk is a @dataclass(eq=True) with ndarray fields, and
                # comparing numpy arrays raises "truth value is ambiguous", so
                # relying on .index() is fragile even if CPython's identity
                # short-circuit currently masks it.
                ready_indices = [
                    i for i, c in enumerate(valid_batch)
                    if c.ref_log_prob is not None and c.tau_samples is not None
                    and (not compute_base or c.base_log_prob is not None)
                ]
                if not ready_indices:
                    continue
                ready_batch = [valid_batch[i] for i in ready_indices]
                ready_modes = [modes[i] for i in ready_indices]
                # Anchor rows in this minibatch. Built once here and reused by
                # every sign-keyed mask below, all of which must treat anchors
                # as a third class rather than as positives.
                anchor_row_mask = torch.tensor(
                    [_is_anchor(c) for c in ready_batch],
                    device=self.device, dtype=torch.bool,
                )
                has_anchor_rows = bool(anchor_row_mask.any())
                signal_row_mask = ~anchor_row_mask
                n_signal_rows = int(signal_row_mask.sum())
                # Row divisor for clip_loss and both KL terms when anchor rows
                # are present: the INTENDED signal-row count (`signal_mb_size`),
                # held CONSTANT across the epoch.
                #
                # Why not the realized count. Dividing by the rows actually
                # present spikes any batch the sampler under-fills: a trailing
                # batch with 1 signal + 3 anchor rows would weight every row at
                # 1.0 instead of 1/signal_mb_size, making a 4-row batch the
                # largest step of the epoch (and, at max_grad_norm=0.5, the only
                # clipped one). A constant divisor makes a row's weight
                # independent of batch composition — an under-filled batch simply
                # contributes proportionally less, which is what it should do.
                #
                # Why the signal count and not the total. It keeps anchor rows
                # ADDITIVE: a signal row's weight is 1/signal_mb_size either way,
                # exactly what it would be in an anchor-free minibatch of that
                # size, so turning anchors on doesn't rescale the rows that drive
                # improvement — and the anchor KL genuinely ADDS a constraint
                # rather than reallocating the existing KL budget across more
                # rows. In the anchor-only path signal_mb_size == mini_batch_size,
                # so a full batch reduces to a plain mean. With no anchor rows
                # the expression below IS row_loss.mean().
                loss_divisor = float(max(signal_mb_size, 1))

                # If all chunks are ready, use tensors as-is (common case)
                if len(ready_batch) == len(valid_batch):
                    ready_actions = actions
                    ready_masks = action_masks
                    ready_noise = initial_noise
                    ready_advantages = advantages
                    ready_backbone = backbone_output
                    ready_state_features = state_features
                    ready_embodiment_id = embodiment_id
                else:
                    idx = torch.tensor(ready_indices, device=self.device)
                    ready_actions = actions[idx]
                    ready_masks = action_masks[idx]
                    ready_noise = initial_noise[idx] if initial_noise is not None else None
                    ready_advantages = advantages[idx]
                    ready_backbone = {
                        k: v[idx] if v is not None and hasattr(v, '__getitem__') else v
                        for k, v in backbone_output.items()
                    }
                    ready_state_features = state_features[idx]
                    ready_embodiment_id = embodiment_id[idx]

                ref_log_probs = torch.tensor(
                    [c.ref_log_prob for c in ready_batch],
                    device=self.device, dtype=torch.float32,
                )
                # Pre-load base_log_probs only when the base-model anchor is
                # active. Skipping the tensor allocation when disabled keeps
                # the vanilla path unchanged.
                if compute_base:
                    base_log_probs = torch.tensor(
                        [c.base_log_prob for c in ready_batch],
                        device=self.device, dtype=torch.float32,
                    )
                else:
                    base_log_probs = None

                # Reconstruct timesteps from stored per-chunk tau_samples.
                # Both copies of a paired chunk reuse the SAME tau_samples
                # and ref_log_prob (they were computed at the original ε).
                # Only the DiT input noise differs between fixed and jitter
                # rows — handled via noise_for_input below.
                tau_np = np.stack([c.tau_samples for c in ready_batch], axis=1)  # [K, B]
                timesteps = torch.from_numpy(tau_np).to(
                    device=self.device, dtype=torch.bfloat16
                )

                # --- Jitter-GRPO: build per-K input noise tensor ---
                # When any row in this minibatch is tagged "jitter", sample a
                # fresh ξ ~ N(0, I) of shape [K, B, H, D] and construct
                # noise_for_input[k, jitter_row] = sqrt(1-λ²)·ε + λ·ξ_k, where
                # λ is jitter_pos for positive-advantage rows and jitter_neg
                # for negative — so each sign gets its own Jacobian-penalty
                # strength. Fixed rows keep noise_for_input[:, fixed_row] = ε
                # (unchanged). The original ε (ready_noise) is still passed as
                # `noise=...` below so velocity_target = a − ε is anchored at
                # the original noise — that asymmetry is what makes the loss in
                # expectation an FM-loss + Frobenius-norm Jacobian regularizer
                # (the core Jitter-GRPO trick).
                lam_pos = self.config.jitter_pos
                lam_neg = self.config.jitter_neg
                if (
                    (lam_pos > 0.0 or lam_neg > 0.0)
                    and ready_noise is not None
                    and any(m == "jitter" for m in ready_modes)
                ):
                    K = len(self.config.tau_centers)
                    B_r, H, D = ready_noise.shape

                    # Unseeded: uses the global torch RNG, mirroring
                    # _sample_jittered_timesteps' τ-jitter sampling. We
                    # deliberately do NOT use a per-mb torch.Generator —
                    # training-time stochasticity (τ jitter, on-policy
                    # collection noise) isn't seeded per-call either.
                    xi = torch.randn(
                        K, B_r, H, D,
                        device=self.device, dtype=ready_noise.dtype,
                    )

                    jitter_mask_dev = torch.tensor(
                        [m == "jitter" for m in ready_modes],
                        device=self.device, dtype=torch.bool,
                    )

                    # Per-row λ selected by the chunk's PRE-renormalization
                    # advantage sign (same classification as the pos/neg
                    # clipfrac split below): jitter_pos for adv > 0, jitter_neg
                    # otherwise. Built in float32 so the scalar keeps full
                    # precision through the sqrt/multiply (matches the old
                    # single-lambda Python-float behavior). A row whose side is
                    # 0.0 collapses to ε — its jitter copy is then identical to
                    # the fixed row.
                    lam_row = torch.where(
                        ready_advantages > 0,
                        ready_advantages.new_full((B_r,), lam_pos, dtype=torch.float32),
                        ready_advantages.new_full((B_r,), lam_neg, dtype=torch.float32),
                    )
                    lam_j = lam_row[jitter_mask_dev]                    # [n_jit]
                    sqrt_one_minus_j = (1.0 - lam_j * lam_j).sqrt()     # [n_jit]

                    # expand returns a view — clone() is REQUIRED before
                    # __setitem__ to allocate writable per-K rows; without
                    # it the assignment would alias across the K dimension.
                    noise_for_input = (
                        ready_noise.unsqueeze(0).expand(K, -1, -1, -1).clone()
                    )
                    # Broadcast per-row λ over [K, n_jit, H, D]. The f32 math is
                    # cast back to ready_noise.dtype explicitly — masked
                    # index-put requires matching dtypes (it will NOT auto-cast
                    # a f32 source into a bf16 destination).
                    noise_for_input[:, jitter_mask_dev] = (
                        sqrt_one_minus_j[None, :, None, None]
                        * ready_noise[jitter_mask_dev].unsqueeze(0)
                        + lam_j[None, :, None, None] * xi[:, jitter_mask_dev]
                    ).to(ready_noise.dtype)

                    # ── Once-per-iteration jitter gap measurement ────────────
                    # Must be taken at theta == theta_ref, i.e. before any
                    # optimizer.step(), or the "gap" picks up policy drift and
                    # stops being a pure input-perturbation effect.
                    #
                    # `n_updates == 0` is exactly that condition: it is
                    # incremented only immediately after optimizer.step() in
                    # _apply_accumulated_grads, and the non-finite-gradient path
                    # there returns before BOTH the step and the increment. So
                    # n_updates == 0 <=> the weights are still the reference
                    # weights. Deliberately NOT `accum_count == 0` as well:
                    # accumulating a micro-batch does not move weights, so under
                    # gradient_accumulation_steps > 1 this correctly gets up to k
                    # chances to land the measurement before the first step.
                    #
                    # Why a retry loop matters at all: this whole block is
                    # conditional on the minibatch CONTAINING jitter rows, and
                    # under jitter_paired=True the entry pool is 50% "fixed", so
                    # an all-fixed first minibatch is possible (~0.4% at
                    # mini_batch_size=8). Without the n_updates guard that case
                    # would silently defer the measurement to a POST-step
                    # minibatch and report drift as gap. With it, we either
                    # measure pre-step or emit nothing (curve gap).
                    if jitter_diag is None and n_updates == 0:
                        # Pure instrumentation: a failure here must cost the
                        # metric, not the iteration. An iteration carries ~13
                        # minutes of collected simulation by the time it reaches
                        # this point, so a diagnostic-only exception (shape
                        # surprise on an odd minibatch, OOM on the extra
                        # forwards) must not discard it. Warn loudly and set a
                        # non-None sentinel so we don't retry every minibatch.
                        try:
                            jitter_diag = self._jitter_gap_diagnostics(
                                ready_backbone=ready_backbone,
                                ready_state_features=ready_state_features,
                                ready_embodiment_id=ready_embodiment_id,
                                ready_actions=ready_actions,
                                ready_masks=ready_masks,
                                ready_noise=ready_noise,
                                timesteps=timesteps,
                                noise_for_input=noise_for_input,
                                lam_row=lam_row,
                                pos_adv_mask=(ready_advantages > 0) & ~anchor_row_mask,
                                fixed_row_mask=(~jitter_mask_dev) & ~anchor_row_mask,
                                jitter_row_mask=jitter_mask_dev & ~anchor_row_mask,
                            )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"  WARNING: jitter gap diagnostics failed "
                                f"({type(exc).__name__}: {exc}) — skipping the "
                                f"jitter/* metrics for this iteration. Training "
                                f"is unaffected."
                            )
                            # If the failure was an OOM on the two extra
                            # forwards, hand the freed blocks back before the
                            # real minibatch runs — otherwise the diagnostic
                            # could turn a survivable iteration into a training
                            # OOM, which is the one way this instrumentation
                            # could still cost you a run.
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            jitter_diag = {}
                else:
                    noise_for_input = None

                # Only compute current model's log-prob (with gradient)
                current_log_probs = compute_fm_log_prob(
                    action_head=self.model.action_head,
                    backbone_output=ready_backbone,
                    state_features=ready_state_features,
                    embodiment_id=ready_embodiment_id,
                    actions=ready_actions,
                    action_mask=ready_masks,
                    timesteps=timesteps,
                    noise=ready_noise,
                    n_samples=len(self.config.tau_centers),
                    noise_for_input=noise_for_input,
                )

                log_ratio = current_log_probs - ref_log_probs
                ratio = log_ratio.exp()

                # --- Advantage renormalization ---
                # After the A_episode/num_chunks division in _build_chunks, per-chunk
                # advantages have small, heterogeneous magnitudes (varying with
                # episode length). Re-normalizing stabilizes gradient scale across
                # iterations and keeps the effective clip threshold meaningful
                # relative to the advantage magnitude.
                #
                # Two modes (config.per_iteration_advantage_norm):
                #   - False (default): PER-MINIBATCH z-score (matches
                #     grpo_cont.py:413-417). Subtracts the minibatch mean — which
                #     the balanced sampler biases off zero — so a genuinely-good
                #     chunk can be pushed to a negative z-scored advantage (sign
                #     flip) and trained as if bad.
                #   - True: BUFFER-WIDE z-score using buffer_adv_mean/std computed
                #     once over all live chunks (above). buffer_adv_mean ≈ 0
                #     (Σ A_ep = 0 per group), so sign is preserved (no flips) and a
                #     chunk's effective advantage is independent of its batchmates.
                #
                # The duplicate-handling note below applies to the per-minibatch path:
                # With Jitter-GRPO paired entries, a chunk's (fixed, jitter)
                # copies share the SAME advantage value, so the minibatch's
                # advantage tensor may have duplicates. Mean and std are still
                # well-defined; std stays positive as long as ≥2 distinct
                # advantages appear in the mb (best-effort under stratification —
                # if the residual mb collapses to a single distinct advantage
                # value, the +1e-8 epsilon zeroes out the renorm and that mb
                # contributes no gradient, same as pre-jitter behavior with
                # single-group minibatches). Note duplicates SHRINK the unbiased
                # std (Bessel correction overcounts independence), so renormalized
                # |advantages| are slightly LARGER per row in duplicate-heavy
                # minibatches — variance of the z-scored output is exactly 1 by
                # construction in either case. Net iter-wide gradient direction
                # is unchanged.

                # Capture advantage sign BEFORE per-mb renormalization for
                # the per-branch×sign clipfrac split. The renorm subtracts
                # the mb mean — for an all-positive-adv mb this puts half
                # the rows below zero, so a post-renorm sign would mean
                # "above/below mb mean" rather than "good/bad chunk". The
                # pre-renorm sign matches the buffer's group-relative GRPO
                # classification, which is what the diagnostic should
                # surface. Anchor rows are excluded: their advantage is a
                # constant, not a group-relative comparison, so they are
                # neither "good" nor "bad" in this sense.
                pre_renorm_pos_adv_mask = (ready_advantages > 0) & ~anchor_row_mask

                # Gated on anchors being enabled for the ITERATION, for the
                # same reason as loss_divisor: a batch the fractional quota left
                # anchor-free must still get the anchor-aware treatment, whose
                # <2-signal-row fallback is what keeps a lone row from entering
                # the surrogate at raw A_ep/num_chunks scale. When the batch has
                # no anchor rows the torch.where below is a no-op, so the branch
                # reduces to the plain path for a FULL batch; an under-filled one
                # is deliberately scaled down by the constant divisor.
                #
                # The two gatings differ only for a batch that is anchor-free AND
                # has <2 signal rows (plain leaves it raw; this path falls back to
                # the buffer-wide scale). That composition is COMMON, not a
                # defensive edge. The credit accumulator lands the final anchor
                # unit on the last batch, but does nothing about the interior
                # batches it leaves anchor-free, and _iter_stratified_minibatches
                # under-fills MID-epoch whenever its global filler order is
                # exhausted while a large group still has queued entries — the
                # skewed group sizes that dynamic collection produces. Measured
                # with groups (40,1,1,1) at mb_size=5: 35 of 40 minibatches are
                # anchor-free with one signal row. Gating per batch instead would
                # leave every one of those rows at raw A_ep/num_chunks scale.
                if not anchors_in_play:
                    if self.config.per_iteration_advantage_norm:
                        # Buffer-wide (per-iteration) z-score: subtract the iteration
                        # mean (≈0 by group-relative construction) and divide by the
                        # iteration std computed ONCE over all live chunks. Because
                        # buffer_adv_mean ≈ 0, sign is preserved → no good chunk flips
                        # to a negative z-scored advantage, and the effective clip
                        # threshold / gradient scale no longer depend on minibatch
                        # composition. buffer_adv_std == 0.0 (< 2 live chunks) skips it.
                        if buffer_adv_std > 1e-8:
                            ready_advantages = (
                                (ready_advantages - buffer_adv_mean)
                                / (buffer_adv_std + 1e-8)
                            )
                    elif ready_advantages.numel() > 1:
                        ready_advantages = (
                            (ready_advantages - ready_advantages.mean())
                            / (ready_advantages.std() + 1e-8)
                        )
                else:
                    # Mixed minibatch: renorm the signal rows exactly as above
                    # (statistics over signal rows ONLY — an anchor row's
                    # constant positive advantage would lift the mean and flip
                    # weak positives), and put anchor rows on the buffer-wide
                    # scale instead. See anchor_scale for why anchors must never
                    # touch minibatch-local statistics.
                    signal_row_mask = ~anchor_row_mask
                    scaled_anchor = ready_advantages * anchor_scale
                    if self.config.per_iteration_advantage_norm:
                        if buffer_adv_std > 1e-8:
                            signal_vals = (
                                (ready_advantages - buffer_adv_mean)
                                / (buffer_adv_std + 1e-8)
                            )
                        else:
                            signal_vals = ready_advantages
                    elif n_signal_rows > 1:
                        _sig = ready_advantages[signal_row_mask]
                        signal_vals = (
                            (ready_advantages - _sig.mean()) / (_sig.std() + 1e-8)
                        )
                    elif buffer_adv_std > 1e-8:
                        # Too few signal rows for a per-minibatch z-score. Fall
                        # back to the buffer-wide one — available whenever the BUFFER
                        # holds >= 2 signal chunks; with exactly one, buffer_adv_std
                        # is 0 and the row stays raw, there being no scale to borrow
                        # — instead of leaving the row RAW: a raw per-chunk
                        # advantage is ~1/num_chunks of a normalized one, so the
                        # row would enter the surrogate at a wildly different
                        # scale from every other minibatch's.
                        signal_vals = (
                            (ready_advantages - buffer_adv_mean)
                            / (buffer_adv_std + 1e-8)
                        )
                    else:
                        signal_vals = ready_advantages
                    ready_advantages = torch.where(
                        anchor_row_mask, scaled_anchor, signal_vals
                    )

                # --- Clipped surrogate loss ---
                surr1 = ready_advantages * ratio
                surr2 = ready_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_eps_low, 1 + self.config.clip_eps_high
                )
                # UNWEIGHTED per-row loss — measured for the dynamic weight BEFORE
                # weighting so the k estimate never feeds back on itself.
                row_loss = -torch.min(surr1, surr2)

                # Dynamic positive-advantage weight (config.positive_advantage_weight_scaling).
                # Deferred commit: measure the alive loss mass and pick k now (k is
                # needed to weight THIS minibatch), but fold the mass into the
                # pooled N_iter/D_iter — and hence the persistent EMA — ONLY after
                # the minibatch clears the non-finite-loss guard below. So a
                # minibatch dropped for non-finite loss (incl. a base-model KL
                # overflow while the ref ratio is finite) contributes no mass,
                # keeping "pooled mass == trained rows" exactly true.
                pending_pos_scale = None
                if scaling:
                    # Rows we amplify: group-good (pre-renorm adv > 0) AND still
                    # reinforcing after renorm (post > 0). Under per-iteration norm
                    # these coincide with the pre-renorm positives; the post-sign
                    # term only bites under per-minibatch norm (renorm flips),
                    # where it stops us amplifying a suppression term.
                    pos_amp_mask = pre_renorm_pos_adv_mask & (ready_advantages > 0)

                    # MEASURE (detached) this minibatch's ALIVE loss mass:
                    #   N = alive erosion = negative-adv rows whose gradient still
                    #       flows (DEAD iff ratio < 1 - clip_eps_low: the clamp
                    #       saturates and torch.min picks the constant branch).
                    #   D = alive reinforcement = exactly the rows we AMPLIFY that
                    #       still have gradient (DEAD iff upper-clipped, i.e.
                    #       ratio > 1 + clip_eps_high — rare here, but filtered so a
                    #       dead winner never inflates D). Keying D on pos_amp_mask
                    #       (not all pre-renorm positives) keeps the denominator
                    #       equal to the mass k actually scales.
                    # Anchor rows are in NEITHER mass: they are not an erosion
                    # term and not a group-relative reinforcement term, so
                    # letting them inflate D would drive k toward 1 and silently
                    # disable the mechanism at high success.
                    with torch.no_grad():
                        r_det = ratio.detach()
                        rl_abs = row_loss.detach().abs()
                        alive_neg_mask = (
                            (~pre_renorm_pos_adv_mask)
                            & (~anchor_row_mask)
                            & (r_det >= 1 - self.config.clip_eps_low)
                        )
                        amp_alive_mask = pos_amp_mask & (
                            r_det <= 1 + self.config.clip_eps_high
                        )
                        n_mass = float(rl_abs[alive_neg_mask].sum())
                        d_mass = float(rl_abs[amp_alive_mask].sum())
                    # k from the pool as finalized by prior TRAINED minibatches
                    # (this mb's own mass is folded in post-guard, not here), so a
                    # dropped minibatch never feeds k or the EMA. Mass is pooled per
                    # trained row — both jitter_paired branches and balanced-sampler
                    # duplicates count, since each applies a real gradient.
                    if not have_prior:
                        k = 1.0  # first update (fresh or post-resume): warm up
                    else:
                        k = min(max(
                            self.config.positive_advantage_weight_target_ratio
                            * (N_seed + N_iter) / (D_seed + D_iter + _POS_SCALE_EPS),
                            1.0,
                        ), self.config.positive_advantage_weight_max)
                    # Stage the mass for post-guard commit (only if finite, so a
                    # ratio overflow can never poison the pool or the EMA).
                    if math.isfinite(n_mass) and math.isfinite(d_mass):
                        pending_pos_scale = (k, n_mass, d_mass)
                    # row_weight is exactly k on amplified rows, 1.0 elsewhere.
                    # (Weighting an upper-clipped positive would be a no-op — its
                    # gradient is already zero — so pos_amp_mask needs no alive
                    # filter for the weighting itself.)
                    row_weight = 1.0 + (k - 1.0) * pos_amp_mask.to(row_loss.dtype)
                    row_loss = row_weight * row_loss

                clip_loss = (
                    row_loss.mean() if not anchors_in_play
                    else row_loss.sum() / loss_divisor
                )

                # --- KL divergence penalties (Schulman k3 estimator) ---
                # KL(p || q) ≈ E[exp(p_lp - q_lp) - (p_lp - q_lp) - 1] with the
                # log-prob of the ANCHOR target on the left. Identity:
                # e^x - x - 1 ≥ 0 for all x, with equality iff x=0. Properties
                # vs the naive (anchor - current).mean():
                #   - Non-negative POINTWISE, not just in expectation.
                #   - Minimum at current ≡ anchor → gradient pulls policies
                #     together symmetrically (the naive estimator's gradient
                #     was one-sided and could *reward* current >> anchor).
                #   - Same expected value (still estimates KL(anchor||current)).
                #   - Lower variance.
                # See Schulman 2020 "Approximating KL Divergence" for derivation.
                #
                # Two anchor targets, summed into total loss:
                #   - last_iter: ref_log_probs (this iter's start-of-update
                #     snapshot). Bounds per-iter drift.
                #   - base_model: base_log_probs (pretrained DiT, LoRA off).
                #     Bounds CUMULATIVE drift from the pretrained policy.
                # Per-row tensors are kept around (not just the mean) so the
                # per-branch fixed/jitter accumulator below can split each
                # term separately. Both terms use loss_divisor for the same
                # additive-anchor reason as clip_loss above.
                inv_log_ratio = ref_log_probs - current_log_probs  # = -log_ratio
                kl_per_row_last_iter = inv_log_ratio.exp() - inv_log_ratio - 1.0
                kl_loss_last_iter = self.config.kl_coef_last_iter * (
                    kl_per_row_last_iter.mean() if not anchors_in_play
                    else kl_per_row_last_iter.sum() / loss_divisor
                )

                if compute_base:
                    inv_log_ratio_base = base_log_probs - current_log_probs
                    kl_per_row_base_model = (
                        inv_log_ratio_base.exp() - inv_log_ratio_base - 1.0
                    )
                    kl_loss_base_model = self.config.kl_coef_base_model * (
                        kl_per_row_base_model.mean() if not anchors_in_play
                        else kl_per_row_base_model.sum() / loss_divisor
                    )
                else:
                    kl_per_row_base_model = None
                    kl_loss_base_model = torch.zeros(
                        (), device=self.device, dtype=torch.float32
                    )

                # --- Total loss ---
                loss = clip_loss + kl_loss_last_iter + kl_loss_base_model

                # NaN/Inf guard: a single bad batch (e.g., bf16 overflow in
                # ratio = log_ratio.exp() when log_ratio is large, or NaN
                # creeping in from numerical edge cases in the backbone)
                # would otherwise propagate through optimizer.step() and
                # silently corrupt the LoRA weights. clip_grad_norm_ does NOT
                # rescue NaN gradients; it only bounds finite norms.
                # Skip this minibatch and log a counter instead.
                #
                # Accumulation policy: skip ONLY this micro-batch's
                # contribution — never the whole window. The guard fires before
                # backward(), so nothing non-finite ever enters the gradient
                # buffer, and the window's already-accumulated (finite,
                # perfectly good) micro-batches are kept rather than thrown
                # away. accum_count is deliberately NOT advanced below, so the
                # window absorbs the next finite micro-batch instead and still
                # closes on k TRAINED micro-batches. `continue` is safe with
                # respect to the trailing flush because the flush lives outside
                # this loop, at the epoch boundary.
                if not torch.isfinite(loss):
                    n_skipped_nonfinite += 1
                    continue

                # Minibatch survived the non-finite guard → commit its dynamic-
                # weight mass and k. Done here (not in the measure block) so a
                # dropped minibatch never folds its mass into N_iter/D_iter or the
                # cross-iteration EMA. Unaffected by accumulation: mass pools per
                # TRAINED micro-batch, which is exactly the set of rows that
                # reach backward() below, regardless of where window boundaries
                # land. ONE rare exception: if a window is later dropped for a
                # non-finite accumulated gradient, its rows' mass is already
                # pooled even though no gradient reached the weights, so the pool
                # over-counts by up to k micro-batches. Accepted rather than
                # staged per window — it only occurs on an anomaly that already
                # warrants investigation, and k is a RATIO so the effect is
                # second-order (n_nonfinite_grad_steps surfaces it).
                if pending_pos_scale is not None:
                    k_last, _n_mass, _d_mass = pending_pos_scale
                    N_iter += _n_mass
                    D_iter += _d_mass

                # --- Backward pass (gradient accumulation window) ---
                # zero_grad ONLY at the start of a window; otherwise this
                # micro-batch's gradient would wipe its predecessors'. At
                # accum_steps=1 that's every micro-batch, i.e. the original
                # zero_grad → backward → clip → step sequence.
                if accum_count == 0:
                    self.optimizer.zero_grad()
                # Scale by 1/k so the accumulated buffer holds the MEAN
                # micro-batch gradient, keeping the gradient magnitude (and
                # therefore the effective step size under max_grad_norm
                # clipping) on the same scale as an un-accumulated run.
                # accum_steps == 1 short-circuits to `loss` itself rather than
                # `loss / 1`: mathematically identical, but it keeps the
                # autograd graph and the resulting gradients bit-identical to
                # the pre-accumulation code path with no extra div node.
                if accum_steps == 1:
                    loss.backward()
                else:
                    (loss / accum_steps).backward()
                accum_count += 1

                if accum_count == accum_steps:
                    _apply_accumulated_grads()

                # --- Track statistics ---
                with torch.no_grad():
                    # Fraction of rows the clamp actually moved. With asymmetric
                    # bounds this is the OR of the two one-sided clips rather than
                    # a single |ratio - 1| > eps threshold.
                    #
                    # This and the mean_ratio / mean_log_ratio_abs accumulators
                    # below cover ALL trained rows, anchor rows included — they
                    # describe the batch the optimizer saw, and anchor rows are
                    # part of it. For the signal-only view use
                    # clipfrac_effective_{pos,neg} (sign-bucketed, anchors
                    # excluded) and mean_ratio_anchor for the anchor split.
                    clipfrac = (
                        (ratio < 1 - self.config.clip_eps_low)
                        | (ratio > 1 + self.config.clip_eps_high)
                    ).float().mean().item()
                    clipfracs.append(clipfrac)
                    total_loss += loss.item()
                    total_clip_loss += clip_loss.item()
                    total_kl_last_iter += kl_loss_last_iter.item()
                    if compute_base:
                        total_kl_base_model += kl_loss_base_model.item()
                    total_ratio += ratio.mean().item()
                    # log_ratio magnitude is the primary diagnostic for DPPO-style
                    # FM log-prob surrogates: large values mean the MSE-based
                    # log-prob is noisy enough that most updates will clip, which
                    # caps the effective gradient signal.
                    total_log_ratio_abs += log_ratio.abs().mean().item()
                    # Ratio distribution tails — when mean_ratio≈1 but
                    # clipfrac jumps, the tail values are doing the clipping.
                    # bf16 `ratio = log_ratio.exp()` can overflow to +inf
                    # even when the clipped loss stays finite (clamp bounds
                    # the loss but not the raw ratio). Filter the same way
                    # as grad_norms to keep TB charts clean.
                    rmax = ratio.max().item()
                    rmin = ratio.min().item()
                    if math.isfinite(rmax):
                        ratio_maxes.append(rmax)
                    if math.isfinite(rmin):
                        ratio_mins.append(rmin)
                    # Counts TRAINED minibatches, not optimizer steps (see the
                    # n_updates / n_micro_batches split above). This is the
                    # divisor for every per-minibatch mean in `result`, so those
                    # curves are invariant to gradient_accumulation_steps.
                    n_micro_batches += 1
                    # Sign-flip diagnostic: group-good chunks (pre-renorm adv>0)
                    # that renorm pushed to <= 0. Nonzero under per-minibatch norm;
                    # 0 under per-iteration norm (buffer_mean≈0 preserves sign) and
                    # also 0 whenever renorm was skipped (< 2 live chunks / std≈0).
                    n_pos_flipped += int(
                        (pre_renorm_pos_adv_mask & (ready_advantages <= 0)).sum().item()
                    )

                    # --- EFFECTIVE clipfrac (clip-term gradient zeroed) -------
                    # See the accumulator declarations for why this is not the
                    # same as `clipfrac`, and clip_killed_gradient() for the
                    # four-case derivation and the KL-term caveat.
                    #
                    # Bucketed by the POST-renorm advantage sign, unlike the
                    # sibling clipfrac_{branch}_{sign} metrics which use the
                    # pre-renorm (group-relative) sign. That is deliberate:
                    # which BOUND a row can die on is decided by the sign of the
                    # advantage the LOSS saw, i.e. post-renorm. Under the default
                    # per-minibatch renorm a group-good row can z-score negative
                    # (counted as n_pos_flipped_by_renorm); bucketing such a row
                    # as "pos" would put a lower-bound death into
                    # clipfrac_effective_pos and break the one property that
                    # makes this metric useful — that _pos stays at 0 unless the
                    # ratio genuinely exceeded 1+clip_eps_high.
                    grad_dead = clip_killed_gradient(
                        ratio, surr1, surr2,
                        self.config.clip_eps_low, self.config.clip_eps_high,
                    )
                    # Anchor rows sit in neither bucket — they have no
                    # group-relative sign, and pooling them into _pos would make
                    # the curve a mix of two different row classes.
                    post_renorm_pos_mask = (ready_advantages > 0) & ~anchor_row_mask
                    post_renorm_neg_mask = (ready_advantages <= 0) & ~anchor_row_mask
                    n_p = int(post_renorm_pos_mask.sum().item())
                    n_n = int(post_renorm_neg_mask.sum().item())
                    if n_p > 0:
                        clipfrac_eff_sum_pos += int(
                            grad_dead[post_renorm_pos_mask].sum().item()
                        )
                        n_rows_pos_total += n_p
                    if n_n > 0:
                        clipfrac_eff_sum_neg += int(
                            grad_dead[post_renorm_neg_mask].sum().item()
                        )
                        n_rows_neg_total += n_n

                    # --- Anchor-row accumulation ---
                    # Row-weighted, and the only place anchor rows are measured.
                    # mean_ratio_anchor is the readout that matters: it starts at
                    # 1.0 and rises toward 1 + clip_eps_high as the anchor term
                    # pulls; it saturating there means the clip is capping the
                    # per-iteration retention move, which is the intended bound.
                    if has_anchor_rows:
                        n_a = int(anchor_row_mask.sum().item())
                        n_rows_anchor += n_a
                        # Ratio/KL sums accumulate over FINITE rows only, with
                        # their own divisor: `ratio = log_ratio.exp()` can
                        # overflow to +inf while the clipped loss stays finite,
                        # and a single inf in a running sum poisons the curve for
                        # the whole run. Same policy as ratio_maxes/ratio_mins
                        # above and the sign-split mean_ratio_* metrics.
                        _r_a = ratio[anchor_row_mask]
                        _k_a = kl_per_row_last_iter[anchor_row_mask]
                        _fin = torch.isfinite(_r_a) & torch.isfinite(_k_a)
                        n_finite_a = int(_fin.sum().item())
                        if n_finite_a > 0:
                            n_rows_anchor_finite += n_finite_a
                            ratio_sum_anchor += _r_a[_fin].sum().item()
                            kl_per_row_sum_anchor += _k_a[_fin].sum().item()

                    # --- Per-branch row-level accumulation (Jitter-GRPO) ---
                    # Only runs when jitter is enabled. Gating on jitter-active
                    # makes the jitter-off path bit-identical at the metrics
                    # layer (no `_fixed`/`_jitter` curves emitted, no extra
                    # per-mb CUDA syncs from .item() calls).
                    #
                    # Aggregation note: legacy aggregated metrics (clipfrac,
                    # mean_ratio, mean_log_ratio_abs, kl_loss_last_iter,
                    # kl_loss_base_model above) are means-of-per-mb-means.
                    # The per-branch versions emitted below are ROW-WEIGHTED
                    # (sum / n_rows). The two differ when minibatch sizes vary
                    # (e.g., last mb smaller than mb_size). The fixed-vs-
                    # jitter gap on mean_log_ratio_abs (and on each of the
                    # sign-split clipfrac_{branch}_{pos,neg} pairs) IS the
                    # empirical Jacobian-norm signal that Jitter-GRPO is
                    # designed to surface — if it shrinks across iters, the
                    # regularizer is working.
                    if lam_pos > 0.0 or lam_neg > 0.0:
                        over_clip = (
                            (ratio < 1 - self.config.clip_eps_low)
                            | (ratio > 1 + self.config.clip_eps_high)
                        ).float()
                        log_ratio_abs = log_ratio.abs()
                        # Anchor rows are always tagged "fixed" but belong to
                        # neither branch — strip them so mean_ratio_fixed stays a
                        # pure signal-row curve.
                        fixed_mask = torch.tensor(
                            [m == "fixed" for m in ready_modes],
                            device=self.device, dtype=torch.bool,
                        ) & ~anchor_row_mask
                        jit_mask = (~fixed_mask) & ~anchor_row_mask
                        # Pre-renorm sign masks for the 4-way clipfrac split.
                        # See pre_renorm_pos_adv_mask capture above.
                        neg_adv_mask = (~pre_renorm_pos_adv_mask) & ~anchor_row_mask
                        fixed_pos_mask = fixed_mask & pre_renorm_pos_adv_mask
                        fixed_neg_mask = fixed_mask & neg_adv_mask
                        jit_pos_mask = jit_mask & pre_renorm_pos_adv_mask
                        jit_neg_mask = jit_mask & neg_adv_mask

                        n_f = int(fixed_mask.sum().item())
                        n_j = int(jit_mask.sum().item())
                        if n_f > 0:
                            ratio_sum_fixed += ratio[fixed_mask].sum().item()
                            log_ratio_abs_sum_fixed += log_ratio_abs[fixed_mask].sum().item()
                            kl_per_row_sum_last_iter_fixed += kl_per_row_last_iter[fixed_mask].sum().item()
                            if compute_base:
                                kl_per_row_sum_base_model_fixed += kl_per_row_base_model[fixed_mask].sum().item()
                            n_rows_fixed += n_f
                        if n_j > 0:
                            ratio_sum_jitter += ratio[jit_mask].sum().item()
                            log_ratio_abs_sum_jitter += log_ratio_abs[jit_mask].sum().item()
                            kl_per_row_sum_last_iter_jitter += kl_per_row_last_iter[jit_mask].sum().item()
                            if compute_base:
                                kl_per_row_sum_base_model_jitter += kl_per_row_base_model[jit_mask].sum().item()
                            n_rows_jitter += n_j
                        # Clipfrac split — accumulate independently per
                        # (branch, adv_sign) bucket. Each bucket may be
                        # empty in a given mb (e.g., a stratified mb that
                        # happens to draw only positive-adv chunks for the
                        # fixed branch), so each .sum().item() is gated on
                        # its own row count to skip the CUDA sync when 0.
                        n_fp = int(fixed_pos_mask.sum().item())
                        n_fn = int(fixed_neg_mask.sum().item())
                        n_jp = int(jit_pos_mask.sum().item())
                        n_jn = int(jit_neg_mask.sum().item())
                        if n_fp > 0:
                            clipfrac_sum_fixed_pos += int(over_clip[fixed_pos_mask].sum().item())
                            ratio_sum_fixed_pos += ratio[fixed_pos_mask].sum().item()
                            log_ratio_abs_sum_fixed_pos += log_ratio_abs[fixed_pos_mask].sum().item()
                            n_rows_fixed_pos += n_fp
                        if n_fn > 0:
                            clipfrac_sum_fixed_neg += int(over_clip[fixed_neg_mask].sum().item())
                            ratio_sum_fixed_neg += ratio[fixed_neg_mask].sum().item()
                            log_ratio_abs_sum_fixed_neg += log_ratio_abs[fixed_neg_mask].sum().item()
                            n_rows_fixed_neg += n_fn
                        if n_jp > 0:
                            clipfrac_sum_jitter_pos += int(over_clip[jit_pos_mask].sum().item())
                            ratio_sum_jitter_pos += ratio[jit_pos_mask].sum().item()
                            log_ratio_abs_sum_jitter_pos += log_ratio_abs[jit_pos_mask].sum().item()
                            n_rows_jitter_pos += n_jp
                        if n_jn > 0:
                            clipfrac_sum_jitter_neg += int(over_clip[jit_neg_mask].sum().item())
                            ratio_sum_jitter_neg += ratio[jit_neg_mask].sum().item()
                            log_ratio_abs_sum_jitter_neg += log_ratio_abs[jit_neg_mask].sum().item()
                            n_rows_jitter_neg += n_jn

            # ── Epoch boundary: flush a partial accumulation window ──────────
            # Anything still in the gradient buffer belongs to trained
            # micro-batches, so dropping it would silently discard real signal.
            # A flush is REQUIRED here (not merely tidy) because the number of
            # micro-batches per epoch is not a multiple of k in general:
            # _iter_balanced_minibatches anchors epoch length to
            # ceil(len(entries) / mb_size) and returns early when the majority
            # pool drains, and the non-finite guard can drop micro-batches
            # anywhere in the epoch. Consequence of the uniform 1/k scale: this
            # step's gradient is (m/k)× the window average for m < k
            # micro-batches — a proportionally smaller step, at most one per
            # epoch. Guarded on accum_count > 0 so an epoch that trained nothing
            # (every minibatch filtered or non-finite) never steps on an empty
            # or stale buffer. When it does fire, accum_count is necessarily in
            # [1, k-1] — a window that reached k already closed itself above —
            # so this is always a genuinely partial window, and at k=1 the
            # branch is unreachable (every micro-batch closes its own window).
            # n_partial_windows counts flush ATTEMPTS: if this one is then
            # dropped for a non-finite accumulated gradient it also lands in
            # n_nonfinite_grad_steps, and n_updates (the source of truth for real
            # steps) does not move.
            if accum_count > 0:
                n_partial_windows += 1
                _apply_accumulated_grads()

        # Model remains in eval mode (it never left)
        # n_updates counts steps that actually reached the weights, so
        # n_updates == 0 means "the model is bit-identical to the start of this
        # iteration" — exactly what train()'s `did_update` gate needs. Two ways
        # to get here: nothing trained at all (no live chunks, every minibatch
        # filtered or non-finite), or every window was dropped for a non-finite
        # accumulated gradient (n_nonfinite_grad_steps > 0 with
        # n_micro_batches > 0). The second case is why this is NOT the same as
        # "n_micro_batches == 0" — report both counters so the operator can tell
        # the two apart instead of seeing a bare empty dict. The divisions below
        # stay safe either way: a step requires at least one backward(), so
        # n_updates > 0 implies n_micro_batches > 0.
        if n_updates == 0:
            early: dict = {}
            if n_skipped_nonfinite:
                early["n_skipped_nonfinite"] = n_skipped_nonfinite
            if n_zero_grad_steps:
                # Reported HERE above all: a zero-gradient window is the reason
                # this iteration has n_updates == 0, so the counter is most
                # needed on exactly this path.
                early["n_zero_grad_steps"] = n_zero_grad_steps
                early["n_micro_batches"] = n_micro_batches
                print(
                    f"  No optimizer step survived this iter: "
                    f"{n_zero_grad_steps} window(s) had an exactly-zero gradient "
                    f"(no learning signal) over {n_micro_batches} trained "
                    f"minibatch(es). Model unchanged; iteration preserved."
                )
            if n_nonfinite_grad_steps:
                early["n_nonfinite_grad_steps"] = n_nonfinite_grad_steps
                early["n_micro_batches"] = n_micro_batches
                print(
                    f"  No optimizer step survived this iter: "
                    f"{n_nonfinite_grad_steps} window(s) dropped for non-finite "
                    f"gradients over {n_micro_batches} trained minibatch(es). "
                    f"Model unchanged."
                )
            if self.config.dynamic_epoch_training:
                early["actual_epochs"] = actual_num_epochs
                if success_frac is not None:
                    early["success_fraction"] = success_frac
            # Diagnostics that are still VALID on a no-step iteration, and are
            # most valuable exactly here:
            #   - the jitter gap was measured at theta == theta_ref, before any
            #     step could have fired, so discarding it because the update was
            #     later thrown away loses a reading that is still correct. It is
            #     also the most likely EXPLANATION for landing in this branch: a
            #     large gap means a large |log_ratio|, and bf16 `ratio =
            #     log_ratio.exp()` overflowing is precisely what trips the
            #     non-finite-loss guard.
            #   - the effective clipfracs come from micro-batches that actually
            #     trained, so they are populated in the dropped-gradient case
            #     (n_micro_batches > 0) and simply absent in the all-non-finite
            #     case, which is the right behaviour in both.
            if jitter_diag:
                early["_jitter_diag"] = jitter_diag
            if n_rows_pos_total > 0:
                early["clipfrac_effective_pos"] = (
                    clipfrac_eff_sum_pos / n_rows_pos_total
                )
            if n_rows_neg_total > 0:
                early["clipfrac_effective_neg"] = (
                    clipfrac_eff_sum_neg / n_rows_neg_total
                )
            return early

        if n_skipped_nonfinite > 0:
            print(
                f"  WARNING: skipped {n_skipped_nonfinite} minibatch(es) for "
                f"non-finite loss (NaN/Inf) — likely bf16 ratio overflow"
            )
        if n_nonfinite_grad_steps > 0:
            print(
                f"  WARNING: dropped {n_nonfinite_grad_steps} optimizer step(s) "
                f"for non-finite ACCUMULATED gradients (backward-side overflow; "
                f"the forward loss was finite). LoRA weights were protected, but "
                f"investigate — this is not expected."
            )

        # Per-minibatch means divide by n_micro_batches (NOT n_updates): with
        # gradient_accumulation_steps=k these differ by ~k, and dividing by the
        # step count would scale every loss/ratio/KL curve up by k — a pure
        # logging artifact that would look like a k-fold jump in loss the moment
        # accumulation is switched on. That keeps these curves on the same SCALE
        # across k; it does not make them bit-identical across k (within a window
        # all k micro-batches see the same un-stepped weights, so the log-probs
        # differ from a k=1 run by a few percent). grad_norm_* is the deliberate
        # exception: one sample per optimizer step, over the accumulated gradient.
        result = {
            "loss": total_loss / n_micro_batches,
            "clip_loss": total_clip_loss / n_micro_batches,
            "kl_loss_last_iter": total_kl_last_iter / n_micro_batches,
            "clipfrac": np.mean(clipfracs) if clipfracs else 0,
            "mean_ratio": total_ratio / n_micro_batches,
            "mean_log_ratio_abs": total_log_ratio_abs / n_micro_batches,
            "n_updates": n_updates,
            "n_micro_batches": n_micro_batches,
            "n_skipped_nonfinite": n_skipped_nonfinite,
            # Optimizer steps dropped to protect the weights from a non-finite
            # accumulated gradient. Emitted unconditionally (even at 0) so the
            # TB curve is a flat zero line you can glance at, rather than a
            # missing series you'd have to know to look for.
            "n_nonfinite_grad_steps": n_nonfinite_grad_steps,
            "n_zero_grad_steps": n_zero_grad_steps,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
            "ratio_max": float(np.max(ratio_maxes)) if ratio_maxes else 1.0,
            "ratio_min": float(np.min(ratio_mins)) if ratio_mins else 1.0,
            "actual_epochs": actual_num_epochs,
            "n_pos_flipped_by_renorm": n_pos_flipped,
        }
        # Accumulation diagnostics, only when accumulation is actually on (k=1
        # runs keep exactly the key set they had before, so no new constant-
        # valued TB curves appear on vanilla runs). n_partial_windows counts
        # end-of-epoch flushes that closed with < k micro-batches — each of
        # those took a proportionally smaller step. It is bounded ABOVE by
        # actual_epochs (at most one flush per epoch, see the flush site), so it
        # simply reports how many epochs had a trained-micro-batch count that
        # wasn't a multiple of k. It is NOT a drop detector: a non-finite skip
        # can make the remainder divide evenly and LOWER this counter. Watch
        # n_skipped_nonfinite for mid-epoch drops.
        if accum_steps > 1:
            result["grad_accum_steps"] = accum_steps
            result["n_partial_windows"] = n_partial_windows
        # Effective balanced-sampler positive ratio (only when balanced sampling
        # ran with entries present). With the dynamic flag it tracks success.
        if eff_pos_ratio is not None:
            result["balanced_pos_ratio"] = eff_pos_ratio
        # Anchor-row metrics. Emitted only when anchor rows actually trained, so
        # runs with include_anchor_groups=False keep exactly their prior key set.
        if n_rows_anchor > 0:
            result["n_anchor_rows_trained"] = n_rows_anchor
            # Emitted only when at least one anchor row had a finite ratio, and
            # only if the resulting mean is itself finite — a non-finite value is
            # DROPPED (leaving a curve gap) rather than written, matching the
            # policy for every other ratio-derived metric here. Note this is a
            # per-anchor-row mean, whereas kl_loss_last_iter in the LOSS divides
            # by signal_mb_size; the two are not comparable side by side.
            if n_rows_anchor_finite > 0:
                _mr = ratio_sum_anchor / n_rows_anchor_finite
                _kl = (self.config.kl_coef_last_iter
                       * kl_per_row_sum_anchor / n_rows_anchor_finite)
                if math.isfinite(_mr):
                    result["mean_ratio_anchor"] = _mr
                if math.isfinite(_kl):
                    result["kl_loss_anchor"] = _kl
        # --- Dynamic positive-advantage weighting: fold this iter's pooled
        # masses into the cross-iteration EMA and surface k. Reached only on the
        # success path (n_updates > 0 is guaranteed by the early return above),
        # so a dead iteration never drags the EMA toward zero. have_prior was
        # snapshotted at iteration start: on the first-ever update it's False, so
        # the EMA is seeded from this iter's own masses.
        # Note: N_iter/D_iter pool across all epochs, so their absolute scale
        # tracks the epoch count. Under a fixed update_epochs (default) the scale
        # is constant across iters and the seed stays a fixed ~prior fraction;
        # under dynamic_epoch_training the count varies, which only shifts how
        # much the seed anchors — k is a RATIO, so this is a second-order
        # smoothing effect, not a bias in k.
        #
        # An ANCHOR-ONLY iteration reaches here with n_updates > 0 but pools
        # zero mass in both terms (anchor rows are in neither N nor D). Folding
        # that in would decay the EMA scale toward zero for no information, and
        # on a first-ever update would set D_ema = 0 — which reads as "not warmed
        # up" and pins k=1 until a signal iteration arrives. Skip the fold
        # instead and leave the prior EMA untouched.
        if scaling:
            # Skip the fold ONLY on an iteration with no SIGNAL rows at all
            # (`not entries`): anchor rows are in neither N nor D, so folding
            # zeros would decay the EMA scale for no information, and on a
            # first-ever update would leave D_ema == 0 and pin k=1. Keyed on
            # `entries` rather than `anchors_in_play` so an iteration that HAS
            # signal rows but pooled zero mass (every row clip-dead) still folds,
            # exactly as the pre-anchor code did.
            if (N_iter > 0.0 or D_iter > 0.0) or entries:
                if not have_prior:
                    self._pos_scale_N_ema, self._pos_scale_D_ema = N_iter, D_iter
                else:
                    self._pos_scale_N_ema = (
                        _POS_SCALE_BETA * self._pos_scale_N_ema + (1 - _POS_SCALE_BETA) * N_iter
                    )
                    self._pos_scale_D_ema = (
                        _POS_SCALE_BETA * self._pos_scale_D_ema + (1 - _POS_SCALE_BETA) * D_iter
                    )
            result["pos_adv_weight_k"] = k_last
            result["pos_adv_alive_neg_mass"] = N_iter
            result["pos_adv_pos_mass"] = D_iter
        # Only emit kl_loss_base_model when the anchor was active this iter.
        # _log_metrics gates on key presence, so vanilla runs see no
        # train/kl_loss_base_model curve at all.
        if compute_base:
            result["kl_loss_base_model"] = total_kl_base_model / n_micro_batches
        if success_frac is not None:
            result["success_fraction"] = success_frac
        # Per-branch metrics (Jitter-GRPO). Only emitted when jitter is active
        # — with jitter off (both jitter_pos and jitter_neg == 0) the
        # per-branch accumulators stay at their zero defaults (the per-mb
        # update block is gated on `lam_pos > 0 or lam_neg > 0`), so
        # n_rows_fixed and n_rows_jitter are both 0 and neither key block
        # below fires. _log_metrics' `if key in update_stats` then skips
        # the corresponding TB scalar, leaving vanilla GRPO runs without
        # any `_fixed`/`_jitter` curves. The aggregation here is row-weighted
        # (see comment in the per-mb update block above) — different from
        # the legacy `clipfrac` / `mean_ratio` / etc. above (mean-of-mb-means);
        # at jitter > 0 with variable mb sizes the two will differ slightly.
        if n_rows_fixed > 0:
            result["mean_ratio_fixed"]         = ratio_sum_fixed / n_rows_fixed
            result["mean_log_ratio_abs_fixed"] = log_ratio_abs_sum_fixed / n_rows_fixed
            result["kl_loss_last_iter_fixed"] = (
                self.config.kl_coef_last_iter
                * (kl_per_row_sum_last_iter_fixed / n_rows_fixed)
            )
            if compute_base:
                result["kl_loss_base_model_fixed"] = (
                    self.config.kl_coef_base_model
                    * (kl_per_row_sum_base_model_fixed / n_rows_fixed)
                )
        if n_rows_jitter > 0:
            result["mean_ratio_jitter"]         = ratio_sum_jitter / n_rows_jitter
            result["mean_log_ratio_abs_jitter"] = log_ratio_abs_sum_jitter / n_rows_jitter
            result["kl_loss_last_iter_jitter"] = (
                self.config.kl_coef_last_iter
                * (kl_per_row_sum_last_iter_jitter / n_rows_jitter)
            )
            if compute_base:
                result["kl_loss_base_model_jitter"] = (
                    self.config.kl_coef_base_model
                    * (kl_per_row_sum_base_model_jitter / n_rows_jitter)
                )
        # Clipfrac split by advantage sign — each {branch}_{sign} bucket
        # is gated on its own row count so a bucket that drew zero rows
        # this iter is simply absent from the result dict (and the TB
        # curve has a gap), rather than emitting a misleading 0.
        if n_rows_fixed_pos > 0:
            result["clipfrac_fixed_pos"] = clipfrac_sum_fixed_pos / n_rows_fixed_pos
            result["mean_ratio_fixed_pos"] = ratio_sum_fixed_pos / n_rows_fixed_pos
            result["mean_log_ratio_abs_fixed_pos"] = (
                log_ratio_abs_sum_fixed_pos / n_rows_fixed_pos
            )
        if n_rows_fixed_neg > 0:
            result["clipfrac_fixed_neg"] = clipfrac_sum_fixed_neg / n_rows_fixed_neg
            result["mean_ratio_fixed_neg"] = ratio_sum_fixed_neg / n_rows_fixed_neg
            result["mean_log_ratio_abs_fixed_neg"] = (
                log_ratio_abs_sum_fixed_neg / n_rows_fixed_neg
            )
        if n_rows_jitter_pos > 0:
            result["clipfrac_jitter_pos"] = clipfrac_sum_jitter_pos / n_rows_jitter_pos
            result["mean_ratio_jitter_pos"] = ratio_sum_jitter_pos / n_rows_jitter_pos
            result["mean_log_ratio_abs_jitter_pos"] = (
                log_ratio_abs_sum_jitter_pos / n_rows_jitter_pos
            )
        if n_rows_jitter_neg > 0:
            result["clipfrac_jitter_neg"] = clipfrac_sum_jitter_neg / n_rows_jitter_neg
            result["mean_ratio_jitter_neg"] = ratio_sum_jitter_neg / n_rows_jitter_neg
            result["mean_log_ratio_abs_jitter_neg"] = (
                log_ratio_abs_sum_jitter_neg / n_rows_jitter_neg
            )
        # Effective clipfrac — emitted for EVERY run (jitter on or off), unlike
        # the branch-split metrics above. This is the honest "how much gradient
        # did the clip actually kill" number; see the accumulator comments.
        if n_rows_pos_total > 0:
            result["clipfrac_effective_pos"] = clipfrac_eff_sum_pos / n_rows_pos_total
        if n_rows_neg_total > 0:
            result["clipfrac_effective_neg"] = clipfrac_eff_sum_neg / n_rows_neg_total
        # Once-per-iteration jitter gap measurement. Keys are namespaced under
        # `jitter/` by _log_metrics rather than `train/`, since they are a
        # single-minibatch snapshot at theta == theta_ref, not a per-mb mean
        # over the iteration like everything else in `result`.
        if jitter_diag:
            result["_jitter_diag"] = jitter_diag
        return result

    def _per_chunk_gap_survey(self, chunks: list) -> dict | None:
        """Measure the jitter gap PER CHUNK on a stratified sample (Stage 1).

        WHY. `_jitter_gap_diagnostics` reports the gap averaged over ~4 positive
        rows of one minibatch. This measures it for individual chunks, which turns
        it into a per-chunk BASIN WIDTH: small gap = the velocity field is locally
        flat, so neighbouring noise vectors produce nearly the same action (robust
        basin); large gap = a small eps perturbation moves the action a lot
        (fragile basin).

        That matters because this problem's defining constraint is ONE BIT of
        reward per ~40 chunks. A continuous per-chunk quantity is the only kind of
        signal that can break that, and this one is nearly free to obtain. The
        three correlations below are the point of the exercise:
          gap vs episode outcome  -> is success "landing in robust basins"? If so
              the gap is usable as an advantage shaper, which is a far better use
              than modulating lambda (see the lambda note at the bottom).
          gap vs normalised position in the episode -> does fragility live early or
              late? This is the measurement that settles the recency-weighting
              direction, which the objective itself does not determine.
          gap vs the chunk's own MSE_ref -> are fragile chunks also poorly-fit
              ones? Bears on the MSE_ref growth seen across iterations.

        COST. The clean leg is FREE: MSE_theta(eps) == -chunk.ref_log_prob, already
        stored by the ref pass. Only the jittered leg needs a forward, so a sample
        of N chunks costs N*K DiT forwards against the ref pass's n_chunks*2K.
        At N=256 that is ~5% of the ref pass's DiT work, ~12 s on a ~1700 s
        iteration (<1%). Sampling ALL chunks would be ~6%, which is why this is
        deliberately a subsample -- 256 resolves |r| > 0.12 at 2 sigma and pins the
        CV to +/-4.4% relative, which is all these questions need.

        A SINGLE PROBE LAMBDA is used for every sampled chunk, NOT the per-sign
        jitter_pos/jitter_neg. gap scales as lambda^2, so the production 0.25/0.05
        split would make gap differ 25x by advantage sign and the outcome
        correlation would just be measuring that split. A uniform probe keeps
        gap_i comparable across chunks.

        RNG: uses a dedicated seeded torch.Generator, so it consumes no global RNG
        and leaves runs bit-comparable to ones recorded before this existed
        (`_jitter_gap_diagnostics` achieves the same by consuming none at all).

        ON LAMBDA MODULATION: this survey deliberately does NOT feed a per-chunk
        lambda. With lambda constant, gap_i is ALREADY proportional to ||J_i||^2,
        so sharp-basin chunks already receive proportionally more flattening
        pressure. "Equalising the gap" (lambda_i ~ 1/||J_i||) would REMOVE that
        scaling; amplifying it drives the sharpest chunks to the mode-collapse
        bound first. Measure first; only add modulation if the numbers below
        suggest a specific policy.

        Returns a dict logged under `chunk_gap/`, or None when disabled or when
        there is not enough usable data.
        """
        N = int(getattr(self.config, "per_chunk_gap_survey_size", 0) or 0)
        if N <= 0:
            return None
        usable = [
            c for c in chunks
            if c.ref_log_prob is not None and c.initial_noise is not None
            and c.tau_samples is not None
        ]
        if len(usable) < 16:
            return None

        # Normalised position within the parent episode. Computed over ALL chunks
        # (not the sample) so it is a true fraction: episodes have different
        # lengths, and successes terminate early, so raw chunk_idx would conflate
        # "late in the episode" with "came from a failure".
        last = {}
        for c in chunks:
            last[c.episode_idx] = max(last.get(c.episode_idx, 0), c.chunk_idx)
        pos = {id(c): (c.chunk_idx / last[c.episode_idx]) if last[c.episode_idx] else 0.0
               for c in usable}

        # Stratify over 10 position bins x {success, failure}. Without this the
        # gap-vs-position estimate is confounded: failures run to the 50-chunk
        # truncation while successes stop at ~36, so late positions would be
        # drawn disproportionately from failures and the outcome effect would
        # masquerade as a position effect.
        rng = np.random.default_rng(self.config.seed + self.iteration * 7919)
        cells: dict = {}
        for c in usable:
            cells.setdefault((min(int(pos[id(c)] * 10), 9), bool(c.episode_success)), []).append(c)
        per_cell = max(1, N // max(len(cells), 1))
        sample = []
        for key in sorted(cells, key=lambda k: (k[1], k[0])):
            pool = cells[key]
            take = min(per_cell, len(pool))
            idx = rng.choice(len(pool), size=take, replace=False)
            sample.extend(pool[i] for i in idx)
        if len(sample) < 16:
            return None

        probe_lam = float(self.config.jitter_pos) or 0.25
        K = len(self.config.tau_centers)
        gaps: list[float] = []
        meta: list[tuple] = []
        gen = torch.Generator(device="cpu").manual_seed(
            self.config.seed + self.iteration * 104729
        )
        bs = max(1, self.config.mini_batch_size * 2)
        with self._model_lock, torch.no_grad():
            for start in range(0, len(sample), bs):
                batch = sample[start:start + bs]
                result = self._prepare_batch([(c, "fixed") for c in batch])
                if result is None:
                    continue
                bd, valid = result
                eps = bd["initial_noise"]
                if eps is None:
                    continue
                B, H, D = eps.shape
                tau = torch.from_numpy(
                    np.stack([c.tau_samples for c in valid], axis=1)
                ).to(device=self.device, dtype=torch.bfloat16)
                xi = torch.randn(K, B, H, D, generator=gen).to(
                    device=self.device, dtype=eps.dtype
                )
                nfi = (
                    math.sqrt(1.0 - probe_lam * probe_lam) * eps.unsqueeze(0)
                    + probe_lam * xi
                ).to(eps.dtype)
                lp_jit = compute_fm_log_prob(
                    action_head=self.model.action_head,
                    backbone_output=bd["backbone_output"],
                    state_features=bd["state_features"],
                    embodiment_id=bd["embodiment_id"],
                    actions=bd["actions"],
                    action_mask=bd["action_masks"],
                    timesteps=tau,
                    noise=eps,
                    n_samples=K,
                    noise_for_input=nfi,
                )
                for i, c in enumerate(valid):
                    # gap = MSE(eps') - MSE(eps) = (-lp_jit) - (-ref_log_prob)
                    gp = float(c.ref_log_prob) - float(lp_jit[i].item())
                    if not math.isfinite(gp):
                        continue
                    gaps.append(gp)
                    meta.append((
                        1.0 if c.episode_success else 0.0,
                        pos[id(c)],
                        -float(c.ref_log_prob),        # this chunk's MSE_ref
                        1.0 if c.advantage > 0 else 0.0,
                    ))
        if len(gaps) < 16:
            return None

        gv = np.asarray(gaps, dtype=np.float64)
        succ, position, mse_ref, posadv = (np.asarray(x, dtype=np.float64)
                                           for x in zip(*meta))

        def corr(a, b):
            if a.std() < 1e-12 or b.std() < 1e-12:
                return None
            return float(np.corrcoef(a, b)[0, 1])

        out = {
            "n": int(gv.size),
            "probe_lambda": probe_lam,
            "mean": float(gv.mean()),
            "p10": float(np.percentile(gv, 10)),
            "p50": float(np.percentile(gv, 50)),
            "p90": float(np.percentile(gv, 90)),
            "max": float(gv.max()),
        }
        if abs(gv.mean()) > 1e-12:
            # THE decision statistic. Compare against the ~4-8% intrinsic
            # xi-sampling floor: at or below it, per-chunk treatment is dead.
            out["cv"] = float(gv.std() / abs(gv.mean()))
        for lbl, mask in (("succ", succ > 0.5), ("fail", succ <= 0.5)):
            if mask.sum() >= 4:
                out[f"mean_{lbl}"] = float(gv[mask].mean())
        for lbl, mask in (("posadv", posadv > 0.5), ("negadv", posadv <= 0.5)):
            if mask.sum() >= 4:
                out[f"mean_{lbl}"] = float(gv[mask].mean())
        for lbl, other in (("r_outcome", succ), ("r_position", position),
                           ("r_ref_mse", mse_ref)):
            c = corr(gv, other)
            if c is not None:
                out[lbl] = c
        print(
            f"  chunk-gap survey: n={out['n']} lam={probe_lam:.2f} "
            f"mean={out['mean']:.5f} CV={out.get('cv', float('nan')):.3f} "
            f"r(outcome)={out.get('r_outcome', float('nan')):+.3f} "
            f"r(position)={out.get('r_position', float('nan')):+.3f} "
            f"r(MSE_ref)={out.get('r_ref_mse', float('nan')):+.3f}"
        )
        return out

    def _jitter_gap_diagnostics(
        self,
        *,
        ready_backbone,
        ready_state_features,
        ready_embodiment_id,
        ready_actions,
        ready_masks,
        ready_noise,
        timesteps,
        noise_for_input,
        lam_row,
        pos_adv_mask,
        fixed_row_mask,
        jitter_row_mask,
    ) -> dict:
        """Measure the fixed-vs-jitter FM-loss gap directly, once per iteration.

        WHY THIS EXISTS. The gap

            gap = E[ MSE_theta(eps') - MSE_theta(eps) ]  ~=  (1-tau)^2 * lam^2 * ||grad_x v_theta||_F^2

        is the single quantity Jitter-GRPO turns on, and it is readable two ways:
          - as the Jacobian penalty (the derivation's framing), and
          - as the EXTRA RATIO HEADROOM on a positive-advantage row: without
            jitter, `log rho <= MSE_ref` caps reinforcement at ~0.01-0.05; with
            jitter the row starts at `rho = e^-gap` and has `MSE_ref + gap` of
            usable room. A positive row cannot be clipped by clip_eps_LOW no
            matter how far its ratio falls (min() picks the unclamped branch —
            see clip_killed_gradient), and the only bound that CAN kill it,
            clip_eps_high, is unreachable at these ratios. So the added room is
            usable in full.

        WHAT `jacobian_fro_sq` ACTUALLY IS. Two caveats on the name:
          - The MSE is a mean over VALID action dims, so this is
            `‖∇_x v_θ‖²_F / D_valid`, not the raw Frobenius norm. It is a proxy
            for trend and for cross-lambda comparison, not an absolute quantity.
          - It divides by `lambda²`, but the actual per-element perturbation
            variance of `eps' - eps` is `(sqrt(1-lambda²) - 1)² + lambda²`. So
            the estimate is biased HIGH by `1 + ((sqrt(1-lam²)-1)/lam)²`:
            +1.6% at lambda=0.25, +7% at lambda=0.50. Small enough that the
            metric stays comparable across the lambda ladder, but it is a bias,
            not exact invariance.
          - `timesteps` arrives as bf16 (built that way in _grpo_update_inner
            from the fp32 `tau_samples` on the chunks), so the `(1-tau)²`
            prefactor divided out here — and the reported `tau{k}_value` — carry
            ~0.4% relative quantization versus the taus the ref pass actually
            used. Third-order against the two biases above; noted so the
            enumeration is complete.

        Until now it was only observable by DIFFING two TB curves
        (`mean_log_ratio_abs_fixed` vs `_jitter`), which requires
        `jitter_paired=True` — so it was unmeasurable in every `nojitterpair`
        run. This measures it directly instead, in any mode.

        WHY IT IS TRUSTWORTHY. Both forwards run here, back to back, on the SAME
        minibatch with the SAME cached backbone features and the SAME tau
        samples, differing ONLY in `noise_for_input`. So unlike the
        curve-differencing approach it carries none of the ~5e-4 bf16/batching
        noise floor between the ref pass (batch = 2*mini_batch_size, fresh
        `_prepare_batch`) and the update pass (batch = mini_batch_size, rebuilt
        from cache), and it does not depend on `ref_log_prob` at all.

        COST AND SAFETY. Two extra no_grad forwards on ONE minibatch per
        iteration: ~12 DiT passes against the ~780+ a normal iteration runs, so
        ~1.5% overhead, and no activations are retained so peak VRAM is strictly
        below a training minibatch. Critically it consumes NO RNG: with both
        `timesteps` and `noise` supplied, compute_fm_log_prob's sampling
        branches are skipped, and the DiT is in eval mode with lora_dropout
        inert — so inserting this does not shift the global torch RNG stream and
        runs stay comparable to ones recorded before it existed.

        Args:
            lam_row: [B] per-row jitter lambda (jitter_pos / jitter_neg by
                pre-renorm advantage sign) — the same tensor used to build
                `noise_for_input`, so the lambda^2 divided out below is exactly
                the one applied.
            pos_adv_mask: [B] bool, PRE-renormalization advantage sign. Matches
                the convention of the clipfrac_{branch}_{pos,neg} buckets.
            fixed_row_mask: [B] bool, True for `jitter_paired=True` "fixed"
                rows. Those rows have noise_for_input == eps, so their gap must
                come out ~0 — a built-in correctness check on this whole
                measurement. Empty in `nojitterpair` mode.

        Returns:
            dict of scalars, TB-prefixed `jitter/` by _log_metrics. Empty when
            there is nothing meaningful to report (no positive-advantage rows in
            this minibatch), which leaves a curve gap rather than a fake 0.
        """
        K = timesteps.shape[0]
        common = dict(
            action_head=self.model.action_head,
            backbone_output=ready_backbone,
            state_features=ready_state_features,
            embodiment_id=ready_embodiment_id,
            actions=ready_actions,
            action_mask=ready_masks,
            timesteps=timesteps,
            noise=ready_noise,
            n_samples=K,
        )
        with torch.no_grad():
            # noise_for_input=None => DiT input is the original eps for EVERY
            # row, including rows tagged "jitter". This is the clean reference
            # leg and it is what makes the gap a pure input-perturbation effect.
            _, lp_clean = compute_fm_log_prob(
                **common, noise_for_input=None, return_per_tau=True
            )  # [K, B]
            _, lp_jit = compute_fm_log_prob(
                **common, noise_for_input=noise_for_input, return_per_tau=True
            )  # [K, B]

        # log_prob = -MSE, so (clean - jittered) = MSE_jittered - MSE_clean = gap.
        # Non-negative in expectation; individual rows can go slightly negative
        # from the finite-xi sample, which is why we report means.
        gap_per_tau = (lp_clean - lp_jit).float()          # [K, B]
        gap_row = gap_per_tau.mean(dim=0)                  # [B]

        # Divide out the analytic prefactor to recover the Jacobian norm itself,
        # using the ACTUAL jittered taus (tau_centers +/- N(0, 0.02)) rather than
        # the nominal centers. Per-row division, then mean — dividing the means
        # would bias the estimate when lam_row is mixed.
        w_row = ((1.0 - timesteps.float()) ** 2).mean(dim=0)   # [B]
        denom = w_row * (lam_row.float() ** 2)                 # [B]
        jac_row = torch.where(
            denom > 1e-12, gap_row / denom.clamp_min(1e-12), torch.zeros_like(gap_row)
        )

        out: dict = {}
        neg_adv_mask = ~pos_adv_mask
        # Restrict every headline number to JITTER rows: a paired-mode "fixed"
        # row has noise_for_input == eps by construction, so including it would
        # dilute the gap toward zero by the fixed/jitter row ratio.
        #
        # Taken as an EXPLICIT mask, never as ~fixed_row_mask. Anchor rows are
        # excluded from BOTH masks by the caller, so complementing one would
        # sweep every anchor row into the other — and since anchors are never
        # jittered their gap is structurally 0, which would drag gap_neg (and
        # hence neg_clip_budget_used, documented as the ceiling on jitter_neg)
        # toward zero by the anchor row fraction.
        jit_mask = jitter_row_mask
        jp = jit_mask & pos_adv_mask
        jn = jit_mask & neg_adv_mask
        n_jp = int(jp.sum().item())
        n_jn = int(jn.sum().item())

        if n_jp > 0:
            gap_pos = float(gap_row[jp].mean().item())
            out["gap_pos"] = gap_pos
            out["jacobian_fro_sq"] = float(jac_row[jp].mean().item())
            out["n_rows_pos"] = n_jp
            # STAGE-0 per-chunk spread. gap_row is already PER CHUNK; everything
            # else here collapses it to a mean. The coefficient of variation across
            # the rows of this one minibatch is the cheapest possible test of
            # whether a per-chunk gap carries information at all:
            #   a single chunk's gap has ~4-8% intrinsic noise from the xi draw
            #   (rel. sd ~ sqrt(2/r_eff)/sqrt(K), r_eff = 50..192 output dims),
            # so CV <~ 8% means the between-chunk spread is all sampling noise and
            # a per-chunk treatment cannot help; CV >~ 15% means real structure.
            # Only ~4 positive rows per iteration, so read this pooled over many
            # iterations, not per-iteration. Free: no extra forward.
            if n_jp > 1:
                gp_rows = gap_row[jp]
                m = float(gp_rows.mean().item())
                if abs(m) > 1e-12:
                    out["gap_pos_cv"] = float(gp_rows.std().item()) / abs(m)
            out["gap_pos_min"] = float(gap_row[jp].min().item())
            out["gap_pos_max"] = float(gap_row[jp].max().item())
            # Per-tau profile, POSITIVE rows only: gap scales as lam^2, and
            # lam_pos is typically ~5x lam_neg (25x in the gap), so pooling the
            # signs would make the profile a mixture of two very different
            # curves. Index k maps to config.tau_centers[k].
            for k in range(K):
                out[f"gap_at_tau{k}"] = float(gap_per_tau[k][jp].mean().item())
                out[f"tau{k}_value"] = float(timesteps[k][jp].float().mean().item())
            # THE jitter metric: how many times more usable log-ratio room a
            # positive row has with jitter than without.
            #
            # SCOPE MISMATCH, stated so it is not mistaken for an identity:
            # `ref_pos` is a chunk-level mean over ALL positive-advantage live
            # chunks of the iteration, while `gap_pos` is a row-level mean over
            # THIS ONE minibatch. Fine for the intended use (an order-of-
            # magnitude read on "is jitter buying gradient room"), but the two
            # terms are not drawn from the same sample.
            #
            # getattr, not attribute access: this is an OPTIONAL upstream stat
            # (absent if the ref pass produced nothing, and absent entirely on
            # any trainer built via __new__ without __init__, as the CPU tests
            # do). A missing diagnostic input must degrade to "no curve", never
            # to an AttributeError hours into a run.
            ref_pos = (getattr(self, "_ref_mse_stats", None) or {}).get("pos_mean")
            if ref_pos is not None and ref_pos > 1e-9:
                out["headroom_multiplier"] = (ref_pos + gap_pos) / ref_pos
                out["headroom_ref_only"] = ref_pos
                out["headroom_with_jitter"] = ref_pos + gap_pos
        if n_jn > 0:
            out["gap_neg"] = float(gap_row[jn].mean().item())
            out["n_rows_neg"] = n_jn
            # Fraction of the erosion clip budget |log(1-clip_eps_low)| that the
            # negative rows' gap consumes BEFORE any policy drift. At
            # jitter_neg=0.05 this is a few percent (harmless); as it approaches
            # 1.0 every negative row is born outside the clip and contributes no
            # gradient at all. This is the hard ceiling on jitter_neg, and it is
            # the only place clip_eps_low interacts with jitter — it cannot clip
            # a POSITIVE row (min() always picks the unclamped branch there).
            lo_budget = -math.log(max(1.0 - self.config.clip_eps_low, 1e-12))
            out["neg_clip_budget_used"] = out["gap_neg"] / lo_budget
        if int(fixed_row_mask.sum().item()) > 0:
            # Self-check, paired mode only: must read ~0 (bounded by bf16 noise,
            # order 1e-4). A non-trivial value means the fixed rows are NOT
            # being fed the original eps and the jitter bookkeeping is wrong.
            out["gap_fixed_rows_selfcheck"] = float(
                gap_row[fixed_row_mask].mean().item()
            )
        return out

    def _min_expected_batches(
        self, entries: list[tuple[ActionChunk, str]], signal_mb_size: int
    ) -> int:
        """LOWER BOUND on the minibatches an epoch yields at `signal_mb_size`.

        Used only to size the anchor slot reservation — `_with_anchor_rows`
        measures the real count and distributes against that.

        A bound, not an exact count, and the direction is the whole point.
        Reserved capacity is `slots x n_batches`, so UNDER-estimating the batch
        count over-reserves slots (costing extra optimizer steps — see the gap
        note below — but never dropping rows), while OVER-estimating
        under-reserves and silently drops anchor rows. Since dropped rows are the
        unrecoverable failure, every term below is conservative:

        - `ceil(n / mb)` is itself a lower bound for the stratified sampler,
          which under-fills mid-epoch when a group's queue drains early and so
          can yield MORE batches than the row count implies. The gap is small for
          near-uniform group sizes but LARGE for skewed ones: measured max
          real - est of 99 over 30k randomized shapes, e.g. groups (109, 1x11) at
          mb=12 yields 109 batches against a bound of 10. Since the reservation
          loop stops at the first sufficient slot count, a loose bound
          over-reserves slots and can multiply the per-iteration step count
          several-fold — not merely "an occasional extra step".
        - `_iter_balanced_minibatches` yields at most that, terminating early
          once its majority pool drains after
          `ceil(len(majority) / n_majority_per_batch)` batches. Mirrors that
          sampler's arithmetic including both of its fallbacks to stratified.

        `test_min_expected_batches_never_overshoots` pins the direction.
        """
        n = len(entries)
        if n == 0 or signal_mb_size <= 0:
            return 1
        n_stratified = math.ceil(n / signal_mb_size)
        if not self.config.balanced_minibatch_training:
            return n_stratified
        n_pos = sum(1 for c, _m in entries if c.advantage > 0)
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return n_stratified                      # sampler falls back
        natural = n_pos / n
        ratio = self._effective_pos_ratio(natural)
        n_pos_per_batch = max(1, round(ratio * signal_mb_size))
        n_neg_per_batch = signal_mb_size - n_pos_per_batch
        if n_neg_per_batch <= 0:
            return n_stratified                      # sampler falls back
        if natural < ratio:
            majority, per_batch = n_neg, n_neg_per_batch
        else:
            majority, per_batch = n_pos, n_pos_per_batch
        return max(1, min(n_stratified, math.ceil(majority / per_batch)))

    def _with_anchor_rows(
        self,
        batch_iter: Iterator[list[tuple[ActionChunk, str]]],
        anchor_entries: list[tuple[ActionChunk, str]],
        max_per_batch: int,
        rng: np.random.Generator,
    ) -> Iterator[list[tuple[ActionChunk, str]]]:
        """Append an anchor-row quota to minibatches from `batch_iter`.

        Anchor rows are added here rather than inside the samplers so both
        sampler paths stay signal-only: their pos/neg pools, stratification, and
        epoch-length anchors are unchanged, and with no anchor entries this is a
        transparent pass-through. The inner sampler is driven at
        `mini_batch_size - max_per_batch` so total rows per minibatch — and hence
        peak VRAM — stay at `mini_batch_size`.

        The epoch's batches are MATERIALIZED first so the per-batch target is
        `pool / n_batches` against the real count. Estimating it instead is what
        made a small pool under-deliver: `ceil(len(entries) / signal_mb_size)` is
        the stratified count, while `_iter_balanced_minibatches` (the default)
        stops early once its majority pool drains, and both fallbacks inside it
        change the count again. At a 1-chunk pool the estimate produced ZERO
        trained anchor rows. Materializing costs only a list of references —
        the samplers do index arithmetic, no tensor work — and consumes the
        epoch's RNG up front, which changes nothing about determinism.

        The target may be fractional, so a running credit accumulator emits
        `floor(credit)` rows per batch and carries the remainder; `+1e-9` before
        truncating because an epoch's credits sum to a whole number only up to
        float error (1/6 * 6 == 0.9999999999999999). Emission is capped at
        `max_per_batch` (the reserved slot count).

        The pool is drawn without replacement. `used` additionally prevents the
        same chunk landing twice in ONE batch if the permutation were exhausted
        mid-batch — unreachable as written, since the target is `pool /
        n_batches` so an epoch's emissions never exceed the pool and the
        permutation never wraps; kept as defense in case the target derivation
        changes. `test_fixed_branch_metrics_exclude_anchors` checks the invariant
        that makes it unreachable, but only on three enumerated (pool, n_batches)
        pairs — it would not catch a target-derivation change that broke the
        invariant at some other shape.
        """
        if not anchor_entries or max_per_batch <= 0:
            yield from batch_iter
            return

        batches = list(batch_iter)
        if not batches:
            return
        capacity = max_per_batch * len(batches)
        if capacity < len(anchor_entries):
            # The reservation was sized from an ESTIMATE of the batch count; if
            # the sampler produced fewer batches than that, the epoch physically
            # cannot place every anchor row without exceeding mini_batch_size.
            # Say so rather than dropping rows silently.
            print(
                f"  WARNING: anchor row capacity {capacity} "
                f"({max_per_batch}/minibatch x {len(batches)} minibatches) is "
                f"below the {len(anchor_entries)}-row pool — "
                f"{len(anchor_entries) - capacity} row(s) will not train this "
                f"epoch. Lower anchor_max_row_frac or raise mini_batch_size."
            )
        rows_per_batch = min(
            len(anchor_entries) / len(batches), float(max_per_batch)
        )

        order = list(rng.permutation(len(anchor_entries)).astype(int))
        ptr = 0
        credit = 0.0
        for batch in batches:
            credit += rows_per_batch
            n_take = min(int(credit + 1e-9), max_per_batch)
            credit -= n_take
            used: set[int] = set()
            while len(used) < n_take:
                if ptr >= len(order):
                    order = list(rng.permutation(len(anchor_entries)).astype(int))
                    ptr = 0
                idx = int(order[ptr])
                ptr += 1
                if idx in used:
                    continue
                used.add(idx)
                batch.append(anchor_entries[idx])
            yield batch

    def _iter_stratified_minibatches(
        self,
        entries: list[tuple[ActionChunk, str]],
        rng: np.random.Generator,
        mb_size: int | None = None,
    ) -> Iterator[list[tuple[ActionChunk, str]]]:
        """Yield minibatches with best-effort per-group stratification.

        Each entry is a (chunk, mode) tuple where mode is "fixed" (no jitter)
        or "jitter" (Jitter-GRPO ε'). Group binning still uses chunk.group_id
        — both copies of a paired chunk share the same group_id, so they
        land in the same group's queue.

        Each minibatch contains (mb_size // n_live_groups) GUARANTEED entries
        from every non-empty group, plus (mb_size % n_live_groups) FILLER
        entries drawn uniformly without replacement from entries not yet
        consumed this epoch. With mb_size=8 and num_groups=5 that's 1
        entry per group plus 3 filler.

        Why stratify: chunks within an episode share an identical advantage
        (A_ep / num_chunks from episode_buffer._build_chunks). A small
        flat-shuffled minibatch dominated by 1-2 episodes has near-zero
        advantage variance, and the per-minibatch z-score renorm in
        _grpo_update_inner then squashes that batch's gradient signal
        toward zero. Forcing every batch to span all live groups
        guarantees the renorm has multiple distinct group-mean
        advantages to work with.

        Why uniform-over-remaining-ENTRIES for filler (vs uniform-over-
        GROUPS): self-balances. With ~equal group sizes, fuller queues
        contribute filler proportionally more often, so all groups drain
        in lockstep and the "≥1 per group" guarantee holds for
        essentially the whole epoch. Uniform-over-groups would drain
        small groups too fast and skew the late epoch.

        Walking a pre-shuffled filler_order left-to-right while skipping
        already-used indices is equivalent to uniform-without-replacement
        from the remaining pool: at any point, the prefix of un-visited
        filler_order entries is itself a uniform random permutation of
        the remaining set.

        Degenerate cases:
          - mb_size < n_live_groups: base_per_group=0, everything becomes
            filler → degrades to flat random shuffle (no stratification).
          - A group's queue empties before others: silently skipped in
            subsequent guaranteed phases (best-effort); other groups
            continue contributing.
          - Last batch may be smaller than mb_size if entries don't
            divide evenly.

        Each entry is yielded exactly once per epoch.

        Args:
            mb_size: Rows per minibatch. Defaults to config.mini_batch_size;
                _grpo_update_inner passes a reduced size when an anchor-row
                quota is appended afterwards (see _with_anchor_rows).
        """
        n_entries = len(entries)
        if n_entries == 0:
            return

        # Bin entry indices by group_id and shuffle each group's order.
        # group_id is propagated from GRPOEpisode in
        # episode_buffer._build_chunks. With paired (fixed, jitter) entries
        # both copies of a chunk share the same group_id, so they land in
        # the same group's queue and may end up in the same or different
        # minibatches across the epoch — either is fine for the Jacobian
        # regularizer expectation argument.
        group_to_queue: dict[int, list[int]] = {}
        for i, (c, _mode) in enumerate(entries):
            group_to_queue.setdefault(c.group_id, []).append(i)
        for gid in group_to_queue:
            rng.shuffle(group_to_queue[gid])

        # Global filler visitation order. Walked once left-to-right;
        # entries already consumed by a guaranteed slot (or an earlier
        # filler pick) are skipped without rewinding the pointer.
        filler_order = np.arange(n_entries)
        rng.shuffle(filler_order)

        group_positions: dict[int, int] = {gid: 0 for gid in group_to_queue}
        filler_pos = 0
        # Tracks entries already placed in some batch this epoch. Both the
        # guaranteed phase and the filler phase can consume any entry, so
        # this is the single source of truth across both paths.
        used = np.zeros(n_entries, dtype=bool)

        n_live_groups = len(group_to_queue)
        mb_size = mb_size or self.config.mini_batch_size
        base_per_group = mb_size // n_live_groups
        n_filler = mb_size - base_per_group * n_live_groups

        while True:
            batch_idx_list: list[int] = []

            # Guaranteed slots: take up to base_per_group UNUSED entries
            # from each non-empty group's shuffled queue. Filler-consumed
            # entries are walked past (pointer advances, taken count does
            # not), so each group always tries hardest to land its quota.
            if base_per_group > 0:
                for gid, queue in group_to_queue.items():
                    taken = 0
                    pos = group_positions[gid]
                    while taken < base_per_group and pos < len(queue):
                        idx = queue[pos]
                        pos += 1
                        if not used[idx]:
                            batch_idx_list.append(idx)
                            used[idx] = True
                            taken += 1
                    group_positions[gid] = pos

            # Filler slots: walk filler_order, skip already-used. Same
            # skip-on-used pattern as the guaranteed phase so an entry
            # that the guaranteed phase already took in this very batch
            # isn't double-counted.
            n_filler_taken = 0
            while n_filler_taken < n_filler and filler_pos < n_entries:
                idx = int(filler_order[filler_pos])
                filler_pos += 1
                if not used[idx]:
                    batch_idx_list.append(idx)
                    used[idx] = True
                    n_filler_taken += 1

            if not batch_idx_list:
                # All entries consumed: both pointers exhausted AND no
                # unused entries remain. Safe termination — argued
                # because each entry is in both filler_order and exactly
                # one group queue, and both pointers advance
                # monotonically through them.
                return

            yield [entries[i] for i in batch_idx_list]

    def _effective_pos_ratio(self, natural_pos_frac: float) -> float:
        """Target positive-advantage fraction for the balanced sampler.

        Fixed at ``balanced_minibatch_positive_adv_ratio`` by default. When
        ``balanced_minibatch_positive_adv_ratio_dynamic`` is set, track the
        natural positive fraction (≈ success rate), clamped to
        ``[balanced_minibatch_positive_adv_ratio,
        balanced_minibatch_positive_adv_ratio_max]`` — so at HIGH success the
        sampler stops oversampling the rare, large-advantage failures (the
        policy-collapse driver), while the floor preserves positive oversampling
        (reinforcement signal) at low success. Single source of truth so the
        sampler and the logged ``balanced_pos_ratio`` never drift.
        """
        base = self.config.balanced_minibatch_positive_adv_ratio
        if not self.config.balanced_minibatch_positive_adv_ratio_dynamic:
            return base
        return min(
            self.config.balanced_minibatch_positive_adv_ratio_max,
            max(base, natural_pos_frac),
        )

    def _iter_balanced_minibatches(
        self,
        entries: list[tuple[ActionChunk, str]],
        rng: np.random.Generator,
        mb_size: int | None = None,
    ) -> Iterator[list[tuple[ActionChunk, str]]]:
        """Yield mini-batches with balanced positive/negative advantage sampling.

        Applies the target pos/neg ratio in BOTH directions:
          - When natural_pos_frac < pos_ratio (too few positives): positives
            are the minority class, oversampled WITH replacement. Negatives
            are sampled WITHOUT replacement and control when the epoch ends.
          - When natural_pos_frac > pos_ratio (too few negatives): negatives
            are the minority class, oversampled WITH replacement. Positives
            are sampled WITHOUT replacement and control when the epoch ends.

        Falls back to _iter_stratified_minibatches only when one sign class
        is entirely absent (can't form a balanced batch).

        This bidirectional design prevents two distinct failure modes:
          - Low success (few positives): gradient dominated by negative
            advantages → oversample positives to provide learning signal.
          - High success (few negatives): minibatch z-score renorm amplifies
            the rare large-negative-advantage failures, causing the policy to
            over-correct toward avoiding those specific failure modes. Cycling
            negatives with replacement caps this amplification.

        Epoch length (number of mini-batches) is anchored to `ceil(n / mb_size)`
        matching the vanilla stratified path. The minority pool cycles with
        replacement; the majority pool is drawn without replacement and may not
        be fully consumed before the epoch anchor is reached.

        Args:
            entries: List of (ActionChunk, mode) tuples from _grpo_update_inner.
            rng:     Per-epoch numpy Generator (caller provides reproducible seed).
            mb_size: Rows per minibatch. Defaults to config.mini_batch_size;
                     reduced by the caller when an anchor-row quota is appended
                     afterwards (see _with_anchor_rows).

        Yields:
            Lists of (ActionChunk, mode) tuples, length <= mini_batch_size.
        """
        if not entries:
            return

        # Split entries by advantage sign. The per-chunk advantage inherits its
        # sign directly from the episode-level group-relative normalization;
        # live_chunks already filtered out zero-advantage (dead group) entries,
        # and anchor entries are held out by the caller — their constant positive
        # advantage would swamp the positive pool at high success and crowd out
        # the genuine mixed-group successes this sampler exists to preserve.
        pos_indices = [i for i, (c, _) in enumerate(entries) if c.advantage > 0]
        neg_indices = [i for i, (c, _) in enumerate(entries) if c.advantage <= 0]

        # Fall back to stratified when one sign class is absent — we can't
        # form a balanced batch without both positive and negative entries.
        if not pos_indices or not neg_indices:
            yield from self._iter_stratified_minibatches(entries, rng, mb_size)
            return

        natural_pos_frac = len(pos_indices) / len(entries)
        # Target positive fraction: fixed config value, or (dynamic mode) the
        # natural fraction clamped to [base, max]. See _effective_pos_ratio.
        pos_ratio = self._effective_pos_ratio(natural_pos_frac)

        mb_size = mb_size or self.config.mini_batch_size
        n_pos_per_batch = max(1, round(pos_ratio * mb_size))
        n_neg_per_batch = mb_size - n_pos_per_batch

        # Guard: if rounding left no room for one sign class (e.g. pos_ratio=0.9375
        # with mb_size=8 causes round(7.5)=8 → n_neg=0), fall back.
        #
        # Logged, not silent: an anchor-row quota SHRINKS mb_size here, so a
        # large anchor pool can turn balanced sampling off. `pos_ratio` is NOT the
        # natural positive fraction — _effective_pos_ratio returns the configured
        # constant unless the dynamic flag is set, then clamps to ..._ratio_max —
        # so at the default 0.5 this needs signal_mb_size == 1, i.e. anchor_slots
        # == mini_batch_size - 1. The pool ratio that implies SCALES with
        # mini_batch_size — measured first fallback at 1.05:1 (mb=4), 3.05:1
        # (mb=8), 5.05:1 (mb=12), 7.05:1 (mb=16) — and is reachable at the default
        # anchor_max_row_frac=1.0, because the budget's one-whole-episode floor
        # can admit an anchor episode far larger than the cap.
        # train/balanced_pos_ratio (computed separately) still reports the target
        # as though it had applied, hence the warning.
        if n_neg_per_batch <= 0:
            print(
                f"  WARNING: balanced sampling disabled this epoch — "
                f"pos_ratio={pos_ratio:.3f} at {mb_size} row(s)/minibatch "
                f"leaves no negative slot. Falling back to stratified sampling. "
                f"Lower anchor_max_row_frac or raise mini_batch_size."
            )
            yield from self._iter_stratified_minibatches(entries, rng, mb_size)
            return

        # Determine minority vs majority pool based on which sign class is
        # underrepresented relative to the target ratio:
        #   - natural_pos_frac < pos_ratio: positives are minority → cycle positives
        #   - natural_pos_frac > pos_ratio: negatives are minority → cycle negatives
        # The minority pool is oversampled with replacement (cycles when exhausted);
        # the majority pool is sampled without replacement and controls epoch end.
        if natural_pos_frac < pos_ratio:
            minority_indices = pos_indices
            majority_indices = neg_indices
            n_minority_per_batch = n_pos_per_batch
            n_majority_per_batch = n_neg_per_batch
        else:
            # natural_pos_frac >= pos_ratio: negatives are minority (or exactly at
            # target, in which case either direction is fine — negatives is the
            # conservative choice since it prevents positive dominance in batches).
            minority_indices = neg_indices
            majority_indices = pos_indices
            n_minority_per_batch = n_neg_per_batch
            n_majority_per_batch = n_pos_per_batch

        # Shuffle both pools independently for this epoch.
        minority_pool = list(rng.permutation(len(minority_indices)).astype(int))
        majority_pool = list(rng.permutation(len(majority_indices)).astype(int))

        # Epoch length is anchored to ceil(n_entries / mb_size) to keep the
        # per-epoch optimizer-step budget comparable to the vanilla stratified
        # path. The minority pool cycles with replacement when exhausted; the
        # majority pool advances a running pointer. When the majority pool
        # drains before n_batches is reached, the epoch terminates early to
        # avoid yielding minority-only batches (same-sign z-score renorm would
        # produce meaningless gradients).
        n_batches = math.ceil(len(entries) / mb_size)
        minority_ptr = 0
        majority_ptr = 0

        for _ in range(n_batches):
            batch: list[tuple[ActionChunk, str]] = []

            # --- Minority slots (oversample with replacement) ---
            for _ in range(n_minority_per_batch):
                if minority_ptr >= len(minority_pool):
                    # Re-shuffle and restart when pool is exhausted
                    minority_pool = list(rng.permutation(len(minority_indices)).astype(int))
                    minority_ptr = 0
                batch.append(entries[minority_indices[minority_pool[minority_ptr]]])
                minority_ptr += 1

            # --- Majority slots (without replacement, stop when exhausted) ---
            taken = 0
            while taken < n_majority_per_batch and majority_ptr < len(majority_pool):
                batch.append(entries[majority_indices[majority_pool[majority_ptr]]])
                majority_ptr += 1
                taken += 1

            yield batch

            # Majority pool exhausted — stop rather than yielding minority-only
            # batches (same-sign z-score renorm would be meaningless).
            if majority_ptr >= len(majority_pool):
                return

    def _prepare_batch(
        self, batch: list[tuple[ActionChunk, str]]
    ) -> Optional[tuple[dict, list[ActionChunk]]]:
        """Convert a list of (ActionChunk, mode) entries into GPU tensors for training.

        This handles:
        - Using raw normalized actions (50x128) for FM log-prob computation
        - Re-encoding observations through the backbone
        - Creating embodiment ID tensors

        The raw_action field is REQUIRED — it's the action in the model's internal
        space (before decode_action slices/denormalizes). Without it, the FM loss
        surrogate would be computed on mismatched dimensions.

        Args:
            batch: List of (ActionChunk, mode) tuples. mode is "fixed" (no
                jitter, DiT input noise = original ε) or "jitter" (DiT input
                noise = ε' = sqrt(1-λ²)·ε + λ·ξ, constructed by the caller).
                Mode is carried through to batch_data["modes"] so the caller
                knows which rows need ξ-jittered input noise.

        Returns:
            Tuple of (tensor_dict, valid_batch_list), or None if batch is invalid.
            tensor_dict["modes"] is parallel to valid_batch_list (length B).
        """
        if not batch:
            return None

        # Filter to entries whose chunk has raw_actions (required for FM
        # log-prob). Preserve ordering and keep mode aligned 1:1.
        valid_pairs = [(c, m) for (c, m) in batch if c.raw_action is not None]
        if not valid_pairs:
            return None

        valid_batch = [c for (c, _) in valid_pairs]
        modes = [m for (_, m) in valid_pairs]

        B = len(valid_batch)

        # --- Raw normalized actions (50×128) for FM log-prob ---
        # This is what the FM loss evaluates — the model's internal action representation
        actions = torch.stack([
            torch.from_numpy(chunk.raw_action).float() for chunk in valid_batch
        ]).to(self.device, dtype=torch.bfloat16)  # [B, 50, 128]

        # --- Action masks ---
        action_masks = torch.stack([
            torch.from_numpy(chunk.action_mask).float() for chunk in valid_batch
        ]).to(self.device, dtype=torch.bfloat16)  # [B, 50, 128]

        # --- Initial noise (the ε₀ that was denoised into these actions) ---
        # GRPO requires evaluating the FM log-prob along the ACTUAL denoising
        # path for both the reference and current passes; the shared ε is what
        # makes the importance ratio a model-quality signal rather than noise.
        # Falling back to a freshly-sampled noise here would break that
        # invariant (ref and current would use different ε), so we hard-fail
        # instead of silently degrading training.
        missing_noise = [c for c in valid_batch if c.initial_noise is None]
        if missing_noise:
            raise RuntimeError(
                f"{len(missing_noise)}/{len(valid_batch)} chunks are missing "
                "initial_noise. GRPO requires captured initial noise from "
                "grpo_server.py; check that GRPOPolicyWrapper is wrapping the "
                "policy and that the bfloat16→numpy conversion succeeded."
            )
        initial_noise = torch.stack([
            torch.from_numpy(chunk.initial_noise).float() for chunk in valid_batch
        ]).to(self.device, dtype=torch.bfloat16)  # [B, 50, 128]

        # --- Advantages ---
        advantages = torch.tensor(
            [chunk.advantage for chunk in valid_batch],
            device=self.device, dtype=torch.float32,
        )  # [B]

        # --- Encode observations through backbone ---
        # Fast path: if _compute_ref_log_probs has already cached per-chunk
        # encoded features for every chunk in this batch, rebuild the batch
        # tensors directly from cache and skip the backbone forward. The Eagle
        # backbone + state encoder are frozen (no LoRA), so their output is
        # identical regardless of LoRA weight updates in between — the cache
        # is semantically valid for the whole iteration.
        #
        # Slow path (fallback): re-encode observations. Taken when the cache
        # is not yet populated (first call from _compute_ref_log_probs) or
        # when any chunk is missing cached features.
        all_cached = all(
            c.cached_backbone_features is not None
            and c.cached_state_features is not None
            and c.cached_embodiment_id is not None
            for c in valid_batch
        )

        if all_cached:
            backbone_output, state_features, embodiment_id = (
                self._rebuild_encoded_from_cache(valid_batch)
            )
        else:
            # Follows the DenoisingLab pattern (denoising_lab.py:190-202):
            #   backbone_inputs, action_inputs = model.prepare_input(**collated)
            #   backbone_output = backbone(backbone_inputs)
            #   features = action_head._encode_features(backbone_output, action_inputs)
            with torch.no_grad():
                encode_result = self._encode_observations(valid_batch)

            if encode_result is None:
                return None

            backbone_output, state_features, embodiment_id = encode_result

        return {
            "actions": actions,
            "action_masks": action_masks,
            "initial_noise": initial_noise,
            "advantages": advantages,
            "backbone_output": backbone_output,
            "state_features": state_features,
            "embodiment_id": embodiment_id,
            # Per-row mode list ("fixed" or "jitter"), parallel to valid_batch.
            # Used by _grpo_update_inner to decide which rows get ξ-jittered
            # input noise. _compute_ref_log_probs ignores this — the ref pass
            # always uses original ε (Jitter-GRPO anchors the cached ref at
            # the original ε so the same baseline serves both branches).
            "modes": modes,
        }, valid_batch

    def _rebuild_encoded_from_cache(self, valid_batch: list[ActionChunk]):
        """Restack per-chunk cached features into batched tensors.

        Each chunk stores its features UNPADDED (at its own seq_len). To put
        them in a minibatch we pad them all to the minibatch's max seq_len.
        This mirrors what the backbone's internal padding does, so the output
        has the same shape contract as _encode_observations() would produce.

        We require the image_mask and backbone_attention_mask cache state to
        be uniform across a minibatch: if some chunks have a mask and others
        don't, zero-filling the missing rows would silently turn their image
        tokens into non-image tokens (changing the cross-attention routing in
        AlternateVLDiT). All chunks come from the same iteration's collection,
        so this should always be uniform — a mismatch means upstream caching
        went wrong and we'd rather fail loudly than train on corrupted masks.
        """
        B = len(valid_batch)

        # Determine padding target
        seq_lens = [c.cached_backbone_features.shape[0] for c in valid_batch]
        max_seq = max(seq_lens)
        D = valid_batch[0].cached_backbone_features.shape[1]
        feat_dtype = valid_batch[0].cached_backbone_features.dtype

        backbone_features = torch.zeros(B, max_seq, D, device=self.device, dtype=feat_dtype)

        # Enforce uniformity: either all chunks have the mask or none do.
        # Explicit raise (not assert) because `python -O` strips asserts and
        # silently zero-filling a missing mask would corrupt image-token routing.
        attn_present = [c.cached_backbone_attn_mask is not None for c in valid_batch]
        img_present = [c.cached_image_mask is not None for c in valid_batch]
        if not (all(attn_present) or not any(attn_present)):
            raise RuntimeError(
                f"Inconsistent cached_backbone_attn_mask across minibatch: {attn_present}. "
                "All chunks must have the same mask cache state."
            )
        if not (all(img_present) or not any(img_present)):
            raise RuntimeError(
                f"Inconsistent cached_image_mask across minibatch: {img_present}. "
                "All chunks must have the same mask cache state."
            )
        has_attn = all(attn_present)
        has_img = all(img_present)

        backbone_attn_mask = None
        image_mask = None

        if has_attn:
            backbone_attn_mask = torch.zeros(
                B, max_seq, device=self.device,
                dtype=valid_batch[0].cached_backbone_attn_mask.dtype,
            )
        if has_img:
            image_mask = torch.zeros(
                B, max_seq, device=self.device,
                dtype=valid_batch[0].cached_image_mask.dtype,
            )

        for i, c in enumerate(valid_batch):
            sl = seq_lens[i]
            backbone_features[i, :sl] = c.cached_backbone_features
            if has_attn:
                backbone_attn_mask[i, :sl] = c.cached_backbone_attn_mask
            if has_img:
                image_mask[i, :sl] = c.cached_image_mask

        state_features = torch.stack(
            [c.cached_state_features for c in valid_batch], dim=0
        )
        embodiment_id = torch.stack(
            [c.cached_embodiment_id for c in valid_batch], dim=0
        )

        backbone_output = {
            "backbone_features": backbone_features,
            "image_mask": image_mask,
            "backbone_attention_mask": backbone_attn_mask,
        }

        return backbone_output, state_features, embodiment_id

    def _encode_observations(self, batch: list[ActionChunk]):
        """Run Eagle backbone on a batch of observations.

        Follows the DenoisingLab pattern (denoising_lab.py:161-217):
        1. Convert each observation to VLAStepData
        2. Process through the Gr00tN1d6Processor
        3. Collate into a batch
        4. model.prepare_input() → backbone_inputs, action_inputs
        5. backbone(backbone_inputs) → backbone_output
        6. action_head._encode_features() → backbone_features + state_features

        Returns:
            Tuple of (backbone_output, state_features, embodiment_id) or None on failure.
            - backbone_output: BatchFeature with processed backbone_features + masks
            - state_features: [B, state_horizon, 1536] encoded state
            - embodiment_id: [B] tensor of embodiment IDs
        """
        from gr00t.data.types import VLAStepData, MessageType
        from gr00t.data.embodiment_tags import EmbodimentTag

        embodiment_tag = EmbodimentTag[self.config.embodiment_tag]

        # Step 1-2: Build VLAStepData for each chunk and process
        processed_inputs = []
        for chunk in batch:
            vla = VLAStepData(
                images=chunk.video_frames,
                states=chunk.state,
                actions={},  # No ground-truth actions needed for feature encoding
                text=chunk.language,
                embodiment=embodiment_tag,
            )
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate into batch
        collated = self.processor.collator(processed_inputs)

        # Step 4: model.prepare_input() splits into backbone and action head inputs
        backbone_inputs, action_inputs = self.model.prepare_input(**collated)

        # Step 5: Run backbone (frozen)
        backbone_output = self.model.backbone(backbone_inputs)

        # Step 6: Encode features (applies vlln + state encoder)
        features = self.model.action_head._encode_features(
            backbone_output, action_inputs
        )

        # Extract what we need
        embodiment_id = action_inputs.embodiment_id

        # Build the backbone_output dict that fm_log_prob expects
        fm_backbone_output = {
            "backbone_features": features.backbone_features,
            "image_mask": getattr(backbone_output, "image_mask", None),
            "backbone_attention_mask": getattr(backbone_output, "backbone_attention_mask", None),
        }

        return fm_backbone_output, features.state_features, embodiment_id

    def _start_server_thread(self):
        """Start the GRPO server in a background thread for collection.

        Uses the GRPOPolicyWrapper directly with the loaded model — no subprocess
        needed since both are in the same venv and share GPU memory.

        The server wraps our trained model to serve actions to the collector subprocess.
        """
        import threading
        from grpo_server import GRPOPolicyWrapper
        from gr00t.policy.server_client import PolicyServer
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import VLAStepData, MessageType

        # Build a minimal policy that reuses our already-loaded model + processor
        # without re-loading from disk (saves time and memory)
        trainer_ref = self  # Capture reference for inner class

        class _InPlacePolicy:
            """Minimal policy interface using the trainer's pre-loaded model."""

            def __init__(self):
                self.strict = False
                self.model = trainer_ref.model
                self.processor = trainer_ref.processor
                self.embodiment_tag = EmbodimentTag[trainer_ref.config.embodiment_tag]
                self.modality_configs = self.processor.get_modality_configs()[
                    self.embodiment_tag.value
                ]
                language_keys = self.modality_configs["language"].modality_keys
                self.language_key = language_keys[0]
                self.collate_fn = self.processor.collator

            def get_action(self, observation, options=None):
                """Standard policy interface: obs → decoded action dict."""
                unbatched = self._unbatch_observation(observation)
                processed_inputs = []
                states_list = []

                for obs in unbatched:
                    vla = VLAStepData(
                        images=obs.get("video", {}),
                        states=obs.get("state", {}),
                        actions={},
                        text=obs.get("language", {}).get(self.language_key, [""])[0],
                        embodiment=self.embodiment_tag,
                    )
                    states_list.append(vla.states)
                    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla}]
                    processed_inputs.append(self.processor(messages))

                collated = self.collate_fn(processed_inputs)

                # model.get_action() internally calls prepare_input() which handles
                # device/dtype conversion via tree.map_structure
                with torch.inference_mode():
                    model_pred = self.model.get_action(**collated)
                normalized_action = model_pred["action_pred"].float()

                batched_states = {}
                for k in self.modality_configs["state"].modality_keys:
                    batched_states[k] = np.stack([s[k] for s in states_list], axis=0)
                unnormalized = self.processor.decode_action(
                    normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
                )
                return {k: v.astype(np.float32) for k, v in unnormalized.items()}, {}

            def _unbatch_observation(self, observation):
                """Split batched observation into list of single observations.

                Batch size is inferred explicitly from the `video` modality
                (first axis of any video array). Relying on dict insertion
                order — picking "the first dict modality value" — is fragile:
                if Gr00tSimPolicyWrapper or any future caller ever inserts
                language first, language values are list[list[str]] (no
                .shape) and the fallback `batch_size = 1` silently drops
                most observations. Video is always present for ROBOCASA_PANDA_OMRON
                and shaped (B, T, H, W, C), so it's a reliable anchor.
                """
                batch_size = None
                video_dict = observation.get("video")
                if isinstance(video_dict, dict) and video_dict:
                    for v in video_dict.values():
                        if hasattr(v, "shape") and len(v.shape) > 0:
                            batch_size = v.shape[0]
                            break

                # Fallback: try state, then any other ndarray-like value.
                # Should never trigger for Panda Omron but kept defensive.
                if batch_size is None:
                    state_dict = observation.get("state")
                    if isinstance(state_dict, dict) and state_dict:
                        for v in state_dict.values():
                            if hasattr(v, "shape") and len(v.shape) > 0:
                                batch_size = v.shape[0]
                                break

                if batch_size is None:
                    raise RuntimeError(
                        "_unbatch_observation: could not determine batch size "
                        "from video or state modalities. Got top-level keys: "
                        f"{list(observation.keys())}. Expected a "
                        "Gr00tPolicy-style nested observation with a non-empty "
                        "'video' or 'state' dict of ndarrays."
                    )

                unbatched = []
                for i in range(batch_size):
                    single = {}
                    for mod_key, mod_val in observation.items():
                        if isinstance(mod_val, dict):
                            single[mod_key] = {k: v[i] for k, v in mod_val.items()}
                        elif hasattr(mod_val, "__getitem__"):
                            single[mod_key] = mod_val[i]
                        else:
                            single[mod_key] = mod_val
                    unbatched.append(single)
                return unbatched

            def reset(self, options=None):
                return {}

            def get_modality_config(self):
                """Return the per-embodiment modality config dict.

                PolicyServer registers a `get_modality_config` endpoint that
                forwards to the wrapped policy. The chain ends at
                _InPlacePolicy, so without this method any client call would
                AttributeError. Mirrors `Gr00tPolicy.get_modality_config`.
                """
                return self.modality_configs

        # Create policy → sim wrapper → GRPO wrapper
        # strict=False avoids observation validation during collection
        # (the collector may send slightly different formats)
        in_place_policy = _InPlacePolicy()
        sim_wrapper = Gr00tSimPolicyWrapper(in_place_policy, strict=False)

        grpo_wrapper = GRPOPolicyWrapper(
            policy=sim_wrapper,
            device=str(self.device),
            model_lock=self._model_lock,
        )

        server = PolicyServer(
            policy=grpo_wrapper,
            host=self.config.server_host,
            port=self.config.server_port,
        )

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        # PolicyServer binds in __init__ (server_client.py), so the port is
        # ready as soon as PolicyServer(...) returns — no need to wait here.
        return server, thread

    def _stop_server_thread(self, server_and_thread):
        """Stop the background server thread cleanly.

        Must properly close the ZMQ socket so the port can be reused next iteration.
        Simply setting running=False isn't enough because the server blocks on recv().
        """
        server, thread = server_and_thread
        server.running = False

        # Send a "kill" message to unblock the server's recv() loop
        # This causes it to exit the while loop and release the socket
        try:
            import zmq
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{self.config.server_host}:{self.config.server_port}")
            from gr00t.policy.server_client import MsgSerializer
            sock.send(MsgSerializer.to_bytes({"endpoint": "kill"}))
            # Brief wait for response (server sends back before exiting)
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            try:
                sock.recv()
            except zmq.error.Again:
                pass
            sock.close()
            ctx.term()
        except Exception:
            pass

        # Wait for thread to actually finish (with timeout)
        thread.join(timeout=3.0)

        # Force-close the socket if thread didn't exit cleanly
        if thread.is_alive():
            try:
                server.socket.close(linger=0)
            except Exception:
                pass

    def _log_config(self):
        """Log every GRPOConfig field to TensorBoard as a single text summary.

        Called once from setup(), so every run (fresh or resumed) records the
        exact hyperparameters that produced its curves — recoverable from the
        TensorBoard log dir alone even after grpo_config.py's defaults have
        since changed. Uses add_text rather than add_hparams: several fields
        are lists or unions (e.g. lora_target_modules, env_names,
        max_episode_steps: int | list[int]) that add_hparams' scalar-only
        hparam_dict rejects, and add_text renders as a readable table in
        TensorBoard's Text tab with no per-field type handling required.
        """
        if self.writer is None:
            return
        # Escape literal '|' in the repr'd value — otherwise a value that
        # happens to contain one (e.g. a custom wandb_run_name or a path)
        # would be misparsed as an extra markdown table column separator.
        # (The replace() is kept out of the f-string's {} to stay valid on
        # Python <3.12, which disallows backslashes inside f-string braces.)
        def _fmt_value(value):
            escaped = repr(value).replace("|", "\\|")
            return escaped

        rows = "\n".join(
            f"| {f.name} | {_fmt_value(getattr(self.config, f.name))} |"
            for f in dataclasses.fields(self.config)
        )
        table = f"| Parameter | Value |\n|---|---|\n{rows}"
        self.writer.add_text("config", table, global_step=0)

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
        """Log training metrics to TensorBoard and wandb."""
        if self.writer is None:
            return

        # Episode stats are only meaningful when the collector returned
        # data this iter. An empty `stats` dict means buffer.stats() saw
        # zero episodes (collection failed entirely); logging `.get(..., 0)`
        # defaults on that path would falsely show "0% success",
        # "0 groups", "0 num_steps", etc. — indistinguishable from a real
        # all-fail iter. Skip the whole episode/* block in that case.
        if stats:
            self.writer.add_scalar("episode/success_rate", stats.get("success_rate", 0), iteration)
            self.writer.add_scalar("episode/mean_reward", stats.get("mean_reward", 0), iteration)
            self.writer.add_scalar("episode/std_reward", stats.get("std_reward", 0), iteration)

            # Episode trajectory length — catches "model is rushing" failure mode
            # (mean_num_steps drops below baseline) before success_rate collapses.
            self.writer.add_scalar("episode/mean_num_steps", stats.get("mean_num_steps", 0), iteration)
            self.writer.add_scalar("episode/std_num_steps", stats.get("std_num_steps", 0), iteration)

            # Group quality. n_dead_groups → how many groups got std<1e-4 in
            # compute_advantages (or were singletons) and contributed zero
            # gradient. group_success_* → distribution shape across groups
            # (an iter-mean of 50% could be "all groups at 50%" or "half at
            # 100%, half at 0%" — very different).
            self.writer.add_scalar("episode/n_groups", stats.get("n_groups", 0), iteration)
            self.writer.add_scalar("episode/n_dead_groups", stats.get("n_dead_groups", 0), iteration)
            self.writer.add_scalar("episode/n_live_groups", stats.get("n_live_groups", 0), iteration)
            self.writer.add_scalar("episode/group_success_min", stats.get("group_success_min", 0), iteration)
            self.writer.add_scalar("episode/group_success_median", stats.get("group_success_median", 0), iteration)
            self.writer.add_scalar("episode/group_success_max", stats.get("group_success_max", 0), iteration)

            # Anchor groups (all-success). Only emitted when the feature is on,
            # so prior runs keep exactly their existing episode/* key set.
            # n_anchor_groups rising while n_live_groups falls is the expected
            # shape at high success — it is the buffer that used to vanish.
            if self.config.include_anchor_groups:
                self.writer.add_scalar(
                    "episode/n_anchor_groups", stats.get("n_anchor_groups", 0), iteration
                )
                self.writer.add_scalar(
                    "episode/n_anchor_episodes",
                    stats.get("n_anchor_episodes", 0),
                    iteration,
                )
                self.writer.add_scalar(
                    "episode/n_anchor_episodes_dropped",
                    stats.get("n_anchor_episodes_dropped", 0),
                    iteration,
                )
                # The two counts the skip decision is keyed on. Without them a
                # TB-only operator cannot tell WHY an iteration was skipped.
                self.writer.add_scalar(
                    "episode/n_signal_chunks", stats.get("n_signal_chunks", 0),
                    iteration,
                )
                self.writer.add_scalar(
                    "episode/n_anchor_chunks", stats.get("n_anchor_chunks", 0),
                    iteration,
                )

            # Advantage signal availability (already in buffer.stats() but
            # previously not surfaced to TB). pct_positive_advantage near 0.5 is
            # healthy; far off means the group-relative normalization is failing.
            self.writer.add_scalar("episode/mean_advantage", stats.get("mean_advantage", 0), iteration)
            self.writer.add_scalar("episode/std_advantage", stats.get("std_advantage", 0), iteration)
            self.writer.add_scalar(
                "episode/pct_positive_advantage",
                stats.get("pct_positive_advantage", 0),
                iteration,
            )

        # Dynamic-epoch diagnostics. Only emitted when dynamic_epoch_training=True
        # AND at least one optimizer step actually fired. Gating on n_updates>0
        # prevents logging "planned" epoch counts on iters where all minibatches
        # were skipped (non-finite loss) — the name "actual_epochs" should reflect
        # what was executed, not what was planned.
        if (self.config.dynamic_epoch_training and update_stats is not None
                and update_stats.get("n_updates", 0) > 0):
            if "actual_epochs" in update_stats:
                self.writer.add_scalar(
                    "balanced/actual_epochs",
                    update_stats["actual_epochs"],
                    iteration,
                )
            if "success_fraction" in update_stats:
                self.writer.add_scalar(
                    "balanced/success_fraction",
                    update_stats["success_fraction"],
                    iteration,
                )

        # Update-counter scalars: log even when n_updates=0, because seeing
        # n_updates=0 IS the diagnostic signal — it pinpoints which iters
        # never fired a step (dead-group filter / non-finite loss). Without
        # logging these, a skipped iter would just show as a gap in TB
        # train/loss instead of a clear n_updates=0 bar.
        if update_stats is not None:
            self.writer.add_scalar(
                "train/n_updates",
                update_stats.get("n_updates", 0),
                iteration,
            )
            # Trained minibatches. Equal to n_updates without gradient
            # accumulation and ~k× larger with it, so the pair separates "how
            # much data did the update see" from "how many optimizer steps did
            # it take". Logged unconditionally (same rationale as n_updates):
            # `.get(..., 0)` is correct on the n_updates=0 early-return path,
            # where no micro-batch trained either.
            self.writer.add_scalar(
                "train/n_micro_batches",
                update_stats.get("n_micro_batches", 0),
                iteration,
            )
            self.writer.add_scalar(
                "train/n_skipped_nonfinite",
                update_stats.get("n_skipped_nonfinite", 0),
                iteration,
            )
            # Optimizer steps dropped because the accumulated gradient was
            # non-finite. Logged unconditionally: a flat zero line is the
            # healthy reading, and anything above zero means LoRA weights were
            # protected from a NaN write — correlate with train/n_updates
            # (which excludes the dropped step) and with the console WARNING.
            self.writer.add_scalar(
                "train/n_nonfinite_grad_steps",
                update_stats.get("n_nonfinite_grad_steps", 0),
                iteration,
            )
            # Accumulation-only counters: absent (no curve) unless
            # gradient_accumulation_steps > 1.
            for key in ("grad_accum_steps", "n_partial_windows"):
                if key in update_stats:
                    self.writer.add_scalar(
                        f"train/{key}", update_stats[key], iteration
                    )

        # Reference-MSE diagnostics. Deliberately OUTSIDE the n_updates > 0
        # gate below: MSE_ref is a property of (collected data x reference
        # policy) measured before any optimizer step, so it is still valid — and
        # still worth seeing — on an iteration whose update was skipped. It is
        # None only when the ref pass itself did not run (no live chunks), which
        # leaves a gap rather than a fake 0. See _summarize_ref_mse for why
        # these curves matter: ref_mse/ratio_ceiling_max vs clip_eps_high tells
        # you whether the upper clip is reachable at all, and ref_mse/pos_mean
        # decaying toward 0 is positive-branch saturation.
        # getattr for the same reason as in _jitter_gap_diagnostics: a trainer
        # built without __init__ (CPU tests) must still be able to log.
        #
        # NON-FINITE FILTER on all three blocks below. Every other numeric TB
        # path in this file filters (ratio_maxes/ratio_mins, grad_norms,
        # phase_times) for the reason phase_times documents: one nan/inf poisons
        # wandb's chart autoscale for the REST OF THE RUN. These metrics derive
        # from ref_log_prob (never isfinite-checked upstream) and from two
        # diagnostic forwards, and np.exp(MSE_ref) overflows above MSE_ref ~709,
        # so inf/nan is reachable in principle. Drop the offending scalar, keep
        # the rest, and say so once.
        # NON-FINITE FILTER for the metrics added alongside this helper.
        #
        # SCOPE, stated precisely because an earlier version of this comment
        # over-claimed: exactly four pre-existing sites filter non-finite values
        # (ratio_maxes/ratio_mins, grad_norms, phase_times). `train/loss`,
        # `train/clipfrac`, `train/mean_ratio` and `train/mean_log_ratio_abs` do
        # NOT — that is pre-existing behaviour this change deliberately leaves
        # alone. Everything routed through _emit below IS filtered.
        #
        # Why it matters: phase_times documents the reason — one nan/inf poisons
        # wandb's chart autoscale for the REST OF THE RUN. And it is reachable
        # for the sign-split ratio metrics specifically: the comment on
        # ratio_maxes notes that bf16 `ratio = log_ratio.exp()` can overflow to
        # +inf while the clipped loss stays FINITE (for A>0,
        # min(A*inf, A*(1+hi)) is the finite bound), so an inf ratio survives the
        # torch.isfinite(loss) guard and lands in ratio_sum_*. ref_log_prob is
        # likewise never isfinite-checked upstream, and np.exp(MSE_ref) overflows
        # above MSE_ref ~709.
        def _emit(prefix: str, d: dict) -> None:
            # Non-numeric values are skipped rather than raising: math.isfinite
            # throws TypeError on a str/None, and the TB half of _log_metrics has
            # no try/except (unlike the _jitter_gap_diagnostics callsite). An
            # iteration carries ~13 minutes of collected simulation by the time it
            # gets here, so a malformed diagnostic entry must cost that entry, not
            # the iteration.
            bad = []
            for key, val in d.items():
                try:
                    ok = math.isfinite(val)
                except TypeError:
                    ok = False
                if ok:
                    self.writer.add_scalar(f"{prefix}/{key}", val, iteration)
                else:
                    bad.append(key)
            if bad:
                print(
                    f"  WARNING: dropped non-finite/non-numeric {prefix}/* "
                    f"scalar(s) {bad} at iteration {iteration} rather than "
                    f"poisoning the charts."
                )

        _ref_mse = getattr(self, "_ref_mse_stats", None)
        if _ref_mse:
            _emit("ref_mse", _ref_mse)

        # Per-chunk gap survey. Ungated on n_updates for the same reason as
        # ref_mse/*: measured at theta == theta_ref, before any step. The headline
        # is chunk_gap/cv -- compare it against the ~4-8% intrinsic xi-sampling
        # floor to decide whether any per-chunk treatment is worth building.
        _cg = getattr(self, "_chunk_gap_stats", None)
        if _cg:
            _emit("chunk_gap", _cg)

        # Jitter gap measurement. Under its own `jitter/` prefix because it is a
        # single-minibatch snapshot taken at theta == theta_ref, not a per-mb
        # mean over the iteration like every `train/` scalar. Absent when jitter
        # is off or when the first minibatch drew no positive-advantage rows,
        # which leaves a curve gap rather than a fake 0.
        #
        # Reading guide:
        #   jitter/headroom_multiplier  = (MSE_ref_pos + gap_pos) / MSE_ref_pos.
        #       How many times more usable log-ratio room a positive row has WITH
        #       jitter than without. ~1.0 means jitter_pos is doing nothing.
        #   jitter/jacobian_fro_sq      = the field's noise-sensitivity with the
        #       (1-tau)^2 * lambda^2 prefactor divided out, so it is comparable
        #       ACROSS different jitter_pos settings.
        #   jitter/gap_at_tau{k}        = per-tau profile (positive rows), k
        #       indexing config.tau_centers. Shows WHERE along the denoising
        #       path the field is noise-sensitive.
        #   jitter/neg_clip_budget_used = fraction of the erosion clip budget the
        #       negative rows' gap eats before any drift. -> 1.0 means every
        #       negative row is born clipped; that is the ceiling on jitter_neg.
        #   jitter/n_rows_{pos,neg}     = row counts backing the two gaps. A
        #       ONE-MINIBATCH count, not an iteration total — the whole jitter/*
        #       block is a single-minibatch snapshot.
        #   jitter/gap_fixed_rows_selfcheck = must be ~0 (paired mode only).
        if update_stats and update_stats.get("_jitter_diag"):
            _emit("jitter", update_stats["_jitter_diag"])

        # Effective clipfrac. Ungated on n_updates for the same reason as the two
        # blocks above: it is populated by micro-batches that TRAINED, which
        # includes the iteration where every gradient window was then dropped —
        # exactly the iteration whose diagnosis this metric helps with.
        #
        # Expect clipfrac_effective_pos == 0 at any sane jitter_pos: a
        # post-renorm-positive row can only die on the UPPER bound, which needs
        # ratio > 1+clip_eps_high (~1.2) against the analytic ceiling
        # e^MSE_ref ~= 1.01-1.05. If _pos goes non-zero, either the ratio really
        # did exceed the upper bound or clip_eps_high was lowered below reach.
        # THAT is the property this metric exists for, and it holds exactly.
        #
        # Do NOT read clipfrac_effective_neg as a drop-in for clipfrac_*_neg:
        # under the default per-minibatch renorm the two have different
        # DENOMINATORS. This one buckets by post-renorm sign, which z-scoring
        # puts near half the rows below zero for, regardless of success rate;
        # clipfrac_{fixed,jitter}_neg uses the true group-relative negative
        # count. Worse, a group-good row carrying lam = jitter_pos (hence a large
        # gap, hence a ratio well under 1-clip_eps_low) that renorm then flipped
        # negative is a genuine lower-bound death booked HERE — so a large
        # jitter_pos inflates _neg. Cross-reference n_pos_flipped_by_renorm
        # before attributing a rise in _neg to real erosion clipping.
        if update_stats:
            _eff = {
                k: update_stats[k]
                for k in ("clipfrac_effective_pos", "clipfrac_effective_neg")
                if k in update_stats
            }
            if _eff:
                _emit("train", _eff)

        # Loss / ratio / grad scalars are only meaningful when at least one
        # optimizer.step() actually fired. `_grpo_update_inner` returns
        # `{"n_skipped_nonfinite": N}` when every minibatch got skipped for
        # non-finite loss — a truthy dict missing every loss/ratio key. The
        # old `if update_stats:` gate then emitted `train/loss=0`,
        # `train/grad_norm_mean=0`, `train/mean_ratio=1` etc. as `.get(...)`
        # defaults — fake values that pollute the TB curves.
        if update_stats and update_stats.get("n_updates", 0) > 0:
            self.writer.add_scalar("train/loss", update_stats.get("loss", 0), iteration)
            self.writer.add_scalar("train/clip_loss", update_stats.get("clip_loss", 0), iteration)
            # Two KL anchors: kl_loss_last_iter (KL to start-of-iter ref policy,
            # always emitted when n_updates>0) and kl_loss_base_model (KL to
            # the pretrained DiT, only emitted when kl_coef_base_model > 0 —
            # the inner loop only adds the key in that case).
            self.writer.add_scalar(
                "train/kl_loss_last_iter",
                update_stats.get("kl_loss_last_iter", 0),
                iteration,
            )
            if "kl_loss_base_model" in update_stats:
                self.writer.add_scalar(
                    "train/kl_loss_base_model",
                    update_stats["kl_loss_base_model"],
                    iteration,
                )
            self.writer.add_scalar("train/clipfrac", update_stats.get("clipfrac", 0), iteration)
            self.writer.add_scalar("train/mean_ratio", update_stats.get("mean_ratio", 1), iteration)
            self.writer.add_scalar(
                "train/mean_log_ratio_abs",
                update_stats.get("mean_log_ratio_abs", 0),
                iteration,
            )
            # Gradient norm BEFORE clipping (mean/max across minibatches).
            # The primary "is anything actually training?" signal — if this
            # stays near 0 across many iters, the FM log-prob gradient
            # vanishes regardless of clip_loss appearance.
            self.writer.add_scalar(
                "train/grad_norm_mean",
                update_stats.get("grad_norm_mean", 0),
                iteration,
            )
            self.writer.add_scalar(
                "train/grad_norm_max",
                update_stats.get("grad_norm_max", 0),
                iteration,
            )
            # Ratio distribution tails. With mean_ratio≈1 and modest
            # clipfrac, large ratio_max/small ratio_min reveal outlier
            # minibatches doing all the clipping work.
            self.writer.add_scalar("train/ratio_max", update_stats.get("ratio_max", 1), iteration)
            self.writer.add_scalar("train/ratio_min", update_stats.get("ratio_min", 1), iteration)

            # Per-branch metrics (Jitter-GRPO). Only emitted when the
            # corresponding branch fired any rows this iter — so vanilla
            # GRPO runs (jitter off) see no `_jitter` curves at all,
            # and a partial iter where one branch's rows were all dead-group-
            # filtered just skips that iter's scalar instead of emitting 0.
            # The fixed/jitter gap on mean_log_ratio_abs IS the empirical
            # Jacobian-norm signal that Jitter-GRPO is designed to surface —
            # if it shrinks across iters, the regularizer is doing its job.
            #
            # KL terms are split into _last_iter and _base_model variants.
            # _base_model_{fixed,jitter} keys are only present when the
            # base-model anchor is active AND that branch fired rows, so
            # the `if key in update_stats` gate naturally suppresses them
            # for vanilla runs and for runs with kl_coef_base_model=0.
            for branch in ("fixed", "jitter"):
                for metric in (
                    "mean_ratio",
                    "mean_log_ratio_abs",
                    "kl_loss_last_iter",
                    "kl_loss_base_model",
                ):
                    key = f"{metric}_{branch}"
                    if key in update_stats:
                        self.writer.add_scalar(
                            f"train/{key}", update_stats[key], iteration
                        )
                # Clipfrac is split by advantage sign (pos/neg) — see
                # `clipfrac_sum_fixed_pos` etc. in _grpo_update_inner.
                # Gated independently of the other branch metrics: a bucket
                # with zero rows this iter is just absent from update_stats.
                # mean_ratio / mean_log_ratio_abs are split the same way: the
                # pooled versions above are dominated by the positive rows'
                # jitter bias, so the sign-split pair is what makes each branch
                # legible (mean_ratio_jitter_pos starting at e^-gap_pos and
                # moving up = positive-branch headroom being consumed).
                #
                # Routed through _emit (unlike the un-split versions above) so
                # the sign-split ratio metrics get the non-finite filter: a bf16
                # `ratio = exp(log_ratio)` overflow reaches ratio_sum_* while the
                # clipped loss stays finite, so +inf here is reachable in a way
                # it is not for the pre-existing curves.
                _split = {
                    f"{metric}_{branch}_{sign}": update_stats[
                        f"{metric}_{branch}_{sign}"
                    ]
                    for sign in ("pos", "neg")
                    for metric in ("clipfrac", "mean_ratio", "mean_log_ratio_abs")
                    if f"{metric}_{branch}_{sign}" in update_stats
                }
                if _split:
                    _emit("train", _split)

            # Effective clipfrac moved OUT of this block — see below. It is
            # populated by any micro-batch that trained, which includes
            # iterations where every gradient window was later dropped
            # (n_updates == 0, n_micro_batches > 0), so gating it on
            # n_updates > 0 would have made the early-return copy dead code.

            # Dynamic positive-advantage weight (pos_adv_*: present only when
            # positive_advantage_weight_scaling ran this iter) and the sign-flip
            # counter (n_pos_flipped_by_renorm: emitted every successful iter).
            # pos_adv_weight_k is the headline curve (sits in [1, ..._max]); the
            # mass terms show what drove it. n_pos_flipped_by_renorm reads 0 under
            # per-iteration norm and >0 under per-minibatch norm (the artifact).
            for key in (
                "pos_adv_weight_k",
                "pos_adv_alive_neg_mass",
                "pos_adv_pos_mass",
                "n_pos_flipped_by_renorm",
                "balanced_pos_ratio",
                # Anchor rows (present only when anchor rows trained this iter).
                # mean_ratio_anchor is the one to watch: it starts at 1.0 and
                # saturating at 1 + clip_eps_high means the clip is bounding the
                # retention move, which is the designed cap.
                "n_anchor_rows_trained",
                "mean_ratio_anchor",
                "kl_loss_anchor",
            ):
                if key in update_stats:
                    self.writer.add_scalar(f"train/{key}", update_stats[key], iteration)

        if lr is not None:
            self.writer.add_scalar("train/learning_rate", lr, iteration)

        if iter_time is not None:
            self.writer.add_scalar("time/iteration_seconds", iter_time, iteration)

        # Phase-time breakdown (collect / collect_rollout / collect_load /
        # advantage / ref_logprob / update). The trainer already times each
        # phase for its console summary; surface them here so TB can answer
        # "which phase regressed?" without parsing logs. `collect` is the total
        # of Phase 1; collect_rollout + collect_load decompose it.
        # NaN values are sentinels indicating "no real work ran this iter
        # for this phase" (e.g., collect=nan when the iter reused cached
        # episodes via resume_from_collected_data) — skip them so the curve
        # has a clean gap rather than a misleading data point at ~0.
        if phase_times is not None:
            for phase_name, secs in phase_times.items():
                if not math.isnan(secs):
                    self.writer.add_scalar(
                        f"time/{phase_name}_seconds", secs, iteration
                    )

        # Cumulative L2 distance of LoRA params from their setup-time snapshot.
        # The "has the policy actually moved?" diagnostic: if this stays near
        # zero across iters, no amount of clip_loss / mean_log_ratio_abs
        # commentary matters — the model is unchanged.
        if lora_delta_norm is not None:
            self.writer.add_scalar("lora/weight_delta_norm", lora_delta_norm, iteration)

        # Wandb logging. Mirror the TB gates so the wandb dashboard doesn't
        # show fake zeros either.
        if self.config.use_wandb:
            try:
                import wandb
                log_dict = {"iteration": iteration}
                if stats:
                    log_dict.update(stats)
                    if not self.config.include_anchor_groups:
                        # buffer.stats() reports the anchor counters
                        # unconditionally, but the TB side gates them on the
                        # flag — drop them here too so an anchors-off run's key
                        # set is exactly what it was before the feature existed.
                        # Must list EVERY anchor-related key stats() adds
                        # unconditionally — it grew from three to five when the
                        # chunk counts were added for the skip decision.
                        for _k in ("n_anchor_groups", "n_anchor_episodes",
                                   "n_anchor_episodes_dropped",
                                   "n_signal_chunks", "n_anchor_chunks"):
                            log_dict.pop(_k, None)
                # Mirror the TB-side ref_mse/* block, which is deliberately
                # ungated on n_updates (see the comment there). getattr, not
                # attribute access: on a trainer built without __init__ a bare
                # read raises AttributeError, and the `except Exception: pass`
                # around this whole block would then silently drop the ENTIRE
                # iteration's wandb payload rather than just this one metric.
                # Non-finite values are filtered for the same
                # chart-autoscale-poisoning reason as the TB side.
                _ref_mse_w = getattr(self, "_ref_mse_stats", None)
                if _ref_mse_w:
                    log_dict.update({
                        f"ref_mse/{k}": v for k, v in _ref_mse_w.items()
                        if math.isfinite(v)
                    })
                _cg_w = getattr(self, "_chunk_gap_stats", None)
                if _cg_w:
                    log_dict.update({
                        f"chunk_gap/{k}": v for k, v in _cg_w.items()
                        if math.isfinite(v)
                    })
                if update_stats is not None:
                    # Counters always; loss/ratio/grad only when n_updates>0
                    # (matching the TB-side gating). n_micro_batches is a
                    # counter too, so it's mirrored here for TB parity — the
                    # gated block below re-sets it to the same value when
                    # n_updates>0, which is a harmless no-op.
                    log_dict["train/n_updates"] = update_stats.get("n_updates", 0)
                    log_dict["train/n_micro_batches"] = (
                        update_stats.get("n_micro_batches", 0)
                    )
                    log_dict["train/n_skipped_nonfinite"] = (
                        update_stats.get("n_skipped_nonfinite", 0)
                    )
                    log_dict["train/n_nonfinite_grad_steps"] = (
                        update_stats.get("n_nonfinite_grad_steps", 0)
                    )
                    if update_stats.get("n_updates", 0) > 0:
                        log_dict.update({
                            f"train/{k}": v
                            for k, v in update_stats.items()
                            if k not in (
                                "n_updates", "n_skipped_nonfinite",
                                # Handled by the gated dynamic_epoch_training
                                # block below; exclude here to avoid both a
                                # spurious train/actual_epochs curve on vanilla
                                # runs and a double-log (train/ + balanced/) on
                                # dynamic-epoch runs.
                                "actual_epochs", "success_fraction",
                                # Nested dict, not a scalar — mirrored under the
                                # jitter/ prefix just below, same as the TB side.
                                "_jitter_diag",
                                # Excluded so the finite-filtered copies added
                                # below are the ONLY source of these keys.
                                # Without this exclusion the unfiltered value
                                # would already be in log_dict and the filtered
                                # block could not remove it — dict.update cannot
                                # un-set a key — so wandb would receive a
                                # non-finite scalar on every n_updates > 0
                                # iteration while the TB side printed
                                # "dropped ... rather than poisoning the charts".
                                "clipfrac_effective_pos",
                                "clipfrac_effective_neg",
                            )
                        })
                    # Mirror the TB-side jitter/* block. Ungated on n_updates for
                    # the same reason ref_mse/* is: the measurement happens at
                    # theta == theta_ref, before any step could have fired.
                    if update_stats.get("_jitter_diag"):
                        log_dict.update({
                            f"jitter/{k}": v
                            for k, v in update_stats["_jitter_diag"].items()
                            if math.isfinite(v)
                        })
                    # Effective clipfrac, also ungated (populated by any
                    # micro-batch that trained, including on a dropped-window
                    # iteration).
                    log_dict.update({
                        f"train/{k}": update_stats[k]
                        for k in ("clipfrac_effective_pos",
                                  "clipfrac_effective_neg")
                        if k in update_stats and math.isfinite(update_stats[k])
                    })
                if lr is not None:
                    log_dict["train/lr"] = lr
                if phase_times is not None:
                    # Skip NaN entries — same sentinel as the TB path. wandb
                    # would otherwise log nan and break its chart autoscale
                    # for the rest of the run.
                    log_dict.update({
                        f"time/{k}_seconds": v
                        for k, v in phase_times.items()
                        if not math.isnan(v)
                    })
                if lora_delta_norm is not None:
                    log_dict["lora/weight_delta_norm"] = lora_delta_norm
                if (self.config.dynamic_epoch_training and update_stats is not None
                        and update_stats.get("n_updates", 0) > 0):
                    if "actual_epochs" in update_stats:
                        log_dict["balanced/actual_epochs"] = update_stats["actual_epochs"]
                    if "success_fraction" in update_stats:
                        log_dict["balanced/success_fraction"] = update_stats["success_fraction"]
                wandb.log(log_dict)
            except Exception:
                pass

    def _compute_lora_delta_norm(self) -> float:
        """L2 norm of (current trainable params − snapshot taken at setup time).

        Tracks cumulative drift of LoRA weights SINCE THIS RUN STARTED.
        Resumed runs reset the baseline at setup (snapshot post-load), so
        the metric measures within-run drift, not drift from PEFT init.

        Diagnostic intent: when training appears to fire (n_updates > 0,
        non-zero loss) but episode metrics don't budge, this number tells
        you whether the weights themselves are actually moving. A flat
        curve here means the optimizer steps are too small to change the
        policy regardless of what the loss says.

        Accumulates the squared-delta sum on-device with a single sync at
        the end. The naive per-param `.item()` pattern triggers one
        GPU→CPU sync per LoRA tensor (~hundreds), measurably stalling the
        log call on real hardware.
        """
        if not getattr(self, "_lora_init_params", None):
            return 0.0
        total_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self._lora_init_params:
                    # All trainable params are fp32 post-upcast (see setup()),
                    # so this cast is a no-op in the common path but keeps
                    # the subtraction safe if a future refactor leaves any
                    # trainable param in bf16.
                    delta = p.detach().float() - self._lora_init_params[name].float()
                    total_sq = total_sq + delta.pow(2).sum()
        return float(total_sq.sqrt().item())

    def _save_checkpoint(self, iteration: int):
        """Save LoRA weights and optimizer state."""
        ckpt_dir = Path(self.config.checkpoint_dir) / f"iter_{iteration:04d}"
        save_lora_checkpoint(self.model, ckpt_dir)

        # Save optimizer state with the param-name list alongside it. The
        # name list is REQUIRED for resume to detect a positional permutation
        # of same-shape LoRA params (see _validate_optimizer_param_names).
        # Wrapping into a dict instead of writing two files keeps the load
        # atomic and removes any risk of mismatched sidecar files.
        torch.save(
            {
                "optimizer_state": self.optimizer.state_dict(),
                "param_names": self._lora_param_names,
            },
            ckpt_dir / "optimizer.pt",
        )
        print(f"  Checkpoint saved: {ckpt_dir}")

    def _save_checkpoint_for_skipped_iter(self, iteration: int):
        """Save a resume point for an iter whose gradient update did NOT fire.

        Names the dir after `_last_updated_iteration` (the iter whose state
        we'd actually be restoring), not the current loop iter. That way:
          - resume from this dir → start_iteration = last_updated + 1, which
            is exactly the skipped iter — it gets a fresh attempt rather than
            being burned from num_iterations.
          - LR scheduling on resume matches what the skipped iter would have
            seen (frac = 1 - (last_updated)/num_iterations), since LR is
            recomputed per-iter from the loop counter.
          - If the dir already exists (e.g., the previous successful iter
            was a save_interval boundary), skip the write — the on-disk
            state is already exactly what we'd be saving.
        """
        target = self._last_updated_iteration
        if target <= 0:
            print(
                f"  Skip checkpoint at iter {iteration}: no successful "
                f"update has run yet — model is still base weights."
            )
            return
        ckpt_dir = Path(self.config.checkpoint_dir) / f"iter_{target:04d}"
        if ckpt_dir.exists():
            print(
                f"  Skip checkpoint at iter {iteration}: iter_{target:04d}/ "
                f"already exists (resume from there to retry iter {target + 1})."
            )
            return
        self._save_checkpoint(target)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Launch GRPO training."""
    try:
        import tyro
        config = tyro.cli(GRPOConfig)
    except ImportError:
        # Fallback: use defaults
        print("tyro not available, using default config")
        config = GRPOConfig()

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create trainer and run. Both setup() and train() are wrapped so that
    # shutdown() runs even when setup() raises mid-way — setup() has many
    # raise paths (cache validator, optimizer state validation, LoRA load
    # mismatch) that can fire after the TensorBoard writer / background
    # server thread are created. Without the wrap, those raises bypass
    # shutdown() and leak the server socket / writer.
    trainer = GRPOTrainer(config)
    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
