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

        # Optional long-running collector server. When configured, collection
        # is an RPC call instead of a subprocess spawn — eliminates per-iter
        # robocasa import + worker startup cost (~10-20s/iter). The server
        # must be started separately (see scripts/grpo/collector_server.py).
        # We ping at __init__ to fail fast if the server is unreachable, is
        # missing any of our configured env_names, or was booted with
        # bake-time args (group_size, n_action_steps, max_episode_steps)
        # that don't match this trainer's config.
        self._collector_client = None
        if self.config.collector_server_host:
            sys.path.insert(0, str(Path(__file__).parent))
            from collector_server import CollectorClient
            # Scale RPC timeout from the EFFECTIVE upper bound on groups
            # this iter. With dynamic mode active (min_alive_groups>0
            # and max_groups>num_groups in EpisodeCollector.collect), the
            # collector may run up to max_groups groups; otherwise it stops
            # at exactly num_groups. ~7 min/group on the user's setup. With
            # num_async_vector_env < group_size each group is collected over
            # turns_per_group sequential turns, so scale by that factor too
            # (turns_per_group == 1 in the default one-env-per-rollout case).
            effective_max_groups = (
                self.config.max_groups
                if self.config.min_alive_groups > 0
                and self.config.max_groups > self.config.num_groups
                else self.config.num_groups
            )
            self._collector_client = CollectorClient(
                host=self.config.collector_server_host,
                port=self.config.collector_server_port,
                # Cap below ZMQ's int32 RCVTIMEO limit (~2.147e9 ms). The scaled
                # value can exceed it at extreme configs (e.g. group_size>=52,
                # num_envs=1, max_groups=100 → turns_per_group=52), which would
                # otherwise raise OverflowError at socket setup. ~2e9 ms (~23
                # days) is effectively unbounded for collection anyway.
                timeout_ms=min(
                    420_000 * effective_max_groups * self._turns_per_group(),
                    2_000_000_000,
                ),
            )
            try:
                info = self._collector_client.ping()
            except Exception as e:
                raise RuntimeError(
                    f"Could not reach collector server at "
                    f"{self.config.collector_server_host}:{self.config.collector_server_port}: "
                    f"{type(e).__name__}: {e}. "
                    f"Start it first with: "
                    f"`gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python "
                    f"scripts/grpo/collector_server.py --env-names ... "
                    f"--max-episode-steps ... --listen-port "
                    f"{self.config.collector_server_port}`."
                )
            self._validate_collector_server_config(info)
            print(
                f"  Collector server: {self.config.collector_server_host}:"
                f"{self.config.collector_server_port} "
                f"(envs: {sorted(info.get('envs', []))}, "
                f"group_size={info.get('group_size')}, "
                f"num_async_vector_env={info.get('num_async_vector_env')}, "
                f"n_action_steps={info.get('n_action_steps')})"
            )

    def setup(self):
        """Load the model + LoRA, configure optimizer, validate the resume
        cache (when ``resume_from_collected_data=True``), and start the
        persistent inference server thread.

        Heavy work that's deferred from ``__init__`` so a misconfigured
        trainer fails fast at construction without paying for the multi-
        minute model load. Note: ``__init__`` ALREADY binds to the
        optional collector RPC server (``collector_server_*`` fields), so
        those fields cannot be mutated post-construction; only the
        non-RPC config fields are still mutable here.
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

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # --- Start persistent server ---
        # The server shares self.model, so LoRA weight updates are reflected automatically
        self._server_handle = self._start_server_thread()

        print("\n" + "=" * 60)
        print("Setup complete. Ready to train.")
        print("=" * 60)

    def shutdown(self):
        """Clean up resources (server thread, tensorboard writer, collector client)."""
        if hasattr(self, '_server_handle') and self._server_handle is not None:
            self._stop_server_thread(self._server_handle)
            self._server_handle = None
        # Close collector RPC client (the server itself stays running for the
        # next trainer instance — that's the whole point of long-running mode).
        if getattr(self, '_collector_client', None) is not None:
            try:
                self._collector_client.close()
            except Exception as e:
                print(f"WARN: failed to close collector client: {e}")
            self._collector_client = None
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
        if self.config.jitter_lambda > 0.0:
            # Surface the doubled-step cost up-front so the user can confirm
            # update_epochs has been halved if they want to match vanilla
            # GRPO's per-iter step budget. Single line, only when active.
            print(
                f"  Jitter-GRPO: lambda={self.config.jitter_lambda} "
                f"(paired scheduling — 2× minibatches per epoch; "
                f"halve update_epochs to match vanilla per-iter step count)"
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
                success_weight=self.config.success_weight,
                max_episode_steps=max_steps,
            )
            stats = self.buffer.stats()
            phase2_time = time.time() - phase2_start

            # Skip update if no gradient signal (all same outcome)
            if stats.get("std_reward", 0) < 1e-8:
                print(f"  Skipping update: all episodes have same reward (no gradient signal)")
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
            self._compute_ref_log_probs()

            # ═══ Phase 3: GRPO Policy Update ═══
            phase3_start = time.time()
            update_stats = self._grpo_update()
            phase3_time = time.time() - phase3_start

            # Treat an iter as "updated" only if at least one optimizer.step()
            # actually fired. Two paths lead to n_updates=0 here that the
            # outer std_reward<1e-8 skip-check above does NOT catch:
            #   1. Every minibatch had non-finite loss (bf16 ratio overflow).
            #   2. Every group's per-group std<1e-4 (so the dead-chunk filter
            #      in _grpo_update_inner left zero live chunks), but the
            #      GLOBAL std_reward exceeded 1e-8 — e.g., a mix of all-fail
            #      groups (rewards=0) and all-succeed-with-identical-num_steps
            #      groups (rewards=constant).
            # In both cases the model + optimizer are bit-identical to the
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
                    "advantage": phase2_time,
                    "update": phase3_time,
                },
                lora_delta_norm=self._compute_lora_delta_norm(),
            )

            collect_label = (
                f"cached ({phase1_time:.2f}s)"
                if used_cached_collection
                else f"{phase1_time:.0f}s"
            )
            print(
                f"  Time: collect={collect_label}, "
                f"advantage={phase2_time:.1f}s, "
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

        Dispatches to a long-running collector_server (when
        config.collector_server_host is set) or to a fresh subprocess of
        collect_episodes.py. Both paths write episodes as .npz files to
        episode_dir; we then load them into self.buffer and run the same
        failure handling for both modes.
        """
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

        # Run collection: RPC to long-running server if configured, else
        # spawn a fresh subprocess.
        if self._collector_client is not None:
            failure_reason = self._collect_via_server(env_name, episode_dir, ff_steps)
        else:
            failure_reason = self._collect_via_subprocess(
                env_name, episode_dir, max_steps, ff_steps,
            )

        # Common post-processing: load episodes, then handle any failure.
        n_loaded = 0
        if failure_reason is None:
            n_loaded = self.buffer.load_episodes(episode_dir)
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
                    f"TIME_WAIT, MUJOCO_GL backend missing, model OOM during inference, "
                    f"or (server mode) collector_server.py not running. "
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

        n_loaded = self.buffer.load_episodes(episode_dir)
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
        call — _collect_via_server is the long-running alternative.
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
            "--success-weight", str(self.config.success_weight),
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

    def _collect_via_server(
        self,
        env_name: str,
        episode_dir: Path,
        ff_steps: int,
    ) -> str | None:
        """Run collection via the long-running collector_server.

        Returns a failure_reason string, or None on success. The server holds
        its own max_episode_steps per env (set at server startup), so we
        don't pass max_steps here — if the trainer's config diverges from
        the server's, restart the server with the new values.

        Distinguishes FATAL server errors (config mismatches like env_name
        typo) from transient ones (timeouts, connection blips). Fatal errors
        re-raise immediately rather than burning the consecutive-failure
        retry budget — there's no point retrying when the cause won't
        self-correct.
        """
        from collector_server import FatalCollectorError
        try:
            result = self._collector_client.collect(
                env_name=env_name,
                output_dir=str(episode_dir),
                # Iter-stride 100_000 must match _collect_via_subprocess so
                # both transports produce identical group seeds for a given
                # (config.seed, iteration). See GROUP_SEED_STRIDE in
                # collect_episodes.py for the within-iter spacing.
                base_seed=self.config.seed + self.iteration * 100_000,
                num_groups=self.config.num_groups,
                success_weight=self.config.success_weight,
                fast_forward_steps=ff_steps,
                fast_forward_pct=self.config.fast_forward_pct,
                min_alive_groups=self.config.min_alive_groups,
                max_groups=self.config.max_groups,
                init_state_npz_path=self.config.init_state_npz_path,
            )
        except FatalCollectorError as e:
            raise RuntimeError(
                f"Collector server reports fatal config error: {e}. This "
                f"won't fix itself on retry — restart the collector server "
                f"with --env-names / --max-episode-steps / --group-size / "
                f"--num-async-vector-env / --n-action-steps matching this "
                f"trainer's config "
                f"(env_names={self.config.env_names}, "
                f"group_size={self.config.group_size}, "
                f"num_async_vector_env={self._resolved_num_async_vector_env()}, "
                f"n_action_steps={self.config.n_action_steps})."
            ) from e
        except TimeoutError as e:
            return f"collector_server timeout: {e}"
        except RuntimeError as e:
            # Server-reported transient error.
            return f"collector_server error: {e}"
        except Exception as e:
            return f"collector_server connection error ({type(e).__name__}): {e}"

        print(
            f"    [collector_server] {result['n_episodes']} episodes, "
            f"{result['n_successes']} successes in {result['elapsed_s']}s"
        )
        return None

    def _validate_collector_server_config(self, info: dict) -> None:
        """Check that the server's bake-time args match this trainer's config.

        Mismatches mean episodes will be collected with values inconsistent
        with what the trainer expects (advantage shape, chunking math).
        Fail fast at __init__ with the exact restart command rather than
        producing silently corrupt training data.
        """
        available_envs = set(info.get("envs", []))
        missing_envs = [e for e in self.config.env_names if e not in available_envs]
        if missing_envs:
            raise RuntimeError(
                f"Collector server is missing envs that this trainer needs: "
                f"{missing_envs}. Restart the server with --env-names "
                f"matching this config: {self.config.env_names}"
            )

        server_group_size = info.get("group_size")
        if server_group_size != self.config.group_size:
            raise RuntimeError(
                f"group_size mismatch: trainer config = {self.config.group_size}, "
                f"server (boot-time) = {server_group_size}. Restart the "
                f"collector server with --group-size {self.config.group_size}."
            )

        # num_async_vector_env: resolve both sides to the effective worker count
        # before comparing, so a trainer with None (→ group_size) matches a
        # server booted with --num-async-vector-env group_size and vice-versa.
        # An OLD server (pre-this-feature) returns None for the key → resolves
        # to its group_size; this passes silently when the trainer also uses the
        # default and correctly fails (with a restart hint) when the trainer
        # lowers num_async_vector_env below group_size.
        trainer_nave = self._resolved_num_async_vector_env()
        server_nave_raw = info.get("num_async_vector_env")
        server_nave = (
            server_group_size if server_nave_raw is None else server_nave_raw
        )
        if server_nave != trainer_nave:
            raise RuntimeError(
                f"num_async_vector_env mismatch: trainer config = {trainer_nave}, "
                f"server (boot-time) = {server_nave}. Restart the collector "
                f"server with --num-async-vector-env {trainer_nave}."
            )

        server_n_action_steps = info.get("n_action_steps")
        if server_n_action_steps != self.config.n_action_steps:
            raise RuntimeError(
                f"n_action_steps mismatch: trainer config = "
                f"{self.config.n_action_steps}, server (boot-time) = "
                f"{server_n_action_steps}. Restart the collector server with "
                f"--n-action-steps {self.config.n_action_steps}."
            )

        server_env_max_steps = info.get("env_max_steps", {})
        for env_name in self.config.env_names:
            expected = self._resolve_max_steps_for_env(env_name)
            actual = server_env_max_steps.get(env_name)
            if actual != expected:
                raise RuntimeError(
                    f"max_episode_steps mismatch for {env_name!r}: trainer "
                    f"config = {expected}, server (boot-time) = {actual}. "
                    f"Restart the collector server with --max-episode-steps "
                    f"matching the per-env values in this trainer config."
                )

    def _resolve_max_steps_for_env(self, env_name: str) -> int:
        """Look up max_episode_steps for one env_name from the trainer config.

        config.max_episode_steps can be an int (broadcast to all envs) or a
        list parallel to config.env_names.
        """
        if isinstance(self.config.max_episode_steps, list):
            idx = self.config.env_names.index(env_name)
            return self.config.max_episode_steps[idx]
        return self.config.max_episode_steps

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
            # Belt-and-suspenders: __post_init__ already enforces this. The
            # double-check makes the caller's contract explicit and protects
            # against direct setup() invocation that bypassed the dataclass
            # validation (e.g., a future refactor that mutates config fields
            # post-construction).
            raise ValueError(
                "_validate_collected_data_cache called without resume_from set "
                "(should have been rejected by GRPOConfig.__post_init__)."
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
        # the trainer's gradient-signal filter (advantage=0 for std<1e-4
        # groups → dropped by the dead-group filter in
        # _grpo_update_inner). Compare against `len(s)` rather than
        # config.group_size so partial groups (worker crashes losing some
        # rollouts) are evaluated by what's actually in the cache — which
        # is what compute_advantages will see. For full groups
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
        # (episode_buffer.py:367), so `== 0.0` would also work; `abs(x) > 1e-12`
        # is defense-in-depth against any future change that introduces float
        # noise in the per-group normalization path.
        n_total = len(chunks)
        chunks = [c for c in chunks if abs(c.advantage) > 1e-12]
        n_live = len(chunks)
        if n_live < n_total:
            print(
                f"  Skipping ref log-prob pass for {n_total - n_live}/"
                f"{n_total} dead-group chunks (advantage == 0)."
            )
        if not chunks:
            # Should have been caught by std_reward < 1e-8 in train(), but
            # guard against the edge case of a non-zero global std but every
            # group still dead (mathematically possible with a single
            # success-vs-failure split across groups).
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
                # uses original ε for the DiT input regardless of jitter_lambda
                # (Jitter-GRPO anchors the cached ref at the original ε so
                # both fixed and jitter branches share the same baseline),
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
        all_chunks = self.buffer._build_chunks()
        live_chunks = [c for c in all_chunks if abs(c.advantage) > 1e-12]
        n_total_chunks = len(all_chunks)
        n_live_chunks = len(live_chunks)
        if n_live_chunks < n_total_chunks:
            print(
                f"  Filtering {n_total_chunks - n_live_chunks}/"
                f"{n_total_chunks} chunks with zero advantage (dead groups). "
                f"Remaining live chunks: {n_live_chunks}."
            )
        if n_live_chunks == 0:
            return {}

        # Jitter-GRPO paired scheduling. When jitter_lambda > 0, every chunk
        # produces TWO entries per epoch: a "fixed" entry (DiT input noise =
        # original ε) and a "jitter" entry (DiT input noise = ε' = sqrt(1-λ²)·ε
        # + λ·ξ). The stratified minibatcher then yields 2× as many minibatches
        # → 2× optimizer steps per epoch. User halves update_epochs MANUALLY
        # to match the per-iter step count of vanilla GRPO. When 0 (default),
        # behavior is bit-identical to pre-jitter code (single "fixed" tag per
        # chunk; ξ-sampling block below is skipped).
        if self.config.jitter_lambda > 0.0:
            entries = (
                [(c, "fixed") for c in live_chunks]
                + [(c, "jitter") for c in live_chunks]
            )
        else:
            entries = [(c, "fixed") for c in live_chunks]

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
        grad_norms: list[float] = []
        ratio_maxes: list[float] = []
        ratio_mins: list[float] = []
        n_updates = 0
        n_skipped_nonfinite = 0  # minibatches dropped for NaN/Inf loss

        # Whether the base-model KL anchor is active this run. Cached locally
        # so the per-mb hot path skips the dict lookup. Drives the decision
        # to load base_log_probs and compute KL(base || current) below.
        compute_base = self.config.kl_coef_base_model > 0.0

        # Per-branch row-level accumulators (Jitter-GRPO). Aggregated metrics
        # above stay per-mb so the jitter_lambda=0 path produces bit-identical
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
        # near update_epochs even when real training signal is sparse.
        #
        # We use episode-level advantage sign (self.buffer.advantages[i] > 0)
        # rather than ep.success, for consistency with _iter_balanced_minibatches
        # which oversamples chunks with c.advantage > 0. With shaped rewards
        # (success_weight < 1.0), a failing episode with high max_progress can
        # have positive advantage — ep.success would undercount these.
        if self.config.dynamic_epoch_training:
            live_group_ids = {c.group_id for c in live_chunks}
            live_ep_indices = [
                i for i, ep in enumerate(self.buffer.episodes)
                if ep.group_id in live_group_ids
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
            m = min(successful_eps, total_eps_collected - successful_eps)
            E = self.config.update_epochs
            n = total_eps_collected
            actual_num_epochs = max(1, (4 * m * E + n) // (2 * n))
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

            if self.config.balanced_minibatch_training:
                batch_iter = self._iter_balanced_minibatches(entries, rng)
            else:
                batch_iter = self._iter_stratified_minibatches(entries, rng)

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
                # noise_for_input[k, jitter_row] = sqrt(1-λ²)·ε + λ·ξ_k. Fixed
                # rows keep noise_for_input[:, fixed_row] = ε (unchanged).
                # The original ε (ready_noise) is still passed as `noise=...`
                # below so velocity_target = a − ε is anchored at the original
                # noise — that asymmetry is what makes the loss in expectation
                # an FM-loss + Frobenius-norm Jacobian regularizer (the core
                # Jitter-GRPO trick).
                lam = self.config.jitter_lambda
                if (
                    lam > 0.0
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

                    # expand returns a view — clone() is REQUIRED before
                    # __setitem__ to allocate writable per-K rows; without
                    # it the assignment would alias across the K dimension.
                    noise_for_input = (
                        ready_noise.unsqueeze(0).expand(K, -1, -1, -1).clone()
                    )
                    sqrt_one_minus = (1.0 - lam * lam) ** 0.5
                    noise_for_input[:, jitter_mask_dev] = (
                        sqrt_one_minus
                        * ready_noise[jitter_mask_dev].unsqueeze(0)
                        + lam * xi[:, jitter_mask_dev]
                    )
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

                # --- Per-minibatch advantage renormalization (matches grpo_cont.py:413-417) ---
                # After the A_episode/num_chunks division in _build_chunks, per-chunk
                # advantages have small, heterogeneous magnitudes (varying with
                # episode length). Re-normalizing within the minibatch stabilizes
                # gradient scale across iterations and keeps the effective clip
                # threshold meaningful relative to the advantage magnitude.
                #
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
                # surface.
                pre_renorm_pos_adv_mask = ready_advantages > 0

                if ready_advantages.numel() > 1:
                    ready_advantages = (
                        (ready_advantages - ready_advantages.mean())
                        / (ready_advantages.std() + 1e-8)
                    )

                # --- Clipped surrogate loss ---
                surr1 = ready_advantages * ratio
                surr2 = ready_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_eps_low, 1 + self.config.clip_eps_high
                )
                clip_loss = -torch.min(surr1, surr2).mean()

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
                # term separately.
                inv_log_ratio = ref_log_probs - current_log_probs  # = -log_ratio
                kl_per_row_last_iter = inv_log_ratio.exp() - inv_log_ratio - 1.0
                kl_loss_last_iter = (
                    self.config.kl_coef_last_iter * kl_per_row_last_iter.mean()
                )

                if compute_base:
                    inv_log_ratio_base = base_log_probs - current_log_probs
                    kl_per_row_base_model = (
                        inv_log_ratio_base.exp() - inv_log_ratio_base - 1.0
                    )
                    kl_loss_base_model = (
                        self.config.kl_coef_base_model
                        * kl_per_row_base_model.mean()
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
                if not torch.isfinite(loss):
                    n_skipped_nonfinite += 1
                    continue

                # --- Backward pass ---
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping. clip_grad_norm_ returns the TOTAL norm
                # of the gradient vector BEFORE clipping — capture it for
                # the train/grad_norm_* diagnostics (independent of whether
                # clipping actually fired this minibatch).
                pre_clip_grad_norm = nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                # `clip_grad_norm_` with default error_if_nonfinite=False can
                # return inf/nan if backward introduced a non-finite gradient
                # that didn't show up in the forward `loss` (which we already
                # guard above). A single inf in `grad_norms` then poisons
                # np.mean/np.max into inf and breaks TB chart autoscale for
                # the rest of the run. Drop non-finite values silently —
                # n_skipped_nonfinite already tracks the upstream loss case.
                gnorm = float(pre_clip_grad_norm)
                if math.isfinite(gnorm):
                    grad_norms.append(gnorm)
                self.optimizer.step()

                # --- Track statistics ---
                with torch.no_grad():
                    # Fraction of rows the clamp actually moved. With asymmetric
                    # bounds this is the OR of the two one-sided clips rather than
                    # a single |ratio - 1| > eps threshold.
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
                    n_updates += 1

                    # --- Per-branch row-level accumulation (Jitter-GRPO) ---
                    # Only runs when jitter is enabled. Gating on lam>0 makes
                    # the jitter_lambda=0 path bit-identical at the metrics
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
                    if lam > 0.0:
                        over_clip = (
                            (ratio < 1 - self.config.clip_eps_low)
                            | (ratio > 1 + self.config.clip_eps_high)
                        ).float()
                        log_ratio_abs = log_ratio.abs()
                        fixed_mask = torch.tensor(
                            [m == "fixed" for m in ready_modes],
                            device=self.device, dtype=torch.bool,
                        )
                        jit_mask = ~fixed_mask
                        # Pre-renorm sign masks for the 4-way clipfrac split.
                        # See pre_renorm_pos_adv_mask capture above.
                        neg_adv_mask = ~pre_renorm_pos_adv_mask
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
                            n_rows_fixed_pos += n_fp
                        if n_fn > 0:
                            clipfrac_sum_fixed_neg += int(over_clip[fixed_neg_mask].sum().item())
                            n_rows_fixed_neg += n_fn
                        if n_jp > 0:
                            clipfrac_sum_jitter_pos += int(over_clip[jit_pos_mask].sum().item())
                            n_rows_jitter_pos += n_jp
                        if n_jn > 0:
                            clipfrac_sum_jitter_neg += int(over_clip[jit_neg_mask].sum().item())
                            n_rows_jitter_neg += n_jn

        # Model remains in eval mode (it never left)
        if n_updates == 0:
            early: dict = {"n_skipped_nonfinite": n_skipped_nonfinite} if n_skipped_nonfinite else {}
            if self.config.dynamic_epoch_training:
                early["actual_epochs"] = actual_num_epochs
                if success_frac is not None:
                    early["success_fraction"] = success_frac
            return early

        if n_skipped_nonfinite > 0:
            print(
                f"  WARNING: skipped {n_skipped_nonfinite} minibatch(es) for "
                f"non-finite loss (NaN/Inf) — likely bf16 ratio overflow"
            )

        result = {
            "loss": total_loss / n_updates,
            "clip_loss": total_clip_loss / n_updates,
            "kl_loss_last_iter": total_kl_last_iter / n_updates,
            "clipfrac": np.mean(clipfracs) if clipfracs else 0,
            "mean_ratio": total_ratio / n_updates,
            "mean_log_ratio_abs": total_log_ratio_abs / n_updates,
            "n_updates": n_updates,
            "n_skipped_nonfinite": n_skipped_nonfinite,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
            "ratio_max": float(np.max(ratio_maxes)) if ratio_maxes else 1.0,
            "ratio_min": float(np.min(ratio_mins)) if ratio_mins else 1.0,
            "actual_epochs": actual_num_epochs,
        }
        # Only emit kl_loss_base_model when the anchor was active this iter.
        # _log_metrics gates on key presence, so vanilla runs see no
        # train/kl_loss_base_model curve at all.
        if compute_base:
            result["kl_loss_base_model"] = total_kl_base_model / n_updates
        if success_frac is not None:
            result["success_fraction"] = success_frac
        # Per-branch metrics (Jitter-GRPO). Only emitted when jitter_lambda > 0
        # — at jitter_lambda == 0 the per-branch accumulators stay at their
        # zero defaults (the per-mb update block is gated on `lam > 0`), so
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
        if n_rows_fixed_neg > 0:
            result["clipfrac_fixed_neg"] = clipfrac_sum_fixed_neg / n_rows_fixed_neg
        if n_rows_jitter_pos > 0:
            result["clipfrac_jitter_pos"] = clipfrac_sum_jitter_pos / n_rows_jitter_pos
        if n_rows_jitter_neg > 0:
            result["clipfrac_jitter_neg"] = clipfrac_sum_jitter_neg / n_rows_jitter_neg
        return result

    def _iter_stratified_minibatches(
        self,
        entries: list[tuple[ActionChunk, str]],
        rng: np.random.Generator,
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
        mb_size = self.config.mini_batch_size
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

    def _iter_balanced_minibatches(
        self,
        entries: list[tuple[ActionChunk, str]],
        rng: np.random.Generator,
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

        Yields:
            Lists of (ActionChunk, mode) tuples, length <= mini_batch_size.
        """
        if not entries:
            return

        pos_ratio = self.config.balanced_minibatch_positive_adv_ratio

        # Split entries by advantage sign. The per-chunk advantage inherits its
        # sign directly from the episode-level group-relative normalization;
        # live_chunks already filtered out zero-advantage (dead group) entries.
        pos_indices = [i for i, (c, _) in enumerate(entries) if c.advantage > 0]
        neg_indices = [i for i, (c, _) in enumerate(entries) if c.advantage <= 0]

        # Fall back to stratified when one sign class is absent — we can't
        # form a balanced batch without both positive and negative entries.
        if not pos_indices or not neg_indices:
            yield from self._iter_stratified_minibatches(entries, rng)
            return

        natural_pos_frac = len(pos_indices) / len(entries)

        mb_size = self.config.mini_batch_size
        n_pos_per_batch = max(1, round(pos_ratio * mb_size))
        n_neg_per_batch = mb_size - n_pos_per_batch

        # Guard: if rounding left no room for one sign class (e.g. pos_ratio=0.9375
        # with mb_size=8 causes round(7.5)=8 → n_neg=0), fall back.
        if n_neg_per_batch <= 0:
            yield from self._iter_stratified_minibatches(entries, rng)
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
            # mean_progress is only meaningful when dense progress actually fed into
            # the shaped reward. With success_weight=1.0 (default) the collector
            # skips compute_dense_progress entirely, so max_progress is a constant 0
            # and logging it here would just produce a flat zero curve.
            if self.config.success_weight < 1.0:
                self.writer.add_scalar("episode/mean_progress", stats.get("mean_progress", 0), iteration)
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
            self.writer.add_scalar(
                "train/n_skipped_nonfinite",
                update_stats.get("n_skipped_nonfinite", 0),
                iteration,
            )

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
            # GRPO runs (jitter_lambda=0) see no `_jitter` curves at all,
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
                for sign in ("pos", "neg"):
                    key = f"clipfrac_{branch}_{sign}"
                    if key in update_stats:
                        self.writer.add_scalar(
                            f"train/{key}", update_stats[key], iteration
                        )

        if lr is not None:
            self.writer.add_scalar("train/learning_rate", lr, iteration)

        if iter_time is not None:
            self.writer.add_scalar("time/iteration_seconds", iter_time, iteration)

        # Phase-time breakdown (collect / advantage / update). The trainer
        # already times each phase for its console summary; surface them
        # here so TB can answer "which phase regressed?" without parsing logs.
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
                if update_stats is not None:
                    # Counters always; loss/ratio/grad only when n_updates>0
                    # (matching the TB-side gating).
                    log_dict["train/n_updates"] = update_stats.get("n_updates", 0)
                    log_dict["train/n_skipped_nonfinite"] = (
                        update_stats.get("n_skipped_nonfinite", 0)
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
                            )
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
    # shutdown() runs even when setup() raises mid-way — important because
    # __init__ may have already opened a ZMQ collector_client (and started
    # background threads inside zmq.Context), and setup() itself has many
    # raise paths (cache validator, optimizer state validation, LoRA load
    # mismatch). Without the wrap, those raises bypass shutdown() entirely
    # and leak the collector socket / context.
    trainer = GRPOTrainer(config)
    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
