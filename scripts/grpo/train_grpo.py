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
import shutil
import threading
import time
import subprocess
from pathlib import Path
from typing import Optional

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
)
from fm_log_prob import compute_fm_log_prob, _sample_jittered_timesteps
from episode_buffer import EpisodeBuffer, ActionChunk


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
            self._collector_client = CollectorClient(
                host=self.config.collector_server_host,
                port=self.config.collector_server_port,
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
                f"n_action_steps={info.get('n_action_steps')})"
            )

    def setup(self):
        """Load model, apply LoRA, setup optimizer, start server.

        This is separate from __init__ so that config can be modified before setup.
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
        start_iteration = 1
        if self.config.resume_from:
            resume_path = Path(self.config.resume_from)
            print(f"\n  Resuming from: {resume_path}")
            load_lora_checkpoint(self.model, resume_path)
            # Extract iteration number from directory name (e.g., "iter_0050" → 50)
            dir_name = resume_path.name
            if dir_name.startswith("iter_"):
                start_iteration = int(dir_name.split("_")[1]) + 1
                print(f"  Continuing from iteration {start_iteration}")
        self._start_iteration = start_iteration

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
        print(f"  Episodes per iteration: {total_eps} ({self.config.num_groups} groups × {self.config.group_size})")
        print(f"  Update epochs: {self.config.update_epochs}")
        print(f"  Mini-batch size: {self.config.mini_batch_size}")
        print(f"  Estimated time: ~{self.config.num_iterations * 5 / 60:.1f} hours")

        for iteration in range(self._start_iteration, self.config.num_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

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

            # ═══ Phase 1: Collect episodes ═══
            phase1_start = time.time()
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
                self._log_metrics(iteration, stats, skip_reason="no_signal")
                # Save anyway on save_interval boundaries: the model state is
                # unchanged from the previous iter (no optimizer.step ran), but
                # a resume point named iter_NNNN should always exist when N is
                # a multiple of save_interval, regardless of whether the
                # gradient update happened to fire.
                if iteration % self.config.save_interval == 0:
                    self._save_checkpoint(iteration)
                continue

            # ═══ Phase 2b: Pre-compute reference log-probs ═══
            self._compute_ref_log_probs()

            # ═══ Phase 3: GRPO Policy Update ═══
            phase3_start = time.time()
            update_stats = self._grpo_update()
            phase3_time = time.time() - phase3_start

            # ═══ Phase 4: Logging and checkpointing ═══
            iter_time = time.time() - iter_start
            self._log_metrics(iteration, stats, update_stats, lr, iter_time)

            print(
                f"  Time: collect={phase1_time:.0f}s, "
                f"advantage={phase2_time:.1f}s, "
                f"update={phase3_time:.0f}s, "
                f"total={iter_time:.0f}s"
            )

            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self._save_checkpoint(iteration)

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

        # Final save
        self._save_checkpoint(self.config.num_iterations)

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
        print(f"  Collecting {self.config.num_groups} groups × {self.config.group_size} = {total_episodes} episodes...")

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
        expected_total = self.config.group_size * self.config.num_groups
        if n_loaded < expected_total:
            pct = 100 * n_loaded / expected_total if expected_total > 0 else 0
            print(
                f"  WARNING: Only {n_loaded}/{expected_total} episodes "
                f"({pct:.0f}%) loaded — some workers may have failed silently. "
                f"Failure counter NOT incremented."
            )

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
            collector_script,
            "--env-name", env_name,
            "--group-size", str(self.config.group_size),
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
        ]

        # Stream collector output line-by-line so the user sees progress
        # instead of waiting for the whole subprocess to finish. Mirror the
        # collector's stdout/stderr to the trainer log with a "[collector]"
        # prefix. A background Timer enforces the wall clock even if the
        # subprocess hangs on stdout with no output (otherwise the blocking
        # read could wait forever).
        timeout_s = 2100  # 35 min
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
            )
        except FatalCollectorError as e:
            raise RuntimeError(
                f"Collector server reports fatal config error: {e}. This "
                f"won't fix itself on retry — restart the collector server "
                f"with --env-names / --max-episode-steps / --group-size / "
                f"--n-action-steps matching this trainer's config "
                f"(env_names={self.config.env_names}, "
                f"group_size={self.config.group_size}, "
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
        with self._model_lock, torch.no_grad():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                result = self._prepare_batch(batch)
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
                n_computed += len(valid_batch)

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

        total_loss = 0.0
        total_clip_loss = 0.0
        total_kl = 0.0
        total_ratio = 0.0
        total_log_ratio_abs = 0.0
        clipfracs = []
        n_updates = 0
        n_skipped_nonfinite = 0  # minibatches dropped for NaN/Inf loss

        for epoch in range(self.config.update_epochs):
            # Iterate over mini-batches (shuffled each epoch)
            for batch in self.buffer.iter_minibatches(
                batch_size=self.config.mini_batch_size,
                shuffle=True,
                seed=self.config.seed + self.iteration * 100 + epoch,
            ):
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

                # --- Compute importance ratio ---
                # Use pre-computed ref_log_probs (from _compute_ref_log_probs)
                # and stored tau_samples for consistency.
                # Build ready_indices directly instead of calling list.index(c):
                # ActionChunk is a @dataclass(eq=True) with ndarray fields, and
                # comparing numpy arrays raises "truth value is ambiguous", so
                # relying on .index() is fragile even if CPython's identity
                # short-circuit currently masks it.
                ready_indices = [
                    i for i, c in enumerate(valid_batch)
                    if c.ref_log_prob is not None and c.tau_samples is not None
                ]
                if not ready_indices:
                    continue
                ready_batch = [valid_batch[i] for i in ready_indices]

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

                # Reconstruct timesteps from stored per-chunk tau_samples
                tau_np = np.stack([c.tau_samples for c in ready_batch], axis=1)  # [K, B]
                timesteps = torch.from_numpy(tau_np).to(
                    device=self.device, dtype=torch.bfloat16
                )

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
                )

                log_ratio = current_log_probs - ref_log_probs
                ratio = log_ratio.exp()

                # --- Per-minibatch advantage renormalization (matches grpo_cont.py:413-417) ---
                # After the A_episode/num_chunks division in _build_chunks, per-chunk
                # advantages have small, heterogeneous magnitudes (varying with
                # episode length). Re-normalizing within the minibatch stabilizes
                # gradient scale across iterations and keeps the effective clip
                # threshold meaningful relative to the advantage magnitude.
                if ready_advantages.numel() > 1:
                    ready_advantages = (
                        (ready_advantages - ready_advantages.mean())
                        / (ready_advantages.std() + 1e-8)
                    )

                # --- Clipped surrogate loss ---
                surr1 = ready_advantages * ratio
                surr2 = ready_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps
                )
                clip_loss = -torch.min(surr1, surr2).mean()

                # --- KL divergence penalty (Schulman k3 estimator) ---
                # KL(ref || current) ≈ E[exp(ref - current) - (ref - current) - 1]
                # Identity: e^x - x - 1 ≥ 0 for all x, with equality iff x=0.
                # Properties vs the naive (ref - current).mean():
                #   - Non-negative POINTWISE, not just in expectation.
                #   - Minimum at current ≡ ref → gradient pulls policies together
                #     symmetrically (the naive estimator's gradient was one-sided
                #     and could *reward* current >> ref).
                #   - Same expected value (still estimates KL(ref||current)).
                #   - Lower variance.
                # See Schulman 2020 "Approximating KL Divergence" for the derivation.
                inv_log_ratio = ref_log_probs - current_log_probs  # = -log_ratio
                kl_loss = self.config.kl_coef * (
                    inv_log_ratio.exp() - inv_log_ratio - 1.0
                ).mean()

                # --- Total loss ---
                loss = clip_loss + kl_loss

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

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # --- Track statistics ---
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()
                    clipfracs.append(clipfrac)
                    total_loss += loss.item()
                    total_clip_loss += clip_loss.item()
                    total_kl += kl_loss.item()
                    total_ratio += ratio.mean().item()
                    # log_ratio magnitude is the primary diagnostic for DPPO-style
                    # FM log-prob surrogates: large values mean the MSE-based
                    # log-prob is noisy enough that most updates will clip, which
                    # caps the effective gradient signal.
                    total_log_ratio_abs += log_ratio.abs().mean().item()
                    n_updates += 1

        # Model remains in eval mode (it never left)
        if n_updates == 0:
            return {"n_skipped_nonfinite": n_skipped_nonfinite} if n_skipped_nonfinite else {}

        if n_skipped_nonfinite > 0:
            print(
                f"  WARNING: skipped {n_skipped_nonfinite} minibatch(es) for "
                f"non-finite loss (NaN/Inf) — likely bf16 ratio overflow"
            )

        return {
            "loss": total_loss / n_updates,
            "clip_loss": total_clip_loss / n_updates,
            "kl_loss": total_kl / n_updates,
            "clipfrac": np.mean(clipfracs) if clipfracs else 0,
            "mean_ratio": total_ratio / n_updates,
            "mean_log_ratio_abs": total_log_ratio_abs / n_updates,
            "n_updates": n_updates,
            "n_skipped_nonfinite": n_skipped_nonfinite,
        }

    def _prepare_batch(self, batch: list[ActionChunk]) -> Optional[tuple[dict, list[ActionChunk]]]:
        """Convert a list of ActionChunks into GPU tensors for training.

        This handles:
        - Using raw normalized actions (50x128) for FM log-prob computation
        - Re-encoding observations through the backbone
        - Creating embodiment ID tensors

        The raw_action field is REQUIRED — it's the action in the model's internal
        space (before decode_action slices/denormalizes). Without it, the FM loss
        surrogate would be computed on mismatched dimensions.

        Args:
            batch: List of ActionChunk objects from the episode buffer.

        Returns:
            Tuple of (tensor_dict, valid_batch_list), or None if batch is invalid.
        """
        if not batch:
            return None

        # Filter to chunks that have raw_actions (required for FM log-prob)
        valid_batch = [c for c in batch if c.raw_action is not None]
        if not valid_batch:
            return None

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

    def _log_metrics(self, iteration, stats, update_stats=None, lr=None, iter_time=None, skip_reason=None):
        """Log training metrics to TensorBoard and wandb."""
        if self.writer is None:
            return

        # Episode stats
        self.writer.add_scalar("episode/success_rate", stats.get("success_rate", 0), iteration)
        # mean_progress is only meaningful when dense progress actually fed into
        # the shaped reward. With success_weight=1.0 (default) the collector
        # skips compute_dense_progress entirely, so max_progress is a constant 0
        # and logging it here would just produce a flat zero curve.
        if self.config.success_weight < 1.0:
            self.writer.add_scalar("episode/mean_progress", stats.get("mean_progress", 0), iteration)
        self.writer.add_scalar("episode/mean_reward", stats.get("mean_reward", 0), iteration)
        self.writer.add_scalar("episode/std_reward", stats.get("std_reward", 0), iteration)

        # Update stats
        if update_stats:
            self.writer.add_scalar("train/loss", update_stats.get("loss", 0), iteration)
            self.writer.add_scalar("train/clip_loss", update_stats.get("clip_loss", 0), iteration)
            self.writer.add_scalar("train/kl_loss", update_stats.get("kl_loss", 0), iteration)
            self.writer.add_scalar("train/clipfrac", update_stats.get("clipfrac", 0), iteration)
            self.writer.add_scalar("train/mean_ratio", update_stats.get("mean_ratio", 1), iteration)
            self.writer.add_scalar(
                "train/mean_log_ratio_abs",
                update_stats.get("mean_log_ratio_abs", 0),
                iteration,
            )
            # Track NaN/Inf-skipped minibatches; sustained nonzero values
            # indicate ratio overflow or numerical instability worth tuning
            # (lower lr or alpha, or cast MSE to fp32).
            self.writer.add_scalar(
                "train/n_skipped_nonfinite",
                update_stats.get("n_skipped_nonfinite", 0),
                iteration,
            )

        if lr is not None:
            self.writer.add_scalar("train/learning_rate", lr, iteration)

        if iter_time is not None:
            self.writer.add_scalar("time/iteration_seconds", iter_time, iteration)

        # Wandb logging
        if self.config.use_wandb:
            try:
                import wandb
                log_dict = {"iteration": iteration, **stats}
                if update_stats:
                    log_dict.update({f"train/{k}": v for k, v in update_stats.items()})
                if lr:
                    log_dict["train/lr"] = lr
                wandb.log(log_dict)
            except Exception:
                pass

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

    # Create trainer and run
    trainer = GRPOTrainer(config)
    trainer.setup()
    try:
        trainer.train()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
