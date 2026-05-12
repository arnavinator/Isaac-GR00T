"""Main GRPO training loop for GR00T N1.6 DiT.

This is the orchestrator that ties everything together:
1. Loads model + applies LoRA
2. Creates frozen reference model
3. Iterates: collect episodes → compute advantages → policy update

Structure mirrors grpo_cont.py's outer loop (lines 242-457):
    for update in range(1, num_updates+1):
        # Collect data (lines 253-312)
        # Compute advantages (lines 325-364)
        # GRPO policy update (lines 379-439)
        # Log metrics (lines 446-455)

Key differences from grpo_cont.py:
- Episode collection is via subprocess (robocasa venv) + ZMQ server
- Log-prob uses FM surrogate instead of Gaussian distribution
- Advantages are episodic (group-relative on shaped rewards)
- Reference model used instead of stored old log-probs

Usage:
    uv run python scripts/grpo/train_grpo.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --num-iterations 200 \\
        --group-size 5 --num-groups 12

Hardware: Fits on A10G (24GB) with batch_size=4 and shared backbone.
"""

import sys
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
    create_reference_dit,
    update_reference_dit,
    save_lora_checkpoint,
    load_lora_checkpoint,
    print_trainable_params,
)
from fm_log_prob import compute_fm_log_prob_pair
from episode_buffer import EpisodeBuffer, ActionChunk


class GRPOTrainer:
    """GRPO training loop for GR00T N1.6 DiT with LoRA.

    This class manages the full training pipeline:
    - Model setup (LoRA injection, reference model creation)
    - Episode collection (launches collector subprocess)
    - Advantage computation (group-relative normalization)
    - Policy gradient update (clipped surrogate + KL penalty)
    - Evaluation and checkpointing

    The training logic directly mirrors grpo_cont.py's structure:
    - Outer loop: collect → advantage → update
    - Inner loop (policy update): multiple epochs × minibatches
    - Loss: -min(ratio*A, clip(ratio)*A) + kl_coef * KL
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
        self.ref_action_head = None
        self.optimizer = None
        self.iteration = 0

        # Episode buffer for current iteration's data
        self.buffer = EpisodeBuffer()

        # Logging
        self.writer = None  # TensorBoard/wandb writer

    def setup(self):
        """Load model, apply LoRA, create reference, setup optimizer.

        This is separate from __init__ so that config can be modified before setup.
        """
        import gr00t.model  # noqa: F401 — registers model classes
        from transformers import AutoModel, AutoProcessor

        print("=" * 60)
        print("GRPO Training Setup")
        print("=" * 60)

        # --- Step 1: Load pretrained model ---
        print(f"\n[1/5] Loading model from {self.config.model_path}...")
        self.model = AutoModel.from_pretrained(self.config.model_path)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.eval()  # Start in eval mode (we manually control train/eval per component)

        # Load processor for action encoding/decoding
        self.processor = AutoProcessor.from_pretrained(self.config.model_path)
        self.processor.eval()

        # --- Step 2: Apply LoRA to DiT ---
        print(f"\n[2/5] Applying LoRA (rank={self.config.lora_rank})...")
        self.model = apply_lora_to_dit(
            self.model,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )
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

        # --- Step 3: Create frozen reference model ---
        # The reference shares the Eagle backbone (frozen, identical) but has
        # its own DiT copy for computing reference log-probs in the importance ratio.
        # If resuming, the reference starts at the same trained state as the current model.
        print("\n[3/5] Creating frozen reference action head...")
        self.ref_action_head = create_reference_dit(self.model)
        self.ref_action_head.to(device=self.device, dtype=torch.bfloat16)

        # --- Step 4: Setup optimizer (only LoRA params) ---
        print("\n[4/5] Setting up optimizer...")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
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
                self.optimizer.load_state_dict(
                    torch.load(opt_path, map_location=self.device)
                )
                print(f"  Optimizer state restored from {opt_path}")
            else:
                print(f"  WARNING: No optimizer.pt found at {opt_path}, starting fresh optimizer")

        print(f"  AdamW: lr={self.config.learning_rate}, wd={self.config.weight_decay}")
        print(f"  Trainable params in optimizer: {sum(p.numel() for p in trainable_params):,}")

        # --- Step 5: Setup logging ---
        print("\n[5/5] Setting up logging...")
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

        print("\n" + "=" * 60)
        print("Setup complete. Ready to train.")
        print("=" * 60)

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
            # With 7 tasks and 200 iterations, each task gets ~28 full training updates.
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
            self.buffer.compute_advantages(success_weight=self.config.success_weight)
            stats = self.buffer.stats()
            phase2_time = time.time() - phase2_start

            # Skip update if no gradient signal (all same outcome)
            if stats.get("std_reward", 0) < 1e-8:
                print(f"  Skipping update: all episodes have same reward (no gradient signal)")
                self._log_metrics(iteration, stats, skip_reason="no_signal")
                continue

            # ═══ Phase 3: GRPO Policy Update ═══
            phase3_start = time.time()
            update_stats = self._grpo_update()
            phase3_time = time.time() - phase3_start

            # ═══ Phase 4: Reference model update ═══
            if iteration % self.config.ref_update_interval == 0:
                print(f"  Updating reference model (every {self.config.ref_update_interval} iters)")
                update_reference_dit(self.ref_action_head, self.model)

            # ═══ Phase 5: Logging and checkpointing ═══
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
        """Launch episode collector subprocess and load results.

        The collector runs in the robocasa venv as a separate process,
        communicating with our model via the ZMQ server.

        This is analogous to grpo_cont.py's collection loop (lines 253-312)
        but uses a subprocess instead of inline env stepping.
        """
        self.buffer.clear()

        # Output directory for this iteration's episodes
        episode_dir = Path(self.config.episode_dir) / f"iter_{self.iteration:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Start the ZMQ server in a background thread
        server_handle = self._start_server_thread()

        try:
            # Launch collector subprocess in robocasa venv
            robocasa_python = str(
                Path(__file__).parent.parent.parent
                / "gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
            )

            collector_script = str(Path(__file__).parent / "collect_episodes.py")

            # Resolve per-task fast_forward_steps (same pattern as max_episode_steps)
            if isinstance(self.config.fast_forward_steps, list):
                ff_steps = self.config.fast_forward_steps[task_idx]
            else:
                ff_steps = self.config.fast_forward_steps

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
                "--seed", str(self.config.seed + self.iteration * 1000),
            ]

            total_episodes = self.config.group_size * self.config.num_groups
            print(f"  Collecting {self.config.num_groups} groups × {self.config.group_size} = {total_episodes} episodes...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                print(f"  WARNING: Collector failed with code {result.returncode}")
                print(f"  stderr: {result.stderr[:500]}")
                return

        finally:
            self._stop_server_thread(server_handle)

        # Load collected episodes into buffer
        n_loaded = self.buffer.load_episodes(episode_dir)
        print(f"  Loaded {n_loaded} episodes ({self.buffer.num_chunks} chunks)")

    def _grpo_update(self) -> dict:
        """Run GRPO clipped surrogate policy gradient update on collected episodes.

        This directly mirrors grpo_cont.py lines 379-439:
            for epoch in range(update_epochs):
                for start in range(0, grouped_batch_size, minibatch_size):
                    # Compute new log-probs
                    # Compute ratio and clipped loss
                    # Backward + step

        Key adaptations for flow-matching:
        - log_prob computed via FM surrogate (fm_log_prob.py) instead of Gaussian
        - Reference model used instead of stored old log-probs
        - Backbone features cached (expensive, only run once per unique observation)

        Returns:
            Dict of update statistics (loss, clipfrac, kl, etc.)
        """
        # Put DiT in training mode (LoRA layers need dropout)
        self.model.action_head.model.train()

        total_loss = 0.0
        total_clip_loss = 0.0
        total_kl = 0.0
        clipfracs = []
        n_updates = 0

        for epoch in range(self.config.update_epochs):
            # Iterate over mini-batches (shuffled each epoch)
            for batch in self.buffer.iter_minibatches(
                batch_size=self.config.mini_batch_size,
                shuffle=True,
                seed=self.config.seed + self.iteration * 100 + epoch,
            ):
                # --- Prepare batch tensors ---
                batch_data = self._prepare_batch(batch)
                if batch_data is None:
                    continue

                actions = batch_data["actions"]           # [B, horizon, dim]
                action_masks = batch_data["action_masks"] # [B, horizon, dim]
                advantages = batch_data["advantages"]     # [B]
                backbone_output = batch_data["backbone_output"]
                state_features = batch_data["state_features"]
                embodiment_id = batch_data["embodiment_id"]

                # --- Compute importance ratio ---
                # This is the FM-surrogate equivalent of grpo_cont.py line 409:
                #   log_ratio = logprob_new - logprob_old
                current_log_probs, ref_log_probs = compute_fm_log_prob_pair(
                    current_action_head=self.model.action_head,
                    ref_action_head=self.ref_action_head,
                    backbone_output=backbone_output,
                    state_features=state_features,
                    embodiment_id=embodiment_id,
                    actions=actions,
                    action_mask=action_masks,
                    n_samples=self.config.n_fm_samples,
                )

                log_ratio = current_log_probs - ref_log_probs
                ratio = log_ratio.exp()

                # --- Clipped surrogate loss ---
                # Mirrors grpo_cont.py lines 421-424:
                #   min1 = advs * ratio
                #   min2 = advs * torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
                #   loss_clip = torch.min(min1, min2).mean()
                surr1 = advantages * ratio
                surr2 = advantages * torch.clamp(
                    ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps
                )
                clip_loss = -torch.min(surr1, surr2).mean()

                # --- KL divergence penalty ---
                # Mirrors grpo_cont.py line 430:
                #   loss_kl = (logprob_old - logprob_new).mean()
                kl_loss = self.config.kl_coef * (ref_log_probs - current_log_probs).mean()

                # --- Total loss ---
                # Mirrors grpo_cont.py line 434:
                #   loss = -loss_clip + kl_coef * loss_kl
                loss = clip_loss + kl_loss

                # --- Backward pass ---
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (mirrors grpo_cont.py line 437)
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
                    n_updates += 1

        # Back to eval mode
        self.model.action_head.model.eval()

        if n_updates == 0:
            return {}

        return {
            "loss": total_loss / n_updates,
            "clip_loss": total_clip_loss / n_updates,
            "kl_loss": total_kl / n_updates,
            "clipfrac": np.mean(clipfracs) if clipfracs else 0,
            "n_updates": n_updates,
            "mean_ratio": ratio.mean().item() if 'ratio' in dir() else 0,
        }

    def _prepare_batch(self, batch: list[ActionChunk]) -> Optional[dict]:
        """Convert a list of ActionChunks into GPU tensors for training.

        This handles:
        - Using raw normalized actions (50×128) for FM log-prob computation
        - Re-encoding observations through the backbone
        - Creating embodiment ID tensors

        The raw_action field is REQUIRED — it's the action in the model's internal
        space (before decode_action slices/denormalizes). Without it, the FM loss
        surrogate would be computed on mismatched dimensions.

        Args:
            batch: List of ActionChunk objects from the episode buffer.

        Returns:
            Dict of tensors ready for the GRPO update, or None if batch is invalid.
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

        # --- Advantages ---
        advantages = torch.tensor(
            [chunk.advantage for chunk in valid_batch],
            device=self.device, dtype=torch.float32,
        )  # [B]

        # --- Encode observations through backbone ---
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
            "advantages": advantages,
            "backbone_output": backbone_output,
            "state_features": state_features,
            "embodiment_id": embodiment_id,
        }

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
        try:
            from gr00t.data.types import VLAStepData, MessageType
            from gr00t.data.embodiment_tags import EmbodimentTag

            embodiment_tag = EmbodimentTag(self.config.embodiment_tag)

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

            # Note: model.prepare_input() handles device/dtype conversion internally
            # (it uses tree.map_structure to move all tensors to model device + dtype)

            # Step 4: model.prepare_input() splits into backbone and action head inputs
            # The collator returns {"inputs": batch_dict}, so **collated unpacks to
            # prepare_input(inputs=batch_dict) matching the method signature
            backbone_inputs, action_inputs = self.model.prepare_input(**collated)

            # Step 5: Run backbone (frozen)
            backbone_output = self.model.backbone(backbone_inputs)

            # Step 6: Encode features (applies vlln + state encoder)
            features = self.model.action_head._encode_features(
                backbone_output, action_inputs
            )

            # Extract what we need
            # features has: backbone_features (processed vl_embeds), state_features
            # backbone_output has: image_mask, backbone_attention_mask (for DiT cross-attn)
            embodiment_id = action_inputs.embodiment_id

            # Build the backbone_output dict that fm_log_prob expects
            # It needs: backbone_features (processed), image_mask, backbone_attention_mask
            fm_backbone_output = {
                "backbone_features": features.backbone_features,
                "image_mask": getattr(backbone_output, "image_mask", None),
                "backbone_attention_mask": getattr(backbone_output, "backbone_attention_mask", None),
            }

            return fm_backbone_output, features.state_features, embodiment_id

        except Exception as e:
            print(f"  WARNING: Failed to encode observations: {e}")
            import traceback
            traceback.print_exc()
            return None

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
                self.embodiment_tag = EmbodimentTag(trainer_ref.config.embodiment_tag)
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
                """Split batched observation into list of single observations."""
                batch_size = 1
                for mod_val in observation.values():
                    if isinstance(mod_val, dict):
                        for v in mod_val.values():
                            if hasattr(v, "shape") and len(v.shape) > 0:
                                batch_size = v.shape[0]
                                break
                        break
                    elif hasattr(mod_val, "shape") and len(mod_val.shape) > 0:
                        batch_size = mod_val.shape[0]
                        break

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
                return self.processor.get_modality_configs()

        # Create policy → sim wrapper → GRPO wrapper
        # strict=False avoids observation validation during collection
        # (the collector may send slightly different formats)
        in_place_policy = _InPlacePolicy()
        sim_wrapper = Gr00tSimPolicyWrapper(in_place_policy, strict=False)

        grpo_wrapper = GRPOPolicyWrapper(
            policy=sim_wrapper,
            initial_seed=self.iteration * 10000,
            device=str(self.device),
        )

        server = PolicyServer(
            policy=grpo_wrapper,
            host=self.config.server_host,
            port=self.config.server_port,
        )

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        time.sleep(1)  # Give server time to bind
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
        self.writer.add_scalar("episode/mean_progress", stats.get("mean_progress", 0), iteration)
        self.writer.add_scalar("episode/mean_reward", stats.get("mean_reward", 0), iteration)
        self.writer.add_scalar("episode/std_reward", stats.get("std_reward", 0), iteration)

        # Update stats
        if update_stats:
            self.writer.add_scalar("train/loss", update_stats.get("loss", 0), iteration)
            self.writer.add_scalar("train/clip_loss", update_stats.get("clip_loss", 0), iteration)
            self.writer.add_scalar("train/kl_loss", update_stats.get("kl_loss", 0), iteration)
            self.writer.add_scalar("train/clipfrac", update_stats.get("clipfrac", 0), iteration)

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

        # Also save optimizer state for resuming
        torch.save(
            self.optimizer.state_dict(),
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
    trainer.train()


if __name__ == "__main__":
    main()
