"""GRPO training configuration for GR00T N1.6 DiT.

This dataclass mirrors the structure of grpo_cont.py's `init_args()` but adapted for:
- Flow-matching diffusion (FM log-prob surrogate instead of Gaussian policy)
- Server-client episode collection (instead of vectorized envs)
- Episodic sparse+shaped rewards (instead of per-step dense rewards)
- LoRA finetuning (instead of full parameter updates)

Usage:
    config = GRPOConfig()                         # defaults
    config = GRPOConfig(lora_rank=32, kl_coef=0.005)  # override
    # Or from CLI via tyro:
    # config = tyro.cli(GRPOConfig)
"""

from dataclasses import dataclass, field
from typing import Optional

from lora_dit import DEFAULT_LORA_TARGET_MODULES


@dataclass
class GRPOConfig:
    """Complete configuration for GRPO + LoRA finetuning of GR00T DiT.

    Organized into logical sections matching the training pipeline stages:
    1. Model & LoRA — what to train
    2. Collection — how to gather episodes
    3. Reward shaping — how to score episodes
    4. GRPO algorithm — how to compute advantages and policy gradient
    5. Optimizer — how to update parameters
    6. Training loop — orchestration
    7. Logging — what to track
    """

    # ─── Model & LoRA ────────────────────────────────────────────────────────

    # Path to pretrained model (HuggingFace hub ID or local path)
    model_path: str = "nvidia/GR00T-N1.6-3B"

    # Embodiment tag for the target robot (determines action dims, cameras, etc.)
    embodiment_tag: str = "ROBOCASA_PANDA_OMRON"

    # LoRA rank — controls trainable param count (~20M at r=16, ~2% of DiT)
    # Higher rank = more expressive but more memory and risk of overfitting
    lora_rank: int = 16

    # LoRA alpha — scaling factor. Standard practice: alpha = 2 * rank
    # Effective LoRA scale = alpha / rank = 2.0
    lora_alpha: int = 32

    # LoRA dropout — regularization inside the LoRA adapter layers.
    # Default 0.0 because the training loop keeps the DiT in eval mode for both
    # the reference and current log-prob passes (see train_grpo._grpo_update),
    # so any dropout you configure here is a no-op in practice. If you want
    # dropout to actually fire, you must ALSO switch the DiT to .train() mode
    # in BOTH _compute_ref_log_probs and _grpo_update — otherwise importance
    # ratios are unaffected and the value here doesn't matter.
    lora_dropout: float = 0.0

    # Which layers in the DiT to apply LoRA to (must be nn.Linear, NOT CategorySpecificLinear)
    # These are the module name patterns within model.action_head.model (AlternateVLDiT).
    # The default list is sourced from lora_dit.DEFAULT_LORA_TARGET_MODULES to keep
    # grpo_config, lora_dit, and grpo_server in sync.
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )

    # Device for model training (typically "cuda" or "cuda:0")
    device: str = "cuda"

    # ─── Episode Collection ──────────────────────────────────────────────────

    # Number of rollouts per group (G = "answers per question")
    # Each group resets G parallel environments with the SAME seed (identical initial state).
    # Different rollouts diverge due to policy noise (denoising randomness).
    # Advantages are computed by comparing outcomes WITHIN a group.
    # Same as grpo_cont.py's args.num_envs = 5
    # Also determines the number of parallel environments (one env per rollout).
    group_size: int = 4

    # Number of groups per iteration ("questions per iteration")
    # Each group gets a unique seed → unique initial kitchen/object configuration.
    # More groups = more diverse gradient signal per update.
    # Same as grpo_cont.py's args.num_groups = 5
    # With dynamic group collection (see min_successful_groups), this is the
    # MINIMUM number of groups; the collector may collect more (up to max_groups)
    # if the success criterion isn't met after the first num_groups.
    num_groups: int = 5

    # Dynamic group collection: after collecting `num_groups` groups, if fewer
    # than `min_successful_groups` had at least one rollout succeed, the
    # collector keeps adding one group at a time until the criterion is met
    # or `max_groups` is reached. Set to 0 to disable (always exactly num_groups).
    # Useful when many groups time out or fail entirely (dead groups contribute
    # zero gradient signal — see the dead-group filter in train_grpo.py).
    min_successful_groups: int = 4

    # Hard cap on dynamic group collection. Bounds worst-case wall time when
    # the task is too hard for the current policy. Must be >= num_groups and
    # <= 100 (the GROUP_SEED_STRIDE limit in collect_episodes.py). The
    # subprocess and RPC timeouts auto-scale from this value at 7 min/group,
    # matching the original 35 min budget for 5 groups.
    max_groups: int = 10

    # Maximum steps per episode before truncation (at 10Hz action rate).
    # Either a single int (applied to all envs) or a list of ints (one per env_name).
    # 720 steps = 72 seconds of sim time. Some tasks need more/less time.
    # Example: [720, 720, 400, 480, 720, 720, 400] for 7 envs with varying difficulty.
    max_episode_steps: int | list[int] = 480

    # How many steps from each 16-step action chunk to actually execute
    # Remaining steps discarded, fresh observation taken, new chunk predicted
    n_action_steps: int = 8

    # ─── Fast-Forward Branching ──────────────────────────────────────────────
    # Skip the early approach phase by fast-forwarding a single env, then
    # branching all group_size envs from that intermediate state. This focuses
    # GRPO signal on the critical manipulation phase (grasp, placement, etc.)
    # rather than the less consequential approach trajectory.
    #
    # When active for a group:
    #   1. One env runs solo for fast_forward_steps outer steps
    #   2. Its MuJoCo sim state is saved
    #   3. All group_size envs restore that state and diverge independently
    # Pattern adapted from scripts/denoising_lab/eval/branching_rollout.py.

    # Number of outer steps (action chunks) to fast-forward before branching.
    # Either a single int (applied to all envs) or a list of ints (one per env_name).
    # 0 = disabled. 10 outer steps = 80 sub-steps at n_action_steps=8.
    fast_forward_steps: int | list[int] = 12

    # Fraction of groups that use fast-forward (rest start from seed normally).
    # Mixing ensures the full trajectory stays in the training distribution,
    # preventing approach-phase drift from lack of gradient signal.
    # 0.0 = never fast-forward, 1.0 = always fast-forward.
    fast_forward_pct: float = 0.8

    # ZMQ server host and port for model inference during collection
    server_host: str = "127.0.0.1"
    server_port: int = 5555

    # Optional long-running collector server (collector_server.py). When
    # collector_server_host is non-empty, the trainer connects via
    # CollectorClient instead of spawning `python collect_episodes.py` per
    # iteration — eliminates the ~10-20s startup cost (robocasa imports +
    # AsyncVectorEnv worker spawn). The server must be started separately
    # with --env-names matching this config; mismatched env_names raise on
    # the first collect() request. Empty host = subprocess fallback.
    collector_server_host: str = ""
    collector_server_port: int = 5556

    # RoboCasa environment names to train on.
    # Tasks are selected round-robin: iteration 1 → task 0, iteration 2 → task 1, etc.
    # Each iteration collects ALL num_groups for a SINGLE task (not distributed across tasks).
    # With 8 tasks and 200 iterations, each task gets 25 full training updates.
    env_names: list[str] = field(default_factory=lambda: [
        "robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env",
        # "robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env",
        # "robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env",
        # "robocasa_panda_omron/TurnOffStove_PandaOmron_Env",
        # "robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env",
        # "robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env",
        # "robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env",
        # "robocasa_panda_omron/TurnOnStove_PandaOmron_Env",
    ])

    # Directory to store collected episode .npz files
    episode_dir: str = "grpo_data/grpo_episodes"

    # How many of the most recent iter_*/ subdirs to keep under episode_dir.
    # After each iteration's episodes are saved, older iter_*/ dirs are pruned
    # to bound disk usage. Default 3 keeps the current iter + 2 prior for
    # post-mortem inspection. Set to 0 to disable pruning (keep everything).
    # At 25 episodes/iter × 90 chunks × ~250KB/chunk ≈ 0.5 GB/iter, 200 iters
    # is ~100 GB if unpruned; /tmp on most GPU hosts is much smaller.
    episode_dirs_to_keep: int = 3

    # ─── Reward Shaping ──────────────────────────────────────────────────────

    # Weight of binary success signal in shaped reward
    # reward = success_weight * success + (1 - success_weight) * max_progress
    success_weight: float = 1.0

    # ─── GRPO Algorithm ──────────────────────────────────────────────────────
    # These directly mirror grpo_cont.py's clipped objective args

    # Clipping epsilon — prevents too-large policy updates
    # Same as grpo_cont.py's args.clip_eps = 0.2
    clip_eps: float = 0.2

    # Number of optimization epochs over collected data per each iteration
    # each epoch shuffles all action chunks from data collection
    # for each iter in num_iterations, we do a grad update (update_epochs * (total action chunks // mini_batch_size))
    # Same as grpo_cont.py's args.update_epochs = 10
    update_epochs: int = 5

    # Mini-batch size (in # of action chunks) for each gradient step within each epoch in update_epochs
    # If we collected 200 action chunks and mini_batch_size=10, then we will do 20 grad updates per epoch
    # Smaller = more updates per epoch but noisier gradients
    mini_batch_size: int = 8

    # KL divergence penalty coefficient (regularization toward reference policy)
    # Same role as grpo_cont.py's args.kl_coef = 0.002
    kl_coef: float = 0.1

    # Timestep centers (τ values) for FM log-prob evaluation during TRAINING ONLY.
    # This does NOT affect inference (action generation always uses exactly 4 Euler steps).
    # K = len(tau_centers) determines how many points along the noise→action interpolation
    # path we probe to estimate how well the model predicts the velocity field.
    # Each center gets small Gaussian jitter (std=0.02) during sampling.
    # Default is late-biased: denser at later τ where velocity prediction errors
    # have more impact on action quality (fewer Euler steps left to correct).
    # Each center = one DiT forward pass. A single shared noise ε is reused across all K.
    tau_centers: list[float] = field(default_factory=lambda: [
        0.0, 0.25, 0.35, 0.5, 0.6, 0.75
    ])

    # ─── Optimizer ───────────────────────────────────────────────────────────

    # Learning rate — 10x lower than supervised finetuning (1e-4)
    # RL gradients are noisier, so we need smaller steps
    learning_rate: float = 1e-5

    # AdamW weight decay (L2 regularization on LoRA weights)
    weight_decay: float = 1e-5

    # Maximum gradient norm for clipping (prevents explosion from rare high-advantage samples)
    # Same role as grpo_cont.py's args.max_grad_norm = 0.5
    max_grad_norm: float = 0.5

    # ─── Training Loop ───────────────────────────────────────────────────────

    # Total number of collect-train iterations
    num_iterations: int = 200

    # Resume from a previous checkpoint directory (e.g., "/tmp/grpo_checkpoints/iter_0050").
    # If set, loads LoRA weights + optimizer state and continues from that iteration.
    # If None, starts fresh training from the base pretrained model.
    resume_from: Optional[str] = None

    # Directory for checkpoints (LoRA weights + optimizer state)
    checkpoint_dir: str = "grpo_data/grpo_checkpoints"

    # Save checkpoint every N iterations
    save_interval: int = 2

    # Random seed for reproducibility
    seed: int = 67

    # ─── Logging ─────────────────────────────────────────────────────────────

    # Whether to use wandb for experiment tracking
    use_wandb: bool = False

    # Wandb project name
    wandb_project: str = "groot-grpo"

    # Wandb run name (auto-generated if None)
    wandb_run_name: Optional[str] = None

    # Suppress collector-side import noise (robosuite [WARNING]/[INFO], mimicgen
    # `print`, gymnasium passive_env_checker UserWarning) and the per-iter
    # process-memory diagnostics ([worker_mem pid=...] from collect_episodes.py
    # and [mem iter ...] from train_grpo.py). Real operational warnings —
    # collector failures, non-finite-loss skips, partial collections — are NOT
    # affected. The trainer propagates this to the collector subprocess via the
    # GRPO_CLEAN_OUTPUT=1 env var, so AsyncVectorEnv workers (spawn) pick it up
    # too. Note: the long-running collector_server (when collector_server_host
    # is set) is launched separately — to silence its output, start it with
    # GRPO_CLEAN_OUTPUT=1 in the env yourself.
    clean_output: bool = True

    def __post_init__(self):
        """Validate config invariants at construction time.

        Catches misconfigurations BEFORE the trainer spends ~1 minute on
        model load + server bind, so the operator gets immediate feedback
        instead of the "subprocess exited 1" or RPC FatalCollectorError
        path that would otherwise surface the same error several minutes in.

        Mirror constraints with EpisodeCollector.collect()'s runtime
        validation (collect_episodes.py:508-525), but check here too so a
        misconfigured trainer never reaches collection.
        """
        if self.num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {self.num_groups}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.save_interval < 1:
            raise ValueError(
                f"save_interval must be >= 1, got {self.save_interval}"
            )
        if self.max_groups < self.num_groups:
            raise ValueError(
                f"max_groups ({self.max_groups}) must be >= num_groups "
                f"({self.num_groups})"
            )
        # GROUP_SEED_STRIDE=1000 in collect_episodes.py × max_groups must
        # stay below the trainer's per-iter seed stride (100_000) or two
        # consecutive iters' seed ranges overlap. max_groups=100 is the
        # boundary (last seed = base + 99_000, next iter at base + 100_000).
        if self.max_groups > 100:
            raise ValueError(
                f"max_groups ({self.max_groups}) must be <= 100 to avoid "
                f"seed-range collisions with the next iter (per-iter stride "
                f"is 100_000 in train_grpo.py, group stride is 1000 in "
                f"collect_episodes.py)."
            )
        if self.min_successful_groups < 0:
            raise ValueError(
                f"min_successful_groups must be >= 0, got "
                f"{self.min_successful_groups}"
            )
        if self.min_successful_groups > self.max_groups:
            raise ValueError(
                f"min_successful_groups ({self.min_successful_groups}) cannot "
                f"exceed max_groups ({self.max_groups}) — criterion would be "
                f"unsatisfiable."
            )
