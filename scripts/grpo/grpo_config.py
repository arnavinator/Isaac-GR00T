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

    # LoRA dropout — regularization to prevent overfitting with RL's noisy gradients
    lora_dropout: float = 0.05

    # Which layers in the DiT to apply LoRA to (must be nn.Linear, NOT CategorySpecificLinear)
    # These are the module name patterns within model.action_head.model (AlternateVLDiT)
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "attn1.to_q",      # Self/cross-attention query projection [1536, 1536] or [2048, 1536]
        "attn1.to_k",      # Self/cross-attention key projection
        "attn1.to_v",      # Self/cross-attention value projection
        "attn1.to_out.0",  # Attention output projection [1536, 1536]
        "ff.net.0.proj",   # FeedForward GEGLU gate projection [1536, 2*inner_dim]
        "ff.net.2",        # FeedForward output projection [inner_dim, 1536]
        "proj_out_1",      # DiT conditioning projection [1536, 3072]
        "proj_out_2",      # DiT final output projection [1536, 1024]
    ])

    # Device for model training (typically "cuda" or "cuda:0")
    device: str = "cuda"

    # ─── Episode Collection ──────────────────────────────────────────────────

    # Number of rollouts per group (G = "answers per question")
    # Each group resets G parallel environments with the SAME seed (identical initial state).
    # Different rollouts diverge due to policy noise (denoising randomness).
    # Advantages are computed by comparing outcomes WITHIN a group.
    # Same as grpo_cont.py's args.num_envs = 5
    # Also determines the number of parallel environments (one env per rollout).
    group_size: int = 5

    # Number of groups per iteration ("questions per iteration")
    # Each group gets a unique seed → unique initial kitchen/object configuration.
    # More groups = more diverse gradient signal per update.
    # Same as grpo_cont.py's args.num_groups = 5
    # Total episodes per iteration = group_size × num_groups
    num_groups: int = 12

    # Maximum steps per episode before truncation (at 10Hz action rate).
    # Either a single int (applied to all envs) or a list of ints (one per env_name).
    # 720 steps = 72 seconds of sim time. Some tasks need more/less time.
    # Example: [720, 720, 400, 480, 720, 720, 400] for 7 envs with varying difficulty.
    max_episode_steps: int | list[int] = 720

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
    fast_forward_steps: int | list[int] = 0

    # Fraction of groups that use fast-forward (rest start from seed normally).
    # Mixing ensures the full trajectory stays in the training distribution,
    # preventing approach-phase drift from lack of gradient signal.
    # 0.0 = never fast-forward, 1.0 = always fast-forward.
    fast_forward_pct: float = 0.5

    # ZMQ server host and port for model inference during collection
    server_host: str = "127.0.0.1"
    server_port: int = 5555

    # RoboCasa environment names to train on.
    # Tasks are selected round-robin: iteration 1 → task 0, iteration 2 → task 1, etc.
    # Each iteration collects ALL num_groups for a SINGLE task (not distributed across tasks).
    # With 7 tasks and 200 iterations, each task gets ~28 full training updates.
    env_names: list[str] = field(default_factory=lambda: [
        "robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env",
        "robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env",
        "robocasa_panda_omron/TurnOffStove_PandaOmron_Env",
        "robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env",
        "robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env",
        "robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env",
        "robocasa_panda_omron/TurnOnStove_PandaOmron_Env",
    ])

    # Directory to store collected episode .npz files
    episode_dir: str = "/tmp/grpo_episodes"

    # ─── Reward Shaping ──────────────────────────────────────────────────────

    # Whether to use dense reward shaping (strongly recommended for sparse envs)
    use_dense_reward: bool = True

    # Weight of binary success signal in shaped reward
    # reward = success_weight * success + (1 - success_weight) * max_progress
    success_weight: float = 0.7

    # ─── GRPO Algorithm ──────────────────────────────────────────────────────
    # These directly mirror grpo_cont.py's clipped objective args

    # Clipping epsilon — prevents too-large policy updates
    # Same as grpo_cont.py's args.clip_eps = 0.2
    clip_eps: float = 0.2

    # Number of optimization epochs over collected data per iteration
    # Same as grpo_cont.py's args.update_epochs = 10
    update_epochs: int = 10

    # Mini-batch size (in action chunks) for each gradient step
    # Smaller = more updates per epoch but noisier gradients
    mini_batch_size: int = 8

    # KL divergence penalty coefficient (regularization toward reference policy)
    # Same role as grpo_cont.py's args.kl_coef = 0.002
    # Higher here (0.01) because FM log-prob surrogate is noisier than Gaussian
    kl_coef: float = 0.01

    # How often to update the reference model (every N iterations)
    # Reference stays frozen between updates, providing stable KL anchor
    ref_update_interval: int = 5

    # Number of timestep samples (K) for FM log-prob variance reduction
    # Each sample adds one DiT forward pass per action chunk.
    # K=4 matches the 4 inference denoising steps (τ = 0, 0.25, 0.5, 0.75).
    # A single shared noise ε is used across all K timestep samples.
    n_fm_samples: int = 4

    # ─── Optimizer ───────────────────────────────────────────────────────────

    # Learning rate — 10x lower than supervised finetuning (1e-4)
    # RL gradients are noisier, so we need smaller steps
    learning_rate: float = 1e-5

    # AdamW weight decay (L2 regularization on LoRA weights)
    weight_decay: float = 1e-5

    # Maximum gradient norm for clipping (prevents explosion from rare high-advantage samples)
    # Same role as grpo_cont.py's args.max_grad_norm = 0.5
    max_grad_norm: float = 1.0

    # ─── Training Loop ───────────────────────────────────────────────────────

    # Total number of collect-train iterations
    num_iterations: int = 200

    # Resume from a previous checkpoint directory (e.g., "/tmp/grpo_checkpoints/iter_0050").
    # If set, loads LoRA weights + optimizer state and continues from that iteration.
    # If None, starts fresh training from the base pretrained model.
    resume_from: Optional[str] = None

    # Directory for checkpoints (LoRA weights + optimizer state)
    checkpoint_dir: str = "/tmp/grpo_checkpoints"

    # Save checkpoint every N iterations
    save_interval: int = 10

    # Random seed for reproducibility
    seed: int = 67

    # ─── Logging ─────────────────────────────────────────────────────────────

    # Whether to use wandb for experiment tracking
    use_wandb: bool = True

    # Wandb project name
    wandb_project: str = "groot-grpo"

    # Wandb run name (auto-generated if None)
    wandb_run_name: Optional[str] = None

    # Whether to run density-aware diagnostics during collection (+12% overhead)
    # Provides per-chunk log-likelihood estimates for monitoring
    use_density_diagnostics: bool = False
