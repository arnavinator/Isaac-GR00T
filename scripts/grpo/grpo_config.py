"""GRPO training configuration for GR00T N1.6 DiT.

This dataclass mirrors the structure of grpo_cont.py's `init_args()` but adapted for:
- Flow-matching diffusion (FM log-prob surrogate instead of Gaussian policy)
- Server-client episode collection (instead of vectorized envs)
- Episodic sparse+shaped rewards (instead of per-step dense rewards)
- LoRA finetuning (instead of full parameter updates)

Usage:
    config = GRPOConfig()                         # defaults
    config = GRPOConfig(lora_rank=32, kl_coef_last_iter=0.005)  # override
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
    # Also the DEFAULT number of parallel environments (one env per rollout),
    # unless num_async_vector_env overrides it (see below).
    group_size: int = 8

    # Number of physical AsyncVectorEnv workers used to collect each group.
    # None → resolves to group_size (one worker per rollout — behavior 100%
    # unchanged from before this knob existed). When set, it must satisfy
    # 1 <= num_async_vector_env <= group_size AND
    # group_size % num_async_vector_env == 0. Each logical group of group_size
    # rollouts is then collected over k = group_size // num_async_vector_env
    # sequential "turns" of num_async_vector_env rollouts each, every turn
    # restarting from the same bit-identical branch-point state (captured via
    # apply_scene_bundle) and tagged with the same group_id. Within-group
    # diversity still comes only from per-query denoising noise (unseeded), so
    # turns are genuinely diverse. Lower this to cap peak worker RAM
    # (group_size MuJoCo workers can exceed host RAM) at the cost of ~k×
    # collection wall time per group.
    num_async_vector_env: Optional[int] = 4

    # Number of groups per iteration ("questions per iteration")
    # Each group gets a unique seed → unique initial kitchen/object configuration.
    # More groups = more diverse gradient signal per update.
    # Same as grpo_cont.py's args.num_groups = 5
    # With dynamic group collection (see min_alive_groups), this is the
    # MINIMUM number of groups; the collector may collect more (up to max_groups)
    # if the alive criterion isn't met after the first num_groups.
    num_groups: int = 3

    # Dynamic group collection: after collecting `num_groups` groups, if fewer
    # than `min_alive_groups` are ALIVE (mixed: 0 < group_successes <
    # group_size, equivalently per-group reward std > 0 under success_weight=
    # 1.0 with time-scaling disabled), the collector keeps adding one group at
    # a time until the criterion is met or `max_groups` is reached. Set to 0
    # to disable (always exactly num_groups).
    #
    # Why "alive" not "≥1 success": only mixed groups produce non-zero
    # gradient signal (compute_advantages in episode_buffer.py drops any
    # group with std < 1e-4 to advantage=0). All-success groups have
    # group_successes=group_size > 0 but std=0 and contribute nothing — under
    # the previous "≥1 success" criterion they were silently counted as
    # satisfying the gate, so an iteration where the policy got too good on
    # a scene could wrap collection happily and then train on zero live
    # chunks. The alive predicate fixes this: in the early/low-success
    # regime it is bit-identical to the old criterion (mixed iff ≥1 success
    # iff ≥1 success AND ≥1 fail when no group is fully solved); in the
    # transition regime it correctly demands actual gradient signal.
    min_alive_groups: int = 2

    # Hard cap on dynamic group collection. Bounds worst-case wall time when
    # the task is too hard for the current policy. Must be >= num_groups and
    # <= 100 (the GROUP_SEED_STRIDE limit in collect_episodes.py). The
    # subprocess and RPC timeouts auto-scale from this value at 7 min/group,
    # matching the original 35 min budget for 5 groups.
    max_groups: int = 5

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

    # ─── Init from saved sim state (overfitting / curriculum) ────────────────
    # When set, every group's branch point is loaded from this saved-state npz
    # instead of being produced by env.reset(seed=...) (and instead of running
    # the current model forward via fast-forward). Used for overfitting GRPO on
    # a specific intermediate state — e.g., step 10 of a known failing episode —
    # for analysis or curriculum.
    #
    # The npz must contain __sim_state__, __model_xml__, __ep_meta__ as produced
    # by scripts/denoising_lab/eval/interactive_rollout.py (or any other saver
    # that follows the branching_rollout.py:182-210 contract).
    #
    # Interactions with other knobs:
    #   - Internally short-circuits the fast-forward path; fast_forward_steps /
    #     fast_forward_pct are ignored. Set fast_forward_pct=0.0 explicitly to
    #     make the intent visible in logs.
    #   - min_alive_groups should be 0 — every group starts from the same
    #     hard state, so "N alive groups" is also not the criterion you want
    #     (and the dynamic loop is meaningless when every group has the same
    #     scene).
    #   - success_weight < 1.0 is strongly recommended; from a hard saved state
    #     binary-only reward typically produces dead groups (every rollout fails
    #     identically → std=0 → zero advantage → no learning). With shaped reward
    #     enabled, max_progress varies with denoising noise and provides signal.
    init_state_npz_path: Optional[str] = None

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
    episode_dirs_to_keep: int = 2

    # ─── Reward Shaping ──────────────────────────────────────────────────────

    # Weight of binary success signal in shaped reward
    # reward = success_weight * success + (1 - success_weight) * max_progress
    success_weight: float = 1.0

    # ─── GRPO Algorithm ──────────────────────────────────────────────────────
    # These directly mirror grpo_cont.py's clipped objective args

    # Clipping epsilons — prevent too-large policy updates. Split into separate
    # lower and upper bounds (DAPO-style "Clip-Higher") so the clipped surrogate
    # can be asymmetric: clamp(ratio, 1 - clip_eps_low, 1 + clip_eps_high).
    # Both default to 0.2 — the symmetric clip identical to the original single
    # clip_eps = 0.2 (same as grpo_cont.py's args.clip_eps = 0.2). Raise
    # clip_eps_high above clip_eps_low to allow more upside exploration.
    # Constraint: each must lie in (0.0, 1.0) (validated in __post_init__). No
    # ordering constraint between the two — any low/high pair is allowed.
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2

    # Number of optimization epochs over collected data per each iteration
    # each epoch shuffles all action chunks from data collection
    # for each iter in num_iterations, we do a grad update (update_epochs * (total action chunks // mini_batch_size))
    # Same as grpo_cont.py's args.update_epochs = 10
    update_epochs: int = 2

    # ─── Balanced Training (two independent mechanisms) ──────────────────────
    # Both address gradient instability from skewed episode outcomes, and each
    # is now toggled by its OWN flag — any of the four on/off combinations is
    # valid. (Previously a single `balanced_training` flag gated both at once.)
    #
    #   1. balanced_minibatch_training — balanced mini-batch sampling: each
    #      mini-batch enforces the target pos/neg ratio in BOTH directions. The
    #      underrepresented sign class is the "minority" and is oversampled with
    #      replacement; the overrepresented class is the "majority" and is drawn
    #      without replacement, controlling when the epoch ends.
    #        - natural_pos_frac < pos_ratio: too few positives → cycle positives
    #        - natural_pos_frac > pos_ratio: too few negatives → cycle negatives
    #      Falls back to stratified sampling only when one sign class is entirely
    #      absent (no positives or no negatives). When False, uses the plain
    #      stratified-minibatch path.
    #
    #   2. dynamic_epoch_training — dynamic epoch count via a tent function of
    #      success_frac:
    #        m = min(successful_eps, total_eps − successful_eps)
    #        actual_epochs = max(1, (4·m·update_epochs + total_eps) // (2·total_eps))
    #      Peaks at success_frac=0.5 (→ full update_epochs); decays to 1 at both
    #      0% and 100% success. Reduces training at asymmetric extremes in either
    #      direction, preventing both under-training (sparse signal) and
    #      over-training (highly asymmetric advantages at high success). When
    #      False, always runs exactly update_epochs epochs.
    balanced_minibatch_training: bool = True
    dynamic_epoch_training: bool = False

    # Target fraction of positive-advantage chunks in each mini-batch.
    # Must be strictly in (0.0, 1.0). Only active when balanced_minibatch_training=True.
    # Default 0.5: equal split between positive and negative advantages.
    # Set higher (e.g. 0.7) to bias the gradient more toward success examples.
    balanced_minibatch_positive_adv_ratio: float = 0.5

    # Mini-batch size (in # of action chunks) for each gradient step within each epoch in update_epochs
    # If we collected 200 action chunks and mini_batch_size=10, then we will do 20 grad updates per epoch
    # Smaller = more updates per epoch but noisier gradients
    mini_batch_size: int = 8

    # KL divergence penalty coefficient — anchor toward THIS ITER'S start-of-update
    # policy (the "ref" snapshot taken in _compute_ref_log_probs before the GRPO
    # epochs fire). Bounds per-iter policy drift to prevent the clipped surrogate
    # from racing too far on noisy gradients within a single iter.
    # Same role as grpo_cont.py's args.kl_coef = 0.002.
    kl_coef_last_iter: float = 0.2

    # KL divergence penalty coefficient — anchor toward the BASE FROZEN DiT
    # (= current model with LoRA adapters disabled, so no extra params loaded).
    # Bounds CUMULATIVE drift from the pretrained policy across all iters,
    # complementing kl_coef_last_iter's per-iter bound. The base log-prob is
    # pre-computed once per iter inside the same no_grad pass that produces
    # ref_log_probs, then cached on each chunk — extra cost is one DiT forward
    # per iter (no second model in VRAM, no LoRA-disabled forward per minibatch).
    # 0.0 disables the term entirely (skips the extra pre-compute pass and the
    # per-mb KL formula). Suggested starting value: same order as kl_coef_last_iter.
    kl_coef_base_model: float = 0.2

    # Jitter-GRPO Jacobian regularizer strength (paired scheduling).
    # When > 0, every action chunk produces TWO entries per epoch: a "fixed"
    # version (DiT input noise = original ε) and a "jitter" version (DiT input
    # noise = ε' = sqrt(1-λ²)*ε + λ*ξ, with fresh Gaussian ξ sampled per τ
    # per minibatch from the global torch RNG). The velocity target a − ε
    # stays at the ORIGINAL ε in both branches; the cached chunk.ref_log_prob
    # (computed at original ε) is reused — the bias is O(λ²) and θ-independent,
    # so the gradient direction is unaffected. In expectation this adds a
    # Frobenius-norm Jacobian penalty (1-t)²·λ²·‖∇_x v_θ‖_F², encouraging the
    # velocity field to be locally smooth around each rolled-out trajectory.
    # Doubles optimizer steps per epoch — halve update_epochs MANUALLY for
    # jitter runs to match the per-iter optimizer-step budget of vanilla GRPO.
    # 0.0 disables (bit-identical to vanilla GRPO). Suggested value 0.05.
    jitter_lambda: float = 0.0

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
    learning_rate: float = 3e-5

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

    # Skip the FIRST resumed iter's collection phase by loading episodes that
    # are already on disk under episode_dir/iter_{start_iteration:04d}/. Only
    # honored when resume_from is also set; rejected at config-construction
    # time otherwise. Common case: a prior run crashed AFTER finishing
    # collection but BEFORE the model update completed — the cached episodes
    # were collected by the policy whose weights live in resume_from, so they
    # remain on-policy for the resumed iter.
    #
    # The trainer pre-flight-validates the cache during setup() (dir exists,
    # >= num_groups distinct group_ids, env_name matches the round-robin task
    # for start_iteration, raw_action / action_mask / initial_noise keys
    # present, min_alive_groups criterion satisfied or max_groups cap
    # reached). Validation failures raise before the model is loaded so the
    # operator gets immediate feedback. Only the first resumed iter consumes
    # the cache; subsequent iters collect normally.
    #
    # When NOT to enable: do NOT set this flag if you've changed any
    # collection-affecting config since the cache was written. The validator
    # catches env_name and group-count mismatches but does NOT detect changes
    # to n_action_steps, fast_forward_steps / fast_forward_pct,
    # init_state_npz_path, max_episode_steps, or success_weight — the cached
    # iter would silently train on episodes from the OLD config while
    # subsequent iters collect under the new one. If in doubt, leave this
    # disabled and pay the collection cost.
    resume_from_collected_data: bool = False

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
        # env_names drives both the round-robin task selection per iter
        # (`env_names[(iteration - 1) % len(env_names)]`) and the
        # cached-data validator's expected-env check. An empty list raises
        # ZeroDivisionError on first iter — fail clearly at construction.
        if not self.env_names:
            raise ValueError(
                "env_names must be a non-empty list (at least one env to "
                "train on). Got empty list."
            )
        # num_async_vector_env: None means "one worker per rollout" (= group_size,
        # the original coupling). When set, it must evenly divide group_size and
        # not exceed it (collecting more physical envs than the logical group
        # size is out of scope and rejected). Validate the RESOLVED value so the
        # stored field can stay None and downstream resolves it identically.
        resolved_nave = (
            self.group_size
            if self.num_async_vector_env is None
            else self.num_async_vector_env
        )
        if resolved_nave < 1:
            raise ValueError(
                f"num_async_vector_env must be >= 1, got "
                f"{self.num_async_vector_env}"
            )
        if resolved_nave > self.group_size:
            raise ValueError(
                f"num_async_vector_env ({resolved_nave}) cannot exceed "
                f"group_size ({self.group_size}); collecting more physical envs "
                f"than the logical group size is out of scope."
            )
        if self.group_size % resolved_nave != 0:
            raise ValueError(
                f"group_size ({self.group_size}) must be divisible by "
                f"num_async_vector_env ({resolved_nave}) so each group is "
                f"collected in a whole number of equal turns "
                f"(k = group_size // num_async_vector_env)."
            )
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
        if self.min_alive_groups < 0:
            raise ValueError(
                f"min_alive_groups must be >= 0, got "
                f"{self.min_alive_groups}"
            )
        if self.min_alive_groups > self.max_groups:
            raise ValueError(
                f"min_alive_groups ({self.min_alive_groups}) cannot "
                f"exceed max_groups ({self.max_groups}) — criterion would be "
                f"unsatisfiable."
            )
        if not (0.0 <= self.jitter_lambda < 1.0):
            raise ValueError(
                f"jitter_lambda must be in [0.0, 1.0), got {self.jitter_lambda}. "
                f"Variance preservation requires λ < 1; use 0.0 to disable."
            )
        # KL coefficients must be NON-NEGATIVE. Each is multiplied by a Schulman
        # k3 KL term (non-negative pointwise) and added to the loss; a negative
        # coef inverts the sign and turns the anchor into a *reward for
        # divergence* — the policy actively flees the anchor. Worse, the
        # base-model branch is gated `compute_base = kl_coef_base_model > 0.0`,
        # so a negative value would silently disable the term entirely (no
        # warning, no anchoring effect at all). Fail fast with a clear message
        # so a CLI typo like `--kl-coef-last-iter -0.1` doesn't quietly run a
        # divergence-reward training campaign for hours.
        if self.kl_coef_last_iter < 0.0:
            raise ValueError(
                f"kl_coef_last_iter must be >= 0, got {self.kl_coef_last_iter}. "
                f"Use 0.0 to disable; negative values would invert the anchor "
                f"into a reward for divergence."
            )
        if self.kl_coef_base_model < 0.0:
            raise ValueError(
                f"kl_coef_base_model must be >= 0, got {self.kl_coef_base_model}. "
                f"Use 0.0 to disable (skips the base-model forward entirely); "
                f"negative values silently disable via the `> 0.0` gate AND "
                f"would invert the anchor's gradient direction if the gate "
                f"were ever loosened."
            )
        # The clipped surrogate clamps the importance ratio to
        # [1 - clip_eps_low, 1 + clip_eps_high]. Each epsilon must lie in the
        # open interval (0, 1); there is NO ordering constraint between them
        # (any low/high pair is allowed, including clip_eps_low > clip_eps_high).
        #   - Upper end (< 1): the importance ratio = exp(log_ratio) is always
        #     strictly positive, so clip_eps_low >= 1 drops the lower bound
        #     1 - clip_eps_low to <= 0 — a floor the ratio can never cross,
        #     silently disabling the downside clip. The same cap is applied to
        #     clip_eps_high for a uniform rule.
        #   - Lower end (> 0): eps == 0 gives a zero-width clip on that side
        #     (the bound is pinned to exactly 1), and eps < 0 inverts the
        #     window; both are degenerate, so require strictly positive values.
        if not (0.0 < self.clip_eps_low < 1.0) or not (0.0 < self.clip_eps_high < 1.0):
            raise ValueError(
                f"clip_eps_low and clip_eps_high must each lie in (0.0, 1.0), got "
                f"clip_eps_low={self.clip_eps_low}, clip_eps_high={self.clip_eps_high}. "
                f"The surrogate clamps the ratio to [1 - clip_eps_low, "
                f"1 + clip_eps_high]; a value >= 1 drops the lower bound to <= 0 "
                f"(downside clip never fires), and a value <= 0 gives a zero-width "
                f"or inverted clip window."
            )
        # success_weight is a probability-like blend weight in the shaped
        # reward (success_weight * success + (1-success_weight) * max_progress);
        # values outside [0, 1] silently invert the progress term (e.g.,
        # success_weight=2.5 → reward = 2.5*success - 1.5*max_progress) and
        # corrupt the training signal without crashing. Cheap to range-check.
        if not (0.0 <= self.success_weight <= 1.0):
            raise ValueError(
                f"success_weight must be in [0.0, 1.0], got {self.success_weight}. "
                f"It blends binary success and dense max_progress in the shaped "
                f"reward; values outside [0, 1] flip the sign of the progress term."
            )

        if self.balanced_minibatch_training and not (
            0.0 < self.balanced_minibatch_positive_adv_ratio < 1.0
        ):
            raise ValueError(
                f"balanced_minibatch_positive_adv_ratio must be strictly in "
                f"(0.0, 1.0) when balanced_minibatch_training=True, got "
                f"{self.balanced_minibatch_positive_adv_ratio}. "
                f"Use a value like 0.5 (equal split) or 0.7 (bias toward positives)."
            )

        # resume_from_collected_data is meaningless without a checkpoint to
        # resume from: the cache it would reuse was collected by the policy
        # whose weights live in resume_from. Reject at construction time so a
        # CLI typo doesn't silently fall through to fresh collection.
        #
        # Use `not self.resume_from` (truthy check) instead of `is None` so
        # an empty/whitespace string is also rejected — `is None` would let
        # `--resume-from ""` through, and the trainer's later `if config.
        # resume_from:` check would silently fall through to start_iteration=1
        # while still treating the cache flag as enabled, leading to a
        # fresh-weight model trained on iter_0001/'s cached episodes.
        if self.resume_from_collected_data:
            if not self.resume_from or not self.resume_from.strip():
                raise ValueError(
                    "resume_from_collected_data=True requires resume_from "
                    "to be a non-empty path. The flag tells the trainer to "
                    "reuse episodes that were collected by the policy at "
                    "resume_from's checkpoint; without a checkpoint, the "
                    "cached episodes have no on-policy guarantee."
                )

        # Resolve episode_dir to an absolute path so a CWD change between
        # config-construction (or setup()) and `train()` cannot make the
        # cached path point at a different filesystem location. Mirrors the
        # `init_state_npz_path` resolution pattern below — same motivation.
        # Path.resolve() works on non-existent paths (the trainer creates
        # episode_dir lazily), so this is safe at construction time.
        # Without this, a relative `--episode-dir grpo_data/...` combined
        # with a between-setup-and-train CWD change (notebook context, a
        # `gr00t/` import that os.chdir's, etc.) makes the loader emit a
        # confusing "directory was deleted" error pointing at a phantom
        # filesystem race.
        from pathlib import Path as _Path
        self.episode_dir = str(_Path(self.episode_dir).expanduser().resolve())

        # ── init_state_npz_path validations ─────────────────────────────────
        # These run at config-construction time so failures surface BEFORE the
        # trainer spends minutes on model load + server bind. The npz path
        # also needs to be resolved to an absolute path here so it remains
        # valid across processes: the long-running collector_server (and the
        # robocasa-venv subprocess) may have a different CWD than the trainer.
        if self.init_state_npz_path is not None:
            # Empty / whitespace path is almost certainly a CLI typo; reject
            # rather than waste a subprocess on np.load("").
            if not self.init_state_npz_path.strip():
                raise ValueError(
                    "init_state_npz_path is empty/whitespace; pass a real "
                    "path (or unset the flag to disable the override)."
                )
            # Embedded NUL bytes survive str checks but raise a cryptic
            # OS-level "embedded null character" deep inside pathlib's
            # stat() — wrap with a clearer message and the offending input.
            if "\x00" in self.init_state_npz_path:
                raise ValueError(
                    f"init_state_npz_path contains an embedded NUL byte "
                    f"({self.init_state_npz_path!r}). Most likely a quoting "
                    f"or env-var-injection bug at the call site."
                )
            from pathlib import Path
            _init_path = Path(self.init_state_npz_path).expanduser().resolve()
            if not _init_path.exists():
                raise FileNotFoundError(
                    f"init_state_npz_path does not exist: {_init_path} "
                    f"(passed as {self.init_state_npz_path!r}). Resolve relative "
                    f"to the trainer's CWD; double-check the path."
                )
            if not _init_path.is_file():
                raise ValueError(
                    f"init_state_npz_path is not a regular file: {_init_path} "
                    f"(maybe a directory?). Pass the .npz file path itself."
                )
            # Overwrite with the resolved absolute path so subprocess /
            # collector_server consumers don't depend on CWD.
            self.init_state_npz_path = str(_init_path)

            # NOTE: deliberately do NOT warn on success_weight=1.0 +
            # init_state or on min_alive_groups>0 + init_state. Both
            # are valid choices:
            #   - success_weight=1.0: pure sparse binary reward is a
            #     legitimate setup; the operator may know the policy
            #     succeeds intermittently from this state, or may want to
            #     deliberately avoid mixing in dense-progress shaping.
            #   - min_alive_groups>0: with all groups starting from
            #     the same saved state, requiring ≥N alive groups is a
            #     stability mechanism — each group draws independent
            #     denoising noise, so ≥N alive (mixed) groups gives a
            #     less noisy gradient direction and reduces policy-
            #     collapse risk from few-group updates.

            # Multiple env_names + a single saved npz is almost certainly a
            # config bug: the npz's sim_state has dims tied to one env's
            # MjModel (nq+nv), and round-robin would apply it to envs with
            # different dims on subsequent iters → MuJoCo errors. Even when
            # dims happen to match, the saved scene/objects belong to one
            # task and don't make sense for another.
            if len(self.env_names) > 1:
                import warnings
                warnings.warn(
                    f"init_state_npz_path is set with multiple env_names="
                    f"{self.env_names}. The saved sim state is tied to a "
                    f"specific env's MjModel; round-robin training will apply "
                    f"it to mismatched envs and crash (or silently corrupt "
                    f"state). Use a single env_name with init_state.",
                    stacklevel=3,
                )
