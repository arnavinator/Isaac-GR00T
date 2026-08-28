"""GRPO training configuration for GR00T N1.6 DiT.

This dataclass mirrors the structure of grpo_cont.py's `init_args()` but adapted for:
- Flow-matching diffusion (FM log-prob surrogate instead of Gaussian policy)
- Server-client episode collection (instead of vectorized envs)
- Episodic sparse binary rewards (instead of per-step dense rewards)
- LoRA finetuning (instead of full parameter updates)

Usage:
    config = GRPOConfig()                         # defaults
    config = GRPOConfig(lora_rank=32, kl_coef_last_iter=0.005)  # override
    # Or from CLI via tyro:
    # config = tyro.cli(GRPOConfig)
"""

from dataclasses import dataclass, field
from typing import Optional

import math

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
    # group_size, equivalently per-group reward std > 0), the collector keeps
    # adding one group at a time until the criterion is met or `max_groups` is
    # reached. Set to 0 to disable (always exactly num_groups).
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

    # Take one camera render per action chunk instead of one per substep.
    # The collector reads observations with video_delta_indices=[0], so only the
    # frame at the end of the chunk ever reaches the policy or the saved episode
    # — the other n_action_steps-1 renders (3 cameras each, for PandaOmron) are
    # discarded. Skipping them is observationally equivalent, not an
    # approximation: robosuite samples a camera observable on the LAST physics
    # substep of each control step, which is the same sim state the wrapper's
    # forced end-of-chunk render captures. See MultiStepWrapper.step for why
    # re-enabling mid-chunk instead would yield blank/stale frames, and
    # scripts/grpo/README.md for the full argument.
    # Set False to restore the old render-every-substep behavior (e.g. to rule
    # out a rendering-related regression).
    skip_intermediate_render: bool = True

    # Video observation keys (substring match) dropped before an observation
    # reaches the policy server or an episode .npz. RoboCasa emits
    # full-resolution passthrough copies next to the keys the model consumes —
    # `res512_image_*` beside every `res256_image_*` for PandaOmron, and
    # `ego_view_res1280x800_freq20` for GR1. Nothing in scripts/grpo,
    # gr00t/eval or gr00t/data reads them (the processor selects the
    # embodiment's configured modality keys), yet they are ~80% of the
    # per-chunk video bytes: 2.36 MB of 512x512 copies against 0.59 MB of the
    # 256x256 frames actually used.
    #
    # Dropping them shortens the npz write, the trainer read-back, the ZMQ
    # round trip per outer step, and the trainer's resident heap — the last of
    # which is what pushes MuJoCo workers into swap (see _release_memory_to_os).
    # Set to [] to keep every key. If a checkpoint ever consumes one of these,
    # the policy server raises on the missing modality key — loud, not silent.
    dropped_video_keys: list[str] = field(
        default_factory=lambda: ["res512", "ego_view_res1280x800"]
    )

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
    #   - From a hard saved state, binary-only reward can produce dead groups
    #     early (every rollout fails identically → std=0 → zero advantage → no
    #     learning). Pick a branch point the policy solves at least
    #     intermittently, or the iteration yields no gradient signal.
    init_state_npz_path: Optional[str] = None

    # ─── Frozen scene seed pool (between-scene variance reduction) ────────────
    # By DEFAULT every iteration trains on brand-new scenes. The trainer passes
    # `--seed config.seed + iteration * 100_000` to the collector, which then
    # derives `group_seed = base_seed + group_idx * GROUP_SEED_STRIDE`, so no
    # RoboCasa seed is ever revisited across a run. Measured between-scene
    # success-rate sd is 0.285, which makes ~84% of the per-iteration
    # `episode/mean_reward` variance pure scene RESAMPLING rather than policy
    # change — the training curve is then uninterpretable, because a swing
    # between consecutive iterations says nothing about the update that happened
    # in between.
    #
    # scene_seed_pool_size = K > 0 freezes a pool of K scene seeds and cycles it
    # deterministically across iterations, so the same scenes recur and
    # iteration-to-iteration differences are (mostly) policy differences. The
    # cursor is a pure function of `iteration` — see
    # GRPOTrainer._scene_seeds_for_iteration for the formula and why it is
    # stateless (it makes --resume-from correct by construction).
    #
    # 0 = DISABLED, and disabled is bit-identical to the behavior before this
    # feature existed: no `--group-seeds` argument is appended to the collector
    # argv at all, and no per-scene TB series are emitted.
    #
    # IMPORTANT when reading the curves: with K > num_groups the PER-ITERATION
    # success rate is exactly as noisy as before — each iteration still samples
    # only num_groups of the K scenes. What the pool buys is that the sequence
    # of iterations covers a FIXED scene set, so the readable unit becomes the
    # PASS mean over K/num_groups consecutive iterations (see
    # `episode/pool_pass`). Set K == num_groups if you want every single
    # iteration to be directly comparable, at the cost of training on only
    # num_groups distinct scenes for the whole run.
    scene_seed_pool_size: int = 0

    # First seed of the pool; the rest are `base + j * GROUP_SEED_STRIDE` for
    # j in [0, K). None → resolved IN __post_init__ (not at the use site, so the
    # resolved value shows up in the TensorBoard config text dump —
    # GRPOTrainer._log_config walks dataclasses.fields and would otherwise
    # record a `None` that tells a later reader nothing about which scenes ran)
    # to `seed + 100_000`.
    #
    # Why that default: `seed + 1 * 100_000` is EXACTLY the seed block that
    # iteration 1 would have drawn under the old per-iteration formula, so a
    # pooled run starts on the same scenes a baseline run's first iteration saw.
    # With the default seed=67 and K=12 the pool is
    # 100067, 101067, ..., 111067.
    scene_seed_pool_base: Optional[int] = None

    # ZMQ server host and port for model inference during collection
    server_host: str = "127.0.0.1"
    server_port: int = 5555

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

    # ─── Advantage normalization & positive-advantage weighting ──────────────

    # Advantage normalization scope. False (default) = z-score PER MINIBATCH
    # (current behavior). True = z-score once per ITERATION using the mean/std
    # over ALL live chunks: ready_adv = (a − buffer_mean) / (buffer_std + 1e-8).
    # buffer_mean ≈ 0 by group-relative construction (Σ A_ep = 0 per group), so
    # this PRESERVES each chunk's good/bad SIGN — unlike per-minibatch renorm,
    # which subtracts the (balanced-sampler-biased) minibatch mean and can flip
    # a genuinely-good chunk negative. Also removes minibatch-composition
    # coupling: a chunk's effective advantage no longer depends on its batchmates.
    per_iteration_advantage_norm: bool = False

    # When True, scale up the per-row clip loss on group-good rows by a live
    # factor k that balances alive positive/negative loss mass (see train_grpo).
    # False (default) = no weighting, bit-identical to current. Positives are
    # rarely upper-clipped (r ≤ e^{MSE_ref} ≈ 1.008 in practice), so k scales the
    # reinforcement gradient A·r·∂MSE_θ/∂θ directly; the ratio cap does not defeat
    # it. Designed to pair with per_iteration_advantage_norm=True: under that norm
    # a chunk's post-renorm sign equals its group sign, so the alive positive/
    # negative mass classification is exact. Under per-minibatch norm renorm
    # sign-flips make the classification approximate (k slightly off) but safe.
    positive_advantage_weight_scaling: bool = False

    # Hard cap on the dynamic weight (k is clamped to [1.0, this]). The natural
    # dynamics are ~10× lopsided toward erosion, so low-single-digits is sensible;
    # 5.0 is a moderate default. Only consulted when the scaling flag is True.
    positive_advantage_weight_max: float = 10.0

    # Desired POST-weighting ratio of amplified-positive loss mass to alive-
    # negative (erosion) loss mass. k solves (k·D)/N = target_ratio, i.e.
    # k = target_ratio · N/D, clamped to [1.0, max] (N = alive-negative mass,
    # D = alive amplified-positive mass). 1.0 equalizes the two (weighted positive
    # mass == erosion mass); because the FM surrogate lets negatives move the
    # ratio ~10× further than positives, the natural balance skews toward erosion,
    # so values > 1 tilt further toward reinforcement. Only consulted when the
    # scaling flag is True.
    positive_advantage_weight_target_ratio: float = 1.0

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
    # When balanced_minibatch_positive_adv_ratio_dynamic=True this value is the
    # FLOOR of the dynamic ratio (see below).
    balanced_minibatch_positive_adv_ratio: float = 0.5

    # Dynamic positive-advantage ratio. When True, the sampler no longer targets a
    # fixed positive fraction; it tracks the NATURAL positive fraction (≈ success
    # rate) per iteration, clamped to
    # [balanced_minibatch_positive_adv_ratio, balanced_minibatch_positive_adv_ratio_max].
    # Why: a fixed 0.5 oversamples whichever sign is RARE. At LOW success that's the
    # positives (desirable — preserves reinforcement signal). But at HIGH success
    # the rare class is the FAILURES, and oversampling those few large-advantage
    # failures — which on a single scene share structure with the successes —
    # over-suppresses the good behavior and can collapse the policy. Tracking the
    # natural fraction stops that negative-oversampling at high success while the
    # floor keeps positive oversampling at low success. Only active when
    # balanced_minibatch_training=True.
    balanced_minibatch_positive_adv_ratio_dynamic: bool = False

    # Upper cap for the dynamic positive-advantage ratio (only consulted when
    # balanced_minibatch_positive_adv_ratio_dynamic=True). Keeps SOME negative
    # (failure-avoidance) signal even at very high success. Must lie in (0, 1) and
    # be >= balanced_minibatch_positive_adv_ratio.
    balanced_minibatch_positive_adv_ratio_max: float = 0.75

    # ─── Anchor groups (all-success groups) ──────────────────────────────────
    # An ALL-SUCCESS group (k == group_size) has per-group reward std == 0, so
    # the group-mean baseline gives every episode advantage exactly 0 and the
    # whole group is dropped as dead. That is correct policy-gradient behavior
    # (nothing to improve), but it also means the trust region never sees the
    # states the policy already solves, and at high success rates most of the
    # buffer disappears while the surviving mixed groups are dominated by rare
    # large-negative failures.
    #
    # These two knobs reclassify all-success groups as ANCHOR groups. ALL-FAIL
    # and singleton groups stay dead — pushing DOWN on every rollout from a
    # state gives no direction to move toward and is the documented cause of the
    # v2 avoidance-gradient collapse. See README "Anchor groups".
    #
    # include_anchor_groups admits all-success chunks into the ref pass and the
    # update. With anchor_advantage == 0 their clip term is identically 0, so the
    # only thing they add to the LOSS is the KL anchors (kl_coef_last_iter /
    # kl_coef_base_model) — a retention constraint with no reward signal of its
    # own. Not a literal no-op, though: the rows occupy minibatch slots, so they
    # change each batch's renorm sample and raise the per-iteration optimizer
    # step count (see README "Anchor groups").
    include_anchor_groups: bool = False

    # Positive advantage assigned to each episode of an anchor group, in the
    # same units as the group-relative z-scores of mixed groups. 0.0 = KL-only
    # (see above). Requires include_anchor_groups=True.
    #
    # Choosing the value — pseudo-count (Beta-Bernoulli) baseline. k == G is not
    # proof that p == 1: at G=8 a state with true p=0.85 returns 8/8 about 27% of
    # the time, so the MLE baseline 1.0 over-estimates and each success really did
    # earn positive advantage. Replace the group mean with the posterior mean
    # under κ pseudo-counts at prior success rate p̄, and divide by a FIXED scale
    # (the group's own std is 0):
    #
    #   b_g      = (Σ r_i + κ·p̄) / (G + κ)
    #   A_anchor = (1 − b_g) / σ_fixed  =  κ(1 − p̄) / ((G + κ)·σ_fixed)
    #
    # With κ=2, p̄=0.5 (Laplace's rule of succession) and σ_fixed=0.5 (the max
    # Bernoulli std, ≈ the std of a balanced G/2 group): A_anchor = 2/(G+2), i.e.
    # 0.200 at group_size=8 and 0.143 at group_size=12. Today's behavior is the
    # κ=0 case. The value is a CONSTANT (κ, p̄, σ_fixed are only identifiable as
    # this one combination), so it is configured directly rather than derived at
    # runtime; recompute it if you change group_size. For scale: at G=12 a
    # balanced 6/12 group's successes sit at ±0.96 and the weakest signal row
    # that exists at all is ±0.29.
    #
    # Deliberately NOT tied to the running success rate: the estimator wants the
    # anchor to fade as success climbs, while the negative-mass asymmetry wants
    # it strongest exactly then. Keeping it fixed makes the effect readable and
    # leaves the asymmetry to the balanced-sampler mechanisms above.
    anchor_advantage: float = 0.0

    # Anchor row budget, as a multiple of the SIGNAL (mixed-group) chunk count.
    # Anchor episodes are kept in index order until the budget is met; the rest
    # are dropped back to dead (logged, never silent). Anchor advantages are not
    # zero-sum within a group, so dropping individual anchor episodes distorts
    # nothing — unlike signal groups, where it would break Σ A_ep = 0.
    # 1.0 = anchor rows may be at most as numerous as signal rows. WAIVED (and
    # logged) whenever there are no signal chunks at all — not just an
    # all-success iteration, but any iteration with no mixed group, e.g. an
    # all-fail + all-success mix, which carries a non-zero std_reward and so does
    # not hit the trainer's outer skip either. Above ~7 the per-batch cap stops
    # an epoch from covering the pool (see README "Row budget and cost"). Each anchor row costs the same
    # len(tau_centers) DiT forwards as a signal row in both the ref pass and
    # every update epoch, so this is the compute knob as well as the strength
    # knob — halving it halves both the cost and the anchor's share of the
    # gradient (the per-minibatch quota is proportional and may be fractional,
    # so this holds for pools smaller than one row per minibatch too).
    anchor_max_row_frac: float = 1.0

    # Mini-batch size (in # of action chunks) for each gradient step within each epoch in update_epochs
    # If we collected 200 action chunks and mini_batch_size=10, then we will do 20 grad updates per epoch
    # Smaller = more updates per epoch but noisier gradients
    mini_batch_size: int = 8

    # Number of consecutive mini-batches whose gradients are ACCUMULATED into a
    # single optimizer step ("k"). 1 (default) is bit-identical to the
    # pre-accumulation behavior: zero_grad → backward → clip_grad_norm_ → step
    # once per mini-batch. k > 1 zeroes the grad buffer once per WINDOW of k
    # mini-batches, scales each mini-batch's loss by 1/k before backward(), and
    # runs a single clip_grad_norm_ + step on the accumulated (averaged)
    # gradient. Effective rows per optimizer step = k * mini_batch_size.
    #
    # Why this knob exists: mini_batch_size cannot be raised. Peak VRAM at
    # mini_batch_size=8 is ~21.5 GB of ~25.3 GB on an A10G (~1.48 GB per row,
    # so ~8-9 rows is the ceiling). The per-row cost is dominated by the K-loop
    # in fm_log_prob.compute_fm_log_prob, which accumulates the log-prob across
    # all len(tau_centers) DiT forward passes and calls backward() ONCE — so
    # autograd retains the activations of all K passes simultaneously.
    # Accumulation is therefore the only route to a larger effective batch: it
    # holds peak VRAM and total forward/backward work constant while cutting the
    # optimizer-step count by ~k and reducing update-direction noise. Step size
    # is unaffected: the LR schedule is per-ITERATION (the monotone anneal ramp
    # in train() — there is no warmup, iteration 1 already runs at the full
    # configured LR), not per-step, so k does not rescale it — hold LR fixed.
    #
    # NOT equivalent to one true (k * mini_batch_size)-row batch, deliberately.
    # The advantage z-score still runs INDEPENDENTLY on each micro-batch of
    # mini_batch_size rows (per_iteration_advantage_norm stays False — see its
    # comment above), so what this averages is k independently normalized
    # micro-batch gradients. That is the intended semantics, not an
    # approximation of a single wide batch: the group-relative binary-reward
    # advantage is strongly asymmetric (at 12.5% success, +2.475 for a success
    # vs -0.354 for a failure, ~7:1), and per-minibatch z-scoring restores
    # symmetry. Switching to per-iteration norm to make accumulation "exact"
    # passes that 7:1 asymmetry straight through, strips the failure-avoidance
    # signal, and silently pins pos_adv_weight_k to its 1.0 floor (disabling
    # PAWS) — it measured much worse on matched iterations.
    #
    # Behavior at the edges (see _grpo_update_inner for the implementation):
    #   - A partial window at the end of an epoch is FLUSHED, never discarded.
    #     Because the scale is a uniform 1/k, a flushed window of m < k
    #     micro-batches yields (m/k)x the average gradient — a proportionally
    #     smaller step. At most one such step per epoch.
    #   - Mini-batches dropped by the non-finite-loss guard never reach
    #     backward(), so they contribute nothing to the buffer AND do not count
    #     toward the window: every full window holds exactly k TRAINED
    #     micro-batches.
    #   - If the ACCUMULATED gradient itself is non-finite (backward-side
    #     overflow with a finite forward loss), the step is dropped and the
    #     window discarded rather than written into the LoRA params — counted by
    #     train/n_nonfinite_grad_steps. Independent of k (it protects a k=1 run
    #     just the same), but note a k > 1 window discards up to k
    #     micro-batches' work when it fires.
    #   - train/n_updates counts real optimizer.step() calls (so it drops by ~k),
    #     while train/n_micro_batches counts trained mini-batches (unchanged by
    #     k). Per-micro-batch metrics (loss, clipfrac, mean_ratio, ...) divide by
    #     n_micro_batches, so they are NOT k-inflated and stay on the same scale
    #     across k — but they are not bit-identical either: within a window all k
    #     micro-batches see the same un-stepped weights, so the log-probs (and
    #     hence loss / ratio / clipfrac) shift somewhat vs a k=1 baseline. Expect
    #     a few percent of difference; that is not a regression signal.
    #     train/grad_norm_* now measures the ACCUMULATED gradient, which is
    #     expected to read lower at k > 1 (noise averaging), not "less signal".
    gradient_accumulation_steps: int = 1

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

    # Jitter-GRPO Jacobian regularizer strength, split by advantage sign:
    # jitter_pos applies to positive-advantage chunks ("good" chunks we
    # reinforce), jitter_neg to negative-advantage chunks ("bad" chunks we
    # suppress). The sign is the chunk's PRE-renormalization group-relative
    # GRPO advantage (matches the *_pos / *_neg metric split).
    #
    # When EITHER is > 0, jitter is active. The jitter_paired flag (below)
    # decides how many entries each chunk contributes; in the default paired
    # mode every action chunk produces TWO entries per epoch: a "fixed" version
    # (DiT input noise = original ε) and a "jitter" version (DiT input noise =
    # ε' = sqrt(1-λ²)*ε + λ*ξ, with fresh Gaussian ξ sampled per τ per minibatch
    # from the global torch RNG). Each jitter row uses λ = jitter_pos or
    # jitter_neg per its advantage sign. The velocity target a − ε stays at the
    # ORIGINAL ε in both branches; the cached chunk.ref_log_prob (computed at
    # original ε) is reused — the bias is O(λ²) and θ-independent, so the
    # gradient direction is unaffected. In expectation this adds a
    # Frobenius-norm Jacobian penalty (1-t)²·λ²·‖∇_x v_θ‖_F² (with the per-sign
    # λ), encouraging the velocity field to be locally smooth around each
    # rolled-out trajectory.
    #
    # In the default paired mode this doubles optimizer steps per epoch — halve
    # update_epochs MANUALLY to match the per-iter optimizer-step budget of
    # vanilla GRPO (see jitter_paired for the jitter-only alternative that keeps
    # a 1× budget). Setting only ONE side to 0 still emits that sign's jitter
    # copy, but with λ=0 it is identical to the fixed row (no Jacobian penalty
    # on that sign, just a redundant forward pass); set BOTH to 0.0 to fully
    # disable (bit-identical to vanilla GRPO). Suggested value 0.05 for each.
    jitter_pos: float = 0.0
    jitter_neg: float = 0.0

    # Jitter scheduling mode (only consulted when jitter is active, i.e.
    # jitter_pos or jitter_neg > 0; otherwise N/A).
    #   True  (default): every chunk produces BOTH a "fixed" and a "jitter"
    #         entry per epoch — 2× minibatches → 2× optimizer steps. Halve
    #         update_epochs MANUALLY to match a vanilla GRPO per-iter step
    #         budget. Keeps the fixed-vs-jitter per-branch diagnostic (the
    #         mean_log_ratio_abs gap that estimates the Jacobian norm).
    #   False: every chunk produces ONLY its "jitter" entry — 1× minibatches,
    #         so the per-iter optimizer-step count matches a vanilla run at the
    #         same update_epochs (directly comparable, no manual halving). No
    #         "fixed" rows means no `_fixed` per-branch metrics and no
    #         fixed-vs-jitter gap diagnostic; the loss is trained purely on the
    #         jittered input noise.
    jitter_paired: bool = True

    # ─── Endpoint-roughness constraint (the "jerk constraint") ───────────────
    # A temporal-smoothness prior on the DiT's IMPLIED ENDPOINT along the action
    # horizon. Orthogonal to Jitter-GRPO: jitter bounds the MAGNITUDE of the
    # velocity field's noise response (`E_xi||J xi||^2 = ||J||_F^2`, isotropic),
    # while this bounds its SPECTRUM along `h`. Measured independence:
    # `jitter/jacobian_fro_sq` fell 32% over the same iterations in which the
    # residual's high-frequency fraction rose 1.7-2.9x and relative seed
    # dispersion rose 4.5x.
    #
    # The constrained quantity is
    #     HF(a_hat(tau)) = R(a_hat) / (6 * M(a_hat).detach())
    #     a_hat(tau)     = x_tau + (1 - tau) * v_theta  ==  a + (1 - tau) * r
    # penalised as a HINGE against the pretrained field's own value:
    #     L = smooth_coef * relu( HF_pooled - smooth_hf_ref )
    #
    # Why the endpoint and not the residual: `HF(a_hat)` separates base from the
    # finetuned field by 100-200x versus the residual's 1.5-2.1x, and the two
    # measured checkpoints rank OPPOSITELY on residual vs chunk roughness.
    # Why a hinge and not a penalty: below the threshold both value and gradient
    # are exactly 0, so it never pushes toward the conditional-mean map — which
    # the `consensus_ns4` eval measures at 0.365 against baseline 0.600.
    # See scripts/grpo/README.md and jerk-constraint.md for the full derivation.
    #
    # 0.0 (default) = feature OFF, bit-identical to a run without it: no extra
    # tensors, no calibration, no metrics, no banner line. Suggested starting
    # value 0.15, which puts the term at ~15% of |clip_loss| at the roughness
    # measured on iter_0011/iter_0017. Bracket +-3x.
    smooth_coef: float = 0.0

    # Frozen SCALAR threshold. The constraint is evaluated at a single tau (=0)
    # on a dedicated clean DiT forward, so there is no per-tau vector. Three forms:
    #   None  (default) -> AUTO-CALIBRATE from the first iteration of a fresh run.
    #                      PEFT initialises lora_B to zeros, so before the first
    #                      optimizer step theta == theta_base and the collected
    #                      chunks ARE base-policy samples (confirmed: fresh runs
    #                      log ref_mse/log_base_ratio_mean == 0 exactly at iter 1,
    #                      versus 0.0572 at a resumed run's first iteration). The
    #                      measurement is taken while n_updates == 0, then scaled
    #                      by `smooth_hf_ref_scale` and frozen.
    #   float           -> flat scalar for every tau. Viable: base HF(a_hat)
    #                      spans 0.0003-0.0068 across observations and tau while
    #                      the lowest finetuned value anywhere is 0.0779, an 11x
    #                      gap, so a flat 0.02 has ~3x margin either side.
    #   list[float]     -> only element [0] is used, with a warning. Accepted
    #                      for backward compatibility with per-tau configs.
    #
    # A single-tau design, so this is a SCALAR. A list is accepted for backward
    # compatibility with the earlier per-tau build; only its first entry is used
    # and a warning is printed.
    # NEVER recomputed from the current policy. A tracking threshold would
    # re-baseline on the roughness the previous iteration introduced, permit a
    # little more, and never bind — the ratchet this design exists to avoid.
    # Persisted to `smooth_ref.json` in each checkpoint and reloaded on resume,
    # because a resumed run's first iteration is NOT base-policy.
    smooth_hf_ref: float | list[float] | None = None

    # Multiplier applied to the auto-calibrated base value. Sets how hard the
    # constraint bites; the SHAPE across tau stays the base field's. The measured
    # base weighted-mean HF(a_hat) is 0.0012-0.0051 depending on the state, so
    # 4.0 lands at ~0.005-0.02 — comfortably above base and 10-40x below the
    # finetuned field. Authority (`R(r)/R(a)`, the fraction of the chunk's D2
    # amplitude the residual can cancel) crosses 1 at roughly 10-13x base, so
    # values much above ~8 risk engaging only after full cancellation is gone.
    # Ignored when `smooth_hf_ref` is set explicitly.
    smooth_hf_ref_scale: float = 4.0

    # Admit `base_motion` into the constrained dim set. OFF by default because
    # `control_mode` gates it — arm and base are mutually exclusive under
    # robosuite's HybridMobileBase — so in arm mode it is commanded but inert,
    # and constraining an inert channel spends adapter capacity on dims that do
    # not move the robot. Discrete keys (`gripper_close`, `control_mode`) are
    # excluded unconditionally: both are 0/1 thresholded at 0.5, so a grasp IS a
    # step function and penalising its second difference would suppress grasping.
    smooth_include_base_motion: bool = False

    # Minimum number of rows the auto-calibration must pool before freezing
    # hf_ref. The pre-first-optimizer-step window is only
    # `gradient_accumulation_steps` micro-batches -- ONE at the defaults, i.e.
    # `mini_batch_size` rows -- and measured on base-like chunks an 8-row window
    # spreads the pooled base HF over 0.64x-1.57x of its large-sample value, with a
    # low-energy window (routine during a grasp or an approach pause) reading up to
    # 17x high. Since hf_ref is frozen and persisted into every checkpoint, one
    # unlucky window would silently neuter the feature for the whole run lineage.
    # Calibration therefore accumulates across the whole iteration and, if still
    # short, into later iterations, with a console line each time.
    smooth_calib_min_rows: int = 512

    # PER-CHUNK jitter-gap survey (measurement only; changes no training math).
    #
    # 0 = off (default, zero cost). N > 0 measures the jitter gap for N individual
    # chunks per iteration, stratified over 10 normalised-position bins x
    # {success, failure}, and logs the distribution plus three correlations under
    # `chunk_gap/`. See GRPOTrainer._per_chunk_gap_survey for the full rationale.
    #
    # What it buys: the per-chunk gap is a BASIN WIDTH measurement — how much the
    # FM loss rises when that chunk's noise is perturbed. Small = neighbouring
    # noise lands on the same action (robust); large = fragile. This problem gives
    # ONE BIT of reward per ~40 chunks, so a continuous per-chunk quantity is the
    # only kind of signal that can break that bottleneck, and the clean leg of the
    # measurement is already computed (MSE_ref = -chunk.ref_log_prob).
    #
    # Cost: N*K extra DiT forwards against the ref pass's n_chunks*2K. At N=256
    # that is ~5% of the ref pass's DiT work, ~12 s on a ~1700 s iteration (<1%).
    # Sampling every chunk would be ~6% of the iteration, which is why this is a
    # subsample: N=256 resolves |r| > 0.12 at 2 sigma and pins the CV to +/-4.4%
    # relative, which is all the three questions need. Values much above ~512 buy
    # precision nothing depends on.
    #
    # Reading the result: `chunk_gap/cv` is the decision statistic. A single
    # chunk's gap carries ~4-8% intrinsic noise from the xi draw, so CV at or below
    # ~8% means the between-chunk spread is pure sampling noise and no per-chunk
    # treatment can help; CV above ~15% means real structure.
    #
    # Requires jitter to be meaningful but not enabled: the survey uses a single
    # probe lambda = jitter_pos (falling back to 0.25 when jitter_pos == 0) for
    # every sampled chunk, because gap ~ lambda^2 and the production 0.25/0.05
    # sign split would otherwise make the outcome correlation measure the split
    # rather than basin width.
    per_chunk_gap_survey_size: int = 0

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
    # init_state_npz_path, or max_episode_steps — the cached
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
    # too.
    clean_output: bool = True

    def __post_init__(self):
        """Validate config invariants at construction time.

        Catches misconfigurations BEFORE the trainer spends ~1 minute on
        model load + server bind, so the operator gets immediate feedback
        instead of the "subprocess exited 1" path that would otherwise
        surface the same error several minutes in.

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
        # Gradient accumulation window size. 1 = one optimizer step per
        # mini-batch (no accumulation, bit-identical to the pre-accumulation
        # code path). Values < 1 are degenerate in two DIFFERENT ways, both
        # fatal, because `accum_count == k` is then unreachable from
        # accum_count=1 and the 1/k scale is applied literally (k=1 is the only
        # value that short-circuits the division):
        #   k == 0: `(loss / 0).backward()` makes the loss non-finite, so
        #     clip_grad_norm_ returns inf and its clip_coef of 0 turns the inf
        #     gradients into NaN — the first end-of-epoch flush writes NaN into
        #     every LoRA param and permanently poisons AdamW's moments. Loud
        #     rather than silent (grad_norm_* reads 0.0 and every later
        #     minibatch trips the non-finite guard, printing its WARNING), but
        #     the damage is already done by then.
        #   k < 0: the loss is finite and the curves look completely normal
        #     (no warning, no banner — the banner is gated on k > 1), yet the
        #     negative scale makes every flush apply the NEGATED sum of the
        #     window's gradients, i.e. gradient ASCENT. This is the genuinely
        #     silent one.
        # Fail fast so `--gradient-accumulation-steps 0` can't burn a run.
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got "
                f"{self.gradient_accumulation_steps}. 1 = one optimizer step "
                f"per mini-batch (no accumulation); k > 1 accumulates k "
                f"mini-batches per step."
            )
        for _jname, _jval in (
            ("jitter_pos", self.jitter_pos),
            ("jitter_neg", self.jitter_neg),
        ):
            if not (0.0 <= _jval < 1.0):
                raise ValueError(
                    f"{_jname} must be in [0.0, 1.0), got {_jval}. "
                    f"Variance preservation requires λ < 1; use 0.0 to disable."
                )

        # ── Endpoint-roughness constraint ────────────────────────────────────
        # smooth_coef == 0.0 is the OFF switch and must stay a total no-op, so
        # only the value range is checked unconditionally; the companion knobs
        # are validated for self-consistency either way so a typo surfaces even
        # before the feature is switched on.
        if self.smooth_coef < 0.0 or not math.isfinite(self.smooth_coef):
            raise ValueError(
                f"smooth_coef must be finite and >= 0, got {self.smooth_coef}. "
                f"It scales relu(HF - hf_ref) >= 0, so a negative value would "
                f"REWARD roughness. Use 0.0 to disable the constraint."
            )
        if self.smooth_hf_ref_scale <= 0.0 or not math.isfinite(
            self.smooth_hf_ref_scale
        ):
            raise ValueError(
                f"smooth_hf_ref_scale must be finite and > 0, got "
                f"{self.smooth_hf_ref_scale}. It multiplies the auto-calibrated "
                f"base HF, so <= 0 would put the threshold at or below zero and "
                f"pin the hinge permanently open."
            )
        if self.smooth_calib_min_rows < 1:
            raise ValueError(
                f"smooth_calib_min_rows must be >= 1, got "
                f"{self.smooth_calib_min_rows}."
            )
        if self.smooth_hf_ref is not None:
            _refs = (
                self.smooth_hf_ref
                if isinstance(self.smooth_hf_ref, (list, tuple))
                else [self.smooth_hf_ref]
            )
            if isinstance(self.smooth_hf_ref, (list, tuple)) and not _refs:
                raise ValueError(
                    "smooth_hf_ref=[] is empty. Pass a single float for the "
                    "threshold (the constraint is evaluated at one tau, so the "
                    "reference is a scalar), or None to auto-calibrate."
                )
            for _r in _refs:
                if not math.isfinite(_r) or _r <= 0.0:
                    raise ValueError(
                        f"smooth_hf_ref entries must be finite and > 0, got "
                        f"{self.smooth_hf_ref}. HF is a positive ratio (1.0 = white "
                        f"along h, 2.9839 the attainable max at H=16); 0.0 would "
                        f"pin the hinge permanently open, which is the same "
                        f"failure smooth_hf_ref_scale refuses."
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
        # Anchor groups (all-success). anchor_advantage is a POSITIVE advantage
        # applied to rows we only ever want to nudge upward; a negative value
        # would turn the retention term into active suppression of behavior that
        # demonstrably works. It is also inert unless the rows are admitted at
        # all, so require the gate explicitly rather than silently ignoring it.
        if self.anchor_advantage < 0.0:
            raise ValueError(
                f"anchor_advantage must be >= 0, got {self.anchor_advantage}. "
                f"Use 0.0 for KL-only anchoring; a negative value would "
                f"suppress trajectories that succeeded."
            )
        if self.anchor_advantage > 0.0 and not self.include_anchor_groups:
            raise ValueError(
                f"anchor_advantage={self.anchor_advantage} requires "
                f"include_anchor_groups=True — with the gate off, all-success "
                f"groups stay dead and the value is never read."
            )
        if self.anchor_max_row_frac <= 0.0:
            raise ValueError(
                f"anchor_max_row_frac must be > 0, got "
                f"{self.anchor_max_row_frac}. It caps anchor rows at this "
                f"multiple of the signal-chunk count; use "
                f"include_anchor_groups=False to disable anchors entirely."
            )
        if (
            self.anchor_max_row_frac != 1.0
            and not self.include_anchor_groups
        ):
            raise ValueError(
                f"anchor_max_row_frac={self.anchor_max_row_frac} requires "
                f"include_anchor_groups=True — with the gate off there are no "
                f"anchor rows to budget and the value is never read. (Mirrors "
                f"the same check on anchor_advantage.)"
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

        # Dynamic positive-advantage weighting bounds (only meaningful when
        # positive_advantage_weight_scaling=True, but validated unconditionally
        # so a bad value is caught even if the flag is toggled on later). k is
        # clamped to [1.0, max]; a cap <= 1 makes the weight a permanent no-op,
        # and a non-positive target ratio is degenerate.
        if self.positive_advantage_weight_max <= 1.0:
            raise ValueError(
                f"positive_advantage_weight_max must be > 1.0, got "
                f"{self.positive_advantage_weight_max}. k is clamped to [1, max]; "
                f"a cap <= 1 makes the dynamic weight a no-op."
            )
        if self.positive_advantage_weight_target_ratio <= 0.0:
            raise ValueError(
                f"positive_advantage_weight_target_ratio must be > 0, got "
                f"{self.positive_advantage_weight_target_ratio}."
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

        # Dynamic-ratio cap. Range is checked unconditionally (catches typos even
        # before the flag is flipped on); the ordering (cap >= base) is required
        # only when the dynamic mode is active, since that's when the sampler
        # clamps the natural fraction to [base, cap] — an inverted interval would
        # be degenerate.
        if not (0.0 < self.balanced_minibatch_positive_adv_ratio_max < 1.0):
            raise ValueError(
                f"balanced_minibatch_positive_adv_ratio_max must be strictly in "
                f"(0.0, 1.0), got {self.balanced_minibatch_positive_adv_ratio_max}."
            )
        if (
            self.balanced_minibatch_positive_adv_ratio_dynamic
            and self.balanced_minibatch_positive_adv_ratio_max
            < self.balanced_minibatch_positive_adv_ratio
        ):
            raise ValueError(
                f"balanced_minibatch_positive_adv_ratio_max "
                f"({self.balanced_minibatch_positive_adv_ratio_max}) must be >= "
                f"balanced_minibatch_positive_adv_ratio "
                f"({self.balanced_minibatch_positive_adv_ratio}) when "
                f"balanced_minibatch_positive_adv_ratio_dynamic=True; the dynamic "
                f"ratio is clamped to [base, max]."
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
            # resume_from=None is ALLOWED and means "fresh model, reuse
            # episode_dir/iter_0001/". That case is on-policy by construction:
            # iteration 1 collects BEFORE any optimizer.step(), so its episodes
            # were produced by the freshly-initialized policy — and setup()
            # seeds torch.manual_seed(config.seed) before LoRA injection, so a
            # fresh run with the same seed reproduces that policy bit-for-bit.
            # Typical use: a run whose UPDATE config was wrong (bad lr, bad
            # gradient_accumulation_steps) but whose collection was fine —
            # restart clean without paying the ~25 min collection again.
            #
            # The reuse is only sound if seed, model_path, and the LoRA
            # geometry (rank/alpha/target_modules) match the run that wrote
            # the cache; none of those are checked by
            # _validate_collected_data_cache, which verifies the
            # COLLECTION-side invariants (env_name, group counts, group sizes,
            # FM keys) instead.
            #
            # An empty/whitespace string is still rejected: unlike None it
            # can only arrive from an explicit `--resume-from ""`, which is a
            # quoting typo rather than a deliberate fresh start.
            if self.resume_from is not None and not self.resume_from.strip():
                raise ValueError(
                    "resume_from_collected_data=True with an empty/whitespace "
                    "resume_from. Pass a real iter_NNNN/ checkpoint path to "
                    "resume a trained policy, or OMIT --resume-from entirely "
                    "to start from a fresh model while reusing the cached "
                    "episodes at episode_dir/iter_0001/."
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
        # valid across processes: the robocasa-venv subprocess may have a
        # different CWD than the trainer.
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
            # Overwrite with the resolved absolute path so the subprocess
            # collector doesn't depend on CWD.
            self.init_state_npz_path = str(_init_path)

            # NOTE: deliberately do NOT warn on min_alive_groups>0 +
            # init_state — it's a valid choice: with all groups starting
            # from the same saved state, requiring ≥N alive groups is a
            # stability mechanism — each group draws independent denoising
            # noise, so ≥N alive (mixed) groups gives a less noisy gradient
            # direction and reduces policy-collapse risk from few-group
            # updates.

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

        # ── Frozen scene seed pool validations ───────────────────────────────
        # Every check below is a HARD error rather than a warning, because each
        # failure mode is silent at runtime: the run completes, the curves look
        # plausible, and the property the pool exists to provide is simply gone.
        #
        # The RANGE check is unconditional — exactly 0 is the "off" switch, so
        # any other value below 1 is a typo rather than a disable. A negative K
        # would otherwise sail past the `> 0` gate below and disable the feature
        # silently, and a fractional 0 < K < 1 (a float arriving from a
        # programmatic caller) would pass the gate and then build an EMPTY pool,
        # whose `% len(pool)` raises ZeroDivisionError inside the seed cursor —
        # after model load, minutes into the run.
        if self.scene_seed_pool_size != 0 and self.scene_seed_pool_size < 1:
            raise ValueError(
                f"scene_seed_pool_size must be >= 1 to enable the frozen scene "
                f"pool, or exactly 0 to disable it, got "
                f"{self.scene_seed_pool_size}. (0 is the only disabling value; "
                f"the disabled path is bit-identical to a run from before the "
                f"feature existed.)"
            )
        # The rest are gated on the feature being ON so a disabled pool stays a
        # total no-op — in particular it must not start rejecting the default
        # min_alive_groups=2 for every existing baseline run.
        if self.scene_seed_pool_size > 0:
            # Resolve the base IN PLACE (see the field comment): the trainer's
            # TB config dump reads the dataclass fields, so a lazily-resolved
            # base would leave the log saying `None` and the actual scene set
            # would be unrecoverable from the run's artifacts.
            if self.scene_seed_pool_base is None:
                self.scene_seed_pool_base = self.seed + 100_000

            # Upper bound on the pool slots a single iteration can consume, and
            # therefore the minimum viable K. It is max(num_groups, max_groups),
            # NOT num_groups: with min_alive_groups > 0 the collector may extend
            # past num_groups chasing mixed groups, up to max_groups
            # (`dynamic_mode = min_alive_groups > 0 and max_groups > num_groups`
            # in collect_episodes.py), and the trainer therefore hands it
            # max(num_groups, max_groups) consecutive slots per iteration. A
            # smaller pool would wrap inside that window.
            #
            # Consequence worth knowing: `K == num_groups` — the one setting where
            # every single iteration is directly comparable — additionally
            # requires `max_groups == num_groups`, i.e. dynamic collection off.
            # Mirrors EpisodeCollector.collect's `_max_reachable_groups`; the two
            # bounds must agree or one of them rejects a run the other accepts.
            _needed = max(self.num_groups, self.max_groups)
            if self.scene_seed_pool_size < _needed:
                raise ValueError(
                    f"scene_seed_pool_size ({self.scene_seed_pool_size}) must be "
                    f">= max(num_groups, max_groups) = {_needed} "
                    f"(num_groups={self.num_groups}, max_groups={self.max_groups}, "
                    f"min_alive_groups={self.min_alive_groups}). A single iteration "
                    f"draws its groups from consecutive pool slots — up to "
                    f"max_groups of them when dynamic collection extends past "
                    f"num_groups — so a smaller pool would wrap WITHIN one "
                    f"iteration and hand two groups the same seed, hence the same "
                    f"scene. GRPO's group-relative advantage assumes each group is "
                    f"an INDEPENDENT scene; two groups sharing one would silently "
                    f"double-count that scene in the iteration mean and correlate "
                    f"their advantages, and nothing downstream would flag it. "
                    f"Either raise K, or set max_groups == num_groups to disable "
                    f"dynamic extension."
                )
            if self.init_state_npz_path is not None:
                raise ValueError(
                    f"scene_seed_pool_size > 0 is incompatible with "
                    f"init_state_npz_path ({self.init_state_npz_path!r}). An init "
                    f"bundle overrides the scene ENTIRELY — every group restores "
                    f"the same saved model_xml + sim_state, and the reset's own "
                    f"seeded scene is immediately overwritten by "
                    f"apply_scene_bundle (see collect_episodes.py, "
                    f"_align_envs_to_group_scene). The pool would therefore be "
                    f"silently INERT: the seeds would be passed, logged, and "
                    f"plotted while having no effect on a single pixel of the "
                    f"scene. Silent inertness is exactly the failure mode worth "
                    f"erroring on. Drop one of the two flags."
                )
            # NOT an error: a pool that is not a whole number of iterations
            # long still cycles deterministically and still never repeats a seed
            # inside one iteration (that is what the K >= max_groups check
            # above guarantees). What it loses is only READABILITY — pass
            # boundaries drift relative to iteration boundaries, so there is no
            # fixed iteration stride whose mean is a full-pool average, and
            # `episode/pool_pass` increments mid-iteration-block. Warn so the
            # user knows the "average K/num_groups consecutive iterations"
            # recipe does not apply to their K.
            if self.scene_seed_pool_size % self.num_groups != 0:
                import warnings
                warnings.warn(
                    f"scene_seed_pool_size ({self.scene_seed_pool_size}) is not "
                    f"a multiple of num_groups ({self.num_groups}). The pool "
                    f"still cycles deterministically and never repeats a seed "
                    f"within an iteration, but a full pass over the pool no "
                    f"longer aligns with a whole number of iterations, so you "
                    f"cannot read a pass mean off a fixed iteration stride. Use "
                    f"a multiple of num_groups (e.g. "
                    f"{self.num_groups * max(1, round(self.scene_seed_pool_size / self.num_groups))}) "
                    f"if you want pass-aligned iteration blocks.",
                    stacklevel=3,
                )
