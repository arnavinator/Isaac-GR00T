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
from smoothness import SMOOTH_INSTRUMENTS


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
    episode_dirs_to_keep: int = 10

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

    # PER-ROW, MSE-REFERENCED lower clip. 0.0 (default) = OFF and bit-identical
    # to a flat `1 - clip_eps_low` floor on every row.
    #
    # WHY. The importance ratio is rho = exp(MSE_ref - MSE_theta), so
    # `clip_eps_low` is specified in LOG-RATIO (nat) units while the quantity
    # that actually diverges is MSE_theta. A flat epsilon therefore grants wildly
    # non-uniform MSE headroom WITHIN ONE ITERATION: at clip_eps_low=0.08 the
    # allowed MSE inflation measured 261x at `ref_mse/p10` and 2.1x at
    # `ref_mse/max`. When > 0 each row instead gets a budget proportional to its
    # own MSE_ref:
    #
    #     budget_i    = min(clip_low_mse_coef * MSE_ref_i,
    #                       |ln(1 - clip_eps_low)|)          # nats
    #     rho_floor_i = exp(-budget_i)
    #
    # so every row is allowed the same RELATIVE inflation, 1 + clip_low_mse_coef,
    # until the ceiling binds.
    #
    # `clip_eps_low` stays an ABSOLUTE CEILING on the budget (the min above), so
    # this mechanism can only ever be TIGHTER than a flat clip, never looser.
    # That is deliberate: MSE_ref GROWS as the field degrades (measured
    # 0.0023 -> 0.0297 over one run), so an uncapped c * MSE_ref budget would
    # WIDEN the clip exactly when it needs to tighten.
    #
    # Pairs with paws_k_floor_at_target: a tighter lower clip kills more negative
    # rows, which shrinks PAWS's alive-erosion mass N and hence LOWERS k — see
    # that field.
    clip_low_mse_coef: float = 0.0

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

    # Floor the MEASURED k at positive_advantage_weight_target_ratio instead of
    # at 1.0. False (default) = today's behaviour.
    #
    # WHY. The measured branch is
    # `k = clamp(tratio * N/D, floor, max)` with N = alive erosion mass and
    # D = alive amplified-positive mass. `clip_low_mse_coef > 0` deliberately
    # kills MORE negative rows, which shrinks N, which under the current
    # controller LOWERS k — so tightening the erosion brake would also weaken
    # reinforcement, the opposite of the intent. Flooring at tratio removes only
    # the "amplify LESS than target" case; it never amplifies more.
    #
    # Measured: N/D sits at 1.04-1.06 in healthy iterations, so the floor is
    # INERT there (k = tratio * 1.05 > tratio); it falls to 0.66 during collapse,
    # which is exactly where it binds. Only consulted when the scaling flag is
    # True. Requires target_ratio >= 1.0 (validated): flooring below 1.0 would
    # pin k under the no-op point and invert the mechanism's intent.
    paws_k_floor_at_target: bool = False

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

    # ─── Trajectory-roughness constraint (the "jerk constraint") ─────────────
    # A temporal-smoothness prior on the DiT's generated action chunk along the
    # action horizon. Orthogonal to Jitter-GRPO: jitter bounds the MAGNITUDE of
    # the velocity field's noise response (`E_xi||J xi||^2 = ||J||_F^2`,
    # isotropic), while this bounds its SPECTRUM along `h`. Measured
    # independence: `jitter/jacobian_fro_sq` fell 32% over the same iterations in
    # which the residual's high-frequency fraction rose 1.7-2.9x and relative
    # seed dispersion rose 4.5x.
    #
    # The constrained quantity is
    #     HF(u) = R(u) / (6 * M(u).detach())        D2 along h
    # penalised as a HINGE against the pretrained field's own value:
    #     L = smooth_coef * relu( HF_pooled - smooth_hf_ref )
    # where `u` is the trajectory named by `smooth_instrument` (default: the
    # 4-step generated chunk).
    # Why a hinge and not a penalty: below the threshold both value and gradient
    # are exactly 0, so it never pushes toward the conditional-mean map — which
    # the `consensus_ns4` eval measures at 0.365 against baseline 0.600.
    # See scripts/grpo/README.md and jerk-constraint.md for the full derivation.
    #
    # 0.0 (default) = feature OFF, bit-identical to a run without it: no extra
    # tensors, no calibration, no metrics, no banner line. Suggested starting
    # range for the CHUNK instrument is 0.15-0.5: the last-step-differentiable
    # rollout retains roughly a quarter of the true 4-step gradient's magnitude
    # (one of four steps carries a graph), so the coefficient may need raising
    # relative to the 0.15 that was calibrated on the endpoint instrument, whose
    # single forward IS its whole gradient. Bracket +-3x either way.
    smooth_coef: float = 0.0

    # WHICH trajectory the (1,-2,1) second-difference operator is applied to.
    # Exactly two values; anything else is a hard config error.
    #
    #   "chunk" (default) -- the 4-step GENERATED chunk, i.e. what the robot
    #       executes. Rolled out on the production sampler schedule from the
    #       collected eps, with only the LAST Euler step differentiable: the
    #       forward VALUE is the exact 4-step chunk, while only 1 graph-forward
    #       is added so VRAM (and hence mini_batch_size=8) is unchanged. The
    #       gradient is biased -- it misses how theta shapes the earlier steps
    #       and the sampler path -- which is an accepted, documented tradeoff
    #       against the ~29.1 GB a fully-differentiated rollout needs versus the
    #       25.3 GB available.
    #
    #   "endpoint" -- the historical instrument: the 1-step implied endpoint
    #       `a_hat(0) = eps + v_theta(eps, 0)` on a dedicated clean forward at
    #       tau = 0. Reproduces the pre-change behaviour bit-for-bit, so runs
    #       calibrated against it stay reproducible.
    #
    # Why the default is "chunk". An empirical sweep over 16 checkpoints of a
    # real training run showed the endpoint does NOT control physical trajectory
    # jerk, which is the quantity the constraint exists to bound:
    #   * Over iterations 10-16 of the unconstrained run the endpoint HF FELL 9%
    #     (0.331 -> 0.300) while EEF path jerk ROSE 11% (0.516 -> 0.572).
    #     Spearman rho between them over that window: +0.00.
    #   * A run WITH the constraint at coef 0.15 pinned endpoint HF at 3-6x base
    #     for six iterations, yet its executed chunks still degraded: chunk HF
    #     2.2x -> 8.6x base, path jerk 1.45x -> 2.86x base.
    # The 4-step chunk's HF, by contrast, correlates with path jerk at
    # rho = +0.98 overall and +0.96 over the late iterations.
    smooth_instrument: str = "chunk"

    # Frozen SCALAR threshold on the pooled HF of whichever instrument
    # `smooth_instrument` selects. One number, not a per-tau vector: both
    # instruments reduce to a single pooled scalar per minibatch. Three forms:
    #   None  (default) -> AUTO-CALIBRATE from the first iteration of a fresh run.
    #                      PEFT initialises lora_B to zeros, so before the first
    #                      optimizer step theta == theta_base and the collected
    #                      chunks ARE base-policy samples (confirmed: fresh runs
    #                      log ref_mse/log_base_ratio_mean == 0 exactly at iter 1,
    #                      versus 0.0572 at a resumed run's first iteration). The
    #                      measurement is taken while n_updates == 0, then scaled
    #                      by `smooth_hf_ref_scale` and frozen.
    #   float           -> flat scalar threshold. Must be in the units of the
    #                      SELECTED instrument -- the two have similar base
    #                      values (chunk 0.00141, endpoint 0.00157) but very
    #                      different useful ranges, so a value carried over from
    #                      the other instrument will mis-bind. See
    #                      `smooth_hf_ref_scale` for measured levels.
    #   list[float]     -> only element [0] is used, with a warning. Accepted
    #                      for backward compatibility with per-tau configs.
    #
    # A single-scalar design. A list is accepted for backward compatibility with
    # the earlier per-tau build; only its first entry is used and a warning is
    # printed.
    # NEVER recomputed from the current policy. A tracking threshold would
    # re-baseline on the roughness the previous iteration introduced, permit a
    # little more, and never bind — the ratchet this design exists to avoid.
    # Persisted to `smooth_ref.json` in each checkpoint and reloaded on resume,
    # because a resumed run's first iteration is NOT base-policy.
    smooth_hf_ref: float | list[float] | None = None

    # Multiplier applied to the auto-calibrated base value. Sets how hard the
    # constraint bites. Ignored when `smooth_hf_ref` is set explicitly.
    #
    # The scale is a MULTIPLE OF THE MEASURED BASE VALUE, and the two instruments
    # have completely different useful ranges even though their base values are
    # similar. Which is why this default moved 4.0 -> 15.0 when the instrument
    # moved endpoint -> chunk: 4.0 was calibrated for the endpoint and is far too
    # tight for the chunk.
    #
    # Measured chunk HF on the control run (base = 0.00141):
    #     iter1 0.0023  iter2 0.0072  iter3 0.0152  iter4 0.0244
    #     iter6 0.0408  iter12 0.0959  iter16 0.1131          -> ~80x base
    # Corresponding EEF path jerk: base 0.0689, iter4 0.2720, iter12 0.5358.
    # A bound near 15-17x base (~0.024) therefore targets the iteration-4
    # roughness level, which carries ~2x LESS path jerk than the control run's
    # peak-success iteration -- a real reduction that is still reachable rather
    # than a threshold the field can never get back under.
    #
    # For `smooth_instrument="endpoint"`, 4.0 remains the right value: the
    # measured base endpoint HF is 0.0012-0.0051 depending on state, so 4.0 lands
    # at ~0.005-0.02, comfortably above base and 10-40x below the finetuned
    # field, and the endpoint's own authority (`R(r)/R(a)`) crosses 1 at roughly
    # 10-13x base.
    smooth_hf_ref_scale: float = 15.0

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

    # ─── Weight-step direction cosines (`lora/cos_step_*`) ────────────────────
    # Always emitted; these two knobs only choose the FROZEN EARLY reference
    # `L_early` that `lora/cos_step_early` is measured against.
    #
    # WHY a frozen reference. Measured across 6 runs, cos(step, L_early) has a
    # minimum of -0.058 over 41 updates on the runs that stayed healthy and
    # reaches -0.49 / -0.62 on the two that collapsed directionally. The
    # `cos_step_cumulative` variant is self-referential — once a run turns,
    # `W_prev - W_init` turns with it — and measured POORLY: it stayed at
    # +0.37..+0.53 straight through a collapse. Read `cos_step_early`.
    #
    # cos_ref_lora_paths: (path_a, path_b), each an existing `iter_NNNN/` LoRA
    #   checkpoint dir or a `lora_weights.pt` file. L_early = W(b) - W(a), loaded
    #   ONCE at setup. Use this to score a NEW run against a KNOWN-GOOD run's
    #   early direction; without it the reference is this run's own early motion,
    #   which cannot detect a run that turned before it was frozen.
    # cos_ref_iterations: when cos_ref_lora_paths is None, freeze
    #   L_early = W_now - W_init after this many LOGGED (non-zero-step)
    #   iterations. Must be >= 1.
    cos_ref_lora_paths: Optional[tuple[str, str]] = None
    cos_ref_iterations: int = 2

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

        # ── Trajectory-roughness constraint ──────────────────────────────────
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
        # Validated unconditionally (even at smooth_coef == 0) for the same
        # reason as the knobs below: a typo must surface at construction, not
        # the first time someone switches the feature on. The two instruments
        # measure DIFFERENT quantities with similar base values (chunk 0.00141
        # vs endpoint 0.00157), so a silent fall-through to a default would be
        # invisible in the numbers.
        if self.smooth_instrument not in SMOOTH_INSTRUMENTS:
            raise ValueError(
                f"smooth_instrument must be one of "
                f"{sorted(SMOOTH_INSTRUMENTS)}, got "
                f"{self.smooth_instrument!r}. 'chunk' constrains the 4-step "
                f"generated chunk (what the robot executes; correlates with EEF "
                f"path jerk at rho=+0.98); 'endpoint' constrains the 1-step "
                f"implied endpoint a_hat(0) = eps + v(eps, 0) and reproduces the "
                f"pre-change behaviour (rho=+0.00 against path jerk over the "
                f"late iterations, which is why it is no longer the default)."
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
                    "threshold (both instruments reduce to one pooled scalar per "
                    "minibatch, so the reference is a scalar), or None to "
                    "auto-calibrate."
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

        # Per-row MSE-referenced lower clip. Validated unconditionally (the same
        # posture as the two epsilons above) so a typo is caught even in a run
        # that leaves the mechanism off. 0.0 IS the off switch, so only negative
        # and non-finite values are rejected.
        if self.clip_low_mse_coef < 0.0 or not math.isfinite(self.clip_low_mse_coef):
            raise ValueError(
                f"clip_low_mse_coef must be finite and >= 0, got "
                f"{self.clip_low_mse_coef}. 0.0 is the OFF switch (flat "
                f"1 - clip_eps_low floor on every row); a positive value gives "
                f"each row a budget of min(coef * MSE_ref_i, "
                f"|ln(1 - clip_eps_low)|) nats. A negative coefficient would put "
                f"the floor ABOVE 1.0, clipping every row that did not move."
            )

        # SILENT-INERTNESS TRAP. `clip_eps_low` is BOTH the off-path floor and the
        # absolute ceiling on the per-row budget, so setting it to the value you
        # actually want as a budget makes the coefficient unreachable: every row
        # pins to the ceiling and the mechanism silently reverts to the flat clip
        # it was added to replace. The run completes, the curves look plausible,
        # and `drift/neg_frac_over_budget` reports against the flat budget, so
        # nothing downstream flags it. Costs a full arm to discover.
        #
        # Probe value is the measured HEALTHY-REGIME per-iteration `ref_mse/mean`
        # (0.0040-0.0050 over iterations 3-9 of the reference run; 0.00398 at the
        # iteration the collapse ignited). The per-row mechanism only does work
        # where `coef * MSE_ref < ceiling`, so if the ceiling already binds at the
        # healthy operating point it binds for essentially the whole run. Compare
        # against train_grpo.MSE_REF_BANNER_PROBES, which brackets the full
        # early-to-degraded range (0.0023-0.0297); this is a point inside it, not
        # its low end, because the low end is only touched at iteration 1.
        # Duplicated rather than imported: train_grpo imports THIS module, so
        # importing back would be circular. Keep them consistent.
        #
        # Correct usage: leave clip_eps_low LOOSE (its current/default value) and
        # let the coefficient set the budget; use clip_low_mse_coef=0 with a tight
        # clip_eps_low for a FLAT-budget arm. Do not tighten both at once.
        if self.clip_low_mse_coef > 0.0:
            _ceil_nats = -math.log(max(1.0 - self.clip_eps_low, 1e-12))
            _binds_at = _ceil_nats / self.clip_low_mse_coef
            if _binds_at <= 0.005:
                import warnings
                warnings.warn(
                    f"clip_low_mse_coef={self.clip_low_mse_coef:g} is effectively "
                    f"INERT at clip_eps_low={self.clip_eps_low:g}: the ceiling "
                    f"|ln(1-clip_eps_low)|={_ceil_nats:.5f} nats binds for every "
                    f"row with MSE_ref >= {_binds_at:.5f}, at or below the top of "
                    f"the measured healthy-regime ref_mse/mean (~0.004-0.005). "
                    f"Nearly every row will "
                    f"pin to the ceiling and the per-row mechanism reduces to the "
                    f"flat clip it replaces. Either raise clip_eps_low (leave it at "
                    f"its default and let the coefficient set the budget) or set "
                    f"clip_low_mse_coef=0.0 for a deliberate flat-budget arm.",
                    stacklevel=2,
                )

            # BORN-DEAD TRAP. `budget_i` is measured from rho = 1, but with
            # jitter active a NEGATIVE-advantage row is not born there: its DiT
            # input is eps' rather than eps while the velocity target stays at
            # a - eps, so its MSE_theta carries an offset
            #     gap_neg = E_k[(1-tau)^2] * jitter_neg^2 * ||J||_F^2 / D
            # from step 0. If gap_neg exceeds the row's whole budget the row is
            # born BELOW its own floor and its erosion gradient is dead on
            # arrival — the per-row clip silently becomes an erosion ABLATION
            # rather than a cap, and it does so most on the rows the reference
            # fits BEST (smallest MSE_ref, hence smallest budget).
            #
            # Estimate: gap_neg ~= 0.9 * jitter_neg^2, calibrated against the
            # measured 0.0016-0.0026 at jitter_neg=0.05 with the shipped
            # tau_centers and jacobian_fro_sq ~ 2. It scales as lambda^2, so
            # halving jitter_neg cuts it 4x. `jitter/gap_neg` logs the real value
            # per iteration; this is only for a config-time sanity check.
            #
            # Compared against `coef * 0.004` (the measured healthy-regime
            # ref_mse/mean), i.e. the budget a TYPICAL row gets. The pre-existing
            # `jitter/neg_clip_budget_used` is the same quantity for the FLAT clip,
            # where its documented wall is ~0.30.
            _gap_est = 0.9 * self.jitter_neg * self.jitter_neg
            _typ_budget = self.clip_low_mse_coef * 0.004
            if _gap_est > 0.5 * _typ_budget:
                import warnings
                warnings.warn(
                    f"jitter_neg={self.jitter_neg:g} implies gap_neg ~= "
                    f"{_gap_est:.5f} nats, against a typical per-row budget of "
                    f"clip_low_mse_coef * ref_mse/mean ~= {_typ_budget:.5f} nats "
                    f"({100 * _gap_est / max(_typ_budget, 1e-12):.0f}% of it). "
                    f"Negative rows are born at rho=exp(-gap_neg), so rows with "
                    f"MSE_ref < {_gap_est / self.clip_low_mse_coef:.5f} start "
                    f"BELOW their own floor and are clip-dead on arrival: the "
                    f"per-row clip becomes an erosion ablation, not a cap, biased "
                    f"toward the rows the reference fits best. Either raise "
                    f"clip_low_mse_coef, or lower jitter_neg (gap_neg scales as "
                    f"jitter_neg^2; 0.0 makes negative rows born at exactly "
                    f"rho=1). Watch drift/neg_frac_born_dead, which is scoped to the "
                    f"pre-step micro-batches where this is measurable — it "
                    f"reads ~1.0 when this has bitten.",
                    stacklevel=2,
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
        # Flooring the measured k at target_ratio only makes sense ABOVE the
        # no-op point. Below 1.0 the floor would PIN k under 1.0, i.e. force the
        # mechanism to de-amplify reinforcement on every iteration whose measured
        # N/D is healthy — the exact inversion of what the flag exists for.
        # Ordering: checked after the > 0 test so a negative target reports the
        # simpler error first.
        if (
            self.paws_k_floor_at_target
            and self.positive_advantage_weight_target_ratio < 1.0
        ):
            raise ValueError(
                f"paws_k_floor_at_target=True requires "
                f"positive_advantage_weight_target_ratio >= 1.0, got "
                f"{self.positive_advantage_weight_target_ratio}. The flag replaces "
                f"the measured k's lower clamp of 1.0 with target_ratio; a target "
                f"below 1.0 would floor k BELOW the no-op point and force "
                f"de-amplification of reinforcement, inverting the intent. Either "
                f"raise the target ratio or leave paws_k_floor_at_target=False."
            )

        # ...and the floor must not sit ABOVE the cap. `k = min(max(measured,
        # floor), max)`, so target_ratio > positive_advantage_weight_max collapses
        # the expression to the constant `max` for EVERY measurement: the
        # measurement-driven controller silently becomes a fixed amplifier, and the
        # banner prints an inverted interval ("clamped to [5, 2]") as if it were a
        # range. Only reachable with the flag on — without it the floor is 1.0,
        # which the `max > 1.0` check above already keeps below the cap.
        if (
            self.paws_k_floor_at_target
            and self.positive_advantage_weight_target_ratio
            >= self.positive_advantage_weight_max
        ):
            raise ValueError(
                f"paws_k_floor_at_target=True requires "
                f"positive_advantage_weight_target_ratio "
                f"({self.positive_advantage_weight_target_ratio}) < "
                f"positive_advantage_weight_max "
                f"({self.positive_advantage_weight_max}). k is computed as "
                f"min(max(measured, target_ratio), max); with the target at or above the "
                f"cap that is the constant `max` for every measurement, which "
                f"turns PAWS from a controller into a fixed amplifier without "
                f"warning. Raise positive_advantage_weight_max or lower the target."
            )

        # Weight-step direction cosines. Both knobs are validated
        # unconditionally: the cosines are always emitted, so a bad value here is
        # never dormant.
        if self.cos_ref_iterations < 1:
            raise ValueError(
                f"cos_ref_iterations must be >= 1, got "
                f"{self.cos_ref_iterations}. It is the number of LOGGED "
                f"iterations after which L_early = W_now - W_init is frozen as "
                f"the lora/cos_step_early reference; 0 would freeze a zero "
                f"reference vector, whose cosine is undefined."
            )
        if self.cos_ref_lora_paths is not None:
            _paths = self.cos_ref_lora_paths
            # `isinstance(..., (list, tuple))` rather than `len(tuple(_paths))`:
            # the latter raises a bare TypeError on a non-iterable (an int from a
            # config file, say) instead of this method's ValueError, and it would
            # silently ACCEPT a 2-character string as a 2-tuple of characters.
            if not isinstance(_paths, (list, tuple)) or len(_paths) != 2:
                raise ValueError(
                    f"cos_ref_lora_paths must be a 2-tuple (path_a, path_b), got "
                    f"{self.cos_ref_lora_paths!r}. L_early = W(path_b) - "
                    f"W(path_a), so exactly two checkpoints are required."
                )
            # Normalise to a tuple so the TB config dump and the setup-time load
            # see one shape regardless of how tyro/YAML delivered it (list vs
            # tuple). Done before the existence check so the error message quotes
            # the resolved value.
            self.cos_ref_lora_paths = (str(_paths[0]), str(_paths[1]))
            # Local alias, NOT a bare `from pathlib import Path`: two later
            # blocks in this same method already bind `Path` / `_Path` locally,
            # and a function-local `import ... as Path` makes that name local for
            # the WHOLE function body — so using the plain name here would raise
            # UnboundLocalError before those imports execute.
            from pathlib import Path as _CosPath
            for _p in self.cos_ref_lora_paths:
                _pp = _CosPath(_p)
                if not _pp.exists():
                    raise ValueError(
                        f"cos_ref_lora_paths entry {_p!r} does not exist. Pass "
                        f"existing iter_NNNN/ checkpoint directories (or "
                        f"lora_weights.pt files) — the reference direction is "
                        f"loaded ONCE at setup, so a bad path must fail before "
                        f"the multi-minute model load rather than hours in."
                    )
                # Existence alone is not enough to keep that promise: a directory
                # without a lora_weights.pt (or a plain wrong directory such as
                # /tmp) validated fine here and then failed inside
                # _load_cos_ref_direction, which runs AFTER the model load.
                if _pp.is_dir() and not (_pp / "lora_weights.pt").exists():
                    raise ValueError(
                        f"cos_ref_lora_paths entry {_p!r} is a directory with no "
                        f"lora_weights.pt in it. Point at an iter_NNNN/ checkpoint "
                        f"directory (which contains lora_weights.pt) or at the "
                        f".pt file itself; the reference is loaded once at setup, "
                        f"after the multi-minute model load."
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
