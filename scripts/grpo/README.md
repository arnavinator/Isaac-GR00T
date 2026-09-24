# GRPO Finetuning for GR00T N1.6

Group Relative Policy Optimization (GRPO) with LoRA adapters for online RL
finetuning of the GR00T N1.6 DiT action head on RoboCasa manipulation tasks.

The pretrained GR00T model is trained with offline imitation learning. This
package layers an episodic RL loop on top: collect group rollouts in
simulation, compute group-relative advantages on the resulting rewards, and
update only LoRA adapters on the DiT via a clipped surrogate objective with a
Flow-Matching (FM) log-probability surrogate.

---

## Contents

| File | Purpose |
|------|---------|
| `train_grpo.py` | Main orchestrator (`GRPOTrainer`): model+LoRA setup, iter loop, ref log-probs, GRPO update, checkpointing. |
| `grpo_config.py` | `GRPOConfig` dataclass — every tunable knob lives here. |
| `grpo_server.py` | Extends `PolicyServer` to capture per-call denoising noise + raw `(B, 50, 128)` action. Required for FM log-prob. |
| `collect_episodes.py` | Runs in the robocasa venv. `EpisodeCollector` does group rollouts via `AsyncVectorEnv`, including fast-forward branching and scene-bundle alignment. |
| `episode_buffer.py` | `EpisodeBuffer`, `GRPOEpisode`, `ActionChunk`. Loads `.npz` episodes, computes group-relative advantages. |
| `fm_log_prob.py` | FM-loss-as-log-prob surrogate (`compute_fm_log_prob`), jittered timestep sampler (`_sample_jittered_timesteps`), production sampler schedule (`inference_schedule`) and the last-step-differentiable chunk rollout (`_smooth_chunk_rollout`) the roughness constraint measures. |
| `lora_dit.py` | `apply_lora_to_dit`, `save_lora_checkpoint`, `load_lora_checkpoint`, default target-module list. |
| `smoothness.py` | Trajectory-roughness ("jerk") constraint primitives: `second_difference`, `roughness_moments`, `pooled_hf`, `roughness_hf`, the continuous-action-dim selector and `build_key_dim_span`. Model-free and fully unit-testable. The 4-step chunk rollout lives in `fm_log_prob._smooth_chunk_rollout` (it needs the DiT); the hinge lives in `train_grpo._grpo_update_inner`. |
| `gripper_release.py` | Post-reopen truncation primitives: `PostReopenFilter`, `gripper_widths`, `close_cross_indices`, `reopen_onset_index`, `post_reopen_detect`. Model-free and sim-free; detects the close→reopen of a failed grasp from the measured `gripper_qpos` — and the ONSET of that reopen, by walking back down the rising edge — so `episode_buffer` can trim the meander that follows. See README "Post-reopen truncation". |
| `eval_lora_from_npz.py` | Eval harness: runs N parallel rollouts of a LoRA policy from a saved `interactive_rollout.py` `.npz`, aggregates per-attempt success/num_steps into `results.json`. Subclasses `EpisodeCollector` in init-state mode. |
| `test_*.py` | Sanity checks for sim-wrapper / `.npz` key roundtrip. `test_grad_accum.py` drives the real `_grpo_update_inner` on CPU to pin the gradient-accumulation semantics and the PAWS mass accounting / cold start. `test_jitter_metrics.py` does the same for the `jitter/*` / `ref_mse/*` / sign-split / effective-clipfrac instrumentation. `test_anchor_groups.py` does the same for anchor groups (classification, row budget, renorm isolation, sampler/PAWS/epoch exclusions). |
| `verify_multiturn_gpu.py` | Real-stack check for multi-turn collection / branch-point integrity. Run on the GPU VM in the robocasa venv. |
| `test_video_key_filter.py` | Covers the unused-video-key filter (`dropped_video_keys`). |
| `test_smoothness.py` | CPU suite for the trajectory-roughness constraint: HF calibration, the `a_hat = a + (1−τ)r` identity, hinge semantics, dim/horizon selection, the derived sampler schedule, the last-step-differentiable rollout (exact value + gradient localized to the final step), the `compute_fm_log_prob` return contract, both instruments' jitter-invariance, the executed-chunk metrics, `smooth_ref.json` guard rejection **including instrument mismatch**, and `smooth_coef=0` bit-identity (stats, weights and RNG stream) through the real `_grpo_update_inner`. |
| `verify_render_skip_gpu.py` | Real-stack check for `skip_intermediate_render`: proves the kept frame is byte-identical to the unskipped path against real MuJoCo/EGL rendering, and reports the render count + speedup. Robocasa venv, no model server. |
| `test_scene_seed_pool.py` | CPU suite for the frozen scene seed pool: base resolution, the stateless cursor + pass alignment, within-iteration seed distinctness (including a non-divisible K), all four config validations plus the pass-alignment warning, `GROUP_SEED_STRIDE` agreement between the two files, byte-identity of the disabled collector argv, the real `EpisodeCollector.collect` consuming `--group-seeds` (and refusing to wrap), and `per_scene_success` → `episode/scene_sr/*` emission through the real `_log_metrics`. |
| `test_gripper_release.py` | CPU suite for the post-reopen truncation: width extraction (state horizon, flat shape, custom key, eight guards incl. NaN/inf, (0,2), (1,2,2) and non-numeric); the hysteresis state machine on both edges incl. strictness at each threshold, the in-band no-op, the never-closed and never-reopened cases and the first-cycle-only rule; the ONSET walk — rising-edge base, floor-relative margin, the `close_idx < onset <= cross_idx` invariant over 3k+ exhaustive traces, mid-hold blip rejection (`episode_0039` / `episode_0044` shapes), the monotonicity guard against a high plateau touching the crossing, and invariance over the threshold box and the margin plateau; `keep_chunks` counting from the onset and overruns being no-ops; the sign-convention / unit-mismatch guard; the `PostReopenFilter` and `GRPOConfig` validation matrices (incl. the four tuning knobs being a hard error while the feature is off, and `onset_margin ≥ open_width − close_width`); and, through the real `compute_advantages` / `_build_chunks`, off-switch bit-identity, successes and anchors left whole, `Σ A_chunk == A_ep` plus the group zero-sum surviving, the per-row magnification being exactly `num_chunks / num_train_chunks`, idempotence + self-clearing (no stale chunk memo), dead groups staying dead, and the anchor row budget seeing truncated signal counts. The `ITER_0001` / `ITER_0001_CROSS` fixtures pin all 48 real episodes' onset AND crossing (so a regression collapsing the two is caught by name) plus the N → dropped-fraction curve; when that collection is on disk the suite re-derives both from the raw `.npz` and re-runs the 19-threshold-pair and margin-plateau invariance on real widths. |
| `test_clip_floor.py` | CPU suite for the per-row MSE-referenced lower clip (`clip_low_mse_coef`), the PAWS `k` floor (`paws_k_floor_at_target`) and the three added diagnostics: off-switch determinism + additivity (the bit-identity-vs-baseline check is an out-of-tree differential, recipe in that test's docstring), the `rho_floor` arithmetic incl. the binding `clip_eps_low` ceiling, agreement of **all six** lower-bound consumers on rows straddling their own floors, positive/anchor-row inertness against the four-case table, both `k` floors and both untouched `k` branches, monotonicity in the coefficient, hand-computed `drift/*` values, `jitter/pos_clip_budget_used`, and the `lora/cos_step_*` cosines incl. the sign flip and the two `L_early` sources. |
| `test_kl_base_adaptive.py` | CPU suite for the closed-loop base-model trust region (`kl_base_adaptive`): off-switch (no emission, no state touched), deadband semantics on both edges, clamps, the relax floor at the starting coefficient, effect-based `action` on both branches, relax pacing (exactly one move per `patience`, never compounding), a missing/NaN reading holding rather than relaxing, the authority linearisation, a replay against runB's **full** it1-14 archive drift series (never truncate that fixture — a shorter window hid a self-disarm bug), the shipped defaults, and the full validation matrix incl. non-finite and bool knobs, the save/refresh paths, the four-case resume matrix, and a source-level wiring check for `setup()` (unreachable from a `__new__` harness). |
| `test_grad_probe.py` | CPU suite for the gradient-decomposition probe (`grad_probe_every`), driving the real `_grpo_update_inner` plus the real `_grad_probe_capture_jittered` / `_grad_probe_finish` / `select_grad_probe_rows` / `aggregate_grad_probes`. Covers: off-switch bit-identity (stats, `p.grad`, weights, RNG stream, and a spy proving `torch.autograd.grad` is never called); the probe ON changing **nothing** about the training step (`autograd.grad` really does not accumulate); the decomposition identity `g_jit − g_R = λ²g_P` against a **hand-derived** closed form on a τ- and `noise_for_input`-sensitive analytic stand-in with a known Jacobian (a re-run of autograd would have agreed with a wrong derivation — this caught a missing `w0` factor); `R → 0` as `λ → 0` and monotonicity in `λ`; both legs sharing ε / τ / rows verbatim, with mutants that mismatch the τ **set** and the row **set** and must be detected; the τ-subset path applied to both legs; the row cap and deterministic tie-breaking; the paired-mode fixed-row exclusion pinned against the ~2× diluted alternative; the `jitter_neg > 0` guard on `g_erosion`; hand-computed percentiles/aggregation; skip, failure and cadence accounting; both legs running at the **same θ** at `gradient_accumulation_steps` 1 and 2; the four-way return unpack (roughness constraint × probe); TB emission incl. the `vram/` split and the non-finite drop; and the full config validation matrix. |
| `test_adam_knobs.py` | CPU suite for the surfaced AdamW knobs (`adam_beta1`, `adam_beta2`, `adam_eps`). Covers: default bit-identity with the previously hard-coded `(0.9, 0.999)` / `1e-5`, asserted on a real `optim.AdamW` `param_group` rather than the dataclass; the validation matrix including the two values that otherwise train silently and wrongly (`beta1 == 1.0` → zero step forever, `beta2 == 1.0` → denominator collapses to `adam_eps`) and `beta == 0.0` which must be **accepted** as the momentum-off ablation; both edges of the eps/beta2 regime warning incl. the strict `< 1e-6` boundary and that it warns rather than raises; a source-level wiring check on `setup()` (unreachable from a `__new__` harness) plus that exactly one `optim.AdamW(` remains; and the two arithmetic claims the `adam_eps` doc rests on, measured against a real AdamW step — a 3.8× gradient gives a >3× step at `eps=1e-5` but <1.1× at `eps=1e-8`, and the eps drop inflates the step ~10×. |
| `calibrate_vel_anchor.py` | Step 1 of the velocity-anchor experiment: one update-only trial of `train_grpo.py` per `vel_anchor_coef` on a cached iteration, then the first update's shrink / cosine against coef 0 (exact rank-2r trace trick, no dense ΔW), guards, and log-interpolated sweep suggestions. See README "Velocity anchor". |
| `test_vel_anchor.py` | CPU suite for `vel_anchor_coef`: the REAL `compute_fm_log_prob` on a stub head holding a real PEFT LoRA layer (off-path contract, D == 0 at the anchor, D against an independent LoRA-formula hand computation for both anchor kinds and jittered inputs, identical current/anchor inputs, finite-difference gradient, adapters restored after an exception, `functional_call` leaving the live model and saved checkpoint untouched, masking, split parts, no RNG, every struct combination), then the real `_grpo_update_inner` / ref pass / jitter diagnostics / `_setup_vel_anchor` / `_log_metrics` / `train()`: loss composition incl. the anchor-row divisor, coef-0 identity, accumulation, the non-finite guard, the force-balance probe against direct gradients with the step bit-identical, every metric, `jac_part_pos`, anchor loading, the first-micro-batch check, config validation, resume, and `stop_after_iterations`. |
| `test_calibrate_vel_anchor.py` | CPU suite for `calibrate_vel_anchor.py`: trace-trick norm/cosine vs dense (fresh and relative to a start), interpolation incl. the non-monotone warning and brackets, `--dry-run` command construction parsed back through tyro into `GRPOConfig`, and the cached-episode guard on synthetic TB event files. |

---

## Architecture

Two processes share the work:

```
┌─────────────────────────┐   ZMQ obs/action    ┌──────────────────────────┐
│ Trainer (main .venv)    │ ◄──────────────────►│ Collector (robocasa venv)│
│  GPU model + LoRA       │  port 5555          │  AsyncVectorEnv workers  │
│  In-process PolicyServer│                     │  Writes .npz per iter    │
└─────────────────────────┘                     └──────────────────────────┘
```

Each iteration the trainer spawns one `collect_episodes.py` subprocess in the
robocasa venv, which connects back to the in-process policy server over ZMQ,
runs the group rollouts, and writes the episodes as `.npz` files.

The trainer spawns the policy server **in a background thread** of its own
process, so the LoRA weights it updates are immediately visible to the next
collection round — no checkpoint shuffling. A re-entrant lock
(`self._model_lock`) serializes forward and backward passes between the
server thread (inference for the collector) and the main thread (ref
log-probs / GRPO update).

The collector runs in a separate venv because robocasa depends on MuJoCo
and gym wrappers that don't coexist cleanly with the main training stack.

---

## Quick Start

### 1. Run training

Run the trainer; it spawns one `collect_episodes.py` subprocess per iteration
(in the robocasa venv) to collect the group rollouts:

```bash
uv run python scripts/grpo/train_grpo.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 200 \
    --group-size 4 --num-groups 5 \
    --checkpoint-dir grpo_data/grpo_checkpoints
```

### 2. Standalone server (debug / eval only)

`scripts/grpo/grpo_server.py` is the standalone variant of the in-process
policy server. Use it to serve a trained LoRA checkpoint without spinning up
the trainer:

```bash
uv run python scripts/grpo/grpo_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --lora-checkpoint grpo_data/grpo_checkpoints/iter_0100 \
    --lora-rank 16 --lora-alpha 32 \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --port 5555
```

Do NOT use `gr00t/eval/run_gr00t_server.py` for GRPO collection — it
does not install the noise/raw-action capture hooks that `_prepare_batch`
requires.

---

## Loading a Trained LoRA Checkpoint for Inference

Each `iter_NNNN/` checkpoint dir contains:

```
iter_NNNN/
  lora_weights.pt   # filtered LoRA-only state dict (~80 MB at rank=16)
  optimizer.pt      # optimizer state + param names + kl_base_coef;
                    # only needed for resuming training, ignored for inference
```

`optimizer.pt` also carries the adaptive-KL controller's coefficient
(`kl_base_coef`, `None` when the feature is off). The four resume cases:

| resuming | into | result |
|---|---|---|
| adaptive checkpoint | adaptive run | coefficient restored |
| pre-feature checkpoint (no key) | adaptive run | starts at `kl_coef_base_model`, prints a warning |
| — (fresh, `resume_from=None`) | adaptive run | starts at `kl_coef_base_model`, no warning |
| adaptive checkpoint | **non**-adaptive run | ignored; and the next `_save_checkpoint` writes `None`, so an adaptive → non-adaptive → adaptive chain **silently discards** the coefficient |

`_kl_base_below_streak` is deliberately NOT persisted — it is at most
`kl_base_relax_patience` iterations of state, and resetting it on resume errs toward
holding the coefficient, which is the safe direction. `_save_checkpoint_for_skipped_iter`
early-returns when the target dir already exists, which is correct for weights and AdamW
moments but NOT for the coefficient (it advances on every iteration reaching phase 2b,
including ones that fire no optimizer step), so that path calls
`_refresh_kl_base_coef_in_checkpoint` to patch just that one key in place.

There are two supported inference paths: a **server-client benchmark** (drop
into the existing denoising-lab eval pipeline) and an **in-process notebook**
(direct `DenoisingLab` API for trajectory experimentation).

### Reproducible benchmark via `robocasa_eval_benchmark.py`

`scripts/denoising_lab/eval/robocasa_eval_benchmark.py` is strategy-agnostic —
it just connects to whatever ZMQ server is running on `--port`. So the only
thing that changes for a LoRA strategy is the **server**: instead of
`gr00t/eval/run_gr00t_server.py` (baseline), use `grpo_server.py`, which
already supports loading a LoRA checkpoint via `--lora-checkpoint`.

**Terminal 1 — model venv, GRPO server with LoRA:**

```bash
uv run python scripts/grpo/grpo_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --lora-checkpoint grpo_data/grpo_checkpoints/iter_0100 \
    --use-sim-policy-wrapper \
    --port 5555 \
    --verbose
```

**Terminal 2 — sim venv, identical to baseline_euler eval:**

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --n-episodes 15 --seed 42 --n-envs 2 --port 5555 \
    --max-episode-steps 480 \
    --output-dir ~/benchmark_results/grpo_iter_0100 \
    --strategy-name grpo_iter_0100
```

Use whatever env(s) the LoRA was trained on (see `GRPOConfig.env_names`) —
benchmarking on tasks the policy never saw will mostly measure the base
model. Override `--lora-rank` / `--lora-alpha` / `--lora-target-modules` on
the server command **only** if you trained with non-default values; mismatch
hard-fails inside `load_lora_checkpoint` (`lora_dit.py:165-185`) rather than
silently degrading.

`grpo_server.py` does not track gradients during inference. The `Gr00tPolicy`
forward pass runs inside `torch.inference_mode()`
(`gr00t/policy/gr00t_policy.py:347`), so the `requires_grad=True` flag that
PEFT sets on the LoRA params is a no-op — no autograd graph is built and the
extra cost beyond the baseline server is just the LoRA matmuls themselves.

### Interactive notebook via `DenoisingLab`

For the trajectory-fan / seed-sweep experiments in
`scripts/denoising_lab/notebooks/`, inject the LoRA into the existing
`DenoisingLab` after it loads the base model. See
`scripts/denoising_lab/notebooks/interactive_denoising_panda_lora_v1.ipynb`
for a working copy of `interactive_denoising_panda_v2.ipynb` with the
injection cell pre-wired. The full pattern:

```python
# After: lab = DenoisingLab(MODEL_PATH, EMBODIMENT_TAG, device=DEVICE)
import sys, os
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "grpo"))
from lora_dit import apply_lora_to_dit, load_lora_checkpoint

apply_lora_to_dit(lab.model, rank=16, alpha=32, dropout=0.0)
load_lora_checkpoint(lab.model, "grpo_data/grpo_checkpoints/iter_0100")
# Pin freshly-injected LoRA Linears to the DiT's device/dtype:
lab.model.action_head.model.to(device=lab.device, dtype=lab.dtype)
```

Caveats:

- **`.to(device=lab.device, dtype=lab.dtype)` is required.** PEFT's
  `inject_adapter_in_model` creates the new Linear submodules at default
  device/dtype; without the cast, the first `lab.denoise(...)` call hits a
  cross-device or cross-dtype error.
- **The `lab.action_head` reference set in `DenoisingLab.__init__` is
  unchanged** — LoRA injection mutates the same `model.action_head.model`
  object in place, so subsequent `lab.encode_features_from_sim_obs(...)` /
  `lab.denoise(...)` calls automatically route through the trained adapters.
- **LoRA only touches the DiT, not the Eagle backbone.** A `BackboneFeatures`
  cached from a base-model run remains valid input to a LoRA `denoise`, and
  vice versa — useful for A/B comparing the same observation through both
  policies.
- **For A/B comparisons**, build a second `DenoisingLab` instance for the
  base model rather than trying to "uninject" LoRA — `merge_lora_weights`
  (`lora_dit.py:205`) is irreversible and there is no `unmerge` helper.

### Parallel evaluation from a saved sim state via `eval_lora_from_npz.py`

`scripts/grpo/eval_lora_from_npz.py` is the eval-side counterpart to the
"Init from saved sim state" training mode (covered later in this README): it
loads the same `interactive_rollout.py` `.npz` (`__sim_state__`,
`__model_xml__`, `__ep_meta__`, optional `__step_info__`) and runs
`--num-attempts` parallel rollouts, all starting bit-identically from that
state. Use it to measure how often a LoRA succeeds from a specific
intermediate state and at what speed — complementary to
`robocasa_eval_benchmark.py`, which measures end-to-end performance from
fresh randomized scenes.

Within-attempt diversity comes from the server's unseeded `torch.randn`
during denoising, NOT from env randomness. AsyncVectorEnv subprocess
workers parallelize the MuJoCo cost: with `--num-envs W < --num-attempts N`,
the script collects N rollouts over `N // W` sequential turns of W rollouts
each (mirroring `num_async_vector_env` in training).

**Terminal 1 — model venv, GRPO server with the LoRA loaded:**

```bash
uv run python scripts/grpo/grpo_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --lora-checkpoint grpo_data/grpo_checkpoints/iter_0100 \
    --use-sim-policy-wrapper --port 5555
```

**Terminal 2 — sim venv:**

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/grpo/eval_lora_from_npz.py \
    --env-name robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --obs-path /tmp/saved_observations/ep000_step010.npz \
    --num-attempts 100 --num-envs 10 \
    --max-episode-steps 480 --n-action-steps 8 \
    --output-dir /tmp/eval_iter_0100 \
    --lora-checkpoint grpo_data/grpo_checkpoints/iter_0100
```

The script writes `results.json` to `--output-dir`:

```json
{
  "lineage": {
    "obs_path": "...", "lora_checkpoint": "...",
    "branch_step": 10, "saved_n_action_steps": 8,
    "consumed_substeps": 80, "remaining_substeps_budget": 400,
    "seed": 42, "timestamp": "...", "duration_s": 432.5,
    "...": "..."
  },
  "summary": {
    "total": 100, "successes": 47, "success_rate": 0.47,
    "mean_num_steps_all": 234.5,
    "mean_num_steps_successful": 156.2,
    "mean_num_steps_failed": 314.6
  },
  "attempts": [{"attempt_idx": 0, "success": true, "num_steps": 142,
                "termination": "success"}, "..."]
}
```

Constraints and caveats:

- **`--num-attempts` must be divisible by `--num-envs`** (the script
  reuses `EpisodeCollector`'s `group_size % num_async_vector_env == 0`
  invariant). The error message lists divisors of the chosen
  `--num-attempts` so you can adjust either knob.
- **`--lora-checkpoint` is metadata only.** The script records the path
  in `results.json` but does NOT load weights itself — the server in
  Terminal 1 is responsible. Mismatch (server running base model or a
  different LoRA than the path you record) cannot be detected
  client-side; verify the server's startup log shows the expected
  checkpoint path before running.
- **Pre-spawn ping fails fast on server-down.** Before paying the
  ~10-20 s robocasa import + AsyncVectorEnv worker spawn cost, the
  script pings the GRPO server with explicit ZMQ `RCVTIMEO`/`SNDTIMEO`
  (5 s budget). If Terminal 1 isn't running, you get a
  `ConnectionError` with a corrected start command, not a 20 s wait
  followed by a hang inside the first `get_action`.
- **No video / image / per-step observation saving.** The
  `EvalCollector` subclass overrides `_extract_video_single` /
  `_extract_state_single` / `_get_actions_from_server` to drop those
  recordings (~12 GB + ~460 MB savings on a 100-attempt Panda run). If
  you want per-step inspection, use `branching_rollout.py` for
  single-trajectory analysis instead.
- **`consumed_substeps` accounting is correct across an
  `n_action_steps` change.** The .npz's saved `n_action_steps` (in
  `__step_info__`) drives `consumed_substeps`, not the eval-time
  `--n-action-steps`, so a chunk-size change between save and replay
  doesn't break budget bookkeeping.

---

## Episode Collection

### Groups, seeds, and within-group variance

- A **group** is `group_size` rollouts started from an identical initial
  state (same RoboCasa seed → same kitchen layout, same object poses).
- Within-group diversity comes **only from policy denoising noise**
  (`torch.randn` inside the DiT). The env contributes zero variance once
  the scene is aligned.
- Different groups use seeds `base_seed + g * 1000` (`GROUP_SEED_STRIDE` in
  `collect_episodes.py`), wide-spaced so consecutive groups land on
  visually-distinct kitchens.
- The trainer's per-iter seed stride is 100,000
  (`config.seed + iteration * 100_000`), so two consecutive iters' group
  ranges never collide. This caps `max_groups` at 100.

### Frozen scene seed pool (`scene_seed_pool_size`)

The per-iter stride above means **every iteration trains on, and reports,
brand-new scenes**. Measured between-scene success-rate sd is **0.285**, so at
`num_groups=4` roughly **84% of the per-iteration `episode/mean_reward` variance
is scene resampling**, not policy change. The training curve is then
uninterpretable: a swing between consecutive iterations says nothing about the
update that happened in between.

`scene_seed_pool_size = K > 0` freezes a pool of K scene seeds and cycles it
deterministically across iterations, so the same scenes recur.

### Multi-env / multi-task support

Pass multiple env names in `config.env_names`. The trainer **round-robins**
tasks: iteration 1 → task 0, iteration 2 → task 1, etc. Each iteration
collects ALL `num_groups` for a SINGLE task; tasks are never mixed within
a group (group-relative normalization only makes sense among rollouts of
the same task with the same initial scene).

Per-task tuning:

- `max_episode_steps: int | list[int]` — single value applied to every env,
  or a list parallel to `env_names`.
- `fast_forward_steps: int | list[int]` — same convention.

With 8 tasks × 200 iters, each task gets 25 updates.

### AsyncVectorEnv + scene-bundle alignment

`num_async_vector_env > 1` uses `gym.vector.AsyncVectorEnv` (subprocess
workers, parallel MuJoCo); `== 1` uses `SyncVectorEnv` (no IPC). RoboCasa
picks layout/textures at env construction via a per-instance RNG, so
identically-seeded parallel workers still render **different** scenes.
`GroupAlignmentWrapper` (`collect_episodes.py`) exposes composite RPCs
(`get_scene_bundle`, `apply_scene_bundle`) that the parent invokes via
`env.call()` to copy env-0's scene XML + flat MuJoCo state to all other
workers. After alignment, every env in the group is bit-identical
(verifiable via `--debug-fast-forward`).

### Decoupling group size from worker count (`num_async_vector_env`)

`group_size` is the **logical** number of rollouts per group;
`num_async_vector_env` is the **physical** number of parallel sim workers.
By default (`None`) they're equal — one worker per rollout, unchanged from
before this knob existed. Set `num_async_vector_env < group_size` to cap
peak worker RAM (each MuJoCo worker is ~5 GiB) on RAM-limited hosts: a group
is then collected over `k = group_size // num_async_vector_env` sequential
**turns** of `num_async_vector_env` rollouts each.


### Fast-Forward Branching

Tasks like "open the right drawer" spend most of an episode on the
**approach phase** (navigate, position the gripper). That phase carries
relatively little reward signal compared to the **manipulation phase**
(grasp + pull). Fast-forward focuses GRPO signal on the latter:

```
   t=0 ──────► t=FF (rollout one env, save MuJoCo state)
                    │
                    │ apply_scene_bundle to all G envs
                    ▼
   t=FF ──── independent rollouts ──── t=done
```

1. After scene-bundle alignment, **one env** runs solo for
   `fast_forward_steps` outer steps (each outer step = `n_action_steps`
   sub-steps + one model query).
2. The env's MuJoCo state is captured and pushed to all G envs via the
   same scene-bundle RPC.
3. All G envs continue independently from that state; within-group
   variance comes from the post-branch denoising noise.

Knobs:

- `fast_forward_steps` (int | list[int]): outer steps to fast-forward.
  Default 12; `0` disables. With `n_action_steps=8`, 12 outer steps = 96
  sub-steps (~9.6 sim seconds at 10 Hz).
- `fast_forward_pct` (float, 0-1): probability that a **single iteration**
  uses FF for ALL its groups. Default 0.8. The Bernoulli draw is once per
  `collect()` call, not per group — mixing FF and non-FF groups within an
  iteration would distort cross-group reward comparisons (FF groups have
  shorter `num_steps` and thus larger time-scaled rewards). Long-run FF
  fraction across iterations still approaches `fast_forward_pct` because
  each call gets a different `base_seed`.

Edge cases handled:

- If any env terminates during the FF prefix (e.g., accidental success),
  the collector falls back to a normal seed-aligned group for that group.
- With `num_async_vector_env < group_size`, the post-FF branch point is
  captured once (turn 1, via `get_scene_bundle`) and re-applied for turns
  2..k — the lockstep FF prefix is **not** re-run on later turns (it would
  diverge, since the model-query denoising noise is unseeded).
- FF prefix steps are **not** counted in `episode.num_steps`, so
  time-scaled rewards compare post-branch effort fairly within a group.
- `--debug-fast-forward` saves a per-group montage of camera views to
  `<output_dir>/debug_ff/group<G>_seed<S>_ff<F>.png` so you can eyeball
  that every env in a group really is bit-identical at the branch point.

### Init from saved sim state

A second, more explicit branching mode: instead of having env 0 run the
*current model* forward N steps to produce a branch state (Fast-Forward),
load a **pre-saved** scene + sim state from a `.npz` and start every env in
every group from there. Intended for overfitting / curriculum experiments —
e.g., training GRPO on a single known-hard intermediate state (step 10 of a
specific failing trajectory) to study how the policy refines its behavior at
that state without burning compute on the upstream approach.

```bash
uv run python scripts/grpo/train_grpo.py \
    --init-state-npz-path /path/to/ep000_step010.npz \
    --fast-forward-pct 0.0 \
    --min-alive-groups 0 \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 50
```

The npz must be produced by `scripts/denoising_lab/eval/interactive_rollout.py`
(or any saver that follows the same contract: `__sim_state__`,
`__model_xml__`, `__ep_meta__` keys; see `branching_rollout.py:182-210`).

Mechanics:

1. `_load_init_bundle` (`collect_episodes.py`) parses the npz once per
   collector process and caches the resulting `{ep_meta, model_xml,
   sim_state, consumed_substeps}` dict, keyed by path. The
   `consumed_substeps` field is what makes the post-restore rollout
   truncate at the **remaining** budget rather than a fresh full one —
   see "Budget accounting" below.
2. `_align_envs_to_group_scene` short-circuits the usual "env 0's bundle →
   all envs" handshake and broadcasts the loaded bundle to every env via
   the same `apply_scene_bundle` RPC (`collect_episodes.py:412-522`).
3. Within-group and across-group divergence comes entirely from per-env
   denoising noise; the env starts bit-identical everywhere.

---

## GRPO Algorithm

### Per-iteration phases

```
for iteration in range(start, num_iterations + 1):
    # Phase 0: pre-flight memory cleanup
    _release_memory_to_os()                                # gc + cuda + malloc_trim

    # Phase 1: collect this iter's task
    env_name = env_names[(iter-1) % len(env_names)]
    _collect_episodes(env_name)                            # via collect_episodes.py subprocess

    # Phase 2: compute advantages
    buffer.compute_advantages(max_steps, anchor_advantage, ...)  # per-group z-score

    # Phase 2b: pre-compute reference log-probs (current model == ref before update)
    _compute_ref_log_probs()                               # caches backbone features

    # Phase 3: GRPO update
    _grpo_update()                                         # update_epochs × minibatches

    # Phase 4: log + checkpoint
    if iteration % save_interval == 0: _save_checkpoint(...)
```

Each phase is timed and logged to TensorBoard:

| scalar | phase |
|---|---|
| `time/iteration_seconds` | `iter_start` → end of Phase 3 (excludes logging + checkpointing, which are sampled after it) |
| `time/collect_seconds` | Phase 1 total |
| `time/collect_rollout_seconds` | collector subprocess (imports + worker spawn + rollouts + npz writes) |
| `time/collect_load_seconds` | trainer-side npz read-back into the buffer |
| `time/advantage_seconds` | Phase 2 |
| `time/ref_logprob_seconds` | Phase 2b |
| `time/update_seconds` | Phase 3 |

`collect_rollout + collect_load` is slightly LESS than `collect`: Phase 1 also
covers `buffer.clear()`, `_prune_old_episode_dirs()` (an `rmtree` of aged iter
dirs) and the stale-`.npz` unlink before the subprocess starts. Likewise
`collect + advantage + ref_logprob + update` is slightly less than
`iteration_seconds`, whose remainder is Phase 0 (`_release_memory_to_os()`: two
`gc.collect()` passes + `malloc_trim`) plus the per-iter task/LR setup. Treat the
residual as "untimed glue", not as a missing phase.

A NaN sub-phase is skipped rather than logged as 0, so cached-episode iters
(`resume_from_collected_data`) show a clean gap on `collect` / `collect_rollout`
instead of dragging the autoscale toward zero — on those iters `collect_load` is
the only Phase 1 curve with data. Covered by `test_phase_timing_logs.py`.

### Reward → advantage

```
reward = float(success)                                  # sparse binary (1.0 on success)
scaled = reward / num_steps * max_episode_steps          # faster = better (currently DISABLED)
A_episode = (reward - group_mean) / (group_std + 1e-8)   # PER GROUP
A_chunk = A_episode / num_chunks_in_episode
```

- The reward is **sparse binary**: `1.0` on task success, `0.0` otherwise.
  There is no reward shaping — the codebase does not compute dense progress.
- Time-scaling (`/ num_steps * max_episode_steps`) would make faster solutions
  get larger reward, creating advantage variance even within all-success
  groups. It is currently **DISABLED** in `compute_advantages`
  (see the block comment there for the ablation rationale); the reward fed to
  the group-relative normalization is the raw binary value.
- `A_chunk = A_episode / num_chunks` preserves the within-group
  zero-sum invariant at the chunk level, so every trajectory contributes
  equal **total** gradient weight regardless of length.

A group with reward std < 1e-4 is **degenerate**: the group-mean baseline gives
every episode an advantage of exactly 0. Under the binary reward this happens
for all-success groups (every rollout succeeded) and all-fail groups (every
rollout failed) — only **mixed** groups produce an improvement gradient. This is
not a threshold artifact: the per-group std is either exactly 0 (all G outcomes
identical) or at least `1/sqrt(G)`, which is 3500× the threshold at G=8, so the
`std_r < 1e-4` test is an exact "were all outcomes the same?" check.

By default degenerate groups are **dead**: their chunks are filtered out before
any forward pass (see "Minibatch construction"). `include_anchor_groups`
reclassifies the all-success half as **anchor** groups instead — see the next
section.

### Anchor groups

An all-success group being zero-advantage is correct policy-gradient behavior:
the group mean *is* the Monte-Carlo value estimate, and no rollout beat it, so
there is nothing to improve. But dropping those groups has two costs:

1. **The trust region never covers the solved states.** `kl_coef_last_iter` and
   `kl_coef_base_model` are evaluated only over live chunks, so the constraint
   binds where the policy is uncertain and is blind to where it succeeds.
   Because group seeds are fresh every iteration
   (`seed + iter*100_000 + group_idx*1000`), the anchor states are *different
   scenes* from the live ones, so admitting them genuinely widens the
   constraint's support.
2. **At high success most of the buffer disappears**, and what survives is
   dominated by rare failures: in a 7/8 group the single failure carries
   −2.47 against successes at +0.35. `balanced_minibatch_training` and the
   tent-shaped epoch decay both exist to damp that asymmetry by reweighting
   scarce data; anchor rows instead restore positive mass that is *real*.

Three-way classification in `compute_advantages` (k = successes, G = group size):

| | condition | advantage | role |
|---|---|---|---|
| **signal** | `0 < k < G` | `(r − mean) / std_r` — formula untouched | improvement |
| **anchor** | `k == G` | `anchor_advantage` (constant) | retention |
| **dead** | `k == 0`, or `G == 1` | 0, filtered | — |

**All-fail groups stay dead, deliberately.** Pushing down on every rollout from
a state gives no target to move toward, and it is the avoidance gradient the v2
ablation identified as the collapse mechanism. Run the pseudo-count baseline
below on a `k == 0` group and it hands every episode a *negative* advantage, so
the asymmetry falls out of the math too — it lives in one `if`.

#### Choosing `anchor_advantage`

`k == G` is not proof that `p == 1`: at G=8 a state with true p=0.85 returns 8/8
about 27% of the time (0.85⁸), so the MLE baseline 1.0 over-estimates and each
success really did earn positive advantage. Replace the group mean with the
Beta-Bernoulli posterior mean under κ pseudo-counts at prior success rate p̄,
and divide by a fixed scale (the group's own std is 0):

```
b_g      = (Σ r_i + κ·p̄) / (G + κ)
A_anchor = (1 − b_g) / σ_fixed  =  κ(1 − p̄) / ((G + κ)·σ_fixed)
```

κ is "how many imaginary rollouts my prior is worth"; p̄ is "what success rate
they had". With κ=2, p̄=0.5 (Laplace's rule of succession) and σ_fixed=0.5 (the
max Bernoulli std, ≈ the std of a balanced G/2 group) this is `2/(G+2)`:

| group_size | `anchor_advantage` (κ=2) | balanced-group success, for scale | weakest signal row |
|---|---|---|---|
| 8 | **0.200** | ±0.935 | ±0.354 |
| 12 | **0.143** | ±0.957 | ±0.289 |
| 16 | 0.111 | ±0.968 | ±0.250 |

Those comparisons are at the **episode** level (`A_episode`), which is where the
value is set. What a row contributes also passes through `÷ num_chunks` and the
iteration-wide scale, so the realized row-level ratio varies with group
composition and episode length — larger against a lopsided group whose
advantages are small, smaller against a balanced one. That is the intended
behavior for a fixed absolute magnitude.

Today's dead-group behavior is the κ=0 case. The correction shrinks as G grows —
more real evidence, less prior — which is what makes it a finite-sample
correction rather than a bonus. Since κ, p̄ and σ_fixed are only identifiable as
this one combination, the value is configured directly; recompute it if you
change `group_size` (κ=3 at G=12 reproduces the G=8, κ=2 magnitude if you want
to hold gradient scale fixed across a group-size change).

It is deliberately **not** tied to the running success rate: the estimator wants
the anchor to fade as success climbs, while the negative-mass asymmetry wants it
strongest exactly then. A fixed value keeps the effect readable and leaves the
asymmetry to the balanced-sampler mechanisms.

`anchor_advantage = 0` with `include_anchor_groups = True` is the KL-only
setting — the rows join the batch and the trust region, but their clip term is
identically 0, so they carry no reward signal of their own. Not a literal no-op,
though: the rows occupy minibatch slots, which changes each batch's renorm sample
and raises the per-iteration step count.

#### Bound

With A > 0 the surrogate is `min(A·ρ, A·(1+clip_eps_high))`, so an anchor row's
gradient dies once ρ exceeds 1.2 — the FM surrogate can improve by at most
log(1.2) ≈ 0.18 nats on that path per iteration, regardless of the constant.
`train/mean_ratio_anchor` saturating near `1 + clip_eps_high` means the clip is
bounding the retention move, which is the designed cap.

#### What anchor rows are excluded from

Anchor rows are a third class, so every mechanism *defined by advantage sign*
skips them. Getting any of these wrong silently defeats the feature:

| mechanism | why anchors are excluded |
|---|---|
| per-minibatch z-score | An anchor-only minibatch has no variance except `anchor_advantage / num_chunks` — i.e. **episode length**. A z-score there amplifies length to ±1 and reproduces the time-scaling gradient that collapsed v2. In a mixed batch, all-positive anchor rows also lift the mean and can flip weak real positives negative. |
| `buffer_adv_mean` / `buffer_adv_std` | Computed over signal rows only, so the mean stays ≈0 and `per_iteration_advantage_norm` keeps its sign-preservation property. |
| balanced sampler pos/neg pools | At high success anchors would *dominate* the positive pool and crowd out the genuine mixed-group successes the sampler exists to preserve, while inflating `natural_pos_frac`. |
| dynamic-epoch `success_frac` | Every anchor episode succeeded, so counting them drives the tent toward 1 epoch exactly when anchors were added. |
| PAWS `N`/`D` alive mass | Inflating D drives k → 1 and silently disables the mechanism. |
| pos/neg clipfrac buckets, `n_pos_flipped_by_renorm`, `mean_ratio_fixed/jitter` | All keyed on group-relative sign. Anchors get their own `*_anchor` curves instead. |
| `ref_mse/*`, `chunk_gap/*` | Split by advantage sign; anchor rows are simply dropped from these diagnostics (no `*_anchor` counterpart). |
| jitter (`jitter_pos`/`jitter_neg`) | Anchor entries are always tagged `"fixed"` — λ is selected by advantage sign. `_jitter_gap_diagnostics` takes the jitter set as an explicit mask rather than `~fixed_row_mask`, since excluding anchors from one mask would otherwise sweep them into the other. |
| balanced-sampler viability | The anchor reservation shrinks the sampler's batch size, which can round the minority slot count to 0 and make `_iter_balanced_minibatches` fall back to stratified. At the default `balanced_minibatch_positive_adv_ratio=0.5` this needs `signal_mb_size == 1`, i.e. `anchor_slots == mini_batch_size − 1`. The pool ratio that implies **scales with `mini_batch_size`** — measured first fallback at 1.05:1 (mb=4), 3.05:1 (mb=8), 5.05:1 (mb=12), 7.05:1 (mb=16) — so it is roughly `(mini_batch_size − 1):1`, not a constant. It IS reachable at the default `anchor_max_row_frac=1.0`: that budget caps *chunks*, but the one-whole-episode floor below can admit an anchor episode several times larger than the cap, so a small mixed group plus one 65-chunk anchor episode reaches 6.5:1. Do not confuse this with the ~7:1 *coverage* limit below; they are different thresholds. The fallback logs a WARNING either way. |

The headline `clipfrac` / `mean_ratio` / `mean_log_ratio_abs` curves are the
deliberate exception: they cover **all** trained rows, anchors included, because
they describe the batch the optimizer saw. Use `clipfrac_effective_{pos,neg}`
for the signal-only view and `train/mean_ratio_anchor` for the anchor split.

Anchor rows instead get their **scale** from the buffer-wide signal std
(`anchor_scale`), so an anchor row's weight does not depend on which rows happen
to share its minibatch. It does still depend on the iteration's signal spread
(`buffer_adv_std` is per-iteration): the anchor is a fixed *absolute* magnitude,
so its weight relative to the signal rows grows as the signal advantages shrink
— which is what happens at high success, and is the direction you want. Anchor
rows also never get the mean subtracted, only the scale divided.
`per_iteration_advantage_norm=True` is the intended pairing: under per-minibatch
norm the signal rows rescale per batch while anchors don't, so the ratio wobbles
with batch composition (the startup banner warns about this).

#### Row budget and cost

Each anchor row costs the same `len(tau_centers)` DiT forwards as a signal row
in the ref pass (×2 with the base-model KL) and in every update epoch. At high
success they can be a large fraction of the buffer, so `anchor_max_row_frac`
caps anchor chunks at that multiple of the signal chunk count — one knob for
both compute and the anchor's share of the gradient. Anchor episodes are kept in
index order — first-fit, so an episode that doesn't fit is skipped and a later
shorter one may still be admitted — until the budget is met, and the rest revert
to dead, logged rather than silently dropped. The budget has an implicit floor of
one whole episode: the first anchor episode is always admitted so a small value
shrinks the anchor share instead of deleting the feature, which at ~30–65
chunks/episode can overshoot a small budget several-fold. Because anchor advantages are constant rather than
zero-sum within a group, dropping individual anchor episodes distorts nothing —
unlike a signal group, where it would break `Σ A_ep = 0`.

The budget is **waived when there are no signal chunks at all**: there is no
denominator to measure it against. That is not only the all-success case — an
all-fail *plus* all-success mix also has zero signal chunks while carrying a
non-zero `std_reward`, so the outer skip doesn't fire either. The waiver is
logged, because it means `anchor_max_row_frac` is bounding nothing that
iteration.

Two more bounds worth knowing:

- Above roughly `anchor_max_row_frac ≈ 7` the per-batch cap (`anchor_slots ≤
  mini_batch_size − 1`) stops the epoch from covering the pool: measured coverage
  is 1.00× up to a 5:1 anchor:signal ratio, then 0.70× at 10:1 and 0.35× at 20:1.
  Far outside the default of 1.0, but the excess rows are simply never trained.
- The budget admits episodes in index order, so a small value systematically
  favours the lowest `group_id`s — the earliest-collected groups. Deterministic
  across runs; group seeds rotate per iteration, so it doesn't compound.
- **`anchor_max_row_frac` is therefore not a hard cap.** The one-episode floor
  admits the first anchor episode whatever its size, so the realized ratio can
  exceed the configured budget several-fold (measured 3.25× with a 10-chunk
  episode against an 8-chunk signal pool at `frac=0.5`). Reason about it as a
  target, not a bound — and note the interaction with first-fit: the overshoot is
  worst when a long episode sorts first, and the shorter episodes that would have
  fit are then dropped.

The iteration skip is keyed on `n_signal_chunks` / `n_anchor_chunks` rather than
`std_reward` — see "Skip semantics". An iteration with no signal chunks trains on
its anchor rows when `anchor_advantage > 0` **or** `kl_coef_base_model > 0`; with
both at 0 it has no gradient at all (clip term identically 0, `KL(ref ‖ current)`
zero at `θ == θ_ref`) and stays skipped rather than firing steps that would apply
only weight decay and carried momentum while consuming an iteration.

#### Additive, not diluting

`clip_loss` and both KL terms divide by `signal_mb_size` — the **intended**
signal-row count, held constant across the epoch — when anchor rows are present,
rather than by the total row count. Two consequences:

- A signal row's weight is `1/signal_mb_size`, exactly what it would be in an
  anchor-free minibatch of that size, so turning anchors on doesn't rescale the
  rows that drive improvement, and the anchor KL genuinely *adds* a constraint
  rather than reallocating the existing KL budget across more rows.
- Using the **realized** signal count instead would spike any batch the sampler
  under-fills: a trailing batch with 1 signal + 3 anchor rows would weight every
  row at 1.0 instead of `1/signal_mb_size`, making a 4-row batch the largest step
  of the epoch and — at `max_grad_norm=0.5` — the only clipped one. A constant
  divisor makes a row's weight independent of batch composition; an under-filled
  batch simply contributes proportionally less.

A minibatch left with fewer than 2 signal rows also can't support a
per-minibatch z-score, so its signal rows fall back to the buffer-wide one
rather than entering the surrogate at raw `A_ep / num_chunks` scale.

The divisor is gated on **anchors being enabled this iteration**, not on "this
minibatch happens to hold an anchor row". Because the quota is fractional, the
credit accumulator leaves some batches anchor-free; gating per batch would send
those through `.mean()` and put the composition-dependent weight straight back —
a 1-signal-row trailing batch would weight its row at 1.0 instead of
`1/signal_mb_size`, and whether it did would be decided by the credit counter.

Two caveats on "additive". The loss *weights* are additive as described, but each
anchor row still carries full `1/signal_mb_size` KL weight, so a batch's pre-clip
gradient norm rises by roughly `1 + n_anchor/signal_mb_size`. At the default
`max_grad_norm=0.5` that can put a batch into active clipping, and the rescale
then applies to the signal gradient too. And anchor rows occupy slots, so the
per-iteration optimizer step count rises (see above).

With no anchor rows in the iteration the expression is `row_loss.mean()`,
bit-identical to the pre-anchor path.

#### Metrics

`episode/n_anchor_groups`, `episode/n_anchor_episodes`,
`episode/n_anchor_episodes_dropped` (only with the flag on — `buffer.stats()`
reports the counters unconditionally, so the wandb bulk-dump strips them too);
`train/n_anchor_rows_trained`, `train/mean_ratio_anchor`, `train/kl_loss_anchor`
(only when anchor rows actually trained, and dropped rather than written if
non-finite). `episode/n_live_groups` still counts **signal** groups only, and
`mean_advantage` / `std_advantage` / `pct_positive_advantage` are still computed
over signal episodes only.

Caveats on cross-run comparability, since not every pre-existing curve is
untouched:

- `clipfrac`, `mean_ratio` and `mean_log_ratio_abs` cover anchor rows too (the
  deliberate exception noted above).
- `train/loss`, `train/clip_loss`, `train/kl_loss_last_iter` **and
  `train/kl_loss_base_model`** switch from `.mean()` to
  `.sum() / signal_mb_size` whenever anchors are in play, so their magnitudes are
  not directly comparable to an anchors-off run.
- `episode/n_dead_groups` falls (an anchor group is no longer dead — inherent to
  the feature), and `episode/pct_positive_advantage` / `episode/std_advantage`
  shift because their denominator is the non-anchor episodes: anchor episodes
  leave the sample entirely rather than contributing zeros as they did when they
  were dead. Only `episode/n_live_groups` and `episode/mean_advantage` are
  numerically preserved.
- `train/kl_loss_anchor` covers the `kl_coef_last_iter` term only; the anchor
  rows' base-model KL contribution is not surfaced separately.
- `train/ratio_max`, `train/ratio_min` and `train/grad_norm_*` also shift, since
  anchor rows enter the ratio extremes and the rescaled loss.
- Budget-DROPPED anchor episodes revert to `is_anchor=False` with advantage 0, so
  they still contribute zeros to `pct_positive_advantage` / `std_advantage`. Only
  admitted anchor episodes leave that sample.
- `train/kl_loss_anchor` is a per-anchor-row mean, whereas the `kl_loss_last_iter`
  term inside the loss divides by `signal_mb_size`. Similar names, different
  normalizations — don't read them side by side as one quantity.

#### Suggested ablation

`init_state_npz_path` single-scene mode is the right harness — it is where
v2/v3 ran, where all-success groups are pure "G/G noise draws succeeded", and
where the buffer actually goes empty at high success. Ladder:
`anchor_advantage` ∈ {0 (KL-only), 0.10, 0.143, 0.25} at `group_size=12`,
against the v3 baseline that held at 0.83. Watch whether success *holds above*
the prior plateau rather than merely reaching it; watch
`train/mean_ratio_anchor` for clip saturation; and watch
`episode/group_success_{min,median,max}` spread for the one genuinely uncertain
risk — reinforcing the model's own (ε → a) mappings is reflow-style
self-distillation, and since all exploration here comes from denoising noise
through a shared DiT, over-sharpening on solved states could shrink within-group
variance everywhere.

#### What not to do instead

Do not create within-group variance among the successes to make the group
non-degenerate. That includes the capped speed multiplier suggested in
`compute_advantages`' block comment: capping the reward at
`min(1.5, max_steps/num_steps)` bounds the reward *spread*, but the advantage
divides by the group's own std, which rescales whatever spread survives back to
±1. An all-success group with rewards in [1.0, 1.5] and std 0.15 yields
advantages of ±1.7 — the same magnitude as real succeed-vs-fail signal, so "be
20% faster" gets weighted like "succeed instead of failing". That is the v2
mechanism exactly, cap or no cap. A speed term would need a fixed-scale
denominator, not the per-group std.

#### CLI usage

```bash
# Layer 1 — retention constraint only. Anchor rows join the batch and the KL
# terms; their clip term is identically 0, so they cannot move the policy.
uv run python scripts/grpo/train_grpo.py --include-anchor-groups

# Layer 2 — add the positive pull. 0.143 is the kappa=2 value at group_size=12.
uv run python scripts/grpo/train_grpo.py \
    --include-anchor-groups \
    --anchor-advantage 0.143 \
    --per-iteration-advantage-norm \
    --group-size 12 \
    --anchor-max-row-frac 0.5
```

`--anchor-advantage` without `--include-anchor-groups` is a hard config error
rather than a silent no-op. The startup banner prints
`Anchor groups: ON (advantage=…, row budget=…× signal rows)`, plus a NOTE when
a positive advantage is combined with per-minibatch renorm.

### Post-reopen truncation (`post_reopen_keep_chunks`)

Anchor groups address *which episodes* contribute. This addresses *which chunks
of a failing episode* contribute.

On a grasp-and-place task a failure has a stereotyped shape: approach, close the
gripper (usually on nothing), reopen a few chunks later, then fly the arm away
and meander until truncation. Under the `A_ep / num_chunks` split that meander
absorbs a large share of the episode's negative gradient weight — and it is a
*consequence* of the failure, not its cause. Worse, no success in the group
exhibits it at all (successes terminate on the place), so it is the single most
discriminative difference between the group's positive and negative rollouts.
The update is being told, loudly, "don't fly away after a failed grasp."

`post_reopen_keep_chunks = N` truncates each **failing** episode to
`onset_idx + N` chunks. Everything *before* the onset — the whole approach and
the closed phase — is always kept.

```bash
uv run python scripts/grpo/train_grpo.py \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --post-reopen-keep-chunks 3
```

**The detector.** Everything runs on the **measured** gripper width
`w = gripper_qpos[0] - gripper_qpos[1]`, not on the commanded
`action.gripper_close`: the measurement is what physically happened, it covers a
close on the mug and a close on nothing identically (no need to tell them
apart), and it is unaffected by a noisy gripper channel — on this data 113
chunks command a gripper value that is not even constant across their own
executed substeps, so "the chunk the command flipped in" is not as crisp a
quantity as it sounds. Three stages:

1. **Close** — a three-state machine, and both states before CLOSED matter.
   Nothing is detected until the gripper has been observed **open**, because a
   close is a *transition*, not a state: if an episode begins mid-grasp the
   transition was never observed and there is no close event to find. CLOSED
   then latches only after `post_reopen_min_closed_chunks` (default 3)
   **consecutive** sub-threshold samples — hysteresis guards against chatter
   *inside* the band but does nothing about a single-sample excursion straight
   through it, and this signal carries ~6 mm one-sample blips. Without either
   guard a lone transient dip, or an episode that starts closed, latches and the
   next open sample reads as the reopen, truncating to `keep_chunks + 1` chunks
   and discarding the whole failed grasp. Both are free on the reference data:
   every episode starts at 0.0793, real closed phases run 7–25 chunks, and no
   episode contains a sub-threshold run shorter than 3. Both fail
   **conservatively** — an undetected close means no truncation, visible as
   `episode/n_post_reopen_detected` falling.
2. **Crossing** — that exit index. Unambiguously open, but **0–2 chunks late**
   (1 on 36 of the 43 measured failures): the fingers take about a control chunk
   to travel, so by the time the width clears the open threshold the reopen is
   already underway.
3. **Onset** — walk *backward* from the crossing down the contiguous rising
   edge, while the width is both above `floor + post_reopen_onset_margin`
   (`floor` = the minimum width during the closed phase, so the test adapts to a
   close on nothing at ~0.001 and a close on the mug at ~0.020 alike) **and**
   strictly below its successor. The first chunk of that edge is the onset — the
   first observation in which the gripper has measurably begun to open. This is
   what `N` is timed from.

Why walk backward rather than scan forward for the first rise: the closed phase
contains isolated upward blips (`episode_0039` reads 0.007 mid-hold between
0.001 samples; `episode_0044` reads 0.007 then falls back to 0.003). A forward
scan fires on those. Anchoring at a confirmed crossing means only the rising
edge that actually reaches the open state is ever traversed. The
`w[o-1] < w[o]` term additionally bounds the walk when a closed phase sets its
floor early and then holds *above* `floor + margin` right up to the release. It
changes no onset on any of the 48 real episodes at any margin inside the plateau
below (1 onset moves at margin 0.002, 3 at 0.0015 and under), so "free" is a
statement about the plateau, not about the term.

Its known cost: a **stepped release** — open partway, hold, open fully — is
indistinguishable in the width trace from "hold the object, then release", so
the walk stops at the top step and reports the onset late by the plateau length.
Two physically identical traces differing only by measurement noise on the
plateau can disagree by several chunks. No rule on the width alone separates the
two shapes, and the error is in the conservative direction (a late onset keeps
*more* chunks), so it is accepted rather than fixed.

`N` counts **from the onset chunk itself**: `N=3` keeps `onset`…`onset+2`;
`N=0` drops the onset chunk too. `None` (default) disables the feature and is
bit-identical to the pre-feature behavior.

A truncation that would leave fewer than `post_reopen_min_train_chunks` (default
5) chunks is **refused** — the episode is kept whole and counted on
`episode/n_post_reopen_implausible`. The approach phase alone is ~13 chunks here,
so a handful of retained chunks means the detector latched onto something that is
not the grasp. This is a secondary backstop; the two state-machine guards above
are what actually prevent the known failure modes, and neither counter can see
one on its own (`detected` reads a perfect hit rate in exactly that case).

### FM log-prob surrogate

Flow-matching has no closed-form log-probability. Following DPPO
(Ren et al. 2024), `compute_fm_log_prob` uses negative FM loss as a
surrogate:

```
x_τ = (1 - τ) ε + τ a            # interpolate noise → action
v_target = a - ε                  # true velocity
v_pred = action_head(x_τ, τ, cond)
log π(a | obs) ≈ −E_τ[MSE(v_pred, v_target)]
```

Critical invariants for the importance ratio:

1. **Same ε** for ref pass and current pass. The collector captures the
   actual noise tensor used at inference time via `grpo_server.py`'s
   `torch.randn` hook (thread-local; see "Noise capture" below); training
   reuses it.
2. **Same τ samples** for ref pass and current pass. After sampling
   jittered timesteps for the ref pass, they are stored on each chunk
   (`chunk.tau_samples`) and replayed during `_grpo_update`.

The MSE is computed in **fp32** even though the model runs in bf16: bf16
mantissa is too coarse to resolve the small (current − ref) differences
GRPO depends on, which otherwise inflate `mean_log_ratio_abs` and clip
fraction.

### Noise capture (`grpo_server.py`)

The denoising loop creates ε via `torch.randn` inside
`Gr00tN1d6ActionHead.get_action_with_features`. To recover it without
breaking other code paths:

- `torch.randn` is patched **once at module import** with a thread-local
  router. Other threads see pass-through.
- `GRPOPolicyWrapper.get_action` sets a thread-local capture context only
  during the denoising call, captures the **first 3-D randn** as ε, and
  clears the context on exit (`try/finally`).
- The raw `(B, 50, 128)` action prediction (before
  `decode_action()` slices to the embodiment's actual dims) is captured
  by monkey-patching `get_action_with_features` for the duration of the
  call.

Both are returned to the collector in the `info` dict and persisted into
each chunk's `.npz`. A `compute_action_mask` derived from the loaded
embodiment's modality config is also returned so FM-MSE ignores padded
dims.

### tau_centers

`compute_fm_log_prob` averages MSE over `K = len(tau_centers)` evaluation
points along the same ε → action path. **One DiT forward pass per
center.**

```python
tau_centers = [0.0, 0.25, 0.35, 0.5, 0.6, 0.75]   # default (late-biased)
```

Each iteration, every center gets a small Gaussian jitter
(`std=0.02`, `_sample_jittered_timesteps` in `fm_log_prob.py`) and is
clamped to `[0, noise_s]` where `noise_s = 0.999`. The jittered samples
are then **shared** between the ref pass and the current pass for every
chunk so the importance ratio reflects only model difference, not
sampling noise.

Why late-biased: at inference time, the model takes only **4 Euler
steps** (`t = 0, 0.25, 0.5, 0.75`). Velocity errors at late τ (closer to
the clean action) have fewer remaining steps to correct, so weighting the
surrogate toward late τ aligns the training signal with what matters at
inference.

This is **independent of inference** — the inference loop always uses
exactly 4 Euler steps regardless of `tau_centers`. `tau_centers` only
affects training log-prob evaluation. Adding more centers improves the
log-prob estimate but linearly increases per-minibatch compute.

### Minibatch construction (stratified, dead-group filter)

`_grpo_update_inner` does NOT use `EpisodeBuffer.iter_minibatches` (a flat
shuffle). It uses `_iter_stratified_minibatches` instead:

1. **Dead-group filter**: drop every chunk with `|advantage| < 1e-12` (advantage
   was set to literal 0 by `compute_advantages` for groups with std < 1e-4),
   keeping anchor chunks when `include_anchor_groups` is on. Filtering here
   keeps every minibatch uniformly live-only and avoids a `(0 - mean) / std`
   term polluting the per-minibatch advantage renorm.

2. **Bin live chunks by `group_id`** and shuffle within each bin.

3. **Each minibatch**:
   - GUARANTEED: take up to `mb_size // n_live_groups` chunks from EACH
     live group (best-effort if a group's queue is short).
   - FILLER: fill the remaining `mb_size % n_live_groups` slots from a
     globally-shuffled pool, skipping chunks already used in this batch.

With `mb_size=8` and 5 live groups: 1 guaranteed per group + 3 filler
chunks. Every chunk is yielded exactly once per epoch (across epochs the
permutation reshuffles).

Why stratify: chunks within an episode share an identical
`A_episode / num_chunks` advantage. A flat-shuffled minibatch dominated
by 1-2 episodes has near-zero advantage variance, and the per-minibatch
z-score renorm in `_grpo_update_inner` then squashes that batch's
gradient signal toward zero. Stratification guarantees every minibatch
spans all live groups.

Why uniform-over-CHUNKS for the filler (vs uniform-over-GROUPS): it
self-balances. Fuller groups contribute filler proportionally more often,
so all groups drain in lockstep and the "≥1 per group" guarantee holds
for essentially the whole epoch.

**Anchor rows** (see "Anchor groups") are appended by `_with_anchor_rows`
*around* whichever sampler ran, not inside it — so both sampler paths stay
signal-only and are a transparent pass-through when no anchors exist. The inner
sampler is driven at `mini_batch_size - anchor_slots` (both samplers take an
optional `mb_size` override), so total rows per minibatch — and hence peak VRAM —
stay at `mini_batch_size`. At least one signal slot is always reserved: at
`anchor_max_row_frac` large enough for the anchor pool to dwarf the signal pool,
the proportional quota would otherwise reach `mini_batch_size`, leaving the inner
sampler a batch size of 0 — which its `mb_size or config.mini_batch_size` default
silently turns back into `mini_batch_size`, overfilling every minibatch. At
`mini_batch_size = 1` there is no room for both, so the anchor rows are skipped
with a WARNING rather than exceeding the budget.

The quota may be FRACTIONAL: `_with_anchor_rows` carries a credit accumulator
and emits `floor(credit)` rows per batch, so one epoch consumes the anchor pool
about once. Flooring it at one row per batch instead would ride a small pool
along in every minibatch — 1 anchor chunk against 100 signal chunks would train
~15× per epoch while every signal row trains once. Within a batch the pool is
drawn without replacement even across a reshuffle, so a chunk is never served
twice into the same minibatch.

Two details make the realized share match the pool share rather than merely
approximating it:

- The reserved slot count is chosen by solving for **delivery capacity**: the
  smallest `slots` whose `slots × n_batches` covers the pool, where `n_batches`
  comes from `_min_expected_batches` (which models both samplers, including the
  balanced one's early termination and both of its fallbacks). `ceil(target)`
  alone is not enough — it implicitly assumes the *stratified* batch count, and
  on the balanced sampler the smaller realized count makes the target exceed the
  cap, pinning every batch. Measured 0.83–0.91× delivery in the band where the
  ceil lands on 1. The reservation is only an estimate; `_with_anchor_rows`
  measures the real count and WARNS if capacity still fell short.
- The target is `pool / n_batches` against the batch count the sampler
  **actually produced** — `_with_anchor_rows` materializes the epoch's batches
  before distributing. Estimating it does not work: `ceil(len(entries) /
  signal_mb_size)` is the *stratified* count, while `_iter_balanced_minibatches`
  (the default) stops early once its majority pool drains, and its fallbacks
  change the count again. Under that estimate a 1-chunk pool trained **zero**
  rows on the balanced path. Measured exposure is now 1.00× on both samplers for
  every pool up to the coverage limit below (~7:1 anchor:signal); above that the
  per-batch cap binds and delivery falls off as documented, with a WARNING naming
  the shortfall.

Under `jitter_paired=True` the anchor share is computed against an entry pool
that jitter has doubled, so the realized anchor:signal **mass** ratio is about
half what the same `anchor_max_row_frac` gives with jitter off. Preserving both
that ratio and 1× exposure per epoch is not possible — pairing doubles signal
mass without doubling anchor mass — so exposure is preserved and the ratio moves.
Raise `anchor_max_row_frac` (or `anchor_advantage`) if you want the same anchor
pressure under paired jitter.

When there are no signal rows at all, the anchor entries go through the
stratified sampler directly at full `mini_batch_size`.

**Anchor rows raise the per-iteration optimizer step count.** They occupy
minibatch slots, so the signal rows spread over more batches — up to ~2× the
steps at the same LR when the anchor pool matches the signal pool. Same caveat as
`jitter_paired`'s 2× warning; the startup banner states it. Lower
`update_epochs` or `anchor_max_row_frac` to match an anchors-off baseline's step
budget.

### Balanced Training

Two **independent** mechanisms that address the common failure mode in
early-stage GRPO where most rollouts fail: negative-advantage chunks
vastly outnumber positives, individual mini-batches carry a weak or
one-sided gradient signal, and a small number of sparse successes are
over- or under-weighted relative to the training budget they warrant.

Each is controlled by its **own** flag — `balanced_minibatch_training`
(mechanism 1) and `dynamic_epoch_training` (mechanism 2) — both default
`True`. They are fully decoupled, so any of the four on/off combinations is
valid. With both off, training is bit-identical to the unmodified
stratified-minibatch, fixed-epoch (`update_epochs`) path.

#### Mechanism 1: balanced mini-batch sampling (`balanced_minibatch_training`)

**What it does.** Each mini-batch enforces `balanced_minibatch_positive_adv_ratio`
(X) in **both directions**. The sign class that is underrepresented relative to
X is the "minority" and is oversampled with replacement; the overrepresented
class is the "majority" and is drawn without replacement, controlling when the
epoch ends.

**When it activates.** Always when both sign classes are present:
- `natural_pos_frac < X`: too few positives → cycle positives, drain negatives
- `natural_pos_frac ≥ X`: too few negatives → cycle negatives, drain positives

Falls back to `_iter_stratified_minibatches` only when one sign class is
entirely absent (all episodes fail or all succeed within live groups). Anchor
rows are not in either pool — the caller holds them out and appends them as a
separate quota (see "Anchor groups").

**Why bidirectional matters.** At high success rates (e.g. 70% positive), the
few negative-advantage chunks (failures) receive a very large magnitude from
per-minibatch z-score renorm, producing an outsized "avoid failure" gradient
that can collapse the policy in the next iteration. Cycling negatives caps this
by ensuring each batch has the targeted proportion regardless of the natural
distribution.

**Sampling strategy.** The minority pool reshuffles when exhausted to give
best-effort equal exposure across minority chunks. The majority pool advances
monotonically and may not be fully consumed before the epoch-length anchor is
reached — some majority chunks go unseen each epoch, which is the documented
cost of the rebalancing.

**Epoch length.** Anchored to `ceil(n_live_chunks / mb_size)`, matching the
vanilla stratified path so `update_epochs` remains directly comparable between
balanced and vanilla runs. When the majority pool drains early, the epoch stops
rather than yielding minority-only tail batches that would defeat the balance
guarantee.

**Relationship to Jitter-GRPO.** With paired jitter active (`jitter_pos` or
`jitter_neg` > 0 and `jitter_paired=True`), `entries` is doubled (`fixed +
jitter` copies of each chunk). Both copies of a positive chunk are independent
entries in the positive pool. The balanced sampler draws from them in shuffled
order; the Jacobian regularizer accumulates at epoch granularity (not within a
single mini-batch), so the pairing requirement is satisfied regardless of
whether fixed and jitter copies land in the same batch. In jitter-only mode
(`jitter_paired=False`) each chunk contributes a single `jitter` entry, so the
pool is the same size as vanilla. The combination of these features is sound.

#### Mechanism 2: dynamic epoch count (`dynamic_epoch_training`)

**What it does.** Scales `update_epochs` using a **tent function** of the
positive-advantage fraction, implemented via exact integer arithmetic:

```
m = min(successful_eps, total_eps − successful_eps)
actual_num_epochs = max(1, (4·m·update_epochs + total_eps) // (2·total_eps))
```

This is the integer form of `floor(2·min(sf, 1−sf)·update_epochs + 0.5)`.
The formula peaks at `success_frac = 0.5` (→ full `update_epochs`) and
decays symmetrically toward both extremes:

- **Near 0% success:** all-failure, purely negative advantages, sparse useful
  signal → 1 epoch
- **Near 50% success:** balanced +/− advantages, most informative → full
  `update_epochs`
- **Near 100% success:** all-success, highly asymmetric advantages (the few
  failures get very large negative advantage from group-relative normalisation,
  dominating gradient direction) → reduced epochs

The integer formula avoids ULP cancellation that can corrupt `float`-based
implementations at specific episode counts when `update_epochs ≥ 6`.

**What counts as `successful_eps / total_eps`.**

- `total_eps` is the number of episodes in **live groups** only — groups
  whose per-group reward std is ≥ 1e-4 and thus produce non-zero gradient
  signal. Dead all-success or all-fail groups are excluded from both
  numerator and denominator to prevent their inflating `success_frac` and
  keeping `actual_num_epochs` near `update_epochs` when real signal is
  sparse. Anchor groups are excluded for the same reason — every one of their
  episodes succeeded, so counting them would drive `success_frac` toward 1 and
  collapse the tent to 1 epoch exactly when anchors were added.
- `successful_eps` counts live-group episodes with **positive advantage**
  (`self.buffer.advantages[i] > 0`), not `ep.success`. Under the sparse
  binary reward these coincide for live (mixed) groups — a group's successes
  get positive advantage, its failures negative — so this equals counting
  `ep.success`, while keeping the epoch formula consistent with mechanism 1,
  which oversamples chunks with `c.advantage > 0`.

**Examples.** 5 groups × 4 rollouts, `update_epochs = 4`:
- `success_frac = 0.25` (2/8 positive): `m=2`, `(32+8)//16 = 2` epochs
- `success_frac = 0.50` (4/8): `m=4`, `(64+8)//16 = 4` epochs (peak)
- `success_frac = 0.70` (14/20): `m=6`, `(96+20)//40 = 2` epochs — fewer
  than the old monotonic formula's 3, preventing overshoot at high success

#### CLI usage

```bash
# Both mechanisms are ON by default. Use the tyro switch flags to toggle them
# (--flag enables, --no-flag disables); booleans take no value.
uv run python scripts/grpo/train_grpo.py \
    --no-dynamic-epoch-training \
    --balanced-minibatch-positive-adv-ratio 0.7 \
    --update-epochs 5 \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env
```

The two flags are independent — e.g. the run above keeps the balanced
sampler (`balanced_minibatch_training` stays on) but runs exactly
`update_epochs` epochs every iteration (`--no-dynamic-epoch-training`). To do
the reverse, pass `--no-balanced-minibatch-training` and leave the dynamic
epochs on.

The startup banner prints one line per enabled mechanism:
`Balanced mini-batch sampling: ON (positive_adv_ratio=…)` and/or
`Dynamic epoch count: ON (tent epochs=max(1, floor(2·min(sf,1-sf)·N+0.5)))`.
When `dynamic_epoch_training` is on, a per-iteration line `Dynamic epochs:
X/Y positive-advantage live-group episodes (tent scale=Z) → A/N epochs` is
printed. TensorBoard logs `balanced/actual_epochs` and
`balanced/success_fraction` (gated on `dynamic_epoch_training` and at least
one optimizer step in that iteration).

#### Files touched

| File | Change |
|------|--------|
| `grpo_config.py` | Adds `balanced_minibatch_training: bool = True`, `dynamic_epoch_training: bool = True`, and `balanced_minibatch_positive_adv_ratio: float = 0.5` with `__post_init__` validation (ratio strictly in `(0, 1)` when `balanced_minibatch_training=True`). |
| `train_grpo.py` | `_grpo_update_inner` computes `actual_num_epochs` via the integer tent formula when `dynamic_epoch_training` is on (else `update_epochs`), and dispatches to `_iter_balanced_minibatches` when `balanced_minibatch_training` is on (else `_iter_stratified_minibatches`). `_iter_balanced_minibatches` applies the target ratio bidirectionally — cycles the minority sign class with replacement, drains the majority without replacement. `_log_metrics` emits `balanced/actual_epochs` and `balanced/success_fraction` (gated on `dynamic_epoch_training` and `n_updates > 0`). |
| `test_balanced_fixes.py` | Unit tests for both mechanisms plus their independence (all four on/off combinations): per-batch ratio in both directions, epoch-length anchor, minority cycling, fallback paths, tent formula correctness including integer ULP cases. |

#### Files touched (anchor groups)

| File | Change |
|------|--------|
| `grpo_config.py` | Adds `include_anchor_groups: bool = False`, `anchor_advantage: float = 0.0`, `anchor_max_row_frac: float = 1.0` with `__post_init__` validation (non-negative advantage; advantage > 0 requires the gate; positive row budget). |
| `episode_buffer.py` | `compute_advantages` takes the three knobs and classifies signal / anchor / dead; `_resolve_anchor_groups` applies the row budget at episode granularity. `GRPOEpisode.is_anchor` / `ActionChunk.is_anchor` carry the flag; `_build_chunks` propagates it. `stats()` adds `n_anchor_{groups,episodes,episodes_dropped}`, keeps `n_live_groups` and the advantage summaries signal-only. `__main__` self-test covers the classification. |
| `train_grpo.py` | `train()` passes the knobs through and makes the `std_reward < 1e-8` skip anchor-aware. `_compute_ref_log_probs` admits anchor chunks (gated on config) and passes signal-only chunks to `_summarize_ref_mse` / `_per_chunk_gap_survey`. `_grpo_update_inner` splits live chunks into signal/anchor, computes `anchor_scale`, excludes anchors from the renorm statistics / sign masks / PAWS mass / sign-keyed metrics, divides the loss by the signal row count, and appends the anchor quota via `_with_anchor_rows`. Both samplers take an optional `mb_size`. `_log_metrics` emits `episode/n_anchor_*` and `train/{n_anchor_rows_trained,mean_ratio_anchor,kl_loss_anchor}`. |
| `collect_episodes.py` | Comment only — the `min_alive_groups` "alive" predicate stays mixed-only. |
| `test_anchor_groups.py` | Buffer classification, row budget, config validation, and the real `_grpo_update_inner` on CPU: bit-identity with no anchor rows, renorm isolation, the anchor-only iteration, additive KL, and the PAWS / balanced-sampler / dynamic-epoch exclusions. |

---

### PAWS: dynamic positive-advantage weighting

`positive_advantage_weight_scaling` scales the per-row clip loss on group-good
rows by a live factor `k`, chosen so that reinforcement mass is
`positive_advantage_weight_target_ratio` times erosion mass:

```
N = alive erosion       = sum |row_loss| over negative-advantage rows still
                          passing gradient (dead iff ratio < rho_floor_i, the
                          row's own lower bound — flat at 1 - clip_eps_low
                          unless clip_low_mse_coef > 0)
D = alive reinforcement = sum |row_loss| over amplified positive rows still
                          passing gradient (dead iff ratio > 1 + clip_eps_high)
k = clamp(target_ratio * N / D, 1.0, positive_advantage_weight_max)
                          # lower clamp becomes target_ratio with
                          # paws_k_floor_at_target=True
```

Mass is measured on the **unweighted** row loss, so the estimate never feeds
back on `k`. Anchor rows are in neither term. Both terms pool per **trained**
micro-batch across the whole iteration.

**Read `pos_adv_realized_ratio`, not `pos_adv_weight_k`** — but read it as a
coarse "which side is this iteration pushing on", not as a precise estimator of
`target_ratio`. It is `Σ kᵢ·Dᵢ / Σ Nᵢ`, pooled per micro-batch at the `k` that
micro-batch was actually weighted by:

- `≈ 1.0` → the two sides are balanced, i.e. the mechanism is off in effect.
  When `target_ratio > 1` this is the reading to worry about: it is what the
  removed cross-iteration EMA produced on every resume, survivable at 52 %
  success and fatal (0.67 → ~0.04 in one iteration) at 67 %. **At the config
  default `target_ratio = 1.0` it is instead the on-target reading** — the two
  cases are only distinguishable by knowing `target_ratio`.
- `>> target_ratio` → erosion is largely clip-dead (`N << D`). The raw ratio
  *diverges* as `N → 0` (measured 12–15 in that regime), so the emitted value is
  clamped to `positive_advantage_weight_max` to keep the curve readable; the
  unclamped terms are always available as `pos_adv_pos_mass` /
  `pos_adv_alive_neg_mass`. Cross-check `clipfrac_effective_neg` and `k_min`.

**Deviation from `target_ratio` does not by itself mean a clamp is binding.**
Each `kᵢ` is a *prefix* estimate (the pool excluding its own micro-batch), so
when the running prefix ratio differs from the whole-iteration ratio the pooled
result drifts off target with no clamp involved — measured +9…+20 % on skewed
group shapes with `k` comfortably inside `[1, max]`. On the 238-micro-batch
reference iterations, where the prefix is stable, it read 1.7500–1.7501 against
a target of 1.75. Use `k_min` / `k_max` to tell a clamp from prefix drift.

`k` itself is a poor headline because it moves for a benign reason: under
per-minibatch renorm the z-score forces `Σ_{post>0}|A| ≡ Σ_{post≤0}|A|`. `N` and
`D` are keyed on the **pre**-renorm sign, so that gives `N/D ≡ 1` only absent
renorm sign flips (a flipped row falls out of *both* masses; one pos→neg flip in
an 8-row minibatch measures `N/D ≈ 0.88`) — watch
`n_pos_flipped_by_renorm`, which read 0 on 15 of 16 iterations of the reference
run. There `N/D` sat at `exp(jitter/gap_pos)`, measured 1.0464 ± 0.0057 over
every non-clipping iteration and matching `exp(gap_pos)` to within 0.5 %, so
`k ≈ target_ratio·1.046`. When drift starts clip-killing negatives, `N` falls and
`k` falls with it — the mechanism correctly tracking a real drop in erosion, not
the mechanism weakening. In that same run `k` slid 1.83 → 1.49 over the last
three iterations while the realized ratio never left 1.750.

`pos_adv_weight_k_{min,max}` bracket the **measured** `k`s (the unmeasured prior
is excluded — it is a config-derived constant, and folding it in would pin
`k_min` to it and hide the real spread). They are absent on an iteration that
never measured. Together with `k_last` they separate a clamp from prefix drift,
and they surface a mid-iteration excursion — e.g. a run of one-sided minibatches
pinning `k` at the cap — that `k_last` alone would miss.

**No cross-iteration state.** `k` is derived from the current iteration's pool
alone. Until the pool holds any amplified-positive mass there is nothing to
measure, so `k` falls back to the analytic prior `k = target_ratio` — the
fallback **tracks the target** instead of being pinned to `1.0` independently of
it. (At the config default `target_ratio = 1.0` the prior *is* 1.0 — correct,
since that config asks for equal masses.) There is no count-based warm-up beyond
that: the prior is unmeasured and is *not* floored by the measurement, so holding
it longer over-amplifies in the clip-dead-erosion regime, where the measured `k`
floors at 1.0 while the prior still says `target_ratio`. Under
`per_iteration_advantage_norm` the minibatch zero-mean identity does not hold, so
there is no prior to stand on and the fallback is `1.0`; that combination
measures much worse overall anyway (see "Gradient accumulation"). See
"Checkpointing & Resuming → Resume" for the resume bug this design replaced.

#### Flooring `k` at `target_ratio` (`paws_k_floor_at_target`)

`paws_k_floor_at_target = False` (default) keeps the historical lower clamp of
`1.0` on the **measured** branch:

```
k = clamp(target_ratio * N / D,  1.0 (default) | target_ratio (flag on),
          positive_advantage_weight_max)
```

**Why.** A tighter lower clip (`clip_low_mse_coef > 0`) deliberately kills more
negative rows, which shrinks `N`, which through `target_ratio · N/D` *lowers*
`k`. So tightening the erosion brake would also weaken reinforcement — the
opposite of the intent. Flooring at `target_ratio` removes only the "amplify
*less* than target" case; it never amplifies more than the measurement asks for,
because the `min(…, max)` cap is still applied afterwards.

Measured: `N/D` sits at **1.04–1.06** on healthy iterations, so
`target_ratio · N/D > target_ratio` and the floor is **inert** there; `N/D` falls
to **0.66** during collapse, which is exactly where it binds.

Two knife edges worth knowing:

- At exactly `N/D == 1` the `D_iter + 1e-8` denominator puts the measured `k` a
  part in 1e8 *below* `target_ratio`, so the floor binds by that much. Harmless,
  but it means "inert" is a statement about `N/D > 1`, not `N/D ≥ 1`.
- Validation rejects `paws_k_floor_at_target=True` with
  `positive_advantage_weight_target_ratio < 1.0`: that would pin `k` *under* the
  no-op point and force de-amplification, inverting the mechanism.

The other two `k` branches are **untouched**, and each for its own reason:

- The unmeasured prior (`D_iter == 0`) is already
  `min(max(target_ratio, 1.0), max)` — it already tracks `target_ratio`, so with
  `target_ratio ≥ 1.0` enforced there is nothing to change.
- The `per_iteration_advantage_norm` fallback stays `k = 1.0` because that path
  has **no prior to stand on** — the buffer-wide z-score breaks the minibatch
  zero-mean identity that makes `N/D ≈ 1`. Flooring an unmeasured, unjustified
  value at `target_ratio` would amplify on the strength of nothing.

---


**Two guards on the flag.** `positive_advantage_weight_target_ratio` must be
`≥ 1.0` (below the no-op point the floor would force de-amplification on every
healthy iteration) **and** `≤ positive_advantage_weight_max`. The second is not
cosmetic: `k = min(max(measured, target_ratio), max)`, so a target above the cap
collapses the expression to the constant `max` for *every* measurement — the
measurement-driven controller silently becomes a fixed amplifier, while the banner
advertises an inverted interval (`clamped to [5, 2]`) as if it were a range.

**`train/pos_adv_k_floor_binds_frac`** and **`train/pos_adv_k_cap_binds_frac`** give the
fraction of MEASURED micro-batches whose `k` the floor / the cap moved. Read them before
`k_min`/`k_max`: a `floor_frac` near 1 means `k_min` **is** the floor and carries no
information about `N/D` — exactly the ambiguity they resolve.

They replaced an earlier `pos_adv_weight_k_raw_min`, which had three defects. `clamp` is
monotone, so `k_min == clamp(k_raw_min)` and a **cap**-bound minimum could never satisfy
`k_raw_min < k_min` — the cap was structurally unreportable. It fired on default unfloored
configs too, since the historical floor is still `1.0`. And `k_raw` is a *prefix* estimate,
so a minimum over ~300 micro-batches selects the shortest, noisiest prefix and reads
exactly `0.0` whenever the first trained micro-batch had no alive negative rows — a
fabricated zero of exactly the kind this codebase avoids elsewhere.

### Clipped surrogate + KL

```
ratio = (current_log_prob - ref_log_prob).exp()
advantages = (A - A.mean()) / (A.std() + 1e-8)            # renorm per-batch
# rho_floor is a [B] tensor, == 1 - clip_eps_low on every row at the default
# clip_low_mse_coef = 0.0 (see "Per-row, MSE-referenced lower clip").
surr1 = A * ratio
surr2 = A * clamp(ratio, min=rho_floor, max=1 + clip_eps_high)
clip_loss = -min(surr1, surr2).mean()

# Schulman k3 KL estimator (non-negative pointwise, symmetric gradient):
inv = ref_log_prob - current_log_prob
kl_loss_last_iter = kl_coef_last_iter * (inv.exp() - inv - 1).mean()

# Optional KL anchor to the base frozen DiT (LoRA disabled). Skipped when
# kl_coef_base_model = 0; otherwise base_log_prob is pre-computed once per
# iter inside the same no_grad pass that produces ref_log_prob, with
# `with disabled_adapters(model.action_head.model)`.
inv_base = base_log_prob - current_log_prob
kl_loss_base_model = _kl_base_coef_now() * (inv_base.exp() - inv_base - 1).mean()

loss = clip_loss + kl_loss_last_iter + kl_loss_base_model
# + vel_anchor_coef * D when vel_anchor_coef > 0 — see "Velocity anchor".
```

`_kl_base_coef_now()` is `config.kl_coef_base_model` unless
`kl_base_adaptive=True`, in which case it is the controlled value — see "Adaptive
base-model trust region" below. **`compute_base` still gates on the CONFIG value**
(`compute_base = self.config.kl_coef_base_model > 0.0`): it decides whether the
base-model forward pass runs at all, and gating that on the controlled value would
let the controller switch off its own input.

When anchor rows are present in the minibatch, all three `.mean()`s become
`.sum() / signal_mb_size` (a constant, not the realized row count) — see
"Anchor groups → Additive, not diluting". With no anchor rows the expression is
exactly the `.mean()` above.

NaN/Inf guard: a minibatch with non-finite loss (typically bf16 ratio
overflow when `|log_ratio|` is large) is **skipped**, the
`n_skipped_nonfinite` counter increments, and training continues.
`clip_grad_norm_` only bounds finite gradients — it does not rescue NaNs.
The guard fires BEFORE `backward()`, so a skipped minibatch never puts
anything into the gradient buffer (see "Gradient accumulation" for what that
means when several minibatches share one optimizer step).

Second, independent guard on the **gradient** side: if `clip_grad_norm_` reports
a non-finite norm — either because `backward()` produced inf/NaN even though the
forward loss was finite, or because the fp32 sum-of-squares of large-but-finite
gradients overflowed — the optimizer step is **dropped**, the gradient buffer is
zeroed, and `n_nonfinite_grad_steps` increments (with a console WARNING).
Clipping cannot save that buffer: `total_norm = inf` gives a clip coefficient of
0, so the buffer becomes either all-NaN (`inf * 0`) or exactly `0.0` (finite
gradients scaled by 0) — nothing to rescue either way. Stepping on the NaN case
would write NaN into every LoRA param, poison AdamW's moments for the rest of the
run, and — because the iteration would still report `n_updates > 0` — persist a
NaN checkpoint that a later `--resume-from` would load, all while `grad_norm_*`
still looked normal (the offending norm is excluded from that average). Dropping
the step instead leaves the weights at their last good value and training
continues with the next window. Expected reading is a flat
`train/n_nonfinite_grad_steps == 0`; anything above zero is worth investigating
even though the run survives it.

If ZERO minibatches commit a gradient step in an iteration (every batch
non-finite, every window dropped, or every group dead), the iteration is
treated as **skipped** and the resume checkpoint is saved under the last
successfully-updated iter's name (see "Checkpointing").

### Adaptive base-model trust region (`kl_base_adaptive`)

`kl_base_adaptive = False` (default) is OFF and bit-identical to HEAD: the
controller returns an empty dict, no `train/kl_base_*` curve is emitted, and no RNG
is consumed. The off-switch claim is an
out-of-tree differential (materialize HEAD with `git show HEAD:scripts/grpo/<f>` and
drive `test_grad_accum.run_update` on both trees); `test_kl_base_adaptive.py` covers the
in-tree half — no emission, no state touched, `_kl_base_coef_now()` falling back to the
config value.

**The problem.** `ref_mse/log_base_ratio_mean` (= `ref_log_prob − base_log_prob`, an
on-policy KL(θ‖base) estimate in loss-native nats) increases NET in all 10 archive
runs that log it — runB 0 → 0.302, arm1 0.018 → 0.111, ref16 0 → 0.193, lr12e4
0 → 0.272 — and nothing bounds it. 6 of the 10 are strictly monotone and 4 dip; the
discriminator is MAGNITUDE, not monotonicity, and `res11` alone stays inside a narrow
band (0.057 → 0.070, net +0.012) while also being the most stable and highest-SR run
measured. Unlike `lora/weight_delta_norm` this quantity is **continuous across every
resume** (arm1 it10 = 0.01809 vs runB it10 = 0.01808), so it is globally comparable
across the whole archive.

The k3 gradient is `coef·(e^x − 1)·(−∂lp/∂θ)`, i.e. a restoring force **linear** in
the drift `x`, against a surrogate per-row gradient of ≈0.9. At the shipped
`kl_coef_base_model = 0.1` that is **0.20 %** authority at runB's ignition point
(x = 0.0181) and 2.9 % even at x = 0.302 once the run is dead. Every run that logs
the quantity used coef 0.1 and the archive-wide peak is 2.90 %, so the mechanism has
never been tested — under-powered by ~100× (20 % authority at ignition needs
coef 10.09).

**Why a controller and not a bigger constant.** The coefficient needed for constant
authority moves **14×** within one run: 20 % needs coef 10.0 at x = 0.018 but coef 0.7
at x = 0.302. Structurally a fixed coefficient is proportional control on a plant that
is itself an integral, which leaves a steady-state offset. Adapting it adds integral
action: the coefficient climbs until the drift stops *growing*, whatever the surrogate
pushes with. Same reason adaptive-KL controllers exist in RLHF rather than fixed
penalties.

**The law.** Deadband on `log_base_ratio_mean` against `kl_base_target`, deliberately
**asymmetric**:

| condition | action |
|---|---|
| `lbr > deadband_hi · target` | `coef ×= kl_base_adapt_rate`, clamped at `coef_max`; streak reset |
| `deadband_lo · target ≤ lbr ≤ deadband_hi · target` | hold; streak reset |
| `lbr < deadband_lo · target` | streak += 1; on reaching `kl_base_relax_patience`, `coef /= kl_base_relax_rate` **floored at the STARTING coefficient**, streak consumed |
| no / non-finite reading | **hold** (never relax), flagged `kl_base_coef_held_no_reading` |

Three properties, each of which fixed a real bug:

- **Asymmetric, and relaxation is floored at the start value.** Low drift is the
  normal healthy state, *not* evidence of over-braking. A symmetric ×2/÷2 controller
  disarms itself: `log_base_ratio_mean` reads exactly **0** at iteration 1 (PEFT
  zero-inits `lora_B`, so iteration 1 IS the base policy), so a fresh run always
  starts below the band and immediately accumulates the streak. Replayed on runB it
  entered the emergency several doublings under-braked. Patience alone only *paces*
  the walk; the floor is what stops it.
- **A missing reading holds.** An absent measurement is not evidence that drift is
  low; relaxing on it would walk the coefficient down over a few iterations.
- **The relax branch can never RAISE the coefficient** (`max(prev/relax,
  min(start, prev))`) — otherwise a resumed run whose restored coefficient is below a
  raised `kl_coef_base_model` would jump *up* on a quiet iteration and report it as a
  relaxation.

**Ordering.** Called in `train()` **after** `_compute_ref_log_probs()` (so the drift is
measured at θ ≡ θ_ref, free of this iteration's own update) and **before**
`_grpo_update()` (which captures the coefficient once per iteration). It sits inside
the Phase-2b timing window, so `time/ref_logprob_seconds` nominally covers it —
negligible against a multi-minute phase.

**`kl_base_target` CANNOT be calibrated from the archive, and this is the honest state of the evidence.** Replaying the real controller against all 10 drift-logging runs
and asking for a target that (a) engages on every run that degraded and (b) spares
every run that did not, the constraint set is **empty by 38×**:

| run | outcome | drift at the decisive iteration | implied bound |
|---|---|---|---|
| `full175` | degraded .583 → .333 | **0.0076** | target < 0.0051 |
| `arm1` | declined .688 → .458 | 0.0578 | target < 0.0386 |
| `CONTROL_r9` | collapsed .646 → .354 | 0.0875 | target < 0.0583 |
| `res11` | never degraded, best SR | max 0.0701 | target ≥ 0.0467 |
| `mse0.8_r13` | declined | max 0.0846 | target ≥ 0.0564 |
| `lr12e4` | **+0.50 SR, the biggest gainer** | max 0.2925 | target ≥ 0.1950 |

`full175` degraded at drift 0.0076 while `lr12e4` thrived at 0.2925 — a **38× overlap**
between the healthy and degrading ranges. Worse, tested as a classifier over all 26
iteration transitions in the corpus, neither the level nor its increment predicts a
subsequent ≥0.15 SR drop:

| predictor | AUC |
|---|---|
| `log_base_ratio_mean` level | **0.317** |
| its per-iteration increment | **0.358** |

Both are *below* chance (0.5). With 6 positives the CI is wide enough not to refute the
mechanism, but the archive provides **no evidence for it**. The original rationale — "it
grows in every run and nothing bounds it, therefore bounding it will help" — conflates
a thing that always happens with a thing that causes harm; something that rises in the
runs that *succeed* cannot discriminate.

So **0.055 is an operating guess, not a derivation.** It is chosen only to sit inside
the two-run window [0.0467, 0.0695] that spares `res11` and still engages on runB's
it12 collapse, and three of the other seven runs contradict it. An earlier default of
0.025 was strictly worse — it changed runB by exactly nothing while driving `res11` and
`ref16` to the cap and pinning them there — but "better than 0.025" is the only claim
0.055 supports.

**Treat this feature as speculative until the readout is fixed.** Success is currently
measured on the very 48 episodes the update consumes, with a 0.094 difference-SE and a
period-3 scene sawtooth; a frozen-weights baseline and a pass-complete paired statistic
are prerequisites for telling whether bounding drift does anything at all.

**This does NOT transfer across harnesses.** Pool runs degrade at drift levels where
single-scene runs thrive: arm1 declines from 0.0578 while `res11` holds SR 0.73 at
0.0572–0.0701. For a small pool, engaging before arm1's onset while leaving runB's
healthy 0.0295 alone needs target ∈ (0.0197, 0.0385) → **~0.03 for a K=4 pool**, with
the named risk that if a pool run needs to reach ~0.06 to learn the way the
single-scene runs did, 0.03 forbids it. `kl_base_coef_max` is the hedge.

`lbr` is measured on this iteration's own rollouts, which is `num_groups` scenes'
worth of episodes **regardless of `scene_seed_pool_size`** — so the UNITS transfer when
K changes; the healthy LEVEL does not.

**`kl_base_coef_max` defaults to 5, not 30.** Over-correction is visible but not cheap:
relaxation is ÷1.1 gated on 3 consecutive below-band iterations, so walking the cap back
down to the start takes **51 iterations**, against archive runs of 14–20 — i.e. not
recoverable inside a run. A cap of 5 bounds the worst case to **~30 %** authority at the
default target (~53 % at drift 0.10) — real pushback that cannot freeze the policy. At 30
it reaches **178 %**, reachable in 3 tightenings. Watch `train/kl_base_coef_at_max`; the
startup banner prints both figures for the resolved config, and the banner is the
authority if it ever disagrees with this file.

**Expect a floor, not a gain.** `res11` held SR 0.73 by barely moving. A working trust
region plausibly converts "collapse" into "flat" — that is the floor this establishes,
not its ceiling. Pair it with something that supplies upside.

**Known limitation.** `_summarize_ref_mse` is called with **signal chunks only**, so
with `include_anchor_groups=True` at high success an all-anchor iteration leaves
`_ref_mse_stats = None` and the controller HOLDS — precisely when the retention term is
most load-bearing. It is flagged, not silent.

**Persistence.** `optimizer.pt` carries `kl_base_coef`; see "Checkpointing & Resuming".

#### CLI usage

```bash
# Recommended: start at 1.0 (~6% authority at the default target — near-inert, but only 2 doublings
# from useful). Leaving kl_coef_base_model at its 0.2 default warns.
uv run python scripts/grpo/train_grpo.py \
    --kl-base-adaptive --kl-coef-base-model 1.0

# K=4 pool: tighter target, per the per-harness note above.
uv run python scripts/grpo/train_grpo.py \
    --kl-base-adaptive --kl-coef-base-model 1.0 --kl-base-target 0.03 \
    --scene-seed-pool-size 4 --scene-seed-pool-base 105067 \
    --num-groups 4 --max-groups 4 --min-alive-groups 0
```

### Velocity anchor (`vel_anchor_coef`)

`vel_anchor_coef = 0.0` (default) is OFF and bit-identical to the tree before it:
no extra forward, no RNG draw, no `vel_anchor/*` curve. The off claim was checked as an
out-of-tree differential against HEAD (recipe in `test_vel_anchor.py`'s docstring);
`test_vel_anchor.py` test 13 is the in-tree half.

**What it adds.**

```
D_row = mean_k  mean_valid  (v_theta(x_k) - v_anchor(x_k))^2     # per row, fp32
loss += vel_anchor_coef * (D.mean()            # no anchor rows in play
                           | D.sum() / signal_mb_size)   # anchor rows in play (as KL)
```

- `x_k` are the **training forward's own DiT inputs** for the K `tau_centers` samples,
  including the jittered `x'_k` on jittered rows (the positive rows at `jitter_neg = 0`).
  The anchor forward runs inside the same K-loop at the identical input, under
  `torch.no_grad()`.
- `v_anchor` is the **base DiT** (`vel_anchor_path = None`, LoRA disabled through
  `disabled_adapters`), or a **frozen LoRA checkpoint** (`vel_anchor_path = iter_NNNN/` or
  `lora_weights.pt`). A checkpoint is loaded once at setup through `_load_lora_state`,
  which hard-fails on key or shape mismatch. It is held as a plain attribute (~58 MB),
  never a module parameter, and evaluated through `torch.func.functional_call`. Only the
  A/B tensors are swapped, so the anchor uses THIS run's `lora_alpha`/`lora_rank`.
  Checkpoints do not store them, so an anchor saved under a different alpha at the same
  rank passes every check (same limitation as `--resume-from`).
- There is no PAWS weighting: D is not advantage-keyed. Anchor rows are included, on the
  same constant divisor as the KL terms.

**Why not the KL knobs.**

1. **Jitter bias.** Both k3 terms compare the *jittered* current log-prob against the
   *clean* reference/base log-prob on positive rows, so at `jitter_pos = 0.125` they
   measure the jitter gap (~0.0155 nats) rather than drift. At iteration 1 (θ = base), the
   plam-0.125 run logs `kl_loss_base_model == kl_loss_last_iter == 7.17e-6`, which is
   gap²/2 × positive fraction × 0.1. Raising the coefficients under jitter adds Jacobian
   contraction instead of a brake.
2. **Wrong direction even with matched inputs.** The k3 term sees one scalar per row,
   `x = MSE_θ − MSE_anchor`. Its gradient is `(eˣ − 1)·∇MSE_θ`, a rescaled copy of the
   row's own FM gradient, which acts like an advantage offset. It never points from v_θ
   toward v_anchor, and at small drift its pull scales with θ's tiny self-consistency
   residual. `∇D = 2·(v_θ − v_anchor)·∂v_θ/∂θ` is a linear restoring force in every output
   direction, with a unique minimum at the anchor. It is also the path-space (Girsanov)
   KL of a flow policy, up to the noise scale.

**Interpretation caveat.** At jittered inputs D also anchors the velocity's
*input-sensitivity*, which is exactly what `jitter_pos` shrinks. A holding arm therefore
means "drift held **and** plam's cumulative effect damped". `vel_anchor/jac_part_pos`
measures the second part.

**Metrics** (all absent when off):

| Metric | When | Read it as |
|---|---|---|
| `vel_anchor/start_mean`, `start_pos`, `start_neg`, `start_p90` | ref pass: clean inputs, start-of-iteration weights, signal chunks | **The drift readout.** A proper squared distance, unlike `ref_mse/log_base_ratio_mean = ‖δ‖² − 2⟨r_θ, δ⟩`. It levels off in a holding arm. p90 catches a tail of runaway rows. |
| `vel_anchor/start_gripper_frac` | ref pass | Pooled share of D in the gripper column (uniform ≈ 1/12). Rising in a declining arm means gripper weighting is the next knob. Absent while every D is 0 (0/0), i.e. iteration 1 of a fresh base-anchor run. |
| `vel_anchor/start_exec_frac` | ref pass | Pooled share in the executed steps 0..`n_action_steps`−1. If drift piles into the never-executed steps, masking them from the policy-gradient term is the follow-up. Absent while every D is 0, like the gripper share. |
| `vel_anchor/train_mean` | update, all trained rows | What the penalty acts on, jittered inputs included. |
| `vel_anchor/train_last_epoch_mean` | update, final epoch | End-of-update distance, i.e. how far one update pushes before the pull catches up. |
| `vel_anchor/loss` | update | `coef × reduced D`, averaged over trained micro-batches. |
| `vel_anchor/grad_ratio`, `grad_cos` | `grad_probe_every` cadence (every Nth trained micro-batch, jitter rows not required) | `‖∇penalty‖ / ‖∇clip_loss‖` and their cosine, pre-Adam, via `torch.autograd.grad` (so `.grad` and the step are untouched), averaged over probes with a non-zero penalty gradient (a probe still AT the anchor, D = 0, is skipped by both). ≪ 0.1 means inert; ≫ 1 means close to frozen. cos < 0 means the penalty is resisting where GRPO wants to go. |
| `vel_anchor/jac_part_pos` | jitter-gap diagnostics (start of iteration) | Positive rows' D(x′) − D(x) at the same weights and taus: the part of the penalty that resists plam rather than drift. Absent when jitter is off (the diagnostics do not run); exactly 0 at `jitter_pos = 0` with `jitter_neg > 0`. |
| `vel_anchor/coef` | every iteration | The coefficient in force. |
| `vel_anchor/source` (text) | once | `base` or the checkpoint path. |

**First-micro-batch check.** When the anchor equals the starting weights (a fresh run
with the base anchor, or `vel_anchor_path` = the resume checkpoint), the first
micro-batch must read `max D < 1e-5`, or setup's anchor mapping is wrong and the run
raises. This check is one-shot. A non-finite D is left to the non-finite guard (that
micro-batch is dropped) and the check stays armed for the next one.

**Smoke test (GPU host).** On a fresh base-anchor run, iteration 1's `start_*` read 0 and
the two `start_*_frac` are absent by construction (D ≡ 0 before any step), so run it for
two iterations before judging that every metric is written.

**Cost (estimates, not yet measured).** Training adds K no-grad anchor forwards per
micro-batch (~+30–35% update time). The ref pass adds K per batch (~+100 s per
iteration). The force balance costs two extra `autograd.grad` backwards per probed
micro-batch (~+5% at cadence 20).

**`stop_after_iterations`.** This exits after N iterations *of this invocation*, through
the normal save path, while the LR still follows `num_iterations`. A 1-iteration trial
therefore trains at exactly the LR that iteration would get in the full run.

**Calibration (`calibrate_vel_anchor.py`).** Run one update-only trial per coefficient on
a cached iteration, then read the shrink and turn of the first update against coef 0:

```bash
.venv/bin/python scripts/grpo/calibrate_vel_anchor.py \
    --coefs 0 0.1 1 10 100 --targets 0.05 0.15 0.35 \
    --out-dir grpo_data/vel_anchor_calib -- <train_grpo.py args for A's config>
# --dry-run prints the commands; --analyze-only re-reads finished trials. Trial dirs must
# not exist yet (a leftover one would be reused silently), so use a fresh --out-dir.
```

- **What each trial adds.** Each trial appends `--vel-anchor-coef c
  --stop-after-iterations 1 --resume-from-collected-data --checkpoint-dir
  <out>/coef_<c>` after the verbatim passthrough (tyro is last-wins). Any `--resume-from`
  / `--vel-anchor-path` in the passthrough are kept.
- **What it measures.** `‖ΔW‖` of the effective update `(α/r)(B@A)` relative to the start
  weights, computed exactly at rank 2r with no dense product. It also reports the cosine
  with the coef-0 update, `vel_anchor/train_last_epoch_mean` and `vel_anchor/grad_ratio`.
- **Suggestions.** c_lo / c_mid / c_hi come from log-interpolating shrink(c) at the
  targets. A target outside the measured range is reported as a bracket, never
  extrapolated.
- **Guards.** The run hard-fails if the trials did not load identical cached episodes
  (same `episode/success_rate` and `episode/num_chunks` at the trained step). It warns
  when shrink is non-monotone.
- **Output.** A table on stdout and `calib_summary.json`.

### Per-row, MSE-referenced lower clip (`clip_low_mse_coef`)

`clip_low_mse_coef = 0.0` (default) is OFF and bit-identical to a flat
`1 − clip_eps_low` floor on every row. `test_clip_floor.py` asserts the
flags-off path's DETERMINISM and the additivity of the new keys; the
bit-identity-vs-baseline check is an **out-of-tree differential** (recipe in that
test's docstring) — it cannot live in-tree, because once this change is committed
`HEAD` contains it.

**Why a flat `clip_eps_low` is the wrong shape.** The importance ratio is
`ρ = exp(MSE_ref − MSE_θ)`, and `MSE_θ ≥ 0`, so the whole reachable range is
`ρ ≤ exp(MSE_ref)` — measured 1.002–1.03 against `1 + clip_eps_high = 1.2`.
Consequences, both measured:

- **Positive-advantage rows are never clipped.**
  `train/clipfrac_effective_pos` is identically 0 in 69 of 69 logged
  iterations. The negative branch is the only one with room to run, so the
  lower clip is the *only* live brake in the objective.
- **A flat epsilon grants wildly non-uniform MSE headroom inside ONE
  iteration.** `clip_eps_low` is in log-ratio (nat) units while the quantity
  that diverges is `MSE_θ`. At `clip_eps_low = 0.08` the allowed MSE inflation
  spanned **261× at `ref_mse/p10` down to 2.1× at `ref_mse/max`** in a single
  iteration.

**The rule.** When `clip_low_mse_coef > 0`, each row gets

```
MSE_ref_i   = max(-ref_log_prob_i, 0)                        # clamped, see below
budget_i    = min(clip_low_mse_coef * MSE_ref_i,
                  |ln(1 - clip_eps_low)|)                    # nats
rho_floor_i = exp(-budget_i)
```

so every row is allowed the same **relative** inflation, `1 + coef`, instead of
the same absolute nat count.

Three properties, all load-bearing:

1. **`clip_eps_low` stays an absolute CEILING on the budget** (the `min`). The
   mechanism can therefore only ever be *tighter* than today, never looser. That
   is enforced, not merely implied: `exp(-(-log(1-eps)))` does **not** round-trip
   to `1-eps` in fp32 (123 of 999 `eps` values land one ULP *below* it), so
   `rho_floor` is `maximum(exp(-budget), 1-clip_eps_low)`, which also makes a
   ceiling-pinned row's floor bitwise equal to the flags-off path's.
   That is deliberate: `MSE_ref` **grows** as the field degrades (measured
   0.0023 → 0.0297 over one run), so an uncapped `c · MSE_ref` budget would
   *widen* the clip exactly when it needs to tighten.
2. `MSE_ref` is clamped at `≥ 0` before use. It is `−MSE` upstream so a
   positive `ref_log_prob` should be impossible; the clamp exists so an fp edge
   case cannot put the floor **above** 1.0, which would clip every row that
   failed to move.
3. `coef == 0.0` is numerically identical to today, RNG stream and every
   pre-existing TB scalar included.

**One tensor, six consumers.** `1 − clip_eps_low` was read at six places, and
if any of them disagreed with the loss then PAWS's alive-erosion mass `N` (hence
`k`) and every clip metric would describe a different clip than the optimizer
applied. So `_grpo_update_inner` materialises **exactly one** `[B]` tensor
`rho_floor` immediately before the surrogate and hands that same object to all
five; none of them consults `config.clip_eps_low` for its bound:

| site | what it feeds | what breaks if it desynchronises |
|---|---|---|
| `surr2 = A · clamp(ρ, min=rho_floor, max=1+hi)` | the loss itself | — |
| `alive_neg_mask = … & (ρ_det >= rho_floor)` | PAWS `N` → `k` | `N` counts rows the loss already clip-killed, inflating `k` |
| `clipfrac = (ρ < rho_floor) \| (ρ > 1+hi)` | `train/clipfrac` | — |
| `clip_killed_gradient(ρ, surr1, surr2, rho_floor, hi)` | `train/clipfrac_effective_{pos,neg}` | — |
| `over_clip = (ρ < rho_floor) \| (ρ > 1+hi)` | `train/clipfrac_{fixed,jitter}_{pos,neg}` | — |

`clip_killed_gradient` accepts `float | Tensor` for its low argument: a float is
an *epsilon* (bound `1 − eps`, the legacy form every existing call site uses), a
Tensor is the **bound itself**. Its four-case table is per row and holds for
either form.

`torch.clamp` with **both** bounds as tensors. Value-identical to the
scalar-bound form (bitwise over 2e5 random rows × fp32/bf16/fp64, exact boundary
values included) **and** gradient-identical to it, including at an exact tie.

An earlier revision used `torch.maximum(torch.minimum(…))`, which is
value-identical but **not** gradient-identical: at `ρ == bound`, `clamp` routes the
full gradient to the selected branch while `maximum`/`minimum` split it 0.5/0.5,
measuring `d(loss)/d(ρ) = −0.75` against clamp's `−1.00` for either advantage
sign. The enclosing `torch.min` does *not* compensate — a differential test
against the pre-change tree diverged in `grad_norm_max` and in the final weights
on a batch containing boundary ratios, i.e. it broke the flags-off invariant.
Ties are rare but reachable: `ρ` is fp32 (`fm_log_prob.py` accumulates in fp32),
and a 4–8 fp32-ULP window of `log_ratio` maps exactly onto each bound (~6e-7 per
row). Clamp also removes an order dependence — `max(min(x,hi),lo)` disagrees with
clamp's `min(max(x,lo),hi)` whenever `lo > hi`, unreachable today given the `eps`
validation but latent. Both bounds must be tensors: `torch.clamp` rejects a mixed
`(Tensor min, Number max)` call.

**Anchor rows** get the same formula — their `ref_log_prob` is a real
measurement and they do pass through the surrogate, so they need an entry. It is
inert for them: an anchor carries a constant *positive* advantage, and by the
four-case table a positive row's `min()` always picks the unclamped branch below
the lower bound. A tighter floor can move only the sign-agnostic
`train/clipfrac`, which is a metric. Anchors are already excluded from the PAWS
masses, so it cannot move `k` through them either.

**`jitter/neg_clip_budget_used` keeps its flat `|ln(1 − clip_eps_low)|`
denominator** so the curve stays comparable across runs; the coefficient does
not change its meaning. Same for the new `pos_clip_budget_used`.

**Pair it with `paws_k_floor_at_target`.** Killing more negative rows shrinks
`N`, which under the current controller *lowers* `k` — so (A) alone tightens the
erosion brake and weakens reinforcement at the same time. See "PAWS → Flooring
`k` at `target_ratio`".

```bash
uv run python scripts/grpo/train_grpo.py \
  --clip-low-mse-coef 8.0 --clip-eps-low 0.2 \
  --positive-advantage-weight-scaling --paws-k-floor-at-target \
  --positive-advantage-weight-target-ratio 1.75
```

The startup banner prints the resolved formula, the uniform inflation factor,
the `MSE_ref` at which the ceiling starts binding, and the budget at two
representative `MSE_ref` values (`train_grpo.MSE_REF_BANNER_PROBES`, which
bracket the measured range) — so a coefficient large enough to pin every row to
the ceiling, i.e. to silently revert to the flat clip, is visible in the first
screen of output.

### Per-row erosion-drift distribution (`drift/*`)

Emitted every iteration, unconditionally — no flag, no cost worth measuring.
**Pooled over every trained micro-batch** of the iteration, over
**pre-renorm-negative advantage, non-anchor** rows.

**Why.** Every `train/*` number is a mean over 266–336 micro-batches, so the
per-**row** spread of erosion drift was completely unmeasured — and a lower-clip
threshold calibrated from iteration means will clip far more rows than intended
the moment `p90` is several times `p50`. These are the numbers
`clip_low_mse_coef` should be picked from.

**Why pooled, and not the first micro-batch** (which an earlier revision used):
`_compute_ref_log_probs` runs *before* `_grpo_update`, so on the first trained
micro-batch `n_updates == 0` and the weights **are** the reference weights —
`log_ratio` is identically 0 for a fixed row and exactly `−gap` for a jittered
one. The percentiles then read ~0 no matter how far the policy drifted, which is
the opposite of what they exist to measure. (The sibling `jitter/*` block gates on
`n_updates == 0` *deliberately*, for the opposite reason: it wants a drift-free
measurement.) Pooling also means a first micro-batch holding no negative row no
longer loses the family for the whole iteration — ~17 % of iterations at an
80 %-positive mix.

**Signed and one-sided.** The clip fires on `ratio < rho_floor`, i.e.
`log_ratio < −budget`, so the quantity is `−log_ratio` (**positive = eroded
downward**) tested against `+budget`. A row that drifted *up* shows as a negative
value and is *not* counted over budget: for a negative-advantage row an upward
move hits the **upper** bound, which by `clip_killed_gradient`'s four-case table
leaves the row **alive**. An earlier revision compared `\|log_ratio\|` against the
budget and over-reported 4× on a batch containing one up-drifted row.

| Scalar | Meaning |
|---|---|
| `drift/neg_down_{p10,p50,p90,max}` | percentiles of per-row `−log_ratio` over that population, pooled across the iteration. Positive = eroded toward the floor; negative = drifted up. `torch.quantile`'s linear interpolation. |
| `drift/neg_rows` | the surviving (finite) row count behind the pooled numbers — an **iteration** total, roughly `n_micro_batches × negative rows per batch`. |
| `drift/budget_mean` | pooled mean of the per-row nat budget, so the percentiles can be read against the constraint without recomputing it. Not a threshold — the budget is per row. |
| `drift/neg_frac_over_budget` | fraction of pooled rows past **their own** budget — the flat `\|ln(1−clip_eps_low)\|` when `clip_low_mse_coef == 0`, and `min(coef·MSE_ref_i, that ceiling)` when it is on (read back off `rho_floor`, one expression for both). Strict `>`: a row exactly at its floor is not clipped. The direct "how much erosion is this clip killing" readout. |
| `drift/neg_over_mseref_{p50,p90,max}` | drift per unit of the row's **own** `MSE_ref`. `clip_low_mse_coef` is a threshold on exactly this ratio, so **`p90` is the coefficient that leaves ~90 % of rows inside their budget** — read it off a control run instead of inferring it from marginals. Denominator clamped at 1e-12 (a row with `MSE_ref == 0` has a zero budget at any coefficient, so `+inf` would poison the percentile). |
| `drift/neg_corr_drift_mseref`, `drift/neg_mseref_top_decile`, `drift/neg_mseref_all` | the **joint** structure, which percentiles alone cannot give. Positive correlation with `top_decile > all` means the runaway rows are the **badly-fit** ones — and a per-row budget `coef·MSE_ref` then grants exactly those rows *more* room, so a flat budget is tighter where it matters. Negative correlation favours the per-row form. Absent when fewer than 2 rows survive or either series has zero variance. |
| `drift/neg_frac_born_dead`, `drift/neg_born_rows` | the same fraction restricted to the **pre-step** micro-batches — those whose FORWARD ran before any `optimizer.step()` — captured at the forward, so it is all `gradient_accumulation_steps` of the first window (inferring it after the step drops the window-closing micro-batch, which cost 1 of every k) — where `θ == θ_ref` and "over budget" therefore means "**born** clip-dead". This is the tripwire the `clip_low_mse_coef` × `jitter_neg` config warning tells you to watch; it reads ~1.0 when the budget is below `gap_neg`. **Read the sample size with it**: the denominator is `neg_born_rows`, at most `gradient_accumulation_steps × mini_batch_size` rows (3–5 at the defaults, so the curve is quantized to `1/n`), and both keys are **absent** on any iteration whose first window held no negative signal row (~17 % at an 80 %-positive mix). |

Non-finite rows are **dropped**, not counted — same policy as the
`ratio_maxes` / `ratio_mins` accumulators, because `ratio = log_ratio.exp()` can
overflow while the clipped loss stays finite. A poisoned micro-batch therefore
shows up as a small `neg_rows` rather than a NaN curve.

Ungated on `n_updates`, like `ref_mse/*` and `jitter/*`: the numbers come off a
micro-batch that *trained*, so they survive an iteration whose gradient windows
were all dropped — and a blown-up per-row drift is a likely *cause* of landing
there. A micro-batch with no pre-renorm-negative signal rows emits **nothing**
(a curve gap, not a fake 0).

### Weight-step direction cosines (`lora/cos_step_*`)

Emitted every iteration alongside the pre-existing `lora/weight_delta_norm`.
`_compute_lora_step_cosines()` is called from the same two `_log_metrics` sites
as `_compute_lora_delta_norm()`.

| Scalar | Meaning |
|---|---|
| `lora/cos_step_early` | `cos(step_now, L_early)`. **This is the one to read.** Measured across 6 runs its minimum is −0.058 over 41 updates on the runs that stayed healthy, and it reaches **−0.49** and **−0.62** on the two that collapsed directionally — i.e. it turns negative before the success curve does. Emitted only when a reference is available. |
| `lora/cos_step_prev` | `cos(step_now, step_prev)`. Step-to-step consistency; useful for telling a genuine direction change from per-iteration sampling noise. |
| `lora/cos_step_cumulative` | `cos(step_now, W_prev − W_init)`. Emitted **because it is free, not because it is informative**: it is self-referential — once a run turns, `W_prev − W_init` turns with it — and measured POORLY, holding +0.37…+0.53 straight through a collapse. |
| `lora/step_norm` | `‖step_now‖`, so a cosine can be read against the size of the move that produced it. |
| `lora/cos_ref_source` | TB **text** summary (`global_step 0`), written once per distinct source: `"paths"`, `"frozen_after_N_logged_iters_of_run"`, or `"none"`. `M = N + 1`, because the first logged iteration only seeds the history and returns before the counter advances. `_of_run` is load-bearing: `setup()` resets the history, so on a **resumed** run the reference is `W_(resume+M) − W_resume` — the run is scored against its own *post-resume* direction. Pass `--cos-ref-lora-paths` to pin a reference from the pre-resume lineage. |

**The reference `L_early`**, two forms:

```
cos_ref_lora_paths: tuple[str, str] | None = None   # (path_a, path_b); L_early = W(b) - W(a)
cos_ref_iterations: int = 2                          # else freeze L_early = W_now - W_init
                                                     # after N logged iterations
```

`cos_ref_lora_paths` points at two existing `iter_NNNN/` LoRA checkpoint dirs (or
`lora_weights.pt` files) and is loaded ONCE at setup by a plain `torch.load` of
the filtered state dict — deliberately *not* `load_lora_checkpoint`, which loads
*into* the live model. Checkpoint keys are DiT-relative while
`named_parameters()` is model-relative, so the prefix is derived by **module
identity** (walking `named_modules()` until the object *is*
`model.action_head.model`) rather than hardcoded. A key-set or per-key shape
mismatch against the live trainable set, or a degenerate `W(b) == W(a)`, is a
hard failure with an explanatory message: a silently-partial reference would make
`cos_step_early` a cosine against a different subspace than the step lives in,
which reads as a plausible number and is meaningless.

**Why the frozen reference matters.** The in-run freeze scores a run against its
*own* early direction, so it cannot detect a run that had already turned by
iteration N. Pass `--cos-ref-lora-paths` from a known-good run's checkpoints for
an external reference. `--cos-ref-iterations` must be `≥ 1`; a zero reference
vector has no defined cosine.

**Memory and syncs.** Two device-resident snapshots of `_lora_init_params`' size
(~80 MB each at rank 16): `_lora_prev_params` (W at the previous logged
iteration) and `_lora_prev_step`. Both are `copy_`-ed **in place** rather than
rebound, so no history accumulates and no rebuild ever holds an old and a new
copy at once. `L_early` lives on **CPU** — it is a frozen constant read once per
~13 minutes of wall clock, so a per-tensor H2D copy costs nothing and keeps
device-resident extra memory at exactly the budgeted two snapshots. Every dot
product and squared norm accumulates into ONE device vector with a single sync at
the end, matching `_compute_lora_delta_norm`'s pattern — no `.item()` per tensor.

**Edges.** The first logged iteration and any zero-step iteration (the early-skip
log path, or an iteration whose every gradient window was dropped) emit nothing
AND leave the history untouched, so the next real step is still compared against
the last real *step* rather than against zero. Nothing is lost at iteration 1:
the only quantity that would be defined there is `step_norm`, and
`lora/weight_delta_norm` already *is* `‖W_1 − W_init‖`.

### Knobs and files: per-row clip floor, `k` floor, new diagnostics

| Knob | Default | Meaning |
|---|---|---|
| `clip_low_mse_coef` (`--clip-low-mse-coef`) | `0.0` = OFF | Per-row lower clip: `budget_i = min(coef·MSE_ref_i, \|ln(1−clip_eps_low)\|)`, `rho_floor_i = exp(−budget_i)`. Uniform *relative* MSE budget `1 + coef`. `clip_eps_low` remains an absolute ceiling, so it can only ever be tighter than the flat clip. |
| `paws_k_floor_at_target` (`--paws-k-floor-at-target`) | `False` | Floor the MEASURED PAWS `k` at `positive_advantage_weight_target_ratio` instead of `1.0`. Requires `1.0 ≤ target_ratio < positive_advantage_weight_max`. Only consulted with `positive_advantage_weight_scaling=True`. |
| `cos_ref_lora_paths` (`--cos-ref-lora-paths`) | `None` | `(path_a, path_b)` — two existing LoRA checkpoint dirs (or `lora_weights.pt` files). `L_early = W(b) − W(a)`, loaded once at setup. |
| `cos_ref_iterations` (`--cos-ref-iterations`) | `2` | When `cos_ref_lora_paths` is unset, freeze `L_early = W_now − W_init` after this many logged iterations. Must be `≥ 1`. |

New TB families: `drift/*` (9 scalars), `lora/cos_step_{prev,cumulative,early}`,
`lora/step_norm`, `lora/cos_ref_source` (text), `jitter/pos_clip_budget_used`.
All are pure additions and emit unconditionally where their inputs exist.

| File | Change |
|------|--------|
| `grpo_config.py` | Adds `clip_low_mse_coef`, `paws_k_floor_at_target`, `cos_ref_lora_paths`, `cos_ref_iterations` + four `__post_init__` validations (`coef ≥ 0` and finite; the flag/`target_ratio ≥ 1.0` interaction; `cos_ref_iterations ≥ 1`; `cos_ref_lora_paths` a 2-tuple of existing paths, normalised to `tuple[str, str]`). |
| `train_grpo.py` | `clip_killed_gradient` takes `float \| Tensor` for its low bound. `_grpo_update_inner` builds the single `rho_floor` tensor and threads it to all six consumers, applies the `k` floor, and pools per-row erosion travel over every trained micro-batch plus a pre-step-scoped born-dead fraction (returned as `_drift_diag`, carried onto the `n_updates == 0` path too). `_jitter_gap_diagnostics` hoists `lo_budget` and adds `pos_clip_budget_used`. New `_dit_param_prefix`, `_load_lora_state`, `_load_cos_ref_direction`, `_compute_lora_step_cosines`; `setup()` resolves `L_early`; `_log_metrics` takes `lora_cosines=` and emits `drift/*`, `lora/cos_*` and the provenance text (wandb mirrored). `train()` gains the two banner blocks; `MSE_REF_BANNER_PROBES` is the banner's `MSE_ref` probe pair. Class-level OFF defaults for the six `_lora_cos_*` attributes so `__new__`-built test trainers degrade instead of raising. |
| `toy_train_grpo.py` | `_log_metrics` override forwards `lora_cosines`; banner prints `Clip low MSE c`. |
| `test_clip_floor.py` | New CPU suite (see the Contents table). |
| `test_jitter_metrics.py` | The `clip_killed_gradient` call-site spy now asserts the low argument is a per-ROW `rho_floor` tensor equal to `1 − clip_eps_low` (was: the float epsilon), with `clip_eps_high` still a scalar. |

### `grad_share/*` — where the loss wants the update, vs. where AdamW sends it

Eight TB scalars, one per `lora_target_modules` entry, giving each module type's
**percent of the iteration's LoRA gradient energy** (`Σ‖g‖²`, energy-weighted
across the iteration's optimizer steps, then normalised). They sum to **100 every
iteration by construction** — normalised by their own sum, over a group map that
`setup()` asserts is exhaustive (and rejects duplicate targets), because a
silently-unmatched param would leave the curves summing to less than 100.

**Why it is worth a curve.** The gradient and the applied step disagree by orders
of magnitude, and nothing else in the metric set shows it. `proj_out_1` +
`proj_out_2` are **0.79 %** of the 14.5 M trainable params but carried **93.5 %**
of gradient energy on one measured arm — they sit after `norm_out` (a non-affine
`LayerNorm`, so the 32 blocks' output scale is deleted before the head acts) and
`proj_out_1` is the only LoRA weight on the timestep-conditioning path. `eps`
decides whether the step follows per-coordinate gradient magnitude or coordinate
**count**, and at 99.2 %/0.8 % those answers differ enormously: on measured arms
the head's share of the applied step was 33 % at `eps=1e-5` and 0.7 % at
`eps=1e-8`, from 93.5 % and 8.6 % of the gradient respectively. Read it alongside
`train/grad_norm_mean` (total magnitude), `lora/step_norm` (how far the weights
moved) and the "AdamW betas / eps" section below (which regime you are in).

**Reading it.** `grad_share` is a *diagnostic of the loss*, not of the optimizer —
it says nothing on its own about whether the step is well spent. A single
module type dominating is normal; what matters is whether the *step* follows it.
Note `lora/weight_delta_norm` is NOT comparable across an `eps` change (it counts
raw A/B factors, which include LoRA's gauge freedom `A→GA, B→BG⁻¹`); for
cross-run drift use `ref_mse/log_base_ratio_mean`.

**Cost.** One fused `_foreach_norm` + `index_add` per optimizer step — a second
read of the already-resident ~58 MB gradient, ~60 µs, against a ~1770 s
iteration. Exactly one host sync per iteration. Taken BEFORE `clip_grad_norm_`
(which rescales `.grad` in place) and committed only on steps that actually
reach the weights, so dropped windows cannot skew the allocation. Absent — a
curve gap, not a fake 0 — on an iteration where `n_updates == 0`.

### AdamW betas / eps — and which regime this run is in

`adam_beta1`, `adam_beta2` and `adam_eps` were hard-coded at the
`optim.AdamW(...)` construction site (`(0.9, 0.999)` / `1e-5`); they are now config
fields, and they land in the TensorBoard `config` dump — a run's own artifacts
previously did not record the optimizer it used.

**Shipped defaults are `beta1=0.9`, `beta2=0.99`, `eps=1e-8` — the NORMALISED
regime, not the pre-knob values.** To reproduce a run recorded before the knobs
existed, pass `--adam-eps 1e-5 --adam-beta2 0.999` explicitly.

**Resuming re-applies config over the checkpoint.** `Optimizer.load_state_dict`
replaces `param_groups` wholesale, keeping only `params`, so a resume would
otherwise silently adopt the *checkpoint's* betas/eps/weight_decay — meaning
`--adam-eps 1e-8` into a pre-knob checkpoint would quietly train at `1e-5`, in the
wrong regime. `lr` escapes this because the annealing line re-sets it every
iteration. `_reapply_optimizer_hyperparams` forces config back on after the load
and prints what the checkpoint held; the AdamW moment state (`exp_avg`,
`exp_avg_sq`, `step`) is untouched.

PyTorch AdamW is `θ -= lr · m̂ / (√v̂ + ε)`. **ε is added to `√v̂`, outside the
sqrt** — which is what makes `adam_eps` a regime switch here rather than a
numerical guard.

**Which regime.** Per-coordinate `√v̂` is on the order of `‖g‖ / √n_trainable`,
with `n_trainable = 14,532,608` (√ = 3812):

| arm | `train/grad_norm_mean` | per-coord `√v̂` | `eps / √v̂` |
|---|---|---|---|
| `jitter_pos=0.125` | 0.0038 | 1.0e-6 | **10×** |
| `jitter_pos=0.25` | 0.0067 | 1.8e-6 | 5.7× |
| `jitter_pos=0.35` | 0.0143 | 3.8e-6 | 2.7× |
| `overfit_step10_v5` | 0.019 | 5.0e-6 | 2.0× |

So at the shipped `eps=1e-5` the **ε floor dominates the denominator** for the
bulk of coordinates and the step is ≈ `lr · m̂ / ε` — closer to SGD+momentum than
to normalised Adam. (Those are RMS figures over a heavy-tailed distribution:
`lora_B` is zero-initialised, so `∂L/∂A = 0` at step 1 and the A/B scales differ.
A high-gradient tail is already in the normalised regime; the bulk is not.)

Two consequences that matter for **reading the experiment log**, not just for
tuning:

1. **Gradient magnitude is not normalised away.** `jitter_pos` moves `‖g‖` by
   3.8× across the arms above — via the FM residual, since the FM loss is
   least-squares so `grad ∝ residual` — so at fixed `learning_rate` a λ change
   silently changes the **step size** too. A λ ablation that does not co-adjust
   `lr` to hold `lora/step_norm` fixed is confounded, and at least one recorded
   pair (`jitter_pos` 0.125 at `lr` 5.9e-5 vs 1.2e-4) is.
2. **`adam_beta2` is nearly inert at `eps=1e-5`.** `√v̂` barely enters the
   denominator, so changing β₂ alone does almost nothing. It becomes load-bearing
   only once ε is lowered below `√v̂`.

**On lowering ε.** `eps=1e-8` makes the step gradient-magnitude invariant
(bounded by ~`lr` per coordinate, SNR-weighted through `m̂/√v̂`), which removes
confound (1). It is **not a bug fix and not obviously an improvement**: `eps=1e-5`
is the deliberate RL convention (CleanRL's PPO, the original baselines) precisely
*because* policy-gradient noise makes the normalised regime amplify low-magnitude
coordinates, and the in-tree value traces to `grpo_cont.py:230`. Expect two
things:

- **A ~10× larger aggregate step at the same `lr`.** In the ε regime
  `‖step‖ = (lr/ε)·‖m̂‖` ≈ 0.046; normalised, `‖step‖ ≈ lr·√n_eff` ≈ 0.46.
  Recalibrate `lr` **down** ~10×, and **measure** the factor (one
  `resume_from_collected_data` update, read `lora/step_norm`) rather than assuming
  it — the heavy tail means the realised factor is smaller than the median-
  coordinate estimate.
- **A qualitative change in *which* parameters move.** The 10× does not come from
  scaling the existing step; it comes from **activating the millions of
  small-magnitude coordinates the ε floor was holding still**. That floor is a de
  facto trust region over 14.5M parameters driven by a gradient whose
  step-to-step coherence (`lora/cos_step_prev`) is only 0.45–0.87. Treat it as a
  one-variable experiment, not a new default.

**If you lower ε, also lower β₂.** β₂ = 0.999 is a 1000-step memory ≈ 24
iterations at ~42 optimizer steps/iter, so `t_eff = (1−β₂ᵗ)/(1−β₂)` reaches only
~650 by iteration 25 — the **entire run** sits inside `v`'s warmup, and `v` lags a
non-stationary gradient scale (one arm's `ref_mse/pos_mean` grew 4× over six
iterations, which `v` would trail by the whole run). β₂ = 0.99 is a 100-step
memory (~2.4 iterations) at the cost of steady-state `√v̂` noise 2.2% → 7.1%; for
a ~1000-step non-stationary objective that is the better trade. Bias correction
removes the *bias* from step 1 either way — what warms up is only the variance of
the estimate (11% at it1, 6.5% at it3). `__post_init__` emits a
`warnings.warn` (not an error) for `adam_eps < 1e-6` with `adam_beta2 >= 0.999`.

**β₁ is the cheaper lever on incoherence, because it does not leave the ε
regime.** 0.9 is a 10-step memory against ~42 optimizer steps per iteration, so
`m` averages only ~24% of an iteration. When the measured pathology is low
coherence, more temporal averaging raises per-step SNR for free — the same thing a
larger `gradient_accumulation_steps` buys, without the compute. 0.95 = 20 steps,
0.98 = 50 steps. The interaction with the ε regime is favourable: there the step
is ≈ proportional to `m̂`, and raising β₁ grows `m̂` on **coherent** directions
while leaving incoherent ones near zero — so it amplifies the consistent
component specifically, rather than scaling everything the way `lr` does. Costs:
lag (slower response to a genuine change in the gradient) and a larger effective
step, so watch `lora/step_norm` and drop `lr` if it overshoots.

**Two neighbours worth knowing are inert.** `weight_decay = 1e-5` is decoupled
(`θ -= lr·wd·θ`), i.e. 1.2e-9 per step per unit of θ at `lr` 1.2e-4 — ~1.3e-6
total relative shrinkage over a 25-iteration run, against the ~0.7
`lora/weight_delta_norm` those steps produce. It is not regularising anything.
And `max_grad_norm = 0.5` never binds: measured `train/grad_norm_mean` is
0.0038–0.019 (26–130× below the bound) with `train/n_nonfinite_grad_steps` and
`train/n_skipped_nonfinite` at 0 throughout. Do not read a slow run as
gradient-clipped.

Covered by `test_adam_knobs.py`.

### Gradient accumulation

`gradient_accumulation_steps = k` (default 1) accumulates the gradients of `k`
consecutive mini-batches into a single optimizer step:

```
per micro-batch that survives the non-finite guard:
    if the window is empty:  optimizer.zero_grad()
    (loss / k).backward()                  # 1/k → the buffer holds the MEAN
    if the window now holds k:             # close the window
        clip_grad_norm_(...); optimizer.step()
at the end of EVERY epoch:
    if the window is non-empty: clip_grad_norm_(...); optimizer.step()  # flush
in either case: if the accumulated gradient is non-finite, drop the step,
    zero the buffer, and count it (n_nonfinite_grad_steps) instead
```

**Why.** `mini_batch_size` cannot be raised: peak VRAM at `mini_batch_size=8`
is ~21.5 GB of ~25.3 GB on an A10G (~1.48 GB per row → ~8-9 rows is the
ceiling). The per-row cost is dominated by the K-loop in
`compute_fm_log_prob`, which accumulates the log-prob across all
`len(tau_centers)` DiT forward passes and calls `backward()` once — so autograd
retains the activations of all K passes simultaneously. Accumulation is
therefore the only route to a larger effective batch. Peak VRAM is unchanged
(each micro-batch's graph is still freed by its own backward; the retained fp32
grad buffers are ~80 MB at rank 16) and total forward/backward work is
unchanged. What changes: the update direction averages `k` micro-batch
gradients, and the optimizer-step count drops by ~`k`. **LR is per-iteration**
(the monotone anneal ramp in `train()` — there is no warmup; iteration 1 already
runs at the full configured LR), not per-step, so `k` does not rescale the step
size — hold LR fixed.

**Deliberately NOT one wide batch.** The advantage z-score still runs
independently on each micro-batch of `mini_batch_size` rows
(`per_iteration_advantage_norm` stays `False`), so what gets averaged is `k`
independently normalized gradients. That is the intent, not an approximation:
the group-relative binary-reward advantage is strongly asymmetric (at 12.5%
success, +2.475 for a success vs −0.354 for a failure, ~7:1), and
per-minibatch z-scoring restores symmetry. Switching to per-iteration norm to
make accumulation "exact" passes that asymmetry straight through, strips the
failure-avoidance signal, and silently pins `pos_adv_weight_k` to its 1.0 floor
(disabling PAWS) — it measured much worse on matched iterations
(success 0.125 / 0.25 / 0.083 / 0.29 vs 0.125 / 0.625 / 0.625 / 0.54).

**Edges.**
- A partial window at an epoch boundary is **flushed, never discarded**. The
  scale is a uniform `1/k`, so a flushed window of `m < k` micro-batches steps
  with `(m/k)×` the average gradient — at most one such step per epoch
  (`train/n_partial_windows`). The flush is required, not cosmetic:
  `_iter_balanced_minibatches` anchors epoch length to `ceil(n / mb_size)` but
  returns early when the majority pool drains, so the micro-batch count per
  epoch is not a multiple of `k`.
- A minibatch dropped by the non-finite guard contributes nothing AND does not
  advance the window, so every full window carries exactly `k` **trained**
  micro-batches. PAWS mass (`N_iter` / `D_iter`) likewise commits per trained
  micro-batch, so the "pooled mass == trained rows" invariant is unchanged —
  except on the rare dropped-window path below, where up to `k` micro-batches'
  mass is pooled without a weight update (accepted; `k` is a ratio, so the
  effect is second-order).
- If the accumulated gradient is non-finite the step is dropped and the window
  discarded (`n_nonfinite_grad_steps`) — see the gradient-side guard under
  "Clipped surrogate + KL". This protects a `k=1` run identically; the only
  k-specific note is that a dropped `k > 1` window forfeits up to `k`
  micro-batches of work instead of one.
- `train/n_updates` counts real `optimizer.step()` calls that actually reached
  the weights (so it drops by ~`k`, excludes dropped windows, and `did_update` /
  checkpoint naming still mean "the model moved");
  `train/n_micro_batches` counts trained mini-batches (unchanged by `k`). Every
  per-minibatch mean — `loss`, `clip_loss`, `kl_loss_*`, `clipfrac`,
  `mean_ratio`, `mean_log_ratio_abs` — divides by `n_micro_batches`, so these
  curves are **not k-inflated** and stay on the same scale across `k`. They are
  NOT bit-identical across `k`: within a window all `k` micro-batches see the
  same un-stepped weights, so the log-probs (and hence loss / ratio / clipfrac)
  shift by a few percent versus a `k=1` baseline — that is expected, not a
  regression. The `_fixed` / `_jitter` branch metrics are row-weighted
  (`sum / n_rows_*`) rather than per-minibatch means, and are likewise
  unaffected by `k`. `train/grad_norm_*` is the deliberate exception: it
  measures the ACCUMULATED gradient, so expect it to read lower at `k > 1` —
  that is noise cancelling between micro-batches, not weaker signal.
- `k = 1` is bit-identical to the pre-accumulation code path, and emits no
  `grad_accum_steps` / `n_partial_windows` curves and no banner line.

`test_grad_accum.py` covers all of the above by driving the real
`_grpo_update` / `_grpo_update_inner` on CPU. It substitutes the GPU-bound and
setup-bound pieces: `_prepare_batch` (tiny CPU tensors instead of a backbone
re-encode), `compute_fm_log_prob` (a 2-parameter analytic stand-in for the
K-loop DiT forward), the model, the episode buffer, and the optimizer (plain SGD
so a reference trajectory is exactly reproducible — production uses AdamW), and
it builds the trainer via `__new__` to skip `setup()`. The accumulation window,
guards, flush, step cadence and every metric divisor are the production ones.

```bash
# k=2: ~150 optimizer steps/iter instead of ~300, same LR, same peak VRAM
uv run python scripts/grpo/train_grpo.py \
  --gradient-accumulation-steps 2 --learning-rate 1.5e-5 \
  --group-size 12 --num-groups 4 --num-iterations 40
```

### Reference log-prob caching

`_compute_ref_log_probs` runs once per iteration, BEFORE the GRPO update,
in a `no_grad` block. It serves two purposes:

1. Captures `ref_log_prob` + `tau_samples` per chunk for reuse in the
   update.
2. **Caches per-chunk Eagle backbone + state encoder features** onto each
   `ActionChunk`. Both are frozen (no LoRA), so their output is identical
   across all `update_epochs × minibatches` in this iteration.

In `_grpo_update`, `_prepare_batch` checks if every chunk in the batch
has cached features and takes the fast path (`_rebuild_encoded_from_cache`)
— restacking cached slices instead of re-running the backbone. This is
the largest single training-time speedup in the loop.

The cache is invalidated each iteration by `buffer.clear()` (called by
`_release_memory_to_os` at iter start).

---

## Jitter-GRPO (Jacobian regularizer)

An optional, feature-flagged extension layered on top of the standard GRPO
loop. Defaults `jitter_pos = 0.0` and `jitter_neg = 0.0` are bit-identical to
vanilla GRPO; setting either (e.g. `--jitter-pos 0.05 --jitter-neg 0.05`)
activates the full mechanism.

### Motivation

Standard GRPO trains the DiT velocity field along the rolled-out denoising
trajectory: each update tightens `v_θ(x_t, t | obs)` toward `(a − ε)` at the
single point `x_t = (1−t)·ε + t·a`. Trajectories from noise samples *near* `ε`
rely entirely on architectural smoothness of the velocity field to land near
`a`. When that smoothness is poor, a successful action chunk's basin can be
narrow — the model is fragile to tiny perturbations of the inference noise.

However, the promise of Flow Matching is to be *noise-resillient* and have the
denoising velocity field push noise into good action basins, whereas today, the 
velocity field is quite sensitive to perturbations in noise, leading to fragility 
when picking between high-advantage and low-advantage actions.
In order to encourage the the velocity field to be more robust, we would like
to encourage neighboring noise to `ε` to also lead to `a`.

Jitter-GRPO adds a Frobenius-norm Jacobian penalty
`(1−t)²·λ²·‖∇_x v_θ‖_F²` *in expectation* to the existing loss, encouraging
the velocity field to be locally smooth along each rolled-out path. The
implementation is a one-line trick: feed the DiT a variance-preserving
jittered noise input `ε' = √(1−λ²)·ε + λ·ξ` (ξ ~ N(0, I)) but keep the
velocity target at the **original** `a − ε`. Taking expectation over ξ gives
the standard FM loss + the Jacobian penalty, with no double-backward and no
architecture changes. The cached `chunk.ref_log_prob` (computed at the
original ε) is reused for both branches — the cached-vs-recomputed-ref bias
is `O(λ²)` and θ-independent, so the gradient direction is unaffected.

### Scheduling: paired vs jitter-only (`jitter_paired`)

`jitter_paired` (default `True`, `--no-jitter-paired` to disable) decides how
many entries each chunk contributes per epoch when jitter is active. It is
N/A when jitter is off.

**Paired (`jitter_paired=True`, default).** Each live chunk produces TWO
entries per epoch:

```python
entries = (
    [(c, "fixed") for c in live_chunks]      # DiT input = original ε
    + [(c, "jitter") for c in live_chunks]   # DiT input = ε'
)
```

Both entries reference the **same** `ActionChunk` object (so they share
`tau_samples`, `ref_log_prob`, `initial_noise`, and the cached backbone
features). The only difference is the DiT input noise during the forward
pass: "fixed" rows use the original `ε`, "jitter" rows use
`ε' = √(1−λ²)·ε + λ·ξ`, where `λ` is `jitter_pos` for positive-advantage
chunks and `jitter_neg` for negative.

Doubling the entries list doubles the number of optimizer steps per epoch.
**Halve `update_epochs` MANUALLY** when running paired jitter (e.g., 4 → 2)
to match the per-iter optimizer-step budget of vanilla GRPO. The trainer
does not auto-halve — the relationship is left explicit so the user can
audit it from the CLI. This mode keeps the fixed-vs-jitter per-branch
diagnostic (the `mean_log_ratio_abs` gap that estimates the Jacobian norm).

**Jitter-only (`jitter_paired=False`).** Each live chunk produces ONLY its
jitter entry:

```python
entries = [(c, "jitter") for c in live_chunks]   # DiT input = ε'
```

The per-iter optimizer-step count then matches a vanilla GRPO run at the
**same** `update_epochs` — no manual halving, directly comparable curves.
The trade-off: with no "fixed" rows, the `_fixed` per-branch metrics and the
fixed-vs-jitter gap diagnostic are unavailable (only `_jitter` metrics are
emitted), and the loss trains purely on the jittered input noise. Use this
when you want an apples-to-apples step-budget comparison against a no-jitter
baseline rather than the paired diagnostic.

### `compute_fm_log_prob`: per-τ jittered input noise

`fm_log_prob.compute_fm_log_prob` gains an optional `noise_for_input` kwarg:

```python
def compute_fm_log_prob(..., noise, noise_for_input=None):
    eps = noise                       # original ε; drives velocity_target
    velocity_target = actions - eps   # ALWAYS at the ORIGINAL ε

    if noise_for_input is not None:   # required shape: [K, B, H, D]
        eps_input_all = noise_for_input
    else:
        eps_input_all = None          # back-compat fallback

    for k in range(n_samples):        # K-loop over tau_centers
        eps_input = eps if eps_input_all is None else eps_input_all[k]
        noisy_trajectory = (1 - t)*eps_input + t*actions
        # ... DiT forward, MSE per row, accumulate
```

Two design choices:

1. **`velocity_target` stays at the ORIGINAL ε.** It's `actions - noise`,
   NOT `actions - noise_for_input`. The asymmetry between input and target
   is what produces the Jacobian regularizer in expectation. Swapping the
   target to ε' would gain an `O(λ²)` model-independent floor that doesn't
   shrink as the model improves.

2. **Per-τ independent ξ_k.** The trainer already probes the FM log-prob
   at `K = len(tau_centers)` different τ values per chunk per minibatch
   (see the `tau_centers` subsection above — defaults to a length-6
   late-biased schedule). Jitter-GRPO draws ONE fresh ξ_k for each of
   those K τ-evaluations, so a paired chunk's jittered forward pass uses
   K different ε'_k = √(1−λ²)·ε + λ·ξ_k along its K τ samples. The
   caller therefore passes a 4-D `[K, B, H, D]` tensor where each
   `noise_for_input[k]` carries the ξ-jitter for one τ-evaluation. This
   gives K independent samples of the Jacobian expectation per minibatch,
   matching the variance-reduction structure of `tau_centers`. Only the
   4-D shape is supported (validated with a shape check); 3-D broadcast
   would diverge from the per-τ-fresh-ξ design.

Backward compat: when `noise_for_input=None` (the default), the function
falls back to `eps_input = eps` and the K-loop is bit-identical to the
pre-Jitter-GRPO code.

### `_iter_stratified_minibatches`: now yields entries

Refactored to operate on `list[(ActionChunk, str)]` instead of
`list[ActionChunk]`. Group binning still uses `chunk.group_id` (read off
the tuple's first element); both copies of a paired chunk share `group_id`
so they land in the same group's queue but typically end up in different
minibatches across the epoch. Yielded type: `list[(ActionChunk, str)]`.

Same deterministic shuffle behavior — with jitter off (both sides 0),
`entries = [(c, "fixed") for c in live_chunks]` has identical length and
ordering to the old `live_chunks`, and the same RNG seed produces the
same minibatch composition.

### `_prepare_batch`: carries mode through

Takes `batch: list[(ActionChunk, str)]`. The order-preserving filter
`valid_pairs = [(c, m) for (c, m) in batch if c.raw_action is not None]`
keeps modes aligned 1:1 with `valid_batch`. Returns the same `batch_data`
dict with one new key:

```python
batch_data["modes"]: list[str]   # length B, parallel to valid_batch
```

### `_compute_ref_log_probs`: always tags as "fixed"

The reference log-prob pass uses the original ε for both branches (per the
cached-ref invariant), so its single call site simply wraps the chunk list
as `[(c, "fixed") for c in batch]` before passing into `_prepare_batch`.
No `noise_for_input` is constructed; the ref pass is bit-identical
regardless of the jitter settings.

### ξ sampling and `noise_for_input` construction

Inside `_grpo_update_inner`, after `_prepare_batch` returns and the
`ready_*` slicing is done:

```python
ready_modes = [batch_data["modes"][i] for i in ready_indices]
lam_pos = self.config.jitter_pos
lam_neg = self.config.jitter_neg

if (lam_pos > 0.0 or lam_neg > 0.0) and any(m == "jitter" for m in ready_modes):
    K = len(self.config.tau_centers)
    B_r, H, D = ready_noise.shape

    # Unseeded; uses global torch RNG, matching _sample_jittered_timesteps.
    xi = torch.randn(K, B_r, H, D,
                     device=self.device, dtype=ready_noise.dtype)

    jitter_mask = torch.tensor(
        [m == "jitter" for m in ready_modes],
        device=self.device, dtype=torch.bool,
    )

    # Per-row λ by PRE-renorm advantage sign: jitter_pos for adv > 0,
    # jitter_neg otherwise. float32 keeps the scalar full-precision through
    # the sqrt/multiply (a 0.0 side collapses that row to ε).
    lam_row = torch.where(
        ready_advantages > 0,
        ready_advantages.new_full((B_r,), lam_pos, dtype=torch.float32),
        ready_advantages.new_full((B_r,), lam_neg, dtype=torch.float32),
    )
    lam_j = lam_row[jitter_mask]
    sqrt_one_minus_j = (1.0 - lam_j * lam_j).sqrt()

    # expand returns a stride-0 view; clone() materializes a writable
    # [K, B_r, H, D] tensor so __setitem__ writes per-K rows independently.
    noise_for_input = (
        ready_noise.unsqueeze(0).expand(K, -1, -1, -1).clone()
    )
    # Explicit .to(dtype): masked index-put will NOT auto-cast f32 -> bf16.
    noise_for_input[:, jitter_mask] = (
        sqrt_one_minus_j[None, :, None, None] * ready_noise[jitter_mask].unsqueeze(0)
        + lam_j[None, :, None, None] * xi[:, jitter_mask]
    ).to(ready_noise.dtype)
else:
    noise_for_input = None
```

Three notable details:

- **ξ is unseeded.** Uses the global torch RNG, matching how
  `_sample_jittered_timesteps` jitters the τ centers. On-policy collection
  noise also isn't seeded per-call, so making ξ a special case would be
  inconsistent with the rest of the training-time stochasticity. Resume
  across iters proceeds without errors but ξ values are not bit-reproducible
  across the resume boundary when jitter is active.
- **`expand+clone` is required.** `unsqueeze(0).expand(K, -1, -1, -1)`
  returns a stride-0 view across the K dim; `__setitem__` on the view would
  alias all K rows. The explicit `.clone()` materializes a writable per-K
  tensor before the assignment.
- **Fixed rows pass through unchanged.** Only
  `noise_for_input[:, jitter_mask]` is overwritten. Rows where
  `mode == "fixed"` retain `ready_noise` from the broadcast clone — ε for
  both target and input, identical to vanilla GRPO behavior.

The constructed `noise_for_input` then flows into:

```python
current_log_probs = compute_fm_log_prob(
    ..., noise=ready_noise, noise_for_input=noise_for_input,
    n_samples=len(self.config.tau_centers),
)
```

When the gate is False (both λ=0, or no jitter rows in this mb),
`noise_for_input=None` and the K-loop takes the original-ε path.

VRAM cost: `xi + noise_for_input ≈ 2 × 614 KB` per minibatch at
`K=6, B=8, H=50, D=128` in bf16. Negligible vs the DiT activations.

### Per-branch metrics (`*_fixed` / `*_jitter` TB scalars)

The KL is refactored to expose `kl_per_row_last_iter` (and optionally
`kl_per_row_base_model`) as named intermediates so they can be indexed by
branch. The final `kl_loss_last_iter = kl_coef_last_iter *
kl_per_row_last_iter.mean()` is numerically identical to the previous
inlined form.

Inside the no-grad accumulator block, **gated on `lam_pos > 0.0 or lam_neg >
0.0`**, we split the per-row tensors by mode and accumulate row-level sums:

```python
if lam_pos > 0.0 or lam_neg > 0.0:
    fixed_mask = torch.tensor([m == "fixed" for m in ready_modes], ...)
    jit_mask = ~fixed_mask

    n_f = int(fixed_mask.sum().item())
    n_j = int(jit_mask.sum().item())
    if n_f > 0:
        ratio_sum_fixed                  += ratio[fixed_mask].sum().item()
        log_ratio_abs_sum_fixed          += log_ratio_abs[fixed_mask].sum().item()
        clipfrac_sum_fixed               += int(over_clip[fixed_mask].sum().item())
        kl_per_row_sum_last_iter_fixed   += kl_per_row_last_iter[fixed_mask].sum().item()
        if compute_base:
            kl_per_row_sum_base_model_fixed += kl_per_row_base_model[fixed_mask].sum().item()
        n_rows_fixed                     += n_f
    # ... analogous for jitter
```

End-of-iter, per-branch metrics are added to `update_stats` only when at
least one row of that branch fired:

```python
if n_rows_fixed > 0:
    result["clipfrac_fixed"]                 = clipfrac_sum_fixed / n_rows_fixed
    result["mean_ratio_fixed"]               = ratio_sum_fixed / n_rows_fixed
    result["mean_log_ratio_abs_fixed"]       = log_ratio_abs_sum_fixed / n_rows_fixed
    result["kl_loss_last_iter_fixed"]        = kl_coef_last_iter * (kl_per_row_sum_last_iter_fixed / n_rows_fixed)
    if compute_base:
        result["kl_loss_base_model_fixed"]   = kl_coef_base_model * (kl_per_row_sum_base_model_fixed / n_rows_fixed)
# ... analogous for jitter
```

The gating on `lam_pos > 0.0 or lam_neg > 0.0` matters: with jitter off (both
sides 0), the per-mb accumulator block is skipped entirely, the per-branch
counters stay at their zero defaults, the result-dict gating
`if n_rows_fixed > 0:` is False, and no `_fixed`/`_jitter` keys are emitted.
Vanilla GRPO runs see exactly the same TB curves they always did.

### Advantage-sign-split ratio metrics and the *effective* clipfrac

`mean_ratio_{fixed,jitter}` pool both advantage signs, and because the gap
scales as `λ²` the two signs sit at very different ratios (at
`jitter_pos=0.25` / `jitter_neg=0.05` the biases are ≈−0.058 vs ≈−0.002), so
the pooled curve is dominated by the positive rows and neither branch is
legible. `train/{mean_ratio,mean_log_ratio_abs}_{fixed,jitter}_{pos,neg}` split
them:

- `mean_ratio_jitter_pos` starts each iteration at `e^-gap_pos` and its
  movement **up** within the iteration is headroom being consumed — the direct
  "is the positive branch learning?" readout.
- `mean_ratio_jitter_neg` starts at ≈1.0 and moves down; that is erosion.

`train/clipfrac_effective_{pos,neg}` counts rows whose **clip-term** gradient
the clamp actually zeroed, which is *not* what `clipfrac` measures. `clipfrac`
is the sign-agnostic test `(ratio < 1−lo) | (ratio > 1+hi)`, and for a
positive-advantage row that is a false positive: with `A>0` and `ρ < 1−lo`,
`min(A·ρ, A·(1−lo)) = A·ρ` — the unclamped branch wins and the gradient is
fully alive. Predicate: `clip_killed_gradient()` (module-level in
`train_grpo.py`, so tests exercise the real expression), which is
`clamp_moved & (surr2 <= surr1)`; positives can only die on the **upper** bound
and negatives only on the **lower** one.

That distinction is cosmetic at today's `jitter_pos` (observed `ratio_max` ≈
1.05) but becomes load-bearing above `jitter_pos ≈ 0.30`, where `gap_pos`
exceeds `|log(1−clip_eps_low)|` and **every** positive row reports as
"clipped" while training normally. Two caveats:

- Buckets by the **post**-renorm advantage sign (unlike the sibling
  `clipfrac_{branch}_{sign}` metrics, which use pre-renorm), because which
  bound a row can die on is decided by the sign the loss saw. Expect
  `_pos` ≡ 0 at any sane `jitter_pos`.
- Do **not** read `_neg` as a drop-in for `clipfrac_*_neg`: under
  per-minibatch renorm the two have different denominators, and a group-good
  row carrying `λ = jitter_pos` that renorm flipped negative is a genuine
  lower-bound death booked here — so a large `jitter_pos` inflates `_neg`.
  Cross-reference `n_pos_flipped_by_renorm`.

Values routed through `_log_metrics._emit` are filtered for non-finite (and
non-numeric) entries before reaching TB/wandb, because a bf16
`ratio = log_ratio.exp()` overflow reaches `ratio_sum_*` while the clipped loss
stays finite, and one `nan`/`inf` poisons wandb's chart autoscale for the rest
of the run. Note the **pre-existing** `train/loss`, `train/clipfrac`,
`train/mean_ratio` and `train/mean_log_ratio_abs` are deliberately left
unfiltered.

`test_jitter_metrics.py` covers all of the above on CPU: the gap / Jacobian /
headroom arithmetic against a closed-form stand-in whose FM residual vanishes
at `ε_in = ε` (so the Taylor expansion the estimator inverts applies), the
`clip_killed_gradient` truth table cross-checked against autograd on the real
loss, the `clipfrac_effective_*` aggregation **values** (forced dead-patterns
pin each bucket's denominator), and the θ ≡ θ_ref property functionally via
AdamW's lazily-populated `optimizer.state`.

### Splitting the jitter term into reinforcement + headroom (`gradprobe/*`)

`grad_probe_every = 0` (default) is bit-identical to a run without this: no extra
forwards, no `return_per_tau=True`, no `retain_graph=True`, no RNG consumed, no
`gradprobe/*` curves, no banner line. Asserted — including `p.grad` and
RNG-stream identity — in `test_grad_probe.py`.

**What the instrument is for.** On a positive-advantage row the loss minimises
`MSE_θ(ε′)` with `ε′ = √(1−λ²)ε + λξ`. Taking expectation over `ξ`, that single
term is **two gradients welded together**:

```
∂MSE_θ(ε′)/∂θ  =  ∂MSE_θ(ε)/∂θ  +  λ²·∂P/∂θ   =   g_R  +  λ²·g_P
```

`g_R` is REINFORCEMENT (fit this successful chunk better at its own noise);
`λ²·g_P` is the Jacobian/HEADROOM term the Jitter-GRPO regulariser adds. They
share one coefficient, so their **ratio** has never been tunable. The probe
measures it:

| Step | How |
|---|---|
| `g_jit = ∇_θ MSE_θ(ε′)` | off the graph the training forward ALREADY built (`return_per_tau=True` gives the un-averaged `[K, B]` terms, so a τ subset is takeable) |
| `g_R = ∇_θ MSE_θ(ε)` | ONE fresh clean-ε forward on ≤ `grad_probe_max_rows` rows |
| `g_head = g_jit − g_R` | formed **in place** (`g_jit.sub_(g_R)`); this IS `λ²·g_P` exactly, no Taylor assumption |
| `g_erosion` | the negative non-anchor rows of the SAME retained graph — **free**, and clean only when `jitter_neg == 0` |

The headline numbers are `R = ‖g_head‖/‖g_R‖` and `cos(g_R, g_head)`. They decide
three open questions: whether to add an AWR-on-success term and at what
coefficient (`c ≈ 0.95(R−1)` balances the blend), whether to instead **lower**
`jitter_pos` (a negative `cos_min` means the two components FIGHT, so a
counterweight is the wrong fix), and what
`positive_advantage_weight_target_ratio` is actually delivering — PAWS balances
`|A·ρ|` LOSS mass, but every positive row it amplifies drags along a `λ²g_P`
gradient component PAWS never measures.

The earlier "R ≈ 14" figure was `jitter/gap_pos / ref_mse/pos_mean`, a ratio of
**loss values** (median 13.6 over 21 iterations). It does not convert:
`‖∇MSE_θ(ε)‖ ~ 2√MSE·‖∇_θ v‖` while `‖∇λ²P‖ ~ 2λ²√P·‖∇_θ∇_x v‖`, and those
second factors are different unlogged objects. `R` was genuinely unknown,
plausibly anywhere in `[1, 15]`; this is the only way to get it.

**Sequencing — the whole reason it is affordable.** Three positions, none of them
free choices:

```
_grad_probe_capture_jittered()   BEFORE loss.backward()   (needs that graph;
                                 retain_graph=True only DELAYS the free, it does
                                 not enlarge the graph, so this costs one 58 MB
                                 vector and nothing else)
loss.backward()                  frees the training graph
_grad_probe_finish()             AFTER the free  → the two graphs never coexist
                                 BEFORE the step → g_R is at the SAME θ as g_jit
```

Both halves of that sandwich are load-bearing. A post-step clean forward would
fold one optimizer step of policy drift into `g_head`; a pre-backward clean
forward would make peak VRAM the sum of two graphs. `test_grad_probe.py` reads
the recorded θ off both legs and asserts they match, at
`gradient_accumulation_steps` 1 **and** 2 (at `k=2` the step fires on the second
micro-batch of each window, which is what an "after the window closes" placement
would corrupt).

**`torch.autograd.grad`, never `.backward()`.** `.backward()` accumulates into
`p.grad`, which is the live gradient-accumulation buffer, so a probe on it would
change the optimizer step the run takes — the instrument would alter what it
measures. `autograd.grad` returns the gradients and leaves `.grad` untouched
(`allow_unused=True`, with `None` slots materialised as zeros by
`flatten_param_grads` so the two vectors stay subtractable).

**VRAM budget**, against the measured ~21.5 GB of ~25.3 GB production peak on an
A10G at `mini_batch_size=8` (~9.7 GB base once the training graph is freed,
~0.247 GB per (row, τ) at K=6):

| Item | Cost |
|---|---|
| retained gradient vectors | 14,532,608 trainable fp32 params × 4 B = **58 MB each**; at most `g_jit` + `g_R` coexist (`g_head` is in place, `g_erosion` is reduced to its norm immediately) → ~116 MB, < 0.5 % of budget |
| `retain_graph=True` | 0 — it delays the free, it does not enlarge the graph |
| clean forward activations | `grad_probe_max_rows × |τ subset| × 0.247 GB` = **~5.9 GB** at 4 rows / full K=6, and they land AFTER the training graph is freed → ~15.6 GB peak |

`vram/grad_probe_peak_delta` reports the peak during the probe minus the peak the
surrounding update had already reached, so the claim is **verified in production**
rather than asserted here. `max_memory_allocated` is monotone within an
iteration, so this reads `0.0` whenever the probe stayed under the training
high-water mark — which is the expected outcome.

**Cost and the recommended cadence.** One probe is one clean forward+backward on
≤ 4 rows, i.e. ≈ +50 % of ONE micro-batch, so probing every Nth of ~300
micro-batches costs ≈ `50/N` %. **Use `--grad-probe-every 15` to `30`** for 10–20
probes per iteration at 1.7–3.3 % overhead. One probe per iteration is not
enough: the DISTRIBUTION is the deliverable, since `ref_mse` spans p10 0.0013 to
max 0.199 across rows and `R` plausibly varies with it.

```bash
# Production settings + the probe. Nothing about training changes.
uv run python scripts/grpo/train_grpo.py \
    --jitter-pos 0.25 --jitter-neg 0 --no-jitter-paired --update-epochs 4 \
    --grad-probe-every 20 \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 40

# Tighter VRAM (or a slower host): fewer rows and half the taus. Both legs get
# the same subset, so the reading stays valid — it is just noisier.
uv run python scripts/grpo/train_grpo.py \
    --grad-probe-every 20 --grad-probe-max-rows 2 --grad-probe-tau-subset 3
```

| Scalar | Meaning |
|---|---|
| `gradprobe/R_{mean,p10,p50,p90,max}` | `‖λ²g_P‖ / ‖g_R‖`. Above 1 the regulariser dominates the term nominally doing reinforcement. |
| `gradprobe/R_first`, `R_last` | first and last **successful** probe of the iteration. θ drifts across the ~300 micro-batches of an update; an `R` that MOVES means a fixed AWR coefficient is the wrong functional form. |
| `gradprobe/cos_reinforce_headroom`, `cos_min` | `cos(g_R, g_head)`. The min is reported beside the mean because a near-zero mean is ambiguous between "consistently orthogonal" (harmless — they do different jobs) and "fighting on some rows" (not). Negative ⇒ lower `jitter_pos` rather than adding a counterweight. |
| `gradprobe/g_reinforce_norm`, `g_headroom_norm`, `g_jit_norm`, `g_erosion_norm` | per-probe means of the four norms. All are **per-row-MEAN** gradients (the summed per-row loss is divided by its row count before `autograd.grad`), so they are comparable across micro-batches with different row counts. |
| `gradprobe/reinforce_over_erosion` | `‖g_R‖ / ‖g_erosion‖` — the **first gradient-resolved** reinforcement-vs-erosion comparison. `pos_adv_realized_ratio` is a loss-mass ratio and cannot supply it: equal loss mass does not imply equal gradient norm, because the two sides' residuals multiply different Jacobians. Present only when `jitter_neg_is_zero == 1.0`. |
| `gradprobe/jitter_neg_is_zero` | 1.0/0.0 provenance for the row above. At `jitter_neg > 0` a negative row carries its own Jacobian component, so the free erosion measurement is **omitted**, not mislabelled. |
| `gradprobe/n_probes`, `n_skipped`, `n_failed` | sample accounting. A probe is SKIPPED when the micro-batch held < 2 positive non-anchor jitter rows, when `initial_noise` is absent (the clean leg would have to sample its own ε), or when any resulting norm is non-finite / `‖g_R‖ == 0`. `n_failed` counts probes that RAISED — the metric is lost, the iteration is not (an iteration carries ~13 minutes of collected simulation by then). |
| `gradprobe/n_pos_rows_mean`, `n_neg_rows_mean`, `tau_subset_size` | what backed the readings. `n_pos_rows_mean` is the CAPPED count, so it saturates at `grad_probe_max_rows`. |
| `vram/grad_probe_peak_delta` | GB the probe added to the iteration's high-water mark. `0.0` (absent) is the expected reading. **Subtract it from `vram/per_row` before extrapolating**: `per_row = (upd_peak − fixed) / mini_batch_size` reads the raw peak, so on a probed run it attributes the probe's transient to the rows and over-states the largest feasible `mini_batch_size`. This is the only pre-existing curve the probe perturbs, and this key is exactly the correction. |

Everything routes through `_log_metrics`' `_emit`, so a non-finite scalar is
dropped with a warning rather than poisoning wandb's chart autoscale. The family
is emitted **outside** the `n_updates > 0` gate, like `jitter/*` and `drift/*`:
the probes come off micro-batches that reached `backward()`, so they survive an
iteration whose gradient windows were all dropped.

**Two scoping decisions worth knowing.**

- **Rows are the JITTERED positive non-anchor rows.** Under
  `jitter_paired=True` half the entries are "fixed" rows whose `ε′` IS `ε`, so
  their `g_head` contribution is identically zero and including them would report
  **exactly half** the true `R`, with no other symptom. With jitter fully off the
  clause is dropped and the probe correctly reads `R = 0` — a null reading, not a
  bug, and the banner says so.
- **Row selection is by PRE-renorm |advantage|**, descending, ties broken by row
  index (`sorted`, not `torch.topk`, whose tie-breaking is not part of its
  contract and differs across devices). Pre- rather than post-renorm because the
  eligibility mask is pre-renorm-keyed: a row renorm flipped negative can carry a
  large post-renorm |advantage| while actually being suppressed.

**Resolution limit.** `R` is a ratio of two nearly-equal fp32 vectors' difference
to one of them. When the probed row count differs from the micro-batch size the
clean leg reduces a differently-shaped tensor, so `g_jit − g_R` carries ~1 fp32
ULP even where the two are mathematically identical — measured `R ≈ 6e-8` at
`λ = 0` on the CPU stand-in, and higher in production where the DiT activations
are bf16. Read `R` below ~1e-3 as "indistinguishable from zero".

| File | Change |
|------|--------|
| `grpo_config.py` | Adds `grad_probe_every: int = 0`, `grad_probe_max_rows: int = 4`, `grad_probe_tau_subset: int = 0` + three hard-fail range checks in `__post_init__` (validated unconditionally, so a companion-knob typo surfaces before the feature is switched on). |
| `train_grpo.py` | Module-level `flatten_param_grads`, `select_grad_probe_rows`, `aggregate_grad_probes` (module-level so the tests exercise the real expressions). `GRPOTrainer._grad_probe_capture_jittered` / `_grad_probe_finish` are the two phases. `_grpo_update_inner` plans the probe before the forward (it selects `return_per_tau`), runs phase 1 before `backward()` and phase 2 after it, and reports via a `_grad_probe_stats()` closure on both the normal and early-return paths. `_log_metrics` emits `gradprobe/*` and splits one key to `vram/`. Startup banner when enabled. |
| `test_grad_probe.py` | New CPU suite (see the Contents table). |

### Bit-identical guarantee with jitter off (both sides `0`)

| Path | Behavior when jitter off |
|------|-----------------|
| `entries` construction | `[(c, "fixed") for c in live_chunks]` — same length and order as old `live_chunks`. |
| `_iter_stratified_minibatches` | Same RNG seed, same shuffle, same minibatch composition; yields the same chunks just wrapped in 1-tuples of `(c, "fixed")`. |
| `_prepare_batch` | Same `valid_batch` ordering; new `modes` list emitted but unused downstream. |
| `compute_fm_log_prob` | `noise_for_input=None` → `eps_input = eps` → K-loop math unchanged. |
| ξ-sampling block | Gated on `lam_pos > 0.0 or lam_neg > 0.0`; not entered. |
| Per-branch metric block | Gated on `lam_pos > 0.0 or lam_neg > 0.0`; not entered. No extra CUDA syncs from `.item()`. |
| Legacy aggregated metrics | Identical formulation; per-mb-mean accumulators preserved. |
| TB scalars | No `_fixed`/`_jitter` keys emitted; legacy TB curves byte-identical. |

Resume across iters with jitter off (both sides 0) is bit-reproducible
end-to-end. With jitter active, ξ samples are not bit-reproducible across the
resume boundary (intentional — ξ uses global torch RNG, matching τ-jitter
and on-policy collection noise).


## Operational Notes

- **GPU**: a single 24-GB+ NVIDIA GPU (training keeps frozen base in
  bf16, only LoRA params in fp32). Tested on A10G with `mini_batch_size=8`.
- **CPU/RAM**: each iteration's collector subprocess spawns
  `num_async_vector_env` MuJoCo workers (default `group_size`) for the one
  task being collected. 64+ GB RAM is comfortable for 5 workers. Lower
  `num_async_vector_env` (collecting each group over multiple turns) to fit
  larger groups on a RAM-limited host.
- **Robocasa venv**: located at
  `gr00t/eval/sim/robocasa/robocasa_uv/.venv/`. The subprocess collector
  path hard-codes this path (see `_collect_via_subprocess` in
  `train_grpo.py`); if you've put robocasa elsewhere, edit that path.
- **Memory creep**: there are small leaks in robosuite/MuJoCo's model
  reload path. The collector workers `gc.collect()` + `malloc_trim(0)`
  after every `apply_scene_bundle`; the trainer does the same at the
  start of each iter. Because a fresh collector subprocess is spawned and
  torn down every iteration, cross-iteration creep in the collector is
  bounded by construction.
- **Consecutive-failure abort**: the trainer aborts after 3 consecutive
  collector failures (timeout, non-zero exit, zero episodes loaded). The
  log line right before the abort lists common causes (wrong venv path,
  stuck port, missing MuJoCo backend, OOM).
