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
| `collector_server.py` | Long-running ZMQ collector daemon. Skips per-iter robocasa import + worker-spawn cost (~10-20s/iter). |
| `episode_buffer.py` | `EpisodeBuffer`, `GRPOEpisode`, `ActionChunk`. Loads `.npz` episodes, computes group-relative advantages. |
| `fm_log_prob.py` | FM-loss-as-log-prob surrogate (`compute_fm_log_prob`), jittered timestep sampler (`_sample_jittered_timesteps`). |
| `lora_dit.py` | `apply_lora_to_dit`, `save_lora_checkpoint`, `load_lora_checkpoint`, default target-module list. |
| `dense_reward.py` | Per-task continuous progress extraction (drawers, doors, PnP, stove, microwave). Used when `success_weight < 1.0`. |
| `test_*.py` | Sanity checks for sim-wrapper / `.npz` key roundtrip. |

---

## Architecture

Three processes share the work:

```
┌─────────────────────────┐   ZMQ obs/action    ┌──────────────────────────┐
│ Trainer (main .venv)    │ ◄──────────────────►│ Collector (robocasa venv)│
│  GPU model + LoRA       │  port 5555          │  AsyncVectorEnv workers  │
│  In-process PolicyServer│                     │  Writes .npz per iter    │
└────────────┬────────────┘                     └──────────────────────────┘
             │ optional ZMQ RPC (port 5556)
             ▼
┌──────────────────────────┐
│ collector_server.py      │  long-running collector for the trainer to call
│ (robocasa venv)          │  (skips per-iter startup)
└──────────────────────────┘
```

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

### 1. Subprocess mode (simpler, ~10-20s/iter startup tax)

Just run the trainer; it spawns one `collect_episodes.py` subprocess per
iteration:

```bash
uv run python scripts/grpo/train_grpo.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 200 \
    --group-size 4 --num-groups 5 \
    --checkpoint-dir grpo_data/grpo_checkpoints
```

### 2. Long-running collector mode (recommended for multi-task runs)

**Terminal 1** — start the collector daemon in the robocasa venv. It will
boot one `EpisodeCollector` per `--env-names` entry, paying the import +
worker-spawn cost once.

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/grpo/collector_server.py \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
                robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env \
    --max-episode-steps 480 480 \
    --group-size 4 --n-action-steps 8 \
    --policy-server-host 127.0.0.1 --policy-server-port 5555 \
    --listen-port 5556
```

**Terminal 2** — start the trainer, pointing it at the collector daemon:

```bash
uv run python scripts/grpo/train_grpo.py \
    --collector-server-host 127.0.0.1 --collector-server-port 5556
```

The trainer pings the collector at startup and **fails fast** if the
daemon's bake-time args (`--group-size`, `--n-action-steps`, per-env
`--max-episode-steps`, env-name set) don't match its own config. Restart
the daemon with matching flags if it does.

### 3. Standalone server (debug / eval only)

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

With 8 tasks × 200 iters, each task gets 25 updates. The
`collector_server` validates that its boot-time env list matches the
trainer's `env_names` and raises on mismatch.

### AsyncVectorEnv + scene-bundle alignment

`group_size > 1` uses `gym.vector.AsyncVectorEnv` (subprocess workers,
parallel MuJoCo). RoboCasa picks layout/textures at env construction via a
per-instance RNG, so identically-seeded parallel workers still render
**different** scenes. `GroupAlignmentWrapper` (`collect_episodes.py`)
exposes composite RPCs (`get_scene_bundle`, `apply_scene_bundle`) that the
parent invokes via `env.call()` to copy env-0's scene XML + flat MuJoCo
state to all other workers. After alignment, every env in the group is
bit-identical (verifiable via `--debug-fast-forward`).

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
- FF prefix steps are **not** counted in `episode.num_steps`, so
  time-scaled rewards compare post-branch effort fairly within a group.
- `--debug-fast-forward` saves a per-group montage of camera views to
  `<output_dir>/debug_ff/group<G>_seed<S>_ff<F>.png` so you can eyeball
  that every env in a group really is bit-identical at the branch point.

### Dynamic group collection

Many RoboCasa tasks have a wide success-rate distribution early in
training: some groups produce 0/G successes and contribute no gradient
signal (per-group reward std falls below the dead-group threshold). To
avoid wasting an iteration on a buffer with zero live signal:

```
config.num_groups = 5              # MINIMUM groups per iter (was fixed)
config.min_successful_groups = 4   # keep adding groups until ≥4 had ≥1 success
config.max_groups = 10             # hard cap on dynamic collection
```

After the first `num_groups` groups, the collector keeps adding **one
group at a time** until either:

1. `successful_groups >= min_successful_groups` (a group is "successful"
   if at least one of its rollouts succeeded), or
2. `group_idx >= max_groups` (hard cap, logs a WARNING).

To disable dynamic collection entirely, set `min_successful_groups = 0` —
the collector then always stops at exactly `num_groups`.

Constraints (enforced in `GRPOConfig.__post_init__`):

- `max_groups >= num_groups`
- `max_groups <= 100` (seed-stride collision boundary)
- `min_successful_groups <= max_groups`

Subprocess/RPC timeouts auto-scale at 7 min/group:
`timeout = 420 * effective_max_groups` seconds.

The dead-group threshold used here ("group had ≥1 success") is an
approximation for "group will produce non-zero gradient signal". The
strict condition is per-group reward std > 1e-4 — see "Minibatch
construction" below for the actual filter.

---

## GRPO Algorithm

### Per-iteration phases

```
for iteration in range(start, num_iterations + 1):
    # Phase 0: pre-flight memory cleanup
    _release_memory_to_os()                                # gc + cuda + malloc_trim

    # Phase 1: collect this iter's task
    env_name = env_names[(iter-1) % len(env_names)]
    _collect_episodes(env_name)                            # via subprocess OR RPC

    # Phase 2: compute advantages
    buffer.compute_advantages(success_weight, max_steps)   # per-group z-score

    # Phase 2b: pre-compute reference log-probs (current model == ref before update)
    _compute_ref_log_probs()                               # caches backbone features

    # Phase 3: GRPO update
    _grpo_update()                                         # update_epochs × minibatches

    # Phase 4: log + checkpoint
    if iteration % save_interval == 0: _save_checkpoint(...)
```

### Reward shaping → advantage

```
shaped = success_weight * success + (1 - success_weight) * max_progress
scaled = shaped / num_steps * max_episode_steps          # faster = better
A_episode = (scaled - group_mean) / (group_std + 1e-8)   # PER GROUP
A_chunk = A_episode / num_chunks_in_episode
```

- `success_weight = 1.0` (default) → pure binary reward, time-scaled. The
  collector skips `compute_dense_progress` entirely in this mode.
- `success_weight < 1.0` → blend in continuous progress from
  `dense_reward.py` (per-task: drawer/door joint position, PnP distance to
  target, stove knob state, etc.). Variance even in all-fail groups.
- Time-scaling (`/ num_steps * max_episode_steps`) means faster solutions
  get larger reward, creating advantage variance even within all-success
  groups.
- `A_chunk = A_episode / num_chunks` preserves the within-group
  zero-sum invariant at the chunk level, so every trajectory contributes
  equal **total** gradient weight regardless of length.

A group with reward std < 1e-4 is **dead**: its chunks get advantage
exactly 0 and are filtered out before any forward pass (see "Minibatch
construction").

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

1. **Dead-group filter**: drop every chunk with `|advantage| < 1e-12`
   (advantage was set to literal 0 by `compute_advantages` for groups with
   std < 1e-4). Filtering here keeps every minibatch uniformly live-only
   and avoids a `(0 - mean) / std` term polluting the per-minibatch
   advantage renorm.

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

### Clipped surrogate + KL

Per minibatch:

```
ratio = (current_log_prob - ref_log_prob).exp()
advantages = (A - A.mean()) / (A.std() + 1e-8)            # renorm per-batch
surr1 = A * ratio
surr2 = A * clamp(ratio, 1 - clip_eps, 1 + clip_eps)
clip_loss = -min(surr1, surr2).mean()

# Schulman k3 KL estimator (non-negative pointwise, symmetric gradient):
inv = ref_log_prob - current_log_prob
kl_loss = kl_coef * (inv.exp() - inv - 1).mean()

loss = clip_loss + kl_loss
```

NaN/Inf guard: a minibatch with non-finite loss (typically bf16 ratio
overflow when `|log_ratio|` is large) is **skipped**, the
`n_skipped_nonfinite` counter increments, and training continues.
`clip_grad_norm_` only bounds finite gradients — it does not rescue NaNs.

If ZERO minibatches commit a gradient step in an iteration (every batch
non-finite, or every group dead), the iteration is treated as **skipped**
and the resume checkpoint is saved under the last successfully-updated
iter's name (see "Checkpointing").

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

## Checkpointing & Resuming

### What gets saved

Every `save_interval` iterations into `<checkpoint_dir>/iter_NNNN/`:

```
iter_0050/
  lora_weights.pt   # only the LoRA A/B tensors (~80 MB at rank=16)
  optimizer.pt      # {"optimizer_state": ..., "param_names": [...]}
```

LoRA weights are extracted by filtering for `"lora_" in name` (works with
the low-level `inject_adapter_in_model` API, where
`get_peft_model_state_dict` is unreliable).

`optimizer.pt` bundles the AdamW state with the **ordered list of
trainable param names**. AdamW serializes its state by integer position;
a peft/torch version bump that reshuffles same-shape LoRA tensors would
silently mis-attach Adam moments without the name check.

### Resume

```bash
uv run python scripts/grpo/train_grpo.py \
    --resume-from grpo_data/grpo_checkpoints/iter_0050
```

On resume:

1. LoRA arch is rebuilt from the **current** `lora_rank` / `lora_alpha` /
   `lora_target_modules`.
2. `load_lora_checkpoint` does a strict two-sided key match and per-key
   shape check — **hard-fails** on:
   - keys in save but not in model (target_modules shrank);
   - keys in model but not in save (target_modules grew);
   - shape mismatch (rank changed).
3. `optimizer.pt` is loaded with `_validate_optimizer_param_names`
   (positional order) + `_validate_optimizer_state` (param count +
   exp_avg shape) — both raise actionable errors on mismatch.
4. `start_iteration = int(dir_name.split("_")[1]) + 1`, so the next iter
   is the one after the checkpoint.

### Skip semantics (preserves iteration budget)

An iteration is "skipped" if no `optimizer.step()` actually fired. Two
paths produce this:

1. Outer skip: global `std_reward < 1e-8` → entire buffer pruned, no
   update.
2. Inner skip: per-iter `n_updates == 0` (every minibatch non-finite, OR
   every group dead with global std still > 1e-8).

In both cases the checkpoint (if scheduled this iter) is written under
the **last successfully-updated** iter's name — not the current loop
iter. Resuming from that dir then sets `start_iteration = last + 1`,
which is exactly the skipped iter, so it gets a fresh attempt rather than
being burned from `num_iterations`. If that dir already exists (the
previous iter was itself a save-interval boundary), the write is skipped
— the on-disk state is bit-identical.

### Episode dir retention

`episode_dirs_to_keep` controls how many `iter_NNNN/` `.npz` directories
under `episode_dir` are kept around for post-mortem inspection. Default
is 3 (current + 2 prior). Pruning runs **before** the current iter's
directory is created, so the on-disk count never temporarily exceeds the
cap.

At ~0.5 GB/iter for the default config, 200 iters unpruned would burn
~100 GB; the default keeps disk under ~1.5 GB.

---

## Configuration Reference

`GRPOConfig` (in `grpo_config.py`) is the single source of truth. CLI
overrides go through `tyro`:

```bash
uv run python scripts/grpo/train_grpo.py \
    --lora-rank 32 --kl-coef 0.005 \
    --num-iterations 500 \
    --tau-centers 0.0 0.3 0.5 0.7 0.9
```

### Key knobs

**Model & LoRA**
- `model_path` (default `nvidia/GR00T-N1.6-3B`)
- `embodiment_tag` (default `ROBOCASA_PANDA_OMRON`)
- `lora_rank` / `lora_alpha` / `lora_dropout` (default 16 / 32 / 0.0)
- `lora_target_modules` — defaults to `DEFAULT_LORA_TARGET_MODULES` from
  `lora_dit.py`: 8 module patterns inside each of the 32 DiT blocks
  (`attn1.to_{q,k,v}`, `attn1.to_out.0`, `ff.net.{0.proj,2}`,
  `proj_out_{1,2}`). ~20M trainable params at rank=16.

**Episode collection**
- `group_size` (G) — rollouts per group; also the number of parallel
  envs. Default 4.
- `num_groups` — minimum groups per iter. Default 5.
- `min_successful_groups` / `max_groups` — see "Dynamic group
  collection". Default 4 / 10.
- `max_episode_steps: int | list[int]` — per-env truncation horizon.
  Default 480.
- `n_action_steps` — sub-steps to execute from each 16-step chunk.
  Default 8.
- `fast_forward_steps: int | list[int]`, `fast_forward_pct` — see
  "Fast-Forward Branching". Default 12 / 0.8.
- `env_names: list[str]` — round-robin task selection.
- `episode_dir`, `episode_dirs_to_keep`.

**ZMQ wiring**
- `server_host` / `server_port` — in-process policy server (default
  `127.0.0.1:5555`).
- `collector_server_host` / `collector_server_port` — long-running
  collector daemon. Empty host disables it (subprocess fallback).

**Reward shaping**
- `success_weight` (0-1) — binary weight in shaped reward. Default 1.0
  (pure binary, no dense progress collected).

**GRPO algorithm**
- `clip_eps` (default 0.2)
- `update_epochs` (default 5)
- `mini_batch_size` (default 8 chunks)
- `kl_coef` (default 0.1)
- `tau_centers` (default `[0.0, 0.25, 0.35, 0.5, 0.6, 0.75]`)

**Optimizer**
- `learning_rate` (default 1e-5; ~10× lower than supervised FT because RL
  gradients are noisier)
- `weight_decay` (1e-5)
- `max_grad_norm` (0.5)

LoRA params are upcast to **fp32 for AdamW** while the frozen base stays
bf16. Without this, Adam moments underflow at small lr × bf16 ULP and
the policy barely moves. PEFT's `LoraLayer.forward` handles the dtype
mismatch internally.

**Training loop**
- `num_iterations` (default 200)
- `resume_from` (default None)
- `checkpoint_dir`, `save_interval` (default every 2 iters)
- `seed` (default 67)

**Logging**
- TensorBoard writer always on; logs at `<checkpoint_dir>/tb_logs/`.
- `use_wandb` + `wandb_project` + `wandb_run_name` for optional W&B.

Logged scalars include `episode/{success_rate,mean_reward,std_reward}`,
`train/{loss,clip_loss,kl_loss,clipfrac,mean_ratio,mean_log_ratio_abs,n_skipped_nonfinite}`,
`train/learning_rate`, and `time/iteration_seconds`. `mean_log_ratio_abs`
is the primary diagnostic for DPPO-style surrogates: large values mean
the FM-MSE log-prob is noisy enough that most updates clip.

---

## Operational Notes

- **GPU**: a single 24-GB+ NVIDIA GPU (training keeps frozen base in
  bf16, only LoRA params in fp32). Tested on A10G with `mini_batch_size=8`.
- **CPU/RAM**: the collector spawns `group_size` MuJoCo workers per env;
  in long-running mode, `len(env_names) × group_size` total. 64+ GB
  RAM is comfortable for 8 tasks × 5 workers.
- **Robocasa venv**: located at
  `gr00t/eval/sim/robocasa/robocasa_uv/.venv/`. The subprocess collector
  path hard-codes this path; if you've put robocasa elsewhere, switch to
  long-running mode and just point the trainer at it.
- **Memory creep**: there are small leaks in robosuite/MuJoCo's model
  reload path. The collector workers `gc.collect()` + `malloc_trim(0)`
  after every `apply_scene_bundle`; the trainer does the same at the
  start of each iter. Even so, restart the long-running collector every
  ~50-100 iterations as a hygiene measure — the trainer's
  consecutive-failure budget will simply retry on the next iter once the
  collector is back up.
- **Consecutive-failure abort**: the trainer aborts after 3 consecutive
  collector failures (timeout, non-zero exit, zero episodes loaded). The
  log line right before the abort lists common causes (wrong venv path,
  stuck port, missing MuJoCo backend, OOM, collector not running).
- **Fatal vs transient errors**: `FatalCollectorError` (raised on
  `ValueError`-class server errors, primarily env-name mismatches) is
  re-raised immediately rather than burning the retry budget. The error
  message includes the exact restart command needed.
