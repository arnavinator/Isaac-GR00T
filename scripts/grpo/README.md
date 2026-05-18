# GRPO + LoRA for GR00T N1.6 DiT

Reinforcement learning finetuning of the GR00T N1.6 diffusion action head using Group Relative Policy Optimization (GRPO) with Low-Rank Adaptation (LoRA).

## Overview

This implements GRPO — a value-function-free policy gradient method — to improve GR00T's manipulation policy on RoboCasa tasks. Instead of a learned critic (which struggles with sparse rewards), GRPO compares groups of episodes against each other to determine which actions were better.

**Key insight**: The pretrained model already achieves ~66% average success. GRPO provides a principled way to push beyond imitation learning by optimizing directly for task success.

### Why GRPO over PPO?

| Method | Value Function | Works with Sparse Rewards | Memory |
|--------|---------------|--------------------------|--------|
| PPO    | Required (hard to train with 0/1 reward) | Poorly | +2GB for critic |
| GRPO   | None needed | Well (group-relative normalization) | Lower |

### Why LoRA?

The DiT action head has 1.1B parameters. Full RL finetuning would be unstable and memory-intensive. LoRA at rank 16 adds ~20M trainable parameters (~2% of DiT), keeping the model stable while enabling adaptation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  GRPO Training Process (GPU, main .venv)                │
│                                                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Eagle VLM Backbone (frozen, shared)                │ │
│  │ Encodes images + language → vision-language embeds  │ │
│  └────────────────────────────────────────────────────┘ │
│                    │                                     │
│  ┌─────────────────┼─────────────────────────────────┐  │
│  │ DiT + LoRA      │   Pre-computed ref_log_probs    │  │
│  │ (trainable)     │   (per-chunk scalars)           │  │
│  │                 │                                  │  │
│  │  32 transformer blocks × LoRA(r=16) ≈ 20M params  │  │
│  └─────────────────┼─────────────────────────────────┘  │
│                    │                                     │
│  ┌─────────────────┼───────────┐                        │
│  │ ZMQ Server (port 5555)     │ ← serves actions +     │
│  │ + initial noise capture    │    returns initial noise │
│  └─────────────────┬──────────┘                         │
└────────────────────┼────────────────────────────────────┘
                     │
┌────────────────────┼────────────────────────────────────┐
│  Episode Collector (CPU, robocasa venv)                  │
│                    │                                     │
│  ┌─────────────────┼──────────┐  ┌───────────────────┐  │
│  │ ZMQ Client      │          │  │ group_size envs   │  │
│  │ (PolicyClient)  │          │  │ (AsyncVectorEnv)  │  │
│  └────────────────────────────┘  └───────────────────┘  │
│                                                          │
│  Saves per episode: obs, actions, raw_actions, initial_noise  │
│  → .npz files on disk                                    │
└──────────────────────────────────────────────────────────┘
```

The collector runs in a **separate Python venv** (`robocasa_uv/.venv/`) from the trainer because robocasa/robosuite/MuJoCo have version conflicts with the model deps. That venv split is why the trainer can never hold an `EpisodeCollector` directly — see [Collection Modes](#collection-modes) for how the cross-venv handshake works.

## Training Loop (per iteration)

```
1. SELECT     → Pick one task (round-robin across env_names)
2. COLLECT    → For each of num_groups groups:
                  Reset group_size envs with the SAME seed (identical initial state)
                  Run all to completion (different outcomes from policy noise)
3. ADVANTAGE  → Group-relative normalization WITHIN each group (time-scaled rewards)
4. REF LOGP   → Pre-compute reference log-probs for all chunks (single no-grad pass)
5. UPDATE     → update_epochs × minibatches of clipped ratio + KL loss
6. LOG        → TensorBoard/wandb metrics, save checkpoint
```

Each iteration trains on ONE task. With 8 tasks and 200 iterations, each task gets ~25 updates. Groups compare rollouts from the same initial state, isolating the effect of policy noise from environmental randomness.

### Fast-Forward Branching (Optional)

The early approach phase (arm moving toward the object) is mostly solved by the pretrained policy. The critical divergence happens at grasp/manipulation time. Fast-forward branching skips the approach by:

1. Running one env solo for `fast_forward_steps` outer steps
2. Saving the MuJoCo sim state at that intermediate point
3. Restoring it into all `group_size` envs (identical manipulation scenario)
4. All envs then diverge independently via policy noise

This focuses gradient signal on the high-impact manipulation phase. The `fast_forward_pct` parameter controls what fraction of groups use fast-forward (the rest start from seed) to keep the full trajectory in the training distribution.

```
fast_forward_pct=0.5 means:
  Group 1: [seed] ──────────────────────────────── full rollout (5 envs)
  Group 2: [seed] ── ff ──┬── branch rollout (5 envs from same midpoint)
  Group 3: [seed] ──────────────────────────────── full rollout (5 envs)
  Group 4: [seed] ── ff ──┬── branch rollout (5 envs from same midpoint)
  ...
```

## Collection Modes

The collector lives in the robocasa venv (separate Python interpreter from the trainer), so the trainer can't import `EpisodeCollector` directly — there's no shared address space. The trainer talks to the collector across a process boundary, in one of two modes:

### Subprocess mode (default)
Each iteration's `_collect_episodes` spawns a fresh `python collect_episodes.py` subprocess via `train_grpo.py:_collect_via_subprocess`. Simple, isolated, self-cleaning — but pays the *full startup cost every iteration*:
- Re-imports robocasa/robosuite per worker (~5-10s each)
- Spawns AsyncVectorEnv subprocess workers (~1-2s)
- Builds MuJoCo models via `gym.make` per worker (~5-10s each)

That's roughly **10-20s of overhead per iteration**, or 30-60 minutes wasted across a 200-iteration run.

### Server mode (optional)
A long-running `collector_server.py` holds one pre-initialized `EpisodeCollector` per task at startup (each with its own AsyncVectorEnv worker pool), then services per-iteration `collect` requests over ZMQ on port 5556. The trainer's `_collect_via_server` is a thin RPC client. **Startup cost is paid once at server boot, not per iteration.**

Why a ZMQ server and not just a persistent Python object? See above — the venv split forces process separation. ZMQ is the same REQ/REP + msgpack pattern that `grpo_server.py` already uses for the model server, so the trainer just becomes a client of one extra service.

The trainer's `setup()` always starts an in-process model server on `--server-port` (default 5555), so server-mode only needs **two** terminals: the long-running collector (which connects to the trainer's in-process model server) and the trainer itself. Start the collector first; it'll wait for the trainer to come up.

```bash
# Terminal 1: Long-running collector server (port 5556).
# Pre-initializes one EpisodeCollector per --env-name. --max-episode-steps,
# --group-size, --n-action-steps must match the trainer config — the trainer
# pings at __init__ and refuses to start on any mismatch. The collector
# connects to the trainer's in-process model server on --policy-server-port.
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python scripts/grpo/collector_server.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
                robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --max-episode-steps 720 480 \
    --group-size 5 --n-action-steps 8 \
    --policy-server-host 127.0.0.1 --policy-server-port 5555 \
    --listen-port 5556

# Terminal 2: Trainer (also hosts the in-process model server on 5555).
uv run python scripts/grpo/train_grpo.py \
    --collector-server-host 127.0.0.1 --collector-server-port 5556 \
    --num-iterations 200
```

> **Don't** run `scripts/grpo/grpo_server.py` standalone alongside the trainer in server mode — both would try to bind port 5555. The trainer's in-process server already plays that role. Standalone `grpo_server.py` is only useful for debugging `collect_episodes.py` CLI without the trainer.

The trainer pings the collector server at `__init__` and validates that the server's bake-time config (env_names, group_size, n_action_steps, per-env max_episode_steps) matches its own. Any mismatch raises with the exact restart command — silent collection of episodes with the wrong shape would be much worse.

### Trade-offs

| Aspect | Subprocess mode | Server mode |
|--------|-----------------|-------------|
| Per-iter startup | ~10-20s overhead | ~0s overhead |
| Setup | One terminal (trainer) | Two terminals (collector server + trainer; trainer's in-process model server fills the third role) |
| Code reload | Each iter picks up edits to `collect_episodes.py` | Restart server to pick up edits |
| Memory | Bounded per iter (process exits) | Slow growth across iters; restart server every ~50-100 iters |
| Failure isolation | One collector crash = one bad iter | Collector crash = service down until restarted |
| `max_episode_steps` change | Picked up next iter | Restart server with new `--max-episode-steps` |
| `--debug-fast-forward` verification | Direct CLI on `collect_episodes.py` (unchanged) | Direct CLI on `collect_episodes.py` (server is bypassed) |

Set `config.collector_server_host = ""` (default) for subprocess mode; set it to a non-empty host (e.g., `"127.0.0.1"`) to use the server.

## Files

| File | Purpose | Runs in |
|------|---------|---------|
| `grpo_config.py` | All hyperparameters (mirrors grpo_cont.py's init_args) | main venv |
| `lora_dit.py` | LoRA injection into DiT, save/load/merge | main venv |
| `fm_log_prob.py` | FM log-prob surrogate (replaces Gaussian dist.log_prob) | main venv |
| `episode_buffer.py` | Episode storage + group-relative advantages | main venv |
| `dense_reward.py` | Extract progress metrics from RoboCasa envs | robocasa venv |
| `collect_episodes.py` | Episode collector subprocess (one-shot CLI) | robocasa venv |
| `collector_server.py` | Long-running episode collector service (optional, see [Collection Modes](#collection-modes)) | robocasa venv |
| `grpo_server.py` | Extended model server (captures initial noise + raw action) | main venv |
| `train_grpo.py` | Main training orchestrator | main venv |

## Quick Start

### Prerequisites

- NVIDIA GPU with >= 16GB VRAM (tested on A10G 24GB)
- Model weights: `nvidia/GR00T-N1.6-3B` (auto-downloads from HuggingFace)
- Both venvs set up (main `.venv/` and `robocasa_uv/.venv/`)

### Run Training

```bash
# Full training (8 tasks, 200 iterations, ~17 hours)
# Each iteration: 1 task × 5 groups × 5 rollouts = 25 episodes
uv run python scripts/grpo/train_grpo.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --num-iterations 200 \
    --group-size 5 \
    --num-groups 5

# Quick smoke test (1 task, 3 iterations, small groups)
uv run python scripts/grpo/train_grpo.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 3 \
    --group-size 3 \
    --num-groups 4
```

### Run Components Separately (for debugging)

```bash
# Terminal 1: Start GRPO server
uv run python scripts/grpo/grpo_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON

# Terminal 2: Collect episodes manually (2 groups × 5 rollouts = 10 episodes)
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python scripts/grpo/collect_episodes.py \
    --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --group-size 5 --num-groups 2 \
    --output-dir /tmp/grpo_test

# Test LoRA injection
uv run python scripts/grpo/lora_dit.py --full-test

# Test episode buffer + advantages
uv run python scripts/grpo/episode_buffer.py

# Test dense reward classification
uv run python scripts/grpo/dense_reward.py

# Test observation key round-trip (no GPU)
uv run python scripts/grpo/test_key_roundtrip.py

# Test SimWrapper compatibility (no GPU)
uv run python scripts/grpo/test_sim_wrapper.py
```

## Checkpointing & Resume

### What's saved

Each checkpoint at `{checkpoint_dir}/iter_NNNN/` contains:
- `lora_weights.pt` — the LoRA adapter state dict (only the `lora_*` keys; ~80MB vs ~6GB for the full model)
- `optimizer.pt` — Adam moments + the trainable-parameter name list (validated against the current model's name order on load to detect a positional permutation from a `peft`/`torch` version bump)

### Save points

The trainer writes a checkpoint at any of:
- End of a successful iter `N` where `N % save_interval == 0`
- End of a **skipped** iter `N` where `N % save_interval == 0` (different naming — see below)
- End of the run (always, under the last successfully-updated iter's name)

### Resume

```bash
uv run python scripts/grpo/train_grpo.py \
    --resume-from ~/my_Isaac-GR00T/grpo_data/grpo_checkpoints/iter_0050 \
    --num-iterations 200
```

The trainer parses `NNNN` from the dir name and sets `start_iteration = NNNN + 1`. LoRA weights and optimizer moments are loaded from that dir; LR is recomputed each iter from the loop counter, so the schedule resumes cleanly.

### What "iter NNNN" actually means

**An `iter_NNNN/` directory is named after the last iter whose gradient update actually fired** — not the loop counter at the moment of the save. An iter is "skipped" (no update) when:
- Collection failed (subprocess timeout, server RPC error, zero episodes produced) → buffer empty → `std_reward = 0` → skip path, OR
- All episodes produced the same shaped reward (e.g., 0/25 success across the whole iter, or all 25 succeeded in identical step counts) → `std_reward < 1e-8` → skip path

In both cases the model weights and optimizer moments are bit-identical to what they were after the previous successful iter. Naming the resulting checkpoint `iter_<current>` would (a) burn the failed iter from the `num_iterations` budget on resume, since `start_iteration = current + 1`, and (b) give the next iter an LR one tick lower than the one the skipped iter should have used. So `_save_checkpoint_for_skipped_iter` (`train_grpo.py:1772`) names the dir after `_last_updated_iteration` instead, and skips the write if that dir already exists:

| Skip-save state | Behavior |
|---|---|
| No update has fired yet (e.g., fresh run, iter 1 collection failed) | Skip save: *"no successful update has run yet — model is still base weights"* |
| Last update was iter K, `iter_K/` doesn't exist on disk | Write `iter_K/` |
| Last update was iter K, `iter_K/` already exists | Skip save: *"resume from there to retry iter K+1"* |

This is what makes resume retry the failed iter rather than burn it from `num_iterations`.

### Worked example: iter 6 times out

Setup: `save_interval=2`, `num_iterations=200`. Iters 1–5 succeed; iter 6's collection hits the 35-minute timeout.

| Iter | Outcome | `_last_updated_iteration` | Save action |
|---|---|---|---|
| 1 | update succeeds | 1 | `1 % 2 ≠ 0` → no save |
| 2 | update succeeds | 2 | save `iter_0002/` |
| 3 | update succeeds | 3 | no save |
| 4 | update succeeds | 4 | save `iter_0004/` |
| 5 | update succeeds | 5 | no save |
| 6 | collection timeout — skip path | 5 (unchanged) | skip-save → target=5, `iter_0005/` doesn't exist → write `iter_0005/` |

`checkpoint_dir/` now contains `iter_0002/, iter_0004/, iter_0005/`. Resume from `iter_0005`:
- `start_iteration = 6` → iter 6 is **retried fresh**, not skipped past
- LR for the retry = `(1 - 5/200) × lr = 0.975 × lr` — exactly what the original (timed-out) iter 6 would have used
- The 35 minutes the timeout cost you don't *also* burn one of your 200 training iterations

If `iter_0005/` had already existed (e.g., `save_interval=1` would have written it on iter 5), the skip-save logs *"iter_0005/ already exists (resume from there to retry iter 6)"* and writes nothing — you still resume from the existing `iter_0005/` and get the same retry behavior.

### Final save

The post-loop final save (`train_grpo.py:442-454`) also writes under `iter_<_last_updated_iteration>/`, skipping if the dir already exists. A run that fails on its last few iters then ends with a checkpoint named after the last iter that actually trained — not the `num_iterations` value, which would otherwise be a misleadingly-named copy of older state.

## Hyperparameters

### Critical (tune these first)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `group_size` | 5 | G = rollouts per group (parallel envs with same seed). More = tighter advantage estimates |
| `num_groups` | 5 | Groups per iteration (different initial states). More = diverse gradients |
| `learning_rate` | 1e-5 | Lower = more stable, higher = faster but risks collapse |
| `kl_coef` | 0.005 | Higher = stays closer to pretrained, lower = more exploration |
| `clip_eps` | 0.2 | Clipping range for surrogate objective (0.1-0.3 typical) |
| `lora_rank` | 16 | Higher = more expressive (~20M params at r=16) |

### Secondary (usually fine at defaults)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `update_epochs` | 10 | More epochs = more updates per data, risk overfitting |
| `tau_centers` | [0, .25, .35, .5, .6, .75] | τ eval points for FM log-prob (K = len(list)) |
| `success_weight` | 1.0 | Balance binary reward vs dense progress signal (1.0 = binary only) |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `fast_forward_steps` | 10 | Outer steps to skip before branching (0=disabled, per-env list OK) |
| `fast_forward_pct` | 0.5 | Fraction of groups using fast-forward (rest start normally) |
| `episode_dirs_to_keep` | 3 | Number of recent `iter_*/` subdirs to retain under `episode_dir`; older dirs are pruned after each successful collection. Set to 0 to disable pruning. Bounds disk usage on `/tmp` over long runs. |
| `collector_server_host` | `""` | Empty = subprocess mode (default). Set to a host (e.g., `"127.0.0.1"`) to use a long-running `collector_server.py` and skip the per-iter startup cost. See [Collection Modes](#collection-modes). |
| `collector_server_port` | 5556 | Port of the long-running collector server. Only used when `collector_server_host` is non-empty. Distinct from the model server's port 5555. |

## Reward Shaping

The shaped reward combines binary task success with continuous progress, then time-scales to reward faster solutions:

```
shaped_reward = success_weight * success + (1 - success_weight) * max_progress
time_scaled_reward = shaped_reward / num_steps * max_episode_steps
```

- **success** (0 or 1): Did the robot complete the task?
- **max_progress** (0 to 1): How far did it get? (e.g., drawer opened 60%)
- **time_scaling**: Faster solutions get proportionally higher reward, creating variance even in all-success groups

With `success_weight=1.0` (default), the reward is purely binary but time-scaled: a success in 200 steps gets 2.6x the reward of a success in 520 steps.

Task-specific progress extraction:
- **Door/drawer**: Joint position from `get_door_state()` (0=closed, 1=fully open)
- **Pick-and-place**: `1 - normalized_distance_to_target`
- **Stove**: Knob rotation state

## FM Log-Probability Surrogate

In a Gaussian policy (like grpo_cont.py), log-prob is:
```
log π(a|s) = -0.5*log(2π) - log(σ) - 0.5*((a - μ)/σ)²
```

For a flow-matching diffusion model, we use the loss as a surrogate:
```
log π(a|s) ≈ -(1/K) Σ_k ||v_θ(x_τk, τk | s) - (a - ε)||²
```

where:
- `ε` is a single noise vector (one per action chunk evaluation)
- `x_τ = (1-τ)ε + τ*a` is the noisy interpolation at timestep τ
- `τk` are K=6 samples from tight Gaussians centered on [0, 0.25, 0.35, 0.5, 0.6, 0.75]
- `v_θ` is the model's predicted velocity field

**Key design choices:**
- **Single ε, multiple τ**: Each action was generated from one denoising trajectory. We evaluate the velocity field at multiple points along that one path, not across unrelated random paths.
- **Inference-aligned τ**: Sampling near the actual denoising timesteps (0, 0.25, 0.5, 0.75) evaluates the model where it matters most for action quality.
- **Shared (τ, ε) for ratio**: The reference log-prob is pre-computed with specific (τ, ε) values, and the current policy is evaluated with the SAME (τ, ε), so the importance ratio reflects only the model quality difference.

## Troubleshooting

### "All episodes have same reward (no gradient signal)"
- Expected for easy tasks (>90% success) or very hard tasks (<5% success)
- The round-robin task selection ensures other tasks provide signal
- Consider adjusting the task mix or increasing group size

### VRAM OOM
- Reduce `mini_batch_size` (default 8 → try 4 or 2)
- Reduce number of τ centers in `tau_centers` (e.g., down to [0.25, 0.5, 0.75])
- Lower `lora_rank` (16 → 8)

### KL divergence explodes
- Increase `kl_coef` (0.005 → 0.02)
- Decrease `learning_rate`
- Check for NaN in loss (gradient clipping issue)

### `RuntimeError: GRPOPolicyWrapper: raw_action / initial_noise capture failed`
`grpo_server.py` hooks `get_action_with_features` (to grab the raw 50×128 action) and `torch.randn` (to grab the 3-D initial noise). This error fires when one of those captures produced `None` at the end of a `get_action` call:
- **raw_action None** → `get_action_with_features` no longer returns `BatchFeature({"action_pred": …})`. Check for a model refactor on the action head return contract.
- **initial_noise None** → the denoising loop stopped calling `torch.randn(...)` with a 3-D size (e.g., switched to `torch.randn_like` or a pre-allocated buffer). Only `torch.randn` is intercepted.

Either fix the capture hook in `grpo_server.py` or update the model to restore the expected contract. This is a hard-fail by design — a silent `None` would propagate as a missing-`initial_noise` error in `_prepare_batch`, which is less obvious to debug.

### `RuntimeError: Collector failed N consecutive iterations`
The trainer aborts after 3 consecutive collection failures (timeout / non-zero subprocess exit / zero `.npz` files produced / RPC error in server mode). Without this guard, a misconfigured robocasa venv or stuck MuJoCo init would leave the trainer in a silent infinite no-op. The error message includes the last failure reason; cross-check with the `[collector] ...` lines streamed above. Common causes:
- **Robocasa venv path wrong** → `gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python` doesn't exist; rerun setup.
- **Server port stuck** → previous server thread didn't release port 5555; wait for TIME_WAIT or change `--server-port`.
- **MUJOCO_GL backend missing** → on a headless GPU host without `egl`, `gym.make` will hang; verify `MUJOCO_GL=egl` is set.
- **Server-side OOM on inference** → check trainer-side stdout for CUDA OOM during the first action-server call.
- **Server mode: `collector_server.py` died** → reason will start with `collector_server connection error` or `collector_server timeout`. Restart the server in its terminal; the trainer will reconnect on the next iteration. If the server's terminal shows a Python traceback, fix the underlying error first.
- **Server mode: env_name mismatch** → reason starts with `collector_server error: env_name 'X' not pre-initialized`. The trainer's config has an env that wasn't passed to `--env-names` at server boot. Restart the server with the full list of env_names matching the trainer config.

### Collection is slow
- **First, check if you're paying per-iter startup cost** — see [Collection Modes](#collection-modes). Server mode eliminates ~10-20s/iter overhead.
- **Then, check for swap thrashing** — see [Memory Management](#memory-management). Each worker uses ~5-6 GiB; 5 workers × 5 GiB plus trainer + system sits right at the 30 GiB RAM cliff. Once you cross it, every step pays swap I/O latency and per-group time roughly doubles.
- Increase `group_size` (more parallel envs per group, more CPU memory)
- Ensure server and collector are on same machine (ZMQ latency)
- Check if MuJoCo rendering is accidentally enabled

### `RuntimeError: LoRA checkpoint contains keys not present in the current model` (or vice versa)
Your `lora_target_modules` (or `lora_rank`) differs between the saved checkpoint and your current config. Resuming would silently load partial weights — either dropping saved adapter behavior or leaving new adapters at random init while the optimizer attaches to them. Fix by matching the config or restarting from scratch. The error message tells you which side has extra keys.

### `RuntimeError: Optimizer state shape mismatch at group 0, position N`
PEFT or PyTorch version changed between save and resume, altering the trainable-parameter traversal order. AdamW state would silently re-attach to the wrong tensors. Pin `peft` and `torch` versions across save and load, or restart training from scratch.

### `episode/kl_loss` is small but always non-negative
Expected. The KL penalty uses Schulman's k3 unbiased estimator (`E[exp(ref-current) - (ref-current) - 1]`), which is non-negative pointwise (zero exactly when current ≡ ref). If kl_loss climbs above ~0.5–1.0, the policy is drifting from ref faster than `kl_coef` can brake — either bump `kl_coef` (0.005 → 0.01) or lower `learning_rate`.

### Verifying fast-forward branching
```bash
# Run with debug mode — saves montage PNGs showing all envs after branch point
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python scripts/grpo/collect_episodes.py \
    --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --group-size 5 --num-groups 2 \
    --fast-forward-steps 10 --fast-forward-pct 1.0 \
    --debug-fast-forward \
    --output-dir /tmp/grpo_ff_test
# Inspect /tmp/grpo_ff_test/debug_ff/*.png — all rows should look identical
```

## Memory Management

Two memory leaks made longer training runs fall off a cliff: a slow trainer-side
accumulator that crept upward across iterations, and a worker-side per-call
leak that ballooned each AsyncVectorEnv subprocess during a single iter. Both
are now patched. This section documents what was wrong, where the fixes live,
and how to interpret the diagnostic logs they emit.

### The setup that exposed both leaks

Default config on a 30 GiB RAM box:

| Component | Memory at steady state |
|---|---|
| 5 AsyncVectorEnv workers (one per `group_size`) | ~5-6 GiB each → 25-30 GiB |
| Trainer process (model on GPU, CPU-side episode buffer) | ~1-2 GiB after cleanup |
| Collector parent + system processes | ~2-3 GiB |
| **Total** | **~30-35 GiB** |

The system is right at the cliff. Any extra megabyte pushes a worker into
swap, and once swap I/O contends with the model server's GPU traffic on
`/mnt/scratch/swapfile` (which also holds the HF cache), every `env.step()`
pays a page-fault tax. Per-group time roughly doubles, and the 35-min
subprocess timeout fires. Without the cleanups, both leaks compound this:
the trainer's heap grows ~3-4 GiB across iters, and each worker's
post-group baseline grows ~700-1000 MiB per group.

### Trainer-side cleanup (`train_grpo.py`)

**Where:** `_release_memory_to_os()` and `_log_mem_snapshot()`, called at
the top of every iteration in `train()` — *before* `_collect_episodes`
spawns the collector subprocess.

**What it does, in order:**

1. `self.buffer.clear()` — drops the previous iter's `EpisodeBuffer`
   (25 episodes worth of numpy arrays loaded from `.npz`, plus per-chunk
   cached backbone features). **This step is load-bearing**: until these
   references are dropped, `gc.collect()` and `malloc_trim()` cannot free
   any of the underlying memory because Python still holds them as
   reachable. The original patch ran cleanup *before* `clear()` and only
   recovered ~130 MiB; moving `clear()` to the start recovers ~3-4 GiB.
2. `gc.collect()` × 2 — breaks reference cycles between `ActionChunk`
   objects and parent episodes that hold each other via shared dicts.
   The second pass picks up any garbage produced by finalizers run during
   the first pass.
3. `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` — drains any
   pending GPU work and asks PyTorch's caching allocator to release
   unused blocks back to the driver. Not the dominant CPU savings, but
   keeps VRAM headroom predictable across iters.
4. `ctypes.CDLL("libc.so.6").malloc_trim(0)` — asks glibc to return
   freed heap pages to the kernel. Without this, the heap stays
   sticky-high even after Python has dropped all refs: glibc keeps freed
   memory in its per-thread cache for fast re-allocation, and on a memory-
   pressured box that cache competes directly with the collector workers
   for RSS.

**Why the order matters:** every step depends on the previous one having
run. Don't reorder unless you understand which step frees what (e.g.,
calling `malloc_trim` before `gc.collect()` is a no-op because nothing has
been freed yet).

### Worker-side cleanup (`collect_episodes.py`)

**Where:** module-level helpers `_read_proc_status_mb()`, `_log_worker_mem()`,
and `_release_worker_memory_to_os()`, called from inside
`GroupAlignmentWrapper.apply_scene_bundle` immediately after step 3
(`robosuite_env.reset_from_xml_string` + `sim.reset()`).

**What's leaking:** `reset_from_xml_string(xml)` builds a fresh
`MjModel` + `MjData` and replaces `robosuite_env.sim`. The previous pair
becomes orphaned, but several mechanisms keep it pinned:

- robosuite caches references to the old `MjModel` and `MjData` in its
  observable wrappers, sensor handlers, and contact processors
- MuJoCo's C-side memory pools (textures, meshes, contact preallocation
  arrays) don't always release at the glibc level even after the Python
  wrapper objects are GC'd
- EGL/GL framebuffer state from the renderer leaks across reloads

Without cleanup, each `apply_scene_bundle` call adds ~700-1000 MiB of
resident memory per worker; over 5 groups per iter that's ~3-4 GiB per
worker per iter, and across 5 workers that crosses the system's RAM
cliff into swap thrashing. The worker cleanup runs the same
`gc.collect()` × 2 + `malloc_trim(0)` recipe as the trainer-side fix
(no CUDA — workers don't have GPU state).

**Why placement matters:** the cleanup is right after step 3 because
that's the moment the old `MjModel` becomes unreachable from the live env
(steps 4-6 work exclusively on the new `MjModel`). Running it earlier
would have nothing to clean; running it later means the orphaned object
sits in memory longer and may interfere with steps 4-6's allocations.

### Diagnostic logs

Both fixes emit log lines so you can verify they're working and detect
regressions. The trainer logs at the top of every iter:

```
[mem iter 7 start (pre-release)] RSS=4923MB Swap=0MB Total=4923MB
[mem iter 7 start (post-release)] RSS=1063MB Swap=0MB Total=1063MB
```

**What to watch:** post-release `Total` should stay roughly flat across
iters (e.g., consistently ~1 GiB). If it climbs by hundreds of MB per
iter, there's another accumulator we haven't found yet — most likely
candidates are cached features on chunks, the autograd graph from
`_grpo_update`, or wandb buffering. Pre-release climbing is *expected*
(it's the buffer the cleanup is about to release).

Each worker logs three lines per `apply_scene_bundle` call (so 5 workers
× 5 groups × 3 = 75 worker_mem lines per iter, prefixed with `[collector]`
by the trainer subprocess pipe):

```
[worker_mem pid=NNNN apply_scene_bundle entry] RSS=…MB Swap=…MB Total=…MB xml=…KB
[worker_mem pid=NNNN apply_scene_bundle post-reset (pre-cleanup)] RSS=…MB Swap=…MB Total=…MB
[worker_mem pid=NNNN apply_scene_bundle post-cleanup] RSS=…MB Swap=…MB Total=…MB
```

**What to watch:**

| Comparison | Healthy | Concerning |
|---|---|---|
| `xml=` value across iters | Roughly constant per env | Drifting → procedural scene complexity varies by seed |
| `post-reset` − `entry` | Negative or small (`reset_from_xml_string` does some implicit cleanup) | Large positive → new scene materially heavier than old |
| `post-cleanup` − `entry` | ~0-100 MiB residual | 500 MiB+ residual → cleanup not catching the dominant allocation |
| `entry` (group N+1) vs `post-cleanup` (group N) | Bounded gap (~25-100 MiB drift) | Climbing → step()-time leak between groups |

### What the cleanups CAN and CANNOT do

**Can:** bound the *cross-iter* and *inter-group* residual. With both
fixes in place, the trainer stays at ~1-2 GiB indefinitely, and worker
post-cleanup baselines drift by only ~25-100 MiB per group cycle. Without
them, you eventually hit swap thrashing and the 35-min subprocess timeout.

**Cannot:** reduce the *during-episode* peak. While each worker is
running its 60-chunk × 8-substep episode loop, transient allocations
(observation rendering buffers, MuJoCo contact arrays, numpy temporaries)
push the worker's working set up by ~3-4 GiB before being released. The
cleanup catches this *between groups*, but for the duration of an episode
the memory is genuinely live. If 5 workers × peak (~8-9 GiB each) exceeds
RAM, you'll still page-fault during episodes regardless of how aggressive
the between-group cleanup is.

When that happens, the fix is upstream of the cleanup:

- **Bigger instance** — most reliable. ~64 GiB RAM removes the cliff entirely.
- **Reduce per-worker peak** — smaller render resolution, fewer cameras,
  shorter `max_episode_steps`. Cuts the transient allocation directly.
- **Raise the subprocess timeout** (`train_grpo.py`, `timeout_s = 2100`)
  to ~3300s. Doesn't fix the slowness, just lets each iter finish through
  the page-fault tax.


## References

- **GRPO**: DeepSeek-R1 (2024) — Group Relative Policy Optimization
- **DPPO**: Ren et al. (2024) — Diffusion Policy Policy Optimization (FM loss as log-prob)
- **LoRA**: Hu et al. (2021) — Low-Rank Adaptation of Large Language Models
- **GR00T N1.6**: NVIDIA (2024) — Generalist Robot 0-shot Transfer
- **Reference implementation**: `~/Desktop/ppo_cont/half_cheetah_ppo_vs_grpo/grpo_cont.py`
