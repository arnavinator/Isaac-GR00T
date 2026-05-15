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
│  │ (PolicyClient)  │          │  │ (SyncVectorEnv)   │  │
│  └────────────────────────────┘  └───────────────────┘  │
│                                                          │
│  Saves per episode: obs, actions, raw_actions, initial_noise  │
│  → .npz files on disk                                    │
└──────────────────────────────────────────────────────────┘
```

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

## Files

| File | Purpose | Runs in |
|------|---------|---------|
| `grpo_config.py` | All hyperparameters (mirrors grpo_cont.py's init_args) | main venv |
| `lora_dit.py` | LoRA injection into DiT, save/load/merge | main venv |
| `fm_log_prob.py` | FM log-prob surrogate (replaces Gaussian dist.log_prob) | main venv |
| `episode_buffer.py` | Episode storage + group-relative advantages | main venv |
| `dense_reward.py` | Extract progress metrics from RoboCasa envs | robocasa venv |
| `collect_episodes.py` | Episode collector subprocess | robocasa venv |
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

### Collection is slow
- Increase `group_size` (more parallel envs per group, more CPU memory)
- Ensure server and collector are on same machine (ZMQ latency)
- Check if MuJoCo rendering is accidentally enabled

### `RuntimeError: GRPOPolicyWrapper: raw_action / initial_noise capture failed`
`grpo_server.py` hooks `get_action_with_features` (to grab the raw 50×128 action) and `torch.randn` (to grab the 3-D initial noise). This error fires when one of those captures produced `None` at the end of a `get_action` call:
- **raw_action None** → `get_action_with_features` no longer returns `BatchFeature({"action_pred": …})`. Check for a model refactor on the action head return contract.
- **initial_noise None** → the denoising loop stopped calling `torch.randn(...)` with a 3-D size (e.g., switched to `torch.randn_like` or a pre-allocated buffer). Only `torch.randn` is intercepted.

Either fix the capture hook in `grpo_server.py` or update the model to restore the expected contract. This is a hard-fail by design — a silent `None` would propagate as a missing-`initial_noise` error in `_prepare_batch`, which is less obvious to debug.

### `RuntimeError: Collector failed N consecutive iterations`
The trainer aborts after 3 consecutive collection failures (timeout / non-zero subprocess exit / zero `.npz` files produced). Without this guard, a misconfigured robocasa venv or stuck MuJoCo init would leave the trainer in a silent infinite no-op. The error message includes the last failure reason; cross-check with the `[collector] ...` lines streamed above. Common causes:
- **Robocasa venv path wrong** → `gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python` doesn't exist; rerun setup.
- **Server port stuck** → previous server thread didn't release port 5555; wait for TIME_WAIT or change `--server-port`.
- **MUJOCO_GL backend missing** → on a headless GPU host without `egl`, `gym.make` will hang; verify `MUJOCO_GL=egl` is set.
- **Server-side OOM on inference** → check trainer-side stdout for CUDA OOM during the first action-server call.

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

## References

- **GRPO**: DeepSeek-R1 (2024) — Group Relative Policy Optimization
- **DPPO**: Ren et al. (2024) — Diffusion Policy Policy Optimization (FM loss as log-prob)
- **LoRA**: Hu et al. (2021) — Low-Rank Adaptation of Large Language Models
- **GR00T N1.6**: NVIDIA (2024) — Generalist Robot 0-shot Transfer
- **Reference implementation**: `~/Desktop/ppo_cont/half_cheetah_ppo_vs_grpo/grpo_cont.py`
