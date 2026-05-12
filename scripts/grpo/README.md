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
│  │ DiT + LoRA      │   Reference DiT (frozen copy)   │  │
│  │ (trainable)     │   (for importance ratio)        │  │
│  │                 │                                  │  │
│  │  32 transformer blocks × LoRA(r=16) ≈ 20M params  │  │
│  └─────────────────┼─────────────────────────────────┘  │
│                    │                                     │
│  ┌─────────────────┼───────────┐                        │
│  │ ZMQ Server (port 5555)     │ ← serves actions +     │
│  │ + noise seed tracking      │    returns noise seeds  │
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
│  Saves per episode: obs, actions, rewards, noise seeds   │
│  → .npz files on disk                                    │
└──────────────────────────────────────────────────────────┘
```

## Training Loop (per iteration)

```
1. SELECT     → Pick one task (round-robin across env_names)
2. COLLECT    → For each of num_groups groups:
                  Reset group_size envs with the SAME seed (identical initial state)
                  Run all to completion (different outcomes from policy noise)
3. ADVANTAGE  → Group-relative normalization WITHIN each group
4. UPDATE     → update_epochs × minibatches of clipped ratio + KL loss
5. LOG        → TensorBoard/wandb metrics, save checkpoint
```

Each iteration trains on ONE task. With 7 tasks and 200 iterations, each task gets ~28 updates. Groups compare rollouts from the same initial state, isolating the effect of policy noise from environmental randomness.

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
| `grpo_server.py` | Extended model server (returns noise seeds) | main venv |
| `train_grpo.py` | Main training orchestrator | main venv |

## Quick Start

### Prerequisites

- NVIDIA GPU with >= 16GB VRAM (tested on A10G 24GB)
- Model weights: `nvidia/GR00T-N1.6-3B` (auto-downloads from HuggingFace)
- Both venvs set up (main `.venv/` and `robocasa_uv/.venv/`)

### Run Training

```bash
# Full training (all 7 tasks, 200 iterations, ~17 hours)
# Each iteration: 1 task × 12 groups × 5 rollouts = 60 episodes
uv run python scripts/grpo/train_grpo.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --num-iterations 200 \
    --group-size 5 \
    --num-groups 12

# Quick smoke test (1 task, 3 iterations, small groups)
uv run python scripts/grpo/train_grpo.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
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
```

## Hyperparameters

### Critical (tune these first)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `group_size` | 5 | G = rollouts per group (parallel envs with same seed). More = tighter advantage estimates |
| `num_groups` | 12 | Groups per iteration (different initial states). More = diverse gradients |
| `learning_rate` | 1e-5 | Lower = more stable, higher = faster but risks collapse |
| `kl_coef` | 0.01 | Higher = stays closer to pretrained, lower = more exploration |
| `clip_eps` | 0.2 | Clipping range for surrogate objective (0.1-0.3 typical) |
| `lora_rank` | 16 | Higher = more expressive (~20M params at r=16) |

### Secondary (usually fine at defaults)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `update_epochs` | 10 | More epochs = more updates per data, risk overfitting |
| `n_fm_samples` | 4 | Timestep samples per action (matches 4 inference steps) |
| `success_weight` | 0.7 | Balance binary reward vs dense progress signal |
| `ref_update_interval` | 5 | How often reference model catches up to current |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `fast_forward_steps` | 0 | Outer steps to skip before branching (0=disabled, per-env list OK) |
| `fast_forward_pct` | 0.5 | Fraction of groups using fast-forward (rest start normally) |

## Reward Shaping

The shaped reward combines binary task success with continuous progress:

```
reward = 0.7 * success + 0.3 * max_progress
```

- **success** (0 or 1): Did the robot complete the task?
- **max_progress** (0 to 1): How far did it get? (e.g., drawer opened 60%)

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
- `τk` are K=4 samples from tight Gaussians centered on the inference schedule (0, 0.25, 0.5, 0.75)
- `v_θ` is the model's predicted velocity field

**Key design choices:**
- **Single ε, multiple τ**: Each action was generated from one denoising trajectory. We evaluate the velocity field at multiple points along that one path, not across unrelated random paths.
- **Inference-aligned τ**: Sampling near the actual denoising timesteps (0, 0.25, 0.5, 0.75) evaluates the model where it matters most for action quality.
- **Shared (τ, ε) for ratio**: Both `π_θ` and `π_ref` use the same (τ, ε), so the importance ratio reflects only the model quality difference.

## Troubleshooting

### "All episodes have same reward (no gradient signal)"
- Expected for easy tasks (>90% success) or very hard tasks (<5% success)
- The round-robin task selection ensures other tasks provide signal
- Consider adjusting the task mix or increasing group size

### VRAM OOM
- Reduce `mini_batch_size` (default 8 → try 4 or 2)
- Reduce `n_fm_samples` (default 4 → try 2)
- Lower `lora_rank` (16 → 8)

### KL divergence explodes
- Increase `kl_coef` (0.01 → 0.05)
- Decrease `learning_rate`
- Increase `ref_update_interval` (slower reference drift)
- Check for NaN in loss (gradient clipping issue)

### Collection is slow
- Increase `group_size` (more parallel envs per group, more CPU memory)
- Ensure server and collector are on same machine (ZMQ latency)
- Check if MuJoCo rendering is accidentally enabled

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
