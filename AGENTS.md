# Isaac-GR00T — Reference

## What This Is
GR00T N1.6 is NVIDIA's 3B-parameter vision-language-action (VLA) model for robot manipulation. It takes camera images + proprioceptive state + a language instruction and outputs action chunks (future EEF deltas) via an Eagle VLM backbone + flow matching diffusion transformer (DiT) action head. Trained via offline imitation learning on demonstrations, NOT online RL.

## Repository Structure
```
gr00t/
  model/gr00t_n1d6/
    gr00t_n1d6.py               # Gr00tN1d6 (full model) + Gr00tN1d6ActionHead (DiT denoising)
    processing_gr00t_n1d6.py    # Processor: obs→model input, decode_action (50,128)→(16,29)
  model/modules/
    dit.py                      # DiT / AlternateVLDiT (32-layer diffusion transformer)
    eagle_backbone.py           # Eagle VLM backbone (encodes images + language)
    embodiment_conditioned_mlp.py  # CategorySpecificMLP, MultiEmbodimentActionEncoder
  policy/
    gr00t_policy.py             # Gr00tPolicy (inference), Gr00tSimPolicyWrapper
    server_client.py            # PolicyServer (ZMQ) and PolicyClient
  eval/
    run_gr00t_server.py         # Launches ZMQ inference server (Terminal 1) [--verbose flag]
    rollout_policy.py           # Runs sim rollouts as client (Terminal 2)
    sim/wrapper/multistep_wrapper.py  # Action chunking, episode truncation
    sim/env_utils.py            # env_name → EmbodimentTag mapping
  data/
    state_action/state_action_processor.py  # Normalization (min-max / mean-std / sin-cos)
    types.py                    # VLAStepData, ModalityConfig, ActionConfig
    embodiment_tags.py          # EmbodimentTag enum
  configs/
    model/gr00t_n1d6.py         # Gr00tN1d6Config (defaults overridden by checkpoint)
    data/embodiment_configs.py  # Posttrain modality configs (pretrain configs in checkpoint)

external_dependencies/
  robocasa/                       # Base RoboCasa (UT Austin) — MuJoCo kitchen sim
  robocasa-gr1-tabletop-tasks/
    robocasa/models/robots/__init__.py      # Key converters (PandaOmronKeyConverter, etc.)
    robocasa/utils/gym_utils/
      gymnasium_groot.py                    # GrootRoboCasaEnv gym wrapper, env registration
      gymnasium_basic.py                    # RoboCasaEnv.step() — packs action dict → flat vector

scripts/explore_vla_pipeline.py   # Single-process VLA explorer (model-only, no env needed)
```

## Model Architecture

### Two-Stage Inference
1. **Eagle VLM Backbone** (runs once per query): tokenized images + language → fused vision-language embeddings `[B, seq_len, 2048]`
2. **DiT Action Head** (runs 4 denoising steps): iteratively refines noise → action chunk

### DiT: AlternateVLDiT (32 layers, ~1.5B params of the 3B total)
- `inner_dim = 32 heads × 48 dim/head = 1536`
- Layers alternate: even=cross-attention, odd=self-attention
- Cross-attention alternates what it attends to: TEXT tokens (idx%4==0) vs IMAGE tokens (idx%4==2)
- Every layer uses **AdaLayerNorm** conditioned on the diffusion timestep embedding
- Output: `[B, 51, 1024]` → `action_decoder` MLP → `[B, 50, 128]` predicted velocity

### Denoising Loop (Flow Matching, NOT DDPM/DDIM)
- 4 Euler steps with `dt=0.25`, timestep schedule: 0 → 250 → 500 → 750
- Each step: DiT predicts velocity, `actions = actions + 0.25 * velocity`
- Raw output shape: `(B, 50, 128)` — padded for multi-embodiment support
- `decode_action()` slices to per-embodiment dims: e.g., Panda → `(B, 16, 29)`

### Training vs Inference
- **`forward()`** (training): single random timestep, MSE loss on predicted vs true velocity. Ground-truth actions required.
- **`get_action()`** (inference): 4-step denoising loop from pure noise. `@torch.no_grad()`.

### Multi-Embodiment Support
- Padded to max dims: 50 timesteps, 128 action dims, 29 state dims (actual values from checkpoint, not code defaults)
- `action_mask` zeros out padded dims during training loss
- `action_encoder` / `action_decoder` use `CategorySpecificLinear` — separate weight matrices per embodiment, indexed by `embodiment_id` (e.g., 13 for Panda)
- Per-embodiment modality configs (cameras, state keys, action keys, `delta_indices`) stored in checkpoint, not in repo

## Observation Format (PandaOmron)
| Modality | Keys | Details |
|----------|------|---------|
| Video | `res256_image_side_0`, `res256_image_side_1`, `res256_image_wrist_0` | 3 cameras: 2 exocentric side + 1 wrist. Captured 512×512, resized to 256×256 |
| State | `gripper_qpos`, `base_position`(3), `base_rotation`(4), `end_effector_position_relative`(3), `end_effector_rotation_relative`(4) | ~16 floats total. EEF-centric, NO joint angles, NO velocities |
| Language | `annotation.human.action.task_description` | e.g., "open the right drawer". Set at reset, constant per episode |

## Action Format (PandaOmron)
Model outputs are **EEF deltas** (Operational Space Control), NOT joint angles or torques:
| Key | Dims | Description |
|-----|------|-------------|
| `end_effector_position` | 3 | Delta XYZ of gripper |
| `end_effector_rotation` | 3 | Delta axis-angle rotation |
| `gripper_close` | 1 | 0/1 discrete (thresholded at 0.5) |
| `base_motion` | 4 | 3D base velocity + 1D torso height |
| `control_mode` | 1 | 0=arm mode, 1=navigation mode (mutually exclusive via HybridMobileBase) |

- `unmap_action()` renames these to robosuite keys (`robot0_right`, `robot0_base`, etc.)
- `gymnasium_basic.py:step()` packs into flat vector for robosuite's OSC controller
- OSC controller internally computes joint torques via Jacobian inverse kinematics
- Training data uses 0/1 encoding for discrete dims; model learns in [0,1], thresholded at 0.5

## Normalization
- **States & actions**: per-dimension min-max normalization to `[-1, 1]`, clipped. Statistics from training data, stored in checkpoint.
- **Denormalization**: `(clipped_value + 1) / 2 * (max - min) + min`
- Some embodiments use mean-std (z-score) or sin-cos encoding instead (configured per modality key)

## Action Chunking
- Model predicts 16 future timesteps (for Panda; varies per embodiment via `delta_indices`)
- `--n_action_steps 8` means `MultiStepWrapper.step()` executes steps 0-7, discards 8-15
- After executing 8 sub-steps, new observation collected and fresh chunk predicted

## Architecture: Server-Client Evaluation
Two terminals, two separate venvs (robocasa not in main venv):
- **Server** (main `.venv/`): loads model on GPU, serves actions over ZMQ port 5555
- **Client** (sim-specific venv): runs simulator, sends observations, executes returned actions

## Key Commands

### Run Evaluation — RoboCasa Panda
```bash
# Terminal 1 — server (uses main .venv, add --verbose for denoising logs)
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper --verbose

# Terminal 2 — client (uses robocasa venv)
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
  --n_episodes 10 --policy_client_host 127.0.0.1 --policy_client_port 5555 \
  --max_episode_steps 720 --n_action_steps 8 --n_envs 5 \
  --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env
```

### Fine-tuning
```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /path/to/lerobot_v2_dataset \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --num-gpus 1 --output-dir /tmp/finetuned --max-steps 2000
```

## Key Embodiment Tags
- `ROBOCASA_PANDA_OMRON` (id=13) — Panda arm + Omron mobile base + gripper (3 cameras, 16-step horizon)
- `GR1` (id=20) — GR1 humanoid arms+waist with Fourier hands (1 ego camera)
- `BEHAVIOR_R1_PRO` (id=24) — Galaxea R1 Pro (Isaac Sim, 32-step horizon, needs RT-core GPU)
- `LIBERO_PANDA` (id=2), `OXE_GOOGLE` (id=0), `OXE_WIDOWX` (id=1), `UNITREE_G1` (id=8)

## Environment Behavior
- Sparse reward: 1.0 on success, 0.0 otherwise
- `ignore_done=True` — env never self-terminates; truncation by MultiStepWrapper at `max_episode_steps`
- `terminate_on_success=True` in eval — episode ends early on success
- `--n_envs` creates parallel envs via `AsyncVectorEnv`; episodes distributed across envs
- Videos saved to `/tmp/sim_eval_videos_*/` (RoboCasa only)

## Hardware Requirements
- Server: NVIDIA GPU >=16GB VRAM, CUDA, Linux (bfloat16, flash-attn)
- RoboCasa client: CPU sufficient (MuJoCo + EGL headless)
- BEHAVIOR client: needs RT-core GPU (L40/RTX 4090; NOT A100/H100)
