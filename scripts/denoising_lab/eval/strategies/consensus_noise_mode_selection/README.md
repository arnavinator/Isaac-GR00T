# Consensus Noise Mode Selection

**Category:** Noise selection (full denoising) | **NFEs:** K * D | **Retraining:** None

## Overview

Samples K noise candidates, fully denoises all K through D Euler steps,
then selects the candidate whose EEF trajectory is closest to the consensus
(mean) of all K candidates.

Motivated by the observation that the initial noise sample is the dominant
source of episode-level variance (ICC = 0.23, ~40% of seeds flip between
runs).  The 4-step Euler solver amplifies noise-specific variance ~70x
compared to a single denoising step, while the mean trajectory stays
approximately the same.  By sampling K candidates and picking the one
closest to the mean, we select for the "typical" action — the one most
candidates agree on.

### Key difference from `noise_space_mode_selection`

| | noise_space_mode_selection | consensus_noise_mode_selection |
|---|---|---|
| Denoising | 1-step preview for K, 3 remaining steps for winner | Full D steps for all K |
| Scoring space | Raw action space (smoothness, velocity magnitude) | Decoded EEF space (position, rotation, jerk) |
| NFEs | K + 3 | K * D |
| Scoring basis | 1-step fully-extrapolated proxy | Fully-denoised trajectories |

## Algorithm

```
Input: observation, K, D, lambda_pos, lambda_rot, lambda_jerk

1. Encode observation through Eagle VLM backbone (once)
2. Sample K noise vectors:  eps_1..eps_K ~ N(0, I),  each (B, 50, 128)
3. Denoise all K through D Euler steps (batched as K*B):
     for step in 0..D-1:
         tau = step / D
         velocity = DiT(actions, tau, VLM_features)
         actions = actions + (1/D) * velocity
4. Extract EEF trajectories from each candidate:
     pos_deltas = actions[:, :16, 0:3]
     rot_deltas = actions[:, :16, 3:6]
     cumpos = cumsum(pos_deltas)
     cumrot = cumsum(rot_deltas)
5. Compute consensus (mean across K candidates):
     mean_pos = mean(cumpos, dim=K)
     mean_rot = mean(cumrot, dim=K)
6. Score each candidate:
     pos_score  = -mean(||cumpos_k - mean_pos||^2)
     rot_score  = -mean(||cumrot_k - mean_rot||^2)
     jerk_score = -sum(||diff3(cumpos_k)||^2)
     total = lambda_pos * pos_score + lambda_rot * rot_score + lambda_jerk * jerk_score
7. Select best:  k* = argmax(total)
8. Return actions[k*]
```

## Configuration

```python
@dataclass
class ConsensusConfig:
    K: int = 8                              # noise candidates
    num_steps: int = 4                      # denoising steps
    lambda_pos: float = 1.0                 # position closeness weight
    lambda_rot: float = 0.5                 # rotation closeness weight
    lambda_jerk: float = 0.1               # jerk minimization weight
    action_horizon: int = 16                # meaningful timesteps (PandaOmron)
    eef_pos_slice: tuple[int, int] = (0, 3) # EEF position indices
    eef_rot_slice: tuple[int, int] = (3, 6) # EEF rotation indices
```

## Quick Start

### Terminal 1 — Server (model venv)

```bash
bash scripts/denoising_lab/eval/strategies/consensus_noise_mode_selection/run_server.sh
```

Override parameters:
```bash
bash scripts/denoising_lab/eval/strategies/consensus_noise_mode_selection/run_server.sh \
    --K 16 --lambda-pos 1.0 --lambda-rot 0.5 --lambda-jerk 0.1
```

### Terminal 2 — Evaluation (sim venv)

```bash
bash scripts/denoising_lab/eval/strategies/consensus_noise_mode_selection/run_eval.sh
```

### Notebook

```python
import sys
sys.path.insert(0, "scripts/denoising_lab/eval/strategies/consensus_noise_mode_selection")

from strategy import denoise_with_lab, ConsensusConfig

cfg = ConsensusConfig(K=8)
actions, best_noise, last_velocity = denoise_with_lab(lab, features, seed=42, cfg=cfg)
decoded = lab.decode_raw_actions(actions, features.states)
```

## Hyperparameter Guide

| Parameter | Range | Notes |
|-----------|-------|-------|
| `K` | 4-32 | More candidates = better consensus estimate, but K*D NFEs. K=8 is a good starting point. |
| `lambda_pos` | 0.5-2.0 | Primary scoring term. Higher values emphasize position consensus. |
| `lambda_rot` | 0.0-1.0 | Secondary term. Set to 0 to ignore rotation in scoring. |
| `lambda_jerk` | 0.0-0.5 | Smoothness regularizer. Penalizes jerky trajectories independently of consensus. |
| `action_horizon` | 16 | Set to embodiment's meaningful horizon. 16 for PandaOmron. |
