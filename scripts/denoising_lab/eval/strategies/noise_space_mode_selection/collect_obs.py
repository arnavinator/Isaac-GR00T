"""Collect observations from a sim env and save to disk as pickle.

This helper runs in the robocasa venv (which has robosuite/robocasa installed).
The main profiler script launches this as a subprocess and loads the results.

Usage (robocasa venv):
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
        scripts/denoising_lab/eval/strategies/noise_space_mode_selection/collect_obs.py \
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
        --n-obs 10 --seed 42 \
        --output-path /tmp/obs_cache.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
)


def batch_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Add a leading batch dimension to a single-env observation."""
    batched: dict[str, Any] = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            batched[key] = val[np.newaxis]
        elif isinstance(val, str):
            batched[key] = (val,)
        else:
            batched[key] = val
    return batched


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sim observations to disk")
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--n-obs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=None, max_episode_steps=args.max_episode_steps),
        multistep=MultiStepConfig(
            n_action_steps=8,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=False,
        ),
    )
    env = create_eval_env(
        env_name=args.env_name, env_idx=0, total_n_envs=1,
        wrapper_configs=wrapper_configs,
    )

    observations = []
    obs, _ = env.reset(seed=args.seed)
    observations.append(batch_obs(obs))

    for i in range(args.n_obs - 1):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(batch_obs(obs))
        if terminated or truncated:
            obs, _ = env.reset(seed=args.seed + len(observations))

    env.close()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(observations, f)

    print(f"Saved {len(observations)} observations to {output_path}")


if __name__ == "__main__":
    main()
