"""Collect observations from sim rollouts and save as .npz files.

Runs in the **sim venv** (robocasa).  Connects to a running GR00T server,
executes episodes, and saves a subset of observations to disk for offline
schedule calibration.

This is a helper script invoked by ``calibrate_schedule.py``.  It can also
be run standalone::

    SIM_PYTHON=gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python
    $SIM_PYTHON scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/_collect_observations.py \
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
        --n-episodes 5 --seed 42 --obs-per-episode 4 \
        --host 127.0.0.1 --port 5556 \
        --output-dir /tmp/calibration_obs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
)
from gr00t.policy.server_client import PolicyClient


# ---------------------------------------------------------------------------
# Observation helpers (mirrors robocasa_eval_benchmark.py)
# ---------------------------------------------------------------------------

def batch_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Add a leading batch dimension to a single-env observation."""
    batched: dict[str, Any] = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            batched[key] = val[np.newaxis]
        elif isinstance(val, str):
            batched[key] = (val,)
        elif isinstance(val, (tuple, list)):
            batched[key] = val
        else:
            batched[key] = val
    return batched


def unbatch_action(action: dict[str, Any]) -> dict[str, Any]:
    """Remove the leading batch dimension from a server action response."""
    unbatched: dict[str, Any] = {}
    for key, val in action.items():
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            unbatched[key] = val[0]
        else:
            unbatched[key] = val
    return unbatched


# ---------------------------------------------------------------------------
# Save in DenoisingLab-compatible format
# ---------------------------------------------------------------------------

def save_observation(obs_dict: dict[str, Any], path: Path) -> None:
    """Save a batched observation to .npz (compatible with DenoisingLab.load_observation)."""
    arrays: dict[str, Any] = {}
    metadata: dict[str, str] = {}

    for key, val in obs_dict.items():
        if isinstance(val, np.ndarray):
            arrays[key] = val
        elif isinstance(val, (tuple, list)):
            metadata[key] = (
                "|".join(str(v) for v in val) if len(val) > 1 else str(val[0])
            )
        elif isinstance(val, str):
            metadata[key] = val

    if metadata:
        arrays["__metadata__"] = np.array(json.dumps(metadata), dtype=object)

    np.savez_compressed(str(path), **arrays)


# ---------------------------------------------------------------------------
# Environment creation (same as benchmark)
# ---------------------------------------------------------------------------

def create_single_env(env_name: str, n_action_steps: int, max_episode_steps: int):
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=None, max_episode_steps=max_episode_steps),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )
    return create_eval_env(
        env_name=env_name, env_idx=0, total_n_envs=1,
        wrapper_configs=wrapper_configs,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect observations from sim rollouts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-per-episode", type=int, default=4)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to server
    client = PolicyClient(host=args.host, port=args.port, strict=False)
    if not client.ping():
        print("ERROR: Server not responding.")
        return

    env = create_single_env(args.env_name, args.n_action_steps, args.max_episode_steps)

    max_chunks = args.max_episode_steps // args.n_action_steps
    save_interval = max(1, max_chunks // args.obs_per_episode)

    # Include env name in filenames to avoid collisions when multiple envs
    # write to the same output directory.
    env_tag = args.env_name.split("/")[-1] if "/" in args.env_name else args.env_name
    obs_idx = 0

    for ep in range(args.n_episodes):
        seed = args.seed + ep
        obs, _ = env.reset(seed=seed)
        batched = batch_obs(obs)
        done = False
        step = 0

        while not done:
            # Save this observation?
            if step % save_interval == 0:
                path = output_dir / f"obs_{env_tag}_{obs_idx:04d}.npz"
                save_observation(batched, path)
                obs_idx += 1

            action, _ = client.get_action(batched)
            obs, reward, terminated, truncated, info = env.step(
                unbatch_action(action)
            )
            step += 1
            done = terminated or truncated
            if not done:
                batched = batch_obs(obs)

        print(f"  ep {ep} (seed={seed}): {step} steps, {obs_idx} obs saved so far")

    env.close()
    print(f"Saved {obs_idx} observations to {output_dir}")


if __name__ == "__main__":
    main()
