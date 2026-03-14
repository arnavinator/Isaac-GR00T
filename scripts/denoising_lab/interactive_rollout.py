"""Interactive rollout — sim-side script for step-by-step env control + obs capture.

Runs in the **sim venv** (robocasa_uv/.venv). Communicates with the model server
over ZMQ via PolicyClient. NO model loading, NO torch dependency beyond what the
sim venv already provides.

Usage::

    # Terminal 1 (model venv) — start the ZMQ server
    uv run python gr00t/eval/run_gr00t_server.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --embodiment-tag ROBOCASA_PANDA_OMRON \\
        --use-sim-policy-wrapper

    # Terminal 2 (sim venv) — run interactive rollout
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
        scripts/denoising_lab/interactive_rollout.py \\
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --host 127.0.0.1 --port 5555 \\
        --n-action-steps 8 --max-episode-steps 720 \\
        --save-dir /tmp/saved_observations
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from gr00t.eval.rollout_policy import (
    WrapperConfigs,
    MultiStepConfig,
    VideoConfig,
    create_eval_env,
)
from gr00t.policy.server_client import PolicyClient


class InteractiveRollout:
    """Interactive step-by-step rollout with observation capture."""

    def __init__(
        self,
        env_name: str,
        host: str = "127.0.0.1",
        port: int = 5555,
        n_action_steps: int = 8,
        max_episode_steps: int = 720,
        save_dir: str = "/tmp/saved_observations",
    ):
        self.env_name = env_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_action_steps = n_action_steps

        # Create PolicyClient (ZMQ)
        self.client = PolicyClient(host=host, port=port, strict=False)

        # Wait for server
        print(f"Connecting to server at {host}:{port}...")
        if not self.client.ping():
            print("WARNING: Server not responding. Make sure the server is running.")
        else:
            print("Server connected.")

        # Create environment
        wrapper_configs = WrapperConfigs(
            video=VideoConfig(video_dir=None),
            multistep=MultiStepConfig(
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                terminate_on_success=True,
            ),
        )
        self.env = create_eval_env(
            env_name=env_name,
            env_idx=0,
            total_n_envs=1,
            wrapper_configs=wrapper_configs,
        )

        self.step_count = 0
        self.episode_count = 0

    def _save_observation(self, observation: dict[str, Any], tag: str = "") -> Path:
        """Save current observation to .npz file."""
        suffix = f"_{tag}" if tag else ""
        filename = f"ep{self.episode_count:03d}_step{self.step_count:03d}{suffix}.npz"
        path = self.save_dir / filename

        # Flatten for saving: wrap single-env obs in batch dim if needed
        arrays: dict[str, Any] = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value[np.newaxis] if value.ndim < 3 else value
            elif isinstance(value, (tuple, list)):
                # Language keys — store as metadata string
                arrays[key] = value
            else:
                arrays[key] = value

        # Use np.savez_compressed directly — no torch needed
        save_dict = {}
        metadata = {}
        for key, val in arrays.items():
            if isinstance(val, np.ndarray):
                save_dict[key] = val
            elif isinstance(val, (tuple, list)):
                metadata[key] = "|".join(str(v) for v in val) if len(val) > 1 else str(val[0])
            else:
                metadata[key] = str(val)

        if metadata:
            import json
            save_dict["__metadata__"] = np.array(json.dumps(metadata), dtype=object)

        np.savez_compressed(str(path), **save_dict)
        return path

    def _print_action_details(self, action: dict[str, Any]) -> None:
        """Print per-sub-step action dimensions."""
        print("\n--- Action Details ---")
        for key, arr in action.items():
            if isinstance(arr, np.ndarray):
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
                # Print first few sub-steps
                n_show = min(3, arr.shape[1] if arr.ndim >= 2 else 1)
                for t in range(n_show):
                    if arr.ndim >= 3:
                        vals = arr[0, t]
                    elif arr.ndim == 2:
                        vals = arr[t]
                    else:
                        vals = arr
                    fmt = " ".join(f"{v:+.4f}" for v in np.atleast_1d(vals))
                    print(f"    t={t}: [{fmt}]")
                if arr.ndim >= 2 and arr.shape[-2] > n_show:
                    print(f"    ... ({arr.shape[-2] - n_show} more steps)")
        print("---\n")

    def run_episode(self) -> dict[str, Any]:
        """Run one interactive episode.

        Returns:
            Episode info dict with success, length, etc.
        """
        self.step_count = 0
        obs, info = self.env.reset()

        # Wrap obs in batch dimension for PolicyClient
        batched_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                batched_obs[key] = val[np.newaxis]
            elif isinstance(val, (str,)):
                batched_obs[key] = (val,)
            elif isinstance(val, (tuple, list)):
                batched_obs[key] = val
            else:
                batched_obs[key] = val

        print(f"\n{'='*60}")
        print(f"Episode {self.episode_count} started. Env: {self.env_name}")
        print(f"{'='*60}")

        done = False
        total_reward = 0.0
        success = False

        while not done:
            # Get action from server
            action, _ = self.client.get_action(batched_obs)

            print(f"\nStep {self.step_count} | Reward so far: {total_reward:.2f}")
            print("Menu: [s]tep  [d]etails  [o]save-obs  [r]e-query  [q]uit")

            while True:
                choice = input("> ").strip().lower()

                if choice in ("s", "step", ""):
                    # Execute action
                    # Remove batch dimension for single env
                    unbatched_action = {}
                    for key, val in action.items():
                        if isinstance(val, np.ndarray) and val.ndim >= 2:
                            unbatched_action[key] = val[0]
                        else:
                            unbatched_action[key] = val

                    obs, reward, terminated, truncated, info = self.env.step(
                        unbatched_action
                    )
                    total_reward += reward
                    self.step_count += 1
                    done = terminated or truncated

                    if "success" in info:
                        ep_success = info["success"]
                        if isinstance(ep_success, (list, np.ndarray)):
                            ep_success = np.any(ep_success)
                        success = success or bool(ep_success)

                    if done:
                        print(f"\nEpisode ended. Reward: {total_reward:.2f}, Success: {success}")
                    else:
                        # Re-batch for next query
                        batched_obs = {}
                        for key, val in obs.items():
                            if isinstance(val, np.ndarray):
                                batched_obs[key] = val[np.newaxis]
                            elif isinstance(val, str):
                                batched_obs[key] = (val,)
                            elif isinstance(val, (tuple, list)):
                                batched_obs[key] = val
                            else:
                                batched_obs[key] = val
                    break

                elif choice in ("d", "details"):
                    self._print_action_details(action)

                elif choice in ("o", "save"):
                    path = self._save_observation(batched_obs)
                    print(f"Observation saved to: {path}")

                elif choice in ("r", "re-query", "requery"):
                    action, _ = self.client.get_action(batched_obs)
                    print("Re-queried server for new action.")

                elif choice in ("q", "quit"):
                    print("Quitting episode.")
                    done = True
                    break

                else:
                    print("Unknown command. Use [s]tep [d]etails [o]save [r]e-query [q]uit")

        self.episode_count += 1
        return {
            "reward": total_reward,
            "length": self.step_count,
            "success": success,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Interactive rollout with observation capture"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        required=True,
        help="Gym environment name (e.g. robocasa_panda_omron/OpenDrawer_PandaOmron_Env)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/tmp/saved_observations",
        help="Directory to save observation .npz files",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes to run (0 = loop until Ctrl-C)",
    )
    args = parser.parse_args()

    runner = InteractiveRollout(
        env_name=args.env_name,
        host=args.host,
        port=args.port,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        save_dir=args.save_dir,
    )

    episode = 0
    try:
        while args.n_episodes == 0 or episode < args.n_episodes:
            result = runner.run_episode()
            print(f"\nEpisode {episode} result: {result}")
            episode += 1

            if args.n_episodes == 0:
                cont = input("\nRun another episode? [y/n] ").strip().lower()
                if cont not in ("y", "yes", ""):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"\nFinished {episode} episodes.")


if __name__ == "__main__":
    main()
