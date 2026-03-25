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

    # Replay an action chunk (after exporting from the notebook)
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
        scripts/denoising_lab/interactive_rollout.py \\
        --replay \\
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --obs-path /tmp/saved_observations/ep000_step001.npz \\
        --action-path /tmp/action_chunks/my_actions.npz \\
        --video-out /tmp/replay.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

from gr00t.eval.rollout_policy import (
    WrapperConfigs,
    MultiStepConfig,
    VideoConfig,
    create_eval_env,
)
from gr00t.policy.server_client import PolicyClient


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types in ep_meta dicts."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


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
        video_dir: str | None = None,
    ):
        self.env_name = env_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_action_steps = n_action_steps
        self.video_dir = video_dir

        # Create PolicyClient (ZMQ)
        self.client = PolicyClient(host=host, port=port, strict=False)

        # Wait for server
        print(f"Connecting to server at {host}:{port}...")
        if not self.client.ping():
            print("WARNING: Server not responding. Make sure the server is running.")
        else:
            print("Server connected.")

        # Create environment (with optional video recording)
        wrapper_configs = WrapperConfigs(
            video=VideoConfig(
                video_dir=video_dir,
                max_episode_steps=max_episode_steps,
            ),
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

    def _get_sim_state(self) -> np.ndarray | None:
        """Try to get the MuJoCo simulation state (qpos/qvel) for replay."""
        try:
            # Navigate wrapper chain: gym wrappers → GrootRoboCasaEnv → RoboCasaEnv → robosuite env
            base_env = self.env.unwrapped  # gymnasium base
            robosuite_env = base_env.env    # robosuite env stored in RoboCasaEnv.env
            return np.array(robosuite_env.sim.get_state().flatten())
        except (AttributeError, TypeError):
            return None

    def _get_ep_meta(self) -> dict | None:
        """Get robosuite episode metadata (layout_id, style_id, etc.) for replay.

        The ep_meta dict captures all the random choices made during
        ``_load_model()`` — kitchen layout, style, object assets, camera poses —
        so that ``set_ep_meta()`` + ``reset()`` can reproduce the exact same
        scene geometry.
        """
        try:
            base_env = self.env.unwrapped
            robosuite_env = base_env.env
            return robosuite_env.get_ep_meta()
        except (AttributeError, TypeError):
            return None

    def _save_camera_snapshot(self, observation: dict[str, Any], path: Path) -> None:
        """Save a camera montage PNG from the observation."""
        frames = []
        for key in sorted(observation.keys()):
            if not key.startswith("video."):
                continue
            val = observation[key]
            if not isinstance(val, np.ndarray):
                continue
            # Skip 512 resolution duplicates — keep 256 for compact snapshots
            if "res512" in key:
                continue
            img = val
            while img.ndim > 3:
                img = img[0]
            frames.append(img)

        if not frames:
            return

        # Resize to common height and concatenate horizontally
        target_h = min(f.shape[0] for f in frames)
        resized = []
        for f in frames:
            if f.shape[0] != target_h:
                scale = target_h / f.shape[0]
                new_w = int(f.shape[1] * scale)
                f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized.append(f)
        montage = np.concatenate(resized, axis=1)

        # Save as PNG (RGB → BGR for cv2)
        cv2.imwrite(str(path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    def _save_observation(self, observation: dict[str, Any], tag: str = "") -> Path:
        """Save current observation to .npz file, with sim state and camera snapshot."""
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

        # Save MuJoCo sim state for replay
        sim_state = self._get_sim_state()
        if sim_state is not None:
            arrays["__sim_state__"] = sim_state

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
            save_dict["__metadata__"] = np.array(json.dumps(metadata), dtype=object)

        # Save episode metadata (layout_id, style_id, object configs, etc.)
        # as a separate JSON blob so replay can restore the exact kitchen layout.
        ep_meta = self._get_ep_meta()
        if ep_meta is not None:
            save_dict["__ep_meta__"] = np.array(
                json.dumps(ep_meta, cls=_NumpyEncoder), dtype=object
            )

        np.savez_compressed(str(path), **save_dict)

        # Save camera snapshot PNG alongside the .npz
        snapshot_path = path.with_suffix(".png")
        self._save_camera_snapshot(observation, snapshot_path)

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

        # Detect language key and initialize prompt tracking
        language_key = None
        for key, val in obs.items():
            if isinstance(val, str):
                language_key = key
                break
        original_prompt = obs.get(language_key, "") if language_key else ""
        prompt_override = None  # None = use env default
        prompt_history: list[tuple[int, str]] = [(0, original_prompt)]
        if language_key:
            print(f"Text prompt: '{original_prompt}'")

        done = False
        total_reward = 0.0
        success = False

        while not done:
            # Get action from server
            action, _ = self.client.get_action(batched_obs)

            print(f"\nStep {self.step_count} | Reward so far: {total_reward:.2f}")
            if prompt_override is not None:
                print(f"  Prompt (modified): '{prompt_override}'")
            print("Menu: [s]tep  [so]save+step  [d]etails  [o]save-obs  [m]odify-text  [r]e-query  [q]uit")

            while True:
                choice = input("> ").strip().lower()

                if choice in ("s", "step", "", "so", "save-step"):
                    # Save observation first if requested
                    if choice in ("so", "save-step"):
                        path = self._save_observation(batched_obs)
                        print(f"Observation saved to: {path}")
                        print(f"Camera snapshot: {path.with_suffix('.png')}")
                        sim_state = self._get_sim_state()
                        if sim_state is not None:
                            print(f"Sim state saved ({sim_state.shape[0]} floats) — replay-ready")
                        ep_meta = self._get_ep_meta()
                        if ep_meta is not None:
                            print(f"Layout saved: layout_id={ep_meta.get('layout_id')}, "
                                  f"style_id={ep_meta.get('style_id')}")

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
                        # Apply persistent prompt override
                        if prompt_override is not None and language_key is not None:
                            batched_obs[language_key] = (prompt_override,)
                    break

                elif choice in ("d", "details"):
                    self._print_action_details(action)

                elif choice in ("o", "save"):
                    path = self._save_observation(batched_obs)
                    print(f"Observation saved to: {path}")
                    print(f"Camera snapshot: {path.with_suffix('.png')}")
                    sim_state = self._get_sim_state()
                    if sim_state is not None:
                        print(f"Sim state saved ({sim_state.shape[0]} floats) — replay-ready")
                    ep_meta = self._get_ep_meta()
                    if ep_meta is not None:
                        print(f"Layout saved: layout_id={ep_meta.get('layout_id')}, "
                              f"style_id={ep_meta.get('style_id')}")

                elif choice in ("r", "re-query", "requery"):
                    action, _ = self.client.get_action(batched_obs)
                    print("Re-queried server for new action.")

                elif choice in ("m", "modify-text", "modify"):
                    if language_key is None:
                        print("No language key found in observation.")
                    else:
                        print("\n--- Text Prompt History ---")
                        for i, (step, prompt) in enumerate(prompt_history):
                            if i == 0:
                                label = f"step {step} (original)"
                            else:
                                label = f"step {step}"
                            print(f"  [{label}]: '{prompt}'")
                        current = prompt_override if prompt_override is not None else original_prompt
                        print(f"\n  Current prompt: '{current}'")
                        print("---")
                        new_prompt = input("Enter new prompt (or press Enter to keep): ").strip()
                        if new_prompt:
                            prompt_override = new_prompt
                            prompt_history.append((self.step_count, new_prompt))
                            batched_obs[language_key] = (new_prompt,)
                            # Re-query so the next [s]tep uses the new prompt
                            action, _ = self.client.get_action(batched_obs)
                            print(f"Prompt changed to: '{new_prompt}' — action re-queried.")
                        else:
                            print("Prompt unchanged.")

                elif choice in ("q", "quit"):
                    print("Quitting episode.")
                    done = True
                    break

                else:
                    print("Unknown command. Use [s]tep [so]save+step [d]etails [o]save [m]odify-text [r]e-query [q]uit")

        self.episode_count += 1
        return {
            "reward": total_reward,
            "length": self.step_count,
            "success": success,
        }


class ReplayRollout:
    """Replay an action chunk from a saved sim state, recording video.

    This allows you to export action chunks from the notebook (different
    denoising strategies) and visualize them in the actual simulator.

    The workflow:
    1. In interactive rollout, press ``[o]`` to save observation + sim state.
    2. In the notebook, load the observation, try denoising strategies, and
       call ``DenoisingLab.save_action_chunk(decoded, path)`` to export.
    3. Run this replay to restore the sim state, execute the action chunk,
       and produce a video.
    """

    def __init__(
        self,
        env_name: str,
        obs_path: str,
        action_path: str,
        video_out: str,
        n_action_steps: int = 8,
        max_episode_steps: int = 720,
    ):
        self.env_name = env_name
        self.obs_path = Path(obs_path)
        self.action_path = Path(action_path)
        self.video_out = Path(video_out)
        self.video_out.parent.mkdir(parents=True, exist_ok=True)

        # Create environment (no video wrapper — we record manually)
        wrapper_configs = WrapperConfigs(
            video=VideoConfig(video_dir=None),
            multistep=MultiStepConfig(
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                terminate_on_success=False,  # don't terminate early during replay
            ),
        )
        self.env = create_eval_env(
            env_name=env_name,
            env_idx=0,
            total_n_envs=1,
            wrapper_configs=wrapper_configs,
        )

    def _get_robosuite_env(self):
        """Navigate wrapper chain to the robosuite env."""
        base_env = self.env.unwrapped
        return base_env.env

    def _restore_sim_state(self, sim_state: np.ndarray) -> None:
        """Restore MuJoCo simulation state from a flattened array."""
        robosuite_env = self._get_robosuite_env()
        robosuite_env.sim.set_state_from_flattened(sim_state)
        robosuite_env.sim.forward()
        if hasattr(robosuite_env, "update_state"):
            robosuite_env.update_state()
        elif hasattr(robosuite_env, "update_sites"):
            robosuite_env.update_sites()

    def _collect_camera_frames(self, obs: dict[str, Any]) -> list[np.ndarray]:
        """Extract camera frames from an observation dict."""
        frames = []
        for key in sorted(obs.keys()):
            if not key.startswith("video."):
                continue
            if "res512" in key:
                continue
            val = obs[key]
            if isinstance(val, np.ndarray):
                img = val
                while img.ndim > 3:
                    img = img[0]
                frames.append(img)
        return frames

    def run(self) -> Path:
        """Execute the replay and save the video.

        Returns:
            Path to the saved video file.
        """
        import av

        # Load observation .npz (contains sim state + ep_meta for layout replay)
        data = dict(np.load(str(self.obs_path), allow_pickle=True))
        if "__sim_state__" not in data:
            raise ValueError(
                f"No sim state found in {self.obs_path}. "
                "Re-save the observation using the updated interactive_rollout.py."
            )
        sim_state = data["__sim_state__"]

        # Load episode metadata (layout_id, style_id, object configs, etc.)
        # so the reset rebuilds the exact same kitchen instead of randomizing.
        ep_meta = None
        if "__ep_meta__" in data:
            ep_meta = json.loads(str(data["__ep_meta__"]))

        # Load action chunk
        action_data = dict(np.load(str(self.action_path), allow_pickle=True))
        print(f"Action chunk keys: {list(action_data.keys())}")
        for k, v in action_data.items():
            print(f"  {k}: shape={v.shape}")

        # Inject ep_meta BEFORE reset so _load_model() builds the same kitchen
        robosuite_env = self._get_robosuite_env()
        if ep_meta is not None:
            robosuite_env.set_ep_meta(ep_meta)
            print(f"Restored ep_meta: layout_id={ep_meta.get('layout_id')}, "
                  f"style_id={ep_meta.get('style_id')}")
        else:
            print("WARNING: No ep_meta in saved observation — kitchen layout "
                  "will be randomized. Re-save with the updated interactive_rollout.py "
                  "for deterministic replay.")

        # Reset env (uses saved layout via ep_meta), then restore exact state
        print("Resetting environment...")
        obs, info = self.env.reset()
        print("Restoring sim state...")
        self._restore_sim_state(sim_state)

        # Re-read observation after state restore to get correct camera frames
        # Chain: robosuite raw obs → get_basic_observation → get_groot_observation
        robosuite_env = self._get_robosuite_env()
        raw_obs = robosuite_env._get_observations()
        base_env = self.env.unwrapped
        basic_obs = base_env.get_basic_observation(raw_obs)
        obs = base_env.get_groot_observation(basic_obs)

        # Collect initial frames
        all_frames: list[np.ndarray] = []
        cam_frames = self._collect_camera_frames(obs)
        if cam_frames:
            # Resize to common height and concatenate
            target_h = min(f.shape[0] for f in cam_frames)
            resized = []
            for f in cam_frames:
                if f.shape[0] != target_h:
                    scale = target_h / f.shape[0]
                    new_w = int(f.shape[1] * scale)
                    f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized.append(f)
            montage = np.concatenate(resized, axis=1)
            all_frames.append(montage)

        # Step through action chunk sub-steps
        # The action chunk has shape (B, T, dim) — remove batch dim, then iterate
        action_horizon = list(action_data.values())[0].shape[1]
        print(f"Stepping through {action_horizon} action sub-steps...")

        for t in range(action_horizon):
            single_step_action = {}
            for key, arr in action_data.items():
                # Add "action." prefix — GrootRoboCasaEnv.step() calls
                # unmap_action() which expects keys like "action.end_effector_position".
                # The notebook's decode_raw_actions() returns unprefixed keys, and
                # Gr00tSimPolicyWrapper._get_action() adds the prefix at line 618.
                prefixed_key = f"action.{key}" if not key.startswith("action.") else key
                single_step_action[prefixed_key] = arr[0, t]  # (dim,)

            # Step the base env (single sub-step, bypassing MultiStepWrapper)
            try:
                base_obs, reward, terminated, truncated, step_info = base_env.step(
                    single_step_action
                )
            except Exception as e:
                print(f"  Step {t}: env.step() failed: {e}")
                break

            success = step_info.get("success", False)
            print(f"  Step {t}: reward={reward:.3f} success={success}")

            cam_frames = self._collect_camera_frames(base_obs)
            if cam_frames:
                target_h = min(f.shape[0] for f in cam_frames)
                resized = []
                for f in cam_frames:
                    if f.shape[0] != target_h:
                        scale = target_h / f.shape[0]
                        new_w = int(f.shape[1] * scale)
                        f = cv2.resize(
                            f, (new_w, target_h), interpolation=cv2.INTER_LINEAR
                        )
                    resized.append(f)
                montage = np.concatenate(resized, axis=1)
                all_frames.append(montage)

        # Write video
        if not all_frames:
            print("No frames captured — cannot write video.")
            return self.video_out

        print(f"Writing {len(all_frames)} frames to {self.video_out}...")
        h, w = all_frames[0].shape[:2]
        # H264 requires even dimensions
        h = h - (h % 2)
        w = w - (w % 2)
        container = av.open(str(self.video_out), mode="w")
        stream = container.add_stream("h264", rate=10)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"crf": "18"}

        for frame_arr in all_frames:
            # Resize to target even dimensions for H264 compatibility
            if frame_arr.shape[:2] != (h, w):
                frame_arr = cv2.resize(frame_arr, (w, h))
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

        print(f"Video saved to: {self.video_out}")
        return self.video_out


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
        "--video-dir",
        type=str,
        default=None,
        help="Directory to save episode videos (None = no video recording)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes to run (0 = loop until Ctrl-C)",
    )

    # Replay mode arguments
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay mode: load saved sim state + action chunk, record video",
    )
    parser.add_argument(
        "--obs-path",
        type=str,
        default=None,
        help="(replay mode) Path to observation .npz with saved sim state",
    )
    parser.add_argument(
        "--action-path",
        type=str,
        default=None,
        help="(replay mode) Path to action chunk .npz exported from notebook",
    )
    parser.add_argument(
        "--video-out",
        type=str,
        default="/tmp/replay.mp4",
        help="(replay mode) Output video path",
    )

    args = parser.parse_args()

    if args.replay:
        # Replay mode
        if args.obs_path is None or args.action_path is None:
            parser.error("--replay requires --obs-path and --action-path")

        replayer = ReplayRollout(
            env_name=args.env_name,
            obs_path=args.obs_path,
            action_path=args.action_path,
            video_out=args.video_out,
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
        )
        replayer.run()
        return

    # Interactive mode
    runner = InteractiveRollout(
        env_name=args.env_name,
        host=args.host,
        port=args.port,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        save_dir=args.save_dir,
        video_dir=args.video_dir,
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
    if args.video_dir:
        print(f"Videos saved to: {args.video_dir}")


if __name__ == "__main__":
    main()
