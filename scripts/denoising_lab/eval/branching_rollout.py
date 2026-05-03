"""Branching rollout — restore sim state, inject custom action, continue autonomously.

Uses Approach 1 (Direct MuJoCo State Restoration): loads ``__sim_state__`` and
``__ep_meta__`` from a saved ``.npz``, restores the exact simulator state, optionally
executes a custom action chunk, then continues rolling out via VLA server until
success or ``max_episode_steps``.

Runs in the **sim venv** (robocasa_uv/.venv). Requires a running model server.

Usage::

    # Terminal 1 (model venv) — start the ZMQ server
    uv run python gr00t/eval/run_gr00t_server.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --embodiment-tag ROBOCASA_PANDA_OMRON \\
        --use-sim-policy-wrapper

    # Terminal 2 (sim venv) — branch from step 12, inject custom action, continue
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
        scripts/denoising_lab/eval/branching_rollout.py \\
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --obs-path /tmp/saved_observations/ep000_step012.npz \\
        --action-path /tmp/action_chunks/my_denoised.npz \\
        --output-dir /tmp/branching_results/exp_01 \\
        --save-observations

    # Without custom action (baseline — pure VLA continuation from saved state)
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
        scripts/denoising_lab/eval/branching_rollout.py \\
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --obs-path /tmp/saved_observations/ep000_step012.npz \\
        --output-dir /tmp/branching_results/baseline
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
)
from gr00t.policy.server_client import PolicyClient


# ---------------------------------------------------------------------------
# Utilities (copied from robocasa_eval_benchmark.py to keep sim-venv deps minimal)
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
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


def batch_obs(obs: dict[str, Any]) -> dict[str, Any]:
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
    unbatched: dict[str, Any] = {}
    for key, val in action.items():
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            unbatched[key] = val[0]
        else:
            unbatched[key] = val
    return unbatched


def extract_success(info: dict[str, Any]) -> bool:
    if "success" not in info:
        return False
    s = info["success"]
    if isinstance(s, (list, np.ndarray)):
        return bool(np.any(s))
    if isinstance(s, (bool, int)):
        return bool(s)
    return False


def _parse_step_from_filename(path: Path) -> int | None:
    m = re.search(r"step(\d+)", path.stem)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# BranchingRollout
# ---------------------------------------------------------------------------


class BranchingRollout:
    """Restore sim state from .npz, inject a custom action, continue autonomously."""

    def __init__(
        self,
        env_name: str,
        obs_path: str,
        action_path: str | None = None,
        host: str = "127.0.0.1",
        port: int = 5555,
        n_action_steps: int = 8,
        max_episode_steps: int = 720,
        custom_action_steps: int | None = None,
        output_dir: str = "/tmp/branching_results",
        save_observations: bool = False,
        video_dir: str | None = None,
        branch_step: int | None = None,
    ):
        self.env_name = env_name
        self.obs_path = Path(obs_path)
        self.action_path = Path(action_path) if action_path else None
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.custom_action_steps = custom_action_steps
        self.save_observations = save_observations

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if save_observations:
            self.obs_dir = self.output_dir / "observations"
            self.obs_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = Path(video_dir) if video_dir else None
        if self.video_dir is not None:
            self.video_dir.mkdir(parents=True, exist_ok=True)

        self.client = PolicyClient(host=host, port=port, strict=False)

        # Never use VideoRecordingWrapper — it can't see Phase 2 custom
        # action steps (which bypass the wrapper) and starts from the wrong
        # frame (randomized reset, before state restoration). We record
        # frames manually across all phases instead.
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

        self.branch_step = branch_step
        self._load_npz_metadata()

    # -- metadata ----------------------------------------------------------

    def _load_npz_metadata(self) -> None:
        data = dict(np.load(str(self.obs_path), allow_pickle=True))
        if "__sim_state__" not in data:
            raise ValueError(
                f"No __sim_state__ in {self.obs_path}. "
                "Re-save using the updated interactive_rollout.py."
            )
        self._sim_state = data["__sim_state__"]

        self._ep_meta = None
        if "__ep_meta__" in data:
            self._ep_meta = json.loads(str(data["__ep_meta__"]))

        self._model_xml = None
        if "__model_xml__" in data:
            self._model_xml = str(data["__model_xml__"])

        if self.branch_step is None and "__step_info__" in data:
            step_info = json.loads(str(data["__step_info__"]))
            self.branch_step = step_info.get("step")

        if self.branch_step is None:
            self.branch_step = _parse_step_from_filename(self.obs_path)

        if self.branch_step is None:
            raise ValueError(
                "Cannot determine branch step. Provide --branch-step or use a "
                ".npz with __step_info__ / filename ep*_step*.npz."
            )

    # -- env helpers -------------------------------------------------------

    def _get_robosuite_env(self):
        base_env = self.env.unwrapped
        return base_env.env

    def _restore_sim_state(self, sim_state: np.ndarray) -> None:
        robosuite_env = self._get_robosuite_env()
        robosuite_env.sim.set_state_from_flattened(sim_state)
        robosuite_env.sim.forward()
        if hasattr(robosuite_env, "update_state"):
            robosuite_env.update_state()
        elif hasattr(robosuite_env, "update_sites"):
            robosuite_env.update_sites()

    def _read_observation_after_restore(self) -> dict[str, Any]:
        robosuite_env = self._get_robosuite_env()
        raw_obs = robosuite_env._get_observations()
        base_env = self.env.unwrapped
        basic_obs = base_env.get_basic_observation(raw_obs)
        return base_env.get_groot_observation(basic_obs)

    def _get_sim_state(self) -> np.ndarray | None:
        try:
            base_env = self.env.unwrapped
            robosuite_env = base_env.env
            return np.array(robosuite_env.sim.get_state().flatten())
        except (AttributeError, TypeError):
            return None

    def _get_ep_meta(self) -> dict | None:
        try:
            base_env = self.env.unwrapped
            robosuite_env = base_env.env
            return robosuite_env.get_ep_meta()
        except (AttributeError, TypeError):
            return None

    def _get_model_xml(self) -> str | None:
        try:
            base_env = self.env.unwrapped
            robosuite_env = base_env.env
            return robosuite_env.sim.model.get_xml()
        except (AttributeError, TypeError):
            return None

    def _sync_wrapper_state(
        self, obs: dict[str, Any], consumed_substeps: int
    ) -> None:
        self.env.reward = [0.0] * consumed_substeps
        self.env.done = [False] * consumed_substeps
        self.env.obs = deque(
            [obs] * (self.env.max_steps_needed + 1),
            maxlen=self.env.max_steps_needed + 1,
        )

    # -- observation saving ------------------------------------------------

    def _save_branch_observation(
        self, obs: dict[str, Any], label: str, step: int,
    ) -> Path:
        path = self.obs_dir / f"{label}.npz"
        save_dict: dict[str, Any] = {}
        metadata: dict[str, str] = {}

        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                save_dict[key] = val[np.newaxis] if val.ndim < 3 else val
            elif isinstance(val, (tuple, list)):
                metadata[key] = (
                    "|".join(str(v) for v in val) if len(val) > 1 else str(val[0])
                )
            else:
                metadata[key] = str(val)

        if metadata:
            save_dict["__metadata__"] = np.array(json.dumps(metadata), dtype=object)

        sim_state = self._get_sim_state()
        if sim_state is not None:
            save_dict["__sim_state__"] = sim_state

        ep_meta = self._get_ep_meta()
        if ep_meta is not None:
            save_dict["__ep_meta__"] = np.array(
                json.dumps(ep_meta, cls=_NumpyEncoder), dtype=object
            )

        model_xml = self._get_model_xml()
        if model_xml is not None:
            save_dict["__model_xml__"] = np.array(model_xml, dtype=object)

        save_dict["__step_info__"] = np.array(
            json.dumps({
                "step": step,
                "n_action_steps": self.n_action_steps,
                "parent_obs_path": str(self.obs_path),
                "branch_step": self.branch_step,
            }),
            dtype=object,
        )

        np.savez_compressed(str(path), **save_dict)

        self._save_camera_snapshot(obs, path.with_suffix(".png"))
        return path

    def _save_camera_snapshot(self, obs: dict[str, Any], path: Path) -> None:
        frames = self._collect_camera_frames(obs)
        if not frames:
            return
        montage = self._montage(frames)
        cv2.imwrite(str(path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    def _collect_camera_frames(self, obs: dict[str, Any]) -> list[np.ndarray]:
        frames = []
        for key in sorted(obs.keys()):
            if not key.startswith("video.") or "res512" in key:
                continue
            val = obs[key]
            if not isinstance(val, np.ndarray):
                continue
            img = val
            while img.ndim > 3:
                img = img[0]
            frames.append(img)
        return frames

    def _montage(self, frames: list[np.ndarray]) -> np.ndarray:
        target_h = min(f.shape[0] for f in frames)
        resized = []
        for f in frames:
            if f.shape[0] != target_h:
                scale = target_h / f.shape[0]
                new_w = int(f.shape[1] * scale)
                f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized.append(f)
        return np.concatenate(resized, axis=1)

    def _write_video(
        self, frames: list[tuple[str, np.ndarray]], path: Path
    ) -> None:
        import av

        if not frames:
            return
        h, w = frames[0][1].shape[:2]
        h = h - (h % 2)
        w = w - (w % 2)
        container = av.open(str(path), mode="w")
        stream = container.add_stream("h264", rate=10)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"crf": "18"}
        for label, frame_arr in frames:
            if frame_arr.shape[:2] != (h, w):
                frame_arr = cv2.resize(frame_arr, (w, h))
            frame_arr = frame_arr.copy()
            cv2.putText(
                frame_arr, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    # -- main run ----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        t0 = time.monotonic()

        # ---- Phase 1: State Restoration ----------------------------------
        print(f"Branch point: step {self.branch_step} from {self.obs_path.name}")

        robosuite_env = self._get_robosuite_env()
        if self._ep_meta is not None:
            robosuite_env.set_ep_meta(self._ep_meta)
            print(
                f"Restored ep_meta: layout_id={self._ep_meta.get('layout_id')}, "
                f"style_id={self._ep_meta.get('style_id')}"
            )

        print("Resetting environment...")
        self.env.reset()

        if self._model_xml is not None:
            # Rebuild the scene from the saved model XML. This ensures
            # robosuite's internal task state (object tracking, success
            # conditions) matches the original episode — not the randomized
            # scene from reset(). Follows playback_dataset.py:reset_to().
            xml = robosuite_env.edit_model_xml(self._model_xml)
            robosuite_env.reset_from_xml_string(xml)
            robosuite_env.sim.reset()
            print("Restored model XML (exact scene reconstruction).")
        else:
            print(
                "WARNING: No __model_xml__ in saved observation. "
                "Task state may not match the original episode. "
                "Re-save observations with the updated interactive_rollout.py."
            )

        print("Restoring sim state...")
        self._restore_sim_state(self._sim_state)
        obs = self._read_observation_after_restore()
        print("State restored successfully.")

        video_frames: list[np.ndarray] = []

        success = False
        total_reward = 0.0
        custom_substeps_executed = 0

        # ---- Phase 2: Custom Action Injection ----------------------------
        if self.action_path is not None:
            action_data = dict(np.load(str(self.action_path), allow_pickle=True))
            print(f"Custom action keys: {list(action_data.keys())}")

            action_horizon = list(action_data.values())[0].shape[1]
            n_custom = min(
                self.custom_action_steps if self.custom_action_steps is not None else action_horizon,
                action_horizon,
            )
            print(f"Executing {n_custom} custom action sub-steps (of {action_horizon} available)...")

            base_env = self.env.unwrapped
            for t in range(n_custom):
                single_step_action = {}
                for key, arr in action_data.items():
                    prefixed = f"action.{key}" if not key.startswith("action.") else key
                    single_step_action[prefixed] = arr[0, t]

                try:
                    base_obs, reward, terminated, truncated, step_info = base_env.step(
                        single_step_action
                    )
                except Exception as e:
                    print(f"  Custom step {t}: env.step() failed: {e}")
                    break

                step_success = step_info.get("success", False)
                total_reward += float(reward)
                custom_substeps_executed += 1
                print(f"  Custom step {t}: reward={reward:.3f} success={step_success}")

                if self.video_dir is not None:
                    cam = self._collect_camera_frames(base_obs)
                    if cam:
                        video_frames.append((f"custom {t}/{n_custom}", self._montage(cam)))

                if self.save_observations:
                    self._save_branch_observation(
                        base_obs,
                        f"branch_step{self.branch_step:03d}_custom_{t:02d}",
                        step=self.branch_step,
                    )

                if step_success:
                    success = True
                    break

            obs = self._read_observation_after_restore()
        else:
            print("No custom action — continuing autonomously from restored state.")

        # ---- Phase 3: Wrapper Synchronization ----------------------------
        consumed_substeps = self.branch_step * self.n_action_steps + custom_substeps_executed
        remaining = self.max_episode_steps - consumed_substeps
        print(
            f"Consumed {consumed_substeps} sub-steps "
            f"({self.branch_step} outer steps × {self.n_action_steps} + "
            f"{custom_substeps_executed} custom). "
            f"Remaining budget: {remaining} sub-steps."
        )

        auto_step = 0

        if success:
            print("Task succeeded during custom action — skipping autonomous phase.")
        elif remaining <= 0:
            print("No step budget remaining — skipping autonomous phase.")
        else:
            self._sync_wrapper_state(obs, consumed_substeps)
            wrapper_obs = self.env._get_obs(
                self.env.video_delta_indices, self.env.state_delta_indices
            )

            # ---- Phase 4: Autonomous Continuation ------------------------
            print("Starting autonomous rollout...")
            print(f"Connecting to server...")
            if not self.client.ping():
                print("WARNING: Server not responding.")
            else:
                print("Server connected.")

            batched = batch_obs(wrapper_obs)
            done = False
            auto_step = 0

            while not done:
                action, _ = self.client.get_action(batched)
                obs, reward, terminated, truncated, info = self.env.step(
                    unbatch_action(action)
                )
                total_reward += float(reward)
                auto_step += 1
                done = terminated or truncated
                success = success or extract_success(info)

                if auto_step % 10 == 0 or done:
                    status = "SUCCESS" if success else "running"
                    print(
                        f"  Auto step {auto_step}: "
                        f"reward={total_reward:.3f} status={status}"
                    )

                raw_obs = self.env.obs[-1]

                if self.video_dir is not None:
                    cam = self._collect_camera_frames(raw_obs)
                    if cam:
                        label = f"auto {auto_step} (step {self.branch_step + auto_step})"
                        video_frames.append((label, self._montage(cam)))

                if self.save_observations and not done:
                    outer_step = self.branch_step + auto_step
                    self._save_branch_observation(
                        raw_obs,
                        f"branch_step{outer_step:03d}",
                        step=outer_step,
                    )

                if not done:
                    batched = batch_obs(obs)

        # ---- Phase 5: Save Results ---------------------------------------
        duration = time.monotonic() - t0

        if self.video_dir is not None and video_frames:
            video_path = self.video_dir / f"branch_from_step{self.branch_step:03d}.mp4"
            self._write_video(video_frames, video_path)
            print(f"Video saved: {video_path} ({len(video_frames)} frames)")

        termination_reason = "success" if success else "truncated"
        if success and custom_substeps_executed > 0 and auto_step == 0:
            termination_reason = "success_during_custom_action"

        lineage = {
            "parent_obs_path": str(self.obs_path.resolve()),
            "branch_step": self.branch_step,
            "consumed_substeps_before_autonomous": consumed_substeps,
            "custom_action_path": str(self.action_path.resolve()) if self.action_path else None,
            "custom_action_substeps_executed": custom_substeps_executed,
            "env_name": self.env_name,
            "max_episode_steps": self.max_episode_steps,
            "n_action_steps": self.n_action_steps,
            "result": {
                "success": success,
                "total_reward": round(total_reward, 4),
                "autonomous_steps": auto_step,
                "termination_reason": termination_reason,
            },
            "duration_s": round(duration, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        lineage_path = self.output_dir / "lineage.json"
        with open(lineage_path, "w") as f:
            json.dump(lineage, f, indent=2)

        status = "SUCCESS" if success else "FAIL"
        print(f"\n{'='*60}")
        print(f"Result: {status}")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Custom sub-steps: {custom_substeps_executed}")
        print(f"  Autonomous steps: {auto_step}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Lineage: {lineage_path}")
        print(f"{'='*60}")

        return lineage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Branch from a saved sim state, inject custom action, continue autonomously",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-name",
        type=str,
        required=True,
        help="Gymnasium env ID (e.g. robocasa_panda_omron/OpenDrawer_PandaOmron_Env)",
    )
    parser.add_argument(
        "--obs-path",
        type=str,
        required=True,
        help="Path to observation .npz with saved sim state (from interactive_rollout.py)",
    )
    parser.add_argument(
        "--action-path",
        type=str,
        default=None,
        help="Path to custom action chunk .npz (from notebook). "
        "Omit for pure VLA continuation from saved state.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument(
        "--custom-action-steps",
        type=int,
        default=None,
        help="How many sub-steps of the custom action to execute (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/branching_results",
    )
    parser.add_argument(
        "--save-observations",
        action="store_true",
        help="Save per-step .npz files in output_dir/observations/",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory for video recording (None = no video)",
    )
    parser.add_argument(
        "--branch-step",
        type=int,
        default=None,
        help="Override branch step (default: read from .npz metadata or filename)",
    )

    args = parser.parse_args()

    runner = BranchingRollout(
        env_name=args.env_name,
        obs_path=args.obs_path,
        action_path=args.action_path,
        host=args.host,
        port=args.port,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        custom_action_steps=args.custom_action_steps,
        output_dir=args.output_dir,
        save_observations=args.save_observations,
        video_dir=args.video_dir,
        branch_step=args.branch_step,
    )
    runner.run()


if __name__ == "__main__":
    main()
