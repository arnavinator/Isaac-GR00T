"""Test that observation keys round-trip correctly through the collection pipeline.

Verifies: extract → save npz → load → ActionChunk has correct keys for VLAStepData.
The processor expects modality keys WITHOUT the flat namespace prefix:
  - Video: "res256_image_side_0" (not "video.res256_image_side_0")
  - State: "gripper_qpos" (not "state.gripper_qpos")

NOTE: collect_episodes.py requires the robocasa venv (gymnasium). This test
reimplements the key extraction logic to verify correctness without that dependency.
"""

import tempfile
from pathlib import Path

import numpy as np

from episode_buffer import EpisodeBuffer


def _extract_video_stripped(observations, env_idx: int) -> dict:
    """Reimplements collect_episodes._extract_video with prefix stripping."""
    frames = {}
    if isinstance(observations, dict):
        for key, value in observations.items():
            if "image" in key or "video" in key:
                if hasattr(value, '__getitem__') and len(value) > env_idx:
                    clean_key = key.removeprefix("video.")
                    frames[clean_key] = np.array(value[env_idx])
    return frames


def _extract_state_stripped(observations, env_idx: int) -> dict:
    """Reimplements collect_episodes._extract_state with prefix stripping."""
    state = {}
    if isinstance(observations, dict):
        for key, value in observations.items():
            if "image" not in key and "video" not in key and "language" not in key:
                if "annotation" in key:
                    continue
                if hasattr(value, '__getitem__') and len(value) > env_idx:
                    clean_key = key.removeprefix("state.")
                    state[clean_key] = np.array(value[env_idx])
    return state


def _save_episodes_minimal(episodes: list, output_dir: str) -> None:
    """Reimplements the save logic from collect_episodes.save_episodes."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, ep in enumerate(episodes):
        save_dict = {
            "language": ep.get("language") or "test",
            "env_name": ep["env_name"],
            "success": ep["success"],
            "max_progress": ep["max_progress"],
            "num_steps": ep["num_steps"],
            "num_chunks": len(ep["actions"]),
            "group_id": ep.get("group_id", 0),
            "env_seed": ep.get("env_seed", 0),
        }

        for chunk_idx in range(len(ep["actions"])):
            if chunk_idx < len(ep["video_frames"]):
                for cam_name, frame in ep["video_frames"][chunk_idx].items():
                    save_dict[f"video_{cam_name}_{chunk_idx}"] = frame
            if chunk_idx < len(ep["states"]):
                for state_key, state_val in ep["states"][chunk_idx].items():
                    save_dict[f"state_{state_key}_{chunk_idx}"] = state_val
            save_dict[f"action_{chunk_idx}"] = ep["actions"][chunk_idx]
            if chunk_idx < len(ep.get("raw_actions", [])) and ep["raw_actions"][chunk_idx] is not None:
                save_dict[f"raw_action_{chunk_idx}"] = ep["raw_actions"][chunk_idx]
            if chunk_idx < len(ep.get("action_masks", [])) and ep["action_masks"][chunk_idx] is not None:
                save_dict[f"action_mask_{chunk_idx}"] = ep["action_masks"][chunk_idx]
            else:
                save_dict[f"action_mask_{chunk_idx}"] = np.ones((50, 128), dtype=np.float32)
            if chunk_idx < len(ep.get("initial_noises", [])) and ep["initial_noises"][chunk_idx] is not None:
                save_dict[f"initial_noise_{chunk_idx}"] = ep["initial_noises"][chunk_idx]

        np.savez_compressed(output_path / f"episode_{idx:04d}.npz", **save_dict)


def test_key_roundtrip():
    """Verify observation keys are stored without modality prefix."""

    # Mock observation as produced by GrootRoboCasaEnv (flat keys with modality prefix)
    mock_obs = {
        "video.res256_image_side_0": np.random.randint(0, 255, (2, 256, 256, 3), dtype=np.uint8),
        "video.res256_image_side_1": np.random.randint(0, 255, (2, 256, 256, 3), dtype=np.uint8),
        "video.res256_image_wrist_0": np.random.randint(0, 255, (2, 256, 256, 3), dtype=np.uint8),
        "state.gripper_qpos": np.random.randn(2, 1).astype(np.float32),
        "state.base_position": np.random.randn(2, 3).astype(np.float32),
        "state.end_effector_position_relative": np.random.randn(2, 3).astype(np.float32),
        "annotation.human.action.task_description": ("pick up the mug", "pick up the mug"),
    }

    # Test _extract_video strips prefix
    frames = _extract_video_stripped(mock_obs, env_idx=0)

    assert "res256_image_side_0" in frames, f"Expected stripped key, got: {list(frames.keys())}"
    assert "res256_image_side_1" in frames
    assert "res256_image_wrist_0" in frames
    assert not any(k.startswith("video.") for k in frames), "Keys should not have video. prefix"
    print("  [PASS] _extract_video strips 'video.' prefix correctly")

    # Test _extract_state strips prefix and filters annotations
    state = _extract_state_stripped(mock_obs, env_idx=0)

    assert "gripper_qpos" in state, f"Expected stripped key, got: {list(state.keys())}"
    assert "base_position" in state
    assert "end_effector_position_relative" in state
    assert not any(k.startswith("state.") for k in state), "Keys should not have state. prefix"
    assert not any("annotation" in k for k in state), "Annotation keys should be filtered out"
    print("  [PASS] _extract_state strips 'state.' prefix and filters annotations")

    # Test full round-trip: save → load → verify keys
    with tempfile.TemporaryDirectory() as tmpdir:
        episode = {
            "video_frames": [frames],
            "states": [state],
            "actions": [np.random.randn(16, 12).astype(np.float32)],
            "raw_actions": [np.random.randn(50, 128).astype(np.float32)],
            "action_masks": [np.ones((50, 128), dtype=np.float32)],
            "initial_noises": [np.random.randn(50, 128).astype(np.float32)],
            "language": "pick up the mug",
            "success": True,
            "max_progress": 0.8,
            "shaped_reward": 0.9,
            "env_name": "robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env",
            "num_steps": 100,
            "group_id": 0,
            "env_seed": 42,
        }

        _save_episodes_minimal([episode], tmpdir)

        # Load back via EpisodeBuffer
        buffer = EpisodeBuffer()
        n_loaded = buffer.load_episodes(tmpdir)
        assert n_loaded == 1, f"Expected 1 episode loaded, got {n_loaded}"

        ep = buffer.episodes[0]

        # Verify video keys don't have prefix
        for cam_name in ep.video_frames[0].keys():
            assert not cam_name.startswith("video."), f"Loaded video key has prefix: {cam_name}"
        assert "res256_image_side_0" in ep.video_frames[0]
        print("  [PASS] Video keys round-trip without prefix")

        # Verify state keys don't have prefix
        for state_key in ep.states[0].keys():
            assert not state_key.startswith("state."), f"Loaded state key has prefix: {state_key}"
        assert "gripper_qpos" in ep.states[0]
        assert "base_position" in ep.states[0]
        print("  [PASS] State keys round-trip without prefix")

    print("\nAll key round-trip tests PASSED.")


if __name__ == "__main__":
    print("=== Key Round-Trip Test ===\n")
    test_key_roundtrip()
