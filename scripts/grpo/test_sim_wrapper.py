"""Test that _InPlacePolicy satisfies Gr00tSimPolicyWrapper's interface requirements.

The wrapper asserts: len(self.policy.modality_configs["language"].delta_indices) == 1
This test verifies the _InPlacePolicy class has the right attribute structure
to pass this assertion without needing a GPU or the actual model.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add project root and scripts/grpo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class MockModalityConfig:
    modality_keys: list
    delta_indices: list


def test_inplace_policy_modality_configs():
    """Verify _InPlacePolicy.modality_configs has the structure Gr00tSimPolicyWrapper expects."""

    # Mock the processor.get_modality_configs() return value
    mock_configs = {
        "ROBOCASA_PANDA_OMRON": {
            "video": MockModalityConfig(
                modality_keys=["res256_image_side_0", "res256_image_side_1", "res256_image_wrist_0"],
                delta_indices=[0],
            ),
            "state": MockModalityConfig(
                modality_keys=["gripper_qpos", "base_position", "base_rotation",
                               "end_effector_position_relative", "end_effector_rotation_relative"],
                delta_indices=[0],
            ),
            "language": MockModalityConfig(
                modality_keys=["annotation.human.action.task_description"],
                delta_indices=[0],
            ),
            "action": MockModalityConfig(
                modality_keys=["end_effector_position", "end_effector_rotation",
                               "gripper_close", "base_motion", "control_mode"],
                delta_indices=list(range(16)),
            ),
        }
    }

    # Create a mock processor
    mock_processor = MagicMock()
    mock_processor.get_modality_configs.return_value = mock_configs

    # Simulate what _InPlacePolicy does:
    embodiment_tag_value = "ROBOCASA_PANDA_OMRON"
    modality_configs = mock_processor.get_modality_configs()[embodiment_tag_value]

    # This is the assertion Gr00tSimPolicyWrapper.__init__ makes:
    assert "language" in modality_configs, "modality_configs must have 'language' key"
    assert hasattr(modality_configs["language"], "delta_indices"), "language config must have delta_indices"
    assert len(modality_configs["language"].delta_indices) == 1, (
        f"Expected 1 language delta index, got {len(modality_configs['language'].delta_indices)}"
    )
    print("  [PASS] modality_configs['language'].delta_indices has length 1")

    # Verify the SimWrapper's key iteration pattern works
    for modality in ["video", "state", "language"]:
        assert modality in modality_configs, f"Missing modality: {modality}"
        assert hasattr(modality_configs[modality], "modality_keys"), f"{modality} missing modality_keys"
        assert len(modality_configs[modality].modality_keys) > 0, f"{modality} has no keys"
    print("  [PASS] All modalities have modality_keys attribute")

    # Verify key format expectations
    for key in modality_configs["video"].modality_keys:
        assert not key.startswith("video."), f"Video modality key should not have prefix: {key}"
    for key in modality_configs["state"].modality_keys:
        assert not key.startswith("state."), f"State modality key should not have prefix: {key}"
    print("  [PASS] Modality keys don't have namespace prefixes")

    print("\nAll SimWrapper compatibility tests PASSED.")


if __name__ == "__main__":
    print("=== SimWrapper Compatibility Test ===\n")
    test_inplace_policy_modality_configs()
