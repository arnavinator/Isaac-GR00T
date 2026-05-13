"""Dense reward extraction from RoboCasa environments.

RoboCasa's default reward is sparse binary (0 on failure, 1 on success).
This module extracts CONTINUOUS PROGRESS metrics from the underlying robosuite
environments, enabling richer GRPO signal.

Why dense rewards matter for GRPO:
- GRPO needs variance within a group to produce non-zero advantages
- With pure binary rewards and a task at 50% success, ~50% of groups have mixed outcomes
- But with dense progress, EVEN all-fail groups have meaningful spread
  (e.g., drawer opened 60% vs 10% → different rewards → non-zero advantages)

Task-specific progress extraction:
- Door/drawer tasks: joint position from get_door_state() or get_drawer_state()
- PnP (pick-and-place) tasks: distance from object to target fixture
- Stove tasks: knob rotation state

These values are available in the underlying robosuite env but NOT exposed in
the gymnasium wrapper's info dict (which only has 'success' and 'grasp_distractor_obj').
We must unwrap to access them.

NOTE: This file runs in the ROBOCASA VENV (not main .venv), since it imports
robocasa/robosuite APIs.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Task type classification
# ---------------------------------------------------------------------------

# Map environment names to task types for reward extraction
TASK_TYPE_MAP = {
    # Door tasks
    "OpenDoor": "open_door",
    "CloseDoor": "close_door",
    "OpenDoubleDoor": "open_door",
    # Drawer tasks
    "OpenDrawer": "open_drawer",
    "CloseDrawer": "close_drawer",
    # Coffee tasks (PnP-style: pick mug, place on target fixture)
    "CoffeeServeMug": "pnp",
    "CoffeeSetupMug": "pnp",
    # Pick-and-place tasks (all PnP* environments)
    "PnP": "pnp",
    # Stove tasks
    "TurnOnStove": "turn_on_stove",
    "TurnOffStove": "turn_off_stove",
    # Microwave tasks
    "TurnOnMicrowave": "turn_on_microwave",
    "TurnOffMicrowave": "turn_off_microwave",
}


def classify_task_type(env_name: str) -> str:
    """Determine the task type from the environment name.

    Args:
        env_name: Full env name like "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
                  or just the task part like "OpenDrawer_PandaOmron_Env".

    Returns:
        Task type string (e.g., "open_drawer", "pnp", "turn_on_stove").
        Returns "unknown" if no match found.
    """
    # Extract task name from full path
    if "/" in env_name:
        env_name = env_name.split("/")[-1]

    # Strip the embodiment suffix (e.g., "_PandaOmron_Env")
    task_name = env_name.split("_")[0] if "_" in env_name else env_name

    # Try prefix matching (handles PnPCounterToSink, PnPMicrowaveToCounter, etc.)
    for prefix, task_type in TASK_TYPE_MAP.items():
        if task_name.startswith(prefix):
            return task_type

    return "unknown"


# ---------------------------------------------------------------------------
# Progress extraction (runs inside robocasa venv)
# ---------------------------------------------------------------------------


def compute_dense_progress(gym_env, task_type: str) -> float:
    """Extract continuous progress metric from RoboCasa environment.

    This unwraps the gymnasium environment to access robosuite internals.
    The returned value is in [0, 1] where 1.0 means task fully completed.

    Args:
        gym_env: The gymnasium environment (may be wrapped in VectorEnv).
            Must support .unwrapped to get the base robosuite env.
        task_type: One of the task types from classify_task_type().

    Returns:
        Progress value in [0, 1]. Higher = closer to task completion.
    """
    try:
        # Unwrap to get the base robosuite/robocasa environment
        # gym_env.unwrapped.env is the robosuite ManipulationEnv
        base_env = _get_base_env(gym_env)
        if base_env is None:
            return 0.0

        if task_type in ("open_door", "open_drawer"):
            return _get_door_drawer_progress(base_env, opening=True)

        elif task_type in ("close_door", "close_drawer"):
            return _get_door_drawer_progress(base_env, opening=False)

        elif task_type == "pnp":
            return _get_pnp_progress(base_env)

        elif task_type in ("turn_on_stove", "turn_off_stove"):
            return _get_stove_progress(base_env, turning_on=(task_type == "turn_on_stove"))

        elif task_type in ("turn_on_microwave", "turn_off_microwave"):
            return _get_microwave_progress(base_env, turning_on=(task_type == "turn_on_microwave"))

        else:
            # Unknown task type — fall back to binary success check
            return float(_check_success(base_env))

    except Exception:
        # If anything fails in progress extraction, return 0 (safe fallback)
        # Don't crash the episode collector over a reward shaping error
        return 0.0


def _get_base_env(gym_env):
    """Unwrap gymnasium env to get the underlying robosuite environment.

    Handles multiple levels of wrapping:
    - gymnasium VectorEnv wrappers
    - MultiStepWrapper
    - GrootRoboCasaEnv
    - robosuite ManipulationEnv (the actual env with fixtures)
    """
    try:
        # For single envs: gym_env.unwrapped.env
        env = gym_env
        while hasattr(env, "unwrapped") and env.unwrapped is not env:
            env = env.unwrapped
        # GrootRoboCasaEnv wraps robosuite env as .env
        if hasattr(env, "env"):
            return env.env
        return env
    except Exception:
        return None


def _get_door_drawer_progress(base_env, opening: bool) -> float:
    """Extract door/drawer joint position as progress metric.

    RoboCasa door fixtures expose get_door_state() which returns a dict of
    joint names → positions in [0, 1] (0 = fully closed, 1 = fully open).

    Success thresholds (from robocasa source):
    - Open door: joint_p >= 0.90
    - Close door: joint_p <= 0.10
    - Open drawer: joint_p >= 0.95
    - Close drawer: joint_p <= 0.05

    Args:
        base_env: Robosuite ManipulationEnv with door_fxtr/drawer attribute.
        opening: True if task is opening, False if closing.

    Returns:
        Progress in [0, 1]. For opening: progress = mean joint position.
        For closing: progress = 1 - mean joint position.
    """
    # Try door fixture first, then drawer
    fixture = getattr(base_env, "door_fxtr", None)
    if fixture is None:
        fixture = getattr(base_env, "drawer", None)
    if fixture is None:
        return 0.0

    try:
        door_state = fixture.get_door_state(env=base_env)
        if not door_state:
            return 0.0

        # Average across all joints (handles double doors)
        mean_position = np.mean(list(door_state.values()))

        if opening:
            return float(np.clip(mean_position, 0.0, 1.0))
        else:
            # For closing tasks, progress = how much it's been closed
            return float(np.clip(1.0 - mean_position, 0.0, 1.0))

    except Exception:
        return 0.0


def _get_pnp_progress(base_env) -> float:
    """Extract pick-and-place progress as normalized distance to target.

    Progress = 1 - clip(dist_to_target / max_dist, 0, 1)
    - 0.0 means object at starting position (or far from target)
    - 1.0 means object at target location

    Uses sim.data.body_xpos for object position and target fixture
    placement sites for target position.

    Args:
        base_env: Robosuite ManipulationEnv with object and target fixture.

    Returns:
        Progress in [0, 1].
    """
    try:
        # Get the manipulated object's position
        # Kitchen stores objects as a dict {name: model}, not a list
        if hasattr(base_env, "objects") and base_env.objects:
            if isinstance(base_env.objects, dict):
                obj = base_env.objects.get("obj") or list(base_env.objects.values())[0]
            elif hasattr(base_env.objects, '__getitem__'):
                obj = base_env.objects[0]
            else:
                return 0.0
            obj_pos = base_env.sim.data.body_xpos[
                base_env.sim.model.body_name2id(obj.root_body)
            ]
        elif hasattr(base_env, "obj"):
            obj_pos = base_env.sim.data.body_xpos[
                base_env.sim.model.body_name2id(base_env.obj.root_body)
            ]
        else:
            return 0.0

        # Get target position from the target fixture
        # Try multiple common attribute names (different tasks use different names)
        target_fixture = getattr(base_env, "target_fixture", None)
        if target_fixture is None:
            target_fixture = getattr(base_env, "placement_fixture", None)
        if target_fixture is None:
            target_fixture = getattr(base_env, "counter", None)

        if target_fixture is not None and hasattr(target_fixture, "get_int_sites"):
            # get_int_sites() returns placement site positions
            int_sites = target_fixture.get_int_sites()
            if int_sites:
                # Use center of interaction sites as target
                target_pos = np.mean(
                    [base_env.sim.data.site_xpos[
                        base_env.sim.model.site_name2id(site)
                    ] for site in int_sites],
                    axis=0,
                )
            else:
                return 0.0
        else:
            return 0.0

        # Compute normalized distance
        dist = np.linalg.norm(obj_pos - target_pos)
        max_dist = 1.5  # Maximum expected distance in workspace (meters)
        progress = 1.0 - np.clip(dist / max_dist, 0.0, 1.0)

        return float(progress)

    except Exception:
        return 0.0


def _get_stove_progress(base_env, turning_on: bool) -> float:
    """Extract stove knob state as progress metric.

    Args:
        base_env: Robosuite env with stove fixture.
        turning_on: True if turning on, False if turning off.

    Returns:
        Progress in [0, 1].
    """
    try:
        stove = getattr(base_env, "stove", None)
        if stove is None:
            return 0.0

        knob_state = stove.get_knob_state(env=base_env)
        if not knob_state:
            return 0.0

        # knob_state is typically {knob_name: value} where value is 0 (off) to 1 (on)
        mean_state = np.mean(list(knob_state.values()))

        if turning_on:
            return float(np.clip(mean_state, 0.0, 1.0))
        else:
            return float(np.clip(1.0 - mean_state, 0.0, 1.0))

    except Exception:
        return 0.0


def _get_microwave_progress(base_env, turning_on: bool) -> float:
    """Extract microwave state as progress metric.

    Args:
        base_env: Robosuite env with microwave fixture.
        turning_on: True if turning on, False if turning off.

    Returns:
        Progress in [0, 1].
    """
    try:
        microwave = getattr(base_env, "microwave", None)
        if microwave is None:
            return 0.0

        door_state = microwave.get_door_state(env=base_env)
        if not door_state:
            return 0.0

        mean_state = np.mean(list(door_state.values()))

        if turning_on:
            return float(np.clip(mean_state, 0.0, 1.0))
        else:
            return float(np.clip(1.0 - mean_state, 0.0, 1.0))

    except Exception:
        return 0.0


def _check_success(base_env) -> bool:
    """Check if the task is complete via the env's success check."""
    try:
        return bool(base_env._check_success())
    except Exception:
        return False


def compute_shaped_reward(
    success: bool,
    max_progress: float,
    success_weight: float = 1.0,
) -> float:
    """Compute shaped reward combining binary success and dense progress.

    Formula: reward = success_weight * success + (1 - success_weight) * max_progress

    Args:
        success: Whether the episode fully succeeded (binary).
        max_progress: Maximum progress achieved during the episode [0, 1].
        success_weight: Weight for binary success term (default 1.0 = pure binary).

    Returns:
        Shaped reward in [0, 1].

    Examples:
        >>> compute_shaped_reward(True, 1.0)              # Full success
        1.0
        >>> compute_shaped_reward(False, 0.6, 0.7)        # Failed but made progress
        0.18
        >>> compute_shaped_reward(False, 0.0)             # Complete failure
        0.0
    """
    return success_weight * float(success) + (1.0 - success_weight) * max_progress


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Dense Reward Self-Test ===\n")

    # Test task type classification
    test_cases = [
        ("robocasa_panda_omron/OpenDrawer_PandaOmron_Env", "open_drawer"),
        ("robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env", "pnp"),
        ("robocasa_panda_omron/TurnOnStove_PandaOmron_Env", "turn_on_stove"),
        ("robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env", "open_door"),
        ("robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env", "pnp"),
        ("robocasa_panda_omron/TurnOffStove_PandaOmron_Env", "turn_off_stove"),
    ]

    for env_name, expected_type in test_cases:
        result = classify_task_type(env_name)
        status = "PASS" if result == expected_type else "FAIL"
        print(f"  [{status}] {env_name} → {result} (expected {expected_type})")

    # Test shaped reward computation
    print("\nShaped reward examples (default success_weight=1.0):")
    print(f"  Success + full progress:           {compute_shaped_reward(True, 1.0):.3f}")
    print(f"  Fail + 60% progress:               {compute_shaped_reward(False, 0.6):.3f}")
    print(f"  Fail + no progress:                {compute_shaped_reward(False, 0.0):.3f}")
    print(f"  Success + 80% progress:            {compute_shaped_reward(True, 0.8):.3f}")
    print(f"  Fail + 60% progress (weight 0.7):  {compute_shaped_reward(False, 0.6, 0.7):.3f}")

    # With success_weight=1.0 the reward is purely binary
    assert compute_shaped_reward(True, 1.0) == 1.0
    assert compute_shaped_reward(False, 0.0) == 0.0
    assert compute_shaped_reward(False, 0.5) == 0.0
    # With success_weight<1.0, dense progress contributes
    assert 0 < compute_shaped_reward(False, 0.5, success_weight=0.7) < 0.5

    print("\nAll tests PASSED.")
