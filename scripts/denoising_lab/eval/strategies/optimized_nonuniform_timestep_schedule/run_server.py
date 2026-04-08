"""Launch GR00T server with optimized non-uniform timestep schedule.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head to use a non-uniform tau schedule for the 4
Euler steps, then starts the ZMQ policy server.
"""

from dataclasses import dataclass
import os
import sys
from pathlib import Path

import tyro

# Make strategy.py importable from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer
from strategy import DEFAULT_SCHEDULE, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the non-uniform timestep schedule server."""

    model_path: str = "nvidia/GR00T-N1.6-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.ROBOCASA_PANDA_OMRON
    """Embodiment tag."""

    device: str = "cuda"
    """Device to run the model on."""

    host: str = "0.0.0.0"
    """Host address for the server."""

    port: int = 5555
    """Port number for the server."""

    use_sim_policy_wrapper: bool = True
    """Whether to use the sim policy wrapper."""

    verbose: bool = True
    """Enable verbose denoising step logging."""

    schedule: tuple[float, ...] = tuple(DEFAULT_SCHEDULE)
    """Tau schedule — sequence of ascending tau values starting at 0.0.
    Default: (0.0, 0.08, 0.35, 0.82).  Override with an empirically
    optimized schedule from ``strategy.find_optimal_schedule()``."""


def main(config: ServerConfig):
    schedule = list(config.schedule)
    print("Starting GR00T server with NON-UNIFORM TIMESTEP SCHEDULE")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  Schedule:   {schedule}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Patch action head with non-uniform schedule
    patch_action_head(policy.model.action_head, schedule=schedule)
    print(f"  Strategy:   optimized_nonuniform_timestep_schedule ({len(schedule)} NFEs)")

    # Wrap for sim if needed
    if config.use_sim_policy_wrapper:
        policy = Gr00tSimPolicyWrapper(policy)

    # Enable verbose logging
    if config.verbose:
        inner = policy.policy if config.use_sim_policy_wrapper else policy
        inner.model.action_head.verbose = True
        print("  Verbose:    enabled")

    server = PolicyServer(policy=policy, host=config.host, port=config.port)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
