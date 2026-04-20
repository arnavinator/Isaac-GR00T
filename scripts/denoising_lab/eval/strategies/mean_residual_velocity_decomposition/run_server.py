"""Launch GR00T server with mean-residual velocity decomposition.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head's denoising loop to apply mean-residual
velocity scaling, then starts the ZMQ policy server.
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
from strategy import patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the mean-residual velocity decomposition server."""

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

    verbose: bool = False
    """Enable verbose denoising step logging."""

    # --- Mean-residual parameters ---

    rho: float = 1.15
    """Residual scaling factor. 1.0 = baseline. >1.0 boosts structural
    detail. <1.0 boosts uniformity."""

    onset: int = 2
    """First denoising step to apply decomposition (0-3)."""

    energy_preserve: bool = True
    """Normalise modified velocity to preserve total magnitude."""


def main(config: ServerConfig):
    print("Starting GR00T server with MEAN-RESIDUAL VELOCITY DECOMPOSITION")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  Params:     rho={config.rho}  onset={config.onset}  "
          f"energy_preserve={config.energy_preserve}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Patch action head with mean-residual denoising
    patch_action_head(
        policy.model.action_head,
        rho=config.rho,
        onset=config.onset,
        energy_preserve=config.energy_preserve,
    )
    print("  Strategy:   mean_residual_velocity_decomposition (4 NFEs, DC-AC split)")

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
