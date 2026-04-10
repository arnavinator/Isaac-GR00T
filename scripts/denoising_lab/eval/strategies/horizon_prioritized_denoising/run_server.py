"""Launch GR00T server with horizon-prioritized denoising.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head's denoising loop to apply position-dependent
velocity gating, then starts the ZMQ policy server.
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
    """Configuration for the horizon-prioritized denoising server."""

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

    # --- Horizon-prioritized parameters ---

    gamma: float = 0.5
    """Boost amplitude for the Gaussian attention window."""

    sigma_w: float = 3.0
    """Width of the Gaussian attention window."""

    effective_horizon: int = 16
    """Actual action steps for the embodiment (16 for PandaOmron)."""


def main(config: ServerConfig):
    print("Starting GR00T server with HORIZON-PRIORITIZED DENOISING")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  Params:     gamma={config.gamma}  sigma_w={config.sigma_w}  "
          f"effective_horizon={config.effective_horizon}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Patch action head with horizon-prioritized denoising
    patch_action_head(
        policy.model.action_head,
        gamma=config.gamma,
        sigma_w=config.sigma_w,
        effective_horizon=config.effective_horizon,
    )
    print("  Strategy:   horizon_prioritized_denoising (4 NFEs, position-dependent gating)")

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
