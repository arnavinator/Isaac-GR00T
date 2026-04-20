"""Launch GR00T server with evolutionary population denoising.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head with population-based denoising (K candidates
with per-step fitness selection, crossover, and mutation), then starts the
ZMQ policy server.
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
from strategy import EvolutionaryConfig, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the evolutionary population denoising server."""

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

    # --- Evolutionary parameters ---

    K: int = 8
    """Population size (must be even)."""

    lambda_smooth: float = 1.0
    """Fitness weight: temporal smoothness."""

    lambda_velocity: float = 0.1
    """Fitness weight: velocity magnitude."""

    lambda_consensus: float = 0.3
    """Fitness weight: inter-particle consensus."""

    sigma_0: float = 0.02
    """Initial mutation strength."""


def main(config: ServerConfig):
    print("Starting GR00T server with EVOLUTIONARY POPULATION DENOISING")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  K={config.K}  smooth={config.lambda_smooth}  "
          f"velocity={config.lambda_velocity}  consensus={config.lambda_consensus}  "
          f"sigma_0={config.sigma_0}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Build config and patch action head
    cfg = EvolutionaryConfig(
        K=config.K,
        lambda_smooth=config.lambda_smooth,
        lambda_velocity=config.lambda_velocity,
        lambda_consensus=config.lambda_consensus,
        sigma_0=config.sigma_0,
    )
    patch_action_head(policy.model.action_head, cfg=cfg)
    print(f"  Strategy:   evolutionary_population_denoising "
          f"({config.K}x4 NFEs, batched to 4 sequential passes)")

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
