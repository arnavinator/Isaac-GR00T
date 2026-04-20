"""Launch GR00T server with curvature-adaptive step-size control.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head with an adaptive Euler-Heun integrator that
automatically adjusts step sizes based on velocity field curvature, then
starts the ZMQ policy server.
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
from strategy import AdaptiveConfig, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the curvature-adaptive step-size server."""

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

    # --- Adaptive parameters ---

    atol: float = 0.05
    """Absolute error tolerance (normalised action space)."""

    max_nfe: int = 8
    """Hard NFE budget for bounded worst-case latency."""

    dt_init: float = 0.25
    """Initial step size."""

    dt_min: float = 0.125
    """Minimum step size."""


def main(config: ServerConfig):
    print("Starting GR00T server with CURVATURE-ADAPTIVE STEP-SIZE CONTROL")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  Adaptive:   atol={config.atol}  max_nfe={config.max_nfe}  "
          f"dt_init={config.dt_init}  dt_min={config.dt_min}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Build config and patch action head
    cfg = AdaptiveConfig(
        atol=config.atol,
        max_nfe=config.max_nfe,
        dt_init=config.dt_init,
        dt_min=config.dt_min,
    )
    patch_action_head(policy.model.action_head, cfg=cfg)
    print(f"  Strategy:   curvature_adaptive_step_size (6-{config.max_nfe} NFEs adaptive)")

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
