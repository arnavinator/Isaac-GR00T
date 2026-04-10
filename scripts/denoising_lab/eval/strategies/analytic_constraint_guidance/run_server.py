"""Launch GR00T server with analytic constraint guidance.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head's denoising loop to apply physics-based
constraint gradients during each Euler step, then starts the ZMQ policy server.
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
from strategy import ConstraintConfig, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the constraint-guided denoising server."""

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

    # --- Constraint guidance parameters ---

    lambda_smooth: float = 0.005
    """Weight for temporal smoothness constraint (jerk minimisation)."""

    lambda_discrete: float = 0.01
    """Weight for discrete decisiveness constraint (gripper + control mode)."""

    lambda_mode: float = 0.003
    """Weight for control-mode temporal consistency constraint."""

    eta: float = 0.1
    """Overall guidance strength (annealed by tau)."""


def main(config: ServerConfig):
    print("Starting GR00T server with ANALYTIC CONSTRAINT GUIDANCE")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  Guidance:   eta={config.eta}  smooth={config.lambda_smooth}  "
          f"discrete={config.lambda_discrete}  mode={config.lambda_mode}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Build constraint config and patch action head
    cfg = ConstraintConfig(
        lambda_smooth=config.lambda_smooth,
        lambda_discrete=config.lambda_discrete,
        lambda_mode=config.lambda_mode,
        eta=config.eta,
    )
    patch_action_head(policy.model.action_head, cfg=cfg)
    print("  Strategy:   analytic_constraint_guidance (4 NFEs + analytic gradients)")

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
