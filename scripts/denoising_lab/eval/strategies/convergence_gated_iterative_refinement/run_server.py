"""Launch GR00T server with convergence-gated iterative refinement.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head with phase-separated denoising + iterative
refinement at fixed tau, then starts the ZMQ policy server.
"""

from dataclasses import dataclass
import os
import sys
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer
from strategy import ConvergenceGatedConfig, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the convergence-gated iterative refinement server."""

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

    # --- Strategy parameters ---

    n_exec: int = 8
    """Standard execution horizon (n_action_steps executed per chunk)."""

    n_min: int = 2
    """Minimum execution horizon (safety floor for adaptive horizon)."""

    tau_refine: int = 750
    """Fixed timestep bucket for the Phase 2 refinement loop."""

    dt_refine: float = 0.25
    """Euler step size for Phase 2 refinement updates."""

    theta: float = 0.5
    """Per-position convergence threshold on velocity L2 norm."""

    K_max: int = 6
    """Maximum number of Phase 2 refinement iterations (budget cap)."""

    K_min: int = 2
    """Minimum iterations before early stopping is allowed."""

    clamp_uncertain: bool = True
    """Replace uncertain tail positions with last converged action."""


def main(config: ServerConfig):
    print("Starting GR00T server with CONVERGENCE-GATED ITERATIVE REFINEMENT")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  tau_refine={config.tau_refine}  theta={config.theta}  "
          f"K=[{config.K_min},{config.K_max}]  "
          f"n_exec={config.n_exec}  n_min={config.n_min}  "
          f"clamp={config.clamp_uncertain}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    cfg = ConvergenceGatedConfig(
        n_exec=config.n_exec,
        n_min=config.n_min,
        tau_refine=config.tau_refine,
        dt_refine=config.dt_refine,
        theta=config.theta,
        K_max=config.K_max,
        K_min=config.K_min,
        clamp_uncertain=config.clamp_uncertain,
    )
    patch_action_head(policy.model.action_head, cfg=cfg)
    nfe_range = f"{2 + config.K_min}-{2 + config.K_max}"
    print(f"  Strategy:   convergence_gated_iterative_refinement ({nfe_range} NFEs)")

    if config.use_sim_policy_wrapper:
        policy = Gr00tSimPolicyWrapper(policy)

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
