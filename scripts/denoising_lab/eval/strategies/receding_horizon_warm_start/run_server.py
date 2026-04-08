"""Launch GR00T server with receding-horizon warm-start denoising.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head to warm-start each chunk from the previous
chunk's un-executed actions (3 NFEs after the first cold-start chunk), and
starts the ZMQ policy server.

The warm-start cache is automatically cleared on ``policy.reset()`` (called
between episodes by the rollout client).
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
    """Configuration for the warm-start denoising server."""

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

    tau_start: float = 0.25
    """Noise level to re-noise the warm-start to (0.25 = skip 1 of 4 steps)."""

    n_executed: int = 8
    """Number of action steps executed from each chunk before re-planning."""


def main(config: ServerConfig):
    print("Starting GR00T server with RECEDING-HORIZON WARM-START")
    print(f"  Model path:  {config.model_path}")
    print(f"  Embodiment:  {config.embodiment_tag}")
    print(f"  Device:      {config.device}")
    print(f"  Host:        {config.host}:{config.port}")
    print(f"  tau_start:   {config.tau_start}")
    print(f"  n_executed:  {config.n_executed}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Patch action head with warm-start denoising
    reset_fn = patch_action_head(
        policy.model.action_head,
        tau_start=config.tau_start,
        n_executed=config.n_executed,
    )
    nfes_warm = round((1.0 - config.tau_start) * 4)
    print(f"  Strategy:    receding_horizon_warm_start ({nfes_warm} NFEs after cold start)")

    # Hook reset_fn into policy.reset() so warm-start cache is cleared
    # between episodes
    _original_reset = policy.reset

    def _patched_reset(options=None):
        reset_fn()
        return _original_reset(options)

    policy.reset = _patched_reset

    # Wrap for sim if needed
    if config.use_sim_policy_wrapper:
        policy = Gr00tSimPolicyWrapper(policy)

    # Enable verbose logging
    if config.verbose:
        inner = policy.policy if config.use_sim_policy_wrapper else policy
        inner.model.action_head.verbose = True
        print("  Verbose:     enabled")

    server = PolicyServer(policy=policy, host=config.host, port=config.port)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
