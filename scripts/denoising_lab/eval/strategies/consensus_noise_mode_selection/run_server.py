"""Launch GR00T server with consensus noise mode selection.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head to sample K noise candidates, fully denoise all
K, score in EEF space, and select the best, then starts the ZMQ policy server.
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
from strategy import ConsensusConfig, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the consensus noise mode selection server."""

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

    # --- Consensus parameters ---

    K: int = 8
    """Number of noise candidates to evaluate."""

    num_steps: int = 4
    """Number of Euler denoising steps (all K candidates go through all D)."""

    lambda_pos: float = 1.0
    """Weight for EEF position closeness-to-mean score."""

    lambda_rot: float = 0.5
    """Weight for EEF rotation closeness-to-mean score."""

    lambda_jerk: float = 0.1
    """Weight for jerk minimization score."""

    action_horizon: int = 16
    """Meaningful timesteps for scoring (PandaOmron=16)."""


def main(config: ServerConfig):
    print("Starting GR00T server with CONSENSUS NOISE MODE SELECTION")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  K={config.K}  steps={config.num_steps}  "
          f"pos={config.lambda_pos}  rot={config.lambda_rot}  "
          f"jerk={config.lambda_jerk}  horizon={config.action_horizon}")
    print(f"  Total NFEs: {config.K * config.num_steps}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    cfg = ConsensusConfig(
        K=config.K,
        num_steps=config.num_steps,
        lambda_pos=config.lambda_pos,
        lambda_rot=config.lambda_rot,
        lambda_jerk=config.lambda_jerk,
        action_horizon=config.action_horizon,
    )
    reset_fn = patch_action_head(policy.model.action_head, cfg=cfg)
    print(f"  Strategy:   consensus_noise_mode_selection ({config.K}x{config.num_steps} NFEs)")

    _original_reset = policy.reset

    def _patched_reset(options=None):
        reset_fn()
        return _original_reset(options)

    policy.reset = _patched_reset

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
