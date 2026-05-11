"""Launch GR00T server with density-aware denoising.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head's denoising loop to use density-aware
divergence estimation, then starts the ZMQ policy server.
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
from strategy import DensityAwareConfig, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the density-aware denoising server."""

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

    # --- Density-aware parameters ---

    mode: str = "guided"
    """Operating mode: "monitor", "guided", or "rank"."""

    h: float = 1e-3
    """Finite-difference perturbation scale."""

    alpha: float = 0.15
    """Guidance strength (guided mode). 0 = pure monitoring."""

    D0: float | None = None
    """Divergence normalization scale. None = auto per step."""

    N: int = 4
    """Number of candidates (rank mode)."""

    score_dims: int | None = 12
    """Leading action dims to probe (None = all). 12 = PandaOmron meaningful dims."""

    score_horizon: int | None = None
    """Leading timesteps to probe (None = all). 16 = PandaOmron action horizon."""

    lambda_anchor: float = 0.0
    """Anchor consistency weight for rank mode (0 = pure log-likelihood)."""

    anchor_decay: float = 0.5
    """Geometric decay for anchor distance weighting."""

    n_exec_steps: int = 8
    """Overlap region for anchor consistency."""


def main(config: ServerConfig):
    print("Starting GR00T server with DENSITY-AWARE DENOISING")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  Mode:       {config.mode}")
    print(f"  Params:     h={config.h}  alpha={config.alpha}  D0={config.D0}  "
          f"N={config.N}  score_dims={config.score_dims}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    cfg = DensityAwareConfig(
        mode=config.mode,
        h=config.h,
        alpha=config.alpha,
        D0=config.D0,
        N=config.N,
        score_dims=config.score_dims,
        score_horizon=config.score_horizon,
        lambda_anchor=config.lambda_anchor,
        anchor_decay=config.anchor_decay,
        n_exec_steps=config.n_exec_steps,
    )

    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    reset_fn = patch_action_head(policy.model.action_head, cfg=cfg)
    print(f"  Strategy:   density_aware_denoising ({config.mode} mode, "
          f"{'batched divergence + guided scaling' if config.mode == 'guided' else config.mode})")

    if config.use_sim_policy_wrapper:
        policy = Gr00tSimPolicyWrapper(policy)

    original_reset = policy.reset
    def augmented_reset(options=None):
        reset_fn(options)
        return original_reset(options)
    policy.reset = augmented_reset

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
