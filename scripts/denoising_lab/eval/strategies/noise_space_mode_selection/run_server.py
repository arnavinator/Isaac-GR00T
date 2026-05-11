"""Launch GR00T server with noise-space mode selection.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head to sample K noise candidates and select the best
via 1-step velocity preview, then starts the ZMQ policy server.
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
from strategy import NoiseSelectionConfig, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the noise-space mode selection server."""

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

    # --- Noise selection parameters ---

    K: int = 8
    """Number of noise candidates to evaluate."""

    lambda_smooth: float = 1.0
    """Weight for temporal smoothness score."""

    lambda_mag: float = 0.1
    """Weight for velocity magnitude score."""

    lambda_anchor: float = 0.5
    """Weight for anchor consistency with previous chunk."""

    anchor_decay: float = 0.5
    """Per-step decay for distance-weighted anchor scoring."""

    noise_type: str = "gaussian"
    """Noise distribution for candidates: "gaussian" or "uniform"."""

    score_dims: int | None = 12
    """Number of leading action dims to score (None = all). Default 12 for PandaOmron."""

    score_horizon: int | None = None
    """Number of leading timesteps to score (None = all). Set to 16 for PandaOmron
    to restrict to meaningful horizon."""

    noise_keyframes: int | None = None
    """Temporal keyframes for smooth noise (None = i.i.d.). Try 4-8."""


def main(config: ServerConfig):
    print("Starting GR00T server with NOISE-SPACE MODE SELECTION")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  K={config.K}  smooth={config.lambda_smooth}  "
          f"mag={config.lambda_mag}  anchor={config.lambda_anchor}  "
          f"decay={config.anchor_decay}  noise={config.noise_type}")
    print(f"  score_dims={config.score_dims}  score_horizon={config.score_horizon}  "
          f"noise_keyframes={config.noise_keyframes}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Build config and patch action head
    cfg = NoiseSelectionConfig(
        K=config.K,
        lambda_smooth=config.lambda_smooth,
        lambda_mag=config.lambda_mag,
        lambda_anchor=config.lambda_anchor,
        anchor_decay=config.anchor_decay,
        noise_type=config.noise_type,
        score_dims=config.score_dims,
        score_horizon=config.score_horizon,
        noise_keyframes=config.noise_keyframes,
    )
    reset_fn = patch_action_head(policy.model.action_head, cfg=cfg)
    print(f"  Strategy:   noise_space_mode_selection ({config.K}+3 NFEs)")

    # Hook reset_fn into policy.reset() so cached prev_actions is cleared
    # between episodes (prevents stale cross-episode anchor distortion)
    _original_reset = policy.reset

    def _patched_reset(options=None):
        reset_fn(options)
        return _original_reset(options)

    policy.reset = _patched_reset

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
