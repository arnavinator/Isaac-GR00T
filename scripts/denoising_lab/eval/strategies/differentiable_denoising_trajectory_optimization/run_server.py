"""Launch GR00T server with Differentiable Denoising Trajectory Optimization (DDTO).

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head to optimise the initial noise via 1-step backprop
through the DiT, then starts the ZMQ policy server.
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
from strategy import DDTOConfig, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the DDTO server."""

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

    # --- DDTO parameters ---

    lambda_smooth: float = 1.0
    """Weight for temporal smoothness loss."""

    lambda_anchor: float = 0.5
    """Weight for anchor consistency with previous chunk."""

    lambda_mode: float = 0.1
    """Weight for on-mode regulariser (0 = Variant B, no HVP)."""

    anchor_decay: float = 0.5
    """Per-step decay for distance-weighted anchor scoring."""

    n_exec_steps: int = 8
    """Number of action steps executed per chunk."""

    eta: float = 0.1
    """Normalised gradient step size."""

    num_steps: int = 4
    """Number of denoising steps."""

    n_action_dims: int | None = 12
    """Active action dims for loss computation (PandaOmron: 12). None = all."""

    n_action_horizon: int | None = 16
    """Active timesteps for loss computation (PandaOmron: 16). None = all."""


def main(config: ServerConfig):
    variant = "A (mode reg)" if config.lambda_mode > 0 else "B (no mode reg)"
    print("Starting GR00T server with DIFFERENTIABLE DENOISING TRAJECTORY OPTIMIZATION (DDTO)")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  Variant:    {variant}")
    print(f"  smooth={config.lambda_smooth}  anchor={config.lambda_anchor}  "
          f"mode={config.lambda_mode}  decay={config.anchor_decay}  "
          f"eta={config.eta}  steps={config.num_steps}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Build config and patch action head
    cfg = DDTOConfig(
        lambda_smooth=config.lambda_smooth,
        lambda_anchor=config.lambda_anchor,
        lambda_mode=config.lambda_mode,
        anchor_decay=config.anchor_decay,
        n_exec_steps=config.n_exec_steps,
        eta=config.eta,
        num_steps=config.num_steps,
        n_action_dims=config.n_action_dims,
        n_action_horizon=config.n_action_horizon,
    )
    reset_fn = patch_action_head(policy.model.action_head, cfg=cfg)
    print(f"  Strategy:   ddto (1 grad + {config.num_steps} Euler = {config.num_steps + 1} NFEs)")

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
