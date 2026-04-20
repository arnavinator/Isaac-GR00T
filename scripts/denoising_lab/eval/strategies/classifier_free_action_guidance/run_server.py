"""Launch GR00T server with classifier-free action guidance.

Same CLI interface as ``gr00t/eval/run_gr00t_server.py``.  Loads the standard
model, patches the action head's denoising loop to apply CFG-style guidance
(amplifying the observation-conditioned velocity direction), then starts the
ZMQ policy server.

Note: Full CFG quality requires the model to be fine-tuned with observation
dropout.  Without that training, moderate guidance weights (w <= 2.0) are
recommended.
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
from strategy import CFGConfig, patch_action_head


@dataclass
class ServerConfig:
    """Configuration for the classifier-free guidance server."""

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

    # --- CFG parameters ---

    guidance_weight: float = 1.5
    """Fixed guidance weight w.  Set to 1.0 to disable guidance."""

    null_embed_path: str | None = None
    """Path to a saved null embedding from fine-tuning.  Uses zeros if not provided."""

    velocity_clamp_ratio: float = 2.0
    """Clamp guided velocity magnitude to this ratio of conditioned velocity."""


def main(config: ServerConfig):
    print("Starting GR00T server with CLASSIFIER-FREE ACTION GUIDANCE")
    print(f"  Model path: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Device:     {config.device}")
    print(f"  Host:       {config.host}:{config.port}")
    print(f"  Guidance:   w={config.guidance_weight}  "
          f"clamp_ratio={config.velocity_clamp_ratio}")
    if config.null_embed_path:
        print(f"  Null embed: {config.null_embed_path}")
    else:
        print("  Null embed: zeros (no fine-tuned null embedding provided)")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Load policy
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    # Build config and patch action head
    cfg = CFGConfig(
        guidance_weight=config.guidance_weight,
        use_sigmoid_schedule=False,  # fixed weight in server mode
        null_embed_path=config.null_embed_path,
        velocity_clamp_ratio=config.velocity_clamp_ratio,
    )
    patch_action_head(policy.model.action_head, cfg=cfg)
    print(f"  Strategy:   classifier_free_action_guidance "
          f"(~5 effective NFEs with batched cond+uncond)")

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
