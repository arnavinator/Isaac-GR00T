"""Extended GR00T policy server for GRPO training.

This server extends the standard PolicyServer (gr00t/policy/server_client.py) to:
1. Capture the initial noise tensor used during denoising (via torch.randn hook)
2. Capture the raw normalized action tensor (50×128) from the DiT output

Why initial_noise matters for GRPO:
- The FM log-prob surrogate evaluates the velocity field along an interpolation path
  x_τ = (1-τ)ε + τ*action. Using the SAME ε for both current and ref models ensures
  the importance ratio reflects only the model difference, not estimation noise.
- We capture ε₀ (the actual noise that was denoised into the action) so training
  can evaluate along the true path the model took, rather than random paths.

Implementation approach:
- Monkey-patch torch.randn for the duration of the policy call
- Capture the first 3D tensor (the denoising noise in get_action_with_features)
- Also hook get_action_with_features to capture the raw action before decoding
- Restore both patches in a try/finally block

This file runs in the MAIN VENV (GPU) alongside the model.

Usage:
    uv run python scripts/grpo/grpo_server.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --embodiment-tag ROBOCASA_PANDA_OMRON \\
        --port 5555
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gr00t.policy.server_client import PolicyServer
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class GRPOServerConfig:
    """Configuration for the GRPO-extended model server.

    Extends the standard ServerConfig with GRPO-specific options.
    """
    # Standard server config
    model_path: str = "nvidia/GR00T-N1.6-3B"
    embodiment_tag: str = "ROBOCASA_PANDA_OMRON"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 5555

    # GRPO extensions
    # Whether to use density-aware diagnostics during inference
    use_density_diagnostics: bool = False

    # LoRA checkpoint to load (None = use base model)
    lora_checkpoint: Optional[str] = None

    # Whether to use the sim policy wrapper (flat keys ↔ nested format)
    use_sim_policy_wrapper: bool = True

    # Verbose logging of denoising steps
    verbose: bool = False


class GRPOPolicyWrapper:
    """Wraps the GR00T policy to capture the initial denoising noise and raw action.

    This is the key GRPO extension. Before each inference call:
    1. Patch torch.randn to capture the first 3D tensor (the denoising noise)
    2. Hook get_action_with_features to capture the raw normalized action output
    Both are returned in the info dict for the collector to store per chunk.
    """

    def __init__(self, policy, device: str = "cuda", action_mask: np.ndarray | None = None):
        """
        Args:
            policy: The underlying Gr00tPolicy or Gr00tSimPolicyWrapper.
            device: Device for the model (used to identify action_head).
            action_mask: Pre-computed (max_horizon, max_dim) mask with 1s for valid
                dims of the current embodiment, 0s for padding. If None, attempts
                to derive it from the wrapped policy's processor/modality_configs.
        """
        self.policy = policy
        self.device = device
        self.action_mask = action_mask if action_mask is not None else compute_action_mask(policy)

    def get_action(self, observation, options=None):
        """Get action with initial noise and raw action capture.

        Extends the standard policy.get_action() interface to additionally capture:
        1. The initial noise tensor used to start the denoising loop
        2. The raw normalized action tensor (50×128) from the model output

        Both are returned in the info dict for the collector to store. During
        GRPO training, the initial noise is used as the shared ε for FM log-prob
        evaluation (evaluating the model along the actual denoising path).

        Args:
            observation: Batched observation dict from the environment.
            options: Optional dict (passed through to inner policy).

        Returns:
            Tuple of (action_dict, info_dict) where info_dict contains:
                - 'initial_noise': numpy array (B, 50, 128) — the noise that was denoised
                - 'raw_actions': numpy array (B, 50, 128) — normalized model output
        """
        # Navigate to the action head for hooking
        inner_policy = self.policy.policy if hasattr(self.policy, "policy") else self.policy
        if hasattr(inner_policy, "model"):
            action_head = inner_policy.model.action_head
        elif hasattr(inner_policy, "policy") and hasattr(inner_policy.policy, "model"):
            action_head = inner_policy.policy.model.action_head
        else:
            action_head = None

        captured_raw_action = [None]
        captured_initial_noise = [None]

        if action_head is not None:
            original_method = action_head.get_action_with_features

            def capturing_get_action_with_features(*args, **kwargs):
                result = original_method(*args, **kwargs)
                captured_raw_action[0] = result["action_pred"].detach().cpu().numpy()
                return result

            # Patch torch.randn to capture the initial noise tensor.
            # In get_action_with_features(), the first 3D randn call is the denoising
            # starting noise: torch.randn(batch_size, action_horizon, action_dim)
            _original_randn = torch.randn

            def _capturing_randn(*args, **kwargs):
                result = _original_randn(*args, **kwargs)
                # Capture the first 3D tensor produced (the denoising noise)
                if captured_initial_noise[0] is None and result.dim() == 3:
                    captured_initial_noise[0] = result.detach().cpu().numpy()
                return result

            action_head.get_action_with_features = capturing_get_action_with_features
            torch.randn = _capturing_randn

        # Call the underlying policy
        try:
            action, info = self.policy.get_action(observation, options)
        finally:
            # Always restore originals
            if action_head is not None:
                action_head.get_action_with_features = original_method
                torch.randn = _original_randn

        # Attach captured data to info for client retrieval
        if not isinstance(info, dict):
            info = {}
        if captured_raw_action[0] is not None:
            info["raw_actions"] = captured_raw_action[0]
        if captured_initial_noise[0] is not None:
            info["initial_noise"] = captured_initial_noise[0]
        if self.action_mask is not None and captured_raw_action[0] is not None:
            B = captured_raw_action[0].shape[0]
            info["action_mask"] = np.broadcast_to(
                self.action_mask[np.newaxis], (B,) + self.action_mask.shape
            ).copy()

        return action, info

    def reset(self, options=None):
        """Reset policy state (delegate to inner policy).

        Args:
            options: Optional reset configuration (passed through to inner policy).
                     The PolicyServer calls this with options={'client_id': ...}.
        """
        if hasattr(self.policy, "reset"):
            return self.policy.reset(options)
        return {}

    def get_modality_config(self):
        """Get modality config (delegate to inner policy)."""
        if hasattr(self.policy, "get_modality_config"):
            return self.policy.get_modality_config()
        return {}


def compute_action_mask(policy) -> np.ndarray | None:
    """Compute the per-embodiment action mask from a wrapped policy.

    The model always outputs a padded (max_action_horizon, max_action_dim) tensor.
    For a given embodiment, only a sub-rectangle corresponds to valid action dims.
    This mask lets FM log-prob ignore the padded region during training.

    Args:
        policy: Gr00tPolicy, Gr00tSimPolicyWrapper, or an _InPlacePolicy wrapping
            a Gr00tN1d6 model. The inner policy must expose `.processor`,
            `.modality_configs` (per-embodiment sub-dict), and `.embodiment_tag`.

    Returns:
        Float32 ndarray of shape (max_action_horizon, max_action_dim), or None
        if the required attributes cannot be resolved.
    """
    inner = getattr(policy, "policy", policy)
    if not (hasattr(inner, "processor") and hasattr(inner, "modality_configs")
            and hasattr(inner, "embodiment_tag")):
        return None

    try:
        processor = inner.processor
        modality_configs = inner.modality_configs
        embodiment_tag = inner.embodiment_tag

        action_horizon = len(modality_configs["action"].delta_indices)
        # Note: processor.action_dim is only populated when set_statistics() is
        # called explicitly; it is NOT populated by from_pretrained. Use the
        # state_action_processor accessor, which is always populated when
        # statistics are loaded from the checkpoint.
        action_dim = int(processor.state_action_processor.get_action_dim(embodiment_tag.value))

        max_horizon = int(getattr(processor, "max_action_horizon", 50))
        max_dim = int(getattr(processor, "max_action_dim", 128))

        mask = np.zeros((max_horizon, max_dim), dtype=np.float32)
        mask[:action_horizon, :action_dim] = 1.0
        return mask
    except (AttributeError, KeyError, TypeError):
        return None


def create_grpo_server(config: GRPOServerConfig) -> PolicyServer:
    """Create a GRPO-extended policy server.

    This loads the model, optionally applies LoRA, wraps with the seed-tracking
    policy wrapper, and creates the ZMQ server.

    Args:
        config: GRPO server configuration.

    Returns:
        PolicyServer instance ready to run.
    """
    import gr00t.model  # noqa: F401 — registers model classes

    print(f"Loading model from {config.model_path}...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag[config.embodiment_tag],
        model_path=config.model_path,
        device=config.device,
    )

    # Apply LoRA if checkpoint provided
    if config.lora_checkpoint:
        from lora_dit import apply_lora_to_dit, load_lora_checkpoint
        print(f"Loading LoRA checkpoint from {config.lora_checkpoint}...")
        apply_lora_to_dit(policy.model)
        load_lora_checkpoint(policy.model, config.lora_checkpoint)
        print("LoRA weights loaded.")

    # Wrap with sim policy wrapper if needed (handles flat ↔ nested observation format)
    if config.use_sim_policy_wrapper:
        policy = Gr00tSimPolicyWrapper(policy)

    # Enable verbose denoising logs if requested
    if config.verbose:
        inner = policy.policy if hasattr(policy, "policy") else policy
        inner.model.action_head.verbose = True

    # Wrap with GRPO seed-tracking policy
    grpo_policy = GRPOPolicyWrapper(
        policy=policy,
        device=config.device,
    )

    # Create and return the ZMQ server
    server = PolicyServer(
        policy=grpo_policy,
        host=config.host,
        port=config.port,
    )

    print(f"\nGRPO server ready on {config.host}:{config.port}")
    print(f"  Model: {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  LoRA: {'loaded' if config.lora_checkpoint else 'none (base model)'}")
    print(f"  Density diagnostics: {config.use_density_diagnostics}")

    return server


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Launch the GRPO-extended model server."""
    import tyro

    config = tyro.cli(GRPOServerConfig)
    server = create_grpo_server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer shut down.")


if __name__ == "__main__":
    main()
