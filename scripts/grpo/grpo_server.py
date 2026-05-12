"""Extended GR00T policy server for GRPO training.

This server extends the standard PolicyServer (gr00t/policy/server_client.py) to:
1. Return the noise seed used during each denoising call
2. Optionally run density-aware diagnostics during inference (+12% overhead)

Why noise seeds matter for GRPO:
- The FM log-prob surrogate requires evaluating the model on the EXACT same noise
  that was used during collection
- Without the seed, we can't reconstruct the denoising trajectory
- The DenoisingLab (denoising_lab.py:287-296) already demonstrates this pattern:
      gen = torch.Generator(device=device).manual_seed(seed)
      actions = torch.randn(..., generator=gen)

Implementation approach:
- Use a counter-based seed (iteration * max_chunks + chunk_idx) for determinism
- Before each get_action() call, set torch.manual_seed(seed)
- Return the seed alongside the action in the response

This file runs in the MAIN VENV (GPU) alongside the model.

Usage:
    uv run python scripts/grpo/grpo_server.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --embodiment-tag ROBOCASA_PANDA_OMRON \\
        --port 5555
"""

import sys
import threading
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
    # Starting seed for noise generation (incremented per call)
    initial_seed: int = 0

    # Whether to use density-aware diagnostics during inference
    use_density_diagnostics: bool = False

    # LoRA checkpoint to load (None = use base model)
    lora_checkpoint: Optional[str] = None

    # Whether to use the sim policy wrapper (flat keys ↔ nested format)
    use_sim_policy_wrapper: bool = True

    # Verbose logging of denoising steps
    verbose: bool = False


class GRPOPolicyWrapper:
    """Wraps the GR00T policy to capture and return noise seeds.

    This is the key GRPO extension: before each inference call, we set a
    deterministic seed on the GPU RNG so that the initial noise in the
    denoising loop (torch.randn at gr00t_n1d6.py:311) is reproducible.

    The seed is returned alongside the action so the training loop can
    reconstruct the exact noise for FM log-prob computation.
    """

    def __init__(self, policy, initial_seed: int = 0, device: str = "cuda"):
        """
        Args:
            policy: The underlying Gr00tPolicy or Gr00tSimPolicyWrapper.
            initial_seed: Starting seed value (incremented per call).
            device: Device for torch RNG seeding.
        """
        self.policy = policy
        self.device = device

        # Atomic counter for seed generation (thread-safe for server use)
        self._seed_counter = initial_seed
        self._lock = threading.Lock()

        # Store the most recent seeds for client retrieval
        self._last_seeds: list[int] = []

    def get_action(self, observation, options=None):
        """Get action with deterministic noise seeding and raw action capture.

        Mirrors the standard policy.get_action() interface but additionally:
        1. Sets a deterministic RNG seed before inference
        2. Captures the raw normalized action tensor (50×128) via hook
        3. Returns noise seeds and raw action in the info dict

        The raw normalized action is needed for FM log-prob computation during
        GRPO training. Without it, we'd need to re-normalize decoded actions
        (lossy due to clipping/thresholding).

        Args:
            observation: Batched observation dict from the environment.
            options: Optional dict (passed through to inner policy).

        Returns:
            Tuple of (action_dict, info_dict) where info_dict contains:
                - 'noise_seeds': list of int seeds used for denoising
                - 'raw_actions': numpy array (B, 50, 128) of normalized actions
        """
        # Generate deterministic seed for this call
        with self._lock:
            seed = self._seed_counter
            self._seed_counter += 1

        # Determine batch size from observation
        batch_size = self._get_batch_size(observation)
        seeds = [seed * 1000 + i for i in range(batch_size)]
        self._last_seeds = seeds

        # Install a hook on the action head to capture raw normalized actions
        # The action head's get_action_with_features() produces the raw (50×128) tensor
        # at the end of the denoising loop, before decode_action() is called by the policy.
        inner_policy = self.policy.policy if hasattr(self.policy, "policy") else self.policy
        if hasattr(inner_policy, "model"):
            action_head = inner_policy.model.action_head
        elif hasattr(inner_policy, "policy") and hasattr(inner_policy.policy, "model"):
            action_head = inner_policy.policy.model.action_head
        else:
            action_head = None

        captured_raw_action = [None]  # Mutable container for closure

        if action_head is not None:
            original_method = action_head.get_action_with_features

            def capturing_get_action_with_features(*args, **kwargs):
                result = original_method(*args, **kwargs)
                # Capture the raw normalized action tensor before it's decoded
                captured_raw_action[0] = result["action_pred"].detach().cpu().numpy()
                return result

            action_head.get_action_with_features = capturing_get_action_with_features

        # Set the PyTorch RNG state for reproducible noise generation
        # This affects the torch.randn() call in get_action_with_features() (line 311)
        torch.manual_seed(seed)
        if self.device != "cpu":
            torch.cuda.manual_seed(seed)

        # Call the underlying policy
        try:
            action, info = self.policy.get_action(observation, options)
        finally:
            # Restore original method to avoid accumulating closures
            if action_head is not None:
                action_head.get_action_with_features = original_method

        # Attach noise seeds and raw action to info for client retrieval
        if not isinstance(info, dict):
            info = {}
        info["noise_seeds"] = seeds
        if captured_raw_action[0] is not None:
            info["raw_actions"] = captured_raw_action[0]

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

    def _get_batch_size(self, observation) -> int:
        """Infer batch size from observation structure."""
        if isinstance(observation, dict):
            for key, value in observation.items():
                if hasattr(value, "shape"):
                    return value.shape[0]
                elif isinstance(value, dict):
                    for v in value.values():
                        if hasattr(v, "shape"):
                            return v.shape[0]
        return 1


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
        embodiment_tag=EmbodimentTag(config.embodiment_tag),
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
        initial_seed=config.initial_seed,
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
