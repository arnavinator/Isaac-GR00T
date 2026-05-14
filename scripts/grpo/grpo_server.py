"""Extended GR00T policy server for GRPO training.

This server extends the standard PolicyServer (gr00t/policy/server_client.py) to:
1. Capture the initial noise tensor used during denoising
2. Capture the raw normalized action tensor (50×128) from the DiT output

Why initial_noise matters for GRPO:
- The FM log-prob surrogate evaluates the velocity field along an interpolation path
  x_τ = (1-τ)ε + τ*action. Using the SAME ε for both current and ref models ensures
  the importance ratio reflects only the model difference, not estimation noise.
- We capture ε₀ (the actual noise that was denoised into the action) so training
  can evaluate along the true path the model took, rather than random paths.

Implementation approach:
- At module import, install a thread-local-aware wrapper around torch.randn
  exactly once. Other threads see a pass-through; only the thread that sets
  a thread-local capture context gets its first 3D randn recorded.
- During GRPOPolicyWrapper.get_action, monkey-patch the action head's
  get_action_with_features. The wrapper sets/clears the thread-local context
  so noise capture is strictly scoped to the denoising call; it also converts
  the raw action tensor to float32 before numpy (bfloat16 has no numpy dtype).
- Restore the action head attribute in a try/finally block.

This file runs in the MAIN VENV (GPU) alongside the model.

Usage:
    uv run python scripts/grpo/grpo_server.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --embodiment-tag ROBOCASA_PANDA_OMRON \\
        --port 5555
"""

import sys
import threading
from contextlib import nullcontext
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


# ---------------------------------------------------------------------------
# Thread-local noise capture
# ---------------------------------------------------------------------------
# The denoising noise is created via torch.randn inside Gr00tN1d6ActionHead.
# We need to capture it, but patching torch.randn globally for every
# GRPOPolicyWrapper.get_action call is a concurrency hazard: if the main
# training thread's torch.randn collides with the server thread's patch, the
# main thread ends up routing through our capture function and gets wrong
# semantics.
#
# Instead, install the torch.randn override ONCE at module load and route
# capture state through threading.local(). Other threads see a pass-through;
# only the thread that set _grpo_tls.capture gets its first 3D randn recorded.
_grpo_tls = threading.local()
_original_randn = torch.randn


def _grpo_capturing_randn(*args, **kwargs):
    result = _original_randn(*args, **kwargs)
    ctx = getattr(_grpo_tls, "capture", None)
    if ctx is not None and ctx.get("noise") is None and result.dim() == 3:
        # Convert to float32 first: numpy has no native bfloat16 representation,
        # so .cpu().numpy() on a bfloat16 tensor raises TypeError. The captured
        # noise is eventually stacked back into bfloat16 on the GPU for training,
        # so the round-trip through float32 numpy is lossless.
        ctx["noise"] = result.detach().float().cpu().numpy()
    return result


# Install exactly once; subsequent imports are no-ops.
if torch.randn is not _grpo_capturing_randn:
    torch.randn = _grpo_capturing_randn


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

    # LoRA checkpoint to load (None = use base model)
    lora_checkpoint: Optional[str] = None

    # LoRA architecture — must match the rank/alpha/targets used at training time.
    # A mismatch silently drops checkpoint keys (load_lora_checkpoint only warns),
    # leaving the server running the base model.
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: Optional[list[str]] = None  # None = use lora_dit defaults

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

    def __init__(
        self,
        policy,
        device: str = "cuda",
        action_mask: np.ndarray | None = None,
        model_lock: "threading.RLock | None" = None,
    ):
        """
        Args:
            policy: The underlying Gr00tPolicy or Gr00tSimPolicyWrapper.
            device: Device for the model (used to identify action_head).
            action_mask: Pre-computed (max_horizon, max_dim) mask with 1s for valid
                dims of the current embodiment, 0s for padding. If None, attempts
                to derive it from the wrapped policy's processor/modality_configs.
            model_lock: Optional re-entrant lock serializing model forward/backward
                across the server thread and a sibling training thread. When set,
                the entire get_action body (including monkey-patching and the
                underlying denoising call) runs inside the lock so the trainer
                can safely take the same lock during _grpo_update / backward
                passes. If None, no locking is applied (standalone-server use).
        """
        self.policy = policy
        self.device = device
        self.action_mask = action_mask if action_mask is not None else compute_action_mask(policy)
        self.model_lock = model_lock

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
        capture_ctx = {"noise": None}

        # Serialize model access with the trainer's update loop when sharing the
        # same Gr00tN1d6 instance across threads (see train_grpo.GRPOTrainer).
        # nullcontext for the standalone-server case (no sibling trainer thread).
        lock_ctx = self.model_lock if self.model_lock is not None else nullcontext()

        with lock_ctx:
            if action_head is not None:
                original_method = action_head.get_action_with_features

                def capturing_get_action_with_features(*args, **kwargs):
                    # Scope noise capture to ONLY the denoising call. torch.randn
                    # calls outside this window (from other threads, or from earlier
                    # stages of this call) are pass-through.
                    _grpo_tls.capture = capture_ctx
                    try:
                        result = original_method(*args, **kwargs)
                    finally:
                        _grpo_tls.capture = None
                    # action_pred inherits vl_embeds.dtype (bfloat16). numpy has no
                    # native bfloat16, so convert via float32 before .numpy().
                    captured_raw_action[0] = (
                        result["action_pred"].detach().float().cpu().numpy()
                    )
                    return result

                action_head.get_action_with_features = capturing_get_action_with_features

            # Call the underlying policy
            try:
                action, info = self.policy.get_action(observation, options)
            finally:
                # Always restore the original method (capture context is already
                # cleared inside capturing_get_action_with_features, so nothing
                # else to undo here).
                if action_head is not None:
                    action_head.get_action_with_features = original_method

        # Fail loudly if a refactor breaks capture. Silent-None propagation
        # would later surface as a hard error in _prepare_batch (missing
        # initial_noise) — but it's cleaner to fail at the capture site so the
        # offending call is obvious. Only assert when the action_head could
        # actually be hooked; with action_head=None we never had a chance to
        # capture and the caller presumably doesn't need GRPO-specific data.
        if action_head is not None:
            if captured_raw_action[0] is None:
                raise RuntimeError(
                    "GRPOPolicyWrapper: raw_action capture failed. "
                    "capturing_get_action_with_features did not see a "
                    "result['action_pred'] — did get_action_with_features's "
                    "return contract change?"
                )
            if capture_ctx["noise"] is None:
                raise RuntimeError(
                    "GRPOPolicyWrapper: initial_noise capture failed. "
                    "No 3-D torch.randn call was observed during denoising — "
                    "did gr00t_n1d6.get_action_with_features switch to "
                    "torch.randn_like or a pre-allocated buffer? The "
                    "_grpo_capturing_randn hook intercepts torch.randn only."
                )

        # Attach captured data to info for client retrieval
        if not isinstance(info, dict):
            info = {}
        if captured_raw_action[0] is not None:
            info["raw_actions"] = captured_raw_action[0]
        if capture_ctx["noise"] is not None:
            info["initial_noise"] = capture_ctx["noise"]
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


def compute_action_mask(policy) -> np.ndarray:
    """Compute the per-embodiment action mask from a wrapped policy.

    The model always outputs a padded (max_action_horizon, max_action_dim) tensor.
    For a given embodiment, only a sub-rectangle corresponds to valid action dims.
    This mask lets FM log-prob ignore the padded region during training.

    A wrong/missing mask silently corrupts the training signal (the FM loss would
    include the padded region, which is meaningless for the current embodiment),
    so this function raises rather than returning None on any error path —
    GRPOPolicyWrapper will then refuse to construct, and the user gets a clear
    error at server-start instead of degraded training quality.

    Args:
        policy: Gr00tPolicy, Gr00tSimPolicyWrapper, or an _InPlacePolicy wrapping
            a Gr00tN1d6 model. The inner policy must expose `.processor`,
            `.modality_configs` (per-embodiment sub-dict), and `.embodiment_tag`.

    Returns:
        Float32 ndarray of shape (max_action_horizon, max_action_dim).

    Raises:
        RuntimeError: if any required attribute is missing or statistics are not
            loaded on the processor.
    """
    inner = getattr(policy, "policy", policy)
    missing = [
        attr for attr in ("processor", "modality_configs", "embodiment_tag")
        if not hasattr(inner, attr)
    ]
    if missing:
        raise RuntimeError(
            f"compute_action_mask: policy is missing required attribute(s) "
            f"{missing}. Expected a Gr00tPolicy/Gr00tSimPolicyWrapper-shaped "
            f"object exposing processor, modality_configs, embodiment_tag. "
            f"Got policy of type {type(policy).__name__}."
        )

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
    except (AttributeError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"compute_action_mask: failed to derive mask for "
            f"embodiment={getattr(inner, 'embodiment_tag', '?')}. Most common "
            f"cause: processor statistics not loaded (set_statistics not called "
            f"or checkpoint missing norm_params). Original error: "
            f"{type(e).__name__}: {e}"
        ) from e


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
        print(
            f"Loading LoRA checkpoint from {config.lora_checkpoint} "
            f"(rank={config.lora_rank}, alpha={config.lora_alpha})..."
        )
        apply_lora_to_dit(
            policy.model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        load_lora_checkpoint(policy.model, config.lora_checkpoint)
        print("LoRA weights loaded.")

    # Wrap with sim policy wrapper if needed (handles flat ↔ nested observation format)
    # strict=False to match train_grpo.py's in-process server setup
    # (train_grpo.py:_start_server_thread constructs the wrapper the same way).
    # Standalone runs also disable validation because the collector sends obs
    # that are already in the flat sim format and exact dtype/shape matches
    # to the validator aren't guaranteed across embodiments.
    if config.use_sim_policy_wrapper:
        policy = Gr00tSimPolicyWrapper(policy, strict=False)

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
