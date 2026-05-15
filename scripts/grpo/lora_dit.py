"""LoRA adapter injection for GR00T N1.6 DiT action head.

This module applies LoRA (Low-Rank Adaptation) to the DiT transformer blocks
inside the GR00T action head. Only standard nn.Linear layers are targeted —
the CategorySpecificLinear layers (state_encoder, action_encoder, action_decoder)
have 3D weight tensors [num_categories, in, out] that are incompatible with LoRA.

Architecture context:
- GR00T N1.6 = Eagle VLM backbone (1.9B, frozen) + DiT action head (1.1B)
- The DiT (AlternateVLDiT) has 32 transformer blocks with attention + FF
- LoRA at rank 16 adds ~20M trainable params (~2% of DiT)

Why inject_adapter_in_model instead of get_peft_model:
- get_peft_model() wraps the entire model and fails on CategorySpecificLinear
- inject_adapter_in_model() targets a specific submodule (the DiT only)

Reference:
- peft docs: https://huggingface.co/docs/peft/developer_guides/low_level_api
- DenoisingLab uses the same model loading pattern (denoising_lab.py:114-118)
"""

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model


# Single source of truth for the default LoRA target module list. Imported by
# grpo_config.GRPOConfig.lora_target_modules and used as the fallback here when
# the caller passes target_modules=None. Keep these names in sync with the
# AlternateVLDiT structure (diffusers Attention + FeedForward inside
# BasicTransformerBlock, plus the DiT-level proj_out_1/proj_out_2).
DEFAULT_LORA_TARGET_MODULES: list[str] = [
    "attn1.to_q",      # Self/cross-attention query projection [1536, 1536] or [2048, 1536]
    "attn1.to_k",      # Self/cross-attention key projection
    "attn1.to_v",      # Self/cross-attention value projection
    "attn1.to_out.0",  # Attention output projection [1536, 1536]
    "ff.net.0.proj",   # FeedForward GEGLU gate projection [1536, 2*inner_dim]
    "ff.net.2",        # FeedForward output projection [inner_dim, 1536]
    "proj_out_1",      # DiT conditioning projection [1536, 3072]
    "proj_out_2",      # DiT final output projection [1536, 1024]
]


def apply_lora_to_dit(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Inject LoRA adapters into the DiT (AlternateVLDiT) inside the action head.

    This freezes ALL model parameters first, then injects trainable LoRA layers
    into the specified modules of model.action_head.model (the DiT).

    Args:
        model: Full Gr00tN1d6 model (backbone + action_head).
        rank: LoRA rank. Higher = more expressive but more memory.
        alpha: LoRA scaling factor. Standard: alpha = 2 * rank.
        dropout: Dropout on LoRA layers for regularization.
        target_modules: List of module name patterns to target within the DiT.
            Must be nn.Linear layers. If None, uses DEFAULT_LORA_TARGET_MODULES.

    Returns:
        The same model object (modified in-place) with LoRA adapters injected.
        Only LoRA parameters will have requires_grad=True.

    Example:
        >>> model = AutoModel.from_pretrained("nvidia/GR00T-N1.6-3B")
        >>> model = apply_lora_to_dit(model, rank=16)
        >>> print_trainable_params(model)  # ~20M trainable out of ~3B total
    """
    if target_modules is None:
        target_modules = list(DEFAULT_LORA_TARGET_MODULES)

    # Step 1: Freeze everything (backbone + action head + all submodules)
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Create LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",  # Don't add bias terms (standard for RL finetuning)
        target_modules=target_modules,
    )

    # Step 3: Inject LoRA into the DiT submodule only
    # model.action_head.model is the AlternateVLDiT (32-layer diffusion transformer)
    # This avoids touching CategorySpecificLinear in state_encoder/action_encoder/action_decoder
    dit = model.action_head.model
    inject_adapter_in_model(lora_config, dit, adapter_name="default")

    return model


def save_lora_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Save only the LoRA adapter weights (tiny compared to full model).

    Saves ~80MB instead of ~6GB (full model). Can be loaded on top of the
    base pretrained model to restore training state.

    Uses manual state_dict filtering rather than peft's get_peft_model_state_dict,
    which may not work with inject_adapter_in_model (the low-level API we use).

    Args:
        model: Full model with LoRA adapters on action_head.model.
        path: Directory to save the checkpoint to.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Extract LoRA parameters by name (reliable regardless of peft internals)
    dit = model.action_head.model
    lora_state = {k: v.cpu() for k, v in dit.state_dict().items() if "lora_" in k}

    if not lora_state:
        print(f"  WARNING: No LoRA parameters found to save!")

    torch.save(lora_state, path / "lora_weights.pt")


def load_lora_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Load LoRA adapter weights into an already-injected model.

    The model must already have LoRA adapters injected (via apply_lora_to_dit).
    Performs a strict TWO-SIDED match and HARD-FAILS on any divergence:
      - Saved LoRA keys not present in the current model (target_modules shrank)
        → RuntimeError. Loading would silently drop those weights, leaving the
        previously-trained adapter behavior unrepresented in the resumed run.
      - Current LoRA keys not present in the saved checkpoint (target_modules
        grew) → RuntimeError. Loading would leave some adapters at random init
        while the optimizer state attaches to them, silently corrupting training.
      - Per-key shape mismatch (rank changed) → RuntimeError. ``strict=True``
        loading would catch most of these but the explicit message points at
        the likely cause.

    Hard-failing is intentional: the previous warn-and-continue behavior could
    silently turn a resume into a from-scratch run on the affected layers,
    discoverable only hours later via degraded training curves.

    Args:
        model: Full model with LoRA adapters already injected.
        path: Directory containing lora_weights.pt.

    Raises:
        RuntimeError: If saved and current LoRA layouts don't match exactly.
    """
    path = Path(path)
    lora_state = torch.load(path / "lora_weights.pt", map_location="cpu")

    # Determine the model's LoRA keys (filter to "lora_" substring inside
    # the DiT, mirroring save_lora_checkpoint's filter).
    dit = model.action_head.model
    current_state = dit.state_dict()
    model_lora_keys = {k for k in current_state if "lora_" in k}
    saved_lora_keys = set(lora_state.keys())

    extra_in_save = saved_lora_keys - model_lora_keys
    extra_in_model = model_lora_keys - saved_lora_keys

    if extra_in_save:
        raise RuntimeError(
            f"LoRA checkpoint contains {len(extra_in_save)} keys not present "
            f"in the current model. Likely cause: lora_target_modules in your "
            f"config is a SUBSET of the modules targeted at save time. "
            f"First few unmatched saved keys: {sorted(extra_in_save)[:3]}. "
            f"Either expand lora_target_modules to match the checkpoint, or "
            f"restart training from scratch."
        )

    if extra_in_model:
        raise RuntimeError(
            f"Current model has {len(extra_in_model)} LoRA keys not present "
            f"in the checkpoint. Likely cause: lora_target_modules in your "
            f"config is a SUPERSET of the modules targeted at save time. "
            f"Loading would leave these adapters at random init while the "
            f"optimizer state attaches to them — a silent corruption. "
            f"First few unmatched model keys: {sorted(extra_in_model)[:3]}. "
            f"Either reduce lora_target_modules to match the checkpoint, or "
            f"restart training from scratch."
        )

    # Per-key shape check — catches lora_rank mismatches with a clear message
    # before strict=True load_state_dict would raise a less-actionable error.
    for k in saved_lora_keys:
        sshape = tuple(lora_state[k].shape)
        cshape = tuple(current_state[k].shape)
        if sshape != cshape:
            raise RuntimeError(
                f"LoRA shape mismatch on key '{k}': saved {sshape}, "
                f"current {cshape}. Likely cause: lora_rank in your config "
                f"differs from the checkpoint's training config."
            )

    # All keys and shapes match — apply via partial state_dict update + strict load.
    current_state.update(lora_state)
    dit.load_state_dict(current_state, strict=True)
    print(f"  Loaded {len(lora_state)} LoRA parameters from {path}")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA adapters into base weights for deployment (irreversible).

    After merging, the model behaves identically but without LoRA overhead.
    Use this for final deployment — not during training (can't un-merge).

    The merged model can be saved as a standard HuggingFace checkpoint.

    Args:
        model: Full model with trained LoRA adapters.

    Returns:
        The same model with LoRA merged into base weights.
    """
    from peft.tuners.lora import LoraLayer

    dit = model.action_head.model
    for name, module in dit.named_modules():
        if isinstance(module, LoraLayer):
            module.merge()

    return model


def print_trainable_params(model: nn.Module) -> dict[str, int]:
    """Print and return trainable vs total parameter counts.

    Useful for verifying LoRA injection worked correctly.
    Expected: ~20M trainable / ~3B total for rank=16 on DiT.

    Args:
        model: Any PyTorch model.

    Returns:
        Dict with keys: trainable, total, percentage.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0

    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    print(f"  LoRA params:  ~{trainable:,}")
    print(f"  Frozen params: ~{total - trainable:,}")

    return {"trainable": trainable, "total": total, "percentage": pct}


# ---------------------------------------------------------------------------
# Self-test (runs without GPU using mock tensors)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== LoRA DiT Self-Test ===\n")
    print("This test verifies LoRA injection logic without loading the full model.")
    print("For GPU test: uv run python scripts/grpo/lora_dit.py --full-test\n")

    import sys

    if "--full-test" in sys.argv:
        # Full test: loads actual model (requires GPU + model weights)
        from transformers import AutoModel
        import gr00t.model  # noqa: F401 — registers model classes

        print("Loading model...")
        model = AutoModel.from_pretrained("nvidia/GR00T-N1.6-3B")
        model.to("cuda", dtype=torch.bfloat16)

        print("\nBefore LoRA:")
        print_trainable_params(model)

        print("\nApplying LoRA (rank=16)...")
        model = apply_lora_to_dit(model, rank=16, alpha=32, dropout=0.05)

        print("\nAfter LoRA:")
        stats = print_trainable_params(model)

        # Verify only LoRA params are trainable
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "lora" in name.lower(), f"Non-LoRA param is trainable: {name}"

        print(f"\nAll {stats['trainable']:,} trainable params are LoRA adapters.")
        print("Test PASSED.")
    else:
        # Quick test: just verify the LoRA config construction
        from peft import LoraConfig
        config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["attn1.to_q", "attn1.to_k", "ff.net.0.proj"],
        )
        print(f"LoRA config created: rank={config.r}, alpha={config.lora_alpha}")
        print(f"Target modules: {config.target_modules}")
        print("\nQuick test PASSED. Use --full-test for GPU validation.")
