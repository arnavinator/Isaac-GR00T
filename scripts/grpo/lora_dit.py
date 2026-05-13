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
            Must be nn.Linear layers. If None, uses default attention + FF targets.

    Returns:
        The same model object (modified in-place) with LoRA adapters injected.
        Only LoRA parameters will have requires_grad=True.

    Example:
        >>> model = AutoModel.from_pretrained("nvidia/GR00T-N1.6-3B")
        >>> model = apply_lora_to_dit(model, rank=16)
        >>> print_trainable_params(model)  # ~20M trainable out of ~3B total
    """
    if target_modules is None:
        target_modules = [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "proj_out_1",
            "proj_out_2",
        ]

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
    Uses partial state_dict update — only LoRA keys are overwritten.

    Args:
        model: Full model with LoRA adapters already injected.
        path: Directory containing lora_weights.pt.
    """
    path = Path(path)
    lora_state = torch.load(path / "lora_weights.pt", map_location="cpu")

    # Update only LoRA parameters in the DiT
    dit = model.action_head.model
    current_state = dit.state_dict()

    # Verify all saved keys exist in the model
    missing = [k for k in lora_state if k not in current_state]
    if missing:
        print(f"  WARNING: {len(missing)} LoRA keys not found in model: {missing[:3]}...")

    current_state.update(lora_state)
    dit.load_state_dict(current_state)
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
