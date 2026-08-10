"""Low-Rank Adaptation (LoRA) for GPT-2.

The pretrained weights stay frozen and each training stage learns a low-rank
update instead:

    W_effective = W + (A @ B) * (alpha / r)

A stage therefore adds a few MB of trainable weights rather than the ~500MB a
full GPT-2 checkpoint would take.

GPT-2 stores its projections in `transformers.Conv1D` modules whose weight is
laid out as [in_features, out_features], i.e. transposed relative to
`nn.Linear`.  Both layouts are handled below.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("c_attn", "c_proj")


def _feature_dims(module: nn.Module) -> tuple[int, int]:
    """Return (in_features, out_features) for a Linear or a GPT-2 Conv1D."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    weight = module.weight  # Conv1D: [in_features, out_features]
    return weight.shape[0], weight.shape[1]


class LoRALayer(nn.Module):
    """Wraps a frozen projection and adds a trainable low-rank branch."""

    def __init__(self, base: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        in_features, out_features = _feature_dims(base)
        self.rank = rank
        self.scaling = alpha / rank
        self.enabled = True

        # The adapter is created on the device and in the dtype of the weight it
        # wraps, so injecting it into an already placed model needs no follow-up
        # call to `.to(...)`.
        like = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, **like))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, **like))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A is initialised like a regular layer and B with zeros, so the adapter
        # is a no-op at step 0 and training starts exactly from the base model.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        update = (self.dropout(x) @ self.lora_A) @ self.lora_B
        return out + update * self.scaling

    @torch.no_grad()
    def merge_(self) -> nn.Module:
        """Fold the adapter into the frozen weight and return the base module."""
        delta = (self.lora_A @ self.lora_B) * self.scaling
        if isinstance(self.base, nn.Linear):
            self.base.weight.add_(delta.T)
        else:
            self.base.weight.add_(delta)
        return self.base


def _iter_named_parents(model: nn.Module):
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            yield module, child_name, child, f"{name}.{child_name}" if name else child_name


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> nn.Module:
    """Replace every targeted projection with a LoRA-wrapped copy.

    The default targets are the attention projections; the MLP blocks stay
    frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

    replacements = [
        (parent, child_name, child)
        for parent, child_name, child, _ in _iter_named_parents(model)
        if child_name in targets and not isinstance(child, LoRALayer)
    ]
    for parent, child_name, child in replacements:
        setattr(parent, child_name, LoRALayer(child, rank=rank, alpha=alpha, dropout=dropout))
    return model


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if "lora_" in name]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.state_dict().items()
        if "lora_" in name
    }


def save_lora(model: nn.Module, path: str, metadata: dict | None = None) -> None:
    torch.save({"lora": lora_state_dict(model), "metadata": metadata or {}}, path)


def load_lora(model: nn.Module, path: str, device: str = "cpu") -> dict:
    """Load adapter weights into a model that already has LoRA injected."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["lora"] if "lora" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if "lora_" in key]
    if unexpected:
        raise RuntimeError(f"Adapter does not match the model: {unexpected[:5]}")
    return checkpoint.get("metadata", {})


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its frozen weight and remove the LoRA wrappers.

    The RL stage merges the supervised adapter before attaching its own, so
    disabling the RL adapter recovers the supervised model without holding a
    second copy of GPT-2 in memory.
    """
    replacements = [
        (parent, child_name, child)
        for parent, child_name, child, _ in _iter_named_parents(model)
        if isinstance(child, LoRALayer)
    ]
    for parent, child_name, child in replacements:
        setattr(parent, child_name, child.merge_())
    return model


@contextmanager
def adapters_disabled(model: nn.Module):
    """Temporarily run the model without its adapters (the reference policy)."""
    layers = [m for m in model.modules() if isinstance(m, LoRALayer)]
    previous = [layer.enabled for layer in layers]
    for layer in layers:
        layer.enabled = False
    try:
        yield model
    finally:
        for layer, state in zip(layers, previous):
            layer.enabled = state


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
