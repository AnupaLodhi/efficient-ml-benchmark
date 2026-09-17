"""Pruning methods.

Honesty note (this is load-bearing for the whole project's credibility, so
it's spelled out here rather than just in the paper):

`torch.nn.utils.prune` -- which both methods below use -- works by
*masking*: it zeroes out selected weights (or `prune.remove`s the mask to
bake the zeros permanently into the tensor). The tensor's shape, dtype, and
on-disk storage are unchanged. This means:

  - `nonzero_params` (see src/models/resnet.py) genuinely drops -- that's a
    real, measurable sparsity result, useful for RQ1.
  - `model_size_mb` (dense state_dict serialization) and CPU latency
    (dense matmul/conv over the same tensor shapes) do NOT improve from
    masking alone, at ANY sparsity level, because PyTorch's default dense
    tensors still store and compute over every zero.
  - Realizing an actual size/latency benefit from *unstructured* sparsity
    requires exporting to a sparse tensor format with runtime support (not
    generally available for conv layers in stock PyTorch/ONNX Runtime as of
    this writing) -- out of scope here, noted as a limitation.
  - *Structured* (channel) pruning CAN translate to real speedup/size
    reduction, but only if the pruned channels are physically removed by
    rebuilding smaller layers (and, for a residual network like ResNet-18,
    propagating the channel-count change through skip connections). The
    `structured_ln` method below only *masks* whole channels to zero -- it
    reports channel-level sparsity honestly, but this project does NOT
    implement dependency-aware physical channel removal (that is
    substantial additional engineering, e.g. what libraries like
    torch-pruning exist to solve), so no size/latency benefit should be
    claimed for it either, and none is claimed in the evaluation code.

Every function/report below is written to make this distinction explicit
rather than silently implying a benefit that wasn't actually produced.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def _prunable_conv_linear_modules(model: nn.Module):
    """Conv2d and Linear layers are the parameter-heavy, prunable layers.
    BatchNorm and the final classifier bias are excluded from pruning by
    convention (pruning BN scale/shift destabilizes training disproportionately
    for the params saved, and the classifier layer is small and accuracy-critical).
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield name, module


def apply_unstructured_pruning(
    model: nn.Module,
    amount: float,
    remove_reparam: bool = True,
) -> nn.Module:
    """L1 unstructured magnitude pruning: zero out the `amount` fraction of
    individual weights (across each layer independently) with the smallest
    absolute value. Modifies `model` in place and also returns it.

    `amount=0.0` is a no-op (returned unchanged) -- this is how the
    "baseline/no pruning" condition in the sparsity sweep is realized.
    """
    if amount <= 0.0:
        return model
    if not (0.0 < amount < 1.0):
        raise ValueError(f"amount must be in (0, 1), got {amount}")

    for _, module in _prunable_conv_linear_modules(model):
        prune.l1_unstructured(module, name="weight", amount=amount)
        if remove_reparam:
            prune.remove(module, "weight")
    return model


def finalize_pruning(model: nn.Module) -> nn.Module:
    """Bake active pruning masks into weight tensors.

    During post-pruning fine-tuning the masks must remain active so optimizer
    updates cannot regrow pruned weights. Call this only after fine-tuning,
    before final evaluation/export.
    """
    for _, module in _prunable_conv_linear_modules(model):
        if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
    return model


def apply_structured_pruning(model: nn.Module, amount: float, dim: int = 0) -> nn.Module:
    """Ln (L2) structured pruning: zero out entire output channels
    (dim=0, i.e. whole filters) with the smallest L2 norm, per layer.

    See module docstring: this masks whole channels to zero but does NOT
    physically shrink the tensors, so no latency/size benefit should be
    assumed from this alone -- only channel-level sparsity.
    """
    if amount <= 0.0:
        return model
    if not (0.0 < amount < 1.0):
        raise ValueError(f"amount must be in (0, 1), got {amount}")

    for _, module in _prunable_conv_linear_modules(model):
        # A layer with very few output channels (e.g. the 10-class final
        # Linear) can't have `amount` fraction validly removed by channel --
        # skip layers where amount would remove all channels.
        n_channels = module.weight.shape[dim]
        if int(amount * n_channels) >= n_channels:
            continue
        prune.ln_structured(module, name="weight", amount=amount, n=2, dim=dim)
        prune.remove(module, "weight")
    return model


def get_sparsity_report(model: nn.Module) -> dict:
    """Per-layer and overall zero-weight fraction, for reporting sparsity
    honestly and granularly (overall figure can hide layers that pruned to
    ~0% because they were skipped, e.g. tiny final Linear under structured
    pruning).
    """
    layer_reports = {}
    total, zero = 0, 0
    for name, module in _prunable_conv_linear_modules(model):
        w = module.weight.data
        n = w.numel()
        z = int((w == 0).sum().item())
        layer_reports[name] = {"total": n, "zero": z, "sparsity": z / n if n else 0.0}
        total += n
        zero += z

    return {
        "overall_sparsity": zero / total if total else 0.0,
        "total_prunable_params": total,
        "zero_prunable_params": zero,
        "per_layer": layer_reports,
    }
