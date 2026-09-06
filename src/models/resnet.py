"""ResNet-18 for CIFAR-10.

We reuse torchvision's ResNet building blocks (BasicBlock) but replace the
ImageNet-style stem. Rationale:

The standard torchvision `resnet18` stem is a 7x7 stride-2 conv followed by
a stride-2 max-pool, designed for 224x224 ImageNet input; applied to 32x32
CIFAR-10 images it downsamples to an 8x8 feature map after the stem alone,
throwing away most spatial information before a single residual block runs.
This is a well-documented issue (see He et al., 2016, and the widely-used
"ResNet for CIFAR" variant) -- the standard fix, which we adopt here, is a
3x3 stride-1 stem with no max-pool. This is a *methodological choice*, not a
free-form architecture invention: it is the standard CIFAR-ResNet recipe used
throughout the pruning/quantization literature we will cite in Phase 8, and
we report it explicitly here so the paper's Methodology section can point to
this exact difference from the ImageNet stem.

Setting `cifar_stem: false` in the config falls back to the unmodified
torchvision stem, kept only as an ablation option -- not recommended for the
main results.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_cifar(num_classes: int = 10, cifar_stem: bool = True) -> nn.Module:
    """Build a ResNet-18 sized for CIFAR-10.

    Args:
        num_classes: number of output classes (10 for CIFAR-10).
        cifar_stem: if True, replace the ImageNet stem (7x7 s2 conv + maxpool)
            with the standard CIFAR stem (3x3 s1 conv, no maxpool).

    Returns:
        An nn.Module. Weights are randomly initialized (pretrained=False) --
        this project trains from scratch on CIFAR-10, since ImageNet-pretrained
        weights assume a different input resolution/stem and would confound
        the compression comparison with a transfer-learning effect.
    """
    model = resnet18(weights=None, num_classes=num_classes)

    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    return model


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def count_nonzero_parameters(model: nn.Module) -> dict:
    """Return total vs. non-zero parameter counts (relevant after pruning).

    For an unpruned dense model these two numbers are equal. After
    unstructured magnitude pruning, `nonzero_params` drops while
    `total_params` (and, absent special sparse storage, the on-disk file
    size) stays the same -- which is precisely the distinction the pruning
    experiments in Phase 3 need to report honestly.
    """
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int(torch.count_nonzero(p).item())
    return {"total_params": total, "nonzero_params": nonzero}


if __name__ == "__main__":
    # Quick manual sanity check: python -m src.models.resnet
    m = build_resnet18_cifar()
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print("output shape:", y.shape)
    print(count_parameters(m))
