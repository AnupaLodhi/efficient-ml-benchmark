"""Quantization methods for CPU deployment.

Distinguishing the three PyTorch-native approaches used here (per the
project spec's requirement to keep these clearly separate):

- **FP16 (half precision)**: every weight/activation stored/computed as
  float16 instead of float32. Halves the theoretical size; CPU speed
  benefit is not guaranteed since many CPU kernels compute fp16 by casting
  up to fp32 internally -- measured honestly below, not assumed.

- **Dynamic INT8 quantization** (`torch.ao.quantization.quantize_dynamic`):
  weights are quantized to int8 ahead of time; activations are quantized
  on-the-fly per-batch at inference time. In PyTorch's eager-mode
  implementation this is only supported for `nn.Linear` (and RNN/LSTM/GRU,
  irrelevant here). **This matters a lot for a Conv-dominated network like
  ResNet-18**: only the final `fc` layer (a small fraction of the model's
  ~11.17M parameters) is actually quantized by this method. Applying
  dynamic quantization to a CNN and reporting a large compression ratio
  would misrepresent what happened -- the report explicitly states how many
  parameters were actually affected.

- **Static (post-training) INT8 quantization**: BOTH weights and
  activations are quantized to int8 ahead of time, with activation ranges
  determined by running calibration data through the model first. Unlike
  dynamic quantization, PyTorch's eager-mode static quantization DOES
  support `nn.Conv2d` (after fusing Conv+BN(+ReLU) and inserting
  Quant/DeQuant stubs), making it the method that actually exercises the
  bulk of ResNet-18's parameters. This is the primary method expected to
  show a real CPU size/latency benefit for this architecture.

- **Quantization-aware training (QAT)** is intentionally NOT implemented
  in this project (also not listed in configs/config.yaml
  `quantization.methods`). It requires re-running training with fake-quant
  ops inserted, which is a substantially larger compute cost than this
  project's other experiments, given the CPU-only training budget
  documented in PHASE_LOG.md. Flagged here as a scoped-out extension for
  the paper's Limitations section, not silently omitted.

Reproducibility risk discovered while building this (record, don't hide):
on the PyTorch build this project was developed against (2.14.0), every
call into `torch.ao.quantization` (used by both dynamic and static
quantization below) raises a `DeprecationWarning` stating that eager-mode
quantization will be removed in PyTorch 2.10+, with `torchao`'s `quantize_`
API and PT2E (`prepare_pt2e`/`convert_pt2e`) named as the replacements.
The API still works correctly as of this version (verified by the tests in
tests/test_quantization.py, including a real measured ~4x size reduction
for static quantization), but this is worth flagging explicitly in the
paper's Methodology/Limitations: results reported here may not reproduce
verbatim on a PyTorch version where this API has actually been removed,
and a `torchao`-based reimplementation would be the natural fix at that
point.
"""
from __future__ import annotations

import copy

import torch
import torch.ao.quantization as tq
import torch.nn as nn
from torch.utils.data import DataLoader


import torch.ao.nn.quantized as nnq
from torchvision.models.resnet import BasicBlock


class _QuantizableBasicBlock(BasicBlock):
    """A ResNet BasicBlock whose residual addition uses
    `nn.quantized.FloatFunctional` instead of the plain `out += identity`.

    Discovered via smoke-testing (not assumed up front): PyTorch's eager-mode
    static quantization converts Conv2d/Linear/BatchNorm to quantized
    equivalents, but does NOT support the `aten::add` op directly on two
    already-quantized tensors -- attempting static quantization on a plain
    torchvision ResNet fails at the skip connection with
    `NotImplementedError: Could not run 'aten::add.out' ... 'QuantizedCPU'`.
    The documented fix (also used in torchvision's own `quantization.resnet`
    reference models) is to route the addition through
    `nnq.FloatFunctional().add(...)`, which knows how to handle quantized
    tensors. This subclass makes that swap; `_make_quantizable` below
    converts an existing (already-trained) model's blocks to use it in
    place, so no weights are lost or re-initialized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_add = nnq.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)
        return out


def _make_quantizable(model: nn.Module) -> nn.Module:
    """Replace every BasicBlock in `model` with a `_QuantizableBasicBlock`
    that reuses the exact same conv/bn/relu/downsample submodules (no
    weights are copied or reinitialized -- only the forward()'s residual-add
    implementation changes). Operates on `model` in place and returns it.
    """
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for i, block in enumerate(layer):
            new_block = _QuantizableBasicBlock.__new__(_QuantizableBasicBlock)
            nn.Module.__init__(new_block)
            new_block.conv1 = block.conv1
            new_block.bn1 = block.bn1
            new_block.relu = block.relu
            new_block.conv2 = block.conv2
            new_block.bn2 = block.bn2
            new_block.downsample = block.downsample
            new_block.stride = block.stride
            new_block.skip_add = nnq.FloatFunctional()
            layer[i] = new_block
    return model


def to_fp16(model: nn.Module) -> nn.Module:
    """Convert a model to float16. Returns a new model; does not mutate the input."""
    return copy.deepcopy(model).half()


def count_actually_quantized_params(model: nn.Module, quantized_module_types: tuple) -> dict:
    """How many parameters live in layers that were actually converted to a
    quantized module type, vs. total -- the honesty check for dynamic
    quantization's limited Conv2d support.

    NOTE: quantized modules (e.g. `torch.ao.nn.quantized.dynamic.Linear`)
    store weights as packed/opaque params that do NOT show up via the
    normal `.parameters()` iterator -- an earlier version of this function
    used `.parameters()` and silently reported 0 quantized params even when
    quantization had visibly succeeded. Fixed to use each quantized module's
    `.weight()` accessor instead, discovered by comparing this function's
    output against a manual inspection of the converted model.
    """
    total = sum(p.numel() for p in model.parameters())
    quantized = 0
    for module in model.modules():
        if isinstance(module, quantized_module_types):
            if hasattr(module, "weight") and callable(getattr(module, "weight")):
                quantized += module.weight().numel()
                bias_accessor = getattr(module, "bias", None)
                if callable(bias_accessor) and bias_accessor() is not None:
                    quantized += bias_accessor().numel()
            else:
                quantized += sum(p.numel() for p in module.parameters())
    # Total must also count quantized-layer params, which live outside the
    # normal .parameters() iterator and are otherwise invisible to `total`.
    total += quantized
    return {"total_params": total, "quantized_layer_params": quantized,
            "fraction_quantized": quantized / total if total else 0.0}


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization. Only nn.Linear is affected for this
    architecture -- see module docstring. Returns a new model.
    """
    model = copy.deepcopy(model).eval()
    quantized = tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return quantized


def _fuse_resnet_modules(model: nn.Module) -> nn.Module:
    """Fuse Conv+BN(+ReLU) triples ahead of static quantization, as PyTorch's
    static quantization workflow expects/benefits from. Fuses the CIFAR stem
    and every BasicBlock's conv1+bn1(+relu) and conv2+bn2.
    """
    to_fuse = [["conv1", "bn1"]] if not hasattr(model, "relu") else [["conv1", "bn1", "relu"]]
    model = tq.fuse_modules(model, to_fuse, inplace=False)

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for block in layer:
            tq.fuse_modules(block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
            if block.downsample is not None:
                tq.fuse_modules(block.downsample, [["0", "1"]], inplace=True)
    return model


class _QuantWrappedResNet(nn.Module):
    """Wraps a ResNet with QuantStub/DeQuantStub at the boundaries, required
    by PyTorch's eager-mode static quantization workflow.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


@torch.no_grad()
def apply_static_quantization(
    model: nn.Module,
    calibration_loader: DataLoader,
    num_calibration_batches: int = 20,
    backend: str = "x86",
) -> nn.Module:
    """Static post-training INT8 quantization: fuse -> insert stubs ->
    calibrate on real (or representative) data -> convert. Returns a new
    quantized model; does not mutate the input.
    """
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError(
            f"Quantization backend '{backend}' not available on this build "
            f"(available: {torch.backends.quantized.supported_engines})."
        )
    torch.backends.quantized.engine = backend

    model = copy.deepcopy(model).eval()
    model = _make_quantizable(model)
    fused = _fuse_resnet_modules(model)
    wrapped = _QuantWrappedResNet(fused)
    wrapped.qconfig = tq.get_default_qconfig(backend)

    prepared = tq.prepare(wrapped, inplace=False)

    # Calibration: run representative data through the model so the
    # observers inserted by `prepare` can estimate activation ranges.
    n_batches = 0
    for images, _ in calibration_loader:
        prepared(images)
        n_batches += 1
        if n_batches >= num_calibration_batches:
            break
    if n_batches == 0:
        raise ValueError("Calibration loader produced zero batches -- cannot calibrate activation ranges.")

    converted = tq.convert(prepared, inplace=False)
    return converted
