import numpy as np
import pytest
import torch
import torch.ao.nn.quantized.dynamic as nnqd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.compression.quantization import (
    apply_dynamic_quantization,
    apply_static_quantization,
    count_actually_quantized_params,
    to_fp16,
)
from src.evaluation.metrics import get_model_size_mb
from src.models.resnet import build_resnet18_cifar, count_parameters


class _TinyDataset(Dataset):
    def __init__(self, n=32, seed=0):
        rng = np.random.default_rng(seed)
        self.images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
        self.labels = rng.integers(0, 10, size=(n,)).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tf = transforms.Compose([transforms.ToTensor()])
        return tf(Image.fromarray(self.images[idx])), self.labels[idx]


def _tiny_loader(n=32, batch_size=8):
    return DataLoader(_TinyDataset(n=n), batch_size=batch_size)


def test_fp16_forward_pass_and_size():
    model = build_resnet18_cifar()
    size_fp32 = get_model_size_mb(model)
    m16 = to_fp16(model)
    size_fp16 = get_model_size_mb(m16)

    x = torch.randn(2, 3, 32, 32).half()
    y = m16(x)
    assert y.shape == (2, 10)
    assert y.dtype == torch.float16
    # FP16 halves storage relative to FP32 -- assert the real ratio, not an
    # assumed one.
    ratio = size_fp32 / size_fp16
    assert 1.9 < ratio < 2.1


def test_fp16_does_not_mutate_original_model():
    model = build_resnet18_cifar()
    to_fp16(model)
    assert next(model.parameters()).dtype == torch.float32


def test_dynamic_quantization_forward_pass():
    model = build_resnet18_cifar()
    mdyn = apply_dynamic_quantization(model)
    x = torch.randn(2, 3, 32, 32)
    y = mdyn(x)
    assert y.shape == (2, 10)


def test_dynamic_quantization_only_affects_fc_layer():
    """Core honesty check from the module docstring: eager-mode dynamic
    quantization on a Conv-dominated CNN only quantizes nn.Linear (here,
    just the final `fc`), not the convolutional backbone -- this must be
    reflected accurately, not overstated as full-model compression."""
    model = build_resnet18_cifar()
    total_params = count_parameters(model)["total_params"]
    mdyn = apply_dynamic_quantization(model)
    report = count_actually_quantized_params(mdyn, (nnqd.Linear,))

    assert report["total_params"] == total_params
    # fc layer only: (512 * 10 weights) + (10 bias) = 5130 params.
    assert report["quantized_layer_params"] == 5130
    assert report["fraction_quantized"] < 0.001, (
        "dynamic quantization is quantizing more than just the fc layer -- "
        "investigate before trusting a compression-ratio claim for this method"
    )


def test_static_quantization_forward_pass_and_real_size_reduction():
    """Unlike pruning (masking only), static INT8 quantization should show
    a REAL measured size reduction -- this is the mechanism the project's
    pruning module explicitly does NOT have, and this test is the guardrail
    that the two aren't confused in results.csv."""
    model = build_resnet18_cifar()
    model.eval()
    size_fp32 = get_model_size_mb(model)

    qmodel = apply_static_quantization(model, _tiny_loader(n=16), num_calibration_batches=2)
    size_int8 = get_model_size_mb(qmodel)

    x = torch.randn(2, 3, 32, 32)
    y = qmodel(x)
    assert y.shape == (2, 10)

    assert size_int8 < size_fp32 * 0.5, (
        f"expected static INT8 ({size_int8:.2f} MB) to be well under half of "
        f"FP32 ({size_fp32:.2f} MB) -- if this fails, static quantization "
        "silently stopped producing a real compression benefit and results "
        "would need re-checking before trusting the compression_ratio column"
    )


def test_static_quantization_does_not_mutate_original_model():
    model = build_resnet18_cifar()
    model.eval()
    original_dtype = next(model.parameters()).dtype
    apply_static_quantization(model, _tiny_loader(n=16), num_calibration_batches=2)
    assert next(model.parameters()).dtype == original_dtype


def test_static_quantization_rejects_unsupported_backend():
    model = build_resnet18_cifar()
    model.eval()
    with pytest.raises(RuntimeError):
        apply_static_quantization(model, _tiny_loader(n=8), backend="not_a_real_backend")
