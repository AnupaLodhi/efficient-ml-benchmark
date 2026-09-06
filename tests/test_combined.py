import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.compression.pruning import apply_unstructured_pruning, get_sparsity_report
from src.compression.quantization import apply_dynamic_quantization, apply_static_quantization
from src.models.resnet import build_resnet18_cifar


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


def test_pruned_then_dynamic_quantized_forward_pass():
    model = build_resnet18_cifar()
    apply_unstructured_pruning(model, 0.5)
    model.eval()
    combined = apply_dynamic_quantization(model)
    x = torch.randn(2, 3, 32, 32)
    y = combined(x)
    assert y.shape == (2, 10)


def test_pruned_then_static_quantized_forward_pass():
    model = build_resnet18_cifar()
    apply_unstructured_pruning(model, 0.5)
    model.eval()
    combined = apply_static_quantization(model, _tiny_loader(n=16), num_calibration_batches=2)
    x = torch.randn(2, 3, 32, 32)
    y = combined(x)
    assert y.shape == (2, 10)


def test_pruning_survives_before_quantization():
    """The sparsity achieved by pruning should still be measurable in the
    pruned-but-not-yet-quantized model right before it's handed to the
    quantization step (sanity check on pipeline ordering)."""
    model = build_resnet18_cifar()
    apply_unstructured_pruning(model, 0.6)
    report = get_sparsity_report(model)
    assert abs(report["overall_sparsity"] - 0.6) < 0.01
