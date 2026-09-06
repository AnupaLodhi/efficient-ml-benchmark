import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.evaluation.metrics import (
    compression_ratio,
    evaluate_classification,
    full_evaluation_report,
    get_model_size_mb,
    measure_latency,
    measure_memory,
)
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


def _tiny_loader(n=32, batch_size=8, seed=0):
    return DataLoader(_TinyDataset(n=n, seed=seed), batch_size=batch_size)


def test_evaluate_classification_returns_expected_fields():
    model = build_resnet18_cifar()
    metrics = evaluate_classification(model, _tiny_loader())
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision_macro <= 1.0
    assert 0.0 <= metrics.recall_macro <= 1.0
    assert 0.0 <= metrics.f1_macro <= 1.0
    assert metrics.n_samples == 32


def test_model_size_is_positive_and_reasonable():
    model = build_resnet18_cifar()
    size_mb = get_model_size_mb(model)
    # ~11.17M FP32 params -> roughly 42-45 MB state_dict.
    assert 30 < size_mb < 60


def test_latency_measurement_shapes():
    model = build_resnet18_cifar()
    latency = measure_latency(model, num_warmup=2, num_runs=5)
    assert latency.num_runs == 5
    assert latency.mean_ms > 0
    assert latency.throughput_samples_per_sec > 0


def test_memory_measurement_nonnegative():
    model = build_resnet18_cifar()
    mem = measure_memory(model, num_runs=5)
    assert mem.peak_rss_delta_mb >= 0.0


def test_compression_ratio_basic():
    assert compression_ratio(100.0, 50.0) == 2.0
    assert compression_ratio(100.0, 100.0) == 1.0


def test_compression_ratio_rejects_zero_size():
    import pytest
    with pytest.raises(ValueError):
        compression_ratio(100.0, 0.0)


def test_full_evaluation_report_has_all_expected_keys():
    model = build_resnet18_cifar()
    report = full_evaluation_report(model, _tiny_loader(), num_warmup=2, num_latency_runs=5)
    expected_prefixes = ["classification_", "model_size_mb", "total_params", "nonzero_params", "latency_", "memory_"]
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in report), f"missing {prefix} in report"
