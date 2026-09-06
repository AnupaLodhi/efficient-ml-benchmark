import pytest
import torch

from src.compression.pruning import apply_structured_pruning, apply_unstructured_pruning, get_sparsity_report
from src.evaluation.metrics import get_model_size_mb
from src.models.resnet import build_resnet18_cifar


def test_unstructured_pruning_achieves_target_sparsity():
    model = build_resnet18_cifar()
    apply_unstructured_pruning(model, 0.5)
    report = get_sparsity_report(model)
    assert abs(report["overall_sparsity"] - 0.5) < 0.01


def test_unstructured_pruning_zero_amount_is_noop():
    model = build_resnet18_cifar()
    report_before = get_sparsity_report(model)
    apply_unstructured_pruning(model, 0.0)
    report_after = get_sparsity_report(model)
    assert report_before["overall_sparsity"] == report_after["overall_sparsity"]


def test_unstructured_pruning_rejects_invalid_amount():
    model = build_resnet18_cifar()
    with pytest.raises(ValueError):
        apply_unstructured_pruning(model, 1.5)


def test_structured_pruning_zeroes_whole_channels():
    model = build_resnet18_cifar()
    apply_structured_pruning(model, 0.5, dim=0)
    # Check conv1: some output channels should be entirely zero.
    w = model.conv1.weight.data  # shape (out_channels, in_channels, kh, kw)
    per_channel_zero = (w.abs().sum(dim=(1, 2, 3)) == 0)
    assert per_channel_zero.sum().item() > 0


def test_pruning_does_not_change_model_size_on_disk():
    """Core honesty check: masking-based pruning (this project's method)
    must NOT be reported as reducing model size, because it genuinely
    doesn't -- the tensor is still stored densely."""
    model = build_resnet18_cifar()
    size_before = get_model_size_mb(model)
    apply_unstructured_pruning(model, 0.8)
    size_after = get_model_size_mb(model)
    assert abs(size_before - size_after) < 0.5, (
        "model size changed after masking-based pruning -- this would be a "
        "surprising result worth investigating, since dense tensor storage "
        "should be unaffected by zeroing values"
    )


def test_pruning_output_shape_unchanged():
    """Masking never changes tensor shapes -- forward pass must still work
    with the same input/output shapes as before pruning."""
    model = build_resnet18_cifar()
    apply_unstructured_pruning(model, 0.6)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
    assert y.shape == (2, 10)


def test_high_sparsity_across_full_config_sweep():
    for sparsity in [0.2, 0.4, 0.6, 0.8]:
        model = build_resnet18_cifar()
        apply_unstructured_pruning(model, sparsity)
        report = get_sparsity_report(model)
        assert abs(report["overall_sparsity"] - sparsity) < 0.01, f"failed at sparsity={sparsity}"
