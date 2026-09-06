import torch

from src.models.resnet import build_resnet18_cifar, count_nonzero_parameters, count_parameters


def test_output_shape_matches_num_classes():
    model = build_resnet18_cifar(num_classes=10, cifar_stem=True)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    assert y.shape == (4, 10)


def test_cifar_stem_changes_first_conv():
    cifar_model = build_resnet18_cifar(cifar_stem=True)
    imagenet_model = build_resnet18_cifar(cifar_stem=False)
    assert cifar_model.conv1.kernel_size == (3, 3)
    assert cifar_model.conv1.stride == (1, 1)
    assert imagenet_model.conv1.kernel_size == (7, 7)


def test_param_count_is_reasonable_for_resnet18():
    model = build_resnet18_cifar()
    counts = count_parameters(model)
    # Standard ResNet-18 has ~11.2M parameters regardless of stem/head tweaks.
    assert 11_000_000 < counts["total_params"] < 11_300_000
    assert counts["trainable_params"] == counts["total_params"]


def test_nonzero_close_to_total_before_pruning():
    # NOTE: this is intentionally NOT an exact equality. PyTorch's default
    # nn.BatchNorm2d initializes bias=0, so a handful of parameters
    # (~4.8k out of ~11.17M here) are exactly zero even before any pruning
    # is applied. This was discovered by an earlier, stricter version of
    # this test failing -- recorded here rather than silently loosened,
    # per the project's "record unexpected results honestly" rule.
    model = build_resnet18_cifar()
    counts = count_nonzero_parameters(model)
    zero_at_init = counts["total_params"] - counts["nonzero_params"]
    assert 0 <= zero_at_init < 10_000, (
        f"unexpectedly many zero params at init: {zero_at_init}"
    )


def test_different_batch_sizes_work():
    model = build_resnet18_cifar()
    model.eval()
    with torch.no_grad():
        for batch_size in (1, 2, 8):
            x = torch.randn(batch_size, 3, 32, 32)
            y = model(x)
            assert y.shape == (batch_size, 10)
