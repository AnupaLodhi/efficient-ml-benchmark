import torch

from src.utils.env_info import collect_environment_info
from src.utils.seed import set_seed


def test_collect_environment_info_has_required_fields():
    info = collect_environment_info()
    for key in ("python_version", "torch_version", "cpu", "memory", "gpu", "timestamp_utc"):
        assert key in info
    assert "cuda_available" in info["gpu"]


def test_set_seed_gives_reproducible_random_tensor():
    set_seed(123, deterministic=True)
    a = torch.rand(5)
    set_seed(123, deterministic=True)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_different_seeds_give_different_tensors():
    set_seed(1)
    a = torch.rand(5)
    set_seed(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)
