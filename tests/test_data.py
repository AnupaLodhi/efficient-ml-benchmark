"""Tests for the data pipeline.

These tests do NOT download CIFAR-10 -- they exercise the transform
pipeline and the reproducible-split logic against a small synthetic
dataset with the same shape/interface as torchvision.datasets.CIFAR10.
This lets the pipeline logic be verified in network-restricted environments
(including this sandbox); a separate integration check
(scripts noted in the README) should be run once against the real dataset
on a machine with internet access before trusting Phase 2+ results.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.cifar10 import build_transforms
from src.data.split_utils import make_seeded_generator
from src.utils.config import load_config


class _FakeCIFAR10(Dataset):
    """Minimal stand-in for torchvision.datasets.CIFAR10: fixed-size 32x32
    RGB images with integer labels in [0, 10), no network access required.
    """

    def __init__(self, n: int = 200, transform=None, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
        self.labels = rng.integers(0, 10, size=(n,)).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def test_transforms_produce_correct_tensor_shape():
    cfg = load_config()
    train_tf, eval_tf = build_transforms(cfg)
    ds = _FakeCIFAR10(n=10, transform=train_tf)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)


def test_eval_transform_is_deterministic():
    cfg = load_config()
    _, eval_tf = build_transforms(cfg)
    ds = _FakeCIFAR10(n=5, transform=eval_tf)
    img_a, _ = ds[0]
    img_b, _ = ds[0]
    assert torch.allclose(img_a, img_b)


def test_split_is_reproducible_given_same_seed():
    n_total = 1000
    generator1 = torch.Generator().manual_seed(42)
    perm1 = torch.randperm(n_total, generator=generator1).tolist()

    generator2 = torch.Generator().manual_seed(42)
    perm2 = torch.randperm(n_total, generator=generator2).tolist()

    assert perm1 == perm2


def test_split_differs_with_different_seed():
    n_total = 1000
    g1 = torch.Generator().manual_seed(42)
    perm1 = torch.randperm(n_total, generator=g1).tolist()

    g2 = torch.Generator().manual_seed(43)
    perm2 = torch.randperm(n_total, generator=g2).tolist()

    assert perm1 != perm2


def test_seeded_dataloader_generator_reproducible():
    g1 = make_seeded_generator(123)
    g2 = make_seeded_generator(123)
    t1 = torch.randperm(50, generator=g1)
    t2 = torch.randperm(50, generator=g2)
    assert torch.equal(t1, t2)
