"""CIFAR-10 loading, preprocessing, and a reproducible train/val/test split.

Design notes
------------
* torchvision's official CIFAR-10 source (www.cs.toronto.edu) is used for
  `download=True`. On networks that block that host, download the archive
  manually and place it at `<data_dir>/cifar-10-python.tar.gz` before
  running -- torchvision will detect and extract it without re-downloading.
* The official CIFAR-10 test set (10,000 images) is reserved exclusively for
  final reporting. Hyperparameter/checkpoint selection uses a validation
  split carved out of the *training* set, so the test set is never touched
  until final evaluation. This split uses its own fixed seed
  (`data.split_seed`), independent of the training seed, so the split is
  stable even if you sweep `training` seeds for statistical repeats.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_transforms(cfg) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform).

    Train transform includes standard CIFAR-10 augmentation (random crop with
    padding + horizontal flip); eval transform is deterministic
    (no augmentation) so validation/test numbers are directly comparable
    across experiments.
    """
    mean = cfg.data.normalize_mean
    std = cfg.data.normalize_std
    aug = cfg.data.augmentation

    train_ops = []
    if aug.random_crop_padding:
        train_ops.append(transforms.RandomCrop(32, padding=aug.random_crop_padding))
    if aug.random_horizontal_flip:
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    eval_ops = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def load_datasets(cfg, download: bool = True):
    """Load CIFAR-10 train/test sets and split train into train/val.

    Returns:
        train_set, val_set, test_set (all torch.utils.data.Dataset),
        plus the class names list.

    Raises a clear, actionable error (not a bare traceback) if the dataset
    cannot be downloaded, since sandboxed/offline environments commonly
    block the upstream host.
    """
    data_dir = cfg.paths.data_dir
    train_tf, eval_tf = build_transforms(cfg)

    try:
        full_train_aug = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_tf)
        full_train_eval = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=eval_tf)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=eval_tf)
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable context
        raise RuntimeError(
            "Failed to obtain CIFAR-10. If this environment cannot reach "
            "https://www.cs.toronto.edu, download 'cifar-10-python.tar.gz' "
            f"on a machine with internet access and place it at "
            f"'{data_dir}/cifar-10-python.tar.gz', then re-run with the same "
            "data_dir (torchvision will extract the local archive)."
        ) from exc

    n_total = len(full_train_aug)
    n_val = int(round(n_total * cfg.data.val_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.data.split_seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    # Use the augmented-transform dataset for train indices and the
    # eval-transform dataset for val indices, so validation is never
    # evaluated under random augmentation.
    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    return train_set, val_set, test_set, CIFAR10_CLASSES


def make_dataloaders(cfg, download: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    from .split_utils import make_seeded_generator  # local import avoids cycle risk

    train_set, val_set, test_set, classes = load_datasets(cfg, download=download)

    g = make_seeded_generator(cfg.project.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    return train_loader, val_loader, test_loader, classes
