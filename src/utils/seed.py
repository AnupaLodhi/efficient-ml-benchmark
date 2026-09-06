"""Deterministic seeding for reproducibility across Python, NumPy and PyTorch."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all relevant RNGs and (optionally) force deterministic algorithms.

    Determinism on CPU is generally achievable; on GPU some ops (e.g. certain
    convolution algorithms) are non-deterministic even with these flags set,
    which is why we still record hardware info and report latency as
    mean +/- std over repeated runs rather than a single number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # older torch versions don't support warn_only
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Seed function for torch.utils.data.DataLoader(worker_init_fn=...).

    Ensures DataLoader worker processes (when num_workers > 0) don't all
    share the same NumPy/random state, while still being reproducible given
    a fixed base seed via the generator passed to the DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
