"""Small helpers shared by data-loading code."""
from __future__ import annotations

import torch


def make_seeded_generator(seed: int) -> torch.Generator:
    """A torch.Generator seeded for reproducible DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
