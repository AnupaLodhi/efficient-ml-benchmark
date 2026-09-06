"""Evaluation metrics shared by every experiment phase.

Everything here is measured the same way regardless of which compression
technique produced the model, so results are comparable across the whole
project (this is the "single reproducible benchmarking framework" the
project spec asks for). Nothing in this module trains anything -- it only
measures a model that's already built.
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.models.resnet import count_nonzero_parameters, count_parameters


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    n_samples: int


@dataclass
class LatencyMetrics:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    num_runs: int
    batch_size: int
    throughput_samples_per_sec: float


@dataclass
class MemoryMetrics:
    peak_rss_delta_mb: float
    note: str = (
        "Process-level peak resident-set-size delta during inference, measured "
        "with psutil. This is a coarse proxy for model memory footprint, not a "
        "precise activation-memory profile."
    )


@torch.no_grad()
def evaluate_classification(model: nn.Module, dataloader: DataLoader, device: str = "cpu") -> ClassificationMetrics:
    """Run the model over `dataloader` and compute accuracy/precision/recall/F1.

    Uses macro averaging across the 10 CIFAR-10 classes (unweighted mean per
    class), appropriate since CIFAR-10 is class-balanced.
    """
    model.eval()
    model.to(device)

    all_preds, all_targets = [], []
    for images, targets in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(np.asarray(targets))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return ClassificationMetrics(
        accuracy=float(acc),
        precision_macro=float(precision),
        recall_macro=float(recall),
        f1_macro=float(f1),
        n_samples=len(y_true),
    )


def get_model_size_mb(model: nn.Module) -> float:
    """On-disk size of `model.state_dict()` when serialized, in MB.

    This is the size that actually matters for deployment/download -- more
    meaningful than a raw parameter-count estimate, since it reflects the
    real dtype of each stored tensor (relevant once quantized models with
    int8 weights are compared to the FP32 baseline in Phase 4+).
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(model.state_dict(), tmp_path)
        size_bytes = os.path.getsize(tmp_path)
    finally:
        os.remove(tmp_path)
    return size_bytes / (1024 * 1024)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_shape: tuple = (1, 3, 32, 32),
    num_warmup: int = 10,
    num_runs: int = 50,
    device: str = "cpu",
) -> LatencyMetrics:
    """Measure per-batch CPU inference latency, repeated `num_runs` times.

    Reports mean +/- std rather than a single number, per the project's
    statistical-rigor requirement. Warmup runs are discarded (first calls
    pay for lazy kernel init / cache effects and would bias the mean).
    """
    model.eval()
    model.to(device)
    x = torch.randn(*input_shape, device=device)

    for _ in range(num_warmup):
        model(x)

    times_ms = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model(x)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    times_ms = np.array(times_ms)
    batch_size = input_shape[0]
    mean_ms = float(times_ms.mean())
    throughput = batch_size / (mean_ms / 1000.0) if mean_ms > 0 else float("nan")

    return LatencyMetrics(
        mean_ms=mean_ms,
        std_ms=float(times_ms.std()),
        min_ms=float(times_ms.min()),
        max_ms=float(times_ms.max()),
        num_runs=num_runs,
        batch_size=batch_size,
        throughput_samples_per_sec=throughput,
    )


@torch.no_grad()
def measure_memory(
    model: nn.Module,
    input_shape: tuple = (1, 3, 32, 32),
    num_runs: int = 20,
    device: str = "cpu",
) -> MemoryMetrics:
    """Measure process RSS delta while running repeated inference.

    Requires `psutil`. This is intentionally simple (process-level, not a
    per-layer activation profiler) -- appropriate for the "memory usage
    where measurable" requirement without overstating precision.
    """
    import psutil

    process = psutil.Process(os.getpid())
    model.eval()
    model.to(device)
    x = torch.randn(*input_shape, device=device)

    # A couple of warmup calls so lazy allocations don't get counted as
    # "inference memory".
    for _ in range(3):
        model(x)

    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    for _ in range(num_runs):
        model(x)
        current_rss = process.memory_info().rss
        peak_rss = max(peak_rss, current_rss)

    delta_mb = (peak_rss - baseline_rss) / (1024 * 1024)
    return MemoryMetrics(peak_rss_delta_mb=max(delta_mb, 0.0))


def compression_ratio(baseline_size_mb: float, compressed_size_mb: float) -> float:
    """baseline_size / compressed_size. >1 means the compressed model is smaller.

    Returns 1.0 (not inflated) if sizes are equal, and raises on a
    zero-size compressed model rather than silently returning inf.
    """
    if compressed_size_mb <= 0:
        raise ValueError(f"compressed_size_mb must be > 0, got {compressed_size_mb}")
    return baseline_size_mb / compressed_size_mb


def full_evaluation_report(
    model: nn.Module,
    dataloader: DataLoader,
    latency_input_shape: tuple = (1, 3, 32, 32),
    num_warmup: int = 10,
    num_latency_runs: int = 50,
    device: str = "cpu",
) -> dict:
    """Run the full standard evaluation suite and return a flat dict, ready
    to become one row of results/raw/*.csv. This is the single function
    every experiment script (baseline, pruning, quantization, combined)
    should call, so every model is scored identically.
    """
    cls_metrics = evaluate_classification(model, dataloader, device=device)
    size_mb = get_model_size_mb(model)
    param_counts = count_parameters(model)
    nonzero_counts = count_nonzero_parameters(model)
    latency = measure_latency(
        model, input_shape=latency_input_shape, num_warmup=num_warmup, num_runs=num_latency_runs, device=device
    )
    memory = measure_memory(model, input_shape=latency_input_shape, device=device)

    report = {}
    report.update({f"classification_{k}": v for k, v in asdict(cls_metrics).items()})
    report["model_size_mb"] = size_mb
    report["total_params"] = param_counts["total_params"]
    report["nonzero_params"] = nonzero_counts["nonzero_params"]
    report.update({f"latency_{k}": v for k, v in asdict(latency).items()})
    report.update({f"memory_{k}": v for k, v in asdict(memory).items() if k != "note"})
    return report
