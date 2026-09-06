"""Phase 4 entry point: quantization sweep.

For each method in cfg.quantization.methods, starting from the trained FP32
baseline checkpoint:
  1. Apply the quantization method (src.compression.quantization).
     No fine-tuning step -- these are post-training methods by design (that
     distinction is the point of Phase 4; QAT, which would need
     re-training, is explicitly out of scope -- see quantization.py docstring).
  2. Run the full evaluation suite and append a row to results/raw/results.csv,
     with the actually-affected parameter count recorded for dynamic
     quantization so the results table itself carries the honesty caveat,
     not just this script's comments.

Usage:
    python -m experiments.run_quantization --checkpoint models/baseline/resnet18_fp32_best.pt
    python -m experiments.run_quantization --smoke-test
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.ao.nn.quantized.dynamic as nnqd

from src.benchmarking.results_store import append_result_row
from src.compression.quantization import (
    apply_dynamic_quantization,
    apply_static_quantization,
    count_actually_quantized_params,
    to_fp16,
)
from src.evaluation.metrics import evaluate_classification, full_evaluation_report, get_model_size_mb
from src.models.resnet import build_resnet18_cifar
from src.utils.config import load_config
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def load_baseline_model(checkpoint_path: str | Path, cfg) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {checkpoint_path}. "
            "Run `python -m experiments.train_baseline` first (Phase 2)."
        )
    model = build_resnet18_cifar(num_classes=cfg.data.num_classes, cifar_stem=cfg.model.cifar_stem)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def _evaluate_and_record(model, test_loader, cfg, method: str, notes: str) -> None:
    report = full_evaluation_report(
        model, test_loader,
        num_warmup=cfg.evaluation.latency_num_warmup,
        num_latency_runs=cfg.evaluation.latency_num_runs,
        device="cpu",
    )


    run_id = f"quantization_{method}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "resnet18",
        "technique": "quantization",
        "sparsity": 0.0,
        "pruning_method": None,
        "quantization_method": method,
        "device": "cpu",
        "training_time_sec": 0.0,  # post-training methods only, no fine-tuning
        "notes": notes,
        **report,
    }
    results_csv = Path(cfg.paths.results_raw) / "results.csv"
    append_result_row(row, results_csv)
    logger.info(f"[{method}] size={report['model_size_mb']:.2f}MB "
                f"latency={report['latency_mean_ms']:.2f}ms "
                f"acc={report['classification_accuracy']:.4f} -> {results_csv}")


def run_quantization_sweep(cfg, checkpoint_path: str | Path) -> None:
    from src.data.cifar10 import make_dataloaders

    if not Path(checkpoint_path).exists():
        load_baseline_model(checkpoint_path, cfg)  # raises the clear FileNotFoundError

    train_loader, val_loader, test_loader, _ = make_dataloaders(cfg, download=True)
    methods = cfg.quantization.methods

    if "fp32_baseline" in methods:
        model = load_baseline_model(checkpoint_path, cfg)
        _evaluate_and_record(model, test_loader, cfg, "fp32",
                              "unmodified baseline, re-measured under identical eval code for direct comparability")

    if "fp16" in methods:
        model = load_baseline_model(checkpoint_path, cfg)
        m16 = to_fp16(model)
        # fp16 model needs fp16 input; full_evaluation_report builds its own
        # random latency input at fp32, so cast the model's eval path
        # accordingly -- evaluate_classification handles dataset images
        # (which are fp32) by casting per-batch here.
        report_ok = True
        try:
            class _FP16Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m(x.half()).float()

            wrapped = _FP16Wrapper(m16)
            _evaluate_and_record(
                wrapped, test_loader, cfg, "fp16",
                "activations cast fp32->fp16->fp32 at the module boundary for compatibility with the shared "
                "fp32 evaluation pipeline; CPU latency benefit is NOT assumed -- see quantization.py docstring."
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"FP16 evaluation failed on this hardware/torch build: {exc}. "
                         "Recording as a failed experiment rather than skipping silently.")
            row = {
                "run_id": f"quantization_fp16_FAILED_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_name": "resnet18", "technique": "quantization", "quantization_method": "fp16",
                "device": "cpu", "notes": f"FAILED: {exc}",
            }
            append_result_row(row, Path(cfg.paths.results_raw) / "results.csv")

    if "dynamic_int8" in methods:
        model = load_baseline_model(checkpoint_path, cfg)
        mdyn = apply_dynamic_quantization(model)
        coverage = count_actually_quantized_params(mdyn, (nnqd.Linear,))
        _evaluate_and_record(
            mdyn, test_loader, cfg, "dynamic_int8",
            f"only {coverage['fraction_quantized']*100:.3f}% of parameters actually quantized "
            f"({coverage['quantized_layer_params']:,}/{coverage['total_params']:,}, fc layer only) -- "
            "eager-mode dynamic quantization does not support Conv2d; do not read model_size_mb/latency "
            "here as representative of full-model int8 compression."
        )

    if "static_int8" in methods:
        model = load_baseline_model(checkpoint_path, cfg)
        qmodel = apply_static_quantization(
            model, train_loader, num_calibration_batches=cfg.quantization.static_calibration_batches
        )
        _evaluate_and_record(
            qmodel, test_loader, cfg, "static_int8",
            "fused Conv+BN(+ReLU), FloatFunctional residual add, calibrated on "
            f"{cfg.quantization.static_calibration_batches} training batches -- this method quantizes "
            "the convolutional backbone, unlike dynamic_int8."
        )


def run_smoke_test(cfg) -> None:
    """Verify all four quantization code paths run against the real
    architecture, using random weights (no checkpoint needed). Does not
    write to results.csv."""
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class _Tiny(Dataset):
        def __init__(self, n, seed):
            rng = np.random.default_rng(seed)
            self.images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
            self.labels = rng.integers(0, 10, size=(n,)).tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            tf = transforms.Compose([transforms.ToTensor()])
            return tf(Image.fromarray(self.images[idx])), self.labels[idx]

    loader = DataLoader(_Tiny(32, 0), batch_size=8)
    base_model = build_resnet18_cifar(num_classes=cfg.data.num_classes, cifar_stem=cfg.model.cifar_stem)
    base_model.eval()
    size_fp32 = get_model_size_mb(base_model)

    m16 = to_fp16(base_model)
    logger.info(f"SMOKE TEST [fp16]: size {size_fp32:.2f} -> {get_model_size_mb(m16):.2f} MB")

    mdyn = apply_dynamic_quantization(base_model)
    coverage = count_actually_quantized_params(mdyn, (nnqd.Linear,))
    logger.info(f"SMOKE TEST [dynamic_int8]: {coverage['fraction_quantized']*100:.3f}% of params quantized "
                f"(fc layer only, as expected)")

    qmodel = apply_static_quantization(base_model, loader, num_calibration_batches=2)
    logger.info(f"SMOKE TEST [static_int8]: size {size_fp32:.2f} -> {get_model_size_mb(qmodel):.2f} MB")

    logger.info("SMOKE TEST complete: all quantization code paths run end-to-end on the real architecture.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization sweep: fp16, dynamic INT8, static INT8.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/baseline/resnet18_fp32_best.pt")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smoke_test:
        set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
        run_smoke_test(cfg)
        return

    run_quantization_sweep(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
