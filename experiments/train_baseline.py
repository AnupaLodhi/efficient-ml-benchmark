"""Phase 2 entry point: train the FP32 baseline ResNet-18 on CIFAR-10 and
run the full evaluation suite.

Usage:
    python -m experiments.train_baseline
    python -m experiments.train_baseline --epochs 5 --config configs/config.yaml
    python -m experiments.train_baseline --smoke-test   # tiny run, see below

IMPORTANT -- read before running:
This script needs real CIFAR-10 (downloaded by torchvision, or a local
cifar-10-python.tar.gz -- see README "Environment & Data Availability") and
enough compute to make ~100 epochs of ResNet-18 training practical (a few
CPU cores or a GPU). If you just want to confirm the training loop itself
is wired correctly, use --smoke-test, which trains 1 epoch on a tiny
synthetic dataset (no download, seconds to run) and does NOT produce a
result meant to go in results.csv -- it's a pipeline check, not an
experiment.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.benchmarking.results_store import append_result_row
from src.evaluation.metrics import full_evaluation_report, get_model_size_mb
from src.models.resnet import build_resnet18_cifar, count_parameters
from src.training.train import train_model
from src.utils.config import load_config, save_config
from src.utils.env_info import collect_environment_info
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run_smoke_test(cfg) -> None:
    """1-epoch training run on a tiny synthetic dataset with the real
    training loop, real model, and real evaluation code -- to prove the
    Phase 2 pipeline is wired correctly without needing CIFAR-10 access or
    real compute budget. Prints results but does NOT write to results.csv,
    since these numbers are meaningless (random data) and must never be
    mistaken for a real experiment.
    """
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class _TinyFakeCIFAR(Dataset):
        def __init__(self, n, seed):
            rng = np.random.default_rng(seed)
            self.images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
            self.labels = rng.integers(0, 10, size=(n,)).tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            from torchvision import transforms

            img = Image.fromarray(self.images[idx])
            tf = transforms.Compose([transforms.ToTensor()])
            return tf(img), self.labels[idx]

    logger.info("SMOKE TEST: training 1 epoch on synthetic random data (not real CIFAR-10).")
    train_ds = _TinyFakeCIFAR(64, seed=0)
    val_ds = _TinyFakeCIFAR(32, seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = build_resnet18_cifar(num_classes=cfg.data.num_classes, cifar_stem=cfg.model.cifar_stem)
    history = train_model(model, train_loader, val_loader, cfg, epochs=1, device="cpu")

    logger.info(f"SMOKE TEST complete. 1-epoch train_loss={history.train_loss[0]:.4f} "
                f"(on random labels -- expected val_acc ~ chance level, ~0.1 for 10 classes). "
                "Pipeline runs end-to-end. Real training requires real CIFAR-10 data.")


def run_baseline_training(cfg, epochs: int | None, seed: int) -> None:
    from src.data.cifar10 import make_dataloaders

    set_seed(seed, deterministic=cfg.project.deterministic)
    logger.info("Loading CIFAR-10 (requires network access or a local archive -- see README) ...")
    train_loader, val_loader, test_loader, classes = make_dataloaders(cfg, download=True)
    logger.info(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = build_resnet18_cifar(num_classes=cfg.data.num_classes, cifar_stem=cfg.model.cifar_stem)
    logger.info(f"Model: ResNet-18 (CIFAR stem={cfg.model.cifar_stem}), {count_parameters(model)['total_params']:,} params")

    checkpoint_path = Path(cfg.paths.models_dir) / "baseline" / "resnet18_fp32_best.pt"
    history = train_model(
        model, train_loader, val_loader, cfg,
        epochs=epochs, checkpoint_path=checkpoint_path, device=cfg.project.device,
    )

    logger.info(f"Training complete in {history.total_train_time_sec:.1f}s. "
                f"Best val accuracy: {history.best_val_accuracy:.4f} @ epoch {history.best_epoch+1}")

    # Reload best checkpoint before final test-set evaluation (not the
    # possibly-overfit final-epoch weights).
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    logger.info("Running full evaluation suite on the held-out TEST set ...")
    report = full_evaluation_report(
        model, test_loader,
        num_warmup=cfg.evaluation.latency_num_warmup,
        num_latency_runs=cfg.evaluation.latency_num_runs,
        device="cpu",  # test-set latency always measured on CPU per RQ4
    )

    run_id = f"baseline_fp32_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "resnet18",
        "technique": "baseline",
        "sparsity": 0.0,
        "pruning_method": None,
        "quantization_method": "fp32",
        "device": "cpu",
        "compression_ratio": 1.0,
        "training_time_sec": history.total_train_time_sec,
        "notes": f"best_epoch={history.best_epoch+1}, stopped_early={history.stopped_early}",
        **report,
    }
    results_csv = Path(cfg.paths.results_raw) / "results.csv"
    append_result_row(row, results_csv)
    logger.info(f"Result row appended -> {results_csv}")

    # Save the resolved config + environment snapshot alongside this run,
    # so every number can be traced back to exactly what produced it.
    run_dir = Path(cfg.paths.results_raw) / run_id
    save_config(cfg, run_dir / "config.yaml")
    with open(run_dir / "environment.json", "w") as f:
        json.dump(collect_environment_info(), f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(vars(history), f, indent=2)
    logger.info(f"Run artifacts saved -> {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the FP32 baseline ResNet-18.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override configs/config.yaml training.epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override configs/config.yaml project.seed")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run a 1-epoch pipeline check on synthetic data instead of real training.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.project.seed = args.seed

    if args.smoke_test:
        set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
        run_smoke_test(cfg)
        return

    run_baseline_training(cfg, epochs=args.epochs, seed=cfg.project.seed)


if __name__ == "__main__":
    main()
