"""Phase 3 entry point: pruning sweep.

For each (method, sparsity) in cfg.pruning, starting from the trained FP32
baseline checkpoint:
  1. Apply pruning (src.compression.pruning).
  2. Fine-tune briefly (cfg.pruning.fine_tune_epochs) to recover accuracy.
  3. Run the full evaluation suite (same metrics as the baseline, so results
     are directly comparable) and append a row to results/raw/results.csv.

Usage:
    python -m experiments.run_pruning --checkpoint models/baseline/resnet18_fp32_best.pt
    python -m experiments.run_pruning --smoke-test   # pipeline check, synthetic data

Requires a trained baseline checkpoint from experiments/train_baseline.py.
Will refuse to run against real CIFAR-10 without one, since fine-tuning
pruned *random* weights would produce meaningless numbers -- rather than
silently doing that, this script raises a clear error.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.benchmarking.results_store import append_result_row
from src.compression.pruning import (
    apply_structured_pruning,
    apply_unstructured_pruning,
    finalize_pruning,
    get_sparsity_report,
)
from src.evaluation.metrics import full_evaluation_report
from src.models.resnet import build_resnet18_cifar
from src.training.train import train_model
from src.utils.config import load_config
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

METHOD_FUNCS = {
    "unstructured_l1": apply_unstructured_pruning,
    "structured_ln": apply_structured_pruning,
}


def load_baseline_model(checkpoint_path: str | Path, cfg) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {checkpoint_path}. "
            "Run `python -m experiments.train_baseline` first (Phase 2) -- "
            "pruning a randomly-initialized model and reporting the result "
            "would violate this project's 'never fabricate results' rule."
        )
    model = build_resnet18_cifar(num_classes=cfg.data.num_classes, cifar_stem=cfg.model.cifar_stem)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model


def run_pruning_sweep(cfg, checkpoint_path: str | Path) -> None:
    from src.data.cifar10 import make_dataloaders

    # Fail fast with the more actionable error if the checkpoint is missing,
    # rather than failing on the (also currently blocked) CIFAR-10 download
    # first and masking the real issue.
    if not Path(checkpoint_path).exists():
        load_baseline_model(checkpoint_path, cfg)  # raises the clear FileNotFoundError

    train_loader, val_loader, test_loader, _ = make_dataloaders(cfg, download=True)

    for method in cfg.pruning.methods:
        prune_fn = METHOD_FUNCS[method]
        for sparsity in cfg.pruning.sparsities:
            set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
            logger.info(f"=== Pruning: method={method} sparsity={sparsity} ===")

            model = load_baseline_model(checkpoint_path, cfg)

            if sparsity > 0.0:
                if method == "unstructured_l1":
                    prune_fn(model, sparsity, remove_reparam=False)
                else:
                    prune_fn(model, sparsity)
                sparsity_report = get_sparsity_report(model)
                logger.info(f"Achieved overall prunable-layer sparsity: {sparsity_report['overall_sparsity']:.3f}")

                fine_tune_ckpt = Path(cfg.paths.models_dir) / "pruning" / f"resnet18_{method}_s{sparsity}.pt"
                history = train_model(
                    model, train_loader, val_loader, cfg,
                    epochs=cfg.pruning.fine_tune_epochs,
                    checkpoint_path=fine_tune_ckpt,
                    device=cfg.project.device,
                )
                model.load_state_dict(torch.load(fine_tune_ckpt, map_location="cpu"))
                if method == "unstructured_l1":
                    finalize_pruning(model)
                sparsity_report = get_sparsity_report(model)
                training_time = history.total_train_time_sec
            else:
                sparsity_report = get_sparsity_report(model)
                training_time = 0.0  # sparsity=0.0 is the baseline re-measured under identical eval code

            report = full_evaluation_report(
                model, test_loader,
                num_warmup=cfg.evaluation.latency_num_warmup,
                num_latency_runs=cfg.evaluation.latency_num_runs,
                device="cpu",
            )

            run_id = f"pruning_{method}_s{sparsity}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            row = {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_name": "resnet18",
                "technique": "pruning",
                "sparsity": sparsity,
                "pruning_method": method,
                "quantization_method": "fp32",
                "device": "cpu",
                "training_time_sec": training_time,
                "notes": (
                    f"overall_prunable_sparsity={sparsity_report['overall_sparsity']:.4f}; "
                    "size_mb/latency NOT expected to improve from masking alone -- see "
                    "src/compression/pruning.py module docstring."
                ),
                **report,
            }
            results_csv = Path(cfg.paths.results_raw) / "results.csv"
            append_result_row(row, results_csv)
            logger.info(f"Result appended -> {results_csv}")


def run_smoke_test(cfg) -> None:
    """Verify the pruning + fine-tune + evaluation pipeline runs correctly on
    a synthetic dataset and a randomly-initialized model, WITHOUT writing to
    results.csv. This checks the code path, not model quality."""
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

    train_loader = DataLoader(_Tiny(32, 0), batch_size=8)
    val_loader = DataLoader(_Tiny(16, 1), batch_size=8)

    model = build_resnet18_cifar(num_classes=cfg.data.num_classes, cifar_stem=cfg.model.cifar_stem)

    for method, prune_fn in METHOD_FUNCS.items():
        m = copy.deepcopy(model)
        if method == "unstructured_l1":
            prune_fn(m, 0.4, remove_reparam=False)
        else:
            prune_fn(m, 0.4)
        sparsity_report = get_sparsity_report(m)
        logger.info(f"SMOKE TEST [{method}]: overall sparsity after pruning = {sparsity_report['overall_sparsity']:.3f}")
        history = train_model(m, train_loader, val_loader, cfg, epochs=1, device="cpu")
        assert len(history.train_loss) == 1

        post_train_report = get_sparsity_report(m)
        logger.info(
            f"SMOKE TEST [{method}]: overall sparsity after fine-tuning = "
            f"{post_train_report['overall_sparsity']:.3f}"
        )

        if method == "unstructured_l1":
            assert abs(post_train_report["overall_sparsity"] - 0.4) < 0.01
            finalize_pruning(m)
            final_report = get_sparsity_report(m)
            assert abs(final_report["overall_sparsity"] - 0.4) < 0.01

    logger.info("SMOKE TEST complete: pruning + fine-tune pipeline runs end-to-end for both methods.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pruning sweep across sparsities and methods.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/baseline/resnet18_fp32_best.pt")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smoke_test:
        set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
        run_smoke_test(cfg)
        return

    run_pruning_sweep(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
