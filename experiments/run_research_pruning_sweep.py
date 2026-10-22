"""Physical structured-pruning sweep for the research ResNet-18 baseline."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch_pruning as tp

from src.data.cifar10 import make_dataloaders
from src.evaluation.metrics import evaluate_classification
from src.models.resnet import build_resnet18_cifar
from src.training.train import train_model
from src.utils.config import load_config
from src.utils.seed import set_seed


BASELINE = Path(
    "models/baseline/resnet18_research_30epochs/best_model.pt"
)
OUTPUT_ROOT = Path("models/research_pruning")
RESULTS_ROOT = Path("results/raw/research_pruning")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        choices=[0.0, 0.1, 0.2, 0.3, 0.4],
    )
    args = parser.parse_args()

    ratio = args.ratio
    name = f"ratio_{int(round(ratio * 100)):02d}"

    output_dir = OUTPUT_ROOT / name
    result_dir = RESULTS_ROOT / name

    checkpoint = output_dir / "best_model.pt"
    full_model = output_dir / "full_model.pt"
    result_path = result_dir / "results.json"

    if checkpoint.exists() or full_model.exists() or result_path.exists():
        raise SystemExit(
            f"Existing output detected for {name}. "
            "Refusing to overwrite an earlier experiment."
        )

    if not BASELINE.exists():
        raise FileNotFoundError(f"Baseline missing: {BASELINE}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/config_mac.yaml")
    set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
    torch.set_num_threads(4)

    cfg.training.lr = 0.01
    cfg.training.warmup_epochs = 0
    cfg.training.early_stopping_patience = 0

    train_loader, val_loader, _, _ = make_dataloaders(
        cfg, download=False
    )

    model = build_resnet18_cifar(
        num_classes=cfg.data.num_classes,
        cifar_stem=cfg.model.cifar_stem,
    )

    model.load_state_dict(
        torch.load(BASELINE, map_location="cpu")
    )
    model.eval()

    original_params = count_parameters(model)

    baseline_val_accuracy = evaluate_classification(
        model, val_loader, device="cpu"
    ).accuracy

    if ratio > 0:
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=torch.randn(1, 3, 32, 32),
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=ratio,
            iterative_steps=1,
            ignored_layers=[model.fc],
        )
        pruner.step()

    pruned_params = count_parameters(model)

    before_accuracy = evaluate_classification(
        model, val_loader, device="cpu"
    ).accuracy

    print("\n===== RESEARCH PRUNING EXPERIMENT =====", flush=True)
    print("Ratio:", ratio, flush=True)
    print("Baseline validation accuracy:", f"{baseline_val_accuracy:.2%}", flush=True)
    print("Original parameters:", original_params, flush=True)
    print("Pruned parameters:", pruned_params, flush=True)
    print("Before fine-tuning:", f"{before_accuracy:.2%}", flush=True)

    history = train_model(
        model,
        train_loader,
        val_loader,
        cfg,
        epochs=3,
        checkpoint_path=checkpoint,
        device="cpu",
    )

    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")
    )
    model.eval()

    after_accuracy = evaluate_classification(
        model, val_loader, device="cpu"
    ).accuracy

    # The full object preserves the physically pruned layer dimensions.
    torch.save(model, full_model)

    result = {
        "dataset": "CIFAR-10",
        "model": "ResNet-18",
        "seed": cfg.project.seed,
        "pruning_method": (
            "unpruned_control" if ratio == 0
            else "physical_structured_magnitude_l2"
        ),
        "requested_pruning_ratio": ratio,
        "baseline_validation_accuracy": baseline_val_accuracy,
        "original_parameters": original_params,
        "pruned_parameters": pruned_params,
        "parameter_reduction_pct": (
            100 * (original_params - pruned_params) / original_params
        ),
        "validation_accuracy_before_finetune": before_accuracy,
        "validation_accuracy_after_finetune": after_accuracy,
        "best_validation_accuracy": history.best_val_accuracy,
        "best_epoch": history.best_epoch + 1,
        "fine_tune_epochs": 3,
        "learning_rate": cfg.training.lr,
        "training_time_sec": history.total_train_time_sec,
        "checkpoint": str(checkpoint),
        "full_model": str(full_model),
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "val_accuracy": history.val_accuracy,
        "epoch_time_sec": history.epoch_time_sec,
    }

    with result_path.open("w") as f:
        json.dump(result, f, indent=2)

    with (result_dir / "results.csv").open("w", newline="") as f:
        scalar_result = {
            k: v for k, v in result.items()
            if not isinstance(v, list)
        }
        writer = csv.DictWriter(
            f, fieldnames=scalar_result.keys()
        )
        writer.writeheader()
        writer.writerow(scalar_result)

    print("\n===== EXPERIMENT COMPLETE =====")
    print("Ratio:", ratio)
    print("Parameter reduction:", f"{result['parameter_reduction_pct']:.2f}%")
    print("Validation accuracy:", f"{after_accuracy:.2%}")
    print("Best fine-tuning epoch:", history.best_epoch + 1)
    print("Results:", result_path)
    print("Model:", full_model)


if __name__ == "__main__":
    main()
