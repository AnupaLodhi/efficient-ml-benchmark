import csv
import json
from pathlib import Path

import torch
import torch_pruning as tp

from src.data.cifar10 import make_dataloaders
from src.models.resnet import build_resnet18_cifar
from src.training.train import train_model
from src.evaluation.metrics import evaluate_classification
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    cfg = load_config("configs/config_mac.yaml")
    set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
    torch.set_num_threads(4)

    train_loader, val_loader, _, _ = make_dataloaders(cfg, download=False)

    model = build_resnet18_cifar(
        num_classes=cfg.data.num_classes,
        cifar_stem=cfg.model.cifar_stem,
    )
    model.load_state_dict(
        torch.load("models/baseline/resnet18_fp32_best.pt", map_location="cpu")
    )
    model.eval()

    original_params = sum(p.numel() for p in model.parameters())

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1, 3, 32, 32),
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=0.2,
        iterative_steps=1,
        ignored_layers=[model.fc],
    )
    pruner.step()

    pruned_params = sum(p.numel() for p in model.parameters())

    before_accuracy = evaluate_classification(
        model, val_loader, device="cpu"
    ).accuracy

    cfg.training.lr = 0.01
    cfg.training.warmup_epochs = 0

    checkpoint = Path("models/pruning/resnet18_structured_physical_s0.2_best.pt")

    history = train_model(
        model,
        train_loader,
        val_loader,
        cfg,
        epochs=3,
        checkpoint_path=checkpoint,
        device="cpu",
    )

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    after_accuracy = evaluate_classification(
        model, val_loader, device="cpu"
    ).accuracy

    output_dir = Path("results/raw/structured")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": "CIFAR-10",
        "model": "ResNet-18",
        "seed": cfg.project.seed,
        "pruning_method": "physical_structured_magnitude_l2",
        "requested_pruning_ratio": 0.2,
        "original_params": original_params,
        "pruned_params": pruned_params,
        "parameter_reduction_pct": 100 * (original_params - pruned_params) / original_params,
        "validation_accuracy_before_finetune": before_accuracy,
        "validation_accuracy_after_finetune": after_accuracy,
        "best_validation_accuracy": history.best_val_accuracy,
        "best_epoch": history.best_epoch + 1,
        "fine_tune_epochs": 3,
        "learning_rate": 0.01,
        "training_time_sec": history.total_train_time_sec,
        "checkpoint": str(checkpoint),
    }

    with (output_dir / "finetune_s0.2.json").open("w") as f:
        json.dump(result, f, indent=2)

    with (output_dir / "finetune_s0.2.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print("\n===== STRUCTURED PRUNING EXPERIMENT =====")
    print("Original parameters:", f"{original_params:,}")
    print("Pruned parameters:", f"{pruned_params:,}")
    print("Parameter reduction:", f"{result['parameter_reduction_pct']:.2f}%")
    print("Validation accuracy before fine-tuning:", f"{before_accuracy:.2%}")
    print("Validation accuracy after fine-tuning:", f"{after_accuracy:.2%}")
    print("Best epoch:", history.best_epoch + 1)
    print("Training time:", f"{history.total_train_time_sec:.1f} seconds")
    print("Checkpoint:", checkpoint)
    print("Results:", output_dir)


if __name__ == "__main__":
    main()
