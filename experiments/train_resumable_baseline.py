"""Resumable CIFAR-10 ResNet-18 training with isolated checkpoints."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data.cifar10 import make_dataloaders
from src.models.resnet import build_resnet18_cifar
from src.training.train import (
    build_optimizer_and_scheduler,
    train_one_epoch,
    compute_val_loss,
)
from src.evaluation.metrics import evaluate_classification
from src.utils.config import load_config
from src.utils.seed import set_seed


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="models/baseline/resnet18_research_30epochs",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    cfg = load_config("configs/config_mac.yaml")
    set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
    torch.set_num_threads(4)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    latest_path = output / "latest_state.pt"
    best_path = output / "best_model.pt"
    history_path = output / "history.json"

    train_loader, val_loader, _, _ = make_dataloaders(
        cfg, download=False
    )

    if args.max_train_samples is not None:
        from torch.utils.data import DataLoader, Subset
        n = args.max_train_samples
        if not 1 <= n <= len(train_loader.dataset):
            raise ValueError("Invalid max-train-samples")
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(n)),
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=0,
            generator=train_loader.generator,
        )

    if args.max_val_samples is not None:
        from torch.utils.data import DataLoader, Subset
        n = args.max_val_samples
        if not 1 <= n <= len(val_loader.dataset):
            raise ValueError("Invalid max-val-samples")
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(n)),
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=0,
        )

    model = build_resnet18_cifar(
        num_classes=cfg.data.num_classes,
        cifar_stem=cfg.model.cifar_stem,
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, epochs=args.epochs
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.training.label_smoothing
    )

    run_config = {
        "seed": cfg.project.seed,
        "total_epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "batch_size": train_loader.batch_size,
        "num_workers": train_loader.num_workers,
        "learning_rate": cfg.training.lr,
        "optimizer": cfg.training.optimizer,
        "momentum": cfg.training.momentum,
        "weight_decay": cfg.training.weight_decay,
        "nesterov": cfg.training.nesterov,
        "scheduler": cfg.training.lr_scheduler,
        "warmup_epochs": cfg.training.warmup_epochs,
        "label_smoothing": cfg.training.label_smoothing,
        "grad_clip_norm": cfg.training.grad_clip_norm,
        "split_seed": cfg.data.split_seed,
        "cifar_stem": cfg.model.cifar_stem,
        "num_classes": cfg.data.num_classes,
        "deterministic": cfg.project.deterministic,
        "cpu_threads": torch.get_num_threads(),
    }

    start_epoch = 0
    best_accuracy = -1.0
    history = []

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: {latest_path} does not exist"
            )

        state = torch.load(latest_path, map_location="cpu")

        if state.get("run_config") != run_config:
            raise ValueError(
                "Resume configuration mismatch or missing metadata. "
                "Use the original settings or start a new run."
            )

        if state["total_epochs"] != args.epochs:
            raise ValueError(
                "Resume requires the same total epoch count. "
                "Changing it would alter the learning-rate schedule."
            )

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        start_epoch = state["completed_epoch"]
        best_accuracy = state["best_accuracy"]
        history = state["history"]

        torch.set_rng_state(state["torch_rng"])

        if "dataloader_rng" not in state:
            raise ValueError(
                "Checkpoint lacks DataLoader RNG state. "
                "Exact shuffle-order continuation cannot be guaranteed."
            )
        train_loader.generator.set_state(state["dataloader_rng"])

        if "numpy_rng" in state:
            import numpy as np
            np.random.set_state(state["numpy_rng"])

        if "python_rng" in state:
            import random
            random.setstate(state["python_rng"])

        print(f"Resuming from epoch {start_epoch}")

    elif latest_path.exists() or best_path.exists():
        raise FileExistsError(
            f"Existing training files found in {output}. "
            "Use --resume or choose a new --output directory."
        )

    print("===== RESEARCH BASELINE TRAINING =====")
    print("Target epochs:", args.epochs)
    print("Starting epoch:", start_epoch + 1)
    print("Output:", output)

    for epoch in range(start_epoch, args.epochs):
        start = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device="cpu",
            grad_clip_norm=cfg.training.grad_clip_norm,
        )

        val_loss = compute_val_loss(
            model, val_loader, criterion, device="cpu"
        )

        val_accuracy = evaluate_classification(
            model, val_loader, device="cpu"
        ).accuracy

        scheduler.step()

        elapsed = time.perf_counter() - start

        record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "epoch_time_sec": float(elapsed),
            "learning_rate_next_epoch": float(
                optimizer.param_groups[0]["lr"]
            ),
        }
        history.append(record)

        if val_accuracy > best_accuracy:
            best_accuracy = float(val_accuracy)
            atomic_save(model.state_dict(), best_path)

        import random
        import numpy as np

        state = {
            "completed_epoch": epoch + 1,
            "total_epochs": args.epochs,
            "run_config": run_config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_accuracy": best_accuracy,
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "dataloader_rng": train_loader.generator.get_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }

        atomic_save(state, latest_path)

        temporary_history = history_path.with_suffix(".json.tmp")
        temporary_history.write_text(json.dumps(history, indent=2))
        temporary_history.replace(history_path)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.2%} | "
            f"best={best_accuracy:.2%} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
            print("Controlled stop after checkpoint save.")
            return

    print("\nTraining complete.")
    print("Best validation accuracy:", f"{best_accuracy:.2%}")
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()
