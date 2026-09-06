"""Baseline training loop, reused (with a much shorter schedule) for the
post-pruning fine-tuning step in Phase 3.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_classification
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_accuracy: list = field(default_factory=list)
    epoch_time_sec: list = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_epoch: int = -1
    total_train_time_sec: float = 0.0
    stopped_early: bool = False
    stopped_at_epoch: Optional[int] = None


def build_optimizer_and_scheduler(model: nn.Module, cfg, epochs: Optional[int] = None):
    t = cfg.training
    epochs = epochs if epochs is not None else t.epochs

    if t.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=t.lr,
            momentum=t.momentum,
            weight_decay=t.weight_decay,
            nesterov=t.nesterov,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {t.optimizer}. Add it here explicitly before using it.")

    if t.lr_scheduler == "cosine":
        warmup_epochs = min(t.warmup_epochs, max(epochs - 1, 0))

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            import math

            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {t.lr_scheduler}. Add it here explicitly before using it.")

    return optimizer, scheduler


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer, criterion, device: str, grad_clip_norm=None) -> float:
    model.train()
    model.to(device)
    running_loss = 0.0
    n_samples = 0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def compute_val_loss(model: nn.Module, dataloader: DataLoader, criterion, device: str) -> float:
    model.eval()
    running_loss = 0.0
    n_samples = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
    return running_loss / max(n_samples, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg,
    epochs: Optional[int] = None,
    checkpoint_path: Optional[str | Path] = None,
    device: str = "cpu",
    log_every_n_epochs: int = 1,
) -> TrainingHistory:
    """Train `model` in place, returning a TrainingHistory. If
    `checkpoint_path` is given, saves the best-val-accuracy state_dict there
    (this is the "save the trained baseline model" step of the spec).

    Early stopping: if val_accuracy hasn't improved for
    `cfg.training.early_stopping_patience` epochs, training stops before
    `epochs` is reached (recorded in the history, not silently truncated).
    """
    epochs = epochs if epochs is not None else cfg.training.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, epochs=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    history = TrainingHistory()
    patience_counter = 0
    total_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip_norm=cfg.training.grad_clip_norm
        )
        val_loss = compute_val_loss(model, val_loader, criterion, device)
        val_metrics = evaluate_classification(model, val_loader, device=device)
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_metrics.accuracy)
        history.epoch_time_sec.append(epoch_time)

        improved = val_metrics.accuracy > history.best_val_accuracy
        if improved:
            history.best_val_accuracy = val_metrics.accuracy
            history.best_epoch = epoch
            patience_counter = 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if (epoch % log_every_n_epochs) == 0 or epoch == epochs - 1:
            logger.info(
                f"epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_metrics.accuracy:.4f} "
                f"(best={history.best_val_accuracy:.4f} @ epoch {history.best_epoch+1}) "
                f"| {epoch_time:.1f}s/epoch"
            )

        if cfg.training.early_stopping_patience and patience_counter >= cfg.training.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no val improvement for {patience_counter} epochs).")
            history.stopped_early = True
            history.stopped_at_epoch = epoch
            break

    history.total_train_time_sec = time.perf_counter() - total_start
    return history
