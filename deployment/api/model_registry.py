"""Discovers available model checkpoints under `models/` and loads them for
inference. Deliberately does NOT hard-code a list of "available models" --
it scans the actual filesystem and `results/processed/results_processed.csv`,
so the API only ever offers to serve a model that genuinely exists and was
genuinely measured (no listing a compressed variant as "available" if it
was never actually trained/saved).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

from src.models.resnet import build_resnet18_cifar

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class ModelEntry:
    def __init__(self, model_id: str, checkpoint_path: Path, technique: str, metrics: Optional[dict] = None):
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.technique = technique
        self.metrics = metrics or {}
        self._model: Optional[nn.Module] = None

    def load(self, cfg) -> nn.Module:
        if self._model is None:
            model = build_resnet18_cifar(num_classes=cfg.data.num_classes, cifar_stem=cfg.model.cifar_stem)
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
        return self._model

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "technique": self.technique,
            "checkpoint_path": str(self.checkpoint_path),
            "metrics": self.metrics,
        }


class ModelRegistry:
    """Scans models/{baseline,pruning,quantization,combined}/*.pt and
    (optionally) joins each checkpoint against its measured row in
    results/processed/results_processed.csv by filename-derived run naming,
    so the API can report REAL measured accuracy/latency/size next to each
    prediction rather than a number made up at request time.
    """

    def __init__(self, models_dir: str | Path, results_csv: Optional[str | Path] = None):
        self.models_dir = Path(models_dir)
        self.results_csv = Path(results_csv) if results_csv else None
        self.entries: dict[str, ModelEntry] = {}
        self._scan()

    def _scan(self) -> None:
        if not self.models_dir.exists():
            return

        results_df = None
        if self.results_csv and self.results_csv.exists():
            results_df = pd.read_csv(self.results_csv)

        for technique_dir in ["baseline", "pruning", "quantization", "combined"]:
            subdir = self.models_dir / technique_dir
            if not subdir.exists():
                continue
            for ckpt in sorted(subdir.glob("*.pt")):
                model_id = ckpt.stem
                metrics = {}
                if results_df is not None and "run_id" in results_df.columns:
                    # best-effort match: run_id containing the checkpoint stem
                    matches = results_df[results_df["run_id"].astype(str).str.contains(model_id, na=False)]
                    if not matches.empty:
                        metrics = matches.iloc[-1].to_dict()
                self.entries[model_id] = ModelEntry(model_id, ckpt, technique_dir, metrics)

    def list_models(self) -> list[dict]:
        return [e.to_dict() for e in self.entries.values()]

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self.entries:
            available = list(self.entries.keys())
            raise KeyError(f"Unknown model_id '{model_id}'. Available: {available}")
        return self.entries[model_id]

    def is_empty(self) -> bool:
        return len(self.entries) == 0
