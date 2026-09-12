"""FastAPI inference service.

Serves whichever trained checkpoints actually exist under `models/` (see
model_registry.py -- nothing is offered that wasn't really produced by the
research pipeline). Endpoints:

  GET  /models             -- list available model variants + their measured metrics
  POST /predict             -- run inference with a chosen model variant
  GET  /health              -- liveness check

Run locally:
    uvicorn deployment.api.main:app --reload --port 8000

Run in Docker: see Dockerfile at the repo root / deployment/README.md.

Honesty note (per project spec): this service reports whatever latency
`results_processed.csv` recorded for that model during the Phase 6
benchmark, AND the actual latency of the request just served -- both are
returned so a compressed model is never implied to be faster than
benchmarking actually showed. It does not claim a speed advantage that
wasn't measured.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from deployment.api.model_registry import CIFAR10_CLASSES, ModelRegistry
from src.evaluation.metrics import get_model_size_mb
from src.utils.config import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Efficient ML Benchmark -- Inference API",
    description="Serves CIFAR-10 ResNet-18 baseline and compressed variants for comparison.",
    version="0.1.0",
)

_cfg = load_config()
_registry = ModelRegistry(
    models_dir=_cfg.paths.models_dir,
    results_csv=Path(_cfg.paths.results_processed) / "results_processed.csv",
)

_eval_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(_cfg.data.normalize_mean, _cfg.data.normalize_std),
])


class PredictionResponse(BaseModel):
    model_id: str
    technique: str
    predicted_class: str
    predicted_class_index: int
    confidence: float
    request_latency_ms: float
    benchmarked_latency_ms: float | None = None
    model_size_mb: float | None = None
    compression_ratio: float | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_available": len(_registry.entries)}


@app.get("/models")
def list_models() -> list[dict]:
    if _registry.is_empty():
        return []
    return _registry.list_models()


@app.post("/predict", response_model=PredictionResponse)
async def predict(model_id: str, file: UploadFile = File(...)) -> PredictionResponse:
    if _registry.is_empty():
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model checkpoints found under 'models/'. Run "
                "experiments/train_baseline.py (and, optionally, the "
                "pruning/quantization/combined scripts) before starting this service."
            ),
        )
    try:
        entry = _registry.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    model = entry.load(_cfg)
    input_tensor = _eval_transform(image).unsqueeze(0)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = probs.max(dim=1)
    request_latency_ms = (time.perf_counter() - start) * 1000.0

    metrics = entry.metrics or {}
    return PredictionResponse(
        model_id=entry.model_id,
        technique=entry.technique,
        predicted_class=CIFAR10_CLASSES[int(predicted_idx.item())],
        predicted_class_index=int(predicted_idx.item()),
        confidence=float(confidence.item()),
        request_latency_ms=request_latency_ms,
        benchmarked_latency_ms=metrics.get("latency_mean_ms"),
        model_size_mb=metrics.get("model_size_mb"),
        compression_ratio=metrics.get("compression_ratio"),
    )
