import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from deployment.api.main import app
from deployment.api.model_registry import ModelRegistry
from src.models.resnet import build_resnet18_cifar

client = TestClient(app)


def _fake_image_bytes() -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_endpoint_returns_list():
    r = client.get("/models")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_predict_unknown_model_returns_404_or_503():
    r = client.post(
        "/predict",
        params={"model_id": "definitely_not_a_real_model"},
        files={"file": ("x.png", _fake_image_bytes(), "image/png")},
    )
    # 503 if no models exist at all yet, 404 if some exist but not this one --
    # either way, must NOT silently succeed or invent a prediction.
    assert r.status_code in (404, 503)


def test_predict_rejects_non_image_file():
    r = client.post(
        "/predict",
        params={"model_id": "anything"},
        files={"file": ("x.txt", b"not an image at all", "text/plain")},
    )
    assert r.status_code in (400, 404, 503)


def test_model_registry_empty_dir_reports_empty(tmp_path):
    registry = ModelRegistry(models_dir=tmp_path / "does_not_exist")
    assert registry.is_empty()
    assert registry.list_models() == []


def test_model_registry_discovers_real_checkpoint(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir(parents=True)
    model = build_resnet18_cifar()
    ckpt_path = baseline_dir / "resnet18_fp32_test.pt"
    torch.save(model.state_dict(), ckpt_path)

    registry = ModelRegistry(models_dir=tmp_path)
    assert not registry.is_empty()
    entry = registry.get("resnet18_fp32_test")
    assert entry.technique == "baseline"

    loaded = entry.load(_fake_cfg())
    assert loaded is not None


def test_model_registry_unknown_id_raises():
    registry = ModelRegistry(models_dir="models")  # may be empty in this repo state
    with pytest.raises(KeyError):
        registry.get("no_such_model")


class _FakeCfg:
    class data:
        num_classes = 10

    class model:
        cifar_stem = True


def _fake_cfg():
    return _FakeCfg()


def test_predict_end_to_end_with_real_checkpoint(tmp_path, monkeypatch):
    """Full happy path: register a real (untrained) checkpoint and confirm
    /predict returns a well-formed response with correctly-typed fields."""
    import deployment.api.main as main_module

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir(parents=True)
    model = build_resnet18_cifar()
    ckpt_path = baseline_dir / "resnet18_fp32_e2e.pt"
    torch.save(model.state_dict(), ckpt_path)

    monkeypatch.setattr(main_module, "_registry", ModelRegistry(models_dir=tmp_path))

    r = client.post(
        "/predict",
        params={"model_id": "resnet18_fp32_e2e"},
        files={"file": ("x.png", _fake_image_bytes(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["request_latency_ms"] > 0
    # No benchmark data was provided for this checkpoint -- must be None,
    # never a guessed number.
    assert body["benchmarked_latency_ms"] is None
    assert body["model_size_mb"] is None
