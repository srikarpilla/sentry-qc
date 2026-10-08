"""
tests/test_api.py
─────────────────
Pytest tests for the Sentry QC FastAPI backend.

Run:
  pytest tests/ -v

Covered:
  - GET /health
  - POST /predict  (valid JPEG)
  - POST /predict  (corrupt file → 400)
  - POST /predict  (wrong type → 415)
  - POST /predict  (oversized → 400)
  - Label-flip: P(good)=0.9 -> "good", P(good)=0.2 -> "defective"
"""
from __future__ import annotations

import io
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Import the app ─────────────────────────────────────────────────────────────
# We patch model_service so we never need the real .keras file during tests
import app.model_service as svc

def _make_mock_service(p_good: float = 0.85):
    """Patch model_service to return synthetic probabilities."""
    p_defective = 1.0 - p_good
    threshold = 0.58
    label = "good" if p_good >= threshold else "defective"
    confidence = p_good if label == "good" else p_defective
    low = threshold - 0.10
    high = threshold + 0.10
    status = "uncertain" if low <= p_good <= high else "ok"

    mock_result = {
        "label": label,
        "p_good": round(p_good, 4),
        "p_defective": round(p_defective, 4),
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "status": status,
        "inference_ms": 12.3,
    }
    return mock_result


def _fake_load_model(settings):
    """Fake lifespan model load — sets internal singletons."""
    svc._class_names = ["defective", "good"]
    svc._threshold = 0.58
    svc._uncertain_band = 0.10
    svc._model_path = "outputs/final_model.keras"
    svc._model = MagicMock()  # not None -> is_loaded() == True


def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    buf = io.BytesIO()
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """TestClient with model_service mocked out."""
    with patch.object(svc, "load_model", _fake_load_model):
        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "tensorflow_version" in data
    assert "threshold" in data
    assert isinstance(data["class_names"], list)


def test_predict_valid_image(client):
    jpeg_bytes = _make_jpeg()
    with patch.object(svc, "predict_single", return_value=_make_mock_service(0.85)):
        resp = client.post("/predict", files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "good"
    assert "p_good" in data
    assert "p_defective" in data
    assert "threshold" in data
    assert "status" in data
    assert "inference_ms" in data


def test_predict_corrupt_file(client):
    bad_bytes = b"this is not an image"
    resp = client.post("/predict", files={"file": ("bad.jpg", bad_bytes, "image/jpeg")})
    assert resp.status_code == 400


def test_predict_wrong_type(client):
    txt_bytes = b"plain text content"
    resp = client.post(
        "/predict",
        files={"file": ("document.txt", txt_bytes, "text/plain")},
    )
    assert resp.status_code == 415


def test_predict_oversize(client):
    # 11 MB file (exceeds 10 MB limit)
    big_bytes = b"\xff\xd8\xff" + b"X" * (11 * 1024 * 1024)
    resp = client.post(
        "/predict",
        files={"file": ("big.jpg", big_bytes, "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "large" in resp.json()["detail"].lower()


def test_label_flip_good(client):
    """P(good)=0.9 -> label must be good"""
    jpeg_bytes = _make_jpeg()
    with patch.object(svc, "predict_single", return_value=_make_mock_service(0.90)):
        resp = client.post("/predict", files={"file": ("t.jpg", jpeg_bytes, "image/jpeg")})
    assert resp.status_code == 200
    assert resp.json()["label"] == "good"


def test_label_flip_defective(client):
    """P(good)=0.2 -> label must be defective"""
    jpeg_bytes = _make_jpeg()
    with patch.object(svc, "predict_single", return_value=_make_mock_service(0.20)):
        resp = client.post("/predict", files={"file": ("t.jpg", jpeg_bytes, "image/jpeg")})
    assert resp.status_code == 200
    assert resp.json()["label"] == "defective"


def test_label_uncertain_band(client):
    """P(good)=0.60 is within [0.48, 0.68] -> status=uncertain"""
    jpeg_bytes = _make_jpeg()
    with patch.object(svc, "predict_single", return_value=_make_mock_service(0.60)):
        resp = client.post("/predict", files={"file": ("t.jpg", jpeg_bytes, "image/jpeg")})
    assert resp.status_code == 200
    assert resp.json()["status"] == "uncertain"


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["input_size"] == [288, 288]
    assert data["threshold"] == 0.58
    assert "metrics" in data


def test_batch_predict(client):
    jpeg_bytes = _make_jpeg()
    mock_batch = [
        {**_make_mock_service(0.85), "filename": "a.jpg"},
        {**_make_mock_service(0.10), "filename": "b.jpg"},
    ]
    with patch.object(svc, "predict_batch", return_value=mock_batch):
        resp = client.post(
            "/predict/batch",
            files=[
                ("files", ("a.jpg", jpeg_bytes, "image/jpeg")),
                ("files", ("b.jpg", jpeg_bytes, "image/jpeg")),
            ],
        )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2
    assert results[0]["label"] == "good"
    assert results[1]["label"] == "defective"
