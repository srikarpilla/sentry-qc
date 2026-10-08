"""
app/model_service.py
────────────────────
Loads the Keras model once at startup and exposes predict() / predict_batch().

Key design decisions
────────────────────
* Model loaded via keras.models.load_model() - works with the full .keras
  SavedModel format so weights, architecture and internal EfficientNet
  rescaling are all restored correctly.
* Inference runs via model(x, training=False) which keeps BatchNorm in
  inference mode and skips Dropout.
* Threadpool execution is handled in main.py (run_in_executor) so the
  event loop is never blocked.
* Warm-up: one dummy prediction at startup so the first real request is fast.
* TTA (optional): average P over [original, h-flip, v-flip].
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from app.preprocess import preprocess_image_bytes

logger = logging.getLogger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────
_model: keras.Model | None = None
_interpreter: Any | None = None
_in_details: Any | None = None
_out_details: Any | None = None
_class_names: list[str] = []
_threshold: float = 0.58
_uncertain_band: float = 0.10
_model_path: str = ""


def load_model(settings: Any) -> None:
    """
    Load model + companion files.  Called once from the lifespan event.
    Fails fast with a clear message if any file is missing.
    """
    global _model, _interpreter, _in_details, _out_details
    global _class_names, _threshold, _uncertain_band, _model_path

    # ── Resolve paths ──────────────────────────────────────────────────────────
    model_path = settings.model_path
    class_names_path = settings.class_names_path
    threshold_path = settings.threshold_path

    for path, label in [
        (model_path, "Model"),
        (class_names_path, "class_names.json"),
        (threshold_path, "threshold.json"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} file not found at '{path}'. "
                f"Set the corresponding env var or place the file there."
            )

    # ── Load class names ───────────────────────────────────────────────────────
    with open(class_names_path, "r", encoding="utf-8") as fh:
        _class_names = json.load(fh)
    logger.info("Class names loaded: %s", _class_names)

    # ── Load threshold ─────────────────────────────────────────────────────────
    with open(threshold_path, "r", encoding="utf-8") as fh:
        thresh_data = json.load(fh)
    _threshold = float(thresh_data["threshold"])
    logger.info("Threshold loaded: %.4f (positive_class=%s)", _threshold, thresh_data.get("positive_class"))

    # ── Load model ─────────────────────────────────────────────────────────────
    _model_path = model_path
    logger.info("Loading model from '%s' ...", model_path)
    if model_path.endswith(".tflite"):
        _interpreter = tf.lite.Interpreter(model_path=model_path)
        _interpreter.allocate_tensors()
        _in_details = _interpreter.get_input_details()
        _out_details = _interpreter.get_output_details()
        logger.info("TFLite model loaded successfully.")
    else:
        _model = keras.models.load_model(model_path)
        logger.info(
            "Keras model loaded — input: %s  output: %s",
            _model.input_shape,
            _model.output_shape,
        )

    # ── Uncertain band ─────────────────────────────────────────────────────────
    _uncertain_band = settings.uncertain_band

    # ── Warm-up run ────────────────────────────────────────────────────────────
    dummy = np.zeros((1, settings.img_height, settings.img_width, 3), dtype=np.float32)
    _ = _run_inference(dummy)
    logger.info("Model warm-up complete.")


def is_loaded() -> bool:
    return _interpreter is not None or _model is not None


def get_class_names() -> list[str]:
    return _class_names


def get_threshold() -> float:
    return _threshold


def get_model_path() -> str:
    return _model_path


def _run_inference(batch: np.ndarray) -> np.ndarray:
    """Run model forward pass. Returns numpy array of shape (N, 2)."""
    if _interpreter is not None:
        preds = []
        for i in range(len(batch)):
            sample = np.expand_dims(batch[i], axis=0)
            _interpreter.set_tensor(_in_details[0]["index"], sample)
            _interpreter.invoke()
            preds.append(_interpreter.get_tensor(_out_details[0]["index"])[0])
        return np.array(preds, dtype=np.float32)
    elif _model is not None:
        preds = _model(batch, training=False)
        return preds.numpy()
    raise RuntimeError("No model loaded")


def _apply_threshold(p_good: float) -> tuple[str, str]:
    """Apply threshold + uncertain-band to p_good. Returns (label, status)."""
    low = _threshold - _uncertain_band
    high = _threshold + _uncertain_band
    label = "good" if p_good >= _threshold else "defective"
    status = "uncertain" if low <= p_good <= high else "ok"
    return label, status


def predict_single(
    image_bytes: bytes,
    filename: str,
    use_tta: bool = False,
) -> dict:
    """
    Predict class for one image.

    Parameters
    ----------
    image_bytes : bytes  – raw image data
    filename    : str    – used only for logging
    use_tta     : bool   – average over 3 TTA views

    Returns
    -------
    dict matching PredictionResult schema
    """
    assert is_loaded(), "Model not loaded"

    # ── Preprocess ─────────────────────────────────────────────────────────────
    batch = preprocess_image_bytes(image_bytes, use_tta=use_tta)
    # batch shape: (1, 288, 288, 3) or (3, 288, 288, 3) when TTA

    # ── Inference ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    probs = _run_inference(batch)                 # (N, 2)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # class_names[0] = "defective", class_names[1] = "good"
    mean_probs = probs.mean(axis=0)              # average over TTA views
    p_defective = float(mean_probs[0])
    p_good = float(mean_probs[1])

    label, status = _apply_threshold(p_good)
    confidence = p_good if label == "good" else p_defective

    logger.info(
        "PREDICT file=%s label=%s p_good=%.4f latency_ms=%.1f",
        filename, label, p_good, elapsed_ms,
    )

    return {
        "label": label,
        "p_good": round(p_good, 4),
        "p_defective": round(p_defective, 4),
        "confidence": round(confidence, 4),
        "threshold": _threshold,
        "status": status,
        "inference_ms": round(elapsed_ms, 2),
    }


def predict_batch(
    images: list[tuple[str, bytes]],
    use_tta: bool = False,
) -> list[dict]:
    """
    Predict for a list of (filename, image_bytes) tuples.
    Returns a list of dicts matching BatchPredictionResult schema.
    """
    results = []
    for filename, image_bytes in images:
        result = predict_single(image_bytes, filename, use_tta=use_tta)
        result["filename"] = filename
        results.append(result)
    return results
