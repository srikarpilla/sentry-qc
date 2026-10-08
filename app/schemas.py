"""
app/schemas.py
──────────────
Pydantic response models for all API endpoints.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    label: Literal["good", "defective"]
    p_good: float = Field(..., ge=0.0, le=1.0, description="Probability of being a good part")
    p_defective: float = Field(..., ge=0.0, le=1.0, description="Probability of being defective")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted class")
    threshold: float = Field(..., description="Decision threshold applied to P(good)")
    status: Literal["ok", "uncertain"] = Field(..., description="ok or uncertain (near threshold)")
    inference_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchPredictionResult(PredictionResult):
    filename: str = Field(..., description="Original uploaded filename")


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    model_loaded: bool
    model_file: str
    class_names: list[str]
    threshold: float
    tensorflow_version: str


class ModelInfoResponse(BaseModel):
    input_size: tuple[int, int]
    input_channels: int
    class_order: list[str]
    threshold: float
    positive_class: str
    metrics: dict
