"""
app/config.py
─────────────
All tunable knobs live here. Read from environment variables with sensible
defaults so the server works out-of-the-box without any extra setup.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Paths
    model_path: str = os.path.join("outputs", "final_model.keras")
    class_names_path: str = os.path.join("outputs", "class_names.json")
    threshold_path: str = os.path.join("outputs", "threshold.json")

    # Model inference
    img_height: int = 288
    img_width: int = 288
    max_batch_size: int = 16

    # File upload validation
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    allowed_content_types: list[str] = [
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp",
        "image/x-bmp",
    ]
    allowed_extensions: list[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    # If P(good) is within [threshold - band, threshold + band], return "uncertain"
    uncertain_band: float = 0.10  # env: UNCERTAIN_BAND

    # TTA env: USE_TTA=true to enable
    use_tta: bool = False

    # CORS
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000",
                                "http://localhost:3000", "http://localhost:5173"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
