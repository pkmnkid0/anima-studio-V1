"""App-wide paths and defaults."""
from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ANIMA_STUDIO_DATA", APP_DIR / "data")).resolve()
DATASETS_DIR = DATA_DIR / "datasets"
PRESETS_DIR = DATA_DIR / "presets"
MODELS_CACHE_DIR = DATA_DIR / "model_cache"
THUMBNAIL_CACHE_DIR = DATA_DIR / "thumb_cache"

for _d in (DATA_DIR, DATASETS_DIR, PRESETS_DIR, MODELS_CACHE_DIR, THUMBNAIL_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
CAPTION_EXTENSIONS = {".txt", ".caption"}

HOST = os.environ.get("ANIMA_STUDIO_HOST", "127.0.0.1")
PORT = int(os.environ.get("ANIMA_STUDIO_PORT", "7864"))
