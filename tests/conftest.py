import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path_factory, monkeypatch):
    """Keep every test away from the real data/ folder.

    config paths are computed at import time, so rather than re-importing
    modules this points the stores that read/write files at a throwaway folder
    for each test (kept outside tmp_path so tests can still assert an empty
    folder). A test that needs a specific path can monkeypatch on top - its
    value wins for that test."""
    from app import config
    from app.services import img_cache, settings_store

    data = tmp_path_factory.mktemp("studio-data")
    subdirs = {"DATASETS_DIR": "datasets", "PRESETS_DIR": "presets",
               "MODELS_CACHE_DIR": "model_cache", "THUMBNAIL_CACHE_DIR": "thumb_cache",
               "BOORU_IMG_CACHE_DIR": "cache/booru_img"}
    for sub in subdirs.values():
        (data / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    for name, sub in subdirs.items():
        monkeypatch.setattr(config, name, data / sub)
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setattr(img_cache, "CACHE_DIR", data / "cache" / "booru_img")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_STUDIO_DATA", str(tmp_path))
    from app import create_app
    return create_app().test_client()
