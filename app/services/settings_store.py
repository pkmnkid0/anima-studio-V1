"""
Tiny JSON-file-backed settings store for small pieces of app-wide state
that don't belong in a training preset or a dataset folder - currently
just the per-source API credentials some booru sites now require (see
booru.py). Deliberately not a database: this is a handful of small
values edited rarely through a settings panel, so a single JSON file
under the app's data dir is the simplest thing that works and is easy
for a user to inspect/edit/back up by hand if they want to.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .. import config

SETTINGS_PATH = config.DATA_DIR / "settings.json"
_lock = threading.Lock()


def _read_all() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_all(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_PATH)


def get(key: str, default=None):
    with _lock:
        return _read_all().get(key, default)


def set(key: str, value) -> None:
    with _lock:
        data = _read_all()
        data[key] = value
        _write_all(data)


def get_booru_credentials() -> dict:
    """{"gelbooru": {"api_key": "...", "user_id": "..."}, "rule34": {...}}
    - only ever holds the sources that actually need credentials."""
    return get("booru_credentials", {}) or {}


def set_booru_credentials(source: str, api_key: str, user_id: str) -> dict:
    creds = get_booru_credentials()
    api_key = (api_key or "").strip()
    user_id = (user_id or "").strip()
    if api_key or user_id:
        creds[source] = {"api_key": api_key, "user_id": user_id}
    else:
        creds.pop(source, None)
    set("booru_credentials", creds)
    return creds
