"""Saved app settings (non-training UI settings) and named presets.

Training parameters are deliberately NOT part of this - they live in the
user's own TOML files."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import settings_store

bp = Blueprint("appsettings", __name__)
KEY = "ui_settings"


def _load() -> dict:
    d = settings_store.get(KEY) or {}
    d.setdefault("current", {})
    d.setdefault("presets", {})
    return d


def _clean(snap) -> dict:
    if not isinstance(snap, dict):
        return {}
    return {str(k): v for k, v in snap.items() if isinstance(v, (str, bool, int, float))}


@bp.get("")
def get_all():
    return jsonify(_load())


@bp.put("/current")
def put_current():
    d = _load()
    d["current"] = _clean((request.get_json(force=True) or {}).get("values"))
    settings_store.set(KEY, d)
    return jsonify({"ok": True})


@bp.post("/presets")
def save_preset():
    body = request.get_json(force=True) or {}
    name = str(body.get("name", "")).strip()
    if not name:
        return jsonify({"error": "Give the preset a name."}), 400
    d = _load()
    d["presets"][name[:60]] = _clean(body.get("values"))
    settings_store.set(KEY, d)
    return jsonify(d)


@bp.delete("/presets/<name>")
def delete_preset(name: str):
    d = _load()
    d["presets"].pop(name, None)
    settings_store.set(KEY, d)
    return jsonify(d)
