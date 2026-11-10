"""Saved app settings (non-training UI settings) and named presets.

Training parameters are deliberately NOT part of this - they live in the
user's own TOML files."""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from ..services import settings_store

bp = Blueprint("appsettings", __name__)
KEY = "ui_settings"


def _load() -> dict:
    d = settings_store.get(KEY) or {}
    d.setdefault("current", {})
    d.setdefault("presets", {})
    d.setdefault("auto_load", True)      # "always load these settings when opening the app"
    d.setdefault("startup", "last")      # "last" or "preset:<name>"
    d.setdefault("log_dir", "")          # dedicated TensorBoard log folder ("" = <output>/logs)
    d.setdefault("keybinds", {})         # Magic eraser shortcuts: {action: [key, key]}
    return d


def _clean_keybinds(raw) -> dict:
    """{action: ["Ctrl+Z", ""]} - short strings only, at most two keys per action."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for action, keys in list(raw.items())[:80]:
        if not isinstance(action, str) or not (1 <= len(action) <= 40) or not isinstance(keys, list):
            continue
        out[action] = [k[:32] if isinstance(k, str) else "" for k in keys[:2]]
    return out


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


@bp.put("/options")
def put_options():
    body = request.get_json(force=True) or {}
    d = _load()
    if "auto_load" in body:
        d["auto_load"] = bool(body["auto_load"])
    if "startup" in body:
        st = str(body["startup"])
        if st == "last" or (st.startswith("preset:") and st[7:] in d["presets"]):
            d["startup"] = st
    if "keybinds" in body:
        d["keybinds"] = _clean_keybinds(body["keybinds"])
    if "log_dir" in body:
        ld = str(body["log_dir"] or "").strip()
        if ld:
            try:
                Path(ld).expanduser().mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return jsonify({"error": f"Can't use that log folder: {exc}"}), 400
        d["log_dir"] = ld
    settings_store.set(KEY, d)
    return jsonify(d)


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
