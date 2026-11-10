from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import updater

bp = Blueprint("update", __name__)


def _err(exc, code=400):
    return jsonify({"error": str(exc)}), code


@bp.get("/status")
def status():
    last = updater.settings_store.get("update_last", {}) or {}
    return jsonify({"version": updater.current_version(), "packaged": updater.is_packaged(),
                    "settings": updater.settings(), "last": last, "progress": updater.progress()})


@bp.post("/settings")
def settings():
    try:
        return jsonify(updater.save_settings(request.get_json(force=True) or {}))
    except updater.UpdateError as exc:
        return _err(exc)


@bp.post("/check")
def check():
    try:
        return jsonify(updater.check())
    except updater.UpdateError as exc:
        return _err(exc)


@bp.post("/apply")
def apply():
    try:
        return jsonify(updater.start(request.get_json(force=True) or {}))
    except updater.UpdateError as exc:
        return _err(exc)


@bp.get("/progress")
def progress():
    return jsonify(updater.progress())


@bp.post("/restart")
def restart():
    try:
        return jsonify(updater.restart())
    except updater.UpdateError as exc:
        return _err(exc)
