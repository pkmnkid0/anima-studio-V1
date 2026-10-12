from __future__ import annotations

import hashlib
import io
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file, abort

from .. import config
from ..services import datasets

bp = Blueprint("files", __name__)


@bp.get("/browse")
def browse():
    path = request.args.get("path") or None
    ext_param = request.args.get("ext")
    file_extensions = tuple(f".{e.strip().lstrip('.').lower()}" for e in ext_param.split(",") if e.strip()) if ext_param else None
    try:
        return jsonify(datasets.list_dir(path, file_extensions=file_extensions))
    except NotADirectoryError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/subsets")
def subsets():
    return jsonify({"subsets": datasets.list_subsets()})


@bp.post("/subsets/add")
def subsets_add():
    body = request.get_json(force=True) or {}
    path = body.get("path")
    if not path:
        return jsonify({"error": "path is required"}), 400
    try:
        return jsonify({"subsets": datasets.register_subset(path)})
    except NotADirectoryError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/subsets/remove")
def subsets_remove():
    body = request.get_json(force=True) or {}
    path = body.get("path")
    if not path:
        return jsonify({"error": "path is required"}), 400
    return jsonify({"subsets": datasets.unregister_subset(path)})


@bp.post("/mkdir")
def mkdir():
    body = request.get_json(force=True) or {}
    path = body.get("path")
    if not path:
        return jsonify({"error": "path is required"}), 400
    created = datasets.create_dir(path)
    return jsonify({"path": created})


@bp.get("/thumbnail")
def thumbnail():
    path = request.args.get("path", "")
    size = int(request.args.get("size", 240))
    p = Path(path)
    if not p.exists() or not p.is_file() or p.suffix.lower() not in config.IMAGE_EXTENSIONS:
        abort(404)

    cache_key = hashlib.sha1(f"{p}|{size}|{p.stat().st_mtime}".encode()).hexdigest()
    cache_path = config.THUMBNAIL_CACHE_DIR / f"{cache_key}.jpg"
    if cache_path.exists():
        return send_file(cache_path, mimetype="image/jpeg")

    from PIL import Image
    img = Image.open(p).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    cache_path.write_bytes(buf.getvalue())
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@bp.get("/image")
def full_image():
    path = request.args.get("path", "")
    p = Path(path)
    if not p.exists() or not p.is_file() or p.suffix.lower() not in config.IMAGE_EXTENSIONS:
        abort(404)
    return send_file(p)
