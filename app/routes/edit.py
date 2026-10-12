from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import datasets, dep_installer, imaging, resizer, upscaler_ai

bp = Blueprint("edit", __name__)

UPSCALER_DEPS_JOB = "upscaler_ultrasharp_deps"


@bp.get("/images")
def images():
    folder = request.args.get("folder", "")
    try:
        imgs = datasets.list_images(folder)
    except NotADirectoryError as exc:
        return jsonify({"error": str(exc)}), 400
    for img in imgs:
        img["has_backup"] = imaging.has_backup(img["path"])
    return jsonify({"images": imgs})


@bp.post("/inpaint")
def inpaint():
    body = request.get_json(force=True) or {}
    image_path = body.get("image_path")
    mask = body.get("mask")
    method = body.get("method", "telea")
    radius = int(body.get("radius", 4))
    save_as_copy = bool(body.get("save_as_copy", False))
    if not image_path or not mask:
        return jsonify({"error": "image_path and mask are required"}), 400
    try:
        result_path = imaging.inpaint(image_path, mask, method=method, radius=radius, save_as_copy=save_as_copy)
    except imaging.ImagingError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "path": result_path})


@bp.post("/crop")
def crop():
    body = request.get_json(force=True) or {}
    image_path = body.get("image_path")
    box = body.get("box")  # [x0, y0, x1, y1]
    save_as_copy = bool(body.get("save_as_copy", False))
    dest_dir = body.get("dest_dir") or None
    upscale_factor = body.get("upscale")
    upscale_method = body.get("upscale_method", "lanczos")
    if not image_path or not box or len(box) != 4:
        return jsonify({"error": "image_path and a 4-value box are required"}), 400
    try:
        result_path = imaging.crop(
            image_path, tuple(box), save_as_copy=save_as_copy, dest_dir=dest_dir,
            upscale_factor=float(upscale_factor) if upscale_factor else None,
            upscale_method=upscale_method,
        )
    except imaging.ImagingError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "path": result_path})


@bp.get("/upscaler/status")
def upscaler_status():
    return jsonify(upscaler_ai.status())


@bp.post("/upscaler/install_deps")
def upscaler_install_deps():
    try:
        return jsonify(dep_installer.start(UPSCALER_DEPS_JOB, upscaler_ai.DEPS))
    except dep_installer.InstallError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/upscaler/install_deps_status")
def upscaler_install_deps_status():
    return jsonify(dep_installer.status(UPSCALER_DEPS_JOB))


@bp.post("/upscaler/download")
def upscaler_download():
    if upscaler_ai.is_downloaded():
        return jsonify({"status": "done"})
    try:
        upscaler_ai.download_async()
    except upscaler_ai.UpscalerError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "downloading"})


@bp.get("/upscaler/download_status")
def upscaler_download_status():
    return jsonify(upscaler_ai.download_progress())


@bp.post("/resize/start")
def resize_start():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(resizer.start(body.get("folder", ""), body.get("resolutions", [])))
    except resizer.ResizeError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/resize/status")
def resize_status():
    return jsonify(resizer.status())


@bp.post("/revert")
def revert():
    body = request.get_json(force=True) or {}
    image_path = body.get("image_path")
    if not image_path:
        return jsonify({"error": "image_path is required"}), 400
    ok = imaging.revert_to_backup(image_path)
    if not ok:
        return jsonify({"error": "No backup found for this image - it hasn't been edited yet."}), 404
    return jsonify({"ok": True})


@bp.get("/duplicates")
def duplicates():
    folder = request.args.get("folder", "")
    threshold = int(request.args.get("threshold", 6))
    try:
        imgs = datasets.list_images(folder)
    except NotADirectoryError as exc:
        return jsonify({"error": str(exc)}), 400
    paths = [img["path"] for img in imgs]
    groups = imaging.find_duplicate_groups(paths, threshold=threshold)
    return jsonify({"groups": groups, "scanned": len(paths)})


@bp.post("/delete_images")
def delete_images():
    body = request.get_json(force=True) or {}
    image_paths = body.get("image_paths", [])
    if not image_paths:
        return jsonify({"error": "image_paths is required"}), 400
    results = datasets.delete_images(image_paths)
    deleted = sum(1 for r in results if r["status"] == "ok")
    return jsonify({"results": results, "deleted": deleted})
