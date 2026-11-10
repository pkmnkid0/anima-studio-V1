from __future__ import annotations

from flask import Blueprint, jsonify, request, send_file

from edit_backend import catalog as edit_catalog

from ..services import datasets, dep_installer, edit_batch, edit_engine, edit_models, edit_setup, imaging, resizer, upscaler_ai

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


@bp.get("/history")
def history():
    path = request.args.get("image_path", "")
    return jsonify(imaging.history_state(path)) if path else (jsonify({"error": "image_path is required"}), 400)


@bp.post("/history/<which>")
def history_step(which):
    body = request.get_json(force=True) or {}
    path = body.get("image_path")
    if which not in ("undo", "redo") or not path:
        return jsonify({"error": "image_path is required"}), 400
    done = imaging.undo_edit(path) if which == "undo" else imaging.redo_edit(path)
    return jsonify({"ok": done, **imaging.history_state(path)})


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


# ---------------------------------------------------------------------------------------------
# AI tools: IOPaint (the engine behind lama-cleaner) running in its own environment
# ---------------------------------------------------------------------------------------------

def _ai_error(exc: Exception, code: int = 400):
    return jsonify({"error": str(exc)}), code


@bp.get("/ai/catalog")
def ai_catalog():
    return jsonify({"catalog": edit_catalog.public_catalog(), "status": edit_engine.status()})


@bp.get("/ai/status")
def ai_status():
    return jsonify(edit_engine.status())


@bp.get("/ai/setup/status")
def ai_setup_status():
    return jsonify(edit_setup.status())


@bp.post("/ai/setup/start")
def ai_setup_start():
    try:
        return jsonify(edit_setup.start())
    except edit_setup.SetupError as exc:
        return _ai_error(exc)


@bp.post("/ai/setup/cancel")
def ai_setup_cancel():
    return jsonify(edit_setup.cancel())


@bp.get("/ai/verify")
def ai_verify():
    return jsonify(edit_setup.verify())


@bp.post("/ai/engine/stop")
def ai_engine_stop():
    return jsonify({"stopped": edit_engine.stop("Stopped from the editor."), "status": edit_engine.status()})


@bp.post("/ai/settings")
def ai_settings():
    body = request.get_json(force=True) or {}
    try:
        idle = int(body["idle_minutes"]) if body.get("idle_minutes") not in (None, "") else None
        return jsonify(edit_engine.set_settings(body.get("models_dir"), idle))
    except (edit_engine.EngineError, ValueError) as exc:
        return _ai_error(exc)


@bp.post("/ai/run")
def ai_run():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(edit_engine.submit(body))
    except edit_engine.EngineError as exc:
        return _ai_error(exc)


@bp.get("/ai/job")
def ai_job():
    job = edit_engine.job(request.args.get("id", ""))
    if job is None:
        return jsonify({"error": "Unknown job."}), 404
    return jsonify(job)


@bp.post("/ai/job/cancel")
def ai_job_cancel():
    body = request.get_json(force=True) or {}
    return jsonify({"cancelled": edit_engine.cancel(body.get("id"))})


@bp.get("/ai/result/<token>")
def ai_result(token):
    path = edit_engine.result_file(token)
    if path is None:
        return jsonify({"error": "That result is no longer available."}), 404
    resp = send_file(str(path), mimetype="image/png", conditional=False, max_age=0)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.delete("/ai/result/<token>")
def ai_result_discard(token):
    return jsonify({"ok": edit_engine.discard_result(token)})


@bp.post("/ai/commit")
def ai_commit():
    body = request.get_json(force=True) or {}
    image_path = body.get("image_path")
    path = edit_engine.result_file(str(body.get("token") or ""))
    if not image_path:
        return jsonify({"error": "image_path is required"}), 400
    if path is None:
        return jsonify({"error": "That result expired - run the tool again."}), 404
    try:
        saved = imaging.save_ai_result(image_path, str(path), save_as_copy=bool(body.get("save_as_copy")),
                                       keep_alpha=bool(body.get("keep_alpha")),
                                       background=str(body.get("background") or "#ffffff"))
    except imaging.ImagingError as exc:
        return _ai_error(exc)
    edit_engine.discard_result(str(body.get("token")))
    return jsonify({"ok": True, "path": saved})


@bp.post("/mask/adjust")
def mask_adjust():
    body = request.get_json(force=True) or {}
    try:
        mask = imaging.adjust_mask(body.get("mask") or "", body.get("op", "expand"), int(body.get("amount", 5)))
    except (imaging.ImagingError, ValueError) as exc:
        return _ai_error(exc)
    return jsonify({"mask": mask})


@bp.post("/ai/batch/start")
def ai_batch_start():
    try:
        return jsonify(edit_batch.start(request.get_json(force=True) or {}))
    except edit_batch.BatchError as exc:
        return _ai_error(exc)


@bp.get("/ai/batch/status")
def ai_batch_status():
    return jsonify(edit_batch.status())


@bp.post("/ai/batch/cancel")
def ai_batch_cancel():
    return jsonify(edit_batch.cancel())


@bp.post("/ai/models/download")
def ai_models_download():
    try:
        return jsonify(edit_models.start(request.get_json(force=True) or {}))
    except edit_models.ModelsError as exc:
        return _ai_error(exc)


@bp.get("/ai/models/download/status")
def ai_models_download_status():
    return jsonify(edit_models.status())


@bp.post("/ai/models/download/cancel")
def ai_models_download_cancel():
    return jsonify(edit_models.cancel())
