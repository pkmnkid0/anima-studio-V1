from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from ..services import captions, datasets, dep_installer, tagger

bp = Blueprint("caption", __name__)

TAGGER_TIMM_DEPS_JOB = "tagger_timm_deps"


@bp.get("/images")
def images():
    folder = request.args.get("folder", "")
    try:
        return jsonify({"images": datasets.list_images(folder)})
    except NotADirectoryError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/stats")
def stats():
    folder = request.args.get("folder", "")
    try:
        return jsonify(datasets.caption_stats(folder))
    except NotADirectoryError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/tag_frequency")
def tag_frequency():
    folder = request.args.get("folder", "")
    try:
        return jsonify({"tags": datasets.tag_frequency(folder, with_examples=True)})
    except NotADirectoryError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/save")
def save():
    body = request.get_json(force=True) or {}
    image_path = body.get("image_path")
    caption = body.get("caption", "")
    if not image_path:
        return jsonify({"error": "image_path is required"}), 400
    captions.write_caption(image_path, caption)
    return jsonify({"ok": True})


@bp.post("/bulk/add_tag")
def bulk_add_tag():
    body = request.get_json(force=True) or {}
    changed = captions.bulk_add_tag(body.get("folder", ""), body.get("tag", ""), body.get("position", "end"))
    return jsonify({"changed": changed})


@bp.post("/bulk/remove_tag")
def bulk_remove_tag():
    body = request.get_json(force=True) or {}
    changed = captions.bulk_remove_tag(body.get("folder", ""), body.get("tag", ""))
    return jsonify({"changed": changed})


@bp.post("/bulk/remove_tags")
def bulk_remove_tags():
    body = request.get_json(force=True) or {}
    changed = captions.bulk_remove_tags(body.get("folder", ""), body.get("tags", []))
    return jsonify({"changed": changed})


@bp.post("/bulk/add_tag_to_images")
def add_tag_to_images():
    body = request.get_json(force=True) or {}
    changed = captions.add_tag_to_images(
        body.get("image_paths", []), body.get("tag", ""), body.get("position", "end")
    )
    return jsonify({"changed": changed})


@bp.post("/bulk/find_replace")
def bulk_find_replace():
    body = request.get_json(force=True) or {}
    changed = captions.bulk_find_replace(
        body.get("folder", ""), body.get("find", ""), body.get("replace", ""),
        whole_tag_only=bool(body.get("whole_tag_only", True)),
    )
    return jsonify({"changed": changed})


@bp.post("/bulk/underscores")
def bulk_underscores():
    body = request.get_json(force=True) or {}
    changed = captions.bulk_underscores_to_spaces(body.get("folder", ""))
    return jsonify({"changed": changed})


@bp.get("/tagger/models")
def tagger_models():
    models = [
        {
            "name": m,
            "downloaded": tagger.is_downloaded(m),
            "backend": tagger.backend_for(m),
            "deps_available": tagger.deps_available(m),
            "needs_deps": tagger.backend_for(m) != "onnx",
        }
        for m in tagger.available_models()
    ]
    return jsonify({"models": models, "default": tagger.DEFAULT_MODEL})


@bp.post("/tagger/install_deps")
def tagger_install_deps():
    try:
        return jsonify(dep_installer.start(TAGGER_TIMM_DEPS_JOB, tagger.TIMM_DEPS))
    except dep_installer.InstallError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/tagger/install_deps_status")
def tagger_install_deps_status():
    return jsonify(dep_installer.status(TAGGER_TIMM_DEPS_JOB))


@bp.post("/tagger/download")
def tagger_download():
    body = request.get_json(force=True) or {}
    model = body.get("model", tagger.DEFAULT_MODEL)
    if tagger.is_downloaded(model):
        return jsonify({"status": "done"})
    try:
        tagger.download_async(model)
    except tagger.TaggerError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "downloading"})


@bp.get("/tagger/download_status")
def tagger_download_status():
    model = request.args.get("model", tagger.DEFAULT_MODEL)
    return jsonify(tagger.download_progress(model))


@bp.post("/autotag")
def autotag():
    body = request.get_json(force=True) or {}
    folder = body.get("folder", "")
    model = body.get("model", tagger.DEFAULT_MODEL)
    general_threshold = float(body.get("general_threshold", 0.35))
    character_threshold = float(body.get("character_threshold", 0.75))
    mode = body.get("mode", "missing_only")  # missing_only | all
    trigger_word = (body.get("trigger_word") or "").strip()
    include_rating_tag = bool(body.get("include_rating_tag", False))

    if not tagger.is_downloaded(model):
        return jsonify({
            "error": f"The {model} tagger model hasn't been downloaded yet - "
                     "click Download in the Auto-tag card first."
        }), 400

    try:
        images = datasets.list_images(folder)
    except NotADirectoryError as exc:
        return jsonify({"error": str(exc)}), 400

    results = []
    for img in images:
        if mode == "missing_only" and img["caption"]:
            results.append({"path": img["path"], "status": "skipped", "reason": "already captioned"})
            continue
        try:
            tagged = tagger.tag_image_file(
                img["path"], model_name=model,
                general_threshold=general_threshold, character_threshold=character_threshold,
            )
        except tagger.TaggerError as exc:
            return jsonify({"error": str(exc)}), 502

        tags = tagged["character"] + tagged["general"]
        tags = [t.replace("_", " ") for t in tags]
        if include_rating_tag and tagged.get("rating"):
            tags = tags + [tagged["rating"].replace("_", " ")]
        if trigger_word:
            tags = [trigger_word] + tags
        caption = ", ".join(tags)
        captions.write_caption(img["path"], caption)
        results.append({"path": img["path"], "status": "ok", "caption": caption})

    tagged_count = sum(1 for r in results if r["status"] == "ok")
    return jsonify({"results": results, "tagged": tagged_count, "total": len(results)})


# ---------------------------------------------------------------------
# Caption rules, variation detection, templates (§4)
# ---------------------------------------------------------------------

@bp.post("/variations")
def detect_variations():
    """Scan a folder's captions for tag variations that may
    represent the same concept (white_hair vs silver_hair)."""
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    return jsonify({"variations": captions.detect_variations(captions.tags_in_folder(folder))})


@bp.post("/standardize")
def standardize_variations():
    """Rename variation tags to the suggested canonical form
    across the whole folder (whole-tag find/replace)."""
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    mapping = body.get("mapping") or {}
    if not isinstance(mapping, dict) or not mapping:
        return jsonify({"error": "mapping of from-tag -> to-tag is required"}), 400
    total = 0
    for from_tag, to_tag in mapping.items():
        if from_tag and to_tag:
            total += captions.bulk_find_replace(folder, from_tag, to_tag, whole_tag_only=True)
    return jsonify({"changed": total})


