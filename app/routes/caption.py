from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from ..services import caption_env, captioner, captions, datasets, dep_installer, tag_tools, tagger

bp = Blueprint("caption", __name__)

TAGGER_TIMM_DEPS_JOB = "tagger_timm_deps"
CAPTIONER_DEPS_JOB = "captioner_deps"


def _snap(folder: str, label: str, image_paths=None) -> None:
    """Remember the captions first so the change can be undone."""
    try:
        tag_tools.snapshot(folder, label, image_paths)
    except Exception:   # noqa: BLE001 - undo is best-effort, never block the edit
        pass


@bp.get("/hf_token")
def hf_token_get():
    return jsonify({"has_token": bool(captioner.hf_token())})


@bp.post("/hf_token")
def hf_token_set():
    from ..services import settings_store
    tok = str((request.get_json(force=True) or {}).get("token") or "").strip()
    if tok and (len(tok) < 8 or any(c.isspace() for c in tok)):
        return jsonify({"error": "That does not look like a Hugging Face token (they start with hf_)."}), 400
    settings_store.set("hf_token", tok)
    return jsonify({"has_token": bool(tok)})


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
    _snap(body.get("folder", ""), "Add tag to all")
    changed = captions.bulk_add_tag(body.get("folder", ""), body.get("tag", ""), body.get("position", "end"))
    return jsonify({"changed": changed})


@bp.post("/bulk/remove_tag")
def bulk_remove_tag():
    body = request.get_json(force=True) or {}
    _snap(body.get("folder", ""), "Remove tag from all")
    changed = captions.bulk_remove_tag(body.get("folder", ""), body.get("tag", ""))
    return jsonify({"changed": changed})


@bp.post("/bulk/remove_tags")
def bulk_remove_tags():
    body = request.get_json(force=True) or {}
    _snap(body.get("folder", ""), "Remove tags from all")
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
    _snap(body.get("folder", ""), "Find & replace")
    changed = captions.bulk_find_replace(
        body.get("folder", ""), body.get("find", ""), body.get("replace", ""),
        whole_tag_only=bool(body.get("whole_tag_only", True)),
    )
    return jsonify({"changed": changed})


@bp.post("/bulk/underscores")
def bulk_underscores():
    body = request.get_json(force=True) or {}
    _snap(body.get("folder", ""), "Underscores to spaces")
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


# --- Natural-language captions ------------------------------------------------
# Anima's text encoder is an LLM, so a caption that reads like a sentence carries
# more than a tag list. Two sources: rephrasing tags that already exist (instant,
# offline), or a local vision-language model describing the image (downloadable).

@bp.get("/nl/models")
def nl_models():
    return jsonify({
        "models": [captioner.model_summary(name) for name in captioner.MODELS],
        "default": captioner.DEFAULT_MODEL,
        "vram_profiles": captioner.VRAM_PROFILES,
    })


@bp.post("/nl/download")
def nl_download():
    body = request.get_json(force=True) or {}
    model = body.get("model", captioner.DEFAULT_MODEL)
    if captioner.is_downloaded(model):
        return jsonify({"status": "done"})
    try:
        captioner.download_async(model)
    except captioner.CaptionerError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "downloading"})


@bp.get("/nl/download_status")
def nl_download_status():
    model = request.args.get("model", captioner.DEFAULT_MODEL)
    progress = captioner.download_progress(model)
    progress["downloaded"] = captioner.is_downloaded(model)
    return jsonify(progress)


@bp.get("/nl/deps")
def nl_deps():
    """Which interpreter will run the model, and whether it can."""
    return jsonify(captioner.deps_status(force=request.args.get("force") == "1", model=request.args.get("model") or None))


@bp.get("/nl/env/status")
def nl_env_status():
    return jsonify(caption_env.status())


@bp.post("/nl/env/start")
def nl_env_start():
    try:
        return jsonify(caption_env.start())
    except caption_env.SetupError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/nl/env/cancel")
def nl_env_cancel():
    return jsonify(caption_env.cancel())


@bp.post("/nl/install_deps")
def nl_install_deps():
    python, _label = captioner.interpreter()
    try:
        return jsonify(dep_installer.start(CAPTIONER_DEPS_JOB, captioner.CAPTION_PIP, python=python))
    except dep_installer.InstallError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/nl/install_deps_status")
def nl_install_deps_status():
    return jsonify(dep_installer.status(CAPTIONER_DEPS_JOB))


@bp.post("/nl/preview")
def nl_preview():
    """Rephrase one caption without touching the folder, so the style can be judged first."""
    body = request.get_json(force=True) or {}
    text = (body.get("caption") or "").strip()
    if not text and body.get("image_path"):
        try:
            text = captions.read_caption(body["image_path"]).strip()
        except OSError:
            text = ""
    if not text:
        return jsonify({"error": "Nothing to rephrase - pick an image that already has a caption."}), 400
    return jsonify({
        "before": text,
        "after": captioner.preview_caption(
            text, trigger_word=(body.get("trigger_word") or "").strip(),
            drop_quality=bool(body.get("drop_quality", True)),
            write=body.get("write", "replace")),
    })


@bp.post("/nl/run")
def nl_run():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(captioner.run(body))
    except captioner.CaptionerError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/nl/run_status")
def nl_run_status():
    return jsonify(captioner.run_status())


@bp.post("/nl/cancel")
def nl_cancel():
    return jsonify(captioner.cancel())


def _threshold(body: dict, key: str, default: float) -> float:
    """A cleared threshold box sends null (parseFloat("") is NaN) - fall
    back to the default instead of float(None) raising a 500."""
    raw = body.get(key, default)
    if raw is None or raw == "":
        return default
    return float(raw)


@bp.post("/autotag")
def autotag():
    body = request.get_json(force=True) or {}
    folder = body.get("folder", "")
    model = body.get("model", tagger.DEFAULT_MODEL)
    try:
        general_threshold = _threshold(body, "general_threshold", 0.35)
        character_threshold = _threshold(body, "character_threshold", 0.75)
    except (TypeError, ValueError):
        return jsonify({"error": "Thresholds must be numbers."}), 400
    mode = body.get("mode", "missing_only")  # missing_only | all
    trigger_word = (body.get("trigger_word") or "").strip()
    include_rating_tag = bool(body.get("include_rating_tag", False))
    use_colour_guard = bool(body.get("colour_guard", False))
    bl = tag_tools.get_blacklist()
    blacklist = tag_tools.blacklist_set() if bl["apply_after_tagging"] else set()

    if not tagger.is_downloaded(model):
        return jsonify({
            "error": f"The {model} tagger model hasn't been downloaded yet - "
                     "click Download in the Auto-tag card first."
        }), 400

    try:
        images = datasets.list_images(folder)
    except NotADirectoryError as exc:
        return jsonify({"error": str(exc)}), 400

    _snap(folder, "Auto-tag")
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
        except (tagger.TaggerError, OSError) as exc:
            # one corrupt/undecodable image must not lose the whole batch
            results.append({"path": img["path"], "status": "error", "reason": str(exc)})
            continue

        tags = tagged["character"] + tagged["general"]
        if use_colour_guard and tagged.get("general_scored"):
            scored = tag_tools.colour_guard(tagged["general_scored"])
            keep = {n for n, _ in scored}
            tags = [t for t in tags if t in keep or t in tagged["character"]]
        tags = [t.replace("_", " ") for t in tags]
        if blacklist:
            tags = [t for t in tags if tag_tools._norm(t) not in blacklist]
        if include_rating_tag and tagged.get("rating"):
            tags = tags + [tagged["rating"].replace("_", " ")]
        if trigger_word:
            tags = [trigger_word] + tags
        caption = ", ".join(tags)
        captions.write_caption(img["path"], caption)
        results.append({"path": img["path"], "status": "ok", "caption": caption})

    tagged_count = sum(1 for r in results if r["status"] == "ok")
    return jsonify({"results": results, "tagged": tagged_count, "total": len(results)})


@bp.post("/region_tags")
def region_tags():
    """Tag only a selected rectangle of one image (for details the tagger misses)."""
    body = request.get_json(force=True) or {}
    model = body.get("model", tagger.DEFAULT_MODEL)
    path = str(body.get("image_path", ""))
    if not path or not Path(path).is_file():
        return jsonify({"error": "Image not found."}), 400
    if not tagger.is_downloaded(model):
        return jsonify({"error": f"The {model} tagger model hasn't been downloaded yet - "
                                 "download it in the Auto-tag card first."}), 400
    try:
        tags = tagger.tag_region(path, body.get("box") or {}, model, int(body.get("top", 40)), str(body.get("query", "")))
    except (tagger.TaggerError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"tags": tags})


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
    _snap(folder, "Standardize variations")
    mapping = body.get("mapping") or {}
    if not isinstance(mapping, dict) or not mapping:
        return jsonify({"error": "mapping of from-tag -> to-tag is required"}), 400
    total = 0
    for from_tag, to_tag in mapping.items():
        if from_tag and to_tag:
            total += captions.bulk_find_replace(folder, from_tag, to_tag, whole_tag_only=True)
    return jsonify({"changed": total})


# ---------------------------------------------------------------------
# Undo, multi-select, blacklist, pruning, sort, co-occurrence
# ---------------------------------------------------------------------

def _folder_error(exc: Exception):
    return jsonify({"error": str(exc)}), 400


@bp.get("/undo/status")
def undo_status():
    return jsonify(tag_tools.undo_status(request.args.get("folder", "")))


@bp.post("/undo")
def undo():
    body = request.get_json(force=True) or {}
    return jsonify(tag_tools.undo(str(body.get("folder", ""))))


@bp.post("/bulk/images")
def bulk_images():
    """Add/remove tags on exactly the selected images."""
    body = request.get_json(force=True) or {}
    paths = [str(p) for p in body.get("image_paths", [])]
    add, remove = body.get("add", []), body.get("remove", [])
    label = "Edit selected images"
    _snap(str(body.get("folder", "")), label, paths)
    changed = tag_tools.edit_images(paths, add, remove, body.get("position", "end"))
    return jsonify({"changed": changed})


@bp.get("/blacklist")
def get_blacklist():
    return jsonify(tag_tools.get_blacklist())


@bp.put("/blacklist")
def put_blacklist():
    body = request.get_json(force=True) or {}
    return jsonify(tag_tools.set_blacklist(body.get("tags", []), bool(body.get("apply_after_tagging", True))))


@bp.post("/bulk/apply_blacklist")
def apply_blacklist():
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    try:
        _snap(folder, "Apply blacklist")
        return jsonify({"changed": tag_tools.apply_blacklist(folder)})
    except NotADirectoryError as exc:
        return _folder_error(exc)


@bp.post("/prune/scan")
def prune_scan():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(tag_tools.scan_prune(str(body.get("folder", ""))))
    except NotADirectoryError as exc:
        return _folder_error(exc)


@bp.post("/prune/apply")
def prune_apply():
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    pairs = body.get("pairs") or []
    if not pairs:
        return jsonify({"error": "Tick at least one pair first."}), 400
    try:
        _snap(folder, "Prune redundant tags")
        return jsonify({"changed": tag_tools.apply_prune(folder, pairs)})
    except NotADirectoryError as exc:
        return _folder_error(exc)


@bp.post("/bulk/dedupe")
def bulk_dedupe():
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    try:
        _snap(folder, "Remove repeated tags")
        return jsonify({"changed": tag_tools.dedupe_captions(folder)})
    except NotADirectoryError as exc:
        return _folder_error(exc)


@bp.post("/bulk/auto_sort")
def bulk_auto_sort():
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    try:
        _snap(folder, "Auto-sort tags")
        changed = tag_tools.auto_sort(folder, int(body.get("keep_first", 1)), body.get("pinned") or [])
        return jsonify({"changed": changed})
    except NotADirectoryError as exc:
        return _folder_error(exc)


@bp.post("/cooccurrence")
def cooccurrence():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(tag_tools.cooccurrence(
            str(body.get("folder", "")), str(body.get("tag", "")),
            int(body.get("min_count", 3)), int(body.get("top", 40)),
        ))
    except NotADirectoryError as exc:
        return _folder_error(exc)
