from __future__ import annotations

from flask import Blueprint, jsonify, request

from .. import config
from ..services import booru, downloader, settings_store, web_images

bp = Blueprint("collect", __name__)


@bp.get("/sources")
def sources():
    return jsonify({"sources": booru.SOURCES, "ratings": ["safe", "sensitive", "questionable", "explicit", "any"]})


@bp.get("/credentials")
def credentials():
    """Which sources need an api_key/user_id and whether one's saved -
    Gelbooru and Rule34 both started requiring these in Aug 2025."""
    return jsonify({"sources": booru.credentials_status()})


@bp.post("/credentials")
def save_credentials():
    body = request.get_json(force=True) or {}
    source = body.get("source", "")
    if source not in booru.SOURCES_REQUIRING_CREDENTIALS:
        return jsonify({"error": f"{source} doesn't take API credentials."}), 400
    settings_store.set_booru_credentials(source, body.get("api_key", ""), body.get("user_id", ""))
    return jsonify({"sources": booru.credentials_status()})


@bp.get("/autocomplete")
def autocomplete():
    source = request.args.get("source", "danbooru")
    query = request.args.get("query", "")
    limit = int(request.args.get("limit", 10))
    results = booru.autocomplete(source, query, limit=limit)
    return jsonify({"results": results})


@bp.post("/tag_options")
def tag_options():
    body = request.get_json(force=True) or {}
    source = body.get("source", "danbooru")
    tags = (body.get("tags") or "").strip()
    rating = body.get("rating", "safe")
    try:
        data = booru.tag_options(source, tags, rating=rating, sample_pages=body.get("sample_pages", 3), per_page=body.get("per_page", 100))
    except booru.CredentialsRequiredError as exc:
        return jsonify({"error": str(exc), "needs_credentials": exc.source}), 401
    except booru.BooruError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify(data)


@bp.post("/smart_dataset")
def smart_dataset():
    body = request.get_json(force=True) or {}
    source = body.get("source", "danbooru")
    tags = (body.get("tags") or "").strip()
    rating = body.get("rating", "safe")
    target = int(body.get("target", 100))
    pool = int(body.get("pool", 5000))
    min_resolution = int(body.get("min_resolution", 512))
    if not tags:
        return jsonify({"error": "Enter a character tag first."}), 400
    try:
        any_tags = [str(t) for t in (body.get("any_tags") or []) if str(t).strip()]
        data = booru.smart_dataset(source, tags, rating=rating, target=target, pool=pool,
                                   min_resolution=min_resolution, any_tags=any_tags)
    except booru.CredentialsRequiredError as exc:
        return jsonify({"error": str(exc), "needs_credentials": exc.source}), 401
    except booru.BooruError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify(data)


@bp.post("/search")
def search():
    body = request.get_json(force=True) or {}
    source = body.get("source", "danbooru")
    tags = body.get("tags", "")
    rating = body.get("rating", "safe")
    limit = int(body.get("limit", 40))
    page = int(body.get("page", 1))
    try:
        results = booru.search(source, tags, rating=rating, limit=limit, page=page)
    except booru.CredentialsRequiredError as exc:
        return jsonify({"error": str(exc), "needs_credentials": exc.source}), 401
    except booru.BooruError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"results": results, "count": len(results)})


@bp.post("/scan_page")
def scan_page():
    body = request.get_json(force=True) or {}
    url = body.get("url", "")
    same_domain_only = bool(body.get("same_domain_only", True))
    if not url:
        return jsonify({"error": "url is required"}), 400
    try:
        results = web_images.scan_page(url, same_domain_only=same_domain_only)
    except web_images.WebScrapeError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"results": results, "count": len(results)})


@bp.post("/from_urls")
def from_urls():
    body = request.get_json(force=True) or {}
    urls = body.get("urls", [])
    if isinstance(urls, str):
        urls = urls.splitlines()
    results = web_images.from_url_list(urls)
    return jsonify({"results": results, "count": len(results)})


@bp.post("/download")
def download():
    body = request.get_json(force=True) or {}
    items = body.get("items", [])
    dest_dir = body.get("dest_dir") or str(config.DATASETS_DIR / "untitled")
    overwrite = bool(body.get("overwrite", False))

    if not items:
        return jsonify({"error": "No items to download"}), 400

    results = downloader.download_batch(items, dest_dir, overwrite=overwrite)   # images only
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errored = sum(1 for r in results if r["status"] == "error")
    return jsonify({
        "results": results, "dest_dir": dest_dir,
        "summary": {"ok": ok, "skipped": skipped, "errored": errored, "total": len(results)},
    })


@bp.post("/upload_local")
def upload_local():
    """Add images straight from the user's own computer into a dataset
    folder - no search involved, just a direct copy-in."""
    dest_dir = request.form.get("dest_dir") or str(config.DATASETS_DIR / "untitled")
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received"}), 400

    saved = downloader.save_uploaded_files(files, dest_dir)
    ok = sum(1 for r in saved if r["status"] == "ok")
    errored = sum(1 for r in saved if r["status"] == "error")
    return jsonify({
        "results": saved, "dest_dir": dest_dir,
        "summary": {"ok": ok, "skipped": 0, "errored": errored, "total": len(saved)},
    })
