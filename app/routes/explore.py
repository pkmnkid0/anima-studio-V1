"""Booru Explorer API: feed/search, post details, tag wiki, related tags,
autocomplete, pools, and an allow-listed image proxy for thumbnails."""
from __future__ import annotations

import requests
from flask import Blueprint, Response, jsonify, request

from app.services import booru_engines as eng
from app.services import netbypass

bp = Blueprint("explore", __name__)


def _err(exc: Exception, code: int = 502):
    return jsonify({"error": str(exc)}), code


@bp.get("/sources")
def sources():
    return jsonify({"sources": eng.list_sources(), "orders": list(eng.ORDERS),
                    "ratings": ["safe", "sensitive", "questionable", "explicit", "any"]})


@bp.get("/search")
def search():
    a = request.args
    try:
        res = eng.explore(
            source=a.get("source", "danbooru"), tags=a.get("tags", ""), rating=a.get("rating", "any"),
            order=a.get("order", "random"), limit=a.get("limit", 40, type=int),
            page=a.get("page", 1, type=int), min_score=a.get("min_score", 0, type=int))
        return jsonify(res)
    except eng.EngineError as exc:
        return _err(exc)


@bp.get("/post/<int:post_id>")
def post_details(post_id: int):
    try:
        return jsonify({"post": eng.get_post(request.args.get("source", "danbooru"), post_id)})
    except eng.EngineError as exc:
        return _err(exc)


@bp.get("/tag/wiki")
def tag_wiki():
    try:
        return jsonify(eng.tag_wiki(request.args.get("source", "danbooru"), request.args.get("tag", "")))
    except eng.EngineError as exc:
        return _err(exc)


@bp.get("/tag/related")
def tag_related():
    try:
        return jsonify({"related": eng.related_tags(request.args.get("source", "danbooru"), request.args.get("tag", ""))})
    except eng.EngineError as exc:
        return _err(exc)


@bp.get("/autocomplete")
def autocomplete():
    try:
        res = eng.autocomplete(request.args.get("source", "danbooru"), request.args.get("query", ""),
                               request.args.get("limit", 10, type=int))
        return jsonify({"results": res})
    except eng.EngineError as exc:
        return _err(exc)


@bp.get("/pools")
def pools():
    try:
        res = eng.pools(request.args.get("source", "danbooru"), request.args.get("query", ""),
                        request.args.get("page", 1, type=int))
        return jsonify({"pools": res})
    except eng.EngineError as exc:
        return _err(exc)


@bp.get("/net")
def net_get():
    return jsonify({"mode": netbypass.get_mode(), "proxy": eng.get_proxy(),
                    "providers": [p[0] for p in netbypass.PROVIDERS]})


@bp.post("/net")
def net_set():
    body = request.get_json(silent=True) or {}
    try:
        if "mode" in body:
            netbypass.set_mode(body["mode"])
        if "proxy" in body:
            eng.set_proxy(body["proxy"])
    except (ValueError, eng.EngineError) as exc:
        return _err(exc, 400)
    return jsonify({"mode": netbypass.get_mode(), "proxy": eng.get_proxy()})


@bp.get("/net/test")
def net_test():
    """How each way of finding a site's address does from this computer."""
    ids = [x for x in request.args.get("sources", "danbooru,gelbooru,konachan,e621").split(",") if x in eng.SOURCES]
    hosts = list(dict.fromkeys(__import__("urllib.parse").parse.urlparse(eng.SOURCES[i]["base"]).hostname for i in ids))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(hosts) or 1) as pool:
        results = list(pool.map(netbypass.test_host, hosts))
    return jsonify({"mode": netbypass.get_mode(), "results": results})


@bp.get("/diagnose")
def diagnose():
    ids = [x for x in request.args.get("sources", "").split(",") if x in eng.SOURCES] or None
    return jsonify({"results": eng.probe_sources(ids), "proxy": eng.get_proxy()})


@bp.post("/proxy")
def set_proxy():
    try:
        return jsonify({"proxy": eng.set_proxy((request.get_json(silent=True) or {}).get("proxy", ""))})
    except eng.EngineError as exc:
        return _err(exc, 400)


@bp.get("/img")
def img():
    """Stream an image from a known booru host with the right Referer, so
    hotlink-protected thumbnails still show up."""
    url = request.args.get("url", "")
    ok, referer = eng.image_proxy_ok(url)
    if not ok:
        return jsonify({"error": "Host not allowed"}), 400
    from urllib.parse import urlparse as _up
    eng.MEDIA_HOSTS.add((_up(url).hostname or "").lower())     # lets secure-DNS cover new CDN hosts too
    try:
        upstream = eng.open_image(url, referer or url, eng.get_proxy(), timeout=(8, 25))
    except requests.exceptions.RequestException as exc:
        return _err(exc)
    if upstream.status_code != 200:
        upstream.close()
        return Response(status=upstream.status_code)
    ctype = upstream.headers.get("Content-Type", "application/octet-stream")
    ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
    if ctype.startswith("application/octet-stream") or ctype.startswith("binary/"):
        guessed = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif",
                   "webp": "image/webp", "avif": "image/avif", "bmp": "image/bmp",
                   "mp4": "video/mp4", "webm": "video/webm"}.get(ext)
        if guessed:
            ctype = guessed
    if not (ctype.startswith("image/") or ctype.startswith("video/")):
        upstream.close()
        return jsonify({"error": "Not an image"}), 415
    resp = Response(upstream.iter_content(chunk_size=1 << 15), content_type=ctype)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    # Content-Length of a compressed body doesn't match what iter_content yields -> truncated images
    if upstream.headers.get("Content-Length") and not upstream.headers.get("Content-Encoding"):
        resp.headers["Content-Length"] = upstream.headers["Content-Length"]
    return resp


@bp.get("/img/test")
def img_test():
    """Why won't this image load? Used by the Explorer's 'images not loading' notice."""
    url = request.args.get("url", "")
    ok, referer = eng.image_proxy_ok(url)
    if not ok:
        return jsonify({"ok": False, "message": "That address isn't allowed."})
    from urllib.parse import urlparse as _up
    host = (_up(url).hostname or "").lower()
    eng.MEDIA_HOSTS.add(host)
    try:
        r = eng.open_image(url, referer or url, eng.get_proxy(), timeout=(6, 10))
        code, ctype = r.status_code, r.headers.get("Content-Type", "")
        r.close()
        if code == 200 and (ctype.startswith("image/") or ctype.startswith("video/") or ctype.startswith("application/octet")):
            return jsonify({"ok": True, "host": host, "message": "reachable"})
        return jsonify({"ok": False, "host": host, "status": code, "message":
                        f"{host} answered HTTP {code} ({ctype or 'no content type'}) - the site is refusing image requests from here."})
    except requests.exceptions.RequestException as exc:
        f = eng._friendly(exc, host)
        return jsonify({"ok": False, "host": host, "message": f"Can't reach {host}: {f}"})
