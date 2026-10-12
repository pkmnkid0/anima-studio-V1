"""
Downloads selected search results into a dataset folder. Only the image
is saved: no `.txt` caption and no `.json` metadata file are written.
"""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .booru_engines import referer_for, open_image, get_proxy

USER_AGENT = "AnimaLoRAStudio/1.0 (dataset collection tool; +local use)"
REQUEST_TIMEOUT = 30
MAX_WORKERS = 6

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_stem(source: str, post_id, file_url: str) -> str:
    if post_id is not None:
        return f"{source}_{post_id}"
    # No post id (plain URL) - hash the URL for a stable, collision-free name.
    digest = hashlib.sha1(file_url.encode("utf-8")).hexdigest()[:16]
    return f"web_{digest}"


def _extension_from_url(url: str, fallback: str = "") -> str:
    ext = fallback.strip(".").lower()
    if not ext:
        path = url.split("?")[0]
        if "." in path.rsplit("/", 1)[-1]:
            ext = path.rsplit(".", 1)[-1].lower()
    if not ext or len(ext) > 5:
        ext = "jpg"
    return ext


def _download_one(item: dict, dest_dir: Path, overwrite: bool) -> dict:
    file_url = item["file_url"]
    ext = _extension_from_url(file_url, item.get("extension", ""))
    stem = _safe_stem(item.get("source", "web"), item.get("id"), file_url)
    image_path = dest_dir / f"{stem}.{ext}"

    if image_path.exists() and not overwrite:
        return {"file_url": file_url, "status": "skipped", "reason": "already exists", "path": str(image_path)}

    try:
        resp = open_image(file_url, referer_for(file_url), get_proxy(), timeout=(10, REQUEST_TIMEOUT))
        resp.raise_for_status()
        real = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif",
                "image/avif": "avif", "image/bmp": "bmp"}.get(resp.headers.get("Content-Type", "").split(";")[0].strip().lower())
        if real and real != ext and not (real == "jpg" and ext == "jpeg"):
            image_path = dest_dir / f"{stem}.{real}"       # derived URLs can guess the wrong extension
        with open(image_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
    except requests.exceptions.RequestException as exc:
        return {"file_url": file_url, "status": "error", "reason": str(exc)}

    return {"file_url": file_url, "status": "ok", "path": str(image_path)}


def download_batch(items: list[dict], dest_dir: str | Path, overwrite: bool = False) -> list[dict]:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_download_one, item, dest, overwrite): item
            for item in items
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def save_uploaded_files(files: list, dest_dir: str | Path) -> list[dict]:
    """Copy browser-uploaded files (werkzeug FileStorage objects)
    straight into a dataset folder, keeping the original name where
    possible and avoiding collisions."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    results = []
    for file_storage in files:
        original_name = file_storage.filename or "upload"
        safe_name = _SAFE_NAME_RE.sub("_", Path(original_name).name)
        if not safe_name or safe_name == "_":
            safe_name = "upload"
        target = dest / safe_name
        stem, ext = target.stem, target.suffix
        counter = 1
        while target.exists():
            target = dest / f"{stem}_{counter}{ext}"
            counter += 1
        try:
            file_storage.save(target)
            results.append({"filename": original_name, "status": "ok", "path": str(target)})
        except OSError as exc:
            results.append({"filename": original_name, "status": "error", "reason": str(exc)})
    return results
