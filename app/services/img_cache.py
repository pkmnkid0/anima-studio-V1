"""A small disk cache for images fetched through the booru proxy.

Thumbnails are re-requested every time you scroll or re-run a search; keeping
them on disk means the second look is served locally instead of hitting the
booru again. The cache is bounded - once it grows past MAX_BYTES the least
recently used files are dropped. A cache miss or a failed write is never an
error: the route just falls back to fetching from the site."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from .. import config

CACHE_DIR = config.BOORU_IMG_CACHE_DIR
MAX_BYTES = 256 * 1024 * 1024          # total size of the cache
MAX_ENTRY_BYTES = 8 * 1024 * 1024      # don't keep full-size originals bigger than this
_PRUNE_EVERY = 32 * 1024 * 1024        # only rescan the folder after this much new data

_prune_lock = threading.Lock()
_added_since_prune = 0


def _paths(url: str) -> tuple[Path, Path]:
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    body = CACHE_DIR / key[:2] / key
    return body, body.with_suffix(".json")


def get(url: str) -> tuple[bytes, str] | None:
    """(body, content-type) for this URL if it is cached, else None."""
    body, meta = _paths(url)
    try:
        ctype = json.loads(meta.read_text(encoding="utf-8")).get("ctype") or "application/octet-stream"
        data = body.read_bytes()
    except (OSError, ValueError):
        return None
    try:
        os.utime(body, None)          # reading refreshes its LRU position
    except OSError:
        pass
    return data, ctype


def put(url: str, data: bytes, ctype: str) -> None:
    global _added_since_prune
    body, meta = _paths(url)
    try:
        body.parent.mkdir(parents=True, exist_ok=True)
        # unique per write: concurrent saves of one URL must not share a
        # .part file and interleave their bytes
        tmp = body.with_name(f"{body.name}.{os.getpid()}_{time.time_ns()}.part")
        tmp.write_bytes(data)
        os.replace(tmp, body)
        meta.write_text(json.dumps({"ctype": ctype, "ts": time.time()}), encoding="utf-8")
    except OSError:
        return                        # a cache write must never break the response
    _added_since_prune += len(data)
    if _added_since_prune >= _PRUNE_EVERY:
        _added_since_prune = 0
        _prune()


def _prune() -> None:
    """Drop the least recently used files until the cache is back under budget."""
    with _prune_lock:
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for f in CACHE_DIR.rglob("*"):
            if f.is_file():
                try:
                    st = f.stat()
                except OSError:
                    continue
                entries.append((st.st_mtime, st.st_size, f))
                total += st.st_size
        if total <= MAX_BYTES:
            return
        entries.sort()                # oldest use first
        for _mtime, size, f in entries:
            if total <= MAX_BYTES:
                break
            try:
                f.unlink()
                total -= size
            except OSError:
                continue
