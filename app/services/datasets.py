"""
Filesystem-facing helpers for the Caption and Train tabs: browsing
folders (this app runs entirely on the user's own machine, so a simple
server-side directory browser stands in for a native "choose folder"
dialog), listing a folder's images alongside their caption/tag-sidecar
state, and basic tag-frequency stats used by the tag editor.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .. import config
from .imaging import is_backup_file, read_text_robust


def list_dir(path: str | None, file_extensions: tuple[str, ...] | None = None) -> dict:
    """List subdirectories of `path` (or the datasets root if omitted),
    for the in-app folder browser. If `file_extensions` is given, also
    lists matching files in that folder (for the file-picker variant used
    to load things like a .toml config straight off disk)."""
    if not path:
        base = config.DATASETS_DIR
    else:
        base = Path(path)

    if not base.exists() or not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    try:
        base = base.resolve()
    except OSError:
        pass
    entries = []
    files = []
    problem = None
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:  # permission denied, cloud placeholder, network drive gone...
        children = []
        problem = f"Can't read this folder ({exc.strerror or exc})."
    # Each child is handled on its own: one protected or broken entry (e.g. a
    # Windows junction like "Application Data") must never hide the rest.
    for child in children:
        try:
            if child.name.startswith("."):
                continue
            if child.is_dir():
                entries.append({"name": child.name, "path": str(child), "image_count": _count_images(child)})
            elif file_extensions and child.is_file() and child.suffix.lower() in file_extensions:
                files.append({"name": child.name, "path": str(child), "modified": child.stat().st_mtime})
        except OSError:
            continue

    parent = str(base.parent) if base.parent != base else None
    roots = _filesystem_roots()

    result = {"path": str(base), "parent": parent, "dirs": entries, "roots": roots}
    if problem:
        result["warning"] = problem
    if file_extensions:
        result["files"] = files
    return result


def _count_images(folder: Path, scan_cap: int = 4000) -> int | None:
    """Image count for the folder list. Returns None if the folder can't be
    read, and stops after scan_cap entries so huge folders (or slow network
    drives) don't make the browser hang."""
    count = scanned = 0
    try:
        with os.scandir(folder) as it:
            for entry in it:
                scanned += 1
                if scanned > scan_cap:
                    break
                try:
                    if entry.is_file() and Path(entry.name).suffix.lower() in config.IMAGE_EXTENSIONS \
                            and not is_backup_file(Path(entry.path)):
                        count += 1
                except OSError:
                    continue
    except OSError:
        return None
    return count


def _filesystem_roots() -> list[dict]:
    """Quick-access shortcuts for the folder browser modal - your home
    directory plus every drive letter (Windows) or "/" (Linux/Mac) - so
    picking a folder that lives outside the app's own datasets folder
    (a different drive, or anywhere else on your PC) doesn't mean
    clicking ".." dozens of times or getting stuck with no way to leave
    the current drive at all."""
    roots = [{"label": "Home", "path": str(Path.home())}]
    if os.name == "nt":
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            try:
                if Path(drive).exists():
                    roots.append({"label": drive, "path": drive})
            except OSError:
                continue
    else:
        roots.append({"label": "/", "path": "/"})
    return roots


def create_dir(path: str) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


def list_subsets() -> list[dict]:
    """Every folder the Caption/Edit tab's subset tabs should offer:
    every folder directly under the datasets root (as before), plus any
    folder anywhere on disk the user has explicitly registered via
    "Add subset from my PC" - so datasets that live outside the app's
    own datasets folder (e.g. an existing dataset from another tool)
    show up too, not just ones copied into DATASETS_DIR."""
    from . import settings_store

    seen: dict[str, dict] = {}
    try:
        for d in list_dir(None).get("dirs", []):
            key = str(Path(d["path"]).resolve())
            seen[key] = {"name": d["name"], "path": d["path"], "image_count": d["image_count"], "external": False}
    except (NotADirectoryError, OSError):
        pass

    for raw_path in settings_store.get("registered_subsets", []) or []:
        p = Path(raw_path)
        if not p.exists() or not p.is_dir():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        image_count = sum(
            1 for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS and not is_backup_file(f)
        )
        seen[key] = {"name": p.name, "path": str(p), "image_count": image_count, "external": True}

    return sorted(seen.values(), key=lambda d: d["name"].lower())


def register_subset(path: str) -> list[dict]:
    """Add an arbitrary folder from anywhere on disk to the subset list -
    used by "Add subset from my PC" so already-organized datasets don't
    need to be copied into the datasets root just to get a quick-switch
    tab for them."""
    from . import settings_store

    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    resolved = str(p.resolve())
    current = settings_store.get("registered_subsets", []) or []
    if resolved not in current:
        current.append(resolved)
        settings_store.set("registered_subsets", current)
    return list_subsets()


def unregister_subset(path: str) -> list[dict]:
    """Removes a folder from the registered-subsets list only - never
    touches anything on disk, so this is safe to offer as a plain
    "remove tab" action."""
    from . import settings_store

    p = Path(path)
    resolved = str(p.resolve()) if p.exists() else path
    current = [x for x in (settings_store.get("registered_subsets", []) or []) if x not in (path, resolved)]
    settings_store.set("registered_subsets", current)
    return list_subsets()


def list_images(folder: str) -> list[dict]:
    """Every image in `folder` with its current caption text (if any)
    and whether a booru tag sidecar exists."""
    base = Path(folder)
    if not base.exists() or not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    out = []
    for f in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        if not f.is_file() or f.suffix.lower() not in config.IMAGE_EXTENSIONS:
            continue
        if is_backup_file(f):
            continue
        caption_path = f.with_suffix(".txt")
        alt_caption_path = f.with_suffix(".caption")
        caption = ""
        if caption_path.exists():
            caption = read_text_robust(caption_path)
        elif alt_caption_path.exists():
            caption = read_text_robust(alt_caption_path)

        out.append({
            "filename": f.name,
            "path": str(f),
            "caption": caption,
        })
    return out


def tag_frequency(folder: str, with_examples: bool = False) -> list[dict]:
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for img in list_images(folder):
        # Split on commas (this app's own convention) or newlines - some
        # datasets tagged by other tools write one tag per line instead
        # of comma-separating them, which used to come through as a
        # single giant "tag" here and made per-tag stats/filtering on an
        # imported dataset look broken.
        for tag in [t.strip() for t in re.split(r"[,\n]+", img["caption"]) if t.strip()]:
            counts[tag] = counts.get(tag, 0) + 1
            if with_examples and tag not in examples:
                examples[tag] = img["path"]
    results = sorted(
        ({"tag": tag, "count": count} for tag, count in counts.items()),
        key=lambda x: (-x["count"], x["tag"]),
    )
    if with_examples:
        for r in results:
            r["example"] = examples.get(r["tag"])
    return results


def caption_stats(folder: str) -> dict:
    images = list_images(folder)
    captioned = sum(1 for i in images if i["caption"])
    return {
        "total": len(images),
        "captioned": captioned,
        "uncaptioned": len(images) - captioned,
        "unique_tags": len(tag_frequency(folder)),
    }


def delete_images(image_paths: list[str]) -> list[dict]:
    """Delete images and everything that goes with them (caption,
    booru-tags sidecar, magic-eraser backup) - used when discarding
    duplicates found by the Edit tab's scan."""
    from .imaging import BACKUP_SUFFIX

    results = []
    for image_path in image_paths:
        p = Path(image_path)
        try:
            for related in (
                p,
                p.with_suffix(".txt"),
                p.with_suffix(".caption"),
                p.with_name(f"{p.stem}.tags.json"),
                p.with_name(f"{p.stem}{BACKUP_SUFFIX}{p.suffix}"),
            ):
                if related.exists():
                    related.unlink()
            results.append({"path": image_path, "status": "ok"})
        except OSError as exc:
            results.append({"path": image_path, "status": "error", "reason": str(exc)})
    return results
