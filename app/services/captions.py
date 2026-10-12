"""
Caption file operations for the tag editor: reading/writing individual
`.txt` captions, and bulk operations (find & replace, add tag, remove
tag, prepend trigger word, underscore -> space cleanup) across a whole
folder at once.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import config
from .imaging import is_backup_file, read_text_robust


def _is_dataset_image(f: Path) -> bool:
    return f.suffix.lower() in config.IMAGE_EXTENSIONS and not is_backup_file(f)


def _caption_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def read_caption(image_path: str) -> str:
    p = _caption_path(Path(image_path))
    if p.exists():
        return read_text_robust(p)
    return ""


def write_caption(image_path: str, caption: str) -> None:
    p = _caption_path(Path(image_path))
    p.write_text(caption.strip(), encoding="utf-8")


def _split(caption: str) -> list[str]:
    # Comma-separated is this app's own convention, but some datasets
    # tagged by other tools write one tag per line instead - without
    # this, every bulk operation (add/remove tag, find & replace) would
    # see a whole newline-separated caption as a single unmatched "tag"
    # and silently no-op on it.
    return [t.strip() for t in re.split(r"[,\n]+", caption) if t.strip()]


def _join(tags: list[str]) -> str:
    return ", ".join(tags)


def add_tag_to_images(image_paths: list[str], tag: str, position: str = "end") -> int:
    """Add `tag` to exactly the given images (not the whole folder) -
    used to undo a Tag Frequency "exclude" click by restoring the tag
    to precisely the images it was removed from."""
    tag = tag.strip()
    if not tag:
        return 0
    changed = 0
    for image_path in image_paths:
        caption_path = _caption_path(Path(image_path))
        current = _split(read_text_robust(caption_path)) if caption_path.exists() else []
        if tag in current:
            continue
        current = [tag] + current if position == "start" else current + [tag]
        caption_path.write_text(_join(current), encoding="utf-8")
        changed += 1
    return changed


def bulk_add_tag(folder: str, tag: str, position: str = "end") -> int:
    """Add `tag` to every caption in the folder that doesn't already have
    it. position: "start" | "end". Returns the number of files changed."""
    tag = tag.strip()
    if not tag:
        return 0
    changed = 0
    for f in Path(folder).iterdir():
        if not _is_dataset_image(f):
            continue
        caption_path = _caption_path(f)
        current = _split(read_text_robust(caption_path)) if caption_path.exists() else []
        if tag in current:
            continue
        current = [tag] + current if position == "start" else current + [tag]
        caption_path.write_text(_join(current), encoding="utf-8")
        changed += 1
    return changed


def bulk_remove_tag(folder: str, tag: str) -> int:
    tag = tag.strip()
    if not tag:
        return 0
    return bulk_remove_tags(folder, [tag])


def bulk_remove_tags(folder: str, tags: list[str]) -> int:
    """Remove any number of tags from every caption in one pass - used
    by the Tag Frequency view's multi-select "exclude" action so
    excluding N tags costs one folder scan instead of N."""
    tag_set = {t.strip() for t in tags if t and t.strip()}
    if not tag_set:
        return 0
    changed = 0
    for f in Path(folder).iterdir():
        if not _is_dataset_image(f):
            continue
        caption_path = _caption_path(f)
        if not caption_path.exists():
            continue
        current = _split(read_text_robust(caption_path))
        new_tags = [t for t in current if t not in tag_set]
        if len(new_tags) != len(current):
            caption_path.write_text(_join(new_tags), encoding="utf-8")
            changed += 1
    return changed


def bulk_find_replace(folder: str, find: str, replace: str, whole_tag_only: bool = True) -> int:
    """If whole_tag_only, only replaces a tag that matches `find` exactly
    (renaming a tag). Otherwise does a substring replace within each
    caption's raw text."""
    find = find.strip()
    if not find:
        return 0
    changed = 0
    for f in Path(folder).iterdir():
        if not _is_dataset_image(f):
            continue
        caption_path = _caption_path(f)
        if not caption_path.exists():
            continue
        original = read_text_robust(caption_path)
        if whole_tag_only:
            tags = _split(original)
            if find not in tags:
                continue
            new_tags = [replace.strip() if t == find else t for t in tags]
            new_tags = [t for t in new_tags if t]  # allow "replace" empty to delete
            new_text = _join(new_tags)
        else:
            if find not in original:
                continue
            new_text = original.replace(find, replace)
        if new_text != original:
            caption_path.write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def bulk_underscores_to_spaces(folder: str) -> int:
    changed = 0
    for f in Path(folder).iterdir():
        if not _is_dataset_image(f):
            continue
        caption_path = _caption_path(f)
        if not caption_path.exists():
            continue
        original = read_text_robust(caption_path)
        tags = _split(original)
        # keep emoticon-style tags with underscores intact, e.g. ">_<", "._.":
        new_tags = [t if t.replace("_", "") == "" else t.replace("_", " ") for t in tags]
        new_text = _join(new_tags)
        if new_text != original:
            caption_path.write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def normalize_tags(tags: list[str]) -> list[str]:
    """Standardize tag formatting: lowercase, underscores ->
    spaces, collapsed whitespace, deduplicated (order kept)."""
    seen: set[str] = set()
    out = []
    for tag in tags:
        normalized = " ".join(tag.strip().lower().replace("_", " ").split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def tags_in_folder(folder: str) -> list[str]:
    """Every tag used across a folder's captions (with
    duplicates - the caller decides how to count them)."""
    tags: list[str] = []
    for f in Path(folder).iterdir():
        if not _is_dataset_image(f):
            continue
        caption_path = _caption_path(f)
        if not caption_path.exists():
            continue
        tags.extend(_split(read_text_robust(caption_path)))
    return tags


# ---------------------------------------------------------------------
# Variation detection (§4): white_hair vs silver_hair etc.
# ---------------------------------------------------------------------

# Prefix-word synonym sets - pairs the tagger and booru
# taggers legitimately spell differently for the same
# concept (white_hair vs silver_hair, and so on).
_SYNONYMS = [
    frozenset({"white", "silver"}),
    frozenset({"black", "dark"}),
    frozenset({"blonde", "yellow"}),
    frozenset({"orange", "ginger"}),
    frozenset({"pink", "rose"}),
    frozenset({"purple", "violet"}),
    frozenset({"1st", "first"}),
    frozenset({"2nd", "second"}),
    frozenset({"3rd", "third"}),
]


def _prefix_variation(a: str, b: str) -> bool:
    """True when two prefix words mean the same concept:
    either they're string-similar or they're in the same
    synonym set."""
    import difflib  # noqa: PLC0415 - stdlib, cheap
    if difflib.SequenceMatcher(None, a, b).ratio() >= 0.72:
        return True
    pair = frozenset({a, b})
    return any(pair <= syn for syn in _SYNONYMS)


def detect_variations(tags: list[str],
                      similarity: float = 0.72) -> list[dict]:
    """Find tags that may represent the same concept.

    Tags are grouped by their last word (hair, eyes, ...),
    then compared pairwise inside each group: the differing
    prefix words are checked for string similarity or known
    synonyms, so white_hair/silver_hair surface as
    candidates to standardize while blue_eyes/green_eyes
    (genuinely different concepts) do not.
    """
    normalized = normalize_tags(tags)
    by_suffix: dict[str, list[str]] = {}
    for tag in normalized:
        parts = tag.split()
        suffix = parts[-1] if len(parts) > 1 else tag
        by_suffix.setdefault(suffix, []).append(tag)

    groups = []
    for suffix, members in by_suffix.items():
        if len(members) < 2:
            continue
        clusters: list[list[str]] = []
        for tag in members:
            placed = False
            for cluster in clusters:
                prefix = tag.rsplit(" ", 1)[0]
                for other in cluster:
                    other_prefix = other.rsplit(" ", 1)[0]
                    if _prefix_variation(prefix, other_prefix):
                        cluster.append(tag)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([tag])
        for cluster in clusters:
            if len(cluster) > 1:
                # suggest the shortest form (usually the canonical one)
                suggestion = sorted(cluster, key=lambda t: (len(t), t))[0]
                groups.append({
                    "tags": sorted(cluster),
                    "suggestion": suggestion,
                    "message": (
                        f"These tags may represent the same concept: "
                        f"{', '.join(cluster)}. Standardize to '{suggestion}'?"
                    ),
                })
    return groups
