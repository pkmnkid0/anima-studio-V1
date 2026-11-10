"""Smart dataset analysis (health score, balance, cleanup).

Everything here is read-only except apply_cleanup(), which
only ever removes images the user explicitly confirmed -
"Never automatically delete without user confirmation."
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from .. import config
from . import imaging

MIN_IMAGE_SIDE = 256          # "tiny image" threshold
EXTREME_ASPECT = 3.0          # > 3:1 or < 1:3
NEAR_DUP_THRESHOLD = 6        # dhash hamming distance


class AnalysisError(RuntimeError):
    pass


def _image_files(folder: str) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise AnalysisError(f"Not a folder: {folder}")
    return sorted(
        (p for p in root.iterdir()
         if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
         and not imaging.is_backup_file(p)),
        key=lambda p: p.name,
    )


def _probe_image(path: Path) -> dict:
    """Open one image and report its stats; corrupt files
    surface as {"corrupt": reason} instead of raising."""
    info = {"path": str(path), "name": path.name,
            "bytes": path.stat().st_size, "corrupt": None}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            info["width"], info["height"] = w, h
            info["aspect"] = round(w / h, 3) if h else 0.0
            info["megapixels"] = round(w * h / 1e6, 2)
    except Exception as exc:  # noqa: BLE001 - any PIL failure is "corrupt"
        info["corrupt"] = f"{type(exc).__name__}: {exc}"[:200]
    return info


def analyze_dataset(folder: str, deep: bool = True) -> dict:
    """Full analysis of a dataset folder.

    deep=True also computes near-duplicate groups (dhash) -
    it reads every image fully, so it's slower; the UI
    offers "quick" (structure only) and "full" modes.
    """
    files = _image_files(folder)
    images = [_probe_image(p) for p in files]

    res_buckets = Counter()
    aspect_buckets = Counter()
    corrupt, tiny, extreme = [], [], []
    exact_hash: dict[str, list[str]] = {}

    for info in images:
        if info["corrupt"]:
            corrupt.append(info)
            continue
        res_buckets[f'{info["width"]}x{info["height"]}'] += 1
        a = info["aspect"]
        bucket = ("wide" if a > 1.25 else
                  "tall" if a < 0.8 else "square")
        aspect_buckets[bucket] += 1
        if min(info["width"], info["height"]) < MIN_IMAGE_SIDE:
            tiny.append(info)
        if a > EXTREME_ASPECT or a < 1 / EXTREME_ASPECT:
            extreme.append(info)
        digest = hashlib.md5(Path(info["path"]).read_bytes()).hexdigest()
        exact_hash.setdefault(digest, []).append(info["name"])

    exact_dups = [names for names in exact_hash.values() if len(names) > 1]

    near_dup_groups: list[list[str]] = []
    if deep and files:
        try:
            groups = imaging.find_duplicate_groups(
                [str(p) for p in files], threshold=NEAR_DUP_THRESHOLD)
            near_dup_groups = [[item["path"] for item in g] for g in groups]
        except Exception:  # noqa: BLE001 - dhash is best-effort
            near_dup_groups = []

    def _has_caption(p: Path) -> bool:
        for suffix in (".txt", ".caption"):
            sidecar = p.with_suffix(suffix)
            if sidecar.is_file() and imaging.read_text_robust(sidecar).strip():
                return True
        return False

    captioned = sum(1 for p in files if _has_caption(p))
    coverage = round(100.0 * captioned / len(files), 1) if files else 0.0
    if files and captioned < len(files):
        coverage = min(coverage, 99.9)   # "100%" with a visibly uncaptioned image reads as a bug

    return {
        "folder": str(folder),
        "count": len(files),
        "total_mb": round(sum(info["bytes"] for info in images) / 1e6, 1),
        "images": images,
        "resolutions": dict(res_buckets.most_common()),
        "aspect_ratio_split": dict(aspect_buckets),
        "corrupt": corrupt,
        "tiny": tiny,
        "extreme_aspect": extreme,
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dup_groups,
        "caption_coverage": coverage,
        "tag_frequency": _top_tags(files),
    }


def _top_tags(files: list[Path], limit: int = 25) -> list[dict]:
    counter: Counter[str] = Counter()
    for p in files:
        sidecar = p.with_suffix(".txt") if p.with_suffix(".txt").is_file() else p.with_suffix(".caption")
        if not sidecar.is_file():
            continue
        text = imaging.read_text_robust(sidecar)
        for tag in re.split(r"[,\n]", text):
            tag = tag.strip()
            if tag:
                counter[tag] += 1
    total = sum(counter.values()) or 1
    return [{"tag": tag, "count": count,
             "pct": round(100.0 * count / total, 1)}
            for tag, count in counter.most_common(limit)]


# ---------------------------------------------------------------------
# Health score
# ---------------------------------------------------------------------

def health_score(analysis: dict) -> dict:
    """Weighted 0-100 score + letter grade.

    Starting from 100, each detected problem subtracts a
    capped penalty so one bad image can't sink a big dataset.
    """
    count = max(analysis.get("count", 0), 1)
    penalties = []

    def penalize(points: float, label: str, detail: str = ""):
        penalties.append({"points": round(points, 1), "label": label, "detail": detail})

    n_corrupt = len(analysis.get("corrupt", []))
    if n_corrupt:
        penalize(min(40, 8 * n_corrupt), "Corrupted images", f"{n_corrupt} file(s) can't be opened")

    n_tiny = len(analysis.get("tiny", []))
    if n_tiny:
        penalize(min(20, 2 * n_tiny), "Low-resolution images", f"{n_tiny} under {MIN_IMAGE_SIDE}px")

    n_extreme = len(analysis.get("extreme_aspect", []))
    if n_extreme:
        penalize(min(15, 1.5 * n_extreme), "Extreme aspect ratios", f"{n_extreme} wider/taller than 3:1")

    n_exact = sum(len(g) - 1 for g in analysis.get("exact_duplicates", []))
    if n_exact:
        penalize(min(20, 2 * n_exact), "Exact duplicates", f"{n_exact} redundant file(s)")

    n_near = sum(len(g) - 1 for g in analysis.get("near_duplicates", []))
    if n_near:
        penalize(min(25, 1.25 * n_near), "Near-duplicates", f"{n_near} visually redundant image(s)")

    coverage = analysis.get("caption_coverage", 0.0)
    if coverage < 100:
        penalize(min(15, 0.5 * (100 - coverage)), "Missing captions",
                 f"{round(100 - coverage, 1)}% of images have no caption")

    score = max(0.0, 100 - sum(p["points"] for p in penalties))
    grade = ("Excellent" if score >= 90 else
             "Good" if score >= 75 else
             "Needs Improvement" if score >= 50 else "Poor")
    return {
        "score": round(score, 1),
        "grade": grade,
        "penalties": penalties,
        "image_count": analysis.get("count", 0),
    }


# ---------------------------------------------------------------------
# Balance analyzer (§14)
# ---------------------------------------------------------------------

# Tag groups the balance analyzer understands. Each group is a
# list of (label, [tag names]) - matching is case-insensitive
# substring so "1st costume" matches "1st_costume" captions.
BALANCE_GROUPS = {
    "Outfits": [
        ("1st Costume", ["1st costume", "costume1", "school uniform"]),
        ("Pajamas", ["pajamas", "pyjamas", "nightwear"]),
        ("Casual", ["casual", "everyday"]),
        ("New Year", ["new year", "kimono", "yukata", "furisode"]),
        ("Swimwear", ["swimsuit", "bikini", "swimwear", "school swimsuit"]),
        ("Formal", ["formal", "dress", "suit", "gown"]),
        ("Gym/Sport", ["gym", "sport", "athletic", "tracksuit"]),
    ],
    "Poses": [
        ("Standing", ["standing", "stand"]),
        ("Sitting", ["sitting", "sit"]),
        ("Walking", ["walking", "walk"]),
        ("Running", ["running", "run"]),
        ("Lying", ["lying", "lying on back", "lying on stomach"]),
    ],
    "Shot types": [
        ("Full body", ["full body", "full_body", "standing"]),
        ("Upper body", ["upper body", "upper_body", "torso"]),
        ("Portrait", ["portrait", "face", "close up", "closeup", "headshot"]),
        ("Back view", ["back view", "back_view", "from behind"]),
        ("Side view", ["side view", "side_view", "profile"]),
    ],
}


def _normalize_tag(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").split())


def _image_tags(path: Path) -> set[str]:
    """The image's authoritative tags: every tag stored in the
    .txt caption, split into tags and normalized.
    Falls back to splitting the .txt caption itself."""
    caption = path.with_suffix(".txt")
    if not caption.is_file():
        caption = path.with_suffix(".caption")
    if caption.is_file():
        return {_normalize_tag(t) for t in
                imaging.read_text_robust(caption).replace("\n", ",").split(",")
                if t.strip()}
    return set()


# Coverage buckets: each maps a human label to the normalized
# tag needles that evidence it. A dataset covers a bucket when
# enough of its images carry at least one matching tag.
COVERAGE_BUCKETS: dict[str, list[str]] = {
    "Front view": ["looking at viewer", "facing viewer", "front view", "front"],
    "Side view": ["side view", "profile", "from the side"],
    "Back view": ["back view", "from behind"],
    "Full body": ["full body"],
    "Upper body": ["upper body"],
    "Face / close-up": ["portrait", "close up", "closeup", "headshot", "face focus"],
    "Expressions": ["smile", "smiling", "laughing", "angry", "sad", "surprised", "happy", "expression"],
    "Outfits": ["outfit", "costume", "uniform", "dress", "kimono", "pajamas", "casual", "swimsuit"],
    "Poses": ["standing", "sitting", "walking", "running", "lying", "fighting"],
    "Backgrounds": ["scenery", "outdoors", "indoors", "city", "nature", "background", "simple background"],
    "Camera angles": ["from above", "from below", "low angle", "high angle", "dutch angle", "overhead"],
}


def _coverage_grade(pct: float) -> str:
    if pct >= 70:
        return "Excellent"
    if pct >= 40:
        return "Good"
    if pct >= 20:
        return "Moderate"
    if pct >= 5:
        return "Weak"
    return "Missing"


def coverage_map(folder: str, per_bucket: int = 8) -> dict:
    """How well the dataset covers each aspect of the concept,
    keyed off the images' normalized tag sets. Each row reports
    count, share, a grade, and the images that evidence it."""
    files = _image_files(folder)
    tag_sets = [_image_tags(p) for p in files]
    total = max(len(files), 1)
    rows = []
    for label, needles in COVERAGE_BUCKETS.items():
        normalized = [_normalize_tag(n) for n in needles]
        matches: list[str] = []
        for p, tags in zip(files, tag_sets):
            blob = " ".join(sorted(tags))
            if any(n in tags or n in blob for n in normalized):
                matches.append(str(p))
        pct = round(100.0 * len(matches) / total, 1)
        rows.append({
            "label": label,
            "count": len(matches),
            "pct": pct,
            "grade": _coverage_grade(pct),
            "images": matches[:per_bucket],
        })
    return {"folder": str(folder), "image_count": len(files), "rows": rows}


def balance_analysis(folder: str, groups: dict | None = None) -> dict:
    """Tag-group distribution with dominance warnings.

    Each image contributes its normalized tag set (from the
    .txt caption when present - authoritative and
    exact - otherwise the .txt caption). A bucket hits an
    image when a needle matches one of its normalized tags
    exactly, or appears in its caption text.
    """
    files = _image_files(folder)
    tag_sets = [_image_tags(p) for p in files]
    caption_blobs = [" ".join(sorted(ts)) for ts in tag_sets]

    groups = groups or BALANCE_GROUPS
    result = {}
    warnings = []
    total = max(len(files), 1)

    for group_name, buckets in groups.items():
        distribution = []
        matched_any = 0
        for label, needles in buckets:
            normalized = [_normalize_tag(n) for n in needles]
            hits = sum(
                1 for tags, blob in zip(tag_sets, caption_blobs)
                if any(n in tags or n in blob for n in normalized))
            matched_any += hits
            distribution.append({
                "label": label, "count": hits,
                "pct": round(100.0 * hits / total, 1),
                "bar": _bar(hits, total),
            })
        distribution.sort(key=lambda d: d["count"], reverse=True)
        result[group_name] = distribution

        if distribution and distribution[0]["count"] > 0:
            top_pct = distribution[0]["pct"]
            if top_pct >= 49:
                warnings.append(
                    f"⚠️ {distribution[0]['label']} represents {top_pct}% of the {group_name.lower()}.")
        if matched_any == 0:
            warnings.append(f"⚠️ No {group_name.lower()} tags found - caption with outfit/pose/shot tags for better analysis.")

    # sparse-bucket warnings (any labeled bucket with very few images)
    for group_name, distribution in result.items():
        sparse = [d for d in distribution if 0 < d["count"] <= max(2, total * 0.03)]
        for d in sparse:
            warnings.append(f"⚠️ Very few {d['label'].lower()} images ({d['count']}).")

    return {"groups": result, "warnings": warnings, "image_count": len(files)}


def _bar(count: int, total: int, width: int = 20) -> str:
    filled = round(width * count / total) if total else 0
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------
# Cleanup plan (§15) - candidates only; removals are explicit
# ---------------------------------------------------------------------

def cleanup_plan(analysis: dict) -> list[dict]:
    """Suggested KEEP/REMOVE/REVIEW decisions. Nothing is
    deleted here - apply_cleanup() acts on user decisions."""
    plan = []
    seen: set[str] = set()

    def add(path: str, reason: str, action: str):
        if path in seen:
            return
        seen.add(path)
        plan.append({"path": path, "reason": reason, "action": action})

    for info in analysis.get("corrupt", []):
        add(info["path"], f"Corrupted: {info['corrupt']}", "remove")
    for info in analysis.get("tiny", []):
        add(info["path"], f"Tiny image ({info['width']}x{info['height']})", "review")
    for info in analysis.get("extreme_aspect", []):
        add(info["path"], f"Extreme aspect ratio ({info['aspect']})", "review")

    for group in analysis.get("exact_duplicates", []):
        for path in group[1:]:  # keep the first, remove the rest
            add(str(Path(analysis["folder"]) / path),
                "Exact duplicate", "remove")
    for group in analysis.get("near_duplicates", []):
        for path in group[1:]:
            add(path, "Near-duplicate (visually redundant)", "review")

    return plan


def apply_cleanup(folder: str, decisions: dict[str, str]) -> dict:
    """Apply user decisions: {"<path>": "remove" | "keep" | "review"}.

    Only explicit "remove" entries are deleted; everything
    else (including unlisted images) is left untouched.
    """
    root = Path(folder)
    removed, kept, reviewed = [], [], []
    for raw_path, action in decisions.items():
        p = Path(raw_path)
        try:
            p.resolve().relative_to(root.resolve())
        except ValueError:
            raise AnalysisError(f"Path outside dataset folder: {raw_path}")
        if not p.is_file():
            continue
        if action == "remove":
            p.unlink()
            # Also remove everything that belongs to the image (sidecars,
            # booru-tags json, magic-eraser backup), or the folder is left
            # with orphan captions and a backup that resurfaces later.
            for related in (p.with_suffix(".txt"),
                            p.with_suffix(".caption"),
                            p.with_name(f"{p.stem}.tags.json"),
                            p.with_name(f"{p.stem}{imaging.BACKUP_SUFFIX}{p.suffix}")):
                try:
                    if related.is_file():
                        related.unlink()
                except OSError:
                    pass
            removed.append(str(p))
        elif action == "review":
            reviewed.append(str(p))
        else:
            kept.append(str(p))
    return {"removed": removed, "kept": kept, "reviewed": reviewed}


# ---------------------------------------------------------------------
# Multi-character detection (part of §2's "image quality problems")
# ---------------------------------------------------------------------
