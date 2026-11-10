"""Tag cleanup tools for the Caption tab.

Undo snapshots, blacklist, implied-tag pruning, auto-sort, duplicate removal,
tag co-occurrence, per-image multi-edit and the colour guard.

Everything here works on the `.txt` caption next to each image, using the same
splitting rules as captions.py (commas or newlines).
"""
from __future__ import annotations

import re
import threading
from collections import Counter
from pathlib import Path

from . import settings_store
from .captions import _caption_path, _is_dataset_image, _join, _split
from .imaging import read_text_robust


def _norm(tag: str) -> str:
    return " ".join(str(tag).strip().lower().replace("_", " ").split())


def _images(folder: str) -> list[Path]:
    p = Path(folder)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    return sorted(f for f in p.iterdir() if _is_dataset_image(f))


def _read(image: Path) -> str | None:
    cp = _caption_path(image)
    return read_text_robust(cp) if cp.exists() else None


def _write(image: Path, tags: list[str]) -> None:
    _caption_path(image).write_text(_join(tags), encoding="utf-8")


# ---------------------------------------------------------------- undo

_UNDO: dict[str, list[dict]] = {}
_LOCK = threading.Lock()
MAX_UNDO = 8


def _key(folder: str) -> str:
    return str(Path(folder)).replace("\\", "/").rstrip("/").lower()


def snapshot(folder: str, label: str, image_paths: list[str] | None = None) -> None:
    """Remember the current captions so the next bulk change can be undone.
    Captions only, never images."""
    paths = [Path(p) for p in image_paths] if image_paths is not None else _images(folder)
    files: dict[str, str | None] = {}
    for img in paths:
        cp = _caption_path(img)
        files[str(cp)] = read_text_robust(cp) if cp.exists() else None
    with _LOCK:
        stack = _UNDO.setdefault(_key(folder), [])
        stack.append({"label": label, "files": files})
        del stack[:-MAX_UNDO]


def undo_status(folder: str) -> dict:
    with _LOCK:
        stack = _UNDO.get(_key(folder), [])
        return {"depth": len(stack), "label": stack[-1]["label"] if stack else ""}


def undo(folder: str) -> dict:
    with _LOCK:
        stack = _UNDO.get(_key(folder), [])
        if not stack:
            return {"restored": 0, "label": "", "depth": 0}
        snap = stack.pop()
        depth = len(stack)
    restored = 0
    for path, text in snap["files"].items():
        cp = Path(path)
        current = read_text_robust(cp) if cp.exists() else None
        if current == text:
            continue
        if text is None:
            cp.unlink(missing_ok=True)
        else:
            cp.write_text(text, encoding="utf-8")
        restored += 1
    return {"restored": restored, "label": snap["label"], "depth": depth}


# ------------------------------------------------- multi-select editing

def edit_images(image_paths: list[str], add: list[str], remove: list[str], position: str = "end") -> int:
    """Add and/or remove tags on exactly the given images."""
    add = [t.strip() for t in add if t and t.strip()]
    remove_n = {_norm(t) for t in remove if t and t.strip()}
    changed = 0
    for p in image_paths:
        img = Path(p)
        original = _read(img)
        tags = _split(original) if original else []
        new = [t for t in tags if _norm(t) not in remove_n]
        have = {_norm(t) for t in new}
        fresh = [t for t in add if _norm(t) not in have]
        # dedupe the incoming list itself
        seen: set[str] = set()
        fresh = [t for t in fresh if not (_norm(t) in seen or seen.add(_norm(t)))]
        new = fresh + new if position == "start" else new + fresh
        if new != tags:
            _write(img, new)
            changed += 1
    return changed


# ------------------------------------------------------------ blacklist

BL_KEY = "caption_blacklist"

COMMON_JUNK = [
    "watermark", "signature", "artist name", "english text", "web address", "dated",
    "username", "patreon username", "twitter username", "pixiv username", "logo",
    "copyright name", "character name", "text focus", "speech bubble", "commentary request",
    "translation request", "bad id", "bad pixiv id", "highres", "absurdres", "lowres",
]


def get_blacklist() -> dict:
    d = settings_store.get(BL_KEY) or {}
    return {
        "tags": [str(t) for t in d.get("tags", [])],
        "apply_after_tagging": bool(d.get("apply_after_tagging", True)),
        "common": COMMON_JUNK,
    }


def set_blacklist(tags: list[str], apply_after_tagging: bool) -> dict:
    seen: set[str] = set()
    clean = []
    for t in tags:
        n = _norm(t)
        if n and n not in seen:
            seen.add(n)
            clean.append(n)
    settings_store.set(BL_KEY, {"tags": clean, "apply_after_tagging": bool(apply_after_tagging)})
    return get_blacklist()


def blacklist_set() -> set[str]:
    return {_norm(t) for t in get_blacklist()["tags"]}


def apply_blacklist(folder: str, tags: set[str] | None = None) -> int:
    bl = tags if tags is not None else blacklist_set()
    if not bl:
        return 0
    changed = 0
    for img in _images(folder):
        original = _read(img)
        if not original:
            continue
        tags_ = _split(original)
        new = [t for t in tags_ if _norm(t) not in bl]
        if len(new) != len(tags_):
            _write(img, new)
            changed += 1
    return changed


# -------------------------------------------------------------- pruning

def scan_prune(folder: str) -> dict:
    """Find "general" tags made redundant by a more specific one in the same
    caption: `shirt` next to `white shirt`, `long hair` next to `very long hair`.

    Returns {"pairs": [...], "groups": [...]}. A pair is (general, specific);
    a group is one general tag with every specific tag that makes it redundant
    and the exact number of captions it would be removed from.
    """
    pairs: Counter = Counter()
    totals: Counter = Counter()
    removable: Counter = Counter()
    display: dict[str, str] = {}
    for img in _images(folder):
        original = _read(img)
        if not original:
            continue
        raw = _split(original)
        norm = {_norm(t) for t in raw}
        for t in raw:
            display.setdefault(_norm(t), t)
        totals.update(norm)
        hit: set[str] = set()
        for b in norm:
            words = b.split()
            for i in range(1, len(words)):
                a = " ".join(words[i:])
                if a in norm:
                    pairs[(a, b)] += 1
                    hit.add(a)
        removable.update(hit)
    out = []
    groups: dict[str, dict] = {}
    for (a, b), n in pairs.most_common():
        out.append({
            "general": display.get(a, a), "specific": display.get(b, b),
            "images": n, "general_total": totals[a],
        })
        g = groups.setdefault(a, {"general": display.get(a, a), "specifics": [],
                                  "images": removable[a], "general_total": totals[a]})
        g["specifics"].append({"tag": display.get(b, b), "images": n})
    ordered = sorted(groups.values(), key=lambda g: (-g["images"], g["general"]))
    return {"pairs": out, "groups": ordered}


def apply_prune(folder: str, pairs: list[dict]) -> int:
    by_general: dict[str, set[str]] = {}
    for p in pairs:
        g, s = _norm(p.get("general", "")), _norm(p.get("specific", ""))
        if g and s:
            by_general.setdefault(g, set()).add(s)
    if not by_general:
        return 0
    changed = 0
    for img in _images(folder):
        original = _read(img)
        if not original:
            continue
        raw = _split(original)
        norm = {_norm(t) for t in raw}
        new = [t for t in raw if not (_norm(t) in by_general and (by_general[_norm(t)] & norm))]
        if len(new) != len(raw):
            _write(img, new)
            changed += 1
    return changed


# ------------------------------------------------------------ duplicates

def dedupe_captions(folder: str) -> int:
    """Remove a tag that is repeated inside one caption (first copy stays)."""
    changed = 0
    for img in _images(folder):
        original = _read(img)
        if not original:
            continue
        raw = _split(original)
        seen: set[str] = set()
        new = []
        for t in raw:
            n = _norm(t)
            if n in seen:
                continue
            seen.add(n)
            new.append(t)
        if len(new) != len(raw):
            _write(img, new)
            changed += 1
    return changed


# ------------------------------------------------------------- auto-sort

# Checked in this order; the first matching bucket wins. Unmatched tags land
# in "other". Order inside a bucket is preserved (stable sort), so the
# tagger's confidence order survives.
_BUCKETS: list[tuple[str, re.Pattern]] = [
    ("subject", re.compile(r"^(\d+\+?(girl|girls|boy|boys|other|others)|solo|solo focus|multiple (girls|boys|others)|no humans|1other)$")),
    ("expression", re.compile(r"smile|blush|open mouth|closed mouth|looking |expression|tears|crying|angry|surprised|^[:;>^=]|tongue|embarrassed|frown|grin|laugh|pout|wink|eyes closed|closed eyes|half.closed|fang|parted lips|shy|sad|happy|annoyed|nervous|serious|smirk|sweat|:d|:o|:3")),
    ("hair", re.compile(r"\bhair\b|\bbangs\b|ahoge|braid|ponytail|twintails|\bbuns?\b|sidelocks|hime cut|bob cut|drill|bald|forehead|hairband|hairclip|hair ")),
    ("eyes", re.compile(r"\beyes?\b|eyebrows|eyelashes|pupils|heterochromia|eyeshadow|eyeliner")),
    ("body", re.compile(r"\bskin\b|breasts|\bbody\b|\btan\b|muscular|thighs|navel|collarbone|\btail\b|\bears?\b|horns?|wings?|\bmole\b|freckles|tattoo|\bthick|\bflat chest|midriff|armpits|abs\b|\blegs\b|\bfeet\b|\bhands?\b(?! on)|nipples|cleavage")),
    ("clothing", re.compile(r"dress|shirt|skirt|jacket|coat|uniform|bikini|swimsuit|pants|shorts|jeans|socks|thighhighs|stockings|pantyhose|shoes|boots|gloves|\bhat\b|\bcap\b|ribbon|\bbow\b|scarf|necktie|\btie\b|sweater|hoodie|kimono|yukata|apron|cape|cloak|armor|\bbra\b|panties|underwear|choker|necklace|earrings|jewelry|glasses|\bbelt\b|collar|sleeves|leotard|bodysuit|maid|headband|headwear|bracelet|\bring\b|capelet|vest|blouse|crop top|tank top|sailor|garter|footwear|legwear|detached|off shoulder|clothes|clothing|naked|nude|topless|lingerie|robe|costume|mask|hood|bandages?|piercing|pendant|brooch|badge|corset|cardigan|pleated|bare")),
    ("pose", re.compile(r"sitting|standing|lying|kneeling|\barms?\b|\bhands? |holding|leaning|walking|running|\bpose\b|from (above|behind|below|side)|upper body|full body|cowboy shot|portrait|close-up|bent over|squatting|crossed legs|spread|jumping|reaching|waving|peace sign|\bv$|dutch angle|facing|profile|head tilt|on back|on side|on stomach|hand on|hands on|arms up|arms behind|pointing|hugging|stretching|dancing|sleeping|eating|drinking|wading|floating|flying")),
    ("setting", re.compile(r"background|outdoors|indoors|\bsky\b|cloud|\btree|flower|night|\bday\b|\bsun\b|moon|water|\broom\b|window|\bcity|street|grass|beach|sunset|\bbed\b|chair|\btable\b|forest|building|simple|gradient|blurry|depth of field|lighting|shadow|light rays|bokeh|stage|classroom|sunlight|rain|snow|petals|\bsea\b|ocean|field|garden|park|bridge|road|stairs|couch|curtains|floor|wall|\bstars?\b|fireworks|lantern")),
    ("meta", re.compile(r"highres|absurdres|commentary|translat|text|watermark|signature|artist|username|rating|lowres|monochrome|greyscale|official art|jpeg artifacts|english|bad id|scan|quality|sketch|chibi|border|letterboxed|pillarboxed|cropped|dated|logo|copyright")),
]
_ORDER = ["subject", "hair", "eyes", "body", "clothing", "expression", "pose", "other", "setting", "meta"]


def bucket_of(tag: str) -> str:
    n = _norm(tag)
    for name, rx in _BUCKETS:
        if rx.search(n):
            return name
    return "other"


def sort_tags(tags: list[str], keep_first: int = 1, pinned: list[str] | None = None) -> list[str]:
    pinned_n = [_norm(t) for t in (pinned or []) if _norm(t)]
    head: list[str] = []
    rest = list(tags)
    # explicit pinned tags go first, in the order given
    for pn in pinned_n:
        for t in list(rest):
            if _norm(t) == pn:
                head.append(t)
                rest.remove(t)
    # then the first N of whatever is left (usually the trigger word)
    k = max(0, int(keep_first))
    head += rest[:k]
    rest = rest[k:]
    rank = {name: i for i, name in enumerate(_ORDER)}
    rest.sort(key=lambda t: rank[bucket_of(t)])   # stable
    return head + rest


def auto_sort(folder: str, keep_first: int = 1, pinned: list[str] | None = None) -> int:
    changed = 0
    for img in _images(folder):
        original = _read(img)
        if not original:
            continue
        raw = _split(original)
        new = sort_tags(raw, keep_first, pinned)
        if new != raw:
            _write(img, new)
            changed += 1
    return changed


# --------------------------------------------------------- co-occurrence

def cooccurrence(folder: str, tag: str = "", min_count: int = 3, top: int = 40) -> dict:
    sets: list[set[str]] = []
    for img in _images(folder):
        original = _read(img)
        if original:
            sets.append({_norm(t) for t in _split(original)})
    total = len(sets)
    freq: Counter = Counter()
    for s in sets:
        freq.update(s)

    t = _norm(tag)
    if t:
        with_t = [s for s in sets if t in s]
        n = len(with_t)
        cnt: Counter = Counter()
        for s in with_t:
            cnt.update(x for x in s if x != t)
        related = [
            {"tag": x, "together": c, "pct": c / n, "of_other": c / freq[x]}
            for x, c in cnt.most_common(top)
        ] if n else []
        return {"tag": t, "images": n, "total": total, "related": related}

    cands = {x for x, c in freq.items() if c >= min_count}
    pairs: Counter = Counter()
    for s in sets:
        keep = sorted(s & cands)
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                pairs[(keep[i], keep[j])] += 1
    out = []
    for (a, b), c in pairs.items():
        pa, pb = c / freq[a], c / freq[b]
        if pa >= 0.9 and pb >= 0.9:
            out.append({"a": a, "b": b, "together": c, "a_total": freq[a], "b_total": freq[b]})
    out.sort(key=lambda r: -r["together"])
    return {"total": total, "pairs": out[:top]}


# ------------------------------------------------------ colour guard

_COLOURS = (
    "black|white|grey|gray|silver|brown|blonde|red|orange|yellow|green|blue|aqua|cyan|teal|"
    "purple|pink|violet|lavender|dark|light|golden|gold|beige|maroon|multicolored|two-tone|gradient"
)
_EYES = re.compile(rf"^({_COLOURS}) eyes$")
_HAIR = re.compile(rf"^({_COLOURS}) hair$")
_SKIN_FANTASY = re.compile(r"^(blue|green|purple|pink|red|yellow|orange|grey|gray|black|white|colored) skin$")
_MULTI_HAIR = re.compile(r"multicolored hair|two-tone hair|gradient hair|streaked hair|colored inner hair|split-color hair|colored tips")

def colour_guard(scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Clean up taggers that over-predict colours.

    - eyes: keep only the most confident colour (unless heterochromia is present)
    - hair: keep only the most confident colour (unless multicolour hair is present)
    - skin: drop coloured-skin tags (blue skin, colored skin...) below 0.8
    `scored` is [(tag_with_underscores, prob)], any order."""
    norm = [(_norm(t), t, p) for t, p in scored]
    names = {n for n, _, _ in norm}
    drop: set[str] = set()

    def single(rx: re.Pattern, skip: bool) -> None:
        if skip:
            return
        hits = [(p, n) for n, _, p in norm if rx.match(n)]
        if len(hits) > 1:
            best = max(hits)[1]
            drop.update(n for _, n in hits if n != best)

    single(_EYES, "heterochromia" in names)
    single(_HAIR, any(_MULTI_HAIR.search(n) for n in names))
    for n, _, p in norm:
        if _SKIN_FANTASY.match(n) and p < 0.8:
            drop.add(n)
    return [(t, p) for n, t, p in norm if n not in drop]
