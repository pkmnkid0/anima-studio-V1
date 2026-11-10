"""
Local image editing for dataset prep: a "magic eraser" that removes a
painted-over region (typically an artist signature or watermark) using
OpenCV's classical inpainting. This runs entirely offline and needs no
model download - it reconstructs the erased area from the surrounding
pixels, which works well for the small, mostly-flat regions a signature
or stamp usually sits on.

The first edit to an image backs up the original bytes alongside it
(`<name>.orig<ext>`) so edits are always reversible.
"""
from __future__ import annotations

import base64
import hashlib
import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

BACKUP_SUFFIX = ".orig"


class ImagingError(RuntimeError):
    pass


def read_text_robust(path: Path) -> str:
    """Reads a text sidecar (caption file) trying common encodings in
    order - these files aren't always written by this app, and a
    dataset captioned or hand-edited elsewhere (e.g. saved from Windows
    Notepad, or written by a tool that adds a UTF-8 BOM) can be UTF-16
    or BOM-prefixed. Reading those straight as UTF-8 with
    errors="replace" turns most of the text into replacement
    characters, which can make an existing, perfectly good caption file
    look empty or garbled once it hits the Caption tab."""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16").strip()
        except UnicodeDecodeError:
            pass
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252").strip()   # captions saved by western-locale tools
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace").strip()


def is_backup_file(path: Path) -> bool:
    """True for the `<name>.orig.<ext>` backups the magic eraser writes -
    these should never be treated as independent dataset images.

    Only counts as a backup if the corresponding "live" file (the one it
    was backed up from) also exists next to it - a real eraser backup
    always has one. Without that check, any imported image that just
    happens to have ".orig" in its name for its own reasons (a fairly
    common convention in other tools/exports) would get silently
    excluded from the whole dataset - the image *and* its caption
    vanishing from the Caption/Edit tabs' listing entirely."""
    if not path.stem.endswith(BACKUP_SUFFIX):
        return False
    original_stem = path.stem[: -len(BACKUP_SUFFIX)]
    if not original_stem:
        return False
    return path.with_name(f"{original_stem}{path.suffix}").exists()


def _backup_path(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + BACKUP_SUFFIX + image_path.suffix)


def has_backup(image_path: str) -> bool:
    return _backup_path(Path(image_path)).exists()


def ensure_backup(image_path: str) -> None:
    p = Path(image_path)
    backup = _backup_path(p)
    if not backup.exists():
        backup.write_bytes(p.read_bytes())


# ----- step-by-step undo/redo of applied edits --------------------------------------------------
# Before an edit overwrites an image in place, the old file is copied into the app's data folder, so Undo in the
# editor can take back each applied edit (not only the unsaved mask strokes) and Redo can bring it back.

HISTORY_KEEP = 12                      # undo steps kept per image
HISTORY_MAX_BYTES = 3 * 1024 ** 3      # oldest histories are dropped beyond this


def _history_dir(image_path: Path) -> Path:
    from ..config import DATA_DIR
    key = hashlib.sha1(str(image_path.resolve()).encode("utf-8", "ignore")).hexdigest()[:20]
    return DATA_DIR / "edit_history" / key


def _hist_files(d: Path, kind: str) -> list[Path]:
    return sorted(d.glob(f"{kind}_*")) if d.is_dir() else []


def _hist_add(d: Path, kind: str, src: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    files = _hist_files(d, kind)
    n = int(files[-1].name.split("_")[1].split(".")[0]) + 1 if files else 1
    shutil.copyfile(src, d / f"{kind}_{n:06d}{src.suffix}")


def push_history(image_path: str) -> None:
    """Call right before an in-place write. Starts a new branch: whatever could have been redone is dropped."""
    p = Path(image_path)
    if not p.is_file():
        return
    d = _history_dir(p)
    for f in _hist_files(d, "r"):
        f.unlink(missing_ok=True)
    _hist_add(d, "u", p)
    for f in _hist_files(d, "u")[:-HISTORY_KEEP]:
        f.unlink(missing_ok=True)
    _prune_history()


def _prune_history() -> None:
    from ..config import DATA_DIR
    root = DATA_DIR / "edit_history"
    if not root.is_dir():
        return
    dirs = [x for x in root.iterdir() if x.is_dir()]
    sizes = {x: sum(f.stat().st_size for f in x.iterdir() if f.is_file()) for x in dirs}
    total = sum(sizes.values())
    for x in sorted(dirs, key=lambda d: d.stat().st_mtime):
        if total <= HISTORY_MAX_BYTES:
            break
        shutil.rmtree(x, ignore_errors=True)
        total -= sizes[x]


def history_state(image_path: str) -> dict:
    d = _history_dir(Path(image_path))
    return {"undo": len(_hist_files(d, "u")), "redo": len(_hist_files(d, "r"))}


def _step(image_path: str, take: str, give: str) -> bool:
    p = Path(image_path)
    d = _history_dir(p)
    files = _hist_files(d, take)
    if not files or not p.is_file():
        return False
    src = files[-1]
    _hist_add(d, give, p)                  # what is on disk now becomes the opposite step
    p.write_bytes(src.read_bytes())
    src.unlink(missing_ok=True)
    return True


def undo_edit(image_path: str) -> bool:
    return _step(image_path, "u", "r")


def redo_edit(image_path: str) -> bool:
    return _step(image_path, "r", "u")


def revert_to_backup(image_path: str) -> bool:
    p = Path(image_path)
    backup = _backup_path(p)
    if not backup.exists():
        return False
    push_history(image_path)
    p.write_bytes(backup.read_bytes())
    return True


def _decode_b64_image(data: str) -> bytes:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data)
    except ValueError as exc:     # binascii.Error is a ValueError
        raise ImagingError("The mask data is invalid - paint over the area again.") from exc


def inpaint(image_path: str, mask_b64: str, method: str = "telea", radius: int = 4, save_as_copy: bool = False) -> str:
    """Erase the masked region of the image. `mask_b64` is a PNG (base64
    or data URL) the same aspect as the image, where any non-black pixel
    marks "erase this". Returns the path actually written to - the
    original path if edited in place (after backing up the original),
    or a new `<name>_edited<ext>` path if `save_as_copy` is set."""
    import cv2

    p = Path(image_path)
    if not p.exists():
        raise ImagingError(f"No such image: {image_path}")

    try:
        image = Image.open(p).convert("RGB")
    except Exception as exc:                              # noqa: BLE001 - PIL raises many types
        raise ImagingError(f"Couldn't open the image: {exc}") from exc
    try:
        mask_img = Image.open(io.BytesIO(_decode_b64_image(mask_b64))).convert("L")
    except Exception as exc:                              # noqa: BLE001
        raise ImagingError("The mask couldn't be read - paint over the area again.") from exc
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.NEAREST)

    mask_arr = np.array(mask_img)
    if not mask_arr.any():
        raise ImagingError("Mask is empty - paint over the area to erase first.")

    img_arr = np.array(image)
    _, mask_bin = cv2.threshold(mask_arr, 10, 255, cv2.THRESH_BINARY)
    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA

    # Both classical algorithms work per-channel on pixel intensity, so
    # RGB in / RGB out is color-correct without a BGR round trip.
    result = cv2.inpaint(img_arr, mask_bin, inpaintRadius=radius, flags=flag)
    out_image = Image.fromarray(result)
    save_kwargs = {"quality": 95} if p.suffix.lower() in (".jpg", ".jpeg") else {}

    if save_as_copy:
        target = _copy_path(p)
        out_image.save(target, **save_kwargs)
        return str(target)

    ensure_backup(image_path)
    push_history(image_path)
    out_image.save(p, **save_kwargs)
    return str(p)


_UPSCALE_METHODS = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC, "nearest": Image.NEAREST}


def _apply_upscale(image: Image.Image, scale: float, method: str = "lanczos") -> Image.Image:
    """Upscale by `scale`. Lanczos/bicubic/nearest are classic (non-AI)
    resamplers - no model download required, so they always work
    offline; Lanczos is the sharpest and the usual default, bicubic a
    softer alternative some people prefer for very small/noisy crops.

    "ultrasharp" instead runs the crop through the 4x-UltraSharp AI
    upscaler (see upscaler_ai.py) - sharper, but needs 'torch' and
    'spandrel' installed plus a one-time model download."""
    if method == "ultrasharp":
        from . import upscaler_ai
        try:
            return upscaler_ai.upscale_image(image, scale=scale)
        except upscaler_ai.UpscalerError as exc:
            raise ImagingError(str(exc)) from exc

    resample = _UPSCALE_METHODS.get(method, Image.LANCZOS)
    w, h = image.size
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), resample)


def _unique_dest_path(dest_dir: Path, stem: str, suffix: str) -> Path:
    candidate = dest_dir / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def upscale(
    image_path: str, scale: float = 2.0, method: str = "lanczos",
    save_as_copy: bool = False, dest_dir: str | None = None,
) -> str:
    """Upscale a full image in place, as a copy, or into a different
    dataset folder (same three destination modes as crop())."""
    p = Path(image_path)
    if not p.exists():
        raise ImagingError(f"No such image: {image_path}")
    if scale <= 1:
        raise ImagingError("Upscale factor must be greater than 1x.")

    try:
        image = Image.open(p).convert("RGB")
    except Exception as exc:                              # noqa: BLE001
        raise ImagingError(f"Couldn't open the image: {exc}") from exc
    result = _apply_upscale(image, scale, method)
    save_kwargs = {"quality": 95} if p.suffix.lower() in (".jpg", ".jpeg") else {}

    if dest_dir:
        target_dir = Path(dest_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = _unique_dest_path(target_dir, p.stem, p.suffix)
        result.save(target, **save_kwargs)
        return str(target)

    if save_as_copy:
        target = _copy_path(p)
        result.save(target, **save_kwargs)
        return str(target)

    ensure_backup(image_path)
    push_history(image_path)
    result.save(p, **save_kwargs)
    return str(p)


def crop(
    image_path: str, box: tuple[float, float, float, float], save_as_copy: bool = False,
    dest_dir: str | None = None, upscale_factor: float | None = None, upscale_method: str = "lanczos",
) -> str:
    """Crop to `box` (x0, y0, x1, y1) in the image's own pixel
    coordinates. Three destination modes (pick at most one):
      - default: edited in place (after backing up the original)
      - `save_as_copy`: written to a new `<name>_edited<ext>` file alongside the original
      - `dest_dir`: written into a different folder entirely (e.g. a new
        training "subset" folder) as a new file, leaving the source
        image untouched - the crop-and-file-into-a-subset workflow.
    If `upscale_factor` > 1 is given, the crop is upscaled by that factor
    (via `upscale_method`) before being saved, in the same step - useful
    since crops are often small and can use the extra resolution."""
    p = Path(image_path)
    if not p.exists():
        raise ImagingError(f"No such image: {image_path}")

    try:
        image = Image.open(p).convert("RGB")
    except Exception as exc:                              # noqa: BLE001
        raise ImagingError(f"Couldn't open the image: {exc}") from exc
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, min(x0, image.width)), max(0, min(x1, image.width))))
    y0, y1 = sorted((max(0, min(y0, image.height)), max(0, min(y1, image.height))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ImagingError("Crop area is too small.")

    cropped = image.crop((int(x0), int(y0), int(x1), int(y1)))
    if upscale_factor and upscale_factor > 1:
        cropped = _apply_upscale(cropped, upscale_factor, upscale_method)
    save_kwargs = {"quality": 95} if p.suffix.lower() in (".jpg", ".jpeg") else {}

    if dest_dir:
        target_dir = Path(dest_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = _unique_dest_path(target_dir, p.stem, p.suffix)
        cropped.save(target, **save_kwargs)
        return str(target)

    if save_as_copy:
        target = _copy_path(p)
        cropped.save(target, **save_kwargs)
        return str(target)

    ensure_backup(image_path)
    push_history(image_path)
    cropped.save(p, **save_kwargs)
    return str(p)


def _copy_path(p: Path) -> Path:
    candidate = p.with_name(f"{p.stem}_edited{p.suffix}")
    counter = 2
    while candidate.exists():
        candidate = p.with_name(f"{p.stem}_edited_{counter}{p.suffix}")
        counter += 1
    return candidate


# ---------------------------------------------------------------------
# Duplicate / near-duplicate detection (difference hash, no extra deps)
# ---------------------------------------------------------------------

def compute_dhash(image_path: str, hash_size: int = 8) -> int:
    """Difference hash: resize to (hash_size+1) x hash_size, grayscale,
    and set each bit based on whether a pixel is brighter than its right
    neighbor. Robust to resizing, re-compression, and minor color
    shifts - exactly the kind of variation reposts/re-encodes have."""
    with Image.open(image_path) as img:
        small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = list(small.getdata())

    bits = 0
    width = hash_size + 1
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * width + col]
            right = pixels[row * width + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_duplicate_groups(image_paths: list[str], threshold: int = 6) -> list[list[dict]]:
    """Groups of near-duplicate images (dHash Hamming distance <=
    threshold). Uses union-find so A~B~C forms one group even if A and
    C alone are just over the threshold - the usual behavior wanted for
    "these all look like the same shot" clustering. threshold=0 means
    only flag hashes that match exactly."""
    hashes: dict[str, int] = {}
    for path in image_paths:
        try:
            hashes[path] = compute_dhash(path)
        except Exception:
            continue  # unreadable/corrupt image - skip rather than fail the whole scan

    parent = {path: path for path in hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    paths = list(hashes.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hamming_distance(hashes[paths[i]], hashes[paths[j]]) <= threshold:
                union(paths[i], paths[j])

    groups: dict[str, list[str]] = {}
    for path in paths:
        root = find(path)
        groups.setdefault(root, []).append(path)

    result = []
    for members in groups.values():
        if len(members) < 2:
            continue
        # Largest file first - a reasonable default "keep this one" guess
        # (usually the highest-resolution/least-compressed copy).
        members_sorted = sorted(members, key=lambda p: Path(p).stat().st_size, reverse=True)
        result.append([
            {"path": m, "size": Path(m).stat().st_size} for m in members_sorted
        ])
    result.sort(key=lambda g: len(g), reverse=True)
    return result


# ---------------------------------------------------------------------
# AI edit results (Edit tab -> AI tools) and mask adjustment
# ---------------------------------------------------------------------

_ALPHA_FORMATS = (".png", ".webp")


def _background_rgb(background: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b); anything else is white."""
    import re

    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(background or "").strip())
    if not m:
        return (255, 255, 255)
    v = int(m.group(1), 16)
    return ((v >> 16) & 255, (v >> 8) & 255, v & 255)


def _has_transparency(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA"):
        return img.convert("RGBA").getchannel("A").getextrema()[0] < 255
    return False


def save_ai_result(image_path: str, result_path: str, save_as_copy: bool = False, keep_alpha: bool = False,
                   background: str = "#ffffff") -> str:
    """Write an AI tool's result (a PNG from the edit engine) over the dataset image, or next to it as a copy.

    The original is backed up first (the same `<name>.orig.<ext>` the magic eraser uses) whenever the file is replaced.
    A result with transparency (background removal) is kept as-is only when `keep_alpha` is set: PNG and WebP images
    take it in place, a JPEG/BMP original is left alone and the cut-out is saved as a new `<name>_edited.png` instead.
    Otherwise the transparency is flattened onto `background`. Returns the path written."""
    p = Path(image_path)
    if not p.exists():
        raise ImagingError(f"No such image: {image_path}")
    try:
        res = Image.open(result_path)
        res.load()
    except Exception as exc:                              # noqa: BLE001 - PIL raises many types
        raise ImagingError("That result is no longer available - run the tool again.") from exc

    ext = p.suffix.lower()
    if keep_alpha and _has_transparency(res):
        out = res.convert("RGBA")
        if ext in _ALPHA_FORMATS:
            target = _copy_path(p) if save_as_copy else p
        else:
            target = _copy_path(p.with_suffix(".png"))
    else:
        if res.mode in ("RGBA", "LA", "PA"):
            rgba = res.convert("RGBA")
            flat = Image.new("RGBA", rgba.size, _background_rgb(background) + (255,))
            out = Image.alpha_composite(flat, rgba).convert("RGB")
        else:
            out = res.convert("RGB")
        target = _copy_path(p) if save_as_copy else p

    save_kwargs = {"quality": 95} if target.suffix.lower() in (".jpg", ".jpeg", ".webp") else {}
    if target == p:
        ensure_backup(image_path)
        push_history(image_path)
    out.save(target, **save_kwargs)
    return str(target)


def adjust_mask(mask_b64: str, op: str, amount: int = 5) -> str:
    """Grow, shrink or invert a painted mask (PNG data URL in, PNG data URL out; white = the area to fill).

    Growing a mask a few pixels before an AI fill usually helps: it covers the soft halo around an object that the
    brush misses."""
    import cv2

    if op not in ("expand", "shrink", "reverse"):
        raise ImagingError("Unknown mask operation.")
    try:
        mask_img = Image.open(io.BytesIO(_decode_b64_image(mask_b64))).convert("L")
    except Exception as exc:                              # noqa: BLE001
        raise ImagingError("The mask couldn't be read - paint over the area again.") from exc
    arr = np.array(mask_img)
    arr = np.where(arr >= 127, 255, 0).astype(np.uint8)
    if op == "reverse":
        arr = 255 - arr
    else:
        amount = max(1, min(200, int(amount)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * amount + 1, 2 * amount + 1))
        arr = cv2.dilate(arr, kernel) if op == "expand" else cv2.erode(arr, kernel)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
