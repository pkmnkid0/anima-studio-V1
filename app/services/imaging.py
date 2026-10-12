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
import io
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
    raw = path.read_bytes()
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


def revert_to_backup(image_path: str) -> bool:
    p = Path(image_path)
    backup = _backup_path(p)
    if not backup.exists():
        return False
    p.write_bytes(backup.read_bytes())
    return True


def _decode_b64_image(data: str) -> bytes:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


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

    image = Image.open(p).convert("RGB")
    mask_img = Image.open(io.BytesIO(_decode_b64_image(mask_b64))).convert("L")
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

    image = Image.open(p).convert("RGB")
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

    image = Image.open(p).convert("RGB")
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
