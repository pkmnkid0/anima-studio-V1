"""
Generates smaller, model-friendly-resolution copies of a whole dataset
tree - lower-resolution training data trains noticeably faster (fewer
pixels per step), which is what you'd want before a first-pass or
low-res training run. Adapted from a script the user shared, with three
bugs fixed along the way:

- Transparent images (RGBA/LA/P) are now composited onto a white
  background before the alpha channel is dropped, instead of just
  discarding alpha and keeping whatever RGB values happened to sit
  underneath it - which for a lot of PNG exporters is black, and showed
  up as dark fringing around transparent edges on cutout-style art.
- The "already small enough, just round to a multiple of 64" path now
  derives height from the rounded width via the aspect ratio, instead
  of rounding each axis independently - the old way could visibly skew
  small/thin images.
- Output layout: one folder per resolution, named after the source folder:

      <folder>/resized images/<folder name> 512/image.png + image.txt
      <folder>/resized images/<folder name> 1024/...

  e.g. "gorgeous mushroom artstyle" -> "resized images/gorgeous mushroom
  artstyle 512" and "... 1024". Images inside subfolders keep their
  subfolder (so kohya-style "10_character" folders survive) inside those.
  Filenames stay as they were (output is always .png); only when art.png and
  art.jpg share a folder does the second get its extension appended.

Runs on a thread pool, not a process pool like the original script did -
this executes inside the already-running Anima Studio server, and
spawning subprocesses from inside a long-running app is a real way to
hang or duplicate the app, particularly on Windows, where multiprocessing
re-imports the launching script in each worker unless very carefully
guarded. Pillow releases the GIL during its own resize/decode/encode
work, so threads still get real parallelism for this without that risk.
"""
from __future__ import annotations

import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

OUTPUT_FOLDER_NAME = "resized images"
_SKIP_DIRS = {"resized images", "resized"}   # never re-process output (incl. the old name)
DEFAULT_RESOLUTIONS = [512, 1024, 1536]
DIVISIBLE = 64
_VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_lock = threading.Lock()
_status: dict = {"stage": "idle", "done": 0, "total": 0, "errors": [], "folder": None}


class ResizeError(RuntimeError):
    pass


def status() -> dict:
    with _lock:
        return dict(_status, errors=list(_status["errors"]))


def is_busy() -> bool:
    with _lock:
        return _status["stage"] == "running"


def _set(**kwargs) -> None:
    with _lock:
        _status.update(kwargs)


def start(folder: str, resolutions: list[int]) -> dict:
    if is_busy():
        raise ResizeError("A resize job is already running - wait for it to finish.")
    base = Path(folder)
    if not base.exists() or not base.is_dir():
        raise ResizeError(f"Not a directory: {folder}")

    clean_resolutions = sorted({int(r) for r in resolutions if int(r) > 0})
    if not clean_resolutions:
        raise ResizeError("Choose at least one resolution.")

    _set(stage="running", done=0, total=0, errors=[], folder=str(base))
    threading.Thread(target=_run, args=(base, clean_resolutions), daemon=True).start()
    return status()


def _find_images(base: Path) -> list[Path]:
    found = []
    for root, dirs, files in os.walk(base):
        # Don't descend into any previous output folder, at any depth, so
        # re-running this doesn't reprocess its own output.
        dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS]
        for fname in files:
            if Path(fname).suffix.lower() in _VALID_EXTENSIONS:
                found.append(Path(root) / fname)
    return found


def _output_dir(base: Path, source_dir: Path, res: int) -> Path:
    """resized images/<base folder name> <res>[/<subfolder>]"""
    try:
        rel = source_dir.relative_to(base)
    except ValueError:
        rel = Path(".")
    root = base / OUTPUT_FOLDER_NAME / f"{base.name or 'images'} {res}"
    return root if str(rel) in ("", ".") else root / rel


def _plan_names(images: list[Path]) -> dict[Path, str]:
    """Output stem for each image. Keeps the original name; only a clash inside
    one folder (art.png vs art.jpg, or differing only by case) gets a suffix."""
    plan: dict[Path, str] = {}
    used: dict[Path, set[str]] = {}
    for path in sorted(images):
        taken = used.setdefault(path.parent, set())
        stem, ext_tag = path.stem, path.suffix.lower().lstrip(".")
        candidate = stem
        if candidate.lower() in taken:
            candidate = f"{stem}_{ext_tag}"
        n = 2
        while candidate.lower() in taken:
            candidate = f"{stem}_{ext_tag}_{n}"
            n += 1
        taken.add(candidate.lower())
        plan[path] = candidate
    return plan


def _run(base: Path, resolutions: list[int]) -> None:
    try:
        images = _find_images(base)
        _set(total=len(images))
        if not images:
            _set(stage="done")
            return

        targets = {res: res * res for res in resolutions}
        names = _plan_names(images)
        errors: list[str] = []
        done = 0
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            futures = {pool.submit(_process_one, base, path, names[path], targets): path for path in images}
            for future in as_completed(futures):
                done += 1
                err = future.result()
                if err:
                    errors.append(err)
                _set(done=done, errors=errors)
        _set(stage="done")
    except Exception as exc:  # noqa: BLE001 - surfaced to the user via status()
        _set(stage="error", errors=[str(exc)])


def _process_one(base: Path, path: Path, out_stem: str, targets: dict[int, int]) -> str | None:
    try:
        source_dir = path.parent
        caption_path = source_dir / f"{path.stem}.txt"

        with Image.open(path) as raw:
            img = _flatten_to_rgb(raw)
            orig_w, orig_h = img.size

            for res, target_area in targets.items():
                target_w, target_h = _target_shape(orig_w, orig_h, target_area)
                is_downscale = (target_w * target_h) <= (orig_w * orig_h)
                resample = Image.Resampling.BOX if is_downscale else Image.Resampling.LANCZOS
                resized = img.resize((target_w, target_h), resample)

                dest_dir = _output_dir(base, source_dir, res)
                dest_dir.mkdir(parents=True, exist_ok=True)
                resized.save(dest_dir / f"{out_stem}.png", format="PNG")

                if caption_path.exists():
                    shutil.copy2(caption_path, dest_dir / f"{out_stem}.txt")
        return None
    except Exception as exc:  # noqa: BLE001 - collected into the job's error list
        return f"{path.name}: {exc}"


def _flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Drops alpha the safe way: composite onto white first. Plain
    .convert('RGB') on an RGBA/LA/P image keeps whatever color values
    sat under the transparent pixels (often black), which shows up as
    dark fringing around transparent edges once alpha is gone."""
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _target_shape(width: int, height: int, target_area: int, divisible: int = DIVISIBLE) -> tuple[int, int]:
    """New size preserving aspect ratio, capped at target_area, rounded
    to the nearest multiple of `divisible` (most training pipelines want
    both dimensions divisible by 64)."""
    current_area = width * height
    aspect_ratio = width / height

    if current_area <= target_area:
        # Already small enough - just round to a valid size instead of
        # upscaling it. Height is derived from the rounded width via the
        # aspect ratio rather than rounded independently, so a small/thin
        # image doesn't visibly skew (64px can be a large fraction of a
        # small image's own dimensions).
        new_w = max(divisible, round(width / divisible) * divisible)
        new_h = max(divisible, round((new_w / aspect_ratio) / divisible) * divisible)
        return new_w, new_h

    target_h = math.sqrt(target_area / aspect_ratio)
    target_w = aspect_ratio * target_h
    new_w = max(divisible, round(target_w / divisible) * divisible)
    new_h = max(divisible, round(target_h / divisible) * divisible)
    return new_w, new_h
