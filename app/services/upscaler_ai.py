"""
Optional AI upscaling using the 4x-UltraSharp ESRGAN-family model, for
people who want a sharper result on crops than the classical resamplers
in imaging.py (Lanczos / Bicubic / Nearest) can give.

This is loaded through `spandrel` (a small architecture-detecting model
loader) rather than a hand-written network definition, since ESRGAN
checkpoints in the wild ship in a couple of slightly different state-dict
layouts and spandrel already knows how to recognize and load them.

Same optional-dependency shape as the timm/EVA02 tagger backend in
tagger.py: nothing here is imported until AI upscaling is actually used,
so the app runs fine without `torch`/`spandrel` installed - the classical
upscale methods just keep working, and this one shows a clear error
telling you what to install.

The weights (~65MB) are fetched from Hugging Face on first use and
cached under data/model_cache, downloaded as a deliberate, visible step
(download_async / download_progress) the same way the WD14 tagger models
are, rather than silently blocking the first crop for a while.
"""
from __future__ import annotations

import threading
from pathlib import Path

from .. import config

MODEL_NAME = "4x-ultrasharp"
MODEL_REPO = "Kim2091/UltraSharp"
MODEL_FILENAME = "4x-UltraSharp.pth"
HF_FILE_BASE = "https://huggingface.co/{repo}/resolve/main/{filename}"

# The checkpoint itself is a fixed 4x model - other requested scale
# factors are reached by running it at 4x and then resampling that
# result to the exact size wanted, which is still sharper than resizing
# the original crop directly.
NATIVE_SCALE = 4

# Tiling keeps CPU/GPU memory bounded on large crops - a naive whole-image
# pass through this architecture can use several GB on anything sizeable.
_TILE = 512
_TILE_OVERLAP = 32

_lock = threading.Lock()
_model_bundle: dict | None = None
_progress_lock = threading.Lock()
_progress: dict = {"status": "idle"}


class UpscalerError(RuntimeError):
    pass


def _model_dir() -> Path:
    d = config.MODELS_CACHE_DIR / MODEL_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _weights_path() -> Path:
    return _model_dir() / MODEL_FILENAME


def is_downloaded() -> bool:
    return _weights_path().exists()


DEPS = ["torch", "spandrel"]


def available() -> bool:
    """Whether the required packages are installed - lets the frontend
    grey the option out instead of letting someone hit an import error
    mid-crop."""
    try:
        import spandrel  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def status() -> dict:
    return {"available": available(), "downloaded": is_downloaded()}


def download_progress() -> dict:
    with _progress_lock:
        return dict(_progress)


def _set_progress(**kwargs) -> None:
    with _progress_lock:
        _progress.update(kwargs)


def download_async() -> None:
    """Kick off a download on a background thread so the request that
    triggered it can return immediately; poll download_progress() for
    status - same pattern as tagger.download_async."""
    current = download_progress()
    if current.get("status") == "downloading":
        return
    _set_progress(status="downloading", bytes=0, total=0, error=None)
    thread = threading.Thread(target=_download_worker, daemon=True)
    thread.start()


def _download_worker() -> None:
    try:
        _ensure_downloaded()
        _set_progress(status="done")
    except Exception as exc:  # noqa: BLE001 - reported through the progress dict
        _set_progress(status="error", error=str(exc))


def _ensure_downloaded() -> Path:
    target = _weights_path()
    if target.exists():
        return target

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a core dependency
        raise UpscalerError("The 'requests' package is required to download the UltraSharp model.") from exc

    url = HF_FILE_BASE.format(repo=MODEL_REPO, filename=MODEL_FILENAME)
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise UpscalerError(
            f"Couldn't download the UltraSharp model ({url}). AI upscaling needs "
            f"internet access the first time it's used. Original error: {exc}"
        ) from exc

    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    _set_progress(total=total)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                downloaded += len(chunk)
                _set_progress(bytes=downloaded)
        tmp.rename(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _get_model() -> dict:
    global _model_bundle
    with _lock:
        if _model_bundle is not None:
            return _model_bundle
        try:
            import torch
            from spandrel import ModelLoader
        except ImportError as exc:
            raise UpscalerError(
                "AI upscaling needs 'torch' and 'spandrel' installed "
                "(`pip install torch spandrel`) - the Lanczos/Bicubic/Nearest "
                "methods work without them."
            ) from exc

        if not is_downloaded():
            raise UpscalerError(
                "The UltraSharp model hasn't been downloaded yet - "
                "click Download next to the Upscale method first."
            )

        weights_path = _weights_path()
        loaded = ModelLoader().load_from_file(str(weights_path))
        loaded.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = loaded.to(device)
        _model_bundle = {"model": loaded, "torch": torch, "device": device}
        return _model_bundle


def upscale_image(image, scale: float = 4.0):
    """Run 4x-UltraSharp on a PIL image and resample the result to the
    exact `scale` requested. Tiled internally so large crops don't blow
    out CPU/GPU memory."""
    import numpy as np
    from PIL import Image

    bundle = _get_model()
    model, torch, device = bundle["model"], bundle["torch"], bundle["device"]
    try:
        return _upscale_with(bundle, model, torch, device, image, scale)
    finally:
        release_gpu()          # never hold VRAM between edits - training needs all of it


def release_gpu() -> None:
    """Move the upscaler off the GPU and hand the cached memory back to the driver."""
    with _lock:
        b = _model_bundle
        if not b:
            return
        try:
            b["model"] = b["model"].to("cpu")
            torch = b["torch"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _upscale_with(bundle, model, torch, device, image, scale):
    import numpy as np
    from PIL import Image
    if device != "cpu":
        model.to(device)
    img = image.convert("RGB")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        if max(w, h) <= _TILE:
            out = model(tensor)
        else:
            out = _tiled_inference(model, tensor, torch)

    out = out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    result = Image.fromarray((out * 255.0 + 0.5).astype("uint8"))

    if abs(scale - NATIVE_SCALE) > 0.01:
        target_w = max(1, round(w * scale))
        target_h = max(1, round(h * scale))
        result = result.resize((target_w, target_h), Image.LANCZOS)
    return result


def _tiled_inference(model, tensor, torch):
    """Splits large images into overlapping tiles so memory stays
    bounded, runs each through the model, and stitches them back
    together, trimming the overlap on inner edges so tiles butt up
    without visible seams."""
    _, _, h, w = tensor.shape
    scale = NATIVE_SCALE
    out = torch.zeros((1, 3, h * scale, w * scale), dtype=tensor.dtype, device=tensor.device)

    step = _TILE - _TILE_OVERLAP
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + _TILE, h)
            x1 = min(x0 + _TILE, w)
            tile = tensor[:, :, y0:y1, x0:x1]
            tile_out = model(tile)

            trim_l = _TILE_OVERLAP // 2 if x0 > 0 else 0
            trim_t = _TILE_OVERLAP // 2 if y0 > 0 else 0
            trim_r = _TILE_OVERLAP // 2 if x1 < w else 0
            trim_b = _TILE_OVERLAP // 2 if y1 < h else 0

            ty0, ty1 = trim_t * scale, tile_out.shape[2] - trim_b * scale
            tx0, tx1 = trim_l * scale, tile_out.shape[3] - trim_r * scale
            oy0, oy1 = (y0 + trim_t) * scale, (y1 - trim_b) * scale
            ox0, ox1 = (x0 + trim_l) * scale, (x1 - trim_r) * scale
            out[:, :, oy0:oy1, ox0:ox1] = tile_out[:, :, ty0:ty1, tx0:tx1]

    return out
