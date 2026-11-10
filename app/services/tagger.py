"""
Local booru-style auto-tagger using a SmilingWolf WD14 ONNX tagger
model - the same family of model kohya-ss's own sd-scripts ships
(tag_images_by_wd14_tagger.py) and that most captioning tools in this
space use. Runs fully locally via onnxruntime once the model is cached;
nothing about a specific image is sent anywhere.

The model + tag list are pulled from Hugging Face on first use and
cached under data/model_cache. Everything after that runs offline.
Downloading is a deliberate, visible step (see download_async /
download_progress) rather than something that silently happens the
first time someone clicks "tag" - a multi-hundred-MB download with no
feedback looks like a hang.
"""
from __future__ import annotations

import csv
import io
import threading
from pathlib import Path

from .. import config

MODEL_REPOS = {
    "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3": "SmilingWolf/wd-convnext-tagger-v3",
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
    # Community "canary" continuation of the EVA02-Large v3 tagger: same
    # base model fine-tuned on everything posted since that model's own
    # training cutoff, plus ~6k new tags (mostly newer characters). Ships
    # as a timm/safetensors checkpoint rather than ONNX, so it needs its
    # own download + inference path below (see MODEL_BACKEND).
    "wd-eva02-tagger-2026-canary": "ashen-sensored/wd-eva02-tagger-2026-canary",
    # PixAI Tagger v1.0 (Apache-2.0, not gated): SAM3-based multi-label tagger, 30,877 tags
    # (general / character / style / copyright / meta / rating), 1008px input. It ships its own
    # model code (tagger_pipeline.py) and is run through a transformers pipeline.
    "pixai-tagger-v1.0": "pixai-labs/pixai-tagger-v1.0",
}
DEFAULT_MODEL = "wd-vit-tagger-v3"
HF_FILE_BASE = "https://huggingface.co/{repo}/resolve/main/{filename}"

# "onnx" models ship model.onnx + selected_tags.csv and run through
# onnxruntime with raw 0-255 BGR pixel input (no normalization) - this is
# the original WD14/v3 tagger family. "timm" models ship model.safetensors
# + config.json + selected_tags.csv and run through timm/torch with
# standard ImageNet-style RGB input normalized to [-1, 1].
MODEL_BACKEND = {
    "wd-eva02-tagger-2026-canary": "timm",
    "pixai-tagger-v1.0": "pixai",
}
# architecture name timm needs for timm.create_model(...) - only used for
# "timm"-backend models.
MODEL_TIMM_ARCH = {
    "wd-eva02-tagger-2026-canary": "eva02_large_patch14_448",
}

_MODEL_FILES = {
    "onnx": ("model.onnx", "selected_tags.csv"),
    "timm": ("model.safetensors", "selected_tags.csv"),
    # weights first; the rest is the model code + configs the transformers pipeline loads
    "pixai": ("model.safetensors", "config.json", "preprocessor_config.json", "tagger_pipeline.py"),
}

_sessions: dict[str, "object"] = {}
_tag_lists: dict[str, dict] = {}
_progress_lock = threading.Lock()
_progress: dict[str, dict] = {}


def _backend(model_name: str) -> str:
    return MODEL_BACKEND.get(model_name, "onnx")


def backend_for(model_name: str) -> str:
    """Public accessor for _backend - "onnx" (default, just needs
    onnxruntime, already in requirements.txt) or "timm" (needs
    torch/timm/safetensors, see TIMM_DEPS)."""
    return _backend(model_name)


class TaggerError(RuntimeError):
    pass


def available_models() -> list[str]:
    return list(MODEL_REPOS.keys())


def _model_dir(model_name: str) -> Path:
    d = config.MODELS_CACHE_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_downloaded(model_name: str) -> bool:
    d = _model_dir(model_name)
    filenames = _MODEL_FILES[_backend(model_name)]
    return all((d / filename).exists() for filename in filenames)


TIMM_DEPS = ["torch", "torchvision", "timm", "safetensors", "transformers", "numpy", "pillow"]


def timm_deps_available() -> bool:
    """Whether the extra packages the timm-backed tagger model(s) need
    are installed - lets the frontend offer an install button instead of
    letting someone hit the ImportError only after clicking the model."""
    try:
        import safetensors  # noqa: F401
        import timm  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def pixai_deps_available() -> bool:
    try:
        import torchvision  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return timm_deps_available()


def deps_available(model_name: str) -> bool:
    b = _backend(model_name)
    if b == "onnx":
        return True
    return pixai_deps_available() if b == "pixai" else timm_deps_available()


def download_progress(model_name: str) -> dict:
    with _progress_lock:
        return dict(_progress.get(model_name, {"status": "idle"}))


def _set_progress(model_name: str, **kwargs) -> None:
    with _progress_lock:
        _progress.setdefault(model_name, {}).update(kwargs)


def download_async(model_name: str) -> None:
    """Kick off a download on a background thread so the request that
    triggered it can return immediately; poll download_progress() for
    status."""
    if model_name not in MODEL_REPOS:
        raise TaggerError(f"Unknown tagger model: {model_name}")
    current = download_progress(model_name)
    if current.get("status") == "downloading":
        return  # already in flight
    _set_progress(model_name, status="downloading", bytes=0, total=0, error=None)
    thread = threading.Thread(target=_download_worker, args=(model_name,), daemon=True)
    thread.start()


def _download_worker(model_name: str) -> None:
    try:
        _ensure_downloaded(model_name, progress_key=model_name)
        _set_progress(model_name, status="done")
    except Exception as exc:  # noqa: BLE001 - reported through the progress dict, not raised
        _set_progress(model_name, status="error", error=str(exc))


def _ensure_downloaded(model_name: str, progress_key: str | None = None) -> tuple[Path, Path]:
    if model_name not in MODEL_REPOS:
        raise TaggerError(f"Unknown tagger model: {model_name}")
    repo = MODEL_REPOS[model_name]
    d = _model_dir(model_name)
    names = _MODEL_FILES[_backend(model_name)]
    files = [(n, d / n) for n in names]
    weights_path, csv_path = files[0][1], files[1][1]

    if all(dest.exists() for _, dest in files):
        return weights_path, csv_path

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a core dependency
        raise TaggerError("The 'requests' package is required to download tagger models.") from exc

    for filename, dest in files:
        if dest.exists():
            continue
        url = HF_FILE_BASE.format(repo=repo, filename=filename)
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise TaggerError(
                f"Couldn't download {filename} for {model_name} ({url}). "
                "Auto-tagging needs internet access the first time it's used "
                f"so it can fetch the model. Original error: {exc}"
            ) from exc

        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        if progress_key:
            _set_progress(progress_key, total=total, bytes=0)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                downloaded += len(chunk)
                if progress_key:
                    _set_progress(progress_key, bytes=downloaded)
        tmp.rename(dest)

    return weights_path, csv_path


def _load_tag_list(csv_path: Path) -> dict:
    rating_tags, general_tags, character_tags = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row["name"]
            category = int(row["category"])
            if category == 9:
                rating_tags.append(name)
            elif category == 4:
                character_tags.append(name)
            else:
                general_tags.append(name)
    # Order in the CSV is the same order model outputs are indexed by,
    # so keep the full ordered name list too.
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)
    return {
        "names": [r["name"] for r in all_rows],
        "categories": [int(r["category"]) for r in all_rows],
        "rating_tags": rating_tags,
        "general_tags": general_tags,
        "character_tags": character_tags,
    }


def _get_session(model_name: str):
    if model_name in _sessions:
        return _sessions[model_name], _tag_lists[model_name]

    backend = _backend(model_name)
    weights_path, csv_path = _ensure_downloaded(model_name)

    if backend == "pixai":
        session = _load_pixai(model_name)
        _sessions[model_name] = session
        _tag_lists[model_name] = {}
        return session, {}
    if backend == "timm":
        session = _load_timm_model(model_name, weights_path)
    else:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise TaggerError(
                "onnxruntime isn't installed. Install it with `pip install onnxruntime` "
                "(or `onnxruntime-gpu` if you have a CUDA setup) to use auto-tagging."
            ) from exc
        session = ort.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])

    tag_list = _load_tag_list(csv_path)

    _sessions[model_name] = session
    _tag_lists[model_name] = tag_list
    return session, tag_list


def _load_pixai(model_name: str):
    """PixAI Tagger v1.0 through its own transformers pipeline (the repo ships the model code,
    hence trust_remote_code). Loaded from the local download folder - no further network use."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise TaggerError(
            "PixAI Tagger needs torch, torchvision, timm and transformers installed "
            "(use the Install button, or `pip install torch torchvision timm transformers`)."
        ) from exc
    d = str(_model_dir(model_name))
    try:
        pipe = pipeline(model=d, image_processor=d, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        raise TaggerError(f"Couldn't load PixAI Tagger: {exc}") from exc
    return {"kind": "pixai", "pipe": pipe, "torch": torch}


def _tag_pixai(session, image_bytes: bytes, general_threshold, character_threshold, include_style) -> dict:
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    thr = {}
    if general_threshold is not None:
        thr["general"] = general_threshold
    if character_threshold is not None:
        thr["character"] = character_threshold
    with session["torch"].no_grad():
        out = session["pipe"](image, threshold=thr) if thr else session["pipe"](image)
    res = out["results"] if isinstance(out, dict) and "results" in out else out
    general = sorted((res.get("general") or {}).items(), key=lambda x: -x[1])
    character = sorted((res.get("character") or {}).items(), key=lambda x: -x[1])
    rating = res.get("rating") or {}
    best_rating = max(rating.items(), key=lambda x: x[1])[0] if rating else None
    extra = []
    if include_style:
        extra = sorted((res.get("style") or {}).items(), key=lambda x: -x[1])
    return {
        "general": [n for n, _ in extra] + [n for n, _ in general],
        "character": [n for n, _ in character],
        "general_scored": general,
        "character_scored": character,
        "rating": best_rating,
    }


def _load_timm_model(model_name: str, weights_path: Path):
    """Loads a timm/safetensors-backed tagger (currently just the EVA02
    2026-canary model) - a plain image classifier head with one sigmoid
    output per tag, same shape as the ONNX WD14 models but run through
    torch instead of onnxruntime."""
    try:
        import timm
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise TaggerError(
            "This tagger model needs 'torch', 'timm' and 'safetensors' installed "
            "(`pip install torch timm safetensors`) - it ships as a timm checkpoint "
            "rather than ONNX. The other WD14 tagger models don't need these."
        ) from exc

    arch = MODEL_TIMM_ARCH[model_name]
    tag_csv = weights_path.with_name("selected_tags.csv")
    with open(tag_csv, encoding="utf-8") as fh:
        num_classes = sum(1 for _ in fh) - 1  # minus header row

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    model.eval()
    return {"kind": "timm", "model": model, "torch": torch, "input_size": 448}


def _preprocess(image_bytes: bytes, target_size: int):
    """Preprocessing for the ONNX WD14/v3 tagger family: pad to square on
    a white background, resize, and feed raw 0-255 BGR pixels (no mean/std
    normalization) - matches how these models were trained."""
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    size = max(w, h)
    padded = Image.new("RGB", (size, size), (255, 255, 255))
    padded.paste(image, ((size - w) // 2, (size - h) // 2))
    if size != target_size:
        padded = padded.resize((target_size, target_size), Image.BICUBIC)

    arr = np.asarray(padded, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR, matches the model's training preprocessing
    arr = np.expand_dims(arr, axis=0)
    return arr


def _preprocess_timm(image_bytes: bytes, target_size: int):
    """Preprocessing for the timm/EVA02 canary tagger: pad to square,
    resize, normalized to [-1, 1] (mean=std=0.5 per the model's own
    pretrained_cfg), then channels flipped to BGR (see below). Unlike the
    ONNX WD14 models it is normalized rather than raw 0-255."""
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    size = max(w, h)
    padded = Image.new("RGB", (size, size), (255, 255, 255))
    padded.paste(image, ((size - w) // 2, (size - h) // 2))
    if size != target_size:
        padded = padded.resize((target_size, target_size), Image.BICUBIC)

    arr = np.asarray(padded, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # -> [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    # The WD taggers were trained on BGR input, including their timm builds:
    # SmilingWolf's reference script (wdv3-timm) flips the channels after the
    # transform with `inputs[:, [2, 1, 0]]`. Skipping the flip swaps red and
    # blue, so red eyes came out as blue_eyes and warm skin as blue skin.
    arr = np.ascontiguousarray(arr[::-1])
    arr = np.expand_dims(arr, axis=0)
    return arr


def tag_image_bytes(
    image_bytes: bytes,
    model_name: str = DEFAULT_MODEL,
    general_threshold: float = 0.35,
    character_threshold: float = 0.75,
    include_rating: bool = True,
) -> dict:
    session, tag_list = _get_session(model_name)
    backend = _backend(model_name)

    if backend == "pixai":
        # PixAI's own default thresholds (general 0.17, character 0.27) were tuned for this model;
        # the WD-style 0.35 / 0.75 sliders would drop most of its tags, so only use the sliders
        # when they were moved off the WD defaults.
        g = None if abs(general_threshold - 0.35) < 1e-9 else general_threshold
        c = None if abs(character_threshold - 0.75) < 1e-9 else character_threshold
        result = _tag_pixai(session, image_bytes, g, c, include_style=True)
        if not include_rating:
            result.pop("rating", None)
        return result

    if backend == "timm":
        torch = session["torch"]
        target_size = session.get("input_size", 448)
        input_array = _preprocess_timm(image_bytes, target_size)
        with torch.no_grad():
            logits = session["model"](torch.from_numpy(input_array))
            probs = torch.sigmoid(logits)[0].numpy()
    else:
        input_meta = session.get_inputs()[0]
        shape = input_meta.shape
        # shape is typically [None, H, W, 3]; fall back to 448 (the common
        # v3 tagger input size) if the dimension isn't static.
        target_size = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else 448

        input_array = _preprocess(image_bytes, target_size)
        input_name = input_meta.name
        output_name = session.get_outputs()[0].name
        probs = session.run([output_name], {input_name: input_array})[0][0]

    names = tag_list["names"]
    categories = tag_list["categories"]

    general, character, rating = [], [], []
    for name, category, prob in zip(names, categories, probs):
        prob = float(prob)
        if category == 9:
            rating.append((name, prob))
        elif category == 4 and prob >= character_threshold:
            character.append((name, prob))
        elif category == 0 and prob >= general_threshold:
            general.append((name, prob))

    general.sort(key=lambda x: -x[1])
    character.sort(key=lambda x: -x[1])
    best_rating = max(rating, key=lambda x: x[1])[0] if rating else "unknown"

    result = {
        "general": [name for name, _ in general],
        "character": [name for name, _ in character],
        "general_scored": general,
        "character_scored": character,
    }
    if include_rating:
        result["rating"] = best_rating
    return result


def tag_image_file(image_path: str, **kwargs) -> dict:
    data = Path(image_path).read_bytes()
    return tag_image_bytes(data, **kwargs)


_REGION_CACHE: dict = {}


def tag_region(image_path: str, box: dict, model_name: str = DEFAULT_MODEL, top: int = 40, query: str = "") -> list[dict]:
    """Tag only a selected part of an image.

    `box` is {x, y, w, h} as fractions (0-1) of the image. The crop is scored by
    the tagger at its full input size, so small details that get drowned out when the
    whole picture is tagged come through. Every tag the model knows is scored; `whole`
    is its score on the whole image, so `lift` shows what the crop revealed.
    `query` searches the model's whole vocabulary (every word must appear in the tag),
    so a tag that scored too low to make the top list can still be found."""
    from PIL import Image  # noqa: PLC0415

    try:
        x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise TaggerError("A region needs x, y, w and h.") from exc
    key = (image_path, Path(image_path).stat().st_mtime_ns, model_name, round(x, 4), round(y, 4), round(w, 4), round(h, 4))
    scores = _REGION_CACHE.get(key)
    if scores is None:
        img = Image.open(image_path)
        img.load()
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        W, H = img.size
        left, top_px = max(0, int(round(x * W))), max(0, int(round(y * H)))
        right, bottom = min(W, int(round((x + w) * W))), min(H, int(round((y + h) * H)))
        if right - left < 8 or bottom - top_px < 8:
            raise TaggerError("That region is too small - drag a larger box.")
        buf = io.BytesIO()
        img.crop((left, top_px, right, bottom)).save(buf, "PNG")
        whole_buf = io.BytesIO()
        img.save(whole_buf, "PNG")

        def scored(data: bytes) -> dict[str, float]:
            r = tag_image_bytes(data, model_name=model_name, general_threshold=0.0,
                                character_threshold=0.0, include_rating=False)
            return {n: float(p) for n, p in r["general_scored"]}

        crop, whole = scored(buf.getvalue()), scored(whole_buf.getvalue())
        scores = [(n, p, whole.get(n, 0.0)) for n, p in crop.items()]
        if len(_REGION_CACHE) > 8:
            _REGION_CACHE.clear()
        _REGION_CACHE[key] = scores

    words = [t for t in query.lower().replace("_", " ").split() if t]
    rows = [(n, p, wh) for n, p, wh in scores if all(wd in n.replace("_", " ").lower() for wd in words)]
    rows.sort(key=lambda r: -r[1])
    return [{"tag": n.replace("_", " "), "score": round(p, 4), "whole": round(wh, 4), "lift": round(p - wh, 4)}
            for n, p, wh in rows[:max(1, int(top))]]
