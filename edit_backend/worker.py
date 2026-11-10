"""The AI edit worker: runs IOPaint (the engine behind lama-cleaner) inside the edit environment.

Anima Studio starts this file with the edit environment's Python (`python -m edit_backend.worker`) and talks to it over
stdin/stdout, one JSON object per line - no network port is opened, so nothing in a browser can reach it.

  request   {"id": 7, "cmd": "inpaint", ...arguments}
  event     {"id": 7, "event": "progress", "step": 12, "total": 50, "message": ""}      (any number, optional)
            {"id": 7, "event": "log", "line": "..."}
  response  {"id": 7, "ok": true, "result": {...}}  or  {"id": 7, "ok": false, "error": "readable message"}

Everything IOPaint prints goes to stderr (Anima Studio shows it in the log); stdout carries only this protocol.
Images travel as files (the caller passes paths), never as base64 through the pipe.

The work mirrors IOPaint's own server (iopaint/api.py): same pre- and post-processing, same ModelManager and plugin
calls, so results match what lama-cleaner gives. IOPaint is imported lazily, inside the handlers.

`python -m edit_backend.worker --check` prints a JSON report of the environment and exits (used by Verify).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import threading
import traceback
from pathlib import Path

from edit_backend import catalog

_PROTO = None                                  # the real stdout, used only for protocol lines
_SEND_LOCK = threading.Lock()
_CURRENT = {"id": None, "total": None}


class WorkerError(RuntimeError):
    """A failure whose message is meant for the person using the editor."""


# ------------------------------------------------------------------ protocol

def send(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=True)
    with _SEND_LOCK:
        out = _PROTO or sys.__stdout__
        out.write(line + "\n")
        out.flush()


def log(line: str) -> None:
    send({"id": _CURRENT["id"], "event": "log", "line": str(line)})


def progress(step: int, total: int | None = None, message: str = "") -> None:
    send({"id": _CURRENT["id"], "event": "progress", "step": int(step),
          "total": total if total is not None else _CURRENT["total"], "message": message})


# ------------------------------------------------------------------ state

class _State:
    manager = None                              # iopaint ModelManager
    manager_key = None                          # the load options it was built with
    manager_name = None
    plugins: dict = {}                          # kind -> (key, instance)


STATE = _State()


def _torch():
    import torch
    return torch


def torch_gc() -> None:
    gc.collect()
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:                          # noqa: BLE001 - freeing memory must never raise
        pass


def resolve_device(name: str | None) -> str:
    torch = _torch()
    name = (name or "auto").lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise WorkerError("This environment has no usable GPU (CUDA). Choose the CPU device, or re-run the edit "
                          "environment setup in the Backend tab.")
    return name


# ------------------------------------------------------------------ image helpers

def load_rgb(path: str):
    """(rgb uint8 [H,W,3], alpha uint8 [H,W] or None), with the EXIF rotation applied like the browser does."""
    import numpy as np
    from PIL import Image, ImageOps

    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:                   # noqa: BLE001
        raise WorkerError(f"Couldn't open the image: {exc}") from None
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:                          # noqa: BLE001
        pass
    alpha = None
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        arr = np.array(rgba)
        alpha = arr[:, :, 3].copy()
        rgb = arr[:, :, :3].copy()
    else:
        rgb = np.array(img.convert("RGB"))
    return rgb, alpha


def load_mask(path: str, size_hw: tuple[int, int]):
    """The mask as uint8 [H,W], strictly 0/255, resized to the image (nearest) when its size differs."""
    import cv2
    import numpy as np
    from PIL import Image

    try:
        m = Image.open(path).convert("L")
    except Exception as exc:                   # noqa: BLE001
        raise WorkerError("The mask couldn't be read - paint over the area again.") from None
    h, w = size_hw
    if m.size != (w, h):
        m = m.resize((w, h), Image.NEAREST)
    arr = np.array(m)
    arr = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)[1]
    if not arr.any():
        raise WorkerError("The mask is empty - paint over the area first.")
    return arr


def save_png(arr, path: str) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path, format="PNG")


def concat_alpha(rgb, alpha):
    import cv2
    import numpy as np

    if alpha is None:
        return rgb
    if alpha.shape[:2] != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]))
    return np.concatenate((rgb, alpha[:, :, None]), axis=-1)


def _image_key(path: str) -> str:
    """Stable identity of the picture on disk, so SAM re-encodes the image only when it really changed."""
    st = os.stat(path)
    return hashlib.md5(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ handlers

def h_ping(req: dict) -> dict:
    return {"pid": os.getpid(), "python": sys.version.split()[0]}


def environment_report() -> dict:
    """What is installed and what works - the checklist behind Verify."""
    rep: dict = {"python": sys.version.split()[0], "platform": sys.platform, "modules": {}}

    def probe(label: str, importer) -> None:
        try:
            rep["modules"][label] = {"ok": True, "version": str(importer() or "")}
        except BaseException as exc:           # noqa: BLE001 - SystemExit included: IOPaint calls exit() on bad deps
            rep["modules"][label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

    def ver(name: str):
        def get():
            from importlib import metadata
            return metadata.version(name)
        return get

    probe("torch", lambda: _torch().__version__)
    for label, dist in (("iopaint", "iopaint"), ("diffusers", "diffusers"), ("transformers", "transformers"),
                        ("huggingface-hub", "huggingface-hub"), ("rembg", "rembg"), ("onnxruntime", "onnxruntime"),
                        ("opencv", "opencv-python"), ("Pillow", "Pillow"), ("numpy", "numpy"),
                        ("controlnet-aux", "controlnet-aux")):
        probe(label, ver(dist))
    # The imports that really matter: the ones IOPaint's model manager and plugins make when they start.
    probe("iopaint.model_manager", lambda: (__import__("iopaint.model_manager"), "ok")[1])
    probe("iopaint.plugins", lambda: (__import__("iopaint.plugins"), "ok")[1])
    cuda = {"available": False}
    try:
        torch = _torch()
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["torch_cuda"] = getattr(torch.version, "cuda", None)
        if cuda["available"]:
            props = torch.cuda.get_device_properties(0)
            cuda["name"] = props.name
            cuda["total_gb"] = round(props.total_memory / 1024 ** 3, 1)
    except Exception as exc:                   # noqa: BLE001
        cuda["error"] = f"{type(exc).__name__}: {exc}"[:200]
    rep["cuda"] = cuda
    return rep


def h_check(req: dict) -> dict:
    return environment_report()


def _hub_checkpoint(url: str) -> bool:
    from iopaint.helper import get_cache_path_by_url
    return os.path.exists(get_cache_path_by_url(url))


_REALESRGAN_FILES = {
    "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}
_FACE_FILES = {
    "gfpgan": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "restoreformer": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
}


def h_models(req: dict) -> dict:
    """Which models are on disk: erase / diffusion models (IOPaint's own scan) and the plugin weights."""
    from iopaint.download import scan_models

    scanned = [m.model_dump() for m in scan_models()]
    have = {m["name"] for m in scanned}
    plugins: dict = {}
    try:
        from iopaint.plugins.interactive_seg import SEGMENT_ANYTHING_MODELS
        for mid, info in SEGMENT_ANYTHING_MODELS.items():
            plugins["sam:" + mid] = _hub_checkpoint(info["url"])
    except Exception as exc:                   # noqa: BLE001
        log(f"Could not check the SAM models: {exc}")
    try:
        for mid, url in _REALESRGAN_FILES.items():
            plugins["realesrgan:" + mid] = _hub_checkpoint(url)
        for mid, url in _FACE_FILES.items():
            plugins["face:" + mid] = _hub_checkpoint(url)
        from iopaint.plugins.anime_seg import ANIME_SEG_MODELS
        plugins["removebg:anime-seg"] = _hub_checkpoint(ANIME_SEG_MODELS["url"])
    except Exception as exc:                   # noqa: BLE001
        log(f"Could not check the plugin models: {exc}")
    return {"scanned": scanned, "downloaded": sorted(have), "plugins": plugins,
            "loaded": STATE.manager_name, "loaded_plugins": {k: v[0] for k, v in STATE.plugins.items()}}


def h_download(req: dict) -> dict:
    """Download a model: an erase model, a Hugging Face diffusion model, or a plugin's weights."""
    kind = req.get("kind", "model")
    name = req.get("name", "")
    if not name:
        raise WorkerError("No model name given.")
    if kind == "model":
        if name == "cv2":
            return {"downloaded": True}
        from iopaint.download import cli_download_model
        log(f"Downloading {name} ...")
        cli_download_model(name)
        return {"downloaded": True}
    # plugin weights are fetched when the plugin starts
    device = resolve_device(req.get("device") or "cpu")
    _plugin(kind, name, device, no_half=bool(req.get("no_half")), bg_upscale=bool(req.get("bg_upscale")))
    return {"downloaded": True}


def _manager_key(opts: dict) -> tuple:
    return tuple((k, opts.get(k)) for k in catalog.MODEL_LOAD_KEYS)


def _clean_options(req: dict) -> dict:
    opts = dict(req.get("options") or {})
    spec = {o["key"]: o["default"] for o in catalog.ENGINE_OPTIONS}
    clean = {k: opts.get(k, spec[k]) for k in catalog.MODEL_LOAD_KEYS}
    clean["device"] = resolve_device(clean["device"])
    for k in catalog.MODEL_LOAD_KEYS:
        if k != "device":
            clean[k] = bool(clean[k])
    return clean


def _diffusion_callback(pipe, step: int, timestep: int, callback_kwargs: dict | None = None):
    progress(step + 1)
    return callback_kwargs if callback_kwargs is not None else {}


def _ensure_model_files(name: str, local_only: bool) -> None:
    from iopaint.download import cli_download_model, scan_models

    if name == "cv2" or name in {m.name for m in scan_models()}:
        return
    if local_only:
        raise WorkerError(f"The model '{name}' is not downloaded and Offline mode is on.")
    log(f"'{name}' is not on disk yet - downloading it (first use only) ...")
    cli_download_model(name)


def ensure_manager(name: str, opts: dict):
    """The ModelManager for `name` with these load options, loading, switching or rebuilding as needed."""
    import torch
    from iopaint.model_manager import ModelManager

    key = _manager_key(opts)
    if STATE.manager is not None and STATE.manager_key == key:
        if STATE.manager_name != name:
            _ensure_model_files(name, opts["local_files_only"])
            STATE.manager.scan_models()
            STATE.manager.switch(name)
            STATE.manager_name = name
            torch_gc()
        return STATE.manager
    STATE.manager = None                       # free the old one before building the new one
    STATE.manager_name = None
    torch_gc()
    _ensure_model_files(name, opts["local_files_only"])
    log(f"Loading {name} on {opts['device']} ...")
    STATE.manager = ModelManager(
        name=name, device=torch.device(opts["device"]), no_half=opts["no_half"], low_mem=opts["low_mem"],
        disable_nsfw=opts["disable_nsfw"], sd_cpu_textencoder=opts["cpu_textencoder"],
        local_files_only=opts["local_files_only"], cpu_offload=opts["cpu_offload"], callback=_diffusion_callback)
    STATE.manager_key = key
    STATE.manager_name = name
    return STATE.manager


def h_load_model(req: dict) -> dict:
    name = req.get("model") or "lama"
    ensure_manager(name, _clean_options(req))
    return {"loaded": name, "device": STATE.manager_key and dict(STATE.manager_key).get("device")}


def build_config(params: dict | None):
    """An iopaint InpaintRequest from the browser's settings; unknown names are rejected with a clear message."""
    from iopaint.schema import InpaintRequest

    params = dict(params or {})
    unknown = sorted(k for k in params if k not in catalog.PARAM_KEYS and k not in catalog.RESERVED_FIELDS)
    if unknown:
        raise WorkerError("Unknown setting(s): " + ", ".join(unknown))
    params = {k: v for k, v in params.items() if v is not None and k not in catalog.RESERVED_FIELDS}
    try:
        return InpaintRequest(**params)
    except Exception as exc:                   # noqa: BLE001 - pydantic's message names the bad field
        raise WorkerError(f"A setting is out of range: {exc}") from None


def h_inpaint(req: dict) -> dict:
    import cv2
    import numpy as np

    rgb, alpha = load_rgb(req["image_path"])
    mask = load_mask(req["mask_path"], rgb.shape[:2]) if req.get("mask_path") else None
    config = build_config(req.get("params"))
    model = req.get("model") or "lama"
    if mask is None:
        if not config.use_extender:
            raise WorkerError("No mask given.")
        mask = np.zeros(rgb.shape[:2], np.uint8)
    _CURRENT["total"] = int(config.sd_steps)
    manager = ensure_manager(model, _clean_options(req))
    out_bgr = manager(rgb, mask, config)
    out_rgb = cv2.cvtColor(out_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if alpha is not None and out_rgb.shape[:2] != rgb.shape[:2]:
        alpha = None                           # an extended picture is returned opaque
    out_rgb = concat_alpha(out_rgb, alpha)
    torch_gc()
    save_png(out_rgb, req["out_path"])
    return {"out_path": req["out_path"], "seed": int(config.sd_seed), "size": [int(out_rgb.shape[1]), int(out_rgb.shape[0])],
            "has_alpha": out_rgb.shape[2] == 4}


# ------------------------------------------------------------------ plugins

def _plugin(kind: str, model: str, device: str, no_half: bool = False, bg_upscale: bool = False,
            bg_model: str = ""):
    """A started plugin, reused while its settings stay the same (one per kind, so memory stays small)."""
    from iopaint.schema import Device

    key = (model, device, bool(no_half), bool(bg_upscale), bg_model)
    cached = STATE.plugins.get(kind)
    if cached and cached[0] == key:
        return cached[1]
    STATE.plugins.pop(kind, None)
    torch_gc()
    dev = Device(device)
    log(f"Starting {kind} ({model or 'default'}) on {device} ...")
    if kind == "sam":
        from iopaint.plugins.interactive_seg import InteractiveSeg
        inst = InteractiveSeg(model, dev)
    elif kind == "removebg":
        if model == "anime-seg":
            from iopaint.plugins.anime_seg import AnimeSeg
            inst = AnimeSeg()
        else:
            from iopaint.plugins.remove_bg import RemoveBG
            inst = RemoveBG(model, dev)
    elif kind == "realesrgan":
        from iopaint.plugins.realesrgan import RealESRGANUpscaler
        inst = RealESRGANUpscaler(model, dev, no_half=no_half)
    elif kind in ("gfpgan", "restoreformer"):
        upscaler = _plugin("realesrgan", bg_model or "RealESRGAN_x4plus_anime_6B", device, no_half) if bg_upscale else None
        if kind == "gfpgan":
            from iopaint.plugins.gfpgan_plugin import GFPGANPlugin
            inst = GFPGANPlugin(dev, upscaler=upscaler)
        else:
            from iopaint.plugins.restoreformer import RestoreFormerPlugin
            inst = RestoreFormerPlugin(dev, upscaler=upscaler)
    else:
        raise WorkerError(f"Unknown plugin '{kind}'.")
    STATE.plugins[kind] = (key, inst)
    return inst


def h_segment(req: dict) -> dict:
    """Click-to-select with Segment Anything: clicks are [[x, y, 1 (include) | 0 (exclude)], ...] in image pixels."""
    import numpy as np

    clicks = [[int(c[0]), int(c[1]), int(c[2])] for c in (req.get("clicks") or [])]
    if not clicks:
        raise WorkerError("Click on the picture first.")
    rgb, _ = load_rgb(req["image_path"])
    model = req.get("model") or "mobile_sam"
    device = resolve_device(req.get("device"))
    sam = _plugin("sam", model, device)
    mask = sam.forward(rgb, clicks, _image_key(req["image_path"]))
    save_png(np.asarray(mask, dtype=np.uint8), req["out_path"])
    torch_gc()
    return {"out_path": req["out_path"]}


def h_plugin(req: dict) -> dict:
    """Remove background / upscale / restore faces on a whole image; op "mask" returns a 0/255 mask instead."""
    import cv2
    import numpy as np
    from iopaint.schema import RunPluginRequest

    kind = req.get("plugin", "")
    op = req.get("op", "image")
    rgb, alpha = load_rgb(req["image_path"])
    device = resolve_device(req.get("device"))
    model = req.get("model") or ""
    no_half = bool(req.get("no_half"))
    if kind == "realesrgan" and not model:
        model = "RealESRGAN_x4plus_anime_6B"
    plugin = _plugin(kind, model, device, no_half=no_half, bg_upscale=bool(req.get("bg_upscale")),
                     bg_model=req.get("bg_model") or "")
    rpr = RunPluginRequest(name=kind, image="", clicks=[], scale=float(req.get("scale") or 2.0))
    if op == "mask":
        if not getattr(plugin, "support_gen_mask", False):
            raise WorkerError(f"{kind} cannot make a mask.")
        out = plugin.gen_mask(rgb, rpr)
        if out.ndim == 3 and out.shape[2] != 1:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        save_png(np.asarray(out, dtype=np.uint8).reshape(out.shape[:2]), req["out_path"])
        torch_gc()
        return {"out_path": req["out_path"], "has_alpha": False, "size": [int(out.shape[1]), int(out.shape[0])]}
    if not getattr(plugin, "support_gen_image", False):
        raise WorkerError(f"{kind} cannot produce an image.")
    out = plugin.gen_image(rgb, rpr)
    if out.shape[2] == 4:
        result = out                            # RGBA straight from the plugin (background removal)
    else:
        result = concat_alpha(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), alpha)
    save_png(result, req["out_path"])
    torch_gc()
    return {"out_path": req["out_path"], "has_alpha": result.shape[2] == 4,
            "size": [int(result.shape[1]), int(result.shape[0])]}


def h_unload(req: dict) -> dict:
    STATE.manager = None
    STATE.manager_key = None
    STATE.manager_name = None
    STATE.plugins.clear()
    torch_gc()
    return {"unloaded": True}


HANDLERS = {
    "ping": h_ping, "check": h_check, "models": h_models, "download": h_download, "load_model": h_load_model,
    "inpaint": h_inpaint, "segment": h_segment, "plugin": h_plugin, "unload": h_unload,
}


# ------------------------------------------------------------------ loop

def friendly_error(exc: BaseException) -> str:
    """A sentence a person can act on; the traceback goes to the log."""
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, WorkerError):
        return str(exc)
    low = text.lower()
    if "out of memory" in low:
        return ("Out of GPU memory. Close other GPU programs (is a training running?), choose the CPU device, turn on "
                "Low memory mode / offload, or pick a smaller model.")
    if isinstance(exc, SystemExit):
        return ("IOPaint stopped the operation (a model file is missing, damaged or could not be downloaded) - see "
                "the log; delete the damaged model and try again.")
    if isinstance(exc, ModuleNotFoundError):
        return f"A package is missing from the edit environment ({exc.name}). Re-run the setup in the Backend tab."
    return text[:600]


def handle(req: dict) -> dict:
    cmd = req.get("cmd")
    fn = HANDLERS.get(cmd)
    if fn is None:
        raise WorkerError(f"Unknown command '{cmd}'.")
    return fn(req)


def serve() -> None:
    global _PROTO
    _PROTO = sys.stdout
    sys.stdout = sys.stderr                    # IOPaint and its libraries print; keep stdout for the protocol only
    try:
        import iopaint                          # noqa: F401 - sets the memory-friendly environment, then:
        iopaint.fix_window_pytorch()
    except BaseException:                      # noqa: BLE001 - optional
        pass
    send({"event": "ready", "pid": os.getpid(), "python": sys.version.split()[0]})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except ValueError:
            send({"id": None, "ok": False, "error": "Bad request line."})
            continue
        rid = req.get("id")
        _CURRENT["id"], _CURRENT["total"] = rid, None
        if req.get("cmd") == "shutdown":
            send({"id": rid, "ok": True, "result": {"bye": True}})
            break
        try:
            send({"id": rid, "ok": True, "result": handle(req)})
        except BaseException as exc:           # noqa: BLE001 - SystemExit too: IOPaint calls exit(-1) on bad downloads
            if not isinstance(exc, WorkerError):
                traceback.print_exc()
            torch_gc_safe()
            send({"id": rid, "ok": False, "error": friendly_error(exc)})
        finally:
            _CURRENT["id"] = None


def torch_gc_safe() -> None:
    try:
        torch_gc()
    except Exception:                          # noqa: BLE001
        pass


def main(argv: list[str]) -> int:
    if "--check" in argv:
        sys.stdout.write(json.dumps(environment_report()))
        sys.stdout.flush()
        return 0
    serve()
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
