"""Natural-language captioning for Anima.

Anima's text encoder is a language model (Qwen3), so a caption that reads like a
sentence - "a girl with long blue hair in twintails, wearing a school uniform,
smiling in a classroom" - gives it more to work with than a tag list does. This
module produces those captions two ways:

* **from tags** - the taggers already installed (WD14, PixAI) plus the tags a
  person typed, grouped the way `tag_tools` groups them (subject, hair, eyes,
  clothing, pose, setting) and phrased into a readable sentence. Nothing to
  download, works offline, instant.
* **from a vision-language model** - a real description of what is in the
  picture, from a local model the user chooses and downloads.

The model runs in its own process (`native_backend/caption_worker.py`) started
with whichever interpreter has PyTorch - normally the training backend's own
environment, so this needs no second PyTorch download - and the process ends
with the batch, which releases the VRAM.

Captions are written with `captions.write_caption`, the same writer every other
part of the app uses, and a batch takes an undo snapshot first.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .. import config
from . import captions, datasets, procs, settings_store, tag_tools

# Qwen3-VL needs transformers 4.57+; bitsandbytes does the 4/8-bit loading that lets
# the 8B models run on an 8 GB card.
CAPTION_DEPS = ["transformers", "accelerate", "sentencepiece", "protobuf", "huggingface-hub",
                "pillow", "bitsandbytes"]
CAPTION_PIP = ["transformers>=4.57", "accelerate>=1.0", "sentencepiece", "protobuf", "huggingface-hub",
               "pillow", "bitsandbytes>=0.46.1"]
MIN_TRANSFORMERS = (4, 57)
MIN_TRANSFORMERS_35 = (5, 3)        # Qwen3.5 models: run in their own environment (caption_env), never the training backend
MIN_BITSANDBYTES = (0, 46)         # transformers refuses 4/8-bit loading of Qwen3-VL below 0.46.1

DEFAULT_MODEL = "joycaption-beta"

# VRAM profiles the picker offers. "auto" reads the card; a number is the budget
# (in GB) the loader plans for, so an 8 GB profile behaves the same on a 24 GB card.
VRAM_PROFILES = ["auto", "8", "10", "12", "16", "24"]

# Every model here is happy to describe adult art: JoyCaption was built for it, the
# rest are "abliterated" (refusal direction removed) Qwen VL checkpoints. `tier` is the
# smallest card the model is comfortable on:
#   low  - runs in 4-bit inside 8 GB with room to spare
#   mid  - 4-bit fits 8 GB; 8-bit wants ~12 GB
#   high - bf16 wants 16-24 GB; 4-bit still fits 8-10 GB
# `family` decides the chat format; `skip_modules` stay 16-bit when quantizing: the
# vision tower and projector are small, but quantizing them wrecks what the model sees.
# `size_gb` is the bf16 weight size; estimated_vram() derives the rest.
_BASE_PROMPT = ("Describe this image in {n} for an image-generation training caption. Name the "
                "subject, their hair and eye colour, what they are wearing (or not wearing), their "
                "expression, pose and body, and where they are. Describe what is actually shown, "
                "including adult or explicit content, in plain factual words. Reply with the "
                "description only - no preamble, no commentary, no lists.")

MODELS: dict[str, dict] = {
    "qwen2.5-vl-3b-uncensored": {
        "label": "Qwen2.5-VL 3B (uncensored)",
        "repo": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
        "kind": "chat", "family": "qwen2vl", "tier": "low",
        "prompt": _BASE_PROMPT.format(n="two or three detailed sentences"),
        "skip_modules": ["visual", "lm_head"],
        "size_gb": 7.5,
        "note": "Smallest and fastest. ~4 GB in 4-bit, so an 8 GB card has room for big images.",
    },
    "qwen3-vl-4b-uncensored": {
        "label": "Qwen3-VL 4B (uncensored)",
        "repo": "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
        "kind": "chat", "family": "qwen3vl", "tier": "low",
        "prompt": _BASE_PROMPT.format(n="three or four detailed sentences"),
        "skip_modules": ["visual", "lm_head"],
        "size_gb": 8.9,
        "note": "Newer and sharper than the 2.5 3B, still comfortable in 8 GB (4-bit).",
    },
    "joycaption-beta": {
        "label": "JoyCaption Beta One",
        "repo": "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "kind": "chat", "family": "llava", "tier": "mid",
        "system": "You are a helpful image captioner.",
        "prompt": ("Write a long descriptive caption for this image in a formal tone. Describe the "
                   "subject's appearance, clothing or nudity, body, pose, expression, the setting and "
                   "the lighting, including any sexual content, in explicit plain words. Do not "
                   "mention the image itself or use lists."),
        "skip_modules": ["vision_tower", "multi_modal_projector", "lm_head"],
        "size_gb": 16.1,
        "note": "Built for captioning adult and anime art for training sets. ~6 GB in 4-bit, so 8 GB works.",
    },
    "qwen3-vl-8b-uncensored": {
        "label": "Qwen3-VL 8B (uncensored)",
        "repo": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
        "kind": "chat", "family": "qwen3vl", "tier": "mid",
        "prompt": _BASE_PROMPT.format(n="four to six detailed sentences"),
        "skip_modules": ["visual", "lm_head"],
        "size_gb": 17.5,
        "note": "Most detailed general model that still fits: 4-bit on 8 GB, 8-bit on 12 GB.",
    },
    "qwen2.5-vl-7b-uncensored": {
        "label": "Qwen2.5-VL 7B (uncensored)",
        "repo": "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated",
        "kind": "chat", "family": "qwen2vl", "tier": "high",
        "prompt": _BASE_PROMPT.format(n="three to five detailed sentences"),
        "skip_modules": ["visual", "lm_head"],
        "size_gb": 16.6,
        "note": "Proven 7B. bf16 on 24 GB, 8-bit on 12-16 GB, 4-bit on 8 GB.",
    },
    "qwen3-vl-8b-caption": {
        "label": "Qwen3-VL 8B (caption-tuned, uncensored)",
        "repo": "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it",
        "kind": "chat", "family": "qwen3vl", "tier": "high",
        "prompt": _BASE_PROMPT.format(n="a long, richly detailed paragraph"),
        "skip_modules": ["visual", "lm_head"],
        "size_gb": 17.5,
        "note": "Caption-tuned 8B for the longest, most thorough descriptions. Best on 16-24 GB.",
    },
    "qwen3.5-2b-caption": {
        "label": "Qwen3.5 2B (caption-tuned, uncensored)",
        "repo": "prithivMLmods/Gliese-Qwen3.5-2B-Abliterated-Caption",
        "kind": "chat", "family": "qwen35", "tier": "low", "env": "caption5",
        "prompt": _BASE_PROMPT.format(n="a detailed paragraph"),
        "skip_modules": ["visual", "lm_head"],
        "size_gb": 4.8,
        "note": "Newest-generation small captioner. Runs in its own environment (transformers 5.x).",
    },
    "qwen3.5-4b-caption": {
        "label": "Qwen3.5 4B (caption-tuned, uncensored)",
        "repo": "prithivMLmods/Gliese-Qwen3.5-4B-Abliterated-Caption",
        "kind": "chat", "family": "qwen35", "tier": "low", "env": "caption5",
        "prompt": _BASE_PROMPT.format(n="a long, detailed paragraph"),
        "skip_modules": ["visual", "lm_head"],
        "size_gb": 9.0,
        "note": "Caption-tuned 4B: 4-bit fits 8 GB. Runs in its own environment (transformers 5.x).",
    },
    "qwen3.5-9b-caption": {
        "label": "Qwen3.5 9B (caption-tuned, uncensored)",
        "repo": "prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption",
        "kind": "chat", "family": "qwen35", "tier": "mid", "env": "caption5",
        "prompt": _BASE_PROMPT.format(n="a long, richly detailed paragraph"),
        "skip_modules": ["visual", "lm_head"],
        "size_gb": 19.0,
        "note": "The strongest caption-tuned model here. 4-bit fits a 10 GB card. Runs in its own environment (transformers 5.x).",
    },
}

_RUNTIME_OVERHEAD_GB = 1.6      # CUDA context + activations for one image, at the default image size


def estimated_vram(info: dict, mode: str) -> float:
    """Rough VRAM needed to caption with `mode` ("4bit", "8bit" or "bf16").

    4-bit NF4 stores ~0.5 byte per weight but the vision tower and lm_head stay
    16-bit (hence 0.30, not 0.25); 8-bit is ~1 byte per weight.
    """
    size = float(info.get("size_gb") or 0)
    factor = {"4bit": 0.30, "8bit": 0.55}.get(mode, 1.0)
    return round(size * factor + _RUNTIME_OVERHEAD_GB, 1)


def pick_mode(info: dict, vram_gb: float, quant: str = "auto") -> str:
    """Which precision to load: the best one that leaves headroom on `vram_gb`."""
    if quant in ("4bit", "8bit", "bf16"):
        return quant
    for mode in ("bf16", "8bit", "4bit"):
        if estimated_vram(info, mode) + 1.0 <= vram_gb:
            return mode
    return "4bit"


# Weight files a repo may carry that transformers would never load - skipping them
# keeps a 7 GB model from turning into a 15 GB download. ".bin" is only skipped
# when the repo also ships safetensors (decided per repo at download time).
_IGNORE = ["*.gguf", "*.pth", "*.h5", "*.msgpack", "*.onnx", "*.ckpt", "*.tflite", "*.mlmodel",
           "original/*", "*.md"]

_progress: dict[str, dict] = {}
_progress_lock = threading.Lock()
_deps_cache: dict = {"at": 0.0, "data": None}


class CaptionerError(RuntimeError):
    pass


# --------------------------------------------------------------------- models

def available_models() -> list[str]:
    return list(MODELS)


def model_info(name: str) -> dict:
    if name not in MODELS:
        raise CaptionerError(f"Unknown caption model: {name}")
    return MODELS[name]


def _env5_ready() -> bool:
    from . import caption_env
    return caption_env.is_ready()


def model_summary(name: str) -> dict:
    info = MODELS[name]
    return {
        "name": name, "label": info["label"], "repo": info["repo"], "size_gb": info["size_gb"],
        "note": info["note"], "tier": info["tier"], "downloaded": is_downloaded(name),
        "vram": {m: estimated_vram(info, m) for m in ("4bit", "8bit", "bf16")},
        "fits": {str(v): pick_mode(info, v) for v in (8, 10, 12, 16, 24)},
        "env": info.get("env", ""),
        "env_ready": info.get("env") != "caption5" or _env5_ready(),
    }


def model_dir(name: str) -> Path:
    return config.MODELS_CACHE_DIR / name


def _set_progress(name: str, **kw) -> None:
    with _progress_lock:
        _progress.setdefault(name, {"status": "idle", "bytes": 0, "total": 0, "error": None,
                                    "file": ""}).update(kw)


def download_progress(name: str) -> dict:
    with _progress_lock:
        return dict(_progress.get(name, {"status": "idle", "bytes": 0, "total": 0,
                                         "error": None, "file": ""}))


def is_downloaded(name: str) -> bool:
    """A complete copy on disk. A half-finished snapshot has no config.json yet."""
    d = model_dir(name)
    if not d.is_dir() or not (d / "config.json").exists():
        return False
    weights = list(d.glob("*.safetensors")) + list(d.glob("*.bin"))
    return any(w.stat().st_size > 1024 and not w.name.endswith(".incomplete") for w in weights)


def download_async(name: str) -> dict:
    if name not in MODELS:
        raise CaptionerError(f"Unknown caption model: {name}")
    if MODELS[name].get("env") == "caption5" and not _env5_ready():
        raise CaptionerError("Set up the Qwen3.5 environment first (the button under the model list) - the download "
                             "runs inside it.")
    if download_progress(name)["status"] == "downloading":
        return download_progress(name)
    _set_progress(name, status="downloading", bytes=0, total=0, error=None, file="")
    threading.Thread(target=_download_worker, args=(name,), daemon=True, name=f"caption-dl-{name}").start()
    return download_progress(name)


def _download_worker(name: str) -> None:
    try:
        _download(name)
        _set_progress(name, status="done", file="")
    except Exception as exc:                              # noqa: BLE001 - surfaced in the UI
        _set_progress(name, status="error", error=f"{type(exc).__name__}: {exc}", file="")


_DOWNLOAD_SCRIPT = r'''
import fnmatch, sys
from huggingface_hub import HfApi, snapshot_download
repo, dest, ignore_arg = sys.argv[1], sys.argv[2], sys.argv[3]
ignore = [p for p in ignore_arg.split(",") if p]
try:
    files = HfApi().list_repo_files(repo)
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")
    keep = [f for f in files
            if not any(fnmatch.fnmatch(f, pat) for pat in ignore) and not f.endswith(".incomplete")]
    sib = HfApi().model_info(repo, files_metadata=True).siblings or []
    total = sum((s.size or 0) for s in sib if s.rfilename in keep)
except Exception:
    total = 0
print("@@TOTAL", int(total), flush=True)
snapshot_download(repo_id=repo, local_dir=dest, ignore_patterns=ignore, max_workers=4)
print("@@DONE", flush=True)
'''


def hf_token() -> str:
    return str(settings_store.get("hf_token", "") or "").strip()


def explain_download_failure(tail: list[str], code: int, size_gb: float | None = None) -> str:
    """A plain-words reason for a failed download, from the last lines the downloader printed.

    Only a real "No module named" means the dependencies are missing. A traceback that merely mentions
    huggingface_hub (every network or disk failure does) is NOT a missing dependency."""
    text = "\n".join(tail)
    low = text.lower()
    last = tail[-1] if tail else f"exit code {code}"
    if "no module named" in low:
        return ("The caption dependencies are not installed in the interpreter that would run the model - "
                "install them from the Backend tab's Feature dependencies card first. (" + last + ")")
    if "no space left" in low or "errno 28" in low or "not enough space" in low:
        need = f" This model needs about {size_gb:.0f} GB free." if size_gb else ""
        return "The disk is full, so the download stopped." + need + " Free some space and press download again - it resumes."
    if "401" in low or "403" in low or "gated" in low or "access to model" in low:
        have = bool(hf_token())
        steps = ("Your saved token was not accepted for this model - open the model's page on huggingface.co, press "
                 "\"Agree and access repository\" while logged in to the same account, then try again."
                 if have else
                 "This model is gated. 1) Open its page on huggingface.co while logged in and press \"Agree and access "
                 "repository\". 2) Create a free Read token at huggingface.co/settings/tokens. 3) Paste it in App settings "
                 "-> Hugging Face token, then press download again.")
        return steps + " Detail: " + last
    if "404" in low or "repositorynotfound" in low or "repository not found" in low:
        return "That model repository was not found on Hugging Face (it may have been renamed or removed). Detail: " + last
    if any(k in low for k in ("connection", "timed out", "timeout", "ssl", "name resolution", "max retries", "reset by peer", "readerror", "remoteprotocol")):
        return ("The connection to Hugging Face dropped during the download. Press download again - it resumes where it stopped. "
                "Detail: " + last)
    return f"The download failed: {last}"


def _download(name: str) -> None:
    """Fetch the model weights.

    The download runs as a subprocess of the interpreter that will later load
    the model (normally the training backend's venv) rather than this process:
    the app's own environment does not necessarily have huggingface_hub, while
    the chosen interpreter does as soon as the caption dependencies are
    installed. Progress is still read off the destination folder.
    """
    python, _label = interpreter(name)
    info = MODELS[name]
    dest = model_dir(name)
    dest.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()

    def watch() -> None:
        # huggingface_hub gives no byte callback, so watch the folder instead:
        # it streams into a .cache subfolder and lands files here as they finish.
        while not stop.wait(0.6):
            used = 0
            newest = ""
            newest_at = 0.0
            for p in dest.rglob("*"):
                try:
                    if p.is_file():
                        used += p.stat().st_size
                        if p.suffix == ".incomplete" and p.stat().st_mtime > newest_at:
                            newest, newest_at = p.name, p.stat().st_mtime
                except OSError:
                    continue
            _set_progress(name, bytes=int(used), file=newest.split(".")[0])
            total = download_progress(name).get("total") or 0
            if total and used >= total:
                break

    watcher = threading.Thread(target=watch, daemon=True, name=f"caption-dl-watch-{name}")
    watcher.start()
    try:
        args = [python, "-c", _DOWNLOAD_SCRIPT, info["repo"], str(dest), ",".join(_IGNORE)]
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        env = dict(os.environ)
        tok = hf_token()
        if tok:
            env["HF_TOKEN"] = tok             # gated models (e.g. the caption-tuned Qwen3-VL) need a login
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, creationflags=creationflags, env=env)
        procs.register(proc)                  # closing the app stops the download too
        tail: list[str] = []
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("@@TOTAL "):
                try:
                    _set_progress(name, total=int(line.split(" ", 1)[1]))
                except (IndexError, ValueError):
                    pass
            elif line.startswith("@@"):
                continue
            elif line:
                tail.append(line)
                del tail[:-8]
        code = proc.wait()
        if code != 0:
            raise CaptionerError(explain_download_failure(tail, code, info.get("size_gb")))
    finally:
        stop.set()


# ------------------------------------------------------- which interpreter/deps

def interpreter(model: str | None = None) -> tuple[str, str]:
    """(python executable, human label) that will load the model.

    The training backend's environment is preferred: it already has PyTorch, and
    it is the CUDA build, so captioning costs no extra download and runs on the
    GPU. Falling back to our own interpreter means installing PyTorch here.

    Qwen3.5 models need transformers 5.x, so they use their own environment (caption_env) instead.
    """
    if model and MODELS.get(model, {}).get("env") == "caption5":
        from . import caption_env
        return str(caption_env.venv_python()), "the Qwen3.5 caption environment"
    from . import backend_setup
    try:
        if backend_setup.is_installed():
            return str(backend_setup.venv_python()), "the backend environment"
    except Exception:                                     # noqa: BLE001
        pass
    return sys.executable, "Anima Studio's own environment"


# pip package names vs the module you actually import, for the ones that differ
_DEP_IMPORTS = {"protobuf": "google.protobuf", "huggingface-hub": "huggingface_hub", "pillow": "PIL"}


def deps_status(force: bool = False, model: str | None = None) -> dict:
    """Whether the chosen interpreter can caption, checked by asking it."""
    env5 = bool(model) and MODELS.get(model, {}).get("env") == "caption5"
    python, label = interpreter(model)
    if env5:
        from . import caption_env
        if not caption_env.is_ready():
            return {"python": python, "label": label, "checked": True, "ok": False, "env": "caption5",
                    "missing": ["the Qwen3.5 environment (not set up yet)"], "torch": None, "error": ""}
    if not force and not env5 and _deps_cache["data"] is not None and time.time() - _deps_cache["at"] < 300:
        return _deps_cache["data"]

    min_tf = MIN_TRANSFORMERS_35 if env5 else MIN_TRANSFORMERS
    required = {p: _DEP_IMPORTS.get(p, p) for p in CAPTION_DEPS}
    probe = ("import importlib.util as u, json, sys\n"
             f"required = {required!r}\n"
             "have = {}\n"
             "for name, mod in required.items():\n"
             "    try:\n"
             "        have[name] = u.find_spec(mod) is not None\n"
             "    except Exception:\n"
             "        have[name] = False\n"
             "try:\n"
             "    import torch\n"
             "    t = {'version': torch.__version__, 'cuda': bool(torch.cuda.is_available())}\n"
             "except Exception as exc:\n"
             "    t = None\n"
             "import importlib.metadata as md\n"
             "def ver(n):\n"
             "    try:\n"
             "        return md.version(n)\n"
             "    except Exception:\n"
             "        return ''\n"
             "tv = ver('transformers')\n"
             "bv = ver('bitsandbytes')\n"
             "berr = ''\n"
             "if have.get('bitsandbytes'):\n"
             "    try:\n"
             "        import bitsandbytes\n"
             "    except BaseException as exc:\n"
             "        berr = (type(exc).__name__ + ': ' + str(exc)).strip().splitlines()[0][:200]\n"
             "print('@@DEPS ' + json.dumps({'modules': have, 'torch': t, 'transformers': tv,\n"
             "                              'bitsandbytes': bv, 'bnb_error': berr, 'accelerate': ver('accelerate')}))\n")
    data = {"python": python, "label": label, "checked": True, "ok": False,
            "missing": list(CAPTION_DEPS), "torch": None, "error": ""}
    try:
        r = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=180,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        line = next((l for l in (r.stdout or "").splitlines() if l.startswith("@@DEPS ")), None)
        if line is None:
            data["error"] = ((r.stderr or r.stdout or "").strip().splitlines() or ["no answer"])[-1]
        else:
            parsed = json.loads(line[len("@@DEPS "):])
            missing = [m for m, ok in parsed["modules"].items() if not ok]
            tv = parsed.get("transformers") or ""
            old = False
            if tv and "transformers" not in missing:
                try:
                    parts = tuple(int(x) for x in re.findall(r"\d+", tv)[:2])
                    old = parts < min_tf
                except ValueError:
                    old = False
            if old:
                missing = missing + [f"transformers>={min_tf[0]}.{min_tf[1]} (have {tv})"]
            bv = parsed.get("bitsandbytes") or ""
            if bv and "bitsandbytes" not in missing:
                try:
                    if tuple(int(x) for x in re.findall(r"\d+", bv)[:2]) < MIN_BITSANDBYTES:
                        missing = missing + [f"bitsandbytes>=0.46.1 (have {bv})"]
                except ValueError:
                    pass
            # find_spec only proves the package is on disk; a bitsandbytes that is installed but will not
            # import (wrong CUDA build, missing library) is what made "installed" dependencies read as missing
            # - and only on the 8B models, the ones big enough to need 4-bit loading.
            if parsed.get("bnb_error") and "bitsandbytes" not in missing:
                missing = missing + [f"bitsandbytes (installed but will not load: {parsed['bnb_error']})"]
            data.update(missing=missing, torch=parsed.get("torch"), ok=not missing, transformers=tv,
                        bitsandbytes=bv, bnb_error=parsed.get("bnb_error") or "")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        data["error"] = str(exc)

    if not env5:
        _deps_cache.update(at=time.time(), data=data)
    return data


# ------------------------------------------------------ tags -> a real sentence

# Quality and meta tags are library metadata, not something you can see in the
# picture, so they are noise in a sentence.
_META_NOISE = {
    "best quality", "high quality", "highres", "absurdres", "lowres", "worst quality", "low quality",
    "masterpiece", "very aesthetic", "aesthetic", "bad quality", "jpeg artifacts", "artist name",
    "signature", "watermark", "web address", "username", "commentary", "commentary request",
    "translated", "english text", "text focus", "anime style", "anime coloring", "anime screencap",
    "traditional media", "sketch", "monochrome", "greyscale", "grayscale",
}

_SUBJECT = {
    "1girl": "a girl", "1boy": "a boy", "1other": "a character", "solo": "", "solo focus": "",
    "2girls": "two girls", "2boys": "two boys", "3girls": "three girls", "3boys": "three boys",
    "multiple girls": "several girls", "multiple boys": "several boys", "no humans": "no people",
    "6+girls": "many girls", "6+boys": "many boys",
}

_EXPRESSION = {
    "smile": "smiling", "blush": "blushing", "cry": "crying", "crying": "crying", "laugh": "laughing",
    "angry": "looking angry", "sad": "looking sad", "surprised": "looking surprised",
    "open mouth": "with an open mouth", "closed mouth": "with a closed mouth",
    "closed eyes": "with closed eyes", "half-closed eyes": "with half-closed eyes",
    "embarrassed": "looking embarrassed", "confused": "looking confused", "wink": "winking",
    "looking at viewer": "looking at the viewer", "looking away": "looking away",
    "looking back": "looking back", "looking down": "looking down", "looking up": "looking up",
    "looking to the side": "looking to the side", "parted lips": "with parted lips",
    "grin": "grinning", "smirk": "smirking", "pout": "pouting", "tears": "in tears",
    "shy": "looking shy", "serious": "with a serious expression", "annoyed": "looking annoyed",
}

_SETTING = {
    "day": "during the day", "night": "at night", "morning": "in the morning", "evening": "in the evening",
    "sunset": "at sunset", "sunrise": "at sunrise", "indoors": "indoors", "outdoors": "outdoors",
    "underwater": "underwater", "white background": "on a white background",
    "grey background": "on a grey background", "gray background": "on a grey background",
    "black background": "on a black background", "simple background": "on a plain background",
    "transparent background": "on a transparent background", "gradient background": "on a gradient background",
    "sky": "under an open sky", "cloudy sky": "under a cloudy sky", "starry sky": "under a starry sky",
    "from above": "seen from above", "from below": "seen from below", "from behind": "seen from behind",
    "from side": "seen from the side", "from side,": "seen from the side",
    "full body": "shown full body", "upper body": "shown from the waist up",
    "cowboy shot": "shown from the thighs up", "portrait": "in a portrait crop",
    "close-up": "in close-up",
}

# How the picture is framed reads best straight after the subject ("A girl,
# shown full body, wearing ..."), so these are lifted out of the setting clause.
_FRAMING = ("full body", "upper body", "lower body", "cowboy shot", "portrait", "close-up",
            "from above", "from below", "from behind", "from side", "from side,")


def _article(phrase: str) -> str:
    """'white shirt' -> 'a white shirt', 'thighhighs' -> 'thighhighs'."""
    words = phrase.split()
    if not words:
        return phrase
    if words[0] in ("a", "an", "the", "some", "her", "his", "their"):
        return phrase
    last = words[-1]
    if last.endswith("s") and not last.endswith(("ss", "us", "is", "ous")):
        return phrase                                   # already plural: sleeves, shorts, glasses
    return ("an " if words[0][:1].lower() in "aeiou" else "a ") + phrase


def _conjoin(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _gerund(tag: str) -> str:
    if tag in _EXPRESSION:
        return _EXPRESSION[tag]
    if tag.endswith("ing"):
        return tag
    if tag.endswith("e") and not tag.endswith("ee"):
        return tag[:-1] + "ing"
    return tag


def _hair_clause(tags: list[str]) -> str:
    """'long hair', 'blue hair', 'twintails' -> 'with long blue hair in twintails'."""
    modifiers, styles = [], []
    for tag in tags:
        if tag.endswith(" hair"):
            modifiers.extend(tag[: -len(" hair")].split())
        elif tag in ("hair", "hair focus"):
            continue
        else:
            styles.append(tag)
    core = " ".join(modifiers)
    if core and styles:
        return f"with {core} hair in {_conjoin(styles)}"
    if core:
        return f"with {core} hair"
    if styles:
        return f"with her hair in {_conjoin(styles)}"
    return ""


def _eyes_clause(tags: list[str]) -> str:
    colours, extras = [], []
    for tag in tags:
        if tag.endswith(" eyes"):
            colours.extend(tag[: -len(" eyes")].split())
        else:
            extras.append(tag)
    core = " ".join(colours)
    if core and extras:
        return f"with {core} eyes and {_conjoin(extras)}"
    if core:
        return f"with {core} eyes"
    if extras:
        return f"with {_conjoin(extras)}"
    return ""


def caption_from_tags(tags: list[str], *, trigger_word: str = "", drop_quality: bool = True) -> str:
    """Phrase a tag list as a readable sentence.

    The tags come from whatever produced them (WD14, PixAI, or a person typing),
    so this only claims to be readable - it groups them the way `tag_tools`
    groups them and phrases each group, rather than pretending to understand the
    picture. An empty result means nothing was left worth saying.
    """
    grouped: dict[str, list[str]] = {}
    for tag in captions.normalize_tags([str(t) for t in tags]):
        if drop_quality and tag in _META_NOISE:
            continue
        # framing and background tags read as setting ("seen from the side"),
        # whatever bucket the shared tag rules would have put them in
        bucket = "setting" if (tag in _SETTING or tag.endswith(" background")) else tag_tools.bucket_of(tag)
        if drop_quality and bucket == "meta":
            continue                     # rankings, watermarks, screencap notes: not the picture
        grouped.setdefault(bucket, []).append(tag)

    if not grouped:
        return ""

    parts: list[str] = []
    subject = _conjoin([_SUBJECT[t] for t in grouped.get("subject", []) if t in _SUBJECT])
    if not subject:
        unknown_subject = [t for t in grouped.get("subject", []) if t not in _SUBJECT]
        subject = _conjoin([_article(t) for t in unknown_subject]) or "a character"
    head = subject[0].upper() + subject[1:] if subject else "A character"

    # the hair clause hangs off the subject, so it reads as one phrase
    hair = _hair_clause(grouped.get("hair", []))
    parts.append(f"{head} {hair}" if hair else head)

    framing = [_SETTING.get(t, t) for t in grouped.get("setting", []) if t in _FRAMING]
    if framing:
        parts.append(_conjoin(framing))

    eyes = _eyes_clause(grouped.get("eyes", []))
    if eyes:
        parts.append(eyes)
    body = grouped.get("body", [])
    if body:
        parts.append("with " + _conjoin([_article(t) for t in body]))

    clothing = grouped.get("clothing", [])
    if clothing:
        parts.append("wearing " + _conjoin([_article(t) for t in clothing]))

    expression = [_gerund(t) for t in grouped.get("expression", [])]
    if expression:
        parts.append(_conjoin(expression))

    pose = [_pose(t) for t in grouped.get("pose", [])]
    if pose:
        parts.append(_conjoin(pose))

    setting = grouped.get("setting", [])
    backgrounds = [t for t in setting if t.endswith(" background")]
    others = [t for t in setting if t not in backgrounds and t not in _FRAMING]
    background = _background_clause(backgrounds)
    clauses = [_SETTING.get(t, f"in {_article(t)}") for t in others]
    if background:
        clauses.append(background)
    if clauses:
        parts.append(_conjoin(clauses))

    leftovers = [t for t in grouped.get("other", []) if t not in _SUBJECT]
    if leftovers:
        parts.append(_conjoin(leftovers))

    # kept ranking tags go last, so they read as a footnote rather than a pose
    meta = [t for t in grouped.get("meta", []) if t not in _SUBJECT]
    if meta:
        parts.append(_conjoin(meta))

    sentence = ", ".join(p.strip() for p in parts if p and p.strip())
    if not sentence:
        return ""
    if trigger_word:
        # the trigger goes first, so the sentence after it reads as a continuation
        return f"{trigger_word}, {sentence[0].lower()}{sentence[1:]}."
    return sentence + "."


def _pose(tag: str) -> str:
    """'holding sword' -> 'holding a sword'; 'standing' stays 'standing'."""
    for verb in ("holding ", "carrying ", "wearing ", "eating ", "reading ", "playing "):
        if tag.startswith(verb):
            return verb + _article(tag[len(verb):])
    return tag


def _background_clause(tags: list[str]) -> str:
    """'grey background' + 'gradient background' -> 'on a grey gradient background'."""
    words: list[str] = []
    for tag in tags:
        for word in tag[: -len(" background")].split():
            word = "plain" if word in ("simple", "flat") else word
            if word not in words:
                words.append(word)
    if not words:
        return ""
    if len(words) > 1:
        words = [w for w in words if w != "plain"] or words       # "plain" adds nothing to "grey gradient"
    return "on a " + " ".join(words) + " background"


def caption_from_caption_text(text: str, **kw) -> str:
    """Turn an existing comma-separated caption into a sentence."""
    return caption_from_tags(captions._split(text), **kw)


def _without_trigger(tags: list[str], trigger: str) -> list[str]:
    """The tags minus the trigger word, so a kept caption does not say it twice."""
    drop = trigger.strip().lower().replace("_", " ")
    if not drop:
        return tags
    return [t for t in tags if t.strip().lower().replace("_", " ") != drop]


def _compose_keep(tags: list[str], sentence: str, trigger: str) -> str:
    """`keep` write mode: the trigger (unless the tags already carry it),
    the tags that were there, then the sentence."""
    seen = {t.strip().lower().replace("_", " ") for t in tags}
    parts = [trigger] if trigger and trigger.strip().lower().replace("_", " ") not in seen else []
    parts.extend(t for t in tags if t.strip())
    sentence = (sentence or "").strip()
    if sentence:
        if parts:                             # mid-caption now, so it reads as a continuation
            sentence = sentence[0].lower() + sentence[1:]
        parts.append(sentence)
    return ", ".join(parts)


def preview_caption(text: str, *, trigger_word: str = "", drop_quality: bool = True,
                    write: str = "replace") -> str:
    """What a run would write for one caption, without touching the folder."""
    tags = captions._split(text)
    if write == "keep":
        sentence = caption_from_tags(_without_trigger(tags, trigger_word), drop_quality=drop_quality)
        return _compose_keep(tags, sentence, trigger_word)
    return caption_from_tags(tags, trigger_word=trigger_word, drop_quality=drop_quality)


# ------------------------------------------------------------------ batch jobs

_JOB: dict = {"status": "idle", "done": 0, "total": 0, "current": "", "results": [],
              "captioned": 0, "skipped": 0, "failed": 0, "error": "", "source": "",
              "started": None, "finished": None, "device": "", "loader": "", "quant": "", "model": ""}
_JOB_LOCK = threading.Lock()
_CANCEL = threading.Event()
_MAX_RESULTS = 300
_proc_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def _job_update(**kw) -> None:
    with _JOB_LOCK:
        _JOB.update(kw)


def _job_add_result(result: dict) -> None:
    with _JOB_LOCK:
        _JOB["results"].append(result)
        if len(_JOB["results"]) > _MAX_RESULTS:
            del _JOB["results"][: len(_JOB["results"]) - _MAX_RESULTS]


def run_status() -> dict:
    with _JOB_LOCK:
        return json.loads(json.dumps(_JOB))


def cancel() -> dict:
    _CANCEL.set()
    with _proc_lock:
        proc = _proc
    if proc is not None and proc.poll() is None:
        from .procs import kill_tree
        kill_tree(proc)
    return run_status()


def run(options: dict) -> dict:
    """Start a captioning batch in the background; returns the initial status."""
    if run_status()["status"] == "running":
        raise CaptionerError("A captioning batch is already running.")
    folder = (options.get("folder") or "").strip().strip('"')
    if not folder or not Path(folder).is_dir():
        raise CaptionerError("Pick the dataset folder to caption first.")
    source = options.get("source", "tags")
    if source not in ("tags", "vlm"):
        raise CaptionerError("Unknown caption source.")
    write = options.get("write", "replace")
    if write not in ("replace", "keep"):
        raise CaptionerError("Unknown caption style.")
    if source == "vlm":
        model = options.get("model") or DEFAULT_MODEL
        if model not in MODELS:
            raise CaptionerError(f"Unknown caption model: {model}")
        if not is_downloaded(model):
            raise CaptionerError(f"{MODELS[model]['label']} is not downloaded yet - press Download first.")

    # Claim the job under the same lock run_status() reads under, so two
    # quick batches (two tabs, double-click) can't both pass the check above
    # and then run over each other's results.
    with _JOB_LOCK:
        if _JOB.get("status") == "running":
            raise CaptionerError("A captioning batch is already running.")
        _JOB["status"] = "running"
    _CANCEL.clear()
    _job_update(done=0, total=0, current="", results=[], captioned=0,
                skipped=0, failed=0, error="", source=source, started=time.time(), finished=None,
                device="", loader="", model=options.get("model", ""))
    threading.Thread(target=_run_job, args=(options,), daemon=True, name="caption-batch").start()
    return run_status()


def _selected_images(folder: str, mode: str, limit: int) -> list[dict]:
    images = datasets.list_images(folder)
    if mode != "all":
        images = [i for i in images if not (i.get("caption") or "").strip()]
    if limit and limit > 0:
        images = images[:limit]
    return images


def _snapshot(folder: str, label: str) -> None:
    try:
        tag_tools.snapshot(folder, label)
    except Exception:                                     # noqa: BLE001 - undo is a bonus, not a requirement
        pass


def _run_job(options: dict) -> None:
    """Thread entry point: any unexpected failure must still land in the
    job status, or the UI would poll a job that says "running" forever."""
    try:
        _run_job_inner(options)
    except Exception as exc:                              # noqa: BLE001
        _job_update(status="error", finished=time.time(), current="",
                    error=f"The captioning job failed: {exc}")


def _run_job_inner(options: dict) -> None:
    folder = (options.get("folder") or "").strip().strip('"')
    source = options.get("source", "tags")
    mode = options.get("mode", "missing_only")
    if source == "tags":
        # an image without a caption has no tags to rephrase, so for this source
        # every image that has tags is in scope no matter what was picked
        mode = "all"
    try:
        limit = int(options.get("limit") or 0)
    except (TypeError, ValueError):
        limit = 0
    try:
        images = _selected_images(folder, mode, limit)
    except Exception as exc:                              # noqa: BLE001
        _job_update(status="error", error=f"Could not read the folder: {exc}", finished=time.time())
        return

    # every image the folder can offer, so the bar shows real progress even when
    # most of them are skipped
    try:
        total = len(datasets.list_images(folder))
    except Exception:                                     # noqa: BLE001
        total = len(images)
    _job_update(total=total, done=0, skipped=max(0, total - len(images)))

    if not images:
        _job_update(status="done", finished=time.time())
        return

    _snapshot(folder, "Natural-language captions")

    if source == "tags":
        _run_tag_job(folder, images, options)
    else:
        _run_vlm_job(folder, images, options)


def _run_tag_job(folder: str, images: list[dict], options: dict) -> None:
    trigger = (options.get("trigger_word") or "").strip()
    drop_quality = bool(options.get("drop_quality", True))
    write = options.get("write", "replace")
    done = skipped = failed = 0
    for img in images:
        if _CANCEL.is_set():
            _job_update(status="cancelled", finished=time.time())
            return
        _job_update(current=img["filename"])
        tags = captions._split(img.get("caption") or "")
        if not tags:
            skipped += 1
            _job_update(skipped=skipped, done=done + skipped)
            _job_add_result({"path": img["path"], "status": "skipped", "reason": "no tags to rephrase"})
            continue
        try:
            if write == "keep":
                # the tags stay where they are, so the sentence is built from what
                # is left after the trigger and written after the tags
                sentence = caption_from_tags(_without_trigger(tags, trigger), drop_quality=drop_quality)
                text = _compose_keep(tags, sentence, trigger)
            else:
                sentence = caption_from_tags(tags, trigger_word=trigger, drop_quality=drop_quality)
                text = sentence
            if not sentence:
                skipped += 1
                _job_add_result({"path": img["path"], "status": "skipped", "reason": "nothing to rephrase"})
            else:
                captions.write_caption(img["path"], text)
                done += 1
                _job_add_result({"path": img["path"], "status": "ok", "caption": text})
        except Exception as exc:                          # noqa: BLE001
            failed += 1
            _job_add_result({"path": img["path"], "status": "error", "error": str(exc)})
        _job_update(done=done + skipped + failed, captioned=done, skipped=skipped, failed=failed)
    _job_update(status="done", finished=time.time(), current="",
                captioned=done, skipped=skipped, failed=failed)


def _vlm_job(model: str, options: dict, images: list[dict]) -> dict:
    """The job handed to native_backend.caption_worker.

    The repo is the copy the Download button filled; the hub id would send
    transformers to the shared Hugging Face cache and fetch the weights again.
    """
    info = MODELS[model]
    return {
        "repo": str(model_dir(model)) if is_downloaded(model) else info["repo"],
        "kind": info["kind"],
        "family": info.get("family", ""),
        "system": info.get("system", ""),
        "prompt": options.get("prompt") or info.get("prompt", ""),
        "max_new_tokens": int(options.get("max_new_tokens") or 320),
        "size_gb": info.get("size_gb", 0),
        "skip_modules": list(info.get("skip_modules", [])),
        "vram_profile": options.get("vram_profile") if options.get("vram_profile") in VRAM_PROFILES else "auto",
        "quant": options.get("quant") if options.get("quant") in ("4bit", "8bit", "bf16") else "auto",
        "images": [{"id": i["path"]} for i in images],
    }


def _run_vlm_job(folder: str, images: list[dict], options: dict) -> None:
    model = options.get("model") or DEFAULT_MODEL
    python, label = interpreter(model)
    deps = deps_status(force=True, model=model)
    if not deps["ok"]:
        _job_update(status="error", finished=time.time(),
                    error=f"The captioning dependencies are missing from {label}"
                          + (f" ({', '.join(deps['missing'])})" if deps["missing"] else "")
                          + ". Press Install dependencies, then try again.")
        return
    if not deps.get("torch"):
        _job_update(status="error", finished=time.time(),
                    error=f"PyTorch is missing from {label} - press Install dependencies first.")
        return

    trigger = (options.get("trigger_word") or "").strip()
    write = options.get("write", "replace")
    job = _vlm_job(model, options, images)
    tmp = Path(tempfile.mkdtemp(prefix="anima-caption-"))
    job_file = tmp / "job.json"
    job_file.write_text(json.dumps(job), encoding="utf-8")

    # A cancel pressed while the dependencies were being probed would be
    # lost below (cancel() had no worker process to kill yet).
    if _CANCEL.is_set():
        shutil.rmtree(tmp, ignore_errors=True)
        _job_update(status="cancelled", finished=time.time())
        return

    done = failed = 0
    skipped = run_status().get("skipped", 0)              # images already captioned before this batch
    _job_update(model=model)
    try:
        proc = subprocess.Popen(
            [python, "-u", "-m", "native_backend.caption_worker", str(job_file)],
            cwd=str(config.APP_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except OSError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        _job_update(status="error", finished=time.time(), error=f"Could not start the captioner: {exc}")
        return

    procs.register(proc)                                  # app shutdown must not orphan the worker
    with _proc_lock:
        globals()["_proc"] = proc
    if _CANCEL.is_set():
        cancel()                                          # pressed between spawn and registration

    by_path = {i["path"]: i for i in images}
    tail: list[str] = []
    for line in proc.stdout:                              # type: ignore[union-attr]
        line = line.rstrip("\r\n")
        if not line:
            continue
        tail.append(line)
        del tail[:-12]
        if line.startswith("@@READY "):
            try:
                ready = json.loads(line[8:])
                _job_update(device=ready.get("device", ""), loader=ready.get("loader", ""),
                            quant=ready.get("quant", ""))
            except ValueError:
                pass
            continue
        if line.startswith("@@CAPTION "):
            try:
                item = json.loads(line[10:])
            except ValueError:
                continue
            img = by_path.get(item.get("id", ""))
            if img is None:
                continue
            _job_update(current=img["filename"])
            if item.get("caption"):
                try:
                    text = item["caption"]
                    if write == "keep":
                        text = _compose_keep(captions._split(img.get("caption") or ""), text, trigger)
                    elif trigger:
                        text = f"{trigger}, {text[0].lower()}{text[1:]}" if text else text
                    captions.write_caption(img["path"], text)
                    done += 1
                    _job_add_result({"path": img["path"], "status": "ok", "caption": text})
                except Exception as exc:                  # noqa: BLE001
                    failed += 1
                    _job_add_result({"path": img["path"], "status": "error", "error": str(exc)})
            else:
                failed += 1
                _job_add_result({"path": img["path"], "status": "error", "error": item.get("error", "failed")})
            _job_update(done=done + skipped + failed, captioned=done, failed=failed)
            continue
        if line.startswith("@@DONE "):
            try:
                payload = json.loads(line[7:])
            except ValueError:
                payload = {}
            if payload.get("error"):
                _job_update(error=payload["error"])

    code = proc.wait()
    with _proc_lock:
        globals()["_proc"] = None
    shutil.rmtree(tmp, ignore_errors=True)

    if _CANCEL.is_set():
        _job_update(status="cancelled", finished=time.time(), current="")
        return
    if code != 0:
        # A worker that died mid-batch (crash, OOM, killed) must never be
        # reported as "done" just because some images had been captioned.
        remaining = max(0, len(images) - done - failed)
        _job_update(status="error", finished=time.time(), current="",
                    captioned=done, skipped=skipped, failed=failed + remaining,
                    error=run_status()["error"]
                    or "The captioner stopped unexpectedly: " + " / ".join(tail[-3:]))
        return
    _job_update(status="done", finished=time.time(), current="",
                captioned=done, skipped=skipped, failed=failed)
