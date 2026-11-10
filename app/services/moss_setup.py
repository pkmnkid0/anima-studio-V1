"""MOSS-Image environment: install, verify, and download the model.

MOSS-Image trains with a vendored fork of sd-scripts (moss_backend/sd_scripts, see moss_backend/PATCHES.md)
inside its own virtual environment, because it needs torch 2.14.1 + CUDA 13 (cuDNN 9.24 - older cuDNN returns
NaN attention gradients) while the Anima/SDXL engine pins something else. This module builds that environment
step by step, reports honest progress, and downloads OpenMOSS-Team/MOSS-Image-Base (~27 GB).

Nothing here needs torch in Anima Studio's own interpreter: every probe runs inside the MOSS venv.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import config
from . import procs


class moss_env:                                              # noqa: N801 - same paths as native_backend/moss_env.py
    """Mirror of native_backend/moss_env.py (kept separate so a packaged app never imports the training package)."""
    MOSS_DIR = config.APP_DIR / "moss_backend"
    SD_SCRIPTS = MOSS_DIR / "sd_scripts"
    VENV = MOSS_DIR / "venv"
    OVERLAY = SD_SCRIPTS / "moss_te_deps"

    @staticmethod
    def venv_python() -> Path:
        return moss_env.VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    @staticmethod
    def has_venv() -> bool:
        return moss_env.venv_python().exists()

    @staticmethod
    def has_trainer() -> bool:
        return (moss_env.SD_SCRIPTS / "moss_train_network.py").is_file() and \
            (moss_env.SD_SCRIPTS / "networks" / "lora_moss.py").is_file()

TORCH_VERSION = "2.14.1"
TORCHVISION_VERSION = "0.29.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
MODEL_REPO = "OpenMOSS-Team/MOSS-Image-Base"
MODEL_PATTERNS = ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"]
MODEL_SIZE_GB = 27
MIN_DRIVER = 580                         # CUDA 13 wheels need an NVIDIA driver >= 580

STEPS = [
    ("venv", "Create the MOSS environment"),
    ("torch", f"Install PyTorch {TORCH_VERSION} (CUDA 13)"),
    ("deps", "Install the trainer's packages"),
    ("overlay", "Install the Qwen3.5 text-encoder overlay (transformers 5)"),
    ("verify", "Check everything"),
]

_lock = threading.Lock()
_state = {"stage": "idle", "step": "", "step_index": 0, "error": None, "log": [], "started": None, "finished": None}
_proc: subprocess.Popen | None = None
_cancel = threading.Event()

_dl_lock = threading.Lock()
_dl = {"status": "idle", "bytes": 0, "total": MODEL_SIZE_GB * 1024 ** 3, "error": None, "dest": "", "file": ""}
_dl_proc: subprocess.Popen | None = None


class SetupError(RuntimeError):
    pass


def default_model_dir() -> Path:
    return moss_env.MOSS_DIR / "models" / "MOSS-Image-Base"


# ------------------------------------------------------------------ state helpers

def _log(line: str) -> None:
    with _lock:
        _state["log"].append(line.rstrip("\n"))
        del _state["log"][:-400]


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def is_busy() -> bool:
    with _lock:
        return _state["stage"] == "installing"


def status() -> dict:
    with _lock:
        s = dict(_state)
        s["log_tail"] = list(_state["log"][-120:])
        s.pop("log", None)
    s["steps"] = [{"id": i, "label": label} for i, label in STEPS]
    s["installed"] = moss_env.has_venv()
    s["trainer_present"] = moss_env.has_trainer()
    s["torch_version"] = TORCH_VERSION
    s["model_size_gb"] = MODEL_SIZE_GB
    return s


# --------------------------------------------------------------------- install

def _run(cmd: list[str], cwd: Path | None = None, env_extra: dict | None = None) -> int:
    """Run one command, streaming its output into the log. Returns the exit code."""
    global _proc
    _log("$ " + " ".join(str(c) for c in cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    env.update(env_extra or {})
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, **procs.NEW_GROUP)
    procs.register(proc)
    with _lock:
        _proc = proc
    for line in proc.stdout:                                  # type: ignore[union-attr]
        line = line.rstrip("\r\n")
        if line:
            _log(line)
    code = proc.wait()
    with _lock:
        _proc = None
    return code


def _find_python() -> list[str]:
    """The interpreter that creates the venv: uv (any Python it can fetch), else a suitable local Python."""
    uv = shutil.which("uv")
    if uv:
        return [uv, "venv", "--python", "3.11", "--seed"]
    v = sys.version_info
    if (3, 10) <= (v.major, v.minor) <= (3, 13):
        return [sys.executable, "-m", "venv"]
    py = shutil.which("python3.11") or shutil.which("python3.12") or shutil.which("python3.10")
    if py:
        return [py, "-m", "venv"]
    raise SetupError(f"Python {v.major}.{v.minor} is too new/old for PyTorch {TORCH_VERSION}. Install Python 3.11 "
                     "(python.org) or uv (astral.sh/uv), then try again.")


def _pip(*args: str) -> list[str]:
    return [str(moss_env.venv_python()), "-m", "pip", *args]


def start() -> dict:
    if not moss_env.has_trainer():
        raise SetupError("moss_backend/sd_scripts is missing from this copy of Anima Studio - redownload it.")
    with _lock:
        if _state["stage"] == "installing":
            raise SetupError("The MOSS environment is already being installed.")
        _state.update(stage="installing", step="venv", step_index=0, error=None, log=[],
                      started=time.time(), finished=None)
    _cancel.clear()
    threading.Thread(target=_install, daemon=True, name="moss-setup").start()
    return status()


def cancel() -> dict:
    _cancel.set()
    with _lock:
        proc = _proc
    procs.kill_tree(proc)
    return status()


def _fail(msg: str) -> None:
    _log("ERROR: " + msg)
    _set(stage="error", error=msg, finished=time.time())


def _install() -> None:
    try:
        sd = moss_env.SD_SCRIPTS
        # 1. venv
        _set(step="venv", step_index=0)
        if not moss_env.has_venv():
            code = _run(_find_python() + [str(moss_env.VENV)])
            if code != 0 or not moss_env.has_venv():
                return _fail("Could not create the virtual environment (see the log).")
        else:
            _log("Reusing the existing MOSS environment.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)
        _run(_pip("install", "--upgrade", "pip", "setuptools", "wheel"))

        # 2. torch
        _set(step="torch", step_index=1)
        code = _run(_pip("install", f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
                         "--index-url", TORCH_INDEX))
        if code != 0:
            return _fail(f"PyTorch {TORCH_VERSION} (CUDA 13) failed to install. It needs an NVIDIA driver >= {MIN_DRIVER}, "
                         "a 64-bit Python 3.10-3.13 and a working internet connection.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)

        # 3. the trainer's packages (includes git+https dependencies, so git must be installed)
        _set(step="deps", step_index=2)
        if shutil.which("git") is None:
            return _fail("Git is not installed (some trainer packages come from GitHub). Install it from "
                         "git-scm.com, restart Anima Studio and try again.")
        code = _run(_pip("install", "-r", "requirements.txt"), cwd=sd)
        if code != 0:
            return _fail("Installing the trainer's packages failed - scroll up in the log for the package that broke.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)

        # 4. transformers 5 overlay for the Qwen3.5 text encoder
        _set(step="overlay", step_index=3)
        code = _run([str(moss_env.venv_python()), "library/moss_te_client.py", "install"], cwd=sd,
                    env_extra={"PYTHONPATH": str(sd)})
        if code != 0:
            return _fail("Installing the text-encoder overlay (transformers 5) failed - see the log.")

        # 5. verify
        _set(step="verify", step_index=4)
        result = verify()
        for c in result["checks"]:
            _log(f"[{c['status']}] {c['label']}: {c['detail']}")
        if any(c["status"] == "fail" for c in result["checks"]):
            return _fail("The environment installed but a check failed - see the log.")
        _set(stage="done", step="", error=None, finished=time.time())
        _log("MOSS-Image environment ready.")
    except SetupError as exc:
        _fail(str(exc))
    except Exception as exc:                                  # noqa: BLE001 - surfaced in the UI
        _fail(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------- verify

_PROBE = r"""
import importlib.util as u, json, sys
out = {"python": sys.version.split()[0], "torch": None, "error": "", "modules": {}, "overlay": ""}
for m in ("accelerate", "diffusers", "safetensors", "bitsandbytes", "toml", "voluptuous", "einops", "library"):
    out["modules"][m] = u.find_spec(m) is not None
try:
    import torch
    cuda = bool(torch.cuda.is_available())
    out["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda or "", "available": cuda,
                    "cudnn": int(torch.backends.cudnn.version() or 0) if cuda else 0,
                    "name": torch.cuda.get_device_name(0) if cuda else "",
                    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if cuda else 0}
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
try:
    import glob, os
    found = glob.glob(os.path.join(sys.argv[1], "transformers-*.dist-info"))
    out["overlay"] = os.path.basename(found[0])[len("transformers-"):-len(".dist-info")] if found else ""
except Exception:
    pass
print("@@MOSS " + json.dumps(out))
"""


def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong),
                            ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong), ("f", ctypes.c_ulonglong)]
            ms = _MS()
            ms.l = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))      # type: ignore[attr-defined]
            return round(ms.total / 1024 ** 3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (OSError, ValueError, AttributeError):
        return None


def verify() -> dict:
    """A checklist, not a yes/no: the interesting failures are partial ones."""
    checks: list[dict] = []

    def add(cid: str, label: str, status_: str, detail: str, fix: str = "") -> None:
        checks.append({"id": cid, "label": label, "status": status_, "detail": detail, "fix": fix})

    if moss_env.has_trainer():
        add("files", "Bundled MOSS-Image trainer", "ok", "moss_backend/sd_scripts is present.")
    else:
        add("files", "Bundled MOSS-Image trainer", "fail", "moss_backend/sd_scripts is missing.",
            "Redownload Anima Studio.")
    if not moss_env.has_venv():
        add("venv", "MOSS environment", "fail", "Not installed yet.", "Press Set up MOSS-Image environment.")
        return {"ok": False, "checks": checks, "vram_gb": None}
    add("venv", "MOSS environment", "ok", str(moss_env.VENV))

    data: dict = {}
    try:
        r = subprocess.run([str(moss_env.venv_python()), "-c", _PROBE, str(moss_env.OVERLAY)],
                           capture_output=True, text=True, timeout=240, cwd=str(moss_env.SD_SCRIPTS),
                           env={**os.environ, "PYTHONPATH": str(moss_env.SD_SCRIPTS)},
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        line = next((l for l in (r.stdout or "").splitlines() if l.startswith("@@MOSS ")), None)
        if line:
            data = json.loads(line[7:])
        else:
            add("probe", "Environment probe", "fail", ((r.stderr or r.stdout or "").strip().splitlines() or ["no answer"])[-1])
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        add("probe", "Environment probe", "fail", str(exc))

    vram = None
    if data:
        t = data.get("torch")
        if not t:
            add("torch", "PyTorch", "fail", data.get("error") or "not importable", "Re-run the setup.")
        else:
            ok_ver = t["version"].startswith(TORCH_VERSION)
            add("torch", "PyTorch", "ok" if ok_ver else "warn",
                f"{t['version']} (CUDA {t['cuda'] or 'none'})",
                "" if ok_ver else f"The trainer is tested with torch {TORCH_VERSION}; Compile/cuDNN options may fail.")
            if t["available"]:
                vram = t["vram_gb"]
                add("gpu", "GPU", "ok", f"{t['name']} · {t['vram_gb']} GB")
                cudnn = t["cudnn"]
                add("cudnn", "cuDNN", "ok" if cudnn >= 92400 else "warn", str(cudnn),
                    "" if cudnn >= 92400 else "Attention 'cudnn' needs cuDNN 9.24+ (torch 2.14.1+cu130). Use Attention = auto.")
            else:
                add("gpu", "GPU", "fail", "PyTorch cannot see an NVIDIA GPU.",
                    f"Update the NVIDIA driver (needs {MIN_DRIVER}+ for CUDA 13) and restart.")
        missing = [m for m, ok in (data.get("modules") or {}).items() if not ok]
        add("packages", "Trainer packages", "fail" if missing else "ok",
            ("Missing: " + ", ".join(missing)) if missing else "All present.", "Re-run the setup." if missing else "")
        ov = data.get("overlay") or ""
        add("overlay", "Text-encoder overlay", "ok" if ov.startswith("5") else "fail",
            f"transformers {ov}" if ov else "not installed", "" if ov.startswith("5") else "Re-run the setup.")

    ram = _ram_gb()
    if ram is not None:
        add("ram", "System RAM", "ok" if ram >= 30 else "warn", f"{ram} GB",
            "" if ram >= 30 else "Block swap and the text-encoder offload keep ~12-20 GB in RAM; 32 GB is comfortable. "
                                 "Use a lower VRAM profile or the NF4 text encoder on smaller RAM.")
    return {"ok": not any(c["status"] == "fail" for c in checks), "checks": checks, "vram_gb": vram}


# ------------------------------------------------------------- model download

def model_status(path: str = "") -> dict:
    """Is the MOSS-Image pipeline folder complete? (cheap: only looks for the folders and a weights file)"""
    d = Path(path) if path else default_model_dir()
    parts = {}
    for sub in ("transformer", "text_encoder", "vae", "tokenizer"):
        folder = d / sub
        parts[sub] = folder.is_dir() and any(f.stat().st_size > 1024 for f in folder.iterdir() if f.is_file())
    complete = (d / "model_index.json").is_file() and all(parts.values())
    return {"path": str(d), "complete": complete, "parts": parts}


def download_status() -> dict:
    with _dl_lock:
        out = dict(_dl)
    out["model"] = model_status(out["dest"])
    return out


_DL_SCRIPT = r"""
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
patterns = sys.argv[3].split(",")
snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=patterns, max_workers=4)
print("@@DONE", flush=True)
"""


def start_download(dest: str = "") -> dict:
    if not moss_env.has_venv():
        raise SetupError("Set up the MOSS environment first (it provides the downloader).")
    target = Path(dest).expanduser() if dest else default_model_dir()
    with _dl_lock:
        if _dl["status"] == "downloading":
            raise SetupError("The model is already downloading.")
        _dl.update(status="downloading", bytes=0, error=None, dest=str(target), file="")
    threading.Thread(target=_download, args=(target,), daemon=True, name="moss-download").start()
    return download_status()


def cancel_download() -> dict:
    with _dl_lock:
        proc = _dl_proc
    procs.kill_tree(proc)
    return download_status()


def _download(target: Path) -> None:
    global _dl_proc
    try:
        target.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target).free / 1024 ** 3
        have = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024 ** 3
        if free + have < MODEL_SIZE_GB + 1:
            raise SetupError(f"Not enough free disk space on that drive: {free:.0f} GB free, ~{MODEL_SIZE_GB} GB needed.")
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(1.0):
                used, newest, newest_at = 0, "", 0.0
                for p in target.rglob("*"):
                    try:
                        if p.is_file():
                            used += p.stat().st_size
                            if p.suffix == ".incomplete" and p.stat().st_mtime > newest_at:
                                newest, newest_at = p.name, p.stat().st_mtime
                    except OSError:
                        continue
                with _dl_lock:
                    _dl.update(bytes=int(used), file=newest.split(".")[0][:40])

        threading.Thread(target=watch, daemon=True).start()
        try:
            proc = subprocess.Popen(
                [str(moss_env.venv_python()), "-c", _DL_SCRIPT, MODEL_REPO, str(target), ",".join(MODEL_PATTERNS)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, **procs.NEW_GROUP)
            procs.register(proc)
            with _dl_lock:
                _dl_proc = proc
            tail: list[str] = []
            for line in proc.stdout:                          # type: ignore[union-attr]
                line = line.strip()
                if line and not line.startswith("@@"):
                    tail.append(line)
                    del tail[:-6]
            code = proc.wait()
        finally:
            stop.set()
        if code != 0:
            raise SetupError("The download failed: " + (tail[-1] if tail else f"exit code {code}"))
        with _dl_lock:
            _dl.update(status="done", file="")
    except Exception as exc:                                  # noqa: BLE001
        with _dl_lock:
            _dl.update(status="error", error=str(exc), file="")
    finally:
        with _dl_lock:
            _dl_proc = None
