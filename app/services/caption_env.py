"""A separate Python environment for the Qwen3.5 caption models.

Qwen3.5 needs transformers 5.x, while the training backend pins transformers 4.54. Installing 5.x into the training
backend could break LoRA training, so these models get their own virtual environment (caption_backend/venv) that is built
on demand: PyTorch (the build matching the NVIDIA driver), then transformers 5.x and the caption packages. It never
touches the training backend, the other engines, or Anima Studio's own packages.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import config
from . import procs
from .edit_setup import SetupError, _detect_index

VENV = config.APP_DIR / "caption_backend" / "venv"
_BIN = "Scripts" if sys.platform == "win32" else "bin"
_EXE = ".exe" if sys.platform == "win32" else ""
PYTHON_VERSIONS = ("3.12", "3.11", "3.10")
MIN_TRANSFORMERS = (5, 3)
PACKAGES = ["transformers>=5.3", "accelerate>=1.0", "sentencepiece", "protobuf", "huggingface-hub", "pillow",
            "bitsandbytes>=0.46.1"]
STEPS = [("venv", "Create the environment"), ("torch", "Install PyTorch (the build for your GPU)"),
         ("pkgs", "Install transformers 5.x and the caption packages"), ("verify", "Check everything")]

_lock = threading.Lock()
_state = {"stage": "idle", "step": "", "step_index": 0, "error": None, "log": [], "started": None, "finished": None}
_proc: subprocess.Popen | None = None
_cancel = threading.Event()


def venv_python() -> Path:
    return VENV / _BIN / ("python" + _EXE)


def has_venv() -> bool:
    return venv_python().exists()


def is_ready() -> bool:
    """Installed and verified earlier (the marker is written only after a passing check)."""
    return has_venv() and (VENV / ".caption_ready").exists()


def _log(line: str) -> None:
    with _lock:
        _state["log"].append(line.rstrip("\n"))
        del _state["log"][:-400]


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def status() -> dict:
    with _lock:
        s = dict(_state)
        s["log_tail"] = list(_state["log"][-120:])
        s.pop("log", None)
    s["steps"] = [{"id": i, "label": label} for i, label in STEPS]
    s["installed"] = is_ready()
    return s


def _run(cmd: list[str]) -> int:
    global _proc
    import os

    _log("$ " + " ".join(str(c) for c in cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    proc = subprocess.Popen([str(c) for c in cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, **procs.NEW_GROUP)
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
    uv = shutil.which("uv")
    if uv:
        return [uv, "venv", "--python", PYTHON_VERSIONS[1], "--seed"]
    v = sys.version_info
    if f"{v.major}.{v.minor}" in PYTHON_VERSIONS:
        return [sys.executable, "-m", "venv"]
    if sys.platform == "win32" and shutil.which("py"):
        for ver in PYTHON_VERSIONS:
            try:
                r = subprocess.run(["py", f"-{ver}", "-c", "pass"], capture_output=True, timeout=30,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0:
                return ["py", f"-{ver}", "-m", "venv"]
    for ver in PYTHON_VERSIONS:
        exe = shutil.which(f"python{ver}")
        if exe:
            return [exe, "-m", "venv"]
    raise SetupError(f"Python {v.major}.{v.minor} cannot be used here: the Qwen3.5 captioner needs Python 3.10-3.12. "
                     "Install Python 3.11 (python.org, with the py launcher) or uv (astral.sh/uv), then try again.")


def _pip(*args: str) -> list[str]:
    return [str(venv_python()), "-m", "pip", *args]


def start() -> dict:
    with _lock:
        if _state["stage"] == "installing":
            raise SetupError("The caption environment is already being installed.")
        _state.update(stage="installing", step="venv", step_index=0, error=None, log=[], started=time.time(),
                      finished=None)
    _cancel.clear()
    threading.Thread(target=_install, daemon=True, name="caption-env-setup").start()
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


PROBE = (
    "import json, importlib.metadata as md\n"
    "out = {}\n"
    "try:\n"
    "    import torch\n"
    "    out['torch'] = {'version': torch.__version__, 'cuda': bool(torch.cuda.is_available())}\n"
    "except Exception as e:\n"
    "    out['torch'] = None; out['torch_error'] = str(e)[:200]\n"
    "def v(n):\n"
    "    try: return md.version(n)\n"
    "    except Exception: return ''\n"
    "out['transformers'] = v('transformers'); out['bitsandbytes'] = v('bitsandbytes')\n"
    "try:\n"
    "    import transformers\n"
    "    out['multimodal_class'] = hasattr(transformers, 'AutoModelForImageTextToText') or hasattr(transformers, 'AutoModelForMultimodalLM')\n"
    "except Exception as e:\n"
    "    out['multimodal_class'] = False; out['import_error'] = str(e)[:200]\n"
    "print('@@PROBE ' + json.dumps(out))\n")


def verify_report(rep: dict) -> list[dict]:
    """The probe's answer -> a checklist (pure, so it can be tested)."""
    checks = []

    def add(cid, label, status_, detail):
        checks.append({"id": cid, "label": label, "status": status_, "detail": detail})

    t = rep.get("torch")
    add("torch", "PyTorch", "ok" if t else "fail", (t or {}).get("version") or rep.get("torch_error", "not installed"))
    if t:
        add("gpu", "GPU", "ok" if t.get("cuda") else "warn",
            "PyTorch sees your GPU" if t.get("cuda") else "PyTorch cannot see a GPU - captioning would run on the CPU (slow).")
    import re
    tv = rep.get("transformers") or ""
    parts = tuple(int(x) for x in re.findall(r"\d+", tv)[:2]) if tv else ()
    add("transformers", "transformers", "ok" if parts and parts >= MIN_TRANSFORMERS else "fail",
        tv or "not installed" if not parts or parts >= MIN_TRANSFORMERS else f"{tv} (needs {MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]}+)")
    add("loader", "Image-text model loader", "ok" if rep.get("multimodal_class") else "fail",
        "available" if rep.get("multimodal_class") else rep.get("import_error", "missing"))
    add("bitsandbytes", "bitsandbytes (4/8-bit loading)", "ok" if rep.get("bitsandbytes") else "warn",
        rep.get("bitsandbytes") or "not installed - only bf16 loading would work")
    return checks


def verify() -> dict:
    try:
        r = subprocess.run([str(venv_python()), "-c", PROBE], capture_output=True, text=True, timeout=300,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        line = next(l for l in (r.stdout or "").splitlines() if l.startswith("@@PROBE "))
        rep = json.loads(line[8:])
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as exc:
        return {"ok": False, "checks": [{"id": "probe", "label": "Environment check", "status": "fail",
                                         "detail": f"The check could not run: {type(exc).__name__}"}]}
    checks = verify_report(rep)
    return {"ok": not any(c["status"] == "fail" for c in checks), "checks": checks}


def _install() -> None:
    try:
        index = _detect_index()
        _log(f"PyTorch index: {index}")
        _set(step="venv", step_index=0)
        if not has_venv():
            VENV.parent.mkdir(parents=True, exist_ok=True)
            if _run(_find_python() + [str(VENV)]) != 0 or not has_venv():
                return _fail("Could not create the virtual environment (see the log).")
        else:
            _log("Reusing the existing caption environment.")
        (VENV / ".caption_ready").unlink(missing_ok=True)
        if _cancel.is_set():
            return _set(stage="idle", error=None)
        _run(_pip("install", "--upgrade", "pip", "setuptools", "wheel"))

        _set(step="torch", step_index=1)
        if _run(_pip("install", "torch", "torchvision", "--index-url", index)) != 0:
            return _fail("PyTorch failed to install. It needs a 64-bit Python 3.10-3.12 and a working internet connection.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)

        _set(step="pkgs", step_index=2)
        if _run(_pip("install", "--default-timeout=120", "--retries", "10", *PACKAGES)) != 0:
            return _fail("Installing transformers and the caption packages failed - scroll up in the log for the package that broke.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)

        _set(step="verify", step_index=3)
        res = verify()
        for c in res["checks"]:
            _log(f"[{c['status']}] {c['label']}: {c['detail']}")
        if not res["ok"]:
            return _fail("The environment installed but a check failed - see the log.")
        (VENV / ".caption_ready").write_text("ok", encoding="utf-8")
        _set(stage="done", step="", error=None, finished=time.time())
        _log("Caption environment ready.")
    except SetupError as exc:
        _fail(str(exc))
    except Exception as exc:                                  # noqa: BLE001 - surfaced in the UI
        _fail(f"{type(exc).__name__}: {exc}")
