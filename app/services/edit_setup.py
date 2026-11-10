"""AI edit environment: install and verify.

The Edit tab's AI tools (LaMa and the other erase models, Stable Diffusion inpainting, Segment Anything, background
removal, Real-ESRGAN, face restoration) are IOPaint's - the engine behind lama-cleaner. IOPaint pins old versions of
Pillow, diffusers, huggingface-hub and gradio, so it lives in its own virtual environment (edit_backend/venv) and never
touches Anima Studio's own packages or the training engines. This module builds that environment step by step (PyTorch
from the index that matches the NVIDIA driver, then IOPaint, then the background-removal extras), reports honest
progress, and verifies the result by running the worker's own `--check` inside it.

Nothing here needs torch in Anima Studio's own interpreter.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from edit_backend import catalog
from edit_backend import env as edit_env

from . import procs

IOPAINT_SPEC = f"iopaint=={catalog.IOPAINT_VERSION}"
# IOPaint 1.6.0 accepts any transformers >= 4.39.1, but it pins huggingface-hub 0.25.2, which the newest transformers
# no longer allow; capping it keeps pip from backtracking through dozens of releases.
TRANSFORMERS_SPEC = "transformers>=4.39.1,<4.50"
EXTRA_SPECS = ("onnxruntime<=1.19.2", "rembg[cpu]")          # what `iopaint install-plugins-packages` installs
TORCH_PACKAGES = ("torch", "torchvision")
PYTHON_VERSIONS = ("3.11", "3.10")                           # IOPaint pins Pillow 9.5.0, which has no wheels past 3.11
INDEX_BASE = "https://download.pytorch.org/whl/"

STEPS = [
    ("venv", "Create the AI edit environment"),
    ("torch", "Install PyTorch (the build for your GPU)"),
    ("iopaint", "Install the AI edit engine"),
    ("extras", "Install the background-removal packages"),
    ("verify", "Check everything"),
]

_lock = threading.Lock()
_state = {"stage": "idle", "step": "", "step_index": 0, "error": None, "log": [], "started": None, "finished": None,
          "torch_index": ""}
_proc: subprocess.Popen | None = None
_cancel = threading.Event()


class SetupError(RuntimeError):
    pass


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
    s["installed"] = edit_env.has_venv() and edit_env.has_iopaint()
    s["trainer_present"] = edit_env.has_worker()               # the key the shared Backend-tab card reads
    s["iopaint"] = catalog.IOPAINT_VERSION
    return s


# --------------------------------------------------------------------- install

def torch_index_tag(smi_output: str) -> str:
    """The PyTorch wheel index for the driver's CUDA version (nvidia-smi's 'CUDA Version: X.Y'). IOPaint runs on any
    torch >= 2.0, so older drivers get the older CUDA builds that still exist instead of an error."""
    m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", smi_output or "")
    if not m:
        raise SetupError("Could not read the CUDA version from nvidia-smi.")
    ver = int(m.group(1)) * 100 + int(m.group(2))
    for floor, tag in ((1300, "cu130"), (1208, "cu128"), (1206, "cu126"), (1204, "cu124"), (1201, "cu121"),
                       (1108, "cu118")):
        if ver >= floor:
            return tag
    raise SetupError(f"Your NVIDIA driver supports CUDA {m.group(1)}.{m.group(2)}, older than any current PyTorch "
                     "build. Update the NVIDIA driver, or choose the CPU device in the editor.")


def _detect_index() -> str:
    """The PyTorch index to install from; the CPU build when there is no NVIDIA GPU (the tools still work, slower)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        _log("nvidia-smi was not found - installing the CPU build of PyTorch (the AI tools will run on the CPU).")
        return INDEX_BASE + "cpu"
    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=30,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupError(f"nvidia-smi failed: {exc}") from None
    return INDEX_BASE + torch_index_tag(r.stdout)


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run one command, streaming its output into the log. Returns the exit code."""
    global _proc
    import os

    _log("$ " + " ".join(str(c) for c in cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
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
    """The command that creates the venv: uv (fetches Python 3.11 itself), else a suitable local Python."""
    uv = shutil.which("uv")
    if uv:
        return [uv, "venv", "--python", PYTHON_VERSIONS[0], "--seed"]
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
    raise SetupError(
        f"Python {v.major}.{v.minor} cannot be used for the AI edit tools: the AI edit engine needs Python 3.10 or 3.11. "
        "Install Python 3.11 (python.org, with the py launcher) or uv (astral.sh/uv), then try again.")


def _pip(*args: str) -> list[str]:
    return [str(edit_env.venv_python()), "-m", "pip", *args]


def start() -> dict:
    if not edit_env.has_worker():
        raise SetupError("edit_backend is missing from this copy of Anima Studio - redownload it.")
    with _lock:
        if _state["stage"] == "installing":
            raise SetupError("The AI edit environment is already being installed.")
        _state.update(stage="installing", step="venv", step_index=0, error=None, log=[],
                      started=time.time(), finished=None, torch_index="")
    _cancel.clear()
    threading.Thread(target=_install, daemon=True, name="edit-setup").start()
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
        index = _detect_index()
        _set(torch_index=index)
        _log(f"PyTorch index: {index}")

        _set(step="venv", step_index=0)
        if not edit_env.has_venv():
            code = _run(_find_python() + [str(edit_env.VENV)])
            if code != 0 or not edit_env.has_venv():
                return _fail("Could not create the virtual environment (see the log).")
        else:
            _log("Reusing the existing AI edit environment.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)
        _run(_pip("install", "--upgrade", "pip", "setuptools", "wheel"))

        _set(step="torch", step_index=1)
        code = _run(_pip("install", *TORCH_PACKAGES, "--index-url", index))
        if code != 0:
            return _fail("PyTorch failed to install. It needs a 64-bit Python 3.10 or 3.11 and a working internet "
                         "connection.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)

        _set(step="iopaint", step_index=2)
        code = _run(_pip("install", IOPAINT_SPEC, TRANSFORMERS_SPEC))
        if code != 0:
            return _fail("Installing the AI edit engine failed - scroll up in the log for the package that broke.")
        if not edit_env.has_iopaint():
            return _fail("The AI edit engine installed but cannot be found in the environment - re-run the setup.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)

        _set(step="extras", step_index=3)
        code = _run(_pip("install", *EXTRA_SPECS))
        if code != 0:
            _log("WARNING: the background-removal packages (rembg, onnxruntime) did not install. Everything else "
                 "works; only the rembg-based 'Remove background' models need them.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)

        _set(step="verify", step_index=4)
        result = verify()
        for c in result["checks"]:
            _log(f"[{c['status']}] {c['label']}: {c['detail']}")
        if any(c["status"] == "fail" for c in result["checks"]):
            return _fail("The environment installed but a check failed - see the log.")
        _set(stage="done", step="", error=None, finished=time.time())
        _log("AI edit environment ready.")
    except SetupError as exc:
        _fail(str(exc))
    except Exception as exc:                                  # noqa: BLE001 - surfaced in the UI
        _fail(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------- verify

def report_checks(rep: dict) -> list[dict]:
    """The worker's `--check` report -> the checklist the Backend tab shows."""
    checks: list[dict] = []

    def add(cid: str, label: str, status_: str, detail: str, fix: str = "") -> None:
        checks.append({"id": cid, "label": label, "status": status_, "detail": detail, "fix": fix})

    mods = rep.get("modules") or {}

    def mod(name: str) -> dict:
        return mods.get(name) or {}

    py = str(rep.get("python", ""))
    add("python", "Python", "ok" if py.startswith(("3.10", "3.11")) else "warn", py,
        "" if py.startswith(("3.10", "3.11")) else "The AI edit engine pins Pillow 9.5.0, which has no wheels for this Python.")
    t = mod("torch")
    add("torch", "PyTorch", "ok" if t.get("ok") else "fail",
        t.get("version") or t.get("error", "not installed"), "" if t.get("ok") else "Re-run the setup.")
    cuda = rep.get("cuda") or {}
    if cuda.get("available"):
        add("gpu", "GPU", "ok", f"{cuda.get('name')} · {cuda.get('total_gb')} GB · CUDA {cuda.get('torch_cuda')}")
    else:
        add("gpu", "GPU", "warn", "PyTorch cannot see a GPU - the AI tools will run on the CPU.",
            "Fine for LaMa and the small tools; Stable Diffusion is very slow on a CPU. To use the GPU, update the "
            "NVIDIA driver, then re-run the setup.")
    io = mod("iopaint")
    if not io.get("ok"):
        add("iopaint", "AI edit engine", "fail", io.get("error", "not installed"), "Re-run the setup.")
    else:
        same = io.get("version") == catalog.IOPAINT_VERSION
        add("iopaint", "AI edit engine", "ok" if same else "warn", io.get("version", ""),
            "" if same else f"Anima Studio was built for AI edit engine {catalog.IOPAINT_VERSION}; re-run the setup.")
    for key, label in (("iopaint.model_manager", "Erase and diffusion models"), ("iopaint.plugins", "Plugins")):
        m = mod(key)
        add(key, label, "ok" if m.get("ok") else "fail", "loads" if m.get("ok") else m.get("error", "failed to load"),
            "" if m.get("ok") else "Re-run the setup (a package version may be wrong).")
    for key, label in (("diffusers", "diffusers"), ("transformers", "transformers"), ("Pillow", "Pillow"),
                       ("numpy", "numpy")):
        m = mod(key)
        add(key, label, "ok" if m.get("ok") else "warn", m.get("version") or m.get("error", "missing"))
    rb = mod("rembg")
    add("rembg", "rembg (background removal models)", "ok" if rb.get("ok") else "warn",
        rb.get("version") or "not installed",
        "" if rb.get("ok") else "Only the U²-Net / IS-Net / BiRefNet models need it; the anime cut-out and BRIA RMBG "
                                "models work without it. Re-run the setup to install it.")
    return checks


def verify() -> dict:
    """A checklist, not a yes/no: the interesting failures are partial ones."""
    checks: list[dict] = []

    def add(cid: str, label: str, status_: str, detail: str, fix: str = "") -> None:
        checks.append({"id": cid, "label": label, "status": status_, "detail": detail, "fix": fix})

    if edit_env.has_worker():
        add("files", "Bundled edit worker", "ok", "edit_backend is present.")
    else:
        add("files", "Bundled edit worker", "fail", "edit_backend is missing.", "Redownload Anima Studio.")
    if not edit_env.has_venv():
        add("venv", "AI edit environment", "fail", "Not installed yet.", "Press Set up AI edit environment.")
        return {"ok": False, "checks": checks}
    add("venv", "AI edit environment", "ok", str(edit_env.VENV))
    try:
        r = subprocess.run([str(edit_env.venv_python()), "-m", "edit_backend.worker", "--check"], capture_output=True,
                           text=True, timeout=300, cwd=str(edit_env.ROOT), env=_check_env(),
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        rep = json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        add("check", "Environment check", "fail", f"The check could not run: {type(exc).__name__}: {exc}"[:300],
            "Re-run the setup.")
        return {"ok": False, "checks": checks}
    checks.extend(report_checks(rep))
    try:
        free = shutil.disk_usage(str(_models_anchor())).free / 1024 ** 3
        add("disk", "Free disk space for models", "ok" if free >= 10 else "warn", f"{free:.0f} GB free",
            "" if free >= 10 else "Stable Diffusion models are 2-7 GB each; the folder can be moved on the editor's "
                                  "engine panel.")
    except OSError:
        pass
    return {"ok": not any(c["status"] == "fail" for c in checks), "checks": checks}


def _check_env() -> dict:
    import os
    return {**os.environ, "PYTHONUTF8": "1"}


def _models_anchor() -> Path:
    from . import edit_engine
    p = edit_engine.models_dir()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p
