"""Low-VRAM trainer environment: install and verify.

KohakuMossTrainer (kohaku_backend/, vendored unmodified) is its own training program with its own pinned packages
(torch 2.13-2.14, transformers 5.3+, diffusers 0.38+, kohaku-engine, LyCORIS 4.1.3+), so it lives in its own virtual
environment (kohaku_backend/venv) and never touches the Anima/SDXL or the sd-scripts MOSS-Image environments.
This module builds it step by step the way its own install helper does (torch from the PyTorch index that
matches the NVIDIA driver, then the package editable with its CUDA extras), reports honest progress, and verifies
the result with its own `kohakumoss-doctor`.

Nothing here needs torch in Anima Studio's own interpreter: every probe runs inside the Kohaku venv.
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
from . import procs
from .moss_setup import _ram_gb


class kohaku_env:                                            # noqa: N801 - same paths as native_backend/kohaku_env.py
    """Mirror of native_backend/kohaku_env.py (kept separate so a packaged app never imports the training package)."""
    DIR = config.APP_DIR / "kohaku_backend"
    VENV = DIR / "venv"
    _BIN = "Scripts" if sys.platform == "win32" else "bin"
    _EXE = ".exe" if sys.platform == "win32" else ""

    @staticmethod
    def exe(name: str) -> Path:
        return kohaku_env.VENV / kohaku_env._BIN / (name + kohaku_env._EXE)

    @staticmethod
    def venv_python() -> Path:
        return kohaku_env.exe("python")

    @staticmethod
    def has_venv() -> bool:
        return kohaku_env.venv_python().exists()

    @staticmethod
    def has_kogine() -> bool:
        return kohaku_env.exe("kogine").exists()

    @staticmethod
    def has_trainer() -> bool:
        return (kohaku_env.DIR / "scripts" / "train" / "moss.py").is_file() and \
            (kohaku_env.DIR / "src" / "kohakumoss" / "__init__.py").is_file()


TORCH_SPEC = "torch>=2.13,<2.15"                 # the one pin of Low-VRAM trainer (its pyproject, install helpers, docs)
TORCH_EXTRAS = ("torchvision",)
PACKAGE_EXTRAS = "[cuda,optim]"                  # cuda = kernels (triton-windows / tilelang) + bitsandbytes; optim = prodigy, schedule-free
PYTHON_VERSION = "3.12"                          # the helper's choice (3.10-3.13 work)
INDEX_BASE = "https://download.pytorch.org/whl/"

STEPS = [
    ("venv", "Create the Low-VRAM trainer environment"),
    ("torch", "Install PyTorch (the build for your NVIDIA driver)"),
    ("package", "Install Low-VRAM trainer and its packages"),
    ("verify", "Check everything (Low-VRAM trainer doctor)"),
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
    s["installed"] = kohaku_env.has_venv() and kohaku_env.has_kogine()
    s["trainer_present"] = kohaku_env.has_trainer()
    s["torch_spec"] = TORCH_SPEC
    return s


# --------------------------------------------------------------------- install

def torch_index_tag(smi_output: str) -> str:
    """The PyTorch wheel index for the driver's CUDA version (nvidia-smi's 'CUDA Version: X.Y'), the way
    its install helper picks it: >= 13.0 cu130, >= 12.8 cu128, >= 12.6 cu126."""
    m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", smi_output or "")
    if not m:
        raise SetupError("Could not read the CUDA version from nvidia-smi. Install a current NVIDIA driver and "
                         "restart Anima Studio.")
    ver = int(m.group(1)) * 100 + int(m.group(2))
    if ver >= 1300:
        return "cu130"
    if ver >= 1208:
        return "cu128"
    if ver >= 1206:
        return "cu126"
    raise SetupError(f"Your NVIDIA driver supports CUDA {m.group(1)}.{m.group(2)}; Low-VRAM trainer needs CUDA 12.6 or "
                     "newer (driver 560+). Update the NVIDIA driver and try again.")


def _detect_index() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        raise SetupError("nvidia-smi was not found - install the NVIDIA driver (and restart Anima Studio) first.")
    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=30,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupError(f"nvidia-smi failed: {exc}") from None
    return INDEX_BASE + torch_index_tag(r.stdout)


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
    """The command that creates the venv: uv (fetches Python 3.12 itself), else a suitable local Python."""
    uv = shutil.which("uv")
    if uv:
        return [uv, "venv", "--python", PYTHON_VERSION, "--seed"]
    v = sys.version_info
    candidates = [shutil.which(n) for n in ("python3.12", "python3.13", "python3.11", "python3.10")]
    if (3, 10) <= (v.major, v.minor) <= (3, 13):
        return [sys.executable, "-m", "venv"]
    if sys.platform == "win32" and shutil.which("py"):
        return ["py", f"-{PYTHON_VERSION}", "-m", "venv"]
    py = next((c for c in candidates if c), None)
    if py:
        return [py, "-m", "venv"]
    raise SetupError(f"Python {v.major}.{v.minor} cannot be used for Low-VRAM trainer. Install Python {PYTHON_VERSION} "
                     "(python.org, with the py launcher) or uv (astral.sh/uv), then try again.")


def _pip(*args: str) -> list[str]:
    return [str(kohaku_env.venv_python()), "-m", "pip", *args]


def start() -> dict:
    if not kohaku_env.has_trainer():
        raise SetupError("kohaku_backend is missing from this copy of Anima Studio - redownload it.")
    with _lock:
        if _state["stage"] == "installing":
            raise SetupError("The Low-VRAM trainer environment is already being installed.")
        _state.update(stage="installing", step="venv", step_index=0, error=None, log=[],
                      started=time.time(), finished=None, torch_index="")
    _cancel.clear()
    threading.Thread(target=_install, daemon=True, name="kohaku-setup").start()
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
        _log(f"PyTorch index for your driver: {index}")

        _set(step="venv", step_index=0)
        if not kohaku_env.has_venv():
            code = _run(_find_python() + [str(kohaku_env.VENV)])
            if code != 0 or not kohaku_env.has_venv():
                return _fail("Could not create the virtual environment (see the log).")
        else:
            _log("Reusing the existing Low-VRAM trainer environment.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)
        _run(_pip("install", "--upgrade", "pip", "setuptools", "wheel"))

        _set(step="torch", step_index=1)
        code = _run(_pip("install", TORCH_SPEC, *TORCH_EXTRAS, "--index-url", index))
        if code != 0:
            return _fail("PyTorch failed to install. It needs a 64-bit Python 3.10-3.13, a recent NVIDIA driver and "
                         "a working internet connection.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)

        _set(step="package", step_index=2)
        code = _run(_pip("install", "-e", f"{kohaku_env.DIR}{PACKAGE_EXTRAS}", TORCH_SPEC, "--extra-index-url", index),
                    cwd=kohaku_env.DIR)
        if code != 0:
            return _fail("Installing Low-VRAM trainer failed - scroll up in the log for the package that broke. "
                         "(On Windows, packages are wheels; a build error usually means a Python newer than 3.13.)")
        if not kohaku_env.has_kogine():
            return _fail("Low-VRAM trainer installed but its 'kogine' runner is missing - re-run the setup.")
        if _cancel.is_set():
            return _set(stage="idle", error=None)

        _set(step="verify", step_index=3)
        result = verify()
        for c in result["checks"]:
            _log(f"[{c['status']}] {c['label']}: {c['detail']}")
        if any(c["status"] == "fail" for c in result["checks"]):
            return _fail("The environment installed but a check failed - see the log.")
        _set(stage="done", step="", error=None, finished=time.time())
        _log("Low-VRAM trainer environment ready.")
    except SetupError as exc:
        _fail(str(exc))
    except Exception as exc:                                  # noqa: BLE001 - surfaced in the UI
        _fail(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------- verify

REQUIRED_PACKAGES = ("kohaku-engine", "anyschedule", "kohakuvault", "lycoris-lora", "diffusers", "transformers")


def doctor_checks(rep: dict) -> list[dict]:
    """its own `kohakumoss-doctor --json` report -> the checklist the Backend tab shows."""
    checks: list[dict] = []

    def add(cid: str, label: str, status_: str, detail: str, fix: str = "") -> None:
        checks.append({"id": cid, "label": label, "status": status_, "detail": detail, "fix": fix})

    pk = rep.get("packages") or {}
    torch_v = pk.get("torch")
    if not torch_v:
        add("torch", "PyTorch", "fail", "not installed", "Re-run the setup.")
    else:
        ok_pin = bool(re.match(r"2\.1[34]\.", torch_v))
        add("torch", "PyTorch", "ok" if ok_pin else "warn", f"{torch_v} (CUDA {rep.get('torch_cuda') or 'none'})",
            "" if ok_pin else "Low-VRAM trainer pins torch 2.13-2.14; re-run the setup to restore it.")
    devs = rep.get("devices") or []
    if devs:
        for d in devs:
            add("gpu", "GPU", "ok", f"{d.get('name')} · {d.get('memory_gib')} GB · {d.get('arch')}")
    else:
        add("gpu", "GPU", "fail", "PyTorch cannot see a GPU.",
            "Update the NVIDIA driver and restart; if torch shows no CUDA above, re-run the setup.")
    absent = [p for p in REQUIRED_PACKAGES if not pk.get(p)]
    add("packages", "Low-VRAM trainer packages", "fail" if absent else "ok",
        ("Missing: " + ", ".join(absent)) if absent else "All present.", "Re-run the setup." if absent else "")
    tl, tw = pk.get("tilelang"), pk.get("triton-windows") or pk.get("triton")
    add("kernels", "Kernel packages", "ok" if (tl or tw) else "warn",
        f"triton {tw or '-'} · tilelang {tl or '-'}", "" if (tl or tw) else "Without them training uses slower fallback kernels.")
    msvc = rep.get("msvc") or {}
    if msvc.get("needed"):
        found = msvc.get("cl_on_path") or msvc.get("vswhere_sees_vs")
        add("msvc", "C++ build tools (Windows)", "ok" if found else "warn",
            "found" if found else "Visual Studio Build Tools were not found.",
            "" if found else "Install VS Build Tools 2022 with the 'Desktop development with C++' workload; the kernels "
                             "compile with it. Without it, training falls back to slower kernels.")
    for row in rep.get("missing") or []:
        what = str(row.get("what", ""))
        low = what.lower()
        bad = "cpu-only" in low or ("driver" in low and "torch" in low)
        add("missing", "Doctor: " + what[:70], "fail" if bad else "warn", what, str(row.get("fix", "")))
    for key, err in (rep.get("backend_failures") or {}).items():
        add("backend", f"Kernel backend failed: {key}", "warn", str(err)[:200],
            "Training falls back to the next kernel backend (slower). Fix the toolchain, then re-run Verify.")
    return checks


def verify() -> dict:
    """A checklist, not a yes/no: the interesting failures are partial ones."""
    checks: list[dict] = []

    def add(cid: str, label: str, status_: str, detail: str, fix: str = "") -> None:
        checks.append({"id": cid, "label": label, "status": status_, "detail": detail, "fix": fix})

    if kohaku_env.has_trainer():
        add("files", "Bundled Low-VRAM trainer", "ok", "kohaku_backend is present.")
    else:
        add("files", "Bundled Low-VRAM trainer", "fail", "kohaku_backend is missing.", "Redownload Anima Studio.")
    if not kohaku_env.has_venv():
        add("venv", "Low-VRAM trainer environment", "fail", "Not installed yet.", "Press Set up Low-VRAM trainer environment.")
        return {"ok": False, "checks": checks, "vram_gb": None}
    add("venv", "Low-VRAM trainer environment", "ok", str(kohaku_env.VENV))
    if not kohaku_env.has_kogine():
        add("kogine", "kogine runner", "fail", "The runner that starts Low-VRAM trainer scripts is missing.", "Re-run the setup.")
        return {"ok": False, "checks": checks, "vram_gb": None}
    add("kogine", "kogine runner", "ok", "present")

    vram = None
    doctor = kohaku_env.exe("kohakumoss-doctor")
    if not doctor.exists():
        add("doctor", "Low-VRAM trainer doctor", "fail", "the check tool is missing.", "Re-run the setup.")
    else:
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="kohaku_doctor_")
        os.close(fd)
        try:
            r = subprocess.run([str(doctor), "--json", tmp], capture_output=True, text=True, timeout=900,
                               cwd=str(kohaku_env.DIR), env={**os.environ, "PYTHONUTF8": "1"},
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            try:
                rep = json.loads(Path(tmp).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                tail = ((r.stderr or r.stdout or "").strip().splitlines() or ["no answer"])[-1]
                add("doctor", "Low-VRAM trainer doctor", "fail", tail, "Re-run the setup.")
            else:
                checks.extend(doctor_checks(rep))
                if rep.get("devices"):
                    vram = rep["devices"][0].get("memory_gib")
        except (OSError, subprocess.SubprocessError) as exc:
            add("doctor", "Low-VRAM trainer doctor", "fail", str(exc))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    ram = _ram_gb()
    if ram is not None:
        add("ram", "System RAM", "ok" if ram >= 32 else "warn", f"{ram} GB",
            "" if ram >= 32 else "The 8 GB profile reads the model through ~32 GB of host memory at build time. On less, "
                                 "use the 12 GB profile with an int8 base, or free memory first.")
    return {"ok": not any(c["status"] == "fail" for c in checks), "checks": checks, "vram_gb": vram}
