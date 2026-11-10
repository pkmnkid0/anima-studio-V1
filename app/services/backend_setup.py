"""
In-app installer for the venv Anima Studio's native backend
(native_backend/) runs training with. The bundled sd_scripts library at
training_backend/sd_scripts/ (see THIRD_PARTY_NOTICES.md - vendored
from 67372a/LoRA_Easy_Training_Scripts, not written by Anima Studio)
ships with Anima Studio already; what's still a separate step is
installing dependencies into its venv - first that library's own
(PyTorch, xformers, etc, via its own installer, a multi-GB download
that has to match your GPU/CUDA setup), then native_backend's own
SDXL-path dependencies (diffusers, peft, etc) into that same venv
right after, so native_backend/server.py has everything it needs for
both training paths in one place. This module runs both of those steps
from a button instead of a terminal, with output streamed live.

The sd_scripts installer itself is interactive (it can ask things like
"are you using this remotely? (y/n)" - auto-answered below, see
_run_interactive - and depending on the machine, follow-up questions
about CUDA/xformers versions) - this deliberately does not try to
script past unknown ones blind, since guessing wrong is worse than
asking: the log below is a real (very small) terminal, and whatever it
asks beyond what's auto-handled, type the answer in the box and click
Send. The native_backend dependency install that runs afterward is
plain and non-interactive, so it just runs straight through.

The input box only marks itself "ready" once the installer has gone
quiet for a bit while still running - not just "the process is alive" -
since that's the whole runtime of a multi-minute install. Without that,
someone reasonably sends an answer to what they *see* on screen, but if
nothing's actually blocked on a read() right then, that input just sits
in the pipe buffer and gets silently fed to whatever the *next* real
prompt turns out to be - answering a question they never got to read.
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

_MAX_LOG_LINES = 800
_IDLE_SECONDS_BEFORE_LIKELY_WAITING = 2.5

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_stage = "idle"  # idle | installing | done | error
_log_lines: list[str] = []
_target_dir: str | None = None
_error: str | None = None
_last_output_at: float = 0.0


class SetupError(RuntimeError):
    pass


def bundled_backend_dir() -> Path:
    """Where the vendored training_backend/ ships, right next to this
    app's own code - the default (and normally only) install target."""
    return config.APP_DIR / "training_backend"


def is_bundled() -> bool:
    return (bundled_backend_dir() / "main.py").exists()


def is_installed() -> bool:
    """Whether the bundled backend's own dependencies have actually been
    installed - checked by looking for the venv its installer creates,
    not just whether *this process* has run an install this session.
    Without this, restarting Anima Studio after a previous, successful
    install would show "idle"/"not started" again, even though it's
    already done."""
    venv = bundled_backend_dir() / "sd_scripts" / "venv"
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not py.exists():
        return False          # folder alone (interrupted install) doesn't count
    # a finished install has PyTorch in it
    for sp in (venv / "Lib" / "site-packages", *venv.glob("lib/python*/site-packages")):
        if (sp / "torch").exists():
            return True
    return False


def is_partial() -> bool:
    """A venv folder exists but the install never completed."""
    return (bundled_backend_dir() / "sd_scripts" / "venv").exists() and not is_installed()


def venv_python() -> Path:
    """The backend venv's interpreter. Public because features outside this
    module (captioning, for one) would rather borrow this environment's
    PyTorch than install a second copy of it."""
    return _venv_python()


def status() -> dict:
    with _lock:
        running = _proc is not None and _proc.poll() is None
        idle_for = time.time() - _last_output_at if running else 0.0
        # NOTE: _estimate_progress takes the already-captured lines and must
        # not re-acquire _lock - this lock is a plain (non-reentrant) Lock, so
        # grabbing it again here would deadlock the request forever.
        progress = _estimate_progress(list(_log_lines))
        return {
            "stage": _stage,
            "installed": is_installed(),
            "partial": is_partial(),
            "waiting_for_input": running and idle_for > _IDLE_SECONDS_BEFORE_LIKELY_WAITING,
            "target_dir": _target_dir,
            "error": _error,
            "log_tail": list(_log_lines[-200:]),
            "bundled": is_bundled(),
            "bundled_dir": str(bundled_backend_dir()),
            "progress": progress,
        }


def _estimate_progress(lines: list[str]) -> int:
    """Rough 0-100 progress estimate for the install, parsed from
    the log output. The installer is a black box (install.bat), so
    this is heuristic: pip/uv download percentages are parsed when
    present, otherwise we fall back to stage-based estimation.

    Takes the log lines as an argument instead of locking, because
    its only caller already holds the lock."""
    import re

    if not lines:
        return 0

    # Look for pip/uv download percentages in recent lines
    pct = None
    for line in reversed(lines[-50:]):
        # Match patterns like "Downloading torch... 45%" or "torch 45.2/100MB"
        m = re.search(r"(\d{1,3})\s*%", line)
        if m:
            pct = int(m.group(1))
            break
        # uv style: "Downloading torch (1.2GB) 45/100"
        m = re.search(r"(\d{1,3})\s*/\s*100", line)
        if m:
            pct = int(m.group(1))
            break

    if pct is not None:
        return min(99, pct)

    # Stage-based fallback: how far through the installer's known steps the log is
    text = "\n".join(lines).lower()
    stages = [("creating venv", 5), ("torch", 15), ("ramtorch", 45), ("customized-optimizers", 50),
              ("xformers", 55), ("torchao", 60), ("lycoris", 65), ("requirements", 72),
              ("accelerate", 80), ("completed installing", 85), ("sdxl-path dependencies", 88),
              ("diffusers", 92)]
    best = 0
    for key, val in stages:
        if key in text:
            best = max(best, val)
    if best:
        return best
    return 2 if "running" in text else 0


def _append(line: str) -> None:
    global _last_output_at
    with _lock:
        _log_lines.append(line.rstrip("\n"))
        if len(_log_lines) > _MAX_LOG_LINES:
            del _log_lines[: len(_log_lines) - _MAX_LOG_LINES]
        _last_output_at = time.time()


def _set_stage(stage: str, error: str | None = None) -> None:
    global _stage, _error
    with _lock:
        _stage = stage
        _error = error


def is_busy() -> bool:
    with _lock:
        return _stage == "installing"


def maybe_autostart_install() -> None:
    """Called once when Anima Studio itself starts up (see app/__init__.py):
    if the bundled backend's source is here but its dependencies have
    never been installed, kick off that install automatically in the
    background - so by the time someone gets to the Train tab, it's
    often already done (or visibly in progress there), instead of
    needing to separately notice and click "Install dependencies" first.

    Fire-and-forget, same as backend_manager.maybe_autostart - never
    raises, never blocks app startup, and does nothing at all if the
    backend isn't bundled, is already installed, or is already running
    (e.g. this got called twice)."""
    if not is_bundled() or is_installed() or is_busy():
        return
    try:
        start()
    except SetupError:
        pass  # best-effort - the Train tab's card still works manually either way


def start(target_dir: str = "") -> dict:
    """Runs the bundled backend's own installer (dependencies only - the
    training code itself is already here). `target_dir` is only used as
    a fallback for the rare case someone points this at a different
    checkout instead of the bundled one."""
    global _target_dir, _last_output_at, _stage, _error
    target = Path(target_dir).expanduser() if target_dir else bundled_backend_dir()
    if not target.exists():
        raise SetupError(f"'{target}' doesn't exist.")

    # Claim the slot under one lock (check + set): two overlapping requests
    # must not both launch the installer, only one of which would be tracked.
    with _lock:
        if _stage == "installing":
            raise SetupError("Setup is already running - check the log below, or wait for it to finish.")
        _log_lines.clear()
        _last_output_at = time.time()
        _stage = "installing"
        _error = None
    _target_dir = str(target)

    threading.Thread(target=_run_install, args=(target,), daemon=True).start()
    return status()


def _run_install(target: Path) -> None:
    try:
        installer = _find_installer(target)
        if installer is None:
            _append(f"Couldn't find an install script (install.sh/install.bat) in {target}.")
            _set_stage("error", "No installer script found.")
            return

        _append(f"Running {installer.name} in {target} - this downloads PyTorch and the rest")
        _append("of the ML stack, so it can take a while.")
        _append("")
        _run_interactive(installer, target)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user via status()
        _append(f"Unexpected error: {exc}")
        _set_stage("error", str(exc))


def _find_installer(target: Path):
    # installer.py is run directly: install.bat ends in `pause`, which keeps
    # the process alive waiting for a key press so "finished" was never seen.
    candidate = target / "installer.py"
    return candidate if candidate.exists() else None


def _python_cmd() -> list[str]:
    import shutil
    for name in (("python", "py") if sys.platform == "win32" else ("python3", "python")):
        path = shutil.which(name)
        if path and "WindowsApps" not in path:
            return [path] + (["-3"] if name == "py" else [])
    if not getattr(sys, "frozen", False):
        return [sys.executable]
    raise SetupError("Python isn't installed (or isn't on PATH). Install Python 3.11 from python.org "
                     "and tick 'Add python.exe to PATH', then try again.")


def _run_interactive(script: Path, cwd: Path) -> None:
    global _proc
    args = _python_cmd() + ["-u", str(script)]
    proc = subprocess.Popen(
        args, cwd=str(cwd),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        # No visible console window for this child process (see the same
        # fix in backend_manager.py) - doesn't affect the piped stdin/
        # stdout this function relies on to feed it answers.
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        # UV_LINK_MODE=copy: venv and uv's cache are often on different drives, where uv can't
        # hardlink and would print its "falling back to full copy" warning; copying is what we want.
        env={**os.environ, "UV_LINK_MODE": "copy", "PYTHONUNBUFFERED": "1",
             "PATH": str(Path.home() / ".local" / "bin") + os.pathsep + os.environ.get("PATH", "")},
        **({"start_new_session": True} if sys.platform != "win32" else {}),
    )
    with _lock:
        _proc = proc

    try:
        # The installer's first question is "are you using this remotely?
        # (y/n)" - answering "y" means "yes, remote", which leads to a
        # different set of follow-up questions (ngrok/cloudflared tunnels)
        # that don't apply here, since Anima Studio always runs locally.
        # Auto-answering this one specific, always-the-same-answer question
        # avoids relying on someone reading the literal wording correctly
        # ("are you remote" reads easy to answer backwards if you're
        # thinking "yes, I'm local").
        proc.stdin.write("n\n")
        proc.stdin.flush()
        _append('(Auto-answered "are you using this remotely?" with "n" - Anima Studio')
        _append("always runs locally. If it asks anything else, use the box below.)")
        _append("")
    except OSError:
        pass  # if this fails, the manual input box below still covers it

    def _reader() -> None:
        global _proc
        try:
            for line in proc.stdout:
                _append(line)
        except Exception:  # noqa: BLE001 - the process going away mid-read is normal
            pass
        finally:
            try:
                code = proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                code = proc.poll()
            with _lock:
                _proc = None
            if code == 0 and not is_installed():
                _append("")
                _append("The installer finished but PyTorch isn't in the environment - something failed above (Git must also be installed: git-scm.com).")
                _set_stage("error", "Install didn't complete - read the log above.")
            elif code == 0 and (_install_native_backend_deps(cwd), missing_modules())[1]:
                _append("")
                _append(f"Still missing packages: {', '.join(missing_modules())} - repairing...")
                _set_stage("done")
                if not repair_runtime_deps():
                    _set_stage("error", "Couldn't install all backend packages - see the log.")
            elif code == 0:
                _append("")
                _append("Installer finished. 'Backend folder' below is already set to this")
                _append("folder - launching the backend automatically...")
                _set_stage("done")
                try:   # first-time install finished -> bring the backend up right away
                    from . import backend_manager
                    backend_manager.maybe_autostart()
                except Exception as exc:  # noqa: BLE001 - never break the installer thread
                    _append(f"Couldn't auto-launch the backend: {exc}")
            else:
                _set_stage("error", f"Installer exited with code {code} - scroll up to see why.")

    threading.Thread(target=_reader, daemon=True).start()


def _install_native_backend_deps(sd_scripts_dir: Path) -> bool:
    """After the interactive installer above finishes setting up
    sd_scripts' own venv, also installs native_backend's SDXL-path
    dependencies (diffusers/peft/etc) into that SAME venv - so one venv
    covers both training paths native_backend/server.py can run,
    instead of needing a second, separate environment just for SDXL."""
    venv_python = bundled_backend_dir() / "sd_scripts" / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    requirements = bundled_backend_dir().parent / "native_backend" / "requirements.txt"
    if not venv_python.exists() or not requirements.exists():
        return False
    _append("")
    _append("Installing native_backend's own SDXL-path dependencies (diffusers, peft, ...)")
    _append("into the same virtual environment...")
    try:
        proc = subprocess.Popen(
            [str(venv_python), "-m", "pip", "install", "-r", str(requirements)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        for line in proc.stdout:
            _append(line)
        return proc.wait() == 0
    except OSError as exc:
        _append(f"Couldn't install native_backend's dependencies automatically: {exc}")
        _append(f"You can run this yourself: {venv_python} -m pip install -r {requirements}")
        return False


def _venv_python() -> Path:
    return bundled_backend_dir() / "sd_scripts" / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


_REQUIRED_MODULES = ("flask", "einops", "accelerate", "transformers", "diffusers", "peft", "safetensors",
                     "toml", "ftfy", "kornia", "voluptuous", "cv2", "scipy", "torchvision")


def missing_modules() -> list[str]:
    """Modules the trainer imports that the backend venv doesn't have (empty = all good)."""
    py = _venv_python()
    if not py.exists():
        return list(_REQUIRED_MODULES)
    code = ("import importlib.util as u;"
            f"print(','.join(m for m in {list(_REQUIRED_MODULES)!r} if u.find_spec(m) is None))")
    try:
        r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        if r.returncode != 0:
            return list(_REQUIRED_MODULES)
        return [m for m in r.stdout.strip().split(",") if m]
    except (OSError, subprocess.SubprocessError):
        return list(_REQUIRED_MODULES)


def _pip(args: list[str]) -> bool:
    py = _venv_python()
    try:
        proc = subprocess.Popen(
            [str(py), "-m", "pip", "install", "--default-timeout=120", "--retries", "10", *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=str(bundled_backend_dir() / "sd_scripts"),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        for line in proc.stdout:
            _append(line)
        return proc.wait() == 0
    except OSError as exc:
        _append(f"pip failed to start: {exc}")
        return False


def _install_requirements_file(path: Path) -> list[str]:
    """pip install -r <file>; if the whole file fails (one bad git dependency
    sinks it), fall back to line by line. Returns the lines that failed."""
    if not path.exists():
        return []
    _append(f"Installing packages from {path.name} ...")
    if _pip(["-r", str(path)]):
        return []
    failed = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #")[0].strip()
        if not line or line.startswith(("#", "-e", "--")):
            continue
        if ";" in line:
            marker = line.split(";", 1)[1]
            if ("Linux" in marker and sys.platform == "win32") or ("Windows" in marker and sys.platform != "win32"):
                continue
        _append(f"> {line}")
        if not _pip([line]):
            failed.append(line)
    return failed


def repair_runtime_deps() -> bool:
    """Installed venv (PyTorch present) but the rest of the packages are missing - e.g. an
    earlier install stopped half way: finish the job (no PyTorch re-download), then launch."""
    if is_busy() or not is_installed():
        return False
    if not missing_modules():
        return False

    def run():
        _set_stage("installing")
        with _lock:
            _log_lines.clear()
        miss = missing_modules()
        _append(f"Missing in the backend environment: {', '.join(miss)} - installing them now.")
        base = bundled_backend_dir()
        failed = _install_requirements_file(base / "sd_scripts" / "requirements.txt")
        sched = base / "custom_scheduler"
        if sched.exists():
            _pip(["-e", str(sched)])
        failed += _install_requirements_file(base / "requirements.txt")
        failed += _install_requirements_file(base.parent / "native_backend" / "requirements.txt")
        still = missing_modules()
        if not still:
            _append("")
            _append("All packages installed - launching the backend...")
            if failed:
                _append("(Optional packages that failed: " + "; ".join(failed) + ")")
            _set_stage("done")
            try:
                from . import backend_manager
                backend_manager.maybe_autostart()
            except Exception as exc:  # noqa: BLE001
                _append(f"Couldn't auto-launch the backend: {exc}")
        else:
            _append("")
            _append(f"Still missing: {', '.join(still)}. Check your internet connection and that Git is installed, then press Install again.")
            _set_stage("error", f"Couldn't install: {', '.join(still)} - see the log.")

    threading.Thread(target=run, daemon=True, name="backend-repair").start()
    return True


def send_input(text: str) -> dict:
    with _lock:
        proc = _proc
    if proc is None or proc.poll() is not None or proc.stdin is None:
        raise SetupError("Nothing is waiting for input right now.")
    try:
        proc.stdin.write(text.rstrip("\n") + "\n")
        proc.stdin.flush()
    except OSError as exc:
        raise SetupError(f"Couldn't send that: {exc}") from exc
    _append(f"> {text}")
    return status()


# ------------------------------------------------------------- verification
# "Is this backend actually able to train?" - a checklist rather than a single
# yes/no, because the interesting failures are partial ones: a venv whose
# PyTorch never finished downloading, a CUDA build that cannot see the GPU, a
# package that pip skipped. is_installed() only proves a venv exists with a
# torch/ folder, which is not the same as a trainer that will start.

# Files the vendored backend cannot start without. Relative to training_backend/.
_REQUIRED_BACKEND_FILES = (
    "main.py",
    "sd_scripts/anima_train_network.py",
    "sd_scripts/sdxl_train_network.py",
    "sd_scripts/networks/lora_anima.py",
    "sd_scripts/library/anima_models.py",
    "custom_scheduler",
)

_MIN_FREE_GB = 10.0

# One interpreter launch answers every question that needs the venv: three
# separate launches would each pay a multi-second `import torch`.
_PROBE_CODE = """
import importlib.util as u, json, sys
required = %(required)r
out = {"python": sys.version.split()[0],
       "missing": [m for m in required if u.find_spec(m) is None],
       "scheduler": u.find_spec("LoraEasyCustomOptimizer") is not None,
       "torch": None, "torch_error": ""}
try:
    import torch
    cuda_ok = bool(torch.cuda.is_available())
    out["torch"] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda or "",
        "available": cuda_ok,
        "name": torch.cuda.get_device_name(0) if cuda_ok else "",
        "count": torch.cuda.device_count() if cuda_ok else 0,
        "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)) if cuda_ok else "",
    }
except Exception as exc:
    out["torch_error"] = "%%s: %%s" %% (type(exc).__name__, exc)
print("@@VERIFY " + json.dumps(out))
""" % {"required": list(_REQUIRED_MODULES)}


def _probe() -> dict:
    """Run the questions that need the venv's own interpreter. Never raises."""
    py = _venv_python()
    if not py.exists():
        return {}
    try:
        r = subprocess.run([str(py), "-c", _PROBE_CODE], capture_output=True, text=True, timeout=180,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"probe_error": str(exc)}
    for line in (r.stdout or "").splitlines():
        if line.startswith("@@VERIFY "):
            try:
                return json.loads(line[len("@@VERIFY "):])
            except ValueError:
                return {"probe_error": "the environment returned unreadable answers"}
    detail = ((r.stderr or r.stdout or "").strip().splitlines() or ["no output"])[-1]
    return {"probe_error": detail}


def _free_gb(path: Path) -> float | None:
    try:
        anchor = path if path.exists() else path.parent
        return shutil.disk_usage(str(anchor)).free / (1024 ** 3)
    except OSError:
        return None


def verify() -> dict:
    """Check the backend install end to end and say what to do about problems.

    Returns {"checks": [{"id", "label", "status", "detail", "fix"}], "ok": bool,
    "failed": int, "warned": int}. `status` is "ok", "warn" or "fail"; only a
    "fail" means training cannot start.
    """
    checks: list[dict] = []

    def add(item_id: str, label: str, status: str, detail: str, fix: str = "") -> None:
        checks.append({"id": item_id, "label": label, "status": status, "detail": detail, "fix": fix})

    base = bundled_backend_dir()

    # --- the backend source that ships with the app ------------------------
    missing = [rel for rel in _REQUIRED_BACKEND_FILES if not (base / rel).exists()]
    if missing:
        add("files", "Backend files", "fail",
            f"{len(missing)} of {len(_REQUIRED_BACKEND_FILES)} expected files are missing from training_backend/: "
            + ", ".join(missing[:6]) + ("…" if len(missing) > 6 else ""),
            "Re-extract the Anima Studio archive: the bundled backend is incomplete.")
    else:
        add("files", "Backend files", "ok",
            f"All {len(_REQUIRED_BACKEND_FILES)} expected files are present in training_backend/.")

    server_py = config.APP_DIR / "native_backend" / "server.py"
    add("trainer_server", "Trainer server code", "ok" if server_py.exists() else "fail",
        f"native_backend/ is present ({server_py.parent})." if server_py.exists()
        else f"native_backend/server.py not found next to the app ({server_py.parent}).",
        "" if server_py.exists() else "Re-extract the Anima Studio archive: native_backend/ is missing.")

    # --- the venv ----------------------------------------------------------
    py = _venv_python()
    if not py.exists():
        add("venv", "Backend environment", "fail",
            f"No interpreter at {py} - the backend dependencies were never installed.",
            "Open the Backend tab and press Install dependencies (a multi-GB download).")
        probe: dict = {}
    else:
        add("venv", "Backend environment", "ok", f"Interpreter found at {py}.")
        probe = _probe()
        if probe.get("probe_error"):
            add("probe", "Environment answers", "warn",
                f"The environment could not be queried: {probe['probe_error']}",
                "If training fails with import errors, run Install dependencies again.")
        elif probe:
            add("probe", "Environment answers", "ok",
                f"Python {probe.get('python', '?')} answered from the backend environment.")

    # --- torch and CUDA: the check that actually decides "can this train" ---
    torch_info = probe.get("torch") if probe else None
    if torch_info:
        cap = f" (compute {torch_info['capability']})" if torch_info.get("capability") else ""
        if torch_info.get("available"):
            add("torch", "PyTorch + CUDA", "ok",
                f"torch {torch_info['version']} (CUDA {torch_info['cuda'] or '?'}), "
                f"{torch_info['count']} GPU(s) visible: {torch_info['name']}{cap}.")
        else:
            add("torch", "PyTorch + CUDA", "fail",
                f"torch {torch_info['version']} is installed but sees no CUDA device "
                f"(build says CUDA {torch_info['cuda'] or 'none'}).",
                "Reinstall the backend with the build that matches your GPU, or update the NVIDIA driver.")
    elif probe.get("torch_error"):
        add("torch", "PyTorch + CUDA", "fail",
            f"PyTorch could not be imported in the backend environment: {probe['torch_error']}",
            "Press Install dependencies in the Backend tab to repair the environment.")
    elif py.exists() and not probe:
        add("torch", "PyTorch + CUDA", "warn", "Could not check PyTorch in the backend environment.",
            "Press Install dependencies to repair, then verify again.")

    # --- the rest of the packages -----------------------------------------
    if probe and "missing" in probe:
        miss = probe["missing"]
        if miss:
            add("packages", "Training packages", "fail",
                f"{len(miss)} required package(s) missing from the backend environment: {', '.join(miss)}.",
                "Press Start on the Backend tab - it installs missing packages automatically before launching.")
        else:
            add("packages", "Training packages", "ok",
                f"All {len(_REQUIRED_MODULES)} required packages import in the backend environment.")
    elif py.exists():
        miss = missing_modules()
        if miss:
            add("packages", "Training packages", "fail",
                f"{len(miss)} required package(s) missing: {', '.join(miss)}.",
                "Press Start on the Backend tab - it installs missing packages automatically before launching.")
        else:
            add("packages", "Training packages", "ok", "Required packages are all present.")

    # --- the custom schedulers (an editable install pip can quietly drop) ---
    if probe and "scheduler" in probe:
        if probe["scheduler"]:
            add("scheduler", "Custom schedulers", "ok", "LoraEasyCustomOptimizer is importable.")
        else:
            add("scheduler", "Custom schedulers", "warn",
                "LoraEasyCustomOptimizer is not importable - RexAnnealing and friends will not exist.",
                "Press Install dependencies in the Backend tab to reinstall it from training_backend/custom_scheduler.")

    # --- is the backend actually running and answering ---------------------
    from . import backend_manager      # imported here: backend_manager imports this module
    proc = backend_manager.status()
    if proc.get("running"):
        add("process", "Backend process", "ok",
            f"Running: {proc.get('command', '')} (started {_ago(proc.get('started_at'))}).")
    else:
        add("process", "Backend process", "warn", "The backend is not running.",
            "Press Start on the Backend tab - training needs it running." + (f" Last error: {proc['error']}" if proc.get("error") else ""))

    try:
        from . import backend_client
        ping = backend_client.ping(config.DEFAULT_BACKEND_URL)
        if ping.get("reachable"):
            add("reachable", "Backend responds", "ok", f"{config.DEFAULT_BACKEND_URL} answered.")
        else:
            add("reachable", "Backend responds", "warn",
                f"{config.DEFAULT_BACKEND_URL} did not answer.",
                "Start the backend, or correct the Backend URL on the Train tab.")
    except Exception as exc:      # noqa: BLE001 - a health check must never take the page down
        add("reachable", "Backend responds", "warn", f"Could not reach the backend: {exc}",
            "Start the backend and check the Backend URL.")

    # --- room to train in --------------------------------------------------
    free = _free_gb(base)
    if free is not None:
        where = base.anchor or str(base)
        if free < _MIN_FREE_GB:
            add("disk", "Free disk space", "warn",
                f"Only {free:.1f} GB free on {where} - cached latents and checkpoints are large.",
                "Free some space, or point the dataset/output folders at another drive.")
        else:
            add("disk", "Free disk space", "ok", f"{free:.1f} GB free on {where}.")

    failed = sum(1 for c in checks if c["status"] == "fail")
    warned = sum(1 for c in checks if c["status"] == "warn")
    return {"checks": checks, "ok": failed == 0, "failed": failed, "warned": warned,
            "installed": is_installed(), "partial": is_partial()}


def _ago(when) -> str:
    if not when:
        return "earlier"
    secs = max(0, int(time.time() - float(when)))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    return f"{secs // 3600}h ago"
