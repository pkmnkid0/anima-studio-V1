"""
Local launcher for Anima Studio's own training backend
(native_backend/, see native_backend/NOTES.md) - this module saves the
user from having to open a second terminal and run it by hand: given
the folder it lives in and the command to start it, it launches that as
a child process, tracks it, and can stop it again.

The default command runs native_backend/server.py using the *bundled*
backend's own venv Python (training_backend/sd_scripts/venv) - that's
what already has Anima training's dependencies installed (via "Set up
backend"), and native_backend's own SDXL-path dependencies
(diffusers/peft/etc, see native_backend/requirements.txt) get installed
into that same venv, so there's one venv to manage, not two.
training_backend/ itself is no longer something you run directly - it's
a library native_backend/anima_trainer.py imports from.

Deliberately simple: one backend process at a time, tracked in memory
(not persisted across an Anima Studio restart - if this process dies, any
backend it launched is treated as gone too, same as closing a terminal
window would). Settings (folder + command + auto-start) ARE persisted via
settings_store so they don't need re-entering every session.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import settings_store
from .. import config


def _default_command() -> str:
    """native_backend/server.py, run with the bundled backend's own venv
    Python so Anima training's dependencies are available to it without
    a second, separate install."""
    venv_python = "training_backend/sd_scripts/venv/Scripts/python.exe" if sys.platform == "win32" \
        else "training_backend/sd_scripts/venv/bin/python"
    return f"{venv_python} -m native_backend.server"


DEFAULT_COMMAND = _default_command()

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_launch_dir: str | None = None
_command: str | None = None
_started_at: float | None = None
_last_error: str | None = None
_log_lines: list[str] = []
_MAX_LOG_LINES = 400


class BackendLaunchError(RuntimeError):
    pass


def get_settings() -> dict:
    # native_backend/server.py is a module (`-m native_backend.server`),
    # so it needs to be launched with cwd=the project root, not
    # training_backend/ - unlike the old vendored server, which was its
    # own standalone script run from inside its own folder.
    default_dir = str(config.APP_DIR) if (config.APP_DIR / "native_backend").exists() else ""
    saved = settings_store.get("backend_launcher") or {}
    out = {"dir": default_dir, "command": DEFAULT_COMMAND, "autostart": True}
    out.update({k: v for k, v in saved.items() if v not in (None, "")})
    # Auto-start is ON by default. A stored False only counts when the user
    # actually flipped the checkbox themselves ("autostart_set"); older
    # versions wrote False automatically, which silently kept it off.
    if not saved.get("autostart_set"):
        out["autostart"] = True
    out.pop("autostart_set", None)
    return out


def save_settings(directory: str, command: str, autostart: bool, explicit: bool = True) -> dict:
    prev = settings_store.get("backend_launcher") or {}
    settings = {"dir": directory or "", "command": command or DEFAULT_COMMAND, "autostart": bool(autostart),
                "autostart_set": bool(explicit or prev.get("autostart_set"))}
    settings_store.set("backend_launcher", settings)
    settings = dict(settings)
    settings.pop("autostart_set", None)
    return settings


def _reader_thread(proc: subprocess.Popen) -> None:
    global _last_error
    try:
        for line in proc.stdout:
            with _lock:
                _log_lines.append(line.rstrip("\n"))
                if len(_log_lines) > _MAX_LOG_LINES:
                    del _log_lines[: len(_log_lines) - _MAX_LOG_LINES]
    except Exception:  # noqa: BLE001 - the process going away mid-read is normal on stop()
        pass
    finally:
        code = proc.poll()
        with _lock:
            if code not in (0, None) and _proc is proc:
                _last_error = f"Backend process exited with code {code}. Check the log below."


def start(directory: str, command: str | None = None) -> dict:
    global _proc, _launch_dir, _command, _started_at, _last_error
    with _lock:
        if _proc is not None and _proc.poll() is None:
            raise BackendLaunchError("A backend process is already running - stop it first.")

        d = Path(directory) if directory else None
        if not d or not d.is_dir():
            raise BackendLaunchError(f"'{directory}' isn't a folder Anima Studio can see.")

        command = (command or DEFAULT_COMMAND).strip()
        try:
            args = shlex.split(command, posix=(sys.platform != "win32"))
        except ValueError as exc:
            raise BackendLaunchError(f"Couldn't parse that start command: {exc}") from exc
        # Windows shlex keeps surrounding quotes ("C:\Program Files\...") -
        # strip matched pairs so a pasted path with spaces still runs.
        args = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a for a in args]
        if not args:
            raise BackendLaunchError("Start command is empty.")

        _log_lines.clear()
        _last_error = None
        try:
            _proc = subprocess.Popen(
                args, cwd=str(d), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
                env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                # Suppresses the console window Windows would otherwise pop
                # up for this child process - Anima Studio runs with no
                # console of its own (desktop.py/the built .exe), and this
                # backend process shouldn't spawn a visible one either.
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                **({"start_new_session": True} if sys.platform != "win32" else {}),
            )
        except OSError as exc:
            _proc = None
            venv_python = d / args[0]
            if not venv_python.exists():
                raise BackendLaunchError(
                    f"'{args[0]}' doesn't exist in {d} - dependencies haven't been installed into "
                    "*this* copy of the backend yet. This is separate from any other copy you may "
                    "have set up before (e.g. if this is a built .exe, it bundles the backend's "
                    "source code fresh, not an existing install) - click \"Set up backend\" above, "
                    "then try Launch again once that finishes."
                ) from exc
            raise BackendLaunchError(
                f"Couldn't launch '{command}' in {d}: {exc}. "
                "Check the folder and command are right (e.g. the backend's own venv Python)."
            ) from exc

        _launch_dir = str(d)
        _command = command
        _started_at = time.time()
        proc_ref = _proc

    threading.Thread(target=_reader_thread, args=(proc_ref,), daemon=True).start()
    return status()


def stop() -> dict:
    global _proc, _last_error
    with _lock:
        proc = _proc
    if proc is None or proc.poll() is not None:
        with _lock:
            _proc = None
        return status()

    from .procs import kill_tree
    # Clear the tracking BEFORE killing: the reader thread reports a
    # non-zero exit as a crash, and a deliberate stop must not surface as one.
    with _lock:
        _proc = None
        _last_error = None
    kill_tree(proc)   # the backend's own helpers (accelerate, trainer) go with it
    return status()


def status() -> dict:
    with _lock:
        running = _proc is not None and _proc.poll() is None
        return {
            "running": running,
            "dir": _launch_dir,
            "command": _command,
            "started_at": _started_at if running else None,
            "error": _last_error,
            "log_tail": list(_log_lines[-60:]),
        }


def maybe_autostart() -> None:
    """Called once at app startup - launches the backend automatically if
    the user has turned that on in the launcher settings."""
    settings = get_settings()
    if not settings.get("autostart") or not settings.get("dir"):
        return
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return
    try:
        start(settings["dir"], settings.get("command"))
    except BackendLaunchError as exc:
        global _last_error
        with _lock:
            _last_error = f"Auto-start failed: {exc}"
