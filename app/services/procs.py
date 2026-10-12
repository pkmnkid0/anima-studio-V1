"""Child-process cleanup, so closing Anima Studio never leaves Python running.

The training backend starts helpers of its own (accelerate, the trainer), so
stopping only the process we launched is not enough - the whole tree goes.
"""
from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys

NEW_GROUP = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True}


def kill_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=15,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
    except Exception:  # noqa: BLE001 - shutdown must never raise
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def shutdown_all() -> None:
    """Stop every process Anima Studio launched. Safe to call more than once."""
    from . import backend_manager, backend_setup
    for mod in (backend_manager, backend_setup):
        try:
            kill_tree(getattr(mod, "_proc", None))
        except Exception:  # noqa: BLE001
            pass


atexit.register(shutdown_all)
