#!/usr/bin/env python3
"""
Anima Studio - desktop entry point.

Runs the exact same app as run.py, but inside its own native window
(via pywebview) instead of opening a browser tab - this is what
build_windows.bat packages into Anima Studio.exe, and what run.bat
launches by default (via pythonw, with no console window at all).

Run directly with:  python desktop.py
(needs `pywebview` installed - see requirements.txt)
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import traceback

from app import create_app
from app import config

_LOG_PATH = config.DATA_DIR / "desktop_launch.log"


def _log_failure(message: str) -> None:
    """Writes startup failures to a file, not just stdout - run.bat and
    the built .exe both launch this with no console window at all
    (pythonw / a windowed PyInstaller build), so a bare print() here
    would vanish with nobody able to see why nothing opened."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{message}\n")
    except OSError:
        pass  # best-effort - stdout below still covers the "has a console" case
    print(message, file=sys.stderr)


def _free_port(preferred: int) -> int:
    """Uses the configured port if it's free, otherwise asks the OS for
    any free one - two copies of the app (or a leftover process from a
    crash) shouldn't stop this one from opening a window."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((config.HOST, preferred))
            return preferred
        except OSError:
            s.bind((config.HOST, 0))
            return s.getsockname()[1]


def _wait_until_up(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def main() -> None:
    try:
        import webview
    except ImportError:
        _log_failure(
            "Desktop mode needs the 'pywebview' package.\n"
            "Install it with:  pip install pywebview\n"
            "(or just run 'python run.py' to use it in your browser instead)"
        )
        return

    try:
        app = create_app()
        port = _free_port(config.PORT)
        server = threading.Thread(
            target=lambda: app.run(host=config.HOST, port=port, debug=False, threaded=True, use_reloader=False),
            daemon=True,
        )
        server.start()

        if not _wait_until_up(config.HOST, port):
            _log_failure("Anima Studio's local server didn't come up in time - try again.")
            return

        webview.create_window(
            "Anima Studio",
            f"http://{config.HOST}:{port}",
            width=1440,
            height=920,
            min_size=(1024, 680),
            background_color="#0a0d0f",
        )
        webview.start()
        # Window closed: stop the training backend and anything it launched, then exit for real.
        from app.services import procs
        procs.shutdown_all()
        os._exit(0)
    except Exception:  # noqa: BLE001 - this IS the top-level handler; there's no console to see a raw traceback in
        _log_failure(f"Anima Studio failed to start:\n{traceback.format_exc()}\n\nLog file: {_LOG_PATH}")


if __name__ == "__main__":
    main()
