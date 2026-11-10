"""Where the AI edit environment lives (stdlib only, shared by the Flask app and the tests)."""
from __future__ import annotations

import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
VENV = DIR / "venv"
_BIN = "Scripts" if sys.platform == "win32" else "bin"
_EXE = ".exe" if sys.platform == "win32" else ""


def venv_python() -> Path:
    return VENV / _BIN / ("python" + _EXE)


def has_venv() -> bool:
    return venv_python().exists()


def has_iopaint() -> bool:
    """IOPaint is installed in the environment (looked up on disk: the app itself has no torch to import it with)."""
    return any(VENV.glob("Lib/site-packages/iopaint/__init__.py")) or \
        any(VENV.glob("lib/python3*/site-packages/iopaint/__init__.py"))


def has_worker() -> bool:
    return (DIR / "worker.py").is_file() and (DIR / "catalog.py").is_file()
