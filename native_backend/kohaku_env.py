"""Where the KohakuMoss training environment lives.

KohakuMossTrainer (by KohakuBlueLeaf, vendored unmodified in kohaku_backend/) is its own training program with its
own packages (torch 2.13-2.14, transformers 5.3+, diffusers 0.38+, kohaku-engine, LyCORIS 4.1.3+), so it gets its own
virtual environment in kohaku_backend/venv - installing it next to the sd-scripts engines would change packages they
depend on. Stdlib only: both the Studio server and the training backend import it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KOHAKU_DIR = ROOT / "kohaku_backend"
VENV = KOHAKU_DIR / "venv"
USER_CONFIGS = KOHAKU_DIR / "configs" / "user"           # configs Anima Studio writes for each run
EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""


def _bin(name: str) -> Path:
    return VENV / ("Scripts" if sys.platform == "win32" else "bin") / (name + EXE_SUFFIX)


def venv_python() -> Path:
    return _bin("python")


def kogine_exe() -> Path:
    """KohakuEngine's runner (`kogine run <script> --config <file> --set KEY=VALUE`)."""
    return _bin("kogine")


def doctor_exe() -> Path:
    return _bin("kohakumoss-doctor")


def config_exe() -> Path:
    return _bin("kohakumoss-config")


def has_venv() -> bool:
    return venv_python().exists()


def has_kogine() -> bool:
    return kogine_exe().exists()


def has_trainer() -> bool:
    return (KOHAKU_DIR / "scripts" / "train" / "moss.py").is_file() and \
        (KOHAKU_DIR / "src" / "kohakumoss" / "__init__.py").is_file()
