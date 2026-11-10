"""Where the MOSS-Image training environment lives.

MOSS-Image trains on its own vendored sd-scripts fork (moss_backend/sd_scripts) inside its own
virtual environment (moss_backend/venv): the fork pins torch 2.14 + CUDA 13 (cuDNN 9.24), which the
Anima / SDXL engine does not. Stdlib only - both the Studio server and the training backend import it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOSS_DIR = ROOT / "moss_backend"
SD_SCRIPTS = MOSS_DIR / "sd_scripts"
VENV = MOSS_DIR / "venv"
OVERLAY = SD_SCRIPTS / "moss_te_deps"          # transformers 5 for the Qwen3.5 text encoder (installed on setup)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def has_venv() -> bool:
    return venv_python().exists()


def has_trainer() -> bool:
    return (SD_SCRIPTS / "moss_train_network.py").is_file() and (SD_SCRIPTS / "networks" / "lora_moss.py").is_file()
