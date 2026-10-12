#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found on PATH. Install Python 3.10+ first."
    exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "Creating a virtual environment in .venv ..."
    python3 -m venv .venv
fi

echo "Installing/updating dependencies ..."
PIP_OPTS="--default-timeout=120 --retries 10"
.venv/bin/python -m pip install $PIP_OPTS --upgrade pip >/dev/null 2>&1 || true
.venv/bin/python -m pip install $PIP_OPTS -r requirements.txt

echo
echo "Starting Anima Studio ..."
.venv/bin/python run.py
