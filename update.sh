#!/usr/bin/env bash
# Updates Anima Studio's code from your GitHub repository (see update.bat for options).
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Run run.sh once first so the app is set up."; exit 1; }
"$PY" update.py "$@"
