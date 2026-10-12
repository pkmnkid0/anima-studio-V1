"""Training history (§18) - every run recorded, comparable.

Runs are appended to DATA_DIR/training_history.jsonl so the
Train tab can list past runs and compare configurations,
loss curves and outputs across attempts.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from .. import config

_HISTORY = config.DATA_DIR / "training_history.jsonl"


def _read_all() -> list[dict]:
    if not _HISTORY.is_file():
        return []
    runs = []
    for line in _HISTORY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def _write_all(runs: list[dict]) -> None:
    _HISTORY.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY.write_text(
        "".join(json.dumps(r, sort_keys=False) + "\n" for r in runs),
        encoding="utf-8")


def dataset_hash(folder: str) -> str:
    """Stable-ish hash of a dataset folder: names + sizes of
    the images, so identical datasets compare equal."""
    root = Path(folder)
    digest = hashlib.sha256()
    if root.is_dir():
        for p in sorted(root.iterdir(), key=lambda p: p.name):
            if p.is_file():
                digest.update(p.name.encode())
                digest.update(str(p.stat().st_size).encode())
    return digest.hexdigest()[:16]


def record(run: dict) -> dict:
    """Append a finished (or failed) run. Required keys:
    name, model_type. Everything else is stored verbatim."""
    entry = dict(run)
    entry.setdefault("id", uuid.uuid4().hex[:8])
    entry.setdefault("started", time.time())
    entry.setdefault("finished", time.time())
    if "dataset_hash" not in entry and entry.get("dataset_folder"):
        entry["dataset_hash"] = dataset_hash(entry["dataset_folder"])
    runs = _read_all()
    runs.append(entry)
    _write_all(runs)
    return entry


def list_runs(limit: int = 100) -> list[dict]:
    return _read_all()[-limit:][::-1]


def get_run(run_id: str) -> dict | None:
    for run in _read_all():
        if run.get("id") == run_id:
            return run
    return None


def delete_run(run_id: str) -> bool:
    runs = _read_all()
    kept = [r for r in runs if r.get("id") != run_id]
    if len(kept) == len(runs):
        return False
    _write_all(kept)
    return True


def compare(run_ids: list[str]) -> list[dict]:
    """Side-by-side subset of the recorded fields."""
    keys = ("name", "model_type", "project_type", "network_dim",
            "network_alpha", "learning_rate", "optimizer_type",
            "lr_scheduler", "max_train_steps", "max_train_epochs",
            "resolution", "batch_size", "final_loss", "steps",
            "dataset_hash", "output")
    rows = []
    for run_id in run_ids:
        run = get_run(run_id)
        if run:
            rows.append({k: run.get(k) for k in keys})
    return rows
