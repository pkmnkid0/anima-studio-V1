"""
Thin client for the LoRA_Easy_Training_Scripts backend - the actual
training engine (sd-scripts fork + custom optimizers) that the uploaded
reference project drives. This app does not reimplement DiT training
itself; it talks to that backend over the same local HTTP API the
reference desktop app uses, so a config built here trains through the
exact engine the user already has installed per that project's README.

Protocol (reverse-engineered from the reference app's MainUI.py):
  POST {url}/validate   body: {"args", "dataset", "accelerate"}  -> {"tags": {...}}
  GET  {url}/train       params: train_mode, sdxl, flux, anima, accelerate_*
  GET  {url}/is_training -> {"training": bool, "errored": bool}
  GET  {url}/stop_training
  GET  {url}/check_path   params: path -> {"exists": bool}
"""
from __future__ import annotations

import requests

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
TIMEOUT_SHORT = 10
TIMEOUT_TRAIN_START = 30


class BackendError(RuntimeError):
    pass


class BackendUnreachable(BackendError):
    pass


def _clean(url: str) -> str:
    return url.rstrip("/") if url else DEFAULT_BACKEND_URL


def ping(url: str) -> dict:
    """Cheap reachability check. Any HTTP response (even a 404) counts as
    "reachable" - we only care whether something is listening."""
    url = _clean(url)
    try:
        resp = requests.get(f"{url}/is_training", timeout=TIMEOUT_SHORT)
        return {"reachable": True, "status_code": resp.status_code}
    except requests.exceptions.RequestException as exc:
        return {"reachable": False, "error": str(exc)}


def check_path(url: str, path: str) -> bool:
    url = _clean(url)
    try:
        resp = requests.get(f"{url}/check_path", params={"path": path}, timeout=TIMEOUT_SHORT)
        resp.raise_for_status()
        data = resp.json()
        return bool(data.get("exists", False))
    except requests.exceptions.RequestException as exc:
        raise BackendUnreachable(str(exc)) from exc


def validate(url: str, args: dict, dataset: dict, accelerate: dict | None = None) -> dict:
    url = _clean(url)
    body = {"args": args, "dataset": dataset, "accelerate": accelerate or {}}
    try:
        resp = requests.post(f"{url}/validate", json=body, timeout=TIMEOUT_SHORT * 3)
    except requests.exceptions.RequestException as exc:
        raise BackendUnreachable(
            f"Could not reach the training backend at {url}. "
            "Make sure it's running (see the backend's own README) and the URL is correct."
        ) from exc
    if resp.status_code != 200:
        raise BackendError(f"Backend rejected the config (HTTP {resp.status_code}): {resp.text}")
    try:
        return resp.json()
    except ValueError:
        return {}


def start_training(
    url: str,
    train_mode: str,
    has_sdxl: bool,
    has_flux: bool,
    has_anima: bool,
    accelerate: dict | None = None,
) -> None:
    url = _clean(url)
    params = {
        "train_mode": train_mode,
        "sdxl": str(bool(has_sdxl)),
        "flux": str(bool(has_flux)),
        "anima": str(bool(has_anima)),
    }
    accelerate = accelerate or {}
    if accelerate.get("enabled"):
        params["accelerate_enabled"] = "True"
        params["accelerate_num_processes"] = str(accelerate.get("num_processes", 2))
        params["accelerate_main_process_port"] = str(accelerate.get("main_process_port", 29500))
    try:
        resp = requests.get(f"{url}/train", params=params, timeout=TIMEOUT_TRAIN_START)
    except requests.exceptions.RequestException as exc:
        raise BackendUnreachable(str(exc)) from exc
    if resp.status_code != 200:
        raise BackendError(f"Backend refused to start training (HTTP {resp.status_code}): {resp.text}")


def is_training(url: str) -> dict:
    url = _clean(url)
    try:
        resp = requests.get(f"{url}/is_training", timeout=TIMEOUT_SHORT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        raise BackendUnreachable(str(exc)) from exc


def get_progress(url: str) -> dict | None:
    """Step/epoch/loss progress, if the backend exposes it - this is a
    native_backend/server.py extra, not part of the original contract
    the vendored backend speaks, so a 404 here just means "that backend
    doesn't report this", not an error worth surfacing."""
    url = _clean(url)
    try:
        resp = requests.get(f"{url}/status", timeout=TIMEOUT_SHORT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        raise BackendUnreachable(str(exc)) from exc


def stop_training(url: str) -> None:
    url = _clean(url)
    try:
        requests.get(f"{url}/stop_training", timeout=TIMEOUT_SHORT)
    except requests.exceptions.RequestException as exc:
        raise BackendUnreachable(str(exc)) from exc
