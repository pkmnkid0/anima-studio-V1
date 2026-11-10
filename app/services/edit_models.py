"""Download several AI edit models in one go (Backend tab -> AI edit tools -> Models).

The Edit tab downloads a model the first time it is used; this lets the person fetch a whole set up front instead,
one after the other through the same engine, skipping what is already on disk. A failure on one model is recorded and
the rest carry on. Cancel stops the engine, so even a large download ends at once.
"""
from __future__ import annotations

import threading
import time

from edit_backend import catalog

from . import edit_engine

_lock = threading.Lock()
_cancel = threading.Event()
_state: dict = {"state": "idle", "total": 0, "done": 0, "skipped": 0, "failed": [], "current": "", "message": "",
                "started": None, "finished": None}


class ModelsError(RuntimeError):
    pass


def status() -> dict:
    with _lock:
        s = dict(_state)
        s["failed"] = list(_state["failed"])
    return s


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def _label_of(kind: str, name: str) -> str:
    for grp in catalog.pick_groups():
        for it in grp["items"]:
            if it["name"] == name and (it["kind"] == kind or (kind in ("gfpgan", "restoreformer") and it["kind"] == "face")):
                return it["label"]
    return name


def resolve(body: dict) -> list[tuple[str, str]]:
    """The (kind, name) list for {"pack": id} or {"items": [{"kind", "name"}, ...]}, validated against the catalog."""
    valid = {(i["kind"], i["name"]) for grp in catalog.pick_groups() for i in grp["items"]}
    if body.get("pack"):
        pack = next((p for p in catalog.MODEL_PACKS if p["id"] == body["pack"]), None)
        if pack is None:
            raise ModelsError("Unknown model set.")
        return list(pack["items"])
    items = []
    for it in body.get("items") or []:
        kind, name = str((it or {}).get("kind") or ""), str((it or {}).get("name") or "")
        # the face models are two plugins named after the model
        if (kind, name) not in valid:
            raise ModelsError(f"Unknown model: {name}")
        pair = (name, name) if kind == "face" else (kind, name)
        if pair not in items:
            items.append(pair)
    if not items:
        raise ModelsError("Choose at least one model.")
    return items


def _have(known: dict, kind: str, name: str) -> bool:
    if kind == "model":
        return name in (known.get("downloaded") or [])
    plugins = known.get("plugins") or {}
    key = {"sam": "sam:", "realesrgan": "realesrgan:", "removebg": "removebg:"}.get(kind)
    if key:
        return bool(plugins.get(key + name))
    return bool(plugins.get("face:" + name))


def start(body: dict) -> dict:
    items = resolve(body)
    if not edit_engine.is_installed():
        raise ModelsError("The AI edit environment is not installed yet - set it up first.")
    for kind, name in items:                                   # fail now on a bad name, not halfway through
        try:
            edit_engine.build_request({"cmd": "download", "kind": kind, "name": name})
        except edit_engine.EngineError as exc:
            raise ModelsError(str(exc)) from None
    with _lock:
        if _state["state"] == "running":
            raise ModelsError("A model download is already running.")
    if not edit_engine.reserve_for_batch():
        raise ModelsError("The AI engine is busy - try again when the current job has finished.")
    _cancel.clear()
    _set(state="running", total=len(items), done=0, skipped=0, failed=[], current="", message="Starting the engine…",
         started=time.time(), finished=None)
    threading.Thread(target=_run, args=(items,), daemon=True, name="edit-models").start()
    return status()


def _run(items: list[tuple[str, str]]) -> None:
    try:
        try:
            known = edit_engine.call("models")
        except edit_engine.EngineError:
            known = {}
        for kind, name in items:
            if _cancel.is_set():
                break
            label = _label_of(kind, name)
            if _have(known, kind, name):
                with _lock:
                    _state["done"] += 1
                    _state["skipped"] += 1
                continue
            _set(current=label, message=f"Downloading {label}…")
            try:
                edit_engine.call("download", {"kind": kind, "name": name, "device": "cpu", "no_half": False})
            except edit_engine.EngineCancelled:
                break
            except edit_engine.EngineError as exc:
                with _lock:
                    _state["failed"].append({"name": label, "error": str(exc)[:300]})
            with _lock:
                _state["done"] += 1
        stopped = _cancel.is_set()
        s = status()
        ok = s["done"] - len(s["failed"]) - s["skipped"]
        msg = "Stopped." if stopped else f"Done: {ok} downloaded, {s['skipped']} already there" + (
            f", {len(s['failed'])} failed." if s["failed"] else ".")
        _set(state="cancelled" if stopped else "done", current="", message=msg, finished=time.time())
    except Exception as exc:                      # noqa: BLE001
        _set(state="error", message=str(exc)[:300], finished=time.time())
    finally:
        edit_engine.release_batch()


def cancel() -> dict:
    if status()["state"] == "running":
        _cancel.set()
        edit_engine.stop("Model download stopped.")
    return status()
