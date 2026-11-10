"""Run one AI edit tool over every image of a dataset folder (Edit tab -> "AI batch").

Remove the background, upscale, restore faces, or apply the same painted mask to every picture (a watermark that sits
in the same place on every image) - one image at a time through the edit engine, each result written back the same
way the single-image editor does it (backup first, or a copy next to the original). A failure on one picture is
recorded and the batch carries on; Cancel stops the engine, so even a long Stable Diffusion step ends at once.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from . import datasets, edit_engine, imaging

_lock = threading.Lock()
_state: dict = {"state": "idle", "total": 0, "done": 0, "failed": [], "current": "", "message": "", "started": None,
                "finished": None}
_cancel = threading.Event()


class BatchError(RuntimeError):
    pass


def status() -> dict:
    with _lock:
        s = dict(_state)
        s["failed"] = list(_state["failed"][-50:])
        s["failed_count"] = len(_state["failed"])
    return s


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def start(body: dict) -> dict:
    folder = str(body.get("folder") or "")
    mode = body.get("mode")
    if mode not in ("plugin", "inpaint"):
        raise BatchError("Choose what to run on the folder.")
    if mode == "inpaint" and not body.get("mask"):
        raise BatchError("Paint the mask on one image first: the same mask is applied to every image.")
    try:
        images = datasets.list_images(folder)
    except NotADirectoryError as exc:
        raise BatchError(str(exc)) from None
    if not images:
        raise BatchError("That folder has no images.")
    if not edit_engine.is_installed():
        raise BatchError("The AI edit environment is not installed yet - set it up in the Backend tab first.")
    # Validate one request now (model name, options, ranges) so a typo fails here, not on image 1 of 400.
    probe = _request(body, images[0]["path"], mode)
    try:
        cmd, args, tmp = edit_engine.build_request(probe)
    except edit_engine.EngineError as exc:
        raise BatchError(str(exc)) from None
    for p in tmp:
        p.unlink(missing_ok=True)
    Path(args["out_path"]).unlink(missing_ok=True)
    with _lock:
        if _state["state"] == "running":
            raise BatchError("A batch is already running.")
    if not edit_engine.reserve_for_batch():
        raise BatchError("The AI engine is busy with another job - wait for it to finish first.")
    _cancel.clear()
    with _lock:
        _state.update(state="running", total=len(images), done=0, failed=[], current="", message="",
                      started=time.time(), finished=None)
    threading.Thread(target=_run, args=(body, mode, [i["path"] for i in images]), daemon=True,
                     name="edit-batch").start()
    return status()


def cancel() -> dict:
    _cancel.set()
    with _lock:
        running = _state["state"] == "running"
    if running:
        edit_engine.stop("")
    return status()


def _request(body: dict, path: str, mode: str) -> dict:
    req = {"cmd": "plugin" if mode == "plugin" else "inpaint", "image_path": path,
           "options": body.get("options") or {}}
    if mode == "plugin":
        for key in ("plugin", "model", "scale", "bg_upscale", "bg_model"):
            if key in body:
                req[key] = body[key]
        req["op"] = "image"
    else:
        req["model"] = body.get("model") or "lama"
        req["params"] = body.get("params") or {}
        req["mask"] = body.get("mask")
    return req


def _run(body: dict, mode: str, paths: list[str]) -> None:
    try:
        for i, path in enumerate(paths):
            if _cancel.is_set():
                _set(state="cancelled", message="Cancelled.", finished=time.time())
                return
            name = Path(path).name
            _set(current=name)
            out = None
            tmp: list[Path] = []
            try:
                cmd, args, tmp = edit_engine.build_request(_request(body, path, mode))
                out = args["out_path"]
                edit_engine.call(cmd, args)
                imaging.save_ai_result(path, out, save_as_copy=bool(body.get("save_as_copy")),
                                       keep_alpha=bool(body.get("keep_alpha")),
                                       background=str(body.get("background") or "#ffffff"))
            except edit_engine.EngineCancelled:
                _set(state="cancelled", message="Cancelled.", finished=time.time())
                return
            except (edit_engine.EngineError, imaging.ImagingError) as exc:
                with _lock:
                    _state["failed"].append({"filename": name, "error": str(exc)[:300]})
            finally:
                for p in tmp:
                    p.unlink(missing_ok=True)
                if out:
                    Path(out).unlink(missing_ok=True)
            _set(done=i + 1)
        with _lock:
            failed = len(_state["failed"])
        _set(state="done", current="", finished=time.time(),
             message=f"Finished. {failed} image(s) failed." if failed else "Finished.")
    except Exception as exc:                                  # noqa: BLE001 - never leave the batch "running"
        _set(state="error", message=f"{type(exc).__name__}: {exc}", finished=time.time())
    finally:
        edit_engine.release_batch()
