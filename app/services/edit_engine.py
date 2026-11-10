"""The AI edit engine: starts and talks to edit_backend/worker.py (IOPaint inside its own environment).

One worker process, one request at a time, JSON lines over stdin/stdout (no network port). Jobs are what the browser
sees: `submit()` returns at once and the page polls `job()` for progress; `call()` is the blocking form the batch runner
uses. The worker is started on first use, stopped after a few idle minutes (so it does not hold GPU memory while you
train) and stopped for good by Cancel, which kills it - the next job simply starts a fresh one.
"""
from __future__ import annotations

import base64
import binascii
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from edit_backend import catalog
from edit_backend import env as edit_env

from .. import config
from . import procs, settings_store

DEFAULT_IDLE_MINUTES = 15
READY_TIMEOUT = 600                  # importing torch + IOPaint on a spinning disk can take minutes the first time
TMP_DIR = config.DATA_DIR / "edit_tmp"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_TOKEN = re.compile(r"^[0-9a-f]{32}$")
_MODEL_NAME = re.compile(r"^[\w .+:/@()\-]{1,200}$", re.UNICODE)
_PCT = re.compile(r"(\d{1,3})%\|")

UI_COMMANDS = ("models", "check", "unload", "load_model", "download", "inpaint", "segment", "plugin")
DOWNLOAD_KINDS = ("model", "sam", "removebg", "realesrgan", "gfpgan", "restoreformer")
PLUGIN_KINDS = ("removebg", "realesrgan", "gfpgan", "restoreformer")


class EngineError(RuntimeError):
    """A failure whose message is meant for the person using the editor."""


class EngineCancelled(EngineError):
    pass


# ------------------------------------------------------------------ settings

def models_dir() -> Path:
    raw = str(settings_store.get("edit_models_dir", "") or "").strip().strip('"')
    return Path(raw).expanduser() if raw else config.MODELS_CACHE_DIR / "edit_models"


def idle_minutes() -> int:
    try:
        return max(0, int(settings_store.get("edit_idle_minutes", DEFAULT_IDLE_MINUTES)))
    except (TypeError, ValueError):
        return DEFAULT_IDLE_MINUTES


def set_settings(models_dir_value: str | None = None, idle: int | None = None) -> dict:
    if models_dir_value is not None:
        raw = str(models_dir_value).strip().strip('"')
        if raw:
            p = Path(raw).expanduser()
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise EngineError(f"Can't use that folder: {exc}") from None
        if raw != str(settings_store.get("edit_models_dir", "") or ""):
            settings_store.set("edit_models_dir", raw)
            stop("The models folder changed.")
    if idle is not None:
        settings_store.set("edit_idle_minutes", max(0, min(24 * 60, int(idle))))
    return status()


# ------------------------------------------------------------------ state

_lock = threading.RLock()                       # guards the fields below
_run_lock = threading.Lock()                    # one worker request at a time
_proc: subprocess.Popen | None = None
_ready = threading.Event()
_stderr: deque = deque(maxlen=300)
_inflight: dict | None = None                   # {"id", "job", "done", "resp", "cancelled"}
_active_job: str | None = None                  # the job (or "batch") currently using the engine
_jobs: dict[str, dict] = {}
_seq = itertools.count(1)
_last_activity = time.time()
_loaded: dict = {"model": None, "device": None}
_known: dict = {"models": None}
_watcher_started = False


def _touch() -> None:
    global _last_activity
    _last_activity = time.time()


def is_installed() -> bool:
    return edit_env.has_venv() and edit_env.has_iopaint() and edit_env.has_worker()


def status() -> dict:
    with _lock:
        alive = _proc is not None and _proc.poll() is None
        return {
            "installed": is_installed(), "worker_present": edit_env.has_worker(), "running": alive,
            "ready": alive and _ready.is_set(), "loaded": dict(_loaded), "busy": _active_job,
            "models_dir": str(models_dir()), "idle_minutes": idle_minutes(), "known": _known["models"],
            "stderr_tail": list(_stderr)[-12:],
        }


# ------------------------------------------------------------------ process

def _child_env() -> dict:
    md = models_dir()
    md.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "XDG_CACHE_HOME": str(md), "U2NET_HOME": str(md), "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONPATH": str(edit_env.ROOT)}
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "TRANSFORMERS_CACHE"):
        env.pop(key, None)                      # one models folder, the one shown in the editor
    return env


def _spawn() -> subprocess.Popen:
    global _proc, _ready, _watcher_started
    if not is_installed():
        raise EngineError("The AI edit environment is not installed yet - open the Backend tab and press "
                          "'Set up AI edit environment'.")
    cleanup_tmp()
    _stderr.clear()
    ready = threading.Event()
    proc = subprocess.Popen(
        [str(edit_env.venv_python()), "-u", "-m", "edit_backend.worker"], cwd=str(edit_env.ROOT), env=_child_env(),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, **procs.NEW_GROUP)
    procs.register(proc)
    with _lock:
        _proc, _ready = proc, ready
        _loaded.update(model=None, device=None)
    threading.Thread(target=_read_stdout, args=(proc, ready), daemon=True, name="edit-stdout").start()
    threading.Thread(target=_read_stderr, args=(proc,), daemon=True, name="edit-stderr").start()
    if not _watcher_started:
        _watcher_started = True
        threading.Thread(target=_watch_idle, daemon=True, name="edit-idle").start()
    return proc


def _read_stdout(proc: subprocess.Popen, ready: threading.Event) -> None:
    try:
        for raw in iter(proc.stdout.readline, b""):                     # type: ignore[union-attr]
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                _note_stderr(line)
                continue
            _on_message(msg, ready)
    finally:
        _on_exit(proc)


def _read_stderr(proc: subprocess.Popen) -> None:
    buf = b""
    try:
        while True:
            chunk = proc.stderr.read(2048)                              # type: ignore[union-attr]
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]+", buf)
            buf = parts.pop()
            for part in parts:
                text = part.decode("utf-8", "replace").rstrip()
                if text.strip():
                    _note_stderr(text)
    finally:
        if buf.strip():
            _note_stderr(buf.decode("utf-8", "replace").rstrip())


def _note_stderr(text: str) -> None:
    text = text[-400:]
    with _lock:
        if _stderr and _PCT.search(text) and _PCT.search(_stderr[-1]):
            _stderr[-1] = text                                           # a progress bar redraws one line
        else:
            _stderr.append(text)
        slot = _inflight
    if slot and slot.get("job") is not None:
        job = slot["job"]
        m = _PCT.search(text)
        if m:
            job["progress"]["percent"] = min(100, int(m.group(1)))
            job["progress"]["message"] = text.strip()[:160]
        _job_log(job, text)


def _job_log(job: dict, text: str) -> None:
    log = job["log"]
    if log and _PCT.search(text) and _PCT.search(log[-1]):
        log[-1] = text
    else:
        log.append(text)
    del log[:-120]


def _on_message(msg: dict, ready: threading.Event) -> None:
    ev = msg.get("event")
    if ev == "ready":
        ready.set()
        return
    with _lock:
        slot = _inflight
    if ev == "log":
        _note_stderr(str(msg.get("line", "")))
        return
    if ev == "progress":
        if slot and slot.get("job") is not None:
            p = slot["job"]["progress"]
            p["step"] = msg.get("step")
            if msg.get("total"):
                p["total"] = msg.get("total")
            if msg.get("message"):
                p["message"] = msg.get("message")
        return
    if "ok" in msg and slot and msg.get("id") == slot["id"]:
        slot["resp"] = msg
        slot["done"].set()


def _on_exit(proc: subprocess.Popen) -> None:
    global _proc
    with _lock:
        if _proc is proc:
            _proc = None
        slot = _inflight
    if slot is not None:
        slot["done"].set()                                              # resp stays None: the worker is gone


def stop(reason: str = "") -> bool:
    """Stop the worker (frees its GPU memory). A job running at that moment ends as cancelled."""
    with _lock:
        proc, slot = _proc, _inflight
    if proc is None or proc.poll() is not None:
        return False
    if slot is not None:
        slot["cancelled"] = True
        slot["why"] = reason
    procs.kill_tree(proc)
    return True


def _watch_idle() -> None:
    while True:
        time.sleep(30)
        try:
            mins = idle_minutes()
            with _lock:
                alive = _proc is not None and _proc.poll() is None
                idle_for = time.time() - _last_activity
                busy = _active_job is not None
            if alive and not busy and mins > 0 and idle_for > mins * 60:
                stop("idle")
        except Exception:                                               # noqa: BLE001 - the watcher must survive
            pass


def _send(proc: subprocess.Popen, obj: dict) -> None:
    data = (json.dumps(obj, ensure_ascii=True) + "\n").encode("utf-8")
    try:
        proc.stdin.write(data)                                          # type: ignore[union-attr]
        proc.stdin.flush()                                              # type: ignore[union-attr]
    except (OSError, ValueError):
        raise EngineError("The AI engine is not running (it may have crashed) - try again.") from None


def _ensure_worker(job: dict | None) -> subprocess.Popen:
    with _lock:
        proc, ready = _proc, _ready
    if proc is None or proc.poll() is not None:
        if job is not None:
            job["progress"]["message"] = "Starting the AI engine (the first start can take a minute) ..."
        proc = _spawn()
        with _lock:
            ready = _ready
    deadline = time.time() + READY_TIMEOUT
    while not ready.wait(0.5):
        if job is not None and job.get("_cancelled"):
            stop("")
            raise EngineCancelled("Stopped.")
        if proc.poll() is not None:
            tail = " | ".join(list(_stderr)[-4:]) or "no output"
            raise EngineError(f"The AI engine failed to start: {tail[:500]}")
        if time.time() > deadline:
            stop("startup timeout")
            raise EngineError("The AI engine did not start in time - see the Backend tab and re-run Verify.")
    return proc


def call(cmd: str, args: dict | None = None, job: dict | None = None) -> dict:
    """Run one worker command and wait for it. Raises EngineError / EngineCancelled with a readable message."""
    global _inflight
    with _run_lock:
        _touch()
        proc = _ensure_worker(job)
        slot = {"id": next(_seq), "job": job, "done": threading.Event(), "resp": None, "cancelled": False}
        with _lock:
            _inflight = slot
        try:
            if job is not None and job.get("_cancelled"):
                raise EngineCancelled("Stopped.")
            _send(proc, {"id": slot["id"], "cmd": cmd, **(args or {})})
            slot["done"].wait()
        finally:
            with _lock:
                _inflight = None
            _touch()
        resp = slot["resp"]
        if resp is None:
            if slot["cancelled"]:
                raise EngineCancelled("Stopped." if slot.get("why") in ("", None) else f"Stopped: {slot['why']}")
            tail = " | ".join(list(_stderr)[-3:]) or "no output"
            raise EngineError(f"The AI engine stopped unexpectedly: {tail[:400]}")
        if not resp.get("ok"):
            raise EngineError(str(resp.get("error") or "The AI engine reported an error."))
        return resp.get("result") or {}


# ------------------------------------------------------------------ temp files

def cleanup_tmp(max_age: float = 24 * 3600) -> None:
    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - max_age
        for p in TMP_DIR.iterdir():
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass
    except OSError:
        pass


def new_token() -> str:
    return uuid.uuid4().hex


def tmp_path(token: str, suffix: str = ".png") -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return TMP_DIR / f"{token}{suffix}"


def result_file(token: str) -> Path | None:
    if not isinstance(token, str) or not _TOKEN.match(token):
        return None
    p = TMP_DIR / f"{token}.png"
    return p if p.is_file() else None


def discard_result(token: str) -> bool:
    p = result_file(token)
    if p is None:
        return False
    try:
        p.unlink()
    except OSError:
        return False
    return True


def decode_data_url(data: str) -> bytes:
    if not isinstance(data, str) or not data:
        raise EngineError("No image data was sent.")
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        raise EngineError("The image data is invalid - paint the area again.") from None
    if len(raw) > MAX_UPLOAD_BYTES:
        raise EngineError("That image is too large.")
    return raw


# ------------------------------------------------------------------ requests from the browser

def _clean_options(raw) -> dict:
    """Engine options the browser may set: only the catalogue's keys, with the right types."""
    raw = raw if isinstance(raw, dict) else {}
    out: dict = {}
    for spec in catalog.ENGINE_OPTIONS:
        key = spec["key"]
        if key not in raw:
            continue
        val = raw[key]
        if spec["kind"] == "bool":
            out[key] = bool(val)
        elif spec["kind"] == "select":
            out[key] = val if val in spec["options"] else spec["default"]
    return out


def model_name(value) -> str:
    name = str(value or "").strip()
    if not name:
        raise EngineError("Choose a model first.")
    if not _MODEL_NAME.match(name) or ".." in name:
        raise EngineError("That model name is not valid.")
    return name


def image_file(path) -> str:
    p = Path(str(path or ""))
    if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
        raise EngineError(f"No such image: {path}")
    return str(p)


def build_request(body: dict) -> tuple[str, dict, list[Path]]:
    """Validate a browser request and turn it into (worker command, worker arguments, temp files to delete after)."""
    cmd = body.get("cmd")
    if cmd not in UI_COMMANDS:
        raise EngineError(f"Unknown command '{cmd}'.")
    options = _clean_options(body.get("options"))
    tmp: list[Path] = []
    if cmd in ("models", "check", "unload"):
        return cmd, {}, tmp
    if cmd == "load_model":
        return cmd, {"model": model_name(body.get("model")), "options": options}, tmp
    if cmd == "download":
        kind = body.get("kind", "model")
        if kind not in DOWNLOAD_KINDS:
            raise EngineError("Unknown download kind.")
        name = str(body.get("name") or "").strip()
        if kind == "model":
            name = model_name(name)
        elif name and not _MODEL_NAME.match(name):
            raise EngineError("That model name is not valid.")
        return cmd, {"kind": kind, "name": name, "device": options.get("device", "auto"),
                     "no_half": options.get("no_half", False)}, tmp

    token = new_token()
    out = tmp_path(token)
    args: dict = {"out_path": str(out), "options": options, "device": options.get("device", "auto")}
    if cmd == "inpaint":
        args["image_path"] = image_file(body.get("image_path"))
        args["model"] = model_name(body.get("model") or "lama")
        params = body.get("params")
        args["params"] = params if isinstance(params, dict) else {}
        if body.get("mask"):
            mask = tmp_path(token, ".mask.png")
            mask.write_bytes(decode_data_url(body["mask"]))
            args["mask_path"] = str(mask)
            tmp.append(mask)
    elif cmd == "segment":
        args["image_path"] = image_file(body.get("image_path"))
        clicks = body.get("clicks")
        if not isinstance(clicks, list) or not clicks or len(clicks) > 200:
            raise EngineError("Click on the picture first.")
        try:
            args["clicks"] = [[int(c[0]), int(c[1]), 1 if int(c[2]) else 0] for c in clicks]
        except (TypeError, ValueError, IndexError):
            raise EngineError("The clicks are not valid.") from None
        args["model"] = str(body.get("model") or "mobile_sam")
        if not _MODEL_NAME.match(args["model"]):
            raise EngineError("That model name is not valid.")
    elif cmd == "plugin":
        kind = body.get("plugin")
        if kind not in PLUGIN_KINDS:
            raise EngineError("Unknown tool.")
        op = body.get("op", "image")
        if op not in ("image", "mask"):
            raise EngineError("Unknown operation.")
        model = str(body.get("model") or "")
        if model and not _MODEL_NAME.match(model):
            raise EngineError("That model name is not valid.")
        try:
            scale = float(body.get("scale") or 2.0)
        except (TypeError, ValueError):
            raise EngineError("The scale is not a number.") from None
        if not 1.0 <= scale <= 8.0:
            raise EngineError("The scale must be between 1 and 8.")
        args.update(image_path=image_file(body.get("image_path")), plugin=kind, op=op, model=model, scale=scale,
                    no_half=bool(options.get("no_half")), bg_upscale=bool(body.get("bg_upscale")),
                    bg_model=str(body.get("bg_model") or ""))
        if args["bg_model"] and not _MODEL_NAME.match(args["bg_model"]):
            raise EngineError("That model name is not valid.")
    return cmd, args, tmp


def public_result(result: dict, out_path: str | None) -> dict:
    """The worker's result without server paths: the picture is referred to by its token."""
    res = {k: v for k, v in result.items() if k != "out_path"}
    if out_path and Path(out_path).is_file():
        res["token"] = Path(out_path).name.split(".")[0]
    return res


# ------------------------------------------------------------------ jobs

def _public_job(job: dict) -> dict:
    return {k: (list(job["log"][-40:]) if k == "log" else v) for k, v in job.items() if not k.startswith("_")}


def submit(body: dict) -> dict:
    """Start a job from a browser request; returns its public view at once."""
    global _active_job
    cmd, args, tmp = build_request(body)
    job = {"id": uuid.uuid4().hex[:12], "cmd": cmd, "state": "running", "progress": {"step": None, "total": None,
           "percent": None, "message": ""}, "log": [], "result": None, "error": None, "started": time.time(),
           "finished": None}
    with _lock:
        if _active_job is not None:
            for p in tmp:
                p.unlink(missing_ok=True)
            raise EngineError("The AI engine is busy with another job - wait for it to finish, or press Cancel.")
        _active_job = job["id"]
        _jobs[job["id"]] = job
        for old in sorted(_jobs.values(), key=lambda j: j["started"])[:-40]:
            _jobs.pop(old["id"], None)
    threading.Thread(target=_run_job, args=(job, cmd, args, tmp), daemon=True, name="edit-job").start()
    return _public_job(job)


def _run_job(job: dict, cmd: str, args: dict, tmp: list[Path]) -> None:
    global _active_job
    try:
        result = call(cmd, args, job=job)
        job["result"] = public_result(result, args.get("out_path"))
        if cmd == "load_model":
            _loaded.update(model=result.get("loaded"), device=result.get("device"))
        elif cmd == "inpaint":
            _loaded.update(model=args.get("model"))
        elif cmd == "models":
            _known["models"] = {k: result.get(k) for k in ("downloaded", "plugins", "loaded", "loaded_plugins")}
        elif cmd == "unload":
            _loaded.update(model=None)
        job["state"] = "done"
    except EngineCancelled as exc:
        job["state"], job["error"] = "cancelled", str(exc)
    except EngineError as exc:
        job["state"], job["error"] = "error", str(exc)
    except Exception as exc:                                            # noqa: BLE001 - never leave a job running
        job["state"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
    finally:
        job["finished"] = time.time()
        for p in tmp:
            p.unlink(missing_ok=True)
        if job["state"] != "done" and args.get("out_path"):
            Path(args["out_path"]).unlink(missing_ok=True)
        with _lock:
            if _active_job == job["id"]:
                _active_job = None


def job(job_id: str) -> dict | None:
    with _lock:
        j = _jobs.get(job_id)
        return _public_job(j) if j else None


def cancel(job_id: str | None = None) -> bool:
    """Cancel the running job by stopping the worker (the only reliable way to interrupt a model)."""
    with _lock:
        active = _active_job
        j = _jobs.get(active) if active else None
    if active is None or (job_id and job_id != active):
        return False
    if j is not None:
        j["_cancelled"] = True                  # honoured even while the worker is still starting up
    stop("")
    return True


def reserve_for_batch() -> bool:
    """Marks the engine busy while a batch runs; False when something else is already using it."""
    global _active_job
    with _lock:
        if _active_job is not None:
            return False
        _active_job = "batch"
        return True


def release_batch() -> None:
    global _active_job
    with _lock:
        if _active_job == "batch":
            _active_job = None
