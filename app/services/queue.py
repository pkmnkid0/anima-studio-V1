"""Training queue.

A persistent, ordered, single-worker queue of training runs. Each job
stores the full Train-tab UI payload plus the backend URL. One daemon
worker takes the next queued job and runs it exactly the way the Start
button does: build the payload, `validate` (which hands the config to the
backend), then `start_training`, then it polls until the run finishes.

Behaviour worth knowing:
  * Jobs run one at a time, in the order shown. Reorder with move_job().
  * If the backend is already training something (started by hand), the
    worker waits for it to go idle instead of clobbering it.
  * If the backend can't be reached, the job goes back to the queue and the
    queue pauses with a note, rather than failing every job in a row.
  * Pause only stops the *next* job from starting; the running one finishes.
  * Jobs are saved to disk. A job that was running when the app closed is
    marked as interrupted (the backend may have kept training, so it is not
    restarted automatically).
"""
from __future__ import annotations

import json
import threading
import time
import uuid

from .. import config, train_config
from . import backend_client

QUEUE_DIR = config.DATA_DIR / "queue"
QUEUE_FILE = QUEUE_DIR / "jobs.json"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
FINISHED = (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED)

POLL_SECONDS = 5.0
START_GRACE_SECONDS = 180.0     # time the backend gets to report "training" after we start it

_io = threading.RLock()          # guards the jobs file (Flask threads + worker)
_state_lock = threading.Lock()
_worker: threading.Thread | None = None
_paused = False
_note = ""
_cancel: set[str] = set()
_recovered = False


# ---------------------------------------------------------------- storage

def _read_all() -> dict:
    if not QUEUE_FILE.is_file():
        return {}
    try:
        data = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_all(data: dict) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(QUEUE_FILE)


def _recover_once() -> None:
    global _recovered
    with _io:
        if _recovered:
            return
        _recovered = True
        jobs = _read_all()
        changed = False
        for j in jobs.values():
            if j.get("status") == STATUS_RUNNING:
                j["status"] = STATUS_ERROR
                j["finished"] = time.time()
                j["result"] = ("Interrupted: Anima Studio was closed while this job was running. "
                               "The backend may have kept training - check it, then use Retry.")
                changed = True
        if changed:
            _write_all(jobs)


def _ordered(jobs: dict) -> list[dict]:
    return sorted(jobs.values(), key=lambda j: (j.get("order", j.get("created", 0)), j.get("created", 0)))


# ---------------------------------------------------------------- public API

def list_jobs() -> list[dict]:
    """Running + queued jobs in run order, then finished jobs newest-first."""
    _recover_once()
    with _io:
        jobs = _read_all()
    live = [j for j in _ordered(jobs) if j["status"] in (STATUS_RUNNING, STATUS_QUEUED)]
    live.sort(key=lambda j: 0 if j["status"] == STATUS_RUNNING else 1)  # stable: keeps order within groups
    done = sorted((j for j in jobs.values() if j["status"] in FINISHED),
                  key=lambda j: j.get("finished") or 0, reverse=True)
    return live + done


def _summarize(ui: dict) -> dict:
    try:
        payload = train_config.build_backend_payload(ui)
        saving = payload["args"].get("saving_args", {})
        return {
            "output_name": saving.get("output_name") or "",
            "model_type": ui.get("model_type") or ("sdxl" if payload["args"].get("general_args", {}).get("sdxl") else "anima"),
            "train_mode": payload.get("train_mode", "lora"),
            "subsets": len(payload.get("subsets", [])),
        }
    except Exception:  # noqa: BLE001 - a summary is cosmetic; never block queueing on it
        return {"output_name": "", "model_type": ui.get("model_type", ""), "train_mode": ui.get("train_mode", "lora"), "subsets": 0}


def add_job(name: str, ui: dict, backend_url: str) -> dict:
    _recover_once()
    if not isinstance(ui, dict) or not ui:
        raise ValueError("No training config to queue.")
    if not (backend_url or "").strip():
        raise ValueError("Set the backend URL first.")
    # Fail now (not hours later) when the config can't even be built.
    try:
        train_config.build_backend_payload(ui)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"This config can't be queued: {exc}") from exc
    summary = _summarize(ui)
    with _io:
        jobs = _read_all()
        name = (name or "").strip() or summary["output_name"] or f"Run {len(jobs) + 1}"
        last = max((j.get("order", 0) for j in jobs.values()), default=0)
        job_id = uuid.uuid4().hex[:8]
        job = {
            "id": job_id, "name": name, "ui": ui, "backend_url": backend_url.strip(),
            "summary": summary, "status": STATUS_QUEUED, "created": time.time(), "order": last + 1,
            "started": None, "finished": None, "result": None, "progress": None,
        }
        jobs[job_id] = job
        _write_all(jobs)
    _ensure_worker()
    return job


def remove_job(job_id: str) -> bool:
    with _io:
        jobs = _read_all()
        j = jobs.get(job_id)
        if not j or j["status"] == STATUS_RUNNING:
            return False
        del jobs[job_id]
        _write_all(jobs)
        return True


def cancel_job(job_id: str) -> dict | None:
    with _io:
        jobs = _read_all()
        j = jobs.get(job_id)
        if not j:
            return None
        if j["status"] == STATUS_RUNNING:
            _cancel.add(job_id)           # the worker stops the backend, then finalises
            return j
        if j["status"] == STATUS_QUEUED:
            j.update(status=STATUS_CANCELLED, finished=time.time(), result="Cancelled before it started.")
            _write_all(jobs)
        return j


def retry_job(job_id: str) -> dict | None:
    with _io:
        jobs = _read_all()
        j = jobs.get(job_id)
        if not j or j["status"] not in FINISHED:
            return None
        j.update(status=STATUS_QUEUED, finished=None, started=None, result=None, progress=None,
                 order=max((x.get("order", 0) for x in jobs.values()), default=0) + 1)
        _write_all(jobs)
    _ensure_worker()
    return j


def move_job(job_id: str, direction: str) -> bool:
    """direction: up | down | top. Only queued jobs are reordered."""
    with _io:
        jobs = _read_all()
        queued = [j for j in _ordered(jobs) if j["status"] == STATUS_QUEUED]
        ids = [j["id"] for j in queued]
        if job_id not in ids:
            return False
        i = ids.index(job_id)
        if direction == "top":
            ids.insert(0, ids.pop(i))
        elif direction == "up" and i > 0:
            ids[i - 1], ids[i] = ids[i], ids[i - 1]
        elif direction == "down" and i < len(ids) - 1:
            ids[i + 1], ids[i] = ids[i], ids[i + 1]
        else:
            return False
        for n, jid in enumerate(ids, start=1):
            jobs[jid]["order"] = n
        _write_all(jobs)
        return True


def clear_finished() -> int:
    with _io:
        jobs = _read_all()
        gone = [k for k, j in jobs.items() if j["status"] in FINISHED]
        for k in gone:
            del jobs[k]
        if gone:
            _write_all(jobs)
        return len(gone)


def rename_job(job_id: str, name: str) -> bool:
    name = (name or "").strip()
    if not name:
        return False
    with _io:
        jobs = _read_all()
        if job_id not in jobs:
            return False
        jobs[job_id]["name"] = name
        _write_all(jobs)
        return True


def state() -> dict:
    return {"paused": _paused, "note": _note, "worker_alive": bool(_worker and _worker.is_alive())}


def set_paused(paused: bool) -> dict:
    global _paused, _note
    _paused = bool(paused)
    if not _paused:
        _note = ""
        _ensure_worker()
    return state()


# ---------------------------------------------------------------- worker

def _ensure_worker() -> None:
    global _worker
    with _state_lock:
        if _worker and _worker.is_alive():
            return
        _worker = threading.Thread(target=_loop, name="train-queue", daemon=True)
        _worker.start()


def _claim_next() -> dict | None:
    with _io:
        jobs = _read_all()
        for j in _ordered(jobs):
            if j["status"] == STATUS_QUEUED:
                j.update(status=STATUS_RUNNING, started=time.time(), result=None, progress=None)
                _write_all(jobs)
                return j
    return None


def _loop() -> None:
    idle_since = time.monotonic()
    while True:
        if _paused:
            time.sleep(1.0)
            continue
        job = _claim_next()
        if job is None:
            # nothing to do; exit after a short idle so the thread doesn't linger forever
            if time.monotonic() - idle_since > 30:
                return
            time.sleep(1.0)
            continue
        idle_since = time.monotonic()
        _run_job(job)
        idle_since = time.monotonic()


def _update(job_id: str, **fields) -> None:
    with _io:
        jobs = _read_all()
        if job_id in jobs:
            jobs[job_id].update(fields)
            _write_all(jobs)


def _requeue(job_id: str, note: str) -> None:
    global _paused, _note
    _update(job_id, status=STATUS_QUEUED, started=None, progress=None)
    _paused = True
    _note = note


def _finish(job_id: str, status: str, result: str) -> None:
    _update(job_id, status=status, finished=time.time(), result=result)


def _stop_backend(url: str) -> None:
    try:
        backend_client.stop_training(url)
    except backend_client.BackendError:
        pass


def _run_job(job: dict) -> None:
    job_id, ui, url = job["id"], job["ui"], job["backend_url"]
    try:
        # 1) wait for the backend to be idle (a hand-started run may be going)
        while True:
            if job_id in _cancel:
                _cancel.discard(job_id)
                return _finish(job_id, STATUS_CANCELLED, "Cancelled while waiting for the backend.")
            try:
                busy = backend_client.is_training(url).get("training")
            except backend_client.BackendUnreachable:
                return _requeue(job_id, f"Paused: the training backend isn't reachable at {url}. "
                                        "Start it (Train > Backend), then press Resume.")
            if not busy:
                break
            _update(job_id, progress={"waiting": True})
            time.sleep(POLL_SECONDS)

        # 2) same steps as the Start button: validate (hands over the config), then start
        payload = train_config.build_backend_payload(ui)
        dataset = {**payload["dataset"], "subsets": payload["subsets"]}
        backend_client.validate(url, payload["args"], dataset)
        backend_client.start_training(
            url, train_mode=payload["train_mode"],
            has_sdxl=bool(payload["args"].get("general_args", {}).get("sdxl")),
            has_flux=bool(payload["args"].get("flux_args")),
            has_anima=bool(payload["args"].get("anima_args")),
            accelerate=ui.get("accelerate", {}),
        )

        # 3) follow it to the end
        started = time.monotonic()
        seen_training = False
        while True:
            time.sleep(POLL_SECONDS)
            if job_id in _cancel:
                _cancel.discard(job_id)
                _stop_backend(url)
                return _finish(job_id, STATUS_CANCELLED, "Cancelled by you; stop signal sent to the backend.")
            try:
                st = backend_client.is_training(url)
            except backend_client.BackendUnreachable:
                continue          # backend busy/restarting; keep waiting
            if st.get("training"):
                seen_training = True
                try:
                    prog = backend_client.get_progress(url)
                    if prog:
                        _update(job_id, progress=prog)
                except backend_client.BackendError:
                    pass
                continue
            if st.get("errored"):
                return _finish(job_id, STATUS_ERROR, "Training errored - check the backend log.")
            if seen_training:
                return _finish(job_id, STATUS_DONE, "Training completed.")
            if time.monotonic() - started > START_GRACE_SECONDS:
                return _finish(job_id, STATUS_ERROR, "The backend accepted the job but never reported training. Check its log.")
    except backend_client.BackendUnreachable:
        _requeue(job_id, f"Paused: lost contact with the training backend at {url}. Start it, then press Resume.")
    except backend_client.BackendError as exc:
        _finish(job_id, STATUS_ERROR, str(exc))
    except Exception as exc:  # noqa: BLE001
        _finish(job_id, STATUS_ERROR, f"Unexpected error: {exc}")
