"""
Anima Studio's own training backend - server layer.

Implements the exact HTTP contract app/services/backend_client.py already
speaks (reverse-engineered from the vendored LoRA_Easy_Training_Scripts
backend's own main.py/utils/validation.py - see NOTES.md in this folder),
so the existing Train tab works against this unchanged: just point
"Backend URL" at wherever this is running instead.

This does NOT shell out to separate training scripts the way the vendored
backend does. /validate keeps the parsed config in memory; /train runs
the matching trainer function (sdxl_trainer.run or anima_trainer.run) on
a background thread and tracks its progress for /is_training to report.

Endpoints:
  POST /validate       body: {"args", "dataset", "accelerate"} -> {"tags": {}}
  GET  /train           params: train_mode, sdxl, flux, anima, accelerate_*
  GET  /is_training     -> {"training": bool, "errored": bool}
  GET  /stop_training    params: force
  GET  /check_path       params: path -> {"exists": bool}
  GET  /status           (extra, not part of the original contract - step/
                           loss/epoch progress for a richer Monitor tab)
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, request

# A log line the console encoding can't represent (non-ASCII dataset paths
# or tags) must not raise UnicodeEncodeError out of the reader thread -
# that used to abandon the trainer with nobody draining its pipe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass

_lock = threading.Lock()
_validated: dict | None = None  # last successful /validate payload
_thread: threading.Thread | None = None
_state = {"training": False, "errored": False, "error_detail": None}
_progress = {"step": 0, "total_steps": None, "epoch": 0, "loss": None}
_stop_requested = threading.Event()
_worker: subprocess.Popen | None = None
_log: list[str] = []
_MAX_LOG = 300
_ROOT = Path(__file__).resolve().parent.parent


_QUIET_PATHS = ("/is_training", "/status", "/check_path", "/progress")


class _QuietPolling(logging.Filter):
    """Drop the access-log lines of the endpoints the app polls every few
    seconds - they buried the real training output in the backend log."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f'"GET {p}' in msg for p in _QUIET_PATHS)


def _configure_logging() -> None:
    wz = logging.getLogger("werkzeug")
    wz.addFilter(_QuietPolling())
    wz.propagate = False            # werkzeug has its own handler; propagating printed every line twice
    if not wz.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        h.addFilter(_QuietPolling())
        wz.addHandler(h)
    wz.setLevel(logging.INFO)


def create_app() -> Flask:
    _configure_logging()
    app = Flask(__name__)

    @app.before_request
    def _no_browsers():
        # Only Anima Studio's own server talks to this one, and Python's
        # `requests` sends neither header. A browser always sends at least one,
        # so a web page can never drive training through here.
        if request.headers.get("Origin") or request.headers.get("Sec-Fetch-Site"):
            return jsonify({"detail": "Browsers may not call the training backend directly."}), 403

    @app.post("/validate")
    def validate():
        global _validated
        body = request.get_json(force=True) or {}
        args = body.get("args")
        dataset = body.get("dataset")
        if not isinstance(args, dict) or not isinstance(dataset, dict):
            return jsonify({"detail": "args and dataset are required"}), 400

        errors = _validate_payload(args, dataset)
        if errors:
            return jsonify({"detail": errors}), 400

        with _lock:
            _validated = {"args": args, "dataset": dataset, "accelerate": body.get("accelerate") or {}}
        return jsonify({"tags": {}})

    @app.get("/train")
    def train():
        global _thread
        with _lock:
            if _state["training"]:
                return jsonify({"detail": "Training Already Running"}), 409
            if _validated is None:
                return jsonify({"detail": "No Previously Validated Args"}), 400
            validated = _validated

        is_sdxl = request.args.get("sdxl", "False") == "True"
        is_flux = request.args.get("flux", "False") == "True"
        is_anima = request.args.get("anima", "False") == "True"
        is_moss = request.args.get("moss", "False") == "True"
        is_kohaku = request.args.get("kohaku", "False") == "True"
        train_mode = request.args.get("train_mode", "lora")

        runner = _pick_runner(train_mode, is_sdxl, is_flux, is_anima, is_moss, is_kohaku)
        if runner is None:
            return jsonify({
                "detail": "Invalid Train Parameters",
                "sdxl": is_sdxl, "flux": is_flux, "anima": is_anima, "moss": is_moss, "kohaku": is_kohaku,
                "train_mode": train_mode,
            }), 400

        with _lock:
            # re-check: a second /train can have passed the check above while
            # this request was validating and picking its runner - only one
            # may start, and the winner owns the stop-request flag
            if _state["training"]:
                return jsonify({"detail": "Training Already Running"}), 409
            _stop_requested.clear()
            _state["training"] = True
            _state["errored"] = False
            _state["error_detail"] = None
            _log.clear()
            _progress.update(step=0, total_steps=None, epoch=0, loss=None)
        _thread = threading.Thread(target=_run_training, args=(runner, validated), daemon=True)
        _thread.start()
        return jsonify({"detail": "Training Started", "training": True})

    @app.get("/is_training")
    def is_training():
        with _lock:
            return jsonify({"training": _state["training"], "errored": _state["errored"]})

    @app.get("/status")
    def status():
        with _lock:
            return jsonify({**_state, **_progress, "log_tail": _log[-60:]})

    @app.get("/stop_training")
    def stop_training():
        with _lock:
            if not _state["training"]:
                return jsonify({"detail": "Not Currently Training"}), 400
        _stop_requested.set()
        with _lock:
            proc = _worker
        if proc is not None:
            _kill_tree(proc)        # immediate - no waiting for the trainer to notice
        return jsonify({"detail": "Training Stopped"})

    @app.get("/check_path")
    def check_path():
        path = request.args.get("path", "")
        return jsonify({"exists": bool(path) and Path(path).exists()})

    return app


def _validate_payload(args: dict, dataset: dict) -> list[str]:
    errors = []
    subsets = dataset.get("subsets") or []
    if not subsets:
        errors.append("dataset.subsets is empty - add at least one image folder.")
    for i, s in enumerate(subsets):
        image_dir = s.get("image_dir")
        if not image_dir:
            errors.append(f"subset {i} has no image_dir.")
        elif not Path(image_dir).is_dir():
            errors.append(f"subset {i}: '{image_dir}' is not a folder.")
    saving = args.get("saving_args", {})
    if not saving.get("output_dir"):
        errors.append("saving_args.output_dir is required.")
    return errors


def _pick_runner(train_mode: str, is_sdxl: bool, is_flux: bool, is_anima: bool, is_moss: bool = False,
                 is_kohaku: bool = False):
    """Which trainer to run: 'sdxl' / 'anima' / 'moss' / 'kohaku', or None for unsupported combinations."""
    if is_kohaku:
        # KohakuMoss trains LoRA-family adapters and full finetunes itself (its own algorithm setting)
        return "kohaku" if not (is_sdxl or is_flux or is_anima or is_moss) else None
    if train_mode != "lora":
        return None  # this backend only implements LoRA training for now
    if is_moss:
        return "moss" if not (is_sdxl or is_flux or is_anima) else None
    if is_sdxl and not is_flux and not is_anima:
        return "sdxl"
    if is_anima and not is_sdxl and not is_flux:
        return "anima"
    return None  # flux, or an inconsistent combination - not implemented here


_PROG_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
_LOSS_RE = re.compile(r"(?:avr_loss|loss)=([0-9.eE+-]+)")
_EPOCH_RE = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)", re.I)


def _handle_line(line: str) -> None:
    line = line.rstrip("\r\n ")
    if not line:
        return
    if line.startswith("@@PROGRESS "):
        try:
            with _lock:
                _progress.update(json.loads(line[len("@@PROGRESS "):]))
        except ValueError:
            pass
        return
    if line.startswith("@@ERROR "):
        with _lock:
            _state["error_detail"] = line[len("@@ERROR "):]
        return
    m = _PROG_RE.search(line)
    if m and ("it/s" in line or "s/it" in line or "steps" in line.lower()):
        upd = {"step": int(m.group(1)), "total_steps": int(m.group(2))}
        lm = _LOSS_RE.search(line)
        if lm:
            try:
                upd["loss"] = float(lm.group(1))
            except ValueError:
                pass
        with _lock:
            _progress.update(upd)
    em = _EPOCH_RE.search(line)
    if em:
        with _lock:
            _progress.update(epoch=int(em.group(1)), max_epoch=int(em.group(2)))
    # keep the log compact: tqdm redraws the same line hundreds of times
    with _lock:
        if m and _log and _PROG_RE.search(_log[-1]):
            _log[-1] = line
        else:
            _log.append(line)
            if len(_log) > _MAX_LOG:
                del _log[: len(_log) - _MAX_LOG]
    print(line, flush=True)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the worker and everything it spawned (accelerate, dataloader workers)."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_training(kind: str, validated: dict) -> None:
    global _worker
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump({"args": validated["args"], "dataset": validated["dataset"]}, tmp)
        tmp.close()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONUTF8": "1"}
        python = sys.executable
        if kind == "moss":
            # MOSS-Image has its own environment (torch 2.14 / CUDA 13); the Anima/SDXL venv cannot run it.
            from . import moss_env
            if not moss_env.has_venv():
                raise RuntimeError("The MOSS-Image environment is not installed - open the Backend tab and press "
                                   "'Set up MOSS-Image environment' first.")
            python = str(moss_env.venv_python())
        elif kind == "kohaku":
            # KohakuMoss has its own environment too (torch 2.13-2.14, transformers 5.3+, kogine).
            from . import kohaku_env
            if not kohaku_env.has_venv():
                raise RuntimeError("The KohakuMoss environment is not installed - open the Backend tab and press "
                                   "'Set up KohakuMoss environment' first.")
            python = str(kohaku_env.venv_python())
        proc = subprocess.Popen(
            [python, "-u", "-m", "native_backend.train_worker", tmp.name, kind],
            cwd=str(_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            **({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
               if sys.platform == "win32" else {"start_new_session": True}),
        )
        with _lock:
            _worker = proc
        if _stop_requested.is_set():
            # /stop_training arrived while this spawn was still in flight and
            # had no process to kill yet - honour it now
            _kill_tree(proc)
        # tqdm separates updates with \r, not \n - split on both
        buf = b""
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]+", buf)
            buf = parts.pop()
            for part in parts:
                _handle_line(part.decode("utf-8", "replace"))
        if buf:
            _handle_line(buf.decode("utf-8", "replace"))
        code = proc.wait()
        if code != 0 and not _stop_requested.is_set():
            with _lock:
                _state["errored"] = True
                if not _state.get("error_detail"):
                    tail = [l for l in _log if l.strip()][-1:] or [f"trainer exited with code {code}"]
                    _state["error_detail"] = tail[0]
    except Exception as exc:  # noqa: BLE001 - reported via /is_training and /status
        traceback.print_exc()
        with _lock:
            proc = _worker
        if proc is not None and proc.poll() is None:
            # if the reader loop itself failed, kill the trainer rather than
            # leave it running untracked with nobody draining its pipe
            _kill_tree(proc)
        with _lock:
            _state["errored"] = True
            _state["error_detail"] = str(exc)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        with _lock:
            _worker = None
            _state["training"] = False


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    create_app().run(host="127.0.0.1", port=port, threaded=True)
