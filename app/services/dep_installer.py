"""
Generic in-app "pip install X Y Z" runner with streamed output, for the
optional heavy dependencies a couple of features need beyond Anima
Studio's own requirements.txt - currently: torch/timm/safetensors for
the EVA02 canary tagger model, and torch/spandrel for the UltraSharp AI
upscaler. Both are plain, non-interactive `pip install` calls (unlike
the training backend's own installer in backend_setup.py, which asks
real interactive questions) - this just runs pip and streams what it
prints, so installing them doesn't require opening a terminal.

Multiple independent installs can run at once (keyed by job_id) since
they're unrelated packages for unrelated features - starting one
doesn't block or affect the other.
"""
from __future__ import annotations

import subprocess
import sys
import threading

_MAX_LOG_LINES = 400
_lock = threading.Lock()
_jobs: dict[str, dict] = {}


class InstallError(RuntimeError):
    pass


def status(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return {"stage": "idle", "log_tail": [], "packages": [], "error": None}
        return {
            "stage": job["stage"],
            "log_tail": list(job["log"][-150:]),
            "packages": job["packages"],
            "error": job.get("error"),
        }


def is_busy(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
        return bool(job and job["stage"] == "installing")


def start(job_id: str, packages: list[str], extra_pip_args: list[str] | None = None,
          python: str | None = None) -> dict:
    """Run `pip install <packages>` in the background.

    `python` picks the interpreter to install into: the app's own by default,
    or - for features that borrow the training backend's environment - the
    backend venv's interpreter.
    """
    if not packages:
        raise InstallError("No packages given.")
    if is_busy(job_id):
        raise InstallError("Already installing - check the log below, or wait for it to finish.")

    with _lock:
        _jobs[job_id] = {"stage": "installing", "log": [], "packages": list(packages), "error": None}

    threading.Thread(target=_run, args=(job_id, packages, extra_pip_args or [], python),
                     daemon=True).start()
    return status(job_id)


def _append(job_id: str, line: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["log"].append(line.rstrip("\n"))
        if len(job["log"]) > _MAX_LOG_LINES:
            del job["log"][: len(job["log"]) - _MAX_LOG_LINES]


def _set_stage(job_id: str, stage: str, error: str | None = None) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job["stage"] = stage
            job["error"] = error


def _run(job_id: str, packages: list[str], extra_pip_args: list[str], python: str | None = None) -> None:
    try:
        if python is None and getattr(sys, "frozen", False):
            raise InstallError(
                "Anima Studio is running as a packaged app, which has no Python for "
                "pip to install into - install these packages into the backend's "
                "Python manually, then restart Anima Studio.")
        args = [python or sys.executable, "-m", "pip", "install", *extra_pip_args, *packages]
        _append(job_id, f"Running: {' '.join(args)}")
        _append(job_id, "")
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        from .procs import register
        register(proc)                  # app shutdown must not orphan the pip process
        for line in proc.stdout:
            _append(job_id, line)
        code = proc.wait()
        if code == 0:
            _append(job_id, "")
            _append(job_id, "Done - try the feature again, it should work now.")
            _set_stage(job_id, "done")
        else:
            _set_stage(job_id, "error", f"pip exited with code {code} - scroll up in the log to see why.")
    except Exception as exc:  # noqa: BLE001 - surfaced to the user via status()
        _append(job_id, f"Unexpected error: {exc}")
        _set_stage(job_id, "error", str(exc))
