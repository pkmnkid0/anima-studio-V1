"""Runs one Low-VRAM trainer training from the Train tab (trainer kind "kohaku").

Anima Studio writes the run's config files (kohaku_config.py), then runs its own scripts with its runner,
`kogine`, from the Kohaku environment: inspect the data plan, build the offline text and latent caches, train. This
module only orchestrates; it needs nothing but the standard library and is started by native_backend/train_worker.py
with the Kohaku environment's interpreter (so a Stop that kills the worker kills every child with it).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import kohaku_config as kc
from . import kohaku_env as env

_STEP_LINE = re.compile(r"\bstep\s+(\d+)\b[^\n]*?\bloss=([-+0-9.eE]+|nan|inf)", re.I)


class KohakuError(RuntimeError):
    pass


def _emit(text: str) -> None:
    print(text, flush=True)


def _child_env() -> dict:
    e = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    e.pop("FORCE_COLOR", None)                       # no ANSI colours / tqdm redraws in the Monitor log
    return e


def _stream(cmd: list[str], on_line) -> int:
    """Run `cmd` in kohaku_backend/, passing every output line (split on \\r and \\n) to `on_line`."""
    proc = subprocess.Popen(cmd, cwd=str(env.KOHAKU_DIR), env=_child_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=0)
    buf = b""
    try:
        while True:
            chunk = proc.stdout.read(4096)                                           # type: ignore[union-attr]
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]+", buf)
            buf = parts.pop()
            for part in parts:
                if part.strip():
                    on_line(part.decode("utf-8", "replace"))
        if buf.strip():
            on_line(buf.decode("utf-8", "replace"))
    finally:
        code = proc.wait()
    return code


class _Metrics(threading.Thread):
    """Follows <run>/metrics.jsonl (one JSON row per logged step) and reports step / loss / s per it."""

    def __init__(self, path: Path, total: int, on_progress):
        super().__init__(daemon=True, name="kohaku-metrics")
        self.path, self.total, self.on_progress = path, total, on_progress
        self.since = time.time() - 5                  # ignore rows of an earlier attempt that is still in the file
        self._halt = threading.Event()
        self._last = -1

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.wait(2.0):
            row = self._last_row()
            if not row or float(row.get("time", 0) or 0) < self.since:
                continue
            step = row.get("step")
            if isinstance(step, int) and step != self._last:
                self._last = step
                upd = {"step": step, "total_steps": self.total}
                for src, dst in (("loss", "loss"), ("lr", "lr"), ("step_time", "s_per_it")):
                    if isinstance(row.get(src), (int, float)):
                        upd[dst] = row[src]
                self.on_progress(**upd)

    def _last_row(self) -> dict | None:
        try:
            with open(self.path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                tail = f.read().splitlines()
        except OSError:
            return None
        for line in reversed(tail):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                return row
        return None


def collect_exports(built: kc.Built) -> list[Path]:
    """Copy the finished adapter(s) next to the output folder's other LoRAs, where the Monitor tab and the
    packager look for them: <output>/<name>.safetensors (ComfyUI keys) and <output>/<name>-lycoris.safetensors."""
    out, src_dir = [], built.run_dir / "export"
    for suffix, dest in (("-comfy.safetensors", f"{built.name}.safetensors"),
                         ("-lycoris.safetensors", f"{built.name}-lycoris.safetensors")):
        src = src_dir / f"{built.name}{suffix}"
        if src.is_file():
            target = built.run_dir.parent / dest
            shutil.copy2(src, target)
            out.append(target)
    return out


def run(args: dict, dataset: dict, on_progress, should_stop) -> None:
    if not env.has_trainer():
        raise KohakuError("kohaku_backend/ is missing from this copy of Anima Studio - redownload it.")
    if not env.has_kogine():
        raise KohakuError("The Low-VRAM trainer environment is not installed (or is incomplete) - open the Backend tab and "
                          "press 'Set up Low-VRAM trainer environment'.")
    k = args.get("kohaku_args") or {}
    saving = args.get("saving_args") or {}
    try:
        built = kc.build(k, dataset.get("subsets") or [], saving.get("output_dir", ""), saving.get("output_name", "") or "",
                         user_dir=env.USER_CONFIGS)
    except kc.KohakuConfigError as exc:
        raise KohakuError(str(exc)) from None
    kc.write(built)
    _emit(f"[kohaku] run '{built.name}' - config written to {built.train_path}")
    for note in built.notes:
        _emit(f"[kohaku] note: {note}")
    ignored = sorted({key for s in dataset.get("subsets") or [] for key in
                      ("shuffle_caption", "keep_tokens", "flip_aug", "color_aug", "random_crop", "caption_dropout_rate")
                      if s.get(key)})
    if ignored:
        _emit("[kohaku] note: per-folder options " + ", ".join(ignored) + " are not used by Low-VRAM trainer - use its "
              "caption and augmentation settings on the Model tab.")

    on_progress(step=0, total_steps=built.steps, loss=None)
    kogine = str(env.kogine_exe())
    for label, script, cfg_name in kc.passes(built):
        cfg_file = built.train_path.parent / cfg_name
        try:                                          # the documented form: a path relative to kohaku_backend/
            cfg = cfg_file.resolve().relative_to(env.KOHAKU_DIR.resolve()).as_posix()
        except ValueError:
            cfg = cfg_file.resolve().as_posix()
        _emit(f"[kohaku] === {label} ===")
        monitor = None
        if script.endswith("train/moss.py"):
            monitor = _Metrics(built.run_dir / "metrics.jsonl", built.steps, on_progress)
            monitor.start()

        def on_line(line: str) -> None:
            m = _STEP_LINE.search(line)
            if m and monitor is not None:
                try:
                    on_progress(step=int(m.group(1)), total_steps=built.steps, loss=float(m.group(2)))
                except ValueError:
                    pass
            _emit(line)

        t0 = time.time()
        try:
            code = _stream([kogine, "run", script, "--config", cfg], on_line)
        finally:
            if monitor is not None:
                monitor.stop()
        if code != 0:
            raise KohakuError(f"{label} failed (exit code {code}) - see the log above for the first error.")
        _emit(f"[kohaku] {label}: done in {time.time() - t0:.0f} s")

    for path in collect_exports(built):
        _emit(f"[kohaku] LoRA saved: {path}")
    on_progress(step=built.steps, total_steps=built.steps, loss=None)
    _emit(f"[kohaku] finished. Checkpoints, samples and metrics are in {built.run_dir}")


if __name__ == "__main__":                                      # pragma: no cover
    print("Use native_backend.train_worker", file=sys.stderr)
