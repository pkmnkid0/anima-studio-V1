"""
Training monitoring: helps answer "is this LoRA training well?" without
fabricating anything Anima Studio can't actually observe.

Three things are read straight from the filesystem and always work:
  - the latest sample images the backend saved (if sample generation
    was enabled in the Train tab), so you can watch quality/overfitting
    visually as training progresses
  - the checkpoints saved so far, so you can see what's actually landed
    on disk
  - basic training status, via the same backend HTTP API the Train tab
    already polls

Loss/learning-rate curves are read from TensorBoard event files if
logging was enabled - genuinely useful, but optional and clearly
labeled as such, since it needs the separate `tensorboard` package and
a run that actually wrote logs.
"""
from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}


def output_dir_exists(output_dir: str) -> bool:
    """Whether the configured output folder is actually on disk.

    The listing helpers below return [] for two very different situations:
    "training hasn't written anything yet" and "this folder does not
    exist". The UI has to tell them apart, because the second one means
    the path is wrong (renamed, deleted, or pointing at a project that no
    longer exists) and polling it forever produces a permanently blank
    panel with no explanation.
    """
    return Path(output_dir).is_dir()


def list_sample_images(output_dir: str, limit: int = 24) -> list[dict]:
    """Newest sample images under <output_dir>/sample - the conventional
    location sd-scripts-family trainers write sample generations to."""
    sample_dir = Path(output_dir) / "sample"
    if not sample_dir.is_dir():
        return []

    files = [f for f in sample_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return [
        {"path": str(f), "filename": f.name, "mtime": f.stat().st_mtime}
        for f in files[:limit]
    ]


def list_checkpoints(output_dir: str) -> list[dict]:
    """Saved model files directly in the output folder, newest first."""
    out = Path(output_dir)
    if not out.is_dir():
        return []

    files = [f for f in out.iterdir() if f.is_file() and f.suffix.lower() in CHECKPOINT_EXTENSIONS]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return [
        {"path": str(f), "filename": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime}
        for f in files
    ]


def tensorboard_available() -> bool:
    try:
        import tensorboard.backend.event_processing.event_accumulator  # noqa: F401
        return True
    except ImportError:
        return False


def read_scalars(logging_dir: str) -> dict:
    """Scalar summaries (loss, learning rate, etc.) from TensorBoard
    event files under `logging_dir`. Returns {"available": False,
    "reason": "..."} if tensorboard isn't installed or nothing was
    logged yet - never raises for those expected cases."""
    if not tensorboard_available():
        return {
            "available": False,
            "reason": "The 'tensorboard' package isn't installed. Run: pip install tensorboard",
        }

    log_path = Path(logging_dir)
    if not log_path.is_dir():
        return {"available": False, "reason": f"No logging directory found at {logging_dir}."}

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # TensorBoard logs to a subfolder per run; search recursively for
    # the actual event files rather than assuming a fixed layout.
    event_dirs = {p.parent for p in log_path.rglob("events.out.tfevents.*")}
    if not event_dirs:
        return {"available": False, "reason": "No TensorBoard event files found yet - they appear once training starts writing logs."}

    # Every training run writes its own folder (and event file). Merging them
    # drew old and new runs on top of each other, so a new run looked frozen.
    # Show only the newest run.
    def _newest(d: Path) -> float:
        try:
            return max(f.stat().st_mtime for f in d.glob("events.out.tfevents.*"))
        except (OSError, ValueError):
            return 0.0

    latest = max(event_dirs, key=_newest)
    run_id = f"{latest}|{max(f.name for f in latest.glob('events.out.tfevents.*'))}"
    event_dirs = {latest}

    series: dict[str, list[dict]] = {}
    for event_dir in event_dirs:
        try:
            acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
            acc.Reload()
        except Exception as exc:  # noqa: BLE001 - a corrupt/partial event file shouldn't break the whole view
            continue
        for tag in acc.Tags().get("scalars", []):
            points = [{"step": e.step, "value": e.value, "wall_time": e.wall_time} for e in acc.Scalars(tag)]
            series.setdefault(tag, []).extend(points)

    for tag in series:
        series[tag].sort(key=lambda p: p["step"])

    if not series:
        return {"available": False, "reason": "Event files exist but contain no scalar data yet."}

    return {"available": True, "series": series, "run": run_id}
