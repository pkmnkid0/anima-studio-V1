"""GPU telemetry for the Train tab's health strip.

Reads `nvidia-smi`, which ships with the NVIDIA driver. That means this works
without installing anything and without torch: the app's own interpreter never
has to import a GPU library, and it still works when the training backend has
not been set up yet.

Results are cached for a moment because the UI polls them. On a machine with no
NVIDIA GPU (or no driver) this reports `available: False` with a reason, and the
UI hides the strip instead of showing an error.
"""
from __future__ import annotations

import subprocess
import sys
import time

CACHE_SECONDS = 1.5
TIMEOUT_SECONDS = 3.0

_FIELDS = ["index", "name", "utilization.gpu", "memory.used", "memory.total",
           "temperature.gpu", "power.draw", "power.limit", "fan.speed"]
_NUMERIC_FIELDS = [f for f in _FIELDS if f != "name"]

_cache: dict = {"at": 0.0, "data": None}


def _run(args: list[str]) -> str:
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW    # never flash a console window
    out = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                         encoding="utf-8", errors="replace", **kwargs)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout or "nvidia-smi failed").strip())
    return out.stdout


def _num(text: str) -> float | None:
    text = (text or "").strip()
    if not text or text.lower() in ("n/a", "[not supported]", "[n/a]", "not supported", "unknown"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


def _parse(text: str) -> list[dict]:
    gpus = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(_FIELDS):
            index, name = parts[0], parts[1]
            rest = parts[2:]
        elif len(parts) == len(_NUMERIC_FIELDS):     # a name containing a comma shifted the split
            index, name, rest = parts[0], "", parts[1:]
        else:
            continue
        values = dict(zip(_NUMERIC_FIELDS[1:], rest))
        total = _num(values["memory.total"])
        used = _num(values["memory.used"])
        gpus.append({
            "index": int(_num(index) or 0),
            "name": name or "GPU",
            "util_pct": _round(_num(values["utilization.gpu"])),
            "mem_used_mb": _round(used),
            "mem_total_mb": _round(total),
            "mem_pct": _round(100 * used / total) if used is not None and total else None,
            "temp_c": _round(_num(values["temperature.gpu"])),
            "power_w": _round(_num(values["power.draw"])),
            "power_limit_w": _round(_num(values["power.limit"])),
            "fan_pct": _round(_num(values["fan.speed"])),
        })
    return gpus


def query(force: bool = False) -> dict:
    """Current GPU state. Cached for CACHE_SECONDS; never raises."""
    now = time.time()
    if not force and _cache["data"] is not None and now - _cache["at"] < CACHE_SECONDS:
        return _cache["data"]

    query_fields = ",".join(_FIELDS)
    try:
        text = _run(["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"])
        gpus = _parse(text)
        data = {"available": bool(gpus), "gpus": gpus, "at": now}
        if not gpus:
            data["reason"] = "nvidia-smi returned no GPUs."
    except FileNotFoundError:
        data = {"available": False, "gpus": [], "at": now,
                "reason": "nvidia-smi was not found - no NVIDIA driver on this machine."}
    except subprocess.TimeoutExpired:
        data = {"available": False, "gpus": [], "at": now, "reason": "nvidia-smi did not answer in time."}
    except (OSError, RuntimeError, ValueError) as exc:
        data = {"available": False, "gpus": [], "at": now, "reason": str(exc)}

    _cache.update(at=now, data=data)
    return data
