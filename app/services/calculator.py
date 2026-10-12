"""Images & steps calculator (§6).

Pure math - no I/O, no torch. Mirrors how the vendored
trainer derives optimizer steps:

    images_per_epoch = images * repeats
    steps_per_epoch  = ceil(images_per_epoch / (batch * grad_accum))
    total_steps      = steps_per_epoch * epochs

Recommended step ranges follow the usual LoRA community
heuristics and adapt to the project type and dataset size.
"""
from __future__ import annotations

import math

PROJECT_TYPES = ("character", "outfit", "style", "concept", "object", "other")

# Base "balanced" step target for a median dataset, per type.
# Character LoRAs converge on fewer steps than broad concepts.
_BASE_STEPS = {
    "character": 2500,
    "outfit": 2200,
    "style": 3000,
    "concept": 3500,
    "object": 2500,
    "other": 3000,
}

# How the target scales with dataset size (images bucket -> factor).
def _size_factor(images: int) -> float:
    if images <= 10:
        return 0.6
    if images <= 30:
        return 0.8
    if images <= 100:
        return 1.0
    if images <= 300:
        return 1.2
    if images <= 1000:
        return 1.4
    return 1.6


def calculate(images: int, repeats: int, epochs: int,
              batch_size: int, grad_accum: int,
              project_type: str = "character") -> dict:
    images = int(images)
    repeats = int(repeats)
    epochs = int(epochs)
    batch_size = int(batch_size)
    grad_accum = int(grad_accum)

    if images < 1:
        raise ValueError("Number of images must be at least 1.")
    if repeats < 1:
        raise ValueError("Repeats must be at least 1.")
    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if grad_accum < 1:
        raise ValueError("Gradient accumulation must be at least 1.")
    if project_type not in PROJECT_TYPES:
        raise ValueError(f"Unknown project type: {project_type}")

    effective_batch = batch_size * grad_accum
    images_per_epoch = images * repeats
    total_exposures = images_per_epoch * epochs
    steps_per_epoch = math.ceil(images_per_epoch / effective_batch)
    total_steps = steps_per_epoch * epochs

    # Recommended ranges, scaled by type + dataset size.
    balanced = round(_BASE_STEPS[project_type] * _size_factor(images))
    ranges = {
        "low": round(balanced * 0.6),
        "balanced": balanced,
        "high": round(balanced * 1.4),
        "very_high": round(balanced * 1.8),
    }

    warnings = []
    if images <= 20 and total_steps > ranges["high"]:
        warnings.append(
            "⚠️ Small dataset + high steps may cause overfitting. "
            "Consider fewer steps, lower rank, or more images.")
    if images >= 500 and total_steps < ranges["low"]:
        warnings.append(
            "⚠️ Large dataset + very low steps may produce a weak LoRA. "
            "Consider more steps or epochs.")
    if images <= 10:
        warnings.append(
            "⚠️ Very small datasets (≤10 images) rarely generalize - "
            "expect to rely on the trigger word heavily.")
    if epochs > 40:
        warnings.append("⚠️ Many epochs on a small dataset is the classic overfitting recipe.")
    if total_steps > 10000:
        warnings.append("⚠️ Very long training - consider capping steps and checking samples.")

    # Effective exposure: how many times, on average, each image
    # is seen (repeats * epochs), plus a diversity note.
    exposure = repeats * epochs

    # Steps needed to hit a target (inverse calculation), if the
    # caller wants "I want ~2500 steps, how many epochs?"
    def epochs_for(target_steps: int) -> float:
        return target_steps / steps_per_epoch if steps_per_epoch else 0.0

    return {
        "images": images,
        "repeats": repeats,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": effective_batch,
        "project_type": project_type,
        "images_per_epoch": images_per_epoch,
        "total_exposures": total_exposures,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "exposure_per_image": exposure,
        "recommended_ranges": ranges,
        "epochs_for_target": {
            str(t): round(epochs_for(t), 1)
            for t in (ranges["low"], ranges["balanced"], ranges["high"])
        },
        "warnings": warnings,
    }


def estimate_duration(total_steps: int, steps_per_sec: float = 0.8) -> dict:
    """Rough wall-clock estimate. steps_per_sec defaults to a
    conservative 0.8 (a 512-step/sec SDXL-ish run is fast;
    Anima DiT on a consumer card is often slower)."""
    if total_steps < 0:
        raise ValueError("total_steps must be >= 0")
    if steps_per_sec <= 0:
        raise ValueError("steps_per_sec must be > 0")
    seconds = total_steps / steps_per_sec
    return {
        "total_steps": total_steps,
        "steps_per_sec": steps_per_sec,
        "seconds": round(seconds),
        "minutes": round(seconds / 60, 1),
        "hours": round(seconds / 3600, 2),
        "human": _human_duration(seconds),
    }


def _human_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
