from __future__ import annotations

from flask import Blueprint, jsonify, request

from .. import config
from ..services import backend_client, gpu, monitor

bp = Blueprint("monitor", __name__)


@bp.get("/samples")
def samples():
    output_dir = request.args.get("output_dir", "")
    if not output_dir:
        return jsonify({"error": "output_dir is required"}), 400
    return jsonify({
        "samples": monitor.list_sample_images(output_dir),
        "output_dir_exists": monitor.output_dir_exists(output_dir),
    })


@bp.get("/checkpoints")
def checkpoints():
    output_dir = request.args.get("output_dir", "")
    if not output_dir:
        return jsonify({"error": "output_dir is required"}), 400
    return jsonify({
        "checkpoints": monitor.list_checkpoints(output_dir),
        "output_dir_exists": monitor.output_dir_exists(output_dir),
    })


@bp.get("/scalars")
def scalars():
    logging_dir = request.args.get("logging_dir", "")
    if not logging_dir:
        return jsonify({"available": False, "reason": "No logging directory configured for this run."})
    return jsonify(monitor.read_scalars(logging_dir))


@bp.get("/gpu")
def gpu_health():
    """Live GPU state for the Train tab's health strip (nvidia-smi, cached briefly)."""
    return jsonify(gpu.query(force=request.args.get("force") == "1"))


@bp.get("/progress")
def progress():
    url = request.args.get("url", config.DEFAULT_BACKEND_URL)
    try:
        data = backend_client.get_progress(url)
    except backend_client.BackendError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify(data or {"available": False})


@bp.get("/status")
def status():
    url = request.args.get("url", config.DEFAULT_BACKEND_URL)
    try:
        return jsonify(backend_client.is_training(url))
    except backend_client.BackendError as exc:
        return jsonify({"error": str(exc)}), 502
