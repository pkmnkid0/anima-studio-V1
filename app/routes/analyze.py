"""Dataset analysis routes (§2, §14, §15)."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import analysis

bp = Blueprint("analyze", __name__)


@bp.post("/dataset")
def analyze_dataset():
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    deep = bool(body.get("deep", True))
    try:
        result = analysis.analyze_dataset(folder, deep=deep)
        result["health"] = analysis.health_score(result)
        result["coverage"] = analysis.coverage_map(folder)
        return jsonify(result)
    except analysis.AnalysisError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/balance")
def balance():
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    try:
        return jsonify(analysis.balance_analysis(folder))
    except analysis.AnalysisError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/cleanup/plan")
def cleanup_plan():
    """Re-analyze and return the KEEP/REMOVE/REVIEW plan.
    Nothing is deleted by this endpoint."""
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    try:
        result = analysis.analyze_dataset(folder, deep=True)
        return jsonify({
            "plan": analysis.cleanup_plan(result),
            "health": analysis.health_score(result),
        })
    except analysis.AnalysisError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/cleanup/apply")
def cleanup_apply():
    """Apply the user's explicit decisions. Only entries the
    user marked "remove" are deleted."""
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    decisions = body.get("decisions") or {}
    if not isinstance(decisions, dict):
        return jsonify({"error": "decisions must be an object of path -> action"}), 400
    try:
        return jsonify(analysis.apply_cleanup(folder, decisions))
    except (analysis.AnalysisError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


