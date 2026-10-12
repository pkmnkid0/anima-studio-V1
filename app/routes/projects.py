"""Project management routes (§1, §21)."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import projects

bp = Blueprint("projects", __name__)


@bp.get("")
def list_projects():
    return jsonify({"projects": projects.list_projects(request.args.get("q", "")),
                    "defaults": projects.get_defaults()})


@bp.post("")
def create_project():
    body = request.get_json(force=True) or {}
    try:
        project = projects.create_project(
            name=str(body.get("name", "")),
            project_type=body.get("project_type") or None,
            model_type=body.get("model_type") or None,
            dataset_folder=str(body.get("dataset_folder") or ""),
        )
    except projects.ProjectError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(project)


@bp.post("/<project_id>/open")
def open_project(project_id: str):
    try:
        return jsonify(projects.touch_project(project_id))
    except projects.ProjectError as exc:
        return jsonify({"error": str(exc)}), 404


@bp.get("/<project_id>")
def get_project(project_id: str):
    try:
        return jsonify(projects.load_project(project_id))
    except projects.ProjectError as exc:
        return jsonify({"error": str(exc)}), 404


@bp.put("/<project_id>")
def update_project(project_id: str):
    body = request.get_json(force=True) or {}
    try:
        return jsonify(projects.update_project(project_id, **body))
    except projects.ProjectError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.delete("/<project_id>")
def delete_project(project_id: str):
    try:
        return jsonify(projects.delete_project(project_id))
    except projects.ProjectError as exc:
        return jsonify({"error": str(exc)}), 404


@bp.get("/types")
def project_types():
    return jsonify({
        "project_types": projects.PROJECT_TYPES,
        "model_types": projects.MODEL_TYPES,
        "layout": projects.LAYOUT_DIRS,
    })
