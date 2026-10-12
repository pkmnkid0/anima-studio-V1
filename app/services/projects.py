"""Project management (one LoRA = one project).

A project is a small folder under DATA_DIR/projects/ with a project.json
manifest and just two sub-folders:

    <project>/
        project.json   manifest: name, type, model, trigger word, remembered settings
        dataset/       training images (+ captions beside them)
        output/        everything that comes out: checkpoints, samples, package

A project may instead *link* an existing dataset folder anywhere on disk
(`dataset_path`), in which case no dataset/ folder is created.

The manifest also remembers the settings used with the project (its
`settings` dict, incl. a snapshot of the app's non-training UI settings), and
`last_opened`, so older projects can be reopened exactly as they were left.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .. import config
from . import settings_store

PROJECT_TYPES = ("character", "outfit", "style", "concept", "object", "other")
MODEL_TYPES = ("anima", "sdxl")

LAYOUT_DIRS = ("dataset", "output")
LEGACY_OUTPUT_DIRS = ("samples", "exports", "checkpoints")
MANIFEST_NAME = "project.json"


class ProjectError(RuntimeError):
    pass


def projects_root() -> Path:
    root = config.DATA_DIR / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "project"


def _manifest_path(project_dir: Path) -> Path:
    return project_dir / MANIFEST_NAME


def _read_manifest(project_dir: Path) -> dict:
    p = _manifest_path(project_dir)
    if not p.is_file():
        raise ProjectError(f"Not a project (no {MANIFEST_NAME}): {project_dir}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ProjectError(f"Couldn't read project manifest: {exc}")
    data["id"] = project_dir.name
    data["path"] = str(project_dir)
    linked = str(data.get("dataset_path") or "").strip()
    data["dataset_dir"] = linked or str(project_dir / "dataset")
    out = project_dir / "output"
    if not out.is_dir():
        # projects made by older versions kept samples/exports separately
        for legacy in LEGACY_OUTPUT_DIRS:
            if (project_dir / legacy).is_dir():
                out = project_dir / legacy
                break
    data["output_dir"] = str(out)
    data.setdefault("settings", {})
    return data


def _count_images(folder: Path) -> int:
    try:
        return sum(1 for f in folder.iterdir()
                   if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS)
    except OSError:
        return 0


def list_projects(q: str = "") -> list[dict]:
    """Projects, most recently opened first. `q` searches name, type, model,
    trigger word, folder path and (legacy) notes - every word must match."""
    words = [w for w in re.split(r"\s+", (q or "").lower().strip()) if w]
    projects = []
    for d in projects_root().iterdir():
        if not d.is_dir() or not _manifest_path(d).is_file():
            continue
        try:
            data = _read_manifest(d)
        except ProjectError:
            continue
        if words:
            hay = " ".join(str(data.get(k, "")) for k in
                           ("name", "project_type", "model_type", "trigger_word", "notes", "path", "dataset_dir")).lower()
            if not all(w in hay for w in words):
                continue
        data["image_count"] = _count_images(Path(data["dataset_dir"]))
        projects.append(data)
    projects.sort(key=lambda p: p.get("last_opened") or p.get("modified") or p.get("created") or 0, reverse=True)
    return projects


def get_defaults() -> dict:
    """Type/model last used when creating a project - prefilled next time."""
    d = settings_store.get("project_defaults") or {}
    return {"project_type": d.get("project_type", "character"), "model_type": d.get("model_type", "anima")}


def create_project(name: str, project_type: str | None = None,
                   model_type: str | None = None, dataset_folder: str = "") -> dict:
    name = name.strip()
    if not name:
        raise ProjectError("Project name is required.")
    defaults = get_defaults()
    project_type = project_type or defaults["project_type"]
    model_type = model_type or defaults["model_type"]
    if project_type not in PROJECT_TYPES:
        raise ProjectError(f"Unknown project type: {project_type}")
    if model_type not in MODEL_TYPES:
        raise ProjectError(f"Unknown model type: {model_type}")

    linked = (dataset_folder or "").strip().strip('"').strip("'")
    if linked and not Path(linked).is_dir():
        raise ProjectError(f"Dataset folder not found: {linked}")

    base = _slug(name)
    project_dir = projects_root() / base
    n = 2
    while project_dir.exists():
        project_dir = projects_root() / f"{base}_{n}"
        n += 1
    project_dir.mkdir(parents=True)
    for sub in LAYOUT_DIRS:
        if sub == "dataset" and linked:
            continue
        (project_dir / sub).mkdir()

    now = time.time()
    manifest = {
        "name": name,
        "project_type": project_type,
        "model_type": model_type,
        "trigger_word": "",
        "dataset_path": linked,
        "settings": {},
        "created": now,
        "modified": now,
        "last_opened": now,
    }
    _manifest_path(project_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    settings_store.set("project_defaults", {"project_type": project_type, "model_type": model_type})
    return _read_manifest(project_dir)


def touch_project(project_id: str) -> dict:
    """Mark a project as just opened (drives the 'recent' ordering)."""
    return _write(project_id, last_opened=time.time(), _keep_modified=True)


def load_project(project_id: str) -> dict:
    return _read_manifest(_project_dir(project_id))


def _write(project_id: str, _keep_modified: bool = False, **fields) -> dict:
    project_dir = _project_dir(project_id)
    data = _read_manifest(project_dir)
    for key in ("id", "path", "dataset_dir", "output_dir"):
        data.pop(key, None)
    allowed = {"name", "project_type", "model_type", "trigger_word", "notes",
               "dataset_path", "settings", "last_opened"}
    for key, value in fields.items():
        if key not in allowed:
            raise ProjectError(f"Unknown field: {key}")
        if key == "name" and not str(value).strip():
            raise ProjectError("Project name is required.")
        if key == "settings":
            if not isinstance(value, dict):
                raise ProjectError("settings must be an object.")
            merged = dict(data.get("settings") or {})
            merged.update(value)
            value = merged
        if key == "dataset_path" and value:
            value = str(value).strip().strip('"').strip("'")
            if not Path(value).is_dir():
                raise ProjectError(f"Dataset folder not found: {value}")
        data[key] = value
    if not _keep_modified:
        data["modified"] = time.time()
    _manifest_path(project_dir).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return _read_manifest(project_dir)


def update_project(project_id: str, **fields) -> dict:
    return _write(project_id, **fields)


def delete_project(project_id: str) -> dict:
    project_dir = _project_dir(project_id)
    name = _read_manifest(project_dir).get("name", project_id)
    import shutil
    shutil.rmtree(project_dir)
    return {"deleted": project_id, "name": name}


def _project_dir(project_id: str) -> Path:
    if not project_id or "/" in project_id or "\\" in project_id or project_id in (".", ".."):
        raise ProjectError("Invalid project id.")
    project_dir = projects_root() / project_id
    if not project_dir.is_dir():
        raise ProjectError(f"No such project: {project_id}")
    return project_dir
