from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path

from flask import Blueprint, jsonify, request, Response

from .. import config, schema, train_config
from ..toml_writer import dumps as toml_dumps
from ..services import backend_client, backend_manager, backend_setup
from ..services import analysis, calculator, history, packager, recommend
from ..services import queue as train_queue

bp = Blueprint("train", __name__)


@bp.post("/parse_toml")
def parse_toml():
    """Load TOML: accepts either raw TOML text (JSON body {"text": "..."})
    or an uploaded file (multipart "file"), parses it, and returns the
    Train tab's UI JSON shape so the form can populate itself."""
    text = None
    if request.files.get("file"):
        text = request.files["file"].read().decode("utf-8", errors="replace")
    else:
        body = request.get_json(silent=True) or {}
        text = body.get("text")

    if not text:
        return jsonify({"error": "No TOML content received."}), 400

    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return jsonify({"error": f"That doesn't look like valid TOML: {exc}"}), 400

    try:
        canonical = train_config.from_toml_document(doc)
        ui = train_config.to_ui_payload(canonical)
    except Exception as exc:  # noqa: BLE001 - surface a clean message either way
        return jsonify({"error": f"Couldn't read that config: {exc}"}), 400

    return jsonify({"ui": ui})


@bp.post("/load_toml_from_disk")
def load_toml_from_disk():
    """Like /parse_toml, but reads the file straight off the server's
    filesystem (via the file browser's path) instead of a browser upload -
    for loading a .toml that already lives on disk, e.g. from a previous
    session or written by another tool."""
    body = request.get_json(force=True) or {}
    path = body.get("path", "")
    p = Path(path)
    if not p.is_file():
        return jsonify({"error": f"No such file: {path}"}), 400
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return jsonify({"error": f"Couldn't read {path}: {exc}"}), 400

    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return jsonify({"error": f"That doesn't look like valid TOML: {exc}"}), 400

    try:
        canonical = train_config.from_toml_document(doc)
        ui = train_config.to_ui_payload(canonical)
    except Exception as exc:  # noqa: BLE001 - surface a clean message either way
        return jsonify({"error": f"Couldn't read that config: {exc}"}), 400

    return jsonify({"ui": ui, "path": str(p)})


@bp.post("/save_toml_to_disk")
def save_toml_to_disk():
    """Writes the current config straight to a folder on the server's
    filesystem (chosen via the folder browser) instead of only offering
    it as a browser download."""
    body = request.get_json(force=True) or {}
    ui = body.get("ui", {})
    directory = body.get("dir", "")
    filename = (body.get("filename") or "").strip()
    if not directory:
        return jsonify({"error": "Choose a destination folder first."}), 400

    payload = train_config.build_backend_payload(ui)
    doc = train_config.to_toml_document(payload)
    text = toml_dumps(doc)

    if not filename:
        filename = (payload["args"].get("saving_args", {}).get("output_name") or "anima_lora_config") + ".toml"
    if not filename.lower().endswith(".toml"):
        filename += ".toml"

    d = Path(directory)
    try:
        d.mkdir(parents=True, exist_ok=True)
        target = d / filename
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        return jsonify({"error": f"Couldn't write to {directory}: {exc}"}), 400

    return jsonify({"ok": True, "path": str(target)})


@bp.get("/schema")
def get_schema():
    return jsonify(schema.full_schema())


_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _caption_warnings(subset: dict) -> list[str]:
    """Same rule as the trainer's own read_caption(): the caption is
    <image path without extension> + caption_extension, in the same folder
    (so 'a.png' needs 'a.txt'; 'a.png.txt' is NOT read)."""
    import os
    try:
        d = str(subset.get("image_dir"))
        if not os.path.isdir(d):
            return []
        ext = subset.get("caption_extension") or ".txt"
        if not ext.startswith("."):
            ext = "." + ext
        imgs = [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in _IMG_EXT]
        if not imgs:
            return []
        missing = []
        for f in imgs:
            cap = os.path.join(d, os.path.splitext(f)[0] + ext)
            ok = False
            if os.path.isfile(cap):
                with open(cap, "rt", encoding="utf-8", errors="ignore") as fh:
                    ok = bool(fh.read().strip())
            if not ok:
                missing.append((f, os.path.basename(os.path.splitext(f)[0] + ext)))
        if not missing:
            return []
        name = subset.get("name") or os.path.basename(d)
        ex = "; ".join(f"{i} needs {c}" for i, c in missing[:3])
        # Captions that exist but are named differently (e.g. 'pic.png_0001.txt' or 'pic.png.txt')
        stray = [t for t in os.listdir(d) if t.lower().endswith(ext.lower())
                 and any(t.startswith(i) for i, _ in missing)]
        if stray:
            ex += (f". Found caption files with the wrong name, e.g. '{stray[0]}' - the caption must be "
                   f"named exactly like the image but ending in {ext} (rename it)")
        return [f"Folder '{name}': {len(missing)} of {len(imgs)} images have no usable '{ext}' caption "
                f"(e.g. {ex}). Folder: {d}"]
    except Exception:  # never let a check break validation
        return []


def _validation_warnings(payload: dict) -> list[str]:
    warnings = []
    general = payload["args"].get("general_args", {})
    anima = payload["args"].get("anima_args", {})
    saving = payload["args"].get("saving_args", {})
    if general.get("sdxl"):
        # SDXL: the base model lives in general_args, and the
        # Anima-specific fields above are legitimately absent.
        if not general.get("pretrained_model_name_or_path"):
            warnings.append("No base model path set.")
    else:
        if not anima.get("pretrained_model_name_or_path"):
            warnings.append("No DiT model path set.")
        if not anima.get("qwen3"):
            warnings.append("No Qwen3 text encoder path set.")
        if not anima.get("vae"):
            warnings.append("No VAE path set.")
    if not saving.get("output_dir"):
        warnings.append("No output folder set.")
    sample = payload["args"].get("sample_args", {})
    if sample and not (sample.get("sample_prompts_text") or sample.get("sample_prompts")):
        warnings.append("Sample images are on but no sample prompt is written - add at least one prompt in Sample images.")
    if not payload["subsets"]:
        warnings.append("No dataset subsets added - training needs at least one image folder.")
    else:
        for s in payload["subsets"]:
            if not s.get("image_dir"):
                warnings.append(f"Subset '{s.get('name')}' has no image folder set.")
            else:
                warnings.extend(_caption_warnings(s))
    return warnings


@bp.post("/build")
def build():
    """Convert the UI payload into the backend structure + TOML text,
    without saving or training - used for live preview and for the
    "Download TOML" action."""
    ui = request.get_json(force=True) or {}
    payload = train_config.build_backend_payload(ui)
    doc = train_config.to_toml_document(payload)
    text = toml_dumps(doc)
    return jsonify({
        "payload": payload,
        "toml": text,
        "warnings": _validation_warnings(payload),
    })


@bp.post("/download_toml")
def download_toml():
    ui = request.get_json(force=True) or {}
    payload = train_config.build_backend_payload(ui)
    doc = train_config.to_toml_document(payload)
    text = toml_dumps(doc)
    filename = (payload["args"].get("saving_args", {}).get("output_name") or "anima_lora_config") + ".toml"
    return Response(
        text, mimetype="application/toml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --- Presets (native JSON - exact round trip of the UI's own shape) ----

@bp.get("/presets")
def list_presets():
    presets = []
    for f in sorted(config.PRESETS_DIR.glob("*.json")):
        presets.append({"name": f.stem, "modified": f.stat().st_mtime})
    return jsonify({"presets": presets})


@bp.get("/presets/<name>")
def load_preset(name: str):
    path = _preset_path(name)
    if not path.exists():
        return jsonify({"error": "Preset not found"}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@bp.post("/presets/<name>")
def save_preset(name: str):
    ui = request.get_json(force=True) or {}
    path = _preset_path(name)
    path.write_text(json.dumps(ui, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "name": name})


@bp.delete("/presets/<name>")
def delete_preset(name: str):
    path = _preset_path(name)
    if path.exists():
        path.unlink()
    return jsonify({"ok": True})


def _preset_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip() or "preset"
    return config.PRESETS_DIR / f"{safe}.json"


# --- Backend control -----------------------------------------------------

@bp.get("/backend/ping")
def backend_ping():
    url = request.args.get("url", config.DEFAULT_BACKEND_URL)
    return jsonify(backend_client.ping(url))



@bp.post("/backend/start")
def backend_start():
    body = request.get_json(force=True) or {}
    ui = body.get("ui", {})
    url = body.get("backend_url", config.DEFAULT_BACKEND_URL)
    payload = train_config.build_backend_payload(ui)
    dataset_with_subsets = {**payload["dataset"], "subsets": payload["subsets"]}

    try:
        backend_client.validate(url, payload["args"], dataset_with_subsets)
        backend_client.start_training(
            url,
            train_mode=payload["train_mode"],
            has_sdxl=bool(payload["args"].get("general_args", {}).get("sdxl")),
            has_flux=bool(payload["args"].get("flux_args")),
            has_anima=bool(payload["args"].get("anima_args")),
            accelerate=ui.get("accelerate", {}),
        )
    except backend_client.BackendError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify({"ok": True, "started_at": time.time()})


@bp.get("/backend/status")
def backend_status():
    url = request.args.get("url", config.DEFAULT_BACKEND_URL)
    try:
        return jsonify(backend_client.is_training(url))
    except backend_client.BackendError as exc:
        return jsonify({"error": str(exc)}), 502


@bp.post("/backend/stop")
def backend_stop():
    body = request.get_json(force=True) or {}
    url = body.get("backend_url", config.DEFAULT_BACKEND_URL)
    try:
        backend_client.stop_training(url)
    except backend_client.BackendError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"ok": True})


# --- Local backend process launcher ---------------------------------------
# Distinct from the /backend/* routes above (which talk *to* an already
# running backend over HTTP) - these start/stop that backend as a child
# process of Anima Studio itself, so the user doesn't have to open a
# second terminal for it.

@bp.get("/backend/process/settings")
def backend_process_settings():
    return jsonify(backend_manager.get_settings())


@bp.post("/backend/process/settings")
def backend_process_save_settings():
    body = request.get_json(force=True) or {}
    saved = backend_manager.save_settings(
        body.get("dir", ""), body.get("command", ""), bool(body.get("autostart", False)),
    )
    return jsonify(saved)


@bp.get("/backend/process/status")
def backend_process_status():
    return jsonify(backend_manager.status())


@bp.post("/backend/process/start")
def backend_process_start():
    body = request.get_json(force=True) or {}
    directory = body.get("dir", "")
    command = body.get("command", "")
    # Starting it is also a good time to remember these settings, so a
    # restart of Anima Studio doesn't lose them even without autostart on.
    current = backend_manager.get_settings()
    backend_manager.save_settings(directory, command, current.get("autostart", True), explicit=False)
    from ..services import backend_setup
    if backend_setup.is_installed() and not backend_setup.is_busy() and backend_setup.repair_runtime_deps():
        return jsonify({**backend_manager.status(), "repairing": True,
                        "note": "Some packages were missing - installing them first; the backend starts automatically afterwards."})
    try:
        result = backend_manager.start(directory, command)
    except backend_manager.BackendLaunchError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@bp.post("/backend/process/stop")
def backend_process_stop():
    return jsonify(backend_manager.stop())


# --- In-app backend installer -----------------------------------------
# Runs the same clone + install steps as setup_backend.sh/.bat, from a
# button instead of a separate terminal. The installer itself can ask
# interactive questions, so /setup/input lets the UI act as a tiny
# terminal for whatever it asks rather than guessing blind.

@bp.post("/backend/setup/start")
def backend_setup_start():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(backend_setup.start(body.get("target_dir", "")))
    except backend_setup.SetupError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/backend/setup/status")
def backend_setup_status():
    return jsonify(backend_setup.status())


@bp.post("/backend/setup/input")
def backend_setup_input():
    body = request.get_json(force=True) or {}
    text = body.get("text", "")
    try:
        return jsonify(backend_setup.send_input(text))
    except backend_setup.SetupError as exc:
        return jsonify({"error": str(exc)}), 400


# --- Training presets (§5) ------------------------------------------
# Distinct from the user-saved UI presets above: these are the
# curated per-(model type, project type) starting points.

@bp.get("/training_presets")
def training_presets():
    model_type = request.args.get("model_type")
    try:
        return jsonify({"presets": recommend.presets(model_type)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/training_presets/<model_type>/<name>")
def training_preset(model_type: str, name: str):
    try:
        return jsonify(recommend.preset(model_type, name))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


# --- Automatic configuration recommendations (§17) -------------------

# --- AI training advisor (§22) ---------------------------------------

@bp.post("/advisor")
def advisor():
    body = request.get_json(force=True) or {}
    folder = str(body.get("folder", ""))
    model_type = str(body.get("model_type", "anima"))
    project_type = str(body.get("project_type", "character"))
    try:
        analysis_result = analysis.analyze_dataset(folder, deep=True)
        health = analysis.health_score(analysis_result)
        vram = body.get("vram_gb")
        report = recommend.advisor_report(
            analysis_result, health, model_type, project_type,
            None if vram in (None, "") else float(vram))
        return jsonify({
            "report": report,
            "analysis": {
                "count": analysis_result["count"],
                "health": health,
                "resolutions": analysis_result["resolutions"],
            },
        })
    except (analysis.AnalysisError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


# --- Images & steps calculator (§6) ----------------------------------

@bp.post("/calculator")
def calculate_steps():
    body = request.get_json(force=True) or {}
    try:
        result = calculator.calculate(
            images=int(body.get("images", 0) or 0),
            repeats=int(body.get("repeats", 1) or 1),
            epochs=int(body.get("epochs", 1) or 1),
            batch_size=int(body.get("batch_size", 1) or 1),
            grad_accum=int(body.get("grad_accum", 1) or 1),
            project_type=str(body.get("project_type", "character")),
        )
        result["duration"] = calculator.estimate_duration(
            result["total_steps"],
            float(body.get("steps_per_sec") or 0.8))
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


# --- Hardware detection (§16) -----------------------------------------

# --- Training history (§18) -------------------------------------------

@bp.get("/history")
def training_history():
    return jsonify({"runs": history.list_runs(int(request.args.get("limit", 100) or 100))})


@bp.post("/history")
def record_history():
    body = request.get_json(force=True) or {}
    if not body.get("name"):
        return jsonify({"error": "Run name is required."}), 400
    try:
        return jsonify(history.record(body))
    except OSError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.delete("/history/<run_id>")
def delete_history(run_id: str):
    if not history.delete_run(run_id):
        return jsonify({"error": "Run not found."}), 404
    return jsonify({"ok": True})


@bp.post("/history/compare")
def compare_history():
    body = request.get_json(force=True) or {}
    return jsonify({"runs": history.compare(list(body.get("run_ids") or []))})


# --- LoRA packaging (§20) ----------------------------------------------

@bp.get("/package/inspect")
def package_inspect():
    try:
        return jsonify(packager.inspect(request.args.get("path", "")))
    except (packager.PackageError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/package")
def package_lora():
    body = request.get_json(force=True) or {}
    lora_path = packager.clean_path(str(body.get("lora_path", "")))
    output_dir = packager.clean_path(str(body.get("output_dir", "")))
    if not lora_path:
        return jsonify({"error": "LoRA file path is required."}), 400
    if not output_dir:
        return jsonify({"error": "Output folder is required."}), 400
    training_config = body.get("training_config") or {}
    if not training_config.get("project_type") and body.get("project_type"):
        training_config["project_type"] = body.get("project_type")
    if not training_config.get("model_type") and body.get("model_type"):
        training_config["model_type"] = body.get("model_type")
    try:
        return jsonify(packager.package(
            lora_path=lora_path,
            output_dir=output_dir,
            name=str(body.get("name") or "") or None,
            trigger_words=body.get("trigger_words") or [],
            description=str(body.get("description", "")),
            training_config=training_config,
            dataset_folder=str(body.get("dataset_folder") or ""),
            sample_images=body.get("sample_images") or [],
        ))
    except (packager.PackageError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


# --- Training queue ---------------------------------------------------------

def _queue_payload():
    return {"jobs": [{k: v for k, v in j.items() if k != "ui"} for j in train_queue.list_jobs()],
            "state": train_queue.state()}


@bp.get("/queue")
def queue_list():
    return jsonify(_queue_payload())


@bp.post("/queue/add")
def queue_add():
    body = request.get_json(force=True) or {}
    try:
        job = train_queue.add_job(body.get("name", ""), body.get("ui") or {}, body.get("backend_url", config.DEFAULT_BACKEND_URL))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"job": {k: v for k, v in job.items() if k != "ui"}, **_queue_payload()})


@bp.post("/queue/<job_id>/cancel")
def queue_cancel(job_id: str):
    if train_queue.cancel_job(job_id) is None:
        return jsonify({"error": "No such job"}), 404
    return jsonify(_queue_payload())


@bp.delete("/queue/<job_id>")
def queue_remove(job_id: str):
    if not train_queue.remove_job(job_id):
        return jsonify({"error": "That job is running or doesn't exist - cancel it first."}), 400
    return jsonify(_queue_payload())


@bp.post("/queue/<job_id>/move")
def queue_move(job_id: str):
    direction = (request.get_json(silent=True) or {}).get("direction", "up")
    if not train_queue.move_job(job_id, direction):
        return jsonify({"error": "Can't move that job (only waiting jobs can be reordered)."}), 400
    return jsonify(_queue_payload())


@bp.post("/queue/<job_id>/retry")
def queue_retry(job_id: str):
    if train_queue.retry_job(job_id) is None:
        return jsonify({"error": "Only finished jobs can be retried."}), 400
    return jsonify(_queue_payload())


@bp.post("/queue/<job_id>/rename")
def queue_rename(job_id: str):
    if not train_queue.rename_job(job_id, (request.get_json(silent=True) or {}).get("name", "")):
        return jsonify({"error": "Couldn't rename that job."}), 400
    return jsonify(_queue_payload())


@bp.post("/queue/pause")
def queue_pause():
    return jsonify({**_queue_payload(), "state": train_queue.set_paused(True)})


@bp.post("/queue/resume")
def queue_resume():
    return jsonify({**_queue_payload(), "state": train_queue.set_paused(False)})


@bp.post("/queue/clear_finished")
def queue_clear():
    n = train_queue.clear_finished()
    return jsonify({"removed": n, **_queue_payload()})
