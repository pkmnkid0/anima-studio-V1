"""LoRA packaging (§20) - bundle a trained LoRA into a
shareable, self-describing folder (and optional zip):

    <name>/
        <name>.safetensors      the LoRA weights
        README.md               description + usage
        trigger_words.txt       the trigger word(s)
        training_config.json    the exact config that produced it
        dataset_info.json       dataset summary + hash
        metadata.json           merged metadata for the hub
        sample_images/          sample generations
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

from .. import config
from . import history


class PackageError(RuntimeError):
    pass


_SAMPLE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def clean_path(raw: str | None) -> str:
    """Pasted paths often carry quotes (Windows "Copy as path") or spaces."""
    return (raw or "").strip().strip('"').strip("'").strip()


def read_safetensors_metadata(path: Path) -> dict:
    """The `__metadata__` dict from a .safetensors header (no tensors loaded)."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return {}
            n = int.from_bytes(raw, "little")
            if n <= 0 or n > 64 * 1024 * 1024:
                return {}
            header = json.loads(f.read(n).decode("utf-8"))
        meta = header.get("__metadata__") or {}
        return {str(k): v for k, v in meta.items()}
    except (OSError, ValueError):
        return {}


def _find_samples(lora: Path, limit: int = 12) -> list[Path]:
    """Sample images the trainer wrote next to the LoRA (sample/, samples/ or
    the same folder, named after the LoRA)."""
    found: list[Path] = []
    for d in (lora.parent / "sample", lora.parent / "samples", lora.parent):
        if not d.is_dir():
            continue
        try:
            for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime):
                if p.is_file() and p.suffix.lower() in _SAMPLE_EXTS and (d != lora.parent or p.stem.startswith(lora.stem)):
                    found.append(p)
        except OSError:
            continue
        if found:
            break
    return found[-limit:]


def inspect(lora_path: str) -> dict:
    """What we can learn about a LoRA file for auto-filling the package form."""
    lora = Path(clean_path(lora_path))
    if not lora.is_file():
        raise PackageError(f"LoRA file not found: {lora}")
    meta = read_safetensors_metadata(lora)
    triggers: list[str] = []
    raw = meta.get("ss_tag_frequency")
    if raw:
        try:
            freq = json.loads(raw) if isinstance(raw, str) else raw
            counts: dict[str, int] = {}
            for ds in freq.values():
                for tag, c in ds.items():
                    counts[tag] = counts.get(tag, 0) + int(c)
            triggers = [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:1]]
        except (ValueError, AttributeError, TypeError):
            pass
    return {
        "name": lora.stem,
        "size_mb": round(lora.stat().st_size / 1e6, 1),
        "network_dim": meta.get("ss_network_dim"),
        "network_alpha": meta.get("ss_network_alpha"),
        "base_model": meta.get("ss_sd_model_name") or meta.get("modelspec.architecture"),
        "steps": meta.get("ss_steps"),
        "epochs": meta.get("ss_epoch"),
        "trigger_words": triggers,
        "sample_images": [str(p) for p in _find_samples(lora)],
        "has_metadata": bool(meta),
    }


def _readme(manifest: dict) -> str:
    name = manifest.get("name", "LoRA")
    lines = [
        f"# {name}",
        "",
        manifest.get("description") or "A LoRA trained with Anima Studio.",
        "",
        "## Usage",
        "",
    ]
    if manifest.get("trigger_words"):
        lines.append(f"Trigger word(s): `{', '.join(manifest['trigger_words'])}`")
    else:
        lines.append("No trigger word.")
    lines += [
        "",
        f"- Base model type: {manifest.get('model_type', '?')}",
        f"- Project type: {manifest.get('project_type', '?')}",
        f"- Rank / alpha: {manifest.get('network_dim', '?')} / {manifest.get('network_alpha', '?')}",
        f"- Trained steps: {manifest.get('max_train_steps', '?')}",
        *( [f"- Base model: {manifest['base_model']}"] if manifest.get("base_model") else [] ),
        *( ["", "## Samples", ""] + [f"![sample {i}](sample_images/{n})" for i, n in enumerate(manifest.get("samples", []), 1)]
           if manifest.get("samples") else [] ),
        "",
        "Trained with [Anima Studio](https://github.com/).",
    ]
    return "\n".join(lines)


def package(lora_path: str, output_dir: str, *,
            name: str | None = None,
            trigger_words: list[str] | None = None,
            description: str = "",
            training_config: dict | None = None,
            dataset_folder: str | None = None,
            sample_images: list[str] | None = None) -> dict:
    lora = Path(clean_path(lora_path))
    if not lora.is_file():
        raise PackageError(f"LoRA file not found: {lora}")
    if lora.suffix.lower() != ".safetensors":
        raise PackageError("Pick a .safetensors LoRA file.")
    output_dir = clean_path(output_dir)
    if not output_dir:
        raise PackageError("Output folder is required.")
    dataset_folder = clean_path(dataset_folder) or None
    name = re.sub(r'[<>:"/\\|?*]+', "_", (name or "").strip()).strip(". ") or None

    out = Path(output_dir)
    if out.exists():
        if not out.is_dir():
            raise PackageError(f"Output path exists and is not a folder: {output_dir}")
        # An existing Anima Studio package can never be reused - that
        # guards against overwriting a previous export's weights.
        if (out / "metadata.json").is_file() or (out / "README.md").is_file():
            raise PackageError(
                f"A package already exists at '{out}'. Choose a fresh folder "
                "or delete that package first.")
        if any(out.iterdir()):
            # A non-empty folder the user picked (the common case with a
            # folder browser) is treated as the *parent*: the package goes
            # into a subfolder named after the LoRA.
            stem = name or lora.stem
            out = out / stem
            if out.exists() and any(out.iterdir()):
                raise PackageError(
                    f"'{out}' already contains files - pick a different output name "
                    "or clean that folder.")
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True)

    stem = name or lora.stem
    weights = out / f"{stem}.safetensors"
    shutil.copy2(lora, weights)

    meta = read_safetensors_metadata(lora)
    tc = training_config or {}

    def pick(key, meta_key):
        v = tc.get(key)
        return v if v not in (None, "") else meta.get(meta_key)

    manifest = {
        "name": stem,
        "model_type": tc.get("model_type", ""),
        "project_type": tc.get("project_type", ""),
        "network_dim": pick("network_dim", "ss_network_dim"),
        "network_alpha": pick("network_alpha", "ss_network_alpha"),
        "max_train_steps": pick("max_train_steps", "ss_steps"),
        "base_model": meta.get("ss_sd_model_name") or meta.get("modelspec.architecture"),
        "description": description,
        "trigger_words": [t for t in (trigger_words or []) if t],
    }

    (out / "trigger_words.txt").write_text(
        "\n".join(manifest["trigger_words"]), encoding="utf-8")
    (out / "README.md").write_text(_readme(manifest), encoding="utf-8")
    (out / "training_config.json").write_text(
        json.dumps(training_config or {}, indent=2), encoding="utf-8")

    dataset_info = {"folder": dataset_folder}
    if dataset_folder:
        dataset_info["hash"] = history.dataset_hash(dataset_folder)
        images = [p for p in Path(dataset_folder).iterdir()
                  if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS] \
            if Path(dataset_folder).is_dir() else []
        dataset_info["image_count"] = len(images)
    (out / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2), encoding="utf-8")

    samples_dir = out / "sample_images"
    samples_dir.mkdir()
    copied_samples = []
    wanted = [Path(clean_path(x)) for x in (sample_images or [])] or _find_samples(lora)
    for p in wanted:
        if p.is_file() and p.suffix.lower() in _SAMPLE_EXTS:
            shutil.copy2(p, samples_dir / p.name)
            copied_samples.append(p.name)
    if not copied_samples:
        samples_dir.rmdir()
    manifest["sample_count"] = len(copied_samples)
    manifest["samples"] = copied_samples
    (out / "README.md").write_text(_readme(manifest), encoding="utf-8")

    metadata = {
        **manifest,
        "weights": weights.name,
        "sample_images": copied_samples,
        "packaged_by": "Anima Studio",
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    # NB: with_suffix() would mangle names containing dots ("my_lora_v1.5"
    # -> "my_lora_v1.zip"), so the extension is appended instead.
    zip_path = out.parent / f"{out.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(out.parent).as_posix())

    return {
        "folder": str(out),
        "zip": str(zip_path),
        "files": sorted(p.relative_to(out).as_posix()
                        for p in out.rglob("*") if p.is_file()),
        "manifest": manifest,
    }
