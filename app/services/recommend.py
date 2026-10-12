"""Training presets (§5), automatic configuration
recommendations (§17) and the AI training advisor (§22).

Presets are the opinionated starting points per project
type; recommendations adapt them to the user's hardware
and dataset. The advisor merges a dataset analysis with a
recommendation into one report with an "apply" payload.
"""
from __future__ import annotations

from . import calculator

# ---------------------------------------------------------------------
# Presets - one block per (model type, project type).
# Values mirror the Train tab's UI payload keys.
# ---------------------------------------------------------------------

_ANIMA_PRESETS = {
    "character": {
        "label": "Anima · Character",
        "network": {"network_type": "lora", "network_dim": 16, "network_alpha": 16},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "1e-4",
                      "lr_scheduler": "cosine_with_restarts", "num_cycles": 2},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 2500},
    },
    "outfit": {
        "label": "Anima · Outfit",
        "network": {"network_type": "lora", "network_dim": 20, "network_alpha": 20},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "1e-4",
                      "lr_scheduler": "cosine_with_restarts", "num_cycles": 2},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 2200},
    },
    "style": {
        "label": "Anima · Style",
        "network": {"network_type": "lora", "network_dim": 32, "network_alpha": 32},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "1e-4",
                      "lr_scheduler": "cosine", "num_cycles": 1},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 3000},
    },
    "concept": {
        "label": "Anima · Concept",
        "network": {"network_type": "lora", "network_dim": 32, "network_alpha": 32},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "8e-5",
                      "lr_scheduler": "cosine", "num_cycles": 1},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 3500},
    },
    "high_fidelity": {
        "label": "Anima · High Fidelity",
        "network": {"network_type": "lora", "network_dim": 48, "network_alpha": 48},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "8e-5",
                      "lr_scheduler": "cosine_with_restarts", "num_cycles": 3},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 4000},
    },
    "strong_character": {
        "label": "Anima · Strong Character",
        "network": {"network_type": "lora", "network_dim": 24, "network_alpha": 24},
        "optimizer": {"optimizer_type": "Prodigy", "learning_rate": "1.0",
                      "lr_scheduler": "constant_with_warmup", "num_cycles": 1},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 3000},
    },
}

_SDXL_PRESETS = {
    "character": {
        "label": "SDXL · Character",
        "network": {"network_type": "lora", "network_dim": 16, "network_alpha": 16},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "1e-4",
                      "lr_scheduler": "cosine_with_restarts", "num_cycles": 2},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 2500},
    },
    "outfit": {
        "label": "SDXL · Outfit",
        "network": {"network_type": "lora", "network_dim": 20, "network_alpha": 20},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "1e-4",
                      "lr_scheduler": "cosine_with_restarts", "num_cycles": 2},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 2200},
    },
    "style": {
        "label": "SDXL · Style",
        "network": {"network_type": "lora", "network_dim": 32, "network_alpha": 32},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "1e-4",
                      "lr_scheduler": "cosine", "num_cycles": 1},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 3000},
    },
    "concept": {
        "label": "SDXL · Concept",
        "network": {"network_type": "lora", "network_dim": 32, "network_alpha": 32},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "8e-5",
                      "lr_scheduler": "cosine", "num_cycles": 1},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 3500},
    },
    "object": {
        "label": "SDXL · Object",
        "network": {"network_type": "lora", "network_dim": 16, "network_alpha": 16},
        "optimizer": {"optimizer_type": "AdamW", "learning_rate": "1e-4",
                      "lr_scheduler": "cosine", "num_cycles": 1},
        "general": {"resolution_width": 1024, "batch_size": 1, "mixed_precision": "bf16",
                    "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": 2000},
    },
}

PRESETS = {"anima": _ANIMA_PRESETS, "sdxl": _SDXL_PRESETS}


def presets(model_type: str | None = None) -> dict:
    if model_type is None:
        return PRESETS
    if model_type not in PRESETS:
        raise ValueError(f"Unknown model type: {model_type}")
    return PRESETS[model_type]


def preset(model_type: str, name: str) -> dict:
    table = presets(model_type)
    if name not in table:
        raise ValueError(f"No preset '{name}' for {model_type}")
    block = dict(table[name])
    block["key"] = name
    return block


# ---------------------------------------------------------------------
# Automatic configuration recommendations (§17)
# ---------------------------------------------------------------------

def recommend(model_type: str, image_count: int, project_type: str,
              vram_gb: float | None = None,
              diversity: str = "mixed") -> dict:
    """Hardware + dataset aware recommendation. Everything is
    overridable - this is a starting point, not a straitjacket."""
    if model_type not in PRESETS:
        raise ValueError(f"Unknown model type: {model_type}")
    if project_type not in calculator.PROJECT_TYPES:
        raise ValueError(f"Unknown project type: {project_type}")

    base = preset(model_type, project_type if project_type in PRESETS[model_type] else "character")

    # Dataset-size driven adjustments
    if image_count <= 10:
        rank, steps_note = 12, "tiny dataset - keep rank low to reduce overfitting"
        repeats = 4
    elif image_count <= 30:
        rank, steps_note = 16, "small dataset - standard character/outfit settings"
        repeats = 3
    elif image_count <= 100:
        rank, steps_note = 16, "medium dataset - standard settings"
        repeats = 2
    elif image_count <= 300:
        rank, steps_note = 24, "large dataset - extra capacity pays off"
        repeats = 1
    else:
        rank, steps_note = 32, "very large dataset - full capacity, no repeats needed"
        repeats = 1

    if diversity == "low":
        rank = max(12, rank - 4)
        steps_note = "low diversity - lower rank helps avoid memorizing near-copies"
    elif diversity == "high":
        rank = min(48, rank + 8)

    # VRAM-driven resolution/batch
    resolution, batch_size, grad_accum = 1024, 1, 4
    if vram_gb is not None:
        if vram_gb >= 16:
            batch_size, grad_accum = 2, 2
        elif vram_gb < 8:
            resolution = 768

    calc = calculator.calculate(
        images=image_count, repeats=repeats, epochs=1,
        batch_size=batch_size, grad_accum=grad_accum,
        project_type=project_type)
    target = calc["recommended_ranges"]["balanced"]
    epochs = max(1, round(target / max(calc["steps_per_epoch"], 1)))

    return {
        "model_type": model_type,
        "project_type": project_type,
        "dataset_summary": {
            "images": image_count, "repeats": repeats,
            "diversity": diversity, "note": steps_note,
        },
        "network": {"network_type": "lora", "network_dim": rank,
                    "network_alpha": rank},
        "general": {"resolution_width": resolution, "batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "mixed_precision": "bf16", "gradient_checkpointing": True},
        "training": {"max_train_mode": "steps", "max_train_value": target,
                     "epochs_equivalent": epochs},
        "optimizer": base["optimizer"],
        "recommended_ranges": calc["recommended_ranges"],
        "warnings": calc["warnings"],
        "message": (
            f"Your dataset contains {image_count} images and looks like a "
            f"{project_type} LoRA."
        ),
    }


# ---------------------------------------------------------------------
# AI training advisor (§22)
# ---------------------------------------------------------------------

def advisor_report(analysis: dict, health: dict, model_type: str,
                   project_type: str, vram_gb: float | None = None) -> dict:
    """Merge a dataset analysis into an actionable report:
    what to clean up, what to add, and how to train."""
    image_count = analysis.get("count", 0)
    recommendation = recommend(model_type, image_count, project_type, vram_gb)

    checks = []
    if health.get("grade") in ("Excellent", "Good"):
        checks.append(("good", f"Dataset health is {health['grade']} ({health['score']}/100)"))
    else:
        checks.append(("warn", f"Dataset health is {health['grade']} ({health['score']}/100)"))

    n_near = sum(len(g) - 1 for g in analysis.get("near_duplicates", []))
    n_exact = sum(len(g) - 1 for g in analysis.get("exact_duplicates", []))
    if n_near or n_exact:
        checks.append(("warn", f"{n_near + n_exact} duplicate/near-duplicate image(s) detected"))
    else:
        checks.append(("good", "No duplicates detected"))

    n_corrupt = len(analysis.get("corrupt", []))
    if n_corrupt:
        checks.append(("warn", f"{n_corrupt} corrupted image(s) should be removed"))

    coverage = analysis.get("caption_coverage", 0.0)
    if coverage >= 100:
        checks.append(("good", "Every image is captioned"))
    else:
        checks.append(("warn", f"{round(100 - coverage, 1)}% of images have no caption"))

    n_tiny = len(analysis.get("tiny", []))
    if n_tiny:
        checks.append(("warn", f"{n_tiny} low-resolution image(s) under 256px"))

    # Outfit-balance check from the analysis' tag frequency
    top_tags = analysis.get("tag_frequency", [])
    if top_tags and top_tags[0]["pct"] >= 40:
        checks.append(("warn",
                       f"'{top_tags[0]['tag']}' dominates captions at {top_tags[0]['pct']}%"))

    actions = []
    if n_near + n_exact:
        actions.append(f"Remove {n_near + n_exact} duplicate image(s) in the Dataset panel.")
    if n_corrupt:
        actions.append(f"Remove {n_corrupt} corrupted image(s).")
    if coverage < 100:
        actions.append("Caption the remaining images before training.")
    if image_count < 20:
        actions.append("Add 5-15 more images (varied poses/outfits) if possible.")
    if not actions:
        actions.append("Dataset looks good - start training with the recommended settings.")

    return {
        "checks": [{"kind": kind, "text": text} for kind, text in checks],
        "recommendation": recommendation,
        "actions": actions,
        "apply_payload": {
            "network": recommendation["network"],
            "general": recommendation["general"],
            "training": recommendation["training"],
            "optimizer": recommendation["optimizer"],
        },
    }
