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

def _moss_presets() -> dict:
    """MOSS-Image LoRA starting points: the Anima ones (both are rectified-flow DiTs trained with AdamW
    LoRA at 1024 px) relabelled. Hardware settings come from MOSS_VRAM_PROFILES, not from here."""
    out = {}
    for key, block in _ANIMA_PRESETS.items():
        if key == "strong_character":
            continue                                    # Prodigy-based; not validated on MOSS-Image
        b = {k: (dict(v) if isinstance(v, dict) else v) for k, v in block.items()}
        b["label"] = block["label"].replace("Anima", "MOSS-Image")
        out[key] = b
    return out


# What each memory budget needs to train the FULL bf16 MOSS-Image DiT (5B, ~10.2 GB of block weights).
# Block swap parks N of the 42 blocks (~0.23 GB each) in system RAM and streams them in per step, so VRAM
# falls roughly 0.23 GB per swapped block and speed falls with it. The 9B text encoder is only needed while
# captions are cached: "offload" keeps exact bf16 weights, streaming the layers that do not fit.
MOSS_VRAM_PROFILES = {
    "8": {
        "label": "8 GB · full bf16 (block swap 36)", "min_vram_gb": 8, "min_ram_gb": 32,
        "model": {"moss_blocks_to_swap": 36, "moss_int8": "off", "moss_compile": "none",
                  "moss_attn_backend": "auto", "moss_te_mode": "offload", "moss_te_gpu_gb": 6.0,
                  "moss_unchecked_blocks": 0},
        "general": {"gradient_checkpointing": True, "batch_size": 1, "cache_latents": True},
        "note": "~6 GB peak at 1024 px. Slowest profile; needs about 32 GB of system RAM (blocks and text encoder "
                "live there). Raise Blocks to swap to 40 if you still run out of memory, or train at 768 px.",
    },
    "10": {
        "label": "10 GB · full bf16 (block swap 30)", "min_vram_gb": 10, "min_ram_gb": 32,
        "model": {"moss_blocks_to_swap": 30, "moss_int8": "off", "moss_compile": "none",
                  "moss_attn_backend": "auto", "moss_te_mode": "offload", "moss_te_gpu_gb": 7.5,
                  "moss_unchecked_blocks": 0},
        "general": {"gradient_checkpointing": True, "batch_size": 1, "cache_latents": True},
        "note": "~7 GB peak at 1024 px, leaving ~3 GB of headroom for the desktop and spikes. About 30% faster "
                "than the 8 GB profile; needs about 32 GB of system RAM. Drop Blocks to swap to 26 for more "
                "speed if you have the room, or raise it to 34 if you run out of memory.",
    },
    "10-int8": {
        "label": "10 GB · INT8 base (no swap, faster, not pure bf16)", "min_vram_gb": 10, "min_ram_gb": 24,
        "model": {"moss_blocks_to_swap": 0, "moss_int8": "on", "moss_compile": "none",
                  "moss_attn_backend": "auto", "moss_te_mode": "offload", "moss_te_gpu_gb": 7.5,
                  "moss_unchecked_blocks": 0},
        "general": {"gradient_checkpointing": True, "batch_size": 1, "cache_latents": True},
        "note": "The bf16 DiT is loaded on the CPU and quantized block by block on the GPU, so the card only ever holds "
                "the int8 weights (~5 GB). The LoRA learns against an INT8 copy of the base. Faster than block swap. "
                "Set Compile = blocks for a further ~25% if Visual Studio Build Tools are installed (needed on Windows).",
    },
    "12": {
        "label": "12 GB · full bf16 (block swap 24)", "min_vram_gb": 12, "min_ram_gb": 32,
        "model": {"moss_blocks_to_swap": 24, "moss_int8": "off", "moss_compile": "none",
                  "moss_attn_backend": "auto", "moss_te_mode": "offload", "moss_te_gpu_gb": 9.0,
                  "moss_unchecked_blocks": 0},
        "general": {"gradient_checkpointing": True, "batch_size": 1, "cache_latents": True},
        "note": "~8 GB peak at 1024 px.",
    },
    "16": {
        "label": "16 GB · full bf16 (block swap 12)", "min_vram_gb": 16, "min_ram_gb": 32,
        "model": {"moss_blocks_to_swap": 12, "moss_int8": "off", "moss_compile": "none",
                  "moss_attn_backend": "auto", "moss_te_mode": "offload", "moss_te_gpu_gb": 13.0,
                  "moss_unchecked_blocks": 0},
        "general": {"gradient_checkpointing": True, "batch_size": 1, "cache_latents": True},
        "note": "~11 GB peak at 1024 px.",
    },
    "24": {
        "label": "24 GB · full bf16, no swap", "min_vram_gb": 24, "min_ram_gb": 24,
        "model": {"moss_blocks_to_swap": 0, "moss_int8": "off", "moss_compile": "blocks",
                  "moss_attn_backend": "cudnn", "moss_te_mode": "full", "moss_unchecked_blocks": 0},
        "general": {"gradient_checkpointing": True, "batch_size": 1, "cache_latents": True},
        "note": "13 GB peak at 1024 px (12 GB with Compile = blocks). The fork's reference setup: 1.4 s per 1024 px image on an RTX 4090.",
    },
    "24-fast": {
        "label": "24 GB · INT8 base (fastest, not pure bf16)", "min_vram_gb": 12, "min_ram_gb": 24,
        "model": {"moss_blocks_to_swap": 0, "moss_int8": "on", "moss_compile": "blocks",
                  "moss_attn_backend": "cudnn", "moss_te_mode": "full", "moss_unchecked_blocks": 0},
        "general": {"gradient_checkpointing": True, "batch_size": 1, "cache_latents": True},
        "note": "The LoRA learns against an INT8-quantised copy of the base (7.5 GB peak, ~1.2 s per image on an "
                "RTX 4090). Not compatible with block swap.",
    },
}


def moss_profile_for(vram_gb: float | None) -> str:
    """The largest profile that fits `vram_gb` (the smallest one when unknown)."""
    if not vram_gb:
        return "8"
    fit = [k for k in ("8", "10", "12", "16", "24") if MOSS_VRAM_PROFILES[k]["min_vram_gb"] <= vram_gb + 0.6]
    return fit[-1] if fit else "8"


# KohakuMoss VRAM profiles: the preset layers under kohaku_backend/configs/moss/vram/ (tests/test_kohaku.py keeps this
# table equal to the layer files). A card selects the largest layer it holds within 5 % of the layer's size - a 10 GB
# card gets the 8 GB layer. The keys are the Train tab's `kohaku_*` fields; "" means "not set by this layer".
def _kp(label, min_gb, profile, tokens, te, vae, minutes, note):
    m = {"kohaku_memory_profile": profile, "kohaku_max_tokens": tokens,
         "kohaku_te_memory": te.get("memory", ""), "kohaku_te_ahead": te.get("ahead", ""),
         "kohaku_te_batch": te.get("batch", ""), "kohaku_te_max_tokens": te.get("max_tokens", ""),
         "kohaku_vae_memory": vae.get("memory", ""), "kohaku_vae_ahead": vae.get("ahead", ""),
         "kohaku_vae_tile": {True: "true", False: "false"}.get(vae.get("tile"), ""),
         "kohaku_save_minutes": minutes}
    return {"label": label, "min_vram_gb": min_gb, "model": m, "note": note}


KOHAKU_VRAM_PROFILES = {
    "6": _kp("6 GB · streamed weights, host optimizer", 6, "ckpt-ring2-actoff-optcpu-gradoff", 4352,
             {"memory": "ring", "ahead": 4}, {}, 30,
             "Two resident blocks, activation and gradient offload, the AdamW update on the host. Text and latent "
             "caches are built offline first, so the card only holds the backbone while training."),
    "8": _kp("8 GB · minimal (also the pick for a 10 GB card)", 8, "minimal", 4352,
             {"memory": "ring", "ahead": 8, "batch": 16, "max_tokens": 8192},
             {"memory": "offload", "ahead": 8, "tile": True}, 30,
             "The low-VRAM trainer's 8 GB layer is checked with the allocator capped at 8 GB, so it should not spill into shared "
             "memory. Needs about 32 GB of system RAM while the model loads and about 19 GB during the steps."),
    "12": _kp("12 GB · four streamed blocks", 12, "ckpt-ring4-actoff-optstream-gradoff", 6144,
              {"memory": "ring", "ahead": 4, "batch": 16, "max_tokens": 8192}, {"memory": "offload", "ahead": 4}, "",
              "Faster than the 8 GB layer. On a 10 GB card try it only if Task Manager shows no shared-memory use."),
    "16": _kp("16 GB · low", 16, "low", 8192, {"memory": "ring", "ahead": 4}, {}, "",
              "Block checkpointing, four resident streamed blocks, activation offload."),
    "24": _kp("24 GB · balanced", 24, "balanced", 12288, {"memory": "ring", "ahead": 8}, {}, "",
              "Block checkpointing with 16 resident streamed blocks."),
    "auto": _kp("Auto · the planner picks", 0, "auto", "", {"memory": "auto", "ahead": 4}, {"memory": "auto"}, "",
                "The low-VRAM trainer picks the fastest profile whose memory estimate fits the card (any size)."),
}


def kohaku_profile_for(vram_gb: float | None) -> str:
    """The low-VRAM trainer's own rule: the largest layer the card holds within 5 % of its size; above 24 GB, auto."""
    if not vram_gb:
        return "8"
    if vram_gb > 24:
        return "auto"
    fit = [k for k in ("6", "8", "12", "16", "24") if vram_gb >= KOHAKU_VRAM_PROFILES[k]["min_vram_gb"] * 0.95]
    return fit[-1] if fit else "6"


PRESETS = {"anima": _ANIMA_PRESETS, "sdxl": _SDXL_PRESETS, "moss": _moss_presets()}


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
        if model_type == "moss":
            batch_size, grad_accum = 1, 4               # swapped blocks make every step slow; accumulate instead

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
