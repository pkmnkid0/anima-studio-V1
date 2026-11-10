"""What Anima Studio's AI edit tools offer, mirrored from IOPaint 1.6.0 (the engine behind lama-cleaner).

Pure data and stdlib only: the Flask app (which has no torch), the worker (which runs inside the edit environment) and
the tests all read the same lists. Every field of IOPaint's `InpaintRequest` is described in `PARAMS`, so no setting the
engine understands is left without a control; tests compare this file against the real schema.
"""
from __future__ import annotations

IOPAINT_VERSION = "1.6.0"

DEVICES = ["auto", "cuda", "cpu"]

# ------------------------------------------------------------------ models

# Erase models: they only fill the painted area. Names are IOPaint's own ids.
ERASE_MODELS = [
    {"id": "lama", "label": "LaMa", "tags": ["erase"],
     "hint": "The standard all-round remover for objects, text and watermarks. Fast."},
    {"id": "anime-lama", "label": "Anime LaMa", "tags": ["erase"],
     "hint": "LaMa fine-tuned on anime and manga art. A good first choice for anime datasets."},
    {"id": "manga", "label": "Manga", "tags": ["erase"],
     "hint": "Made for manga pages: keeps line art and screentone."},
    {"id": "zits", "label": "ZITS", "tags": ["erase", "zits"],
     "hint": "Structure-aware: continues lines and edges across the hole. Heavier than LaMa."},
    {"id": "mat", "label": "MAT", "tags": ["erase"],
     "hint": "Transformer-based, handles large holes. Comfortable only on a GPU."},
    {"id": "fcf", "label": "FcF", "tags": ["erase"],
     "hint": "Fourier-convolution model, aimed at large holes."},
    {"id": "ldm", "label": "LDM", "tags": ["erase", "ldm"],
     "hint": "Latent-diffusion eraser. Slower than LaMa."},
    {"id": "migan", "label": "MI-GAN", "tags": ["erase"],
     "hint": "Small and very fast, lower quality than LaMa."},
]

# Generative models: they paint new content from a prompt (erase-and-replace, outpainting). Hugging Face repo ids.
DIFFUSION_MODELS = [
    {"id": "Sanster/anything-4.0-inpainting", "label": "Anything 4.0 inpainting (anime, SD 1.5)", "family": "sd",
     "tags": ["diffusion", "prompt"]},
    {"id": "runwayml/stable-diffusion-inpainting", "label": "Stable Diffusion 1.5 inpainting", "family": "sd",
     "tags": ["diffusion", "prompt"]},
    {"id": "Sanster/Realistic_Vision_V1.4-inpainting", "label": "Realistic Vision 1.4 inpainting (SD 1.5)",
     "family": "sd", "tags": ["diffusion", "prompt"]},
    {"id": "Uminosachi/realisticVisionV51_v51VAE-inpainting", "label": "Realistic Vision 5.1 inpainting (SD 1.5)",
     "family": "sd", "tags": ["diffusion", "prompt"]},
    {"id": "redstonehero/dreamshaper-inpainting", "label": "DreamShaper inpainting (SD 1.5)", "family": "sd",
     "tags": ["diffusion", "prompt"]},
    {"id": "stabilityai/stable-diffusion-2-inpainting", "label": "Stable Diffusion 2 inpainting", "family": "sd2",
     "tags": ["diffusion", "prompt"]},
    {"id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "label": "Stable Diffusion XL inpainting",
     "family": "sdxl", "tags": ["diffusion", "prompt"]},
    {"id": "eienmojiki/Anything-XL", "label": "Anything XL (anime, SDXL)", "family": "sdxl",
     "tags": ["diffusion", "prompt"]},
    {"id": "SG161222/RealVisXL_V5.0", "label": "RealVisXL 5.0 (SDXL)", "family": "sdxl",
     "tags": ["diffusion", "prompt"]},
    {"id": "RunDiffusion/Juggernaut-XI-v11", "label": "Juggernaut XI (SDXL)", "family": "sdxl",
     "tags": ["diffusion", "prompt"]},
    {"id": "Fantasy-Studio/Paint-by-Example", "label": "Paint by Example (fill from a reference image)",
     "family": "other", "tags": ["diffusion", "pbe"]},
    {"id": "timbrooks/instruct-pix2pix", "label": "InstructPix2Pix (edit by instruction)", "family": "other",
     "tags": ["diffusion", "prompt", "p2p"]},
    {"id": "kandinsky-community/kandinsky-2-2-decoder-inpaint", "label": "Kandinsky 2.2 inpainting",
     "family": "other", "tags": ["diffusion", "prompt"]},
    {"id": "Sanster/PowerPaint-V1-stable-diffusion-inpainting", "label": "PowerPaint v1 (tasks: remove, fill, outpaint)",
     "family": "other", "tags": ["diffusion", "prompt", "powerpaint"]},
]

# Tags a model that is not in the lists above gets (a custom Hugging Face repo, a local diffusers folder or a
# single-file .safetensors checkpoint put in the models folder): treated as a prompt-driven diffusion model.
CUSTOM_MODEL_TAGS = ["diffusion", "prompt"]

# ------------------------------------------------------------------ plugins

SAM_MODELS = [
    {"id": "mobile_sam", "label": "MobileSAM (≈40 MB, fastest)"},
    {"id": "vit_b", "label": "SAM ViT-B (≈375 MB)"},
    {"id": "vit_l", "label": "SAM ViT-L (≈1.2 GB)"},
    {"id": "vit_h", "label": "SAM ViT-H (≈2.4 GB, best)"},
    {"id": "sam_hq_vit_b", "label": "SAM-HQ ViT-B (sharper edges)"},
    {"id": "sam_hq_vit_l", "label": "SAM-HQ ViT-L"},
    {"id": "sam_hq_vit_h", "label": "SAM-HQ ViT-H"},
    {"id": "sam2_tiny", "label": "SAM 2 tiny"},
    {"id": "sam2_small", "label": "SAM 2 small"},
    {"id": "sam2_base", "label": "SAM 2 base+"},
    {"id": "sam2_large", "label": "SAM 2 large"},
    {"id": "sam2_1_tiny", "label": "SAM 2.1 tiny"},
    {"id": "sam2_1_small", "label": "SAM 2.1 small"},
    {"id": "sam2_1_base", "label": "SAM 2.1 base+"},
    {"id": "sam2_1_large", "label": "SAM 2.1 large"},
]

# "anime-seg" is IOPaint's anime-character cut-out plugin (always CPU); the rest are rembg / RMBG models.
REMOVE_BG_MODELS = [
    {"id": "anime-seg", "label": "Anime character cut-out (anime-segmentation)"},
    {"id": "isnet-general-use", "label": "IS-Net general"},
    {"id": "u2net", "label": "U²-Net"},
    {"id": "u2netp", "label": "U²-Net small"},
    {"id": "u2net_human_seg", "label": "U²-Net human"},
    {"id": "u2net_cloth_seg", "label": "U²-Net clothes"},
    {"id": "silueta", "label": "Silueta"},
    {"id": "briaai/RMBG-1.4", "label": "BRIA RMBG 1.4 (GPU capable)"},
    {"id": "briaai/RMBG-2.0", "label": "BRIA RMBG 2.0 (GPU capable)"},
    {"id": "birefnet-general", "label": "BiRefNet general"},
    {"id": "birefnet-general-lite", "label": "BiRefNet general lite"},
    {"id": "birefnet-portrait", "label": "BiRefNet portrait"},
    {"id": "birefnet-dis", "label": "BiRefNet DIS"},
    {"id": "birefnet-hrsod", "label": "BiRefNet HRSOD"},
    {"id": "birefnet-cod", "label": "BiRefNet COD"},
    {"id": "birefnet-massive", "label": "BiRefNet massive"},
]

REALESRGAN_MODELS = [
    {"id": "RealESRGAN_x4plus_anime_6B", "label": "RealESRGAN x4 anime (6B)"},
    {"id": "realesr-general-x4v3", "label": "Real-ESRGAN general x4 v3"},
    {"id": "RealESRGAN_x4plus", "label": "Real-ESRGAN x4 plus"},
]

FACE_MODELS = [
    {"id": "gfpgan", "label": "GFPGAN 1.4"},
    {"id": "restoreformer", "label": "RestoreFormer"},
]

# What each Actions button runs on the worker.
PLUGIN_IDS = ("removebg", "animeseg", "realesrgan", "gfpgan", "restoreformer", "sam")

# ------------------------------------------------------------------ download packs (Backend tab)
# Each item is (kind, name) exactly as the worker's "download" command takes it.

def _erase(ids=None):
    return [("model", m["id"]) for m in ERASE_MODELS if ids is None or m["id"] in ids]


MODEL_PACKS = [
    {"id": "essentials", "label": "Essentials (about 1 GB)",
     "items": _erase(["lama", "anime-lama", "manga"]) + [("sam", "mobile_sam"), ("removebg", "anime-seg"),
                                                          ("realesrgan", "RealESRGAN_x4plus_anime_6B"), ("gfpgan", "gfpgan")]},
    {"id": "erase", "label": "All erase models (LaMa, Anime LaMa, Manga, ZITS, MAT, FcF, LDM, MI-GAN)", "items": _erase()},
    {"id": "all_small", "label": "Everything except Stable Diffusion models (a few GB)",
     "items": _erase() + [("sam", "mobile_sam"), ("sam", "vit_b"), ("removebg", "anime-seg"), ("removebg", "isnet-general-use"),
                          ("realesrgan", "RealESRGAN_x4plus_anime_6B"), ("realesrgan", "realesr-general-x4v3"),
                          ("realesrgan", "RealESRGAN_x4plus"), ("gfpgan", "gfpgan"), ("restoreformer", "restoreformer")]},
]


def pick_groups() -> list[dict]:
    """Every downloadable model, grouped, for the 'Choose models' list."""
    def g(title, kind, rows, large=False):
        return {"title": title, "large": large,
                "items": [{"kind": kind, "name": r["id"], "label": r["label"]} for r in rows]}
    return [
        g("Erase models", "model", ERASE_MODELS),
        g("Segment Anything (click to select)", "sam", SAM_MODELS),
        g("Background removal", "removebg", REMOVE_BG_MODELS),
        g("Upscalers", "realesrgan", REALESRGAN_MODELS),
        g("Face restoration", "face", FACE_MODELS),
        g("Stable Diffusion models (2-7 GB each)", "model", DIFFUSION_MODELS, large=True),
    ]


# ------------------------------------------------------------------ enumerations

HD_STRATEGIES = ["Crop", "Resize", "Original"]
LDM_SAMPLERS = ["plms", "ddim"]
SD_SAMPLERS = [
    "UniPC", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ 2M SDE", "DPM++ 2M SDE Karras", "DPM++ SDE", "DPM++ SDE Karras",
    "DPM2", "DPM2 Karras", "DPM2 a", "DPM2 a Karras", "Euler", "Euler a", "Heun", "LMS", "LMS Karras", "DDIM", "PNDM",
    "LCM",
]
CV2_FLAGS = ["INPAINT_NS", "INPAINT_TELEA"]
POWERPAINT_TASKS = ["text-guided", "context-aware", "shape-guided", "object-remove", "outpainting"]
SD_CONTROLNETS = [
    "lllyasviel/control_v11p_sd15_canny", "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_inpaint", "lllyasviel/control_v11f1p_sd15_depth",
]
SD2_CONTROLNETS = [
    "thibaud/controlnet-sd21-canny-diffusers", "thibaud/controlnet-sd21-depth-diffusers",
    "thibaud/controlnet-sd21-openpose-diffusers",
]
SDXL_CONTROLNETS = [
    "thibaud/controlnet-openpose-sdxl-1.0", "destitech/controlnet-inpaint-dreamer-sdxl",
    "diffusers/controlnet-canny-sdxl-1.0", "diffusers/controlnet-canny-sdxl-1.0-mid",
    "diffusers/controlnet-canny-sdxl-1.0-small", "diffusers/controlnet-depth-sdxl-1.0",
    "diffusers/controlnet-depth-sdxl-1.0-mid", "diffusers/controlnet-depth-sdxl-1.0-small",
]
ALL_CONTROLNETS = SD_CONTROLNETS + SD2_CONTROLNETS + SDXL_CONTROLNETS
SD_BRUSHNETS = ["Sanster/brushnet_random_mask", "Sanster/brushnet_segmentation_mask"]
SDXL_BRUSHNETS = ["Regulus0725/random_mask_brushnet_ckpt_sdxl_regulus_v1"]
ALL_BRUSHNETS = SD_BRUSHNETS + SDXL_BRUSHNETS

# ------------------------------------------------------------------ engine options (ModelManager arguments)

ENGINE_OPTIONS = [
    {"key": "device", "label": "Device", "kind": "select", "default": "auto", "options": DEVICES,
     "hint": "auto uses the GPU when there is one. Pick cpu to leave the GPU free for training."},
    {"key": "no_half", "label": "Full precision (fp32)", "kind": "bool", "default": False,
     "hint": "Turn on if a diffusion model returns black or green images. Uses more memory."},
    {"key": "low_mem", "label": "Low memory mode", "kind": "bool", "default": False,
     "hint": "Attention slicing and VAE tiling for diffusion models."},
    {"key": "cpu_offload", "label": "Offload diffusion model to RAM", "kind": "bool", "default": False,
     "hint": "Keeps the diffusion model's weights in system RAM: far less VRAM, slower."},
    {"key": "cpu_textencoder", "label": "Text encoder on CPU", "kind": "bool", "default": False,
     "hint": "Runs the diffusion model's text encoder on the CPU to save VRAM."},
    {"key": "disable_nsfw", "label": "Disable NSFW checker", "kind": "bool", "default": True,
     "hint": "Stable Diffusion's safety checker replaces images it flags with a black picture. Leave this on (checker disabled) to avoid blank results."},
    {"key": "local_files_only", "label": "Offline (no downloads)", "kind": "bool", "default": False,
     "hint": "Only use models already downloaded; never contact Hugging Face."},
]

MODEL_LOAD_KEYS = ("device", "no_half", "low_mem", "cpu_offload", "cpu_textencoder", "disable_nsfw", "local_files_only")

# ------------------------------------------------------------------ InpaintRequest

# Group titles in display order.
GROUPS = [
    ("large", "Large images (erase models)"),
    ("classic", "OpenCV (classic methods)"),
    ("ldm", "LDM"),
    ("zits", "ZITS"),
    ("prompt", "Prompt"),
    ("gen", "Generation"),
    ("regions", "Crop / extend regions"),
    ("pbe", "Paint by Example"),
    ("p2p", "InstructPix2Pix"),
    ("controlnet", "ControlNet"),
    ("brushnet", "BrushNet"),
    ("powerpaint", "PowerPaint"),
]


def _p(key, label, kind, default, group, models, hint="", **extra):
    spec = {"key": key, "label": label, "kind": kind, "default": default, "group": group, "models": models, "hint": hint}
    spec.update(extra)
    return spec


# `models` says for which kind of model the control is shown: erase / ldm / zits / diffusion / prompt / pbe / p2p /
# powerpaint / classic (the OpenCV methods). Everything is always accepted by the engine.
PARAMS = [
    _p("hd_strategy", "Large-image strategy", "select", "Crop", "large", ["erase"],
       "Crop: work on the area around the mask (best for big images). Resize: shrink to the limit, then paste back. "
       "Original: run the whole image.", options=HD_STRATEGIES),
    _p("hd_strategy_crop_trigger_size", "Crop above (px)", "int", 800, "large", ["erase"],
       "Crop strategy only: used when the image's longer side is larger than this.", min=64, max=8192, step=16),
    _p("hd_strategy_crop_margin", "Crop margin (px)", "int", 128, "large", ["erase"],
       "Crop strategy: context kept around the masked area.", min=0, max=1024, step=8),
    _p("hd_strategy_resize_limit", "Resize limit (px)", "int", 1280, "large", ["erase"],
       "Resize strategy: the longer side is shrunk to this before inpainting.", min=128, max=8192, step=16),

    _p("cv2_flag", "Algorithm", "select", "INPAINT_NS", "classic", ["classic"],
       "Navier-Stokes or Telea. (The plain Telea / Navier-Stokes methods of the eraser use these.)", options=CV2_FLAGS),
    _p("cv2_radius", "Radius (px)", "int", 4, "classic", ["classic"],
       "Neighbourhood considered around each filled pixel.", min=1, max=100, step=1),

    _p("ldm_steps", "Steps", "int", 20, "ldm", ["ldm"], "Denoising steps of the LDM eraser.", min=1, max=200, step=1),
    _p("ldm_sampler", "Sampler", "select", "plms", "ldm", ["ldm"], "", options=LDM_SAMPLERS),

    _p("zits_wireframe", "Wireframe detection", "bool", True, "zits", ["zits"],
       "Detect straight lines (buildings, panels) and continue them across the hole."),

    _p("prompt", "Prompt", "area", "", "prompt", ["prompt"],
       "What to paint in the masked area. For an eraser, leave it empty or describe the background."),
    _p("negative_prompt", "Negative prompt", "area", "", "prompt", ["prompt"], "What to avoid."),

    _p("sd_scale", "Work scale", "float", 1.0, "gen", ["diffusion"],
       "Shrink the image before generating (0.1 - 1); the area outside the mask keeps full quality.",
       min=0.1, max=1.0, step=0.05),
    _p("sd_strength", "Strength", "float", 1.0, "gen", ["diffusion"],
       "How much noise is added: 1 = ignore the existing pixels under the mask, lower = stay closer to them.",
       min=0.0, max=1.0, step=0.01),
    _p("sd_steps", "Steps", "int", 50, "gen", ["diffusion"], "Denoising steps.", min=1, max=200, step=1),
    _p("sd_guidance_scale", "Guidance scale", "float", 7.5, "gen", ["diffusion"],
       "Higher follows the prompt more closely, usually at some cost in quality.", min=0.0, max=30.0, step=0.1),
    _p("sd_sampler", "Sampler", "select", "UniPC", "gen", ["diffusion"], "", options=SD_SAMPLERS),
    _p("sd_seed", "Seed", "int", 42, "gen", ["diffusion"], "-1 = a random seed each time.", min=-1, max=2147483647, step=1),
    _p("sd_mask_blur", "Mask blur (px)", "int", 11, "gen", ["diffusion"],
       "Softens the mask edge so the new area blends in.", min=0, max=128, step=1),
    _p("sd_match_histograms", "Match colours to the original", "bool", False, "gen", ["diffusion"],
       "Match the generated area's histogram to the unmasked part."),
    _p("sd_keep_unmasked_area", "Keep the unmasked area unchanged", "bool", True, "gen", ["erase", "diffusion"],
       "Paste the original pixels back outside the mask."),
    _p("sd_lcm_lora", "LCM-LoRA (fast, few steps)", "bool", False, "gen", ["diffusion"],
       "Use the LCM LoRA: set Steps to about 4-8, Guidance 1-2 and Sampler LCM."),
    _p("sd_outpainting_softness", "Outpainting softness", "float", 20.0, "gen", ["diffusion"],
       "Blend width where extended areas meet the picture.", min=0.0, max=200.0, step=1.0),
    _p("sd_outpainting_space", "Outpainting space", "float", 20.0, "gen", ["diffusion"],
       "How far the new area overlaps the existing picture.", min=0.0, max=200.0, step=1.0),

    _p("use_croper", "Generate inside a crop box", "bool", False, "regions", ["diffusion"],
       "Run the diffusion model on a smaller box only (much faster on big images). The Erase button fills the box "
       "from the mask automatically when this is on and the numbers are 0."),
    _p("croper_x", "Crop x", "int", 0, "regions", ["diffusion"], "", min=0, max=100000, step=1),
    _p("croper_y", "Crop y", "int", 0, "regions", ["diffusion"], "", min=0, max=100000, step=1),
    _p("croper_width", "Crop width", "int", 512, "regions", ["diffusion"], "", min=64, max=100000, step=8),
    _p("croper_height", "Crop height", "int", 512, "regions", ["diffusion"], "", min=64, max=100000, step=8),
    _p("use_extender", "Extend (outpaint) the image", "bool", False, "regions", ["diffusion"],
       "Set by the Extend tool. The extender box is in image pixels and may start outside the image (negative x / y)."),
    _p("extender_x", "Extender x", "int", 0, "regions", ["diffusion"], "", min=-100000, max=100000, step=1),
    _p("extender_y", "Extender y", "int", 0, "regions", ["diffusion"], "", min=-100000, max=100000, step=1),
    _p("extender_width", "Extender width", "int", 640, "regions", ["diffusion"], "", min=64, max=100000, step=8),
    _p("extender_height", "Extender height", "int", 640, "regions", ["diffusion"], "", min=64, max=100000, step=8),

    _p("paint_by_example_example_image", "Example image", "image", None, "pbe", ["pbe"],
       "The picture whose content is painted into the mask."),

    _p("p2p_image_guidance_scale", "Image guidance", "float", 1.5, "p2p", ["p2p"],
       "How closely the result follows the original image.", min=0.0, max=10.0, step=0.1),

    _p("enable_controlnet", "Use ControlNet", "bool", False, "controlnet", ["diffusion"],
       "Needs a Stable Diffusion model; reloads the model."),
    _p("controlnet_method", "ControlNet model", "select", SD_CONTROLNETS[0], "controlnet", ["diffusion"],
       "Pick one that matches the base model (SD 1.5, SD 2 or SDXL).", options=ALL_CONTROLNETS),
    _p("controlnet_conditioning_scale", "ControlNet strength", "float", 0.4, "controlnet", ["diffusion"], "",
       min=0.0, max=1.0, step=0.05),

    _p("enable_brushnet", "Use BrushNet", "bool", False, "brushnet", ["diffusion"],
       "Works with regular (non-inpainting) Stable Diffusion models. Turns ControlNet off."),
    _p("brushnet_method", "BrushNet model", "select", SD_BRUSHNETS[0], "brushnet", ["diffusion"], "",
       options=ALL_BRUSHNETS),
    _p("brushnet_conditioning_scale", "BrushNet strength", "float", 1.0, "brushnet", ["diffusion"], "",
       min=0.0, max=1.0, step=0.05),

    _p("enable_powerpaint_v2", "Use PowerPaint v2", "bool", False, "powerpaint", ["diffusion"],
       "For regular Stable Diffusion models."),
    _p("powerpaint_task", "Task", "select", "text-guided", "powerpaint", ["diffusion", "powerpaint"],
       "What PowerPaint should do with the mask.", options=POWERPAINT_TASKS),
    _p("fitting_degree", "Fitting degree", "float", 1.0, "powerpaint", ["diffusion", "powerpaint"],
       "How closely generated objects follow the mask shape (0.1 - 1).", min=0.05, max=1.0, step=0.05),
]

# Fields of IOPaint's InpaintRequest that Anima Studio fills itself (the picture and the mask).
RESERVED_FIELDS = ("image", "mask")

PARAM_KEYS = tuple(p["key"] for p in PARAMS)


def default_params() -> dict:
    return {p["key"]: p["default"] for p in PARAMS}


def _all_models() -> list[dict]:
    return ERASE_MODELS + DIFFUSION_MODELS


def model_info(model_id: str) -> dict:
    """The catalogue entry for a model id; unknown ids (custom repos, local files) count as prompt-driven diffusion."""
    if model_id in ("cv2", "telea", "ns"):
        return {"id": "cv2", "label": "OpenCV", "tags": ["classic"]}
    for m in _all_models():
        if m["id"] == model_id:
            return m
    return {"id": model_id, "label": model_id, "family": "custom", "tags": list(CUSTOM_MODEL_TAGS)}


def is_erase_model(model_id: str) -> bool:
    return "erase" in model_info(model_id)["tags"]


def controlnets_for(family: str) -> list[str]:
    return {"sd2": SD2_CONTROLNETS, "sdxl": SDXL_CONTROLNETS}.get(family, SD_CONTROLNETS)


def public_catalog() -> dict:
    """Everything the browser needs to draw the AI controls."""
    return {
        "iopaint_version": IOPAINT_VERSION,
        "erase_models": ERASE_MODELS,
        "diffusion_models": DIFFUSION_MODELS,
        "sam_models": SAM_MODELS,
        "remove_bg_models": REMOVE_BG_MODELS,
        "realesrgan_models": REALESRGAN_MODELS,
        "face_models": FACE_MODELS,
        "engine_options": ENGINE_OPTIONS,
        "custom_model_tags": CUSTOM_MODEL_TAGS,
        "packs": [{"id": p["id"], "label": p["label"], "count": len(p["items"])} for p in MODEL_PACKS],
        "pick_groups": pick_groups(),
        "groups": [{"id": g, "title": t} for g, t in GROUPS],
        "params": PARAMS,
        "devices": DEVICES,
        "defaults": default_params(),
        "custom_model_tags": CUSTOM_MODEL_TAGS,
    }
