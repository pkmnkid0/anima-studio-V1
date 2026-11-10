"""Anima Studio's Train tab -> Low-VRAM trainer config files.

KohakuMossTrainer (kohaku_backend/, vendored unmodified) is driven by `kogine run <script> --config <file>`, where a
config is a Python file of module-level UPPER_CASE knobs that `use_config()` the preset layers under
kohaku_backend/configs/moss/ and then overrides knobs. This module turns the Train tab's `kohaku_args` (plus the
dataset folders and the output folder) into two such files - the train config and the data-preparation config
(`kohakumoss-config new train` writes the same pair) - and returns the commands to run. Stdlib only, so it is tested
without torch and runs in any interpreter.

Every value goes through `repr()` of a coerced Python literal, and the "extra knobs" box only accepts literals
(`ast.literal_eval`), so nothing a user types can become code in the generated file.
"""
from __future__ import annotations

import ast
import os
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .kohaku_env import KOHAKU_DIR, USER_CONFIGS

P = "kohaku_"                                   # prefix of every key of the Train tab's kohaku_args

# VRAM layer file stems under configs/moss/vram/ (the Train tab's "vram" choice)
VRAM_LAYERS = {"6": "vram_6g", "8": "vram_8g", "12": "vram_12g", "16": "vram_16g", "24": "vram_24g",
               "auto": "vram_auto", "24-full": "vram_24g_full"}
ALGO_LAYERS = {"lora": "algo_lora", "loha": "algo_loha", "lokr": "algo_lokr", "oft": "algo_oft",
               "dora": "algo_dora", "full": "algo_full"}
ALGOS = ("lora", "locon", "loha", "lokr", "oft", "boft", "glora", "ia3", "dylora", "tlora", "dora", "full")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class KohakuConfigError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- coercion

def _b(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _s(v) -> str:
    return "" if v is None else str(v).strip().strip('"')


def _i(v, name: str) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        raise KohakuConfigError(f"{name}: '{v}' is not a whole number.") from None


def _f(v, name: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise KohakuConfigError(f"{name}: '{v}' is not a number.") from None


def _json_dict(v, name: str) -> dict:
    import json
    text = _s(v)
    if not text:
        return {}
    try:
        out = json.loads(text) if isinstance(v, str) else v
    except ValueError as exc:
        raise KohakuConfigError(f"{name}: not valid JSON ({exc}).") from None
    if not isinstance(out, dict):
        raise KohakuConfigError(f"{name}: must be a JSON object like {{\"factor\": 8}}.")
    return out


def literal(v) -> str:
    """A Python literal for the config file (repr of plain data only)."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return repr(v)
    if isinstance(v, (list, tuple)):
        inner = ", ".join(literal(x) for x in v)
        return f"({inner},)" if isinstance(v, tuple) and len(v) == 1 else (f"({inner})" if isinstance(v, tuple) else f"[{inner}]")
    if isinstance(v, dict):
        return "{" + ", ".join(f"{literal(k)}: {literal(x)}" for k, x in v.items()) + "}"
    raise KohakuConfigError(f"Cannot write {type(v).__name__} into a config.")


def posix(path: str | Path) -> str:
    return Path(str(path).strip().strip('"')).expanduser().as_posix()


def parse_extra_knobs(text: str) -> dict:
    """'KEY = value' lines (Python literals only) -> {KEY: value}; the long tail of Kohaku's knobs."""
    out: dict = {}
    for n, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise KohakuConfigError(f"Extra knobs, line {n}: write KNOB_NAME = value (knob names are UPPER_CASE).")
        try:
            out[key] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            raise KohakuConfigError(f"Extra knobs, line {n}: '{value.strip()}' is not a Python value "
                                    "(put text in quotes, e.g. \"bf16\").") from None
    return out


# ------------------------------------------------------------------------------------- layer helpers

def layer_path(group: str, stem: str, root: Path = KOHAKU_DIR) -> Path:
    path = root / "configs" / "moss" / group / f"{stem}.py"
    if not path.is_file():
        raise KohakuConfigError(f"Low-VRAM trainer preset '{stem}' is missing from {path.parent} - the bundled copy is incomplete.")
    return path


def layer_knobs(path: Path) -> dict:
    """The literal module-level assignments of a layer file (what use_config would apply), via ast."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return out


def merged_dicts(layers: list[Path], names: tuple[str, ...] = ("MEMORY_OVERRIDES",)) -> dict:
    """`use_config` replaces whole values, so a dict knob that several layers set is merged here, in layer order
    (what `kohakumoss-config new` does for MEMORY_OVERRIDES)."""
    out = {}
    for name in names:
        found = [k[name] for k in map(layer_knobs, layers) if name in k]
        if len(found) > 1:
            out[name] = {key: v for part in found for key, v in part.items()}
    return out


# ------------------------------------------------------------------------------------------- weights

def _under(weights: Path, sub: str, value: str, defaults: tuple[str, ...]) -> Path | None:
    if value:
        for cand in (Path(value), weights / sub / value, weights / sub / (value + ".safetensors")):
            if cand.exists():
                return cand
        return None
    for d in defaults:
        cand = weights / sub / (d + ".safetensors")
        if cand.exists():
            return cand
    return None


def inspect_weights(weights: str, variant: str = "base", transformer: str = "", text_encoder: str = "",
                    vae: str = "") -> dict:
    """What the weights folder holds, with the same default-file rules as Low-VRAM trainer (docs/guides/weights.md):
    returns {"layout", "te", "vae", "dit", "problems"}; te / vae are what the cache encoders get as `path`."""
    problems: list[str] = []
    if not weights:
        return {"layout": "", "te": "", "vae": "", "dit": "", "problems": ["Set the Low-VRAM trainer weights folder."]}
    w = Path(weights).expanduser()
    if not w.exists():
        return {"layout": "", "te": weights, "vae": weights, "dit": "",
                "problems": [f"The weights path does not exist: {weights}"]}
    if w.is_file():
        # a single DiT file: the text encoder and the VAE must be named
        te = _under(w.parent, "text_encoders", text_encoder, ()) or (Path(text_encoder) if text_encoder else None)
        va = _under(w.parent, "vae", vae, ()) or (Path(vae) if vae else None)
        if not (te and Path(te).exists()):
            problems.append("With a single DiT file as the weights path, set the Text encoder (file or folder).")
        if not (va and Path(va).exists()):
            problems.append("With a single DiT file as the weights path, set the VAE (file).")
        return {"layout": "file", "te": str(te or ""), "vae": str(va or ""), "dit": str(w), "problems": problems}

    if (w / "diffusion_models").is_dir() or (w / "text_encoders").is_dir():
        layout = "comfy"
        dit = _under(w, "diffusion_models", transformer, (f"moss_image_{variant}_int8_convrot", f"moss_image_{variant}_bf16"))
        te = _under(w, "text_encoders", text_encoder, ("qwen3.5_9b_int8_convrot", "qwen3.5_9b_bf16", "qwen3.5_9b_w4a8_gptq"))
        va = _under(w, "vae", vae, ("flux2-vae_small", "flux2-vae"))
        for label, found, value, sub in (("DiT", dit, transformer, "diffusion_models"), ("Text encoder", te, text_encoder, "text_encoders"),
                                         ("VAE", va, vae, "vae")):
            if found is None:
                problems.append(f"{label}: '{value}' was not found in {w / sub}." if value
                                else f"{label}: no default file found in {w / sub} (see the weights guide for the file names).")
        return {"layout": layout, "te": str(te or w), "vae": str(va or w), "dit": str(dit or ""), "problems": problems}

    root = w / "base" if (w / "base").is_dir() and not (w / "transformer").is_dir() else w
    if (root / "transformer").is_dir():
        te = _under(root, "text_encoder", text_encoder, ()) if text_encoder else None
        va = _under(root, "vae", vae, ()) if vae else None
        if text_encoder and te is None:
            problems.append(f"Text encoder: '{text_encoder}' was not found.")
        if vae and va is None:
            problems.append(f"VAE: '{vae}' was not found.")
        for sub in ("text_encoder", "vae"):
            if not (root / sub).is_dir() and not (text_encoder if sub == "text_encoder" else vae):
                problems.append(f"The pipeline folder has no {sub}/ folder.")
        return {"layout": "hf", "te": str(te or root), "vae": str(va or root), "dit": str(root / "transformer"),
                "problems": problems}

    return {"layout": "", "te": str(w), "vae": str(w), "dit": "",
            "problems": [f"{w} is neither a ComfyUI models folder (diffusion_models/, text_encoders/, vae/) nor a "
                         "MOSS-Image pipeline folder (transformer/, text_encoder/, vae/)."]}


# ---------------------------------------------------------------------------------------------- build

@dataclass
class Built:
    name: str
    train_path: Path
    data_path: Path
    train_text: str
    data_text: str
    run_dir: Path
    steps: int
    cache_passes: bool
    inspect: bool
    notes: list[str] = field(default_factory=list)
    knobs: dict = field(default_factory=dict)


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-._")
    return s[:60] or "anima-studio-lora"


def steps_per_epoch(items: int, batching: str, side: int, max_tokens: int, batch_size: int, grad_acc: int) -> float:
    """Optimizer steps in one pass over `items` samples (images x repeats). Token batching packs about
    max_tokens / (tokens of one image + caption) images into a batch; count batching uses the batch size."""
    if batching == "count":
        per_batch = max(1, batch_size)
    else:
        per_batch = max(1, max_tokens // int((side * side) // 256 + 200))
    return max(1.0, items / (per_batch * max(1, grad_acc)))


def _opt_str(own: dict, knob: str, value) -> None:
    v = _s(value)
    if v:
        own[knob] = v


def _group(a: dict, key: str, default=None):
    return a.get(P + key, default)


def build(kohaku_args: dict, subsets: list[dict], output_dir: str, output_name: str = "",
          root: Path = KOHAKU_DIR, user_dir: Path | None = None) -> Built:
    """The train + data configs for one run. Raises KohakuConfigError with a readable message."""
    a = dict(kohaku_args or {})
    g = lambda key, default=None: _group(a, key, default)          # noqa: E731
    user_dir = user_dir or (root / "configs" / "user")
    notes: list[str] = []

    weights = posix(_s(g("weights"))) if _s(g("weights")) else ""
    if not weights:
        raise KohakuConfigError("Set the Low-VRAM trainer weights folder on the Train tab (Model).")
    variant = _s(g("variant", "base")) or "base"
    info = inspect_weights(weights, variant, _s(g("transformer")), _s(g("text_encoder")), _s(g("vae")))
    if info["problems"]:
        raise KohakuConfigError(" ".join(info["problems"]))

    if not subsets:
        raise KohakuConfigError("Add at least one dataset folder.")
    if not _s(output_dir):
        raise KohakuConfigError("Set the output folder (Output tab).")
    name = slug(output_name)

    # ---- layers ----------------------------------------------------------------------------------
    algo = _s(g("algo", "lora")) or "lora"
    if algo not in ALGOS:
        raise KohakuConfigError(f"Unknown algorithm '{algo}'.")
    full = algo == "full"
    vram = _s(g("vram", "8")) or "8"
    if full:
        vram = "24-full" if vram in ("24", "24-full") else "auto"
    if vram not in VRAM_LAYERS:
        raise KohakuConfigError(f"Unknown VRAM profile '{vram}'.")
    precision = _s(g("precision", "bf16")) or "bf16"
    if precision == "auto":
        precision = "bf16"
    if precision not in ("bf16", "fp16"):
        raise KohakuConfigError("Precision must be bf16, fp16 or auto.")
    layers = [
        root / "configs" / "moss" / "_base" / ("base_full.py" if full else "base_lora.py"),
        layer_path("algo", ALGO_LAYERS.get(algo, "algo_lora"), root),
        layer_path("vram", VRAM_LAYERS[vram], root),
        layer_path("precision", f"precision_{precision}", root),
    ]
    if not layers[0].is_file():
        raise KohakuConfigError(f"{layers[0]} is missing - the bundled Low-VRAM trainer copy is incomplete.")

    # ---- knobs (everything the Train tab sets wins over the layers) ------------------------------
    own: dict = {"WEIGHTS": weights, "RUN_NAME": name, "OUTPUT_DIR": posix(output_dir)}
    if variant != "base":
        own["VARIANT"] = variant
    for key, knob in (("transformer", "TRANSFORMER"), ("text_encoder", "TEXT_ENCODER"), ("vae", "VAE"),
                      ("text_device", "TEXT_DEVICE"), ("vae_dtype", "VAE_DTYPE")):
        _opt_str(own, knob, g(key))
    if _b(g("train_text_refiner", False)):
        own["TRAIN_TEXT_REFINER"] = True

    # -- data
    mode = _s(g("caption_mode", "tags")) or "tags"
    if mode not in ("tags", "nl"):
        raise KohakuConfigError("Caption style must be tags or natural language.")
    datasets, total_items = [], 0
    for i, s in enumerate(subsets):
        d = _s(s.get("image_dir"))
        if not d or not Path(d).is_dir():
            raise KohakuConfigError(f"Dataset folder {i + 1} is missing or not a folder: {d or '(empty)'}")
        imgs = sum(1 for f in Path(d).iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT)
        if imgs == 0:
            raise KohakuConfigError(f"No images found in {d}.")
        rep = max(1, _i(s.get("num_repeats", 1), "Repeats"))
        total_items += imgs * rep
        datasets.append({"source": "folder", "path": posix(d), "txt_as": mode, "recursive": False, "weight": float(rep)})
        ext = _s(s.get("caption_extension", ".txt")).lower()
        if ext not in ("", ".txt", ".caption"):
            notes.append(f"Folder {i + 1}: caption extension '{ext}' is not read by Low-VRAM trainer (it reads .txt, .caption and .json).")
    own["DATASETS"] = datasets
    if mode == "tags":
        cap = {"template": "tags", "keep": [], "shuffle": _b(g("caption_shuffle", True)),
               "tag_dropout": _f(g("caption_tag_dropout", 0.0), "Tag dropout"),
               "underscores": _b(g("caption_underscores", True)), "unescape": _b(g("caption_unescape", True))}
    else:
        cap = {"template": "nl"}
    own["CAPTION"] = cap
    own["TEXT_DROP"] = _f(g("text_drop", 0.1), "Caption dropout")
    res_mode = _s(g("resolution_mode", "resize"))
    if res_mode:
        own["RESOLUTION_MODE"] = res_mode
    side = _i(g("resolution", 1024), "Resolution")
    if side < 256:
        raise KohakuConfigError("Resolution must be at least 256.")
    own["TARGET_AREA"] = side * side
    own["SIZE_STEP"] = _i(g("size_step", 16), "Size step")
    own["ASPECT_RANGE"] = (_f(g("aspect_min", 0.33), "Aspect min"), _f(g("aspect_max", 3.0), "Aspect max"))
    for key, knob in (("upscale", "UPSCALE"), ("batching", "BATCHING"), ("overlong", "OVERLONG"),
                      ("token_budget", "TOKEN_BUDGET"), ("crop", "CROP"), ("cache_dtype", "CACHE_DTYPE"),
                      ("latent_cache", "LATENT_CACHE"), ("text_cache", "TEXT_CACHE")):
        _opt_str(own, knob, g(key))
    if g("max_tokens") not in (None, "", 0, "0"):
        own["MAX_TOKENS"] = _i(g("max_tokens"), "Max tokens")
    for key, knob, default in (("batch_size", "BATCH_SIZE", 4), ("max_samples", "MAX_SAMPLES", 0),
                               ("caption_variants", "CAPTION_VARIANTS", 4), ("num_workers", "NUM_WORKERS", 2),
                               ("seed", "SEED", 0)):
        if g(key) not in (None, ""):
            own[knob] = _i(g(key), key)
    if _b(g("flip", False)):
        own["FLIP"] = True
    cache_dir = _s(g("cache_dir")) or str(Path(posix(output_dir)) / "kohaku_cache")
    own["CACHE_DIR"] = posix(cache_dir)

    # -- text encoder / VAE placement (the layer's dict, edited in the UI)
    te_plan = {}
    for key, knob, conv in (("te_memory", "memory", str), ("te_ahead", "ahead", int), ("te_batch", "batch", int),
                            ("te_max_tokens", "max_tokens", int), ("te_precision", "precision", str),
                            ("te_stream_resident", "stream_resident", int)):
        v = g(key)
        if v not in (None, ""):
            te_plan[knob] = conv(v) if conv is str else _i(v, key)
    if te_plan:
        own["TEXT_ENCODE"] = te_plan
    vae_plan = {}
    for key, knob, conv in (("vae_memory", "memory", str), ("vae_ahead", "ahead", int), ("vae_batch", "batch", int)):
        v = g(key)
        if v not in (None, ""):
            vae_plan[knob] = conv(v) if conv is str else _i(v, key)
    if g("vae_tile") not in (None, ""):
        vae_plan["tile"] = _b(g("vae_tile"))
    if vae_plan:
        own["VAE_ENCODE"] = vae_plan

    # -- adapter
    own["ALGO"] = "lora" if algo == "dora" else algo
    if not full:
        if _s(g("preset")):
            own["PRESET"] = _s(g("preset"))
        own["DIM"] = _i(g("dim", 32), "Rank")
        own["ALPHA"] = _f(g("alpha", 32), "Alpha")
        kwargs = _json_dict(g("algo_kwargs"), "Algorithm arguments")
        if kwargs:
            own["ALGO_KWARGS"] = kwargs
        if algo == "dora" or _b(g("dora", False)):
            own["DORA"] = True
            if g("dora_detach") not in (None, ""):
                own["DORA_DETACH"] = _b(g("dora_detach"))
        own["DROPOUT"] = {"input": _f(g("dropout_input", 0.0), "Input dropout"), "rank": _f(g("dropout_rank", 0.0), "Rank dropout"),
                          "module": _f(g("dropout_module", 0.0), "Module dropout")}
        _opt_str(own, "ADAPTER_MODE", g("adapter_mode"))
        if g("loraplus_ratio") not in (None, ""):
            own["LORAPLUS_RATIO"] = _f(g("loraplus_ratio"), "LoRA+ ratio")
        if _b(g("train_norms", False)):
            own["TRAIN_NORMS"] = True


    # -- objective
    ts = _s(g("timestep", "moss")) or "moss"
    own["TIMESTEP"] = ts
    if ts == "moss":
        own["TIMESTEP_KWARGS"] = {"mu": _f(g("ts_mu", -1.09), "mu"), "s": _f(g("ts_s", 0.95), "s"),
                                  "shift": _f(g("ts_shift", 1.0), "shift"), "cutoff": _f(g("ts_cutoff", 0.95), "cutoff"),
                                  "uniform_mix": _f(g("ts_uniform_mix", 0.01), "uniform mix")}
    for key, knob in (("loss", "LOSS"), ("weighting", "WEIGHTING")):
        _opt_str(own, knob, g(key))
    if g("hard_weight") not in (None, ""):
        own["HARD_WEIGHT"] = _f(g("hard_weight"), "Data loss weight")

    # -- optimiser
    for key, knob in (("optimizer", "OPTIMIZER"), ("wd_rules", "WD_RULES"), ("step_sync", "STEP_SYNC")):
        _opt_str(own, knob, g(key))
    own["LR"] = _f(g("lr", 1e-4), "Learning rate")
    own["BETAS"] = (_f(g("beta1", 0.9), "Beta 1"), _f(g("beta2", 0.95), "Beta 2"))
    if g("eps") not in (None, ""):
        own["EPS"] = _f(g("eps"), "Epsilon")
    own["WEIGHT_DECAY"] = _f(g("weight_decay", 0.01), "Weight decay")
    own["GRAD_CLIP"] = _f(g("grad_clip", 1.0), "Gradient clip")
    sched_mode = _s(g("schedule_mode", "cosine")) or "cosine"
    sched = {"mode": sched_mode, "warmup": _i(g("warmup", 100), "Warmup"), "end": -1}
    if sched_mode != "constant":
        sched["min_value"] = _f(g("min_value", 0.1), "Final LR fraction")
    own["SCHEDULE"] = sched
    okw = _json_dict(g("optimizer_kwargs"), "Optimizer arguments")
    if okw:
        own["OPTIMIZER_KWARGS"] = okw
    if g("ema") not in (None, ""):
        own["EMA"] = _f(g("ema"), "EMA")

    # -- memory
    profile = _s(g("memory_profile"))
    if profile:
        own["MEMORY_PROFILE"] = profile
    if profile == "manual":
        for key, knob in (("checkpoint", "CHECKPOINT"), ("stream", "STREAM"), ("optim_state", "OPTIM_STATE")):
            _opt_str(own, knob, g(key))
        if g("stream_resident") not in (None, ""):
            own["STREAM_RESIDENT"] = _i(g("stream_resident"), "Resident blocks")
        for key, knob in (("pinned", "PINNED"), ("act_offload", "ACT_OFFLOAD"), ("grad_offload", "GRAD_OFFLOAD"),
                          ("fused_step", "FUSED_STEP")):
            if g(key) not in (None, ""):
                own[knob] = _b(g(key))
    if _s(g("base_precision")):
        own["BASE_PRECISION"] = _s(g("base_precision"))
    for key, knob in (("fp8_recipe", "FP8_RECIPE"), ("fp8_dgrad", "FP8_DGRAD"), ("int8_dgrad", "INT8_DGRAD")):
        _opt_str(own, knob, g(key))
    if _b(g("int8_compute", False)):
        own["INT8_COMPUTE"] = True
    sr = _s(g("step_release"))
    if sr:
        own["STEP_RELEASE"] = int(sr) if sr.isdigit() else sr
    _opt_str(own, "TUNE", g("tune"))
    _opt_str(own, "COMPILE", g("compile"))
    if precision == "fp16" and _s(g("fp16_fix")):
        own["FP16_FIX"] = _s(g("fp16_fix"))

    # -- runtime
    grad_acc = max(1, _i(g("grad_acc", 1), "Gradient accumulation"))
    save_every_steps = None
    if _s(g("duration", "steps")) == "epochs":
        # The engine only counts optimizer steps, so an epoch count is turned into steps from the dataset size.
        epochs = _f(g("epochs", 10), "Epochs")
        if epochs <= 0:
            raise KohakuConfigError("Epochs must be more than 0.")
        per_epoch = steps_per_epoch(total_items, _s(g("batching", "tokens")) or "tokens", side,
                                    _i(g("max_tokens"), "Max tokens") if g("max_tokens") not in (None, "", 0, "0") else 4352,
                                    _i(g("batch_size", 4), "Batch size"), grad_acc)
        steps = max(1, math.ceil(epochs * per_epoch))
        every = _f(g("save_every_epochs", 1), "Save every (epochs)")
        save_every_steps = max(0, round(every * per_epoch)) if every > 0 else 0
        notes.append(f"{epochs:g} epochs = {steps} steps ({total_items} images x repeats, about {per_epoch:.1f} steps per epoch). "
                     "Batches are packed by token budget, so the real epoch length varies a little with image shapes.")
    else:
        steps = _i(g("steps", 2000), "Steps")
    if steps < 1:
        raise KohakuConfigError("Steps must be at least 1.")
    own["STEPS"] = steps
    own["GRAD_ACC"] = grad_acc
    own["LOG_EVERY"] = max(1, _i(g("log_every", 5), "Log every"))
    if g("metrics_every") not in (None, "", 0, "0"):
        own["METRICS_EVERY"] = _i(g("metrics_every"), "Metrics every")
    for key, knob in (("logger", "LOGGER"), ("metrics_level", "METRICS_LEVEL")):
        _opt_str(own, knob, g(key))

    # -- checkpoints / export
    own["SAVE_EVERY"] = save_every_steps if save_every_steps is not None else max(0, _i(g("save_every", 500), "Save every"))
    if g("save_minutes") not in (None, ""):
        own["SAVE_MINUTES"] = max(0, _i(g("save_minutes"), "Save every N minutes"))
    own["KEEP_LAST"] = max(0, _i(g("keep_last", 3), "Keep last"))
    fc = _s(g("final_checkpoint"))
    if fc:
        own["FINAL_CHECKPOINT"] = fc if fc == "auto" else _b(fc)
    resume = _s(g("resume"))
    if resume:
        own["RESUME"] = resume
    _opt_str(own, "INIT_FROM", g("init_from"))
    own["SAVE_FORMAT"] = _s(g("save_format", "comfy")) or "comfy"
    if _b(g("save_merged", False)):
        own["SAVE_MERGED"] = True
    fq = _s(g("save_fused_qkv"))
    if fq:
        own["SAVE_FUSED_QKV"] = "auto" if fq == "auto" else _b(fq)
    if g("checkpoint_export") not in (None, ""):
        own["CHECKPOINT_EXPORT"] = _b(g("checkpoint_export"))

    # -- sampling
    every = max(0, _i(g("sample_every", 0), "Sample every"))
    own["SAMPLE_EVERY"] = every
    prompts = [l.strip() for l in str(g("sample_prompts") or "").splitlines() if l.strip()]
    if prompts:
        own["SAMPLE_PROMPTS"] = prompts
    if every and not prompts:
        raise KohakuConfigError("Sampling is on but there are no sample prompts - add one per line or set Sample every to 0.")
    if every:
        _opt_str(own, "SAMPLE_NEGATIVE", g("sample_negative"))
        own["SAMPLE_STEPS"] = _i(g("sample_steps", 50), "Sample steps")
        own["SAMPLE_CFG"] = _f(g("sample_cfg", 3.0), "Sample CFG")
        own["SAMPLE_SHIFT"] = _f(g("sample_shift", 5.0), "Sample shift")
        own["SAMPLE_SIZE"] = (_i(g("sample_width", 1024), "Sample width"), _i(g("sample_height", 1024), "Sample height"))
        seeds = [_i(x, "Sample seeds") for x in re.split(r"[,\s]+", _s(g("sample_seeds", "0"))) if x] or [0]
        own["SAMPLE_SEEDS"] = tuple(seeds)
        skip = [_i(x, "Sample skip layers") for x in re.split(r"[,\s]+", _s(g("sample_skip_layers", "22"))) if x]
        own["SAMPLE_SKIP_LAYERS"] = tuple(skip)
        if g("sample_batch") not in (None, ""):
            own["SAMPLE_BATCH"] = _i(g("sample_batch"), "Sample batch")
        if g("sample_columns") not in (None, ""):
            own["SAMPLE_COLUMNS"] = _i(g("sample_columns"), "Sample columns")
        if _b(g("sample_at_start", False)):
            own["SAMPLE_AT_START"] = True

    # -- the long tail, last so it can override anything above
    own.update(parse_extra_knobs(str(g("extra_knobs") or "")))

    # a dict knob several layers set is merged here (use_config replaces whole values)
    for knob, value in merged_dicts(layers).items():
        if knob not in own:
            own[knob] = value

    # ---- render --------------------------------------------------------------------------------
    base = user_dir.resolve()
    train_path = user_dir / f"studio_{name}.py"
    data_path = user_dir / f"studio_{name}_data.py"

    def rel(target: Path) -> str:
        try:
            return Path(os.path.relpath(target.resolve(), base)).as_posix()
        except ValueError:
            return target.resolve().as_posix()

    head = [f'"""Anima Studio run "{name}": drives scripts/train/moss.py. Written by Anima Studio on every run - edit the',
            'Train tab, not this file."""', "", "from kohakuengine import use_config", ""]
    body = [f'use_config("{rel(p)}")' for p in layers] + ["", "# knobs below win over the layers above"]
    body += [f"{k} = {literal(v)}" for k, v in own.items()]
    train_text = "\n".join(head + body) + "\n"

    te_path = info["te"] or weights
    vae_path = info["vae"] or weights
    data_text = "\n".join([
        f'"""Anima Studio run "{name}": the train config\'s data knobs plus the cache encoders. Drives',
        'scripts/data/inspect_data.py, scripts/data/cache_text.py and scripts/data/cache_latents.py."""', "",
        "from kohakuengine import use_config", "", f'use_config("{train_path.name}")', "",
        'TEXT_ENCODER = "kohakumoss.encoders.text_encoder"',
        f'TEXT_ENCODER_KWARGS = {literal({"path": posix(te_path), "device": "cuda"})}',
        'IMAGE_ENCODER = "kohakumoss.encoders.vae_encoder"',
        f'IMAGE_ENCODER_KWARGS = {literal({"path": posix(vae_path), "device": "cuda"})}',
    ]) + "\n"

    caches_needed = own.get("LATENT_CACHE", "offline") == "offline" or own.get("TEXT_CACHE", "offline") == "offline"
    return Built(name=name, train_path=train_path, data_path=data_path, train_text=train_text, data_text=data_text,
                 run_dir=Path(posix(output_dir)) / name, steps=steps, cache_passes=caches_needed,
                 inspect=_b(g("run_inspect", True)), notes=notes, knobs=own)


def write(built: Built) -> None:
    built.train_path.parent.mkdir(parents=True, exist_ok=True)
    built.train_path.write_text(built.train_text, encoding="utf-8")
    built.data_path.write_text(built.data_text, encoding="utf-8")


def passes(built: Built) -> list[tuple[str, str, str]]:
    """The ordered kogine jobs: (label, script, config file name)."""
    out = []
    if built.inspect:
        out.append(("Checking the data plan", "scripts/data/inspect_data.py", built.data_path.name))
    if built.cache_passes:
        k = built.knobs
        if k.get("TEXT_CACHE", "offline") == "offline":
            out.append(("Caching caption embeddings (text encoder)", "scripts/data/cache_text.py", built.data_path.name))
        if k.get("LATENT_CACHE", "offline") == "offline":
            out.append(("Caching image latents (VAE)", "scripts/data/cache_latents.py", built.data_path.name))
    out.append(("Training", "scripts/train/moss.py", built.train_path.name))
    return out
