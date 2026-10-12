"""
Anima LoRA training for Anima Studio's native backend.

Unlike sdxl_trainer.py (a genuinely new training loop built on official
diffusers/peft primitives), this deliberately does NOT reimplement
Anima's training loop from scratch. Anima is an obscure, very new
architecture - a Diffusion Transformer with a Qwen3 LLM text encoder
bridged through a custom adapter, trained via rectified flow - and
mainstream libraries have no support for it. The vendored backend at
training_backend/sd_scripts/ does (anima_train_network.py +
library/anima_models.py, Apache-2.0), and re-deriving that from
fragments read out of context would risk getting a subtle, unverifiable
detail wrong. What's actually new here: translating Anima Studio's own
config shape into what that real trainer expects, and invoking it
in-process (a direct Python call) instead of as a subprocess.

Concretely: the real training entry point
(anima_train_network.py's `if __name__ == "__main__"` block) builds its
`args` by parsing real CLI defaults, then does a couple of small fixups,
then calls `AnimaNetworkTrainer().train(args)`. This mirrors that exact
sequence - args come from the real parser's own defaults
(`setup_parser().parse_args([])`), not a hand-built Namespace guessing
at field names, so anything this code doesn't explicitly override still
gets a real, tested default rather than a missing-attribute crash deep
inside the trainer.

Dataset handoff: the base trainer only reads datasets from a TOML file
path (`--dataset_config`), verified against its actual schema in
library/config_util.py (top-level "datasets" list of DreamBooth-style
subsets, "image_dir" the only required field) - so Anima Studio's
dataset dict is written out as a real TOML file here, not passed
in-memory, since there's no in-memory entry point to hook into.

Honest limitation: this has NOT been run against real Anima model
weights (Qwen3 + the DiT checkpoint + the Qwen-Image VAE) - those are
large, separately-hosted downloads outside what could be fetched and
verified in the environment this was built in. What IS verified:
argument translation produces a valid, schema-correct config (tested
against the real parser and the real TOML schema), and the import path
into the vendored trainer resolves correctly.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from . import dataset as ds

_SD_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "training_backend" / "sd_scripts"


class TrainerError(RuntimeError):
    pass


def _get(d: dict, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_sd_scripts_importable() -> None:
    if not _SD_SCRIPTS_DIR.is_dir():
        raise TrainerError(
            f"Anima training needs the bundled backend's sd_scripts folder, not found at {_SD_SCRIPTS_DIR}. "
            "Run \"Set up backend\" in the Train tab first."
        )
    if str(_SD_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SD_SCRIPTS_DIR))


_SUBSET_KEYS = (
    "image_dir", "num_repeats", "caption_extension", "color_aug", "gamma_aug", "gamma_aug_range", "gamma_aug_rate",
    "face_crop_aug_range", "flip_aug", "random_crop", "random_crop_padding_percent", "shuffle_caption", "keep_tokens",
    "keep_tokens_separator", "secondary_separator", "caption_separator", "enable_wildcard", "token_warmup_min",
    "token_warmup_step", "caption_prefix", "caption_suffix", "resize_interpolation", "caption_dropout_every_n_epochs",
    "caption_dropout_rate", "caption_tag_dropout_rate", "protected_tags_file", "is_reg", "is_val", "alpha_mask",
    "class_tokens", "cache_info",
)
_DATASET_KEYS = {"validation_seed", "validation_split", "network_multiplier", "resize_interpolation",
                 "skip_image_resolution", "resolution_jitter_resolutions", "resolution_jitter_batch_sizes",
                 "resolution_jitter_weights", "multires_training"}


def write_dataset_toml(dataset: dict, resolution, dest_path: Path, batch_size: int = 1) -> Path:
    """Anima Studio's dataset dict -> a real kohya-style TOML file, the
    only format the vendored trainer's dataset loader actually reads
    (see library/config_util.py's ConfigSanitizer - DB_SUBSET_DISTINCT_SCHEMA
    requires image_dir; everything else here is optional with the
    trainer's own defaults). `resolution` may be a scalar int or a
    [width, height] pair - the sanitizer accepts both."""
    import toml  # ships with the vendored backend's own venv

    subsets = dataset.get("subsets") or []
    if not subsets:
        raise TrainerError("No subsets in dataset config.")

    toml_subsets = []
    for s in subsets:
        image_dir = s.get("image_dir")
        if not image_dir or not Path(image_dir).is_dir():
            raise TrainerError(f"Subset image_dir is missing or not a folder: {image_dir!r}")
        _ext = s.get("caption_extension") or ".txt"
        if not _ext.startswith("."):
            _ext = "." + _ext
        import os as _os
        _imgs = [f for f in _os.listdir(image_dir) if _os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")]
        _have = sum(1 for f in _imgs if _os.path.isfile(_os.path.join(image_dir, _os.path.splitext(f)[0] + _ext))
                    and _os.path.getsize(_os.path.join(image_dir, _os.path.splitext(f)[0] + _ext)) > 0)
        print(f"[anima-studio] captions: {_have}/{len(_imgs)} images in {image_dir} have a '{_ext}' file", flush=True)
        if _imgs and _have == 0:
            print(f"[anima-studio] WARNING: no captions found in {image_dir} - training on EMPTY captions "
                  "will produce a LoRA with almost no effect.", flush=True)
        entry = {
            "image_dir": str(image_dir),
            "num_repeats": int(s.get("num_repeats", 1)),
            # The vendored trainer's own default is ".caption", not ".txt" -
            # Anima Studio's own dataset convention (and every other tab in
            # this app) writes ".txt" captions, so this has to be explicit
            # or captions get silently dropped with no error (confirmed by
            # actually running this against the real dataset loader, not
            # just read from its source - see native_backend/NOTES.md).
            "caption_extension": s.get("caption_extension") or ".txt",
        }
        # Every other per-folder setting the UI offers (caption shuffle, keep tokens, flip/colour
        # aug, random crop, caption dropout, prefix/suffix, regularisation flag...). These used to
        # be dropped, so they had no effect on training.
        for key in _SUBSET_KEYS:
            if key in ("image_dir", "num_repeats", "caption_extension"):
                continue
            # Same rule as LoRA Easy Training Scripts: with Cache latents on, random crop and colour
            # augmentation are forced off (a cached latent can't be re-cropped; the trainer would
            # refuse to start otherwise). Masked loss also turns random crop off.
            if _CACHE_LATENTS_REQUESTED and key in ("random_crop", "color_aug", "gamma_aug", "gamma_aug_range", "gamma_aug_rate"):
                continue
            if _MASKED_LOSS_REQUESTED and key == "random_crop":
                continue
            val = s.get(key)
            if val is None or val == "" or val is False:
                continue
            entry[key] = val
        toml_subsets.append(entry)

    global _SUBSET_CAPTION_AUG, _SUBSET_PIXEL_AUG
    _SUBSET_CAPTION_AUG = any(e.get(k) for e in toml_subsets for k in (
        "shuffle_caption", "caption_dropout_rate", "caption_tag_dropout_rate", "caption_dropout_every_n_epochs",
        "token_warmup_step", "enable_wildcard"))
    _SUBSET_PIXEL_AUG = any(e.get(k) for e in toml_subsets for k in (
        "random_crop", "color_aug", "face_crop_aug_range", "gamma_aug"))
    ds_entry = {
        "resolution": resolution,
        "batch_size": max(1, int(batch_size)),
        "subsets": toml_subsets,
    }
    # Bucketing is the loader's default-OFF; without it any image bigger than the training
    # resolution aborts with "image too large, but cropping and bucketing are disabled".
    buckets = dict(dataset.get("bucket_args") or {})
    buckets.setdefault("enable_bucket", True)
    for key, typ in (("enable_bucket", bool), ("bucket_no_upscale", bool), ("min_bucket_reso", int),
                     ("max_bucket_reso", int), ("bucket_reso_steps", int), ("multires_training", bool)):
        if buckets.get(key) is not None:
            ds_entry[key] = typ(buckets[key])
    # dataset-level extras the user added (validation split, resize interpolation, ...)
    for src in (dataset.get("extra_args") or {}, ):
        for key, val in src.items():
            if key in _DATASET_KEYS and val not in (None, ""):
                ds_entry[key] = val
    doc = {"datasets": [ds_entry]}
    dest_path.write_text(toml.dumps(doc), encoding="utf-8")
    return dest_path


def _dataset_general(dataset: dict) -> dict:
    """Dataset-level settings (resolution, batch_size) live in the
    payload's dataset section - build_backend_payload puts them there,
    not in args.general_args."""
    return (dataset or {}).get("general_args") or {}


def _dataset_resolution(dataset: dict, args: dict):
    """The UI's resolution, as the vendored dataset schema wants it:
    a scalar int or a [width, height] pair (both are valid there).
    Falls back to args.general_args for hand-built payloads."""
    raw = _dataset_general(dataset).get("resolution") or _get(args, "general_args", "resolution")
    if raw in (None, ""):
        return 1024
    return raw


def _dataset_batch_size(dataset: dict) -> int:
    raw = _dataset_general(dataset).get("batch_size")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


_BARE_BUILTIN = ("SGD", "SGDNesterov", "AdamW", "AdamW8bit", "SGDNesterov8bit", "Lion", "Lion8bit", "PagedAdamW",
                 "PagedAdamW8bit", "PagedLion8bit", "PagedAdamW32bit", "Adafactor", "Prodigy", "DAdaptation",
                 "DAdaptAdamPreprint", "DAdaptAdaGrad", "DAdaptAdam", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion",
                 "DAdaptSGD", "AdamWScheduleFree", "RAdamScheduleFree", "SGDScheduleFree")


def resolve_optimizer_name(name: str) -> str:
    """Optimizer names are case-sensitive in the trainer: "Came" is looked up as
    torch.optim.Came and fails, while the real class is LoraEasyCustomOptimizer.CAME.
    Match the name case-insensitively against what is actually installed."""
    name = str(name).strip()
    if "." in name:
        return name
    for b in _BARE_BUILTIN:
        if b.lower() == name.lower():
            return b
    _ensure_sd_scripts_importable()
    import importlib
    for module in ("LoraEasyCustomOptimizer", "adv_optm.optim", "prodigyplus.prodigy_plus_schedulefree"):
        try:
            mod = importlib.import_module(module)
        except Exception:  # noqa: BLE001 - package not installed / import problem: try the next one
            continue
        for attr in dir(mod):
            if attr.lower() == name.lower() and callable(getattr(mod, attr, None)):
                return f"{module}.{attr}"
    return name


_LYCORIS_ONLY = {"algo", "preset", "conv_dim", "conv_alpha", "use_tucker", "train_norm", "rescaled", "constraint",
                 "dora_wd", "bypass_mode", "dropout", "unit", "down_lr_weight", "mid_lr_weight", "up_lr_weight",
                 "block_dims", "block_alphas", "conv_block_dims", "conv_block_alphas"}


def _apply_ui_overrides(ns, args: dict, dataset_item_count: int, dataset_batch_size: int) -> None:
    """Translate Anima Studio's payload sections onto a Namespace that
    already carries the vendored trainer's real defaults. Pure dict ->
    attribute translation, so it can be unit-tested without importing
    the trainer (which needs torch). Every field named here exists in
    that trainer's parser - library/anima_train_utils.py
    (add_anima_training_arguments), library/train_util.py
    (add_dit_training_arguments / the shared parser) and
    anima_train_network.setup_parser."""
    anima_args = args.get("anima_args", {})
    general = args.get("general_args", {})
    optimizer = args.get("optimizer_args", {})
    network = args.get("network_args", {})
    saving = args.get("saving_args", {})

    if _KIND == "sdxl":
        ns.pretrained_model_name_or_path = general.get("pretrained_model_name_or_path") or anima_args.get("pretrained_model_name_or_path")
        if not ns.pretrained_model_name_or_path:
            raise TrainerError("SDXL training needs a base model checkpoint set.")
        if general.get("vae"):
            ns.vae = general["vae"]
        ns.network_module = "networks.lora"
        anima_args = {}          # Anima-only options must not leak into an SDXL run
    else:
        ns.pretrained_model_name_or_path = anima_args.get("pretrained_model_name_or_path")
        ns.qwen3 = anima_args.get("qwen3")
        ns.vae = anima_args.get("vae")
        missing = [k for k in ("pretrained_model_name_or_path", "qwen3", "vae") if not getattr(ns, k)]
        if missing:
            raise TrainerError(f"Anima training needs anima_args.{', anima_args.'.join(missing)} set.")
        ns.network_module = "networks.lora_anima"
    ns.network_dim = int(_get(network, "network_dim", default=16))
    ns.network_alpha = float(_get(network, "network_alpha", default=8.0))

    # Extra network kwargs (rank/module dropout, LyCORIS algo options, user-added entries)
    # travel as "key=value" strings, exactly what --network_args takes. They were never
    # forwarded before, so e.g. rank dropout silently did nothing.
    net_kwargs = network.get("network_args")
    if isinstance(net_kwargs, dict) and net_kwargs:
        def _fmt(v):
            return str(v).lower() if isinstance(v, bool) else str(v)
        use = dict(net_kwargs)
        if "algo" in use:                       # a LyCORIS algorithm was chosen in the UI
            try:
                _ensure_sd_scripts_importable()
                import importlib
                importlib.import_module("lycoris.kohya")
                ns.network_module = "lycoris.kohya"
                print(f"[anima-studio] using LyCORIS ({use['algo']}) via lycoris.kohya", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[anima-studio] WARNING: LyCORIS isn't available ({exc}); training a standard LoRA instead. "
                      "Ignored: " + ", ".join(sorted(k for k in use if k in _LYCORIS_ONLY)), flush=True)
                use = {k: v for k, v in use.items() if k not in _LYCORIS_ONLY}
        if use:
            ns.network_args = [f"{k}={_fmt(v)}" for k, v in use.items()]

    if optimizer.get("learning_rate"):
        ns.learning_rate = float(optimizer["learning_rate"])
    if optimizer.get("optimizer_type"):
        ns.optimizer_type = resolve_optimizer_name(optimizer["optimizer_type"])

    # Optimizer/scheduler knobs. optimizer_args and
    # lr_scheduler_args are "key=value" string lists - the
    # exact shape the vendored trainer's parsers expect
    # (each entry is unpacked with arg.split("=")).
    if optimizer.get("optimizer_args"):
        ns.optimizer_args = [str(a) for a in optimizer["optimizer_args"]]
    if optimizer.get("lr_scheduler"):
        ns.lr_scheduler = optimizer["lr_scheduler"]
    if optimizer.get("lr_scheduler_type"):
        ns.lr_scheduler_type = optimizer["lr_scheduler_type"]
    if optimizer.get("lr_scheduler_args"):
        ns.lr_scheduler_args = [str(a) for a in optimizer["lr_scheduler_args"]]
    if optimizer.get("lr_scheduler_num_cycles") is not None:
        ns.lr_scheduler_num_cycles = int(optimizer["lr_scheduler_num_cycles"])
    if optimizer.get("lr_scheduler_power") is not None:
        ns.lr_scheduler_power = float(optimizer["lr_scheduler_power"])
    if optimizer.get("lr_decay_steps") is not None:
        ns.lr_decay_steps = optimizer["lr_decay_steps"]
    if optimizer.get("lr_scheduler_timescale") is not None:
        ns.lr_scheduler_timescale = int(optimizer["lr_scheduler_timescale"])
    if optimizer.get("lr_scheduler_min_lr_ratio") is not None:
        ns.lr_scheduler_min_lr_ratio = float(optimizer["lr_scheduler_min_lr_ratio"])
    custom_scheduler = bool(optimizer.get("lr_scheduler_type"))
    if optimizer.get("warmup_ratio") is not None and not custom_scheduler:
        # (custom schedulers such as RexAnnealingWarmRestarts do their own warmup; the
        # trainer raises "does not require num_warmup_steps" if one is passed)
        # The parser takes the warmup as a step count or,
        # when < 1, as a ratio of total steps - the UI's
        # warmup ratio maps straight onto the float form.
        ns.lr_warmup_steps = float(optimizer["warmup_ratio"])
    if optimizer.get("zero_lr_warmup") and not custom_scheduler:
        ns.zero_lr_warmup = True

    # Steps vs epochs: the UI sends exactly one. In epochs mode the
    # vendored trainer derives max_train_steps itself once the
    # dataloader exists (train_network.py: "if args.max_train_epochs
    # is not None: ..."), so forward the epochs and mirror that
    # derivation here - purely so the progress total shown from step
    # 0 is already right. The trainer recomputes it identically.
    if general.get("max_train_steps"):
        ns.max_train_steps = int(general["max_train_steps"])
    elif general.get("max_train_epochs"):
        ns.max_train_epochs = int(general["max_train_epochs"])
        grad_accum = max(1, int(_get(args, "general_args", "gradient_accumulation_steps", default=1)))
        steps_per_epoch = math.ceil(dataset_item_count / max(1, dataset_batch_size))
        ns.max_train_steps = ns.max_train_epochs * math.ceil(steps_per_epoch / grad_accum)
    # else: leave the parser's own default (1600) - the real script's
    # behavior when neither is given.

    # Memory / dataloader / reproducibility knobs the UI exposes.
    if general.get("gradient_accumulation_steps"):
        ns.gradient_accumulation_steps = int(general["gradient_accumulation_steps"])
    if general.get("gradient_checkpointing"):
        ns.gradient_checkpointing = True
    if general.get("max_data_loader_n_workers") is not None:
        ns.max_data_loader_n_workers = int(general["max_data_loader_n_workers"])
    if general.get("seed") is not None:
        ns.seed = int(general["seed"])
    if general.get("max_token_length") is not None:
        ns.max_token_length = int(general["max_token_length"])
    if general.get("xformers"):
        ns.xformers = True
    if general.get("sdpa"):
        ns.sdpa = True

    # Full-precision modes: the shared trainer asserts
    # mixed_precision matches the full_* flag (train_network.py:
    # "if args.full_fp16: assert args.mixed_precision == 'fp16'"),
    # so the two always travel together.
    if general.get("full_fp16"):
        ns.full_fp16 = True
        ns.mixed_precision = "fp16"
    elif general.get("full_bf16"):
        ns.full_bf16 = True
        ns.mixed_precision = "bf16"
    elif general.get("mixed_precision"):
        ns.mixed_precision = general["mixed_precision"]

    # Anima-specific fields (add_anima_training_arguments unless
    # noted). Each is guarded so a UI that doesn't send it leaves
    # the parser's own default in place.
    if anima_args.get("t5_tokenizer_path"):
        ns.t5_tokenizer_path = anima_args["t5_tokenizer_path"]
    if anima_args.get("qwen3_max_token_length") is not None:
        ns.qwen3_max_token_length = int(anima_args["qwen3_max_token_length"])
    if anima_args.get("t5_max_token_length") is not None:
        ns.t5_max_token_length = int(anima_args["t5_max_token_length"])
    if anima_args.get("timestep_sampling"):
        ns.timestep_sampling = anima_args["timestep_sampling"]
    if anima_args.get("sigmoid_scale") is not None:
        ns.sigmoid_scale = float(anima_args["sigmoid_scale"])
    if anima_args.get("discrete_flow_shift") is not None:
        ns.discrete_flow_shift = float(anima_args["discrete_flow_shift"])
    if anima_args.get("attn_mode"):
        ns.attn_mode = anima_args["attn_mode"]
    if anima_args.get("split_attn"):
        ns.split_attn = True
    if anima_args.get("vae_chunk_size") is not None:
        ns.vae_chunk_size = int(anima_args["vae_chunk_size"])
    if anima_args.get("vae_disable_cache"):
        ns.vae_disable_cache = True
    if anima_args.get("blocks_to_swap") is not None:  # add_dit_training_arguments
        ns.blocks_to_swap = int(anima_args["blocks_to_swap"])
    if anima_args.get("unsloth_offload_checkpointing"):  # anima_train_network's own parser
        ns.unsloth_offload_checkpointing = True
    if saving.get("output_name"):
        ns.output_name = saving["output_name"]
    else:
        ns.output_name = "sdxl_lora" if _KIND == "sdxl" else "anima_lora"
    if saving.get("save_every_n_steps"):
        ns.save_every_n_epochs = None
        ns.save_every_n_steps = int(saving["save_every_n_steps"])


_NEVER_FORWARD = {"dataset_config", "output_dir", "config_file", "output_config", "warmup_ratio",
                  "lr_warmup_steps", "zero_lr_warmup", "network_module", "optimizer_args", "lr_scheduler_args",
                  "network_args", "max_train_steps", "max_train_epochs", "sample_prompts_text"}


def _forward_remaining_args(ns, defaults, args: dict) -> None:
    """The hand-written mapping above only covers some settings; everything else the UI
    sends (cache_latents, cache_text_encoder_outputs, train-unet-only, ...) used to be
    dropped silently, so the trainer ran with its slow defaults. Forward every remaining
    setting whose name is a real trainer option and whose type fits, exactly as a kohya
    TOML config would."""
    forwarded, unapplied = [], []
    for section in args.values():
        if not isinstance(section, dict):
            continue
        for key, val in section.items():
            if key in _NEVER_FORWARD or val == "" or val is None:
                continue                     # unset / empty fields keep the trainer's own default
            if not hasattr(ns, key):
                unapplied.append(f"{key} (not a trainer option)")
                continue
            if getattr(ns, key) != getattr(defaults, key, None):
                continue                     # already set by the explicit mapping
            dflt = getattr(defaults, key, None)
            if isinstance(val, bool):
                if dflt is not None and not isinstance(dflt, bool):
                    unapplied.append(f"{key} (type mismatch)")
                    continue
            elif isinstance(val, (int, float)):
                if isinstance(dflt, bool) or isinstance(dflt, (str, list)):
                    continue
                if isinstance(dflt, int) and not isinstance(dflt, bool):
                    val = int(val)
                elif isinstance(dflt, float):
                    val = float(val)
            elif isinstance(val, str):
                if isinstance(dflt, (bool, int, float, list)):
                    continue
            elif isinstance(val, list):
                if not (dflt is None or isinstance(dflt, list)):
                    continue
                val = [x for x in val]
            else:
                unapplied.append(f"{key} (unsupported value)")
                continue
            if key == "text_encoder_lr" and not isinstance(val, list):
                val = [float(val)]
            setattr(ns, key, val)
            forwarded.append(key)
    if unapplied:
        print("[anima-studio] WARNING settings NOT applied: " + ", ".join(sorted(unapplied)), flush=True)
    if forwarded:
        print("[anima-studio] extra settings applied: " + ", ".join(sorted(forwarded)), flush=True)


_LAST_DEFAULTS = None
_KIND = "anima"          # "anima" or "sdxl": which vendored sd-scripts trainer this run uses
_SUBSET_CAPTION_AUG = False
_CACHE_LATENTS_REQUESTED = False
_MASKED_LOSS_REQUESTED = False
_SUBSET_PIXEL_AUG = False


def _auto_optimize(ns) -> None:
    """Free speed and VRAM that cost no quality: only switched on when the settings that
    would make them change the result (random crop, colour aug, caption shuffle/dropout)
    are off. Never turns anything off that the user set."""
    g = lambda k, d=None: getattr(ns, k, d)
    notes = []
    pixel_aug = g("random_crop") or g("color_aug") or g("face_crop_aug_range") or _SUBSET_PIXEL_AUG
    if not g("cache_latents") and not pixel_aug:
        ns.cache_latents = True
        notes.append("cache_latents")
    caption_aug = g("shuffle_caption") or (g("caption_dropout_rate") or 0) > 0 or (g("caption_tag_dropout_rate") or 0) > 0 \
        or (g("caption_dropout_every_n_epochs") or 0) > 0 or g("token_warmup_step") \
        or _SUBSET_CAPTION_AUG or g("shuffle_caption")
    if g("network_train_unet_only") and not g("cache_text_encoder_outputs") and not caption_aug:
        ns.cache_text_encoder_outputs = True
        notes.append("cache_text_encoder_outputs")
    workers = g("max_data_loader_n_workers") or 0
    if workers > 0 and not g("persistent_data_loader_workers"):
        ns.persistent_data_loader_workers = True      # no worker respawn every epoch (slow on Windows)
        notes.append("persistent_data_loader_workers")
    if notes:
        print("[anima-studio] auto-enabled (no quality change): " + ", ".join(notes), flush=True)


def _log_environment(ns) -> None:
    """One block in the log with everything that decides speed and VRAM, so a slow run can be
    compared line by line with another trainer."""
    import importlib
    lines = []
    try:
        import torch
        lines.append(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
                     f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"torch import failed: {exc}")
    for mod in ("xformers", "flash_attn", "triton", "bitsandbytes", "sageattention"):
        try:
            m = importlib.import_module(mod)
            lines.append(f"{mod} {getattr(m, '__version__', 'installed')}")
        except Exception:  # noqa: BLE001
            lines.append(f"{mod}: not installed")
    d = _LAST_DEFAULTS
    changed = []
    if d is not None:
        for k, v in sorted(vars(ns).items()):
            if k.startswith("_") or k in ("pretrained_model_name_or_path", "qwen3", "vae", "output_dir", "dataset_config"):
                continue
            if v != getattr(d, k, None):
                changed.append(f"{k}={v}")
    print("[anima-studio] environment: " + " | ".join(lines), flush=True)
    print("[anima-studio] effective settings: " + ", ".join(changed), flush=True)


def _trainer_module():
    _ensure_sd_scripts_importable()
    if _KIND == "sdxl":
        import sdxl_train_network
        return sdxl_train_network
    import anima_train_network
    return anima_train_network


def build_args(args: dict, dataset_item_count: int, dataset_batch_size: int,
               dataset_toml_path: Path, output_dir: Path):
    """A real argparse.Namespace from the vendored trainer's own parser -
    every field starts at its real, tested default; only what Anima
    Studio's UI actually configures gets overridden on top. See module
    docstring for why this is safer than hand-building the Namespace."""
    _ensure_sd_scripts_importable()
    parser = _trainer_module().setup_parser()
    ns = parser.parse_args([])

    _apply_ui_overrides(ns, args, dataset_item_count, dataset_batch_size)
    defaults = parser.parse_args([])
    _forward_remaining_args(ns, defaults, args)
    _auto_optimize(ns)
    global _LAST_DEFAULTS
    _LAST_DEFAULTS = defaults

    ns.dataset_config = str(dataset_toml_path)
    ns.output_dir = str(output_dir)

    # TensorBoard needs a log folder; the trainer refuses to start without one.
    if getattr(ns, "log_with", None) in ("tensorboard", "all") and not getattr(ns, "logging_dir", None):
        ns.logging_dir = str(Path(output_dir) / "logs")
        print(f"[anima-studio] no log folder set - TensorBoard logs go to {ns.logging_dir}", flush=True)

    if _KIND == "anima":
        if getattr(ns, "attn_mode", None) == "sdpa":
            ns.attn_mode = "torch"  # backward-compat fixup, matches the real script's __main__ block
        ns._anima_model = True  # bypasses a generic "rectified flow" log/guard in the shared base trainer

    return ns


def run(args: dict, dataset: dict, on_progress, should_stop) -> None:
    run_kind("anima", args, dataset, on_progress, should_stop)


def run_kind(kind: str, args: dict, dataset: dict, on_progress, should_stop) -> None:
    global _KIND, _CACHE_LATENTS_REQUESTED, _MASKED_LOSS_REQUESTED
    _KIND = kind
    gen = args.get("general_args", {}) or {}
    _CACHE_LATENTS_REQUESTED = bool(gen.get("cache_latents"))
    _MASKED_LOSS_REQUESTED = bool((args.get("optimizer_args", {}) or {}).get("masked_loss") or gen.get("masked_loss"))
    if _CACHE_LATENTS_REQUESTED:
        print("[anima-studio] Cache latents is on, so Random crop / Color / Gamma augmentation are turned off "
              "(they can't be combined; same as LoRA Easy Training Scripts).", flush=True)
    _ensure_sd_scripts_importable()
    mod = _trainer_module()
    from library import train_util

    saving = args.get("saving_args", {})
    output_dir_str = saving.get("output_dir")
    if not output_dir_str:
        raise TrainerError("saving_args.output_dir is required.")
    output_dir = Path(output_dir_str).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = _dataset_resolution(dataset, args)
    batch_size = _dataset_batch_size(dataset)
    # Fail fast on a genuinely missing/empty folder before handing off to
    # the vendored trainer, which would otherwise report a much less
    # specific error from deep inside its own dataset loader.
    items = ds.build_dataset_items(dataset)

    with tempfile.TemporaryDirectory(prefix="anima_studio_dataset_") as tmp:
        toml_path = write_dataset_toml(dataset, resolution, Path(tmp) / "dataset.toml", batch_size)
        # Sample prompts typed into the app: write them to a temp file for the trainer.
        sample_args = args.get("sample_args") or {}
        text = (sample_args.get("sample_prompts_text") or "").strip()
        if text:
            pf = Path(tmp) / "sample_prompts.txt"
            pf.write_text(text + "\n", encoding="utf-8")
            args = {**args, "sample_args": {**sample_args, "sample_prompts": str(pf)}}
        ns = build_args(args, len(items), batch_size, toml_path, output_dir)
        ns = train_util.read_config_from_file(ns, mod.setup_parser())

        _log_environment(ns)
        on_progress(step=0, total_steps=ns.max_train_steps, loss=None)

        # The real trainer reports its own progress via tqdm/accelerate,
        # not a plain callback - patching those internals would be
        # fragile across versions, so progress here is start/finish only.
        # (Anima Studio's Monitor tab already reads sample images and
        # checkpoints from output_dir directly for the vendored-backend
        # path too, so this matches that, not a regression from it.)
        trainer = mod.SdxlNetworkTrainer() if kind == "sdxl" else mod.AnimaNetworkTrainer()
        trainer.train(ns)

        on_progress(step=ns.max_train_steps, total_steps=ns.max_train_steps, loss=None)
