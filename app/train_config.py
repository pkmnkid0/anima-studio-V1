"""
Translates the JSON settings payload the Train tab sends into the exact
nested structure the LoRA_Easy_Training_Scripts backend expects, both as

  - a TOML config file (for --config_file / "Save TOML", round trippable
    with the reference desktop app), and
  - the flat {"args": {...}, "dataset": {...}} JSON body the backend's
    HTTP API (/validate, /train) consumes.

Every conditional here mirrors a specific `edit_args(name, value,
optional=True)` call in the reference app: "optional" means the key is
dropped entirely when the value is falsy (0, "", False, None), matching
sd-scripts' own convention of treating an absent key as "use the
default". Comments point back to which widget/section the logic came
from so this stays auditable against the original.
"""
from __future__ import annotations

from typing import Any

from app import schema

# ---------------------------------------------------------------------
# Optimizer name resolution.
#
# The vendored engine resolves an optimizer in exactly two ways:
#   - fixed built-in branches (torch.optim, bitsandbytes,
#     schedulefree, prodigyopt, dadaptation, lion-pytorch,
#     transformers) that compare the name case-folded, and
#   - a generic fallback that does
#     importlib.import_module(<module>) + getattr(<module>, <class>)
#     for dotted "module.Class" names.
# Every exotic optimizer the backend ships (LoraEasyCustomOptimizer,
# adv_optm, prodigyplus) is therefore only reachable through its
# dotted import path - a bare "ABMOG" would crash with
# "AttributeError: module 'torch.optim' has no attribute 'ABMOG'".
# The UI keeps the friendly names; the payload translates them.
# ---------------------------------------------------------------------

# Names the engine's built-in branches handle bare.
_BARE_OPTIMIZERS = frozenset({
    "SGD", "SGDNesterov", "AdamW", "AdamW8bit", "SGDNesterov8bit",
    "Lion", "Lion8bit", "PagedAdamW", "PagedAdamW8bit",
    "PagedLion8bit", "PagedAdamW32bit", "Adafactor", "Prodigy",
    "DAdaptation", "DAdaptAdamPreprint", "DAdaptAdaGrad",
    "DAdaptAdam", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion",
    "DAdaptSGD", "AdamWScheduleFree", "RAdamScheduleFree",
    "SGDScheduleFree",
})

# adv_optm.optim exports exactly these eight.
_ADV_OPTIMIZERS = frozenset({
    "AdamW_adv", "Adopt_adv", "Lion_adv", "Prodigy_adv",
    "Muon_adv", "AdaMuon_adv", "SignSGD_adv", "SinkSGD_adv",
})

_DOTTED_OPTIMIZERS: dict[str, str] = {}
for _name in schema.OPTIMIZERS_ALL:
    if _name in _BARE_OPTIMIZERS:
        continue
    if _name in _ADV_OPTIMIZERS:
        _DOTTED_OPTIMIZERS[_name] = f"adv_optm.optim.{_name}"
    elif _name == "ProdigyPlusScheduleFree":
        _DOTTED_OPTIMIZERS[_name] = (
            "prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree"
        )
    else:
        _DOTTED_OPTIMIZERS[_name] = f"LoraEasyCustomOptimizer.{_name}"

_FRIENDLY_OPTIMIZERS = {v: k for k, v in _DOTTED_OPTIMIZERS.items()}


def _drop_falsy(d: dict) -> dict:
    """Remove keys whose value is falsy - mirrors edit_args(..., optional=True)."""
    return {k: v for k, v in d.items() if v not in (None, "", False) and v != []}


def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------
# Section builders. Each returns (args_dict, dataset_args_dict).
# ---------------------------------------------------------------------

def _build_model_and_general(ui: dict) -> tuple[dict, dict]:
    model = ui.get("model", {})
    general = ui.get("general", {})
    model_type = ui.get("model_type", "anima")

    args: dict[str, Any] = {}

    if model_type == "sdxl":
        args["pretrained_model_name_or_path"] = model.get("sdxl_checkpoint", "")
        args["v2"] = False
        args["sdxl"] = True
        if model.get("sdxl_vae"):
            args["vae"] = model["sdxl_vae"]
        elif general.get("vae"):
            args["vae"] = general["vae"]
    else:
        # Base model field: the backend's general_args table always
        # carries pretrained_model_name_or_path (GeneralUI wires it
        # without optional=True). For an Anima run the DiT checkpoint
        # is the closest equivalent, so we mirror it here too -
        # harmless if the backend only reads the model path from
        # anima_args, and satisfies the field if it validates
        # general_args independently.
        args["pretrained_model_name_or_path"] = model.get("dit_model", "")
        args["v2"] = False
        args["sdxl"] = False
        if general.get("vae"):
            args["vae"] = general["vae"]

    args["no_half_vae"] = bool(general.get("no_half_vae"))
    args["lowram"] = bool(general.get("lowram"))
    args["highvram"] = bool(general.get("highvram"))

    if general.get("full_fp16"):
        args["full_fp16"] = True
    elif general.get("full_bf16"):
        args["full_bf16"] = True
    else:
        args["mixed_precision"] = general.get("mixed_precision", "bf16")
    # fp8_base is unsupported for Anima DiT (mirrors ArgsListUI's
    # update_fp8_for_anima, which force-disables it) - only ever sent
    # for SDXL, where sd-scripts does support it.
    if model_type == "sdxl" and general.get("fp8_base"):
        args["fp8_base"] = True

    args["gradient_checkpointing"] = bool(general.get("gradient_checkpointing"))
    if general.get("gradient_accumulation_enabled"):
        args["gradient_accumulation_steps"] = int(general.get("gradient_accumulation_steps", 1))

    args["seed"] = int(general.get("seed", 42))
    args["max_data_loader_n_workers"] = int(general.get("max_data_loader_n_workers", 1))
    args["max_token_length"] = int(general.get("max_token_length", 225))
    args["prior_loss_weight"] = float(general.get("prior_loss_weight", 1.0))

    xformers = bool(general.get("xformers"))
    sdpa = bool(general.get("sdpa"))
    if xformers and sdpa:
        sdpa = False
    if xformers:
        args["xformers"] = True
    if sdpa:
        args["sdpa"] = True

    max_mode = general.get("max_train_mode", "epochs")
    max_value = int(general.get("max_train_value", 10))
    args["max_train_epochs" if max_mode == "epochs" else "max_train_steps"] = max_value

    if general.get("cache_latents"):
        args["cache_latents"] = True
        if general.get("cache_latents_to_disk"):
            args["cache_latents_to_disk"] = True

    if general.get("keep_tokens_separator_enabled"):
        args["keep_tokens_separator"] = general.get("keep_tokens_separator", "")
    if general.get("training_comment_enabled"):
        args["training_comment"] = general.get("training_comment", "")
    if general.get("protected_tags_file_enabled"):
        args["protected_tags_file"] = general.get("protected_tags_file", "")

    # experimental (advanced / flow-matching extras, applicable mainly
    # to SDXL rectified-flow experiments per upstream README - included
    # for parity but off by default and independent of the Anima path)
    exp = ui.get("experimental", {})
    if exp.get("vae_batch_size_enabled"):
        args["vae_batch_size"] = int(exp.get("vae_batch_size", 1))
    if exp.get("vae_reflection"):
        args["vae_reflection"] = True
    if exp.get("vae_custom_scale_enabled"):
        args["vae_custom_scale"] = float(exp.get("vae_custom_scale", 0.0))
    if exp.get("vae_custom_shift_enabled"):
        args["vae_custom_shift"] = float(exp.get("vae_custom_shift", 0.0))
    if exp.get("zero_cond_dropout"):
        args["zero_cond_dropout"] = True
    if exp.get("debiased_estimation_loss"):
        args["debiased_estimation_loss"] = True

    dataset_args: dict[str, Any] = {}
    height_enabled = bool(general.get("resolution_height_enabled"))
    width = int(general.get("resolution_width", 1024))
    if height_enabled:
        dataset_args["resolution"] = [width, int(general.get("resolution_height", width))]
    else:
        dataset_args["resolution"] = width
    dataset_args["batch_size"] = int(general.get("batch_size", 1))

    return args, dataset_args


def _build_anima(ui: dict) -> dict:
    if ui.get("model_type", "anima") != "anima":
        return {}
    model = ui.get("model", {})
    args: dict[str, Any] = {
        "pretrained_model_name_or_path": model.get("dit_model", ""),
        "qwen3": model.get("qwen3_model", ""),
        "vae": model.get("vae_model", ""),
    }
    if model.get("t5_tokenizer_path"):
        args["t5_tokenizer_path"] = model["t5_tokenizer_path"]

    args["qwen3_max_token_length"] = int(model.get("qwen3_max_token_length", 512))
    args["t5_max_token_length"] = int(model.get("t5_max_token_length", 512))

    sampling = model.get("timestep_sampling", "sigmoid")
    args["timestep_sampling"] = sampling
    if sampling in ("sigmoid", "shift", "flux_shift"):
        args["sigmoid_scale"] = float(model.get("sigmoid_scale", 1.0))
    if sampling in ("sigma", "shift"):
        args["discrete_flow_shift"] = float(model.get("discrete_flow_shift", 3.0))

    if int(model.get("vae_chunk_size", 0)) > 0:
        args["vae_chunk_size"] = int(model["vae_chunk_size"])
    if model.get("vae_disable_cache"):
        args["vae_disable_cache"] = True
    if int(model.get("blocks_to_swap", 0)) > 0:
        args["blocks_to_swap"] = int(model["blocks_to_swap"])

    xformers = bool(ui.get("general", {}).get("xformers"))
    if model.get("flash_attn"):
        args["attn_mode"] = "flash"
    # split_attn is force-enabled whenever xFormers is selected (mirrors
    # update_split_attn_from_general); SDPA enables it internally on the
    # backend side so we don't need to send it in that case.
    if xformers or model.get("split_attn"):
        args["split_attn"] = True
    if model.get("unsloth_offload_checkpointing"):
        args["unsloth_offload_checkpointing"] = True

    return args


def _build_network(ui: dict) -> dict:
    net = ui.get("network", {})
    algo = net.get("algo", "lora")
    is_lycoris = algo not in ("lora", "locon", "dylora")
    is_kohya = algo in ("lora", "locon", "dylora")

    args: dict[str, Any] = {
        "network_dim": int(net.get("network_dim", 32)),
        "network_alpha": round(float(net.get("network_alpha", 16.0)), 2),
        "min_timestep": int(net.get("min_timestep", 0)),
        "max_timestep": int(net.get("max_timestep", 1000)),
    }
    # GoRA / RaLoRA force alpha == dim on the backend side; keep the UI's
    # value as sent (the UI itself mirrors dim -> alpha when this algo is
    # selected, same as the reference NetworkUI).

    train_target = net.get("train_target", "both")
    if train_target == "unet":
        args["network_train_unet_only"] = True
    elif train_target == "te":
        args["network_train_text_encoder_only"] = True

    if net.get("network_dropout_enabled") and not is_lycoris:
        args["network_dropout"] = float(net.get("network_dropout", 0.1))

    if net.get("cache_te_outputs"):
        args["cache_text_encoder_outputs"] = True
        if net.get("cache_te_to_disk"):
            args["cache_text_encoder_outputs_to_disk"] = True

    if is_kohya and net.get("lora_fa"):
        args["fa"] = True

    if net.get("ip_noise_gamma_enabled"):
        args["ip_noise_gamma"] = round(float(net.get("ip_noise_gamma", 0.1)), 4)

    # --- network_args sub-table ---
    network_args: dict[str, Any] = {}
    if is_lycoris:
        network_args["algo"] = algo.split(" ")[0].lower()
        if net.get("preset"):
            network_args["preset"] = net["preset"]
        if net.get("conv_dim_enabled") and algo != "ortholora":
            network_args["conv_dim"] = int(net.get("conv_dim", 16))
            network_args["conv_alpha"] = float(net.get("conv_alpha", 32.0))
        if net.get("use_tucker"):
            network_args["use_tucker"] = True
        if net.get("train_norm"):
            network_args["train_norm"] = True
        if net.get("rescaled"):
            network_args["rescaled"] = True
        if net.get("constraint_enabled"):
            network_args["constraint"] = float(net.get("constraint", 0.0))
        dora_capable = algo in ("locon (lycoris)", "loha", "lokr", "abba", "gora",
                                 "ralora", "lora2", "ortholora")
        bypass = bool(net.get("bypass_mode")) and algo != "glora"
        dora = bool(net.get("dora")) and dora_capable and not bypass
        if dora:
            network_args["dora_wd"] = True
        if bypass and dora_capable:
            network_args["bypass_mode"] = True
        if net.get("network_dropout_enabled") and algo != "ia3":
            network_args["dropout"] = float(net.get("network_dropout", 0.1))
        if net.get("rank_dropout_enabled") and algo != "ia3":
            network_args["rank_dropout"] = float(net.get("rank_dropout", 0.1))
        if net.get("module_dropout_enabled") and algo != "ia3":
            network_args["module_dropout"] = float(net.get("module_dropout", 0.1))

    if not is_lycoris:
        # lora_anima reads these two as network kwargs, so they work for plain LoRA too
        if net.get("rank_dropout_enabled"):
            network_args["rank_dropout"] = float(net.get("rank_dropout", 0.1))
        if net.get("module_dropout_enabled"):
            network_args["module_dropout"] = float(net.get("module_dropout", 0.1))

    if algo == "dylora":
        network_args["unit"] = int(net.get("dylora_unit", 4))

    for group_key, arg_name in (
        ("block_weight", None),
        ("block_dims", "block_dims"),
        ("block_alphas", "block_alphas"),
        ("conv_block_dims", "conv_block_dims"),
        ("conv_block_alphas", "conv_block_alphas"),
    ):
        group = net.get(group_key, {})
        if not group or not group.get("enabled"):
            continue
        if group_key == "block_weight":
            if group.get("down"):
                network_args["down_lr_weight"] = group["down"]
            if group.get("mid") is not None:
                network_args["mid_lr_weight"] = group["mid"]
            if group.get("up"):
                network_args["up_lr_weight"] = group["up"]
        else:
            values = group.get("values")
            if values:
                network_args[arg_name] = values

    for extra in net.get("extra_network_args", []) or []:
        name = (extra.get("name") or "").strip()
        value = extra.get("value")
        if name and value not in (None, ""):
            network_args[name] = _coerce(value)

    if network_args:
        args["network_args"] = network_args

    return args


def _coerce(value: str):
    if isinstance(value, (int, float, bool)):
        return value
    v = str(value).strip()
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _build_optimizer(ui: dict) -> dict:
    opt = ui.get("optimizer", {})
    optimizer_type = opt.get("optimizer_type", "AdamW")
    args: dict[str, Any] = {
        # Exotic optimizers travel as dotted import paths -
        # see _DOTTED_OPTIMIZERS above.
        "optimizer_type": _DOTTED_OPTIMIZERS.get(optimizer_type, optimizer_type),
        "learning_rate": _coerce(opt.get("learning_rate", "1e-4")),
        "max_grad_norm": float(opt.get("max_grad_norm", 1.0)),
        "loss_type": opt.get("loss_type", "l2"),
    }

    if opt.get("unet_lr_enabled"):
        args["unet_lr"] = _coerce(opt.get("unet_lr", "1e-4"))
    if opt.get("te_lr_enabled"):
        te_lr = opt.get("te_lr", "1e-4")
        if isinstance(te_lr, str) and "," in te_lr:
            try:
                args["text_encoder_lr"] = [float(p.strip()) for p in te_lr.split(",") if p.strip()]
            except ValueError:
                args["text_encoder_lr"] = 0.0
        else:
            args["text_encoder_lr"] = _coerce(te_lr)

    if opt.get("warmup_ratio_enabled"):
        args["warmup_ratio"] = round(float(opt.get("warmup_ratio", 0.0)), 2)
    if opt.get("zero_lr_warmup"):
        args["zero_lr_warmup"] = True

    scheduler = opt.get("lr_scheduler", "cosine")
    scheduler_key = scheduler.replace(" ", "_")
    lr_scheduler_args: dict[str, Any] = {}

    if scheduler_key == "cosine_with_restarts":
        args["lr_scheduler_num_cycles"] = int(opt.get("num_cycles", 1))
    elif scheduler_key in ("cosine_annealing_warm_restarts_(CAWR)", "cosine_annealing_warmup_restarts"):
        args["lr_scheduler_type"] = (
            "LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts.CosineAnnealingWarmRestarts"
        )
        if opt.get("min_lr") not in (None, "", 0, "0", "0.0"):
            lr_scheduler_args["min_lr"] = _coerce(opt.get("min_lr"))
        args["lr_scheduler_num_cycles"] = int(opt.get("num_cycles", 1))
        lr_scheduler_args["gamma"] = round(1 - float(opt.get("gamma", 0.9)), 2)
    elif scheduler_key in ("rex_annealing_warm_restarts_(RAWR)", "rex"):
        args["lr_scheduler_type"] = (
            "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
        )
        if opt.get("min_lr") not in (None, "", 0, "0", "0.0"):
            lr_scheduler_args["min_lr"] = _coerce(opt.get("min_lr"))
        args["lr_scheduler_num_cycles"] = int(opt.get("num_cycles", 1))
        lr_scheduler_args["gamma"] = round(1 - float(opt.get("gamma", 0.9)), 2)
        lr_scheduler_args["d"] = round(float(opt.get("d_param", 0.9)), 4)
    elif scheduler_key == "polynomial":
        args["lr_scheduler_power"] = float(opt.get("poly_power", 1.0))
    elif scheduler_key in ("warmup_stable_decay", "wsd"):
        args["lr_scheduler_num_cycles"] = int(opt.get("num_cycles", 1))
        args["lr_decay_steps"] = round(float(opt.get("decay_ratio", 0.1)), 2)
        lr_scheduler_args["decay_type"] = opt.get("decay_type", "1-sqrt").lower()
        # WSD's floor is lr_scheduler_min_lr_ratio (a ratio of
        # the initial LR), not a scheduler kwarg - sending it
        # as "min_lr" would be an unknown kwarg for that scheduler.
        if opt.get("min_lr_ratio") not in (None, "", 0, "0", "0.0"):
            args["lr_scheduler_min_lr_ratio"] = round(float(opt["min_lr_ratio"]), 4)
    elif scheduler_key in ("CosineAnnealingLR", "cosineannealinglr"):
        if opt.get("min_lr") not in (None, "", 0, "0", "0.0"):
            lr_scheduler_args["min_lr"] = _coerce(opt.get("min_lr"))
    elif scheduler_key == "inverse_sqrt":
        if opt.get("timescale") not in (None, "", 0, "0", "0.0"):
            args["lr_scheduler_timescale"] = int(opt["timescale"])
    elif scheduler_key == "cosine_with_min_lr":
        args["lr_scheduler_num_cycles"] = int(opt.get("num_cycles", 1))
        if opt.get("min_lr_ratio") not in (None, "", 0, "0", "0.0"):
            args["lr_scheduler_min_lr_ratio"] = round(float(opt["min_lr_ratio"]), 4)

    # Adafactor's schedule takes its initial LR inline in
    # the name ("adafactor:0.01") - a bare "adafactor"
    # would crash the backend's name.split(":")[1] lookup.
    if scheduler_key == "adafactor":
        args["lr_scheduler"] = f"adafactor:{opt.get('initial_lr', 0.01)}"
    else:
        args["lr_scheduler"] = scheduler

    # Extra scheduler kwargs (e.g. piecewise_constant's
    # step_rules) ride along as additional "key=value" entries.
    extra_scheduler_args = [
        f"{kv.get('name')}={kv.get('value')}"
        for kv in (opt.get("lr_scheduler_extra_args") or [])
        if kv.get("name")
    ]
    if lr_scheduler_args or extra_scheduler_args:
        # "key=value" strings - the exact shape --lr_scheduler_args
        # parses (it unpacks each entry with arg.split("="), so a
        # dict here would crash the backend).
        args["lr_scheduler_args"] = [f"{k}={v}" for k, v in lr_scheduler_args.items()] + extra_scheduler_args

    if opt.get("scale_weight_norms_enabled"):
        args["scale_weight_norms"] = float(opt.get("scale_weight_norms", 1.0))
    if opt.get("min_snr_gamma_enabled"):
        args["min_snr_gamma"] = float(opt.get("min_snr_gamma", 5.0))
    if opt.get("zero_terminal_snr"):
        args["zero_terminal_snr"] = True
    if opt.get("masked_loss"):
        args["masked_loss"] = True

    loss_type = args["loss_type"]
    huber_family = loss_type not in ("l2", "x_sigmoid", "log_cosh", "squared_logarithmic",
                                      "focal_frequency", "frequency_distribution")
    if huber_family:
        args["huber_c"] = round(float(opt.get("huber_param", 0.1)), 4)
        needs_schedule = loss_type in ("huber", "smooth_l1", "standard_pseudo_huber",
                                        "standard_huber", "standard_smooth_l1", "soft_welsch",
                                        "scaled_quadratic", "smooth_l2_log")
        if needs_schedule:
            args["huber_schedule"] = opt.get("huber_schedule", "snr")

    extra_args = {}
    for extra in opt.get("extra_optimizer_args", []) or []:
        name = (extra.get("name") or "").strip()
        value = extra.get("value")
        if name and value not in (None, ""):
            extra_args[name] = value
    if extra_args:
        # "key=value" strings - the exact shape --optimizer_args
        # parses (it unpacks each entry with arg.split("="), so a
        # dict here would crash the backend).
        args["optimizer_args"] = [f"{k}={v}" for k, v in extra_args.items()]

    return args


def _build_saving(ui: dict) -> dict:
    s = ui.get("saving", {})
    args: dict[str, Any] = {
        "output_dir": s.get("output_dir", ""),
        "save_precision": s.get("save_precision", "fp16"),
        "save_model_as": s.get("save_model_as", "safetensors"),
    }
    if s.get("output_name_enabled"):
        args["output_name"] = s.get("output_name", "")
    if s.get("resume_enabled"):
        args["resume"] = s.get("resume", "")
    if s.get("save_only_last_enabled"):
        key = "save_last_n_epochs" if s.get("save_last_mode", "epochs") == "epochs" else "save_last_n_steps"
        args[key] = int(s.get("save_last_value", 1))
    if s.get("save_ratio_enabled"):
        args["save_n_epoch_ratio"] = float(s.get("save_ratio", 1))
    if s.get("save_tag_enabled"):
        args["tag_occurrence"] = True
        args["tag_file_location"] = s.get("save_tag_dir", "")
    if s.get("save_freq_enabled"):
        key = "save_every_n_epochs" if s.get("save_freq_mode", "epochs") == "epochs" else "save_every_n_steps"
        args[key] = int(s.get("save_freq_value", 1))
    if s.get("save_toml_enabled"):
        args["save_toml"] = True
        args["save_toml_location"] = s.get("save_toml_dir", "")
    if s.get("save_state"):
        args["save_state"] = True
        if s.get("save_last_state_enabled"):
            key = ("save_last_n_epochs_state" if s.get("save_last_state_mode", "epochs") == "epochs"
                    else "save_last_n_steps_state")
            args[key] = int(s.get("save_last_state_value", 1))
    return args


def _build_bucket(ui: dict) -> dict:
    b = ui.get("bucket", {})
    dataset_args: dict[str, Any] = {"enable_bucket": bool(b.get("enabled", True))}
    if dataset_args["enable_bucket"]:
        dataset_args["bucket_no_upscale"] = bool(b.get("no_upscale", True))
        dataset_args["multires_training"] = bool(b.get("multires_training", False))
        dataset_args["min_bucket_reso"] = int(b.get("min_reso", 256))
        dataset_args["max_bucket_reso"] = int(b.get("max_reso", 2048))
        dataset_args["bucket_reso_steps"] = int(b.get("reso_steps", 64))
    return dataset_args


def _build_noise(ui: dict) -> dict:
    n = ui.get("noise", {})
    args: dict[str, Any] = {}
    if n.get("noise_offset_enabled"):
        args["noise_offset"] = float(n.get("noise_offset", 0.1))
    if n.get("pyramid_enabled"):
        args["multires_noise_iterations"] = int(n.get("pyramid_iterations", 6))
        args["multires_noise_discount"] = round(float(n.get("pyramid_discount", 0.3)), 4)
    return args


def _build_sample(ui: dict) -> dict:
    s = ui.get("sample", {})
    if not s.get("enabled"):
        return {}
    args: dict[str, Any] = {"sample_sampler": s.get("sampler", "ddim").lower()}
    key = "sample_every_n_steps" if s.get("freq_mode", "epochs") == "steps" else "sample_every_n_epochs"
    args[key] = int(s.get("freq_value", 1))
    text = (s.get("prompts_text") or "").strip()
    if text:
        args["sample_prompts_text"] = text      # written to a temporary file by the trainer
    elif s.get("prompts_file"):
        args["sample_prompts"] = s["prompts_file"]
    return args


def _build_logging(ui: dict) -> dict:
    lg = ui.get("logging", {})
    if not lg.get("enabled"):
        return {}
    args: dict[str, Any] = {
        "log_with": lg.get("log_with", "tensorboard").lower(),
        "logging_dir": lg.get("logging_dir", ""),
    }
    prefix_mode = lg.get("log_prefix_mode", "disabled")
    args["log_prefix_mode"] = prefix_mode
    if prefix_mode == "manual":
        args["log_prefix"] = lg.get("log_prefix", "")
    run_mode = lg.get("run_name_mode", "default")
    args["run_name_mode"] = run_mode
    if run_mode == "manual":
        args["run_name"] = lg.get("run_name", "")
    if lg.get("tracker_name_enabled"):
        args["log_tracker_name"] = lg.get("tracker_name", "")
    if lg.get("log_with", "tensorboard").lower() in ("wandb", "all"):
        args["wandb_api_key"] = lg.get("wandb_api_key", "")
    return args


def _build_edm_loss(ui: dict) -> dict:
    e = ui.get("edm_loss", {})
    if not e.get("enabled"):
        return {}
    args: dict[str, Any] = {
        "edm2_loss_weighting": True,
        "edm2_loss_weighting_optimizer": e.get("optimizer_type", "AdamW"),
        "edm2_loss_weighting_optimizer_lr": e.get("learning_rate", "1e-2"),
        "edm2_loss_weighting_num_channels": int(e.get("num_channels", 128)),
    }
    if e.get("optimizer_args"):
        args["edm2_loss_weighting_optimizer_args"] = e["optimizer_args"]
    if e.get("scheduler_enabled"):
        args["edm2_loss_weighting_lr_scheduler_warmup_percent"] = float(e.get("warmup_percent", 0.0))
        args["edm2_loss_weighting_lr_scheduler_constant_percent"] = float(e.get("constant_percent", 0.0))
    if e.get("initial_weights"):
        args["edm2_loss_weighting_initial_weights"] = e["initial_weights"]
    return args


def _build_extra_args(ui: dict) -> tuple[dict, dict]:
    args: dict[str, Any] = {}
    dataset_args: dict[str, Any] = {}
    for extra in ui.get("extra_args", []) or []:
        name = (extra.get("name") or "").strip()
        value = extra.get("value")
        if not name or value in (None, ""):
            continue
        target = dataset_args if extra.get("dataset") else args
        target[name] = _coerce(value)
    return args, dataset_args


def _build_subset(raw: dict) -> dict:
    out: dict[str, Any] = {
        "name": raw.get("name") or "subset",
        "image_dir": raw.get("image_dir", ""),
        "num_repeats": int(raw.get("num_repeats", 1)),
        "caption_extension": raw.get("caption_extension", ".txt"),
        "random_crop_padding_percent": float(raw.get("random_crop_padding_percent", 0.05)),
    }
    if raw.get("target_image_dir"):
        out["target_image_dir"] = raw["target_image_dir"]
    if raw.get("conditioning_data_dir"):
        out["conditioning_data_dir"] = raw["conditioning_data_dir"]
    if raw.get("shuffle_caption"):
        out["shuffle_caption"] = True
    if raw.get("flip_aug"):
        out["flip_aug"] = True
    if raw.get("keep_tokens"):
        out["keep_tokens"] = int(raw["keep_tokens"])
    if raw.get("color_aug"):
        out["color_aug"] = True
    if raw.get("random_crop"):
        out["random_crop"] = True
    if raw.get("is_reg"):
        out["is_reg"] = True
    if raw.get("is_val"):
        out["is_val"] = True

    if raw.get("face_crop_enabled"):
        out["face_crop_aug_range"] = [
            float(raw.get("face_crop_width", 1.0)), float(raw.get("face_crop_height", 1.0))
        ]
    if raw.get("caption_dropout_enabled"):
        if raw.get("caption_dropout_rate"):
            out["caption_dropout_rate"] = float(raw["caption_dropout_rate"])
        if raw.get("caption_dropout_every_n_epochs"):
            out["caption_dropout_every_n_epochs"] = int(raw["caption_dropout_every_n_epochs"])
        if raw.get("caption_tag_dropout_rate"):
            out["caption_tag_dropout_rate"] = float(raw["caption_tag_dropout_rate"])
    if raw.get("gamma_aug_enabled"):
        out["gamma_aug"] = True
        out["gamma_aug_range"] = [float(raw.get("gamma_aug_min", 0.95)), float(raw.get("gamma_aug_max", 1.05))]
        if raw.get("gamma_aug_rate"):
            out["gamma_aug_rate"] = float(raw["gamma_aug_rate"])
    if raw.get("shuffle_caption_sigma_enabled") and raw.get("shuffle_caption_sigma"):
        out["shuffle_caption_sigma"] = raw["shuffle_caption_sigma"]
    if raw.get("token_warmup_enabled"):
        out["token_warmup_min"] = int(raw.get("token_warmup_min", 1))
        out["token_warmup_step"] = int(raw.get("token_warmup_step", 1))
    if raw.get("protected_tags_file"):
        out["protected_tags_file"] = raw["protected_tags_file"]
    return out


def build_backend_payload(ui: dict) -> dict:
    """Returns {"args": {...}, "dataset": {...}, "subsets": [...],
    "train_mode": "lora"} - the canonical form this app works with
    internally. Callers derive both the TOML file and the HTTP request
    body for /validate + /train from this single structure."""

    general_args, general_dataset = _build_model_and_general(ui)
    bucket_dataset = _build_bucket(ui)
    extra_args, extra_dataset = _build_extra_args(ui)

    args = {
        "general_args": general_args,
        "anima_args": _build_anima(ui),
        "network_args": _build_network(ui),
        "optimizer_args": _build_optimizer(ui),
        "saving_args": _build_saving(ui),
        "noise_args": _build_noise(ui),
        "sample_args": _build_sample(ui),
        "logging_args": _build_logging(ui),
        "edm_loss_args": _build_edm_loss(ui),
        "extra_args": extra_args,
    }
    args = {k: v for k, v in args.items() if v}

    dataset = {
        "general_args": general_dataset,
        "bucket_args": bucket_dataset,
        "extra_args": extra_dataset,
    }
    dataset = {k: v for k, v in dataset.items() if v}

    subsets = [_build_subset(s) for s in ui.get("subsets", [])]

    return {
        "args": args,
        "dataset": dataset,
        "subsets": subsets,
        "train_mode": ui.get("train_mode", "lora"),
    }


def to_toml_document(payload: dict) -> dict:
    """Reshape the canonical payload into the exact table layout the
    reference desktop app's save_toml()/process_toml() round trip uses,
    so files produced here also open cleanly in that app and vice
    versa."""
    doc: dict[str, Any] = {}
    for section, section_args in payload["args"].items():
        doc.setdefault(section, {})["args"] = section_args
    for section, section_dataset in payload["dataset"].items():
        doc.setdefault(section, {})["dataset_args"] = section_dataset
    if payload["subsets"]:
        doc["subsets"] = payload["subsets"]
    doc["train_mode"] = {"train_mode": payload["train_mode"]}
    return doc


def from_toml_document(doc: dict) -> dict:
    """Inverse of to_toml_document - used when importing a TOML file
    produced by either this app or the reference desktop app."""
    args: dict[str, Any] = {}
    dataset: dict[str, Any] = {}
    subsets = doc.get("subsets", [])
    train_mode = "lora"
    for key, value in doc.items():
        if key == "subsets":
            continue
        if key == "train_mode":
            train_mode = value.get("train_mode", "lora")
            continue
        if not isinstance(value, dict):
            continue
        if "args" in value:
            args[key] = value["args"]
        if "dataset_args" in value:
            dataset[key] = value["dataset_args"]
    return {"args": args, "dataset": dataset, "subsets": subsets, "train_mode": train_mode}


# ---------------------------------------------------------------------
# Inverse mapping: canonical {args, dataset, subsets, train_mode} ->
# the UI's JSON shape (mirrors default_ui_payload() field for field).
# Used for "Load TOML". This is a best-effort reconstruction: a couple
# of native-kohya algo choices ("lora" vs "locon" with no lycoris
# extras) are genuinely indistinguishable from the saved args alone, so
# those default to "lora" - everything else round-trips exactly, which
# is checked in the test suite by re-running build_backend_payload on
# the reconstructed UI and comparing against the original config.
# ---------------------------------------------------------------------

def default_ui_payload() -> dict:
    """The same defaults the frontend's Train tab starts from - kept
    here too so "Load TOML" always returns a complete, valid shape even
    when a config only sets a handful of fields."""
    return {
        "train_mode": "lora",
        "model_type": "anima",
        "model": {
            "dit_model": "", "qwen3_model": "", "vae_model": "", "t5_tokenizer_path": "",
            "qwen3_max_token_length": 512, "t5_max_token_length": 512,
            "timestep_sampling": "sigmoid", "discrete_flow_shift": 3.0, "sigmoid_scale": 1.0,
            "vae_chunk_size": 0, "vae_disable_cache": False, "blocks_to_swap": 0,
            "flash_attn": False, "split_attn": False, "unsloth_offload_checkpointing": False,
            "sdxl_checkpoint": "", "sdxl_vae": "",
        },
        "general": {
            "resolution_width": 1024, "resolution_height_enabled": False, "resolution_height": 1024,
            "batch_size": 1, "mixed_precision": "bf16", "full_fp16": False, "full_bf16": False, "fp8_base": False,
            "xformers": False, "sdpa": True, "seed": 42, "prior_loss_weight": 1.0,
            "max_data_loader_n_workers": 1, "max_train_mode": "epochs", "max_train_value": 10,
            "gradient_checkpointing": True, "gradient_accumulation_enabled": False, "gradient_accumulation_steps": 1,
            "cache_latents": True, "cache_latents_to_disk": False,
            "lowram": False, "highvram": False, "no_half_vae": False,
            "keep_tokens_separator_enabled": False, "keep_tokens_separator": "",
            "training_comment_enabled": False, "training_comment": "",
            "protected_tags_file_enabled": False, "protected_tags_file": "", "vae": "",
        },
        "network": {
            "algo": "lora", "preset": "", "network_dim": 32, "network_alpha": 16,
            "conv_dim_enabled": False, "conv_dim": 16, "conv_alpha": 32,
            "min_timestep": 0, "max_timestep": 1000, "train_target": "both",
            "network_dropout_enabled": False, "network_dropout": 0.1,
            "rank_dropout_enabled": False, "rank_dropout": 0.1,
            "module_dropout_enabled": False, "module_dropout": 0.1,
            "cache_te_outputs": False, "cache_te_to_disk": False, "dylora_unit": 4,
            "bypass_mode": False, "use_tucker": False, "train_norm": False, "dora": False, "rescaled": False,
            "constraint_enabled": False, "constraint": 0,
            "lora_fa": False, "ip_noise_gamma_enabled": False, "ip_noise_gamma": 0.1,
            "extra_network_args": [],
            "block_weight": {"enabled": False, "down": "", "mid": "", "up": ""},
            "block_dims": {"enabled": False, "values": ""},
            "block_alphas": {"enabled": False, "values": ""},
            "conv_block_dims": {"enabled": False, "values": ""},
            "conv_block_alphas": {"enabled": False, "values": ""},
        },
        "optimizer": {
            "optimizer_type": "AdamW", "learning_rate": "1e-4",
            "unet_lr_enabled": False, "unet_lr": "1e-4", "te_lr_enabled": False, "te_lr": "1e-4",
            "lr_scheduler": "cosine", "num_cycles": 1, "poly_power": 1.0,
            "min_lr": 0, "gamma": 0.9, "d_param": 0.9, "decay_ratio": 0.1, "decay_type": "1-sqrt",
            "timescale": None, "min_lr_ratio": None, "initial_lr": 0.01,
            "lr_scheduler_extra_args": [],
            "warmup_ratio_enabled": False, "warmup_ratio": 0.05, "zero_lr_warmup": False,
            "scale_weight_norms_enabled": False, "scale_weight_norms": 1.0, "max_grad_norm": 1.0,
            "min_snr_gamma_enabled": False, "min_snr_gamma": 5.0,
            "zero_terminal_snr": False, "masked_loss": False,
            "loss_type": "l2", "huber_schedule": "snr", "huber_param": 0.1,
            "extra_optimizer_args": [{"name": "weight_decay", "value": "0.1"}],
        },
        "saving": {
            "output_dir": "", "output_name_enabled": False, "output_name": "",
            "save_precision": "fp16", "save_model_as": "safetensors",
            "resume_enabled": False, "resume": "",
            "save_only_last_enabled": False, "save_last_mode": "epochs", "save_last_value": 4,
            "save_ratio_enabled": False, "save_ratio": 1,
            "save_tag_enabled": False, "save_tag_dir": "",
            "save_freq_enabled": True, "save_freq_mode": "epochs", "save_freq_value": 1,
            "save_toml_enabled": False, "save_toml_dir": "",
            "save_state": False, "save_last_state_enabled": False, "save_last_state_mode": "epochs", "save_last_state_value": 1,
        },
        "bucket": {"enabled": True, "no_upscale": True, "multires_training": False, "min_reso": 256, "max_reso": 2048, "reso_steps": 64},
        "noise": {"noise_offset_enabled": False, "noise_offset": 0.1, "pyramid_enabled": False, "pyramid_iterations": 6, "pyramid_discount": 0.3},
        "sample": {"enabled": False, "sampler": "ddim", "freq_mode": "epochs", "freq_value": 1, "prompts_file": "", "prompts_text": ""},
        "logging": {
            "enabled": True, "log_with": "tensorboard", "logging_dir": "",
            "log_prefix_mode": "disabled", "log_prefix": "", "run_name_mode": "default", "run_name": "",
            "tracker_name_enabled": False, "tracker_name": "", "wandb_api_key": "",
        },
        "edm_loss": {
            "enabled": False, "optimizer_type": "AdamW", "learning_rate": "1e-2", "optimizer_args": "",
            "scheduler_enabled": False, "warmup_percent": 0, "constant_percent": 0,
            "initial_weights": "", "num_channels": 128,
        },
        "experimental": {
            "vae_batch_size_enabled": False, "vae_batch_size": 1, "vae_reflection": False,
            "vae_custom_scale_enabled": False, "vae_custom_scale": 0,
            "vae_custom_shift_enabled": False, "vae_custom_shift": 0,
            "zero_cond_dropout": False, "debiased_estimation_loss": False,
        },
        "accelerate": {"enabled": False, "num_processes": 2, "main_process_port": 29500},
        "extra_args": [],
        "subsets": [],
    }


def _algo_lookup() -> dict:
    from . import schema
    return {a["value"].split(" ")[0].lower(): a["value"] for a in schema.NETWORK_ALGOS if a["lycoris"]}


def _num_str(values) -> str:
    return ",".join(str(v) for v in values) if values else ""


def _ui_general(args: dict, dataset: dict, ui: dict) -> None:
    g = ui["general"]
    is_sdxl = bool(args.get("sdxl"))
    ui["model_type"] = "sdxl" if is_sdxl else "anima"
    if is_sdxl:
        ui["model"]["sdxl_checkpoint"] = args.get("pretrained_model_name_or_path", "")
        if "vae" in args:
            ui["model"]["sdxl_vae"] = args["vae"]
    elif "vae" in args:
        g["vae"] = args["vae"]
    g["no_half_vae"] = bool(args.get("no_half_vae"))
    g["lowram"] = bool(args.get("lowram"))
    g["highvram"] = bool(args.get("highvram"))
    if args.get("full_fp16"):
        g["full_fp16"] = True
    elif args.get("full_bf16"):
        g["full_bf16"] = True
    elif "mixed_precision" in args:
        g["mixed_precision"] = args["mixed_precision"]
    if args.get("fp8_base"):
        g["fp8_base"] = True
    g["gradient_checkpointing"] = bool(args.get("gradient_checkpointing"))
    if "gradient_accumulation_steps" in args:
        g["gradient_accumulation_enabled"] = True
        g["gradient_accumulation_steps"] = args["gradient_accumulation_steps"]
    if "seed" in args:
        g["seed"] = args["seed"]
    if "max_data_loader_n_workers" in args:
        g["max_data_loader_n_workers"] = args["max_data_loader_n_workers"]
    if "prior_loss_weight" in args:
        g["prior_loss_weight"] = args["prior_loss_weight"]
    if args.get("xformers"):
        g["xformers"], g["sdpa"] = True, False
    elif args.get("sdpa"):
        g["xformers"], g["sdpa"] = False, True
    if "max_train_epochs" in args:
        g["max_train_mode"], g["max_train_value"] = "epochs", args["max_train_epochs"]
    elif "max_train_steps" in args:
        g["max_train_mode"], g["max_train_value"] = "steps", args["max_train_steps"]
    if args.get("cache_latents"):
        g["cache_latents"] = True
        g["cache_latents_to_disk"] = bool(args.get("cache_latents_to_disk"))
    if "keep_tokens_separator" in args:
        g["keep_tokens_separator_enabled"], g["keep_tokens_separator"] = True, args["keep_tokens_separator"]
    if "training_comment" in args:
        g["training_comment_enabled"], g["training_comment"] = True, args["training_comment"]
    if "protected_tags_file" in args:
        g["protected_tags_file_enabled"], g["protected_tags_file"] = True, args["protected_tags_file"]

    resolution = dataset.get("resolution")
    if isinstance(resolution, list):
        g["resolution_height_enabled"] = True
        g["resolution_width"], g["resolution_height"] = resolution[0], resolution[1]
    elif resolution is not None:
        g["resolution_width"] = resolution
    if "batch_size" in dataset:
        g["batch_size"] = dataset["batch_size"]

    exp = ui["experimental"]
    if "vae_batch_size" in args:
        exp["vae_batch_size_enabled"], exp["vae_batch_size"] = True, args["vae_batch_size"]
    if args.get("vae_reflection"):
        exp["vae_reflection"] = True
    if "vae_custom_scale" in args:
        exp["vae_custom_scale_enabled"], exp["vae_custom_scale"] = True, args["vae_custom_scale"]
    if "vae_custom_shift" in args:
        exp["vae_custom_shift_enabled"], exp["vae_custom_shift"] = True, args["vae_custom_shift"]
    if args.get("zero_cond_dropout"):
        exp["zero_cond_dropout"] = True
    if args.get("debiased_estimation_loss"):
        exp["debiased_estimation_loss"] = True


def _ui_anima(args: dict, ui: dict) -> None:
    m = ui["model"]
    if "pretrained_model_name_or_path" in args:
        m["dit_model"] = args["pretrained_model_name_or_path"]
    if "qwen3" in args:
        m["qwen3_model"] = args["qwen3"]
    if "vae" in args:
        m["vae_model"] = args["vae"]
    if "t5_tokenizer_path" in args:
        m["t5_tokenizer_path"] = args["t5_tokenizer_path"]
    for key in ("qwen3_max_token_length", "t5_max_token_length", "blocks_to_swap", "vae_chunk_size"):
        if key in args:
            m[key] = args[key]
    if "timestep_sampling" in args:
        m["timestep_sampling"] = args["timestep_sampling"]
    if "sigmoid_scale" in args:
        m["sigmoid_scale"] = args["sigmoid_scale"]
    if "discrete_flow_shift" in args:
        m["discrete_flow_shift"] = args["discrete_flow_shift"]
    if args.get("vae_disable_cache"):
        m["vae_disable_cache"] = True
    if args.get("attn_mode") == "flash":
        m["flash_attn"] = True
    if args.get("split_attn"):
        m["split_attn"] = True
    if args.get("unsloth_offload_checkpointing"):
        m["unsloth_offload_checkpointing"] = True


def _ui_network(args: dict, ui: dict) -> None:
    n = ui["network"]
    if "network_dim" in args:
        n["network_dim"] = args["network_dim"]
    if "network_alpha" in args:
        n["network_alpha"] = args["network_alpha"]
    if "min_timestep" in args:
        n["min_timestep"] = args["min_timestep"]
    if "max_timestep" in args:
        n["max_timestep"] = args["max_timestep"]
    if args.get("network_train_unet_only"):
        n["train_target"] = "unet"
    elif args.get("network_train_text_encoder_only"):
        n["train_target"] = "te"
    if "network_dropout" in args:
        n["network_dropout_enabled"], n["network_dropout"] = True, args["network_dropout"]
    if args.get("cache_text_encoder_outputs"):
        n["cache_te_outputs"] = True
        n["cache_te_to_disk"] = bool(args.get("cache_text_encoder_outputs_to_disk"))
    if args.get("fa"):
        n["lora_fa"] = True
    if "ip_noise_gamma" in args:
        n["ip_noise_gamma_enabled"], n["ip_noise_gamma"] = True, args["ip_noise_gamma"]

    na = dict(args.get("network_args") or {})
    algo_short = na.pop("algo", None)
    if algo_short:
        n["algo"] = _algo_lookup().get(algo_short, "lora")
    elif "unit" in na:
        n["algo"] = "dylora"
    # else: leave as the default "lora" - native "lora" vs "locon" with no
    # lycoris extras produce identical args in this schema, so it can't be
    # told apart from the file alone.

    if "preset" in na:
        n["preset"] = na.pop("preset")
    if "conv_dim" in na:
        n["conv_dim_enabled"] = True
        n["conv_dim"] = na.pop("conv_dim")
        n["conv_alpha"] = na.pop("conv_alpha", n["conv_alpha"])
    for flag in ("use_tucker", "train_norm", "rescaled", "bypass_mode"):
        if na.pop(flag, False):
            n[flag] = True
    if na.pop("dora_wd", False):
        n["dora"] = True
    if "constraint" in na:
        n["constraint_enabled"], n["constraint"] = True, na.pop("constraint")
    if "dropout" in na:
        n["network_dropout_enabled"], n["network_dropout"] = True, na.pop("dropout")
    if "rank_dropout" in na:
        n["rank_dropout_enabled"], n["rank_dropout"] = True, na.pop("rank_dropout")
    if "module_dropout" in na:
        n["module_dropout_enabled"], n["module_dropout"] = True, na.pop("module_dropout")
    if "unit" in na:
        n["dylora_unit"] = na.pop("unit")

    if "down_lr_weight" in na or "mid_lr_weight" in na or "up_lr_weight" in na:
        n["block_weight"] = {
            "enabled": True,
            "down": _num_str(na.pop("down_lr_weight", [])),
            "mid": str(na.pop("mid_lr_weight")) if na.get("mid_lr_weight") is not None else "",
            "up": _num_str(na.pop("up_lr_weight", [])),
        }
    for key in ("block_dims", "block_alphas", "conv_block_dims", "conv_block_alphas"):
        if key in na:
            n[key] = {"enabled": True, "values": _num_str(na.pop(key))}

    # Anything left over is a genuinely custom network_args entry.
    n["extra_network_args"] = [{"name": k, "value": str(v)} for k, v in na.items()]


def _ui_optimizer(args: dict, ui: dict) -> None:
    o = ui["optimizer"]
    if "optimizer_type" in args:
        # Dotted import paths round-trip back to the
        # friendly dropdown name.
        o["optimizer_type"] = _FRIENDLY_OPTIMIZERS.get(
            args["optimizer_type"], args["optimizer_type"])
    if "learning_rate" in args:
        o["learning_rate"] = str(args["learning_rate"])
    if "unet_lr" in args:
        o["unet_lr_enabled"], o["unet_lr"] = True, str(args["unet_lr"])
    if "text_encoder_lr" in args:
        te_lr = args["text_encoder_lr"]
        o["te_lr_enabled"] = True
        o["te_lr"] = ",".join(str(x) for x in te_lr) if isinstance(te_lr, list) else str(te_lr)
    if "lr_scheduler" in args:
        o["lr_scheduler"] = args["lr_scheduler"]
    if "lr_scheduler_num_cycles" in args:
        o["num_cycles"] = args["lr_scheduler_num_cycles"]
    if "lr_scheduler_power" in args:
        o["poly_power"] = args["lr_scheduler_power"]
    if "lr_decay_steps" in args:
        o["decay_ratio"] = args["lr_decay_steps"]
    if "lr_scheduler_timescale" in args:
        o["timescale"] = args["lr_scheduler_timescale"]
    if "lr_scheduler_min_lr_ratio" in args:
        o["min_lr_ratio"] = args["lr_scheduler_min_lr_ratio"]
    # Adafactor's initial LR travels inline in the
    # scheduler name ("adafactor:0.01").
    if isinstance(args.get("lr_scheduler"), str) and args["lr_scheduler"].startswith("adafactor:"):
        o["lr_scheduler"] = "adafactor"
        o["initial_lr"] = _coerce(args["lr_scheduler"].split(":", 1)[1])
    lsa = args.get("lr_scheduler_args") or []
    if isinstance(lsa, dict):  # payloads from before the list-form fix
        lsa = [f"{k}={v}" for k, v in lsa.items()]
    scheduler_extras = []
    for item in lsa:
        if "=" not in str(item):
            continue
        key, value = str(item).split("=", 1)
        key, value = key.strip(), value.strip()
        if key == "min_lr":
            o["min_lr"] = _coerce(value)
        elif key == "gamma":
            o["gamma"] = round(1 - float(value), 4)
        elif key == "d":
            o["d_param"] = _coerce(value)
        elif key == "decay_type":
            o["decay_type"] = value
        else:
            scheduler_extras.append({"name": key, "value": value})
    if scheduler_extras:
        o["lr_scheduler_extra_args"] = scheduler_extras
    if "warmup_ratio" in args:
        o["warmup_ratio_enabled"], o["warmup_ratio"] = True, args["warmup_ratio"]
    if args.get("zero_lr_warmup"):
        o["zero_lr_warmup"] = True
    if "scale_weight_norms" in args:
        o["scale_weight_norms_enabled"], o["scale_weight_norms"] = True, args["scale_weight_norms"]
    if "max_grad_norm" in args:
        o["max_grad_norm"] = args["max_grad_norm"]
    if "min_snr_gamma" in args:
        o["min_snr_gamma_enabled"], o["min_snr_gamma"] = True, args["min_snr_gamma"]
    if args.get("zero_terminal_snr"):
        o["zero_terminal_snr"] = True
    if args.get("masked_loss"):
        o["masked_loss"] = True
    if "loss_type" in args:
        o["loss_type"] = args["loss_type"]
    if "huber_c" in args:
        o["huber_param"] = args["huber_c"]
    if "huber_schedule" in args:
        o["huber_schedule"] = args["huber_schedule"]
    if args.get("optimizer_args"):
        oa = args["optimizer_args"]
        if isinstance(oa, dict):  # payloads from before the list-form fix
            oa = [f"{k}={v}" for k, v in oa.items()]
        o["extra_optimizer_args"] = []
        for item in oa:
            if "=" not in str(item):
                continue
            name, value = str(item).split("=", 1)
            o["extra_optimizer_args"].append({"name": name.strip(), "value": value.strip()})


def _ui_saving(args: dict, ui: dict) -> None:
    s = ui["saving"]
    for key in ("output_dir", "save_precision", "save_model_as"):
        if key in args:
            s[key] = args[key]
    if "output_name" in args:
        s["output_name_enabled"], s["output_name"] = True, args["output_name"]
    if "resume" in args:
        s["resume_enabled"], s["resume"] = True, args["resume"]
    if "save_last_n_epochs" in args:
        s["save_only_last_enabled"], s["save_last_mode"], s["save_last_value"] = True, "epochs", args["save_last_n_epochs"]
    elif "save_last_n_steps" in args:
        s["save_only_last_enabled"], s["save_last_mode"], s["save_last_value"] = True, "steps", args["save_last_n_steps"]
    if "save_n_epoch_ratio" in args:
        s["save_ratio_enabled"], s["save_ratio"] = True, args["save_n_epoch_ratio"]
    if args.get("tag_occurrence"):
        s["save_tag_enabled"], s["save_tag_dir"] = True, args.get("tag_file_location", "")
    if "save_every_n_epochs" in args:
        s["save_freq_enabled"], s["save_freq_mode"], s["save_freq_value"] = True, "epochs", args["save_every_n_epochs"]
    elif "save_every_n_steps" in args:
        s["save_freq_enabled"], s["save_freq_mode"], s["save_freq_value"] = True, "steps", args["save_every_n_steps"]
    else:
        s["save_freq_enabled"] = False
    if args.get("save_toml"):
        s["save_toml_enabled"], s["save_toml_dir"] = True, args.get("save_toml_location", "")
    if args.get("save_state"):
        s["save_state"] = True
        if "save_last_n_epochs_state" in args:
            s["save_last_state_enabled"], s["save_last_state_mode"], s["save_last_state_value"] = \
                True, "epochs", args["save_last_n_epochs_state"]
        elif "save_last_n_steps_state" in args:
            s["save_last_state_enabled"], s["save_last_state_mode"], s["save_last_state_value"] = \
                True, "steps", args["save_last_n_steps_state"]


def _ui_bucket(dataset: dict, ui: dict) -> None:
    b = ui["bucket"]
    b["enabled"] = bool(dataset.get("enable_bucket", True))
    if b["enabled"]:
        b["no_upscale"] = bool(dataset.get("bucket_no_upscale", True))
        b["multires_training"] = bool(dataset.get("multires_training", False))
        for key, ui_key in (("min_bucket_reso", "min_reso"), ("max_bucket_reso", "max_reso"), ("bucket_reso_steps", "reso_steps")):
            if key in dataset:
                b[ui_key] = dataset[key]


def _ui_noise(args: dict, ui: dict) -> None:
    n = ui["noise"]
    if "noise_offset" in args:
        n["noise_offset_enabled"], n["noise_offset"] = True, args["noise_offset"]
    if "multires_noise_iterations" in args:
        n["pyramid_enabled"] = True
        n["pyramid_iterations"] = args["multires_noise_iterations"]
        n["pyramid_discount"] = args.get("multires_noise_discount", n["pyramid_discount"])


def _ui_sample(args: dict, ui: dict) -> None:
    if not args:
        return
    s = ui["sample"]
    s["enabled"] = True
    if "sample_sampler" in args:
        s["sampler"] = args["sample_sampler"]
    if "sample_every_n_steps" in args:
        s["freq_mode"], s["freq_value"] = "steps", args["sample_every_n_steps"]
    elif "sample_every_n_epochs" in args:
        s["freq_mode"], s["freq_value"] = "epochs", args["sample_every_n_epochs"]
    if "sample_prompts" in args:
        s["prompts_file"] = args["sample_prompts"]
    if "sample_prompts_text" in args:
        s["prompts_text"] = args["sample_prompts_text"]


def _ui_logging(args: dict, ui: dict) -> None:
    if not args:
        return
    lg = ui["logging"]
    lg["enabled"] = True
    if "log_with" in args:
        lg["log_with"] = args["log_with"]
    if "logging_dir" in args:
        lg["logging_dir"] = args["logging_dir"]
    if "log_prefix_mode" in args:
        lg["log_prefix_mode"] = args["log_prefix_mode"]
    if "log_prefix" in args:
        lg["log_prefix"] = args["log_prefix"]
    if "run_name_mode" in args:
        lg["run_name_mode"] = args["run_name_mode"]
    if "run_name" in args:
        lg["run_name"] = args["run_name"]
    if "log_tracker_name" in args:
        lg["tracker_name_enabled"], lg["tracker_name"] = True, args["log_tracker_name"]
    if "wandb_api_key" in args:
        lg["wandb_api_key"] = args["wandb_api_key"]


def _ui_edm_loss(args: dict, ui: dict) -> None:
    if not args.get("edm2_loss_weighting"):
        return
    e = ui["edm_loss"]
    e["enabled"] = True
    if "edm2_loss_weighting_optimizer" in args:
        e["optimizer_type"] = args["edm2_loss_weighting_optimizer"]
    if "edm2_loss_weighting_optimizer_lr" in args:
        e["learning_rate"] = str(args["edm2_loss_weighting_optimizer_lr"])
    if "edm2_loss_weighting_optimizer_args" in args:
        e["optimizer_args"] = args["edm2_loss_weighting_optimizer_args"]
    if "edm2_loss_weighting_lr_scheduler_warmup_percent" in args:
        e["scheduler_enabled"] = True
        e["warmup_percent"] = args["edm2_loss_weighting_lr_scheduler_warmup_percent"]
        e["constant_percent"] = args.get("edm2_loss_weighting_lr_scheduler_constant_percent", 0)
    if "edm2_loss_weighting_initial_weights" in args:
        e["initial_weights"] = args["edm2_loss_weighting_initial_weights"]
    if "edm2_loss_weighting_num_channels" in args:
        e["num_channels"] = args["edm2_loss_weighting_num_channels"]


def _ui_subset(raw: dict) -> dict:
    s = {
        "name": raw.get("name", "subset"),
        "image_dir": raw.get("image_dir", ""),
        "target_image_dir": raw.get("target_image_dir", ""),
        "conditioning_data_dir": raw.get("conditioning_data_dir", ""),
        "num_repeats": raw.get("num_repeats", 1),
        "caption_extension": raw.get("caption_extension", ".txt"),
        "random_crop_padding_percent": raw.get("random_crop_padding_percent", 0.05),
        "shuffle_caption": bool(raw.get("shuffle_caption")),
        "flip_aug": bool(raw.get("flip_aug")),
        "keep_tokens": raw.get("keep_tokens", 0),
        "color_aug": bool(raw.get("color_aug")),
        "random_crop": bool(raw.get("random_crop")),
        "is_reg": bool(raw.get("is_reg")),
        "is_val": bool(raw.get("is_val")),
        "face_crop_enabled": False, "face_crop_width": 1, "face_crop_height": 1,
        "caption_dropout_enabled": False, "caption_dropout_rate": 0,
        "caption_dropout_every_n_epochs": 0, "caption_tag_dropout_rate": 0,
        "gamma_aug_enabled": False, "gamma_aug_min": 0.95, "gamma_aug_max": 1.05, "gamma_aug_rate": 0.5,
        "shuffle_caption_sigma_enabled": False, "shuffle_caption_sigma": 0,
        "token_warmup_enabled": False, "token_warmup_min": 1, "token_warmup_step": 1,
        "protected_tags_file": raw.get("protected_tags_file", ""),
    }
    if "face_crop_aug_range" in raw:
        s["face_crop_enabled"] = True
        s["face_crop_width"], s["face_crop_height"] = raw["face_crop_aug_range"]
    if any(k in raw for k in ("caption_dropout_rate", "caption_dropout_every_n_epochs", "caption_tag_dropout_rate")):
        s["caption_dropout_enabled"] = True
        s["caption_dropout_rate"] = raw.get("caption_dropout_rate", 0)
        s["caption_dropout_every_n_epochs"] = raw.get("caption_dropout_every_n_epochs", 0)
        s["caption_tag_dropout_rate"] = raw.get("caption_tag_dropout_rate", 0)
    if raw.get("gamma_aug"):
        s["gamma_aug_enabled"] = True
        rng = raw.get("gamma_aug_range", [0.95, 1.05])
        s["gamma_aug_min"], s["gamma_aug_max"] = rng[0], rng[1]
        s["gamma_aug_rate"] = raw.get("gamma_aug_rate", 0.5)
    if "shuffle_caption_sigma" in raw:
        s["shuffle_caption_sigma_enabled"], s["shuffle_caption_sigma"] = True, raw["shuffle_caption_sigma"]
    if "token_warmup_min" in raw or "token_warmup_step" in raw:
        s["token_warmup_enabled"] = True
        s["token_warmup_min"] = raw.get("token_warmup_min", 1)
        s["token_warmup_step"] = raw.get("token_warmup_step", 1)
    return s


def to_ui_payload(payload: dict) -> dict:
    """Best-effort inverse of build_backend_payload - turns a parsed
    TOML's canonical form back into the Train tab's JSON shape, used by
    "Load TOML"."""
    ui = default_ui_payload()
    ui["train_mode"] = payload.get("train_mode", "lora")
    # default_ui_payload() seeds a "weight_decay: 0.1" suggestion for a
    # brand-new config; a loaded file should reflect exactly what it
    # contains instead, so start empty and let _ui_optimizer refill it
    # only if the file actually has extra optimizer args.
    ui["optimizer"]["extra_optimizer_args"] = []
    args = payload.get("args", {})
    dataset = payload.get("dataset", {})

    _ui_general(args.get("general_args", {}), dataset.get("general_args", {}), ui)
    _ui_anima(args.get("anima_args", {}), ui)
    _ui_network(args.get("network_args", {}), ui)
    _ui_optimizer(args.get("optimizer_args", {}), ui)
    _ui_saving(args.get("saving_args", {}), ui)
    _ui_bucket(dataset.get("bucket_args", {}), ui)
    _ui_noise(args.get("noise_args", {}), ui)
    _ui_sample(args.get("sample_args", {}), ui)
    _ui_logging(args.get("logging_args", {}), ui)
    _ui_edm_loss(args.get("edm_loss_args", {}), ui)

    extra_args = args.get("extra_args", {})
    extra_dataset = dataset.get("extra_args", {})
    ui["extra_args"] = (
        [{"name": k, "value": str(v), "dataset": False} for k, v in extra_args.items()]
        + [{"name": k, "value": str(v), "dataset": True} for k, v in extra_dataset.items()]
    )

    ui["subsets"] = [_ui_subset(s) for s in payload.get("subsets", [])]
    return ui
