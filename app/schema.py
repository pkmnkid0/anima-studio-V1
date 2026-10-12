"""
Canonical schema for Anima DiT LoRA training settings.

Every field name, default value, and enumerated option here mirrors the
argument names used by the LoRA_Easy_Training_Scripts (67372a "refresh"
fork) backend, which is the training engine this app talks to. Keeping
this file as the single source of truth means the frontend (which fetches
it via /api/train/schema), the TOML writer, and the backend HTTP client
all agree on field names without duplicating the list in three places.
"""

# --- Optimizers -------------------------------------------------------
# "Recommended" are shown first / highlighted in the UI. The full list
# covers the custom optimizer packages the backend installs
# (LoraEasyCustomOptimizer, adv_optm, prodigyplus) plus everything the
# vendored backend resolves natively (torch.optim, bitsandbytes,
# schedulefree, prodigyopt, dadaptation, lion-pytorch, transformers).
# The exotic names only resolve through dotted import paths, so the
# payload builder (train_config._DOTTED_OPTIMIZERS) translates them
# before sending - the dropdown names here stay the friendly ones.
OPTIMIZERS_RECOMMENDED = ["CAME", "FFTDescent", "OCGOpt", "OCGOptV2", "SimplifiedAdEMAMix"]

OPTIMIZERS_ALL = [
    "ABMOG", "AdaGC", "AdaMuon_adv", "AdamWScheduleFreePlus", "ADOPT", "Adopt_adv",
    "ADOPTAOScheduleFree", "ADOPTEMAMixScheduleFree", "ADOPTMARSScheduleFree",
    "ADOPTNesterovScheduleFree", "ADOPTScheduleFree", "AdEMAMix", "AdaBelief", "AdaFactor",
    "AdamMini", "AdamW", "AdamW4bitAO", "AdamW_adv", "AdamW8bit", "AdamW8bitAO",
    "AdamW8bitKahan", "AdamWScheduleFree", "AdamWfp8AO",
    "Adan", "AINOOpt", "Alice", "AMUSE", "BCOS", "BilatMuon", "BilatMuonNS",
    "CAME", "CASCADE",
    "Compass", "Compass8BitBNB", "CompassAO",
    "CompassADOPT", "CompassADOPTMARS", "CompassPlus", "CStableAdamW", "DAdaptAdaGrad",
    "DAdaptAdam", "DAdaptAdan", "DAdaptSGD", "Dehaze", "FADOPTMARSScheduleFree",
    "FADOPTScheduleFree", "FARMSCrop", "FARMSCropV2", "FCompass", "FCompassADOPT",
    "FCompassADOPTMARS", "FCompassPlus", "FFTDescent", "Fira", "FMARSCrop", "FMARSCropV2",
    "FMARSCropV2ExMachina", "FMARSCropV3", "FMARSCropV3ExMachina", "FishMonger",
    "FishMonger8BitBNB", "GaLore",
    "Glyph", "GOODDOG", "GrokFastAdamW", "LaProp", "LPFAdamW", "Lion", "Lion_adv", "MODA",
    "MomentusCaution", "Muon_adv", "Mythical", "NorMuonScheduleFree", "OAGOpt", "OCGOpt",
    "OCGOptV2", "Prodigy",
    "Prodigy_adv", "ProdigyPlusScheduleFree", "ProjectiveAdam", "RACS", "RAdamScheduleFree",
    "REMASTER", "RMSProp", "RMSPropADOPT", "RMSPropADOPTMARS", "Ranger21", "SCION",
    "SGDNesterov", "SGDNesterov8bit", "SGDScheduleFree", "SGDSaI", "ScalableShampoo",
    "ScheduleFreeWrapper", "SCGOpt", "SCORN", "SCORNMachina", "SimplifiedAdEMAMix",
    "SimplifiedAdEMAMixAINO", "SimplifiedAdEMAMixExM", "SignSGD_adv", "SingState",
    "SinkSGD_adv", "SNOO_ASGD", "SODA", "SODAWrapper", "StableSPAM", "TALON",
    "VSGD", "WarpAdam", "WarpAINO", "WiwiOpt",
]

# --- LR Schedulers ------------------------------------------------------
# UI label -> (lr_scheduler value, extra fields it enables)
LR_SCHEDULERS = [
    {"label": "Cosine", "value": "cosine", "fields": []},
    {"label": "Cosine With Restarts", "value": "cosine_with_restarts", "fields": ["num_cycles"]},
    {"label": "Cosine Annealing Warm Restarts (CAWR)", "value": "cosine_annealing_warm_restarts_(CAWR)",
     "fields": ["num_cycles", "min_lr", "gamma"]},
    {"label": "Cosine Annealing LR", "value": "CosineAnnealingLR", "fields": ["min_lr"]},
    {"label": "Cosine With Min LR", "value": "cosine_with_min_lr", "fields": ["num_cycles", "min_lr_ratio"]},
    {"label": "Inverse Sqrt", "value": "inverse_sqrt", "fields": ["timescale"]},
    {"label": "Piecewise Constant", "value": "piecewise_constant", "fields": []},
    {"label": "Linear", "value": "linear", "fields": []},
    {"label": "Constant", "value": "constant", "fields": []},
    {"label": "Constant With Warmup", "value": "constant_with_warmup", "fields": []},
    {"label": "Adafactor", "value": "adafactor", "fields": ["initial_lr"]},
    {"label": "Polynomial", "value": "polynomial", "fields": ["poly_power"]},
    {"label": "Rex Annealing Warm Restarts (RAWR)", "value": "rex_annealing_warm_restarts_(RAWR)",
     "fields": ["num_cycles", "min_lr", "gamma", "d_param"]},
    {"label": "Warmup Stable Decay", "value": "warmup_stable_decay",
     "fields": ["num_cycles", "decay_ratio", "decay_type", "min_lr_ratio"]},
]

LOSS_TYPES = [
    {"label": "L2", "value": "l2", "huber": False},
    {"label": "Huber", "value": "huber", "huber": True},
    {"label": "Smooth L1", "value": "smooth_l1", "huber": True},
    {"label": "X Sigmoid", "value": "x_sigmoid", "huber": False},
    {"label": "Log Cosh", "value": "log_cosh", "huber": False},
    {"label": "Standard Pseudo Huber", "value": "standard_pseudo_huber", "huber": True},
    {"label": "Standard Huber", "value": "standard_huber", "huber": True},
    {"label": "Standard Smooth L1", "value": "standard_smooth_l1", "huber": True},
    {"label": "Squared Logarithmic", "value": "squared_logarithmic", "huber": False},
    {"label": "Soft Welsch", "value": "soft_welsch", "huber": True},
    {"label": "Focal Frequency", "value": "focal_frequency", "huber": False},
    {"label": "Scaled Quadratic", "value": "scaled_quadratic", "huber": True},
    {"label": "Smooth L2 Log", "value": "smooth_l2_log", "huber": True},
    {"label": "MSE Pyramid 2D", "value": "mse_pyramid_2d", "huber": False},
]

HUBER_SCHEDULES = ["snr", "exponential", "constant"]
WSD_DECAY_TYPES = ["1-sqrt", "cosine", "linear"]

# --- Network / LoRA algorithms -----------------------------------------
# value -> lycoris algo string (None = native kohya LoRA, not a lycoris algo)
NETWORK_ALGOS = [
    {"label": "LoRA", "value": "lora", "lycoris": False},
    {"label": "LoCon", "value": "locon", "lycoris": False},
    {"label": "DyLoRA", "value": "dylora", "lycoris": False},
    {"label": "LoCon (LyCORIS)", "value": "locon (lycoris)", "lycoris": True},
    {"label": "LoHa", "value": "loha", "lycoris": True},
    {"label": "IA3", "value": "ia3", "lycoris": True},
    {"label": "Lokr", "value": "lokr", "lycoris": True},
    {"label": "BOFT", "value": "boft", "lycoris": True},
    {"label": "Diag-OFT", "value": "diag-oft", "lycoris": True},
    {"label": "Full", "value": "full", "lycoris": True},
    {"label": "GLoRA", "value": "glora", "lycoris": True},
    {"label": "ABBA", "value": "abba", "lycoris": True},
    {"label": "TLora", "value": "tlora", "lycoris": True},
    {"label": "GoRA", "value": "gora", "lycoris": True},
    {"label": "RaLoRA", "value": "ralora", "lycoris": True},
    {"label": "LoRA2", "value": "lora2", "lycoris": True},
    {"label": "OrthoLoRA", "value": "ortholora", "lycoris": True},
]

TRAIN_TARGETS = [
    {"label": "UNet + Text Encoder", "value": "both"},
    {"label": "UNet Only", "value": "unet"},
    {"label": "Text Encoder Only", "value": "te"},
]

TIMESTEP_SAMPLING = ["sigmoid", "shift", "sigma", "flux_shift", "uniform"]

SAMPLERS = ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver",
            "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]

CAPTION_EXTENSIONS = [".txt", ".caption"]

SAVE_PRECISIONS = ["fp16", "bf16", "float"]
SAVE_AS = ["safetensors", "pt", "ckpt"]
MIXED_PRECISIONS = ["fp16", "bf16", "float"]
MAX_TOKEN_LENGTHS = [75, 150, 225]

# --- Section defaults ----------------------------------------------------
# These map 1:1 to each widget's DEFAULTS dict in the reference app, so a
# freshly created config matches its out-of-the-box behavior.

DEFAULTS = {
    "general_args": {
        "seed": 42,
        "clip_skip": 2,
        "max_train_epochs": 1,
        "max_train_mode": "epochs",  # epochs | steps
        "max_data_loader_n_workers": 1,
        "max_token_length": 225,
        "prior_loss_weight": 1.0,
        "mixed_precision": "bf16",
        "resolution": 1024,
        "batch_size": 1,
        "resolution_height_enabled": False,
        "resolution_height": 1024,
    },
    "anima_args": {
        "qwen3_max_token_length": 512,
        "t5_max_token_length": 512,
        "timestep_sampling": "sigmoid",
        "discrete_flow_shift": 3.0,
        "sigmoid_scale": 1.0,
        "vae_chunk_size": 0,
        "blocks_to_swap": 0,
    },
    "network_args": {
        "network_dim": 32,
        "network_alpha": 16.0,
        "min_timestep": 0,
        "max_timestep": 1000,
        "algo": "lora",
        "conv_dim": 16,
        "conv_alpha": 32.0,
        "rank_dropout": 0.1,
        "module_dropout": 0.1,
        "network_dropout": 0.1,
        "dylora_unit": 4,
    },
    "optimizer_args": {
        "optimizer_type": "AdamW",
        "lr_scheduler": "cosine",
        "learning_rate": "1e-4",
        "unet_lr": "1e-4",
        "text_encoder_lr": "1e-4",
        "max_grad_norm": 1.0,
        "loss_type": "l2",
        "warmup_ratio": 0.0,
        "min_snr_gamma": 5.0,
        "num_cycles": 1,
        "poly_power": 1.0,
        "gamma": 0.9,
        "d_param": 0.9,
        "decay_ratio": 0.1,
        "decay_type": "1-sqrt",
        "min_lr": 0.0,
        "timescale": None,
        "min_lr_ratio": None,
        "initial_lr": 0.01,
        "lr_scheduler_extra_args": [],
        "huber_param": 0.1,
        "huber_schedule": "snr",
        "extra_optimizer_args": [{"name": "weight_decay", "value": "0.1"}],
    },
    "saving_args": {
        "save_precision": "fp16",
        "save_model_as": "safetensors",
        "save_freq_mode": "epochs",
        "save_freq": 1,
    },
    "bucket_args": {
        "enable_bucket": True,
        "bucket_no_upscale": True,
        "multires_training": False,
        "min_bucket_reso": 256,
        "max_bucket_reso": 2048,
        "bucket_reso_steps": 64,
    },
    "noise_args": {
        "noise_offset": 0.1,
        "pyramid_iterations": 6,
        "pyramid_discount": 0.3,
    },
    "sample_args": {
        "sample_sampler": "ddim",
        "steps_epochs_mode": "epochs",
        "steps_epoch_value": 1,
    },
    "logging_args": {
        "log_with": "tensorboard",
        "log_prefix_mode": "disabled",
        "run_name_mode": "default",
    },
    "edm_loss_args": {
        "optimizer_type": "AdamW",
        "learning_rate": "1e-2",
        "warmup_percent": 0.0,
        "constant_percent": 0.0,
        "num_channels": 128,
    },
    "accelerate_args": {
        "num_processes": 2,
        "main_process_port": 29500,
    },
    "subset_defaults": {
        "num_repeats": 1,
        "caption_extension": ".txt",
        "random_crop_padding_percent": 0.05,
    },
}


def full_schema() -> dict:
    """Everything the frontend needs to render the Train tab's form."""
    return {
        "optimizers": {"recommended": OPTIMIZERS_RECOMMENDED, "all": OPTIMIZERS_ALL},
        "lr_schedulers": LR_SCHEDULERS,
        "loss_types": LOSS_TYPES,
        "huber_schedules": HUBER_SCHEDULES,
        "wsd_decay_types": WSD_DECAY_TYPES,
        "network_algos": NETWORK_ALGOS,
        "train_targets": TRAIN_TARGETS,
        "timestep_sampling": TIMESTEP_SAMPLING,
        "samplers": SAMPLERS,
        "caption_extensions": CAPTION_EXTENSIONS,
        "save_precisions": SAVE_PRECISIONS,
        "save_as": SAVE_AS,
        "mixed_precisions": MIXED_PRECISIONS,
        "max_token_lengths": MAX_TOKEN_LENGTHS,
        "defaults": DEFAULTS,
    }
