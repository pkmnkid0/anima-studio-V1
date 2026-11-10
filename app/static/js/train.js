// ==========================================================================
// Train tab. Renders every setting from the reference project's schema,
// keeps a single `state` object in the UI's payload shape, and posts it
// to /api/train/build for a live TOML preview and to the backend for
// actual training.
// ==========================================================================

const Train = (() => {
  let addDatasetFolder = null;
  let schemaData = null;

  const state = {
    train_mode: "lora",
    model_type: "anima",
    model: {
      dit_model: "", qwen3_model: "", vae_model: "", t5_tokenizer_path: "",
      qwen3_max_token_length: 512, t5_max_token_length: 512,
      timestep_sampling: "sigmoid", discrete_flow_shift: 3.0, sigmoid_scale: 1.0,
      vae_chunk_size: 0, vae_disable_cache: false, blocks_to_swap: 0,
      flash_attn: false, split_attn: false, unsloth_offload_checkpointing: false,
      sdxl_checkpoint: "", sdxl_vae: "",
      // MOSS-Image (defaults follow the 8 GB VRAM profile)
      moss_model: "", moss_text_encoder: "", moss_tokenizer: "", moss_vae: "", moss_max_token_length: 512,
      moss_lora_save_format: "both", moss_timestep_sampling: "moss",
      moss_discrete_flow_shift: 1.0, moss_sigmoid_scale: 1.0,
      moss_ts_mu: -1.09, moss_ts_s: 0.95, moss_ts_cutoff: 0.95, moss_ts_uniform_mix: 0.01,
      moss_blocks_to_swap: 36, moss_int8: "off", moss_attn_backend: "auto", moss_compile: "none",
      moss_unchecked_blocks: 0, moss_fp32_modulation: false, moss_vae_sample_posterior: false,
      moss_sample_skip_layers: "22", moss_te_mode: "offload", moss_te_gpu_gb: 6.0,
      moss_te_cpu_gb: 48.0, moss_vram_profile: "8",
      // which engine trains MOSS-Image: "fork" (sd-scripts moss fork) or "kohaku" (Low-VRAM trainer)
      moss_engine: "fork",
      // <kohaku-defaults> Low-VRAM trainer engine; generated from KOHAKU_DEFAULTS in app/train_config.py (tests keep them equal)
      kohaku_weights: "",
      kohaku_variant: "base",
      kohaku_transformer: "",
      kohaku_text_encoder: "",
      kohaku_vae: "",
      kohaku_text_device: "",
      kohaku_vae_dtype: "auto",
      kohaku_train_text_refiner: false,
      kohaku_vram: "8",
      kohaku_memory_profile: "minimal",
      kohaku_base_precision: "",
      kohaku_int8_compute: false,
      kohaku_int8_dgrad: "",
      kohaku_fp8_recipe: "",
      kohaku_fp8_dgrad: "",
      kohaku_step_release: "",
      kohaku_compile: "",
      kohaku_tune: "table",
      kohaku_checkpoint: "",
      kohaku_stream: "",
      kohaku_stream_resident: "",
      kohaku_optim_state: "",
      kohaku_pinned: "",
      kohaku_act_offload: "",
      kohaku_grad_offload: "",
      kohaku_fused_step: "",
      kohaku_te_memory: "ring",
      kohaku_te_ahead: 8,
      kohaku_te_batch: 16,
      kohaku_te_max_tokens: 8192,
      kohaku_te_precision: "",
      kohaku_te_stream_resident: "",
      kohaku_vae_memory: "offload",
      kohaku_vae_ahead: 8,
      kohaku_vae_batch: "",
      kohaku_vae_tile: "true",
      kohaku_caption_mode: "nl",
      kohaku_caption_shuffle: true,
      kohaku_caption_tag_dropout: 0.0,
      kohaku_caption_underscores: true,
      kohaku_caption_unescape: true,
      kohaku_text_drop: 0.1,
      kohaku_resolution: 1024,
      kohaku_resolution_mode: "resize",
      kohaku_size_step: 16,
      kohaku_aspect_min: 0.33,
      kohaku_aspect_max: 3.0,
      kohaku_upscale: "never",
      kohaku_batching: "tokens",
      kohaku_max_tokens: 4352,
      kohaku_batch_size: 4,
      kohaku_overlong: "drop_tokens",
      kohaku_max_samples: "",
      kohaku_token_budget: "estimate",
      kohaku_latent_cache: "offline",
      kohaku_text_cache: "offline",
      kohaku_cache_dtype: "bf16",
      kohaku_caption_variants: 4,
      kohaku_flip: false,
      kohaku_crop: "center",
      kohaku_num_workers: 2,
      kohaku_seed: 0,
      kohaku_cache_dir: "",
      kohaku_algo: "lora",
      kohaku_preset: "attn-mlp",
      kohaku_dim: 32,
      kohaku_alpha: 32,
      kohaku_algo_kwargs: "",
      kohaku_dora: false,
      kohaku_dora_detach: "",
      kohaku_dropout_input: 0.0,
      kohaku_dropout_rank: 0.0,
      kohaku_dropout_module: 0.0,
      kohaku_adapter_mode: "bypass",
      kohaku_loraplus_ratio: "",
      kohaku_train_norms: false,
      kohaku_timestep: "moss",
      kohaku_ts_mu: -1.09,
      kohaku_ts_s: 0.95,
      kohaku_ts_shift: 1.0,
      kohaku_ts_cutoff: 0.95,
      kohaku_ts_uniform_mix: 0.01,
      kohaku_loss: "mse",
      kohaku_weighting: "none",
      kohaku_hard_weight: 1.0,
      kohaku_optimizer: "adamw",
      kohaku_lr: 0.0001,
      kohaku_beta1: 0.9,
      kohaku_beta2: 0.95,
      kohaku_eps: "",
      kohaku_weight_decay: 0.01,
      kohaku_wd_rules: "moss",
      kohaku_grad_clip: 1.0,
      kohaku_schedule_mode: "cosine",
      kohaku_warmup: 100,
      kohaku_min_value: 0.1,
      kohaku_ema: "",
      kohaku_step_sync: "step",
      kohaku_optimizer_kwargs: "",
      kohaku_precision: "bf16",
      kohaku_fp16_fix: "",
      kohaku_duration: "steps",
      kohaku_epochs: 10,
      kohaku_save_every_epochs: 1,
      kohaku_steps: 300,
      kohaku_grad_acc: 4,
      kohaku_log_every: 5,
      kohaku_metrics_every: "",
      kohaku_logger: "console",
      kohaku_metrics_level: "train",
      kohaku_run_inspect: true,
      kohaku_save_every: 100,
      kohaku_save_minutes: 30,
      kohaku_keep_last: 3,
      kohaku_final_checkpoint: "auto",
      kohaku_resume: "",
      kohaku_init_from: "",
      kohaku_save_format: "comfy",
      kohaku_save_merged: false,
      kohaku_save_fused_qkv: "",
      kohaku_checkpoint_export: true,
      kohaku_sample_every: 0,
      kohaku_sample_prompts: "",
      kohaku_sample_negative: "",
      kohaku_sample_steps: 50,
      kohaku_sample_cfg: 3.0,
      kohaku_sample_shift: 5.0,
      kohaku_sample_width: 1024,
      kohaku_sample_height: 1024,
      kohaku_sample_seeds: "0",
      kohaku_sample_skip_layers: "22",
      kohaku_sample_batch: "",
      kohaku_sample_columns: "",
      kohaku_sample_at_start: false,
      kohaku_extra_knobs: "",
      // </kohaku-defaults>
    },
    general: {
      resolution_width: 1024, resolution_height_enabled: false, resolution_height: 1024,
      batch_size: 1, mixed_precision: "bf16", full_fp16: false, full_bf16: false, fp8_base: false,
      xformers: false, sdpa: true,
      seed: 42, prior_loss_weight: 1.0, max_data_loader_n_workers: 1,
      max_train_mode: "epochs", max_train_value: 10,
      gradient_checkpointing: true, gradient_accumulation_enabled: false, gradient_accumulation_steps: 1,
      cache_latents: true, cache_latents_to_disk: false,
      lowram: false, highvram: false, no_half_vae: false,
      keep_tokens_separator_enabled: false, keep_tokens_separator: "",
      training_comment_enabled: false, training_comment: "",
      protected_tags_file_enabled: false, protected_tags_file: "",
      vae: "",
    },
    network: {
      algo: "lora", preset: "",
      network_dim: 32, network_alpha: 16,
      conv_dim_enabled: false, conv_dim: 16, conv_alpha: 32,
      min_timestep: 0, max_timestep: 1000,
      train_target: "both",
      network_dropout_enabled: false, network_dropout: 0.1,
      rank_dropout_enabled: false, rank_dropout: 0.1,
      module_dropout_enabled: false, module_dropout: 0.1,
      cache_te_outputs: false, cache_te_to_disk: false,
      dylora_unit: 4,
      bypass_mode: false, use_tucker: false, train_norm: false, dora: false, rescaled: false,
      constraint_enabled: false, constraint: 0,
      lora_fa: false, ip_noise_gamma_enabled: false, ip_noise_gamma: 0.1,
      extra_network_args: [],
      block_weight: { enabled: false, down: "", mid: "", up: "" },
      block_dims: { enabled: false, values: "" },
      block_alphas: { enabled: false, values: "" },
      conv_block_dims: { enabled: false, values: "" },
      conv_block_alphas: { enabled: false, values: "" },
    },
    optimizer: {
      optimizer_type: "AdamW", learning_rate: "1e-4",
      unet_lr_enabled: false, unet_lr: "1e-4",
      te_lr_enabled: false, te_lr: "1e-4",
      lr_scheduler: "cosine", num_cycles: 1, poly_power: 1.0,
      min_lr: 0, gamma: 0.9, d_param: 0.9, decay_ratio: 0.1, decay_type: "1-sqrt",
      timescale: null, min_lr_ratio: null, initial_lr: 0.01,
      lr_scheduler_extra_args: [],
      warmup_ratio_enabled: false, warmup_ratio: 0.05, zero_lr_warmup: false,
      scale_weight_norms_enabled: false, scale_weight_norms: 1.0,
      max_grad_norm: 1.0,
      min_snr_gamma_enabled: false, min_snr_gamma: 5.0,
      zero_terminal_snr: false, masked_loss: false,
      loss_type: "l2", huber_schedule: "snr", huber_param: 0.1,
      extra_optimizer_args: [{ name: "weight_decay", value: "0.1" }],
    },
    saving: {
      output_dir: "", output_name_enabled: false, output_name: "",
      save_precision: "fp16", save_model_as: "safetensors",
      resume_enabled: false, resume: "",
      save_only_last_enabled: false, save_last_mode: "epochs", save_last_value: 4,
      save_ratio_enabled: false, save_ratio: 1,
      save_tag_enabled: false, save_tag_dir: "",
      save_freq_enabled: true, save_freq_mode: "epochs", save_freq_value: 1,
      save_toml_enabled: false, save_toml_dir: "",
      save_state: false, save_last_state_enabled: false, save_last_state_mode: "epochs", save_last_state_value: 1,
    },
    bucket: { enabled: true, no_upscale: true, multires_training: false, min_reso: 256, max_reso: 2048, reso_steps: 64 },
    noise: { noise_offset_enabled: false, noise_offset: 0.1, pyramid_enabled: false, pyramid_iterations: 6, pyramid_discount: 0.3 },
    sample: { enabled: false, sampler: "ddim", freq_mode: "epochs", freq_value: 1, prompts_file: "", prompts_text: "" },
    logging: {
      enabled: true, log_with: "tensorboard", logging_dir: "",
      log_prefix_mode: "disabled", log_prefix: "", run_name_mode: "default", run_name: "",
      tracker_name_enabled: false, tracker_name: "", wandb_api_key: "",
    },
    edm_loss: {
      enabled: false, optimizer_type: "AdamW", learning_rate: "1e-2", optimizer_args: "",
      scheduler_enabled: false, warmup_percent: 0, constant_percent: 0,
      initial_weights: "", num_channels: 128,
    },
    experimental: {
      vae_batch_size_enabled: false, vae_batch_size: 1, vae_reflection: false,
      vae_custom_scale_enabled: false, vae_custom_scale: 0, vae_custom_shift_enabled: false, vae_custom_shift: 0,
      zero_cond_dropout: false, debiased_estimation_loss: false,
    },
    accelerate: { enabled: false, num_processes: 2, main_process_port: 29500 },
    extra_args: [],
    subsets: [],
  };

  // ------------------------------ generic field helpers ------------------------------

  function field(labelText, inputEl, opts = {}) {
    const label = h("label", {}, [labelText]);
    if (opts.hint) label.appendChild(h("span", { class: "hint-dot", title: opts.hint }, "?"));
    const cls = "field" + (opts.span ? ` span-${opts.span}` : "");
    return h("div", { class: cls }, [label, inputEl]);
  }

  function textInput(value, onChange, opts = {}) {
    const el = h("input", { type: "text", placeholder: opts.placeholder || "" });
    el.value = value ?? "";
    el.addEventListener("input", () => onChange(el.value));
    return el;
  }

  function pathInput(value, onChange) {
    const input = textInput(value, onChange, { placeholder: "path" });
    input.classList.add("mono");
    const browseBtn = h("button", { class: "btn small", type: "button", onclick: () => {
      FolderBrowser.open(input.value || null, (path) => { input.value = path; onChange(path); });
    } }, "…");
    return h("div", { class: "path-input" }, [input, browseBtn]);
  }

  function numberInput(value, onChange, opts = {}) {
    const el = h("input", { type: "number", step: opts.step ?? "any", min: opts.min, max: opts.max });
    el.value = value ?? "";
    el.addEventListener("input", () => onChange(el.value === "" ? "" : Number(el.value)));
    return el;
  }

  function selectInput(options, value, onChange) {
    // options: [{value,label}] or [{group,options:[{value,label}]}]
    const el = h("select", {});
    for (const opt of options) {
      if (opt.group) {
        const og = h("optgroup", { label: opt.group });
        for (const o of opt.options) og.appendChild(h("option", { value: o.value }, o.label ?? o.value));
        el.appendChild(og);
      } else {
        el.appendChild(h("option", { value: opt.value }, opt.label ?? opt.value));
      }
    }
    el.value = value;
    el.addEventListener("change", () => onChange(el.value));
    return el;
  }

  function toggleRow(checked, onChange, labelText, subText) {
    const input = h("input", { type: "checkbox" });
    input.checked = !!checked;
    input.addEventListener("change", () => onChange(input.checked));
    const label = h("label", { class: "toggle-row" }, [
      input, h("span", { class: "toggle" }),
      h("span", { class: "label" }, [labelText, subText ? h("small", {}, subText) : null]),
    ]);
    return label;
  }

  function checkboxRow(checked, onChange, labelText) {
    const input = h("input", { type: "checkbox" });
    input.checked = !!checked;
    input.addEventListener("change", () => onChange(input.checked));
    return h("label", { class: "checkbox-row" }, [input, labelText]);
  }

  function pillGroup(options, value, onChange) {
    const wrap = h("div", { class: "pill-group" });
    const btns = options.map((opt) => h("button", {
      type: "button",
      class: opt.value === value ? "active" : "",
      onclick: (e) => {
        wrap.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        e.currentTarget.classList.add("active");
        onChange(opt.value);
      },
    }, opt.label));
    btns.forEach((b) => wrap.appendChild(b));
    return wrap;
  }

  // "checkbox that reveals a field below it" - the pervasive `_enabled` pattern.
  function optionalField(checked, onToggle, labelText, innerEl, opts = {}) {
    const body = h("div", { style: checked ? "" : "display:none", class: "stack" }, [innerEl]);
    const toggle = toggleRow(checked, (v) => {
      onToggle(v);
      body.style.display = v ? "" : "none";
    }, labelText, opts.subText);
    const cls = "field" + (opts.span ? ` span-${opts.span}` : "");
    return h("div", { class: cls }, [toggle, body]);
  }

  function keyValueList(items, opts = {}) {
    const list = h("div", { class: "key-value-list" });
    function renderRows() {
      list.innerHTML = "";
      items.forEach((item, idx) => {
        const nameInput = textInput(item.name, (v) => { item.name = v; }, { placeholder: "name" });
        const valueInput = textInput(item.value, (v) => { item.value = v; }, { placeholder: "value" });
        const row = h("div", { class: "key-value-row" }, [nameInput, valueInput]);
        if (opts.datasetFlag) {
          row.appendChild(checkboxRow(item.dataset, (v) => { item.dataset = v; }, "dataset"));
        }
        row.appendChild(h("button", {
          class: "btn small icon-only", type: "button",
          onclick: () => { items.splice(idx, 1); renderRows(); },
        }, "✕"));
        list.appendChild(row);
      });
    }
    renderRows();
    const addBtn = h("button", {
      class: "btn small", type: "button",
      onclick: () => { items.push({ name: "", value: "", dataset: false }); renderRows(); },
    }, "+ Add");
    return h("div", { class: "stack" }, [list, addBtn]);
  }

  // ------------------------------ section (collapsible card) ------------------------------

  function section(id, title, desc, bodyChildren, opts = {}) {
    const isOpen = opts.open ?? false;
    const sec = h("div", { class: `section${isOpen ? " open" : ""}`, id: `section-${id}` });
    const head = h("div", {
      class: "section-head",
      onclick: () => sec.classList.toggle("open"),
    }, [
      h("svg", { class: "chev", width: "10", height: "10", viewBox: "0 0 10 10", html: "<path d='M2 1l5 4-5 4' stroke='currentColor' stroke-width='1.6' fill='none'/>" }),
      h("span", { class: "title" }, title),
      h("span", { class: "desc" }, desc || ""),
    ]);
    if (opts.pillId) {
      const pill = h("span", { class: "toggle-pill", id: opts.pillId }, "off");
      head.appendChild(pill);
    }
    sec.appendChild(head);
    sec.appendChild(h("div", { class: "section-body" }, bodyChildren));
    return sec;
  }

  function wideSec(sec) { sec.classList.add("wide"); return sec; }

  function sub(title, children, hint) {
    const kids = Array.isArray(children) ? children : [children];
    const head = [h("div", { class: "subgroup-title" }, title)];
    if (hint) head.push(h("div", { class: "subgroup-hint" }, hint));
    return h("div", { class: "subgroup" }, [...head, ...kids.filter(Boolean)]);
  }

  function setPill(pillId, on) {
    const el = document.getElementById(pillId);
    if (!el) return;
    el.textContent = on ? "on" : "off";
    el.classList.toggle("on", on);
  }

  // ------------------------------ Model (Anima) ------------------------------

  function buildModelSection() {
    const m = state.model;
    const typeToggle = h("div", { class: "field span-full" }, [
      h("label", {}, "Model type"),
      pillGroup(
        [{ value: "anima", label: "Anima (DiT)" }, { value: "sdxl", label: "SDXL" }, { value: "moss", label: "MOSS-Image" }],
        state.model_type || "anima",
        (v) => { state.model_type = v; renderAllSections(); refreshPreview(); },
      ),
    ]);

    if (state.model_type === "moss") return buildMossSection(typeToggle);

    if ((state.model_type || "anima") === "sdxl") {
      const grid = h("div", { class: "field-grid cols-3" }, [
        typeToggle,
        field("Base checkpoint", pathInput(m.sdxl_checkpoint, (v) => m.sdxl_checkpoint = v), { span: "full",
          hint: "An SDXL .safetensors checkpoint - bundles both CLIP text encoders already." }),
        field("External VAE", pathInput(m.sdxl_vae, (v) => m.sdxl_vae = v), { span: "full",
          hint: "Optional - leave blank to use the VAE baked into the checkpoint." }),
      ]);
      return section("model", "Model", "SDXL checkpoint, VAE", [grid], { open: true });
    }

    const grid = h("div", { class: "field-grid cols-3" }, [typeToggle]);

    grid.append(
      field("DiT model (Anima checkpoint)", pathInput(m.dit_model, (v) => m.dit_model = v), { span: "full" }),
      field("Qwen3 text encoder", pathInput(m.qwen3_model, (v) => m.qwen3_model = v), { span: "full" }),
      field("VAE", pathInput(m.vae_model, (v) => m.vae_model = v), { span: "full" }),
      field("T5 tokenizer path", pathInput(m.t5_tokenizer_path, (v) => m.t5_tokenizer_path = v),
        { hint: "Optional - leave blank to use the tokenizer bundled with the DiT model." }),
      field("Qwen3 max token length", numberInput(m.qwen3_max_token_length, (v) => m.qwen3_max_token_length = v, { step: 1, min: 1 })),
      field("T5 max token length", numberInput(m.t5_max_token_length, (v) => m.t5_max_token_length = v, { step: 1, min: 1 })),
    );

    const sigmoidScaleField = field("Sigmoid scale", numberInput(m.sigmoid_scale, (v) => m.sigmoid_scale = v, { step: 0.1 }));
    const flowShiftField = field("Discrete flow shift", numberInput(m.discrete_flow_shift, (v) => m.discrete_flow_shift = v, { step: 0.1 }));
    function syncTimestepFields() {
      sigmoidScaleField.style.display = ["sigmoid", "shift", "flux_shift"].includes(m.timestep_sampling) ? "" : "none";
      flowShiftField.style.display = ["sigma", "shift"].includes(m.timestep_sampling) ? "" : "none";
    }
    const samplingOptions = (schemaData.timestep_sampling || ["sigmoid", "shift", "sigma", "flux_shift", "uniform"])
      .map((v) => ({ value: v, label: v }));
    const samplingSelect = selectInput(samplingOptions, m.timestep_sampling, (v) => { m.timestep_sampling = v; syncTimestepFields(); });
    grid.append(field("Timestep sampling", samplingSelect), sigmoidScaleField, flowShiftField);
    syncTimestepFields();

    grid.append(
      field("VAE chunk size", numberInput(m.vae_chunk_size, (v) => m.vae_chunk_size = v, { step: 1, min: 0 }),
        { hint: "0 disables chunking." }),
      field("Blocks to swap", numberInput(m.blocks_to_swap, (v) => m.blocks_to_swap = v, { step: 1, min: 0 }),
        { hint: "Offloads N transformer blocks to CPU to reduce VRAM use. 0 disables it." }),
    );

    const toggles = h("div", { class: "field-grid cols-3" }, [
      toggleRow(m.vae_disable_cache, (v) => m.vae_disable_cache = v, "Disable VAE cache"),
      toggleRow(m.flash_attn, (v) => m.flash_attn = v, "Flash attention"),
      toggleRow(m.split_attn, (v) => m.split_attn = v, "Split attention", "Auto-enabled if xFormers is selected below."),
      toggleRow(m.unsloth_offload_checkpointing, (v) => m.unsloth_offload_checkpointing = v, "Unsloth offload checkpointing"),
    ]);

    return section("model", "Model", "DiT checkpoint, text encoder, VAE", [grid, toggles], { open: true });
  }

  // ------------------------------ Model (MOSS-Image) ------------------------------

  let mossProfiles = null;           // {key: {label, model, general, note, ...}} from /api/train/moss/profiles
  let mossProfilesLoading = false;

  function applyMossProfile(key) {
    const p = mossProfiles && mossProfiles[key];
    if (!p) return;
    Object.assign(state.model, p.model, { moss_vram_profile: key });
    Object.assign(state.general, p.general);
  }

  function buildMossSection(typeToggle) {
    const m = state.model;
    if (m.moss_engine === "kohaku") return buildKohakuSection(typeToggle);
    if (!mossProfiles && !mossProfilesLoading) {
      mossProfilesLoading = true;
      Api.mossProfiles().then((d) => { mossProfiles = d.profiles; })
        .catch(() => {}).finally(() => { mossProfilesLoading = false; renderAllSections(); });
    }
    const profileKeys = mossProfiles ? Object.keys(mossProfiles) : ["8", "10", "10-int8", "12", "16", "24", "24-fast"];
    const profileLabel = (k) => (mossProfiles && mossProfiles[k] ? mossProfiles[k].label : k);
    const profileSelect = selectInput(profileKeys.map((k) => ({ value: k, label: profileLabel(k) })),
      m.moss_vram_profile in (mossProfiles || {}) || !mossProfiles ? m.moss_vram_profile : "8",
      (v) => { applyMossProfile(v); renderAllSections(); refreshPreview(); });
    const note = h("div", { class: "subgroup-hint" },
      (mossProfiles && mossProfiles[m.moss_vram_profile] && mossProfiles[m.moss_vram_profile].note) || "");
    const suggestBtn = h("button", { class: "btn small", type: "button", onclick: async () => {
      try {
        const g = await Api.monitorGpu();
        const gpu = g && g.available ? (g.gpus || [])[0] : null;
        if (!gpu) return toast("No NVIDIA GPU detected - pick a profile by hand.", "info");
        const gb = gpu.mem_total_mb / 1024;
        const d = await Api.mossProfiles(gb);
        mossProfiles = d.profiles;
        applyMossProfile(d.suggested);
        renderAllSections(); refreshPreview();
        toast(`${gpu.name} (${gb.toFixed(0)} GB): applied the "${profileLabel(d.suggested)}" profile.`, "success");
      } catch (err) { toast(err.message, "error"); }
    } }, "Pick for my GPU");

    const profileRow = h("div", { class: "field span-full" }, [
      h("label", {}, ["VRAM profile ", h("span", { class: "hint-dot", title: "Sets Blocks to swap, INT8, compile, attention and how the 9B text encoder is loaded for your card. Every profile except the INT8 one trains the full bf16 DiT. You can still change any value below." }, "?")]),
      h("div", { class: "row", style: "gap:8px;align-items:center" }, [profileSelect, suggestBtn]),
      note,
    ]);

    const grid = h("div", { class: "field-grid cols-3" }, [
      typeToggle, engineToggle(), profileRow,
      field("MOSS-Image model (folder or DiT file)", pathInput(m.moss_model, (v) => m.moss_model = v), { span: "full",
        hint: "The OpenMOSS-Team/MOSS-Image-Base folder (transformer/, text_encoder/, tokenizer/, vae/). Base is recommended; LoRAs trained on Base also work on SFT. Or the path of a single bf16 DiT .safetensors file - then fill in the Text encoder, Tokenizer folder and VAE fields below yourself (file names do not matter). Quantized checkpoints cannot be trained." }),
      field("Text encoder", pathInput(m.moss_text_encoder, (v) => m.moss_text_encoder = v),
        { hint: "Qwen3.5-9B in bf16: either the HF text_encoder/ folder, or ONE bf16 .safetensors file (ComfyUI export - no config.json needed; quantized/int8 files are refused). Empty only if the model is a pipeline folder (then its text_encoder/ is used)." }),
      field("Tokenizer folder (optional)", pathInput(m.moss_tokenizer, (v) => m.moss_tokenizer = v),
        { hint: "The tokenizer/ folder of OpenMOSS-Team/MOSS-Image-Base (a few MB). Optional: if empty, a single-file text encoder downloads just the tokenizer (a few MB) once." }),
      field("VAE", pathInput(m.moss_vae, (v) => m.moss_vae = v),
        { hint: "Empty only if the model is a pipeline folder (then its vae/ is used - recommended, its statistics define the latent space)." }),
      field("Max caption tokens", numberInput(m.moss_max_token_length, (v) => m.moss_max_token_length = v, { step: 32, min: 32, max: 4096 }),
        { hint: "Qwen3.5 tokens including the chat template. Natural-language captions are long; 512 fits most." }),
      field("LoRA key format", selectInput([{ value: "both" }, { value: "split" }, { value: "fused" }], m.moss_lora_save_format, (v) => m.moss_lora_save_format = v),
        { hint: "both = split q/k/v plus fused qkv keys, so ComfyUI/Forge apply it on any MOSS-Image checkpoint." }),
    ]);

    // timestep sampling
    const tsFields = h("div", { class: "field-grid cols-3" });
    const flowShift = field("Flow shift", numberInput(m.moss_discrete_flow_shift, (v) => m.moss_discrete_flow_shift = v, { step: 0.1, min: 0.01 }),
      { hint: "MOSS-Image trains with 1.0; its inference shift of 5.0 is applied only when sampling." });
    const sigmoid = field("Sigmoid scale", numberInput(m.moss_sigmoid_scale, (v) => m.moss_sigmoid_scale = v, { step: 0.1 }));
    const mossTs = [
      field("mu", numberInput(m.moss_ts_mu, (v) => m.moss_ts_mu = v, { step: 0.01 })),
      field("s", numberInput(m.moss_ts_s, (v) => m.moss_ts_s = v, { step: 0.01, min: 0.01 })),
      field("cutoff", numberInput(m.moss_ts_cutoff, (v) => m.moss_ts_cutoff = v, { step: 0.01, min: 0.05, max: 1 })),
      field("uniform mix", numberInput(m.moss_ts_uniform_mix, (v) => m.moss_ts_uniform_mix = v, { step: 0.01, min: 0, max: 1 })),
    ];
    const syncTs = () => {
      mossTs.forEach((f) => { f.style.display = m.moss_timestep_sampling === "moss" ? "" : "none"; });
      flowShift.style.display = ["moss", "sigma", "shift"].includes(m.moss_timestep_sampling) ? "" : "none";
      sigmoid.style.display = ["sigmoid", "shift", "flux_shift"].includes(m.moss_timestep_sampling) ? "" : "none";
    };
    const tsSelect = selectInput(["moss", "sigmoid", "uniform", "sigma", "shift", "flux_shift"].map((v) => ({ value: v })),
      m.moss_timestep_sampling, (v) => { m.moss_timestep_sampling = v; syncTs(); });
    tsFields.append(field("Timestep sampling", tsSelect, { hint: "moss = the MOSS-Image training recipe (biased toward high noise). Recommended." }),
      flowShift, sigmoid, ...mossTs);
    syncTs();

    // memory + speed
    const memGrid = h("div", { class: "field-grid cols-3" }, [
      field("Blocks to swap (0-40)", numberInput(m.moss_blocks_to_swap, (v) => { m.moss_blocks_to_swap = v; if (v > 0 && m.moss_int8 !== "off") { m.moss_int8 = "off"; renderAllSections(); } }, { step: 1, min: 0, max: 40 }),
        { hint: "Parks N of the 42 blocks (~0.23 GB each) in system RAM. Roughly 0.23 GB of VRAM saved per block, at the cost of speed. 36 trains the full bf16 DiT in ~6 GB." }),
      field("Unchecked blocks", numberInput(m.moss_unchecked_blocks, (v) => m.moss_unchecked_blocks = v, { step: 1, min: 0, max: 42 }),
        { hint: "Skip gradient checkpointing for the last N blocks: slightly faster, ~1 GB more VRAM each at 1024 px." }),
      field("Sample CFG skip layers", textInput(m.moss_sample_skip_layers, (v) => m.moss_sample_skip_layers = v, { placeholder: "22" }),
        { hint: "Blocks skipped in the negative pass of sample images (official default 22). Empty = plain CFG." }),
      field("Attention", selectInput(["auto", "cudnn", "efficient", "flash", "torch"].map((v) => ({ value: v })), m.moss_attn_backend, (v) => m.moss_attn_backend = v),
        { hint: "cudnn is ~2.4x faster but needs cuDNN 9.24+ (the environment installs torch 2.14.1+cu130); the Backend tab's check tells you." }),
      field("Compile", selectInput(["none", "situ", "blocks"].map((v) => ({ value: v })), m.moss_compile, (v) => m.moss_compile = v),
        { hint: "torch.compile (needs Triton; on Windows Visual Studio Build Tools). blocks is fastest; with Blocks to swap it falls back to situ." }),
      field("INT8 base", selectInput([{ value: "off", label: "off (full bf16)" }, { value: "on", label: "on" }, { value: "on_bwd", label: "on (int8 backward)" }],
        m.moss_int8, (v) => { m.moss_int8 = v; if (v !== "off" && m.moss_blocks_to_swap > 0) { m.moss_blocks_to_swap = 0; renderAllSections(); } }),
        { hint: "Quantizes the frozen base to INT8: faster and ~4.5 GB lighter, but the LoRA learns against a quantized base and it cannot be combined with Blocks to swap." }),
    ]);
    const toggles = h("div", { class: "field-grid cols-3" }, [
      toggleRow(m.moss_fp32_modulation, (v) => m.moss_fp32_modulation = v, "FP32 modulation", "Keeps the timestep MLP in fp32. Negligible cost."),
      toggleRow(m.moss_vae_sample_posterior, (v) => m.moss_vae_sample_posterior = v, "Sample VAE posterior", "Instead of its mode, when caching latents."),
    ]);

    // text encoder loading
    const teGpu = field("Text-encoder GPU budget (GB)", numberInput(m.moss_te_gpu_gb, (v) => m.moss_te_gpu_gb = v, { step: 0.5, min: 1 }),
      { hint: "How much VRAM the 9B text encoder may use while captions are cached. The rest of its layers stream from RAM." });
    const teCpu = field("Text-encoder RAM budget (GB)", numberInput(m.moss_te_cpu_gb, (v) => m.moss_te_cpu_gb = v, { step: 1, min: 8 }));
    const syncTe = () => { const off = m.moss_te_mode === "offload"; teGpu.style.display = off ? "" : "none"; teCpu.style.display = off ? "" : "none"; };
    const teSelect = selectInput([
      { value: "full", label: "full - bf16 on the GPU (needs ~18 GB)" },
      { value: "offload", label: "offload - exact bf16, streamed from RAM (8-16 GB cards)" },
      { value: "nf4", label: "nf4 - 4-bit on the GPU (fast, slightly different embeddings)" },
      { value: "int8", label: "int8 - 8-bit on the GPU (slightly different embeddings)" },
    ], m.moss_te_mode, (v) => { m.moss_te_mode = v; syncTe(); });
    const teGrid = h("div", { class: "field-grid cols-3" }, [
      field("Text encoder loading", teSelect, { span: "full",
        hint: "The 9B Qwen3.5 text encoder only runs while captions are cached, before the DiT loads. offload keeps the exact bf16 weights, so nothing about the training changes - it is just slower (once per dataset) and needs ~20 GB of free RAM. nf4/int8 are quick but encode captions slightly differently from the bf16 model that inference uses." }),
      teGpu, teCpu,
    ]);
    syncTe();

    return section("model", "Model · MOSS-Image", "5B DiT, Qwen3.5-9B text encoder, FLUX.2 VAE", [
      grid,
      sub("Timestep sampling", tsFields),
      sub("Memory & speed", [memGrid, toggles]),
      sub("Text encoder (caption caching)", teGrid,
        "Text-encoder outputs are always cached for MOSS-Image, so caption shuffling, tag dropout and token warmup are ignored (caption dropout still works)."),
    ], { open: true });
  }

  // ------------------------------ Model (MOSS-Image · low-VRAM trainer) ------------------------------

  let kohakuProfiles = null;         // {key: {label, model, note, min_vram_gb}} from /api/train/kohaku/profiles
  let kohakuProfilesLoading = false;
  let kohakuParts = null;            // the cards of the low-VRAM engine, built once per render and placed on the tabs

  function engineToggle() {
    return h("div", { class: "field span-full" }, [
      h("label", {}, ["Trainer engine ", h("span", { class: "hint-dot", title: "Both engines train MOSS-Image and share your dataset folders and output folder. Standard = the bundled sd-scripts fork. Low-VRAM = a second trainer built for small cards (its own environment, set up on the Backend tab): trains straight on the int8 files and streams weights, so 6-12 GB cards work. Try the same data with each and compare." }, "?")]),
      pillGroup(
        [{ value: "fork", label: "Standard trainer (sd-scripts)" }, { value: "kohaku", label: "Low-VRAM trainer (6 GB+)" }],
        state.model.moss_engine || "fork",
        (v) => { state.model.moss_engine = v; renderAllSections(); refreshPreview(); },
      ),
    ]);
  }

  function applyKohakuProfile(key) {
    const p = kohakuProfiles && kohakuProfiles[key];
    if (!p) return;
    Object.assign(state.model, p.model, { kohaku_vram: key });
  }

  // one low-VRAM trainer option -> a labelled input bound to state.model[key]
  function kf(key, label, kind, opts = {}) {
    const m = state.model;
    const set = (v) => { m[key] = v; if (opts.after) opts.after(); };
    let input;
    if (kind === "bool") return toggleRow(m[key], set, label, opts.hint);
    if (kind === "path") input = pathInput(m[key], set);
    else if (kind === "num") input = numberInput(m[key], set, { step: opts.step, min: opts.min, max: opts.max });
    else if (kind === "select") input = selectInput((opts.options || []).map((o) => (typeof o === "object" ? o : { value: o, label: o === "" ? "default" : o })), m[key], set);
    else if (kind === "area") {
      input = h("textarea", { rows: opts.rows || 3, placeholder: opts.placeholder || "", class: "mono" });
      input.value = m[key] ?? "";
      input.addEventListener("input", () => set(input.value));
    } else input = textInput(m[key], set, { placeholder: opts.placeholder || "" });
    return field(label, input, { hint: opts.hint, span: opts.span });
  }

  function buildKohakuSection(typeToggle) {
    const m = state.model;
    if (!kohakuProfiles && !kohakuProfilesLoading) {
      kohakuProfilesLoading = true;
      Api.kohakuProfiles().then((d) => { kohakuProfiles = d.profiles; })
        .catch(() => {}).finally(() => { kohakuProfilesLoading = false; renderAllSections(); });
    }
    const keys = kohakuProfiles ? Object.keys(kohakuProfiles) : ["6", "8", "12", "16", "24", "auto"];
    const label = (k) => (kohakuProfiles && kohakuProfiles[k] ? kohakuProfiles[k].label : k);
    const profileSelect = selectInput(keys.map((k) => ({ value: k, label: label(k) })), m.kohaku_vram,
      (v) => { applyKohakuProfile(v); renderAllSections(); refreshPreview(); });
    const note = h("div", { class: "subgroup-hint" }, (kohakuProfiles && kohakuProfiles[m.kohaku_vram] && kohakuProfiles[m.kohaku_vram].note) || "");
    const suggestBtn = h("button", { class: "btn small", type: "button", onclick: async () => {
      try {
        const g = await Api.monitorGpu();
        const gpu = g && g.available ? (g.gpus || [])[0] : null;
        if (!gpu) return toast("No NVIDIA GPU detected - pick a profile by hand.", "info");
        const gb = gpu.mem_total_mb / 1024;
        const d = await Api.kohakuProfiles(gb);
        kohakuProfiles = d.profiles;
        applyKohakuProfile(d.suggested);
        renderAllSections(); refreshPreview();
        toast(`${gpu.name} (${gb.toFixed(0)} GB): applied the "${label(d.suggested)}" layer.`, "success");
      } catch (err) { toast(err.message, "error"); }
    } }, "Pick for my GPU");

    // ---- model files
    const checkOut = h("div", { class: "subgroup-hint", style: "white-space:pre-wrap" }, "");
    const checkBtn = h("button", { class: "btn small", type: "button", onclick: async () => {
      try {
        const r = await Api.kohakuCheck({ weights: m.kohaku_weights, variant: m.kohaku_variant, transformer: m.kohaku_transformer,
          text_encoder: m.kohaku_text_encoder, vae: m.kohaku_vae });
        checkOut.textContent = r.problems && r.problems.length
          ? "Problem: " + r.problems.join("\n")
          : `Layout: ${r.layout}\nDiT: ${r.dit}\nText encoder: ${r.te}\nVAE: ${r.vae}`;
      } catch (err) { toast(err.message, "error"); }
    } }, "Check these files");
    const files = h("div", { class: "field-grid cols-3" }, [
      typeToggle, engineToggle(),
      field("Weights folder", pathInput(m.kohaku_weights, (v) => m.kohaku_weights = v), { span: "full",
        hint: "A ComfyUI models folder (diffusion_models/, text_encoders/, vae/ - the Comfy-Org/MOSS-Image repack; its int8 ConvRot files train as they are) or an OpenMOSS-Team/MOSS-Image-Base pipeline folder (transformer/, text_encoder/, vae/). A single DiT .safetensors also works if you fill in the Text encoder and VAE below. Taken exactly as typed." }),
      kf("kohaku_variant", "Variant", "select", { options: ["base", "sft"], hint: "base is recommended; LoRAs trained on Base also work on SFT." }),
      kf("kohaku_transformer", "DiT file (optional)", "text", { placeholder: "empty = the default file in the weights folder", span: "full",
        hint: "A file name under diffusion_models/ (with or without .safetensors) or any path. Empty picks moss_image_<variant>_int8_convrot, else _bf16." }),
      kf("kohaku_text_encoder", "Text encoder (optional)", "text", { placeholder: "empty = qwen3.5_9b_int8_convrot, else _bf16", span: "full",
        hint: "A file name under text_encoders/, any path, or a Qwen3.5 folder. No config.json or tokenizer needed: the trainer ships both." }),
      kf("kohaku_vae", "VAE (optional)", "text", { placeholder: "empty = flux2-vae_small, else flux2-vae", span: "full" }),
      h("div", { class: "field span-full" }, [h("div", { class: "row", style: "gap:8px;align-items:center" }, [checkBtn]), checkOut]),
    ]);

    // ---- hardware / memory
    const manual = h("div", { class: "field-grid cols-3" }, [
      kf("kohaku_checkpoint", "Checkpointing", "text", { placeholder: "block | sub-block | selective | none" }),
      kf("kohaku_stream", "Weight streaming", "select", { options: ["", "off", "ring", "swap"] }),
      kf("kohaku_stream_resident", "Resident blocks", "num", { min: 0, step: 1 }),
      kf("kohaku_optim_state", "Optimizer state", "select", { options: ["", "gpu", "stream", "cpu"] }),
      kf("kohaku_pinned", "Pinned host memory", "select", { options: ["", "true", "false"] }),
      kf("kohaku_act_offload", "Activation offload", "select", { options: ["", "true", "false"] }),
      kf("kohaku_grad_offload", "Gradient offload", "select", { options: ["", "true", "false"] }),
      kf("kohaku_fused_step", "Fused step", "select", { options: ["", "true", "false"] }),
    ]);
    const syncManual = () => { manual.style.display = m.kohaku_memory_profile === "manual" ? "" : "none"; };
    syncManual();
    const memGrid = h("div", { class: "field-grid cols-3" }, [
      h("div", { class: "field span-full" }, [
        h("label", {}, ["VRAM layer ", h("span", { class: "hint-dot", title: "Presets for 6 / 8 / 12 / 16 / 24 GB cards. A card gets the largest layer it holds within 5 %, so a 10 GB card gets the 8 GB layer (its allocator is capped at 8 GB in the authors' tests). Applying one fills the fields below; you can still change any of them." }, "?")]),
        h("div", { class: "row", style: "gap:8px;align-items:center" }, [profileSelect, suggestBtn]), note,
      ]),
      kf("kohaku_memory_profile", "Memory profile", "text", { placeholder: "minimal", after: syncManual,
        hint: "A profile name: speed, balanced, low, minimal, auto, a catalog name such as ckpt-ring2-actoff-optcpu-gradoff, or manual (then set the streaming fields shown below)." }),
      kf("kohaku_base_precision", "Base precision", "select", { options: ["", "auto", "compute", "int8", "fp8", "fp8_compute", "mxfp8"],
        hint: "auto (default) keeps what the DiT file stores: an int8 ConvRot file trains on its int8 weights. int8 quantizes a bf16 file while loading. compute dequantizes to 16 bit." }),
      kf("kohaku_precision", "Compute precision", "select", { options: ["bf16", "fp16"], hint: "bf16 on RTX 30-series and newer. fp16 only for older cards (uses the online residual fix)." }),
      kf("kohaku_fp16_fix", "FP16 overflow fix", "select", { options: ["", "online", "weights", "fp32_residual", "none"],
        hint: "Only used with fp16 compute. Empty keeps the fp16 layer's default (online, c = 1/16); weights folds the scales into the weights; fp32_residual keeps the residual stream in fp32." }),
      kf("kohaku_int8_compute", "INT8 compute", "bool", { hint: "Int8 bases run int8 matmuls instead of weight-only (CUDA)." }),
      kf("kohaku_int8_dgrad", "INT8 input gradient", "select", { options: ["", "int8", "16bit"] }),
      kf("kohaku_fp8_recipe", "FP8 scaling", "select", { options: ["", "tensor", "row", "block"] }),
      kf("kohaku_fp8_dgrad", "FP8 input gradient", "select", { options: ["", "fp8", "16bit"] }),
      kf("kohaku_compile", "torch.compile", "select", { options: ["", "off", "block", "model"] }),
      kf("kohaku_tune", "Kernel tuning", "select", { options: ["table", "off"], hint: "table times the kernels once per card before step 0 and stores the result; off skips all timing (quick smoke run)." }),
      kf("kohaku_step_release", "Release cached VRAM", "text", { placeholder: "none | empty_cache | N", hint: "none keeps the allocator's cache between steps; empty_cache frees it every step (slower); N every N steps." }),
    ]);
    const placement = h("div", { class: "field-grid cols-3" }, [
      kf("kohaku_te_memory", "Text encoder placement", "select", { options: ["", "auto", "resident", "ring", "offload", "cpu"],
        hint: "ring streams its layers through the GPU (about 2 GB at a time) and frees them between passes." }),
      kf("kohaku_te_ahead", "Batches per pass", "num", { min: 1, step: 1 }),
      kf("kohaku_te_batch", "Prompts per call", "num", { min: 1, step: 1 }),
      kf("kohaku_te_max_tokens", "Tokens per call", "num", { min: 256, step: 256 }),
      kf("kohaku_te_precision", "Text encoder precision", "select", { options: ["", "auto", "compute", "int8", "fp8", "fp8_compute", "mxfp8"] }),
      kf("kohaku_te_stream_resident", "Ring slots", "num", { min: 1, step: 1 }),
      kf("kohaku_text_device", "Text encoder device", "text", { placeholder: "empty = the training GPU; cpu", hint: "cpu only with the offline text cache." }),
      kf("kohaku_vae_memory", "VAE placement", "select", { options: ["", "auto", "resident", "offload", "ring", "cpu"] }),
      kf("kohaku_vae_ahead", "VAE batches per pass", "num", { min: 1, step: 1 }),
      kf("kohaku_vae_batch", "Images per VAE call", "num", { min: 1, step: 1 }),
      kf("kohaku_vae_tile", "VAE tiling", "select", { options: ["", "true", "false"], hint: "Tiled encode saves VRAM on big images." }),
      kf("kohaku_vae_dtype", "VAE precision", "select", { options: ["auto", "bf16", "fp16", "fp32"] }),
    ]);

    // ---- adapter
    const algos = ["lora", "locon", "loha", "lokr", "oft", "boft", "glora", "ia3", "dylora", "tlora", "dora", "full"];
    const ALGO_DEFAULTS = {
      lora: { kohaku_dim: 32, kohaku_alpha: 32, kohaku_lr: 1e-4 }, locon: { kohaku_dim: 32, kohaku_alpha: 32, kohaku_lr: 1e-4 },
      loha: { kohaku_dim: 16, kohaku_alpha: 8, kohaku_lr: 2e-4 }, lokr: { kohaku_dim: 10000, kohaku_alpha: 1, kohaku_lr: 2e-4 },
      oft: { kohaku_dim: 32, kohaku_alpha: 0, kohaku_lr: 1e-4 }, dora: { kohaku_dim: 16, kohaku_alpha: 16, kohaku_lr: 1e-4 },
      full: { kohaku_lr: 5e-5, kohaku_optimizer: "muon", kohaku_schedule_mode: "cosine", kohaku_warmup: 500, kohaku_ema: 0.9999 },
    };
    const adapter = h("div", { class: "field-grid cols-3" }, [
      field("Algorithm", selectInput(algos.map((v) => ({ value: v })), m.kohaku_algo, (v) => {
        m.kohaku_algo = v; Object.assign(m, ALGO_DEFAULTS[v] || {}); renderAllSections(); refreshPreview();
      }), { hint: "Choosing an algorithm fills its rank, alpha and learning-rate defaults from its presets. full = full finetune (needs a 24 GB card with host streaming)." }),
      kf("kohaku_preset", "Target modules", "text", { placeholder: "attn-mlp", hint: "attn | attn-mlp | full-block | all, plus +text-refiner +modulation +embed, or a LyCORIS-style toml path." }),
      kf("kohaku_dim", "Rank (DIM)", "num", { min: 1, step: 1, hint: "For OFT this is the block size; LoKr uses 10000 for full matrices." }),
      kf("kohaku_alpha", "Alpha", "num", { step: 0.5 }),
      kf("kohaku_adapter_mode", "Adapter mode", "select", { options: ["bypass", "merge"] }),
      kf("kohaku_loraplus_ratio", "LoRA+ ratio", "num", { step: 0.5, min: 1, hint: "Learning-rate ratio of the up / down factors; empty = off." }),
      kf("kohaku_dropout_input", "Dropout", "num", { step: 0.01, min: 0, max: 1 }),
      kf("kohaku_dropout_rank", "Rank dropout", "num", { step: 0.01, min: 0, max: 1 }),
      kf("kohaku_dropout_module", "Module dropout", "num", { step: 0.01, min: 0, max: 1 }),
      kf("kohaku_algo_kwargs", "Algorithm arguments (JSON)", "text", { span: "full", placeholder: '{"factor": 8}',
        hint: "Passed unchanged to the LyCORIS module, e.g. {\"factor\": 8} for LoKr or {\"use_tucker\": true}. Empty keeps the preset's." }),
      kf("kohaku_dora", "DoRA (weight decomposition)", "bool"),
      kf("kohaku_dora_detach", "DoRA: detach the norm", "select", { options: ["", "true", "false"] }),
      kf("kohaku_train_norms", "Train norm layers", "bool"),
      kf("kohaku_train_text_refiner", "Also train the 2 text-refiner blocks", "bool"),
    ]);

    // ---- optimiser
    const optim = h("div", { class: "field-grid cols-3" }, [
      kf("kohaku_optimizer", "Optimizer", "select", { options: ["adamw", "adamw16", "adamw8bit", "adamw8bit_kernel", "muon", "prodigy", "schedule_free"] }),
      kf("kohaku_lr", "Learning rate", "num", { step: "any", min: 0 }),
      kf("kohaku_weight_decay", "Weight decay", "num", { step: 0.001, min: 0 }),
      kf("kohaku_beta1", "Beta 1", "num", { step: 0.01 }),
      kf("kohaku_beta2", "Beta 2", "num", { step: 0.01 }),
      kf("kohaku_eps", "Epsilon", "num", { step: "any", hint: "empty = 1e-8" }),
      kf("kohaku_wd_rules", "Weight-decay rules", "select", { options: ["moss", "none", "all"] }),
      kf("kohaku_grad_clip", "Gradient clip", "num", { step: 0.1, min: 0 }),
      kf("kohaku_ema", "EMA", "num", { step: 0.0001, min: 0, max: 1, hint: "empty or 0 = off; e.g. 0.9999." }),
      kf("kohaku_schedule_mode", "LR schedule", "select", { options: ["cosine", "constant"] }),
      kf("kohaku_warmup", "Warmup steps", "num", { min: 0, step: 1 }),
      kf("kohaku_min_value", "Final LR fraction", "num", { min: 0, max: 1, step: 0.05, hint: "Cosine only: the learning rate ends at this fraction of the peak." }),
      kf("kohaku_step_sync", "Step sync", "select", { options: ["step", "deferred"] }),
      kf("kohaku_optimizer_kwargs", "Optimizer arguments (JSON)", "text", { span: "full", placeholder: '{"weight_decay": 0.01}' }),
    ]);

    // ---- objective
    const tsFields = [
      kf("kohaku_ts_mu", "mu", "num", { step: 0.01 }), kf("kohaku_ts_s", "s", "num", { step: 0.01, min: 0.01 }),
      kf("kohaku_ts_shift", "shift", "num", { step: 0.1, min: 0.01 }), kf("kohaku_ts_cutoff", "cutoff", "num", { step: 0.01, min: 0.05, max: 1 }),
      kf("kohaku_ts_uniform_mix", "uniform mix", "num", { step: 0.01, min: 0, max: 1 }),
    ];
    const syncTs = () => tsFields.forEach((f) => { f.style.display = m.kohaku_timestep === "moss" ? "" : "none"; });
    const objective = h("div", { class: "field-grid cols-3" }, [
      kf("kohaku_timestep", "Timestep sampling", "select", { options: ["moss", "uniform", "logit_normal"], after: syncTs,
        hint: "moss = the MOSS-Image training recipe (biased toward high noise). Recommended." }),
      ...tsFields,
      kf("kohaku_loss", "Loss", "select", { options: ["mse", "huber", "l1"] }),
      kf("kohaku_weighting", "Loss weighting", "select", { options: ["none", "x0", "eps"] }),
      kf("kohaku_hard_weight", "Data loss weight", "num", { step: 0.1, min: 0 }),
    ]);
    syncTs();

    // ---- data
    const data = h("div", { class: "field-grid cols-3" }, [
      kf("kohaku_caption_mode", "Caption style", "select", { options: [{ value: "nl", label: "Natural language" }, { value: "tags", label: "Tags (comma separated)" }],
        hint: "MOSS-Image reads captions with a Qwen3.5 language model, so natural language is the usual choice. Captions come from the .txt file next to each image." }),
      kf("kohaku_caption_shuffle", "Shuffle tags", "bool", { hint: "Tags style only." }),
      kf("kohaku_caption_tag_dropout", "Tag dropout", "num", { step: 0.05, min: 0, max: 1, hint: "Tags style only." }),
      kf("kohaku_caption_underscores", "Underscores to spaces", "bool"),
      kf("kohaku_caption_unescape", "Unescape \\( \\)", "bool"),
      kf("kohaku_text_drop", "Empty-caption chance", "num", { step: 0.01, min: 0, max: 1, hint: "Probability a sample trains on the empty prompt (caption dropout)." }),
      kf("kohaku_resolution", "Resolution (px side)", "num", { step: 64, min: 256, hint: "The target area is side x side; the aspect ratio of each image is kept." }),
      kf("kohaku_resolution_mode", "Resolution mode", "select", { options: ["resize", "bucket", "shifted_square", "square_crop"] }),
      kf("kohaku_size_step", "Size step", "num", { step: 16, min: 16 }),
      kf("kohaku_aspect_min", "Aspect min", "num", { step: 0.01 }),
      kf("kohaku_aspect_max", "Aspect max", "num", { step: 0.01 }),
      kf("kohaku_upscale", "Upscale small images", "select", { options: ["never", "to_area"] }),
      kf("kohaku_batching", "Batching", "select", { options: ["tokens", "tokens_ffd", "count"], hint: "tokens packs images until the token budget below is full (no padding waste); count uses a fixed batch size." }),
      kf("kohaku_max_tokens", "Tokens per batch", "num", { step: 256, min: 512, hint: "Image + caption tokens per batch. 4352 = one 1 MP image; the VRAM layer sets it." }),
      kf("kohaku_batch_size", "Batch size (count mode)", "num", { step: 1, min: 1 }),
      kf("kohaku_overlong", "Too-long samples", "select", { options: ["drop", "truncate_text", "drop_tokens"] }),
      kf("kohaku_max_samples", "Max samples per batch", "num", { step: 1, min: 0 }),
      kf("kohaku_token_budget", "Token budget", "select", { options: ["estimate", "render", "exact"] }),
      kf("kohaku_latent_cache", "Latent cache", "select", { options: ["offline", "online", "off"], hint: "offline encodes every image once before training, so the VAE is not held while training." }),
      kf("kohaku_text_cache", "Text cache", "select", { options: ["offline", "online", "off"], hint: "offline encodes every caption once before training, then frees the 9B text encoder." }),
      kf("kohaku_cache_dtype", "Cache precision", "select", { options: ["bf16", "fp16", "fp32", "fp8"] }),
      kf("kohaku_caption_variants", "Caption variants per image", "num", { step: 1, min: 1 }),
      kf("kohaku_flip", "Horizontal flip", "bool"),
      kf("kohaku_crop", "Crop", "select", { options: ["center", "random"], hint: "Offline latent caches need center." }),
      kf("kohaku_num_workers", "Data workers", "num", { step: 1, min: 0 }),
      kf("kohaku_seed", "Seed", "num", { step: 1 }),
      kf("kohaku_cache_dir", "Cache folder (optional)", "path", { span: "full", hint: "Empty = <output folder>/kohaku_cache." }),
    ]);

    // ---- duration (steps or epochs)
    const epochMode = m.kohaku_duration === "epochs";
    const perBatch = (() => {
      const px = Number(m.kohaku_resolution) || 1024, tokens = Number(m.kohaku_max_tokens) || 4352;
      return m.kohaku_batching === "count" ? Math.max(1, Number(m.kohaku_batch_size) || 1)
        : Math.max(1, Math.floor(tokens / (px * px / 256 + 200)));
    })();
    const acc = Number(m.kohaku_grad_acc) || 1;
    const stepsHint = `About ${perBatch} image${perBatch === 1 ? "" : "s"} per batch x ${acc} accumulation = ${perBatch * acc} images per step (repeats counted). A step is one optimizer update.`;
    const durationToggle = h("div", { class: "field span-full" }, [
      h("label", {}, ["Train for ", h("span", { class: "hint-dot", title: "Steps count optimizer updates. Epochs count passes over your whole dataset (images x repeats); the trainer itself only counts steps, so the epoch number is converted to steps when training starts, from your dataset size, batch size and accumulation. Batches are packed by token budget, so the result is a close estimate rather than exact." }, "?")]),
      pillGroup([{ value: "steps", label: "Steps" }, { value: "epochs", label: "Epochs" }], m.kohaku_duration || "steps",
        (v) => { m.kohaku_duration = v; renderAllSections(); refreshPreview(); }),
    ]);
    const duration = h("div", { class: "field-grid cols-3" }, [
      durationToggle,
      epochMode
        ? kf("kohaku_epochs", "Epochs", "num", { step: 1, min: 0.1, hint: `Passes over the dataset. Steps per epoch = images x repeats / (${perBatch} per batch x ${acc} accumulation); the exact step count is shown when training starts.` })
        : kf("kohaku_steps", "Steps", "num", { step: 10, min: 1, hint: stepsHint }),
      kf("kohaku_grad_acc", "Gradient accumulation", "num", { step: 1, min: 1 }),
      epochMode
        ? kf("kohaku_save_every_epochs", "Save every (epochs)", "num", { step: 0.5, min: 0, hint: "0 = only at the end." })
        : kf("kohaku_save_every", "Save every (steps)", "num", { step: 10, min: 0 }),
    ]);

    // ---- checkpoints / logging
    const run = h("div", { class: "field-grid cols-3" }, [
      kf("kohaku_save_minutes", "Also save every (minutes)", "num", { step: 5, min: 0 }),
      kf("kohaku_keep_last", "Keep last checkpoints", "num", { step: 1, min: 0 }),
      kf("kohaku_final_checkpoint", "Final resume checkpoint", "select", { options: ["auto", "true", "false"] }),
      kf("kohaku_save_format", "LoRA file format", "select", { options: ["comfy", "hf", "both", "none"], hint: "comfy = the key layout ComfyUI's MOSS-Image LoRA loader reads. The finished file is copied to <output folder>/<name>.safetensors." }),
      kf("kohaku_save_fused_qkv", "Fused q/k/v keys", "select", { options: ["", "auto", "true", "false"] }),
      kf("kohaku_save_merged", "Also save the merged DiT", "bool"),
      kf("kohaku_checkpoint_export", "Export inside every checkpoint", "bool"),
      kf("kohaku_resume", "Resume", "text", { placeholder: "latest, or a checkpoint folder", hint: "Resumes an interrupted run of the same name." }),
      kf("kohaku_init_from", "Start from checkpoint (optional)", "path", { span: "full" }),
    ]);
    const logging = h("div", { class: "field-grid cols-3" }, [
      kf("kohaku_log_every", "Log every (steps)", "num", { step: 1, min: 1 }),
      kf("kohaku_metrics_every", "Metrics every (steps)", "num", { step: 1, min: 1 }),
      kf("kohaku_logger", "Logger", "select", { options: ["console", "wandb"] }),
      kf("kohaku_metrics_level", "Metrics detail", "select", { options: ["train", "bench", "debug"] }),
      kf("kohaku_run_inspect", "Print the data plan first", "bool", { hint: "Runs the trainer's inspect step (epoch size, tokens per batch, sample captions) before caching." }),
    ]);

    // ---- sampling
    const sampling = h("div", { class: "field-grid cols-3" }, [
      kf("kohaku_sample_every", "Sample every (steps)", "num", { step: 10, min: 0, hint: "0 = off. Sampling loads the text encoder and runs the full DiT - slow and VRAM-heavy on small cards." }),
      kf("kohaku_sample_prompts", "Sample prompts (one per line)", "area", { span: "full", rows: 3 }),
      kf("kohaku_sample_negative", "Negative prompt", "text", { span: "full" }),
      kf("kohaku_sample_steps", "Sampler steps", "num", { step: 1, min: 1 }),
      kf("kohaku_sample_cfg", "CFG", "num", { step: 0.1, min: 0 }),
      kf("kohaku_sample_shift", "Shift", "num", { step: 0.1, min: 0 }),
      kf("kohaku_sample_width", "Width", "num", { step: 64, min: 256 }),
      kf("kohaku_sample_height", "Height", "num", { step: 64, min: 256 }),
      kf("kohaku_sample_seeds", "Seeds", "text", { placeholder: "0, 1" }),
      kf("kohaku_sample_skip_layers", "CFG skip layers", "text", { placeholder: "22" }),
      kf("kohaku_sample_batch", "Prompts per forward", "num", { step: 1, min: 0 }),
      kf("kohaku_sample_columns", "Sheet columns", "num", { step: 1, min: 0 }),
      kf("kohaku_sample_at_start", "Also sample at step 0", "bool"),
    ]);

    const extra = kf("kohaku_extra_knobs", "Extra trainer knobs", "area", { span: "full", rows: 4,
      placeholder: 'ARCH_OVERRIDES = {}\nSTREAM_LOSS = {"concentration": {"weight": 0.01, "k": 32}}',
      hint: "Anything the fields in the other tabs do not cover: one KNOB_NAME = value per line (Python values; text in quotes). These are applied last and win over every field. Knob names are listed in kohaku_backend/scripts/train/moss.py." });

    // One card per topic, spread over the tabs like the sd-scripts engine: Model / Dataset / Training / Output / Advanced.
    kohakuParts = {
      adapter: section("kohaku-adapter", "Adapter", "Algorithm, rank and target modules", [adapter], { open: true }),
      optim: section("kohaku-optim", "Optimizer & schedule", "Learning rate, optimizer, warmup", [optim], { open: true }),
      reading: wideSec(section("kohaku-data", "Dataset reading", "Captions, resolution, batching and caches",
        [h("p", { class: "card-hint", style: "margin:0 0 8px" }, "Dataset folders and repeats are set above. Repeats become sampling weights."), data], { open: true })),
      duration: section("kohaku-duration", "Duration", "Steps or epochs, accumulation, how often to save", [duration], { open: true }),
      memory: section("kohaku-memory", "Memory & precision", "VRAM layer, precision, streaming",
        [h("p", { class: "card-hint", style: "margin:0 0 8px" }, "Applies to the DiT. The text encoder and VAE only run while captions and latents are cached (offline), then are freed."), memGrid, manual], { open: true }),
      placement: section("kohaku-placement", "Text encoder & VAE", "Where they run and how they are loaded", [placement]),
      objective: section("kohaku-objective", "Objective", "Timestep sampling and loss", [objective], { open: true }),
      checkpoints: section("kohaku-checkpoints", "Checkpoints", "File format, keeping, resuming", [run], { open: true }),
      logging: section("kohaku-logging", "Logging", "Console, metrics and the data plan", [logging]),
      sampling: wideSec(section("kohaku-sampling", "Sample images during training", "Prompts and sampler settings", [sampling])),
      extra: section("kohaku-extra", "Extra trainer knobs", "Advanced / passthrough", [extra], { open: true }),
    };

    return section("model", "Model · MOSS-Image (Low-VRAM trainer)", "Trains the int8 files directly; streams weights on 6-12 GB cards", [files], { open: true });
  }

  // ------------------------------ General ------------------------------

  function buildGeneralSection() {
    const g = state.general;
    const resGrid = h("div", { class: "field-grid cols-3" });
    const grid = h("div", { class: "field-grid cols-3" });

    const heightField = field("Height", numberInput(g.resolution_height, (v) => g.resolution_height = v, { step: 64, min: 64 }));
    heightField.style.display = g.resolution_height_enabled ? "" : "none";
    resGrid.append(
      field("Resolution (width)", numberInput(g.resolution_width, (v) => g.resolution_width = v, { step: 64, min: 64 })),
      h("div", { class: "field" }, [
        h("label", {}, "\u00A0"),
        toggleRow(g.resolution_height_enabled, (v) => {
          g.resolution_height_enabled = v;
          heightField.style.display = v ? "" : "none";
        }, "Non-square"),
      ]),
      heightField,
    );

    const maxTrainField = field("Amount", numberInput(g.max_train_value, (v) => g.max_train_value = v, { step: 1, min: 1 }));
    grid.append(
      field("Train for", pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }],
        g.max_train_mode, (v) => g.max_train_mode = v)),
      maxTrainField,
      field("Batch size", numberInput(g.batch_size, (v) => g.batch_size = v, { step: 1, min: 1 })),
      field("Seed", numberInput(g.seed, (v) => g.seed = v, { step: 1 })),
    );

    const precisionRow = h("div", { class: "field-grid cols-3" }, [
      field("Mixed precision", selectInput([{ value: "fp16" }, { value: "bf16" }, { value: "float" }], g.mixed_precision, (v) => g.mixed_precision = v)),
      toggleRow(g.full_fp16, (v) => g.full_fp16 = v, "Full fp16", "Overrides mixed precision."),
      toggleRow(g.full_bf16, (v) => g.full_bf16 = v, "Full bf16", "Overrides mixed precision."),
      (state.model_type || "anima") === "sdxl"
        ? toggleRow(g.fp8_base, (v) => g.fp8_base = v, "fp8 base weights", "Not supported for Anima - SDXL only.")
        : null,
    ]);

    const attnWrap = h("div", { class: "field" }, [
      h("label", {}, "Attention"),
      pillGroup(
        [{ value: "sdpa", label: "SDPA" }, { value: "xformers", label: "xFormers" }, { value: "none", label: "None" }],
        g.xformers ? "xformers" : g.sdpa ? "sdpa" : "none",
        (v) => { g.sdpa = v === "sdpa"; g.xformers = v === "xformers"; },
      ),
    ]);

    const memoryGrid = h("div", { class: "field-grid cols-3" }, [
      toggleRow(g.gradient_checkpointing, (v) => g.gradient_checkpointing = v, "Gradient checkpointing", "Saves VRAM, a little slower."),
      toggleRow(g.lowram, (v) => g.lowram = v, "Low RAM mode"),
      toggleRow(g.highvram, (v) => g.highvram = v, "High VRAM mode"),
      toggleRow(g.no_half_vae, (v) => g.no_half_vae = v, "No half VAE"),
    ]);
    const toggles = h("div", { class: "field-grid cols-3" }, [
      toggleRow(g.cache_latents, (v) => { g.cache_latents = v; cacheDiskField.style.display = v ? "" : "none"; }, "Cache latents"),
      (state.model_type || "anima") !== "sdxl" && state.network
        ? toggleRow(state.network.cache_te_outputs, (v) => state.network.cache_te_outputs = v, "Cache text encoder outputs",
            "Faster and lighter. Only when the text encoder isn't trained.")
        : null,
      field("Dataloader workers", numberInput(g.max_data_loader_n_workers, (v) => g.max_data_loader_n_workers = v, { step: 1, min: 0 })),
    ]);
    var cacheDiskField = optionalField(g.cache_latents_to_disk, (v) => g.cache_latents_to_disk = v, "Cache latents to disk", h("div"));
    cacheDiskField.style.display = g.cache_latents ? "" : "none";
    toggles.appendChild(cacheDiskField);

    const gradAccumInner = numberInput(g.gradient_accumulation_steps, (v) => g.gradient_accumulation_steps = v, { step: 1, min: 1 });
    const advanced = h("div", { class: "field-grid cols-3" }, [
      optionalField(g.gradient_accumulation_enabled, (v) => g.gradient_accumulation_enabled = v, "Gradient accumulation", gradAccumInner),
      field("Prior loss weight", numberInput(g.prior_loss_weight, (v) => g.prior_loss_weight = v, { step: 0.1 }),
        { hint: "Only matters if you use regularization images." }),
      optionalField(g.keep_tokens_separator_enabled, (v) => g.keep_tokens_separator_enabled = v, "Keep-tokens separator",
        textInput(g.keep_tokens_separator, (v) => g.keep_tokens_separator = v, { placeholder: "|||" })),
      optionalField(g.training_comment_enabled, (v) => g.training_comment_enabled = v, "Training comment",
        textInput(g.training_comment, (v) => g.training_comment = v)),
      optionalField(g.protected_tags_file_enabled, (v) => g.protected_tags_file_enabled = v, "Protected tags file",
        pathInput(g.protected_tags_file, (v) => g.protected_tags_file = v)),
      field("External VAE override", pathInput(g.vae, (v) => g.vae = v),
        { hint: "Rarely needed for Anima - it already has its own VAE above." }),
    ]);

    return section("general", "Training basics", "Length, resolution, precision, memory, caching", [
      sub("Length & batch", grid),
      sub("Resolution", resGrid),
      sub("Precision & attention", [precisionRow, attnWrap]),
      sub("Memory", memoryGrid),
      sub("Caching & loading (speed)", toggles),
      sub("Other", advanced),
    ], { open: true });
  }

  // ------------------------------ Subsets ------------------------------

  function subsetCard(subset, idx, onRemove, startOpen) {
    const card = h("div", { class: `subset-card${startOpen ? " open" : ""}` });
    const head = h("div", { class: "subset-head", onclick: (e) => {
      if (e.target.tagName !== "INPUT" && e.target.tagName !== "BUTTON") card.classList.toggle("open");
    } }, [
      h("svg", { class: "chev", width: "10", height: "10", viewBox: "0 0 10 10", html: "<path d='M2 1l5 4-5 4' stroke='currentColor' stroke-width='1.6' fill='none'/>" }),
      textInput(subset.name, (v) => subset.name = v),
      h("span", { style: "flex:1" }),
      h("button", { class: "btn small danger", type: "button", onclick: (e) => { e.stopPropagation(); onRemove(); } }, "Remove"),
    ]);

    const folderGrid = h("div", { class: "field-grid cols-3" }, [
      field("Image folder", pathInput(subset.image_dir, (v) => subset.image_dir = v), { span: "full" }),
      field("Repeats", numberInput(subset.num_repeats, (v) => subset.num_repeats = v, { step: 1, min: 1 })),
      field("Caption extension", selectInput([{ value: ".txt" }, { value: ".caption" }], subset.caption_extension, (v) => subset.caption_extension = v)),
      h("div", { class: "field" }, [
        h("label", {}, "Role of this folder"),
        checkboxRow(subset.is_reg, (v) => subset.is_reg = v, "Regularization images"),
        checkboxRow(subset.is_val, (v) => subset.is_val = v, "Validation images"),
      ]),
    ]);

    const cropPadding = numberInput(subset.random_crop_padding_percent, (v) => subset.random_crop_padding_percent = v, { step: 0.01, min: 0, max: 1 });
    const pairedPaths = h("div", { class: "field-grid cols-3" }, [
      field("Target/output folder", pathInput(subset.target_image_dir, (v) => subset.target_image_dir = v),
        { hint: "For control-net style paired datasets." }),
      field("Conditioning data folder", pathInput(subset.conditioning_data_dir, (v) => subset.conditioning_data_dir = v)),
    ]);

    const faceCropW = numberInput(subset.face_crop_width, (v) => subset.face_crop_width = v, { step: 0.1 });
    const faceCropH = numberInput(subset.face_crop_height, (v) => subset.face_crop_height = v, { step: 0.1 });
    const captionDropoutGroup = h("div", { class: "field-grid cols-3" }, [
      field("Dropout rate", numberInput(subset.caption_dropout_rate, (v) => subset.caption_dropout_rate = v, { step: 0.05, min: 0, max: 1 })),
      field("Dropout every N epochs", numberInput(subset.caption_dropout_every_n_epochs, (v) => subset.caption_dropout_every_n_epochs = v, { step: 1, min: 0 })),
      field("Per-tag dropout rate", numberInput(subset.caption_tag_dropout_rate, (v) => subset.caption_tag_dropout_rate = v, { step: 0.05, min: 0, max: 1 })),
    ]);
    const gammaMin = numberInput(subset.gamma_aug_min, (v) => subset.gamma_aug_min = v, { step: 0.01 });
    const gammaMax = numberInput(subset.gamma_aug_max, (v) => subset.gamma_aug_max = v, { step: 0.01 });
    const gammaRate = numberInput(subset.gamma_aug_rate, (v) => subset.gamma_aug_rate = v, { step: 0.05, min: 0, max: 1 });
    const gammaGroup = h("div", { class: "field-grid cols-3" }, [
      field("Min gamma", gammaMin), field("Max gamma", gammaMax), field("Rate", gammaRate),
    ]);
    const tokenWarmupMin = numberInput(subset.token_warmup_min, (v) => subset.token_warmup_min = v, { step: 1, min: 1 });
    const tokenWarmupStep = numberInput(subset.token_warmup_step, (v) => subset.token_warmup_step = v, { step: 1, min: 1 });
    const tokenWarmupGroup = h("div", { class: "field-grid cols-3" }, [
      field("Warmup min tokens", tokenWarmupMin), field("Warmup step", tokenWarmupStep),
    ]);

    const captionGrid = h("div", { class: "field-grid cols-3" }, [
      checkboxRow(subset.shuffle_caption, (v) => subset.shuffle_caption = v, "Shuffle caption tags"),
      field("Keep first N tags fixed", numberInput(subset.keep_tokens, (v) => subset.keep_tokens = v, { step: 1, min: 0 }),
        { hint: "Tags at the start (e.g. a trigger word) that shuffling won't move." }),
      field("Protected tags file", pathInput(subset.protected_tags_file, (v) => subset.protected_tags_file = v)),
      optionalField(!!subset.caption_dropout_enabled, (v) => subset.caption_dropout_enabled = v, "Caption dropout", captionDropoutGroup, { span: "full" }),
      optionalField(!!subset.token_warmup_enabled, (v) => subset.token_warmup_enabled = v, "Token warmup", tokenWarmupGroup, { span: "full" }),
    ]);

    const augGrid = h("div", { class: "field-grid cols-3" }, [
      checkboxRow(subset.flip_aug, (v) => subset.flip_aug = v, "Flip augment"),
      checkboxRow(subset.color_aug, (v) => subset.color_aug = v, "Color augment (off while Cache latents is on)"),
      optionalField(subset.random_crop, (v) => subset.random_crop = v, "Random crop (off while Cache latents is on)", field("Padding %", cropPadding)),
      optionalField(!!subset.gamma_aug_enabled, (v) => subset.gamma_aug_enabled = v, "Gamma augmentation", gammaGroup, { span: "full" }),
      optionalField(!!subset.face_crop_enabled, (v) => subset.face_crop_enabled = v, "Face crop aug range",
        h("div", { class: "field-grid cols-2" }, [field("Width", faceCropW), field("Height", faceCropH)]), { span: "full" }),
    ]);

    const grp = (title, el, cls) => h("div", { class: `subgroup${cls ? " " + cls : ""}` }, [h("div", { class: "subgroup-title" }, title), el]);
    const body = h("div", { class: "subset-body subset-cols" }, [
      grp("Folder", folderGrid),
      grp("Captions", captionGrid),
      grp("Image augmentation", augGrid),
      grp("Paired / control-net data (optional)", pairedPaths, "wide"),
    ]);

    card.append(head, body);
    return card;
  }

  function newSubset(name) {
    return {
      name: name || `subset_${state.subsets.length + 1}`, image_dir: "", target_image_dir: "", conditioning_data_dir: "",
      num_repeats: 1, caption_extension: ".txt", random_crop_padding_percent: 0.05,
      shuffle_caption: true, flip_aug: false, keep_tokens: 0, color_aug: false, random_crop: false,
      is_reg: false, is_val: false,
      face_crop_enabled: false, face_crop_width: 1, face_crop_height: 1,
      caption_dropout_enabled: false, caption_dropout_rate: 0, caption_dropout_every_n_epochs: 0, caption_tag_dropout_rate: 0,
      gamma_aug_enabled: false, gamma_aug_min: 0.95, gamma_aug_max: 1.05, gamma_aug_rate: 0.5,
      shuffle_caption_sigma_enabled: false, shuffle_caption_sigma: 0,
      token_warmup_enabled: false, token_warmup_min: 1, token_warmup_step: 1,
      protected_tags_file: "",
    };
  }

  function buildSubsetsSection() {
    const list = h("div");
    function renderList() {
      list.innerHTML = "";
      if (state.subsets.length === 0) {
        list.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "No dataset folders added yet.")]));
      }
      state.subsets.forEach((subset, idx) => {
        list.appendChild(subsetCard(subset, idx, () => { state.subsets.splice(idx, 1); renderList(); }, true));
      });
    }
    renderList();

    function addFolder(path) {
      if (state.subsets.find((s) => s.image_dir === path)) return false;
      const name = path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || "dataset";
      state.subsets.push({ ...newSubset(name), image_dir: path });
      renderList();
      return true;
    }
    addDatasetFolder = addFolder;

    const addBtn = h("button", { class: "btn primary small", type: "button", onclick: () => {
      state.subsets.push(newSubset());
      renderList();
    } }, "+ Add dataset folder");

    const useActiveBtn = h("button", {
      class: "btn small", type: "button",
      title: "Use the folder you were just working with in Collect/Edit/Caption",
      onclick: () => {
        const active = ActiveDataset.get();
        if (!active) return toast("No dataset folder yet - collect or caption one first.", "info");
        if (!addFolder(active)) return toast("That folder is already added.", "info");
        renderList();
        toast(`Added "${active}".`, "success");
      },
    }, "Use current dataset");

    // Every subset already known to Collect/Caption/Edit (their side-rail
    // tabs) - so a whole dataset's worth of folders can be picked from or
    // added in one click, instead of browsing to each one by hand.
    const knownPicker = h("select", { style: "width:auto" }, [h("option", { value: "" }, "Pick a known subset…")]);
    const addKnownBtn = h("button", { class: "btn small", type: "button" }, "Add");
    const addAllBtn = h("button", {
      class: "btn small", type: "button",
      title: "Add every subset known to Collect/Caption/Edit that isn't already in this list",
      onclick: async () => {
        try {
          const { subsets } = await Api.listSubsets();
          const added = subsets.filter((s) => addFolder(s.path)).length;
          renderList();
          toast(added ? `Added ${added} subset${added === 1 ? "" : "s"}.` : "Nothing new to add - they're all already here.", added ? "success" : "info");
        } catch (err) {
          toast(err.message, "error");
        }
      },
    }, "Add all subsets");

    Api.listSubsets().then(({ subsets }) => {
      subsets.forEach((s) => {
        knownPicker.appendChild(h("option", { value: s.path }, `${s.name} (${s.image_count})`));
      });
    }).catch(() => {});

    addKnownBtn.addEventListener("click", () => {
      const path = knownPicker.value;
      if (!path) return;
      if (!addFolder(path)) { toast("That folder is already added.", "info"); return; }
      renderList();
      knownPicker.value = "";
    });

    return section("subsets", "Dataset folders", `${state.subsets.length || ""}`, [
      list,
      h("div", { class: "row", style: "flex-wrap:wrap;gap:8px" }, [addBtn, useActiveBtn, knownPicker, addKnownBtn, addAllBtn]),
    ], { open: true });
  }

  // ------------------------------ Bucketing ------------------------------

  function buildBucketSection() {
    const b = state.bucket;
    const inner = h("div", { class: "field-grid cols-3" }, [
      field("Min resolution", numberInput(b.min_reso, (v) => b.min_reso = v, { step: 64, min: 64 })),
      field("Max resolution", numberInput(b.max_reso, (v) => b.max_reso = v, { step: 64, min: 64 })),
      field("Resolution step", numberInput(b.reso_steps, (v) => b.reso_steps = v, { step: 8, min: 8 })),
      checkboxRow(b.no_upscale, (v) => b.no_upscale = v, "Don't upscale small images"),
      checkboxRow(b.multires_training, (v) => b.multires_training = v, "Multi-resolution training"),
    ]);
    inner.style.display = b.enabled ? "" : "none";
    const toggle = toggleRow(b.enabled, (v) => { b.enabled = v; inner.style.display = v ? "" : "none"; }, "Enable aspect-ratio bucketing");
    return section("bucket", "Bucketing", "Aspect-ratio buckets for mixed-size datasets", [toggle, inner], { open: true });
  }

  // ------------------------------ Network (LoRA / LyCORIS) ------------------------------

  function buildNetworkSection() {
    const n = state.network;
    const algoOptions = (schemaData.network_algos || []).map((a) => ({ value: a.value, label: a.label }));
    const grid = h("div", { class: "field-grid cols-3" });

    const lycorisGroup = h("div", { class: "field-grid cols-3" });
    const dyloraField = field("DyLoRA unit", numberInput(n.dylora_unit, (v) => n.dylora_unit = v, { step: 1, min: 1 }));
    const loraFaToggle = checkboxRow(n.lora_fa, (v) => n.lora_fa = v, "LoRA-FA (freeze A matrix)");

    function algoInfo(value) {
      return (schemaData.network_algos || []).find((a) => a.value === value) || { lycoris: false };
    }

    function syncAlgoGroups() {
      const info = algoInfo(n.algo);
      lycorisGroup.style.display = info.lycoris ? "" : "none";
      dyloraField.style.display = n.algo === "dylora" ? "" : "none";
      loraFaToggle.style.display = !info.lycoris ? "" : "none";
    }

    grid.append(
      field("Algorithm", selectInput(algoOptions, n.algo, (v) => { n.algo = v; syncAlgoGroups(); })),
      field("LyCORIS preset", textInput(n.preset, (v) => n.preset = v, { placeholder: "full / attn-mlp / …" })),
      field("Network dim (rank)", numberInput(n.network_dim, (v) => n.network_dim = v, { step: 1, min: 1 })),
      field("Network alpha", numberInput(n.network_alpha, (v) => n.network_alpha = v, { step: 1, min: 0.1 })),
      field("Min timestep", numberInput(n.min_timestep, (v) => n.min_timestep = v, { step: 10, min: 0, max: 1000 })),
      field("Max timestep", numberInput(n.max_timestep, (v) => n.max_timestep = v, { step: 10, min: 0, max: 1000 })),
      field("Train target", selectInput([
        { value: "both", label: "UNet + Text Encoder" }, { value: "unet", label: "UNet only" }, { value: "te", label: "Text encoder only" },
      ], n.train_target, (v) => n.train_target = v)),
    );

    const convDimEnabled = h("div", { class: "field-grid cols-2", style: n.conv_dim_enabled ? "" : "display:none" }, [
      field("Conv dim", numberInput(n.conv_dim, (v) => n.conv_dim = v, { step: 1, min: 1 })),
      field("Conv alpha", numberInput(n.conv_alpha, (v) => n.conv_alpha = v, { step: 1, min: 0.1 })),
    ]);
    lycorisGroup.append(
      optionalField(n.conv_dim_enabled, (v) => { n.conv_dim_enabled = v; convDimEnabled.style.display = v ? "" : "none"; },
        "Separate conv dim/alpha", h("div"), { span: "full" }),
      convDimEnabled,
      optionalField(n.rank_dropout_enabled, (v) => n.rank_dropout_enabled = v, "Rank dropout",
        numberInput(n.rank_dropout, (v) => n.rank_dropout = v, { step: 0.05, min: 0, max: 1 })),
      optionalField(n.module_dropout_enabled, (v) => n.module_dropout_enabled = v, "Module dropout",
        numberInput(n.module_dropout, (v) => n.module_dropout = v, { step: 0.05, min: 0, max: 1 })),
      optionalField(n.constraint_enabled, (v) => n.constraint_enabled = v, "Constraint",
        numberInput(n.constraint, (v) => n.constraint = v, { step: 0.1 })),
      checkboxRow(n.use_tucker, (v) => n.use_tucker = v, "Use Tucker decomposition"),
      checkboxRow(n.train_norm, (v) => n.train_norm = v, "Train norm layers"),
      checkboxRow(n.dora, (v) => n.dora = v, "DoRA weight decompose"),
      checkboxRow(n.rescaled, (v) => n.rescaled = v, "Rescaled"),
      checkboxRow(n.bypass_mode, (v) => n.bypass_mode = v, "Bypass mode"),
    );

    const dropoutGrid = h("div", { class: "field-grid cols-3" }, [
      optionalField(n.network_dropout_enabled, (v) => n.network_dropout_enabled = v, "Network dropout",
        numberInput(n.network_dropout, (v) => n.network_dropout = v, { step: 0.05, min: 0, max: 1 })),
      optionalField(n.ip_noise_gamma_enabled, (v) => n.ip_noise_gamma_enabled = v, "IP noise gamma",
        numberInput(n.ip_noise_gamma, (v) => n.ip_noise_gamma = v, { step: 0.01, min: 0 })),
      dyloraField,
    ]);

    const cacheTeInner = checkboxRow(n.cache_te_to_disk, (v) => n.cache_te_to_disk = v, "Also cache to disk");
    const cacheTeWrap = h("div", { style: n.cache_te_outputs ? "" : "display:none" }, [cacheTeInner]);
    const otherToggles = h("div", { class: "field-grid cols-3" }, [
      toggleRow(n.cache_te_outputs, (v) => { n.cache_te_outputs = v; cacheTeWrap.style.display = v ? "" : "none"; }, "Cache text encoder outputs"),
      cacheTeWrap,
      loraFaToggle,
    ]);

    dyloraField.style.display = n.algo === "dylora" ? "" : "none";
    loraFaToggle.style.display = !algoInfo(n.algo).lycoris ? "" : "none";

    // Block weights / block dims (advanced, simplified to comma-separated text)
    const blockWeightInner = h("div", { class: "field-grid cols-3" }, [
      field("Down (12 values)", textInput(n.block_weight.down, (v) => n.block_weight.down = v, { placeholder: "1,1,1,1,1,1,1,1,1,1,1,1" })),
      field("Mid (1 value)", textInput(n.block_weight.mid, (v) => n.block_weight.mid = v, { placeholder: "1" })),
      field("Up (12 values)", textInput(n.block_weight.up, (v) => n.block_weight.up = v, { placeholder: "1,1,1,1,1,1,1,1,1,1,1,1" })),
    ]);
    function simpleValuesField(obj, placeholder) {
      return field("Values", textInput(obj.values, (v) => obj.values = v, { placeholder }));
    }
    const blockExtras = h("div", { class: "field-grid cols-2" }, [
      optionalField(n.block_dims.enabled, (v) => n.block_dims.enabled = v, "Per-block dims",
        simpleValuesField(n.block_dims, "comma-separated, 1 per block")),
      optionalField(n.block_alphas.enabled, (v) => n.block_alphas.enabled = v, "Per-block alphas",
        simpleValuesField(n.block_alphas, "comma-separated, 1 per block")),
      optionalField(n.conv_block_dims.enabled, (v) => n.conv_block_dims.enabled = v, "Per-block conv dims",
        simpleValuesField(n.conv_block_dims, "comma-separated, 1 per block")),
      optionalField(n.conv_block_alphas.enabled, (v) => n.conv_block_alphas.enabled = v, "Per-block conv alphas",
        simpleValuesField(n.conv_block_alphas, "comma-separated, 1 per block")),
    ]);
    const blockSection = h("div", { class: "subgroup" }, [
      h("div", { class: "subgroup-title" }, "Block weighting (advanced)"),
      optionalField(n.block_weight.enabled, (v) => n.block_weight.enabled = v, "Per-block LR weights", blockWeightInner, { span: "full" }),
      blockExtras,
    ]);

    const extraArgsSection = h("div", { class: "subgroup" }, [
      h("div", { class: "subgroup-title" }, "Extra network_args (advanced / custom)"),
      keyValueList(n.extra_network_args),
    ]);

    syncAlgoGroups();

    return section("network", "Network (LoRA)", "Rank, algorithm, dropout", [
      grid, lycorisGroup, dropoutGrid, otherToggles, blockSection, extraArgsSection,
    ], { open: true });
  }

  // ------------------------------ Optimizer / Scheduler / Loss ------------------------------

  function buildOptimizerSection() {
    const o = state.optimizer;
    const optimizerOptions = [
      { group: "Recommended", options: (schemaData.optimizers?.recommended || []).map((v) => ({ value: v })) },
      { group: "All optimizers", options: (schemaData.optimizers?.all || []).map((v) => ({ value: v })) },
    ];

    const grid = h("div", { class: "field-grid cols-3" }, [
      field("Optimizer", selectInput(optimizerOptions, o.optimizer_type, (v) => o.optimizer_type = v)),
      field("Learning rate", textInput(o.learning_rate, (v) => o.learning_rate = v, { placeholder: "1e-4" }), { hint: "Scientific notation is fine, e.g. 1e-4" }),
      field("Max grad norm", numberInput(o.max_grad_norm, (v) => o.max_grad_norm = v, { step: 0.1, min: 0 })),
      optionalField(o.unet_lr_enabled, (v) => o.unet_lr_enabled = v, "Separate UNet LR",
        textInput(o.unet_lr, (v) => o.unet_lr = v, { placeholder: "1e-4" })),
      optionalField(o.te_lr_enabled, (v) => o.te_lr_enabled = v, "Separate text encoder LR",
        textInput(o.te_lr, (v) => o.te_lr = v, { placeholder: "1e-4" })),
    ]);

    const schedulerOptions = (schemaData.lr_schedulers || []).map((s) => ({ value: s.value, label: s.label }));
    const numCyclesField = field("Num cycles", numberInput(o.num_cycles, (v) => o.num_cycles = v, { step: 1, min: 1 }), { hint: "How many times the learning rate restarts." });
    const polyPowerField = field("Poly power", numberInput(o.poly_power, (v) => o.poly_power = v, { step: 0.1 }));
    const minLrField = field("Min LR", numberInput(o.min_lr, (v) => o.min_lr = v, { step: 0.00001 }), { hint: "Lowest learning rate the schedule reaches." });
    const gammaField = field("Gamma", numberInput(o.gamma, (v) => o.gamma = v, { step: 0.01, min: 0, max: 1 }), { hint: "How much lower the peak learning rate is after each restart. Only matters with more than 1 cycle." });
    const dParamField = field("Curve shape (D)", numberInput(o.d_param, (v) => o.d_param = v, { step: 0.01, min: 0, max: 1 }), { hint: "Shape of the rex decay. 0 = straight line down; higher = the learning rate stays high longer, then drops faster near the end. 0.9 is the usual value." });
    const decayRatioField = field("Decay ratio", numberInput(o.decay_ratio, (v) => o.decay_ratio = v, { step: 0.05, min: 0, max: 1 }));
    const decayTypeField = field("Decay type", selectInput([{ value: "1-sqrt" }, { value: "cosine" }, { value: "linear" }], o.decay_type, (v) => o.decay_type = v));
    const timescaleField = field("Timescale", numberInput(o.timescale, (v) => o.timescale = v, { step: 1, min: 1 }), { hint: "Inverse sqrt timescale - defaults to the warmup step count when empty" });
    const minLrRatioField = field("Min LR ratio", numberInput(o.min_lr_ratio, (v) => o.min_lr_ratio = v, { step: 0.01, min: 0, max: 1 }), { hint: "Floor as a fraction of the initial learning rate" });
    const initialLrField = field("Initial LR", numberInput(o.initial_lr, (v) => o.initial_lr = v, { step: 0.001, min: 0 }), { hint: "Sent to the backend as adafactor:<initial lr>" });
    const schedulerExtras = h("div", { class: "field-grid cols-3" }, [
      numCyclesField, polyPowerField, minLrField, gammaField, dParamField, decayRatioField, decayTypeField,
      timescaleField, minLrRatioField, initialLrField,
    ]);
    const schedulerArgsSection = h("div", { class: "subgroup" }, [
      h("div", { class: "subgroup-title" }, "Extra scheduler args (key=value, e.g. step_rules for Piecewise Constant)"),
      keyValueList(o.lr_scheduler_extra_args),
    ]);

    function syncScheduler() {
      const info = (schemaData.lr_schedulers || []).find((s) => s.value === o.lr_scheduler) || { fields: [] };
      const show = new Set(info.fields || []);
      numCyclesField.style.display = show.has("num_cycles") ? "" : "none";
      polyPowerField.style.display = show.has("poly_power") ? "" : "none";
      minLrField.style.display = show.has("min_lr") ? "" : "none";
      gammaField.style.display = show.has("gamma") ? "" : "none";
      dParamField.style.display = show.has("d_param") ? "" : "none";
      decayRatioField.style.display = show.has("decay_ratio") ? "" : "none";
      decayTypeField.style.display = show.has("decay_type") ? "" : "none";
      timescaleField.style.display = show.has("timescale") ? "" : "none";
      minLrRatioField.style.display = show.has("min_lr_ratio") ? "" : "none";
      initialLrField.style.display = show.has("initial_lr") ? "" : "none";
    }

    const zeroWarmupToggle = checkboxRow(o.zero_lr_warmup, (v) => o.zero_lr_warmup = v, "Warm up from zero");
    const zeroWarmupWrap = h("div", { style: o.warmup_ratio_enabled ? "" : "display:none" }, [zeroWarmupToggle]);
    const warmupInner = h("div", { class: "stack" }, [
      numberInput(o.warmup_ratio, (v) => o.warmup_ratio = v, { step: 0.01, min: 0, max: 1 }),
    ]);

    const schedulerRow = h("div", { class: "field-grid cols-3" }, [
      field("LR scheduler", selectInput(schedulerOptions, o.lr_scheduler, (v) => { o.lr_scheduler = v; syncScheduler(); })),
      optionalField(o.warmup_ratio_enabled, (v) => {
        o.warmup_ratio_enabled = v; zeroWarmupWrap.style.display = v ? "" : "none";
      }, "Warmup ratio", warmupInner),
      zeroWarmupWrap,
    ]);
    syncScheduler();

    const lossOptions = (schemaData.loss_types || []).map((l) => ({ value: l.value, label: l.label }));
    const huberParamField = field("Huber c", numberInput(o.huber_param, (v) => o.huber_param = v, { step: 0.01 }));
    const huberScheduleField = field("Huber schedule", selectInput(
      (schemaData.huber_schedules || ["snr", "exponential", "constant"]).map((v) => ({ value: v })),
      o.huber_schedule, (v) => o.huber_schedule = v,
    ));
    function syncLoss() {
      const info = (schemaData.loss_types || []).find((l) => l.value === o.loss_type) || { huber: false };
      huberParamField.style.display = info.huber ? "" : "none";
      huberScheduleField.style.display = info.huber ? "" : "none";
    }

    const otherRow = h("div", { class: "field-grid cols-3" }, [
      field("Loss type", selectInput(lossOptions, o.loss_type, (v) => { o.loss_type = v; syncLoss(); })),
      huberParamField, huberScheduleField,
      optionalField(o.scale_weight_norms_enabled, (v) => o.scale_weight_norms_enabled = v, "Scale weight norms",
        numberInput(o.scale_weight_norms, (v) => o.scale_weight_norms = v, { step: 0.1 })),
      optionalField(o.min_snr_gamma_enabled, (v) => o.min_snr_gamma_enabled = v, "Min-SNR gamma",
        numberInput(o.min_snr_gamma, (v) => o.min_snr_gamma = v, { step: 0.5 })),
      checkboxRow(o.zero_terminal_snr, (v) => o.zero_terminal_snr = v, "Zero terminal SNR"),
      checkboxRow(o.masked_loss, (v) => o.masked_loss = v, "Masked loss"),
    ]);
    syncLoss();

    const extraArgsSection = h("div", { class: "subgroup" }, [
      h("div", { class: "subgroup-title" }, "Extra optimizer_args (e.g. weight_decay, betas)"),
      keyValueList(o.extra_optimizer_args),
    ]);

    return section("optimizer", "Optimizer, scheduler & loss", `${o.optimizer_type} · ${o.lr_scheduler}`, [
      sub("Optimizer & learning rate", [grid, extraArgsSection]),
      sub("Scheduler", [schedulerRow, schedulerExtras, schedulerArgsSection],
        "Warmup is ignored by the custom schedulers (Rex, CAWR...) - they handle it themselves."),
      sub("Loss & regularization", otherRow),
    ], { open: true });
  }

  // ------------------------------ Saving ------------------------------

  function buildSavingSection() {
    const s = state.saving;
    const grid = h("div", { class: "field-grid cols-3" }, [
      field("Output folder", pathInput(s.output_dir, (v) => s.output_dir = v), { span: "full" }),
      optionalField(s.output_name_enabled, (v) => s.output_name_enabled = v, "Custom output name",
        textInput(s.output_name, (v) => s.output_name = v, { placeholder: "my_anima_lora" })),
      field("Save precision", selectInput([{ value: "fp16" }, { value: "bf16" }, { value: "float" }], s.save_precision, (v) => s.save_precision = v)),
      field("Save format", selectInput([{ value: "safetensors" }, { value: "pt" }, { value: "ckpt" }], s.save_model_as, (v) => s.save_model_as = v)),
      optionalField(s.resume_enabled, (v) => s.resume_enabled = v, "Resume from state",
        pathInput(s.resume, (v) => s.resume = v)),
    ]);

    const freqValueField = field("Every", numberInput(s.save_freq_value, (v) => s.save_freq_value = v, { step: 1, min: 1 }));
    const freqModeField = pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }], s.save_freq_mode, (v) => s.save_freq_mode = v);
    const freqExtras = h("div", { class: "field-grid cols-3", style: s.save_freq_enabled ? "" : "display:none" }, [freqModeField, freqValueField]);
    const freqRow = h("div", { class: "field-grid cols-3" }, [
      optionalField(s.save_freq_enabled, (v) => { s.save_freq_enabled = v; freqExtras.style.display = v ? "" : "none"; },
        "Save checkpoints periodically", h("div")),
      freqExtras,
    ]);

    const lastValueField = field("Keep last N", numberInput(s.save_last_value, (v) => s.save_last_value = v, { step: 1, min: 1 }));
    const lastModeField = pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }], s.save_last_mode, (v) => s.save_last_mode = v);
    const lastExtras = h("div", { class: "field-grid cols-3", style: s.save_only_last_enabled ? "" : "display:none" }, [lastModeField, lastValueField]);
    const lastRow = h("div", { class: "field-grid cols-3" }, [
      optionalField(s.save_only_last_enabled, (v) => { s.save_only_last_enabled = v; lastExtras.style.display = v ? "" : "none"; },
        "Limit saved checkpoints", h("div")),
      lastExtras,
      optionalField(s.save_ratio_enabled, (v) => s.save_ratio_enabled = v, "Save ratio",
        numberInput(s.save_ratio, (v) => s.save_ratio = v, { step: 1, min: 1 })),
    ]);

    const otherRow = h("div", { class: "field-grid cols-3" }, [
      optionalField(s.save_tag_enabled, (v) => s.save_tag_enabled = v, "Write tag occurrence file",
        pathInput(s.save_tag_dir, (v) => s.save_tag_dir = v)),
      optionalField(s.save_toml_enabled, (v) => s.save_toml_enabled = v, "Save a copy of the config",
        pathInput(s.save_toml_dir, (v) => s.save_toml_dir = v)),
    ]);

    const saveStateInner = h("div", { class: "field-grid cols-3" }, [
      pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }], s.save_last_state_mode, (v) => s.save_last_state_mode = v),
      numberInput(s.save_last_state_value, (v) => s.save_last_state_value = v, { step: 1, min: 1 }),
    ]);
    const saveStateRow = h("div", { class: "field-grid cols-2" }, [
      checkboxRow(s.save_state, (v) => { s.save_state = v; saveStateWrap.style.display = v ? "" : "none"; }, "Save optimizer/training state"),
    ]);
    var saveStateWrap = optionalField(s.save_last_state_enabled, (v) => s.save_last_state_enabled = v, "Limit saved states", saveStateInner);
    saveStateWrap.style.display = s.save_state ? "" : "none";

    return section("saving", "Saving", s.output_dir ? s.output_dir : "", [
      grid, freqRow, lastRow, otherRow, saveStateRow, saveStateWrap,
    ], { open: true });
  }

  // ------------------------------ Noise / Sample / Logging ------------------------------

  function buildNoiseSection() {
    const n = state.noise;
    const pyramidInner = h("div", { class: "field-grid cols-2" }, [
      field("Iterations", numberInput(n.pyramid_iterations, (v) => n.pyramid_iterations = v, { step: 1, min: 1 })),
      field("Discount", numberInput(n.pyramid_discount, (v) => n.pyramid_discount = v, { step: 0.01, min: 0, max: 1 })),
    ]);
    const grid = h("div", { class: "field-grid cols-2" }, [
      optionalField(n.noise_offset_enabled, (v) => n.noise_offset_enabled = v, "Noise offset",
        numberInput(n.noise_offset, (v) => n.noise_offset = v, { step: 0.01, min: 0 })),
      optionalField(n.pyramid_enabled, (v) => n.pyramid_enabled = v, "Pyramid (multi-res) noise", pyramidInner),
    ]);
    return section("noise", "Noise offset", "Optional regularization noise", [grid]);
  }

  function buildSampleSection() {
    const s = state.sample;
    const samplerOptions = (schemaData.samplers || ["ddim"]).map((v) => ({ value: v }));
    const body = h("div", { class: "field-grid cols-3" }, [
      field("Sampler", selectInput(samplerOptions, s.sampler, (v) => s.sampler = v)),
      field("Sample every", pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }], s.freq_mode, (v) => s.freq_mode = v)),
      field("Amount", numberInput(s.freq_value, (v) => s.freq_value = v, { step: 1, min: 1 })),
      field("Sample prompts", (() => {
        const ta = h("textarea", { rows: "5", placeholder: "one prompt per line, e.g.\n1girl, solo, smile --w 1024 --h 1024 --s 28 --l 4 --d 42 --n lowres, bad anatomy" });
        ta.value = s.prompts_text || "";
        ta.addEventListener("input", () => s.prompts_text = ta.value);
        return ta;
      })(), { span: "full", hint: "Write the prompts here, one per line. Optional per-line flags: --n negative prompt, --w width, --h height, --s steps, --l CFG scale, --d seed." }),
      s.prompts_file ? field("Prompts file (older setting)", pathInput(s.prompts_file, (v) => s.prompts_file = v), { span: "full", hint: "Only used when the box above is empty." }) : null,
    ]);
    body.style.display = s.enabled ? "" : "none";
    const toggle = toggleRow(s.enabled, (v) => { s.enabled = v; body.style.display = v ? "" : "none"; }, "Generate sample images during training");
    return section("sample", "Sample images", "Preview generations while training", [toggle, body], { open: true });
  }

  function buildLoggingSection() {
    const l = state.logging;
    const wandbField = field("Wandb API key", (() => {
      const el = h("input", { type: "password" });
      el.value = l.wandb_api_key;
      el.addEventListener("input", () => l.wandb_api_key = el.value);
      return el;
    })());
    const prefixManualField = field("Prefix text", textInput(l.log_prefix, (v) => l.log_prefix = v));
    const runNameManualField = field("Run name", textInput(l.run_name, (v) => l.run_name = v));

    function sync() {
      wandbField.style.display = ["wandb", "all"].includes(l.log_with) ? "" : "none";
      prefixManualField.style.display = l.log_prefix_mode === "manual" ? "" : "none";
      runNameManualField.style.display = l.run_name_mode === "manual" ? "" : "none";
    }

    const body = h("div", { class: "field-grid cols-3" }, [
      field("Log with", selectInput([{ value: "tensorboard" }, { value: "wandb" }, { value: "all" }], l.log_with, (v) => { l.log_with = v; sync(); })),
      field("Log directory", pathInput(l.logging_dir, (v) => l.logging_dir = v)),
      wandbField,
      field("Log prefix", selectInput([{ value: "disabled" }, { value: "default" }, { value: "manual" }], l.log_prefix_mode, (v) => { l.log_prefix_mode = v; sync(); })),
      prefixManualField,
      field("Run name", selectInput([{ value: "default" }, { value: "manual" }], l.run_name_mode, (v) => { l.run_name_mode = v; sync(); })),
      runNameManualField,
      optionalField(l.tracker_name_enabled, (v) => l.tracker_name_enabled = v, "Tracker project name",
        textInput(l.tracker_name, (v) => l.tracker_name = v)),
    ]);
    body.style.display = l.enabled ? "" : "none";
    sync();
    const toggle = toggleRow(l.enabled, (v) => { l.enabled = v; body.style.display = v ? "" : "none"; }, "Enable experiment logging");
    return section("logging", "Logging", "TensorBoard / Weights & Biases", [toggle, body], { open: true });
  }

  // ------------------------------ EDM² loss weighting ------------------------------

  function buildEdmSection() {
    const e = state.edm_loss;
    const schedulerInner = h("div", { class: "field-grid cols-2" }, [
      field("Warmup %", numberInput(e.warmup_percent, (v) => e.warmup_percent = v, { step: 0.05, min: 0, max: 1 })),
      field("Constant %", numberInput(e.constant_percent, (v) => e.constant_percent = v, { step: 0.05, min: 0, max: 1 })),
    ]);
    const optimizerOptions = [
      { group: "Recommended", options: (schemaData.optimizers?.recommended || []).map((v) => ({ value: v })) },
      { group: "All optimizers", options: (schemaData.optimizers?.all || []).map((v) => ({ value: v })) },
    ];
    const body = h("div", { class: "field-grid cols-3" }, [
      field("Optimizer", selectInput(optimizerOptions, e.optimizer_type, (v) => e.optimizer_type = v)),
      field("Learning rate", textInput(e.learning_rate, (v) => e.learning_rate = v)),
      field("Num channels", numberInput(e.num_channels, (v) => e.num_channels = v, { step: 1, min: 1 })),
      field("Optimizer args", textInput(e.optimizer_args, (v) => e.optimizer_args = v, { placeholder: "raw args string, optional" }), { span: "full" }),
      field("Initial weights", pathInput(e.initial_weights, (v) => e.initial_weights = v), { span: "full" }),
      optionalField(e.scheduler_enabled, (v) => e.scheduler_enabled = v, "LR scheduler for the loss weights", schedulerInner, { span: "full" }),
    ]);
    body.style.display = e.enabled ? "" : "none";
    const toggle = toggleRow(e.enabled, (v) => { e.enabled = v; body.style.display = v ? "" : "none"; }, "Enable EDM² loss weighting");
    return section("edm", "EDM\u00B2 loss weighting", "Learned per-timestep loss weights", [toggle, body]);
  }

  // ------------------------------ Experimental / Advanced ------------------------------

  function buildExperimentalSection() {
    const x = state.experimental;
    const body = h("div", { class: "field-grid cols-3" }, [
      optionalField(x.vae_batch_size_enabled, (v) => x.vae_batch_size_enabled = v, "VAE batch size",
        numberInput(x.vae_batch_size, (v) => x.vae_batch_size = v, { step: 1, min: 1 })),
      optionalField(x.vae_custom_scale_enabled, (v) => x.vae_custom_scale_enabled = v, "Custom VAE scale",
        numberInput(x.vae_custom_scale, (v) => x.vae_custom_scale = v, { step: 0.01 })),
      optionalField(x.vae_custom_shift_enabled, (v) => x.vae_custom_shift_enabled = v, "Custom VAE shift",
        numberInput(x.vae_custom_shift, (v) => x.vae_custom_shift = v, { step: 0.01 })),
      checkboxRow(x.vae_reflection, (v) => x.vae_reflection = v, "VAE reflection padding"),
      checkboxRow(x.zero_cond_dropout, (v) => x.zero_cond_dropout = v, "Zero conditioning dropout"),
      checkboxRow(x.debiased_estimation_loss, (v) => x.debiased_estimation_loss = v, "Debiased estimation loss"),
    ]);
    return section("experimental", "Experimental", "Mostly relevant to SDXL rectified-flow setups", [body]);
  }

  function buildAccelerateSection() {
    const a = state.accelerate;
    const body = h("div", { class: "field-grid cols-3" }, [
      field("Number of GPUs", numberInput(a.num_processes, (v) => a.num_processes = v, { step: 1, min: 1 })),
      field("Main process port", numberInput(a.main_process_port, (v) => a.main_process_port = v, { step: 1, min: 1024, max: 65535 })),
    ]);
    body.style.display = a.enabled ? "" : "none";
    const toggle = toggleRow(a.enabled, (v) => { a.enabled = v; body.style.display = v ? "" : "none"; }, "Multi-GPU (accelerate) training");
    return section("accelerate", "Multi-GPU", "Launch with accelerate across multiple GPUs", [toggle, body]);
  }

  function buildExtraArgsSection() {
    const body = h("div", { class: "stack" }, [
      h("p", { class: "card-hint", style: "margin:0" }, "Passed straight through to the backend - useful for flags this app doesn't have a dedicated field for yet."),
      keyValueList(state.extra_args, { datasetFlag: true }),
    ]);
    return section("extra", "Extra arguments", "Advanced / passthrough", [body]);
  }

  // ------------------------------ Orchestration ------------------------------

  function renderAllSections() {
    // Each panel is a two-column grid of cards; "wide" sections take the full row.
    const wide = (sec) => { sec.classList.add("wide"); return sec; };
    const kohaku = state.model_type === "moss" && state.model.moss_engine === "kohaku";
    const note = (text) => h("div", { class: "card", style: "padding:14px;font-size:12px;color:var(--text-dim)" }, text);
    const modelCard = kohaku ? wide(buildModelSection()) : null;      // builds kohakuParts as a side effect
    const kp = kohakuParts;
    const panels = kohaku ? {
      dataset: [buildSubsetsSection(), kp.reading],
      model: [modelCard, kp.adapter, kp.optim],
      training: [kp.duration, kp.memory, kp.placement, kp.objective],
      output: [buildSavingSection(), kp.checkpoints, kp.logging, kp.sampling],
      advanced: [kp.extra, note("Accelerate, experimental and extra sd-scripts arguments do not apply to the low-VRAM trainer. Use the extra knobs box here instead.")],
    } : {
      dataset: [buildSubsetsSection(), buildBucketSection()],
      model: [wide(buildModelSection()), buildNetworkSection(), buildOptimizerSection()],
      training: [buildGeneralSection(), buildNoiseSection(), buildEdmSection()],
      output: [buildSavingSection(), buildLoggingSection(), wide(buildSampleSection())],
      advanced: [buildAccelerateSection(), buildExperimentalSection(), buildExtraArgsSection()],
    };
    for (const [name, sections] of Object.entries(panels)) {
      const container = document.getElementById(`train-panel-${name}`);
      container.innerHTML = "";
      if (name === "model" || name === "output" || (kohaku && name === "training")) {
        const grid = h("div", { class: "panel-grid" });
        grid.append(...sections);
        container.append(grid);
      } else {
        container.append(...sections);
      }
    }
  }

  function initSubtabs() {
    const bar = document.getElementById("train-subtabs");
    bar.addEventListener("click", (e) => {
      const btn = e.target.closest(".train-subtab");
      if (!btn) return;
      bar.querySelectorAll(".train-subtab").forEach((b) => b.classList.toggle("active", b === btn));
      if (btn.dataset.subtab === "queue") refreshQueue();
      document.querySelectorAll(".train-subtab-panel").forEach((p) => {
        p.style.display = p.id === `train-panel-${btn.dataset.subtab}` ? "block" : "none";
      });
    });
  }

  function currentUiPayload() {
    // Deep-clone and post-process the a la carte fields (comma lists -> arrays)
    const ui = JSON.parse(JSON.stringify(state));
    for (const subset of ui.subsets) {
      // nothing extra to post-process currently - kept for parity/extension
    }
    const bw = ui.network.block_weight;
    if (bw && bw.enabled) {
      bw.down = splitNumbers(bw.down);
      bw.mid = bw.mid === "" ? null : Number(bw.mid);
      bw.up = splitNumbers(bw.up);
    }
    for (const key of ["block_dims", "block_alphas", "conv_block_dims", "conv_block_alphas"]) {
      const g = ui.network[key];
      if (g && g.enabled) g.values = splitNumbers(g.values);
    }
    return ui;
  }

  function splitNumbers(text) {
    return (text || "").split(",").map((s) => s.trim()).filter((s) => s !== "").map(Number);
  }

  async function refreshPreview() {
    try {
      const data = await Api.trainBuild(currentUiPayload());
      document.getElementById("toml-preview").textContent = data.toml || "(empty)";
      renderWarnings(data.warnings || []);
      return data;
    } catch (err) {
      document.getElementById("toml-preview").textContent = `Error building config:\n${err.message}`;
    }
  }

  function renderWarnings(warnings) {
    const box = document.getElementById("train-warnings");
    const list = document.getElementById("train-warnings-list");
    if (!warnings.length) { box.style.display = "none"; return; }
    box.style.display = "block";
    list.innerHTML = "";
    warnings.forEach((w) => list.appendChild(h("li", {}, w)));
  }

  async function downloadToml() {
    const ui = currentUiPayload();
    try {
      const resp = await fetch("/api/train/download_toml", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(ui),
      });
      if (!resp.ok) { toast("Couldn't build the config file.", "error"); return; }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = (ui.saving.output_name || "anima_lora_config") + ".toml";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) { toast(err.message, "error"); }
  }

  async function loadTomlFile(file) {
    try {
      const text = await file.text();
      const data = await Api.parseToml(text);
      Object.keys(state).forEach((key) => delete state[key]);
      Object.assign(state, data.ui);
      renderAllSections();
      await refreshPreview();
      toast(`Loaded ${file.name}.`, "success");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function browseTomlFromDisk() {
    FileBrowser.open(null, "toml", async (path) => {
      try {
        const data = await Api.loadTomlFromDisk(path);
        Object.keys(state).forEach((key) => delete state[key]);
        Object.assign(state, data.ui);
        renderAllSections();
        await refreshPreview();
        toast(`Loaded ${path.split(/[\\/]/).pop()}.`, "success");
      } catch (err) {
        toast(err.message, "error");
      }
    });
  }

  function saveTomlToDisk() {
    FolderBrowser.open(null, async (dir) => {
      const filename = prompt("File name:", (state.saving?.output_name || "anima_lora_config") + ".toml");
      if (!filename) return;
      try {
        const data = await Api.saveTomlToDisk(currentUiPayload(), dir, filename);
        toast(`Saved to ${data.path}`, "success");
      } catch (err) {
        toast(err.message, "error");
      }
    });
  }

  // --- Presets ---

  async function refreshPresetList() {
    const data = await Api.listPresets();
    const select = document.getElementById("preset-select");
    select.innerHTML = "";
    select.appendChild(h("option", { value: "" }, "Load a preset…"));
    data.presets.forEach((p) => select.appendChild(h("option", { value: p.name }, p.name)));
  }

  async function savePreset() {
    const name = prompt("Preset name:");
    if (!name) return;
    try {
      await Api.savePreset(name, currentUiPayload());
      toast(`Saved preset "${name}".`, "success");
      await refreshPresetList();
    } catch (err) { toast(err.message, "error"); }
  }

  async function loadPresetByName(name) {
    if (!name) return;
    try {
      const ui = await Api.loadPreset(name);
      Object.assign(state, ui);
      renderAllSections();
      await refreshPreview();
      toast(`Loaded preset "${name}".`, "success");
    } catch (err) { toast(err.message, "error"); }
  }

  // --- Backend control ---

  function setConnDot(dotId, textId, ok, text) {
    const dot = document.getElementById(dotId);
    dot.classList.remove("ok", "bad");
    dot.classList.add(ok ? "ok" : "bad");
    document.getElementById(textId).textContent = text;
  }

  async function checkBackend() {
    const url = document.getElementById("backend-url").value.trim();
    try {
      const result = await Api.backendPing(url);
      if (result.reachable) {
        setConnDot("train-conn-dot", "train-conn-text", true, "Connected");
        setConnDot("backend-dot", "backend-status-text", true, "Backend online");
      } else {
        setConnDot("train-conn-dot", "train-conn-text", false, "Not reachable");
        setConnDot("backend-dot", "backend-status-text", false, "Backend offline");
      }
    } catch (err) {
      setConnDot("train-conn-dot", "train-conn-text", false, "Not reachable");
    }
  }

  let startArmed = null;

  async function startTraining() {
    const btn = document.getElementById("start-training-btn");
    // From another tab the bar would start whatever is configured on the Train tab; make that explicit.
    if (!document.getElementById("view-train").classList.contains("active") && !startArmed) {
      btn.textContent = "Confirm: use Train-tab settings";
      startArmed = setTimeout(() => { startArmed = null; btn.textContent = "Start training"; }, 4500);
      return;
    }
    if (startArmed) { clearTimeout(startArmed); startArmed = null; btn.textContent = "Start training"; }
    const url = document.getElementById("backend-url").value.trim();
    const ui = currentUiPayload();
    btn.disabled = true;
    try {
      await Api.backendStart(ui, url);
      toast("Training started.", "success");
      document.dispatchEvent(new CustomEvent("studio:training-started"));
      dockKick();
    } catch (err) {
      showTrainError("Couldn't start training", err.message, []);
    } finally {
      btn.disabled = false;
    }
  }

  // ---- error panel (above the bar) ----
  let dismissedError = "";
  function explainError(msg) {
    const m = String(msg || "");
    if (/No module named '([^']+)'/.test(m)) return `A Python package is missing in the backend environment (${RegExp.$1}). Open the Backend tab and press Start - it installs missing packages automatically.`;
    if (/CUDA out of memory|OutOfMemory/i.test(m)) return "The GPU ran out of memory. Lower the batch size or resolution, enable gradient checkpointing / block swapping, or close other GPU programs.";
    if (/has no attribute '(\w+)'/.test(m) && /torch\.optim/.test(m)) return "That optimizer name isn't recognised. Check the optimizer type on the Train tab.";
    if (/already set/.test(m)) return "A previous failed run left state behind. Training now runs in a fresh process, so just start again.";
    return "";
  }
  function showTrainError(title, detail, log) {
    const key = `${title}|${detail}`;
    if (key === dismissedError) return;
    const box = document.getElementById("train-error");
    document.getElementById("train-error-title").textContent = title;
    const msg = document.getElementById("train-error-msg");
    msg.textContent = String(detail || "Unknown error");
    const old = box.querySelector(".train-error-hint");
    if (old) old.remove();
    const hint = explainError(`${detail}\n${(log || []).join("\n")}`);
    if (hint) { const d = document.createElement("div"); d.className = "train-error-hint"; d.textContent = hint; msg.after(d); }
    const logEl = document.getElementById("train-error-log");
    logEl.textContent = (log || []).join("\n");
    box.querySelector("details").style.display = (log && log.length) ? "" : "none";
    box.dataset.key = key;
    box.hidden = false;
  }
  function hideTrainError(dismiss) {
    const box = document.getElementById("train-error");
    if (dismiss && box.dataset.key) dismissedError = box.dataset.key;
    box.hidden = true;
  }

  async function stopTraining() {
    const url = document.getElementById("backend-url").value.trim();
    try {
      // A queue job must be cancelled through the queue, otherwise it would be recorded as "completed".
      const running = ((dock.queue && dock.queue.jobs) || []).find((j) => j.status === "running");
      if (running) renderQueue(await Api.queueAction(running.id, "cancel"));
      else await Api.backendStop(url);
      toast("Training stopped.", "info");
      dockKick();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // --- Always-visible training bar ---
  // One polling loop (any tab) drives the bar, the Queue tab and its badge.
  const dock = { timer: null, busy: false, seenSince: null, queue: null, reachable: null, live: false };

  function dockKick() {
    if (dock.timer) clearTimeout(dock.timer);
    dockTick();
  }

  async function dockTick() {
    if (dock.busy) return;
    dock.busy = true;
    const url = document.getElementById("backend-url").value.trim();
    let status = null, prog = null;
    try { status = await Api.backendStatus(url); dock.reachable = true; } catch { dock.reachable = false; }
    if (status && (status.training || status.errored)) { try { prog = await Api.monitorProgress(url); } catch { /* no progress endpoint */ } }
    try { dock.queue = await Api.queueList(); renderQueue(dock.queue); } catch { /* keep last */ }
    try { renderDock(status, prog); } catch (e) { console.error(e); }
    dock.busy = false;
    const wait = document.hidden ? 15000 : dock.live ? 2000 : dock.reachable ? 5000 : 10000;
    dock.timer = setTimeout(dockTick, wait);
  }

  // Self-correcting ETA: the estimator itself lives in ui.js so that the
  // Monitor tab can show the same number.
  const runEta = createEtaEstimator();

  function renderDock(status, prog) {
    const el = document.getElementById("train-dock");
    const live = !!(status && status.training);
    const q = dock.queue || { jobs: [], state: {} };
    const runningJob = q.jobs.find((j) => j.status === "running");
    const waiting = q.jobs.filter((j) => j.status === "queued");
    const paused = !!(q.state && q.state.paused);
    dock.live = live;
    el.classList.toggle("live", live);
    el.classList.toggle("idle", !live);
    el.classList.toggle("compact", !live);
    const stop = document.getElementById("stop-training-btn");
    stop.style.display = live ? "inline-flex" : "none";

    // queue chip (any state)
    const chip = document.getElementById("dock-queue-chip");
    const activeCount = waiting.length + (runningJob ? 1 : 0);
    chip.hidden = !(activeCount || paused);
    if (activeCount || paused) {
      chip.textContent = paused ? `Queue paused · ${waiting.length}` : waiting.length ? `Queue · ${waiting.length} waiting` : "Queue · last job";
      chip.classList.toggle("paused", paused);
    }

    const dot = document.getElementById("train-conn-dot");
    dot.classList.toggle("ok", dock.reachable !== false);
    dot.classList.toggle("bad", dock.reachable === false);

    if (live) hideTrainError(false);
    if (!live && status && status.errored) {
      const p = prog || {};
      showTrainError("Training failed", p.error_detail || "The trainer stopped with an error.", p.log_tail || []);
    }
    if (!live) {
      runEta.reset(); dock.seenSince = null;
      const text = dock.reachable === false ? "Backend not reachable"
        : status && status.errored ? "Last run failed - details above"
        : "Backend connected · idle";
      document.getElementById("train-conn-text").textContent = text;
      document.getElementById("dock-pill").textContent = dock.reachable === false ? "\u25B6 Train (offline)" : "\u25B6 Train";
      return;
    }

    // ---- live view ----
    const now = Date.now() / 1000;
    if (!dock.seenSince) dock.seenSince = runningJob && runningJob.started ? runningJob.started : now;
    const elapsed = now - dock.seenSince;
    const p = prog || {};
    const step = Number(p.step), total = Number(p.total_steps || p.max_step || 0);
    const hasSteps = Number.isFinite(step) && step > 0 && total > 0;
    const stalledReporting = !hasSteps && elapsed > 90;       // trainer only reports start/finish
    const fill = document.getElementById("dock-fill");
    let stats = "", sub = [];
    if (hasSteps) {
      const pct = Math.min(100, (step / total) * 100);
      fill.classList.remove("indeterminate");
      fill.style.width = `${pct.toFixed(1)}%`;
      const bits = [`step ${step}/${total}`];
      if (p.epoch) bits.push(`epoch ${p.epoch}${p.max_epoch ? "/" + p.max_epoch : ""}`);
      if (p.loss != null) bits.push(`loss ${Number(p.loss).toFixed(4)}`);
      stats = `${bits.join(" · ")} · ${pct.toFixed(0)}%`;
      const eta = runEta.update(step, total);
      sub.push(`elapsed ${fmtDuration(elapsed)}`);
      if (eta && eta.stalled) {
        sub.push(`no progress for ${fmtDuration(now - eta.stallSince)} - saving a checkpoint?`);
      } else if (eta) {
        sub.push(`~${fmtDuration(eta.seconds)} remaining · ${eta.rate.toFixed(2)} it/s`);
      } else {
        sub.push("estimating time remaining…");
      }
    } else {
      fill.classList.add("indeterminate");
      fill.style.width = "";
      stats = elapsed < 90 ? "starting…" : "running";
      sub.push(`elapsed ${fmtDuration(elapsed)}`);
      sub.push(stalledReporting ? "time remaining unavailable (this trainer reports progress only at start and finish)" : "waiting for first step…");
    }
    if (waiting.length) sub.push(`queue: ${waiting.length} waiting, next "${waiting[0].name}"`);
    else if (runningJob) sub.push("queue: this is the last job");
    document.getElementById("dock-title").textContent = runningJob ? runningJob.name : "Training";
    document.getElementById("dock-stats").textContent = stats;
    document.getElementById("dock-sub").textContent = sub.join("  ·  ");
    document.getElementById("train-conn-text").textContent = "Training…";
  }

  function initDock() {
    document.getElementById("train-error-dismiss").addEventListener("click", () => hideTrainError(true));
    document.getElementById("train-error-copy").addEventListener("click", async () => {
      const t = `${document.getElementById("train-error-title").textContent}\n${document.getElementById("train-error-msg").textContent}\n\n${document.getElementById("train-error-log").textContent}`;
      try { await navigator.clipboard.writeText(t); toast("Copied.", "success"); } catch { toast("Couldn't copy - select the text manually.", "error"); }
    });
    document.getElementById("dock-queue-chip").addEventListener("click", () => {
      document.querySelector('.nav-item[data-view="train"]').click();
      const tab = document.querySelector('.train-subtab[data-subtab="queue"]');
      if (tab) tab.click();
    });
    dockKick();
  }

  // --- GPU health read-out ---
  // nvidia-smi through the app's own process: no torch, no extra packages. One
  // poll feeds both read-outs: the strip at the top of the Monitor tab, and -
  // only while a run is actually live - a condensed copy on the training bar,
  // where VRAM and temperature are the things you want to watch while training.
  // It lives here with the dock code because the bar's copy needs the dock's
  // live state; keeping one poller means nvidia-smi is asked once per tick.
  let gpuTimer = null;

  function gpuState(g) {
    if (g.temp_c != null && g.temp_c >= 85) return "hot";
    return g.util_pct != null && g.util_pct > 5 ? "active" : "idle";
  }

  function gpuBarColour(pct) {
    return pct >= 92 ? "var(--danger)" : pct >= 75 ? "var(--warn, #c9a227)" : "var(--accent)";
  }

  function gpuFirst(data) {
    return data && data.available ? (data.gpus || [])[0] || null : null;
  }

  function paintGpuBar(el, pct) {
    const pctClamped = Math.min(100, Math.max(0, pct ?? 0));
    el.style.width = `${pctClamped}%`;
    el.style.background = gpuBarColour(pctClamped);
  }

  function renderGpuStrip(data) {
    const strip = document.getElementById("gpu-strip");
    if (!strip) return;
    const g = gpuFirst(data);
    if (!g) { strip.hidden = true; return; }
    strip.hidden = false;
    strip.dataset.state = gpuState(g);
    document.getElementById("gpu-strip-name").textContent =
      data.gpus.length > 1 ? `${g.name} +${data.gpus.length - 1}` : g.name;

    const gb = (mb) => (mb == null ? "?" : (mb / 1024).toFixed(1));
    document.getElementById("gpu-strip-vram").textContent =
      g.mem_total_mb != null ? `VRAM ${gb(g.mem_used_mb)} / ${gb(g.mem_total_mb)} GB` : "";
    paintGpuBar(document.getElementById("gpu-strip-bar"), g.mem_pct);
    document.getElementById("gpu-strip-util").textContent =
      g.util_pct != null ? `${Math.round(g.util_pct)}% util` : "";
    document.getElementById("gpu-strip-temp").textContent =
      g.temp_c != null ? `${Math.round(g.temp_c)}°C` : "";
    document.getElementById("gpu-strip-power").textContent =
      g.power_w != null ? `${Math.round(g.power_w)}${g.power_limit_w ? `/${Math.round(g.power_limit_w)}` : ""} W` : "";
    document.getElementById("gpu-strip-note").textContent =
      g.fan_pct != null ? `fan ${Math.round(g.fan_pct)}%` : "";
  }

  function renderDockGpu(data) {
    const chip = document.getElementById("dock-gpu");
    if (!chip) return;
    const g = gpuFirst(data);
    if (!g) { chip.hidden = true; return; }
    chip.hidden = false;
    chip.dataset.state = gpuState(g);

    const gb = (mb) => (mb == null ? "?" : (mb / 1024).toFixed(1));
    document.getElementById("dock-gpu-util").textContent =
      g.util_pct != null ? `${Math.round(g.util_pct)}%` : "";
    document.getElementById("dock-gpu-mem").textContent =
      g.mem_total_mb != null ? `${gb(g.mem_used_mb)}/${gb(g.mem_total_mb)} GB` : "";
    document.getElementById("dock-gpu-temp").textContent =
      g.temp_c != null ? `${Math.round(g.temp_c)}°C` : "";
    paintGpuBar(document.getElementById("dock-gpu-bar"), g.mem_pct);
    chip.title = `${g.name} - utilisation, VRAM in use and temperature`;
  }

  async function refreshGpu() {
    let data = null;
    try { data = await Api.monitorGpu(); } catch { data = null; }
    renderGpuStrip(data);
    renderDockGpu(data);
  }

  // Only ask when one of the two read-outs is actually on screen: the strip needs
  // the Monitor tab, and the bar's copy only exists while a run is live.
  function gpuWanted() {
    if (document.hidden) return false;
    if (dock.live) return true;
    const monitor = document.getElementById("view-monitor");
    return !!(monitor && monitor.classList.contains("active"));
  }

  function scheduleGpu(delay) {
    if (gpuTimer) clearTimeout(gpuTimer);
    gpuTimer = setTimeout(() => {
      if (gpuWanted()) refreshGpu();
      scheduleGpu(dock.live ? 3000 : 5000);
    }, delay == null ? (dock.live ? 3000 : 5000) : delay);
  }

  function initGpuReadout() {
    if (gpuWanted()) refreshGpu();
    scheduleGpu(1500);
    // coming back to the window should not wait for the next tick to show it
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) refreshGpu();
    });
    // nor should switching to the Monitor tab, where the full strip lives
    document.addEventListener("studio:viewchange", (e) => {
      if (e.detail && e.detail.view === "monitor") refreshGpu();
    });
  }

  // --- Training queue ---

  const QUEUE_LABEL = { queued: "Waiting", running: "Training", done: "Done", error: "Failed", cancelled: "Cancelled" };
  let queueBusy = false;

  async function addToQueue(nameOverride) {
    const url = document.getElementById("backend-url").value.trim();
    const nameEl = document.getElementById("queue-name");
    const name = (nameOverride !== undefined ? nameOverride : nameEl.value).trim();
    try {
      const data = await Api.queueAdd(name, currentUiPayload(), url);
      nameEl.value = "";
      renderQueue(data);
      dockKick();
      toast(`Added "${data.job.name}" to the queue (${data.jobs.filter((j) => j.status === "queued" || j.status === "running").length} active).`, "success");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function queueDo(fn) {
    if (queueBusy) return;
    queueBusy = true;
    try { renderQueue(await fn()); dockKick(); } catch (err) { toast(err.message, "error"); } finally { queueBusy = false; }
  }

  async function refreshQueue() {
    try { renderQueue(await Api.queueList()); } catch { /* backend of this app unreachable; keep last view */ }
  }

  function renderQueue(data) {
    const jobs = data.jobs || [];
    const active = jobs.filter((j) => j.status === "queued" || j.status === "running");
    const badge = document.getElementById("queue-count");
    badge.textContent = active.length;
    badge.style.display = active.length ? "inline-block" : "none";

    const pauseBtn = document.getElementById("queue-pause-btn");
    const paused = data.state && data.state.paused;
    pauseBtn.textContent = paused ? "Resume queue" : "Pause queue";
    pauseBtn.classList.toggle("primary", !!paused);
    pauseBtn.onclick = () => queueDo(() => (paused ? Api.queueResume() : Api.queuePause()));
    const note = document.getElementById("queue-note");
    note.style.display = data.state && data.state.note ? "block" : "none";
    note.textContent = (data.state && data.state.note) || "";

    const list = document.getElementById("queue-list");
    list.innerHTML = "";
    if (!jobs.length) {
      list.appendChild(h("div", { class: "scout-empty", style: "margin-top:12px" }, "The queue is empty. Set up a run in the other tabs, then click \"Add current config to queue\"."));
      return;
    }
    const waiting = jobs.filter((j) => j.status === "queued");
    jobs.forEach((j) => {
      const idx = waiting.findIndex((x) => x.id === j.id);
      const sm = j.summary || {};
      const meta = [sm.model_type, sm.train_mode, sm.output_name ? `→ ${sm.output_name}` : "", sm.subsets ? `${sm.subsets} dataset folder${sm.subsets === 1 ? "" : "s"}` : ""].filter(Boolean).join(" · ");
      let detail = "";
      if (j.status === "running") {
        const p = j.progress || {};
        if (p.waiting) detail = "Waiting for the backend to finish its current run…";
        else {
          const bits = [];
          if (p.epoch != null) bits.push(`epoch ${p.epoch}${p.max_epoch ? "/" + p.max_epoch : ""}`);
          if (p.step != null) bits.push(`step ${p.step}${p.total_steps ? "/" + p.total_steps : ""}`);
          if (p.loss != null) bits.push(`loss ${Number(p.loss).toFixed(4)}`);
          detail = bits.length ? bits.join(" · ") : "Training…";
        }
        if (j.started) detail += ` · running ${fmtDuration(Date.now() / 1000 - j.started)}`;
      } else if (j.status === "queued") {
        detail = idx === 0 ? "Next up" : `Position ${idx + 1} in line`;
      } else {
        detail = j.result || "";
        if (j.started && j.finished) detail += ` (took ${fmtDuration(j.finished - j.started)})`;
      }
      const btn = (label, title, fn, cls = "") => h("button", { class: `btn small ${cls}`, type: "button", title, onclick: fn }, label);
      const actions = [];
      if (j.status === "queued") {
        actions.push(btn("↑", "Move up", () => queueDo(() => Api.queueAction(j.id, "move", { direction: "up" }))));
        actions.push(btn("↓", "Move down", () => queueDo(() => Api.queueAction(j.id, "move", { direction: "down" }))));
        actions.push(btn("⤒", "Run next", () => queueDo(() => Api.queueAction(j.id, "move", { direction: "top" }))));
        actions.push(btn("Cancel", "Take it out of the line", () => queueDo(() => Api.queueAction(j.id, "cancel"))));
      } else if (j.status === "running") {
        actions.push(btn("Stop", "Stop this run and move on", () => queueDo(() => Api.queueAction(j.id, "cancel")), "danger"));
      } else {
        actions.push(btn("Retry", "Put it back at the end of the queue", () => queueDo(() => Api.queueAction(j.id, "retry"))));
        actions.push(btn("✕", "Remove from the list", () => queueDo(() => Api.queueRemove(j.id))));
      }
      list.appendChild(h("div", { class: `queue-row ${j.status}` }, [
        h("span", { class: `queue-badge ${j.status}` }, QUEUE_LABEL[j.status] || j.status),
        h("div", { class: "queue-main" }, [
          h("div", { class: "queue-title" }, j.name),
          h("div", { class: "queue-meta" }, meta),
          h("div", { class: `queue-detail ${j.status}` }, detail),
        ]),
        h("div", { class: "queue-actions" }, actions),
      ]));
    });
  }

  function initQueue() {
    document.getElementById("queue-add-btn").addEventListener("click", () => addToQueue());
    document.getElementById("queue-bar-btn").addEventListener("click", () => addToQueue(""));
    document.getElementById("queue-clear-btn").addEventListener("click", () => queueDo(() => Api.queueClear()));
    refreshQueue();
  }

  // --- Local backend launcher ---

  let launcherStatusTimer = null;

  function renderLauncherStatus(status) {
    const dot = document.getElementById("backend-launch-dot");
    dot.classList.remove("ok", "bad");
    dot.classList.add(status.running ? "ok" : "bad");
    document.getElementById("backend-launch-status-text").textContent =
      status.running ? `Running (${status.command})` : (status.error || "Not running");
    document.getElementById("backend-launch-start").style.display = status.running ? "none" : "inline-flex";
    document.getElementById("backend-launch-stop").style.display = status.running ? "inline-flex" : "none";
    const log = document.getElementById("backend-launch-log");
    if (status.log_tail && status.log_tail.length) {
      log.style.display = "block";
      log.textContent = status.log_tail.join("\n");
      log.scrollTop = log.scrollHeight;
    }
  }

  async function refreshLauncherStatus() {
    try {
      renderLauncherStatus(await Api.backendProcessStatus());
    } catch { /* non-critical */ }
    // also keeps the installer card live when the install was started by the
    // app itself at boot (no click here, so nothing else is polling it)
    try { await refreshSetupStatus(); } catch { /* non-critical */ }
  }

  function pollLauncherStatus() {
    if (launcherStatusTimer) clearInterval(launcherStatusTimer);
    launcherStatusTimer = visibleInterval(refreshLauncherStatus, 2500);
  }

  async function launchBackend() {
    const dir = document.getElementById("backend-launch-dir").value.trim();
    const command = document.getElementById("backend-launch-command").value.trim();
    if (!dir) return toast("Set the backend's folder first.", "error");
    const btn = document.getElementById("backend-launch-start");
    btn.disabled = true;
    try {
      const res = await Api.backendProcessStart(dir, command);
      renderLauncherStatus(res);
      toast(res.repairing ? "Installing missing packages first — watch the Backend tab. It starts automatically afterwards." : "Backend launching…", "success");
      pollLauncherStatus();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  async function stopLauncherBackend() {
    try {
      renderLauncherStatus(await Api.backendProcessStop());
      if (launcherStatusTimer) clearInterval(launcherStatusTimer);
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function saveLauncherSettingsFromForm() {
    const dir = document.getElementById("backend-launch-dir").value.trim();
    const command = document.getElementById("backend-launch-command").value.trim();
    const autostart = document.getElementById("backend-launch-autostart").checked;
    try {
      await Api.backendProcessSaveSettings(dir, command, autostart);
    } catch { /* saved opportunistically - not worth interrupting the user over */ }
  }

  // --- Backend integrity check -------------------------------------------
  // The interesting failures here are partial ones (a venv whose PyTorch never
  // finished, a CUDA build that cannot see the card), so the results are shown
  // as a checklist with a suggested fix per row rather than one pass/fail.
  const VERIFY_ICON = { ok: "✓", warn: "!", fail: "✕" };

  function renderVerifyResults(data) {
    const box = document.getElementById("backend-verify-results");
    const summary = document.getElementById("backend-verify-summary");
    box.replaceChildren(...(data.checks || []).map((c) => h("div", { class: "verify-row", "data-status": c.status }, [
      h("span", { class: "verify-icon" }, VERIFY_ICON[c.status] || "?"),
      h("div", { class: "verify-body" }, [
        h("b", {}, c.label),
        c.detail ? h("small", {}, c.detail) : null,
        c.fix ? h("small", { class: "verify-fix" }, c.fix) : null,
      ]),
    ])));
    box.style.display = "block";
    const warnings = data.warned || 0;
    summary.textContent = data.ok
      ? (warnings ? `Everything needed is in place (${warnings} warning${warnings === 1 ? "" : "s"})` : "Everything needed is in place.")
      : `${data.failed} problem${data.failed === 1 ? "" : "s"} would stop training`;
    summary.style.color = data.ok ? "" : "var(--danger)";
  }

  async function runBackendVerify() {
    const btn = document.getElementById("backend-verify-btn");
    const spinner = document.getElementById("backend-verify-spinner");
    const summary = document.getElementById("backend-verify-summary");
    btn.disabled = true;
    spinner.style.display = "inline-block";
    summary.textContent = "Checking the environment…";
    summary.style.color = "";
    try {
      renderVerifyResults(await Api.backendVerify());
    } catch (err) {
      summary.textContent = err.message || "The check could not run.";
      summary.style.color = "var(--danger)";
    } finally {
      btn.disabled = false;
      spinner.style.display = "none";
    }
  }

  async function initLauncher() {
    try {
      const settings = await Api.backendProcessSettings();
      document.getElementById("backend-launch-dir").value = settings.dir || "";
      document.getElementById("backend-launch-command").value = settings.command || "python backend.py";
      document.getElementById("backend-launch-autostart").checked = !!settings.autostart;
    } catch { /* first run - defaults are fine */ }
    await refreshLauncherStatus();
    pollLauncherStatus();

    document.getElementById("backend-launch-browse").addEventListener("click", () => {
      FolderBrowser.open(document.getElementById("backend-launch-dir").value || null, (path) => {
        document.getElementById("backend-launch-dir").value = path;
        saveLauncherSettingsFromForm();
      });
    });
    document.getElementById("backend-launch-start").addEventListener("click", launchBackend);
    document.getElementById("backend-launch-stop").addEventListener("click", stopLauncherBackend);
    document.getElementById("backend-launch-autostart").addEventListener("change", saveLauncherSettingsFromForm);
    document.getElementById("backend-launch-command").addEventListener("change", saveLauncherSettingsFromForm);
    document.getElementById("backend-launch-dir").addEventListener("change", saveLauncherSettingsFromForm);
    document.getElementById("backend-verify-btn").addEventListener("click", runBackendVerify);
  }

  // --- In-app backend installer (clone + run installer, streamed) ---

  let setupStatusTimer = null;

  function renderSetupStatus(status) {
    const dot = document.getElementById("backend-setup-dot");
    dot.classList.remove("ok", "bad");
    const busy = status.stage === "installing";

    // status.installed reflects the venv actually being on disk right
    // now - not just what this process's own install run did - so a
    // previous session's completed install (or the auto-install that
    // may have started before this tab was even opened) shows correctly
    // instead of looking like "Not started".
    let label;
    if (busy) {
      label = "Running the installer…";
    } else if (status.stage === "error") {
      dot.classList.add("bad");
      label = status.error || "Something went wrong";
    } else if (status.installed) {
      dot.classList.add("ok");
      label = "Installed" + (status.stage === "done" ? " - see the log for next steps" : "");
    } else {
      label = "Not started";
    }
    document.getElementById("backend-setup-status-text").textContent = label;
    document.getElementById("backend-setup-start").disabled = busy;

    const track = document.getElementById("backend-setup-progress-track");
    const fill = document.getElementById("backend-setup-progress-fill");
    if (track && fill) {
      if (busy) {
        track.style.display = "block";
        fill.style.width = `${status.progress || 0}%`;
      } else {
        track.style.display = "none";
        fill.style.width = "0%";
      }
    }

    const log = document.getElementById("backend-setup-log");
    if (status.log_tail && status.log_tail.length) {
      log.style.display = "block";
      log.textContent = status.log_tail.join("\n");
      log.scrollTop = log.scrollHeight;
    }
    document.getElementById("backend-setup-input-row").style.display =
      status.stage === "installing" && status.waiting_for_input ? "flex" : "none";

    if (!busy && setupStatusTimer) {
      clearInterval(setupStatusTimer);
      setupStatusTimer = null;
    }
  }

  async function refreshSetupStatus() {
    try {
      renderSetupStatus(await Api.backendSetupStatus());
    } catch { /* non-critical */ }
  }

  function pollSetupStatus() {
    if (setupStatusTimer) clearInterval(setupStatusTimer);
    setupStatusTimer = visibleInterval(refreshSetupStatus, 1500);
  }

  async function startBackendSetup() {
    try {
      renderSetupStatus(await Api.backendSetupStart());
      toast("Installing backend dependencies - watch the log below.", "success");
      pollSetupStatus();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function sendSetupInput() {
    const input = document.getElementById("backend-setup-input");
    const text = input.value;
    if (!text) return;
    try {
      renderSetupStatus(await Api.backendSetupInput(text));
      input.value = "";
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function initSetup() {
    const s = await Api.backendSetupStatus().catch(() => null);
    if (s) {
      renderSetupStatus(s);
      document.getElementById("backend-setup-target-label").textContent =
        s.bundled ? `Installing into: ${s.bundled_dir}` : "Bundled backend folder not found.";
      document.getElementById("backend-setup-start").disabled = !s.bundled || s.stage === "installing";
      if (s.stage === "installing") pollSetupStatus();
    }

    document.getElementById("backend-setup-start").addEventListener("click", startBackendSetup);
    document.getElementById("backend-setup-send").addEventListener("click", sendSetupInput);
    document.getElementById("backend-setup-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendSetupInput();
    });
  }

  async function init() {
    // Best-effort warm-ups: a failed fetch here must not leave the rest of
    // the tab (buttons, queue, dock, GPU readout, setup) unwired.
    try {
      schemaData = await Api.trainSchema();
      renderAllSections();
      initSubtabs();
      await refreshPreview();
    } catch (err) { toast(err.message, "error"); }
    try { await refreshPresetList(); } catch (err) { toast(err.message, "error"); }

    document.getElementById("preview-refresh-btn").addEventListener("click", refreshPreview);
    document.getElementById("download-toml-btn").addEventListener("click", downloadToml);
    document.getElementById("load-toml-btn").addEventListener("click", () => {
      document.getElementById("load-toml-file").click();
    });
    document.getElementById("load-toml-file").addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) loadTomlFile(file);
      e.target.value = ""; // allow re-selecting the same file later
    });
    document.getElementById("load-toml-disk-btn").addEventListener("click", browseTomlFromDisk);
    document.getElementById("save-toml-disk-btn").addEventListener("click", saveTomlToDisk);
    document.getElementById("preset-save-btn").addEventListener("click", savePreset);
    document.getElementById("preset-select").addEventListener("change", (e) => loadPresetByName(e.target.value));
    document.getElementById("backend-check-btn").addEventListener("click", checkBackend);
    document.getElementById("start-training-btn").addEventListener("click", startTraining);
    document.getElementById("stop-training-btn").addEventListener("click", stopTraining);
    initQueue();
    initDock();
    initGpuReadout();
    await initSetup();
    await initLauncher();
  }

  return {
    init,
    schemaData: () => schemaData,
    state,
    addDatasetFolder: (path) => addDatasetFolder ? addDatasetFolder(path) : false,
    renderAllSections,
    field, textInput, pathInput, numberInput, selectInput, toggleRow, checkboxRow,
    pillGroup, optionalField, keyValueList, section, setPill,
    buildModelSection, buildGeneralSection, buildSubsetsSection, buildBucketSection,
    buildNetworkSection, buildOptimizerSection, buildSavingSection,
    buildNoiseSection, buildSampleSection, buildLoggingSection, buildEdmSection,
    buildExperimentalSection, buildAccelerateSection, buildExtraArgsSection,
    refreshPreview,
  };
})();
