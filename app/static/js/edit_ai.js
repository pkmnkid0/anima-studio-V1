// Edit tab: the AI tools of the Magic eraser. The engine is IOPaint (the project behind lama-cleaner), run by the
// server in its own environment; this file is the screen for it:
//   - Method list: OpenCV (instant, no setup) + AI erase models + Stable Diffusion erase-and-replace / extend
//   - "AI tools" side panel: engine options, the model, EVERY setting the selected model understands (from the catalogue)
//   - Select tool: Segment Anything click-to-select;  Extend tool: outpainting;  Mask: grow / shrink / invert
//   - One-click tools: remove background, upscale (Real-ESRGAN), restore faces (GFPGAN / RestoreFormer)
//   - Results are previewed first: Accept saves (the original is backed up), Discard throws away, Hold to compare
//   - AI batch card: run a tool over the whole folder
const EditAI = (() => {
  const LS_KEY = "anima.edit.ai.v3";
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  let B = null; // bridge to the editor (edit.js)
  let cat = null; // /api/edit/ai/catalog
  let status = null; // /api/edit/ai/status
  let busy = false;
  let currentJob = null;
  let result = null; // {token, img, opts, hasAlpha} - a result waiting for Accept / Discard
  let samClicks = [];
  let excludeMode = false;
  let batchTimer = null;
  let batchSeen = false; // a batch has been seen running: keep polling through hiccups

  const prefs = {
    method: "", custom: "", open: true, engine: {}, params: {}, sam: "mobile_sam",
    bg: { model: "anime-seg", mode: "fill", color: "#ffffff" },
    up: { model: "RealESRGAN_x4plus_anime_6B", scale: 2 },
    face: { model: "gfpgan", bg_upscale: false, bg_model: "RealESRGAN_x4plus_anime_6B" },
    models_dir: "", idle: "",
  };

  function loadPrefs() {
    try {
      const raw = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
      for (const k of Object.keys(prefs)) {
        if (raw[k] === undefined) continue;
        prefs[k] = (prefs[k] && typeof prefs[k] === "object" && !Array.isArray(prefs[k])) ? { ...prefs[k], ...raw[k] } : raw[k];
      }
    } catch { /* storage unavailable: defaults */ }
  }
  function save() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(prefs)); } catch { /* ignore */ }
  }

  // ------------------------------------------------------------------ catalogue helpers

  const entries = () => [...cat.erase_models, ...cat.diffusion_models];
  const isAiMethod = (v) => typeof v === "string" && (v.startsWith("ai:") || v === "custom");
  const methodValue = () => $("eraser-method").value;

  function currentModelId() {
    const v = methodValue();
    if (v === "custom") return (prefs.custom || "").trim();
    return v.startsWith("ai:") ? v.slice(3) : "";
  }
  function entryFor(id) { return entries().find((m) => m.id === id) || null; }
  function tagsFor(id) { const m = entryFor(id); return new Set(m ? m.tags : (cat.custom_model_tags || ["diffusion", "prompt"])); }
  function specVisible(spec, tags, method) {
    if (spec.key === "use_extender" || spec.key.startsWith("extender_")) return false; // driven by the Extend tool
    if (method.startsWith("cv:")) return spec.models.includes("classic") && spec.key !== "cv2_flag";
    return spec.models.some((t) => tags.has(t));
  }

  function engineOptions() {
    const out = {};
    for (const o of cat.engine_options) out[o.key] = o.key in prefs.engine ? prefs.engine[o.key] : o.default;
    return out;
  }

  // The settings sent with an erase / replace / extend request: every control that is visible for the model.
  function paramsPayload(tags) {
    const out = {};
    for (const spec of cat.params) {
      if (!specVisible(spec, tags, "ai:x")) continue;
      const v = spec.key in prefs.params ? prefs.params[spec.key] : spec.default;
      if (v === null || v === undefined || v === "" && spec.kind === "image") continue;
      out[spec.key] = v;
    }
    if (!out.use_croper) for (const k of ["croper_x", "croper_y", "croper_width", "croper_height"]) delete out[k];
    return out;
  }

  function needInstalled() {
    if (status && !status.installed) {
      toast("The AI edit environment isn't installed yet - open the Backend tab and press 'Set up AI edit environment'.", "error");
      renderSide();
      return false;
    }
    return true;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("The result picture could not be loaded."));
      img.src = src;
    });
  }

  // ------------------------------------------------------------------ jobs and the result bar

  function showBar(opts) {
    const bar = $("eraser-result-bar");
    bar.style.display = opts ? "flex" : "none";
    if (!opts) return;
    $("eraser-result-text").textContent = opts.text || "";
    $("eraser-busy-spinner").style.display = opts.busy ? "inline-block" : "none";
    $("eraser-busy-track").style.display = opts.busy ? "block" : "none";
    $("eraser-job-cancel").style.display = opts.busy ? "inline-flex" : "none";
    for (const id of ["eraser-compare", "eraser-discard", "eraser-accept"]) $(id).style.display = opts.done ? "inline-flex" : "none";
    $("eraser-result-bg").style.display = opts.done && opts.alpha ? "flex" : "none";
    if (opts.percent !== undefined) $("eraser-busy-fill").style.width = `${Math.max(3, Math.min(100, opts.percent))}%`;
  }

  function setBusy(on, text) {
    busy = on;
    $("eraser-apply").disabled = on;
    if (on) showBar({ busy: true, text, percent: 3 });
    else if (!result) showBar(null);
  }

  function jobText(job, label) {
    const p = job.progress || {};
    if (p.step && p.total) return `${label} ${p.step}/${p.total}`;
    if (p.message) return p.message.length > 70 ? p.message.slice(-70) : p.message;
    return label;
  }

  async function runJob(body, label) {
    const started = await Api.aiRun(body);
    currentJob = started.id;
    try {
      let cur = started;
      for (;;) {
        const p = cur.progress || {};
        const pct = p.percent != null ? p.percent : (p.step && p.total ? (100 * p.step) / p.total : 6);
        showBar({ busy: true, text: jobText(cur, label), percent: pct });
        if (cur.state !== "running") break;
        await sleep(600);
        cur = await Api.aiJob(started.id);
      }
      if (cur.state === "done") return cur.result || {};
      const err = new Error(cur.error || "The AI engine failed.");
      err.cancelled = cur.state === "cancelled";
      throw err;
    } finally {
      currentJob = null;
      refreshStatus();
    }
  }

  function reportError(err) {
    if (err && err.cancelled) toast("Stopped.", "info");
    else toast(err && err.message ? err.message : String(err), "error");
  }

  async function runAndPreview(body, label, opts, alpha) {
    if (busy) return toast("The AI engine is busy with another step.", "info");
    const path = B.imagePath();
    setBusy(true, label);
    try {
      const res = await runJob(body, label);
      if (!res.token) throw new Error("The AI engine returned no picture.");
      if (B.imagePath() !== path) {               // you moved to another picture while it was working
        Api.aiDiscard(res.token);
        showBar(null);
        return toast("The result was thrown away because you opened a different picture.", "info");
      }
      const img = await loadImage(Api.aiResultUrl(res.token));
      const hasAlpha = !!(alpha || res.has_alpha);
      result = { token: res.token, img, opts, hasAlpha, path };
      showPreview();
      showBar({ done: true, alpha: hasAlpha, text: `Result ready (${img.naturalWidth}×${img.naturalHeight}). Accept to save it - the original is backed up.` });
    } catch (err) {
      showBar(null);
      reportError(err);
    } finally {
      busy = false;
      $("eraser-apply").disabled = false;
    }
  }

  function showPreview() {
    if (!result) return;
    const fill = result.hasAlpha && $("eraser-result-bgmode").value === "fill";
    B.setPreview({ img: result.img, ...result.opts, bg: fill ? $("eraser-result-color").value : null });
  }

  async function accept() {
    if (!result) return;
    const r = result;
    const mode = $("eraser-result-bgmode").value;
    try {
      const data = await Api.aiCommit({
        image_path: r.path, token: r.token, save_as_copy: B.saveAsCopy(),
        keep_alpha: r.hasAlpha && mode === "alpha", background: $("eraser-result-color").value,
      });
      result = null;
      showBar(null);
      await B.afterCommit(data.path);
      toast("Saved.", "success");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function discard() {
    if (B.hasPreview()) B.discardPreview(); // -> onPreviewCleared below
    else onPreviewCleared();
  }

  function onPreviewCleared() {
    if (result) Api.aiDiscard(result.token);
    result = null;
    if (!busy) showBar(null);
  }

  // ------------------------------------------------------------------ erase / replace / extend

  function fitCroper(params, bounds) {
    const s = B.size();
    const margin = 64;
    const size = (need, full) => Math.min(full, Math.max(512, Math.ceil((need + 2 * margin) / 8) * 8));
    const w = size(bounds.x1 - bounds.x0, s.w), hgt = size(bounds.y1 - bounds.y0, s.h);
    const cx = (bounds.x0 + bounds.x1) / 2, cy = (bounds.y0 + bounds.y1) / 2;
    params.croper_width = w;
    params.croper_height = hgt;
    params.croper_x = Math.max(0, Math.min(s.w - w, Math.round(cx - w / 2)));
    params.croper_y = Math.max(0, Math.min(s.h - hgt, Math.round(cy - hgt / 2)));
    for (const k of ["croper_x", "croper_y", "croper_width", "croper_height"]) prefs.params[k] = params[k];
    save();
  }

  async function runInpaint() {
    if (!B.hasImage() || !needInstalled()) return;
    const model = currentModelId();
    if (!model) return toast("Type the model's name under Custom model in the AI tools panel.", "error");
    if (!B.hasMask()) return toast("Paint over (or select) the area first.", "error");
    const params = paramsPayload(tagsFor(model));
    if (params.use_croper) {
      const b = B.maskBounds();
      const inside = b && b.x0 >= params.croper_x && b.y0 >= params.croper_y &&
        b.x1 <= params.croper_x + params.croper_width && b.y1 <= params.croper_y + params.croper_height;
      if (b && !inside) { fitCroper(params, b); renderSide(); toast("Crop box moved to cover the mask.", "info"); }
    }
    const label = tagsFor(model).has("erase") ? "Erasing" : "Generating";
    await runAndPreview({
      cmd: "inpaint", image_path: B.imagePath(), model, params, options: engineOptions(), mask: B.maskDataUrl(),
    }, label, { mode: "same" });
  }

  async function runExtend() {
    if (!B.hasImage() || !needInstalled()) return;
    const model = currentModelId();
    if (!model || !tagsFor(model).has("diffusion")) {
      return toast("Extend needs a Stable Diffusion model: pick one under Method -> AI replace.", "error");
    }
    const pad = B.getPad(), s = B.size();
    if (pad.l + pad.t + pad.r + pad.b === 0) return toast("Set how many pixels to add on at least one side.", "error");
    const params = paramsPayload(tagsFor(model));
    for (const k of ["use_croper", "croper_x", "croper_y", "croper_width", "croper_height"]) delete params[k];
    Object.assign(params, {
      use_extender: true, extender_x: -pad.l, extender_y: -pad.t,
      extender_width: s.w + pad.l + pad.r, extender_height: s.h + pad.t + pad.b,
    });
    await runAndPreview({
      cmd: "inpaint", image_path: B.imagePath(), model, params, options: engineOptions(),
    }, "Extending", { mode: "extend", dx: pad.l, dy: pad.t });
  }

  // ------------------------------------------------------------------ Segment Anything (Select tool)

  async function onSelectClick(pt, include) {
    if (busy) return;
    if (!needInstalled()) return;
    if (excludeMode) include = !include;
    samClicks.push([Math.round(pt.x), Math.round(pt.y), include ? 1 : 0]);
    syncMarkers();
    setBusy(true, "Selecting");
    try {
      const res = await runJob({
        cmd: "segment", image_path: B.imagePath(), clicks: samClicks, model: prefs.sam, options: engineOptions(),
      }, "Selecting");
      const img = await loadImage(Api.aiResultUrl(res.token));
      Api.aiDiscard(res.token);
      B.setSelection(img);
      $("eraser-sel-status").textContent = `${samClicks.length} click${samClicks.length === 1 ? "" : "s"}`;
    } catch (err) {
      samClicks.pop();
      syncMarkers();
      reportError(err);
    } finally {
      busy = false;
      $("eraser-apply").disabled = false;
      showBar(null);
    }
  }

  function syncMarkers() {
    B.setMarkers(samClicks.map((c) => ({ x: c[0], y: c[1], include: !!c[2] })));
  }

  function resetSelectionSession(keep) {
    samClicks = [];
    if (keep) B.commitSelection(); else B.clearSelection();
    $("eraser-sel-status").textContent = "";
  }

  // ------------------------------------------------------------------ mask tools and extend controls

  async function adjustMask(op) {
    if (busy) return;
    if (op !== "reverse" && !B.hasMask()) return toast("Paint or select an area first.", "info");
    try {
      const amount = parseInt($("eraser-mask-amount").value, 10) || 8;
      const data = await Api.maskAdjust(B.maskDataUrl(), op, amount);
      B.setMaskFromImage(await loadImage(data.mask));
      samClicks = [];
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function readPad() {
    const v = (id) => Math.max(0, Math.min(4096, parseInt($(id).value, 10) || 0));
    return { l: v("eraser-ext-l"), t: v("eraser-ext-t"), r: v("eraser-ext-r"), b: v("eraser-ext-b") };
  }

  function writePad(p) {
    $("eraser-ext-l").value = p.l; $("eraser-ext-t").value = p.t; $("eraser-ext-r").value = p.r; $("eraser-ext-b").value = p.b;
    syncPad();
  }

  function syncPad() {
    const p = readPad(), s = B.size();
    B.setPad(p);
    $("eraser-ext-info").textContent = s ? `New size ${s.w + p.l + p.r} × ${s.h + p.t + p.b}` : "";
  }

  function applyPreset(value) {
    const s = B.size();
    if (!value || !s) return;
    if (value.startsWith("all")) {
      const n = parseInt(value.slice(3), 10);
      return writePad({ l: n, t: n, r: n, b: n });
    }
    const [aw, ah] = value.split(":").map(Number);
    const target = aw / ah, cur = s.w / s.h;
    const snap = (n) => Math.floor(n / 8) * 8;
    if (cur < target) {
      const add = Math.max(0, Math.round((s.h * target) / 8) * 8 - s.w);
      const l = snap(add / 2);
      writePad({ l, t: 0, r: add - l, b: 0 });
    } else {
      const add = Math.max(0, Math.round(s.w / target / 8) * 8 - s.h);
      const t = snap(add / 2);
      writePad({ l: 0, t, r: 0, b: add - t });
    }
  }

  // ------------------------------------------------------------------ one-click tools

  async function runPlugin(kind, extra, label, opts) {
    if (!B.hasImage() || !needInstalled()) return;
    await runAndPreview({ cmd: "plugin", plugin: kind, image_path: B.imagePath(), op: "image", options: engineOptions(), ...extra }, label, opts);
  }

  async function selectSubject() {
    if (!B.hasImage() || !needInstalled() || busy) return;
    setBusy(true, "Finding the subject");
    try {
      const res = await runJob({
        cmd: "plugin", plugin: "removebg", model: prefs.bg.model, op: "mask", image_path: B.imagePath(), options: engineOptions(),
      }, "Finding the subject");
      const img = await loadImage(Api.aiResultUrl(res.token));
      Api.aiDiscard(res.token);
      B.setSelection(img);
      B.commitSelection();
      toast("Subject selected. Use Mask -> Invert to select the background instead.", "success");
    } catch (err) {
      reportError(err);
    } finally {
      busy = false;
      $("eraser-apply").disabled = false;
      showBar(null);
    }
  }

  async function downloadModel() {
    const id = currentModelId();
    if (!id || !needInstalled() || busy) return;
    setBusy(true, `Downloading ${id}`);
    try {
      await runJob({ cmd: "download", kind: "model", name: id, options: engineOptions() }, `Downloading ${id}`);
      toast(`${id} is ready.`, "success");
    } catch (err) {
      reportError(err);
    } finally {
      busy = false;
      $("eraser-apply").disabled = false;
      showBar(null);
    }
  }

  // ------------------------------------------------------------------ side panel

  const field = (label, control, hint) => h("div", { class: "ai-field" }, [
    label ? h("div", { class: "ai-label" }, label) : null, control, hint ? h("small", {}, hint) : null,
  ].filter(Boolean));

  function selectEl(options, value, onchange) {
    const sel = h("select", { onchange: (e) => onchange(e.target.value) },
      options.map((o) => { const v = typeof o === "object" ? o.id : o; const t = typeof o === "object" ? o.label : o; return h("option", { value: v }, t); }));
    sel.value = value;
    return sel;
  }

  function checkEl(label, checked, onchange, hint) {
    return h("div", { class: "ai-field" }, [
      h("label", { class: "checkbox-row" }, [h("input", { type: "checkbox", checked: !!checked, onchange: (e) => onchange(e.target.checked) }), label]),
      hint ? h("small", {}, hint) : null,
    ].filter(Boolean));
  }

  function numberEl(value, spec, onchange) {
    return h("input", {
      type: "number", value, min: spec.min, max: spec.max, step: spec.step,
      onchange: (e) => {
        let n = spec.kind === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
        if (Number.isNaN(n)) n = spec.default;
        if (spec.min !== undefined) n = Math.max(spec.min, n);
        if (spec.max !== undefined) n = Math.min(spec.max, n);
        e.target.value = n;
        onchange(n);
      },
    });
  }

  function paramField(spec) {
    const cur = spec.key in prefs.params ? prefs.params[spec.key] : spec.default;
    const set = (v) => { prefs.params[spec.key] = v; save(); };
    if (spec.kind === "bool") return checkEl(spec.label, cur, (v) => { set(v); if (spec.key === "use_croper") renderSide(); }, spec.hint);
    if (spec.kind === "select") return field(spec.label, selectEl(spec.options, cur, set), spec.hint);
    if (spec.kind === "int" || spec.kind === "float") return field(spec.label, numberEl(cur, spec, set), spec.hint);
    if (spec.kind === "area") {
      const ta = h("textarea", { placeholder: spec.label });
      ta.value = cur || "";
      ta.addEventListener("input", () => set(ta.value));
      return field(spec.label, ta, spec.hint);
    }
    if (spec.kind === "image") {
      const thumb = h("img", { class: "ai-example", style: cur ? "" : "display:none" });
      if (cur) thumb.src = cur;
      const file = h("input", { type: "file", accept: "image/*" });
      file.addEventListener("change", async () => {
        const f = file.files && file.files[0];
        if (!f) return;
        try {
          const url = await shrinkToDataUrl(f, 1024);
          set(url);
          thumb.src = url;
          thumb.style.display = "";
        } catch (err) { toast(err.message, "error"); }
      });
      const clear = h("button", { class: "btn small ghost", onclick: () => { set(null); thumb.style.display = "none"; file.value = ""; } }, "Clear");
      return field(spec.label, h("div", {}, [file, clear, thumb]), spec.hint);
    }
    return null;
  }

  function shrinkToDataUrl(file, maxSide) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        const k = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
        const c = document.createElement("canvas");
        c.width = Math.max(1, Math.round(img.naturalWidth * k));
        c.height = Math.max(1, Math.round(img.naturalHeight * k));
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        URL.revokeObjectURL(url);
        resolve(c.toDataURL("image/png"));
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("That file is not a picture.")); };
      img.src = url;
    });
  }

  function details(title, body, open) {
    const d = h("details", {}, [h("summary", {}, title), h("div", { class: "ai-body" }, body)]);
    if (open) d.open = true;
    return d;
  }

  function engineSection() {
    const st = status || {};
    const line = !status ? "Checking the AI engine…"
      : !st.installed ? "The AI edit environment is not installed."
      : st.running ? (st.ready ? "Engine running" + (st.busy ? " (busy)" : "") : "Engine starting…")
      : "Engine idle - it starts when you run a tool and stops by itself after a few idle minutes.";
    const loaded = st.loaded && st.loaded.model ? ` Loaded: ${st.loaded.model}.` : "";
    const rows = [h("div", { class: "ai-status" }, line + loaded)];
    if (status && !st.installed) {
      rows.push(h("div", { class: "ai-row" }, [h("button", { class: "btn small primary", onclick: () => openView("backend") }, "Open the Backend tab")]));
      rows.push(h("small", {}, "Press 'Set up AI edit environment' there (once). The OpenCV methods work without it."));
    } else {
      rows.push(h("div", { class: "ai-row" }, [
        h("button", { class: "btn small", onclick: async () => { await Api.aiEngineStop(); refreshStatus(); toast("AI engine stopped - GPU memory is free.", "info"); } }, "Free GPU memory"),
        h("button", { class: "btn small ghost", onclick: refreshStatus }, "Refresh"),
      ]));
    }
    for (const o of cat.engine_options) {
      const cur = o.key in prefs.engine ? prefs.engine[o.key] : o.default;
      const set = (v) => { prefs.engine[o.key] = v; save(); };
      rows.push(o.kind === "bool" ? checkEl(o.label, cur, set, o.hint) : field(o.label, selectEl(o.options, cur, set), o.hint));
    }
    const dir = h("input", { type: "text", class: "mono", placeholder: "models folder", value: st.models_dir || "" });
    const idle = h("input", { type: "number", min: 0, max: 1440, value: st.idle_minutes != null ? st.idle_minutes : 15 });
    rows.push(field("Models folder", dir, "Where downloaded models are kept (Stable Diffusion models are 2-7 GB each). Changing it restarts the engine."));
    rows.push(field("Stop the engine after idle minutes (0 = never)", idle));
    rows.push(h("div", { class: "ai-row" }, [h("button", { class: "btn small", onclick: async () => {
      try {
        const s = await Api.aiSettings({ models_dir: dir.value.trim(), idle_minutes: idle.value });
        status = s;
        toast("Saved.", "success");
      } catch (err) { toast(err.message, "error"); }
    } }, "Save folder / idle time")]));
    return details("AI engine", rows, !status || !st.installed);
  }

  function modelSection() {
    const v = methodValue();
    const rows = [];
    if (v.startsWith("cv:")) {
      rows.push(h("small", {}, "OpenCV fills the area by smoothing the pixels around it: instant and needs no setup, but it smears textured areas. Pick an AI method for those."));
    } else {
      const id = currentModelId();
      const m = entryFor(id);
      if (v === "custom") {
        const inp = h("input", { type: "text", class: "mono", placeholder: "e.g. author/model, a folder, or a .safetensors in the models folder", value: prefs.custom });
        inp.addEventListener("change", () => { prefs.custom = inp.value; save(); renderSide(); });
        rows.push(field("Custom model", inp, "A Hugging Face repo id, a diffusers folder, or a single-file checkpoint placed in the models folder. It is treated as a prompt-driven Stable Diffusion model."));
      } else if (m) {
        rows.push(h("div", { class: "ai-status" }, [h("b", {}, m.label), m.hint ? h("div", {}, m.hint) : null].filter(Boolean)));
      }
      rows.push(h("small", {}, "Models download automatically the first time they are used."));
      if (id) rows.push(h("div", { class: "ai-row" }, [h("button", { class: "btn small", onclick: downloadModel }, "Download now")]));
    }
    return details("Model", rows, true);
  }

  function settingsSections() {
    const v = methodValue();
    const model = currentModelId();
    const tags = v.startsWith("cv:") ? new Set(["classic"]) : tagsFor(model);
    const out = [];
    for (const g of cat.groups) {
      const specs = cat.params.filter((p) => p.group === g.id && specVisible(p, tags, v));
      if (!specs.length) continue;
      const body = specs.map(paramField).filter(Boolean);
      if (g.id === "regions" && !v.startsWith("cv:")) {
        body.push(h("div", { class: "ai-row" }, [h("button", { class: "btn small", onclick: () => {
          const b = B.maskBounds();
          if (!b) return toast("Paint or select an area first.", "info");
          const tmp = {};
          fitCroper(tmp, b);
          renderSide();
        } }, "Fit crop box to the mask")]));
        body.push(h("small", {}, "The Extend tool sets the extend box for you (Extend by left/top/right/bottom)."));
      }
      out.push(details(g.title, body, ["large", "classic", "prompt", "gen", "ldm", "zits", "pbe", "p2p", "powerpaint"].includes(g.id)));
    }
    return out;
  }

  function actionsSection() {
    const bg = prefs.bg, up = prefs.up, face = prefs.face;
    const bgBox = [
      field("Model", selectEl(cat.remove_bg_models, bg.model, (v) => { bg.model = v; save(); })),
      field("Result", selectEl([{ id: "fill", label: "Fill with colour (for training images)" }, { id: "alpha", label: "Transparent PNG" }], bg.mode, (v) => { bg.mode = v; save(); $("eraser-result-bgmode").value = v; })),
      field("Fill colour", h("input", { type: "color", value: bg.color, onchange: (e) => { bg.color = e.target.value; save(); $("eraser-result-color").value = e.target.value; } })),
      h("div", { class: "ai-row" }, [
        h("button", { class: "btn small primary", onclick: () => { $("eraser-result-bgmode").value = bg.mode; $("eraser-result-color").value = bg.color;
          runPlugin("removebg", { model: bg.model }, "Removing the background", { mode: "same" }); } }, "Remove background"),
        h("button", { class: "btn small", title: "Turns the subject into the mask, so you can then invert it and replace the background", onclick: selectSubject }, "Select subject as mask"),
      ]),
    ];
    const upBox = [
      field("Model", selectEl(cat.realesrgan_models, up.model, (v) => { up.model = v; save(); })),
      field("Scale (1-8)", numberEl(up.scale, { kind: "float", min: 1, max: 8, step: 0.5, default: 2 }, (v) => { up.scale = v; save(); }), "The model works at 4x and the result is resized to this scale."),
      h("div", { class: "ai-row" }, [h("button", { class: "btn small primary", onclick: () =>
        runPlugin("realesrgan", { model: up.model, scale: up.scale }, "Upscaling", { mode: "scale" }) }, "Upscale")]),
    ];
    const faceBox = [
      field("Model", selectEl(cat.face_models, face.model, (v) => { face.model = v; save(); })),
      checkEl("Also upscale the background (Real-ESRGAN)", face.bg_upscale, (v) => { face.bg_upscale = v; save(); }, "Slower; the picture keeps its size."),
      field("Background model", selectEl(cat.realesrgan_models, face.bg_model, (v) => { face.bg_model = v; save(); })),
      h("div", { class: "ai-row" }, [h("button", { class: "btn small primary", onclick: () =>
        runPlugin(face.model, { bg_upscale: face.bg_upscale, bg_model: face.bg_model }, "Restoring faces", { mode: "same" }) }, "Restore faces")]),
    ];
    return [details("Remove background", bgBox, false), details("Upscale (Real-ESRGAN)", upBox, false), details("Restore faces", faceBox, false)];
  }

  function renderSide() {
    const side = $("eraser-ai-side");
    if (!side || !cat) return;
    const keepScroll = side.scrollTop;
    side.replaceChildren(engineSection(), modelSection(), ...settingsSections(), ...actionsSection());
    side.scrollTop = keepScroll;
  }

  function setPanel(open) {
    prefs.open = open;
    save();
    $("eraser-ai-side").style.display = open ? "block" : "none";
    $("eraser-resizer").style.display = open ? "block" : "none";
    $("eraser-ai-toggle").textContent = open ? "AI tools ▾" : "AI tools ▸";
    if (open) { renderSide(); refreshStatus(); }
    B.redraw();
  }

  async function refreshStatus() {
    try {
      status = await Api.aiStatus();
      if (prefs.open) renderSide();
    } catch { /* the server may be restarting */ }
  }

  // ------------------------------------------------------------------ method list

  function fillMethods() {
    const sel = $("eraser-method");
    const group = (label, items) => h("optgroup", { label }, items.map(([v, t]) => h("option", { value: v }, t)));
    sel.replaceChildren(
      group("Fast - OpenCV (no setup)", [["cv:telea", "Telea"], ["cv:ns", "Navier-Stokes"]]),
      group("AI erase (fills the painted area)", cat.erase_models.map((m) => ["ai:" + m.id, m.label])),
      group("AI replace / extend (paints from a prompt)", cat.diffusion_models.map((m) => ["ai:" + m.id, m.label])),
      group("Other", [["custom", "Custom model…"]]),
    );
    const want = prefs.method || (status && status.installed ? "ai:lama" : "cv:telea"); // LaMa is far better than OpenCV: default to it once installed
    sel.value = [...sel.options].some((o) => o.value === want) ? want : "cv:telea";
  }

  // ------------------------------------------------------------------ AI batch (Edit tab card)

  function batchBody() {
    const op = $("ai-batch-op").value;
    const common = { folder: B.folder(), options: engineOptions(), save_as_copy: $("ai-batch-copy").checked };
    if (op === "inpaint") {
      if (!B.hasMask()) throw new Error("Open an image, paint the area to erase (the same place is erased on every picture), then press Start.");
      let model = currentModelId();
      if (!model) model = "lama";
      if (!tagsFor(model).has("erase") && !tagsFor(model).has("diffusion")) model = "lama";
      return { ...common, mode: "inpaint", model, params: paramsPayload(tagsFor(model)), mask: B.maskDataUrl() };
    }
    const base = { ...common, mode: "plugin", plugin: op, background: prefs.bg.color, keep_alpha: false };
    if (op === "removebg") return { ...base, model: prefs.bg.model, keep_alpha: prefs.bg.mode === "alpha" };
    if (op === "realesrgan") return { ...base, model: prefs.up.model, scale: prefs.up.scale };
    return { ...base, bg_upscale: prefs.face.bg_upscale, bg_model: prefs.face.bg_model };
  }

  function renderBatch(s) {
    const running = s.state === "running";
    $("ai-batch-start").disabled = running;
    $("ai-batch-cancel").style.display = running ? "inline-flex" : "none";
    $("ai-batch-dot").className = "conn-dot" + (running ? " ok" : "");
    const text = s.state === "idle" ? "Not started"
      : running ? `Running ${s.done}/${s.total}${s.current ? " - " + s.current : ""}`
      : s.message || s.state;
    $("ai-batch-text").textContent = text;
    $("ai-batch-track").style.display = s.state === "idle" ? "none" : "block";
    $("ai-batch-fill").style.width = `${s.total ? Math.round((100 * s.done) / s.total) : 0}%`;
    const errs = $("ai-batch-errors");
    errs.style.display = s.failed && s.failed.length ? "block" : "none";
    errs.textContent = (s.failed || []).map((f) => `${f.filename}: ${f.error}`).join("\n");
  }

  async function pollBatch() {
    clearTimeout(batchTimer);
    try {
      const s = await Api.aiBatchStatus();
      renderBatch(s);
      if (s.state === "running") { batchSeen = true; batchTimer = setTimeout(pollBatch, 1200); }
      else if (s.state === "done" && batchSeen) { batchSeen = false; document.getElementById("edit-load").click(); }
    } catch { if (batchSeen) batchTimer = setTimeout(pollBatch, 3000); }
  }

  async function startBatch() {
    try {
      if (!B.folder()) throw new Error("Load a folder first.");
      if (status && !status.installed) throw new Error("Set up the AI edit environment in the Backend tab first.");
      const body = batchBody();
      if (!body.save_as_copy && !confirm("This replaces every image in the folder with the AI result. Each original is backed up first (Revert to original brings it back). Continue?")) return;
      renderBatch(await Api.aiBatchStart(body));
      pollBatch();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // ------------------------------------------------------------------ init

  function init(bridge) {
    B = bridge;
    loadPrefs();
    if (!B) return;
    Object.assign(B.hooks, {
      isAiMethod, runInpaint, runExtend, onSelectClick, onPreviewCleared,
      cvRadius: () => {
        const r = parseInt(prefs.params.cv2_radius, 10);
        return Number.isFinite(r) && r > 0 ? r : 4;
      },
      onToolChange: (tool) => {
        if (tool !== "select" && samClicks.length) resetSelectionSession(true);
        if (tool === "extend") syncPad();
        if (cat) renderSide();
      },
      onImageChange: () => { samClicks = []; $("eraser-sel-status").textContent = ""; writePadQuiet(); },
      onMaskCleared: () => { samClicks = []; $("eraser-sel-status").textContent = ""; },
    });

    // toolbar wiring
    $("eraser-ai-toggle").addEventListener("click", () => setPanel(!prefs.open));
    $("eraser-method").addEventListener("change", () => {
      prefs.method = methodValue();
      save();
      if (isAiMethod(prefs.method) && !prefs.open) setPanel(true); else renderSide();
    });
    document.querySelectorAll("[data-mask-op]").forEach((b) => b.addEventListener("click", () => adjustMask(b.dataset.maskOp)));
    $("eraser-accept").addEventListener("click", accept);
    $("eraser-discard").addEventListener("click", discard);
    $("eraser-job-cancel").addEventListener("click", async () => { if (currentJob) await Api.aiJobCancel(currentJob); });
    const cmp = $("eraser-compare");
    for (const ev of ["pointerdown"]) cmp.addEventListener(ev, (e) => { e.preventDefault(); B.setCompare(true); });
    for (const ev of ["pointerup", "pointerleave", "pointercancel"]) cmp.addEventListener(ev, () => B.setCompare(false));
    $("eraser-result-bgmode").addEventListener("change", showPreview);
    $("eraser-result-color").addEventListener("input", showPreview);
    // select tool row
    document.querySelectorAll("#eraser-sel-mode button").forEach((b) => b.addEventListener("click", () => {
      excludeMode = b.dataset.mode === "exclude";
      document.querySelectorAll("#eraser-sel-mode button").forEach((x) => x.classList.toggle("active", x === b));
    }));
    $("eraser-sel-reset").addEventListener("click", () => resetSelectionSession(false));
    $("eraser-sel-done").addEventListener("click", () => resetSelectionSession(true));
    $("eraser-sam-model").addEventListener("change", (e) => { prefs.sam = e.target.value; save(); });
    // extend row
    for (const id of ["eraser-ext-l", "eraser-ext-t", "eraser-ext-r", "eraser-ext-b"]) $(id).addEventListener("input", syncPad);
    $("eraser-ext-preset").addEventListener("change", (e) => { applyPreset(e.target.value); e.target.value = ""; });
    $("eraser-ext-reset").addEventListener("click", () => writePad({ l: 0, t: 0, r: 0, b: 0 }));
    // batch
    $("ai-batch-start").addEventListener("click", startBatch);
    $("ai-batch-cancel").addEventListener("click", async () => { renderBatch(await Api.aiBatchCancel()); pollBatch(); });
    pollBatch();

    document.addEventListener("studio:viewchange", (e) => { if (e.detail && e.detail.view === "edit") refreshStatus(); });

    Api.aiCatalog().then((data) => {
      cat = data.catalog;
      status = data.status;
      fillMethods();
      const sam = $("eraser-sam-model");
      sam.replaceChildren(...cat.sam_models.map((m) => h("option", { value: m.id }, m.label)));
      sam.value = cat.sam_models.some((m) => m.id === prefs.sam) ? prefs.sam : "mobile_sam";
      $("eraser-result-bgmode").value = prefs.bg.mode;
      $("eraser-result-color").value = prefs.bg.color;
      setPanel(prefs.open);
    }).catch(() => {
      // The OpenCV methods keep working; only the AI controls are missing.
      $("eraser-ai-toggle").style.display = "none";
    });
  }

  function writePadQuiet() {
    for (const id of ["eraser-ext-l", "eraser-ext-t", "eraser-ext-r", "eraser-ext-b"]) $(id).value = 0;
    $("eraser-ext-info").textContent = "";
  }

  return { init, isAiMethod, runInpaint, runExtend, paramsPayload: (tags) => paramsPayload(tags) };
})();
