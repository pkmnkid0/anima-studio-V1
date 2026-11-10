// App settings: remembers the non-training settings of the app (Collect, Edit,
// Caption, Explorer, Monitor folders, default log folder, backend URL). Saved automatically, per active project and
// globally, and also as named presets. Training parameters are deliberately
// excluded - those live in the user's own TOML files.
const AppSettings = (() => {
  const SCOPES = ["#view-collect", "#view-edit", "#view-caption", "#view-explore", "#view-monitor"];
  const EXTRA_IDS = ["backend-url"];
  // folders that follow the active project, and secrets, are never captured
  const SKIP_IDS = new Set(["collect-dest-dir", "caption-folder", "edit-folder", "package-out"]);
  const SKIP_RE = /token|password|secret|api.?key|^new-project|^(sel-|blacklist-|cooc-|colour-|prune-|slim-|appsettings-)/i;

  let state = { current: {}, presets: {}, auto_load: true, startup: "last", log_dir: "" };
  let applying = false;
  let saveTimer = null;
  let lastProjectId = null;
  let switchSeq = 0;      // guards an old project's settings load from winning

  const $ = (id) => document.getElementById(id);

  function fields() {
    const out = [];
    for (const sc of SCOPES) {
      document.querySelectorAll(`${sc} input, ${sc} select, ${sc} textarea`).forEach((e) => out.push(e));
    }
    for (const id of EXTRA_IDS) { const e = $(id); if (e) out.push(e); }
    return out.filter((e) => e.id && !SKIP_IDS.has(e.id) && !SKIP_RE.test(e.id)
      && !["file", "password", "button", "submit", "hidden"].includes(e.type));
  }

  const norm = (p) => String(p || "").replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();

  // The Monitor output folder normally follows the active project. Only an
  // explicitly different folder is worth remembering - otherwise a stale path
  // from another project would be restored over the project's own.
  function followsProject(e) {
    if (e.id !== "monitor-output-dir" || typeof Context === "undefined") return false;
    const p = Context.active();
    if (!p) return false;
    const own = p.output_dir || `${String(p.path || "").replace(/[\\/]+$/, "")}/output`;
    return norm(e.value) === norm(own);
  }

  function snapshot() {
    const snap = {};
    for (const e of fields()) {
      if (followsProject(e)) continue;
      if (e.id === "monitor-logging-dir" && state.log_dir) continue;   // the default log folder owns it
      snap[e.id] = e.type === "checkbox" || e.type === "radio" ? !!e.checked : e.value;
    }
    return snap;
  }

  function apply(snap) {
    if (!snap) return 0;
    applying = true;
    let n = 0;
    try {
      for (const e of fields()) {
        if (!(e.id in snap)) continue;
        const v = snap[e.id];
        if (e.type === "checkbox" || e.type === "radio") {
          if (e.checked === !!v) continue;
          e.checked = !!v;
        } else {
          if (e.value === String(v)) continue;
          e.value = v;
          if (e.tagName === "SELECT" && e.value !== String(v)) continue;   // option no longer exists
        }
        e.dispatchEvent(new Event("change", { bubbles: true }));
        n++;
      }
    } finally { applying = false; }
    return n;
  }

  // The dedicated log folder: fills the Train tab's Log directory and Monitor's
  // Logging folder so neither has to be typed again.
  let lastLogDir = "";
  function applyLogDir() {
    const ld = state.log_dir || "";
    if (!ld) { lastLogDir = ""; return; }
    const mon = $("monitor-logging-dir");
    if (mon && mon.value !== ld) {
      mon.value = ld;
      applying = true;
      try { mon.dispatchEvent(new Event("change", { bubbles: true })); } finally { applying = false; }
    }
    if (typeof Train !== "undefined" && Train.state && Train.state.logging) {
      const lg = Train.state.logging;
      if (!lg.logging_dir || lg.logging_dir === lastLogDir) {
        lg.logging_dir = ld;
        try { Train.renderAllSections(); } catch { /* schema not loaded yet */ }
      }
    }
    lastLogDir = ld;
  }

  function scheduleSave() {
    if (applying) return;
    clearTimeout(saveTimer);
    // Snapshot now: by the time the timer fires the user may have switched
    // projects, and the DOM would then hold the next project's settings.
    const snap = snapshot();
    const p = typeof Context !== "undefined" ? Context.active() : null;
    saveTimer = setTimeout(() => save(snap, p ? p.id : null), 900);
  }

  async function save(snap, projectId) {
    state.current = snap;
    try {
      await Api.appSettingsSaveCurrent(snap);
      if (projectId) await Api.projectsUpdate(projectId, { settings: { ui: snap } });
    } catch (err) { console.warn("app settings save failed", err); }
  }

  // ---- panel ----
  function ensurePanel() {
    let m = $("appsettings-modal");
    if (m) return m;
    m = h("div", { class: "modal-backdrop", id: "appsettings-modal" }, [
      h("div", { class: "modal" }, [
        h("div", { class: "modal-head" }, [h("h3", {}, "App settings")]),
        h("div", { class: "modal-body", style: "padding:14px 18px" }, [
          h("p", { class: "card-hint", style: "margin-top:0" },
            "Your Collect, Edit, Caption, Explorer and Monitor (output and log folder) settings are remembered automatically — per project and overall. " +
            "Training parameters are not included (use your .toml files). Save named presets to switch between setups."),
          h("div", { style: "margin:4px 0 14px;padding:10px 12px;border:1px solid var(--border,#2a2d3a);border-radius:8px" }, [
            h("label", { class: "checkbox-row", style: "margin:0" }, [
              h("input", { type: "checkbox", id: "appsettings-autoload" }),
              h("span", {}, "Always load these settings when opening the app"),
            ]),
            h("div", { class: "row", style: "gap:8px;margin-top:8px;align-items:center" }, [
              h("span", { class: "card-hint", style: "margin:0" }, "Load:"),
              h("select", { id: "appsettings-startup", style: "flex:1" }),
            ]),
            h("div", { class: "card-hint", style: "margin:6px 0 0" },
              "Off = the app starts with default values every time (your settings are still saved and your presets are still here)."),
          ]),
          h("div", { style: "margin:0 0 14px;padding:10px 12px;border:1px solid var(--border,#2a2d3a);border-radius:8px" }, [
            h("div", { style: "font-weight:600;margin-bottom:6px" }, "Default log folder"),
            h("div", { class: "row", style: "gap:6px" }, [
              h("input", { type: "text", id: "appsettings-logdir", class: "mono", style: "flex:1", placeholder: "e.g. D:\\anima-logs  (blank = <output>/logs)" }),
              h("button", { class: "btn small", id: "appsettings-logdir-browse", type: "button" }, "…"),
              h("button", { class: "btn small primary", id: "appsettings-logdir-save", type: "button" }, "Set"),
            ]),
            h("div", { class: "card-hint", style: "margin:6px 0 0" },
              "Set once: every training writes its TensorBoard logs here, and Monitor's Logging folder is filled in automatically."),
          ]),
          h("div", { style: "margin:0 0 14px;padding:10px 12px;border:1px solid var(--border,#2a2d3a);border-radius:8px" }, [
            h("div", { style: "font-weight:600;margin-bottom:6px" }, "Hugging Face token"),
            h("div", { class: "row", style: "gap:6px" }, [
              h("input", { type: "password", id: "appsettings-hftoken", class: "mono", style: "flex:1", autocomplete: "off", placeholder: "hf_… (only for gated models)" }),
              h("button", { class: "btn small primary", id: "appsettings-hftoken-save", type: "button" }, "Save"),
              h("button", { class: "btn small", id: "appsettings-hftoken-clear", type: "button" }, "Remove"),
            ]),
            h("div", { class: "card-hint", style: "margin:6px 0 0", id: "appsettings-hftoken-hint" },
              "Some caption models (like Qwen3-VL 8B caption-tuned) are gated: accept the licence on the model's Hugging Face page, " +
              "then paste a free Read token from huggingface.co/settings/tokens. It is stored only on this computer and used for downloads."),
          ]),
          h("div", { id: "appsettings-updates", style: "margin:0 0 14px;padding:10px 12px;border:1px solid var(--border,#2a2d3a);border-radius:8px" }),
          h("details", { style: "margin:0 0 14px;padding:10px 12px;border:1px solid var(--border,#2a2d3a);border-radius:8px", id: "appsettings-keybinds-box" }, [
            h("summary", { style: "font-weight:600;cursor:pointer" }, "Eraser keybinds"),
            h("div", { id: "appsettings-keybinds", style: "margin-top:8px" }),
          ]),
          h("div", { class: "row", style: "gap:6px" }, [
            h("input", { type: "text", id: "appsettings-name", placeholder: "Preset name", style: "flex:1" }),
            h("button", { class: "btn primary", id: "appsettings-save" }, "Save current as preset"),
          ]),
          h("div", { id: "appsettings-list", style: "margin-top:12px" }),
        ]),
        h("div", { class: "modal-foot" }, [h("button", { class: "btn ghost", id: "appsettings-close" }, "Close")]),
      ]),
    ]);
    document.body.appendChild(m);
    if (typeof Updates !== "undefined") Updates.mount($("appsettings-updates"));
    const hfShow = (has) => { $("appsettings-hftoken").placeholder = has ? "Token saved - paste to replace" : "hf_… (only for gated models)"; };
    Api.hfTokenStatus().then((r) => hfShow(r.has_token)).catch(() => {});
    $("appsettings-hftoken-save").addEventListener("click", async () => {
      try { const r = await Api.hfTokenSet($("appsettings-hftoken").value.trim()); $("appsettings-hftoken").value = ""; hfShow(r.has_token); toast(r.has_token ? "Hugging Face token saved" : "Token removed", "success"); }
      catch (err) { toast(err.message, "error"); }
    });
    $("appsettings-hftoken-clear").addEventListener("click", async () => {
      try { await Api.hfTokenSet(""); hfShow(false); toast("Token removed", "success"); } catch (err) { toast(err.message, "error"); }
    });
    $("appsettings-close").addEventListener("click", () => m.classList.remove("open"));
    $("appsettings-keybinds-box").addEventListener("toggle", (e) => { if (e.target.open) Keybinds.renderInto($("appsettings-keybinds")); });
    m.addEventListener("click", (e) => { if (e.target === m) m.classList.remove("open"); });
    $("appsettings-autoload").addEventListener("change", () => saveOptions());
    $("appsettings-startup").addEventListener("change", () => saveOptions());
    $("appsettings-logdir-browse").addEventListener("click", () => {
      FolderBrowser.open($("appsettings-logdir").value || null, (p) => { $("appsettings-logdir").value = p; });
    });
    $("appsettings-logdir-save").addEventListener("click", async () => {
      try {
        state = await Api.appSettingsSaveOptions({ log_dir: $("appsettings-logdir").value.trim() });
        applyLogDir();
        toast(state.log_dir ? "Default log folder set" : "Default log folder cleared", "success");
      } catch (err) { toast(err.message, "error"); }
    });
    $("appsettings-save").addEventListener("click", async () => {
      const name = $("appsettings-name").value.trim();
      try {
        state = await Api.appSettingsSavePreset(name, snapshot());
        $("appsettings-name").value = "";
        toast(`Saved preset "${name}"`, "success");
        renderList();
      } catch (err) { toast(err.message, "error"); }
    });
    return m;
  }

  async function saveOptions() {
    const auto_load = $("appsettings-autoload").checked;
    const startup = $("appsettings-startup").value || "last";
    $("appsettings-startup").disabled = !auto_load;
    try { state = await Api.appSettingsSaveOptions({ auto_load, startup }); }
    catch (err) { toast(err.message, "error"); }
  }

  function renderOptions() {
    const cb = $("appsettings-autoload"), sel = $("appsettings-startup");
    if (!cb || !sel) return;
    $("appsettings-logdir").value = state.log_dir || "";
    cb.checked = state.auto_load !== false;
    sel.innerHTML = "";
    sel.appendChild(h("option", { value: "last" }, "My last-used settings"));
    Object.keys(state.presets || {}).forEach((n) => sel.appendChild(h("option", { value: `preset:${n}` }, `Preset: ${n}`)));
    sel.value = state.startup || "last";
    if (sel.value !== (state.startup || "last")) sel.value = "last";   // that preset was deleted
    sel.disabled = !cb.checked;
  }

  function renderList() {
    renderOptions();
    const box = $("appsettings-list");
    box.innerHTML = "";
    const names = Object.keys(state.presets || {});
    if (!names.length) { box.appendChild(h("div", { class: "card-hint" }, "No presets saved yet.")); return; }
    for (const name of names) {
      box.appendChild(h("div", { class: "row", style: "gap:6px;padding:6px 0;border-top:1px solid var(--border,#2a2d3a)" }, [
        h("span", { style: "flex:1" }, name),
        h("button", { class: "btn small", onclick: () => {
          const n = apply(state.presets[name]); save();
          toast(`Loaded "${name}" (${n} setting${n === 1 ? "" : "s"} changed)`, "success");
        } }, "Load"),
        h("button", { class: "btn small ghost", onclick: async () => {
          state = await Api.appSettingsDeletePreset(name); renderList();
        } }, "Delete"),
      ]));
    }
  }

  async function init() {
    try { state = await Api.appSettingsGet(); } catch { /* first run */ }
    if (typeof Keybinds !== "undefined") Keybinds.hydrate(state.keybinds);
    // "Always load these settings when opening the app" - on by default.
    if (state.auto_load !== false) {
      const st = state.startup || "last";
      const snap = st.startsWith("preset:") ? (state.presets || {})[st.slice(7)] : state.current;
      apply(snap || state.current);
    }
    applyLogDir();   // independent of the startup option: it is a standing setting
    document.addEventListener("change", (e) => {
      if (applying || !e.target || !e.target.id) return;
      if (fields().includes(e.target)) scheduleSave();
    });
    const btn = $("appsettings-btn");
    if (btn) btn.addEventListener("click", () => { ensurePanel().classList.add("open"); renderList(); });
    // restore a project's own settings when you switch to it
    if (typeof Context !== "undefined") {
      Context.onChange(async (p) => {
        const id = p ? p.id : null;
        if (id === lastProjectId) return;
        lastProjectId = id;
        const seq = ++switchSeq;
        if (!id) return;
        let fresh = p;
        try { fresh = await Api.projectsGet(id); } catch { /* use cached */ }
        if (seq !== switchSeq) return;    // a newer project switch won
        const ui = fresh && fresh.settings && fresh.settings.ui;
        if (ui) apply(ui);
      });
    }
  }

  return { init, snapshot, apply };
})();
