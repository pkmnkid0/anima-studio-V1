// App settings: remembers the non-training settings of the app (Collect, Edit,
// Caption, Explorer, backend URL). Saved automatically, per active project and
// globally, and also as named presets. Training parameters are deliberately
// excluded - those live in the user's own TOML files.
const AppSettings = (() => {
  const SCOPES = ["#view-collect", "#view-edit", "#view-caption", "#view-explore"];
  const EXTRA_IDS = ["backend-url"];
  // folders that follow the active project, and secrets, are never captured
  const SKIP_IDS = new Set(["collect-dest-dir", "caption-folder", "edit-folder", "monitor-output-dir", "package-out"]);
  const SKIP_RE = /token|password|secret|api.?key|^new-project/i;

  let state = { current: {}, presets: {} };
  let applying = false;
  let saveTimer = null;
  let lastProjectId = null;

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

  function snapshot() {
    const snap = {};
    for (const e of fields()) snap[e.id] = e.type === "checkbox" || e.type === "radio" ? !!e.checked : e.value;
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

  function scheduleSave() {
    if (applying) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(save, 900);
  }

  async function save() {
    const snap = snapshot();
    state.current = snap;
    try {
      await Api.appSettingsSaveCurrent(snap);
      const p = typeof Context !== "undefined" ? Context.active() : null;
      if (p) await Api.projectsUpdate(p.id, { settings: { ui: snap } });
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
            "Your Collect, Edit, Caption and Explorer settings are remembered automatically — per project and overall. " +
            "Training parameters are not included (use your .toml files). Save named presets to switch between setups."),
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
    $("appsettings-close").addEventListener("click", () => m.classList.remove("open"));
    m.addEventListener("click", (e) => { if (e.target === m) m.classList.remove("open"); });
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

  function renderList() {
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
    apply(state.current);
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
        if (!id) return;
        let fresh = p;
        try { fresh = await Api.projectsGet(id); } catch { /* use cached */ }
        const ui = fresh && fresh.settings && fresh.settings.ui;
        if (ui) apply(ui);
      });
    }
  }

  return { init, snapshot, apply };
})();
