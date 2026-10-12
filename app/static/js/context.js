// Active-project context (§ the "which LoRA am I on?" problem).
//
// Before this existed each tab held its own folder path and nothing tied
// them together: you could point Collect at one folder, Caption at another,
// and Train at a third, with no indication anywhere that it was happening.
// A project already exists server-side (projects.py, one folder per LoRA with
// a fixed layout), so this module makes it the *selected* thing and derives
// every tab's folder from it.
//
// Two design points worth knowing before editing:
//
// 1. Switching a project does NOT clobber fields the user has touched. Each
//    target input records the value we last wrote; if the current value
//    differs, the user edited it and we leave it alone (and say so). Without
//    this, typing a custom path into Collect and then switching projects
//    would silently destroy it.
//
// 2. Tab modules are loaded but their init() may not have run yet (Train
//    fetches its schema asynchronously). So writes go through each module's
//    optional setter when one exists and fall back to setting the input
//    directly; the input is always written so a later init() that reads the
//    DOM picks up the right value.
const Context = (() => {
  const STORAGE_KEY = "animaStudio.activeProject";
  const TYPE_LABELS = {
    character: "Character", outfit: "Outfit", style: "Style",
    concept: "Concept", object: "Object", other: "Other",
  };

  let projects = [];
  let activeId = null;
  // input id -> the value we last wrote into it, used to detect user edits
  const lastWritten = new Map();

  const $ = (id) => document.getElementById(id);

  function active() {
    return projects.find((p) => p.id === activeId) || null;
  }

  function readStored() {
    try {
      return localStorage.getItem(STORAGE_KEY) || "";
    } catch {
      return "";
    }
  }

  function writeStored(id) {
    try {
      if (id) localStorage.setItem(STORAGE_KEY, id);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* private mode / disabled storage - the switcher still works for this
         session, it just won't survive a reload. Not worth a toast. */
    }
  }

  function join(base, sub) {
    const root = String(base).replace(/[\\/]+$/, "");
    // Match the separator style of the base path. A project path comes back
    // from the server as "C:\...\shiori_8a61af", and appending "/" gave us
    // "C:\...\shiori_8a61af/dataset" - which works in Python but looks like
    // a mistake to anyone reading the input box.
    const sep = /^[A-Za-z]:[\\/]/.test(root) ? "\\" : "/";
    return `${root}${sep}${sub}`;
  }

  // Where each tab should read from, given a project. Only inputs that map
  // unambiguously onto the project layout appear here - anything ambiguous
  // is left alone rather than guessed at.
  const datasetOf = (p) => p.dataset_dir || join(p.path, "dataset");
  const outputOf = (p) => p.output_dir || join(p.path, "output");

  function targetsFor(project) {
    return [
      { id: "collect-dest-dir", value: datasetOf(project), label: "Collect" },
      { id: "caption-folder", value: datasetOf(project), label: "Caption" },
      { id: "edit-folder", value: datasetOf(project), label: "Edit" },
      { id: "monitor-output-dir", value: outputOf(project), label: "Monitor" },
      { id: "package-out", value: outputOf(project), label: "Package" },
    ];
  }

  // Returns "changed" only when the value actually moved, so callers can
  // avoid pointless reloads when the project resolves to the same path.
  function setInput(id, value, { force = false } = {}) {
    const el = $(id);
    if (!el) return "missing";
    const current = el.value || "";
    const wasWritten = lastWritten.get(id);
    const untouched = !current || current === wasWritten;
    if (!force && !untouched) return "user-edited";
    if (current === value) return "unchanged";
    el.value = value;
    lastWritten.set(id, value);
    return "changed";
  }

  // Fired after the project changes so tabs can react (load the folder, etc).
  const listeners = new Set();
  function onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }

  function emit() {
    const project = active();
    listeners.forEach((fn) => {
      try { fn(project); } catch (err) { console.error("context listener", err); }
    });
  }

  function apply({ silent = false } = {}) {
    const project = active();
    const skipped = [];

    if (project) {
      let monitorMoved = false;
      for (const t of targetsFor(project)) {
        const result = setInput(t.id, t.value);
        if (result === "user-edited") skipped.push(t.label);
        if (t.id === "monitor-output-dir" && result === "changed") monitorMoved = true;
      }
      // Setting .value directly doesn't fire a change event, so Monitor's
      // own listener never runs. Without this nudge its panel would sit on
      // the previous project's samples - or stay blank, since its polling
      // is gated on the tab being visible.
      if (monitorMoved && typeof Monitor !== "undefined") {
        try { Monitor.onFolderChanged(); } catch (err) { console.warn("context: monitor refresh", err); }
      }
      // Train keeps its dataset folders and output dir in JS state rather
      // than the DOM, so it needs handling separately.
      //
      // NOTE: these modules are declared with `const X = (() => {...})()` in
      // classic <script> tags. That creates a *lexical* global binding, which
      // is NOT a property of `window`. Testing `window.Train` therefore
      // silently reports undefined forever and every Train write below is
      // skipped - so probe the bare identifier instead.
      if (typeof Train !== "undefined") {
        try { Train.addDatasetFolder(datasetOf(project)); } catch (err) { console.warn("context: addDatasetFolder", err); }
        const outputDir = outputOf(project);
        const prev = lastWritten.get("train-output-dir");
        const saving = Train.state && Train.state.saving;
        if (saving && (!saving.output_dir || saving.output_dir === prev)) {
          saving.output_dir = outputDir;
          lastWritten.set("train-output-dir", outputDir);
          // Re-render so the Saving section's path input picks this up.
          try { Train.renderAllSections(); } catch { /* schema not loaded yet */ }
        }
      }
    }

    renderContextBar();
    emit();

    if (!silent) {
      if (project && skipped.length) {
        toast(
          `Switched to "${project.name}". Left your own folder in ${skipped.join(", ")} alone.`,
          "info",
        );
      } else if (project) {
        toast(`Project: ${project.name}`, "success");
      }
    }
  }

  function renderSelect() {
    const sel = $("project-switch-select");
    if (!sel) return;
    sel.innerHTML = "";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = projects.length ? "No project selected" : "No projects yet";
    sel.appendChild(blank);
    for (const p of projects) {
      const opt = document.createElement("option");
      opt.value = p.id;
      const imgs = Number(p.image_count || 0);
      opt.textContent = `${p.name} (${TYPE_LABELS[p.project_type] || p.project_type}, ${imgs} img)`;
      sel.appendChild(opt);
    }
    sel.value = activeId || "";
    const wrap = $("project-switch");
    if (wrap) wrap.classList.toggle("is-empty", !activeId);
  }

  function renderContextBar() {
    const bar = $("context-bar");
    const project = active();
    if (!bar) return;
    if (!project) {
      bar.style.display = "none";
      return;
    }
    bar.style.display = "flex";

    const name = $("context-name");
    if (name) name.textContent = project.name;

    const type = $("context-type");
    if (type) type.textContent = TYPE_LABELS[project.project_type] || project.project_type;

    const model = $("context-model");
    if (model) model.textContent = String(project.model_type || "anima").toUpperCase();

    const stats = $("context-stats");
    if (stats) {
      const imgs = Number(project.image_count || 0);
      stats.innerHTML = "";
      const mk = (text) => {
        const s = document.createElement("span");
        s.textContent = text;
        return s;
      };
      stats.append(mk(`${imgs} image${imgs === 1 ? "" : "s"}`));
      if (project.trigger_word) {
        const sep = mk("·");
        sep.className = "dot-sep";
        stats.append(sep, mk(`trigger "${project.trigger_word}"`));
      }
    }

    const path = $("context-path");
    if (path) {
      path.textContent = project.path;
      path.title = project.path;
    }
  }

  function setActive(id, opts) {
    activeId = id || null;
    writeStored(activeId);
    if (activeId) {
      // bump "last opened" so the project floats to the top of the recent list
      Api.projectsOpen(activeId).then((p) => {
        const i = projects.findIndex((x) => x.id === p.id);
        if (i >= 0) projects[i] = { ...projects[i], last_opened: p.last_opened };
      }).catch(() => {});
    }
    // renderSelect() owns the .is-empty highlight, and it's the only thing
    // that knows whether the dropdown reflects activeId - so it has to run
    // on every selection change, not just on a project-list refresh.
    renderSelect();
    apply(opts);
  }

  // Re-read the project list. Called on boot and after any create/delete so
  // the switcher never offers a project that has been removed.
  async function refresh({ keepSelection = true } = {}) {
    let data;
    try {
      data = await Api.projectsList();
    } catch (err) {
      console.error("context: project list failed", err);
      return;
    }
    projects = (data && data.projects) || [];

    const stored = readStored();
    if (!keepSelection) activeId = null;
    if (activeId && !projects.some((p) => p.id === activeId)) {
      // The selected project was deleted out from under us.
      activeId = null;
    }
    if (!activeId && stored && projects.some((p) => p.id === stored)) {
      activeId = stored;
    }

    renderSelect();
    if (activeId) {
      // Restore the derived folders without a toast - this is boot, and
      // repeated "Project: X" toasts on every page load would be noise.
      apply({ silent: true });
    } else {
      renderContextBar();
    }
  }

  function goToView(name) {
    const btn = document.querySelector(`.nav-item[data-view="${name}"]`);
    if (btn) btn.click();
  }

  function wire() {
    const sel = $("project-switch-select");
    if (sel) {
      sel.addEventListener("change", (e) => setActive(e.target.value));
    }

    const manage = $("project-switch-manage");
    if (manage) manage.addEventListener("click", () => goToView("projects"));

    const goCaption = $("context-go-caption");
    if (goCaption) {
      goCaption.addEventListener("click", () => {
        const project = active();
        if (!project) return;
        const path = datasetOf(project);
        // Caption.setFolder also triggers its load(); fall back to just
        // setting the input if the module isn't up yet.
        if (typeof Caption !== "undefined" && Caption.setFolder) {
          Caption.setFolder(path);
        } else {
          setInput("caption-folder", path, { force: true });
        }
        goToView("caption");
      });
    }

    const goTrain = $("context-go-train");
    if (goTrain) {
      goTrain.addEventListener("click", () => {
        const project = active();
        if (!project) return;
        if (typeof Train !== "undefined") {
          const added = Train.addDatasetFolder(datasetOf(project));
          if (!added) toast("That dataset is already in the Train tab", "info");
        }
        goToView("train");
      });
    }

      }

  function init() {
    wire();
    return refresh();
  }

  return { init, refresh, onChange, active, setActive, get projects() { return projects; } };
})();