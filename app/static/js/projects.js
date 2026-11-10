// Projects view (§1, §21) - one folder per LoRA.
// A project's layout is created server-side; this module
// lists projects and hands their folders to the other tabs.
const Projects = (() => {
  const TYPE_LABELS = {
    character: "Character", outfit: "Outfit", style: "Style",
    concept: "Concept", object: "Object", other: "Other",
  };

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null) node.setAttribute(k, v);
    }
    for (const child of children.flat()) {
      if (child == null) continue;
      node.append(child.nodeType ? child : document.createTextNode(child));
    }
    return node;
  }

  function switchView(name) {
    const btn = document.querySelector(`.nav-item[data-view="${name}"]`);
    if (btn) btn.click();
  }

  function fmtDate(ts) {
    if (!ts) return "";
    const d = new Date(ts * 1000);
    return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  }

  function projectCard(project) {
    const badge = el("span", { class: "badge", style: "background:var(--accent-soft);color:var(--accent)" },
      TYPE_LABELS[project.project_type] || project.project_type);
    const model = el("span", { class: "badge rating-good" }, (project.model_type || "anima").toUpperCase());

    const isActive = typeof Context !== "undefined" && Context.active() && Context.active().id === project.id;
    const actions = el("div", { class: "row", style: "gap:6px;margin-top:auto;padding-top:10px;flex-wrap:wrap" },
      [
        el("button", {
          class: "btn small primary", title: "Make this the active project and restore the settings you used with it",
          onclick: () => {
            Context.setActive(project.id);
            load();
          },
        }, isActive ? "Active ✓" : "Open"),
        el("button", {
          class: "btn small", title: "Use this project's dataset folder in Collect",
          onclick: () => {
            const input = document.getElementById("collect-dest-dir");
            if (input) input.value = (project.dataset_dir || project.path + "/dataset");
            switchView("collect");
            toast(`Collect → ${project.name}/dataset`, "info");
          },
        }, "Collect"),
        el("button", {
          class: "btn small", title: "Open this project's dataset in the Caption tab",
          onclick: () => {
            const input = document.getElementById("caption-folder");
            if (input) input.value = (project.dataset_dir || project.path + "/dataset");
            const load = document.getElementById("caption-load");
            if (load) load.click();
            switchView("caption");
          },
        }, "Caption"),
        el("button", {
          class: "btn small", title: "Add this project's dataset to the Train tab",
          onclick: () => {
            const added = typeof Train !== "undefined" && Train.addDatasetFolder((project.dataset_dir || project.path + "/dataset"));
            if (!added) toast("Dataset folder already added to Train", "info");
            switchView("train");
          },
        }, "Train"),
        el("button", {
          class: "btn small ghost", title: "Delete this project and all its contents",
          onclick: async () => {
            if (!confirm(`Delete project "${project.name}"? This removes its entire folder, including dataset, checkpoints and samples.`)) return;
            try {
              await Api.projectsDelete(project.id);
              toast(`Deleted project "${project.name}"`, "info");
              load();
            } catch (err) {
              toast(err.message, "error");
            }
          },
        }, "Delete"),
      ]);

    return el("div", { class: "card project-card" },
      [
        el("div", { class: "project-head" },
          [
            el("div", { class: "project-title" }, [project.name]),
            el("div", { class: "row", style: "gap:6px" }, [badge, model]),
          ]),
        el("div", { class: "project-meta" },
          [
            el("span", {}, `${project.image_count || 0} images`),
            el("span", {}, `·`),
            el("span", {}, fmtDate(project.created)),
          ]),
        project.trigger_word ? el("div", { class: "project-meta" }, `trigger: ${project.trigger_word}`) : null,
        el("div", { class: "project-meta" }, project.last_opened ? `last opened ${fmtDate(project.last_opened)}` : ""),
        el("div", { class: "project-path mono", title: project.path }, project.path),
        actions,
      ]);
  }

  let searchTimer = null;
  let loadSeq = 0;

  async function load() {
    const grid = document.getElementById("project-grid");
    const empty = document.getElementById("projects-empty");
    if (!grid) return;
    const seq = ++loadSeq;         // a slow listing must not append twice
    try {
      const q = (document.getElementById("project-search") || {}).value || "";
      const { projects, defaults } = await Api.projectsList(q.trim());
      if (seq !== loadSeq) return;
      if (defaults && !document.getElementById("new-project-type").dataset.touched) {
        document.getElementById("new-project-type").value = defaults.project_type;
        document.getElementById("new-project-model").value = defaults.model_type;
      }
      const cnt = document.getElementById("project-count");
      if (cnt) cnt.textContent = `${projects.length} project${projects.length === 1 ? "" : "s"}`;
      // Keep the topbar switcher in step: creating or deleting a project
      // here has to be reflected in the dropdown immediately, or it keeps
      // offering a deleted project until the next page load.
      // Probe the bare identifier: Context is a `const` in a classic script,
      // so it is a lexical global and never a property of window.
      if (typeof Context !== "undefined") await Context.refresh();
      if (seq !== loadSeq) return;
      grid.innerHTML = "";
      if (!projects.length) {
        empty.style.display = "block";
        empty.querySelector("h3").textContent = q.trim() ? "No projects match" : "No projects yet";
        return;
      }
      empty.style.display = "none";
      projects.forEach((p) => grid.appendChild(projectCard(p)));
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function init() {
    const btn = document.getElementById("new-project-btn");
    const nameEl = document.getElementById("new-project-name");
    for (const id of ["new-project-type", "new-project-model"]) {
      document.getElementById(id).addEventListener("change", (e) => { e.target.dataset.touched = "1"; });
    }
    document.getElementById("new-project-dataset-browse").addEventListener("click", () => {
      const inp = document.getElementById("new-project-dataset");
      FolderBrowser.open(inp.value || null, (p) => { inp.value = p; });
    });
    const create = async () => {
      const name = nameEl.value.trim();
      if (!name) { toast("Give the project a name first.", "error"); return; }
      const type = document.getElementById("new-project-type").value;
      const model = document.getElementById("new-project-model").value;
      const ds = document.getElementById("new-project-dataset").value.trim();
      const status = document.getElementById("new-project-status");
      status.textContent = "Creating…";
      try {
        const project = await Api.projectsCreate(name, type, model, ds);
        toast(`Created project "${project.name}"`, "success");
        nameEl.value = "";
        document.getElementById("new-project-dataset").value = "";
        status.textContent = `Created at ${project.path}`;
        await load();
        Context.setActive(project.id);   // start working in it right away
      } catch (err) {
        status.textContent = "";
        toast(err.message, "error");
      }
    };
    btn.addEventListener("click", create);
    nameEl.addEventListener("keydown", (e) => { if (e.key === "Enter") create(); });
    document.getElementById("project-search").addEventListener("input", () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(load, 200);
    });
    const refresh = document.getElementById("project-refresh-btn");
    if (refresh) refresh.addEventListener("click", load);
    load();
  }

  return { init, load };
})();
