// Small DOM + UX helpers shared across tabs.

function h(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (v === false || v === null || v === undefined) continue;   // false must NOT become checked="false"
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  // .flat(Infinity) absorbs any accidentally-nested arrays (e.g. a bare
  // .map() result dropped into a children array without being spread).
  const flatChildren = [].concat(children).flat(Infinity);
  for (const child of flatChildren) {
    if (child === null || child === undefined || child === false) continue;
    el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function toast(message, type = "info") {
  const stack = document.getElementById("toast-stack");
  const el = h("div", { class: `toast ${type}` }, message);
  stack.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

function ratingBadge(rating) {
  const r = rating || "unknown";
  return h("span", { class: `badge rating-${r}` }, [h("span", { class: `rating-dot ${r}` }), r]);
}

function debounce(fn, wait = 350) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

/** setInterval that skips ticks while the tab is hidden — a minimised window
 *  otherwise keeps polling the app (and the GPU) for nobody. */
function visibleInterval(fn, ms) {
  return setInterval(() => {
    if (!document.hidden) fn();
  }, ms);
}

/** Switch to another tab, exactly as clicking its nav item would. */
function openView(name) {
  const btn = document.querySelector(`.nav-item[data-view="${name}"]`);
  if (btn) btn.click();
}

// ------------------------- Subset tab strip (Caption + Edit tabs) -------------------------
// A rail of little tabs for quickly switching between datasets: every
// folder under the datasets root, plus any folder anywhere on disk the
// user has explicitly added via "+ Add subset" - and, separately, every
// immediate subfolder of whatever folder is currently loaded (the
// common "10_concept" style layout, where the real image subsets live
// one level below the folder you'd naturally point this at) so those
// show up as their own tabs too instead of silently looking empty.
const SubsetTabs = (() => {
  // Per-container "which folder's children are we showing as the 'In
  // this folder' group" - kept stable across clicking between siblings,
  // so picking one subset doesn't make its neighbors disappear (they'd
  // otherwise be recomputed as children of whatever's now loaded, and a
  // leaf subset - the normal case, just full of images - has none).
  const _browseRoots = new Map(); // containerId -> folder path
  const _browseChildren = new Map(); // containerId -> that folder's children (cached)

  function samePath(a, b) {
    if (!a || !b) return false;
    const norm = (p) => p.replace(/[\\/]+$/, "").toLowerCase();
    return norm(a) === norm(b);
  }

  function isRootOrKnownChild(path, root, children) {
    return samePath(path, root) || children.some((c) => samePath(c.path, path));
  }

  // ---- per-viewer sidebar prefs (collapsed sections, hidden rows, hide-empty) ----
  const PREF_KEY = "anima.subsetSidebar.v1";
  function loadPrefs() {
    try { return Object.assign({ collapsed: {}, hidden: [], hideEmpty: false }, JSON.parse(localStorage.getItem(PREF_KEY) || "{}")); }
    catch { return { collapsed: {}, hidden: [], hideEmpty: false }; }
  }
  function savePrefs(p) { try { localStorage.setItem(PREF_KEY, JSON.stringify(p)); } catch { /* storage unavailable - fine */ } }
  const normPath = (p) => (p || "").replace(/[\\/]+$/, "").toLowerCase();

  function tabEl(entry, activePath, onSelect, containerId, render, prefs) {
    const isActive = samePath(entry.path, activePath);
    const empty = Number(entry.image_count) === 0;
    const tab = h("button", {
      type: "button",
      class: `subset-tab${isActive ? " active" : ""}${empty ? " is-empty" : ""}`,
      title: entry.path,
      onclick: () => onSelect(entry.path),
    }, [
      h("span", { class: "subset-tab-name" }, entry.name),
      h("span", { class: "subset-tab-count" }, String(entry.image_count)),
    ]);
    if (entry.external) {
      tab.appendChild(h("span", {
        class: "subset-tab-remove",
        title: "Remove from this list (your files are untouched)",
        onclick: async (e) => {
          e.stopPropagation();
          try {
            await Api.removeSubset(entry.path);
            render(containerId, activePath, onSelect);
          } catch (err) {
            toast(err.message, "error");
          }
        },
      }, "×"));
    } else if (!isActive) {
      tab.appendChild(h("span", {
        class: "subset-tab-remove",
        title: "Hide from this list (nothing is deleted)",
        onclick: (e) => {
          e.stopPropagation();
          prefs.hidden.push(normPath(entry.path));
          savePrefs(prefs);
          render(containerId, activePath, onSelect);
        },
      }, "×"));
    }
    return tab;
  }

  async function render(containerId, activePath, onSelect) {
    const container = document.getElementById(containerId);
    if (!container) return;

    let subsets = [];
    try {
      subsets = (await Api.listSubsets()).subsets || [];
    } catch { /* tabs are a convenience layer - the manual path field above still works */ }

    // Subfolders of a stable "root" folder - this is what makes a
    // "<dataset>/10_concept/20_other" layout usable: loading the outer
    // folder shows nothing directly, but its subsets appear here as
    // one-click tabs. The root only changes when the newly active
    // folder isn't part of the current sibling group at all (a genuinely
    // different dataset was loaded) - selecting a sibling within the
    // group keeps the whole group visible instead of collapsing it.
    let root = _browseRoots.get(containerId) || null;
    let children = _browseChildren.get(containerId) || [];

    if (!activePath) {
      root = null;
      children = [];
      _browseRoots.delete(containerId);
      _browseChildren.delete(containerId);
    } else if (!root || !isRootOrKnownChild(activePath, root, children)) {
      root = activePath;
      try {
        const data = await Api.browse(root);
        children = (data.dirs || []).filter((d) => !subsets.some((s) => samePath(s.path, d.path)));
      } catch {
        children = []; // root might not be browsable (e.g. not created yet) - fine, just skip
      }
      _browseRoots.set(containerId, root);
      _browseChildren.set(containerId, children);
    }
    // else: activePath is the current root or one of its already-known
    // children/siblings - keep showing the same group, just re-render
    // below with the new tab marked active.

    container.innerHTML = "";
    const prefs = loadPrefs();
    const isHidden = (e) => prefs.hidden.includes(normPath(e.path)) && !samePath(e.path, activePath);
    const isEmptyHidden = (e) => prefs.hideEmpty && Number(e.image_count) === 0 && !samePath(e.path, activePath);

    // A collapsible group: header (chevron, title, row count) + its rows.
    function group(key, title, entries, hint) {
      const visible = entries.filter((e) => !isHidden(e) && !isEmptyHidden(e));
      if (!entries.length) return;
      const collapsed = !!prefs.collapsed[key] && !entries.some((e) => samePath(e.path, activePath));
      const head = h("button", {
        type: "button", class: "subset-sidebar-label subset-group-head", title: hint || "",
        onclick: () => { prefs.collapsed[key] = !collapsed; savePrefs(prefs); render(containerId, activePath, onSelect); },
      }, [
        h("span", { class: "subset-chevron" }, collapsed ? "▸" : "▾"),
        h("span", { class: "subset-group-title" }, title),
        h("span", { class: "subset-group-count" }, String(visible.length)),
      ]);
      container.appendChild(head);
      if (!collapsed) visible.forEach((e) => container.appendChild(tabEl(e, activePath, onSelect, containerId, render, prefs)));
    }

    const rootName = root ? (root.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || root) : "";
    group("datasets", "Your datasets", subsets, "Datasets in your library and folders you added with + Add subset");
    group("inside", rootName ? `Inside "${rootName}"` : "Subfolders", children,
      "Subfolders of the folder you have open");

    // Footer: hide-empty switch, and a way back for rows you hid.
    const all = [...subsets, ...children];
    const emptyCount = all.filter((e) => Number(e.image_count) === 0 && !samePath(e.path, activePath)).length;
    const hiddenCount = all.filter((e) => prefs.hidden.includes(normPath(e.path)) && !samePath(e.path, activePath)).length;
    if (emptyCount) {
      container.appendChild(h("label", { class: "checkbox-row subset-opt" }, [
        h("input", { type: "checkbox", checked: prefs.hideEmpty,
          onchange: (e) => { prefs.hideEmpty = e.target.checked; savePrefs(prefs); render(containerId, activePath, onSelect); } }),
        h("span", {}, `Hide empty folders (${emptyCount})`),
      ]));
    }
    if (hiddenCount) {
      container.appendChild(h("button", { type: "button", class: "subset-opt subset-opt-link",
        onclick: () => { prefs.hidden = []; savePrefs(prefs); render(containerId, activePath, onSelect); } },
        `Show ${hiddenCount} hidden`));
    }

    container.appendChild(h("button", {
      type: "button",
      class: "subset-tab subset-tab-add",
      title: "Add a folder from anywhere on your PC as a subset tab",
      onclick: () => {
        FolderBrowser.open(null, async (path) => {
          try {
            await Api.addSubset(path);
            onSelect(path);
          } catch (err) {
            toast(err.message, "error");
          }
        });
      },
    }, "+ Add subset"));
  }

  return { render };
})();

// ------------------------- Folder browser modal -------------------------
const FolderBrowser = (() => {
  let currentPath = null;
  let onChoose = null;

  const backdrop = () => document.getElementById("folder-modal");
  const pathEl = () => document.getElementById("folder-modal-path");
  const listEl = () => document.getElementById("folder-modal-list");
  const rootsEl = () => document.getElementById("folder-modal-roots");

  function renderRoots(roots) {
    const el = rootsEl();
    el.innerHTML = "";
    for (const r of roots || []) {
      const isHere = currentPath && r.path.replace(/[\\/]+$/, "").toLowerCase() === currentPath.replace(/[\\/]+$/, "").toLowerCase();
      el.appendChild(h("button", {
        type: "button",
        class: `modal-root-chip${isHere ? " active" : ""}`,
        onclick: () => refresh(r.path),
      }, r.label));
    }
  }

  async function refresh(path) {
    try {
      const data = await Api.browse(path);
      currentPath = data.path;
      pathEl().textContent = data.path;
      renderRoots(data.roots);
      const list = listEl();
      list.innerHTML = "";
      if (data.parent) {
        list.appendChild(rowEl("..", data.parent, null, () => refresh(data.parent)));
      }
      for (const d of data.dirs) {
        list.appendChild(rowEl(d.name, d.path, d.image_count, () => refresh(d.path)));
      }
      if (data.warning) {
        list.appendChild(h("div", { class: "empty-state" }, [h("p", {}, data.warning)]));
      } else if (data.dirs.length === 0) {
        list.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "No subfolders here. You can still choose this folder.")]));
      }
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function rowEl(name, path, imageCount, onClick) {
    const row = h("div", { class: "modal-dir-row", onclick: onClick }, [
      h("span", {}, "📁"),
      h("span", {}, name),
    ]);
    if (imageCount !== null && imageCount !== undefined) {
      row.appendChild(h("span", { class: "img-count" }, imageCount ? `${imageCount}${imageCount >= 4000 ? "+" : ""} images` : ""));
    }
    return row;
  }

  function open(startPath, callback) {
    onChoose = callback;
    currentPath = null;               // "Use this folder" must not fire on a stale path
    backdrop().classList.add("open");
    refresh(startPath || null);
  }

  function close() {
    backdrop().classList.remove("open");
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("folder-modal-cancel").addEventListener("click", close);
    document.getElementById("folder-modal-choose").addEventListener("click", () => {
      if (onChoose && currentPath) onChoose(currentPath);
      close();
    });
    document.getElementById("folder-modal-new").addEventListener("click", async () => {
      if (!currentPath) return toast("Still loading the folder list - try again in a moment.", "info");
      const name = prompt("New folder name:");
      if (!name) return;
      const newPath = currentPath.replace(/[\\/]+$/, "") + "/" + name;
      try {
        await Api.mkdir(newPath);
        refresh(newPath);
      } catch (err) {
        toast(err.message, "error");
      }
    });
    backdrop().addEventListener("click", (e) => {
      if (e.target === backdrop()) close();
    });
  });

  return { open, close };
})();

// ------------------------- File browser modal (pick a single file) -------------------------
const FileBrowser = (() => {
  let currentPath = null;
  let extFilter = null;
  let onChoose = null;

  const backdrop = () => document.getElementById("file-modal");
  const pathEl = () => document.getElementById("file-modal-path");
  const listEl = () => document.getElementById("file-modal-list");

  async function refresh(path) {
    try {
      const data = await Api.browseFiles(path, extFilter);
      currentPath = data.path;
      pathEl().textContent = data.path;
      const rootsBox = document.getElementById("file-modal-roots");
      if (rootsBox) {
        rootsBox.innerHTML = "";
        for (const r of data.roots || []) {
          rootsBox.appendChild(h("button", {
            type: "button", class: "modal-root-chip", onclick: () => refresh(r.path),
          }, r.label));
        }
      }
      const list = listEl();
      list.innerHTML = "";
      if (data.parent) {
        list.appendChild(dirRow("..", data.parent, () => refresh(data.parent)));
      }
      for (const d of data.dirs) {
        list.appendChild(dirRow(d.name, d.path, () => refresh(d.path)));
      }
      for (const f of (data.files || [])) {
        list.appendChild(fileRow(f.name, f.path));
      }
      if (data.warning) {
        list.appendChild(h("div", { class: "empty-state" }, [h("p", {}, data.warning)]));
      } else if (data.dirs.length === 0 && (data.files || []).length === 0) {
        list.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "Nothing here.")]));
      }
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function dirRow(name, path, onClick) {
    return h("div", { class: "modal-dir-row", onclick: onClick }, [h("span", {}, "📁"), h("span", {}, name)]);
  }

  function fileRow(name, path) {
    return h("div", { class: "modal-dir-row", onclick: () => { if (onChoose) onChoose(path); close(); } }, [
      h("span", {}, "📄"), h("span", {}, name),
    ]);
  }

  function open(startPath, ext, callback) {
    onChoose = callback;
    extFilter = ext || null;
    backdrop().classList.add("open");
    refresh(startPath || null);
  }

  function close() {
    backdrop().classList.remove("open");
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("file-modal-cancel").addEventListener("click", close);
    backdrop().addEventListener("click", (e) => {
      if (e.target === backdrop()) close();
    });
  });

  return { open, close };
})();

// ------------------------- Active dataset (shared across tabs) -------------------------
// Collect, Edit, and Caption all work on "a folder"; this remembers the
// most recent one so moving between tabs - and adding it to Train -
// doesn't mean re-browsing the filesystem every time. Persisted in
// localStorage so it survives a page reload too.
const ActiveDataset = (() => {
  let folder = "";
  try { folder = localStorage.getItem("animaStudio.activeDataset") || ""; } catch { /* storage blocked: session-only */ }
  const listeners = [];

  function set(path) {
    if (!path || path === folder) return;
    folder = path;
    try { localStorage.setItem("animaStudio.activeDataset", path); } catch { /* storage blocked: session-only */ }
    listeners.forEach((fn) => fn(folder));
  }

  function get() {
    return folder;
  }

  function onChange(fn) {
    listeners.push(fn);
  }

  return { set, get, onChange };
})();


function openLightbox(src, captionText, action) {
  document.getElementById("lightbox-img").src = src;
  const capEl = document.getElementById("lightbox-caption");
  if (captionText) {
    capEl.textContent = captionText;
    capEl.style.display = "block";
  } else {
    capEl.style.display = "none";
  }
  const actionBtn = document.getElementById("lightbox-action");
  if (action) {
    actionBtn.textContent = action.label;
    actionBtn.className = `btn primary${action.active ? "" : " ghost"}`;
    actionBtn.style.display = "inline-flex";
    actionBtn.onclick = (e) => {
      e.stopPropagation();
      action.onClick();
    };
  } else {
    actionBtn.style.display = "none";
    actionBtn.onclick = null;
  }
  document.getElementById("lightbox").classList.add("open");
}
document.addEventListener("DOMContentLoaded", () => {
  const lb = document.getElementById("lightbox");
  lb.addEventListener("click", (e) => {
    if (e.target.id !== "lightbox-img" && e.target.id !== "lightbox-action") lb.classList.remove("open");
  });
});

// ------------------------- Training ETA estimator -------------------------
// Estimates how long a run has left from the steps it reports, and keeps
// correcting itself as the run speeds up or slows down:
//
//  * the rate is a least-squares fit over a rolling window, so one slow step
//    cannot throw it off the way a plain "last two samples" average does;
//  * the window is short enough that a real change (a checkpoint pause, a
//    dataset switch, thermal throttling, another job sharing the GPU) is
//    picked up instead of being averaged away;
//  * the fitted rate and the ETA are both damped with an EWMA, so the number
//    settles instead of jumping on every poll;
//  * if steps stop arriving while the run is still alive, `stalled` is set so
//    the UI can say "saving a checkpoint?" rather than count down a lie.
//
// Feed it every poll: update(step, total) -> null | {seconds, rate, stalled}.
function createEtaEstimator(opts) {
  const cfg = { windowSeconds: 240, maxSamples: 60, damp: 0.3, rateDamp: 0.35, stallAfter: 45, ...(opts || {}) };
  const samples = [];
  let rate = null, seconds = null, stallSince = null;

  function reset() {
    samples.length = 0;
    rate = null; seconds = null; stallSince = null;
  }

  function update(step, total) {
    if (!Number.isFinite(step)) return null;
    const now = Date.now() / 1000;
    const last = samples[samples.length - 1];
    if (last && step < last.step) reset();                       // a new run started
    if (!samples.length || step !== samples[samples.length - 1].step) samples.push({ t: now, step });
    while (samples.length > 2 && now - samples[0].t > cfg.windowSeconds) samples.shift();
    if (samples.length > cfg.maxSamples) samples.splice(0, samples.length - cfg.maxSamples);

    const quietFor = samples.length ? now - samples[samples.length - 1].t : 0;
    stallSince = quietFor > cfg.stallAfter ? (stallSince ?? now - quietFor) : null;

    // Steps per second from a least-squares fit over the window.
    let fitted = 0;
    if (samples.length >= 2) {
      const n = samples.length;
      let mt = 0, ms = 0;
      for (const s of samples) { mt += s.t; ms += s.step; }
      mt /= n; ms /= n;
      let num = 0, den = 0;
      for (const s of samples) { num += (s.t - mt) * (s.step - ms); den += (s.t - mt) ** 2; }
      if (den > 1e-9) {
        fitted = num / den;
      } else {
        // every sample landed in the same instant (two polls inside one
        // millisecond): no slope to fit, so use the endpoints if they differ
        const first = samples[0], newest = samples[samples.length - 1];
        if (newest.t > first.t) fitted = (newest.step - first.step) / (newest.t - first.t);
      }
    }

    if (fitted > 0) {
      rate = rate == null ? fitted : rate + (fitted - rate) * cfg.rateDamp;
      if (Number.isFinite(total)) {
        const raw = Math.max(0, (total - step) / rate);
        seconds = seconds == null ? raw : seconds + (raw - seconds) * cfg.damp;
      }
    }

    if (stallSince != null) return { seconds, rate, stalled: true, stallSince };
    if (rate == null || seconds == null) return null;
    return { seconds, rate, stalled: false, stallSince: null };
  }

  return { update, reset };
}

// Compact "1h 05m" / "12m 30s" / "45s" duration for progress read-outs.
function fmtDuration(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec % 60}s` : `${sec}s`;
}
