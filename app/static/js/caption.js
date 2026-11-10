const Caption = (() => {
  let currentFolder = "";
  let images = [];
  let allTags = []; // full tag frequency list, cached for the Tag Frequency subview
  // tag -> { count, example, imagePaths } snapshot taken at the moment a
  // tag is excluded, so clicking it again restores it to exactly the
  // images it came from (not every image in the folder).
  const excludedTags = new Map();
  let tagFreqSort = "count";
  let tagFilter = "";                 // show only images carrying this tag
  const selected = new Set();         // image paths ticked in the gallery
  let lastPicked = -1;                // index (in the visible list) of the last tick, for shift-range
  let copiedTags = [];                // tags copied from one image, to paste onto the selection
  let visibleImages = [];

  // Splits on commas (this app's own convention) or newlines - some
  // datasets tagged elsewhere write one tag per line instead of
  // comma-separating them; matches the same splitting the backend's
  // tag_frequency() does so per-tag counts/filtering agree with the UI.
  function splitTags(text) {
    return (text || "").split(/[,\n]+/).map((t) => t.trim()).filter(Boolean);
  }

  const normTag = (t) => String(t || "").trim().toLowerCase().replace(/_/g, " ").replace(/\s+/g, " ");

  // Autocomplete and the Tag Frequency view both read this list, so keep it
  // fresh after per-image edits without reloading the whole folder on every key.
  const refreshTagList = debounce(async () => {
    if (!currentFolder) return;
    try {
      const [t, st] = await Promise.all([Api.tagFrequency(currentFolder), Api.captionStats(currentFolder)]);
      allTags = t.tags;
      renderStats(st);
      renderTagFrequency(allTags);
      renderTagFreqCloud();
    } catch { /* next edit retries */ }
  }, 1500);

  // One timer per image: editing image A and then B inside the debounce window
  // must save both (a single shared debounce would drop A's save).
  const saveTimers = new Map();
  function saveCaption(path, caption) {
    clearTimeout(saveTimers.get(path));
    saveTimers.set(path, setTimeout(async () => {
      saveTimers.delete(path);
      try {
        await Api.captionSave(path, caption);
        refreshTagList();
      } catch (err) {
        toast(err.message, "error");
      }
    }, 500));
  }

  function init() {
    const folderInput = document.getElementById("caption-folder");
    if (!folderInput.value && ActiveDataset.get()) folderInput.value = ActiveDataset.get();
    ActiveDataset.onChange((path) => {
      if (!folderInput.value) folderInput.value = path;
    });

    document.getElementById("caption-browse").addEventListener("click", () => {
      FolderBrowser.open(currentFolder || null, (path) => {
        document.getElementById("caption-folder").value = path;
        load();
      });
    });
    document.getElementById("caption-load").addEventListener("click", load);
    document.getElementById("caption-folder").addEventListener("keydown", (e) => {
      if (e.key === "Enter") load();
    });
    document.getElementById("caption-search").addEventListener("input", debounce(renderGrid, 150));

    SubsetTabs.render("caption-subset-tabs", currentFolder, selectSubsetTab);

    // autocomplete on every plain tag box
    for (const id of ["bulk-add-tag", "bulk-trigger-front", "bulk-remove-tag", "bulk-find", "sel-add-tag", "sel-remove-tag"]) {
      const el = document.getElementById(id);
      if (el) Autocomplete.attach(el, { getTags: () => allTags, onPick: (t) => { el.value = t; el.focus(); } });
    }
    document.querySelectorAll(".caption-undo-btn").forEach((b) => b.addEventListener("click", undoLast));
    document.getElementById("caption-tag-filter-clear").addEventListener("click", () => setTagFilter(""));
    document.getElementById("sel-all").addEventListener("click", () => {
      visibleImages.forEach((i) => selected.add(i.path));
      renderGrid();
    });
    document.getElementById("sel-clear").addEventListener("click", () => { selected.clear(); renderGrid(); });
    document.getElementById("sel-add-btn").addEventListener("click", () => editSelected("add"));
    document.getElementById("sel-remove-btn").addEventListener("click", () => editSelected("remove"));
    document.getElementById("sel-paste-btn").addEventListener("click", () => editSelected("paste"));

    document.getElementById("bulk-add-btn").addEventListener("click", async () => {
      const tag = document.getElementById("bulk-add-tag").value.trim();
      if (!tag) return;
      const r = await Api.bulkAddTag(currentFolder, tag, "end");
      toast(`Added "${tag}" to ${r.changed} images.`, "success");
      document.getElementById("bulk-add-tag").value = "";
      await refresh();
    });

    document.getElementById("bulk-trigger-front-btn").addEventListener("click", async () => {
      const tag = document.getElementById("bulk-trigger-front").value.trim();
      if (!tag) return;
      const r = await Api.bulkAddTag(currentFolder, tag, "start");
      toast(`Prepended "${tag}" to the front of ${r.changed} caption(s).`, "success");
      document.getElementById("bulk-trigger-front").value = "";
      await refresh();
    });

    document.getElementById("bulk-remove-btn").addEventListener("click", async () => {
      const tag = document.getElementById("bulk-remove-tag").value.trim();
      if (!tag) return;
      const r = await Api.bulkRemoveTag(currentFolder, tag);
      toast(`Removed "${tag}" from ${r.changed} images.`, "success");
      document.getElementById("bulk-remove-tag").value = "";
      await refresh();
    });

    document.getElementById("bulk-replace-btn").addEventListener("click", async () => {
      const find = document.getElementById("bulk-find").value.trim();
      const replace = document.getElementById("bulk-replace").value.trim();
      if (!find) return;
      const r = await Api.bulkFindReplace(currentFolder, find, replace, true);
      toast(`Updated ${r.changed} images.`, "success");
      document.getElementById("bulk-find").value = "";
      document.getElementById("bulk-replace").value = "";
      await refresh();
    });

    document.getElementById("bulk-underscores-btn").addEventListener("click", async () => {
      const r = await Api.bulkUnderscores(currentFolder);
      toast(`Cleaned up ${r.changed} captions.`, "success");
      await refresh();
    });

    document.getElementById("autotag-btn").addEventListener("click", runAutotag);
    document.getElementById("tagger-download-btn").addEventListener("click", startTaggerDownload);
    document.getElementById("tagger-model").addEventListener("change", () => {
      refreshTaggerStatus();
      updateTaggerDepsRow();
    });
    document.getElementById("tagger-deps-goto").addEventListener("click", () => openView("backend"));

    // Dependency installs live on the Backend tab now; re-check when the user
    // comes back so a warning here clears as soon as they are installed.
    document.addEventListener("studio:viewchange", (e) => {
      if (!e.detail || e.detail.view !== "caption") return;
      Api.taggerModels().then((data) => {
        taggerModelsData = data.models;
        updateTaggerDepsRow();
      }).catch(() => {});
      refreshNlDeps(false);
    });

    document.querySelectorAll("#caption-subview-toggle button").forEach((btn) => {
      btn.addEventListener("click", () => switchSubview(btn.dataset.subview));
    });

    // Booru tagging and natural-language captions are two ways to do the same
    // job, so only one card shows at a time; remember which one was left open.
    let savedTool = "";
    try { savedTool = localStorage.getItem("animaStudio.captionTool") || ""; } catch { /* storage unavailable - fine */ }
    setCaptionTool(savedTool === "nl" ? "nl" : "booru");
    document.querySelectorAll("#caption-tool-toggle button").forEach((btn) => {
      btn.addEventListener("click", () => setCaptionTool(btn.dataset.tool));
    });
    document.getElementById("tagfreq-search").addEventListener("input", debounce(renderTagFreqCloud, 150));
    document.querySelectorAll("#tagfreq-sort-toggle button").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#tagfreq-sort-toggle button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        tagFreqSort = btn.dataset.sort;
        renderTagFreqCloud();
      });
    });

    Api.taggerModels().then((data) => {
      taggerModelsData = data.models;
      const select = document.getElementById("tagger-model");
      select.innerHTML = "";
      data.models.forEach((m) => {
        select.appendChild(h("option", { value: m.name, selected: m.name === data.default ? "selected" : null }, m.name));
      });
      refreshTaggerStatus();
      updateTaggerDepsRow();
    }).catch(() => {});

    initNlCaptions();
  }

  // ------------------------------ WD14 model download ------------------------------

  let taggerPollTimer = null;
  let taggerModelsData = [];

  function updateTaggerDepsRow() {
    const model = document.getElementById("tagger-model").value;
    const info = taggerModelsData.find((m) => m.name === model);
    const row = document.getElementById("tagger-deps-row");
    if (!info || !info.needs_deps || info.deps_available) {
      row.style.display = "none";
      return;
    }
    row.style.display = "block";
  }

  function setTaggerReady(ready) {
    document.getElementById("autotag-btn").disabled = !ready;
    const badge = document.getElementById("tagger-status-badge");
    badge.textContent = ready ? "ready" : "not downloaded";
    badge.className = `badge ${ready ? "rating-safe" : "rating-questionable"}`;
    document.getElementById("tagger-download-btn").style.display = ready ? "none" : "inline-flex";
  }

  async function refreshTaggerStatus() {
    const model = document.getElementById("tagger-model").value;
    if (!model) return;
    try {
      const status = await Api.taggerDownloadStatus(model);
      applyTaggerStatus(status);
    } catch {
      setTaggerReady(false);
    }
  }

  function applyTaggerStatus(status) {
    const track = document.getElementById("tagger-progress-track");
    const fill = document.getElementById("tagger-progress-fill");
    const text = document.getElementById("tagger-progress-text");

    if (status.status === "downloading") {
      setTaggerReady(false);
      document.getElementById("tagger-download-btn").style.display = "none";
      track.style.display = "block";
      const pct = status.total ? Math.min(100, Math.round((status.bytes / status.total) * 100)) : 0;
      fill.style.width = `${pct}%`;
      text.textContent = status.total ? `${pct}% (${(status.bytes / 1e6).toFixed(0)} / ${(status.total / 1e6).toFixed(0)} MB)` : "starting…";
      document.getElementById("tagger-status-badge").textContent = "downloading";
      document.getElementById("tagger-status-badge").className = "badge rating-sensitive";
      if (!taggerPollTimer) taggerPollTimer = setInterval(refreshTaggerStatus, 800);
      return;
    }

    if (taggerPollTimer) { clearInterval(taggerPollTimer); taggerPollTimer = null; }
    track.style.display = "none";
    text.textContent = "";

    if (status.status === "error") {
      setTaggerReady(false);
      document.getElementById("tagger-status-badge").textContent = "download failed";
      document.getElementById("tagger-status-badge").className = "badge rating-explicit";
      toast(status.error || "Model download failed.", "error");
      return;
    }

    if (status.status === "done") {
      setTaggerReady(true);
      return;
    }

    setTaggerReady(false);
  }

  async function startTaggerDownload() {
    const model = document.getElementById("tagger-model").value;
    try {
      const status = await Api.taggerDownload(model);
      applyTaggerStatus(status);
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // ------------------------------ natural-language captions ------------------------------
  // Two sources behind one button: rephrasing the tags already on disk (instant,
  // offline), or a local vision-language model describing the picture. The model
  // runs in its own process, so the batch reports progress by polling rather than
  // by holding the request open.

  let nlModelsData = [];
  let nlPollTimer = null;
  let nlDlPollTimer = null;
  let nlModeVlm = "missing_only";       // the VLM's own choice, kept while the tags source owns the select

  // Why a whole batch ended up skipped, in words that say what to do about it.
  const NL_SKIP_NOTES = {
    "no tags to rephrase": "these images have no tags yet - tag them first with the Auto-tag card",
    "nothing to rephrase": "nothing left to say after dropping library tags",
  };

  const nlEl = (id) => document.getElementById(id);
  const nlModel = () => nlEl("nl-model").value;
  const nlIsVlm = () => nlEl("nl-source").value === "vlm";

  const NL_MODE_WORDS = { "4bit": "4-bit", "8bit": "8-bit", "bf16": "full bf16" };

  // What this model will do on the chosen VRAM profile ("auto" reads the card at run time).
  function nlFitText(info) {
    const prof = nlEl("nl-vram").value;
    const gb = prof === "auto" ? null : prof;
    if (!gb) return `Needs about ${info.vram["4bit"]} GB in 4-bit, ${info.vram["8bit"]} GB in 8-bit, ${info.vram.bf16} GB in bf16.`;
    const mode = info.fits[gb];
    return `On ${gb} GB it loads ${NL_MODE_WORDS[mode] || mode} (about ${info.vram[mode]} GB).`;
  }

  function updateNlRows() {
    const vlm = nlIsVlm();
    const mode = nlEl("nl-mode");
    if (vlm) {
      mode.disabled = false;
      mode.value = nlModeVlm;
    } else {
      // rephrasing works on the tags already on disk, so an image without a
      // caption can never be rephrased - the choice only exists for the VLM
      mode.value = "all";
      mode.disabled = true;
    }
    nlEl("nl-vlm-row").style.display = vlm ? "flex" : "none";
    nlEl("nl-model").disabled = !vlm;
    nlEl("nl-vram-field").style.display = vlm ? "" : "none";
    nlEl("nl-preview-btn").style.display = vlm ? "none" : "inline-flex";
    nlEl("nl-once-btn").style.display = vlm ? "inline-flex" : "none";
    if (!vlm) {
      nlEl("nl-deps-row").style.display = "none";
      nlEl("nl-dl-track").style.display = "none";
      nlEl("nl-dl-text").textContent = "";
      return;
    }
    const info = nlModelsData.find((m) => m.name === nlModel());
    nlEl("nl-download-btn").style.display = info && info.downloaded ? "none" : "inline-flex";
    nlEl("nl-model-note").textContent = info
      ? `${info.downloaded ? "Downloaded. " : `Download ~${info.size_gb} GB. `}${info.note} ${nlFitText(info)}`
      : "";
    refreshNlDeps(false);
  }

  async function refreshNlDeps(force) {
    if (!nlIsVlm()) return;
    const row = nlEl("nl-deps-row");
    const text = nlEl("nl-deps-text");
    try {
      const deps = await Api.nlDeps(force, nlModel());
      const own = deps.env === "caption5" || (nlModelsData.find((m) => m.name === nlModel()) || {}).env === "caption5";
      nlEl("nl-env-setup").style.display = own ? "" : "none";
      nlEl("nl-deps-goto").style.display = own ? "none" : "";
      if (deps.ok) {                        // environment ready - nothing to warn about here
        row.style.display = "none";
        return;
      }
      const bits = [];
      if (deps.missing && deps.missing.length) bits.push(`Needs ${deps.missing.join(", ")}`);
      if (deps.error) bits.push(deps.error);
      if (!bits.length) bits.push("The caption environment is not ready");
      text.textContent = own
        ? `${bits.join(" · ")} - this model runs in its own environment (it keeps transformers 5.x away from training). One-time download, a few GB.`
        : `${bits.join(" · ")} - install once from the Backend tab.`;
      row.style.display = "block";
    } catch (err) {
      text.textContent = err.message || "could not check the environment";
      row.style.display = "block";
    }
  }

  let nlEnvTimer = null;
  function pollNlEnv() {
    clearInterval(nlEnvTimer);
    const log = nlEl("nl-env-log"), btn = nlEl("nl-env-setup");
    const tick = async () => {
      try {
        const st = await Api.nlEnvStatus();
        log.style.display = st.stage === "idle" ? "none" : "block";
        log.textContent = (st.log_tail || []).slice(-14).join("\n");
        log.scrollTop = log.scrollHeight;
        btn.disabled = st.stage === "installing";
        if (st.stage === "installing") {
          const step = (st.steps || [])[st.step_index];
          nlEl("nl-deps-text").textContent = `Setting up… ${step ? step.label : ""} (${(st.step_index || 0) + 1}/${(st.steps || []).length}). This can take several minutes.`;
          return;
        }
        clearInterval(nlEnvTimer);
        if (st.stage === "error") nlEl("nl-deps-text").textContent = st.error;
        if (st.stage === "done") { toast("Qwen3.5 environment ready", "success"); Api.nlModels().then((d) => { nlModelsData = d.models; updateNlRows(); }).catch(() => {}); refreshNlDeps(true); }
      } catch { /* keep polling */ }
    };
    nlEnvTimer = setInterval(tick, 1500); tick();
  }

  async function startNlDownload() {
    const btn = nlEl("nl-download-btn");
    btn.disabled = true;
    try {
      applyNlDownload(await Api.nlDownload(nlModel()));
      if (!nlDlPollTimer) nlDlPollTimer = setInterval(pollNlDownload, 800);
    } catch (err) {
      toast(err.message, "error");
      btn.disabled = false;
    }
  }

  async function pollNlDownload() {
    try {
      applyNlDownload(await Api.nlDownloadStatus(nlModel()));
    } catch { /* keep polling */ }
  }

  function applyNlDownload(st) {
    const track = nlEl("nl-dl-track");
    const fill = nlEl("nl-dl-fill");
    const text = nlEl("nl-dl-text");
    const btn = nlEl("nl-download-btn");
    const gb = (bytes) => (bytes / 1024 ** 3).toFixed(2);
    if (st.status === "downloading") {
      track.style.display = "block";
      fill.classList.toggle("indeterminate", !st.total);
      fill.style.width = `${st.total ? Math.min(100, (st.bytes / st.total) * 100) : 0}%`;
      text.textContent = st.total
        ? `${Math.round((st.bytes / st.total) * 100)}% · ${gb(st.bytes)} / ${gb(st.total)} GB`
        : `downloading… ${gb(st.bytes)} GB so far`;
      btn.disabled = true;
      return;
    }
    track.style.display = "none";
    btn.disabled = false;
    if (nlDlPollTimer) { clearInterval(nlDlPollTimer); nlDlPollTimer = null; }
    if (st.status === "error") {
      text.textContent = st.error || "download failed";
      toast(st.error || "Download failed", "error");
      return;
    }
    if (st.status === "done") {
      text.textContent = "Downloaded - this model can caption now.";
      const info = nlModelsData.find((m) => m.name === nlModel());
      if (info) info.downloaded = true;
      updateNlRows();
      toast("Caption model downloaded.", "success");
    }
  }

  function applyNlRun(st) {
    const track = nlEl("nl-run-track");
    const fill = nlEl("nl-run-fill");
    const text = nlEl("nl-run-text");
    if (st.status === "running") {
      track.style.display = "block";
      fill.classList.remove("indeterminate");
      const pct = st.total ? Math.min(100, ((st.done || 0) / st.total) * 100) : 0;
      fill.style.width = `${pct}%`;
      nlEl("nl-cancel-btn").style.display = "inline-flex";
      text.textContent = `${st.done || 0} / ${st.total || "?"} images`
        + (st.current ? ` · ${st.current}` : "")
        + (st.device ? ` · model on ${st.device}${st.quant ? ` (${st.quant})` : ""}` : "");
      return;
    }
    if (st.status === "idle") return;
    track.style.display = "none";
    resetNlRun();
    if (st.status === "error") {
      text.textContent = st.error || "captioning failed";
      toast(st.error || "Captioning failed", "error");
      return;
    }
    const failed = st.failed || 0;
    let note = "";
    if (!st.captioned && !failed && st.skipped && st.status === "done") {
      const seen = new Set();
      for (const r of st.results || []) if (r.reason) seen.add(r.reason);
      if (seen.size) note = [...seen].map((r) => NL_SKIP_NOTES[r] || r).join("; ");
      else if (nlEl("nl-mode").value === "missing_only") note = "every image already has a caption";
    }
    text.textContent = `${st.captioned || 0} captioned`
      + (st.skipped ? `, ${st.skipped} skipped` : "")
      + (failed ? `, ${failed} failed` : "")
      + (st.status === "cancelled" ? " · stopped" : "")
      + (note ? ` · ${note}` : "");
    if (st.status === "cancelled") toast("Stopped.", "info");
    else if (note) toast(`Nothing to caption - ${note}.`, "info");
    else toast(`Captioning finished: ${st.captioned || 0} images written.`, failed ? "info" : "success");
    if ((st.captioned || 0) > 0) refresh();
  }

  function resetNlRun() {
    if (nlPollTimer) { clearInterval(nlPollTimer); nlPollTimer = null; }
    nlEl("nl-run-btn").disabled = false;
    nlEl("nl-spinner").style.display = "none";
    nlEl("nl-cancel-btn").style.display = "none";
  }

  async function runNlCaptions(limit) {
    if (!currentFolder) return toast("Load a dataset folder first.", "error");
    nlEl("nl-run-btn").disabled = true;
    nlEl("nl-spinner").style.display = "inline-block";
    try {
      applyNlRun(await Api.nlRun({
        folder: currentFolder,
        source: nlEl("nl-source").value,
        model: nlModel(),
        vram_profile: nlEl("nl-vram").value,
        mode: nlEl("nl-mode").value,
        write: nlEl("nl-write").value,
        trigger_word: nlEl("nl-trigger").value.trim(),
        drop_quality: nlEl("nl-drop-quality").checked,
        limit: limit || 0,
      }));
      if (!nlPollTimer) nlPollTimer = setInterval(pollNlRun, 700);
    } catch (err) {
      toast(err.message, "error");
      resetNlRun();
    }
  }

  async function pollNlRun() {
    try {
      applyNlRun(await Api.nlRunStatus());
    } catch { /* keep polling */ }
  }

  async function cancelNlCaptions() {
    try {
      applyNlRun(await Api.nlCancel());
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function previewNlCaption() {
    const img = (images || []).find((i) => (i.caption || "").trim());
    if (!img) return toast("No captioned image here to rephrase yet.", "error");
    try {
      const r = await Api.nlPreview({
        caption: img.caption,
        write: nlEl("nl-write").value,
        trigger_word: nlEl("nl-trigger").value.trim(),
        drop_quality: nlEl("nl-drop-quality").checked,
      });
      nlEl("nl-preview-box").style.display = "block";
      nlEl("nl-preview-before").textContent = r.before;
      nlEl("nl-preview-after").textContent = r.after;
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function initNlCaptions() {
    nlEl("nl-source").addEventListener("change", updateNlRows);
    nlEl("nl-model").addEventListener("change", updateNlRows);
    nlEl("nl-vram").addEventListener("change", updateNlRows);
    nlEl("nl-mode").addEventListener("change", () => {
      if (nlIsVlm()) nlModeVlm = nlEl("nl-mode").value;    // remember it for the next VLM run
    });
    nlEl("nl-download-btn").addEventListener("click", startNlDownload);
    nlEl("nl-deps-goto").addEventListener("click", () => openView("backend"));
    nlEl("nl-env-setup").addEventListener("click", async () => {
      try { await Api.nlEnvStart(); pollNlEnv(); } catch (err) { toast(err.message, "error"); }
    });
    nlEl("nl-run-btn").addEventListener("click", () => runNlCaptions(0));
    nlEl("nl-once-btn").addEventListener("click", () => runNlCaptions(1));
    nlEl("nl-cancel-btn").addEventListener("click", cancelNlCaptions);
    nlEl("nl-preview-btn").addEventListener("click", previewNlCaption);
    Api.nlModels().then((data) => {
      nlModelsData = data.models;
      const select = nlEl("nl-model");
      const TIER = { low: "8 GB", mid: "8-12 GB", high: "16 GB+" };
      select.replaceChildren(...data.models.map((m) => h("option", {
        value: m.name, title: m.note, selected: m.name === data.default ? "selected" : null,
      }, `${m.label} · ${TIER[m.tier] || ""}`)));
      updateNlRows();
    }).catch(() => {});
  }

  function selectSubsetTab(path) {
    document.getElementById("caption-folder").value = path;
    return load();
  }

  async function load() {
    const newFolder = document.getElementById("caption-folder").value.trim();
    if (!newFolder) return toast("Enter or browse to a folder first.", "error");
    if (newFolder !== currentFolder) { excludedTags.clear(); selected.clear(); tagFilter = ""; lastPicked = -1; }
    currentFolder = newFolder;
    await refresh();
    ActiveDataset.set(currentFolder);
    SubsetTabs.render("caption-subset-tabs", currentFolder, selectSubsetTab);
  }

  async function refresh() {
    try {
      const [imgData, statsData, tagData] = await Promise.all([
        Api.captionImages(currentFolder),
        Api.captionStats(currentFolder),
        Api.tagFrequency(currentFolder),
      ]);
      images = imgData.images;
      allTags = tagData.tags;
      document.getElementById("caption-empty").style.display = "none";
      document.getElementById("caption-content").style.display = "block";
      renderStats(statsData);
      renderGrid();
      renderTagFrequency(allTags);
      renderTagFreqCloud();
      refreshUndo();
      await showNestedHintIfEmpty();
    } catch (err) {
      // Blank the gallery rather than keeping the previous folder's images
      // on screen under the new path - the bulk actions all target
      // currentFolder, so stale cards would edit the wrong dataset.
      images = [];
      visibleImages = [];
      allTags = [];
      selected.clear();
      renderStats({ total: 0, captioned: 0, uncaptioned: 0, unique_tags: 0 });
      renderGrid();
      renderTagFrequency([]);
      renderTagFreqCloud();
      toast(err.message, "error");
    }
  }

  async function showNestedHintIfEmpty() {
    const hint = document.getElementById("caption-nested-hint");
    if (!hint) return;
    if (images.length > 0) { hint.style.display = "none"; return; }
    try {
      const data = await Api.browse(currentFolder);
      const count = (data.dirs || []).length;
      if (count > 0) {
        hint.textContent = `This folder has no images directly, but ${count} subfolder${count === 1 ? "" : "s"} below might - pick one from the sidebar.`;
        hint.style.display = "block";
      } else {
        hint.style.display = "none";
      }
    } catch {
      hint.style.display = "none";
    }
  }

  function renderStats(stats) {
    document.getElementById("stat-total").textContent = stats.total;
    document.getElementById("stat-captioned").textContent = stats.captioned;
    document.getElementById("stat-uncaptioned").textContent = stats.uncaptioned;
    document.getElementById("stat-unique").textContent = stats.unique_tags;
  }

  function renderGrid() {
    const grid = document.getElementById("caption-grid");
    grid.innerHTML = "";
    const query = (document.getElementById("caption-search").value || "").trim().toLowerCase();
    let visible = images;
    if (tagFilter) visible = visible.filter((img) => splitTags(img.caption).some((t) => normTag(t) === tagFilter));
    if (query) visible = visible.filter((img) => img.filename.toLowerCase().includes(query) || img.caption.toLowerCase().includes(query));
    visibleImages = visible;
    // forget selections that are no longer in the folder
    const live = new Set(images.map((i) => i.path));
    for (const p of [...selected]) if (!live.has(p)) selected.delete(p);

    document.getElementById("caption-grid-count").textContent =
      visible.length !== images.length ? `${visible.length} of ${images.length}` : `${images.length} images`;
    const banner = document.getElementById("caption-tag-filter");
    banner.classList.toggle("open", !!tagFilter);
    document.getElementById("caption-tag-filter-name").textContent = tagFilter;
    grid.classList.toggle("has-selection", selected.size > 0);
    renderSelBar();

    visible.forEach((img, idx) => grid.appendChild(buildCard(img, idx)));
    if (visible.length === 0) {
      grid.appendChild(h("div", { class: "empty-state" }, [h("p", {}, tagFilter && !query ? "No image has that tag any more." : "No images match that search.")]));
    }
  }

  function renderSelBar() {
    const bar = document.getElementById("caption-selbar");
    bar.classList.toggle("open", selected.size > 0);
    document.getElementById("sel-count").textContent = `${selected.size} selected`;
    const paste = document.getElementById("sel-paste-btn");
    paste.disabled = copiedTags.length === 0;
    paste.textContent = copiedTags.length ? `Paste ${copiedTags.length} copied tag${copiedTags.length === 1 ? "" : "s"}` : "Paste copied tags";
  }

  function toggleSelect(img, idx, shift) {
    // lastPicked can point past the end after a search/filter shrinks the
    // visible list - fall back to a plain toggle instead of a crash.
    if (shift && lastPicked >= 0 && lastPicked < visibleImages.length) {
      const [a, b] = [Math.min(lastPicked, idx), Math.max(lastPicked, idx)];
      for (let i = a; i <= b; i++) selected.add(visibleImages[i].path);
    } else if (selected.has(img.path)) {
      selected.delete(img.path);
    } else {
      selected.add(img.path);
    }
    lastPicked = idx;
    renderGrid();
  }

  async function editSelected(kind) {
    const paths = [...selected];
    if (!paths.length) return;
    let add = [], remove = [];
    if (kind === "add") {
      const el = document.getElementById("sel-add-tag");
      add = splitTags(el.value);
      if (!add.length) return toast("Type a tag to add first.", "error");
      el.value = "";
    } else if (kind === "remove") {
      const el = document.getElementById("sel-remove-tag");
      remove = splitTags(el.value);
      if (!remove.length) return toast("Type a tag to remove first.", "error");
      el.value = "";
    } else {
      add = copiedTags;
    }
    try {
      const r = await Api.captionEditImages(currentFolder, paths, add, remove);
      toast(`Updated ${r.changed} of ${paths.length} selected image${paths.length === 1 ? "" : "s"}.`, "success");
      await refresh();
    } catch (err) { toast(err.message, "error"); }
  }

  // The chip editor for one image (type + Enter to add, drag to reorder, x to
  // remove, Ctrl+Z to undo, Aa for plain text). Used by the gallery card and by
  // the zoomed viewer, so both behave the same.
  function editorParts(img) {
    let tags = splitTags(img.caption);
    const hist = [];                         // earlier versions of this caption, newest last
    let rawMode = false;
    const countEl = h("span", { class: "tag-count" });
    const undoBtn = h("button", { class: "mini-btn", type: "button", title: "Undo the last change to this caption (Ctrl+Z)", disabled: true }, "↶");
    const editor = h("div", { class: "chip-editor" });
    const input = h("input", { type: "text", class: "chip-input", placeholder: "add tag…" });
    const raw = h("textarea", { class: "raw-caption", style: "display:none", placeholder: "no caption yet" });

    function sync(push) {
      if (push) {
        const prev = splitTags(img.caption).join(", ");
        if (hist[hist.length - 1] !== prev) hist.push(prev);
        if (hist.length > 50) hist.shift();
      }
      img.caption = tags.join(", ");
      countEl.textContent = `${tags.length} tags`;
      undoBtn.disabled = hist.length === 0;
      saveCaption(img.path, img.caption);
    }
    function change(newTags) {
      tags = newTags;
      sync(true);
      paint();
    }
    function addTags(list) {
      const have = new Set(tags.map(normTag));
      const fresh = [];
      for (const t of list) {
        const n = normTag(t);
        if (n && !have.has(n)) { have.add(n); fresh.push(t.trim()); }
      }
      if (fresh.length) change(tags.concat(fresh));
    }
    function undo() {
      if (!hist.length) return;
      tags = splitTags(hist.pop());
      sync(false);
      paint();
    }

    let dragFrom = -1;
    function paint() {
      editor.innerHTML = "";
      tags.forEach((t, i) => {
        const chip = h("span", { class: `chip${i === 0 ? " first" : ""}`, draggable: "true", title: i === 0 ? "First tag (drag to reorder)" : "Drag to reorder" }, [
          h("span", {}, t),
          h("button", { class: "x", type: "button", title: "Remove", onclick: (e) => { e.stopPropagation(); change(tags.filter((_, j) => j !== i)); } }, "×"),
        ]);
        chip.addEventListener("dragstart", (e) => {
          dragFrom = i;
          chip.classList.add("dragging");
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("text/plain", t);
        });
        chip.addEventListener("dragend", () => { dragFrom = -1; paint(); });
        chip.addEventListener("dragover", (e) => {
          if (dragFrom < 0) return;
          e.preventDefault();
          const r = chip.getBoundingClientRect();
          const before = e.clientX < r.left + r.width / 2;
          chip.classList.toggle("drop-before", before);
          chip.classList.toggle("drop-after", !before);
        });
        chip.addEventListener("dragleave", () => chip.classList.remove("drop-before", "drop-after"));
        chip.addEventListener("drop", (e) => {
          if (dragFrom < 0) return;
          e.preventDefault();
          const r = chip.getBoundingClientRect();
          let to = e.clientX < r.left + r.width / 2 ? i : i + 1;
          const moved = tags[dragFrom];
          const next = tags.filter((_, j) => j !== dragFrom);
          if (dragFrom < to) to -= 1;
          next.splice(to, 0, moved);
          dragFrom = -1;
          if (next.join("\n") !== tags.join("\n")) change(next); else paint();
        });
        editor.appendChild(chip);
      });
      editor.appendChild(input);
      countEl.textContent = `${tags.length} tags`;
    }

    Autocomplete.attach(input, { getTags: () => allTags, onPick: (t) => { input.value = ""; addTags([t]); input.focus(); } });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === ",") {
        e.preventDefault();
        const parts = splitTags(input.value);
        input.value = "";
        addTags(parts);
        input.focus();
      } else if (e.key === "Backspace" && !input.value && tags.length) {
        change(tags.slice(0, -1));
        input.focus();
      }
    });
    input.addEventListener("paste", (e) => {
      const text = (e.clipboardData || window.clipboardData).getData("text");
      if (/[,\n]/.test(text)) {
        e.preventDefault();
        addTags(splitTags(text));
      }
    });
    input.addEventListener("blur", () => {
      const parts = splitTags(input.value);
      if (parts.length) { input.value = ""; addTags(parts); }
    });
    editor.addEventListener("click", (e) => { if (e.target === editor) input.focus(); });
    editor.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
    });
    undoBtn.addEventListener("click", undo);

    raw.addEventListener("input", () => {
      tags = splitTags(raw.value);
      sync(false);
    });
    raw.addEventListener("focus", () => { raw.dataset.before = splitTags(img.caption).join(", "); });
    raw.addEventListener("blur", () => {
      const before = raw.dataset.before;
      if (before !== undefined && before !== splitTags(img.caption).join(", ") && hist[hist.length - 1] !== before) {
        hist.push(before);
        undoBtn.disabled = false;
      }
    });
    const modeBtn = h("button", { class: "mini-btn", type: "button", title: "Edit the caption as plain text" }, "Aa");
    modeBtn.addEventListener("click", () => {
      rawMode = !rawMode;
      editor.style.display = rawMode ? "none" : "";
      raw.style.display = rawMode ? "" : "none";
      modeBtn.textContent = rawMode ? "Chips" : "Aa";
      if (rawMode) { raw.value = tags.join(", "); raw.focus(); } else paint();
    });
    const copyBtn = h("button", { class: "mini-btn", type: "button", title: "Copy this image's tags (then paste them onto selected images)" }, "⧉");
    copyBtn.addEventListener("click", () => {
      copiedTags = [...tags];
      renderSelBar();
      toast(`Copied ${tags.length} tag${tags.length === 1 ? "" : "s"}. Select images, then press Paste.`, "info");
    });

    paint();
    return { editor, raw, countEl, undoBtn, copyBtn, modeBtn, addTags };
  }

  // One caption card: thumbnail with a select dot, plus the chip editor.
  function buildCard(img, idx) {
    const { editor, raw, countEl, undoBtn, copyBtn, modeBtn } = editorParts(img);
    const isSel = selected.has(img.path);
    const dot = h("div", { class: "sel-dot", title: "Select (shift-click for a range)", role: "button" }, isSel ? "✓" : "");
    dot.addEventListener("click", (e) => { e.stopPropagation(); toggleSelect(img, idx, e.shiftKey); });

    return h("div", { class: `caption-card${isSel ? " selected" : ""}` }, [
      h("div", { class: "thumb-wrap" }, [
        dot,
        h("img", { src: Api.thumbUrl(img.path), loading: "lazy", decoding: "async", title: "Click to zoom", onclick: () => openViewer(idx) }),
      ]),
      h("div", { class: "body" }, [
        h("div", { class: "filename" }, img.filename),
        editor,
        raw,
        h("div", { class: "card-foot" }, [countEl, h("span", { class: "spacer" }), undoBtn, copyBtn, modeBtn]),
      ]),
    ]);
  }

  // ------------------------------ zoomed viewer ------------------------------
  // Big image on the left, the same editable chips on the right. Left/Right
  // arrows (or the buttons) step through the images that are currently shown.
  let viewerIdx = -1;
  let viewerEl = null;

  function openViewer(idx) {
    if (!visibleImages.length) return;
    viewerIdx = Math.max(0, Math.min(idx, visibleImages.length - 1));
    if (!viewerEl) {
      viewerEl = h("div", { class: "cap-viewer", role: "dialog" });
      viewerEl.addEventListener("click", (e) => { if (e.target.closest(".region-mode")) return; if (e.target === viewerEl || e.target.classList.contains("cap-v-stage")) closeViewer(); });
      document.body.appendChild(viewerEl);
      document.addEventListener("keydown", onViewerKey, true);
    }
    document.body.style.overflow = "hidden";
    paintViewer();
  }

  function closeViewer() {
    if (!viewerEl) return;
    // Let a half-typed tag commit before the box disappears (removing the
    // focused input from the DOM never fires its blur handler).
    if (document.activeElement && viewerEl.contains(document.activeElement)) document.activeElement.blur();
    Autocomplete.close();
    document.removeEventListener("keydown", onViewerKey, true);
    viewerEl.remove();
    viewerEl = null;
    document.body.style.overflow = "";
    renderGrid();          // the cards behind pick up whatever was edited
  }

  function stepViewer(d) {
    const n = viewerIdx + d;
    if (n < 0 || n >= visibleImages.length) return;
    // let the chip box commit a half-typed tag before we leave this image
    if (document.activeElement && viewerEl && viewerEl.contains(document.activeElement)) document.activeElement.blur();
    viewerIdx = n;
    paintViewer();
  }

  function onViewerKey(e) {
    if (!viewerEl) return;
    const t = e.target;
    const typing = t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.value !== ""));
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeViewer(); }
    else if (e.key === "ArrowLeft" && !typing) { e.preventDefault(); stepViewer(-1); }
    else if (e.key === "ArrowRight" && !typing) { e.preventDefault(); stepViewer(1); }
    else if (e.key === "PageUp") { e.preventDefault(); stepViewer(-1); }
    else if (e.key === "PageDown") { e.preventDefault(); stepViewer(1); }
  }

  function paintViewer() {
    const img = visibleImages[viewerIdx];
    if (!img) return closeViewer();
    const { editor, raw, countEl, undoBtn, copyBtn, modeBtn, addTags } = editorParts(img);
    viewerEl.innerHTML = "";
    const prev = h("button", { class: "cap-v-nav prev", type: "button", title: "Previous (←)", onclick: (e) => { e.stopPropagation(); stepViewer(-1); } }, "‹");
    const next = h("button", { class: "cap-v-nav next", type: "button", title: "Next (→)", onclick: (e) => { e.stopPropagation(); stepViewer(1); } }, "›");
    prev.style.visibility = viewerIdx > 0 ? "visible" : "hidden";
    next.style.visibility = viewerIdx < visibleImages.length - 1 ? "visible" : "hidden";
    viewerEl.append(
      h("div", { class: "cap-v-stage" }, [
        prev,
        h("img", { src: Api.imageUrl(img.path), alt: img.filename, draggable: "false" }),
        next,
      ]),
      h("div", { class: "cap-v-side" }, [
        h("div", { class: "cap-v-head" }, [
          h("div", {}, [
            h("div", { class: "cap-v-name" }, img.filename),
            h("div", { class: "cap-v-count" }, `${viewerIdx + 1} of ${visibleImages.length}`),
          ]),
          h("button", { class: "btn small ghost", type: "button", title: "Close (Esc)", onclick: closeViewer }, "✕"),
        ]),
        editor,
        raw,
        h("div", { class: "card-foot" }, [countEl, h("span", { class: "spacer" }), undoBtn, copyBtn, modeBtn]),
        regionPanel(img, addTags),
        h("p", { class: "card-hint", style: "margin-top:10px" }, "← → switch image · drag chips to reorder · Ctrl+Z undo · Esc close"),
      ]),
    );
    // warm the cache for the neighbours so stepping feels instant
    [visibleImages[viewerIdx + 1], visibleImages[viewerIdx - 1]].forEach((n) => { if (n) new Image().src = Api.imageUrl(n.path); });
  }

  // ------------------------------ tag a selected region ------------------------------
  // For details the tagger misses on the whole image: draw a box, the tagger scores just
  // that crop, and the closest tags are offered as chips to add.
  function regionPanel(img, addTags) {
    const box = h("div", { class: "region-box" });
    const results = h("div", { class: "region-results" });
    const btn = h("button", { class: "btn small", type: "button", title: "Draw a box around a detail and get tag suggestions for just that part" }, "▢ Tag a region");
    const status = h("div", { class: "card-hint", style: "margin:6px 0" });
    let on = false;
    let lastBox = null;
    const search = h("input", { type: "text", class: "region-search", placeholder: "Search all tags for this region, e.g. plate, crotch…", style: "display:none;width:100%;margin:6px 0" });
    let searchTimer = null;
    search.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => lastBox && run(lastBox, true), 300); });
    search.addEventListener("keydown", (e) => e.stopPropagation());

    const stage = () => viewerEl && viewerEl.querySelector(".cap-v-stage");
    const imgEl = () => stage() && stage().querySelector("img");
    // where the picture really is inside the (object-fit: contain) element
    function shown() {
      const el = imgEl(), st = stage();
      const r = el.getBoundingClientRect(), sr = st.getBoundingClientRect();
      const k = Math.min(r.width / el.naturalWidth, r.height / el.naturalHeight);
      const w = el.naturalWidth * k, hh = el.naturalHeight * k;
      return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - hh) / 2, w, h: hh, sl: sr.left, st: sr.top };
    }
    function setOn(v) {
      on = v;
      btn.classList.toggle("primary", on);
      btn.textContent = on ? "▢ Drag on the image…" : "▢ Tag a region";
      const st = stage();
      if (!st) return;
      st.classList.toggle("region-mode", on);
      if (!on) box.remove();
    }
    btn.addEventListener("click", () => { setOn(!on); status.textContent = on ? "Drag a box around the detail on the image." : ""; });

    async function run(frac, fromSearch) {
      lastBox = frac;
      search.style.display = "";
      status.textContent = "Reading that region…";
      if (!fromSearch) search.value = "";
      results.replaceChildren();
      try {
        const r = await Api.captionRegionTags({
          image_path: img.path, box: frac, model: document.getElementById("tagger-model").value, top: 40, query: search.value.trim(),
        });
        const have = new Set(splitTags(img.caption).map(normTag));
        status.textContent = r.tags.length
          ? "Closest tags for this region - click to add. Bar = score on the region; ★ = the whole image missed it."
          : "No tag in this model's vocabulary matches that search.";
        results.replaceChildren(...r.tags.map((t) => {
          const owned = have.has(normTag(t.tag));
          const chip = h("button", { type: "button", class: `region-tag${owned ? " owned" : ""}`, title: `region ${Math.round(t.score * 100)}% · whole image ${Math.round(t.whole * 100)}%` }, [
            h("span", { class: "bar", style: `width:${Math.min(100, Math.round(t.score * 100))}%` }),
            h("span", { class: "lbl" }, `${t.lift >= 0.15 && t.whole < 0.35 ? "★ " : ""}${t.tag}`),
            h("span", { class: "pct" }, `${t.score < 0.01 ? (t.score * 100).toFixed(1) : Math.round(t.score * 100)}%`),
          ]);
          chip.addEventListener("click", () => {
            addTags([t.tag]);
            have.add(normTag(t.tag));
            chip.classList.add("owned");
          });
          return chip;
        }));
      } catch (err) { status.textContent = ""; toast(err.message, "error"); }
    }

    // drawing happens on the stage; attach once it exists
    const panel = h("div", { class: "region-panel" }, [btn, status, search, results]);
    queueMicrotask(() => {
      const st = stage();
      if (!st || !panel.isConnected) return;   // a repaint may have replaced this panel
      st.addEventListener("pointerdown", (e) => {
        if (!on || e.target.closest(".cap-v-nav")) return;
        const s = shown();
        const x0 = Math.max(s.left, Math.min(s.left + s.w, e.clientX)), y0 = Math.max(s.top, Math.min(s.top + s.h, e.clientY));
        let x1 = x0, y1 = y0;
        e.preventDefault();
        st.setPointerCapture(e.pointerId);
        st.appendChild(box);
        const draw = () => {
          Object.assign(box.style, { left: `${Math.min(x0, x1) - s.sl}px`, top: `${Math.min(y0, y1) - s.st}px`,
            width: `${Math.abs(x1 - x0)}px`, height: `${Math.abs(y1 - y0)}px` });
        };
        draw();
        const move = (ev) => {
          x1 = Math.max(s.left, Math.min(s.left + s.w, ev.clientX));
          y1 = Math.max(s.top, Math.min(s.top + s.h, ev.clientY));
          draw();
        };
        const up = () => {
          st.removeEventListener("pointermove", move);
          st.removeEventListener("pointerup", up);
          const w = Math.abs(x1 - x0), hh = Math.abs(y1 - y0);
          if (w < 10 || hh < 10) { box.remove(); return; }
          run({ x: (Math.min(x0, x1) - s.left) / s.w, y: (Math.min(y0, y1) - s.top) / s.h, w: w / s.w, h: hh / s.h });
        };
        st.addEventListener("pointermove", move);
        st.addEventListener("pointerup", up);
      });
    });
    return panel;
  }

  function setTagFilter(tag) {
    tagFilter = normTag(tag);
    lastPicked = -1;
    renderGrid();
  }

  function showImagesWithTag(tag) {
    switchSubview("gallery");
    setTagFilter(tag);
    document.getElementById("caption-grid").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------ undo (bulk changes) ------------------------------

  async function refreshUndo() {
    let st = { depth: 0, label: "" };
    if (currentFolder) { try { st = await Api.captionUndoStatus(currentFolder); } catch { /* ignore */ } }
    document.querySelectorAll(".caption-undo-btn").forEach((b) => {
      b.classList.toggle("open", st.depth > 0);
      b.textContent = st.depth > 0 ? `↶ Undo: ${st.label}` : "";
    });
  }

  async function undoLast() {
    try {
      const r = await Api.captionUndo(currentFolder);
      toast(r.label ? `Undid "${r.label}" (${r.restored} caption${r.restored === 1 ? "" : "s"} restored).` : "Nothing to undo.", r.label ? "success" : "info");
      await refresh();
    } catch (err) { toast(err.message, "error"); }
  }

  function renderTagFrequency(tags) {
    const list = document.getElementById("tag-freq-list");
    list.innerHTML = "";
    document.getElementById("tag-freq-count").textContent = `${tags.length} unique`;
    tags.slice(0, 300).forEach((t) => {
      list.appendChild(h("div", {
        class: "tag-freq-item",
        onclick: () => { document.getElementById("bulk-remove-tag").value = t.tag; },
        title: "Click to fill into 'Remove tag from all'",
      }, [h("span", {}, t.tag), h("span", { class: "n" }, t.count)]));
    });
  }

  // ------------------------------ caption tool switcher ------------------------------

  // Booru tagger vs natural-language writer - one card at a time, so the
  // gallery column stays short. The choice is remembered between visits.
  function setCaptionTool(name) {
    document.querySelectorAll("#caption-tool-toggle button").forEach((b) => b.classList.toggle("active", b.dataset.tool === name));
    document.getElementById("caption-booru-card").style.display = name === "booru" ? "block" : "none";
    document.getElementById("caption-nl-card").style.display = name === "nl" ? "block" : "none";
    try { localStorage.setItem("animaStudio.captionTool", name); } catch { /* storage unavailable - fine */ }
  }

  // ------------------------------ Tag Frequency (full view / exclude) ------------------------------

  function switchSubview(name) {
    document.querySelectorAll("#caption-subview-toggle button").forEach((b) => b.classList.toggle("active", b.dataset.subview === name));
    document.getElementById("caption-subview-gallery").style.display = name === "gallery" ? "block" : "none";
    document.getElementById("caption-subview-tagfreq").style.display = name === "tagfreq" ? "block" : "none";
    const rules = document.getElementById("caption-subview-rules");
    if (rules) rules.style.display = name === "rules" ? "block" : "none";
  }

  function visibleTags() {
    const query = (document.getElementById("tagfreq-search").value || "").trim().toLowerCase();
    // Live tags from the backend, plus any tag the user has excluded this
    // session that's no longer live (it was fully removed) - shown struck
    // through using its remembered count so it doesn't just vanish.
    const liveNames = new Set(allTags.map((t) => t.tag));
    const combined = [...allTags];
    for (const [tag, snap] of excludedTags) {
      if (!liveNames.has(tag)) combined.push({ tag, count: snap.count, example: snap.example });
    }
    let list = combined;
    if (query) list = list.filter((t) => t.tag.toLowerCase().includes(query));
    if (tagFreqSort === "az") {
      list = [...list].sort((a, b) => a.tag.localeCompare(b.tag));
    } else {
      list = [...list].sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
    }
    return list;
  }

  function renderTagFreqCloud() {
    const cloud = document.getElementById("tagfreq-cloud");
    const tags = visibleTags();
    cloud.innerHTML = "";
    document.getElementById("tagfreq-full-count").textContent = `${allTags.length} unique`;

    if (tags.length === 0) {
      cloud.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "No tags match that filter.")]));
      return;
    }

    tags.forEach((t) => {
      const isExcluded = excludedTags.has(t.tag);
      const chip = h("span", {
        class: `tagfreq-chip${isExcluded ? " excluded" : ""}`,
        onclick: () => toggleTagExcluded(t, chip),
      }, [
        h("span", { class: "name" }, t.tag),
        h("span", { class: "n" }, `${t.count}`),
        h("button", {
          class: "tagfreq-show-btn", type: "button", title: "Show only the images with this tag",
          onclick: (e) => { e.stopPropagation(); showImagesWithTag(t.tag); },
        }, "▦"),
        h("button", {
          class: "tagfreq-edit-btn", type: "button", title: "Rename this tag",
          onclick: (e) => { e.stopPropagation(); startEditTag(t, chip); },
        }, "✎"),
      ]);
      cloud.appendChild(chip);
    });
  }

  function startEditTag(tagEntry, chipEl) {
    const input = h("input", { type: "text", class: "tagfreq-edit-input" });
    input.value = tagEntry.tag;
    chipEl.innerHTML = "";
    chipEl.classList.add("editing");
    chipEl.appendChild(input);
    input.focus();
    input.select();

    let settled = false;
    const commit = async () => {
      if (settled) return;
      settled = true;
      const newName = input.value.trim();
      if (!newName || newName === tagEntry.tag) return renderTagFreqCloud();
      try {
        const r = await Api.bulkFindReplace(currentFolder, tagEntry.tag, newName, true);
        toast(`Renamed "${tagEntry.tag}" to "${newName}" in ${r.changed} images.`, "success");
        excludedTags.delete(tagEntry.tag); // a rename supersedes any pending exclusion of the old name
        await refresh();
      } catch (err) {
        toast(err.message, "error");
        renderTagFreqCloud();
      }
    };
    const cancel = () => {
      if (settled) return;
      settled = true;
      renderTagFreqCloud();
    };

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); commit(); }
      if (e.key === "Escape") { e.preventDefault(); cancel(); }
    });
    input.addEventListener("blur", commit);
  }

  async function toggleTagExcluded(tagEntry, chipEl) {
    // The chip's click listener survives startEditTag() (h() uses
    // addEventListener), so a click inside the rename box must not run
    // "exclude this tag" - that would bulk-remove it from every caption.
    if (chipEl.classList.contains("editing")) return;
    const tag = tagEntry.tag;
    chipEl.classList.add("pending");
    try {
      if (excludedTags.has(tag)) {
        // Restore - re-add the tag to exactly the images it came from.
        const snap = excludedTags.get(tag);
        await Api.addTagToImages(snap.imagePaths, tag);
        excludedTags.delete(tag);
        toast(`Restored "${tag}" to ${snap.imagePaths.length} image${snap.imagePaths.length === 1 ? "" : "s"}.`, "success");
      } else {
        // Exclude - snapshot which currently-loaded images have it, then remove everywhere.
        const imagePaths = images
          .filter((img) => splitTags(img.caption).includes(tag))
          .map((img) => img.path);
        await Api.bulkRemoveTags(currentFolder, [tag]);
        excludedTags.set(tag, { count: tagEntry.count, example: tagEntry.example, imagePaths });
        toast(`Removed "${tag}" from ${imagePaths.length} image${imagePaths.length === 1 ? "" : "s"}.`, "success");
      }
      await refresh();
    } catch (err) {
      toast(err.message, "error");
      chipEl.classList.remove("pending");
    }
  }

  async function runAutotag() {
    if (!currentFolder) return toast("Load a folder first.", "error");
    const modelInfo = taggerModelsData.find((m) => m.name === document.getElementById("tagger-model").value);
    if (modelInfo && modelInfo.needs_deps && !modelInfo.deps_available) {
      return toast("This model needs its extra dependencies installed first - see the notice above the button.", "error");
    }
    const btn = document.getElementById("autotag-btn");
    const spinner = document.getElementById("autotag-spinner");
    btn.disabled = true;
    spinner.style.display = "inline-block";
    try {
      const data = await Api.autotag({
        folder: currentFolder,
        model: document.getElementById("tagger-model").value,
        general_threshold: parseFloat(document.getElementById("tagger-general-thresh").value),
        character_threshold: parseFloat(document.getElementById("tagger-char-thresh").value),
        mode: document.getElementById("tagger-mode").value,
        colour_guard: document.getElementById("tagger-colour-guard").checked,
      });
      toast(`Tagged ${data.tagged} of ${data.total} images.`, "success");
      await refresh();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
      spinner.style.display = "none";
    }
  }

  return {
    init,
    setFolder: (path) => { document.getElementById("caption-folder").value = path; load(); },
    folder: () => currentFolder,
    tagList: () => allTags,
    refresh,
    showImagesWithTag,
  };
})();
