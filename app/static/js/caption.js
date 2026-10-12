const Caption = (() => {
  let currentFolder = "";
  let images = [];
  let allTags = []; // full tag frequency list, cached for the Tag Frequency subview
  // tag -> { count, example, imagePaths } snapshot taken at the moment a
  // tag is excluded, so clicking it again restores it to exactly the
  // images it came from (not every image in the folder).
  const excludedTags = new Map();
  let tagFreqSort = "count";

  // Splits on commas (this app's own convention) or newlines - some
  // datasets tagged elsewhere write one tag per line instead of
  // comma-separating them; matches the same splitting the backend's
  // tag_frequency() does so per-tag counts/filtering agree with the UI.
  function splitTags(text) {
    return (text || "").split(/[,\n]+/).map((t) => t.trim()).filter(Boolean);
  }

  const saveCaption = debounce(async (path, caption) => {
    try {
      await Api.captionSave(path, caption);
    } catch (err) {
      toast(err.message, "error");
    }
  }, 500);

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
    document.getElementById("tagger-deps-install-btn").addEventListener("click", startTaggerDepsInstall);

    document.querySelectorAll("#caption-subview-toggle button").forEach((btn) => {
      btn.addEventListener("click", () => switchSubview(btn.dataset.subview));
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
  }

  // ------------------------------ WD14 model download ------------------------------

  let taggerPollTimer = null;
  let taggerDepsPollTimer = null;
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

  async function startTaggerDepsInstall() {
    const btn = document.getElementById("tagger-deps-install-btn");
    const spinner = document.getElementById("tagger-deps-spinner");
    btn.disabled = true;
    spinner.style.display = "inline-block";
    try {
      renderTaggerDepsStatus(await Api.taggerInstallDeps());
      if (!taggerDepsPollTimer) taggerDepsPollTimer = setInterval(pollTaggerDepsStatus, 1000);
    } catch (err) {
      toast(err.message, "error");
      btn.disabled = false;
      spinner.style.display = "none";
    }
  }

  async function pollTaggerDepsStatus() {
    try {
      renderTaggerDepsStatus(await Api.taggerInstallDepsStatus());
    } catch { /* transient - next poll retries */ }
  }

  function renderTaggerDepsStatus(status) {
    const log = document.getElementById("tagger-deps-log");
    const btn = document.getElementById("tagger-deps-install-btn");
    const spinner = document.getElementById("tagger-deps-spinner");

    if (status.log_tail && status.log_tail.length) {
      log.style.display = "block";
      log.textContent = status.log_tail.join("\n");
      log.scrollTop = log.scrollHeight;
    }

    if (status.stage === "installing") {
      btn.disabled = true;
      spinner.style.display = "inline-block";
      return;
    }

    spinner.style.display = "none";
    if (taggerDepsPollTimer) { clearInterval(taggerDepsPollTimer); taggerDepsPollTimer = null; }

    if (status.stage === "done") {
      const info = taggerModelsData.find((m) => m.name === document.getElementById("tagger-model").value);
      if (info) info.deps_available = true;
      updateTaggerDepsRow();
      toast("Dependencies installed - the model is ready to use.", "success");
    } else if (status.stage === "error") {
      btn.disabled = false;
      toast(status.error || "Install failed - see the log.", "error");
    } else {
      btn.disabled = false;
    }
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

  function selectSubsetTab(path) {
    document.getElementById("caption-folder").value = path;
    return load();
  }

  async function load() {
    const newFolder = document.getElementById("caption-folder").value.trim();
    if (!newFolder) return toast("Enter or browse to a folder first.", "error");
    if (newFolder !== currentFolder) excludedTags.clear();
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
      await showNestedHintIfEmpty();
    } catch (err) {
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
    const visible = query
      ? images.filter((img) => img.filename.toLowerCase().includes(query) || img.caption.toLowerCase().includes(query))
      : images;
    document.getElementById("caption-grid-count").textContent =
      query ? `${visible.length} of ${images.length}` : `${images.length} images`;

    for (const img of visible) {
      const tagCount = img.caption ? splitTags(img.caption).length : 0;
      const countEl = h("span", { class: "tag-count" }, `${tagCount} tags`);
      const textarea = h("textarea", {
        placeholder: "no caption yet",
        oninput: (e) => {
          img.caption = e.target.value;
          countEl.textContent = `${splitTags(e.target.value).length} tags`;
          saveCaption(img.path, e.target.value);
        },
      });
      textarea.value = img.caption;

      const card = h("div", { class: "caption-card" }, [
        h("div", { class: "thumb-wrap" }, [
          h("img", { src: Api.thumbUrl(img.path), loading: "lazy", onclick: () => openLightbox(Api.imageUrl(img.path), img.caption) }),
        ]),
        h("div", { class: "body" }, [
          h("div", { class: "filename" }, img.filename),
          textarea,
          h("div", { class: "card-foot" }, [
            countEl,
          ]),
        ]),
      ]);
      grid.appendChild(card);
    }
    if (query && visible.length === 0) {
      grid.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "No images match that search.")]));
    }
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
    chipEl.onclick = null;
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

  return { init, setFolder: (path) => { document.getElementById("caption-folder").value = path; load(); } };
})();
