// Caption > Cleanup panels: prune redundant tags, blacklist, tidy & sort,
// colour fix and tag relationships. The Gallery and Tag Frequency live in
// caption.js; this file only adds the Cleanup tools.
const CaptionTools = (() => {
  const $ = (id) => document.getElementById(id);
  const folder = () => Caption.folder();
  const splitList = (text) => (text || "").split(/[,\n]+/).map((t) => t.trim()).filter(Boolean);

  function needFolder() {
    if (!folder()) { toast("Load a dataset folder first.", "error"); return false; }
    return true;
  }

  async function run(label, fn) {
    if (!needFolder()) return false;
    try {
      const r = await fn();
      toast(`${label}: changed ${r.changed} caption${r.changed === 1 ? "" : "s"}.`, "success");
      await Caption.refresh();
      return true;
    } catch (err) { toast(err.message, "error"); return false; }
  }

  // ------------------------------------------------- review lists (shared)
  // Prune and Variations use the same layout: a scan button, a live summary,
  // All / None, one primary action, and a list of rows with a checkbox.
  const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
  const countOf = (tag) => {
    const want = tag.toLowerCase().replace(/_/g, " ");
    const hit = Caption.tagList().find((t) => t.tag.toLowerCase().replace(/_/g, " ") === want);
    return hit ? hit.count : 0;
  };
  const chip = (text, cls, extra) => h("span", { class: `rtag ${cls || ""}` }, [text, extra ? h("em", {}, extra) : null]);

  function reviewBar(prefix, { any, picked, summary }) {
    $(`${prefix}-summary`).textContent = summary;
    $(`${prefix}-all`).style.display = any ? "" : "none";
    $(`${prefix}-none`).style.display = any ? "" : "none";
    $(`${prefix}-apply-btn`).disabled = !picked;
  }

  function emptyState(box, text) {
    box.innerHTML = "";
    box.appendChild(h("div", { class: "review-empty" }, text));
  }

  // ---------------------------------------------------------------- prune
  let pruneGroups = [];

  async function pruneScan() {
    if (!needFolder()) return;
    emptyState($("prune-list"), "Scanning…");
    try {
      pruneGroups = (await Api.pruneScan(folder())).groups || [];
    } catch (err) { emptyState($("prune-list"), ""); return toast(err.message, "error"); }
    // pre-tick the ones where the general tag is almost always the redundant one
    pruneGroups.forEach((g) => { g.on = g.general_total ? g.images / g.general_total >= 0.8 : false; });
    renderPrune();
  }

  function renderPrune() {
    const box = $("prune-list");
    if (!pruneGroups.length) {
      reviewBar("prune", { any: false, picked: 0, summary: "" });
      return emptyState(box, "Nothing redundant found. Your captions are already tidy.");
    }
    box.innerHTML = "";
    pruneGroups.forEach((g) => {
      const pct = g.general_total ? Math.round((g.images / g.general_total) * 100) : 0;
      const cb = h("input", { type: "checkbox" });
      cb.checked = g.on;
      cb.addEventListener("change", () => { g.on = cb.checked; row.classList.toggle("on", g.on); updatePruneBar(); });
      const row = h("label", { class: `review-row${g.on ? " on" : ""}` }, [
        h("span", { class: "checkbox-row", style: "margin:0" }, [cb, h("span", {})]),
        h("div", { class: "review-main" }, [
          h("div", { class: "review-line" }, [
            chip(g.general, "drop"),
            h("span", { class: "rlabel" }, "is redundant next to"),
            ...g.specifics.slice(0, 4).map((x) => chip(x.tag, "keep")),
            g.specifics.length > 4 ? h("span", { class: "rmore" }, `+${g.specifics.length - 4} more`) : null,
          ]),
        ]),
        h("div", { class: "review-meta" }, [
          h("div", { class: "rbar", title: `${g.images} of ${g.general_total} captions with "${g.general}"` }, [h("i", { style: `width:${pct}%` })]),
          h("span", {}, `${plural(g.images, "caption")}`),
        ]),
      ]);
      box.appendChild(row);
    });
    updatePruneBar();
  }

  function updatePruneBar() {
    const on = pruneGroups.filter((g) => g.on);
    const caps = on.reduce((n, g) => n + g.images, 0);
    reviewBar("prune", {
      any: true, picked: on.length,
      summary: `${plural(pruneGroups.length, "redundant tag")} found · ${on.length} selected${on.length ? ` · up to ${plural(caps, "caption")} change` : ""}`,
    });
  }

  async function pruneApply() {
    const pairs = [];
    pruneGroups.filter((g) => g.on).forEach((g) => g.specifics.forEach((x) => pairs.push({ general: g.general, specific: x.tag })));
    if (!pairs.length) return toast("Tick at least one tag first.", "error");
    if (!(await run("Pruned", () => Api.pruneApply(folder(), pairs)))) return;
    pruneGroups = [];
    reviewBar("prune", { any: false, picked: 0, summary: "Done. Scan again to check." });
    $("prune-list").innerHTML = "";
  }

  // ----------------------------------------------------------- variations
  let variationGroups = [];

  async function variationsScan() {
    if (!needFolder()) return;
    emptyState($("variations-list"), "Scanning…");
    try {
      variationGroups = ((await Api.captionVariations(folder())).variations || []).map((g) => ({
        tags: g.tags, keep: g.suggestion, on: true,
      }));
    } catch (err) { emptyState($("variations-list"), ""); return toast(err.message, "error"); }
    renderVariations();
  }

  function renderVariations() {
    const box = $("variations-list");
    if (!variationGroups.length) {
      reviewBar("variations", { any: false, picked: 0, summary: "" });
      return emptyState(box, "No look-alike tags found.");
    }
    box.innerHTML = "";
    variationGroups.forEach((g) => {
      const cb = h("input", { type: "checkbox" });
      cb.checked = g.on;
      const chips = h("div", { class: "review-line" });
      const meta = h("span", {});
      const paint = () => {
        chips.innerHTML = "";
        g.tags.forEach((t) => {
          const isKeep = t === g.keep;
          const b = h("button", { type: "button", class: `rtag pick ${isKeep ? "keep" : "drop"}`, title: isKeep ? "This one is kept" : "Click to keep this spelling instead" }, [
            t, h("em", {}, String(countOf(t))),
          ]);
          b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); g.keep = t; paint(); updateVariationBar(); });
          chips.appendChild(b);
        });
        const moved = g.tags.filter((t) => t !== g.keep).reduce((n, t) => n + countOf(t), 0);
        meta.textContent = `${plural(moved, "tag use")} renamed`;
      };
      cb.addEventListener("change", () => { g.on = cb.checked; row.classList.toggle("on", g.on); updateVariationBar(); });
      const row = h("div", { class: `review-row${g.on ? " on" : ""}` }, [
        h("label", { class: "checkbox-row", style: "margin:0" }, [cb, h("span", {})]),
        h("div", { class: "review-main" }, [
          chips,
          h("div", { class: "rhint" }, "Click the spelling you want to keep."),
        ]),
        h("div", { class: "review-meta" }, [meta]),
      ]);
      paint();
      box.appendChild(row);
    });
    updateVariationBar();
  }

  function updateVariationBar() {
    const on = variationGroups.filter((g) => g.on);
    reviewBar("variations", {
      any: true, picked: on.length,
      summary: `${plural(variationGroups.length, "group")} found · ${on.length} selected`,
    });
  }

  async function variationsApply() {
    const mapping = {};
    variationGroups.filter((g) => g.on).forEach((g) => g.tags.forEach((t) => {
      if (t === g.keep) return;
      mapping[t] = g.keep;
      // captions that still use underscores
      mapping[t.replace(/ /g, "_")] = g.keep.replace(/ /g, "_");
    }));
    if (!Object.keys(mapping).length) return toast("Tick at least one group first.", "error");
    if (!(await run("Merged", () => Api.captionStandardize(folder(), mapping)))) return;
    variationGroups = [];
    reviewBar("variations", { any: false, picked: 0, summary: "Done. Scan again to check." });
    $("variations-list").innerHTML = "";
  }

  // ------------------------------------------------------------ blacklist
  let commonJunk = [];

  async function loadBlacklist() {
    try {
      const d = await Api.blacklistGet();
      commonJunk = d.common || [];
      $("blacklist-text").value = (d.tags || []).join(", ");
      $("blacklist-auto").checked = d.apply_after_tagging !== false;
    } catch { /* offline on first paint */ }
  }

  async function saveBlacklist(quiet) {
    try {
      const d = await Api.blacklistSave(splitList($("blacklist-text").value), $("blacklist-auto").checked);
      $("blacklist-text").value = (d.tags || []).join(", ");
      if (!quiet) toast(`Blacklist saved (${d.tags.length} tag${d.tags.length === 1 ? "" : "s"}).`, "success");
      return true;
    } catch (err) { toast(err.message, "error"); return false; }
  }

  // --------------------------------------------------------- relationships
  async function showRelations() {
    if (!needFolder()) return;
    const tag = $("cooc-tag").value.trim();
    const box = $("cooc-list");
    box.textContent = "Working…";
    let d;
    try { d = await Api.cooccurrence(folder(), tag); } catch (err) { box.textContent = ""; return toast(err.message, "error"); }
    box.innerHTML = "";
    const link = (t) => h("a", { onclick: () => Caption.showImagesWithTag(t), title: "Show the images with this tag" }, t);
    if (tag) {
      if (!d.images) { box.appendChild(h("p", { class: "card-hint" }, `No image has "${tag}".`)); return; }
      box.appendChild(h("p", { class: "card-hint", style: "margin:0 0 6px" },
        `"${d.tag}" is in ${d.images} of ${d.total} images. Tags that appear with it:`));
      d.related.forEach((r) => {
        const pct = Math.round(r.pct * 100);
        box.appendChild(h("div", { class: "cleanup-row" }, [
          link(r.tag),
          h("div", { class: "cleanup-bar" }, [h("i", { style: `width:${pct}%` })]),
          h("span", {}, `${pct}% of these images`),
          h("span", { class: "meta" }, `${r.together} together · ${Math.round(r.of_other * 100)}% of all "${r.tag}"`),
        ]));
      });
    } else {
      if (!d.pairs.length) { box.appendChild(h("p", { class: "card-hint" }, "No tags are glued together (a pair that appears together in 90%+ of the images that have either one).")); return; }
      box.appendChild(h("p", { class: "card-hint", style: "margin:0 0 6px" },
        "Pairs that almost always appear together. The model will learn them as one thing:"));
      d.pairs.forEach((r) => {
        box.appendChild(h("div", { class: "cleanup-row" }, [
          link(r.a), h("span", { class: "arrow" }, "+"), link(r.b),
          h("span", { class: "meta" }, `${r.together} images`),
        ]));
      });
    }
  }

  function init() {
    $("prune-scan-btn").addEventListener("click", pruneScan);
    $("prune-apply-btn").addEventListener("click", pruneApply);
    $("prune-all").addEventListener("click", () => { pruneGroups.forEach((g) => { g.on = true; }); renderPrune(); });
    $("prune-none").addEventListener("click", () => { pruneGroups.forEach((g) => { g.on = false; }); renderPrune(); });
    $("variations-scan-btn").addEventListener("click", variationsScan);
    $("variations-apply-btn").addEventListener("click", variationsApply);
    $("variations-all").addEventListener("click", () => { variationGroups.forEach((g) => { g.on = true; }); renderVariations(); });
    $("variations-none").addEventListener("click", () => { variationGroups.forEach((g) => { g.on = false; }); renderVariations(); });

    $("blacklist-common-btn").addEventListener("click", () => {
      const have = new Set(splitList($("blacklist-text").value).map((t) => t.toLowerCase()));
      const add = commonJunk.filter((t) => !have.has(t.toLowerCase()));
      $("blacklist-text").value = [...splitList($("blacklist-text").value), ...add].join(", ");
      toast(add.length ? `Added ${add.length} common junk tags - press Save.` : "They are all in the list already.", "info");
    });
    $("blacklist-save-btn").addEventListener("click", () => saveBlacklist(false));
    $("blacklist-auto").addEventListener("change", () => saveBlacklist(true));
    $("blacklist-apply-btn").addEventListener("click", async () => {
      if (!needFolder()) return;
      if (!(await saveBlacklist(true))) return;
      await run("Blacklist applied", () => Api.blacklistApply(folder()));
    });

    $("dedupe-btn").addEventListener("click", () => run("Repeated tags removed", () => Api.dedupeTags(folder())));
    $("sort-btn").addEventListener("click", () =>
      run("Sorted", () => Api.autoSortTags(folder(), Number($("sort-keep").value || 0), splitList($("sort-pinned").value))));

    const cooc = $("cooc-tag");
    Autocomplete.attach(cooc, { getTags: () => Caption.tagList(), onPick: (t) => { cooc.value = t; showRelations(); } });
    cooc.addEventListener("keydown", (e) => { if (e.key === "Enter") showRelations(); });
    $("cooc-btn").addEventListener("click", showRelations);

    loadBlacklist();
  }

  return { init };
})();
