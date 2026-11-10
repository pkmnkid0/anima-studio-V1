// Studio module: wires the spec's higher-level features
// (dataset health, caption rules/templates, step
// calculator, training history, LoRA packaging)
// into the UI IDs declared in index.html.
const Studio = (() => {
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === "dataset" && v && typeof v === "object") {
        for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
      }
      else if (v !== undefined && v !== null) node.setAttribute(k, v);
    }
    for (const child of children.flat()) {
      if (child == null) continue;
      node.append(child.nodeType ? child : document.createTextNode(child));
    }
    return node;
  }

  const $ = (id) => document.getElementById(id);

  const fmtNum = (n) => (typeof n === "number" ? n.toLocaleString() : n);

  function toastError(err) { toast(err.message || String(err), "error"); }

  // ------------------------------------------------------------------
  // Collect tab: Dataset Health (§2, §14, §15)
  // ------------------------------------------------------------------
  let healthData = null;

  function initHealth() {
    const folderEl = $("health-folder");
    if (!folderEl) return;
    $("health-browse")?.addEventListener("click", () => {
      FolderBrowser.open(folderEl.value || null, (p) => { folderEl.value = p; });
    });
    const run = async (deep) => {
      const folder = folderEl.value.trim();
      if (!folder) return toast("Pick a dataset folder first.", "error");
      $("health-spinner").style.display = "inline-block";
      $("health-status").textContent = deep ? "Analyzing (full scan)…" : "Analyzing…";
      try {
        healthData = await Api.analyzeDataset(folder, deep);
        renderHealth(healthData);
      } catch (err) { toastError(err); }
      finally { $("health-spinner").style.display = "none"; }
    };
    $("health-analyze-btn")?.addEventListener("click", () => run(true));
    $("health-quick-btn")?.addEventListener("click", () => run(false));
    $("health-cleanup-select-dupes")?.addEventListener("click", () => {
      document.querySelectorAll("#health-cleanup-list input[type=checkbox]").forEach((box) => {
        box.checked = /duplicate/i.test(box.dataset.reason || "");
      });
      refreshCleanupApply();
    });
    // The apply button reflects whatever is currently ticked (it used to
    // stay disabled forever - nothing ever enabled it).
    $("health-cleanup-list")?.addEventListener("change", (e) => {
      if (e.target.matches("input[type=checkbox]")) refreshCleanupApply();
    });
    $("health-cleanup-apply")?.addEventListener("click", async () => {
      const decisions = {};
      document.querySelectorAll("#health-cleanup-list input[type=checkbox]:checked").forEach((box) => {
        decisions[box.dataset.path] = "remove";
      });
      if (!Object.keys(decisions).length) return toast("Tick the items you want removed first.", "error");
      if (!confirm(`Permanently delete ${Object.keys(decisions).length} item(s)? Their .txt sidecars are removed too.`)) return;
      try {
        const res = await Api.analyzeCleanupApply(folderEl.value.trim(), decisions);
        $("health-cleanup-status").textContent = `Removed ${res.removed.length} item(s).`;
        toast(`Removed ${res.removed.length} item(s).`, "success");
        run(false);
      } catch (err) { toastError(err); }
    });
  }

  function renderHealth(data) {
    $("health-results").style.display = "block";
    const h = data.health || {};
    $("health-score").textContent = h.score ?? "–";
    $("health-grade").textContent = h.grade || "";
    const ring = $("health-ring");
    if (ring) ring.dataset.grade = (h.grade || "").toLowerCase().replace(/\s+/g, "-");

    // Summary stats
    const stats = [
      ["Images", fmtNum(data.count)],
      ["Total size", `${data.total_mb} MB`],
      ["Caption coverage", `${data.caption_coverage}%`],
      ["Corrupt", data.corrupt.length],
      ["Tiny (<256px)", data.tiny.length],
      ["Exact duplicates", data.exact_duplicates.length ? data.exact_duplicates.reduce((a, g) => a + g.length - 1, 0) : 0],
      ["Near-duplicates", data.near_duplicates.length ? data.near_duplicates.reduce((a, g) => a + g.length - 1, 0) : 0],
      ["Extreme aspects", data.extreme_aspect.length],
    ];
    const summary = $("health-summary");
    summary.innerHTML = "";
    stats.forEach(([label, value]) => {
      summary.appendChild(el("div", { class: "summary-stat" }, [
        el("span", { class: "v" }, String(value)),
        el("span", { class: "l" }, label),
      ]));
    });

    // Penalties
    const pen = $("health-penalties");
    pen.innerHTML = "";
    if (!h.penalties || !h.penalties.length) {
      pen.appendChild(el("p", { class: "card-hint", style: "margin:0" }, "No problems found - this dataset is in great shape."));
    } else {
      h.penalties.forEach((p) => {
        pen.appendChild(el("div", { class: "penalty-row" }, [
          el("span", { class: "penalty-pts" }, `-${p.points}`),
          el("span", {}, p.label),
          el("span", { class: "penalty-detail" }, p.detail ? ` — ${p.detail}` : ""),
        ]));
      });
    }

    // Distributions
    const resolutions = $("health-resolutions");
    resolutions.innerHTML = "";
    Object.entries(data.resolutions).forEach(([res, count]) => {
      resolutions.appendChild(el("div", { class: "dist-row" }, [
        el("span", { class: "dist-label mono" }, res),
        el("span", { class: "dist-bar" }, [el("span", { class: "dist-fill", style: `width:${Math.max(4, 100 * count / data.count)}%` })]),
        el("span", { class: "dist-count" }, count),
      ]));
    });
    const aspects = $("health-aspects");
    aspects.innerHTML = "";
    Object.entries(data.aspect_ratio_split).forEach(([label, count]) => {
      aspects.appendChild(el("div", { class: "dist-row" }, [
        el("span", { class: "dist-label" }, label),
        el("span", { class: "dist-bar" }, [el("span", { class: "dist-fill", style: `width:${Math.max(4, 100 * count / data.count)}%` })]),
        el("span", { class: "dist-count" }, count),
      ]));
    });

    // Balance
    renderBalance(data);

    // Coverage map (§1)
    if (data.coverage) renderCoverage(data.coverage);

    // Cleanup plan
    renderCleanup(data);

    const statusEl = $("health-status");
    if (statusEl) statusEl.textContent = `Analyzed ${data.count} image(s) in ${data.folder}.`;
  }

  async function renderBalance(data) {
    const container = $("health-balance");
    container.innerHTML = "";
    try {
      const bal = await Api.analyzeBalance(data.folder);
      Object.entries(bal.groups || {}).forEach(([groupName, dist]) => {
        const block = el("div", { class: "balance-group" }, [el("div", { class: "section-label" }, groupName)]);
        dist.forEach((d) => {
          block.appendChild(el("div", { class: "dist-row" }, [
            el("span", { class: "dist-label" }, d.label),
            el("span", { class: "dist-bar" }, [el("span", { class: "dist-fill", style: `width:${Math.max(2, d.pct)}%` })]),
            el("span", { class: "dist-count" }, `${d.count} (${d.pct}%)`),
          ]));
        });
        container.appendChild(block);
      });
      const warnings = $("health-balance-warnings");
      if (bal.warnings && bal.warnings.length) {
        warnings.style.display = "block";
        warnings.replaceChildren(...bal.warnings.map((w) => el("div", {}, w)));
      } else {
        warnings.style.display = "none";
      }
    } catch (err) {
      container.appendChild(el("p", { class: "card-hint" }, "Balance analysis unavailable for this folder."));
    }
  }

  function renderCoverage(coverage) {
    const box = $("health-coverage");
    if (!box) return;
    box.innerHTML = "";
    (coverage.rows || []).forEach((row) => {
      const gradeClass = `grade-${row.grade.toLowerCase()}`;
      const rowEl = el("div", { class: "dist-row compact", style: "cursor:pointer" }, [
        el("span", { class: "dist-label" }, row.label),
        el("span", { class: "dist-bar" }, [el("span", { class: "dist-fill", style: `width:${Math.max(2, row.pct)}%` })]),
        el("span", { class: "dist-count" }, `${row.count} (${row.pct}%)`),
        el("span", { class: gradeClass, style: "min-width:72px;text-align:right;font-size:11.5px;font-weight:600" }, row.grade),
      ]);
      rowEl.addEventListener("click", () => showCoverageImages(row));
      box.appendChild(rowEl);
    });
  }

  function showCoverageImages(row) {
    const wrap = $("health-coverage-images");
    const grid = $("health-coverage-images-grid");
    const label = $("health-coverage-images-label");
    if (!wrap || !grid) return;
    wrap.style.display = "block";
    label.textContent = `${row.label} — ${row.count} image(s)`;
    grid.innerHTML = "";
    (row.images || []).forEach((path) => {
      grid.appendChild(el("img", {
        src: `/api/files/image?path=${encodeURIComponent(path)}`,
        alt: row.label, style: "width:100%;border-radius:6px", loading: "lazy", decoding: "async",
      }));
    });
  }

  function renderCleanup(data) {
    const list = $("health-cleanup-list");
    list.innerHTML = "";
    const plan = cleanupPlan(data);
    $("health-cleanup-count").textContent = plan.length;
    refreshCleanupApply();
    if (!plan.length) {
      list.appendChild(el("p", { class: "card-hint" }, "No cleanup candidates found."));
      return;
    }
    plan.forEach((item) => {
      list.appendChild(el("label", { class: "cleanup-row" }, [
        el("input", { type: "checkbox", dataset: { path: item.path, reason: item.reason } }),
        el("span", { class: "badge", style: item.action === "remove" ? "background:var(--danger-soft);color:var(--danger)" : "background:var(--accent-soft);color:var(--accent)", text: item.action }),
        el("span", { class: "cleanup-reason" }, item.reason),
        el("span", { class: "mono cleanup-path" }, item.path.replace(/^.*[\\/]/, "")),
      ]));
    });
  }

  // Join a folder + child name the way the server's Path/folder join does:
  // follow the folder's own separator and don't double a trailing one.
  function cleanupPathJoin(folder, name) {
    const sep = folder.includes("\\") ? "\\" : "/";
    return `${folder.replace(/[\\/]+$/, "")}${sep}${name}`;
  }

  function refreshCleanupApply() {
    const anyChecked = [...document.querySelectorAll("#health-cleanup-list input[type=checkbox]")].some((b) => b.checked);
    $("health-cleanup-apply").disabled = !anyChecked;
  }

  // Client-side mirror of the server's cleanup_plan so the UI
  // can render without a second round-trip.
  function cleanupPlan(analysis) {
    const plan = [];
    const seen = new Set();
    const add = (path, reason, action) => {
      if (seen.has(path)) return;
      seen.add(path);
      plan.push({ path, reason, action });
    };
    (analysis.corrupt || []).forEach((i) => add(i.path, `Corrupted: ${i.corrupt}`, "remove"));
    (analysis.tiny || []).forEach((i) => add(i.path, `Tiny image (${i.width}x${i.height})`, "review"));
    (analysis.extreme_aspect || []).forEach((i) => add(i.path, `Extreme aspect ratio (${i.aspect})`, "review"));
    (analysis.exact_duplicates || []).forEach((group) => group.slice(1).forEach((name) => add(cleanupPathJoin(analysis.folder, name), "Exact duplicate", "remove")));
    (analysis.near_duplicates || []).forEach((group) => group.slice(1).forEach((p) => add(p, "Near-duplicate (visually redundant)", "review")));
    return plan;
  }

  // ------------------------------------------------------------------
  // Train studio tab (§5, §6, §16, §17, §18, §20, §22)
  // ------------------------------------------------------------------
  async function initStudioTab() {
    const modelSel = $("studio-model");

    // Calculator
    $("calc-run-btn")?.addEventListener("click", async () => {
      try {
        const res = await Api.trainCalculator({
          images: Number($("calc-images").value),
          repeats: Number($("calc-repeats").value),
          epochs: Number($("calc-epochs").value),
          batch_size: Number($("calc-batch").value),
          grad_accum: Number($("calc-gradacc").value),
          project_type: $("calc-type").value,
          steps_per_sec: Number($("calc-sps").value),
        });
        renderCalculator(res);
      } catch (err) { toastError(err); }
    });

    // History
    const refreshHistory = async () => {
      try {
        const { runs } = await Api.trainHistory();
        renderHistory(runs);
      } catch (err) { toastError(err); }
    };
    $("history-refresh-btn")?.addEventListener("click", refreshHistory);
    $("history-compare-btn")?.addEventListener("click", async () => {
      const ids = [...document.querySelectorAll(".history-check:checked")].map((b) => b.dataset.id);
      if (ids.length < 2) return toast("Select at least two runs to compare.", "error");
      try {
        const res = await Api.trainHistoryCompare(ids);
        renderHistoryCompare(res.runs);
      } catch (err) { toastError(err); }
    });
    refreshHistory();

    // Packaging
    const wireBrowse = (inputId, btnId) => {
      $(btnId)?.addEventListener("click", () => {
        FolderBrowser.open($(inputId).value || null, (p) => { $(inputId).value = p; });
      });
    };
    // ---- LoRA slimmer ----
    let slimPick = null;
    const slimPathVal = () => $("slim-path").value.trim().replace(/^"|"$/g, "");
    async function slimAnalyze() {
      const path = slimPathVal();
      if (!path) return toast("Pick a LoRA file first.", "error");
      $("slim-spinner").style.display = "inline-block";
      $("slim-options").style.display = "none";
      try {
        const r = await Api.slimAnalyze(path, $("slim-dtype").value);
        $("slim-info").textContent = `${r.size_mb} MB · ${r.layers} layers · rank ${r.max_rank} · ${r.dtype}` +
          (r.base_model ? ` · ${r.base_model}` : "") + (r.skipped ? ` · ${r.skipped} other tensors copied untouched` : "");
        const t = $("slim-table");
        t.replaceChildren(
          h("thead", {}, h("tr", {}, ["", "Option", "Size", "Saved", "Kept", "Avg rank"].map((x) => h("th", {}, x)))),
          h("tbody", {}, r.options.map((o) => h("tr", {}, [
            h("td", {}, h("input", { type: "radio", name: "slim-opt", value: o.id,
              checked: o.id === r.recommended, onchange: () => { slimPick = o; } })),
            h("td", {}, o.label + (o.id === r.recommended ? "  ★ recommended" : "")),
            h("td", {}, `${o.size_mb} MB`), h("td", {}, `${o.saved_pct}%`),
            h("td", {}, `${o.kept_pct}%`), h("td", {}, String(o.avg_rank))])))
        );
        slimPick = r.options.find((o) => o.id === r.recommended) || null;
        $("slim-options").style.display = "block";
        if (!r.recommended) toast("No option keeps 99% - pick one yourself.", "info");
      } catch (err) { toastError(err); }
      finally { $("slim-spinner").style.display = "none"; }
    }
    $("slim-analyze-btn")?.addEventListener("click", slimAnalyze);
    $("slim-browse")?.addEventListener("click", () => {
      const cur = slimPathVal();
      FileBrowser.open(cur ? cur.replace(/[\\/][^\\/]*$/, "") : null, "safetensors", (p) => {
        $("slim-path").value = p; slimAnalyze();
      });
    });
    $("slim-dtype")?.addEventListener("change", () => { if (slimPathVal() && $("slim-options").style.display !== "none") slimAnalyze(); });
    $("slim-run-btn")?.addEventListener("click", async () => {
      if (!slimPick) return toast("Pick one of the options first.", "error");
      $("slim-spinner").style.display = "inline-block";
      try {
        const r = await Api.slimLora({ path: slimPathVal(), mode: slimPick.mode, value: slimPick.value, dtype: $("slim-dtype").value });
        $("slim-status").textContent = `Saved ${r.out_path} — ${r.size_before_mb} → ${r.size_after_mb} MB, kept ${r.kept_pct}%`;
        toast("Slim copy saved.", "success");
      } catch (err) { toastError(err); }
      finally { $("slim-spinner").style.display = "none"; }
    });

    // ---- LoRA metadata ----
    const metaPathVal = () => $("meta-path").value.trim().replace(/^"|"$/g, "");
    let metaInfo = null;
    async function metaRead() {
      const path = metaPathVal();
      if (!path) return toast("Pick a .safetensors file first.", "error");
      $("meta-spinner").style.display = "inline-block";
      try {
        metaInfo = await Api.loraMetadata(path);
        $("meta-summary").textContent = metaInfo.has_metadata
          ? `${metaInfo.key_count} metadata key(s) · ${metaInfo.tensors} tensors · ${metaInfo.size_mb} MB`
          : `No metadata in this file · ${metaInfo.tensors} tensors · ${metaInfo.size_mb} MB`;
        $("meta-table").replaceChildren(
          h("thead", {}, h("tr", {}, ["Keep", "Key", "Value", "Size"].map((x) => h("th", {}, x)))),
          h("tbody", {}, metaInfo.keys.map((k) => h("tr", {}, [
            h("td", {}, h("input", { type: "checkbox", "data-key": k.key })),
            h("td", { class: "mono" }, k.key),
            h("td", { title: k.value }, k.value.length > 120 ? `${k.value.slice(0, 120)}…` : k.value),
            h("td", {}, `${k.size} B`),
          ])))
        );
        $("meta-panel").style.display = metaInfo.has_metadata ? "block" : "none";
        if (!metaInfo.has_metadata) toast("This file has no metadata.", "info");
      } catch (err) { toastError(err); $("meta-panel").style.display = "none"; }
      finally { $("meta-spinner").style.display = "none"; }
    }
    $("meta-read-btn")?.addEventListener("click", metaRead);
    $("meta-browse")?.addEventListener("click", () => {
      const cur = metaPathVal();
      FileBrowser.open(cur ? cur.replace(/[\\/][^\\/]*$/, "") : null, "safetensors", (p) => {
        $("meta-path").value = p;
        metaRead();
      });
    });
    $("meta-all-btn")?.addEventListener("click", () => {
      document.querySelectorAll("#meta-table input[type=checkbox]").forEach((b) => { b.checked = true; });
    });
    $("meta-none-btn")?.addEventListener("click", () => {
      document.querySelectorAll("#meta-table input[type=checkbox]").forEach((b) => { b.checked = false; });
    });
    $("meta-strip-btn")?.addEventListener("click", async () => {
      if (!metaInfo || !metaInfo.has_metadata) return toast("Read a file with metadata first.", "error");
      const keep = [...document.querySelectorAll("#meta-table input[type=checkbox]:checked")].map((b) => b.dataset.key);
      $("meta-spinner").style.display = "inline-block";
      try {
        const r = await Api.loraStripMetadata({ path: metaPathVal(), keep });
        $("meta-status").textContent = `Saved ${r.out_path} — ${r.size_before_mb} → ${r.size_after_mb} MB · ` +
          `${r.removed.length} key(s) removed${r.kept.length ? `, ${r.kept.length} kept` : ""} · weights verified unchanged`;
        toast("Copy saved without the unticked metadata.", "success");
      } catch (err) { toastError(err); }
      finally { $("meta-spinner").style.display = "none"; }
    });

    wireBrowse("package-out", "package-out-browse");
    wireBrowse("package-dataset", "package-dataset-browse");
    $("package-lora-browse")?.addEventListener("click", () => {
      const cur = $("package-lora").value.trim().replace(/^"|"$/g, "");
      const startDir = cur ? cur.replace(/[\\/][^\\/]*$/, "") : null;
      FileBrowser.open(startDir || null, "safetensors", (p) => {
        $("package-lora").value = p;
        autofillPackage(p);
      });
    });
    $("package-lora")?.addEventListener("change", () => {
      const v = $("package-lora").value.trim().replace(/^"|"$/g, "");
      $("package-lora").value = v;
      if (v) autofillPackage(v);
    });
    async function autofillPackage(path) {
      try {
        const info = await Api.packageInspect(path);
        if (!$("package-name").value.trim()) $("package-name").value = info.name || "";
        if (!$("package-triggers").value.trim() && (info.trigger_words || []).length) {
          $("package-triggers").value = info.trigger_words.join(", ");
        }
        const s = [];
        if (info.network_dim) s.push(`rank ${info.network_dim}`);
        if (info.base_model) s.push(info.base_model);
        if (info.sample_images && info.sample_images.length) s.push(`${info.sample_images.length} sample images will be included`);
        $("package-status").textContent = s.length ? `Detected: ${s.join(" · ")}` : "";
      } catch (err) { /* inspection is a convenience only */ }
    }
    $("package-run-btn")?.addEventListener("click", async () => {
      const lora = $("package-lora").value.trim();
      const out = $("package-out").value.trim();
      if (!lora || !out) return toast("LoRA file and output folder are required.", "error");
      $("package-spinner").style.display = "inline-block";
      try {
        const res = await Api.trainPackage({
          lora_path: lora,
          output_dir: out,
          name: $("package-name").value.trim() || null,
          trigger_words: $("package-triggers").value.split(",").map((s) => s.trim()).filter(Boolean),
          description: $("package-desc").value.trim(),
          dataset_folder: $("package-dataset").value.trim() || null,
          model_type: modelSel ? modelSel.value : "anima",
          project_type: $("calc-type").value,
        });
        $("package-status").textContent = `Packaged to ${res.folder} (+ ${res.zip}) — ${res.files.length} files` +
          (res.manifest && res.manifest.sample_count ? `, ${res.manifest.sample_count} samples` : "");
        toast("LoRA packaged.", "success");
      } catch (err) { toastError(err); }
      finally { $("package-spinner").style.display = "none"; }
    });
  }

  function renderCalculator(res) {
    $("calc-results").style.display = "block";
    const stats = [
      ["Images / epoch", fmtNum(res.images_per_epoch)],
      ["Total exposures", fmtNum(res.total_exposures)],
      ["Steps / epoch", fmtNum(res.steps_per_epoch)],
      ["Total steps", fmtNum(res.total_steps)],
      ["Eff. batch", res.effective_batch],
      ["Exposure / image", `${res.exposure_per_image}×`],
      ["Est. duration", res.duration ? res.duration.human : "–"],
      ["Epochs for target", Object.entries(res.epochs_for_target || {}).map(([k, v]) => `${v} @${k}`).join(", ")],
    ];
    const statsBox = $("calc-stats");
    statsBox.innerHTML = "";
    stats.forEach(([label, value]) => {
      statsBox.appendChild(el("div", { class: "summary-stat" }, [
        el("span", { class: "v" }, String(value)),
        el("span", { class: "l" }, label),
      ]));
    });
    const warnings = $("calc-warnings");
    if (res.warnings && res.warnings.length) {
      warnings.style.display = "block";
      warnings.replaceChildren(...res.warnings.map((w) => el("div", {}, w)));
    } else {
      warnings.style.display = "none";
    }
  }

  function renderHistory(runs) {
    const box = $("history-list");
    box.innerHTML = "";
    $("history-count").textContent = runs.length;
    $("history-compare-btn").disabled = runs.length < 2;
    if (!runs.length) {
      box.appendChild(el("p", { class: "card-hint" }, "No runs recorded yet."));
      return;
    }
    runs.forEach((run) => {
      box.appendChild(el("div", { class: "history-row" }, [
        el("input", { type: "checkbox", class: "history-check", "data-id": run.id }),
        el("div", { class: "history-main" }, [
          el("strong", {}, run.name || run.id),
          el("span", { class: "history-meta" }, [
            `${run.model_type || "?"} · ${run.project_type || "?"} · rank ${run.network_dim ?? "?"} · ${fmtNum(run.max_train_steps)} steps · loss ${run.final_loss ?? "–"}`,
          ]),
        ]),
        el("button", {
          class: "btn small ghost", text: "Delete",
          onclick: async () => {
            try { await Api.trainHistoryDelete(run.id); refreshHistoryInline(); toast("Run removed.", "info"); }
            catch (err) { toastError(err); }
          },
        }),
      ]));
    });
    function refreshHistoryInline() {
      Api.trainHistory().then(({ runs }) => renderHistory(runs)).catch(toastError);
    }
  }

  function renderHistoryCompare(rows) {
    $("history-compare").style.display = "block";
    const table = $("history-compare-table");
    table.innerHTML = "";
    if (!rows.length) return;
    const keys = Object.keys(rows[0]);
    const thead = el("thead", {}, [el("tr", {}, keys.map((k) => el("th", {}, k)))]);
    const tbody = el("tbody", {}, rows.map((row) => el("tr", {}, keys.map((k) => el("td", {}, String(row[k] ?? ""))))));
    table.append(thead, tbody);
  }

  function initDragDrop() {
    // Dropping a file anywhere outside a real drop zone would make the
    // browser navigate to it, silently replacing the whole app.
    ["dragover", "drop"].forEach((ev) =>
      document.addEventListener(ev, (e) => {
        if (!e.target.closest || !e.target.closest("[data-drop]")) e.preventDefault();
      }));
    document.querySelectorAll("[data-drop]").forEach((zone) => {
      ["dragenter", "dragover"].forEach((ev) =>
        zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drop-active"); }));
      ["dragleave", "drop"].forEach((ev) =>
        zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drop-active"); }));
      zone.addEventListener("drop", (e) => {
        const files = e.dataTransfer?.files;
        if (!files || !files.length) return;
        const path = files[0].path;
        if (!path) return toast("Drop a file from the desktop, not the browser.", "error");
        if (zone.dataset.drop === "folder") {
          const isImage = /\.(png|jpe?g|webp|gif|bmp)$/i.test(path);
          const folder = isImage ? path.replace(/[\\/][^\\/]+$/, "") : path;
          $("health-folder").value = folder;
          toast(`Dataset folder set: ${folder}`, "success");
        }
      });
    });
  }

  function init() {
    initHealth();
    initStudioTab();
    initDragDrop();
  }

  return { init };
})();

document.addEventListener("DOMContentLoaded", () => {
  try { Studio.init(); } catch (err) { console.error("Studio init failed", err); }
  try { Projects.init(); } catch (err) { console.error("Projects init failed", err); }
});
