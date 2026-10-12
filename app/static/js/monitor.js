const Monitor = (() => {
  let autoRefreshTimer = null;
  // Whether the Monitor tab is the one on screen. Polling is gated on this
  // because the tab refreshes every 6s and, before this, kept doing so
  // while hidden - ~130 pointless requests per 2 minutes of work elsewhere.
  let isVisible = false;
  let lastRun = null;      // id of the run currently drawn
  let startWatch = null;
  let staleRun = null;     // previous run, ignored right after "Start training"

  function init() {
    // Prefill from whatever the Train tab currently has configured, if anything.
    const outputInput = document.getElementById("monitor-output-dir");
    const loggingInput = document.getElementById("monitor-logging-dir");
    if (!outputInput.value && Train.state.saving.output_dir) outputInput.value = Train.state.saving.output_dir;
    if (!loggingInput.value && Train.state.logging.logging_dir) loggingInput.value = Train.state.logging.logging_dir;

    document.getElementById("monitor-browse-output").addEventListener("click", () => {
      FolderBrowser.open(outputInput.value || null, (path) => { outputInput.value = path; refreshAll(); });
    });
    document.getElementById("monitor-browse-logging").addEventListener("click", () => {
      FolderBrowser.open(loggingInput.value || null, (path) => { loggingInput.value = path; refreshAll(); });
    });
    document.getElementById("monitor-refresh-btn").addEventListener("click", refreshAll);
    outputInput.addEventListener("change", refreshAll);
    loggingInput.addEventListener("change", refreshAll);
    document.getElementById("monitor-auto-refresh").addEventListener("change", () => {
      applyPolling();
      // Ticking the box while hidden shouldn't queue a refresh that then
      // runs 5 seconds before anyone can see it.
      if (isVisible) refreshAll();
    });

    document.addEventListener("studio:viewchange", (e) => {
      const wasVisible = isVisible;
      isVisible = e.detail && e.detail.view === "monitor";
      if (isVisible === wasVisible) return;
      applyPolling();
      // Coming back to the tab should show current numbers, not whatever was
      // on screen when you left.
      if (isVisible) refreshAll();
    });

    // A new run: drop the old charts right away and keep re-checking until
    // the new run's log file appears (the trainer needs a moment to start).
    document.addEventListener("studio:training-started", () => {
      staleRun = lastRun;   // charts of this run are the OLD run until a new one shows up
      lastRun = null;
      document.getElementById("monitor-charts").innerHTML = "";
      const empty = document.getElementById("monitor-chart-empty");
      empty.style.display = "block";
      empty.textContent = "New run starting - charts appear once it logs its first steps.";
      let n = 0;
      clearInterval(startWatch);
      startWatch = setInterval(() => {
        refreshAll();
        if (++n >= 30 || lastRun) clearInterval(startWatch);
      }, 4000);
      refreshAll();
    });

    // The user may land directly on Monitor via a deep link or a reload of a
    // session where that tab was selected.
    isVisible = document.getElementById("view-monitor").classList.contains("active");
    applyPolling();
    if (isVisible) refreshAll();
  }

  // The interval runs only when both conditions hold: the user wants
  // auto-refresh AND the tab is actually on screen. Keeping them separate
  // means unchecking the box while hidden takes effect on the next visit
  // rather than being silently reverted.
  function applyPolling() {
    const wanted = document.getElementById("monitor-auto-refresh").checked;
    if (wanted && isVisible) startAutoRefresh();
    else stopAutoRefresh();
  }

  function startAutoRefresh() {
    stopAutoRefresh();
    autoRefreshTimer = setInterval(refreshAll, 6000);
  }

  function stopAutoRefresh() {
    if (autoRefreshTimer) clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }

  async function refreshAll() {
    await Promise.all([refreshStatus(), refreshProgress(), refreshSamples(), refreshCheckpoints(), refreshScalars()]);
  }

  async function refreshStatus() {
    const backendUrl = document.getElementById("backend-url")?.value?.trim() || "http://127.0.0.1:8000";
    const dot = document.getElementById("monitor-status-dot");
    const text = document.getElementById("monitor-status-text");
    try {
      const status = await Api.monitorStatus(backendUrl);
      dot.classList.remove("ok", "bad");
      dot.classList.add(status.training ? "ok" : "bad");
      text.textContent = status.training ? "Training…" : (status.errored ? "Errored" : "Idle");
    } catch {
      dot.classList.remove("ok");
      dot.classList.add("bad");
      text.textContent = "Backend not reachable";
    }
  }

  async function refreshProgress() {
    const backendUrl = document.getElementById("backend-url")?.value?.trim() || "http://127.0.0.1:8000";
    const row = document.getElementById("monitor-progress-row");
    try {
      const p = await Api.monitorProgress(backendUrl);
      // native_backend/server.py is the only backend that reports this
      // (step/total_steps/epoch/loss) - anything else (unreachable, or
      // a backend that just doesn't have this endpoint) just hides the
      // row instead of showing a confusing error for an optional extra.
      if (!p || p.available === false || typeof p.step !== "number") {
        row.style.display = "none";
        return;
      }
      row.style.display = "block";
      const stepsText = p.total_steps ? `Step ${p.step} / ${p.total_steps}` : `Step ${p.step}`;
      document.getElementById("monitor-progress-steps").textContent =
        p.epoch ? `${stepsText} · epoch ${p.epoch}` : stepsText;
      document.getElementById("monitor-progress-loss").textContent =
        typeof p.loss === "number" ? `loss ${p.loss.toFixed(4)}` : "";
      const pct = p.total_steps ? Math.min(100, Math.round((p.step / p.total_steps) * 100)) : 0;
      document.getElementById("monitor-progress-fill").style.width = `${pct}%`;
    } catch {
      row.style.display = "none";
    }
  }

  async function refreshSamples() {
    const outputDir = document.getElementById("monitor-output-dir").value.trim();
    const grid = document.getElementById("monitor-samples-grid");
    const empty = document.getElementById("monitor-samples-empty");
    if (!outputDir) {
      grid.innerHTML = "";
      empty.style.display = "block";
      empty.textContent = "Set an output folder above to see samples here.";
      document.getElementById("monitor-samples-count").textContent = "";
      return;
    }
    try {
      const data = await Api.monitorSamples(outputDir);
      grid.innerHTML = "";
      document.getElementById("monitor-samples-count").textContent = data.samples.length ? `${data.samples.length}` : "";
      empty.style.display = data.samples.length ? "none" : "block";
      // Distinguish "no samples yet" from "that folder isn't there". The
      // second case means the path is wrong or its project was deleted, and
      // an eternally-blank grid gives no hint which.
      empty.textContent = data.output_dir_exists === false
        ? `Output folder not found: ${outputDir}`
        : "No samples yet. Enable sample generation in the Train tab to see them here.";
      data.samples.forEach((s) => {
        grid.appendChild(h("div", { class: "result-card", style: "cursor:zoom-in" }, [
          h("img", {
            src: Api.thumbUrl(s.path, 240), loading: "lazy",
            onclick: () => openLightbox(Api.imageUrl(s.path), s.filename),
          }),
        ]));
      });
    } catch {
      // output dir doesn't exist yet, or no sample folder - treat as empty rather than erroring loudly
      grid.innerHTML = "";
      empty.style.display = "block";
    }
  }

  async function refreshCheckpoints() {
    const outputDir = document.getElementById("monitor-output-dir").value.trim();
    const list = document.getElementById("monitor-checkpoints-list");
    const empty = document.getElementById("monitor-checkpoints-empty");
    if (!outputDir) {
      list.innerHTML = "";
      empty.style.display = "block";
      empty.textContent = "Set an output folder above to see checkpoints here.";
      document.getElementById("monitor-checkpoints-count").textContent = "";
      return;
    }
    try {
      const data = await Api.monitorCheckpoints(outputDir);
      list.innerHTML = "";
      document.getElementById("monitor-checkpoints-count").textContent = data.checkpoints.length ? `${data.checkpoints.length}` : "";
      empty.style.display = data.checkpoints.length ? "none" : "block";
      empty.textContent = data.output_dir_exists === false
        ? `Output folder not found: ${outputDir}`
        : "No checkpoints saved yet.";
      data.checkpoints.forEach((ckpt) => {
        list.appendChild(h("div", { class: "checkpoint-row" }, [
          h("span", { class: "name" }, ckpt.filename),
          h("span", { class: "meta" }, formatBytes(ckpt.size)),
          h("span", { class: "meta" }, formatRelativeTime(ckpt.mtime)),
        ]));
      });
    } catch {
      list.innerHTML = "";
      empty.style.display = "block";
    }
  }

  async function refreshScalars() {
    // Blank log folder = the trainer's default, <output folder>/logs.
    const outDir = document.getElementById("monitor-output-dir").value.trim();
    const loggingDir = document.getElementById("monitor-logging-dir").value.trim()
      || (outDir ? outDir.replace(/[\\/]+$/, "") + "/logs" : "");
    const container = document.getElementById("monitor-charts");
    const emptyMsg = document.getElementById("monitor-chart-empty");
    if (!loggingDir) {
      container.innerHTML = "";
      emptyMsg.style.display = "block";
      emptyMsg.textContent = "Set a logging folder above to see charts here.";
      return;
    }
    try {
      const data = await Api.monitorScalars(loggingDir);
      if (!data.available) {
        container.innerHTML = "";
        emptyMsg.style.display = "block";
        emptyMsg.textContent = data.reason || "No chart data available yet.";
        return;
      }
      if (staleRun && data.run === staleRun) {
        container.innerHTML = "";
        emptyMsg.style.display = "block";
        emptyMsg.textContent = "New run starting - charts appear once it logs its first steps.";
        return;
      }
      staleRun = null;
      emptyMsg.style.display = "none";
      lastRun = data.run || "x";
      container.innerHTML = "";
      Object.entries(data.series).forEach(([tag, points]) => {
        container.appendChild(renderChart(tag, points));
      });
    } catch (err) {
      container.innerHTML = "";
      emptyMsg.style.display = "block";
      emptyMsg.textContent = `Couldn't read logs: ${err.message}`;
    }
  }

  // ------------------------------ tiny dependency-free line chart ------------------------------

  function renderChart(tag, points) {
    const width = 640, height = 150, padding = 8;
    const values = points.map((p) => p.value);
    const steps = points.map((p) => p.step);
    const minV = Math.min(...values), maxV = Math.max(...values);
    const minS = Math.min(...steps), maxS = Math.max(...steps);
    const spanV = maxV - minV || 1;
    const spanS = maxS - minS || 1;

    const toX = (s) => padding + ((s - minS) / spanS) * (width - padding * 2);
    const toY = (v) => height - padding - ((v - minV) / spanV) * (height - padding * 2);

    const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.step).toFixed(1)},${toY(p.value).toFixed(1)}`).join(" ");
    const latest = points[points.length - 1];

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("class", "chart-svg");
    svg.setAttribute("preserveAspectRatio", "none");

    const axis = document.createElementNS(svgNS, "line");
    axis.setAttribute("x1", padding); axis.setAttribute("y1", height - padding);
    axis.setAttribute("x2", width - padding); axis.setAttribute("y2", height - padding);
    axis.setAttribute("class", "axis");
    svg.appendChild(axis);

    const path = document.createElementNS(svgNS, "path");
    path.setAttribute("d", pathD);
    path.setAttribute("class", "line");
    svg.appendChild(path);

    if (latest) {
      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("x", width - padding);
      label.setAttribute("y", padding + 10);
      label.setAttribute("text-anchor", "end");
      label.setAttribute("class", "latest-value");
      label.textContent = `${fmtNum(latest.value)} @ step ${latest.step}`;
      svg.appendChild(label);
    }

    return h("div", { class: "chart-block" }, [
      h("div", { class: "chart-title" }, `${tag}   (y-axis: ${fmtNum(minV)} to ${fmtNum(maxV)}, auto-scaled)`),
      svg,
    ]);
  }

  // ------------------------------ formatting helpers ------------------------------

  // Small values (learning rates ~1e-5) used to print as "0.0000".
  function fmtNum(v) {
    const a = Math.abs(v);
    if (a !== 0 && a < 0.001) return v.toExponential(2);
    return v.toFixed(4);
  }

  function formatBytes(n) {
    if (n > 1e9) return `${(n / 1e9).toFixed(2)} GB`;
    if (n > 1e6) return `${(n / 1e6).toFixed(1)} MB`;
    if (n > 1e3) return `${(n / 1e3).toFixed(0)} KB`;
    return `${n} B`;
  }

  function formatRelativeTime(unixSeconds) {
    const diff = Date.now() / 1000 - unixSeconds;
    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  // Called when something else (the project switcher) repoints the output
  // folder. Setting .value doesn't fire change, so this stands in for the
  // user editing the field. Refreshes only if the tab is on screen - a
  // hidden tab has nothing to show, and it will refresh on the way in.
  function onFolderChanged() {
    if (isVisible) refreshAll();
  }

  return { init, refreshAll, onFolderChanged, isVisible: () => isVisible };
})();
