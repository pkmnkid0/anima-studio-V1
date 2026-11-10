// Backend tab: one big status hero + 3-step tracker, driven by the same
// endpoints the launcher/installer cards use. The detailed controls live in
// those cards; the hero buttons just forward to them.
const BackendTab = (() => {
  const $ = (id) => document.getElementById(id);
  let timer = null;
  let wasInstalling = false;

  function step(id, state, text) {
    const el = $(`be-step-${id}`);
    if (!el) return;
    el.classList.remove("ok", "busy", "bad");
    if (state) el.classList.add(state);
    $(`be-step-${id}-t`).textContent = text;
  }

  function fmtUp(t) {
    if (!t) return "";
    const s = Math.max(0, Math.floor(Date.now() / 1000 - t));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
  }

  function render(setup, proc, reachable) {
    const installing = setup && setup.stage === "installing";
    const installed = setup && setup.installed;
    const running = proc && proc.running;
    let state, title, sub;

    if (installing) {
      state = "installing"; title = `Installing the backend… ${setup.progress || 0}%`;
      sub = "This downloads PyTorch and friends (several GB) — it can take 10+ minutes. Keep the app open; it starts by itself when done.";
    } else if (setup && setup.stage === "error") {
      state = "error"; title = "Install failed"; sub = setup.error || "See the log below.";
    } else if (reachable) {
      state = "online"; title = "Backend online";
      sub = `${$("backend-url").value.trim()}${running ? " · up " + fmtUp(proc.started_at) : ""}`;
    } else if (running) {
      state = "starting"; title = "Starting…"; sub = "Process is up, waiting for it to answer.";
    } else if (proc && proc.error) {
      state = "error"; title = "Backend stopped"; sub = proc.error;
    } else {
      state = "offline"; title = installed ? "Backend offline" : (setup && setup.partial ? "Install incomplete" : "Not installed yet");
      sub = installed ? "Press Start — it also starts automatically with the app."
        : "Press “Install dependencies” below. It will say clearly when it's finished.";
    }
    $("be-hero").dataset.state = state;
    $("be-title").textContent = title;
    $("be-sub").textContent = sub;
    const pr = $("be-progress");
    pr.style.display = installing ? "block" : "none";
    $("be-progress-fill").style.width = `${setup && setup.progress || 0}%`;
    const lines = (setup && setup.log_tail) || [];
    $("be-now").textContent = installing ? (lines.filter((l) => l.trim()).slice(-1)[0] || "").slice(0, 160) : "";
    if (wasInstalling && !installing) {
      if (installed) { toast("Backend install finished ✓", "success"); }
      else if (setup && setup.stage === "error") { toast("Backend install failed — see the log on the Backend tab", "error"); }
    }
    wasInstalling = installing;
    $("be-start").style.display = running || installing ? "none" : "inline-flex";
    $("be-stop").style.display = running ? "inline-flex" : "none";
    $("be-start").disabled = !installed;

    step("install", installing ? "busy" : installed ? "ok" : (setup && setup.stage === "error" ? "bad" : ""),
      installing ? `${setup.progress || 0}%` : installed ? "Dependencies ready" : "Not installed");
    step("run", running ? "ok" : "", running ? `Up ${fmtUp(proc.started_at)}` : "Not running");
    step("conn", reachable ? "ok" : running ? "busy" : "", reachable ? "Answering requests" : running ? "Waiting…" : "Not connected");

    const dot = $("be-nav-dot");
    if (dot) dot.className = "nav-step be-nav-dot " + (reachable ? "ok" : installing || running ? "busy" : state === "error" ? "bad" : "");
  }

  let ticking = false;
  async function tick() {
    if (ticking) return;          // a slow poll must not stack a second chain
    ticking = true;
    let setup = null, proc = null, reachable = false;
    try { setup = await Api.backendSetupStatus(); } catch { /* ignore */ }
    try { proc = await Api.backendProcessStatus(); } catch { /* ignore */ }
    try {
      const r = await Api.backendPing($("backend-url").value.trim());
      reachable = !!r.reachable;
    } catch { /* ignore */ }
    render(setup, proc, reachable);
    ticking = false;
    timer = setTimeout(tick, document.hidden ? 15000 : 3000);
  }

  // --- Feature dependencies ------------------------------------------------
  // The optional pip sets a few features need (timm tagger models, the
  // vision-language captioner, the AI upscaler). These installs used to live
  // inside the feature tabs themselves; they run from here now so those tabs
  // only show a one-line note when something is missing.
  const FD = {
    tagger: { label: "Auto-tagger", text: "fdeps-tagger-text", btn: "fdeps-tagger-btn",
              spinner: "fdeps-tagger-spinner", log: "fdeps-tagger-log" },
    nl: { label: "Vision-language captions", text: "fdeps-nl-text", btn: "fdeps-nl-btn",
          spinner: "fdeps-nl-spinner", log: "fdeps-nl-log" },
    upscaler: { label: "AI upscaler", text: "fdeps-upscaler-text", btn: "fdeps-upscaler-btn",
                spinner: "fdeps-upscaler-spinner", log: "fdeps-upscaler-log" },
  };
  const fdStart = {
    tagger: () => Api.taggerInstallDeps(),
    nl: () => Api.nlInstallDeps(),
    upscaler: () => Api.upscalerInstallDeps(),
  };
  const fdStatus = {
    tagger: () => Api.taggerInstallDepsStatus(),
    nl: () => Api.nlInstallDepsStatus(),
    upscaler: () => Api.upscalerInstallDepsStatus(),
  };
  const fdActive = new Set();
  let fdTimer = null;

  function fdShow(kind, text, needsInstall) {
    const r = FD[kind];
    $(r.text).textContent = text;
    $(r.btn).style.display = needsInstall ? "inline-flex" : "none";
  }

  async function refreshFeatureDeps() {
    // Auto-tagger: the timm-based models need torch/timm/safetensors in the
    // app's own environment; the ONNX models need nothing at all.
    if (!fdActive.has("tagger")) {
      try {
        const data = await Api.taggerModels();
        const need = (data.models || []).filter((m) => m.needs_deps);
        const missing = need.filter((m) => !m.deps_available).length;
        fdShow("tagger", !need.length ? "not needed"
          : missing ? `missing for ${missing} of ${need.length} timm models (torch, timm, safetensors)`
          : "installed", missing > 0);
      } catch (err) { fdShow("tagger", err.message || "could not check", true); }
    }

    // Captions: the probe itself says which interpreter would run the model.
    if (!fdActive.has("nl")) {
      try {
        const d = await Api.nlDeps(false);
        const bits = [`runs in ${d.label}`];
        if (d.torch) bits.push(`torch ${d.torch.version}${d.torch.cuda ? " (CUDA)" : " (CPU only - slow)"}`);
        if (d.missing && d.missing.length) bits.push(`missing: ${d.missing.join(", ")}`);
        if (d.error) bits.push(d.error);
        fdShow("nl", bits.join(" · "), !d.ok);
      } catch (err) { fdShow("nl", err.message || "could not check", true); }
    }

    // Upscaler: torch + spandrel in the app's own environment.
    if (!fdActive.has("upscaler")) {
      try {
        const s = await Api.upscalerStatus();
        fdShow("upscaler", s.available ? "installed" : "missing torch + spandrel", !s.available);
      } catch (err) { fdShow("upscaler", err.message || "could not check", true); }
    }

    // An install may still be running from before a reload - pick it back up.
    for (const kind of Object.keys(FD)) {
      if (fdActive.has(kind)) continue;
      try {
        const st = await fdStatus[kind]();
        if (st.stage === "installing") { fdActive.add(kind); await renderFdJob(kind, st); }
      } catch { /* ignore */ }
    }
    if (fdActive.size && !fdTimer) fdTimer = setInterval(pollFeatureDeps, 1000);
  }

  async function startFeatureDep(kind) {
    const r = FD[kind];
    $(r.btn).disabled = true;
    $(r.spinner).style.display = "inline-block";
    try {
      fdActive.add(kind);
      await renderFdJob(kind, await fdStart[kind]());
      if (!fdTimer) fdTimer = setInterval(pollFeatureDeps, 1000);
    } catch (err) {
      fdActive.delete(kind);
      toast(err.message, "error");
      $(r.btn).disabled = false;
      $(r.spinner).style.display = "none";
    }
  }

  async function pollFeatureDeps() {
    for (const kind of [...fdActive]) {
      try {
        await renderFdJob(kind, await fdStatus[kind]());
      } catch { /* transient - the next poll retries */ }
    }
  }

  async function renderFdJob(kind, st) {
    const r = FD[kind];
    const log = $(r.log);
    if (st.log_tail && st.log_tail.length) {
      log.style.display = "block";
      log.textContent = st.log_tail.join("\n");
      log.scrollTop = log.scrollHeight;
    }
    if (st.stage === "installing") {
      $(r.btn).disabled = true;
      $(r.spinner).style.display = "inline-block";
      return;
    }
    fdActive.delete(kind);
    if (!fdActive.size && fdTimer) { clearInterval(fdTimer); fdTimer = null; }
    $(r.btn).disabled = false;
    $(r.spinner).style.display = "none";
    if (st.stage === "done") {
      toast(`${r.label} dependencies installed.`, "success");
      if (kind === "nl") await Api.nlDeps(true).catch(() => {});   // refresh the shared probe cache
      refreshFeatureDeps();
    } else if (st.stage === "error") {
      toast(st.error || "Install failed - see the log.", "error");
    }
  }

  function init() {
    $("be-start").addEventListener("click", () => $("backend-launch-start").click());
    $("be-stop").addEventListener("click", () => $("backend-launch-stop").click());
    $("be-check").addEventListener("click", async () => {
      $("backend-check-btn").click();
      if (timer) clearTimeout(timer);
      tick();
    });
    for (const kind of Object.keys(FD)) {
      $(FD[kind].btn).addEventListener("click", () => startFeatureDep(kind));
    }
    // Statuses are fetched when the tab is opened rather than at boot: the
    // caption check spawns a probe process, and the tab may never be visited.
    if ($("view-backend").classList.contains("active")) refreshFeatureDeps();
    document.addEventListener("studio:viewchange", (e) => {
      if (e.detail && e.detail.view === "backend") refreshFeatureDeps();
    });
    tick();
  }

  return { init };
})();
