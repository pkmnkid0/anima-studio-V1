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

  async function tick() {
    let setup = null, proc = null, reachable = false;
    try { setup = await Api.backendSetupStatus(); } catch { /* ignore */ }
    try { proc = await Api.backendProcessStatus(); } catch { /* ignore */ }
    try {
      const r = await Api.backendPing($("backend-url").value.trim());
      reachable = !!r.reachable;
    } catch { /* ignore */ }
    render(setup, proc, reachable);
    timer = setTimeout(tick, document.hidden ? 15000 : 3000);
  }

  function init() {
    $("be-start").addEventListener("click", () => $("backend-launch-start").click());
    $("be-stop").addEventListener("click", () => $("backend-launch-stop").click());
    $("be-check").addEventListener("click", async () => {
      $("backend-check-btn").click();
      if (timer) clearTimeout(timer);
      tick();
    });
    tick();
  }

  return { init };
})();
