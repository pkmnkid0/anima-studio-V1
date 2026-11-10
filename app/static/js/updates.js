// "Updates" block of the App settings window: check GitHub, install the new code, restart.
const Updates = (() => {
  const $ = (id) => document.getElementById(id);
  const h = (tag, attrs = {}, kids = []) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    (Array.isArray(kids) ? kids : [kids]).forEach((c) => e.append(c));
    return e;
  };
  let info = null, poll = null, root = null;

  const set = (id, text) => { const e = $(id); if (e) e.textContent = text; };

  function mount(el) {
    root = el;
    el.append(
      h("div", { style: "font-weight:600;margin-bottom:6px" }, "Updates"),
      h("div", { class: "card-hint", id: "upd-version", style: "margin:0 0 8px" }, ""),
      h("div", { class: "row", style: "gap:6px;align-items:center" }, [
        h("input", { type: "text", id: "upd-repo", class: "mono", style: "flex:1", placeholder: "GitHub repository: owner/name" }),
        h("select", { id: "upd-channel", title: "Where updates come from" }, [
          h("option", { value: "release" }, "Releases"), h("option", { value: "main" }, "Latest on main")]),
      ]),
      h("div", { class: "row", style: "gap:6px;margin-top:6px" }, [
        h("input", { type: "password", id: "upd-token", class: "mono", style: "flex:1", autocomplete: "off",
          placeholder: "Access token (only for a private repository)" }),
      ]),
      h("div", { class: "row", style: "gap:6px;margin-top:8px;align-items:center" }, [
        h("button", { class: "btn small", id: "upd-check", type: "button" }, "Check for updates"),
        h("button", { class: "btn small primary", id: "upd-apply", type: "button", style: "display:none" }, "Update now"),
        h("button", { class: "btn small primary", id: "upd-restart", type: "button", style: "display:none" }, "Restart now"),
        h("label", { class: "checkbox-row", style: "margin:0 0 0 auto" }, [
          h("input", { type: "checkbox", id: "upd-auto" }), h("span", {}, "Check on start")]),
      ]),
      h("div", { id: "upd-msg", class: "card-hint", style: "margin:8px 0 0;white-space:pre-wrap" }, ""),
      h("div", { class: "card-hint", style: "margin:6px 0 0" },
        "Only the app's code is replaced. Your projects, data, models and installed environments are never touched, " +
        "and a backup of the old code is kept in data/update_backups."),
    );
    $("upd-check").addEventListener("click", check);
    $("upd-apply").addEventListener("click", apply);
    $("upd-restart").addEventListener("click", restart);
    $("upd-repo").addEventListener("change", save);
    $("upd-channel").addEventListener("change", save);
    $("upd-token").addEventListener("change", save);
    $("upd-auto").addEventListener("change", save);
    load();
  }

  async function load() {
    try {
      const s = await Api.updateStatus();
      set("upd-version", `Installed version: ${s.version}` + (s.packaged ? "  (packaged app)" : ""));
      $("upd-repo").value = s.settings.repo || "";
      $("upd-channel").value = s.settings.channel;
      $("upd-auto").checked = !!s.settings.auto_check;
      $("upd-token").placeholder = s.settings.has_token ? "Token saved - type to replace" : "Access token (only for a private repository)";
      if (s.progress && ["downloading", "backup", "installing", "deps"].includes(s.progress.state)) watch();
      else if (s.progress && s.progress.state === "done") showProgress(s.progress);
    } catch (err) { set("upd-msg", err.message); }
  }

  async function save() {
    const body = { repo: $("upd-repo").value, channel: $("upd-channel").value, auto_check: $("upd-auto").checked };
    const tok = $("upd-token").value;
    if (tok) body.token = tok;
    try {
      const s = await Api.updateSettings(body);
      $("upd-token").value = "";
      $("upd-token").placeholder = s.has_token ? "Token saved - type to replace" : "Access token (only for a private repository)";
      info = null; $("upd-apply").style.display = "none";
    } catch (err) { set("upd-msg", err.message); }
  }

  async function check(silent) {
    set("upd-msg", "Checking…");
    try {
      await save();
      info = await Api.updateCheck();
      const btn = $("upd-apply");
      if (!info.available) {
        set("upd-msg", `You're up to date (${info.latest || info.installed}).`);
        btn.style.display = "none"; markDot(false);
        return info;
      }
      markDot(true);
      const notes = (info.notes || "").trim();
      set("upd-msg", `New version available: ${info.latest}` + (notes ? `\n\n${notes}` : ""));
      if (info.packaged) {
        set("upd-msg", `New version available: ${info.latest}. This packaged app can't replace itself - download it from ${info.url}`);
        btn.style.display = "none";
      } else { btn.style.display = ""; btn.textContent = `Update to ${info.latest}`; }
      return info;
    } catch (err) { if (!silent) set("upd-msg", err.message); else set("upd-msg", ""); }
  }

  async function apply() {
    if (!info) return;
    $("upd-apply").disabled = true;
    try { await Api.updateApply({ zip: info.zip, latest: info.latest, sha: info.sha }); watch(); }
    catch (err) { set("upd-msg", err.message); $("upd-apply").disabled = false; }
  }

  function showProgress(p) {
    if (p.state === "error") { set("upd-msg", p.error); $("upd-apply").disabled = false; return true; }
    if (p.state === "done") {
      set("upd-msg", p.message + (p.error ? `\n${p.error}` : ""));
      $("upd-apply").style.display = "none"; $("upd-restart").style.display = ""; markDot(false);
      return true;
    }
    set("upd-msg", `${p.message} ${p.percent || 0}%`);
    return false;
  }

  function watch() {
    clearInterval(poll);
    poll = setInterval(async () => {
      try { if (showProgress(await Api.updateProgress())) clearInterval(poll); }
      catch { /* keep polling */ }
    }, 700);
  }

  async function restart() {
    if (!confirm("Restart Anima Studio now? Anything still running (training, captioning) will be stopped.")) return;
    try {
      await Api.updateRestart();
      set("upd-msg", "Restarting… this page reloads when the app is back.");
      const t0 = Date.now();
      const tick = setInterval(async () => {
        try { const r = await fetch("/api/update/status", { cache: "no-store" }); if (r.ok && Date.now() - t0 > 4000) { clearInterval(tick); location.reload(); } }
        catch { /* still down */ }
        if (Date.now() - t0 > 90000) { clearInterval(tick); set("upd-msg", "The app did not come back. Start it again with run.bat / run.sh."); }
      }, 1500);
    } catch (err) { set("upd-msg", err.message); }
  }

  function markDot(on) {
    const b = $("appsettings-btn");
    if (b) b.classList.toggle("has-update", !!on);
  }

  // quiet once-a-day check at startup (only when a repository is set)
  async function autoCheck() {
    try {
      const s = await Api.updateStatus();
      if (!s.settings.auto_check || !s.settings.repo || s.packaged) {
        if (s.last && s.last.available) markDot(true);
        return;
      }
      if (s.last && Date.now() / 1000 - (s.last.at || 0) < 86400) { if (s.last.available) markDot(true); return; }
      const r = await Api.updateCheck();
      markDot(!!r.available);
    } catch { /* offline or no repo: stay quiet */ }
  }

  return { mount, autoCheck };
})();
