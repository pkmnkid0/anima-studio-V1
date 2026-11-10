// Backend tab: an "environment" card - install the trainer's own Python environment, then verify it.
// Used by the MOSS-Image (sd-scripts fork) card and the Low-VRAM trainer card; each passes its element-id prefix and API calls.
const EnvCard = (() => {
  const ICON = { ok: "✓", warn: "!", fail: "✕" };

  function create(cfg) {
    const $ = (id) => document.getElementById(`${cfg.prefix}-${id}`);
    let timer = null;
    let wasInstalling = false;

    function renderVerify(result) {
      const box = $("verify-results");
      box.style.display = "block";
      box.replaceChildren(...result.checks.map((c) => h("div", { class: "verify-row", "data-status": c.status }, [
        h("span", { class: "verify-icon" }, ICON[c.status] || "?"),
        h("div", { class: "verify-body" }, [
          h("b", {}, c.label),
          c.detail ? h("small", {}, c.detail) : null,
          c.fix ? h("small", { class: "verify-fix" }, c.fix) : null,
        ]),
      ])));
    }

    function renderSetup(s) {
      const installing = s.stage === "installing";
      const idx = s.step_index || 0;
      $("setup-btn").disabled = installing || !s.trainer_present;
      $("setup-btn").textContent = s.installed ? cfg.repairLabel : cfg.setupLabel;
      $("setup-cancel").style.display = installing ? "inline-flex" : "none";
      $("spinner").style.display = installing ? "inline-block" : "none";
      const label = (s.steps[idx] || {}).label || "";
      $("setup-text").textContent = installing
        ? `Step ${idx + 1}/${s.steps.length}: ${label}…`
        : s.stage === "error" ? `Failed: ${s.error}`
        : !s.trainer_present ? cfg.missingText
        : s.installed ? "Environment installed." : "Not installed yet.";
      $("setup-track").style.display = installing ? "block" : "none";
      $("setup-fill").style.width = `${Math.round(100 * idx / s.steps.length) + 5}%`;
      const log = $("setup-log");
      const show = installing || s.stage === "error";
      log.style.display = show ? "block" : "none";
      if (show) {
        log.textContent = (s.log_tail || []).join("\n");
        log.scrollTop = log.scrollHeight;
      }
      if (wasInstalling && !installing) {
        if (s.stage === "done") toast(`${cfg.name} environment ready ✓`, "success");
        else if (s.stage === "error") toast(`${cfg.name} setup failed - see the log on the Backend tab`, "error");
      }
      wasInstalling = installing;
    }

    async function tick() {
      try { renderSetup(await cfg.api.status()); } catch { /* backend tab not ready */ }
    }

    function poll() {
      if (timer) clearTimeout(timer);
      const view = document.getElementById("view-backend");
      const visible = !view || view.offsetParent !== null;
      const run = visible ? tick() : Promise.resolve();
      run.finally(() => { timer = setTimeout(poll, wasInstalling ? 1500 : 5000); });
    }

    function init() {
      if (!document.getElementById(`${cfg.prefix}-env-card`)) return;
      $("setup-btn").addEventListener("click", async () => {
        try { renderSetup(await cfg.api.start()); } catch (err) { toast(err.message, "error"); }
      });
      $("setup-cancel").addEventListener("click", async () => {
        try { renderSetup(await cfg.api.cancel()); } catch (err) { toast(err.message, "error"); }
      });
      $("verify-btn").addEventListener("click", async () => {
        $("spinner").style.display = "inline-block";
        try { renderVerify(await cfg.api.verify()); } catch (err) { toast(err.message, "error"); }
        finally { $("spinner").style.display = "none"; }
      });
      poll();
    }

    return { init };
  }

  return { create };
})();
