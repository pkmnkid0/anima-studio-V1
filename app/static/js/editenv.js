// Backend tab: the AI edit tools card (IOPaint, the engine behind lama-cleaner) - environment install and verify.
const EditEnv = EnvCard.create({
  prefix: "editai", name: "AI edit tools", setupLabel: "Set up AI edit environment", repairLabel: "Repair / update the environment",
  missingText: "The bundled edit engine (edit_backend) is missing - redownload Anima Studio.",
  api: { status: () => Api.aiSetupStatus(), start: () => Api.aiSetupStart(), cancel: () => Api.aiSetupCancel(), verify: () => Api.aiVerify() },
});

// Backend tab -> AI edit tools -> "Models": fetch a whole set of models in one go instead of one by one in the Edit tab.
// One row (a set to pick + one button); the single-model list only opens when "Choose models…" is picked.
const EditModels = (() => {
  let cat = null, running = false, timer = null;

  async function init() {
    const box = document.getElementById("editai-models");
    if (!box) return;
    try { cat = (await Api.aiModelsCatalog()).catalog; } catch { return; }
    const picked = new Set();
    const sel = h("select", { style: "min-width:260px" }, [
      ...cat.packs.map((p) => h("option", { value: p.id }, p.label)),
      h("option", { value: "choose" }, "Choose models…"),
    ]);
    const btn = h("button", { class: "btn", type: "button" }, "Download models");
    const stop = h("button", { class: "btn ghost", type: "button", style: "display:none" }, "Stop");
    const text = h("span", { class: "label" }, "");
    const list = h("details", { style: "display:none;margin-top:10px" }, [h("summary", { style: "cursor:pointer;font-size:12px" }, "Models to download")]);
    const key = (i) => `${i.kind}:${i.name}`;
    cat.pick_groups.forEach((g) => {
      const inner = h("div", { style: "display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:2px 12px;margin:4px 0 8px" },
        g.items.map((i) => {
          const cb = h("input", { type: "checkbox" });
          cb.addEventListener("change", () => { cb.checked ? picked.add(key(i)) : picked.delete(key(i)); });
          return h("label", { style: "display:flex;gap:6px;font-size:12px;cursor:pointer;align-items:center" }, [cb, i.label]);
        }));
      list.appendChild(h("div", { style: "font-weight:600;font-size:12px;margin-top:8px" }, g.title + (g.large ? "  (large)" : "")));
      list.appendChild(inner);
    });
    sel.addEventListener("change", () => { list.style.display = sel.value === "choose" ? "block" : "none"; if (sel.value === "choose") list.open = true; });
    box.replaceChildren(
      h("div", { class: "row", style: "gap:8px;flex-wrap:wrap;align-items:center" }, [h("b", { style: "font-size:12.5px" }, "Models"), sel, btn, stop, text]),
      h("div", { class: "progress-track", style: "display:none;margin-top:8px" }, [h("div", { class: "progress-fill", style: "width:0%" })]),
      list,
    );
    const track = box.querySelector(".progress-track"), fill = track.firstChild;

    function show(s) {
      running = s.state === "running";
      btn.disabled = running; stop.style.display = running ? "inline-flex" : "none";
      track.style.display = running ? "block" : "none";
      fill.style.width = `${s.total ? Math.round(100 * s.done / s.total) : 0}%`;
      text.textContent = running ? `${s.done}/${s.total} · ${s.message}` : (s.state === "idle" ? "Models otherwise download the first time you use them." : s.message);
      if (!running && s.failed && s.failed.length) text.title = s.failed.map((f) => `${f.name}: ${f.error}`).join("\n");
    }
    async function poll() {
      clearTimeout(timer);
      try {
        const s = await Api.aiModelsStatus(); show(s);
        if (running) timer = setTimeout(poll, 1500);
      } catch { timer = setTimeout(poll, 5000); }
    }
    btn.addEventListener("click", async () => {
      const body = sel.value === "choose"
        ? { items: [...picked].map((k) => { const [kind, ...n] = k.split(":"); return { kind, name: n.join(":") }; }) }
        : { pack: sel.value };
      try { show(await Api.aiModelsDownload(body)); timer = setTimeout(poll, 1500); } catch (err) { toast(err.message, "error"); }
    });
    stop.addEventListener("click", async () => { try { show(await Api.aiModelsCancel()); } catch (err) { toast(err.message, "error"); } });
    poll();
  }
  return { init };
})();
