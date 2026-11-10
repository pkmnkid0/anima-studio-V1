// Magic eraser keybinds: the list of actions with their default keys, the user's changes (saved in the browser and in
// the app's settings file), matching of key events to actions, and the editor shown in App settings.
//
// A key is written "Ctrl+Shift+G": modifiers in the order Ctrl, Alt, Shift, then the key (a letter in capitals, or a
// name such as Enter, Delete, ArrowLeft, Space). Shift is only part of a key for letters and named keys, so symbols
// such as "?" or "+" are written alone, as the keyboard produces them. Each action has two slots.
const Keybinds = (() => {
  const LS_KEY = "anima.keybinds.v1";

  // id, label, group, default keys, hold (true = active while the key is down)
  const ACTIONS = [
    { id: "run", label: "Run the tool / accept the result", group: "General", keys: ["Enter"] },
    { id: "help", label: "Show the shortcut list", group: "General", keys: ["?"] },
    { id: "aiPanel", label: "Show / hide the AI panel", group: "General", keys: ["A"] },
    { id: "prev", label: "Previous picture", group: "General", keys: ["ArrowLeft"] },
    { id: "next", label: "Next picture", group: "General", keys: ["ArrowRight"] },
    { id: "toolBrush", label: "Brush", group: "Tools", keys: ["B"] },
    { id: "toolRect", label: "Rectangle", group: "Tools", keys: ["R"] },
    { id: "toolSelect", label: "Select (Segment Anything)", group: "Tools", keys: ["S"] },
    { id: "toolExtend", label: "Extend", group: "Tools", keys: ["E"] },
    { id: "toolCrop", label: "Crop", group: "Tools", keys: ["C"] },
    { id: "brushSmaller", label: "Smaller brush", group: "Brush and mask", keys: ["["] },
    { id: "brushLarger", label: "Larger brush", group: "Brush and mask", keys: ["]"] },
    { id: "maskGrow", label: "Grow the mask", group: "Brush and mask", keys: ["G"] },
    { id: "maskShrink", label: "Shrink the mask", group: "Brush and mask", keys: ["Shift+G"] },
    { id: "maskInvert", label: "Invert the mask", group: "Brush and mask", keys: ["I"] },
    { id: "clear", label: "Clear the mask", group: "Brush and mask", keys: ["Delete"] },
    { id: "selectToggle", label: "Include / exclude (Select tool)", group: "Brush and mask", keys: ["X"] },
    { id: "undo", label: "Undo", group: "Brush and mask", keys: ["Ctrl+Z"] },
    { id: "redo", label: "Redo", group: "Brush and mask", keys: ["Ctrl+Y", "Ctrl+Shift+Z"] },
    { id: "zoomIn", label: "Zoom in", group: "View", keys: ["+", "="] },
    { id: "zoomOut", label: "Zoom out", group: "View", keys: ["-", "_"] },
    { id: "fit", label: "Fit to window", group: "View", keys: ["0", "F"] },
    { id: "actual", label: "Actual size (100%)", group: "View", keys: ["1"] },
    { id: "pan", label: "Move around (hold, then drag)", group: "View", keys: ["Space"], hold: true },
    { id: "compare", label: "Compare with the original (hold)", group: "View", keys: ["H"], hold: true },
  ];
  const BY_ID = Object.fromEntries(ACTIONS.map((a) => [a.id, a]));
  const MODIFIER_KEYS = new Set(["Control", "Shift", "Alt", "Meta", "AltGraph", "CapsLock", "OS"]);
  // keys that are always reserved: Esc discards / closes and cancels a capture
  const RESERVED = new Set(["Escape"]);

  let custom = {};          // {action: [k1, k2]} - only actions the user changed
  let listeners = [];
  let map = null;           // combo -> action id (rebuilt after a change)

  // ------------------------------------------------------------ key strings

  function baseKey(e) {
    const k = e.key;
    if (k === " ") return "Space";
    if (k.length === 1) return k.toUpperCase();
    return k;
  }

  function comboOf(e) {
    if (MODIFIER_KEYS.has(e.key)) return "";
    const base = baseKey(e);
    const named = base.length > 1;
    const letter = /^[A-Z]$/.test(base);
    const parts = [];
    if (e.ctrlKey || e.metaKey) parts.push("Ctrl");
    if (e.altKey) parts.push("Alt");
    if (e.shiftKey && (letter || named)) parts.push("Shift");
    parts.push(base);
    return parts.join("+");
  }

  // "Ctrl+Shift+G" -> {mods: ["Ctrl","Shift"], key: "G"}; "+" and "Ctrl++" have the plus sign itself as the key.
  function splitCombo(combo) {
    if (!combo) return { mods: [], key: "" };
    if (combo === "+") return { mods: [], key: "+" };
    if (combo.endsWith("++")) return { mods: combo.slice(0, -2).split("+"), key: "+" };
    const parts = combo.split("+");
    return { mods: parts.slice(0, -1), key: parts[parts.length - 1] };
  }

  const PRETTY = { ArrowLeft: "←", ArrowRight: "→", ArrowUp: "↑", ArrowDown: "↓", Delete: "Del", Backspace: "⌫", Escape: "Esc" };
  function pretty(combo) {
    const { mods, key } = splitCombo(combo);
    return combo ? [...mods, PRETTY[key] || key].join("+") : "";
  }

  // ------------------------------------------------------------ the saved keys

  function keysOf(id) {
    const a = BY_ID[id];
    if (!a) return [];
    const k = custom[id] || a.keys;
    return [k[0] || "", k[1] || ""];
  }

  function rebuild() {
    map = new Map();
    for (const a of ACTIONS) for (const k of keysOf(a.id)) if (k && !map.has(k)) map.set(k, a.id);
  }

  function actionFor(e) {
    if (!map) rebuild();
    const c = comboOf(e);
    return c ? map.get(c) || null : null;
  }

  // For hold actions: was this the key being held (ignoring whatever modifiers are down now)?
  function releasedHold(e) {
    const base = baseKey(e);
    for (const a of ACTIONS) {
      if (!a.hold) continue;
      for (const k of keysOf(a.id)) if (k && splitCombo(k).key === base) return a.id;
    }
    return null;
  }

  function label(id) {
    const ks = keysOf(id).filter(Boolean);
    return ks.length ? ks.map(pretty).join(" / ") : "unassigned";
  }

  function first(id) { return pretty(keysOf(id).filter(Boolean)[0] || ""); }

  function notify() { rebuild(); listeners.forEach((fn) => { try { fn(); } catch (err) { console.warn(err); } }); }
  function onChange(fn) { listeners.push(fn); }

  // ------------------------------------------------------------ storing

  function load() {
    try {
      const raw = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
      custom = sanitize(raw);
    } catch { custom = {}; }
    rebuild();
  }

  function sanitize(raw) {
    const out = {};
    if (!raw || typeof raw !== "object") return out;
    for (const [id, keys] of Object.entries(raw)) {
      if (!BY_ID[id] || !Array.isArray(keys)) continue;
      const k = [String(keys[0] || ""), String(keys[1] || "")];
      if (JSON.stringify(k) !== JSON.stringify(padded(BY_ID[id].keys))) out[id] = k;
    }
    return out;
  }

  const padded = (k) => [k[0] || "", k[1] || ""];

  let saveTimer = null;
  function persist() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(custom)); } catch { /* private mode */ }
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      if (typeof Api !== "undefined") Api.appSettingsSaveOptions({ keybinds: custom }).catch(() => {});
    }, 400);
  }

  // The settings file wins over this browser's copy when it has keys (e.g. a fresh browser or another device).
  function hydrate(fromServer) {
    const s = sanitize(fromServer);
    if (Object.keys(s).length) {
      custom = s;
      try { localStorage.setItem(LS_KEY, JSON.stringify(custom)); } catch { /* ignore */ }
      notify();
    } else if (Object.keys(custom).length && typeof Api !== "undefined") {
      Api.appSettingsSaveOptions({ keybinds: custom }).catch(() => {});     // push this browser's keys up
    }
  }

  // Give `combo` to slot `slot` of `id`; a key already used elsewhere is taken from that action.
  function assign(id, slot, combo) {
    const stolen = [];
    if (combo) {
      for (const a of ACTIONS) {
        const ks = keysOf(a.id);
        let changed = false;
        ks.forEach((k, i) => { if (k === combo && !(a.id === id && i === slot)) { ks[i] = ""; changed = true; } });
        if (changed) { custom[a.id] = ks; stolen.push(a.label); }
      }
    }
    const mine = keysOf(id);
    mine[slot] = combo;
    custom[id] = mine;
    custom = sanitize(custom);
    persist();
    notify();
    return stolen;
  }

  function resetAction(id) { delete custom[id]; persist(); notify(); }
  function resetAll() { custom = {}; persist(); notify(); }

  // ------------------------------------------------------------ the editor (App settings)

  let capturing = null;     // {btn, restore}

  function stopCapture() {
    if (!capturing) return;
    window.removeEventListener("keydown", capturing.onKey, true);
    capturing = null;
  }

  function capture(btn, id, slot, rerender) {
    stopCapture();
    const old = btn.textContent;
    btn.textContent = "Press a key…";
    btn.classList.add("capturing");
    const onKey = (e) => {
      if (MODIFIER_KEYS.has(e.key)) return;             // wait for the real key
      e.preventDefault();
      e.stopPropagation();
      stopCapture();
      if (RESERVED.has(e.key)) { btn.textContent = old; btn.classList.remove("capturing"); return; }   // Esc = cancel
      const stolen = assign(id, slot, comboOf(e));
      if (stolen.length && typeof toast === "function") toast(`${pretty(comboOf(e))} moved here from: ${stolen.join(", ")}`, "info");
      rerender();
    };
    capturing = { btn, onKey };
    window.addEventListener("keydown", onKey, true);
  }

  function renderInto(box) {
    if (!box) return;
    const redraw = () => renderInto(box);
    box.innerHTML = "";
    box.appendChild(h("p", { class: "card-hint", style: "margin:0 0 8px" },
      "Click a key, then press the new one (Esc cancels). Each action has two slots; × clears a slot. Changes save by themselves. " +
      "Esc (discard / close), the mouse wheel and middle-drag are fixed."));
    let group = "";
    for (const a of ACTIONS) {
      if (a.group !== group) {
        group = a.group;
        box.appendChild(h("div", { style: "font-weight:600;font-size:12px;margin:10px 0 2px;color:var(--text-dim)" }, group));
      }
      const ks = keysOf(a.id);
      const changed = !!custom[a.id];
      const cell = (slot) => {
        const btn = h("button", { type: "button", class: "btn small keybind-btn" + (ks[slot] ? "" : " empty"), title: "Click, then press a key" },
          ks[slot] ? pretty(ks[slot]) : "—");
        btn.addEventListener("click", () => capture(btn, a.id, slot, redraw));
        const clear = h("button", { type: "button", class: "btn small ghost icon-only", title: "Clear this slot", style: ks[slot] ? "" : "visibility:hidden" }, "×");
        clear.addEventListener("click", () => { assign(a.id, slot, ""); redraw(); });
        return h("span", { class: "row", style: "gap:2px" }, [btn, clear]);
      };
      box.appendChild(h("div", { class: "row keybind-row", style: "gap:8px;padding:3px 0;align-items:center" }, [
        h("span", { style: "flex:1;font-size:12.5px" }, a.label),
        cell(0), cell(1),
        h("button", { type: "button", class: "btn small ghost", style: changed ? "" : "visibility:hidden", title: "Back to the default keys",
          onclick: () => { resetAction(a.id); redraw(); } }, "Reset"),
      ]));
    }
    box.appendChild(h("div", { class: "row", style: "margin-top:12px;gap:8px" }, [
      h("button", { type: "button", class: "btn small", onclick: () => { resetAll(); redraw(); } }, "Reset all to defaults"),
    ]));
  }

  load();
  return { ACTIONS, actionFor, releasedHold, keysOf, label, first, pretty, onChange, hydrate, assign, resetAll, resetAction, renderInto, comboOf, load };
})();
