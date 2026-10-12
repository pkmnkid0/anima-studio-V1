const Theme = (() => {
  const STORAGE_KEY = "animaStudio.theme";
  const DEFAULT_THEME = { bg: "#0a0d0f", accent: "#ce9e5a" };

  const PRESETS = [
    { name: "Deep Studio", bg: "#0a0d0f", accent: "#ce9e5a" },
    { name: "Ember", bg: "#14151b", accent: "#e2593f" },
    { name: "Midnight", bg: "#12141c", accent: "#7c93e6" },
    { name: "Forest", bg: "#121815", accent: "#6fbf8f" },
    { name: "Warm Slate", bg: "#1a1714", accent: "#e0b354" },
    { name: "Ink", bg: "#0f1115", accent: "#9aa5b1" },
  ];

  function clamp255(n) {
    return Math.max(0, Math.min(255, n));
  }

  function hexToRgb(hex) {
    const num = parseInt(hex.replace("#", ""), 16);
    return { r: (num >> 16) & 0xff, g: (num >> 8) & 0xff, b: num & 0xff };
  }

  function rgbToHex(r, g, b) {
    return `#${[r, g, b].map((v) => clamp255(Math.round(v)).toString(16).padStart(2, "0")).join("")}`;
  }

  function shade(hex, amount) {
    // amount in [-255, 255]: positive lightens, negative darkens.
    const { r, g, b } = hexToRgb(hex);
    return rgbToHex(r + amount, g + amount, b + amount);
  }

  function hexToRgba(hex, alpha) {
    const { r, g, b } = hexToRgb(hex);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function isValidHex(value) {
    return /^#[0-9a-fA-F]{6}$/.test(value);
  }

  function apply(bgHex, accentHex) {
    const root = document.documentElement.style;
    root.setProperty("--bg", bgHex);
    root.setProperty("--bg-raised", shade(bgHex, 8));
    root.setProperty("--bg-card", shade(bgHex, 14));
    root.setProperty("--bg-card-hover", shade(bgHex, 22));
    root.setProperty("--bg-input", shade(bgHex, 5));
    root.setProperty("--border", shade(bgHex, 26));
    root.setProperty("--border-soft", shade(bgHex, 16));
    root.setProperty("--border-strong", shade(bgHex, 40));

    root.setProperty("--accent", accentHex);
    root.setProperty("--accent-hover", shade(accentHex, 18));
    root.setProperty("--accent-soft", hexToRgba(accentHex, 0.14));
    root.setProperty("--accent-border", hexToRgba(accentHex, 0.35));
  }

  function save(bgHex, accentHex) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ bg: bgHex, accent: accentHex }));
  }

  function load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return DEFAULT_THEME;
      const parsed = JSON.parse(raw);
      if (isValidHex(parsed.bg) && isValidHex(parsed.accent)) return parsed;
    } catch {
      /* fall through to default */
    }
    return DEFAULT_THEME;
  }

  function applySaved() {
    const { bg, accent } = load();
    apply(bg, accent);
    return { bg, accent };
  }

  function syncInputs(bgHex, accentHex) {
    document.getElementById("theme-bg-color").value = bgHex;
    document.getElementById("theme-bg-hex").value = bgHex;
    document.getElementById("theme-accent-color").value = accentHex;
    document.getElementById("theme-accent-hex").value = accentHex;
  }

  function currentValues() {
    return {
      bg: document.getElementById("theme-bg-color").value,
      accent: document.getElementById("theme-accent-color").value,
    };
  }

  function handleChange() {
    const { bg, accent } = currentValues();
    apply(bg, accent);
    save(bg, accent);
    syncInputs(bg, accent);
  }

  function init() {
    const current = applySaved();
    syncInputs(current.bg, current.accent);

    const presetRow = document.getElementById("theme-presets");
    PRESETS.forEach((preset) => {
      const swatch = h("div", {
        class: "theme-preset",
        title: preset.name,
        style: `background:${preset.bg}`,
        onclick: () => {
          apply(preset.bg, preset.accent);
          save(preset.bg, preset.accent);
          syncInputs(preset.bg, preset.accent);
        },
      }, [h("div", { class: "swatch-accent", style: `background:${preset.accent}` })]);
      presetRow.appendChild(swatch);
    });

    document.getElementById("appearance-btn").addEventListener("click", () => {
      document.getElementById("appearance-modal").classList.add("open");
    });
    document.getElementById("appearance-close-btn").addEventListener("click", () => {
      document.getElementById("appearance-modal").classList.remove("open");
    });
    document.getElementById("appearance-modal").addEventListener("click", (e) => {
      if (e.target.id === "appearance-modal") document.getElementById("appearance-modal").classList.remove("open");
    });

    document.getElementById("theme-bg-color").addEventListener("input", handleChange);
    document.getElementById("theme-accent-color").addEventListener("input", handleChange);
    document.getElementById("theme-bg-hex").addEventListener("change", (e) => {
      if (isValidHex(e.target.value)) { document.getElementById("theme-bg-color").value = e.target.value; handleChange(); }
      else toast("Enter a valid hex color like #14151b", "error");
    });
    document.getElementById("theme-accent-hex").addEventListener("change", (e) => {
      if (isValidHex(e.target.value)) { document.getElementById("theme-accent-color").value = e.target.value; handleChange(); }
      else toast("Enter a valid hex color like #ce9e5a", "error");
    });
    document.getElementById("theme-reset-btn").addEventListener("click", () => {
      apply(DEFAULT_THEME.bg, DEFAULT_THEME.accent);
      save(DEFAULT_THEME.bg, DEFAULT_THEME.accent);
      syncInputs(DEFAULT_THEME.bg, DEFAULT_THEME.accent);
      toast("Appearance reset to default.", "success");
    });
  }

  return { init, applySaved };
})();

// Apply the saved theme immediately (before DOMContentLoaded finishes
// building the rest of the UI) so there's no flash of default colors.
Theme.applySaved();
