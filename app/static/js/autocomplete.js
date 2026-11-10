// Tag autocomplete: a small dropdown under a text input, fed from the tags
// already used in the loaded dataset, most common first.
//
//   Autocomplete.attach(input, { getTags: () => [{tag, count}], onPick: (tag) => ... })
//
// Attach BEFORE adding your own keydown handler: when the menu has a highlighted
// item, Enter/Tab picks it and stops the event so your handler never sees it.
const Autocomplete = (() => {
  let menu = null;
  let owner = null;     // input the menu currently belongs to
  let items = [];
  let index = -1;
  let pick = null;

  function ensureMenu() {
    if (menu) return menu;
    menu = document.createElement("div");
    menu.className = "ac-menu";
    menu.style.display = "none";
    document.body.appendChild(menu);
    return menu;
  }

  function close() {
    if (menu) menu.style.display = "none";
    owner = null; items = []; index = -1; pick = null;
  }

  function paint() {
    menu.innerHTML = "";
    items.forEach((it, i) => {
      const row = document.createElement("div");
      row.className = "ac-item" + (i === index ? " active" : "");
      const name = document.createElement("span");
      name.textContent = it.tag;
      const n = document.createElement("span");
      n.className = "n";
      n.textContent = it.count;
      row.append(name, n);
      row.addEventListener("mousedown", (e) => { e.preventDefault(); choose(i); });
      menu.appendChild(row);
    });
  }

  function place(input) {
    const r = input.getBoundingClientRect();
    menu.style.left = `${Math.round(r.left)}px`;
    menu.style.top = `${Math.round(r.bottom + 2)}px`;
    menu.style.minWidth = `${Math.max(180, Math.round(r.width))}px`;
    menu.style.display = "block";
  }

  function choose(i) {
    const it = items[i];
    const fn = pick;
    close();
    if (it && fn) fn(it.tag);
  }

  function match(query, tags, max) {
    const q = query.trim().toLowerCase().replace(/_/g, " ");
    if (!q) return [];
    const starts = [], contains = [];
    for (const t of tags) {
      const name = t.tag.toLowerCase();
      if (name === q) continue;                // already typed in full
      if (name.startsWith(q)) starts.push(t);
      else if (name.includes(q)) contains.push(t);
      if (starts.length >= max && contains.length >= max) break;
    }
    return starts.concat(contains).slice(0, max);
  }

  function attach(input, { getTags, onPick, max = 8 }) {
    const refresh = () => {
      const found = match(input.value, getTags() || [], max);
      if (!found.length) { if (owner === input) close(); return; }
      ensureMenu();
      owner = input; items = found; index = -1; pick = onPick;
      paint();
      place(input);
    };
    input.addEventListener("input", refresh);
    // Re-check ownership when the timer fires: fast Tab-to-next-input typing
    // may have re-opened this menu (or another) inside the 80 ms grace.
    input.addEventListener("blur", () => { if (owner === input) setTimeout(() => { if (owner === input) close(); }, 80); });
    input.addEventListener("keydown", (e) => {
      if (owner !== input || !items.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        index = (index + (e.key === "ArrowDown" ? 1 : -1) + items.length + 1) % (items.length + 1);
        if (index === items.length) index = -1;
        paint();
      } else if ((e.key === "Enter" || e.key === "Tab") && index >= 0) {
        e.preventDefault();
        e.stopImmediatePropagation();
        choose(index);
      } else if (e.key === "Escape") {
        close();
      }
    });
  }

  return { attach, close };
})();
