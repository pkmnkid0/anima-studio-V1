// Booru Explorer - browse any supported booru like the site itself:
// a thumbnail wall that fills in without searching, a tag sidebar with
// categories, tag wiki + related tags, a post viewer, and one-click import
// into the active dataset.
const Explore = (() => {
  const LS = {
    source: "animaStudio.bx.source", rating: "animaStudio.bx.rating", minScore: "animaStudio.bx.minScore",
    blacklist: "animaStudio.bx.blacklist", saved: "animaStudio.bx.saved",
  };
  const CATS = [["artist", "Artists"], ["copyright", "Copyrights"], ["character", "Characters"], ["general", "General"], ["meta", "Meta"]];

  const st = {
    sources: [], source: "danbooru", order: "random", page: 1, hasMore: false,
    posts: [], visible: [], tags: [], seq: 0, wikiSeq: 0, loaded: false, inited: false, retryHooked: false,
    selected: new Map(), viewer: -1, panel: "", acItems: [], acIndex: -1,
  };
  const $ = (id) => document.getElementById(id);
  let acSeq = 0;   // out-of-order autocomplete responses get dropped
  const srcInfo = () => st.sources.find((s) => s.id === st.source) || { features: {}, name: st.source, base: "", engine: "", engine_label: "" };
  const img = (url) => (url ? `/api/explore/img?url=${encodeURIComponent(url)}` : "");
  // Broken-image safety net: retry through the proxy once, then load the original
  // address directly (no referrer), then show a neutral placeholder.
  function imgFallback(el, original) {
    let step = 0;
    el.addEventListener("error", () => {
      step += 1;
      if (!original) return;
      if (step === 1) el.src = img(original) + "&r=" + Date.now();
      else if (step === 2) { el.referrerPolicy = "no-referrer"; el.src = original; }
      else { el.classList.add("bx-broken"); el.removeAttribute("src"); reportBrokenImage(original); }
    });
    return el;
  }
  // When thumbnails fail, find out why (once per host) and say so instead of leaving a wall of blanks.
  const brokenHosts = new Set();
  async function reportBrokenImage(url) {
    let host = "";
    try { host = new URL(url).hostname; } catch { return; }
    if (brokenHosts.has(host)) return;
    brokenHosts.add(host);
    let msg = `Images from ${host} can't be loaded.`;
    try {
      const r = await api("img/test", { url });
      if (r && r.ok) { brokenHosts.delete(host); return; }     // that image was just a one-off
      if (r && r.message) msg = r.message;
    } catch { /* keep the generic message */ }
    let bar = document.getElementById("bx-imgwarn");
    if (!bar) {
      bar = h("div", { id: "bx-imgwarn", class: "bx-imgwarn" });
      const grid = $("bx-grid");
      grid.parentNode.insertBefore(bar, grid);
    }
    bar.innerHTML = "";
    bar.append(h("span", {}, msg + " "),
      h("button", { class: "btn small", type: "button", onclick: () => { window.scrollTo({ top: 0 }); togglePanel("net"); } }, "Unblock sites…"),
      h("button", { class: "btn small ghost", type: "button", onclick: () => { bar.remove(); brokenHosts.clear(); } }, "Dismiss"));
  }
  const readJSON = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } };
  const fmt = (n) => (n == null ? "" : n >= 1000 ? `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k` : String(n));
  const pretty = (t) => String(t).replace(/_/g, " ");
  const setChips = (order) => $("bx-modes").querySelectorAll(".bx-chip").forEach((c) => c.classList.toggle("active", c.dataset.order === order));

  async function getJSON(url, opts = {}, timeoutMs = 45000) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    let resp;
    try { resp = await fetch(url, { ...opts, signal: ctl.signal }); }
    catch (e) {
      throw new Error(e.name === "AbortError"
        ? "The request timed out - the site (or your connection to it) is too slow right now."
        : `Can't reach the Anima Studio server: ${e.message}`);
    } finally { clearTimeout(timer); }
    let data = null;
    try { data = await resp.json(); } catch { /* ignore */ }
    if (!resp.ok) throw new Error((data && data.error) || `HTTP ${resp.status}`);
    return data;
  }
  const api = (path, params) => getJSON(`/api/explore/${path}?${new URLSearchParams(params).toString()}`);

  // ------------------------------------------------------------ init
  async function init() {
    if (st.inited) return;
    st.inited = true;
    // Registered before the fetch below: if the source list fails to load,
    // entering the tab again retries the whole init instead of leaving the
    // tab dead until a full page reload.
    if (!st.retryHooked) {
      st.retryHooked = true;
      document.addEventListener("studio:viewchange", (e) => {
        if (!e.detail || e.detail.view !== "explore") return;
        if (!st.inited) init();
        else ensureLoaded();
      });
    }
    try {
      st.sources = (await getJSON("/api/explore/sources")).sources;
    } catch (e) {
      st.inited = false;
      $("bx-status").textContent = `Couldn't load source list: ${e.message}`;
      $("bx-status").classList.add("error");
      return;
    }
    st.source = localStorage.getItem(LS.source) || "danbooru";
    if (!st.sources.some((s) => s.id === st.source)) st.source = "danbooru";
    buildSourceSelect();
    $("bx-source").value = st.source;
    $("bx-rating").value = localStorage.getItem(LS.rating) || "any";
    const ms = localStorage.getItem(LS.minScore);
    if (ms !== null) $("bx-minscore").value = ms;
    applySourceUI();
    bind();
    if ($("view-explore").classList.contains("active")) ensureLoaded();
  }

  function ensureLoaded() {
    if (!st.loaded && st.inited && st.sources.length) { st.loaded = true; run(1); }
  }

  function buildSourceSelect() {
    const sel = $("bx-source");
    sel.innerHTML = "";
    const groups = {};
    st.sources.forEach((s) => { (groups[s.engine_label] = groups[s.engine_label] || []).push(s); });
    Object.entries(groups).forEach(([label, list]) => {
      const og = h("optgroup", { label });
      list.forEach((s) => og.appendChild(h("option", { value: s.id }, s.name + (s.nsfw ? "  (18+ possible)" : ""))));
      sel.appendChild(og);
    });
  }

  function applySourceUI() {
    const s = srcInfo();
    $("bx-pools-btn").style.display = s.features.pools ? "" : "none";
    $("bx-tags").title = s.engine === "philomena" ? "Tags use spaces on this site; typing underscores works too." : "";
    updateSiteLink();
  }

  function updateSiteLink() {
    const s = srcInfo();
    const q = encodeURIComponent($("bx-tags").value.trim());
    let url = s.base;
    if (s.engine === "danbooru" || s.engine === "e621") url = `${s.base}/posts?tags=${q}`;
    else if (s.engine === "gelbooru") url = `${s.base}/index.php?page=post&s=list&tags=${q}`;
    else if (s.engine === "moebooru") url = `${s.base}/post?tags=${q}`;
    else if (s.engine === "philomena") url = `${s.base}/search?q=${q}`;
    const a = $("bx-site-link");
    a.href = url;
    a.textContent = `Open on ${s.name} ↗`;
  }

  function bind() {
    $("bx-source").addEventListener("change", () => {
      st.source = $("bx-source").value;
      localStorage.setItem(LS.source, st.source);
      applySourceUI(); closePanel();
      run(1);
    });
    $("bx-rating").addEventListener("change", () => { localStorage.setItem(LS.rating, $("bx-rating").value); run(1); });
    $("bx-minscore").addEventListener("change", () => { localStorage.setItem(LS.minScore, $("bx-minscore").value); run(1); });
    $("bx-search-btn").addEventListener("click", () => { closeAc(); onSearchSubmit(); });
    $("bx-tags").addEventListener("keydown", onTagsKey);
    $("bx-tags").addEventListener("input", debounce(onTagsInput, 200));
    $("bx-tags").addEventListener("input", updateSiteLink);
    $("bx-tags").addEventListener("blur", () => setTimeout(closeAc, 120));
    $("bx-modes").querySelectorAll(".bx-chip").forEach((c) => c.addEventListener("click", () => setOrder(c.dataset.order)));
    $("bx-shuffle").addEventListener("click", () => { if (st.order !== "random") setOrder("random"); else run(1); });
    $("bx-prev").addEventListener("click", () => run(Math.max(1, st.page - 1)));
    $("bx-next").addEventListener("click", () => run(st.page + 1));
    $("bx-blacklist-btn").addEventListener("click", () => togglePanel("blacklist"));
    $("bx-saved-btn").addEventListener("click", () => togglePanel("saved"));
    $("bx-pools-btn").addEventListener("click", () => togglePanel("pools"));
    $("bx-net-btn").addEventListener("click", () => togglePanel("net"));
    $("bx-clear-sel").addEventListener("click", () => { st.selected.clear(); renderGrid(); updateSelbar(); });
    $("bx-browse-dest").addEventListener("click", () => FolderBrowser.open($("bx-dest").value || null, (p) => { $("bx-dest").value = p; }));
    $("bx-add-btn").addEventListener("click", addSelectedToDataset);
    $("bx-v-close").addEventListener("click", closeViewer);
    $("bx-v-prev").addEventListener("click", () => stepViewer(-1));
    $("bx-v-next").addEventListener("click", () => stepViewer(1));
    $("bx-viewer").addEventListener("click", (e) => { if (e.target.id === "bx-viewer") closeViewer(); });
    document.addEventListener("keydown", (e) => {
      if ($("bx-viewer").style.display === "none") return;
      if (e.key === "Escape") closeViewer();
      else if (e.key === "ArrowLeft") stepViewer(-1);
      else if (e.key === "ArrowRight") stepViewer(1);
    });
    // clicks on [[tag]] links inside rendered wiki text
    document.addEventListener("click", (e) => {
      const a = e.target.closest && e.target.closest("a.bx-taglink");
      if (!a) return;
      e.preventDefault();
      closeViewer();
      searchTag(a.dataset.tag);
    });
  }

  // ------------------------------------------------------------ search
  function setOrder(order) {
    st.order = order;
    setChips(order);
    run(1);
  }

  function onSearchSubmit() {
    // A typed search is a "look it up" action: random order makes no sense there.
    if ($("bx-tags").value.trim() && st.order === "random") { st.order = "popular"; setChips("popular"); }
    run(1);
  }

  function searchTag(tag, order) {
    $("bx-tags").value = tag;
    if (order) st.order = order;
    else if (st.order === "random") st.order = "popular";
    setChips(st.order);
    updateSiteLink();
    window.scrollTo({ top: 0 });
    run(1);
  }

  function addToQuery(tag, negate) {
    const inp = $("bx-tags");
    const parts = inp.value.trim().split(/\s+/).filter(Boolean).filter((t) => t.replace(/^-/, "") !== tag);
    parts.push(negate ? `-${tag}` : tag);
    inp.value = parts.join(" ");
    updateSiteLink();
    if (st.order === "random") { st.order = "popular"; setChips("popular"); }
    run(1);
  }

  async function run(page) {
    const seq = ++st.seq;
    st.page = page;
    const tags = $("bx-tags").value.trim();
    const s = srcInfo();
    const status = $("bx-status");
    status.classList.remove("error");
    status.textContent = st.order === "random" && !tags ? `Pulling random posts from ${s.name}…` : `Loading from ${s.name}…`;
    showSkeleton();
    try {
      const data = await api("search", {
        source: st.source, tags, rating: $("bx-rating").value, order: st.order,
        limit: Math.min(s.max_limit || 40, 40), page, min_score: Math.max(0, parseInt($("bx-minscore").value, 10) || 0),
      });
      if (seq !== st.seq) return; // a newer request superseded this one
      st.posts = data.posts; st.hasMore = data.has_more; st.tags = data.tags;
      st.orderNote = data.order_note || "";
      applyBlacklist();
      renderGrid();
      renderSidebar(data.tags);
      renderCharacters(data.characters, tags);
      renderPager();
      const hidden = st.posts.length - st.visible.length;
      const s2 = srcInfo();
      status.textContent = st.posts.length
        ? `${st.visible.length} posts${hidden ? ` · ${hidden} hidden by blacklist` : ""}${data.total ? ` · ${data.total.toLocaleString()} total` : ""} · ${s2.name}${st.orderNote ? ` · ${st.orderNote}` : ""}`
        : "";
      if (!st.visible.length) {
        $("bx-grid").innerHTML = "";
        $("bx-grid").appendChild(h("div", { class: "bx-empty" },
          st.posts.length ? "Everything on this page is hidden by your blacklist."
            : `No posts found${(parseInt($("bx-minscore").value, 10) || 0) > 0 ? " - try lowering Min score" : ""}${$("bx-rating").value === "safe" ? " or widening Rating" : ""}.`));
      }
      updateSiteLink();
      loadTagHeader(tags);
    } catch (e) {
      if (seq !== st.seq) return;
      $("bx-grid").innerHTML = "";
      $("bx-chars").innerHTML = "";
      $("bx-taglists").innerHTML = "";
      $("bx-taghead").innerHTML = "";
      status.textContent = "";
      showLoadError(e.message);
    }
  }

  // Shown instead of an endless spinner when a site can't be reached.
  function showLoadError(message) {
    const g = $("bx-grid");
    g.innerHTML = "";
    const box = h("div", { class: "bx-empty bx-error" });
    box.appendChild(h("div", { style: "font-weight:600;margin-bottom:6px;color:var(--danger)" }, `Couldn't load ${srcInfo().name}`));
    box.appendChild(h("div", { style: "margin-bottom:12px" }, message));
    const result = h("div", { class: "bx-diag" });
    box.appendChild(h("div", { class: "row", style: "display:flex;gap:8px;justify-content:center;flex-wrap:wrap" }, [
      h("button", { class: "btn primary small", type: "button", onclick: () => run(st.page) }, "Retry"),
      h("button", { class: "btn small", type: "button", onclick: () => runDiagnose(result) }, "Check which sites I can reach"),
      h("button", { class: "btn small", type: "button", onclick: () => { window.scrollTo({ top: 0 }); togglePanel("net"); } }, "Unblock sites…"),
    ]));
    box.appendChild(result);
    box.appendChild(proxyRow());
    g.appendChild(box);
  }

  function proxyRow() {
    const inp = h("input", { type: "text", class: "mono", placeholder: "http://127.0.0.1:7890  or  socks5h://127.0.0.1:1080", style: "width:min(420px,90%)" });
    api("diagnose", { sources: "danbooru" }).then((d) => { if (d.proxy) inp.value = d.proxy; }).catch(() => {});
    return h("div", { style: "margin-top:14px;font-size:11.5px;color:var(--text-dim)" }, [
      h("div", { style: "margin-bottom:6px" }, "If your network blocks these sites, a VPN or proxy fixes it. Proxy for all booru traffic (leave empty for none):"),
      inp,
      h("button", { class: "btn small", type: "button", style: "margin-left:6px", onclick: async () => {
        try {
          await getJSON("/api/explore/proxy", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ proxy: inp.value.trim() }) }, 15000);
          toast(inp.value.trim() ? "Proxy saved" : "Proxy cleared", "success");
          run(st.page);
        } catch (e) { toast(e.message, "error"); }
      } }, "Save & retry"),
    ]);
  }

  async function runDiagnose(out) {
    out.innerHTML = "";
    out.appendChild(h("div", { class: "scout-note", style: "margin-top:10px" }, "Testing every site (about 10 seconds)…"));
    try {
      const d = await api("diagnose", {});
      out.innerHTML = "";
      const ok = d.results.filter((r) => r.ok);
      out.appendChild(h("div", { style: "margin:10px 0 6px" }, ok.length
        ? `${ok.length} of ${d.results.length} sites reachable - click one to browse it:`
        : "None of the sites are reachable from this computer. Check your internet, VPN or proxy."));
      out.appendChild(h("div", { class: "bx-related", style: "justify-content:center" }, ok.map((r) =>
        h("button", { class: "bx-chip-tag", type: "button", onclick: () => {
          st.source = r.id; $("bx-source").value = r.id; localStorage.setItem(LS.source, r.id); applySourceUI(); run(1);
        } }, `${r.name} · ${r.ms} ms`))));
      const bad = d.results.filter((r) => !r.ok);
      if (bad.length && ok.length) out.appendChild(h("div", { style: "margin-top:8px;font-size:11px" }, `Unreachable: ${bad.map((r) => r.name).join(", ")}`));
    } catch (e) {
      out.innerHTML = "";
      out.appendChild(h("div", { class: "scout-note" }, e.message));
    }
  }

  function showSkeleton() {
    const g = $("bx-grid");
    g.innerHTML = "";
    for (let i = 0; i < 10; i++) g.appendChild(h("div", { class: "bx-skel", style: `height:${150 + ((i * 53) % 130)}px` }));
  }

  // ------------------------------------------------------------ blacklist
  const blacklist = () => readJSON(LS.blacklist, []);
  function applyBlacklist() {
    const bl = new Set(blacklist().map((t) => t.toLowerCase()));
    st.visible = bl.size ? st.posts.filter((p) => !Object.values(p.tags).flat().some((t) => bl.has(t.toLowerCase()))) : st.posts.slice();
  }

  // ------------------------------------------------------------ grid
  function renderGrid() {
    const g = $("bx-grid");
    g.innerHTML = "";
    st.visible.forEach((p, idx) => {
      const key = `${p.source}:${p.id}`;
      const card = h("a", { class: `bx-card${st.selected.has(key) ? " sel" : ""}`, href: "#", onclick: (e) => { e.preventDefault(); openViewer(idx); } }, [
        imgFallback(h("img", { src: img(p.thumb_url || p.preview_url), loading: "lazy", decoding: "async", alt: `#${p.id}` }), p.thumb_url || p.preview_url),
        h("button", { class: "bx-pick", type: "button", title: "Select for dataset", onclick: (e) => { e.preventDefault(); e.stopPropagation(); toggleSelect(p, card); } }, st.selected.has(key) ? "✓" : "+"),
        h("div", { class: "bx-ov" }, [
          h("span", { class: `rdot ${p.rating}`, title: p.rating }),
          h("span", { class: "grow" }, `#${p.id}`),
          h("span", {}, p.width ? `${p.width}×${p.height}` : ""),
          h("span", {}, `★${p.score ?? 0}`),
        ]),
      ]);
      g.appendChild(card);
    });
  }

  function toggleSelect(p, card) {
    const key = `${p.source}:${p.id}`;
    if (st.selected.has(key)) st.selected.delete(key); else st.selected.set(key, p);
    if (card) {
      card.classList.toggle("sel", st.selected.has(key));
      const b = card.querySelector(".bx-pick");
      if (b) b.textContent = st.selected.has(key) ? "✓" : "+";
    }
    updateSelbar();
  }

  function updateSelbar() {
    const n = st.selected.size;
    $("bx-selbar").style.display = n ? "flex" : "none";
    $("bx-selcount").textContent = `${n} selected`;
    if (n && !$("bx-dest").value) $("bx-dest").value = ActiveDataset.get() || "";
  }

  async function addSelectedToDataset() {
    const items = [...st.selected.values()];
    if (!items.length) return;
    const dest = $("bx-dest").value.trim();
    if (!dest) return toast("Choose a destination folder first (Browse…).", "error");
    const btn = $("bx-add-btn");
    btn.disabled = true; btn.textContent = "Downloading…";
    try {
      const data = await Api.collectDownload({ items, dest_dir: dest });
      const s = data.summary;
      toast(`Added ${s.ok} image${s.ok === 1 ? "" : "s"} to ${data.dest_dir}${s.errored ? ` (${s.errored} failed)` : ""}`, s.errored ? "info" : "success");
      ActiveDataset.set(data.dest_dir);
      st.selected.clear(); renderGrid(); updateSelbar();
    } catch (e) {
      toast(e.message, "error");
    } finally {
      btn.disabled = false; btn.textContent = "Add to dataset";
    }
  }

  function renderPager() {
    const random = st.order === "random";
    $("bx-pager").style.display = random || (st.page === 1 && !st.hasMore) ? "none" : "flex";
    $("bx-prev").disabled = st.page <= 1;
    $("bx-next").disabled = !st.hasMore;
    $("bx-page").textContent = `Page ${st.page}`;
    $("bx-shuffle").textContent = random ? "Shuffle again" : "Shuffle";
  }

  // ------------------------------------------------------------ character strip
  function renderCharacters(chars, query) {
    const box = $("bx-chars");
    box.innerHTML = "";
    const q = new Set(query.split(/\s+/).filter(Boolean));
    const list = (chars || []).filter((c) => !q.has(c.name)).slice(0, 24);
    if (!list.length) return;
    box.appendChild(h("div", { class: "bx-chars-title", style: "display:block" }, "Characters in this feed - click one to explore"));
    list.forEach((c) => box.appendChild(h("button", { class: "bx-char", type: "button", title: pretty(c.name), onclick: () => searchTag(c.name, "popular") }, [
      imgFallback(h("img", { src: img(c.cover), loading: "lazy", decoding: "async", alt: "" }), c.cover),
      h("span", {}, pretty(c.name)),
      h("em", {}, `${c.count} here`),
    ])));
  }

  // ------------------------------------------------------------ sidebar
  function renderSidebar(tags) {
    const box = $("bx-taglists");
    box.innerHTML = "";
    const bl = new Set(blacklist().map((t) => t.toLowerCase()));
    CATS.forEach(([cat, label]) => {
      const rows = tags.filter((t) => t.category === cat && !bl.has(t.name.toLowerCase())).slice(0, cat === "general" ? 30 : 15);
      if (!rows.length) return;
      box.appendChild(h("div", { class: "bx-group-title" }, label));
      rows.forEach((t) => box.appendChild(tagRow(t.name, cat, t.count)));
    });
    if (!box.children.length) box.appendChild(h("div", { class: "scout-note" }, "No tag data for this page."));
  }

  function tagRow(name, cat, count) {
    return h("div", { class: `bx-tag ${cat}` }, [
      h("button", { class: "ta", type: "button", title: "Add to search", onclick: () => addToQuery(name, false) }, "+"),
      h("button", { class: "ta", type: "button", title: "Exclude from search", onclick: () => addToQuery(name, true) }, "−"),
      h("button", { class: "tn", type: "button", title: `Search ${pretty(name)}`, onclick: () => searchTag(name) }, pretty(name)),
      h("button", { class: "ta", type: "button", title: "Tag wiki", onclick: () => showWiki(name) }, "?"),
      count != null ? h("span", { class: "tc" }, `${count}×`) : null,
    ]);
  }

  function singleTag(tags) {
    const parts = tags.split(/\s+/).filter(Boolean);
    if (parts.length !== 1) return "";
    const t = parts[0];
    if (/^[-~]/.test(t) || /^(order|sort|rating|score|id|pool|parent|user|fav|width|height|status|limit|date):/i.test(t)) return "";
    return t;
  }

  function loadTagHeader(tags) {
    const box = $("bx-taghead");
    const tag = singleTag(tags);
    if (tag) return showWiki(tag);
    box.className = "bx-taghead";
    box.innerHTML = "";
    const s = srcInfo();
    const titles = { random: "Random discovery", popular: "Trending", latest: "Latest uploads", score: "Top scored", favorites: "Most favorited" };
    box.appendChild(h("h3", {}, tags ? "Search results" : titles[st.order]));
    box.appendChild(h("div", { class: "bx-meta" }, `${s.name} · ${s.engine_label}`));
    box.appendChild(h("div", { class: "scout-note" }, tags
      ? "Hover a tag below for + (add), − (exclude) and ? (wiki)."
      : "Nothing typed - this is the site's own front page. Click a character card or any tag to dig in."));
    return null;
  }

  async function showWiki(tag) {
    const seq = ++st.wikiSeq;
    const box = $("bx-taghead");
    box.className = "bx-taghead";
    box.innerHTML = "";
    box.appendChild(h("h3", {}, pretty(tag)));
    box.appendChild(h("div", { class: "scout-note" }, "Loading wiki…"));
    let w;
    try {
      w = await api("tag/wiki", { source: st.source, tag });
    } catch (e) {
      if (seq !== st.wikiSeq) return;
      box.innerHTML = "";
      box.appendChild(h("h3", {}, pretty(tag)));
      box.appendChild(h("div", { class: "scout-note" }, `Wiki unavailable: ${e.message}`));
      return;
    }
    if (seq !== st.wikiSeq) return;
    box.innerHTML = "";
    box.appendChild(h("h3", {}, w.title || pretty(tag)));
    const meta = h("div", { class: "bx-meta" });
    if (w.category) meta.appendChild(h("span", { class: `bx-badge ${w.category}` }, w.category));
    if (w.post_count != null) meta.appendChild(h("span", {}, `${w.post_count.toLocaleString()} posts`));
    box.appendChild(meta);
    if (w.other_names && w.other_names.length) box.appendChild(h("div", { class: "bx-aka" }, `Also: ${w.other_names.slice(0, 8).join(", ")}`));
    if (w.body_html) {
      const wiki = h("div", { class: "bx-wiki", html: w.body_html });
      box.appendChild(wiki);
      if (w.body_html.length > 450) {
        box.appendChild(h("div", { class: "bx-links" }, [h("button", { type: "button", onclick: (e) => {
          wiki.classList.toggle("open"); e.target.textContent = wiki.classList.contains("open") ? "Show less" : "Read more";
        } }, "Read more")]));
      }
    } else {
      box.appendChild(h("div", { class: "scout-note" }, w.available ? "No wiki page has been written for this tag yet." : `${srcInfo().name} has no wiki API - use the link below.`));
    }
    (w.extra || []).forEach((x) => {
      if (x.links) {
        box.appendChild(h("div", { class: "bx-links" }, x.links.filter((u) => /^https?:\/\//i.test(u)).map((u) => {
          let host = u; try { host = new URL(u).hostname; } catch { /* keep raw */ }
          return h("a", { href: u, target: "_blank", rel: "noopener noreferrer" }, host);
        })));
      }
      if (x.tags) {
        box.appendChild(h("div", { class: "bx-group-title" }, x.label));
        box.appendChild(h("div", { class: "bx-related" }, x.tags.map((t) => h("button", { class: "bx-chip-tag", type: "button", onclick: () => searchTag(t.replace(/ /g, "_")) }, t))));
      }
    });
    box.appendChild(h("div", { class: "bx-links" }, [
      h("a", { href: w.wiki_url, target: "_blank", rel: "noopener noreferrer" }, "Wiki on site ↗"),
      h("a", { href: w.search_url, target: "_blank", rel: "noopener noreferrer" }, "Search on site ↗"),
    ]));
    if (w.related && w.related.length) {
      box.appendChild(h("div", { class: "bx-group-title" }, "Related tags"));
      box.appendChild(h("div", { class: "bx-related" }, w.related.slice(0, 24).map((r) =>
        h("button", { class: `bx-chip-tag ${r.category || ""}`, type: "button", title: "Add to search", onclick: () => addToQuery(r.name, false) }, pretty(r.name)))));
    }
  }

  // ------------------------------------------------------------ viewer
  function openViewer(idx) {
    st.viewer = idx;
    $("bx-viewer").style.display = "flex";
    document.body.style.overflow = "hidden";
    paintViewer();
  }
  function closeViewer() {
    if (st.viewer < 0 && $("bx-viewer").style.display === "none") return;
    $("bx-viewer").style.display = "none";
    document.body.style.overflow = "";
    st.viewer = -1;
    $("bx-v-stage").innerHTML = "";
  }
  function stepViewer(d) {
    const n = st.viewer + d;
    if (n < 0 || n >= st.visible.length) return;
    st.viewer = n;
    paintViewer();
  }

  function paintViewer() {
    const p = st.visible[st.viewer];
    if (!p) return;
    const stage = $("bx-v-stage");
    stage.className = "bx-v-stage";
    stage.innerHTML = "";
    if (["mp4", "webm"].includes((p.extension || "").toLowerCase())) {
      stage.appendChild(h("video", { src: img(p.file_url), controls: "", autoplay: "", loop: "" }));
    } else {
      const im = h("img", { src: img(p.sample_url || p.file_url), alt: `#${p.id}`, onclick: () => {
        const full = stage.classList.toggle("full");
        im.src = img(full ? p.file_url : (p.sample_url || p.file_url));
      } });
      imgFallback(im, p.sample_url || p.file_url);
      stage.appendChild(im);
    }
    $("bx-v-prev").style.display = st.viewer > 0 ? "" : "none";
    $("bx-v-next").style.display = st.viewer < st.visible.length - 1 ? "" : "none";
    paintViewerSide(p);
    // fresh details (categorized tags, pools, uploader) arrive in the background
    const mine = st.viewer;
    api(`post/${p.id}`, { source: p.source }).then((d) => {
      if (st.viewer !== mine) return;
      Object.assign(p, d.post);
      paintViewerSide(p);
    }).catch(() => { /* the listing data is already on screen */ });
  }

  function paintViewerSide(p) {
    const side = $("bx-v-side");
    side.innerHTML = "";
    const key = `${p.source}:${p.id}`;
    side.appendChild(h("h3", {}, `Post #${p.id}`));
    side.appendChild(h("div", { class: "bx-v-actions" }, [
      h("button", { class: `btn small ${st.selected.has(key) ? "primary" : ""}`, type: "button", onclick: () => {
        toggleSelect(p, null); renderGrid(); paintViewerSide(p);
      } }, st.selected.has(key) ? "✓ Selected" : "+ Select for dataset"),
      h("a", { class: "btn small", href: p.post_url, target: "_blank", rel: "noopener noreferrer", style: "text-decoration:none" }, "Open on site ↗"),
      h("a", { class: "btn small", href: img(p.file_url), target: "_blank", style: "text-decoration:none" }, "Full image"),
    ]));
    const kv = h("dl", { class: "bx-kv" });
    const row = (k, v) => { if (v === undefined || v === null || v === "") return; kv.appendChild(h("dt", {}, k)); kv.appendChild(h("dd", {}, v)); };
    row("Rating", p.rating); row("Score", p.score); row("Favorites", p.fav_count);
    row("Size", p.width ? `${p.width}×${p.height} ${p.extension || ""}` : p.extension);
    row("File", p.file_size ? `${(p.file_size / 1048576).toFixed(2)} MB` : "");
    row("Uploaded", String(p.created_at || "").replace("T", " ").slice(0, 16)); row("Uploader", p.uploader);
    if (p.source_url) {
      kv.appendChild(h("dt", {}, "Source"));
      kv.appendChild(h("dd", {}, /^https?:\/\//i.test(p.source_url) ? h("a", { href: p.source_url, target: "_blank", rel: "noopener noreferrer" }, p.source_url) : p.source_url));
    }
    side.appendChild(kv);
    const rel = [];
    const jump = (q) => () => { closeViewer(); searchTag(q, "latest"); };
    if (p.parent_id) rel.push(h("button", { class: "bx-chip-tag", type: "button", onclick: jump(`parent:${p.parent_id}`) }, `Parent #${p.parent_id}`));
    if (p.has_children) rel.push(h("button", { class: "bx-chip-tag", type: "button", onclick: jump(`parent:${p.id}`) }, "Has children"));
    (p.pool_ids || []).forEach((id) => rel.push(h("button", { class: "bx-chip-tag", type: "button", onclick: jump(`pool:${id}`) }, `Pool #${id}`)));
    if (rel.length) { side.appendChild(h("div", { class: "bx-group-title" }, "Related posts")); side.appendChild(h("div", { class: "bx-related" }, rel)); }
    CATS.forEach(([cat, label]) => {
      const tags = (p.tags && p.tags[cat]) || [];
      if (!tags.length) return;
      side.appendChild(h("div", { class: "bx-group-title" }, label));
      tags.forEach((t) => side.appendChild(h("div", { class: `bx-tag ${cat}` }, [
        h("button", { class: "ta", type: "button", title: "Add to search", onclick: () => { closeViewer(); addToQuery(t, false); } }, "+"),
        h("button", { class: "ta", type: "button", title: "Exclude", onclick: () => { closeViewer(); addToQuery(t, true); } }, "−"),
        h("button", { class: "tn", type: "button", onclick: () => { closeViewer(); searchTag(t); } }, pretty(t)),
        h("button", { class: "ta", type: "button", title: "Tag wiki", onclick: () => { closeViewer(); window.scrollTo({ top: 0 }); showWiki(t); } }, "?"),
        // total posts on the site for this tag (artist / copyright / character)
        (p.tag_counts && p.tag_counts[t] != null)
          ? h("span", { class: "tc", title: `${p.tag_counts[t].toLocaleString()} posts on ${srcInfo().name}` }, `${p.tag_counts[t].toLocaleString()} posts`)
          : null,
      ])));
    });
  }

  // ------------------------------------------------------------ autocomplete
  function currentToken() {
    const inp = $("bx-tags");
    const pos = inp.selectionStart ?? inp.value.length;
    const before = inp.value.slice(0, pos);
    const start = before.lastIndexOf(" ") + 1;
    let token = before.slice(start);
    const prefix = /^[-~]/.test(token) ? token[0] : "";
    token = token.replace(/^[-~]/, "");
    return { token, start, end: pos, prefix };
  }

  async function onTagsInput() {
    const { token } = currentToken();
    if (token.length < 2) return closeAc();
    const mySource = st.source;
    const seq = ++acSeq;
    try {
      const data = await api("autocomplete", { source: st.source, query: token, limit: 10 });
      if (mySource !== st.source || seq !== acSeq) return;
      renderAc(data.results || []);
    } catch { if (seq === acSeq) closeAc(); }
  }

  function renderAc(items) {
    st.acItems = items; st.acIndex = -1;
    const list = $("bx-autocomplete");
    list.innerHTML = "";
    if (!items.length) return closeAc();
    items.forEach((it, i) => list.appendChild(h("div", { class: "autocomplete-item", onmousedown: (e) => { e.preventDefault(); applyAc(i); } }, [
      h("span", { class: `cat-dot ${it.category || "general"}` }),
      h("span", { class: "name" }, pretty(it.name)),
      h("span", { class: "count" }, fmt(it.post_count)),
    ])));
    list.classList.add("open");
  }
  function closeAc() {
    st.acItems = []; st.acIndex = -1;
    const l = $("bx-autocomplete");
    if (l) { l.classList.remove("open"); l.innerHTML = ""; }
  }
  function applyAc(i) {
    const it = st.acItems[i];
    if (!it) return;
    const inp = $("bx-tags");
    const { start, end, prefix } = currentToken();
    inp.value = `${inp.value.slice(0, start)}${prefix}${it.name} ${inp.value.slice(end)}`.replace(/\s+$/, " ");
    closeAc(); updateSiteLink(); inp.focus();
  }
  function onTagsKey(e) {
    const open = st.acItems.length > 0;
    if (open && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      st.acIndex = (st.acIndex + (e.key === "ArrowDown" ? 1 : -1) + st.acItems.length) % st.acItems.length;
      [...$("bx-autocomplete").children].forEach((c, i) => c.classList.toggle("active", i === st.acIndex));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (open && st.acIndex >= 0) applyAc(st.acIndex); else { closeAc(); onSearchSubmit(); }
    } else if (e.key === "Escape") closeAc();
  }

  // ------------------------------------------------------------ panels
  function closePanel() { st.panel = ""; $("bx-panel").style.display = "none"; }
  function togglePanel(name) {
    if (st.panel === name) return closePanel();
    st.panel = name;
    const p = $("bx-panel");
    p.innerHTML = "";
    p.style.display = "block";
    if (name === "blacklist") panelBlacklist(p);
    else if (name === "saved") panelSaved(p);
    else if (name === "pools") panelPools(p);
    else if (name === "net") panelNet(p);
  }

  // Unblocking: secure DNS mode, proxy/VPN, and a test that shows which part works.
  async function panelNet(p) {
    p.appendChild(h("h4", {}, "Connection - getting past blocked sites"));
    let cfg;
    try { cfg = await api("net", {}); } catch (e) { p.appendChild(h("div", { class: "scout-note" }, e.message)); return; }
    const save = async (body, msg) => {
      try { await getJSON("/api/explore/net", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }, 15000); toast(msg, "success"); }
      catch (e) { toast(e.message, "error"); }
    };
    const modes = [
      ["auto", "Automatic (recommended)", "Normal lookup first; if the site is blocked or answers with a fake address, uses encrypted DNS (Cloudflare / Google / Quad9)."],
      ["secure", "Always use encrypted DNS", "Skips your network's DNS completely for booru sites. Use this if Automatic is slow."],
      ["direct", "Off", "Plain system behaviour, nothing is bypassed."],
    ];
    const box = h("div", { style: "display:flex;flex-direction:column;gap:6px;margin-bottom:12px" });
    modes.forEach(([id, label, desc]) => {
      const r = h("input", { type: "radio", name: "bx-netmode", value: id });
      if (cfg.mode === id) r.checked = true;
      r.addEventListener("change", () => save({ mode: id }, `Connection mode: ${label}`));
      box.appendChild(h("label", { style: "display:flex;gap:8px;align-items:flex-start;font-size:12.5px;cursor:pointer" }, [r, h("span", {}, [h("b", {}, label), h("br"), h("span", { class: "scout-note" }, desc)])]));
    });
    p.appendChild(box);

    const frag = h("input", { type: "checkbox" });
    frag.checked = cfg.fragment !== false;
    frag.addEventListener("change", () => save({ fragment: frag.checked }, frag.checked ? "Site-name unblocking on" : "Site-name unblocking off"));
    p.appendChild(h("label", { style: "display:flex;gap:8px;align-items:flex-start;font-size:12.5px;cursor:pointer;margin-bottom:12px" }, [frag,
      h("span", {}, [h("b", {}, "Also get past blocks on the site name"), h("br"),
        h("span", { class: "scout-note" }, "When a site can't be reached, retries through a built-in local relay that sends the secure-connection hello in small pieces, which filters that look for the site name can't match. Nothing leaves your computer except the normal connection to the site. On by default.")])]));

    const pin = h("input", { type: "text", class: "mono", value: cfg.proxy || "", placeholder: "http://127.0.0.1:7890   or   socks5h://127.0.0.1:1080", style: "width:min(430px,100%)" });
    p.appendChild(h("div", { style: "font-size:12.5px;margin-bottom:4px" }, [h("b", {}, "Proxy / VPN port "), h("span", { class: "scout-note" }, "- for networks that block by address or by site name, which DNS alone can't fix. Run a VPN, Cloudflare WARP (proxy mode) or Tor, and paste its local port here. Leave empty for none.")]));
    p.appendChild(h("div", { class: "row" }, [pin, h("button", { class: "btn small", type: "button", onclick: () => save({ proxy: pin.value.trim() }, pin.value.trim() ? "Proxy saved" : "Proxy cleared") }, "Save proxy")]));

    const out = h("div", { style: "margin-top:12px;font-size:12px" });
    p.appendChild(h("div", { class: "row" }, [
      h("button", { class: "btn primary small", type: "button", onclick: async () => {
        out.innerHTML = ""; out.appendChild(h("div", { class: "scout-note" }, "Testing…"));
        try {
          const d = await api("net/test", { sources: "danbooru,gelbooru,konachan,e621,safebooru" });
          out.innerHTML = "";
          d.results.forEach((r) => out.appendChild(h("div", { style: "padding:2px 0" },
            `${r.host}:  your network ${r.system.ok ? "✓ answers" : "✕ " + (r.system.error || "blocked")}  ·  encrypted DNS ${r.secure.ok ? "✓ works (" + r.secure.provider + ")" : "✕ " + r.secure.error.slice(0, 60)}${r.relay && r.relay.tested ? "  ·  site-name unblocking " + (r.relay.ok ? "✓ gets through" : "✕ " + (r.relay.error || "no")) : ""}`)));
          const bad = d.results.filter((r) => !r.system.ok && r.secure.ok).length;
          const none = d.results.filter((r) => !r.secure.ok).length;
          out.appendChild(h("div", { class: "scout-note", style: "margin-top:8px" },
            none ? "Encrypted DNS can't reach the resolvers from here (or the site doesn't exist) - use a proxy/VPN above."
              : bad ? "Your network's DNS is blocking these; the Automatic mode already routes around it. Click Retry on the feed."
                : "No DNS blocking detected. If sites still fail, the block is by address or site name - use a proxy/VPN above."));
        } catch (e) { out.innerHTML = ""; out.appendChild(h("div", { class: "scout-note" }, e.message)); }
      } }, "Test connection"),
      h("button", { class: "btn small", type: "button", onclick: () => { closePanel(); run(1); } }, "Close & retry feed"),
    ]));
    p.appendChild(out);
  }

  function panelBlacklist(p) {
    p.appendChild(h("h4", {}, "Blacklist - posts with any of these tags are hidden"));
    const ta = h("textarea", { placeholder: "tags separated by spaces or new lines" }, blacklist().join(" "));
    p.appendChild(ta);
    p.appendChild(h("div", { class: "row" }, [
      h("button", { class: "btn primary small", type: "button", onclick: () => {
        const tags = ta.value.split(/\s+/).map((t) => t.trim()).filter(Boolean);
        localStorage.setItem(LS.blacklist, JSON.stringify([...new Set(tags)]));
        applyBlacklist(); renderGrid(); renderSidebar(st.tags);
        $("bx-status").textContent = `${st.visible.length} posts · ${st.posts.length - st.visible.length} hidden by blacklist`;
        toast("Blacklist saved", "success");
      } }, "Save"),
      h("span", { class: "scout-note" }, "Stored in this browser; applies to what's on screen and to every later search."),
    ]));
  }

  function panelSaved(p) {
    p.appendChild(h("h4", {}, "Saved searches"));
    const list = h("div", { class: "bx-list" });
    const render = () => {
      list.innerHTML = "";
      const saved = readJSON(LS.saved, []);
      if (!saved.length) list.appendChild(h("div", { class: "scout-note" }, "Nothing saved yet. Type a search, then click \"Save current\"."));
      saved.forEach((s, i) => list.appendChild(h("div", { class: "bx-list-item", onclick: () => {
        st.source = s.source; $("bx-source").value = s.source; localStorage.setItem(LS.source, s.source);
        $("bx-tags").value = s.tags; $("bx-rating").value = s.rating; applySourceUI();
        st.order = s.order; setChips(s.order);
        closePanel(); run(1);
      } }, [
        h("span", { class: "grow" }, s.tags || "(browse)"),
        h("span", { class: "dim" }, `${(st.sources.find((x) => x.id === s.source) || {}).name || s.source} · ${s.order}`),
        h("button", { class: "btn small", type: "button", onclick: (e) => { e.stopPropagation(); saved.splice(i, 1); localStorage.setItem(LS.saved, JSON.stringify(saved)); render(); } }, "✕"),
      ])));
    };
    p.appendChild(list);
    p.appendChild(h("div", { class: "row" }, [h("button", { class: "btn primary small", type: "button", onclick: () => {
      const saved = readJSON(LS.saved, []);
      saved.unshift({ source: st.source, tags: $("bx-tags").value.trim(), rating: $("bx-rating").value, order: st.order });
      localStorage.setItem(LS.saved, JSON.stringify(saved.slice(0, 50)));
      render();
    } }, "Save current")]));
    render();
  }

  function panelPools(p) {
    p.appendChild(h("h4", {}, `Pools on ${srcInfo().name}`));
    const inp = h("input", { type: "text", placeholder: "Search pool name (empty = recently updated)", class: "mono" });
    const list = h("div", { class: "bx-list", style: "margin-top:8px" });
    const go = async () => {
      list.innerHTML = ""; list.appendChild(h("div", { class: "scout-note" }, "Loading…"));
      try {
        const d = await api("pools", { source: st.source, query: inp.value.trim() });
        list.innerHTML = "";
        if (!d.pools.length) list.appendChild(h("div", { class: "scout-note" }, "No pools found."));
        d.pools.forEach((pl) => list.appendChild(h("div", { class: "bx-list-item", title: pl.description, onclick: () => { closePanel(); searchTag(`pool:${pl.id}`, "latest"); } }, [
          h("span", { class: "grow" }, pl.name), h("span", { class: "dim" }, `${pl.post_count ?? "?"} posts · #${pl.id}`),
        ])));
      } catch (e) { list.innerHTML = ""; list.appendChild(h("div", { class: "scout-note" }, e.message)); }
    };
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    p.appendChild(h("div", { class: "row" }, [inp, h("button", { class: "btn small", type: "button", onclick: go }, "Search")]));
    p.appendChild(list);
    go();
  }

  return { init, searchTag };
})();
window.Explore = Explore;
