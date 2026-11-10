const COLLECT_SOURCES = [
  ["danbooru", "Danbooru"], ["aibooru", "AIBooru"], ["safebooru_donmai", "Safebooru (Donmai)"],
  ["gelbooru", "Gelbooru"], ["safebooru", "Safebooru"], ["rule34", "Rule34"], ["tbib", "TBIB"],
  ["xbooru", "Xbooru"], ["realbooru", "Realbooru"],
  ["konachan", "Konachan.com"], ["konachan_net", "Konachan.net"], ["yandere", "yande.re"],
  ["sakugabooru", "Sakugabooru"],
  ["e621", "e621"], ["e926", "e926"],
  ["derpibooru", "Derpibooru"], ["furbooru", "Furbooru"], ["ponybooru", "Ponybooru"],
];

const Collect = (() => {
  const selectedSources = new Set(["danbooru"]);
  let currentMode = "booru"; // booru | webpage | urls | local
  let results = [];
  const selected = new Set();
  let currentPage = 1;
  let isBooruSearch = false; // whether Prev/Next should re-run a booru search vs. do nothing

  // Booru media sits on CDNs that refuse a hotlink from the app's own page
  // (cdn.donmai.us answers 403, others check the Referer), which is what left
  // the results grid full of broken thumbnails. Everything goes through the
  // app's image proxy instead - the same one the Booru Explorer uses: it sends
  // the site's own Referer and covers the hosts that need the secure-DNS
  // fallback.
  const imgUrl = (url) => (url ? `/api/explore/img?url=${encodeURIComponent(url)}` : "");

  // Broken-image safety net, same as the Explorer's: retry through the proxy,
  // then the original address with no referrer, then say why (once per host)
  // instead of leaving a silent blank.
  const brokenHosts = new Set();
  function imgFallback(el, original) {
    let step = 0;
    el.addEventListener("error", () => {
      step += 1;
      if (!original) return;
      if (step === 1) el.src = imgUrl(original) + "&r=" + Date.now();
      else if (step === 2) { el.referrerPolicy = "no-referrer"; el.src = original; }
      else { el.removeAttribute("src"); explainBrokenImage(original); }
    });
    return el;
  }
  async function explainBrokenImage(url) {
    let host = "";
    try { host = new URL(url).hostname; } catch { return; }
    if (brokenHosts.has(host)) return;
    brokenHosts.add(host);
    let msg = `Images from ${host} can't be loaded.`;
    try {
      const r = await fetch(`/api/explore/img/test?url=${encodeURIComponent(url)}`).then((x) => x.json());
      if (r && r.ok) { brokenHosts.delete(host); return; }     // that image was just a one-off
      if (r && r.message) msg = r.message;
    } catch { /* keep the generic message */ }
    toast(`${msg} Try another source, or set a proxy under Booru Explorer → Connection.`, "error");
  }

  // The lightbox renders an <img>, so a video post (Sakugabooru clips) shows
  // its still frame instead of something that can never render.
  function viewerSrc(item) {
    if (/^(mp4|webm|mkv|mov)$/i.test((item.extension || "").toLowerCase())) {
      return item.preview_url || item.thumb_url || item.sample_url || "";
    }
    return item.file_url || item.sample_url || item.preview_url || "";
  }

  function init() {
    const destInput = document.getElementById("collect-dest-dir");
    if (!destInput.value && ActiveDataset.get()) destInput.value = ActiveDataset.get();
    const localDestInput = document.getElementById("collect-local-dest-dir");
    if (!localDestInput.value && ActiveDataset.get()) localDestInput.value = ActiveDataset.get();

    document.querySelectorAll(".source-tab").forEach((tab) => {
      tab.addEventListener("click", () => selectMode(tab.dataset.source));
    });
    const pillBox = document.getElementById("collect-source-pills");
    COLLECT_SOURCES.forEach(([id, name]) => {
      pillBox.appendChild(h("button", {
        class: `source-pill${selectedSources.has(id) ? " active" : ""}`, "data-source": id, type: "button",
        onclick: () => toggleSourcePill(id),
      }, name));
    });
    document.querySelectorAll(".booru-subtab").forEach((tab) => {
      tab.addEventListener("click", () => selectBooruSubview(tab.dataset.booruSubview));
    });
    document.getElementById("collect-search-btn").addEventListener("click", doSearch);
    document.getElementById("collect-scan-btn").addEventListener("click", doScanPage);
    document.getElementById("collect-urls-btn").addEventListener("click", doFromUrls);
    document.getElementById("collect-select-all").addEventListener("click", () => setAllSelected(true));
    document.getElementById("collect-select-none").addEventListener("click", () => setAllSelected(false));
    document.getElementById("collect-browse-dest").addEventListener("click", () => {
      FolderBrowser.open(destInput.value || null, (path) => {
        destInput.value = path;
      });
    });
    document.getElementById("collect-local-browse-dest").addEventListener("click", () => {
      FolderBrowser.open(localDestInput.value || null, (path) => {
        localDestInput.value = path;
      });
    });
    document.getElementById("collect-local-add-btn").addEventListener("click", doLocalUpload);
    document.getElementById("collect-download-btn").addEventListener("click", () => doDownload());
    document.getElementById("collect-download-all-btn").addEventListener("click", doDownloadAll);
    document.getElementById("collect-prev-btn").addEventListener("click", () => changePage(-1));
    document.getElementById("collect-next-btn").addEventListener("click", () => changePage(1));

    document.getElementById("collect-tags").addEventListener("keydown", onTagsKeydown);
    document.getElementById("collect-tags").addEventListener("input", debounce(onTagsInput, 200));
    document.getElementById("collect-tags").addEventListener("input", updateDanbooruTagHint);
    document.getElementById("collect-rating").addEventListener("change", updateDanbooruTagHint);
    updateDanbooruTagHint();
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".autocomplete-wrap")) closeAutocomplete();
    });

    const discoverBtn=document.getElementById("discover-tags-btn");
    if(discoverBtn) discoverBtn.addEventListener("click", discoverTagOptions);
    const buildBtn=document.getElementById("build-dataset-btn");
    if(buildBtn) buildBtn.addEventListener("click", searchAndCurate);
    renderSelectedFilterTags();
    selectBooruSubview("builder");
  }

  // ------------------------------ merged booru dataset builder ------------------------------
  const selectedFilterTags = new Set();
  const filterGroupTags = new Map();
  let lastDiscoveryKey = "";
  // Cache the last discovered tag groups so toggling a filter
  // re-renders the same list instead of wiping it (which made
  // it impossible to pick more than one filter tag).
  let lastGroups = null;

  // Character / manual tags: every image must have all of these.
  function baseSearchTags() {
    return document.getElementById("collect-tags").value.trim();
  }

  // Smart curation only: ticked filters are alternatives. An image needs the base tags plus ANY
  // ONE of the ticked filters (not all of them at once). One query per filter.
  function searchQueries() {
    const base = baseSearchTags();
    const baseTokens = new Set(base.split(/\s+/).filter(Boolean));
    const filters = [...selectedFilterTags].filter((t) => !baseTokens.has(t));
    return filters.length ? filters.map((t) => `${base} ${t}`.trim()) : [base];
  }

  // Normal Search: every ticked tag is combined (AND), as before.
  function combinedSearchTags() {
    const tokens = new Set(baseSearchTags().split(/\s+/).filter(Boolean));
    [...selectedFilterTags].forEach((t) => tokens.add(t));
    return [...tokens].join(" ").trim();
  }

  // Query the Danbooru 2-tag hint should count: the OR queries in smart mode, otherwise the combined one.
  function composeSearchTags() {
    const smartOn = document.getElementById("smart-enabled")?.checked && selectedSources.size === 1;
    if (!smartOn) return combinedSearchTags();
    return searchQueries().reduce((a, b) => (b.split(/\s+/).length > a.split(/\s+/).length ? b : a), "");
  }

  function renderSelectedFilterTags() {
    const wrap = document.getElementById("selected-filter-tags");
    if (!wrap) return;
    wrap.innerHTML = "";
    if (!selectedFilterTags.size) {
      wrap.innerHTML = '<span class="selected-filter-empty">No optional filters selected.</span>';
      updateDanbooruTagHint();
      return;
    }
    [...selectedFilterTags].forEach((tag) => {
      const chip = h("button", { type:"button", class:"filter-chip", onclick:() => toggleFilterTag(tag, filterGroupTags.get(tag) || "other") }, [
        h("span", {}, tag.replace(/_/g," ")), h("span", { class:"filter-chip-x" }, "×"),
      ]);
      wrap.appendChild(chip);
    });
    updateDanbooruTagHint();
  }

  function optionLabel(tag) { return tag.replace(/_/g, " "); }

  function toggleFilterTag(tag, group) {
    // Every group, including outfit, allows multiple selections.
    if (selectedFilterTags.has(tag)) selectedFilterTags.delete(tag);
    else selectedFilterTags.add(tag);
    renderTagOptionGroups(lastGroups);
    renderSelectedFilterTags();
  }

  function renderTagOptionGroups(groups = null) {
    const root = document.getElementById("tag-option-groups");
    if (!root) return;
    if (!groups) {
      root.innerHTML = '<div class="tag-options-empty">Enter a character and click <b>Discover tags</b> to see tags found on that character.</div>';
      return;
    }
    root.innerHTML = "";
    const order = ["outfit","view","footwear","pose","expression","details","other"];
    const labels = { outfit:"Outfits / Clothing", view:"Framing & View", footwear:"Footwear", pose:"Pose / Action", expression:"Expressions", details:"Details", other:"Other useful tags" };
    let any = false;
    order.forEach((group) => {
      const items = groups[group] || [];
      if (!items.length) return;
      any = true;
      const section = h("div", { class:`tag-option-group tag-group-${group}` });
      const title = h("div", { class:"tag-option-heading" }, [
        h("strong", {}, labels[group]),
        h("span", {}, "Choose any"),
      ]);
      const list = h("div", { class:"tag-option-list" });
      items.forEach((item) => {
        filterGroupTags.set(item.tag, group);
        const active = selectedFilterTags.has(item.tag);
        list.appendChild(h("label", { class:`tag-option${active ? " active" : ""}` }, [
          h("input", { type:"checkbox", checked:active, onchange:() => toggleFilterTag(item.tag, group) }),
          h("span", { class:"tag-option-name" }, optionLabel(item.tag)),
          h("span", { class:"tag-option-count" }, `${item.count}`),
        ]));
      });
      section.appendChild(title); section.appendChild(list); root.appendChild(section);
    });
    if (!any) root.innerHTML='<div class="tag-options-empty">No useful tags were found in the sample. You can still type any booru tag manually.</div>';
  }

  async function discoverTagOptions() {
    const status=document.getElementById("tag-discovery-status");
    const btn=document.getElementById("discover-tags-btn");
    const tags=document.getElementById("collect-tags").value.trim();
    const source=[...selectedSources][0];
    const rating=document.getElementById("collect-rating").value;
    if(!tags) return toast("Enter a character tag first.","error");
    const discoveryKey=`${source}|${rating}|${tags}`;
    if(discoveryKey!==lastDiscoveryKey){
      selectedFilterTags.clear(); filterGroupTags.clear(); renderSelectedFilterTags();
    }
    if(selectedSources.size>1) toast("Tag discovery uses the first selected source; select one source for character-specific tag suggestions.","info");
    btn.disabled=true; btn.textContent="Discovering…";
    status.style.display="block"; status.textContent="Sampling character posts and finding useful tags…";
    try {
      const d=await Api.collectTagOptions({source,tags,rating,sample_pages:3,per_page:100});
      lastDiscoveryKey=discoveryKey;
      lastGroups=d.groups||{};
      renderTagOptionGroups(lastGroups);
      status.textContent=`Found options from ${d.sampled_posts||0} sampled posts. Select an outfit, then add views/footwear/etc. if you want.`;
    } catch(e) {
      status.textContent=e.message||"Could not discover tags.";
      toast(e.message||"Could not discover tags.","error");
    } finally { btn.disabled=false; btn.textContent="Discover tags"; }
  }

  async function searchAndCurate() {
    const smart=document.getElementById("smart-enabled").checked;
    const tags=baseSearchTags();
    const any_tags=[...selectedFilterTags];
    if(!tags) return toast("Enter a character tag first.","error");
    if(smart && selectedSources.size===1) {
      const btn=document.getElementById("build-dataset-btn");
      const status=document.getElementById("smart-dataset-status");
      const target=Math.max(1,Math.min(1000,parseInt(document.getElementById("smart-target").value,10)||100));
      const pool=Math.max(target,Math.min(5000,parseInt(document.getElementById("smart-pool").value,10)||5000));
      const min_resolution=Math.max(0,parseInt(document.getElementById("smart-min-resolution").value,10)||0);
      const source=[...selectedSources][0];
      const rating=document.getElementById("collect-rating").value;
      btn.disabled=true; btn.textContent="Curating…"; status.style.display="block";
      status.textContent=`Searching images with any of ${selectedFilterTags.size||"no"} selected filter${selectedFilterTags.size===1?"":"s"} and selecting ${target.toLocaleString()}…`;
      try {
        const d=await Api.collectSmartDataset({source,tags,any_tags,rating,target,pool,min_resolution});
        setResults(d.results||[]); selected.clear(); (d.results||[]).forEach((_,i)=>selected.add(i)); renderResults();
        const c=d.coverage||{}, a=d.coverage_available||{};
        const labels={back:"Back",side:"Side",full_body:"Full body",footwear:"Footwear",outfit:"Outfits",action:"Actions",expression:"Expressions",detail:"Details"};
        const coverage=Object.keys(labels).map(k=>`${labels[k]} ${c[k]||0}/${a[k]||0}`).join(" . ");
        status.textContent=`Selected ${d.selected_count||0} from ${d.candidate_count||0} usable candidates. ${coverage}`;
        if(!(d.results||[]).length) toast("No usable images matched those filters.","info");
        else toast(`Curated ${d.selected_count} images. Review them before downloading.`,"success");
      } catch(e) { status.textContent=e.message||"Curation failed."; toast(e.message||"Curation failed.","error"); }
      finally { btn.disabled=false; btn.textContent="Search & curate"; }
      return;
    }
    if(smart && selectedSources.size>1) toast("Smart selection needs one source, so this search will use normal gallery mode.","info");
    currentPage=1; isBooruSearch=selectedSources.size===1; await runSearch();
  }

  // ------------------------------ tag autocomplete ------------------------------

  let acItems = [];
  let acIndex = -1;

  function currentToken(input) {
    const value = input.value;
    const upTo = value.slice(0, input.selectionStart ?? value.length);
    const lastSpace = upTo.lastIndexOf(" ");
    const token = upTo.slice(lastSpace + 1);
    return { token: token.replace(/^-/, ""), negated: token.startsWith("-"), start: lastSpace + 1 };
  }

  async function onTagsInput() {
    if (currentMode !== "booru" || selectedSources.size === 0) return closeAutocomplete();
    const input = document.getElementById("collect-tags");
    const { token } = currentToken(input);
    if (token.length < 2) return closeAutocomplete();
    try {
      // Autocomplete against the first selected source - ambiguous to
      // merge suggestions across sites, and this covers the common case
      // of exactly one source anyway.
      const source = [...selectedSources][0];
      const data = await Api.collectAutocomplete(source, token, 8);
      renderAutocomplete(data.results || []);
    } catch {
      closeAutocomplete();
    }
  }

  function formatCount(n) {
    if (n === null || n === undefined) return "";
    if (n >= 1000) return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`;
    return String(n);
  }

  function renderAutocomplete(items) {
    acItems = items;
    acIndex = -1;
    const list = document.getElementById("collect-autocomplete");
    list.innerHTML = "";
    if (!items.length) return closeAutocomplete();
    items.forEach((item, idx) => {
      list.appendChild(h("div", {
        class: "autocomplete-item",
        onmousedown: (e) => { e.preventDefault(); applySuggestion(idx); },
      }, [
        h("span", { class: `cat-dot ${item.category || "general"}` }),
        h("span", { class: "name" }, item.name.replace(/_/g, " ")),
        h("span", { class: "count" }, formatCount(item.post_count)),
      ]));
    });
    list.classList.add("open");
  }

  function closeAutocomplete() {
    acItems = [];
    acIndex = -1;
    const list = document.getElementById("collect-autocomplete");
    if (list) { list.classList.remove("open"); list.innerHTML = ""; }
  }

  function applySuggestion(idx) {
    const item = acItems[idx];
    if (!item) return;
    const input = document.getElementById("collect-tags");
    const { negated, start } = currentToken(input);
    const before = input.value.slice(0, start);
    const after = input.value.slice(input.selectionStart ?? input.value.length);
    const inserted = (negated ? "-" : "") + item.name;
    input.value = `${before}${inserted} ${after}`.replace(/\s+/g, " ").replace(/^\s+/, "");
    const caret = (before + inserted + " ").length;
    input.focus();
    input.setSelectionRange(caret, caret);
    closeAutocomplete();
  }

  function highlightAutocomplete() {
    const list = document.getElementById("collect-autocomplete");
    [...list.children].forEach((el, i) => el.classList.toggle("active", i === acIndex));
  }

  function onTagsKeydown(e) {
    const list = document.getElementById("collect-autocomplete");
    const isOpen = list.classList.contains("open");
    if (isOpen && e.key === "ArrowDown") {
      e.preventDefault();
      acIndex = Math.min(acIndex + 1, acItems.length - 1);
      highlightAutocomplete();
      return;
    }
    if (isOpen && e.key === "ArrowUp") {
      e.preventDefault();
      acIndex = Math.max(acIndex - 1, 0);
      highlightAutocomplete();
      return;
    }
    if (isOpen && e.key === "Escape") {
      closeAutocomplete();
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      if (isOpen && acIndex >= 0) applySuggestion(acIndex);
      else doSearch();
    }
  }

  function selectMode(mode) {
    currentMode = mode;
    closeAutocomplete();
    document.querySelectorAll(".source-tab").forEach((t) => t.classList.toggle("active", t.dataset.source === mode));
    const subnav = document.getElementById("booru-subnav");
    if (subnav) subnav.style.display = mode === "booru" ? "flex" : "none";
    document.getElementById("collect-webpage-fields").style.display = mode === "webpage" ? "grid" : "none";
    document.getElementById("collect-urls-fields").style.display = mode === "urls" ? "grid" : "none";
    document.getElementById("collect-local-fields").style.display = mode === "local" ? "grid" : "none";
    document.querySelectorAll(".booru-subview").forEach((panel) => {
      panel.style.display = mode === "booru" && panel.classList.contains("active") ? "block" : "none";
    });
  }

  function selectBooruSubview(view) {
    document.querySelectorAll(".booru-subtab").forEach((tab) => {
      const active = tab.dataset.booruSubview === view;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll(".booru-subview").forEach((panel) => {
      const active = panel.dataset.booruPanel === view;
      panel.classList.toggle("active", active);
      panel.style.display = active && currentMode === "booru" ? "block" : "none";
    });
  }

  function toggleSourcePill(source) {
    if (selectedSources.has(source)) {
      if (selectedSources.size === 1) return; // keep at least one selected
      selectedSources.delete(source);
    } else {
      selectedSources.add(source);
    }
    document.querySelectorAll(".source-pill").forEach((p) => p.classList.toggle("active", selectedSources.has(p.dataset.source)));
    updateSearchButtonLabel();
    updateDanbooruTagHint();
  }

  function updateDanbooruTagHint() {
    const hint = document.getElementById("danbooru-tag-limit-hint");
    if (!selectedSources.has("danbooru")) { hint.style.display = "none"; return; }

    const userTags = composeSearchTags().split(/\s+/).filter(Boolean).length;
    const ratingAddsTag = document.getElementById("collect-rating").value !== "any" ? 1 : 0;
    const total = userTags + ratingAddsTag;

    if (total > 2) {
      hint.textContent = `Danbooru only lets anonymous search send 2 tags at once - your other `
        + `${total - 2} tag${total - 2 === 1 ? "" : "s"} still apply, just as a filter on each result `
        + `afterward instead of the search itself, so this may take a little longer and return fewer `
        + `results per page than usual.`;
      hint.style.display = "block";
    } else {
      hint.style.display = "none";
    }
  }

  function updateSearchButtonLabel() {
    const btn = document.getElementById("collect-search-btn");
    if (btn.disabled) return; // mid-search - don't clobber the "Searching…" label
    btn.textContent = selectedSources.size > 1 ? `Search & merge (${selectedSources.size})` : "Search";
  }

  async function doSearch() {
    currentPage = 1;
    isBooruSearch = selectedSources.size === 1; // paging a merged multi-source result set isn't well-defined
    await runSearch();
  }

  async function changePage(delta) {
    if (!isBooruSearch) return;
    const nextPage = currentPage + delta;
    if (nextPage < 1) return;
    const previousPage = currentPage;
    currentPage = nextPage;
    const gotResults = await runSearch({ silentEmpty: delta > 0 });
    if (delta > 0 && !gotResults) {
      currentPage = previousPage; // bounced off the end - stay put
      toast("No more results.", "info");
    }
  }

  async function runSearch({ silentEmpty = false } = {}) {
    const queries = [combinedSearchTags()];   // normal gallery search = all tags together
    const rating = document.getElementById("collect-rating").value;
    const limit = parseInt(document.getElementById("collect-limit").value, 10) || 40;
    const sources = [...selectedSources];
    const btn = document.getElementById("collect-search-btn");
    btn.disabled = true;
    btn.textContent = sources.length > 1 ? `Searching ${sources.length} sources…` : "Searching…";

    const outcomes = await Promise.allSettled(
      sources.flatMap((source) => queries.map((tags) => Api.collectSearch({ source, tags, rating, limit, page: currentPage }))),
    );
    const outSources = sources.flatMap((source) => queries.map(() => source));

    const merged = [];
    const seenIds = new Set();
    const failedMessages = [];
    let credentialsNeeded = null;
    outcomes.forEach((outcome, i) => {
      if (outcome.status === "fulfilled") {
        outcome.value.results.forEach((r) => {
          const k = `${r.source}|${r.id}`;
          if (r.id != null && seenIds.has(k)) return;
          seenIds.add(k); merged.push(r);
        });
      }
      else {
        const reason = outcome.reason;
        if (reason && reason.needsCredentials) credentialsNeeded = reason.needsCredentials;
        const detail = reason && reason.message ? reason.message : "unknown error";
        failedMessages.push(sources.length > 1 ? `${outSources[i]}: ${detail}` : detail);
      }
    });

    const gotResults = merged.length > 0;

    // Bouncing off the last page (silentEmpty) must leave the current
    // page on screen - it used to clear the grid and the page indicator
    // before changePage() reverted the page number.
    if (silentEmpty && !gotResults) {
      btn.disabled = false;
      updateSearchButtonLabel();
      return false;
    }

    setResults(merged);
    if (credentialsNeeded) {
      toast(`${credentialsNeeded} needs an API key - click "API keys…" above the sources to add one.`, "error");
    } else if (failedMessages.length) {
      toast(failedMessages.join(" | "), "error");
    }
    if (!gotResults && !failedMessages.length && !silentEmpty) toast("No results for that search.", "info");

    btn.disabled = false;
    updateSearchButtonLabel();
    return gotResults;
  }

  function updatePaginationBar() {
    const bar = document.getElementById("collect-pagination");
    bar.style.display = isBooruSearch ? "flex" : "none";
    document.getElementById("collect-page-indicator").textContent = `Page ${currentPage}`;
    document.getElementById("collect-prev-btn").disabled = currentPage <= 1;
  }

  async function doScanPage() {
    const url = document.getElementById("collect-page-url").value.trim();
    if (!url) return toast("Enter a page URL first.", "error");
    isBooruSearch = false;
    const btn = document.getElementById("collect-scan-btn");
    btn.disabled = true;
    btn.textContent = "Scanning…";
    try {
      const data = await Api.collectScanPage({ url, same_domain_only: true });
      setResults(data.results);
      if (data.results.length === 0) toast("No images found on that page.", "info");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Scan page";
    }
  }

  async function doFromUrls() {
    const raw = document.getElementById("collect-urls").value;
    const urls = raw.split("\n").map((u) => u.trim()).filter(Boolean);
    if (urls.length === 0) return toast("Paste at least one URL.", "error");
    isBooruSearch = false;
    const btn = document.getElementById("collect-urls-btn");
    btn.disabled = true;
    try {
      const data = await Api.collectFromUrls({ urls });
      setResults(data.results);
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  async function doLocalUpload() {
    const destDir = document.getElementById("collect-local-dest-dir").value.trim();
    const fileInput = document.getElementById("collect-local-files");
    const files = fileInput.files;
    if (!destDir) return toast("Choose a destination folder first.", "error");
    if (!files || files.length === 0) return toast("Choose at least one image first.", "error");

    const btn = document.getElementById("collect-local-add-btn");
    const spinner = document.getElementById("collect-local-spinner");
    const statusEl = document.getElementById("collect-local-status");
    btn.disabled = true;
    spinner.style.display = "inline-block";
    statusEl.textContent = `Adding ${files.length} image${files.length === 1 ? "" : "s"}…`;

    const formData = new FormData();
    formData.append("dest_dir", destDir);
    for (const file of files) formData.append("files", file);

    try {
      const resp = await fetch("/api/collect/upload_local", { method: "POST", body: formData });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      const s = data.summary;
      statusEl.textContent = `Done: ${s.ok} added, ${s.errored} failed.`;
      toast(`Added ${s.ok} images to ${data.dest_dir}`, "success");
      ActiveDataset.set(data.dest_dir);
      fileInput.value = "";
    } catch (err) {
      toast(err.message, "error");
      statusEl.textContent = "Add failed.";
    } finally {
      btn.disabled = false;
      spinner.style.display = "none";
    }
  }

  function setResults(list) {
    results = list;
    selected.clear();
    renderResults();
    updatePaginationBar();
  }

  function showPreview(idx) {
    const item = results[idx];
    const tagsText = flattenTagsForDisplay(item.tags);
    const isSelected = selected.has(idx);
    openLightbox(imgUrl(viewerSrc(item)), tagsText, {
      label: isSelected ? "✓ Selected — click to remove" : "Select for download",
      active: isSelected,
      onClick: () => { toggleSelect(idx); showPreview(idx); },
    });
  }

  function renderResults() {
    const card = document.getElementById("collect-results-card");
    const grid = document.getElementById("collect-results-grid");
    grid.innerHTML = "";
    document.getElementById("collect-results-count").textContent = results.length ? `${results.length} found` : "";
    card.style.display = results.length ? "block" : "none";

    results.forEach((item, idx) => {
      const isSelected = selected.has(idx);
      const el = h("div", {
        class: `result-card${isSelected ? " selected" : ""}`,
        onclick: () => toggleSelect(idx),
      }, [
        imgFallback(h("img", { src: imgUrl(item.preview_url || item.thumb_url || item.file_url), loading: "lazy", decoding: "async", alt: "" }),
          item.preview_url || item.thumb_url || item.file_url),
        h("div", { class: "check" }, [
          h("svg", { width: "11", height: "9", viewBox: "0 0 11 9", html: "<path d='M1 4.5L4 7.5L10 1' stroke='white' stroke-width='1.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/>" }),
        ]),
        h("button", {
          class: "preview-btn", type: "button", title: "View & select individually",
          onclick: (e) => { e.stopPropagation(); showPreview(idx); },
        }, [
          h("svg", { width: "12", height: "12", viewBox: "0 0 16 16", fill: "none", html: "<path d='M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5Z' stroke='white' stroke-width='1.3'/><circle cx='8' cy='8' r='2' stroke='white' stroke-width='1.3'/>" }),
        ]),
        h("div", { class: "meta" }, [
          ratingBadge(item.rating),
          item.score !== undefined && item.score !== null
            ? h("span", { class: "score" }, `★${item.score}`)
            : null,
        ]),
      ]);
      grid.appendChild(el);
    });

    updateDownloadCard();
  }

  function flattenTagsForDisplay(tags) {
    if (!tags) return "";
    const order = ["artist", "copyright", "character", "general", "meta"];
    const flat = [];
    for (const cat of order) (tags[cat] || []).forEach((t) => flat.push(t.replace(/_/g, " ")));
    return flat.join(", ");
  }

  function toggleSelect(idx) {
    if (selected.has(idx)) selected.delete(idx);
    else selected.add(idx);
    renderResults();
  }

  function setAllSelected(state) {
    selected.clear();
    if (state) results.forEach((_, idx) => selected.add(idx));
    renderResults();
  }

  function updateDownloadCard() {
    const card = document.getElementById("collect-download-card");
    card.style.display = results.length > 0 ? "block" : "none";
    document.getElementById("collect-selected-count").textContent = selected.size ? `${selected.size} selected` : "";
    document.getElementById("collect-download-btn").disabled = selected.size === 0;
  }

  async function doDownload(items) {
    const destDir = document.getElementById("collect-dest-dir").value.trim();
    if (!destDir) return toast("Choose a destination folder first.", "error");
    items = items || [...selected].map((idx) => results[idx]);
    if (!items.length) return toast("Nothing to download - select at least one image, or use Download all.", "error");
    const btn = document.getElementById("collect-download-btn");
    const allBtn = document.getElementById("collect-download-all-btn");
    const spinner = document.getElementById("collect-download-spinner");
    const statusEl = document.getElementById("collect-download-status");
    btn.disabled = true;
    allBtn.disabled = true;
    spinner.style.display = "inline-block";
    statusEl.textContent = `Downloading ${items.length} images…`;
    try {
      const data = await Api.collectDownload({
        items,
        dest_dir: destDir,
      });
      const s = data.summary;
      statusEl.textContent = `Done: ${s.ok} saved, ${s.skipped} skipped, ${s.errored} failed.`;
      toast(`Saved ${s.ok} images to ${data.dest_dir}`, "success");
      ActiveDataset.set(data.dest_dir);
    } catch (err) {
      toast(err.message, "error");
      statusEl.textContent = "Download failed.";
    } finally {
      btn.disabled = selected.size === 0;
      allBtn.disabled = false;
      spinner.style.display = "none";
    }
  }

  function doDownloadAll() {
    return doDownload(results.slice());
  }

  return { init };
})();
