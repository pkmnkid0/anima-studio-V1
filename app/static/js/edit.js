const Edit = (() => {
  let currentFolder = "";
  let images = [];
  let currentIndex = -1;

  let tool = "brush";
  let brushSize = 28;
  let isDrawing = false;
  let strokes = []; // {type:'brush', points:[{x,y}], size} | {type:'rect', x0,y0,x1,y1}
  let cropRect = null; // {x0,y0,x1,y1} in canvas (natural image) pixels
  let cropDragMode = null; // null | 'move' | 'nw'|'n'|'ne'|'w'|'e'|'sw'|'s'|'se'
  let cropDragStart = null; // {x, y, rect} snapshot taken on pointerdown
  let cropAspect = null; // null (free) or a number like 1, 4/3, 16/9
  const CROP_HANDLE_CSS_SIZE = 11;
  const CROP_MIN_SIZE = 16;
  let baseImage = null;
  let zoom = null; // null = fit to container, otherwise a multiplier like 1.5
  let redoStack = []; // strokes taken back by Undo, restored by Redo
  let preview = null; // an AI result waiting for Accept / Discard: {img, mode: "same"|"scale"|"extend", dx, dy, bg}
  let comparing = false; // holding "Hold to compare": show the original instead of the result
  let pad = { l: 0, t: 0, r: 0, b: 0 }; // Extend tool: pixels added on each side
  let markers = []; // Select tool: [{x, y, include}]
  let liveSel = null; // the selection stroke Segment Anything is currently refining
  const hooks = {}; // set by EditAI: onPreviewCleared, onToolChange, onSelectClick, runInpaint, runExtend, isAiMethod, cvRadius

  // duplicate finder selection state: path -> "keep" | "delete"
  const dupSelection = new Map();

  const maskCanvas = document.createElement("canvas");

  function init() {
    const folderInput = document.getElementById("edit-folder");
    if (!folderInput.value && ActiveDataset.get()) folderInput.value = ActiveDataset.get();
    ActiveDataset.onChange((path) => {
      if (!folderInput.value) folderInput.value = path;
    });

    document.getElementById("edit-browse").addEventListener("click", () => {
      FolderBrowser.open(currentFolder || null, (path) => {
        document.getElementById("edit-folder").value = path;
        load();
      });
    });
    document.getElementById("edit-load").addEventListener("click", load);
    document.getElementById("edit-folder").addEventListener("keydown", (e) => {
      if (e.key === "Enter") load();
    });
    document.getElementById("edit-search").addEventListener("input", debounce(renderGrid, 150));

    SubsetTabs.render("edit-subset-tabs", currentFolder, selectSubsetTab);

    document.querySelectorAll("#eraser-tool-toggle button").forEach((btn) => {
      btn.addEventListener("click", () => selectTool(btn.dataset.tool));
    });
    document.getElementById("eraser-brush-size").addEventListener("input", (e) => setBrushSize(parseInt(e.target.value, 10)));
    document.getElementById("eraser-clear-mask").addEventListener("click", clearStrokes);
    document.getElementById("eraser-undo").addEventListener("click", undoStroke);
    document.getElementById("eraser-redo").addEventListener("click", redoStroke);
    const delModal = document.getElementById("eraser-delete-modal");
    if (delModal) delModal.addEventListener("click", deleteCurrentImage);
    const delBtnGrid = document.getElementById("eraser-delete");
    if (delBtnGrid) delBtnGrid.addEventListener("click", deleteCurrentImage);
    document.getElementById("eraser-close").addEventListener("click", closeEditor);
    document.getElementById("eraser-apply").addEventListener("click", applyErase);
    document.getElementById("eraser-crop-apply").addEventListener("click", applyCrop);
    document.getElementById("eraser-revert").addEventListener("click", revertImage);
    document.getElementById("eraser-prev-overlay").addEventListener("click", () => step(-1));
    document.getElementById("eraser-next-overlay").addEventListener("click", () => step(1));
    document.getElementById("eraser-zoom-in").addEventListener("click", () => setZoom(curScale() * 1.25));
    document.getElementById("eraser-zoom-out").addEventListener("click", () => setZoom(curScale() / 1.25));
    document.getElementById("eraser-zoom-label").addEventListener("click", () => setZoom(null));
    document.getElementById("eraser-zoom-100").addEventListener("click", () => setZoom(1));
    document.getElementById("eraser-keys-btn").addEventListener("click", toggleKeys);
    initViewport();
    restorePrefs();
    refreshKeyHints();
    Keybinds.onChange(refreshKeyHints);
    document.getElementById("eraser-aspect").addEventListener("change", (e) => {
      cropAspect = parseAspect(e.target.value);
      if (cropRect) {
        // re-fit the current rect, anchored top-left, then pull it back
        // inside the canvas so the saved crop matches the chosen aspect
        cropRect = clampRectToCanvas(applyAspectToRect(cropRect, "se"));
        redraw();
      }
    });
    document.getElementById("eraser-crop-to-subset").addEventListener("change", (e) => {
      document.getElementById("eraser-crop-dest-wrap").style.display = e.target.checked ? "flex" : "none";
    });
    document.getElementById("eraser-upscale-method").addEventListener("change", refreshUpscalerStatusIfNeeded);
    document.getElementById("eraser-upscaler-download-btn").addEventListener("click", startUpscalerDownload);
    document.getElementById("eraser-upscaler-deps-goto").addEventListener("click", () => openView("backend"));
    refreshUpscalerStatusIfNeeded();
    // Dependency installs live on the Backend tab now; re-check when the user
    // comes back so the badge clears as soon as they are installed.
    document.addEventListener("studio:viewchange", (e) => {
      if (e.detail && e.detail.view === "edit") refreshUpscalerStatusIfNeeded();
    });
    document.getElementById("resize-start-btn").addEventListener("click", startResizeJob);
    refreshResizeStatus();
    document.getElementById("eraser-crop-dest-browse").addEventListener("click", () => {
      FolderBrowser.open(document.getElementById("eraser-crop-dest-dir").value || null, (path) => {
        document.getElementById("eraser-crop-dest-dir").value = path;
      });
    });

    const canvas = document.getElementById("eraser-canvas");
    canvas.addEventListener("mousedown", onPointerDown);
    canvas.addEventListener("mousemove", updateCursorPosition);
    canvas.addEventListener("mouseenter", () => {
      if (tool === "brush") document.getElementById("eraser-cursor").style.display = "block";
    });
    canvas.addEventListener("mouseleave", () => {
      document.getElementById("eraser-cursor").style.display = "none";
    });
    // mousemove/mouseup are bound on the window, not the canvas: a drag
    // (brush stroke, rect, or crop-handle resize) that moves faster than
    // the canvas bounds - or briefly slips past its edge, which happens
    // constantly when resizing a crop box near the image border - used to
    // stop updating the moment the pointer left the canvas element,
    // making the drag look "stuck" until you released the mouse
    // somewhere else. Tracking on window keeps it continuous.
    window.addEventListener("mousemove", onPointerMove);
    window.addEventListener("mouseup", onPointerUp);
    canvas.addEventListener("touchstart", onPointerDown, { passive: false });
    canvas.addEventListener("touchmove", onPointerMove, { passive: false });
    canvas.addEventListener("touchend", onPointerUp);
    // right-click excludes while Segment Anything is selecting, so the browser menu must not open there
    canvas.addEventListener("contextmenu", (e) => { if (tool === "select") e.preventDefault(); });

    document.getElementById("eraser-modal").addEventListener("click", (e) => {
      if (e.target.id === "eraser-modal") closeEditor();
    });
    const modalEl = document.getElementById("eraser-modal");
    // After a click or a choice in a dropdown, give the keyboard back to the editor (otherwise Enter / B / R keep
    // going to the focused button or list instead of the hotkeys).
    modalEl.addEventListener("click", (e) => { const b = e.target.closest && e.target.closest("button"); if (b && e.detail) b.blur(); });
    modalEl.addEventListener("change", (e) => { if (e.target.matches && e.target.matches("select, input[type=range], input[type=checkbox]")) e.target.blur(); });
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);

    // Duplicate finder
    const thresholdInput = document.getElementById("dup-threshold");
    thresholdInput.addEventListener("input", () => {
      document.getElementById("dup-threshold-value").textContent = thresholdInput.value;
    });
    document.getElementById("dup-scan-btn").addEventListener("click", scanForDuplicates);
  }

  // ------------------------------ folder / grid ------------------------------

  function selectSubsetTab(path) {
    document.getElementById("edit-folder").value = path;
    return load();
  }

  // ------------------------------ AI upscaler (UltraSharp) model download ------------------------------

  let upscalerPollTimer = null;

  async function refreshUpscalerStatusIfNeeded() {
    const method = document.getElementById("eraser-upscale-method").value;
    const row = document.getElementById("eraser-upscaler-status-row");
    if (method !== "ultrasharp") { row.style.display = "none"; return; }
    row.style.display = "flex";
    try {
      applyUpscalerStatus(await Api.upscalerStatus());
    } catch {
      /* the crop request itself will surface a real error if this is unreachable */
    }
  }

  function applyUpscalerStatus(status) {
    const badge = document.getElementById("eraser-upscaler-badge");
    const btn = document.getElementById("eraser-upscaler-download-btn");
    const depsBtn = document.getElementById("eraser-upscaler-deps-goto");
    if (!status.available) {
      badge.textContent = "needs torch + spandrel";
      badge.className = "badge rating-explicit";
      btn.style.display = "none";
      depsBtn.style.display = "inline-flex";
      return;
    }
    depsBtn.style.display = "none";
    if (status.downloaded) {
      badge.textContent = "ready";
      badge.className = "badge rating-safe";
      btn.style.display = "none";
      if (upscalerPollTimer) { clearInterval(upscalerPollTimer); upscalerPollTimer = null; }
      return;
    }
    badge.textContent = "not downloaded";
    badge.className = "badge rating-questionable";
    btn.style.display = "inline-flex";
  }

  async function startUpscalerDownload() {
    try {
      await Api.upscalerDownload();
      if (!upscalerPollTimer) upscalerPollTimer = setInterval(pollUpscalerDownload, 800);
      pollUpscalerDownload();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function pollUpscalerDownload() {
    let status;
    try {
      status = await Api.upscalerDownloadStatus();
    } catch {
      return; // transient - next poll retries
    }
    const track = document.getElementById("eraser-upscaler-progress-track");
    const fill = document.getElementById("eraser-upscaler-progress-fill");
    const text = document.getElementById("eraser-upscaler-progress-text");
    const badge = document.getElementById("eraser-upscaler-badge");

    if (status.status === "downloading") {
      track.style.display = "block";
      const pct = status.total ? Math.min(100, Math.round((status.bytes / status.total) * 100)) : 0;
      fill.style.width = `${pct}%`;
      text.textContent = status.total ? `${pct}%` : "starting…";
      badge.textContent = "downloading";
      badge.className = "badge rating-sensitive";
      return;
    }

    track.style.display = "none";
    text.textContent = "";
    if (status.status === "error") {
      badge.textContent = "download failed";
      badge.className = "badge rating-explicit";
      toast(status.error || "UltraSharp model download failed.", "error");
      if (upscalerPollTimer) { clearInterval(upscalerPollTimer); upscalerPollTimer = null; }
      return;
    }
    if (status.status === "done") {
      applyUpscalerStatus(await Api.upscalerStatus());
    }
  }

  // ------------------------------ Multi-resolution export ------------------------------

  let resizePollTimer = null;

  function collectResolutions() {
    const checked = [...document.querySelectorAll(".resize-res-check:checked")].map((el) => parseInt(el.value, 10));
    const custom = document.getElementById("resize-custom").value.trim();
    if (custom) {
      const val = parseInt(custom, 10);
      if (!Number.isNaN(val) && val > 0) checked.push(val);
      else toast(`Ignoring "${custom}" - not a valid resolution.`, "error");
    }
    return [...new Set(checked)];
  }

  async function startResizeJob() {
    if (!currentFolder) return toast("Load a folder first.", "error");
    const resolutions = collectResolutions();
    if (!resolutions.length) return toast("Pick at least one resolution.", "error");
    try {
      renderResizeStatus(await Api.resizeStart(currentFolder, resolutions));
      toast("Resize job started - see progress below.", "success");
      if (!resizePollTimer) resizePollTimer = setInterval(refreshResizeStatus, 1000);
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function refreshResizeStatus() {
    try {
      renderResizeStatus(await Api.resizeStatus());
    } catch { /* non-critical */ }
  }

  function renderResizeStatus(status) {
    const dot = document.getElementById("resize-dot");
    const text = document.getElementById("resize-status-text");
    const track = document.getElementById("resize-progress-track");
    const fill = document.getElementById("resize-progress-fill");
    const errBox = document.getElementById("resize-errors");
    const btn = document.getElementById("resize-start-btn");

    dot.classList.remove("ok", "bad");
    btn.disabled = status.stage === "running";
    if (status.stage === "running" && !resizePollTimer) {
      // A job started before this page loaded (or a reloaded tab) still
      // needs its progress to tick, not just a static snapshot.
      resizePollTimer = setInterval(refreshResizeStatus, 1000);
    }

    if (status.stage === "running") {
      text.textContent = status.total ? `Resizing ${status.done}/${status.total}…` : "Scanning folder…";
      track.style.display = "block";
      fill.style.width = status.total ? `${Math.round((status.done / status.total) * 100)}%` : "4%";
    } else if (status.stage === "done") {
      dot.classList.add("ok");
      text.textContent = status.total
        ? `Done - ${status.total} image${status.total === 1 ? "" : "s"} processed.`
        : "Done - no images found in that folder.";
      track.style.display = "none";
      if (resizePollTimer) { clearInterval(resizePollTimer); resizePollTimer = null; }
    } else if (status.stage === "error") {
      dot.classList.add("bad");
      text.textContent = "Something went wrong - see below.";
      track.style.display = "none";
      if (resizePollTimer) { clearInterval(resizePollTimer); resizePollTimer = null; }
    } else {
      text.textContent = "Not started";
      track.style.display = "none";
    }

    if (status.errors && status.errors.length) {
      errBox.style.display = "block";
      errBox.textContent = status.errors.join("\n");
    } else {
      errBox.style.display = "none";
    }
  }

  async function load() {
    currentFolder = document.getElementById("edit-folder").value.trim();
    if (!currentFolder) return toast("Enter or browse to a folder first.", "error");
    try {
      const data = await Api.editImages(currentFolder);
      images = data.images;
      document.getElementById("edit-empty").style.display = "none";
      document.getElementById("edit-content").style.display = "block";
      renderGrid();
      document.getElementById("dup-results").innerHTML = "";
      dupSelection.clear();
      ActiveDataset.set(currentFolder);
      SubsetTabs.render("edit-subset-tabs", currentFolder, selectSubsetTab);
      await showNestedHintIfEmpty();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function showNestedHintIfEmpty() {
    const hint = document.getElementById("edit-nested-hint");
    if (!hint) return;
    if (images.length > 0) { hint.style.display = "none"; return; }
    try {
      const data = await Api.browse(currentFolder);
      const count = (data.dirs || []).length;
      if (count > 0) {
        hint.textContent = `This folder has no images directly, but ${count} subfolder${count === 1 ? "" : "s"} below might - pick one from the sidebar.`;
        hint.style.display = "block";
      } else {
        hint.style.display = "none";
      }
    } catch {
      hint.style.display = "none";
    }
  }

  function computeVisibleIndices() {
    const query = (document.getElementById("edit-search").value || "").trim().toLowerCase();
    if (!query) return images.map((_, i) => i);
    const out = [];
    images.forEach((img, i) => { if (img.filename.toLowerCase().includes(query)) out.push(i); });
    return out;
  }

  function renderGrid() {
    const grid = document.getElementById("edit-grid");
    grid.innerHTML = "";
    const indices = computeVisibleIndices();
    const query = (document.getElementById("edit-search").value || "").trim();
    document.getElementById("edit-grid-count").textContent =
      query ? `${indices.length} of ${images.length}` : `${images.length} images`;
    if (images.length === 0) {
      grid.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "No images in this folder.")]));
      return;
    }
    if (indices.length === 0) {
      grid.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "No images match that search.")]));
      return;
    }
    indices.forEach((idx) => {
      const img = images[idx];
      const card = h("div", { class: "caption-card" }, [
        h("div", { class: "thumb-wrap", style: "cursor:pointer", onclick: () => openEditorAt(idx) }, [
          h("img", { src: Api.thumbUrl(img.path, 240), loading: "lazy", decoding: "async" }),
        ]),
        h("div", { class: "body" }, [
          h("div", { class: "filename" }, img.filename),
          img.caption
            ? h("div", { class: "tag-preview", title: img.caption }, img.caption)
            : h("div", { class: "tag-preview empty" }, "No tags yet"),
          h("div", { class: "card-foot" }, [
            img.has_backup
              ? h("span", { class: "badge rating-sensitive" }, "edited")
              : h("span", { class: "badge rating-safe" }, "original"),
            h("button", { class: "btn small ghost", style: "margin-left:auto", onclick: () => openEditorAt(idx) }, "Edit"),
            h("button", { class: "btn small danger", title: "Delete this image and its caption file", onclick: () => deleteOneImage(img) }, "Delete"),
          ]),
        ]),
      ]);
      grid.appendChild(card);
    });
  }

  // ------------------------------ eraser modal ------------------------------

  function updatePrevNextButtons() {
    const indices = computeVisibleIndices();
    const pos = indices.indexOf(currentIndex);
    document.getElementById("eraser-prev-overlay").disabled = pos <= 0;
    document.getElementById("eraser-next-overlay").disabled = pos === -1 || pos >= indices.length - 1;
  }

  function step(delta) {
    const indices = computeVisibleIndices();
    const pos = indices.indexOf(currentIndex);
    if (pos === -1) return;
    const newPos = pos + delta;
    if (newPos < 0 || newPos >= indices.length) return;
    openEditorAt(indices[newPos]);
  }

  function selectTool(newTool) {
    if (preview && newTool !== tool) discardPreview();
    tool = newTool;
    document.getElementById("eraser-canvas").style.cursor = "";      // the crop tool leaves a crosshair / resize cursor behind
    document.querySelectorAll("#eraser-tool-toggle button").forEach((b) => b.classList.toggle("active", b.dataset.tool === tool));
    const wrap = document.getElementById("eraser-canvas-wrap");
    wrap.classList.toggle("tool-rect", tool === "rect");
    wrap.classList.toggle("tool-crop", tool === "crop");
    wrap.classList.toggle("tool-select", tool === "select");
    wrap.classList.toggle("tool-extend", tool === "extend");
    const show = (id, on, display = "flex") => { document.getElementById(id).style.display = on ? display : "none"; };
    document.getElementById("eraser-cursor").style.display = tool === "brush" ? "" : "none";
    show("eraser-brush-size-row", tool === "brush");
    show("eraser-method-row", tool !== "crop");
    show("eraser-aspect-row", tool === "crop");
    show("eraser-upscale-row", tool === "crop");
    show("eraser-crop-dest-row", tool === "crop");
    show("eraser-mask-tools", tool === "brush" || tool === "rect" || tool === "select");
    show("eraser-select-row", tool === "select");
    show("eraser-extend-row", tool === "extend");
    show("eraser-apply", tool !== "crop", "inline-flex");
    show("eraser-crop-apply", tool === "crop", "inline-flex");
    document.getElementById("eraser-apply").textContent = applyLabel();
    const keys = "";
    renderKeybar();
    document.getElementById("eraser-hint").textContent = ({
      brush: "Paint over the area to remove.",
      rect: "Drag rectangles over the area to remove.",
      select: "Click an object to select it (Segment Anything).",
      extend: "Set how much to add on each side and pick a Stable Diffusion method.",
      crop: "Drag the box to choose the crop.",
    }[tool] || "") + keys;
    if (tool === "crop" && !cropRect && baseImage) initCropRect();
    if (hooks.onToolChange) hooks.onToolChange(tool);
    redraw();
  }

  function applyLabel() { return tool === "extend" ? "Extend" : "Erase"; }

  function initCropRect() {
    const canvas = document.getElementById("eraser-canvas");
    const marginX = canvas.width * 0.1;
    const marginY = canvas.height * 0.1;
    cropRect = { x0: marginX, y0: marginY, x1: canvas.width - marginX, y1: canvas.height - marginY };
    if (cropAspect) cropRect = clampRectToCanvas(applyAspectToRect(cropRect, "se"));
  }

  function parseAspect(value) {
    if (value === "free") return null;
    const [w, h] = value.split(":").map(Number);
    return w / h;
  }

  function applyAspectToRect(rect, mode) {
    if (!cropAspect) return rect;
    let { x0, y0, x1, y1 } = rect;
    const w = x1 - x0, h = y1 - y0;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;

    if (mode === "nw" || mode === "ne") {
      y0 = y1 - w / cropAspect;
    } else if (mode === "sw" || mode === "se") {
      y1 = y0 + w / cropAspect;
    } else if (mode === "n" || mode === "s") {
      const newW = h * cropAspect;
      x0 = cx - newW / 2;
      x1 = cx + newW / 2;
    } else if (mode === "w" || mode === "e") {
      const newH = w / cropAspect;
      y0 = cy - newH / 2;
      y1 = cy + newH / 2;
    }
    return { x0, y0, x1, y1 };
  }

  function handleSizeInCanvasUnits() {
    const canvas = document.getElementById("eraser-canvas");
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return CROP_HANDLE_CSS_SIZE;
    return CROP_HANDLE_CSS_SIZE * (canvas.width / rect.width);
  }

  function cropHandlePositions(rect) {
    const { x0, y0, x1, y1 } = rect;
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    return { nw: [x0, y0], n: [mx, y0], ne: [x1, y0], w: [x0, my], e: [x1, my], sw: [x0, y1], s: [mx, y1], se: [x1, y1] };
  }

  function hitTestCrop(pt) {
    if (!cropRect) return null;
    const hs = handleSizeInCanvasUnits();
    const handles = cropHandlePositions(cropRect);
    for (const [name, [hx, hy]] of Object.entries(handles)) {
      if (Math.abs(pt.x - hx) <= hs && Math.abs(pt.y - hy) <= hs) return name;
    }
    const { x0, y0, x1, y1 } = cropRect;
    if (pt.x > x0 && pt.x < x1 && pt.y > y0 && pt.y < y1) return "move";
    return null;
  }

  function cursorForHandle(mode) {
    const map = {
      nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
      n: "ns-resize", s: "ns-resize", w: "ew-resize", e: "ew-resize", move: "move",
    };
    return map[mode] || "crosshair";
  }

  function openEditorAt(idx) {
    if (idx < 0 || idx >= images.length) return;
    currentIndex = idx;
    const imgMeta = images[idx];
    document.getElementById("eraser-modal").classList.add("open");
    document.getElementById("eraser-revert").disabled = !imgMeta.has_backup;
    updatePrevNextButtons();
    strokes = [];
    redoStack = [];
    markers = [];
    liveSel = null;
    pad = { l: 0, t: 0, r: 0, b: 0 };
    if (hooks.onImageChange) hooks.onImageChange();
    const hadPreview = !!preview;
    clearPreviewQuiet();
    if (hadPreview && hooks.onPreviewCleared) hooks.onPreviewCleared();
    cropRect = null;
    cropDragMode = null;
    cropDragStart = null;
    zoom = null;
    updateZoomLabel();

    const canvas = document.getElementById("eraser-canvas");
    const loader = new Image();
    const reqIdx = idx;   // clicking through images fast: an older load finishing late must not win
    loader.onload = () => {
      if (currentIndex !== reqIdx) return;
      baseImage = loader;
      canvas.width = loader.naturalWidth;
      canvas.height = loader.naturalHeight;
      maskCanvas.width = loader.naturalWidth;
      maskCanvas.height = loader.naturalHeight;
      if (tool === "crop") initCropRect();
      applyZoomStyle();
      redraw();
    };
    loader.onerror = () => { if (currentIndex === reqIdx) toast("Couldn't load that image.", "error"); };
    loader.src = Api.imageUrl(imgMeta.path, Date.now());
  }

  function closeEditor() {
    if (preview) discardPreview();
    document.getElementById("eraser-modal").classList.remove("open");
  }

  // ------------------------------ zoom ------------------------------

  const scrollEl = () => document.getElementById("eraser-scroll");
  let spaceDown = false;
  let pan = null;
  let keysOpen = false;

  // The scale that makes the whole picture fit the window (never beyond 3x for tiny pictures).
  function fitScale() {
    const sc = scrollEl(), c = document.getElementById("eraser-canvas");
    if (!sc || !c.width || !c.height) return 1;
    const w = sc.clientWidth - 24, h = sc.clientHeight - 24;
    if (w <= 0 || h <= 0) return 1;
    return Math.max(0.02, Math.min(w / c.width, h / c.height, 3));
  }

  function curScale() { return zoom === null ? fitScale() : zoom; }

  function viewCenter() {
    const r = scrollEl().getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }

  // Zoom (null = fit) keeping the picture point under `anchor` (client coordinates) where it is.
  function setZoom(newZoom, anchor) {
    if (newZoom !== null) newZoom = Math.max(0.02, Math.min(32, newZoom));
    const canvas = document.getElementById("eraser-canvas");
    const sc = scrollEl();
    anchor = anchor || viewCenter();
    const before = canvas.getBoundingClientRect();
    const u = before.width ? (anchor.x - before.left) / before.width : 0.5;
    const v = before.height ? (anchor.y - before.top) / before.height : 0.5;
    zoom = newZoom;
    applyZoomStyle();
    updateZoomLabel();
    showHud(zoom === null ? `Fit ${Math.round(fitScale() * 100)}%` : `${Math.round(zoom * 100)}%`);
    if (zoom !== null) {
      const after = canvas.getBoundingClientRect();
      sc.scrollLeft += after.left + u * after.width - anchor.x;
      sc.scrollTop += after.top + v * after.height - anchor.y;
    }
    updateCursorSize();
    drawMaskScreen();
  }

  function applyZoomStyle() {
    const canvas = document.getElementById("eraser-canvas");
    const k = curScale();
    canvas.style.width = `${canvas.width * k}px`;
    canvas.style.height = `${canvas.height * k}px`;
    scrollEl().classList.toggle("pixelated", k >= 8);   // smooth when zoomed in; hard pixels only at extreme zoom
    updateZoomLabel();
    drawMaskScreen();
  }

  function updateZoomLabel() {
    document.getElementById("eraser-zoom-label").textContent = zoom === null ? `Fit ${Math.round(fitScale() * 100)}%` : `${Math.round(zoom * 100)}%`;
  }

  function setBrushSize(n) {
    brushSize = Math.max(2, Math.min(400, Math.round(n)));
    document.getElementById("eraser-brush-size").value = brushSize;
    document.getElementById("eraser-brush-val").textContent = brushSize;
    updateCursorSize();
    savePref("brush", brushSize);
    showHud(`Brush ${brushSize}px`);
  }

  // ------------------------------ remembered settings, hud, shortcut strip ------------------------------

  const PREF_KEY = "anima.eraser.prefs";
  function loadPrefs() { try { return JSON.parse(localStorage.getItem(PREF_KEY) || "{}") || {}; } catch { return {}; } }
  function savePref(k, v) { try { const p = loadPrefs(); p[k] = v; localStorage.setItem(PREF_KEY, JSON.stringify(p)); } catch { /* private mode */ } }

  let hudTimer = null;
  function showHud(text) {
    const el = document.getElementById("eraser-hud");
    if (!el || !document.getElementById("eraser-modal").classList.contains("open")) return;
    el.textContent = text;
    el.classList.add("show");
    clearTimeout(hudTimer);
    hudTimer = setTimeout(() => el.classList.remove("show"), 900);
  }

  // Always-visible hints for the current tool, built from the user's own keys.
  function renderKeybar() {
    const el = document.getElementById("eraser-keybar");
    if (!el) return;
    const has = (id) => Keybinds.keysOf(id).some(Boolean);
    const k = (id) => `<kbd>${Keybinds.label(id)}</kbd>`;
    // a hint is shown only when every action it mentions still has a key
    const hint = (ids, text) => (ids.every(has) ? ids.map(k).join(" ") + " " + text : "");
    const common = [hint(["run"], "run / accept"), `<kbd>Wheel</kbd> zoom`, hint(["pan"], "+drag move"),
      [hint(["fit"], "fit"), hint(["actual"], "100%")].filter(Boolean).join("  "),
      [hint(["undo"], "undo"), hint(["redo"], "redo")].filter(Boolean).join("  "), hint(["aiPanel"], "AI panel"), hint(["help"], "all keys")];
    const byTool = {
      brush: [hint(["brushSmaller", "brushLarger"], "brush size"), `<kbd>Alt</kbd>+wheel size`,
        [hint(["maskGrow"], "grow"), hint(["maskShrink"], "shrink"), hint(["maskInvert"], "invert")].filter(Boolean).join("  ")],
      rect: [[hint(["maskGrow"], "grow"), hint(["maskInvert"], "invert")].filter(Boolean).join("  ")],
      select: [has("selectToggle") ? `click add  right-click / ${k("selectToggle")} remove` : "click add  right-click remove"],
      extend: [], crop: [],
    };
    el.innerHTML = [...(byTool[tool] || []), ...common].filter(Boolean).map((t) => `<span>${t}</span>`).join("");
  }

  // The "?" popup lists every action with its current keys.
  function renderKeysPop() {
    const pop = document.getElementById("eraser-keys-pop");
    if (!pop) return;
    pop.innerHTML = "<b>Keyboard</b>";
    let group = "";
    for (const a of Keybinds.ACTIONS) {
      if (a.group !== group) { group = a.group; pop.insertAdjacentHTML("beforeend", `<div class="keys-group">${group}</div>`); }
      pop.insertAdjacentHTML("beforeend",
        `<div><kbd>${Keybinds.label(a.id)}</kbd> ${a.label}</div>`);
    }
    pop.insertAdjacentHTML("beforeend",
      `<div class="keys-group">Fixed</div><div><kbd>Esc</kbd> discard a result, or close</div>` +
      `<div><kbd>Wheel</kbd> zoom to the cursor, <kbd>Ctrl</kbd>/<kbd>Alt</kbd>+wheel brush size, middle-drag to move</div>` +
      `<div class="keys-foot">Change these in <b>App settings → Eraser keybinds</b>.</div>`);
  }

  function applyKeyTitles() {
    const t = (id, text) => { const el = document.getElementById(id); if (el) el.title = text; };
    t("eraser-undo", `Undo (${Keybinds.label("undo")})`);
    t("eraser-redo", `Redo (${Keybinds.label("redo")})`);
    t("eraser-zoom-out", `Zoom out (${Keybinds.label("zoomOut")} or mouse wheel)`);
    t("eraser-zoom-in", `Zoom in (${Keybinds.label("zoomIn")} or mouse wheel)`);
    t("eraser-zoom-label", `Fit to window (${Keybinds.label("fit")})`);
    t("eraser-zoom-100", `Actual size (${Keybinds.label("actual")})`);
    t("eraser-keys-btn", `Keyboard shortcuts (${Keybinds.label("help")})`);
    t("eraser-brush-size", `Brush size (${Keybinds.label("brushSmaller")} and ${Keybinds.label("brushLarger")}, or Ctrl/Alt + mouse wheel)`);
    t("eraser-prev-overlay", `Previous image (${Keybinds.label("prev")})`);
    t("eraser-next-overlay", `Next image (${Keybinds.label("next")})`);
    document.querySelectorAll("#eraser-tool-toggle button").forEach((btn) => {
      const id = { brush: "toolBrush", rect: "toolRect", select: "toolSelect", extend: "toolExtend", crop: "toolCrop" }[btn.dataset.tool];
      const key = btn.querySelector("kbd");
      if (key && id) key.textContent = Keybinds.first(id) || "";
    });
  }

  function refreshKeyHints() { renderKeybar(); renderKeysPop(); applyKeyTitles(); }

  function restorePrefs() {
    const p = loadPrefs();
    if (typeof p.brush === "number") { brushSize = Math.max(2, Math.min(400, p.brush)); document.getElementById("eraser-brush-size").value = brushSize; document.getElementById("eraser-brush-val").textContent = brushSize; }
    if (typeof p.copy === "boolean") document.getElementById("eraser-save-as-copy").checked = p.copy;
    if (typeof p.maskAmount === "number") document.getElementById("eraser-mask-amount").value = p.maskAmount;
    if (typeof p.sideW === "number") document.documentElement.style.setProperty("--eraser-side-w", `${p.sideW}px`);
    document.getElementById("eraser-save-as-copy").addEventListener("change", (e) => savePref("copy", e.target.checked));
    document.getElementById("eraser-mask-amount").addEventListener("change", (e) => savePref("maskAmount", parseInt(e.target.value, 10) || 8));
    // drag the edge of the AI panel to resize it
    const handle = document.getElementById("eraser-resizer");
    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      handle.classList.add("dragging");
      const move = (ev) => {
        const w = Math.max(280, Math.min(720, window.innerWidth - ev.clientX));
        document.documentElement.style.setProperty("--eraser-side-w", `${w}px`);
      };
      const up = () => {
        handle.classList.remove("dragging");
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        savePref("sideW", parseInt(getComputedStyle(document.documentElement).getPropertyValue("--eraser-side-w"), 10) || 380);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });
  }

  function toggleKeys(force) {
    keysOpen = typeof force === "boolean" ? force : !keysOpen;
    document.getElementById("eraser-keys-pop").style.display = keysOpen ? "block" : "none";
  }

  // Mouse wheel = zoom to the cursor (Alt + wheel = brush size), Space-drag or middle-drag = move around, and the fit
  // follows the window while "Fit" is on.
  function initViewport() {
    const sc = scrollEl();
    sc.addEventListener("wheel", (e) => {
      if (!baseImage) return;
      e.preventDefault();
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      // Ctrl or Alt + wheel = brush size (Ctrl would otherwise zoom the whole page); the other tools keep zooming.
      if ((e.altKey || e.ctrlKey || e.metaKey) && tool === "brush") { setBrushSize(brushSize * Math.exp(-(dy || e.deltaX) * 0.002)); return; }
      setZoom(curScale() * Math.exp(-dy * 0.0015), { x: e.clientX, y: e.clientY });
    }, { passive: false });
    // Ctrl + wheel anywhere over the editor must never reach the browser's own page zoom.
    document.getElementById("eraser-modal").addEventListener("wheel", (e) => {
      if (e.ctrlKey || e.metaKey) e.preventDefault();
    }, { passive: false, capture: true });
    sc.addEventListener("mousedown", (e) => {
      if (e.button === 1 || (e.button === 0 && spaceDown)) {
        e.preventDefault();
        e.stopPropagation();
        pan = { x: e.clientX, y: e.clientY, l: sc.scrollLeft, t: sc.scrollTop };
        sc.classList.add("panning");
      }
    }, true);
    window.addEventListener("mousemove", (e) => {
      if (!pan) return;
      sc.scrollLeft = pan.l - (e.clientX - pan.x);
      sc.scrollTop = pan.t - (e.clientY - pan.y);
    });
    window.addEventListener("mouseup", () => { if (pan) { pan = null; sc.classList.remove("panning"); } });
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => { if (baseImage && zoom === null) applyZoomStyle(); else drawMaskScreen(); }).observe(sc);
    sc.addEventListener("scroll", drawMaskScreen, { passive: true });
    }
  }

  function setSpace(on) {
    spaceDown = on;
    const sc = scrollEl();
    sc.classList.toggle("space-pan", on);
    document.getElementById("eraser-cursor").style.display = on || tool !== "brush" ? "none" : "";
  }

  function onKeyDown(e) {
    const modal = document.getElementById("eraser-modal");
    if (!modal.classList.contains("open")) return;
    // Not while the user is typing or on a control that uses the keys itself.
    const t = e.target;
    const textual = t && t.tagName === "INPUT" && !["range", "checkbox", "radio", "button", "color"].includes(t.type);
    if (t && (textual || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
    if (e.key === "Escape") {
      if (keysOpen) return toggleKeys(false);
      if (preview) discardPreview(); else closeEditor();
      return;
    }
    const action = Keybinds.actionFor(e);
    if (!action) return;
    if (t && t.tagName === "INPUT" && t.type === "range" && (action === "prev" || action === "next")) return;
    const hold = action === "pan" || action === "compare";
    if (e.repeat && hold) { e.preventDefault(); return; }
    switch (action) {
      case "run": e.preventDefault(); if (!e.repeat) applyOrAccept(); break;
      case "undo": e.preventDefault(); undoStroke(); break;
      case "redo": e.preventDefault(); redoStroke(); break;
      case "prev": step(-1); break;
      case "next": step(1); break;
      case "toolBrush": e.preventDefault(); selectTool("brush"); break;
      case "toolRect": e.preventDefault(); selectTool("rect"); break;
      case "toolSelect": e.preventDefault(); selectTool("select"); break;
      case "toolExtend": e.preventDefault(); selectTool("extend"); break;
      case "toolCrop": e.preventDefault(); selectTool("crop"); break;
      case "brushSmaller": e.preventDefault(); setBrushSize(Math.min(brushSize - 1, brushSize / 1.12)); break;
      case "brushLarger": e.preventDefault(); setBrushSize(Math.max(brushSize + 1, brushSize * 1.12)); break;
      case "zoomIn": e.preventDefault(); setZoom(curScale() * 1.25); break;
      case "zoomOut": e.preventDefault(); setZoom(curScale() / 1.25); break;
      case "fit": e.preventDefault(); setZoom(null); break;
      case "actual": e.preventDefault(); setZoom(1); break;
      case "clear": e.preventDefault(); clearStrokes(); break;
      case "pan": e.preventDefault(); setSpace(true); break;
      case "compare": if (preview) { comparing = true; redraw(); } break;
      case "help": toggleKeys(); break;
      case "selectToggle": {
        if (tool === "select") { const b = document.querySelector("#eraser-sel-mode button:not(.active)"); if (b) b.click(); }
        break;
      }
      case "aiPanel": {
        const b = document.getElementById("eraser-ai-toggle");
        if (b && b.offsetParent !== null) { e.preventDefault(); b.click(); }
        break;
      }
      case "maskGrow": case "maskShrink": case "maskInvert": {
        const op = action === "maskInvert" ? "reverse" : action === "maskShrink" ? "shrink" : "expand";
        const b = document.querySelector(`#eraser-mask-tools [data-mask-op="${op}"]`);
        if (b && document.getElementById("eraser-mask-tools").offsetParent !== null) {
          e.preventDefault();
          b.click();
          showHud(`Mask: ${op === "reverse" ? "inverted" : op === "expand" ? "grown" : "shrunk"}`);
        }
        break;
      }
      default: break;
    }
  }

  function onKeyUp(e) {
    if (!document.getElementById("eraser-modal").classList.contains("open")) return;
    const released = Keybinds.releasedHold(e);
    if (released === "pan") { e.preventDefault(); setSpace(false); }
    if (released === "compare" && comparing && preview) { comparing = false; redraw(); }
  }

  // Enter: accept a waiting AI result, otherwise run the current tool (Erase / Extend / Apply crop).
  function applyOrAccept() {
    const accept = document.getElementById("eraser-accept");
    if (preview && accept && accept.offsetParent !== null) return accept.click();
    const id = tool === "crop" ? "eraser-crop-apply" : "eraser-apply";
    const btn = document.getElementById(id);
    if (btn && !btn.disabled) btn.click();
  }

  // ------------------------------ drawing ------------------------------

  // What the canvas shows right now: the picture, the picture with the Extend margins, or an AI result.
  function viewSize() {
    const canvas = document.getElementById("eraser-canvas");
    if (!baseImage) return { w: canvas.width, h: canvas.height };
    const bw = baseImage.naturalWidth, bh = baseImage.naturalHeight;
    if (preview) return { w: preview.img.naturalWidth, h: preview.img.naturalHeight };
    if (tool === "extend") return { w: bw + pad.l + pad.r, h: bh + pad.t + pad.b };
    return { w: bw, h: bh };
  }

  // The painted mask over the picture, drawn on its own canvas at SCREEN resolution (not the picture's), so the fill
  // and its outline stay smooth at any zoom. One flat see-through tint (overlapping strokes do not get darker) plus
  // a thin solid line around the painted area (about 1.6 screen pixels at any zoom).
  const MASK_ALPHA = 0.38;
  let fillLayer = null, ringLayer = null;
  function drawMaskScreen() {
    const ov = document.getElementById("eraser-mask-overlay");
    const wrap = document.getElementById("eraser-canvas-wrap");
    const canvas = document.getElementById("eraser-canvas");
    if (!ov || !wrap) return;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(wrap.clientWidth * dpr)), H = Math.max(1, Math.round(wrap.clientHeight * dpr));
    if (ov.width !== W || ov.height !== H) { ov.width = W; ov.height = H; }
    const ctx = ov.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!strokes.length || preview || tool === "extend" || !baseImage || !canvas.width) return;
    const cr = canvas.getBoundingClientRect(), wr = wrap.getBoundingClientRect();
    const k = cr.width / canvas.width;
    if (!(k > 0)) return;
    if (!fillLayer) { fillLayer = document.createElement("canvas"); ringLayer = document.createElement("canvas"); }
    for (const c of [fillLayer, ringLayer]) if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const f = fillLayer.getContext("2d");
    f.setTransform(1, 0, 0, 1, 0, 0);
    f.globalCompositeOperation = "source-over";
    f.clearRect(0, 0, W, H);
    f.imageSmoothingEnabled = true; f.imageSmoothingQuality = "high";
    f.setTransform(dpr * k, 0, 0, dpr * k, dpr * (cr.left - wr.left), dpr * (cr.top - wr.top));
    for (const stroke of strokes) drawStroke(f, stroke, "white");
    f.setTransform(1, 0, 0, 1, 0, 0);
    f.globalCompositeOperation = "source-in";
    f.fillStyle = "rgb(206, 158, 90)";
    f.fillRect(0, 0, W, H);
    ctx.globalAlpha = MASK_ALPHA;
    ctx.drawImage(fillLayer, 0, 0);
    ctx.globalAlpha = 1;
    // outline: the shape shifted around in a circle, then the inside cut out
    const r = ringLayer.getContext("2d");
    r.globalCompositeOperation = "source-over";
    r.clearRect(0, 0, W, H);
    const w = 1.6 * dpr;
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      r.drawImage(fillLayer, Math.cos(a) * w, Math.sin(a) * w);
    }
    r.globalCompositeOperation = "source-in";
    r.fillStyle = "#f0cf94";
    r.fillRect(0, 0, W, H);
    r.globalCompositeOperation = "destination-out";
    r.drawImage(fillLayer, 0, 0);
    ctx.drawImage(ringLayer, 0, 0);
  }

  function renderMask() {
    const mctx = maskCanvas.getContext("2d");
    mctx.fillStyle = "black";
    mctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    for (const stroke of strokes) drawStroke(mctx, stroke, "white");
  }

  function redraw() {
    redrawPicture();
    drawMaskScreen();
  }

  function redrawPicture() {
    const canvas = document.getElementById("eraser-canvas");
    const size = viewSize();
    if (canvas.width !== size.w || canvas.height !== size.h) {
      canvas.width = size.w;
      canvas.height = size.h;
      applyZoomStyle();
    }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (preview) {
      if (!comparing) {
        if (preview.bg) { ctx.fillStyle = preview.bg; ctx.fillRect(0, 0, canvas.width, canvas.height); }
        ctx.drawImage(preview.img, 0, 0);
      } else if (baseImage) {
        if (preview.mode === "extend") ctx.drawImage(baseImage, preview.dx, preview.dy);
        else ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);
      }
      return;
    }

    if (tool === "extend" && baseImage) {
      ctx.fillStyle = "rgba(206, 158, 90, 0.18)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(baseImage, pad.l, pad.t);
      ctx.strokeStyle = "#e7edf0";
      ctx.lineWidth = Math.max(2, canvas.width / 400);
      ctx.setLineDash([canvas.width / 60, canvas.width / 90]);
      ctx.strokeRect(ctx.lineWidth / 2, ctx.lineWidth / 2, canvas.width - ctx.lineWidth, canvas.height - ctx.lineWidth);
      ctx.setLineDash([]);
      return;
    }

    if (baseImage) ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);

    renderMask();

    if (tool === "select" && markers.length) {
      const r = Math.max(5, canvas.width / 150);
      for (const m of markers) {
        ctx.beginPath();
        ctx.arc(m.x, m.y, r, 0, Math.PI * 2);
        ctx.fillStyle = m.include ? "#3ddc84" : "#ff5252";
        ctx.fill();
        ctx.lineWidth = Math.max(1.5, r / 3);
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      }
    }

    if (tool === "crop" && cropRect) {
      drawCropOverlay(ctx, canvas);
    }
  }

  function drawCropOverlay(ctx, canvas) {
    const x = Math.min(cropRect.x0, cropRect.x1);
    const y = Math.min(cropRect.y0, cropRect.y1);
    const w = Math.abs(cropRect.x1 - cropRect.x0);
    const hh = Math.abs(cropRect.y1 - cropRect.y0);

    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fillRect(0, 0, canvas.width, y); // top
    ctx.fillRect(0, y + hh, canvas.width, canvas.height - (y + hh)); // bottom
    ctx.fillRect(0, y, x, hh); // left
    ctx.fillRect(x + w, y, canvas.width - (x + w), hh); // right

    ctx.strokeStyle = "#e7edf0";
    ctx.lineWidth = Math.max(1, canvas.width / 500);
    ctx.setLineDash([canvas.width / 100, canvas.width / 150]);
    ctx.strokeRect(x, y, w, hh);
    ctx.setLineDash([]);

    // rule-of-thirds guide lines, drawn faint - standard crop-tool convention
    ctx.strokeStyle = "rgba(231, 237, 240, 0.35)";
    ctx.lineWidth = Math.max(1, canvas.width / 800);
    for (let i = 1; i <= 2; i++) {
      ctx.beginPath();
      ctx.moveTo(x + (w * i) / 3, y);
      ctx.lineTo(x + (w * i) / 3, y + hh);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y + (hh * i) / 3);
      ctx.lineTo(x + w, y + (hh * i) / 3);
      ctx.stroke();
    }

    // resize handles - small filled squares at each corner and edge midpoint
    const hs = handleSizeInCanvasUnits();
    ctx.fillStyle = "#e7edf0";
    ctx.strokeStyle = "#0a0d0f";
    ctx.lineWidth = Math.max(1, canvas.width / 600);
    Object.values(cropHandlePositions(cropRect)).forEach(([hx, hy]) => {
      ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
      ctx.strokeRect(hx - hs / 2, hy - hs / 2, hs, hs);
    });
  }

  function drawStroke(ctx, stroke, color) {
    if (stroke.type === "sel") {                 // an AI selection: pre-rendered as a tinted and a white-on-transparent layer
      ctx.drawImage(color === "white" ? stroke.white : stroke.tint, 0, 0);
      return;
    }
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (stroke.type === "brush") {
      ctx.lineWidth = stroke.size;
      if (stroke.points.length > 1) {
        ctx.beginPath();
        stroke.points.forEach((pt, i) => {
          if (i === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.stroke();
      }
      // Round caps at every point so a single click (no drag) still paints a dot.
      stroke.points.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, stroke.size / 2, 0, Math.PI * 2);
        ctx.fill();
      });
    } else if (stroke.type === "rect") {
      const x = Math.min(stroke.x0, stroke.x1);
      const y = Math.min(stroke.y0, stroke.y1);
      const w = Math.abs(stroke.x1 - stroke.x0);
      const hh = Math.abs(stroke.y1 - stroke.y0);
      ctx.fillRect(x, y, w, hh);
    }
  }

  function canvasPoint(e) {
    const canvas = document.getElementById("eraser-canvas");
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: Math.max(0, Math.min(canvas.width, (clientX - rect.left) * scaleX)),
      y: Math.max(0, Math.min(canvas.height, (clientY - rect.top) * scaleY)),
    };
  }

  function cssBrushDiameter() {
    // brushSize is in canvas (image-pixel) units; convert to on-screen
    // CSS pixels so the circle matches what will actually get erased.
    const canvas = document.getElementById("eraser-canvas");
    const rect = canvas.getBoundingClientRect();
    if (!canvas.width || !rect.width) return brushSize;
    const scale = rect.width / canvas.width;
    return brushSize * scale;
  }

  function updateCursorPosition(e) {
    if (tool !== "brush") return;
    const cursor = document.getElementById("eraser-cursor");
    const wrapRect = document.getElementById("eraser-canvas-wrap").getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    cursor.style.left = `${clientX - wrapRect.left}px`;
    cursor.style.top = `${clientY - wrapRect.top}px`;
    updateCursorSize();
  }

  function updateCursorSize() {
    const cursor = document.getElementById("eraser-cursor");
    const diameter = cssBrushDiameter();
    cursor.style.width = `${diameter}px`;
    cursor.style.height = `${diameter}px`;
  }

  function clampRectToCanvas(rect) {
    const canvas = document.getElementById("eraser-canvas");
    let { x0, y0, x1, y1 } = rect;
    x0 = Math.max(0, x0); y0 = Math.max(0, y0);
    x1 = Math.min(canvas.width, x1); y1 = Math.min(canvas.height, y1);
    return { x0, y0, x1, y1 };
  }

  function onPointerDown(e) {
    if (!baseImage || preview || tool === "extend") return;
    e.preventDefault();
    const pt = canvasPoint(e);
    if (tool === "select") {
      if (hooks.onSelectClick) hooks.onSelectClick(pt, !(e.button === 2 || e.shiftKey));
      return;
    }
    if (tool === "brush" || tool === "rect") redoStack = [];
    if (tool === "brush") {
      isDrawing = true;
      strokes.push({ type: "brush", points: [pt], size: brushSize });
    } else if (tool === "rect") {
      isDrawing = true;
      strokes.push({ type: "rect", x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y });
    } else if (tool === "crop") {
      if (!cropRect) initCropRect();
      const hit = hitTestCrop(pt);
      if (hit) {
        cropDragMode = hit;
        isDrawing = true;
        cropDragStart = { x: pt.x, y: pt.y, rect: { ...cropRect } };
      } else {
        // Clicked outside the current box and not on a handle - draw a
        // brand new crop rectangle from scratch here, the same way the
        // Rectangle tool already lets you draw fresh, instead of leaving
        // the click with nothing to do.
        cropDragMode = "new";
        isDrawing = true;
        cropDragStart = { x: pt.x, y: pt.y, prevRect: cropRect ? { ...cropRect } : null };
        cropRect = { x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y };
      }
    }
    redraw();
  }

  function onPointerMove(e) {
    // mousemove is now bound on window (see init) so drags stay smooth
    // past the canvas edge - guard here so that doesn't mean doing work
    // on every mouse move across the whole page while the editor is closed.
    if (!baseImage || preview || !document.getElementById("eraser-modal").classList.contains("open")) return;
    if (tool === "crop") {
      // Update the hover cursor even when not actively dragging.
      if (!isDrawing) {
        const hover = hitTestCrop(canvasPoint(e));
        document.getElementById("eraser-canvas").style.cursor = hover ? cursorForHandle(hover) : "crosshair";
      }
    }
    if (!isDrawing) return;
    e.preventDefault();
    const pt = canvasPoint(e);
    if (tool === "brush") {
      const last = strokes[strokes.length - 1];
      if (last) last.points.push(pt);
    } else if (tool === "rect") {
      const last = strokes[strokes.length - 1];
      if (last) { last.x1 = pt.x; last.y1 = pt.y; }
    } else if (tool === "crop" && cropDragMode) {
      updateCropDrag(pt);
    }
    redraw();
  }

  function updateCropDrag(pt) {
    if (cropDragMode === "new") {
      let x0 = cropDragStart.x, y0 = cropDragStart.y, x1 = pt.x, y1 = pt.y;
      if (cropAspect) {
        // Keep the drag's width, derive height from it, preserving
        // whichever vertical direction the drag is currently going.
        const w = x1 - x0;
        const signY = y1 - y0 < 0 ? -1 : 1;
        y1 = y0 + (signY * Math.abs(w)) / cropAspect;
      }
      cropRect = clampRectToCanvas({
        x0: Math.min(x0, x1), y0: Math.min(y0, y1),
        x1: Math.max(x0, x1), y1: Math.max(y0, y1),
      });
      return;
    }

    const r0 = cropDragStart.rect;
    const dx = pt.x - cropDragStart.x;
    const dy = pt.y - cropDragStart.y;
    let { x0, y0, x1, y1 } = r0;

    if (cropDragMode === "move") {
      const w = r0.x1 - r0.x0, h = r0.y1 - r0.y0;
      const canvas = document.getElementById("eraser-canvas");
      const mdx = Math.max(-r0.x0, Math.min(canvas.width - r0.x1, dx));
      const mdy = Math.max(-r0.y0, Math.min(canvas.height - r0.y1, dy));
      cropRect = { x0: r0.x0 + mdx, y0: r0.y0 + mdy, x1: r0.x1 + mdx, y1: r0.y1 + mdy };
      return;
    }

    if (cropDragMode.includes("n")) y0 = Math.min(r0.y0 + dy, r0.y1 - CROP_MIN_SIZE);
    if (cropDragMode.includes("s")) y1 = Math.max(r0.y1 + dy, r0.y0 + CROP_MIN_SIZE);
    if (cropDragMode.includes("w")) x0 = Math.min(r0.x0 + dx, r0.x1 - CROP_MIN_SIZE);
    if (cropDragMode.includes("e")) x1 = Math.max(r0.x1 + dx, r0.x0 + CROP_MIN_SIZE);

    let rect = { x0, y0, x1, y1 };
    if (cropAspect) rect = applyAspectToRect(rect, cropDragMode);
    cropRect = clampRectToCanvas(rect);
  }

  function onPointerUp() {
    if (tool === "crop" && cropDragMode === "new" && cropRect) {
      const w = cropRect.x1 - cropRect.x0, h = cropRect.y1 - cropRect.y0;
      if (w < CROP_MIN_SIZE || h < CROP_MIN_SIZE) {
        // A stray click (no real drag) - fall back to whatever box was
        // there before instead of leaving a sliver-sized crop rect.
        cropRect = (cropDragStart && cropDragStart.prevRect) || null;
        if (!cropRect && baseImage) initCropRect();
        redraw();
      }
    }
    isDrawing = false;
    cropDragMode = null;
    cropDragStart = null;
  }

  function clearStrokes() {
    strokes = [];
    redoStack = [];
    markers = [];
    liveSel = null;
    if (hooks.onMaskCleared) hooks.onMaskCleared();
    if (tool === "crop") initCropRect();
    redraw();
  }

  // Undo / Redo: first the strokes you have painted but not applied; once there are none, the edits already applied
  // to the picture (each erase, crop or AI result is saved aside before it overwrites the file).
  function undoStroke() {
    if (preview) return;
    if (tool === "crop") { initCropRect(); redraw(); return; }
    if (strokes.length) {
      const popped = strokes.pop();
      redoStack.push(popped);
      if (popped === liveSel) {
        liveSel = null;
        markers = [];
        if (hooks.onMaskCleared) hooks.onMaskCleared();
      }
      redraw();
      return;
    }
    stepImage("undo");
  }

  function redoStroke() {
    if (preview) return;
    if (tool !== "crop" && redoStack.length) {
      strokes.push(redoStack.pop());
      redraw();
      return;
    }
    stepImage("redo");
  }

  let stepping = false;
  async function stepImage(step) {
    if (currentIndex < 0 || stepping) return;
    const imgMeta = images[currentIndex];
    stepping = true;
    try {
      const r = await Api.editHistory({ step, image_path: imgMeta.path });
      if (!r.ok) { showHud(step === "undo" ? "Nothing to undo" : "Nothing to redo"); return; }
      strokes = []; redoStack = []; markers = []; liveSel = null;
      if (hooks.onMaskCleared) hooks.onMaskCleared();
      await openEditorAt(currentIndex);
      renderGrid();
      showHud(step === "undo" ? `Undid the last edit (${r.undo} more)` : `Redid the edit (${r.redo} more)`);
    } catch (err) {
      toast(err.message, "error");
    } finally {
      stepping = false;
    }
  }

  // ------------------------------ AI helpers (used through `bridge`) ------------------------------

  // A black/white mask picture (white = the area) -> a stroke drawn as a tinted overlay and as white on the mask canvas.
  function maskToStroke(src) {
    const w = maskCanvas.width, h = maskCanvas.height;
    const tmp = document.createElement("canvas");
    tmp.width = w; tmp.height = h;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(src, 0, 0, w, h);
    const px = tctx.getImageData(0, 0, w, h).data;
    const mk = (r, g, b, a) => {
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const cctx = c.getContext("2d");
      const out = cctx.createImageData(w, h);
      for (let i = 0; i < px.length; i += 4) {
        if (px[i] >= 127) { out.data[i] = r; out.data[i + 1] = g; out.data[i + 2] = b; out.data[i + 3] = a; }
      }
      cctx.putImageData(out, 0, 0);
      return c;
    };
    return { type: "sel", tint: mk(206, 158, 90, 140), white: mk(255, 255, 255, 255) };
  }

  function maskBounds() {
    renderMask();
    const w = maskCanvas.width, h = maskCanvas.height;
    const px = maskCanvas.getContext("2d").getImageData(0, 0, w, h).data;
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) {
      const row = y * w * 4;
      for (let x = 0; x < w; x++) {
        if (px[row + x * 4] >= 127) {
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          y1 = y;
        }
      }
    }
    return x1 < 0 ? null : { x0, y0, x1: x1 + 1, y1: y1 + 1 };
  }

  function setSelection(img) {
    const stroke = maskToStroke(img);
    const i = liveSel ? strokes.indexOf(liveSel) : -1;
    if (i >= 0) strokes[i] = stroke; else strokes.push(stroke);
    liveSel = stroke;
    redoStack = [];
    redraw();
  }

  function setMaskFromImage(img) {
    strokes = [maskToStroke(img)];
    redoStack = [];
    liveSel = null;
    markers = [];
    redraw();
  }

  function clearSelection() {
    const i = liveSel ? strokes.indexOf(liveSel) : -1;
    if (i >= 0) strokes.splice(i, 1);
    liveSel = null;
    markers = [];
    redraw();
  }

  function setPreview(p) {
    preview = { mode: "same", dx: 0, dy: 0, bg: null, ...p };
    comparing = false;
    document.getElementById("eraser-canvas-wrap").classList.add("has-preview");
    redraw();
  }

  function clearPreviewQuiet() {
    preview = null;
    comparing = false;
    document.getElementById("eraser-canvas-wrap").classList.remove("has-preview");
  }

  function discardPreview() {
    clearPreviewQuiet();
    redraw();
    if (hooks.onPreviewCleared) hooks.onPreviewCleared();
  }

  const samePath = (a, b) => String(a).replace(/\\/g, "/").toLowerCase() === String(b).replace(/\\/g, "/").toLowerCase();

  // A result was written to disk by /api/edit/ai/commit: reload the picture, or refresh the list when a copy was made.
  async function afterCommit(savedPath) {
    const imgMeta = images[currentIndex];
    clearPreviewQuiet();
    const inPlace = imgMeta && samePath(savedPath, imgMeta.path);
    if (inPlace) {
      imgMeta.has_backup = true;
      openEditorAt(currentIndex);
      renderGrid();
    } else {
      toast(`Saved: ${String(savedPath).split(/[\\/]/).pop()}`, "success");
      await load();
    }
  }

  const bridge = {
    hooks,
    hasImage: () => currentIndex >= 0 && !!baseImage,
    imagePath: () => (currentIndex >= 0 && images[currentIndex] ? images[currentIndex].path : ""),
    folder: () => currentFolder,
    tool: () => tool,
    size: () => (baseImage ? { w: baseImage.naturalWidth, h: baseImage.naturalHeight } : null),
    saveAsCopy: () => document.getElementById("eraser-save-as-copy").checked,
    hasMask: () => strokes.length > 0,
    maskDataUrl: () => { renderMask(); return maskCanvas.toDataURL("image/png"); },
    maskBounds,
    getPad: () => ({ ...pad }),
    setPad: (p) => { pad = { l: p.l | 0, t: p.t | 0, r: p.r | 0, b: p.b | 0 }; if (tool === "extend" && !preview) redraw(); },
    setMarkers: (list) => { markers = list; redraw(); },
    setSelection,
    clearSelection,
    commitSelection: () => { liveSel = null; markers = []; redraw(); },
    setMaskFromImage,
    setPreview,
    discardPreview,
    setCompare: (on) => { if (preview) { comparing = !!on; redraw(); } },
    hasPreview: () => !!preview,
    afterCommit,
    redraw,
  };

  // ------------------------------ apply actions ------------------------------

  async function applyErase() {
    if (currentIndex < 0) return;
    if (preview) return toast("Accept or discard the AI result first.", "info");
    if (tool === "extend") return hooks.runExtend ? hooks.runExtend() : toast("The AI tools are not loaded.", "error");
    const methodValue = document.getElementById("eraser-method").value;
    if (hooks.isAiMethod && hooks.isAiMethod(methodValue)) return hooks.runInpaint();
    if (strokes.length === 0) return toast("Paint over the area to erase first.", "error");
    const imgMeta = images[currentIndex];
    const maskB64 = maskCanvas.toDataURL("image/png");
    const saveAsCopy = document.getElementById("eraser-save-as-copy").checked;
    const btn = document.getElementById("eraser-apply");
    btn.disabled = true;
    btn.textContent = "Erasing…";
    try {
      const data = await Api.editInpaint({
        image_path: imgMeta.path,
        mask: maskB64,
        method: methodValue.replace(/^cv:/, ""),
        radius: hooks.cvRadius ? hooks.cvRadius() : 4,
        save_as_copy: saveAsCopy,
      });
      if (saveAsCopy) {
        toast(`Saved a copy: ${data.path.split(/[\\/]/).pop()}`, "success");
        await load();
      } else {
        imgMeta.has_backup = true;
        toast("Erased.", "success");
        openEditorAt(currentIndex); // reload so the real inpainted result is visible
        renderGrid();
      }
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = applyLabel();
    }
  }

  async function applyCrop() {
    if (currentIndex < 0 || !cropRect) return toast("Drag a rectangle to select the crop area first.", "error");
    const imgMeta = images[currentIndex];
    const box = [
      Math.min(cropRect.x0, cropRect.x1), Math.min(cropRect.y0, cropRect.y1),
      Math.max(cropRect.x0, cropRect.x1), Math.max(cropRect.y0, cropRect.y1),
    ];
    const saveAsCopy = document.getElementById("eraser-save-as-copy").checked;
    const sendToSubset = document.getElementById("eraser-crop-to-subset").checked;
    const destDir = sendToSubset ? document.getElementById("eraser-crop-dest-dir").value.trim() : "";
    if (sendToSubset && !destDir) return toast("Choose or create a destination folder first.", "error");
    const upscale = parseFloat(document.getElementById("eraser-upscale-factor").value);
    const upscaleMethod = document.getElementById("eraser-upscale-method").value;
    if (upscale > 1 && upscaleMethod === "ultrasharp") {
      try {
        const status = await Api.upscalerStatus();
        if (!status.available) return toast("AI upscaling needs 'torch' and 'spandrel' installed on the server - see the Upscale row.", "error");
        if (!status.downloaded) return toast("Download the UltraSharp model first (see the Upscale row).", "error");
      } catch { /* let the crop request itself surface any error */ }
    }
    const btn = document.getElementById("eraser-crop-apply");
    btn.disabled = true;
    btn.textContent = "Cropping…";
    try {
      const data = await Api.editCrop({
        image_path: imgMeta.path, box, save_as_copy: saveAsCopy,
        dest_dir: destDir || undefined,
        upscale: upscale > 1 ? upscale : undefined,
        upscale_method: upscaleMethod,
      });
      if (destDir) {
        toast(`Sent crop to ${destDir.split(/[\\/]/).pop()}: ${data.path.split(/[\\/]/).pop()}`, "success");
        SubsetTabs.render("edit-subset-tabs", currentFolder, selectSubsetTab);
      } else if (saveAsCopy) {
        toast(`Saved a copy: ${data.path.split(/[\\/]/).pop()}`, "success");
        await load();
      } else {
        imgMeta.has_backup = true;
        toast("Cropped.", "success");
        cropRect = null;
        openEditorAt(currentIndex);
        renderGrid();
      }
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Apply crop";
    }
  }

  async function revertImage() {
    if (currentIndex < 0) return;
    const imgMeta = images[currentIndex];
    try {
      await Api.editRevert(imgMeta.path);
      imgMeta.has_backup = false;
      toast("Reverted to original.", "success");
      openEditorAt(currentIndex);
      renderGrid();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // ------------------------------ duplicate finder ------------------------------

  async function scanForDuplicates() {
    if (!currentFolder) return toast("Load a folder first.", "error");
    const threshold = document.getElementById("dup-threshold").value;
    const btn = document.getElementById("dup-scan-btn");
    const spinner = document.getElementById("dup-spinner");
    btn.disabled = true;
    spinner.style.display = "inline-block";
    try {
      const data = await Api.editDuplicates(currentFolder, threshold);
      dupSelection.clear();
      renderDuplicateGroups(data.groups);
      if (data.groups.length === 0) toast("No duplicates found at this threshold.", "info");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
      spinner.style.display = "none";
    }
  }

  function formatBytes(n) {
    if (n > 1e6) return `${(n / 1e6).toFixed(1)} MB`;
    if (n > 1e3) return `${(n / 1e3).toFixed(0)} KB`;
    return `${n} B`;
  }

  function renderDuplicateGroups(groups) {
    const container = document.getElementById("dup-results");
    container.innerHTML = "";
    if (groups.length === 0) return;

    groups.forEach((group) => {
      // Default: keep the first (largest file) entry, mark the rest for deletion.
      group.forEach((member, i) => {
        if (!dupSelection.has(member.path)) dupSelection.set(member.path, i === 0 ? "keep" : "delete");
      });

      const thumbs = group.map((member) => {
        const filename = member.path.split(/[\\/]/).pop();
        const state = dupSelection.get(member.path);
        const thumb = h("div", {
          class: `dup-thumb ${state}`,
          title: "Click to toggle keep / delete",
          onclick: () => {
            dupSelection.set(member.path, dupSelection.get(member.path) === "keep" ? "delete" : "keep");
            thumb.className = `dup-thumb ${dupSelection.get(member.path)}`;
            label.textContent = dupSelection.get(member.path) === "keep" ? "Keep" : "Delete";
          },
        }, [
          h("img", { src: Api.thumbUrl(member.path, 160), loading: "lazy", decoding: "async" }),
        ]);
        const label = h("span", { class: "dup-label" }, state === "keep" ? "Keep" : "Delete");
        thumb.appendChild(label);
        thumb.title = `${filename} - ${formatBytes(member.size)}`;
        return thumb;
      });

      const groupEl = h("div", { class: "dup-group" }, [
        h("div", { class: "dup-group-head" }, [`${group.length} similar images`]),
        h("div", { class: "dup-thumbs" }, thumbs),
      ]);
      container.appendChild(groupEl);
    });

    container.appendChild(h("div", { class: "row", style: "margin-top:6px" }, [
      h("button", { class: "btn danger", onclick: deleteMarkedDuplicates }, "Delete all marked"),
    ]));
  }

  async function deleteOneImage(img) {
    if (!confirm(`Delete "${img.filename}" and its caption file? This can't be undone.`)) return;
    try {
      const data = await Api.editDeleteImages([img.path]);
      const r = (data.results || []).find((x) => x.path === img.path);
      if (r && r.status === "error") throw new Error(r.reason || "Couldn't delete that file.");
      toast(`Deleted "${img.filename}".`, "success");
      images = images.filter((i) => i.path !== img.path);
      renderGrid();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function deleteMarkedDuplicates() {
    const toDelete = [...dupSelection.entries()].filter(([, state]) => state === "delete").map(([path]) => path);
    if (toDelete.length === 0) return toast("Nothing marked for deletion.", "info");
    if (!confirm(`Delete ${toDelete.length} image${toDelete.length === 1 ? "" : "s"}? This can't be undone.`)) return;
    try {
      const data = await Api.editDeleteImages(toDelete);
      const failed = (data.results || []).filter((r) => r.status === "error").length;
      toast(`Deleted ${data.deleted} image${data.deleted === 1 ? "" : "s"}.${failed ? ` ${failed} couldn't be deleted.` : ""}`,
            failed ? "error" : "success");
      document.getElementById("dup-results").innerHTML = "";
      dupSelection.clear();
      await load();
    } catch (err) {
      toast(err.message, "error");
    }
  }


  async function deleteCurrentImage() {
    if (currentIndex < 0 || currentIndex >= images.length) return;
    const img = images[currentIndex];
    if (!confirm(`Delete "${img.filename}" and its caption file? This can't be undone.`)) return;
    try {
      const data = await Api.editDeleteImages([img.path]);
      const r = (data.results || []).find((x) => x.path === img.path);
      if (r && r.status === "error") throw new Error(r.reason || "Couldn't delete that file.");
      toast(`Deleted "${img.filename}".`, "success");
      closeEditor();
      images = images.filter((i) => i.path !== img.path);
      renderGrid();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  return { init, bridge, setFolder: (path) => { document.getElementById("edit-folder").value = path; load(); } };
})();
