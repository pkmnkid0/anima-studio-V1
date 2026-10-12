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
    document.getElementById("eraser-brush-size").addEventListener("input", (e) => {
      brushSize = parseInt(e.target.value, 10);
      updateCursorSize();
    });
    document.getElementById("eraser-clear-mask").addEventListener("click", clearStrokes);
    document.getElementById("eraser-undo").addEventListener("click", undoStroke);
    const delModal = document.getElementById("eraser-delete-modal");
    if (delModal) delModal.addEventListener("click", deleteCurrentImage);
    const delBtn = document.getElementById("eraser-delete");
    if (delBtn) delBtn.addEventListener("click", deleteCurrentImage);
    const delModalBtn = document.getElementById("eraser-delete-modal");
    if (delModalBtn) delModalBtn.addEventListener("click", deleteCurrentImage);
    const delBtnGrid = document.getElementById("eraser-delete");
    if (delBtnGrid) delBtnGrid.addEventListener("click", deleteCurrentImage);
    document.getElementById("eraser-close").addEventListener("click", closeEditor);
    document.getElementById("eraser-apply").addEventListener("click", applyErase);
    document.getElementById("eraser-crop-apply").addEventListener("click", applyCrop);
    document.getElementById("eraser-revert").addEventListener("click", revertImage);
    document.getElementById("eraser-prev-overlay").addEventListener("click", () => step(-1));
    document.getElementById("eraser-next-overlay").addEventListener("click", () => step(1));
    document.getElementById("eraser-zoom-in").addEventListener("click", () => setZoom((zoom || 1) * 1.25));
    document.getElementById("eraser-zoom-out").addEventListener("click", () => setZoom((zoom || 1) / 1.25));
    document.getElementById("eraser-zoom-label").addEventListener("click", () => setZoom(null));
    document.getElementById("eraser-aspect").addEventListener("change", (e) => {
      cropAspect = parseAspect(e.target.value);
      if (cropRect) {
        cropRect = applyAspectToRect(cropRect, "se"); // re-fit the current rect, anchored top-left
        redraw();
      }
    });
    document.getElementById("eraser-crop-to-subset").addEventListener("change", (e) => {
      document.getElementById("eraser-crop-dest-wrap").style.display = e.target.checked ? "flex" : "none";
    });
    document.getElementById("eraser-upscale-method").addEventListener("change", refreshUpscalerStatusIfNeeded);
    document.getElementById("eraser-upscaler-download-btn").addEventListener("click", startUpscalerDownload);
    document.getElementById("eraser-upscaler-deps-btn").addEventListener("click", startUpscalerDepsInstall);
    refreshUpscalerStatusIfNeeded();
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

    document.getElementById("eraser-modal").addEventListener("click", (e) => {
      if (e.target.id === "eraser-modal") closeEditor();
    });
    document.addEventListener("keydown", (e) => {
      if (!document.getElementById("eraser-modal").classList.contains("open")) return;
      if (e.key === "Escape") closeEditor();
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
    });

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
    const depsBtn = document.getElementById("eraser-upscaler-deps-btn");
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

  let upscalerDepsPollTimer = null;

  async function startUpscalerDepsInstall() {
    const btn = document.getElementById("eraser-upscaler-deps-btn");
    const spinner = document.getElementById("eraser-upscaler-deps-spinner");
    btn.disabled = true;
    spinner.style.display = "inline-block";
    try {
      renderUpscalerDepsStatus(await Api.upscalerInstallDeps());
      if (!upscalerDepsPollTimer) upscalerDepsPollTimer = setInterval(pollUpscalerDepsStatus, 1000);
    } catch (err) {
      toast(err.message, "error");
      btn.disabled = false;
      spinner.style.display = "none";
    }
  }

  async function pollUpscalerDepsStatus() {
    try {
      renderUpscalerDepsStatus(await Api.upscalerInstallDepsStatus());
    } catch { /* transient - next poll retries */ }
  }

  function renderUpscalerDepsStatus(status) {
    const log = document.getElementById("eraser-upscaler-deps-log");
    const btn = document.getElementById("eraser-upscaler-deps-btn");
    const spinner = document.getElementById("eraser-upscaler-deps-spinner");

    if (status.log_tail && status.log_tail.length) {
      log.style.display = "block";
      log.textContent = status.log_tail.join("\n");
      log.scrollTop = log.scrollHeight;
    }

    if (status.stage === "installing") {
      btn.disabled = true;
      spinner.style.display = "inline-block";
      return;
    }

    spinner.style.display = "none";
    if (upscalerDepsPollTimer) { clearInterval(upscalerDepsPollTimer); upscalerDepsPollTimer = null; }

    if (status.stage === "done") {
      toast("Dependencies installed - UltraSharp is ready to use.", "success");
      refreshUpscalerStatusIfNeeded();
    } else if (status.stage === "error") {
      btn.disabled = false;
      toast(status.error || "Install failed - see the log.", "error");
    } else {
      btn.disabled = false;
    }
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
          h("img", { src: Api.thumbUrl(img.path, 240), loading: "lazy" }),
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
    tool = newTool;
    document.querySelectorAll("#eraser-tool-toggle button").forEach((b) => b.classList.toggle("active", b.dataset.tool === tool));
    const wrap = document.getElementById("eraser-canvas-wrap");
    wrap.classList.toggle("tool-rect", tool === "rect");
    wrap.classList.toggle("tool-crop", tool === "crop");
    document.getElementById("eraser-cursor").style.display = tool === "brush" ? "" : "none";
    document.getElementById("eraser-brush-size-row").style.display = tool === "brush" ? "flex" : "none";
    document.getElementById("eraser-method-row").style.display = tool === "crop" ? "none" : "flex";
    document.getElementById("eraser-aspect-row").style.display = tool === "crop" ? "flex" : "none";
    document.getElementById("eraser-upscale-row").style.display = tool === "crop" ? "flex" : "none";
    document.getElementById("eraser-crop-dest-row").style.display = tool === "crop" ? "flex" : "none";
    document.getElementById("eraser-apply").style.display = tool === "crop" ? "none" : "inline-flex";
    document.getElementById("eraser-crop-apply").style.display = tool === "crop" ? "inline-flex" : "none";
    if (tool === "crop" && !cropRect && baseImage) initCropRect();
    redraw();
  }

  function initCropRect() {
    const canvas = document.getElementById("eraser-canvas");
    const marginX = canvas.width * 0.1;
    const marginY = canvas.height * 0.1;
    cropRect = { x0: marginX, y0: marginY, x1: canvas.width - marginX, y1: canvas.height - marginY };
    if (cropAspect) cropRect = applyAspectToRect(cropRect, "se");
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
    cropRect = null;
    cropDragMode = null;
    cropDragStart = null;
    zoom = null;
    updateZoomLabel();

    const canvas = document.getElementById("eraser-canvas");
    const loader = new Image();
    loader.onload = () => {
      baseImage = loader;
      canvas.width = loader.naturalWidth;
      canvas.height = loader.naturalHeight;
      maskCanvas.width = loader.naturalWidth;
      maskCanvas.height = loader.naturalHeight;
      if (tool === "crop") initCropRect();
      applyZoomStyle();
      redraw();
    };
    loader.onerror = () => toast("Couldn't load that image.", "error");
    loader.src = Api.imageUrl(imgMeta.path, Date.now());
  }

  function closeEditor() {
    document.getElementById("eraser-modal").classList.remove("open");
  }

  // ------------------------------ zoom ------------------------------

  function setZoom(newZoom) {
    if (newZoom !== null) newZoom = Math.max(0.1, Math.min(6, newZoom));
    zoom = newZoom;
    applyZoomStyle();
    updateZoomLabel();
  }

  function applyZoomStyle() {
    const canvas = document.getElementById("eraser-canvas");
    if (zoom === null) {
      canvas.style.width = "";
      canvas.style.height = "";
    } else {
      canvas.style.width = `${canvas.width * zoom}px`;
      canvas.style.height = `${canvas.height * zoom}px`;
    }
  }

  function updateZoomLabel() {
    document.getElementById("eraser-zoom-label").textContent = zoom === null ? "Fit" : `${Math.round(zoom * 100)}%`;
  }

  // ------------------------------ drawing ------------------------------

  function redraw() {
    const canvas = document.getElementById("eraser-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (baseImage) ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);

    const mctx = maskCanvas.getContext("2d");
    mctx.fillStyle = "black";
    mctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

    for (const stroke of strokes) {
      drawStroke(ctx, stroke, "rgba(206, 158, 90, 0.55)");
      drawStroke(mctx, stroke, "white");
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
    if (!baseImage) return;
    e.preventDefault();
    const pt = canvasPoint(e);
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
    if (!baseImage || !document.getElementById("eraser-modal").classList.contains("open")) return;
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
    if (tool === "crop") initCropRect();
    redraw();
  }

  function undoStroke() {
    if (tool === "crop") initCropRect();
    else strokes.pop();
    redraw();
  }

  // ------------------------------ apply actions ------------------------------

  async function applyErase() {
    if (currentIndex < 0) return;
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
        method: document.getElementById("eraser-method").value,
        radius: 4,
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
      btn.textContent = "Erase";
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
          h("img", { src: Api.thumbUrl(member.path, 160), loading: "lazy" }),
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
      toast(`Deleted ${data.deleted} image${data.deleted === 1 ? "" : "s"}.`, "success");
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
      await Api.editDeleteImages([img.path]);
      toast(`Deleted "${img.filename}".`, "success");
      closeEditor();
      images = images.filter((i) => i.path !== img.path);
      renderGrid();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  return { init, setFolder: (path) => { document.getElementById("edit-folder").value = path; load(); } };
})();
