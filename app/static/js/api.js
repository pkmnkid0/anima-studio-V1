// Thin fetch wrapper. Every call returns parsed JSON and throws a
// readable Error on failure so callers can just try/catch + toast.
const Api = (() => {
  async function request(method, url, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    let resp;
    try {
      resp = await fetch(url, opts);
    } catch (err) {
      throw new Error(`Network error calling ${url}: ${err.message}`);
    }
    let data = null;
    const text = await resp.text();
    if (text) {
      try { data = JSON.parse(text); } catch { data = null; }
    }
    if (!resp.ok) {
      // Flask HTML error pages (500s) have no JSON body - skim a text
      // snippet so the toast names the real error instead of "HTTP 500".
      const fallback = (text || "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 300);
      const msg = (data && (data.error || data.message)) || fallback || `HTTP ${resp.status}`;
      const err = new Error(msg);
      if (data && data.needs_credentials) err.needsCredentials = data.needs_credentials;
      throw err;
    }
    return data;
  }

  const get = (url) => request("GET", url);
  const post = (url, body) => request("POST", url, body ?? {});
  const put = (url, body) => request("PUT", url, body ?? {});
  const del = (url) => request("DELETE", url);

  return {
    // Collect
    collectSearch: (body) => post("/api/collect/search", body),
    collectScanPage: (body) => post("/api/collect/scan_page", body),
    collectFromUrls: (body) => post("/api/collect/from_urls", body),
    collectDownload: (body) => post("/api/collect/download", body),
    collectAutocomplete: (source, query, limit = 8) =>
      get(`/api/collect/autocomplete?source=${encodeURIComponent(source)}&query=${encodeURIComponent(query)}&limit=${limit}`),
    collectCredentials: () => get("/api/collect/credentials"),
    collectSaveCredentials: (source, apiKey, userId) =>
      post("/api/collect/credentials", { source, api_key: apiKey, user_id: userId }),

    // Caption
    captionImages: (folder) => get(`/api/caption/images?folder=${encodeURIComponent(folder)}`),
    captionStats: (folder) => get(`/api/caption/stats?folder=${encodeURIComponent(folder)}`),
    tagFrequency: (folder) => get(`/api/caption/tag_frequency?folder=${encodeURIComponent(folder)}`),
    captionSave: (imagePath, caption) => post("/api/caption/save", { image_path: imagePath, caption }),
    bulkAddTag: (folder, tag, position) => post("/api/caption/bulk/add_tag", { folder, tag, position }),
    bulkRemoveTag: (folder, tag) => post("/api/caption/bulk/remove_tag", { folder, tag }),
    bulkRemoveTags: (folder, tags) => post("/api/caption/bulk/remove_tags", { folder, tags }),
    addTagToImages: (imagePaths, tag, position = "end") =>
      post("/api/caption/bulk/add_tag_to_images", { image_paths: imagePaths, tag, position }),
    bulkFindReplace: (folder, find, replace, wholeTagOnly) =>
      post("/api/caption/bulk/find_replace", { folder, find, replace, whole_tag_only: wholeTagOnly }),
    bulkUnderscores: (folder) => post("/api/caption/bulk/underscores", { folder }),
    taggerModels: () => get("/api/caption/tagger/models"),
    collectTagOptions: (body) => post("/api/collect/tag_options", body),
    collectSmartDataset: (body) => post("/api/collect/smart_dataset", body),
    taggerDownload: (model) => post("/api/caption/tagger/download", { model }),
    taggerDownloadStatus: (model) => get(`/api/caption/tagger/download_status?model=${encodeURIComponent(model)}`),
    taggerInstallDeps: () => post("/api/caption/tagger/install_deps"),
    taggerInstallDepsStatus: () => get("/api/caption/tagger/install_deps_status"),
    autotag: (body) => post("/api/caption/autotag", body),
    nlModels: () => get("/api/caption/nl/models"),
    hfTokenStatus: () => get("/api/caption/hf_token"),
    hfTokenSet: (token) => post("/api/caption/hf_token", { token }),
    nlDownload: (model) => post("/api/caption/nl/download", { model }),
    nlDownloadStatus: (model) => get(`/api/caption/nl/download_status?model=${encodeURIComponent(model)}`),
    nlDeps: (force, model) => get(`/api/caption/nl/deps?${force ? "force=1&" : ""}model=${encodeURIComponent(model || "")}`),
    nlEnvStatus: () => get("/api/caption/nl/env/status"),
    nlEnvStart: () => post("/api/caption/nl/env/start"),
    nlEnvCancel: () => post("/api/caption/nl/env/cancel"),
    nlInstallDeps: () => post("/api/caption/nl/install_deps"),
    nlInstallDepsStatus: () => get("/api/caption/nl/install_deps_status"),
    nlPreview: (body) => post("/api/caption/nl/preview", body),
    nlRun: (body) => post("/api/caption/nl/run", body),
    nlRunStatus: () => get("/api/caption/nl/run_status"),
    nlCancel: () => post("/api/caption/nl/cancel"),

    // Train
    trainSchema: () => get("/api/train/schema"),
    trainBuild: (ui) => post("/api/train/build", ui),
    parseToml: (text) => post("/api/train/parse_toml", { text }),
    loadTomlFromDisk: (path) => post("/api/train/load_toml_from_disk", { path }),
    saveTomlToDisk: (ui, dir, filename) => post("/api/train/save_toml_to_disk", { ui, dir, filename }),
    listPresets: () => get("/api/train/presets"),
    loadPreset: (name) => get(`/api/train/presets/${encodeURIComponent(name)}`),
    savePreset: (name, ui) => post(`/api/train/presets/${encodeURIComponent(name)}`, ui),
    deletePreset: (name) => del(`/api/train/presets/${encodeURIComponent(name)}`),
    backendPing: (url) => get(`/api/train/backend/ping?url=${encodeURIComponent(url)}`),
    backendStart: (ui, backendUrl) => post("/api/train/backend/start", { ui, backend_url: backendUrl }),
    backendStatus: (url) => get(`/api/train/backend/status?url=${encodeURIComponent(url)}`),
    backendVerify: () => get("/api/train/backend/verify"),
    mossProfiles: (vramGb) => get("/api/train/moss/profiles" + (vramGb ? `?vram_gb=${encodeURIComponent(vramGb)}` : "")),
    mossSetupStatus: () => get("/api/train/moss/setup/status"),
    mossSetupStart: () => post("/api/train/moss/setup/start", {}),
    mossSetupCancel: () => post("/api/train/moss/setup/cancel", {}),
    mossVerify: () => get("/api/train/moss/verify"),
    kohakuProfiles: (vramGb) => get("/api/train/kohaku/profiles" + (vramGb ? `?vram_gb=${encodeURIComponent(vramGb)}` : "")),
    kohakuSetupStatus: () => get("/api/train/kohaku/setup/status"),
    kohakuSetupStart: () => post("/api/train/kohaku/setup/start", {}),
    kohakuSetupCancel: () => post("/api/train/kohaku/setup/cancel", {}),
    kohakuVerify: () => get("/api/train/kohaku/verify"),
    kohakuCheck: (body) => post("/api/train/kohaku/check", body),
    mossModelStatus: (path) => get("/api/train/moss/model/status" + (path ? `?path=${encodeURIComponent(path)}` : "")),
    mossModelDownload: (dest) => post("/api/train/moss/model/download", { dest: dest || "" }),
    mossModelCancel: () => post("/api/train/moss/model/cancel", {}),
    queueList: () => get("/api/train/queue"),
    queueAdd: (name, ui, backendUrl) => post("/api/train/queue/add", { name, ui, backend_url: backendUrl }),
    queueAction: (id, action, body) => post(`/api/train/queue/${id}/${action}`, body || {}),
    queueRemove: (id) => del(`/api/train/queue/${id}`),
    queuePause: () => post("/api/train/queue/pause", {}),
    queueResume: () => post("/api/train/queue/resume", {}),
    queueClear: () => post("/api/train/queue/clear_finished", {}),
    backendStop: (backendUrl) => post("/api/train/backend/stop", { backend_url: backendUrl }),
    backendProcessSettings: () => get("/api/train/backend/process/settings"),
    backendProcessSaveSettings: (dir, command, autostart) =>
      post("/api/train/backend/process/settings", { dir, command, autostart }),
    backendProcessStatus: () => get("/api/train/backend/process/status"),
    backendProcessStart: (dir, command) => post("/api/train/backend/process/start", { dir, command }),
    backendProcessStop: () => post("/api/train/backend/process/stop"),
    backendSetupStart: () => post("/api/train/backend/setup/start", {}),
    backendSetupStatus: () => get("/api/train/backend/setup/status"),
    backendSetupInput: (text) => post("/api/train/backend/setup/input", { text }),

    // Files
    browse: (path) => get(`/api/files/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`),
    browseFiles: (path, ext) => {
      const params = [];
      if (path) params.push(`path=${encodeURIComponent(path)}`);
      if (ext) params.push(`ext=${encodeURIComponent(ext)}`);
      return get(`/api/files/browse${params.length ? `?${params.join("&")}` : ""}`);
    },
    packageInspect: (path) => get(`/api/train/package/inspect?path=${encodeURIComponent(path)}`),
    mkdir: (path) => post("/api/files/mkdir", { path }),
    listSubsets: () => get("/api/files/subsets"),
    addSubset: (path) => post("/api/files/subsets/add", { path }),
    removeSubset: (path) => post("/api/files/subsets/remove", { path }),
    thumbUrl: (path, size = 240) => `/api/files/thumbnail?path=${encodeURIComponent(path)}&size=${size}`,
    imageUrl: (path, bust) => `/api/files/image?path=${encodeURIComponent(path)}${bust ? `&t=${bust}` : ""}`,

    // Edit (magic eraser)
    editImages: (folder) => get(`/api/edit/images?folder=${encodeURIComponent(folder)}`),
    editInpaint: (body) => post("/api/edit/inpaint", body),
    editRevert: (imagePath) => post("/api/edit/revert", { image_path: imagePath }),
    editCrop: (body) => post("/api/edit/crop", body),
    resizeStart: (folder, resolutions) => post("/api/edit/resize/start", { folder, resolutions }),
    resizeStatus: () => get("/api/edit/resize/status"),
    upscalerStatus: () => get("/api/edit/upscaler/status"),
    upscalerDownload: () => post("/api/edit/upscaler/download"),
    upscalerDownloadStatus: () => get("/api/edit/upscaler/download_status"),
    upscalerInstallDeps: () => post("/api/edit/upscaler/install_deps"),
    upscalerInstallDepsStatus: () => get("/api/edit/upscaler/install_deps_status"),
    editDuplicates: (folder, threshold) => get(`/api/edit/duplicates?folder=${encodeURIComponent(folder)}&threshold=${threshold}`),
    editDeleteImages: (imagePaths) => post("/api/edit/delete_images", { image_paths: imagePaths }),

    // Edit: AI tools (IOPaint / lama-cleaner engine in its own environment)
    aiCatalog: () => get("/api/edit/ai/catalog"),
    aiStatus: () => get("/api/edit/ai/status"),
    aiSetupStatus: () => get("/api/edit/ai/setup/status"),
    aiSetupStart: () => post("/api/edit/ai/setup/start", {}),
    aiSetupCancel: () => post("/api/edit/ai/setup/cancel", {}),
    aiVerify: () => get("/api/edit/ai/verify"),
    aiEngineStop: () => post("/api/edit/ai/engine/stop", {}),
    aiSettings: (body) => post("/api/edit/ai/settings", body),
    aiRun: (body) => post("/api/edit/ai/run", body),
    aiJob: (id) => get(`/api/edit/ai/job?id=${encodeURIComponent(id)}`),
    aiJobCancel: (id) => post("/api/edit/ai/job/cancel", { id }),
    aiResultUrl: (token) => `/api/edit/ai/result/${encodeURIComponent(token)}?t=${Date.now()}`,
    aiDiscard: (token) => fetch(`/api/edit/ai/result/${encodeURIComponent(token)}`, { method: "DELETE" }).catch(() => {}),
    aiCommit: (body) => post("/api/edit/ai/commit", body),
    maskAdjust: (mask, op, amount) => post("/api/edit/mask/adjust", { mask, op, amount }),
    aiBatchStart: (body) => post("/api/edit/ai/batch/start", body),
    aiBatchStatus: () => get("/api/edit/ai/batch/status"),
    updateStatus: () => get("/api/update/status"),
  updateSettings: (body) => post("/api/update/settings", body),
  updateCheck: () => post("/api/update/check", {}),
  updateApply: (info) => post("/api/update/apply", info),
  updateProgress: () => get("/api/update/progress"),
  updateRestart: () => post("/api/update/restart", {}),
  editHistory: (body) => post(`/api/edit/history/${body.step}`, { image_path: body.image_path }),
    aiModelsCatalog: () => get("/api/edit/ai/catalog"),
    aiModelsDownload: (body) => post("/api/edit/ai/models/download", body),
    aiModelsStatus: () => get("/api/edit/ai/models/download/status"),
    aiModelsCancel: () => post("/api/edit/ai/models/download/cancel", {}),
    aiBatchCancel: () => post("/api/edit/ai/batch/cancel", {}),

    // Monitor
    monitorSamples: (outputDir) => get(`/api/monitor/samples?output_dir=${encodeURIComponent(outputDir)}`),
    monitorCheckpoints: (outputDir) => get(`/api/monitor/checkpoints?output_dir=${encodeURIComponent(outputDir)}`),
    monitorScalars: (loggingDir) => get(`/api/monitor/scalars?logging_dir=${encodeURIComponent(loggingDir)}`),
    monitorStatus: (url) => get(`/api/monitor/status?url=${encodeURIComponent(url)}`),
    monitorGpu: () => get("/api/monitor/gpu"),
    monitorProgress: (url) => get(`/api/monitor/progress?url=${encodeURIComponent(url)}`),

    // Projects
    projectsList: (q) => get(`/api/projects${q ? `?q=${encodeURIComponent(q)}` : ""}`),
    projectsCreate: (name, projectType, modelType, datasetFolder) =>
      post("/api/projects", { name, project_type: projectType, model_type: modelType, dataset_folder: datasetFolder || "" }),
    projectsOpen: (id) => post(`/api/projects/${encodeURIComponent(id)}/open`, {}),
    appSettingsGet: () => get("/api/appsettings"),
    appSettingsSaveCurrent: (values) => put("/api/appsettings/current", { values }),
    appSettingsSaveOptions: (opts) => put("/api/appsettings/options", opts),
    appSettingsSavePreset: (name, values) => post("/api/appsettings/presets", { name, values }),
    appSettingsDeletePreset: (name) => del(`/api/appsettings/presets/${encodeURIComponent(name)}`),
    projectsGet: (id) => get(`/api/projects/${encodeURIComponent(id)}`),
    projectsUpdate: (id, fields) => put(`/api/projects/${encodeURIComponent(id)}`, fields),
    projectsDelete: (id) => del(`/api/projects/${encodeURIComponent(id)}`),
    projectsTypes: () => get("/api/projects/types"),

    // Dataset analysis
    analyzeDataset: (folder, deep) => post("/api/analyze/dataset", { folder, deep }),
    analyzeBalance: (folder) => post("/api/analyze/balance", { folder }),
    analyzeCleanupPlan: (folder) => post("/api/analyze/cleanup/plan", { folder }),
    analyzeCleanupApply: (folder, decisions) =>
      post("/api/analyze/cleanup/apply", { folder, decisions }),
    captionUndoStatus: (folder) => get(`/api/caption/undo/status?folder=${encodeURIComponent(folder)}`),
    captionUndo: (folder) => post("/api/caption/undo", { folder }),
    captionEditImages: (folder, imagePaths, add, remove, position = "end") =>
      post("/api/caption/bulk/images", { folder, image_paths: imagePaths, add, remove, position }),
    blacklistGet: () => get("/api/caption/blacklist"),
    blacklistSave: (tags, applyAfterTagging) => put("/api/caption/blacklist", { tags, apply_after_tagging: applyAfterTagging }),
    blacklistApply: (folder) => post("/api/caption/bulk/apply_blacklist", { folder }),
    pruneScan: (folder) => post("/api/caption/prune/scan", { folder }),
    pruneApply: (folder, pairs) => post("/api/caption/prune/apply", { folder, pairs }),
    dedupeTags: (folder) => post("/api/caption/bulk/dedupe", { folder }),
    autoSortTags: (folder, keepFirst, pinned) => post("/api/caption/bulk/auto_sort", { folder, keep_first: keepFirst, pinned }),
    cooccurrence: (folder, tag, minCount = 3) => post("/api/caption/cooccurrence", { folder, tag, min_count: minCount }),
    captionVariations: (folder) => post("/api/caption/variations", { folder }),
    captionStandardize: (folder, mapping) => post("/api/caption/standardize", { folder, mapping }),
    trainCalculator: (body) => post("/api/train/calculator", body),
    trainHistory: (limit) => get(`/api/train/history${limit ? `?limit=${limit}` : ""}`),
    trainHistoryRecord: (run) => post("/api/train/history", run),
    trainHistoryDelete: (runId) => del(`/api/train/history/${encodeURIComponent(runId)}`),
    trainHistoryCompare: (runIds) => post("/api/train/history/compare", { run_ids: runIds }),
    captionRegionTags: (body) => post("/api/caption/region_tags", body),
    slimAnalyze: (path, dtype) => get(`/api/train/slim/analyze?path=${encodeURIComponent(path)}&dtype=${encodeURIComponent(dtype || "keep")}`),
    slimLora: (body) => post("/api/train/slim", body),
    loraMetadata: (path) => get(`/api/train/metadata?path=${encodeURIComponent(path)}`),
    loraStripMetadata: (body) => post("/api/train/metadata/strip", body),
    trainPackage: (body) => post("/api/train/package", body),
  };
})();