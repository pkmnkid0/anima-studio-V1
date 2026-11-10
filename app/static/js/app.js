document.addEventListener("DOMContentLoaded", async () => {
  const titles = {
    projects: ["Projects", "— one small folder per LoRA: dataset + output"],
    explore: ["Booru Explorer", "— browse any booru for character inspiration, with tag wiki"],
    collect: ["Collect", "— gather images from booru sites or the web"],
    edit: ["Edit", "— erase signatures and watermarks before captioning"],
    caption: ["Caption", "— tag your dataset before training"],
    train: ["Train", "— configure and launch an Anima LoRA run"],
    backend: ["Backend", "— install, start and monitor the training engine"],
    monitor: ["Monitor", "— watch samples, checkpoints, and loss as it trains"],
  };

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.dataset.view;
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
      const [title, sub] = titles[view];
      document.getElementById("topbar-title").innerHTML = `${title}<span class="dim">${sub}</span>`;
      // Tabs that poll in the background need to know whether anyone is
      // actually looking at them. Monitor polls every 6s and was previously
      // doing it unconditionally, which meant a hidden tab kept hitting the
      // server forever.
      document.dispatchEvent(new CustomEvent("studio:viewchange", { detail: { view } }));
    });
  });

  Collect.init();
  if (window.Explore && Explore.init) Explore.init();
  Edit.init();
  if (typeof EditAI !== "undefined") EditAI.init(Edit.bridge);
  Caption.init();
  CaptionTools.init();
  Theme.init();
  try {
    await Train.init();
  } catch (err) {
    toast(`Couldn't load training settings: ${err.message}`, "error");
  }
  Monitor.init();
  if (typeof BackendTab !== "undefined") BackendTab.init();
  if (typeof MossEnv !== "undefined") MossEnv.init();
  if (typeof KohakuEnv !== "undefined") KohakuEnv.init();
  if (typeof EditEnv !== "undefined") EditEnv.init();
  if (typeof EditModels !== "undefined") EditModels.init();
  if (typeof Updates !== "undefined") Updates.autoCheck();
  if (window.Tester && Tester.init) Tester.init(); // optional module; not shipped in this build

  // Project context last: it derives folder paths for every tab above, so
  // it has to wait until their inputs exist. Train may have failed to init,
  // hence the try/catch - the switcher still works without it.
  try {
    await AppSettings.init();   // before Context so the saved values are in place
  } catch (err) {
    console.error("app settings failed to load", err);
  }
  try {
    await Context.init();
  } catch (err) {
    console.error("project context failed to load", err);
  }

  // Best-effort initial backend reachability check, silent on failure.
  try {
    const url = document.getElementById("backend-url").value.trim();
    const result = await Api.backendPing(url);
    const dot = document.getElementById("backend-dot");
    dot.classList.add(result.reachable ? "ok" : "bad");
    document.getElementById("backend-status-text").textContent = result.reachable ? "Backend online" : "Backend offline";
  } catch {
    /* backend not running yet - fine, the user hasn't necessarily started it */
  }
});
