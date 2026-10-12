# Anima Studio

A single, clean workspace for the full LoRA pipeline - Anima DiT **and**
SDXL: **collect** a dataset from booru sites, the web, or your own
computer, **edit** it with a local magic eraser/Windows-Photos-style
crop tool, **caption** it (booru tags or a local WD14 auto-tagger),
**train** with every setting from your reference
`LoRA_Easy_Training_Scripts` project exposed in one form, and
**monitor** the run as it trains. A folder you work on in one tab
follows you into the next. Fully re-themeable to match your taste.

It's a small local web app - a Python/Flask backend plus a plain
HTML/JS frontend - that you run on your own machine and open in a
browser. Nothing about your images, tags, or models is sent anywhere
except the requests you explicitly make (booru search/autocomplete,
WD14 model download, and your own training backend).

## What's real here, and what it hands off

- **Collect**, **Edit**, **Caption**, and **Monitor** are fully
  self-contained. They talk directly to booru sites' public search and
  autocomplete APIs (or a page/URL/local file you give them), erase
  watermarks and crop with OpenCV/Pillow locally, run the WD14 tagger
  locally via `onnxruntime`, and read training progress straight off
  the filesystem and (optionally) TensorBoard logs.
- **Train** builds a complete, validated config for either an Anima DiT
  or SDXL run, and can export/import a `.toml` file or send the config
  straight to a **training backend** and start the run.

  That backend is the actual training engine - the sd-scripts fork with
  the custom optimizers - from `67372a/LoRA_Easy_Training_Scripts`
  (the project you shared), now bundled directly at `training_backend/`
  (see `THIRD_PARTY_NOTICES.md`). This app is a frontend for it, not a
  reimplementation of it: I don't have that engine's exact model
  architecture or its ~90 experimental optimizer implementations, and
  guessing at them would produce something that silently trains
  garbage - worse than not having it. The code itself now ships with
  Anima Studio; see "Getting the training backend" below for the one
  remaining step (installing *its* dependencies).

## Setup

**Windows:** double-click `run.bat`. **macOS/Linux:** `./run.sh`. Or
manually: `pip install -r requirements.txt && python run.py`. Opens
`http://127.0.0.1:7864`.

### Desktop app (Windows)

Two options, same app either way:

- **Native window, no build step:** `pip install pywebview` then
  `python desktop.py` - opens in its own window instead of a browser
  tab. Good for everyday use without setting up a build.
- **A real standalone Anima Studio.exe:** run `build_windows.bat` (or
  `pyinstaller anima_studio.spec --noconfirm` yourself). This has to be
  run *on Windows* - PyInstaller builds for whatever OS it's run on, it
  can't cross-compile a `.exe` from Linux/Mac. Output lands in
  `dist\Anima Studio\Anima Studio.exe`; the whole `dist\Anima Studio\`
  folder is what you'd zip up to share, not just the `.exe` alone.
  Needs the WebView2 Runtime, which ships with Windows 11 and current
  Windows 10 - if it's somehow missing, Windows will prompt for it.

The desktop window is the same Flask app as `run.py`, just without a
browser tab around it - same datasets folder, same settings, same
training backend hookup below.

### Getting the training backend

The actual training engine now ships bundled with Anima Studio at
`training_backend/` (vendored from `67372a/LoRA_Easy_Training_Scripts` -
see `THIRD_PARTY_NOTICES.md` for exactly what and its license). There's
nothing to clone anymore - Anima Studio still doesn't reimplement the
training engine itself, but you no longer need a separate download to
get it.

What's still a separate step: *that* code's own dependencies (PyTorch,
xformers, the rest of the ML stack) - a multi-GB install that has to
match your specific GPU/CUDA setup, so there's no way to freeze it into
this zip the way the bundled source code is. Two ways to run that
install:

- **From the Train tab:** the "Set up backend" card runs the installer
  right there, with its output streamed live (and a box to answer any
  questions it asks, since it's interactive).
- **From a terminal:** `setup_backend.sh` / `setup_backend.bat` run the
  exact same installer directly.

Once that's done, "Backend folder" in the Train tab is already set to
`training_backend/` by default - click "Launch backend", then "Check
connection". With "Auto-start" checked, Anima Studio launches and stops
that process itself on its own startup/shutdown - you never need a
second terminal window open.

### Optional extras

- **Auto-tagger** (Caption tab): needs `onnxruntime` (in
  requirements.txt already) plus internet the first time you click
  Download for a model.
- **Loss/metric charts** (Monitor tab): needs `pip install tensorboard`
  and a run that had logging enabled in the Train tab. Without it,
  Monitor still shows training status, sample images, and checkpoints.

## Using it

1. **Booru Explorer** - a real booru-style browser for finding characters
   worth training. It opens already filled with **random posts** (no search
   needed); switch to Trending / Latest / Top score / Most faved, or type tags
   with live autocomplete. A strip of **"Characters in this feed"** cards lets
   you jump straight into a character; the left sidebar lists the page's
   tags by category (artist / copyright / character / general / meta) with
   `+` add, `-` exclude and `?` wiki on every tag. Searching a single tag
   shows its **tag wiki** (text, aliases, post count, category, related tags,
   artist links). Click a thumbnail for the viewer (categorized tags,
   score/size/source, parent/children, pools, prev/next with the arrow keys),
   "Open on site" jumps to the booru's own page, and the **+** on a card
   selects it for **Add to dataset**. Also: pool browser, saved searches,
   a local blacklist, and a min-score filter.
   **20 sources**: Danbooru, AIBooru, ATFBooru, Gelbooru, Safebooru, Rule34,
   TBIB, Xbooru, Realbooru, HypnoHub, Konachan.com/.net, yande.re,
   Lolibooru, Sakugabooru, e621, e926, Derpibooru, Furbooru, Ponybooru.
   Not included (login/token or no public API): Sankaku, Pixiv, Twitter/X,
   Zerochan, Anime-Pictures, e-shuushuu. Adding another site that runs one
   of the supported engines is a single entry in `booru_engines.py`.
   Wiki / related tags / pools depend on what each site's API offers;
   Gelbooru-family sites have no wiki API, so those link to the site's wiki.
   Rating defaults to **Safe**.
   **Blocked sites:** Explorer > **Connection**. *Automatic* (default) looks a site
   up normally and, if your network blocks it or answers with a fake address,
   switches to encrypted DNS (Cloudflare / Google / Quad9) - this also covers
   Collect and downloads. *Test connection* shows which part works. If the
   block is by IP address or site name, DNS can't help: run a VPN, Cloudflare
   WARP (proxy mode) or Tor and enter its local port as the proxy there.
   If a site can't be reached the error panel offers Retry, a "which sites can
   I reach" check, and a proxy box (also used for downloads).
2. **Collect** - Booru search across all 20 sources above - select more than one
   to search and merge results at once - a web page, a URL list, or
   Local files to add images straight from your computer. Live tag
   suggestions as you type. Click the eye icon on any result to preview
   it full-size and select it for download right from that view.
   Prev/Next pages through results.
3. **Edit** - search by filename to find something in a big folder.
   Click any image to open the eraser: paint a watermark away (brush or
   rectangle), or switch to Crop for a resizable, draggable selection
   with corner/edge handles and aspect-ratio lock, just like Windows
   Photos. Zoom in for precision, check "Save as copy" to keep the
   original untouched, and use the overlay arrows right on the image to
   move between photos. Use **Find duplicates** to catch resizes,
   recompresses, and minor edits, and delete the ones you don't want.
4. **Caption** - edit captions per image, search by filename or tag
   content, or use the bulk tools. Switch to **Tag Frequency** to see
   every tag as a chip with its count - click one to strike it through
   and remove it from every caption instantly (click again to restore
   it to exactly the images it came from), or click the pencil icon to
   rename a tag everywhere in one step. Auto-tag with WD14 after
   downloading the model once, with a visible progress bar.
5. **Train** - (a **training bar** stays at the bottom of every tab: a small "Train" pill when idle that expands on hover with Start / Add to queue, and a full bar while training with progress, step/epoch/loss, estimated time remaining and the queue. Some trainers only report start and finish, so for those it shows elapsed time instead of an estimate. Also a **Queue** tab: "Add to queue" snapshots the current config as a job; jobs train one after another, can be reordered, cancelled, retried, and the queue pauses by itself if the backend is down) - choose **Anima (DiT)** or **SDXL** at the top of the
   Model section - the whole form adapts. Click "Use current dataset"
   to wire in the same folder with one click. Download/Load `.toml`, or
   hit **Start training** once your backend is running.
6. **Monitor** - point it at your run's output folder to see the
   latest sample images and saved checkpoints as they land, and (with
   TensorBoard installed + logging enabled) loss/learning-rate charts.

Click **Appearance** in the sidebar to pick a background and accent
color (or one of six presets) - it's saved and applied every time you
open the app.

## Rating filter (Collect tab)

Defaults to **Safe** and is a plain pass-through to each site's own
rating tag - the same filter that site's own search box uses. Nothing
here bypasses a site's own gating.

## Project layout

```
app/
  schema.py          canonical optimizer/scheduler/loss/network options
  train_config.py     Train tab settings <-> TOML for both Anima and
                      SDXL, both directions tested to round-trip exactly
  toml_writer.py       dependency-free TOML serializer
  services/
    booru.py            7 booru-family API clients + tag autocomplete
    web_images.py         page/URL image discovery
    downloader.py           concurrent download + local file import
    imaging.py                magic eraser, crop, save-copy, backups,
                               and duplicate detection (dHash)
    captions.py                caption read/write + bulk edits,
                                including exact per-image tag restore
                                and instant rename
    tagger.py                    WD14 ONNX auto-tagger, explicit
                                  progress-tracked download
    datasets.py                   folder browsing, image listing,
                                   tag stats, delete
    monitor.py                     sample images, checkpoints,
                                    TensorBoard scalar reading
    backend_client.py              HTTP client for the training backend
  routes/               Flask blueprints (collect/edit/caption/train/
                        files/monitor)
  static/               CSS + vanilla JS frontend (ActiveDataset shares
                        the working folder across tabs, theme.js drives
                        Appearance customization)
run.py / run.sh / run.bat       start Anima Studio (browser)
desktop.py                      start Anima Studio in its own window (needs pywebview)
anima_studio.spec / build_windows.bat   build a standalone Anima Studio.exe
training_backend/               bundled training engine (see THIRD_PARTY_NOTICES.md)
setup_backend.sh / .bat         install the bundled backend's own dependencies
setup_backend.sh / .bat         clone + install the real training backend
```

## Notes

- Two native-kohya algorithm choices ("lora" vs "locon" with no LyCORIS
  extras enabled) produce identical config output in this schema, so a
  loaded `.toml` that used plain "locon" will show as "lora" after
  reload - everything else round-trips exactly, checked across every
  optimizer/scheduler/loss/algorithm combination for both Anima and
  SDXL.
- The magic eraser and crop tool use classical (non-AI) processing -
  fast, run on any CPU, no model download.
- Duplicate detection uses a difference hash (dHash), robust to
  resizing, recompression, and minor color/brightness changes but won't
  catch a genuinely redrawn or heavily edited repost.

## Tests

```
pip install pytest
python -m pytest
```

The suite needs no GPU and no torch. It checks that Training-tab settings reach the trainer config,
that the dataset/caption checks match the trainer's own rules, the cache-latents/random-crop rule,
the PixAI tagger wiring, the Flask routes, and that every JS file parses. Two scheduler tests run
only when torch is installed.
