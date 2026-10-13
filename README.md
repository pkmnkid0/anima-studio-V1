<div align="center">

<img src="docs/screenshots/logo.png" width="96" alt="Anima Studio logo">

# Anima Studio

**The whole LoRA pipeline in one local app.**
Collect → Edit → Caption → Train → Monitor, for **Anima (DiT)** and **SDXL**.

![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![flask](https://img.shields.io/badge/backend-Flask-000000?logo=flask)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-555)
![gpu](https://img.shields.io/badge/training-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)
![tests](https://img.shields.io/badge/tests-44%20passing-2ea44f)
![local](https://img.shields.io/badge/runs-100%25%20local-d2a45a)

<img src="docs/screenshots/train-dataset.png" alt="Anima Studio - Train tab" width="900">

</div>

---

## What it is

Anima Studio is a small local web app (Python/Flask backend, plain HTML/JS frontend) that runs on your own machine, in a browser tab or its own desktop window. A folder you work on in one tab follows you into the next, so a dataset goes from "images on a booru" to "trained LoRA" without leaving the app.

Nothing about your images, tags or models leaves your machine except the requests you explicitly make: booru search, tagger model download, and your own training backend.

```mermaid
flowchart LR
    A[Collect<br/>20 booru sources,<br/>web, local files] --> B[Edit<br/>erase, crop,<br/>de-duplicate]
    B --> C[Caption<br/>6 auto-taggers,<br/>tag tools]
    C --> D[Train<br/>Anima / SDXL,<br/>queue + presets]
    D --> E[Monitor<br/>samples, checkpoints,<br/>loss + LR charts]
```

## Feature sheet

| | Tab | What you get |
|---|---|---|
| 🔎 | **Booru Explorer** | A booru-style browser with random / trending / top-score feeds, tag autocomplete, a tag sidebar by category, tag wiki, pools, saved searches and a local blacklist. |
| ⬇️ | **Collect** | Search 20 booru sources at once (or a web page, URL list, or local files). **Smart curation** picks a balanced subset: ticked filter tags are alternatives (any one of them), then it de-duplicates and balances views, outfits and poses. |
| ✂️ | **Edit** | Paint-out watermark eraser, Windows-Photos-style crop with aspect lock, **duplicate finder** (dHash), and **multi-resolution export** (512 / 1024 / 1536 / custom copies of your dataset). |
| 🏷️ | **Caption** | **Edit tags** on any image (auto-saved) or across the whole dataset, **Tag Frequency** view (strike a tag from every caption in one click, rename everywhere), caption-variation detection, and auto-tagging. |
| 🧪 | **Train** | One form for every setting, grouped into Dataset / Model & network / Training / Output / Advanced. Presets, a job **queue**, `.toml` import/export, pre-run warnings, and a Studio tab. |
| 📈 | **Monitor** | Live step / loss / ETA, newest run's **loss and learning-rate charts** (TensorBoard), sample images and saved checkpoints. |
| ⚙️ | **Backend** | Installs, starts and stops the bundled training engine and shows its log. |

## Screenshots

> Captured from the running app with a small synthetic demo dataset.

### Collect
<img src="docs/screenshots/collect.png" alt="Collect tab" width="900">

### Edit
<img src="docs/screenshots/edit.png" alt="Edit tab" width="900">

### Caption
<img src="docs/screenshots/caption.png" alt="Caption tab" width="900">

<details>
<summary><b>Tag Frequency view</b></summary>

<img src="docs/screenshots/caption-tags.png" alt="Tag frequency" width="900">

</details>

### Train
The Training tab is split into five sub-tabs. Related options sit together, and each tick option sits next to the settings it controls.

<table>
<tr>
<td align="center"><b>Dataset</b><br><img src="docs/screenshots/train-dataset.png" alt="Train: Dataset"></td>
<td align="center"><b>Model &amp; network</b><br><img src="docs/screenshots/train-model.png" alt="Train: Model and network"></td>
</tr>
<tr>
<td align="center"><b>Training</b><br><img src="docs/screenshots/train-training.png" alt="Train: Training"></td>
<td align="center"><b>Output</b><br><img src="docs/screenshots/train-output.png" alt="Train: Output"></td>
</tr>
</table>

<details>
<summary><b>Advanced</b></summary>

<img src="docs/screenshots/train-advanced.png" alt="Advanced" width="900">

</details>

### Monitor
<img src="docs/screenshots/monitor.png" alt="Monitor tab" width="900">

### Projects and Backend
<table>
<tr>
<td align="center"><b>Projects</b><br><img src="docs/screenshots/projects.png" alt="Projects tab"></td>
<td align="center"><b>Backend</b><br><img src="docs/screenshots/backend.png" alt="Backend tab"></td>
</tr>
</table>

## Complete feature reference

Every tab, every option. (Built from the app's own screens, so it matches what you see.)

### Projects
- One folder per LoRA, with a fixed `dataset` and `output` layout created for you
- Project **type** (Character / Outfit / Style / Concept / Object / Other) and **model** (Anima / SDXL), remembered for next time
- Link an **existing dataset folder** instead of creating an empty one
- Search projects by name, type, model, trigger word or folder; most recently opened first
- Every project remembers the settings you used
- **Active project** switching fills in the folder of every tab (Collect, Edit, Caption, Train, Monitor). It never overwrites a path you typed yourself, and tells you when it left one alone

### Booru Explorer
- Opens already filled with **random posts**; one-click feeds: Random, Trending, Latest, Top score, Most faved, plus Shuffle
- **20 sources** with a source picker; **rating filter** (Safe / Sensitive / Questionable / Explicit / Any), default Safe; **minimum score** filter
- Tag search with **live autocomplete**, exclusion (`-tag`), and **Open on site** to jump to the booru's own page
- **"Characters in this feed"** cards to jump straight into a character
- Sidebar of the page's tags grouped by category (artist, copyright, character, general, meta) with **+ add, - exclude and ? wiki** on every tag
- **Tag wiki** when searching a single tag: text, aliases, post count, category, related tags, artist links
- Full-screen **viewer** with categorized tags, score, size, source, parent/children posts, pools, and **previous/next with the arrow keys** (Esc closes)
- **Pools** browser, **Saved searches**, a local **Blacklist** (posts with those tags are hidden)
- Select posts with **+** on a card, then **Add to dataset** into any destination folder (Browse, Clear selection)
- **Connection panel** for blocked sites: Automatic mode switches to encrypted DNS (Cloudflare / Google / Quad9), a **Test connection** button shows which part works, and a **proxy** setting is used for searches and downloads
- Error panel with Retry, "which sites can I reach" check and proxy box
- Per-site **API keys** (for sites that now require them), stored locally

### Collect
- Four ways in: **Booru search**, **Web page**, **URL list**, **Local files**
- **Multi-source search**: pick several boorus and the results are merged
- Live tag suggestions, results per page, rating filter, Prev / Next paging
- **Eye icon** on any result to preview full size and select it for download from there
- **Select all / Clear**, choose the destination folder, **Download selected / Download all**
- **Web page**: paste a URL and **Scan page** to list its images; **URL list**: one image URL per line
- **Local files**: pick images from your computer and add them to a dataset
- **Dataset Builder**
  - **Discover tags**: samples the character's posts and groups useful tags (outfits, views, footwear, poses, expressions, details, other) with counts
  - **Filters**: tick any tags; smart curation takes images with the character plus any one of them (normal Search still requires all)
  - **Smart curation**: target image count (1 to 1000), search pool (up to 5000), minimum resolution; removes low-quality and duplicate images and balances coverage, with a coverage readout (back, side, full body, footwear, outfits, actions, expressions, details)
  - Notice when Danbooru's anonymous 2-tag limit applies (extra tags become a filter on each result)
- **Smart Dataset Manager** (Dataset Health): see below

### Dataset Health (Collect > Smart Dataset Manager)
- **Health score** 0 to 100 with a letter grade
- **Quick** analysis or **Full** analysis (adds near-duplicate scan using perceptual hashing)
- **Distribution** charts: resolutions and aspect ratios
- **Balance analyzer**: tag-group distribution with dominance warnings
- **Dataset coverage map**: how well each aspect of the concept is covered, with the evidence images one click away
- **Cleanup plan**: suggested keep / remove / review for every image, **Select duplicates only**, **Remove selected**, **Save selection** to another folder. Nothing is deleted without your confirmation
- Drop a folder or images onto the path row

### Edit
- Load any folder; **search by filename**; click an image to open the editor; **arrow keys** (and on-image arrows) move to the next or previous image
- **Magic eraser**: paint over a watermark or signature with a **brush** (adjustable size) or a **rectangle**, choose the **Telea** or **Navier-Stokes** method, then Erase; **Undo** and **Clear**
- **Crop** with a resizable, draggable selection and corner/edge handles, **aspect lock** (Free, 1:1, 4:3, 3:4, 16:9, 9:16), then Apply crop
- **Upscale** the crop: Off, 1.5x, 2x, 3x, 4x with **Lanczos** (sharp), **Bicubic** (soft) or **AI (UltraSharp)** (optional model download and one-click dependency install)
- **Zoom** in/out and Fit
- **Save as copy** keeps the original untouched; **Revert to original** restores from automatic backups
- **Send crop to folder / subset** files the crop straight into another dataset folder
- **Delete** removes an image and its caption file
- **Find duplicates**: similarity threshold slider (resizes, recompressions, minor edits), then review and delete
- **Multi-resolution export**: 512, 1024, 1536 or a custom size; copies the whole folder tree at each size with captions alongside, one folder per resolution inside `resized images/`, with a progress status. Originals are never touched
- **Subset sidebar** (also in Caption): collapsible groups "Your datasets" and "Inside <folder>", **+ Add subset** from anywhere on disk, **Hide empty folders**, hide or remove a row with the x, "Show N hidden"

### Caption
- Stats bar: images, captioned, uncaptioned, unique tags; three views: **Gallery**, **Tag Frequency**, **Cleanup**
- **Edit tags on any image:** each image has its own caption box. Type to add, remove, reorder or reword tags and it **saves automatically** as you type. Search by filename or tag content to find the image you want
- **Edit tags across the whole dataset** without opening every image: add, remove, find and replace, rename (pencil in Tag Frequency), strike, standardise variants, and convert underscores to spaces
- **Auto-tagging** with 6 models: WD ViT v3, WD SwinV2 v3, WD ConvNeXt v3, WD EVA02-Large v3, WD EVA02 2026 canary and **PixAI tagger v1.0**
  - One-click model download with progress, then fully offline; one-click dependency install for the timm/PixAI models
  - General and character **threshold** sliders; apply to **images without captions** or **all images (overwrite)**
- **Bulk edit**: add a tag to all, **prepend a trigger word** (always the first tag), remove a tag from all, find and replace a tag, convert underscores to spaces
- **Tag Frequency**: every tag as a chip with its count, sort by count or A to Z, filter box; **click a tag to strike it from every caption** (click again to restore it to exactly the images it came from); **pencil** to rename a tag everywhere
- **Tag variation detection** (Cleanup): finds tags meaning the same thing (white_hair vs silver_hair), tick groups and **Standardize selected**
- Top tags panel

### Train
- **Anima (DiT)** or **SDXL** at the top; the whole form adapts
- **Presets**: **Save preset** stores your whole configuration under a name, and **Load a preset** brings it back
- **Before you train** panel lists problems by name (missing model paths, no dataset, a caption file that is missing or empty and what it should be called)
- Sub-tabs: Dataset, Model & network, Training, Output, Advanced, Studio, Queue, Config preview

**Dataset**
- Add any number of **dataset folders**, **Use current dataset**, **Add all subsets**
- Per folder: image folder, **repeats**, **caption extension**, role (**regularization** or **validation** images), **shuffle caption tags**, **keep first N tags fixed**, **protected tags file**, **caption dropout** (rate, per-tag), **token warmup**, **flip**, **color augment**, **random crop** (with padding), **gamma augment**, **face crop range**, **paired / control-net data**
- **Aspect-ratio bucketing** with min / max resolution, step, **don't upscale small images**, and **multi-resolution training**

**Model & network**
- Model files: **DiT**, **Qwen3 text encoder**, **VAE**, **T5 tokenizer**, max token lengths, **timestep sampling**, **sigmoid scale**, **discrete flow shift**, **VAE chunk size**, **blocks to swap**, **disable VAE cache**, **flash attention**, **split attention**, **Unsloth offload checkpointing**
- Network: **17 algorithms** (LoRA, LoCon, LyCORIS variants and more), LyCORIS preset, **rank and alpha**, min/max timestep, train target, **separate conv dim and alpha**, rank and module dropout, constraint, **Tucker decomposition**, **train norm layers**, **DoRA**, rescaled, **bypass mode**, network dropout, IP noise gamma, **DyLoRA** unit, **cache text encoder outputs** (also to disk), **LoRA-FA**
- **Per-block** LR weights (down / mid / up), dims, alphas, conv dims and conv alphas
- **Optimizer** (113 options, 5 recommended), learning rate, **max grad norm**, **separate UNet / text-encoder LR**, optimizer arguments
- **LR scheduler** (14) with warmup ratio, **warm up from zero**, cycles, poly power, min LR, gamma, curve shape, decay ratio and type, timescale, min LR ratio, initial LR
- **Loss** (14 types) with Huber c and schedule, **scale weight norms**, **min-SNR gamma**, **zero terminal SNR**, **masked loss**

**Training**
- **Length** by epochs or steps, **batch size**, **seed**, **gradient accumulation**
- **Resolution** with non-square support
- **Precision and attention**: mixed precision, full fp16, full bf16, SDPA / xFormers / none
- **Memory**: gradient checkpointing, low RAM, high VRAM, no half VAE
- **Caching and loading**: cache latents (also to disk), cache text-encoder outputs, dataloader workers
- Prior loss weight, keep-tokens separator, training comment, protected tags file, external VAE override
- **Noise offset**, **pyramid (multi-res) noise** with iterations and discount, **EDM2 loss weighting** (optimizer, LR, channels, warmup, constant phase, initial weights)

**Output**
- Output folder, custom output name, **save precision**, **save format** (safetensors), **resume from state**
- **Save checkpoints periodically** by epochs or steps, **limit saved checkpoints**, save ratio, write tag-occurrence file, **save a copy of the config**, save optimizer/training state, limit saved states
- **Logging**: TensorBoard or Wandb, log directory (defaults to `<output>/logs`), prefix, run name, tracker project name
- **Sample images during training**: sampler (16), every N epochs or steps, sample prompts

**Advanced**
- **Multi-GPU** (accelerate): number of GPUs, main process port
- **Experimental**: VAE batch size, custom VAE scale/shift, VAE reflection padding, zero conditioning dropout, **debiased estimation**
- **Extra arguments** field for anything not on a form; unknown options are forwarded to the trainer

**Studio**
- **Images and steps calculator**: images, repeats, epochs, project type, batch size, gradient accumulation and steps per second give total steps and an **estimated duration**
- **Training history**: every run is recorded; **Compare selected** runs side by side
- **LoRA packaging**: pick a `.safetensors`, read its metadata to auto-fill the form, add package name, trigger words, description and dataset folder, and bundle a shareable, self-describing folder (and zip)

**Queue**
- **Add current config to queue** (a snapshot), jobs train one after another
- Reorder (up / down / top), rename, cancel, retry, **Pause queue**, **Clear finished**; pauses itself if the backend is down

**Config preview**
- Live TOML preview, **Download .toml**, **Load .toml** (upload or from disk), **Save .toml to disk**; settings round-trip exactly

**Training bar (every tab)**
- A **Train** pill that expands on hover with **Start training** and **Add to queue**; while training it shows progress, step / epoch / loss, estimated time remaining (or elapsed time for trainers that only report start and finish) and the queue
- **Stop** button; from another tab, Start asks you to confirm it will use the Train tab's settings
- Error panel with a plain-language hint (out of memory, missing package, unknown optimizer) and a **Copy** button for the log

### Monitor
- Output folder and logging folder (the log folder falls back to `<output>/logs`)
- **Status** light, **step / epoch / loss** progress bar
- **Loss and metric charts** for the newest run only, with y-range in each title and clean scientific notation for tiny values; clears and re-checks when a new run starts
- **Latest samples** grid with click-to-zoom, **checkpoints** list with size and age
- **Auto-refresh** that only polls while the tab is open, plus Refresh now

### Backend
- Three-step status: **Installed**, **Process running**, **Connected**
- **Install dependencies** with live log and an input box to answer the installer's questions; nothing installs automatically
- **Start / Stop**, **Auto-start with Anima Studio**, editable start command, backend URL, **Check connection**, and a backend log

### App-wide
- **Appearance**: background colour, accent colour, six presets, Reset to default, saved between launches
- **App settings**: Collect / Edit / Caption / Explorer settings are saved automatically per project and globally, and as **named presets**; secrets and project folders are never captured
- **Desktop window** via pywebview, **standalone `.exe`** build, or plain browser tab
- Folder browser with new-folder button, drag and drop of folders and images
- **Warnings by name**: a missing or misnamed caption file is reported with the exact name it should have (`a.png` needs `a.txt`), and the trainer log prints how many images have captions
- **Run it once, no surprise installs**: opening the app never downloads anything; `run.bat` installs its own dependencies once (`run.bat --update` to refresh)
- Local-only: nothing leaves your machine except searches, model downloads and your own backend
- Test suite (44 tests, no GPU needed)

## Specifications

| | |
|---|---|
| **Models** | Anima (DiT, with Qwen3 text encoder and Qwen-Image VAE) and SDXL |
| **Training engine** | Vendored [sd-scripts](https://github.com/kohya-ss/sd-scripts) fork with custom optimizers and schedulers, from `67372a/LoRA_Easy_Training_Scripts` (see `THIRD_PARTY_NOTICES.md`) |
| **Network types** | 17 algorithms (LoRA, LoCon, LyCORIS variants and more) |
| **Optimizers** | 113 (5 recommended defaults), including CAME, Prodigy, ScheduleFree and `adv_optm` |
| **LR schedulers** | 14, including Rex annealing with warm restarts (cycle length is set automatically) |
| **Loss types** | 14, plus min-SNR, debiased estimation, masked loss, noise offset and Huber |
| **Sample samplers** | 16 |
| **Dataset sources** | 20 booru sites (Danbooru, Gelbooru, Safebooru, Konachan, yande.re, e621 and more), web pages, URL lists, local files |
| **Auto-taggers** | WD ViT / SwinV2 / ConvNeXt / EVA02-Large v3, EVA02 2026 canary, and **PixAI tagger v1.0** |
| **Image tools** | Magic eraser, crop with AI or classical upscale, duplicate detection, multi-resolution export |
| **Monitoring** | Progress, loss and learning-rate charts (TensorBoard), samples, checkpoints |
| **Run control** | Queue with reorder / cancel / retry, presets, `.toml` import and export |
| **Packaging** | `run.bat` / `run.sh`, `desktop.py` (own window via pywebview), or a standalone `.exe` via `build_windows.bat` |
| **Frontend** | Vanilla JS, no build step, themeable via Appearance |
| **Tests** | 44 passing, 2 skipped without torch; needs no GPU |

## Quick start

**Windows:** double-click `run.bat`. **macOS / Linux:** `./run.sh`.
Or manually:

```bash
pip install -r requirements.txt
python run.py            # opens http://127.0.0.1:7864
```

On first run, open the **Backend** tab and press **Install** once. This sets up the bundled training engine's own environment (PyTorch and the rest of the ML stack, a multi-GB download that has to match your GPU). Nothing is installed automatically. After that the backend starts with the app.

Optional: `pip install tensorboard` for loss charts, `pip install pywebview` then `python desktop.py` for a native window.

## Notes on training quality

- The bundled trainer includes a fix for Rex annealing with warm restarts: its cycle length defaulted to one step, which collapsed the learning rate to nearly zero. The app now sets it to the full run length.
- Captions are read exactly the way the trainer reads them (`image.png` needs `image.txt`). The Train tab warns by name when one is missing.
- With "Cache latents" on, random crop and colour/gamma augmentation are dropped, because they can't apply to cached latents. The UI says so.

## Project layout

```
app/                 Flask app: routes/, services/, schema, TOML config, static/ (JS + CSS), templates/
native_backend/      Trainer server that drives the training engine
training_backend/    Bundled sd-scripts fork (own venv, installed from the Backend tab)
tests/               pytest suite (no GPU needed)
docs/                GUIDE.md (full usage guide) and screenshots
run.py / run.bat / run.sh   start the app          desktop.py   native window
anima_studio.spec / build_windows.bat               standalone Windows build
```

## Docs

- [`docs/GUIDE.md`](docs/GUIDE.md) - full usage guide, each tab in detail
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) - vendored code and licences

## Tests

```bash
pip install pytest
python -m pytest
```

## Licences

Anima Studio bundles third-party code; see `THIRD_PARTY_NOTICES.md`. Models (Anima, Qwen3, VAEs, taggers) have their own licences, so check them before any commercial use.
