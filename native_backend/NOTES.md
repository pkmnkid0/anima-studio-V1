# Anima Studio's native backend

A from-scratch training backend, built to speak the exact same HTTP
contract `app/services/backend_client.py` already expects (reverse-
engineered against the vendored backend's own `main.py` /
`utils/validation.py` - see below) - so pointing the Train tab's
"Backend URL" here instead of `training_backend/` is a drop-in swap,
nothing else in Anima Studio needs to change.

## Why two different designs for the two model families

**SDXL** (`sdxl_trainer.py`): a genuinely new training loop, built on
Hugging Face's official `diffusers` + `peft` libraries. SDXL is well-
documented and diffusers' support for it is mature and widely used, so
writing new orchestration code on top of those primitives is
responsible: the risk of a subtle, hard-to-detect numerical bug is low
because the primitives themselves are already correct and tested by a
huge community.

**Anima** (`anima_trainer.py`): deliberately does NOT reimplement the
model architecture. Anima is an obscure, very new model - a Diffusion
Transformer with a Qwen3 LLM text encoder bridged through a custom
adapter, trained via rectified flow - and no mainstream library
supports it. The vendored backend at `training_backend/sd_scripts/`
does (Apache-2.0), in ~5,000 lines of real, tested implementation
(`library/anima_models.py` alone is 1,666 lines of custom attention/
rotary-embedding/adapter code). Re-deriving that from fragments with no
way to validate it would be reckless, not "from scratch" - it'd be
guessing at someone else's unpublished research architecture. Instead,
this calls the REAL, complete training pipeline
(`anima_train_network.AnimaNetworkTrainer.train()`) directly as a
Python function, in-process, instead of as a subprocess - what's
actually new here is the translation from Anima Studio's own config
shape into what that real trainer expects.

## What's actually been tested, and how

Being specific here on purpose, since "it runs" and "it's correct" are
different claims for ML training code:

- **Server contract** (`server.py`): fully tested. Validate/train/stop/
  status state transitions, error handling, and a full mocked training
  lifecycle (start, progress reporting, mid-training stop, exception
  handling) were all run and verified against a Flask test client.
- **Dataset scanning** (`dataset.py`): fully tested against real files -
  multi-subset merging, `num_repeats`, and every error path (missing
  folder, empty folder, missing `image_dir`).
- **SDXL trainer**: tested genuinely end-to-end. A tiny but
  architecturally-real SDXL model (real `UNet2DConditionModel` with
  SDXL's `addition_embed_type="text_time"` conditioning, real dual CLIP
  text encoders, real VAE) was constructed with diffusers' own classes
  (randomly initialized - no download needed) and run through the
  actual training loop: real LoRA injection via `peft`, real forward/
  backward passes, finite (non-NaN) loss across multiple steps, and a
  correctly-saved LoRA weights file verified to contain genuinely
  LoRA-named tensors. This is real verification of the mechanics, not a
  claim that any specific LoRA trained this way will produce good
  generation quality - that needs a real base model, a real dataset,
  and real training time to know.
- **Anima trainer**: the *integration* is tested end-to-end against the
  real vendored pipeline - argument translation was verified against
  the real parser (not a hand-guessed Namespace), the generated dataset
  TOML was validated against the real `ConfigSanitizer` schema, and a
  full run was traced through real dataset loading, real config
  parsing, and into the real Qwen3-loading code, stopping only at the
  expected point (a fake, nonexistent model path - there was no real
  Anima checkpoint available to test against, since those are large,
  separately-hosted downloads). This caught and fixed a real bug along
  the way: the vendored trainer's own caption-file-extension default is
  `.caption`, not `.txt` - without an explicit override, captions would
  have been silently dropped (no error, just empty captions) for every
  real Anima Studio dataset, since this app's convention is `.txt`.
  **Not tested: an actual full run against real Anima model weights.**

## The API contract (for reference)

```
POST /validate   {"args": {...}, "dataset": {"subsets": [...]}, "accelerate": {...}}
GET  /train       ?train_mode=lora&sdxl=True|flux=True|anima=True&accelerate_*
GET  /is_training -> {"training": bool, "errored": bool}
GET  /status       (extra - not in the original contract) step/loss/epoch
GET  /stop_training ?force=
GET  /check_path   ?path= -> {"exists": bool}
```

`dataset.subsets` is required by the real backend's own validator
(confirmed by reading `utils/validation.py` directly) - Anima Studio's
existing `routes/train.py` had a real, separate bug where this never
actually got included in the request body, meaning Start Training
against the vendored backend has likely never worked. That's fixed as
part of this same round of work, independent of which backend you use.

## Running it

```
pip install -r native_backend/requirements.txt   # SDXL path deps
python -m native_backend.server 8000
```

Then set the Train tab's Backend URL to `http://127.0.0.1:8000` and
click Check connection.

For Anima training specifically, run this using the SAME Python
interpreter/venv as the bundled backend
(`training_backend/sd_scripts/venv`), since that's what has its
dependencies already installed (einops, kornia, RamTorch, etc.) - this
backend doesn't reinstall those separately.

## Known gaps / not implemented

- **Flux** is not implemented in either trainer - `/train` returns 400
  for it.
- **Full fine-tuning** (non-LoRA) is not implemented - only
  `train_mode=lora`.
- **Progress reporting for Anima training** is start/finish only, not
  live per-step - the real trainer reports progress through its own
  tqdm/accelerate hooks internally, and patching those was judged too
  fragile across versions to be worth it for this; the Monitor tab's
  existing sample-image/checkpoint-folder polling still works
  regardless of which backend is running.
