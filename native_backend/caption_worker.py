"""Caption worker: write natural-language captions with a local vision-language model.

This runs as its own process, launched by `app/services/captioner.py`, for two
reasons: the model's memory (several GB of VRAM) is released the moment the
batch ends instead of sitting inside the Anima Studio process, and the imports
that need PyTorch never touch the app's own interpreter.

It is started with whichever interpreter has the caption dependencies - normally
the backend venv, which already has PyTorch - so a working training backend
needs no second multi-GB download:

    python -m native_backend.caption_worker <job.json>

job.json:
    {"repo": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
     "kind": "chat",
     "family": "llava" | "qwen2vl" | "qwen3vl",
     "system": "You are a helpful image captioner.",   # optional system turn
     "prompt": "Describe this image.",
     "max_new_tokens": 320,
     "size_gb": 7.5,                         # bf16 weight size, drives the precision choice
     "skip_modules": ["visual", "lm_head"],  # kept in 16-bit when quantizing
     "vram_profile": "auto" | "8" | "12" | "16" | "24",
     "quant": "auto" | "4bit" | "8bit" | "bf16",
     "images": [{"id": "<image path>"}]}

Output, one line per image, flushed as it goes:

    @@READY   {"device": "cuda", "loader": "AutoModelForImageTextToText", "quant": "4bit"}
    @@CAPTION {"id": "<image path>", "caption": "..."}
    @@CAPTION {"id": "<image path>", "error": "..."}
    @@DONE    {"count": 12, "error": ""}

Nothing here is allowed to be fatal for a single image: a model that cannot
read one file should still caption the rest.
"""
from __future__ import annotations

import json
import sys
import traceback


def _emit(tag: str, payload: dict) -> None:
    print(f"@@{tag} {json.dumps(payload)}", flush=True)


def _pick_device(torch) -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:                                     # noqa: BLE001 - driver trouble must not be fatal
        pass
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and getattr(mps, "is_available", lambda: False)():
        return "mps"
    return "cpu"


# Mirrors app/services/captioner.estimated_vram(): bf16 weights, ~0.55x in 8-bit
# and ~0.30x in 4-bit (the vision tower and lm_head stay 16-bit), plus a fixed
# runtime overhead. Kept in sync by tests/test_caption_quant.py.
_OVERHEAD_GB = 1.6
_HEADROOM_GB = 1.0
_FACTOR = {"4bit": 0.30, "8bit": 0.55, "bf16": 1.0}

# Longest image side the processor may feed the vision tower. A 4K illustration
# would otherwise turn into thousands of vision tokens and overflow an 8 GB card
# on the first image; captions do not need that resolution.
_MAX_PIXELS_BY_VRAM = {8: 640 * 28 * 28, 12: 900 * 28 * 28, 16: 1200 * 28 * 28}
_MAX_PIXELS_DEFAULT = 1600 * 28 * 28


def _pick_quant(kind: str, size_gb: float, quant: str, device: str, vram_gb: float) -> str | None:
    """Return "4bit"/"8bit" to load quantized, or None for the native dtype.

    "auto" picks the highest precision that leaves headroom on this card:
    bf16, then 8-bit, then 4-bit. Quantization needs CUDA (bitsandbytes), so
    anything else loads in the native dtype.
    """
    if device != "cuda":
        return None
    if quant in ("4bit", "8bit"):
        return quant
    if quant == "bf16" or not size_gb:
        return None
    for mode in ("bf16", "8bit", "4bit"):
        if size_gb * _FACTOR[mode] + _OVERHEAD_GB + _HEADROOM_GB <= vram_gb:
            return None if mode == "bf16" else mode
    return "4bit"


def _vram_budget(torch, device: str, profile: str) -> float:
    """GB the loader plans for: the chosen profile, else the card's real size."""
    if device != "cuda":
        return 0.0
    try:
        if str(profile).isdigit():
            return float(profile)
    except (TypeError, ValueError):
        pass
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:                                     # noqa: BLE001 - advisory only
        return 0.0


def _load(repo: str, kind: str, torch, device: str, quant: str = "auto", size_gb: float = 0.0,
          skip_modules: list | None = None, vram_gb: float = 0.0, family: str = ""):
    """Load a processor + model, trying the general auto-classes in turn.

    Returns (processor, model, loader class name, quantization).
    """
    import transformers

    proc_kw = {}
    if family in ("qwen2vl", "qwen3vl", "qwen35"):
        budget = int(vram_gb) if vram_gb else 0
        pix = next((v for k, v in sorted(_MAX_PIXELS_BY_VRAM.items()) if budget and budget <= k),
                   _MAX_PIXELS_DEFAULT)
        proc_kw["max_pixels"] = pix
    try:
        processor = transformers.AutoProcessor.from_pretrained(repo, trust_remote_code=False, **proc_kw)
    except TypeError:
        processor = transformers.AutoProcessor.from_pretrained(repo, trust_remote_code=False)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    mode = _pick_quant(kind, size_gb, quant, device, vram_gb)
    bnb_config = None
    if mode:
        try:
            import bitsandbytes  # noqa: F401
        except BaseException as exc:                      # noqa: BLE001 - a broken CUDA build raises more than ImportError
            why = (f"{type(exc).__name__}: {exc}".strip().splitlines() or [""])[0][:240]
            raise RuntimeError("this model needs 4/8-bit loading on this GPU, and bitsandbytes is installed "
                               f"but would not load ({why}). Press Install dependencies in the Backend tab "
                               "(it upgrades bitsandbytes), or choose a smaller model / the bf16 option")
        skip = list(skip_modules or [])
        if mode == "4bit":
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=skip or None)
        else:
            bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True,
                                                         llm_int8_skip_modules=skip or None)

    errors = []
    # AutoModelForVision2Seq is deprecated in transformers 4.5x+; ImageTextToText covers
    # LLaVA (JoyCaption), Qwen2.5-VL and Qwen3-VL.
    # Qwen3.5 must never fall back to the plain language-model class: that would load the text part only and
    # caption without ever seeing the picture.
    classes = (("AutoModelForImageTextToText", "AutoModelForMultimodalLM") if family == "qwen35"
               else ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"))
    for name in classes:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            if bnb_config is not None:
                # Quantized weights come up in place via device_map; a .to()
                # afterwards is what would silently dissolve the quantization.
                model = cls.from_pretrained(repo, quantization_config=bnb_config,
                                            device_map={"": 0}, torch_dtype=dtype)
            else:
                model = cls.from_pretrained(repo, torch_dtype=dtype)
                model.to(device)
            model.eval()
            return processor, model, name, (mode or "")
        except Exception as exc:                          # noqa: BLE001 - try the next class
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            try:
                torch.cuda.empty_cache()
            except Exception:                             # noqa: BLE001
                pass
    raise RuntimeError("no usable model class for this repository - " + " | ".join(errors))


def _limit_pixels(image, max_pixels: int):
    """Shrink a picture whose pixel count exceeds the budget (keeps aspect; never enlarges)."""
    w, h = image.size
    if max_pixels and w * h > max_pixels:
        k = (max_pixels / float(w * h)) ** 0.5
        image = image.resize((max(28, int(w * k)), max(28, int(h * k))))
    return image


def _strip_thinking(text: str) -> str:
    """Drop a <think>...</think> block (and an unclosed one) some Qwen3.5 checkpoints emit."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.replace("</think>", "").strip()


def _caption_chat(processor, model, image, prompt: str, max_new_tokens: int, device: str,
                  system: str = "", dtype=None, family: str = "", max_pixels: int = 0) -> str:
    if family == "qwen35":
        image = _limit_pixels(image, max_pixels)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
    try:
        extra = {"enable_thinking": False} if family == "qwen35" else {}
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **extra)
    except Exception:                                     # noqa: BLE001 - older processors want the placeholder
        text = f"<image>\n{prompt}"
    inputs = processor(images=[image], text=[text], return_tensors="pt")
    inputs = inputs.to(device)
    # LLaVA-style processors return float32 pixel_values; the vision tower is bf16
    # (and stays bf16 under quantization), so feed it the dtype it was loaded in.
    if dtype is not None:
        for key in ("pixel_values", "pixel_values_videos"):
            if key in inputs and inputs[key] is not None:
                inputs[key] = inputs[key].to(dtype)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                               repetition_penalty=1.05)
    # Decode only what the model added: the full sequence also contains the
    # formatted chat prompt.
    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)[0]
    return _strip_thinking(decoded) if family == "qwen35" else decoded.strip()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        _emit("DONE", {"count": 0, "error": "usage: caption_worker <job.json>"})
        return 2
    try:
        with open(argv[1], "r", encoding="utf-8") as fh:
            job = json.load(fh)
    except (OSError, ValueError) as exc:
        _emit("DONE", {"count": 0, "error": f"could not read the job file: {exc}"})
        return 2

    images = [i for i in job.get("images", []) if i.get("id")]
    try:
        import torch
        from PIL import Image
    except Exception as exc:                              # noqa: BLE001
        _emit("DONE", {"count": 0, "error": f"this interpreter cannot caption ({type(exc).__name__}: {exc})"})
        return 3

    device = _pick_device(torch)
    vram_gb = _vram_budget(torch, device, job.get("vram_profile") or "auto")
    try:
        processor, model, loader, quant = _load(job["repo"], job.get("kind", "chat"), torch, device,
                                                quant=job.get("quant") or "auto",
                                                size_gb=float(job.get("size_gb") or 0),
                                                skip_modules=job.get("skip_modules") or [],
                                                vram_gb=vram_gb, family=job.get("family") or "")
    except Exception as exc:                              # noqa: BLE001
        _emit("DONE", {"count": 0, "error": f"{type(exc).__name__}: {exc}"})
        return 4
    _emit("READY", {"device": device, "loader": loader, "quant": quant or "bf16"})

    max_new_tokens = int(job.get("max_new_tokens") or 320)
    prompt = job.get("prompt") or "Describe this image in one or two sentences."
    system = job.get("system") or ""
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    budget = int(vram_gb) if vram_gb else 0
    pix_cap = next((v for k, v in sorted(_MAX_PIXELS_BY_VRAM.items()) if budget and budget <= k), _MAX_PIXELS_DEFAULT)

    done = 0
    for item in images:
        try:
            with Image.open(item["id"]) as img:
                rgb = img.convert("RGB")
                with torch.inference_mode():
                    caption = _caption_chat(processor, model, rgb, prompt, max_new_tokens, device,
                                            system=system, dtype=dtype, family=job.get("family") or "",
                                            max_pixels=pix_cap)
            caption = " ".join((caption or "").split())
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()              # keeps an 8 GB card from fragmenting across a batch
                except Exception:                         # noqa: BLE001
                    pass
            if not caption:
                _emit("CAPTION", {"id": item["id"], "error": "the model returned nothing"})
            else:
                _emit("CAPTION", {"id": item["id"], "caption": caption})
                done += 1
        except Exception as exc:                          # noqa: BLE001 - one bad image is not the batch
            _emit("CAPTION", {"id": item["id"],
                              "error": f"{type(exc).__name__}: {exc}",
                              "trace": traceback.format_exc(limit=3)})

    _emit("DONE", {"count": done, "error": ""})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
