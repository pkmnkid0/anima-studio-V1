# MOSS-Image text encoder worker.
#
# Qwen3.5 needs transformers>=5.3, which is not installed in the training environment. The trainer runs this
# script as a subprocess with a transformers 5 overlay prepended to PYTHONPATH (see moss_te_client.py).
# It must not import anything from `library`.
#
# Protocol: one JSON request per stdin line, one reply per stdout line prefixed with REPLY_PREFIX.
#   {"captions": [...], "out": "<path>.safetensors"} -> writes emb_<i> ([L, hidden] tensors), replies {"ok": true, "n": N}
#   {"cmd": "exit"}                                  -> exits
# Everything else the libraries print goes to stderr.

import argparse
import json
import os
import sys
import traceback

REPLY_PREFIX = "@@MOSS_TE@@ "


def reply(stdout, payload: dict):
    stdout.write(REPLY_PREFIX + json.dumps(payload) + "\n")
    stdout.flush()


TOKENIZER_REPOS = ("OpenMOSS-Team/MOSS-Image-Base", "Qwen/Qwen3.5-9B")


def download_tokenizer() -> str:
    """No tokenizer next to the text encoder: fetch just the tokenizer files (a few MB, no weights) from the
    MOSS-Image repo into the normal Hugging Face cache, and reuse them on later runs."""
    errors = []
    for repo in TOKENIZER_REPOS:
        try:
            from huggingface_hub import snapshot_download
            patterns = ["tokenizer/*"] if "MOSS" in repo else [
                "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
                "chat_template*", "preprocessor_config.json"]
            root = snapshot_download(repo_id=repo, allow_patterns=patterns)
            for cand in (os.path.join(root, "tokenizer"), root):
                if os.path.isfile(os.path.join(cand, "tokenizer.json")) or os.path.isfile(os.path.join(cand, "tokenizer_config.json")):
                    print(f"[moss-te] tokenizer downloaded from {repo}", file=sys.stderr, flush=True)
                    return cand
            errors.append(f"{repo}: no tokenizer files in it")
        except Exception as exc:                                  # noqa: BLE001
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError("No tokenizer next to the text encoder and it could not be downloaded ("
                            + " | ".join(errors) + "). Connect to the internet once, or set 'Tokenizer folder'.")


def resolve_paths(path: str, tokenizer_path: str | None):
    """Accept a pipeline root (with text_encoder/ and tokenizer/) or a text_encoder folder."""
    if os.path.isfile(path) and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")):
        # model.safetensors inside a normal Hugging Face text_encoder/ folder: load the folder the usual way
        path = os.path.dirname(os.path.abspath(path))
    if os.path.isfile(path):          # a bare bf16 .safetensors file (ComfyUI export, no config beside it)
        folder = os.path.dirname(os.path.abspath(path))
        if tokenizer_path is None:
            for cand in (os.path.join(folder, "tokenizer"), folder, os.path.join(os.path.dirname(folder), "tokenizer")):
                if os.path.isfile(os.path.join(cand, "tokenizer.json")) or os.path.isfile(os.path.join(cand, "tokenizer_config.json")):
                    tokenizer_path = cand
                    break
        if tokenizer_path is None:
            tokenizer_path = download_tokenizer()
        return path, tokenizer_path
    model_dir = os.path.join(path, "text_encoder") if os.path.isdir(os.path.join(path, "text_encoder")) else path
    if tokenizer_path is None:
        for cand in (os.path.join(path, "tokenizer"), os.path.join(os.path.dirname(model_dir), "tokenizer"), model_dir):
            if os.path.isfile(os.path.join(cand, "tokenizer.json")) or os.path.isfile(os.path.join(cand, "tokenizer_config.json")):
                tokenizer_path = cand
                break
    if tokenizer_path is None:
        raise FileNotFoundError(f"No tokenizer found next to {model_dir}; pass --tokenizer")
    return model_dir, tokenizer_path


_STRIP_PREFIXES = ("model.language_model.", "language_model.model.", "language_model.", "text_model.", "model.")


def normalize_key(key: str) -> str:
    """ComfyUI / HF exports prefix the same tensor differently ("model.layers.0...", "language_model.layers.0...",
    "model.language_model.layers.0..."); compare them with the prefix stripped."""
    changed = True
    while changed:
        changed = False
        for pre in _STRIP_PREFIXES:
            if key.startswith(pre):
                key = key[len(pre):]
                changed = True
    return key


def map_single_file_keys(expected: list, file_keys: list):
    """-> (mapping {expected key: file key}, missing expected keys). Vision-tower keys are ignored."""
    by_norm = {}
    for fk in file_keys:
        if fk.startswith(("visual.", "model.visual.", "vision_tower.", "model.vision_tower.", "mtp.")):
            continue
        by_norm.setdefault(normalize_key(fk), fk)
    mapping, missing = {}, []
    for ek in expected:
        fk = by_norm.get(normalize_key(ek))
        if fk is None:
            missing.append(ek)
        else:
            mapping[ek] = fk
    return mapping, missing


QWEN35_ROPE = {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
               "mrope_section": [11, 11, 10], "mrope_interleaved": True}


def infer_text_config(shapes: dict) -> dict:
    """Qwen3_5TextConfig kwargs read off tensor shapes ({normalized key: shape}), so a bare .safetensors file needs
    no config.json. Layer count, widths, head counts and the full/linear attention layout all follow from the weights;
    only the RoPE settings (theta 1e7, rotary fraction 0.25) and the 128-wide linear key heads are Qwen3.5 constants."""
    vocab, hidden = shapes["embed_tokens.weight"]
    layer_ids = sorted({int(k.split(".")[1]) for k in shapes if k.startswith("layers.") and k.split(".")[1].isdigit()})
    n = layer_ids[-1] + 1
    layer_types = ["full_attention" if f"layers.{i}.self_attn.q_proj.weight" in shapes else "linear_attention"
                   for i in range(n)]
    cfg = {"vocab_size": vocab, "hidden_size": hidden, "num_hidden_layers": n, "layer_types": layer_types,
           "intermediate_size": shapes["layers.0.mlp.up_proj.weight"][0], "rope_parameters": dict(QWEN35_ROPE)}
    full = next((i for i, t in enumerate(layer_types) if t == "full_attention"), None)
    if full is not None:
        head_dim = shapes[f"layers.{full}.self_attn.q_norm.weight"][0]
        cfg.update(head_dim=head_dim,
                   num_attention_heads=shapes[f"layers.{full}.self_attn.q_proj.weight"][0] // (head_dim * 2),
                   num_key_value_heads=shapes[f"layers.{full}.self_attn.k_proj.weight"][0] // head_dim)
    lin = next((i for i, t in enumerate(layer_types) if t == "linear_attention"), None)
    if lin is not None:
        pre = f"layers.{lin}.linear_attn."
        v_heads = shapes[pre + "A_log"][0]
        v_dim = shapes[pre + "in_proj_z.weight"][0]
        qkv = shapes[pre + "in_proj_qkv.weight"][0]
        k_head_dim = 128
        k_total = (qkv - v_dim) // 2
        cfg.update(linear_num_value_heads=v_heads, linear_value_head_dim=v_dim // v_heads,
                   linear_key_head_dim=k_head_dim, linear_num_key_heads=k_total // k_head_dim,
                   linear_conv_kernel_dim=shapes[pre + "conv1d.weight"][-1])
    return cfg


def load_single_file(path: str, config_dir: str, dtype, torch, transformers):
    """Build the Qwen3.5 text model from its config and fill it from one bf16 .safetensors file.

    The model is built on the CPU without random initialisation and each tensor is copied in as it is read, so
    the ~18 GB file is never held in RAM twice (the previous build-on-meta + load_state_dict path built a second
    full dict of tensors)."""
    from safetensors import safe_open
    try:                                           # transformers 5
        from transformers.initialization import no_init_weights
    except ImportError:                            # transformers 4.x
        from transformers.modeling_utils import no_init_weights

    with safe_open(path, framework="pt", device="cpu") as f0:
        file_keys0 = list(f0.keys())
        shapes = {normalize_key(k): f0.get_slice(k).get_shape() for k in file_keys0
                  if not k.startswith(("visual.", "model.visual.", "vision_tower.", "model.vision_tower.", "mtp."))}
    if config_dir and os.path.isfile(os.path.join(config_dir, "config.json")):
        print(f"[moss-te] reading config from {config_dir}", file=sys.stderr, flush=True)
        cfg = transformers.AutoConfig.from_pretrained(config_dir)
    else:
        kw = infer_text_config(shapes)
        print(f"[moss-te] no config.json - config read from the weights: {kw['num_hidden_layers']} layers, "
              f"hidden {kw['hidden_size']}, {kw.get('num_attention_heads')} heads", file=sys.stderr, flush=True)
        cfg = transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig(**kw)
    print("[moss-te] building the model on the CPU (no random init)", file=sys.stderr, flush=True)
    with no_init_weights():
        if type(cfg).__name__ == "Qwen3_5TextConfig":
            model = transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel(cfg)
        else:
            model = transformers.AutoModel.from_config(cfg, torch_dtype=dtype)
    for prm in model.parameters():                 # weights only: buffers (RoPE inv_freq, ...) stay fp32 like from_pretrained
        prm.data = prm.data.to(dtype)
    text_model = model.language_model if hasattr(model, "language_model") else model
    params = dict(text_model.state_dict())
    with safe_open(path, framework="pt", device="cpu") as f:
        file_keys = list(f.keys())
        for k in file_keys[:50]:
            dt = f.get_slice(k).get_dtype()
            if dt not in ("BF16", "F16", "F32"):
                raise RuntimeError(f"{os.path.basename(path)} holds {dt} tensors ({k}) - it is quantized. The trainer needs the "
                                   "full bf16 text encoder (the text_encoder/ folder of MOSS-Image-Base or a bf16 single file).")
        mapping, missing = map_single_file_keys(list(params.keys()), file_keys)
        if missing:
            raise RuntimeError(f"{len(missing)} weights of the Qwen3.5 text model are not in {os.path.basename(path)} "
                               f"(e.g. {missing[:3]}); it is not a full bf16 Qwen3.5-9B text encoder.")
        print(f"[moss-te] copying {len(mapping)} tensors", file=sys.stderr, flush=True)
        with torch.no_grad():
            for ek, fk in mapping.items():
                params[ek].copy_(f.get_tensor(fk))
    return model, text_model

# Qwen3.5 chat format with thinking on (what MOSS-Image trains against): "<|im_start|>assistant\n<think>\n".
FALLBACK_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}"
    "{% endif %}")


def ensure_chat_template(tokenizer, tokenizer_dir: str) -> str:
    """A tokenizer folder copied without its chat template (tokenizer_config.json / chat_template.jinja) cannot format
    captions. Find the template: beside the tokenizer, then in the MOSS-Image / Qwen3.5 repos (a tiny file), then the
    built-in Qwen3.5 format. Done before the 9B model loads, so a missing template costs seconds, not a 15 min load.
    Returns where it came from."""
    if getattr(tokenizer, "chat_template", None):
        return "tokenizer"
    for d in (tokenizer_dir, os.path.dirname(os.path.abspath(tokenizer_dir))):
        f = os.path.join(d, "chat_template.jinja")
        if os.path.isfile(f):
            tokenizer.chat_template = open(f, encoding="utf-8").read()
            return f
    try:
        from huggingface_hub import snapshot_download
        for repo in TOKENIZER_REPOS:
            try:
                root = snapshot_download(repo_id=repo, allow_patterns=["tokenizer/chat_template*", "tokenizer/tokenizer_config.json",
                                                                        "chat_template*", "tokenizer_config.json"])
            except Exception:                                     # noqa: BLE001
                continue
            for cand in (os.path.join(root, "tokenizer", "chat_template.jinja"), os.path.join(root, "chat_template.jinja")):
                if os.path.isfile(cand):
                    tokenizer.chat_template = open(cand, encoding="utf-8").read()
                    return f"{repo}/{os.path.basename(cand)}"
            for cand in (os.path.join(root, "tokenizer", "tokenizer_config.json"), os.path.join(root, "tokenizer_config.json")):
                if os.path.isfile(cand):
                    tpl = json.load(open(cand, encoding="utf-8")).get("chat_template")
                    if isinstance(tpl, str) and tpl:
                        tokenizer.chat_template = tpl
                        return f"{repo}/tokenizer_config.json"
    except ImportError:
        pass
    tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    return "built-in Qwen3.5 template"


def warm_file_cache(model_path: str, max_gb: float = 40.0) -> None:
    """Read the weight files once, front to back, so the OS keeps them in RAM before from_pretrained touches them.

    safetensors loads tensor by tensor in file order of the *index*, which on a hard disk turns into thousands of
    seeks (a 18 GB encoder took 15 minutes); one sequential pass runs at the disk's full speed (~2-3 minutes) and the
    loader then reads from the cache. Harmless on an SSD, and the cache is evictable, so it cannot cause an out-of-memory."""
    import time
    if os.path.isfile(model_path):
        files = [model_path]
    else:
        files = sorted(os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".safetensors"))
    total = sum(os.path.getsize(f) for f in files)
    if not files or total > max_gb * 1024 ** 3:
        return
    t0, done, buf = time.time(), 0, bytearray(64 * 1024 * 1024)
    print(f"[moss-te] reading {total / 1024 ** 3:.1f} GB of weights sequentially (fast on hard disks)...", file=sys.stderr, flush=True)
    last = t0
    for f in files:
        with open(f, "rb", buffering=0) as fh:
            while True:
                n = fh.readinto(buf)
                if not n:
                    break
                done += n
                if time.time() - last > 15:
                    last = time.time()
                    print(f"[moss-te]   {done / 1024 ** 3:.1f} / {total / 1024 ** 3:.1f} GB", file=sys.stderr, flush=True)
    print(f"[moss-te] weights read in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_length", type=int, default=512)
    # Low-VRAM loading (Anima Studio addition). full: whole model on the GPU (~18 GB). offload: exact bf16 weights,
    # as many layers as fit under --gpu_gb on the GPU and the rest streamed from CPU RAM (needs ~20 GB RAM, slower).
    # nf4 / int8: bitsandbytes-quantized text encoder on the GPU (small and fast, but embeddings differ slightly
    # from bf16, so the LoRA is trained against marginally different text features than inference will see).
    parser.add_argument("--mode", default="full", choices=["full", "offload", "nf4", "int8"])
    parser.add_argument("--gpu_gb", type=float, default=6.0, help="GPU memory budget for the offload mode")
    parser.add_argument("--config_dir", default=None,
                        help="Single-file text encoder only: folder with the Qwen3.5 config.json "
                             "(default: the Hugging Face hub config of Qwen/Qwen3.5-9B)")
    parser.add_argument("--cpu_gb", type=float, default=48.0, help="CPU RAM budget for the offload mode")
    args = parser.parse_args()

    # keep the protocol channel clean: libraries print to stdout
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    import faulthandler
    faulthandler.enable(file=sys.stderr, all_threads=True)      # a native crash (exit 0xC0000005) prints where it died
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer

    major = int(transformers.__version__.split(".")[0])
    if major < 5:
        reply(protocol_out, {"ok": False, "error": f"transformers>=5.3 required in the worker, found {transformers.__version__}"})
        return 1

    model_dir, tokenizer_dir = resolve_paths(args.model, args.tokenizer)
    if args.mode == "full" and str(args.device).startswith("cuda"):
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception:
            total_gb = 0.0
        if 0 < total_gb < 22:
            print(f"[moss-te] this GPU has {total_gb:.1f} GB; the 9B text encoder needs ~18 GB in 'full' mode - "
                  f"switching to 'offload' (exact bf16, streamed from RAM)", file=sys.stderr, flush=True)
            args.mode = "offload"
            args.gpu_gb = min(args.gpu_gb, max(2.0, total_gb - 3.0))
    single_file = os.path.isfile(model_dir)
    config_dir = args.config_dir or (os.path.join(os.path.dirname(os.path.abspath(model_dir)), "text_encoder")
                                     if single_file else None)
    dtype = getattr(torch, args.dtype)
    print(f"[moss-te] transformers {transformers.__version__}, loading {model_dir} ({args.dtype}) on {args.device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    print(f"[moss-te] chat template: {ensure_chat_template(tokenizer, tokenizer_dir)}", file=sys.stderr, flush=True)
    _probe = tokenizer.apply_chat_template([{"role": "user", "content": "a"}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True)
    print(f"[moss-te] template check: {_probe!r}", file=sys.stderr, flush=True)
    on_gpu = str(args.device).startswith("cuda")
    try:
        warm_file_cache(model_dir)
    except OSError as exc:
        print(f"[moss-te] could not pre-read the weights ({exc}); loading normally", file=sys.stderr, flush=True)
    if single_file:
        cfg_src = config_dir if config_dir and os.path.isfile(os.path.join(config_dir, "config.json")) else None
        print(f"[moss-te] single-file text encoder {args.model} (config: {cfg_src or 'inferred from the weights'})", file=sys.stderr)
        if args.mode in ("nf4", "int8"):
            reply(protocol_out, {"ok": False, "error": "A single-file text encoder supports 'full' and 'offload' modes, not nf4/int8"})
            return 1
        model, text_model = load_single_file(model_dir, cfg_src, dtype, torch, transformers)
        text_model.eval().requires_grad_(False)
        if args.mode == "offload" and on_gpu:
            from accelerate import dispatch_model, infer_auto_device_map
            gpu_index = int(str(args.device).split(":")[1]) if ":" in str(args.device) else 0
            max_memory = {gpu_index: f"{args.gpu_gb:g}GiB", "cpu": f"{args.cpu_gb:g}GiB"}
            device_map = infer_auto_device_map(text_model, max_memory=max_memory,
                                               no_split_module_classes=list(getattr(text_model, "_no_split_modules", None) or []))
            dispatch_model(text_model, device_map=device_map)
            input_device = f"cuda:{gpu_index}"
        else:
            text_model.to(args.device)
            input_device = args.device
    elif args.mode == "full" or not on_gpu:
        model = AutoModel.from_pretrained(model_dir, dtype=dtype)
        # only the language model is needed for text prompts; the vision tower stays on the CPU
        text_model = model.language_model if hasattr(model, "language_model") else model
        text_model.to(args.device).eval().requires_grad_(False)
        input_device = args.device
    elif args.mode == "offload":
        gpu_index = int(str(args.device).split(":")[1]) if ":" in str(args.device) else 0
        max_memory = {gpu_index: f"{args.gpu_gb:g}GiB", "cpu": f"{args.cpu_gb:g}GiB"}
        print(f"[moss-te] offload mode: <= {args.gpu_gb:g} GiB on the GPU, the rest streamed from CPU RAM", file=sys.stderr)
        model = AutoModel.from_pretrained(model_dir, dtype=dtype, device_map="auto", max_memory=max_memory,
                                          low_cpu_mem_usage=True)
        text_model = model.language_model if hasattr(model, "language_model") else model
        text_model.eval().requires_grad_(False)
        input_device = f"cuda:{gpu_index}"
    else:
        from transformers import BitsAndBytesConfig
        gpu_index = int(str(args.device).split(":")[1]) if ":" in str(args.device) else 0
        if args.mode == "nf4":
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                       bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["visual", "lm_head"])
        else:
            quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual", "lm_head"])
        print(f"[moss-te] {args.mode} mode (quantized: embeddings differ slightly from bf16)", file=sys.stderr)
        model = AutoModel.from_pretrained(model_dir, dtype=dtype, quantization_config=quant,
                                          device_map={"": gpu_index}, low_cpu_mem_usage=True)
        text_model = model.language_model if hasattr(model, "language_model") else model
        text_model.eval().requires_grad_(False)
        input_device = f"cuda:{gpu_index}"
    print("[moss-te] ready", file=sys.stderr)
    reply(protocol_out, {"ready": True, "transformers": transformers.__version__})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("cmd") == "exit":
                break
            tensors = {}
            with torch.no_grad():
                for i, caption in enumerate(request["captions"]):
                    # exact training-time chat template: thinking enabled, no system prompt
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": caption}], tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)["input_ids"]
                    # unpadded single forwards: padding changes Qwen3.5's gated delta-net (linear attention) states
                    out = text_model(input_ids=ids.to(input_device), use_cache=False, return_dict=True)
                    tensors[f"emb_{i}"] = out.last_hidden_state[0].to("cpu", dtype=dtype).contiguous()
            save_file(tensors, request["out"])
            reply(protocol_out, {"ok": True, "n": len(tensors)})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            reply(protocol_out, {"ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
