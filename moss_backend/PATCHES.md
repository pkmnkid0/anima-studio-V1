# Changes to the vendored sd-scripts (RichyRich515, branch `moss-image`)

Everything else is unmodified upstream.

1. `library/moss_te_worker.py` - low-VRAM text-encoder loading. New flags `--mode {full,offload,nf4,int8}`,
   `--gpu_gb`, `--cpu_gb`. Upstream loads the whole 9B Qwen3.5 encoder on the GPU (~18 GB) while captions are
   cached, which cannot happen on 8 GB. `offload` keeps exact bf16 weights and streams layers from CPU RAM
   (accelerate `device_map="auto"` + `max_memory`); `nf4`/`int8` quantize the encoder (small and fast, but
   embeddings differ slightly from bf16).
2. `library/moss_te_client.py`, `library/moss_train_utils.py`, `moss_train_network.py` - new options
   `--moss_te_mode`, `--moss_te_gpu_gb`, `--moss_te_cpu_gb`, passed through to that worker.

# Note on training_backend/ (Anima / SDXL engine)

Re-added from upstream 67372a/LoRA_Easy_Training_scripts_Backend (+ 67372a/sd-scripts) because the zip supplied
for this build did not contain it. One Anima Studio patch: `sd_scripts/library/train_util.py` injects
`first_cycle_max_steps` for RexAnnealingWarmRestarts when it is not set (see tests/test_scheduler_fix.py).

3. `library/moss_te_worker.py` - single-file text encoder. `--model` may be one bf16 `.safetensors` file (ComfyUI export):
   the Qwen3.5 model is built from its config and filled from the file with prefix-insensitive key mapping; quantized
   files are refused. Needs `--tokenizer` (the pipeline's `tokenizer/`). Supports full and offload modes. UNTESTED on a real file.

4. `library/moss_te_worker.py` (later fix) - `model.safetensors` with a `config.json` beside it is loaded as a normal HF folder;
   the bare single-file loader now streams tensors into a CPU-built model; `full` mode on a GPU under 22 GB switches to
   `offload`; `faulthandler` prints where a native crash (exit code 3221225477 / 0xC0000005) happened.

5. `library/moss_te_worker.py` - a bare text-encoder `.safetensors` no longer needs a config.json: the Qwen3.5 text config
   is inferred from tensor shapes (`infer_text_config`); RoPE theta 1e7 / rotary fraction 0.25 are constants taken from
   Qwen3.5's published config from memory - UNVERIFIED against the real file.

6. `library/moss_te_worker.py` - a bare text-encoder file with no tokenizer beside it downloads only the tokenizer files
   (a few MB) from OpenMOSS-Team/MOSS-Image-Base (fallback Qwen/Qwen3.5-9B) into the Hugging Face cache. Needs internet once.

7. `library/moss_te_worker.py` - verified on a tiny random Qwen3.5 (tests/test_moss_te_e2e.py): fixed the transformers-5
   `no_init_weights` import location and a bf16 cast of RoPE buffers; a bare single file is now bit-identical to a normal folder load.

8. `library/moss_te_worker.py` - a tokenizer folder without its chat template no longer fails after the model load:
   the template is found (beside the tokenizer, then the MOSS/Qwen repos, then a built-in Qwen3.5 format) and checked
   BEFORE the 9B model loads.

9. `library/moss_te_worker.py` - `warm_file_cache`: reads the text-encoder weights front to back once before loading, so a
   hard disk streams at full speed instead of seeking tensor by tensor (a 15 min load in the field).

10. `moss_train_network.py`, `library/moss_image_models.py` - `--moss_int8` loads the bf16 DiT on the CPU and quantizes each
    projection on the GPU one at a time (`quantize_blocks_int8(..., device=)`), so the GPU never holds the ~10 GB bf16 weights.
    Stock code loaded them all onto the GPU first (out of memory on a 10 GB card). Quantization checked on CPU: identical output.
