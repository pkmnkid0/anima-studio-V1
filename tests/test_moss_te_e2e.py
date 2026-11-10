"""Runs the real text-encoder worker on a tiny random Qwen3.5 (skipped unless torch + transformers>=5 are installed).

A bare single-file text encoder (ComfyUI-style key names, no config.json) must give bit-identical embeddings to the
same weights loaded as a normal Hugging Face folder."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("needs transformers>=5", allow_module_level=True)

WORKER = Path(__file__).resolve().parent.parent / "moss_backend" / "sd_scripts" / "library" / "moss_te_worker.py"


def _run(args, out):
    p = subprocess.Popen([sys.executable, "-P", str(WORKER)] + args + ["--device", "cpu", "--max_length", "64"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p.stdin.write(json.dumps({"captions": ["hi there a girl"], "out": str(out)}) + "\n")
    p.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
    p.stdin.flush()
    o, e = p.communicate(timeout=300)
    assert p.returncode == 0, e[-2000:]
    assert '"ok": true' in o, o + e[-1000:]
    from safetensors.torch import load_file
    return load_file(str(out))["emb_0"], e


def test_bare_single_file_matches_a_hf_folder(tmp_path):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    rope = {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10], "mrope_interleaved": True}
    cfg = Qwen3_5TextConfig(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_value_heads=4,
                            linear_value_head_dim=64, linear_key_head_dim=128, linear_num_key_heads=2,
                            linear_conv_kernel_dim=4, rope_parameters=rope)
    torch.manual_seed(0)
    sd = {k: v.contiguous() for k, v in Qwen3_5TextModel(cfg).to(torch.bfloat16).state_dict().items()}

    vocab = {"<unk>": 0, "hi": 1, "there": 2, "a": 3, "girl": 4, "<|im_start|>": 5, "<|im_end|>": 6}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>")
    fast.chat_template = ("{% for m in messages %}<|im_start|> {{ m['content'] }} <|im_end|> {% endfor %}"
                          "{% if add_generation_prompt %}<|im_start|> {% endif %}")
    fast.save_pretrained(tmp_path / "tokenizer")

    bare = tmp_path / "bare.safetensors"             # ComfyUI-style names, no config.json beside it
    save_file({"model.language_model." + k: v for k, v in sd.items()}, str(bare))
    folder = tmp_path / "hf" / "text_encoder"
    folder.mkdir(parents=True)
    save_file(sd, str(folder / "model.safetensors"))
    cfg.save_pretrained(folder)
    shutil.copytree(tmp_path / "tokenizer", tmp_path / "hf" / "tokenizer")

    a, _ = _run(["--model", str(bare), "--tokenizer", str(tmp_path / "tokenizer")], tmp_path / "a.safetensors")
    b, _ = _run(["--model", str(folder / "model.safetensors")], tmp_path / "b.safetensors")
    assert torch.isfinite(a.float()).all()
    assert torch.equal(a, b)

    # a tokenizer folder copied WITHOUT its chat template must not break the run (it did, after a 15 minute load)
    notpl = tmp_path / "tok_no_template"
    shutil.copytree(tmp_path / "tokenizer", notpl)
    cfgf = notpl / "tokenizer_config.json"
    data = json.loads(cfgf.read_text())
    data.pop("chat_template", None)
    cfgf.write_text(json.dumps(data))
    for extra in notpl.glob("chat_template*"):
        extra.unlink()
    os.environ["HF_HUB_OFFLINE"] = "1"
    c, err = _run(["--model", str(bare), "--tokenizer", str(notpl)], tmp_path / "c.safetensors")
    assert "chat template: built-in Qwen3.5 template" in err
    assert "<|im_start|>assistant\\n<think>\\n" in err                  # the template check line, repr()'d
    assert torch.isfinite(c.float()).all()
