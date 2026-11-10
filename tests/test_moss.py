"""MOSS-Image integration: UI state -> trainer args, runner routing, VRAM profiles, and a static check that every
option the Studio emits is a real option of the vendored trainer's parser (torch is not available to import it)."""
import copy
import re
from pathlib import Path

import pytest

from app import train_config as tc
from app.services import recommend

ROOT = Path(__file__).resolve().parent.parent
SD = ROOT / "moss_backend" / "sd_scripts"


def _moss_ui(**model):
    ui = tc.default_ui_payload()
    ui["model_type"] = "moss"
    ui["model"]["moss_model"] = "/models/MOSS-Image-Base"
    ui["model"].update(model)
    return ui


def test_moss_payload_has_moss_args_and_no_anima_args():
    p = tc.build_backend_payload(_moss_ui())
    assert "anima_args" not in p["args"] and "moss_args" in p["args"]
    m = p["args"]["moss_args"]
    assert m["pretrained_model_name_or_path"] == "/models/MOSS-Image-Base"
    assert m["timestep_sampling"] == "moss" and m["moss_ts_cutoff"] == 0.95 and m["discrete_flow_shift"] == 1.0
    assert m["blocks_to_swap"] == 36 and "moss_int8" not in m          # 8 GB profile defaults: full bf16
    assert m["moss_te_mode"] == "offload" and m["moss_te_gpu_gb"] == 6.0
    assert m["moss_sample_uncond_skip_layers"] == "22"
    g = p["args"]["general_args"]
    assert g["pretrained_model_name_or_path"] == "/models/MOSS-Image-Base" and g["sdxl"] is False
    assert g["mixed_precision"] == "bf16" and "full_fp16" not in g


def test_moss_never_sends_fp16():
    ui = _moss_ui()
    ui["general"]["full_fp16"] = True
    g = tc.build_backend_payload(ui)["args"]["general_args"]
    assert "full_fp16" not in g and g["mixed_precision"] == "bf16"
    ui["general"]["full_bf16"] = True
    assert tc.build_backend_payload(ui)["args"]["general_args"]["full_bf16"] is True


def test_int8_and_block_swap_are_exclusive():
    m = tc.build_backend_payload(_moss_ui(moss_int8="on", moss_blocks_to_swap=0))["args"]["moss_args"]
    assert m["moss_int8"] is True and "blocks_to_swap" not in m and "moss_int8_bwd" not in m
    m = tc.build_backend_payload(_moss_ui(moss_int8="on_bwd", moss_blocks_to_swap=0))["args"]["moss_args"]
    assert m["moss_int8_bwd"] == "int8"
    m = tc.build_backend_payload(_moss_ui(moss_int8="on", moss_blocks_to_swap=20))["args"]["moss_args"]
    assert "moss_int8" not in m and m["blocks_to_swap"] == 20          # the trainer would refuse both
    assert tc.build_backend_payload(_moss_ui(moss_blocks_to_swap=99))["args"]["moss_args"]["blocks_to_swap"] == 40


def test_te_mode_full_sends_nothing_and_skip_layers_none():
    m = tc.build_backend_payload(_moss_ui(moss_te_mode="full", moss_sample_skip_layers=""))["args"]["moss_args"]
    assert "moss_te_mode" not in m and m["moss_sample_uncond_skip_layers"] == "none"
    m = tc.build_backend_payload(_moss_ui(moss_te_mode="nf4"))["args"]["moss_args"]
    assert m["moss_te_mode"] == "nf4" and "moss_te_gpu_gb" not in m


def test_moss_roundtrips_through_toml():
    ui = _moss_ui(moss_blocks_to_swap=24, moss_te_mode="nf4", moss_attn_backend="cudnn", moss_compile="blocks",
                  moss_lora_save_format="fused", moss_fp32_modulation=True, moss_unchecked_blocks=3)
    payload = tc.build_backend_payload(ui)
    back = tc.to_ui_payload(tc.from_toml_document(tc.to_toml_document(payload)))
    assert back["model_type"] == "moss"
    for key in ("moss_model", "moss_blocks_to_swap", "moss_te_mode", "moss_attn_backend", "moss_compile",
                "moss_lora_save_format", "moss_fp32_modulation", "moss_unchecked_blocks", "moss_int8"):
        assert back["model"][key] == ui["model"][key], key
    assert tc.build_backend_payload(back)["args"]["moss_args"] == payload["args"]["moss_args"]


def test_anima_and_sdxl_payloads_unchanged_by_moss_fields():
    for mt in ("anima", "sdxl"):
        ui = tc.default_ui_payload()
        ui["model_type"] = mt
        assert "moss_args" not in tc.build_backend_payload(ui)["args"]


def test_runner_routing():
    from native_backend import server
    assert server._pick_runner("lora", False, False, False, True) == "moss"
    assert server._pick_runner("lora", False, False, True, False) == "anima"
    assert server._pick_runner("lora", True, False, False, False) == "sdxl"
    assert server._pick_runner("lora", False, False, False, False) is None
    assert server._pick_runner("lora", True, False, False, True) is None
    assert server._pick_runner("full", False, False, False, True) is None


def test_trainer_override_mapping_for_moss():
    """anima_trainer._apply_ui_overrides turns moss_args into the fork's parser fields (no torch needed)."""
    from argparse import Namespace
    from native_backend import anima_trainer as at
    payload = tc.build_backend_payload(_moss_ui(moss_te_mode="nf4", moss_int8="on", moss_blocks_to_swap=0,
                                                moss_compile="blocks", moss_attn_backend="cudnn"))
    ns = Namespace(network_dim=None, network_alpha=None, pretrained_model_name_or_path=None)
    old = at._KIND
    at._KIND = "moss"
    try:
        at._apply_ui_overrides(ns, payload["args"], 10, 1)
    finally:
        at._KIND = old
    assert ns.pretrained_model_name_or_path == "/models/MOSS-Image-Base"
    assert ns.network_module == "networks.lora_moss"
    assert ns.moss_te_mode == "nf4" and ns.moss_int8 is True and ns.moss_compile == "blocks"
    assert ns.moss_attn_backend == "cudnn" and ns.timestep_sampling == "moss"
    assert ns.moss_ts_mu == pytest.approx(-1.09) and ns.moss_max_token_length == 512
    assert ns.output_name == "moss_lora"


def test_trainer_requires_a_model_for_moss():
    from argparse import Namespace
    from native_backend import anima_trainer as at
    ui = _moss_ui()
    ui["model"]["moss_model"] = ""
    payload = tc.build_backend_payload(ui)
    at._KIND = "moss"
    try:
        with pytest.raises(at.TrainerError):
            at._apply_ui_overrides(Namespace(), payload["args"], 1, 1)
    finally:
        at._KIND = "anima"


def test_every_emitted_option_exists_in_the_vendored_parser():
    """moss_args keys must be real --options of moss_train_network's parser."""
    src = "\n".join((SD / p).read_text(encoding="utf-8") for p in (
        "library/moss_train_utils.py", "library/train_util.py", "train_network.py"))
    declared = set(re.findall(r'["\']--([a-z0-9_]+)["\']', src))
    emitted = set(tc.build_backend_payload(_moss_ui(
        moss_int8="on_bwd", moss_blocks_to_swap=0, moss_te_mode="offload", moss_unchecked_blocks=2,
        moss_fp32_modulation=True, moss_vae_sample_posterior=True, moss_vae="/v",
        moss_text_encoder="/t"))["args"]["moss_args"])
    # trainer consumes this key itself rather than as a parser option
    emitted -= {"pretrained_model_name_or_path"}
    assert emitted <= declared, sorted(emitted - declared)


def test_low_vram_text_encoder_is_wired_through_the_fork():
    utils = (SD / "library/moss_train_utils.py").read_text(encoding="utf-8")
    client = (SD / "library/moss_te_client.py").read_text(encoding="utf-8")
    worker = (SD / "library/moss_te_worker.py").read_text(encoding="utf-8")
    trainer = (SD / "moss_train_network.py").read_text(encoding="utf-8")
    for name in ("moss_te_mode", "moss_te_gpu_gb", "moss_te_cpu_gb"):
        assert f"--{name}" in utils
    assert "args.moss_te_mode" in trainer and '"--mode"' in client
    assert 'choices=["full", "offload", "nf4", "int8"]' in worker and "max_memory" in worker


def test_vram_profiles_cover_8gb_and_stay_bf16_below_24():
    P = recommend.MOSS_VRAM_PROFILES
    assert recommend.moss_profile_for(None) == "8" and recommend.moss_profile_for(8) == "8" and recommend.moss_profile_for(10) == "10"
    assert recommend.moss_profile_for(11.5) == "12" and recommend.moss_profile_for(24) == "24"
    assert recommend.moss_profile_for(6) == "8"
    for key in ("8", "10", "12", "16"):
        m = P[key]["model"]
        assert m["moss_int8"] == "off" and m["moss_blocks_to_swap"] > 0       # full bf16, swapped
    swaps = [P[k]["model"]["moss_blocks_to_swap"] for k in ("8", "10", "12", "16", "24")]
    assert swaps == sorted(swaps, reverse=True) and swaps[-1] == 0
    assert P["24-fast"]["model"]["moss_int8"] == "on" and P["24-fast"]["model"]["moss_blocks_to_swap"] == 0
    # every profile keeps the fields the payload builder accepts
    for key, prof in P.items():
        ui = _moss_ui(**prof["model"])
        assert "moss_args" in tc.build_backend_payload(ui)["args"], key


def test_moss_routes(client, tmp_path):
    j = client.get("/api/train/moss/profiles?vram_gb=8").get_json()
    assert j["suggested"] == "8" and "24-fast" in j["profiles"]
    assert client.get("/api/train/moss/profiles?vram_gb=24").get_json()["suggested"] == "24"
    s = client.get("/api/train/moss/setup/status").get_json()
    assert s["trainer_present"] is True and s["installed"] is False and len(s["steps"]) == 5
    v = client.get("/api/train/moss/verify").get_json()
    assert v["ok"] is False and v["checks"][0]["id"] == "files" and v["checks"][0]["status"] == "ok"
    st = client.get("/api/train/moss/model/status").get_json()
    assert st["model"]["complete"] is False
    assert client.post("/api/train/moss/model/download", json={}).status_code == 400    # no venv yet
    # a complete-looking pipeline folder is recognised
    for sub in ("transformer", "text_encoder", "vae", "tokenizer"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "w.safetensors").write_bytes(b"x" * 4096)
    (tmp_path / "model_index.json").write_text("{}")
    assert client.get(f"/api/train/moss/model/status?path={tmp_path}").get_json()["model"]["complete"] is True


def test_build_route_warns_about_missing_moss_model(client):
    ui = _moss_ui()
    ui["model"]["moss_model"] = ""
    j = client.post("/api/train/build", json=ui).get_json()
    assert any("MOSS-Image model folder" in w for w in j["warnings"])
    assert "moss_args" in j["payload"]["args"] and "[moss_args.args]" in j["toml"]


def test_project_accepts_moss_model_type():
    from app.services import projects
    assert "moss" in projects.MODEL_TYPES


def test_vendored_trainer_is_complete():
    for rel in ("moss_train_network.py", "train_network.py", "networks/lora_moss.py", "library/moss_image_models.py",
                "library/moss_image_vae.py", "library/moss_te_client.py", "library/moss_te_worker.py",
                "library/strategy_moss.py", "library/moss_train_utils.py", "LICENSE.md", "requirements.txt"):
        assert (SD / rel).is_file(), rel


def test_single_file_text_encoder_key_mapping():
    """A ComfyUI-style single bf16 text-encoder file names tensors differently from the HF folder; the worker maps them."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("moss_te_worker", SD / "library" / "moss_te_worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    expected = ["embed_tokens.weight", "layers.0.mlp.up_proj.weight", "norm.weight"]
    for prefix in ("", "model.", "language_model.", "model.language_model."):
        file_keys = [prefix + k for k in expected] + ["model.visual.blocks.0.attn.qkv.weight", "lm_head.weight"]
        mapping, missing = w.map_single_file_keys(expected, file_keys)
        assert not missing and mapping["norm.weight"] == prefix + "norm.weight", prefix
    _, missing = w.map_single_file_keys(expected, ["model.embed_tokens.weight"])
    assert missing == ["layers.0.mlp.up_proj.weight", "norm.weight"]
    # a lone file without a tokenizer is refused with a message that says what to do
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "te.safetensors")
        open(f, "wb").write(b"x")
        w.download_tokenizer = lambda: "/cache/tokenizer"          # no network in tests: a bare file fetches it
        assert w.resolve_paths(f, None) == (f, "/cache/tokenizer")
        os.makedirs(os.path.join(d, "tokenizer"))
        open(os.path.join(d, "tokenizer", "tokenizer.json"), "w").write("{}")
        assert w.resolve_paths(f, None) == (f, os.path.join(d, "tokenizer"))


def test_paths_are_used_exactly_as_typed(tmp_path):
    """Like the Anima tab: no guessing from file names."""
    (tmp_path / "qwen_model.safetensors").write_bytes(b"x")
    m = tc.build_backend_payload(_moss_ui(moss_model=str(tmp_path)))["args"]["moss_args"]
    assert m["pretrained_model_name_or_path"] == str(tmp_path)
    assert "moss_text_encoder" not in m and "vae" not in m and "moss_tokenizer" not in m
    m = tc.build_backend_payload(_moss_ui(moss_model="/m/dit.safetensors", moss_text_encoder="/m/te.safetensors",
                                          moss_vae="/m/vae.safetensors", moss_tokenizer="/m/tok"))["args"]["moss_args"]
    assert (m["pretrained_model_name_or_path"], m["moss_text_encoder"], m["vae"], m["moss_tokenizer"]) == (
        "/m/dit.safetensors", "/m/te.safetensors", "/m/vae.safetensors", "/m/tok")


def test_model_safetensors_inside_a_hf_text_encoder_folder_loads_as_that_folder(tmp_path):
    """E:\\moss training\\text_encoder\\model.safetensors (config.json beside it) is a normal HF folder, not a bare file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("moss_te_worker2", SD / "library" / "moss_te_worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    te = tmp_path / "text_encoder"
    te.mkdir()
    (te / "model.safetensors").write_bytes(b"x")
    (te / "config.json").write_text("{}")
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tokenizer.json").write_text("{}")
    assert w.resolve_paths(str(te / "model.safetensors"), None) == (str(te), str(tmp_path / "tokenizer"))


def test_text_config_is_inferred_from_tensor_shapes():
    """No config.json needed: a Qwen3.5-9B-shaped file gives the 9B config (32 layers, every 4th full attention)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("moss_te_worker3", SD / "library" / "moss_te_worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    sh = {"embed_tokens.weight": [248320, 4096], "norm.weight": [4096]}
    for i in range(32):
        sh[f"layers.{i}.mlp.up_proj.weight"] = [12288, 4096]
        if (i + 1) % 4 == 0:
            sh[f"layers.{i}.self_attn.q_proj.weight"] = [16 * 256 * 2, 4096]
            sh[f"layers.{i}.self_attn.k_proj.weight"] = [4 * 256, 4096]
            sh[f"layers.{i}.self_attn.q_norm.weight"] = [256]
        else:
            p = f"layers.{i}.linear_attn."
            sh[p + "A_log"] = [32]
            sh[p + "in_proj_z.weight"] = [32 * 128, 4096]
            sh[p + "in_proj_qkv.weight"] = [16 * 128 * 2 + 32 * 128, 4096]
            sh[p + "conv1d.weight"] = [8192, 1, 4]
    c = w.infer_text_config(sh)
    assert (c["vocab_size"], c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"]) == (248320, 4096, 32, 12288)
    assert (c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]) == (16, 4, 256)
    assert (c["linear_num_value_heads"], c["linear_num_key_heads"], c["linear_value_head_dim"],
            c["linear_key_head_dim"], c["linear_conv_kernel_dim"]) == (32, 16, 128, 128, 4)
    assert c["layer_types"].count("full_attention") == 8 and c["layer_types"][3] == "full_attention"
    assert c["rope_parameters"]["rope_theta"] == 10000000


def test_vendored_moss_files_parse_on_python_310():
    """The MOSS environment can be Python 3.10-3.13; a nested same-quote f-string only parses on 3.12+."""
    import re
    for f in list((ROOT / "moss_backend" / "sd_scripts").glob("*.py")) + list((ROOT / "moss_backend" / "sd_scripts" / "library").glob("moss_*.py")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert not re.search(r'\bf"([^"\\]|\\.)*\{[^{}"]*"[^{}"]*"[^{}]*\}', line), f"{f.name}:{n} nested quotes in an f-string"
            assert not re.search(r"\bf'([^'\\]|\\.)*\{[^{}']*'[^{}']*'[^{}]*\}", line), f"{f.name}:{n} nested quotes in an f-string"


def test_int8_loads_on_cpu_and_quantizes_block_by_block_on_the_gpu():
    """The stock fork put the whole ~10 GB bf16 DiT on the GPU before quantizing, which a 10 GB card cannot hold."""
    net = (SD / "moss_train_network.py").read_text(encoding="utf-8")
    assert 'loading_device = "cpu" if (self.is_swapping_blocks or args.moss_int8)' in net
    assert "quantize_blocks_int8(args.moss_int8_bwd, device=accelerator.device)" in net
    models = (SD / "library" / "moss_image_models.py").read_text(encoding="utf-8")
    assert "weight.data.to(target)" in models
    P = recommend.MOSS_VRAM_PROFILES["10-int8"]
    assert P["model"]["moss_int8"] == "on" and P["model"]["moss_blocks_to_swap"] == 0
