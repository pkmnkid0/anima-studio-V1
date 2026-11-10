"""KohakuMoss engine: config generation, preset tables, the runner, routes and the Train tab's defaults.

No GPU, torch or KohakuMoss environment is needed: the generated config files are checked as data (ast), against the
knob names and preset layers of the vendored kohaku_backend/, and the runner is driven with a fake `kogine`.
"""
from __future__ import annotations

import ast
import json
import re
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KOHAKU = ROOT / "kohaku_backend"

from app import train_config  # noqa: E402
from app.services import kohaku_setup, recommend  # noqa: E402
from native_backend import kohaku_config as kc  # noqa: E402
from native_backend import kohaku_env, kohaku_runner, server  # noqa: E402


# ------------------------------------------------------------------------------------------ fixtures

@pytest.fixture()
def weights(tmp_path):
    w = tmp_path / "models"
    for sub, name in (("diffusion_models", "moss_image_base_int8_convrot"), ("text_encoders", "qwen3.5_9b_int8_convrot"),
                      ("vae", "flux2-vae_small")):
        (w / sub).mkdir(parents=True)
        (w / sub / f"{name}.safetensors").write_bytes(b"x")
    return w


@pytest.fixture()
def dataset(tmp_path):
    a, b = tmp_path / "chars", tmp_path / "extra"
    a.mkdir(), b.mkdir()
    for i in range(3):
        (a / f"{i}.png").write_bytes(b"x")
        (a / f"{i}.txt").write_text("a girl", encoding="utf-8")
    (b / "z.jpg").write_bytes(b"x")
    return [{"image_dir": str(a), "num_repeats": 2, "caption_extension": ".txt"},
            {"image_dir": str(b), "num_repeats": 1}]


def _args(weights, **over):
    a = {k: v for k, v in train_config.KOHAKU_DEFAULTS.items()}
    a["kohaku_weights"] = str(weights)
    a.update(over)
    return a


def _knobs(text: str) -> dict:
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def _script_knobs() -> set[str]:
    tree = ast.parse((KOHAKU / "scripts" / "train" / "moss.py").read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            names.add(node.targets[0].id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {n for n in names if n.isupper()}


# ---------------------------------------------------------------------------------------- vendoring

def test_vendored_trainer_is_complete_and_licensed():
    assert kohaku_env.has_trainer()
    assert (KOHAKU / "LICENSE").read_text(encoding="utf-8").startswith("                               KohakuMoss License")
    assert (KOHAKU / "pyproject.toml").is_file() and (KOHAKU / "setup.py").is_file()
    for layer in ("_base/base_lora.py", "algo/algo_lora.py", "vram/vram_8g.py", "precision/precision_bf16.py"):
        assert (KOHAKU / "configs" / "moss" / layer).is_file()


def test_env_paths_mirror_each_other():
    assert kohaku_setup.kohaku_env.DIR == kohaku_env.KOHAKU_DIR
    assert kohaku_setup.kohaku_env.VENV == kohaku_env.VENV
    assert kohaku_setup.kohaku_env.venv_python() == kohaku_env.venv_python()
    assert kohaku_setup.kohaku_env.exe("kogine") == kohaku_env.kogine_exe()


# ------------------------------------------------------------------------------------ preset tables

def _layer(group: str, stem: str) -> dict:
    return kc.layer_knobs(KOHAKU / "configs" / "moss" / group / f"{stem}.py")


@pytest.mark.parametrize("key,stem", [("6", "vram_6g"), ("8", "vram_8g"), ("12", "vram_12g"), ("16", "vram_16g"),
                                      ("24", "vram_24g"), ("auto", "vram_auto")])
def test_vram_profiles_equal_the_layer_files(key, stem):
    base = _layer("_base", "base_lora")
    knobs = {**base, **_layer("vram", stem)}
    m = recommend.KOHAKU_VRAM_PROFILES[key]["model"]
    assert m["kohaku_memory_profile"] == knobs["MEMORY_PROFILE"]
    if "MAX_TOKENS" in _layer("vram", stem):
        assert m["kohaku_max_tokens"] == knobs["MAX_TOKENS"]
    te = knobs["TEXT_ENCODE"]
    assert m["kohaku_te_memory"] == te.get("memory", "")
    assert m["kohaku_te_ahead"] == te.get("ahead", "")
    assert m["kohaku_te_batch"] == te.get("batch", "")
    assert m["kohaku_te_max_tokens"] == te.get("max_tokens", "")
    vae = _layer("vram", stem).get("VAE_ENCODE", {})
    assert m["kohaku_vae_memory"] == vae.get("memory", "")
    assert m["kohaku_vae_ahead"] == vae.get("ahead", "")
    assert m["kohaku_vae_tile"] == {True: "true", False: "false"}.get(vae.get("tile"), "")
    assert m["kohaku_save_minutes"] == _layer("vram", stem).get("SAVE_MINUTES", "")


def test_default_state_is_the_8gb_layer():
    d = train_config.KOHAKU_DEFAULTS
    p = recommend.KOHAKU_VRAM_PROFILES["8"]["model"]
    for k, v in p.items():
        assert d[k] == v, k
    assert d["kohaku_vram"] == "8"


def test_profile_pick_follows_kohakumoss_rule():
    f = recommend.kohaku_profile_for
    assert f(10) == "8" and f(8) == "8" and f(7.6) == "8" and f(12) == "12" and f(11.5) == "12"
    assert f(16) == "16" and f(24) == "24" and f(32) == "auto" and f(5) == "6" and f(None) == "8"


def test_algorithm_defaults_in_the_ui_match_the_algo_layers():
    js = (ROOT / "app/static/js/train.js").read_text(encoding="utf-8")
    for algo, stem in (("lora", "algo_lora"), ("loha", "algo_loha"), ("lokr", "algo_lokr"), ("oft", "algo_oft"), ("dora", "algo_dora")):
        row = re.search(r"\b%s: \{ kohaku_dim: ([\d.]+), kohaku_alpha: ([\d.]+), kohaku_lr: ([\d.e-]+) \}" % algo, js)
        assert row, algo
        layer = _layer("algo", stem)
        assert float(row.group(1)) == layer["DIM"] and float(row.group(2)) == layer["ALPHA"]
        assert float(row.group(3)) == layer.get("LR", 1e-4)


# ---------------------------------------------------------------------------------- UI <-> backend

def test_js_defaults_equal_python_defaults():
    js = (ROOT / "app/static/js/train.js").read_text(encoding="utf-8")
    block = js.split("// <kohaku-defaults>")[1].split("// </kohaku-defaults>")[0]
    found = {}
    for m in re.finditer(r"^\s+(kohaku_\w+): (.+),$", block, re.M):
        found[m.group(1)] = json.loads(m.group(2))
    assert found == train_config.KOHAKU_DEFAULTS


def test_payload_roundtrip_and_engine_gating():
    ui = train_config.default_ui_payload()
    assert ui["model"]["moss_engine"] == "fork"
    ui["model_type"] = "moss"
    fork = train_config.build_backend_payload(ui)["args"]
    assert "moss_args" in fork and "kohaku_args" not in fork
    ui["model"]["moss_engine"] = "kohaku"
    ui["model"]["kohaku_weights"] = "E:/models"
    ui["model"]["kohaku_steps"] = 123
    payload = train_config.build_backend_payload(ui)
    assert "kohaku_args" in payload["args"] and "moss_args" not in payload["args"] and "anima_args" not in payload["args"]
    assert payload["args"]["kohaku_args"]["kohaku_steps"] == 123
    back = train_config.to_ui_payload(payload)
    assert back["model_type"] == "moss" and back["model"]["moss_engine"] == "kohaku"
    assert back["model"]["kohaku_weights"] == "E:/models" and back["model"]["kohaku_steps"] == 123
    train_config.to_toml_document(payload)          # serialises


def test_server_picks_the_kohaku_runner():
    assert server._pick_runner("lora", False, False, False, False, True) == "kohaku"
    assert server._pick_runner("lora", False, False, True, False, True) is None
    assert server._pick_runner("full", False, False, False, False, True) == "kohaku"
    assert server._pick_runner("lora", False, False, False, True) == "moss"      # unchanged


# ----------------------------------------------------------------------------------- config build

def test_build_writes_valid_configs_with_only_real_knobs(weights, dataset, tmp_path):
    b = kc.build(_args(weights), dataset, str(tmp_path / "out"), "My Char v1", user_dir=KOHAKU / "configs" / "user")
    k = _knobs(b.train_text)
    assert not set(k) - _script_knobs(), set(k) - _script_knobs()          # every knob exists in scripts/train/moss.py
    assert k["RUN_NAME"] == "My-Char-v1" and k["STEPS"] == 300 and k["GRAD_ACC"] == 4
    assert k["MEMORY_PROFILE"] == "minimal" and k["MAX_TOKENS"] == 4352
    assert k["TARGET_AREA"] == 1024 * 1024 and k["CAPTION"] == {"template": "nl"}
    assert k["DATASETS"] == [
        {"source": "folder", "path": (tmp_path / "chars").as_posix(), "txt_as": "nl", "recursive": False, "weight": 2.0},
        {"source": "folder", "path": (tmp_path / "extra").as_posix(), "txt_as": "nl", "recursive": False, "weight": 1.0}]
    assert k["CACHE_DIR"] == (tmp_path / "out" / "kohaku_cache").as_posix()
    assert k["TEXT_ENCODE"] == {"memory": "ring", "ahead": 8, "batch": 16, "max_tokens": 8192}
    assert k["VAE_ENCODE"] == {"memory": "offload", "ahead": 8, "tile": True}
    assert k["SAMPLE_EVERY"] == 0 and k["SAVE_FORMAT"] == "comfy"
    # the layers come first, in KohakuMoss' own order, as relative paths inside the repository
    uses = re.findall(r'^use_config\("([^"]+)"\)$', b.train_text, re.M)
    assert uses == ["../moss/_base/base_lora.py", "../moss/algo/algo_lora.py", "../moss/vram/vram_8g.py",
                    "../moss/precision/precision_bf16.py"]
    d = _knobs(b.data_text)
    assert d["TEXT_ENCODER"] == "kohakumoss.encoders.text_encoder"
    assert d["TEXT_ENCODER_KWARGS"]["path"].endswith("qwen3.5_9b_int8_convrot.safetensors")
    assert d["IMAGE_ENCODER_KWARGS"]["path"].endswith("flux2-vae_small.safetensors")
    assert [p[1] for p in kc.passes(b)] == ["scripts/data/inspect_data.py", "scripts/data/cache_text.py",
                                            "scripts/data/cache_latents.py", "scripts/train/moss.py"]


def test_algorithms_and_precision_pick_their_layers(weights, dataset, tmp_path):
    for algo in ("lokr", "loha", "oft", "dora"):
        b = kc.build(_args(weights, kohaku_algo=algo), dataset, str(tmp_path), "n", user_dir=KOHAKU / "configs" / "user")
        assert f"../moss/algo/algo_{algo}.py" in b.train_text
    b = kc.build(_args(weights, kohaku_algo="dora"), dataset, str(tmp_path), "n", user_dir=KOHAKU / "configs" / "user")
    k = _knobs(b.train_text)
    assert k["ALGO"] == "lora" and k["DORA"] is True
    b = kc.build(_args(weights, kohaku_algo="glora"), dataset, str(tmp_path), "n", user_dir=KOHAKU / "configs" / "user")
    assert "algo_lora.py" in b.train_text and _knobs(b.train_text)["ALGO"] == "glora"
    b = kc.build(_args(weights, kohaku_precision="fp16", kohaku_fp16_fix="online"), dataset, str(tmp_path), "n",
                 user_dir=KOHAKU / "configs" / "user")
    assert "precision_fp16.py" in b.train_text and _knobs(b.train_text)["FP16_FIX"] == "online"
    b = kc.build(_args(weights, kohaku_algo="full", kohaku_vram="24"), dataset, str(tmp_path), "n", user_dir=KOHAKU / "configs" / "user")
    k = _knobs(b.train_text)
    assert "base_full.py" in b.train_text and "vram_24g_full.py" in b.train_text
    assert k["ALGO"] == "full" and "DIM" not in k and "DORA" not in k


@pytest.mark.parametrize("vram", ["6", "8", "12", "16", "24", "auto"])
def test_every_vram_layer_builds(weights, dataset, tmp_path, vram):
    b = kc.build(_args(weights, kohaku_vram=vram), dataset, str(tmp_path), "n", user_dir=KOHAKU / "configs" / "user")
    assert f"../moss/vram/vram_{vram}{'g' if vram != 'auto' else ''}.py" in b.train_text


def test_tags_mode_and_sampling_and_extras(weights, dataset, tmp_path):
    b = kc.build(_args(weights, kohaku_caption_mode="tags", kohaku_caption_tag_dropout=0.1, kohaku_sample_every=50,
                       kohaku_sample_prompts="a cat\n\na dog", kohaku_sample_seeds="1, 2", kohaku_sample_skip_layers="22 23",
                       kohaku_extra_knobs="# comment\nSTREAM_LOSS = {\"rms\": {\"weight\": 0.001}}\nSEED = 7\n"),
                 dataset, str(tmp_path), "n", user_dir=KOHAKU / "configs" / "user")
    k = _knobs(b.train_text)
    assert k["CAPTION"]["template"] == "tags" and k["CAPTION"]["tag_dropout"] == 0.1 and k["CAPTION"]["keep"] == []
    assert k["DATASETS"][0]["txt_as"] == "tags"
    assert k["SAMPLE_PROMPTS"] == ["a cat", "a dog"] and k["SAMPLE_SEEDS"] == (1, 2) and k["SAMPLE_SKIP_LAYERS"] == (22, 23)
    assert k["STREAM_LOSS"] == {"rms": {"weight": 0.001}} and k["SEED"] == 7          # extras win
    assert not set(k) - _script_knobs()


def test_manual_memory_knobs_only_with_manual_profile(weights, dataset, tmp_path):
    b = kc.build(_args(weights, kohaku_stream="ring", kohaku_act_offload="true"), dataset, str(tmp_path), "n",
                 user_dir=KOHAKU / "configs" / "user")
    assert "STREAM" not in _knobs(b.train_text)                       # a profile is set: they must stay unset
    b = kc.build(_args(weights, kohaku_memory_profile="manual", kohaku_stream="ring", kohaku_stream_resident=2,
                       kohaku_act_offload="true", kohaku_checkpoint="block"), dataset, str(tmp_path), "n",
                 user_dir=KOHAKU / "configs" / "user")
    k = _knobs(b.train_text)
    assert k["STREAM"] == "ring" and k["STREAM_RESIDENT"] == 2 and k["ACT_OFFLOAD"] is True and k["CHECKPOINT"] == "block"
    assert not set(k) - _script_knobs()


def test_user_text_cannot_become_code(weights, dataset, tmp_path):
    evil = 'x"\n__import__("os").system("echo hacked")\n#'
    b = kc.build(_args(weights, kohaku_sample_every=10, kohaku_sample_prompts=evil, kohaku_resume=evil),
                 dataset, str(tmp_path), evil, user_dir=KOHAKU / "configs" / "user")
    tree = ast.parse(b.train_text)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert [c.func.id for c in calls] == ["use_config"] * 4          # nothing but the four layer calls
    assert _knobs(b.train_text)["RESUME"] == evil
    for bad in ("STEPS = __import__('os')", "lower = 1", "NAME 5"):
        with pytest.raises(kc.KohakuConfigError):
            kc.parse_extra_knobs(bad)


@pytest.mark.parametrize("over,needle", [
    ({"kohaku_weights": ""}, "weights folder"), ({"kohaku_steps": 0}, "Steps"), ({"kohaku_dim": "abc"}, "Rank"),
    ({"kohaku_algo": "nope"}, "Unknown algorithm"), ({"kohaku_vram": "99"}, "VRAM profile"),
    ({"kohaku_sample_every": 10, "kohaku_sample_prompts": ""}, "no sample prompts"), ({"kohaku_resolution": 100}, "Resolution"),
    ({"kohaku_caption_mode": "x"}, "Caption style")])
def test_bad_settings_give_readable_errors(weights, dataset, tmp_path, over, needle):
    with pytest.raises(kc.KohakuConfigError, match=needle):
        kc.build(_args(weights, **over), dataset, str(tmp_path), "n", user_dir=KOHAKU / "configs" / "user")


def test_bad_dataset_and_output_errors(weights, dataset, tmp_path):
    ud = KOHAKU / "configs" / "user"
    with pytest.raises(kc.KohakuConfigError, match="at least one dataset"):
        kc.build(_args(weights), [], str(tmp_path), "n", user_dir=ud)
    with pytest.raises(kc.KohakuConfigError, match="output folder"):
        kc.build(_args(weights), dataset, "", "n", user_dir=ud)
    with pytest.raises(kc.KohakuConfigError, match="missing or not a folder"):
        kc.build(_args(weights), [{"image_dir": str(tmp_path / "nope")}], str(tmp_path), "n", user_dir=ud)
    (tmp_path / "empty").mkdir()
    with pytest.raises(kc.KohakuConfigError, match="No images"):
        kc.build(_args(weights), [{"image_dir": str(tmp_path / "empty")}], str(tmp_path), "n", user_dir=ud)


def test_write_creates_both_files(weights, dataset, tmp_path):
    b = kc.build(_args(weights), dataset, str(tmp_path / "o"), "n", user_dir=tmp_path / "cfg")
    kc.write(b)
    assert b.train_path.read_text(encoding="utf-8") == b.train_text and b.data_path.is_file()


def test_slug():
    assert kc.slug("My Char v1!") == "My-Char-v1" and kc.slug("") == "anima-studio-lora" and kc.slug("../../x") == "x"


# ----------------------------------------------------------------------------------- weights layouts

def test_inspect_weights_layouts(weights, tmp_path):
    r = kc.inspect_weights(str(weights))
    assert r["layout"] == "comfy" and not r["problems"] and r["te"].endswith("qwen3.5_9b_int8_convrot.safetensors")
    assert kc.inspect_weights(str(weights), text_encoder="nope")["problems"]
    (weights / "text_encoders" / "mine.safetensors").write_bytes(b"x")
    assert kc.inspect_weights(str(weights), text_encoder="mine")["te"].endswith("mine.safetensors")
    assert kc.inspect_weights("")["problems"] and kc.inspect_weights(str(tmp_path / "missing"))["problems"]
    assert kc.inspect_weights(str(tmp_path))["layout"] == ""                      # neither layout

    hf = tmp_path / "hf"
    for sub in ("transformer", "text_encoder", "vae", "tokenizer"):
        (hf / sub).mkdir(parents=True)
    r = kc.inspect_weights(str(hf))
    assert r["layout"] == "hf" and not r["problems"]
    root = tmp_path / "root"
    (root / "base" / "transformer").mkdir(parents=True)
    (root / "base" / "text_encoder").mkdir()
    (root / "base" / "vae").mkdir()
    assert kc.inspect_weights(str(root))["layout"] == "hf"

    single = tmp_path / "dit.safetensors"
    single.write_bytes(b"x")
    assert kc.inspect_weights(str(single))["problems"]                                  # needs a TE and a VAE
    te, va = tmp_path / "te.safetensors", tmp_path / "vae.safetensors"
    te.write_bytes(b"x"), va.write_bytes(b"x")
    r = kc.inspect_weights(str(single), text_encoder=str(te), vae=str(va))
    assert r["layout"] == "file" and not r["problems"]


# ---------------------------------------------------------------------------------------- the runner

FAKE_KOGINE = """#!/bin/sh
# fake kogine: logs, writes metrics rows, and creates the export files like KohakuMoss' train script does
echo "kogine $@"
case "$2" in
  *train/moss.py)
    mkdir -p "$FAKE_OUT/$FAKE_NAME/export"
    printf '{"step": 1, "time": %s, "loss": 0.5, "step_time": 2.0}\\n' "$(date +%s)" > "$FAKE_OUT/$FAKE_NAME/metrics.jsonl"
    sleep 3
    echo "step 2 [loss=0.4321, grad_norm=1.0, lr=1e-4]"
    echo fake > "$FAKE_OUT/$FAKE_NAME/export/$FAKE_NAME-comfy.safetensors"
    [ "$FAKE_FAIL" = "train" ] && exit 3
    ;;
  *cache_text.py) [ "$FAKE_FAIL" = "text" ] && { echo "boom"; exit 2; } ;;
esac
exit 0
"""


@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("fake kogine is a shell script")
    exe = tmp_path / "kogine"
    exe.write_text(FAKE_KOGINE, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(kohaku_env, "kogine_exe", lambda: exe)
    monkeypatch.setattr(kohaku_env, "has_kogine", lambda: True)
    monkeypatch.setattr(kohaku_env, "USER_CONFIGS", tmp_path / "userconf")
    monkeypatch.setenv("FAKE_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("FAKE_NAME", "My-Char-v1")
    return tmp_path


def _payload(weights, dataset, tmp_path):
    return ({"kohaku_args": _args(weights), "saving_args": {"output_dir": str(tmp_path / "out"), "output_name": "My Char v1"}},
            {"subsets": dataset})


def test_runner_runs_the_passes_and_copies_the_lora(fake_env, weights, dataset, capsys):
    args, ds = _payload(weights, dataset, fake_env)
    seen = []
    kohaku_runner.run(args, ds, lambda **kw: seen.append(kw), lambda: False)
    out = capsys.readouterr().out
    runs = [l for l in out.splitlines() if l.startswith("kogine run")]
    assert len(runs) == 4, out
    for line, script in zip(runs, ("inspect_data.py", "cache_text.py", "cache_latents.py", "train/moss.py")):
        assert "run scripts/" in line and script in line and "--config" in line, line
    assert out.index("inspect_data.py") < out.index("cache_text.py") < out.index("cache_latents.py") < out.index("train/moss.py")
    assert "LoRA saved" in out
    assert (fake_env / "out" / "My-Char-v1.safetensors").read_text().strip() == "fake"
    assert any(p.get("step") == 2 and p.get("loss") == pytest.approx(0.4321) for p in seen)
    assert seen[-1]["step"] == seen[-1]["total_steps"] == 300
    assert (fake_env / "userconf" / "studio_My-Char-v1.py").is_file() and (fake_env / "userconf" / "studio_My-Char-v1_data.py").is_file()


def test_runner_reports_the_failing_pass(fake_env, weights, dataset, monkeypatch):
    args, ds = _payload(weights, dataset, fake_env)
    monkeypatch.setenv("FAKE_FAIL", "text")
    with pytest.raises(kohaku_runner.KohakuError, match="Caching caption embeddings.*exit code 2"):
        kohaku_runner.run(args, ds, lambda **kw: None, lambda: False)
    monkeypatch.setenv("FAKE_FAIL", "train")
    with pytest.raises(kohaku_runner.KohakuError, match="Training failed"):
        kohaku_runner.run(args, ds, lambda **kw: None, lambda: False)


def test_runner_refuses_without_environment_or_with_bad_settings(fake_env, weights, dataset, monkeypatch):
    args, ds = _payload(weights, dataset, fake_env)
    monkeypatch.setattr(kohaku_env, "has_kogine", lambda: False)
    with pytest.raises(kohaku_runner.KohakuError, match="Backend tab"):
        kohaku_runner.run(args, ds, lambda **kw: None, lambda: False)
    monkeypatch.setattr(kohaku_env, "has_kogine", lambda: True)
    args["kohaku_args"]["kohaku_steps"] = 0
    with pytest.raises(kohaku_runner.KohakuError, match="Steps"):
        kohaku_runner.run(args, ds, lambda **kw: None, lambda: False)


def test_worker_dispatches_kohaku(fake_env, weights, dataset, tmp_path):
    import subprocess
    args, ds = _payload(weights, dataset, fake_env)
    pf = tmp_path / "payload.json"
    pf.write_text(json.dumps({"args": args, "dataset": ds}), encoding="utf-8")
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    code = (f"import native_backend.kohaku_env as e, pathlib;"
            f"e.kogine_exe=lambda: pathlib.Path({str(fake_env / 'kogine')!r});e.has_kogine=lambda: True;"
            f"e.USER_CONFIGS=pathlib.Path({str(fake_env / 'userconf')!r});"
            f"import sys;sys.argv=['w',{str(pf)!r},'kohaku'];"
            f"from native_backend import train_worker;raise SystemExit(train_worker.main())")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**env, "FAKE_OUT": str(fake_env / "out"),
                                                                                           "FAKE_NAME": "My-Char-v1"}, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "@@PROGRESS" in r.stdout and "LoRA saved" in r.stdout


# ---------------------------------------------------------------------------------- setup + verify

def test_torch_index_follows_the_driver():
    smi = "| NVIDIA-SMI 581.15  Driver Version: 581.15  CUDA Version: {v} |"
    assert kohaku_setup.torch_index_tag(smi.format(v="13.0")) == "cu130"
    assert kohaku_setup.torch_index_tag(smi.format(v="13.1")) == "cu130"
    assert kohaku_setup.torch_index_tag(smi.format(v="12.9")) == "cu128"
    assert kohaku_setup.torch_index_tag(smi.format(v="12.8")) == "cu128"
    assert kohaku_setup.torch_index_tag(smi.format(v="12.6")) == "cu126"
    assert kohaku_setup.torch_index_tag("CUDA UMD Version: 13.2") == "cu130"
    with pytest.raises(kohaku_setup.SetupError, match="12.6"):
        kohaku_setup.torch_index_tag(smi.format(v="12.4"))
    with pytest.raises(kohaku_setup.SetupError, match="Could not read"):
        kohaku_setup.torch_index_tag("garbage")


def test_doctor_report_becomes_a_checklist():
    rep = {"packages": {"torch": "2.14.1", "kohaku-engine": "0.2", "anyschedule": "1", "kohakuvault": "1.1", "lycoris-lora": "4.1.3",
                        "diffusers": "0.38", "transformers": "5.3", "triton-windows": "3.8.0", "tilelang": "0.1.15"},
           "torch_cuda": "13.0", "devices": [{"name": "RTX 3080", "memory_gib": 10.0, "arch": "sm86"}],
           "msvc": {"needed": True, "cl_on_path": False, "vswhere_sees_vs": False}, "missing": [
               {"what": "Windows SDK missing", "fix": "winget ..."}], "backend_failures": {"triton:attn": "cl.exe not found"}}
    by = {}
    for c in kohaku_setup.doctor_checks(rep):
        by.setdefault(c["id"], []).append(c)
    assert by["torch"][0]["status"] == "ok" and by["gpu"][0]["status"] == "ok" and by["packages"][0]["status"] == "ok"
    assert by["msvc"][0]["status"] == "warn" and by["missing"][0]["status"] == "warn" and by["backend"][0]["status"] == "warn"
    rep2 = {**rep, "devices": [], "packages": {"torch": "2.9.0"}, "missing": [{"what": "CPU-only torch next to an NVIDIA GPU", "fix": "x"}]}
    by2 = {c["id"]: c for c in kohaku_setup.doctor_checks(rep2)}
    assert by2["gpu"]["status"] == "fail" and by2["packages"]["status"] == "fail" and by2["torch"]["status"] == "warn"
    assert by2["missing"]["status"] == "fail"


def test_verify_without_environment_is_a_clean_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(kohaku_setup.kohaku_env, "VENV", tmp_path / "novenv")
    r = kohaku_setup.verify()
    assert not r["ok"] and any(c["id"] == "venv" and c["status"] == "fail" for c in r["checks"])
    assert not kohaku_setup.status()["installed"]


# ------------------------------------------------------------------------------------------ routes

def test_routes(client, weights, monkeypatch):
    r = client.get("/api/train/kohaku/profiles?vram_gb=10").get_json()
    assert r["suggested"] == "8" and set(r["profiles"]) == {"6", "8", "12", "16", "24", "auto"}
    st = client.get("/api/train/kohaku/setup/status").get_json()
    assert st["trainer_present"] and [s["id"] for s in st["steps"]] == ["venv", "torch", "package", "verify"]
    c = client.post("/api/train/kohaku/check", json={"weights": str(weights)}).get_json()
    assert c["layout"] == "comfy" and not c["problems"]
    assert client.post("/api/train/kohaku/check", json={"weights": "/nope"}).get_json()["problems"]
    monkeypatch.setattr(kohaku_setup, "start", lambda: (_ for _ in ()).throw(kohaku_setup.SetupError("no driver")))
    assert client.post("/api/train/kohaku/setup/start").status_code == 400
    assert client.get("/api/train/kohaku/verify").get_json()["ok"] is False             # no venv in the test tree


def test_build_warns_about_kohaku_problems(client, weights, dataset):
    ui = train_config.default_ui_payload()
    ui["model_type"] = "moss"
    ui["model"]["moss_engine"] = "kohaku"
    ui["subsets"] = [{**dataset[0], "name": "chars"}]
    res = client.post("/api/train/build", json=ui).get_json()
    warnings = " ".join(res.get("warnings") or [])
    assert "Low-VRAM trainer weights folder" in warnings
    ui["model"]["kohaku_weights"] = str(weights)
    res = client.post("/api/train/build", json=ui).get_json()
    warnings = " ".join(res.get("warnings") or [])
    assert "weights folder" not in warnings and "DiT model path" not in warnings and "Qwen3" not in warnings


def test_ui_has_the_kohaku_pieces(client):
    html = client.get("/").get_data(as_text=True)
    for needle in ("kohaku-env-card", "kohaku-setup-btn", "js/envcard.js", "js/kohakuenv.js"):
        assert needle in html
    js = (ROOT / "app/static/js/train.js").read_text(encoding="utf-8")
    for needle in ("buildKohakuSection", "engineToggle", "kohaku_extra_knobs", "Trainer engine"):
        assert needle in js


def test_epochs_become_steps(weights, dataset, tmp_path):
    # dataset: 2 repeats of the first folder + 1 of the second; 1 MP images pack 1 per batch, accumulation 4
    imgs = sum(1 for d in dataset for f in Path(d["image_dir"]).iterdir() if f.suffix.lower() in kc.IMAGE_EXT
               for _ in range(int(d.get("num_repeats", 1))))
    per_epoch = kc.steps_per_epoch(imgs, "tokens", 1024, 4352, 4, 4)
    b = kc.build(_args(weights, kohaku_duration="epochs", kohaku_epochs=3, kohaku_grad_acc=4, kohaku_resolution=1024,
                       kohaku_max_tokens=4352, kohaku_save_every_epochs=1), dataset, str(tmp_path / "o"), "n",
                 user_dir=KOHAKU / "configs" / "user")
    import math
    assert b.steps == max(1, math.ceil(3 * per_epoch))
    assert b.knobs["STEPS"] == b.steps
    assert b.knobs["SAVE_EVERY"] == max(0, round(per_epoch))
    assert any("epochs" in n for n in b.notes)
    # steps mode is unchanged
    b2 = kc.build(_args(weights, kohaku_duration="steps", kohaku_steps=77), dataset, str(tmp_path / "o"), "n",
                  user_dir=KOHAKU / "configs" / "user")
    assert b2.steps == 77
    with pytest.raises(kc.KohakuConfigError):
        kc.build(_args(weights, kohaku_duration="epochs", kohaku_epochs=0), dataset, str(tmp_path / "o"), "n",
                 user_dir=KOHAKU / "configs" / "user")


def test_steps_per_epoch_count_mode():
    assert kc.steps_per_epoch(40, "count", 1024, 4352, 4, 2) == 5.0
    assert kc.steps_per_epoch(1, "tokens", 1024, 4352, 4, 8) == 1.0
