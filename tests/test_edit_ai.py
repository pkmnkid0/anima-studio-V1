"""The Edit tab's AI tools: catalogue, requests, engine (against a fake worker), batch, routes, images and the UI wiring."""
import base64
import io
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from PIL import Image

from edit_backend import catalog
from edit_backend import env as edit_env

ROOT = Path(__file__).resolve().parent.parent

FAKE_WORKER = textwrap.dedent('''
    import json, sys, time
    from PIL import Image

    def send(o):
        sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()

    send({"event": "ready"})
    for raw in sys.stdin:
        req = json.loads(raw)
        rid, cmd = req["id"], req["cmd"]
        if cmd == "shutdown":
            send({"id": rid, "ok": True, "result": {}}); break
        try:
            if cmd == "plugin" and req.get("model") == "slow":
                time.sleep(60)
            if cmd == "plugin" and req.get("model") == "boom":
                send({"id": rid, "ok": False, "error": "Out of GPU memory."}); continue
            if cmd in ("inpaint", "segment", "plugin"):
                src = Image.open(req["image_path"]).convert("RGB")
                if cmd == "segment":
                    out = Image.new("L", src.size, 255)
                elif cmd == "plugin" and req["plugin"] == "removebg" and req["op"] == "image":
                    out = src.convert("RGBA"); out.putalpha(Image.new("L", src.size, 0))
                else:
                    out = Image.new("RGB", src.size, (200, 10, 10))
                if cmd == "plugin" and req["plugin"] == "realesrgan":
                    out = out.resize((int(src.width * req["scale"]), int(src.height * req["scale"])))
                send({"id": rid, "event": "progress", "step": 1, "total": 2})
                out.save(req["out_path"])
                send({"id": rid, "ok": True, "result": {"out_path": req["out_path"], "has_alpha": out.mode == "RGBA", "seen": sorted(req)}})
            else:
                send({"id": rid, "ok": True, "result": {"cmd": cmd, "downloaded": ["lama"], "plugins": {}, "loaded": None}})
        except Exception as exc:
            send({"id": rid, "ok": False, "error": str(exc)})
''')


# ------------------------------------------------------------------ helpers

def _png(size=(64, 48), color=(10, 120, 200), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mask_url(size=(64, 48), box=(10, 10, 30, 30)):
    m = Image.new("L", size, 0)
    m.paste(255, box)
    buf = io.BytesIO()
    m.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture()
def fake_engine(tmp_path, monkeypatch):
    from app.services import edit_engine

    root = tmp_path / "fakeroot"
    (root / "edit_backend").mkdir(parents=True)
    (root / "edit_backend" / "__init__.py").write_text("")
    (root / "edit_backend" / "worker.py").write_text(FAKE_WORKER)
    monkeypatch.setattr(edit_env, "ROOT", root)
    monkeypatch.setattr(edit_env, "venv_python", lambda: Path(sys.executable))
    monkeypatch.setattr(edit_engine, "is_installed", lambda: True)
    monkeypatch.setattr(edit_engine, "TMP_DIR", tmp_path / "edit_tmp")
    yield edit_engine
    edit_engine.stop("test over")
    time.sleep(0.3)


@pytest.fixture()
def folder(tmp_path):
    d = tmp_path / "set"
    d.mkdir()
    for n in ("a.png", "b.png"):
        (d / n).write_bytes(_png())
    (d / "c.jpg").write_bytes(_png(mode="RGB"))
    return d


def _wait(engine, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        j = engine.job(job_id)
        if j["state"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


# ------------------------------------------------------------------ catalogue

def test_catalogue_is_consistent():
    keys = [p["key"] for p in catalog.PARAMS]
    assert len(keys) == len(set(keys)), "duplicate setting keys"
    groups = {g for g, _ in catalog.GROUPS}
    kinds = {"select", "int", "float", "bool", "area", "image"}
    for p in catalog.PARAMS:
        assert p["group"] in groups and p["kind"] in kinds and p["models"], p["key"]
        if p["kind"] == "select":
            assert p["default"] in p["options"], p["key"]
        if p["kind"] in ("int", "float") and "min" in p:
            assert p["min"] <= p["default"] <= p["max"], p["key"]
    assert {m["id"] for m in catalog.ERASE_MODELS} >= {"lama", "ldm", "zits", "mat", "fcf", "manga", "migan"}
    assert any(m["id"] == "Fantasy-Studio/Paint-by-Example" for m in catalog.DIFFUSION_MODELS)
    pub = catalog.public_catalog()
    assert pub["defaults"]["sd_seed"] == 42 and "custom_model_tags" in pub
    assert catalog.model_info("some/custom-repo")["tags"] == catalog.CUSTOM_MODEL_TAGS
    assert catalog.is_erase_model("lama") and not catalog.is_erase_model("runwayml/stable-diffusion-inpainting")


def test_nsfw_checker_is_off_by_default():
    opt = {o["key"]: o for o in catalog.ENGINE_OPTIONS}
    assert opt["disable_nsfw"]["default"] is True


def test_catalogue_matches_the_real_iopaint_schema():
    """Runs only where IOPaint is importable (the edit environment): every InpaintRequest field has a control."""
    schema = pytest.importorskip("iopaint.schema")
    fields = set(schema.InpaintRequest.model_fields) - set(catalog.RESERVED_FIELDS)
    assert fields == set(catalog.PARAM_KEYS)


# ------------------------------------------------------------------ requests

def test_build_request_validates(fake_engine, folder):
    e = fake_engine
    img = str(folder / "a.png")
    with pytest.raises(e.EngineError):
        e.build_request({"cmd": "rm -rf"})
    with pytest.raises(e.EngineError):
        e.build_request({"cmd": "inpaint", "image_path": img, "model": "../../etc/passwd", "mask": _mask_url()})
    with pytest.raises(e.EngineError):
        e.build_request({"cmd": "inpaint", "image_path": str(folder / "nope.png"), "model": "lama", "mask": _mask_url()})
    with pytest.raises(e.EngineError):
        e.build_request({"cmd": "inpaint", "image_path": str(folder / "a.txt"), "model": "lama"})
    with pytest.raises(e.EngineError):
        e.build_request({"cmd": "segment", "image_path": img, "clicks": []})
    with pytest.raises(e.EngineError):
        e.build_request({"cmd": "segment", "image_path": img, "clicks": [["x", 1, 1]]})
    with pytest.raises(e.EngineError):
        e.build_request({"cmd": "plugin", "plugin": "evil", "image_path": img})
    with pytest.raises(e.EngineError):
        e.build_request({"cmd": "plugin", "plugin": "realesrgan", "image_path": img, "scale": 99})
    with pytest.raises(e.EngineError):
        e.build_request({"cmd": "download", "kind": "sam; rm", "name": "x"})


def test_build_request_shapes_and_cleans_options(fake_engine, folder):
    e = fake_engine
    cmd, args, tmp = e.build_request({
        "cmd": "inpaint", "image_path": str(folder / "a.png"), "model": "lama", "mask": _mask_url(),
        "options": {"device": "cpu", "no_half": 1, "bogus": "x", "device_x": 1}, "params": {"hd_strategy": "Crop"}})
    assert cmd == "inpaint" and args["options"] == {"device": "cpu", "no_half": True}
    assert Path(args["mask_path"]).is_file() and args["params"] == {"hd_strategy": "Crop"}
    for p in tmp:
        p.unlink()
    cmd, args, _ = e.build_request({"cmd": "segment", "image_path": str(folder / "a.png"),
                                    "clicks": [[1.0, 2, True], [3, 4, 0]]})
    assert args["clicks"] == [[1, 2, 1], [3, 4, 0]] and args["model"] == "mobile_sam"


def test_options_with_a_bad_device_fall_back_to_the_default(fake_engine):
    assert fake_engine._clean_options({"device": "tpu"}) == {"device": "auto"}


# ------------------------------------------------------------------ the engine against a fake worker

def test_inpaint_job_end_to_end(fake_engine, folder):
    e = fake_engine
    job = e.submit({"cmd": "inpaint", "image_path": str(folder / "a.png"), "model": "lama", "mask": _mask_url()})
    done = _wait(e, job["id"])
    assert done["state"] == "done", done
    token = done["result"]["token"]
    assert "out_path" not in done["result"] and re.fullmatch(r"[0-9a-f]{32}", token)
    assert done["progress"]["step"] == 1 and done["progress"]["total"] == 2
    assert Image.open(e.result_file(token)).getpixel((0, 0)) == (200, 10, 10)
    assert e.status()["running"] and e.status()["ready"]
    assert e.discard_result(token) and e.result_file(token) is None


def test_worker_errors_reach_the_user(fake_engine, folder):
    e = fake_engine
    job = e.submit({"cmd": "plugin", "plugin": "removebg", "model": "boom", "image_path": str(folder / "a.png")})
    done = _wait(e, job["id"])
    assert done["state"] == "error" and "Out of GPU memory" in done["error"]
    assert e.status()["busy"] is None


def test_cancel_stops_the_worker_and_the_next_job_restarts_it(fake_engine, folder):
    e = fake_engine
    job = e.submit({"cmd": "plugin", "plugin": "removebg", "model": "slow", "image_path": str(folder / "a.png")})
    time.sleep(1.5)
    assert e.cancel(job["id"]) is True
    done = _wait(e, job["id"])
    assert done["state"] == "cancelled"
    job2 = e.submit({"cmd": "inpaint", "image_path": str(folder / "a.png"), "model": "lama", "mask": _mask_url()})
    assert _wait(e, job2["id"])["state"] == "done"


def test_only_one_job_at_a_time(fake_engine, folder):
    e = fake_engine
    slow = e.submit({"cmd": "plugin", "plugin": "removebg", "model": "slow", "image_path": str(folder / "a.png")})
    with pytest.raises(e.EngineError, match="busy"):
        e.submit({"cmd": "models"})
    e.cancel(slow["id"])
    _wait(e, slow["id"])


def test_not_installed_gives_a_clear_message(tmp_path, monkeypatch):
    from app.services import edit_engine

    monkeypatch.setattr(edit_engine, "is_installed", lambda: False)
    monkeypatch.setattr(edit_engine, "TMP_DIR", tmp_path / "t")
    job = edit_engine.submit({"cmd": "models"})
    done = _wait(edit_engine, job["id"])
    assert done["state"] == "error" and "Set up AI edit environment" in done["error"]


def test_result_tokens_cannot_escape_the_temp_folder(fake_engine):
    for bad in ("../secret", "a" * 32 + ".png", "", None, "G" * 32):
        assert fake_engine.result_file(bad) is None


# ------------------------------------------------------------------ images

def test_save_ai_result_in_place_backs_up_and_copy_does_not(tmp_path):
    from app.services import imaging

    src = tmp_path / "x.jpg"
    src.write_bytes(_png(color=(1, 2, 3)))
    res = tmp_path / "r.png"
    res.write_bytes(_png(color=(9, 9, 9)))
    out = imaging.save_ai_result(str(src), str(res), save_as_copy=True)
    assert out != str(src) and Path(out).is_file() and not imaging.has_backup(str(src))
    out = imaging.save_ai_result(str(src), str(res))
    assert out == str(src) and imaging.has_backup(str(src))
    assert imaging.revert_to_backup(str(src))


def test_save_ai_result_transparency(tmp_path):
    from app.services import imaging

    res = tmp_path / "cut.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(res)
    jpg = tmp_path / "x.jpg"
    jpg.write_bytes(_png((8, 8)))
    flat = Path(imaging.save_ai_result(str(jpg), str(res), save_as_copy=True, background="#00ff00"))
    assert Image.open(flat).convert("RGB").getpixel((0, 0))[1] > 200          # flattened onto green
    kept = Path(imaging.save_ai_result(str(jpg), str(res), keep_alpha=True))
    assert kept.suffix == ".png" and Image.open(kept).mode == "RGBA"
    assert jpg.exists()                                                       # the JPEG is never turned into a PNG


def test_adjust_mask():
    from app.services import imaging

    def count(url):
        return sum(1 for v in Image.open(io.BytesIO(base64.b64decode(url.split(",")[1]))).getdata() if v)

    base = _mask_url()
    assert count(imaging.adjust_mask(base, "expand", 4)) > count(base) > count(imaging.adjust_mask(base, "shrink", 3))
    assert count(imaging.adjust_mask(base, "reverse")) == 64 * 48 - count(base)
    with pytest.raises(imaging.ImagingError):
        imaging.adjust_mask(base, "explode")


# ------------------------------------------------------------------ batch

def test_batch_removebg_over_a_folder(fake_engine, folder):
    from app.services import edit_batch

    s = edit_batch.start({"folder": str(folder), "mode": "plugin", "plugin": "removebg", "save_as_copy": True,
                          "background": "#ffffff"})
    assert s["state"] == "running" and s["total"] == 3
    end = time.time() + 40
    while edit_batch.status()["state"] == "running" and time.time() < end:
        time.sleep(0.2)
    st = edit_batch.status()
    assert st["state"] == "done" and st["done"] == 3 and st["failed_count"] == 0, st
    assert len(list(folder.glob("*_edited*"))) == 3
    assert fake_engine.status()["busy"] is None


def test_batch_inpaint_needs_a_mask_and_continues_after_failures(fake_engine, folder):
    from app.services import edit_batch

    with pytest.raises(edit_batch.BatchError, match="mask"):
        edit_batch.start({"folder": str(folder), "mode": "inpaint"})
    with pytest.raises(edit_batch.BatchError):
        edit_batch.start({"folder": str(folder / "missing"), "mode": "plugin", "plugin": "removebg"})
    s = edit_batch.start({"folder": str(folder), "mode": "plugin", "plugin": "removebg", "model": "boom",
                          "save_as_copy": True})
    end = time.time() + 40
    while edit_batch.status()["state"] == "running" and time.time() < end:
        time.sleep(0.2)
    st = edit_batch.status()
    assert st["state"] == "done" and st["failed_count"] == 3 and "Out of GPU memory" in st["failed"][0]["error"]


# ------------------------------------------------------------------ setup helpers

def test_torch_index_for_driver():
    from app.services import edit_setup as s

    smi = lambda v: f"| NVIDIA-SMI 570 | Driver Version: 570 | CUDA Version: {v} |"
    assert s.torch_index_tag(smi("13.0")) == "cu130" and s.torch_index_tag(smi("12.8")) == "cu128"
    assert s.torch_index_tag(smi("12.4")) == "cu124" and s.torch_index_tag(smi("11.8")) == "cu118"
    with pytest.raises(s.SetupError):
        s.torch_index_tag(smi("10.2"))
    with pytest.raises(s.SetupError):
        s.torch_index_tag("no gpu here")


def test_verify_checklist_and_report(monkeypatch, tmp_path):
    from app.services import edit_setup as s

    monkeypatch.setattr(edit_env, "VENV", tmp_path / "nope")
    r = s.verify()
    assert not r["ok"] and any(c["id"] == "venv" and c["status"] == "fail" for c in r["checks"])
    rep = {"python": "3.11.9", "modules": {"torch": {"ok": True, "version": "2.5"}, "iopaint": {"ok": True, "version": "1.6.0"},
                                           "iopaint.model_manager": {"ok": True}, "iopaint.plugins": {"ok": False, "error": "x"},
                                           "rembg": {"ok": False}}, "cuda": {"available": False}}
    by = {c["id"]: c["status"] for c in s.report_checks(rep)}
    assert by["torch"] == "ok" and by["iopaint"] == "ok" and by["iopaint.plugins"] == "fail"
    assert by["gpu"] == "warn" and by["rembg"] == "warn" and by["python"] == "ok"


def test_python_version_choice_is_limited_to_what_iopaint_supports():
    from app.services import edit_setup as s

    assert s.PYTHON_VERSIONS == ("3.11", "3.10")
    assert s.IOPAINT_SPEC == "iopaint==" + catalog.IOPAINT_VERSION


# ------------------------------------------------------------------ routes

def test_routes(client, fake_engine, folder):
    r = client.get("/api/edit/ai/catalog").get_json()
    assert r["catalog"]["iopaint_version"] == catalog.IOPAINT_VERSION and "installed" in r["status"]
    assert client.get("/api/edit/ai/setup/status").get_json()["steps"]
    assert client.get("/api/edit/ai/job?id=nope").status_code == 404
    assert client.get("/api/edit/ai/result/" + "0" * 32).status_code == 404
    assert client.post("/api/edit/ai/run", json={"cmd": "nope"}).status_code == 400
    assert client.post("/api/edit/mask/adjust", json={"mask": _mask_url(), "op": "expand", "amount": 3}).status_code == 200
    assert client.post("/api/edit/mask/adjust", json={"mask": "###", "op": "expand"}).status_code == 400

    img = str(folder / "a.png")
    job = client.post("/api/edit/ai/run", json={"cmd": "inpaint", "image_path": img, "model": "lama",
                                                "mask": _mask_url()}).get_json()
    done = _wait(fake_engine, job["id"])
    token = done["result"]["token"]
    assert client.get(f"/api/edit/ai/result/{token}").mimetype == "image/png"
    assert client.post("/api/edit/ai/commit", json={"image_path": img, "token": token}).get_json()["ok"]
    assert client.post("/api/edit/ai/commit", json={"image_path": img, "token": token}).status_code == 404
    assert Image.open(img).getpixel((0, 0)) == (200, 10, 10)
    assert client.get("/api/edit/images", query_string={"folder": str(folder)}).get_json()["images"][0]["has_backup"] in (True, False)


# ------------------------------------------------------------------ the screen

def test_every_element_the_ai_script_uses_exists_in_the_page():
    js = (ROOT / "app/static/js/edit_ai.js").read_text(encoding="utf-8")
    html = (ROOT / "app/templates/index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'\$\("([\w-]+)"\)', js)) | set(re.findall(r'getElementById\("([\w-]+)"\)', js))
    missing = sorted(i for i in ids if f'id="{i}"' not in html)
    assert not missing, f"ids used by edit_ai.js but missing from index.html: {missing}"


def test_every_bridge_function_the_ai_script_calls_exists():
    js = (ROOT / "app/static/js/edit_ai.js").read_text(encoding="utf-8")
    bridge = (ROOT / "app/static/js/edit.js").read_text(encoding="utf-8")
    used = set(re.findall(r"\bB\.(\w+)\b", js)) - {"hooks"}
    block = bridge[bridge.index("const bridge = {"):]
    block = block[:block.index("\n  };")]
    defined = set(re.findall(r"^\s*(\w+)\s*[:,(]", block, re.M)) | set(re.findall(r"^\s*(\w+),", block, re.M))
    assert not (used - defined), f"bridge functions missing: {sorted(used - defined)}"


def test_every_api_call_the_ai_script_makes_is_defined():
    js = (ROOT / "app/static/js/edit_ai.js").read_text(encoding="utf-8")
    api = (ROOT / "app/static/js/api.js").read_text(encoding="utf-8")
    used = set(re.findall(r"\bApi\.(\w+)\b", js))
    assert not [u for u in used if not re.search(rf"\b{u}\s*:", api)]


def test_edit_ai_script_is_loaded_after_edit_js():
    html = (ROOT / "app/templates/index.html").read_text(encoding="utf-8")
    assert html.index("js/edit.js") < html.index("js/edit_ai.js")
    assert "editai-env-card" in html and "ai-batch-card" in html


def test_eraser_fills_the_window_and_shows_its_shortcuts():
    css = (ROOT / "app/static/css/pages.css").read_text(encoding="utf-8")
    assert re.search(r"\.eraser-modal\s*\{[^}]*width:\s*100vw[^}]*height:\s*100vh", css, re.S)
    js = (ROOT / "app/static/js/edit.js").read_text(encoding="utf-8")
    for key in ("wheel", "Keybinds"):
        assert key in js, key
    html = (ROOT / "app/templates/index.html").read_text(encoding="utf-8")
    for el in ("eraser-keybar", "eraser-zoomhud", "eraser-hud", "eraser-resizer", "eraser-rail"):
        assert el in html, el


# ------------------------------------------------------------------ keybinds

def _node(js_body):
    """Run `js_body` with keybinds.js loaded (browser globals stubbed); returns the printed JSON."""
    import json
    import shutil

    if not shutil.which("node"):
        pytest.skip("node is not installed")
    src = (ROOT / "app/static/js/keybinds.js").read_text(encoding="utf-8")
    prog = ("const store = {}; global.localStorage = {getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; }};\n"
            "global.Api = {appSettingsSaveOptions: async (o) => { global.__saved = o; }};\n"
            + src.replace("const Keybinds =", "global.Keybinds =") + "\n"
            "const ev = (key, m = {}) => ({key, ctrlKey: !!m.ctrl, metaKey: false, altKey: !!m.alt, shiftKey: !!m.shift});\n"
            + js_body)
    out = subprocess.run(["node", "-e", prog], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_keybinds_defaults_and_matching():
    r = _node('''console.log(JSON.stringify({
      run: Keybinds.actionFor(ev("Enter")), undo: Keybinds.actionFor(ev("z", {ctrl: true})),
      redo2: Keybinds.actionFor(ev("Z", {ctrl: true, shift: true})), shrink: Keybinds.actionFor(ev("G", {shift: true})),
      grow: Keybinds.actionFor(ev("g")), help: Keybinds.actionFor(ev("?", {shift: true})), plus: Keybinds.actionFor(ev("+", {shift: true})),
      none: Keybinds.actionFor(ev("q")), space: Keybinds.actionFor(ev(" ")), mod: Keybinds.actionFor(ev("Shift", {shift: true})),
      label: Keybinds.label("redo"), plusLabel: Keybinds.label("zoomIn")}));''')
    assert r["run"] == "run" and r["undo"] == "undo" and r["redo2"] == "redo" and r["shrink"] == "maskShrink"
    assert r["grow"] == "maskGrow" and r["help"] == "help" and r["plus"] == "zoomIn" and r["space"] == "pan"
    assert r["none"] is None and r["mod"] is None
    assert r["label"] == "Ctrl+Y / Ctrl+Shift+Z" and r["plusLabel"] == "+ / ="


def test_keybinds_rebind_conflict_reset_and_save():
    r = _node('''
      const before = Keybinds.actionFor(ev("q"));
      const stolen = Keybinds.assign("run", 0, "Q");           // Enter -> Q
      const r1 = [Keybinds.actionFor(ev("q")), Keybinds.actionFor(ev("Enter"))];
      const stolen2 = Keybinds.assign("clear", 0, "Q");         // Q already means "run": it moves
      const r2 = [Keybinds.actionFor(ev("q")), Keybinds.keysOf("run")];
      const savedLocal = JSON.parse(store["anima.keybinds.v1"]);
      Keybinds.resetAll();
      const r3 = [Keybinds.actionFor(ev("Enter")), Keybinds.actionFor(ev("q")), store["anima.keybinds.v1"]];
      Keybinds.hydrate({run: ["W", ""], bogus: ["X"], undo: "nope"});
      const r4 = [Keybinds.actionFor(ev("w")), Keybinds.actionFor(ev("x"))];
      Keybinds.assign("zoomIn", 0, "+");                         // the plus sign as a key
      console.log(JSON.stringify({before, stolen, r1, stolen2, r2, savedLocal, r3, r4, label: Keybinds.label("zoomIn"), hold: Keybinds.releasedHold(ev(" "))}));''')
    assert r["before"] is None and r["stolen"] == []
    assert r["r1"] == ["run", None]
    assert r["stolen2"] == ["Run the tool / accept the result"] and r["r2"][0] == "clear" and r["r2"][1] == ["", ""]
    assert r["savedLocal"]["clear"] == ["Q", ""]
    assert r["r3"] == ["run", None, "{}"]
    assert r["r4"] == ["run", "selectToggle"]                        # server copy applied; the junk entry changed nothing
    assert r["label"] == "+ / =" and r["hold"] == "pan"


def test_keybinds_are_validated_by_the_server(client):
    r = client.put("/api/appsettings/options", json={"keybinds": {
        "run": ["Q", ""], "x" * 50: ["A"], "undo": "bad", "redo": ["a" * 99, 5, "third"]}})
    kb = r.get_json()["keybinds"]
    assert kb["run"] == ["Q", ""] and "undo" not in kb and len(kb["redo"]) == 2 and len(kb["redo"][0]) == 32
    assert not any(len(k) > 40 for k in kb)
    assert client.get("/api/appsettings").get_json()["keybinds"] == kb
    assert client.put("/api/appsettings/options", json={"keybinds": "nonsense"}).get_json()["keybinds"] == {}


def test_editor_uses_the_keymap_everywhere():
    js = (ROOT / "app/static/js/edit.js").read_text(encoding="utf-8")
    assert "Keybinds.actionFor" in js and "Keybinds.releasedHold" in js
    assert 'k === "["' not in js and 'low === "g"' not in js          # no hard-coded letters left in the handler
    kb = (ROOT / "app/static/js/keybinds.js").read_text(encoding="utf-8")
    for action in re.findall(r'case "(\w+)":', js[js.index("function onKeyDown"):js.index("function onKeyUp")]):
        assert f'id: "{action}"' in kb, f"{action} is handled but not in the keybind list"
    for action in re.findall(r'id: "(\w+)", label', kb):
        assert f'"{action}"' in js, f"{action} is in the keybind list but never handled"
