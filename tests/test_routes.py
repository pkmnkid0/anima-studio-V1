import pytest

from app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_STUDIO_DATA", str(tmp_path))
    return create_app().test_client()


def test_index_loads(client):
    assert client.get("/").status_code == 200


def test_train_schema(client):
    r = client.get("/api/train/schema")
    assert r.status_code == 200 and r.is_json


def test_build_with_default_payload(client):
    from app import train_config as tc
    r = client.post("/api/train/build", json=tc.default_ui_payload())
    assert r.status_code == 200
    body = r.get_json()
    assert "warnings" in body


def test_build_warns_about_missing_captions(client, tmp_path):
    from app import train_config as tc
    img = tmp_path / "set"
    img.mkdir()
    (img / "a.png").write_bytes(b"x")
    (img / "a.png_0001.txt").write_text("1girl")
    ui = tc.default_ui_payload()
    ui["subsets"] = [{"name": "set", "image_dir": str(img), "num_repeats": 1, "caption_extension": ".txt"}]
    w = client.post("/api/train/build", json=ui).get_json()["warnings"]
    assert any("a.png needs a.txt" in x for x in w)


def test_tagger_models_listing_includes_pixai(client):
    models = client.get("/api/caption/tagger/models").get_json()["models"]
    names = {m["name"] for m in models}
    assert "pixai-tagger-v1.0" in names
    pix = next(m for m in models if m["name"] == "pixai-tagger-v1.0")
    assert pix["needs_deps"] is True


def test_appsettings_options_roundtrip(client, tmp_path, monkeypatch):
    from app.services import settings_store
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", tmp_path / "settings.json")   # never touch real settings
    r = client.get("/api/appsettings").get_json()
    assert r["auto_load"] is True and r["startup"] == "last"          # on by default
    client.post("/api/appsettings/presets", json={"name": "mine", "values": {"monitor-logging-dir": "D:/logs"}})
    r = client.put("/api/appsettings/options", json={"auto_load": False, "startup": "preset:mine"}).get_json()
    assert r["auto_load"] is False and r["startup"] == "preset:mine"
    r = client.put("/api/appsettings/options", json={"startup": "preset:missing"}).get_json()
    assert r["startup"] == "preset:mine"                               # unknown preset is ignored
    assert client.get("/api/appsettings").get_json()["auto_load"] is False


def test_appsettings_log_dir(client, tmp_path, monkeypatch):
    from app.services import settings_store
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", tmp_path / "settings.json")
    target = tmp_path / "logs-here"
    r = client.put("/api/appsettings/options", json={"log_dir": str(target)})
    assert r.status_code == 200 and r.get_json()["log_dir"] == str(target)
    assert target.is_dir()
    assert client.get("/api/appsettings").get_json()["log_dir"] == str(target)
    assert client.put("/api/appsettings/options", json={"log_dir": ""}).get_json()["log_dir"] == ""


def test_caption_tools_routes(client, tmp_path, monkeypatch):
    from PIL import Image
    from app.services import settings_store
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", tmp_path / "settings.json")
    d = tmp_path / "ds"
    d.mkdir()
    for i, t in enumerate(["x, shirt, white shirt", "x, shirt, white shirt", "x, smile, smile"]):
        Image.new("RGB", (8, 8), (i * 40, 1, 1)).save(d / f"{i}.png")
        (d / f"{i}.txt").write_text(t)
    folder = str(d)
    scan = client.post("/api/caption/prune/scan", json={"folder": folder}).get_json()
    pairs = scan["pairs"]
    assert pairs and pairs[0]["general"] == "shirt" and scan["groups"][0]["images"] == 2
    assert client.post("/api/caption/prune/apply", json={"folder": folder, "pairs": []}).status_code == 400
    r = client.post("/api/caption/prune/apply", json={"folder": folder, "pairs": [pairs[0]]}).get_json()
    assert r["changed"] == 2
    st = client.get("/api/caption/undo/status", query_string={"folder": folder}).get_json()
    assert st["depth"] == 1 and st["label"] == "Prune redundant tags"
    assert client.post("/api/caption/undo", json={"folder": folder}).get_json()["restored"] == 2
    assert client.post("/api/caption/bulk/dedupe", json={"folder": folder}).get_json()["changed"] == 1
    assert client.put("/api/caption/blacklist", json={"tags": ["Smile"], "apply_after_tagging": False}).get_json()["tags"] == ["smile"]
    assert client.post("/api/caption/bulk/apply_blacklist", json={"folder": folder}).get_json()["changed"] == 1


def test_api_js_has_no_duplicate_keys():
    """A repeated key in the Api object silently replaces the earlier one (this broke the Edit tab once)."""
    import collections
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "static" / "js" / "api.js").read_text(encoding="utf-8")
    keys = re.findall(r"^\s{4}(\w+):", src, re.M)
    assert [k for k, n in collections.Counter(keys).items() if n > 1] == []


def test_cross_site_requests_are_refused(client, tmp_path, monkeypatch):
    from app.services import settings_store
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", tmp_path / "settings.json")
    # a web page on another origin trying to drive the local API
    r = client.post("/api/caption/bulk/dedupe", data="{}", headers={"Origin": "https://evil.example", "Content-Type": "text/plain"})
    assert r.status_code == 403
    assert client.get("/api/health", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/health", headers={"Sec-Fetch-Site": "same-site"}).status_code == 403
    # DNS rebinding: a foreign host name that points at 127.0.0.1
    assert client.get("/api/health", headers={"Host": "evil.example:7864"}).status_code == 403
    # legitimate: the app's own page, curl-style calls, and the browser's same-origin calls
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    ok = client.put("/api/appsettings/options", json={"auto_load": True},
                     headers={"Origin": "http://localhost", "Host": "localhost"})
    assert ok.status_code == 200
    assert "access-control-allow-origin" not in client.get("/api/health", headers={"Origin": "https://evil.example"}).headers
    assert client.get("/api/health").headers["X-Content-Type-Options"] == "nosniff"


def test_lora_slim_routes(client, tmp_path):
    from tests.test_services import _make_lora
    src = tmp_path / "l.safetensors"
    _make_lora(src, rank=16)
    r = client.get("/api/train/slim/analyze", query_string={"path": str(src)})
    assert r.status_code == 200 and r.get_json()["options"]
    r = client.post("/api/train/slim", json={"path": str(src), "mode": "rank", "value": 4})
    j = r.get_json()
    assert r.status_code == 200 and j["rank_after"] == 4 and j["out_path"] != str(src)
    assert src.exists()
    assert client.post("/api/train/slim", json={"path": str(tmp_path / "no.safetensors")}).status_code == 400


def test_lora_metadata_routes(client, tmp_path):
    from tests.test_services import _make_lora
    src = tmp_path / "m.safetensors"
    _make_lora(src)
    r = client.get("/api/train/metadata", query_string={"path": str(src)})
    assert r.status_code == 200 and r.get_json()["key_count"] == 3
    assert client.get("/api/train/metadata").status_code == 400

    r = client.post("/api/train/metadata/strip", json={"path": str(src), "keep": ["ss_network_dim"]})
    j = r.get_json()
    assert r.status_code == 200 and j["verified"] is True and j["out_path"] != str(src)
    assert j["removed"] == ["ss_base_model_version", "sshs_model_hash"]
    assert src.exists()                                    # the original is never modified
    assert client.post("/api/train/metadata/strip", json={"path": str(tmp_path / "no.safetensors")}).status_code == 400


def test_gpu_and_backend_verify_routes(client, monkeypatch):
    from app.services import backend_client, gpu
    monkeypatch.setattr(gpu, "_run", lambda args: "0, NVIDIA GeForce RTX 4090, 10, 1024, 24564, 45, 100.0, 450.0, 30\n")
    gpu._cache.update(at=0.0, data=None)
    j = client.get("/api/monitor/gpu").get_json()
    assert j["available"] is True and j["gpus"][0]["name"].startswith("NVIDIA")
    assert j["gpus"][0]["mem_pct"] == 4.2
    gpu._cache.update(at=0.0, data=None)

    monkeypatch.setattr(backend_client, "ping", lambda url: {"reachable": False})
    r = client.get("/api/train/backend/verify")
    assert r.status_code == 200
    j = r.get_json()
    assert isinstance(j["ok"], bool) and j["checks"]
    assert {c["status"] for c in j["checks"]} <= {"ok", "warn", "fail"}
    assert all(c["label"] for c in j["checks"])


def test_nl_caption_routes(client, tmp_path, monkeypatch):
    from app.services import captioner
    j = client.get("/api/caption/nl/models").get_json()
    assert {m["name"] for m in j["models"]} == set(captioner.MODELS)
    assert "florence-2-large" not in {m["name"] for m in j["models"]}
    assert {m["tier"] for m in j["models"]} == {"low", "mid", "high"}
    assert j["default"] == "joycaption-beta" and j["vram_profiles"][0] == "auto"
    assert all(set(m["fits"]) == {"8", "10", "12", "16", "24"} and m["vram"]["4bit"] <= 8 for m in j["models"])
    assert all(m["downloaded"] is False for m in j["models"])       # nothing downloaded in a fresh checkout
    assert all(m["note"] and m["repo"] for m in j["models"])

    r = client.post("/api/caption/nl/preview", json={"caption": "1girl, blue_hair, smile, best quality"})
    assert r.status_code == 200
    assert r.get_json()["after"] == "A girl with blue hair, smiling."
    r = client.post("/api/caption/nl/preview",
                    json={"caption": "1girl, blue_hair, smile", "write": "keep", "trigger_word": "mychar"})
    assert r.status_code == 200
    assert r.get_json()["after"] == "mychar, 1girl, blue_hair, smile, a girl with blue hair, smiling."
    assert client.post("/api/caption/nl/preview", json={}).status_code == 400
    assert client.get("/api/caption/nl/run_status").get_json()["status"] in ("idle", "done")

    assert client.post("/api/caption/nl/run", json={"folder": str(tmp_path / "nope")}).status_code == 400
    assert client.post("/api/caption/nl/run",
                       json={"folder": str(tmp_path), "source": "vlm", "model": "joycaption-beta"}).status_code == 400

    monkeypatch.setattr(captioner, "deps_status", lambda force=False, model=None: {
        "checked": True, "ok": True, "missing": [], "label": "test env", "python": "python",
        "torch": {"version": "2.0", "cuda": True}, "error": ""})
    assert client.get("/api/caption/nl/deps").get_json()["ok"] is True


def test_img_proxy_serves_the_second_request_from_cache(client, tmp_path, monkeypatch):
    from app.services import booru_engines as eng, img_cache
    monkeypatch.setattr(img_cache, "CACHE_DIR", tmp_path / "imgcache")
    calls = []

    class Resp:
        status_code = 200
        headers = {"Content-Type": "image/jpeg"}

        def iter_content(self, chunk_size=0):
            yield b"\xff\xd8\xff\xe0jpeg"

        def close(self):
            pass

    def fake_open(*a, **k):
        calls.append(a)
        return Resp()

    monkeypatch.setattr(eng, "open_image", fake_open)
    url = "https://cdn.donmai.us/preview/ab/cd/cached.jpg"
    r1 = client.get("/api/explore/img", query_string={"url": url})
    assert r1.status_code == 200 and r1.data == b"\xff\xd8\xff\xe0jpeg"
    assert r1.headers["X-Cache"] == "MISS"
    r2 = client.get("/api/explore/img", query_string={"url": url})
    assert r2.status_code == 200 and r2.data == b"\xff\xd8\xff\xe0jpeg"
    assert r2.headers["X-Cache"] == "HIT"
    assert len(calls) == 1                     # the site was hit exactly once
