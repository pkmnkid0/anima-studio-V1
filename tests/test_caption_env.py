import app.services.caption_env as ce
from app.services import captioner


def test_qwen35_models_registered_with_own_env():
    names = [n for n, m in captioner.MODELS.items() if m.get("family") == "qwen35"]
    assert len(names) == 3
    for n in names:
        m = captioner.MODELS[n]
        assert m["env"] == "caption5" and m["repo"].startswith("prithivMLmods/Gliese-Qwen3.5")
        assert "visual" in m["skip_modules"]
        assert captioner.model_summary(n)["env"] == "caption5"


def test_interpreter_routes_per_model(monkeypatch):
    py, label = captioner.interpreter("qwen3.5-9b-caption")
    assert "caption_backend" in py and "Qwen3.5" in label
    # other models never use it
    assert "caption_backend" not in captioner.interpreter("joycaption-beta")[0]


def test_not_ready_env_blocks_download_and_reports_missing(monkeypatch):
    monkeypatch.setattr(ce, "is_ready", lambda: False)
    d = captioner.deps_status(model="qwen3.5-4b-caption")
    assert not d["ok"] and "environment" in d["missing"][0]
    try:
        captioner.download_async("qwen3.5-4b-caption")
        assert False
    except captioner.CaptionerError as exc:
        assert "environment first" in str(exc)


def test_verify_report_flags_old_transformers():
    rep = {"torch": {"version": "2.9", "cuda": True}, "transformers": "4.57.1", "bitsandbytes": "0.46.1",
           "multimodal_class": True}
    checks = {c["id"]: c for c in ce.verify_report(rep)}
    assert checks["transformers"]["status"] == "fail" and "5.3" in checks["transformers"]["detail"]
    rep["transformers"] = "5.3.0"
    assert all(c["status"] != "fail" for c in ce.verify_report(rep))


def test_worker_helpers():
    from native_backend import caption_worker as w
    assert w._strip_thinking("<think>hmm</think>A girl.") == "A girl."
    assert w._strip_thinking("A girl.<think>never closed") == "A girl."
    from PIL import Image
    big = Image.new("RGB", (4000, 3000))
    small = w._limit_pixels(big, 1_000_000)
    assert small.size[0] * small.size[1] <= 1_000_000 and abs(small.size[0] / small.size[1] - 4 / 3) < 0.01
    assert w._limit_pixels(Image.new("RGB", (100, 100)), 1_000_000).size == (100, 100)


def test_routes_exist():
    from app import create_app
    c = create_app().test_client()
    assert c.get("/api/caption/nl/env/status").get_json()["installed"] is False
    models = c.get("/api/caption/nl/models").get_json()["models"]
    assert any(m["env"] == "caption5" and not m["env_ready"] for m in models)
    r = c.get("/api/caption/nl/deps?model=qwen3.5-9b-caption").get_json()
    assert r["ok"] is False and r["env"] == "caption5"


def test_worker_never_uses_text_only_class_for_qwen35():
    import inspect
    from native_backend import caption_worker as w
    src = inspect.getsource(w._load)
    assert '("AutoModelForImageTextToText", "AutoModelForMultimodalLM") if family == "qwen35"' in src
