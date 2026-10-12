import io

from PIL import Image

from app.services import booru_engines, tagger


def test_image_variants_include_original_and_dedupe():
    v = booru_engines._image_variants("https://tbib.org//images/12/abc.jpg")
    assert v[0].startswith("https://tbib.org") and len(v) == len(set(v))


def test_pixai_model_is_registered():
    assert "pixai-tagger-v1.0" in tagger.available_models()
    assert tagger.backend_for("pixai-tagger-v1.0") == "pixai"
    assert tagger.backend_for("wd-vit-tagger-v3") == "onnx"


def test_pixai_threshold_defaults_and_overrides(monkeypatch):
    import contextlib
    seen = []

    class Torch:
        def no_grad(self):
            return contextlib.nullcontext()

    def pipe(img, threshold=None):
        seen.append(threshold)
        return {"results": {"general": {"1girl": .9, "solo": .8}, "character": {"leina": .9},
                            "style": {"flat color": .5}, "rating": {"rating:g": .1, "rating:e": .8}}}

    name = "pixai-tagger-v1.0"
    tagger._sessions[name] = {"kind": "pixai", "pipe": pipe, "torch": Torch()}
    tagger._tag_lists[name] = {}
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    r = tagger.tag_image_bytes(buf.getvalue(), name)
    assert r["character"] == ["leina"] and "1girl" in r["general"] and r["rating"] == "rating:e"
    assert seen[-1] is None                       # WD sliders untouched -> PixAI's own defaults
    tagger.tag_image_bytes(buf.getvalue(), name, general_threshold=0.2)
    assert seen[-1] == {"general": 0.2}
    tagger._sessions.pop(name); tagger._tag_lists.pop(name)


def test_smart_dataset_filters_are_or(monkeypatch):
    """Ticked filter tags are alternatives: one query per filter, results merged."""
    from app.services import booru
    queries = []

    def fake_search(source, tags, rating="safe", limit=40, page=1):
        queries.append(tags)
        if page > 1:
            return []
        t = tags.split()[-1]
        return [{"source": source, "id": f"{t}-{i}", "file_url": "http://x/y.png", "width": 1024,
                 "height": 1024 + i, "tags": {"general": [t, f"u{t}{i}"]}} for i in range(3)]

    monkeypatch.setattr(booru, "search", fake_search)
    out = booru.smart_dataset("danbooru", "char", target=100, pool=100, min_resolution=0,
                              any_tags=["outfit_a", "outfit_b"])
    assert queries == ["char outfit_a", "char outfit_b"]
    assert out["candidate_count"] == 6
