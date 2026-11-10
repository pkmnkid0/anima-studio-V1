import io
from pathlib import Path

import requests
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


# ---------------------------------------------------------------- tag tools

def _cap_folder(tmp_path, captions):
    from PIL import Image
    for i, text in enumerate(captions):
        Image.new("RGB", (8, 8), (i * 10 % 255, 0, 0)).save(tmp_path / f"i{i}.png")
        if text is not None:
            (tmp_path / f"i{i}.txt").write_text(text, encoding="utf-8")
    return str(tmp_path)


def test_tag_tools_undo_prune_sort_dedupe(tmp_path):
    from app.services import tag_tools as tt
    f = _cap_folder(tmp_path, [
        "mia, shirt, white shirt, smile, smile, red hair, 1girl",
        "mia, shirt, long hair, very long hair",
        "mia, shirt",
    ])
    res = tt.scan_prune(f)
    got = {(p["general"], p["specific"]): p["images"] for p in res["pairs"]}
    g = {x["general"]: x for x in res["groups"]}
    assert g["shirt"]["images"] == 1 and g["shirt"]["general_total"] == 3 and g["shirt"]["specifics"][0]["tag"] == "white shirt"
    assert got[("shirt", "white shirt")] == 1 and got[("long hair", "very long hair")] == 1

    tt.snapshot(f, "prune")
    assert tt.apply_prune(f, [{"general": "shirt", "specific": "white shirt"}]) == 1
    assert "shirt," not in (tmp_path / "i0.txt").read_text().replace("white shirt", "")
    assert (tmp_path / "i2.txt").read_text() == "mia, shirt"          # no specific tag there: untouched
    assert tt.undo_status(f)["label"] == "prune"
    assert tt.undo(f)["restored"] == 1
    assert "shirt, white shirt" in (tmp_path / "i0.txt").read_text()

    assert tt.dedupe_captions(f) == 1
    assert (tmp_path / "i0.txt").read_text().count("smile") == 1

    tt.auto_sort(f, keep_first=1)
    assert (tmp_path / "i0.txt").read_text().split(", ")[:2] == ["mia", "1girl"]   # trigger kept, subject next


def test_tag_tools_blacklist_cooccurrence_colour_guard(tmp_path, monkeypatch):
    from app.services import settings_store, tag_tools as tt
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", tmp_path / "settings.json")
    sub = tmp_path / "ds"
    sub.mkdir()
    f = _cap_folder(sub, ["a, watermark, blue eyes, red scarf", "a, red scarf, green eyes, hat", "a, red scarf, blue eyes",
                          "a, red scarf, hat"])
    tt.set_blacklist(["Watermark"], True)
    assert tt.apply_blacklist(f) == 1

    rel = tt.cooccurrence(f, "red scarf")
    assert rel["images"] == 4 and any(r["tag"] == "blue eyes" for r in rel["related"])
    assert any({p["a"], p["b"]} == {"a", "red scarf"} for p in tt.cooccurrence(f, min_count=3)["pairs"])

    # colour guard: keep the best eye colour, drop low-confidence blue skin
    scored = [("blue_eyes", 0.6), ("red_eyes", 0.9), ("blue_skin", 0.5), ("smile", 0.8)]
    assert [t for t, _ in tt.colour_guard(scored)] == ["red_eyes", "smile"]
    assert "heterochromia" and [t for t, _ in tt.colour_guard(scored + [("heterochromia", 0.5)])].count("blue_eyes") == 1


def test_tag_tools_edit_images(tmp_path):
    from app.services import tag_tools as tt
    f = _cap_folder(tmp_path, ["a, b", "a"])
    paths = [str(tmp_path / "i0.png"), str(tmp_path / "i1.png")]
    assert tt.edit_images(paths, ["c"], ["b"]) == 2
    assert (tmp_path / "i0.txt").read_text() == "a, c" and (tmp_path / "i1.txt").read_text() == "a, c"
    assert tt.edit_images(paths, ["c"], []) == 0


def test_timm_preprocess_is_bgr():
    """The WD timm taggers expect BGR; a pure red image must land in the LAST channel."""
    import pytest
    pytest.importorskip("numpy")
    from PIL import Image
    import io
    from app.services import tagger
    buf = io.BytesIO()
    Image.new("RGB", (448, 448), (255, 0, 0)).save(buf, "PNG")
    arr = tagger._preprocess_timm(buf.getvalue(), 448)
    assert arr.shape == (1, 3, 448, 448)
    assert arr[0, 2].mean() > 0.99 and arr[0, 0].mean() < -0.99


def test_get_post_adds_total_post_counts(monkeypatch):
    from app.services import booru_engines as eng
    raw = {"id": 7, "file_url": "http://x/a.png", "large_file_url": "http://x/a.png", "preview_file_url": "http://x/t.png",
           "tag_string_general": "1girl", "tag_string_character": "hanako_(test)", "tag_string_copyright": "test_series",
           "tag_string_artist": "", "tag_string_meta": "", "rating": "g", "image_width": 10, "image_height": 10,
           "file_ext": "png", "score": 1, "md5": "abc", "created_at": "2020-01-01T00:00:00.000+00:00"}
    monkeypatch.setattr(eng, "_get_json", lambda *a, **k: raw)
    monkeypatch.setattr(eng, "categorize", lambda source, names, **k: {
        "hanako_(test)": {"category": "character", "post_count": 12345},
        "test_series": {"category": "copyright", "post_count": 99}})
    post = eng.get_post("danbooru", 7)
    assert post["tag_counts"] == {"hanako_(test)": 12345, "test_series": 99}


# ------------------------------------------------------------------ LoRA slimmer

def _make_lora(path, rank=16, bf16=False, alpha=8.0, seed=0):
    """A synthetic LoRA with a decaying spectrum: linear layers, one conv layer, a text-encoder layer
    and a LoHa-style layer that must be copied untouched."""
    import numpy as np
    from app.services import lora_slim as ls
    rng = np.random.default_rng(seed)

    def lowrank(out, inn, r):
        decay = np.exp(-np.arange(r) / 3.0)
        return (rng.standard_normal((out, r)) * decay).astype("f4") * 0.2, rng.standard_normal((r, inn)).astype("f4") * 0.2

    tensors, truth = {}, {}
    def add(name, arr, dt="F32"):
        raw = ls._from_f32(arr, "BF16") if dt == "BF16" else arr.astype("<f4").tobytes()
        tensors[name] = {"dtype": dt, "shape": list(arr.shape), "raw": raw}
    dt = "BF16" if bf16 else "F32"
    for pre, out, inn in (("lora_unet_a", 48, 40), ("lora_unet_b", 32, 64), ("lora_te_c", 24, 24)):
        up, down = lowrank(out, inn, rank)
        add(pre + ".lora_up.weight", up, dt); add(pre + ".lora_down.weight", down, dt)
        add(pre + ".alpha", np.array(alpha, "f4"), "F32")
        truth[pre] = ls._to_f32(tensors[pre + ".lora_up.weight"]) @ ls._to_f32(tensors[pre + ".lora_down.weight"]) * (alpha / rank)
    up, down = lowrank(20, 3 * 3 * 6, rank)                      # conv: down (r, 6, 3, 3), up (20, r, 1, 1)
    add("lora_unet_conv.lora_up.weight", up.reshape(20, rank, 1, 1), dt)
    add("lora_unet_conv.lora_down.weight", down.reshape(rank, 6, 3, 3), dt)
    add("lora_unet_conv.alpha", np.array(alpha, "f4"), "F32")
    truth["lora_unet_conv"] = (up @ down) * (alpha / rank)
    add("lora_unet_d.hada_w1_a", rng.standard_normal((8, 4)).astype("f4"))       # not plain LoRA: passthrough
    ls._write(str(path), tensors, {"ss_network_dim": str(rank), "ss_base_model_version": "anima", "sshs_model_hash": "abc"})
    return truth


def _delta(path, prefix):
    from app.services import lora_slim as ls
    t, _ = ls._read(str(path))
    up, down = ls._to_f32(t[prefix + ".lora_up.weight"]), ls._to_f32(t[prefix + ".lora_down.weight"])
    a = float(ls._to_f32(t[prefix + ".alpha"])) if prefix + ".alpha" in t else 1.0
    return up.reshape(up.shape[0], up.shape[1]) @ down.reshape(down.shape[0], -1) * (a / down.shape[0])


def test_lora_slim_is_exact_at_full_rank_and_optimal_when_cut(tmp_path):
    import numpy as np
    from app.services import lora_slim as ls
    src = tmp_path / "x.safetensors"
    truth = _make_lora(src, rank=16)

    # Full rank, only changing number format: the change must be unchanged
    res = ls.slim(str(src), "rank", 16, "fp16", str(tmp_path / "full.safetensors"))
    assert res["layers_changed"] == 0 and res["kept_pct"] == 100.0 and res["size_after_mb"] <= res["size_before_mb"]

    # Cut to rank 4: error must equal the best possible (dropped singular values), nothing worse
    res = ls.slim(str(src), "rank", 4, "keep", str(tmp_path / "r4.safetensors"))
    assert res["rank_after"] == 4 and res["layers_changed"] == 4 and res["size_after_mb"] < res["size_before_mb"]
    for pre, full in truth.items():
        got = _delta(tmp_path / "r4.safetensors", pre)
        s = np.linalg.svd(full, compute_uv=False)
        best = np.sqrt((s[4:] ** 2).sum() / (s ** 2).sum())
        err = np.linalg.norm(got - full) / np.linalg.norm(full)
        assert abs(err - best) < 1e-3, (pre, err, best)

    # the file is valid for the real safetensors library, metadata updated, passthrough layer intact
    from safetensors import safe_open
    with safe_open(str(tmp_path / "r4.safetensors"), "np") as f:
        meta = f.metadata()
        assert "lora_unet_d.hada_w1_a" in f.keys() and f.get_tensor("lora_unet_conv.lora_down.weight").shape == (4, 6, 3, 3)
        assert float(f.get_tensor("lora_unet_a.alpha")) == 4.0
    assert meta["ss_network_dim"] == "4" and "sshs_model_hash" not in meta and "slimmed_from" in meta
    assert src.exists()                                  # original untouched


def test_lora_slim_energy_mode_bf16_and_analyze(tmp_path):
    from app.services import lora_slim as ls
    src = tmp_path / "b.safetensors"
    _make_lora(src, rank=16, bf16=True)
    info = ls.analyze(str(src), "fp16")
    assert info["layers"] == 4 and info["max_rank"] == 16 and info["dtype"] == "bf16" and info["skipped"] == 1
    rec = next(o for o in info["options"] if o["id"] == info["recommended"])
    assert rec["kept_pct"] >= 99 and rec["size_mb"] <= info["size_mb"]
    ranks = [o for o in info["options"] if o["mode"] == "rank" and o["id"] != "same"]
    assert all(a["size_mb"] <= b["size_mb"] for a, b in zip(ranks, ranks[1:]))        # bigger rank = bigger file
    res = ls.slim(str(src), "energy", 99.0, "bf16", str(tmp_path / "e.safetensors"))
    assert res["rank_after"] < 16 and res["kept_pct"] >= 99.0 - 1e-6
    res2 = ls.slim(str(src), "energy", 99.0, "bf16", str(tmp_path / "e.safetensors"))   # never overwrites
    assert res2["out_path"].endswith("e_2.safetensors")


def test_lora_slim_rejects_bad_input(tmp_path):
    import pytest
    from app.services import lora_slim as ls
    with pytest.raises(ls.SlimError):
        ls.analyze(str(tmp_path / "missing.safetensors"))
    src = tmp_path / "c.safetensors"
    _make_lora(src)
    with pytest.raises(ls.SlimError):
        ls.slim(str(src), "rank", 4, "keep", str(src))                      # would overwrite the original
    with pytest.raises(ls.SlimError):
        ls.slim(str(src), "energy", 150)


def test_lora_metadata_read_and_strip(tmp_path):
    from safetensors import safe_open
    from app.services import lora_slim as ls
    src = tmp_path / "m.safetensors"
    _make_lora(src)          # writes ss_network_dim, ss_base_model_version and sshs_model_hash

    info = ls.metadata(str(src))
    assert info["has_metadata"] and info["key_count"] == 3 and info["tensors"] == 13
    assert {k["key"] for k in info["keys"]} == {"ss_network_dim", "ss_base_model_version", "sshs_model_hash"}

    # keep one key, drop the rest
    kept = tmp_path / "keep.safetensors"
    res = ls.strip_metadata(str(src), ["ss_network_dim"], str(kept))
    assert res["verified"] is True
    assert res["removed"] == ["ss_base_model_version", "sshs_model_hash"] and res["kept"] == ["ss_network_dim"]
    with safe_open(str(kept), "np") as f:
        assert f.metadata() == {"ss_network_dim": "16"}

    # keep nothing: no metadata block at all, and every tensor byte for byte identical
    bare = tmp_path / "bare.safetensors"
    res = ls.strip_metadata(str(src), None, str(bare))
    assert res["has_metadata_after"] is False and res["keys_after"] == 0
    with safe_open(str(bare), "np") as f:
        assert not f.metadata() and len(f.keys()) == 13
    assert ls._digest(ls._read(str(src))[0]) == ls._digest(ls._read(str(bare))[0])

    # dry run writes nothing; the original is never touched
    planned = tmp_path / "nope.safetensors"
    res = ls.strip_metadata(str(src), None, str(planned), dry_run=True)
    assert res["dry_run"] is True and not planned.exists() and src.exists()
    assert ls.metadata(str(src))["key_count"] == 3


def test_lora_metadata_rejects_bad_input(tmp_path):
    import pytest
    from app.services import lora_slim as ls
    with pytest.raises(ls.SlimError):
        ls.metadata(str(tmp_path / "missing.safetensors"))
    src = tmp_path / "nm.safetensors"
    _make_lora(src)
    out = tmp_path / "already.safetensors"
    ls.strip_metadata(str(src), None, str(out))
    with pytest.raises(ls.SlimError):
        ls.strip_metadata(str(out), None)                      # nothing left to remove
    with pytest.raises(ls.SlimError):
        ls.strip_metadata(str(src), None, str(src))            # would overwrite the original


def test_tag_region_crops_and_ranks(tmp_path, monkeypatch):
    from PIL import Image
    from app.services import tagger
    img = tmp_path / "a.png"
    Image.new("RGB", (200, 100), "white").save(img)
    sizes = []

    def fake(data, model_name=None, **kw):
        import io
        sizes.append(Image.open(io.BytesIO(data)).size)
        if len(sizes) == 1:   # the crop
            return {"general_scored": [("pelvic_curtain", 0.8), ("skirt", 0.4)]}
        return {"general_scored": [("skirt", 0.5), ("pelvic_curtain", 0.1)]}
    monkeypatch.setattr(tagger, "tag_image_bytes", fake)
    out = tagger.tag_region(str(img), {"x": 0.5, "y": 0, "w": 0.5, "h": 1}, "m")
    assert sizes[0] == (100, 100) and sizes[1] == (200, 100)
    assert out[0]["tag"] == "pelvic curtain" and out[0]["lift"] == 0.7 and out[0]["whole"] == 0.1
    import pytest
    with pytest.raises(tagger.TaggerError):
        tagger.tag_region(str(img), {"x": 0, "y": 0, "w": 0.001, "h": 0.001}, "m")


def test_tag_region_vocabulary_search_finds_low_scoring_tags(tmp_path, monkeypatch):
    from PIL import Image
    from app.services import tagger
    tagger._REGION_CACHE.clear()
    img = tmp_path / "b.png"
    Image.new("RGB", (100, 100), "white").save(img)
    calls = []

    def fake(data, model_name=None, **kw):
        calls.append(kw)
        return {"general_scored": [("skirt", 0.9), ("crotch_plate", 0.004), ("metal", 0.2)]}
    monkeypatch.setattr(tagger, "tag_image_bytes", fake)
    box = {"x": 0, "y": 0, "w": 1, "h": 1}
    assert [t["tag"] for t in tagger.tag_region(str(img), box, "m", 1)] == ["skirt"]
    found = tagger.tag_region(str(img), box, "m", 40, "crotch plate")
    assert [t["tag"] for t in found] == ["crotch plate"] and found[0]["score"] == 0.004
    assert len(calls) == 2          # second call reused the cached scores


# ---------------------------------------------------- natural-language captions

def test_caption_from_tags_phrases_and_orders():
    from app.services import captioner
    text = captioner.caption_from_tags(
        ["1girl", "long_hair", "blue hair", "twintails", "blue eyes", "school uniform", "smile",
         "looking at viewer", "standing", "classroom", "best quality", "absurdres"])
    assert text.startswith("A girl with long blue hair in twintails")
    assert "with blue eyes" in text and "wearing a school uniform" in text
    assert "smiling and looking at the viewer" in text and text.endswith("in a classroom.")
    assert "best quality" not in text and "absurdres" not in text      # library metadata, not the picture

    # tags are normalised first, and a caption with nothing in it stays empty
    assert captioner.caption_from_tags(["_long_hair_"]) == "A character with long hair."
    assert captioner.caption_from_tags([]) == ""
    assert captioner.caption_from_caption_text("1girl, blue hair, smile") == "A girl with blue hair, smiling."


def test_caption_trigger_word_and_kept_library_tags():
    from app.services import captioner
    # the trigger leads, so the sentence after it reads as a continuation
    assert captioner.caption_from_tags(["1girl", "blue hair"], trigger_word="mychar") == "mychar, a girl with blue hair."
    # library tags can be kept when someone wants the ranking in the caption
    kept = captioner.caption_from_tags(["1girl", "best quality"], drop_quality=False)
    assert "best quality" in kept


def test_caption_from_tags_framing_and_backgrounds():
    from app.services import captioner
    text = captioner.caption_from_tags(
        ["1boy", "full body", "grey background", "gradient background", "simple background"])
    assert "shown full body" in text
    assert text.count("background") == 1 and "grey gradient background" in text


def test_caption_batch_from_existing_tags(tmp_path):
    import time as _time
    from PIL import Image
    from app.services import captioner, captions
    folder = tmp_path / "ds"
    folder.mkdir()
    for i, cap in enumerate(["1girl, blue hair, smile", "1boy, short hair", ""]):
        Image.new("RGB", (8, 8), "white").save(folder / f"i{i}.png")
        if cap:
            captions.write_caption(str(folder / f"i{i}.png"), cap)

    status = captioner.run({"folder": str(folder), "source": "tags", "mode": "all"})
    assert status["status"] == "running"
    for _ in range(200):                                   # the batch runs on a thread
        status = captioner.run_status()
        if status["status"] != "running":
            break
        _time.sleep(0.05)
    assert status["status"] == "done", status
    assert status["captioned"] == 2 and status["skipped"] == 1
    assert captions.read_caption(str(folder / "i0.png")) == "A girl with blue hair, smiling."
    assert captions.read_caption(str(folder / "i1.png")) == "A boy with short hair."
    assert captions.read_caption(str(folder / "i2.png")) == ""      # nothing to rephrase, left alone

    import pytest
    with pytest.raises(captioner.CaptionerError):
        captioner.run({"folder": str(tmp_path / "missing"), "source": "tags"})
    with pytest.raises(captioner.CaptionerError):
        captioner.run({"folder": str(folder), "source": "vlm", "model": "not-a-model"})
    with pytest.raises(captioner.CaptionerError):
        captioner.run({"folder": str(folder), "source": "vlm", "model": captioner.DEFAULT_MODEL})  # not downloaded


def test_caption_batch_keeps_the_tags(tmp_path):
    """`keep` writes the trigger first, the tags that were there, then the sentence."""
    import time as _time
    from PIL import Image
    from app.services import captioner, captions
    folder = tmp_path / "ds-keep"
    folder.mkdir()
    for i, cap in enumerate(["1girl, blue hair, smile", "1boy, short hair"]):
        Image.new("RGB", (8, 8), "white").save(folder / f"i{i}.png")
        captions.write_caption(str(folder / f"i{i}.png"), cap)

    status = captioner.run({"folder": str(folder), "source": "tags", "mode": "all",
                            "write": "keep", "trigger_word": "mychar"})
    assert status["status"] == "running"
    for _ in range(200):                                   # the batch runs on a thread
        status = captioner.run_status()
        if status["status"] != "running":
            break
        _time.sleep(0.05)
    assert status["status"] == "done", status
    assert captions.read_caption(str(folder / "i0.png")) == \
        "mychar, 1girl, blue hair, smile, a girl with blue hair, smiling."
    assert captions.read_caption(str(folder / "i1.png")) == \
        "mychar, 1boy, short hair, a boy with short hair."

    import pytest
    # a trigger already in the tags is not written twice
    assert captioner.preview_caption("mychar, 1boy", trigger_word="mychar", write="keep") == \
        "mychar, 1boy, a boy."
    with pytest.raises(captioner.CaptionerError):
        captioner.run({"folder": str(folder), "source": "tags", "write": "nonsense"})


def test_caption_batch_tags_source_ignores_the_without_captions_choice(tmp_path):
    """An image without a caption has no tags to rephrase, so the tags source
    looks at every image no matter what the Images choice says."""
    import time as _time
    from PIL import Image
    from app.services import captioner, captions
    folder = tmp_path / "ds-mode"
    folder.mkdir()
    for i, cap in enumerate(["1girl, blue hair", ""]):
        Image.new("RGB", (8, 8), "white").save(folder / f"i{i}.png")
        if cap:
            captions.write_caption(str(folder / f"i{i}.png"), cap)

    status = captioner.run({"folder": str(folder), "source": "tags", "mode": "missing_only"})
    for _ in range(200):                                   # the batch runs on a thread
        status = captioner.run_status()
        if status["status"] != "running":
            break
        _time.sleep(0.05)
    assert status["status"] == "done", status
    assert status["captioned"] == 1 and status["skipped"] == 1
    assert captions.read_caption(str(folder / "i0.png")) == "A girl with blue hair."
    assert captions.read_caption(str(folder / "i1.png")) == ""


def test_vlm_job_loads_the_downloaded_copy(monkeypatch):
    """The worker is pointed at the folder Download filled, not at the hub id."""
    from app.services import captioner
    monkeypatch.setattr(captioner, "is_downloaded", lambda name: True)
    job = captioner._vlm_job("joycaption-beta", {}, [{"path": "a.png"}])
    assert job["repo"] == str(captioner.model_dir("joycaption-beta"))
    assert job["kind"] == "chat" and job["family"] == "llava"
    assert job["images"] == [{"id": "a.png"}]

    monkeypatch.setattr(captioner, "is_downloaded", lambda name: False)
    assert captioner._vlm_job("joycaption-beta", {}, [])["repo"] == "fancyfeast/llama-joycaption-beta-one-hf-llava"


# ------------------------------------------------------------------ GPU telemetry

_FAKE_SMI = """0, NVIDIA GeForce RTX 4090, 87, 12288, 24564, 62, 320.5, 450.0, 55
1, NVIDIA GeForce RTX 3060, 0, 512, 12288, 41, 18.0, 170.0, [N/A]
"""


def test_gpu_parses_nvidia_smi(monkeypatch):
    from app.services import gpu
    gpu._cache.update(at=0.0, data=None)
    monkeypatch.setattr(gpu, "_run", lambda args: _FAKE_SMI)
    data = gpu.query()
    assert data["available"] is True and len(data["gpus"]) == 2
    first, second = data["gpus"]
    assert first["name"] == "NVIDIA GeForce RTX 4090" and first["util_pct"] == 87
    assert first["mem_used_mb"] == 12288 and first["mem_pct"] == 50.0
    assert first["temp_c"] == 62 and first["power_w"] == 320.5 and first["fan_pct"] == 55
    # "[N/A]" (a laptop GPU with no fan reading) must become None, not a crash
    assert second["fan_pct"] is None and second["index"] == 1


def test_gpu_reports_missing_driver_without_raising(monkeypatch):
    from app.services import gpu
    gpu._cache.update(at=0.0, data=None)

    def no_smi(args):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(gpu, "_run", no_smi)
    data = gpu.query()
    assert data["available"] is False and data["gpus"] == [] and "driver" in data["reason"]

    # and it caches: a second call must not shell out again
    calls = []
    monkeypatch.setattr(gpu, "_run", lambda args: calls.append(args) or _FAKE_SMI)
    data = gpu.query(force=True)
    assert data["gpus"][0]["name"].startswith("NVIDIA") and len(calls) == 1
    gpu.query()
    assert len(calls) == 1          # served from the cache
    gpu._cache.update(at=0.0, data=None)


# ------------------------------------------------------------------ backend verify

def test_backend_verify_lists_what_is_missing(monkeypatch):
    from app.services import backend_client, backend_manager, backend_setup
    monkeypatch.setattr(backend_client, "ping", lambda url: {"reachable": False})
    monkeypatch.setattr(backend_manager, "status", lambda: {"running": False})
    # pretend the backend venv was never installed (this machine may have one)
    monkeypatch.setattr(backend_setup, "_venv_python", lambda: Path("Z:/no-such-venv/python.exe"))
    result = backend_setup.verify()
    by_id = {c["id"]: c for c in result["checks"]}
    # the bundled source is in this checkout, so that part must pass
    assert by_id["files"]["status"] == "ok"
    # ...but a missing backend venv is a blocker
    assert by_id["venv"]["status"] == "fail" and "Install dependencies" in by_id["venv"]["fix"]
    assert by_id["process"]["status"] == "warn"
    assert result["ok"] is False and result["failed"] >= 1
    assert all(c["status"] in ("ok", "warn", "fail") for c in result["checks"])


# ------------------------------------------------------------------ collect downloads

class _FakeResponse:
    """Minimal stand-in for a requests response, as open_image returns one."""

    status_code = 200

    def __init__(self, chunks, ctype="image/jpeg"):
        self._chunks = chunks
        self.headers = {"Content-Type": ctype}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=0):
        yield from self._chunks

    def close(self):
        pass


def _dl_item(url="https://cdn.donmai.us/original/ab/cd/abcd.jpg"):
    return {"file_url": url, "source": "danbooru", "id": 7, "extension": "jpg"}


def test_download_stream_failure_leaves_no_broken_file(tmp_path, monkeypatch):
    from app.services import downloader

    def chunks():
        yield b"\xff\xd8\xff\xe0" + b"x" * 2048
        raise requests.exceptions.ChunkedEncodingError("connection lost mid-download")

    monkeypatch.setattr(downloader, "open_image", lambda *a, **k: _FakeResponse(chunks()))
    out = downloader._download_one(_dl_item(), tmp_path, False)
    assert out["status"] == "error" and "connection lost" in out["reason"]
    assert list(tmp_path.iterdir()) == []      # no truncated jpg, no .part left behind


def test_download_rejects_a_web_page_served_as_an_image(tmp_path, monkeypatch):
    from app.services import downloader
    monkeypatch.setattr(downloader, "open_image",
                        lambda *a, **k: _FakeResponse([b"<!DOCTYPE html><html>403 blocked</html>"]))
    out = downloader._download_one(_dl_item(), tmp_path, False)
    assert out["status"] == "error" and "web page" in out["reason"]
    assert list(tmp_path.iterdir()) == []


def test_download_rewrites_an_empty_leftover_file(tmp_path, monkeypatch):
    from app.services import downloader
    target = tmp_path / "danbooru_7.jpg"
    target.write_bytes(b"")                    # what an interrupted run can leave behind
    monkeypatch.setattr(downloader, "open_image", lambda *a, **k: _FakeResponse([b"\xff\xd8\xff\xe0jpeg body"]))
    out = downloader._download_one(_dl_item(), tmp_path, False)
    assert out["status"] == "ok" and target.stat().st_size == 13
    assert not (tmp_path / "danbooru_7.jpg.part").exists()


def test_download_still_skips_a_real_file(tmp_path, monkeypatch):
    from app.services import downloader
    (tmp_path / "danbooru_7.jpg").write_bytes(b"\xff\xd8\xff\xe0already here")

    def boom(*a, **k):
        raise AssertionError("a complete file must not be downloaded again")
    monkeypatch.setattr(downloader, "open_image", boom)
    assert downloader._download_one(_dl_item(), tmp_path, False)["status"] == "skipped"


def test_download_batch_reports_a_crash_instead_of_losing_the_batch(tmp_path, monkeypatch):
    from app.services import downloader

    def crash(item, dest, overwrite):
        raise RuntimeError("disk full")
    monkeypatch.setattr(downloader, "_download_one", crash)
    out = downloader.download_batch([_dl_item()], tmp_path)
    assert len(out) == 1 and out[0]["status"] == "error" and "disk full" in out[0]["reason"]


# ------------------------------------------------------------------ image proxy cache

def test_img_cache_roundtrip_and_prune(tmp_path, monkeypatch):
    import os
    import time
    from app.services import img_cache
    monkeypatch.setattr(img_cache, "CACHE_DIR", tmp_path / "imgcache")
    monkeypatch.setattr(img_cache, "MAX_BYTES", 400)

    old = "https://cdn.donmai.us/preview/ab/cd/old.jpg"
    new = "https://cdn.donmai.us/preview/ab/cd/new.jpg"
    assert img_cache.get(old) is None
    img_cache.put(old, b"a" * 200, "image/jpeg")
    assert img_cache.get(old) == (b"a" * 200, "image/jpeg")
    img_cache.put(new, b"b" * 200, "image/jpeg")

    stale = time.time() - 600                  # make "old" clearly the least recently used
    for p in img_cache._paths(old):
        os.utime(p, (stale, stale))
    img_cache._prune()
    assert img_cache.get(new) == (b"b" * 200, "image/jpeg")
    assert img_cache.get(old) is None
