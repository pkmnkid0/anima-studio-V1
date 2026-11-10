from native_backend import caption_worker
from app.services import captioner


def test_auto_picks_the_best_precision_that_fits():
    pick = caption_worker._pick_quant
    # a 16 GB model: 4-bit on 8 GB, 8-bit on 12 GB, bf16 on 24 GB
    assert pick("chat", 16.1, "auto", "cuda", 8.0) == "4bit"
    assert pick("chat", 16.1, "auto", "cuda", 12.0) == "8bit"
    assert pick("chat", 16.1, "auto", "cuda", 24.0) is None
    # a 3B model is already small enough for 12 GB bf16, and 4-bit on 8 GB
    assert pick("chat", 7.5, "auto", "cuda", 12.0) is None
    assert pick("chat", 7.5, "auto", "cuda", 8.0) == "8bit"
    assert pick("chat", 7.5, "auto", "cuda", 6.0) == "4bit"
    # explicit choices win; nothing quantizes away from CUDA
    assert pick("chat", 16.1, "4bit", "cuda", 24.0) == "4bit"
    assert pick("chat", 16.1, "8bit", "cuda", 24.0) == "8bit"
    assert pick("chat", 16.1, "bf16", "cuda", 8.0) is None
    assert pick("chat", 16.1, "auto", "cpu", 0.0) is None
    assert pick("chat", 0, "auto", "cuda", 8.0) is None


def test_worker_and_app_agree_on_the_precision():
    """caption_worker keeps its own copy of the VRAM maths; it must match captioner's."""
    for name, info in captioner.MODELS.items():
        for vram in (8, 10, 12, 16, 24):
            app_mode = captioner.pick_mode(info, vram)
            worker = caption_worker._pick_quant("chat", info["size_gb"], "auto", "cuda", float(vram))
            assert (worker or "bf16") == app_mode, (name, vram)


def test_every_model_has_an_8gb_path_and_skips_its_vision_tower():
    for name, info in captioner.MODELS.items():
        assert captioner.estimated_vram(info, "4bit") <= 8.0, name       # all of them run on an 8 GB card
        assert "lm_head" in info["skip_modules"] and info["skip_modules"][0] in ("visual", "vision_tower"), name
        assert info["tier"] in ("low", "mid", "high")
    assert captioner.MODELS[captioner.DEFAULT_MODEL]["tier"] != "high"


def test_vram_profile_and_quant_are_sanitised_in_the_job(monkeypatch):
    monkeypatch.setattr(captioner, "is_downloaded", lambda name: True)
    job = captioner._vlm_job("joycaption-beta", {"vram_profile": "8", "quant": "bogus"}, [])
    assert job["vram_profile"] == "8" and job["quant"] == "auto"
    assert job["system"] and "vision_tower" in job["skip_modules"]
    assert captioner._vlm_job("joycaption-beta", {"vram_profile": "999"}, [])["vram_profile"] == "auto"
