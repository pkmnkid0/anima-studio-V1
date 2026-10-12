import sys
from pathlib import Path

from app.routes.train import _caption_warnings


def _mk(d: Path, name: str, caption: str | None, cap_name: str | None = None):
    (d / name).write_bytes(b"x")
    if caption is not None:
        (d / (cap_name or Path(name).with_suffix(".txt").name)).write_text(caption, encoding="utf-8")


def test_all_captions_present(tmp_path):
    _mk(tmp_path, "a.png", "1girl")
    _mk(tmp_path, "b.jpg", "solo")
    assert _caption_warnings({"image_dir": str(tmp_path), "caption_extension": ".txt"}) == []


def test_names_the_missing_file(tmp_path):
    _mk(tmp_path, "a.png", "1girl")
    _mk(tmp_path, "d.png", "dog", cap_name="d.png_0001.txt")  # tagger-style name: NOT read by the trainer
    w = _caption_warnings({"image_dir": str(tmp_path), "caption_extension": ".txt"})
    assert len(w) == 1 and "d.png needs d.txt" in w[0]


def test_empty_caption_counts_as_missing(tmp_path):
    _mk(tmp_path, "a.png", "   ")
    assert _caption_warnings({"image_dir": str(tmp_path)})


def test_extension_without_dot_does_not_crash(tmp_path):
    _mk(tmp_path, "a.png", "x")
    assert _caption_warnings({"image_dir": str(tmp_path), "caption_extension": "txt"}) == []


def test_dotted_image_names(tmp_path):
    _mk(tmp_path, "b.v2.jpg", "boy")
    assert _caption_warnings({"image_dir": str(tmp_path)}) == []


def test_missing_folder_is_not_an_error():
    assert _caption_warnings({"image_dir": "/definitely/not/here"}) == []


def test_dataset_toml_drops_crop_and_color_aug_with_cache_latents(tmp_path):
    from native_backend import anima_trainer as at
    toml = pytest_import("toml")
    img = tmp_path / "imgs"
    img.mkdir()
    ds = {"subsets": [{"image_dir": str(img), "num_repeats": 2, "random_crop": True,
                       "color_aug": True, "flip_aug": True}]}
    at._CACHE_LATENTS_REQUESTED = True
    try:
        out = at.write_dataset_toml(ds, 1024, tmp_path / "d.toml", 1)
    finally:
        at._CACHE_LATENTS_REQUESTED = False
    sub = toml.load(out)["datasets"][0]["subsets"][0]
    assert "random_crop" not in sub and "color_aug" not in sub
    assert sub["flip_aug"] is True and sub["num_repeats"] == 2


def test_dataset_toml_keeps_crop_without_cache_latents(tmp_path):
    from native_backend import anima_trainer as at
    toml = pytest_import("toml")
    img = tmp_path / "imgs"
    img.mkdir()
    ds = {"subsets": [{"image_dir": str(img), "random_crop": True}]}
    at._CACHE_LATENTS_REQUESTED = False
    out = at.write_dataset_toml(ds, 1024, tmp_path / "d.toml", 1)
    assert toml.load(out)["datasets"][0]["subsets"][0]["random_crop"] is True


def test_dataset_toml_enables_bucketing_by_default(tmp_path):
    from native_backend import anima_trainer as at
    toml = pytest_import("toml")
    img = tmp_path / "imgs"
    img.mkdir()
    out = at.write_dataset_toml({"subsets": [{"image_dir": str(img)}]}, 1024, tmp_path / "d.toml", 1)
    assert toml.load(out)["datasets"][0].get("enable_bucket") is True


def pytest_import(name):
    import pytest
    return pytest.importorskip(name)
