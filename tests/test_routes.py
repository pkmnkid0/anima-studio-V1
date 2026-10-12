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
