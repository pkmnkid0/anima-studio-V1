"""Backend tab: download a set of AI edit models in one go."""
import time

import pytest

from app.services import edit_engine, edit_models
from edit_backend import catalog


def test_every_pack_item_is_known_and_valid():
    for pack in catalog.MODEL_PACKS:
        assert pack["items"]
        for kind, name in pack["items"]:
            edit_engine.build_request({"cmd": "download", "kind": kind, "name": name})
    ids = {(i["kind"], i["name"]) for g in catalog.pick_groups() for i in g["items"]}
    assert ("model", "lama") in ids and ("face", "gfpgan") in ids
    pub = catalog.public_catalog()
    assert pub["packs"] and pub["pick_groups"]


def _wait():
    for _ in range(200):
        if edit_models.status()["state"] != "running":
            return
        time.sleep(0.02)
    raise AssertionError("did not finish")


@pytest.fixture()
def fake(monkeypatch):
    calls = []
    monkeypatch.setattr(edit_engine, "is_installed", lambda: True)

    def call(cmd, args=None, job=None):
        if cmd == "models":
            return {"downloaded": ["lama"], "plugins": {"sam:mobile_sam": True}}
        calls.append((args["kind"], args["name"]))
        if args["name"] == "manga":
            raise edit_engine.EngineError("network down")
        return {"downloaded": True}

    monkeypatch.setattr(edit_engine, "call", call)
    monkeypatch.setattr(edit_engine, "stop", lambda why="": True)
    return calls


def test_pack_skips_existing_and_records_failures(fake):
    edit_models.start({"pack": "essentials"})
    _wait()
    s = edit_models.status()
    assert s["state"] == "done" and s["total"] == 7 and s["done"] == 7 and s["skipped"] == 2
    assert [f["name"] for f in s["failed"]] == ["Manga"]
    assert ("model", "lama") not in fake and ("sam", "mobile_sam") not in fake and ("gfpgan", "gfpgan") in fake
    assert edit_engine.reserve_for_batch()            # engine released afterwards
    edit_engine.release_batch()


def test_custom_choice_validated(fake):
    with pytest.raises(edit_models.ModelsError):
        edit_models.start({"items": []})
    with pytest.raises(edit_models.ModelsError):
        edit_models.start({"items": [{"kind": "model", "name": "../../evil"}]})
    with pytest.raises(edit_models.ModelsError):
        edit_models.start({"pack": "nope"})
    edit_models.start({"items": [{"kind": "face", "name": "restoreformer"}, {"kind": "model", "name": "zits"}]})
    _wait()
    assert ("restoreformer", "restoreformer") in fake and ("model", "zits") in fake


def test_routes(client, fake):
    r = client.post("/api/edit/ai/models/download", json={"pack": "erase"})
    assert r.status_code == 200
    _wait()
    assert client.get("/api/edit/ai/models/download/status").get_json()["state"] == "done"
    assert client.post("/api/edit/ai/models/download", json={"pack": "zzz"}).status_code == 400
