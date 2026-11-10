"""Undo / redo of edits that were already applied to a picture."""
import base64
import io

from PIL import Image

from app.services import imaging


def _img(tmp_path, name="a.png", color=(200, 30, 30)):
    p = tmp_path / name
    Image.new("RGB", (64, 64), color).save(p)
    return p


def _mask(box):
    m = Image.new("L", (64, 64), 0)
    m.paste(255, box)
    buf = io.BytesIO(); m.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_undo_and_redo_walk_through_each_edit(client, tmp_path):
    p = _img(tmp_path)
    original = p.read_bytes()
    for box in ((5, 5, 20, 20), (30, 30, 50, 50)):
        r = client.post("/api/edit/inpaint", json={"image_path": str(p), "mask": _mask(box), "method": "telea"})
        assert r.status_code == 200
    after2 = p.read_bytes()
    assert client.get("/api/edit/history", query_string={"image_path": str(p)}).get_json() == {"undo": 2, "redo": 0}

    r = client.post("/api/edit/history/undo", json={"image_path": str(p)}).get_json()
    assert r["ok"] and r["undo"] == 1 and r["redo"] == 1
    assert p.read_bytes() != after2
    r = client.post("/api/edit/history/undo", json={"image_path": str(p)}).get_json()
    assert r["ok"] and p.read_bytes() == original
    assert not client.post("/api/edit/history/undo", json={"image_path": str(p)}).get_json()["ok"]

    assert client.post("/api/edit/history/redo", json={"image_path": str(p)}).get_json()["ok"]
    assert client.post("/api/edit/history/redo", json={"image_path": str(p)}).get_json()["ok"]
    assert p.read_bytes() == after2
    assert not client.post("/api/edit/history/redo", json={"image_path": str(p)}).get_json()["ok"]


def test_new_edit_after_undo_drops_redo(client, tmp_path):
    p = _img(tmp_path)
    client.post("/api/edit/inpaint", json={"image_path": str(p), "mask": _mask((5, 5, 20, 20)), "method": "telea"})
    client.post("/api/edit/history/undo", json={"image_path": str(p)})
    client.post("/api/edit/inpaint", json={"image_path": str(p), "mask": _mask((8, 8, 30, 30)), "method": "telea"})
    assert client.get("/api/edit/history", query_string={"image_path": str(p)}).get_json()["redo"] == 0


def test_crop_and_revert_are_undoable(client, tmp_path):
    p = _img(tmp_path)
    original = p.read_bytes()
    client.post("/api/edit/crop", json={"image_path": str(p), "box": [0, 0, 32, 32]})
    assert Image.open(p).size == (32, 32)
    client.post("/api/edit/revert", json={"image_path": str(p)})
    assert Image.open(p).size == (64, 64)
    client.post("/api/edit/history/undo", json={"image_path": str(p)})          # undo the revert
    assert Image.open(p).size == (32, 32)
    client.post("/api/edit/history/undo", json={"image_path": str(p)})
    assert p.read_bytes() == original


def test_copies_are_not_tracked_and_bad_requests(client, tmp_path):
    p = _img(tmp_path)
    client.post("/api/edit/inpaint", json={"image_path": str(p), "mask": _mask((5, 5, 20, 20)), "method": "telea", "save_as_copy": True})
    assert client.get("/api/edit/history", query_string={"image_path": str(p)}).get_json()["undo"] == 0
    assert client.post("/api/edit/history/undo", json={}).status_code == 400
    assert client.post("/api/edit/history/nope", json={"image_path": str(p)}).status_code == 400


def test_history_is_capped(client, tmp_path, monkeypatch):
    monkeypatch.setattr(imaging, "HISTORY_KEEP", 3)
    p = _img(tmp_path)
    for i in range(6):
        client.post("/api/edit/inpaint", json={"image_path": str(p), "mask": _mask((i, i, i + 10, i + 10)), "method": "telea"})
    assert client.get("/api/edit/history", query_string={"image_path": str(p)}).get_json()["undo"] == 3
