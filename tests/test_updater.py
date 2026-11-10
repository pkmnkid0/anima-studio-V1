import io
import json
import time
import zipfile

import pytest

from app.services import updater


def make_zip(files, top="owner-repo-abc123"):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, v in files.items():
            z.writestr(f"{top}/{k}" if top else k, v)
    return b.getvalue()


class FakeResp:
    def __init__(self, data=b"", js=None, status=200):
        self._d, self._j, self.status_code, self.headers = data, js, status, {"Content-Length": str(len(data))}

    def json(self):
        return self._j

    def iter_content(self, n):
        yield self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    app = tmp_path / "app_dir"
    (app / "app").mkdir(parents=True)
    (app / "data").mkdir()
    (app / ".venv").mkdir()
    (app / "moss_backend" / "venv").mkdir(parents=True)
    (app / "run.py").write_text("old")
    (app / "app" / "__init__.py").write_text("old")
    (app / "app" / "gone.py").write_text("old, removed in new release")
    (app / "VERSION").write_text("1.0.0")
    (app / "data" / "keep.json").write_text("mine")
    (app / ".venv" / "lib.py").write_text("venv")
    (app / "moss_backend" / "venv" / "x.py").write_text("venv")
    store = {}
    monkeypatch.setattr(updater, "APP_DIR", app)
    monkeypatch.setattr(updater, "VERSION_FILE", app / "VERSION")
    monkeypatch.setattr(updater, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(updater, "BACKUP_DIR", tmp_path / "bk")
    monkeypatch.setattr(updater.settings_store, "get", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(updater.settings_store, "set", lambda k, v: store.__setitem__(k, v))
    store["update_repo"] = "owner/repo"
    # first run records what is installed so "gone.py" is known to be ours
    updater._save_state({"files": ["run.py", "app/__init__.py", "app/gone.py"]})
    return app, store


def run_apply(monkeypatch, data):
    monkeypatch.setattr(updater.requests, "get", lambda *a, **k: FakeResp(data))
    updater._run("https://api.github.com/x", "1.1.0", "")
    return updater.progress()


def test_version_compare():
    assert updater.newer("v1.10.0", "1.9.9")
    assert not updater.newer("1.0.0", "1.0.0")
    assert updater.newer("2", "1.9")


def test_settings_validation_and_no_token_leak(env):
    app, store = env
    s = updater.save_settings({"repo": "https://github.com/a/b.git", "token": "ghp_secret", "channel": "main"})
    assert s["repo"] == "a/b" and s["has_token"] and "ghp_secret" not in json.dumps(s)
    with pytest.raises(updater.UpdateError):
        updater.save_settings({"repo": "not a repo"})


def test_check_release(env, monkeypatch):
    rel = {"tag_name": "v1.2.0", "body": "notes", "zipball_url": "https://api.github.com/z", "html_url": "u", "assets": []}
    monkeypatch.setattr(updater.requests, "get", lambda *a, **k: FakeResp(js=rel))
    out = updater.check()
    assert out["available"] and out["latest"] == "1.2.0" and out["zip"].endswith("/z")
    rel["tag_name"] = "v1.0.0"
    assert not updater.check()["available"]


def test_check_errors(env, monkeypatch):
    monkeypatch.setattr(updater.requests, "get", lambda *a, **k: FakeResp(status=404))
    with pytest.raises(updater.UpdateError, match="Not found"):
        updater.check()


def test_apply_replaces_code_only(env, monkeypatch):
    app, _ = env
    z = make_zip({"run.py": "new", "app/__init__.py": "new", "app/new.py": "n", "VERSION": "1.1.0",
                  "data/evil.json": "x", ".venv/evil.py": "x", "moss_backend/venv/evil.py": "x",
                  "requirements.txt": "flask"})
    p = run_apply(monkeypatch, z)
    assert p["state"] == "done" and p["needs_restart"], p
    assert (app / "run.py").read_text() == "new"
    assert (app / "app" / "new.py").exists()
    assert not (app / "app" / "gone.py").exists()                      # dropped file removed
    assert (app / "data" / "keep.json").read_text() == "mine"          # data untouched
    assert not (app / "data" / "evil.json").exists()
    assert (app / ".venv" / "lib.py").read_text() == "venv"
    assert not (app / ".venv" / "evil.py").exists()
    assert not (app / "moss_backend" / "venv" / "evil.py").exists()
    bks = list((app.parent / "bk").glob("code-*.zip"))
    assert len(bks) == 1
    names = zipfile.ZipFile(bks[0]).namelist()
    assert "run.py" in names and not any(n.startswith(("data/", ".venv/")) for n in names)


def test_rejects_path_traversal(env, monkeypatch):
    app, _ = env
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("top/run.py", "x")
        z.writestr("top/app/__init__.py", "x")
        z.writestr("top/../../escape.txt", "x")
    p = run_apply(monkeypatch, b.getvalue())
    assert p["state"] == "error"
    assert (app / "run.py").read_text() == "old"                       # nothing changed
    assert not (app.parent / "escape.txt").exists()


def test_rejects_foreign_zip(env, monkeypatch):
    app, _ = env
    p = run_apply(monkeypatch, make_zip({"README.md": "hi"}))
    assert p["state"] == "error" and "not an Anima Studio" in p["error"]
    assert (app / "run.py").read_text() == "old"


def test_start_refuses_unknown_host_and_packaged(env, monkeypatch):
    with pytest.raises(updater.UpdateError):
        updater.start({"zip": "https://evil.example/x.zip"})
    monkeypatch.setattr(updater, "is_packaged", lambda: True)
    with pytest.raises(updater.UpdateError):
        updater.start({"zip": "https://api.github.com/x"})


def test_routes_status_has_no_token(env, client=None):
    from app import create_app
    c = create_app().test_client()
    updater.save_settings({"token": "ghp_abc"})
    r = c.get("/api/update/status")
    assert r.status_code == 200 and b"ghp_abc" not in r.data
