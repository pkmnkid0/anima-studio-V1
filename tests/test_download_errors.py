from app.services.captioner import explain_download_failure as ex


def test_hub_traceback_is_not_reported_as_missing_deps():
    tail = ["File \".../huggingface_hub/file_download.py\", line 1, in x",
            "huggingface_hub.errors.HfHubHTTPError: ConnectionError: timed out"]
    msg = ex(tail, 1, 17.5)
    assert "not installed" not in msg and "connection" in msg.lower()


def test_real_missing_module():
    assert "not installed" in ex(["ModuleNotFoundError: No module named 'huggingface_hub'"], 1)


def test_disk_full_names_size():
    msg = ex(["OSError: [Errno 28] No space left on device"], 1, 17.5)
    assert "disk is full" in msg and "18 GB" in msg


def test_gated_and_404_and_fallback():
    assert "gated" in ex(["HTTPError: 403 Client Error"], 1)
    assert "not found" in ex(["RepositoryNotFoundError: 404"], 1)
    assert "download failed: boom" in ex(["boom"], 1)


def test_gated_message_gives_steps_and_token_never_returned(monkeypatch):
    from app import create_app
    from app.services import captioner
    store = {}
    monkeypatch.setattr(captioner.settings_store, "get", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(captioner.settings_store, "set", lambda k, v: store.__setitem__(k, v))
    assert "Read token" in ex(["Access to model x is restricted. 403"], 1)
    c = create_app().test_client()
    assert c.post("/api/caption/hf_token", json={"token": "hf_abcdefghijk"}).get_json() == {"has_token": True}
    r = c.get("/api/caption/hf_token")
    assert r.get_json() == {"has_token": True} and b"hf_abc" not in r.data
    assert "saved token was not accepted" in ex(["403 gated"], 1)
    assert c.post("/api/caption/hf_token", json={"token": "bad token"}).status_code == 400
