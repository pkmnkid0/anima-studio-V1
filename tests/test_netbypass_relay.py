"""The local relay that sends the TLS hello in pieces, and the retry through it."""
import socket
import ssl
import threading

import pytest
import requests

from app.services import netbypass


def _client_hello(host: str) -> bytes:
    """A real ClientHello for `host`, captured from Python's own TLS stack."""
    ctx = ssl.create_default_context()
    inb, outb = ssl.MemoryBIO(), ssl.MemoryBIO()
    obj = ctx.wrap_bio(inb, outb, server_hostname=host)
    try:
        obj.do_handshake()
    except ssl.SSLWantReadError:
        pass
    return outb.read()


def test_sni_found_and_split_in_the_middle():
    data = _client_hello("danbooru.donmai.us")
    a, b = netbypass.sni_range(data)
    assert data[a:b] == b"danbooru.donmai.us"
    cuts = netbypass.split_points(data)
    assert any(a < c < b for c in cuts)            # the name never travels in one piece
    assert cuts == sorted(cuts) and all(0 < c < len(data) for c in cuts)


def test_split_points_for_junk_is_safe():
    assert netbypass.sni_range(b"GET / HTTP/1.1\r\n") is None
    assert netbypass.split_points(b"x" * 100)


def test_relay_forwards_connect_and_fragments_hello(monkeypatch):
    seen = []
    up = socket.socket(); up.bind(("127.0.0.1", 0)); up.listen(1)

    def upstream():
        c, _ = up.accept()
        c.settimeout(2)
        try:
            while True:
                chunk = c.recv(65536)
                if not chunk:
                    break
                seen.append(chunk)
                if len(b"".join(seen)) >= len(hello):
                    c.sendall(b"pong")
        except OSError:
            pass
        c.close()

    hello = _client_hello("example.booru.test")
    threading.Thread(target=upstream, daemon=True).start()
    monkeypatch.setattr(netbypass, "resolve_doh", lambda host: ["127.0.0.1"])
    port = up.getsockname()[1]
    px = netbypass.relay_url().rsplit(":", 1)[1]
    cl = socket.create_connection(("127.0.0.1", int(px)), timeout=3)
    cl.sendall(f"CONNECT example.booru.test:{port} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    assert b"200" in cl.recv(200)
    cl.sendall(hello)
    assert cl.recv(10) == b"pong"
    cl.close()
    assert b"".join(seen) == hello                  # nothing lost or reordered
    assert len(seen) >= 2                           # but it did not arrive in one packet
    assert not any(b"example.booru.test" in chunk for chunk in seen)


def test_adapter_falls_back_to_relay_and_remembers(monkeypatch):
    calls = []
    monkeypatch.setattr(netbypass, "_host_check", lambda h: True)
    monkeypatch.setattr(netbypass, "get_mode", lambda: "auto")
    monkeypatch.setattr(netbypass, "fragment_enabled", lambda: True)
    netbypass._needs_relay.clear()

    def fake_send(self, request, **kw):
        via = bool((kw.get("proxies") or {}).get("https"))
        calls.append(via)
        if not via:
            raise requests.exceptions.ConnectionError("reset by peer")
        return "ok"

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    ad = netbypass.BypassAdapter()
    req = requests.Request("GET", "https://blocked.example/x").prepare()
    assert ad.send(req) == "ok"
    assert calls == [False, True]
    calls.clear()
    assert ad.send(req) == "ok"                      # next time it goes straight through the relay
    assert calls == [True]


def test_adapter_untouched_when_user_has_a_proxy_or_mode_direct(monkeypatch):
    monkeypatch.setattr(netbypass, "_host_check", lambda h: True)
    monkeypatch.setattr(netbypass, "fragment_enabled", lambda: True)
    req = requests.Request("GET", "https://blocked.example/x").prepare()
    monkeypatch.setattr(netbypass, "get_mode", lambda: "direct")
    assert not netbypass._eligible(req, None)
    monkeypatch.setattr(netbypass, "get_mode", lambda: "auto")
    assert not netbypass._eligible(req, {"https": "http://127.0.0.1:1"})
    assert netbypass._eligible(req, {})


def test_caption_deps_probe_flags_old_or_broken_bitsandbytes():
    from app.services import captioner
    assert "bitsandbytes>=0.46.1" in captioner.CAPTION_PIP
    src = open(captioner.__file__, encoding="utf-8").read()
    assert "import bitsandbytes" in src and "will not load" in src
