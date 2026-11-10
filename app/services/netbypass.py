"""Connection modes for reaching booru sites on networks that block them.

Most blocks are done by the network's DNS server: it answers "no such site" or
a fake address for the booru's name. This module answers that by looking the
name up through encrypted DNS-over-HTTPS (Cloudflare / Google / Quad9, tried by
IP so that no DNS lookup is needed to reach them) and connecting to the real
address. The TLS certificate and Host header still use the site's real name,
so nothing about the connection is weakened.

Modes (setting `net_mode`):
  auto    - normal DNS first; on a failure or an obviously fake answer, secure
            DNS. Remembers which hosts needed it. (default)
  secure  - always secure DNS for booru hosts.
  direct  - never touch DNS; plain system behaviour.

Blocks that read the site name in the TLS handshake (SNI - the common one on
filtered ISPs) are handled by a small local relay inside the app: when a booru
connection fails, the same request is retried through it. The relay forwards
the connection untouched but sends the first TLS packet in several tiny pieces,
split in the middle of the site name, which the network's filter cannot match.
The browser-grade TLS session still runs end to end between this app and the
site (certificates are fully verified); the relay only sees encrypted bytes.

What this can NOT fix: blocks by IP address. For those, set a proxy (Explorer >
Connection) - e.g. the local port of a VPN / Cloudflare WARP / Tor client.

Only hosts the app itself talks to for boorus are affected; pip, update checks
and everything else use the normal system resolver.
"""
from __future__ import annotations

import ipaddress
import socket
import struct
import threading
import time
from urllib.parse import urlparse

import requests
import urllib3.util.connection as _u3conn
from requests.adapters import HTTPAdapter

from . import settings_store

MODES = ("auto", "secure", "direct")

# (label, DoH JSON endpoint). IP-literal URLs: reaching them needs no DNS.
PROVIDERS = [
    ("Cloudflare", "https://1.1.1.1/dns-query"),
    ("Cloudflare (alt)", "https://1.0.0.1/dns-query"),
    ("Google", "https://8.8.8.8/resolve"),
    ("Quad9", "https://9.9.9.9:5053/dns-query"),
]
_DOH_TIMEOUT = (3.5, 4.5)
_CACHE_TTL = 600
_NEG_TTL = 30
_PREFER_TTL = 1800

_lock = threading.Lock()
_cache: dict[str, tuple[float, list[str]]] = {}
_fail: dict[str, tuple[float, str]] = {}
_prefer_doh: dict[str, float] = {}      # host -> until: skip system DNS, it failed recently
_last_provider = {"name": ""}
_installed = False
_orig_create_connection = _u3conn.create_connection
_host_check = None                      # set by install(): callable(host) -> bool


def get_mode() -> str:
    m = settings_store.get("net_mode", "auto")
    return m if m in MODES else "auto"


def set_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    settings_store.set("net_mode", mode)
    with _lock:
        _prefer_doh.clear()
    return mode


def _bogus(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return a.is_private or a.is_loopback or a.is_unspecified or a.is_link_local or a.is_multicast or a.is_reserved


def _doh_query(url: str, host: str, rtype: str) -> list[str]:
    r = requests.get(url, params={"name": host, "type": rtype}, headers={"Accept": "application/dns-json"},
                     timeout=_DOH_TIMEOUT)
    r.raise_for_status()
    want = 1 if rtype == "A" else 28
    return [a["data"] for a in (r.json().get("Answer") or []) if a.get("type") == want and a.get("data")]


def resolve_doh(host: str) -> list[str]:
    """Public IPs for `host` via DoH (IPv4 first). Cached. Raises OSError if every provider fails."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(host)
        if hit and hit[0] > now:
            if not hit[1]:
                raise OSError(f"secure DNS found no address for {host}")
            return list(hit[1])
        fail = _fail.get(host)
        if fail and fail[0] > now:                 # every resolver just failed: don't make each request wait again
            raise OSError(fail[1])
    last: Exception | None = None
    for name, url in PROVIDERS:
        try:
            ips = [ip for ip in _doh_query(url, host, "A") if not _bogus(ip)]
            if not ips:
                ips = [ip for ip in _doh_query(url, host, "AAAA") if not _bogus(ip)]
            with _lock:
                _cache[host] = (now + (_CACHE_TTL if ips else _NEG_TTL), ips)
                _last_provider["name"] = name
            if ips:
                return ips
            last = OSError(f"{name} has no address for {host}")
            break                      # authoritative "doesn't exist": other providers will agree
        except (requests.RequestException, ValueError, KeyError) as exc:
            last = exc
            continue
    msg = f"secure DNS lookup for {host} failed: {last}"
    with _lock:
        _fail[host] = (now + 20, msg)
    raise OSError(msg)


def _connect_ips(ips: list[str], port: int, timeout, source_address, socket_options):
    err: Exception | None = None
    for ip in ips:
        try:
            return _orig_create_connection((ip, port), timeout, source_address, socket_options)
        except OSError as exc:
            err = exc
    raise err or OSError("no address to connect to")


def _create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, socket_options=None, **kw):
    host, port = address
    mode = get_mode()
    if mode == "direct" or _host_check is None or not _host_check(host):
        return _orig_create_connection(address, timeout, source_address, socket_options, **kw)
    if mode == "secure" or _prefer_doh.get(host, 0) > time.monotonic():
        return _connect_ips(resolve_doh(host), port, timeout, source_address, socket_options)
    # auto: system DNS first, but never trust a fake answer or a dead address
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        ips = [i[4][0] for i in infos]
        if not ips or all(_bogus(ip) for ip in ips):
            raise OSError("system DNS returned a fake address")
        return _connect_ips([ip for ip in ips if not _bogus(ip)], port, timeout, source_address, socket_options)
    except OSError as first:
        try:
            sock = _connect_ips(resolve_doh(host), port, timeout, source_address, socket_options)
        except OSError:
            raise first               # report the original, simpler error
        with _lock:
            _prefer_doh[host] = time.monotonic() + _PREFER_TTL
        return sock


def install(host_check) -> None:
    """Hook secure DNS into every `requests`/urllib3 connection this process makes.
    `host_check(host)` decides which hosts it may touch."""
    global _installed, _host_check
    _host_check = host_check
    if not _installed:
        _u3conn.create_connection = _create_connection
        _installed = True


def test_host(host: str) -> dict:
    """Does each resolver path work for this host? Used by the Connection panel."""
    out = {"host": host, "system": {"ok": False, "ips": [], "error": ""}, "secure": {"ok": False, "ips": [], "error": "", "provider": ""},
           "relay": {"ok": False, "error": "", "tested": False}}
    try:
        ips = sorted({i[4][0] for i in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
        out["system"].update(ips=ips, ok=bool(ips) and not all(_bogus(i) for i in ips),
                             error="" if not all(_bogus(i) for i in ips) else "fake/blocked address")
    except OSError as exc:
        out["system"]["error"] = str(exc)
    try:
        with _lock:
            _cache.pop(host, None)
            _fail.pop(host, None)
        ips = resolve_doh(host)
        out["secure"].update(ok=True, ips=ips, provider=_last_provider["name"])
    except OSError as exc:
        out["secure"]["error"] = str(exc)
    if fragment_enabled():
        out["relay"] = {**test_relay(host), "tested": True}
    return out


# ----------------------------------------------------------------------------------------------
# Local relay that sends the TLS hello in pieces (gets past SNI-based blocking)
# ----------------------------------------------------------------------------------------------

_relay_lock = threading.Lock()
_relay: dict = {"server": None, "port": 0}
_needs_relay: dict[str, float] = {}            # host -> until: this host failed directly but works through the relay
_RELAY_TTL = 6 * 3600


def fragment_enabled() -> bool:
    return bool(settings_store.get("net_fragment", True))


def set_fragment(on: bool) -> bool:
    settings_store.set("net_fragment", bool(on))
    return bool(on)


def sni_range(data: bytes) -> tuple[int, int] | None:
    """(start, end) of the server name inside a TLS ClientHello record, or None."""
    try:
        if len(data) < 48 or data[0] != 0x16 or data[5] != 0x01:
            return None
        pos = 5 + 4 + 2 + 32                               # record header, handshake header, version, random
        pos += 1 + data[pos]                               # session id
        pos += 2 + struct.unpack(">H", data[pos:pos + 2])[0]     # cipher suites
        pos += 1 + data[pos]                               # compression methods
        end = pos + 2 + struct.unpack(">H", data[pos:pos + 2])[0]
        pos += 2
        while pos + 4 <= min(end, len(data)):
            etype, elen = struct.unpack(">HH", data[pos:pos + 4])
            if etype == 0:                                 # server_name: list len(2) type(1) name len(2) name
                nlen = struct.unpack(">H", data[pos + 7:pos + 9])[0]
                return pos + 9, pos + 9 + nlen
            pos += 4 + elen
    except (IndexError, struct.error):
        pass
    return None


def split_points(data: bytes) -> list[int]:
    """Where to cut the first packet: inside the record header and in the middle of the site name."""
    cuts = {2}
    rng = sni_range(data)
    if rng:
        cuts.add(rng[0] + max(1, (rng[1] - rng[0]) // 2))
        cuts.add(rng[0] + 1)
    else:
        cuts.add(min(len(data) - 1, 40))
    return sorted(c for c in cuts if 0 < c < len(data))


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        for sk in (src, dst):
            try:
                sk.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _handle_client(client: socket.socket) -> None:
    upstream = None
    try:
        client.settimeout(15)
        head = b""
        while b"\r\n\r\n" not in head and len(head) < 8192:
            part = client.recv(4096)
            if not part:
                return
            head += part
        first = head.split(b"\r\n", 1)[0].decode("latin-1").split()
        if len(first) < 2 or first[0].upper() != "CONNECT":
            client.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            return
        host, _, port = first[1].rpartition(":")
        port = int(port or 443)
        try:
            upstream = _connect_ips(resolve_doh(host), port, 8, None, None)
        except OSError:
            upstream = _orig_create_connection((host, port), 8)
        client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        client.settimeout(None)
        upstream.settimeout(None)
        for sk in (client, upstream):
            try:
                sk.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        hello = client.recv(65536)
        if not hello:
            return
        cuts = split_points(hello)
        prev = 0
        for c in cuts + [len(hello)]:
            upstream.sendall(hello[prev:c])
            prev = c
            if c != len(hello):
                time.sleep(0.03)
        threading.Thread(target=_pipe, args=(upstream, client), daemon=True).start()
        _pipe(client, upstream)
    except (OSError, ValueError):
        pass
    finally:
        for sk in (client, upstream):
            if sk is not None:
                try:
                    sk.close()
                except OSError:
                    pass


def _serve(server: socket.socket) -> None:
    while True:
        try:
            client, _ = server.accept()
        except OSError:
            return
        threading.Thread(target=_handle_client, args=(client,), daemon=True).start()


def relay_url() -> str:
    """Start the relay on first use and return its proxy URL (127.0.0.1 only)."""
    with _relay_lock:
        if _relay["server"] is None:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.bind(("127.0.0.1", 0))
            srv.listen(64)
            _relay.update(server=srv, port=srv.getsockname()[1])
            threading.Thread(target=_serve, args=(srv,), daemon=True, name="netbypass-relay").start()
        return f"http://127.0.0.1:{_relay['port']}"


def _eligible(request, proxies) -> bool:
    if get_mode() == "direct" or not fragment_enabled() or _host_check is None:
        return False
    if (proxies or {}).get("https") or not request.url.lower().startswith("https://"):
        return False
    return bool(_host_check(urlparse(request.url).hostname or ""))


_RETRYABLE = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


class BypassAdapter(HTTPAdapter):
    """HTTPAdapter that falls back to the fragmenting relay when a booru host cannot be reached directly,
    and remembers which hosts need it so later requests go straight there."""

    def send(self, request, **kwargs):
        if not _eligible(request, kwargs.get("proxies")):
            return super().send(request, **kwargs)
        host = urlparse(request.url).hostname or ""
        via = {"http": relay_url(), "https": relay_url()}
        if _needs_relay.get(host, 0) > time.monotonic():
            try:
                return super().send(request, **{**kwargs, "proxies": via})
            except _RETRYABLE:
                _needs_relay.pop(host, None)               # the relay stopped helping: try the plain route again
        try:
            return super().send(request, **kwargs)
        except _RETRYABLE as first:
            try:
                resp = super().send(request, **{**kwargs, "proxies": via})
            except _RETRYABLE:
                raise first
            _needs_relay[host] = time.monotonic() + _RELAY_TTL
            return resp


def test_relay(host: str) -> dict:
    """Does an HTTPS request to the site get through the relay?"""
    out = {"ok": False, "error": ""}
    try:
        px = relay_url()
        r = requests.head(f"https://{host}/", proxies={"https": px, "http": px}, timeout=(6, 8), allow_redirects=False)
        out["ok"] = r.status_code < 600
    except requests.RequestException as exc:
        out["error"] = str(exc).splitlines()[0][:120]
    return out
