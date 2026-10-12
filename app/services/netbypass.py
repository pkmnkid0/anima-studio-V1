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

What this can NOT fix: networks that block by IP address or by reading the
site name in the TLS handshake (SNI). For those, set a proxy (Explorer >
Connection) - e.g. the local port of a VPN / Cloudflare WARP / Tor client.

Only hosts the app itself talks to for boorus are affected; pip, update checks
and everything else use the normal system resolver.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
import time

import requests
import urllib3.util.connection as _u3conn

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
    out = {"host": host, "system": {"ok": False, "ips": [], "error": ""}, "secure": {"ok": False, "ips": [], "error": "", "provider": ""}}
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
    return out
