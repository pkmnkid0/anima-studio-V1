"""Guards for a local web app that must not be drivable by other websites.

Anima Studio listens on 127.0.0.1, but a browser on the same machine will happily
send requests there on behalf of ANY web page you have open. Without these
checks a malicious page could list folders, read images, rewrite captions or
start training. Three defences, all cheap:

1. No CORS. The UI is served by this same server, so it never needs it.
2. Same-origin only on /api: browsers label every request with
   `Sec-Fetch-Site` and (for writes) `Origin`; anything that is not
   same-origin is refused. curl, Python and the desktop window send neither
   header and are unaffected.
3. Host allow-list, which stops DNS-rebinding (a hostile domain that resolves
   to 127.0.0.1 and so looks "same-origin" to the browser).
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from flask import Flask, jsonify, request

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _allowed_hosts(bind_host: str) -> set[str] | None:
    """None = do not check (the user deliberately bound a non-loopback address,
    e.g. to use the app from another device on their network)."""
    if bind_host not in _LOOPBACK:
        return None
    extra = {h.strip().lower() for h in os.environ.get("ANIMA_STUDIO_ALLOWED_HOSTS", "").split(",") if h.strip()}
    return _LOOPBACK | extra


def install(app: Flask, bind_host: str) -> None:
    allowed = _allowed_hosts(bind_host)

    @app.before_request
    def _guard():
        if allowed is not None:
            host = (request.host or "").rsplit(":", 1)[0].strip("[]").lower() if not request.host.startswith("[") \
                else request.host.split("]")[0].strip("[").lower()
            if host not in allowed:
                return jsonify({"error": "Unexpected host name."}), 403
        if not request.path.startswith("/api/"):
            return None
        site = request.headers.get("Sec-Fetch-Site")
        if site and site not in ("same-origin", "none"):
            return jsonify({"error": "Cross-site requests are not allowed."}), 403
        origin = request.headers.get("Origin")
        if origin and request.method not in ("GET", "HEAD", "OPTIONS"):
            if urlparse(origin).netloc.lower() != (request.host or "").lower():
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
        return None

    @app.after_request
    def _headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")          # no clickjacking via an iframe
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp
