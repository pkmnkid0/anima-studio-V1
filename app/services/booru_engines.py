"""
Booru source registry and engine adapters (Grabber-style).

Grabber works by describing each site as an *engine* (Danbooru, Gelbooru
0.2, Moebooru, e621, Philomena ...) plus a base URL. This module does the
same: SOURCES lists every site, ENGINES holds one adapter per engine
family, and everything the Booru Explorer needs goes through the same few
functions no matter which site is selected:

    explore()       - a page of posts (random / popular / latest / search),
                      plus sidebar tags and "characters in this feed"
    get_post()      - one post with categorized tags and metadata
    tag_wiki()      - wiki text, category, post count, aliases, related tags
    related_tags()  - related / co-occurring tags
    autocomplete()  - live tag suggestions
    pools()         - pool search (Danbooru-family, e621, Moebooru)
    image_proxy_ok()- host allow-list for the thumbnail proxy

Only public, unauthenticated API endpoints are used, the same ones each
site's own pages use. Nothing here gets around a site's login or rating
gate: the rating filter is translated into each site's own rating syntax
and the Explorer defaults to the safest option.

Not covered (they need a login/token or have no public API): Sankaku,
Pixiv, Twitter/X, Zerochan, Anime-Pictures, e-shuushuu. The registry is
data, so adding a site that runs one of the supported engines is one entry.
"""
from __future__ import annotations

import datetime
import html as _html
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import netbypass, settings_store

USER_AGENT = "AnimaLoRAStudio/1.0 (dataset collection tool; +local use)"
# Some Moebooru sites (Lolibooru behind Cloudflare, etc.) refuse non-browser clients.
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0.0.0 Safari/537.36")
TIMEOUT = (8, 20)       # (connect, read) - fail fast instead of hanging the page
FAST_TIMEOUT = (6, 10)

CATEGORIES = ("artist", "copyright", "character", "general", "meta")


class EngineError(RuntimeError):
    """`network` is True for "couldn't get a usable answer from the site"
    (down, blocked, timed out, bot check) as opposed to a bad request."""
    network = False
    kind = ""     # timeout | conn | blocked | rate | server | client | json | api


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

def _src(name, engine, base, **kw):
    d = {"name": name, "engine": engine, "base": base.rstrip("/"), "nsfw": False,
         "sfw_only": False, "anon_tag_limit": None, "random_native": True}
    d.update(kw)
    return d


SOURCES: dict[str, dict] = {
    # Danbooru family
    "danbooru": _src("Danbooru", "danbooru", "https://danbooru.donmai.us", anon_tag_limit=2),
    "aibooru": _src("AIBooru", "danbooru", "https://aibooru.online", nsfw=True, anon_tag_limit=2),
    "safebooru_donmai": _src("Safebooru (Donmai)", "danbooru", "https://safebooru.donmai.us", sfw_only=True),
    # Gelbooru 0.2 family (dapi)
    "gelbooru": _src("Gelbooru", "gelbooru", "https://gelbooru.com", api="https://gelbooru.com/index.php",
                     rating_names=("general", "sensitive", "questionable", "explicit"), creds=True),
    "safebooru": _src("Safebooru", "gelbooru", "https://safebooru.org", api="https://safebooru.org/index.php",
                      sfw_only=True, rating_names=("safe", "safe", "questionable", "explicit")),
    "rule34": _src("Rule34", "gelbooru", "https://rule34.xxx", api="https://api.rule34.xxx/index.php", nsfw=True,
                   rating_names=("safe", "safe", "questionable", "explicit"), creds=True, html_fallback=True),
    "tbib": _src("TBIB", "gelbooru", "https://tbib.org", api="https://tbib.org/index.php", nsfw=True,
                 rating_names=("safe", "safe", "questionable", "explicit")),
    "xbooru": _src("Xbooru", "gelbooru", "https://xbooru.com", api="https://xbooru.com/index.php", nsfw=True,
                   rating_names=("safe", "safe", "questionable", "explicit")),
    "realbooru": _src("Realbooru", "gelbooru", "https://realbooru.com", api="https://realbooru.com/index.php", nsfw=True,
                      rating_names=("safe", "safe", "questionable", "explicit"), random_native=False),
    # Moebooru family
    "konachan": _src("Konachan.com", "moebooru", "https://konachan.com", nsfw=True),
    "konachan_net": _src("Konachan.net", "moebooru", "https://konachan.net", sfw_only=True),
    "yandere": _src("yande.re", "moebooru", "https://yande.re", nsfw=True),
    "sakugabooru": _src("Sakugabooru", "moebooru", "https://www.sakugabooru.com"),
    # e621 family
    "e621": _src("e621", "e621", "https://e621.net", nsfw=True),
    "e926": _src("e926", "e621", "https://e926.net", sfw_only=True),
    # Philomena family
    "derpibooru": _src("Derpibooru", "philomena", "https://derpibooru.org"),
    "furbooru": _src("Furbooru", "philomena", "https://furbooru.org"),
    "ponybooru": _src("Ponybooru", "philomena", "https://ponybooru.org"),
}

# Sources the legacy Collect code (booru.py) already handles itself.
LEGACY_SOURCES = ("danbooru", "gelbooru", "safebooru", "e621", "rule34", "konachan", "yandere")
EXTRA_SOURCE_IDS = [s for s in SOURCES if s not in LEGACY_SOURCES]

ENGINE_LABELS = {
    "danbooru": "Danbooru", "gelbooru": "Gelbooru 0.2", "moebooru": "Moebooru",
    "e621": "e621", "philomena": "Philomena",
}

ENGINE_FEATURES = {
    "danbooru": {"wiki": True, "related": True, "pools": True, "categories": True, "autocomplete": True},
    "e621": {"wiki": True, "related": True, "pools": True, "categories": True, "autocomplete": True},
    "moebooru": {"wiki": True, "related": True, "pools": True, "categories": False, "autocomplete": True},
    "gelbooru": {"wiki": False, "related": False, "pools": False, "categories": False, "autocomplete": True},
    "philomena": {"wiki": True, "related": False, "pools": False, "categories": False, "autocomplete": True},
}

ORDERS = ("random", "popular", "latest", "score", "favorites")

_RATING_ORDER = ("safe", "sensitive", "questionable", "explicit")


def source_info(source: str) -> dict:
    if source not in SOURCES:
        raise EngineError(f"Unknown source: {source}")
    return SOURCES[source]


def list_sources() -> list[dict]:
    out = []
    for sid, d in SOURCES.items():
        feats = dict(ENGINE_FEATURES[d["engine"]])
        out.append({
            "id": sid, "name": d["name"], "engine": d["engine"],
            "engine_label": ENGINE_LABELS[d["engine"]], "base": d["base"],
            "nsfw": d["nsfw"], "sfw_only": d["sfw_only"], "features": feats,
            "max_limit": _max_limit(d["engine"]),
        })
    return out


def _max_limit(engine: str) -> int:
    return {"danbooru": 200, "gelbooru": 100, "moebooru": 100, "e621": 200, "philomena": 50}[engine]


def _host_allowed_suffixes() -> set[str]:
    suffixes = {urlparse(d["base"]).hostname for d in SOURCES.values()}
    suffixes |= {urlparse(d.get("api", d["base"])).hostname for d in SOURCES.values()}
    # CDNs the sites serve media from
    suffixes |= {"donmai.us", "gelbooru.com", "rule34.xxx", "e621.net", "e926.net", "derpicdn.net",
                 "furbooru.org", "furrycdn.org", "ponybooru.org", "aibooru.online",
                 "yande.re", "konachan.com", "konachan.net", "sakugabooru.com",
                 "aibooru.download", "tbib.org", "xbooru.com", "realbooru.com",
                 "safebooru.org", "sakugabooru.com", "ponycdn.net", "furbooru.com"}
    return {s for s in suffixes if s}


MEDIA_HOSTS: set[str] = set()      # image/CDN hosts seen in post data (boorus move media to new CDNs)


def is_booru_host(host: str) -> bool:
    """True for the booru sites and CDNs this app talks to (the only hosts secure DNS touches)."""
    host = _strip_www((host or "").lower())
    if host in MEDIA_HOSTS:
        return True
    return any(host == _strip_www(suf) or host.endswith("." + _strip_www(suf)) for suf in _host_allowed_suffixes())


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def _public_host(host: str) -> bool:
    """Not localhost / a private or loopback IP literal (keeps the image proxy from being pointed inward)."""
    import ipaddress
    if host in ("localhost",) or host.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except ValueError:
        return "." in host


def image_proxy_ok(url: str) -> tuple[bool, str]:
    """(allowed, referer). Any public http(s) host: boorus keep moving their media to new
    CDNs, and a fixed allow-list made those thumbnails show as broken images."""
    try:
        p = urlparse(url)
    except ValueError:
        return False, ""
    if p.scheme not in ("http", "https") or not p.hostname:
        return False, ""
    host = _strip_www(p.hostname.lower())
    if not _public_host(host):
        return False, ""
    return True, referer_for(url) or f"{p.scheme}://{p.hostname}/"


def referer_for(url: str) -> str:
    try:
        host = _strip_www((urlparse(url).hostname or "").lower())
    except ValueError:
        return ""
    best = ""
    for d in SOURCES.values():
        bh = _strip_www(urlparse(d["base"]).hostname or "")
        root = ".".join(bh.split(".")[-2:])
        if host == bh or host.endswith("." + root) or host.endswith(bh):
            best = d["base"] + "/"
            break
    if not best:
        cdn_map = {"derpicdn.net": "https://derpibooru.org/", "furrycdn.org": "https://furbooru.org/",
                   "donmai.us": "https://danbooru.donmai.us/"}
        for k, v in cdn_map.items():
            if host.endswith(k):
                best = v
    return best


# ---------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------

_host_lock = threading.Lock()
_host_last: dict[str, float] = {}
_MIN_GAP = 0.12


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=1, connect=1, read=1, status=1, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET"}), raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,text/html;q=0.8,*/*;q=0.5"})
    px = get_proxy()
    if px:
        s.proxies.update({"http": px, "https": px})
    return s


netbypass.install(lambda h: is_booru_host(h))   # secure-DNS fallback for booru hosts (see netbypass.py)


def get_proxy() -> str:
    """Optional proxy for all booru traffic: the Explorer's proxy box, or the
    ANIMA_BOORU_PROXY environment variable. (System HTTP(S)_PROXY is honoured too.)"""
    import os
    return (settings_store.get("booru_proxy", "") or os.environ.get("ANIMA_BOORU_PROXY", "") or "").strip()


def set_proxy(url: str) -> str:
    url = (url or "").strip()
    if url and not re.match(r"^(https?|socks5h?)://", url, re.I):
        raise EngineError("Proxy must look like http://host:port or socks5h://host:port")
    settings_store.set("booru_proxy", url)
    return url


def probe_sources(ids: list[str] | None = None, seconds: float = 9.0) -> list[dict]:
    """Quick reachability check of each site (used when the Explorer can't load)."""
    ids = ids or list(SOURCES)

    def probe_request(sid):
        d = SOURCES[sid]
        eng_, base = d["engine"], d["base"]
        if eng_ in ("danbooru", "e621"):
            return f"{base}/posts.json", {"limit": 1}, "json"
        if eng_ == "moebooru":
            return f"{base}/post.json", {"limit": 1}, "json"
        if eng_ == "philomena":
            return f"{base}/api/v1/json/search/images", {"q": "safe", "per_page": 1}, "json"
        if d.get("html_fallback") or sid == "gelbooru":   # these gate the JSON API; browsing uses the public page
            return f"{base}/index.php", {"page": "post", "s": "list", "tags": "all"}, "html"
        return d["api"], {"page": "dapi", "s": "post", "q": "index", "json": 1, "limit": 1}, "json"

    def one(sid):
        d = SOURCES[sid]
        t0 = time.monotonic()
        sess = _session()
        sess.headers.update({"User-Agent": BROWSER_UA, "Referer": d["base"] + "/"})
        url, params, kind = probe_request(sid)
        try:
            r = sess.get(url, params=params, timeout=(5, 7))
            r.raise_for_status()
            if kind == "json":
                r.json()  # must really be the API, not a block/captcha page
            return {"id": sid, "name": d["name"], "ok": True, "ms": int((time.monotonic() - t0) * 1000), "error": ""}
        except (requests.exceptions.RequestException, ValueError) as exc:
            return {"id": sid, "name": d["name"], "ok": False, "ms": None, "error": str(_friendly(exc, d["name"]))}
        finally:
            sess.close()

    pool = ThreadPoolExecutor(max_workers=10)
    futs = {sid: pool.submit(one, sid) for sid in ids}
    wait(list(futs.values()), timeout=seconds)
    pool.shutdown(wait=False)
    out = []
    for sid, f in futs.items():
        if f.done():
            out.append(f.result())
        else:
            out.append({"id": sid, "name": SOURCES[sid]["name"], "ok": False, "ms": None, "error": "Timed out"})
    return out


def _pace(url: str) -> None:
    host = urlparse(url).hostname or ""
    with _host_lock:
        now = time.monotonic()
        wait_for = _host_last.get(host, 0.0) + _MIN_GAP - now
        _host_last[host] = max(now, _host_last.get(host, 0.0) + _MIN_GAP)
    if wait_for > 0:
        time.sleep(wait_for)


def _kind_of(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "conn"
    if isinstance(exc, requests.exceptions.HTTPError):
        code = exc.response.status_code if exc.response is not None else 0
        return "blocked" if code in (401, 403) else "rate" if code == 429 else "server" if code >= 500 else "client"
    if isinstance(exc, ValueError):
        return "json"
    return "other"


def _friendly(exc: Exception, source: str) -> EngineError:
    err = _friendly_inner(exc, source)
    err.network = True
    err.kind = _kind_of(exc)
    return err


def _friendly_inner(exc: Exception, source: str) -> EngineError:
    text = str(exc)
    if isinstance(exc, requests.exceptions.Timeout):
        return EngineError(f"{source} took too long to answer (timed out). Your connection to it is slow or filtered - "
                           "try another source, a VPN, or set a proxy (Explorer > connection help).")
    if isinstance(exc, requests.exceptions.ConnectionError):
        if "NameResolutionError" in text or "getaddrinfo" in text:
            return EngineError(
                f"Couldn't look up {source}'s address - your network's DNS is blocking it or is down. "
                f"Open Connection > Test connection; Automatic mode uses encrypted DNS to get around this, "
                f"or set a proxy/VPN there.")
        return EngineError(f"Couldn't connect to {source}. Check your internet connection, firewall or VPN.")
    if isinstance(exc, requests.exceptions.HTTPError):
        code = exc.response.status_code if exc.response is not None else "?"
        if code in (401, 403):
            body = ""
            try:
                body = (exc.response.text or "")[:400]
            except Exception:  # noqa: BLE001
                pass
            low = body.lower()
            if "just a moment" in low or "cf-chl" in low or "attention required" in low or "cloudflare" in low:
                return EngineError(f"{source} is showing a Cloudflare bot check (HTTP {code}) to this app. "
                                   "Try a VPN/proxy, another source, or open the site in your browser first.")
            if "allowlist" in low or "not allowed" in low or "blocked" in low:
                snippet = " ".join(body.split())[:140]
                return EngineError(f"Your network (firewall, proxy, school/work or ISP filter) blocked {source}: \"{snippet}\"")
            return EngineError(f"{source} refused the request (HTTP {code}). It may need a login or API key, "
                               "or it is blocking automated requests right now.")
        if code == 429:
            return EngineError(f"{source} is rate-limiting you (HTTP 429). Wait a bit and try again.")
        return EngineError(f"{source} returned HTTP {code}.")
    if isinstance(exc, ValueError):
        return EngineError(f"{source} returned a non-JSON response (rate limit or bot check). Try again shortly.")
    return EngineError(f"Request to {source} failed: {exc}")


def _get(session, url, params=None, timeout=TIMEOUT):
    _pace(url)
    resp = session.get(url, params=params, timeout=timeout)
    if resp.status_code in (401, 403, 404, 406, 429, 503) and session.headers.get("User-Agent") != BROWSER_UA:
        # Many boorus (and the Cloudflare rules in front of them) refuse a non-browser
        # User-Agent. Retry once looking like a normal browser before giving up.
        hdrs = {"User-Agent": BROWSER_UA, "Referer": urlparse(url).scheme + "://" + (urlparse(url).hostname or "") + "/"}
        retry = session.get(url, params=params, timeout=timeout, headers=hdrs)
        if retry.status_code < resp.status_code or retry.status_code < 400:
            resp = retry
    resp.raise_for_status()
    return resp


def _get_json(session, url, params=None, timeout=TIMEOUT, source=""):
    try:
        resp = _get(session, url, params, timeout)
        try:
            return resp.json()
        except ValueError:
            if session.headers.get("User-Agent") == BROWSER_UA:
                raise
            # an HTML bot-check instead of JSON: try again as a browser
            hdrs = {"User-Agent": BROWSER_UA, "Referer": urlparse(url).scheme + "://" + (urlparse(url).hostname or "") + "/"}
            r2 = session.get(url, params=params, timeout=timeout, headers=hdrs)
            r2.raise_for_status()
            return r2.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise _friendly(exc, source or url) from exc


def _site_root(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.hostname}/"


def _image_variants(url: str) -> list[str]:
    """Alternative spellings of an image URL that different booru frontends accept."""
    out: list[str] = []
    p = urlparse(url)
    host, path = p.hostname or "", p.path
    q = ("?" + p.query) if p.query else ""
    clean = re.sub(r"/{2,}", "/", path)
    cand_paths = [clean]
    if "/images/" in clean:
        cand_paths.append(clean.replace("/images/", "//images/", 1))
    hosts = [host]
    hosts.append(host[4:] if host.startswith("www.") else "www." + host)
    for h in hosts:
        for pp in cand_paths:
            u = f"{p.scheme}://{h}{pp}{q}"
            if u != url and u not in out:
                out.append(u)
    # same file under another extension (some sites report .jpg for png/webp originals)
    m = re.search(r"\.(jpe?g|png|webp|gif)$", clean, re.I)
    if m and "/images/" in clean and "/thumbnails/" not in clean:
        for ext in ("jpg", "png", "jpeg", "webp", "gif"):
            if ext != m.group(1).lower():
                u = f"{p.scheme}://{host}{clean[:m.start()]}.{ext}{q}"
                if u not in out:
                    out.append(u)
    return out[:8]


def open_image(url: str, referer: str = "", proxy: str = "", timeout=(8, 25), stream: bool = True):
    """GET an image, working around the usual booru refusals. Tries the app's own
    User-Agent first (what has always worked), then a browser User-Agent, then no
    Referer, then alternative spellings of the URL. Returns the first HTTP 200 response
    (caller must close it), or the last non-200 response. Network errors propagate."""
    proxies = {"http": proxy, "https": proxy} if proxy else None
    ref = referer or _site_root(url)
    attempts = [(url, USER_AGENT, ref), (url, BROWSER_UA, ref), (url, BROWSER_UA, None), (url, BROWSER_UA, _site_root(url))]
    attempts += [(u, BROWSER_UA, ref) for u in _image_variants(url)]
    last = None
    seen = set()
    for i, (u, ua, rf) in enumerate(attempts):
        key = (u, ua, rf)
        if key in seen:
            continue
        seen.add(key)
        hdrs = {"User-Agent": ua, "Accept": "image/avif,image/webp,image/*,video/*,*/*;q=0.8"}
        if rf:
            hdrs["Referer"] = rf
        if last is not None:
            try:
                last.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            last = requests.get(u, headers=hdrs, timeout=timeout, stream=stream, proxies=proxies)
        except requests.exceptions.RequestException:
            if i == 0:
                raise            # unreachable host: more attempts would only repeat the wait
            last = None
            continue
        ctype = last.headers.get("Content-Type", "")
        if last.status_code == 200 and not ctype.startswith("text/"):
            return last
        if last.status_code in (429,):
            return last
    if last is None:
        raise requests.exceptions.ConnectionError("image host unreachable")
    return last


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _split(s) -> list[str]:
    return [t for t in (s or "").split() if t]


def _empty_tags() -> dict:
    return {c: [] for c in CATEGORIES}


def _abs(base: str, url: str | None) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return base + url
    return url


def _ext(url: str, fallback: str = "") -> str:
    if fallback:
        return fallback.lower().lstrip(".")
    path = (url or "").split("?")[0]
    last = path.rsplit("/", 1)[-1]
    return last.rsplit(".", 1)[-1].lower() if "." in last else ""


_CAT_MAP = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta", 6: "meta"}


def _norm_rating(raw) -> str:
    r = str(raw or "").lower()
    return {"g": "safe", "general": "safe", "safe": "safe", "s": "safe" , "sensitive": "sensitive",
            "q": "questionable", "questionable": "questionable", "suggestive": "sensitive",
            "e": "explicit", "explicit": "explicit"}.get(r, "unknown")


def _safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------

def _rating_token(source: str, rating: str) -> str | None:
    d = SOURCES[source]
    if not rating or rating == "any":
        return None
    eng = d["engine"]
    if eng == "danbooru":
        return {"safe": "rating:g", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"}.get(rating)
    if eng in ("e621", "moebooru"):
        return {"safe": "rating:s", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"}.get(rating)
    if eng == "gelbooru":
        names = d["rating_names"]
        idx = _RATING_ORDER.index(rating) if rating in _RATING_ORDER else None
        return f"rating:{names[idx]}" if idx is not None else None
    if eng == "philomena":
        return {"safe": "safe", "sensitive": "suggestive", "questionable": "questionable", "explicit": "explicit"}.get(rating)
    return None


def _order_token(source: str, order: str) -> str | None:
    eng = SOURCES[source]["engine"]
    order = order if order in ORDERS else "latest"
    if order == "latest":
        return None
    if eng == "danbooru":
        return {"random": "order:random", "popular": "order:rank", "score": "order:score",
                "favorites": "order:favcount"}[order]
    if eng == "e621":
        return {"random": "order:random", "popular": "order:rank", "score": "order:score",
                "favorites": "order:favcount"}[order]
    if eng == "moebooru":
        return {"random": "order:random", "popular": "order:score", "score": "order:score",
                "favorites": "order:score"}[order]
    if eng == "gelbooru":
        if order == "random":
            return "sort:random" if SOURCES[source]["random_native"] else None
        return "sort:score:desc"
    return None  # philomena uses request params


def _min_score_token(source: str, min_score: int) -> str | None:
    if min_score <= 0:
        return None
    eng = SOURCES[source]["engine"]
    if eng in ("gelbooru", "moebooru", "danbooru", "e621"):
        return f"score:>={min_score}"
    if eng == "philomena":
        return f"score.gte:{min_score}"
    return None


# ---------------------------------------------------------------------
# Post normalization
# ---------------------------------------------------------------------

def _post_url(source: str, pid) -> str:
    d = SOURCES[source]
    base, eng = d["base"], d["engine"]
    if eng in ("danbooru", "e621"):
        return f"{base}/posts/{pid}"
    if eng == "gelbooru":
        return f"{base}/index.php?page=post&s=view&id={pid}"
    if eng == "moebooru":
        return f"{base}/post/show/{pid}"
    return f"{base}/images/{pid}"


def site_search_url(source: str, tags: str) -> str:
    d = SOURCES[source]
    base, eng = d["base"], d["engine"]
    q = quote((tags or "").strip())
    if eng in ("danbooru", "e621"):
        return f"{base}/posts?tags={q}"
    if eng == "gelbooru":
        return f"{base}/index.php?page=post&s=list&tags={q}"
    if eng == "moebooru":
        return f"{base}/post?tags={q}"
    q = quote((tags or "").strip().replace("_", " "))
    return f"{base}/search?q={q}"


def site_wiki_url(source: str, tag: str) -> str:
    d = SOURCES[source]
    base, eng = d["base"], d["engine"]
    if eng == "danbooru":
        return f"{base}/wiki_pages/{quote(tag)}"
    if eng == "e621":
        return f"{base}/wiki_pages/show_or_new?title={quote(tag)}"
    if eng == "moebooru":
        return f"{base}/wiki/show?title={quote(tag)}"
    if eng == "gelbooru":
        return f"{base}/index.php?page=wiki&s=list&search={quote(tag)}"
    return f"{base}/tags/{_philomena_slug(tag)}"


def _base_post(source: str, pid) -> dict:
    return {
        "id": pid, "source": source, "post_url": _post_url(source, pid),
        "file_url": "", "sample_url": "", "preview_url": "", "thumb_url": "",
        "width": None, "height": None, "rating": "unknown", "score": 0,
        "extension": "", "tags": _empty_tags(), "created_at": "", "source_url": "",
        "fav_count": None, "uploader": "", "parent_id": None, "has_children": False,
        "pool_ids": [], "md5": "", "file_size": None, "tags_categorized": False,
    }


def _norm_danbooru(source: str, p: dict) -> dict | None:
    file_url = p.get("file_url") or p.get("large_file_url")
    if not file_url:
        return None
    out = _base_post(source, p["id"])
    base = SOURCES[source]["base"]
    thumb = ""
    ma = p.get("media_asset") or {}
    variants = {v.get("type"): v.get("url") for v in (ma.get("variants") or []) if isinstance(v, dict)}
    for key in ("360x360", "720x720", "180x180"):
        if variants.get(key):
            thumb = variants[key]
            break
    preview = p.get("preview_file_url") or thumb or file_url
    out.update({
        "file_url": _abs(base, file_url),
        "sample_url": _abs(base, p.get("large_file_url") or file_url),
        "preview_url": _abs(base, preview),
        "thumb_url": _abs(base, thumb or preview),
        "width": p.get("image_width"), "height": p.get("image_height"),
        "rating": _norm_rating(p.get("rating")), "score": p.get("score", 0),
        "extension": p.get("file_ext", "") or _ext(file_url),
        "created_at": p.get("created_at", ""), "source_url": p.get("source", "") or "",
        "fav_count": p.get("fav_count"), "parent_id": p.get("parent_id"),
        "has_children": bool(p.get("has_children")), "md5": p.get("md5", ""),
        "file_size": p.get("file_size"), "tags_categorized": True,
        "pool_ids": [int(x) for x in re.findall(r"pool:(\d+)", p.get("pool_string", "") or "")],
        "tags": {
            "general": _split(p.get("tag_string_general")), "character": _split(p.get("tag_string_character")),
            "copyright": _split(p.get("tag_string_copyright")), "artist": _split(p.get("tag_string_artist")),
            "meta": _split(p.get("tag_string_meta")),
        },
    })
    return out


def _norm_e621(source: str, p: dict) -> dict | None:
    f = p.get("file") or {}
    if not f.get("url"):
        return None
    out = _base_post(source, p.get("id"))
    sample = p.get("sample") or {}
    prev = (p.get("preview") or {}).get("url") or ""
    t = p.get("tags") or {}
    rel = p.get("relationships") or {}
    srcs = p.get("sources") or []
    out.update({
        "file_url": f["url"], "sample_url": (sample.get("url") if sample.get("has") else None) or f["url"],
        "preview_url": prev or f["url"], "thumb_url": prev or f["url"],
        "width": f.get("width"), "height": f.get("height"),
        "rating": _norm_rating(p.get("rating")), "score": (p.get("score") or {}).get("total", 0),
        "extension": f.get("ext", ""), "created_at": p.get("created_at", ""),
        "source_url": srcs[0] if srcs else "", "fav_count": p.get("fav_count"),
        "parent_id": rel.get("parent_id"), "has_children": bool(rel.get("has_children")),
        "md5": f.get("md5", ""), "file_size": f.get("size"), "tags_categorized": True,
        "pool_ids": [int(x) for x in (p.get("pools") or []) if str(x).isdigit()],
        "tags": {
            "general": list(t.get("general", [])), "character": list(t.get("character", [])),
            "copyright": list(t.get("copyright", [])), "artist": list(t.get("artist", [])),
            "meta": list(t.get("meta", [])) + list(t.get("species", [])) + list(t.get("lore", [])),
        },
    })
    return out


def _norm_moebooru(source: str, p: dict) -> dict | None:
    file_url = p.get("file_url") or p.get("jpeg_url")
    if not file_url:
        return None
    out = _base_post(source, p.get("id"))
    created = p.get("created_at")
    if isinstance(created, (int, float)):
        created = time.strftime("%Y-%m-%d %H:%M", time.gmtime(created))
    out.update({
        "file_url": file_url, "sample_url": p.get("sample_url") or p.get("jpeg_url") or file_url,
        "preview_url": p.get("preview_url") or file_url, "thumb_url": p.get("preview_url") or file_url,
        "width": p.get("width"), "height": p.get("height"),
        "rating": _norm_rating(p.get("rating")), "score": p.get("score", 0),
        "extension": _ext(file_url), "created_at": created or "", "source_url": p.get("source", "") or "",
        "uploader": p.get("author", "") or "", "parent_id": p.get("parent_id"),
        "has_children": bool(p.get("has_children")), "md5": p.get("md5", ""), "file_size": p.get("file_size"),
        "tags": {**_empty_tags(), "general": _split(p.get("tags"))},
    })
    return out


def _norm_gelbooru(source: str, p: dict) -> dict | None:
    d = SOURCES[source]
    base = d["base"]
    file_url = p.get("file_url")
    directory, image, md5 = p.get("directory"), p.get("image"), p.get("hash") or p.get("md5")
    if not file_url and directory and image:
        file_url = f"{base}/images/{directory}/{image}"
    if not file_url:
        return None
    file_url = _abs(base, file_url)
    preview = p.get("preview_url")
    if not preview and directory and md5:
        preview = f"{base}/thumbnails/{directory}/thumbnail_{md5}.jpg"
    sample = p.get("sample_url")
    if not sample and str(p.get("sample")) in ("1", "true", "True") and directory and md5:
        sample = f"{base}/samples/{directory}/sample_{md5}.jpg"
    out = _base_post(source, p.get("id"))
    out.update({
        "file_url": file_url, "sample_url": _abs(base, sample) or file_url,
        "preview_url": _abs(base, preview) or file_url, "thumb_url": _abs(base, preview) or file_url,
        "width": p.get("width"), "height": p.get("height"),
        "rating": _norm_rating(p.get("rating")), "score": _safe_int(p.get("score")),
        "extension": _ext(file_url), "created_at": p.get("created_at", "") or "",
        "source_url": p.get("source", "") or "", "uploader": p.get("owner", "") or "",
        "parent_id": p.get("parent_id") if p.get("parent_id") not in (0, "0", None, "") else None,
        "has_children": str(p.get("has_children")).lower() == "true", "md5": md5 or "",
        "tags": {**_empty_tags(), "general": _split(p.get("tags"))},
    })
    return out


def _philomena_cat(tag: str) -> str:
    if tag.startswith("artist:"):
        return "artist"
    if tag.startswith("oc:"):
        return "character"
    if tag in ("safe", "suggestive", "questionable", "explicit", "semi-grimdark", "grimdark", "grotesque"):
        return "meta"
    return "general"


def _norm_philomena(source: str, p: dict) -> dict | None:
    rep = p.get("representations") or {}
    full = rep.get("full") or p.get("view_url")
    if not full:
        return None
    base = SOURCES[source]["base"]
    out = _base_post(source, p.get("id"))
    tags = _empty_tags()
    for t in p.get("tags") or []:
        t = t.replace(" ", "_")  # keep the app-wide "underscore, no spaces" tag shape
        tags[_philomena_cat(t)].append(t)
    src = p.get("source_url") or ""
    out.update({
        "file_url": _abs(base, full), "sample_url": _abs(base, rep.get("large") or full),
        "preview_url": _abs(base, rep.get("thumb") or rep.get("small") or full),
        "thumb_url": _abs(base, rep.get("thumb") or rep.get("small") or full),
        "width": p.get("width"), "height": p.get("height"),
        "rating": _rating_from_philomena_tags(p.get("tags") or []), "score": p.get("score", 0),
        "extension": p.get("format", "") or _ext(full), "created_at": p.get("created_at", ""),
        "source_url": src, "fav_count": p.get("faves"), "uploader": p.get("uploader") or "",
        "md5": p.get("sha512_hash", "")[:32], "file_size": p.get("size"),
        "tags": tags, "tags_categorized": False,
    })
    return out


def _rating_from_philomena_tags(tags: list[str]) -> str:
    s = set(tags)
    if "explicit" in s:
        return "explicit"
    if "questionable" in s:
        return "questionable"
    if "suggestive" in s:
        return "sensitive"
    if "safe" in s:
        return "safe"
    return "unknown"


_NORMALIZERS = {"danbooru": _norm_danbooru, "e621": _norm_e621, "moebooru": _norm_moebooru,
                "gelbooru": _norm_gelbooru, "philomena": _norm_philomena}


def _normalize_list(source: str, raw: list) -> list[dict]:
    fn = _NORMALIZERS[SOURCES[source]["engine"]]
    out = []
    for p in raw or []:
        if not isinstance(p, dict):
            continue
        try:
            n = fn(source, p)
        except (KeyError, TypeError, ValueError):
            n = None
        if n:
            out.append(n)
    return out


# ---------------------------------------------------------------------
# Fetching posts
# ---------------------------------------------------------------------

def _creds(source: str) -> dict:
    try:
        return settings_store.get_booru_credentials().get(source, {}) or {}
    except Exception:  # noqa: BLE001 - settings problems must never break browsing
        return {}


def _philomena_query(tags: list[str], rating_token: str | None, min_token: str | None) -> str:
    parts = [t.replace("_", " ") for t in tags]
    if rating_token:
        parts.append(rating_token)
    if min_token:
        parts.append(min_token)
    return ", ".join(parts) if parts else "*"


def _post_filter(posts: list[dict], extra_tags: list[str], min_score: int) -> list[dict]:
    if not extra_tags and min_score <= 0:
        return posts
    out = []
    for p in posts:
        all_tags = {t.lower() for ts in p["tags"].values() for t in ts}
        ok = True
        for t in extra_tags:
            t = t.strip().lower()
            if not t:
                continue
            if t.startswith("-"):
                if t[1:] in all_tags:
                    ok = False
                    break
            elif t not in all_tags:
                ok = False
                break
        if ok and min_score > 0 and _safe_int(p.get("score")) < min_score:
            ok = False
        if ok:
            out.append(p)
    return out


def _window_token(order: str) -> str:
    """A date window keeps "random" and "top score" fast on Danbooru/e621. Sorting or
    shuffling the whole site (millions of posts) is what makes those sites time out
    for anonymous users; sorting one week or six months is cheap."""
    today = datetime.date.today()
    if order == "random":
        start, end = datetime.date(2013, 1, 1), today - datetime.timedelta(days=14)
        d0 = start + datetime.timedelta(days=random.randint(0, (end - start).days))
        d1 = d0 + datetime.timedelta(days=7)
    else:
        d1, d0 = today, today - datetime.timedelta(days=180)
    return f"date:{d0.isoformat()}..{d1.isoformat()}"


def fetch_posts(source: str, tags: str = "", rating: str = "safe", order: str = "latest",
                limit: int = 40, page: int = 1, min_score: int = 0) -> tuple[list[dict], int | None, int]:
    """Return (posts, total_if_known, raw_count). raw_count is the number of
    posts the site returned before post-filtering, used for has_more."""
    d = SOURCES.get(source)
    if d is None:
        raise EngineError(f"Unknown source: {source}")
    eng = d["engine"]
    limit = max(1, min(int(limit), _max_limit(eng)))
    page = max(1, int(page))
    user_tags = [t for t in (tags or "").split() if t]
    if order == "random_page":                  # last resort "random": a random page of the popular listing
        order, page = "popular", random.randint(1, 25)
    rating_tok = None if d["sfw_only"] and rating == "any" else _rating_token(source, rating)
    order_tok = _order_token(source, order)
    if eng in ("danbooru", "e621") and not user_tags and order in ("random", "score", "favorites"):
        user_tags = [_window_token(order)]
        order_tok = "order:favcount" if order == "favorites" else "order:score"
    min_tok = _min_score_token(source, int(min_score or 0))
    session = _session()
    try:
        # sites without native random: jump to a random page
        if order == "random" and eng != "philomena" and not order_tok:
            page = random.randint(1, 60) if not user_tags else random.randint(1, 8)

        if eng == "danbooru":
            return _fetch_danbooru(session, source, d, user_tags, rating_tok, order_tok, min_tok,
                                   int(min_score or 0), limit, page)
        if eng == "e621":
            q = " ".join(user_tags + [x for x in (rating_tok, min_tok, order_tok) if x])
            data = _get_json(session, f"{d['base']}/posts.json", {"tags": q, "limit": limit, "page": page}, source=source)
            raw = data.get("posts", []) if isinstance(data, dict) else []
            return _normalize_list(source, raw), None, len(raw)
        if eng == "moebooru":
            q = " ".join(user_tags + [x for x in (rating_tok, min_tok, order_tok) if x])
            session.headers.update({"User-Agent": BROWSER_UA, "Referer": d["base"] + "/"})
            raw, last_exc = None, None
            # Moebooru forks expose the API at /post.json and/or the older /post/index.json
            for path in ("/post.json", "/post/index.json"):
                try:
                    raw = _get_json(session, f"{d['base']}{path}", {"tags": q, "limit": limit, "page": page}, source=source)
                    if isinstance(raw, list):
                        break
                except EngineError as exc:
                    last_exc = exc
                    if getattr(exc, "network", False):
                        raise          # the site itself is unreachable - the other path won't help
            if not isinstance(raw, list):
                raise last_exc or EngineError(f"{d['name']} returned an unexpected response.")
            return _normalize_list(source, raw), None, len(raw)
        if eng == "gelbooru":
            return _fetch_gelbooru(session, source, d, user_tags, rating_tok, order_tok, min_tok, limit, page, rating, min_score)
        if eng == "philomena":
            q = _philomena_query(user_tags, rating_tok, min_tok)
            params = {"q": q, "per_page": limit, "page": page}
            sf, sd = {"random": ("random", "desc"), "popular": ("wilson_score", "desc"), "score": ("score", "desc"),
                      "favorites": ("faves", "desc")}.get(order, ("first_seen_at", "desc"))
            params.update({"sf": sf, "sd": sd})
            data = _get_json(session, f"{d['base']}/api/v1/json/search/images", params, source=source)
            raw = data.get("images", []) if isinstance(data, dict) else []
            return _normalize_list(source, raw), _safe_int(data.get("total"), None) if isinstance(data, dict) else None, len(raw)
    finally:
        session.close()
    raise EngineError(f"Unhandled engine for {source}")


def _fetch_danbooru(session, source, d, user_tags, rating_tok, order_tok, min_tok, min_score, limit, page):
    base = d["base"]
    cap = d["anon_tag_limit"]
    search_tags = ([rating_tok] if rating_tok else []) + user_tags
    extra: list[str] = []
    if cap is not None and len(search_tags) > cap:
        extra = search_tags[cap:]
        search_tags = search_tags[:cap]
    # min score is always a post-filter on Danbooru (anonymous tag limit)
    post_min = min_score if cap is not None else 0
    if cap is None and min_tok:
        search_tags = search_tags + [min_tok]
    q = " ".join(search_tags + ([order_tok] if order_tok else []))

    def call(query, lim, pg):
        try:
            data = _get_json(session, f"{base}/posts.json", {"tags": query, "limit": lim, "page": pg}, source=source)
        except EngineError:
            raise
        if isinstance(data, dict) and data.get("success") is False:
            err = EngineError(data.get("message") or f"{source} search failed")
            err.kind = "api"
            raise err
        return data if isinstance(data, list) else []

    if not extra:
        # a score floor is applied locally here, so over-fetch to still fill a page
        fetch_lim = min(200, limit * 4) if post_min else limit
        try:
            raw = call(q, fetch_lim, page)
        except EngineError as exc:
            if order_tok and "tag" in str(exc).lower():
                raw = call(" ".join(search_tags), fetch_lim, random.randint(1, 40) if page == 1 else page)
            else:
                raise
        posts = _post_filter(_normalize_list(source, raw), [], post_min)[:limit]
        return posts, None, (limit if len(raw) >= fetch_lim else len(raw))

    matched: list[dict] = []
    api_page, skip, raw_total = 1, (page - 1) * limit, 0
    for _ in range(5):
        raw = call(q, 200, api_page)
        raw_total += len(raw)
        if not raw:
            break
        matched.extend(_post_filter(_normalize_list(source, raw), extra, post_min))
        if len(matched) >= skip + limit:
            break
        api_page += 1
    window = matched[skip:skip + limit]
    return window, None, (limit if len(matched) > skip + limit else len(window))


def _fetch_gelbooru(session, source, d, user_tags, rating_tok, order_tok, min_tok, limit, page, rating, min_score):
    q = " ".join(user_tags + [x for x in (rating_tok, min_tok, order_tok) if x])
    params = {"page": "dapi", "s": "post", "q": "index", "json": 1, "tags": q, "limit": limit, "pid": page - 1}
    creds = {k: v for k, v in _creds(source).items() if v} if d.get("creds") else {}
    params.update(creds)
    exc_saved: Exception | None = None
    data = None
    try:
        data = _get_json(session, d["api"], params, source=source)
    except EngineError as exc:
        exc_saved = exc
    # Gelbooru and Rule34 answer "missing authentication" without an API key; Rule34's JSON
    # also comes back empty/odd. In that case fall back to the site's public list pages.
    bad = exc_saved is not None or (isinstance(data, dict) and (data.get("success") is False or "post" not in data
                                                               and not (data.get("@attributes") or {}).get("count")))
    if d.get("creds") and (bad or (isinstance(data, dict) and not data.get("post") and not creds)):
        posts = _html_fallback(source, user_tags, rating_tok, limit, page)
        if posts:
            return _post_filter(posts, [], int(min_score or 0)), None, len(posts)
        if not creds:
            err = EngineError(f"{d['name']} now requires a free API key for searches, and its public pages didn't answer "
                              f"either. Add your key in Collect > API keys (account > Options > API Access Credentials on "
                              f"{d['base'].split('//')[1]}), or use another source.")
            err.kind = "api"
            raise err
    if exc_saved is not None:
        raise exc_saved
    raw = []
    total = None
    if isinstance(data, dict):
        raw = data.get("post", []) or []
        if isinstance(raw, dict):
            raw = [raw]
        total = _safe_int((data.get("@attributes") or {}).get("count"), None)
    elif isinstance(data, list):
        raw = data
    return _normalize_list(source, raw), total, len(raw)


_THUMB_RE = re.compile(r"/thumbnails?/(?P<dir>[^/]+(?:/[^/]+)?)/thumbnail_(?P<hash>[0-9a-f]{32})\.\w+", re.I)


def _scrape_public_list(source: str, user_tags: list[str], rating_tok: str | None, limit: int, page: int) -> list[dict]:
    """Read a Gelbooru-0.2-style site's public thumbnail page (no API key needed)."""
    from html.parser import HTMLParser

    d = SOURCES[source]
    base = d["base"]
    tags = " ".join(([rating_tok] if rating_tok else []) + list(user_tags))
    sess = _session()
    sess.headers.update({"User-Agent": BROWSER_UA, "Referer": base + "/", "Accept": "text/html,*/*;q=0.8"})
    try:
        resp = _get(sess, f"{base}/index.php", {"page": "post", "s": "list", "tags": tags or "all", "pid": (page - 1) * 42})
        html_text = resp.text
    finally:
        sess.close()

    class P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.cur, self.items = None, []

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "a":
                m = re.search(r"[?&]id=(\d+)", a.get("href", "") or "")
                if m and "s=view" in (a.get("href") or ""):
                    self.cur = {"id": int(m.group(1)), "href": _abs(base, a.get("href"))}
            elif tag == "img" and self.cur is not None:
                self.cur["src"] = a.get("data-src") or a.get("data-original") or a.get("src") or ""
                self.cur["alt"] = a.get("alt") or a.get("title") or ""

        def handle_endtag(self, tag):
            if tag == "a" and self.cur is not None:
                if self.cur.get("src"):
                    self.items.append(self.cur)
                self.cur = None

    parser = P()
    parser.feed(html_text)
    out, seen = [], set()
    for it in parser.items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        thumb = _abs(base, it["src"].split("?")[0] if "rule34" in source else it["src"])
        full = thumb
        if "/thumbnails/" in thumb:
            full = re.sub(r"/thumbnails/([^/]+(?:/[^/]+)?)/thumbnail_([0-9a-f]{32})\.\w+", r"/images/\1/\2.jpg", thumb)
        post = _base_post(source, it["id"])
        post.update({"file_url": full, "sample_url": full, "preview_url": thumb, "thumb_url": thumb,
                     "rating": "unknown", "score": 0, "extension": "jpg", "post_url": it["href"],
                     "tags": {**_empty_tags(), "general": _split(it.get("alt", "").replace(",", " "))}})
        out.append(post)
        if len(out) >= limit:
            break
    return out


def _html_fallback(source, user_tags, rating_tok, limit, page):
    try:
        return _scrape_public_list(source, user_tags, rating_tok, limit, page)
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------
# Tag categories
# ---------------------------------------------------------------------

_tag_cache: dict[tuple[str, str], dict] = {}
_tag_lock = threading.Lock()
_PHILOMENA_CAT = {"character": "character", "oc": "character", "origin": "copyright", "content-official": "general",
                  "content-fanmade": "general", "rating": "meta", "species": "meta", "spoiler": "meta",
                  "error": "meta"}


def _philomena_slug(tag: str) -> str:
    t = tag.replace("-", "-dash-").replace(" ", "+").replace(".", "-dot-").replace(":", "-colon-")
    t = t.replace("/", "-fwslash-").replace("\\", "-bwslash-")
    return quote(t, safe="+-")


def _lookup_tags(source: str, names: list[str]) -> dict[str, dict]:
    """name -> {category, post_count}. Missing names are omitted."""
    d = SOURCES[source]
    eng, base = d["engine"], d["base"]
    out: dict[str, dict] = {}
    session = _session()
    try:
        if eng == "danbooru":
            for i in range(0, len(names), 100):
                chunk = names[i:i + 100]
                data = _get_json(session, f"{base}/tags.json",
                                 {"search[name_comma]": ",".join(chunk), "limit": 100}, FAST_TIMEOUT, source)
                for t in data if isinstance(data, list) else []:
                    out[t.get("name", "")] = {"category": _CAT_MAP.get(t.get("category"), "general"),
                                              "post_count": t.get("post_count")}
        elif eng == "gelbooru":
            if source == "gelbooru":
                for i in range(0, len(names), 80):
                    chunk = names[i:i + 80]
                    params = {"page": "dapi", "s": "tag", "q": "index", "json": 1, "names": " ".join(chunk)}
                    params.update({k: v for k, v in _creds(source).items() if v})
                    data = _get_json(session, d["api"], params, FAST_TIMEOUT, source)
                    tl = data.get("tag", []) if isinstance(data, dict) else []
                    if isinstance(tl, dict):
                        tl = [tl]
                    for t in tl:
                        out[t.get("name", "")] = {"category": _CAT_MAP.get(_safe_int(t.get("type")), "general"),
                                                  "post_count": _safe_int(t.get("count"), None)}
            else:
                for n in names:
                    data = _get_json(session, d["api"], {"page": "dapi", "s": "tag", "q": "index", "json": 1, "name": n},
                                     FAST_TIMEOUT, source)
                    tl = data.get("tag", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    if isinstance(tl, dict):
                        tl = [tl]
                    for t in tl:
                        if t.get("name") == n:
                            out[n] = {"category": _CAT_MAP.get(_safe_int(t.get("type")), "general"),
                                      "post_count": _safe_int(t.get("count"), None)}
        elif eng == "e621":
            for n in names:
                data = _get_json(session, f"{base}/tags.json", {"search[name_matches]": n, "limit": 5}, FAST_TIMEOUT, source)
                for t in data if isinstance(data, list) else []:
                    if t.get("name") == n:
                        cat = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta", 6: "meta",
                               7: "meta", 8: "meta"}.get(t.get("category"), "general")
                        out[n] = {"category": cat, "post_count": t.get("post_count")}
        elif eng == "moebooru":
            for n in names:
                data = _get_json(session, f"{base}/tag.json", {"name": n, "limit": 5}, FAST_TIMEOUT, source)
                for t in data if isinstance(data, list) else []:
                    if t.get("name") == n:
                        out[n] = {"category": _CAT_MAP.get(_safe_int(t.get("type")), "general"),
                                  "post_count": _safe_int(t.get("count"), None)}
        elif eng == "philomena":
            for n in names:
                if n.startswith("artist:") or n.startswith("oc:"):
                    out[n] = {"category": _philomena_cat(n), "post_count": None}
                    continue
                spaced = n.replace("_", " ")
                data = _get_json(session, f"{base}/api/v1/json/search/tags", {"q": f'name:"{spaced}"', "per_page": 5},
                                 FAST_TIMEOUT, source)
                for t in data.get("tags", []) if isinstance(data, dict) else []:
                    if (t.get("name") or "").lower() == spaced.lower():
                        out[n] = {"category": _PHILOMENA_CAT.get(t.get("category") or "", "general"),
                                  "post_count": t.get("images")}
    except EngineError:
        pass  # categorisation is best-effort; browsing continues without it
    finally:
        session.close()
    return out


def categorize(source: str, names: list[str], budget: int = 40, seconds: float = 8.0) -> dict[str, dict]:
    """Cached category lookup for tags that came without categories. At most
    `budget` uncached tags are looked up per call, with a time limit."""
    result: dict[str, dict] = {}
    todo: list[str] = []
    with _tag_lock:
        for n in names:
            hit = _tag_cache.get((source, n))
            if hit is not None:
                result[n] = hit
            else:
                todo.append(n)
    if not todo:
        return result
    eng = SOURCES[source]["engine"]
    todo = todo[:budget if eng not in ("gelbooru",) or source != "gelbooru" else 400]
    if eng == "danbooru" or source == "gelbooru":
        chunks = [todo[i:i + 80] for i in range(0, len(todo), 80)]
    else:
        chunks = [todo[i:i + 6] for i in range(0, len(todo), 6)]
    pool = ThreadPoolExecutor(max_workers=6)
    futures = [pool.submit(_lookup_tags, source, c) for c in chunks]
    done, _ = wait(futures, timeout=seconds)
    pool.shutdown(wait=False)
    found: dict[str, dict] = {}
    for f in done:
        try:
            found.update(f.result())
        except Exception:  # noqa: BLE001
            pass
    with _tag_lock:
        for n in todo:
            if n in found:
                _tag_cache[(source, n)] = found[n]
                result[n] = found[n]
            elif any(f in done for f in futures):
                # looked up, no match: remember as general so we don't retry forever
                if len(_tag_cache) < 50000:
                    _tag_cache.setdefault((source, n), {"category": "general", "post_count": None})
    return result


def apply_categories(source: str, posts: list[dict], budget: int = 40) -> None:
    """Re-bucket tags of category-less posts using cached/looked-up categories."""
    if not posts or SOURCES[source]["engine"] in ("danbooru", "e621"):
        return
    freq: dict[str, int] = {}
    for p in posts:
        for t in p["tags"]["general"]:
            freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq, key=lambda t: -freq[t])
    cats = categorize(source, ranked, budget=budget)
    for p in posts:
        buckets = _empty_tags()
        for cat, tags in p["tags"].items():
            for t in tags:
                c = cats.get(t, {}).get("category", cat)
                buckets[c if c in buckets else "general"].append(t)
        p["tags"] = buckets
        p["tags_categorized"] = True


# ---------------------------------------------------------------------
# Explorer: feed + sidebar
# ---------------------------------------------------------------------

def _aggregate(posts: list[dict]) -> tuple[list[dict], list[dict]]:
    counts: dict[tuple[str, str], int] = {}
    chars: dict[str, dict] = {}
    for p in posts:
        for cat, tags in p["tags"].items():
            for t in tags:
                counts[(cat, t)] = counts.get((cat, t), 0) + 1
        ch = p["tags"].get("character", [])
        # posts showing exactly one character make the best "cover" for it
        for t in ch:
            e = chars.setdefault(t, {"name": t, "count": 0, "cover": p.get("thumb_url") or p.get("preview_url"),
                                      "post_id": p["id"], "solo_cover": False})
            e["count"] += 1
            if len(ch) == 1 and not e["solo_cover"]:
                e.update({"cover": p.get("thumb_url") or p.get("preview_url"), "post_id": p["id"], "solo_cover": True})
    tag_rows = [{"name": t, "category": c, "count": n} for (c, t), n in counts.items()]
    tag_rows.sort(key=lambda r: (-r["count"], r["name"]))
    char_rows = sorted(chars.values(), key=lambda r: (-r["count"], r["name"]))
    for r in char_rows:
        r.pop("solo_cover", None)
    return tag_rows[:150], char_rows[:40]


def _explore_once(source: str = "danbooru", tags: str = "", rating: str = "safe", order: str = "random",
            limit: int = 40, page: int = 1, min_score: int = 0) -> dict:
    d = source_info(source)
    posts, total, raw_count = fetch_posts(source, tags, rating, order, limit, page, min_score)
    apply_categories(source, posts)
    tag_rows, characters = _aggregate(posts)
    lim = max(1, min(int(limit), _max_limit(d["engine"])))
    return {
        "source": source, "posts": posts, "page": page, "limit": lim, "total": total,
        "has_more": raw_count >= lim and order not in ("random", "random_page"),
        "tags": tag_rows, "characters": characters,
        "query": (tags or "").strip(), "order": order, "rating": rating,
        "site_url": site_search_url(source, tags),
    }


_CHAIN = {
    "random": ["random", "random_page", "popular", "latest"],
    "score": ["score", "popular", "latest"],
    "favorites": ["favorites", "popular", "latest"],
    "popular": ["popular", "latest"],
    "latest": ["latest"],
}
_ORDER_LABEL = {"random": "Random", "random_page": "shuffled Trending posts", "popular": "Trending",
                "latest": "Latest", "score": "Top score", "favorites": "Most faved"}
_RETRY_KINDS = {"timeout", "server", "client", "json", "api", "other", ""}   # not: conn / blocked / rate


def explore(source: str = "danbooru", tags: str = "", rating: str = "safe", order: str = "random",
            limit: int = 40, page: int = 1, min_score: int = 0) -> dict:
    """One page of posts. If the requested ordering fails in a way a simpler query can
    avoid (site timeout, 5xx, rejected query) the next cheaper ordering is used and
    `order_note` says so; a site that is down/blocked fails straight away."""
    chain = _CHAIN.get(order, [order])
    first: EngineError | None = None
    failed: list[str] = []
    for i, o in enumerate(chain):
        try:
            res = _explore_once(source, tags, rating, o, limit, page, min_score)
        except EngineError as exc:
            first = first or exc
            if exc.kind not in _RETRY_KINDS or i == len(chain) - 1:
                raise first
            failed.append(o)
            continue
        if failed:
            res["order_used"] = o
            res["order_note"] = (f"{_ORDER_LABEL[failed[0]]} didn't load from {SOURCES[source]['name']} "
                                 f"(it timed out or was refused), so this shows {_ORDER_LABEL[o]} instead.")
        return res
    raise first or EngineError("No result.")


def get_post(source: str, post_id: int) -> dict:
    d = source_info(source)
    eng, base = d["engine"], d["base"]
    session = _session()
    try:
        post = None
        if eng == "danbooru":
            raw = _get_json(session, f"{base}/posts/{int(post_id)}.json", source=source)
            post = _norm_danbooru(source, raw) if isinstance(raw, dict) else None
        elif eng == "e621":
            raw = _get_json(session, f"{base}/posts/{int(post_id)}.json", source=source)
            post = _norm_e621(source, raw.get("post", {})) if isinstance(raw, dict) else None
        elif eng == "philomena":
            raw = _get_json(session, f"{base}/api/v1/json/images/{int(post_id)}", source=source)
            post = _norm_philomena(source, raw.get("image", {})) if isinstance(raw, dict) else None
        elif eng == "moebooru":
            raw = _get_json(session, f"{base}/post.json", {"tags": f"id:{int(post_id)}", "limit": 1}, source=source)
            lst = _normalize_list(source, raw if isinstance(raw, list) else [])
            post = lst[0] if lst else None
        elif eng == "gelbooru":
            params = {"page": "dapi", "s": "post", "q": "index", "json": 1, "id": int(post_id)}
            if d.get("creds"):
                params.update({k: v for k, v in _creds(source).items() if v})
            raw = _get_json(session, d["api"], params, source=source)
            lst = raw.get("post", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
            if isinstance(lst, dict):
                lst = [lst]
            norm = _normalize_list(source, lst)
            post = norm[0] if norm else None
    finally:
        session.close()
    if not post:
        raise EngineError("That post couldn't be loaded (deleted, restricted, or the site refused the request).")
    apply_categories(source, [post], budget=60)
    return post


# ---------------------------------------------------------------------
# Tag wiki
# ---------------------------------------------------------------------

_URL_RE = re.compile(r"^https?://", re.I)


def render_markup(text: str, base: str) -> str:
    """DText / Moebooru / Philomena markup -> small, safe HTML. Everything is
    HTML-escaped first; only a fixed set of tags is ever emitted."""
    if not text:
        return ""
    t = _html.escape(text.replace("\r\n", "\n"))

    def tag_link(m):
        inner = m.group(1)
        target, _, label = inner.partition("|")
        target = target.strip()
        label = (label or target).strip()
        slug = target.lower().replace(" ", "_")
        return f'<a class="bx-taglink" data-tag="{slug}" href="#">{label}</a>'

    t = re.sub(r"\[\[([^\[\]]+?)\]\]", tag_link, t)
    t = re.sub(r"\{\{([^{}]+?)\}\}", lambda m: f'<a class="bx-taglink" data-tag="{m.group(1).strip().replace(" ", "_")}" href="#">{m.group(1).strip()}</a>', t)

    def ext_link(m):
        label, url = m.group(1), m.group(2).strip("[]")
        url = _html.unescape(url)
        if url.startswith("/"):
            url = base + url
        if not _URL_RE.match(url):
            return label
        return f'<a href="{_html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{label}</a>'

    t = re.sub(r'&quot;(.+?)&quot;:(\[[^\]]+\]|\S+)', ext_link, t)
    for code, tagname in (("b", "strong"), ("i", "em"), ("u", "u"), ("s", "s")):
        t = re.sub(rf"\[{code}\](.+?)\[/{code}\]", rf"<{tagname}>\1</{tagname}>", t, flags=re.S | re.I)
    t = re.sub(r"\[/?(?:quote|expand(?:=[^\]]*)?|spoiler|code|nodtext|tn|table|thead|tbody|tr|th|td|color(?:=[^\]]*)?)\]", "", t, flags=re.I)

    blocks, cur_list = [], []

    def flush_list():
        if cur_list:
            blocks.append("<ul>" + "".join(f"<li>{x}</li>" for x in cur_list) + "</ul>")
            cur_list.clear()

    para: list[str] = []

    def flush_para():
        if para:
            blocks.append("<p>" + "<br>".join(para) + "</p>")
            para.clear()

    for line in t.split("\n"):
        s = line.strip()
        if not s:
            flush_list(); flush_para(); continue
        m = re.match(r"^h([1-6])\.\s*(.*)$", s)
        if m:
            flush_list(); flush_para(); blocks.append(f"<h5>{m.group(2)}</h5>"); continue
        m = re.match(r"^\*+\s+(.*)$", s)
        if m:
            flush_para(); cur_list.append(m.group(1)); continue
        flush_list(); para.append(s)
    flush_list(); flush_para()
    return "".join(blocks)


def _first(data, key="id"):
    return data[0] if isinstance(data, list) and data else None


def tag_wiki(source: str, tag: str) -> dict:
    d = source_info(source)
    eng, base = d["engine"], d["base"]
    tag = (tag or "").strip().lstrip("-~")
    if not tag:
        raise EngineError("No tag given.")
    out = {"source": source, "tag": tag, "category": None, "post_count": None, "title": tag.replace("_", " "),
           "body_html": "", "other_names": [], "available": ENGINE_FEATURES[eng]["wiki"],
           "wiki_url": site_wiki_url(source, tag), "search_url": site_search_url(source, tag),
           "extra": [], "related": []}
    session = _session()
    try:
        if eng == "danbooru":
            info = _lookup_tags(source, [tag]).get(tag)
            if info:
                out.update({"category": info["category"], "post_count": info["post_count"]})
            try:
                wiki = _get_json(session, f"{base}/wiki_pages/{quote(tag)}.json", source=source)
            except EngineError:
                wiki = None
            if isinstance(wiki, dict) and wiki.get("body") is not None:
                out["title"] = (wiki.get("title") or tag).replace("_", " ")
                out["body_html"] = render_markup(wiki.get("body", ""), base)
                out["other_names"] = wiki.get("other_names") or []
            if out["category"] == "artist":
                try:
                    art = _get_json(session, f"{base}/artists.json", {"search[name]": tag, "limit": 1}, FAST_TIMEOUT, source)
                    a = _first(art)
                    if a:
                        out["other_names"] = list(dict.fromkeys((out["other_names"] or []) + (a.get("other_names") or [])))
                        urls = [u.get("url") for u in (a.get("urls") or []) if isinstance(u, dict) and u.get("url")]
                        if urls:
                            out["extra"].append({"label": "Links", "links": urls[:8]})
                except EngineError:
                    pass
        elif eng == "e621":
            info = _lookup_tags(source, [tag]).get(tag)
            if info:
                out.update({"category": info["category"], "post_count": info["post_count"]})
            wiki = _get_json(session, f"{base}/wiki_pages.json", {"search[title]": tag, "limit": 1}, source=source)
            w = _first(wiki)
            if w:
                out["title"] = (w.get("title") or tag).replace("_", " ")
                out["body_html"] = render_markup(w.get("body", ""), base)
                out["other_names"] = w.get("other_names") or []
        elif eng == "moebooru":
            info = _lookup_tags(source, [tag]).get(tag)
            if info:
                out.update({"category": info["category"], "post_count": info["post_count"]})
            wiki = _get_json(session, f"{base}/wiki.json", {"query": tag, "limit": 5}, source=source)
            if isinstance(wiki, list):
                match = next((w for w in wiki if (w.get("title") or "").lower() == tag.lower()), None)
                if match:
                    out["title"] = (match.get("title") or tag).replace("_", " ")
                    out["body_html"] = render_markup(match.get("body", ""), base)
        elif eng == "philomena":
            data = _get_json(session, f"{base}/api/v1/json/search/tags", {"q": f'name:"{tag.replace("_", " ")}"', "per_page": 5}, source=source)
            for t in data.get("tags", []) if isinstance(data, dict) else []:
                if (t.get("name") or "").lower() == tag.replace("_", " ").lower():
                    out["category"] = _PHILOMENA_CAT.get(t.get("category") or "", "general")
                    out["post_count"] = t.get("images")
                    out["title"] = t.get("name") or tag
                    out["body_html"] = render_markup(t.get("description") or t.get("short_description") or "", base)
                    out["other_names"] = t.get("aliases") or []
                    if t.get("implied_tags"):
                        out["extra"].append({"label": "Implies", "tags": t["implied_tags"][:20]})
                    break
        elif eng == "gelbooru":
            info = _lookup_tags(source, [tag]).get(tag) if source == "gelbooru" else None
            if info:
                out.update({"category": info["category"], "post_count": info["post_count"]})
    finally:
        session.close()
    out["related"] = related_tags(source, tag)
    return out


# ---------------------------------------------------------------------
# Related tags
# ---------------------------------------------------------------------

def _parse_related(data) -> list[dict]:
    rows: list[dict] = []
    if isinstance(data, dict):
        for key in ("related_tags", "tags"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if isinstance(data, dict):  # moebooru: {"tag": [[name, count], ...]}
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        return rows
    for it in data:
        if isinstance(it, (list, tuple)) and it:
            rows.append({"name": str(it[0]), "category": "general", "score": it[1] if len(it) > 1 else None})
        elif isinstance(it, dict):
            t = it.get("tag") if isinstance(it.get("tag"), dict) else it
            name = t.get("name")
            if not name:
                continue
            cat = t.get("category")
            rows.append({"name": name, "category": _CAT_MAP.get(cat, cat if isinstance(cat, str) else "general"),
                         "score": it.get("cosine_similarity") or it.get("score"), "post_count": t.get("post_count")})
    return rows


def related_tags(source: str, tag: str, limit: int = 30) -> list[dict]:
    d = source_info(source)
    eng, base = d["engine"], d["base"]
    session = _session()
    try:
        data = None
        try:
            if eng == "danbooru":
                for params in ({"search[query]": tag, "limit": limit}, {"query": tag, "limit": limit}):
                    data = _get_json(session, f"{base}/related_tag.json", params, FAST_TIMEOUT, source)
                    if _parse_related(data):
                        break
            elif eng == "e621":
                data = _get_json(session, f"{base}/related_tag.json", {"query": tag}, FAST_TIMEOUT, source)
            elif eng == "moebooru":
                data = _get_json(session, f"{base}/tag/related.json", {"tags": tag}, FAST_TIMEOUT, source)
        except EngineError:
            data = None
        rows = _parse_related(data)[:limit]
    finally:
        session.close()
    if not rows:  # sites without a related-tags API: tags that co-occur in the top posts
        try:
            posts, _, _ = fetch_posts(source, tag, "safe", "score", 40, 1)
            apply_categories(source, posts, budget=40)
            freq: dict[tuple[str, str], int] = {}
            for p in posts:
                for cat, tags_ in p["tags"].items():
                    for t in tags_:
                        if t != tag:
                            freq[(cat, t)] = freq.get((cat, t), 0) + 1
            rows = [{"name": t, "category": c, "score": n / max(len(posts), 1)}
                    for (c, t), n in sorted(freq.items(), key=lambda kv: -kv[1])[:limit]]
        except EngineError:
            rows = []
    names = [r["name"] for r in rows if r.get("category", "general") == "general"]
    if names and eng in ("danbooru",):
        cats = categorize(source, names, budget=30)
        for r in rows:
            if r["name"] in cats:
                r["category"] = cats[r["name"]]["category"]
    return rows


# ---------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------

def autocomplete(source: str, query: str, limit: int = 10) -> list[dict]:
    d = source_info(source)
    eng, base = d["engine"], d["base"]
    query = (query or "").strip()
    if not query:
        return []
    limit = max(1, min(int(limit), 20))
    session = _session()
    out: list[dict] = []
    try:
        if eng == "danbooru":
            data = _get_json(session, f"{base}/autocomplete.json",
                             {"search[query]": query, "search[type]": "tag_query", "limit": limit}, FAST_TIMEOUT, source)
            for it in data if isinstance(data, list) else []:
                if isinstance(it, dict) and (it.get("value") or it.get("label")):
                    out.append({"name": it.get("value") or it.get("label"),
                                "category": _CAT_MAP.get(it.get("category"), "general"),
                                "post_count": it.get("post_count"), "antecedent": it.get("antecedent")})
        elif eng == "e621":
            data = _get_json(session, f"{base}/tags/autocomplete.json", {"search[name_matches]": query, "expiry": 7},
                             FAST_TIMEOUT, source)
            for it in data if isinstance(data, list) else []:
                if isinstance(it, dict) and it.get("name"):
                    out.append({"name": it["name"], "category": _CAT_MAP.get(it.get("category"), "general"),
                                "post_count": it.get("post_count")})
        elif eng == "moebooru":
            data = _get_json(session, f"{base}/tag.json", {"name": query, "order": "count", "limit": limit}, FAST_TIMEOUT, source)
            rows = [t for t in data if isinstance(t, dict)] if isinstance(data, list) else []
            if not rows:
                data = _get_json(session, f"{base}/tag.json", {"name_pattern": query, "order": "count", "limit": limit},
                                 FAST_TIMEOUT, source)
                rows = [t for t in data if isinstance(t, dict)] if isinstance(data, list) else []
            for t in rows:
                if t.get("name") and query.lower() in t["name"].lower():
                    out.append({"name": t["name"], "category": _CAT_MAP.get(_safe_int(t.get("type")), "general"),
                                "post_count": _safe_int(t.get("count"), None)})
        elif eng == "gelbooru":
            urls = [(f"{base}/autocomplete.php", {"q": query}), (f"{base}/public/autocomplete.php", {"q": query}),
                    (d["api"], {"page": "autocomplete2", "term": query, "type": "tag_query", "limit": limit})]
            for url, params in urls:
                try:
                    data = _get_json(session, url, params, FAST_TIMEOUT, source)
                except EngineError:
                    continue
                for it in data if isinstance(data, list) else []:
                    if not isinstance(it, dict):
                        continue
                    name = it.get("value") or it.get("label")
                    if not name:
                        continue
                    count = it.get("post_count")
                    if count is None:
                        m = re.search(r"\(([\d,]+)\)\s*$", it.get("label", "") or "")
                        count = int(m.group(1).replace(",", "")) if m else None
                    cat = it.get("type") or it.get("category")
                    out.append({"name": name, "category": _CAT_MAP.get(_safe_int(cat, -1), cat if cat in CATEGORIES else "general"),
                                "post_count": count})
                if out:
                    break
            if not out:
                params = {"page": "dapi", "s": "tag", "q": "index", "json": 1, "name_pattern": f"{query}%",
                          "orderby": "count", "order": "DESC", "limit": limit}
                if d.get("creds"):
                    params.update({k: v for k, v in _creds(source).items() if v})
                try:
                    data = _get_json(session, d["api"], params, FAST_TIMEOUT, source)
                    tl = data.get("tag", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    if isinstance(tl, dict):
                        tl = [tl]
                    for t in tl:
                        out.append({"name": t.get("name"), "category": _CAT_MAP.get(_safe_int(t.get("type")), "general"),
                                    "post_count": _safe_int(t.get("count"), None)})
                except EngineError:
                    pass
        elif eng == "philomena":
            data = _get_json(session, f"{base}/api/v1/json/search/tags", {"q": f"{query.replace('_', ' ')}*", "per_page": limit},
                             FAST_TIMEOUT, source)
            for t in data.get("tags", []) if isinstance(data, dict) else []:
                if t.get("name"):
                    out.append({"name": t["name"].replace(" ", "_"),
                                "category": _PHILOMENA_CAT.get(t.get("category") or "", "general"),
                                "post_count": t.get("images")})
    except EngineError:
        return []
    finally:
        session.close()
    out = [o for o in out if o.get("name")]
    out.sort(key=lambda o: -(o.get("post_count") or 0))
    return out[:limit]


# ---------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------

def pools(source: str, query: str = "", page: int = 1, limit: int = 20) -> list[dict]:
    d = source_info(source)
    eng, base = d["engine"], d["base"]
    if not ENGINE_FEATURES[eng]["pools"]:
        raise EngineError(f"{d['name']} has no pool API.")
    q = (query or "").strip()
    session = _session()
    try:
        if eng in ("danbooru", "e621"):
            params = {"limit": limit, "page": page, "search[order]": "post_count" if q else "updated_at"}
            if q:
                params["search[name_matches]"] = f"*{q}*"
            data = _get_json(session, f"{base}/pools.json", params, source=source)
            rows = [p for p in data if isinstance(p, dict) and not p.get("is_deleted")] if isinstance(data, list) else []
            return [{"id": p["id"], "name": (p.get("name") or "").replace("_", " "),
                     "post_count": p.get("post_count"), "description": (p.get("description") or "")[:200]} for p in rows]
        data = _get_json(session, f"{base}/pool.json", {"query": q, "page": page}, source=source)
        rows = [p for p in data if isinstance(p, dict)] if isinstance(data, list) else []
        return [{"id": p["id"], "name": (p.get("name") or "").replace("_", " "),
                 "post_count": p.get("post_count"), "description": (p.get("description") or "")[:200]} for p in rows[:limit]]
    finally:
        session.close()
