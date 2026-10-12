"""
Thin wrappers around each booru site's public JSON search API. These are
the same documented, public, unauthenticated endpoints each site's own
search page uses - this module just lets the app call them from server
side and normalize the results into one shape the rest of the app works
with, the same way tools like gallery-dl or Grabber do.

Nothing here bypasses a site's own content gating: the "rating" filter
is translated to each site's own rating tag syntax (the same tag you'd
type into that site's search box yourself), and defaults to the
safest/most restrictive option. Explicit/questionable content is only
ever fetched if the user explicitly widens the rating filter themselves.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import booru_engines as engines
from . import settings_store

USER_AGENT = "AnimaLoRAStudio/1.0 (dataset collection tool; +local use)"
REQUEST_TIMEOUT = 20
AUTOCOMPLETE_TIMEOUT = 8

# API credentials are optional. Some sites gate their JSON/DAPI endpoint,
# but their normal public search pages can still be read without an account.
SOURCES_REQUIRING_CREDENTIALS = set()
CREDENTIAL_SIGNUP_URL = {}

# Danbooru/e621 share this numeric tag-category scheme.
_DANBOORU_CATEGORY = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}

# Normalized rating -> per-site rating tag(s)
_RATING_MAP = {
    "danbooru": {"safe": "rating:g", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"},
    "gelbooru": {"safe": "rating:safe", "sensitive": "rating:sensitive",
                 "questionable": "rating:questionable", "explicit": "rating:explicit"},
    "safebooru": {"safe": "rating:safe", "sensitive": "rating:safe",
                  "questionable": "rating:questionable", "explicit": "rating:explicit"},
    "e621": {"safe": "rating:s", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"},
    "rule34": {"safe": "rating:safe", "sensitive": "rating:safe",
               "questionable": "rating:questionable", "explicit": "rating:explicit"},
    # moebooru sites (konachan/yandere) only have three tiers - "sensitive"
    # maps to the same tag as "safe" since there's no separate tier for it.
    "konachan": {"safe": "rating:s", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"},
    "yandere": {"safe": "rating:s", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"},
}

SOURCES = ["danbooru", "gelbooru", "safebooru", "e621", "rule34", "konachan", "yandere"] + list(engines.EXTRA_SOURCE_IDS)


class BooruError(RuntimeError):
    pass


class CredentialsRequiredError(BooruError):
    """Raised instead of making a doomed request when a source requires
    an api_key/user_id we don't have configured yet."""

    def __init__(self, source: str):
        signup = CREDENTIAL_SIGNUP_URL.get(source, "")
        super().__init__(
            f"{source.capitalize()} now requires a free account's API key & user ID for every "
            f"search (a site-wide change, not something this app can work around). "
            f"Get yours from {signup} and add it under Collect \u2192 API keys."
        )
        self.source = source


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=2, connect=2, read=2, status=2, backoff_factor=0.6,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET"}),
                  raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter); s.mount("http://", adapter)
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/json;q=0.9,*/*;q=0.8"})
    _px = engines.get_proxy()
    if _px:
        s.proxies.update({"http": _px, "https": _px})
    return s


def _credentials(source: str) -> dict:
    return settings_store.get_booru_credentials().get(source, {})


def _split_tags(field: str) -> list[str]:
    return [t for t in (field or "").split() if t]


def _normalize_danbooru(post: dict) -> dict | None:
    file_url = post.get("file_url") or post.get("large_file_url")
    if not file_url:
        return None
    rating_code = post.get("rating", "g")
    rating = {"g": "safe", "s": "sensitive", "q": "questionable", "e": "explicit"}.get(rating_code, "unknown")
    return {
        "id": post["id"],
        "source": "danbooru",
        "post_url": f"https://danbooru.donmai.us/posts/{post['id']}",
        "file_url": file_url,
        "preview_url": post.get("preview_file_url") or file_url,
        "width": post.get("image_width"),
        "height": post.get("image_height"),
        "rating": rating,
        "score": post.get("score", 0),
        "extension": post.get("file_ext", ""),
        "tags": {
            "general": _split_tags(post.get("tag_string_general", "")),
            "character": _split_tags(post.get("tag_string_character", "")),
            "copyright": _split_tags(post.get("tag_string_copyright", "")),
            "artist": _split_tags(post.get("tag_string_artist", "")),
            "meta": _split_tags(post.get("tag_string_meta", "")),
        },
    }


_GELBOORU_LIKE_DOMAINS = {
    "gelbooru": "gelbooru.com",
    "safebooru": "safebooru.org",
    "rule34": "rule34.xxx",
}


def _normalize_gelbooru_like(post: dict, source: str) -> dict | None:
    file_url = post.get("file_url")
    if not file_url:
        return None
    rating_raw = str(post.get("rating", "safe")).lower()
    rating = {"general": "safe", "safe": "safe", "sensitive": "sensitive",
              "questionable": "questionable", "explicit": "explicit"}.get(rating_raw, "unknown")
    # Gelbooru/Safebooru/Rule34 don't split tags by category in the basic
    # API - everything comes back as one space-separated string.
    all_tags = _split_tags(post.get("tags", ""))
    domain = _GELBOORU_LIKE_DOMAINS.get(source, "gelbooru.com")
    return {
        "id": post.get("id"),
        "source": source,
        "post_url": f"https://{domain}/index.php?page=post&s=view&id={post.get('id')}",
        "file_url": file_url,
        "preview_url": post.get("preview_url") or file_url,
        "width": post.get("width"),
        "height": post.get("height"),
        "rating": rating,
        "score": post.get("score", 0),
        "extension": (file_url.rsplit(".", 1)[-1] if "." in file_url else ""),
        "tags": {"general": all_tags, "character": [], "copyright": [], "artist": [], "meta": []},
    }


def _normalize_moebooru(post: dict, source: str) -> dict | None:
    """Konachan/yande.re - same underlying "moebooru" software, tags come
    back as one space-separated string with no category split, and a
    3-tier rating (s/q/e - no separate "sensitive" tier)."""
    file_url = post.get("file_url") or post.get("jpeg_url")
    if not file_url:
        return None
    rating = {"s": "safe", "q": "questionable", "e": "explicit"}.get(post.get("rating"), "unknown")
    domain = "konachan.com" if source == "konachan" else "yande.re"
    return {
        "id": post.get("id"),
        "source": source,
        "post_url": f"https://{domain}/post/show/{post.get('id')}",
        "file_url": file_url,
        "preview_url": post.get("preview_url") or file_url,
        "width": post.get("width"),
        "height": post.get("height"),
        "rating": rating,
        "score": post.get("score", 0),
        "extension": (file_url.rsplit(".", 1)[-1] if "." in file_url else ""),
        "tags": {"general": _split_tags(post.get("tags", "")), "character": [], "copyright": [], "artist": [], "meta": []},
    }


def _normalize_e621(post: dict) -> dict | None:
    file_info = post.get("file", {})
    file_url = file_info.get("url")
    if not file_url:
        return None
    rating = {"s": "safe", "q": "questionable", "e": "explicit"}.get(post.get("rating"), "unknown")
    tags = post.get("tags", {})
    return {
        "id": post.get("id"),
        "source": "e621",
        "post_url": f"https://e621.net/posts/{post.get('id')}",
        "file_url": file_url,
        "preview_url": (post.get("preview", {}) or {}).get("url") or file_url,
        "width": file_info.get("width"),
        "height": file_info.get("height"),
        "rating": rating,
        "score": (post.get("score", {}) or {}).get("total", 0),
        "extension": file_info.get("ext", ""),
        "tags": {
            "general": tags.get("general", []),
            "character": tags.get("character", []),
            "copyright": tags.get("copyright", []),
            "artist": tags.get("artist", []),
            "meta": tags.get("meta", []) + tags.get("species", []) + tags.get("lore", []),
        },
    }


_DANBOORU_ANON_TAG_LIMIT = 2
_DANBOORU_EXTRA_TAG_MAX_PAGES = 5  # cap on extra pages fetched hunting for post-filter matches


def _post_matches_extra_tags(post: dict, extra_tags: list[str]) -> bool:
    """Whether a normalized Danbooru post's own tags satisfy every tag
    beyond the 2 Danbooru's anonymous API actually accepted - this is how
    more than 2 tags get supported at all: extra tags aren't sent to the
    API (it would just reject the request), they're checked against each
    post's own returned tag list instead, which Danbooru already gives us
    for free with every result. Supports Danbooru's "-tag" exclude syntax
    the same way the API itself does."""
    all_post_tags = {t.lower() for tags in post.get("tags", {}).values() for t in tags}
    for tag in extra_tags:
        tag = tag.strip()
        if not tag:
            continue
        if tag.startswith("-"):
            if tag[1:].lower() in all_post_tags:
                return False
        elif tag.lower() not in all_post_tags:
            return False
    return True



class _SearchPageParser(HTMLParser):
    def __init__(self, base):
        super().__init__(convert_charrefs=True); self.base=base; self.cur=None; self.items=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag == "a":
            href=a.get("href","")
            m=re.search(r"[?&]id=(\d+)",href)
            if "page=post" in href and "s=view" in href and m:
                self.cur={"id":int(m.group(1)),"href":urljoin(self.base,href),"src":None,"alt":""}
        elif tag == "img" and self.cur is not None:
            self.cur["src"]=a.get("data-src") or a.get("data-original") or a.get("src")
            self.cur["alt"]=a.get("alt") or a.get("title") or ""
    def handle_endtag(self, tag):
        if tag=="a" and self.cur is not None:
            if self.cur.get("src"): self.items.append(self.cur)
            self.cur=None

def _public_search_page(source, tags, rating, limit, page):
    base={"gelbooru":"https://gelbooru.com/","rule34":"https://rule34.xxx/"}[source]
    rating_tag=_RATING_MAP[source].get(rating) if rating!="any" else None
    query=" ".join(([rating_tag] if rating_tag else []) + _split_tags(tags))
    resp=_session().get(urljoin(base,"index.php"),params={"page":"post","s":"list","tags":query,"pid":max(0,page-1)*42},headers={"User-Agent":engines.BROWSER_UA,"Referer":base},timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    parser=_SearchPageParser(base); parser.feed(resp.text)
    out=[]; seen=set()
    for item in parser.items:
        if item["id"] in seen: continue
        seen.add(item["id"]); src=urljoin(base,item["src"])
        out.append({"id":item["id"],"source":source,"post_url":item["href"],"file_url":src,"preview_url":src,
            "width":None,"height":None,"rating":("safe" if rating=="safe" else rating if rating!="any" else "unknown"),
            "score":0,"extension":src.rsplit(".",1)[-1].split("?",1)[0] if "." in src else "",
            "tags":{"general":_split_tags(item.get("alt","")),"character":[],"copyright":[],"artist":[],"meta":[]}})
        if len(out)>=limit: break
    return out

_SMART_COVERAGE = {
    "back": {"from_behind", "rear_view", "looking_back", "back", "back_view", "from_side_back"},
    "side": {"profile", "side_view", "sideways", "looking_away", "three-quarter_view", "3/4_view"},
    "full_body": {"full_body", "cowboy_shot", "wide_shot", "long_shot", "standing", "walking", "running"},
    "footwear": {"shoes", "boots", "sandals", "slippers", "sneakers", "high_heels", "heels", "loafers", "footwear", "thigh_boots", "ankle_boots"},
    "outfit": {"alternate_costume", "alternate_outfit", "different_outfit", "school_uniform", "casual", "swimsuit", "kimono", "dress", "uniform", "armor", "maid", "bikini"},
    "action": {"fighting", "holding_weapon", "sword", "gun", "jumping", "running", "sitting", "kneeling", "lying", "dancing", "walking", "playing_instrument"},
    "expression": {"smile", "open_mouth", "closed_mouth", "blush", "crying", "angry", "sad", "surprised", "expressionless", "smirk", "laughing"},
    "detail": {"close-up", "portrait", "upper_body", "face", "eyes", "hands", "feet", "hair", "jewelry", "earrings"},
}


def _smart_tags(item: dict) -> set[str]:
    return {str(t).lower().strip() for values in (item.get("tags") or {}).values() for t in (values or [])}


def _smart_coverage(item: dict) -> list[str]:
    tags = _smart_tags(item)
    out = []
    for name, needles in _SMART_COVERAGE.items():
        if tags.intersection(needles):
            out.append(name)
    # Infer full body from common framing tags even when the source omitted it.
    w, h = item.get("width"), item.get("height")
    if "full_body" not in out and w and h and h >= w * 1.35:
        out.append("full_body")
    return out


def _smart_quality(item: dict) -> float:
    tags = _smart_tags(item)
    score = 0.0
    w, h = item.get("width"), item.get("height")
    if w and h:
        pixels = w * h
        score += min(12.0, pixels / (1024 * 1024) * 3.0)
        if min(w, h) >= 700: score += 4
        if min(w, h) >= 1000: score += 3
    score += min(8.0, max(0, float(item.get("score") or 0)) / 10.0)
    if "solo" in tags: score += 8
    if "1girl" in tags or "1boy" in tags: score += 4
    if "multiple_girls" in tags or "multiple_boys" in tags or "group" in tags: score -= 8
    if "text" in tags or "watermark" in tags: score -= 5
    if "comic" in tags or "manga" in tags: score -= 4
    return score


def _smart_signature(item: dict) -> tuple[str, ...]:
    # A conservative metadata duplicate key. We do not claim pixel-level
    # duplicate detection here; this catches reposts/identical tag sets cheaply.
    tags = _smart_tags(item)
    useful = sorted(t for t in tags if t not in {"rating:safe", "rating:general"})
    return tuple(useful)


def smart_dataset(source: str, tags: str, rating: str = "safe", target: int = 100,
                  pool: int = 5000, min_resolution: int = 512, any_tags=None) -> dict:
    """Build a configurable-size character dataset from a larger booru pool.

    The selector uses metadata only: it fetches up to ``pool`` posts, removes
    obvious low-quality/duplicate candidates, then greedily maximizes coverage
    and diversity with diminishing returns. It deliberately does not impose
    hard category quotas, so 50, 100, 200, etc. all scale naturally.
    """
    if source not in SOURCES:
        raise BooruError(f"Unknown source: {source}")
    target = max(1, min(int(target), 1000))
    pool = max(target, min(int(pool), 5000))
    min_resolution = max(0, int(min_resolution))

    # Filter tags are alternatives (OR): the pool is every image that has the
    # base tags plus ANY ONE of the filters. One search per filter, merged.
    any_tags = [t for t in (any_tags or []) if t]
    queries = [f"{tags} {t}".strip() for t in any_tags] or [tags]
    per_query = max(1, -(-pool // len(queries)))   # ceil: split the pool evenly

    candidates = []
    seen = set()
    page = 1
    for query in queries:
        got = 0
        qpage = 1
        while got < per_query and len(candidates) < pool:
            batch_size = min(200, per_query - got)
            batch = search(source, query, rating=rating, limit=batch_size, page=qpage)
            if not batch:
                break
            before = got
            for item in batch:
                key = (item.get("source"), item.get("id"))
                if key in seen or not item.get("file_url"):
                    continue
                seen.add(key)
                got += 1
                w, h = item.get("width"), item.get("height")
                if min_resolution and w and h and min(w, h) < min_resolution:
                    continue
                item["smart_coverage"] = _smart_coverage(item)
                item["smart_quality"] = round(_smart_quality(item), 2)
                item["smart_artist"] = ((item.get("tags") or {}).get("artist") or [""])[0]
                candidates.append(item)
            if len(batch) < batch_size or got == before:
                break
            qpage += 1
        page = max(page, qpage)

    if not candidates:
        return {"results": [], "candidate_count": 0, "selected_count": 0,
                "coverage": {}, "coverage_available": {}, "pages_fetched": page}

    # Remove exact metadata duplicates while keeping the higher-quality item.
    unique = {}
    for item in candidates:
        sig = _smart_signature(item)
        old = unique.get(sig)
        if old is None or item["smart_quality"] > old["smart_quality"]:
            unique[sig] = item
    candidates = list(unique.values())

    selected = []
    selected_ids = set()
    coverage_counts = {k: 0 for k in _SMART_COVERAGE}
    artist_counts = {}
    tag_counts = {}
    remaining = candidates[:]

    # Greedy maximum-marginal-value selection. Coverage gets diminishing
    # returns, while rare tags/viewpoints keep contributing value.
    while remaining and len(selected) < target:
        best = None
        best_value = float("-inf")
        for item in remaining:
            if item["id"] in selected_ids:
                continue
            cov = item["smart_coverage"]
            gain = 0.0
            for c in cov:
                count = coverage_counts[c]
                gain += 24.0 / (1.0 + count * 0.65)
            tags_set = _smart_tags(item)
            rare_gain = sum(2.5 / (1.0 + tag_counts.get(t, 0)) for t in tags_set if t)
            artist = item.get("smart_artist") or ""
            artist_penalty = max(0, artist_counts.get(artist, 0) - 4) * 2.2 if artist else 0
            similarity_penalty = 0.0
            # Metadata similarity penalty: discourage selecting another image
            # with nearly the same useful tag set as an already-selected one.
            for prev in selected[-25:]:
                a, b = _smart_tags(item), _smart_tags(prev)
                if not a or not b:
                    continue
                inter = len(a & b); union = len(a | b)
                if union and inter / union >= 0.82:
                    similarity_penalty += 12.0
                    break
            value = item["smart_quality"] + gain + rare_gain - artist_penalty - similarity_penalty
            # Small deterministic bonus for filling an uncovered category.
            if any(coverage_counts[c] == 0 for c in cov):
                value += 12.0
            if value > best_value:
                best_value, best = value, item
        if best is None:
            break
        selected.append(best)
        selected_ids.add(best["id"])
        for c in best["smart_coverage"]:
            coverage_counts[c] += 1
        artist = best.get("smart_artist") or ""
        if artist: artist_counts[artist] = artist_counts.get(artist, 0) + 1
        for t in _smart_tags(best):
            tag_counts[t] = tag_counts.get(t, 0) + 1
        remaining.remove(best)

    available = {k: sum(1 for x in candidates if k in x["smart_coverage"]) for k in _SMART_COVERAGE}
    return {
        "results": selected,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "coverage": coverage_counts,
        "coverage_available": available,
        "pages_fetched": page,
    }

def search(source: str, tags: str, rating: str = "safe", limit: int = 40, page: int = 1) -> list[dict]:
    """Search a booru source. `rating` is one of safe/sensitive/questionable/
    explicit/any. Returns a list of normalized post dicts.

    Danbooru specifically only accepts 2 tags per search for anonymous
    (no API key) requests - more than that here doesn't fail, it sends
    the rating tag plus your first tag to the API (keeping rating
    enforcement server-side, since that's safety-relevant) and applies
    any further tags as a post-filter against each result's own tags,
    fetching a few extra pages as needed to still fill up to `limit`
    matches - which is why a search with more tags can take a bit longer."""
    if source not in SOURCES:
        raise BooruError(f"Unknown source: {source}")
    if source in engines.EXTRA_SOURCE_IDS:
        try:
            posts, _total, _raw = engines.fetch_posts(source, tags, rating, "latest", limit, page)
        except engines.EngineError as exc:
            raise BooruError(str(exc)) from exc
        return posts
    limit = max(1, min(int(limit), 200))
    page = max(1, int(page))

    user_tags = (tags or "").strip().split()
    rating_tag = _RATING_MAP[source].get(rating) if rating != "any" else None
    query = " ".join(([rating_tag] if rating_tag else []) + user_tags)

    session = _session()
    try:
        if source == "danbooru":
            api_tags = ([rating_tag] if rating_tag else []) + user_tags
            extra_tags = []
            if len(api_tags) > _DANBOORU_ANON_TAG_LIMIT:
                extra_tags = api_tags[_DANBOORU_ANON_TAG_LIMIT:]
                api_tags = api_tags[:_DANBOORU_ANON_TAG_LIMIT]
            api_query = " ".join(api_tags)

            if not extra_tags:
                resp = session.get(
                    "https://danbooru.donmai.us/posts.json",
                    params={"tags": api_query, "limit": limit, "page": page},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                posts = resp.json()
                if isinstance(posts, dict) and "success" in posts and not posts.get("success", True):
                    raise BooruError(posts.get("message", "Danbooru search failed"))
                return [p for p in (_normalize_danbooru(p) for p in posts) if p]

            # More than 2 tags: page through the 2-tag API query ourselves,
            # post-filtering each batch, until `limit` matches are found or
            # we hit the page cap - `page` becomes "the Nth page of
            # already-filtered results" from the caller's perspective.
            matched: list[dict] = []
            api_page = 1
            skip = (page - 1) * limit
            for _ in range(_DANBOORU_EXTRA_TAG_MAX_PAGES):
                resp = session.get(
                    "https://danbooru.donmai.us/posts.json",
                    params={"tags": api_query, "limit": 200, "page": api_page},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                raw_posts = resp.json()
                if isinstance(raw_posts, dict) and "success" in raw_posts and not raw_posts.get("success", True):
                    raise BooruError(raw_posts.get("message", "Danbooru search failed"))
                if not raw_posts:
                    break
                for raw in raw_posts:
                    normalized = _normalize_danbooru(raw)
                    if normalized and _post_matches_extra_tags(normalized, extra_tags):
                        matched.append(normalized)
                if len(matched) >= skip + limit:
                    break
                api_page += 1
            return matched[skip:skip + limit]

        if source in ("gelbooru", "safebooru"):
            if source == "gelbooru" and not _credentials("gelbooru"):
                return _public_search_page("gelbooru", tags, rating, limit, page)
            base="https://gelbooru.com/index.php" if source=="gelbooru" else "https://safebooru.org/index.php"
            params={**_credentials("gelbooru")} if source=="gelbooru" else {}
            resp=session.get(base,params={"page":"dapi","s":"post","q":"index","json":1,"tags":query,"limit":limit,"pid":page-1,**params},timeout=REQUEST_TIMEOUT)
            if resp.status_code==401 and source=="gelbooru": return _public_search_page("gelbooru",tags,rating,limit,page)
            resp.raise_for_status(); data=resp.json(); posts=data.get("post",[]) if isinstance(data,dict) else (data or [])
            if isinstance(posts,dict): posts=[posts]
            return [p for p in (_normalize_gelbooru_like(p,source) for p in posts) if p]

        if source == "rule34":
            creds=_credentials("rule34")
            if not creds.get("api_key") or not creds.get("user_id"):
                return _public_search_page("rule34",tags,rating,limit,page)
            resp=session.get("https://api.rule34.xxx/index.php",params={"page":"dapi","s":"post","q":"index","json":1,"tags":query,"limit":limit,"pid":page-1,**creds},timeout=REQUEST_TIMEOUT)
            if resp.status_code==401: return _public_search_page("rule34",tags,rating,limit,page)
            resp.raise_for_status(); data=resp.json(); posts=data if isinstance(data,list) else (data.get("post",[]) if isinstance(data,dict) else [])
            if isinstance(posts,dict): posts=[posts]
            return [p for p in (_normalize_gelbooru_like(p,"rule34") for p in posts) if p]

        if source == "e621":
            resp = session.get(
                "https://e621.net/posts.json",
                params={"tags": query, "limit": limit, "page": page},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            posts = data.get("posts", [])
            return [p for p in (_normalize_e621(p) for p in posts) if p]

        if source in ("konachan", "yandere"):
            domain = "konachan.com" if source == "konachan" else "yande.re"
            resp = session.get(
                f"https://{domain}/post.json",
                params={"tags": query, "limit": limit, "page": page},
                headers={"User-Agent": engines.BROWSER_UA, "Referer": f"https://{domain}/"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            posts = resp.json()
            return [p for p in (_normalize_moebooru(p, source) for p in posts) if p]
    except requests.exceptions.ConnectionError as exc:
        if "NameResolutionError" in str(exc) or "getaddrinfo" in str(exc):
            raise BooruError(
                f"Couldn't look up {source}'s address - this is a DNS problem, not something "
                f"wrong with Anima Studio itself. Usually either your internet is down, or "
                f"your ISP/network is blocking that specific site (common in some regions for "
                f"booru-style sites). Try switching your DNS to 8.8.8.8 (Google) or 1.1.1.1 "
                f"(Cloudflare), or open the site directly in a browser to check."
            ) from exc
        raise BooruError(
            f"Couldn't connect to {source} - check your internet connection, or a firewall/VPN "
            f"may be blocking it."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise BooruError(f"Request to {source} failed: {exc}") from exc
    except ValueError as exc:
        # The request succeeded (2xx) but the body wasn't valid JSON - most
        # often a Cloudflare/bot-check challenge page or a rate-limit
        # notice instead of the real API response.
        raise BooruError(
            f"{source} returned an unexpected (non-JSON) response - it may be rate-limiting "
            "or blocking automated requests right now. Try again in a bit, or a different source."
        ) from exc

    raise BooruError(f"Unhandled source: {source}")




_TAG_OPTION_GROUPS = {
    "outfit": {
        "dress", "uniform", "school_uniform", "swimsuit", "bikini", "kimono", "yukata",
        "maid", "armor", "casual", "pajamas", "lingerie", "bodysuit", "tracksuit",
        "sailor_uniform", "gym_uniform", "sportswear", "nurse", "waitress", "military",
        "business_suit", "wedding_dress", "gothic", "lolita", "hanfu", "qipao", "cheongsam",
        "hoodie", "jacket", "coat", "shirt", "blouse", "skirt", "shorts", "pants",
        "leggings", "leotard", "robe", "cloak", "cape", "gown", "uniform", "costume",
        "outfit", "alternate_costume", "alternate_outfit", "different_outfit",
    },
    "view": {
        "full_body", "cowboy_shot", "wide_shot", "long_shot", "upper_body", "portrait",
        "close-up", "from_behind", "rear_view", "looking_back", "back_view", "profile",
        "side_view", "sideways", "looking_away", "three-quarter_view", "3/4_view",
    },
    "footwear": {
        "shoes", "boots", "sandals", "slippers", "sneakers", "high_heels", "heels",
        "loafers", "footwear", "thigh_boots", "ankle_boots", "platform_footwear",
        "barefoot", "flip-flops",
    },
    "pose": {
        "standing", "sitting", "kneeling", "lying", "walking", "running", "jumping",
        "dancing", "fighting", "holding_weapon", "sword", "gun", "playing_instrument",
    },
    "expression": {
        "smile", "open_mouth", "closed_mouth", "blush", "crying", "angry", "sad",
        "surprised", "expressionless", "smirk", "laughing",
    },
    "details": {
        "hands", "feet", "hair", "jewelry", "earrings", "necklace", "gloves", "hat",
        "hair_ribbon", "hair_ornament", "weapon", "backpack", "bag",
    },
}

_TAG_OPTION_LABELS = {
    "outfit": "Outfits / Clothing", "view": "Framing & View", "footwear": "Footwear",
    "pose": "Pose / Action", "expression": "Expressions", "details": "Details", "other": "Other useful tags",
}

_TAG_OPTION_EXCLUDE = {
    "solo", "1girl", "1boy", "multiple_girls", "multiple_boys", "group", "simple_background",
    "white_background", "transparent_background", "rating:safe", "rating:general", "rating:sensitive",
    "rating:questionable", "rating:explicit", "text", "watermark", "signature",
}

def _tag_option_group(tag: str) -> str:
    t = tag.lower().strip()
    for group, exact in _TAG_OPTION_GROUPS.items():
        if t in exact:
            return group
    # Catch specific outfit/color variants such as red_dress, blue_school_uniform,
    # black_kimono, etc. without requiring an enormous hard-coded tag list.
    outfit_suffixes = (
        "_dress", "_uniform", "_costume", "_outfit", "_kimono", "_yukata", "_armor",
        "_swimsuit", "_bikini", "_pajamas", "_lingerie", "_bodysuit", "_tracksuit",
        "_suit", "_shirt", "_blouse", "_skirt", "_shorts", "_pants", "_leggings",
        "_jacket", "_coat", "_hoodie", "_robe", "_gown", "_cloak", "_cape", "_maid",
    )
    if t.endswith(outfit_suffixes) or t.startswith(("school_", "gym_", "sailor_", "military_")):
        return "outfit"
    return "other"


def tag_options(source: str, tags: str, rating: str = "safe", sample_pages: int = 3, per_page: int = 100) -> dict:
    """Discover useful tag filters from a character's actual search results.

    This is intentionally a small metadata sample rather than a full pool scan,
    so the UI can populate outfit/view/footwear options quickly.
    """
    if source not in SOURCES:
        raise BooruError(f"Unknown source: {source}")
    tags = (tags or "").strip()
    if not tags:
        raise BooruError("Enter a character tag first.")
    sample_pages = max(1, min(int(sample_pages), 5))
    per_page = max(20, min(int(per_page), 100))

    # For Danbooru, use the character query + rating as the actual API query.
    # We intentionally don't add discovered tags here.
    posts = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pages = range(1, sample_pages + 1)
    def fetch(page):
        try:
            return page, search(source, tags, rating=rating, limit=per_page, page=page), None
        except BooruError as exc:
            return page, [], exc
    with ThreadPoolExecutor(max_workers=sample_pages) as ex:
        futures = [ex.submit(fetch, p) for p in pages]
        for f in as_completed(futures):
            _, batch, err = f.result()
            posts.extend(batch)

    counts = {}
    for item in posts:
        for cat in ("general", "meta"):
            for tag in (item.get("tags", {}).get(cat) or []):
                tag = str(tag).lower().strip()
                if not tag or tag in _TAG_OPTION_EXCLUDE or tag in set(tags.lower().split()):
                    continue
                counts[tag] = counts.get(tag, 0) + 1

    grouped = {k: [] for k in list(_TAG_OPTION_GROUPS) + ["other"]}
    for tag, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        group = _tag_option_group(tag)
        # Avoid flooding the UI with weak singletons; specific outfit tags are
        # especially useful, so allow them at count 2+.
        if count < 2:
            continue
        grouped[group].append({"tag": tag, "count": count, "coverage": round(count / max(len(posts), 1) * 100, 1)})

    limits = {"outfit": 24, "view": 18, "footwear": 18, "pose": 18, "expression": 16, "details": 18, "other": 24}
    options = {}
    for group, items in grouped.items():
        options[group] = items[:limits[group]]
    return {"source": source, "sampled_posts": len(posts), "groups": options, "labels": _TAG_OPTION_LABELS}


def credentials_status() -> dict:
    """{"gelbooru": {"required": True, "configured": False, "signup_url": "..."}, ...}
    for every source that needs an api_key/user_id, for the settings panel."""
    configured = settings_store.get_booru_credentials()
    out = {}
    for source in SOURCES_REQUIRING_CREDENTIALS:
        creds = configured.get(source, {})
        out[source] = {
            "required": True,
            "configured": bool(creds.get("api_key") and creds.get("user_id")),
            "api_key": creds.get("api_key", ""),
            "user_id": creds.get("user_id", ""),
            "signup_url": CREDENTIAL_SIGNUP_URL.get(source, ""),
        }
    return out


def autocomplete(source: str, query: str, limit: int = 10) -> list[dict]:
    """Live tag suggestions as the user types - the same feature each
    site's own search box offers. Returns normalized
    {name, category, post_count} entries. Any parsing surprise from a
    given site's response degrades to an empty list rather than raising,
    so a shape mismatch on one site never breaks typing."""
    query = (query or "").strip()
    if not query or source not in SOURCES:
        return []
    if source in engines.EXTRA_SOURCE_IDS:
        return engines.autocomplete(source, query, limit)
    limit = max(1, min(int(limit), 20))
    session = _session()

    try:
        if source == "danbooru":
            resp = session.get(
                "https://danbooru.donmai.us/autocomplete.json",
                params={"search[query]": query, "search[type]": "tag_query", "limit": limit},
                timeout=AUTOCOMPLETE_TIMEOUT,
            )
            resp.raise_for_status()
            out = []
            for it in resp.json():
                if not isinstance(it, dict):
                    continue
                name = it.get("value") or it.get("label")
                if not name:
                    continue
                out.append({
                    "name": name,
                    "category": _DANBOORU_CATEGORY.get(it.get("category"), "general"),
                    "post_count": it.get("post_count"),
                })
            return out[:limit]

        if source == "e621":
            resp = session.get(
                "https://e621.net/tags/autocomplete.json",
                params={"search[name_matches]": query},
                timeout=AUTOCOMPLETE_TIMEOUT,
            )
            resp.raise_for_status()
            out = []
            for it in resp.json():
                if not isinstance(it, dict):
                    continue
                name = it.get("name") or it.get("value")
                if not name:
                    continue
                out.append({
                    "name": name,
                    "category": _DANBOORU_CATEGORY.get(it.get("category"), "general"),
                    "post_count": it.get("post_count"),
                })
            return out[:limit]

        if source in ("gelbooru", "safebooru", "rule34"):
            base = {
                "gelbooru": "https://gelbooru.com/index.php",
                "safebooru": "https://safebooru.org/index.php",
                "rule34": "https://api.rule34.xxx/index.php",
            }[source]
            extra = {}
            if source in SOURCES_REQUIRING_CREDENTIALS:
                creds = _credentials(source)
                if not creds.get("api_key") or not creds.get("user_id"):
                    return []  # autocomplete degrades silently - the search itself will explain why
                extra = creds
            resp = session.get(
                base,
                params={"page": "autocomplete2", "term": query, "type": "tag_query", "limit": limit, **extra},
                timeout=AUTOCOMPLETE_TIMEOUT,
            )
            resp.raise_for_status()
            out = []
            for it in resp.json():
                if not isinstance(it, dict):
                    continue
                name = it.get("value") or it.get("label")
                if not name:
                    continue
                count = it.get("post_count")
                if count is None:
                    m = re.search(r"\(([\d,]+)\)", it.get("label", ""))
                    if m:
                        try:
                            count = int(m.group(1).replace(",", ""))
                        except ValueError:
                            count = None
                out.append({"name": name, "category": "general", "post_count": count})
            return out[:limit]

        # konachan/yandere (moebooru) don't expose a comparable tag
        # autocomplete endpoint - fall through to the empty result below
        # rather than guessing at one.
    except (requests.exceptions.RequestException, ValueError):
        return []

    return []
