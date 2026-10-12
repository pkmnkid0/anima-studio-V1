"""
Generic "from the web" dataset source: either a pasted list of direct
image URLs, or a page URL to scan for <img> tags. This is the plain-
HTML-parsing equivalent of what a person would do by hand (view a page,
save the images they want) - just automated. No login walls, paywalls,
or anti-bot measures are bypassed; pages that need those simply won't
parse and are reported as such.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import requests

USER_AGENT = "AnimaLoRAStudio/1.0 (dataset collection tool; +local use)"
REQUEST_TIMEOUT = 20
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")

_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_SRC_RE = re.compile(r'(?:src|data-src|data-original)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_SRCSET_RE = re.compile(r'srcset\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


class WebScrapeError(RuntimeError):
    pass


def _looks_like_image(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(IMAGE_EXTENSIONS)


def from_url_list(urls: list[str]) -> list[dict]:
    """Treat each given URL as a direct image link. Minimal validation
    only - this never fetches page HTML."""
    results = []
    for raw in urls:
        url = raw.strip()
        if not url:
            continue
        results.append({
            "source": "url",
            "file_url": url,
            "preview_url": url,
            "id": None,
            "tags": {"general": [], "character": [], "copyright": [], "artist": [], "meta": []},
            "rating": "unknown",
        })
    return results


def scan_page(page_url: str, same_domain_only: bool = True) -> list[dict]:
    """Fetch a page and pull out candidate image URLs from <img> tags
    (src, data-src, and the highest-resolution entry in srcset)."""
    try:
        resp = requests.get(page_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise WebScrapeError(f"Couldn't fetch {page_url}: {exc}") from exc

    html = resp.text
    page_domain = urlparse(page_url).netloc
    found: dict[str, dict] = {}

    for tag in _IMG_TAG_RE.findall(html):
        candidates = []
        m = _SRC_RE.search(tag)
        if m:
            candidates.append(m.group(1))
        m = _SRCSET_RE.search(tag)
        if m:
            # srcset: "url1 1x, url2 2x" or "url1 480w, url2 960w" - take the last (usually largest)
            parts = [p.strip().split(" ")[0] for p in m.group(1).split(",") if p.strip()]
            if parts:
                candidates.append(parts[-1])

        for src in candidates:
            if not src or src.startswith("data:"):
                continue
            absolute = urljoin(page_url, src)
            if same_domain_only and urlparse(absolute).netloc != page_domain:
                continue
            if not _looks_like_image(absolute):
                continue
            found[absolute] = {
                "source": "url",
                "file_url": absolute,
                "preview_url": absolute,
                "id": None,
                "tags": {"general": [], "character": [], "copyright": [], "artist": [], "meta": []},
                "rating": "unknown",
            }

    return list(found.values())
