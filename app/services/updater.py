"""In-app updates from a GitHub repository.

Checks the repo's latest release (or the newest commit on `main`), downloads it, and replaces the app's CODE only:
your data folder, every virtual environment (.venv, */venv), downloaded models and caches are never touched, so an
update never means reinstalling anything. Before replacing anything the current code is zipped into
<data>/update_backups (the last three are kept), and files that an older version had but the new one dropped are
removed. Restart afterwards to run the new code.

Works when the app runs from source (run.bat / run.sh). A packaged .exe cannot replace itself while running, so there
it only reports that a new version exists and links to it.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath

import requests

from .. import config
from . import settings_store

APP_DIR = config.APP_DIR
VERSION_FILE = APP_DIR / "VERSION"
STATE_FILE = config.DATA_DIR / "update_state.json"
BACKUP_DIR = config.DATA_DIR / "update_backups"
API = "https://api.github.com"
MAX_ZIP_BYTES = 600 * 1024 * 1024
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
KEEP_BACKUPS = 3

# never replaced, never deleted, never backed up
PROTECTED_TOP = {"data", ".venv", "venv", ".git", ".github", "build", "dist"}
PROTECTED_PARTS = {"venv", "__pycache__", ".venv", "models", "node_modules"}


class UpdateError(RuntimeError):
    pass


_lock = threading.Lock()
_progress: dict = {"state": "idle", "percent": 0, "message": "", "error": "", "needs_restart": False}


# ------------------------------------------------------------------ small helpers

def current_version() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip() or "0.0.0"
    except OSError:
        return "0.0.0"


def is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False))


def _vtuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:4]) or (0,)


def newer(latest: str, installed: str) -> bool:
    return _vtuple(latest) > _vtuple(installed)


def settings() -> dict:
    return {"repo": settings_store.get("update_repo", "") or "",
            "channel": settings_store.get("update_channel", "release") or "release",
            "has_token": bool(settings_store.get("update_token", "")),
            "auto_check": bool(settings_store.get("update_auto_check", True))}


def save_settings(body: dict) -> dict:
    if "repo" in body:
        repo = str(body["repo"] or "").strip()
        repo = re.sub(r"^https?://github\.com/", "", repo).strip("/").removesuffix(".git")
        if repo and not REPO_RE.match(repo):
            raise UpdateError("The repository should look like owner/name, e.g. yourname/anima-studio.")
        settings_store.set("update_repo", repo)
    if body.get("channel") in ("release", "main"):
        settings_store.set("update_channel", body["channel"])
    if "token" in body and body["token"] is not None:
        settings_store.set("update_token", str(body["token"]).strip())
    if "auto_check" in body:
        settings_store.set("update_auto_check", bool(body["auto_check"]))
    return settings()


def _state() -> dict:
    try:
        d = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(d: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(d), encoding="utf-8")


def _headers(accept: str = "application/vnd.github+json") -> dict:
    h = {"Accept": accept, "User-Agent": "anima-studio-updater", "X-GitHub-Api-Version": "2022-11-28"}
    tok = settings_store.get("update_token", "")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get(url: str, **kw):
    try:
        r = requests.get(url, headers=_headers(), timeout=(8, 20), **kw)
    except requests.RequestException as exc:
        raise UpdateError(f"Could not reach GitHub: {str(exc).splitlines()[0][:140]}") from None
    if r.status_code == 404:
        raise UpdateError("Not found on GitHub. Check the repository name - and for a private repository, add an access token.")
    if r.status_code in (401, 403):
        left = r.headers.get("X-RateLimit-Remaining")
        raise UpdateError("GitHub refused the request" + (" (rate limit reached - try again later or add a token)" if left == "0"
                                                          else " - for a private repository, check the access token."))
    if r.status_code >= 400:
        raise UpdateError(f"GitHub answered {r.status_code}.")
    return r


# ------------------------------------------------------------------ checking

def check() -> dict:
    s = settings()
    out = {"installed": current_version(), "repo": s["repo"], "channel": s["channel"], "packaged": is_packaged(),
           "available": False, "latest": "", "notes": "", "url": "", "checked": time.time()}
    if not s["repo"]:
        raise UpdateError("Enter your GitHub repository first (owner/name).")
    st = _state()
    if s["channel"] == "main":
        r = _get(f"{API}/repos/{s['repo']}/commits/main").json()
        sha = str(r.get("sha") or "")
        out.update(latest=sha[:7], notes=str((r.get("commit") or {}).get("message") or "")[:2000],
                   url=r.get("html_url") or "", zip=f"{API}/repos/{s['repo']}/zipball/main", sha=sha,
                   available=bool(sha) and sha != st.get("sha"))
    else:
        r = _get(f"{API}/repos/{s['repo']}/releases/latest").json()
        tag = str(r.get("tag_name") or "")
        asset = next((a for a in r.get("assets") or [] if str(a.get("name", "")).lower().endswith(".zip")), None)
        out.update(latest=tag.lstrip("vV"), notes=str(r.get("body") or "")[:4000], url=r.get("html_url") or "",
                   zip=(asset or {}).get("url") if asset else r.get("zipball_url"), sha="",
                   asset=bool(asset), available=newer(tag, current_version()))
    settings_store.set("update_last", {"latest": out["latest"], "available": out["available"], "at": out["checked"]})
    return out


# ------------------------------------------------------------------ applying

def _protected(rel: str) -> bool:
    parts = PurePosixPath(rel).parts
    return not parts or parts[0] in PROTECTED_TOP or any(p in PROTECTED_PARTS for p in parts) or parts[-1] == ".env"


def _set(**kw) -> None:
    with _lock:
        _progress.update(kw)


def progress() -> dict:
    with _lock:
        return dict(_progress)


def start(body: dict) -> dict:
    if is_packaged():
        raise UpdateError("This is the packaged app, which cannot replace itself. Download the new version instead.")
    info = body or {}
    url = str(info.get("zip") or "")
    if not re.match(r"^https://(api\.github\.com|codeload\.github\.com|github\.com|objects\.githubusercontent\.com)/", url):
        raise UpdateError("Check for updates first.")
    with _lock:
        if _progress["state"] in ("downloading", "backup", "installing", "deps"):
            raise UpdateError("An update is already running.")
        _progress.update(state="downloading", percent=0, message="Downloading…", error="", needs_restart=False)
    threading.Thread(target=_run, args=(url, str(info.get("latest") or ""), str(info.get("sha") or "")), daemon=True, name="updater").start()
    return progress()


def _download(url: str, dest: Path) -> None:
    try:
        with requests.get(url, headers=_headers("application/octet-stream"), stream=True, timeout=(8, 30)) as r:
            if r.status_code >= 400:
                raise UpdateError(f"The download failed (GitHub answered {r.status_code}).")
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    got += len(chunk)
                    if got > MAX_ZIP_BYTES:
                        raise UpdateError("The download is larger than expected - stopped.")
                    f.write(chunk)
                    if total:
                        _set(percent=min(55, int(55 * got / total)))
    except requests.RequestException as exc:
        raise UpdateError(f"The download failed: {str(exc).splitlines()[0][:140]}") from None


def _members(zf: zipfile.ZipFile) -> list[tuple[str, zipfile.ZipInfo]]:
    """(path inside the app, entry) for every file, with a single wrapping folder (GitHub's zipball) removed."""
    names = [i for i in zf.infolist() if not i.is_dir()]
    tops = {PurePosixPath(i.filename).parts[0] for i in names if PurePosixPath(i.filename).parts}
    strip = len(tops) == 1 and not any(PurePosixPath(i.filename).name == "run.py" and len(PurePosixPath(i.filename).parts) == 1 for i in names)
    out = []
    for i in names:
        parts = PurePosixPath(i.filename).parts
        if strip:
            parts = parts[1:]
        if not parts:
            continue
        if any(p in ("..", "") or p.endswith(":") for p in parts) or PurePosixPath(i.filename).is_absolute():
            raise UpdateError("The download contains an unsafe path - stopped.")
        out.append(("/".join(parts), i))
    return out


def _backup() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKUP_DIR / f"code-{current_version()}-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(APP_DIR):
            rel_root = Path(root).relative_to(APP_DIR).as_posix()
            dirs[:] = [d for d in dirs if not _protected(f"{rel_root}/{d}".lstrip("./"))]
            for f in files:
                rel = f"{rel_root}/{f}".lstrip("./")
                if not _protected(rel) and not rel.endswith((".pyc", ".safetensors", ".ckpt", ".pt", ".onnx")):
                    z.write(Path(root) / f, rel)
    for old in sorted(BACKUP_DIR.glob("code-*.zip"))[:-KEEP_BACKUPS]:
        old.unlink(missing_ok=True)
    return path


def _run(url: str, latest: str, sha: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="anima-update-"))
    try:
        zpath = tmp / "update.zip"
        _download(url, zpath)
        try:
            zf = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile:
            raise UpdateError("The downloaded file is not a valid zip.") from None
        with zf:
            members = _members(zf)
            rels = {r for r, _ in members}
            if "run.py" not in rels or "app/__init__.py" not in rels:
                raise UpdateError("That download is not an Anima Studio release (run.py / app/ missing) - nothing was changed.")
            _set(state="backup", percent=60, message="Saving a backup of the current version…")
            _backup()
            _set(state="installing", percent=70, message="Installing the new files…")
            old_req = hashlib.sha1((APP_DIR / "requirements.txt").read_bytes()).hexdigest() if (APP_DIR / "requirements.txt").exists() else ""
            written = []
            for rel, info in members:
                if _protected(rel):
                    continue
                dest = (APP_DIR / rel).resolve()
                if APP_DIR.resolve() not in dest.parents:
                    raise UpdateError("The download contains an unsafe path - stopped.")
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmpf = dest.with_name(dest.name + ".new")
                with zf.open(info) as src, open(tmpf, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.replace(tmpf, dest)
                written.append(rel)
        # drop what the previous release had and this one no longer has
        st = _state()
        for rel in set(st.get("files") or []) - set(written):
            if not _protected(rel):
                p = (APP_DIR / rel)
                if p.is_file() and APP_DIR.resolve() in p.resolve().parents:
                    p.unlink(missing_ok=True)
        new_req = hashlib.sha1((APP_DIR / "requirements.txt").read_bytes()).hexdigest() if (APP_DIR / "requirements.txt").exists() else ""
        if new_req != old_req:
            _set(state="deps", percent=85, message="Updating the app's Python packages…")
            req = APP_DIR / "requirements.txt"
            code = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], capture_output=True, text=True, timeout=900)
            if code.returncode != 0:
                _set(error="The new files are installed, but some Python packages failed to update: " + (code.stderr or "").strip().splitlines()[-1][:200]
                     if (code.stderr or "").strip() else "Some Python packages failed to update.")
        _save_state({"sha": sha or st.get("sha", ""), "version": latest, "files": written, "at": time.time()})
        settings_store.set("update_last", {"latest": latest, "available": False, "at": time.time()})
        _set(state="done", percent=100, needs_restart=True,
             message=f"Updated to {latest or 'the latest version'}. Restart the app to use it.")
    except UpdateError as exc:
        _set(state="error", error=str(exc), message="")
    except Exception as exc:               # noqa: BLE001
        _set(state="error", error=f"Update failed: {str(exc)[:200]}", message="")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def restart() -> dict:
    """Start the app again with the same command, then exit this process."""
    if is_packaged():
        raise UpdateError("Restart the packaged app yourself.")
    cmd = [sys.executable, *sys.argv]
    helper = ("import subprocess, sys, time; time.sleep(2.5); subprocess.Popen(sys.argv[1:], cwd=%r)" % str(APP_DIR))
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = 0x00000008 | 0x00000200          # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    subprocess.Popen([sys.executable, "-c", helper, *cmd], **kw)
    threading.Timer(0.8, lambda: os._exit(0)).start()
    return {"restarting": True}
