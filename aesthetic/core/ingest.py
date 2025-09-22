# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
from urllib.parse import urlparse
import tempfile
import shutil
import urllib.request
import logging
from typing import Optional

LOG = logging.getLogger("aesthetic.ingest")

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
URL_SCHEMES = ("http", "https", "file")

def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return bool(u.scheme) and u.scheme.lower() in URL_SCHEMES
    except Exception:
        return False

def _looks_like_video_path(path: str) -> bool:
    lp = path.lower()
    # allow querystrings after an extension, e.g. .mp4?sig=...
    for ext in VIDEO_EXTS:
        if ext in lp:
            # ensure it ends with ext or ext followed by query/fragment
            if lp.endswith(ext) or f"{ext}?" in lp or f"{ext}#" in lp:
                return True
    return False

def _tempfile_with_ext(ext: str = ".mp4") -> str:
    ext = ext if ext.startswith(".") else f".{ext}"
    fd, tmp_path = tempfile.mkstemp(prefix="aesthetic_", suffix=ext)
    os.close(fd)
    return tmp_path

def _guess_ext_from_headers(url: str, headers: dict) -> str:
    ctype = (headers.get("Content-Type") or headers.get("content-type") or "").lower()
    if "mp4" in ctype: return ".mp4"
    if "quicktime" in ctype or "mov" in ctype: return ".mov"
    if "matroska" in ctype or "mkv" in ctype: return ".mkv"
    if "webm" in ctype: return ".webm"
    if "x-msvideo" in ctype or "avi" in ctype: return ".avi"
    # fallback: sniff from URL path
    up = urlparse(url)
    for ext in VIDEO_EXTS:
        if up.path.lower().endswith(ext):
            return ext
    return ".mp4"

def _download_to_tempfile(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AESTHETIC/0.3 (+https://localhost) Python-urllib",
            "Accept": "*/*",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ext = _guess_ext_from_headers(url, dict(resp.headers))
            tmp_path = _tempfile_with_ext(ext)
            LOG.info("ingest: downloading %s -> %s", url, tmp_path)
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            return tmp_path
    except Exception:
        # do not leave partials lying around
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

def _resolve_local_path(p: str, project_root: Optional[Path]) -> Path:
    # expand ~ and env vars
    p = os.path.expanduser(os.path.expandvars(p))
    path = Path(p)
    if not path.is_absolute() and project_root is not None:
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    return path

def resolve_source(source: str, config: Optional[dict] = None) -> str:
    """
    Return a cv2.VideoCapture-friendly string for the given source.
      - http(s) video URLs (even with querystrings) are returned directly
      - file:// URLs are resolved to local filesystem paths
      - other http(s) URLs are downloaded to a temp file
      - local paths are expanded, resolved (relative to project root), and validated

    Raises:
        FileNotFoundError if a local path cannot be found.
        URLError/HTTPError on network failures.
    """
    cfg = config or {}
    # determine project root (…/aesthetic/core -> project)
    try:
        project_root = Path(__file__).resolve().parents[2]
    except Exception:
        project_root = None

    if _is_url(source):
        u = urlparse(source)
        scheme = u.scheme.lower()
        if scheme == "file":
            local_p = _resolve_local_path(u.path, project_root)
            if not local_p.exists():
                raise FileNotFoundError(f"file URL not found: {source}")
            LOG.info("ingest: using file URL path %s", str(local_p))
            return str(local_p)
        # http/https
        if _looks_like_video_path(u.path):
            LOG.info("ingest: using stream URL %s", source)
            return source
        # unknown resource type → download then open
        return _download_to_tempfile(source)

    # local filesystem path
    path = _resolve_local_path(source, project_root)
    if not path.exists():
        raise FileNotFoundError(f"source not found: {source}")
    return str(path)
