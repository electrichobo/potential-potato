# lines 1–160
from __future__ import annotations
# --- ingest: URL or local path -> cached local file ---
import os, re, hashlib
from pathlib import Path
from typing import Tuple
import requests

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def _cache_dir() -> Path:
    d = Path('aesthetic/data/cache'); d.mkdir(parents=True, exist_ok=True); return d

def is_url(s: str) -> bool:
    return bool(_URL_RE.match(s.strip()))

def _try_ytdlp(url: str, out_path: Path) -> bool:
    try:
        import yt_dlp  # optional
    except Exception:
        return False
    ydl_opts = {
        "outtmpl": str(out_path),
        "quiet": True,
        "noprogress": True,
        "format": "mp4/bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False

def _http_stream(url: str, out_path: Path, chunk: int = 1024*1024) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for b in r.iter_content(chunk_size=chunk):
                    if b: f.write(b)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False

def resolve_source(src: str) -> Tuple[str, bool]:
    p = Path(src)
    if p.exists() and p.is_file():
        return str(p.resolve()), False
    if is_url(src):
        key = _sha1(src); out = _cache_dir() / f"{key}.mp4"
        if out.exists() and out.stat().st_size > 0:
            return str(out), False
        if _try_ytdlp(src, out): return str(out), True
        if _http_stream(src, out): return str(out), True
        raise RuntimeError("Failed to download media from URL.")
    raise FileNotFoundError("Source not found. Provide a valid path or URL.")
# end 1–160
