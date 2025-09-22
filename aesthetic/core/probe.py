# lines 1–120
from __future__ import annotations
# --- probe: basic metadata via ffprobe or OpenCV fallback ---
import json, shutil, subprocess
from typing import Dict, Any, Optional
import cv2

def _run_ffprobe(path: str, ffprobe_bin: Optional[str]) -> Optional[Dict[str, Any]]:
    if not ffprobe_bin: return None
    try:
        out = subprocess.check_output(
            [ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
            stderr=subprocess.STDOUT
        )
        return json.loads(out.decode("utf-8"))
    except Exception:
        return None

def probe_media(path: str, ffprobe_bin: Optional[str] = None) -> Dict[str, Any]:
    meta = _run_ffprobe(path, ffprobe_bin)
    if meta:
        streams = meta.get("streams", [])
        vstreams = [s for s in streams if s.get("codec_type") == "video"]
        width = vstreams[0].get("width") if vstreams else None
        height = vstreams[0].get("height") if vstreams else None
        fps = None
        if vstreams and "avg_frame_rate" in vstreams[0]:
            num, den = vstreams[0]["avg_frame_rate"].split("/")
            fps = float(num) / float(den) if float(den) != 0 else None
        return {"width": width, "height": height, "fps": fps, "raw": meta}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"width": None, "height": None, "fps": None, "raw": None}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"width": w, "height": h, "fps": fps, "frames": count, "raw": None}
# end 1–120
