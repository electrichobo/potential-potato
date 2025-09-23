# -*- coding: utf-8 -*-
"""
Scene boundary discovery utilities.

- Prefers PySceneDetect via the modern `open_video` API (avoids VideoManager deprecation).
- Falls back to a lightweight HSV-histogram content-change detector when PySceneDetect
  is unavailable or returns no spans.
- Returns both spans and diagnostics for the GUI/pipeline.

This module intentionally keeps helpers small and well-documented.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOG = logging.getLogger("aesthetic.scene_candidates")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SceneDiagnostics:
    method: str
    threshold: float
    downscale: int
    fallback_used: bool
    total_scenes: int
    spans: List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clean_breaks(breaks: Sequence[int], min_len: int) -> List[Tuple[int, int]]:
    """Convert scene break indices into inclusive spans with a minimum length."""
    if not breaks:
        return [(0, 0)]
    ordered = sorted(int(b) for b in breaks)
    if ordered[0] != 0:
        ordered.insert(0, 0)
    spans: List[Tuple[int, int]] = []
    for start, end in zip(ordered[:-1], ordered[1:]):
        a, b = int(start), int(end) - 1
        if b < a:
            continue
        if (b - a + 1) >= min_len:
            spans.append((a, b))
    if not spans:
        spans = [(0, max(0, ordered[-1] - 1))]
    return spans


def _fallback_hsv_detector(
    cap: cv2.VideoCapture, threshold: float, downscale: int, min_len: int
) -> List[Tuple[int, int]]:
    """Histogram-based detector used when PySceneDetect is unavailable."""
    LOG.info("scene detect: using HSV histogram fallback")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_w = max(64, width // max(1, downscale)) if width > 0 else 320

    def resize(frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] <= target_w:
            return frame
        scale = target_w / float(frame.shape[1])
        return cv2.resize(
            frame,
            (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    ok, prev = cap.read()
    if not ok or prev is None:
        return [(0, 0)]
    prev_small = resize(prev)
    prev_hist = cv2.calcHist(
        [cv2.cvtColor(prev_small, cv2.COLOR_BGR2HSV)],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )
    prev_hist = cv2.normalize(prev_hist, None).flatten()

    breaks = [0]
    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        small = resize(frame)
        hist = cv2.calcHist(
            [cv2.cvtColor(small, cv2.COLOR_BGR2HSV)],
            [0, 1],
            None,
            [32, 32],
            [0, 180, 0, 256],
        )
        hist = cv2.normalize(hist, None).flatten()
        diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) * 100.0
        if diff >= threshold and (idx - breaks[-1]) >= min_len:
            breaks.append(idx)
            prev_hist = hist
        idx += 1

    breaks.append(idx)
    return _clean_breaks(breaks, min_len)


# ---------------------------------------------------------------------------
# PySceneDetect path (modern API; no VideoManager)
# ---------------------------------------------------------------------------

def _pyscenedetect(
    cap: cv2.VideoCapture,
    source: str,
    method: str,
    threshold: float,
    downscale: int,
    min_len: int,
) -> Optional[List[Tuple[int, int]]]:
    """
    Run PySceneDetect if available; returns None when not usable.

    Uses `scenedetect.open_video(...)` to avoid the deprecated VideoManager API.
    """
    try:
        # Modern API (PySceneDetect ≥ 0.6): open_video + SceneManager
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except Exception as exc:  # pragma: no cover - optional dependency
        LOG.info("scene detect: PySceneDetect unavailable: %s", exc)
        return None

    try:
        video = open_video(source, backend="opencv")
        # Downscale: PySceneDetect handles this via options on the VideoStream;
        # when not exposed, we just proceed (threshold typically dominates).
        manager = SceneManager()
        detector = (
            AdaptiveDetector(adaptive_threshold=threshold)
            if method == "adaptive"
            else ContentDetector(threshold=threshold)
        )
        manager.add_detector(detector)
        manager.detect_scenes(video=video)
        scene_list = manager.get_scene_list()
    except Exception as exc:
        LOG.info("scene detect: PySceneDetect failed (%s); falling back", exc)
        return []

    breaks: List[int] = [0]
    for start, end in scene_list:
        # start/end are Timecodes; keep frame indices for downstream consistency
        try:
            s = int(start.get_frames())
            e = int(end.get_frames())
        except Exception:
            # Older versions may expose frames via properties; fail soft.
            continue
        if e - s < min_len:
            continue
        breaks.extend([s, e])

    if len(breaks) <= 1:
        return []
    breaks = sorted(set(breaks))
    return _clean_breaks(breaks, min_len)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scenes(source: str, cfg: Dict[str, Any]) -> SceneDiagnostics:
    scenes_cfg = (cfg.get("scenes") or {})
    method = str(scenes_cfg.get("method", "content")).lower()
    threshold = float(scenes_cfg.get("threshold", 27.0))
    downscale = int(scenes_cfg.get("downscale", 2))
    min_len = int((cfg.get("extract") or {}).get("min_scene_len_frames", 12))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        LOG.warning("scene detect: could not open source %s", source)
        return SceneDiagnostics(
            method="unopened",
            threshold=threshold,
            downscale=downscale,
            fallback_used=True,
            total_scenes=1,
            spans=[(0, 0)],
        )

    try:
        spans = _pyscenedetect(cap, source, method, threshold, downscale, min_len)
        fallback_used = spans is None or len(spans) == 0
        if not spans:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            spans = _fallback_hsv_detector(cap, threshold, downscale, min_len)
            fallback_used = True
        return SceneDiagnostics(
            method="pyscenedetect" if not fallback_used else "histogram",
            threshold=threshold,
            downscale=downscale,
            fallback_used=fallback_used,
            total_scenes=len(spans),
            spans=spans,
        )
    finally:
        cap.release()


def spans_to_metadata(spans: Iterable[Tuple[int, int]]) -> List[Dict[str, int]]:
    return [
        {"start": int(start), "end": int(end), "length": int(max(0, end - start + 1))}
        for start, end in spans
    ]


__all__ = ["SceneDiagnostics", "detect_scenes", "spans_to_metadata"]
