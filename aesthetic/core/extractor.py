# -*- coding: utf-8 -*-
from __future__ import annotations
# AESTHETIC — extractor: frame index utilities (even spacing + scene-aware sampling)
from typing import List, Tuple, Optional
import numpy as np
import cv2

# ---------- helpers ----------
def total_frames(path: str) -> int:
    """
    Best-effort frame count using OpenCV metadata. Falls back to 0 if unknown.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n and n > 0:
            return n
        # Some backends don’t expose frame count; try estimating from duration * fps.
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        dur_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)  # often 0 before reading
        if fps > 0.0 and dur_ms > 0.0:
            est = int(round((dur_ms / 1000.0) * fps))
            return max(0, est)
        return 0
    finally:
        cap.release()

def _sample_scene_frames(start: int, end: int, per_scene_candidates: int,
                         jitter_frac: float, min_gap: int) -> List[int]:
    """
    Jittered, roughly-even picks inside a scene span [start, end] (inclusive).
    Enforces a minimum frame gap to reduce near-duplicates.
    """
    if end < start:
        return []
    length = end - start + 1
    if length <= 0 or per_scene_candidates <= 0:
        return []
    step = max(1, length // max(1, per_scene_candidates))
    picks: List[int] = []
    jitter = int(step * max(0.0, float(jitter_frac)))
    for i in range(per_scene_candidates):
        base = start + i * step + step // 2
        j = 0
        if jitter > 0:
            # deterministic-ish jitter using numpy for reproducibility if caller seeds np.random
            j = int(np.random.randint(-jitter, jitter + 1))
        f = max(start, min(end, base + j))
        if not picks or (f - picks[-1]) >= int(min_gap):
            picks.append(f)
    # unique + sorted
    return sorted(set(picks))

# ---------- public: evenly spaced frame indices ----------
def indices_evenly_spaced(path: str, count: int) -> List[int]:
    """
    Return `count` evenly spaced frame indices from [0, total-1].
    If the video is very short, clamps to available frames.
    """
    total = total_frames(path)
    if total <= 0:
        return []
    k = max(1, min(int(count), total))
    return np.linspace(0, total - 1, num=k, dtype=int).tolist()

# ---------- public: scene-driven sampling ----------
def indices_from_scene_spans(
    spans: List[Tuple[int, int]],
    per_scene_candidates: int,
    jitter_frac: float = 0.12,
    min_gap: int = 6,
    max_total: Optional[int] = None,
) -> List[int]:
    """
    Convert a list of scene spans [(start,end), ...] into a set of candidate frame indices
    by sampling each span with jitter & min-gap constraints.
    If max_total is provided, down-samples the global index set (evenly) to that budget.
    """
    picks: List[int] = []
    for (a, b) in spans:
        picks.extend(_sample_scene_frames(int(a), int(b), int(per_scene_candidates),
                                          float(jitter_frac), int(min_gap)))
    picks = sorted(set(int(p) for p in picks))
    if max_total is not None and len(picks) > int(max_total) and int(max_total) > 0:
        # Evenly thin to the requested budget
        k = int(max_total)
        idx = np.linspace(0, len(picks) - 1, num=k, dtype=int)
        picks = [picks[i] for i in idx]
    return picks

def indices_via_scenecut(
    path: str,
    per_scene_candidates: int = 9,
    jitter_frac: float = 0.12,
    min_gap: int = 6,
    max_total: Optional[int] = None,
) -> List[int]:
    """
    Convenience: use PySceneDetect (or the fallback in scene_candidates) to get scene spans,
    then sample indices per scene with jitter + gap control.
    Respects the global config.yaml inside scene_candidates.get_scene_spans.
    """
    # Local import to avoid hard dependency at module import time.
    try:
        from aesthetic.core.scene_candidates import get_scene_spans
    except Exception:
        # If the module isn’t available, return an empty list rather than crashing.
        return []
    spans = get_scene_spans(path)
    return indices_from_scene_spans(
        spans=spans,
        per_scene_candidates=per_scene_candidates,
        jitter_frac=jitter_frac,
        min_gap=min_gap,
        max_total=max_total,
    )
# end
