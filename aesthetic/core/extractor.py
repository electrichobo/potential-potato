# lines 1–120
from __future__ import annotations
# --- extractor_v0: evenly spaced indices (no scene detect) ---
from typing import List
import numpy as np
import cv2

# --- helpers ---
def total_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, n)

# --- public: evenly spaced frame indices ---
def indices_evenly_spaced(path: str, count: int) -> List[int]:
    """
    Return `count` evenly spaced frame indices from [0, total-1].
    If the video is very short, clamps to available frames.
    """
    total = total_frames(path)
    if total <= 0:
        return []
    k = max(1, min(count, total))
    return np.linspace(0, total - 1, num=k, dtype=int).tolist()
# end 1–120
