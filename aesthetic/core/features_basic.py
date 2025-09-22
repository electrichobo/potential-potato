# lines 1–200
from __future__ import annotations
# --- features_v0: cheap technical features + simple score ---
from typing import Tuple, Dict
import numpy as np
import cv2

# --- internals ---
def _luma_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    # ITU-R BT.601 luma approximation, BGR order from OpenCV
    return 0.114 * img_bgr[..., 0] + 0.587 * img_bgr[..., 1] + 0.299 * img_bgr[..., 2]

def _contrast_pctl(y: np.ndarray) -> float:
    y = np.clip(y, 0, 255)
    p1, p99 = np.percentile(y, [1, 99])
    return float(p99 - p1)

def _sharpness_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _border_cleanliness(edges: np.ndarray) -> float:
    h, w = edges.shape
    b = max(4, int(0.05 * min(h, w)))
    mask = np.zeros_like(edges, dtype=bool)
    mask[:b, :] = True; mask[-b:, :] = True; mask[:, :b] = True; mask[:, -b:] = True
    density = float(edges[mask].mean() / 255.0) if edges[mask].size else 0.0
    return float(1.0 - density)

# --- public: compute score + metrics ---
def compute_score_and_metrics(frame_bgr: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Returns (score, metrics_dict).
    Score is a simple 0..~1-ish scalar (we rescale to 0..100 later for filenames).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    y = _luma_from_bgr(frame_bgr)
    edges = cv2.Canny(frame_bgr, 80, 160)

    contrast = _contrast_pctl(y)
    sharp = _sharpness_laplacian(gray)
    border_clean = _border_cleanliness(edges)

    # lightweight score (we’ll replace with learned ranker later)
    score = 0.5 * (contrast / 255.0) + 0.4 * (sharp / (sharp + 1000.0)) + 0.1 * border_clean

    metrics = {
        "contrast": contrast,
        "sharpness": sharp,
        "border_clean": border_clean
    }
    return float(score), metrics
# end 1–200
