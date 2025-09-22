# -*- coding: utf-8 -*-
from __future__ import annotations
# AESTHETIC — core.features_basic: cheap technical features + fused score (robust)
from typing import Tuple, Dict
import numpy as np
import cv2

__all__ = [
    "compute_score_and_metrics",
    "cheap_feature_vector",
]

# ------------------ low-level helpers ------------------
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _luma_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma approximation, BGR order (OpenCV)."""
    img = np.asarray(img_bgr, dtype=np.float32)
    return 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]

def _contrast_pctl(y: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> float:
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, 0, 255)
    p1, p99 = np.percentile(y, [lo, hi])
    return float(max(0.0, p99 - p1))

def _sharpness_laplacian(gray: np.ndarray) -> float:
    gray = np.asarray(gray, dtype=np.uint8)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())

def _border_cleanliness(edges: np.ndarray) -> float:
    """Penalize clutter touching the image border."""
    h, w = edges.shape[:2]
    b = max(4, int(0.05 * min(h, w)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:b, :] = True; mask[-b:, :] = True; mask[:, :b] = True; mask[:, -b:] = True
    if mask.sum() == 0:
        return 1.0
    density = float((edges[mask] > 0).mean())
    return float(1.0 - density)

def _colorfulness_bv(img_bgr: np.ndarray) -> float:
    """
    Hasler & Süsstrunk colorfulness (approx):
    M = sqrt(σ_rg^2 + σ_yb^2) + 0.3 * sqrt(μ_rg^2 + μ_yb^2)
    """
    img = np.asarray(img_bgr, dtype=np.float32)
    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    sr, sy = np.std(rg), np.std(yb)
    mr, my = np.mean(rg), np.mean(yb)
    return float(np.sqrt(sr * sr + sy * sy) + 0.3 * np.sqrt(mr * mr + my * my))

def _saturation_mean(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[..., 1]))  # 0..255

def _exposure_stats(y: np.ndarray) -> Dict[str, float]:
    """
    Basic exposure / clipping stats on luma (0..255):
      - mean_luma
      - frac_low_clip (<= 2)
      - frac_high_clip (>= 253)
      - mid_deviation (|mean - 128| / 128)
    """
    y = np.clip(np.asarray(y, dtype=np.float32), 0, 255)
    n = float(y.size) if y.size else 1.0
    mean = float(np.mean(y)) if y.size else 128.0
    frac_low = float(np.mean(y <= 2.0)) if y.size else 0.0
    frac_high = float(np.mean(y >= 253.0)) if y.size else 0.0
    mid_dev = abs(mean - 128.0) / 128.0  # 0..~1
    return {
        "mean_luma": mean,
        "frac_low_clip": frac_low,
        "frac_high_clip": frac_high,
        "mid_deviation": float(mid_dev),
    }

# ------------------ public: cheap feature vector ------------------
def cheap_feature_vector(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a small technical feature vector used elsewhere:
      [mean_gray, std_gray, laplacian_var, saturation_mean]
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    lap = _sharpness_laplacian(gray)
    sat = _saturation_mean(frame_bgr)
    return np.asarray([mean, std, lap, sat], dtype=np.float32)

# ------------------ public: compute score + metrics ------------------
def compute_score_and_metrics(frame_bgr: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Returns (score, metrics_dict).
    Score is a robust 0..1 scalar (later you can scale to 0..100 for filenames).
    Metrics include the original keys ("contrast", "sharpness", "border_clean")
    plus additional diagnostics for debugging/plots.
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[-1] != 3:
        # defensive: return neutral values
        return 0.5, {
            "contrast": 0.0, "contrast_n": 0.0,
            "sharpness": 0.0, "sharpness_n": 0.0,
            "border_clean": 0.5,
            "colorfulness": 0.0, "colorfulness_n": 0.0,
            "saturation": 0.0, "saturation_n": 0.0,
            "mean_luma": 128.0,
            "frac_low_clip": 0.0,
            "frac_high_clip": 0.0,
            "exposure_good": 1.0,
        }

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    y = _luma_from_bgr(frame_bgr)

    # Canny with mild adaptivity keeps cost low and results stable
    med = float(np.median(gray))
    lo = int(max(20, 0.66 * med))
    hi = int(min(255, 1.33 * med + 40))
    edges = cv2.Canny(gray, lo, hi)

    # --- raw metrics (your original trio retained) ---
    contrast = _contrast_pctl(y)                 # 0..255
    sharp = _sharpness_laplacian(gray)           # unbounded, typical < ~5e3
    border_clean = _border_cleanliness(edges)    # 0..1 (higher is cleaner)

    # --- additional raw metrics ---
    colorfulness = _colorfulness_bv(frame_bgr)   # ~0..>50 scene dependent
    saturation = _saturation_mean(frame_bgr)     # 0..255
    expo = _exposure_stats(y)

    # --- normalized components (0..1-ish, monotone) ---
    contrast_n = contrast / (contrast + 80.0)          # 80–120 is good
    sharp_n    = sharp / (sharp + 1200.0)              # 800–1500 is good
    color_n    = colorfulness / (colorfulness + 20.0)  # 10–40 is good
    sat_raw    = saturation / 255.0
    sat_n      = _clip01(1.0 - 2.0 * abs(sat_raw - 0.5))  # prefer mid saturation

    # Exposure goodness: penalize clipping and big mid-level shifts
    clip_pen = 0.5 * (expo["frac_low_clip"] + expo["frac_high_clip"])
    mid_pen  = min(1.0, expo["mid_deviation"])
    exposure_good = _clip01(1.0 - (0.6 * clip_pen + 0.4 * mid_pen))

    # --- fused score ---
    w_contrast = 0.30
    w_sharp    = 0.30
    w_expo     = 0.20
    w_color    = 0.10
    w_border   = 0.05
    w_sat      = 0.05

    score = (
        w_contrast * contrast_n
        + w_sharp  * sharp_n
        + w_expo   * exposure_good
        + w_color  * color_n
        + w_border * border_clean
        + w_sat    * sat_n
    )
    if not np.isfinite(score):
        score = 0.5
    score = _clip01(float(score))

    metrics: Dict[str, float] = {
        # original keys (kept for compatibility)
        "contrast": float(contrast),
        "sharpness": float(sharp),
        "border_clean": float(border_clean),
        # extras
        "colorfulness": float(colorfulness),
        "saturation": float(saturation),
        "mean_luma": float(expo["mean_luma"]),
        "frac_low_clip": float(expo["frac_low_clip"]),
        "frac_high_clip": float(expo["frac_high_clip"]),
        # normalized/derived
        "contrast_n": float(_clip01(contrast_n)),
        "sharpness_n": float(_clip01(sharp_n)),
        "colorfulness_n": float(_clip01(color_n)),
        "saturation_n": float(_clip01(sat_n)),
        "exposure_good": float(exposure_good),
    }
    return score, metrics
