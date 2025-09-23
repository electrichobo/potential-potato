# -*- coding: utf-8 -*-
"""
Phase 1 technical metrics built with OpenCV primitives.

This module returns a deterministic per-frame technical score along with a
compact 8D feature vector for fast selection. Everything runs on CPU and
sticks to simple, robust measurements so the rest of the pipeline stays
responsive.

Public API:
    - TechnicalResult (dataclass)
    - cheap_feature_vector(frame_bgr) -> np.ndarray(8,)
    - compute_score_and_metrics(frame_bgr) -> TechnicalResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TechnicalResult:
    """Container for the technical pillar output of a single frame."""
    score: float                       # scalar 0..1
    metrics: Dict[str, float]          # named metrics (0..1 where sensible)
    feature_vector: np.ndarray         # 8D float32 vector (cheap, fast)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clip_unit(value: float) -> float:
    """Clamp to [0,1] as float."""
    return float(max(0.0, min(1.0, value)))


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _grey_world_kelvin(bgr: np.ndarray) -> Tuple[float, float]:
    """
    Very rough white-balance proxy via grey-world assumption.
    Returns (r_over_g, b_over_g) normalized to ~0..2.
    """
    b, g, r = [float(np.mean(c)) for c in cv2.split(bgr.astype(np.float32))]
    eps = 1e-6
    r_over_g = r / (g + eps)
    b_over_g = b / (g + eps)
    # Normalise a bit toward 0..1 range around 1.0
    return (_clip_unit((r_over_g - 0.5) / 1.5), _clip_unit((b_over_g - 0.5) / 1.5))


def _banding_proxy(gray: np.ndarray) -> float:
    """
    Low-precision histogram entropy as a banding/posterization proxy.
    Higher entropy -> smoother gradients (less banding).
    """
    # Reduce to 5-bit to emphasise banding in near-quantised ranges
    g5 = (gray.astype(np.uint8) >> 3)  # 0..31
    hist = cv2.calcHist([g5], [0], None, [32], [0, 32]).flatten().astype(np.float32)
    hist /= max(1.0, hist.sum())
    entropy = -float(np.sum(hist * np.log2(hist + 1e-8)))
    # max entropy for 32 bins == log2(32) = 5
    return _clip_unit(entropy / 5.0)


def _palette_stats(hsv: np.ndarray) -> Tuple[float, float]:
    """
    Palette diversity based on coarse hue histogram.
    Returns (flatness, entropy_norm) where:
      - flatness high when one bin dominates (i.e., low diversity)
      - entropy_norm is 0..1
    """
    hue = hsv[..., 0].astype(np.float32)
    hist = cv2.calcHist([hue], [0], None, [18], [0, 180]).astype(np.float32)
    hist = hist / max(1.0, hist.sum())
    top_bin = float(hist.max())
    entropy = -float(np.sum(hist * np.log2(hist + 1e-8)))
    entropy_norm = _clip_unit(entropy / np.log2(18))
    return float(top_bin), float(entropy_norm)


def _colorfulness(bgr: np.ndarray) -> float:
    """
    Hasler–Süsstrunk colorfulness proxy, normalised to ~0..1.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    mean_rg, mean_yb = float(np.mean(rg)), float(np.mean(yb))
    std_rg, std_yb = float(np.std(rg)), float(np.std(yb))
    cf = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    # Squash roughly into 0..1 for typical content
    return _clip_unit(cf / 200.0)


def _center_focus_ratio(gray: np.ndarray) -> float:
    """
    Sharpness bias toward the center: Laplacian variance centre / whole.
    """
    h, w = gray.shape[:2]
    ch, cw = int(h * 0.4), int(w * 0.4)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    center = gray[y0:y0 + ch, x0:x0 + cw]
    lap_all = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_ctr = float(cv2.Laplacian(center, cv2.CV_64F).var())
    if lap_all <= 1e-6:
        return 0.0
    ratio = lap_ctr / lap_all
    # Normalise: 0..2 => 0..1
    return _clip_unit(ratio / 2.0)


def _border_edge_density(gray: np.ndarray) -> float:
    """
    Fraction of edges along borders (lower is better -> cleaner frame edges).
    We return 'cleanliness' = 1 - density.
    """
    edges = cv2.Canny(gray, 80, 160)
    h, w = gray.shape[:2]
    bw = max(2, int(min(h, w) * 0.04))
    mask = np.zeros_like(edges, dtype=np.uint8)
    mask[:bw, :] = 1
    mask[-bw:, :] = 1
    mask[:, :bw] = 1
    mask[:, -bw:] = 1
    border_edges = float((edges * mask > 0).sum())
    total_border = float(mask.sum())
    density = (border_edges / total_border) if total_border > 0 else 0.0
    return _clip_unit(1.0 - density * 4.0)  # amplify a bit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cheap_feature_vector(
    frame_bgr: np.ndarray,
    *,
    gray: Optional[np.ndarray] = None,
    hsv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return the fast 8D feature vector used for cheap technical screening:
        [avg_luma, contrast, sharpness, sat_mean, sat_std,
         colorfulness, border_clean, center_focus]
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return np.zeros((8,), dtype=np.float32)

    frame = np.asarray(frame_bgr, dtype=np.uint8)
    g = gray if gray is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = hsv if hsv is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    avg_luma = float(g.mean()) / 255.0
    contrast = float(np.std(g)) / 128.0  # 0..~2 -> clamp later
    sharpness = float(cv2.Laplacian(g, cv2.CV_64F).var()) / 1000.0
    sat = h[..., 1].astype(np.float32)
    sat_mean = float(sat.mean()) / 255.0
    sat_std = float(np.std(sat)) / 128.0
    cf = _colorfulness(frame)
    border_clean = _border_edge_density(g)
    center_focus = _center_focus_ratio(g)

    vec = np.array(
        [
            _clip_unit(avg_luma),
            _clip_unit(contrast),
            _clip_unit(sharpness),
            _clip_unit(sat_mean),
            _clip_unit(sat_std),
            _clip_unit(cf),
            _clip_unit(border_clean),
            _clip_unit(center_focus),
        ],
        dtype=np.float32,
    )
    return vec


def compute_score_and_metrics(frame_bgr: np.ndarray) -> TechnicalResult:
    """
    Compute a robust technical score and a breakdown of named metrics.
    The score blends exposure, contrast, sharpness, palette diversity,
    border cleanliness, and center focus — all clamped to 0..1.
    """
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Cheap vector (8D)
    feat = cheap_feature_vector(frame, gray=gray, hsv=hsv)

    # Additional signals for the human-readable metrics
    top_bin, hue_entropy = _palette_stats(hsv)
    banding = _banding_proxy(gray)
    r_over_g, b_over_g = _grey_world_kelvin(frame)

    avg_luma, contrast, sharpness, sat_mean, sat_std, colorfulness, border_clean, center_focus = feat.tolist()

    # Exposure fitness prefers mid-range luminance (bell curve around 0.5)
    exposure_fit = _clip_unit(1.0 - abs(avg_luma - 0.5) * 2.0)

    # Palette diversity prefers higher hue entropy (ignore top-bin dominance)
    palette_diversity = hue_entropy

    # Final technical score (balanced blend)
    score = _mean(
        [
            exposure_fit,
            _clip_unit(contrast),
            _clip_unit(sharpness),
            palette_diversity,
            _clip_unit(colorfulness),
            _clip_unit(border_clean),
            _clip_unit(center_focus),
            banding,
        ]
    )

    metrics: Dict[str, float] = {
        "exposure_fit": float(exposure_fit),
        "contrast": float(_clip_unit(contrast)),
        "sharpness": float(_clip_unit(sharpness)),
        "saturation_mean": float(_clip_unit(sat_mean)),
        "saturation_std": float(_clip_unit(sat_std)),
        "colorfulness": float(_clip_unit(colorfulness)),
        "border_clean": float(_clip_unit(border_clean)),
        "center_focus": float(_clip_unit(center_focus)),
        "palette_diversity": float(_clip_unit(palette_diversity)),
        "banding_entropy": float(_clip_unit(banding)),
        "rg_over_g": float(r_over_g),
        "b_over_g": float(b_over_g),
    }

    return TechnicalResult(score=float(_clip_unit(score)), metrics=metrics, feature_vector=feat)
