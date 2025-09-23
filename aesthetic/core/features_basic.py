"""Phase 1 technical metrics built with OpenCV primitives.

The functions below compute a deterministic per-frame technical score along with
individual metric breakdowns that are later surfaced in sidecars.  Everything is
CPU friendly and returns floats in the 0–1 range whenever reasonable so the
selector can blend metrics without additional scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TechnicalResult:
    score: float
    metrics: Dict[str, float]
    feature_vector: np.ndarray


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _clip_unit(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _grey_world_kelvin(bgr: np.ndarray) -> Tuple[float, float]:
    b, g, r = [float(np.mean(chan)) for chan in cv2.split(bgr.astype(np.float32))]
    eps = 1e-6
    avg = (r + g + b) / 3.0
    if avg <= eps:
        return 6500.0, 0.0
    rw, gw, bw = r / (avg + eps), g / (avg + eps), b / (avg + eps)
    temp = 6500.0 * (2.0 - (rw + gw) / 2.0)
    deviation = abs(rw - bw)
    return float(temp), float(deviation)


def _histogram_skew(values: np.ndarray) -> float:
    vals = values.astype(np.float32).flatten()
    if vals.size == 0:
        return 0.0
    mean = float(np.mean(vals))
    std = float(np.std(vals) + 1e-6)
    centered = vals - mean
    skew = float(np.mean((centered / std) ** 3))
    return skew


def _saliency_map(gray: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = cv2.GaussianBlur(np.abs(lap), (5, 5), 0)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    saliency = 0.6 * lap + 0.4 * sat
    saliency = cv2.GaussianBlur(saliency, (7, 7), 0)
    saliency = cv2.normalize(saliency, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return saliency


def _centre_of_mass(map_: np.ndarray) -> Tuple[float, float]:
    h, w = map_.shape[:2]
    if map_.sum() <= 1e-6:
        return 0.5, 0.5
    y_coords, x_coords = np.indices((h, w))
    weights = map_.astype(np.float64)
    cx = float((x_coords * weights).sum() / weights.sum()) / max(1, w - 1)
    cy = float((y_coords * weights).sum() / weights.sum()) / max(1, h - 1)
    return cx, cy


def _distance_to_rule_of_thirds(cx: float, cy: float) -> float:
    thirds = [(1 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (2 / 3, 2 / 3)]
    dist = min(((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 for tx, ty in thirds)
    max_dist = (2 ** 0.5) / 3
    return _clip_unit(1.0 - dist / max_dist)


def _symmetry_score(edges: np.ndarray) -> float:
    flipped = cv2.flip(edges, 1)
    diff = cv2.absdiff(edges, flipped)
    norm = cv2.normalize(diff, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return _clip_unit(1.0 - float(norm.mean()))


def _negative_space_ratio(saliency: np.ndarray, threshold: float = 0.5) -> float:
    mask = saliency > threshold
    if mask.size == 0:
        return 0.5
    ratio = 1.0 - float(mask.mean())
    return _clip_unit(ratio)


def _motion_blur_proxy(gray: np.ndarray) -> float:
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return _clip_unit(lap_var / (lap_var + 800.0))


def _banding_proxy(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    diff = cv2.absdiff(gray, blur)
    hist = cv2.calcHist([diff], [0], None, [32], [0, 256])
    hist = hist / max(1.0, hist.sum())
    entropy = -float(np.sum(hist * np.log2(hist + 1e-8)))
    return _clip_unit(entropy / np.log2(32))


def _palette_stats(hsv: np.ndarray) -> Tuple[float, float]:
    hue = hsv[..., 0].astype(np.float32)
    hist = cv2.calcHist([hue], [0], None, [18], [0, 180])
    hist = hist / max(1.0, hist.sum())
    top_bin = float(hist.max())
    entropy = -float(np.sum(hist * np.log2(hist + 1e-8))) / np.log2(hist.size)
    return _clip_unit(1.0 - top_bin), _clip_unit(entropy)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cheap_feature_vector(frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
         avg_luma = float(gray.mean()) / 255.0
    contrast = float(np.std(gray)) / 128.0
    saturation_mean = float(hsv[..., 1].mean()) / 255.0
    saturation_std = float(hsv[..., 1].std()) / 128.0
    symmetry = _symmetry_score(cv2.Canny(gray, 50, 150))
    neg_space = _negative_space_ratio(_saliency_map(gray, hsv))
    motion = _motion_blur_proxy(gray)
    banding = _banding_proxy(gray)

    features = np.array(
        [
            _clip_unit(avg_luma),
            _clip_unit(contrast),
            _clip_unit(saturation_mean),
            _clip_unit(saturation_std),
            symmetry,
            neg_space,
            motion,
            banding,
        ],
        dtype=np.float32,
    )
    return features


def compute_score_and_metrics(frame_bgr: np.ndarray) -> TechnicalResult:
    if frame_bgr is None or frame_bgr.size == 0:
        zero = np.zeros((8,), dtype=np.float32)
        return TechnicalResult(score=0.0, metrics={}, feature_vector=zero)

    frame = np.asarray(frame_bgr, dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saliency = _saliency_map(gray, hsv)
    edges = cv2.Canny(gray, 40, 120)
    luma = gray.astype(np.float32)

    # Exposure ---------------------------------------------------------------
    avg_luma = float(luma.mean()) / 255.0
    median_luma = float(np.median(luma)) / 255.0
    p10, p90 = np.percentile(luma, [10, 90])
    p10_norm, p90_norm = p10 / 255.0, p90 / 255.0
    clipped_shadows = float(np.mean(luma <= 12.0))
    clipped_highlights = float(np.mean(luma >= 243.0))
    luma_std = float(np.std(luma)) / 64.0
    skew = _histogram_skew(luma)
    exposure_balance = _clip_unit(1.0 - abs(avg_luma - 0.55) * 2.0)
    clip_score = _clip_unit(1.0 - 0.6 * (clipped_shadows + clipped_highlights))
    contrast_score = _clip_unit(luma_std / (luma_std + 0.6))
    temporal_uniformity = 0.5  # placeholder until temporal window is added

    exposure_score = _mean([exposure_balance, clip_score, contrast_score])

    # Lighting ---------------------------------------------------------------
    dynamic_range = _clip_unit((p90 - p10) / 255.0)
    bright_mask = luma >= 0.7 * 255
    mid_mask = (luma >= 0.3 * 255) & (luma < 0.7 * 255)
    bright_ratio = float(bright_mask.mean())
    mid_ratio = float(mid_mask.mean())
    key_fill_ratio = _clip_unit(1.0 - abs(np.log2((bright_ratio + 1e-4) / (mid_ratio + 1e-4))) / 4.0)
    dark_mask = luma < 0.25 * 255
    if dark_mask.any():
        shadow_detail = _clip_unit(float(edges[dark_mask].mean()) / 255.0)
    else:
        shadow_detail = 0.5
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    spill_softness = _clip_unit(1.0 - float(grad_mag.mean()) / 128.0)

    lighting_score = _mean([dynamic_range, key_fill_ratio, shadow_detail, spill_softness])

    # Composition ------------------------------------------------------------
    cx, cy = _centre_of_mass(saliency)
    thirds = _distance_to_rule_of_thirds(cx, cy)
    centre_distance = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    occupancy = _clip_unit(1.0 - centre_distance * 2.0)
    symmetry = _symmetry_score(edges)
    neg_space = _negative_space_ratio(saliency)
    composition_score = _mean([thirds, occupancy, symmetry, neg_space])

    # Movement ---------------------------------------------------------------
    flow_smoothness = 0.5  # placeholder until optical flow window is wired
    jerkiness = 0.5
    motion_blur = _motion_blur_proxy(gray)
    movement_score = _mean([flow_smoothness, 1.0 - jerkiness, motion_blur])

    # Color ------------------------------------------------------------------
    wb_kelvin, wb_dev = _grey_world_kelvin(frame)
    wb_deviation = _clip_unit(1.0 - min(1.0, wb_dev))
    sat_mean = float(hsv[..., 1].mean()) / 255.0
    sat_std = float(hsv[..., 1].std()) / 128.0
    palette_balance, palette_entropy = _palette_stats(hsv)
    banding = _banding_proxy(gray)
    color_score = _mean(
        [
            wb_deviation,
            _clip_unit(1.0 - abs(sat_mean - 0.6) * 1.4),
            _clip_unit(1.0 - abs(sat_std - 0.3) * 2.0),
            palette_entropy,
            banding,
        ]
    )

    weights = {
        "exposure": 0.25,
        "lighting": 0.2,
        "composition": 0.2,
        "movement": 0.15,
        "color": 0.2,
    }
    score = (
        weights["exposure"] * exposure_score
        + weights["lighting"] * lighting_score
        + weights["composition"] * composition_score
        + weights["movement"] * movement_score
        + weights["color"] * color_score
    )
    score = _clip_unit(score)

    metrics: Dict[str, float] = {
        # Exposure
        "exposure_score": exposure_score,
        "average_luma": avg_luma,
        "median_luma": median_luma,
        "p10_luma": p10_norm,
        "p90_luma": p90_norm,
        "clipped_shadows_pct": clipped_shadows,
        "clipped_highlights_pct": clipped_highlights,
        "histogram_std_norm": _clip_unit(luma_std),
        "histogram_skew": float(skew),
        "temporal_uniformity": temporal_uniformity,
        # Lighting
        "lighting_score": lighting_score,
        "dynamic_range_norm": dynamic_range,
        "key_fill_ratio": key_fill_ratio,
        "shadow_detail": shadow_detail,
        "spill_softness": spill_softness,
        # Composition
        "composition_score": composition_score,
        "thirds_adherence": thirds,
        "occupancy_center_of_mass": occupancy,
        "symmetry_score": symmetry,
        "negative_space_ratio": neg_space,
        # Movement
        "movement_score": movement_score,
        "flow_smoothness": flow_smoothness,
        "jerkiness": 1.0 - jerkiness,
        "motion_blur_amount": motion_blur,
        # Color
        "color_score": color_score,
        "wb_kelvin_estimate": wb_kelvin,
        "wb_deviation_norm": wb_deviation,
        "saturation_mean": sat_mean,
        "saturation_std": sat_std,
        "palette_balance": palette_balance,
        "palette_entropy": palette_entropy,
        "artifact_banding_proxy": banding,
        # Final
        "technical_score": score,
    }
 
    return TechnicalResult(score=score, metrics=metrics, feature_vector=cheap_feature_vector(frame))


__all__ = ["TechnicalResult", "compute_score_and_metrics", "cheap_feature_vector"]
