# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import yaml

def _cfg_path() -> str:
    """
    Resolve config.yaml path:
      1) env AESTHETIC_CONFIG if set
      2) <project_root>/config.yaml  (…/aesthetic/core/.. -> project root)
    """
    p = os.environ.get("AESTHETIC_CONFIG", "").strip()
    if p:
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(root, "config.yaml")

def _load_cfg() -> Dict[str, Any]:
    path = _cfg_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _fallback_cv_scenes(src_path: str, cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Fallback scene detector using HSV 2D histogram + Bhattacharyya distance.
    Mirrors the logic used in pipeline._scene_breaks and returns inclusive spans.
    """
    import cv2
    scenes_cfg = (cfg.get("scenes") or {})
    extract_cfg = (cfg.get("extract") or {})
    threshold = float(scenes_cfg.get("threshold", 27.0))
    downscale = int(scenes_cfg.get("downscale", 2))
    min_len   = int(extract_cfg.get("min_scene_len_frames", 12))

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return [(0, 0)]

    def _resize_w(frame_bgr, target_w: Optional[int]) -> "np.ndarray":
        if not target_w or target_w <= 0:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        if w <= target_w:
            return frame_bgr
        s = target_w / float(w)
        return cv2.resize(frame_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

    target_w = 720  # same default as pipeline.processing.resize_width
    ok, prev = cap.read()
    if not ok or prev is None:
        cap.release()
        return [(0, 0)]
    prev = _resize_w(prev, max(64, target_w // max(1, downscale)))
    prev_h = cv2.calcHist([cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    prev_h = cv2.normalize(prev_h, None).flatten()

    breaks = [0]
    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        small = _resize_w(frame, max(64, target_w // max(1, downscale)))
        h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        h = cv2.normalize(h, None).flatten()
        diff = cv2.compareHist(prev_h, h, cv2.HISTCMP_BHATTACHARYYA) * 100.0  # 0..100
        if diff >= threshold and (idx - breaks[-1]) >= min_len:
            breaks.append(idx)
            prev_h = h
        idx += 1
    cap.release()
    breaks.append(idx - 1)

    # Convert breaks to inclusive spans & enforce min_len
    spans: List[Tuple[int, int]] = []
    cleaned = [breaks[0]]
    for b in breaks[1:]:
        if b - cleaned[-1] >= min_len:
            cleaned.append(b)
    for i in range(len(cleaned) - 1):
        a, b = cleaned[i], cleaned[i + 1] - 1
        if b >= a and (b - a + 1) >= min_len:
            spans.append((a, b))
    return spans or [(0, max(0, idx - 2))]

def get_scene_spans(src_path: str) -> List[Tuple[int, int]]:
    """
    Returns list of (start_frame, end_frame) scene spans.
    Prefers PySceneDetect (content/adaptive detector) and falls back to OpenCV histogram
    when PySceneDetect is unavailable or fails.
    Respects:
      scenes.method: "content" | "adaptive"
      scenes.threshold: float
      scenes.downscale: int
      extract.min_scene_len_frames: int
    """
    cfg = _load_cfg()
    scenes_cfg = (cfg.get("scenes") or {})
    extract_cfg = (cfg.get("extract") or {})
    method    = str(scenes_cfg.get("method", "content")).lower()
    threshold = float(scenes_cfg.get("threshold", 27.0))
    downscale = int(scenes_cfg.get("downscale", 2))
    min_len   = int(extract_cfg.get("min_scene_len_frames", 12))

    # Try PySceneDetect path
    try:
        from scenedetect import SceneManager
        from scenedetect.detectors import ContentDetector, AdaptiveDetector
        from scenedetect.video_manager import VideoManager

        vm = VideoManager([src_path])
        try:
            vm.set_downscale_factor(max(1, downscale))
            vm.start()
            sm = SceneManager()
            if method == "adaptive":
                det = AdaptiveDetector(adaptive_threshold=threshold)
            else:
                det = ContentDetector(threshold=threshold)
            sm.add_detector(det)
            sm.detect_scenes(frame_source=vm)
            scene_list = sm.get_scene_list(base_timecode=vm.get_base_timecode())
        finally:
            vm.release()

        spans: List[Tuple[int, int]] = []
        for s, e in scene_list:
            a, b = s.get_frames(), e.get_frames() - 1  # inclusive end
            if b < a:
                continue
            if (b - a + 1) < max(1, min_len):
                continue
            spans.append((a, b))
        if spans:
            return spans
        # fall-through to fallback if PySceneDetect produced nothing
    except Exception:
        # PySceneDetect missing or failed → fallback
        pass

    return _fallback_cv_scenes(src_path, cfg)
