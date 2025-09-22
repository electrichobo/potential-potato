# lines 1–120  (aesthetic/core/scene_candidates.py)
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import os
import yaml

def _load_cfg() -> Dict[str, Any]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    path = os.path.join(root, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_scene_spans(src_path: str) -> List[Tuple[int, int]]:
    """
    Returns list of (start_frame, end_frame) scene spans using PySceneDetect.
    Respects config.scenes.{method, threshold, downscale} and extract.min_scene_len_frames.
    """
    from scenedetect import SceneManager
    from scenedetect.detectors import ContentDetector, AdaptiveDetector
    from scenedetect.video_manager import VideoManager

    cfg = _load_cfg()
    scenes_cfg = (cfg.get("scenes") or {})
    extract_cfg = (cfg.get("extract") or {})
    method    = str(scenes_cfg.get("method", "content")).lower()
    threshold = float(scenes_cfg.get("threshold", 27.0))
    downscale = int(scenes_cfg.get("downscale", 2))
    min_len   = int(extract_cfg.get("min_scene_len_frames", 12))

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
        a, b = s.get_frames(), e.get_frames() - 1
        if b < a:
            continue
        if (b - a + 1) < max(1, min_len):
            continue
        spans.append((a, b))
    return spans
# end 1–120
