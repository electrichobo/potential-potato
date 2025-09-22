# lines 1–60
from __future__ import annotations
# --- config loader (shallow-merge defaults) ---
import os, yaml

_DEFAULT_CFG = {
    "extract": {"mode": "hybrid", "random_frames": 200, "scene_sample": "middle", "fps_limit": 4},
    "select": {"top_k": 5, "overcluster_factor": 3.0, "per_cluster_top": 3, "quality_floor": None},
    "weights": {"technical": 0.5, "subjective": 0.5},
    "output": {"save_audits": True, "score_prefix": True, "folder": "aesthetic/outputs"},
    "ffmpeg": {"bin": r"C:\ffmpeg\bin\ffmpeg.exe", "probe": r"C:\ffmpeg\bin\ffprobe.exe"},
    "logging": {"verbose": True, "save_log": True},
    "models": {"clip": None},
}

def load_config(path: str = "config.yaml") -> dict:
    if not os.path.isfile(path):
        return _DEFAULT_CFG
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = _DEFAULT_CFG.copy()
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merged = cfg[k].copy(); merged.update(v); cfg[k] = merged
        else:
            cfg[k] = v
    return cfg
# end 1–60
