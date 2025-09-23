# -*- coding: utf-8 -*-
"""
Utility helpers shared across the AESTHETIC pipeline.

Centralizes configuration loading/validation, deterministic seeding,
path normalization, lightweight feature caching, and manifest/JSON helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Union

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

LOG = logging.getLogger("aesthetic.utils")

# ---------------------------------------------------------------------------
# Project paths (this module lives at aesthetic/core/utils.py)
# parents[2] is the repo root (e.g., E:\AestheticApp)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# Default config (kept aligned with README/agents)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "sampling": {"random_seed": 42, "jitter_frac": 0.12},
    "extract": {
        "per_scene_candidates": 9,
        "per_scene_keep_pct": 0.4,
        "min_scene_len_frames": 12,
        "min_candidate_gap_frames": 6,
    },
    "select": {"top_k": 6, "overcluster_factor": 4.0, "per_cluster_top": 2},
    "dedup": {
        "enable": True,
        "embed_min_cosine_dist": 0.02,
        "dhash": True,
        "dhash_threshold": 6,
    },
    "output": {
        "folder": "aesthetic/outputs",
        "score_prefix": True,
        "save_audits": True,
        "write_metrics_txt": True,
        "hero_video": {
            "enabled": True,
            "seconds_before": 0.8,
            "seconds_after": 1.2,
            "max_total_duration": 45.0,
            "fps": 30,
            "contact_sheet": True,
        },
    },
    "gpu": {"enabled": True, "device": "cuda:0", "batch_size": 8},
    "heavy": {
        "clip": {
            "enabled": True,
            "model": "ViT-B-32",
            "precision": "fp16",
            "run_mode": "subprocess",
            "timeout_sec": 90,
        }
    },
    "advanced_metrics": {
        "enabled": False,
        "exposure": True,
        "lighting": True,
        "composition": True,
        "movement": True,
        "color": True,
    },
    "pillars": {
        "technical_weight": 0.6,
        "creative_weight": 0.25,
        "subjective_weight": 0.15,
    },
}

# ---------------------------------------------------------------------------
# Config I/O + merging
# ---------------------------------------------------------------------------

def load_config_file(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load YAML config from *path* or from DEFAULT_CONFIG_PATH if omitted."""
    import yaml  # local import

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        LOG.warning("config not found at %s; using defaults", cfg_path)
        return json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy
    with cfg_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return merge_dicts(_DEFAULT_CONFIG, loaded)


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    out: Dict[str, Any] = json.loads(json.dumps(base))  # deep copy
    stack: list[Tuple[MutableMapping[str, Any], Mapping[str, Any]]] = [(out, override)]
    while stack:
        dest, src = stack.pop()
        for key, value in src.items():
            if isinstance(value, Mapping) and isinstance(dest.get(key), MutableMapping):
                stack.append((dest[key], value))  # type: ignore[arg-type]
            else:
                dest[key] = value  # type: ignore[index]
    return out


def _clamp(value: Any, lo: float, hi: float, default: float, *, key: str) -> float:
    """Clamp numeric config values with logging."""
    try:
        v = float(value)
    except Exception:
        LOG.warning("config[%s]=%r invalid; using default %.3f", key, value, default)
        return float(default)
    if v < lo:
        LOG.warning("config[%s]=%.3f below %.3f; clamped", key, v, lo)
        return float(lo)
    if v > hi:
        LOG.warning("config[%s]=%.3f above %.3f; clamped", key, v, hi)
        return float(hi)
    return v


def apply_cli_overrides(cfg: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply dot-notation overrides to a config copy."""
    out = json.loads(json.dumps(cfg))
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cursor: Any = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
        cursor[parts[-1]] = value  # type: ignore[index]
    return out


def prepare_config(raw_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate and normalize configuration:
    - merge with defaults
    - clamp numeric ranges
    - ensure optional blocks exist
    """
    cfg = merge_dicts(_DEFAULT_CONFIG, raw_cfg or {})

    # Sampling
    sampling = cfg.setdefault("sampling", {})
    sampling["random_seed"] = int(sampling.get("random_seed", 42))
    sampling["jitter_frac"] = _clamp(
        sampling.get("jitter_frac", 0.12), 0.0, 0.5, 0.12, key="sampling.jitter_frac"
    )

    # Extract
    extract = cfg.setdefault("extract", {})
    extract["per_scene_candidates"] = max(1, int(extract.get("per_scene_candidates", 9)))
    extract["per_scene_keep_pct"] = _clamp(
        extract.get("per_scene_keep_pct", 0.4), 0.05, 1.0, 0.4, key="extract.per_scene_keep_pct"
    )
    extract["min_scene_len_frames"] = max(1, int(extract.get("min_scene_len_frames", 12)))
    extract["min_candidate_gap_frames"] = max(1, int(extract.get("min_candidate_gap_frames", 6)))

    # Selection
    select = cfg.setdefault("select", {})
    select["top_k"] = max(1, int(select.get("top_k", 6)))
    select["overcluster_factor"] = max(1.0, float(select.get("overcluster_factor", 4.0)))
    select["per_cluster_top"] = max(1, int(select.get("per_cluster_top", 2)))

    # Dedup
    dedup = cfg.setdefault("dedup", {})
    dedup["enable"] = bool(dedup.get("enable", True))
    dedup["embed_min_cosine_dist"] = float(dedup.get("embed_min_cosine_dist", 0.02))
    dedup["dhash"] = bool(dedup.get("dhash", True))
    dedup["dhash_threshold"] = max(0, int(dedup.get("dhash_threshold", 6)))

    # Output
    output = cfg.setdefault("output", {})
    output["folder"] = str(Path(output.get("folder", "aesthetic/outputs")))
    hero_video = output.setdefault("hero_video", {})
    hero_video["enabled"] = bool(hero_video.get("enabled", True))
    hero_video["seconds_before"] = float(hero_video.get("seconds_before", 0.8))
    hero_video["seconds_after"] = float(hero_video.get("seconds_after", 1.2))
    hero_video["max_total_duration"] = float(hero_video.get("max_total_duration", 45.0))
    hero_video["fps"] = int(hero_video.get("fps", 30))
    hero_video["contact_sheet"] = bool(hero_video.get("contact_sheet", True))

    # GPU/Heavy
    gpu = cfg.setdefault("gpu", {})
    gpu["enabled"] = bool(gpu.get("enabled", True))
    gpu["device"] = str(gpu.get("device", "cuda:0"))
    gpu["batch_size"] = int(gpu.get("batch_size", 8))
    heavy = cfg.setdefault("heavy", {})
    clip = heavy.setdefault("clip", {})
    clip["enabled"] = bool(clip.get("enabled", True))
    clip["model"] = str(clip.get("model", "ViT-B-32"))
    clip["precision"] = str(clip.get("precision", "fp16"))
    clip["run_mode"] = str(clip.get("run_mode", "subprocess"))
    clip["timeout_sec"] = int(clip.get("timeout_sec", 90))

    # Pillars
    pillars = cfg.setdefault("pillars", {})
    pillars["technical_weight"] = float(pillars.get("technical_weight", 0.6))
    pillars["creative_weight"] = float(pillars.get("creative_weight", 0.25))
    pillars["subjective_weight"] = float(pillars.get("subjective_weight", 0.15))

    return cfg

# ---------------------------------------------------------------------------
# Output tree and logging directories
# ---------------------------------------------------------------------------

def ensure_output_tree(cfg: Mapping[str, Any]) -> Tuple[Path, Path, Path]:
    """
    Ensure output, log, and cache directories exist based on config.
    Returns (out_dir, log_dir, cache_dir).
    """
    out_dir = Path(cfg.get("output", {}).get("folder", "aesthetic/outputs")).expanduser().resolve()
    log_dir = out_dir / "logs"
    cache_dir = out_dir / "cache"
    for p in (out_dir, log_dir, cache_dir):
        p.mkdir(parents=True, exist_ok=True)
    return out_dir, log_dir, cache_dir

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed Python, random, and NumPy for reproducible runs."""
    random.seed(seed)
    try:
        os.environ["PYTHONHASHSEED"] = str(int(seed))
    except Exception:
        pass
    if _np is not None:
        try:
            _np.random.seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Hashes, caching, JSON helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheKey:
    media_hash: str
    feature_name: str
    cfg_signature: str
    frame_index: int


def media_content_hash(path: str) -> str:
    """Return a stable content hash for the input media path."""
    p = Path(path)
    s = f"{p.resolve()}::{p.stat().st_size if p.exists() else 0}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def stable_config_signature(cfg: Mapping[str, Any]) -> str:
    """Return a short signature for config-impacting fields."""
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serialisable forms:
      * NumPy scalars -> Python scalars
      * NumPy arrays  -> lists
      * Path          -> str
      * sets/tuples   -> lists
      * mappings/sequences -> converted recursively
    """
    # NumPy scalar?
    if _np is not None and isinstance(obj, _np.generic):  # type: ignore[attr-defined]
        return obj.item()
    # NumPy array?
    if _np is not None and isinstance(obj, _np.ndarray):  # type: ignore[attr-defined]
        return obj.tolist()
    # Path
    if isinstance(obj, Path):
        return str(obj)
    # Basic builtins
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Mapping
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # Iterable (list/tuple/set)
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # Fallback: string repr to avoid crashes in late-stage exports
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"

def save_cache_json(root: Path, key: CacheKey, payload: Mapping[str, Any]) -> None:
    path = root / f"{key.media_hash}_{key.feature_name}_{key.cfg_signature}_{key.frame_index}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)

def load_cache_json(root: Path, key: CacheKey) -> Optional[Dict[str, Any]]:
    path = root / f"{key.media_hash}_{key.feature_name}_{key.cfg_signature}_{key.frame_index}.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cache_npz(root: Path, key: CacheKey, array: "Any") -> None:
    import numpy as np  # local
    path = root / f"{key.media_hash}_{key.feature_name}_{key.cfg_signature}_{key.frame_index}.npz"
    np.savez_compressed(str(path), data=array)

def load_cache_npz(root: Path, key: CacheKey) -> Optional["Any"]:
    import numpy as np  # local
    path = root / f"{key.media_hash}_{key.feature_name}_{key.cfg_signature}_{key.frame_index}.npz"
    if path.exists():
        try:
            with np.load(str(path)) as data:
                return data["data"]
        except Exception:
            return None
    return None

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with robust coercion for NumPy/Path types."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)

# ---------------------------------------------------------------------------
# Light summaries
# ---------------------------------------------------------------------------

def summarise_candidates(items: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    total = 0
    scenes = set()
    for it in items:
        total += 1
        scenes.add(int(it.get("scene_index", 0))) if isinstance(it, Mapping) else None
    return {"total": total, "unique_scenes": len(scenes)}

# ---------------------------------------------------------------------------
# Pillar weights — accepts cfg *or* 3 floats
# ---------------------------------------------------------------------------

def pillar_weight_normaliser(
    a: Union[Mapping[str, Any], float],
    b: Optional[float] = None,
    c: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Return normalised (wt, wc, ws) in [0..1] summing to 1.

    Accepts either:
      - pillar_weight_normaliser(cfg_mapping)
      - pillar_weight_normaliser(wt, wc, ws)
    """
    if isinstance(a, Mapping) and b is None and c is None:
        pillars = (a.get("pillars") or {})
        wt = float(pillars.get("technical_weight", 0.6))
        wc = float(pillars.get("creative_weight", 0.25))
        ws = float(pillars.get("subjective_weight", 0.15))
    else:
        wt = float(a)  # type: ignore[arg-type]
        wc = float(b if b is not None else 0.0)
        ws = float(c if c is not None else 0.0)
    s = max(1e-6, wt + wc + ws)
    return (wt / s, wc / s, ws / s)
