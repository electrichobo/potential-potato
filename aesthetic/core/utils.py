"""Utility helpers shared across the AESTHETIC pipeline.

This module centralises configuration loading/validation, deterministic seeding,
path normalisation, lightweight feature caching, and manifest/JSON helpers.  The
goal is to keep the core pipeline lean while providing clearly documented
building blocks that are easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

try:  # NumPy is optional in a few helper paths (e.g. seeding during tests)
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - NumPy is a runtime dependency in prod
    _np = None  # type: ignore


LOG = logging.getLogger("aesthetic.utils")


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Config helpers
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


def load_config_file(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load a YAML configuration file if it exists, otherwise return defaults."""

    import yaml  # Local import to keep the module importable during tests

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        LOG.warning("config not found at %s; using defaults", cfg_path)
        return json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy via JSON
    with cfg_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return merge_dicts(_DEFAULT_CONFIG, loaded)


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* returning a new dictionary."""

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
    """Return a new config with dot-notation overrides applied."""

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
    Validate and normalise configuration:

    * Merge with defaults.
    * Clamp numeric ranges.
    * Normalise relative paths to absolute.
    * Ensure optional blocks exist (e.g. hero_video dict).
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
    extract["min_candidate_gap_frames"] = max(
        1, int(extract.get("min_candidate_gap_frames", 6))
    )

    # Selection
    select = cfg.setdefault("select", {})
    select["top_k"] = max(1, int(select.get("top_k", 6)))
    select["overcluster_factor"] = max(
        1.0, float(select.get("overcluster_factor", 4.0))
    )
    select["per_cluster_top"] = max(1, int(select.get("per_cluster_top", 2)))

    # Dedup
    dedup = cfg.setdefault("dedup", {})
    dedup["enable"] = bool(dedup.get("enable", True))
    dedup["embed_min_cosine_dist"] = _clamp(
        dedup.get("embed_min_cosine_dist", 0.02), 0.0, 1.0, 0.02, key="dedup.embed_min_cosine_dist"
    )
    dedup["dhash"] = bool(dedup.get("dhash", True))
    dedup["dhash_threshold"] = max(0, int(dedup.get("dhash_threshold", 6)))

    # Output paths
    output = cfg.setdefault("output", {})
    base_folder = Path(output.get("folder", "aesthetic/outputs"))
    if not base_folder.is_absolute():
        base_folder = (PROJECT_ROOT / base_folder).resolve()
    output["folder"] = str(base_folder)
    hero = output.setdefault("hero_video", {})
    hero["enabled"] = bool(hero.get("enabled", True))
    hero["seconds_before"] = max(0.0, float(hero.get("seconds_before", 0.8)))
    hero["seconds_after"] = max(0.0, float(hero.get("seconds_after", 1.2)))
    hero["max_total_duration"] = max(1.0, float(hero.get("max_total_duration", 45.0)))
    hero["fps"] = max(1, int(hero.get("fps", 30)))
    hero["contact_sheet"] = bool(hero.get("contact_sheet", True))

    # Heavy settings (clip already validated downstream)
    gpu = cfg.setdefault("gpu", {})
    gpu["enabled"] = bool(gpu.get("enabled", True))
    gpu["device"] = gpu.get("device", "cuda:0")
    gpu["batch_size"] = max(1, int(gpu.get("batch_size", 8)))

    heavy = cfg.setdefault("heavy", {})
    clip = heavy.setdefault("clip", {})
    clip.setdefault("enabled", True)
    clip.setdefault("model", "ViT-B-32")
    clip.setdefault("precision", "fp16")
    clip.setdefault("run_mode", "subprocess")
    clip.setdefault("timeout_sec", 90)

    # Pillar weights must be positive; normalise lazily when used.
    pillars = cfg.setdefault("pillars", {})
    pillars.setdefault("technical_weight", 0.6)
    pillars.setdefault("creative_weight", 0.25)
    pillars.setdefault("subjective_weight", 0.15)

    advanced = cfg.setdefault("advanced_metrics", {})
    advanced.setdefault("enabled", False)
    advanced.setdefault("exposure", True)
    advanced.setdefault("lighting", True)
    advanced.setdefault("composition", True)
    advanced.setdefault("movement", True)
    advanced.setdefault("color", True)

    return cfg


def ensure_output_tree(cfg: Mapping[str, Any]) -> Tuple[Path, Path, Path]:
    """Ensure output/, output/logs, and output/cache exist."""

    out_dir = Path((cfg.get("output") or {}).get("folder", "aesthetic/outputs"))
    logs = out_dir / "logs"
    cache = out_dir / "cache"
    for path in (out_dir, logs, cache):
        path.mkdir(parents=True, exist_ok=True)
    return out_dir, logs, cache


def stable_config_signature(cfg: Mapping[str, Any]) -> str:
    """Return a deterministic hash of the configuration dictionary."""

    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def media_content_hash(source: str) -> str:
    """Compute a stable content hash for caching.

    * Local files: SHA1 of file bytes (streamed).
    * URLs / virtual sources: hash of the source string with a prefix.
    """

    try:
        path = Path(source)
        if path.exists() and path.is_file():
            sha1 = hashlib.sha1()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1 << 20)
                    if not chunk:
                        break
                    sha1.update(chunk)
            return sha1.hexdigest()
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("media_content_hash: fallback hashing for %s (%s)", source, exc)
    return hashlib.sha1(f"virtual::{source}".encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy PRNGs for deterministic behaviour."""

    random.seed(seed)
    if _np is not None:
        try:
            _np.random.seed(seed)
        except Exception:  # pragma: no cover - guard for alternative numpy impls
            pass


# ---------------------------------------------------------------------------
# Feature cache helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheKey:
    media_hash: str
    feature: str
    signature: str
    frame_index: int

    def path(self, cache_dir: Path, ext: str) -> Path:
        return cache_dir / self.media_hash / self.feature / f"{self.signature}_f{self.frame_index:08d}{ext}"


def load_cache_json(cache_dir: Path, key: CacheKey) -> Optional[Dict[str, Any]]:
    path = key.path(cache_dir, ".json")
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - corrupted cache should be ignored
        LOG.warning("cache json read failed for %s: %s", path, exc)
        return None


def save_cache_json(cache_dir: Path, key: CacheKey, payload: Mapping[str, Any]) -> None:
    path = key.path(cache_dir, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_cache_npz(cache_dir: Path, key: CacheKey) -> Optional[Any]:
    path = key.path(cache_dir, ".npz")
    if not path.exists():
        return None
    try:
        import numpy as np

        data = np.load(path)
        if "arr_0" in data:
            return data["arr_0"]
        return data
    except Exception as exc:  # pragma: no cover - treat as cache miss
        LOG.warning("cache npz read failed for %s: %s", path, exc)
        return None


def save_cache_npz(cache_dir: Path, key: CacheKey, array: Any) -> None:
    path = key.path(cache_dir, ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np

        np.savez_compressed(path, array)
    except Exception as exc:  # pragma: no cover - warn but continue
        LOG.warning("cache npz write failed for %s: %s", path, exc)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def as_absolute(path: str | Path, base: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    base = base or PROJECT_ROOT
    return (base / p).resolve()


def pillar_weight_normaliser(cfg: Mapping[str, Any]) -> Tuple[float, float, float]:
    pillars = cfg.get("pillars") or {}
    technical = max(0.0, float(pillars.get("technical_weight", 0.0)))
    creative = max(0.0, float(pillars.get("creative_weight", 0.0)))
    subjective = max(0.0, float(pillars.get("subjective_weight", 0.0)))
    total = technical + creative + subjective
    if total <= 0.0:
        return 1.0, 0.0, 0.0
    return technical / total, creative / total, subjective / total


def summarise_candidates(values: Iterable[float]) -> Dict[str, float]:
    vals = list(float(v) for v in values)
    if not vals:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    import statistics

    return {
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean": float(statistics.fmean(vals)),
    }


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_PATH",
    "load_config_file",
    "merge_dicts",
    "prepare_config",
    "apply_cli_overrides",
    "ensure_output_tree",
    "stable_config_signature",
    "media_content_hash",
    "seed_everything",
    "CacheKey",
    "load_cache_json",
    "save_cache_json",
    "load_cache_npz",
    "save_cache_npz",
    "write_json",
    "as_absolute",
    "pillar_weight_normaliser",
    "summarise_candidates",
]

