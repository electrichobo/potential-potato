"""End-to-end pipeline for the AESTHETIC hero-frame selector.

The module is intentionally organised into small, well-documented helpers so a
future maintainer can reason about each stage in isolation.  The top-level
``run_pipeline`` function orchestrates the stages and acts as the single entry
point that both the GUI and the CLI reuse.

Stages
------
1. **Configuration & ingest** – resolve the source, seed RNGs, and gather video
   metadata.
2. **Scene sampling** – detect scenes (with a histogram fallback) and sample
   candidate frame indices per scene.
3. **Feature extraction** – compute the cheap technical pillar metrics (cached)
   alongside placeholder creative/subjective data and optional advanced
   metrics.
4. **Heavy features** – invoke CLIP in a subprocess when enabled, storing
   embeddings in the feature cache so subsequent runs are fast.
5. **Scoring & pruning** – blend pillar scores, prune per-scene, and pass the
   shortlist to the k-medoids/facility-location selector.
6. **Deduplication & export** – remove near-duplicates, write frames + sidecars,
   and optionally produce a hero video + contact sheet.

Every helper returns serialisable diagnostics so the GUI can surface detailed
status information.  The "always-write" invariant is preserved by falling back
to the global top candidate whenever downstream stages fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from aesthetic.core.utils import (
    CacheKey,
    apply_cli_overrides,
    ensure_output_tree,
    load_config_file,
    media_content_hash,
    pillar_weight_normaliser,
    prepare_config,
    save_cache_json,
    load_cache_json,
    load_cache_npz,
    save_cache_npz,
    seed_everything,
    stable_config_signature,
    summarise_candidates,
    write_json,
)
from aesthetic.core.ingest import resolve_source
from aesthetic.core.scene_candidates import detect_scenes, spans_to_metadata
from aesthetic.core.extractor import indices_from_scene_spans
from aesthetic.core.features_basic import compute_score_and_metrics, TechnicalResult
from aesthetic.core.features_advanced import compute_advanced_metrics
from aesthetic.core.video_export import VideoInfo, export_hero_video
from aesthetic.core.selector import select_hybrid, SelectionDiagnostics
from aesthetic.core import creative, subjective
from aesthetic.features.deep_clip import embed_frames_if_enabled


LOG = logging.getLogger("aesthetic.pipeline")

ProgressCb = Optional[Callable[[float, str], None]]
CancelFn = Optional[Callable[[], bool]]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoMetadata:
    """Describe the video capture so timestamps and exports are reliable."""

    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


@dataclass(frozen=True)
class PipelineContext:
    """Capture configuration, paths, and deterministic identifiers."""

    cfg: Dict[str, Any]
    output_dir: Path
    log_dir: Path
    cache_dir: Path
    cfg_signature: str
    seed: int
    resolved_source: str
    media_hash: str


@dataclass
class Candidate:
    """All per-frame information required for selection and exports."""

    scene_index: int
    frame_index: int
    timestamp: float
    frame: np.ndarray
    tech: TechnicalResult
    advanced: Dict[str, Optional[float]]
    creative: Dict[str, Optional[float]]
    subjective: Dict[str, Optional[float]]
    cheap_vector: np.ndarray
    clip_embedding: Optional[np.ndarray] = None
    prune_score: float = 0.0
    final_score: float = 0.0
    pillar_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    selection_meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration & ingest helpers
# ---------------------------------------------------------------------------


def _initialise_context(
    source: str,
    config: Optional[Dict[str, Any]],
    config_path: Optional[Path],
    cli_overrides: Optional[Mapping[str, Any]],
) -> PipelineContext:
    """Load configuration, seed RNGs, resolve the source, and prep directories."""

    base_cfg = config or load_config_file(config_path)
    if cli_overrides:
        base_cfg = apply_cli_overrides(base_cfg, cli_overrides)
    cfg = prepare_config(base_cfg)
    out_dir, log_dir, cache_dir = ensure_output_tree(cfg)
    cfg_signature = stable_config_signature(cfg)

    seed = int((cfg.get("sampling") or {}).get("random_seed", 42))
    seed_everything(seed)

    resolved = resolve_source(source, config=cfg)
    media_hash = media_content_hash(resolved)

    return PipelineContext(
        cfg=cfg,
        output_dir=out_dir,
        log_dir=log_dir,
        cache_dir=cache_dir,
        cfg_signature=cfg_signature,
        seed=seed,
        resolved_source=resolved,
        media_hash=media_hash,
    )


def _open_capture(source: str) -> Tuple[cv2.VideoCapture, VideoMetadata]:
    """Open the video and extract essential metadata."""

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (frame_count / fps) if fps > 0 else 0.0

    meta = VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration=duration,
    )
    return cap, meta


def _scene_index(frame_index: int, spans: Sequence[Tuple[int, int]]) -> int:
    """Return the zero-based scene index for *frame_index* (fallbacks to 0)."""

    for idx, (start, end) in enumerate(spans):
        if start <= frame_index <= end:
            return idx
    return 0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _resize_to_width(frame: np.ndarray, width: Optional[int]) -> np.ndarray:
    if not width or width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _extract_candidates(
    cap: cv2.VideoCapture,
    indices: Sequence[int],
    spans: Sequence[Tuple[int, int]],
    meta: VideoMetadata,
    context: PipelineContext,
    resize_width: Optional[int],
    progress_cb: ProgressCb,
    cancel_fn: CancelFn,
) -> Tuple[List[Candidate], np.ndarray]:
    """Read frames and compute cheap features with caching."""

    candidates: List[Candidate] = []
    norms: List[float] = []
    basic_key_prefix = "basic_v1"

    for idx, frame_index in enumerate(indices):
        if cancel_fn and cancel_fn():
            raise RuntimeError("pipeline cancelled")
        if progress_cb and indices:
            progress_cb(0.12 + 0.1 * (idx / max(1, len(indices))), "extracting frames")

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = _resize_to_width(frame, resize_width)

        timestamp = (frame_index / meta.fps) if meta.fps > 0 else 0.0
        scene_index = _scene_index(frame_index, spans)

        cache_key = CacheKey(
            context.media_hash,
            basic_key_prefix,
            context.cfg_signature,
            int(frame_index),
        )

        cached = load_cache_json(context.cache_dir, cache_key)
        if cached:
            feature_vector = np.asarray(cached.get("feature_vector", []), dtype=np.float32)
            tech = TechnicalResult(
                score=float(cached.get("technical_score", 0.0)),
                metrics=cached.get("metrics", {}),
                feature_vector=feature_vector,
            )
        else:
            tech = compute_score_and_metrics(frame)
            payload = {
                "technical_score": float(tech.score),
                "metrics": tech.metrics,
                "feature_vector": tech.feature_vector.tolist(),
            }
            save_cache_json(context.cache_dir, cache_key, payload)
        advanced_metrics = compute_advanced_metrics(frame, context.cfg)

        creative_metrics: Dict[str, Optional[float]] = {}
        creative_metrics.update(creative.lighting_style_adherence(None))
        creative_metrics.update(creative.composition_creativity(None))
        creative_metrics.update(creative.movement_impact(None))
        creative_metrics.update(creative.palette_mood_accuracy(None))
        creative_metrics.update(creative.motif_repetition(None))

        subjective_metrics: Dict[str, Optional[float]] = {}
        subjective_metrics.update(subjective.perceived_clarity(None))
        subjective_metrics.update(subjective.emotional_response(None))
        subjective_metrics.update(subjective.engagement_signals(None))
        subjective_metrics.update(subjective.aesthetic_impression(None))
        subjective_metrics.update(subjective.memorability_iconic(None))

        cand = Candidate(
            scene_index=scene_index,
            frame_index=int(frame_index),
            timestamp=float(timestamp),
            frame=frame,
            tech=tech,
            advanced=advanced_metrics,
            creative=creative_metrics,
            subjective=subjective_metrics,
            cheap_vector=tech.feature_vector,
        )
        candidates.append(cand)
        norms.append(float(np.linalg.norm(tech.feature_vector)))

    if not candidates:
        raise RuntimeError("no frames sampled from scenes")

    return candidates, np.asarray(norms, dtype=np.float32)


def _embed_clip_embeddings(
    candidates: Sequence[Candidate],
    context: PipelineContext,
    clip_signature: str,
) -> Dict[str, Any]:
    """Populate CLIP embeddings (cached when possible)."""

    clip_cfg = ((context.cfg.get("heavy") or {}).get("clip") or {})
    clip_enabled = bool(clip_cfg.get("enabled", False))
    status = {"enabled": clip_enabled, "timed_out": False, "computed": 0}
    if not clip_enabled:
        return status

    clip_feature_key = "clip_v1"
    pending: List[Tuple[Candidate, CacheKey]] = []

    for cand in candidates:
        cache_key = CacheKey(
            context.media_hash,
            clip_feature_key,
            clip_signature,
            cand.frame_index,
        )
        cached = load_cache_npz(context.cache_dir, cache_key)
        if cached is not None:
            cand.clip_embedding = np.asarray(cached, dtype=np.float32).flatten()
            cand.selection_meta["clip_cached"] = True
        else:
            pending.append((cand, cache_key))

    if not pending:
        return status

    frames = [cand.frame for cand, _ in pending]
    clip_result = embed_frames_if_enabled(frames, context.cfg)
    if not clip_result:
        return status

    status["timed_out"] = bool(clip_result.get("timed_out", False))
    feats = clip_result.get("feat_clip")
    if feats is None:
        return status

    for (cand, cache_key), vec in zip(pending, np.asarray(feats)):
        cand.clip_embedding = np.asarray(vec, dtype=np.float32).flatten()
        save_cache_npz(context.cache_dir, cache_key, cand.clip_embedding)
        status["computed"] += 1

    return status


# ---------------------------------------------------------------------------
# Scoring & selection utilities
# ---------------------------------------------------------------------------


def _scale_unit(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    scaled = (values - vmin) / (vmax - vmin)
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled.astype(np.float32)


def _aggregate_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def _compute_final_score(
    tech: float,
    creative_score: Optional[float],
    subjective_score: Optional[float],
    weights: Tuple[float, float, float],
) -> Tuple[float, Dict[str, Optional[float]]]:
    wt, wc, ws = weights
    parts: List[Tuple[float, float]] = []
    if tech is not None:
        parts.append((wt, float(tech)))
    if creative_score is not None:
        parts.append((wc, float(creative_score)))
    if subjective_score is not None:
        parts.append((ws, float(subjective_score)))
    if not parts:
        return float(tech), {
            "technical": tech,
            "creative": creative_score,
            "subjective": subjective_score,
        }
    total_weight = sum(w for w, _ in parts)
    total = sum(w * s for w, s in parts)
    final = total / total_weight if total_weight > 0 else float(tech)
    return float(final), {
        "technical": tech,
        "creative": creative_score,
        "subjective": subjective_score,
    }


def _score_candidates(
    candidates: Sequence[Candidate],
    cheap_norms: np.ndarray,
    clip_status: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Blend pillar scores and compute prune heuristics."""

    cheap_scaled = _scale_unit(cheap_norms.astype(np.float32))
    clip_norms = []
    for cand in candidates:
        clip_norms.append(
            float(np.linalg.norm(cand.clip_embedding)) if cand.clip_embedding is not None else math.nan
        )
    clip_array = np.asarray(clip_norms, dtype=np.float32)
    clip_scaled = (
        _scale_unit(np.nan_to_num(clip_array, nan=0.0))
        if np.any(np.isfinite(clip_array))
        else np.zeros_like(cheap_scaled)
    )

    weights = pillar_weight_normaliser(cfg)
    for idx, cand in enumerate(candidates):
        tech_score = float(cand.tech.score)
        creative_score = _aggregate_optional(cand.creative.values())
        subjective_score = _aggregate_optional(cand.subjective.values())
        final_score, pillars = _compute_final_score(tech_score, creative_score, subjective_score, weights)
        cand.final_score = final_score
        cand.pillar_scores = pillars
        if cand.clip_embedding is not None:
            cand.prune_score = 0.6 * clip_scaled[idx] + 0.4 * cheap_scaled[idx]
        else:
            cand.prune_score = cheap_scaled[idx]

    stage = {
        "cheap_norm": summarise_candidates(cheap_scaled),
        "clip_available": int(sum(1 for c in candidates if c.clip_embedding is not None)),
        "clip_status": dict(clip_status),
    }
    return stage


def _prune_candidates(
    candidates: Sequence[Candidate],
    per_scene_candidates: int,
    keep_pct: float,
    top_k_hint: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Retain the strongest candidates per scene to avoid over-selection."""

    per_scene_bins: Dict[int, List[int]] = {}
    for idx, cand in enumerate(candidates):
        per_scene_bins.setdefault(cand.scene_index, []).append(idx)

    per_scene_keep = max(1, int(math.ceil(per_scene_candidates * keep_pct)))
    kept: List[int] = []
    for indices in per_scene_bins.values():
        sorted_scene = sorted(indices, key=lambda i: candidates[i].prune_score, reverse=True)
        count = max(1, min(int(math.ceil(len(indices) * keep_pct)), len(sorted_scene), per_scene_keep))
        kept.extend(sorted_scene[:count])

    if not kept:
        budget = max(1, top_k_hint * 3)
        kept = sorted(
            range(len(candidates)),
            key=lambda i: candidates[i].final_score,
            reverse=True,
        )[:budget]

    stage = {
        "kept": len(kept),
        "total": len(candidates),
        "per_scene_keep_pct": keep_pct,
    }
    return kept, stage


def _select_candidates(
    candidates: Sequence[Candidate],
    kept_indices: Sequence[int],
    cfg: Mapping[str, Any],
) -> Tuple[List[int], Dict[int, int], Dict[str, Any], SelectionDiagnostics]:
    """Run the hybrid selector and return selected indices + diagnostics."""

    select_cfg = cfg.get("select") or {}
    top_k = int(select_cfg.get("top_k", 6))
    overcluster = float(select_cfg.get("overcluster_factor", 4.0))
    per_cluster_top = int(select_cfg.get("per_cluster_top", 2))
    prefer_clip = any(c.clip_embedding is not None for c in candidates)

    items: List[Dict[str, Any]] = []
    for idx in kept_indices:
        cand = candidates[idx]
        item = {"feat": cand.cheap_vector, "score": cand.final_score, "index": idx}
        if cand.clip_embedding is not None:
            item["feat_clip"] = cand.clip_embedding
        items.append(item)

    selected_items, diag = select_hybrid(
        items,
        max(1, top_k),
        overcluster_factor=overcluster,
        per_cluster_top=per_cluster_top,
        prefer_clip=prefer_clip,
        dedup_clip_cosine_min=(cfg.get("dedup") or {}).get("embed_min_cosine_dist", 0.02),
        return_debug=True,
    )

    facility_rank: Dict[int, int] = {}
    cluster_map: Dict[int, int] = {}
    shortlist: set[int] = set()

    if isinstance(diag, SelectionDiagnostics):
        for feature_idx, label in zip(diag.feature_indices, diag.labels):
            if 0 <= feature_idx < len(items):
                cand_idx = int(items[feature_idx]["index"])
                cluster_map[cand_idx] = int(label)
        shortlist = {
            int(items[pos]["index"])
            for pos in getattr(diag, "shortlist_indices", [])
            if 0 <= pos < len(items)
        }

    for rank, item in enumerate(selected_items, start=1):
        facility_rank[int(item["index"])] = rank

    selected_indices = [int(item["index"]) for item in selected_items]
    if not selected_indices:
        selected_indices = sorted(
            kept_indices,
            key=lambda i: candidates[i].final_score,
            reverse=True,
        )[: max(1, top_k)]

    for idx in kept_indices:
        cand = candidates[idx]
        cand.selection_meta["shortlisted"] = idx in shortlist
        if idx in cluster_map:
            cand.selection_meta["cluster_id"] = cluster_map[idx]

    stage = {
        "top_k": top_k,
        "overcluster_k": getattr(diag, "overcluster_k", None),
        "pool": len(items),
        "selected": len(selected_indices),
        "feature_space": getattr(diag, "feature_space", None),
    }

    return selected_indices, facility_rank, stage, diag


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _dhash(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for i, flag in enumerate(diff.flatten()):
        if flag:
            bits |= (1 << i)
    return bits


def _hamming64(a: int, b: int) -> int:
    return int(bin((a ^ b) & ((1 << 64) - 1)).count("1"))


def _deduplicate(
    candidates: Sequence[Candidate],
    selected_indices: Sequence[int],
    cfg: Mapping[str, Any],
) -> Tuple[List[int], Dict[str, Any]]:
    """Apply cosine + dHash deduplication to the selected indices."""

    dedup_cfg = cfg.get("dedup") or {}
    final_indices = list(selected_indices)
    if dedup_cfg.get("enable", True) and len(selected_indices) > 1:
        max_sim = 1.0 - float(dedup_cfg.get("embed_min_cosine_dist", 0.02))
        use_dhash = bool(dedup_cfg.get("dhash", True))
        dhash_thr = int(dedup_cfg.get("dhash_threshold", 6))
        dhashes: Dict[int, int] = {}
        if use_dhash:
            for idx in selected_indices:
                dhashes[idx] = _dhash(candidates[idx].frame)

        ordered = sorted(selected_indices, key=lambda i: candidates[i].final_score, reverse=True)
        deduped: List[int] = []
        for idx in ordered:
            cand = candidates[idx]
            duplicate = False
            for kept in deduped:
                other = candidates[kept]
                if cand.clip_embedding is not None and other.clip_embedding is not None:
                    sim = _cosine_similarity(cand.clip_embedding, other.clip_embedding)
                    if sim > max_sim:
                        cand.selection_meta["dedup_rejected"] = f"clip_sim={sim:.3f}"
                        duplicate = True
                        break
                if use_dhash and not duplicate:
                    dist = _hamming64(dhashes[idx], dhashes[kept])
                    if dist <= dhash_thr:
                        cand.selection_meta["dedup_rejected"] = f"dhash={dist}"
                        duplicate = True
                        break
            if not duplicate:
                deduped.append(idx)
        final_indices = deduped if deduped else ordered[:1]

    if not final_indices:
        final_indices = [
            max(range(len(candidates)), key=lambda i: candidates[i].final_score)
        ]

    stage = {"before": len(selected_indices), "after": len(final_indices)}
    return final_indices, stage


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_image(path: Path, frame: np.ndarray) -> bool:
    if cv2.imwrite(str(path), frame):
        return True
    png_path = path.with_suffix(".png")
    if cv2.imwrite(str(png_path), frame):
        return True
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        path.write_bytes(buf.tobytes())
        return True
    return False


def _candidate_export_payload(cand: Candidate) -> Dict[str, Any]:
    return {
        "frame_index": cand.frame_index,
        "scene_index": cand.scene_index,
        "timestamp": cand.timestamp,
        "score": cand.final_score,
        "scores": cand.pillar_scores,
    }


def _write_outputs(
    candidates: Sequence[Candidate],
    final_indices: Sequence[int],
    context: PipelineContext,
    facility_rank: Mapping[int, int],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Write hero frames + sidecars. Returns written metadata and order."""

    out_dir = context.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    score_prefix = bool(((context.cfg.get("output") or {}).get("score_prefix", True)))
    ordered = sorted(final_indices, key=lambda i: candidates[i].final_score, reverse=True)

    for idx in ordered:
        if idx in facility_rank:
            candidates[idx].selection_meta["facility_rank"] = facility_rank[idx]

    written: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ordered, start=1):
        cand = candidates[idx]
        score_int = int(round(cand.final_score * 100))
        prefix = f"{score_int:03d}_" if score_prefix else ""
        filename = f"{prefix}s{cand.scene_index:03d}_f{cand.frame_index:06d}_r{rank:02d}.jpg"
        path = out_dir / filename
        if not _write_image(path, cand.frame):
            continue
        entry = {
            "path": str(path),
            "score": cand.final_score,
            "scene": cand.scene_index,
            "frame_index": cand.frame_index,
            "timestamp": cand.timestamp,
        }
        sidecar = {
            "schema_version": "1.0",
            "source": {
                "path": context.resolved_source,
                "media_hash": context.media_hash,
                "scene_index": cand.scene_index,
                "frame_index": cand.frame_index,
                "timestamp_sec": cand.timestamp,
            },
            "scores": {"final": cand.final_score, **cand.pillar_scores},
            "metrics": {
                "technical": cand.tech.metrics,
                "advanced": cand.advanced,
                "creative": cand.creative,
                "subjective": cand.subjective,
                "features": {
                    "cheap_norm": float(np.linalg.norm(cand.cheap_vector)),
                    "clip_norm": (
                        float(np.linalg.norm(cand.clip_embedding))
                        if cand.clip_embedding is not None
                        else None
                    ),
                },
            },
            "selection": {
                "rank": rank,
                "prune_score": cand.prune_score,
                **cand.selection_meta,
            },
        }
        write_json(path.with_suffix(".txt"), sidecar)
        written.append(entry)

    if not written:
        idx = max(range(len(candidates)), key=lambda i: candidates[i].final_score)
        cand = candidates[idx]
        fallback_path = out_dir / f"FALLBACK_s{cand.scene_index:03d}_f{cand.frame_index:06d}.jpg"
        if _write_image(fallback_path, cand.frame):
            written.append({"path": str(fallback_path), "score": cand.final_score})
            ordered = [idx]

    return written, ordered


def _build_manifest(
    context: PipelineContext,
    stage_log: Mapping[str, Any],
    written: Sequence[Mapping[str, Any]],
    hero_result: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Path]:
    manifest = {
        "schema_version": "1.0",
        "source": stage_log.get("ingest", {}),
        "config_signature": context.cfg_signature,
        "random_seed": context.seed,
        "scenes": stage_log.get("scene_detection", {}),
        "sampling": stage_log.get("sampling", {}),
        "features": stage_log.get("features", {}),
        "selection": stage_log.get("selection", {}),
        "dedup": stage_log.get("dedup", {}),
        "outputs": {
            "frames": list(written),
            "hero_video": dict(hero_result),
        },
    }
    manifest_path = context.output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    return manifest, manifest_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    source: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    progress_cb: ProgressCb = None,
    cancel_fn: CancelFn = None,
) -> Dict[str, Any]:
    """Execute the hero-frame pipeline and return a manifest summary."""

    progress_cb = progress_cb or (lambda frac, msg: None)
    cancel_fn = cancel_fn or (lambda: False)

    context = _initialise_context(source, config, config_path, cli_overrides)
    stage_log: Dict[str, Any] = {}

    progress_cb(0.02, "opening source")
    cap, meta = _open_capture(context.resolved_source)

    stage_log["ingest"] = {
        "source": context.resolved_source,
        "media_hash": context.media_hash,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "resolution": {"width": meta.width, "height": meta.height},
        "duration_sec": meta.duration,
    }

    try:
        progress_cb(0.06, "detecting scenes")
        scene_diag = detect_scenes(context.resolved_source, context.cfg)
        stage_log["scene_detection"] = {
            "method": scene_diag.method,
            "threshold": scene_diag.threshold,
            "downscale": scene_diag.downscale,
            "fallback_used": scene_diag.fallback_used,
            "count": scene_diag.total_scenes,
            "spans": spans_to_metadata(scene_diag.spans),
        }

        extract_cfg = context.cfg.get("extract") or {}
        sampling_cfg = context.cfg.get("sampling") or {}
        per_scene_candidates = int(extract_cfg.get("per_scene_candidates", 9))
        keep_pct = float(extract_cfg.get("per_scene_keep_pct", 0.4))
        jitter_frac = float(sampling_cfg.get("jitter_frac", 0.12))
        min_gap = int(extract_cfg.get("min_candidate_gap_frames", 6))

        indices = indices_from_scene_spans(
            spans=scene_diag.spans,
            per_scene_candidates=per_scene_candidates,
            jitter_frac=jitter_frac,
            min_gap=min_gap,
        )
        stage_log["sampling"] = {
            "total_candidates": len(indices),
            "per_scene_candidates": per_scene_candidates,
            "jitter_frac": jitter_frac,
            "min_gap": min_gap,
        }

        progress_cb(0.12, f"sampling {len(indices)} candidates")
        resize_width = (context.cfg.get("processing") or {}).get("resize_width")
        candidates, cheap_norms = _extract_candidates(
            cap,
            indices,
            scene_diag.spans,
            meta,
            context,
            resize_width,
            progress_cb,
            cancel_fn,
        )
    finally:
        cap.release()

    clip_signature = stable_config_signature(
        {
            "clip": (context.cfg.get("heavy") or {}).get("clip", {}),
            "gpu": context.cfg.get("gpu", {}),
        }
    )
    clip_status = _embed_clip_embeddings(candidates, context, clip_signature)

    stage_log["features"] = _score_candidates(candidates, cheap_norms, clip_status, context.cfg)
    progress_cb(0.45, "computed pillar scores")

    select_cfg = context.cfg.get("select") or {}
    top_k_hint = int(select_cfg.get("top_k", 6))
    kept_indices, prune_stage = _prune_candidates(
        candidates,
        per_scene_candidates,
        keep_pct,
        top_k_hint,
    )
    stage_log["prune"] = prune_stage

    selected_indices, facility_rank, selection_stage, diag = _select_candidates(
        candidates,
        kept_indices,
        context.cfg,
    )
    stage_log["selection"] = selection_stage
    progress_cb(0.65, f"selected {len(selected_indices)} candidates")

    final_indices, dedup_stage = _deduplicate(candidates, selected_indices, context.cfg)
    stage_log["dedup"] = dedup_stage
    progress_cb(0.75, f"deduped to {len(final_indices)} frames")

    written, ordered_indices = _write_outputs(candidates, final_indices, context, facility_rank)
    progress_cb(0.88, f"wrote {len(written)} frames")

    info = VideoInfo(fps=meta.fps, frame_count=meta.frame_count, duration=meta.duration)
    hero_cfg = ((context.cfg.get("output") or {}).get("hero_video") or {})
    frames_for_sheet = [candidates[i].frame for i in ordered_indices] if hero_cfg.get("contact_sheet", True) else None
    hero_result = export_hero_video(
        context.resolved_source,
        context.output_dir,
        [_candidate_export_payload(candidates[i]) for i in ordered_indices],
        scene_diag.spans,
        info,
        hero_cfg,
        frames=frames_for_sheet,
    )
    progress_cb(0.94, "hero export complete")

    stage_log["export"] = {
        "enabled": bool(hero_cfg.get("enabled", True)),
        "segments": len(hero_result.get("segments", [])) if hero_result else 0,
        "ffmpeg_ok": bool(hero_result.get("ffmpeg_ok", False)) if hero_result else False,
    }

    manifest, manifest_path = _build_manifest(context, stage_log, written, hero_result or {})
    progress_cb(1.0, "pipeline complete")

    return {
        "output_dir": str(context.output_dir),
        "written": written,
        "hero_video": hero_result,
        "manifest": str(manifest_path),
        "stage_log": stage_log,
        "clip_status": clip_status,
        "selector_diag": diag,
    }


__all__ = [
    "run_pipeline",
    "Candidate",
    "PipelineContext",
    "VideoMetadata",
    "_cosine_similarity",
    "_dhash",
    "_hamming64",
]