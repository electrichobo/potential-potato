# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List, Tuple
import cv2, numpy as np, logging, math, json, random
from pathlib import Path

from aesthetic.core.ingest import resolve_source
from aesthetic.features.deep_clip import embed_frames_if_enabled
from aesthetic.core.features_basic import compute_score_and_metrics

LOG = logging.getLogger("aesthetic.pipeline")

ProgressCb = Optional[Callable[[float, str], None]]
CancelFn   = Optional[Callable[[], bool]]

# ---------- utils ----------
def _scale_scores_0_100(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    smin, smax = float(np.nanmin(scores)), float(np.nanmax(scores))
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        return np.full_like(scores, 50.0)
    out = (scores - smin) / (smax - smin) * 100.0
    out[~np.isfinite(out)] = 50.0
    return out

def _l2norm_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-normalize with NaN/Inf protection."""
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < eps, eps, n)
    Y = X / n
    Y[~np.isfinite(Y)] = 0.0
    return Y

def _cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity on row-normalized A, B; clamps/cleans output."""
    S = A @ B.T
    S = np.clip(S, -1.0, 1.0)
    S[~np.isfinite(S)] = 0.0
    return S

def _resize_w(frame_bgr: np.ndarray, target_w: Optional[int]) -> np.ndarray:
    if not target_w or target_w <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= target_w:
        return frame_bgr
    s = target_w / float(w)
    return cv2.resize(frame_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def _ensure_dirs(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    out_dir = Path(((cfg.get("output") or {}).get("folder")) or "aesthetic/outputs").resolve()
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, log_dir

def _write_sidecar_txt(path: Path, payload: Dict[str, Any]) -> None:
    txt = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(txt, encoding="utf-8")

# ---------- simple content scene detect ----------
def _scene_breaks(cap: cv2.VideoCapture, cfg: Dict[str, Any], target_w: int) -> List[int]:
    scenes_cfg = cfg.get("scenes") or {}
    thr = float(scenes_cfg.get("threshold", 27.0))
    down = int(scenes_cfg.get("downscale", 2))
    min_len = int((cfg.get("extract") or {}).get("min_scene_len_frames", 12))

    breaks = [0]
    ok, prev = cap.read()
    if not ok or prev is None:
        return [0]
    prev = _resize_w(prev, max(64, target_w // max(1, down)))
    prev_h = cv2.calcHist([cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    prev_h = cv2.normalize(prev_h, None).flatten()

    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        small = _resize_w(frame, max(64, target_w // max(1, down)))
        h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        h = cv2.normalize(h, None).flatten()
        diff = cv2.compareHist(prev_h, h, cv2.HISTCMP_BHATTACHARYYA) * 100.0  # 0..100
        if diff >= thr and (idx - breaks[-1]) >= min_len:
            breaks.append(idx)
            prev_h = h
        idx += 1

    breaks.append(idx - 1)
    cleaned = [breaks[0]]
    for b in breaks[1:]:
        if b - cleaned[-1] >= min_len:
            cleaned.append(b)
    return cleaned

# ---------- cheap per-frame feature ----------
def _cheap_feats(frames: List[np.ndarray]) -> np.ndarray:
    """
    Simple 4-D features per frame:
    [mean(gray), std(gray), Laplacian var (sharpness), mean saturation]
    """
    feats = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std  = float(gray.std())
        sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        sat = float(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1].mean())
        feats.append([mean, std, sharp, sat])
    return np.asarray(feats, dtype=np.float32)

# ---------- combined feature (cheap + optional CLIP) ----------
def _combined_feature(cheap: np.ndarray, clip: Optional[np.ndarray]) -> np.ndarray:
    """Normalize blocks then concat → normalize again (cosine-friendly)."""
    Cq = _l2norm_rows(cheap.astype(np.float32))
    if clip is None:
        return Cq
    Cc = _l2norm_rows(clip.astype(np.float32))
    return _l2norm_rows(np.concatenate([Cq, Cc], axis=1))

# ---------- per-scene micro-pool with jitter & min-gap ----------
def _sample_scene_frames(start: int, end: int, per_scene_candidates: int, jitter_frac: float, gap: int) -> List[int]:
    if end <= start:
        return []
    length = end - start + 1
    step = max(1, length // max(1, per_scene_candidates))
    picks: List[int] = []
    for i in range(per_scene_candidates):
        base = start + i * step + step // 2
        jitter = int(step * jitter_frac)
        j = random.randint(-jitter, jitter) if jitter > 0 else 0
        f = max(start, min(end, base + j))
        if not picks or (f - picks[-1]) >= gap:
            picks.append(f)
    return sorted(set(picks))

# ---------- k-medoids (PAM-lite, robust) ----------
def _pam_k_medoids(X: np.ndarray, k: int, iters: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (medoid_indices, labels)
    X must be row-normalized. Distance = 1 - cosine_similarity.
    Robust to degenerate/duplicate points (guards probabilities).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    if k >= n:
        medoids = np.arange(n, dtype=np.int32)
        labels = np.arange(n, dtype=np.int32)
        return medoids, labels

    sim = _cosine_sim(X, X)  # n x n

    # kmeans++-like init over cosine distance d = 1 - best_sim
    medoids = [int(rng.integers(n))]
    for _ in range(1, k):
        best_sim = np.max(sim[:, medoids], axis=1)
        d = 1.0 - best_sim                   # [0, 2]
        d[~np.isfinite(d)] = 0.0
        d = np.maximum(d, 0.0)
        total = float(d.sum())
        if total <= 1e-12:
            # distribution collapsed → pick a random unused index
            candidates = [i for i in range(n) if i not in medoids]
            medoids.append(int(rng.choice(candidates)))
            continue
        probs = d / total
        probs = np.maximum(probs, 0.0)
        probs = probs / (probs.sum() + 1e-12)
        medoids.append(int(rng.choice(n, p=probs)))
    medoids = np.asarray(medoids, dtype=np.int32)

    # PAM-lite updates
    for _ in range(max(1, iters)):
        sim_to_m = sim[:, medoids]                # n x k
        labels = np.argmax(sim_to_m, axis=1)
        for j in range(k):
            members = np.where(labels == j)[0]
            if members.size == 0:
                continue
            sub = sim[np.ix_(members, members)]
            best_local = members[np.argmax(sub.sum(axis=1))]
            medoids[j] = int(best_local)

    labels = np.argmax(sim[:, medoids], axis=1)
    return medoids, labels

# ---------- facility-location greedy ----------
def _facility_location_greedy(Sim: np.ndarray, budget: int) -> List[int]:
    """
    Greedy maximize F(A) = sum_i max_{j in A} Sim[i,j]
    Sim: n x n cosine-similarity (row/col aligned)
    """
    n = Sim.shape[0]
    if budget >= n:
        return list(range(n))
    covered = np.zeros(n, dtype=np.float32)
    chosen: List[int] = []
    for _ in range(budget):
        best_idx, best_gain = -1, -1.0
        base_sum = float(covered.sum())
        # linear greedy; good enough for our pool sizes
        for j in range(n):
            if j in chosen:
                continue
            m = np.maximum(covered, Sim[:, j])
            gain = float(m.sum() - base_sum)
            if gain > best_gain:
                best_gain, best_idx = gain, j
        if best_idx < 0:
            break
        covered = np.maximum(covered, Sim[:, best_idx])
        chosen.append(best_idx)
    return chosen

# ---------- perceptual dHash (optional dedup helper) ----------
def _dhash_8(f_bgr: np.ndarray) -> int:
    """8x8 dHash → 64-bit int."""
    small = cv2.resize(cv2.cvtColor(f_bgr, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            bits |= (1 << i)
    return bits

def _hamming64(a: int, b: int) -> int:
    return int(bin((a ^ b) & ((1 << 64) - 1)).count("1"))

# ---------- main ----------
def run_pipeline(
    source: str,
    config: Optional[Dict[str, Any]] = None,
    progress_cb: ProgressCb = None,
    cancel_fn: CancelFn = None,
) -> None:
    cfg: Dict[str, Any] = config or {}
    progress_cb = progress_cb or (lambda f, m: None)
    cancel_fn = cancel_fn or (lambda: False)
    out_dir, _ = _ensure_dirs(cfg)

    # resolve & open
    resolved = resolve_source(source, config=cfg)
    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source: {resolved}")

    processing = cfg.get("processing") or {}
    target_w   = int(processing.get("resize_width", 720))
    random.seed(int((cfg.get("sampling") or {}).get("random_seed", 42)))

    progress_cb(0.05, "detecting scenes")
    breaks = _scene_breaks(cap, cfg, target_w)  # indices in decode order
    LOG.info("scenes: %s", breaks)
    cap.release(); cap = cv2.VideoCapture(str(resolved))

    extract = cfg.get("extract") or {}
    per_scene_candidates = int(extract.get("per_scene_candidates", 7))
    keep_pct = extract.get("per_scene_keep_pct", None)
    per_scene_keep = int(extract.get("per_scene_keep", 3))
    min_gap = int(extract.get("min_candidate_gap_frames", 6))
    jitter = float((cfg.get("sampling") or {}).get("jitter_frac", 0.12))

    all_frames: List[np.ndarray] = []
    all_meta: List[Dict[str, Any]] = []
    tech_scores: List[float] = []
    tech_metrics: List[Dict[str, Any]] = []

    # per-scene micro-pools
    for s in range(len(breaks) - 1):
        start, end = breaks[s], breaks[s + 1]
        picks = _sample_scene_frames(start, end, per_scene_candidates, jitter, min_gap)
        for fidx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fidx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = _resize_w(frame, target_w)
            all_frames.append(frame)
            all_meta.append({"scene": s, "frame_index": int(fidx)})

            # compute once per sampled frame (technical score/metrics)
            sc, mt = compute_score_and_metrics(frame)
            tech_scores.append(float(sc))
            tech_metrics.append(mt)

    cap.release()
    if not all_frames:
        raise RuntimeError("no frames sampled from scenes")
    progress_cb(0.25, f"sampled {len(all_frames)} candidates from {len(breaks)-1} scenes")

    # ---------- features ----------
    cheap = _cheap_feats(all_frames)
    LOG.info("features: cheap=%s", cheap.shape)

    clip_arr = None
    clip_out = embed_frames_if_enabled(all_frames, cfg)
    if clip_out and "feat_clip" in clip_out:
        clip_arr = clip_out["feat_clip"]
        LOG.info("features: feat_clip attached shape=%s", getattr(clip_arr, "shape", None))
    else:
        LOG.info("features: no CLIP embeddings (continuing with cheap only)")

    # per-scene ranking & keep (blend used only for pruning)
    per_scene_bins: Dict[int, List[int]] = {}
    for i, m in enumerate(all_meta):
        per_scene_bins.setdefault(m["scene"], []).append(i)

    # blend: prefer CLIP when present
    q = np.linalg.norm(cheap, axis=1)
    if clip_arr is not None:
        c = np.linalg.norm(clip_arr, axis=1)
        s_blend = 0.6 * _scale_scores_0_100(c) + 0.4 * _scale_scores_0_100(q)
    else:
        s_blend = _scale_scores_0_100(q)

    kept_indices: List[int] = []
    for scn, idxs in per_scene_bins.items():
        k = int(math.ceil(len(idxs) * float(keep_pct))) if keep_pct is not None else per_scene_keep
        k = max(1, min(k, len(idxs)))
        order = sorted(idxs, key=lambda i: s_blend[i], reverse=True)[:k]
        kept_indices.extend(order)
    LOG.info("keep: scenes=%d -> kept=%d of %d", len(per_scene_bins), len(kept_indices), len(all_frames))
    if not kept_indices:
        # hard fallback: take best N globally by blend
        N = max(1, min(int((cfg.get("select") or {}).get("top_k", 6))*3, len(all_frames)))
        kept_indices = np.argsort(-s_blend)[:N].tolist()
        LOG.warning("keep: empty after per-scene; falling back to global top-%d", N)

    # combined features (row-normalized) for selection
    F_all = _combined_feature(cheap, clip_arr)
    F = F_all[kept_indices]
    LOG.info("select: feature_matrix=%s", F.shape)

    # k-medoids overcluster
    select_cfg = cfg.get("select") or {}
    top_k      = int(select_cfg.get("top_k", 6))
    over       = float(select_cfg.get("overcluster_factor", 4.0))
    per_cluster_top = int(select_cfg.get("per_cluster_top", 2))
    K = max(1, int(min(len(F), max(1, int(round(top_k * over))))))

    progress_cb(0.55, f"k-medoids overcluster (k={K})")
    medoids, labels = _pam_k_medoids(F, k=K, iters=5, seed=42)
    LOG.info("kmedoids: labels=%s unique_clusters=%d", labels.shape, len(set(labels.tolist())) if labels.size else 0)

    # per-cluster shortlist by vector norm
    norms = np.linalg.norm(F, axis=1)
    prelim: List[int] = []
    for j in range(K):
        members = np.where(labels == j)[0]
        if members.size == 0:
            continue
        order = members[np.argsort(-norms[members])]
        prelim.extend(order[:per_cluster_top])
    prelim = sorted(set(prelim))
    if len(prelim) == 0:
        prelim = medoids.tolist()
    LOG.info("prelim: %d", len(prelim))

    # facility location on prelim with budget top_k
    if len(prelim) <= top_k:
        chosen_local = prelim
    else:
        P = F[prelim]
        Sim = _cosine_sim(P, P)
        chosen_local_idx = _facility_location_greedy(Sim, budget=max(1, top_k))
        chosen_local = [prelim[i] for i in chosen_local_idx]
    LOG.info("chosen_local: %d (budget=%d)", len(chosen_local), top_k)

    # map back to global and tie-break by score
    final_global = sorted([kept_indices[i] for i in chosen_local],
                          key=lambda i: s_blend[i], reverse=True)[:max(1, top_k)]
    LOG.info("final_global initial: %d", len(final_global))

    # safety net: if still empty, just take top_k by score from kept pool
    if not final_global:
        final_global = sorted(kept_indices, key=lambda i: s_blend[i], reverse=True)[:max(1, top_k)]
        LOG.warning("final_global: empty after selection; falling back to score-top-%d", len(final_global))

    progress_cb(0.70, f"facility-location select → {len(final_global)}")

    # optional de-dup
    dedup_cfg = cfg.get("dedup") or {}
    do_dedup = bool(dedup_cfg.get("enable", True))
    if do_dedup and len(final_global) > 1:
        keep: List[int] = []
        min_cos_dist = float(dedup_cfg.get("embed_min_cosine_dist", 0.02))
        max_sim_allowed = 1.0 - max(0.0, min_cos_dist)
        use_dhash = bool(dedup_cfg.get("dhash", True))
        dhash_thr = int(dedup_cfg.get("dhash_threshold", 6))
        dhashes = {gi: _dhash_8(all_frames[gi]) for gi in final_global} if use_dhash else {}

        for gi in final_global:
            if not keep:
                keep.append(gi); continue
            f_vec = F_all[gi][None, :]
            sims = _cosine_sim(f_vec, F_all[np.array(keep)]).flatten()
            if float(np.max(sims)) > max_sim_allowed:
                continue
            if use_dhash:
                h = dhashes[gi]
                if any(_hamming64(h, dhashes[kj]) <= dhash_thr for kj in keep):
                    continue
            keep.append(gi)
        LOG.info("dedup: %d -> %d", len(final_global), len(keep))
        final_global = keep
        progress_cb(0.80, f"dedup → {len(final_global)}")

    # ---- WRITE OUTPUTS (with resilient write) ----
    written = 0
    score_prefix = bool((cfg.get("output") or {}).get("score_prefix", True))

    def _safe_write(path: Path, img: np.ndarray) -> bool:
        ok = cv2.imwrite(str(path), img)
        if ok:
            return True
        try:
            ok2, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok2:
                path.write_bytes(buf.tobytes())
                return True
        except Exception as e:
            LOG.error("imencode fallback failed for %s: %s", path, e)
        return False

    # absolute safety: if everything above collapses, pick the best-scored frame
    if not final_global and len(all_frames) > 0:
        final_global = [int(np.argmax(s_blend))]
        LOG.warning("final_global empty after dedup; using single best-scored frame.")

    for rank, gi in enumerate(final_global, start=1):
        meta = all_meta[gi]
        frame = all_frames[gi]
        scr = float(s_blend[gi])
        prefix = f"{int(round(scr)):03d}_" if score_prefix else ""
        fname = f"{prefix}s{meta['scene']:03d}_f{meta['frame_index']:06d}_r{rank:02d}.jpg"
        fpath = out_dir / fname

        if _safe_write(fpath, frame):
            written += 1
        else:
            LOG.error("write failed: %s", fpath)

        side = {
            "scene": meta["scene"],
            "frame_index": meta["frame_index"],
            "rank": rank,
            "score_blend": scr,
            "tech": {
                "score": float(tech_scores[gi]) if gi < len(tech_scores) else None,
                **(tech_metrics[gi] if gi < len(tech_metrics) else {}),
            },
            "features": {
                "cheap_norm": float(np.linalg.norm(cheap[gi])),
                "clip_norm": float(np.linalg.norm(clip_arr[gi])) if clip_arr is not None else None,
            },
            "selection": {
                "kmedoids_k": K,
                "overcluster_factor": over,
                "per_cluster_top": per_cluster_top,
                "facility_budget": top_k
            }
        }
        try:
            _write_sidecar_txt(fpath.with_suffix(".txt"), side)
        except Exception as e:
            LOG.error("sidecar write failed: %s", e)

    # last-resort: ensure at least one file exists
    if written == 0 and len(all_frames) > 0:
        gi = int(np.argmax(s_blend))
        meta = all_meta[gi]; frame = all_frames[gi]; scr = float(s_blend[gi])
        fpath = out_dir / f"FALLBACK_s{meta['scene']:03d}_f{meta['frame_index']:06d}.jpg"
        if _safe_write(fpath, frame):
            written = 1
            LOG.warning("no images written earlier; wrote hard fallback: %s", fpath)

    LOG.info("written: %d", written)
    progress_cb(1.0, f"wrote {written} frames + sidecars")
