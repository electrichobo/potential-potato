# -*- coding: utf-8 -*-
from __future__ import annotations
# AESTHETIC — selector:
# k-medoids (overcluster) ➜ facility-location,
# with smart feature picking: prefer CLIP embeddings when present.

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


# ---------- small helpers ----------
def _l2norm(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalization with tiny epsilon and NaN/Inf guards."""
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < eps, eps, n)
    Y = X / n
    Y[~np.isfinite(Y)] = 0.0
    return Y

def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix with numerical safety."""
    X = np.asarray(X, dtype=np.float32)
    G = X @ X.T
    nrm = np.sum(X * X, axis=1, keepdims=True)
    D = nrm + nrm.T - 2.0 * G
    # clamp tiny negatives to zero and clean NaNs/Infs
    np.maximum(D, 0.0, out=D)
    D[~np.isfinite(D)] = 0.0
    return D

def _init_ff(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-first seeding for k-medoids (robust, no probabilities)."""
    n = X.shape[0]
    k = max(1, min(k, n))
    medoids = [int(rng.integers(0, n))]
    D = _pairwise_sq_dists(X)
    min_d = D[medoids[0]].copy()
    for _ in range(1, k):
        j = int(np.argmax(min_d))
        medoids.append(j)
        # track distance to nearest chosen medoid
        min_d = np.minimum(min_d, D[j])
    return np.asarray(medoids, dtype=np.int32)

def _kmedoids_labels(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42) -> np.ndarray:
    """Simple PAM-like k-medoids to get cluster labels (we don’t need medoids back)."""
    n = X.shape[0]
    k = max(1, min(k, n))
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    rng = np.random.default_rng(seed)
    D = _pairwise_sq_dists(X)
    medoids = _init_ff(X, k, rng)
    labels = np.argmin(D[:, medoids], axis=1)
    for _ in range(max_iter):
        changed = False
        for c in range(k):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                continue
            subD = D[np.ix_(idxs, idxs)]
            costs = subD.sum(axis=1)
            best_local = idxs[int(np.argmin(costs))]
            if medoids[c] != best_local:
                medoids[c] = best_local
                changed = True
        new_labels = np.argmin(D[:, medoids], axis=1)
        if not np.any(new_labels != labels):
            if not changed:
                break
        labels = new_labels
    return labels

def _cosine_sim(X: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (clamped/cleaned for safety)."""
    Xn = _l2norm(X)
    S = Xn @ Xn.T
    S = np.clip(S, -1.0, 1.0)
    S[~np.isfinite(S)] = 0.0
    return S

def _facility_location(S: np.ndarray, candidates: List[int], k: int) -> List[int]:
    """Greedy facility-location on a similarity matrix S for a candidate index set."""
    k = max(1, min(k, len(candidates)))
    if k >= len(candidates):
        return candidates[:]
    selected: List[int] = []
    covered = np.zeros(S.shape[0], dtype=np.float32)
    pool = candidates.copy()
    # linear greedy (sufficient for small pools)
    for _ in range(k):
        best_idx, best_gain = -1, -1.0
        base_sum = float(covered.sum())
        for j in pool:
            m = np.maximum(covered, S[:, j])
            gain = float(m.sum() - base_sum)
            if gain > best_gain:
                best_gain, best_idx = gain, j
        if best_idx < 0:
            break
        covered = np.maximum(covered, S[:, best_idx])
        selected.append(best_idx)
        pool.remove(best_idx)
        if not pool:
            break
    return selected


# ---------- feature gathering ----------
def _gather_features(
    items: List[Dict[str, Any]],
    k: int,
    prefer_clip: bool,
    quality_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Decide which feature space to use (CLIP vs cheap), then collect:
      X  = (N, D) float32 feature matrix
      q  = (N,) quality scores (float32)
      idx= (N,) original indices
      key_used = "feat_clip" or "feat"
    Rule: if prefer_clip and we have at least max(k, 5) valid clip features,
          use CLIP; otherwise fall back to "feat".
    """
    def _collect(key: str):
        feats, quals, keep = [], [], []
        for i, it in enumerate(items):
            f = it.get(key, None)
            if f is None:
                continue
            arr = np.asarray(f, dtype=np.float32)
            if arr.ndim == 1:
                # ensure row-shape (1, D)
                feats.append(arr)
                quals.append(float(it.get(quality_key, 0.0)))
                keep.append(i)
        if not feats:
            return None
        return np.vstack(feats), np.asarray(quals, np.float32), np.asarray(keep, np.int64)

    use_key = "feat_clip" if prefer_clip else "feat"
    got = _collect(use_key)

    if prefer_clip:
        enough = (got is not None) and (got[0].shape[0] >= max(5, int(k)))
        if not enough:
            use_key = "feat"
            got = _collect(use_key)

    if got is None:
        # Final fallback: nothing available -> return empty
        return (np.empty((0, 0), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int64),
                use_key)

    X, q, idx = got
    return X, q, idx, use_key


def _prefilter_clip_dedup(
    X: np.ndarray, q: np.ndarray, idx: np.ndarray, min_cos_dist: Optional[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optional pre-filter on CLIP features: sort by quality and drop near-duplicates
    with cosine distance < min_cos_dist.
    """
    if min_cos_dist is None or X.size == 0:
        return X, q, idx

    # Normalize for cosine
    Xn = _l2norm(X)
    order = np.argsort(-q)  # high quality first
    keep: List[int] = []
    for i in order:
        xi = Xn[i]
        ok = True
        for j in keep:
            # cosine distance = 1 - dot
            if 1.0 - float(np.dot(xi, Xn[j])) < float(min_cos_dist):
                ok = False
                break
        if ok:
            keep.append(i)
    keep = np.asarray(keep, dtype=int)
    return X[keep], q[keep], idx[keep]


# ---------- main entry ----------
def select_hybrid(
    items: List[Dict[str, Any]],
    k: int,
    *,
    feat_key: str = "feat",             # kept for backward compatibility (unused if prefer_clip=True)
    quality_key: str = "score",
    overcluster_factor: float = 3.0,
    per_cluster_top: int = 3,
    min_quality: Optional[float] = None,
    prefer_clip: bool = True,           # prefer CLIP if enough available
    dedup_clip_cosine_min: Optional[float] = None,  # optional pre-cluster CLIP de-dup
) -> List[Dict[str, Any]]:
    """
    Select k diverse high-quality items using (overclustered) k-medoids + facility-location.
    - If prefer_clip is True and enough 'feat_clip' exist, selection runs in CLIP space.
      Otherwise it falls back to 'feat' (cheap 4-D feature).
    """
    if k <= 0:
        return []
    if len(items) <= k:
        return items

    # 1) choose feature space & gather
    X, q, idx, key_used = _gather_features(items, k, prefer_clip, quality_key)
    if X.size == 0:  # no features anywhere → quality sort fallback
        order = np.argsort([float(d.get(quality_key, 0.0)) for d in items])[::-1]
        return [items[i] for i in order[:k]]

    # 2) optional hard quality floor
    if min_quality is not None:
        sel = np.where(q >= float(min_quality))[0]
        if sel.size > 0:
            X, q, idx = X[sel], q[sel], idx[sel]

    # 3) optional pre-cluster CLIP de-dup
    if key_used == "feat_clip":
        X, q, idx = _prefilter_clip_dedup(X, q, idx, dedup_clip_cosine_min)

    if X.shape[0] <= k:
        return [items[int(i)] for i in idx]

    # 4) overcluster in chosen space
    m = int(np.ceil(overcluster_factor * k))
    m = max(1, min(m, X.shape[0]))
    labels = _kmedoids_labels(X, m, max_iter=50, seed=42)

    # 5) take top-q per cluster to form pool
    pooled: List[int] = []
    for c in range(m):
        loc = np.where(labels == c)[0]
        if loc.size == 0:
            continue
        order = loc[np.argsort(q[loc])[::-1]]
        pooled.extend(order[:min(per_cluster_top, order.size)].tolist())

    # dedupe while preserving order
    pooled = list(dict.fromkeys(pooled))
    if not pooled:
        # pathological: fall back to best k by q
        best = np.argsort(q)[::-1][:k]
        return [items[int(idx[i])] for i in best]

    # 6) facility-location on the pool (cosine sim in current space)
    Xp = X[pooled]
    Sp = _cosine_sim(Xp)
    cand = list(range(len(pooled)))
    sel_local = _facility_location(Sp, cand, k)

    final_idx = [int(idx[pooled[i]]) for i in sel_local]
    return [items[j] for j in final_idx]
