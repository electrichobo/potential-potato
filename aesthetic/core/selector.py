"""Selection helpers: over-clustered k-medoids + facility location."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Low-level maths utilities
# ---------------------------------------------------------------------------


def _l2norm(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < eps, eps, n)
    Y = X / n
    Y[~np.isfinite(Y)] = 0.0
    return Y


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    G = X @ X.T
    nrm = np.sum(X * X, axis=1, keepdims=True)
    D = nrm + nrm.T - 2.0 * G
    np.maximum(D, 0.0, out=D)
    D[~np.isfinite(D)] = 0.0
    return D


def _init_ff(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    k = max(1, min(k, n))
    medoids = [int(rng.integers(0, n))]
    D = _pairwise_sq_dists(X)
    min_d = D[medoids[0]].copy()
    for _ in range(1, k):
        j = int(np.argmax(min_d))
        medoids.append(j)
        min_d = np.minimum(min_d, D[j])
    return np.asarray(medoids, dtype=np.int32)


def _kmedoids_labels(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    k = max(1, min(k, n))
    if n == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
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
    return labels, medoids


def _cosine_sim(X: np.ndarray) -> np.ndarray:
    Xn = _l2norm(X)
    S = Xn @ Xn.T
    S = np.clip(S, -1.0, 1.0)
    S[~np.isfinite(S)] = 0.0
    return S


def _facility_location(S: np.ndarray, candidates: List[int], k: int) -> List[int]:
    k = max(1, min(k, len(candidates)))
    if k >= len(candidates):
        return candidates[:]
    selected: List[int] = []
    covered = np.zeros(S.shape[0], dtype=np.float32)
    pool = candidates.copy()
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


# ---------------------------------------------------------------------------
# Diagnostics container
# ---------------------------------------------------------------------------


@dataclass
class SelectionDiagnostics:
    feature_space: str
    overcluster_k: int
    per_cluster_top: int
    pool_size: int
    facility_budget: int
    labels: List[int]
    medoid_indices: List[int]
    shortlist_indices: List[int]
    selected_indices: List[int]
    prefilter_size: int
    feature_indices: List[int]


# ---------------------------------------------------------------------------
# Feature gathering utilities
# ---------------------------------------------------------------------------


def _gather_features(
    items: List[Dict[str, Any]],
    k: int,
    prefer_clip: bool,
    quality_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    def _collect(key: str):
        feats, quals, keep = [], [], []
        for i, it in enumerate(items):
            f = it.get(key)
            if f is None:
                continue
            arr = np.asarray(f, dtype=np.float32)
            if arr.ndim != 1:
                continue
            feats.append(arr)
            quals.append(float(it.get(quality_key, 0.0)))
            keep.append(i)
        if not feats:
            return None
        return np.vstack(feats), np.asarray(quals, np.float32), np.asarray(keep, np.int64)

    use_key = "feat_clip" if prefer_clip else "feat"
    got = _collect(use_key)
    if prefer_clip:
        enough = got is not None and got[0].shape[0] >= max(5, int(k))
        if not enough:
            use_key = "feat"
            got = _collect(use_key)

    if got is None:
        return (
            np.empty((0, 0), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int64),
            use_key,
        )

    X, q, idx = got
    return X, q, idx, use_key


def _prefilter_clip_dedup(
    X: np.ndarray, q: np.ndarray, idx: np.ndarray, min_cos_dist: Optional[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min_cos_dist is None or X.size == 0:
        return X, q, idx
    Xn = _l2norm(X)
    order = np.argsort(-q)
    keep: List[int] = []
    for i in order:
        xi = Xn[i]
        ok = True
        for j in keep:
            if 1.0 - float(np.dot(xi, Xn[j])) < float(min_cos_dist):
                ok = False
                break
        if ok:
            keep.append(i)
    keep = np.asarray(keep, dtype=int)
    return X[keep], q[keep], idx[keep]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def select_hybrid(
    items: List[Dict[str, Any]],
    k: int,
    *,
    quality_key: str = "score",
    overcluster_factor: float = 3.0,
    per_cluster_top: int = 3,
    min_quality: Optional[float] = None,
    prefer_clip: bool = True,
    dedup_clip_cosine_min: Optional[float] = None,
    return_debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[SelectionDiagnostics]]:
    diag = SelectionDiagnostics(
        feature_space="feat",
        overcluster_k=0,
        per_cluster_top=int(per_cluster_top),
        pool_size=len(items),
        facility_budget=int(k),
        labels=[],
        medoid_indices=[],
        shortlist_indices=[],
        selected_indices=[],
        prefilter_size=0,
        feature_indices=[],
    )

    if k <= 0:
        return [], diag if return_debug else None
    if len(items) <= k:
        diag.selected_indices = list(range(len(items)))
        return items, diag if return_debug else None

    X, q, idx, key_used = _gather_features(items, k, prefer_clip, quality_key)
    diag.feature_space = key_used
    diag.prefilter_size = int(len(idx))
    diag.feature_indices = idx.tolist()
    if X.size == 0:
        order = np.argsort([float(d.get(quality_key, 0.0)) for d in items])[::-1][:k]
        diag.selected_indices = order.tolist()
        return [items[i] for i in order], diag if return_debug else None

    if min_quality is not None:
        sel = np.where(q >= float(min_quality))[0]
        if sel.size > 0:
            X, q, idx = X[sel], q[sel], idx[sel]

    if key_used == "feat_clip":
        X, q, idx = _prefilter_clip_dedup(X, q, idx, dedup_clip_cosine_min)

    if X.shape[0] <= k:
        diag.shortlist_indices = idx.tolist()
        diag.selected_indices = idx.tolist()
        return [items[int(i)] for i in idx], diag if return_debug else None

    m = int(np.ceil(overcluster_factor * k))
    m = max(1, min(m, X.shape[0]))
    labels, medoids = _kmedoids_labels(X, m, max_iter=50, seed=42)
    diag.overcluster_k = int(m)
    diag.labels = labels.tolist()
    diag.medoid_indices = medoids.tolist()

    pooled: List[int] = []
    for c in range(m):
        loc = np.where(labels == c)[0]
        if loc.size == 0:
            continue
        order = loc[np.argsort(q[loc])[::-1]]
        pooled.extend(order[:min(per_cluster_top, order.size)])
    pooled = list(dict.fromkeys(int(p) for p in pooled))
    diag.shortlist_indices = [int(idx[p]) for p in pooled]

    if not pooled:
        best = np.argsort(q)[::-1][:k]
        diag.selected_indices = [int(idx[i]) for i in best]
        return [items[int(idx[i])] for i in best], diag if return_debug else None

    Xp = X[pooled]
    Sp = _cosine_sim(Xp)
    cand = list(range(len(pooled)))
    sel_local = _facility_location(Sp, cand, k)
    final_idx = [int(idx[pooled[i]]) for i in sel_local]
    diag.selected_indices = final_idx
    return [items[j] for j in final_idx], diag if return_debug else None


__all__ = ["SelectionDiagnostics", "select_hybrid"]

