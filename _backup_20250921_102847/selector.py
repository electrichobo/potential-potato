# lines 1–310
from __future__ import annotations
# --- selector: k-medoids (overcluster) -> facility-location on cosine sims ---
from typing import List, Dict, Any, Optional, Callable
import numpy as np

# ----------------- small math utils -----------------
def _l2norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    G = X @ X.T
    nrm = np.sum(X * X, axis=1, keepdims=True)
    D = nrm + nrm.T - 2.0 * G
    np.maximum(D, 0.0, out=D)
    return D

def _init_ff(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    medoids = [int(rng.integers(0, n))]
    D = _pairwise_sq_dists(X)
    min_d = D[medoids[0]].copy()
    for _ in range(1, k):
        j = int(np.argmax(min_d))
        medoids.append(j)
        min_d = np.minimum(min_d, D[j])
    return np.array(medoids, dtype=int)

def _kmedoids_labels(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42) -> np.ndarray:
    n = X.shape[0]; k = max(1, min(k, n))
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
    Xn = _l2norm(X)
    return Xn @ Xn.T

def _facility_location(S: np.ndarray, candidates: List[int], k: int) -> List[int]:
    selected: List[int] = []
    covered = np.zeros(S.shape[0], dtype=np.float32)
    C = candidates.copy()
    for _ in range(min(k, len(C))):
        gains = [np.maximum(covered, S[:, j]).sum() - covered.sum() for j in C]
        j_pick = C[int(np.argmax(gains))]
        selected.append(j_pick)
        covered = np.maximum(covered, S[:, j_pick])
        C.remove(j_pick)
        if not C: 
            break
    return selected

# ----------------- optional CLIP embedding hook -----------------
def _maybe_embed_clip(items: List[Dict[str, Any]], on_progress: Optional[Callable[[float,str],None]] = None) -> None:
    """
    If the deep CLIP module is available/enabled, compute embeddings and
    store them under 'feat_clip' in-place. Otherwise no-op.
    """
    try:
        # Our helper is expected to read config internally and decide enabled/disabled.
        from ..features.deep_clip import embed_frames_if_enabled  # type: ignore
    except Exception:
        return

    if on_progress:
        on_progress(0.01, "CLIP embeddings (fp16, batched)…")
    try:
        # Should fill each item with item['feat_clip'] if enabled, else no-op.
        embed_frames_if_enabled(items)
    finally:
        if on_progress:
            on_progress(0.02, "CLIP embeddings done")

# ----------------- main selection -----------------
def select_hybrid(
    items: List[Dict[str, Any]],
    k: int,
    feat_key: str = "feat",            # cheap features key (backup)
    quality_key: str = "score",
    overcluster_factor: float = 3.0,
    per_cluster_top: int = 3,
    min_quality: Optional[float] = None,
    on_progress: Optional[Callable[[float, str], None]] = None,  # <-- progress hook
) -> List[Dict[str, Any]]:
    """
    1) (optional) embed CLIP -> items[i]['feat_clip'] when available
    2) use 'feat_clip' if present, else fall back to 'feat'
    3) overcluster via k-medoids
    4) facility-location on cosine similarities
    """
    if len(items) <= k:
        return items

    # Try CLIP (no-op if not available/disabled).
    _maybe_embed_clip(items, on_progress=on_progress)
    
    try:
        _maybe_embed_clip(items, on_progress=on_progress)
    except Exception:
        if on_progress:
            on_progress(0.02, "CLIP skipped (error)")
        # continue with cheap features

    # Prefer high-capacity features if present.
    preferred_feat_keys = ("feat_clip", feat_key)

    feats, quals, keep = [], [], []
    for i, it in enumerate(items):
        f = None
        for key in preferred_feat_keys:
            if key in it and it[key] is not None:
                f = it[key]
                break
        if f is None:
            continue
        feats.append(np.asarray(f, dtype=np.float32))
        quals.append(float(it.get(quality_key, 0.0)))
        keep.append(i)

    if not feats:
        # fall back to pure quality sort if no features exist at all
        return sorted(items, key=lambda d: float(d.get(quality_key, 0.0)), reverse=True)[:k]

    X   = np.vstack(feats)
    q   = np.asarray(quals, dtype=np.float32)
    idx = np.asarray(keep, dtype=int)

    # optional quality floor
    if min_quality is not None:
        sel = np.where(q >= float(min_quality))[0]
        if sel.size == 0:
            sel = np.arange(len(idx))
        X, q, idx = X[sel], q[sel], idx[sel]

    # overcluster with k-medoids
    m = int(np.ceil(overcluster_factor * k))
    m = max(1, min(m, len(idx)))
    labels = _kmedoids_labels(X, m, max_iter=50, seed=42)

    # take top-N per cluster by quality
    pooled = []
    for c in range(m):
        loc = np.where(labels == c)[0]
        if loc.size == 0:
            continue
        order = loc[np.argsort(q[loc])[::-1]]  # high-quality first
        pooled.extend(order[:min(per_cluster_top, order.size)].tolist())

    # keep unique while preserving order
    pooled = list(dict.fromkeys(pooled))

    # facility location on cosine sims among pooled candidates
    Xp = X[pooled]
    Sp = _cosine_sim(Xp)
    cand = list(range(len(pooled)))
    sel_local = _facility_location(Sp, cand, k)
    final_idx = [idx[pooled[i]] for i in sel_local]
    return [items[j] for j in final_idx]
# end 1–310
