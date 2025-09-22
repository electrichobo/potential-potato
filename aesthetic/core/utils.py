# lines 1–60
from __future__ import annotations
# --- utility math helpers ---
import numpy as np
from typing import List, Dict, Any

def l2norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    Xn = l2norm(X)
    return Xn @ Xn.T

def top_k_by_key(items: List[Dict[str, Any]], k: int, key: str) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda it: float(it.get(key, 0.0)), reverse=True)[:k]
# end 1–60
