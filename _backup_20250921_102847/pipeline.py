# lines 1–470
from __future__ import annotations
# --- pipeline_v0.3: per-scene micro-selection, then hybrid global selection ---
from typing import Dict, Any, List, Callable, Optional, Tuple
import os
import cv2
import numpy as np

ProgressCb = Optional[Callable[[float, str], None]]
CancelFn   = Optional[Callable[[], bool]]

# ---------- helpers (stateless) ----------
def _scale_scores_0_100(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    smin, smax = float(scores.min()), float(scores.max())
    if smax <= smin:
        return np.full_like(scores, 50.0)
    return (scores - smin) / (smax - smin) * 100.0

def _downscale(frame_bgr: np.ndarray, target_w: int) -> np.ndarray:
    if target_w is None or target_w <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= target_w:
        return frame_bgr
    scale = target_w / float(w)
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def _luma_bgr(img: np.ndarray) -> np.ndarray:
    return 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]

def _contrast(y: np.ndarray) -> float:
    y = np.clip(y, 0, 255)
    p1, p99 = np.percentile(y, [1, 99])
    return float(p99 - p1)

def _sharp(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _border_clean(edges: np.ndarray) -> float:
    h, w = edges.shape
    b = max(4, int(0.05 * min(h, w)))
    m = np.zeros_like(edges, dtype=bool)
    m[:b, :] = True; m[-b:, :] = True; m[:, :b] = True; m[:, -b:] = True
    den = float(edges[m].mean() / 255.0) if edges[m].size else 0.0
    return float(1.0 - den)

def _center_focus(gray: np.ndarray) -> float:
    h, w = gray.shape
    cy0, cy1 = int(h * 0.33), int(h * 0.66)
    cx0, cx1 = int(w * 0.33), int(w * 0.66)
    roi = gray[cy0:cy1, cx0:cx1]
    full_var = float(cv2.Laplacian(gray, cv2.CV_64F).var() + 1e-6)
    roi_var  = float(cv2.Laplacian(roi,  cv2.CV_64F).var() + 1e-6)
    return float(min(2.0, roi_var / full_var))

def _feat_and_score(small_bgr: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Return (feat[4], score, metrics_dict) computed on a downscaled frame."""
    gray  = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    y     = _luma_bgr(small_bgr)
    edges = cv2.Canny(small_bgr, 80, 160)
    c  = _contrast(y)
    s  = _sharp(gray)
    b  = _border_clean(edges)
    cf = _center_focus(gray)
    feat  = np.array([c/255.0, s/(s+1000.0), b, cf], dtype=np.float32)
    score = 0.45*(c/255.0) + 0.4*(s/(s+1000.0)) + 0.1*b + 0.05*cf
    return feat, float(score), {"contrast": c, "sharpness": s, "border_clean": b, "center_focus": cf}

# ---------- main pipeline ----------
def process_local_video(
    cfg: Dict[str, Any],
    src_path: str,
    on_progress: ProgressCb = None,
    is_cancelled: CancelFn = None,
) -> List[Dict[str, Any]]:
    """
    Phase B micro-selection:
      - detect scenes (scenedetect)
      - for each scene: sample N candidates; compute features; keep top m or top %
      - merge kept frames and run hybrid selector (k-medoids -> facility-location)
      - save JPGs with score prefix + .txt sidecar (toggle: output.write_metrics_txt)
    """
    from .scene_candidates import get_scene_spans
    from .selector import select_hybrid

    # ------- config knobs -------
    out_dir      = cfg.get("output", {}).get("folder", "aesthetic/outputs")
    score_prefix = bool(cfg.get("output", {}).get("score_prefix", True))
    write_txt    = bool(cfg.get("output", {}).get("write_metrics_txt", True))

    target_w          = int(cfg.get("processing", {}).get("resize_width", 720))
    max_frames_global = int(cfg.get("processing", {}).get("max_frames", 800))

    per_scene_candidates = int(cfg.get("extract", {}).get("per_scene_candidates", 7))
    per_scene_keep       = cfg.get("extract", {}).get("per_scene_keep", 3)
    per_scene_keep_pct   = cfg.get("extract", {}).get("per_scene_keep_pct", None)
    min_scene_len        = int(cfg.get("extract", {}).get("min_scene_len_frames", 12))
    min_gap              = int(cfg.get("extract", {}).get("min_candidate_gap_frames", 4))

    # sampling jitter
    jitter_frac = float(cfg.get("sampling", {}).get("jitter_frac", 0.0))  # 0..0.5
    rng_seed    = cfg.get("sampling", {}).get("random_seed", None)
    rng         = np.random.default_rng(int(rng_seed)) if rng_seed is not None else np.random.default_rng()

    # dedup + per-scene quota
    dedup_cfg   = cfg.get("dedup", {})
    dd_enable   = bool(dedup_cfg.get("enable", True))
    dd_cos_min  = float(dedup_cfg.get("embed_min_cosine_dist", 0.0))
    dd_phash    = bool(dedup_cfg.get("dhash", True))
    dd_ph_thr   = int(dedup_cfg.get("dhash_threshold", 6))
    scene_quota = int(dedup_cfg.get("per_scene_quota", 0))  # 0 = no cap

    # global selection
    top_k          = int(cfg.get("select", {}).get("top_k", 5))
    over           = float(cfg.get("select", {}).get("overcluster_factor", 3.0))
    per_cluster_top= int(cfg.get("select", {}).get("per_cluster_top", 3))
    min_quality    = cfg.get("select", {}).get("quality_floor", None)

    os.makedirs(out_dir, exist_ok=True)

    # ------- scenes -------
    spans: List[Tuple[int, int]] = get_scene_spans(src_path)
    if not spans:
        cap_tmp = cv2.VideoCapture(src_path)
        total = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap_tmp.release()
        spans = [(0, max(0, total - 1))]

    # helper must be defined BEFORE we use it (and can see jitter/rng via closure)
    def _even_positions(start: int, end: int, count: int, min_gap_val: int = 1) -> list[int]:
        """~Even indices in [start,end], with optional jitter and a minimum spacing."""
        if end < start:
            return []
        length = end - start + 1
        k = max(1, min(count, length))
        base = np.linspace(start, end, num=k, dtype=float)
        if jitter_frac > 0.0:
            jitter = (rng.random(k) * 2.0 - 1.0) * (length * float(jitter_frac))
            base = np.clip(base + jitter, start, end)
        idxs = np.unique(base.astype(int)).tolist()
        out, last = [], -10**9
        for i in idxs:
            if i - last >= max(1, min_gap_val):
                out.append(i); last = i
        return out

    # target indices per scene
    per_scene_targets: List[List[int]] = []
    for (start, end) in spans:
        if end - start + 1 < max(1, min_scene_len):
            continue
        idxs = _even_positions(start, end, per_scene_candidates, min_gap)
        per_scene_targets.append(idxs)

    # ------- single decode pass -------
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video. Try a different file/container.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    scene_i = 0
    items_global: List[Dict[str, Any]] = []
    scene_buffer: List[Dict[str, Any]] = []

    def current_targets_set() -> set[int]:
        return set(per_scene_targets[scene_i]) if 0 <= scene_i < len(per_scene_targets) else set()

    targets_set = current_targets_set()
    cur, reported = 0, -1

    def bump(pct: float, msg: str):
        if on_progress:
            on_progress(max(0.0, min(1.0, pct)), msg)

    bump(0.0, "Decoding (per-scene)…")

    while True:
        if is_cancelled and is_cancelled():
            break
        ok, frame = cap.read()
        if not ok:
            break

        # flush scenes as we pass their end
        while scene_i < len(spans) and cur > spans[scene_i][1]:
            if scene_buffer:
                scene_buffer.sort(key=lambda d: d["score"], reverse=True)
                keep_n = (max(1, int(round(len(scene_buffer) * float(per_scene_keep_pct))))
                          if per_scene_keep_pct is not None else int(per_scene_keep))
                items_global.extend(scene_buffer[:max(1, keep_n)])
                scene_buffer.clear()
            scene_i += 1
            targets_set = current_targets_set()

        # if this frame is planned for the active scene, compute features/score
        if scene_i < len(spans) and cur in targets_set:
            small = _downscale(frame, target_w)
            feat, sc, met = _feat_and_score(small)
            scene_buffer.append({
                "idx": cur,
                "frame": frame,
                "score": sc,      # FIX: use computed score
                "metrics": met,
                "feat": feat,     # FIX: store feature for selector/dedup
                "scene_id": scene_i
            })

        cur += 1

        if total_frames > 0:
            mark = int((cur / float(total_frames)) * 100)
            if mark != reported:
                bump(cur / float(total_frames), f"Decoding {mark}%")
                reported = mark

        if max_frames_global and len(items_global) > max_frames_global:
            break

    # flush last scene
    if scene_buffer:
        scene_buffer.sort(key=lambda d: d["score"], reverse=True)
        keep_n = (max(1, int(round(len(scene_buffer) * float(per_scene_keep_pct))))
                  if per_scene_keep_pct is not None else int(per_scene_keep))
        items_global.extend(scene_buffer[:max(1, keep_n)])
        scene_buffer.clear()

    cap.release()

    if not items_global:
        raise RuntimeError("No candidate frames found. Try different extract settings.")

    # ------- hybrid selection -------
    winners = select_hybrid(
        items_global, top_k,
        overcluster_factor=over,
        per_cluster_top=per_cluster_top,
        min_quality=min_quality,
        on_progress=bump            # <-- wire GUI progress into selector/deep CLIP
    )

    # ------- dedup + per-scene quota -------
    def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - float(np.dot(a, b)))  # a,b assumed L2-normalized

    def _dhash8(img_bgr: np.ndarray) -> int:
        g = cv2.cvtColor(cv2.resize(img_bgr, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        dif = g[:, 1:] > g[:, :-1]
        h = 0
        for bit in dif.flatten():
            h = (h << 1) | int(bit)
        return h

    def _ham(a: int, b: int) -> int:
        return int((a ^ b).bit_count())

    # precompute any missing features and phash
    for it in winners:
        if "feat" not in it:
            f = np.asarray([
                it["metrics"]["contrast"] / 255.0,
                it["metrics"]["sharpness"] / (it["metrics"]["sharpness"] + 1000.0),
                it["metrics"]["border_clean"],
                it["metrics"]["center_focus"]
            ], dtype=np.float32)
            n = float(np.linalg.norm(f)) or 1.0
            it["feat"] = (f / n).astype(np.float32)
        if dd_phash:
            it["_ph"] = _dhash8(it["frame"])

    kept: List[Dict[str, Any]] = []
    scene_counts: Dict[int, int] = {}
    for cand in sorted(winners, key=lambda d: d["score"], reverse=True):
        sid = cand.get("scene_id", -1)

        # per-scene soft cap
        if scene_quota > 0 and scene_counts.get(sid, 0) >= scene_quota:
            continue

        too_close = False
        if dd_enable and dd_cos_min > 0.0 and kept:
            for k in kept:
                if _cos_dist(cand["feat"], k["feat"]) < dd_cos_min:
                    too_close = True
                    break
        if not too_close and dd_enable and dd_phash and kept:
            for k in kept:
                if _ham(cand["_ph"], k["_ph"]) <= dd_ph_thr:
                    too_close = True
                    break

        if not too_close:
            kept.append(cand)
            scene_counts[sid] = scene_counts.get(sid, 0) + 1

    winners = kept[:top_k] if len(kept) >= top_k else kept

    # ------- save results -------
    scores = np.array([w["score"] for w in winners], dtype=np.float32)
    scaled = _scale_scores_0_100(scores)

    saved: List[Dict[str, Any]] = []
    for w, s in zip(winners, scaled):
        prefix = f"{int(round(s)):02d}_" if score_prefix else ""
        out_path = os.path.join(out_dir, f"{prefix}frame_{w['idx']:08d}.jpg")
        cv2.imwrite(out_path, w["frame"])
        saved.append({"path": out_path, "idx": w["idx"], "score": float(w["score"]), "metrics": w["metrics"]})
        if write_txt:
            base, _ = os.path.splitext(out_path)
            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(f"frame_index: {w['idx']}\n")
                f.write(f"score_raw: {w['score']:.6f}\n")
                f.write(f"score_scaled_0_100: {int(round(s))}\n")
                for k, v in w["metrics"].items():
                    f.write(f"{k}: {float(v):.6f}\n")

    if on_progress:
        on_progress(1.0, f"Done. Scenes≈{len(per_scene_targets)}  Kept={len(items_global)}  Final={len(saved)}")

    return saved
# end 1–470
