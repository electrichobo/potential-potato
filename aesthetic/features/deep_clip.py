# lines 1–240
from __future__ import annotations
"""
AESTHETIC — CLIP embeddings (sandboxed worker)

Why a worker?
- Native/CUDA faults can kill the whole process without a Python traceback.
  We isolate CLIP in a subprocess so the GUI lives on and logs get captured.

Contract:
- embed_frames_if_enabled(items, on_progress) mutates each item by attaching
  item["feat_clip"] = np.ndarray[float32, (D,)] when CLIP is enabled.
- On failure, it logs a short message via on_progress and returns (no raise).

Worker:
- aesthetic/features/clip_worker.py, invoked via `python -m aesthetic.features.clip_worker`
  It writes feats.npy and log files into <output>/logs/clip_job_*.
"""
from typing import List, Dict, Any, Optional, Callable
import os
import sys
import tempfile
import subprocess
import numpy as np

# ---- types ----
ProgressCb = Optional[Callable[[float, str], None]]

# ---- cfg cache ----
_CFG_CACHE: Optional[Dict[str, Any]] = None


# ---- small utils ----
def _log(cb: ProgressCb, pct: float, msg: str) -> None:
    """Bound the progress value and emit a short status line."""
    if cb:
        cb(max(0.0, min(1.0, float(pct))), str(msg))


def _project_root() -> str:
    """Repo root = two levels up from this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def _load_cfg(reload: bool = False) -> Dict[str, Any]:
    """Read config.yaml once (cached)."""
    global _CFG_CACHE
    if _CFG_CACHE is not None and not reload:
        return _CFG_CACHE
    import yaml
    cfg_path = os.path.join(_project_root(), "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        _CFG_CACHE = yaml.safe_load(f) or {}
    return _CFG_CACHE


def _collect_rgb(items: List[Dict[str, Any]]) -> tuple[list[int], np.ndarray]:
    """
    Extract usable RGB frames from items.
    Returns:
        (idxs, rgb_stack) where idxs maps back to items list.
    """
    idxs: list[int] = []
    rgbs: list[np.ndarray] = []
    for i, it in enumerate(items):
        fr = it.get("frame", None)
        if fr is None or not hasattr(fr, "shape") or fr.ndim != 3 or fr.shape[2] != 3:
            continue
        # BGR -> RGB and ensure contiguous memory (PIL safety in worker)
        rgb = np.ascontiguousarray(fr[:, :, ::-1])
        rgbs.append(rgb)
        idxs.append(i)
    if not rgbs:
        return [], np.empty((0, 1, 1, 3), dtype=np.uint8)
    return idxs, np.stack(rgbs, axis=0)


# ---- public API ----
def embed_frames_if_enabled(items: List[Dict[str, Any]], on_progress: ProgressCb = None) -> None:
    """
    Sandboxed CLIP embedding:
      1) Collect RGB frames -> write compressed NPZ.
      2) Spawn worker (`python -m aesthetic.features.clip_worker`) with model/device/batch flags.
      3) Read back N×D float32 features and attach to items[idx]['feat_clip'].

    Never raises (by design). On any error, logs a short message and returns.
    """
    try:
        cfg = _load_cfg()
    except Exception:
        _log(on_progress, 0.0, "config load error; skipping CLIP")
        return

    heavy = (cfg.get("heavy") or {}).get("clip") or {}
    if not bool(heavy.get("enabled", False)):
        _log(on_progress, 0.0, "CLIP disabled")
        return

    gpu      = cfg.get("gpu") or {}
    device   = str(gpu.get("device", "cuda:0"))
    batch_sz = int(gpu.get("batch_size", 16))
    model    = str(heavy.get("model", "ViT-B-32"))
    prec     = str(heavy.get("precision", "fp32")).lower()
    fp16     = 1 if prec == "fp16" else 0

    # Collect frames
    idxs, rgb = _collect_rgb(items)
    if rgb.shape[0] == 0:
        _log(on_progress, 0.02, "No frames for CLIP")
        return

    # Prepare IO & logs
    out_dir_cfg = (cfg.get("output") or {}).get("folder", "aesthetic/outputs")
    out_dir_abs = os.path.abspath(os.path.join(_project_root(), out_dir_cfg))
    logs_dir = os.path.join(out_dir_abs, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        # fallback to project root if output path invalid
        logs_dir = os.path.join(_project_root(), "aesthetic", "outputs", "logs")
        os.makedirs(logs_dir, exist_ok=True)

    tmp_dir  = tempfile.mkdtemp(prefix="clip_job_", dir=logs_dir)
    in_npz   = os.path.join(tmp_dir, "batch.npz")
    out_npy  = os.path.join(tmp_dir, "feats.npy")
    log_txt  = os.path.join(tmp_dir, "worker.log")

    # Save batch
    try:
        np.savez_compressed(in_npz, rgb=rgb)
    except Exception:
        _log(on_progress, 0.02, "CLIP: failed to write batch")
        return

    # Build worker command
    exe = sys.executable  # same interpreter as the GUI/parent process
    cmd = [
        exe, "-m", "aesthetic.features.clip_worker",
        "--in", in_npz,
        "--out", out_npy,
        "--model", model,
        "--device", device,
        "--fp16", str(fp16),
        "--batch", str(batch_sz),
        "--log", log_txt,
    ]

    _log(on_progress, 0.02, f"CLIP worker start (N={rgb.shape[0]}, bs={batch_sz}, {model}, {device}, fp16={bool(fp16)})")

    # Run worker; capture ALL output to a file so native faults are preserved
    try:
        with open(log_txt, "w", encoding="utf-8") as f:
            f.write("AESTHETIC CLIP worker start\n")
            f.write(f"cmd: {' '.join(cmd)}\n\n")
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=f, cwd=_project_root())
        rc = int(proc.returncode)
    except Exception:
        _log(on_progress, 0.02, f"CLIP worker launch failed; see {log_txt}")
        return

    if rc != 0 or not os.path.exists(out_npy):
        _log(on_progress, 0.02, f"CLIP worker failed (rc={rc}); see {log_txt}")
        return

    # Map features back to items
    try:
        feats = np.load(out_npy)  # (N, D) float32
        if feats.shape[0] != len(idxs):
            _log(on_progress, 0.02, f"CLIP mismatch N={feats.shape[0]}!={len(idxs)}; see {log_txt}")
            return
        for j, i in enumerate(idxs):
            items[i]["feat_clip"] = feats[j]
        _log(on_progress, 0.2, f"CLIP worker done → feats={feats.shape}")
    except Exception:
        _log(on_progress, 0.02, f"CLIP readback error; see {log_txt}")
        return
# end 1–240
