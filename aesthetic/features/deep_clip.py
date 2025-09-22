# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Sequence, Optional, Dict, Any
import numpy as np
import logging, os, sys, time, tempfile, json, subprocess, shutil
from pathlib import Path

LOG = logging.getLogger("aesthetic.deep_clip")

def _as_bool(d: dict, key: str, default: bool) -> bool:
    try:
        return bool(d.get(key, default))
    except Exception:
        return default

def _precision_to_fp16(precision: str | None) -> bool:
    if not precision:
        return True
    p = str(precision).strip().lower()
    if p in ("fp16", "half", "float16", "16", "mixed"):
        return True
    if p in ("fp32", "float32", "32", "full"):
        return False
    return True

def _resolve_clip_cfg(global_cfg: Optional[dict]) -> dict:
    cfg = global_cfg or {}
    heavy = cfg.get("heavy") or {}
    clip  = heavy.get("clip") or {}
    gpu   = cfg.get("gpu") or {}
    out   = cfg.get("output") or {}

    enabled    = _as_bool(clip, "enabled", False)
    model      = clip.get("model", "ViT-B-32")
    fp16       = _precision_to_fp16(clip.get("precision"))
    device     = gpu.get("device", "cuda:0")
    batch      = int(gpu.get("batch_size", 16))
    run_mode   = (clip.get("run_mode") or "subprocess").lower()  # default to subprocess (safe)
    timeout_s  = int(clip.get("timeout_sec", 120))
    out_dir    = Path((out.get("folder") or "aesthetic/outputs")).resolve()
    log_dir    = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    return dict(
        enabled=enabled, model=model, fp16=fp16, device=device, batch=batch,
        run_mode=run_mode, timeout_s=timeout_s, log_dir=str(log_dir)
    )

def _write_npz_rgb(frames_bgr: Sequence[np.ndarray], path: str) -> None:
    # pack BGR frames as RGB uint8 for the worker CLI
    frames = [np.ascontiguousarray(f[..., ::-1], dtype=np.uint8) for f in frames_bgr]
    arr = np.stack(frames, axis=0)  # (N,H,W,3) RGB
    np.savez_compressed(path, rgb=arr)

def _call_worker_subprocess(args: dict, frames_bgr: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """
    Use CLI worker in a separate process with a timeout.
    Returns feats array or None on failure.
    """
    tmpdir = tempfile.mkdtemp(prefix="aesthetic_clip_")
    try:
        in_npz  = os.path.join(tmpdir, "batch.npz")
        out_npy = os.path.join(tmpdir, "feats.npy")
        log_path = os.path.join(args["log_dir"], "clip_worker.log")

        _write_npz_rgb(frames_bgr, in_npz)

        cmd = [
            sys.executable, "-m", "aesthetic.features.clip_worker",
            "--in", in_npz,
            "--out", out_npy,
            "--model", args["model"],
            "--device", args["device"],
            "--fp16", "1" if args["fp16"] else "0",
            "--batch", str(int(args["batch"])),
            "--log", log_path,
        ]
        LOG.info("CLIP(subprocess): %s", " ".join(cmd))

        try:
            res = subprocess.run(
                cmd,
                timeout=max(10, int(args["timeout_s"])),
                check=False,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            LOG.warning("CLIP(subprocess): timed out after %ds; skipping CLIP.", int(args["timeout_s"]))
            return None

        if res.stdout:
            LOG.debug("clip_worker stdout:\n%s", res.stdout)
        if res.stderr:
            LOG.debug("clip_worker stderr:\n%s", res.stderr)

        if res.returncode != 0:
            LOG.warning("CLIP(subprocess): worker exit code=%s; see %s", res.returncode, log_path)
            return None

        if not os.path.exists(out_npy):
            LOG.warning("CLIP(subprocess): feats file missing; see %s", log_path)
            return None

        feats = np.load(out_npy)
        if isinstance(feats, np.lib.npyio.NpzFile):
            # shouldn't happen; worker writes .npy
            LOG.warning("CLIP(subprocess): unexpected NPZ output; skipping.")
            return None
        if not isinstance(feats, np.ndarray) or feats.ndim != 2:
            LOG.warning("CLIP(subprocess): bad feature array shape=%s", getattr(feats, "shape", None))
            return None
        return feats.astype(np.float32, copy=False)
    except Exception as e:
        LOG.exception("CLIP(subprocess) failed: %s", e)
        return None
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

def _call_worker_inprocess(args: dict, frames_bgr: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """
    In-process path (no timeout). Used only if run_mode='inprocess'.
    """
    try:
        from aesthetic.features.clip_worker import compute_clip_embeddings
    except Exception as e:
        LOG.warning("CLIP(inprocess): worker API not available: %s", e)
        return None

    # Try requested device first, then CPU if CUDA fails.
    order = [args["device"]]
    if not str(args["device"]).startswith("cpu"):
        order.append("cpu")

    for dev in order:
        t0 = time.time()
        try:
            LOG.info("CLIP(inprocess): model=%s device=%s fp16=%s batch=%d",
                     args["model"], dev, args["fp16"], args["batch"])
            feats = compute_clip_embeddings(
                frames_bgr=list(frames_bgr),
                model=args["model"], fp16=args["fp16"], device=dev, batch=int(args["batch"])
            )
            if not isinstance(feats, np.ndarray) or feats.ndim != 2:
                LOG.warning("CLIP(inprocess): bad features on device %s", dev); continue
            LOG.info("CLIP: embeddings shape=%s in %.2fs", feats.shape, time.time()-t0)
            return feats.astype(np.float32, copy=False)
        except Exception as e:
            LOG.warning("CLIP(inprocess): failed on device %s: %s", dev, e)
            continue
    return None

def embed_frames_if_enabled(frames_bgr: Sequence[np.ndarray], global_config: Optional[dict]) -> Optional[Dict[str, Any]]:
    """
    Compute CLIP embeddings if enabled. Returns {"feat_clip": np.ndarray [N,D]} or None.
    NEVER raises to caller; logs and returns None on any failure or timeout.
    """
    args = _resolve_clip_cfg(global_config)
    if not args["enabled"]:
        LOG.info("CLIP: disabled")
        return None
    if len(frames_bgr) == 0:
        LOG.info("CLIP: no frames given")
        return None

    # Prefer subprocess mode (timeout-safe)
    if args["run_mode"] == "subprocess":
        feats = _call_worker_subprocess(args, frames_bgr)
    else:
        feats = _call_worker_inprocess(args, frames_bgr)

    if feats is None:
        LOG.warning("CLIP: proceeding without embeddings.")
        return None
    return {"feat_clip": feats}
