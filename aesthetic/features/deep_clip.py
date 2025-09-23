"""CLIP embedding orchestration for the AESTHETIC pipeline.

The GUI drives this module to obtain optional CLIP embeddings.  We keep the
logic here lightweight, deterministic, and well-guarded so a failed GPU call
never takes down the main process.  The worker can run either as a separate
subprocess (default) or in-process for debugging.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


LOG = logging.getLogger("aesthetic.features.deep_clip")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _as_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    """Best-effort bool parsing with a safe fallback."""

    try:
        return bool(cfg.get(key, default))
    except Exception:
        return default


def _precision_to_fp16(precision: Optional[str]) -> bool:
    """Return True when the requested precision implies fp16 usage."""

    if precision is None:
        return True
    value = str(precision).strip().lower()
    if value in {"fp16", "half", "float16", "16", "mixed"}:
        return True
    if value in {"fp32", "float32", "32", "full"}:
        return False
    return True


def _resolve_clip_cfg(global_cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalise configuration for the CLIP worker."""

    cfg = dict(global_cfg or {})
    clip_cfg = ((cfg.get("heavy") or {}).get("clip") or {})
    gpu_cfg = cfg.get("gpu") or {}
    output_cfg = cfg.get("output") or {}

    output_dir = Path(output_cfg.get("folder") or "aesthetic/outputs").expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "enabled": _as_bool(clip_cfg, "enabled", False),
        "model": clip_cfg.get("model", "ViT-B-32"),
        "fp16": _precision_to_fp16(clip_cfg.get("precision")),
        "device": gpu_cfg.get("device", "cuda:0"),
        "batch": int(gpu_cfg.get("batch_size", 16)),
        "run_mode": str(clip_cfg.get("run_mode") or "subprocess").strip().lower(),
        "timeout_s": int(clip_cfg.get("timeout_sec", 180)),
        "log_dir": str(log_dir),
    }
    return resolved


# ---------------------------------------------------------------------------
# Worker invocation helpers
# ---------------------------------------------------------------------------


def _write_npz_rgb(frames_bgr: Sequence[np.ndarray], path: Path) -> None:
    """Persist BGR frames as an RGB NPZ file for the subprocess worker."""

    frames_rgb = [np.ascontiguousarray(frame[..., ::-1], dtype=np.uint8) for frame in frames_bgr]
    array = np.stack(frames_rgb, axis=0)
    np.savez_compressed(str(path), rgb=array)


def _call_worker_subprocess(args: Mapping[str, Any], frames_bgr: Sequence[np.ndarray]) -> Tuple[Optional[np.ndarray], bool]:
    """Invoke the CLI worker with a timeout and return (features, timed_out)."""

    timeout = max(10, int(args.get("timeout_s", 180)))
    log_dir = Path(args["log_dir"])
    log_path = log_dir / "clip_worker.log"

    with tempfile.TemporaryDirectory(prefix="aesthetic_clip_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        in_npz = tmp_root / "batch.npz"
        out_npy = tmp_root / "feats.npy"
        _write_npz_rgb(frames_bgr, in_npz)

        cmd = [
            sys.executable,
            "-m",
            "aesthetic.features.clip_worker",
            "--in",
            str(in_npz),
            "--out",
            str(out_npy),
            "--model",
            str(args["model"]),
            "--device",
            str(args["device"]),
            "--fp16",
            "1" if args["fp16"] else "0",
            "--batch",
            str(int(args["batch"])),
            "--log",
            str(log_path),
        ]

        LOG.info("CLIP subprocess launch: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                check=False,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            LOG.warning("CLIP subprocess exceeded %ds timeout; skipping CLIP run.", timeout)
            return None, True

        if result.stdout:
            LOG.debug("clip_worker stdout:%s%s", os.linesep, result.stdout)
        if result.stderr:
            LOG.debug("clip_worker stderr:%s%s", os.linesep, result.stderr)

        if result.returncode != 0:
            LOG.warning("CLIP subprocess failed with code %s; consult %s", result.returncode, log_path)
            return None, False

        if not out_npy.exists():
            LOG.warning("CLIP subprocess produced no output file; consult %s", log_path)
            return None, False

        try:
            feats = np.load(out_npy)
        except Exception as exc:
            LOG.warning("CLIP subprocess output unreadable: %s", exc)
            return None, False

        if not isinstance(feats, np.ndarray) or feats.ndim != 2:
            LOG.warning("CLIP subprocess emitted unexpected array shape=%s", getattr(feats, "shape", None))
            return None, False

        return feats.astype(np.float32, copy=False), False


def _call_worker_inprocess(args: Mapping[str, Any], frames_bgr: Sequence[np.ndarray]) -> Tuple[Optional[np.ndarray], bool]:
    """Use the Python API directly.  Always returns timed_out=False."""

    try:
        from aesthetic.features.clip_worker import compute_clip_embeddings
    except Exception as exc:  # pragma: no cover - import failure path
        LOG.warning("CLIP in-process path unavailable: %s", exc)
        return None, False

    requested_device = str(args["device"])
    device_order = [requested_device]
    if not requested_device.lower().startswith("cpu"):
        device_order.append("cpu")

    for device in device_order:
        start = time.time()
        try:
            LOG.info(
                "CLIP in-process: model=%s device=%s fp16=%s batch=%d",
                args["model"],
                device,
                args["fp16"],
                int(args["batch"]),
            )
            feats = compute_clip_embeddings(
                frames_bgr=list(frames_bgr),
                model=str(args["model"]),
                fp16=bool(args["fp16"]),
                device=device,
                batch=int(args["batch"]),
            )
            if isinstance(feats, np.ndarray) and feats.ndim == 2:
                LOG.info("CLIP in-process succeeded in %.2fs (shape=%s)", time.time() - start, feats.shape)
                return feats.astype(np.float32, copy=False), False
            LOG.warning("CLIP in-process returned invalid output on device %s", device)
        except Exception as exc:  # pragma: no cover - GPU specific failure paths
            LOG.warning("CLIP in-process failed on device %s: %s", device, exc)
            continue

    return None, False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def embed_frames_if_enabled(
    frames_bgr: Sequence[np.ndarray],
    global_config: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compute CLIP embeddings when enabled in the config."""

    args = _resolve_clip_cfg(global_config)
    if not args["enabled"]:
        LOG.info("CLIP embeddings disabled in configuration.")
        return None

    if not frames_bgr:
        LOG.info("CLIP embeddings skipped: no frames provided.")
        return None

    if args["run_mode"] == "subprocess":
        feats, timed_out = _call_worker_subprocess(args, frames_bgr)
    else:
        feats, timed_out = _call_worker_inprocess(args, frames_bgr)

    if feats is None:
        LOG.warning("CLIP embeddings unavailable; proceeding without them.")
        return {"feat_clip": None, "timed_out": timed_out}

    return {"feat_clip": feats, "timed_out": timed_out}


__all__ = [
    "embed_frames_if_enabled",
]

