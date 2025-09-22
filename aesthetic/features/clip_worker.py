# -*- coding: utf-8 -*-
from __future__ import annotations
"""
AESTHETIC — OpenCLIP worker

Python API:
  compute_clip_embeddings(frames_bgr, model="ViT-B-32", fp16=True, device="cuda:0", batch=16) -> np.ndarray [N, D]

CLI (kept for tooling/back-compat):
  python -m aesthetic.features.clip_worker --in <batch.npz> --out <feats.npy>
                                           --model ViT-B-32 --device cuda:0
                                           --fp16 0|1 --batch 16 --log <worker.log>

- Expects RGB frames if called via API: Sequence[np.ndarray(H,W,3), dtype=uint8] (BGR ok; we convert)
- Uses open_clip create_model_and_transforms
- L2-normalizes embeddings; returns float32
- Handles CUDA OOM by reducing batch size
"""
import argparse, os, sys, time
from typing import Sequence, Optional, Tuple
import numpy as np

def _log(msg: str) -> None:
    print(msg, flush=True)

# ---------- Core encode (shared by API & CLI) ----------
def _load_model(model_name: str, device_str: str, use_fp16: bool):
    import torch, open_clip
    use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(device_str if use_cuda else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai", device=device)
    model.eval()
    if use_fp16 and use_cuda:
        model = model.half()
    # Optional perf knobs (guard for PyTorch versions)
    if use_cuda:
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
        except Exception:
            pass
        try:
            # torch >= 2.0
            import torch as _t
            if hasattr(_t, "set_float32_matmul_precision"):
                _t.set_float32_matmul_precision("high")
        except Exception:
            pass
    return model, preprocess, device, use_cuda

def _frames_to_tensor(chunk: np.ndarray, preprocess, device, use_fp16: bool):
    """
    chunk: np.ndarray (B, H, W, 3) uint8 RGB or BGR (we detect/order-correct)
    returns: torch.Tensor (B, C, H, W)
    """
    from PIL import Image
    import torch

    # Heuristic: treat as BGR if mean(B) far from mean(R) after swap—a cheap test is messy.
    # Simpler: assume input is BGR (OpenCV read) and convert to RGB.
    if chunk.ndim != 4 or chunk.shape[-1] != 3:
        raise ValueError(f"frames must be (B,H,W,3), got {chunk.shape}")
    # BGR -> RGB
    rgb = chunk[..., ::-1].copy()

    imgs = [preprocess(Image.fromarray(x)) for x in rgb]
    X = torch.stack(imgs, dim=0).to(device)
    if use_fp16 and device.type == "cuda":
        X = X.half()
    return X

def _encode_batches(
    frames_rgb: np.ndarray,
    model,
    preprocess,
    device,
    use_cuda: bool,
    use_fp16_amp: bool,
    batch_size: int,
) -> np.ndarray:
    import torch

    N = frames_rgb.shape[0]
    feats_all = []
    done = 0
    bs = max(1, int(batch_size))
    t0 = time.time()
    _log(f"encode N={N}, start bs={bs}, fp16_amp={use_fp16_amp}")

    while done < N:
        this_step = min(bs, N - done)
        chunk = frames_rgb[done:done + this_step]  # (B, H, W, 3)
        try:
            with torch.no_grad():
                X = _frames_to_tensor(chunk, preprocess, device, use_fp16_amp)
                if use_fp16_amp and device.type == "cuda":
                    # autocast only helps on CUDA
                    with torch.cuda.amp.autocast():
                        F = model.encode_image(X).float()
                else:
                    F = model.encode_image(X).float()
                # L2 normalize
                F = F / (F.norm(dim=1, keepdim=True) + 1e-8)
                feats_all.append(F.detach().cpu().numpy().astype(np.float32))
            done += this_step
            _log(f"batch ok: {done}/{N} (bs={this_step})")
            del X, F
            if use_cuda:
                torch.cuda.empty_cache()
        except RuntimeError as re:
            msg = str(re).lower()
            if ("out of memory" in msg) or ("cuda" in msg and "oom" in msg):
                old = bs
                new_bs = max(1, bs // 2)
                if new_bs == bs and bs > 1:
                    new_bs = bs - 1
                if new_bs < 1:
                    raise
                _log(f"[OOM] reducing batch {old} -> {new_bs}; retrying this window")
                bs = new_bs
                if use_cuda:
                    try:
                        import torch as _t
                        _t.cuda.empty_cache()
                    except Exception:
                        pass
                time.sleep(0.05)
                continue
            raise
    feats = np.concatenate(feats_all, axis=0) if feats_all else np.zeros((0, 512), np.float32)
    _log(f"done in {time.time()-t0:.2f}s -> feats {feats.shape}")
    return feats

# ---------- Public API ----------
def compute_clip_embeddings(
    frames_bgr: Sequence[np.ndarray],
    model: str = "ViT-B-32",
    fp16: bool = True,
    device: str = "cuda:0",
    batch: int = 16,
) -> np.ndarray:
    """
    Encode a sequence of frames (BGR uint8) to L2-normalized CLIP embeddings (float32).
    Returns: np.ndarray [N, D]
    """
    # Fast path: pack to contiguous array
    if len(frames_bgr) == 0:
        return np.zeros((0, 512), np.float32)
    # Ensure uint8 and contiguous
    frames = [np.ascontiguousarray(x, dtype=np.uint8) for x in frames_bgr]
    arr = np.stack(frames, axis=0)  # (N, H, W, 3)

    try:
        import torch, open_clip  # noqa: F401
    except Exception as e:
        _log(f"[FATAL] torch/open_clip import failed: {e}")
        raise

    model_obj, preprocess, device_obj, use_cuda = _load_model(model, device, fp16)
    use_fp16_amp = bool(fp16 and use_cuda)

    feats = _encode_batches(
        frames_rgb=arr,
        model=model_obj,
        preprocess=preprocess,
        device=device_obj,
        use_cuda=use_cuda,
        use_fp16_amp=use_fp16_amp,
        batch=int(batch),
    )
    return feats

# ---------- CLI ----------
def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_npz", required=True)
    ap.add_argument("--out", dest="out_npy", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)

    # Redirect stdout/err to --log if provided
    if args.log and os.path.abspath(args.log) != os.devnull:
        try:
            os.makedirs(os.path.dirname(args.log), exist_ok=True)
            sys.stdout = open(args.log, "a", encoding="utf-8")
            sys.stderr = sys.stdout
            _log("\n==== worker restart ====")
        except Exception:
            pass

    _log(f"worker args: {args}")

    # Load frames from NPZ as RGB uint8
    try:
        data = np.load(args.in_npz)
        rgb = data["rgb"]  # (N, H, W, 3) uint8 RGB
    except Exception as e:
        _log(f"[FATAL] could not read input NPZ: {e}")
        return 2

    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        _log(f"[FATAL] bad RGB shape/dtype: {rgb.shape} {rgb.dtype}")
        return 2

    # Compute
    try:
        feats = compute_clip_embeddings(
            frames_bgr=[rgb[i, :, :, ::-1] for i in range(rgb.shape[0])],  # convert RGB->BGR for API, which expects BGR
            model=args.model,
            fp16=bool(args.fp16),
            device=args.device,
            batch=int(args.batch),
        )
    except Exception as e:
        _log(f"[FATAL] encode failed: {type(e).__name__}: {e}")
        return 4

    # Save
    try:
        out_dir = os.path.dirname(os.path.abspath(args.out_npy))
        os.makedirs(out_dir, exist_ok=True)
        np.save(args.out_npy, feats)
    except Exception as e:
        _log(f"[FATAL] could not write output: {e}")
        return 5

    _log("success")
    return 0

if __name__ == "__main__":
    sys.exit(_cli())
