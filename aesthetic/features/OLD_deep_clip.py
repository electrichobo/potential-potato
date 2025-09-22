# lines 1–240  (aesthetic/features/deep_clip.py)
from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Tuple
import os, math
import numpy as np
import cv2

ProgressCb = Optional[Callable[[float, str], None]]

_CFG_CACHE: Optional[Dict[str, Any]] = None
_MODEL_CACHE: Optional[Tuple[object, object, object, str, bool, int]] = None
# (model, preprocess, device, model_name, use_fp16, batch_size)

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))

def _load_cfg() -> Dict[str, Any]:
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    import yaml
    cfg_path = os.path.join(_project_root(), "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        _CFG_CACHE = yaml.safe_load(f) or {}
    return _CFG_CACHE

def _log(cb: ProgressCb, pct: float, msg: str):
    if cb:
        cb(max(0.0, min(1.0, pct)), msg)

def _get_model(cb: ProgressCb) -> Optional[Tuple[object, object, object, str, bool, int]]:
    """Create (or return cached) open-clip model/preprocess w/ device + settings."""
    global _MODEL_CACHE
    cfg = _load_cfg()
    heavy = (cfg.get("heavy") or {}).get("clip") or {}
    if not bool(heavy.get("enabled", False)):
        return None

    try:
        import torch, open_clip
    except Exception:
        _log(cb, 0.0, "CLIP disabled (deps missing)")
        return None

    gpu = cfg.get("gpu") or {}
    dev_str = str(gpu.get("device", "cuda:0"))
    
# TEMPORARY SAFE MODE -> force CPU + fp32 to rule out CUDA hard-crash
    use_cuda = False                          # <— force CPU for now
    device   = torch.device("cpu")
    precision = str(heavy.get("precision", "fp16")).lower()
    use_fp16  = False                         # <— fp32
    bs_cfg    = int(gpu.get("batch_size", 32))
    
#    use_cuda = bool(gpu.get("enabled", True)) and torch.cuda.is_available()
#    device = getattr(torch, "device")(dev_str if use_cuda else "cpu")
#    precision = str(heavy.get("precision", "fp16")).lower()
#    use_fp16 = (precision == "fp16") and use_cuda
#    bs_cfg = int(gpu.get("batch_size", 32))

    model_name = str(heavy.get("model", "ViT-B-32"))

    # fast-return cache if settings match
    if _MODEL_CACHE is not None:
        m, pp, dev_cached, name_cached, fp16_cached, bs_cached = _MODEL_CACHE
        if str(dev_cached) == str(device) and name_cached == model_name and fp16_cached == use_fp16:
            return _MODEL_CACHE

    try:
        _log(cb, 0.01, f"CLIP init ({model_name}, fp16={use_fp16})…")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", device=device
        )
        model.eval()
        if use_fp16:
            model = model.half()
        if use_cuda:
            try:
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = True
                import torch as _torch
                _torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        _MODEL_CACHE = (model, preprocess, device, model_name, use_fp16, bs_cfg)
        return _MODEL_CACHE
    except Exception as e:
        _log(cb, 0.0, f"CLIP init failed; skipping ({type(e).__name__})")
        return None

def embed_frames_if_enabled(items: List[Dict[str, Any]], on_progress: ProgressCb = None) -> None:
    """
    Compute CLIP embeddings (if enabled) and attach L2-normalized float32 at 'feat_clip'.
    Never raises: on errors it logs and returns so the pipeline can continue.
    """
    pack = _get_model(on_progress)
    if pack is None:
        return  # disabled or missing deps
    model, preprocess, device, _name, use_fp16, bs_cfg = pack

    try:
        import torch
        from PIL import Image
    except Exception:
        _log(on_progress, 0.0, "CLIP deps missing; skipping")
        return

    # which items still need embeddings?
    todo_idx = []
    for i, it in enumerate(items):
        if it.get("feat_clip") is not None:
            continue
        fr = it.get("frame")
        if fr is None or not hasattr(fr, "shape") or fr.ndim != 3 or fr.shape[2] != 3 or fr.size == 0:
            continue
        todo_idx.append(i)

    if not todo_idx:
        _log(on_progress, 0.02, "CLIP cached (nothing to do)")
        return

    # gentle start; we back off on OOM automatically
    bs = max(4, int(bs_cfg))
    done = 0
    total = len(todo_idx)
    _log(on_progress, 0.02, f"CLIP embeddings (N={total}, bs={bs})…")

    while done < total:
        this_step = min(bs, total - done)
        batch_ids = todo_idx[done: done + this_step]

        # build a safe PIL batch
        pil_list, valid_ids = [], []
        for idx in batch_ids:
            bgr = items[idx]["frame"]
            if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
                continue
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])  # contiguous RGB
            if rgb.shape[1] > 4096 or rgb.shape[0] > 4096:  # pre-shrink pathological frames
                w2, h2 = rgb.shape[1] // 2, rgb.shape[0] // 2
                rgb = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_AREA)
            pil_list.append(Image.fromarray(rgb))
            valid_ids.append(idx)

        if not valid_ids:
            done += this_step
            continue

        try:
            with torch.no_grad():
                tens = [preprocess(p) for p in pil_list]
                X = torch.stack(tens, dim=0).to(device)
                if use_fp16:
                    X = X.half()

                feats = model.encode_image(X).float()
                feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
                feats_np = feats.detach().cpu().numpy().astype(np.float32)

            for j, idx in enumerate(valid_ids):
                items[idx]["feat_clip"] = feats_np[j]

            done += this_step
            _log(on_progress, min(0.2, 0.02 + 0.15 * (done / total)), f"CLIP {done}/{total}")

        except RuntimeError as re:
            msg = str(re).lower()
            if ("out of memory" in msg) or ("cuda" in msg and "oom" in msg):
                bs = max(4, bs // 2)
                _log(on_progress, 0.02, f"CLIP OOM → reducing batch to {bs}")
                if device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue  # retry this segment with smaller batch
            else:
                _log(on_progress, 0.02, "CLIP runtime error; skipping remainder")
                break
        except Exception:
            _log(on_progress, 0.02, "CLIP error; skipping remainder")
            break
        finally:
            # free ASAP
            try:
                del X, feats
            except Exception:
                pass
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    _log(on_progress, 0.2, "CLIP embeddings done")
# end 1–240
