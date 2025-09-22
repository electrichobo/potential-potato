# lines 1–160
from __future__ import annotations
# --- model registry: optional/late-bound ML loaders ---
from typing import Optional, Any, Dict

_REG: Dict[str, Any] = {}

def register(name: str, obj: Any) -> None:
    if name in _REG: raise ValueError(f"Model already registered: {name}")
    _REG[name] = obj

def get(name: str) -> Any:
    if name not in _REG: raise KeyError(f"Model not loaded: {name}")
    return _REG[name]

def try_load_clip(device: str = "cpu") -> Optional[Any]:
    """
    Optional CLIP loader (Transformers). Import only if installed.
    Returns a tiny wrapper (encode_image -> np.ndarray) or None on failure.
    """
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        import numpy as np
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        def encode_image_pil(pil_image) -> np.ndarray:
            inputs = proc(images=pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)  # (1, D)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().reshape(-1)
        wrapper = {"encode_image": encode_image_pil, "dim": int(model.visual_projection.out_features)}
        register("clip", wrapper)
        return wrapper
    except Exception:
        return None
# end 1–160
