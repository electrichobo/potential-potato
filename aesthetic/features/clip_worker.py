# lines 1–240
from __future__ import annotations
"""
AESTHETIC — OpenCLIP worker subprocess

CLI:
  python -m aesthetic.features.clip_worker --in <batch.npz> --out <feats.npy>
                                           --model ViT-B-32 --device cuda:0
                                           --fp16 0|1 --batch 16 --log <worker.log>

- Loads RGB frames (N, H, W, 3) uint8 from --in
- Runs OpenCLIP encode_image with fp16/fp32 on CPU/GPU
- Auto batch backoff on CUDA OOM
- Saves L2-normalized float32 features to --out (N, D)
- Prints progress & errors to --log (stdout+stderr)
"""
import argparse, os, sys, time
import numpy as np

def _log(msg: str):
    print(msg, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_npz", required=True)
    ap.add_argument("--out", dest="out_npy", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    # Redirect stdout/err to --log if provided (even inside subprocess this helps)
    if args.log and os.path.abspath(args.log) != os.devnull:
        try:
            os.makedirs(os.path.dirname(args.log), exist_ok=True)
            sys.stdout = open(args.log, "a", encoding="utf-8")
            sys.stderr = sys.stdout
            _log("\n==== worker restart ====")
        except Exception:
            pass

    _log(f"worker args: {args}")

    try:
        data = np.load(args.in_npz)
        rgb = data["rgb"]  # (N, H, W, 3) uint8
    except Exception as e:
        _log(f"[FATAL] could not read input NPZ: {e}")
        return 2

    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        _log(f"[FATAL] bad RGB shape: {rgb.shape}")
        return 2

    try:
        import torch, open_clip
    except Exception as e:
        _log(f"[FATAL] torch/open_clip import failed: {e}")
        return 2

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    use_fp16 = bool(args.fp16) and use_cuda
    bs = max(4, int(args.batch))

    _log(f"torch: {torch.__version__}  cuda? {use_cuda}  device: {device}")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained="openai", device=device
        )
        model.eval()
        if use_fp16:
            model = model.half()
        if use_cuda:
            try:
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception as e:
        _log(f"[FATAL] model init failed: {type(e).__name__}: {e}")
        return 3

    # Run in batches with OOM backoff
    N = rgb.shape[0]
    feats_all = []
    done = 0
    _log(f"encode N={N}, start bs={bs}, fp16={use_fp16}")
    t0 = time.time()

    while done < N:
        this_step = min(bs, N - done)
        chunk = rgb[done:done+this_step]  # (B, H, W, 3)
        try:
            with torch.no_grad():
                imgs = [preprocess(open_clip.to_pil_image(x)) for x in chunk]
                X = torch.stack(imgs, dim=0).to(device)
                if use_fp16:
                    with torch.cuda.amp.autocast():
                        F = model.encode_image(X).float()
                else:
                    F = model.encode_image(X).float()
                F = F / (F.norm(dim=1, keepdim=True) + 1e-8)  # L2 norm
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
                bs = max(4, bs // 2)
                _log(f"[OOM] reducing batch {old} -> {bs}; retrying this window")
                if use_cuda:
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                time.sleep(0.1)
                continue
            _log(f"[FATAL] runtime error: {re}")
            return 4
        except Exception as e:
            _log(f"[FATAL] encode error: {type(e).__name__}: {e}")
            return 4

    feats = np.concatenate(feats_all, axis=0) if feats_all else np.zeros((0, 512), np.float32)
    _log(f"done in {time.time()-t0:.2f}s -> feats {feats.shape}")

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
    sys.exit(main())
# end 1–240
