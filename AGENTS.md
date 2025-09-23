# AGENTS.md — AESTHETIC App

> **Purpose**  
> This document describes the *agents* (modular responsibilities) that make up AESTHETIC, how they communicate, what they consume/produce, and how to extend or replace them. It’s a practical map for contributors and for wiring future ML features (CineScale, CameraBench, Emotion AI, Academic Reviews, Narrative Resonance, Simulated Audience Training).

---

## 1) Mission (one-liner)

Surface the most “hero” frames from any video by balancing **technical quality**, **creative language**, and **subjective feel**, with **explainable metrics**, GPU acceleration, and deterministic/reproducible configs.

---

## 2) High-level flow (v0.3)

```
GUI (Tk) ──► Orchestrator (pipeline)
             │
             ├─► Scene Agent (PySceneDetect wrapper) ──► scene spans
             ├─► Micro-Select Agent (cheap per-frame features) ──► per-scene winners
             ├─► Embedding Agent (CLIP, subprocess) [optional] ──► feat_clip
             ├─► Selector Agent (k-medoids overcluster → facility-location)
             ├─► De-dup Agent (cosine + pHash + per-scene quota)
             └─► Output Agent (jpg + .txt sidecar + logs)
```

- **Determinism:** sampling jitter + seed in `config.yaml`.
- **Stability:** CLIP runs in its own process to avoid GPU hard-crash taking down the GUI.
- **Explainability:** per-frame `.txt` sidecars with the metrics that decided the outcome.

---

## 3) Agents (responsibilities & contracts)

### 3.1 GUI Agent (`aesthetic/gui/main_window.py`)
- **Inputs:** user path/URL; output directory; `config.yaml`.
- **Outputs:** progress updates; logs; opens output folder; crash logs written to `outputs/logs`.
- **API:** calls `process_local_video(cfg, src, on_progress)` in a background thread.
- **Notes:** Remembers window geometry; “Copy Error” copies latest traceback/log hint.

---

### 3.2 Orchestrator (Pipeline Agent) (`aesthetic/core/pipeline.py`)
- **Inputs:** `cfg` (YAML), `src_path`, `on_progress`.
- **Outputs:** list of saved frames with metrics and scaled (0–100) score prefixes.
- **Steps:**
  1. **Scene planning** → spans `(start,end)`.
  2. **Per-scene sampling** (jitter, min-gap) → candidate indices.
  3. **Cheap features** per candidate → `feat` (4D) + `score` + `metrics`.
  4. **Per-scene micro-keep** (top m or m%).
  5. **Embedding Agent (optional)** → `feat_clip`.
  6. **Selector Agent** (prefers `feat_clip` if present).
  7. **De-dup Agent** (cosine + pHash + per-scene quota).
  8. **Output Agent** (jpg + `.txt` sidecar).

**Contract (item dict):**
```py
{
  "idx": int,                       # frame index
  "frame": np.ndarray(H, W, 3),     # BGR uint8
  "score": float,                   # cheap score
  "metrics": { "contrast":..., "sharpness":..., "border_clean":..., "center_focus":... },
  "feat": np.ndarray(4, float32),   # cheap 4D (L2-normed)
  "feat_clip": np.ndarray(D)?       # optional CLIP (L2-normed), if enabled
  "scene_id": int
}
```

---

### 3.3 Scene Agent (`aesthetic/core/scene_candidates.py`)
- **Inputs:** `src_path`; method/threshold/downscale from `config.yaml`.
- **Outputs:** `List[Tuple[int,int]]` (inclusive spans per scene).
- **Implementation notes:** wraps PySceneDetect content detector; falls back to full-range span if detection yields none.

---

### 3.4 Micro-Select Agent (inside `pipeline.py`)
- **Inputs:** decoded frames at planned indices; `processing.resize_width`.
- **Outputs:** per-scene buffer of candidates; each with `feat`, `score`, `metrics`.
- **Metrics (current):**
  - `contrast`: (P99–P1) on luma.
  - `sharpness`: Laplacian variance.
  - `border_clean`: low edge density on frame borders.
  - `center_focus`: center sharp vs whole (≤2.0 clamp).
- **Planned Technical Metrics (placeholders):**
  - Dynamic Range, Motion Blur/Directionality, Rolling Shutter, WB/Color Cast, Skin Neutrality.

---

### 3.5 Embedding Agent — CLIP Worker (`aesthetic/features/deep_clip.py` + `clip_worker.py`)
- **Inputs:** batched RGB frames (NPZ), `heavy.clip.*`, `gpu.*`.
- **Outputs:** `feat_clip` (float32, L2-normalized).
- **Execution model:** spawned subprocess via `python -m aesthetic.features.clip_worker` to sandbox CUDA/native faults. Logs go to `outputs/logs/clip_job_*/stderr.log`.
- **Config toggles:**
  - `heavy.clip.enabled`: bool
  - `heavy.clip.model`: e.g., `ViT-B-32`
  - `heavy.clip.precision`: `fp16` or `fp32`
  - `gpu.device`: e.g., `cuda:0`
  - `gpu.batch_size`: int

---

### 3.6 Selector Agent (`aesthetic/core/selector.py`)
- **Goal:** maximize quality **and** diversity.
- **Algorithm:** k-medoids (overcluster by factor `m≈k*overcluster_factor`) → pool top `per_cluster_top` per cluster → **facility-location** in similarity space.
- **Feature space:**
  - If available: **CLIP** cosine similarity (preferred).
  - Else: cheap 4D cosine similarity.
- **Inputs (config):** `select.top_k`, `select.overcluster_factor`, `select.per_cluster_top`, `select.quality_floor`, `selector.use_clip_if_available`, `selector.dedup_clip_cosine_min`.
- **Outputs:** final list of item dicts.

---

### 3.7 De-dup Agent (post-select, inside `pipeline.py`)
- **Inputs:** winners, `dedup.*` thresholds.
- **Outputs:** filtered winners.
- **Rules:**
  - cheap-feature cosine distance floor (`dedup.embed_min_cosine_dist`)
  - pHash Hamming distance (`dedup.dhash_threshold`)
  - per-scene quota (`dedup.per_scene_quota`)

---

### 3.8 Output Agent (inside `pipeline.py`)
- **Outputs:** `outputs/`:
  - `NN_frame_XXXXXXXX.jpg` (prefix `NN` = 0–100 scaled score)
  - `NN_frame_XXXXXXXX.txt` (sidecar metrics)
  - logs in `outputs/logs/` (GUI crash logs, CLIP worker stderr, etc.)
- **Guarantees:** creates `outputs/` and `outputs/logs/` up front.

---

## 4) Configuration (what matters)

**`config.yaml` (v0.3):**
```yaml
extract:
  per_scene_candidates: 9
  per_scene_keep: 3
  per_scene_keep_pct: 0.4
  min_scene_len_frames: 12
  min_candidate_gap_frames: 6

select:
  top_k: 6
  overcluster_factor: 4.0
  per_cluster_top: 2
  quality_floor: null

selector:
  use_clip_if_available: true
  dedup_clip_cosine_min: null   # optional early de-dup in CLIP space

processing:
  resize_width: 720
  max_frames: 800

sampling:
  jitter_frac: 0.12
  random_seed: 42

dedup:
  enable: true
  embed_min_cosine_dist: 0.02
  dhash: true
  dhash_threshold: 6
  per_scene_quota: 2

output:
  write_metrics_txt: true
  score_prefix: true
  folder: "aesthetic/outputs"

gpu:
  enabled: true
  device: "cuda:0"
  batch_size: 16

heavy:
  clip:
    enabled: false            # set true to use CLIP worker
    model: "ViT-B-32"
    precision: "fp16"
```

---

## 5) Training methodology (Phase C/D outline)

- **Data:** curated stills/clips labeled per pillar and overall; reference decks per film/DP/genre; negative examples for failure modes.
- **Targets:** overall keeper (binary/Likert), pillar subscores, pairwise preferences.
- **Optimizers:** pairwise logistic/ranking + Optuna Bayesian search on metric/pillar weights; isotonic/Platt calibration to 0–100.
- **Profiles:** “factory default” + user profiles persisted to `config.yaml` (weights section).
- **Auditability:** write out per-run CSV and aggregate histograms (planned).

---

## 6) Roadmap inspirations → concrete features

- **CineScale:** shot scale classifier (WS/MS/CU/ECU) from face/person scale + CLIP priors.  
- **CameraBench:** camera motion tags (locked/handheld/pan/tilt/dolly/crane) via flow patterns & stabilization tests.  
- **Emotion AI:** valence/arousal/expression using Py-Feat/DeepFace/MorphCast (optional, privacy-conscious).  
- **Academic Reviews:** genre/era/DP reference decks; CLIP proximity as a style head.  
- **Narrative Resonance:** ASR/text sentiment + tempo to estimate arc position (tension/release).  
- **Simulated Audience Training:** VLM-based pairwise A/B judgments filtered by consistency, calibrated with human-in-the-loop.

Each will attach metrics under the correct pillar and expose knobs in `config.yaml`.

---

## 7) Libraries & tooling

- **Core:** OpenCV, NumPy, Pillow, PySceneDetect, FFmpeg/ffprobe (installed at `C:\ffmpeg\bin`).
- **ML:** PyTorch (CUDA 12.1 for RTX 3090), open-clip-torch; (later) scikit-learn, Optuna, FAISS.
- **Text/Affect (later):** spaCy, TextBlob, Py-Feat, DeepFace, MorphCast.
- **Audio (later):** librosa, torchaudio.
- **GUI/Packaging:** Tkinter (now), PyInstaller (packaging).
- **Config/Logging:** PyYAML; structured logs to `outputs/logs/`.

---

## 8) Runbooks

### Local dev
```powershell
# 1) activate venv
E:\AestheticApp\.venv\Scripts\Activate.ps1

# 2) run GUI
python -m aesthetic.app
```

### GPU sanity check
```powershell
python - <<'PY'
import torch; print("CUDA?", torch.cuda.is_available(), "dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

### CLIP toggle
- Enable in `config.yaml` → `heavy.clip.enabled: true`
- Batch size tune: `gpu.batch_size: 8/16/32` (worker auto-logs OOM crashes).

---

## 9) Failure modes & guardrails

- **GPU hard crash (TDR/driver):** CLIP in subprocess; logs in `outputs/logs/clip_job_*`.
- **OOM:** batch auto-backoff inside worker; surface status updates to GUI.
- **Silent stalling:** GUI progress heartbeat; scene count + kept + final reported on completion.
- **Codec pitfalls:** OpenCV may fail on some containers—fallback is FFmpeg pre-probe (future).

---

## 10) Contribution guide (short)

- Keep each “agent” small & testable; prefer pure functions for metrics.
- All new metrics:
  - live under a clear pillar,
  - document inputs/outputs & value ranges,
  - add to `.txt` sidecar output,
  - wire to selection (via quality_floor or weights when we add learned weights).
- For ML additions:
  - make them optional via `config.yaml`,
  - provide CPU fallback or graceful skip with a user-friendly log line.

---

## 11) Open TODOs / Placeholders

- [ ] Technical: Dynamic Range, Motion Blur, Rolling Shutter, WB/Skin Neutrality.  
- [ ] Creative: Color Mood/Palette Cohesion, Lens Vibe proxies, Shot Scale/Grammar.  
- [ ] Subjective: Emotion/Valence, Narrative Resonance, Audience Appeal model.  
- [ ] Reference Deck UI + profile manager.  
- [ ] Batch processing queue + watchdog.  
- [ ] Export pack: PNG/WebP, contact sheet, CSV/JSON bundle.

---

> **Drop-in**: save this as `AGENTS.md` at the repo root.  
> Want me to also add a tiny table-of-metrics matrix and link it from `README.md`?
