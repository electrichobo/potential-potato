# AESTHETIC

**AESTHETIC** is a Windows-first Python application that automatically extracts **hero frames** and **hero scenes** from video.  
It scores candidates on three judging pillars:

1. **Technical** – measurable, objective metrics (exposure, composition, motion, color, etc.)  
2. **Creative** – stylistic adherence and artistic intent (placeholders for future)  
3. **Subjective** – audience-based impressions (placeholders for future)

The app is modular, deterministic, and user-friendly, with a verbose GUI for transparency. GPU acceleration is optional.

---

## Features (current + planned)

✅ Scene detection and frame sampling  
✅ Cheap technical metrics (exposure, sharpness, saturation)  
✅ k-medoids overcluster + facility-location selection  
✅ Deduplication (cosine + dHash)  
✅ GUI with logging + progress bar  
✅ CLIP embeddings (in-process, CUDA optional)  

🔜 Always-write invariant (guarantee at least `top_k` frames per run)  
🔜 CLIP subprocess + timeout handling  
🔜 Verbose GUI status window with per-stage breakdown  
🔜 Hero-scenes video export (`hero_scenes.mp4`) with stitched clips  
🔜 Feature caching for faster re-runs  

◻️ Advanced technical metrics (SSIM, VMAF, depth, grading uniformity)  
◻️ Creative pillar metrics (lighting style, composition creativity, palette mood, motif tracking)  
◻️ Subjective pillar metrics (clarity, emotional response, memorability)  
◻️ Contact sheets + run manifests  
◻️ Profile weighting by DP/film reference sets

---

## Directory Structure

```
aesthetic/
  app.py                # Entry point (GUI launcher)
  main_window.py        # GUI with verbose status pane
  config.yaml           # Single config file (root-level)

  core/
    pipeline.py         # End-to-end orchestrator
    ingest.py           # Resolve local/URL sources
    extractor.py        # Frame indexing
    scene_candidates.py # Scene detection
    selector.py         # k-medoids, facility-location, dedup
    features_basic.py   # Fast OpenCV-based technical metrics
    features_advanced.py# Advanced metrics (SSIM, VMAF, etc., opt-in)
    video_export.py     # Hero-scenes video export
    utils.py            # Hashing, caching, logging

  features/
    deep_clip.py        # CLIP controller (in-process/subprocess)
    clip_worker.py      # Subprocess worker for CLIP

  data/
    references/
      by_dp/            # Placeholder: DP reference profiles
      by_film/          # Placeholder: film reference profiles

  outputs/
    logs/               # Logs + crash reports
```

---

## Config (example)

```yaml
extract:
  per_scene_candidates: 9
  per_scene_keep_pct: 0.4
  min_scene_len_frames: 12
select:
  top_k: 6
  overcluster_factor: 4.0
  per_cluster_top: 2
dedup:
  enable: true
  embed_min_cosine_dist: 0.02
  dhash: true
output:
  folder: "aesthetic/outputs"
  score_prefix: true
  hero_video:
    enabled: true
    seconds_before: 0.8
    seconds_after: 1.2
    max_total_duration: 45.0
gpu:
  enabled: true
  device: "cuda:0"
  batch_size: 8
pillars:
  technical_weight: 0.6
  creative_weight: 0.25
  subjective_weight: 0.15
```

---

## Technical Metrics (Phase 1 — implemented / in progress)

**Exposure**
- Average luminance  
- Clipped shadows/highlights %  
- Histogram spread (std, skewness)  
- Temporal exposure uniformity (scene-level)  

**Lighting**
- Dynamic range (brightest/darkest)  
- Key/fill ratio proxy  
- Shadow detail detection  
- Spill/softness proxy  

**Composition**
- Rule-of-thirds adherence (saliency center)  
- Occupancy center of mass  
- Symmetry/asymmetry score  
- Negative space ratio  

**Camera Movement**
- Optical flow smoothness  
- Jerkiness/acceleration spikes  
- Stabilization/jitter proxy  
- Motion blur estimate  

**Color**
- White balance deviation (Kelvin proxy)  
- Saturation/variance  
- Palette bins & entropy  
- Artifact detection (banding proxy)  

**Phase 2/3 (planned opt-in)**
- SSIM, PSNR, VMAF  
- Face detection + thirds adherence  
- Depth (MiDaS)  
- Genre-based shot timing  
- Grading uniformity  
- Chroma noise models  

---

## Creative Metrics (placeholders)

- Lighting style adherence (vs. references)  
- Composition creativity / novelty detection  
- Camera movement impact (story beat alignment)  
- Palette emotional accuracy (color psychology mapping)  
- Motif/repetition tracking  

---

## Subjective Metrics (placeholders)

- Perceived clarity (crowd/expert ratings)  
- Emotional response (sentiment analysis)  
- Engagement (viewer focus, eye-tracking if available)  
- Aesthetic impression (beauty scoring models)  
- Memorability / iconic frame detection  

---

## Hero-Scenes Video Export

- Export stitched `hero_scenes.mp4` around each selected frame  
- Seconds before/after per config  
- Deduplicate/merge overlaps  
- Cap total duration (default 45s)  
- Fade 6–8 frames between clips  
- Optional title card or overlay captions  
- Contact sheet preview (`contact_sheet.jpg`)  

---

## GUI (Main Window)

- **Input/Output controls**: file/URL chooser, run buttons  
- **Verbose status window**: per-stage logs (ingest, scenes, sampling, features, selection, export)  
- **Progress bar**: live percentage + message  
- **Action buttons**: Open Output, Open Logs, Copy Error, Kill Worker, Show EDL, Rerun with Seed  
- **Always responsive** (pipeline runs in background thread)  

---

## Roadmap

- [x] Config loading + normalization  
- [x] Ingest, scene detection, sampling  
- [x] Cheap technical metrics  
- [x] k-medoids + facility-location  
- [x] Deduplication  
- [x] Basic GUI + logging  
- [x] CLIP integration (in-process)  

- [ ] Always-write invariant (fallback top-K)  
- [ ] CLIP subprocess + timeout handling  
- [ ] Verbose GUI status pane  
- [ ] Hero-scenes export (MVP)  
- [ ] Feature caching  

- [ ] Advanced metrics (SSIM, VMAF, depth)  
- [ ] Creative pillar metrics  
- [ ] Subjective pillar metrics  
- [ ] Contact sheets + manifest export  
- [ ] Profile weighting by DP/film  

---

## Acceptance Criteria

- Every valid run writes ≥ `top_k` frames with sidecars  
- Optional hero-scenes video within configured cap  
- `run_manifest.json` records all metrics + selection decisions  
- Identical runs (same config + seed) → identical outputs  
- Failures/timeouts degrade gracefully (frames still written, logs show `"timed_out": true`)  
