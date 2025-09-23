"""Hero-scene export helpers.

This module focuses on the final export stages of the pipeline.  The goals are:

* Convert selected frames into time segments around each hero frame.
* Merge and clamp those segments so they remain within duration + budget caps.
* Invoke ``ffmpeg`` deterministically while capturing stderr for diagnostics.
* Optionally build a contact sheet preview that mirrors the top selections.

The functions are intentionally small and composable so future contributors can
swap out the export strategy (e.g. hardware acceleration) without rewriting the
callers inside :mod:`aesthetic.core.pipeline`.
"""

from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


LOG = logging.getLogger("aesthetic.video_export")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoInfo:
    """Describe the source media so downstream helpers can make safe decisions."""

    fps: float
    frame_count: int
    duration: float


def _find_scene(spans: Sequence[Tuple[int, int]], frame_index: int) -> Tuple[int, int]:
    """Return the scene boundaries that contain *frame_index*.

    A fallback is provided so we never raise if detection produced no spans.
    """

    for start, end in spans:
        if start <= frame_index <= end:
            return start, end
    if spans:
        return spans[0]
    return frame_index, frame_index


def _merge_segments(segments: List[Tuple[float, float]], fade: float) -> List[Tuple[float, float]]:
    """Merge overlapping clips while accounting for cross-fade padding."""

    if not segments:
        return []
    segments.sort()
    merged: List[Tuple[float, float]] = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end - fade:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _cap_segments(segments: List[Tuple[float, float]], cap: float) -> List[Tuple[float, float]]:
    """Clip the stitched duration to the configured *cap* (seconds)."""

    out: List[Tuple[float, float]] = []
    total = 0.0
    for start, end in segments:
        duration = max(0.0, end - start)
        if total + duration <= cap:
            out.append((start, end))
            total += duration
            continue
        remaining = max(0.0, cap - total)
        if remaining <= 0.0:
            break
        out.append((start, start + remaining))
        break
    return out


def build_segments(
    selections: Sequence[Mapping[str, Any]],
    spans: Sequence[Tuple[int, int]],
    info: VideoInfo,
    hero_cfg: Mapping[str, Any],
) -> List[Dict[str, float]]:
    """Build time segments around each selected frame.

    The caller passes the selections (typically candidates serialised to dicts),
    scene spans expressed in frame indices, and the known video info.  We jitter
    around each frame according to the config, clamp to scene boundaries, merge
    the windows, and cap the total stitched duration.  All durations are in
    seconds.
    """

    before = float(hero_cfg.get("seconds_before", 0.8))
    after = float(hero_cfg.get("seconds_after", 1.2))
    fps = info.fps if info.fps > 0 else float(hero_cfg.get("fps", 30))
    fps = max(1.0, float(fps))
    fade = max(0.0, 6.0 / fps)
    max_duration = max(0.0, float(hero_cfg.get("max_total_duration", 45.0)))

    raw_segments: List[Tuple[float, float]] = []
    video_cap = info.duration if info.duration > 0 else None
    for sel in selections:
        frame_index = int(sel.get("frame_index", 0))
        scene_start, scene_end = _find_scene(spans, frame_index)
        start_frame = max(scene_start, int(frame_index - before * fps))
        end_frame = min(scene_end, int(frame_index + after * fps))
        start = max(0.0, start_frame / fps)
        end = max(start + fade, end_frame / fps)
        if video_cap is not None:
            end = min(end, video_cap)
        raw_segments.append((start, end))

    merged = _merge_segments(raw_segments, fade)
    capped = _cap_segments(merged, max_duration)
    return [
        {"start": s, "end": e, "duration": max(0.0, e - s)}
        for s, e in capped
    ]


def _ffmpeg_command(
    source: Path,
    segments: Sequence[Mapping[str, float]],
    fps: float,
    fade: float,
    output_path: Path,
    ffmpeg_bin: str,
) -> List[str]:
    """Build an ``ffmpeg`` command to stitch *segments* with optional crossfades."""

    cmd: List[str] = [ffmpeg_bin, "-y"]
    for seg in segments:
        cmd += ["-ss", f"{seg['start']:.3f}", "-to", f"{seg['end']:.3f}", "-i", str(source)]

    if len(segments) == 1:
        cmd += ["-r", f"{fps:.3f}", "-pix_fmt", "yuv420p", "-an", str(output_path)]
        return cmd

    filter_parts: List[str] = []
    labels: List[str] = []
    for idx, _ in enumerate(segments):
        label = f"v{idx}"
        filter_parts.append(
            f"[{idx}:v:0]setpts=PTS-STARTPTS,fps={fps:.3f},format=yuv420p[{label}]"
        )
        labels.append(f"[{label}]")

    current_label = labels[0]
    offset = segments[0]["duration"] - fade
    for idx in range(1, len(labels)):
        next_label = labels[idx]
        out_label = f"vxf{idx}"
        off = max(0.0, offset)
        filter_parts.append(
            f"{current_label}{next_label}xfade=transition=fade:duration={fade:.3f}:offset={off:.3f}[{out_label}]"
        )
        current_label = f"[{out_label}]"
        offset += segments[idx]["duration"] - fade

    filter_complex = ";".join(filter_parts)
    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        current_label,
        "-r",
        f"{fps:.3f}",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    return cmd


def _run_ffmpeg(cmd: List[str]) -> Tuple[bool, str]:
    """Execute *cmd* returning ``(ok, stderr)`` while logging failures."""

    LOG.info("ffmpeg command: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)
    if res.returncode != 0:
        LOG.warning("ffmpeg failed (%s): %s", res.returncode, res.stderr.strip())
        return False, res.stderr
    return True, res.stderr


def _contact_sheet(frames: Sequence[np.ndarray], path: Path, columns: int = 3) -> Optional[Path]:
    """Write a simple contact sheet for quick QA. Returns the path on success."""

    if not frames:
        return None
    cols = max(1, columns)
    rows = int(math.ceil(len(frames) / cols))
    h, w = frames[0].shape[:2]
    sheet = np.full((rows * h, cols * w, 3), 16, dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
    if cv2.imwrite(str(path), sheet):
        return path
    return None


def export_hero_video(
    source_path: str,
    output_dir: Path,
    selections: Sequence[Mapping[str, Any]],
    spans: Sequence[Tuple[int, int]],
    info: VideoInfo,
    hero_cfg: Mapping[str, Any],
    frames: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Export a stitched hero video and optional contact sheet.

    The function degrades gracefully: when disabled or if ``ffmpeg`` fails we
    still return a result dictionary explaining what happened so the caller can
    surface status in the GUI/logs without raising.
    """

    result: Dict[str, Any] = {"enabled": bool(hero_cfg.get("enabled", True))}
    if not result["enabled"] or not selections:
        return result

    segments = build_segments(selections, spans, info, hero_cfg)
    result["segments"] = segments
    if not segments:
        result["video_path"] = None
        result["error"] = "no segments"
        return result

    fps = info.fps if info.fps > 0 else float(hero_cfg.get("fps", 30))
    fps = max(1.0, float(fps))
    fade = max(0.0, 6.0 / fps)
    ffmpeg_bin = str(hero_cfg.get("ffmpeg_bin", "ffmpeg"))

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "hero_scenes.mp4"
    cmd = _ffmpeg_command(Path(source_path), segments, fps, fade, video_path, ffmpeg_bin)
    ok, stderr = _run_ffmpeg(cmd)
    result["video_path"] = str(video_path) if ok else None
    result["ffmpeg_ok"] = ok
    if not ok:
        result["error"] = stderr.strip()

    if hero_cfg.get("contact_sheet", True) and frames:
        sheet_path = output_dir / "contact_sheet.jpg"
        sheet = _contact_sheet(frames, sheet_path)
        result["contact_sheet"] = str(sheet) if sheet else None
    else:
        result["contact_sheet"] = None

    return result


__all__ = ["VideoInfo", "build_segments", "export_hero_video"]
