"""Placeholder implementations for optional heavy metrics.

Advanced metrics (SSIM, VMAF, depth, etc.) will be filled in later.  For now we
return deterministic placeholders so the rest of the pipeline can toggle the
module on/off without raising errors.  Each metric provider returns a mapping of
metric name ➜ float|None as per the coding guidelines.
"""

from __future__ import annotations

from typing import Dict, Optional


def compute_advanced_metrics(frame_bgr, config: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return deterministic placeholder metrics for advanced pillars."""

    if config is None or not (config.get("advanced_metrics") or {}).get("enabled", False):
        return {}

    return {
        "advanced_exposure": None,
        "advanced_lighting": None,
        "advanced_composition": None,
        "advanced_movement": None,
        "advanced_color": None,
    }


__all__ = ["compute_advanced_metrics"]