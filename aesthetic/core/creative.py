"""Placeholder creative-pillar metrics.

Each helper returns a ``{metric_name: Optional[float]}`` mapping so the pipeline can
log deterministic keys even before the creative models are implemented.  Keeping
these helpers isolated makes it straightforward to replace the internals later
without touching the orchestrator or the sidecar schema.
"""

from __future__ import annotations

from typing import Dict, Optional


def lighting_style_adherence(reference_profile: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return placeholder lighting-style adherence metrics."""

    return {"lighting_style_adherence": None}


def composition_creativity(reference_sets: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return placeholder composition-creativity metrics."""

    return {"composition_creativity": None}


def movement_impact(context_tags: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return placeholder movement-impact metrics."""

    return {"movement_impact": None}


def palette_mood_accuracy(ref_mood: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return placeholder palette-mood metrics."""

    return {"palette_mood_accuracy": None}


def motif_repetition(reference_tags: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return placeholder motif-repetition metrics."""

    return {"motif_repetition": None}


__all__ = [
    "lighting_style_adherence",
    "composition_creativity",
    "movement_impact",
    "palette_mood_accuracy",
    "motif_repetition",
]
