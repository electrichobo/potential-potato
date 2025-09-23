"""Placeholder subjective-pillar metrics."""

from __future__ import annotations

from typing import Dict, Optional


def perceived_clarity(ratings_db: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return placeholder perceived-clarity metrics."""

    return {"perceived_clarity": None}


def emotional_response(ratings_db: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return placeholder emotional-response metrics."""

    return {"emotional_response": None}


def engagement_signals(telemetry: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return placeholder engagement-signal metrics."""

    return {"engagement": None}


def aesthetic_impression(model: Optional[object] = None) -> Dict[str, Optional[float]]:
    """Return placeholder aesthetic-impression metrics."""

    return {"aesthetic_impression": None}


def memorability_iconic(reference_sets: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Return placeholder memorability metrics."""

    return {"memorability": None}


__all__ = [
    "perceived_clarity",
    "emotional_response",
    "engagement_signals",
    "aesthetic_impression",
    "memorability_iconic",
]