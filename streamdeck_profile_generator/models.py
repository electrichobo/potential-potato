"""Domain models for Stream Deck profile generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

GRID_COLS = 8
GRID_ROWS = 4
BUTTONS_PER_PAGE = GRID_COLS * GRID_ROWS


@dataclass
class Button:
    """Represents a single Stream Deck button."""

    col: int
    row: int
    title: str
    action: str
    icon: Optional[str] = None
    tooltip: Optional[str] = None


@dataclass
class Page:
    """Represents a Stream Deck page (folder) of buttons."""

    name: str
    buttons: List[Button]


@dataclass
class Profile:
    """Holds the generated profile metadata."""

    software_title: str
    pages: List[Page]
