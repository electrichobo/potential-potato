"""Deterministic fallback layout for Stream Deck profiles."""

from __future__ import annotations

from typing import List

from .models import BUTTONS_PER_PAGE, GRID_COLS, Button, Page


def build_default_pages() -> List[Page]:
    """Return a minimal multi-page layout when OpenAI is unavailable."""

    pages: List[Page] = []

    master_buttons = [
        Button(0, 0, "Combat", "switch_page:Combat"),
        Button(1, 0, "Inventory", "switch_page:Inventory"),
        Button(2, 0, "Map", "switch_page:Map"),
        Button(3, 0, "Quests", "switch_page:Quests"),
        Button(4, 0, "Settings", "switch_page:Settings"),
        Button(5, 0, "Shortcuts", "switch_page:Shortcuts"),
        Button(6, 0, "Help", "switch_page:Help"),
        Button(7, 3, "Default", "profile:Default"),
    ]

    master_buttons.extend(_fill_remaining_slots(master_buttons))
    pages.append(Page("Master", master_buttons))

    combat_buttons = [
        Button(0, 0, "Space", "hotkey:Space"),
        Button(1, 0, "Group", "hotkey:G"),
        Button(2, 0, "Hide", "hotkey:H"),
        Button(3, 0, "Sneak", "hotkey:C"),
        Button(4, 0, "Shove", "hotkey:R"),
        Button(5, 0, "Throw", "hotkey:T"),
        Button(6, 0, "End Turn", "hotkey:Enter"),
        Button(7, 0, "Highlight", "hotkey:Tab"),
        Button(0, 1, "Select 1", "hotkey:1"),
        Button(1, 1, "Select 2", "hotkey:2"),
        Button(2, 1, "Select 3", "hotkey:3"),
        Button(3, 1, "Select 4", "hotkey:4"),
        Button(4, 1, "Center 1", "hotkey:F1"),
        Button(5, 1, "Center 2", "hotkey:F2"),
        Button(6, 1, "Center 3", "hotkey:F3"),
        Button(7, 1, "Center 4", "hotkey:F4"),
        Button(0, 3, "RTB", "switch_page:Master"),
    ]

    combat_buttons.extend(_fill_remaining_slots(combat_buttons, include_master_controls=False))
    pages.append(Page("Combat", combat_buttons))

    inventory_buttons = [
        Button(0, 0, "Inventory", "hotkey:I"),
        Button(1, 0, "Spellbook", "hotkey:B"),
        Button(2, 0, "Character", "hotkey:K"),
        Button(3, 0, "Journal", "hotkey:L"),
        Button(4, 0, "Interact", "hotkey:F"),
        Button(5, 0, "Examine", "hotkey:E"),
        Button(6, 0, "Quick Save", "hotkey:F5"),
        Button(7, 0, "Quick Load", "hotkey:F8"),
        Button(0, 3, "RTB", "switch_page:Master"),
    ]

    inventory_buttons.extend(_fill_remaining_slots(inventory_buttons, include_master_controls=False))
    pages.append(Page("Inventory", inventory_buttons))

    map_buttons = [
        Button(0, 0, "Map", "hotkey:M"),
        Button(1, 0, "Screenshot", "hotkey:F10"),
        Button(2, 0, "Fullscreen", "hotkey:F11"),
        Button(3, 0, "Menu", "hotkey:Esc"),
        Button(0, 3, "RTB", "switch_page:Master"),
    ]

    map_buttons.extend(_fill_remaining_slots(map_buttons, include_master_controls=False))
    pages.append(Page("Map", map_buttons))

    return pages


def _fill_remaining_slots(existing: List[Button], include_master_controls: bool = True) -> List[Button]:
    """Pad a page with placeholder buttons while respecting RTB/default rules."""

    grid = {(button.col, button.row) for button in existing}
    placeholders: List[Button] = []

    for idx in range(BUTTONS_PER_PAGE):
        col = idx % GRID_COLS
        row = idx // GRID_COLS

        if (col, row) in grid:
            continue

        if include_master_controls and col == 7 and row == 3:
            placeholders.append(Button(col, row, "Default", "profile:Default"))
        elif not include_master_controls and col == 0 and row == 3:
            placeholders.append(Button(col, row, "RTB", "switch_page:Master"))
        else:
            placeholders.append(Button(col, row, "—", "noop"))

    return placeholders
