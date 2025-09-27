"""Layout planning utilities for Stream Deck profile generation."""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List, cast

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

from .models import BUTTONS_PER_PAGE, GRID_COLS, GRID_ROWS, Button, Page

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LayoutGenerationError(RuntimeError):
    """Raised when an AI powered layout cannot be produced."""


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def _build_openai_client() -> "OpenAI":
    """Create an OpenAI client or raise a detailed error."""

    if OpenAI is None:
        raise LayoutGenerationError(
            "openai package is not installed; install `openai` to enable layout generation"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LayoutGenerationError(
            "OPENAI_API_KEY is not set in the environment; export a key before running"
        )

    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:  # pragma: no cover - network configuration issues
        raise LayoutGenerationError("failed to initialise OpenAI client") from exc


def call_openai_layout(software_title: str) -> Dict[str, Any]:
    """Request a layout suggestion from the OpenAI API."""

    client = _build_openai_client()

    prompt = textwrap.dedent(
        f"""
        You are designing pages for an Elgato Stream Deck XL (8x4 grid).
        Return JSON with a `pages` list where each page has a `name` and a
        `buttons` list. Every button requires `col`, `row`, `title`, and
        `action` fields. Reserve column 0, row 3 on every non-master page for
        an RTB button that navigates back to the Master page. Reserve column 7,
        row 3 on the Master page for a button that switches to the Default
        profile. Software title: {software_title}.
        """
    ).strip()

    try:
        response = client.responses.create(  # type: ignore[attr-defined]
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=prompt,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise LayoutGenerationError("OpenAI layout request failed") from exc

    try:
        content = response.output[0].content[0].text  # type: ignore[index]
        return cast(Dict[str, Any], json.loads(content))
    except Exception as exc:
        raise LayoutGenerationError("OpenAI layout response was not valid JSON") from exc


# ---------------------------------------------------------------------------
# Layout sanitising
# ---------------------------------------------------------------------------

def sanitize_pages(ai_response: Dict[str, Any]) -> List[Page]:
    """Normalise an AI layout response into Page objects."""

    if "pages" not in ai_response:
        raise LayoutGenerationError("OpenAI response did not include a `pages` field")

    pages: List[Page] = []

    for page_data in ai_response.get("pages", []):
        name = str(page_data.get("name", "Custom"))
        button_grid: Dict[int, Button] = {}

        for raw_button in page_data.get("buttons", []):
            try:
                col = max(0, min(GRID_COLS - 1, int(raw_button.get("col", 0))))
                row = max(0, min(GRID_ROWS - 1, int(raw_button.get("row", 0))))
                title = str(raw_button.get("title", "")).strip()[:16] or "—"
                action = str(raw_button.get("action", "noop")).strip() or "noop"
                idx = row * GRID_COLS + col
                button_grid[idx] = Button(col, row, title, action)
            except Exception:
                continue

        buttons: List[Button] = []
        for idx in range(BUTTONS_PER_PAGE):
            col = idx % GRID_COLS
            row = idx // GRID_COLS
            if idx in button_grid:
                buttons.append(button_grid[idx])
            elif name != "Master" and col == 0 and row == 3:
                buttons.append(Button(col, row, "RTB", "switch_page:Master"))
            elif name == "Master" and col == 7 and row == 3:
                buttons.append(Button(col, row, "Default", "profile:Default"))
            else:
                buttons.append(Button(col, row, "—", "noop"))

        pages.append(Page(name, buttons))

    if not pages:
        raise LayoutGenerationError("OpenAI response did not define any pages")

    if not any(page.name == "Master" for page in pages):
        raise LayoutGenerationError("OpenAI response omitted a Master page")

    return pages


def build_profile_pages(software_title: str) -> List[Page]:
    """Resolve pages via OpenAI or raise if the request cannot be satisfied."""

    ai_layout = call_openai_layout(software_title)
    return sanitize_pages(ai_layout)
