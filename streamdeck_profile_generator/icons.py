"""Icon rendering helpers for Stream Deck profiles."""

from __future__ import annotations

import os
from typing import Iterable

from .models import Page

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None


def render_icons(pages: Iterable[Page], outdir: str) -> None:
    """Render simple text icons for all non-empty buttons."""

    if Image is None:
        return

    os.makedirs(outdir, exist_ok=True)
    width, height = 144, 144
    bg = (24, 24, 28)
    fg = (235, 235, 245)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 28)
        except Exception:
            font = ImageFont.load_default()

    for page in pages:
        for button in page.buttons:
            if button.title == "—" or button.action == "noop":
                continue

            text = button.title or ""
            image = Image.new("RGB", (width, height), bg)
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle([(2, 2), (width - 3, height - 3)], radius=18, outline=(80, 80, 90), width=3)

            wrapped: list[str] = []
            for piece in text.replace("\n", " ").split():
                if not wrapped:
                    wrapped.append(piece)
                else:
                    candidate = f"{wrapped[-1]} {piece}"
                    if draw.textlength(candidate, font=font) <= width - 24:
                        wrapped[-1] = candidate
                    else:
                        wrapped.append(piece)
                if len(wrapped) == 3:
                    break

            line_height = int(font.size * 1.1)
            block_height = len(wrapped) * line_height
            y_pos = (height - block_height) // 2
            for line in wrapped:
                text_width = draw.textlength(line, font=font)
                x_pos = (width - text_width) // 2
                draw.text((x_pos, y_pos), line, fill=fg, font=font)
                y_pos += line_height

            icon_name = f"{_random_icon_name()}.png"
            image.save(os.path.join(outdir, icon_name), "PNG")
            button.icon = icon_name


def _random_icon_name(length: int = 27) -> str:
    """Generate a pseudo-random icon filename that mirrors Stream Deck IDs."""

    import random
    import string

    characters = string.ascii_uppercase + string.digits
    return "".join(random.choice(characters) for _ in range(length))
