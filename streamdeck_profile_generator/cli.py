"""Command line entry-point for Stream Deck profile generation."""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional

from .layout import build_profile_pages
from .models import Profile
from .packaging import create_streamdeck_profile


def generate_profile(software_title: str) -> Profile:
    """Generate a profile for the supplied software title."""

    pages = build_profile_pages(software_title)
    return Profile(software_title=software_title, pages=pages)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point used by `python -m streamdeck_profile_generator.cli`."""

    if argv is None:
        argv = sys.argv[1:]

    if argv:
        software = " ".join(argv).strip()
    else:
        try:
            software = input("Software title: ").strip()
        except EOFError:
            software = ""

    if not software:
        print("Aborted: no software title provided.")
        return 1

    try:
        print(f"Generating StreamDeck profile for: {software}")
        profile = generate_profile(software)
        print(f"Profile generated with {len(profile.pages)} page(s)")

        output_basename = os.path.abspath(software.replace(" ", "_") + "_XL")
        print(f"Creating profile file: {output_basename}.streamDeckProfile")

        archive = create_streamdeck_profile(profile, output_basename)
        print(f"✅ SUCCESS: Generated StreamDeck Profile: {archive}")
        print("📁 Import this .streamDeckProfile file into Stream Deck software.")
        return 0
    except Exception as exc:  # pragma: no cover - defensive CLI logging
        print("❌ ERROR: Failed to generate Stream Deck profile.")
        print("--- Recursive traceback (most recent call last) ---")
        for line in traceback.TracebackException.from_exception(exc).format(chain=True):
            print(line, end="")
        print("--- End traceback ---")
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI behavior
    raise SystemExit(main())
