"""Profile packaging utilities for Stream Deck."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from typing import Dict, Iterable, List

from .icons import render_icons
from .models import Button, Page, Profile


def create_streamdeck_profile(profile: Profile, output_basename: str) -> str:
    """Create a `.streamDeckProfile` archive from the supplied profile."""

    temp_dir = tempfile.mkdtemp(prefix="streamdeck_")

    try:
        icons_dir = os.path.join(temp_dir, "Icons")
        render_icons(profile.pages, icons_dir)

        manifest = _build_manifest(profile)

        with open(os.path.join(temp_dir, "manifest.json"), "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

        archive_path = f"{output_basename}.streamDeckProfile"
        if os.path.exists(archive_path):
            os.remove(archive_path)

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive.write(file_path, os.path.relpath(file_path, temp_dir))

        return archive_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _build_manifest(profile: Profile) -> Dict[str, object]:
    """Create a Stream Deck 7 manifest payload."""

    manifest: Dict[str, object] = {
        "Actions": {},
        "Author": "StreamDeck Profile Generator",
        "Category": "Automation",
        "CategoryIcon": "",
        "Description": f"StreamDeck profile for {profile.software_title}",
        "Name": f"{profile.software_title} Profile",
        "Version": "1.0.0",
        "DeviceType": 7,
        "Icon": "",
        "URL": "",
        "OS": [{"Platform": "windows", "MinimumVersion": "10"}],
        "Software": {"MinimumVersion": "6.0"},
        "ApplicationsToMonitor": {
            "windows": _candidate_executables(profile.software_title),
        },
        "Profiles": [
            {
                "Name": f"{profile.software_title} Profile",
                "DeviceType": 7,
                "Pages": [],
            }
        ],
    }

    pages_payload = manifest["Profiles"][0]["Pages"]  # type: ignore[index]

    for page in profile.pages:
        page_payload: Dict[str, object] = {"Name": page.name, "Buttons": {}}
        page_payload["Buttons"].update(_serialise_buttons(page.buttons))  # type: ignore[index]
        pages_payload.append(page_payload)  # type: ignore[arg-type]

    return manifest


def _candidate_executables(software_title: str) -> List[str]:
    """Produce executable name candidates for Stream Deck monitoring."""

    slug = software_title.lower().replace(" ", "")
    underscored = software_title.replace(" ", "_")
    return [
        f"{slug}.exe",
        f"{underscored}.exe",
        f"{software_title}.exe",
    ]


def _serialise_buttons(buttons: Iterable[Button]) -> Dict[str, object]:
    """Convert non-empty buttons into Stream Deck manifest entries."""

    serialised: Dict[str, object] = {}

    for button in buttons:
        if button.action == "noop" or button.title == "—":
            continue

        button_key = f"{button.col},{button.row}"
        button_payload: Dict[str, object] = {
            "Name": button.title,
            "State": 0,
            "States": [
                {
                    "Image": f"Icons/{button.icon}" if button.icon else "",
                    "Title": button.title,
                    "TitleAlignment": "middle",
                    "FontSize": "16",
                }
            ],
        }

        action = button.action
        if action.startswith("hotkey:"):
            button_payload.update(
                {
                    "UUID": "com.elgato.streamdeck.system.hotkey",
                    "Settings": {"Hotkey": action.replace("hotkey:", "", 1)},
                }
            )
        elif action.startswith("run:"):
            button_payload.update(
                {
                    "UUID": "com.elgato.streamdeck.system.open",
                    "Settings": {"Path": action.replace("run:", "", 1)},
                }
            )
        elif action.startswith("switch_page:") or action.startswith("profile:"):
            button_payload.update(
                {
                    "UUID": "com.elgato.streamdeck.system.profile",
                    "Settings": {"ProfileUUID": "", "ProfileName": action.split(":", 1)[1]},
                }
            )
        else:
            button_payload.update({"UUID": "com.elgato.streamdeck.system.blank", "Settings": {}})

        serialised[button_key] = button_payload

    return serialised
