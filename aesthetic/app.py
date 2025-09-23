# -*- coding: utf-8 -*-
"""
Application entry point (GUI launcher + CLI).
Run:
  python -m aesthetic.app --in "E:\\AestheticApp\\aesthetic\\data\\tests\\clip.mp4"
Or launch GUI:
  python -m aesthetic.app --gui
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from aesthetic.core import pipeline
from aesthetic.core.utils import load_config_file

LOG = logging.getLogger("aesthetic.app")


def _apply_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Map CLI flags to config override keys."""
    overrides: Dict[str, Any] = {}
    if args.out:
        overrides["output.folder"] = str(Path(args.out))
    if args.seed is not None:
        overrides["sampling.random_seed"] = int(args.seed)
    if args.top_k is not None:
        overrides["select.top_k"] = int(args.top_k)
    if args.hero_video is not None:
        hv = args.hero_video.strip().lower()
        overrides["output.hero_video.enabled"] = hv in {"on", "1", "true", "yes"}
    if args.clip_mode:
        mode = args.clip_mode.strip().lower()
        if mode == "off":
            overrides["heavy.clip.enabled"] = False
        else:
            overrides["heavy.clip.run_mode"] = mode
            overrides["heavy.clip.enabled"] = True
    return overrides


def _run_cli(args: argparse.Namespace) -> int:
    """CLI execution path."""
    cfg = load_config_file(Path(args.config) if args.config else None)
    overrides = _apply_overrides(args)
    try:
        result = pipeline.run_pipeline(
            args.input,
            config=cfg,
            cli_overrides=overrides,
            progress_cb=lambda frac, msg: LOG.info("%03d%% %s", int(frac * 100), msg),
        )
    except Exception as exc:  # pragma: no cover
        LOG.exception("pipeline failed: %s", exc)
        return 1
    LOG.info("wrote %d frames", len(result.get("written", [])))
    LOG.info("manifest: %s", result.get("manifest"))
    return 0


def _run_gui(config_path: Optional[str]) -> int:
    """GUI execution path."""
    try:
        # Import lazily so CLI runs in headless environments.
        from aesthetic.gui.main_window import MainWindow
        import tkinter as tk  # noqa
    except Exception as exc:  # pragma: no cover
        LOG.exception("failed to start GUI: %s", exc)
        return 1
    win = MainWindow(Path(config_path) if config_path else None)
    win.mainloop()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AESTHETIC hero-frame selector")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI")
    parser.add_argument("--in", dest="input", help="Input video path or URL")
    parser.add_argument("--out", dest="out", help="Output directory override")
    parser.add_argument("--seed", dest="seed", type=int, help="Random seed override")
    parser.add_argument("--top_k", dest="top_k", type=int, help="Top-K frames to keep")
    parser.add_argument("--hero-video", dest="hero_video", help="Hero video: on/off")
    parser.add_argument("--clip-mode", dest="clip_mode", help="CLIP: off|subprocess|inproc")
    parser.add_argument("--config", dest="config", default="", help="Path to config.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.gui or not args.input:
        return _run_gui(args.config or None)
    return _run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
