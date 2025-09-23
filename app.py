"""Application entry point (GUI launcher + CLI)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from aesthetic.core import pipeline
from aesthetic.core.utils import load_config_file


LOG = logging.getLogger("aesthetic.app")


def _apply_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.out:
        overrides["output.folder"] = str(Path(args.out))
    if args.seed is not None:
        overrides["sampling.random_seed"] = int(args.seed)
    if args.top_k is not None:
        overrides["select.top_k"] = int(args.top_k)
    if args.hero_video is not None:
        overrides["output.hero_video.enabled"] = args.hero_video.lower() == "on"
    if args.clip_mode:
        mode = args.clip_mode.lower()
        if mode == "off":
            overrides["heavy.clip.enabled"] = False
        else:
            overrides["heavy.clip.run_mode"] = mode
            overrides["heavy.clip.enabled"] = True
    return overrides


def _run_cli(args: argparse.Namespace) -> int:
    cfg = load_config_file(args.config)
    overrides = _apply_overrides(args)
    try:
        result = pipeline.run_pipeline(
            args.input,
            config=cfg,
            cli_overrides=overrides,
            progress_cb=lambda frac, msg: LOG.info("%03d%% %s", int(frac * 100), msg),
        )
    except Exception as exc:  # pragma: no cover - CLI feedback
        LOG.exception("pipeline failed: %s", exc)
        return 1
    LOG.info("wrote %d frames", len(result.get("written", [])))
    LOG.info("manifest: %s", result.get("manifest"))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AESTHETIC video hero-frame selector")
    parser.add_argument("--in", dest="input", help="Input video path or URL")
    parser.add_argument("--out", dest="out", help="Output directory override")
    parser.add_argument("--seed", dest="seed", type=int, help="Random seed override")
    parser.add_argument("--top_k", dest="top_k", type=int, help="Top-K frames to keep")
    parser.add_argument("--hero-video", dest="hero_video", choices=["on", "off"], help="Toggle hero video export")
    parser.add_argument("--clip", dest="clip_mode", choices=["subprocess", "inprocess", "off"], help="Select CLIP execution mode")
    parser.add_argument("--config", dest="config", type=Path, help="Explicit config path")
    parser.add_argument("--no-gui", dest="no_gui", action="store_true", help="Run headless even without --in")

    args = parser.parse_args(argv)

    if args.input or args.no_gui:
        if not args.input:
            parser.error("--no-gui requires --in to be set")
        return _run_cli(args)

    from aesthetic.main_window import run_app

    run_app(config_path=args.config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
