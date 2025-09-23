# -*- coding: utf-8 -*-
"""Tkinter GUI with verbose status panel for AESTHETIC.

Enhancements:
- Source field accepts local file paths OR URLs; "Paste URL" helper.
- Checkbox to toggle hero video export on/off per run.
- "Open Logs" opens the logs folder in Explorer/Finder.
"""

from __future__ import annotations

import logging, os, subprocess, sys, threading, tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from aesthetic.core import pipeline
from aesthetic.core.utils import ensure_output_tree, load_config_file, prepare_config

LOG = logging.getLogger("aesthetic.gui")

def _is_url(s: str) -> bool:
    try:
        u = urlparse(s.strip())
        return bool(u.scheme) and u.scheme.lower() in {"http", "https", "file"}
    except Exception:
        return False

PLACEHOLDER = "File path or YouTube/Vimeo URL"
PLACEHOLDER_COLOR = "#888"
ENTRY_COLOR = "#000"

class StagePanel(ttk.Frame):
    def __init__(self, master: tk.Widget, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.vars: Dict[str, tk.StringVar] = {}
        stages = [
            ("ingest", "Ingest"),
            ("scene_detection", "Scene detection"),
            ("sampling", "Sampling"),
            ("features", "Features"),
            ("selection", "Selection"),
            ("dedup", "Dedup"),
            ("export", "Export"),
        ]
        for row, (key, label) in enumerate(stages):
            ttk.Label(self, text=label + ":", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="nw", padx=(0, 6), pady=2)
            var = tk.StringVar(value="pending")
            ttk.Label(self, textvariable=var, wraplength=240, justify="left").grid(row=row, column=1, sticky="w", pady=2)
            self.vars[key] = var

    def update_stage(self, key: str, text: str) -> None:
        if key in self.vars:
            self.vars[key].set(text)

class MainWindow(tk.Tk):
    def __init__(self, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.title("AESTHETIC — Hero Frame Selector")
        self.geometry("1200x720")
        self.minsize(1000, 600)

        self.config_path = config_path
        self.config_dict = prepare_config(load_config_file(config_path))
        ensure_output_tree(self.config_dict)

        self.progress = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="idle")
        self.error_text = ""
        self.last_manifest: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(3, weight=1)

        # Left column ---------------------------------------------------
        input_frame = ttk.LabelFrame(container, text="Source")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.source_var = tk.StringVar()
        self.source_entry = ttk.Entry(input_frame, textvariable=self.source_var, width=64)
        self.source_entry.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        ttk.Button(input_frame, text="Browse…", command=self._browse).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(input_frame, text="Paste URL", command=self._paste_url).grid(row=0, column=2, padx=(0, 6), pady=6)
        input_frame.columnconfigure(0, weight=1)

        # Placeholder behavior
        self._apply_placeholder()
        self.source_entry.bind("<FocusIn>", self._placeholder_focus_in)
        self.source_entry.bind("<FocusOut>", self._placeholder_focus_out)

        # Run options row
        options = ttk.Frame(container)
        options.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.export_var = tk.BooleanVar(value=bool(((self.config_dict.get("output") or {}).get("hero_video") or {}).get("enabled", True)))
        ttk.Checkbutton(options, text="Export hero video", variable=self.export_var).pack(side="left")
        ttk.Label(options, text=" ").pack(side="left")  # spacer

        # Buttons row
        buttons = ttk.Frame(container)
        buttons.grid(row=2, column=0, sticky="w", pady=8)
        ttk.Button(buttons, text="Run", command=self._run_async).pack(side="left")
        ttk.Button(buttons, text="Reload config", command=self._reload_config).pack(side="left", padx=6)
        ttk.Button(buttons, text="Open Output", command=self._open_output).pack(side="left", padx=6)
        ttk.Button(buttons, text="Open Logs", command=self._open_logs).pack(side="left", padx=6)
        ttk.Button(buttons, text="Rerun with Seed…", command=self._rerun_with_seed).pack(side="left", padx=6)

        # Extra controls
        extra_buttons = ttk.Frame(container)
        extra_buttons.grid(row=2, column=1, sticky="e", pady=8)
        self.copy_btn = ttk.Button(extra_buttons, text="Copy Error", command=self._copy_error, state="disabled")
        self.copy_btn.pack(side="left", padx=6)

        # Progress + log
        ttk.Label(container, textvariable=self.status_var).grid(row=3, column=0, sticky="w")
        ttk.Progressbar(container, variable=self.progress, maximum=1.0).grid(row=3, column=0, sticky="we", pady=(4, 8))
        self.log = tk.Text(container, height=16)
        self.log.grid(row=4, column=0, sticky="nsew")

        # Right column --------------------------------------------------
        status_frame = ttk.LabelFrame(container, text="Status & Logs")
        status_frame.grid(row=0, column=1, rowspan=5, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)

        device_info = ttk.LabelFrame(status_frame, text="Device")
        device_info.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        gpu_cfg = self.config_dict.get("gpu") or {}
        heavy_clip = (self.config_dict.get("heavy") or {}).get("clip", {})
        info_lines = [
            f"GPU: {'enabled' if gpu_cfg.get('enabled', True) else 'disabled'} ({gpu_cfg.get('device', 'cpu')})",
            f"CLIP: {'on' if heavy_clip.get('enabled', True) else 'off'} mode={heavy_clip.get('run_mode', 'subprocess')}",
        ]
        ttk.Label(device_info, text="\n".join(info_lines), justify="left").grid(row=0, column=0, sticky="w")

        self.stages = StagePanel(status_frame)
        self.stages.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        status_frame.rowconfigure(1, weight=1)

    # Placeholder mechanics
    def _apply_placeholder(self) -> None:
        self.source_entry.configure(foreground=PLACEHOLDER_COLOR)
        self.source_var.set(PLACEHOLDER)

    def _placeholder_focus_in(self, _e=None) -> None:
        if self.source_var.get() == PLACEHOLDER:
            self.source_var.set("")
            self.source_entry.configure(foreground=ENTRY_COLOR)

    def _placeholder_focus_out(self, _e=None) -> None:
        if not self.source_var.get().strip():
            self._apply_placeholder()

    # Helpers
    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v"), ("All files", "*.*")]
        )
        if path:
            self.source_entry.configure(foreground=ENTRY_COLOR)
            self.source_var.set(path)

    def _paste_url(self) -> None:
        try:
            text = self.clipboard_get().strip()
        except Exception:
            text = ""
        if not text:
            messagebox.showinfo("AESTHETIC", "Clipboard is empty.")
            return
        if not _is_url(text):
            messagebox.showwarning("AESTHETIC", "Clipboard doesn't look like a URL.")
            return
        self.source_entry.configure(foreground=ENTRY_COLOR)
        self.source_var.set(text)

    def _reload_config(self) -> None:
        self.config_dict = prepare_config(load_config_file(self.config_path))
        ensure_output_tree(self.config_dict)
        self._log("Config reloaded.")

    def _open_output(self) -> None:
        out_dir = Path((self.config_dict.get("output") or {}).get("folder", "aesthetic/outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(out_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(out_dir)], check=False)
            else:
                subprocess.run(["xdg-open", str(out_dir)], check=False)
        except Exception as exc:
            messagebox.showerror("AESTHETIC", f"Could not open output folder\n{exc}")

    def _open_logs(self) -> None:
        out_dir = Path((self.config_dict.get("output") or {}).get("folder", "aesthetic/outputs"))
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(log_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(log_dir)], check=False)
            else:
                subprocess.run(["xdg-open", str(log_dir)], check=False)
        except Exception as exc:
            messagebox.showerror("AESTHETIC", f"Could not open logs folder\n{exc}")

    def _copy_error(self) -> None:
        if not self.error_text:
            return
        self.clipboard_clear()
        self.clipboard_append(self.error_text)
        self.update()
        messagebox.showinfo("Copied", "Error details copied to clipboard")

    def _log(self, msg: str) -> None:
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    # Run orchestration
    def _run_async(self) -> None:
        src = self.source_var.get().strip()
        if not src or src == PLACEHOLDER:
            messagebox.showwarning("AESTHETIC", "Please choose a file or paste a URL.")
            return

        # Apply the "Export hero video" toggle into the config copy we send to the pipeline
        cfg = dict(self.config_dict)  # shallow copy ok (pipeline normalizes)
        cfg.setdefault("output", {}).setdefault("hero_video", {})["enabled"] = bool(self.export_var.get())

        self.progress.set(0.0)
        self.status_var.set("starting…")
        self.error_text = ""
        try:
            self.copy_btn.state(["disabled"])
        except Exception:
            pass

        self._log(f"Running pipeline on: {src} | hero_video={'on' if self.export_var.get() else 'off'}")

        def worker() -> None:
            try:
                result = pipeline.run_pipeline(
                    src,
                    config=cfg,
                    progress_cb=lambda frac, msg: self.after(0, lambda: self._update_progress(frac, msg)),
                )
                self.after(0, lambda: self._on_success(result))
            except Exception as exc:
                LOG.exception("pipeline failed: %s", exc)
                self.after(0, lambda e=exc: self._on_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def _update_progress(self, frac: float, message: str) -> None:
        self.progress.set(max(0.0, min(1.0, float(frac))))
        self.status_var.set(message)
        self._log(message)

    def _on_success(self, result: Dict[str, Any]) -> None:
        self.progress.set(1.0)
        self.status_var.set("done")
        self.last_manifest = result.get("manifest")
        stage_log = result.get("stage_log", {})
        for key, data in stage_log.items():
            if key == "ingest":
                summary = f"fps={data.get('fps', 0):.2f} frames={data.get('frame_count')}"
            elif key == "scene_detection":
                summary = f"{data.get('count', 0)} scenes (method={data.get('method')})"
            elif key == "sampling":
                summary = f"candidates={data.get('total_candidates', 0)}"
            elif key == "features":
                summary = f"clip={data.get('clip_status', {}).get('computed', 0)}"
            elif key == "selection":
                summary = f"selected={data.get('selected', 0)}"
            elif key == "dedup":
                summary = f"{data.get('before', 0)}→{data.get('after', 0)}"
            else:
                summary = str(data)
            self.stages.update_stage(key, summary)
        hero = result.get("hero_video", {})
        if hero.get("enabled", False):
            status = "ok" if hero.get("ffmpeg_ok", False) else hero.get("error", "failed")
            self.stages.update_stage("export", f"hero={status}")
        else:
            self.stages.update_stage("export", "disabled")
        try:
            self.copy_btn.state(["!disabled"])
        except Exception:
            pass

    def _on_error(self, exc: Exception) -> None:
        self.status_var.set("error")
        self.error_text = f"{type(exc).__name__}: {exc}"
        self._log(self.error_text)
        try:
            self.copy_btn.state(["!disabled"])
        except Exception:
            pass
        messagebox.showerror("AESTHETIC", self.error_text)

    def _rerun_with_seed(self) -> None:
        seed = simpledialog.askinteger("Rerun", "Seed:", minvalue=0, maxvalue=1_000_000)
        if seed is None:
            return
        self.config_dict.setdefault("sampling", {})["random_seed"] = int(seed)
        self._log(f"Seed set to {seed}")
        self._run_async()

def run_app(config_path: Optional[Path] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    app = MainWindow(config_path=config_path)
    app.mainloop()

__all__ = ["run_app", "MainWindow"]
