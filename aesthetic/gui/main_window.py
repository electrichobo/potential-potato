# -*- coding: utf-8 -*-
from __future__ import annotations
# AESTHETIC — modular GUI (kept under aesthetic/gui/)
# - Loads PROJECT_ROOT/config.yaml (no env var lookups)
# - Background thread for pipeline
# - Logs → outputs/logs/run.log
# - Copy Error, Open Output, post-run summary

import logging, logging.handlers, threading, traceback, datetime, glob, os, sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml

from aesthetic.core.pipeline import run_pipeline

# ----- paths -----
GUI_DIR = Path(__file__).resolve().parent           # .../aesthetic/gui
PKG_DIR = GUI_DIR.parent                             # .../aesthetic
PROJECT_ROOT = PKG_DIR.parent                        # .../AestheticApp
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"   # ALWAYS this file

LOG = logging.getLogger("aesthetic.gui")

# ---------- logging ----------
def _ensure_logging(cfg: Dict[str, Any]) -> None:
    out_dir = Path(((cfg.get("output") or {}).get("folder")) or "aesthetic/outputs")
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

# ---------- config ----------
def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Always load config from PROJECT_ROOT/config.yaml unless an explicit path is given.
    Environment variables are intentionally ignored.
    """
    p = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not p.exists():
        LOG.warning("config: %s not found, using defaults", p)
        return {}
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Normalize output folder to absolute path so every module agrees
    out_rel = ((cfg.get("output") or {}).get("folder")) or "aesthetic/outputs"
    out_abs = Path(out_rel)
    if not out_abs.is_absolute():
        out_abs = (PROJECT_ROOT / out_rel).resolve()
    cfg.setdefault("output", {})["folder"] = str(out_abs)
    return cfg

# ---------- GUI ----------
class App(tk.Tk):
    """AESTHETIC 2.0 — modular GUI shell."""
    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__()
        self.title("AESTHETIC 2.0")
        self.geometry(self._load_geometry())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.cfg: Dict[str, Any] = cfg or load_config()
        _ensure_logging(self.cfg)
        logging.getLogger("aesthetic.gui").info("config loaded from %s", DEFAULT_CONFIG_PATH)

        # UI state
        self.source_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="idle")
        self.progress = tk.DoubleVar(value=0.0)
        self.last_error_text = ""

        # Layout
        root = ttk.Frame(self, padding=10); root.pack(fill="both", expand=True)
        ttk.Label(root, text="Source (file path or URL):").grid(row=0, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.source_var, width=72).grid(row=1, column=0, columnspan=3, sticky="ew")
        ttk.Button(root, text="Browse", command=self._browse).grid(row=1, column=3, sticky="e", padx=(6,0))

        btns = ttk.Frame(root); btns.grid(row=2, column=0, columnspan=4, sticky="we", pady=(10, 0))
        ttk.Button(btns, text="Run", command=self._run_async).pack(side="left")
        ttk.Button(btns, text="Reload config", command=self._reload_config).pack(side="left", padx=(8,0))
        ttk.Button(btns, text="Open Output", command=self._open_out).pack(side="left", padx=(8,0))
        self.copy_btn = ttk.Button(btns, text="Copy Error", command=self._copy_error, state="disabled")
        self.copy_btn.pack(side="left", padx=(8,0))

        ttk.Label(root, textvariable=self.status_var).grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Progressbar(root, variable=self.progress, maximum=1.0).grid(row=4, column=0, columnspan=4, sticky="ew", pady=(4, 8))

        self.log = tk.Text(root, height=18, width=100)
        self.log.grid(row=5, column=0, columnspan=4, sticky="nsew")
        root.rowconfigure(5, weight=1)
        root.columnconfigure(0, weight=1)

    # ----- UI actions -----
    def _browse(self) -> None:
        path = filedialog.askopenfilename(title="Choose video")
        if path: self.source_var.set(path)

    def _reload_config(self) -> None:
        self.cfg = load_config()
        _ensure_logging(self.cfg)
        self._logline("Config reloaded.")
        self.status_var.set("config reloaded")

    def _open_out(self) -> None:
        out_dir = Path((self.cfg.get("output") or {}).get("folder", "aesthetic/outputs"))
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(str(out_dir))   # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{out_dir}"')
            else:
                os.system(f'xdg-open "{out_dir}"')
        except Exception as e:
            messagebox.showerror("AESTHETIC", f"Could not open folder:\n{out_dir}\n\n{e}")

    def _copy_error(self) -> None:
        if not self.last_error_text:
            return
        self.clipboard_clear()
        self.clipboard_append(self.last_error_text)
        self.update()
        messagebox.showinfo("Copied", "Error details copied to clipboard.")

    # ----- run (background) -----
    def _run_async(self) -> None:
        src = self.source_var.get().strip()
        if not src:
            messagebox.showerror("AESTHETIC", "Please provide a file path or URL")
            return
        self.status_var.set("running…")
        self.progress.set(0.0)
        self.last_error_text = ""
        try:
            self.copy_btn.state(['disabled'])
        except Exception:
            pass
        self._logline("Running pipeline…")

        out_dir = Path((self.cfg.get("output") or {}).get("folder", "aesthetic/outputs"))
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        run_started = datetime.datetime.now()

        def on_progress(frac: float, msg: str) -> None:
            self.after(0, lambda: self._update_progress(frac, msg))

        def worker():
            try:
                run_pipeline(
                    source=src,
                    config=self.cfg,
                    progress_cb=on_progress,
                    cancel_fn=lambda: False,
                )
                saved = self._list_new_outputs(out_dir, since=run_started)
                self.after(0, lambda: self._on_success(saved))
            except Exception as e:
                logging.getLogger("aesthetic.gui").exception("pipeline failed: %s", e)
                tb = "".join(traceback.format_exc())
                msg = f"Processing failed: {type(e).__name__}: {e}\n\n{tb}"
                self.after(0, lambda: self._on_error(msg))

        threading.Thread(target=worker, daemon=True, name="pipeline").start()

    # ----- helpers -----
    def _update_progress(self, frac: float, msg: str) -> None:
        try:
            self.progress.set(max(0.0, min(1.0, float(frac))))
        except Exception:
            self.progress.set(0.0)
        self.status_var.set(str(msg))
        self._logline(str(msg))
        self.update_idletasks()

    def _on_success(self, saved: List[Dict[str, Any]]) -> None:
        self.progress.set(1.0)
        self.status_var.set("done")
        lines = [f"Saved {len(saved)} frames:"]
        for w in saved:
            name = os.path.basename(w.get("path", ""))
            score = float(w.get("score", 0.0))
            lines.append(f" - {name:<26}  score={score:.3f}")
        self._logblock("\n".join(lines))
        try:
            self.copy_btn.state(['!disabled'])
        except Exception:
            pass

    def _on_error(self, full_msg: str) -> None:
        self.status_var.set("error")
        self._logblock(full_msg)
        self.last_error_text = full_msg
        try:
            self.copy_btn.state(['!disabled'])
        except Exception:
            pass
        messagebox.showerror("AESTHETIC", full_msg.splitlines()[0])

    def _list_new_outputs(self, out_dir: Path, since: datetime.datetime) -> List[Dict[str, Any]]:
        jpgs = sorted(glob.glob(str(out_dir / "*.jpg")))
        out: List[Dict[str, Any]] = []
        for p in jpgs:
            try:
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(p))
                if mtime < since:
                    continue
                side = os.path.splitext(p)[0] + ".txt"
                score = 0.0
                if os.path.exists(side):
                    try:
                        import json as _json
                        with open(side, "r", encoding="utf-8") as f:
                            meta = _json.load(f)
                        score = float(meta.get("score_blend", meta.get("tech", {}).get("score", 0.0)))
                    except Exception:
                        pass
                out.append({"path": p, "score": score})
            except Exception:
                continue
        out.sort(key=lambda d: (os.path.getmtime(d["path"]), d["score"]), reverse=True)
        return out

    # ----- window state -----
    def _load_geometry(self) -> str:
        gfile = PROJECT_ROOT / ".aesthetic_ui_geom.txt"
        try:
            if gfile.exists():
                return gfile.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return "1000x640"

    def _on_close(self) -> None:
        gfile = PROJECT_ROOT / ".aesthetic_ui_geom.txt"
        try:
            gfile.write_text(self.geometry(), encoding="utf-8")
        except Exception:
            pass
        self.destroy()

def main() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    _ensure_logging(cfg)
    App(cfg).mainloop()

if __name__ == "__main__":
    main()
