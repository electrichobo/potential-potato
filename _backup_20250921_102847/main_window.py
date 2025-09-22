# lines 1–36
from __future__ import annotations
# AESTHETIC 2.0 — minimal GUI shell (Phase A/B ready)

import os, sys, json, threading, subprocess, traceback, datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
# end 1–36

# lines 37–79
# ---------- crash-log + global hooks ----------
def _write_exception_log(base_dir: str, where: str, exc: BaseException) -> str:
    """Write a detailed crash log and return full path."""
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"crash_{where}_{ts}.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"[{ts}] {where}\n\n")
            f.write("Exception: " + repr(exc) + "\n\n")
            f.write("Traceback:\n")
            f.write("".join(traceback.format_exc()))
        return path
    except Exception:
        return ""

def _install_global_crash_hooks(log_dir_supplier):
    """Log ANY uncaught exception (main or background thread)."""
    def _log_and_print(where, exc_type, exc, tb):
        try:
            out_dir = log_dir_supplier()
            _write_exception_log(out_dir, where, exc)
        finally:
            traceback.print_exception(exc_type, exc, tb)

    def _sys_hook(exc_type, exc, tb): _log_and_print("sys", exc_type, exc, tb)
    def _thread_hook(args: threading.ExceptHookArgs):
        _log_and_print(f"thread:{args.thread.name}", args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _sys_hook
    try:
        threading.excepthook = _thread_hook  # Py 3.8+
    except Exception:
        pass
# end 37–79


# lines 80–265
class App(tk.Tk):
    """Tiny Tk app wrapping the pipeline with a friendly face."""
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self.title("AESTHETIC 2.0")

        # remember window placement
        self._state_path = os.path.join(os.path.dirname(__file__), "window_state.json")
        self._restore_window_geometry(default="900x600+120+80")

        # inputs row
        self.columnconfigure(1, weight=1)
        ttk.Label(self, text="Video file or URL").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.video_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.video_var).grid(row=0, column=1, sticky="we", padx=(0, 4))
        ttk.Button(self, text="Browse", command=self.browse_video).grid(row=0, column=2, padx=(0, 8))

        ttk.Label(self, text="Output folder").grid(row=1, column=0, padx=8, sticky="w")
        self.out_var = tk.StringVar(value=self.cfg.get("output", {}).get("folder", "aesthetic/outputs"))
        ttk.Entry(self, textvariable=self.out_var).grid(row=1, column=1, sticky="we", padx=(0, 4))
        ttk.Button(self, text="Browse", command=self.browse_out).grid(row=1, column=2, padx=(0, 8))

        ttk.Label(self, text="Diversity: k-medoids overcluster + facility location").grid(
            row=2, column=0, columnspan=3, padx=8, pady=(0, 6), sticky="w"
        )

        # progress + status
        self.pbar = ttk.Progressbar(self, mode='determinate', maximum=100)
        self.pbar.grid(row=3, column=0, columnspan=3, sticky='we', padx=8)
        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var).grid(row=4, column=0, columnspan=3, sticky="w", padx=8)

        # log box
        self.log = tk.Text(self, height=18, width=100)
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=8, pady=8)
        self.rowconfigure(5, weight=1)

        # buttons row
        btns = ttk.Frame(self); btns.grid(row=6, column=0, columnspan=3, sticky="e", padx=8, pady=(0, 10))
        self.copy_btn = ttk.Button(btns, text="Copy Error", command=self.copy_error, state="disabled")
        self.copy_btn.pack(side="right", padx=6)
        ttk.Button(btns, text="Open Output", command=self.open_out).pack(side="right", padx=6)
        ttk.Button(btns, text="Run", command=self.run_pipeline).pack(side="right", padx=6)

        self.last_error_text = ""
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # install crash hooks early; resolve current output dir for logs
        def _log_dir_supplier():
            return self._resolve_out_dir(self.out_var.get())
        _install_global_crash_hooks(_log_dir_supplier)

    # ----- window state -----
    def _restore_window_geometry(self, default: str = "900x600+100+100"):
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            x, y = int(st.get("x", 100)), int(st.get("y", 100))
            w, h = int(st.get("w", 900)), int(st.get("h", 600))
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            self.geometry(default)

    def _save_window_geometry(self):
        try:
            self.update_idletasks()
            geo = self.geometry()          # "WxH+X+Y"
            wh, xy = geo.split("+", 1)
            w, h = (int(v) for v in wh.split("x", 1))
            x, y = (int(v) for v in xy.split("+", 1))
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump({"x": x, "y": y, "w": w, "h": h}, f)
        except Exception:
            pass

    def _on_close(self):
        self._save_window_geometry()
        self.destroy()

    # ----- path helper -----
    def _resolve_out_dir(self, raw_path: str) -> str:
        p = (raw_path or "").strip()
        if not p:
            p = self.cfg.get('output', {}).get('folder', 'aesthetic/outputs')
        p = os.path.expanduser(os.path.expandvars(p))
        if not os.path.isabs(p):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            p = os.path.normpath(os.path.join(project_root, p))
        return p

    # ----- UI actions -----
    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4;*.mov;*.mkv;*.avi;*.m4v;*.webm;*.flv;*.wmv"), ("All files", "*.*")]
        )
        if path:
            self.video_var.set(path)

    def browse_out(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.out_var.set(path)

    def open_out(self):
        path = self._resolve_out_dir(self.out_var.get())
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Open Output", f"Could not create folder:\n{path}\n\n{e}")
            return
        try:
            if os.name == "nt": os.startfile(path)
            elif sys.platform == "darwin": subprocess.run(["open", path], check=False)
            else: subprocess.run(["xdg-open", path], check=False)
            self.log.insert('end', f"Opened: {path}\n"); self.log.see('end')
        except Exception as e:
            messagebox.showerror("Open Output", f"Could not open folder:\n{path}\n\n{e}")

    def copy_error(self):
        if not self.last_error_text: return
        self.clipboard_clear(); self.clipboard_append(self.last_error_text); self.update()
        messagebox.showinfo("Copied", "Error details copied to clipboard.")

    def _show_error(self, msg: str) -> None:
        self.last_error_text = msg
        try: self.copy_btn.state(['!disabled'])
        except Exception: pass
        self.log.insert('end', f"\nERROR: {msg}\n"); self.log.see('end')
        self.status_var.set("Error"); self.pbar['value'] = 0

    def _on_done(self, saved: list[dict]) -> None:
        """Called after background worker completes successfully."""
        try:
            self.pbar['value'] = 100
            self.status_var.set("Done.")
            out_dir = (self.cfg.get('output') or {}).get('folder', 'aesthetic/outputs')
            lines = [f"Saved {len(saved)} frames to {out_dir}"]
            for w in saved:
                name = os.path.basename(w.get("path", ""))
                score = float(w.get("score", 0.0))
                lines.append(f" - {name:<26}  score={score:.3f}")
            self.log.insert('end', "\n" + "\n".join(lines) + "\n"); self.log.see('end')
            self.copy_btn.state(['!disabled'])
        except Exception as e:
            self._show_error(f"Finalization failed: {type(e).__name__}: {e}")

    # ----- run pipeline (background) -----
    def run_pipeline(self):
        src = self.video_var.get().strip()
        if not src:
            self.log.insert('end', 'Pick a local video file or URL.\n'); self.log.see('end'); return

        # reset UI
        self.copy_btn.state(['disabled']); self.last_error_text = ""
        self.pbar['value'] = 0; self.status_var.set("Starting…")
        self.log.insert('end', 'Running pipeline…\n'); self.log.see('end')

        # ensure absolute output paths + create logs dir
        out_dir_abs = self._resolve_out_dir(self.out_var.get())
        self.cfg.setdefault('output', {})['folder'] = out_dir_abs
        os.makedirs(out_dir_abs, exist_ok=True)
        os.makedirs(os.path.join(out_dir_abs, "logs"), exist_ok=True)

        def on_progress(pct: float, msg: str):
            self.after(0, lambda: (self.pbar.configure(value=int(round(pct*100))),
                                   self.status_var.set(msg)))

        def worker():
            try:
                from ..core.pipeline import process_local_video
                saved = process_local_video(self.cfg, src, on_progress=on_progress)
                self.after(0, lambda: self._on_done(saved))
            except Exception as e:
                out_dir = (self.cfg.get("output") or {}).get("folder", "aesthetic/outputs")
                log_path = _write_exception_log(out_dir, "pipeline", e)
                msg = f"Processing failed: {type(e).__name__}: {e}"
                if log_path: msg += f"\nDetails saved to:\n{log_path}"
                self.after(0, lambda m=msg: self._show_error(m))

        threading.Thread(target=worker, daemon=True, name="pipeline").start()
# end 80–265
