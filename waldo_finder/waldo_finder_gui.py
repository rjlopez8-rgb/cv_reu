"""
Waldo Finder GUI
================
Tkinter interface for waldo_finder.py — browse a folder of Where's Waldo
images and see ranked striped-pattern candidates.

IMPORTANT: keep this file in the SAME FOLDER as waldo_finder.py
(it imports the detection logic from there).

Features:
  * Open a folder of images, cycle with Prev / Next / Random
  * Adjustable number of candidates
  * Tuning sliders for scan quality (red saturation / white brightness),
    since every book scan has slightly different colors
  * Click a candidate in the list (or on the image) to select it and see
    a zoomed-in preview

Requires:  pip install opencv-python numpy pillow
Run with:  python waldo_finder_gui.py
"""

import os
import random
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import waldo_finder as wf
except ImportError:
    raise SystemExit(
        "Could not import waldo_finder.py — make sure waldo_finder_gui.py "
        "and waldo_finder.py are in the same folder."
    )

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
CANVAS_W, CANVAS_H = 760, 500
ZOOM_W, ZOOM_H = 280, 220

GOLD = "#f0b429"
GRAY_BOX = (160, 160, 160)


class WaldoFinderApp:
    def __init__(self, root):
        self.root = root
        root.title("Waldo Finder")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)

        self.folder = None
        self.image_files = []
        self.index = -1
        self.original = None
        self.candidates = []
        self.selected = 0
        self.scale = 1.0
        self.offset = (0, 0)
        self.tk_photo = None
        self.tk_zoom = None

        self._build_layout()

    # ── layout ──────────────────────────────────────────────────────────────────
    def _build_layout(self):
        btn = dict(bg="#89b4fa", fg="#1e1e2e", relief="flat",
                   font=("Segoe UI", 10, "bold"), padx=12, pady=5,
                   activebackground="#b4befe", cursor="hand2")

        bar = tk.Frame(self.root, bg="#1e1e2e")
        bar.pack(fill="x", padx=12, pady=(12, 6))
        tk.Button(bar, text="Open Folder", command=self.pick_folder, **btn).pack(side="left")
        tk.Button(bar, text="◀ Prev", command=self.prev_image, **btn).pack(side="left", padx=(8, 0))
        tk.Button(bar, text="Next ▶", command=self.next_image, **btn).pack(side="left", padx=(8, 0))
        tk.Button(bar, text="🎲 Random", command=self.random_image, **btn).pack(side="left", padx=(8, 0))

        tk.Label(bar, text="Candidates:", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10)).pack(side="left", padx=(18, 4))
        self.top_n = tk.Spinbox(bar, from_=1, to=20, width=3, font=("Segoe UI", 10),
                                command=self.analyze_current)
        self.top_n.delete(0, "end"); self.top_n.insert(0, "5")
        self.top_n.pack(side="left")

        self.file_label = tk.Label(bar, text="No folder selected", bg="#1e1e2e",
                                   fg="#a6adc8", font=("Segoe UI", 9))
        self.file_label.pack(side="right")

        # Tuning sliders for scan/print quality
        tune = tk.Frame(self.root, bg="#1e1e2e")
        tune.pack(fill="x", padx=12)
        slider_kw = dict(orient="horizontal", length=180, bg="#1e1e2e", fg="#cdd6f4",
                         troughcolor="#313244", highlightthickness=0,
                         font=("Segoe UI", 8))
        tk.Label(tune, text="Red saturation min", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9)).pack(side="left")
        self.red_s_min = tk.Scale(tune, from_=40, to=200, **slider_kw)
        self.red_s_min.set(120)
        self.red_s_min.pack(side="left", padx=(4, 16))
        tk.Label(tune, text="White brightness min", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9)).pack(side="left")
        self.white_v_min = tk.Scale(tune, from_=120, to=230, **slider_kw)
        self.white_v_min.set(170)
        self.white_v_min.pack(side="left", padx=(4, 16))
        tk.Button(tune, text="Re-analyze", command=self.analyze_current, **btn).pack(side="left")

        # Main area
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=12, pady=(6, 6))

        self.canvas = tk.Canvas(main, width=CANVAS_W, height=CANVAS_H,
                                bg="#11111b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left")
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        panel = tk.Frame(main, bg="#181825", width=300)
        panel.pack(side="left", fill="y", padx=(12, 0))
        panel.pack_propagate(False)

        tk.Label(panel, text="Candidates (click to inspect)", bg="#181825",
                 fg="#cdd6f4", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        self.list_frame = tk.Frame(panel, bg="#181825")
        self.list_frame.pack(fill="x", padx=12)

        tk.Label(panel, text="Zoom preview", bg="#181825", fg="#cdd6f4",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(14, 4))
        self.zoom_canvas = tk.Canvas(panel, width=ZOOM_W, height=ZOOM_H,
                                     bg="#11111b", highlightthickness=0)
        self.zoom_canvas.pack(padx=12)

        self.status = tk.Label(self.root, text="Open a folder of Waldo images to begin",
                               bg="#181825", fg="#f9e2af", font=("Segoe UI", 10),
                               anchor="w", padx=10, pady=6)
        self.status.pack(fill="x", padx=12, pady=(0, 12))

    # ── folder / navigation ─────────────────────────────────────────────────────
    def pick_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder of Waldo images")
        if not folder:
            return
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(IMAGE_EXTENSIONS))
        if not files:
            self.status.config(text="No images found in that folder!")
            return
        self.folder, self.image_files, self.index = folder, files, 0
        self.load_current()

    def next_image(self):
        if self.image_files:
            self.index = (self.index + 1) % len(self.image_files)
            self.load_current()

    def prev_image(self):
        if self.image_files:
            self.index = (self.index - 1) % len(self.image_files)
            self.load_current()

    def random_image(self):
        if self.image_files:
            self.index = random.randrange(len(self.image_files))
            self.load_current()

    def load_current(self):
        path = os.path.join(self.folder, self.image_files[self.index])
        img = cv2.imread(path)
        if img is None:
            self.status.config(text=f"Couldn't load {self.image_files[self.index]}")
            return
        self.original = img
        self.file_label.config(
            text=f"{self.image_files[self.index]}  ({self.index + 1}/{len(self.image_files)})")
        self.analyze_current()

    # ── detection ───────────────────────────────────────────────────────────────
    def analyze_current(self):
        if self.original is None:
            return
        self.status.config(text="Analyzing…")
        self.root.update_idletasks()

        # Push slider values into the waldo_finder module before running it
        s_min = int(self.red_s_min.get())
        v_min = int(self.white_v_min.get())
        wf.RED_RANGES = [
            (np.array([0,   s_min, 90]), np.array([8,   255, 255])),
            (np.array([170, s_min, 90]), np.array([180, 255, 255])),
        ]
        wf.WHITE_RANGE = (np.array([0, 0, v_min]), np.array([180, 60, 255]))

        try:
            n = max(1, min(20, int(self.top_n.get())))
        except ValueError:
            n = 5

        self.candidates = wf.find_candidates(self.original, top_n=n)
        self.selected = 0
        if self.candidates:
            self.status.config(
                text=f"{len(self.candidates)} candidate(s) — best score "
                     f"{self.candidates[0]['score']:.1f}, purity {self.candidates[0]['purity']:.2f}")
        else:
            self.status.config(text="No striped candidates found — try lowering the "
                                    "sliders for this scan's colors")
        self.refresh_display()
        self.refresh_list()
        self.refresh_zoom()

    # ── display ────────────────────────────────────────────────────────────────
    def _annotated(self):
        out = self.original.copy()
        for rank, c in enumerate(self.candidates):
            x0, y0, x1, y1 = c["box"]
            if rank == self.selected:
                color, thick = (41, 180, 240), 4      # GOLD-ish in BGR
            else:
                color, thick = GRAY_BOX, 2
            cv2.rectangle(out, (x0, y0), (x1, y1), color, thick)
            label = f"#{rank + 1}"
            cv2.rectangle(out, (x0, y0 - 22), (x0 + 34, y0), color, -1)
            cv2.putText(out, label, (x0 + 4, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def refresh_display(self):
        if self.original is None:
            return
        img = self._annotated()
        h, w = img.shape[:2]
        self.scale = min(CANVAS_W / w, CANVAS_H / h, 1.0)
        dw, dh = int(w * self.scale), int(h * self.scale)
        self.offset = ((CANVAS_W - dw) // 2, (CANVAS_H - dh) // 2)
        rgb = cv2.cvtColor(cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        self.tk_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(*self.offset, anchor="nw", image=self.tk_photo)

    def refresh_list(self):
        for wgt in self.list_frame.winfo_children():
            wgt.destroy()
        for rank, c in enumerate(self.candidates):
            sel = rank == self.selected
            row = tk.Frame(self.list_frame, bg=GOLD if sel else "#313244", cursor="hand2")
            row.pack(fill="x", pady=2)
            txt = (f"#{rank+1}  score {c['score']:.1f}   purity {c['purity']:.2f}   "
                   f"rows {c['stripe_rows']}")
            lbl = tk.Label(row, text=txt, bg=row["bg"],
                           fg="#1e1e2e" if sel else "#cdd6f4",
                           font=("Segoe UI", 9, "bold" if sel else "normal"),
                           anchor="w", padx=8, pady=4)
            lbl.pack(fill="x")
            for wgt in (row, lbl):
                wgt.bind("<Button-1>", lambda e, r=rank: self.select_candidate(r))

    def refresh_zoom(self):
        self.zoom_canvas.delete("all")
        if not self.candidates:
            return
        x0, y0, x1, y1 = self.candidates[self.selected]["box"]
        # pad the crop slightly for context
        pad = 14
        h, w = self.original.shape[:2]
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        crop = self.original[y0:y1, x0:x1]
        if crop.size == 0:
            return
        ch, cw = crop.shape[:2]
        zscale = min(ZOOM_W / cw, ZOOM_H / ch)
        crop = cv2.resize(crop, (int(cw * zscale), int(ch * zscale)),
                          interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        self.tk_zoom = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.zoom_canvas.create_image(ZOOM_W // 2, ZOOM_H // 2, image=self.tk_zoom)

    # ── interaction ────────────────────────────────────────────────────────────
    def select_candidate(self, rank):
        self.selected = rank
        self.refresh_display()
        self.refresh_list()
        self.refresh_zoom()

    def on_canvas_click(self, event):
        if self.original is None or not self.candidates:
            return
        x = (event.x - self.offset[0]) / self.scale
        y = (event.y - self.offset[1]) / self.scale
        # pick the smallest candidate box containing the click
        hits = [(i, c) for i, c in enumerate(self.candidates)
                if c["box"][0] <= x <= c["box"][2] and c["box"][1] <= y <= c["box"][3]]
        if hits:
            hits.sort(key=lambda ic: (ic[1]["box"][2] - ic[1]["box"][0]) *
                                     (ic[1]["box"][3] - ic[1]["box"][1]))
            self.select_candidate(hits[0][0])


if __name__ == "__main__":
    root = tk.Tk()
    app = WaldoFinderApp(root)
    root.mainloop()
