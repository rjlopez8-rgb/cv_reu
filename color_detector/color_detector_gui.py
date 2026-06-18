"""
Color Detector GUI
==================
A tkinter interface for the OpenCV color detector.

Features:
  * Pick a folder of images — cycle through them with Next / Previous / Random
  * Top colors panel — every detected color ranked by % coverage
  * Click anywhere on the image to identify the color at that exact spot
  * Toggle bounding-box overlay on/off
  * Works with .png, .jpg, .jpeg, .bmp, .webp

Requires:  pip install opencv-python numpy pillow
Run with:  python color_detector_gui.py
"""

import os
import random
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Color definitions ──────────────────────────────────────────────────────────
# COMPLETE PARTITION of HSV space — every possible (H,S,V) pixel maps to
# exactly ONE color, so "Unknown" is impossible.
#
# Design (OpenCV HSV: H 0-180, S 0-255, V 0-255):
#   Achromatic:  V<=45 -> Black | S<=35 & V>=200 -> White | S<=35 -> Gray
#   Chromatic (S>=36, V>=46): 10 hue bands from the standard color wheel,
#   each split into three tiers:
#       dark   (low V)              e.g. Maroon, Brown, Navy, Plum
#       vivid  (high S, high V)     e.g. Red, Orange, Blue, Magenta
#       soft   (low S, high V)      e.g. Salmon, Tan, Sky Blue, Lavender
#
# Hue bands (degrees/2): Red 0-5 & 172-180 | Orange 6-19 | Yellow 20-32 |
#   Lime 33-44 | Green 45-87 | Cyan 88-100 | Blue 101-125 | Violet 126-144 |
#   Magenta 145-159 | Pink 160-171
COLORS = {
    # ── Red band  H 0-5 & 172-180 ─────────────────────────────────────────────
    "Red": (
        [(np.array([0,   150, 121]), np.array([5,   255, 255])),
         (np.array([172, 150, 121]), np.array([180, 255, 255]))],
        (0, 0, 220),
    ),
    "Salmon": (
        [(np.array([0,   36,  121]), np.array([5,   149, 255])),
         (np.array([172, 36,  121]), np.array([180, 149, 255]))],
        (122, 160, 250),
    ),
    "Maroon": (
        [(np.array([0,   36,  46]),  np.array([5,   255, 120])),
         (np.array([172, 36,  46]),  np.array([180, 255, 120]))],
        (32, 32, 128),
    ),
    # ── Orange band  H 6-19  (Brown gets a taller V range: browns are dark
    #    oranges and real-world browns reach V~160) ────────────────────────────
    "Orange": (
        [(np.array([6,   150, 170]), np.array([19,  255, 255]))],
        (0, 140, 255),
    ),
    "Tan": (
        [(np.array([6,   36,  170]), np.array([19,  149, 255]))],
        (140, 180, 210),
    ),
    "Brown": (
        [(np.array([6,   36,  46]),  np.array([19,  255, 169]))],
        (42, 75, 130),
    ),
    # ── Yellow band  H 20-32  (dark yellow = olive, reaches V~150) ────────────
    "Yellow": (
        [(np.array([20,  150, 150]), np.array([32,  255, 255]))],
        (0, 220, 220),
    ),
    "Beige": (
        [(np.array([20,  36,  150]), np.array([32,  149, 255]))],
        (179, 222, 245),
    ),
    "Olive": (
        [(np.array([20,  36,  46]),  np.array([32,  255, 149])),
         (np.array([33,  36,  46]),  np.array([44,  255, 120]))],  # + dark lime
        (0, 128, 128),
    ),
    # ── Lime band  H 33-44 (true chartreuse only — green starts at 45) ────────
    "Lime": (
        [(np.array([33,  150, 121]), np.array([44,  255, 255]))],
        (0, 255, 127),
    ),
    # ── Green band  H 45-87 ───────────────────────────────────────────────────
    "Green": (
        [(np.array([45,  150, 121]), np.array([87,  255, 255]))],
        (0, 170, 0),
    ),
    "Sage": (   # soft/pale greens, spans lime+green bands
        [(np.array([33,  36,  121]), np.array([87,  149, 255]))],
        (154, 190, 178),
    ),
    "Dark Green": (
        [(np.array([45,  36,  46]),  np.array([87,  255, 120]))],
        (0, 100, 0),
    ),
    # ── Cyan band  H 88-100  (dark cyan = teal, reaches V~140) ────────────────
    "Cyan": (
        [(np.array([88,  150, 140]), np.array([100, 255, 255]))],
        (255, 220, 0),
    ),
    "Seafoam": (
        [(np.array([88,  36,  140]), np.array([100, 149, 255]))],
        (191, 226, 159),
    ),
    "Teal": (
        [(np.array([88,  36,  46]),  np.array([100, 255, 139]))],
        (128, 128, 0),
    ),
    # ── Blue band  H 101-125 ──────────────────────────────────────────────────
    "Blue": (
        [(np.array([101, 150, 121]), np.array([125, 255, 255]))],
        (220, 50, 0),
    ),
    "Sky Blue": (
        [(np.array([101, 36,  121]), np.array([125, 149, 255]))],
        (235, 206, 135),
    ),
    "Navy": (
        [(np.array([101, 36,  46]),  np.array([125, 255, 120]))],
        (128, 30, 0),
    ),
    # ── Violet band  H 126-144 ────────────────────────────────────────────────
    "Purple": (
        [(np.array([126, 150, 121]), np.array([144, 255, 255]))],
        (255, 0, 160),
    ),
    "Lavender": (
        [(np.array([126, 36,  121]), np.array([144, 149, 255]))],
        (250, 180, 230),
    ),
    "Indigo": (
        [(np.array([126, 36,  46]),  np.array([144, 255, 120]))],
        (130, 0, 75),
    ),
    # ── Magenta band  H 145-159 ───────────────────────────────────────────────
    "Magenta": (
        [(np.array([145, 150, 121]), np.array([159, 255, 255]))],
        (255, 0, 255),
    ),
    "Orchid": (
        [(np.array([145, 36,  121]), np.array([159, 149, 255]))],
        (214, 112, 218),
    ),
    "Plum": (
        [(np.array([145, 36,  46]),  np.array([159, 255, 120]))],
        (133, 69, 142),
    ),
    # ── Pink band  H 160-171 ──────────────────────────────────────────────────
    "Pink": (
        [(np.array([160, 150, 121]), np.array([171, 255, 255]))],
        (180, 105, 255),
    ),
    "Light Pink": (
        [(np.array([160, 36,  121]), np.array([171, 149, 255]))],
        (193, 182, 255),
    ),
    "Berry": (
        [(np.array([160, 36,  46]),  np.array([171, 255, 120]))],
        (87, 38, 135),
    ),
    # ── Achromatic ────────────────────────────────────────────────────────────
    "Black": (
        [(np.array([0,   0,   0]),   np.array([180, 255, 45]))],
        (30, 30, 30),
    ),
    "White": (
        [(np.array([0,   0,   200]), np.array([180, 35,  255]))],
        (245, 245, 245),
    ),
    "Gray": (
        [(np.array([0,   0,   46]),  np.array([180, 35,  199]))],
        (130, 130, 130),
    ),
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
CANVAS_W, CANVAS_H = 720, 480


# ── Core detection helpers ──────────────────────────────────────────────────────
def build_mask(hsv_img, ranges):
    """Combine all HSV ranges of one color into a single binary mask."""
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_img, lo, hi))
    return mask


def classify_pixel(hsv_pixel):
    """Return the color name a single HSV pixel falls into, or 'Unknown'."""
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
    for name, (ranges, _) in COLORS.items():
        for lo, hi in ranges:
            if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
                return name
    return "Unknown"


def analyze_image(bgr_img):
    """
    Run full color detection.
    Returns (results, annotated_image) where results is a list of
    (name, coverage_percent, contour_count, bgr_display_color) sorted by coverage.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, (7, 7), 0)
    annotated = bgr_img.copy()
    total = bgr_img.shape[0] * bgr_img.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    results = []
    for name, (ranges, bgr) in COLORS.items():
        mask = build_mask(hsv_blur, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 500]
        coverage = cv2.countNonZero(mask) / total * 100

        if contours and coverage >= 0.5:
            results.append((name, coverage, len(contours), bgr))
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), bgr, 2)
                label = f"{name} ({coverage:.1f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x, y - th - 6), (x + tw + 4, y), bgr, -1)
                cv2.putText(annotated, label, (x + 2, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    results.sort(key=lambda r: r[1], reverse=True)
    return results, annotated


def bgr_to_hex(bgr):
    """Convert an OpenCV (B,G,R) tuple to a tkinter hex color string."""
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


# ── GUI application ─────────────────────────────────────────────────────────────
class ColorDetectorApp:
    def __init__(self, root):
        self.root = root
        root.title("Color Detector")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)

        self.folder = None
        self.image_files = []
        self.index = -1
        self.original = None      # current image (BGR, full size)
        self.annotated = None     # annotated version (BGR, full size)
        self.hsv_full = None      # HSV of full-size image, for click lookup
        self.scale = 1.0          # display scale factor
        self.offset = (0, 0)      # image offset inside canvas (for letterboxing)
        self.show_boxes = tk.BooleanVar(value=True)
        self.tk_photo = None      # keep a reference or tkinter garbage-collects it

        self._build_layout()

    # ── layout ──────────────────────────────────────────────────────────────────
    def _build_layout(self):
        # Top toolbar
        bar = tk.Frame(self.root, bg="#1e1e2e")
        bar.pack(fill="x", padx=12, pady=(12, 6))

        style_btn = dict(bg="#89b4fa", fg="#1e1e2e", relief="flat",
                         font=("Segoe UI", 10, "bold"), padx=14, pady=6,
                         activebackground="#b4befe", cursor="hand2")

        tk.Button(bar, text="Open Folder", command=self.pick_folder, **style_btn).pack(side="left")
        tk.Button(bar, text="◀ Prev", command=self.prev_image, **style_btn).pack(side="left", padx=(8, 0))
        tk.Button(bar, text="Next ▶", command=self.next_image, **style_btn).pack(side="left", padx=(8, 0))
        tk.Button(bar, text="🎲 Random", command=self.random_image, **style_btn).pack(side="left", padx=(8, 0))

        tk.Checkbutton(bar, text="Show detection boxes", variable=self.show_boxes,
                       command=self.refresh_display, bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244", activebackground="#1e1e2e",
                       activeforeground="#cdd6f4", font=("Segoe UI", 10)).pack(side="left", padx=(16, 0))

        self.file_label = tk.Label(bar, text="No folder selected", bg="#1e1e2e",
                                   fg="#a6adc8", font=("Segoe UI", 9))
        self.file_label.pack(side="right")

        # Main area: canvas on left, results panel on right
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        self.canvas = tk.Canvas(main, width=CANVAS_W, height=CANVAS_H,
                                bg="#11111b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left")
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_hover)

        panel = tk.Frame(main, bg="#181825", width=260)
        panel.pack(side="left", fill="y", padx=(12, 0))
        panel.pack_propagate(False)

        tk.Label(panel, text="Top Colors", bg="#181825", fg="#cdd6f4",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=12, pady=(12, 4))

        # Scrollable results list
        self.results_frame = tk.Frame(panel, bg="#181825")
        self.results_frame.pack(fill="both", expand=True, padx=12)

        # Bottom status bar: hover + click readouts
        status = tk.Frame(self.root, bg="#181825")
        status.pack(fill="x", padx=12, pady=(0, 12))

        self.hover_swatch = tk.Frame(status, width=18, height=18, bg="#181825")
        self.hover_swatch.pack(side="left", padx=(10, 6), pady=8)
        self.hover_label = tk.Label(status, text="Hover over the image…",
                                    bg="#181825", fg="#a6adc8", font=("Segoe UI", 10))
        self.hover_label.pack(side="left", pady=8)

        self.click_label = tk.Label(status, text="Click a spot to identify its color",
                                    bg="#181825", fg="#f9e2af", font=("Segoe UI", 10, "bold"))
        self.click_label.pack(side="right", padx=10, pady=8)

    # ── folder / image navigation ───────────────────────────────────────────────
    def pick_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder of images")
        if not folder:
            return
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(IMAGE_EXTENSIONS))
        if not files:
            self.file_label.config(text="No images found in that folder!")
            return
        self.folder = folder
        self.image_files = files
        self.index = 0
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
            self.file_label.config(text=f"Couldn't load {self.image_files[self.index]}")
            return

        self.original = img
        self.hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.results, self.annotated = analyze_image(img)

        self.file_label.config(
            text=f"{self.image_files[self.index]}  ({self.index + 1}/{len(self.image_files)})")
        self.click_label.config(text="Click a spot to identify its color")
        self.refresh_display()
        self.refresh_results_panel()

    # ── display ────────────────────────────────────────────────────────────────
    def refresh_display(self):
        if self.original is None:
            return
        img = self.annotated if self.show_boxes.get() else self.original

        h, w = img.shape[:2]
        self.scale = min(CANVAS_W / w, CANVAS_H / h, 1.0)
        disp_w, disp_h = int(w * self.scale), int(h * self.scale)
        self.offset = ((CANVAS_W - disp_w) // 2, (CANVAS_H - disp_h) // 2)

        resized = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas.delete("all")
        self.canvas.create_image(self.offset[0], self.offset[1],
                                 anchor="nw", image=self.tk_photo)

    def refresh_results_panel(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if not self.results:
            tk.Label(self.results_frame, text="No colors detected",
                     bg="#181825", fg="#6c7086", font=("Segoe UI", 10)).pack(anchor="w")
            return

        for name, coverage, n_regions, bgr in self.results:
            row = tk.Frame(self.results_frame, bg="#181825")
            row.pack(fill="x", pady=3)

            sw = tk.Frame(row, width=22, height=22, bg=bgr_to_hex(bgr))
            sw.pack(side="left")
            sw.pack_propagate(False)

            txt = f"{name}  —  {coverage:.1f}%"
            if n_regions > 1:
                txt += f"  ({n_regions} regions)"
            tk.Label(row, text=txt, bg="#181825", fg="#cdd6f4",
                     font=("Segoe UI", 10)).pack(side="left", padx=8)

            # coverage bar
            bar_bg = tk.Frame(row, bg="#313244", height=6, width=80)
            bar_bg.pack(side="right", padx=(0, 4))
            bar_bg.pack_propagate(False)
            fill_w = max(2, int(80 * min(coverage, 100) / 100))
            tk.Frame(bar_bg, bg=bgr_to_hex(bgr), height=6, width=fill_w).pack(side="left")

    # ── canvas interaction ─────────────────────────────────────────────────────
    def _canvas_to_image_coords(self, event):
        """Convert a canvas click position to original-image pixel coords, or None."""
        if self.original is None:
            return None
        x = (event.x - self.offset[0]) / self.scale
        y = (event.y - self.offset[1]) / self.scale
        h, w = self.original.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return int(x), int(y)
        return None

    def on_canvas_click(self, event):
        coords = self._canvas_to_image_coords(event)
        if coords is None:
            return
        x, y = coords

        # Sample a 5x5 region around the click for stability
        h, w = self.original.shape[:2]
        x0, x1 = max(0, x - 2), min(w, x + 3)
        y0, y1 = max(0, y - 2), min(h, y + 3)
        region_hsv = self.hsv_full[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        region_bgr = self.original[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)

        name = classify_pixel(region_hsv)
        b, g, r = (int(v) for v in region_bgr)
        self.click_label.config(
            text=f"({x}, {y})  →  {name}   RGB({r}, {g}, {b})")

        # Draw a small marker on the canvas at the click point
        self.refresh_display()
        cx, cy = event.x, event.y
        self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                                outline="#f9e2af", width=2)
        self.canvas.create_line(cx - 10, cy, cx + 10, cy, fill="#f9e2af")
        self.canvas.create_line(cx, cy - 10, cx, cy + 10, fill="#f9e2af")

    def on_canvas_hover(self, event):
        coords = self._canvas_to_image_coords(event)
        if coords is None:
            self.hover_label.config(text="Hover over the image…")
            self.hover_swatch.config(bg="#181825")
            return
        x, y = coords
        hsv_px = self.hsv_full[y, x]
        bgr_px = self.original[y, x]
        name = classify_pixel(hsv_px)
        self.hover_label.config(
            text=f"({x}, {y})  {name}   HSV({hsv_px[0]}, {hsv_px[1]}, {hsv_px[2]})")
        self.hover_swatch.config(bg=bgr_to_hex(tuple(int(v) for v in bgr_px)))


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorDetectorApp(root)
    root.mainloop()
