"""
Color Detector & Image Processing GUI
=====================================
A Tkinter application that:
  1. Loads an image with a button.
  2. Detects the color under the mouse cursor and matches it to the
     closest of 60 reference colors (full spectrum, white -> black),
     showing the color name, HEX code, and RGB values.
  3. Applies image-processing operations selected from a dropdown:
     Edge Detection, FFT Spectrum, Low-Pass Filter (FFT & Gaussian),
     Black & White, Grayscale, Inverted, Sepia, Sharpen, High-Pass.

Dependencies:  pip install pillow numpy
Run:           python color_detector_app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageFilter, ImageOps


# ----------------------------------------------------------------------
# 1. BUILD THE 60-COLOR REFERENCE PALETTE  (white -> full spectrum -> black)
# ----------------------------------------------------------------------
def build_palette():
    """
    60 colors total:
      - 12 hues x 4 variations (Light, Pure, Dark, Deep)  = 48 colors
      - 12 grayscale steps from White to Black             = 12 colors
    """
    import colorsys

    hue_names = [
        "Red", "Orange", "Yellow", "Chartreuse", "Green", "Spring Green",
        "Cyan", "Azure", "Blue", "Violet", "Magenta", "Rose",
    ]
    # (prefix, saturation, value)
    variations = [
        ("Light ", 0.45, 1.00),
        ("",       1.00, 1.00),   # pure hue
        ("Dark ",  1.00, 0.65),
        ("Deep ",  1.00, 0.35),
    ]

    palette = []  # list of (name, (r, g, b))
    for i, hname in enumerate(hue_names):
        h = i / 12.0
        for prefix, s, v in variations:
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            palette.append((prefix + hname,
                            (round(r * 255), round(g * 255), round(b * 255))))

    # 12 grayscale steps: 255 -> 0
    gray_names = ["White", "Gray 90%", "Gray 80%", "Gray 70%", "Gray 60%",
                  "Gray 50%", "Gray 40%", "Gray 30%", "Gray 20%", "Gray 10%",
                  "Near Black", "Black"]
    levels = np.linspace(255, 0, 12).round().astype(int)
    for name, lv in zip(gray_names, levels):
        palette.append((name, (int(lv), int(lv), int(lv))))

    return palette


PALETTE = build_palette()
PALETTE_ARRAY = np.array([rgb for _, rgb in PALETTE], dtype=np.int32)  # (60, 3)


def nearest_color(rgb):
    """Return (name, (r,g,b), hex) of the palette color closest to rgb
    using Euclidean distance in RGB space."""
    diff = PALETTE_ARRAY - np.array(rgb, dtype=np.int32)
    dist = np.einsum("ij,ij->i", diff, diff)      # squared distances
    idx = int(np.argmin(dist))
    name, prgb = PALETTE[idx]
    return name, prgb, "#{:02X}{:02X}{:02X}".format(*prgb)


# ----------------------------------------------------------------------
# 2. IMAGE PROCESSING OPERATIONS
# ----------------------------------------------------------------------
def to_gray_array(img):
    return np.asarray(img.convert("L"), dtype=np.float64)


def fft_spectrum(img):
    """Log-magnitude spectrum of the 2-D FFT (centered)."""
    gray = to_gray_array(img)
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(F))
    mag = (mag / mag.max() * 255).astype(np.uint8)
    return Image.fromarray(mag).convert("RGB")


def fft_low_pass(img, radius_ratio=0.10):
    """Keep only low frequencies inside a circular mask, then inverse FFT."""
    gray = to_gray_array(img)
    F = np.fft.fftshift(np.fft.fft2(gray))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = radius_ratio * min(h, w)
    mask = (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2
    out = np.fft.ifft2(np.fft.ifftshift(F * mask)).real
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")


def fft_high_pass(img, radius_ratio=0.05):
    """Remove low frequencies -> only edges/details remain."""
    gray = to_gray_array(img)
    F = np.fft.fftshift(np.fft.fft2(gray))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = radius_ratio * min(h, w)
    mask = (Y - cy) ** 2 + (X - cx) ** 2 > radius ** 2
    out = np.fft.ifft2(np.fft.ifftshift(F * mask)).real
    out = np.abs(out)
    out = (out / out.max() * 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")


def sobel_edges(img):
    """Edge detection with Sobel kernels implemented in NumPy."""
    gray = to_gray_array(img)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = Kx.T
    pad = np.pad(gray, 1, mode="edge")
    # Build sliding windows and convolve
    win = np.lib.stride_tricks.sliding_window_view(pad, (3, 3))
    gx = np.einsum("ijkl,kl->ij", win, Kx)
    gy = np.einsum("ijkl,kl->ij", win, Ky)
    mag = np.hypot(gx, gy)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    return Image.fromarray(mag).convert("RGB")


def black_and_white(img, threshold=128):
    gray = img.convert("L")
    return gray.point(lambda p: 255 if p >= threshold else 0).convert("RGB")


def sepia(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    M = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]])
    out = np.clip(arr @ M.T, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


FILTERS = {
    "Original":                lambda im: im.convert("RGB"),
    "Grayscale":               lambda im: im.convert("L").convert("RGB"),
    "Black & White":           black_and_white,
    "Inverted":                lambda im: ImageOps.invert(im.convert("RGB")),
    "Edge Detection (Sobel)":  sobel_edges,
    "Edge Detection (PIL)":    lambda im: im.convert("RGB").filter(ImageFilter.FIND_EDGES),
    "FFT Spectrum":            fft_spectrum,
    "Low Pass (FFT)":          fft_low_pass,
    "Low Pass (Gaussian Blur)":lambda im: im.convert("RGB").filter(ImageFilter.GaussianBlur(4)),
    "High Pass (FFT)":         fft_high_pass,
    "Sharpen":                 lambda im: im.convert("RGB").filter(ImageFilter.SHARPEN),
    "Sepia":                   sepia,
    "Emboss":                  lambda im: im.convert("RGB").filter(ImageFilter.EMBOSS),
}


# ----------------------------------------------------------------------
# 3. THE TKINTER APPLICATION
# ----------------------------------------------------------------------
class ColorDetectorApp:
    CANVAS_W, CANVAS_H = 760, 560

    def __init__(self, root):
        self.root = root
        root.title("Color Detector & Image Processing Studio")
        root.configure(bg="#1e1e2e")

        self.original = None        # full-resolution PIL image
        self.display_img = None     # processed + resized PIL image on canvas
        self.display_arr = None     # numpy array of display_img (pixel lookup)
        self.tk_img = None          # ImageTk reference (must keep alive)
        self.offset = (0, 0)        # image position inside the canvas

        self._build_ui()

    # ---------------- UI LAYOUT ----------------
    def _build_ui(self):
        # Left: canvas
        left = tk.Frame(self.root, bg="#1e1e2e")
        left.pack(side="left", padx=10, pady=10)

        self.canvas = tk.Canvas(left, width=self.CANVAS_W, height=self.CANVAS_H,
                                bg="#11111b", highlightthickness=1,
                                highlightbackground="#45475a", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", lambda e: self.status.config(
            text="Move the cursor over the image"))

        # Right: controls
        right = tk.Frame(self.root, bg="#1e1e2e")
        right.pack(side="right", fill="y", padx=(0, 12), pady=10)

        tk.Button(right, text="📂  Load Image", command=self.load_image,
                  font=("Segoe UI", 12, "bold"), bg="#89b4fa", fg="#11111b",
                  activebackground="#74c7ec", relief="flat", padx=10, pady=8
                  ).pack(fill="x", pady=(0, 14))

        tk.Label(right, text="Filter / Operation", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.filter_var = tk.StringVar(value="Original")
        combo = ttk.Combobox(right, textvariable=self.filter_var,
                             values=list(FILTERS.keys()), state="readonly",
                             font=("Segoe UI", 10), width=26)
        combo.pack(fill="x", pady=(4, 14))
        combo.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())

        # --- Color info panel ---
        panel = tk.LabelFrame(right, text=" Color Under Cursor ", bg="#1e1e2e",
                              fg="#cdd6f4", font=("Segoe UI", 10, "bold"),
                              labelanchor="n", bd=1, relief="groove")
        panel.pack(fill="x", pady=(0, 10))

        self.swatch_exact = tk.Canvas(panel, width=220, height=40,
                                      bg="#11111b", highlightthickness=0)
        self.swatch_exact.pack(pady=(8, 2))
        self.lbl_exact = tk.Label(panel, text="Exact:  —", bg="#1e1e2e",
                                  fg="#a6adc8", font=("Consolas", 10))
        self.lbl_exact.pack(anchor="w", padx=8)

        self.swatch_best = tk.Canvas(panel, width=220, height=40,
                                     bg="#11111b", highlightthickness=0)
        self.swatch_best.pack(pady=(10, 2))
        self.lbl_best = tk.Label(panel, text="Best match (of 60):  —",
                                 bg="#1e1e2e", fg="#f9e2af",
                                 font=("Consolas", 10, "bold"),
                                 wraplength=230, justify="left")
        self.lbl_best.pack(anchor="w", padx=8, pady=(0, 8))

        # --- 60-color palette preview ---
        pal = tk.LabelFrame(right, text=" 60-Color Reference Palette ",
                            bg="#1e1e2e", fg="#cdd6f4",
                            font=("Segoe UI", 10, "bold"), labelanchor="n",
                            bd=1, relief="groove")
        pal.pack(fill="x")
        grid = tk.Canvas(pal, width=240, height=120, bg="#1e1e2e",
                         highlightthickness=0)
        grid.pack(padx=6, pady=6)
        sw, sh = 24, 24
        for i, (_, rgb) in enumerate(PALETTE):
            r, c = divmod(i, 10)
            grid.create_rectangle(c * sw, r * sh, c * sw + sw - 2,
                                  r * sh + sh - 2,
                                  fill="#{:02X}{:02X}{:02X}".format(*rgb),
                                  outline="#45475a")

        self.status = tk.Label(right, text="Load an image to begin",
                               bg="#1e1e2e", fg="#6c7086",
                               font=("Segoe UI", 9), wraplength=240,
                               justify="left")
        self.status.pack(anchor="w", pady=(10, 0))

    # ---------------- ACTIONS ----------------
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self.original = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")
            return
        self.filter_var.set("Original")
        self.apply_filter()
        self.status.config(text=f"Loaded: {path.split('/')[-1]}  "
                                f"({self.original.width}x{self.original.height})")

    def apply_filter(self):
        if self.original is None:
            return
        name = self.filter_var.get()
        try:
            processed = FILTERS[name](self.original)
        except Exception as exc:
            messagebox.showerror("Filter error", str(exc))
            return
        self._show(processed)

    def _show(self, img):
        # Fit image inside the canvas while keeping aspect ratio
        img = img.copy()
        img.thumbnail((self.CANVAS_W, self.CANVAS_H), Image.LANCZOS)
        self.display_img = img
        self.display_arr = np.asarray(img, dtype=np.uint8)
        self.tk_img = ImageTk.PhotoImage(img)

        ox = (self.CANVAS_W - img.width) // 2
        oy = (self.CANVAS_H - img.height) // 2
        self.offset = (ox, oy)

        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.tk_img)

    # ---------------- MOUSE COLOR DETECTION ----------------
    def on_mouse_move(self, event):
        if self.display_arr is None:
            return
        ox, oy = self.offset
        x, y = event.x - ox, event.y - oy
        h, w = self.display_arr.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        r, g, b = (int(v) for v in self.display_arr[y, x][:3])
        exact_hex = f"#{r:02X}{g:02X}{b:02X}"
        name, prgb, best_hex = nearest_color((r, g, b))

        # Exact swatch
        self.swatch_exact.delete("all")
        self.swatch_exact.create_rectangle(0, 0, 220, 40, fill=exact_hex,
                                           outline="#45475a")
        self.lbl_exact.config(text=f"Exact:  {exact_hex}  RGB({r},{g},{b})")

        # Best-match swatch
        self.swatch_best.delete("all")
        self.swatch_best.create_rectangle(0, 0, 220, 40, fill=best_hex,
                                          outline="#45475a")
        self.lbl_best.config(text=f"Best match: {name}\n{best_hex}  "
                                  f"RGB{prgb}")

        self.status.config(text=f"Cursor at image pixel ({x}, {y})")


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorDetectorApp(root)
    root.mainloop()
