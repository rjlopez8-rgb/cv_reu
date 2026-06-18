"""
Real-Time Shape Detection GUI
==============================
Uses OpenCV for computer vision and Tkinter for the GUI.
Detects: Circle, Triangle, Rectangle, Square, Pentagon, Hexagon, and more.

Run: python shape_detector_gui.py
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import json
from collections import defaultdict, deque
import os


# ─────────────────────────────────────────────────────────────────────────────
#  Shape Detection Engine
# ─────────────────────────────────────────────────────────────────────────────

SHAPE_COLORS = {
    "Circle":    (0,   200, 255),
    "Triangle":  (0,   255, 100),
    "Square":    (255, 150, 0  ),
    "Rectangle": (180, 0,   255),
    "Pentagon":  (0,   100, 255),
    "Hexagon":   (255, 0,   150),
    "Octagon":   (0,   255, 220),
    "Polygon":   (200, 200, 0  ),
    "Parallelogram": (255, 80, 80),
}

def classify_shape(contour):
    """Return (shape_name, vertices_count, confidence)."""
    peri     = cv2.arcLength(contour, True)
    approx   = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)
    area     = cv2.contourArea(contour)

    if area < 500:
        return None, vertices, 0.0

    # Circularity metric (perfect circle = 1.0)
    circularity = (4 * np.pi * area) / (peri ** 2) if peri > 0 else 0

    shape_map = {
        3: "Triangle",
        4: None,      # handled separately below
        5: "Pentagon",
        6: "Hexagon",
        8: "Octagon",
    }

    # Circle: raised threshold to 0.88 AND require many vertices so that
    # blocky hexagons (~0.91 circularity but only 6 approx vertices) are
    # not misclassified as circles.
    if circularity > 0.88 and vertices > 6:
        return "Circle", vertices, round(circularity, 2)

    if vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h) if h else 1

        # Check interior angles -- parallelograms have non-90-degree corners.
        pts = approx.reshape(4, 2).astype(np.float32)
        angles = []
        for i in range(4):
            p0 = pts[(i - 1) % 4]
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            v1 = p0 - p1
            v2 = p2 - p1
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

        max_angle_dev = max(abs(a - 90) for a in angles)

        if max_angle_dev > 20:          # clearly not a rectangle/square
            return "Parallelogram", vertices, round(1 - max_angle_dev / 90, 2)

        name = "Square" if 0.9 <= aspect <= 1.1 else "Rectangle"
        confidence = 1 - abs(1 - aspect) if name == "Square" else 0.9
        return name, vertices, round(min(confidence, 1.0), 2)

    name = shape_map.get(vertices, "Polygon")
    return name, vertices, 0.85


def detect_shapes(frame, settings):
    """
    Detect shapes in a frame.
    Returns annotated frame and list of detected shape dicts.
    """
    blur_k    = settings.get("blur", 5)
    canny_lo  = settings.get("canny_lo", 50)
    canny_hi  = settings.get("canny_hi", 150)
    min_area  = settings.get("min_area", 500)
    show_info = settings.get("show_info", True)

    output   = frame.copy()
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (blur_k | 1, blur_k | 1), 0)
    edges    = cv2.Canny(blurred, canny_lo, canny_hi)
    dilated  = cv2.dilate(edges, None, iterations=1)

    img_area = frame.shape[0] * frame.shape[1]
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    seen_rects = set()   # deduplicate near-identical contours from RETR_LIST
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Skip contours that are clearly the image border (>70% of frame)
        if area > 0.70 * img_area:
            continue

        name, verts, conf = classify_shape(cnt)
        if not name:
            continue

        # Deduplicate: RETR_LIST returns inner/outer edge of thick strokes.
        # Keep only the largest contour per (shape, centroid-grid-cell).
        M2 = cv2.moments(cnt)
        if M2["m00"] != 0:
            kcx = int(M2["m10"] / M2["m00"]) // 30
            kcy = int(M2["m01"] / M2["m00"]) // 30
        else:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            kcx, kcy = (bx + bw // 2) // 30, (by + bh // 2) // 30
        key = (name, kcx, kcy)
        if key in seen_rects:
            continue
        seen_rects.add(key)

        color = SHAPE_COLORS.get(name, (255, 255, 255))
        cv2.drawContours(output, [cnt], -1, color, 2)

        M  = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

        if show_info:
            label = f"{name} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            # Place label above centroid; clamp so it stays within frame
            frame_w = output.shape[1]
            tx = max(4, min(cx - tw // 2, frame_w - tw - 8))
            ty = cy - 10
            cv2.rectangle(output, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4),
                          (0, 0, 0), -1)
            cv2.putText(output, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        detected.append({"name": name, "vertices": verts,
                         "confidence": conf, "cx": cx, "cy": cy,
                         "area": cv2.contourArea(cnt)})

    return output, edges, detected


# ─────────────────────────────────────────────────────────────────────────────
#  Main GUI Application
# ─────────────────────────────────────────────────────────────────────────────

class ShapeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔷 Real-Time Shape Detector")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        # State
        self.cap           = None
        self.running       = False
        self.paused        = False
        self.source_type   = "webcam"   # webcam | image | video
        self.source_path   = None
        self.frame_lock    = threading.Lock()
        self.current_frame = None
        self.detection_log = deque(maxlen=200)
        self.shape_counts  = defaultdict(int)
        self.fps_history   = deque(maxlen=30)
        self.last_time     = time.time()
        self.show_edges    = tk.BooleanVar(value=False)
        self.freeze        = False

        # Settings (mutable via sliders)
        self.settings = {
            "blur":     5,
            "canny_lo": 50,
            "canny_hi": 150,
            "min_area": 500,
            "show_info": True,
        }

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Top toolbar
        tb = tk.Frame(self.root, bg="#16213e", pady=6)
        tb.pack(fill="x")

        tk.Label(tb, text="🔷 Shape Detector", font=("Arial", 16, "bold"),
                 fg="#00d4ff", bg="#16213e").pack(side="left", padx=12)

        btn_cfg = dict(bg="#0f3460", fg="white", font=("Arial", 10, "bold"),
                       relief="flat", padx=10, pady=4, cursor="hand2",
                       activebackground="#1a5276", activeforeground="white")

        self.btn_webcam = tk.Button(tb, text="📷 Webcam", command=self._start_webcam, **btn_cfg)
        self.btn_webcam.pack(side="left", padx=4)

        tk.Button(tb, text="🖼 Image", command=self._open_image, **btn_cfg).pack(side="left", padx=4)
        tk.Button(tb, text="🎬 Video", command=self._open_video, **btn_cfg).pack(side="left", padx=4)

        self.btn_pause = tk.Button(tb, text="⏸ Pause", command=self._toggle_pause,
                                   state="disabled", **btn_cfg)
        self.btn_pause.pack(side="left", padx=4)

        tk.Button(tb, text="💾 Save", command=self._save_frame, **btn_cfg).pack(side="left", padx=4)
        tk.Button(tb, text="📊 Export Log", command=self._export_log, **btn_cfg).pack(side="left", padx=4)
        tk.Button(tb, text="🔄 Reset", command=self._reset_counts, **btn_cfg).pack(side="left", padx=4)

        tk.Checkbutton(tb, text="Show Edges", variable=self.show_edges,
                       bg="#16213e", fg="#aaa", selectcolor="#0f3460",
                       activebackground="#16213e", font=("Arial", 10)).pack(side="left", padx=8)

        # Status bar (fps etc.)
        self.status_var = tk.StringVar(value="Ready — choose a source above")
        tk.Label(tb, textvariable=self.status_var, font=("Arial", 9),
                 fg="#aaa", bg="#16213e").pack(side="right", padx=12)

        # Main content area
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill="both", expand=True)

        # Left: video panel
        left = tk.Frame(main, bg="#1a1a2e")
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.canvas = tk.Label(left, bg="#0d0d1a",
                               text="No source selected\nClick 📷 Webcam, 🖼 Image or 🎬 Video",
                               fg="#555", font=("Arial", 13), width=80, height=25)
        self.canvas.pack(fill="both", expand=True)

        # Right panel
        right = tk.Frame(main, bg="#1a1a2e", width=300)
        right.pack(side="right", fill="y", padx=4, pady=8)
        right.pack_propagate(False)

        # Shape stats
        stats_frame = tk.LabelFrame(right, text="  📊 Detection Stats  ",
                                    bg="#16213e", fg="#00d4ff",
                                    font=("Arial", 11, "bold"), padx=8, pady=6)
        stats_frame.pack(fill="x", padx=4, pady=4)

        self.stat_labels = {}
        for shape in SHAPE_COLORS:
            row = tk.Frame(stats_frame, bg="#16213e")
            row.pack(fill="x", pady=1)
            color_hex = "#{:02x}{:02x}{:02x}".format(*SHAPE_COLORS[shape][::-1])
            tk.Label(row, text="■", fg=color_hex, bg="#16213e",
                     font=("Arial", 14)).pack(side="left")
            tk.Label(row, text=shape, fg="white", bg="#16213e",
                     font=("Arial", 10), width=10, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="0", fg="#00d4ff", bg="#16213e",
                           font=("Arial", 10, "bold"), width=6, anchor="e")
            lbl.pack(side="right")
            self.stat_labels[shape] = lbl

        # Live count
        cnt_frame = tk.Frame(stats_frame, bg="#16213e")
        cnt_frame.pack(fill="x", pady=(6, 2))
        tk.Label(cnt_frame, text="Total this frame:", fg="#aaa", bg="#16213e",
                 font=("Arial", 9)).pack(side="left")
        self.lbl_total = tk.Label(cnt_frame, text="0", fg="#ff6b6b", bg="#16213e",
                                  font=("Arial", 11, "bold"))
        self.lbl_total.pack(side="right")

        # Sliders
        ctrl_frame = tk.LabelFrame(right, text="  ⚙️ Detection Settings  ",
                                   bg="#16213e", fg="#00d4ff",
                                   font=("Arial", 11, "bold"), padx=8, pady=6)
        ctrl_frame.pack(fill="x", padx=4, pady=4)

        self._make_slider(ctrl_frame, "Blur Kernel", "blur", 1, 21, 5, 2)
        self._make_slider(ctrl_frame, "Canny Low",   "canny_lo", 0, 255, 50, 1)
        self._make_slider(ctrl_frame, "Canny High",  "canny_hi", 0, 255, 150, 1)
        self._make_slider(ctrl_frame, "Min Area",    "min_area", 100, 5000, 500, 100)

        # Detection log
        log_frame = tk.LabelFrame(right, text="  📋 Detection Log  ",
                                  bg="#16213e", fg="#00d4ff",
                                  font=("Arial", 11, "bold"), padx=6, pady=6)
        log_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.log_text = tk.Text(log_frame, bg="#0d0d1a", fg="#88ff88",
                                font=("Courier", 8), wrap="word",
                                state="disabled", height=12)
        sb = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

    def _make_slider(self, parent, label, key, lo, hi, default, step):
        row = tk.Frame(parent, bg="#16213e")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, fg="#ccc", bg="#16213e",
                 font=("Arial", 9), width=12, anchor="w").pack(side="left")
        val_lbl = tk.Label(row, text=str(default), fg="#00d4ff", bg="#16213e",
                           font=("Arial", 9, "bold"), width=5)
        val_lbl.pack(side="right")
        def on_change(v, k=key, lbl=val_lbl):
            iv = int(float(v))
            self.settings[k] = iv
            lbl.config(text=str(iv))
        s = tk.Scale(row, from_=lo, to=hi, orient="horizontal",
                     bg="#16213e", fg="#ccc", troughcolor="#0f3460",
                     highlightthickness=0, sliderlength=14,
                     resolution=step, command=on_change)
        s.set(default)
        s.pack(side="left", fill="x", expand=True)

    # ── Source Controls ───────────────────────────────────────────────────────

    def _start_webcam(self):
        self._stop_capture()
        self.source_type = "webcam"
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.\nMake sure a camera is connected.")
            return
        self.running = True
        self.paused  = False
        self.btn_pause.config(state="normal")
        self.status_var.set("🟢 Webcam active")
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Cannot read image file.")
            return
        self._stop_capture()
        self.source_type  = "image"
        self.current_frame = frame
        self._process_and_show(frame)
        self.status_var.set(f"🖼 Image: {os.path.basename(path)}")
        self.btn_pause.config(state="disabled")

    def _open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")])
        if not path:
            return
        self._stop_capture()
        self.source_type = "video"
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return
        self.running = True
        self.paused  = False
        self.btn_pause.config(state="normal")
        self.status_var.set(f"🎬 Video: {os.path.basename(path)}")
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _stop_capture(self):
        self.running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(text="▶ Resume" if self.paused else "⏸ Pause")

    # ── Capture & Processing Loop ─────────────────────────────────────────────

    def _capture_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            if not self.cap or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            with self.frame_lock:
                self.current_frame = frame.copy()

            self._process_and_show(frame)

            # FPS
            now = time.time()
            elapsed = now - self.last_time
            self.last_time = now
            if elapsed > 0:
                self.fps_history.append(1 / elapsed)
            fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            self.root.after(0, self.status_var.set,
                            f"{'🟢 Webcam' if self.source_type=='webcam' else '🎬 Video'}"
                            f" | FPS: {fps:.1f}")

            time.sleep(0.01)

    def _process_and_show(self, frame):
        annotated, edges, detected = detect_shapes(frame, self.settings)

        # Choose display
        if self.show_edges.get():
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            display = annotated

        # Resize to fit canvas (max 760x480)
        h, w = display.shape[:2]
        max_w, max_h = 760, 480
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        display = cv2.resize(display, (nw, nh))

        img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img  = ImageTk.PhotoImage(pil_img)

        self.root.after(0, self._update_canvas, tk_img, detected)

    def _update_canvas(self, tk_img, detected):
        self.canvas.config(image=tk_img, text="")
        self.canvas.image = tk_img   # prevent GC

        # Update stats
        frame_counts = defaultdict(int)
        for d in detected:
            frame_counts[d["name"]] += 1
            self.shape_counts[d["name"]] += 1

        for shape, lbl in self.stat_labels.items():
            lbl.config(text=str(self.shape_counts[shape]))

        self.lbl_total.config(text=str(len(detected)))

        # Log
        if detected:
            ts = time.strftime("%H:%M:%S")
            lines = [f"[{ts}] " + ", ".join(
                f"{d['name']}({d['confidence']:.0%})" for d in detected)]
            self.detection_log.extend(lines)
            self.log_text.config(state="normal")
            for line in lines:
                self.log_text.insert("end", line + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _save_frame(self):
        with self.frame_lock:
            frame = self.current_frame
        if frame is None:
            messagebox.showinfo("Save", "No frame to save.")
            return
        annotated, _, _ = detect_shapes(frame, self.settings)
        path = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, annotated)
            messagebox.showinfo("Saved", f"Frame saved to:\n{path}")

    def _export_log(self):
        if not self.detection_log:
            messagebox.showinfo("Export", "No detections to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("JSON", "*.json")])
        if path:
            with open(path, "w") as f:
                f.write("\n".join(self.detection_log))
            messagebox.showinfo("Exported", f"Log saved to:\n{path}")

    def _reset_counts(self):
        for k in self.shape_counts:
            self.shape_counts[k] = 0
        for lbl in self.stat_labels.values():
            lbl.config(text="0")
        self.detection_log.clear()
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _on_close(self):
        self._stop_capture()
        self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pillow required for ImageTk
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow",
                               "--break-system-packages", "-q"])
        from PIL import Image, ImageTk

    root = tk.Tk()
    app  = ShapeDetectorApp(root)
    root.mainloop()
