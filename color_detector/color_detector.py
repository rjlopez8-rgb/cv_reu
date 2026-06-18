import cv2
import numpy as np
import sys

# ── Color definitions ──────────────────────────────────────────────────────────
# Each entry: "name" -> ([(lo_HSV, hi_HSV), ...], BGR_display_color)
#
# OpenCV HSV ranges:  H 0-180  S 0-255  V 0-255
# Colors are ordered around the hue wheel (0→180) so the table is easy to
# extend.  Colors that lack strong saturation (black, white, gray) are
# detected purely on Value/Saturation thresholds instead of Hue.
#
# PRIORITY NOTE: achromatic colors (black, white, gray) are tested LAST so
# that vivid-but-dark pixels aren't swallowed by the "black" bucket.
COLORS = {
    # Ranges calibrated by sampling actual HSV values from test patches.
    # OpenCV HSV: H 0-180, S 0-255, V 0-255
    # Achromatic colors (Black/White/Gray) are placed LAST so vivid-but-dark
    # pixels don't get swallowed by the "Black" range before reaching their
    # correct colour bucket.

    # ── Red  H≈0, S high, V mid-high (hue wraps at 0/180) ────────────────────
    "Red": (
        [(np.array([0,   150, 80]),  np.array([8,   255, 255])),
         (np.array([172, 150, 80]),  np.array([180, 255, 255]))],
        (0, 0, 220),
    ),
    # ── Coral  desaturated/pastel red ─────────────────────────────────────────
    "Coral": (
        [(np.array([0,   50,  160]), np.array([8,   149, 255])),
         (np.array([172, 50,  160]), np.array([180, 149, 255]))],
        (80, 100, 255),
    ),
    # ── Orange  H≈14, S=255, V=255 ────────────────────────────────────────────
    "Orange": (
        [(np.array([9,   230, 120]), np.array([18,  255, 255]))],
        (0, 120, 255),
    ),
    # ── Brown  same hue as orange but S≈200, V≈140 ────────────────────────────
    "Brown": (
        [(np.array([8,   60,  30]),  np.array([20,  229, 170]))],
        (30, 80, 140),
    ),
    # ── Yellow  H≈30, S=255, V=220 ────────────────────────────────────────────
    "Yellow": (
        [(np.array([22,  150, 120]), np.array([35,  255, 255]))],
        (0, 220, 220),
    ),
    # ── Lime  H≈51, S=255, V=255 ──────────────────────────────────────────────
    "Lime": (
        [(np.array([38,  100, 80]),  np.array([57,  255, 255]))],
        (0, 255, 80),
    ),
    # ── Green  H≈60, S=255, V=180 ─────────────────────────────────────────────
    "Green": (
        [(np.array([57,  80,  40]),  np.array([87,  255, 255]))],
        (0, 180, 0),
    ),
    # ── Teal  H≈90, S=255, V=128  (dark cyan-green) ───────────────────────────
    "Teal": (
        [(np.array([87,  120, 30]),  np.array([93,  255, 190]))],
        (128, 128, 0),
    ),
    # ── Cyan  H≈94, S=255, V=255  (bright, full saturation) ──────────────────
    "Cyan": (
        [(np.array([87,  150, 180]), np.array([100, 255, 255]))],
        (255, 220, 0),
    ),
    # ── Sky Blue  H≈100-112, lower S ──────────────────────────────────────────
    "Sky Blue": (
        [(np.array([100, 40,  130]), np.array([112, 149, 255]))],
        (235, 206, 135),
    ),
    # ── Blue  H≈113, S=255, V=220 (bright) ───────────────────────────────────
    "Blue": (
        [(np.array([105, 150, 111]), np.array([120, 255, 255]))],
        (220, 50, 0),
    ),
    # ── Navy  H≈114, S=255, V=100 (dark) ─────────────────────────────────────
    "Navy": (
        [(np.array([105, 100, 20]),  np.array([120, 255, 110]))],
        (80, 10, 0),
    ),
    # ── Indigo  H≈120-138, mid brightness ─────────────────────────────────────
    "Indigo": (
        [(np.array([120, 60,  40]),  np.array([138, 255, 179]))],
        (180, 30, 75),
    ),
    # ── Magenta  H≈141, S=255, V=255 ──────────────────────────────────────────
    "Magenta": (
        [(np.array([138, 120, 100]), np.array([146, 255, 255]))],
        (255, 0, 200),
    ),
    # ── Purple  H≈150, S=255, V=200 ───────────────────────────────────────────
    "Purple": (
        [(np.array([146, 60,  50]),  np.array([160, 255, 255]))],
        (180, 0, 180),
    ),
    # ── Pink  H≈164, S≈235, V=255 ─────────────────────────────────────────────
    "Pink": (
        [(np.array([158, 60,  120]), np.array([172, 255, 255]))],
        (147, 20, 255),
    ),
    # ── Achromatic — last so vivid darks aren't misclassified as Black ────────
    "Black": (
        [(np.array([0,   0,   0]),   np.array([180, 255, 45]))],
        (30, 30, 30),
    ),
    # White: S very low, V very high (pure white = S=0, V=255)
    "White": (
        [(np.array([0,   0,   200]), np.array([180, 30,  255]))],
        (220, 220, 220),
    ),
    # Gray: S low, V mid-range
    "Gray": (
        [(np.array([0,   0,   46]),  np.array([180, 60,  199]))],
        (130, 130, 130),
    ),
}


def detect_colors(image_path: str) -> None:
    # ── Load image ─────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not load '{image_path}'")
        sys.exit(1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blur slightly to reduce noise before thresholding
    hsv_blur = cv2.GaussianBlur(hsv, (7, 7), 0)

    result = img.copy()
    total_pixels = img.shape[0] * img.shape[1]

    print(f"\nAnalyzing: {image_path}  ({img.shape[1]}x{img.shape[0]} px)\n")
    print(f"{'Color':<14}  {'Coverage':>9}  {'Contours':>9}")
    print("-" * 40)

    for color_name, (ranges, bgr) in COLORS.items():
        # Build a combined mask for all HSV ranges of this color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_blur, lo, hi))

        # Clean up the mask with morphological ops
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)   # remove specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill gaps

        # Find contours (individual regions of that color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 500]  # ignore tiny blobs

        # Coverage percentage
        coverage = (cv2.countNonZero(mask) / total_pixels) * 100

        if contours:
            print(f"{color_name:<14}  {coverage:>8.1f}%  {len(contours):>9}")

            # Draw bounding boxes and labels on result image
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x + w, y + h), bgr, 2)
                label = f"{color_name} ({coverage:.1f}%)"
                # Small white background behind text for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(result, (x, y - th - 6), (x + tw + 4, y), bgr, -1)
                cv2.putText(result, label, (x + 2, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            print(f"{color_name:<14}  {'not found':>10}")

    print()

    # ── Save annotated output ──────────────────────────────────────────────────
    out_path = image_path.replace(".png", "_detected.png").replace(".jpg", "_detected.jpg")
    cv2.imwrite(out_path, result)
    print(f"Annotated image saved → {out_path}\n")

    # ── Show result (press any key to close) ──────────────────────────────────
    cv2.imshow("Color Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  python color_detector.py <image_path>")
        print("Example: python color_detector.py test_image1_shapes.png")
        sys.exit(1)

    detect_colors(sys.argv[1])
