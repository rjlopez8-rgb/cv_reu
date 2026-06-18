"""
Waldo Finder v2 — multi-feature detector
=========================================
Finds Waldo by combining THREE independent feature detectors and fusing them.
This is far more robust than the v1 stripe-only approach on real book pages.

The three detectors:
  1. TORSO  — horizontally alternating red/white stripes (his jumper)
  2. HEAD   — his face signature: a pair of small round glasses lenses
              (white discs ringed by dark), a thin white hat-band with red
              above it, and pale skin below
  3. FUSION — when a HEAD sits directly above a TORSO, the combined score is
              boosted heavily. A head+torso stack is almost always Waldo.

Each detector is built only from the OpenCV techniques in the tutorials you
studied: HSV masks (cv2.inRange), pixel-shift pattern tests, connected
components, integral images for fast box statistics, and non-max suppression.

Usage:
    python waldo_finder.py <image_path> [num_candidates]
"""

import sys
import cv2
import numpy as np

# ── Tunable HSV ranges (loosened for real print/scan quality) ────────────────────
RED_RANGES = [
    (np.array([0,   90, 70]),  np.array([10,  255, 255])),
    (np.array([168, 90, 70]),  np.array([180, 255, 255])),
]
WHITE_RANGE = (np.array([0, 0, 150]), np.array([180, 80, 255]))
SKIN_RANGE  = (np.array([0, 25, 150]), np.array([22, 120, 255]))
DARK_RANGE  = (np.array([0, 0, 0]),   np.array([180, 255, 135]))
BLUE_RANGE  = (np.array([95, 80, 60]), np.array([130, 255, 255]))

# Torso (stripe) detector params
STRIPE_OFFSETS = [2, 3, 4, 5, 6, 8, 10]
MIN_CLUSTER_AREA = 40
DENSITY_KERNEL = 21
TOP_N_DEFAULT = 5


# ── Shared helpers ────────────────────────────────────────────────────────────
def build_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RED_RANGES:
        red = cv2.bitwise_or(red, cv2.inRange(hsv, lo, hi))
    white = cv2.inRange(hsv, *WHITE_RANGE)
    skin  = cv2.inRange(hsv, *SKIN_RANGE)
    dark  = cv2.inRange(hsv, *DARK_RANGE)
    blue  = cv2.inRange(hsv, *BLUE_RANGE)
    return red, white, skin, dark, blue


def _shift(mask, dx, dy):
    """Shift a mask by (dx, dy), zero-filling the exposed edge."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xd0, xd1 = max(0, -dx), min(w, w - dx)
    yd0, yd1 = max(0, -dy), min(h, h - dy)
    out[ys0:ys1, xs0:xs1] = mask[yd0:yd1, xd0:xd1]
    return out


def _integral(arr):
    ii = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(arr.astype(np.float64), 0), 1)
    return ii


def _boxsum(ii, x0, y0, x1, y1):
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(ii.shape[1] - 1, x1), min(ii.shape[0] - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def _boxfrac(ii, x0, y0, x1, y1):
    area = max(1, (min(ii.shape[1] - 1, x1) - max(0, x0)) *
                  (min(ii.shape[0] - 1, y1) - max(0, y0)))
    return _boxsum(ii, x0, y0, x1, y1) / area


# ── Detector 1: striped TORSO ────────────────────────────────────────────────
def stripe_transition_map(red, white):
    h, w = red.shape
    t = np.zeros((h, w), np.float32)
    for dy in STRIPE_OFFSETS:
        t[:-dy] += ((red[:-dy] > 0) & (white[dy:] > 0)).astype(np.float32)
        t[:-dy] += ((white[:-dy] > 0) & (red[dy:] > 0)).astype(np.float32)
    return t


def horizontal_transition_map(red, white):
    h, w = red.shape
    t = np.zeros((h, w), np.float32)
    for dx in STRIPE_OFFSETS:
        t[:, :-dx] += ((red[:, :-dx] > 0) & (white[:, dx:] > 0)).astype(np.float32)
        t[:, :-dx] += ((white[:, :-dx] > 0) & (red[:, dx:] > 0)).astype(np.float32)
    return t


def find_torsos(bgr, red, white, blue, dark):
    trans = stripe_transition_map(red, white)
    h_trans = horizontal_transition_map(red, white)
    density = cv2.boxFilter(trans, -1, (DENSITY_KERNEL, DENSITY_KERNEL))
    thresh = max(0.8, float(density.max()) * 0.18)
    stripy = (density >= thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    stripy = cv2.morphologyEx(stripy, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(stripy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CLUSTER_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ex = int(w * 0.4)
        x0, y0 = max(0, x - ex), max(0, y - int(h * 1.2))
        x1, y1 = min(bgr.shape[1], x + w + ex), min(bgr.shape[0], y + h + int(h * 1.5))

        region_trans = float(trans[y:y+h, x:x+w].sum())
        region_area = w * h
        rows = int((trans[y:y+h, x:x+w].sum(axis=1) > 0).sum())
        ratio = w / max(h, 1)
        if ratio < 1:
            ratio = 1 / ratio
        shape_penalty = 1.0 / (1.0 + max(0.0, ratio - 1.6) ** 2)
        eff_rows = min(rows, 15)
        v_sum = region_trans
        h_sum = float(h_trans[y:y+h, x:x+w].sum())
        purity = v_sum / max(v_sum + h_sum, 1e-6)
        score = (region_trans / max(region_area, 1)) * eff_rows * shape_penalty * (purity ** 3)
        out.append({
            "box": (x0, y0, x1, y1),
            "stripe_box": (x, y, x + w, y + h),
            "center": ((x0 + x1) / 2, (y0 + y1) / 2),
            "score": score,
            "stripe_rows": rows,
            "purity": purity,
        })
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


# ── Detector 2: HEAD (glasses + hat-band + face) ─────────────────────────────
def _ring_fraction(darkb, cx, cy, W, H):
    """Fraction of 16 compass points (r=2,3) that are dark — high for a ringed disc."""
    hits = tot = 0
    for r in (2, 3):
        for ddx, ddy in [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]:
            x, y = int(cx + ddx*r), int(cy + ddy*r)
            if 0 <= x < W and 0 <= y < H:
                tot += 1
                hits += bool(darkb[y, x])
    return hits / max(tot, 1)


def _band_run(whiteb, cx, cy):
    """Height of the white run above the lenses — a hat BAND is thin (1-7px)."""
    x, y, skipped = int(cx), int(cy) - 3, 0
    while y > 0 and not whiteb[y, x] and skipped < 8:
        y -= 1; skipped += 1
    run = 0
    while y > 0 and whiteb[y, x] and run < 25:
        y -= 1; run += 1
    return run


def find_heads(bgr, red, white, skin, dark):
    H, W = red.shape
    pale = cv2.bitwise_or(white, skin)

    # Lens cores: white pixels enclosed by dark on all four sides
    lens = np.zeros((H, W), np.uint8)
    for d in (2, 3, 4):
        lens |= ((white > 0) &
                 (_shift(dark, 0, d) > 0) & (_shift(dark, 0, -d) > 0) &
                 (_shift(dark, d, 0) > 0) & (_shift(dark, -d, 0) > 0)).astype(np.uint8)

    hat_map = np.zeros((H, W), np.float32)
    for du in (2, 3, 4, 5, 6, 7):
        hat_map += ((white > 0) & (_shift(red, 0, du) > 0)).astype(np.float32)
    stripes = stripe_transition_map(red, white)

    I_dark = _integral(dark > 0)
    I_hat = _integral(hat_map)
    I_red = _integral(red > 0)
    I_pale = _integral(pale > 0)
    I_str = _integral(stripes)
    darkb, whiteb = dark > 0, white > 0

    # Candidate lens blobs
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(lens, connectivity=8)
    blobs = []
    for i in range(1, n):
        if (stats[i, cv2.CC_STAT_AREA] <= 30 and
                stats[i, cv2.CC_STAT_WIDTH] <= 8 and stats[i, cv2.CC_STAT_HEIGHT] <= 8):
            cx, cy = cents[i]
            r8 = _ring_fraction(darkb, cx, cy, W, H)
            if r8 >= 0.5:
                blobs.append((cx, cy, r8))
    blobs.sort()

    raw = []
    for i in range(len(blobs)):
        x1, y1, r1 = blobs[i]
        for j in range(i + 1, len(blobs)):
            x2, y2, r2 = blobs[j]
            dxp = x2 - x1
            if dxp > 14:
                break
            if not (4 <= dxp <= 14 and abs(y2 - y1) <= 2.5):
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            icx, icy = int(cx), int(cy)
            hat_ev = _boxsum(I_hat, icx-11, icy-18, icx+12, icy-2)
            if hat_ev < 4 or hat_ev > 120:
                continue
            red_above = _boxfrac(I_red, icx-9, icy-22, icx+10, icy-8)
            if red_above < 0.18:
                continue
            face_pale = _boxfrac(I_pale, icx-7, icy+1, icx+8, icy+9)
            if face_pale < 0.25:
                continue
            d_fr = _boxfrac(I_dark, icx-12, icy-3, icx+13, icy+13)
            if not (0.12 <= d_fr <= 0.60):
                continue
            runs = [r for r in (_band_run(whiteb, x1, y1),
                                _band_run(whiteb, x2, y2),
                                _band_run(whiteb, cx, cy)) if r > 0]
            if not runs or not (1 <= sorted(runs)[len(runs)//2] <= 7):
                continue
            strv = _boxsum(I_str, icx-14, icy-20, icx+15, icy+30)
            score = (min(hat_ev, 60) * (r1 + r2) * (0.5 + red_above)
                     * (0.5 + face_pale) * (1 + min(strv, 60) / 60.0))
            raw.append({"center": (cx, cy), "score": score,
                        "lens_gap": dxp, "hat": hat_ev,
                        "red_above": red_above, "face": face_pale})

    raw.sort(key=lambda c: c["score"], reverse=True)
    final = []
    for c in raw:
        cx, cy = c["center"]
        if all((cx - k["center"][0])**2 + (cy - k["center"][1])**2 > 14**2 for k in final):
            final.append(c)
    return final


# ── Detector 3: FUSION ───────────────────────────────────────────────────────
def find_candidates(bgr, top_n=TOP_N_DEFAULT):
    """
    Returns a unified, ranked candidate list.  Each candidate dict has:
      box, center, score, kind ('fused'|'head'|'torso'), and feature notes.
    """
    red, white, skin, dark, blue = build_masks(bgr)
    torsos = find_torsos(bgr, red, white, blue, dark)
    heads = find_heads(bgr, red, white, skin, dark)

    # Normalize each detector's scores to 0..1 so they can be combined fairly
    def _norm(items, key="score"):
        if not items:
            return
        mx = max(i[key] for i in items) or 1.0
        for i in items:
            i["nscore"] = i[key] / mx
    _norm(torsos)
    _norm(heads)

    candidates = []
    used_torsos = set()

    # The HEAD detector is the most discriminative feature (glasses + hat-band
    # are nearly unique to Waldo), so head score is the backbone of ranking.
    # A torso only adds a modest BONUS, and only when it's tightly stacked
    # directly beneath the head — not merely "somewhere nearby". This stops
    # the page's many red-white distractors from boosting every head equally.
    for hd in heads:
        hx, hy = hd["center"]
        best_t, best_i, best_align = None, None, 0.0
        for idx, t in enumerate(torsos):
            tx0, ty0, tx1, ty1 = t["box"]
            # Use the tight stripe_box for centering (the padded box drifts when
            # an arm or held object inflates one side).
            sx0, sy0, sx1, sy1 = t["stripe_box"]
            tcx = (sx0 + sx1) / 2
            horiz_off = abs(hx - tcx)
            vert_gap = sy0 - hy
            # Wider horizontal window (22px): Waldo often holds a sign/cane that
            # shifts his visible stripe block sideways from his head.
            if horiz_off <= 22 and -12 <= vert_gap <= 50:
                align = (1 - horiz_off / 22) * (1 - abs(vert_gap - 15) / 40)
                if align > best_align:
                    best_t, best_i, best_align = t, idx, align

        # Base score is the head score (scaled up). Torso adds up to +25%.
        base = 60.0 * hd["nscore"] + 10.0
        if best_t is not None and best_align > 0.25:
            used_torsos.add(best_i)
            tx0, ty0, tx1, ty1 = best_t["box"]
            box = (min(tx0, int(hx) - 18), int(hy) - 24,
                   max(tx1, int(hx) + 18), max(ty1, int(hy) + 10))
            score = base * (1 + 0.25 * best_align * best_t["nscore"])
            kind = "fused"
        else:
            box = (int(hx) - 20, int(hy) - 24, int(hx) + 20, int(hy) + 40)
            score = base
            kind = "head"
        candidates.append({"box": box, "center": (hx, hy), "score": score,
                           "kind": kind, "purity": best_t["purity"] if best_t else 0,
                           "stripe_rows": best_t["stripe_rows"] if best_t else 0})

    # Remaining torsos (no head found) — weaker, lots of red-white distractors
    for idx, t in enumerate(torsos):
        if idx in used_torsos:
            continue
        candidates.append({"box": t["box"], "center": t["center"],
                           "score": 35.0 * t["nscore"], "kind": "torso",
                           "purity": t.get("purity", 0),
                           "stripe_rows": t.get("stripe_rows", 0)})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_n]


# ── Drawing ──────────────────────────────────────────────────────────────────
def annotate(bgr, candidates):
    out = bgr.copy()
    for rank, c in enumerate(candidates, start=1):
        x0, y0, x1, y1 = c["box"]
        if rank == 1:
            color, thick = (0, 215, 255), 3
            label = f"#1 WALDO? ({c['kind']})"
        else:
            color, thick = (160, 160, 160), 2
            label = f"#{rank} {c['kind']}"
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = y0 - 6 if y0 - th - 8 > 0 else y1 + th + 6
        cv2.rectangle(out, (x0, ly - th - 4), (x0 + tw + 6, ly + 3), color, -1)
        cv2.putText(out, label, (x0 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage:  python waldo_finder.py <image_path> [num_candidates]")
        sys.exit(1)
    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_N_DEFAULT

    bgr = cv2.imread(path)
    if bgr is None:
        print(f"Error: could not load '{path}'")
        sys.exit(1)

    candidates = find_candidates(bgr, top_n)
    if not candidates:
        print("No candidates found. Try loosening the HSV ranges for this scan.")
        sys.exit(0)

    print(f"\nTop {len(candidates)} candidates (best first):\n")
    for rank, c in enumerate(candidates, start=1):
        cx, cy = c["center"]
        print(f"  #{rank}  score={c['score']:6.1f}  kind={c['kind']:<6}  "
              f"at ({cx:.0f},{cy:.0f})")

    out = annotate(bgr, candidates)
    out_path = path.rsplit(".", 1)[0] + "_waldo.png"
    cv2.imwrite(out_path, out)
    print(f"\nAnnotated image saved -> {out_path}")

    cv2.imshow("Waldo Finder", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
