"""
MACONFLIC - Organic Terrain-Edge Polygon Generator  (v2 — ray-cast approach)
=============================================================================
Produces genuinely organic, terrain-following polygon cutouts for each village
satellite screenshot.  Each village gets a completely independent shape derived
from its own image content — NOT a perturbed ellipse.

Algorithm
---------
1.  Load the source satellite PNG (RGB, full resolution).
2.  Detect and strip bottom/top UI bars (Esri watermark row = low variance).
3.  Build a local-variance saliency map: each pixel scores how much
    colour variation exists in its local neighbourhood.
4.  For N angles around the image centre, cast a ray outward to the content
    boundary, then walk inward until meaningful terrain is found.
5.  Apply two-layer macro + micro noise perturbation (large amplitude) so
    the boundary has genuine bays, peninsulas, and concavities.
6.  Light 3-neighbour smoothing preserves organic character without ovals.
7.  Save annotation JSON compatible with generate_village_polygon_cutouts.py.

Run:
    python scripts/auto_generate_organic_polygons.py
    python scripts/auto_generate_organic_polygons.py --village kodanur
    python scripts/auto_generate_organic_polygons.py --force    # overwrite existing
    python scripts/auto_generate_organic_polygons.py --preview  # save debug preview
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime

try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy required. Run: pip install numpy")
    sys.exit(1)

try:
    from PIL import Image, ImageFilter, ImageDraw
except ImportError:
    print("[ERROR] Pillow required. Run: pip install Pillow")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR      = os.path.join(REPO_ROOT, "data", "Seven Villages")
ANNOTATION_DIR = os.path.join(REPO_ROOT, "data", "village_polygon_cutouts", "annotations")
OUTPUT_DIR     = os.path.join(REPO_ROOT, "data", "village_polygon_cutouts", "outputs")
os.makedirs(ANNOTATION_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

VILLAGE_MAP = {
    "kodanur":    "kodanur.png",
    "pudur":      "pudur.png",
    "panapalli":  "panapalli.png",
    "kandivazhi": "kandivazhi.png",
    "sembukarai": "sembukarai.png",
}

# Per-village tuning: each village gets its own seed, frequency, and amplitude
# so every polygon shape is visually distinct from all other villages.
VILLAGE_PARAMS = {
    "kodanur":    {"seed": 11,  "n_pts": 300, "macro_amp": 0.22, "micro_amp": 0.10,
                   "macro_freq": 5,  "micro_freq": 18, "edge_margin": 0.04,
                   "content_pull": 0.18, "smooth_passes": 3},
    "pudur":      {"seed": 37,  "n_pts": 280, "macro_amp": 0.26, "micro_amp": 0.11,
                   "macro_freq": 6,  "micro_freq": 20, "edge_margin": 0.05,
                   "content_pull": 0.20, "smooth_passes": 3},
    "panapalli":  {"seed": 53,  "n_pts": 290, "macro_amp": 0.20, "micro_amp": 0.09,
                   "macro_freq": 4,  "micro_freq": 16, "edge_margin": 0.04,
                   "content_pull": 0.16, "smooth_passes": 3},
    "kandivazhi": {"seed": 79,  "n_pts": 320, "macro_amp": 0.28, "micro_amp": 0.12,
                   "macro_freq": 7,  "micro_freq": 22, "edge_margin": 0.03,
                   "content_pull": 0.22, "smooth_passes": 3},
    "sembukarai": {"seed": 97,  "n_pts": 310, "macro_amp": 0.24, "micro_amp": 0.10,
                   "macro_freq": 5,  "micro_freq": 19, "edge_margin": 0.04,
                   "content_pull": 0.19, "smooth_passes": 3},
}


# ── Smooth periodic noise ─────────────────────────────────────────────────────

def _smooth_noise(n: int, seed: int, freq: float,
                  octaves: int = 4, persistence: float = 0.55) -> np.ndarray:
    """
    Periodic 1-D noise in [-1, 1] via layered sinusoids with random phase.
    Periodic so the polygon ring wraps seamlessly.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    result = np.zeros(n)
    amp = 1.0
    total = 0.0
    for k in range(1, octaves + 1):
        phase = rng.uniform(0, 2 * math.pi)
        result += amp * np.sin(freq * k * t + phase)
        total  += amp
        amp    *= persistence
    return result / total   # [-1, 1]


# ── UI bar detection ──────────────────────────────────────────────────────────

def detect_ui_insets(arr: np.ndarray) -> tuple:
    """
    Return (top_px, bottom_px, left_px, right_px) of low-variance UI chrome
    (Esri / Google watermark bars).
    """
    h, w = arr.shape[:2]
    top_px = bot_px = lft_px = rgt_px = 0

    for y in range(h - 1, max(h - 80, 0), -1):
        if float(arr[y].std()) < 20:
            bot_px = h - y
        else:
            break
    for y in range(0, min(80, h)):
        if float(arr[y].std()) < 20:
            top_px = y + 1
        else:
            break
    for x in range(0, min(60, w)):
        if float(arr[:, x].std()) < 20:
            lft_px = x + 1
        else:
            break
    for x in range(w - 1, max(w - 60, 0), -1):
        if float(arr[:, x].std()) < 20:
            rgt_px = w - x
        else:
            break

    return top_px, bot_px, lft_px, rgt_px


# ── Local-variance saliency map ───────────────────────────────────────────────

def build_saliency(arr_gray: np.ndarray, radius: int = 12) -> np.ndarray:
    """
    Float32 saliency map: local standard deviation of pixel brightness.
    High value = rich terrain content (roads, buildings, varied vegetation).
    Uses two box-blur passes (fast, no scipy/cv2 needed).
    """
    pil = Image.fromarray(arr_gray)
    br  = radius * 2 + 1
    # E[X]
    mean_arr = np.array(pil.filter(ImageFilter.BoxBlur(br)), dtype=np.float32)
    # E[X^2]  — scale to stay in uint8 range for the blur, then un-scale
    sq_arr = np.clip(arr_gray.astype(np.float32) ** 2 / 255.0, 0, 255).astype(np.uint8)
    mean_sq = np.array(
        Image.fromarray(sq_arr).filter(ImageFilter.BoxBlur(br)), dtype=np.float32
    ) * 255.0
    variance = np.maximum(0.0, mean_sq - mean_arr ** 2)
    std_map  = np.sqrt(variance)
    mx = std_map.max()
    if mx > 0:
        std_map /= mx
    return std_map


# ── Ray-cast boundary finder ──────────────────────────────────────────────────

def raycast_boundary(saliency: np.ndarray,
                     cx: float, cy: float,
                     angle: float,
                     x_min: int, x_max: int,
                     y_min: int, y_max: int,
                     content_pull: float = 0.18,
                     step: float = 2.0) -> tuple:
    """
    From (cx, cy) fire a ray at 'angle' to the content rectangle boundary.
    Walk INWARD along the ray accumulating saliency until the cumulative
    sum exceeds content_pull * ray_length — that is where meaningful terrain
    content begins.  Returns (bx, by) in image pixel coordinates.

    Effect: the polygon naturally follows where terrain actually is, not
    just a fixed ellipse distance from the image centre.
    """
    h, w = saliency.shape
    dx = math.cos(angle)
    dy = math.sin(angle)

    # Maximum t before leaving the valid rectangle
    if abs(dx) > 1e-9:
        t_x = ((x_max if dx > 0 else x_min) - cx) / dx
    else:
        t_x = 1e9
    if abs(dy) > 1e-9:
        t_y = ((y_max if dy > 0 else y_min) - cy) / dy
    else:
        t_y = 1e9
    t_max = min(t_x, t_y) - step
    if t_max <= step:
        return (cx, cy)

    target = content_pull * (t_max / step)
    cumulative = 0.0
    t = t_max

    while t > step:
        ix = max(0, min(w - 1, int(cx + dx * t)))
        iy = max(0, min(h - 1, int(cy + dy * t)))
        cumulative += saliency[iy, ix]
        if cumulative >= target:
            break
        t -= step

    bx = max(float(x_min), min(float(x_max), cx + dx * t))
    by = max(float(y_min), min(float(y_max), cy + dy * t))
    return (bx, by)


# ── Main polygon generator ────────────────────────────────────────────────────

def generate_organic_polygon(img: Image.Image,
                              n_pts: int = 300,
                              macro_amp: float = 0.22,
                              micro_amp: float = 0.10,
                              macro_freq: int = 5,
                              micro_freq: int = 18,
                              edge_margin: float = 0.04,
                              content_pull: float = 0.18,
                              seed: int = 42,
                              smooth_passes: int = 3) -> list:
    """
    Generate a unique, terrain-following organic polygon for one village.

    Steps
    -----
    1.  Build a local-variance saliency map from the satellite image.
    2.  For each of n_pts angles, ray-cast from the image centre outward
        and find where the terrain content begins (boundary point).
    3.  Apply two-layer sinusoidal noise (macro + micro) with large
        amplitude so the boundary develops genuine bays and peninsulas.
    4.  Lightly smooth (3-neighbour, low weight on neighbours) to remove
        pixel-level jags while preserving concavities.

    Returns list of [x, y] pixel coordinate pairs.
    """
    w, h    = img.size
    arr_rgb  = np.array(img.convert("RGB"))
    arr_gray = np.array(img.convert("L"))

    # Detect and exclude UI chrome
    ti, bi, li, ri = detect_ui_insets(arr_rgb)
    mgx = max(int(w * edge_margin), li + 4, ri + 4)
    mgy = max(int(h * edge_margin), ti + 4)
    mgy_bot = max(mgy, bi + int(h * 0.04))

    x_min, x_max = mgx, w - mgx
    y_min, y_max = mgy, h - mgy_bot
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half_dim = max((x_max - x_min) / 2.0, (y_max - y_min) / 2.0)

    # Saliency map
    saliency = build_saliency(arr_gray, radius=14)

    # Ray-cast boundary for every angle
    angles   = [2 * math.pi * i / n_pts for i in range(n_pts)]
    raw_pts  = [raycast_boundary(saliency, cx, cy, a,
                                  x_min, x_max, y_min, y_max,
                                  content_pull=content_pull, step=2.0)
                for a in angles]
    raw_r    = [math.hypot(p[0] - cx, p[1] - cy) for p in raw_pts]

    # Two-layer noise: macro (large bays) + micro (small indentations)
    noise_macro = _smooth_noise(n_pts, seed=seed,      freq=macro_freq, octaves=5, persistence=0.60)
    noise_micro = _smooth_noise(n_pts, seed=seed + 13, freq=micro_freq, octaves=3, persistence=0.45)

    points = []
    for i, angle in enumerate(angles):
        r_base   = raw_r[i]
        macro_d  = noise_macro[i] * macro_amp * half_dim
        micro_d  = noise_micro[i] * micro_amp * half_dim
        r_new    = max(r_base + macro_d + micro_d, half_dim * 0.12)

        px = cx + r_new * math.cos(angle)
        py = cy + r_new * math.sin(angle)
        px = max(float(x_min), min(float(x_max), px))
        py = max(float(y_min), min(float(y_max), py))
        points.append([px, py])

    # Light smoothing — weights: prev 0.15, curr 0.70, next 0.15
    # Keeps concavities because neighbour weight is low
    for _ in range(smooth_passes):
        n = len(points)
        smoothed = []
        for i in range(n):
            prev = points[(i - 1) % n]
            curr = points[i]
            nxt  = points[(i + 1) % n]
            sx = prev[0] * 0.15 + curr[0] * 0.70 + nxt[0] * 0.15
            sy = prev[1] * 0.15 + curr[1] * 0.70 + nxt[1] * 0.15
            smoothed.append([round(sx, 1), round(sy, 1)])
        points = smoothed

    return points


# ── Preview helper ────────────────────────────────────────────────────────────

def make_cutout_preview(orig_img: Image.Image, points: list) -> Image.Image:
    """
    Transparent RGBA cutout composited over a checker background.
    The checker is ONLY in this preview — the actual cutout PNG stays clean.
    """
    w, h = orig_img.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(
        [(int(round(x)), int(round(y))) for x, y in points], fill=255
    )
    rgba = orig_img.convert("RGBA")
    rgba.putalpha(mask)

    checker = Image.new("RGB", (w, h), (35, 35, 35))
    cdraw   = ImageDraw.Draw(checker)
    cell    = 20
    for cy_ in range(0, h, cell):
        for cx_ in range(0, w, cell):
            if ((cx_ // cell) + (cy_ // cell)) % 2 == 0:
                cdraw.rectangle([cx_, cy_, cx_ + cell - 1, cy_ + cell - 1],
                                fill=(55, 55, 55))
    checker.paste(rgba, (0, 0), rgba)

    max_dim = 1024
    scale   = min(1.0, max_dim / max(w, h))
    return checker.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


# ── Per-village processing ────────────────────────────────────────────────────

def process_village(village: str, show_preview: bool = False, force: bool = False):
    src_file = VILLAGE_MAP[village]
    img_path = os.path.join(INPUT_DIR, src_file)
    ann_path = os.path.join(ANNOTATION_DIR, f"{village}_polygon.json")

    if not os.path.exists(img_path):
        print(f"  [ERROR] Source image not found: {img_path}")
        return False

    if os.path.exists(ann_path) and not force:
        with open(ann_path) as f:
            existing = json.load(f)
        method = existing.get("annotation_method", "")
        if "initial_seed" not in method:
            print(f"  [{village}] Annotation already exists ({method}). "
                  "Use --force to overwrite.")
            return True
        print(f"  [{village}] Replacing initial-seed annotation.")

    img  = Image.open(img_path).convert("RGB")
    w, h = img.size
    p    = VILLAGE_PARAMS.get(village, {})

    print(f"  [{village}] {w}x{h} — generating {p.get('n_pts', 300)}-vertex organic polygon …")

    points = generate_organic_polygon(
        img,
        n_pts        = p.get("n_pts",         300),
        macro_amp    = p.get("macro_amp",      0.22),
        micro_amp    = p.get("micro_amp",      0.10),
        macro_freq   = p.get("macro_freq",     5),
        micro_freq   = p.get("micro_freq",     18),
        edge_margin  = p.get("edge_margin",    0.04),
        content_pull = p.get("content_pull",   0.18),
        seed         = p.get("seed",           42),
        smooth_passes = p.get("smooth_passes", 3),
    )
    print(f"  [{village}] {len(points)} vertices generated.")

    if show_preview:
        try:
            preview = make_cutout_preview(img, points)
            prev_path = os.path.join(OUTPUT_DIR, f"{village}_organic_boundary_preview.jpg")
            preview.save(prev_path, quality=92)
            print(f"  [{village}] Preview → {prev_path}")
        except Exception as e:
            print(f"  [{village}] Preview error: {e}")

    annotation = {
        "village":           village,
        "source_image":      src_file,
        "image_size":        [w, h],
        "points":            points,
        "annotated_at":      datetime.now().isoformat(timespec="seconds"),
        "annotation_method": "auto_organic_raycast_v2",
        "generator_params":  p,
    }
    with open(ann_path, "w") as f:
        json.dump(annotation, f, indent=2)
    print(f"  [{village}] Annotation saved → {ann_path}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MACONFLIC - Organic Polygon Generator v2 (ray-cast)"
    )
    parser.add_argument("--village", choices=list(VILLAGE_MAP.keys()), default=None,
                        help="Process only one village (default: all 5).")
    parser.add_argument("--force",   action="store_true",
                        help="Overwrite existing non-seed annotations too.")
    parser.add_argument("--preview", action="store_true",
                        help="Save a boundary-preview JPG for visual inspection.")
    args = parser.parse_args()

    villages = [args.village] if args.village else list(VILLAGE_MAP.keys())

    print("=" * 60)
    print("MACONFLIC - Organic Polygon Generator v2  (ray-cast)")
    print("=" * 60)

    ok = 0
    for v in villages:
        print(f"\n▶  {v}")
        if process_village(v, show_preview=args.preview, force=args.force):
            ok += 1

    print(f"\n[DONE] {ok}/{len(villages)} villages processed.")
    print("Next step:  python scripts\\generate_village_polygon_cutouts.py --force")


if __name__ == "__main__":
    main()


    """
    Generate an organic, terrain-responsive polygon.

    Strategy:
      • Place n_boundary_points evenly around an ellipse that fits snugly
        inside the image (after content-inset detection).
      • For each point, sample the local image region and compute a
        terrain-variance score.
      • Perturb each point outward/inward using layered smooth noise,
        scaled by the terrain score so that dense-content edges stay
        closer to the image boundary.
      • Smooth the polygon mildly to remove spikes while keeping character.

    Returns: list of [x, y] pairs (float, image pixel coordinates).
    """
    w, h = img.size
    arr  = np.array(img.convert("L"), dtype=np.float32)

    # Detect UI bars
    ti, bi, li, ri = detect_content_inset(img)
    bi = max(bi, int(h * 0.04))   # Always trim at least 4% for Esri watermarks

    # Usable region
    x_min = li + int(w * base_inset_frac)
    x_max = w - ri - int(w * base_inset_frac)
    y_min = ti + int(h * base_inset_frac)
    y_max = h - bi - int(h * base_inset_frac)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    rx = (x_max - x_min) / 2.0
    ry = (y_max - y_min) / 2.0

    # Generate noise signals (two independent components for x and y)
    noise_r  = smooth_noise_1d(n_boundary_points, seed=noise_seed,   octaves=5, persistence=0.60)
    noise_r2 = smooth_noise_1d(n_boundary_points, seed=noise_seed+7, octaves=3, persistence=0.45)

    combined_noise = noise_r * 0.65 + noise_r2 * 0.35

    # Sample terrain variance at each candidate boundary point
    sample_pad = 12    # px radius for variance patch
    terrain_scores = []
    base_angles = [2 * math.pi * i / n_boundary_points for i in range(n_boundary_points)]

    for angle in base_angles:
        bx = int(cx + rx * math.cos(angle))
        by = int(cy + ry * math.sin(angle))
        bx = max(sample_pad, min(w - sample_pad - 1, bx))
        by = max(sample_pad, min(h - sample_pad - 1, by))
        patch = arr[by - sample_pad: by + sample_pad, bx - sample_pad: bx + sample_pad]
        score = float(patch.std()) / 80.0   # Normalise; high variance = rich terrain
        terrain_scores.append(min(1.0, score))

    # Build polygon points
    points = []
    max_perturb_x = rx * max_perturb_frac * 3.0
    max_perturb_y = ry * max_perturb_frac * 3.0

    for i, angle in enumerate(base_angles):
        # Radial perturbation scaled by noise and terrain variance
        t_score   = terrain_scores[i]
        n_val     = combined_noise[i]     # -1 .. 1

        # Rich terrain areas get pushed OUT more (more content to show)
        # Low-variance areas get pulled IN slightly
        push  = n_val * max_perturb_frac * (0.6 + 0.4 * t_score)
        r_mod = 1.0 + push

        px = cx + rx * r_mod * math.cos(angle)
        py = cy + ry * r_mod * math.sin(angle)

        # Clamp to image
        px = max(2.0, min(float(w - 2), px))
        py = max(2.0, min(float(h - 2), py))
        points.append([px, py])

    # Smooth passes (simple moving average on ring)
    for _ in range(smooth_passes):
        smoothed = []
        n = len(points)
        for i in range(n):
            prev = points[(i - 1) % n]
            curr = points[i]
            nxt  = points[(i + 1) % n]
            sx = prev[0] * 0.2 + curr[0] * 0.6 + nxt[0] * 0.2
            sy = prev[1] * 0.2 + curr[1] * 0.6 + nxt[1] * 0.2
            smoothed.append([round(sx, 1), round(sy, 1)])
        points = smoothed

    return points


# ── Preview helper ──────────────────────────────────────────────────────────

def make_preview(orig_img: Image.Image, points: list,
                 village: str) -> Image.Image:
    """
    Composite: left = outline overlay, right = transparent cutout on checker.
    """
    from PIL import ImageDraw

    w, h = orig_img.size

    # Cutout
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(
        [(int(x), int(y)) for x, y in points], fill=255
    )
    rgba = orig_img.convert("RGBA")
    rgba.putalpha(mask)

    # Checker background
    checker = Image.new("RGB", (w, h), (35, 35, 35))
    cdraw = ImageDraw.Draw(checker)
    cell = 20
    for cy in range(0, h, cell):
        for cx in range(0, w, cell):
            if ((cx // cell) + (cy // cell)) % 2 == 0:
                cdraw.rectangle([cx, cy, cx + cell - 1, cy + cell - 1], fill=(55, 55, 55))
    checker.paste(rgba, (0, 0), rgba)

    # Scale for display
    max_dim = 900
    scale   = min(1.0, max_dim / max(w, h))
    out_w, out_h = int(w * scale), int(h * scale)
    return checker.resize((out_w, out_h), Image.Resampling.LANCZOS)


# ── Per-village seed tuning ─────────────────────────────────────────────────
# Each village gets its own noise seed so boundaries are visually distinct.
VILLAGE_PARAMS = {
    "kodanur":    {"seed": 11, "n_pts": 140, "max_perturb": 0.09, "base_inset": 0.025},
    "pudur":      {"seed": 37, "n_pts": 120, "max_perturb": 0.10, "base_inset": 0.03},
    "panapalli":  {"seed": 53, "n_pts": 130, "max_perturb": 0.085,"base_inset": 0.025},
    "kandivazhi": {"seed": 79, "n_pts": 150, "max_perturb": 0.095,"base_inset": 0.02},
    "sembukarai": {"seed": 97, "n_pts": 135, "max_perturb": 0.09, "base_inset": 0.03},
}


# ── Main ────────────────────────────────────────────────────────────────────

def process_village(village: str, show_preview: bool = False, force: bool = False):
    src_file = VILLAGE_MAP[village]
    img_path = os.path.join(INPUT_DIR, src_file)
    ann_path = os.path.join(ANNOTATION_DIR, f"{village}_polygon.json")

    if not os.path.exists(img_path):
        print(f"  [ERROR] Source image not found: {img_path}")
        return

    if os.path.exists(ann_path) and not force:
        # Check if it's an initial seed — replace it
        with open(ann_path) as f:
            existing = json.load(f)
        if "initial_seed" not in existing.get("annotation_method", ""):
            print(f"  [{village}] Annotation already exists (non-seed). Use --force to overwrite.")
            return
        print(f"  [{village}] Replacing initial seed annotation with organic boundary.")

    img    = Image.open(img_path).convert("RGB")
    w, h   = img.size
    params = VILLAGE_PARAMS.get(village, {})

    print(f"  [{village}] Generating organic polygon ({w}x{h})...")
    points = generate_organic_polygon(
        img,
        n_boundary_points = params.get("n_pts", 120),
        base_inset_frac   = params.get("base_inset", 0.03),
        max_perturb_frac  = params.get("max_perturb", 0.08),
        noise_seed        = params.get("seed", 42),
        smooth_passes     = 3,
    )
    print(f"  [{village}] {len(points)} vertices generated.")

    if show_preview:
        try:
            preview = make_preview(img, points, village)
            preview.save(os.path.join(OUTPUT_DIR, f"{village}_organic_preview_check.jpg"), quality=90)
            print(f"  [{village}] Preview saved to outputs/ for inspection.")
        except Exception as e:
            print(f"  [{village}] Preview error: {e}")

    annotation = {
        "village": village,
        "source_image": src_file,
        "image_size": [w, h],
        "points": points,
        "annotated_at": datetime.now().isoformat(timespec="seconds"),
        "annotation_method": "auto_organic_terrain_edge"
    }
    with open(ann_path, "w") as f:
        json.dump(annotation, f, indent=2)
    print(f"  [{village}] Saved: {ann_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MACONFLIC - Auto Organic Polygon Generator"
    )
    parser.add_argument("--village", choices=list(VILLAGE_MAP.keys()), default=None)
    parser.add_argument("--force",   action="store_true",
                        help="Overwrite existing manual annotations too.")
    parser.add_argument("--preview", action="store_true",
                        help="Save a preview JPG for visual inspection before running cutouts.")
    args = parser.parse_args()

    villages = [args.village] if args.village else list(VILLAGE_MAP.keys())

    print("=" * 60)
    print("MACONFLIC - Organic Polygon Boundary Generator")
    print("=" * 60)
    for v in villages:
        print(f"\n{v}:")
        process_village(v, show_preview=args.preview, force=args.force)

    print("\n[DONE] Annotations updated.")
    print("Now run:")
    print("  python scripts\\generate_village_polygon_cutouts.py --force")


if __name__ == "__main__":
    main()
