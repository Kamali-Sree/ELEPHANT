"""
MACONFLIC - Organic Terrain Edge Polygon Generator
===================================================
Automatically generates dense, organic-looking polygon boundaries for each
village satellite screenshot. The boundary closely follows the actual terrain
content edges, producing cutouts that look like the reference Tamil Nadu
silhouette style -- irregular, organic, landscape-following outlines.

Method:
  1. Detect the actual content region of the image (excludes watermarks/UI bars).
  2. Generate a tight initial polygon by sampling around the image perimeter
     at a controlled inset from each edge.
  3. Apply terrain-responsive perturbation: varies the boundary inward/outward
     based on local pixel brightness variance, mimicking the irregular edge
     where dense forest or terrain meets a notional "edge".
  4. Smooth slightly to avoid single-pixel jags while preserving character.
  5. Save as polygon annotation JSON (compatible with generate_village_polygon_cutouts.py).

Run:
    python scripts/auto_generate_organic_polygons.py
    python scripts/auto_generate_organic_polygons.py --village kodanur
    python scripts/auto_generate_organic_polygons.py --preview    # Show preview before saving
"""

import os
import sys
import json
import math
import argparse
import random
from datetime import datetime

try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy required. Run: pip install numpy")
    sys.exit(1)

try:
    from PIL import Image, ImageFilter, ImageDraw, ImageFont
except ImportError:
    print("[ERROR] Pillow required. Run: pip install Pillow")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────
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

# ── Perlin-style noise (pure numpy, no external deps) ──────────────────────

def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def _lerp(a, b, t):
    return a + t * (b - a)

def smooth_noise_1d(n_points: int, seed: int = 42, octaves: int = 4,
                    persistence: float = 0.55) -> np.ndarray:
    """
    Generate smooth 1-D noise in range [-1, 1] with multiple octaves.
    Used to drive organic boundary perturbation.
    """
    rng = np.random.default_rng(seed)
    result = np.zeros(n_points)
    amplitude = 1.0
    frequency = 1.0
    max_amp   = 0.0

    for _ in range(octaves):
        # Random control points
        ctrl_len = max(4, int(n_points * frequency / n_points * 20) + 2)
        ctrl = rng.uniform(-1, 1, ctrl_len)
        # Interpolate to full length
        xs  = np.linspace(0, ctrl_len - 1, n_points)
        idx = np.clip(np.floor(xs).astype(int), 0, ctrl_len - 2)
        t   = xs - idx
        t_f = _fade(t)
        interp = _lerp(ctrl[idx], ctrl[idx + 1], t_f)
        result   += interp * amplitude
        max_amp  += amplitude
        amplitude *= persistence
        frequency *= 2.0

    return result / max_amp   # normalise to [-1, 1]


# ── Core polygon generator ──────────────────────────────────────────────────

def detect_content_inset(img: Image.Image) -> tuple:
    """
    Detect watermark / UI bar regions at image borders.
    Returns (top_inset, bottom_inset, left_inset, right_inset) in pixels.
    Esri screenshots typically have a status bar at the bottom.
    """
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    top_in, bot_in, lft_in, rgt_in = 0, 0, 0, 0

    # Bottom bar: look for rows with very low variance (solid UI bar)
    for y in range(h - 1, max(h - 60, 0), -1):
        row_var = float(arr[y].std())
        if row_var < 18:        # Very uniform row = UI bar
            bot_in = h - y
        else:
            break

    return top_in, bot_in, lft_in, rgt_in


def generate_organic_polygon(img: Image.Image,
                              n_boundary_points: int = 120,
                              base_inset_frac: float = 0.03,
                              max_perturb_frac: float = 0.08,
                              noise_seed: int = 42,
                              smooth_passes: int = 2) -> list:
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
