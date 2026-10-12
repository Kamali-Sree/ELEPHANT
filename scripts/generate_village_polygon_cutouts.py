"""
MACONFLIC - Village Polygon Cutout Batch Processor
====================================================
Reads saved polygon annotations and produces final output files:

  <village>_polygon_mask.png          -- Binary mask (white inside, black outside)
  <village>_polygon_cutout.png        -- RGBA cutout (transparent outside polygon)
  <village>_polygon_outline.png       -- RGB preview with polygon perimeter drawn
  <village>_polygon_preview.jpg       -- Preview composited over contrast background
  <village>_polygon_metadata.json     -- Processing metadata

Usage:
    python scripts/generate_village_polygon_cutouts.py

    # Process single village:
    python scripts/generate_village_polygon_cutouts.py --village kodanur

    # Force regenerate even if outputs exist:
    python scripts/generate_village_polygon_cutouts.py --force
"""

import os
import sys
import json
import argparse
from datetime import datetime

try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy is required. Run: pip install numpy")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("[ERROR] Pillow is required. Run: pip install Pillow")
    sys.exit(1)

# ---- Paths ---------------------------------------------------------------
REPO_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR      = os.path.join(REPO_ROOT, "data", "Seven Villages")
ANNOTATION_DIR = os.path.join(REPO_ROOT, "data", "village_polygon_cutouts", "annotations")
OUTPUT_DIR     = os.path.join(REPO_ROOT, "data", "village_polygon_cutouts", "outputs")
VALIDATION_OUT = os.path.join(REPO_ROOT, "results", "village_polygon_cutout_validation.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(VALIDATION_OUT), exist_ok=True)

ALL_VILLAGES = {
    "kodanur":    "kodanur.png",
    "pudur":      "pudur.png",
    "panapalli":  "panapalli.png",
    "kandivazhi": "kandivazhi.png",
    "sembukarai": "sembukarai.png",
    "dhoomanur":  None,   # Screenshot not yet available
    "thuvaipathy": None,  # Screenshot not yet available
}


# ---- Core cutout generation -----------------------------------------------

def make_mask(image_size: tuple, points: list) -> Image.Image:
    """Return an 'L' mode mask image: white inside polygon, black outside."""
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    flat = [(int(round(x)), int(round(y))) for x, y in points]
    draw.polygon(flat, fill=255)
    return mask


def make_cutout(orig_img: Image.Image, mask: Image.Image) -> Image.Image:
    """Return RGBA image: original pixels inside polygon, alpha=0 outside."""
    rgba = orig_img.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def make_outline_preview(orig_img: Image.Image, points: list,
                          line_color=(0, 220, 100), line_width=3) -> Image.Image:
    """Return RGB image with polygon perimeter drawn over original."""
    preview = orig_img.convert("RGB").copy()
    draw    = ImageDraw.Draw(preview)
    flat    = [(int(round(x)), int(round(y))) for x, y in points]
    # Draw closed polygon outline
    for i in range(len(flat)):
        a = flat[i]
        b = flat[(i + 1) % len(flat)]
        draw.line([a, b], fill=line_color, width=line_width)
    # Draw vertex dots
    r = 4
    for i, (px, py) in enumerate(flat):
        dot_color = (255, 80, 80) if i == 0 else (255, 220, 0)
        draw.ellipse([px - r, py - r, px + r, py + r],
                     fill=dot_color, outline="white")
    return preview


def make_checker_bg(w: int, h: int, cell: int = 16) -> Image.Image:
    """Dark checker for transparency visualisation (preview only)."""
    img  = Image.new("RGB", (w, h), (35, 35, 35))
    draw = ImageDraw.Draw(img)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1],
                                fill=(55, 55, 55))
    return img


def process_village(village: str, force: bool = False) -> dict:
    """
    Process one village: load annotation, generate outputs.
    Returns a result dict for validation.
    """
    result = {
        "village": village,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": None,
        "source_image": None,
        "annotation_exists": False,
        "vertex_count": 0,
        "outputs": {},
        "validation": {},
        "errors": []
    }

    # --- Check source image
    expected_src = ALL_VILLAGES.get(village)
    if expected_src is None:
        result["status"] = "SKIPPED_NO_SOURCE_IMAGE"
        result["errors"].append(f"No source screenshot available for {village}. Add the image to data/Seven Villages/.")
        return result

    img_path = os.path.join(INPUT_DIR, expected_src)
    result["source_image"] = expected_src
    if not os.path.exists(img_path):
        result["status"] = "ERROR_SOURCE_MISSING"
        result["errors"].append(f"Source image not found: {img_path}")
        return result

    # --- Check annotation
    ann_path = os.path.join(ANNOTATION_DIR, f"{village}_polygon.json")
    if not os.path.exists(ann_path):
        result["status"] = "AWAITING_ANNOTATION"
        result["errors"].append(
            f"No annotation found. Run: python scripts/annotate_village_polygons.py --village {village}"
        )
        return result

    result["annotation_exists"] = True
    try:
        with open(ann_path) as f:
            ann = json.load(f)
        points = ann.get("points", [])
    except Exception as e:
        result["status"] = "ERROR_ANNOTATION_PARSE"
        result["errors"].append(f"Could not parse annotation: {e}")
        return result

    if len(points) < 3:
        result["status"] = "ERROR_TOO_FEW_VERTICES"
        result["errors"].append(f"Annotation has {len(points)} vertices; need at least 3.")
        return result

    result["vertex_count"] = len(points)

    # --- Check whether regeneration is needed
    mask_path    = os.path.join(OUTPUT_DIR, f"{village}_polygon_mask.png")
    cutout_path  = os.path.join(OUTPUT_DIR, f"{village}_polygon_cutout.png")
    outline_path = os.path.join(OUTPUT_DIR, f"{village}_polygon_outline.png")
    preview_path = os.path.join(OUTPUT_DIR, f"{village}_polygon_preview.jpg")
    meta_path    = os.path.join(OUTPUT_DIR, f"{village}_polygon_metadata.json")

    all_exist = all(os.path.exists(p) for p in [mask_path, cutout_path, outline_path, preview_path])
    if all_exist and not force:
        print(f"  [{village}] Outputs already exist. Skipping (use --force to regenerate).")
        result["status"] = "OK_CACHED"
        result["outputs"] = {
            "mask": mask_path, "cutout": cutout_path,
            "outline": outline_path, "preview": preview_path,
            "metadata": meta_path
        }
        _validate_outputs(result, points, img_path)
        return result

    # --- Load source image
    try:
        orig_img = Image.open(img_path).convert("RGB")
        img_w, img_h = orig_img.size
    except Exception as e:
        result["status"] = "ERROR_IMAGE_OPEN"
        result["errors"].append(f"Could not open source image: {e}")
        return result

    # Validate annotation image size matches
    ann_size = ann.get("image_size", [img_w, img_h])
    if ann_size[0] != img_w or ann_size[1] != img_h:
        print(f"  [WARN] Annotation size {ann_size} != image size [{img_w},{img_h}]. Proceeding anyway.")

    print(f"  [{village}] Generating outputs ({img_w}x{img_h}, {len(points)} vertices)...")

    # 1. Mask
    mask = make_mask((img_w, img_h), points)
    mask.save(mask_path)

    # 2. RGBA Cutout
    cutout = make_cutout(orig_img, mask)
    cutout.save(cutout_path)

    # 3. Outline Preview
    outline = make_outline_preview(orig_img, points)
    outline.save(outline_path)

    # 4. Composite Preview (cutout over checker)
    checker = make_checker_bg(img_w, img_h)
    checker.paste(cutout, (0, 0), cutout)
    # Save as JPEG for compact storage; resize if large
    max_dim = 1200
    if max(img_w, img_h) > max_dim:
        scale = max_dim / max(img_w, img_h)
        checker = checker.resize((int(img_w * scale), int(img_h * scale)), Image.Resampling.LANCZOS)
    checker.save(preview_path, quality=92)

    # 5. Metadata
    metadata = {
        "village": village,
        "source_image": expected_src,
        "image_size": [img_w, img_h],
        "vertex_count": len(points),
        "annotation_file": ann_path,
        "annotation_method": ann.get("annotation_method", "manual_polygon"),
        "annotated_at": ann.get("annotated_at", "unknown"),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": {
            "mask": mask_path,
            "cutout": cutout_path,
            "outline": outline_path,
            "preview": preview_path
        }
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    result["outputs"] = {
        "mask": mask_path, "cutout": cutout_path,
        "outline": outline_path, "preview": preview_path,
        "metadata": meta_path
    }
    result["status"] = "OK"
    _validate_outputs(result, points, img_path)
    print(f"  [{village}] Done.")
    return result


def _validate_outputs(result: dict, points: list, img_path: str):
    """Fill result['validation'] with actual checks on generated files."""
    v = {}
    outputs = result.get("outputs", {})

    # Source exists
    v["source_exists"] = os.path.exists(img_path)

    # Annotation
    v["annotation_exists"] = result.get("annotation_exists", False)
    v["vertex_count_ok"]   = result.get("vertex_count", 0) >= 3

    # Mask
    mask_path = outputs.get("mask")
    if mask_path and os.path.exists(mask_path):
        v["mask_exists"] = True
        mask = Image.open(mask_path).convert("L")
        arr  = list(mask.getdata())
        has_fg = any(p > 128 for p in arr)
        has_bg = any(p < 128 for p in arr)
        v["mask_has_foreground"] = has_fg
        v["mask_has_background"] = has_bg
        v["mask_not_empty"]      = has_fg
        v["mask_not_full"]       = has_bg
    else:
        v["mask_exists"] = False

    # Cutout
    cutout_path = outputs.get("cutout")
    if cutout_path and os.path.exists(cutout_path):
        v["cutout_exists"] = True
        cutout = Image.open(cutout_path)
        v["cutout_is_rgba"] = (cutout.mode == "RGBA")
        if cutout.mode == "RGBA":
            alpha_data = list(cutout.split()[3].getdata())
            v["cutout_transparent_outside"] = any(a == 0 for a in alpha_data)
            v["cutout_opaque_inside"]        = any(a == 255 for a in alpha_data)
    else:
        v["cutout_exists"] = False

    # Outline
    outline_path = outputs.get("outline")
    if outline_path and os.path.exists(outline_path):
        v["outline_exists"] = True
        outline = Image.open(outline_path)
        v["outline_nonempty"] = (outline.width > 0 and outline.height > 0)
    else:
        v["outline_exists"] = False

    # Preview
    preview_path = outputs.get("preview")
    v["preview_exists"] = preview_path is not None and os.path.exists(preview_path)

    result["validation"] = v


# ---- Comparison sheet -------------------------------------------------------

def generate_comparison_sheet(results: list):
    """
    Builds data/village_polygon_cutouts/outputs/village_polygon_comparison.png
    Columns per row: Original | Mask | Cutout-on-checker | Outline Preview
    Only includes villages that have been processed (status OK or OK_CACHED).
    """
    from PIL import ImageFont

    completed = [r for r in results if r["status"] in ("OK", "OK_CACHED")]
    if not completed:
        print("[WARN] No completed villages to include in comparison sheet.")
        return

    THUMB_W  = 400
    THUMB_H  = 225
    PAD      = 15
    HEADER_H = 50
    LABEL_H  = 28
    COLS     = 4

    total_w = PAD + COLS * (THUMB_W + PAD)
    total_h = HEADER_H + len(completed) * (LABEL_H + THUMB_H + PAD) + PAD

    canvas = Image.new("RGB", (total_w, total_h), (22, 24, 28))
    draw   = ImageDraw.Draw(canvas)

    # Try to load a real font; fall back to default
    try:
        font_title  = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
        font_header = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 11)
        font_label  = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 13)
    except Exception:
        font_title  = ImageFont.load_default()
        font_header = font_title
        font_label  = font_title

    # Header banner
    draw.rectangle([0, 0, total_w, HEADER_H], fill=(12, 17, 34))
    draw.text((PAD, 10), "MACONFLIC - Seven-Village Satellite Polygon Cutout Sheet",
              fill=(255, 255, 255), font=font_title)
    col_labels = ["Original Screenshot", "Polygon Mask", "Transparent Cutout", "Outline Preview"]
    col_x = [PAD + i * (THUMB_W + PAD) for i in range(COLS)]
    for i, lbl in enumerate(col_labels):
        draw.text((col_x[i] + THUMB_W // 2 - 60, 32), lbl,
                  fill=(148, 163, 184), font=font_header)

    y_off = HEADER_H + PAD

    for r in completed:
        village  = r["village"]
        src_img  = r["source_image"]
        outputs  = r["outputs"]

        # Row label
        draw.text((PAD, y_off), f"{village.upper()}  |  {src_img}  |  {r['vertex_count']} vertices",
                  fill=(56, 189, 248), font=font_label)
        y_off += LABEL_H

        # Define the four thumbnail sources
        thumb_sources = [
            os.path.join(INPUT_DIR, src_img),
            outputs.get("mask"),
            outputs.get("preview"),
            outputs.get("outline"),
        ]

        for col_idx, src_path in enumerate(thumb_sources):
            x = PAD + col_idx * (THUMB_W + PAD)
            if src_path and os.path.exists(src_path):
                try:
                    img = Image.open(src_path).convert("RGB")
                    thumb = img.resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
                    canvas.paste(thumb, (x, y_off))
                except Exception as e:
                    draw.rectangle([x, y_off, x + THUMB_W, y_off + THUMB_H],
                                   fill=(40, 40, 40), outline=(80, 80, 80))
                    draw.text((x + 10, y_off + THUMB_H // 2), f"Error: {e}",
                              fill=(200, 80, 80), font=font_header)
            else:
                draw.rectangle([x, y_off, x + THUMB_W, y_off + THUMB_H],
                               fill=(40, 40, 40), outline=(80, 80, 80))
                draw.text((x + 10, y_off + THUMB_H // 2), "Missing",
                          fill=(150, 150, 150), font=font_header)

            # Border
            draw.rectangle([x, y_off, x + THUMB_W, y_off + THUMB_H],
                           outline=(60, 70, 90), width=1)

        y_off += THUMB_H + PAD

    sheet_path = os.path.join(OUTPUT_DIR, "village_polygon_comparison.png")
    canvas.save(sheet_path)
    print(f"[OK] Comparison sheet saved: {sheet_path}")
    return sheet_path


# ---- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MACONFLIC Village Polygon Cutout Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--village",
                        choices=list(ALL_VILLAGES.keys()),
                        default=None,
                        help="Process a single village only.")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if outputs already exist.")
    args = parser.parse_args()

    villages = [args.village] if args.village else list(ALL_VILLAGES.keys())

    print("=" * 60)
    print("MACONFLIC - Village Polygon Cutout Processor")
    print("=" * 60)

    all_results = []
    for v in villages:
        print(f"\nProcessing: {v}")
        res = process_village(v, force=args.force)
        all_results.append(res)
        status = res["status"]
        if res["errors"]:
            for err in res["errors"]:
                print(f"  [!] {err}")
        if status == "OK":
            print(f"  [OK] {v} completed successfully.")
        elif status == "OK_CACHED":
            print(f"  [OK] {v} outputs cached.")
        elif status == "AWAITING_ANNOTATION":
            print(f"  [--] {v} awaits polygon annotation.")
        elif status == "SKIPPED_NO_SOURCE_IMAGE":
            print(f"  [--] {v} has no source screenshot yet.")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['village']:15s} | {r['status']}")

    # Generate comparison sheet for completed ones
    print()
    generate_comparison_sheet(all_results)

    # Write validation report
    with open(VALIDATION_OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Validation report: {VALIDATION_OUT}")

    # Final counts
    ok_count     = sum(1 for r in all_results if r["status"] in ("OK", "OK_CACHED"))
    await_count  = sum(1 for r in all_results if r["status"] == "AWAITING_ANNOTATION")
    missing_count= sum(1 for r in all_results if r["status"] == "SKIPPED_NO_SOURCE_IMAGE")
    error_count  = sum(1 for r in all_results if r["status"].startswith("ERROR"))
    print(f"\nCompleted: {ok_count} | Awaiting annotation: {await_count} | "
          f"No source image: {missing_count} | Errors: {error_count}")

    if await_count > 0:
        print("\nTo annotate a village, run:")
        print("  python scripts/annotate_village_polygons.py --village <name>")
        print("Available village names:")
        for v, src in ALL_VILLAGES.items():
            if src and not os.path.exists(os.path.join(ANNOTATION_DIR, f"{v}_polygon.json")):
                print(f"    {v}")


if __name__ == "__main__":
    main()
