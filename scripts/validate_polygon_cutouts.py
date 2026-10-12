"""
MACONFLIC - Village Polygon Cutout Validator
=============================================
Reads existing output files and runs exhaustive checks.
Does NOT regenerate files -- for regeneration run:
    python scripts/generate_village_polygon_cutouts.py

Usage:
    python scripts/validate_polygon_cutouts.py
"""

import os
import sys
import json
from datetime import datetime

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow is required. Run: pip install Pillow")
    sys.exit(1)

REPO_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR      = os.path.join(REPO_ROOT, "data", "Seven Villages")
ANNOTATION_DIR = os.path.join(REPO_ROOT, "data", "village_polygon_cutouts", "annotations")
OUTPUT_DIR     = os.path.join(REPO_ROOT, "data", "village_polygon_cutouts", "outputs")
VALIDATION_OUT = os.path.join(REPO_ROOT, "results", "village_polygon_cutout_validation.json")

ALL_VILLAGES = {
    "kodanur":     "kodanur.png",
    "pudur":       "pudur.png",
    "panapalli":   "panapalli.png",
    "kandivazhi":  "kandivazhi.png",
    "sembukarai":  "sembukarai.png",
    "dhoomanur":   None,
    "thuvaipathy": None,
}


def validate_village(village: str) -> dict:
    result = {
        "village": village,
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "source_image": ALL_VILLAGES.get(village),
        "checks": {},
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "overall": None
    }
    checks = result["checks"]

    src = ALL_VILLAGES.get(village)
    if src is None:
        checks["source_image_available"] = "SKIPPED (no screenshot for this village)"
        result["overall"] = "INCOMPLETE_NO_SOURCE"
        result["skipped"] = 20
        return result

    img_path    = os.path.join(INPUT_DIR, src)
    ann_path    = os.path.join(ANNOTATION_DIR, f"{village}_polygon.json")
    mask_path   = os.path.join(OUTPUT_DIR, f"{village}_polygon_mask.png")
    cutout_path = os.path.join(OUTPUT_DIR, f"{village}_polygon_cutout.png")
    outline_path= os.path.join(OUTPUT_DIR, f"{village}_polygon_outline.png")
    preview_path= os.path.join(OUTPUT_DIR, f"{village}_polygon_preview.jpg")
    meta_path   = os.path.join(OUTPUT_DIR, f"{village}_polygon_metadata.json")

    def chk(name, value, message=""):
        status = "PASS" if value else "FAIL"
        checks[name] = status if not message else f"{status}: {message}"
        if value:
            result["passed"] += 1
        else:
            result["failed"] += 1

    # 1. Source image
    chk("source_exists", os.path.exists(img_path), img_path if not os.path.exists(img_path) else "")

    # 2-4. Annotation
    ann_ok = os.path.exists(ann_path)
    chk("annotation_exists", ann_ok, "" if ann_ok else f"Run annotator: python scripts/annotate_village_polygons.py --village {village}")

    points = []
    if ann_ok:
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            points = ann.get("points", [])
            chk("annotation_valid_json", True)
        except Exception as e:
            chk("annotation_valid_json", False, str(e))

        chk("annotation_has_3plus_vertices", len(points) >= 3,
            f"Only {len(points)} vertices" if len(points) < 3 else "")
    else:
        chk("annotation_valid_json", False, "file missing")
        chk("annotation_has_3plus_vertices", False, "file missing")

    # 5-9. Mask
    mask_exists = os.path.exists(mask_path)
    chk("mask_exists", mask_exists, "" if mask_exists else f"Run: python scripts/generate_village_polygon_cutouts.py --village {village}")

    if mask_exists and os.path.exists(img_path):
        try:
            orig  = Image.open(img_path)
            mask  = Image.open(mask_path).convert("L")
            chk("mask_correct_dimensions",
                mask.size == orig.size,
                f"mask {mask.size} != image {orig.size}" if mask.size != orig.size else "")
            arr = list(mask.getdata())
            has_fg = any(p > 128 for p in arr)
            has_bg = any(p < 128 for p in arr)
            chk("mask_has_foreground", has_fg)
            chk("mask_has_background", has_bg)
            chk("mask_not_empty",      has_fg)
            chk("mask_not_full_image", has_bg)
        except Exception as e:
            chk("mask_correct_dimensions", False, str(e))
            chk("mask_has_foreground",     False, str(e))
            chk("mask_has_background",     False, str(e))
            chk("mask_not_empty",          False, str(e))
            chk("mask_not_full_image",     False, str(e))
    else:
        for n in ["mask_correct_dimensions", "mask_has_foreground",
                  "mask_has_background", "mask_not_empty", "mask_not_full_image"]:
            checks[n] = "SKIP (mask or source missing)"
            result["skipped"] += 1

    # 10-13. Cutout
    cutout_exists = os.path.exists(cutout_path)
    chk("cutout_exists", cutout_exists)
    if cutout_exists:
        try:
            cutout = Image.open(cutout_path)
            chk("cutout_is_rgba", cutout.mode == "RGBA", f"mode={cutout.mode}")
            if cutout.mode == "RGBA":
                alpha_data = list(cutout.split()[3].getdata())
                chk("cutout_transparent_outside", any(a == 0 for a in alpha_data))
                chk("cutout_opaque_inside",        any(a == 255 for a in alpha_data))
            else:
                chk("cutout_transparent_outside", False, "not RGBA")
                chk("cutout_opaque_inside",        False, "not RGBA")
        except Exception as e:
            chk("cutout_is_rgba",            False, str(e))
            chk("cutout_transparent_outside",False, str(e))
            chk("cutout_opaque_inside",      False, str(e))
    else:
        for n in ["cutout_is_rgba", "cutout_transparent_outside", "cutout_opaque_inside"]:
            checks[n] = "SKIP (cutout missing)"
            result["skipped"] += 1

    # 14-15. Outline
    outline_exists = os.path.exists(outline_path)
    chk("outline_exists", outline_exists)
    if outline_exists:
        try:
            ol = Image.open(outline_path)
            chk("outline_nonempty", ol.width > 0 and ol.height > 0)
        except Exception as e:
            chk("outline_nonempty", False, str(e))
    else:
        checks["outline_nonempty"] = "SKIP (outline missing)"
        result["skipped"] += 1

    # 16. Preview
    chk("preview_exists", os.path.exists(preview_path))

    # 17. Metadata
    meta_exists = os.path.exists(meta_path)
    chk("metadata_exists", meta_exists)
    if meta_exists:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            chk("metadata_has_source_image", "source_image" in meta)
            chk("metadata_has_method",       "annotation_method" in meta)
        except Exception as e:
            chk("metadata_has_source_image", False, str(e))
            chk("metadata_has_method",       False, str(e))
    else:
        checks["metadata_has_source_image"] = "SKIP (metadata missing)"
        checks["metadata_has_method"]        = "SKIP (metadata missing)"
        result["skipped"] += 2

    # Overall verdict
    if result["failed"] == 0 and result["passed"] > 0:
        result["overall"] = "PASS"
    elif result["failed"] > 0 and result["passed"] > 0:
        result["overall"] = "PARTIAL"
    else:
        result["overall"] = "FAIL"

    return result


def main():
    print("=" * 60)
    print("MACONFLIC - Polygon Cutout Validator")
    print("=" * 60)

    all_results = []
    for village in ALL_VILLAGES:
        print(f"\nValidating: {village}")
        r = validate_village(village)
        all_results.append(r)
        print(f"  Overall: {r['overall']}  |  PASS={r['passed']}  FAIL={r['failed']}  SKIP={r['skipped']}")
        for name, status in r["checks"].items():
            icon = "[OK]" if str(status).startswith("PASS") else (
                   "[--]" if str(status).startswith("SKIP") else "[!!]")
            print(f"    {icon} {name}: {status}")

    os.makedirs(os.path.dirname(VALIDATION_OUT), exist_ok=True)
    with open(VALIDATION_OUT, "w") as f:
        json.dump(all_results, f, indent=2)

    total_pass   = sum(r["passed"] for r in all_results)
    total_fail   = sum(r["failed"] for r in all_results)
    total_skip   = sum(r["skipped"] for r in all_results)
    overall_pass = sum(1 for r in all_results if r["overall"] == "PASS")
    print(f"\n{'='*60}")
    print(f"Total checks: PASS={total_pass}  FAIL={total_fail}  SKIP={total_skip}")
    print(f"Villages fully passing: {overall_pass}/{len(ALL_VILLAGES)}")
    print(f"Report saved: {VALIDATION_OUT}")


if __name__ == "__main__":
    main()
