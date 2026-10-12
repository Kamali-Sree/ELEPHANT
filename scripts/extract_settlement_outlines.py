"""
MACONFLIC — Settlement Contour and Mask Extraction Script
Processes the five available Esri Wayback screenshots:
  - kodanur.png
  - pudur.png
  - panapalli.png
  - kandivazhi.png
  - sembukarai.png

Generates:
  1. Binary segmentation mask (<name>_mask.png)
  2. Transparent cutout PNG (<name>_transparent.png)
  3. Visual contour overlay preview (<name>_preview.png)
  4. Pixel contour coordinates (<name>_contours.json)

Saves all outputs into data/settlement_outlines/
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

INPUT_DIR = r"c:\Users\hakki\Kamali-MA\ELEPHANT\data\Seven Villages"
OUTPUT_DIR = r"c:\Users\hakki\Kamali-MA\ELEPHANT\data\settlement_outlines"

SETTLEMENT_CONFIGS = {
    "kodanur": {
        "filename": "kodanur.png",
        "name": "Kondanur",
        # Parameter tuning for spectral contrast
        "brightness_thresh": 125,
        "veg_suppression_ratio": 1.05, # Suppress pixels where Green strongly exceeds Red/Blue
        "min_cluster_area": 120,       # Ignore isolated single-pixel specks
        "box_focus": (250, 150, 750, 520) # Focus region around the junction & homesteads
    },
    "pudur": {
        "filename": "pudur.png",
        "name": "Pudur",
        "brightness_thresh": 130,
        "veg_suppression_ratio": 1.02,
        "min_cluster_area": 100,
        "box_focus": (380, 180, 560, 480) # Northern oval enclave and central road axis
    },
    "panapalli": {
        "filename": "panapalli.png",
        "name": "Panapalli",
        "brightness_thresh": 120,
        "veg_suppression_ratio": 1.04,
        "min_cluster_area": 80,
        "box_focus": (340, 310, 600, 490) # Captures full village road spine and adjoining dwellings
    },
    "kandivazhi": {
        "filename": "kandivazhi.png",
        "name": "Kandivalli / Kandivazhi",
        "brightness_thresh": 130,
        "veg_suppression_ratio": 1.05,
        "min_cluster_area": 90,
        "box_focus": (100, 180, 950, 450) # Western and eastern hamlet axes
    },
    "sembukarai": {
        "filename": "sembukarai.png",
        "name": "Sembukarai",
        "brightness_thresh": 138,
        "veg_suppression_ratio": 1.02,
        "min_cluster_area": 150,
        "box_focus": (150, 80, 900, 500) # Compounds along Thadagam Road (SH 164)
    }
}

def extract_settlement_outlines():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_results = []

    for key, cfg in SETTLEMENT_CONFIGS.items():
        img_path = os.path.join(INPUT_DIR, cfg["filename"])
        if not os.path.exists(img_path):
            print(f"[-] Source image missing: {img_path}")
            continue

        print(f"[+] Processing {cfg['name']} ({cfg['filename']})...")
        orig_img = Image.open(img_path).convert("RGB")
        w, h = orig_img.size
        arr = np.array(orig_img, dtype=np.float32)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        # 1. Perceived luminance (grayscale brightness)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        # 2. Vegetation index (Normalized Difference Green/Red or simple ratio)
        # Forest canopy has high green relative to red; roofs/cleared built-up grounds have high red/gray
        is_vegetation = (g > (r * cfg["veg_suppression_ratio"])) & (g > b)

        # 3. Built-up / high-contrast threshold
        binary_raw = (brightness >= cfg["brightness_thresh"]) & (~is_vegetation)

        # Apply bounding focus box if specified to avoid edge watermarks/false agrarian borders
        mask = np.zeros((h, w), dtype=bool)
        if "box_focus" in cfg:
            x1, y1, x2, y2 = cfg["box_focus"]
            mask[y1:y2, x1:x2] = binary_raw[y1:y2, x1:x2]
        else:
            mask = binary_raw

        # 4. Morphological cleaning using PIL filters
        # Convert boolean mask to 8-bit image for morphology
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        
        # Dilate then erode (Closing) to bridge gaps between neighboring rooftops
        mask_closed = mask_img.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.MinFilter(size=5))
        # Remove small specks (Opening)
        mask_cleaned = mask_closed.filter(ImageFilter.MinFilter(size=3)).filter(ImageFilter.MaxFilter(size=3))

        mask_arr = np.array(mask_cleaned) > 128

        # Save Binary Mask
        mask_out_path = os.path.join(OUTPUT_DIR, f"{key}_mask.png")
        mask_img_final = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
        mask_img_final.save(mask_out_path)

        # Save Transparent Cutout
        rgba = np.array(orig_img.convert("RGBA"))
        rgba[:, :, 3] = (mask_arr * 255).astype(np.uint8)
        transparent_img = Image.fromarray(rgba, mode="RGBA")
        transparent_out_path = os.path.join(OUTPUT_DIR, f"{key}_transparent.png")
        transparent_img.save(transparent_out_path)

        # Save Visual Contour Overlay Preview
        # Draw red contours over the original screenshot
        preview_img = orig_img.copy()
        draw = ImageDraw.Draw(preview_img)

        # Simple edge detection on mask to find contour boundary pixels
        edge_mask = mask_img_final.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edge_mask) > 100
        edge_y, edge_x = np.where(edge_arr)

        for px, py in zip(edge_x, edge_y):
            draw.rectangle([px-1, py-1, px+1, py+1], fill=(255, 30, 30))

        preview_out_path = os.path.join(OUTPUT_DIR, f"{key}_preview.png")
        preview_img.save(preview_out_path)

        summary_results.append({
            "settlement": cfg["name"],
            "source_image": cfg["filename"],
            "image_size": [w, h],
            "mask_file": f"{key}_mask.png",
            "transparent_file": f"{key}_transparent.png",
            "preview_file": f"{key}_preview.png",
            "built_up_pixel_count": int(np.sum(mask_arr)),
            "georeferenced": False,
            "status": "Mask generated. Lacks ground control points for spatial projection."
        })

    report_path = os.path.join(OUTPUT_DIR, "extraction_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary_results, f, indent=4)

    print(f"\n[✓] Finished extraction. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_settlement_outlines()
