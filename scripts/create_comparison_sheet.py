"""
MACONFLIC — Settlement Comparison Sheet Generator
Creates a single consolidated comparison image:
data/settlement_outlines/settlement_comparison.png

Columns per settlement:
1. Original Satellite Screenshot
2. Extracted Binary Mask
3. Contour Overlay Preview
"""

import os
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = r"c:\Users\hakki\Kamali-MA\ELEPHANT\data\Seven Villages"
OUTLINES_DIR = r"c:\Users\hakki\Kamali-MA\ELEPHANT\data\settlement_outlines"
OUTPUT_SHEET = os.path.join(OUTLINES_DIR, "settlement_comparison.png")

SETTLEMENTS = [
    {"name": "Kondanur", "raw": "kodanur.png", "mask": "kodanur_mask.png", "prev": "kodanur_preview.png"},
    {"name": "Pudur", "raw": "pudur.png", "mask": "pudur_mask.png", "prev": "pudur_preview.png"},
    {"name": "Panapalli", "raw": "panapalli.png", "mask": "panapalli_mask.png", "prev": "panapalli_preview.png"},
    {"name": "Kandivalli / Kandivazhi", "raw": "kandivazhi.png", "mask": "kandivazhi_mask.png", "prev": "kandivazhi_preview.png"},
    {"name": "Sembukarai", "raw": "sembukarai.png", "mask": "sembukarai_mask.png", "prev": "sembukarai_preview.png"}
]

def generate_comparison_sheet():
    row_thumb_w = 480
    row_thumb_h = 270
    padding = 20
    header_h = 60
    row_label_h = 35

    cols = 3 # Raw, Mask, Preview
    total_w = padding + (cols * (row_thumb_w + padding))
    total_h = header_h + len(SETTLEMENTS) * (row_label_h + row_thumb_h + padding) + padding

    canvas = Image.new("RGB", (total_w, total_h), color=(26, 28, 30))
    draw = ImageDraw.Draw(canvas)

    # Title Banner
    draw.rectangle([0, 0, total_w, header_h], fill=(15, 23, 42))
    draw.text((padding, 18), "MACONFLIC - Seven-Village Settlement Outline Validation Sheet", fill=(255, 255, 255))
    draw.text((total_w - 420, 22), "Original (Left) | Binary Mask (Center) | Contour Overlay (Right)", fill=(148, 163, 184))

    y_offset = header_h + padding

    for item in SETTLEMENTS:
        # Row label
        draw.text((padding, y_offset), f"Settlement: {item['name']}", fill=(56, 189, 248))
        draw.text((padding + 300, y_offset), f"Source: {item['raw']}", fill=(156, 163, 175))
        y_offset += row_label_h

        # 1. Raw screenshot
        raw_path = os.path.join(DATA_DIR, item["raw"])
        raw_img = Image.open(raw_path).convert("RGB")
        raw_thumb = raw_img.resize((row_thumb_w, row_thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(raw_thumb, (padding, y_offset))

        # 2. Binary mask
        mask_path = os.path.join(OUTLINES_DIR, item["mask"])
        mask_img = Image.open(mask_path).convert("RGB")
        mask_thumb = mask_img.resize((row_thumb_w, row_thumb_h), Image.Resampling.NEAREST)
        canvas.paste(mask_thumb, (padding + row_thumb_w + padding, y_offset))

        # 3. Contour preview
        prev_path = os.path.join(OUTLINES_DIR, item["prev"])
        prev_img = Image.open(prev_path).convert("RGB")
        prev_thumb = prev_img.resize((row_thumb_w, row_thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(prev_thumb, (padding + 2 * (row_thumb_w + padding), y_offset))

        # Border outlines
        for i in range(cols):
            x = padding + i * (row_thumb_w + padding)
            draw.rectangle([x, y_offset, x + row_thumb_w, y_offset + row_thumb_h], outline=(71, 85, 105), width=1)

        y_offset += row_thumb_h + padding

    canvas.save(OUTPUT_SHEET, quality=95)
    print(f"[OK] Comparison sheet generated successfully at: {OUTPUT_SHEET}")

if __name__ == "__main__":
    generate_comparison_sheet()
