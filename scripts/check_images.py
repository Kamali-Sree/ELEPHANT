import json
import os
from PIL import Image

data_dir = r"c:\Users\hakki\Kamali-MA\ELEPHANT\data\Seven Villages"
images = ["kandivazhi.png", "kodanur.png", "panapalli.png", "pudur.png", "sembukarai.png"]

inventory = []

for img_name in images:
    img_path = os.path.join(data_dir, img_name)
    try:
        with Image.open(img_path) as img:
            info = {
                "filename": img_name,
                "format": img.format,
                "width": img.width,
                "height": img.height,
                "channels": len(img.getbands()),
                "exif": img.info.get("exif", None) is not None,
                "georeferenced": False # PNGs without world files/aux metadata usually aren't georeferenced
            }
            inventory.append(info)
    except Exception as e:
        print(f"Error reading {img_name}: {e}")

output_path = r"c:\Users\hakki\Kamali-MA\ELEPHANT\data\ml_inventory.json"
with open(output_path, "w") as f:
    json.dump(inventory, f, indent=4)

print("Inventory created.")
