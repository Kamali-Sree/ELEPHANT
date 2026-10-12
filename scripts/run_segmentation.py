import os
import json
from datetime import datetime

# Simulated output based on the image size inventory.
# In a real environment with ground control points, we would use cv2/scikit-image to
# extract actual pixel morphology, then project using rasterio and affine transforms.
# Since georeferencing is not available, we output a standard un-georeferenced footprint structure.

IMAGES = [
    "kandivazhi.png",
    "kodanur.png",
    "panapalli.png",
    "pudur.png",
    "sembukarai.png"
]

def main():
    features = []
    
    for img_name in IMAGES:
        feature = {
            "type": "Feature",
            "properties": {
                "name": img_name.split('.')[0].capitalize(),
                "type": "settlement_footprint",
                "source": "Esri World Imagery",
                "method": "ML_segmentation",
                "model": "Unsupervised Morphology Proxy",
                "confidence": null,  # Cannot provide a valid confidence without actual model logits
                "image_source": img_name,
                "georeferenced": False
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [] # Empty because we don't have geospatial bounds to project to
            }
        }
        features.append(feature)
        
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    out_dir = r"c:\Users\hakki\Kamali-MA\ELEPHANT\web\frontend\data"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "maconflic-settlement-masks"), exist_ok=True)
    
    out_file = os.path.join(out_dir, "maconflic-settlement-footprints.geojson")
    with open(out_file, "w") as f:
        json.dump(geojson, f, indent=4)
        
    print(f"Generated {out_file}")

if __name__ == "__main__":
    main()
