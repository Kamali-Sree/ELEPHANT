import os
import json
import numpy as np
from PIL import Image, ImageFilter

def generate_red_corridor():
    # Load historical observations
    obs_path = r"c:\Users\hakki\Kamali-MA\ELEPHANT\data\elephant_observations.geojson"
    with open(obs_path, 'r') as f:
        data = json.load(f)
        
    points = []
    # Bounds for Anaikatti (approx)
    min_lat, max_lat = 10.6, 11.2
    min_lon, max_lon = 76.5, 77.1
    
    width, height = 800, 800
    
    # Raster map
    # We will build a continuous heat/connectivity map
    # 0 = transparent, 255 = solid red
    intensity = np.zeros((height, width), dtype=np.float32)
    
    for feature in data['features']:
        coords = feature['geometry']['coordinates']
        lon, lat = coords[0], coords[1]
        
        # Convert to pixel space
        px = int(((lon - min_lon) / (max_lon - min_lon)) * width)
        py = int((1.0 - (lat - min_lat) / (max_lat - min_lat)) * height)
        
        if 0 <= px < width and 0 <= py < height:
            points.append((px, py))
            intensity[py, px] = 1.0

    # If no points, just create an empty raster
    if len(points) == 0:
        print("No valid points in bounding box.")
    else:
        # Create a connectivity corridor via morphological dilation / gaussian blur
        # To simulate a natural corridor, we use a large gaussian filter to connect nearby points
        # Then we apply a threshold to make it a continuous soft blob
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(intensity, sigma=15)
        
        # Normalize and create the red channel
        max_val = np.max(blurred)
        if max_val > 0:
            blurred = blurred / max_val
            
    # Apply thresholding to create an irregular corridor shape rather than a generic heatmap
    # Opacity 25-45% for the corridor
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            val = blurred[y, x]
            if val > 0.05: # Cutoff for the corridor boundary
                alpha = int(min(0.45, (val * 2.0)) * 255) # max 45% opacity
                rgba[y, x] = [255, 0, 0, alpha] # RED only
                
    img = Image.fromarray(rgba)
    
    out_dir = r"c:\Users\hakki\Kamali-MA\ELEPHANT\web\frontend\public"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "maconflic-red-corridor.png")
    img.save(out_path)
    
    print(f"Generated RED potential elephant path corridor at {out_path}")
    
    # Generate the report
    report = {
        "historical_observation_count": len(points),
        "source": "maconflic.db incidents table",
        "spatial_extent": {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon},
        "algorithm": "Heuristic Spatial Connectivity (Gaussian KDE proxy)",
        "environmental_features_used": ["distance to historical observations"],
        "is_ml_prediction": False,
        "is_spatial_heuristic": True,
        "validation_status": "Awaiting georeferenced environmental variables for true supervised ML.",
        "missing_datasets": ["GPS collar tracks", "Forest cover boundary masks", "DEM/Elevation"]
    }
    
    report_dir = r"c:\Users\hakki\Kamali-MA\ELEPHANT\reports"
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "elephant_path_model_report.json"), "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    generate_red_corridor()
