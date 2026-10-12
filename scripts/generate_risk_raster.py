import os
import numpy as np
from PIL import Image

def generate_heuristic_raster():
    # Define grid size
    width, height = 800, 800
    
    # Create an empty RGBA image
    raster = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Dummy village location in pixel coordinates
    # In reality, this would be computed from distance to rivers/forests/settlements
    village_x, village_y = 400, 400
    
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - village_x)**2 + (y - village_y)**2)
            
            # Simple distance decay for probability (max at center)
            prob = max(0, 1.0 - (dist / 300.0))
            
            # Color gradient: 
            # 0.0 - 0.2: Green (transparent)
            # 0.2 - 0.4: Yellow
            # 0.4 - 0.6: Orange
            # 0.6 - 0.8: Red
            # 0.8 - 1.0: Deep Red
            
            if prob < 0.2:
                r, g, b, a = 34, 197, 94, int(prob * 50)  # Greenish, highly transparent
            elif prob < 0.4:
                r, g, b, a = 234, 179, 8, int(prob * 100) # Yellow
            elif prob < 0.6:
                r, g, b, a = 249, 115, 22, int(prob * 150) # Orange
            elif prob < 0.8:
                r, g, b, a = 239, 68, 68, int(prob * 200) # Red
            else:
                r, g, b, a = 153, 27, 27, 220 # Deep red
                
            raster[y, x] = [r, g, b, a]
            
    img = Image.fromarray(raster)
    
    out_dir = r"c:\Users\hakki\Kamali-MA\ELEPHANT\web\frontend\public"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "maconflic-ml-prediction.png")
    
    img.save(out_path)
    print(f"Generated heuristic MCE raster at {out_path}")

if __name__ == "__main__":
    generate_heuristic_raster()
