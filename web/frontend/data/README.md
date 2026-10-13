# MACONFLIC GIS Data

## MACONFLIC Study Area

**Location:**  
Anakatty, Tamil Nadu, India

**Boundary type:**  
Project-defined approximate GIS boundary

**Source:**  
Project-generated GeoJSON for prototype demonstration.

**Purpose:**  
Research visualization and MACONFLIC risk mapping.

**Disclaimer:**  
This polygon is an approximate MACONFLIC research study area and should NOT be interpreted as an official administrative, forest, protected-area, or elephant-corridor boundary. It serves strictly as a visualization bounding region for the Satellite Risk Map module.

---

## Remote Sensing Data

**1. Source:** Microsoft Planetary Computer
**2. API / Catalog:** STAC (SpatioTemporal Asset Catalog) API
**3. Collection:** Sentinel-2 Level-2A (`sentinel-2-l2a`)
**4. Study-area Source:** Bound by the coordinates [76.680, 11.080, 76.780, 11.150] (Anaikatti Region)
**5. Search Method:** Dynamic STAC query for recent imagery intersecting the bounding box.
**6. Cloud Filtering:** Progressively relaxes from 20% to 100% until a suitable image is found.
**7. RGB Bands:** B04 (Red), B03 (Green), B02 (Blue) - Rendered as Natural Color.
**8. False-Color Bands:** B08 (Near-Infrared), B04 (Red), B03 (Green) - Used to highlight dense vegetation (which appears red/magenta).
**9. NDVI Formula:** `(B08 - B04) / (B08 + B04)` - Normalized Difference Vegetation Index.

### Important Scientific Disclaimer:
Sentinel-2 imagery provides environmental and spectral information. It does not directly detect elephants or determine human-elephant conflict risk. The MACONFLIC risk overlays (Danger Zones in Red/Orange/Yellow) are entirely separate project-defined polygons and risk logic. Sentinel-2 false colors or NDVI values do NOT represent elephant presence.
