# Settlement Outline Extraction & Georeferencing Pipeline Report
# MACONFLIC — Mitigating Human-Elephant Conflict Using AI & GIS

## 1. Executive Summary
This document records the visual inspection, feature characterization, segmentation parameters, and geographic alignment results for the target settlements in the Anaikatti region of the Western Ghats / Nilgiris corridor.

---

## 2. Source Image Characterization (Five Available Screenshots)

### 1. `kodanur.png` (Kondanur)
* **Image Dimensions / Channels:** 1362 x 775 pixels (3 channels, PNG) — Total: 1,055,550 pixels
* **Measured Built-up Foreground:** 15,051 pixels (1.426% of image)
* **Visual Content:**
  * Displays a broad agrarian valley nestled along mountain slopes.
  * Central junction contains the primary high-density settlement core (bright roofs, compact layout).
  * A secondary spur of homesteads runs along the western dirt road.
  * Terraced agricultural fields occupy the lowlands; dense deciduous canopy on mountain flanks.
* **Segmentation Focus:**
  * High-reflectance roof clusters and built-up patches concentrated around the central three-way road junction (`box_focus: [250, 150, 750, 520]`).
* **Georeferencing Status:**
  * **NOT GEOREFERENCED.** Needs at least 4 GCPs + 1 Check Point. Approximate center: `[76.7910, 11.1080]`.

---

### 2. `pudur.png` (Pudur)
* **Image Dimensions / Channels:** 927 x 586 pixels (3 channels, PNG) — Total: 543,222 pixels
* **Measured Built-up Foreground:** 4,208 pixels (0.775% of image)
* **Visual Content:**
  * Clearly captures an enclosed northern settlement enclave (farmhouses/residences organized within an oval ring road).
  * Secondary southern village cluster along the feeder road.
  * Flanked on east and west by dense forest cover and agricultural groves.
* **Segmentation Focus:**
  * High-contrast building footprints within the northern oval enclave and the southern road cluster (`box_focus: [380, 180, 560, 480]`).
* **Georeferencing Status:**
  * **NOT GEOREFERENCED.** Needs at least 4 GCPs + 1 Check Point. Approximate center: `[76.7700, 11.1300]`.

---

### 3. `panapalli.png` (Panapalli)
* **Image Dimensions / Channels:** 1350 x 777 pixels (3 channels, PNG) — Total: 1,048,950 pixels
* **Measured Built-up Foreground:** 1,233 pixels (0.1175% of image)
* **Segmentation Parameters:**
  * Brightness threshold: `>= 120`
  * Vegetation suppression ratio: `1.04`
  * Minimum cluster area: `80` pixels
  * Box focus: `(340, 310, 600, 490)`
* **Visual Content:**
  * Compact tribal village arranged almost linearly along a central mountain contour road.
  * Intricate terraced step-farming surrounds the hamlet on all slopes.
  * Heavy forest canopy visible to the south and northeast.
* **Segmentation Focus:**
  * Clustered residential structures lining the central winding road spine and adjoining dwellings.
* **Georeferencing Status:**
  * **NOT GEOREFERENCED.** Needs at least 4 GCPs + 1 Check Point. Center supported by WTI Elephant Corridor report (`[76.78808, 11.09305]`), but raster extent uncalibrated.

---

### 4. `kandivazhi.png` (Kandivalli / Kandivazhi)
* **Image Dimensions / Channels:** 1420 x 781 pixels (3 channels, PNG) — Total: 1,109,020 pixels
* **Measured Built-up Foreground:** 15,176 pixels (1.368% of image)
* **Visual Content:**
  * Multi-nucleus agrarian settlement.
  * Distinct western cluster on hillside, central farm buildings with open clearings, and eastern roadway hamlet.
* **Segmentation Focus:**
  * Western and eastern residential roof clusters (`box_focus: [100, 180, 950, 450]`).
* **Georeferencing Status:**
  * **NOT GEOREFERENCED.** Needs at least 4 GCPs + 1 Check Point. Approximate center: `[76.8000, 11.1167]`.

---

### 5. `sembukarai.png` (Sembukarai / Chembukarai)
* **Image Dimensions / Channels:** 1337 x 842 pixels (3 channels, PNG) — Total: 1,125,754 pixels
* **Measured Built-up Foreground:** 9,329 pixels (0.829% of image)
* **Visual Content:**
  * Broad landscape view containing major transportation corridor: **Thadagam Road / State Highway 164**.
  * Shows route markers (`164`), road bends, and roadside institutions/compounds.
  * Contains an embedded Esri watermark in the bottom-right corner: **`1:8,859 | Level 16`**.
* **Segmentation Focus:**
  * Roadside compounds, clusters along Thadagam Road, and agricultural clearings (`box_focus: [150, 80, 900, 500]`).
* **Georeferencing Status:**
  * **PARTIAL METADATA.** Zoom level (`Level 16`) and scale (`1:8,859`) provide pixel-to-ground scale (~2.38 m/px at lat 11°), but absolute bounding coordinates lack verified tie points. Needs at least 4 GCPs + 1 Check Point.

---

## 3. Measured Settlement Extraction Summary

| Settlement | Source File | Total Pixels | Built-up Pixels (Measured) | Area % | Calibration Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kondanur** | `kodanur.png` | 1,055,550 | 15,051 | 1.426% | Uncalibrated (0 GCPs) |
| **Pudur** | `pudur.png` | 543,222 | 4,208 | 0.775% | Uncalibrated (0 GCPs) |
| **Panapalli** | `panapalli.png` | 1,048,950 | 1,233 | 0.118% | Uncalibrated (0 GCPs) |
| **Kandivalli** | `kandivazhi.png` | 1,109,020 | 15,176 | 1.368% | Uncalibrated (0 GCPs) |
| **Sembukarai** | `sembukarai.png` | 1,125,754 | 9,329 | 0.829% | Partial Scale Only (0 GCPs) |

---

## 4. Settlements Without Source Imagery

### 6. Dhoomanur / Dhoomanoor
* **Source Image:** None in repository.
* **Status:** Ground truth coordinate unresolved (previously defaulted to generic Anaikatti centroid).
* **Policy:** No outline fabricated.

### 7. Thuvaipathy / Thuvaipathi
* **Source Image:** None in repository.
* **Status:** Verified point coordinate available (`[76.7800, 11.1200]`), but no satellite screenshot present.
* **Policy:** Point reference displayed; no outline fabricated.

---

## 5. Control Point (GCP) Collection & Georeferencing Methodology

### Ground Truth Alignment Protocol
To translate uncalibrated pixel masks into valid geographic polygons without introducing arbitrary distortions:
1. **Source Imagery Comparison:** Open the original PNG screenshots alongside Esri Wayback Imagery (or Google Earth Pro high-resolution satellite imagery) centered on the respective coordinates.
2. **Feature Identification (Tie Points):** Select at least 4 unambiguous, invariant geographic features per screenshot:
   - Sharp road intersections (T-junctions, cross-roads).
   - Distinct corners of permanent institutional or masonry compounds.
   - Fixed bridges, culverts, or stream confluence bends crossing roads.
3. **Record Coordinate Pairs:**
   - **Raster Pixel Coordinates:** `(u, v)` measured from top-left `(0, 0)` in image space.
   - **Geographic Coordinates:** `(lon, lat)` in WGS84 decimal degrees (EPSG:4326).
4. **Validation via Independent Check Points:** Designate at least 1 additional point per image exclusively as a validation Check Point (not used in matrix fitting).
5. **Execute Calibration Utility:** Run `scripts/georeference_settlement.py` to calculate the 2D affine transformation matrix and verify that Root Mean Square Error (RMSE) is within acceptable tolerance ($\le 15.0\text{ m}$).
6. **Polygon Export Safety Rule:** GeoJSON/Shapefile polygon conversion is strictly withheld until acceptable calibration error is mathematically demonstrated.
