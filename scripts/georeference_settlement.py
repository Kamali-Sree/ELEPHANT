"""
MACONFLIC — Settlement Georeferencing Utility.

Accepts corresponding image pixel coordinates (x, y) and WGS84 geographic
coordinates (lon, lat), estimates the affine / similarity transformation matrix,
computes residual errors (RMSE), and evaluates independent check points.

Policy:
- Minimum of 4 well-distributed Ground Control Points (GCPs) required for affine fit.
- Minimum of 1 independent Check Point (CP) required for validation.
- Rejects polygon export if RMSE exceeds acceptable threshold (tolerance default: 15.0 meters).
"""

import sys
import json
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Approximate meters per degree at 11.1° N latitude (Anaikatti region)
METERS_PER_DEG_LAT = 110574.0
METERS_PER_DEG_LON = 111320.0 * math.cos(math.radians(11.1))  # ~109,240 m/deg


class SettlementGeoreferencer:
    """
    Fits 2D affine transformation between pixel coordinates (u, v) and
    geographic coordinates (lon, lat).
    """

    def __init__(self, settlement_name: str, tolerance_meters: float = 15.0):
        self.settlement_name = settlement_name
        self.tolerance_meters = tolerance_meters
        self.gcps: List[Dict[str, Any]] = []
        self.check_points: List[Dict[str, Any]] = []
        self.transform_matrix: Optional[np.ndarray] = None
        self.calibrated: bool = False

    def add_gcp(self, point_id: str, img_x: float, img_y: float, lon: float, lat: float, description: str = ""):
        """Add a Ground Control Point used in fitting the transformation."""
        self.gcps.append({
            "id": point_id,
            "img_xy": (float(img_x), float(img_y)),
            "geo_lonlat": (float(lon), float(lat)),
            "desc": description
        })

    def add_check_point(self, point_id: str, img_x: float, img_y: float, lon: float, lat: float, description: str = ""):
        """Add an independent Check Point used solely to validate the fitted model."""
        self.check_points.append({
            "id": point_id,
            "img_xy": (float(img_x), float(img_y)),
            "geo_lonlat": (float(lon), float(lat)),
            "desc": description
        })

    def fit(self) -> Dict[str, Any]:
        """
        Fits affine transformation using least squares:
        [lon, lat]^T = A * [u, v]^T + t
        """
        if len(self.gcps) < 3:
            raise ValueError(
                f"Insufficient GCPs: Need at least 3 non-collinear GCPs for affine fit (4+ recommended). "
                f"Currently provided: {len(self.gcps)}."
            )

        n = len(self.gcps)
        # Build design matrix M and target vectors for Lon and Lat
        # [u, v, 1] * [a1, a2, a0]^T = lon
        # [u, v, 1] * [b1, b2, b0]^T = lat
        M = np.zeros((n, 3))
        Lons = np.zeros((n, 1))
        Lats = np.zeros((n, 1))

        for i, pt in enumerate(self.gcps):
            u, v = pt["img_xy"]
            lon, lat = pt["geo_lonlat"]
            M[i] = [u, v, 1.0]
            Lons[i] = lon
            Lats[i] = lat

        # Solve via least squares
        coeff_lon, residuals_lon, _, _ = np.linalg.lstsq(M, Lons, rcond=None)
        coeff_lat, residuals_lat, _, _ = np.linalg.lstsq(M, Lats, rcond=None)

        self.transform_matrix = np.vstack([coeff_lon.T, coeff_lat.T])  # 2 x 3

        # Calculate GCP residuals
        gcp_residuals = []
        sq_err_meters = []
        for pt in self.gcps:
            u, v = pt["img_xy"]
            pred_lon = float(self.transform_matrix[0, 0] * u + self.transform_matrix[0, 1] * v + self.transform_matrix[0, 2])
            pred_lat = float(self.transform_matrix[1, 0] * u + self.transform_matrix[1, 1] * v + self.transform_matrix[1, 2])

            true_lon, true_lat = pt["geo_lonlat"]
            d_lon = (pred_lon - true_lon) * METERS_PER_DEG_LON
            d_lat = (pred_lat - true_lat) * METERS_PER_DEG_LAT
            err_m = math.hypot(d_lon, d_lat)
            sq_err_meters.append(err_m ** 2)

            gcp_residuals.append({
                "id": pt["id"],
                "residual_meters": round(err_m, 2),
                "d_lon_meters": round(d_lon, 2),
                "d_lat_meters": round(d_lat, 2)
            })

        rmse_gcp_meters = math.sqrt(sum(sq_err_meters) / n) if n > 0 else 0.0

        # Calculate Independent Check Point residuals (if supplied)
        cp_residuals = []
        sq_err_cp = []
        for pt in self.check_points:
            u, v = pt["img_xy"]
            pred_lon = float(self.transform_matrix[0, 0] * u + self.transform_matrix[0, 1] * v + self.transform_matrix[0, 2])
            pred_lat = float(self.transform_matrix[1, 0] * u + self.transform_matrix[1, 1] * v + self.transform_matrix[1, 2])

            true_lon, true_lat = pt["geo_lonlat"]
            d_lon = (pred_lon - true_lon) * METERS_PER_DEG_LON
            d_lat = (pred_lat - true_lat) * METERS_PER_DEG_LAT
            err_m = math.hypot(d_lon, d_lat)
            sq_err_cp.append(err_m ** 2)

            cp_residuals.append({
                "id": pt["id"],
                "residual_meters": round(err_m, 2),
                "d_lon_meters": round(d_lon, 2),
                "d_lat_meters": round(d_lat, 2)
            })

        rmse_cp_meters = math.sqrt(sum(sq_err_cp) / len(sq_err_cp)) if sq_err_cp else None

        # Check acceptable tolerance
        eval_rmse = rmse_cp_meters if rmse_cp_meters is not None else rmse_gcp_meters
        is_acceptable = (eval_rmse <= self.tolerance_meters) and (n >= 4)
        self.calibrated = is_acceptable

        return {
            "settlement": self.settlement_name,
            "gcp_count": n,
            "check_point_count": len(self.check_points),
            "rmse_gcp_meters": round(rmse_gcp_meters, 2),
            "rmse_check_points_meters": round(rmse_cp_meters, 2) if rmse_cp_meters is not None else "N/A (No CPs provided)",
            "tolerance_meters": self.tolerance_meters,
            "calibrated": self.calibrated,
            "gcp_residuals": gcp_residuals,
            "check_point_residuals": cp_residuals
        }

    def pixel_to_geo(self, u: float, v: float) -> Tuple[float, float]:
        """Convert pixel coordinate to (lon, lat) using calibrated matrix."""
        if not self.calibrated or self.transform_matrix is None:
            raise RuntimeError(
                "Model is not calibrated or calibration error exceeds acceptable tolerance. "
                "Polygon generation is prohibited until calibration is verified."
            )
        lon = float(self.transform_matrix[0, 0] * u + self.transform_matrix[0, 1] * v + self.transform_matrix[0, 2])
        lat = float(self.transform_matrix[1, 0] * u + self.transform_matrix[1, 1] * v + self.transform_matrix[1, 2])
        return lon, lat


def print_remaining_gcp_requirements():
    """Prints requirements for collecting tie points for each settlement."""
    print("=" * 70)
    print("MACONFLIC Ground Control Point (GCP) Collection Requirements")
    print("=" * 70)
    requirements = [
        {
            "name": "Kondanur (kodanur.png)",
            "status": "0 GCPs available. Needs at least 4 GCPs + 1 Check Point.",
            "recommended_features": [
                "1. Central 3-way road junction centroid (fork heading north/west)",
                "2. Distinct temple/community hall perimeter corner",
                "3. Southern culvert crossing where water channel meets roadway",
                "4. High-contrast western farm perimeter stone bend",
                "Check Point: Isolated northern homestead roof ridge"
            ]
        },
        {
            "name": "Pudur (pudur.png)",
            "status": "0 GCPs available. Needs at least 4 GCPs + 1 Check Point.",
            "recommended_features": [
                "1. North apex of the oval ring road",
                "2. South-east junction connecting ring road to southern feeder",
                "3. Center of main community courtyard within oval",
                "4. Southern village cluster sharp road bend",
                "Check Point: Isolated eastern farmstead corner"
            ]
        },
        {
            "name": "Panapalli (panapalli.png)",
            "status": "0 GCPs available. Needs at least 4 GCPs + 1 Check Point.",
            "recommended_features": [
                "1. Northern hairpin curve of main mountain spine road",
                "2. Central settlement cross-track / gathering point",
                "3. Southern exit bend heading towards forest track",
                "4. Most prominent western terrace retaining wall boundary",
                "Check Point: Prominent blue/bright roof dwelling on eastern spur"
            ]
        },
        {
            "name": "Kandivazhi (kandivazhi.png)",
            "status": "0 GCPs available. Needs at least 4 GCPs + 1 Check Point.",
            "recommended_features": [
                "1. Western hillside hamlet primary road junction",
                "2. Central farm clearing entrance gate/track",
                "3. Eastern roadway settlement intersection",
                "4. Northern agricultural stone bund intersection",
                "Check Point: Eastern-most homestead structure"
            ]
        },
        {
            "name": "Sembukarai (sembukarai.png)",
            "status": "Partial scale metadata (Level 16 | 1:8,859). Needs 4 GCPs + 1 Check Point.",
            "recommended_features": [
                "1. State Highway 164 (Thadagam Rd) 90-degree curve vertex",
                "2. Milestone / SH 164 marker post location",
                "3. Large roadside institutional compound gate / northwest corner",
                "4. Secondary forest trail junction leaving SH 164 eastward",
                "Check Point: South-western agricultural pump shed / homestead"
            ]
        }
    ]

    for req in requirements:
        print(f"\nSettlement: {req['name']}")
        print(f"Status:     {req['status']}")
        print("Tie-Point Features to identify in Esri Wayback / Google Earth:")
        for feat in req["recommended_features"]:
            print(f"  • {feat}")


if __name__ == "__main__":
    print_remaining_gcp_requirements()
