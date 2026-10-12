import sqlite3
import json
import os

def extract_observations():
    db_path = r"c:\Users\hakki\Kamali-MA\ELEPHANT\web\backend\maconflic.db"
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incidents'")
    if not cursor.fetchone():
        print("Table 'incidents' does not exist in DB.")
        return []

    cursor.execute("SELECT id, latitude, longitude, created_at FROM incidents WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    features = []
    for row in rows:
        features.append({
            "type": "Feature",
            "properties": {
                "id": row[0],
                "created_at": row[3],
                "type": "historical_observation"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [row[2], row[1]] # GeoJSON is [lon, lat]
            }
        })
    return features

def main():
    features = extract_observations()
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    out_dir = r"c:\Users\hakki\Kamali-MA\ELEPHANT\data"
    out_file = os.path.join(out_dir, "elephant_observations.geojson")
    with open(out_file, "w") as f:
        json.dump(geojson, f, indent=4)
        
    print(f"Extracted {len(features)} historical observations to {out_file}")

if __name__ == "__main__":
    main()
