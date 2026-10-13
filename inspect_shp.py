import sys
try:
    import geopandas as gpd
    print("Geopandas found.")
    gdf = gpd.read_file('data/TAMILNADU.shp')
    print("CRS:", gdf.crs)
    print("Geometry type:", gdf.geom_type.unique())
    print("Columns:", gdf.columns.tolist())
    
    # Check for relevant districts/villages
    if 'DISTRICT' in gdf.columns:
        cbe = gdf[gdf['DISTRICT'].str.contains('Coimbatore', case=False, na=False)]
        print(f"Found {len(cbe)} Coimbatore features.")
    
    # Check for ANAIKATTI
    anaikatti = gdf[gdf.apply(lambda row: row.astype(str).str.contains('ANAIKATTI', case=False).any(), axis=1)]
    print(f"Found {len(anaikatti)} ANAIKATTI features.")
    for idx, row in anaikatti.iterrows():
        print(row.to_dict())
        
except ImportError:
    print("Geopandas not found. Trying to parse DBF manually...")
    # Very basic DBF parser or just strings
    with open('data/TAMILNADU.dbf', 'rb') as f:
        content = f.read(2000).decode('ascii', errors='ignore')
        print("DBF snippet:", content)
