"use client";

import { useEffect, useState } from "react";
import { MapContainer, TileLayer, WMSTileLayer, CircleMarker, Circle, Popup, LayersControl, LayerGroup, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import studyAreaData from "@/data/maconflic-study-area.json";
import dangerZonesData from "@/data/maconflic-danger-zones.json";
import settlementsData from "@/data/maconflic-settlements.json";

function FitBounds({ geojsonData }) {
  const map = useMap();
  useEffect(() => {
    if (geojsonData) {
      const layer = L.geoJSON(geojsonData);
      map.fitBounds(layer.getBounds(), { padding: [20, 20] });
    }
  }, [map, geojsonData]);
  return null;
}

const { BaseLayer, Overlay } = LayersControl;

const SEVERITY_COLORS = {
  low: "#22c55e",
  medium: "#f59e0b",
  high: "#f97316",
  critical: "#ef4444",
};

const CALL_TYPE_ICONS = {
  Roar: "🦁",
  Rumble: "🔊",
  Trumpet: "🎺",
};

export default function SatelliteMapComponent({ geojson, alerts = [], sensors = [], mapTilerKey, activeSentinelLayer }) {
  const center = [11.111, 76.733]; // Anakatty, Tamil Nadu
  const zoom = 12;

  const [sentinelData, setSentinelData] = useState(null);

  useEffect(() => {
    async function fetchSentinel() {
      try {
        const res = await fetch("http://localhost:8000/api/sentinel/search");
        const data = await res.json();
        if (data.success) {
          setSentinelData(data);
        }
      } catch (err) {
        console.error("Failed to fetch Sentinel data", err);
      }
    }
    fetchSentinel();
  }, []);

  if (!mapTilerKey) {
    return (
      <div style={{ height: "600px", display: "flex", justifyContent: "center", alignItems: "center", background: "#111", color: "#ef4444", borderRadius: "12px", border: "1px solid #333" }}>
        <div style={{ textAlign: "center" }}>
          <h2 style={{ marginBottom: "1rem" }}>🛰️ Satellite Imagery Unavailable</h2>
          <p>Please configure NEXT_PUBLIC_MAPTILER_API_KEY in your .env.local file.</p>
        </div>
      </div>
    );
  }
  
  return (
    <div style={{ position: "relative", width: "100%", height: "700px", borderRadius: "var(--radius-lg)", overflow: "hidden" }}>
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: "100%", width: "100%", zIndex: 1 }}
      >
      <FitBounds geojsonData={studyAreaData} />
      <LayersControl position="topright">
        {/* BASEMAPS */}
        <BaseLayer checked name="Natural Satellite (MapTiler)">
          <TileLayer
            url={`https://api.maptiler.com/maps/satellite/{z}/{x}/{y}.jpg?key=${mapTilerKey}`}
            attribution='&copy; <a href="https://www.maptiler.com/">MapTiler</a>'
          />
        </BaseLayer>

        {/* STUDY AREA */}
        <Overlay checked name="Study Area">
          <LayerGroup>
            <GeoJSON 
              data={studyAreaData} 
              style={{ color: "#ffffff", weight: 2, fillOpacity: 0.05, dashArray: "5, 10" }}
            >
              <Popup>MACONFLIC Study Area: Anakatty</Popup>
            </GeoJSON>
          </LayerGroup>
        </Overlay>

        {/* DANGER SPOTS */}
        <Overlay checked name="MACONFLIC Danger Spots">
          <LayerGroup>
            <GeoJSON 
              data={dangerZonesData} 
              style={(feature) => {
                const risk = feature.properties.risk_level?.toLowerCase();
                const color = risk === 'high' ? '#ef4444' : risk === 'medium' ? '#f97316' : '#facc15';
                const opacity = risk === 'high' ? 0.35 : risk === 'medium' ? 0.25 : 0.20;
                return {
                  color: color,
                  weight: 2,
                  fillColor: color,
                  fillOpacity: opacity,
                  dashArray: "3, 6"
                };
              }}
              onEachFeature={(feature, layer) => {
                if (feature.properties) {
                  layer.bindPopup(`
                    <div style="font-family: Inter, sans-serif; font-size: 0.85rem; min-width: 200px">
                      <div style="font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem">${feature.properties.village}</div>
                      <div style="background: var(--bg-secondary); padding: 0.5rem; border-radius: var(--radius-sm); margin-bottom: 0.5rem">
                        <div><strong>Danger Spot:</strong> ${feature.properties.id}</div>
                        <div><strong>Risk:</strong> ${feature.properties.risk_level}</div>
                        <div><strong>Type:</strong> ${feature.properties.zone_type}</div>
                        <div><strong>Source:</strong> ${feature.properties.source}</div>
                        <div style="margin-top: 0.5rem"><strong>Description:</strong> ${feature.properties.description}</div>
                      </div>
                    </div>
                  `);
                }
              }}
            />
          </LayerGroup>
        </Overlay>

        {/* SETTLEMENTS */}
        <Overlay checked name="MACONFLIC Settlements">
          <LayerGroup>
            {settlementsData?.features?.map((feature) => {
              if (!feature.geometry || !feature.geometry.coordinates) return null;
              const props = feature.properties;
              return (
                <CircleMarker
                  key={`settlement-${props.id}`}
                  center={[feature.geometry.coordinates[1], feature.geometry.coordinates[0]]}
                  radius={6}
                  pathOptions={{
                    color: "#ffffff",
                    weight: 2,
                    fillColor: "#3b82f6",
                    fillOpacity: 1
                  }}
                >
                  <Popup>
                    <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.85rem", minWidth: "200px" }}>
                      <div style={{ fontWeight: 700, fontSize: "1rem", color: "#3b82f6", marginBottom: "0.5rem" }}>
                        🛖 {props.name}
                      </div>
                      <div style={{ background: "var(--bg-secondary)", padding: "0.5rem", borderRadius: "var(--radius-sm)", marginBottom: "0.5rem" }}>
                        <div><strong>Type:</strong> {props.settlement_type}</div>
                        <div><strong>Status:</strong> {props.status}</div>
                        <div style={{ marginTop: "0.5rem" }}><strong>Source:</strong> {props.source}</div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>{props.source_note}</div>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </LayerGroup>
        </Overlay>

        {/* RISK HEATMAP */}
        <Overlay checked name="Risk Heatmap">
          <LayerGroup>
            {geojson?.features?.map((feature) => {
              const props = feature.properties;
              if (!feature.geometry.coordinates || feature.geometry.coordinates[0] === null) return null;
              if (!props.risk_score) return null;
              
              // Normalize risk score for opacity and radius
              const intensityOpacity = (props.risk_score / 100) * 0.5;
              const heatRadius = 200 + (props.risk_score * 5); // meters
              const color = SEVERITY_COLORS[props.severity] || "#ef4444";
              
              return (
                <Circle
                  key={`heat-${props.id}`}
                  center={[feature.geometry.coordinates[1], feature.geometry.coordinates[0]]}
                  radius={heatRadius}
                  pathOptions={{
                    color: "transparent",
                    fillColor: color,
                    fillOpacity: intensityOpacity,
                    stroke: false
                  }}
                />
              );
            })}
          </LayerGroup>
        </Overlay>

        {/* RISK ZONES (Alerts) */}
        <Overlay checked name="Risk Zones">
          <LayerGroup>
            {alerts?.map((alert) => {
              if (!alert.latitude || !alert.longitude) return null;
              const color = SEVERITY_COLORS[alert.risk_level?.toLowerCase()] || "#ef4444";
              
              return (
                <Circle
                  key={`alert-${alert.id}`}
                  center={[alert.latitude, alert.longitude]}
                  radius={alert.radius_km * 1000} // Radius in meters
                  pathOptions={{
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.15,
                    weight: 2,
                    dashArray: "8, 8"
                  }}
                >
                  <Popup>
                    <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.85rem", minWidth: "200px" }}>
                      <div style={{ fontWeight: 700, fontSize: "1rem", color, marginBottom: "0.5rem" }}>
                        🚨 ALERT ZONE: {alert.risk_level}
                      </div>
                      <div style={{ background: "var(--bg-secondary)", padding: "0.5rem", borderRadius: "var(--radius-sm)", marginBottom: "0.5rem" }}>
                        <div><strong>Incident:</strong> #{alert.incident_id}</div>
                        <div><strong>Risk Score:</strong> {alert.risk_score}/100</div>
                        <div><strong>Radius:</strong> {alert.radius_km} km</div>
                        <div><strong>Status:</strong> {alert.status}</div>
                      </div>
                    </div>
                  </Popup>
                </Circle>
              );
            })}
          </LayerGroup>
        </Overlay>

        {/* INCIDENTS */}
        <Overlay checked name="Incidents">
          <LayerGroup>
            {geojson?.features?.map((feature) => {
              const { coordinates } = feature.geometry;
              if (!coordinates || coordinates[0] === null) return null;
              const props = feature.properties;
              const color = SEVERITY_COLORS[props.severity] || "#6b7280";
              const radius = props.severity === "critical" ? 8 : props.severity === "high" ? 6 : 5;

              return (
                <CircleMarker
                  key={`incident-${props.id}`}
                  center={[coordinates[1], coordinates[0]]}
                  radius={radius}
                  pathOptions={{ color, fillColor: color, fillOpacity: 0.8, weight: 2 }}
                >
                  <Popup>
                    <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.85rem", minWidth: "220px" }}>
                      <div style={{ fontWeight: 700, fontSize: "1rem", marginBottom: "0.5rem" }}>
                        ELEPHANT INCIDENT
                      </div>
                      <div><strong>Call Type:</strong> {CALL_TYPE_ICONS[props.call_type] || "🔈"} {props.call_type}</div>
                      <div><strong>Confidence:</strong> {props.confidence ? `${(props.confidence * 100).toFixed(0)}%` : "—"}</div>
                      <div style={{ margin: "0.5rem 0", padding: "0.5rem", background: "var(--bg-secondary)", borderRadius: "var(--radius-sm)" }}>
                        <div><strong>Risk Level:</strong> <span style={{ color, fontWeight: 600 }}>{props.severity?.toUpperCase()}</span></div>
                        <div><strong>Risk Score:</strong> {props.risk_score ? `${props.risk_score}/100` : "—"}</div>
                      </div>
                      <div><strong>Source:</strong> {props.source}</div>
                      <div style={{ color: "var(--text-secondary)", fontSize: "0.75rem", marginTop: "0.5rem" }}>
                        {new Date(props.created_at).toLocaleString()}
                      </div>
                      <div style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
                        {coordinates[1].toFixed(5)}, {coordinates[0].toFixed(5)}
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </LayerGroup>
        </Overlay>
        
        {/* SENSORS */}
        <Overlay checked name="Sensors">
          <LayerGroup>
            {sensors?.map((sensor, idx) => (
              <CircleMarker
                key={`sensor-${idx}`}
                center={[sensor.latitude, sensor.longitude]}
                radius={4}
                pathOptions={{ color: "#ffffff", fillColor: "#0ea5e9", fillOpacity: 1, weight: 2 }}
              >
                <Popup>🎙️ {sensor.name}</Popup>
              </CircleMarker>
            ))}
          </LayerGroup>
        </Overlay>
      </LayersControl>

      {/* ACTIVE SENTINEL-2 LAYER OVERRIDE (PRELOADED FOR INSTANT SWITCHING) */}
      {sentinelData && sentinelData.item_id && (
        <>
          <TileLayer
            url={`http://localhost:8000/api/sentinel/tiles/rgb/{z}/{x}/{y}.png`}
            attribution={`&copy; Microsoft Planetary Computer | Sentinel-2 L2A RGB`}
            maxZoom={16}
            zIndex={10}
            opacity={activeSentinelLayer === 'rgb' ? 1.0 : 0}
          />
          <TileLayer
            url={`http://localhost:8000/api/sentinel/tiles/false-color/{z}/{x}/{y}.png`}
            attribution={`&copy; Microsoft Planetary Computer | Sentinel-2 L2A FALSE COLOR`}
            maxZoom={16}
            zIndex={10}
            opacity={activeSentinelLayer === 'false-color' ? 1.0 : 0}
          />
          <TileLayer
            url={`http://localhost:8000/api/sentinel/tiles/ndvi/{z}/{x}/{y}.png`}
            attribution={`&copy; Microsoft Planetary Computer | Sentinel-2 L2A NDVI`}
            maxZoom={16}
            zIndex={10}
            opacity={activeSentinelLayer === 'ndvi' ? 0.8 : 0}
          />
        </>
      )}
      </MapContainer>

      {/* MAP LEGEND */}
      <div style={{
        position: "absolute",
        bottom: "20px",
        left: "20px",
        zIndex: 1000,
        background: "rgba(10, 10, 10, 0.85)",
        backdropFilter: "blur(10px)",
        border: "1px solid #333",
        borderRadius: "8px",
        padding: "12px",
        color: "#e5e7eb",
        fontFamily: "Inter, sans-serif",
        fontSize: "0.8rem",
        boxShadow: "0 4px 6px rgba(0,0,0,0.3)",
        maxWidth: "280px"
      }}>
        <div style={{ fontWeight: 700, marginBottom: "8px", color: "#fff", fontSize: "0.9rem" }}>MAP LEGEND</div>
        
        <div style={{ marginBottom: "8px" }}>
          {activeSentinelLayer === 'ndvi' && (
            <>
              <div style={{ fontWeight: 600, color: "#9ca3af", marginBottom: "2px", fontSize: "0.75rem" }}>SENTINEL-2 NDVI</div>
              <div style={{ lineHeight: "1.3", fontSize: "0.75rem" }}>
                <strong>NDVI:</strong> (B08 - B04) / (B08 + B04)<br/>
                Vegetation index raster.<br/>
                <span style={{ color: "#ef4444" }}>Red</span> = Low vegetation<br/>
                <span style={{ color: "#22c55e" }}>Green</span> = High vegetation
              </div>
            </>
          )}
          {activeSentinelLayer === 'false-color' && (
            <>
              <div style={{ fontWeight: 600, color: "#9ca3af", marginBottom: "2px", fontSize: "0.75rem" }}>SENTINEL-2 FALSE COLOR</div>
              <div style={{ lineHeight: "1.3", fontSize: "0.75rem" }}>
                <strong>FALSE COLOR:</strong> B08 / B04 / B03<br/>
                Multispectral visualization highlighting vegetation and surface characteristics.
              </div>
            </>
          )}
          {activeSentinelLayer === 'rgb' && (
            <>
              <div style={{ fontWeight: 600, color: "#9ca3af", marginBottom: "2px", fontSize: "0.75rem" }}>SENTINEL-2 RGB</div>
              <div style={{ lineHeight: "1.3", fontSize: "0.75rem" }}>
                <strong>RGB:</strong> B04 / B03 / B02<br/>
                Natural-color Sentinel-2 imagery.
              </div>
            </>
          )}
          {!activeSentinelLayer && (
            <>
              <div style={{ fontWeight: 600, color: "#9ca3af", marginBottom: "2px", fontSize: "0.75rem" }}>SENTINEL-2 REMOTE SENSING</div>
              <div style={{ lineHeight: "1.3", fontSize: "0.75rem", color: "#6b7280" }}>
                <em>No Sentinel-2 layer active. MapTiler basemap is showing.</em>
              </div>
            </>
          )}
        </div>
        
        <div style={{ marginBottom: "8px" }}>
          <div style={{ fontWeight: 600, color: "#9ca3af", marginBottom: "2px", fontSize: "0.75rem" }}>MACONFLIC RISK OVERLAY</div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}><span style={{ width: "12px", height: "12px", background: "#ef4444", display: "inline-block", opacity: 0.8 }}></span> RED = HIGH / CRITICAL risk</div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}><span style={{ width: "12px", height: "12px", background: "#f97316", display: "inline-block", opacity: 0.8 }}></span> ORANGE = MEDIUM / conflict zone</div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}><span style={{ width: "12px", height: "12px", background: "#facc15", display: "inline-block", opacity: 0.8 }}></span> YELLOW = WATCH / approach zone</div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}><span style={{ width: "12px", height: "12px", background: "#22c55e", display: "inline-block", opacity: 0.8 }}></span> GREEN = Lower/background</div>
        </div>

        <div style={{ marginBottom: "8px" }}>
          <div style={{ fontWeight: 600, color: "#9ca3af", marginBottom: "2px", fontSize: "0.75rem" }}>LAYERS</div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}><span style={{ width: "12px", height: "12px", borderRadius: "50%", background: "#3b82f6", display: "inline-block" }}></span> Settlement Point</div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}><span style={{ width: "12px", height: "12px", borderRadius: "50%", background: "#ef4444", display: "inline-block" }}></span> Live Incident</div>
        </div>

        {sentinelData && sentinelData.success && (
          <div style={{ marginTop: "12px", borderTop: "1px solid #333", paddingTop: "8px", fontSize: "0.75rem", lineHeight: "1.4" }}>
            <div style={{ fontWeight: 600, color: "#9ca3af", marginBottom: "4px" }}>SENTINEL-2 INFORMATION</div>
            <div><strong>Source:</strong> {sentinelData.source}</div>
            <div><strong>Acquisition date:</strong> {new Date(sentinelData.acquisition_date).toLocaleDateString()}</div>
            <div><strong>Cloud cover:</strong> {sentinelData.cloud_cover.toFixed(1)}%</div>
            <div><strong>Processing:</strong> Sentinel-2 Level-2A</div>
          </div>
        )}
      </div>
    </div>
  );
}
