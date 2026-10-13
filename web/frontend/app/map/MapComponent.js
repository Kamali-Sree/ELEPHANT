"use client";

import { useState, useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Circle, Popup, useMap, ImageOverlay } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import settlementsData from "@/data/maconflic-settlement-outlines.json";

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

export default function MapComponent({ geojson, userLocation, alerts = [] }) {
  const [showPath, setShowPath] = useState(false);
  const [pathOpacity, setPathOpacity] = useState(0.40);
  const [showSettlements, setShowSettlements] = useState(true);

  // Center on Kerala/Tamil Nadu/Karnataka elephant corridor region
  const center = [11.11, 76.77];
  const zoom = 12;
  const mlBounds = [[11.07, 76.68], [11.16, 76.79]];

  function FlyToLocation({ location }) {
    const map = useMap();
    useEffect(() => {
      if (location) {
        map.flyTo(location, 12, { animate: true });
      }
    }, [location, map]);
    return null;
  }

  return (
    <MapContainer
      center={center}
      zoom={zoom}
      style={{ height: "600px", width: "100%", borderRadius: "var(--radius-lg)" }}
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | Esri'
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
      />

      {/* Layer Control Panel */}
      <div style={{
        position: 'absolute',
        top: 10,
        right: 10,
        zIndex: 1000,
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(8px)',
        padding: '14px',
        borderRadius: '10px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        minWidth: '260px',
        color: '#1f2937',
        fontSize: '0.85rem'
      }}>
        <div style={{ fontWeight: 700, marginBottom: '10px', borderBottom: '1px solid #e5e7eb', paddingBottom: '6px' }}>
          Map Layers
        </div>

        {/* Settlement Outlines & References Toggle */}
        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showSettlements}
              onChange={(e) => setShowSettlements(e.target.checked)}
              style={{ transform: 'scale(1.15)', accentColor: '#3b82f6' }}
            />
            Settlement Outlines & References
          </label>
          <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '2px', marginLeft: '22px' }}>
            Image-derived footprints & verified centers
          </div>
        </div>

        {/* Potential Elephant Path Toggle */}
        <div style={{ borderTop: '1px solid #f3f4f6', paddingTop: '8px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showPath}
              onChange={(e) => setShowPath(e.target.checked)}
              style={{ transform: 'scale(1.15)', accentColor: '#ef4444' }}
            />
            Potential Elephant Movement Path
          </label>
          
          {showPath && (
            <div style={{ marginTop: '8px', fontSize: '0.8rem', paddingLeft: '4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <span style={{ display: 'inline-block', width: 14, height: 14, background: 'rgba(255, 0, 0, 0.40)', borderRadius: '2px' }}></span>
                RED: Corridor Heuristic
              </div>
              
              <div style={{ marginBottom: '6px' }}>
                <label style={{ display: 'block', marginBottom: '2px', fontWeight: 600, fontSize: '0.75rem' }}>Opacity</label>
                <input
                  type="range"
                  min="0.1"
                  max="0.8"
                  step="0.05"
                  value={pathOpacity}
                  onChange={(e) => setPathOpacity(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginTop: '6px', paddingTop: '6px', borderTop: '1px solid #f3f4f6', color: '#9ca3af', fontSize: '0.7rem' }}>
                <div><strong>Model:</strong> Spatial Connectivity Heuristic</div>
                <div><strong>Data:</strong> Simulated/Test Observations</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Corridor Overlay */}
      {showPath && (
        <ImageOverlay
          url="/maconflic-red-corridor.png"
          bounds={mlBounds}
          opacity={pathOpacity}
        />
      )}

      {/* Settlement Layer (Independent Toggle) */}
      {showSettlements && settlementsData?.features?.map((feature) => {
        const props = feature.properties;
        const coords = props.approx_center;
        if (!coords || !coords[0] || !coords[1]) return null;

        return (
          <CircleMarker
            key={`settlement-${props.id}`}
            center={[coords[1], coords[0]]}
            radius={7}
            pathOptions={{
              color: "#ffffff",
              weight: 2,
              fillColor: "#3b82f6",
              fillOpacity: 0.9,
            }}
          >
            <Popup>
              <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.85rem", minWidth: "220px", color: "#111827" }}>
                <div style={{ fontWeight: 700, fontSize: "1rem", color: "#1d4ed8", marginBottom: "0.25rem" }}>
                  🛖 {props.name}
                </div>
                <div style={{
                  display: "inline-block",
                  fontSize: "0.7rem",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  background: "#dbeafe",
                  color: "#1e40af",
                  marginBottom: "0.5rem"
                }}>
                  {props.classification}
                </div>

                <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", padding: "8px", borderRadius: "6px", fontSize: "0.78rem" }}>
                  <div><strong>Source Image:</strong> {props.source_image || "Not Available"}</div>
                  <div style={{ marginTop: "4px" }}><strong>Method:</strong> {props.extraction_method}</div>
                  <div style={{ marginTop: "4px", color: "#b45309" }}>
                    <strong>Georeference Status:</strong> {props.alignment_status}
                  </div>
                  <div style={{ marginTop: "6px", color: "#4b5563", fontStyle: "italic", lineHeight: 1.4 }}>
                    {props.notes}
                  </div>
                </div>
              </div>
            </Popup>
          </CircleMarker>
        );
      })}

      <FlyToLocation location={userLocation} />

      {/* User Live Location Marker */}
      {userLocation && (
        <CircleMarker
          center={userLocation}
          radius={12}
          pathOptions={{ color: "#3b82f6", fillColor: "#3b82f6", fillOpacity: 0.8, weight: 3 }}
        >
          <Popup>
            <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.9rem", fontWeight: "bold" }}>
              📍 Your Live Location
            </div>
          </Popup>
        </CircleMarker>
      )}

      {/* Incident Circles */}
      {geojson?.features?.map((feature) => {
        const { coordinates } = feature.geometry;
        if (!coordinates || coordinates[0] === null || coordinates[1] === null) return null;
        
        const props = feature.properties;
        const color = SEVERITY_COLORS[props.severity] || "#6b7280";
        const radius = props.severity === "critical" ? 12 : props.severity === "high" ? 10 : 8;

        return (
          <CircleMarker
            key={props.id}
            center={[coordinates[1], coordinates[0]]}
            radius={radius}
            pathOptions={{
              color: color,
              fillColor: color,
              fillOpacity: 0.6,
              weight: 2,
              opacity: 0.9,
            }}
          >
            <Popup>
              <div style={{
                fontFamily: "Inter, sans-serif",
                fontSize: "0.85rem",
                lineHeight: 1.6,
                minWidth: "200px",
              }}>
                <div style={{ fontWeight: 700, fontSize: "1rem", marginBottom: "0.5rem" }}>
                  {CALL_TYPE_ICONS[props.call_type] || "🔈"} {props.call_type || "Unknown"}
                </div>
                <div><strong>Confidence:</strong> {props.confidence ? `${(props.confidence * 100).toFixed(0)}%` : "—"}</div>
                
                <div style={{ margin: "0.5rem 0", padding: "0.5rem", background: "var(--bg-secondary)", borderRadius: "var(--radius-sm)" }}>
                  <div><strong>Risk Level:</strong> <span style={{ color, fontWeight: 600 }}>{props.severity ? props.severity.toUpperCase() : "UNKNOWN"}</span></div>
                  <div><strong>Risk Score:</strong> {props.risk_score !== null ? `${props.risk_score}/100` : "—"}</div>
                </div>

                {props.risk_reason && (
                  <div style={{ marginTop: "0.5rem", color: "var(--text-secondary)", fontSize: "0.8rem", fontStyle: "italic" }}>
                    {props.risk_reason.split(" | ").map((reason, i) => <div key={i}>• {reason}</div>)}
                  </div>
                )}

                <div style={{ marginTop: "0.5rem", color: "#666", fontSize: "0.75rem" }}>
                  {props.created_at ? new Date(props.created_at).toLocaleString() : ""}
                </div>
              </div>
            </Popup>
          </CircleMarker>
        );
      })}

      {/* Alert Circles */}
      {alerts?.map((alert) => {
        if (!alert.latitude || !alert.longitude) return null;
        const color = SEVERITY_COLORS[alert.risk_level?.toLowerCase()] || "#ef4444";
        
        return (
          <Circle
            key={`alert-${alert.id}`}
            center={[alert.latitude, alert.longitude]}
            radius={alert.radius_km * 1000}
            pathOptions={{
              color: color,
              fillColor: color,
              fillOpacity: 0.15,
              weight: 2,
              dashArray: "5, 10"
            }}
          >
            <Popup>
              <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.85rem", minWidth: "220px" }}>
                <div style={{ fontWeight: 700, fontSize: "1rem", color: color, marginBottom: "0.5rem" }}>
                  🚨 ALERT: {alert.risk_level}
                </div>
                <div style={{ marginBottom: "0.5rem" }}>
                  {alert.message?.split('\n').map((line, i) => <div key={i}>{line}</div>)}
                </div>
                <div style={{ background: "var(--bg-secondary)", padding: "0.5rem", borderRadius: "var(--radius-sm)", marginBottom: "0.5rem" }}>
                  <div><strong>Risk Score:</strong> {alert.risk_score ? `${alert.risk_score}/100` : "—"}</div>
                  <div><strong>Incident:</strong> #{alert.incident_id}</div>
                  <div><strong>Radius:</strong> {alert.radius_km} km</div>
                  <div><strong>Status:</strong> {alert.status}</div>
                </div>
                <div style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
                  <div><strong>Created:</strong> {new Date(alert.created_at).toLocaleString()}</div>
                  <div><strong>Expires:</strong> {new Date(alert.expires_at).toLocaleString()}</div>
                </div>
              </div>
            </Popup>
          </Circle>
        );
      })}
    </MapContainer>
  );
}
